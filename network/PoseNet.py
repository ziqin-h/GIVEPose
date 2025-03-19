import torch
import torch.nn as nn
import torch.nn.functional as F
import absl.flags as flags
import cv2
import os
FLAGS = flags.FLAGS

from network.backbone import convnext_backbone
from network.xyz_head import TopDownMaskXyzHead, TopDownXyzHead
from network.conv_pnp_net import ConvPnPNet, ConvPnPNet_T, ConvPnPNet_R, ResPnPNet, MAPEncoder
from network.attention_pnp_net import AttentionPnPNet, CrossAttentionPnPNet, MAPTransformerEncoer
from network.pose_head import SizeHead, RHead, THead, Conv_feat_finetune, ChannelReducer, TRHead_linear, Auxi_Head
from tools.rot_utils import get_rot_mat_y_first
from tools.umeyama import pose_from_umeyama
import torch.nn.functional as F
from .pose_utils.pose_from_pred import pose_from_pred
from .pose_utils.pose_from_pred_centroid_z import pose_from_pred_centroid_z
from .pose_utils.pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs
import copy
import torch
import torch.nn as nn
import numpy as np
from .pose_utils.pose_error import re, te
from .pose_utils.pose_utils import quat2mat_torch
from .pose_utils.rot_reps import rot6d_to_mat_batch, euler2mat_batch, rot6d_fixed_y_to_mat_batch, rot6d_fixed_z_to_mat_batch, rot6d_fixed_x_to_mat_batch
from network.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
from network.att_mask_head import AttentionMaskHead
from torchvision.transforms import Resize, InterpolationMode
from network.resnet import resnet18, resnet34, resnet50, resnet101, resnets
import itertools
from losses.pose_loss import get_closest_rot_batch, xyz_to_region_batch, xyz_to_laplace_batch
from timm.models.layers import trunc_normal_
from tools.utils.align import ransac_pnp

def get_rot_mat(rot, rot_type):
    if rot_type in ["ego_quat", "allo_quat"]:
        rot_m = quat2mat_torch(rot)
    elif rot_type in ["ego_rot6d", "allo_rot6d", "allo_rot6d_sym"]:
        rot_m = rot6d_to_mat_batch(rot)
    elif rot_type in ['allo_rot6d_sym_y', 'allo_rot6d_y']:
        rot_m = rot6d_fixed_y_to_mat_batch(rot)
    elif rot_type in ['allo_rot6d_z']:
        rot_m = rot6d_fixed_z_to_mat_batch(rot)
    elif rot_type in ['allo_rot6d_x']:
        rot_m = rot6d_fixed_x_to_mat_batch(rot)
    elif rot_type == 'euler':
        rot_m = euler2mat_batch(rot)
    else:
        raise ValueError(f"Wrong pred_rot type: {rot_type}")
    return rot_m


def get_mask_prob(pred_mask, mask_loss_type):
    # (b,c,h,w)
    # output: (b, 1, h, w)
    bs, c, h, w = pred_mask.shape
    if mask_loss_type == "L1":
        assert c == 1, c
        mask_max = torch.max(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        mask_min = torch.min(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        # [0, 1]
        mask_prob = (pred_mask - mask_min) / (mask_max - mask_min)  # + 1e-6)
    elif mask_loss_type in ["BCE", "RW_BCE", "dice"]:
        assert c == 1, c
        mask_prob = torch.sigmoid(pred_mask)
    elif mask_loss_type == "CE":
        mask_prob = torch.softmax(pred_mask, dim=1, keepdim=True)[:, 1:2, :, :]
    else:
        raise NotImplementedError(f"Unknown mask loss type: {mask_loss_type}")
    return mask_prob


def compute_mean_re_te(pred_transes, pred_rots, gt_transes, gt_rots):
    pred_transes = pred_transes.detach().cpu().numpy()
    pred_rots = pred_rots.detach().cpu().numpy()
    gt_transes = gt_transes.detach().cpu().numpy()
    gt_rots = gt_rots.detach().cpu().numpy()

    bs = pred_rots.shape[0]
    R_errs = np.zeros((bs,), dtype=np.float32)
    T_errs = np.zeros((bs,), dtype=np.float32)
    for i in range(bs):
        R_errs[i] = re(pred_rots[i], gt_rots[i])
        T_errs[i] = te(pred_transes[i], gt_transes[i])
    return R_errs.mean(), T_errs.mean()

def add_name(img, text):
    height, width = img.shape[:2]
    margin_height = 20
    new_image = np.ones((height + margin_height, width, 3), dtype=np.uint8)*255
    new_image[margin_height:, :] = img

    # font setting
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    font_color = (0, 0, 0)

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = (width - text_width) // 2
    text_y = margin_height - 5
    cv2.putText(new_image, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    return new_image

def symmetry_rotation_matrix_y(number=30):
    result = []
    for i in range(number):
        theta = 2 * np.pi / number * i
        r = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        result.append(r)
    result = np.stack(result)
    return result

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        featdim = 128
        backbone = FLAGS.main_backbone
        self.backbone_name = FLAGS.main_backbone
        pnp_input_feat = 5
        # -----------------backbone
        assert backbone == 'convnext'
        self.backbone = convnext_backbone()
        feature_channel = 1024
        self.xyz_nocs_head = TopDownXyzHead(in_dim=feature_channel, xyz_num_classes=1)

        size_feature_channel = feature_channel
        self.size_head = SizeHead(in_dim=size_feature_channel, out_dim=FLAGS.size_head_out_dim)

        # -----------------pose head
        rot_dim = 4 if 'quat' in FLAGS.r_type else 6
        if FLAGS.nocsmap_encoder =='conv':
            self.nocs_encoder = MAPEncoder(3, featdim=256)
        elif FLAGS.nocsmap_encoder == 'att':
            self.nocs_encoder = MAPTransformerEncoer()
        else:
            raise NotImplementedError
        self.feat_reducer = nn.Conv2d(feature_channel, 256, kernel_size=1)
        self.act = get_nn_act_func("lrelu")
        self.xyz_deform_head = TopDownXyzHead(in_dim=512, xyz_num_classes=1)

        self.pnp_net = ConvPnPNet(pnp_input_feat, featdim=featdim, mask_attention_type=FLAGS.mask_attention_type, rot_dim=rot_dim, flat_op=FLAGS.flat_op)


        self.out_res = FLAGS.out_res
        self.ROT_TYPE = FLAGS.r_type
        self.TRANS_TYPE = "centroid_z"
        self.Z_TYPE = 'REL'

        self.resize_func_out = Resize(FLAGS.out_res, interpolation=InterpolationMode.NEAREST)
        self.resize_func_input = Resize(224, interpolation=InterpolationMode.NEAREST)

    def forward(self, data, device, do_loss=False, pred_scale=None):
        img = data['roi_img'].to(device)
        K = data['cam_K']
        if do_loss:
            mask = data['roi_mask_deform'].to(device)
        else:
            mask = data['roi_mask'].to(device)
        mask_out = self.resize_func_out(mask)
        # 1. extract feature
        if self.backbone_name == 'convnext':
            feat = self.backbone(img)
            pred_size = self.size_head(feat)
            coor_x_nocs, coor_y_nocs, coor_z_nocs = self.xyz_nocs_head(feat)
            coor_xyz_nocs = torch.cat([coor_x_nocs, coor_y_nocs, coor_z_nocs], dim=1)
        else:
            raise NotImplementedError

        # 2. predict pose
        nocs_feat = self.nocs_encoder(coor_xyz_nocs)
        conv_feat256 = self.feat_reducer(feat[0])
        feat_cat = torch.cat([conv_feat256, nocs_feat], dim=1)
        coor_nocs2ivfc_x, coor_nocs2ivfc_y, coor_nocs2ivfc_z = self.xyz_deform_head(feat_cat)
        coor_xyz_ivfc = torch.cat([coor_nocs2ivfc_x, coor_nocs2ivfc_y, coor_nocs2ivfc_z], dim=1)
        coor_xyz_ivfc_2d = torch.cat([coor_xyz_ivfc, data['roi_coord_2d'].to(device)], dim=1)
        pred_rot_, pred_t_, _ = self.pnp_net(coor_feat=coor_xyz_ivfc_2d, mask_attention=mask_out)

        mean_size = data['mean_size'].to(device)
        mean_scale = mean_size.norm(dim=1)
        norm_mean_size = mean_size/mean_scale.unsqueeze(-1)
        pred_size = pred_size + norm_mean_size

        # convert pred_rot to rot mat -------------------------
        rot_type = self.ROT_TYPE
        pred_rot_m = get_rot_mat(pred_rot_, rot_type)

        roi_cams = data['cam_K'].to(device)
        roi_whs = data['roi_wh'].to(device)
        roi_centers = data['bbox_center'].to(device)
        resize_ratios = data['resize_ratio'].to(device)

        # convert pred_rot_m and pred_t to ego pose -----------------------------
        assert self.TRANS_TYPE == "centroid_z"
        pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
            pred_rot_m,
            pred_centroids=pred_t_[:, :2] if FLAGS.t_type == 'site' else pred_t_[:, :2]*0,
            pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
            roi_cams=roi_cams,
            roi_centers=roi_centers,
            resize_ratios=resize_ratios,
            roi_whs=roi_whs,
            eps=1e-4,
            is_allo="allo" in rot_type,
            z_type=self.Z_TYPE,
            is_train=do_loss,
            dataset_name=FLAGS.dataset,
            )
        out_dict = {"rot": pred_ego_rot, "trans": pred_trans, "size": pred_size,
                    "mask": mask_out, 'nocs_coor': coor_xyz_nocs, 'ivfc_coor': coor_xyz_ivfc}
        return out_dict

    def build_params_optimizer(self, training_stage_freeze=None):
        #  training_stage is a list that controls whether to freeze each module
        params_lr_list = []
        if 'backbone' in training_stage_freeze:
            for param in self.backbone_nocs.parameters():
                with torch.no_grad():
                    param.requires_grad = False
            for param in self.backbone_socs.parameters():
                with torch.no_grad():
                    param.requires_grad = False

        # backbone
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, self.parameters()),
                "lr": float(FLAGS.lr),
            }
        )

        return params_lr_list
