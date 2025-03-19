import torch
import torch.nn as nn
import torch.nn.functional as F
import absl.flags as flags
from absl import app
import mmcv
FLAGS = flags.FLAGS  # can control the weight of each term here
from tools.training_utils import get_gt_v
from tools.rot_utils import get_rot_vec_vert_batch, get_rot_mat_y_first, get_vertical_rot_vec
from tools.dataset_utils import xyz_to_region, xyz_to_delta, xyz_to_laplace
import numpy as np

class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()
        if FLAGS.pose_loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='none')
        elif FLAGS.pose_loss_type == 'smoothl1':   # same as MSE
            self.loss_func = nn.SmoothL1Loss(beta=0.5, reduction='none')
        else:
            raise NotImplementedError
        self.loss_func_sml1 = nn.SmoothL1Loss(beta=0.2, reduction='none')
        self.symmetry_rotation_matrix_list = symmetry_rotation_matrix_y(number=360)
        self.symmetry_rotation_matrix_list_tensor_np = symmetry_rotation_matrix_y(number=FLAGS.rot_sym_num)
        self.symmetry_rotation_matrix_list_tensor = None
        self.threshold = 0.03
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum", weight=None)
        self.soft_cross_entropy = CrossEntropyLossWithSoftLabels()

    def forward(self, pred_dict, data):
        loss_dict = {}
        device = pred_dict["rot"].device
        bs = pred_dict["rot"].shape[0]
        gt_rotation_origin = data['rotation'].to(device)
        gt_translation = data["translation"].to(device)
        gt_size = data["real_size"].to(device)
        gt_mask = data["roi_mask_output"].to(device)
        gt_mask_sp = data["roi_ivfc_mask_output"].to(device)
        sym = data['sym_info'].to(device)
        nocs_scale = data['nocs_scale'].to(device).unsqueeze(-1)

        gt_size_norm = gt_size / nocs_scale
        gt_translation_norm = gt_translation / nocs_scale
        sym_mask = sym[:, 0] == 1
        gt_nocs_coor = data['nocs_coord'].to(device)
        gt_sp2d_coor = data['ivfc_coord'].to(device)

        if sym_mask.sum() > 0 and 'sym' not in FLAGS.r_type:
            if self.symmetry_rotation_matrix_list_tensor is None:
                result = []
                for rotation_matrix in self.symmetry_rotation_matrix_list_tensor_np:
                    rotation_matrix = torch.from_numpy(rotation_matrix).float().to(device)
                    result.append(rotation_matrix)
                self.symmetry_rotation_matrix_list_tensor = result
            sym_infos = []
            for i in range(bs):
                if sym[i, 0] == 1:
                    sym_infos.append(self.symmetry_rotation_matrix_list)
                else:
                    sym_infos.append(None)
            assert FLAGS.coor_gt_sym == 'rot'
            gt_rotation = get_closest_rot_batch(pred_dict["rot"], gt_rotation_origin, sym_infos)
            rot_sym = torch.bmm(gt_rotation.transpose(1, 2), gt_rotation_origin)
            gt_nocs_coor_flat = torch.reshape(gt_nocs_coor, [bs, 3, -1])
            gt_nocs_coor_sym = torch.bmm(rot_sym, gt_nocs_coor_flat)
            gt_nocs_coor_sym = gt_nocs_coor_sym.reshape([bs, 3, FLAGS.out_res, FLAGS.out_res])
            gt_sp2d_coor_flat = torch.reshape(gt_sp2d_coor, [bs, 3, -1])
            gt_sp2d_coor_sym = torch.bmm(rot_sym, gt_sp2d_coor_flat)
            gt_sp2d_coor_sym = gt_sp2d_coor_sym.reshape([bs, 3, FLAGS.out_res, FLAGS.out_res])

        else:
            gt_nocs_coor_sym = gt_nocs_coor
            gt_sp2d_coor_sym = gt_sp2d_coor
            gt_rotation = gt_rotation_origin
        syms = sym if 'sym' in FLAGS.r_type else None
        if FLAGS.r_loss == 'l1':
            loss_dict["Rot1"] = FLAGS.rot_1_w * self.cal_loss_Rot1(pred_dict["rot"], gt_rotation, syms)
        elif FLAGS.r_loss == 'angle':
            loss_dict["Rot1"] = FLAGS.rot_1_w * self.cal_loss_Rot_angle(pred_dict["rot"], gt_rotation)
        else:
            raise NotImplementedError
        loss_dict["Tran"] = FLAGS.tran_w * self.cal_loss_tran_scale(pred_dict["trans"], gt_translation_norm)

        loss_dict["Size"] = FLAGS.size_w * self.cal_loss_tran_scale(pred_dict["size"], gt_size_norm)
        model_point_norm = data['model_point'].to(device)

        loss_dict["Point_matching"] = FLAGS.prop_pm_w * self.point_matching_loss(model_point_norm,
                                                                          pred_dict['rot'],
                                                                          pred_dict["trans"],
                                                                          gt_rotation,
                                                                          gt_translation_norm, syms)

        loss_dict["nocs_coor"] = FLAGS.coor_w * self.cal_coor_loss(pred_dict["nocs_coor"], gt_nocs_coor_sym, gt_mask, sym,None, sym_mask)
        loss_dict['sp2d_coor'] = FLAGS.coor_w * self.cal_coor_loss(pred_dict['ivfc_coor'], gt_sp2d_coor_sym, gt_mask_sp, sym, None, sym_mask)

        return loss_dict

    def cal_loss_Rot1(self, pred_v, gt_v, syms):
        if syms is not None:
            mask = torch.ones_like(pred_v)
            for i in range(syms.shape[0]):
                if syms[i, 0] == 1:
                    mask[i, :, 2] = 0 #z aixs
                    mask[i, :, 0] = 0 #x aixs
            res = self.loss_func(pred_v*mask, gt_v*mask)
        else:
            res = self.loss_func(pred_v, gt_v)
        return res.mean()

    def cal_loss_Rot_angle(self, R_pred, R_gt):
        R_diff = torch.bmm(R_gt, R_pred.transpose(1, 2))  # batch matrix-matrix product
        trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)  # 计算矩阵的迹
        angle_error = torch.acos(torch.clip((trace - 1) / 2, -0.99999, 0.99999))
        res = self.loss_func_sml1(angle_error, torch.zeros_like(angle_error))
        return res.mean()

    def cal_loss_Rot2(self, pred_v, gt_v, sym):
        res = self.loss_func(pred_v, gt_v)
        valid_mask = sym[:, 0] == 0
        resw_valid = res[valid_mask]
        if resw_valid.shape[0] > 0:
            return resw_valid.mean()
        else:
            return torch.zeros(1, device=pred_v.device).squeeze()

    def cal_cosine_dis(self, pred_v, gt_v, sym=None):
        # pred_v  bs x 6, gt_v bs x 6
        res = (1.0 - torch.sum(pred_v * gt_v, dim=1)) * 2.0
        if sym is None:
            return torch.mean(res)
        else:
            valid_mask = sym[:, 0] == 0
            resw_valid = res[valid_mask]
            if resw_valid.shape[0] > 0:
                return resw_valid.mean()
            else:
                return torch.zeros(1, device=pred_v.device).squeeze()


    def cal_rot_regular_angle(self, pred_v1, pred_v2, sym):
        bs = pred_v1.shape[0]
        res = torch.zeros(1, device=pred_v1.device).squeeze()
        valid = 0.0
        for i in range(bs):
            if sym[i, 0] == 1:
                continue
            y_direction = pred_v1[i]
            z_direction = pred_v2[i]
            residual = torch.dot(y_direction, z_direction)
            res += torch.abs(residual)
            valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res

    def cal_loss_tran_scale(self, pred_v, gt_v):
        res = self.loss_func(pred_v, gt_v)
        return res.mean()

    def point_matching_loss(self, points, p_rot, p_t, g_rot, g_t, syms):
        # Notice that this loss function do not back-propagate the grad of f_g_vec and f_r_vec
        # bs = points.shape[0]
        points = points.permute(0, 2, 1)
        if syms is not None:
            for i in range(syms.shape[0]):
                if syms[i, 0] == 1:
                    points[i, 0, :] = 0 #z aixs
                    points[i, 2, :] = 0 #x aixs
        pred_points = torch.bmm(p_rot, points) # + p_t[..., None]
        gt_points = torch.bmm(g_rot, points) # + g_t[..., None]
        return self.loss_func(pred_points, gt_points).mean()

    def cal_region_loss_for_batch(self, pred_region, gt_region, gt_mask):
        diff = torch.abs(pred_region - gt_region)
        lower_corr_loss = torch.pow(diff, 2) / (2.0 * self.threshold)
        higher_corr_loss = diff - self.threshold / 2.0
        corr_loss_matrix = torch.where(diff > self.threshold, higher_corr_loss, lower_corr_loss)
        corr_loss_matrix = gt_mask * corr_loss_matrix
        corr_loss = torch.sum(corr_loss_matrix, dim=[1,2,3]) / (torch.sum(gt_mask, dim=[1,2,3]) + 1e-5)
        return corr_loss.mean()

    def cal_coor_loss(self, pred_coor, gt_coor, gt_mask, sym, log_var, sym_mask):
        # filter out invalid point
        pred_coor = pred_coor * gt_mask
        gt_coor = gt_coor * gt_mask
        assert FLAGS.coor_gt_sym != 'coor'
        return self.cal_coor_loss_for_batch(pred_coor, gt_coor, gt_mask, log_var, sym_mask)
    def cal_coor_loss_for_batch(self, coords, nocs, gt_mask, log_variance, sym_mask):
        assert not FLAGS.coor_gt_sym == 'radius'
        diff = torch.abs(coords - nocs)
        lower_corr_loss = torch.pow(diff, 2) / (2.0 * self.threshold)
        higher_corr_loss = diff - self.threshold / 2.0
        corr_loss_matrix = torch.where(diff > self.threshold, higher_corr_loss, lower_corr_loss)
        corr_loss_matrix = gt_mask * corr_loss_matrix
        corr_loss = torch.sum(corr_loss_matrix, dim=[1,2,3]) / (torch.sum(gt_mask, dim=[1,2,3]) + 1e-5)
        return corr_loss.mean()

class Pnp_Loss(nn.Module):
    def __init__(self):
        super(Pnp_Loss, self).__init__()
        if FLAGS.pose_loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='none')
        elif FLAGS.pose_loss_type == 'smoothl1':   # same as MSE
            self.loss_func = nn.SmoothL1Loss(beta=0.5, reduction='none')
        else:
            raise NotImplementedError
        self.symmetry_rotation_matrix_list = symmetry_rotation_matrix_y(number=360)
        self.symmetry_rotation_matrix_list_tensor_np = symmetry_rotation_matrix_y(number=FLAGS.rot_sym_num)
        self.symmetry_rotation_matrix_list_tensor = None
        self.threshold = 0.03

    def forward(self, pred_dict, data):
        loss_dict = {}
        device = pred_dict["rot"].device
        bs = pred_dict["rot"].shape[0]
        gt_rotation_origin = data['rotation'].to(device)
        gt_translation = data["translation"].to(device)
        sym = data['sym_info'].to(device)
        nocs_scale = data['nocs_scale'].to(device).unsqueeze(-1)
        gt_translation_norm = gt_translation / nocs_scale
        sym_mask = sym[:, 0] == 1

        if sym_mask.sum() > 0:
            if self.symmetry_rotation_matrix_list_tensor is None:
                result = []
                for rotation_matrix in self.symmetry_rotation_matrix_list_tensor_np:
                    rotation_matrix = torch.from_numpy(rotation_matrix).float().to(device)
                    result.append(rotation_matrix)
                self.symmetry_rotation_matrix_list_tensor = result
            sym_infos = []
            for i in range(bs):
                if sym[i, 0] == 1:
                    sym_infos.append(self.symmetry_rotation_matrix_list)
                else:
                    sym_infos.append(None)
            assert FLAGS.coor_gt_sym == 'rot'
            gt_rotation = get_closest_rot_batch(pred_dict["rot"], gt_rotation_origin, sym_infos)
        else:
            gt_rotation = gt_rotation_origin

        loss_dict["Rot1"] = FLAGS.rot_1_w * self.cal_loss_Rot1(pred_dict["rot"], gt_rotation)
        loss_dict["Tran"] = FLAGS.tran_w * self.cal_loss_tran_scale(pred_dict["trans"], gt_translation_norm)

        model_point_norm = data['model_point'].to(device)
        loss_dict["Point_matching"] = FLAGS.prop_pm_w * self.point_matching_loss(model_point_norm,
                                                                          pred_dict['rot'],
                                                                          pred_dict["trans"],
                                                                          gt_rotation,
                                                                          gt_translation_norm)
        return loss_dict


    def cal_loss_Rot1(self, pred_v, gt_v):
        res = self.loss_func(pred_v, gt_v)
        return res.mean()

    def cal_cosine_dis(self, pred_v, gt_v, sym=None):
        # pred_v  bs x 6, gt_v bs x 6
        res = (1.0 - torch.sum(pred_v * gt_v, dim=1)) * 2.0
        if sym is None:
            return torch.mean(res)
        else:
            valid_mask = sym[:, 0] == 0
            resw_valid = res[valid_mask]
            if resw_valid.shape[0] > 0:
                return resw_valid.mean()
            else:
                return torch.zeros(1, device=pred_v.device).squeeze()


    def cal_rot_regular_angle(self, pred_v1, pred_v2, sym):
        bs = pred_v1.shape[0]
        res = torch.zeros(1, device=pred_v1.device).squeeze()
        valid = 0.0
        for i in range(bs):
            if sym[i, 0] == 1:
                continue
            y_direction = pred_v1[i]
            z_direction = pred_v2[i]
            residual = torch.dot(y_direction, z_direction)
            res += torch.abs(residual)
            valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res

    def cal_loss_tran_scale(self, pred_v, gt_v):
        res = self.loss_func(pred_v, gt_v)
        return res.mean()

    def point_matching_loss(self, points, p_rot, p_t, g_rot, g_t):
        # Notice that this loss function do not back-propagate the grad of f_g_vec and f_r_vec
        # bs = points.shape[0]
        points = points.permute(0, 2, 1)
        pred_points = torch.bmm(p_rot, points) # + p_t[..., None]
        gt_points = torch.bmm(g_rot, points) # + g_t[..., None]
        return self.loss_func(pred_points, gt_points).mean()


class CrossEntropyLossWithSoftLabels(nn.Module):
    def __init__(self):
        super(CrossEntropyLossWithSoftLabels, self).__init__()

    def forward(self, pred, target):
        pred_p = F.softmax(pred, dim=1)
        target[target==0] = 1
        log_target = torch.log(target)

        loss = -torch.sum(log_target * pred_p)
        return loss

def region2laplace(gt_nocs, region_num, ref_pc, gt_region, pred_region):
    laplace_label = torch.zeros_like(pred_region)
    for i in range(3):
        pass
        # res = xyz_to_delta(gt_nocs, region_num, ref_pc[:,i], i, gt_region[:,i])


def symmetry_rotation_matrix_y(number=30):
    result = []
    for i in range(number):
        theta = 2 * np.pi / number * i
        r = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        result.append(r)
    result = np.stack(result)
    return result


def get_closest_rot(rot_est, rot_gt, sym_info):
    """get the closest rot_gt given rot_est and sym_info.

    rot_est: ndarray
    rot_gt: ndarray
    sym_info: None or Kx3x3 ndarray, m2m
    """
    if sym_info is None:
        return rot_gt
    if isinstance(sym_info, torch.Tensor):
        sym_info = sym_info.cpu().numpy()
    if len(sym_info.shape) == 2:
        sym_info = sym_info.reshape((1, 3, 3))
    # find the closest rot_gt with smallest re
    r_err = re(rot_est, rot_gt)
    closest_rot_gt = rot_gt
    for i in range(sym_info.shape[0]):
        # R_gt_m2c x R_sym_m2m ==> R_gt_sym_m2c
        rot_gt_sym = rot_gt.dot(sym_info[i])
        cur_re = re(rot_est, rot_gt_sym)
        if cur_re < r_err:
            r_err = cur_re
            closest_rot_gt = rot_gt_sym

    return closest_rot_gt

def get_sym_max_nocs_rot(gt_coor, gt_rot, sym_info, nocs_coor=None):
    if sym_info is None:
        return gt_rot, gt_coor
    if isinstance(sym_info, torch.Tensor):
        sym_info = sym_info.cpu().numpy()
    if len(sym_info.shape) == 2:
        sym_info = sym_info.reshape((1, 3, 3))
    assert gt_coor.shape[0] == 3
    gt_coor_flat = np.reshape(gt_coor, [3, -1])
    max_coor_xz_sum = xyz_sum(gt_coor_flat)
    closest_sym_info = sym_info[0]
    for i in range(sym_info.shape[0]):
        # R_gt_m2c x R_sym_m2m ==> R_gt_sym_m2c
        gt_coor_flat_sym = np.dot(sym_info[i], gt_coor_flat)
        cur_coor_xz_sum = xyz_sum(gt_coor_flat_sym)
        if cur_coor_xz_sum > max_coor_xz_sum:
            max_coor_xz_sum = cur_coor_xz_sum
            closest_sym_info = sym_info[i]
    if nocs_coor is None:
        max_sym_coor_flat = np.dot(closest_sym_info, gt_coor_flat)
        max_sym_coor = np.reshape(max_sym_coor_flat, [3,64,64])
        max_sym_rot = np.dot(gt_rot, closest_sym_info)
    else:
        # when use closet point, trans nocs_coor
        nocs_coor_flat = np.reshape(nocs_coor, [3, -1])
        max_sym_coor_flat = np.dot(closest_sym_info, nocs_coor_flat)
        max_sym_coor = np.reshape(max_sym_coor_flat, [3, 64, 64])
        max_sym_rot = np.dot(gt_rot, closest_sym_info)
    return max_sym_rot, max_sym_coor
def xyz_sum(coor):
    return np.sum(coor)


class Scale_loss(nn.Module):
    def __init__(self):
        super(Scale_loss, self).__init__()
        if FLAGS.pose_loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='none')
        elif FLAGS.pose_loss_type == 'smoothl1':   # same as MSE
            self.loss_func = nn.SmoothL1Loss(beta=0.5, reduction='none')
    def forward(self, pred_scale, gt_scale):
        # device = pred_scale.device
        # gt_scale = gt_scale.to(device)
        loss = self.loss_func(pred_scale, gt_scale)
        return loss.mean()

def get_closest_rot_batch(pred_rots, gt_rots, sym_infos):
    """
    get closest gt_rots according to current predicted poses_est and sym_infos
    --------------------
    pred_rots: [B, 4] or [B, 3, 3]
    gt_rots: [B, 4] or [B, 3, 3]
    sym_infos: list [Kx3x3 or None],
        stores K rotations regarding symmetries, if not symmetric, None
    -----
    closest_gt_rots: [B, 3, 3]
    """
    batch_size = pred_rots.shape[0]
    device = pred_rots.device

    closest_gt_rots = gt_rots.clone().cpu().numpy()  # B,3,3

    for i in range(batch_size):
        if sym_infos[i] is None:
            closest_gt_rots[i] = gt_rots[i].cpu().numpy()
        else:
            closest_rot = get_closest_rot(
                pred_rots[i].detach().cpu().numpy(),
                gt_rots[i].cpu().numpy(),
                sym_infos[i],
            )
            closest_gt_rots[i] = closest_rot
    closest_gt_rots = torch.tensor(closest_gt_rots, device=device, dtype=gt_rots.dtype)
    return closest_gt_rots

def get_sym_max_rot_coor_batch(gt_coor, gt_rots, sym_infos):
    batch_size = gt_rots.shape[0]
    device = gt_rots.device

    max_sym_gt_rots = gt_rots.clone().cpu().numpy()
    max_sym_nocs_coors = gt_coor.clone().cpu().numpy()
    for i in range(batch_size):
        if sym_infos[i] is None:
            max_sym_gt_rots[i] = gt_rots[i].cpu().numpy()
        else:
            max_sym_rot, max_sym_nocs_coor = get_sym_max_nocs_rot(
                gt_coor[i].cpu().numpy(),
                gt_rots[i].cpu().numpy(),
                sym_infos[i],
            )
            max_sym_gt_rots[i] = max_sym_rot
            max_sym_nocs_coors[i] = max_sym_nocs_coor
    max_sym_gt_rots = torch.tensor(max_sym_gt_rots, device=device, dtype=gt_rots.dtype)
    max_sym_nocs_coors = torch.tensor(max_sym_nocs_coors, device=device, dtype=gt_rots.dtype)
    return max_sym_gt_rots, max_sym_nocs_coors

def re(R_est, R_gt):
    """Rotational Error.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error.
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    rotation_diff = np.dot(R_est, R_gt.T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    rd_deg = np.rad2deg(np.arccos(error_cos))

    return rd_deg


def laplacian_aleatoric_uncertainty_loss(input, target, log_variance, mask, balance_weight=10, reduction='mean', sum_last_dim=True):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum', 'none']
    if sum_last_dim:
        loss = balance_weight * 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target).sum(1).unsqueeze(1) + 0.5 * log_variance
    else:
        loss = balance_weight * 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target) + 0.5 * log_variance
    loss = loss * mask if mask is not None else loss
    if reduction == 'none':
        return loss
    return loss.mean() if reduction == 'mean' else loss.sum()


def gaussian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    References:
        What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, Neuips'17
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 0.5 * torch.exp(-log_variance) * torch.abs(input - target)**2 + 0.5 * log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()


def xyz_to_region_batch(xyz_crop, fps_points, sym_infos, gt_regions):
    bs = xyz_crop.shape[0]
    device = xyz_crop.device
    gt_sym_rot_regions = gt_regions.clone().cpu().numpy()
    for i in range(bs):
        if sym_infos[i] is None:
            gt_sym_rot_regions[i] = gt_regions[i].cpu().numpy()
        else:
            gt_sym_rot_regions_i = []
            for j in range(3):
                fps_points_j = fps_points[i].detach().cpu().numpy()[j]
                gt_sym_rot_regions_i.append(xyz_to_region(xyz_crop[i].detach().cpu().numpy().transpose(1,2,0), fps_points_j))
            gt_sym_rot_regions[i] = np.stack(gt_sym_rot_regions_i)
    gt_sym_rot_regions = torch.tensor(gt_sym_rot_regions, device=device, dtype=gt_regions.dtype)

    return gt_sym_rot_regions

def xyz_to_laplace_batch(xyz_crop, gt_regions, fps_points, sym_infos, gt_regions_laplace):
    bs = xyz_crop.shape[0]
    device = xyz_crop.device
    gt_sym_rot_regions_laplace = gt_regions_laplace.clone().cpu().numpy()
    for i in range(bs):
        if sym_infos[i] is None:
            gt_sym_rot_regions_laplace[i] = gt_regions_laplace[i].cpu().numpy()
        else:
            gt_sym_rot_regions_laplace_i = []
            for j in range(3):
                fps_points_j = fps_points[i].detach().cpu().numpy()[j]
                roi_region_delta = xyz_to_delta(xyz_crop[i].detach().cpu().numpy().transpose(1,2,0), fps_points_j, j)
                roi_region_laplace = xyz_to_laplace(gt_regions[i][j].cpu().numpy(), roi_region_delta, FLAGS.region_num)
                gt_sym_rot_regions_laplace_i.append(roi_region_laplace)
            gt_sym_rot_regions_laplace[i] = np.stack(gt_sym_rot_regions_laplace_i)
    gt_sym_rot_regions_laplace = torch.tensor(gt_sym_rot_regions_laplace, device=device, dtype=xyz_crop.dtype)
    return gt_sym_rot_regions_laplace
