import torch
import numpy as np
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init
from network.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
from network.torch_utils.layers.dropblock import DropBlock2D, LinearScheduler
from timm.models.layers import StdConv2d
from torchvision.transforms import Resize, InterpolationMode
from network.coordatt import CoordAtt
from network.resnet import resnets

import absl.flags as flags
FLAGS = flags.FLAGS
from network.dcnv3 import DCNv3_PyTorch, DCNv3_C


class ConvPnPNet(nn.Module):
    def __init__(
        self,
        nIn,
        mask_attention_type="none",
        featdim=128,
        add_dim=0,
        rot_dim=6,
        num_stride2_layers=3,
        num_extra_layers=0,
        use_ws=False,
        norm="GN",
        num_gn_groups=32,
        act="relu",
        drop_prob=0.0,
        dropblock_size=5,
        flat_op="flatten",
        final_spatial_size=(8, 8),
        denormalize_by_extent=False,
    ):
        """
        Args:
            nIn: input feature channel
            spatial_pooltype: max | soft
            spatial_topk: 1
            flat_op: flatten | avg | avg-max | avg-max-min
        """
        super().__init__()
        self.featdim = featdim
        self.mask_attention_type = mask_attention_type
        self.flat_op = flat_op
        self.denormalize_by_extent = denormalize_by_extent
        self.rot_dim = rot_dim
        assert add_dim == 0
        if self.mask_attention_type == "concat":
            nIn += 1

        conv_act = get_nn_act_func(act)
        if act == "relu":
            self.act = get_nn_act_func("lrelu")  # legacy model
        else:
            self.act = get_nn_act_func(act)
        # -----------------------------------
        self.drop_prob = drop_prob
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=dropblock_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=5000,
        )


        conv_layer = StdConv2d if use_ws else nn.Conv2d
        self.features = nn.ModuleList()
        for i in range(num_stride2_layers):
            _in_channels = nIn if i == 0 else featdim
            self.features.append(
                conv_layer(
                    _in_channels,
                    featdim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)

        for i in range(num_extra_layers):
            self.features.append(
                conv_layer(
                    featdim,
                    featdim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)

        final_h, final_w = final_spatial_size
        fc_in_dim = {
            "flatten": featdim * final_h * final_w,
            "avg": featdim,
            "avg-max": featdim * 2,
            "avg-max-min": featdim * 3,
        }[flat_op]

        # self.fc1 = nn.Linear(featdim * 8 * 8 + 128, 1024)  # NOTE: 128 for extents feature
        self.fc1 = nn.Linear(fc_in_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)


        self.fc1_z = nn.Linear(fc_in_dim, 1024)
        self.fc2_z = nn.Linear(1024, 256)

        self.fc_z = nn.Linear(256, 1)
        self.fc_r = nn.Linear(256, rot_dim)  # quat or rot6d
        self.fc_t = nn.Linear(256, 2)

        self.resize_func_input = Resize(FLAGS.out_res, interpolation=InterpolationMode.NEAREST)

        # init ------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_r, std=0.01)
        normal_init(self.fc_t, std=0.01)


    def forward(self, coor_feat=None,  mask_attention=None, cat_id=None, sp2d=None):
        """
        Args:
             since this is the actual correspondence
            x: (B,C,H,W)
            extents: (B, 3)
        Returns:

        """
        if self.mask_attention_type != "none" and coor_feat is not None:
            assert mask_attention is not None
            # mask_attention = self.resize_func_input(mask_attention)
            if self.mask_attention_type == "mul":
                coor_feat = coor_feat * mask_attention
            elif self.mask_attention_type == "concat":
                coor_feat = torch.cat([coor_feat, mask_attention], dim=1)
            else:
                raise ValueError(f"Wrong mask attention type: {self.mask_attention_type}")

        if sp2d is not None:
            x = torch.cat([sp2d, coor_feat], dim=1) if coor_feat is not None else sp2d
        else:
            x = coor_feat
        if self.drop_prob > 0:
            self.dropblock.step()  # increment number of iterations
            x = self.dropblock(x)

        for _i, layer in enumerate(self.features):
            x = layer(x)

        pnp_feat = x
        flat_conv_feat = x.flatten(2)  # [B,featdim,*]
        if self.flat_op == "flatten":
            flat_conv_feat = flat_conv_feat.flatten(1)
        elif self.flat_op == "avg":
            flat_conv_feat = flat_conv_feat.mean(-1)  # spatial global average pooling
        elif self.flat_op == "avg-max":
            flat_conv_feat = torch.cat([flat_conv_feat.mean(-1), flat_conv_feat.max(-1)[0]], dim=-1)
        elif self.flat_op == "avg-max-min":
            flat_conv_feat = torch.cat(
                [
                    flat_conv_feat.mean(-1),
                    flat_conv_feat.max(-1)[0],
                    flat_conv_feat.min(-1)[0],
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Invalid flat_op: {self.flat_op}")

        #
        x = self.act(self.fc1(flat_conv_feat))
        x = self.act(self.fc2(x))
        #
        rot = self.fc_r(x)
        t = self.fc_t(x)

        xz = self.act(self.fc1_z(flat_conv_feat))
        xz = self.act(self.fc2_z(xz))
        z = self.fc_z(xz)


        t = torch.cat([t, z], dim=1)

        return rot, t,  pnp_feat

class MAPEncoder(nn.Module):
    def __init__(
        self,
        nIn,
        mask_attention_type="none",
        featdim=128,
        outdim=256,
        add_dim=0,
        rot_dim=6,
        num_stride2_layers=3,
        num_extra_layers=0,
        use_ws=False,
        norm="GN",
        num_gn_groups=32,
        act="relu",
        drop_prob=0.0,
        dropblock_size=5,
        flat_op="flatten",
        final_spatial_size=(8, 8),
        denormalize_by_extent=False,
    ):
        """
        Args:
            nIn: input feature channel
            spatial_pooltype: max | soft
            spatial_topk: 1
            flat_op: flatten | avg | avg-max | avg-max-min
        """
        super().__init__()
        self.featdim = featdim
        self.mask_attention_type = mask_attention_type
        self.flat_op = flat_op
        self.denormalize_by_extent = denormalize_by_extent
        self.rot_dim = rot_dim
        assert add_dim == 0
        if self.mask_attention_type == "concat":
            nIn += 1

        conv_act = get_nn_act_func(act)
        if act == "relu":
            self.act = get_nn_act_func("lrelu")  # legacy model
        else:
            self.act = get_nn_act_func(act)
        # -----------------------------------
        self.drop_prob = drop_prob
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=dropblock_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=5000,
        )
        if FLAGS.use_dcn == 'dcnv3':
            conv_layer = DCNv3_C
        else:
            assert FLAGS.use_dcn == ''
            conv_layer = nn.Conv2d
        self.features = nn.ModuleList()
        for i in range(num_stride2_layers):
            _in_channels = nIn if i == 0 else featdim
            featdim = outdim if i == num_stride2_layers-1 else featdim
            self.features.append(
                conv_layer(
                    _in_channels,
                    featdim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)


        for i in range(num_extra_layers):
            self.features.append(
                conv_layer(
                    featdim,
                    featdim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)

        # init ------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)


    def forward(self, coor_feat=None,  mask_attention=None, cat_id=None, sp2d=None):
        """
        Args:
             since this is the actual correspondence
            x: (B,C,H,W)
            extents: (B, 3)
        Returns:

        """
        if self.mask_attention_type != "none" and coor_feat is not None:
            assert mask_attention is not None
            # mask_attention = self.resize_func_input(mask_attention)
            if self.mask_attention_type == "mul":
                coor_feat = coor_feat * mask_attention
            elif self.mask_attention_type == "concat":
                coor_feat = torch.cat([coor_feat, mask_attention], dim=1)
            else:
                raise ValueError(f"Wrong mask attention type: {self.mask_attention_type}")

        if sp2d is not None:
            x = torch.cat([sp2d, coor_feat], dim=1) if coor_feat is not None else sp2d
        else:
            x = coor_feat
        if self.drop_prob > 0:
            self.dropblock.step()  # increment number of iterations
            x = self.dropblock(x)

        for _i, layer in enumerate(self.features):
            x = layer(x)
        return x

class ResPnPNet(nn.Module):
    def __init__(self, nIn, rot_dim=6):
        super(ResPnPNet, self).__init__()
        feature_extractor = resnets(in_dim=nIn)
        self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu,
                                             feature_extractor.layer1, feature_extractor.layer2, feature_extractor.layer3, feature_extractor.layer4,
                                            feature_extractor.avgpool
                                             # resnet.avgpool
                                             )
        feature_channel = self.cnn_encoder[-2][-1].conv1.weight.shape[1]

        rt_z_fc_out = [256, 256]
        self.head_rt = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                nn.Flatten(),
                nn.BatchNorm1d(int(feature_channel)),
                nn.Linear(int(feature_channel), rt_z_fc_out[0]),
                nn.BatchNorm1d(rt_z_fc_out[0]),
                nn.ReLU(inplace=True),
                # nn.Sigmoid()
            )

        self.head_z = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                nn.Flatten(),
                nn.BatchNorm1d(int(feature_channel)),
                nn.Linear(int(feature_channel), rt_z_fc_out[1]),
                nn.BatchNorm1d(rt_z_fc_out[1]),
                nn.ReLU(inplace=True),
                # nn.Sigmoid()
            )

        if 'CAT' in FLAGS.z_type:
            self.fc_z = nn.Linear(256, 6)
            self.fc_r = nn.Linear(256, rot_dim*6)  # quat or rot6d
            self.fc_t = nn.Linear(256, 2*6)

        else:
            self.fc_z = nn.Linear(256, 1)
            self.fc_r = nn.Linear(256, rot_dim)  # quat or rot6d
            self.fc_t = nn.Linear(256, 2)

    def forward(self, coor_feat,  mask_attention=None, region=None, cat_id=None):
        if region is not None:
            if coor_feat is not None:
                x = torch.cat([coor_feat, region], dim=1)
            else:
                x = region
        else:
            x = coor_feat
        x = self.cnn_encoder(x) #[bs, 512, 8, 8]
        x_rt = self.head_rt(x)
        x_z = self.head_z(x)

        rot = self.fc_r(x_rt)
        t = self.fc_t(x_rt)
        z = self.fc_z(x_z)

        if 'CAT' in FLAGS.z_type:
            bs = z.shape[0]
            z = z[np.arange(bs), cat_id]
            z = z.view(-1, 1)
            t = t.view(bs, 6, 2)
            t = t[np.arange(bs), cat_id]
            rot = rot.view(bs, 6, self.rot_dim)
            rot = rot[np.arange(bs), cat_id]
        t = torch.cat([t, z], dim=1)

        return rot, t, None

class ConvPnPNet_T(nn.Module):
    def __init__(
        self,
        nIn,
        mask_attention_type="none",
        featdim=128,
        add_dim=0,
        rot_dim=6,
        num_stride2_layers=3,
        num_extra_layers=0,
        use_ws=False,
        norm="GN",
        num_gn_groups=32,
        act="relu",
        drop_prob=0.0,
        dropblock_size=5,
        flat_op="flatten",
        final_spatial_size=(8, 8),
        denormalize_by_extent=False,
    ):
        """
        Args:
            nIn: input feature channel
            spatial_pooltype: max | soft
            spatial_topk: 1
            flat_op: flatten | avg | avg-max | avg-max-min
        """
        super().__init__()
        self.featdim = featdim
        self.mask_attention_type = mask_attention_type
        self.flat_op = flat_op
        self.denormalize_by_extent = denormalize_by_extent
        self.rot_dim = rot_dim

        if self.mask_attention_type == "concat":
            nIn += 1

        conv_act = get_nn_act_func(act)
        if act == "relu":
            self.act = get_nn_act_func("lrelu")  # legacy model
        else:
            self.act = get_nn_act_func(act)
        # -----------------------------------
        self.drop_prob = drop_prob
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=dropblock_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=5000,
        )


        conv_layer = StdConv2d if use_ws else nn.Conv2d
        self.features = nn.ModuleList()

        for i in range(num_stride2_layers):
            add_dim1 = add_dim if i == 1 else 0
            _in_channels = nIn if i == 0 else featdim + add_dim1
            self.features.append(
                conv_layer(
                    _in_channels,
                    featdim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)

        for i in range(num_extra_layers):
            self.features.append(
                conv_layer(
                    featdim,
                    featdim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)

        final_h, final_w = final_spatial_size
        fc_in_dim = {
            "flatten": featdim * final_h * final_w,
            "avg": featdim,
            "avg-max": featdim * 2,
            "avg-max-min": featdim * 3,
        }[flat_op]

        # self.fc1 = nn.Linear(featdim * 8 * 8 + 128, 1024)  # NOTE: 128 for extents feature
        self.fc1 = nn.Linear(fc_in_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)


        self.fc1_z = nn.Linear(fc_in_dim, 1024)
        self.fc2_z = nn.Linear(1024, 256)
        if 'CAT' in FLAGS.z_type:
            self.fc_z = nn.Linear(256, 6)
            self.fc_t = nn.Linear(256, 2*6)

        else:
            self.fc_z = nn.Linear(256, 1)
            self.fc_t = nn.Linear(256, 2)

        self.resize_func_input = Resize(FLAGS.out_res, interpolation=InterpolationMode.NEAREST)

        # init ------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_t, std=0.01)


    def forward(self, coor_feat,  mask_attention=None, region=None, cat_id=None):
        """
        Args:
             since this is the actual correspondence
            x: (B,C,H,W)
            extents: (B, 3)
        Returns:

        """

        if region is not None:
            if coor_feat is not None:
                x = torch.cat([coor_feat, region], dim=1)
            else:
                x = region
        else:
            x = coor_feat

        if self.mask_attention_type != "none":
            assert mask_attention is not None
            # mask_attention = self.resize_func_input(mask_attention)
            if self.mask_attention_type == "mul":
                x = x * mask_attention
            elif self.mask_attention_type == "concat":
                x = torch.cat([x, mask_attention], dim=1)
            else:
                raise ValueError(f"Wrong mask attention type: {self.mask_attention_type}")

        if self.drop_prob > 0:
            self.dropblock.step()  # increment number of iterations
            x = self.dropblock(x)


        for _i, layer in enumerate(self.features):
            x = layer(x)

        flat_conv_feat = x.flatten(2)  # [B,featdim,*]
        if self.flat_op == "flatten":
            flat_conv_feat = flat_conv_feat.flatten(1)
        elif self.flat_op == "avg":
            flat_conv_feat = flat_conv_feat.mean(-1)  # spatial global average pooling
        elif self.flat_op == "avg-max":
            flat_conv_feat = torch.cat([flat_conv_feat.mean(-1), flat_conv_feat.max(-1)[0]], dim=-1)
        elif self.flat_op == "avg-max-min":
            flat_conv_feat = torch.cat(
                [
                    flat_conv_feat.mean(-1),
                    flat_conv_feat.max(-1)[0],
                    flat_conv_feat.min(-1)[0],
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Invalid flat_op: {self.flat_op}")

        #
        x = self.act(self.fc1(flat_conv_feat))
        x = self.act(self.fc2(x))
        #
        t = self.fc_t(x)

        xz = self.act(self.fc1_z(flat_conv_feat))
        xz = self.act(self.fc2_z(xz))
        z = self.fc_z(xz)

        if 'CAT' in FLAGS.z_type:
            bs = z.shape[0]
            z = z[np.arange(bs), cat_id]
            z = z.view(-1, 1)
            t = t.view(bs, 6, 2)
            t = t[np.arange(bs), cat_id]

        t = torch.cat([t, z], dim=1)

        return t

class ConvPnPNet_R(nn.Module):
    def __init__(
        self,
        nIn,
        mask_attention_type="none",
        featdim=128,
        add_dim=0,
        rot_dim=6,
        num_stride2_layers=3,
        num_extra_layers=0,
        use_ws=False,
        norm="GN",
        num_gn_groups=32,
        act="relu",
        drop_prob=0.0,
        dropblock_size=5,
        flat_op="flatten",
        final_spatial_size=(8, 8),
        denormalize_by_extent=False,
    ):
        """
        Args:
            nIn: input feature channel
            spatial_pooltype: max | soft
            spatial_topk: 1
            flat_op: flatten | avg | avg-max | avg-max-min
        """
        super().__init__()
        self.featdim = featdim
        self.mask_attention_type = mask_attention_type
        self.flat_op = flat_op
        self.denormalize_by_extent = denormalize_by_extent
        self.rot_dim = rot_dim

        if self.mask_attention_type == "concat":
            nIn += 1

        conv_act = get_nn_act_func(act)
        if act == "relu":
            self.act = get_nn_act_func("lrelu")  # legacy model
        else:
            self.act = get_nn_act_func(act)
        # -----------------------------------
        self.drop_prob = drop_prob
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=dropblock_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=5000,
        )


        conv_layer = StdConv2d if use_ws else nn.Conv2d
        self.features = nn.ModuleList()

        for i in range(num_stride2_layers):
            add_dim1 = add_dim if i == 1 else 0
            _in_channels = nIn if i == 0 else featdim + add_dim1
            self.features.append(
                conv_layer(
                    _in_channels,
                    featdim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)

        for i in range(num_extra_layers):
            self.features.append(
                conv_layer(
                    featdim,
                    featdim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)

        final_h, final_w = final_spatial_size
        fc_in_dim = {
            "flatten": featdim * final_h * final_w,
            "avg": featdim,
            "avg-max": featdim * 2,
            "avg-max-min": featdim * 3,
        }[flat_op]

        # self.fc1 = nn.Linear(featdim * 8 * 8 + 128, 1024)  # NOTE: 128 for extents feature
        self.fc1 = nn.Linear(fc_in_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_r = nn.Linear(256, 3)  # quat or rot6d

        self.resize_func_input = Resize(FLAGS.out_res, interpolation=InterpolationMode.NEAREST)

        # init ------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_r, std=0.01)


    def forward(self, coor_feat,  mask_attention=None, region=None, cat_id=None):
        """
        Args:
             since this is the actual correspondence
            x: (B,C,H,W)
            extents: (B, 3)
        Returns:

        """
        assert region is None
        x = coor_feat

        if self.mask_attention_type != "none":
            assert mask_attention is not None
            # mask_attention = self.resize_func_input(mask_attention)
            if self.mask_attention_type == "mul":
                x = x * mask_attention
            elif self.mask_attention_type == "concat":
                x = torch.cat([x, mask_attention], dim=1)
            else:
                raise ValueError(f"Wrong mask attention type: {self.mask_attention_type}")

        if self.drop_prob > 0:
            self.dropblock.step()  # increment number of iterations
            x = self.dropblock(x)


        for _i, layer in enumerate(self.features):
            x = layer(x)

        flat_conv_feat = x.flatten(2)  # [B,featdim,*]
        if self.flat_op == "flatten":
            flat_conv_feat = flat_conv_feat.flatten(1)
        elif self.flat_op == "avg":
            flat_conv_feat = flat_conv_feat.mean(-1)  # spatial global average pooling
        elif self.flat_op == "avg-max":
            flat_conv_feat = torch.cat([flat_conv_feat.mean(-1), flat_conv_feat.max(-1)[0]], dim=-1)
        elif self.flat_op == "avg-max-min":
            flat_conv_feat = torch.cat(
                [
                    flat_conv_feat.mean(-1),
                    flat_conv_feat.max(-1)[0],
                    flat_conv_feat.min(-1)[0],
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Invalid flat_op: {self.flat_op}")

        #
        x = self.act(self.fc1(flat_conv_feat))
        x = self.act(self.fc2(x))
        #
        rot_axis = self.fc_r(x)

        return rot_axis

if __name__ == '__main__':
    pnp_net = ConvPnPNet(3+2)
    import ipdb
    ipdb.set_trace()
    a=1
