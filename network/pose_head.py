import torch.nn as nn
import torch
import torch.nn.functional as F
import absl.flags as flags
from absl import app
from mmcv.cnn import normal_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm
from network.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
import numpy as np
from torch.nn import init
from config import *
FLAGS = flags.FLAGS
# Point_center  encode the segmented point cloud
# one more conv layer compared to original paper


class SizeHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SizeHead, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feat_dim = FLAGS.feat_ts

        self.conv1 = torch.nn.Conv1d(self.in_dim, self.feat_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.feat_dim, self.out_dim, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(self.feat_dim)
        self._init_weights()

    def forward(self, x):
        if isinstance(x, (tuple, list)) and len(x) == 1:
            x = x[0]
        # bs,1024,8,8
        x = x.flatten(2,3).max(dim=-1, keepdim=True).values
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = self.conv2(x)

        x = x.squeeze(2)
        x = x.contiguous()
        x1 = x[:, :3]
        return x1

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)

class Conv_feat_finetune(nn.Module):
    def __init__(self, in_dim=0, norm="GN",):
        super(Conv_feat_finetune, self).__init__()
        self.dino_feat_finetune = nn.ModuleList()
        conv_layer = nn.Conv2d
        conv_act = get_nn_act_func('relu')

        self.dino_feat_finetune.append(conv_layer(1024, 512, kernel_size=3, stride=2, padding=1, bias=False, ))
        self.dino_feat_finetune.append(get_norm(norm, 512, num_gn_groups=32))
        self.dino_feat_finetune.append(conv_act)
        self.dino_feat_finetune.append(conv_layer(512, 128, kernel_size=3, stride=2, padding=1, bias=False, ))
        self.dino_feat_finetune.append(get_norm(norm, 1024, num_gn_groups=32))
        self.dino_feat_finetune.append(conv_act)

    def forward(self, dino_feat):
        x = dino_feat
        for _i, layer in enumerate(self.dino_feat_finetune):
            x = layer(x)
        return x

class ChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels, norm="GN",):
        super(ChannelReducer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.norm = get_norm(norm, out_channels, num_gn_groups=32)
        self.act = get_nn_act_func('relu')

    def forward(self, x):
        if isinstance(x, (tuple, list)) and len(x) == 1:
            x = x[0]
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class RHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RHead, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feat_dim = FLAGS.feat_ts

        self.conv1 = torch.nn.Conv1d(self.in_dim, self.feat_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.feat_dim, self.out_dim, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(self.feat_dim)
        self._init_weights()

    def forward(self, x):
        if isinstance(x, (tuple, list)) and len(x) == 1:
            x = x[0]
        # bs,1024,8,8
        x = x.flatten(2,3).max(dim=-1, keepdim=True).values
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = self.conv2(x)

        x = x.squeeze(2)
        x_ = x.contiguous()
        return x_

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)


class THead(nn.Module):
    def __init__(self, in_dim):
        super(THead, self).__init__()
        self.in_dim = in_dim
        self.out_dim = 3
        self.feat_dim = FLAGS.feat_ts

        self.conv1 = torch.nn.Conv1d(self.in_dim, self.feat_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.feat_dim, self.out_dim, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(self.feat_dim)
        self._init_weights()

    def forward(self, x):
        if isinstance(x, (tuple, list)) and len(x) == 1:
            x = x[0]
        # bs,1024,8,8
        x = x.flatten(2, 3).max(dim=-1, keepdim=True).values
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = self.conv2(x)

        x = x.squeeze(2)
        x_ = x.contiguous()
        return x_

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)

class TRHead_linear(nn.Module):
    def __init__(self, in_dim, feat_dim=128, flat_op='flatten'):
        super(TRHead_linear, self).__init__()
        fc_in_dim = {
            "flatten": feat_dim * 8 * 8,
            "avg": feat_dim,
            "avg-max": feat_dim * 2,
            "avg-max-min": feat_dim * 3,
        }[flat_op]
        add_dim = 0
        self.flat_op = flat_op
        self.act = get_nn_act_func("lrelu")
        self.conv1 = nn.Conv2d(in_dim, feat_dim, kernel_size=1)
        self.fc1 = nn.Linear(fc_in_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_r = nn.Linear(256+add_dim, 6)  #rot6d
        self.fc_t = nn.Linear(256+add_dim, 2)

        self.fc1_z = nn.Linear(fc_in_dim, 1024)
        self.fc2_z = nn.Linear(1024, 256)
        self.fc_z = nn.Linear(256+add_dim, 1)
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

    def forward(self, x, info2d=None):
        if isinstance(x, (tuple, list)) and len(x) == 1:
            x = x[0]
        # bs,1024,8,8
        x = self.act(self.conv1(x))
        flat_conv_feat = x.flatten(2)
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
        x = self.act(self.fc1(flat_conv_feat))
        x = self.act(self.fc2(x))
        #
        if info2d is not None:
            x = torch.cat([x, info2d], dim=1)
        rot = self.fc_r(x)
        t = self.fc_t(x)

        xz = self.act(self.fc1_z(flat_conv_feat))
        xz = self.act(self.fc2_z(xz))
        if info2d is not None:
            xz = torch.cat([xz, info2d], dim=1)
        z = self.fc_z(xz)

        t = torch.cat([t, z], dim=1)

        return rot, t

class Auxi_Head(nn.Module):
    def __init__(self, feat_dim=128, flat_op='flatten'):
        super(Auxi_Head, self).__init__()
        fc_in_dim = {
            "flatten": feat_dim * 8 * 8,
            "avg": feat_dim,
            "avg-max": feat_dim * 2,
            "avg-max-min": feat_dim * 3,
        }[flat_op]
        self.act = get_nn_act_func("lrelu")
        self.flat_op = flat_op
        self.fc1_cat = nn.Linear(fc_in_dim, 1024)
        self.fc2_cat = nn.Linear(1024, 256)
        self.fc_cat = nn.Linear(256, 6)  # rot6d

        self.fc1_view = nn.Linear(fc_in_dim, 1024)
        self.fc2_view = nn.Linear(1024, 256)
        self.fc_view = nn.Linear(256, 3*6)

    def forward(self, x, cat_0base_id=None):
        if isinstance(x, (tuple, list)) and len(x) == 1:
            x = x[0]
        bs = x.shape[0]
        # bs,1024,8,8
        flat_conv_feat = x.flatten(2)
        if self.flat_op == "flatten":
            flat_conv_feat = flat_conv_feat.flatten(1)
        else:
            raise NotImplementedError
        x_cat = self.act(self.fc1_cat(flat_conv_feat))
        x_cat = self.act(self.fc2_cat(x_cat))
        cat = self.fc_cat(x_cat)

        x_view = self.act(self.fc1_view(flat_conv_feat))
        x_view = self.act(self.fc2_view(x_view))
        view_axis = self.fc_view(x_view)
        view_axis = view_axis.view(bs, 6, 3)
        view_axis = view_axis[np.arange(bs), cat_0base_id]
        return cat, view_axis

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)

def main(argv):
    feature = torch.rand(3, 3, 1000)
    obj_id = torch.randint(low=0, high=15, size=[3, 1])
    net = SizeHead()
    out = net(feature, obj_id)
    t = 1

if __name__ == "__main__":
    app.run(main)
