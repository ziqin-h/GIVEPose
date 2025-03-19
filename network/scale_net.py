import torch.nn as nn
import torch
import torchvision
import absl.flags as flags
from absl import app
from mmcv.cnn import normal_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
from torch.nn import init
import random
import torch
from config.config import *
torch.autograd.set_detect_anomaly(True)
FLAGS = flags.FLAGS

from datasets.load_data_nocs import NocsDataset
import time
from torch.utils.tensorboard import SummaryWriter
# from network.resnet import resnet18, resnet34, resnet50


class Scale_net(nn.Module):
    def __init__(self,feat_dim=8, use_hw=True, backbone='mobilenetv3s', pretrained=True, cats_num=6):
        super(Scale_net, self).__init__()
        feature_extractor_bbox = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        feature_extractor_full = torchvision.models.mobilenet_v3_small(pretrained=pretrained)

        self.feat_encoder_bbox = nn.Sequential(feature_extractor_bbox.features, feature_extractor_bbox.avgpool, nn.Flatten())
        self.feat_encoder_full = nn.Sequential(feature_extractor_full.features, feature_extractor_full.avgpool, nn.Flatten())
        in_dim = feature_extractor_bbox.features[-1].out_channels * 2

        self.drop = nn.Dropout(p=0.2, inplace=True)

        self.line1 = nn.Linear(in_dim, 128)
        self.line2 = nn.Linear(128 + cats_num, feat_dim)
        self.relu = nn.ReLU(inplace=True)
        self.use_hw = use_hw
        if use_hw:
            feat_dim += 2
        self.line3 = nn.Linear(feat_dim + cats_num, 1)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten())

    def forward(self, data, device, mode=''):
        roi_img = data['roi_img'].to(device)
        full_img = data['full_img'].to(device)
        one_hot = data['one_hot'].to(device)
        feat_roi = self.feat_encoder_bbox(roi_img)
        feat_roi = self.drop(feat_roi)
        feat_full = self.feat_encoder_full(full_img)
        feat_full = self.drop(feat_full)
        feat = torch.cat([feat_roi,feat_full], dim=1)
        x = self.line1(feat)
        x = self.relu(x)
        x = torch.cat([x, one_hot], dim=1)
        x = self.line2(x)
        x = self.relu(x)
        x = torch.cat([x, one_hot], dim=1)
        hw = data["roi_wh"].to(device) / 100
        if self.use_hw:
            x = torch.cat([x,hw], dim=1)
        resi_scale = self.line3(x).squeeze()
        mean_scale = data['mean_size'].to(device).norm(dim=1)
        scale = resi_scale + mean_scale
        return scale

    def build_params_optimizer(self, training_stage_freeze=None):
        params_lr_list = []
        if 'backbone' in training_stage_freeze:
            for param in self.feat_encoder.parameters():
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

def main(argv):
    torch.autograd.set_detect_anomaly(True)
    device = 'cuda'
    network = Scale_net(backbone='resnet18').to(device)
    network.train()
    start_epoch = 0
    train_dataset = NocsDataset(source=FLAGS.dataset, mode='train',
                                data_dir=FLAGS.dataset_dir, per_obj=FLAGS.per_obj)
    # start training datasets sampler
    st_time = time.time()
    train_steps = FLAGS.train_size // FLAGS.batch_size
    global_step = train_steps * start_epoch  # record the number iteration
    train_size = train_steps * FLAGS.batch_size
    indices = []
    page_start = - train_size

    for epoch in range(start_epoch, FLAGS.total_epoch):
        # train one epoch
        print('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                                ', ' + 'Epoch %02d' % epoch + ', ' + 'Training started'))
        # create optimizer and adjust learning rate accordingly
        # sample train subset
        page_start += train_size
        len_last = len(indices) - page_start
        if len_last < train_size:
            indices = indices[page_start:]
            if FLAGS.dataset == 'CAMERA+Real':
                # CAMERA : Real = 3 : 1
                syn_ratio = FLAGS.syn_ratio
                camera_len = train_dataset.subset_len[0]
                real_len = train_dataset.subset_len[1]
                real_indices = list(range(camera_len, camera_len + real_len))
                camera_indices = list(range(camera_len))
                n_repeat = (train_size - len_last) // ((syn_ratio + 1) * real_len) + 1
                data_list = random.sample(camera_indices, int(syn_ratio * n_repeat * real_len)) + real_indices
                random.shuffle(data_list)
                indices += data_list
            else:
                data_list = list(range(train_dataset.length))
                for i_step in range((train_size - len_last) // train_dataset.length + 1):
                    random.shuffle(data_list)
                    indices += data_list
            page_start = 0
        train_idx = indices[page_start:(page_start + train_size)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                       sampler=train_sampler,
                                                       num_workers=FLAGS.num_workers, pin_memory=True)
        time1 = time.time()
        for i_step, data in enumerate(train_dataloader, 1):
            output_dict = network(data, device)
if __name__=="__main__":
    app.run(main)