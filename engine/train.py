import os.path
import random
import torch
from absl import app
import sys
from config.config import *
from tools.eval_utils import setup_logger
torch.autograd.set_detect_anomaly(True)

FLAGS = flags.FLAGS

from datasets.load_data_nocs import NocsDataset
import numpy as np
import time
import mmcv
from tqdm import tqdm
from network.point_sample.point_sample import Depth2PC
from network.PoseNet import PoseNet
from losses.pose_loss import PoseLoss
import traceback
from tools.training_utils import build_lr_rate, build_optimizer
from torch.utils.tensorboard import SummaryWriter
from datasets.data_augmentation import pc_augment

torch.autograd.set_detect_anomaly(True)
device = 'cuda'

def train(argv):
    assert FLAGS.per_obj in ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug', 'all', 'sym', 'asym']
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    FLAGS.append_flags_into_file(os.path.join(FLAGS.model_save, 'flags.txt'))
    logger = setup_logger('train_log', os.path.join(FLAGS.model_save, 'log.txt'))
    writer = SummaryWriter(log_dir=FLAGS.model_save)
    network = PoseNet().to(device)
    network.train()

    if FLAGS.resume_point > 0:
        FLAGS.resume_model = os.path.join(FLAGS.model_save, f'model_{FLAGS.resume_point-1:02d}.pth')
        start_epoch = FLAGS.resume_point

    if len(FLAGS.resume_model) > 0:
        model_dict = network.state_dict()
        resume_model_dict = torch.load(FLAGS.resume_model)
        model_dict.update(resume_model_dict)
        network.load_state_dict(model_dict)
        start_epoch = FLAGS.resume_point
    else:
        start_epoch = 0

    pose_loss = PoseLoss()


    train_dataset = NocsDataset(source=FLAGS.dataset, mode='train',
                                data_dir=FLAGS.dataset_dir, per_obj=FLAGS.per_obj)

    # start training datasets sampler
    st_time = time.time()
    train_steps = FLAGS.train_size // FLAGS.batch_size
    global_step = train_steps * start_epoch  # record the number iteration
    train_size = train_steps * FLAGS.batch_size
    indices = []
    page_start = - train_size

    #  build optimizer
    param_list = network.build_params_optimizer(training_stage_freeze=[])
    if FLAGS.optimizer_type == 'Ranger':
        optimizer = build_optimizer(param_list)
    elif FLAGS.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(param_list, lr=FLAGS.lr)
    else:
        optimizer = torch.optim.Adam(param_list)

    if FLAGS.resume_point > 0:
        optimizer_path = os.path.join(FLAGS.model_save, f'last_optimizer.pth')
        optimizer_dict = torch.load(optimizer_path)
        optimizer.load_state_dict(optimizer_dict)

    optimizer.zero_grad()  # first clear the grad
    scheduler = build_lr_rate(optimizer, total_iters=train_steps * FLAGS.total_epoch // FLAGS.accumulate)

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
                n_repeat = (train_size - len_last) // ((syn_ratio+1) * real_len) + 1
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
            # estimate pose
            output_dict = network(data, device, do_loss=True)

            # loss & backward
            loss_dict = pose_loss(output_dict, data)
            total_loss = sum(loss_dict.values())
            total_loss /= FLAGS.accumulate
            try:
                if global_step % FLAGS.accumulate == 0:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
            except Exception as e:
                optimizer.zero_grad()
                logger.warning(f'error occur! {str(e)}! traceback:')
                logger.warning(traceback.print_exc())
                exception_path = os.path.join(FLAGS.model_save, 'exception')
                if not os.path.exists(exception_path):
                    os.makedirs(exception_path)
                torch.save(data, os.path.join(exception_path, 'data.pth'))
                torch.save(network.state_dict(), os.path.join(exception_path, 'model.pth'))
                torch.save(output_dict, os.path.join(exception_path, 'output_dict.pth'))
                continue

            global_step += 1
            writer.add_scalar(f'Loss/total_loss', total_loss.item(), global_step)
            for loss_name in loss_dict.keys():
                writer.add_scalar(f'Loss/{loss_name}', loss_dict[loss_name].item(), global_step)

            if i_step % FLAGS.log_every == 0:
                msg = 'Batch {0} Loss:{1:f}'.format(i_step, total_loss.item())
                for loss_name in loss_dict.keys():
                    msg += f' {loss_name}={loss_dict[loss_name].item()}'
                logger.info(msg)

        # save model
        if (epoch + 1) % FLAGS.save_every == 0 or (epoch + 1) == FLAGS.total_epoch:
            torch.save(network.state_dict(), '{0}/model_{1:02d}.pth'.format(FLAGS.model_save, epoch))
            torch.save(optimizer.state_dict(), '{0}/last_optimizer.pth'.format(FLAGS.model_save, epoch))

if __name__ == "__main__":
    app.run(train)