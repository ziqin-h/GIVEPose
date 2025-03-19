import os
import torch
import mmcv
from config.config import *
from absl import app

FLAGS = flags.FLAGS
from evaluation.load_data_eval import NocsDataset
from evaluation.load_data_eval_wild import WildDataset
# from evaluation.load_data_eval_gt_mask import NocsDatasetGtMask
from datasets.load_data_nocs import convert_to_one_hot
import torch.nn as nn
import numpy as np
import time

# from evaluation.eval_utils import setup_logger, compute_mAP
from evaluation.eval_utils_cass import compute_degree_cm_mAP, setup_logger
from tqdm import tqdm

from network.PoseNet import PoseNet
from network.scale_net import Scale_net
import pickle

device = 'cuda'


def evaluate(argv):
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    model_name = os.path.basename(FLAGS.resume_model).split('.')[0]
    # FLAGS.model_save = os.path.dirname(FLAGS.resume_model)
    # build dataset and dataloader
    if FLAGS.dataset == 'wild6d':
        val_dataset = WildDataset(source=FLAGS.dataset, mode='test')
    else:
        val_dataset = NocsDataset(source=FLAGS.dataset, mode='test')
    output_path = os.path.join(FLAGS.model_save, f'eval_result_{model_name}')
    print(f"save to {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pred_result_save_path = os.path.join(output_path, 'pred_result.pkl')
    logger = setup_logger('eval_log', os.path.join(output_path, 'log_eval.txt'))
    if FLAGS.dataset == 'wild6d':
        logger = setup_logger('eval_log', os.path.join(output_path, f'log_eval_{FLAGS.per_obj}.txt'))
        pred_result_save_path = os.path.join(output_path, f'pred_result_{FLAGS.per_obj}.pkl')
    if os.path.exists(pred_result_save_path):
        with open(pred_result_save_path, 'rb') as file:
            pred_results = pickle.load(file)
            print(f'load from {pred_result_save_path}')
    else:
        network = PoseNet().to(device)

        model_dict = network.state_dict()
        resume_model_dict = torch.load(FLAGS.resume_model)
        model_dict.update(resume_model_dict)
        network.load_state_dict(model_dict)

        # start to test
        network = network.eval()
        assert FLAGS.use_scale_net
        scale_net = Scale_net(feat_dim=FLAGS.feat_dim, use_hw=True, backbone=FLAGS.backbone).to(device)
        model_dict = scale_net.state_dict()
        if FLAGS.dataset=='Real' or FLAGS.dataset=='wild6d':
            print('Eval on Real')
            if len(FLAGS.sn_path) != 0:
                resume_model_dict = torch.load(FLAGS.sn_path)
                print(f'use scale net:{FLAGS.sn_path}')
            else:
                resume_model_dict = torch.load("/path/to/scale_net_Real")
                # raise NotImplementedError
        else:
            print('Load CAMERA scale_net')
            if len(FLAGS.sn_path) != 0:
                resume_model_dict = torch.load(FLAGS.sn_path)
            else:
                resume_model_dict = torch.load("/path/to/scale_net_CAMERA")
                #raise NotImplementedError
        model_dict.update(resume_model_dict)
        scale_net.load_state_dict(model_dict)
        # start to test
        scale_net = scale_net.eval()

        pred_results = []
        def my_collate_fn(list_of_data):
            if len(list_of_data) == 1:
                return list_of_data[0]
            else:
                return NotImplementedError
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=FLAGS.eval_batch_size, collate_fn=my_collate_fn,
                                                     num_workers=1, pin_memory=True)
        for i, data in tqdm(enumerate(val_dataloader, 1)):
            if data is None:
                continue
            data_origin, detection_dict, gts = data
            origin_object_num = len(data_origin['cat_id_0_base'])
            batch_per_max = FLAGS.batch_size
            batch_step_num = np.ceil(origin_object_num / batch_per_max)
            batch_step_num = int(batch_step_num)
            assert batch_step_num <= 1, f"true: {batch_step_num}"

            data = data_origin
            one_hot = [convert_to_one_hot(i, 6) for i in data['cat_id_0_base']]
            data['one_hot'] = torch.as_tensor(np.array(one_hot), dtype=torch.float32).contiguous()
            sym = data['sym_info'].to(device)
            if len(data['cat_id_0_base']) == 0:
                detection_dict['pred_RTs'] = np.zeros((0, 4, 4))
                detection_dict['pred_scales'] = np.zeros((0, 4, 4))
                pred_results.append(detection_dict)
                continue

            with torch.no_grad():
                pred_scale = scale_net(data, device, 'test')
                pred_scale = pred_scale.cpu()
                output_dict = network.forward(data, device, pred_scale=pred_scale)
                pred_rot = output_dict['rot']
                pred_trans = output_dict['trans']
                pred_size = output_dict['size']
                pred_size = torch.nn.functional.normalize(pred_size, p=2, dim=1)
                # pred_scale = output_dict['nocs_scale'].cpu()
                bs = pred_rot.shape[0]
                pred_RT = torch.zeros([bs, 4, 4])
                pred_RT[:, :3, :3] = pred_rot.cpu()
                pred_RT[:, :3, 3] = pred_trans.cpu()
                pred_RT[:, 3, 3] = 1
                pred_RT[:, :3, :] = pred_RT[:, :3, :] * pred_scale[:, None, None]
                pred_RT = pred_RT.detach().cpu().numpy()
                pred_size = pred_size.detach().cpu().numpy()
            if pred_RT is not None:
                detection_dict['pred_RTs'] = pred_RT
                detection_dict['pred_scales'] = pred_size
            else:
                assert NotImplementedError
            pred_results.append(detection_dict)
        #if not FLAGS.save_nocs_map:
        with open(pred_result_save_path, 'wb') as file:
            pickle.dump(pred_results, file)

    if FLAGS.eval_inference_only:
        import sys
        sys.exit()
    if FLAGS.eval_precise:
        degree_thres_list = list(range(0, 71, 1))
        shift_thres_list = [i / 2 for i in range(51)]
        iou_thres_list = [i / 100 for i in range(101)]
    else:
        degree_thres_list = [5, 10, 360]
        shift_thres_list = [5, 10, 1e4]
        iou_thres_list = [0.1, 0.25, 0.5, 0.75]

    if FLAGS.ban_mug:
        synset_names = ['BG'] + ['bottle', 'bowl', 'camera', 'can', 'laptop']
    else:
        synset_names = ['BG'] + ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

    if FLAGS.per_obj in synset_names:
        idx = synset_names.index(FLAGS.per_obj)
    else:
        idx = -1

    iou_aps, pose_aps = compute_degree_cm_mAP(pred_results, synset_names, output_path, degree_thres_list, shift_thres_list,
                              iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=FLAGS.use_match_for_pose,plot_figure=False)
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_05_idx = shift_thres_list.index(5)
    shift_10_idx = shift_thres_list.index(10)

    messages = []

    if FLAGS.per_obj in synset_names:
        messages.append('mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
        messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
        messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
        messages.append('10 degree: {:.1f}'.format(pose_aps[idx, degree_10_idx, -1] * 100))
        messages.append('10cm: {:.1f}'.format(pose_aps[idx, -1, shift_10_idx] * 100))
    else:
        messages.append('average mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
        messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
        messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
        messages.append('10 degree: {:.1f}'.format(pose_aps[idx, degree_10_idx, -1] * 100))
        messages.append('10cm: {:.1f}'.format(pose_aps[idx, -1, shift_10_idx] * 100))
        for idx in range(1, len(synset_names)):
            messages.append('category {}'.format(synset_names[idx]))
            messages.append('mAP:')
            messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
            messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
            messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
            messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
            messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
            messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))

    for msg in messages:
        logger.info(msg)

    if FLAGS.eval_precise:
        degree_thres_list = list(range(0, 61, 1))
        shift_thres_list = [i for i in range(51)]
        iou_thres_list = [i / 100 for i in range(101)]
    else:
        degree_thres_list = [5, 10]
        shift_thres_list = [5, 10, 20, 50]
        iou_thres_list = [0.1, 0.25, 0.5, 0.75]

    norm_results = []
    for curr_result in pred_results:
        gt_rts = curr_result['gt_RTs']
        gt_scale = np.cbrt(np.linalg.det(gt_rts[:, :3, :3]))
        gt_rts[:, :3, :] = gt_rts[:, :3, :] / gt_scale[:, None, None]
        # gt_scale_abs = np.abs(gt_scale)
        # gt_rts[:, :3, :3] = gt_rts[:, :3, :3] / gt_scale[:, None, None]
        # gt_rts[:, :3, 3] = gt_rts[:, :3, 3] / gt_scale_abs[:, None]
        curr_result['gt_RTs'] = gt_rts
        pred_rts = curr_result['pred_RTs']
        pred_scale = np.cbrt(np.linalg.det(pred_rts[:, :3, :3]))
        pred_rts[:, :3, :] = pred_rts[:, :3, :] / pred_scale[:, None, None]
        curr_result['pred_RTs'] = pred_rts
        norm_results.append(curr_result)

    if FLAGS.per_obj in synset_names:
        idx = synset_names.index(FLAGS.per_obj)
    else:
        idx = -1

    iou_aps, pose_aps = compute_degree_cm_mAP(norm_results, synset_names, output_path, degree_thres_list, shift_thres_list,
                              iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=FLAGS.use_match_for_pose,)

    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_05_idx = shift_thres_list.index(20)
    shift_10_idx = shift_thres_list.index(50)

    messages = []

    if FLAGS.per_obj in synset_names:
        messages.append('mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
        messages.append('5 degree, 20%: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 20%: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
        messages.append('10 degree, 50%: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
        messages.append('10 degree: {:.1f}'.format(pose_aps[idx, degree_10_idx, -1] * 100))
        messages.append('20%: {:.1f}'.format(pose_aps[idx, -1, shift_05_idx] * 100))
        messages.append('50%: {:.1f}'.format(pose_aps[idx, -1, shift_10_idx] * 100))
    else:
        messages.append('average mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
        messages.append('5 degree, 20%: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 20%: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
        messages.append('10 degree, 50%: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
        messages.append('10 degree: {:.1f}'.format(pose_aps[idx, degree_10_idx, -1] * 100))
        messages.append('20%: {:.1f}'.format(pose_aps[idx, -1, shift_05_idx] * 100))
        messages.append('50%: {:.1f}'.format(pose_aps[idx, -1, shift_10_idx] * 100))
        for idx in range(1, len(synset_names)):
            messages.append('category {}'.format(synset_names[idx]))
            messages.append('mAP:')
            messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
            messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
            messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
            messages.append('5 degree, 20%: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
            messages.append('10 degree, 20%: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
            messages.append('10 degree, 50%: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
            messages.append('10 degree: {:.1f}'.format(pose_aps[idx, degree_10_idx, -1] * 100))
            messages.append('20%: {:.1f}'.format(pose_aps[idx, -1, shift_05_idx] * 100))
            messages.append('50%: {:.1f}'.format(pose_aps[idx, -1, shift_10_idx] * 100))

    for msg in messages:
        logger.info(msg)

if __name__ == "__main__":
    app.run(evaluate)