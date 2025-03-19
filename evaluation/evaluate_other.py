import os
import torch
from config.config import *
from absl import app

FLAGS = flags.FLAGS
from evaluation.load_data_eval import NocsDataset
from evaluation.load_data_eval_gt_mask import NocsDatasetGtMask
import torch.nn as nn
import numpy as np
import time

# from creating log
from evaluation.eval_utils_cass import compute_degree_cm_mAP, setup_logger
from tqdm import tqdm

from network.PoseNet import PoseNet
import pickle
import _pickle as cPickle

device = 'cuda'


def evaluate(argv):
    output_path = FLAGS.model_save
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    logger = setup_logger('eval_log', os.path.join(FLAGS.model_save, 'log_eval.txt'))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if FLAGS.dataset == 'Real':
        img_list_path = 'Real/test_list.txt'
    else:
        img_list_path = 'CAMERA/val_list.txt'
    data_dir = FLAGS.dataset_dir
    img_list = [os.path.join(img_list_path.split('/')[0], line.rstrip('\n'))
                 for line in open(os.path.join(data_dir, img_list_path))]
    pred_result_save_path = os.path.join(output_path, 'pred_result.pkl')
    if os.path.exists(pred_result_save_path):
        with open(pred_result_save_path, 'rb') as file:
            pred_results = pickle.load(file)
    else:
        pred_results = []
        for i, img_path in tqdm(enumerate(img_list, 1)):
            scene = img_path.split('/')[-2]
            img_id = img_path.split('/')[-1]
            if FLAGS.dataset == 'Real':
                dataset_split = 'REAL275'
                detection_file = os.path.join(FLAGS.result_dir, f'results_test_{scene}_{img_id}.pkl')
            else:
                dataset_split = 'CAMERA25'
                detection_file = os.path.join(FLAGS.result_dir, f'results_val_{scene}_{img_id}.pkl')
            if not os.path.exists(detection_file):
                print(detection_file)
                continue
            with open(detection_file, 'rb') as file:
                detection_dict = cPickle.load(file)
                detection_dict['image_path'] = img_path.replace('Real/real_', 'data/real/')
            pred_results.append(detection_dict)
        with open(pred_result_save_path, 'wb') as file:
            pickle.dump(pred_results, file)

    if FLAGS.eval_inference_only:
        import sys
        sys.exit()

    if FLAGS.eval_precise:
        degree_thres_list = list(range(0, 61, 1))
        shift_thres_list = [i / 2 for i in range(21)]
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
                              iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=FLAGS.use_match_for_pose,)

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
            messages.append('10 degree: {:.1f}'.format(pose_aps[idx, degree_10_idx, -1] * 100))
            messages.append('10cm: {:.1f}'.format(pose_aps[idx, -1, shift_10_idx] * 100))

    for msg in messages:
        logger.info(msg)

    if FLAGS.eval_precise:
        degree_thres_list = list(range(0, 61, 1))
        shift_thres_list = [i for i in range(151)]
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
        curr_result['gt_RTs'] = gt_rts
        pred_rts = curr_result['pred_RTs']
        pred_scale = np.cbrt(np.linalg.det(pred_rts[:, :3, :3]))
        pred_rts[:, :3, :] = pred_rts[:, :3, :] / pred_scale[:, None, None]
        curr_result['pred_RTs'] = pred_rts
        norm_results.append(curr_result)


    iou_aps, pose_aps = compute_degree_cm_mAP(norm_results, synset_names, output_path, degree_thres_list, shift_thres_list,
                              iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=FLAGS.use_match_for_pose,plot_figure=True)

    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_05_idx = shift_thres_list.index(20)
    shift_10_idx = shift_thres_list.index(50)

    messages = []
    if FLAGS.per_obj in synset_names:
        idx = synset_names.index(FLAGS.per_obj)
    else:
        idx = -1

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