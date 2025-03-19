import copy
import os
import cv2
import math
import random
import numpy as np
import _pickle as cPickle
from config.config import *
FLAGS = flags.FLAGS

import torch
import json
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from tools.eval_utils import load_depth, get_real_hw, get_bbox_ori, calculate_iou
from tools.dataset_utils import *
from evaluation.eval_utils_cass import get_3d_bbox, transform_coordinates_3d, compute_3d_iou_new

class WildDataset(data.Dataset):
    def __init__(self, source=None, mode='test',
                img_size=256):
        '''
        :param source: 'CAMERA' or 'Real' or 'CAMERA+Real'
        :param mode: 'train' or 'test'
        :param data_dir: 'path to dataset'
        :param n_pts: 'number of selected sketch point', no use here
        :param img_size: cropped image size
        '''
        self.source = source
        self.mode = mode
        data_dir = FLAGS.dataset_dir
        file_path = 'test_list_{}.txt'.format(FLAGS.per_obj)
        self.cat_name2id = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}


        self.img_list = [line.rstrip('\n').replace('rgbd', 'images').replace('UCSD_POSE_RGBD', 'Wild6D') \
                    for line in open(os.path.join('./data/Wild6D/test_set/', file_path))] # dataset path
        self.length = len(self.img_list)
        self.real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]],
                                        dtype=np.float32)
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)
        print('{} images found.'.format(self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_list[index]
        mask_path = img_path.replace('.jpg', '-mask.png')
        if not os.path.exists(mask_path):
            return None
        raw_rgb = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        # dataset path
        meta = json.load(open(
            os.path.join("./data/Wild6D/test_set", FLAGS.per_obj, img_path.split('/')[-4], img_path.split('/')[-3], 'metadata')))
        cam = np.array(meta['K']).reshape(3, 3).T

        im_H, im_W = raw_rgb.shape[0], raw_rgb.shape[1]
        mask = mask[:, :, 2]
        mask = mask / 255.
        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)
        horizontal_indicies = np.where(np.any(mask, axis=0))[0]
        vertical_indicies = np.where(np.any(mask, axis=1))[0]
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bbox_center = np.array([cx, cy])  # (w/2, h/2)
        if FLAGS.DZI_TYPE == 'none':
            FLAGS.DZI_PAD_SCALE = 1.0
        img_scale = max(y2 - y1, x2 - x1)
        img_scale = min(img_scale, max(im_H, im_W)) * 1.5

        roi_imgs = []
        roi_masks = []
        sym_infos = []
        roi_whs = []
        obj_ids_0base = []
        mean_shapes = []
        roi_coord_2ds = []
        resize_ratios = []
        bbox_centers = []


        num_instance = 1
        cat_id = self.cat_name2id[FLAGS.per_obj]

        for j in range(num_instance):
            roi_img = crop_resize_by_warp_affine(
                raw_rgb, bbox_center, img_scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
            )
            roi_coord_2d = crop_resize_by_warp_affine(
                coord_2d, bbox_center, img_scale, FLAGS.out_res, interpolation=cv2.INTER_NEAREST
            ).transpose(2, 0, 1)
            mask_target = mask.copy().astype(np.float32)
            roi_mask = crop_resize_by_warp_affine(
                mask_target, bbox_center, img_scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
            )
            roi_mask = np.expand_dims(roi_mask, axis=0)
            roi_img = (roi_img / 255.0 - self.img_mean) / self.img_std
            roi_img = roi_img.transpose(2, 0, 1)
            sym_info = self.get_sym_info(cat_id)
            bw, bh = x2-x1, y2-y1

            mean_shape = self.get_mean_shape(FLAGS.per_obj)
            mean_shape = mean_shape / 1000.0

            roi_imgs.append(roi_img)
            roi_masks.append(roi_mask)
            sym_infos.append(sym_info)
            obj_ids_0base.append(cat_id-1)
            roi_whs.append(np.array([bw, bh], dtype=np.float32))
            mean_shapes.append(mean_shape)
            roi_coord_2ds.append(roi_coord_2d)
            resize_ratios.append(FLAGS.out_res / img_scale)
            bbox_centers.append(bbox_center)
        full_img = cv2.resize(raw_rgb, (256, 256)) if FLAGS.resize_full else raw_rgb
        full_img = (full_img / 255.0 - self.img_mean) / self.img_std
        full_img = full_img.transpose(2, 0, 1)

        roi_imgs = np.array(roi_imgs)
        roi_masks = np.array(roi_masks)
        sym_infos = np.array(sym_infos)
        obj_ids_0base = np.array(obj_ids_0base)
        mean_shapes = np.array(mean_shapes)
        roi_coord_2ds = np.array(roi_coord_2ds)

        data_dict = {}
        out_camK = np.array([cam] * len(roi_imgs))
        full_imgs = np.array([full_img] * len(roi_imgs))
        data_dict['roi_img'] = torch.as_tensor(roi_imgs.astype(np.float32)).contiguous()
        data_dict['roi_mask'] = torch.as_tensor(roi_masks.astype(np.float32)).contiguous()
        data_dict['full_img'] = torch.as_tensor(full_imgs.astype(np.float32)).contiguous()
        data_dict['cam_K'] = torch.as_tensor(out_camK.astype(np.float32)).contiguous()
        data_dict['mean_size'] = torch.as_tensor(mean_shapes, dtype=torch.float32).contiguous()
        data_dict['roi_wh'] = torch.as_tensor(np.array(roi_whs), dtype=torch.float32).contiguous()
        data_dict['sym_info'] = torch.as_tensor(sym_infos.astype(np.float32)).contiguous()
        data_dict['cat_id_0_base'] = torch.as_tensor(obj_ids_0base)
        data_dict['roi_coord_2d'] = torch.as_tensor(roi_coord_2ds, dtype=torch.float32).contiguous()
        data_dict["resize_ratio"] = torch.as_tensor(np.array(resize_ratios), dtype=torch.float32).contiguous()
        data_dict["bbox_center"] = torch.as_tensor(np.array(bbox_centers), dtype=torch.float32).contiguous()
        data_dict['img_path'] = img_path

        detection_dict = {}
        gt_path = os.path.join('./data/Wild6D/test_set/pkl_annotations/', FLAGS.per_obj, \
                           FLAGS.per_obj + '-' + img_path.split('/')[-4] + '-' + img_path.split('/')[-3] + '.pkl')

        if not os.path.exists(gt_path):
            print("Not found the ground truth from {}".format(gt_path))
            return None
        gts = cPickle.load(open(gt_path, 'rb'))
        frame_idx = int(img_path.split('/')[-1].split('.jpg')[0])
        if frame_idx >= len(gts['annotations']):
            return None
        gts = gts['annotations'][frame_idx]
        if FLAGS.per_obj=='camera':
            gts['class_id'] = 2
        elif FLAGS.per_obj=='laptop':
            gts['class_id'] = 4
        elif FLAGS.per_obj=='mug':
            gts['class_id'] = 5
        detection_dict['gt_class_ids'] = np.array([gts['class_id']+1])
        detection_dict['gt_bboxes'] = np.array([[y1, x1, y2, x2]])
        gt_RTs = np.eye(4)
        gt_RTs[:3, :3] = -gts['rotation']*np.linalg.norm(gts['size'])
        gt_RTs[:3, 3] = gts['translation']
        detection_dict['gt_RTs'] = gt_RTs[np.newaxis, ...]
        detection_dict['gt_scales'] = gts['size'][np.newaxis, ...]/np.linalg.norm(gts['size'])
        detection_dict['gt_handle_visibility'] = np.array([1])
        detection_dict['cam_K'] = np.array(meta['K']).reshape(3, 3).T

        detection_dict['pred_class_ids'] = np.array([gts['class_id']+1])
        detection_dict['pred_bboxes'] =np.array([[y1, x1, y2, x2]])
        detection_dict['pred_scores'] = np.array([1.0])
        detection_dict['image_path'] = img_path

        return data_dict, detection_dict, gts

    def get_sym_info(self, c):
        #  sym_info  c0 : face classfication  c1, c2, c3:Three view symmetry, correspond to xy, xz, yz respectively
        # c0: 0 no symmetry 1 axis symmetry 2 two reflection planes 3 unimplemented type
        #  Y axis points upwards, x axis pass through the handle, z axis otherwise
        #
        # for specific defination, see sketch_loss
        if c == 1:
            sym = np.array([1, 1, 0, 1], dtype=int)
        elif c == 2:
            sym = np.array([1, 1, 0, 1], dtype=int)
        elif c == 3:
            sym = np.array([0, 0, 0, 0], dtype=int)
        elif c == 4:
            sym = np.array([1, 1, 1, 1], dtype=int)
        elif c == 5:
            sym = np.array([0, 1, 0, 0], dtype=int)
        elif c == 6:
            sym = np.array([0, 1, 0, 0], dtype=int)  # for mug, we currently mark it as no symmetry
        else:
            raise NotImplementedError(f'sym info {c}')
        return sym

    def get_mean_shape(self, c):
        if c == 'bottle':
            unitx = 87
            unity = 220
            unitz = 89
        elif c == 'bowl':
            unitx = 165
            unity = 80
            unitz = 165
        elif c == 'camera':
            unitx = 88
            unity = 128
            unitz = 156
        elif c == 'can':
            unitx = 68
            unity = 146
            unitz = 72
        elif c == 'laptop':
            unitx = 346
            unity = 200
            unitz = 335
        elif c == 'mug':
            unitx = 146
            unity = 83
            unitz = 114
        else:
            unitx = 0
            unity = 0
            unitz = 0
            print('This category is not recorded in my little brain.')
            # scale residual
        return np.array([unitx, unity, unitz])










