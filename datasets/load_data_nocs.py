import copy
import os
import cv2
import math
import random
from absl import app
import mmcv
import numpy as np
import _pickle as cPickle
from config.config import *
from datasets.data_augmentation import defor_2D, get_rotation
FLAGS = flags.FLAGS

import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from tools.eval_utils import load_depth, get_bbox_square, get_real_hw, get_bbox_ori
from tools.dataset_utils import *
from nnutils.tps_utils import TPS, torch_tps_transform
from losses.pose_loss import symmetry_rotation_matrix_y, get_sym_max_nocs_rot

class NocsDataset(data.Dataset):
    def __init__(self, source=None, mode='train', data_dir=None, per_obj=''):
        '''
        :param source: 'CAMERA' or 'Real' or 'CAMERA+Real'
        :param mode: 'train' or 'test'
        :param data_dir: 'path to dataset'
        '''
        self.source = source
        self.mode = mode
        self.data_dir = data_dir

        assert source in ['CAMERA', 'Real', 'CAMERA+Real']
        assert mode in ['train', 'test']
        img_list_path = ['CAMERA/train_list.txt', 'Real/train_list.txt',
                         'CAMERA/val_list.txt', 'Real/test_list.txt']
        model_file_path = ['obj_models/camera_train.pkl', 'obj_models/real_train.pkl',
                           'obj_models/camera_val.pkl', 'obj_models/real_test.pkl']

        if mode == 'train':
            del img_list_path[2:]
            del model_file_path[2:]
        else:
            del img_list_path[:2]
            del model_file_path[:2]
        if source == 'CAMERA':
            del img_list_path[-1]
            del model_file_path[-1]
        elif source == 'Real':
            del img_list_path[0]
            del model_file_path[0]
        else:
            # only use Real to test when source is CAMERA+Real
            if mode == 'test':
                del img_list_path[0]
                del model_file_path[0]

        img_list = []
        subset_len = []
        #  aggregate all availabel datasets
        for path in img_list_path:
            img_list += [os.path.join(path.split('/')[0], line.rstrip('\n'))
                         for line in open(os.path.join(data_dir, path))]
            subset_len.append(len(img_list))
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1] - subset_len[0]]
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.cat_name2id = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}
        self.id2cat_name = {'1': 'bottle', '2': 'bowl', '3': 'camera', '4': 'can', '5': 'laptop', '6': 'mug'}
        self.id2cat_name_CAMERA = {'1': '02876657',
                                   '2': '02880940',
                                   '3': '02942699',
                                   '4': '02946921',
                                   '5': '03642806',
                                   '6': '03797390'}
        self.id2cat_name_real = self.id2cat_name
        if source == 'CAMERA':
            self.id2cat_name = self.id2cat_name_CAMERA
        self.per_obj = per_obj
        self.per_obj_id = None
        # only train one object
        if self.per_obj in self.cat_names:
            self.per_obj_id = self.cat_name2id[self.per_obj]
            img_list_cache_dir = os.path.join(self.data_dir, 'img_list')
            if not os.path.exists(img_list_cache_dir):
                os.makedirs(img_list_cache_dir)
            img_list_cache_filename = os.path.join(img_list_cache_dir, f'{per_obj}_{source}_{mode}_img_list.txt')
            if os.path.exists(img_list_cache_filename):
                print(f'read image list cache from {img_list_cache_filename}')
                img_list_obj = [line.rstrip('\n') for line in open(os.path.join(data_dir, img_list_cache_filename))]
            else:
                # needs to reorganize img_list
                s_obj_id = self.cat_name2id[self.per_obj]
                img_list_obj = []
                from tqdm import tqdm
                for i in tqdm(range(len(img_list))):
                    gt_path = os.path.join(self.data_dir, img_list[i] + '_label.pkl')
                    try:
                        with open(gt_path, 'rb') as f:
                            gts = cPickle.load(f)
                        id_list = gts['class_ids']
                        if s_obj_id in id_list:
                            img_list_obj.append(img_list[i])
                    except:
                        print(f'WARNING {gt_path} is empty')
                        continue
                with open(img_list_cache_filename, 'w') as f:
                    for img_path in img_list_obj:
                        f.write("%s\n" % img_path)
                print(f'save image list cache to {img_list_cache_filename}')
                # iter over  all img_list, cal sublen

            if len(subset_len) == 2:
                camera_len  = 0
                real_len = 0
                for i in range(len(img_list_obj)):
                    if 'CAMERA' in img_list_obj[i].split('/'):
                        camera_len += 1
                    else:
                        real_len += 1
                self.subset_len = [camera_len, real_len]
            #  if use only one dataset
            #  directly load all data
            img_list = img_list_obj

        self.img_list = img_list
        self.length = len(self.img_list)

        models = {}
        for path in model_file_path:
            with open(os.path.join(data_dir, path), 'rb') as f:
                models.update(cPickle.load(f))
        self.models = models

        # move the center to the body of the mug
        # meta info for re-label mug category
        with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
            self.mug_meta = cPickle.load(f)

        self.camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]],
                                          dtype=np.float32)  # [fx, fy, cx, cy]
        self.real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float32)

        self.color_aug_prob = FLAGS.color_aug_prob
        self.color_aug_type = FLAGS.color_aug_type
        self.color_aug_code = FLAGS.color_aug_code
        if mode == "train" and self.color_aug_prob > 0:
            self.color_augmentor = self._get_color_augmentor(aug_type=self.color_aug_type, aug_code=self.color_aug_code)
        else:
            self.color_augmentor = None

        invalid_list_cache_path = os.path.join(self.data_dir, f'invalid_list_cache_dict_{source}_.txt')
        self.invalid_list_cache_path = invalid_list_cache_path
        self.invalid_dict = {}
        if os.path.exists(invalid_list_cache_path):
            with open(invalid_list_cache_path, 'r') as file:
                while True:
                    line = file.readline()
                    if not line:
                        break
                    path, inst_id = line.split()
                    inst_id = int(inst_id)
                    if path in self.invalid_dict.keys():
                        self.invalid_dict[path].append(inst_id)
                    else:
                        self.invalid_dict[path] = [inst_id]

        self.mug_sym = mmcv.load(os.path.join(self.data_dir, 'Real/mug_handle.pkl'))

        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

        print('{} images found.'.format(self.length))
        print('{} models loaded.'.format(len(self.models)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #   load ground truth
        img_path = os.path.join(self.data_dir, self.img_list[index])
        sub_path=self.img_list[index]
        try:
            with open(img_path + '_label.pkl', 'rb') as f:
                gts = cPickle.load(f)
        except:
            return self.__getitem__((index + 1) % self.__len__())
        if 'CAMERA' in img_path.split('/'):
            out_camK = self.camera_intrinsics
            img_type = 'syn'
        else:
            out_camK = self.real_intrinsics
            img_type = 'real'

        # select one foreground object,
        # if specified, then select the object

        if self.per_obj in self.cat_names:
            idx_per_obj = []
            for item in enumerate(gts['class_ids']):
                if item[1] == self.per_obj_id:
                    idx_per_obj.append(item[0])
            idx = random.choice(idx_per_obj)
        else:
            idx = random.randint(0, len(gts['instance_ids']) - 1)
            if FLAGS.ban_mug:
                while gts['class_ids'][idx] == 6:
                    idx = random.randint(0, len(gts['instance_ids']) - 1)
        if img_path in self.invalid_dict.keys():
            if gts['instance_ids'][idx] in self.invalid_dict[img_path]:
                return self.__getitem__((index + 1) % self.__len__())
        if gts['class_ids'][idx] == 6 and img_type == 'real' and self.mode == 'train':
            assert not FLAGS.ban_mug
            handle_tmp_path = img_path.split('/')
            scene_label = handle_tmp_path[-2] + '_res'
            img_id = int(handle_tmp_path[-1])
            mug_handle = self.mug_sym[scene_label][img_id]
        else:
            mug_handle = 1

        image = cv2.imread(img_path + '_color.png')
        if image is not None:
            image = image[:, :, :3]
        else:
            return self.__getitem__((index + 1) % self.__len__())

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        im_H, im_W = image.shape[0], image.shape[1]
        if self.mode == "train" and self.color_aug_prob > 0 and self.color_augmentor is not None:
            if np.random.rand() < self.color_aug_prob:
                if FLAGS.COLOR_AUG_SYN_ONLY:
                    if img_type == 'syn':
                        image = self._color_aug(image, self.color_aug_type)
                else:
                    image = self._color_aug(image, self.color_aug_type)
        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)


        mask_path = img_path + '_mask.png'
        mask = cv2.imread(mask_path)
        if mask is not None:
            mask = mask[:, :, 2]
        else:
            return self.__getitem__((index + 1) % self.__len__())

        cat_id = gts['class_ids'][idx] - 1  # convert to 0-indexed
        model_name = gts['model_list'][idx]

        nocs_coord = cv2.imread(img_path + '_coord.png')
        if nocs_coord is not None:
            nocs_coord = nocs_coord[:, :, :3]
        else:
            return self.__getitem__((index + 1) % self.__len__())
        nocs_coord = nocs_coord[:, :, (2, 1, 0)] # BGR2RGB
        nocs_coord = np.array(nocs_coord, dtype=np.float32) / 255
        nocs_coord[:, :, 2] = 1 - nocs_coord[:, :, 2]
        # [0, 1] -> [-0.5, 0.5]
        nocs_coord = nocs_coord - 0.5

        # adjust nocs coords for mug category
        if cat_id == 5:
            T0 = self.mug_meta[model_name][0]
            s0 = self.mug_meta[model_name][1]
            nocs_coord = s0 * (nocs_coord + T0)

        inst_id = gts['instance_ids'][idx]
        rmin, rmax, cmin, cmax = get_bbox_ori(gts['bboxes'][idx])

        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        bbox_center, img_scale = aug_bbox_DZI(FLAGS, bbox_xyxy, im_H, im_W)
        info2d = np.array([bbox_center[0]/im_W, bbox_center[1]/im_H, img_scale/im_W, img_scale/im_H])
        bw,bh = get_real_hw(gts['bboxes'][idx])

        # load roi info
        roi_img = crop_resize_by_warp_affine(
            image, bbox_center, img_scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        )

        roi_img_origin = roi_img.copy().transpose(2, 0, 1) / 255.0
        roi_img = (roi_img / 255.0 - self.img_mean) / self.img_std
        roi_img = roi_img.transpose(2, 0, 1)

        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, img_scale, FLAGS.out_res, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)

        mask_target = mask.copy().astype(np.float32)
        mask_target[mask != inst_id] = 0.0
        mask_target[mask == inst_id] = 1.0
        nocs_coord[mask_target==0] = 0

        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, img_scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_mask = np.expand_dims(roi_mask, axis=0)
        roi_mask_output = crop_resize_by_warp_affine(
            mask_target, bbox_center, img_scale, FLAGS.out_res, interpolation=cv2.INTER_NEAREST
        )
        roi_mask_output = np.expand_dims(roi_mask_output, axis=0)

        roi_nocs_coord = crop_resize_by_warp_affine(
            nocs_coord, bbox_center, img_scale, FLAGS.out_res, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)

        # ivfc
        ivfc_coor_path = './data/IVFC/' + sub_path + f'_coor_{idx}.png' # dataset path
        assert os.path.exists(ivfc_coor_path), ivfc_coor_path
        ivfc_coord = cv2.imread(ivfc_coor_path)
        ivfc_coord = ivfc_coord[:, :, (2, 1, 0)]
        ivfc_coord = np.array(ivfc_coord, dtype=np.float32) / 255
        mask_target_ivfc = np.ones_like(mask_target)
        mask_target_ivfc[ivfc_coord[:, :, 0] == 0] = 0
        ivfc_coord[:, :, 2] = 1 - ivfc_coord[:, :, 2]
        # [0, 1] -> [-0.5, 0.5]
        ivfc_coord = ivfc_coord - 0.5

        # adjust ivfc coords for mug category
        if cat_id == 5:
            T0 = self.mug_meta[model_name][0]
            s0 = self.mug_meta[model_name][1]
            ivfc_coord = s0 * (ivfc_coord + T0)
        ivfc_coord[mask_target_ivfc == 0] = 0
        roi_ivfc_coord = crop_resize_by_warp_affine(
            ivfc_coord, bbox_center, img_scale, FLAGS.out_res, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)
        roi_ivfc_mask_output = crop_resize_by_warp_affine(
            mask_target_ivfc, bbox_center, img_scale, FLAGS.out_res, interpolation=cv2.INTER_NEAREST
        )
        roi_ivfc_mask_output = np.expand_dims(roi_ivfc_mask_output, axis=0)

        model = self.models[model_name].astype(np.float32)  # 1024 points
        nocs_scale = gts['scales'][idx]  # model bounding box diagonal length !!
        rotation = gts['rotations'][idx]
        translation = gts['translations'][idx]

        real_size, mean_size = self.get_fs_net_scale(self.id2cat_name[str(cat_id + 1)], model, nocs_scale)
        if FLAGS.dataset == 'CAMERA':
            real_size, mean_size = self.get_fs_net_scale(self.id2cat_name_CAMERA[str(cat_id + 1)], model, nocs_scale)
        real_size = real_size / 1000.0
        mean_size = mean_size / 1000.0

        sym_info = self.get_sym_info(cat_id + 1, mug_handle=mug_handle)
        roi_mask_def = defor_2D(roi_mask, rand_r=FLAGS.roi_mask_r, rand_pro=FLAGS.roi_mask_pro)

        bb_aug, rt_aug_t, rt_aug_R = self.generate_aug_parameters()

        full_img = cv2.resize(image, (256, 256)) if FLAGS.resize_full else image
        full_img = (full_img / 255.0 - self.img_mean) / self.img_std
        full_img = full_img.transpose(2, 0, 1)
        one_hot = convert_to_one_hot(cat_id, 6)


        data_dict = {}
        data_dict['full_img'] = torch.as_tensor(full_img.astype(np.float32)).contiguous()
        data_dict['roi_img_origin'] = torch.as_tensor(roi_img_origin.astype(np.float32)).contiguous() ###
        data_dict['roi_img'] = torch.as_tensor(roi_img.astype(np.float32)).contiguous()
        data_dict['cam_K'] = torch.as_tensor(out_camK.astype(np.float32)).contiguous() ###
        data_dict['roi_mask'] = torch.as_tensor(roi_mask.astype(np.float32)).contiguous() ###
        data_dict['cat_id_0_base'] = torch.as_tensor(cat_id, dtype=torch.int).contiguous()
        data_dict['rotation'] = torch.as_tensor(rotation, dtype=torch.float32).contiguous() ###
        data_dict['translation'] = torch.as_tensor(translation, dtype=torch.float32).contiguous() ###
        data_dict['real_size'] = torch.as_tensor(real_size, dtype=torch.float32).contiguous() ###
        data_dict['sym_info'] = torch.as_tensor(sym_info.astype(np.float32)).contiguous() ###
        data_dict['roi_coord_2d'] = torch.as_tensor(roi_coord_2d, dtype=torch.float32).contiguous() ###
        data_dict['mean_size'] = torch.as_tensor(mean_size, dtype=torch.float32).contiguous() ###
        data_dict['aug_bb'] = torch.as_tensor(bb_aug, dtype=torch.float32).contiguous()
        data_dict['aug_rt_t'] = torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous()
        data_dict['aug_rt_R'] = torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous()
        data_dict['roi_mask_deform'] = torch.as_tensor(roi_mask_def, dtype=torch.float32).unsqueeze(0).contiguous()
        data_dict['roi_mask_output'] = torch.as_tensor(roi_mask_output, dtype=torch.float32).contiguous() ###
        data_dict['roi_ivfc_mask_output'] = torch.as_tensor(roi_ivfc_mask_output, dtype=torch.float32).contiguous()
        data_dict['model_point'] = torch.as_tensor(model, dtype=torch.float32).contiguous() ###
        data_dict['nocs_scale'] = torch.as_tensor(nocs_scale, dtype=torch.float32).contiguous() ###
        data_dict['nocs_coord'] = torch.as_tensor(roi_nocs_coord, dtype=torch.float32).contiguous() ###
        data_dict['ivfc_coord'] = torch.as_tensor(roi_ivfc_coord, dtype=torch.float32).contiguous()
        data_dict['img_path'] = img_path
        data_dict['inst_id'] = inst_id
        data_dict['roi_wh'] = torch.as_tensor([bw, bh], dtype=torch.float32).contiguous() ###
        data_dict["img_scale"] = torch.as_tensor(img_scale, dtype=torch.float32).contiguous()
        data_dict["resize_ratio"] = torch.as_tensor(FLAGS.out_res / img_scale, dtype=torch.float32).contiguous() ###
        data_dict["bbox_center"] = torch.as_tensor(bbox_center, dtype=torch.float32).contiguous() ###
        data_dict['one_hot'] = torch.as_tensor(one_hot, dtype=torch.float32).contiguous()
        data_dict['info_2d'] = torch.as_tensor(info2d, dtype=torch.float32).contiguous()
        return data_dict


    def generate_aug_parameters(self, s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2), ax=50, ay=50, az=50, a=15):
        # for bb aug
        ex, ey, ez = np.random.rand(3)
        ex = ex * (s_x[1] - s_x[0]) + s_x[0]
        ey = ey * (s_y[1] - s_y[0]) + s_y[0]
        ez = ez * (s_z[1] - s_z[0]) + s_z[0]
        # for R, t aug
        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
        dx = np.random.rand() * 2 * ax - ax
        dy = np.random.rand() * 2 * ay - ay
        dz = np.random.rand() * 2 * az - az
        return np.array([ex, ey, ez], dtype=np.float32), np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm


    def get_fs_net_scale(self, c, model, nocs_scale):
        # model pc x 3
        lx = 2 * max(max(model[:, 0]), -min(model[:, 0]))
        ly = max(model[:, 1]) - min(model[:, 1])
        lz = max(model[:, 2]) - min(model[:, 2])

        # real scale
        lx_t = lx * nocs_scale * 1000
        ly_t = ly * nocs_scale * 1000
        lz_t = lz * nocs_scale * 1000

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
        elif c == '02876657':
            unitx = 324 / 4
            unity = 874 / 4
            unitz = 321 / 4
            # (81, 218.5, 80.25)
        elif c == '02880940':
            unitx = 675 / 4
            unity = 271 / 4
            unitz = 675 / 4
            # (167.75, 67.75, 167.75)
        elif c == '02942699':
            unitx = 464 / 4
            unity = 487 / 4
            unitz = 702 / 4
        elif c == '02946921':
            unitx = 450 / 4
            unity = 753 / 4
            unitz = 460 / 4
        elif c == '03642806':
            unitx = 581 / 4
            unity = 445 / 4
            unitz = 672 / 4
        elif c == '03797390':
            unitx = 670 / 4
            unity = 540 / 4
            unitz = 497 / 4
        else:
            unitx = 0
            unity = 0
            unitz = 0
            print('This category is not recorded in my little brain.')
            raise NotImplementedError
        return np.array([lx_t, ly_t, lz_t]), np.array([unitx, unity, unitz])

    def get_sym_info(self, c, mug_handle=1):
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
        elif c == 6 and mug_handle == 1:
            sym = np.array([0, 1, 0, 0], dtype=int)  # for mug, we currently mark it as no symmetry
        elif c == 6 and mug_handle == 0:
            sym = np.array([1, 0, 0, 0], dtype=int)
        else:
            raise NotImplementedError(f'sym info {c}')
        return sym

    def get_z_scale_aver(self, c):
        if c == 1:
            z_mul_scale = 474
        elif c == 2:
            z_mul_scale = 398
        elif c == 3:
            z_mul_scale = 443
        elif c == 4:
            z_mul_scale = 487
        elif c == 5:
            z_mul_scale = 471
        elif c == 6:
            z_mul_scale = 371
        else:
            raise NotImplementedError(f'z_scale info {c}')
        all_z_mul_scale = 442
        return z_mul_scale, all_z_mul_scale
    def _get_color_augmentor(self, aug_type="aae", aug_code=None):
        # fmt: off
        if aug_type.lower() == "aae":
            import imgaug.augmenters as iaa  # noqa
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation, pillike,
                                           LinearContrast)  # noqa
            aug_code = """Sequential([
                # Sometimes(0.5, PerspectiveTransform(0.05)),
                # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
                # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
                Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),
                Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),
                Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
                Sometimes(0.3, Invert(0.2, per_channel=True)),
                Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
                Sometimes(0.5, Multiply((0.6, 1.4))),
                Sometimes(0.5, LinearContrast((0.5, 2.2), per_channel=0.3))
                ], random_order = False)"""
            color_augmentor = eval(aug_code)
        elif aug_type.lower() == 'cosy+aae':
            import imgaug.augmenters as iaa
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation, pillike,
                                           LinearContrast)  # noqa
            aug_code = """Sequential([
            Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),
            Sometimes(0.4, GaussianBlur((0., 3.))),
            Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),
            Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),
            Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),
            Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),
            Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
            Sometimes(0.3, Invert(0.2, per_channel=True)),
            Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
            Sometimes(0.5, Multiply((0.6, 1.4))),
            Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),
            Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),
            # Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),
            ], random_order=True)"""
            color_augmentor = eval(aug_code)
        elif aug_type.lower() == "new":  # assume imgaug
            import imgaug.augmenters as iaa
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation, pillike,
                                           LinearContrast)  # noqa
            color_augmentor = Sequential([
                        Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 2.))),
                        Sometimes(0.5, pillike.EnhanceContrast(factor=(0.5, 1.5))),
                        Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.5, 1.5))),
                        Sometimes(0.3, pillike.EnhanceColor(factor=(0., 3.))),
                        ], random_order=True)
        elif aug_type.lower() == "new1":  # assume imgaug
            import imgaug.augmenters as iaa
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation, pillike, MultiplyHueAndSaturation,
                                           LinearContrast)  # noqa
            color_augmentor = Sequential([
                        Sometimes(0.5, pillike.EnhanceSharpness(factor=(0., 2.))),
                        Sometimes(0.8, pillike.EnhanceContrast(factor=(0.5, 1.5))),
                        Sometimes(0.8, pillike.EnhanceBrightness(factor=(0.5, 1.5))),
                        Sometimes(0.8, MultiplyHueAndSaturation(mul_hue=(0.8, 1.2), mul_saturation=(0.5, 1.5), per_channel=True)),
                        Sometimes(0.5, pillike.EnhanceColor(factor=(0., 3.))),
                        ], random_order=True)
        elif aug_type.lower() == "code":  # assume imgaug
            import imgaug.augmenters as iaa
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation, pillike,
                                           LinearContrast)  # noqa
            color_augmentor = eval(aug_code)
        else:
            color_augmentor = None
        # fmt: on
        return color_augmentor

    def _color_aug(self, image, aug_type="code"):
        # assume image in [0, 255] uint8
        if aug_type.lower() in ["aae", "code", "cosy+aae", "new", "new1"]:
            # imgaug need uint8
            return self.color_augmentor.augment_image(image)
        elif aug_type.lower() in ["code_albu"]:
            augmented = self.color_augmentor(image=image)
            return augmented["image"]
        else:
            raise ValueError("aug_type: {} is not supported.".format(aug_type))

    def add_invalid_path(self, invalid_path_dict):
        for path in invalid_path_dict.keys():
            if path in self.invalid_dict.keys():
                self.invalid_dict[path].append(invalid_path_dict[path])
            else:
                self.invalid_dict[path] = [invalid_path_dict[path]]
        with open(self.invalid_list_cache_path, 'a') as file:
            for path in invalid_path_dict:
                file.write(path + ' ' + str(invalid_path_dict[path]) + '\n')

def check(argv):
    train_dataset = NocsDataset(source='CAMERA+Real', mode='train',
                                data_dir='./data/NOCS', per_obj='all')
    for test_data in train_dataset:
        if 'scene_3' in test_data['img_path']:
            mask = test_data['roi_mask_output']
            mask = np.array(mask.to('cpu')[0]*255, dtype=np.uint8)
            cv2.imwrite('scene_3.png', mask)
            print(test_data['img_path'])


def convert_to_one_hot(category_id, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[category_id] = 1
    return one_hot

if __name__=="__main__":
    app.run(check)