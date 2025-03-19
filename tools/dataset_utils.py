import copy

import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.stats import laplace

def get_2d_coord_np(width, height, low=0, high=1, fmt="CHW", norm=True):
    """
    Args:
        width:
        height:
    Returns:
        xy: (2, height, width)
    """
    # coords values are in [low, high]  [0,1] or [-1,1]
    x = np.linspace(0, width-1, width, dtype=np.float32)
    y = np.linspace(0, height-1, height, dtype=np.float32)
    if norm:
        # TODO (0,1)
        x = (x - (width-1)/2) / ((width-1)/2)
        y = (y - (height-1)/2) / ((height-1)/2)
    xy = np.asarray(np.meshgrid(x, y))
    if fmt == "HWC":
        xy = xy.transpose(1, 2, 0)
    elif fmt == "CHW":
        pass
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return xy

def aug_bbox_DZI(FLAGS, bbox_xyxy, im_H, im_W, ori=False):
    """Used for DZI, the augmented box is a square (maybe enlarged)
    Args:
        bbox_xyxy (np.ndarray):
    Returns:
        center, scale
    """
    x1, y1, x2, y2 = bbox_xyxy.copy()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1
    if ori:
        bbox_center = np.array([cx, cy])  # (w/2, h/2)
        scale = max(y2 - y1, x2 - x1)
        scale = min(scale, max(im_H, im_W)) * 1.0
        return bbox_center, scale

    if FLAGS.DZI_TYPE.lower() == "uniform":
        scale_ratio = 1 + FLAGS.DZI_SCALE_RATIO * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
        shift_ratio = FLAGS.DZI_SHIFT_RATIO * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
        bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
        scale = max(y2 - y1, x2 - x1) * scale_ratio * FLAGS.DZI_PAD_SCALE
    elif FLAGS.DZI_TYPE.lower() == "uniform_sr":
        scale_ratio = 1 - 0.25 * np.random.random_sample()  # [1-0.25,1]
        shift_ratio = FLAGS.DZI_SHIFT_RATIO * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
        bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
        scale = max(y2 - y1, x2 - x1) * scale_ratio * FLAGS.DZI_PAD_SCALE
    elif FLAGS.DZI_TYPE.lower() == "roi10d":
        # shift (x1,y1), (x2,y2) by 15% in each direction
        _a = -0.15
        _b = 0.15
        x1 += bw * (np.random.rand() * (_b - _a) + _a)
        x2 += bw * (np.random.rand() * (_b - _a) + _a)
        y1 += bh * (np.random.rand() * (_b - _a) + _a)
        y2 += bh * (np.random.rand() * (_b - _a) + _a)
        x1 = min(max(x1, 0), im_W)
        x2 = min(max(x1, 0), im_W)
        y1 = min(max(y1, 0), im_H)
        y2 = min(max(y2, 0), im_H)
        bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
        scale = max(y2 - y1, x2 - x1) * FLAGS.DZI_PAD_SCALE
    elif FLAGS.DZI_TYPE.lower() == "truncnorm":
        raise NotImplementedError("DZI truncnorm not implemented yet.")
    elif FLAGS.DZI_TYPE.lower() == "none":
        bbox_center = np.array([cx, cy])  # (w/2, h/2)
        scale = max(y2 - y1, x2 - x1)
    else:
        raise NotImplementedError
    scale = min(scale, max(im_H, im_W)) * 1.0
    return bbox_center, scale

def aug_bbox_eval(bbox_xyxy, im_H, im_W):
    """Used for DZI, the augmented box is a square (maybe enlarged)
    Args:
        bbox_xyxy (np.ndarray):
    Returns:
        center, scale
    """
    x1, y1, x2, y2 = bbox_xyxy.copy()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1
    bbox_center = np.array([cx, cy])  # (w/2, h/2)
    scale = max(y2 - y1, x2 - x1)
    scale = min(scale, max(im_H, im_W)) * 1.0
    return bbox_center, scale

def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def shrink_mask(input_mask, shrink_pixel=1):
    '''
    this func shrink a mask by a fixed pixel num
    :param input_mask: [h,w]
    :return:
    '''
    w, h = input_mask.shape
    for _ in range(shrink_pixel):
        left_pixel = np.zeros_like(input_mask)
        for x in range(w):
            for y in range(h):
                if x in [0,w-1] or y in [0,h-1]:
                    continue
                elif input_mask[x-1,y] == 1 and  input_mask[x+1,y] == 1 and input_mask[x,y-1] == 1 and input_mask[x,y+1] == 1:
                    left_pixel[x,y] = 1
        input_mask = copy.deepcopy(left_pixel)
    return input_mask

def xyz_to_region(xyz_crop, fps_points):
    bh, bw = xyz_crop.shape[:2]
    mask_crop = ((xyz_crop[:, :, 0] != 0) | (xyz_crop[:, :, 1] != 0) | (xyz_crop[:, :, 2] != 0)).astype("uint8")
    dists = cdist(xyz_crop.reshape(bh * bw, 3), fps_points)  # (hw, f)
    region_ids = np.argmin(dists, axis=1).reshape(bh, bw) + 1  # NOTE: 1 to num_fps
    # (bh, bw)
    return mask_crop * region_ids  # 0 means bg

def xyz_to_delta(xyz_crop, fps_points, i):
    # {0:x,1:y,2:z}
    bh, bw = xyz_crop.shape[:2]
    mask_crop = ((xyz_crop[:, :, 0] != 0) | (xyz_crop[:, :, 1] != 0) | (xyz_crop[:, :, 2] != 0)).astype("uint8")
    xyz_flat = xyz_crop.reshape(bh * bw, 3)
    xyz_flat_one_dim = np.zeros_like(xyz_flat)
    xyz_flat_one_dim[:, i] = xyz_flat[:, i]
    region_width = fps_points[1,i] - fps_points[0,i]
    top, down = np.max(fps_points), np.min(fps_points)
    xyz_flat_one_dim = xyz_flat_one_dim.clip(down, top)
    dists = cdist(xyz_flat_one_dim, fps_points)  # (hw, f)
    region_ids = np.argmin(dists, axis=1)# .reshape(bh, bw) + 1  # NOTE: 1 to num_fps
    res_delta = xyz_flat_one_dim[:, i] - fps_points[region_ids,i]
    res_delta = res_delta.reshape(bh, bw) / region_width
    # (bh, bw)
    return mask_crop * res_delta  # 0 means bg

def xyz_to_laplace(roi_region, roi_region_delta, each_region_num, b=1):
    region_1d = roi_region.reshape(-1)
    delta = roi_region_delta.reshape(-1)
    sub_region_len = 16*b/(2*each_region_num-1)
    delta_true = delta*sub_region_len
    left = np.ones_like(delta)*-8*b
    left_true = left - delta_true
    probability = laplace_generate(left_true, 2*each_region_num-1, sub_region_len, b=b)
    l_idx = each_region_num - region_1d
    # region each_region_num-> start from 0
    # region 0 ->start from each_region_num; removed by mask later
    indices = l_idx + np.arange(each_region_num)[:, np.newaxis]
    indices = np.minimum(indices, 2*each_region_num-2)
    probability_res = probability[indices, np.arange(64*64)]
    probability_res = probability_res / np.sum(probability_res, axis=0)
    probability_res = probability_res.reshape(each_region_num, 64, 64)
    return probability_res


def adapt_region_by_size(fps_points, real_size, mean_size):
    real_size_norm = real_size/np.linalg.norm(real_size)
    mean_size_norm = mean_size/np.linalg.norm(mean_size)
    x_scale = real_size_norm[0]/mean_size_norm[0]
    y_scale = real_size_norm[1]/mean_size_norm[1]
    z_scale = real_size_norm[2]/mean_size_norm[2]
    fps_points_scaled = np.column_stack((fps_points[:,0]*x_scale, fps_points[:,1]*y_scale, fps_points[:,2]*z_scale))

    return fps_points_scaled

def save_pc2ply(pc, name, colors):
    from plyfile import PlyData, PlyElement
    vertex = np.core.records.fromarrays([pc[:, 0], pc[:, 1], pc[:, 2],colors[:, 0], colors[:, 1], colors[:, 2]], names='x, y, z, red, green, blue', formats='f4, f4, f4, u1, u1, u1')
    element = PlyElement.describe(vertex, 'vertex')

    ply_data = PlyData([element])

    ply_data.write(name)
    return

def get_up2down_region(sp, region_num):
    max_y = np.max(sp[:, 1])
    min_y = np.min(sp[:, 1])
    region_center = np.zeros((region_num, 3))
    d = (max_y - min_y)/(3*region_num - 5)
    bottom = min_y - d
    for i in range(region_num):
        region_center[i,1] = max(bottom, -1)
        bottom += 3*d
    return region_center

def get_left2right_region(sp, region_num):
    max_z = np.max(sp[:, 2])
    min_z = np.min(sp[:, 2])
    region_center = np.zeros((region_num, 3))
    d = (max_z - min_z)/(3*region_num - 5)
    bottom = min_z - d
    for i in range(region_num):
        region_center[i,2] = max(bottom, -1)
        bottom += 3*d
    return region_center

def get_back2front_region(sp, region_num):
    max_x = np.max(sp[:, 0])
    min_x = np.min(sp[:, 0])
    region_center = np.zeros((region_num, 3))
    d = (max_x - min_x)/(3*region_num - 5)
    bottom = min_x - d
    for i in range(region_num):
        region_center[i,0] = max(bottom, -1)
        bottom += 3*d
    return region_center

def get_aver_div3d(sp, region_num, i):
    # {0:up2down,1:left2right,2:back2font}
    if i == 0:
        fps_points = get_up2down_region(sp, region_num)
    elif i == 1:
        fps_points = get_left2right_region(sp, region_num)
    elif i == 2:
        fps_points = get_back2front_region(sp, region_num)
    else:
        raise NotImplementedError
    return fps_points

def get_real_div3d(real_model, region_num, i):
    max_v = np.max(real_model[:, i])
    min_v = np.min(real_model[:, i])
    region_center = np.zeros((region_num, 3))
    width = (max_v - min_v) / region_num
    bottom = min_v
    for j in range(region_num):
        region_center[j, i] = max(bottom, -1)
        bottom += width
    return region_center

def laplace_generate(left_true, bins_num, sub_len, b):
    res = np.zeros((bins_num, 64*64))
    l = left_true
    for i in range(bins_num):
        l_p = laplace_F(l, b)
        r_p = laplace_F(l+sub_len, b)
        res[i] = r_p-l_p
        l = l+sub_len
    return res

def laplace_F(array, b=1, u=0):
    res = np.zeros_like(array)
    res[array<u] = (0.5*np.exp((array-u)/b))[array<u]
    res[array>=u] = (1-0.5*np.exp((u-array)/b))[array>=u]
    return res



