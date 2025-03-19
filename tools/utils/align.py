import time
import numpy as np
import cv2
import torch

def ransacPnP_LM(p2d, p3d, K):
    dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    pts_2d = np.ascontiguousarray(p2d.astype(np.float64))
    pts_3d = np.ascontiguousarray(p3d.astype(np.float64))
    K = K.astype(np.float64)

    try:
        _, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, K, dist_coeffs, reprojectionError=5,
                                                    iterationsCount=10000, flags=cv2.SOLVEPNP_EPNP)

        rvec, tvec = cv2.solvePnPRefineLM(pts_3d, pts_2d, K, dist_coeffs, rvec, tvec)

        rotation = cv2.Rodrigues(rvec)[0]

        pose = np.concatenate([rotation, tvec], axis=-1)
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)

        inliers = [] if inliers is None else inliers

        return rotation, tvec, pose_homo, inliers
    except cv2.error:
        print("CV ERROR")
        return np.eye(3), np.array([0,0,1]).reshape(3,1), [], []

def ransac_pnp(roi_coord_2d, roi_coord_3d, mask_out, K, pred_scale):
    pred_rot_m, pred_trans = [],[]
    width, height = 640, 480
    for i in range(roi_coord_2d.shape[0]):
        scale_i = pred_scale[i]
        mask_out_i = mask_out[i]
        p_2d = roi_coord_2d[i]
        p_2d[0] = p_2d[0] * ((width-1)/2) + ((width-1)/2)
        p_2d[1] = p_2d[1] * ((height-1)/2) + ((height-1)/2)
        p_2d = p_2d.transpose(1, 2, 0)
        p_3d = roi_coord_3d[i].transpose(1, 2, 0)*scale_i
        sel_mask = (mask_out_i==1)
        sel_mask = sel_mask[0]
        model_point = p_3d[sel_mask].reshape(-1, 3)
        image_point = p_2d[sel_mask].reshape(-1, 2)
        K_i = K[i]
        rot, trans, _, _ = ransacPnP_LM(image_point, model_point, K_i)
        pred_rot_m.append(rot)
        pred_trans.append(trans[:,0]/scale_i)
    pred_rot_m = np.array(pred_rot_m)
    pred_trans = np.array(pred_trans)
    return torch.as_tensor(pred_rot_m).contiguous(), torch.as_tensor(pred_trans).contiguous()









