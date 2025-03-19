import torch
from tools.align_utils import estimateSimilarityTransform, estimateSimilarityUmeyama
import numpy as np
import matplotlib
from PIL import Image
import open3d as o3d

def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True)  # ((1)xhxwx4)
    colors = colors[:, :, :3]  # Discard alpha component
    return Image.fromarray(colors)

def pose_from_umeyama(xyz_coor, coor_2d, camK, Depth, obj_mask):
    PC, nocs_coord, obj_mask = get_PC_nocs(coor_2d, camK, Depth, obj_mask, xyz_coor)
    # PC = output_dict['PC'].detach().cpu().numpy()
    # nocs_coord = output_dict['NOCS_coord'].detach().cpu().numpy()
    bs = PC.shape[0]
    rots, trans, scales = [],[],[]
    for i in range(bs):
        PC_i = PC[i]
        nocs_coord_i = nocs_coord[i]
        obj_mask_i = obj_mask[i]
        PC_choose = PC_i[obj_mask_i,:]
        nocs_coord_choose = nocs_coord_i[obj_mask_i,:] # + 0.5
        scale, rotation, translation, pred_sRT = estimateSimilarityTransform(nocs_coord_choose, PC_choose)
        if scale is None:
            scale = 1
            rotation = np.eye(3)
            translation = np.zeros(3)
        rots.append(rotation)
        trans.append(translation)
        scales.append(scale)
    return torch.as_tensor(np.array(scales).astype(np.float32)).contiguous(), torch.as_tensor(np.array(rots).astype(np.float32)).contiguous(), torch.as_tensor(np.array(trans).astype(np.float32)).contiguous()
    # p_s = output_dict['Pred_s'].detach()
    # pred_s = p_s.detach().cpu().numpy()


def get_PC_nocs(coor_2d, camK, Depth, obj_mask, xyz_coor):
    bs = coor_2d.shape[0]
    depth = Depth.squeeze(dim=1).cpu()
    obj_mask = obj_mask.squeeze(dim=1).bool()
    obj_mask = obj_mask.view(bs, 64*64).cpu().numpy()
    xyz_coor = np.array(xyz_coor.cpu())
    xyz_coor = xyz_coor.transpose(0, 2, 3, 1)
    xyz_coor = xyz_coor.reshape(bs, 64*64, 3)
    x_label = coor_2d[:, 0, :, :].cpu()
    y_label = coor_2d[:, 1, :, :].cpu()

    fx = camK[:, 0, 0]
    fy = camK[:, 1, 1]
    ux = camK[:, 0, 2]
    uy = camK[:, 1, 2]
    x_now = (x_label - ux[:, None, None]) * depth/fx[:,None,None]
    y_now = (y_label - uy[:, None, None]) * depth/fy[:,None,None]
    p_n_now = torch.stack([x_now, y_now, depth], dim=-1).view(bs, 64*64, 3).numpy()
    return p_n_now, xyz_coor ,obj_mask





