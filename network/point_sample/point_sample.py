import torch
import torch.nn.functional as F
import numpy as np
import absl.flags as flags

FLAGS = flags.FLAGS


def Depth2PC(obj_mask, Depth, camK, coor2d, seman_feature, nocs_coord=None, batch_operation=True):
    '''
    :param Depth: bs x 1 x h x w
    :param camK:
    :param coor2d:
    :param rgb_feature
    :return:
    '''
    # handle obj_mask
    if obj_mask.shape[1] == 2:   # predicted mask
        obj_mask = F.softmax(obj_mask, dim=1)
        _, obj_mask = torch.max(obj_mask, dim=1)
    bs, H, W = Depth.shape[0], Depth.shape[2], Depth.shape[3]
    seman_dim = seman_feature.shape[-1]

    x_label = coor2d[:, 0, :, :]
    y_label = coor2d[:, 1, :, :]


    if batch_operation:
        depth = Depth.squeeze(dim=1)
        obj_mask = obj_mask.squeeze(dim=1)
        dp_mask = (depth > 0.0)
        fuse_mask = obj_mask.float() * dp_mask.float()
        fuse_mask = fuse_mask.bool()
        fx = camK[:, 0, 0]
        fy = camK[:, 1, 1]
        ux = camK[:, 0, 2]
        uy = camK[:, 1, 2]
        x_now = (x_label - ux[:,None,None]) * depth / fx[:,None,None]
        y_now = (y_label - uy[:,None,None]) * depth / fy[:,None,None]
        p_n_now = torch.stack([x_now, y_now, depth], dim=-1)
        p_seman = seman_feature
        p_part = seman_feature[..., 0]
        if nocs_coord is not None:
            p_nocs = nocs_coord.permute(0, 2, 3, 1)
            p_cat_feat = torch.cat([p_n_now, p_seman, p_nocs], dim=-1)
        else:
            p_cat_feat = torch.cat([p_n_now, p_seman], dim=-1)

        p_select_list = []
        part_idx_list = []

        for i in range(bs):
            p_n_f_now = p_cat_feat[i, fuse_mask[i]]
            p_part_label_now = p_part[i, fuse_mask[i]]

            if FLAGS.sample_method == 'random':
                # basic sampling
                samplenum = FLAGS.point_number
                l_all = p_n_f_now.shape[0]
                if l_all <= 1.0:
                    print('NO POINT IN POINT CLOUD')
                    return None, None
                if l_all >= samplenum:
                    replace_rnd = False
                else:
                    replace_rnd = True

                choose = np.random.choice(l_all, samplenum, replace=replace_rnd)  # can selected more than one times
                p_select = p_n_f_now[choose, :]
                part_idx = torch.zeros([FLAGS.part_num, FLAGS.point_per_part], device=depth.device, dtype=torch.int)
                p_select_part = p_select[:, 3]
                samplenum = FLAGS.point_per_part
                for i_part in range(FLAGS.part_num):
                    curr_part_idx = torch.argwhere(p_select_part == i_part).squeeze()
                    l_all = curr_part_idx.shape[0]
                    if l_all <= 1.0:
                        continue
                    if l_all >= samplenum:
                        replace_rnd = False
                    else:
                        replace_rnd = True
                    choose_part = np.random.choice(l_all, samplenum, replace=replace_rnd)
                    part_idx[i_part] = curr_part_idx[choose_part]
            elif FLAGS.sample_method == 'part':
                choose_list = []
                samplenum = FLAGS.point_per_part
                part_idx = torch.zeros([FLAGS.part_num, FLAGS.point_per_part], device=depth.device, dtype=torch.int)
                for i_part in range(FLAGS.part_num):
                    p_part_now = p_n_f_now[p_part_label_now == i_part]
                    l_all = p_part_now.shape[0]
                    if l_all < 1:
                        # print('part not visible', i_part)
                        choose_list.append(torch.zeros([samplenum, p_part_now.shape[1]], device=depth.device))
                        continue
                    else:
                        if l_all >= samplenum:
                            replace_rnd = False
                        else:
                            replace_rnd = True
                        choose_now = np.random.choice(l_all, samplenum, replace=replace_rnd)  # can selected more than one times
                        choose_list.append(p_part_now[choose_now, :])
                    part_idx[i_part, :] = torch.arange(i_part * samplenum, (i_part+1) * samplenum).to(depth.device)
                p_select = torch.cat(choose_list, dim=0)
            else:
                raise NotImplementedError(FLAGS.sample_method)

            p_select_list.append(p_select)
            part_idx_list.append(part_idx)

        p_select = torch.stack(p_select_list, dim=0)
        part_idx = torch.stack(part_idx_list, dim=0)

        '''
        p_select_x = (p_select[:, 0] - ux) * p_select[:, 2] / fx
        p_select[:, 0] = p_select_x
        p_select_y = (p_select[:, 1] - uy) * p_select[:, 2] / fy
        p_select[:, 1] = p_select_y
        '''

        PC = p_select[:, :, :3]
        if nocs_coord is not None:
            PC_nocs = p_select[:, :, -3:]
            PC_seman = p_select[:, :, 3:-3]
        else:
            PC_seman = p_select[:, :, 3:]
            PC_nocs = None
    else:
        raise NotImplementedError
    return PC / 1000.0, PC_seman, PC_nocs, part_idx