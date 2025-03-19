# @Time    : 06/05/2021
# @Author  : Wei Chen
# @Project : Pycharm
import numpy as np
import torch
import cv2
import math
import struct
import os
import random
import _pickle as cPickle
from tqdm import tqdm
from scipy.optimize import leastsq
from scipy.optimize import least_squares
import matplotlib.pyplot as plt





import torch
import cv2

ratio_dict={}
grids={}
cat_property={1:{'mean_shape':[1,1,1],'ratio':[1,1,1],'xz_ratio':1,'base':6},
              2:{'mean_shape':[1,1,1],'ratio':[1,1,1],'xz_ratio':1,'base':0},
              3:{'mean_shape':[1,1,1],'ratio':[1,1,1],'xz_ratio':1,'base':2}}

def get_ratio(model_dict,cat_names,model_name2cat_name):
    for cat_name in cat_names:
        ratio_dict[cat_name]=[1,1,1]
    for model_name,model in model_dict.items():
        cat_name=model_name2cat_name[model_name]
        if cat_name=='mug':
            lx = 2 * np.amax(np.abs(model[:, 0]))
            ly = 2 * np.amax(np.abs(model[:, 1]))
            lz = 2 * np.amax(np.abs(model[:, 2]))
        else:
            lx = max(model[:, 0]) - min(model[:, 0])
            ly = max(model[:, 1]) - min(model[:, 1])
            lz = max(model[:, 2]) - min(model[:, 2])
        scale=np.array([lx, ly, lz])
        if cat_name in ['bottle','can','bowl']:
            if np.abs(lx-lz)>0.05:
                continue
        max_scale=scale.max()
        ratio=max_scale/scale
        flag=False
        for i,ratio_i in enumerate(ratio):
            if ratio_i>ratio_dict[cat_name][i]:
                if ratio_i>4:
                    continue
                else:
                    ratio_dict[cat_name][i]=ratio_i
    return

def cat_get_property(models,obj_id2cat_id,cat_ids):
    cat_scales={}
    for cat_id in cat_ids:
        cat_scales[cat_id]=[]
    for model_id,model in models.items():
        cat_id=obj_id2cat_id[model_id]
        lx = max(model[:, 0]) - min(model[:, 0])
        ly = max(model[:, 1]) - min(model[:, 1])
        lz = max(model[:, 2]) - min(model[:, 2])
        scale=np.array([lx, ly, lz])
        cat_scales[cat_id].append(scale)
    for cat_id,scales in cat_scales.items():
        num_inst=len(scales)
        sum_scale=np.array([0.0,0.0,0.0])
        max_ratio=np.array([0.0,0.0,0.0])
        sum_xz_r=0
        for scale in scales:
            sum_scale+=scale
            max_scale=scale.max()
            ratio=max_scale/scale
            for i,ratio_i in enumerate(ratio):
                if ratio_i>max_ratio[i]:
                    max_ratio[i]=ratio_i
            sum_xz_r+=scale[0]/scale[2]
        mean_scale=sum_scale/num_inst
        mean_xz=sum_xz_r/num_inst
        cat_property[cat_id]['mean_shape']=mean_scale.tolist()
        cat_property[cat_id]['ratio']=max_ratio.tolist()
        cat_property[cat_id]['xz_ratio']=mean_xz


    return



def make_scatter(n_points_uniform,boxsize):
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize * (points_uniform - 0.5)
    points = points_uniform
    return points


def make_grid(fake_grid_num,scale):
    max_scale=scale.max()
    voxel_dim=round(math.pow(fake_grid_num*max_scale**3 / scale[0]/scale[1]/scale[2], 1/3))
    box_size = 1
    voxel_origin = [-0.5*box_size]*3
    voxel_size=box_size/(voxel_dim-1)
    overall_index = torch.arange(0, voxel_dim ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(voxel_dim ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % voxel_dim
    samples[:, 1] = (overall_index.long() // voxel_dim) % voxel_dim
    samples[:, 0] = ((overall_index.long() // voxel_dim) // voxel_dim) % voxel_dim

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    index=torch.logical_and(np.abs(samples[:,0])<=scale[0]/2/max_scale,
                            np.abs(samples[:,2])<=scale[2]/2/max_scale)
    index=torch.logical_and(index,np.abs(samples[:,1])<=scale[1]/2/max_scale)
    samples=samples[index]
    return samples





# 该函数将深度值进行映射，生成带颜色的深度图
def convert_depth(depth,max_depth=2000):
    histogram=[0 for _ in range(max_depth)]
    h=depth.shape[0]
    w=depth.shape[1]
    total_num=0
    for y in range(h):
        for x in range(w):
            d=depth[y,x]
            if d>0:
                histogram[d]+=1
                total_num+=1
    for i in range(1,max_depth):
        histogram[i]+=histogram[i-1]
    rgb=np.zeros((h,w,3))
    for y in range(h):
        for x in range(w):
            d=depth[y,x]
            if d>0:
                d=depth[y,x]
                r=(256 * (1.0-histogram[d]/total_num))
                rgb[y,x]=np.array([r,r,r])
    return rgb



def show_open3d(points1,points2,points3=None,color_2=None):
    import open3d as o3d
    pcd_1=o3d.geometry.PointCloud()
    pcd_1.points=o3d.utility.Vector3dVector(points1+np.array([0,0,0]))
    color_1=np.zeros_like(points1)
    color_1[:,2]=1
    pcd_1.colors=o3d.utility.Vector3dVector(color_1)
    pcd_2=o3d.geometry.PointCloud()
    pcd_2.points=o3d.utility.Vector3dVector(points2+np.array([0,0,0]))

    if color_2 is None:
        color_2=np.zeros_like(points2)
        color_2[:,0]=1
    pcd_2.colors=o3d.utility.Vector3dVector(color_2)

    frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    if points3 is not None:
        pcd_3=o3d.geometry.PointCloud()
        pcd_3.points=o3d.utility.Vector3dVector(points3+np.array([0,0,0]))
        color_3=np.zeros_like(points3)
        color_3[:,1]=1
        pcd_3.colors=o3d.utility.Vector3dVector(color_3)

        o3d.visualization.draw_geometries([pcd_1,pcd_2,pcd_3],width=800,height=600)
    else:
        o3d.visualization.draw_geometries([pcd_1,pcd_2],width=800,height=600)


def show_open3d(points1,points2,points3=None,color_2=None):
    import open3d as o3d
    pcd_1=o3d.geometry.PointCloud()
    pcd_1.points=o3d.utility.Vector3dVector(points1+np.array([0,0,0]))
    color_1=np.zeros_like(points1)
    color_1[:,2]=1
    pcd_1.colors=o3d.utility.Vector3dVector(color_1)
    pcd_2=o3d.geometry.PointCloud()
    pcd_2.points=o3d.utility.Vector3dVector(points2+np.array([0,0,0]))

    if color_2 is None:
        color_2=np.zeros_like(points2)
        color_2[:,0]=1
    pcd_2.colors=o3d.utility.Vector3dVector(color_2)

    frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    if points3 is not None:
        pcd_3=o3d.geometry.PointCloud()
        pcd_3.points=o3d.utility.Vector3dVector(points3+np.array([0,0,0]))
        color_3=np.zeros_like(points3)
        color_3[:,1]=1
        pcd_3.colors=o3d.utility.Vector3dVector(color_3)

        o3d.visualization.draw_geometries([pcd_1,pcd_2,pcd_3],width=800,height=600)
    else:
        o3d.visualization.draw_geometries([pcd_1,pcd_2],width=800,height=600)

def get_neighbor(query_point,support_points,num_neighber):
    from EQNet.eqnet.ops.knn.knn_utils import knn_query
    from EQNet.eqnet.ops.grouping.grouping_utils import grouping_operation
    query_cnt=query_point.new_zeros(1).int()
    query_cnt[0]=query_point.shape[0]
    support_cnt=support_points.new_zeros(1).int()
    support_cnt[0]=support_points.shape[0]
    index_pair = knn_query(
        num_neighber,
        support_points, support_cnt,
        query_point, query_cnt).int()
    neighbor_pos=grouping_operation(
        support_points, support_cnt, index_pair, query_cnt).permute(0,2,1)
    return neighbor_pos


def show_3d(points1,points2):
    x1=points1[:,0]
    y1=points1[:,1]
    z1=points1[:,2]

    x2=points2[:,0]
    y2=points2[:,1]
    z2=points2[:,2]

    #开始绘图
    fig=plt.figure(dpi=120)
    ax=fig.add_subplot(111,projection='3d')
    #标题
    plt.title('point cloud')
    #利用xyz的值，生成每个点的相应坐标（x,y,z）
    ax.scatter(x1,y1,z1,c='b',marker='*',s=30,linewidth=0,alpha=1,cmap='spectral')

    ax.scatter(x2,y2,z2,c='r',marker='.',s=30,linewidth=0,alpha=1,cmap='spectral')

    # ax.axis('scaled')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_box_aspect((np.ptp(x1), np.ptp(y1), np.ptp(z1)))
    #显示
    plt.show()




def gettrans(kps,h):
    # print(kps.shape) ##N*3
    # print(h.shape)##N,100,3
    # tess
    hss=[]
    # print(h)
    # print(kps.shape) ##3*N
    # kps
    # print(kps.shape)
    # tess
    kps=kps.reshape(-1,3)
    for i in range(h.shape[1]):
        # print(i)
        # print(h[:,i,:].shape #N*3
        # tss

        P = kps.T - kps.T.mean(1).reshape((3, 1))
        #
        Q= h[:,i,:].T - h[:,i,:].T.mean(1).reshape((3,1))
        # print(P.shape,Q.shape)
        # print(kps,h[:,i,:])
        # tess

        # print(P.T,Q.T)
        R=kabsch(P.T,Q.T) ##N*3, N*3

        T=h[:,i,:]-np.dot(R,kps.T).T

        # print(np.mean(T,0))
        # tess
        # print(T.shape)
        hh = np.zeros((3, 4), dtype=np.float32)
        hh[0:3,0:3]=R
        hh[0:3,3]=np.mean(T,0)
        # print(R)
        hss.append(hh)
        # print(hh)
        # if i==3:
        #     tess
    # print(hss)
    return hss


def kabsch(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    #print(P.shape,Q.shape)
    # print(np.mean(P,0))
    # P= P-np.mean(P,0)
    # Q =Q - np.mean(Q, 0)
    # print(P)
    # tests
    C = np.dot(P.T, Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    U, S, V = np.linalg.svd(C)
    #S=np.diag(S)
    #print(C)
    # print(S)
    #print(np.dot(U,np.dot(S,V)))
    d = (np.linalg.det(V.T) * np.linalg.det(U.T)) <0.0

    # d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    # E = np.diag(np.array([1, 1, 1]))
    # if d:
    #     S[-1] = -S[-1]
    #     V[:, -1] = -V[:, -1]
    E = np.diag(np.array([1, 1, (np.linalg.det(V.T) * np.linalg.det(U.T))]))


    # print(E)

    # Create Rotation matrix U
    #print(V)
    #print(U)
    R = np.dot(V.T ,np.dot(E,U.T))

    return R

class Rodrigues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rvec):
        R, jac = cv2.Rodrigues(rvec.detach().cpu().numpy())
        jac = torch.from_numpy(jac).to(rvec.device)
        ctx.save_for_backward(jac)
        return torch.from_numpy(R).to(rvec.device)

    @staticmethod
    def backward(ctx, grad_output):
        jac, = ctx.saved_tensors
        return jac @ grad_output.to(jac.device).reshape(-1)

def draw_axis(img,K,R,T):


    length=100
    origin=K @ (np.array([0,0,0])[:,None]+T)


    g_green_y_point=K @ (R @ np.array([0,1,0])[:,None]*length+T)
    g_red_x_point=K @ (R @ np.array([1,0,0])[:,None]*length+T)


    origin=(origin/origin[2]).squeeze()
    g_green_y_point=(g_green_y_point/g_green_y_point[2]).squeeze()
    g_red_x_point=(g_red_x_point/g_red_x_point[2]).squeeze()
    cv2.line(img, (np.int(origin[0]),np.int(origin[1])), (np.int(g_green_y_point[0]), np.int(g_green_y_point[1])), (255,255,255),2)
    cv2.line(img, (np.int(origin[0]),np.int(origin[1])), (np.int(g_red_x_point[0]), np.int(g_red_x_point[1])), (255,255,255), 2)




    return img

def fill_missing(
        dpt, cam_scale, scale_2_80m, fill_type='multiscale',
        extrapolate=False, show_process=False, blur_type='bilateral'
):
    dpt = dpt / cam_scale * scale_2_80m
    projected_depth = dpt.copy()
    if fill_type == 'fast':
        final_dpt = fill_in_fast(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            # max_depth=2.0
        )
    elif fill_type == 'multiscale':
        final_dpt, process_dict = fill_in_multiscale(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            show_process=show_process,
            max_depth=3.0
        )
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))
    dpt = final_dpt / scale_2_80m * cam_scale
    return dpt

CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)


FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)


def fill_in_multiscale(depth_map, max_depth=8.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.01) & (depths_in <= 1.0)
    valid_pixels_med = (depths_in > 1.0) & (depths_in <= 2.0)
    valid_pixels_far = (depths_in > 2.0)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.01)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.01)
    valid_pixels_med = (dilated_med > 0.01)
    valid_pixels_far = (dilated_far > 0.01)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.01)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.01)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.01)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.01, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
            pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.01) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.01) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.01) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.01)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    return depths_out, process_dict


def normalize_vector( v, dim =1, return_mag =False):
    v_mag = torch.sqrt(v.pow(2).sum(dim=dim, keepdim=True))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.expand_as(v)
    v = v/v_mag
    return v
def cross_product(u, v):
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    out = torch.cat((i.unsqueeze(1), j.unsqueeze(1), k.unsqueeze(1)),1)#batch*3
    return out

def Ortho6d2Mat(x_raw, y_raw):
    y = normalize_vector(y_raw)
    z = cross_product(x_raw, y)
    z = normalize_vector(z)#batch*3
    x = cross_product(y,z)#batch*3

    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    z = z.unsqueeze(2)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix


from scipy.spatial.transform import Rotation as R
import numpy as np
import math


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def siyuanshu2rotation_matrix(Rq):
    # Rq = [-0.35, 1.23e-06, 4.18e-08, 0.39]
    Rm = R.from_quat(Rq)
    rotation_matrix = Rm.as_matrix()
    return rotation_matrix

def Rm_t2T(Rm, t):
    return np.array([list(Rm[0]) + [t[0]],
                     list(Rm[1]) + [t[1]],
                     list(Rm[2]) + [t[2]],
                     [0, 0, 0, 1]])

def inv_T(T):
    R=np.array([[T[0][0],T[0][1],T[0][2]],
                [T[1][0],T[1][1],T[1][2]],
                [T[2][0],T[2][1],T[2][2]]])
    t=np.array([[T[0][3]],[T[1][3]],[T[2][3]]])
    RT=R.T
    t_=-RT@t
    T_ = np.array([[RT[0][0], RT[0][1], RT[0][2], t_[0][0]],
                   [RT[1][0], RT[1][1],RT[1][2], t_[1][0]],
                   [RT[2][0], RT[2][1], RT[2][2], t_[2][0]],
                   [0,0,0,1]])
    return T_




def get_3D_corner_aligned(pc):
    x_max=max(pc[:,0])
    x_min=min(pc[:,0])
    y_max=max(pc[:,1])
    y_min=min(pc[:,1])
    z_max=max(pc[:,2])
    z_min=min(pc[:,2])
    l_x=x_max-x_min
    l_y=y_max-y_min
    l_z=z_max-z_min

    or1=np.array([x_max,y_min,z_min])
    or2=np.array([x_max,y_max,z_min])
    or3=np.array([x_max,y_max,z_max])
    or4=np.array([x_max,y_min,z_max])

    or5=np.array([x_min,y_min,z_min])
    or6=np.array([x_min,y_max,z_min])
    or7=np.array([x_min,y_max,z_max])
    or8=np.array([x_min,y_min,z_max])

    OR=np.array([or1,or2,or3,or4,or5,or6,or7,or8])
    scale=np.array([l_x,l_y,l_z])
    center=np.array([(x_max+x_min)/2,(y_max+y_min)/2,(z_max+z_min)/2])

    return OR, scale, center

import numpy




def cdist(K: numpy.ndarray, B: numpy.ndarray) -> numpy.ndarray:
    """Calculate Euclidean distance between K[i, :] and B[j, :].
    Arguments
    ---------
        K : numpy.array
        B : numpy.array
    """
    K = numpy.atleast_2d(K)
    B = numpy.atleast_2d(B)
    assert K.ndim == 2
    assert B.ndim == 2

    K = numpy.expand_dims(K, 1)
    B = numpy.expand_dims(B, 0)
    D = K - B
    return numpy.linalg.norm(D, axis=2)


def pairwise_radial_basis(K: numpy.ndarray, B: numpy.ndarray) -> numpy.ndarray:
    """Compute the TPS radial basis function phi(r) between every row-pair of K
    and B where r is the Euclidean distance.
    Arguments
    ---------
        K : numpy.array
            n by d vector containing n d-dimensional points.
        B : numpy.array
            m by d vector containing m d-dimensional points.
    Return
    ------
        P : numpy.array
            n by m matrix where.
            P(i, j) = phi( norm( K(i,:) - B(j,:) ) ),
            where phi(r) = r^2*log(r), if r >= 1
                           r*log(r^r), if r <  1
    """
    # r_mat(i, j) is the Euclidean distance between K(i, :) and B(j, :).
    r_mat = cdist(K, B)

    pwise_cond_ind1 = r_mat >= 1
    pwise_cond_ind2 = r_mat < 1
    r_mat_p1 = r_mat[pwise_cond_ind1]
    r_mat_p2 = r_mat[pwise_cond_ind2]

    # P correcponds to the matrix K from [1].
    P = numpy.empty(r_mat.shape)
    P[pwise_cond_ind1] = (r_mat_p1**2) * numpy.log(r_mat_p1)
    P[pwise_cond_ind2] = r_mat_p2 * numpy.log(numpy.power(r_mat_p2, r_mat_p2))

    return P


def find_coefficients(control_points: numpy.ndarray,
                      target_points: numpy.ndarray,
                      lambda_: float = 0.,
                      solver: str = 'exact') -> numpy.ndarray:
    """Given a set of control points and their corresponding points, compute the
    coefficients of the TPS interpolant deforming surface.
    Arguments
    ---------
        control_points : numpy.array
            p by d vector of control points
        target_points : numpy.array
            p by d vector of corresponding target points on the deformed
            surface
        lambda_ : float
            regularization parameter
        solver : str
            the solver to get the coefficients. default is 'exact' for the exact
            solution. Or use 'lstsq' for the least square solution.
    Return
    ------
        coef : numpy.ndarray
            the coefficients
    .. seealso::
        http://cseweb.ucsd.edu/~sjb/pami_tps.pdf
    """
    # ensure data type and shape
    control_points = numpy.atleast_2d(control_points)
    target_points = numpy.atleast_2d(target_points)
    if control_points.shape != target_points.shape:
        raise ValueError(
            'Shape of and control points {cp} and target points {tp} are not the same.'.
                format(cp=control_points.shape, tp=target_points.shape))

    p, d = control_points.shape

    # The matrix
    K = pairwise_radial_basis(control_points, control_points)
    P = numpy.hstack([numpy.ones((p, 1)), control_points])

    # Relax the exact interpolation requirement by means of regularization.
    K = K + lambda_ * numpy.identity(p)

    # Target points
    M = numpy.vstack([
        numpy.hstack([K, P]),
        numpy.hstack([P.T, numpy.zeros((d + 1, d + 1))])
    ])
    Y = numpy.vstack([target_points, numpy.zeros((d + 1, d))])

    # solve for M*X = Y.
    # At least d+1 control points should not be in a subspace; e.g. for d=2, at
    # least 3 points are not on a straight line. Otherwise M will be singular.
    solver = solver.lower()
    if solver == 'exact':
        X = numpy.linalg.solve(M, Y)
    elif solver == 'lstsq':
        X, _, _, _ = numpy.linalg.lstsq(M, Y, None)
    else:
        raise ValueError('Unknown solver: ' + solver)

    return X


def transform(source_points: numpy.ndarray, control_points: numpy.ndarray,
              coefficient: numpy.ndarray) -> numpy.ndarray:
    """Transform the source points form the original surface to the destination
    (deformed) surface.
    Arguments
    ---------
        source_points : numpy.array
            n by d array of source points to be transformed
        control_points : numpy.array
            the control points used in the function `find_coefficients`
        coefficient : numpy.array
            the computed coefficients
    Return
    ------
        deformed_points : numpy.array
            n by d array of the transformed point on the target surface
    """
    source_points = numpy.atleast_2d(source_points)
    control_points = numpy.atleast_2d(control_points)
    if source_points.shape[-1] != control_points.shape[-1]:
        raise ValueError(
            'Dimension of source points ({sd}D) and control points ({cd}D) are not the same.'.
                format(sd=source_points.shape[-1], cd=control_points.shape[-1]))

    n = source_points.shape[0]

    A = pairwise_radial_basis(source_points, control_points)
    K = numpy.hstack([A, numpy.ones((n, 1)), source_points])

    deformed_points = numpy.dot(K, coefficient)
    return deformed_points




class TPS:
    """The thin plate spline deformation warpping.
    """

    def __init__(self,
                 control_points: numpy.ndarray,
                 target_points: numpy.ndarray,
                 lambda_: float = 0.4,
                 solver: str = 'lstsq'):
        """Create a instance that preserve the TPS coefficients.
        Arguments
        ---------
            control_points : numpy.array
                p by d vector of control points
            target_points : numpy.array
                p by d vector of corresponding target points on the deformed
                surface
            lambda_ : float
                regularization parameter
            solver : str
                the solver to get the coefficients. default is 'exact' for the
                exact solution. Or use 'lstsq' for the least square solution.
        """
        self.control_points = control_points
        self.coefficient = find_coefficients(
            control_points, target_points, lambda_, solver)

    def __call__(self, source_points):
        """Transform the source points form the original surface to the
        destination (deformed) surface.
        Arguments
        ---------
            source_points : numpy.array
                n by d array of source points to be transformed
        """
        return transform(source_points, self.control_points,
                         self.coefficient)


def torch_tps_transform(source,coefficient,control_points):
    source = torch.as_tensor(source, dtype=torch.float32).contiguous().unsqueeze(0)
    coefficient = torch.as_tensor(coefficient, dtype=torch.float32).contiguous().unsqueeze(0)
    control_points = torch.as_tensor(control_points, dtype=torch.float32).contiguous().unsqueeze(0)

    distance=torch.norm(source.unsqueeze(2)-control_points.unsqueeze(1),dim=-1)
    pwise_cond_ind1 = distance >= 1
    pwise_cond_ind2 = distance < 1
    r_mat_p1 = distance[pwise_cond_ind1]
    r_mat_p2 = distance[pwise_cond_ind2]

    # P correcponds to the matrix K from [1].
    P = torch.zeros_like(distance)
    P[pwise_cond_ind1] = (r_mat_p1**2) * torch.log(r_mat_p1)
    P[pwise_cond_ind2] = r_mat_p2 * torch.log(torch.pow(r_mat_p2, r_mat_p2))
    source_num=source.shape[1]
    pad=torch.ones_like(source)[:,:,:1]
    K = torch.cat([P, pad, source],dim=-1)
    deformed_points = K @ coefficient
    return np.array(deformed_points.squeeze())

def torch_tps_regular_transform(source,coefficient,control_points):
    source = torch.as_tensor(source,dtype=torch.float32).contiguous().unsqueeze(0)
    coefficient = torch.as_tensor(coefficient,dtype=torch.float32).contiguous().unsqueeze(0)
    control_points = torch.as_tensor(control_points,dtype=torch.float32).contiguous().unsqueeze(0)

    # for batch
    distance=torch.norm(source.unsqueeze(2)-control_points.unsqueeze(1),dim=-1)
    pwise_cond_ind1 = distance >= 1
    pwise_cond_ind2 = distance < 1
    r_mat_p1 = distance[pwise_cond_ind1]
    r_mat_p2 = distance[pwise_cond_ind2]

    # P correcponds to the matrix K from [1].
    P = torch.zeros_like(distance)
    P[pwise_cond_ind1] = (r_mat_p1**2) * torch.log(r_mat_p1)
    P[pwise_cond_ind2] = r_mat_p2 * torch.log(torch.pow(r_mat_p2, r_mat_p2))
    delta=P @ coefficient
    return source+delta






class TPS_reguler:
    """The thin plate spline deformation warpping.
    """

    def __init__(self,
                 control_points: numpy.ndarray,
                 target_points: numpy.ndarray,
                 lambda_: float = 0.4,
                 solver: str = 'lstsq'):
        """Create a instance that preserve the TPS coefficients.
        Arguments
        ---------
            control_points : numpy.array
                p by d vector of control points
            target_points : numpy.array
                p by d vector of corresponding target points on the deformed
                surface
            lambda_ : float
                regularization parameter
            solver : str
                the solver to get the coefficients. default is 'exact' for the
                exact solution. Or use 'lstsq' for the least square solution.
        """
        self.control_points = control_points
        self.coefficient = self.find_coefficients(
            control_points, target_points, lambda_, solver)
    def find_coefficients(self,control_points, target_points, lambda_, solver):
        control_points = numpy.atleast_2d(control_points)
        target_points = numpy.atleast_2d(target_points)
        if control_points.shape != target_points.shape:
            raise ValueError(
                'Shape of and control points {cp} and target points {tp} are not the same.'.
                    format(cp=control_points.shape, tp=target_points.shape))

        p, d = control_points.shape

        K = pairwise_radial_basis(control_points, control_points)
        P = numpy.ones((p, 1))

        # Target points
        M = numpy.vstack([K,P.T,control_points.T])
        target_Y = numpy.vstack([target_points, numpy.zeros((d + 1, d))])
        source_Y=numpy.vstack([control_points, numpy.zeros((d + 1, d))])

        Y=target_Y-source_Y

        # solve for M*X = Y.
        # At least d+1 control points should not be in a subspace; e.g. for d=2, at
        # least 3 points are not on a straight line. Otherwise M will be singular.
        solver = solver.lower()
        if solver == 'exact':
            X = numpy.linalg.solve(M, Y)
        elif solver == 'lstsq':
            X, _, _, _ = numpy.linalg.lstsq(M, Y, None)
        else:
            raise ValueError('Unknown solver: ' + solver)

        return X
    def transform(self,source_points, control_points,coefficient):
        source_points = numpy.atleast_2d(source_points)
        control_points = numpy.atleast_2d(control_points)
        if source_points.shape[-1] != control_points.shape[-1]:
            raise ValueError(
                'Dimension of source points ({sd}D) and control points ({cd}D) are not the same.'.
                    format(sd=source_points.shape[-1], cd=control_points.shape[-1]))

        n = source_points.shape[0]

        A = pairwise_radial_basis(source_points, control_points)

        delta= numpy.dot(A, coefficient)

        return source_points+delta
    def __call__(self, source_points):
        """Transform the source points form the original surface to the
        destination (deformed) surface.
        Arguments
        ---------
            source_points : numpy.array
                n by d array of source points to be transformed
        """
        return self.transform(source_points, self.control_points,
                              self.coefficient)

















def get_key_fea(points,control_points):
    distance=torch.norm(points.unsqueeze(1)-control_points.unsqueeze(0),dim=-1)
    pwise_cond_ind1 = distance >= 1
    pwise_cond_ind2 = distance < 1
    r_mat_p1 = distance[pwise_cond_ind1]
    r_mat_p2 = distance[pwise_cond_ind2]

    # P correcponds to the matrix K from [1].
    P = torch.zeros_like(distance)
    P[pwise_cond_ind1] = (r_mat_p1**2) * torch.log(r_mat_p1)
    P[pwise_cond_ind2] = r_mat_p2 * torch.log(torch.pow(r_mat_p2, r_mat_p2))
    return P

def show_error(surface,error=None,outpath=None):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import mpl_toolkits.mplot3d.axes3d as p3
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    plt.axis('off')
    ax.set_box_aspect([1,1,1])
    limit=0.5
    ax.set(xlim3d=(-limit, limit), xlabel='X')
    ax.set(ylim3d=(-limit, limit), ylabel='Y')
    ax.set(zlim3d=(-limit, limit), zlabel='Z')
    azim=180
    dist=90
    elev=50

    if azim is not None:
        ax.azim = azim
    # if dist is not None:
    #     ax.dist = dist
    if elev is not None:
        ax.elev = elev

    import matplotlib as mpl
    cm=plt.cm.get_cmap('jet')
    norm = mpl.colors.Normalize(vmin=0, vmax=0.3)


    surface_x=-surface[:,0]
    surface_y=surface[:,1]
    surface_z=surface[:,2]
    if error is None:
        ax.scatter(surface_x, surface_y, surface_z, s=10,c=(0.5,0.5,0.5), marker='o',zdir='y')
    else:
        ax.scatter(surface_x, surface_y, surface_z, s=10,c=cm(error), marker='o',zdir='y')

    # fig, ax = plt.subplots(figsize=(1, 10))
    # fig.subplots_adjust(right=0.5)
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm),
    #              ax=ax, orientation='vertical')
    if outpath is not None:
        import time
        plt.savefig(outpath)
    else:
        plt.show()
    return



def show_error_2(surface,sample,error=None,outpath=None):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import mpl_toolkits.mplot3d.axes3d as p3
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    plt.axis('off')
    ax.set_box_aspect([1,1,1])
    limit=0.5
    ax.set(xlim3d=(-limit, limit), xlabel='X')
    ax.set(ylim3d=(-limit, limit), ylabel='Y')
    ax.set(zlim3d=(-limit, limit), zlabel='Z')
    azim=-137
    dist=90
    elev=32

    if azim is not None:
        ax.azim = azim
    # if dist is not None:
    #     ax.dist = dist
    if elev is not None:
        ax.elev = elev

    import matplotlib as mpl
    cm=plt.cm.get_cmap('jet')
    norm = mpl.colors.Normalize(vmin=0, vmax=0.3)
    surface_x=-surface[:,0]
    surface_y=surface[:,1]
    surface_z=surface[:,2]
    ax.scatter(surface_x, surface_y, surface_z, s=10,c=np.array([0.5,0.5,0.5]), marker='o',zdir='y')

    sample_x=-sample[:,0]
    sample_y=sample[:,1]
    sample_z=sample[:,2]
    if error is None:
        ax.scatter(sample_x, sample_y, sample_z, s=5,c=(0.5,0.5,0.5), marker='o',zdir='y')
    else:
        ax.scatter(sample_x, sample_y, sample_z, s=4,c=cm(error), marker='o',zdir='y')



    # fig, ax = plt.subplots(figsize=(1, 10))
    # fig.subplots_adjust(right=0.5)
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm),
    #              ax=ax, orientation='vertical')
    if outpath is not None:
        import time
        plt.savefig(outpath)
    else:
        plt.show()
    return






def show_nocs(surface,query=None,nocs=None,tsne=None,outpath=None):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import mpl_toolkits.mplot3d.axes3d as p3
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    plt.axis('off')
    ax.set_box_aspect([1,1,1])
    limit=0.5
    ax.set(xlim3d=(-limit, limit), xlabel='X')
    ax.set(ylim3d=(-limit, limit), ylabel='Y')
    ax.set(zlim3d=(-limit, limit), zlabel='Z')
    azim=180
    dist=90
    elev=50

    if azim is not None:
        ax.azim = azim
    # if dist is not None:
    #     ax.dist = dist
    if elev is not None:
        ax.elev = elev


    nocs_color=nocs
    surface_x=-surface[:,0]
    surface_y=surface[:,1]
    surface_z=surface[:,2]
    ax.scatter(surface_x, surface_y, surface_z, s=20,c=nocs_color, marker='o',zdir='y')

    if outpath is not None:
        import time
        plt.savefig(outpath)
    else:
        plt.show()
    return tsne

def show_sample(surface,sample,outpath=None):
    import mpl_toolkits.mplot3d.axes3d as p3
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    plt.axis('off')
    ax.set_box_aspect([1,1,1])
    limit=0.5
    ax.set(xlim3d=(-limit, limit), xlabel='X')
    ax.set(ylim3d=(-limit, limit), ylabel='Y')
    ax.set(zlim3d=(-limit, limit), zlabel='Z')
    azim=-140
    dist=90
    elev=32

    if azim is not None:
        ax.azim = azim
    # if dist is not None:
    #     ax.dist = dist
    if elev is not None:
        ax.elev = elev

    surface_color=np.zeros_like(surface)
    surface_color[:]=np.array([0,0,1])
    surface_size=np.zeros_like(surface)[:,0]
    surface_size[:]=10

    sample_color=np.zeros_like(sample)
    sample_color[:]=np.array([0.5,0.5,0.5])
    sample_size=np.zeros_like(sample)[:,0]
    sample_size[:]=10

    combine=np.concatenate([surface,sample],axis=0)
    combine_color=np.concatenate([surface_color,sample_color],axis=0)
    combine_size=np.concatenate([surface_size,sample_size],axis=0)

    combine_x=-combine[:,0]
    combine_y=combine[:,1]
    combine_z=combine[:,2]
    ax.scatter(combine_x, combine_y, combine_z, s=combine_size,c=combine_color, marker='o',zdir='y')


    if outpath is not None:
        import time
        plt.savefig(outpath)
    else:
        plt.show()
    return

def show_modelnet(surface,sample,color=[0,1,0],outpath=None):
    import mpl_toolkits.mplot3d.axes3d as p3
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    plt.axis('off')
    ax.set_box_aspect([1,1,1])
    limit=0.5
    ax.set(xlim3d=(-limit, limit), xlabel='X')
    ax.set(ylim3d=(-limit, limit), ylabel='Y')
    ax.set(zlim3d=(-limit, limit), zlabel='Z')
    azim=126
    dist=6
    elev=20

    if azim is not None:
        ax.azim = azim
    # if dist is not None:
        ax.dist = dist
    if elev is not None:
        ax.elev = elev

    surface_color=np.zeros_like(surface)
    surface_color[:]=np.array([0,0,1])
    surface_size=np.zeros_like(surface)[:,0]
    surface_size[:]=1.5

    sample_color=np.zeros_like(sample)
    sample_color[:]=np.array(color)
    sample_size=np.zeros_like(sample)[:,0]
    sample_size[:]=1.5

    combine=np.concatenate([surface,sample],axis=0)
    combine_color=np.concatenate([surface_color,sample_color],axis=0)
    combine_size=np.concatenate([surface_size,sample_size],axis=0)

    combine_x=-combine[:,0]
    combine_y=combine[:,1]
    combine_z=combine[:,2]
    ax.scatter(combine_x, combine_y, combine_z, s=combine_size,c=combine_color, marker='o',zdir='y')


    if outpath is not None:
        import time
        plt.savefig(outpath)
    else:
        plt.show()
    return