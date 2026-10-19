import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import copy
import os
import pickle
import cv2
import open3d as o3d
import numpy as np
from nuscenes.nuscenes import transform_matrix, Quaternion
from tqdm import tqdm


def load_pcd_open3d(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.asarray(pcd.points, dtype=np.float32)


def polar_bev_projection(pcl_input, measure_range, resolution, w=100, h=100, downsample=True):
    if downsample:
        pcl_input_pc = o3d.geometry.PointCloud()
        pcl_input_pc.points = o3d.utility.Vector3dVector(pcl_input)
        pcl_input_pc = pcl_input_pc.voxel_down_sample(voxel_size=resolution)
        pcl_input_np = np.asarray(pcl_input_pc.points)
    else:
        pcl_input_np = np.asarray(pcl_input)

    mat_global_image = np.zeros((h, w), dtype=np.uint8)

    yaw = np.arctan2(pcl_input_np[:, 1], pcl_input_np[:, 0])
    scan_r = np.linalg.norm(pcl_input_np[:, :2], axis=1)

    yaw_deg = np.degrees(yaw)
    yaw_mask = (yaw_deg >= -60) & (yaw_deg <= 60)
    yaw = yaw[yaw_mask]
    scan_r = scan_r[yaw_mask]

    yaw_min = -np.pi / 3
    yaw_max = np.pi / 3
    w_ind_normalized = (yaw - yaw_min) / (yaw_max - yaw_min)
    w_ind = np.floor(w_ind_normalized * w).astype(int)
    h_ind = np.floor(scan_r / measure_range * h).astype(int)

    valid_mask = (w_ind >= 0) & (w_ind < w) & (h_ind >= 0) & (h_ind < h)

    valid_w_ind = w_ind[valid_mask]
    valid_h_ind = h_ind[valid_mask]

    np.add.at(mat_global_image, (valid_h_ind, valid_w_ind), 1)
    mat_global_image = np.clip(mat_global_image, 0, 10)

    mat_global_image = mat_global_image * 10

    mat_global_image[np.where(mat_global_image > 255)] = 255
    mat_global_image = mat_global_image / np.max(mat_global_image) * 255

    return mat_global_image


def gen_bev(bev_fileroot, info_filepath, bs_whole):
    if not os.path.exists(bev_fileroot):
        os.makedirs(bev_fileroot)
    lidar_bev_root = os.path.join(bev_fileroot, 'lidar')
    if not os.path.exists(lidar_bev_root):
        os.makedirs(lidar_bev_root)
    radar_bev_root = os.path.join(bev_fileroot, 'radar')
    if not os.path.exists(radar_bev_root):
        os.makedirs(radar_bev_root)

    with open(info_filepath, 'rb') as f:
        infos = pickle.load(f)

    for i in tqdm(range(0, bs_whole.shape[0])):
        cur_ind = bs_whole[i][0]
        cur_info = infos[int(cur_ind)]

        lidar_filename = cur_info['lidar_infos']['LIDAR_TOP']['filename']
        lidar_pcd = load_pcd_open3d(lidar_filename)
        ground_mask = lidar_pcd[:, 2] > -1.5
        lidar_pcd = lidar_pcd[ground_mask]
        dist_mask = np.linalg.norm(lidar_pcd[:, :3], 2, axis=1) > 2.0
        lidar_pcd = lidar_pcd[dist_mask]
        t_r_l = transform_matrix(
            np.array([-0.07, 0.0, 0.115]),
            Quaternion(np.array([0.7071, 0.0000, 0.0000, 0.7071])),
            inverse=False
        )
        lidar_pcd = t_r_l.dot(np.concatenate([lidar_pcd, np.ones_like(lidar_pcd[:, 0:1])], axis=-1).T).T[:, :3]
        l_bev_img = polar_bev_projection(lidar_pcd, 80.0, 0.8, 100, 100, downsample=True)

        radar_filename = cur_info['radar_infos']['RADAR_TOP']['filename']
        radar_pcd = load_pcd_open3d(radar_filename)
        dist_mask = np.linalg.norm(radar_pcd[:, :3], 2, axis=1) < 180.0
        radar_pcd = radar_pcd[dist_mask]
        r_bev_img = polar_bev_projection(radar_pcd, 80.0, 0.8, 100, 100, downsample=False)

        filename_l = os.path.splitext(os.path.basename(lidar_filename))[0] + '.png'
        filename_r = os.path.splitext(os.path.basename(radar_filename))[0] + '.png'
        cv2.imwrite(os.path.join(lidar_bev_root, filename_l), l_bev_img)
        cv2.imwrite(os.path.join(radar_bev_root, filename_r), r_bev_img)


if __name__ == '__main__':
    from config.paths import data_root, raw_root
    dataset_root = data_root("snail", "snail")
    bev_root = os.path.join(dataset_root, 'bev', 'mm_bev')

    # --------------------------train------------------------------
    infos_path = os.path.join(dataset_root, 'info', 'snail_infos-train.pkl')
    db_ind_root = os.path.join(dataset_root, 'index', 'snail_train_db.npy')
    query_ind_root = os.path.join(dataset_root, 'index', 'snail_train_query.npy')
    ind_db = np.load(db_ind_root)
    ind_query = np.load(query_ind_root)
    ind_whole = np.concatenate([ind_db, ind_query], axis=0)
    gen_bev(bev_root, infos_path, ind_whole)

    # --------------------------test------------------------------
    infos_path = os.path.join(dataset_root, 'info', 'snail_infos-test.pkl')
    db_ind_root = os.path.join(dataset_root, 'index', 'snail_test_db.npy')
    query_ind_root = os.path.join(dataset_root, 'index', 'snail_test_query.npy')
    ind_db = np.load(db_ind_root)
    ind_query = np.load(query_ind_root)
    ind_whole = np.concatenate([ind_db, ind_query], axis=0)
    gen_bev(bev_root, infos_path, ind_whole)
