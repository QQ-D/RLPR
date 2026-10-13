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
from dataset.MulRan.utils import load_pc, load_radar_polar_pool
from tqdm import tqdm


def lidar_polar_bev_projection(pcl_input, measure_range, resolution, w=900, h=200):
    pcl_input_pc = o3d.geometry.PointCloud()
    pcl_input_pc.points = o3d.utility.Vector3dVector(pcl_input)
    pcl_input_pc = pcl_input_pc.voxel_down_sample(voxel_size=resolution)
    pcl_input_np = np.asarray(pcl_input_pc.points)

    mat_global_image = np.zeros((h, w), dtype=np.uint8)

    yaw = -np.arctan2(pcl_input_np[:, 1], pcl_input_np[:, 0])
    scan_r = np.linalg.norm(pcl_input_np[:, :2], axis=1)

    w_ind = np.floor((0.5 * (yaw / np.pi + 1.0)) * w).astype(int)
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

        radar_filename = cur_info['radar_infos']['RADAR_TOP']['filename']
        r_bev_img = load_radar_polar_pool(radar_filename, 225, 50)
        lidar_filename = cur_info['lidar_infos']['LIDAR_TOP']['filename']
        lidar_pcd = load_pc(lidar_filename)
        l_bev_img = lidar_polar_bev_projection(lidar_pcd, 80.0, 1.6, 225, 50)

        filename_l = os.path.basename(cur_info['lidar_infos']['LIDAR_TOP']['filename']).split('.')[0] + '.png'
        filename_r = os.path.basename(cur_info['radar_infos']['RADAR_TOP']['filename']).split('.')[0] + '.png'
        cv2.imwrite(os.path.join(radar_bev_root, filename_r), r_bev_img)
        cv2.imwrite(os.path.join(lidar_bev_root, filename_l), l_bev_img)


if __name__ == '__main__':
    from config.paths import data_root, raw_root
    dataset_root = data_root("mulran", "mulran")

    # --------------------------Sejong------------------------------
    bev_root = os.path.join(dataset_root, 'bev', 'mm_bev')
    infos_path = os.path.join(dataset_root, 'info', 'mulran_infos-sejong.pkl')
    db_ind_root = os.path.join(dataset_root, 'index', 'sejong_db.npy')
    test_db_ind_root = os.path.join(dataset_root, 'index', 'sejong_test_db.npy')
    test_query_ind_root = os.path.join(dataset_root, 'index', 'sejong_test_query.npy')
    ind_db = np.load(db_ind_root)
    ind_tdb_query = np.load(test_db_ind_root)
    ind_test_query = np.load(test_query_ind_root)
    ind_whole = np.concatenate([ind_db, ind_tdb_query, ind_test_query], axis=0)

    gen_bev(bev_root, infos_path, ind_whole)

    # -------------------------Riverside-----------------------------
    infos_path = os.path.join(dataset_root, 'info', 'mulran_infos-riverside.pkl')
    db_ind_root = os.path.join(dataset_root, 'index', 'riverside_test_db.npy')
    query_ind_root = os.path.join(dataset_root, 'index', 'riverside_test_query.npy')
    ind_db = np.load(db_ind_root)
    ind_query = np.load(query_ind_root)
    ind_whole = np.concatenate([ind_db, ind_query], axis=0)

    gen_bev(bev_root, infos_path, ind_whole)

    # ----------------------------DCC--------------------------------
    infos_path = os.path.join(dataset_root, 'info', 'mulran_infos-dcc.pkl')
    db_ind_root = os.path.join(dataset_root, 'index', 'dcc_test_db.npy')
    query_ind_root = os.path.join(dataset_root, 'index', 'dcc_test_query.npy')
    ind_db = np.load(db_ind_root)
    ind_query = np.load(query_ind_root)
    ind_whole = np.concatenate([ind_db, ind_query], axis=0)

    gen_bev(bev_root, infos_path, ind_whole)

    # ----------------------------KAIST--------------------------------
    infos_path = os.path.join(dataset_root, 'info', 'mulran_infos-kaist.pkl')
    db_ind_root = os.path.join(dataset_root, 'index', 'kaist_db.npy')
    query_ind_root = os.path.join(dataset_root, 'index', 'kaist_train_query.npy')
    ind_db = np.load(db_ind_root)
    ind_query = np.load(query_ind_root)
    ind_whole = np.concatenate([ind_db, ind_query], axis=0)

    gen_bev(bev_root, infos_path, ind_whole)
