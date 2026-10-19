"""
Generate LiDAR / radar 100×100 polar BEV PNGs for Hercules (Snail-style layout).

Output: ``<dataroot>/bev/mm_bev_r80_h100/{lidar,radar}/<stem>.png``

Run from repo root::

    python dataset/Hercules/gen_bev.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import os
import pickle
import warnings

import numpy as np
from PIL import Image

from config.paths import data_root, ensure_dir

try:
    from tqdm import tqdm as _tqdm
except ImportError:

    def _tqdm(it, **_kw):
        return it


def voxel_downsample_xyz(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    if xyz.size == 0:
        return xyz
    xyz = np.asarray(xyz, dtype=np.float32)
    grid = np.floor(xyz / float(voxel_size)).astype(np.int64)
    _, unique_idx = np.unique(grid, axis=0, return_index=True)
    return xyz[unique_idx]


# LiDAR → radar (row points: p' = p_hom @ T_RL.T).
T_RL = np.array(
    [
        [1.0, 0.0, 0.0, -1.445],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def transform_lidar_xyz_to_radar_frame(xyz: np.ndarray, t_r_l: np.ndarray) -> np.ndarray:
    if xyz.size == 0:
        return xyz
    xyz = np.asarray(xyz, dtype=np.float64)
    hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    return (hom @ t_r_l.T)[:, :3].astype(np.float32)


def load_lidar_bin(file_path: str, record_size: int = 29) -> np.ndarray:
    with open(file_path, "rb") as f:
        file_content = f.read()
    num_records = len(file_content) // record_size
    trimmed = file_content[: num_records * record_size]
    dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("reflectivity", np.float32),
            ("velocity", np.float32),
            ("time_offset_ns", np.int32),
            ("line_index", np.uint8),
            ("intensity", np.float32),
        ]
    )
    data = np.frombuffer(trimmed, dtype=dtype)
    return np.vstack(
        [
            data["x"],
            data["y"],
            data["z"],
            data["velocity"],
            np.sqrt(data["x"] ** 2 + data["y"] ** 2 + data["z"] ** 2),
            data["reflectivity"],
        ]
    ).T


def load_radar_bin(file_path: str, record_size: int = 29) -> np.ndarray:
    with open(file_path, "rb") as f:
        file_content = f.read()
    num_records = len(file_content) // record_size
    trimmed = file_content[: num_records * record_size]
    dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("v", np.float32),
            ("r", np.float32),
            ("RCS", np.int8),
            ("azimuth", np.float32),
            ("elevation", np.float32),
        ]
    )
    data = np.frombuffer(trimmed, dtype=dtype)
    return np.vstack(
        [
            data["x"],
            data["y"],
            data["z"],
            data["v"],
            data["r"],
            data["RCS"].astype(np.float32),
            data["azimuth"],
            data["elevation"],
        ]
    ).T


def polar_bev_projection(
    pcl_input: np.ndarray,
    measure_range: float,
    resolution: float,
    w: int = 100,
    h: int = 100,
    downsample: bool = True,
) -> np.ndarray:
    if pcl_input.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    pcl_input_np = np.asarray(pcl_input, dtype=np.float32)
    if downsample:
        pcl_input_np = voxel_downsample_xyz(pcl_input_np, resolution)
    if pcl_input_np.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    mat_global_image = np.zeros((h, w), dtype=np.uint8)
    yaw = np.arctan2(pcl_input_np[:, 1], pcl_input_np[:, 0])
    scan_r = np.linalg.norm(pcl_input_np[:, :2], axis=1)

    yaw_deg = np.degrees(yaw)
    yaw_mask = (yaw_deg >= -60) & (yaw_deg <= 60)
    yaw = yaw[yaw_mask]
    scan_r = scan_r[yaw_mask]
    if yaw.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    yaw_min = -np.pi / 3
    yaw_max = np.pi / 3
    w_ind = np.floor((yaw - yaw_min) / (yaw_max - yaw_min) * w).astype(int)
    h_ind = np.floor(scan_r / measure_range * h).astype(int)

    valid_mask = (w_ind >= 0) & (w_ind < w) & (h_ind >= 0) & (h_ind < h)
    valid_w_ind = w_ind[valid_mask]
    valid_h_ind = h_ind[valid_mask]

    np.add.at(mat_global_image, (valid_h_ind, valid_w_ind), 1)
    mat_global_image = np.clip(mat_global_image, 0, 10) * 10
    mat_global_image[np.where(mat_global_image > 255)] = 255
    peak = float(np.max(mat_global_image))
    if peak > 0:
        mat_global_image = (mat_global_image / peak * 255.0).astype(np.uint8)
    return mat_global_image


def filter_lidar_xyz(points_xyz: np.ndarray, ground_z: float = 0.0, min_dist: float = 2.0):
    ground_mask = points_xyz[:, 2] > ground_z
    points_xyz = points_xyz[ground_mask]
    dist_mask = np.linalg.norm(points_xyz[:, :3], axis=1) > min_dist
    return points_xyz[dist_mask]


def filter_radar_xyz(points_xyz: np.ndarray, max_dist: float = 180.0):
    dist_mask = np.linalg.norm(points_xyz[:, :3], axis=1) < max_dist
    return points_xyz[dist_mask]


def save_png(out_dir: str, src_bin_path: str, img: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src_bin_path))[0]
    Image.fromarray(img).save(os.path.join(out_dir, stem + ".png"))


def process_infos_to_bev(
    infos: list,
    bs_whole: np.ndarray,
    bev_root: str,
    ground_z: float,
    split_tag: str = "",
):
    lidar_dir = os.path.join(bev_root, "lidar")
    radar_dir = os.path.join(bev_root, "radar")
    ensure_dir(lidar_dir)
    ensure_dir(radar_dir)

    desc = f"{split_tag} BEV" if split_tag else "Hercules BEV"
    for i in _tqdm(range(bs_whole.shape[0]), desc=desc):
        cur_ind = int(bs_whole[i, 0])
        cur_info = infos[cur_ind]

        lidar_filename = cur_info["lidar_infos"]["LIDAR_TOP"]["filename"]
        radar_filename = cur_info["radar_infos"]["RADAR_TOP"]["filename"]
        if not os.path.isfile(lidar_filename) or not os.path.isfile(radar_filename):
            warnings.warn(f"Skip missing bin: {lidar_filename} / {radar_filename}", stacklevel=2)
            continue

        lidar_xyz = load_lidar_bin(lidar_filename)[:, :3]
        lidar_xyz = transform_lidar_xyz_to_radar_frame(lidar_xyz, T_RL)
        radar_xyz = load_radar_bin(radar_filename)[:, :3]

        lidar_f = filter_lidar_xyz(lidar_xyz, ground_z=ground_z, min_dist=2.0)
        radar_f = filter_radar_xyz(radar_xyz, max_dist=180.0)

        l_bev = polar_bev_projection(lidar_f, 80.0, 0.8, 100, 100, downsample=True)
        r_bev = polar_bev_projection(radar_f, 80.0, 0.8, 100, 100, downsample=False)

        save_png(lidar_dir, lidar_filename, l_bev)
        save_png(radar_dir, radar_filename, r_bev)


def main():
    parser = argparse.ArgumentParser(description="Hercules 100×100 polar BEV generation.")
    parser.add_argument(
        "--dataroot",
        type=str,
        default=data_root("hercules", "hercules"),
        help="Contains info/, index/, bev/ subdirectories.",
    )
    parser.add_argument("--ground_z", type=float, default=0.0, help="Keep LiDAR points with z > this value.")
    args = parser.parse_args()

    dataroot = args.dataroot
    info_dir = os.path.join(dataroot, "info")
    index_dir = os.path.join(dataroot, "index")
    bev_root = ensure_dir(os.path.join(dataroot, "bev", "mm_bev_r80_h100"))
    print(f"Writing Hercules BEV PNGs to {bev_root}")

    splits = [
        (os.path.join(info_dir, "hercules_infos-rl.pkl"), os.path.join(index_dir, "hercules_rl_whole.npy"), "RiverIsland"),
        (os.path.join(info_dir, "hercules_infos-sc.pkl"), os.path.join(index_dir, "hercules_sc_whole.npy"), "SportsComplex"),
    ]

    for info_path, ind_path, tag in splits:
        if not os.path.isfile(info_path):
            warnings.warn(f"Skip {tag}: missing info {info_path}", stacklevel=2)
            continue
        if not os.path.isfile(ind_path):
            warnings.warn(f"Skip {tag}: missing index {ind_path}", stacklevel=2)
            continue

        with open(info_path, "rb") as f:
            infos = pickle.load(f)
        if not infos:
            warnings.warn(f"Skip {tag}: empty info pickle", stacklevel=2)
            continue

        bs_whole = np.load(ind_path)
        if bs_whole.size == 0:
            warnings.warn(f"Skip {tag}: empty index {ind_path}", stacklevel=2)
            continue

        print(f"=== {tag}: {len(infos)} infos, {bs_whole.shape[0]} index rows ===")
        process_infos_to_bev(infos, bs_whole, bev_root, ground_z=args.ground_z, split_tag=tag)

    print("Done.")


if __name__ == "__main__":
    main()
