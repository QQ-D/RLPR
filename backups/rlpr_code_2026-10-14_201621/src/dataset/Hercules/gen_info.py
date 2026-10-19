"""
Build Hercules multimodal info pickles (SnailRadar-compatible schema).

Each sample pairs one LiDAR ``*.bin`` under ``<seq>/Aeva`` with the temporally closest
radar ``*.bin`` under ``<seq>/continental`` if |Δt| <= ``t_tolerance`` (seconds).

Pose convention (per user):
  - ``lidar_infos['LIDAR_TOP']``: ego pose interpolated from ``Aeva_gt.txt`` at the LiDAR timestamp.
  - ``radar_infos['RADAR_TOP']``: ego pose interpolated from ``Continental_gt.txt`` at the
    paired radar timestamp (not the LiDAR time).

Timestamps in filenames and ``*_gt.txt`` first column are assumed to be **nanoseconds**
(as on the released Hercules exports).

Default outputs:
  - ``hercules_infos-sc.pkl``: SportsComplex/01, SportsComplex/02
  - ``hercules_infos-rl.pkl``: RiverIsland/01, RiverIsland/02

Run from repo root, e.g.:
  python dataset/Hercules/gen_info.py
"""
from __future__ import annotations

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import os
from config.paths import data_root, raw_root
import pickle
import warnings

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def find_nearest_ndx(ts, timestamps):
    """Nearest index in sorted ``timestamps`` (same semantics as SnailRadar.utils)."""
    ndx = int(np.searchsorted(timestamps, ts))
    if ndx == 0:
        return 0
    if ndx == len(timestamps):
        return ndx - 1
    if ts - timestamps[ndx - 1] < timestamps[ndx] - ts:
        return ndx - 1
    return ndx


def pos_interpolate_batch(keytimes, keypositions, querytimes):
    interp_positions = np.zeros((len(querytimes), 3))
    for i in range(keypositions.shape[1]):
        interp_positions[:, i] = np.interp(querytimes, keytimes, keypositions[:, i])
    return interp_positions


def rot_slerp_batch(keytimes, keyquats, querytimes):
    keyrots = R.from_quat(keyquats)
    slerp = Slerp(keytimes, keyrots)
    return slerp(querytimes).as_quat()


def interpolation(gt_times, gt_positions, gt_quats, query_times):
    """Interpolate SE(3) poses (SnailRadar.utils.interpolation)."""
    interp_positions = pos_interpolate_batch(gt_times, gt_positions, query_times)
    interp_quats = rot_slerp_batch(gt_times, gt_quats, query_times)
    poses = []
    for i, q_i in enumerate(interp_quats):
        rot_mat = R.from_quat(q_i).as_matrix()
        xyz = interp_positions[i].reshape(3, 1)
        pose = np.hstack((rot_mat, xyz))
        pose = np.vstack((pose, [0, 0, 0, 1]))
        poses.append(pose)
    return poses


def matrix_to_euler(rot_mat, seq="ZYX", degrees=False):
    return R.from_matrix(rot_mat).as_euler(seq, degrees=degrees)


def read_gt_poses(poses_filepath: str):
    """Parse ``t x y z qx qy qz qw`` trajectory; return sorted arrays (t in file units)."""
    times, positions, quats = [], [], []
    with open(poses_filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#") or len(parts) < 8:
                continue
            t = float(parts[0])
            times.append(t)
            positions.append(list(map(float, parts[1:4])))
            quats.append(list(map(float, parts[4:8])))

    if not times:
        raise ValueError(f"No pose rows in {poses_filepath}")

    system_timestamps = np.asarray(times, dtype=np.float64)
    poses = np.asarray(positions, dtype=np.float64)
    quats = np.asarray(quats, dtype=np.float64)

    sorted_ndx = np.argsort(system_timestamps, axis=0)
    system_timestamps = system_timestamps[sorted_ndx]
    poses = poses[sorted_ndx]
    quats = quats[sorted_ndx]
    return system_timestamps, poses, quats


def read_synchronized_data(
    aeva_gt_path: str,
    continental_gt_path: str,
    lidar_filepath: str,
    radar_filepath: str,
    t_tolerance_s: float,
):
    """
    For each LiDAR bin (anchor), find nearest radar bin; reject if time delta too large.
    LiDAR pose from Aeva_gt at lidar time; radar pose from Continental_gt at radar time.
    """
    aeva_times, aeva_pos, aeva_quat = read_gt_poses(aeva_gt_path)
    cont_times, cont_pos, cont_quat = read_gt_poses(continental_gt_path)

    t_min_a, t_max_a = float(aeva_times[0]), float(aeva_times[-1])
    t_min_c, t_max_c = float(cont_times[0]), float(cont_times[-1])

    lidar_files = sorted([f for f in os.listdir(lidar_filepath) if f.endswith(".bin")])
    radar_files = sorted([f for f in os.listdir(radar_filepath) if f.endswith(".bin")])

    if not lidar_files:
        warnings.warn(f"No LiDAR .bin under {lidar_filepath}", stacklevel=2)
        return []
    if not radar_files:
        warnings.warn(f"No radar .bin under {radar_filepath}", stacklevel=2)
        return []

    lidar_ts_arr = np.array([float(os.path.splitext(f)[0]) for f in lidar_files], dtype=np.float64)
    radar_ts_arr = np.array([float(os.path.splitext(f)[0]) for f in radar_files], dtype=np.float64)

    lidar_valid = (lidar_ts_arr >= t_min_a) & (lidar_ts_arr <= t_max_a)
    lidar_files = [lidar_files[i] for i in np.nonzero(lidar_valid)[0]]
    lidar_ts_arr = lidar_ts_arr[lidar_valid]

    if lidar_ts_arr.size == 0:
        warnings.warn(
            f"No LiDAR timestamps overlap Aeva_gt range [{t_min_a}, {t_max_a}]",
            stacklevel=2,
        )
        return []

    tol = float(t_tolerance_s) * 1e9
    synchronized_data_list = []
    count_rejected_time = 0
    count_rejected_radar_pose = 0

    for ndx, lidar_ts in enumerate(lidar_ts_arr):
        closest_radar_ndx = find_nearest_ndx(lidar_ts, radar_ts_arr)
        radar_ts = radar_ts_arr[closest_radar_ndx]
        if abs(lidar_ts - radar_ts) > tol:
            count_rejected_time += 1
            continue

        if radar_ts < t_min_c or radar_ts > t_max_c:
            count_rejected_radar_pose += 1
            continue

        interp_lidar = interpolation(
            aeva_times, aeva_pos, aeva_quat, np.array([lidar_ts], dtype=np.float64)
        )
        pose_lidar = interp_lidar[0]
        yaw_lidar = float(matrix_to_euler(pose_lidar[:3, :3])[0])

        interp_radar = interpolation(
            cont_times, cont_pos, cont_quat, np.array([radar_ts], dtype=np.float64)
        )
        pose_radar = interp_radar[0]
        yaw_radar = float(matrix_to_euler(pose_radar[:3, :3])[0])

        synchronized_data_list.append(
            {
                "lidar_ts": float(lidar_ts),
                "lidar_filename": lidar_files[ndx],
                "radar_ts": float(radar_ts),
                "radar_filename": radar_files[closest_radar_ndx],
                "lidar_pose": pose_lidar,
                "lidar_yaw": yaw_lidar,
                "radar_pose": pose_radar,
                "radar_yaw": yaw_radar,
            }
        )

    print(
        f"{len(synchronized_data_list)} samples with valid pairs; "
        f"{count_rejected_time} rejected (|Δt| > {t_tolerance_s}s); "
        f"{count_rejected_radar_pose} rejected (radar_ts outside Continental_gt range)"
    )
    return synchronized_data_list


def gen_info_sequence(
    sequence_name: str,
    aeva_gt: str,
    continental_gt: str,
    lidar_filepath: str,
    radar_filepath: str,
    t_tolerance_s: float,
):
    infos = []
    synchronized_data_list = read_synchronized_data(
        aeva_gt,
        continental_gt,
        lidar_filepath,
        radar_filepath,
        t_tolerance_s,
    )

    for item in synchronized_data_list:
        info = {
            "sequence_name": sequence_name,
            "timestamp": item["lidar_ts"],
            "lidar_infos": {
                "LIDAR_TOP": {
                    "ego_pose": item["lidar_pose"],
                    "yaw": item["lidar_yaw"],
                    "filename": os.path.join(lidar_filepath, item["lidar_filename"]),
                }
            },
            "radar_infos": {
                "RADAR_TOP": {
                    "ego_pose": item["radar_pose"],
                    "yaw": item["radar_yaw"],
                    "filename": os.path.join(radar_filepath, item["radar_filename"]),
                }
            },
        }
        infos.append(info)

    return infos


def gen_info(dataset_root: str, sequences: list[str], t_tolerance_s: float):
    all_infos = []
    for sequence in sequences:
        sequence_path = os.path.join(dataset_root, sequence)
        if not os.path.isdir(sequence_path):
            warnings.warn(f"Skip missing sequence: {sequence_path}", stacklevel=2)
            continue

        aeva_gt = os.path.join(sequence_path, "Aeva_gt.txt")
        continental_gt = os.path.join(sequence_path, "Continental_gt.txt")
        lidar_path = os.path.join(sequence_path, "Aeva")
        radar_path = os.path.join(sequence_path, "continental")

        for p, label in (
            (aeva_gt, "Aeva_gt.txt"),
            (continental_gt, "Continental_gt.txt"),
            (lidar_path, "Aeva/"),
            (radar_path, "continental/"),
        ):
            if not os.path.exists(p):
                warnings.warn(f"Skip {sequence}: missing {label} ({p})", stacklevel=2)
                break
        else:
            cur = gen_info_sequence(
                sequence,
                aeva_gt,
                continental_gt,
                lidar_path,
                radar_path,
                t_tolerance_s,
            )
            all_infos.extend(cur)

    return all_infos


def main():
    parser = argparse.ArgumentParser(description="Build Hercules multimodal info pickles.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=raw_root("hercules", "/path/to/Hercules"),
        help="Root containing SportsComplex/, RiverIsland/, ...",
    )
    parser.add_argument(
        "--info_root",
        type=str,
        default=os.path.join(data_root("hercules", "hercules"), "info"),
        help="Directory for output .pkl files (created if missing).",
    )
    parser.add_argument(
        "--t_tolerance",
        type=float,
        default=0.1,
        help="Max |lidar_ts - radar_ts| in seconds (default 0.1, same as Snail).",
    )
    args = parser.parse_args()

    os.makedirs(args.info_root, exist_ok=True)

    sequences_sc = ["SportsComplex/01", "SportsComplex/02"]
    sequences_rl = ["RiverIsland/01", "RiverIsland/02"]

    out_sc = os.path.join(args.info_root, "hercules_infos-sc.pkl")
    out_rl = os.path.join(args.info_root, "hercules_infos-rl.pkl")

    print("=== hercules_infos-rl.pkl ===")
    infos_rl = gen_info(args.dataset_root, sequences_rl, args.t_tolerance)
    with open(out_rl, "wb") as f:
        pickle.dump(infos_rl, f)
    print(f"Wrote {len(infos_rl)} entries -> {out_rl}")

    print("=== hercules_infos-sc.pkl ===")
    infos_sc = gen_info(args.dataset_root, sequences_sc, args.t_tolerance)
    with open(out_sc, "wb") as f:
        pickle.dump(infos_sc, f)
    print(f"Wrote {len(infos_sc)} entries -> {out_sc}")


if __name__ == "__main__":
    main()
