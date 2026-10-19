import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import os
import numpy as np
import pickle
from dataset.SnailRadar.utils import find_nearest_ndx, interpolation, matrix_to_euler


def read_synchronized_data(poses_filepath, lidar_filepath, radar_filepath, t_tolerance):
    times, positions, quats = [], [], []
    with open(poses_filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith('#') or len(parts) < 8:
                continue
            t = float(parts[0])
            times.append(t)
            positions.append(list(map(float, parts[1:4])))
            quats.append(list(map(float, parts[4:8])))
    system_timestamps = np.array(times)
    t_min, t_max = times[0], times[-1]
    poses = np.array(positions)
    quats = np.array(quats)
    # Ensure timestamps and poses are sorted in ascending order
    sorted_ndx = np.argsort(system_timestamps, axis=0)
    system_timestamps = system_timestamps[sorted_ndx]
    poses = poses[sorted_ndx]
    quats = quats[sorted_ndx]

    # list & filter LiDAR data files
    lidar_all_files = sorted([f for f in os.listdir(lidar_filepath) if f.endswith('.pcd')])
    lidar_all_times = np.array([float(f[:-4]) for f in lidar_all_files])
    lidar_valid_mask = (lidar_all_times >= t_min) & (lidar_all_times <= t_max)
    lidar_files = [lidar_all_files[i] for i in np.nonzero(lidar_valid_mask)[0]]
    all_lidar_timestamps = lidar_all_times[lidar_valid_mask]

    # list & filter radar data files
    radar_all_files = sorted([f for f in os.listdir(radar_filepath) if f.endswith('.pcd')])
    radar_all_times = np.array([float(f[:-4]) for f in radar_all_files])
    radar_valid_mask = (radar_all_times >= t_min) & (radar_all_times <= t_max)
    radar_files = [radar_all_files[i] for i in np.nonzero(radar_valid_mask)[0]]
    all_radar_timestamps = radar_all_times[radar_valid_mask]

    interp_poses_lidar = interpolation(system_timestamps, poses, quats, all_lidar_timestamps)

    synchronized_data_list = []
    count_rejected = 0

    for ndx, lidar_ts in enumerate(all_lidar_timestamps):
        synchronized_data_dict = dict()

        # Find index of the closest timestamp
        closest_radar_ts_ndx = find_nearest_ndx(lidar_ts, all_radar_timestamps)
        radar_ts = all_radar_timestamps[closest_radar_ts_ndx]
        delta_s = abs(lidar_ts - radar_ts)
        # Timestamp is in nanoseconds = 1e-9 second
        if delta_s > t_tolerance:
            count_rejected += 1
            continue

        synchronized_data_dict['lidar_ts'] = lidar_ts
        synchronized_data_dict['lidar_filename'] = lidar_files[ndx]
        synchronized_data_dict['radar_ts'] = radar_ts
        synchronized_data_dict['radar_filename'] = radar_files[closest_radar_ts_ndx]
        synchronized_data_dict['pose'] = interp_poses_lidar[ndx]
        euler_angles = matrix_to_euler(synchronized_data_dict['pose'][:3, :3])
        yaw = euler_angles[0]
        synchronized_data_dict['yaw'] = yaw

        synchronized_data_list.append(synchronized_data_dict)

    print(f'{len(synchronized_data_list)} samples with valid pose, {count_rejected} rejected due to unknown pose')
    return synchronized_data_list


def gen_info_sequence(sequence_name, poses_file, lidar_filepath, radar_filepath, t_tolerance):
    infos = list()
    synchronized_data_list = read_synchronized_data(poses_file, lidar_filepath, radar_filepath, t_tolerance)

    for synchronized_data_dict in synchronized_data_list:
        # store scene info
        info = dict()
        info['sequence_name'] = sequence_name
        info['timestamp'] = synchronized_data_dict['lidar_ts']
        lidar_names = ['LIDAR_TOP']
        radar_names = ['RADAR_TOP']
        lidar_infos = dict()
        radar_infos = dict()

        for lidar_name in lidar_names:
            lidar_info = dict()
            lidar_info['ego_pose'] = synchronized_data_dict['pose']
            lidar_info['yaw'] = synchronized_data_dict['yaw']
            lidar_info['filename'] = os.path.join(lidar_filepath, str(synchronized_data_dict['lidar_filename']))
            lidar_infos[lidar_name] = lidar_info

        for radar_name in radar_names:
            radar_info = dict()
            radar_info['ego_pose'] = synchronized_data_dict['pose']
            radar_info['yaw'] = synchronized_data_dict['yaw']
            radar_info['filename'] = os.path.join(radar_filepath, str(synchronized_data_dict['radar_filename']))
            radar_infos[radar_name] = radar_info

        info['lidar_infos'] = lidar_infos
        info['radar_infos'] = radar_infos
        infos.append(info)

    return infos


def gen_info(dataset_root, sequences, pose_time_tolerance):
    all_infos = list()
    for sequence in sequences:
        sequence_path = os.path.join(dataset_root, sequence)
        assert os.path.exists(sequence_path), f'Cannot access sequence: {sequence_path}'

        pose_file = os.path.join(str(sequence_path), 'utm50r_T_xt32.txt')
        assert os.path.exists(pose_file), f'Cannot access global pose file: {pose_file}'

        rel_lidar_path = os.path.join(sequence, 'xt32')
        lidar_path = os.path.join(dataset_root, rel_lidar_path)
        assert os.path.exists(lidar_path), f'Cannot access lidar scans: {lidar_path}'

        rel_radar_path = os.path.join(sequence, 'eagleg7/enhanced')
        radar_path = os.path.join(dataset_root, rel_radar_path)
        assert os.path.exists(radar_path), f'Cannot access radar scans: {radar_path}'

        cur_infos = gen_info_sequence(sequence, pose_file, lidar_path, radar_path, pose_time_tolerance)
        all_infos.extend(cur_infos)
    return all_infos


if __name__ == '__main__':
    from config.paths import data_root as get_data_root, raw_root, ensure_dir
    data_root = raw_root("snail", "/path/to/Snail-Radar")
    info_root = ensure_dir(os.path.join(get_data_root("snail", "snail"), "info"))
    print(f"Writing Snail info pickles to {info_root}")

    sequences = ['20231208/data4', '20240116/data5', '20240123/data2']
    all_infos = gen_info(data_root, sequences, 0.1)
    with open(os.path.join(info_root, 'snail_infos-train.pkl'), 'wb') as f:
        pickle.dump(all_infos, f)

    sequences = ['20240116_eve/data3', '20240116/data2']
    all_infos = gen_info(data_root, sequences, 0.1)
    with open(os.path.join(info_root, 'snail_infos-test.pkl'), 'wb') as f:
        pickle.dump(all_infos, f)
