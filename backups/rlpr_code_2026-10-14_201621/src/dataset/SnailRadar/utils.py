import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import struct
import open3d as o3d


def find_nearest_ndx(ts, timestamps):
    ndx = np.searchsorted(timestamps, ts)
    if ndx == 0:
        return ndx
    elif ndx == len(timestamps):
        return ndx - 1
    else:
        assert timestamps[ndx-1] <= ts <= timestamps[ndx]
        if ts - timestamps[ndx-1] < timestamps[ndx] - ts:
            return ndx - 1
        else:
            return ndx


def rot_slerp_batch(keytimes, keyquats, querytimes):
    keyrots = R.from_quat(keyquats)
    slerp = Slerp(keytimes, keyrots)
    interp_rots = slerp(querytimes)
    return interp_rots.as_quat()

def pos_interpolate_batch(keytimes, keypositions, querytimes):
    interp_positions = np.zeros((len(querytimes), 3))
    for i in range(keypositions.shape[1]):
        interp_positions[:, i] = np.interp(querytimes, keytimes, keypositions[:, i])
    return interp_positions


def interpolation(gt_times, gt_positions, gt_quats, radar_times):
    interp_positions = pos_interpolate_batch(gt_times, gt_positions, radar_times)
    interp_quats = rot_slerp_batch(gt_times, gt_quats, radar_times)
    poses = []
    for i, q_i in enumerate(interp_quats):
        quat_max = R.from_quat(q_i).as_matrix()
        xyz = interp_positions[i].reshape(3, 1)
        pose = np.hstack((quat_max, xyz))
        pose = np.vstack((pose, [0, 0, 0, 1]))
        poses.append(pose)
    return poses


def matrix_to_euler(rot_mat, seq='ZYX', degrees=False):
    rotation = R.from_matrix(rot_mat)
    euler = rotation.as_euler(seq, degrees=degrees)
    return euler


hesai_fieldtypecodes = ('f', 'f', 'f', 'f', 'd', 'H')
hesai_unpack_numbytes = (4, 4, 4, 4, 8, 2)


def read_pcd_file(filename):
    """Read a PCD file and return (fields, points: list of lists)."""
    with open(filename, 'rb') as file:
        fields = []
        # Parse header
        while True:
            line = file.readline().decode('utf-8')
            if 'FIELDS' in line:
                fields = line.split()[1:]
            elif 'DATA' in line:
                data_type = line.split()[1]
                break

        # Read each point
        points = []
        while True:
            point_data = {}
            for i, field in enumerate(fields):
                # Choose byte‐length & unpack code
                if "xt32" in filename:
                    num_bytes = hesai_unpack_numbytes[i]
                    code = hesai_fieldtypecodes[i]
                else:
                    num_bytes = 4
                    code = 'f'
                buf = file.read(num_bytes)
                if not buf:
                    return fields, points
                point_data[field] = struct.unpack(code, buf)[0]
            points.append([point_data[f] for f in fields])


def load_pcd(filename):
    """
    Returns an (N×3) float32 array of [x,y,z] by leveraging read_pcd_file.
    """
    fields, pts_list = read_pcd_file(filename)
    pts = np.array(pts_list, dtype=np.float32)
    ix, iy, iz = fields.index('x'), fields.index('y'), fields.index('z')
    return pts[:, [ix, iy, iz]]


def load_pcd_open3d(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.asarray(pcd.points, dtype=np.float32)


def filter_by_yaw(pos_index_array, query_index_n_pos, database_index_n_pos, angle_threshold):
    angle_threshold = angle_threshold * np.pi / 180
    filtered_pos_index_array = []

    for i in range(len(pos_index_array)):
        query_yaw = query_index_n_pos[i, 3]
        candidate_indices = pos_index_array[i]

        if len(candidate_indices) == 0:
            filtered_pos_index_array.append(np.array([], dtype=int))
            continue

        candidate_yaws = database_index_n_pos[candidate_indices, 3]

        angle_diffs = np.abs(query_yaw - candidate_yaws)
        min_angles = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)

        valid_mask = min_angles < angle_threshold
        filtered_indices = candidate_indices[valid_mask]

        filtered_pos_index_array.append(filtered_indices)
    return np.array(filtered_pos_index_array, dtype=object)
