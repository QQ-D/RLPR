"""
Build Hercules retrieval indices from multimodal info pickles (SnailRadar logic).

Uses ``lidar_infos['LIDAR_TOP']`` for xy / yaw, same column layout as
``dataset/SnailRadar/multimodal/gen_index.py``: ``[info_idx, x, y, yaw]``.

River Island (train split from ``hercules_infos-rl.pkl``):
  - ``hercules_rl_whole.npy``, ``hercules_rl_01.npy`` (database), ``hercules_rl_02.npy`` (query)

Sports Complex (test split from ``hercules_infos-sc.pkl``):
  - ``hercules_sc_whole.npy``, ``hercules_sc_01.npy`` (database), ``hercules_sc_02.npy`` (query)

Default thresholds match Snail ``__main__``: ``dis_th_db=0.2``, ``angle_th_db=1`` (deg),
``pos_th=9.0``, ``angle_th=30`` (deg).

Run from repo root:
  python dataset/Hercules/gen_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import os
from config.paths import data_root
import pickle
import random
import warnings

import numpy as np
from sklearn.neighbors import NearestNeighbors


def process_infos_pair(
    infos_filepath: str,
    database_seq: str,
    query_seq: str,
    dis_th_db: float = 0.2,
    angle_th_db: float = 1.0,
    pos_th: float = 9.0,
    angle_th: float = 30.0,
):
    """
    Like Snail ``process_infos_tests``, but database = ``database_seq`` rows only and
    query = ``query_seq`` rows only (no multi-sequence DB except the one named).
    """
    with open(infos_filepath, "rb") as f:
        infos = pickle.load(f)

    if not infos:
        raise ValueError(f"No entries in info pickle: {infos_filepath}")

    pos_xy = []
    yaw_whole = []
    sequence_names = []

    for i, info in enumerate(infos):
        pos = info["lidar_infos"]["LIDAR_TOP"]["ego_pose"]
        pos_xy.append(np.array([pos[0, 3], pos[1, 3]]))
        yaw_whole.append(info["lidar_infos"]["LIDAR_TOP"]["yaw"])
        sequence_names.append(str(info["sequence_name"]))

    pos_xy = np.array(pos_xy, dtype=np.float32)
    yaw_whole = np.array(yaw_whole, dtype=np.float32).reshape(-1, 1)

    angle_th_db_rad = float(angle_th_db) * np.pi / 180.0
    angle_th_rad = float(angle_th) * np.pi / 180.0

    allowed = {database_seq, query_seq}
    bad = set(sequence_names) - allowed
    if bad:
        raise ValueError(
            f"Unexpected sequence_name in {infos_filepath}: {sorted(bad)} "
            f"(expected only {database_seq!r} and {query_seq!r})"
        )

    print("total frames: ", pos_xy.shape[0])

    fi_db = np.array([i for i, s in enumerate(sequence_names) if s == database_seq], dtype=np.int64)
    fi_q = np.array([i for i, s in enumerate(sequence_names) if s == query_seq], dtype=np.int64)

    if fi_db.size == 0:
        raise ValueError(f"No frames with sequence_name == {database_seq!r}")
    if fi_q.size == 0:
        raise ValueError(f"No frames with sequence_name == {query_seq!r}")

    pos_whole = np.concatenate(
        (np.arange(len(pos_xy), dtype=np.int32).reshape(-1, 1), pos_xy),
        axis=1,
    ).astype(np.float32)

    pos_db_train = pos_whole[fi_db]
    pos_db = pos_db_train[0, :].reshape(1, -1)
    yaw_db_train = yaw_whole[fi_db]
    yaw_db = yaw_db_train[0].reshape(1, -1)
    for i in range(1, pos_db_train.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_db[:, 1:3])
        dis, index = knn.kneighbors(pos_db_train[i, 1:3].reshape(1, -1), 1, return_distance=True)
        ref_yaw = yaw_db[index.flatten()[0], 0]
        current_yaw = yaw_db_train[i, 0]
        angle_diff = np.abs(current_yaw - ref_yaw)
        min_yaw_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        if dis > dis_th_db or min_yaw_diff > angle_th_db_rad:
            pos_db = np.concatenate((pos_db, pos_db_train[i, :].reshape(1, -1)), axis=0)
            yaw_db = np.concatenate((yaw_db, yaw_db_train[i].reshape(1, -1)), axis=0)
    print("database frames: ", pos_db.shape[0])

    pos_val_test = pos_whole[fi_q]
    pos_val_test_dsp = pos_val_test[0, :].reshape(1, -1)
    yaw_val_test = yaw_whole[fi_q]
    yaw_val_test_dsp = yaw_val_test[0].reshape(1, -1)
    for i in range(1, pos_val_test.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_val_test_dsp[:, 1:3])
        dis, index = knn.kneighbors(pos_val_test[i, 1:3].reshape(1, -1), 1, return_distance=True)
        ref_yaw = yaw_val_test_dsp[index.flatten()[0], 0]
        current_yaw = yaw_val_test[i, 0]
        angle_diff = np.abs(current_yaw - ref_yaw)
        min_yaw_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        if dis > dis_th_db or min_yaw_diff > angle_th_db_rad:
            pos_val_test_dsp = np.concatenate((pos_val_test_dsp, pos_val_test[i, :].reshape(1, -1)), axis=0)
            yaw_val_test_dsp = np.concatenate((yaw_val_test_dsp, yaw_val_test[i].reshape(1, -1)), axis=0)
    pos_val_test = pos_val_test_dsp
    yaw_val_test = yaw_val_test_dsp

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_db[:, 1:3])
    pos_val_test_new = []
    yaw_val_test_new = []
    for i in range(len(pos_val_test)):
        dis, index = knn.radius_neighbors(
            pos_val_test[i, 1:3].reshape(1, -1),
            radius=pos_th,
            return_distance=True,
            sort_results=True,
        )
        dis = dis.flatten()[0]
        if len(dis) == 0:
            continue
        index = index.flatten()[0]
        current_yaw = yaw_val_test[i]
        angle_diffs = np.abs(current_yaw - yaw_db[index].flatten())
        min_angles = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        min_angle = np.min(min_angles)
        if dis[0] < pos_th and min_angle < angle_th_rad:
            pos_val_test_new.append(pos_val_test[i, :])
            yaw_val_test_new.append(yaw_val_test[i])

    if len(pos_val_test_new) == 0:
        pos_val_test = np.zeros((0, pos_whole.shape[1]), dtype=np.float32)
        yaw_val_test = np.zeros((0, 1), dtype=np.float32)
        warnings.warn("No query frames passed radius+yaw filter.", stacklevel=2)
    else:
        pos_val_test = np.array(pos_val_test_new)
        yaw_val_test = np.array(yaw_val_test_new)
    print("test query frames: ", pos_val_test.shape[0])

    pos_whole = np.concatenate((pos_whole, yaw_whole), axis=-1)
    pos_db = np.concatenate((pos_db, yaw_db), axis=-1)
    pos_val_test = np.concatenate((pos_val_test, yaw_val_test), axis=-1)

    return pos_whole, pos_db, pos_val_test


def main():
    parser = argparse.ArgumentParser(description="Build Hercules gen_index npy files.")
    parser.add_argument(
        "--dataroot",
        type=str,
        default=data_root("hercules", "hercules"),
        help="Contains info/ and index/ subdirectories.",
    )
    parser.add_argument("--dis_th_db", type=float, default=0.2)
    parser.add_argument("--angle_th_db", type=float, default=1.0)
    parser.add_argument("--pos_th", type=float, default=9.0)
    parser.add_argument("--angle_th", type=float, default=30.0)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    info_dir = os.path.join(args.dataroot, "info")
    index_dir = os.path.join(args.dataroot, "index")
    os.makedirs(index_dir, exist_ok=True)

    rl_infos = os.path.join(info_dir, "hercules_infos-rl.pkl")
    sc_infos = os.path.join(info_dir, "hercules_infos-sc.pkl")

    kw = dict(
        dis_th_db=args.dis_th_db,
        angle_th_db=args.angle_th_db,
        pos_th=args.pos_th,
        angle_th=args.angle_th,
    )

    print("=== RiverIsland (train) ===")
    pos_whole_rl, pos_db_rl, pos_q_rl = process_infos_pair(
        rl_infos,
        database_seq="RiverIsland/01",
        query_seq="RiverIsland/02",
        **kw,
    )
    np.save(os.path.join(index_dir, "hercules_rl_whole.npy"), pos_whole_rl)
    np.save(os.path.join(index_dir, "hercules_rl_01.npy"), pos_db_rl)
    np.save(os.path.join(index_dir, "hercules_rl_02.npy"), pos_q_rl)

    print("=== SportsComplex (test) ===")
    pos_whole_sc, pos_db_sc, pos_q_sc = process_infos_pair(
        sc_infos,
        database_seq="SportsComplex/01",
        query_seq="SportsComplex/02",
        **kw,
    )
    np.save(os.path.join(index_dir, "hercules_sc_whole.npy"), pos_whole_sc)
    np.save(os.path.join(index_dir, "hercules_sc_01.npy"), pos_db_sc)
    np.save(os.path.join(index_dir, "hercules_sc_02.npy"), pos_q_sc)
    print(pos_db_sc.shape, pos_q_sc.shape, pos_db_rl.shape, pos_q_rl.shape)

    print("Saved under:", index_dir)


if __name__ == "__main__":
    main()
