"""Build cross-modal negative caches from training database/query frames only."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from dataset.cross_negatives import mine_cross_modal_negatives


def make_cross_modal_cache_loader(loader, *, batch_size=32):
    return DataLoader(loader.dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=loader.num_workers, pin_memory=loader.pin_memory,
                      collate_fn=loader.collate_fn, generator=torch.Generator().manual_seed(0))


def update_cross_modal_cache(model, triplets, loader, device, cache_root, epoch, hard_count):
    started = time.monotonic()
    frames = loader.dataset
    for name in ('database_index_n_pos', 'query_index_n_pos'):
        if not np.array_equal(getattr(triplets, name), getattr(frames, name)):
            raise ValueError('Training cache index order differs from triplet ' + name)
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last or loader.generator is None:
        raise ValueError('Training cache needs a complete sequential loader with its own generator')
    n_db = len(triplets.database_index_n_pos)
    n_q = len(triplets.query_index_n_pos)
    if len(frames) != n_db + n_q:
        raise ValueError('Training cache must contain database frames followed by query frames')
    lidar_database, radar_queries = [], []
    offset = 0
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc='Cross-modal training cache'):
                lidar, radar = model(batch['lidar_bev'].to(device), batch['radar_bev'].to(device))[:2]
                count = len(batch['lidar_bev'])
                if lidar.ndim != 2 or radar.shape != lidar.shape or len(lidar) != count:
                    raise ValueError('Expected one equally sized LiDAR/radar descriptor per cache frame')
                if offset < n_db:
                    lidar_database.append(lidar[:min(count, n_db - offset)].cpu().numpy())
                if offset + count > n_db:
                    radar_queries.append(radar[max(0, n_db - offset):].cpu().numpy())
                offset += count
    finally:
        model.train(was_training)
    if offset != n_db + n_q or not lidar_database or not radar_queries:
        raise ValueError('Incomplete training descriptor cache')
    mined = mine_cross_modal_negatives(np.concatenate(lidar_database), np.concatenate(radar_queries),
                                      triplets.neg_index, hard_count)
    geographic = np.linalg.norm(triplets.database_index_n_pos[mined, 1:]
                                - triplets.query_index_n_pos[:, None, 1:], axis=-1)
    minimum = float(geographic.min())
    if not np.isfinite(geographic).all() or minimum <= triplets.neg_dist_thres:
        raise ValueError('Hard negative violates the geographic exclusion radius')
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'cross_negatives_epoch_{epoch}.npz'
    np.savez_compressed(path, hard_indices=mined,
                        database_frame_ids=triplets.database_index_n_pos[:, 0].astype(np.int64),
                        query_frame_ids=triplets.query_index_n_pos[:, 0].astype(np.int64))
    report = dict(epoch=int(epoch), hard_count=int(hard_count), database_frames=n_db, query_frames=n_q,
                  direction='radar query to lidar database', minimum_negative_distance_m=minimum,
                  cache_path=str(path), cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  seconds=time.monotonic() - started)
    path.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    triplets.cross_modal_hard_negatives = mined
    return report
