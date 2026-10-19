"""Optional radar-query to LiDAR-database mining, restricted to allowed negatives."""
import numpy as np


def _indices(values, database_size):
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('Negative candidate indices must be a one-dimensional integer array')
    if values.size and (values.min() < 0 or values.max() >= database_size):
        raise ValueError('Negative candidate index outside the training database')
    return np.unique(values.astype(np.int64, copy=False))


def mine_cross_modal_negatives(lidar_database, radar_queries, allowed_negatives, hard_count, *, block_size=128):
    database = np.asarray(lidar_database, dtype=np.float32)
    queries = np.asarray(radar_queries, dtype=np.float32)
    if (database.ndim != 2 or queries.ndim != 2 or database.shape[1] != queries.shape[1]
            or database.shape[1] == 0 or len(allowed_negatives) != len(queries)):
        raise ValueError('Expected compatible database/query descriptors and one candidate set per query')
    if not np.isfinite(database).all() or not np.isfinite(queries).all():
        raise ValueError('Non-finite descriptors in hard-negative mining')
    if hard_count < 1 or block_size < 1:
        raise ValueError('Hard count and block size must be positive')
    result = np.empty((len(queries), hard_count), dtype=np.int64)
    database_norm = np.einsum('ij,ij->i', database, database)
    for start in range(0, len(queries), block_size):
        batch = queries[start:start + block_size]
        distances = (np.einsum('ij,ij->i', batch, batch)[:, None] + database_norm[None, :]
                     - 2 * batch @ database.T)
        for offset, row in enumerate(distances):
            candidates = _indices(allowed_negatives[start + offset], len(database))
            if len(candidates) < hard_count:
                raise ValueError('Too few geographically valid negatives for the requested hard count')
            # Break exact distance ties by database index, independently of input order.
            order = np.lexsort((candidates, row[candidates]))[:hard_count]
            result[start + offset] = candidates[order]
    return result


def sample_mixed_negatives(hard_indices, allowed_indices, total_count, rng=np.random):
    hard = np.asarray(hard_indices)
    allowed = np.asarray(allowed_indices)
    if (hard.ndim != 1 or allowed.ndim != 1 or not np.issubdtype(hard.dtype, np.integer)
            or not np.issubdtype(allowed.dtype, np.integer) or len(np.unique(hard)) != len(hard)
            or not np.isin(hard, allowed).all() or total_count < len(hard)):
        raise ValueError('Hard negatives must be distinct allowed training candidates')
    remaining = np.setdiff1d(allowed, hard)
    random_count = total_count - len(hard)
    if random_count and len(remaining) == 0:
        raise ValueError('No remaining geographically valid random negatives')
    randoms = rng.choice(remaining, random_count, replace=len(remaining) < random_count)
    return np.concatenate((hard, randoms)).astype(np.int64, copy=False)


def sample_mixed_negatives_preserving_rng(hard_indices, allowed_indices, total_count):
    """Consume the old negative draw so future positive choices remain identical."""
    local = np.random.RandomState(0)
    local.set_state(np.random.get_state())
    np.random.choice(allowed_indices, total_count)
    return sample_mixed_negatives(hard_indices, allowed_indices, total_count, local)
