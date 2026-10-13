import random
from torch.utils.data import DataLoader, default_collate


def make_collate_fn():
    def collate_fn(batch):
        if None not in batch:
            return default_collate(batch)
        none_batch_idx = [ind for ind, value in enumerate(batch) if value is None]
        all_ind = list(range(len(batch)))
        alternate_batch_idx_list = [ind for ind in all_ind if ind not in none_batch_idx]
        alternate_batch_idx = random.sample(alternate_batch_idx_list, len(none_batch_idx))
        for i in range(len(alternate_batch_idx)):
            batch[none_batch_idx[i]] = batch[alternate_batch_idx[i]]
        return default_collate(batch)
    return collate_fn
