"""Add positions for the exact frames sampled by the existing triplet dataset."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.MulRan.mulran_dataset import TripletDataset


class SpatialTripletDataset(TripletDataset):
    @classmethod
    def from_existing(cls,dataset):
        if type(dataset) is not TripletDataset:
            raise ValueError('Expected the original MulRan triplet dataset')
        result=cls.__new__(cls)
        result.__dict__.update(dataset.__dict__)
        result._positions={}
        for row in np.concatenate([dataset.database_index_n_pos,dataset.query_index_n_pos]):
            key=int(row[0]);xy=np.asarray(row[1:],dtype=np.float64)
            if row[0]!=key or xy.shape!=(2,) or not np.isfinite(xy).all():
                raise ValueError('Invalid frame index or training coordinates')
            if key in result._positions and not np.array_equal(result._positions[key],xy):
                raise ValueError('Conflicting positions for the same training frame')
            result._positions[key]=xy.copy()
        result._sampled_frames=None
        return result

    def load_data(self,index,*args,**kwargs):
        if self._sampled_frames is not None:
            self._sampled_frames.append(int(index))
        return super().load_data(index,*args,**kwargs)

    def __getitem__(self,index):
        self._sampled_frames=[]
        try:
            item=super().__getitem__(index)
            frames=self._sampled_frames
        finally:
            self._sampled_frames=None
        if item is None:return None
        if len(frames)!=1+self.n_pos+self.n_neg:
            raise ValueError('Sampled frame/position count mismatch')
        item['tuple_positions']=torch.from_numpy(np.stack([self._positions[f] for f in frames]))
        return item


class SpatialDataLoader(DataLoader):
    tuple_positions=None

    def __iter__(self):
        try:
            for batch in super().__iter__():
                self.tuple_positions=batch['tuple_positions']
                yield batch
        finally:
            self.tuple_positions=None


def attach_positions(loader):
    if loader.persistent_workers:
        raise ValueError('Exact resume requires nonpersistent workers')
    return SpatialDataLoader(dataset=SpatialTripletDataset.from_existing(loader.dataset),
        batch_size=loader.batch_size,sampler=loader.sampler,num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,pin_memory=loader.pin_memory,drop_last=loader.drop_last,
        timeout=loader.timeout,worker_init_fn=loader.worker_init_fn,generator=loader.generator,
        multiprocessing_context=loader.multiprocessing_context,
        prefetch_factor=loader.prefetch_factor if loader.num_workers else None)
