"""
Hercules multimodal place-recognition datasets.

BEV tensors follow the same layout as SnailRadar (PNG under ``bev_dataset_root/lidar`` and
``.../radar``, basename from paired ``*.bin`` stems). We reuse
:class:`TripletDataset` and :class:`QueryDataset` from ``dataset.SnailRadar.snail_dataset``.
"""

from dataset.SnailRadar.snail_dataset import BaseDataset, QueryDataset, TripletDataset

__all__ = ["BaseDataset", "QueryDataset", "TripletDataset"]
