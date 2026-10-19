import random
from torch.utils.data import DataLoader
from config.params_full import ModelParamsFull
from dataset.NuScenes.nuscenes_dataset import TripletDataset, QueryDataset
from dataset.collate import make_collate_fn


def make_dataloader(working_mode, params: ModelParamsFull):
    dp = params.nusc_param

    if working_mode == 'train':
        train_dataset = TripletDataset(
            dp.train_info_root,
            dp.train_bev_dataset_root,
            dp.train_database_index_list,
            dp.train_query_index_list,
            params.pos_num,
            params.neg_num,
            dp.neg_dist_threshold,
            dp.pos_dist_threshold,
            params.output_dim,
            l_enable=params.l_enable,
            r_enable=params.r_enable,
            r_bev_point_drop_ratio=dp.r_bev_point_drop_ratio,
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            collate_fn=make_collate_fn(),
            num_workers=params.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        train_query_dataset = QueryDataset(
            dp.train_info_root,
            dp.train_bev_dataset_root,
            dp.train_database_index_list,
            dp.train_query_index_list,
            dp.gth_dist_threshold,
            l_enable=params.l_enable,
            r_enable=params.r_enable,
            r_bev_point_drop_ratio=dp.r_bev_point_drop_ratio,
        )
        train_query_loader = DataLoader(
            dataset=train_query_dataset,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=params.num_test_workers,
            pin_memory=True,
        )

        val_query_dataset = QueryDataset(
            dp.train_info_root[0:1],
            dp.train_bev_dataset_root,
            dp.train_database_index_list[0:1],
            dp.train_val_query_index_list,
            dp.gth_dist_threshold,
            l_enable=params.l_enable,
            r_enable=params.r_enable,
            r_bev_point_drop_ratio=dp.r_bev_point_drop_ratio,
        )
        val_query_loader = DataLoader(
            dataset=val_query_dataset,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=params.num_test_workers,
            pin_memory=True,
        )

        return train_loader, None, train_query_loader, val_query_loader, val_query_dataset

    if working_mode == 'test':
        test_dataset = QueryDataset(
            dp.test_info_root,
            dp.test_bev_dataset_root,
            dp.test_database_index_list,
            dp.test_query_index_list,
            dp.gth_dist_threshold,
            l_enable=params.l_enable,
            r_enable=params.r_enable,
            r_bev_point_drop_ratio=dp.r_bev_point_drop_ratio,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=params.num_test_workers,
            pin_memory=True,
        )
        return test_loader, test_dataset
