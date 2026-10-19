from torch.utils.data import DataLoader
from config.params_full import ModelParamsFull
from dataset.MulRan.mulran_dataset import TripletDataset, QueryDataset
from dataset.collate import make_collate_fn


def make_dataloader(working_mode, params: ModelParamsFull):
    dp = params.mulran_param

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
        )
        train_query_loader = DataLoader(
            dataset=train_query_dataset,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
        )

        return train_loader, train_query_loader

    if working_mode == 'test':
        test_dataset = QueryDataset(
            dp.test_info_root,
            dp.test_bev_dataset_root,
            dp.test_database_index_list,
            dp.test_query_index_list,
            dp.gth_dist_threshold,
            l_enable=params.l_enable,
            r_enable=params.r_enable,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
        )
        return test_loader, test_dataset
