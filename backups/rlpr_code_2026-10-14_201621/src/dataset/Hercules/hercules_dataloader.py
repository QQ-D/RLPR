from torch.utils.data import DataLoader
from config.params_full import ModelParamsFull
from dataset.Hercules.hercules_dataset import QueryDataset, TripletDataset
from dataset.collate import make_collate_fn


def make_dataloader(working_mode, params: ModelParamsFull):
    hp = params.hercules_param

    if working_mode == "train":
        train_dataset = TripletDataset(
            hp.train_info_root,
            hp.train_bev_dataset_root,
            hp.train_database_index_list,
            hp.train_query_index_list,
            params.pos_num,
            params.neg_num,
            hp.neg_dist_threshold,
            hp.pos_dist_threshold,
            hp.neg_rot_threshold,
            hp.pos_rot_threshold,
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
            hp.train_info_root,
            hp.train_bev_dataset_root,
            hp.train_database_index_list,
            hp.train_query_index_list,
            hp.gth_dist_threshold,
            hp.gth_rot_threshold,
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

    if working_mode == "test":
        test_dataset = QueryDataset(
            hp.test_info_root,
            hp.test_bev_dataset_root,
            hp.test_database_index_list,
            hp.test_query_index_list,
            hp.gth_dist_threshold,
            hp.gth_rot_threshold,
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

    raise ValueError(f"Unknown working_mode: {working_mode!r}")
