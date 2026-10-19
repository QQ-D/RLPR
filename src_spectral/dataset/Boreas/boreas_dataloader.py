from torch.utils.data import DataLoader
from config.params_full import ModelParamsFull
from dataset.Boreas.boreas_dataset import QueryDataset


def make_dataloader(working_mode, params: ModelParamsFull):
    if working_mode == 'train':
        raise NotImplementedError("Boreas is evaluation-only in RLPR.")

    if working_mode == 'test':
        dp = params.boreas_param
        test_dataset = QueryDataset(
            dp.test_info_root,
            dp.test_bev_dataset_root,
            dp.test_database_index_list,
            dp.test_query_index_list,
            dp.gth_dist_threshold,
            dp.pos_rot_threshold,
            l_enable=params.l_enable,
            c_enable=params.c_enable,
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
