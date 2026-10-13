import os
from config.paths import data_root, raw_root


class ModelParams:
    def __init__(self):
        self.raw_root = raw_root("hercules", "/path/to/Hercules")
        self.base_root = data_root("hercules", "hercules")
        root = self.base_root

        self.measure_range = 80.0
        self.w = 100
        self.h = 100
        self.res = 1

        self.train_info_root = [os.path.join(root, "info", "hercules_infos-rl.pkl")]
        self.train_bev_dataset_root = os.path.join(root, "bev", "mm_bev_r80_h100")
        self.train_database_index_list = [os.path.join(root, "index", "hercules_rl_01.npy")]
        self.train_query_index_list = [os.path.join(root, "index", "hercules_rl_02.npy")]

        self.test_info_root = [os.path.join(root, "info", "hercules_infos-sc.pkl")]
        self.test_bev_dataset_root = os.path.join(root, "bev", "mm_bev_r80_h100")
        self.test_database_index_list = [os.path.join(root, "index", "hercules_sc_01.npy")]
        self.test_query_index_list = [os.path.join(root, "index", "hercules_sc_02.npy")]

        self.pos_dist_threshold = 9
        self.neg_dist_threshold = 12
        self.gth_dist_threshold = 9
        self.pos_rot_threshold = 30
        self.neg_rot_threshold = 60
        self.gth_rot_threshold = 30
        self.r_bev_point_drop_ratio = 0
