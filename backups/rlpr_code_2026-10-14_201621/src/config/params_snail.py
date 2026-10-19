import os
from config.paths import data_root, raw_root


class ModelParams:
    def __init__(self):
        self.raw_root = raw_root("snail", "/path/to/Snail-Radar")
        self.base_root = data_root("snail", "snail")
        root = self.base_root

        self.train_info_root = [os.path.join(root, "info", "snail_infos-train.pkl")]
        self.train_bev_dataset_root = os.path.join(root, "bev", "mm_bev")
        self.train_database_index_list = [os.path.join(root, "index", "snail_train_db.npy")]
        self.train_query_index_list = [os.path.join(root, "index", "snail_train_query.npy")]

        self.test_info_root = [os.path.join(root, "info", "snail_infos-test.pkl")]
        self.test_bev_dataset_root = os.path.join(root, "bev", "mm_bev")
        self.test_database_index_list = [os.path.join(root, "index", "snail_test_db.npy")]
        self.test_query_index_list = [os.path.join(root, "index", "snail_test_query.npy")]

        self.pos_dist_threshold = 9
        self.neg_dist_threshold = 12
        self.gth_dist_threshold = 9
        self.pos_rot_threshold = 30
        self.neg_rot_threshold = 60
        self.gth_rot_threshold = 30
