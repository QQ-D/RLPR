import os
from config.paths import data_root, raw_root


class ModelParams:
    def __init__(self):
        self.raw_root = raw_root("nusc", "/path/to/nuscenes")
        self.base_root = data_root("nusc", "nusc")
        root = self.base_root

        self.train_info_root = [os.path.join(root, "info", "nuscenes_infos-bs.pkl")]
        self.train_bev_dataset_root = os.path.join(root, "bev", "bs_shv_bev")
        self.train_database_index_list = [os.path.join(root, "index", "bs_db.npy")]
        self.train_query_index_list = [os.path.join(root, "index", "bs_train_query.npy")]
        self.train_val_query_index_list = [os.path.join(root, "index", "bs_test_query.npy")]

        test_split = os.environ.get("RLPR_NUSC_TEST_SPLIT", "son")
        if test_split == "bs":
            self.test_info_root = [os.path.join(root, "info", "nuscenes_infos-bs.pkl")]
            self.test_bev_dataset_root = os.path.join(root, "bev", "bs_shv_bev")
            self.test_database_index_list = [os.path.join(root, "index", "bs_db.npy")]
            self.test_query_index_list = [os.path.join(root, "index", "bs_test_query.npy")]
        elif test_split == "sq":
            self.test_info_root = [os.path.join(root, "info", "nuscenes_infos-sq.pkl")]
            self.test_bev_dataset_root = os.path.join(root, "bev", "sq_bev")
            self.test_database_index_list = [os.path.join(root, "index", "sq_db.npy")]
            self.test_query_index_list = [os.path.join(root, "index", "sq_test_query.npy")]
        else:
            self.test_info_root = [os.path.join(root, "info", "nuscenes_infos-son.pkl")]
            self.test_bev_dataset_root = os.path.join(root, "bev", "son_bev")
            self.test_database_index_list = [os.path.join(root, "index", "son_db.npy")]
            self.test_query_index_list = [os.path.join(root, "index", "son_test_query.npy")]

        self.pos_dist_threshold = 9
        self.neg_dist_threshold = 12
        self.gth_dist_threshold = 9
        self.r_bev_point_drop_ratio = 0
