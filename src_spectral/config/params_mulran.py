import os
from config.paths import data_root, raw_root


class ModelParams:
    def __init__(self):
        self.raw_root = raw_root("mulran", "/path/to/MulRan")
        self.base_root = data_root("mulran", "mulran")
        root = self.base_root

        self.train_info_root = [
            os.path.join(root, "info", "mulran_infos-sejong.pkl"),
            os.path.join(root, "info", "mulran_infos-kaist.pkl"),
        ]
        self.train_bev_dataset_root = os.path.join(root, "bev", "mm_bev")
        self.train_database_index_list = [
            os.path.join(root, "index", "sejong_db.npy"),
            os.path.join(root, "index", "kaist_db.npy"),
        ]
        self.train_query_index_list = [
            os.path.join(root, "index", "sejong_train_query.npy"),
            os.path.join(root, "index", "kaist_train_query.npy"),
        ]

        test_split = os.environ.get("RLPR_MULRAN_TEST_SPLIT", "dcc")
        if test_split == "sejong":
            self.test_info_root = [os.path.join(root, "info", "mulran_infos-sejong.pkl")]
            self.test_bev_dataset_root = os.path.join(root, "bev", "mm_bev")
            self.test_database_index_list = [os.path.join(root, "index", "sejong_test_db.npy")]
            self.test_query_index_list = [os.path.join(root, "index", "sejong_test_query.npy")]
        elif test_split == "riverside":
            self.test_info_root = [os.path.join(root, "info", "mulran_infos-riverside.pkl")]
            self.test_bev_dataset_root = os.path.join(root, "bev", "mm_bev")
            self.test_database_index_list = [os.path.join(root, "index", "riverside_test_db.npy")]
            self.test_query_index_list = [os.path.join(root, "index", "riverside_test_query.npy")]
        else:
            self.test_info_root = [os.path.join(root, "info", "mulran_infos-dcc.pkl")]
            self.test_bev_dataset_root = os.path.join(root, "bev", "mm_bev")
            self.test_database_index_list = [os.path.join(root, "index", "dcc_test_db.npy")]
            self.test_query_index_list = [os.path.join(root, "index", "dcc_test_query.npy")]

        self.pos_dist_threshold = 9
        self.neg_dist_threshold = 12
        self.gth_dist_threshold = 9
