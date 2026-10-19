import os
from config.paths import data_root, raw_root

_BOREAS_TEST_SPLITS = {
    "bor-snow": "2021-01-26-11-22",
    "bor-clear": "2021-09-14-20-00",
}


class ModelParams:
    def __init__(self):
        self.raw_root = raw_root("boreas", "/path/to/Boreas")
        self.base_root = data_root("boreas", "boreas")
        root = self.base_root

        test_split = os.environ.get("RLPR_BOREAS_TEST_SPLIT", "bor-clear").lower()
        if test_split not in _BOREAS_TEST_SPLITS:
            valid = ", ".join(sorted(_BOREAS_TEST_SPLITS))
            raise ValueError(
                f"Unknown RLPR_BOREAS_TEST_SPLIT '{test_split}'. Valid values: {valid}"
            )

        stem = f"boreas_infos-{_BOREAS_TEST_SPLITS[test_split]}"
        self.test_info_root = [os.path.join(root, "info", f"{stem}.pkl")]
        self.test_bev_dataset_root = os.path.join(root, "bev", "mm_bev")
        self.test_database_index_list = [os.path.join(root, "index", f"{stem}_db.npy")]
        self.test_query_index_list = [
            os.path.join(root, "index", f"{stem}_test_query.npy")
        ]

        self.pos_dist_threshold = 9
        self.neg_dist_threshold = 12
        self.gth_dist_threshold = 9
        self.pos_rot_threshold = 60
