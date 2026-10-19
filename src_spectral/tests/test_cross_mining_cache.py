import tempfile
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.cross_mining import make_cross_modal_cache_loader, update_cross_modal_cache
from dataset.MulRan.mulran_dataset import TripletDataset
from config.params_full import ModelParamsFull
from utils.training_state import validate_training_settings


class CacheFrames(Dataset):
    def __init__(self):
        self.database_index_n_pos = np.array([[10, 0., 0.], [20, 20., 0.], [30, 40., 0.]])
        self.query_index_n_pos = self.database_index_n_pos[[0, 2]].copy()

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {'lidar_bev': torch.tensor([float(index), 0.]),
                'radar_bev': torch.tensor([float([100, 100, 100, 2, 0][index]), 0.])}


class CacheModel(torch.nn.Module):
    def forward(self, lidar, radar):
        return lidar, radar, None, None


class CrossMiningCacheTests(unittest.TestCase):
    def test_legacy_random_sampling_checkpoint_cannot_silently_resume_with_hard_mining(self):
        validate_training_settings({}, {'cross_hard_negative_count': 0})
        with self.assertRaisesRegex(ValueError, 'cross_hard_negative_count'):
            validate_training_settings({}, {'cross_hard_negative_count': 5})

    def test_mining_option_defaults_off_and_validates_tuple_capacity(self):
        with patch.dict(os.environ, {'RLPR_TRAINING_ROOT': '', 'RLPR_CROSS_HARD_NEGATIVES': '0'}):
            self.assertEqual(ModelParamsFull().cross_hard_negative_count, 0)
        with patch.dict(os.environ, {'RLPR_TRAINING_ROOT': '', 'RLPR_CROSS_HARD_NEGATIVES': '5'}):
            self.assertEqual(ModelParamsFull().cross_hard_negative_count, 5)
        for value in ('-1', '11', 'bad'):
            with self.subTest(value=value), patch.dict(os.environ, {'RLPR_TRAINING_ROOT': '', 'RLPR_CROSS_HARD_NEGATIVES': value}):
                with self.assertRaisesRegex(ValueError, 'RLPR_CROSS_HARD_NEGATIVES'):
                    ModelParamsFull()

    def test_triplet_sampling_uses_mixed_cache_only_when_enabled(self):
        triplets = TripletDataset.__new__(TripletDataset)
        triplets.query_index_n_pos = np.array([[20, 0., 0.]])
        triplets.database_index_n_pos = np.column_stack((np.arange(30), np.zeros((30, 2))))
        triplets.pos_index = [np.array([0])]
        triplets.neg_index = [np.arange(1, 30)]
        triplets.n_pos, triplets.n_neg = 1, 10
        triplets.l_enable = triplets.r_enable = True
        triplets.augmentation = triplets.is_batch_hard_mining = False
        triplets.load_data = lambda index, *args: (torch.tensor([float(index)]), torch.tensor([float(index)]))
        before = np.random.get_state()
        try:
            np.random.seed(17)
            np.random.choice(triplets.pos_index[0], 1)
            expected = np.random.choice(triplets.neg_index[0], 10)
            np.random.seed(17)
            original = triplets[0]['lidar_bev'].flatten().numpy()
            np.testing.assert_array_equal(original[2:], expected)
            triplets.cross_modal_hard_negatives = np.array([[29, 28, 27, 26, 25]])
            mixed = triplets[0]['lidar_bev'].flatten().numpy()
            np.testing.assert_array_equal(mixed[2:7], [29, 28, 27, 26, 25])
            self.assertEqual(len(set(mixed[2:])), 10)
            self.assertTrue(set(mixed[2:]).issubset(set(triplets.neg_index[0])))
        finally:
            np.random.set_state(before)

    def setup_data(self):
        frames = CacheFrames()
        triplets = SimpleNamespace(database_index_n_pos=frames.database_index_n_pos.copy(),
                                   query_index_n_pos=frames.query_index_n_pos.copy(),
                                   neg_index=[np.array([1, 2]), np.array([0, 1])],
                                   neg_dist_thres=12.)
        loader = make_cross_modal_cache_loader(DataLoader(frames, batch_size=1, num_workers=0), batch_size=2)
        return triplets, loader

    def test_cache_uses_lidar_database_and_radar_query_offsets_without_advancing_training_rng(self):
        triplets, loader = self.setup_data()
        model = CacheModel().train()
        before = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as root:
            report = update_cross_modal_cache(model, triplets, loader, torch.device('cpu'), root, 1, 1)
            np.testing.assert_array_equal(triplets.cross_modal_hard_negatives, [[2], [0]])
            self.assertEqual(report['database_frames'], 3)
            self.assertEqual(report['query_frames'], 2)
            self.assertEqual(report['minimum_negative_distance_m'], 40.)
            saved = np.load(report['cache_path'])
            np.testing.assert_array_equal(saved['database_frame_ids'], [10, 20, 30])
            np.testing.assert_array_equal(saved['query_frame_ids'], [10, 30])
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_misordered_cache_metadata_is_rejected(self):
        triplets, loader = self.setup_data()
        triplets.query_index_n_pos = triplets.query_index_n_pos[::-1]
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(ValueError, 'index'):
            update_cross_modal_cache(CacheModel(), triplets, loader, torch.device('cpu'), root, 1, 1)

    def test_geographically_invalid_negative_cannot_enter_the_cache(self):
        triplets, loader = self.setup_data()
        triplets.neg_index[0] = np.array([0])
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(ValueError, 'geographic'):
            update_cross_modal_cache(CacheModel(), triplets, loader, torch.device('cpu'), root, 1, 1)


if __name__ == '__main__':
    unittest.main()
