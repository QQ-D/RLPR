import unittest

import numpy as np

from dataset.cross_negatives import (mine_cross_modal_negatives, sample_mixed_negatives,
                                     sample_mixed_negatives_preserving_rng)


class CrossNegativeTests(unittest.TestCase):
    def test_radar_queries_rank_lidar_database_with_geographic_exclusions(self):
        database = np.array([[0., 0.], [1., 0.], [3., 0.], [5., 0.]])
        queries = np.array([[.1, 0.], [4.9, 0.]])
        allowed = [np.array([1, 2, 3]), np.array([0, 1, 2])]
        mined = mine_cross_modal_negatives(database, queries, allowed, 2)
        np.testing.assert_array_equal(mined, [[1, 2], [2, 1]])

    def test_ties_are_stable_and_candidate_order_does_not_change_ranking(self):
        database = np.array([[-1., 0.], [1., 0.], [0., 2.]])
        queries = np.zeros((1, 2))
        a = mine_cross_modal_negatives(database, queries, [[2, 1, 0]], 2)
        b = mine_cross_modal_negatives(database, queries, [[0, 2, 1]], 2)
        np.testing.assert_array_equal(a, [[0, 1]])
        np.testing.assert_array_equal(a, b)

    def test_insufficient_candidates_and_invalid_descriptors_fail(self):
        for database, queries, allowed in (
            (np.ones((3, 2)), np.ones((1, 2)), [[1]]),
            (np.full((3, 2), np.nan), np.ones((1, 2)), [[0, 1, 2]]),
            (np.ones((3, 2)), np.ones((1, 3)), [[0, 1, 2]]),
            (np.ones((3, 2)), np.ones((1, 2)), [[0, 1, 3]]),
        ):
            with self.subTest(), self.assertRaises(ValueError):
                mine_cross_modal_negatives(database, queries, allowed, 2)

    def test_mixed_sampling_keeps_hard_examples_and_draws_distinct_allowed_randoms(self):
        allowed = np.arange(20)
        hard = np.array([19, 18, 17, 16, 15])
        a = sample_mixed_negatives(hard, allowed, 10, np.random.default_rng(42))
        b = sample_mixed_negatives(hard, allowed, 10, np.random.default_rng(42))
        np.testing.assert_array_equal(a[:5], hard)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(a)), 10)
        self.assertTrue(set(a).issubset(set(allowed)))
        self.assertTrue(set(a[5:]).isdisjoint(set(hard)))

    def test_mixed_sampling_rejects_positive_or_duplicate_hard_indices(self):
        for hard in ([0, 99], [0, 0]):
            with self.subTest(hard=hard), self.assertRaises(ValueError):
                sample_mixed_negatives(hard, np.arange(10), 4, np.random.default_rng(0))

    def test_mining_does_not_change_later_positive_sampling_random_stream(self):
        before = np.random.get_state()
        try:
            allowed, hard = np.arange(30), np.arange(25, 30)
            np.random.seed(42)
            np.random.choice(allowed, 10)
            expected_next_positives = np.random.choice(np.arange(8), 20)
            np.random.seed(42)
            actual = sample_mixed_negatives_preserving_rng(hard, allowed, 10)
            np.testing.assert_array_equal(np.random.choice(np.arange(8), 20), expected_next_positives)
            np.testing.assert_array_equal(actual[:5], hard)
            self.assertEqual(len(set(actual)), 10)
        finally:
            np.random.set_state(before)


if __name__ == '__main__':
    unittest.main()
