import unittest

import numpy as np

from trainer.eval import max_f1_score, recall_at_k


class EvaluationMetricTests(unittest.TestCase):
    def test_zero_precision_recall_does_not_poison_max_f1(self):
        with np.errstate(divide='raise', invalid='raise'):
            score = max_f1_score([0., .8], [0., .5])
        self.assertAlmostEqual(score, 2 * .8 * .5 / (.8 + .5))

    def test_empty_or_all_zero_curve_has_zero_f1(self):
        self.assertEqual(max_f1_score([], []), 0.)
        self.assertEqual(max_f1_score([0., 0.], [0., 0.]), 0.)

    def test_unsorted_thresholds_are_counted_independently(self):
        prediction = np.arange(30)[None, :]
        actual = recall_at_k(prediction, [[16]], [1, 5, 10, 20, 13])
        np.testing.assert_array_equal(actual, [0., 0., 0., 100., 0.])

    def test_queries_without_ground_truth_are_excluded(self):
        prediction = np.array([[0, 1], [0, 1]])
        actual = recall_at_k(prediction, [[0], []], [1, 2])
        np.testing.assert_array_equal(actual, [100., 100.])

    def test_no_valid_queries_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'ground truth'):
            recall_at_k(np.array([[0]]), [[]], [1])

    def test_mismatched_query_counts_are_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, 'query count'):
            recall_at_k(np.array([[0], [1]]), [[0]], [1])


if __name__ == '__main__':
    unittest.main()
