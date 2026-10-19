import unittest

import numpy as np
import torch
from scipy.signal import coherence

from modules.coherence import descriptor_coherence, coherence_alignment_loss, spectral_coherence_loss


class CoherenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(2)

    def test_welch_estimator_matches_scipy_and_is_not_trivially_one(self):
        x, y = torch.randn(2, 256, dtype=torch.float64)
        actual = descriptor_coherence(x, y, return_bins=True)
        _, expected = coherence(x.numpy(), y.numpy(), nperseg=32, noverlap=16, detrend=False)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-8, atol=1e-9)
        self.assertLess(float(actual.mean()), .3)
        self.assertGreater(float(descriptor_coherence(x, x)), .999)

    def test_zero_energy_is_finite_and_short_input_is_rejected(self):
        z = torch.zeros(2, 256, requires_grad=True)
        score = descriptor_coherence(z, z)
        self.assertEqual(float(score.mean()), 0)
        score.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        with self.assertRaises(ValueError):
            descriptor_coherence(torch.ones(1, 32), torch.ones(1, 32))

    def test_correct_pairs_beat_permuted_pairs_and_teacher_is_detached(self):
        radar = torch.randn(1, 12, 256, requires_grad=True)
        radar.data[:, 1] = radar.data[:, 0] + .01 * torch.randn(256)
        lidar = radar.detach().clone().add(.02 * torch.randn_like(radar)).requires_grad_()
        matched = coherence_alignment_loss(lidar, radar)
        mismatched = coherence_alignment_loss(lidar.roll(3, dims=1), radar)
        self.assertLess(float(matched), float(mismatched))
        matched.backward()
        self.assertIsNone(radar.grad)
        self.assertTrue(torch.isfinite(lidar.grad).all())
        self.assertGreater(float(lidar.grad.abs().sum()), 0)

    def test_identical_collapsed_descriptors_pay_negative_margin(self):
        x = torch.randn(1, 1, 256).expand(1, 12, -1)
        self.assertGreaterEqual(float(coherence_alignment_loss(x, x)), .199)

    def test_alignment_penalizes_sign_reversal_seen_by_euclidean_retrieval(self):
        radar = torch.nn.functional.normalize(torch.randn(1, 12, 256), dim=-1)
        radar[:, 1] = radar[:, 0]
        same = coherence_alignment_loss(radar, radar)
        opposite = coherence_alignment_loss(-radar, radar)
        self.assertLess(float(same), 1e-5)
        self.assertGreater(float(opposite), float(same) + 2.)

    def test_alignment_matches_signed_cosine_ranking_and_detaches_teacher(self):
        radar = torch.randn(2, 12, 256, requires_grad=True)
        lidar = torch.randn_like(radar, requires_grad=True)
        r = radar.detach()
        score = torch.nn.functional.cosine_similarity
        matched = score(lidar, r, dim=-1)
        forward = score(lidar, r[:, :1], dim=-1)
        reverse = score(lidar[:, :1], r, dim=-1)
        near = .5 * (forward[:, 1] + reverse[:, 1])
        positive = .5 * (matched[:, 0] + near)
        negatives = torch.cat((forward[:, 2:], reverse[:, 2:]), dim=1)
        expected = (1. - matched.mean() + .5 * (1. - near.mean())
                    + torch.relu(.2 + negatives - positive[:, None]).mean())
        actual = coherence_alignment_loss(lidar, radar)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNone(radar.grad)
        self.assertTrue(torch.isfinite(lidar.grad).all())
        self.assertGreater(float(lidar.grad.abs().sum()), 0.)

    def test_alignment_zero_descriptors_have_finite_gradients(self):
        lidar = torch.zeros(1, 12, 256, requires_grad=True)
        radar = torch.zeros_like(lidar, requires_grad=True)
        loss = coherence_alignment_loss(lidar, radar)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(lidar.grad).all())
        self.assertIsNone(radar.grad)

    def test_complex_frequency_coherence_averages_before_normalization(self):
        radar = torch.randn(2, 8, 9, 11, dtype=torch.complex64, requires_grad=True)
        lidar = (radar.detach() + .2 * torch.randn_like(radar)).requires_grad_()
        weights = torch.ones_like(radar.real)
        matched = spectral_coherence_loss(radar, lidar, weights)
        unrelated = spectral_coherence_loss(radar, torch.randn_like(radar), weights)
        self.assertLess(float(matched), float(unrelated))
        self.assertLess(float(spectral_coherence_loss(radar, radar.detach(), weights)), 1e-5)
        matched.backward()
        self.assertIsNone(radar.grad)
        self.assertTrue(torch.isfinite(lidar.grad).all())
        self.assertGreater(float(lidar.grad.abs().sum()), 0)
        zero = torch.zeros_like(lidar, requires_grad=True)
        empty = spectral_coherence_loss(zero.detach(), zero, torch.zeros_like(weights))
        empty.backward()
        self.assertEqual(float(empty), 0)
        self.assertTrue(torch.isfinite(zero.grad).all())


if __name__ == '__main__':
    unittest.main()
