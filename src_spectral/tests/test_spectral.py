import unittest

import torch

from modules.spectral import SpectralFilterLayer
from modules.loss import phase_coherence_loss


class SpectralLayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(2)

    def test_shapes_real_reconstruction_and_gradients(self):
        layer = SpectralFilterLayer(8)
        for shape in [(2, 8, 13, 57), (2, 8, 25, 25), (2, 8, 12, 24)]:
            x = torch.randn(shape, requires_grad=True)
            y, aux = layer(x)
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(aux['phase'].shape, (*shape[:-1], shape[-1] // 2 + 1))
            self.assertTrue(torch.isfinite(y).all())
            spectrum = torch.fft.rfft2(y, norm='ortho')
            valid = spectrum.abs() > 1e-4
            error = 1 - torch.cos(torch.angle(spectrum) - aux['phase'])
            self.assertLess(float(error[valid].abs().max()), 1e-4)
            layer.zero_grad()
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(layer.cutoff_logit.grad.abs().sum()), 0)
            self.assertGreater(float(layer.conditioner[-1].weight.grad.abs().sum()), 0)

    def test_mask_changes_with_input_and_attenuates_high_frequencies(self):
        layer = SpectralFilterLayer(8, phase_attention=False)
        _, a = layer(torch.randn(2, 8, 13, 57))
        _, b = layer(torch.randn(2, 8, 13, 57) * 10)
        self.assertFalse(torch.allclose(a['mask'], b['mask']))
        self.assertGreater(float(a['mask'][..., 0, 0].mean()), float(a['mask'][..., 6, -1].mean()))

    def test_zero_input_is_finite_and_has_no_phase_confidence(self):
        layer = SpectralFilterLayer(8)
        x = torch.zeros(1, 8, 13, 57, requires_grad=True)
        y, aux = layer(x)
        self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(float(aux['confidence'].sum()), 0)
        loss = phase_coherence_loss(aux['phase'].detach(), aux['phase'], aux['confidence'])
        self.assertEqual(float(loss), 0)
        (y.square().mean() + loss).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_phase_loss_trains_features_and_attention(self):
        layer = SpectralFilterLayer(8)
        x = torch.randn(1, 8, 13, 57, requires_grad=True)
        _, aux = layer(x)
        target = torch.randn_like(aux['phase'])
        loss = phase_coherence_loss(target, aux['phase'], aux['confidence'])
        loss.backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertGreater(float(layer.phase_attention.query.weight.grad.abs().sum()), 0)

    def test_amplitude_mask_precedes_phase_attention(self):
        layer = SpectralFilterLayer(8)
        observed = []
        handle = layer.phase_attention.register_forward_pre_hook(
            lambda module, args: observed.append(args[0].detach().clone()))
        x = torch.randn(1, 8, 13, 57)
        _, aux = layer(x)
        handle.remove()
        expected = torch.fft.rfft2(x, norm='ortho') * aux['mask']
        torch.testing.assert_close(observed[0], expected)


class PhaseLossTests(unittest.TestCase):
    def test_circular_phase_and_teacher_detach(self):
        radar = torch.tensor([0.2, -3.13], requires_grad=True)
        lidar = (radar.detach() + 2 * torch.pi).requires_grad_()
        loss = phase_coherence_loss(radar, lidar)
        self.assertLess(float(loss), 1e-6)
        loss.backward()
        self.assertIsNone(radar.grad)
        self.assertTrue(torch.isfinite(lidar.grad).all())

    def test_weights_ignore_unreliable_bins(self):
        radar = torch.zeros(2)
        lidar = torch.tensor([0., torch.pi], requires_grad=True)
        self.assertEqual(float(phase_coherence_loss(radar, lidar, torch.tensor([1., 0.]))), 0)
        self.assertGreater(float(phase_coherence_loss(radar, lidar)), .9)


if __name__ == '__main__':
    unittest.main()
