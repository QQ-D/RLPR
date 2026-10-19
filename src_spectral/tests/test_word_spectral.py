import unittest

import torch

from modules.word_spectral import WordSpectralFilterLayer
from modules.phase_transformer import PhaseTransformerEncoderLayer


class WordSpectralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)

    def test_measured_amplitude_is_discarded_and_mask_remains_dynamic(self):
        layer = WordSpectralFilterLayer(8)
        x = torch.randn(2, 8, 13, 57)
        a, aux_a = layer(x)
        b, aux_b = layer(x * 7)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(aux_a['mask'], aux_b['mask'])
        _, changed = layer(torch.randn_like(x))
        self.assertFalse(torch.allclose(aux_a['mask'], changed['mask']))
        self.assertGreater(float(aux_a['mask'][..., 0, 0].mean()),
                           float(aux_a['mask'][..., 6, -1].mean()))

    def test_zero_input_is_zero_and_backward_is_finite(self):
        layer = WordSpectralFilterLayer(8)
        x = torch.zeros(1, 8, 13, 57, requires_grad=True)
        y, aux = layer(x)
        self.assertEqual(float(y.abs().sum()), 0)
        self.assertEqual(float(aux['confidence'].sum()), 0)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_mask_and_phase_transformer_are_trainable_for_both_shapes(self):
        layer = WordSpectralFilterLayer(8)
        encoder = PhaseTransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0)
        for height, width in [(13, 57), (25, 25)]:
            layer.zero_grad(); encoder.zero_grad()
            x = torch.randn(2, 8, height, width, requires_grad=True)
            _, aux = layer(x)
            tokens = torch.cat((aux['spectrum'].real, aux['spectrum'].imag), 1).flatten(2).transpose(1, 2)
            out = encoder(tokens)
            target = torch.randn_like(out)
            (out - target).square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(layer.cutoff_logit.grad.abs().sum()), 0)
            self.assertGreater(float(layer.conditioner[-1].weight.grad.abs().sum()), 0)
            self.assertGreater(float(encoder.self_attn.in_proj_weight.grad.abs().sum()), 0)

    def test_transformer_qk_read_circular_phase_not_input_amplitude(self):
        encoder = PhaseTransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0)
        observed = []
        hook = encoder.self_attn.register_forward_pre_hook(lambda module, args: observed.append(args))
        tokens = torch.randn(2, 7, 16)
        encoder(tokens)
        encoder(tokens * torch.rand(2, 7, 1).add(.5))
        hook.remove()
        torch.testing.assert_close(observed[0][0], observed[1][0])
        torch.testing.assert_close(observed[0][1], observed[1][1])
        self.assertFalse(torch.equal(observed[0][2], observed[1][2]))


if __name__ == '__main__':
    unittest.main()
