import copy
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from modules.word_models import WordCrossPolar, WordLidarPolar, WordRadarPolar
from modules.word_spectral import WordSpectralFilterLayer
from trainer.unimodal import UnimodalTrainer
from utils import pretrain


class WordAmplitudeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        torch.set_num_threads(2)

    def make_model(self, cls, amplitude):
        with patch.dict(os.environ, RLPR_WORD_AMPLITUDE=str(int(amplitude)),
                        RLPR_SPECTRAL_CROSS_FUSION='1'):
            return cls.create()

    def checkpoint(self, model):
        return {'model_variant': model.model_variant, 'spectral_config': model.spectral_config,
                'state_dict': model.state_dict()}

    def test_raw_magnitude_retains_scale_without_changing_phase_content(self):
        layer = WordSpectralFilterLayer(8, retain_amplitude=True)
        x = torch.randn(2, 8, 13, 57)
        y, a = layer(x)
        scaled, b = layer(7 * x)
        torch.testing.assert_close(scaled, 7 * y, atol=1e-5, rtol=2e-5)
        torch.testing.assert_close(a['phase_content'], b['phase_content'], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(a['source_amplitude'], torch.fft.rfft2(x, norm='ortho').abs())
        zero = torch.zeros_like(x, requires_grad=True)
        value, _ = layer(zero)
        self.assertEqual(float(value.abs().sum()), 0)
        value.square().sum().backward()
        self.assertTrue(torch.isfinite(zero.grad).all())

    def test_both_attention_stages_retain_measured_magnitude(self):
        pure = self.make_model(WordCrossPolar, False).eval()
        amp = self.make_model(WordCrossPolar, True).eval()
        amp.load_state_dict(pure.state_dict(), strict=True)
        observed = [[], []]
        hooks = [m.encoder_l.register_forward_pre_hook(
            lambda mod, args, dest=observed[i]: dest.append(args[0].detach().clone()))
                 for i, m in enumerate((pure, amp))]
        x = torch.randn(1, 2, 3, 50, 225)
        with torch.no_grad():
            pure(x, x)
            amp(x, x)
        for hook in hooks:
            hook.remove()
        torch.testing.assert_close(observed[0][0], observed[1][0], atol=0, rtol=0)
        amp.train()
        _, _, aux, _ = amp(x, torch.randn_like(x))
        expected = aux['source_amplitude'] * aux['mask'] * aux['valid_phase']
        # Include DC and all real-spectrum boundary rows/columns.
        for key in ('spectrum', 'fusion_spectrum'):
            torch.testing.assert_close(aux[key].abs(), expected,
                                       atol=2e-6, rtol=2e-5)
        loss = aux['fusion_local_descriptor'].square().sum() + aux['spectrum'].real.mean()
        loss.backward()
        self.assertTrue(torch.isfinite(amp.spectral_l.cutoff_logit.grad).all())
        self.assertGreater(float(amp.spectral_l.cutoff_logit.grad.abs().sum()), 0)

    def test_even_width_boundary_phase_cancellation_keeps_magnitude_and_real_signal(self):
        from modules.word_spectral import restore_measured_amplitude
        layer = WordSpectralFilterLayer(2, retain_amplitude=True)
        _, aux = layer(torch.randn(1, 2, 6, 8))
        # Pure imaginary DC/Nyquist proposals project to zero: use measured phase there.
        proposal = torch.ones_like(aux['spectrum']) * 1j
        restored = restore_measured_amplitude(proposal, aux)
        expected = aux['mask'] * aux['source_amplitude'] * aux['valid_phase']
        torch.testing.assert_close(restored.abs(), expected, atol=2e-6, rtol=2e-5)
        roundtrip = torch.fft.rfft2(torch.fft.irfft2(restored, s=(6, 8), norm='ortho'), norm='ortho')
        torch.testing.assert_close(restored, roundtrip, atol=2e-6, rtol=2e-5)

    def test_default_config_is_compatible_but_mode_mismatch_is_rejected(self):
        pure = self.make_model(WordLidarPolar, False)
        amp = self.make_model(WordLidarPolar, True)
        self.assertTrue(pure.spectral_config['pure_phase'])
        self.assertNotIn('amplitude_mode', pure.spectral_config)
        self.assertFalse(amp.spectral_config['pure_phase'])
        self.assertEqual(amp.spectral_config['amplitude_mode'], 'raw_magnitude_v1')
        with self.assertRaisesRegex(ValueError, 'spectral_config'):
            pretrain.validate_checkpoint_variant(amp, self.checkpoint(pure))
        with patch.dict(os.environ, RLPR_WORD_AMPLITUDE='oops'):
            with self.assertRaisesRegex(ValueError, 'RLPR_WORD_AMPLITUDE'):
                WordLidarPolar.create()

    def test_explicit_unary_warmstart_allows_only_amplitude_delta(self):
        source = self.make_model(WordLidarPolar, False)
        target = self.make_model(WordLidarPolar, True)
        checkpoint = self.checkpoint(source)
        pretrain.warmstart_word_unimodal(target, checkpoint)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value), key)
        for bad in (self.checkpoint(self.make_model(WordRadarPolar, False)),
                    self.checkpoint(self.make_model(WordCrossPolar, False))):
            with self.assertRaises(ValueError):
                pretrain.warmstart_word_unimodal(target, bad)
        incomplete = copy.deepcopy(checkpoint)
        del incomplete['state_dict'][next(iter(incomplete['state_dict']))]
        with self.assertRaises(ValueError):
            pretrain.warmstart_word_unimodal(target, incomplete)
        incompatible = copy.deepcopy(checkpoint)
        incompatible['spectral_config']['coherence_stride'] = 8
        with self.assertRaisesRegex(ValueError, 'spectral_config'):
            pretrain.warmstart_word_unimodal(target, incompatible)

    def test_trainer_warmstart_does_not_resume_optimizer_or_epoch(self):
        source = self.make_model(WordLidarPolar, False)
        target = self.make_model(WordLidarPolar, True)
        checkpoint = dict(self.checkpoint(source), epoch=5, optimizer={'invalid': True})
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'initial.pth')
            torch.save(checkpoint, path)
            trainer = UnimodalTrainer.__new__(UnimodalTrainer)
            trainer.model = target
            trainer.resume = False
            trainer.params = SimpleNamespace(initial_checkpoint_path=path)
            trainer.optimizer = torch.optim.Adam(target.parameters(), lr=5e-5)
            self.assertEqual(trainer._init_weights(), 0)
            self.assertEqual(trainer.optimizer.state_dict()['state'], {})
            self.assertTrue(torch.equal(target.conv1_l.weight, source.conv1_l.weight))


if __name__ == '__main__':
    unittest.main()
