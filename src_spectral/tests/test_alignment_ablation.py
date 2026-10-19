import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from config.params_full import ModelParamsFull
from modules.loss import InfoNCE
from trainer.cross_modal import CrossModalTrainer


class AlignmentAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(2)

    def trainer_and_outputs(self, alignment='infonce', phase_weight=.1):
        trainer = CrossModalTrainer.__new__(CrossModalTrainer)
        trainer.model = SimpleNamespace(model_variant='spectral_word')
        trainer.params = SimpleNamespace(batch_size=1, phase_loss_weight=phase_weight,
                                        cross_fusion_loss_weight=.5, word_alignment_loss=alignment)
        trainer.info_nce = InfoNCE()
        aux = []
        for _ in range(2):
            spectrum = torch.randn(12, 2, 3, 3, dtype=torch.complex64, requires_grad=True)
            aux.append(dict(local_descriptor=torch.randn(12, 64, requires_grad=True),
                            global_descriptor=torch.randn(12, 64, requires_grad=True),
                            spectrum=spectrum, phase=torch.angle(spectrum),
                            confidence=torch.ones_like(spectrum.real)))
        return trainer, (None, None, *aux)

    def test_word_loss_choice_defaults_to_signed_and_rejects_unknown_values(self):
        with patch.dict(os.environ, {'RLPR_TRAINING_ROOT': ''}):
            os.environ.pop('RLPR_WORD_ALIGNMENT_LOSS', None)
            params = ModelParamsFull()
            self.assertTrue(hasattr(params, 'word_alignment_loss'))
            self.assertEqual(params.word_alignment_loss, 'signed_coherency_v2')
        with patch.dict(os.environ, {'RLPR_TRAINING_ROOT': '', 'RLPR_WORD_ALIGNMENT_LOSS': 'infonce'}):
            self.assertEqual(ModelParamsFull().word_alignment_loss, 'infonce')
        with patch.dict(os.environ, {'RLPR_TRAINING_ROOT': '', 'RLPR_WORD_ALIGNMENT_LOSS': 'typo'}):
            with self.assertRaisesRegex(ValueError, 'RLPR_WORD_ALIGNMENT_LOSS'):
                ModelParamsFull()

    def test_infonce_ablation_uses_original_teacher_to_student_direction(self):
        trainer, outputs = self.trainer_and_outputs()
        loss, parts = trainer._compute_loss(outputs)
        lidar, radar = outputs[2:]
        expected = []
        for descriptor in ('local_descriptor', 'global_descriptor'):
            logits = F.normalize(radar[descriptor].detach(), dim=1) @ F.normalize(lidar[descriptor], dim=1).T / .07
            expected.append(F.cross_entropy(logits, torch.arange(12)))
        self.assertIn('infonce_local', parts)
        torch.testing.assert_close(parts['infonce_local'], expected[0])
        torch.testing.assert_close(parts['infonce_global'], expected[1])
        self.assertIn('spectral_coherence_independent', parts)
        torch.testing.assert_close(loss, sum(expected) + .1 * parts['phase'])
        loss.backward()
        self.assertIsNone(radar['local_descriptor'].grad)
        self.assertIsNone(radar['global_descriptor'].grad)
        self.assertIsNone(radar['spectrum'].grad)
        self.assertGreater(float(lidar['local_descriptor'].grad.abs().sum()), 0.)

    def test_infonce_ablation_keeps_fused_supervision_and_auxiliary_losses(self):
        trainer, outputs = self.trainer_and_outputs()
        lidar = outputs[2]
        for name in ('local_descriptor', 'global_descriptor', 'spectrum', 'phase', 'confidence'):
            lidar['fusion_' + name] = lidar[name].clone()
        loss, parts = trainer._compute_loss(outputs)
        self.assertIn('infonce_global_fused', parts)
        self.assertIn('spectral_coherence_fused', parts)
        torch.testing.assert_close(loss, parts['infonce_local'] + parts['infonce_global'] + .1 * parts['phase'])

    def test_zero_auxiliary_weight_removes_spectral_supervision(self):
        trainer, outputs = self.trainer_and_outputs('signed_coherency_v2', phase_weight=0.)
        loss, parts = trainer._compute_loss(outputs)
        torch.testing.assert_close(loss, parts['coherence_local'] + parts['coherence_global'])
        loss.backward()
        self.assertTrue(torch.isfinite(outputs[2]['local_descriptor'].grad).all())
        self.assertGreater(float(outputs[2]['local_descriptor'].grad.abs().sum()), 0.)
        self.assertEqual(int(torch.count_nonzero(outputs[2]['spectrum'].grad)), 0)


if __name__ == '__main__':
    unittest.main()
