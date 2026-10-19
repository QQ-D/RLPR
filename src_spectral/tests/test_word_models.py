import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from modules.word_models import WordCrossPolar, WordCrossCart, WordLidarPolar, WordRadarPolar
from modules.spectral_models import SpectralRadarPolar
from trainer.cross_modal import CrossModalTrainer
from trainer.eval import CrossModalTester
from utils.pretrain import freeze_radar_branch_cross, load_unimodal_pretrained, validate_checkpoint_variant


class WordModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)

    def test_both_geometries_use_phase_encoder_and_test_descriptors_are_independent(self):
        with patch.dict(os.environ, {'RLPR_SPECTRAL_CROSS_FUSION': '1'}):
            for cls, shape in [(WordCrossPolar, (50, 225)), (WordCrossCart, (100, 100))]:
                model = cls.create().eval()
                self.assertEqual(model.model_variant, 'spectral_word')
                self.assertNotIn('encoder_layer_l', dict(model.named_children()))
                calls = []
                h = model.encoder_l.register_forward_pre_hook(lambda m, args: calls.append(args[0].shape))
                l = torch.rand(1, 2, 3, *shape); r = torch.rand_like(l)
                with torch.no_grad():
                    first = model(l, r); other = model(l, r * .3)
                    reverse = model(l * .3, r)
                h.remove()
                self.assertEqual(first[0].shape, (2, 512))
                self.assertTrue(torch.equal(first[0], other[0]))
                self.assertTrue(torch.equal(first[1], reverse[1]))
                self.assertEqual(calls[0][-1], 128)
                self.assertNotIn('fusion_phase', first[2])

    def test_word_teachers_load_and_coherence_trains_all_student_modules(self):
        with patch.dict(os.environ, {'RLPR_SPECTRAL_CROSS_FUSION': '1'}):
            radar, lidar, cross = WordRadarPolar.create(), WordLidarPolar.create(), WordCrossPolar.create()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, model in [('r', radar), ('l', lidar)]:
                path = os.path.join(directory, name + '.pth')
                torch.save({'state_dict': model.state_dict(), 'model_variant': model.model_variant,
                            'spectral_config': model.spectral_config}, path)
                paths.append(path)
            load_unimodal_pretrained(cross, *paths)
        before = {k: v.clone() for k, v in cross.state_dict().items()
                  if k.split('.')[0].endswith('_r') or k.startswith('net_vlad_r_')}
        cross.train()
        trainer = CrossModalTrainer.__new__(CrossModalTrainer)
        trainer.model = cross
        trainer.params = SimpleNamespace(batch_size=1, phase_loss_weight=.1, cross_fusion_loss_weight=.5)
        trainer.info_nce = lambda *args: self.fail('Word mode must not call InfoNCE')
        x = torch.rand(1, 12, 3, 50, 225)
        total, parts = trainer._compute_loss(cross(x, torch.rand_like(x)))
        self.assertIn('coherence_global_fused', parts)
        self.assertNotIn('infonce_local', parts)
        torch.testing.assert_close(total, parts['coherence_local'] + parts['coherence_global'] + .1 * parts['phase'])
        total.backward()
        for p in [cross.spectral_l.cutoff_logit, cross.encoder_l.layers[0].self_attn.in_proj_weight,
                  cross.cross_fusion.query.weight, cross.linear_l.weight,
                  cross.net_vlad_l_g.cluster_weights]:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0)
        for name, p in cross.named_parameters():
            if name.split('.')[0].endswith('_r') or name.startswith('net_vlad_r_'):
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)
        for k, v in before.items():
            self.assertTrue(torch.equal(v, cross.state_dict()[k]), k)

    def test_prior_teacher_is_rejected_instead_of_frozen_untrained_word_modules(self):
        old = SpectralRadarPolar.create()
        checkpoint = {'state_dict': old.state_dict(), 'model_variant': 'spectral',
                      'spectral_config': old.spectral_config}
        with self.assertRaisesRegex(ValueError, 'variant'):
            validate_checkpoint_variant(WordCrossPolar.create(), checkpoint)

    def test_word_checkpoint_roundtrip_and_configuration_validation(self):
        with patch.dict(os.environ, {'RLPR_SPECTRAL_CROSS_FUSION': '1'}):
            model = WordCrossPolar.create().eval()
            target = WordCrossPolar.create().eval()
        checkpoint = {'state_dict': model.state_dict(), 'model_variant': model.model_variant,
                      'spectral_config': model.spectral_config, 'cross_fusion_enabled': True}
        validate_checkpoint_variant(target, checkpoint)
        target.load_state_dict(checkpoint['state_dict'])
        x = torch.rand(1, 1, 3, 50, 225)
        with torch.no_grad():
            torch.testing.assert_close(model(x, x)[0], target(x, x)[0], atol=0, rtol=0)
        bad = dict(checkpoint, spectral_config={'pure_phase': False})
        with self.assertRaisesRegex(ValueError, 'spectral_config'):
            validate_checkpoint_variant(target, bad)
        with patch.dict(os.environ, {'RLPR_SPECTRAL_CROSS_FUSION': '0'}):
            with self.assertRaisesRegex(ValueError, 'cross-fusion'):
                validate_checkpoint_variant(WordCrossPolar.create(), checkpoint)

    def test_evaluation_restores_frozen_teacher_after_loading_checkpoint(self):
        with patch.dict(os.environ, {'RLPR_SPECTRAL_CROSS_FUSION': '1'}):
            source, target = WordCrossPolar.create(), WordCrossPolar.create()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'cross.pth')
            torch.save({'state_dict': source.state_dict(), 'model_variant': source.model_variant,
                        'spectral_config': source.spectral_config, 'cross_fusion_enabled': True}, path)
            tester = CrossModalTester.__new__(CrossModalTester)
            tester.model = target
            tester.params = SimpleNamespace(resume_checkpoint=True, checkpoint_path=path)
            tester._load_checkpoint()
        self.assertFalse(target.encoder_r.layers[0].self_attn.in_proj_weight.requires_grad)
        self.assertTrue(target.encoder_l.layers[0].self_attn.in_proj_weight.requires_grad)


if __name__ == '__main__':
    unittest.main()
