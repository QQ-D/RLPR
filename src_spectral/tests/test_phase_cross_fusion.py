import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from modules.phase_cross import PhaseCrossAttention
from modules.spectral_models import SpectralCrossPolar, SpectralCrossCart, SpectralLidarPolar, SpectralRadarPolar
from modules.loss import InfoNCE
from trainer.cross_modal import CrossModalTrainer
from utils.pretrain import freeze_radar_branch_cross, load_unimodal_pretrained, validate_checkpoint_variant


class PhaseCrossFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)

    def test_other_modality_changes_output_and_teacher_has_no_gradient(self):
        layer = PhaseCrossAttention(8)
        lidar = torch.randn(2, 8, 5, 7, dtype=torch.complex64, requires_grad=True)
        radar = torch.randn_like(lidar, requires_grad=True)
        fused = layer(lidar, radar)
        changed = layer(lidar, radar * 2)
        self.assertEqual(fused.shape, lidar.shape)
        self.assertFalse(torch.allclose(fused, changed))
        fused.abs().square().mean().backward()
        self.assertIsNone(radar.grad)
        self.assertTrue(torch.isfinite(lidar.grad).all())
        self.assertGreater(float(layer.query.weight.grad.abs().sum()), 0)
        self.assertGreater(float(layer.value.weight.grad.abs().sum()), 0)

    def test_zero_context_leaves_student_unchanged(self):
        layer = PhaseCrossAttention(8)
        lidar = torch.randn(1, 8, 5, 7, dtype=torch.complex64)
        torch.testing.assert_close(layer(lidar, torch.zeros_like(lidar)), lidar)

    def test_training_fuses_both_geometries_but_evaluation_stays_independent(self):
        with patch.dict(os.environ, {'RLPR_SPECTRAL_CROSS_FUSION': '1'}):
            for model_class, shape in [(SpectralCrossPolar, (50, 225)), (SpectralCrossCart, (100, 100))]:
                model = model_class.create()
                freeze_radar_branch_cross(model)
                lidar = torch.rand(1, 2, 3, *shape)
                radar = torch.rand_like(lidar)
                model.train()
                dl, dr, al, ar = model(lidar, radar)
                self.assertIn('fusion_phase', al)
                self.assertEqual(al['fusion_local_descriptor'].shape, (2, 256))
                self.assertTrue(torch.isfinite(al['fusion_phase']).all())
                model.eval()
                with torch.no_grad():
                    first = model(lidar, radar)
                    other = model(lidar, radar * 2)
                self.assertNotIn('fusion_phase', first[2])
                self.assertTrue(torch.equal(first[0], other[0]))

    def test_old_unimodal_weights_load_without_freezing_new_fusion(self):
        with patch.dict(os.environ, {'RLPR_SPECTRAL_CROSS_FUSION': '1'}):
            cross = SpectralCrossPolar.create()
            models = [SpectralRadarPolar.create(), SpectralLidarPolar.create()]
        initial_fusion = cross.cross_fusion.query.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, model in enumerate(models):
                path = os.path.join(directory, str(index) + '.pth')
                torch.save({'state_dict': model.state_dict(), 'model_variant': 'spectral',
                            'spectral_config': model.spectral_config}, path)
                paths.append(path)
            load_unimodal_pretrained(cross, paths[0], paths[1])
        self.assertTrue(torch.equal(cross.cross_fusion.query.weight, initial_fusion))
        self.assertTrue(cross.cross_fusion.query.weight.requires_grad)
        self.assertFalse(cross.conv1_r.weight.requires_grad)
        cross.train()
        teacher_stats = cross.bn1_r.running_mean.clone()
        trainer = CrossModalTrainer.__new__(CrossModalTrainer)
        trainer.params = SimpleNamespace(batch_size=1, output_dim=512, phase_loss_weight=.1,
                                         cross_fusion_loss_weight=.5)
        trainer.info_nce = InfoNCE()
        x = torch.rand(1, 12, 3, 50, 225)
        loss, components = trainer._compute_loss(cross(x, x * .7))
        self.assertIn('infonce_local_fused', components)
        loss.backward()
        self.assertGreater(float(cross.cross_fusion.query.weight.grad.abs().sum()), 0)
        self.assertGreater(float(cross.cross_fusion.value.weight.grad.abs().sum()), 0)
        self.assertIsNone(cross.conv1_r.weight.grad)
        torch.testing.assert_close(cross.bn1_r.running_mean, teacher_stats)

    def test_full_checkpoint_requires_matching_fusion_mode(self):
        with patch.dict(os.environ, {'RLPR_SPECTRAL_CROSS_FUSION': '0'}):
            old = SpectralCrossPolar.create()
        with patch.dict(os.environ, {'RLPR_SPECTRAL_CROSS_FUSION': '1'}):
            new = SpectralCrossPolar.create()
        old_checkpoint = {'state_dict': old.state_dict(), 'model_variant': 'spectral',
                          'spectral_config': old.spectral_config}
        with self.assertRaisesRegex(ValueError, 'cross-fusion'):
            validate_checkpoint_variant(new, old_checkpoint)
        new_checkpoint = {'state_dict': new.state_dict(), 'model_variant': 'spectral',
                          'spectral_config': new.spectral_config, 'cross_fusion_enabled': True}
        with self.assertRaisesRegex(ValueError, 'cross-fusion'):
            validate_checkpoint_variant(old, new_checkpoint)


if __name__ == '__main__':
    unittest.main()
