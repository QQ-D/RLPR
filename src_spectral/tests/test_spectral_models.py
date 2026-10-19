import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from modules.spectral_models import SpectralCrossPolar, SpectralCrossCart, SpectralLidarPolar, SpectralRadarPolar
from modules.loss import phase_coherence_loss
from utils.pretrain import load_unimodal_pretrained, freeze_radar_branch_cross
from trainer.cross_modal import CrossModalTrainer
from modules.loss import InfoNCE


class SpectralModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_polar_and_4d_shapes_and_independent_descriptors(self):
        for model_class, shape in [(SpectralCrossPolar, (50, 225)), (SpectralCrossCart, (100, 100))]:
            model = model_class.create().eval()
            lidar = torch.randn(1, 2, 3, *shape)
            radar = torch.randn_like(lidar)
            with torch.no_grad():
                dl, dr, al, ar = model(lidar, radar)
                dl_other, _, _, _ = model(lidar, radar * 2)
            self.assertEqual(dl.shape, (2, 512))
            self.assertEqual(dr.shape, (2, 512))
            self.assertTrue(torch.equal(dl, dl_other))
            self.assertEqual(al['local_descriptor'].shape, (2, 256))
            self.assertEqual(al['phase'].shape, ar['phase'].shape)
            self.assertTrue(torch.isfinite(dl).all())

    def test_unimodal_loading_and_frozen_teacher_stats(self):
        lidar = SpectralLidarPolar.create()
        radar = SpectralRadarPolar.create()
        cross = SpectralCrossPolar.create()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, model in [('l', lidar), ('r', radar)]:
                path = os.path.join(directory, name + '.pth.tar')
                torch.save({'state_dict': model.state_dict(), 'model_variant': 'spectral',
                            'spectral_config': model.spectral_config}, path)
                paths.append(path)
            load_unimodal_pretrained(cross, paths[1], paths[0])
        self.assertTrue(torch.equal(cross.spectral_r.cutoff_logit, radar.spectral_r.cutoff_logit))
        self.assertFalse(cross.spectral_r.cutoff_logit.requires_grad)
        teacher_before = {k: v.clone() for k, v in cross.state_dict().items() if k.split('.')[0].endswith('_r')}
        cross.train()
        self.assertFalse(cross.bn1_r.training)
        self.assertFalse(cross.layer1_r.training)
        x = torch.randn(1, 2, 3, 50, 225)
        dl, dr, al, ar = cross(x, x * .7)
        weights = torch.minimum(al['confidence'], ar['confidence'])
        loss = (dl - dr.detach()).square().mean() + .1 * phase_coherence_loss(ar['phase'], al['phase'], weights)
        loss.backward()
        self.assertTrue(torch.isfinite(cross.conv1_l.weight.grad).all())
        self.assertGreater(float(cross.spectral_l.cutoff_logit.grad.abs().sum()), 0)
        self.assertIsNone(cross.conv1_r.weight.grad)
        for key, before in teacher_before.items():
            self.assertTrue(torch.equal(cross.state_dict()[key], before), key)

    def test_old_checkpoint_cannot_silently_supply_untrained_spectral_teacher(self):
        model = SpectralRadarPolar.create()
        old_state = {k: v for k, v in model.state_dict().items() if not k.startswith('spectral_')}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'old.pth.tar')
            torch.save({'state_dict': old_state}, path)
            with self.assertRaisesRegex(ValueError, 'variant'):
                load_unimodal_pretrained(SpectralCrossPolar.create(), path, path)

    def test_freeze_radar_includes_netvlad_on_resume(self):
        cross = SpectralCrossPolar.create()
        freeze_radar_branch_cross(cross)
        for name, parameter in cross.named_parameters():
            module = name.split('.')[0]
            radar_module = module.endswith('_r') or module.startswith('net_vlad_r_')
            if radar_module:
                self.assertFalse(parameter.requires_grad, name)
        self.assertTrue(cross.spectral_l.cutoff_logit.requires_grad)

    def test_incomplete_teacher_checkpoint_is_rejected_before_loading(self):
        cross = SpectralCrossPolar.create()
        radar = SpectralRadarPolar.create()
        lidar = SpectralLidarPolar.create()
        before = cross.conv1_r.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            r_path, l_path = os.path.join(directory, 'r.pth'), os.path.join(directory, 'l.pth')
            torch.save({'state_dict': {'spectral_r.cutoff_logit': radar.spectral_r.cutoff_logit},
                        'model_variant': 'spectral', 'spectral_config': cross.spectral_config}, r_path)
            torch.save({'state_dict': lidar.state_dict(), 'model_variant': 'spectral',
                        'spectral_config': cross.spectral_config}, l_path)
            with self.assertRaisesRegex(ValueError, 'Incomplete radar'):
                load_unimodal_pretrained(cross, r_path, l_path)
        self.assertTrue(torch.equal(cross.conv1_r.weight, before))
        self.assertTrue(cross.conv1_r.weight.requires_grad)

    def test_cross_checkpoint_cannot_be_used_as_unimodal_teacher(self):
        cross = SpectralCrossPolar.create()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'cross.pth')
            torch.save({'state_dict': cross.state_dict(), 'model_variant': 'spectral',
                        'spectral_config': cross.spectral_config}, path)
            with self.assertRaisesRegex(ValueError, 'unimodal'):
                load_unimodal_pretrained(cross, path, path)
        self.assertTrue(cross.conv1_l.weight.requires_grad)

    def test_wrong_geometry_checkpoint_is_rejected_before_loading(self):
        cross = SpectralCrossCart.create()
        radar, lidar = SpectralRadarPolar.create(), SpectralLidarPolar.create()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, model in [('r', radar), ('l', lidar)]:
                path = os.path.join(directory, name + '.pth')
                torch.save({'state_dict': model.state_dict(), 'model_variant': 'spectral',
                            'spectral_config': model.spectral_config}, path)
                paths.append(path)
            with self.assertRaisesRegex(ValueError, 'shape'):
                load_unimodal_pretrained(cross, paths[0], paths[1])

    def test_trainer_uses_local_global_and_phase_losses(self):
        trainer = CrossModalTrainer.__new__(CrossModalTrainer)
        trainer.params = SimpleNamespace(batch_size=1, output_dim=512, phase_loss_weight=.1)
        trainer.info_nce = InfoNCE()
        lidar = torch.randn(12, 512, requires_grad=True)
        radar = torch.randn(12, 512)
        phase_l = torch.ones(12, 8, 3, 4, requires_grad=True)
        phase_r = torch.zeros_like(phase_l)
        al = {'local_descriptor': lidar[:, 256:], 'global_descriptor': lidar[:, :256],
              'phase': phase_l, 'confidence': torch.ones_like(phase_l)}
        ar = {'local_descriptor': radar[:, 256:], 'global_descriptor': radar[:, :256],
              'phase': phase_r, 'confidence': torch.ones_like(phase_r)}
        total, parts = trainer._compute_loss((lidar, radar, al, ar))
        expected = parts['infonce_local'] + parts['infonce_global'] + .1 * parts['phase']
        self.assertTrue(torch.equal(total, expected))
        total.backward()
        self.assertGreater(float(phase_l.grad.abs().sum()), 0)

    def test_original_trainer_loss_is_unchanged(self):
        trainer = CrossModalTrainer.__new__(CrossModalTrainer)
        trainer.params = SimpleNamespace(batch_size=1, output_dim=512)
        trainer.info_nce = InfoNCE()
        lidar = torch.randn(12, 512)
        radar = torch.randn(12, 512)
        total, parts = trainer._compute_loss((lidar, radar, None, None))
        self.assertTrue(torch.equal(total, trainer.info_nce(radar, lidar)))


if __name__ == '__main__':
    unittest.main()
