import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.amplitude_ablation import (experiment_settings, stage_environment, select_epoch,
                                        verify_training_validation_split, audit_checkpoint,
                                        stage_epoch_budget, choose_stage_epoch, write_json)


class AmplitudeProtocolTests(unittest.TestCase):
    def test_official_code_schedule_is_five_total_unary_epochs_then_one_alignment(self):
        settings = experiment_settings('official_code')
        self.assertEqual(settings['unimodal_epochs'], 5)
        self.assertEqual(settings['unimodal_additional_epochs'], 0)
        self.assertEqual(settings['cross_epochs'], 1)
        self.assertEqual(settings['unimodal_initialization'], 'resnet18_imagenet')
        for stage, expected in [('l', 5), ('r', 5), ('cross', 1)]:
            self.assertEqual(stage_epoch_budget(settings, stage), expected)
        rows = {str(e): {'validation': {'Recall@1': 100. - e, 'F1': .9}}
                for e in range(1, 6)}
        self.assertEqual(choose_stage_epoch(rows, settings, 'l'), 5)
        self.assertEqual(choose_stage_epoch(rows, settings, 'r'), 5)
        with self.assertRaises(ValueError):
            choose_stage_epoch({'1': rows['1']}, settings, 'l')
        self.assertEqual(choose_stage_epoch({'1': rows['1']}, settings, 'cross'), 1)
        old = experiment_settings()
        self.assertEqual(stage_epoch_budget(old, 'l'), 3)
        self.assertEqual(choose_stage_epoch(rows, old, 'l'), 1)
        with self.assertRaises(ValueError):
            experiment_settings('unknown')

    def test_failed_flush_leaves_previous_status_record_readable(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.json'
            path.write_text(json.dumps({'status': 'PAUSED'}))
            with patch('scripts.amplitude_ablation.os.fsync', side_effect=OSError('flush failed')):
                with self.assertRaises(OSError):
                    write_json(path, {'status': 'RUNNING'})
            self.assertEqual(json.loads(path.read_text()), {'status': 'PAUSED'})

    def test_equal_budget_and_explicit_modes_override_inherited_settings(self):
        settings = experiment_settings()
        self.assertEqual(settings['unimodal_additional_epochs'], 3)
        self.assertEqual(settings['cross_epochs'], 2)
        root = Path('/tmp/amplitude-protocol-test')
        pure = stage_environment(root, 'pure_phase', 'l', 1)
        amp = stage_environment(root, 'amplitude', 'l', 1)
        self.assertEqual(pure['RLPR_WORD_AMPLITUDE'], '0')
        self.assertEqual(amp['RLPR_WORD_AMPLITUDE'], '1')
        for name in ('RLPR_SAMPLING_SEED', 'RLPR_DETERMINISTIC', 'RLPR_CROSS_HARD_NEGATIVES',
                     'RLPR_PHASE_LOSS_WEIGHT', 'RLPR_CROSS_FUSION_WEIGHT', 'CUBLAS_WORKSPACE_CONFIG'):
            self.assertEqual(pure[name], amp[name])
        self.assertEqual(amp['RLPR_RESUME'], '0')
        second = stage_environment(root, 'amplitude', 'l', 2)
        self.assertEqual(second['RLPR_RESUME'], '1')
        self.assertEqual(second['RLPR_RESUME_EPOCH'], '1')
        with self.assertRaises(ValueError):
            stage_environment(root, 'oops', 'l', 1)

    def test_selection_uses_only_validation_and_fixed_tie_breaks(self):
        rows = {'1': {'validation': {'Recall@1': 70., 'F1': .8}, 'test': 100.},
                '2': {'validation': {'Recall@1': 71., 'F1': .82}, 'test': 0.},
                '3': {'validation': {'Recall@1': 71., 'F1': .81}, 'test': 100.}}
        self.assertEqual(select_epoch(rows, cross=True), 2)
        self.assertEqual(select_epoch(rows, cross=False), 2)
        rows['3']['validation']['F1'] = .83
        self.assertEqual(select_epoch(rows, cross=True), 3)
        self.assertEqual(select_epoch(rows, cross=False), 2)
        with self.assertRaises(ValueError):
            select_epoch({}, cross=False)

    def test_isolation_guard_includes_training_database(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            train, validation = Path(directory) / 'train', Path(directory) / 'validation'
            for folder in (train, validation):
                (folder / 'index').mkdir(parents=True)
            for name in ('sejong_test_db.npy', 'sejong_test_query.npy'):
                for folder in (train, validation):
                    np.save(folder / 'index' / name, [[9, 100., 0.]])
            np.save(train / 'index/sejong_train_query.npy', [[0, 0., 0.]])
            np.save(train / 'index/sejong_db.npy', [[1, 1., 0.]])
            self.assertEqual(verify_training_validation_split(train, validation), 99.)
            np.save(train / 'index/sejong_db.npy', [[9, 100., 0.]])
            with self.assertRaisesRegex(ValueError, 'overlap'):
                verify_training_validation_split(train, validation)

    def test_sampling_audit_is_recoverable_from_atomic_checkpoint_alone(self):
        import hashlib
        import torch
        generator = torch.Generator().manual_seed(0).get_state()
        saved = dict(epoch=1, model_variant='spectral_word', state_dict={'value': torch.ones(1)},
                     optimizer={'state': {0: {'step': torch.tensor(2071)}}}, scheduler={'last_epoch': 1},
                     training_state={'loader_generator': generator, 'sampler_generator': generator},
                     training_settings={'deterministic_algorithms': True})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'epoch.pth.tar'
            torch.save(saved, path)
            audit = audit_checkpoint(path, 1)
            expected = hashlib.sha256(generator.numpy().tobytes()).hexdigest()
            self.assertEqual(audit['sampling'], {'worker': expected, 'sampler': expected})


if __name__ == '__main__':
    unittest.main()
