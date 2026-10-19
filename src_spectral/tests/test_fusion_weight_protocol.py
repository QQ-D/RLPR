"""Guard the scientific comparison: isolated variables, honest selection, replay."""
import os
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from config.params_full import ModelParamsFull
from scripts.fusion_weight_ablation import fusion_environment, choose_configuration, verify_control


class FusionWeightProtocolTests(unittest.TestCase):
    def test_training_never_uses_the_held_out_validation_root(self):
        from scripts import fusion_weight_ablation as experiment
        initial = {'l': {'path': '/fixed/l5.pth'}, 'r': {'path': '/fixed/r5.pth'}}
        with tempfile.TemporaryDirectory() as tmp:
            for mode, split, expected in [('train', 'sejong', experiment.TRAIN),
                                          ('evaluate', 'sejong', experiment.VALIDATION),
                                          ('evaluate', 'dcc', experiment.TRAIN),
                                          ('evaluate', 'riverside', experiment.TRAIN)]:
                env = experiment.execution_environment(Path(tmp), 'fusion_01', initial, mode, split)
                with patch.dict(os.environ, env, clear=True):
                    params = ModelParamsFull()
                self.assertEqual(params.mulran_param.base_root, str(expected))
                if mode == 'train':
                    self.assertEqual(params.mulran_param.train_info_root[1],
                                     str(experiment.TRAIN / 'info/mulran_infos-kaist.pkl'))

    def test_requested_weight_reaches_trainer_without_inherited_experiment_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            initial = {'l': {'path': '/fixed/l5.pth'}, 'r': {'path': '/fixed/r5.pth'}}
            hostile = {'RLPR_WORD_AMPLITUDE': '0', 'RLPR_CROSS_FUSION_WEIGHT': '.9',
                       'RLPR_PHASE_LOSS_WEIGHT': '3', 'RLPR_RESUME': '1',
                       'RLPR_WORD_ALIGNMENT_LOSS': 'infonce', 'RLPR_CROSS_HARD_NEGATIVES': '0'}
            for arm, weight in [('fusion_05', .5), ('fusion_01', .1), ('fusion_00', 0.)]:
                with patch.dict(os.environ, hostile):
                    env = fusion_environment(Path(tmp), arm, initial)
                with patch.dict(os.environ, env, clear=True):
                    params = ModelParamsFull()
                self.assertEqual(params.cross_fusion_loss_weight, weight)
                self.assertEqual(params.phase_loss_weight, .1)
                self.assertEqual(params.word_alignment_loss, 'signed_coherency_v2')
                self.assertEqual(params.cross_hard_negative_count, 5)
                self.assertEqual(params.pretrain_l_path, '/fixed/l5.pth')
                self.assertEqual(params.pretrain_r_path, '/fixed/r5.pth')
                self.assertEqual(env['RLPR_RESUME'], '0')
                self.assertEqual(env['RLPR_WORD_AMPLITUDE'], '1')
                self.assertEqual(env['RLPR_SPECTRAL_CROSS_FUSION'], '1')
                self.assertEqual(params.training_root, str(Path(tmp) / arm))
            with self.assertRaises(ValueError):
                fusion_environment(Path(tmp), 'typo', initial)

    def test_validation_selection_ignores_test_scores_and_requires_every_configuration(self):
        rows = {
            'fusion_05': {'validation': {'Recall@1': 74., 'F1': .86}, 'benchmarks': {'riverside': {'Recall@1': 99.}}},
            'fusion_01': {'validation': {'Recall@1': 75., 'F1': .85}},
            'fusion_00': {'validation': {'Recall@1': 75., 'F1': .84}},
        }
        self.assertEqual(choose_configuration(rows), 'fusion_01')
        rows['fusion_00']['validation']['F1'] = .85
        self.assertEqual(choose_configuration(rows), 'fusion_01')
        with self.assertRaises(ValueError):
            choose_configuration({'fusion_05': rows['fusion_05']})
        rows['fusion_01']['validation']['Recall@1'] = float('nan')
        with self.assertRaises(ValueError):
            choose_configuration(rows)

    def test_control_replay_rejects_changed_or_missing_model_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, replay = Path(tmp) / 'ref.pt', Path(tmp) / 'replay.pt'
            original = {'weight': torch.tensor([1., 2.]), 'counter': torch.tensor(7)}
            torch.save({'state_dict': original}, reference)
            torch.save({'state_dict': {k: v.clone() for k, v in original.items()}}, replay)
            verify_control(replay, reference)
            changed = {**original, 'weight': torch.tensor([1., 2.001])}
            torch.save({'state_dict': changed}, replay)
            with self.assertRaisesRegex(ValueError, 'replay'):
                verify_control(replay, reference)
            torch.save({'state_dict': {'weight': original['weight']}}, replay)
            with self.assertRaisesRegex(ValueError, 'replay'):
                verify_control(replay, reference)
            torch.save({'state_dict': {'weight': torch.tensor([0.])}}, reference)
            torch.save({'state_dict': {'weight': torch.tensor([-0.])}}, replay)
            with self.assertRaisesRegex(ValueError, 'replay'):
                verify_control(replay, reference)

    def test_resume_recovers_a_published_selection_before_status_was_saved(self):
        from scripts import fusion_weight_ablation as experiment
        records = {a: {'validation': {'Recall@1': score, 'F1': .8}}
                   for a, score in [('fusion_05', 74.), ('fusion_01', 75.), ('fusion_00', 73.)]}
        with tempfile.TemporaryDirectory() as tmp:
            original = {'configurations': copy.deepcopy(records)}
            experiment.lock_selection(Path(tmp), original, 'validation only')
            published = (Path(tmp) / 'selection.json').read_bytes()
            interrupted = {'configurations': copy.deepcopy(records)}
            self.assertEqual(experiment.lock_selection(Path(tmp), interrupted, 'validation only'), 'fusion_01')
            self.assertEqual(interrupted['selection_sha256'], original['selection_sha256'])
            self.assertEqual((Path(tmp) / 'selection.json').read_bytes(), published)
            interrupted['configurations']['fusion_05']['validation']['Recall@1'] = 72.
            with self.assertRaises(ValueError):
                experiment.lock_selection(Path(tmp), interrupted, 'validation only')

    def test_resume_rejects_changed_checkpoint_before_reusing_cached_metrics(self):
        from scripts.fusion_weight_ablation import audit_trial
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / '1.pth.tar'
            checkpoint.write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError, 'Recorded checkpoint'):
                audit_trial(checkpoint, .1, '/fixed/teacher.pt', previous={'checkpoint_sha256': 'old_digest'})


if __name__ == '__main__':
    unittest.main()
