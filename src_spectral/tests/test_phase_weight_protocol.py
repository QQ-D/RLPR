"""Isolation and selection guards for the next phase-auxiliary comparison."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from config.params_full import ModelParamsFull
from scripts.phase_weight_ablation import execution_environment, choose_configuration, lock_selection, audit_trial
from scripts.amplitude_ablation import TRAIN, VALIDATION


class PhaseWeightProtocolTests(unittest.TestCase):
    def test_phase_weight_is_only_changed_training_setting_and_roots_stay_isolated(self):
        initial = {'l': {'path': '/fixed/l5.pt'}, 'r': {'path': '/fixed/r5.pt'}}
        with tempfile.TemporaryDirectory() as tmp:
            for arm, weight in [('phase_01', .1), ('phase_003', .03), ('phase_00', 0.)]:
                for mode, split, data in [('train', 'sejong', TRAIN), ('evaluate', 'sejong', VALIDATION),
                                           ('evaluate', 'dcc', TRAIN), ('evaluate', 'riverside', TRAIN)]:
                    with patch.dict(os.environ, {'RLPR_PHASE_LOSS_WEIGHT':'7', 'RLPR_CROSS_FUSION_WEIGHT':'.5',
                                                  'RLPR_WORD_AMPLITUDE':'0', 'RLPR_RESUME':'1'}):
                        env = execution_environment(Path(tmp), arm, initial, mode, split)
                    with patch.dict(os.environ, env, clear=True):
                        params = ModelParamsFull()
                    self.assertEqual(params.phase_loss_weight, weight)
                    self.assertEqual(params.cross_fusion_loss_weight, 0.)
                    self.assertEqual(params.cross_hard_negative_count, 5)
                    self.assertEqual(params.word_alignment_loss, 'signed_coherency_v2')
                    self.assertEqual(params.pretrain_l_path, '/fixed/l5.pt')
                    self.assertEqual(params.pretrain_r_path, '/fixed/r5.pt')
                    self.assertEqual(params.training_root, str(Path(tmp)/arm))
                    self.assertEqual(params.mulran_param.base_root, str(data))
                    self.assertEqual(env['RLPR_WORD_AMPLITUDE'], '1')
                    self.assertEqual(env['RLPR_SPECTRAL_CROSS_FUSION'], '1')
                    self.assertEqual(env['RLPR_RESUME'], '0')
            with self.assertRaises(ValueError):
                execution_environment(Path(tmp), 'typo', initial, 'train', 'sejong')

    def records(self):
        return {'phase_01': {'checkpoint_sha256':'a', 'validation':{'Recall@1':75., 'F1':.86}},
                'phase_003': {'checkpoint_sha256':'b', 'validation':{'Recall@1':76., 'F1':.87}},
                'phase_00': {'checkpoint_sha256':'c', 'validation':{'Recall@1':76., 'F1':.86}}}

    def test_selection_uses_validation_requires_every_arm_and_prefers_control_on_exact_ties(self):
        rows=self.records()
        rows['phase_01']['benchmarks']={'riverside':{'Recall@1':99.}}
        self.assertEqual(choose_configuration(rows), 'phase_003')
        with self.assertRaises(ValueError):
            choose_configuration({'phase_01':rows['phase_01']})
        rows['phase_01']['validation']={'Recall@1':76.,'F1':.87}
        self.assertEqual(choose_configuration(rows),'phase_01')
        rows['phase_00']['validation']['F1']=float('nan')
        with self.assertRaises(ValueError):
            choose_configuration(rows)

    def test_orphaned_lock_recovers_but_changed_checkpoint_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            report={'configurations':self.records()}
            self.assertEqual(lock_selection(Path(tmp),report,'validation only'),'phase_003')
            published=(Path(tmp)/'selection.json').read_bytes()
            resumed={'configurations':self.records()}
            lock_selection(Path(tmp),resumed,'validation only')
            self.assertEqual(report['selection_sha256'],resumed['selection_sha256'])
            self.assertEqual((Path(tmp)/'selection.json').read_bytes(),published)
            resumed['configurations']['phase_00']['checkpoint_sha256']='changed'
            with self.assertRaises(ValueError):
                lock_selection(Path(tmp),resumed,'validation only')

    def test_phase_checkpoint_audit_checks_actual_weights_and_rejects_changed_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, teacher = Path(tmp)/'1.pt',Path(tmp)/'teacher.pt'
            state={'weight_l':torch.tensor([1.]),'weight_r':torch.tensor([2.])}
            settings=dict(alignment_loss='signed_coherency_v2',batch_size=8,phase_loss_weight=.03,
                cross_fusion_loss_weight=0.,learning_rate_initial=5e-5,scheduler_gamma=.8,
                deterministic_algorithms=True,cublas_workspace_config=':4096:8',cross_hard_negative_count=5,sampling_seed=0)
            saved=dict(epoch=1,model_variant='spectral_word',state_dict=state,
                optimizer={'state':{0:{'step':torch.tensor(2071.),'exp_avg':torch.tensor([.1])}}},
                scheduler={'last_epoch':1},training_settings=settings,
                training_state={'sampler_generator':torch.Generator().manual_seed(0).get_state(),
                                'loader_generator':torch.Generator().manual_seed(0).get_state()},
                spectral_config={'pure_phase':False},cross_fusion_enabled=True)
            torch.save(saved,checkpoint);torch.save({'state_dict':{'weight_r':state['weight_r']}},teacher)
            audit=audit_trial(checkpoint,.03,teacher)
            self.assertEqual(audit['optimizer_steps'],2071)
            self.assertTrue(audit['radar_unchanged'])
            with self.assertRaisesRegex(ValueError,'settings'):
                audit_trial(checkpoint,.1,teacher)
            changed=copy.deepcopy(saved);changed['state_dict']['weight_l']+=1
            torch.save(changed,checkpoint)
            with self.assertRaisesRegex(ValueError,'Recorded checkpoint'):
                audit_trial(checkpoint,.03,teacher,previous=audit)


if __name__ == '__main__':
    unittest.main()
