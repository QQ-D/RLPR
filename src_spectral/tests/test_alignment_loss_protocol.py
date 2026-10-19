"""Objective choice isolation on the validation-selected amplitude configuration."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from config.params_full import ModelParamsFull
from scripts.amplitude_ablation import TRAIN, VALIDATION
from scripts.alignment_loss_ablation import execution_environment, choose_configuration, lock_selection, audit_trial


class AlignmentLossProtocolTests(unittest.TestCase):
    def test_only_objective_changes_and_training_never_uses_validation_root(self):
        initial={'l':{'path':'/fixed/l5.pt'},'r':{'path':'/fixed/r5.pt'}}
        with tempfile.TemporaryDirectory() as tmp:
            for arm,loss in [('signed','signed_coherency_v2'),('infonce','infonce')]:
                for mode,split,data in [('train','sejong',TRAIN),('evaluate','sejong',VALIDATION),
                                        ('evaluate','dcc',TRAIN),('evaluate','riverside',TRAIN)]:
                    with patch.dict(os.environ,{'RLPR_WORD_ALIGNMENT_LOSS':'typo','RLPR_PHASE_LOSS_WEIGHT':'0',
                                                 'RLPR_CROSS_FUSION_WEIGHT':'.5','RLPR_WORD_AMPLITUDE':'0','RLPR_RESUME':'1'}):
                        env=execution_environment(Path(tmp),arm,initial,mode,split)
                    with patch.dict(os.environ,env,clear=True):
                        params=ModelParamsFull()
                    self.assertEqual(params.word_alignment_loss,loss)
                    self.assertEqual(params.phase_loss_weight,.1)
                    self.assertEqual(params.cross_fusion_loss_weight,0.)
                    self.assertEqual(params.cross_hard_negative_count,5)
                    self.assertEqual(params.pretrain_l_path,'/fixed/l5.pt')
                    self.assertEqual(params.pretrain_r_path,'/fixed/r5.pt')
                    self.assertEqual(params.mulran_param.base_root,str(data))
                    self.assertEqual(params.training_root,str(Path(tmp)/arm))
                    self.assertEqual(env['RLPR_RESUME'],'0')
                    self.assertEqual(env['RLPR_WORD_AMPLITUDE'],'1')
                    self.assertEqual(env['RLPR_SPECTRAL_CROSS_FUSION'],'1')
            with self.assertRaises(ValueError):
                execution_environment(Path(tmp),'typo',initial,'train','sejong')

    def records(self):
        return {'signed':{'checkpoint_sha256':'a','validation':{'Recall@1':75.,'F1':.866}},
                'infonce':{'checkpoint_sha256':'b','validation':{'Recall@1':76.,'F1':.86}}}

    def test_both_validations_required_and_test_scores_cannot_select_the_model(self):
        rows=self.records();rows['signed']['benchmarks']={'riverside':{'Recall@1':99.}}
        self.assertEqual(choose_configuration(rows),'infonce')
        rows['signed']['validation']={'Recall@1':76.,'F1':.86}
        self.assertEqual(choose_configuration(rows),'signed')
        with self.assertRaises(ValueError):choose_configuration({'signed':rows['signed']})
        rows['infonce']['validation']['Recall@1']=float('nan')
        with self.assertRaises(ValueError):choose_configuration(rows)

    def test_published_lock_recovers_and_refuses_a_changed_validation_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            a={'configurations':self.records()}
            self.assertEqual(lock_selection(Path(tmp),a,'validation only'),'infonce')
            published=(Path(tmp)/'selection.json').read_bytes()
            b={'configurations':self.records()}
            lock_selection(Path(tmp),b,'validation only')
            self.assertEqual(a['selection_sha256'],b['selection_sha256'])
            self.assertEqual((Path(tmp)/'selection.json').read_bytes(),published)
            b['configurations']['signed']['validation']['F1']=.85
            with self.assertRaises(ValueError):lock_selection(Path(tmp),b,'validation only')

    def test_audit_validates_the_saved_objective_and_rejects_replaced_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint,teacher=Path(tmp)/'1.pt',Path(tmp)/'teacher.pt'
            state={'weight_l':torch.tensor([1.]),'weight_r':torch.tensor([2.])}
            settings=dict(alignment_loss='infonce',batch_size=8,phase_loss_weight=.1,cross_fusion_loss_weight=0.,
                          learning_rate_initial=5e-5,scheduler_gamma=.8,deterministic_algorithms=True,
                          cublas_workspace_config=':4096:8',cross_hard_negative_count=5,sampling_seed=0)
            saved=dict(epoch=1,model_variant='spectral_word',state_dict=state,
                       optimizer={'state':{0:{'step':torch.tensor(2071.),'exp_avg':torch.tensor([.1])}}},
                       scheduler={'last_epoch':1},training_settings=settings,
                       training_state={'sampler_generator':torch.Generator().manual_seed(0).get_state(),
                                       'loader_generator':torch.Generator().manual_seed(0).get_state()},
                       spectral_config={'pure_phase':False},cross_fusion_enabled=True)
            torch.save(saved,checkpoint);torch.save({'state_dict':{'weight_r':state['weight_r']}},teacher)
            row=audit_trial(checkpoint,'infonce',teacher)
            self.assertEqual(row['optimizer_steps'],2071);self.assertTrue(row['radar_unchanged'])
            with self.assertRaisesRegex(ValueError,'settings'):audit_trial(checkpoint,'signed_coherency_v2',teacher)
            other=copy.deepcopy(saved);other['state_dict']['weight_l']+=1
            torch.save(other,checkpoint)
            with self.assertRaisesRegex(ValueError,'Recorded checkpoint'):
                audit_trial(checkpoint,'infonce',teacher,previous=row)


if __name__ == '__main__':
    unittest.main()
