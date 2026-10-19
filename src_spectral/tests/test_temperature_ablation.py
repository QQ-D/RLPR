"""Temperature must affect the actual loss, with fixed auxiliaries and F1 selection."""
import importlib
import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F


class TemperatureTests(unittest.TestCase):
    def module(self,name):
        try:return importlib.import_module(name)
        except ModuleNotFoundError:self.fail('Temperature experiment is not implemented: '+name)

    def test_temperature_changes_real_local_global_loss_and_keeps_teacher_detached(self):
        m=self.module('trainer.temperature_cross_modal')
        from tests.test_alignment_ablation import AlignmentAblationTests
        fixture=AlignmentAblationTests();fixture.setUp()
        baseline,outputs=fixture.trainer_and_outputs()
        original,original_parts=baseline._compute_loss(outputs)
        trainer=m.TemperatureCrossModalTrainer.__new__(m.TemperatureCrossModalTrainer)
        trainer.__dict__.update(baseline.__dict__)
        trainer.configure_temperature(.07)
        control,parts=trainer._compute_loss(outputs)
        self.assertTrue(torch.equal(control,original))
        for value in (.03,.10):
            trainer.configure_temperature(value)
            loss,parts=trainer._compute_loss(outputs)
            expected=[]
            for key in ('local_descriptor','global_descriptor'):
                logits=F.normalize(outputs[3][key].detach(),dim=1)@F.normalize(outputs[2][key],dim=1).T/value
                expected.append(F.cross_entropy(logits,torch.arange(12)))
            torch.testing.assert_close(loss,sum(expected)+.1*original_parts['phase'])
            torch.testing.assert_close(parts['phase'],original_parts['phase'])
            self.assertFalse(torch.equal(loss,original))
        loss.backward()
        self.assertIsNone(outputs[3]['local_descriptor'].grad)
        self.assertGreater(float(outputs[2]['local_descriptor'].grad.abs().sum()),0.)
        for value in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):trainer.configure_temperature(value)

    def test_temperature_is_saved_in_actual_training_settings(self):
        m=self.module('trainer.temperature_cross_modal')
        trainer=m.TemperatureCrossModalTrainer.__new__(m.TemperatureCrossModalTrainer)
        trainer.model=SimpleNamespace(model_variant='spectral_word')
        trainer.params=SimpleNamespace(batch_size=8,phase_loss_weight=.1,cross_fusion_loss_weight=0.,
            learning_rate=5e-5,sampling_seed=2,word_alignment_loss='infonce')
        trainer.cross_hard_negative_count=5
        trainer.configure_temperature(.03)
        self.assertEqual(trainer._training_settings()['contrastive_temperature'],.03)
        trainer.configure_temperature(.10)
        self.assertEqual(trainer._training_settings()['contrastive_temperature'],.10)

    def test_all_nine_validations_are_required_and_test_scores_do_not_select(self):
        m=self.module('scripts.temperature_ablation')
        rows={a:{str(s):{'validation':{'F1':v,'Recall@1':70.}} for s in (0,1,2)}
              for a,v in [('tau_003',.88),('tau_007',.87),('tau_010',.86)]}
        rows['tau_010']['0']['benchmarks']={'riverside':{'F1':1.}}
        self.assertEqual(m.choose_temperature(rows),'tau_003')
        for row in rows['tau_007'].values():row['validation']['F1']=.88
        self.assertEqual(m.choose_temperature(rows),'tau_007')
        del rows['tau_010']['2']
        with self.assertRaises(ValueError):m.choose_temperature(rows)

    def test_checkpoint_audit_rejects_wrong_temperature_and_wrong_seed(self):
        m=self.module('scripts.temperature_ablation')
        import os
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'1.pt';teacher=Path(tmp)/'r.pt'
            state={'weight_l':torch.tensor([1.]),'weight_r':torch.tensor([2.])}
            settings=dict(alignment_loss='infonce',batch_size=8,phase_loss_weight=.1,cross_fusion_loss_weight=0.,
                learning_rate_initial=5e-5,scheduler_gamma=.8,deterministic_algorithms=True,cublas_workspace_config=':4096:8',
                cross_hard_negative_count=5,sampling_seed=2,contrastive_temperature=.03)
            saved=dict(epoch=1,model_variant='spectral_word',state_dict=state,alignment_loss='infonce',
                optimizer={'state':{0:{'step':torch.tensor(2071.),'exp_avg':torch.tensor([.1])}}},scheduler={'last_epoch':1},
                spectral_config={'pure_phase':False},cross_fusion_enabled=True,training_settings=settings,
                training_state={'sampler_generator':torch.Generator().manual_seed(2).get_state(),
                                'loader_generator':torch.Generator().manual_seed(2).get_state()})
            torch.save(saved,path);torch.save({'state_dict':{'weight_r':state['weight_r']}},teacher)
            self.assertEqual(m.audit_temperature(path,.03,2,teacher)['optimizer_steps'],2071)
            with self.assertRaises(ValueError):m.audit_temperature(path,.10,2,teacher)
            with self.assertRaises(ValueError):m.audit_temperature(path,.03,1,teacher)


if __name__=='__main__':unittest.main()
