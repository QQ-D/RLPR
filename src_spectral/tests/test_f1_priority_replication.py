"""Guard F1-first selection, actual seed changes and official metric semantics."""
import copy
import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class F1PriorityTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('scripts.f1_priority_replication')
        except ModuleNotFoundError:
            self.fail('F1-priority replication is not implemented')

    def records(self):
        return {str(s): {
            'signed': {'validation': {'F1': .8+s*.01, 'Recall@1': 90.}},
            'infonce': {'validation': {'F1': .82+s*.01, 'Recall@1': 70.}}}
            for s in (0, 1, 2)}

    def test_selects_mean_validation_f1_without_using_test_or_best_seed(self):
        m=self.module(); rows=self.records()
        rows['0']['signed']['benchmarks']={'dcc': {'F1':1.}}
        selected,stats=m.choose_loss(rows)
        self.assertEqual(selected,'infonce')
        self.assertAlmostEqual(stats['signed']['F1']['mean'],.81)
        self.assertAlmostEqual(stats['infonce']['F1']['sd'],.01)
        rows['1']['signed']['validation']['F1']=1.
        self.assertEqual(m.choose_loss(rows)[0],'signed')
        for broken in ({'0':rows['0']}, dict(rows, **{'3':rows['0']})):
            with self.assertRaises(ValueError):m.choose_loss(broken)
        rows['2']['infonce']['validation']['F1']=float('nan')
        with self.assertRaises(ValueError):m.choose_loss(rows)

    def test_environment_pairs_true_seed_and_sampler_without_validation_training(self):
        m=self.module()
        from config.params_full import ModelParamsFull
        from scripts.amplitude_ablation import TRAIN,VALIDATION
        initial={'l':{'path':'/fixed/l.pt'},'r':{'path':'/fixed/r.pt'}}
        for seed in (1,2):
            for arm in ('signed','infonce'):
                for mode,data in [('train',TRAIN),('evaluate',VALIDATION)]:
                    env=m.environment(Path('/tmp/repl'),seed,arm,initial,mode,'sejong')
                    with patch.dict(os.environ,env,clear=True):p=ModelParamsFull()
                    self.assertEqual(p.sampling_seed,seed)
                    self.assertEqual(env['RLPR_EXPERIMENT_SEED'],str(seed))
                    self.assertEqual(p.mulran_param.base_root,str(data))
                    self.assertEqual(p.cross_fusion_loss_weight,0.)
                    self.assertEqual(p.phase_loss_weight,.1)
                    self.assertEqual(p.training_root,f'/tmp/repl/seed{seed}/{arm}')
        import torch
        for seed in (1,2):
            m.initialize_seed(seed); a=torch.rand(8)
            m.initialize_seed(seed); self.assertTrue(torch.equal(a,torch.rand(8)))
        m.initialize_seed(1);a=torch.rand(8)
        m.initialize_seed(2);self.assertFalse(torch.equal(a,torch.rand(8)))

    def test_pr_uses_original_conditional_recall_and_strict_threshold(self):
        m=self.module()
        # Two correct retrievals, one wrong, one query with no positive.
        d=np.array([.1,.2,.3,.4]); correct=np.array([True,False,True,False])
        at=m.threshold_counts(d,correct,.3)
        self.assertEqual(at['tp'],1);self.assertEqual(at['fp'],1)
        self.assertEqual(at['fn'],1);self.assertEqual(at['tn'],1)
        self.assertEqual(at['F1'],.5)
        best=m.legacy_curve(d,correct)
        self.assertAlmostEqual(best['F1'],.8)
        self.assertEqual(best['tp'],2);self.assertEqual(best['fp'],1)
        self.assertEqual(m.legacy_curve(d,np.zeros(4,dtype=bool))['F1'],0.)

    def test_selection_lock_and_bound_artifact_refuse_changed_evidence(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); report={'seeds':self.records()}
            self.assertEqual(m.lock_selection(root,report),'infonce')
            before=(root/'selection.json').read_bytes()
            recovered={'seeds':self.records()}
            m.lock_selection(root,recovered)
            self.assertEqual(before,(root/'selection.json').read_bytes())
            report['seeds']['1']['signed']['validation']['F1']=.99
            with self.assertRaises(ValueError):m.lock_selection(root,report)
            path=root/'metrics.json';path.write_text('{}')
            artifact=m.bind_artifact(path)
            m.verify_artifact(artifact)
            path.write_text('{"F1":1}')
            with self.assertRaises(ValueError):m.verify_artifact(artifact)

    def test_audit_rejects_wrong_seed_and_unbound_seed_provenance(self):
        m=self.module()
        import hashlib
        import json
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);path=folder/'1.pt';teacher=folder/'teacher.pt'
            state={'weight_l':torch.tensor([1.]),'weight_r':torch.tensor([2.])}
            saved=dict(epoch=1,model_variant='spectral_word',state_dict=state,
                optimizer={'state':{0:{'step':torch.tensor(2071.),'exp_avg':torch.tensor([.1])}}},
                scheduler={'last_epoch':1},spectral_config={'pure_phase':False},cross_fusion_enabled=True,
                training_settings=dict(alignment_loss='infonce',batch_size=8,phase_loss_weight=.1,cross_fusion_loss_weight=0.,
                    learning_rate_initial=5e-5,scheduler_gamma=.8,deterministic_algorithms=True,
                    cublas_workspace_config=':4096:8',cross_hard_negative_count=5,sampling_seed=1),
                training_state={'sampler_generator':torch.Generator().manual_seed(1).get_state(),
                                'loader_generator':torch.Generator().manual_seed(1).get_state()})
            torch.save(saved,path);torch.save({'state_dict':{'weight_r':state['weight_r']}},teacher)
            self.assertEqual(m.audit_trial(path,1,'infonce',teacher)['optimizer_steps'],2071)
            with self.assertRaises(ValueError):m.audit_trial(path,2,'infonce',teacher)
            seedfile=folder/'seed_start.json'
            seedfile.write_text(json.dumps(dict(seed=1,sampling_seed=1,
                torch_rng_sha256=hashlib.sha256(torch.Generator().manual_seed(1).get_state().numpy().tobytes()).hexdigest())))
            self.assertTrue(hasattr(m,'audit_seed_start'),'Missing true-seed provenance audit')
            m.audit_seed_start(folder,1)
            with self.assertRaises(ValueError):m.audit_seed_start(folder,2)

    def test_cached_metrics_must_match_bound_json_and_checkpoint(self):
        m=self.module()
        import json
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);checkpoint=folder/'weights.pt';checkpoint.write_bytes(b'weights')
            metrics=folder/'sejong.json';metrics.write_text(json.dumps({'F1':.8,'Recall@1':70.}))
            diag=folder/'sejong_diagnostics.json'
            diag.write_text(json.dumps({'checkpoint':m.bind_artifact(checkpoint)}))
            row={'checkpoint':str(checkpoint),'checkpoint_sha256':m.digest(checkpoint),
                 'validation':{'F1':.8,'Recall@1':70.},
                 'artifacts':{'sejong_metrics':m.bind_artifact(metrics),
                              'sejong_diagnostics.json':m.bind_artifact(diag)}}
            self.assertTrue(hasattr(m,'verify_metrics'),'Missing cached metric-to-checkpoint binding')
            m.verify_metrics(row)
            row['validation']['F1']=.9
            with self.assertRaises(ValueError):m.verify_metrics(row)
            row['validation']['F1']=.8
            diag.write_text(json.dumps({'checkpoint':{'path':str(checkpoint),'sha256':'wrong'}}))
            row['artifacts']['sejong_diagnostics.json']=m.bind_artifact(diag)
            with self.assertRaises(ValueError):m.verify_metrics(row)


if __name__=='__main__':unittest.main()
