"""Guard F1-only screening and frozen validation-based family selection."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest


class SpatialProtocolTests(unittest.TestCase):
    def module(self):
        try:return importlib.import_module('scripts.spatial_positive_ablation')
        except ModuleNotFoundError:self.fail('Spatial experiment driver is missing')

    def test_screening_and_selection_never_use_test_scores_or_best_seed(self):
        m=self.module()
        base={'F1':.87,'Recall@1':75.};candidate={'F1':.86,'Recall@1':90.}
        self.assertFalse(m.promote(candidate,base))
        self.assertTrue(m.promote({'F1':.871,'Recall@1':70.},base))
        baseline={str(s):{'validation':dict(base),'benchmarks':{'dcc':{'F1':1.}}} for s in (0,1,2)}
        trials={str(s):{'validation':{'F1':v,'Recall@1':70.}} for s,v in enumerate((.88,.86,.88))}
        self.assertEqual(m.select_family(baseline,trials,[0,1,2]),'multi')
        self.assertEqual(m.select_family(baseline,{'0':{'validation':candidate}},[0]),'single')
        trials['1']['validation']['F1']=float('nan')
        with self.assertRaises(ValueError):m.select_family(baseline,trials,[0,1,2])
        with self.assertRaises(ValueError):m.select_family(baseline,{'0':trials['0']},[0,1,2])

    def test_published_lock_recovers_and_changed_validation_is_rejected(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'selection.json';payload={'selected':'multi','F1':.87}
            sha=m.lock_payload(path,payload)
            self.assertEqual(m.lock_payload(path,payload),sha)
            with self.assertRaises(ValueError):m.lock_payload(path,dict(payload,F1=.88),sha)
            path.unlink()
            with self.assertRaises(ValueError):m.lock_payload(path,payload,sha)

    def test_candidate_checkpoint_audit_records_geographic_objective_and_seed(self):
        m=self.module()
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'1.pt';teacher=Path(tmp)/'teacher.pt'
            state={'weight_l':torch.tensor([1.]),'weight_r':torch.tensor([2.])}
            settings=dict(alignment_loss='spatial_multi_infonce_v1',batch_size=8,phase_loss_weight=.1,cross_fusion_loss_weight=0.,
                learning_rate_initial=5e-5,scheduler_gamma=.8,deterministic_algorithms=True,cublas_workspace_config=':4096:8',
                cross_hard_negative_count=5,sampling_seed=2,positive_radius_m=9.,negative_radius_m=12.,temperature=.07,
                positive_reduction='log_probability_mass',negative_scope='within_original_12_frame_tuple')
            saved=dict(epoch=1,model_variant='spectral_word',alignment_loss='spatial_multi_infonce_v1',
                state_dict=state,optimizer={'state':{0:{'step':torch.tensor(2071.),'exp_avg':torch.tensor([.1])}}},
                scheduler={'last_epoch':1},training_settings=settings,spectral_config={'pure_phase':False},cross_fusion_enabled=True,
                training_state={'sampler_generator':torch.Generator().manual_seed(2).get_state(),
                                'loader_generator':torch.Generator().manual_seed(2).get_state()})
            torch.save(saved,path);torch.save({'state_dict':{'weight_r':state['weight_r']}},teacher)
            self.assertEqual(m.audit_candidate(path,2,teacher)['optimizer_steps'],2071)
            with self.assertRaises(ValueError):m.audit_candidate(path,1,teacher)
            saved['training_settings']['positive_radius_m']=12.
            torch.save(saved,path)
            with self.assertRaises(ValueError):m.audit_candidate(path,2,teacher)

    def test_report_uses_frozen_official_comparison_instead_of_mutable_history(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'history.json';target=root/'run';target.mkdir()
            source.write_text('{"F1":0.8057}')
            self.assertTrue(hasattr(m,'freeze_official'),'Official comparison snapshot is missing')
            frozen=m.freeze_official(target,source)
            source.write_text('{"F1":1.0}')
            m.verify_artifact(frozen)
            self.assertEqual(json.loads(Path(frozen['path']).read_text())['F1'],.8057)
            Path(frozen['path']).write_text('{"F1":1.0}')
            with self.assertRaises(ValueError):m.verify_artifact(frozen)


if __name__=='__main__':unittest.main()
