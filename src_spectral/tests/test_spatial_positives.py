"""Guard geographic positives, original tuple sampling and detached teachers."""
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F


class SpatialPositiveTests(unittest.TestCase):
    def module(self,name):
        try:return importlib.import_module(name)
        except ModuleNotFoundError:self.fail('Spatial multi-positive implementation missing: '+name)

    def test_geographic_masks_respect_boundaries_and_reject_nonfinite_coordinates(self):
        m=self.module('modules.spatial_alignment')
        xy=torch.tensor([[[0.,0.],[9.,0.],[12.,0.],[13.,0.]]],dtype=torch.float64)
        positive,valid=m.spatial_masks(xy)
        self.assertEqual(positive[0,0].tolist(),[True,True,False,False])
        self.assertEqual(valid[0,0].tolist(),[True,True,False,True])
        self.assertTrue(torch.equal(positive,positive.transpose(1,2)))
        xy[0,0,0]=float('nan')
        with self.assertRaises(ValueError):m.spatial_masks(xy)

    def test_single_positive_matches_existing_infonce_and_teacher_is_detached(self):
        m=self.module('modules.spatial_alignment')
        from modules.loss import InfoNCE
        torch.manual_seed(37)
        teacher=torch.randn(2,4,8,requires_grad=True)
        student=torch.randn(2,4,8,requires_grad=True)
        xy=torch.tensor([[[0.,0.],[20.,0.],[40.,0.],[60.,0.]]]*2,dtype=torch.float64)
        loss=m.spatial_multi_infonce(teacher,student,xy)
        expected=sum(InfoNCE()(t.detach(),s) for t,s in zip(teacher,student))/2
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad.abs().sum()),0.)

    def test_positive_mass_excludes_ambiguous_pairs_and_all_positive_loss_is_zero(self):
        m=self.module('modules.spatial_alignment')
        teacher=torch.tensor([[[1.,0.],[1.,0.],[0.,1.],[-1.,0.]]])
        student=teacher.clone().requires_grad_()
        xy=torch.tensor([[[0.,0.],[2.,0.],[11.,0.],[30.,0.]]],dtype=torch.float64)
        loss=m.spatial_multi_infonce(teacher,student,xy)
        logits=F.normalize(teacher,dim=-1)@F.normalize(student,dim=-1).transpose(1,2)/.07
        # Hand-derived rows: positives {0,1}, {0,1,2}, {1,2}, {3}; excluded 0<->2.
        positives=[[0,1],[0,1,2],[1,2],[3]];valid=[[0,1,3],[0,1,2,3],[1,2,3],[0,1,2,3]]
        expected=sum(torch.logsumexp(logits[0,i,valid[i]],0)-torch.logsumexp(logits[0,i,positives[i]],0) for i in range(4))/4
        torch.testing.assert_close(loss,expected)
        zero=m.spatial_multi_infonce(teacher,student,torch.zeros(1,4,2))
        self.assertEqual(float(zero),0.)
        zero.backward();self.assertTrue(torch.isfinite(student.grad).all())

    def test_tuple_metadata_preserves_actual_sampling_and_collation_order(self):
        m=self.module('dataset.MulRan.spatial_triplets')
        from dataset.MulRan.mulran_dataset import TripletDataset,BaseDataset
        raw=TripletDataset.__new__(TripletDataset)
        raw.query_index_n_pos=np.array([[0.,0.,0.]])
        raw.database_index_n_pos=np.array([[0.,0.,0.],[1.,2.,0.]]+[[100.+i,20.+20*i,0.] for i in range(10)])
        raw.n_pos=1;raw.n_neg=10;raw.pos_index=[np.array([1])];raw.neg_index=[np.arange(2,12)]
        raw.augmentation=False;raw.l_enable=raw.r_enable=True;raw.is_batch_hard_mining=False
        def load(self,index,*args):return torch.full((3,2,3),float(index)),torch.full((3,2,3),float(index))
        positioned=m.SpatialTripletDataset.from_existing(raw)
        with patch.object(BaseDataset,'load_data',load):
            np.random.seed(81);original=raw[0];state=np.random.get_state()
            np.random.seed(81);item=positioned[0];actual=np.random.get_state()
        self.assertEqual(state[2:],actual[2:]);np.testing.assert_array_equal(state[1],actual[1])
        self.assertTrue(torch.equal(original['lidar_bev'],item['lidar_bev']))
        lookup={int(v[0]):v[1:] for v in raw.database_index_n_pos}
        ids=item['lidar_bev'][:,0,0,0].to(torch.int64).tolist()
        np.testing.assert_array_equal(item['tuple_positions'].numpy(),np.stack([lookup[i] for i in ids]))
        from dataset.collate import make_collate_fn
        batch=make_collate_fn()([item,item])
        self.assertEqual(tuple(batch['lidar_bev'].shape),(2,12,3,2,3))
        self.assertEqual(tuple(batch['tuple_positions'].shape),(2,12,2))

    def test_trainer_changes_alignment_only_and_keeps_phase_and_fusion_paths(self):
        m=self.module('trainer.spatial_cross_modal')
        from tests.test_alignment_ablation import AlignmentAblationTests
        from modules.spatial_alignment import spatial_multi_infonce
        fixture=AlignmentAblationTests();fixture.setUp()
        baseline,outputs=fixture.trainer_and_outputs()
        for name,value in list(outputs[2].items()):outputs[2]['fusion_'+name]=value.clone()
        old,parts=baseline._compute_loss(outputs)
        trainer=m.SpatialCrossModalTrainer.__new__(m.SpatialCrossModalTrainer)
        trainer.__dict__.update(baseline.__dict__);trainer.positive_mode='single'
        xy=torch.tensor([[[0.,0.],[2.,0.]]+[[20.*i,0.] for i in range(1,11)]],dtype=torch.float64)
        trainer.train_loader=SimpleNamespace(tuple_positions=xy)
        control,control_parts=trainer._compute_loss(outputs)
        self.assertTrue(torch.equal(old,control))
        trainer.positive_mode='multi'
        loss,new=trainer._compute_loss(outputs)
        expected=[]
        for kind in ('local','global'):
            key=kind+'_descriptor'
            independent=spatial_multi_infonce(outputs[3][key].view(1,12,-1),outputs[2][key].view(1,12,-1),xy)
            fused=spatial_multi_infonce(outputs[3][key].view(1,12,-1),outputs[2]['fusion_'+key].view(1,12,-1),xy)
            expected.append(.5*independent+.5*fused)
        torch.testing.assert_close(loss,sum(expected)+.1*parts['phase'])
        torch.testing.assert_close(new['phase'],parts['phase'])
        self.assertIn('infonce_local_fused',new)
        loss.backward()
        self.assertIsNone(outputs[3]['local_descriptor'].grad)
        self.assertGreater(float(outputs[2]['local_descriptor'].grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
