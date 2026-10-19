"""Isolated geographic InfoNCE trainer; original trainer remains unchanged."""
from trainer.cross_modal import CrossModalTrainer
from dataset.MulRan.spatial_triplets import attach_positions
from modules.spatial_alignment import spatial_multi_infonce,spatial_masks


class SpatialCrossModalTrainer(CrossModalTrainer):
    def __init__(self,params,model_class,make_dataloader_fn,positive_mode):
        if positive_mode not in ('single','multi') or params.word_alignment_loss!='infonce':
            raise ValueError('Expected single/multi positives on the InfoNCE configuration')
        self.positive_mode=positive_mode
        super().__init__(params,model_class,make_dataloader_fn)
        if self.model.model_variant!='spectral_word':
            raise ValueError('This comparison requires the Word model')
        self.train_loader=attach_positions(self.train_loader)

    def _alignment_loss_name(self):
        return 'spatial_multi_infonce_v1' if self.positive_mode=='multi' else super()._alignment_loss_name()

    def _training_settings(self):
        settings=super()._training_settings()
        if self.positive_mode=='multi':
            settings.update(positive_radius_m=9.,negative_radius_m=12.,temperature=.07,
                            positive_reduction='log_probability_mass',negative_scope='within_original_12_frame_tuple')
        return settings

    def _compute_loss(self,outputs):
        original,parts=super()._compute_loss(outputs)
        if self.positive_mode=='single':return original,parts
        xy=self.train_loader.tuple_positions
        if xy is None or tuple(xy.shape)!=(self.params.batch_size,12,2):
            raise ValueError('Missing ordered training tuple positions')
        lidar,radar=outputs[2:]
        weight=self.params.cross_fusion_loss_weight
        terms=[]
        for kind in ('local','global'):
            name=kind+'_descriptor'
            teacher=radar[name].view(self.params.batch_size,12,-1)
            independent=spatial_multi_infonce(teacher,lidar[name].view(self.params.batch_size,12,-1),xy)
            combined=independent
            if 'fusion_phase' in lidar:
                fused=spatial_multi_infonce(teacher,lidar['fusion_'+name].view(self.params.batch_size,12,-1),xy)
                combined=(1-weight)*independent+weight*fused
                parts['infonce_'+kind+'_independent']=independent
                parts['infonce_'+kind+'_fused']=fused
            parts['infonce_'+kind]=combined
            terms.append(combined)
        positive,valid=spatial_masks(xy)
        parts['spatial_positive_count_per_anchor']=positive.sum(-1).float().mean()
        parts['spatial_ignored_count_per_anchor']=(~valid).sum(-1).float().mean()
        # The original phase/fusion forward paths and auxiliary terms are reused.
        return terms[0]+terms[1]+self.params.phase_loss_weight*parts['phase'],parts
