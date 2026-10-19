import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from modules.loss import InfoNCE, phase_coherence_loss
from modules.coherence import coherence_alignment_loss, spectral_coherence_loss
from utils.pretrain import (
    freeze_params_named_like_r_checkpoint,
    freeze_radar_branch_cross,
    load_snail_crossmodal_init,
    load_unimodal_pretrained,
    validate_checkpoint_variant,
)
from utils.training_state import (configure_sampling, capture_training_state,
                                  restore_training_state, validate_training_settings, require_new_epochs,
                                  save_checkpoint_atomic)
from utils.cross_mining import make_cross_modal_cache_loader, update_cross_modal_cache


class CrossModalTrainer:
    def __init__(self, params, model_class, make_dataloader_fn, hercules_mode=False):
        self.params = params
        self.hercules_mode = hercules_mode

        self.weights_root = params.weights_root
        self.logs_root = params.logs_root
        self.cache_root = params.cache_root
        self.cross_hard_negative_count = getattr(params, 'cross_hard_negative_count', 0)
        self.resume = params.resume_checkpoint
        self.resume_epoch = params.resume_epoch

        self.writer = SummaryWriter(self.logs_root)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class.create(weights=ResNet18_Weights.DEFAULT)
        n_params = sum(p.nelement() for p in self.model.parameters())
        print(f"Number of model parameters: {n_params}")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), params.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.8)
        self.info_nce = InfoNCE()

        loaders = make_dataloader_fn("train", params)
        if len(loaders) == 5:
            self.train_loader, _, self.train_query_loader, _, _ = loaders
        else:
            self.train_loader, self.train_query_loader = loaders

        configure_sampling(self.train_loader, getattr(params, 'sampling_seed', None))
        if self.cross_hard_negative_count:
            if self.train_loader.dataset.__class__.__module__ != 'dataset.MulRan.mulran_dataset':
                raise ValueError('Cross-modal hard-negative sampling is currently supported for MulRan only')
            self.mining_loader = make_cross_modal_cache_loader(self.train_query_loader)

        self.total_iter = len(self.train_loader)

    def _freeze_radar(self):
        if self.hercules_mode or getattr(self.model, 'model_variant', 'rlpr') in ('spectral', 'spectral_word'):
            freeze_radar_branch_cross(self.model)
        else:
            freeze_params_named_like_r_checkpoint(self.model, self.params.pretrain_r_path)

    def _init_weights(self):
        if self.hercules_mode:
            load_snail_crossmodal_init(self.model, self.params.snail_crossmodal_path)
            if self.resume:
                resume_path = os.path.join(self.weights_root, f"{self.resume_epoch}.pth.tar")
                if os.path.exists(resume_path):
                    print(f"\nResuming Hercules from {resume_path}")
                    checkpoint = torch.load(resume_path)
                    validate_checkpoint_variant(self.model, checkpoint)
                    validate_training_settings(checkpoint, self._training_settings())
                    self.model.load_state_dict(checkpoint["state_dict"])
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                    self.scheduler.load_state_dict(checkpoint["scheduler"])
                    self._freeze_radar()
                    restore_training_state(checkpoint.get('training_state'), self.train_loader)
                    return checkpoint["epoch"]
            print("\nStarting Hercules fine-tuning from Snail cross-modal checkpoint")
            self._freeze_radar()
            return 0

        if self.resume:
            resume_filename = os.path.join(self.weights_root, f"{self.resume_epoch}.pth.tar")
            print(f"\nResuming from {resume_filename}")
            checkpoint = torch.load(resume_filename)
            validate_checkpoint_variant(self.model, checkpoint)
            validate_training_settings(checkpoint, self._training_settings())
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self._freeze_radar()
            restore_training_state(checkpoint.get('training_state'), self.train_loader)
            return checkpoint["epoch"]

        print("\nTraining From Scratch")
        load_unimodal_pretrained(self.model, self.params.pretrain_r_path, self.params.pretrain_l_path)
        return 0

    def _alignment_loss_name(self):
        if getattr(getattr(self, 'model', None), 'model_variant', 'rlpr') == 'spectral_word':
            return getattr(self.params, 'word_alignment_loss', 'signed_coherency_v2')
        return 'infonce'

    def _training_settings(self):
        return dict(alignment_loss=self._alignment_loss_name(), batch_size=self.params.batch_size,
                    phase_loss_weight=self.params.phase_loss_weight,
                    cross_fusion_loss_weight=self.params.cross_fusion_loss_weight,
                    learning_rate_initial=self.params.learning_rate, scheduler_gamma=.8,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                    cross_hard_negative_count=getattr(self, 'cross_hard_negative_count', 0),
                    sampling_seed=getattr(self.params, 'sampling_seed', None))

    def _compute_loss(self, outputs):
        descriptors_l, descriptors_r, aux_l, aux_r = outputs
        word_mode = getattr(getattr(self, 'model', None), 'model_variant', 'rlpr') == 'spectral_word'
        use_coherence = self._alignment_loss_name() == 'signed_coherency_v2'
        def contrastive(lidar, radar):
            lidar = lidar.view(self.params.batch_size, 12, -1)
            radar = radar.view(self.params.batch_size, 12, -1)
            if use_coherence:
                return coherence_alignment_loss(lidar, radar)
            if word_mode:
                radar = radar.detach()
            return sum(self.info_nce(r, l) for l, r in zip(lidar, radar)) / self.params.batch_size
        loss_prefix = 'coherence' if use_coherence else 'infonce'
        if isinstance(aux_l, dict) and 'phase' in aux_l:
            local = contrastive(aux_l['local_descriptor'], aux_r['local_descriptor'])
            global_loss = contrastive(aux_l['global_descriptor'], aux_r['global_descriptor'])
            confidence = torch.minimum(aux_l['confidence'], aux_r['confidence'])
            phase = phase_coherence_loss(aux_r['phase'], aux_l['phase'], confidence)
            components = {loss_prefix + '_local': local, loss_prefix + '_global': global_loss, 'phase': phase}
            if word_mode:
                statistical = spectral_coherence_loss(aux_r['spectrum'], aux_l['spectrum'], confidence)
                components.update(phase_circular_independent=phase, spectral_coherence_independent=statistical)
                phase = phase + statistical
                components['phase'] = phase
            if 'fusion_phase' in aux_l:
                weight = self.params.cross_fusion_loss_weight
                fused_local = contrastive(aux_l['fusion_local_descriptor'], aux_r['local_descriptor'])
                fused_global = contrastive(aux_l['fusion_global_descriptor'], aux_r['global_descriptor'])
                fused_phase = phase_coherence_loss(aux_r['phase'], aux_l['fusion_phase'], aux_l['fusion_confidence'])
                if word_mode:
                    fused_statistical = spectral_coherence_loss(aux_r['spectrum'], aux_l['fusion_spectrum'], aux_l['fusion_confidence'])
                    components.update(phase_circular_fused=fused_phase, spectral_coherence_fused=fused_statistical)
                    fused_phase = fused_phase + fused_statistical
                components.update({loss_prefix + '_local_independent': local,
                                   loss_prefix + '_global_independent': global_loss,
                                   'phase_independent': phase, loss_prefix + '_local_fused': fused_local,
                                   loss_prefix + '_global_fused': fused_global, 'phase_fused': fused_phase})
                local = (1 - weight) * local + weight * fused_local
                global_loss = (1 - weight) * global_loss + weight * fused_global
                phase = (1 - weight) * phase + weight * fused_phase
                components.update({loss_prefix + '_local': local, loss_prefix + '_global': global_loss, 'phase': phase})
            loss = local + global_loss + self.params.phase_loss_weight * phase
            return loss, components
        loss = contrastive(descriptors_l, descriptors_r)
        return loss, {'infonce': loss}

    def do_train(self):
        starting_epoch = self._init_weights()
        require_new_epochs(starting_epoch, self.params.epochs)
        for epoch in range(starting_epoch + 1, self.params.epochs + 1):
            print(f"\nTraining Process-epoch {epoch}-------->")
            if self.cross_hard_negative_count:
                mining = update_cross_modal_cache(self.model, self.train_loader.dataset, self.mining_loader,
                                                  self.device, self.cache_root, epoch, self.cross_hard_negative_count)
                print(f"Mined {mining['query_frames']} training queries: "
                      f"{self.cross_hard_negative_count} hard + {self.params.neg_num - self.cross_hard_negative_count} random; "
                      f"minimum geographic distance={mining['minimum_negative_distance_m']:.3f}m")
            query_num = np.zeros([1], dtype=np.float32)
            total_loss = np.zeros([1], dtype=np.float32)
            self.model.train()

            for batch_idx, batch in enumerate(tqdm(self.train_loader)):
                self.optimizer.zero_grad()
                l_bev = batch["lidar_bev"].to(self.device)
                r_bev = batch["radar_bev"].to(self.device)
                loss, components = self._compute_loss(self.model(l_bev, r_bev))
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite alignment loss')
                loss.backward()
                query_num[0] += 1
                total_loss[0] += loss.item()
                self.writer.add_scalar("loss 0", loss.item(), (epoch - 1) * self.total_iter + batch_idx)
                for name, component in components.items():
                    self.writer.add_scalar('alignment/' + name, component.detach().item(),
                                           (epoch - 1) * self.total_iter + batch_idx)
                self.optimizer.step()
                torch.cuda.empty_cache()

            loss_cur_epoch = total_loss / query_num
            print(f"epoch {epoch} finishes\nloss {loss_cur_epoch}")
            self.writer.add_scalar("total loss 0", loss_cur_epoch[0], epoch)
            self.scheduler.step()

            save_checkpoint_atomic(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "training_settings": self._training_settings(),
                    "training_state": capture_training_state(self.train_loader),
                    "model_variant": getattr(self.model, 'model_variant', 'rlpr'),
                    "alignment_loss": self._alignment_loss_name(),
                    "cross_hard_negative_count": self.cross_hard_negative_count,
                    "spectral_config": getattr(self.model, 'spectral_config', None),
                    "phase_loss_weight": self.params.phase_loss_weight,
                    "cross_fusion_enabled": getattr(self.model, 'cross_fusion_enabled', False),
                    "cross_fusion_loss_weight": self.params.cross_fusion_loss_weight,
                },
                os.path.join(self.weights_root, f"{epoch}.pth.tar"),
            )
