"""Spectral variants sharing the original descriptor heads/checkpoint names."""

import os

import torch
from torch.nn import functional as F

from modules.LR_cross_hffc_225_dis import LRCross as CrossPolar
from modules.LR_cross_hffcn_225_dis import LRCross as CrossCart
from modules.LCRFusionGCNExpLHFFC_225 import LCRFusionGCNExpL as LidarPolar
from modules.LCRFusionGCNExpRHFFC import LCRFusionGCNExpR as RadarPolar
from modules.LCRFusionGCNExpLHFFCN_225 import LCRFusionGCNExpL as LidarCart
from modules.LCRFusionGCNExpRHFFCN import LCRFusionGCNExpR as RadarCart
from modules.spectral import SpectralFilterLayer, real_spectrum_projection
from modules.phase_cross import PhaseCrossAttention


class _SpectralFeatures:
    model_variant = 'spectral'

    def _install_spectral(self, modalities):
        self.spectral_config = {
            'phase_attention': os.environ.get('RLPR_SPECTRAL_PHASE_ATTENTION', '1') != '0',
            'dynamic': os.environ.get('RLPR_SPECTRAL_DYNAMIC', '1') != '0',
        }
        for modality in modalities:
            delattr(self, 'importance_' + modality)
            self.add_module('spectral_' + modality, SpectralFilterLayer(64, **self.spectral_config))

    def _spectral_stem(self, bev, modality):
        if bev.ndim == 5:
            bev = bev.flatten(0, 1)
        if bev.ndim != 4 or bev.shape[1] != 3:
            raise ValueError(f'Expected BEV [B, K, 3, H, W] or [N, 3, H, W], got {tuple(bev.shape)}')
        def layer(name):
            return getattr(self, name + '_' + modality)
        features = layer('conv1')(bev)
        features = self.relu(layer('bn1')(features))
        features = layer('maxpool1')(features)
        return layer('spectral')(features)

    def _decode_branch(self, features, modality, aux):
        def layer(name):
            return getattr(self, name + '_' + modality)
        features = layer('layer2')(layer('layer1')(features))
        mask = torch.zeros(features.shape[0], *features.shape[-2:], device=features.device, dtype=features.dtype)
        tokens = (features + layer('pos_emb')(mask)).flatten(2).transpose(1, 2)
        global_descriptor = F.normalize(getattr(self, 'net_vlad_' + modality + '_g')(layer('encoder')(tokens)), dim=1)
        local_descriptor = F.normalize(layer('linear')(layer('gem')(tokens)), dim=1)
        aux['local_descriptor'] = local_descriptor
        aux['global_descriptor'] = global_descriptor
        return torch.cat((global_descriptor, local_descriptor), dim=-1), aux

    def _branch_forward(self, bev, modality):
        features, aux = self._spectral_stem(bev, modality)
        return self._decode_branch(features, modality, aux)


class _SpectralCrossMixin(_SpectralFeatures):
    def __init__(self, pretrained):
        super().__init__(pretrained)
        self._install_spectral(('l', 'r'))
        self.cross_fusion_enabled = os.environ.get('RLPR_SPECTRAL_CROSS_FUSION', '0') == '1'
        if self.cross_fusion_enabled:
            self.cross_fusion = PhaseCrossAttention(64)

    def forward(self, l_bev, r_bev):
        lidar, aux_lidar = self._branch_forward(l_bev, 'l')
        radar, aux_radar = self._branch_forward(r_bev, 'r')
        if self.training and self.cross_fusion_enabled:
            height, width = aux_lidar['spatial_size']
            spectrum = self.cross_fusion(aux_lidar['spectrum'], aux_radar['spectrum'])
            spectrum = real_spectrum_projection(spectrum, width)
            fused_features = torch.fft.irfft2(spectrum, s=(height, width), norm='ortho')
            fused_aux = {'phase': torch.angle(spectrum),
                         'confidence': torch.minimum(aux_lidar['confidence'], aux_radar['confidence'])}
            _, fused_aux = self._decode_branch(fused_features, 'l', fused_aux)
            for key in ('phase', 'confidence', 'local_descriptor', 'global_descriptor'):
                aux_lidar['fusion_' + key] = fused_aux[key]
        return lidar, radar, aux_lidar, aux_radar

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for name, child in self.named_children():
                if (name.endswith('_r') or name.startswith('net_vlad_r_')) and not any(p.requires_grad for p in child.parameters()):
                    child.eval()
        return self


class _SpectralUnimodalMixin(_SpectralFeatures):
    def __init__(self, pretrained):
        super().__init__(pretrained)
        self._install_spectral((self.modality,))

    def forward(self, img, anchor_points, meta, l_bev, r_bev):
        descriptor, _ = self._branch_forward(l_bev if self.modality == 'l' else r_bev, self.modality)
        return descriptor, descriptor, descriptor, descriptor


class SpectralCrossPolar(_SpectralCrossMixin, CrossPolar):
    pass


class SpectralCrossCart(_SpectralCrossMixin, CrossCart):
    pass


class SpectralLidarPolar(_SpectralUnimodalMixin, LidarPolar):
    modality = 'l'


class SpectralRadarPolar(_SpectralUnimodalMixin, RadarPolar):
    modality = 'r'


class SpectralLidarCart(_SpectralUnimodalMixin, LidarCart):
    modality = 'l'


class SpectralRadarCart(_SpectralUnimodalMixin, RadarCart):
    modality = 'r'


MODEL_VARIANTS = {
    CrossPolar: SpectralCrossPolar,
    CrossCart: SpectralCrossCart,
    LidarPolar: SpectralLidarPolar,
    RadarPolar: SpectralRadarPolar,
    LidarCart: SpectralLidarCart,
    RadarCart: SpectralRadarCart,
}
