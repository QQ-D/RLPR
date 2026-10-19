"""Complete Word prose mode, isolated from previously evaluated spectral models."""

import os

import torch
from torch import nn
from torch.nn import functional as F

from modules.spectral_models import (
    _SpectralCrossMixin, _SpectralUnimodalMixin,
    CrossPolar, CrossCart, LidarPolar, RadarPolar, LidarCart, RadarCart,
)
from modules.word_spectral import WordSpectralFilterLayer, restore_measured_amplitude
from modules.phase_transformer import PhaseTransformerEncoderLayer, unit_complex
from modules.phase_cross import PhaseCrossAttention
from modules.spectral import real_spectrum_projection


class _WordFeatures:
    model_variant = 'spectral_word'

    def _install_spectral(self, modalities):
        amplitude = os.environ.get('RLPR_WORD_AMPLITUDE', '0')
        if amplitude not in ('0', '1'):
            raise ValueError('RLPR_WORD_AMPLITUDE must be 0 or 1')
        self.retain_amplitude = amplitude == '1'
        self.spectral_config = {'pure_phase': True, 'dynamic': True,
                                'phase_transformer': True, 'alignment': 'welch_coherence_v1',
                                'coherence_segment': 32, 'coherence_stride': 16, 'coherence_margin': .2}
        if self.retain_amplitude:
            self.spectral_config.update(pure_phase=False, amplitude_mode='raw_magnitude_v1')
        for modality in modalities:
            delattr(self, 'importance_' + modality)
            delattr(self, 'encoder_layer_' + modality)
            self.add_module('spectral_' + modality,
                            WordSpectralFilterLayer(64, retain_amplitude=self.retain_amplitude))
            encoder = PhaseTransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.)
            self.add_module('encoder_' + modality, nn.TransformerEncoder(encoder, 1, enable_nested_tensor=False))

    def _spectral_stem(self, bev, modality):
        _, aux = super()._spectral_stem(bev, modality)
        spectrum = aux['phase_content'] if self.retain_amplitude else aux['spectrum']
        batch, channels, height, width = spectrum.shape
        tokens = torch.cat((spectrum.real, spectrum.imag), 1).flatten(2).transpose(1, 2)
        tokens = getattr(self, 'encoder_' + modality)(tokens)
        content = tokens.transpose(1, 2).reshape(batch, 2 * channels, height, width)
        real, imag = content.chunk(2, dim=1)
        spectrum = unit_complex(torch.complex(real, imag)) * aux['mask'] * aux['valid_phase']
        if self.retain_amplitude:
            spectrum = restore_measured_amplitude(spectrum, aux)
        else:
            spectrum = real_spectrum_projection(spectrum, aux['spatial_size'][1])
        aux.update(spectrum=spectrum, phase=torch.angle(spectrum))
        spatial = torch.fft.irfft2(spectrum, s=aux['spatial_size'], norm='ortho')
        return spatial, aux

    def _decode_branch(self, features, modality, aux):
        def layer(name):
            return getattr(self, name + '_' + modality)
        features = layer('layer2')(layer('layer1')(features))
        mask = torch.zeros(features.shape[0], *features.shape[-2:], device=features.device, dtype=features.dtype)
        tokens = (features + layer('pos_emb')(mask)).flatten(2).transpose(1, 2)
        # The original encoder now operates on complex frequency tokens before iFFT.
        global_descriptor = F.normalize(getattr(self, 'net_vlad_' + modality + '_g')(tokens), dim=1)
        local_descriptor = F.normalize(layer('linear')(layer('gem')(tokens)), dim=1)
        aux.update(local_descriptor=local_descriptor, global_descriptor=global_descriptor)
        return torch.cat((global_descriptor, local_descriptor), -1), aux


class _WordCrossMixin(_WordFeatures, _SpectralCrossMixin):
    def __init__(self, pretrained):
        super().__init__(pretrained)
        if 'RLPR_SPECTRAL_CROSS_FUSION' not in os.environ:
            self.cross_fusion_enabled = True
            self.cross_fusion = PhaseCrossAttention(64)

    def forward(self, l_bev, r_bev):
        lidar, al = self._branch_forward(l_bev, 'l')
        radar, ar = self._branch_forward(r_bev, 'r')
        if self.training and self.cross_fusion_enabled:
            spectrum = self.cross_fusion(al['spectrum'], ar['spectrum'])
            spectrum = unit_complex(spectrum) * al['mask'] * al['valid_phase']
            if self.retain_amplitude:
                spectrum = restore_measured_amplitude(spectrum, al)
            else:
                spectrum = real_spectrum_projection(spectrum, al['spatial_size'][1])
            spatial = torch.fft.irfft2(spectrum, s=al['spatial_size'], norm='ortho')
            fused_aux = {'phase': torch.angle(spectrum),
                         'spectrum': spectrum,
                         'confidence': torch.minimum(al['confidence'], ar['confidence'])}
            _, fused_aux = self._decode_branch(spatial, 'l', fused_aux)
            for key in ('phase', 'confidence', 'spectrum', 'local_descriptor', 'global_descriptor'):
                al['fusion_' + key] = fused_aux[key]
        return lidar, radar, al, ar


class WordCrossPolar(_WordCrossMixin, CrossPolar):
    pass


class WordCrossCart(_WordCrossMixin, CrossCart):
    pass


class WordLidarPolar(_WordFeatures, _SpectralUnimodalMixin, LidarPolar):
    modality = 'l'


class WordRadarPolar(_WordFeatures, _SpectralUnimodalMixin, RadarPolar):
    modality = 'r'


class WordLidarCart(_WordFeatures, _SpectralUnimodalMixin, LidarCart):
    modality = 'l'


class WordRadarCart(_WordFeatures, _SpectralUnimodalMixin, RadarCart):
    modality = 'r'


MODEL_VARIANTS = {CrossPolar: WordCrossPolar, CrossCart: WordCrossCart,
                  LidarPolar: WordLidarPolar, RadarPolar: WordRadarPolar,
                  LidarCart: WordLidarCart, RadarCart: WordRadarCart}
