"""Spatial Fourier processing; phase here is not radar RF signal phase."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PhaseDominantAttention(nn.Module):
    """Within-modality attention: circular phase Q/K, complex content V."""

    def __init__(self, channels, attention_dim=32, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = attention_dim // heads
        self.phase_norm = nn.LayerNorm(2 * channels)
        self.query = nn.Linear(2 * channels, attention_dim, bias=False)
        self.key = nn.Linear(2 * channels, attention_dim, bias=False)
        self.value = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.output = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.strength = nn.Parameter(torch.tensor(-3.0))

    def forward(self, spectrum):
        batch, channels, height, width = spectrum.shape
        unit_phase = spectrum / spectrum.abs().clamp_min(1e-6)
        circular = torch.cat((unit_phase.real, unit_phase.imag), dim=1)
        circular = self.phase_norm(circular.flatten(2).transpose(1, 2))
        content = torch.cat((spectrum.real, spectrum.imag), dim=1).flatten(2).transpose(1, 2)
        query = self.query(circular).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(circular).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(content).view(batch, -1, self.heads, 2 * channels // self.heads).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.)
        attended = attended.transpose(1, 2).reshape(batch, -1, 2 * channels)
        attended = self.output(attended).transpose(1, 2).reshape(batch, 2 * channels, height, width)
        real, imag = attended.chunk(2, dim=1)
        return spectrum + self.strength.sigmoid() * torch.complex(real, imag)


def real_spectrum_projection(spectrum, spatial_width):
    """Enforce Hermitian symmetry at the rFFT DC/Nyquist columns."""
    height = spectrum.shape[-2]
    reverse = (-torch.arange(height, device=spectrum.device)) % height
    projected = spectrum.clone()
    columns = [0]
    if spatial_width % 2 == 0:
        columns.append(spectrum.shape[-1] - 1)
    for column_index in columns:
        column = spectrum[..., column_index]
        projected[..., column_index] = .5 * (column + column.index_select(-1, reverse).conj())
    return projected


class SpectralFilterLayer(nn.Module):
    """Dynamic low-pass amplitude mask, phase attention, and real inverse FFT.

    Shapes are derived from the input. The positive amplitude mask preserves
    phase by itself; subsequent attention can change phase and propagate
    supervision to the preceding CNN and, indirectly, the mask conditioner.
    """

    def __init__(self, channels, phase_attention=True, dynamic=True):
        super().__init__()
        self.channels = channels
        self.dynamic = dynamic
        self.cutoff_logit = nn.Parameter(torch.full((channels,), math.log(.65 / .35)))
        hidden = max(8, channels // 4)
        self.conditioner = nn.Sequential(nn.Linear(channels, hidden), nn.SiLU(), nn.Linear(hidden, channels))
        self.phase_attention = PhaseDominantAttention(channels) if phase_attention else None

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'Expected [B, {self.channels}, H, W], got {tuple(x.shape)}')
        height, width = x.shape[-2:]
        # FP16 FFT does not support these non-power-of-two BEV dimensions.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft2(x.float(), norm='ortho')
            amplitude = spectrum.abs()
            summary = amplitude.log1p().mean(dim=(-2, -1))
            offset = self.conditioner(summary) if self.dynamic else torch.zeros_like(summary)
            cutoff = (self.cutoff_logit[None, :] + .25 * offset).sigmoid()[..., None, None]
            radial_frequency = torch.fft.fftfreq(height, device=x.device).view(-1, 1) / .5
            angular_frequency = torch.fft.rfftfreq(width, device=x.device).view(1, -1) / .5
            radius = torch.sqrt(radial_frequency.square() + angular_frequency.square()) / math.sqrt(2.)
            mask = torch.sigmoid(8. * (cutoff - radius[None, None]))
            filtered = spectrum * mask
            if self.phase_attention is not None:
                filtered = self.phase_attention(filtered)
            filtered = real_spectrum_projection(filtered, width)
            spatial = torch.fft.irfft2(filtered, s=(height, width), norm='ortho')

            # Reliability cannot be optimized to hide phase disagreement.
            mean_amplitude = amplitude.detach().mean(dim=(-2, -1), keepdim=True)
            confidence = (amplitude.detach() / (mean_amplitude + 1e-6)).clamp(max=1.)
            confidence *= (amplitude.detach() > mean_amplitude * 1e-4 + 1e-6)
            confidence *= mask.detach()
            confidence = confidence.clone()
            confidence[..., 0, 0] = 0  # DC does not describe spatial geometry.
            aux = {'phase': torch.angle(filtered), 'confidence': confidence, 'mask': mask,
                   'spectrum': filtered, 'spatial_size': (height, width)}
        return spatial.to(x.dtype), aux
