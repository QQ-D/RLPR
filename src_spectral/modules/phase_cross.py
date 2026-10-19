"""Asymmetric radar-to-LiDAR phase cross-attention for paired training."""

import torch
from torch import nn
from torch.nn import functional as F


class PhaseCrossAttention(nn.Module):
    """LiDAR phase Q reads frozen radar phase K and complex radar V."""

    def __init__(self, channels, attention_dim=32, heads=4):
        super().__init__()
        if attention_dim % heads or 2 * channels % heads:
            raise ValueError('Attention and value dimensions must be divisible by heads')
        self.channels = channels
        self.heads = heads
        self.head_dim = attention_dim // heads
        self.phase_norm = nn.LayerNorm(2 * channels)
        self.query = nn.Linear(2 * channels, attention_dim, bias=False)
        self.key = nn.Linear(2 * channels, attention_dim, bias=False)
        self.value = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.output = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.strength = nn.Parameter(torch.tensor(-3.0))

    def forward(self, lidar_spectrum, radar_spectrum):
        if (lidar_spectrum.ndim != 4 or radar_spectrum.ndim != 4
                or lidar_spectrum.shape[:2] != radar_spectrum.shape[:2]
                or lidar_spectrum.shape[1] != self.channels
                or not lidar_spectrum.is_complex() or not radar_spectrum.is_complex()):
            raise ValueError('Expected complex [N, C, H, W] spectra with matching batch/channels')
        batch, channels, height, width = lidar_spectrum.shape
        with torch.autocast(device_type=lidar_spectrum.device.type, enabled=False):
            lidar = lidar_spectrum.to(torch.complex64)
            radar = radar_spectrum.detach().to(torch.complex64)
            def phase_tokens(spectrum):
                unit = spectrum / spectrum.abs().clamp_min(1e-6)
                circular = torch.cat((unit.real, unit.imag), dim=1).flatten(2).transpose(1, 2)
                return self.phase_norm(circular)
            query = self.query(phase_tokens(lidar)).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
            key = self.key(phase_tokens(radar)).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
            content = torch.cat((radar.real, radar.imag), dim=1).flatten(2).transpose(1, 2)
            value = self.value(content).view(batch, -1, self.heads, 2 * channels // self.heads).transpose(1, 2)
            fused = F.scaled_dot_product_attention(query, key, value, dropout_p=0.)
            fused = fused.transpose(1, 2).reshape(batch, -1, 2 * channels)
            fused = self.output(fused).transpose(1, 2).reshape(batch, 2 * channels, height, width)
            real, imag = fused.chunk(2, dim=1)
            return lidar + self.strength.sigmoid() * torch.complex(real, imag)
