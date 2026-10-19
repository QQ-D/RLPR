"""Transformer encoder architecture with phase-only queries and keys."""

import torch
from torch import nn


def unit_complex(spectrum):
    """Discard measured magnitude; suppress undefined near-zero phases."""
    amplitude = spectrum.abs()
    valid = amplitude > 1e-7
    denominator = torch.where(valid, amplitude, torch.ones_like(amplitude))
    return torch.where(valid, spectrum, torch.zeros_like(spectrum)) / denominator


class PhaseTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Keep MHA, residuals, norms and FFN; compute Q/K from circular phase."""

    def __init__(self, d_model=128, nhead=8, dim_feedforward=512, dropout=0.):
        if d_model % 2:
            raise ValueError('Real/imaginary feature width must be even')
        super().__init__(d_model, nhead, dim_feedforward, dropout, batch_first=True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        real, imag = src.chunk(2, dim=-1)
        phase = unit_complex(torch.complex(real, imag))
        circular = torch.cat((phase.real, phase.imag), dim=-1)
        attended = self.self_attn(circular, circular, src,
                                 attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
                                 need_weights=False, is_causal=is_causal)[0]
        x = self.norm1(src + self.dropout1(attended))
        return self.norm2(x + self._ff_block(x))
