"""Word prose mode: measured magnitude discarded, learned frequency gates retained."""

import math

import torch
from torch import nn

from modules.phase_transformer import unit_complex
from modules.spectral import real_spectrum_projection


def restore_measured_amplitude(proposal, aux):
    """Project phase before restoring magnitude, avoiding boundary attenuation.

    An undefined projected phase uses measured phase. Measured zero bins
    remain zero. This keeps DC/Nyquist realizable without discarding magnitude.
    """
    width = aux['spatial_size'][1]
    phase = unit_complex(real_spectrum_projection(proposal, width))
    fallback = unit_complex(aux['phase_content'])
    phase = torch.where(phase.abs() > 0, phase, fallback)
    spectrum = phase * aux['source_amplitude'] * aux['mask'] * aux['valid_phase']
    return real_spectrum_projection(spectrum, width)


class WordSpectralFilterLayer(nn.Module):
    def __init__(self, channels, retain_amplitude=False):
        super().__init__()
        self.channels = channels
        self.retain_amplitude = retain_amplitude
        self.cutoff_logit = nn.Parameter(torch.full((channels,), math.log(.65 / .35)))
        self.conditioner = nn.Sequential(nn.Linear(2 * channels, max(8, channels // 4)),
                                         nn.SiLU(), nn.Linear(max(8, channels // 4), channels))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'Expected [N, {self.channels}, H, W]')
        height, width = x.shape[-2:]
        with torch.autocast(device_type=x.device.type, enabled=False):
            raw = torch.fft.rfft2(x.float(), norm='ortho')
            phase = unit_complex(raw)
            summary = torch.cat((phase.real, phase.imag), dim=1).mean((-2, -1))
            cutoff = (self.cutoff_logit[None] + .25 * self.conditioner(summary)).sigmoid()[..., None, None]
            fy = torch.fft.fftfreq(height, device=x.device).view(-1, 1) / .5
            fx = torch.fft.rfftfreq(width, device=x.device).view(1, -1) / .5
            radius = torch.sqrt(fy.square() + fx.square()) / math.sqrt(2.)
            mask = torch.sigmoid(8. * (cutoff - radius[None, None]))
            spectrum = real_spectrum_projection(phase * mask, width)
            phase_content = spectrum
            if self.retain_amplitude:
                spectrum = real_spectrum_projection(spectrum * raw.abs(), width)
            confidence = mask.detach() * (phase.abs().detach() > 0)
            confidence = confidence.clone()
            confidence[..., 0, 0] = 0
            spatial = torch.fft.irfft2(spectrum, s=(height, width), norm='ortho')
            aux = {'phase': torch.angle(spectrum), 'confidence': confidence, 'mask': mask,
                   'spectrum': spectrum, 'spatial_size': (height, width),
                   'valid_phase': phase.abs().detach() > 0}
            if self.retain_amplitude:
                aux.update(phase_content=phase_content, source_amplitude=raw.abs())
        return spatial.to(x.dtype), aux
