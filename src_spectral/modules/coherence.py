"""Signed descriptor alignment and statistical coherence of frequency fields."""

import torch
from torch.nn import functional as F


def descriptor_coherence(student, teacher, *, return_bins=False, segment=32, stride=16):
    """Average complex cross/auto spectra across windows, then form MSC.

    The descriptor dimension is the windowed axis. Require at least two
    windows: one periodogram makes magnitude-squared coherence trivial.
    Zero-power bins have zero coherence. Teacher detachment is done by the loss.
    """
    if student.shape[-1] != teacher.shape[-1] or student.shape[-1] < segment + stride:
        raise ValueError('Matching descriptor dimensions and at least two windows are required')
    if segment < 2 or not 0 < stride <= segment:
        raise ValueError('Invalid coherence segment/stride')
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=student.device.type, enabled=False):
        window = torch.hann_window(segment, device=student.device, dtype=dtype)
        x = torch.fft.rfft(student.to(dtype).unfold(-1, segment, stride) * window, norm='ortho')
        y = torch.fft.rfft(teacher.to(dtype).unfold(-1, segment, stride) * window, norm='ortho')
        pxy = (x * y.conj()).mean(dim=-2)
        pxx = x.abs().square().mean(dim=-2)
        pyy = y.abs().square().mean(dim=-2)
        power = pxx * pyy
        bins = (pxy.abs().square() / power.clamp_min(torch.finfo(dtype).tiny)).clamp(0, 1)
        bins = torch.where(power > 0, bins, torch.zeros_like(bins))
    return bins if return_bins else bins.mean(dim=-1)


def coherence_alignment_loss(lidar, radar, margin=.2):
    """Maximize signed coherency; geographic negatives rank lower.

    Use the real normalized inner product (cosine), equivalent by Parseval to
    the real summed cross spectrum divided by total spectral norms. Unlike
    Welch magnitude-squared coherence, it retains the sign of the relationship.
    For unit descriptors its ranking agrees with squared L2 = 2 - 2 * score.
    The separate raw frequency-field auxiliary still uses statistical MSC.

    Tuple layout is [query, positive, ten mined negatives]. Every scan has a
    synchronized radar/LiDAR match. Only the mined negatives are repelled;
    the nearby positive is never treated as a negative. No InfoNCE/softmax.
    """
    if lidar.shape != radar.shape or lidar.ndim != 3 or lidar.shape[1] < 3:
        raise ValueError('Expected matching [B, query+positive+negatives, D] descriptors')
    dtype = torch.float64 if lidar.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=lidar.device.type, enabled=False):
        lidar = F.normalize(lidar.to(dtype), dim=-1)
        radar = F.normalize(radar.detach().to(dtype), dim=-1)
        matched = (lidar * radar).sum(dim=-1).clamp(-1., 1.)
        forward = (lidar * radar[:, :1]).sum(dim=-1).clamp(-1., 1.)
        reverse = (lidar[:, :1] * radar).sum(dim=-1).clamp(-1., 1.)
    near = .5 * (forward[:, 1] + reverse[:, 1])
    positive = .5 * (matched[:, 0] + near)
    negative = torch.cat((forward[:, 2:], reverse[:, 2:]), dim=1)
    return 1. - matched.mean() + .5 * (1. - near.mean()) + F.relu(margin + negative - positive[:, None]).mean()


def spectral_coherence_loss(radar_spectrum, lidar_spectrum, weights, kernel=3):
    """MSC of complex 2D fields, estimated with local frequency smoothing.

    Cross/auto spectra are smoothed BEFORE division. Circular consistency
    is a separate term because MSC cannot distinguish constant phase offsets.
    """
    if (radar_spectrum.shape != lidar_spectrum.shape or weights.shape != lidar_spectrum.shape
            or lidar_spectrum.ndim != 4 or not lidar_spectrum.is_complex()
            or not radar_spectrum.is_complex() or kernel < 3 or kernel % 2 == 0):
        raise ValueError('Expected matching complex [N,C,H,W], real confidence, and odd kernel >=3')
    radar = radar_spectrum.detach()
    def average(x):
        return F.avg_pool2d(x, kernel, stride=1, padding=kernel // 2, count_include_pad=False)
    cross = lidar_spectrum * radar.conj()
    numerator = average(cross.real).square() + average(cross.imag).square()
    power = average(lidar_spectrum.abs().square()) * average(radar.abs().square())
    score = (numerator / power.clamp_min(1e-12)).clamp(0, 1)
    confidence = weights.detach().clamp_min(0) * (power.detach() > 1e-12)
    return ((1. - score) * confidence).sum() / confidence.sum().clamp_min(1e-6)
