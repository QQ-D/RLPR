"""Cross-modal positive probability mass within geographic training neighborhoods."""
import math
import torch
from torch.nn import functional as F


def spatial_masks(positions, positive_radius=9., negative_radius=12.):
    if positions.ndim!=3 or positions.shape[-1]!=2 or positions.shape[1]<1:
        raise ValueError('Expected B x N x 2 training positions')
    if not 0<=positive_radius<negative_radius or not torch.isfinite(positions).all():
        raise ValueError('Invalid geographic thresholds or positions')
    xy=positions.detach().to(dtype=torch.float64)
    squared=(xy[:,:,None,:]-xy[:,None,:,:]).square().sum(-1)
    positive=squared<=positive_radius**2
    valid=positive | (squared>negative_radius**2)
    return positive,valid


def spatial_multi_infonce(teacher,student,positions,temperature=.07):
    if teacher.shape!=student.shape or teacher.ndim!=3 or teacher.shape[:2]!=positions.shape[:2]:
        raise ValueError('Descriptor and position shapes must match')
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Temperature must be positive and finite')
    positive,valid=spatial_masks(positions)
    logits=F.normalize(teacher.detach(),dim=-1) @ F.normalize(student,dim=-1).transpose(1,2)/temperature
    positive,valid=positive.to(logits.device),valid.to(logits.device)
    # Every synchronous diagonal is positive. Ambiguous 9--12m pairs are absent
    # from both sums; no cross-tuple negatives or extra frames are introduced.
    numerator=torch.logsumexp(logits.masked_fill(~positive,-torch.inf),dim=-1)
    denominator=torch.logsumexp(logits.masked_fill(~valid,-torch.inf),dim=-1)
    return (denominator-numerator).mean()
