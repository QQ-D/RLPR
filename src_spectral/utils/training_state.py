"""Reproducible epoch-boundary checkpoints for nonpersistent data workers."""
import random
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


def save_checkpoint_atomic(checkpoint, path):
    """Never expose an interrupted write as a resumable final checkpoint."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def configure_sampling(loader, seed=None):
    if seed is None:
        return
    if not hasattr(loader.sampler, 'generator'):
        raise ValueError('A sampling seed requires a random sampler')
    loader.generator = torch.Generator().manual_seed(seed)
    loader.sampler.generator = torch.Generator().manual_seed(seed)


def capture_training_state(loader):
    if getattr(loader, 'persistent_workers', False):
        raise ValueError('Exact epoch-boundary resume requires nonpersistent workers')
    def generator_state(generator):
        return None if generator is None else generator.get_state()
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        'loader_generator': generator_state(loader.generator),
        'sampler_generator': generator_state(getattr(loader.sampler, 'generator', None)),
    }


def restore_training_state(state, loader):
    if state is None:
        return False  # Legacy checkpoints remain loadable, without an exact RNG guarantee.
    if getattr(loader, 'persistent_workers', False):
        raise ValueError('Exact epoch-boundary resume requires nonpersistent workers')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    for obj, key in ((loader, 'loader_generator'), (loader.sampler, 'sampler_generator')):
        saved = state[key]
        if saved is not None:
            generator = getattr(obj, 'generator', None)
            if generator is None:
                generator = torch.Generator()
                obj.generator = generator
            generator.set_state(saved.cpu())
        elif hasattr(obj, 'generator'):
            obj.generator = None
    if state['torch_cuda'] is not None:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('Saved CUDA random state requires the same CUDA device count')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])
    return True


def validate_training_settings(checkpoint, expected):
    # Older checkpoints record objective metadata at the top level.
    saved = {**checkpoint, **checkpoint.get('training_settings', {})}
    # Cross-modal hard mining did not exist in older checkpoints.
    saved.setdefault('cross_hard_negative_count', 0)
    for key, value in expected.items():
        if key in saved and saved[key] != value:
            raise ValueError(f'Resume {key} differs: checkpoint={saved[key]!r}, requested={value!r}')


def require_new_epochs(starting_epoch, total_epochs):
    if total_epochs <= starting_epoch:
        raise ValueError(f'Target epochs={total_epochs} must exceed completed epoch={starting_epoch}')
