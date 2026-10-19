import os
import torch


def validate_checkpoint_variant(model, checkpoint, *, check_cross_fusion=True):
    expected = getattr(model, 'model_variant', 'rlpr')
    actual = checkpoint.get('model_variant')
    if actual is None:
        actual = 'spectral' if any(k.startswith('spectral_') for k in checkpoint['state_dict']) else 'rlpr'
    if actual != expected:
        raise ValueError(f'Checkpoint variant {actual!r} does not match model variant {expected!r}. '
                         'Train spectral unimodal branches before spectral cross-modal alignment.')
    if expected in ('spectral', 'spectral_word') and checkpoint.get('spectral_config') != model.spectral_config:
        raise ValueError('Checkpoint spectral_config differs from current attention/dynamic-mask settings')
    if expected in ('spectral', 'spectral_word') and check_cross_fusion:
        expected_fusion = getattr(model, 'cross_fusion_enabled', False)
        actual_fusion = checkpoint.get('cross_fusion_enabled', any(
            key.startswith('cross_fusion.') for key in checkpoint['state_dict']))
        if expected_fusion != actual_fusion:
            raise ValueError('Checkpoint cross-fusion mode differs from model. '
                             'Use --cross-fusion for fusion checkpoints; omit it for old checkpoints.')


def require_checkpoint(path, env_name):
    if not path:
        raise ValueError(
            f"{env_name} is required but not set. "
            f"Export {env_name}=/path/to/checkpoint.pth.tar before running."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"{env_name} points to a missing file: {path}")


def load_checkpoint(model, path, env_name, strict=True):
    require_checkpoint(path, env_name)
    checkpoint = torch.load(path, map_location="cpu")
    validate_checkpoint_variant(model, checkpoint)
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    return checkpoint


def _spectral_branch_state(model, checkpoint, modality):
    """Validate the complete branch, including buffers, before changing model."""
    validate_checkpoint_variant(model, checkpoint, check_cross_fusion=False)
    branch = 'radar' if modality == 'r' else 'LiDAR'
    expected = {
        key: value for key, value in model.state_dict().items()
        if key.split('.')[0].endswith('_' + modality)
        or key.split('.')[0].startswith('net_vlad_' + modality + '_')
    }
    supplied = checkpoint['state_dict']
    unexpected = set(supplied) - set(expected)
    if unexpected:
        raise ValueError(f'Expected a {branch} unimodal checkpoint; unexpected keys: {sorted(unexpected)[:3]}')
    missing = set(expected) - set(supplied)
    if missing:
        raise ValueError(f'Incomplete {branch} checkpoint; missing keys: {sorted(missing)[:3]}')
    for key, target in expected.items():
        if not torch.is_tensor(supplied[key]) or supplied[key].shape != target.shape:
            raise ValueError(f'{branch} checkpoint shape mismatch for {key}')
    return supplied


def warmstart_word_unimodal(model, checkpoint):
    """Explicit model-only initialization for training, never a frozen teacher load.

    Only the pure-phase -> raw-magnitude interpretation may change. Optimizer,
    scheduler, epoch and random state are deliberately not restored here.
    """
    modality = getattr(model, 'modality', None)
    if getattr(model, 'model_variant', None) != 'spectral_word' or modality not in ('l', 'r'):
        raise ValueError('Warm start requires a Word unimodal model')
    source = checkpoint.get('spectral_config')
    target = model.spectral_config
    allowed_source = dict(target)
    if target.get('amplitude_mode') == 'raw_magnitude_v1':
        allowed_source.pop('amplitude_mode')
        allowed_source['pure_phase'] = True
    if source != target and source != allowed_source:
        raise ValueError('Warm-start spectral_config has an unrelated configuration change')
    adjusted = dict(checkpoint, spectral_config=target)
    state = _spectral_branch_state(model, adjusted, modality)
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Warm-start checkpoint contains nonfinite values')
    model.load_state_dict(state, strict=True)


def load_unimodal_pretrained(model, pretrain_r_path, pretrain_l_path, freeze_r=True):
    require_checkpoint(pretrain_r_path, "RLPR_PRETRAIN_R")
    require_checkpoint(pretrain_l_path, "RLPR_PRETRAIN_L")

    if getattr(model, 'model_variant', 'rlpr') in ('spectral', 'spectral_word'):
        ckpt_r = torch.load(pretrain_r_path, map_location='cpu')
        ckpt_l = torch.load(pretrain_l_path, map_location='cpu')
        radar_state = _spectral_branch_state(model, ckpt_r, 'r')
        lidar_state = _spectral_branch_state(model, ckpt_l, 'l')
        fusion_state = {key: value for key, value in model.state_dict().items()
                        if key.startswith('cross_fusion.')}
        model.load_state_dict({**radar_state, **lidar_state, **fusion_state}, strict=True)
        if freeze_r:
            freeze_radar_branch_cross(model)
        return

    ckpt_r = load_checkpoint(model, pretrain_r_path, "RLPR_PRETRAIN_R", strict=False)
    if freeze_r:
        layers_r = set(ckpt_r["state_dict"].keys())
        for name, param in model.named_parameters():
            if name in layers_r:
                param.requires_grad = False

    load_checkpoint(model, pretrain_l_path, "RLPR_PRETRAIN_L", strict=False)


def freeze_params_named_like_r_checkpoint(model, pretrain_r_path):
    require_checkpoint(pretrain_r_path, "RLPR_PRETRAIN_R")
    ckpt_r = torch.load(pretrain_r_path, map_location="cpu")
    layers_r = set(ckpt_r["state_dict"].keys())
    for name, param in model.named_parameters():
        if name in layers_r:
            param.requires_grad = False


def freeze_radar_branch_cross(model):
    for name, param in model.named_parameters():
        module_name = name.split(".")[0]
        if module_name.endswith("_r") or module_name.startswith('net_vlad_r_'):
            param.requires_grad = False


def load_snail_crossmodal_init(model, snail_cross_path):
    load_checkpoint(model, snail_cross_path, "RLPR_SNAIL_CROSS_CHECKPOINT", strict=True)
