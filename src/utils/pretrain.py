import os
import torch


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
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    return checkpoint


def load_unimodal_pretrained(model, pretrain_r_path, pretrain_l_path, freeze_r=True):
    require_checkpoint(pretrain_r_path, "RLPR_PRETRAIN_R")
    require_checkpoint(pretrain_l_path, "RLPR_PRETRAIN_L")

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
        if module_name.endswith("_r"):
            param.requires_grad = False


def load_snail_crossmodal_init(model, snail_cross_path):
    load_checkpoint(model, snail_cross_path, "RLPR_SNAIL_CROSS_CHECKPOINT", strict=True)
