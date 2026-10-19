#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np

from scripts.registry import CROSS_MODAL, build_params, get_model_variant
from trainer.cross_modal import CrossModalTrainer
from utils.seed import set_global_seed

sys.maxsize = 3000
np.set_printoptions(threshold=sys.maxsize)


def main():
    parser = argparse.ArgumentParser(description="Cross-modal LiDAR-radar alignment training (TACMA).")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(CROSS_MODAL.keys()),
        help="Dataset name: nusc, mulran, snail, hercules",
    )
    parser.add_argument('--model', choices=['rlpr', 'spectral', 'spectral_word'], default=os.environ.get('RLPR_MODEL_VARIANT', 'rlpr'))
    parser.add_argument('--epochs', type=int, default=None, help='Target total epochs; overrides RLPR_EPOCHS and dataset default')
    parser.add_argument('--cross-fusion', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Paired radar-to-LiDAR phase cross-attention during spectral alignment training')
    args = parser.parse_args()
    if args.cross_fusion is None:
        args.cross_fusion = os.environ.get('RLPR_SPECTRAL_CROSS_FUSION', '1' if args.model == 'spectral_word' else '0') == '1'
    if args.cross_fusion and args.model not in ('spectral', 'spectral_word'):
        parser.error('--cross-fusion requires --model spectral or spectral_word')
    os.environ['RLPR_SPECTRAL_CROSS_FUSION'] = '1' if args.cross_fusion else '0'

    cfg = CROSS_MODAL[args.dataset]
    set_global_seed(0)
    CrossModalTrainer(
        build_params(cfg["epochs"], epochs_override=args.epochs),
        get_model_variant(cfg["model"], args.model),
        cfg["dataloader"],
        hercules_mode=cfg.get("hercules_mode", False),
    ).do_train()


if __name__ == "__main__":
    main()
