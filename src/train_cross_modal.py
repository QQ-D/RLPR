#!/usr/bin/env python3

import argparse
import sys

import numpy as np

from scripts.registry import CROSS_MODAL, build_params
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
    args = parser.parse_args()

    cfg = CROSS_MODAL[args.dataset]
    set_global_seed(0)
    CrossModalTrainer(
        build_params(cfg["epochs"]),
        cfg["model"],
        cfg["dataloader"],
        hercules_mode=cfg.get("hercules_mode", False),
    ).do_train()


if __name__ == "__main__":
    main()
