#!/usr/bin/env python3

import argparse
import sys

import numpy as np

from scripts.registry import UNIMODAL, UNIMODAL_EPOCHS, build_params
from trainer.unimodal import UnimodalTrainer
from utils.seed import set_global_seed

sys.maxsize = 3000
np.set_printoptions(threshold=sys.maxsize)


def main():
    parser = argparse.ArgumentParser(description="Unimodal LiDAR or radar branch pre-training.")
    parser.add_argument("--dataset", required=True, choices=sorted({k[0] for k in UNIMODAL.keys()}))
    parser.add_argument("--modality", required=True, choices=["l", "r"], help="l=LiDAR, r=radar")
    args = parser.parse_args()

    key = (args.dataset, args.modality)
    if key not in UNIMODAL:
        raise ValueError(f"Unsupported combination: dataset={args.dataset}, modality={args.modality}")

    cfg = UNIMODAL[key]
    set_global_seed(0)
    UnimodalTrainer(
        build_params(UNIMODAL_EPOCHS),
        cfg["model"],
        cfg["dataloader"],
        modality=args.modality,
        cache_name=cfg["cache_name"],
        hard_mining_from_epoch=cfg["hard_mining_from_epoch"],
    ).do_train()


if __name__ == "__main__":
    main()
