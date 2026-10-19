#!/usr/bin/env python3

import argparse
import sys

import numpy as np

from scripts.registry import TEST, build_test_params
from trainer.eval import CrossModalTester
from utils.seed import set_global_seed

sys.maxsize = 3000
np.set_printoptions(threshold=sys.maxsize)


def main():
    parser = argparse.ArgumentParser(description="Cross-modal place recognition evaluation.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(TEST.keys()),
        help="nusc, mulran, snail, hercules, boreas",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Evaluate without loading a checkpoint (same as RLPR_TEST_ZERO_SHOT=1)",
    )
    args = parser.parse_args()

    cfg = TEST[args.dataset]
    params = build_test_params()
    if args.zero_shot:
        params.resume_checkpoint = False
    set_global_seed(0)
    CrossModalTester(params, cfg["model"], cfg["dataloader"]).do_test()


if __name__ == "__main__":
    main()
