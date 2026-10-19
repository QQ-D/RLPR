#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np

from scripts.registry import TEST, build_test_params, get_model_variant
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
    parser.add_argument('--model', choices=['rlpr', 'spectral', 'spectral_word'], default=os.environ.get('RLPR_MODEL_VARIANT', 'rlpr'))
    parser.add_argument('--cross-fusion', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Load a model trained with cross-attention; retrieval descriptors remain independent')
    args = parser.parse_args()
    if args.cross_fusion is None:
        args.cross_fusion = os.environ.get('RLPR_SPECTRAL_CROSS_FUSION', '1' if args.model == 'spectral_word' else '0') == '1'
    if args.cross_fusion and args.model not in ('spectral', 'spectral_word'):
        parser.error('--cross-fusion requires --model spectral or spectral_word')
    os.environ['RLPR_SPECTRAL_CROSS_FUSION'] = '1' if args.cross_fusion else '0'

    cfg = TEST[args.dataset]
    params = build_test_params()
    if args.zero_shot:
        params.resume_checkpoint = False
    set_global_seed(0)
    CrossModalTester(params, get_model_variant(cfg["model"], args.model), cfg["dataloader"]).do_test()


if __name__ == "__main__":
    main()
