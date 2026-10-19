#!/usr/bin/env bash
# MulRan full pipeline: preprocess -> unimodal L/R (5 ep) -> cross-modal (1 ep) -> eval
# Usage: RLPR_MULRAN_RAW=/path/to/MulRan RLPR_MULRAN_DATA=/path/to/data ./scripts/mulran.sh
# Optional: SKIP_PREP=1  SKIP_TRAIN=1  RLPR_MULRAN_TEST_SPLIT=sejong|riverside|dcc
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RLPR_MULRAN_RAW="${RLPR_MULRAN_RAW:-/path/to/MulRan}"
export RLPR_MULRAN_DATA="${RLPR_MULRAN_DATA:-$HOME/RLPR_data/mulran}"
export RLPR_MULRAN_TEST_SPLIT="${RLPR_MULRAN_TEST_SPLIT:-dcc}"
OUT="${RLPR_OUTPUT_ROOT:-./outputs}"

if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  echo "=== preprocess ==="
  python dataset/MulRan/gen_info.py
  python dataset/MulRan/gen_index.py
  python dataset/MulRan/gen_bev.py
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "=== unimodal LiDAR (5 epochs) ==="
  export RLPR_TRAINING_ROOT="$OUT/l_mulran" RLPR_RESUME=0
  python train_unimodal.py --dataset mulran --modality l

  echo "=== unimodal radar (5 epochs) ==="
  export RLPR_TRAINING_ROOT="$OUT/r_mulran" RLPR_RESUME=0
  python train_unimodal.py --dataset mulran --modality r

  echo "=== cross-modal (1 epoch) ==="
  export RLPR_TRAINING_ROOT="$OUT/r2l_mulran" RLPR_RESUME=0
  export RLPR_PRETRAIN_L="$OUT/l_mulran/weights/5.pth.tar"
  export RLPR_PRETRAIN_R="$OUT/r_mulran/weights/5.pth.tar"
  python train_cross_modal.py --dataset mulran
fi

echo "=== eval (epoch 1, split=$RLPR_MULRAN_TEST_SPLIT) ==="
export RLPR_TRAINING_ROOT="$OUT/r2l_mulran" RLPR_RESUME_EPOCH=1
python test_cross_modal.py --dataset mulran
