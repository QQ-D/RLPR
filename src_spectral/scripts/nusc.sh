#!/usr/bin/env bash
# NuScenes full pipeline: preprocess -> unimodal L/R (5 ep) -> cross-modal (3 ep) -> eval
# Usage: RLPR_NUSC_RAW=/path/to/nuscenes RLPR_NUSC_DATA=/path/to/data ./scripts/nusc.sh
# Optional: SKIP_PREP=1  SKIP_TRAIN=1  RLPR_NUSC_TEST_SPLIT=son|sq|bs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RLPR_NUSC_RAW="${RLPR_NUSC_RAW:-/path/to/nuscenes}"
export RLPR_NUSC_DATA="${RLPR_NUSC_DATA:-$HOME/RLPR_data/nusc}"
export RLPR_NUSC_TEST_SPLIT="${RLPR_NUSC_TEST_SPLIT:-bs}"
OUT="${RLPR_OUTPUT_ROOT:-./outputs}"

if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  echo "=== preprocess ==="
  python dataset/NuScenes/gen_info.py
  python dataset/NuScenes/gen_index.py
  python dataset/NuScenes/gen_bev.py
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "=== unimodal LiDAR (5 epochs) ==="
  export RLPR_TRAINING_ROOT="$OUT/l_nusc" RLPR_RESUME=0
  python train_unimodal.py --dataset nusc --modality l

  echo "=== unimodal radar (5 epochs) ==="
  export RLPR_TRAINING_ROOT="$OUT/r_nusc" RLPR_RESUME=0
  python train_unimodal.py --dataset nusc --modality r

  echo "=== cross-modal (3 epochs) ==="
  export RLPR_TRAINING_ROOT="$OUT/r2l_nusc" RLPR_RESUME=0
  export RLPR_PRETRAIN_L="$OUT/l_nusc/weights/5.pth.tar"
  export RLPR_PRETRAIN_R="$OUT/r_nusc/weights/5.pth.tar"
  python train_cross_modal.py --dataset nusc
fi

echo "=== eval (epoch 3, split=$RLPR_NUSC_TEST_SPLIT) ==="
export RLPR_TRAINING_ROOT="$OUT/r2l_nusc" RLPR_RESUME_EPOCH=3
python test_cross_modal.py --dataset nusc
