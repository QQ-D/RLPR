#!/usr/bin/env bash
# SnailRadar full pipeline: preprocess -> unimodal L/R (5 ep) -> cross-modal (4 ep) -> eval
# Usage: RLPR_SNAIL_RAW=/path/to/Snail-Radar RLPR_SNAIL_DATA=/path/to/data ./scripts/snail.sh
# Optional: SKIP_PREP=1  SKIP_TRAIN=1
# Note: hard mining starts at unimodal epoch 3 automatically.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RLPR_SNAIL_RAW="${RLPR_SNAIL_RAW:-/path/to/Snail-Radar}"
export RLPR_SNAIL_DATA="${RLPR_SNAIL_DATA:-$HOME/RLPR_data/snail}"
OUT="${RLPR_OUTPUT_ROOT:-./outputs}"

if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  echo "=== preprocess ==="
  python dataset/SnailRadar/gen_info.py
  python dataset/SnailRadar/gen_index.py
  python dataset/SnailRadar/gen_bev.py
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "=== unimodal LiDAR (5 epochs) ==="
  export RLPR_TRAINING_ROOT="$OUT/l_snail" RLPR_RESUME=0
  python train_unimodal.py --dataset snail --modality l

  echo "=== unimodal radar (5 epochs) ==="
  export RLPR_TRAINING_ROOT="$OUT/r_snail" RLPR_RESUME=0
  python train_unimodal.py --dataset snail --modality r

  echo "=== cross-modal (4 epochs) ==="
  export RLPR_TRAINING_ROOT="$OUT/r2l_snail" RLPR_RESUME=0
  export RLPR_PRETRAIN_L="$OUT/l_snail/weights/5.pth.tar"
  export RLPR_PRETRAIN_R="$OUT/r_snail/weights/5.pth.tar"
  python train_cross_modal.py --dataset snail
fi

echo "=== eval (epoch 4) ==="
export RLPR_TRAINING_ROOT="$OUT/r2l_snail" RLPR_RESUME_EPOCH=4
python test_cross_modal.py --dataset snail
