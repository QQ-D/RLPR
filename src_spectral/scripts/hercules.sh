#!/usr/bin/env bash
# Hercules full pipeline: preprocess -> cross-modal fine-tune (1 ep, Snail init) -> eval
# Usage: RLPR_HERCULES_RAW=/path/to/Hercules RLPR_HERCULES_DATA=/path/to/data ./scripts/hercules.sh
# Requires Snail cross-modal checkpoint (run ./scripts/snail.sh first, or set path below).
# Optional: SKIP_PREP=1  SKIP_TRAIN=1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RLPR_HERCULES_RAW="${RLPR_HERCULES_RAW:-/path/to/Hercules}"
export RLPR_HERCULES_DATA="${RLPR_HERCULES_DATA:-$HOME/RLPR_data/hercules}"
OUT="${RLPR_OUTPUT_ROOT:-./outputs}"
export RLPR_SNAIL_CROSS_CHECKPOINT="${RLPR_SNAIL_CROSS_CHECKPOINT:-$OUT/r2l_snail/weights/4.pth.tar}"

if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  echo "=== preprocess ==="
  python dataset/Hercules/gen_info.py
  python dataset/Hercules/gen_index.py
  python dataset/Hercules/gen_bev.py
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "=== cross-modal fine-tune (1 epoch, init=$RLPR_SNAIL_CROSS_CHECKPOINT) ==="
  export RLPR_TRAINING_ROOT="$OUT/r2l_hercules" RLPR_RESUME=0
  python train_cross_modal.py --dataset hercules
fi

echo "=== eval (epoch 1) ==="
export RLPR_TRAINING_ROOT="$OUT/r2l_hercules" RLPR_RESUME_EPOCH=1
python test_cross_modal.py --dataset hercules
