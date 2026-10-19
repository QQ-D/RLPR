#!/usr/bin/env bash
# Boreas: preprocess -> eval (test-only; uses MulRan cross-modal weights)
# Usage: RLPR_BOREAS_RAW=/path/to/Boreas RLPR_BOREAS_DATA=/path/to/data ./scripts/boreas.sh
# Requires MulRan cross-modal checkpoint (run ./scripts/mulran.sh first, or set RLPR_TRAINING_ROOT).
# Optional: SKIP_PREP=1  RLPR_BOREAS_TEST_SPLIT=bor-clear|bor-snow
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RLPR_BOREAS_RAW="${RLPR_BOREAS_RAW:-/path/to/Boreas}"
export RLPR_BOREAS_DATA="${RLPR_BOREAS_DATA:-$HOME/RLPR_data/boreas}"
export RLPR_BOREAS_TEST_SPLIT="${RLPR_BOREAS_TEST_SPLIT:-bor-clear}"
OUT="${RLPR_OUTPUT_ROOT:-./outputs}"

if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  echo "=== preprocess ==="
  python dataset/Boreas/gen_info.py
  python dataset/Boreas/gen_index.py
  python dataset/Boreas/gen_bev.py
fi

echo "=== eval (MulRan cross-modal epoch 1, split=$RLPR_BOREAS_TEST_SPLIT) ==="
export RLPR_TRAINING_ROOT="${RLPR_TRAINING_ROOT:-$OUT/r2l_mulran}" RLPR_RESUME_EPOCH=1
python test_cross_modal.py --dataset boreas
