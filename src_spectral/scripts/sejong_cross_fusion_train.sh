#!/usr/bin/env bash
# Reuse spectral unimodal checkpoints and indices; train only stage two.
set -euo pipefail

RLPR_SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RLPR_PROJECT_ROOT="$(cd -- "$RLPR_SOURCE_ROOT/.." && pwd)"
# Re-enter the current mount before Conda calls os.getcwd().
cd "$RLPR_SOURCE_ROOT"
source /home/rwj/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/data300/envs/rlpr

export RLPR_PROJECT_ROOT
export RLPR_SPECTRAL_CROSS_FUSION=1
export TORCH_HOME="${TORCH_HOME:-$RLPR_PROJECT_ROOT/cache/torch}"
export RLPR_MULRAN_DATA="${RLPR_MULRAN_DATA:-$RLPR_PROJECT_ROOT/processed/mulran_sejong}"
export RLPR_CROSS_INIT_ROOT="${RLPR_CROSS_INIT_ROOT:-$RLPR_PROJECT_ROOT/runs/sejong_spectral_2026-10-14_202056}"
export RLPR_CROSS_RUN_ROOT="${RLPR_CROSS_RUN_ROOT:-$RLPR_PROJECT_ROOT/runs/sejong_cross_fusion_$(TZ=Asia/Shanghai date +%Y-%m-%d_%H%M%S)}"
mkdir -p "$(dirname -- "$RLPR_CROSS_RUN_ROOT")"
mkdir "$RLPR_CROSS_RUN_ROOT"
export RLPR_TRAINING_ROOT="$RLPR_CROSS_RUN_ROOT/r2l_mulran"

python -u - <<'PY' 2>&1 | tee "$RLPR_CROSS_RUN_ROOT/run.log"
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from config.params_full import ModelParamsFull
from dataset.MulRan.mulran_dataloader import make_dataloader
from scripts.registry import CrossPolar, get_model_variant
from trainer.cross_modal import CrossModalTrainer
from utils.seed import set_global_seed

init_root = Path(os.environ['RLPR_CROSS_INIT_ROOT'])
run_root = Path(os.environ['RLPR_CROSS_RUN_ROOT'])
data_root = Path(os.environ['RLPR_MULRAN_DATA'])
metadata = {
    'status': 'RUNNING', 'task': 'stage-two phase cross-fusion training',
    'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'code_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
    'init_root': str(init_root), 'seed': 0, 'batch_size': 1,
    'cross_fusion_enabled': True, 'epochs': int(os.environ.get('RLPR_CROSS_EPOCHS', '2')),
}
def save_status(status):
    metadata['status'] = status
    (run_root / 'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
save_status('RUNNING')
trainer = None
try:
    set_global_seed(0)
    params = ModelParamsFull()
    params.resume_checkpoint = False
    params.epochs = metadata['epochs']
    if params.epochs < 1:
        raise ValueError('RLPR_CROSS_EPOCHS must be positive')
    params.batch_size = 1
    params.test_batch_size = 1
    params.num_workers = 1
    params.pretrain_l_path = str(init_root / 'l_mulran/weights/2.pth.tar')
    params.pretrain_r_path = str(init_root / 'r_mulran/weights/2.pth.tar')
    dp = params.mulran_param
    dp.train_info_root = [str(data_root / 'info/mulran_infos-sejong.pkl')]
    dp.train_bev_dataset_root = str(init_root / 'data/bev')
    dp.train_database_index_list = [str(init_root / 'indices/sejong_db.npy')]
    dp.train_query_index_list = [str(init_root / 'indices/sejong_train_query.npy')]
    for path in [params.pretrain_l_path, params.pretrain_r_path, *dp.train_info_root,
                 *dp.train_database_index_list, *dp.train_query_index_list]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    metadata.update({
        'database_frames': len(np.load(dp.train_database_index_list[0])),
        'query_frames': len(np.load(dp.train_query_index_list[0])),
        'pretrain_l': params.pretrain_l_path, 'pretrain_r': params.pretrain_r_path,
        'phase_loss_weight': params.phase_loss_weight,
        'cross_fusion_loss_weight': params.cross_fusion_loss_weight,
        'scope': 'Existing Sejong training indices; no test-region data used for training',
        'test_policy': 'Independent descriptors; paired fusion used only during training',
    })
    save_status('RUNNING')
    print(f"Cross-fusion train: database={metadata['database_frames']}, queries={metadata['query_frames']}, "
          f"epochs={params.epochs}, fusion_weight={params.cross_fusion_loss_weight}", flush=True)
    trainer = CrossModalTrainer(params, get_model_variant(CrossPolar, 'spectral'), make_dataloader)
    trainer.do_train()
    save_status('COMPLETE')
    print(f'Training complete. Checkpoint: {Path(params.weights_root) / f"{params.epochs}.pth.tar"}', flush=True)
except BaseException:
    save_status('FAILED_OR_INTERRUPTED')
    raise
finally:
    if trainer is not None:
        trainer.writer.close()
PY
