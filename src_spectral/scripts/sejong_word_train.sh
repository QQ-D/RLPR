#!/usr/bin/env bash
# Full Word architecture: retrain both teachers, then coherence alignment.
set -euo pipefail
RLPR_SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RLPR_PROJECT_ROOT="$(cd -- "$RLPR_SOURCE_ROOT/.." && pwd)"
cd "$RLPR_SOURCE_ROOT"
source /home/rwj/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/data300/envs/rlpr
export RLPR_MODEL_VARIANT=spectral_word
export RLPR_SPECTRAL_CROSS_FUSION=1
export RLPR_MULRAN_DATA="${RLPR_MULRAN_DATA:-$RLPR_PROJECT_ROOT/processed/mulran_sejong}"
export TORCH_HOME="${TORCH_HOME:-$RLPR_PROJECT_ROOT/cache/torch}"
export RLPR_WORD_INPUT_ROOT="${RLPR_WORD_INPUT_ROOT:-$RLPR_PROJECT_ROOT/runs/sejong_spectral_2026-10-14_202056}"
export RLPR_WORD_RUN_ROOT="${RLPR_WORD_RUN_ROOT:-$RLPR_PROJECT_ROOT/runs/sejong_word_$(TZ=Asia/Shanghai date +%Y-%m-%d_%H%M%S)}"
mkdir -p "$(dirname -- "$RLPR_WORD_RUN_ROOT")"
mkdir "$RLPR_WORD_RUN_ROOT"

python -u - <<'PY' 2>&1 | tee "$RLPR_WORD_RUN_ROOT/run.log"
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from config.params_full import ModelParamsFull
from dataset.MulRan.mulran_dataloader import make_dataloader
from scripts.registry import CrossPolar, LidarPolar, RadarPolar, get_model_variant
from trainer.cross_modal import CrossModalTrainer
from trainer.unimodal import UnimodalTrainer
from utils.seed import set_global_seed

run_root = Path(os.environ['RLPR_WORD_RUN_ROOT'])
input_root = Path(os.environ['RLPR_WORD_INPUT_ROOT'])
data_root = Path(os.environ['RLPR_MULRAN_DATA'])
metadata = {
    'status': 'RUNNING', 'model_variant': 'spectral_word',
    'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'code_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
    'epochs_per_stage': int(os.environ.get('RLPR_WORD_EPOCHS', '2')),
    'input_root': str(input_root), 'data_root': str(data_root), 'seed': 0, 'batch_size': 1,
    'cross_fusion_enabled': True, 'measured_amplitude': 'discarded',
    'alignment': 'welch_coherence_v1', 'phase_loss_weight': float(os.environ.get('RLPR_PHASE_LOSS_WEIGHT', '.1')),
    'cross_fusion_loss_weight': float(os.environ.get('RLPR_CROSS_FUSION_WEIGHT', '.5')),
    'training_scope': 'Same existing Sejong training indices; geographic test regions excluded',
    'teacher_policy': 'Retrain both unimodal branches with spectral_word; do not load old spectral weights',
    'test_policy': 'Independent radar and LiDAR descriptors', 'torch': torch.__version__,
}
def save_status(status):
    metadata['status'] = status
    (run_root / 'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
save_status('RUNNING')
trainer = None
try:
    epochs = metadata['epochs_per_stage']
    if epochs < 1:
        raise ValueError('RLPR_WORD_EPOCHS must be positive')
    info = data_root / 'info/mulran_infos-sejong.pkl'
    bev = input_root / 'data/bev'
    if not info.is_file() or not bev.is_dir():
        raise FileNotFoundError('Existing Sejong info/BEV required; check RLPR_WORD_INPUT_ROOT and RLPR_MULRAN_DATA')
    indices = run_root / 'indices'
    indices.mkdir()
    for name in ('sejong_db.npy', 'sejong_train_query.npy'):
        values = np.load(input_root / 'indices' / name, allow_pickle=False)
        np.save(indices / name, values)
        metadata['database_frames' if name == 'sejong_db.npy' else 'query_frames'] = len(values)
    save_status('RUNNING')
    print(f"Word train: database={metadata['database_frames']}, queries={metadata['query_frames']}, epochs={epochs} per stage", flush=True)

    def parameters(stage, lidar=True, radar=True):
        os.environ['RLPR_TRAINING_ROOT'] = str(run_root / stage)
        params = ModelParamsFull()
        params.resume_checkpoint = False
        params.epochs = epochs
        params.batch_size = params.test_batch_size = 1
        params.num_workers = 1
        params.l_enable, params.r_enable = lidar, radar
        dp = params.mulran_param
        dp.train_info_root = [str(info)]
        dp.train_bev_dataset_root = str(bev)
        dp.train_database_index_list = [str(indices / 'sejong_db.npy')]
        dp.train_query_index_list = [str(indices / 'sejong_train_query.npy')]
        return params

    for modality, stage, model in [('l', 'l_mulran', LidarPolar), ('r', 'r_mulran', RadarPolar)]:
        print(f'\n===== Word {stage}: {epochs} epochs =====', flush=True)
        set_global_seed(0)
        params = parameters(stage, lidar=modality == 'l', radar=modality == 'r')
        trainer = UnimodalTrainer(params, get_model_variant(model), make_dataloader,
                                  modality, 'mulran', hard_mining_from_epoch=0)
        try:
            trainer.do_train()
        finally:
            trainer.writer.close()
            trainer = None
    print(f'\n===== Word cross-modal coherence: {epochs} epochs =====', flush=True)
    set_global_seed(0)
    params = parameters('r2l_mulran')
    params.pretrain_l_path = str(run_root / f'l_mulran/weights/{epochs}.pth.tar')
    params.pretrain_r_path = str(run_root / f'r_mulran/weights/{epochs}.pth.tar')
    trainer = CrossModalTrainer(params, get_model_variant(CrossPolar), make_dataloader)
    trainer.do_train()
    metadata['checkpoint'] = str(run_root / f'r2l_mulran/weights/{epochs}.pth.tar')
    metadata['spectral_config'] = trainer.model.spectral_config
    save_status('COMPLETE')
    print(f"\nTraining complete. Checkpoint: {metadata['checkpoint']}", flush=True)
except BaseException:
    save_status('FAILED_OR_INTERRUPTED')
    raise
finally:
    if trainer is not None:
        trainer.writer.close()
PY
