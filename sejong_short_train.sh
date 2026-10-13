#!/usr/bin/env bash
# Reproduction smoke run using the existing RLPR trainers and Sejong training regions.
set -euo pipefail

RLPR_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/rwj/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/data300/envs/rlpr
cd "$RLPR_PROJECT_ROOT/src"

export RLPR_PROJECT_ROOT
export RLPR_MULRAN_DATA="${RLPR_MULRAN_DATA:-$RLPR_PROJECT_ROOT/processed/mulran_sejong}"
export TORCH_HOME="${TORCH_HOME:-$RLPR_PROJECT_ROOT/cache/torch}"
export RLPR_SMOKE_ROOT="${RLPR_SMOKE_ROOT:-$RLPR_PROJECT_ROOT/runs/sejong_short_train}"

python -u - <<'PY'
import os
from pathlib import Path

import numpy as np

from config.params_full import ModelParamsFull
from dataset.MulRan.gen_bev import gen_bev
from dataset.MulRan.gen_index import process_infos_sejong
from dataset.MulRan.mulran_dataloader import make_dataloader
from scripts.registry import CrossPolar, LidarPolar, RadarPolar
from trainer.cross_modal import CrossModalTrainer
from trainer.unimodal import UnimodalTrainer
from utils.seed import set_global_seed

data_root = Path(os.environ['RLPR_MULRAN_DATA'])
run_root = Path(os.environ['RLPR_SMOKE_ROOT'])
info_file = data_root / 'info/mulran_infos-sejong.pkl'
index_root = run_root / 'indices'
bev_root = run_root / 'data/bev'
epochs = int(os.environ.get('RLPR_SMOKE_EPOCHS', '2'))
db_count = int(os.environ.get('RLPR_SMOKE_DB', '1024'))
query_count = int(os.environ.get('RLPR_SMOKE_QUERY', '256'))
assert info_file.is_file(), f'First complete Sejong preprocessing: {info_file}'
assert epochs > 0 and db_count >= 16 and query_count > 0
index_root.mkdir(parents=True, exist_ok=True)

# Keep the original geographic training/test split. Sample only training rows.
_, train_db, train_query, _, _ = process_infos_sejong(str(info_file), 'sejong02', 9.0)
training_rows = np.concatenate([train_db, train_query], axis=0)
rng = np.random.default_rng(0)
selected = np.sort(rng.choice(len(training_rows), min(db_count, len(training_rows)), replace=False))
database = training_rows[selected]
selected_queries = np.sort(rng.choice(len(database), min(query_count, len(database)), replace=False))
queries = database[selected_queries]
db_file = index_root / 'sejong_db.npy'
query_file = index_root / 'sejong_train_query.npy'
np.save(db_file, database)
np.save(query_file, queries)
print(f'\nShort train: database={len(database)}, queries={len(queries)}, epochs={epochs}, batch_size=1', flush=True)
gen_bev(str(bev_root), str(info_file), database)

def parameters(stage, lidar=True, radar=True):
    os.environ['RLPR_TRAINING_ROOT'] = str(run_root / stage)
    params = ModelParamsFull()
    params.resume_checkpoint = False
    params.epochs = epochs
    params.batch_size = 1
    params.test_batch_size = 1
    params.num_workers = 1
    params.l_enable = lidar
    params.r_enable = radar
    dp = params.mulran_param
    dp.train_info_root = [str(info_file)]
    dp.train_bev_dataset_root = str(bev_root)
    dp.train_database_index_list = [str(db_file)]
    dp.train_query_index_list = [str(query_file)]
    return params

for modality, stage, model in [('l', 'l_mulran', LidarPolar), ('r', 'r_mulran', RadarPolar)]:
    print(f'\n===== {stage}: {epochs} epochs =====', flush=True)
    set_global_seed(0)
    params = parameters(stage, lidar=modality == 'l', radar=modality == 'r')
    trainer = UnimodalTrainer(params, model, make_dataloader, modality, 'mulran', hard_mining_from_epoch=0)
    trainer.do_train()
    trainer.writer.close()
    del trainer

print(f'\n===== cross-modal: {epochs} epochs =====', flush=True)
set_global_seed(0)
params = parameters('r2l_mulran')
params.pretrain_l_path = str(run_root / f'l_mulran/weights/{epochs}.pth.tar')
params.pretrain_r_path = str(run_root / f'r_mulran/weights/{epochs}.pth.tar')
trainer = CrossModalTrainer(params, CrossPolar, make_dataloader)
trainer.do_train()
trainer.writer.close()
print(f'\nTraining complete. Checkpoint: {run_root / f"r2l_mulran/weights/{epochs}.pth.tar"}', flush=True)
PY
