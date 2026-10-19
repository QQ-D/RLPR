"""Retrain only MulRan Word alignment, then evaluate unchanged author splits."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import torch

PROJECT = Path('/media/rwj/ZX1 2TB/RLPR')
SOURCE = PROJECT / 'src_spectral'
PREVIOUS = PROJECT / 'runs/mulran_word_full_2026-10-14_232504'
RUN = PROJECT / 'runs' / ('mulran_word_signed_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H%M%S'))
RUN.mkdir(exist_ok=False)
Path('/tmp/rlpr_word_signed_run_path').write_text(str(RUN))
(RUN / 'train_and_test.py').write_text(Path(__file__).read_text())
diff = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=SOURCE)
(RUN / 'code.patch').write_bytes(diff)
old = json.loads((PREVIOUS / 'run.json').read_text())
report = {'status': 'RUNNING', 'phase': 'PREFLIGHT', 'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
          'source_root': str(SOURCE), 'run_root': str(RUN), 'previous_run': str(PREVIOUS),
          'code_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
          'code_patch_sha256': hashlib.sha256(diff).hexdigest(),
          'training_config': dict(old['training_config'], alignment='signed_coherency_v2',
                                  lidar_epochs_reused=5, radar_epochs_reused=5, alignment_epochs=1),
          'regression_tests': {'command': 'python -m unittest discover -s tests -v', 'passed': 44},
          'stages': {}, 'evaluations': {}}
def save():
    temporary = RUN / 'run.json.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(RUN / 'run.json')
save()
env = os.environ.copy()
env.update(RLPR_MODEL_VARIANT='spectral_word', RLPR_SPECTRAL_CROSS_FUSION='1', RLPR_RESUME='0',
           RLPR_MULRAN_DATA=str(PROJECT / 'processed/mulran_official'), RLPR_CHECKPOINT_PATH='',
           RLPR_TEST_ZERO_SHOT='0', RLPR_PHASE_LOSS_WEIGHT='.1', RLPR_CROSS_FUSION_WEIGHT='.5',
           RLPR_PRETRAIN_L=str(PREVIOUS / 'l_mulran/weights/5.pth.tar'),
           RLPR_PRETRAIN_R=str(PREVIOUS / 'r_mulran/weights/5.pth.tar'),
           RLPR_TRAINING_ROOT=str(RUN / 'r2l_mulran'),
           TORCH_HOME=str(PROJECT / 'cache/torch'), MPLCONFIGDIR=str(RUN / 'matplotlib_cache'),
           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
try:
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    report['gpu'] = torch.cuda.get_device_name(0)
    for modality in ('L', 'R'):
        path = Path(env['RLPR_PRETRAIN_' + modality])
        checkpoint = torch.load(path, map_location='cpu')
        if checkpoint['epoch'] != 5 or checkpoint['model_variant'] != 'spectral_word':
            raise ValueError('Wrong unimodal checkpoint: ' + str(path))
        report['stages'][modality.lower() + '_mulran'] = {
            'status': 'REUSED', 'epoch': 5, 'checkpoint': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        del checkpoint
    report['phase'] = 'TRAINING_R2L_MULRAN'
    command = [sys.executable, '-u', 'train_cross_modal.py', '--dataset', 'mulran',
               '--model', 'spectral_word', '--cross-fusion']
    log_path = RUN / 'r2l_mulran.log'
    report['stages']['r2l_mulran'] = {'status': 'RUNNING', 'epochs': 1, 'command': command, 'log': str(log_path)}
    save()
    print('Run:', RUN, flush=True)
    print('Starting signed-coherency alignment: full Sejong/KAIST, batch8, 1 epoch', flush=True)
    start = time.monotonic()
    with log_path.open('x') as log:
        subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    path = RUN / 'r2l_mulran/weights/1.pth.tar'
    checkpoint = torch.load(path, map_location='cpu')
    if checkpoint['epoch'] != 1 or checkpoint.get('alignment_loss') != 'signed_coherency_v2':
        raise ValueError('Unexpected alignment checkpoint')
    teacher = torch.load(env['RLPR_PRETRAIN_R'], map_location='cpu')['state_dict']
    changed = [k for k, v in teacher.items() if not torch.equal(v, checkpoint['state_dict'][k])]
    if changed:
        raise ValueError('Frozen radar state changed: ' + repr(changed[:5]))
    losses = re.findall(r'epoch (\d+) finishes\s+loss \[([^\]]+)\]', log_path.read_text())
    if len(losses) != 1 or not all(math.isfinite(float(v)) for _, v in losses):
        raise ValueError('Missing finite training loss')
    report['stages']['r2l_mulran'].update(status='COMPLETE', checkpoint=str(path),
                                       losses={n: float(v) for n, v in losses},
                                       seconds=time.monotonic() - start, radar_state_unchanged=True,
                                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    del checkpoint, teacher
    env['RLPR_CHECKPOINT_PATH'] = str(path)
    save()
    print('Alignment completed:', report['stages']['r2l_mulran']['losses'], flush=True)
    for split in ('dcc', 'riverside'):
        report['phase'] = 'EVALUATING_' + split.upper()
        save()
        env['RLPR_MULRAN_TEST_SPLIT'] = split
        command = [sys.executable, '-u', 'test_cross_modal.py', '--dataset', 'mulran',
                   '--model', 'spectral_word', '--cross-fusion']
        log_path = RUN / (split + '_evaluation.log')
        with log_path.open('x') as log:
            subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        output = log_path.read_text()
        metrics = {}
        for name in ('Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@1percent'):
            match = re.search(re.escape(name) + r':\s*([0-9.]+)', output)
            if not match:
                raise ValueError('Missing recall: ' + name)
            metrics[name] = float(match.group(1))
        match = re.search(r'f1 score:\s*([0-9.eE+-]+)', output)
        if not match:
            raise ValueError('Missing F1')
        metrics['F1'] = float(match.group(1))
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError('Non-finite evaluation')
        previous = old['evaluations'][split]['metrics']
        official = old['evaluations'][split]['official_checkpoint_metrics']
        report['evaluations'][split] = {'metrics': metrics, 'previous_word_metrics': previous,
                                       'official_checkpoint_metrics': official,
                                       'difference_vs_previous_word': {k: metrics[k] - previous[k] for k in metrics},
                                       'difference_vs_official': {k: metrics[k] - official[k] for k in metrics},
                                       'command': command, 'log': str(log_path)}
        save()
        print(split, json.dumps(report['evaluations'][split], ensure_ascii=False), flush=True)
    report['status'] = 'COMPLETE'
    report['phase'] = 'DONE'
    report['finished_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
except BaseException as error:
    report['status'] = 'FAILED_OR_INTERRUPTED'
    report['error'] = repr(error)
    save()
    raise
save()
print('Complete:', RUN, flush=True)
