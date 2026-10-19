"""Five-epoch alignment experiment with validation isolated from training RNG."""
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PROJECT = Path('/media/rwj/ZX1 2TB/RLPR')
SOURCE = PROJECT / 'src_spectral'
INIT = PROJECT / 'runs/mulran_word_full_2026-10-14_232504'
CONTROL = PROJECT / 'runs/mulran_word_ablations_2026-10-15_120339/control'
TRAIN_DATA = PROJECT / 'processed/mulran_official'
VALIDATION_DATA = PROJECT / 'processed/mulran_sejong'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def train_epoch(run, epoch):
    import torch
    sys.path.insert(0, str(SOURCE))
    from scripts.registry import build_params, CROSS_MODAL, get_model_variant
    from trainer.cross_modal import CrossModalTrainer
    from utils.seed import set_global_seed
    set_global_seed(0)
    torch.set_num_threads(4)
    cfg = CROSS_MODAL['mulran']
    trainer = CrossModalTrainer(build_params(cfg['epochs'], epochs_override=epoch),
                               get_model_variant(cfg['model'], 'spectral_word'), cfg['dataloader'])
    try:
        if epoch > 1:
            previous = torch.load(Path(run) / 'r2l_mulran/weights' / f'{epoch - 1}.pth.tar', map_location='cpu')
            if previous.get('training_state') is None or previous.get('epoch') != epoch - 1:
                raise ValueError('Exact resume needs the previous epoch and complete random state')
            del previous
        trainer.do_train()
        write_json(Path(run) / f'epoch_{epoch}_sampling.json', {
            'queries': len(trainer.train_loader.dataset), 'iterations': len(trainer.train_loader),
            'sampler_sha256': hashlib.sha256(trainer.train_loader.sampler.generator.get_state().numpy().tobytes()).hexdigest(),
            'worker_sha256': hashlib.sha256(trainer.train_loader.generator.get_state().numpy().tobytes()).hexdigest(),
        })
    finally:
        trainer.writer.close()


def run_all():
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    torch.set_num_threads(4)
    run = PROJECT / 'runs' / ('mulran_word_longer_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H%M%S'))
    run.mkdir(exist_ok=False)
    driver = run / 'run_experiment.py'
    driver.write_text(Path(__file__).read_text())
    Path('/tmp/rlpr_longer_run_path').write_text(str(run))
    (run / 'code.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=SOURCE))
    source_hashes = {str(p.relative_to(SOURCE)): digest(p) for p in SOURCE.rglob('*.py')
                     if '.git' not in p.parts and 'outputs' not in p.parts}
    protocol = dict(created_at=now(), epochs=list(range(1, 6)), batch_size=8, seed=0,
                    sampling_seed=0, initial_learning_rate=5e-5, scheduler_gamma=.8,
                    alignment_loss='signed_coherency_v2', phase_loss_weight=.1, cross_fusion_loss_weight=.5,
                    train_data=str(TRAIN_DATA), validation_data=str(VALIDATION_DATA),
                    pretrain_l=str(INIT / 'l_mulran/weights/5.pth.tar'),
                    pretrain_r=str(INIT / 'r_mulran/weights/5.pth.tar'),
                    epoch1_reference=str(CONTROL / 'r2l_mulran/weights/1.pth.tar'),
                    deterministic_algorithms=True, cublas_workspace_config=':4096:8',
                    historical_epoch1_comparison='Informational only: old CuBLAS/SDPA policy did not guarantee bitwise reproducibility; compare epochs within this strict run',
                    selection='Maximum Sejong Recall@1, then F1, then earliest epoch; lock before benchmarks',
                    benchmark_epochs='Unique set of epoch1, selected epoch and epoch5; no benchmark-based reselection',
                    benchmark_history='DCC/Riverside have been inspected before; exploratory comparison',
                    random_state='Fresh epoch1; resume complete epoch-boundary RNG, optimizer and scheduler for epochs2..5',
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
                    source_sha256=source_hashes,
                    tests={'command': 'python -m unittest discover -s tests -v', 'passed': 54})
    report = dict(status='RUNNING', phase='PREFLIGHT', started_at=now(), run_root=str(run), epochs={})
    def save():
        write_json(run / 'run.json', report)
    def verify_source():
        for name, expected in source_hashes.items():
            if digest(SOURCE / name) != expected:
                raise ValueError('Runtime source changed: ' + name)
    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('GPU is required')
        report['gpu'] = torch.cuda.get_device_name(0)
        for modality in ('l', 'r'):
            p = Path(protocol['pretrain_' + modality])
            c = torch.load(p, map_location='cpu')
            if c['epoch'] != 5 or c['model_variant'] != 'spectral_word':
                raise ValueError('Wrong unimodal initialization')
            protocol['pretrain_' + modality + '_sha256'] = digest(p)
        train = np.load(TRAIN_DATA / 'index/sejong_train_query.npy')
        held = np.concatenate([np.load(VALIDATION_DATA / 'index' / name)
                               for name in ('sejong_test_db.npy', 'sejong_test_query.npy')])
        if np.intersect1d(train[:, 0], held[:, 0]).size:
            raise ValueError('Train/validation frame overlap')
        minimum = float(cKDTree(train[:, 1:]).query(held[:, 1:])[0].min())
        if minimum < 50.:
            raise ValueError('Sejong geographic buffer below50m')
        for name in ('sejong_test_db.npy', 'sejong_test_query.npy'):
            if not np.array_equal(np.load(TRAIN_DATA / 'index' / name), np.load(VALIDATION_DATA / 'index' / name)):
                raise ValueError('Validation index mismatch')
        protocol['nearest_train_validation_m'] = minimum
        write_json(run / 'protocol.json', protocol)
        report['protocol_sha256'] = digest(run / 'protocol.json')
        base = os.environ.copy()
        base.update(RLPR_MODEL_VARIANT='spectral_word', RLPR_SAMPLING_SEED='0', RLPR_DETERMINISTIC='1',
                    RLPR_SPECTRAL_CROSS_FUSION='1', RLPR_CROSS_FUSION_WEIGHT='.5', RLPR_PHASE_LOSS_WEIGHT='.1',
                    RLPR_WORD_ALIGNMENT_LOSS='signed_coherency_v2', RLPR_CHECKPOINT_PATH='', RLPR_TEST_ZERO_SHOT='0',
                    RLPR_PRETRAIN_L=protocol['pretrain_l'], RLPR_PRETRAIN_R=protocol['pretrain_r'],
                    RLPR_TRAINING_ROOT=str(run / 'r2l_mulran'), RLPR_MULRAN_DATA=str(TRAIN_DATA),
                    TORCH_HOME=str(PROJECT / 'cache/torch'), MPLCONFIGDIR=str(run / 'matplotlib_cache'),
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        def evaluate(epoch, split):
            verify_source()
            env = base.copy()
            env.update(RLPR_MULRAN_DATA=str(VALIDATION_DATA if split == 'sejong' else TRAIN_DATA),
                       RLPR_MULRAN_TEST_SPLIT=split, RLPR_CHECKPOINT_PATH=str(run / f'r2l_mulran/weights/{epoch}.pth.tar'))
            command = [sys.executable, '-u', 'test_cross_modal.py', '--dataset', 'mulran', '--model', 'spectral_word', '--cross-fusion']
            logpath = run / f'epoch_{epoch}_{split}.log'
            with logpath.open('x') as log:
                subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            output = logpath.read_text()
            metrics = {}
            for name in ('Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@1percent', 'F1'):
                label = 'f1 score' if name == 'F1' else re.escape(name)
                match = re.search(label + r':\s*([0-9.eE+-]+)', output)
                if match is None or not math.isfinite(float(match.group(1))):
                    raise ValueError('Missing finite metric: ' + name)
                metrics[name] = float(match.group(1))
            return dict(metrics=metrics, log=str(logpath), completed_at=now())
        save()
        print('Run:', run, flush=True)
        for epoch in protocol['epochs']:
            verify_source()
            report['phase'] = f'TRAINING_EPOCH_{epoch}'
            save()
            env = base.copy()
            env.update(RLPR_RESUME='0' if epoch == 1 else '1', RLPR_RESUME_EPOCH=str(epoch - 1))
            logpath = run / f'epoch_{epoch}_train.log'
            started = time.monotonic()
            print('Training epoch', epoch, flush=True)
            with logpath.open('x') as log:
                subprocess.run([sys.executable, '-u', str(driver), 'train', str(run), str(epoch)],
                               cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            path = run / f'r2l_mulran/weights/{epoch}.pth.tar'
            c = torch.load(path, map_location='cpu')
            if c['epoch'] != epoch or c['alignment_loss'] != protocol['alignment_loss'] or c.get('training_state') is None:
                raise ValueError('Wrong epoch checkpoint')
            if not all(torch.isfinite(v).all() for v in c['state_dict'].values()):
                raise ValueError('Non-finite model')
            radar = torch.load(protocol['pretrain_r'], map_location='cpu')['state_dict']
            if any(not torch.equal(v, c['state_dict'][key]) for key, v in radar.items()):
                raise ValueError('Frozen teacher changed')
            student = torch.load(protocol['pretrain_l'], map_location='cpu')['state_dict']
            changed = sum(not torch.equal(v, c['state_dict'][key]) for key, v in student.items())
            if not changed:
                raise ValueError('Student did not update')
            lr = c['optimizer']['param_groups'][0]['lr']
            if not math.isclose(lr, 5e-5 * .8 ** epoch, rel_tol=1e-12):
                raise ValueError('Wrong learning rate schedule')
            steps = {int(s['step']) for s in c['optimizer']['state'].values() if 'step' in s}
            if steps != {2071 * epoch}:
                raise ValueError('Wrong optimizer update count: ' + str(steps))
            matches = re.findall(r'epoch (\d+) finishes\s+loss \[([^\]]+)\]', logpath.read_text())
            if len(matches) != 1 or int(matches[0][0]) != epoch or not math.isfinite(float(matches[0][1])):
                raise ValueError('Expected one finite training epoch')
            result = dict(checkpoint=str(path), checkpoint_sha256=digest(path), loss=float(matches[0][1]),
                          seconds=time.monotonic() - started, learning_rate_next=lr, optimizer_steps=2071 * epoch,
                          radar_unchanged=True, student_changed_tensors=changed,
                          sampling=json.loads((run / f'epoch_{epoch}_sampling.json').read_text()))
            if epoch == 1:
                reference = torch.load(protocol['epoch1_reference'], map_location='cpu')['state_dict']
                maxdiff = max(float((v - reference[key]).abs().max()) for key, v in c['state_dict'].items())
                result['epoch1_control_max_abs_diff'] = maxdiff
                result['epoch1_control_bitwise_identical'] = maxdiff == 0
            del c, radar, student
            report['epochs'][str(epoch)] = result
            report['phase'] = f'VALIDATING_EPOCH_{epoch}'
            save()
            result['validation'] = evaluate(epoch, 'sejong')
            save()
            print('Epoch', epoch, 'loss', result['loss'], 'validation', result['validation']['metrics'], flush=True)
        winner = max(protocol['epochs'], key=lambda e: (report['epochs'][str(e)]['validation']['metrics']['Recall@1'],
                                                       report['epochs'][str(e)]['validation']['metrics']['F1'], -e))
        selection = dict(selected_epoch=winner, locked_at=now(), rule=protocol['selection'],
                         benchmark_results_consulted_in_this_run=False,
                         validation={k: v['validation']['metrics'] for k, v in report['epochs'].items()})
        write_json(run / 'selection.json', selection)
        report['selection'] = selection
        report['selection_sha256'] = digest(run / 'selection.json')
        save()
        for epoch in sorted({1, winner, 5}):
            report['phase'] = f'BENCHMARKING_EPOCH_{epoch}'
            save()
            result = report['epochs'][str(epoch)]
            result['benchmarks'] = {}
            for split in ('dcc', 'riverside'):
                result['benchmarks'][split] = evaluate(epoch, split)
                save()
                print('Benchmark', epoch, split, result['benchmarks'][split]['metrics'], flush=True)
        fields = ['epoch', 'split', 'loss', 'selected', 'Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@1percent', 'F1']
        with (run / 'comparison.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for key, result in report['epochs'].items():
                for split, evaluation in dict(sejong=result['validation'], **result.get('benchmarks', {})).items():
                    writer.writerow(dict(epoch=int(key), split=split, loss=result['loss'], selected=int(key) == winner, **evaluation['metrics']))
        report.update(status='COMPLETE', phase='COMPLETE', completed_at=now())
        save()
        print('COMPLETE:', run, flush=True)
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc), failed_at=now())
        save()
        raise


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_epoch(sys.argv[2], int(sys.argv[3]))
    else:
        run_all()
