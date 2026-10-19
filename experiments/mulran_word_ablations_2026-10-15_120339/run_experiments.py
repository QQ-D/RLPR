"""Fixed four-arm MulRan ablation; choose on Sejong before benchmark evaluation."""
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PROJECT = Path('/media/rwj/ZX1 2TB/RLPR')
SOURCE = PROJECT / 'src_spectral'
INIT = PROJECT / 'runs/mulran_word_full_2026-10-14_232504'
HISTORICAL = PROJECT / 'runs/mulran_word_signed_2026-10-15_074007/run.json'
TRAIN_DATA = PROJECT / 'processed/mulran_official'
VALIDATION_DATA = PROJECT / 'processed/mulran_sejong'
ARMS = [
    dict(name='control', label='有符号相关度完整对照', fusion=True, phase_weight=.1, alignment='signed_coherency_v2'),
    dict(name='no_fusion', label='仅关闭交叉融合', fusion=False, phase_weight=.1, alignment='signed_coherency_v2'),
    dict(name='no_frequency_aux', label='仅关闭频域辅助损失', fusion=True, phase_weight=0., alignment='signed_coherency_v2'),
    dict(name='infonce', label='仅替换描述子损失为InfoNCE', fusion=True, phase_weight=.1, alignment='infonce'),
]

def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

def train_one(run, name):
    # Environment is set in the parent before any model/config imports.
    import torch
    sys.path.insert(0, str(SOURCE))
    from scripts.registry import build_params, CROSS_MODAL, get_model_variant
    from trainer.cross_modal import CrossModalTrainer
    from utils.seed import set_global_seed
    set_global_seed(0)
    torch.set_num_threads(4)
    params = build_params(1)
    cfg = CROSS_MODAL['mulran']
    trainer = CrossModalTrainer(params, get_model_variant(cfg['model'], 'spectral_word'), cfg['dataloader'])
    # Both random streams are independent of model construction. The no-fusion
    # architecture consumes fewer initialization draws but gets identical batches.
    trainer.train_loader.generator = torch.Generator().manual_seed(0)
    trainer.train_loader.sampler.generator = torch.Generator().manual_seed(0)
    try:
        trainer.do_train()
        state = {
            'sampler_seed': 0, 'worker_seed': 0,
            'sampler_final_state_sha256': hashlib.sha256(trainer.train_loader.sampler.generator.get_state().numpy().tobytes()).hexdigest(),
            'worker_final_state_sha256': hashlib.sha256(trainer.train_loader.generator.get_state().numpy().tobytes()).hexdigest(),
            'dataset_queries': len(trainer.train_loader.dataset), 'iterations': len(trainer.train_loader),
        }
        write_json(Path(run) / name / 'sampling.json', state)
    finally:
        trainer.writer.close()

def run_all():
    import numpy as np
    import torch
    from scipy.spatial import cKDTree

    run = PROJECT / 'runs' / ('mulran_word_ablations_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H%M%S'))
    run.mkdir(exist_ok=False)
    Path('/tmp/rlpr_word_ablation_run_path').write_text(str(run))
    driver = run / 'run_experiments.py'
    driver.write_text(Path(__file__).read_text())
    (run / 'code.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=SOURCE))
    protocol = {
        'created_at': now(), 'arms': ARMS, 'epochs_per_arm': 1, 'batch_size': 8,
        'seed': 0, 'sampler_seed': 0, 'worker_seed': 0, 'learning_rate': 5e-5,
        'epoch_lr_gamma': .8, 'cross_fusion_loss_weight': .5,
        'train_data': str(TRAIN_DATA), 'validation_data': str(VALIDATION_DATA),
        'validation_split': 'Sejong held-out geographic regions',
        'selection': 'Maximum Sejong Recall@1, then maximum Sejong F1, then earliest declared arm; lock before DCC/Riverside',
        'final_evaluation': 'Evaluate all four fixed configurations once on DCC and Riverside after selection; do not reselect on benchmark scores',
        'benchmark_history': 'DCC and Riverside were inspected in earlier work; this batch is exploratory, not a previously unseen benchmark',
        'control_retraining_reason': 'Separate sampler/worker generators make training batches identical across architectures; historical control used global RNG',
        'pretrain_l': str(INIT / 'l_mulran/weights/5.pth.tar'),
        'pretrain_r': str(INIT / 'r_mulran/weights/5.pth.tar'),
        'source_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
        'source_sha256': {p: digest(SOURCE / p) for p in ('trainer/cross_modal.py', 'config/params_full.py', 'modules/coherence.py', 'modules/word_models.py', 'tests/test_alignment_ablation.py')},
        'regression_tests': {'command': 'python -m unittest discover -s tests -v', 'passed': 48},
    }
    report = {'status': 'RUNNING', 'phase': 'PREFLIGHT', 'started_at': now(), 'run_root': str(run),
              'protocol': protocol, 'arms': {}, 'historical_signed_run': str(HISTORICAL)}
    def save():
        write_json(run / 'run.json', report)
    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required')
        report['gpu'] = torch.cuda.get_device_name(0)
        for modality in ('l', 'r'):
            path = Path(protocol['pretrain_' + modality])
            c = torch.load(path, map_location='cpu')
            if c['epoch'] != 5 or c['model_variant'] != 'spectral_word':
                raise ValueError('Wrong unimodal initialization')
            protocol['pretrain_' + modality + '_sha256'] = digest(path)
            del c
        train = np.load(TRAIN_DATA / 'index/sejong_train_query.npy')
        db = np.load(VALIDATION_DATA / 'index/sejong_test_db.npy')
        query = np.load(VALIDATION_DATA / 'index/sejong_test_query.npy')
        held = np.concatenate([db, query])
        if np.intersect1d(train[:, 0], held[:, 0]).size:
            raise ValueError('Validation frame leakage')
        minimum = float(cKDTree(train[:, 1:]).query(held[:, 1:])[0].min())
        if minimum < 50.:
            raise ValueError('Expected Sejong geographic buffer >=50m')
        for name in ('sejong_test_db.npy', 'sejong_test_query.npy'):
            if not np.array_equal(np.load(TRAIN_DATA / 'index' / name), np.load(VALIDATION_DATA / 'index' / name)):
                raise ValueError('Validation index mismatch')
        infos = pickle.loads((VALIDATION_DATA / 'info/mulran_infos-sejong.pkl').read_bytes())
        full_infos = pickle.loads((TRAIN_DATA / 'info/mulran_infos-sejong.pkl').read_bytes())
        for idx in held[:, 0].astype(int):
            for modality, key, sensor in [('lidar', 'lidar_infos', 'LIDAR_TOP'), ('radar', 'radar_infos', 'RADAR_TOP')]:
                filename = Path(infos[idx][key][sensor]['filename']).stem + '.png'
                if filename != Path(full_infos[idx][key][sensor]['filename']).stem + '.png':
                    raise ValueError('Validation info ordering mismatch')
                if not (VALIDATION_DATA / 'bev/mm_bev' / modality / filename).is_file():
                    raise FileNotFoundError(filename)
        protocol['data_verification'] = {'sejong_training_frames': len(train), 'validation_database_frames': len(db),
                                         'validation_query_frames': len(query), 'train_validation_frame_overlap': 0,
                                         'nearest_train_validation_m': minimum,
                                         'full_training_database_frames': len(train) + len(np.load(TRAIN_DATA / 'index/kaist_db.npy')),
                                         'full_training_query_frames': len(train) + len(np.load(TRAIN_DATA / 'index/kaist_train_query.npy'))}
        write_json(run / 'protocol.json', protocol)
        report['protocol_sha256'] = digest(run / 'protocol.json')
        save()
        print('Run:', run, flush=True)
        print('Fixed protocol: four arms, identical seeded batches, Sejong-only selection.', flush=True)
        base = os.environ.copy()
        base.update(RLPR_MODEL_VARIANT='spectral_word', RLPR_RESUME='0', RLPR_CHECKPOINT_PATH='',
                    RLPR_TEST_ZERO_SHOT='0', RLPR_CROSS_FUSION_WEIGHT='.5',
                    RLPR_PRETRAIN_L=protocol['pretrain_l'], RLPR_PRETRAIN_R=protocol['pretrain_r'],
                    TORCH_HOME=str(PROJECT / 'cache/torch'), MPLCONFIGDIR=str(run / 'matplotlib_cache'),
                    OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        def arm_env(arm):
            env = base.copy()
            env.update(RLPR_SPECTRAL_CROSS_FUSION='1' if arm['fusion'] else '0',
                       RLPR_PHASE_LOSS_WEIGHT=str(arm['phase_weight']), RLPR_WORD_ALIGNMENT_LOSS=arm['alignment'],
                       RLPR_TRAINING_ROOT=str(run / arm['name'] / 'r2l_mulran'))
            return env
        def evaluate(arm, split, validation=False):
            name = arm['name']
            env = arm_env(arm)
            env.update(RLPR_MULRAN_DATA=str(VALIDATION_DATA if validation else TRAIN_DATA),
                       RLPR_MULRAN_TEST_SPLIT=split,
                       RLPR_CHECKPOINT_PATH=report['arms'][name]['training']['checkpoint'])
            command = [sys.executable, '-u', 'test_cross_modal.py', '--dataset', 'mulran', '--model', 'spectral_word',
                       '--cross-fusion' if arm['fusion'] else '--no-cross-fusion']
            path = run / name / (split + ('_validation.log' if validation else '_evaluation.log'))
            with path.open('x') as log:
                subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            output = path.read_text()
            metrics = {}
            for metric in ('Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@1percent'):
                match = re.search(re.escape(metric) + r':\s*([0-9.]+)', output)
                if not match:
                    raise ValueError('Missing metric ' + metric)
                metrics[metric] = float(match.group(1))
            match = re.search(r'f1 score:\s*([0-9.eE+-]+)', output)
            if not match:
                raise ValueError('Missing F1')
            metrics['F1'] = float(match.group(1))
            if not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError('Non-finite evaluation')
            return {'metrics': metrics, 'log': str(path), 'command': command, 'completed_at': now()}
        for arm in ARMS:
            name = arm['name']
            arm_root = run / name
            arm_root.mkdir()
            report['phase'] = 'TRAINING_' + name.upper()
            command = [sys.executable, '-u', str(driver), 'train', str(run), name]
            log_path = arm_root / 'train.log'
            report['arms'][name] = {'config': arm, 'training': {'status': 'RUNNING', 'command': command, 'log': str(log_path)},
                                    'evaluations': {}}
            save()
            env = arm_env(arm)
            env['RLPR_MULRAN_DATA'] = str(TRAIN_DATA)
            print('Training:', name, flush=True)
            started = time.monotonic()
            with log_path.open('x') as log:
                subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            path = arm_root / 'r2l_mulran/weights/1.pth.tar'
            c = torch.load(path, map_location='cpu')
            if (c['epoch'] != 1 or c['alignment_loss'] != arm['alignment']
                    or c['cross_fusion_enabled'] != arm['fusion'] or c['phase_loss_weight'] != arm['phase_weight']):
                raise ValueError('Wrong checkpoint training configuration')
            if not all(torch.isfinite(v).all() for v in c['state_dict'].values()):
                raise ValueError('Non-finite checkpoint')
            teacher = torch.load(protocol['pretrain_r'], map_location='cpu')['state_dict']
            if any(not torch.equal(v, c['state_dict'][k]) for k, v in teacher.items()):
                raise ValueError('Frozen radar teacher changed')
            student = torch.load(protocol['pretrain_l'], map_location='cpu')['state_dict']
            changed = sum(not torch.equal(v, c['state_dict'][k]) for k, v in student.items())
            if not changed:
                raise ValueError('Student did not update')
            del c, teacher, student
            matches = re.findall(r'epoch (\d+) finishes\s+loss \[([^\]]+)\]', log_path.read_text())
            if len(matches) != 1 or not math.isfinite(float(matches[0][1])):
                raise ValueError('Missing finite training loss')
            sampling = json.loads((arm_root / 'sampling.json').read_text())
            report['arms'][name]['training'].update(status='COMPLETE', checkpoint=str(path), checkpoint_sha256=digest(path),
                                                   loss=float(matches[0][1]), seconds=time.monotonic() - started,
                                                   radar_state_unchanged=True, student_changed_tensors=changed, sampling=sampling)
            report['phase'] = 'VALIDATING_' + name.upper()
            save()
            report['arms'][name]['validation'] = evaluate(arm, 'sejong', validation=True)
            save()
            print('Sejong validation:', name, json.dumps(report['arms'][name]['validation']['metrics']), flush=True)
        states = [report['arms'][a['name']]['training']['sampling'] for a in ARMS]
        for field in ('sampler_final_state_sha256', 'worker_final_state_sha256', 'dataset_queries', 'iterations'):
            if len({s[field] for s in states}) != 1:
                raise ValueError('Training sampling mismatch: ' + field)
        winner = max(enumerate(ARMS), key=lambda pair: (
            report['arms'][pair[1]['name']]['validation']['metrics']['Recall@1'],
            report['arms'][pair[1]['name']]['validation']['metrics']['F1'], -pair[0]))[1]['name']
        selection = {'selected_arm': winner, 'locked_at': now(), 'rule': protocol['selection'],
                     'all_validation_metrics': {a['name']: report['arms'][a['name']]['validation']['metrics'] for a in ARMS},
                     'benchmark_results_consulted_in_this_batch': False}
        write_json(run / 'selection.json', selection)
        report['selection'] = selection
        report['selection_sha256'] = digest(run / 'selection.json')
        report['sampling_verified_identical'] = True
        save()
        print('Selection locked before benchmarks:', winner, flush=True)
        for arm in ARMS:
            for split in ('dcc', 'riverside'):
                report['phase'] = 'EVALUATING_' + arm['name'].upper() + '_' + split.upper()
                save()
                report['arms'][arm['name']]['evaluations'][split] = evaluate(arm, split)
                save()
                print('Benchmark:', arm['name'], split, json.dumps(report['arms'][arm['name']]['evaluations'][split]['metrics']), flush=True)
        if digest(run / 'protocol.json') != report['protocol_sha256'] or digest(run / 'selection.json') != report['selection_sha256']:
            raise ValueError('Protocol or locked selection changed')
        for modality in ('l', 'r'):
            if digest(protocol['pretrain_' + modality]) != protocol['pretrain_' + modality + '_sha256']:
                raise ValueError('Unimodal checkpoint changed')
        with (run / 'comparison.csv').open('x', newline='') as f:
            fields = ['arm', 'split', 'selected_on_validation', 'Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@1percent', 'F1']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for arm in ARMS:
                entry = report['arms'][arm['name']]
                for split in ('sejong', 'dcc', 'riverside'):
                    result = entry['validation'] if split == 'sejong' else entry['evaluations'][split]
                    writer.writerow(dict(arm=arm['name'], split=split, selected_on_validation=arm['name'] == winner, **result['metrics']))
        report.update(status='COMPLETE', phase='DONE', finished_at=now())
    except BaseException as error:
        report['status'] = 'FAILED_OR_INTERRUPTED'
        report['error'] = repr(error)
        save()
        raise
    save()
    print('Completed all fixed experiments:', run, flush=True)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_one(sys.argv[2], sys.argv[3])
    else:
        run_all()
