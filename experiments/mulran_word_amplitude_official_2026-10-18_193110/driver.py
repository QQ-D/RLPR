"""Paired pure-phase / measured-amplitude experiments on full MulRan training data."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

SOURCE = Path(__file__).resolve().parents[1]
# A frozen copy of this driver also runs from a run directory.
if SOURCE.name != 'src_spectral':
    SOURCE = Path('/media/rwj/ZX1 2TB/RLPR/src_spectral')
PROJECT = SOURCE.parent
INITIAL = PROJECT / 'runs/mulran_word_full_2026-10-14_232504'
TRAIN = PROJECT / 'processed/mulran_official'
VALIDATION = PROJECT / 'processed/mulran_sejong'
ARMS = ('pure_phase', 'amplitude')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('w') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def experiment_settings(schedule='continuation'):
    if schedule not in ('continuation', 'official_code'):
        raise ValueError('Unknown training schedule')
    settings = dict(schedule=schedule, unimodal_epochs=3, unimodal_additional_epochs=3,
                unimodal_initialization='historical_word_weights', historical_unimodal_epochs=5, cross_epochs=2,
                arms=list(ARMS), batch_size=8, seed=0, sampling_seed=0, learning_rate=5e-5,
                scheduler_gamma=.8, cache_batch_size=32, cross_hard_negative_count=5,
                phase_loss_weight=.1, cross_fusion_loss_weight=.5, alignment_loss='signed_coherency_v2',
                deterministic_algorithms=True, cublas_workspace_config=':4096:8',
                teacher_selection='Sejong unimodal Recall@1, then earliest additional epoch',
                cross_selection='Sejong cross-modal Recall@1, then F1, then earliest epoch',
                benchmarks='DCC and Riverside only after both arms lock all selections',
                caveat='Exploratory single-seed ablation; benchmark scenes have been inspected previously',
                representation='Phase-only Transformer inputs; amplitude arm restores measured raw FFT magnitude after self attention and training-only cross attention; no spatial residual',
                initialization='Same historical L5/R5 model-only weights; fresh Adam and scheduler in both arms')
    if schedule == 'official_code':
        settings.update(unimodal_epochs=5, unimodal_additional_epochs=0, historical_unimodal_epochs=0,
                        unimodal_initialization='resnet18_imagenet', cross_epochs=1,
                        teacher_selection='Fixed epoch5 for both unary branches; validation is diagnostic only',
                        cross_selection='Fixed epoch1, the sole alignment epoch',
                        epoch_source='Original src/scripts/registry.py: UNIMODAL_EPOCHS=5; MulRan cross epochs=1. Paper does not specify MulRan epoch count.',
                        initialization='Fresh seed0 model with cached torchvision ResNet18 ImageNet backbone in both arms; no RLPR branch warm start')
    return settings


def stage_epoch_budget(protocol, stage):
    if stage not in ('l', 'r', 'cross'):
        raise ValueError('Unknown training stage')
    return protocol['cross_epochs'] if stage == 'cross' else protocol.get(
        'unimodal_epochs', protocol['unimodal_additional_epochs'])


def choose_stage_epoch(rows, protocol, stage):
    if protocol.get('schedule', 'continuation') == 'official_code':
        count = stage_epoch_budget(protocol, stage)
        if set(rows) != {str(e) for e in range(1, count + 1)}:
            raise ValueError('Fixed official-code checkpoint requires the complete stage budget')
        return count
    return select_epoch(rows, cross=stage == 'cross')


def select_epoch(rows, *, cross):
    if not rows:
        raise ValueError('Cannot select an epoch from empty validation records')
    return max((int(e) for e in rows), key=lambda e: (
        rows[str(e)]['validation']['Recall@1'],
        rows[str(e)]['validation']['F1'] if cross else 0., -e))


def verify_training_validation_split(train_root, validation_root):
    import numpy as np
    from scipy.spatial import cKDTree
    train_root, validation_root = Path(train_root), Path(validation_root)
    train = np.concatenate([np.load(train_root / 'index' / name)
                            for name in ('sejong_train_query.npy', 'sejong_db.npy')])
    held = np.concatenate([np.load(validation_root / 'index' / name)
                           for name in ('sejong_test_db.npy', 'sejong_test_query.npy')])
    if np.intersect1d(train[:, 0], held[:, 0]).size:
        raise ValueError('Train/validation frame overlap')
    minimum = float(cKDTree(train[:, 1:]).query(held[:, 1:])[0].min())
    if minimum < 50.:
        raise ValueError('Sejong geographic buffer below50m')
    for name in ('sejong_test_db.npy', 'sejong_test_query.npy'):
        if not np.array_equal(np.load(train_root / 'index' / name), np.load(validation_root / 'index' / name)):
            raise ValueError('Validation index differs')
    return minimum


def stage_environment(root, arm, stage, epoch):
    if arm not in ARMS or stage not in ('l', 'r', 'cross') or epoch < 1:
        raise ValueError('Invalid amplitude experiment arm, stage or epoch')
    root = Path(root)
    env = os.environ.copy()
    env.update(RLPR_MODEL_VARIANT='spectral_word', RLPR_WORD_AMPLITUDE=str(int(arm == 'amplitude')),
               RLPR_SAMPLING_SEED='0', RLPR_DETERMINISTIC='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               RLPR_CROSS_HARD_NEGATIVES='5', RLPR_SPECTRAL_CROSS_FUSION='1',
               RLPR_PHASE_LOSS_WEIGHT='.1', RLPR_CROSS_FUSION_WEIGHT='.5',
               RLPR_WORD_ALIGNMENT_LOSS='signed_coherency_v2', RLPR_EPOCHS=str(epoch),
               RLPR_RESUME=str(int(epoch > 1)), RLPR_RESUME_EPOCH=str(max(1, epoch - 1)),
               RLPR_CHECKPOINT_PATH='', RLPR_TEST_ZERO_SHOT='0', RLPR_PRETRAIN_L='', RLPR_PRETRAIN_R='',
               RLPR_TRAINING_ROOT=str(root / arm / stage), RLPR_MULRAN_DATA=str(TRAIN),
               RLPR_MULRAN_TEST_SPLIT='sejong', TORCH_HOME=str(PROJECT / 'cache/torch'),
               MPLCONFIGDIR=str(root / 'matplotlib_cache'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTHONUNBUFFERED='1')
    return env


def child_train(root, arm, stage, epoch):
    import torch
    sys.path.insert(0, str(SOURCE))
    from scripts.registry import build_params, CROSS_MODAL, UNIMODAL, get_model_variant
    from trainer.cross_modal import CrossModalTrainer
    from trainer.unimodal import UnimodalTrainer
    from utils.cross_mining import make_cross_modal_cache_loader
    from utils.seed import set_global_seed
    set_global_seed(0)
    torch.set_num_threads(4)
    params = build_params(epochs_override=epoch)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    if stage == 'cross':
        cfg = CROSS_MODAL['mulran']
        trainer = CrossModalTrainer(params, get_model_variant(cfg['model'], 'spectral_word'), cfg['dataloader'])
    else:
        protocol = json.loads((Path(root) / 'protocol.json').read_text())
        if protocol.get('schedule', 'continuation') == 'continuation':
            params.initial_checkpoint_path = str(INITIAL / f'{stage}_mulran/weights/5.pth.tar')
        cfg = UNIMODAL['mulran', stage]
        trainer = UnimodalTrainer(params, get_model_variant(cfg['model'], 'spectral_word'),
                                  cfg['dataloader'], stage, cfg['cache_name'], cfg['hard_mining_from_epoch'])
        trainer.train_query_loader = make_cross_modal_cache_loader(trainer.train_query_loader, batch_size=32)
    try:
        if len(trainer.train_loader.dataset) != 16568 or len(trainer.train_loader) != 2071:
            raise ValueError('Unexpected full training query count')
        trainer.do_train()
    finally:
        trainer.writer.close()


def child_evaluate(stage, checkpoint, output):
    import faiss
    import numpy as np
    import torch
    from tqdm import tqdm
    sys.path.insert(0, str(SOURCE))
    from scripts.registry import build_test_params, CROSS_MODAL, UNIMODAL, get_model_variant
    from trainer.eval import CrossModalTester, recall_at_k
    from utils.pretrain import validate_checkpoint_variant
    from utils.seed import set_global_seed
    set_global_seed(0)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    params = build_test_params()
    params.checkpoint_path = str(checkpoint)
    if stage == 'cross':
        cfg = CROSS_MODAL['mulran']
        tester = CrossModalTester(params, get_model_variant(cfg['model'], 'spectral_word'), cfg['dataloader'])
        metrics = {}
        original = tester._print_f1
        def record_f1(*args):
            metrics['F1'] = original(*args)
            return metrics['F1']
        tester._print_f1 = record_f1
        scores = tester.do_test()
        metrics.update({f'Recall@{k}': float(v) for k, v in zip((1, 5, 10, 20, '1percent'), scores)})
    else:
        cfg = UNIMODAL['mulran', stage]
        model = get_model_variant(cfg['model'], 'spectral_word').create().cuda().eval()
        saved = torch.load(checkpoint, map_location='cpu')
        validate_checkpoint_variant(model, saved)
        model.load_state_dict(saved['state_dict'], strict=True)
        loader, dataset = cfg['dataloader']('test', params)
        descriptors = []
        with torch.no_grad():
            for batch in tqdm(loader):
                bev = batch['lidar_bev' if stage == 'l' else 'radar_bev'].cuda()
                descriptors.append(model(bev, bev, bev, bev, bev)[3].cpu().numpy())
        descriptors = np.concatenate(descriptors).astype('float32')
        if len(descriptors) != dataset.num_db + dataset.num_query:
            raise ValueError('Incomplete unary validation extraction')
        index = faiss.IndexFlatL2(params.output_dim)
        index.add(descriptors[:dataset.num_db])
        _, predictions = index.search(descriptors[dataset.num_db:], min(20, dataset.num_db))
        metrics = {f'Recall@{k}': float(v) for k, v in zip((1, 5, 10, 20),
                    recall_at_k(predictions, dataset.get_positives(), (1, 5, 10, 20)))}
    write_json(output, metrics)
    print(json.dumps(metrics), flush=True)


def audit_checkpoint(path, epoch, *, teacher=None):
    import torch
    saved = torch.load(path, map_location='cpu')
    if saved['epoch'] != epoch or saved['model_variant'] != 'spectral_word':
        raise ValueError('Unexpected checkpoint identity')
    if not all(torch.isfinite(v).all() for v in saved['state_dict'].values()):
        raise ValueError('Nonfinite model state')
    steps = {int(s['step']) for s in saved['optimizer']['state'].values()}
    if steps != {2071 * epoch} or saved['scheduler']['last_epoch'] != epoch:
        raise ValueError('Optimizer update count or scheduler epoch mismatch')
    if saved.get('training_state') is None or not saved['training_settings']['deterministic_algorithms']:
        raise ValueError('Missing strict resume state')
    if teacher:
        radar = torch.load(teacher, map_location='cpu')['state_dict']
        for name, value in radar.items():
            if not torch.equal(value, saved['state_dict'][name]):
                raise ValueError('Frozen radar teacher changed: ' + name)
    sampling = {name: hashlib.sha256(saved['training_state'][key].numpy().tobytes()).hexdigest()
                for name, key in [('sampler', 'sampler_generator'), ('worker', 'loader_generator')]}
    return dict(checkpoint=str(path), checkpoint_sha256=digest(path), optimizer_steps=2071 * epoch,
                all_finite=True, radar_unchanged=bool(teacher), sampling=sampling)


def run_experiment(root=None, schedule='continuation'):
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    continuing = root is not None
    prefix = 'mulran_word_amplitude_official_' if schedule == 'official_code' else 'mulran_word_amplitude_'
    root = Path(root) if continuing else PROJECT / 'runs' / (prefix +
             datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H%M%S'))
    if not continuing:
        root.mkdir(exist_ok=False)
        protocol = experiment_settings(schedule)
        protocol.update(created_at=now(), source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
            source_sha256={str(p.relative_to(SOURCE)): digest(p)
                           for directory in ('config', 'dataset', 'modules', 'scripts', 'trainer', 'utils', 'tests')
                           for p in (SOURCE / directory).rglob('*.py')},
            initialization={m: dict(path=str(INITIAL / f'{m}_mulran/weights/5.pth.tar'),
                              sha256=digest(INITIAL / f'{m}_mulran/weights/5.pth.tar'))
                            for m in (('l', 'r') if schedule == 'continuation' else ())})
        if schedule == 'official_code':
            backbone = PROJECT / 'cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
            protocol['initialization'] = {'backbone': dict(path=str(backbone), sha256=digest(backbone))}
        protocol['nearest_train_validation_m'] = verify_training_validation_split(TRAIN, VALIDATION)
        write_json(root / 'protocol.json', protocol)
        (root / 'driver.py').write_text(Path(__file__).read_text())
        report = dict(status='RUNNING', started_at=now(), run_root=str(root),
                      protocol_sha256=digest(root / 'protocol.json'), arms={a: {} for a in ARMS})
        write_json(root / 'run.json', report)
    else:
        protocol = json.loads((root / 'protocol.json').read_text())
        report = json.loads((root / 'run.json').read_text())
        if report['protocol_sha256'] != digest(root / 'protocol.json'):
            raise ValueError('Protocol was modified')
        if digest(root / 'driver.py') != digest(Path(__file__)):
            raise ValueError('Resume must use the unchanged frozen driver')
    Path('/tmp/rlpr_amplitude_run_path').write_text(str(root))
    print('RUN', root, flush=True)

    def save():
        write_json(root / 'run.json', report)

    report['status'] = 'RUNNING'
    report.pop('error', None)
    save()

    def verify_source():
        for name, expected in protocol['source_sha256'].items():
            if digest(SOURCE / name) != expected:
                raise ValueError('Source changed during experiment: ' + name)
        for init in protocol['initialization'].values():
            if digest(init['path']) != init['sha256']:
                raise ValueError('Initialization checkpoint changed')

    def subprocess_logged(command, env, path):
        verify_source()
        log = path
        attempt = 0
        while log.exists():
            attempt += 1
            log = path.with_suffix(f'.retry{attempt}.log')
        with log.open('x') as handle:
            subprocess.run([sys.executable, '-u', str(root / 'driver.py'), *command],
                           cwd=SOURCE, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        return log

    def evaluate(arm, stage, epoch, split):
        folder = root / arm / stage
        checkpoint = folder / f'weights/{epoch}.pth.tar'
        output = folder / f'epoch_{epoch}_{split}.json'
        env = stage_environment(root, arm, stage, epoch)
        env.update(RLPR_MULRAN_TEST_SPLIT=split,
                   RLPR_MULRAN_DATA=str(VALIDATION if split == 'sejong' else TRAIN),
                   RLPR_CHECKPOINT_PATH=str(checkpoint))
        subprocess_logged(['--child', 'evaluate', '--stage', stage, '--checkpoint', str(checkpoint),
                           '--output', str(output)], env, folder / f'epoch_{epoch}_{split}.log')
        return json.loads(output.read_text())

    try:
        for arm in ARMS:
            for stage in ('l', 'r', 'cross'):
                record = report['arms'][arm].setdefault(stage, {'epochs': {}})
                count = stage_epoch_budget(protocol, stage)
                for epoch in range(1, count + 1):
                    if str(epoch) in record['epochs']:
                        previous = record['epochs'][str(epoch)]
                        if digest(previous['checkpoint']) != previous['checkpoint_sha256']:
                            raise ValueError('Recorded checkpoint changed')
                        continue
                    env = stage_environment(root, arm, stage, epoch)
                    radar_path = None
                    if stage == 'cross':
                        for m in ('l', 'r'):
                            selected = report['arms'][arm][m]['selected_epoch']
                            env['RLPR_PRETRAIN_' + m.upper()] = str(root / arm / m / f'weights/{selected}.pth.tar')
                        radar_path = env['RLPR_PRETRAIN_R']
                    folder = root / arm / stage
                    folder.mkdir(parents=True, exist_ok=True)
                    checkpoint = folder / f'weights/{epoch}.pth.tar'
                    report['phase'] = f'{arm}/{stage}/epoch{epoch}'
                    save()
                    if not checkpoint.exists():
                        subprocess_logged(['--child', 'train', '--root', str(root), '--arm', arm,
                                           '--stage', stage, '--epoch', str(epoch)], env,
                                          folder / f'epoch_{epoch}_train.log')
                    audited = audit_checkpoint(checkpoint, epoch, teacher=radar_path)
                    write_json(folder / f'epoch_{epoch}_sampling.json', audited['sampling'])
                    expected_mode = arm == 'pure_phase'
                    if torch.load(checkpoint, map_location='cpu')['spectral_config']['pure_phase'] != expected_mode:
                        raise ValueError('Checkpoint amplitude mode differs from assigned arm')
                    logs = sorted(folder.glob(f'epoch_{epoch}_train*.log'), key=lambda p: p.stat().st_mtime)
                    losses = re.findall(r'loss \[([\deE.+-]+)\]', logs[-1].read_text())
                    if not losses:
                        raise ValueError('Training log has no final loss')
                    audited['loss'] = float(losses[-1])
                    audited['validation'] = evaluate(arm, stage, epoch, 'sejong')
                    record['epochs'][str(epoch)] = audited
                    save()
                    print(report['phase'], 'loss', audited['loss'], audited['validation'], flush=True)
                record['selected_epoch'] = choose_stage_epoch(record['epochs'], protocol, stage)
                record['selection_locked_at'] = now()
                save()
                print('SELECTED', arm, stage, record['selected_epoch'], flush=True)
        for stage in ('l', 'r', 'cross'):
            count = stage_epoch_budget(protocol, stage)
            for epoch in range(1, count + 1):
                sampling = [json.loads((root / arm / stage / f'epoch_{epoch}_sampling.json').read_text())
                            for arm in ARMS]
                if sampling[0] != sampling[1]:
                    raise ValueError('Paired sampler/worker streams differ')
        report['paired_sampling_streams_match'] = True
        write_json(root / 'selection.json', {a: {s: report['arms'][a][s]['selected_epoch']
                    for s in ('l', 'r', 'cross')} for a in ARMS})
        report['selection_sha256'] = digest(root / 'selection.json')
        save()
        for arm in ARMS:
            record = report['arms'][arm]['cross']
            epoch = record['selected_epoch']
            record.setdefault('benchmarks', {})
            for split in ('dcc', 'riverside'):
                if split not in record['benchmarks']:
                    record['benchmarks'][split] = evaluate(arm, 'cross', epoch, split)
                    save()
                print('TEST', arm, split, record['benchmarks'][split], flush=True)
        verify_source()
        if digest(root / 'selection.json') != report['selection_sha256']:
            raise ValueError('Selection changed after benchmark evaluation')
        report.update(status='COMPLETE', finished_at=now(), phase='COMPLETE')
        save()
        print('COMPLETE', root, flush=True)
    except BaseException as error:
        report.update(status='FAILED', error=repr(error), failed_at=now())
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--schedule', choices=('continuation', 'official_code'), default='continuation')
    parser.add_argument('--child', choices=('train', 'evaluate'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--stage', choices=('l', 'r', 'cross'))
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.child == 'train':
        child_train(args.root, args.arm, args.stage, args.epoch)
    elif args.child == 'evaluate':
        child_evaluate(args.stage, args.checkpoint, args.output)
    else:
        run_experiment(args.run_root, args.schedule)
