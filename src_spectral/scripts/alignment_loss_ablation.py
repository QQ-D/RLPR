"""Signed-coherency versus existing InfoNCE on the selected amplitude configuration."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

SOURCE = Path('/media/rwj/ZX1 2TB/RLPR/src_spectral')
sys.path.insert(0, str(SOURCE))
from scripts.amplitude_ablation import (
    PROJECT, TRAIN, VALIDATION, now, digest, write_json, stage_environment,
    child_train, child_evaluate, audit_checkpoint, verify_training_validation_split,
)
from scripts.phase_weight_ablation import execution_environment as phase_execution_environment
from scripts.fusion_weight_ablation import verify_control

REFERENCE = PROJECT / 'runs/mulran_amplitude_phase_2026-10-19_130536'
LOSSES = {'signed': 'signed_coherency_v2', 'infonce': 'infonce'}


def execution_environment(root, arm, initial, mode, split):
    if arm not in LOSSES:
        raise ValueError('Unknown alignment configuration')
    env = phase_execution_environment(root, 'phase_01', initial, mode, split)
    env.update(RLPR_TRAINING_ROOT=str(Path(root) / arm),
               RLPR_WORD_ALIGNMENT_LOSS=LOSSES[arm])
    return env


def choose_configuration(records):
    if set(records) != set(LOSSES):
        raise ValueError('All configurations must finish validation before selection')
    for row in records.values():
        for key in ('Recall@1', 'F1'):
            if not math.isfinite(row.get('validation', {}).get(key, float('nan'))):
                raise ValueError('Incomplete or nonfinite validation scores')
    return max(LOSSES, key=lambda arm: (
        records[arm]['validation']['Recall@1'], records[arm]['validation']['F1'],
        -list(LOSSES).index(arm)))


def audit_trial(checkpoint, loss, teacher, previous=None):
    import torch
    if previous and digest(checkpoint) != previous['checkpoint_sha256']:
        raise ValueError('Recorded checkpoint changed; cached metrics cannot be reused')
    audit = audit_checkpoint(checkpoint, 1, teacher=teacher)
    saved = torch.load(checkpoint, map_location='cpu')
    expected = dict(alignment_loss=loss, batch_size=8,
                    phase_loss_weight=.1, cross_fusion_loss_weight=0.,
                    learning_rate_initial=5e-5, scheduler_gamma=.8,
                    deterministic_algorithms=True, cublas_workspace_config=':4096:8',
                    cross_hard_negative_count=5, sampling_seed=0)
    if saved['training_settings'] != expected:
        raise ValueError('Trial training settings differ from the frozen comparison')
    if saved['spectral_config'].get('pure_phase', True) or not saved['cross_fusion_enabled']:
        raise ValueError('Expected amplitude model with identical fusion computation in every arm')
    for state in saved['optimizer']['state'].values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError('Nonfinite optimizer state')
    return audit


def lock_selection(root, report, rule):
    records = report['configurations']
    selected = choose_configuration(records)
    payload = dict(selected=selected, rule=rule,
                   validation={a: r['validation'] for a, r in records.items()},
                   checkpoints={a: r.get('checkpoint_sha256') for a, r in records.items()})
    path = Path(root) / 'selection.json'
    if path.exists():
        locked = json.loads(path.read_text())
        if any(locked.get(k) != value for k, value in payload.items()):
            raise ValueError('Locked selection no longer matches validation evidence')
        previous_hash = report.get('selection_sha256')
        if previous_hash is not None and digest(path) != previous_hash:
            raise ValueError('Locked selection file changed')
    else:
        if report.get('selection_sha256') is not None:
            raise ValueError('Previously recorded selection file is missing')
        write_json(path, dict(payload, locked_at=now()))
    # A valid lock can survive a crash before its hash reached run.json.
    report.update(selected_configuration=selected, selection_sha256=digest(path))
    return selected


def run(root=None):
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    if root is None:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        root = PROJECT / 'runs' / ('mulran_amplitude_alignment_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H%M%S'))
        root.mkdir(exist_ok=False)
        previous = json.loads((REFERENCE / 'run.json').read_text())
        prior_protocol = json.loads((REFERENCE / 'protocol.json').read_text())
        if previous['status'] != 'COMPLETE' or digest(REFERENCE / 'protocol.json') != previous['protocol_sha256']:
            raise ValueError('Reference experiment is incomplete or changed')
        # Existing model/trainer/data code must remain identical to the successful run.
        for name, expected in prior_protocol['source_sha256'].items():
            if digest(SOURCE / name) != expected:
                raise ValueError('Source differs from amplitude reference: ' + name)
        if digest(REFERENCE / 'selection.json') != previous['selection_sha256']:
            raise ValueError('Reference validation selection changed')
        if previous['selected_configuration'] != 'phase_01':
            raise ValueError('This comparison requires validation-selected phase auxiliary .1')
        initial = {stage: dict(prior_protocol['initial'][stage]) for stage in ('l', 'r')}
        row = previous['configurations']['phase_01']
        if row['weight'] != .1 or prior_protocol['fixed_fusion_loss_weight'] != 0. or digest(row['checkpoint']) != row['checkpoint_sha256']:
            raise ValueError('Reference checkpoint changed')
        initial['reference'] = dict(path=row['checkpoint'], sha256=row['checkpoint_sha256'])
        protocol = dict(created_at=now(), configurations=LOSSES, seed=0, sampling_seed=0,
                        epochs=1, initial=initial, reference_root=str(REFERENCE),
                        variable='word_alignment_loss', fixed_fusion_loss_weight=0., fixed_phase_loss_weight=.1,
                        selection='Sejong Recall@1, then F1, then signed control on exact tie',
                        objective_semantics='Existing signed pull/ranking versus existing Word InfoNCE: separate local/global teacher-to-student diagonal12-way matches at temperature.07, including the nearby positive as a separate instance. No new multi-positive mask. Both retain the same phase auxiliaries and zero fusion supervision with identical forward paths.',
                        test_policy='Both validations and bitwise control replay must pass before locking selection and reading benchmarks',
                        caveat='Single-seed exploratory comparison; DCC/Riverside have already been inspected',
                        nearest_train_validation_m=verify_training_validation_split(TRAIN, VALIDATION),
                        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
                        source_sha256={str(p.relative_to(SOURCE)): digest(p)
                            for directory in ('config', 'dataset', 'modules', 'scripts', 'trainer', 'utils', 'tests')
                            for p in (SOURCE / directory).rglob('*.py')})
        write_json(root / 'protocol.json', protocol)
        (root / 'driver.py').write_text(Path(__file__).read_text())
        report = dict(status='RUNNING', started_at=now(), phase='PREFLIGHT',
                      protocol_sha256=digest(root / 'protocol.json'), configurations={})
        write_json(root / 'run.json', report)
    else:
        root = Path(root)
        protocol = json.loads((root / 'protocol.json').read_text())
        report = json.loads((root / 'run.json').read_text())
        if digest(root / 'protocol.json') != report['protocol_sha256'] or digest(root / 'driver.py') != digest(Path(__file__)):
            raise ValueError('Resume requires unchanged protocol and frozen driver')
    print('RUN', root, flush=True)
    Path('/tmp/rlpr_alignment_run_path').write_text(str(root))

    def verify_inputs():
        for name, expected in protocol['source_sha256'].items():
            if digest(SOURCE / name) != expected:
                raise ValueError('Frozen source changed: ' + name)
        for initial in protocol['initial'].values():
            if digest(initial['path']) != initial['sha256']:
                raise ValueError('Frozen initialization changed')

    def save():
        write_json(root / 'run.json', report)

    def child(arm, mode, split='sejong'):
        verify_inputs()
        folder = root / arm
        env = execution_environment(root, arm, protocol['initial'], mode, split)
        logfile = folder / ('train.log' if mode == 'train' else split + '.log')
        n = 0
        while logfile.exists():
            n += 1
            logfile = folder / (('train' if mode == 'train' else split) + f'.retry{n}.log')
        with logfile.open('x') as handle:
            subprocess.run([sys.executable, '-u', str(root / 'driver.py'), '--child', mode,
                            '--root', str(root), '--arm', arm, '--split', split],
                           cwd=SOURCE, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    try:
        report.update(status='RUNNING')
        report.pop('error', None)
        save()
        for arm, loss in LOSSES.items():
            folder = root / arm
            folder.mkdir(exist_ok=True)
            checkpoint = folder / 'weights/1.pth.tar'
            report['phase'] = arm + '/TRAIN'
            save()
            if not checkpoint.exists():
                child(arm, 'train')
            audited = audit_trial(checkpoint, loss, protocol['initial']['r']['path'],
                                  previous=report['configurations'].get(arm))
            if arm == 'signed':
                verify_control(checkpoint, protocol['initial']['reference']['path'])
                report['control_replay_bitwise_equal'] = True
            if arm != 'signed' and audited['sampling'] != report['configurations']['signed']['sampling']:
                raise ValueError('Paired sampling streams differ')
            report['phase'] = arm + '/VALIDATION'
            save()
            result = folder / 'sejong.json'
            if not result.exists():
                child(arm, 'evaluate')
            existing = report['configurations'].get(arm, {})
            report['configurations'][arm] = dict(existing, **audited, alignment_loss=loss,
                validation=json.loads(result.read_text()))
            save()
            print('VALIDATION', arm, report['configurations'][arm]['validation'], flush=True)
        selected = lock_selection(root, report, protocol['selection'])
        save()
        selection_path = root / 'selection.json'
        for arm in LOSSES:
            row = report['configurations'][arm]
            row.setdefault('benchmarks', {})
            for split in ('dcc', 'riverside'):
                report['phase'] = arm + '/' + split
                save()
                if split not in row['benchmarks']:
                    child(arm, 'evaluate', split)
                    row['benchmarks'][split] = json.loads((root / arm / (split + '.json')).read_text())
                    save()
                print('TEST', arm, split, row['benchmarks'][split], flush=True)
        verify_inputs()
        if digest(selection_path) != report['selection_sha256']:
            raise ValueError('Selection changed after benchmarks')
        comparisons = []
        official = json.loads((PROJECT / 'runs/mulran_official_eval_2026-10-14/results.json').read_text())
        for split in ('dcc', 'riverside'):
            for method, field in [('paper', 'paper_metrics'), ('released_official', 'metrics')]:
                comparisons.append(dict(dataset=split, configuration=method, selected=False,
                                        **official['splits'][split][field]))
            for arm, row in report['configurations'].items():
                comparisons.append(dict(dataset=split, configuration=arm, selected=arm == selected,
                                        **row['benchmarks'][split]))
        with (root / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['dataset', 'configuration', 'selected',
                'Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@1percent', 'F1'])
            writer.writeheader()
            writer.writerows(comparisons)
        report.update(status='COMPLETE', phase='COMPLETE', finished_at=now(), paired_sampling_streams_match=True)
        save()
        write_json(root / 'results_summary.json', dict(selected_configuration=selected,
            comparisons=comparisons, control_replay_bitwise_equal=True,
            paired_sampling_streams_match=True, caveat=protocol['caveat']))
        print('COMPLETE', root, 'SELECTED', selected, flush=True)
    except BaseException as error:
        report.update(status='FAILED', error=repr(error), failed_at=now())
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--child', choices=('train', 'evaluate'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--arm', choices=tuple(LOSSES))
    parser.add_argument('--split', choices=('sejong', 'dcc', 'riverside'), default='sejong')
    args = parser.parse_args()
    if args.child == 'train':
        child_train(args.root, 'amplitude', 'cross', 1)
    elif args.child == 'evaluate':
        child_evaluate('cross', args.root / args.arm / 'weights/1.pth.tar', args.root / args.arm / (args.split + '.json'))
    else:
        run(args.run_root)
