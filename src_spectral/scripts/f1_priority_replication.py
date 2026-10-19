"""F1-first paired cross-stage replication; historical sources remain immutable."""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

SOURCE = Path('/media/rwj/ZX1 2TB/RLPR/src_spectral')
sys.path.insert(0, str(SOURCE))
from scripts.amplitude_ablation import PROJECT, now, digest, write_json, audit_checkpoint
from scripts.alignment_loss_ablation import LOSSES, execution_environment

REFERENCE = PROJECT / 'runs/mulran_amplitude_alignment_2026-10-19_135238'
SEEDS = (0, 1, 2)
RULE = 'Mean Sejong F1 over seeds 0/1/2, then mean Sejong Recall@1, then signed; no seed selection'


def environment(root, seed, arm, initial, mode, split):
    if seed not in SEEDS:
        raise ValueError('Unexpected seed')
    env = execution_environment(Path(root) / f'seed{seed}', arm, initial, mode, split)
    env.update(RLPR_SAMPLING_SEED=str(seed), RLPR_EXPERIMENT_SEED=str(seed))
    return env


def initialize_seed(seed):
    from utils.seed import set_global_seed
    set_global_seed(seed)


def choose_loss(records):
    if set(records) != {str(s) for s in SEEDS}:
        raise ValueError('Exactly the three declared seeds are required')
    stats = {}
    for arm in LOSSES:
        stats[arm] = {}
        for metric in ('F1', 'Recall@1'):
            try:
                values = [records[str(s)][arm]['validation'][metric] for s in SEEDS]
            except KeyError:
                raise ValueError('Incomplete validation') from None
            if any(not math.isfinite(v) for v in values):
                raise ValueError('Nonfinite validation')
            stats[arm][metric] = dict(mean=statistics.mean(values), sd=statistics.stdev(values))
    selected = max(LOSSES, key=lambda a: (stats[a]['F1']['mean'], stats[a]['Recall@1']['mean'],
                                         -list(LOSSES).index(a)))
    return selected, stats


def bind_artifact(path):
    return dict(path=str(path), sha256=digest(path))


def verify_artifact(record):
    if digest(record['path']) != record['sha256']:
        raise ValueError('Recorded artifact changed: ' + record['path'])


def verify_metrics(row):
    for name, artifact in row.get('artifacts', {}).items():
        if not name.endswith('_metrics'):
            continue
        split = name.removesuffix('_metrics')
        verify_artifact(artifact)
        cached = json.loads(Path(artifact['path']).read_text())
        recorded = row['validation'] if split == 'sejong' else row['benchmarks'][split]
        if cached != recorded:
            raise ValueError('Recorded metrics differ from bound evaluation: '+split)
        diag = row['artifacts'][split+'_diagnostics.json']
        verify_artifact(diag)
        binding = json.loads(Path(diag['path']).read_text())['checkpoint']
        if binding != dict(path=row['checkpoint'],sha256=row['checkpoint_sha256']):
            raise ValueError('Metrics belong to a different checkpoint: '+split)


def lock_selection(root, report):
    selected, stats = choose_loss(report['seeds'])
    payload = dict(selected=selected, rule=RULE, validation_stats=stats,
        evidence={s: {a: {k: row.get(k) for k in ('validation', 'checkpoint_sha256', 'validation_artifact')}
                      for a, row in arms.items()} for s, arms in report['seeds'].items()})
    path = Path(root) / 'selection.json'
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(k) != v for k, v in payload.items()):
            raise ValueError('Locked validation selection changed')
        if report.get('selection_sha256') and digest(path) != report['selection_sha256']:
            raise ValueError('Selection hash changed')
    else:
        if report.get('selection_sha256'):
            raise ValueError('Selection disappeared')
        write_json(path, dict(payload, locked_at=now()))
    report.update(selected_loss=selected, selection_sha256=digest(path))
    return selected


def threshold_counts(distances, correct, threshold):
    import numpy as np
    accepted = np.asarray(distances) < threshold
    correct = np.asarray(correct, dtype=bool)
    tp, fp = int((accepted & correct).sum()), int((accepted & ~correct).sum())
    fn, tn = int((~accepted & correct).sum()), int((~accepted & ~correct).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return dict(threshold=float(threshold), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall,
                F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def legacy_curve(distances, correct):
    import numpy as np
    lower = distances.min()
    if lower - .1 > 0:
        lower -= .1
    rows = [threshold_counts(distances, correct, t)
            for t in np.linspace(lower, distances.max()+.1, 1000)]
    return max(rows, key=lambda r: r['F1'])


def child_train(seed, output):
    import torch
    from scripts.registry import build_params, CROSS_MODAL, get_model_variant
    from trainer.cross_modal import CrossModalTrainer
    initialize_seed(seed)
    torch.set_num_threads(4)
    import hashlib
    write_json(output / 'seed_start.json', dict(seed=seed, sampling_seed=seed,
        torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
    params = build_params(epochs_override=1)
    if params.sampling_seed != seed or not torch.cuda.is_available():
        raise ValueError('Seed configuration or GPU unavailable')
    cfg = CROSS_MODAL['mulran']
    trainer = CrossModalTrainer(params, get_model_variant(cfg['model'], 'spectral_word'), cfg['dataloader'])
    try:
        if len(trainer.train_loader.dataset) != 16568 or len(trainer.train_loader) != 2071:
            raise ValueError('Unexpected training split')
        trainer.do_train()
    finally:
        trainer.writer.close()


def evaluate(checkpoint, output, threshold=None, variant='spectral_word'):
    """Capture descriptors through the unchanged evaluator and verify F1 parity."""
    import faiss
    import numpy as np
    import torch
    from scripts.registry import build_test_params, CROSS_MODAL, get_model_variant
    from trainer.eval import CrossModalTester
    initialize_seed(0)
    torch.set_num_threads(4)
    params = build_test_params()
    params.checkpoint_path = str(checkpoint)
    cfg = CROSS_MODAL['mulran']
    tester = CrossModalTester(params, get_model_variant(cfg['model'], variant), cfg['dataloader'])
    original = tester._print_f1
    metrics = {}
    diagnostics = {}

    def capture(query, database, gt):
        metrics['F1'] = original(query, database, gt)
        index = faiss.IndexFlatL2(query.shape[1]); index.add(database)
        distances, predictions = index.search(query, len(database))
        correct = np.array([pred[0] in g for pred, g in zip(predictions, gt)])
        best = legacy_curve(distances[:, 0], correct)
        if not math.isclose(best['F1'], metrics['F1'], abs_tol=1e-12):
            raise ValueError('Diagnostic F1 differs from original evaluator')
        ds = tester.test_dataset
        db_xy = ds.database_index_n_pos[:, 1:]
        query_xy = ds.query_index_n_pos[:, 1:]
        errors = np.linalg.norm(query_xy-db_xy[predictions[:, 0]], axis=1)
        diagnostics.update(best_threshold=best,
            queries=len(query), database=len(database), valid_queries=int(sum(len(g)>0 for g in gt)),
            correct_top1=int(correct.sum()), wrong_top1=int((~correct).sum()),
            wrong_top1_median_m=float(np.median(errors[~correct])) if (~correct).any() else None,
            wrong_top1_p90_m=float(np.percentile(errors[~correct],90)) if (~correct).any() else None,
            checkpoint=bind_artifact(checkpoint), metric_definition='Original conditional-recall max F1; wrong rejected top1 is TN, not FN')
        if threshold is not None:
            diagnostics['validation_frozen_threshold'] = threshold_counts(distances[:,0],correct,threshold)
        np.savez_compressed(output.with_suffix('.npz'), distances=distances[:, :20],
            predictions=predictions[:, :20], correct=correct, error_m=errors,
            database_index_n_pos=ds.database_index_n_pos, query_index_n_pos=ds.query_index_n_pos)
        with output.with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.writer(handle)
            writer.writerow(['query_row','query_frame','top1_db_row','top1_db_frame','correct','error_m','descriptor_l2_squared','gt_count','first_gt_rank'])
            for i,(pred,g) in enumerate(zip(predictions,gt)):
                ranks=np.flatnonzero(np.isin(pred,g))
                writer.writerow([i,int(ds.query_index_n_pos[i,0]),int(pred[0]),int(ds.database_index_n_pos[pred[0],0]),
                    int(correct[i]),float(errors[i]),float(distances[i,0]),len(g),int(ranks[0])+1 if len(ranks) else ''])
        return metrics['F1']

    tester._print_f1 = capture
    scores = tester.do_test()
    metrics.update({f'Recall@{k}':float(v) for k,v in zip((1,5,10,20,'1percent'),scores)})
    write_json(output.with_name(output.stem+'_diagnostics.json'), diagnostics)
    write_json(output,metrics)


def audit_trial(checkpoint, seed, arm, teacher, previous=None):
    import torch
    if previous and digest(checkpoint) != previous['checkpoint_sha256']:
        raise ValueError('Recorded checkpoint changed')
    result = audit_checkpoint(checkpoint,1,teacher=teacher)
    saved = torch.load(checkpoint,map_location='cpu')
    expected = dict(alignment_loss=LOSSES[arm], batch_size=8, phase_loss_weight=.1,
        cross_fusion_loss_weight=0.,learning_rate_initial=5e-5,scheduler_gamma=.8,
        deterministic_algorithms=True,cublas_workspace_config=':4096:8',
        cross_hard_negative_count=5,sampling_seed=seed)
    if saved['training_settings'] != expected or saved['spectral_config'].get('pure_phase',True) or not saved['cross_fusion_enabled']:
        raise ValueError('Unexpected saved training settings')
    for state in saved['optimizer']['state'].values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError('Nonfinite optimizer')
    return result


def audit_seed_start(folder, seed):
    import hashlib
    import torch
    path = Path(folder)/'seed_start.json'
    actual = json.loads(path.read_text())
    expected = dict(seed=seed, sampling_seed=seed, torch_rng_sha256=hashlib.sha256(
        torch.Generator().manual_seed(seed).get_state().numpy().tobytes()).hexdigest())
    if actual != expected:
        raise ValueError('Training model seed provenance differs from declared seed')
    return bind_artifact(path)


def run(root=None):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    if root is None:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        root = PROJECT/'runs'/('mulran_f1_replication_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H%M%S'))
        root.mkdir(exist_ok=False)
        prior = json.loads((REFERENCE/'run.json').read_text())
        prior_protocol = json.loads((REFERENCE/'protocol.json').read_text())
        if prior['status']!='COMPLETE' or digest(REFERENCE/'protocol.json')!=prior['protocol_sha256'] or digest(REFERENCE/'selection.json')!=prior['selection_sha256']:
            raise ValueError('Reference evidence changed')
        for name,sha in prior_protocol['source_sha256'].items():
            if digest(SOURCE/name)!=sha:
                raise ValueError('Historical source changed: '+name)
        protocol = dict(created_at=now(), seeds=SEEDS, new_training_seeds=[1,2], epochs=1,
            initial={k:copy.deepcopy(prior_protocol['initial'][k]) for k in ('l','r')},
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
            source_sha256={str(p.relative_to(SOURCE)):digest(p) for directory in ('config','dataset','modules','scripts','trainer','utils','tests') for p in (SOURCE/directory).rglob('*.py')},
            reference_run=bind_artifact(REFERENCE/'run.json'), selection=RULE,
            fixed_phase_auxiliary=.1,fixed_fusion_supervision=0.,
            caveat='Exploratory after earlier benchmark inspection; fixed seed-0 L5/R5, only cross stage replicated. No seed selection. Official F1 is per-split maximum over 1000 test thresholds; frozen validation threshold is a separate diagnostic.')
        for arm,row in prior['configurations'].items():
            if digest(row['checkpoint'])!=row['checkpoint_sha256']:
                raise ValueError('Seed-0 checkpoint changed')
        write_json(root/'protocol.json',protocol)
        (root/'driver.py').write_text(Path(__file__).read_text())
        report=dict(status='RUNNING',phase='PREFLIGHT',started_at=now(),protocol_sha256=digest(root/'protocol.json'),
                    seeds={'0':copy.deepcopy(prior['configurations'])})
        write_json(root/'run.json',report)
    else:
        root=Path(root)
        report=json.loads((root/'run.json').read_text())
        protocol=json.loads((root/'protocol.json').read_text())
        if digest(root/'protocol.json')!=report['protocol_sha256'] or digest(root/'driver.py')!=digest(Path(__file__)):
            raise ValueError('Resume protocol/driver changed')
    print('RUN',root,flush=True)
    Path('/tmp/rlpr_f1_replication_run_path').write_text(str(root))

    def save():write_json(root/'run.json',report)
    def verify():
        verify_artifact(protocol['reference_run'])
        for name,sha in protocol['source_sha256'].items():
            if digest(SOURCE/name)!=sha:raise ValueError('Frozen source changed: '+name)
        for initial in protocol['initial'].values():
            if digest(initial['path'])!=initial['sha256']:raise ValueError('Initialization changed')
        for arms in report['seeds'].values():
            for row in arms.values():
                if digest(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Recorded checkpoint changed')
                for artifact in row.get('artifacts',{}).values():verify_artifact(artifact)
                verify_metrics(row)
    def child(seed,arm,mode,split='sejong'):
        verify()
        folder=root/f'seed{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
        env=environment(root,seed,arm,protocol['initial'],mode,split)
        name='train' if mode=='train' else split
        logfile=folder/(name+'.log');retry=0
        while logfile.exists():
            retry+=1;logfile=folder/(name+f'.retry{retry}.log')
        argv=[sys.executable,'-u',str(root/'driver.py'),'--child',mode,'--folder',str(folder),'--seed',str(seed),'--split',split]
        if mode=='evaluate' and split!='sejong':
            diag=json.loads((folder/'sejong_diagnostics.json').read_text())
            argv += ['--threshold',str(diag['best_threshold']['threshold'])]
        with logfile.open('x') as handle:
            subprocess.run(argv,cwd=SOURCE,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    def evaluation(seed,arm,split):
        folder=root/f'seed{seed}'/arm
        row=report['seeds'][str(seed)][arm]
        key=split+'_metrics'
        if key not in row.setdefault('artifacts',{}):
            child(seed,arm,'evaluate',split)
            for suffix in ('.json','_diagnostics.json','.npz','.csv'):
                row['artifacts'][split+suffix]=bind_artifact(folder/(split+suffix))
            row['artifacts'][key]=bind_artifact(folder/(split+'.json'))
            if split=='sejong':
                row['validation']=json.loads((folder/'sejong.json').read_text())
                row['validation_artifact']=row['artifacts'][key]
            else:
                row.setdefault('benchmarks',{})[split]=json.loads((folder/(split+'.json')).read_text())
            save()

    try:
        report.update(status='RUNNING');report.pop('error',None);save()
        verify()  # Check recorded bindings before any resumed trial can rebind them.
        # Re-extract seed 0 diagnostics; require exact parity with historical metrics.
        for arm in LOSSES:
            row=report['seeds']['0'][arm]
            for split in ('sejong','dcc','riverside'):
                folder=root/'seed0'/arm;folder.mkdir(parents=True,exist_ok=True)
                if split+'_metrics' not in row.setdefault('artifacts',{}):
                    env=environment(root,0,arm,protocol['initial'],'evaluate',split)
                    argv=[sys.executable,'-u',str(root/'driver.py'),'--child','evaluate','--folder',str(folder),'--checkpoint',row['checkpoint'],'--split',split]
                    if split!='sejong':
                        argv+=['--threshold',str(json.loads((folder/'sejong_diagnostics.json').read_text())['best_threshold']['threshold'])]
                    verify()
                    with (folder/(split+'.log')).open('w') as handle:
                        subprocess.run(argv,cwd=SOURCE,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
                    expected=row['validation'] if split=='sejong' else row['benchmarks'][split]
                    actual=json.loads((folder/(split+'.json')).read_text())
                    if any(not math.isclose(actual[k],v,abs_tol=1e-10) for k,v in expected.items()):
                        raise ValueError('Seed-0 diagnostics changed historical metrics')
                    for suffix in ('.json','_diagnostics.json','.npz','.csv'):
                        row['artifacts'][split+suffix]=bind_artifact(folder/(split+suffix))
                    row['artifacts'][split+'_metrics']=bind_artifact(folder/(split+'.json'))
                    if split=='sejong':row['validation_artifact']=row['artifacts']['sejong_metrics']
                    save()
        for seed in (1,2):
            records=report['seeds'].setdefault(str(seed),{})
            for arm in LOSSES:
                report['phase']=f'seed{seed}/{arm}/TRAIN';save()
                folder=root/f'seed{seed}'/arm
                checkpoint=folder/'weights/1.pth.tar'
                if not checkpoint.exists():child(seed,arm,'train')
                audited=audit_trial(checkpoint,seed,arm,protocol['initial']['r']['path'],records.get(arm))
                records[arm]=dict(records.get(arm,{}),**audited)
                records[arm].setdefault('artifacts',{})['seed_start']=audit_seed_start(folder,seed)
                for suffix in ('.npz','.json'):
                    records[arm]['artifacts']['mining'+suffix]=bind_artifact(folder/('cache/cross_negatives_epoch_1'+suffix))
                save()
                if arm=='infonce':
                    if audited['sampling']!=records['signed']['sampling']:raise ValueError('Paired random sampling differs')
                    a=np.load(root/f'seed{seed}/signed/cache/cross_negatives_epoch_1.npz')
                    b=np.load(folder/'cache/cross_negatives_epoch_1.npz')
                    if a.files!=b.files or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Paired mined negatives differ')
                report['phase']=f'seed{seed}/{arm}/VALIDATION';save()
                evaluation(seed,arm,'sejong')
                print('VALIDATION',seed,arm,records[arm]['validation'],flush=True)
        verify()
        lock_selection(root,report);save()
        for seed in (1,2):
            for arm in LOSSES:
                for split in ('dcc','riverside'):
                    report['phase']=f'seed{seed}/{arm}/{split}';save()
                    evaluation(seed,arm,split)
                    print('TEST',seed,arm,split,report['seeds'][str(seed)][arm]['benchmarks'][split],flush=True)
        verify()
        if digest(root/'selection.json')!=report['selection_sha256']:raise ValueError('Selection changed')
        summary={}
        for split in ('sejong','dcc','riverside'):
            summary[split]={}
            for arm in LOSSES:
                rows=[report['seeds'][str(s)][arm]['validation'] if split=='sejong' else report['seeds'][str(s)][arm]['benchmarks'][split] for s in SEEDS]
                summary[split][arm]={k:dict(mean=statistics.mean([r[k] for r in rows]),sd=statistics.stdev([r[k] for r in rows])) for k in ('F1','Recall@1')}
            summary[split]['paired_infonce_minus_signed']={k:dict(mean=statistics.mean([ (report['seeds'][str(s)]['infonce']['validation'] if split=='sejong' else report['seeds'][str(s)]['infonce']['benchmarks'][split])[k] - (report['seeds'][str(s)]['signed']['validation'] if split=='sejong' else report['seeds'][str(s)]['signed']['benchmarks'][split])[k] for s in SEEDS])) for k in ('F1','Recall@1')}
        write_json(root/'results_summary.json',dict(selected_loss=report['selected_loss'],summary=summary,caveat=protocol['caveat']))
        report.update(status='COMPLETE',phase='COMPLETE',finished_at=now());save()
        print('COMPLETE',root,report['selected_loss'],flush=True)
    except BaseException as error:
        report.update(status='FAILED',error=repr(error),failed_at=now());save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path)
    parser.add_argument('--child',choices=('train','evaluate'))
    parser.add_argument('--folder',type=Path)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--split',choices=('sejong','dcc','riverside'),default='sejong')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--threshold',type=float)
    parser.add_argument('--variant',choices=('spectral_word','rlpr'),default='spectral_word')
    args=parser.parse_args()
    if args.child=='train':child_train(args.seed,args.folder)
    elif args.child=='evaluate':evaluate(args.checkpoint or args.folder/'weights/1.pth.tar',args.folder/(args.split+'.json'),args.threshold,args.variant)
    else:run(args.run_root)
