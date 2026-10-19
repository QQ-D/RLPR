"""Background F1-first geographic-positive experiment with automatic audit/report."""
import argparse
import copy
import csv
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

SOURCE=Path('/media/rwj/ZX1 2TB/RLPR/src_spectral')
sys.path.insert(0,str(SOURCE))
from scripts.amplitude_ablation import PROJECT,TRAIN,now,digest,write_json,audit_checkpoint
from scripts.f1_priority_replication import (environment,initialize_seed,evaluate,audit_seed_start,
    bind_artifact,verify_artifact,verify_metrics,audit_trial as audit_single)
from scripts.fusion_weight_ablation import verify_control

REFERENCE=PROJECT/'runs/mulran_f1_replication_2026-10-19_160042'
REPORT_BUILDER=PROJECT/'reports/2026-10-19/build_spatial_positive_report.py'
OFFICIAL_RESULTS=PROJECT/'runs/mulran_official_eval_2026-10-14/results.json'


def score(row):
    values=tuple(row.get(k,float('nan')) for k in ('F1','Recall@1'))
    if not all(math.isfinite(v) for v in values):raise ValueError('Incomplete or nonfinite validation')
    return values


def promote(candidate,baseline):return score(candidate)>score(baseline)


def select_family(baseline,trials,seeds):
    if seeds not in ([0],[0,1,2]) or set(trials)!={str(s) for s in seeds}:
        raise ValueError('Incomplete predeclared seed set')
    means={}
    for arm,records in [('single',baseline),('multi',trials)]:
        try:values=[score(records[str(s)]['validation']) for s in seeds]
        except KeyError:raise ValueError('Missing validation evidence') from None
        means[arm]=tuple(statistics.mean([v[i] for v in values]) for i in (0,1))
    return 'multi' if means['multi']>means['single'] else 'single'


def lock_payload(path,payload,previous_sha=None):
    path=Path(path)
    if path.exists():
        old=json.loads(path.read_text())
        if any(old.get(k)!=v for k,v in payload.items()):raise ValueError('Locked evidence changed')
        if previous_sha and digest(path)!=previous_sha:raise ValueError('Locked file changed')
    elif previous_sha:raise ValueError('Previously recorded lock disappeared')
    else:write_json(path,dict(payload,locked_at=now()))
    return digest(path)


def freeze_official(root,source=OFFICIAL_RESULTS):
    original=bind_artifact(source)
    snapshot=Path(root)/'official_results.json'
    shutil.copyfile(source,snapshot)
    result=bind_artifact(snapshot)
    if result['sha256']!=original['sha256']:raise ValueError('Official comparison changed while copying')
    return result


def audit_candidate(path,seed,teacher,previous=None):
    import torch
    if previous and digest(path)!=previous['checkpoint_sha256']:raise ValueError('Recorded checkpoint changed')
    result=audit_checkpoint(path,1,teacher=teacher)
    saved=torch.load(path,map_location='cpu')
    expected=dict(alignment_loss='spatial_multi_infonce_v1',batch_size=8,phase_loss_weight=.1,
        cross_fusion_loss_weight=0.,learning_rate_initial=5e-5,scheduler_gamma=.8,
        deterministic_algorithms=True,cublas_workspace_config=':4096:8',cross_hard_negative_count=5,
        sampling_seed=seed,positive_radius_m=9.,negative_radius_m=12.,temperature=.07,
        positive_reduction='log_probability_mass',negative_scope='within_original_12_frame_tuple')
    if saved['training_settings']!=expected or saved.get('alignment_loss')!='spatial_multi_infonce_v1':
        raise ValueError('Unexpected multi-positive training settings')
    if saved['spectral_config'].get('pure_phase',True) or not saved['cross_fusion_enabled']:
        raise ValueError('Amplitude/fusion computation differs from reference')
    for state in saved['optimizer']['state'].values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():raise ValueError('Nonfinite optimizer')
    return result


def train_child(folder,seed,mode):
    import torch
    import hashlib
    from scripts.registry import build_params,CROSS_MODAL,get_model_variant
    from trainer.spatial_cross_modal import SpatialCrossModalTrainer
    initialize_seed(seed);torch.set_num_threads(4)
    write_json(folder/'seed_start.json',dict(seed=seed,sampling_seed=seed,
        torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
    params=build_params(epochs_override=1)
    dp=params.mulran_param
    if dp.base_root!=str(TRAIN) or params.sampling_seed!=seed or not torch.cuda.is_available():
        raise ValueError('Training root, seed or GPU mismatch')
    if dp.pos_dist_threshold!=9 or dp.neg_dist_threshold!=12:raise ValueError('Geographic thresholds changed')
    cfg=CROSS_MODAL['mulran']
    trainer=SpatialCrossModalTrainer(params,get_model_variant(cfg['model'],'spectral_word'),cfg['dataloader'],mode)
    try:
        if len(trainer.train_loader.dataset)!=16568 or len(trainer.train_loader)!=2071:raise ValueError('Training query count changed')
        trainer.do_train()
    finally:trainer.writer.close()


def stats(rows):
    return {metric:dict(mean=statistics.mean([r[metric] for r in rows]),
        sd=statistics.stdev([r[metric] for r in rows]) if len(rows)>1 else None,n=len(rows))
        for metric in ('F1','Recall@1')}


def run(root=None,regression_log=Path('/tmp/rlpr_spatial_regression.log')):
    import torch
    import numpy as np
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    if root is None:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        prior=json.loads((REFERENCE/'run.json').read_text())
        pp=json.loads((REFERENCE/'protocol.json').read_text())
        if prior['status']!='COMPLETE' or digest(REFERENCE/'protocol.json')!=prior['protocol_sha256']:
            raise ValueError('Incomplete or changed baseline experiment')
        if digest(REFERENCE/'selection.json')!=prior['selection_sha256'] or prior['selected_loss']!='infonce':
            raise ValueError('Baseline family selection changed')
        for name,sha in pp['source_sha256'].items():
            if digest(SOURCE/name)!=sha:raise ValueError('Historical source changed: '+name)
        tests=regression_log.read_text()
        match=re.search(r'Ran (\d+) tests',tests)
        if not match or '\nOK\n' not in tests:raise ValueError('Passing full regression log required')
        root=PROJECT/'runs'/('mulran_spatial_positives_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H%M%S'))
        root.mkdir(exist_ok=False)
        baseline={s:copy.deepcopy(rows['infonce']) for s,rows in prior['seeds'].items()}
        protocol=dict(created_at=now(),reference_run=bind_artifact(REFERENCE/'run.json'),
            initial=copy.deepcopy(pp['initial']),new_epochs_per_trial=1,new_sampling_seeds=[0,1,2],
            fixed_unary_seed=0,temperature=.07,positive_radius_m=9.,negative_radius_m=12.,
            objective='Mean over 12 anchors per original tuple: logsumexp(valid logits)-logsumexp(geographic-positive logits). Teacher radar detached. <=9m positive; >12m negative; other pairs ignored; no cross-tuple negatives.',
            screening='Seed0 candidate (Sejong F1, Recall@1) strictly exceeds seed0 baseline to promote; otherwise stop further seeds and retain baseline.',
            selection='Mean Sejong F1, then mean Recall@1, then single-positive control on exact tie; evaluated seed set is [0] or promoted [0,1,2]; no individual seed selection.',
            caveat='Exploratory after benchmark inspection. Fixed seed0 L5/R5; cross-stage replication only. Historical baselines reused after source/weight/metric verification. Gate stopping is disclosed; no test score controls promotion.',
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
            source_sha256={str(p.relative_to(SOURCE)):digest(p) for d in ('config','dataset','modules','scripts','trainer','utils','tests') for p in (SOURCE/d).rglob('*.py')},
            report_builder=bind_artifact(REPORT_BUILDER),official_results=freeze_official(root),
            regression_log=bind_artifact(regression_log),regression_tests=int(match.group(1)))
        write_json(root/'protocol.json',protocol);shutil.copyfile(regression_log,root/'regression_tests.log')
        (root/'driver.py').write_text(Path(__file__).read_text())
        report=dict(status='RUNNING',phase='PREFLIGHT',started_at=now(),protocol_sha256=digest(root/'protocol.json'),
            baseline=baseline,trials={},training_status='RUNNING',evaluation_status='PENDING',report_status='PENDING')
        write_json(root/'run.json',report)
    else:
        root=Path(root);report=json.loads((root/'run.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
        if digest(root/'protocol.json')!=report['protocol_sha256'] or digest(root/'driver.py')!=digest(Path(__file__)):
            raise ValueError('Resume requires identical protocol/driver')
    print('RUN',root,flush=True)
    Path('/tmp/rlpr_spatial_run_path').write_text(str(root))
    def save():write_json(root/'run.json',report)
    def rows():return list(report['baseline'].values())+list(report['trials'].values())+([report['control']] if 'control' in report else [])
    def verify():
        verify_artifact(protocol['reference_run']);verify_artifact(protocol['report_builder'])
        verify_artifact(protocol['official_results'])
        if digest(root/'regression_tests.log')!=protocol['regression_log']['sha256']:raise ValueError('Regression evidence changed')
        for name,sha in protocol['source_sha256'].items():
            if digest(SOURCE/name)!=sha:raise ValueError('Frozen source changed: '+name)
        for initial in protocol['initial'].values():
            if digest(initial['path'])!=initial['sha256']:raise ValueError('Unary initialization changed')
        for row in rows():
            if digest(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Recorded weights changed')
            for artifact in row.get('artifacts',{}).values():verify_artifact(artifact)
            verify_metrics(row)
        for name in ('screening','selection'):
            if report.get(name+'_sha256') and digest(root/(name+'.json'))!=report[name+'_sha256']:
                raise ValueError('Frozen '+name+' changed')
    def invoke(folder,seed,mode,task,split='sejong'):
        verify();folder.mkdir(parents=True,exist_ok=True)
        env=environment(root,seed,'infonce',protocol['initial'],'train' if task=='train' else 'evaluate',split)
        env['RLPR_TRAINING_ROOT']=str(folder)
        argv=[sys.executable,'-u',str(root/'driver.py'),'--child',task,'--folder',str(folder),'--seed',str(seed),'--mode',mode,'--split',split]
        if task=='evaluate' and split!='sejong':
            threshold=json.loads((folder/'sejong_diagnostics.json').read_text())['best_threshold']['threshold']
            argv+=['--threshold',str(threshold)]
        name='train' if task=='train' else split
        logfile=folder/(name+'.log');n=0
        while logfile.exists():n+=1;logfile=folder/(name+f'.retry{n}.log')
        with logfile.open('x') as handle:subprocess.run(argv,cwd=SOURCE,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    def evaluate_trial(row,folder,seed,mode,split):
        key=split+'_metrics'
        if key in row.setdefault('artifacts',{}):verify_metrics(row);return
        invoke(folder,seed,mode,'evaluate',split)
        metrics=json.loads((folder/(split+'.json')).read_text())
        for suffix in ('.json','_diagnostics.json','.npz','.csv'):
            row['artifacts'][split+suffix]=bind_artifact(folder/(split+suffix))
        row['artifacts'][key]=bind_artifact(folder/(split+'.json'))
        if split=='sejong':row['validation']=metrics
        else:row.setdefault('benchmarks',{})[split]=metrics
        verify_metrics(row);save()
    def train_trial(folder,seed,mode):
        key='control' if mode=='single' else str(seed)
        previous=report.get('control') if mode=='single' else report['trials'].get(key)
        report['phase']=f'{mode}/seed{seed}/TRAIN';save()
        path=folder/'weights/1.pth.tar'
        if not path.exists():invoke(folder,seed,mode,'train')
        audit=(audit_single(path,seed,'infonce',protocol['initial']['r']['path'],previous) if mode=='single'
               else audit_candidate(path,seed,protocol['initial']['r']['path'],previous))
        row=dict(previous or {},**audit)
        row.setdefault('artifacts',{})['seed_start']=audit_seed_start(folder,seed)
        for suffix in ('.npz','.json'):
            row['artifacts']['mining'+suffix]=bind_artifact(folder/('cache/cross_negatives_epoch_1'+suffix))
        baseline=report['baseline'][str(seed)]
        if audit['sampling']!=baseline['sampling']:raise ValueError('Paired sampler/worker random streams differ')
        original_cache=Path(baseline['checkpoint']).parents[1]/'cache/cross_negatives_epoch_1.npz'
        with np.load(original_cache) as a,np.load(folder/'cache/cross_negatives_epoch_1.npz') as b:
            if a.files!=b.files or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Paired mined negatives differ')
        if mode=='single':
            verify_control(path,baseline['checkpoint']);report['control_replay_bitwise_equal']=True;report['control']=row
        else:report['trials'][key]=row
        save();report['phase']=f'{mode}/seed{seed}/VALIDATION';save()
        evaluate_trial(row,folder,seed,mode,'sejong')
        if mode=='single' and row['validation']!=baseline['validation']:raise ValueError('Control validation replay differs')
        print('VALIDATION',mode,seed,row['validation'],flush=True)
        return row
    try:
        report.update(status='RUNNING');report.pop('error',None);save();verify()
        launch=dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),source_commit=protocol['source_commit'])
        if (root/'runtime_at_launch.json').exists():
            if json.loads((root/'runtime_at_launch.json').read_text())!=launch:raise ValueError('Runtime changed on resume')
        else:write_json(root/'runtime_at_launch.json',launch)
        train_trial(root/'control',0,'single')
        candidate=train_trial(root/'multi/seed0',0,'multi')
        gate=promote(candidate['validation'],report['baseline']['0']['validation'])
        evidence=dict(promoted=gate,rule=protocol['screening'],candidate=candidate['validation'],
            baseline=report['baseline']['0']['validation'],candidate_sha256=candidate['checkpoint_sha256'],
            baseline_sha256=report['baseline']['0']['checkpoint_sha256'])
        report['screening_sha256']=lock_payload(root/'screening.json',evidence,report.get('screening_sha256'))
        report['promoted']=gate;save()
        seeds=[0,1,2] if gate else [0]
        for seed in seeds[1:]:train_trial(root/f'multi/seed{seed}',seed,'multi')
        verify();selected=select_family(report['baseline'],report['trials'],seeds)
        payload=dict(selected=selected,seeds=seeds,rule=protocol['selection'],
            baseline={str(s):dict(validation=report['baseline'][str(s)]['validation'],checkpoint_sha256=report['baseline'][str(s)]['checkpoint_sha256']) for s in seeds},
            multi={s:dict(validation=r['validation'],checkpoint_sha256=r['checkpoint_sha256']) for s,r in report['trials'].items()})
        report['selection_sha256']=lock_payload(root/'selection.json',payload,report.get('selection_sha256'))
        report.update(selected_family=selected,evaluated_seeds=seeds,training_status='COMPLETE',evaluation_status='RUNNING');save()
        for seed in seeds:
            for split in ('dcc','riverside'):
                report['phase']=f'multi/seed{seed}/{split}';save()
                evaluate_trial(report['trials'][str(seed)],root/f'multi/seed{seed}',seed,'multi',split)
                print('TEST',seed,split,report['trials'][str(seed)]['benchmarks'][split],flush=True)
        verify()
        summary={}
        for split in ('sejong','dcc','riverside'):
            summary[split]={}
            for arm,records in [('single',report['baseline']),('multi',report['trials'])]:
                data=[records[str(s)]['validation'] if split=='sejong' else records[str(s)]['benchmarks'][split] for s in seeds]
                summary[split][arm]=stats(data)
        write_json(root/'results_summary.json',dict(selected_family=selected,promoted=gate,evaluated_seeds=seeds,
            summary=summary,caveat=protocol['caveat']))
        with (root/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['split','arm','seed','Recall@1','Recall@5','Recall@10','Recall@20','Recall@1percent','F1'])
            writer.writeheader()
            for split in ('dcc','riverside'):
                for arm,records in [('single',report['baseline']),('multi',report['trials'])]:
                    for seed in seeds:writer.writerow(dict(split=split,arm=arm,seed=seed,**records[str(seed)]['benchmarks'][split]))
        audits={'control':audit_single(report['control']['checkpoint'],0,'infonce',protocol['initial']['r']['path'],report['control'])}
        for seed in seeds:audits[str(seed)]=audit_candidate(report['trials'][str(seed)]['checkpoint'],seed,protocol['initial']['r']['path'],report['trials'][str(seed)])
        verify()
        write_json(root/'completion_verification.json',dict(all_checks_passed=True,verified_at=now(),
            control_replay_bitwise_equal=True,paired_sampling_match=True,paired_mining_match=True,teacher_frozen=True,
            checkpoint_audits=audits,regression_tests=protocol['regression_tests'],source_files=len(protocol['source_sha256'])))
        report.update(phase='REPORT',evaluation_status='COMPLETE',report_status='RUNNING',results_finished_at=now());save()
        verify_artifact(protocol['report_builder'])
        with (root/'report.log').open('a') as handle:
            subprocess.run(['/usr/bin/python3',str(REPORT_BUILDER),'--root',str(root)],cwd=SOURCE,stdout=handle,stderr=subprocess.STDOUT,check=True)
        pdf=root/'report.pdf'
        pdftext=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
        if '\ufffd' in pdftext or any(f"{summary[s][a]['F1']['mean']:.4f}" not in pdftext for s in ('dcc','riverside') for a in ('single','multi')):
            raise ValueError('PDF text/metrics verification failed')
        write_json(root/'report_verification.json',dict(pdf=bind_artifact(pdf),metric_text_verified=True,
            pdfinfo=subprocess.check_output(['pdfinfo',str(pdf)],text=True)))
        report.update(status='COMPLETE',phase='COMPLETE',report_status='COMPLETE',finished_at=now());save()
        print('COMPLETE',root,selected,flush=True)
    except BaseException as error:
        report.update(status='FAILED',error=repr(error),failed_at=now());save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path)
    parser.add_argument('--regression-log',type=Path,default=Path('/tmp/rlpr_spatial_regression.log'))
    parser.add_argument('--child',choices=('train','evaluate'))
    parser.add_argument('--folder',type=Path)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--mode',choices=('single','multi'),default='multi')
    parser.add_argument('--split',choices=('sejong','dcc','riverside'),default='sejong')
    parser.add_argument('--threshold',type=float)
    args=parser.parse_args()
    if args.child=='train':train_child(args.folder,args.seed,args.mode)
    elif args.child=='evaluate':evaluate(args.folder/'weights/1.pth.tar',args.folder/(args.split+'.json'),args.threshold)
    else:
        with (PROJECT/'runs/.spatial_positive.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args.run_root,args.regression_log)
