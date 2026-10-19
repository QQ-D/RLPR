"""Run all fixed temperature/seed combinations in the background, then report."""
import argparse
import copy
import csv
import fcntl
import hashlib
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
    bind_artifact,verify_artifact,verify_metrics,audit_trial as audit_baseline)
from scripts.spatial_positive_ablation import lock_payload,freeze_official,stats

REFERENCE=PROJECT/'runs/mulran_f1_replication_2026-10-19_160042'
REPORT_BUILDER=PROJECT/'reports/2026-10-19/build_temperature_report.py'
TEMPERATURES={'tau_007':.07,'tau_003':.03,'tau_010':.10}
SEEDS=(0,1,2)


def choose_temperature(records):
    if set(records)!=set(TEMPERATURES):raise ValueError('All three temperatures are required')
    scores={}
    for arm,rows in records.items():
        if set(rows)!={str(s) for s in SEEDS}:raise ValueError('All three seed validations are required')
        values=[]
        for seed in SEEDS:
            row=rows[str(seed)].get('validation',{})
            value=tuple(row.get(k,float('nan')) for k in ('F1','Recall@1'))
            if not all(math.isfinite(v) for v in value):raise ValueError('Incomplete/nonfinite validation')
            values.append(value)
        scores[arm]=tuple(statistics.mean([v[i] for v in values]) for i in (0,1))
    return max(TEMPERATURES,key=lambda a:(*scores[a],-list(TEMPERATURES).index(a)))


def audit_temperature(path,temperature,seed,teacher,previous=None):
    import torch
    if previous and digest(path)!=previous['checkpoint_sha256']:raise ValueError('Recorded weights changed')
    result=audit_checkpoint(path,1,teacher=teacher)
    saved=torch.load(path,map_location='cpu')
    expected=dict(alignment_loss='infonce',batch_size=8,phase_loss_weight=.1,cross_fusion_loss_weight=0.,
        learning_rate_initial=5e-5,scheduler_gamma=.8,deterministic_algorithms=True,
        cublas_workspace_config=':4096:8',cross_hard_negative_count=5,sampling_seed=seed,
        contrastive_temperature=temperature)
    if saved['training_settings']!=expected or saved.get('alignment_loss')!='infonce':raise ValueError('Saved temperature/training settings differ')
    if saved['spectral_config'].get('pure_phase',True) or not saved['cross_fusion_enabled']:raise ValueError('Model configuration changed')
    for state in saved['optimizer']['state'].values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():raise ValueError('Nonfinite optimizer')
    return result


def train_child(folder,seed,temperature):
    import torch
    from scripts.registry import build_params,CROSS_MODAL,get_model_variant
    from trainer.temperature_cross_modal import TemperatureCrossModalTrainer
    initialize_seed(seed);torch.set_num_threads(4)
    write_json(folder/'seed_start.json',dict(seed=seed,sampling_seed=seed,
        torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
    params=build_params(epochs_override=1)
    if params.mulran_param.base_root!=str(TRAIN) or params.sampling_seed!=seed or not torch.cuda.is_available():
        raise ValueError('Training root, seed or GPU mismatch')
    cfg=CROSS_MODAL['mulran']
    trainer=TemperatureCrossModalTrainer(params,get_model_variant(cfg['model'],'spectral_word'),cfg['dataloader'],temperature)
    try:
        if len(trainer.train_loader.dataset)!=16568 or len(trainer.train_loader)!=2071:raise ValueError('Wrong training split')
        trainer.do_train()
    finally:trainer.writer.close()


def summarize(records):
    summary={};paired={}
    for split in ('sejong','dcc','riverside'):
        summary[split]={};paired[split]={}
        for arm in TEMPERATURES:
            values=[records[arm][str(s)]['validation'] if split=='sejong' else records[arm][str(s)]['benchmarks'][split] for s in SEEDS]
            summary[split][arm]=stats(values)
            if arm!='tau_007':
                deltas=[values[s]['F1']-(records['tau_007'][str(s)]['validation'] if split=='sejong' else records['tau_007'][str(s)]['benchmarks'][split])['F1'] for s in SEEDS]
                paired[split][arm]=dict(values=deltas,mean=statistics.mean(deltas),sd=statistics.stdev(deltas),positive_seed_count=sum(v>0 for v in deltas))
    return summary,paired


def run(root=None,regression_log=Path('/tmp/rlpr_temperature_regression.log')):
    import torch
    import numpy as np
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    if root is None:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        prior=json.loads((REFERENCE/'run.json').read_text());pp=json.loads((REFERENCE/'protocol.json').read_text())
        if prior['status']!='COMPLETE' or digest(REFERENCE/'protocol.json')!=prior['protocol_sha256'] or digest(REFERENCE/'selection.json')!=prior['selection_sha256']:
            raise ValueError('Baseline evidence changed')
        for name,sha in pp['source_sha256'].items():
            if digest(SOURCE/name)!=sha:raise ValueError('Historical source changed: '+name)
        tests=regression_log.read_text();match=re.search(r'Ran (\d+) tests',tests)
        if not match or '\nOK\n' not in tests:raise ValueError('Passing complete regression log required')
        root=PROJECT/'runs'/('mulran_temperature_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H%M%S'));root.mkdir(exist_ok=False)
        protocol=dict(created_at=now(),reference_run=bind_artifact(REFERENCE/'run.json'),
            initial=copy.deepcopy(pp['initial']),temperatures=TEMPERATURES,seeds=list(SEEDS),new_trials=6,
            fixed_unary_seed=0,cross_epochs=1,positive_definition='Original synchronous diagonal12-way single-positive InfoNCE; no spatial-positive masks',
            selection='Mean Sejong F1 across three seeds, then mean Recall@1, then temperature .07 on exact tie; no seed0 early screening or individual-seed selection',
            test_policy='All nine validations and selection lock before any new DCC/Riverside evaluation',
            caveat='Exploratory: benchmarks were inspected previously. Fixed seed0 L5/R5; only cross stage replicated. Temperature.07 results reused after source/checkpoint/metric verification.',
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
            source_sha256={str(p.relative_to(SOURCE)):digest(p) for d in ('config','dataset','modules','scripts','trainer','utils','tests') for p in (SOURCE/d).rglob('*.py')},
            report_builder=bind_artifact(REPORT_BUILDER),official_results=freeze_official(root),
            regression_log=bind_artifact(regression_log),regression_tests=int(match.group(1)))
        write_json(root/'protocol.json',protocol);shutil.copyfile(regression_log,root/'regression_tests.log')
        (root/'driver.py').write_text(Path(__file__).read_text())
        report=dict(status='RUNNING',phase='PREFLIGHT',started_at=now(),protocol_sha256=digest(root/'protocol.json'),
            configurations={'tau_007':{s:copy.deepcopy(v['infonce']) for s,v in prior['seeds'].items()},'tau_003':{},'tau_010':{}},
            training_status='RUNNING',evaluation_status='PENDING',report_status='PENDING')
        write_json(root/'run.json',report)
    else:
        root=Path(root);report=json.loads((root/'run.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
        if digest(root/'protocol.json')!=report['protocol_sha256'] or digest(root/'driver.py')!=digest(Path(__file__)):
            raise ValueError('Resume requires unchanged protocol and frozen driver')
    print('RUN',root,flush=True);Path('/tmp/rlpr_temperature_run_path').write_text(str(root))
    def save():write_json(root/'run.json',report)
    def verify():
        for key in ('reference_run','report_builder','official_results'):verify_artifact(protocol[key])
        if digest(root/'regression_tests.log')!=protocol['regression_log']['sha256']:raise ValueError('Regression log changed')
        for name,sha in protocol['source_sha256'].items():
            if digest(SOURCE/name)!=sha:raise ValueError('Frozen source changed: '+name)
        for item in protocol['initial'].values():
            if digest(item['path'])!=item['sha256']:raise ValueError('Unary weights changed')
        for records in report['configurations'].values():
            for row in records.values():
                if digest(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Checkpoint changed')
                for artifact in row.get('artifacts',{}).values():verify_artifact(artifact)
                verify_metrics(row)
        if report.get('selection_sha256') and digest(root/'selection.json')!=report['selection_sha256']:raise ValueError('Selection changed')
    def invoke(arm,seed,task,split='sejong'):
        verify();folder=root/arm/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
        env=environment(root,seed,'infonce',protocol['initial'],'train' if task=='train' else 'evaluate',split)
        env['RLPR_TRAINING_ROOT']=str(folder)
        argv=[sys.executable,'-u',str(root/'driver.py'),'--child',task,'--folder',str(folder),'--seed',str(seed),'--arm',arm,'--split',split]
        if task=='evaluate' and split!='sejong':
            argv+=['--threshold',str(json.loads((folder/'sejong_diagnostics.json').read_text())['best_threshold']['threshold'])]
        label='train' if task=='train' else split;log=folder/(label+'.log');n=0
        while log.exists():n+=1;log=folder/(label+f'.retry{n}.log')
        with log.open('x') as handle:subprocess.run(argv,cwd=SOURCE,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    def evaluation(arm,seed,split):
        row=report['configurations'][arm][str(seed)];key=split+'_metrics';folder=root/arm/f'seed{seed}'
        if key in row.setdefault('artifacts',{}):verify_metrics(row);return
        invoke(arm,seed,'evaluate',split)
        for suffix in ('.json','_diagnostics.json','.npz','.csv'):row['artifacts'][split+suffix]=bind_artifact(folder/(split+suffix))
        row['artifacts'][key]=bind_artifact(folder/(split+'.json'))
        if split=='sejong':row['validation']=json.loads((folder/'sejong.json').read_text())
        else:row.setdefault('benchmarks',{})[split]=json.loads((folder/(split+'.json')).read_text())
        verify_metrics(row);save()
    def runtime():return dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),source_commit=protocol['source_commit'])
    try:
        report.update(status='RUNNING');report.pop('error',None);save();verify()
        launch=runtime()
        if (root/'runtime_at_launch.json').exists():
            if json.loads((root/'runtime_at_launch.json').read_text())!=launch:raise ValueError('Runtime changed')
        else:write_json(root/'runtime_at_launch.json',launch)
        for seed in SEEDS:
            row=report['configurations']['tau_007'][str(seed)]
            audit_baseline(row['checkpoint'],seed,'infonce',protocol['initial']['r']['path'],row)
        for arm in ('tau_003','tau_010'):
            for seed in SEEDS:
                records=report['configurations'][arm];previous=records.get(str(seed));folder=root/arm/f'seed{seed}';checkpoint=folder/'weights/1.pth.tar'
                report['phase']=f'{arm}/seed{seed}/TRAIN';save()
                if not checkpoint.exists():invoke(arm,seed,'train')
                audit=audit_temperature(checkpoint,TEMPERATURES[arm],seed,protocol['initial']['r']['path'],previous)
                baseline=report['configurations']['tau_007'][str(seed)]
                if audit['sampling']!=baseline['sampling']:raise ValueError('Paired sampler/worker streams differ')
                oldcache=Path(baseline['checkpoint']).parents[1]/'cache/cross_negatives_epoch_1.npz'
                with np.load(oldcache) as a,np.load(folder/'cache/cross_negatives_epoch_1.npz') as b:
                    if a.files!=b.files or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Paired mined negatives differ')
                row=dict(previous or {},**audit);row.setdefault('artifacts',{})['seed_start']=audit_seed_start(folder,seed)
                for suffix in ('.json','.npz'):row['artifacts']['mining'+suffix]=bind_artifact(folder/('cache/cross_negatives_epoch_1'+suffix))
                records[str(seed)]=row;save();report['phase']=f'{arm}/seed{seed}/VALIDATION';save()
                evaluation(arm,seed,'sejong');print('VALIDATION',arm,seed,row['validation'],flush=True)
        verify();selected=choose_temperature(report['configurations'])
        payload=dict(selected=selected,temperature=TEMPERATURES[selected],rule=protocol['selection'],
            evidence={a:{s:dict(validation=r['validation'],checkpoint_sha256=r['checkpoint_sha256']) for s,r in rows.items()} for a,rows in report['configurations'].items()})
        report['selection_sha256']=lock_payload(root/'selection.json',payload,report.get('selection_sha256'))
        report.update(selected_configuration=selected,training_status='COMPLETE',evaluation_status='RUNNING');save()
        for arm in ('tau_003','tau_010'):
            for seed in SEEDS:
                for split in ('dcc','riverside'):
                    report['phase']=f'{arm}/seed{seed}/{split}';save();evaluation(arm,seed,split)
                    print('TEST',arm,seed,split,report['configurations'][arm][str(seed)]['benchmarks'][split],flush=True)
        verify();summary,paired=summarize(report['configurations'])
        write_json(root/'results_summary.json',dict(selected_configuration=selected,selected_temperature=TEMPERATURES[selected],summary=summary,
            paired_F1_delta_from_007=paired,caveat=protocol['caveat']))
        with (root/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['split','temperature','seed','Recall@1','Recall@5','Recall@10','Recall@20','Recall@1percent','F1']);writer.writeheader()
            for split in ('dcc','riverside'):
                for arm,t in TEMPERATURES.items():
                    for seed in SEEDS:writer.writerow(dict(split=split,temperature=t,seed=seed,**report['configurations'][arm][str(seed)]['benchmarks'][split]))
        audits={}
        for arm,t in TEMPERATURES.items():
            audits[arm]={}
            for seed in SEEDS:
                row=report['configurations'][arm][str(seed)]
                audits[arm][str(seed)]=(audit_baseline(row['checkpoint'],seed,'infonce',protocol['initial']['r']['path'],row) if arm=='tau_007'
                    else audit_temperature(row['checkpoint'],t,seed,protocol['initial']['r']['path'],row))
        verify();completion=runtime()
        if completion!=launch:raise ValueError('Completion runtime differs')
        write_json(root/'runtime_at_completion.json',completion)
        write_json(root/'completion_verification.json',dict(all_checks_passed=True,verified_at=now(),checkpoint_count=9,new_training_trials=6,
            paired_sampling_match=True,paired_mining_match=True,teacher_frozen=True,metric_checkpoint_binding_match=True,
            regression_tests=protocol['regression_tests'],source_files=len(protocol['source_sha256']),checkpoint_audits=audits))
        report.update(phase='REPORT',evaluation_status='COMPLETE',report_status='RUNNING',results_finished_at=now());save()
        verify_artifact(protocol['report_builder'])
        with (root/'report.log').open('a') as handle:subprocess.run(['/usr/bin/python3',str(REPORT_BUILDER),'--root',str(root)],cwd=SOURCE,stdout=handle,stderr=subprocess.STDOUT,check=True)
        pdf=root/'report.pdf';text=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
        if '\ufffd' in text or any(f"{summary[s][a]['F1']['mean']:.4f}" not in text for s in ('dcc','riverside') for a in TEMPERATURES):raise ValueError('PDF metrics/text verification failed')
        write_json(root/'report_verification.json',dict(pdf=bind_artifact(pdf),metric_text_verified=True,pdfinfo=subprocess.check_output(['pdfinfo',str(pdf)],text=True)))
        report.update(status='COMPLETE',phase='COMPLETE',report_status='COMPLETE',finished_at=now());save();print('COMPLETE',root,TEMPERATURES[selected],flush=True)
    except BaseException as error:
        report.update(status='FAILED',error=repr(error),failed_at=now());save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path);parser.add_argument('--regression-log',type=Path,default=Path('/tmp/rlpr_temperature_regression.log'))
    parser.add_argument('--child',choices=('train','evaluate'));parser.add_argument('--folder',type=Path)
    parser.add_argument('--seed',type=int,choices=SEEDS,default=0);parser.add_argument('--arm',choices=tuple(TEMPERATURES),default='tau_003')
    parser.add_argument('--split',choices=('sejong','dcc','riverside'),default='sejong');parser.add_argument('--threshold',type=float)
    args=parser.parse_args()
    if args.child=='train':train_child(args.folder,args.seed,TEMPERATURES[args.arm])
    elif args.child=='evaluate':evaluate(args.folder/'weights/1.pth.tar',args.folder/(args.split+'.json'),args.threshold)
    else:
        with (PROJECT/'runs/.temperature_ablation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args.run_root,args.regression_log)
