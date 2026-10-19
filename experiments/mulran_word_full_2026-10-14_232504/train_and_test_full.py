import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

PROJECT=Path('/media/rwj/ZX1 2TB/RLPR')
SOURCE=PROJECT/'src_spectral'
RUN=Path(Path('/tmp/rlpr_word_full_run_path').read_text())
PYTHON='/mnt/data300/envs/rlpr/bin/python'
DATA=PROJECT/'processed/mulran_official'
BASELINE=json.loads((PROJECT/'runs/mulran_official_eval_2026-10-14/results.json').read_text())
report={'status':'RUNNING','phase':'WAITING_FOR_DATA','model_variant':'spectral_word',
        'source_root':str(SOURCE),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        'run_root':str(RUN),'data_root':str(DATA),'started_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'stages':{},'evaluations':{},'training_config':{
            'lidar_epochs':5,'radar_epochs':5,'alignment_epochs':1,'batch_size':8,'seed':0,
            'learning_rate':5e-5,'epoch_lr_gamma':.8,'positive_m':9,'negative_m':12,'negatives_per_tuple':10,
            'phase_loss_weight':.1,'cross_fusion_loss_weight':.5,'alignment':'welch_coherence_v1',
            'teacher_frozen':True,'data':'Full Sejong training regions + KAIST01/02',
            'test_splits':['dcc','riverside'],'test_policy':'Independent descriptors'},
        'torch':torch.__version__,'cuda_available':torch.cuda.is_available()}
assert report['cuda_available'],'GPU is required for this full run'
report['gpu']=torch.cuda.get_device_name(0)
def save():
    temporary=RUN/'run.json.tmp'
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(RUN/'run.json')
save()
env=os.environ.copy()
env.update(RLPR_MODEL_VARIANT='spectral_word',RLPR_SPECTRAL_CROSS_FUSION='1',RLPR_RESUME='0',
           RLPR_MULRAN_DATA=str(DATA),RLPR_CHECKPOINT_PATH='',RLPR_TEST_ZERO_SHOT='0',
           RLPR_PHASE_LOSS_WEIGHT='.1',RLPR_CROSS_FUSION_WEIGHT='.5',
           TORCH_HOME=str(PROJECT/'cache/torch'),MPLCONFIGDIR=str(RUN/'matplotlib_cache'),
           OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
try:
    print('Waiting for full Sejong/KAIST preparation',flush=True)
    while True:
        try: preparation=json.loads((RUN/'preparation.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(2);continue
        if preparation['status']=='FAILED_OR_INTERRUPTED':raise RuntimeError('Training data preparation failed')
        if preparation['status']=='COMPLETE':break
        time.sleep(2)
    report['preparation']=preparation;save()
    print('Full data ready:',preparation['total_database_frames'],'database',preparation['total_query_frames'],'queries',flush=True)
    for stage,modality,epochs in [('l_mulran','l',5),('r_mulran','r',5),('r2l_mulran',None,1)]:
        report['phase']='TRAINING_'+stage.upper();save()
        stage_root=RUN/stage
        if stage_root.exists():raise FileExistsError(f'Refusing to overwrite training directory {stage_root}')
        env['RLPR_TRAINING_ROOT']=str(stage_root)
        env['RLPR_PRETRAIN_L']=str(RUN/'l_mulran/weights/5.pth.tar')
        env['RLPR_PRETRAIN_R']=str(RUN/'r_mulran/weights/5.pth.tar')
        if modality:
            command=[PYTHON,'-u','train_unimodal.py','--dataset','mulran','--modality',modality,'--model','spectral_word']
        else:
            command=[PYTHON,'-u','train_cross_modal.py','--dataset','mulran','--model','spectral_word','--cross-fusion']
        log_path=RUN/(stage+'.log')
        report['stages'][stage]={'status':'RUNNING','epochs':epochs,'command':command,'log':str(log_path)};save()
        print('Starting',stage,epochs,'epochs, batch size 8',flush=True)
        start=time.monotonic()
        with log_path.open('x') as log:
            subprocess.run(command,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        checkpoint=stage_root/'weights'/f'{epochs}.pth.tar'
        metadata=torch.load(checkpoint,map_location='cpu')
        if metadata['epoch']!=epochs or metadata['model_variant']!='spectral_word':
            raise ValueError('Unexpected trained checkpoint metadata')
        losses=re.findall(r'epoch (\d+) finishes\s+loss \[([^\]]+)\]',log_path.read_text())
        report['stages'][stage].update(status='COMPLETE',seconds=time.monotonic()-start,
                                       checkpoint=str(checkpoint),losses={n:float(v) for n,v in losses},
                                       spectral_config=metadata['spectral_config'])
        del metadata
        save();print('Completed',stage,report['stages'][stage]['losses'],flush=True)
    env['RLPR_CHECKPOINT_PATH']=str(RUN/'r2l_mulran/weights/1.pth.tar')
    env['RLPR_TRAINING_ROOT']=str(RUN/'r2l_mulran')
    for split in ('dcc','riverside'):
        report['phase']='EVALUATING_'+split.upper();save()
        env['RLPR_MULRAN_TEST_SPLIT']=split
        command=[PYTHON,'-u','test_cross_modal.py','--dataset','mulran','--model','spectral_word','--cross-fusion']
        log_path=RUN/(split+'_evaluation.log')
        print('Evaluating trained Word model on',split,flush=True)
        with log_path.open('x') as log:
            subprocess.run(command,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=log_path.read_text();metrics={}
        for name in ['Recall@1','Recall@5','Recall@10','Recall@20','Recall@1percent']:
            match=re.search(re.escape(name)+r':\s*([0-9.]+)',text)
            if not match:raise ValueError('Missing metric '+name)
            metrics[name]=float(match.group(1))
        match=re.search(r'f1 score:\s*([0-9.eE+-]+)',text)
        if not match:raise ValueError('Missing finite max F1')
        metrics['F1']=float(match.group(1))
        official=BASELINE['splits'][split]['metrics'];paper=BASELINE['splits'][split]['paper_metrics']
        report['evaluations'][split]={'metrics':metrics,'official_checkpoint_metrics':official,'paper_metrics':paper,
                                     'difference_vs_official':{k:metrics[k]-official[k] for k in metrics},
                                     'difference_vs_paper':{k:metrics[k]-paper[k] for k in paper},
                                     'command':command,'log':str(log_path)}
        save();print(split,json.dumps(report['evaluations'][split],ensure_ascii=False),flush=True)
    report['status']='COMPLETE';report['phase']='DONE'
    report['finished_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
except BaseException:
    report['status']='FAILED_OR_INTERRUPTED';save();raise
save()
print('Training and evaluation complete:',RUN,flush=True)
