"""Read-only sensor retrieval diagnostic for already selected checkpoints."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo
PROJECT=Path('/media/rwj/ZX1 2TB/RLPR')
SOURCE=PROJECT/'src_spectral'

def child(checkpoint, variant, output):
    import faiss
    import numpy as np
    import torch
    from tqdm import tqdm
    sys.path.insert(0,str(SOURCE))
    from scripts.registry import TEST,build_test_params,get_model_variant
    from trainer.eval import CrossModalTester,recall_at_k
    from utils.seed import set_global_seed
    set_global_seed(0);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    cfg=TEST['mulran'];params=build_test_params()
    tester=CrossModalTester(params,get_model_variant(cfg['model'],variant),cfg['dataloader'])
    tester._load_checkpoint();tester.model.eval()
    lidar=[];radar=[]
    with torch.no_grad():
        for batch in tqdm(tester.test_loader):
            ld,rd,*_=tester.model(batch['lidar_bev'].to(tester.device),batch['radar_bev'].to(tester.device))
            lidar.append(ld.cpu().numpy());radar.append(rd.cpu().numpy())
    lidar=np.concatenate(lidar).astype('float32');radar=np.concatenate(radar).astype('float32')
    n=tester.test_dataset.num_db;gt=tester.test_dataset.get_positives();metrics={}
    if len(lidar)!=n+tester.test_dataset.num_query:raise ValueError('Incomplete feature extraction')
    for name,queries,database in [('R2L',radar[n:],lidar[:n]),('R2R',radar[n:],radar[:n]),('L2L',lidar[n:],lidar[:n])]:
        index=faiss.IndexFlatL2(params.output_dim);index.add(database)
        _,prediction=index.search(queries,min(len(database),20))
        scores=recall_at_k(prediction,gt,[1,5,10,20])
        metrics[name]={f'Recall@{k}':float(v) for k,v in zip([1,5,10,20],scores)}
    result=dict(checkpoint=str(checkpoint),variant=variant,split=os.environ['RLPR_MULRAN_TEST_SPLIT'],metrics=metrics,
                note='R2R diagnoses the fixed radar teacher; L2L uses the LiDAR branch after cross-modal alignment, not its unimodal initialization.')
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

def main(experiment):
    report=json.loads((experiment/'run.json').read_text())
    if report['status']!='COMPLETE':raise ValueError('Finish the declared experiment and lock selection before diagnostics')
    epoch=report['selection']['selected_epoch'];selected=report['epochs'][str(epoch)]
    checkpoint=Path(selected['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=selected['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H%M%S')
    root=PROJECT/'runs'/('mulran_teacher_diagnostic_'+stamp);root.mkdir()
    driver=root/'diagnose_teacher.py';driver.write_text(Path(__file__).read_text())
    protocol=dict(selected_experiment=str(experiment),selected_epoch=epoch,selection_rule=report['selection']['rule'],
                  diagnostics='Fixed selected Word model and released official model, R2L/R2R/L2L; no training or checkpoint reselection',
                  strict=True,cublas_workspace=':4096:8',batch_size=1)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    results=[]
    for name,variant,path,fusion in [('official','rlpr',PROJECT/'pretrained/mulran.pth.tar','0'),('word_selected','spectral_word',checkpoint,'1')]:
        for split in ['dcc','riverside']:
            env=os.environ.copy();env.update(RLPR_MODEL_VARIANT=variant,RLPR_SPECTRAL_CROSS_FUSION=fusion,
                RLPR_DETERMINISTIC='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',RLPR_SAMPLING_SEED='0',RLPR_CROSS_HARD_NEGATIVES='0',
                RLPR_MULRAN_DATA=str(PROJECT/'processed/mulran_official'),RLPR_MULRAN_TEST_SPLIT=split,
                RLPR_CHECKPOINT_PATH=str(path),RLPR_TEST_ZERO_SHOT='0',RLPR_TRAINING_ROOT=str(root/name/split),
                TORCH_HOME=str(PROJECT/'cache/torch'),MPLCONFIGDIR=str(root/'matplotlib_cache'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            output=root/f'{name}_{split}.json'
            with (root/f'{name}_{split}.log').open('x') as log:
                subprocess.run([sys.executable,'-u',str(driver),'--child','--checkpoint',str(path),'--variant',variant,'--output',str(output)],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads(output.read_text());result['name']=name
            if name=='word_selected':
                expected=selected['benchmarks'][split]['metrics']['Recall@1']
                if round(result['metrics']['R2L']['Recall@1'],2)!=expected:raise ValueError('Diagnostic does not reproduce locked R2L test')
            results.append(result);print(name,split,result['metrics'],flush=True)
    (root/'results.json').write_text(json.dumps(dict(status='COMPLETE',protocol=protocol,results=results),indent=2)+'\n')
    with (root/'comparison.csv').open('w') as f:
        fields=['model','split','direction','Recall@1','Recall@5','Recall@10','Recall@20'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in results:
            for direction,m in r['metrics'].items():writer.writerow(dict(model=r['name'],split=r['split'],direction=direction,**m))
    Path('/tmp/rlpr_teacher_diagnostic_path').write_text(str(root))
    print('COMPLETE',root,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path);p.add_argument('--child',action='store_true');p.add_argument('--checkpoint',type=Path);p.add_argument('--variant');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.child:child(a.checkpoint,a.variant,a.output)
    else:main(a.experiment)
