"""Matched-epoch analysis fixed before the hard-negative model selection."""
import csv,hashlib,json,math,os,re,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);report=json.loads((root/'run.json').read_text())
if report['status']!='COMPLETE':raise ValueError('Finish locked experiment first')
protocol=json.loads((root/'protocol.json').read_text());baseline_root=Path(protocol['baseline_run'])
if hashlib.sha256((baseline_root/'run.json').read_bytes()).hexdigest()!=protocol['baseline_report_sha256']:raise ValueError('Baseline report changed')
baseline=json.loads((baseline_root/'run.json').read_text());epoch=report['selection']['selected_epoch'];key=str(epoch)
base=baseline['epochs'][key];hard=report['epochs'][key];results={};project=root.parent.parent;source=project/'src_spectral'
if hashlib.sha256(Path(base['checkpoint']).read_bytes()).hexdigest()!=base['checkpoint_sha256']:raise ValueError('Baseline checkpoint changed')
for split in ['dcc','riverside']:
    if split in base.get('benchmarks',{}):
        entry=base['benchmarks'][split];origin='Existing declared baseline evaluation'
    else:
        env=os.environ.copy();env.update(RLPR_MODEL_VARIANT='spectral_word',RLPR_SPECTRAL_CROSS_FUSION='1',RLPR_DETERMINISTIC='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',RLPR_SAMPLING_SEED='0',RLPR_CROSS_HARD_NEGATIVES='0',RLPR_MULRAN_DATA=str(project/'processed/mulran_official'),RLPR_MULRAN_TEST_SPLIT=split,RLPR_CHECKPOINT_PATH=base['checkpoint'],RLPR_TEST_ZERO_SHOT='0',RLPR_TRAINING_ROOT=str(root/f'matched_baseline_epoch_{epoch}'),TORCH_HOME=str(project/'cache/torch'),MPLCONFIGDIR=str(root/'matplotlib_cache'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        logpath=root/f'matched_baseline_epoch_{epoch}_{split}.log'
        with logpath.open('x') as log:subprocess.run([sys.executable,'-u','test_cross_modal.py','--dataset','mulran','--model','spectral_word','--cross-fusion'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        output=logpath.read_text();metrics={}
        for name in ['Recall@1','Recall@5','Recall@10','Recall@20','Recall@1percent','F1']:
            label='f1 score' if name=='F1' else re.escape(name);match=re.search(label+r':\s*([0-9.eE+-]+)',output)
            if not match or not math.isfinite(float(match.group(1))):raise ValueError('Missing finite '+name)
            metrics[name]=float(match.group(1))
        entry={'metrics':metrics,'log':str(logpath)};origin='Additional evaluation chosen only by hard-negative validation-selected epoch'
    results[split]={'epoch':epoch,'random_negatives':entry,'hard_negatives':hard['benchmarks'][split],'baseline_metric_origin':origin}
    print(split,results[split],flush=True)
output={'status':'COMPLETE','selected_epoch':epoch,'protocol':str(root/'matched_epoch_protocol.json'),'baseline_report_changed':False,'results':results}
(root/'matched_epoch_results.json').write_text(json.dumps(output,indent=2)+'\n')
with (root/'matched_epoch_comparison.csv').open('w') as f:
    fields=['method','epoch','split','Recall@1','Recall@5','Recall@10','Recall@20','Recall@1percent','F1'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for split,result in results.items():
        for method in ['random_negatives','hard_negatives']:writer.writerow(dict(method=method,epoch=epoch,split=split,**result[method]['metrics']))
print('COMPLETE matched epoch',epoch,flush=True)
