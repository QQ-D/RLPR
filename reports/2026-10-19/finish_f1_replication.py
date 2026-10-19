"""After training completes, reproduce official diagnostics and build audit/PDF."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE=Path('/media/rwj/ZX1 2TB/RLPR/src_spectral')
sys.path.insert(0,str(SOURCE))
from scripts.f1_priority_replication import environment
from scripts.amplitude_ablation import digest, now, write_json

HERE=Path(__file__).resolve().parent
ROOT=Path(Path('/tmp/rlpr_f1_replication_run_path').read_text())
STATUS=ROOT/'postprocess_status.json'

def status(stage,**extra):write_json(STATUS,dict(stage=stage,time=now(),**extra))

try:
    status('WAITING_FOR_TRAINING')
    while True:
        report=json.loads((ROOT/'run.json').read_text())
        if report['status']=='COMPLETE':break
        if report['status']!='RUNNING':raise RuntimeError('Training stopped: '+report['status'])
        time.sleep(30)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    if digest(ROOT/'protocol.json')!=report['protocol_sha256']:raise ValueError('Protocol changed')
    official=SOURCE.parent/'pretrained/mulran.pth.tar'
    if digest(official)!='51868acc02079672810a9a517c130980558705d74963e98f1023d0ca0e27e8dc':
        raise ValueError('Official weights changed')
    folder=ROOT/'official_diagnostics';folder.mkdir(exist_ok=True)
    manifest={}
    for split in ('sejong','dcc','riverside'):
        status('OFFICIAL_'+split.upper())
        env=environment(ROOT,0,'signed',protocol['initial'],'evaluate',split)
        env.update(RLPR_MODEL_VARIANT='rlpr',RLPR_SPECTRAL_CROSS_FUSION='0')
        argv=[sys.executable,'-u',str(ROOT/'driver.py'),'--child','evaluate','--folder',str(folder),
              '--checkpoint',str(official),'--variant','rlpr','--split',split]
        if split!='sejong':
            diag=json.loads((folder/'sejong_diagnostics.json').read_text())
            argv+=['--threshold',str(diag['best_threshold']['threshold'])]
        with (folder/(split+'.log')).open('x') as handle:
            subprocess.run(argv,cwd=SOURCE,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
        manifest[split]={suffix:digest(folder/(split+suffix)) for suffix in ('.json','_diagnostics.json','.csv','.npz')}
    write_json(folder/'manifest.json',dict(checkpoint_sha256=digest(official),results=manifest,finished_at=now()))
    status('AUDITING')
    with (ROOT/'completion_audit.log').open('x') as handle:
        subprocess.run([sys.executable,str(HERE/'audit_f1_replication.py')],cwd=SOURCE,stdout=handle,stderr=subprocess.STDOUT,check=True)
    status('BUILDING_PDF')
    subprocess.run(['/usr/bin/python3',str(HERE/'build_f1_replication_report.py')],cwd=SOURCE,check=True)
    status('COMPLETE',report=str(HERE/'RLPR_F1优先多种子复验_2026-10-19.pdf'))
except BaseException as error:
    status('FAILED',error=repr(error))
    raise
