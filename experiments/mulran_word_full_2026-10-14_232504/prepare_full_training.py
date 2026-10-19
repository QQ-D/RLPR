import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path('/media/rwj/ZX1 2TB/RLPR')
RUN = Path('/tmp/rlpr_word_full_run_path').read_text()
RUN = Path(RUN)
sys.path.insert(0, str(PROJECT / 'src'))
from dataset.MulRan.gen_info import gen_info
from dataset.MulRan.gen_index import process_infos_sejong, process_infos_tests
from dataset.MulRan.gen_bev import gen_bev

DATA = PROJECT / 'processed/mulran_official'
RAW = PROJECT.parent / 'RLPR_dataset/MulRan'
report = {'status': 'RUNNING', 'phase': 'PREPARING_SEJONG', 'groups': {},
          'processed_root': str(DATA), 'raw_root': str(RAW),
          'training_scope': 'Full Sejong training regions and KAIST01/02; no Riverside/DCC training',
          'preprocessing': 'Original author synchronization, indices, 80m range, 1.6m LiDAR voxels, 50x225 BEV'}
def save():
    (RUN / 'preparation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
save()

def generate_missing(name, infos, rows):
    roots = [PROJECT/'processed/mulran_sejong/bev/mm_bev',
             PROJECT/'runs/sejong_spectral_2026-10-14_202056/data/bev',
             PROJECT/'runs/sejong_short_train/data/bev']
    missing = []; reused = 0
    for row in rows:
        frame = infos[int(row[0])]
        for sensor, key, channel in [('lidar','lidar_infos','LIDAR_TOP'), ('radar','radar_infos','RADAR_TOP')]:
            filename = Path(frame[key][channel]['filename']).stem + '.png'
            dest = DATA/'bev/mm_bev'/sensor/filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                for root in roots:
                    source = root/sensor/filename
                    if source.is_file():
                        image = cv2.imread(str(source), 0)
                        if image is None or image.shape != (50,225):
                            raise ValueError(f'Invalid cached BEV: {source}')
                        shutil.copy2(source, dest)
                        reused += 1
                        break
        names = [Path(frame[key][channel]['filename']).stem+'.png'
                 for key,channel in [('lidar_infos','LIDAR_TOP'),('radar_infos','RADAR_TOP')]]
        if any(not (DATA/'bev/mm_bev'/sensor/f).is_file() for sensor,f in zip(('lidar','radar'),names)):
            missing.append(row)
    report['groups'][name].update(reused_bev_files=reused, frames_requiring_generation=len(missing))
    save()
    if missing:
        print(name, 'generating',len(missing),'missing BEV pairs',flush=True)
        gen_bev(str(DATA/'bev/mm_bev'), str(DATA/'info'/('mulran_infos-'+name+'.pkl')),np.asarray(missing))
    for row in rows:
        frame = infos[int(row[0])]
        for sensor,key,channel in [('lidar','lidar_infos','LIDAR_TOP'),('radar','radar_infos','RADAR_TOP')]:
            p=DATA/'bev/mm_bev'/sensor/(Path(frame[key][channel]['filename']).stem+'.png')
            if not p.is_file() or p.stat().st_size==0:
                raise IOError(f'Missing BEV: {p}')

try:
    for sub in ('info','index','bev/mm_bev/lidar','bev/mm_bev/radar'):
        (DATA/sub).mkdir(parents=True,exist_ok=True)
    info=DATA/'info/mulran_infos-sejong.pkl'
    if not info.exists():
        shutil.copy2(PROJECT/'processed/mulran_sejong/info/mulran_infos-sejong.pkl',info)
    with info.open('rb') as f: infos=pickle.load(f)
    whole, db, query, test_db, test_query=process_infos_sejong(str(info),'Sejong02',9.)
    combined=np.concatenate((db,query),axis=0)
    for key,array in [('whole',whole),('db',combined),('train_query',combined),('test_db',test_db),('test_query',test_query)]:
        np.save(DATA/'index'/('sejong_'+key+'.npy'),array)
    report['groups']['sejong']={'synchronized_frames':len(infos),'database_frames':len(combined),'query_frames':len(combined),
                                'train_seq01_frames':len(db),'train_seq02_frames':len(query)}
    generate_missing('sejong',infos,combined)
    report['groups']['sejong']['status']='COMPLETE'
    report['phase']='WAITING_FOR_KAIST_EXTRACTION';save()
    print('Sejong full training data ready:', len(combined),flush=True)
    while True:
        try:
            extraction=json.loads((RUN/'extraction.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            time.sleep(2);continue
        if extraction['status']=='FAILED_OR_INTERRUPTED':
            raise RuntimeError('KAIST extraction failed')
        if extraction['status']=='COMPLETE':break
        time.sleep(2)
    report['phase']='PREPARING_KAIST';save()
    info=DATA/'info/mulran_infos-kaist.pkl'
    if not info.exists():
        infos=gen_info(str(RAW),['KAIST01','KAIST02'],1.)
        with info.open('xb') as f:pickle.dump(infos,f)
    else:
        with info.open('rb') as f:infos=pickle.load(f)
    whole,db,query=process_infos_tests(str(info),'KAIST02',1.,9.)
    for key,array in [('whole',whole),('db',db),('train_query',query)]:
        np.save(DATA/'index'/('kaist_'+key+'.npy'),array)
    report['groups']['kaist']={'synchronized_frames':len(infos),'database_frames':len(db),'query_frames':len(query)}
    generate_missing('kaist',infos,np.concatenate((db,query),axis=0))
    report['groups']['kaist']['status']='COMPLETE'
    report['status']='COMPLETE';report['phase']='READY'
    report['total_database_frames']=sum(g['database_frames'] for g in report['groups'].values())
    report['total_query_frames']=sum(g['query_frames'] for g in report['groups'].values())
except BaseException:
    report['status']='FAILED_OR_INTERRUPTED';save();raise
save()
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
