"""Copy code, reports and experiment logs to an existing personal-repo checkout.

Never copies dataset/weights/caches, never changes training source, never pushes.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PROJECT=Path('/media/rwj/ZX1 2TB/RLPR')
BLOCK_PARTS={'.git','__pycache__','.venv','weights','checkpoints','cache','matplotlib_cache','processed','pretrained','wheels'}
BLOCK_EXTS={'.pth','.pt','.ckpt','.safetensors','.onnx','.bin','.pcd','.bag','.mcap','.npy','.npz','.pkl','.pickle','.h5','.hdf5','.whl','.zip','.7z','.tar','.gz'}
RESULT_EXTS={'.json','.csv','.log','.txt','.md','.py','.sh','.pdf','.patch'}


def allowed(path,relative):
    return not path.is_symlink() and not any(p in BLOCK_PARTS for p in relative.parts) and path.suffix.lower() not in BLOCK_EXTS and path.name not in ('.env','id_ed25519','id_rsa')


def copy_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def export(destination):
    destination=destination.resolve()
    if not (destination/'.git').is_dir():raise ValueError('Destination must be an existing independent Git clone')
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=destination,text=True).strip()
    if remote not in ('https://github.com/QQ-D/RLPR.git','git@github.com:QQ-D/RLPR.git'):raise ValueError('Unexpected export target')
    versions={};groups=Counter()
    for directory in ('src','src_spectral'):
        root=PROJECT/directory
        tracked=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
        versions[directory]=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        for name in filter(None,tracked):
            rel=Path(name);path=root/rel
            if not allowed(path,rel):raise ValueError('Forbidden source item: '+name)
            copy_file(path,destination/directory/rel);groups[directory]+=1
    for directory in ('reports','backups'):
        for path in (PROJECT/directory).rglob('*'):
            if not path.is_file():continue
            rel=path.relative_to(PROJECT)
            if allowed(path,rel):copy_file(path,destination/rel);groups[directory]+=1
    for path in PROJECT.glob('*.sh'):copy_file(path,destination/path.name);groups['root']+=1
    copy_file(PROJECT/'environment.lock.txt',destination/'environment.lock.txt')
    for path in PROJECT.glob('*.docx'):copy_file(path,destination/'design'/path.name);groups['design']+=1
    for path in (PROJECT/'runs').rglob('*'):
        if not path.is_file():continue
        rel=path.relative_to(PROJECT/'runs')
        if not allowed(path,rel):continue
        if path.suffix.lower() not in RESULT_EXTS and not path.name.startswith('events.out.tfevents.'):continue
        copy_file(path,destination/'experiments'/rel);groups['experiments']+=1
    history=destination/'source_history';history.mkdir(exist_ok=True)
    for directory in ('src','src_spectral'):
        extra=PROJECT/directory/'README.full.md'
        if extra.is_file():copy_file(extra,history/(directory+'_README.full.md'))
    changes=subprocess.check_output(['git','diff','--binary',versions['src'],versions['src_spectral']],cwd=PROJECT/'src_spectral')
    (history/'spectral_changes.patch').write_bytes(changes)
    log=subprocess.check_output(['git','log','--format=fuller',versions['src']+'..'+versions['src_spectral']],cwd=PROJECT/'src_spectral')
    (history/'spectral_commits.log').write_bytes(log)
    (history/'source_versions.json').write_text(json.dumps(versions,indent=2)+'\n')
    folders=sorted(p for p in (PROJECT/'runs').iterdir() if p.is_dir() and (destination/'experiments'/p.name).exists())
    index=['# 实验记录索引','','本目录保留配置、运行日志、结果、核验记录及 TensorBoard 标量日志；模型、数据输入、BEV 图和 NumPy 缓存留在本机。历史失败或中止记录也保留，不能将其视为完成实验。','','| 实验目录 | 记录状态 | 结果入口 |','|---|---|---|']
    for folder in folders:
        target=destination/'experiments'/folder.name;state='旧记录，见目录文件'
        for name in ('run.json','results.json'):
            p=target/name
            if p.exists():
                data=json.loads(p.read_text());state=str(data.get('status',state));break
        links=['['+name+']('+folder.name+'/'+name+')' for name in ('run.json','results_summary.json','comparison.csv','completion_verification.json','report.pdf') if (target/name).exists()]
        index.append('| ['+folder.name+']('+folder.name+'/) | '+state+' | '+' · '.join(links)+' |')
    (destination/'experiments/INDEX.md').write_text('\n'.join(index)+'\n')
    files={};total=0
    for path in sorted(destination.rglob('*')):
        if not path.is_file() or '.git' in path.relative_to(destination).parts:continue
        rel=path.relative_to(destination)
        if not allowed(path,rel):raise ValueError('Forbidden publication file: '+str(rel))
        if path.name in ('SHA256SUMS.txt','UPLOAD_MANIFEST.json') and len(rel.parts)==1:continue
        content=path.read_bytes();files[str(rel)]=dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest());total+=len(content)
    manifest=dict(created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),target='https://github.com/QQ-D/RLPR',
        source_versions=versions,groups=dict(groups),file_count=len(files),total_bytes=total,
        excluded=['raw and processed datasets','point clouds and BEV dataset images','all model/checkpoint files','descriptor/mining NumPy caches','Conda environments and wheels','private Git metadata and credentials'],
        files=files)
    (destination/'UPLOAD_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    sums={name:row['sha256'] for name,row in files.items()}
    sums['UPLOAD_MANIFEST.json']=hashlib.sha256((destination/'UPLOAD_MANIFEST.json').read_bytes()).hexdigest()
    (destination/'SHA256SUMS.txt').write_text(''.join(sums[name]+'  '+name+'\n' for name in sorted(sums)))
    print(json.dumps(dict(target=str(destination),file_count=len(files),total_bytes=total,groups=dict(groups)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',required=True,type=Path)
    export(parser.parse_args().destination)
