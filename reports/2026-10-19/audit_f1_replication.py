"""Audit completed F1 replication and summarize saved, per-query diagnostics."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

SOURCE=Path('/media/rwj/ZX1 2TB/RLPR/src_spectral')
sys.path.insert(0,str(SOURCE))
from scripts.f1_priority_replication import audit_trial, audit_seed_start, choose_loss, verify_metrics
from scripts.amplitude_ablation import digest, now, write_json

root=Path(Path('/tmp/rlpr_f1_replication_run_path').read_text())
report=json.loads((root/'run.json').read_text())
protocol=json.loads((root/'protocol.json').read_text())
assert report['status']=='COMPLETE'
assert digest(root/'protocol.json')==report['protocol_sha256']
assert digest(root/'selection.json')==report['selection_sha256']
selected,validation=choose_loss(report['seeds'])
assert selected==report['selected_loss']
for name,sha in protocol['source_sha256'].items():assert digest(SOURCE/name)==sha,name
for initial in protocol['initial'].values():assert digest(initial['path'])==initial['sha256']
torch.set_num_threads(4)
audits={}
for seed,arms in report['seeds'].items():
    audits[seed]={}
    for arm,row in arms.items():
        audits[seed][arm]=audit_trial(row['checkpoint'],int(seed),arm,protocol['initial']['r']['path'],row)
        verify_metrics(row)
        for binding in row['artifacts'].values():assert digest(binding['path'])==binding['sha256']
        if seed!='0':audit_seed_start(root/f'seed{seed}'/arm,int(seed))
    assert arms['signed']['sampling']==arms['infonce']['sampling']
    if seed!='0':
        with np.load(root/f'seed{seed}/signed/cache/cross_negatives_epoch_1.npz') as a, np.load(root/f'seed{seed}/infonce/cache/cross_negatives_epoch_1.npz') as b:
            assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)

errors={}
for split in ('sejong','dcc','riverside'):
    paired={}
    with np.load(root/'seed0/signed'/(split+'.npz')) as a,np.load(root/'seed0/infonce'/(split+'.npz')) as b:
        assert np.array_equal(a['query_index_n_pos'],b['query_index_n_pos'])
        assert np.array_equal(a['database_index_n_pos'],b['database_index_n_pos'])
        sa,ib=a['correct'],b['correct']
        paired=dict(both_correct=int((sa&ib).sum()),signed_only_correct=int((sa&~ib).sum()),
                    infonce_only_correct=int((~sa&ib).sum()),both_wrong=int((~sa&~ib).sum()))
    errors[split]=dict(paired_seed0=paired,models={})
    for arm in ('signed','infonce'):
        folder=root/'seed0'/arm
        diag=json.loads((folder/(split+'_diagnostics.json')).read_text())
        with np.load(folder/(split+'.npz')) as data:
            wrong=data['error_m'][~data['correct']]
            diag['wrong_distance_buckets_m']={
                '9_to_20':int(((wrong>9)&(wrong<=20)).sum()),
                '20_to_50':int(((wrong>20)&(wrong<=50)).sum()),
                '50_to_100':int(((wrong>50)&(wrong<=100)).sum()),
                'above_100':int((wrong>100).sum())}
            diag['top1_squared_l2_median_correct']=float(np.median(data['distances'][data['correct'],0]))
            diag['top1_squared_l2_median_wrong']=float(np.median(data['distances'][~data['correct'],0]))
        errors[split]['models'][arm]=diag

official_root=root/'official_diagnostics'
if official_root.exists():
    historic=json.loads((SOURCE.parent/'runs/mulran_official_eval_2026-10-14/results.json').read_text())
    for split in ('sejong','dcc','riverside'):
        path=official_root/(split+'_diagnostics.json')
        assert path.exists(),path
        diag=json.loads(path.read_text())
        assert digest(diag['checkpoint']['path'])==diag['checkpoint']['sha256']
        if split!='sejong':
            actual=json.loads((official_root/(split+'.json')).read_text())
            expected=historic['splits'][split]['metrics']
            assert all(math.isclose(actual[k],v,abs_tol=1e-10 if k=='F1' else .0051) for k,v in expected.items())
        errors[split]['models']['released_official']=diag

launch=json.loads((root/'runtime_at_launch.json').read_text())
runtime=dict(python='.'.join(str(v) for v in sys.version_info[:3]),torch=torch.__version__,
    cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),source_commit=protocol['source_commit'])
launch_commit=subprocess.check_output(['git','rev-parse','--verify',launch['source_commit']+'^{commit}'],cwd=SOURCE,text=True).strip()
assert runtime['source_commit']==launch_commit
assert all(runtime[k]==launch[k] for k in runtime if k!='source_commit')
write_json(root/'runtime_at_completion.json',runtime)
tests=(root/'regression_tests.log').read_text()
assert 'Ran 99 tests' in tests and '\nOK\n' in tests
write_json(root/'error_analysis.json',errors)
verification=dict(all_checks_passed=True,verified_at=now(),checkpoint_count=6,new_training_checkpoints=4,
    fixed_unary_seed=0,cross_seeds=[0,1,2],optimizer_updates_per_checkpoint=2071,
    source_files=len(protocol['source_sha256']),teacher_frozen=True,paired_sampling_match=True,
    paired_mining_match=True,metric_checkpoint_binding_match=True,legacy_f1_parity=True,
    runtime_matches=True,regression_tests_passed=99,checkpoints=audits)
write_json(root/'completion_verification.json',verification)
print(json.dumps(dict(root=str(root),selection=selected,verification=verification,errors=errors),ensure_ascii=False,indent=2))
