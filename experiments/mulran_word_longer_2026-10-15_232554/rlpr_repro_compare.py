import json, os, subprocess, sys
from pathlib import Path
import torch
torch.set_num_threads(4)
p=json.loads(Path('/media/rwj/ZX1 2TB/RLPR/runs/mulran_word_longer_2026-10-15_232554/protocol.json').read_text())
e=os.environ.copy();e.update(RLPR_MODEL_VARIANT='spectral_word',RLPR_SAMPLING_SEED='0',RLPR_SPECTRAL_CROSS_FUSION='1',RLPR_RESUME='0',RLPR_WORD_ALIGNMENT_LOSS='signed_coherency_v2',RLPR_PRETRAIN_L=p['pretrain_l'],RLPR_PRETRAIN_R=p['pretrain_r'],RLPR_MULRAN_DATA=p['train_data'],RLPR_TRAINING_ROOT='/tmp/rlpr_probe',MPLCONFIGDIR='/tmp/rlpr-matplotlib',TORCH_HOME='/media/rwj/ZX1 2TB/RLPR/cache/torch',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
for mode in ['normal','strict']:
 env=e.copy()
 if mode=='strict':env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 for i in range(2):
  r=subprocess.run([sys.executable,'-u','/tmp/rlpr_repro_probe.py',mode,f'/tmp/rlpr_{mode}_{i}'],env=env)
  if r.returncode:sys.exit(r.returncode)
 a,b=[torch.load(f'/tmp/rlpr_{mode}_{i}.pth',map_location='cpu') for i in range(2)]
 print('MODE',mode,'maxdiff',max(float((v-b[k]).abs().max()) for k,v in a.items()),flush=True)
