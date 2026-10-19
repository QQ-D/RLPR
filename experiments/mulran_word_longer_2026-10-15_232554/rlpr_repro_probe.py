import hashlib, json, os, sys
from pathlib import Path
import torch
sys.path.insert(0,'/media/rwj/ZX1 2TB/RLPR/src_spectral')
from scripts.registry import CROSS_MODAL, build_params, get_model_variant
from trainer.cross_modal import CrossModalTrainer
from utils.seed import set_global_seed
set_global_seed(0);torch.set_num_threads(4)
mode=sys.argv[1]
if mode=='strict': torch.use_deterministic_algorithms(True)
cfg=CROSS_MODAL['mulran'];t=CrossModalTrainer(build_params(1,epochs_override=1),get_model_variant(cfg['model'],'spectral_word'),cfg['dataloader'])
t._init_weights();t.model.train(); out=[]
try:
 for i,b in enumerate(t.train_loader):
  if i==3: break
  t.optimizer.zero_grad();l=b['lidar_bev'].cuda();r=b['radar_bev'].cuda()
  loss,_=t._compute_loss(t.model(l,r));loss.backward();t.optimizer.step()
  out.append(dict(step=i,loss=loss.item(),batch=hashlib.sha256(b['lidar_bev'].numpy().tobytes()+b['radar_bev'].numpy().tobytes()).hexdigest()))
 torch.save(t.model.state_dict(),sys.argv[2]+'.pth')
 Path(sys.argv[2]+'.json').write_text(json.dumps(out,indent=2))
 print(out,flush=True)
finally:t.writer.close()
