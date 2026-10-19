"""Single-positive InfoNCE temperature only; original trainer is unchanged."""
import math
import os
import torch
from modules.loss import InfoNCE
from trainer.cross_modal import CrossModalTrainer


class TemperatureCrossModalTrainer(CrossModalTrainer):
    def __init__(self,params,model_class,make_dataloader_fn,temperature):
        if params.word_alignment_loss!='infonce':raise ValueError('Temperature study requires single-positive InfoNCE')
        self.configure_temperature(temperature)
        super().__init__(params,model_class,make_dataloader_fn)
        self.configure_temperature(temperature)

    def configure_temperature(self,temperature):
        if not math.isfinite(temperature) or temperature<=0:raise ValueError('Invalid InfoNCE temperature')
        self.temperature=float(temperature)
        self.info_nce=InfoNCE(self.temperature)

    def _training_settings(self):
        return dict(super()._training_settings(),contrastive_temperature=self.temperature)

    def _init_weights(self):
        if self.resume:
            path=os.path.join(self.weights_root,f'{self.resume_epoch}.pth.tar')
            saved=torch.load(path,map_location='cpu')
            if saved.get('training_settings',{}).get('contrastive_temperature')!=self.temperature:
                raise ValueError('Resume requires an explicitly matching recorded temperature')
        return super()._init_weights()
