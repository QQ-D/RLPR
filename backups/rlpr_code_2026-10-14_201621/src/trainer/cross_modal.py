import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from modules.loss import InfoNCE
from utils.pretrain import (
    freeze_params_named_like_r_checkpoint,
    freeze_radar_branch_cross,
    load_snail_crossmodal_init,
    load_unimodal_pretrained,
)


class CrossModalTrainer:
    def __init__(self, params, model_class, make_dataloader_fn, hercules_mode=False):
        self.params = params
        self.hercules_mode = hercules_mode

        self.weights_root = params.weights_root
        self.logs_root = params.logs_root
        self.resume = params.resume_checkpoint
        self.resume_epoch = params.resume_epoch

        self.writer = SummaryWriter(self.logs_root)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class.create(weights=ResNet18_Weights.DEFAULT)
        n_params = sum(p.nelement() for p in self.model.parameters())
        print(f"Number of model parameters: {n_params}")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), params.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.8)
        self.info_nce = InfoNCE()

        loaders = make_dataloader_fn("train", params)
        if len(loaders) == 5:
            self.train_loader, _, _, _, _ = loaders
        else:
            self.train_loader, _ = loaders

        self.total_iter = len(self.train_loader)

    def _freeze_radar(self):
        if self.hercules_mode:
            freeze_radar_branch_cross(self.model)
        else:
            freeze_params_named_like_r_checkpoint(self.model, self.params.pretrain_r_path)

    def _init_weights(self):
        if self.hercules_mode:
            load_snail_crossmodal_init(self.model, self.params.snail_crossmodal_path)
            if self.resume:
                resume_path = os.path.join(self.weights_root, f"{self.resume_epoch}.pth.tar")
                if os.path.exists(resume_path):
                    print(f"\nResuming Hercules from {resume_path}")
                    checkpoint = torch.load(resume_path)
                    self.model.load_state_dict(checkpoint["state_dict"])
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                    self.scheduler.load_state_dict(checkpoint["scheduler"])
                    self._freeze_radar()
                    return checkpoint["epoch"]
            print("\nStarting Hercules fine-tuning from Snail cross-modal checkpoint")
            self._freeze_radar()
            return 0

        if self.resume:
            resume_filename = os.path.join(self.weights_root, f"{self.resume_epoch}.pth.tar")
            print(f"\nResuming from {resume_filename}")
            checkpoint = torch.load(resume_filename)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self._freeze_radar()
            return checkpoint["epoch"]

        print("\nTraining From Scratch")
        load_unimodal_pretrained(self.model, self.params.pretrain_r_path, self.params.pretrain_l_path)
        return 0

    def do_train(self):
        starting_epoch = self._init_weights()
        for epoch in range(starting_epoch + 1, self.params.epochs + 1):
            print(f"\nTraining Process-epoch {epoch}-------->")
            query_num = np.zeros([1], dtype=np.float32)
            total_loss = np.zeros([1], dtype=np.float32)
            self.model.train()

            for batch_idx, batch in enumerate(tqdm(self.train_loader)):
                self.optimizer.zero_grad()
                l_bev = batch["lidar_bev"].to(self.device)
                r_bev = batch["radar_bev"].to(self.device)
                descriptors_l, descriptors_r, _, _ = self.model(l_bev, r_bev)
                descriptors_l = descriptors_l.view(self.params.batch_size, 12, self.params.output_dim)
                descriptors_r = descriptors_r.view(self.params.batch_size, 12, self.params.output_dim)

                loss = 0.0
                for descriptor_l, descriptor_r in zip(descriptors_l, descriptors_r):
                    loss += self.info_nce(descriptor_r, descriptor_l)
                loss /= self.params.batch_size
                loss.backward()

                if torch.isnan(loss):
                    raise ValueError("Loss of current batch is nan!")
                query_num[0] += 1
                total_loss[0] += loss.item()
                self.writer.add_scalar("loss 0", loss.item(), (epoch - 1) * self.total_iter + batch_idx)
                self.optimizer.step()
                torch.cuda.empty_cache()

            loss_cur_epoch = total_loss / query_num
            print(f"epoch {epoch} finishes\nloss {loss_cur_epoch}")
            self.writer.add_scalar("total loss 0", loss_cur_epoch[0], epoch)
            self.scheduler.step()

            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                os.path.join(self.weights_root, f"{epoch}.pth.tar"),
            )
