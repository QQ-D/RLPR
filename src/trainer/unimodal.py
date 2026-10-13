import os
import pickle

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from modules.loss import triplet_loss


class UnimodalTrainer:
    def __init__(self, params, model_class, make_dataloader_fn, modality, cache_name,
                 hard_mining_from_epoch=0):
        assert modality in ("l", "r")
        self.params = params
        self.modality = modality
        self.cache_name = cache_name
        self.hard_mining_from_epoch = hard_mining_from_epoch

        self.weights_root = params.weights_root
        self.logs_root = params.logs_root
        self.cache_root = params.cache_root
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

        loaders = make_dataloader_fn("train", params)
        if len(loaders) == 5:
            self.train_loader, _, self.train_query_loader, _, _ = loaders
        else:
            self.train_loader, self.train_query_loader = loaders

        self.total_iter = len(self.train_loader)

    def _get_bev(self, batch):
        key = "radar_bev" if self.modality == "r" else "lidar_bev"
        return batch[key].to(self.device)

    def _forward_descriptors(self, bev):
        outputs = self.model(bev, bev, bev, bev, bev)
        return outputs[3]

    def _init_weights(self):
        if self.resume:
            resume_filename = os.path.join(self.weights_root, f"{self.resume_epoch}.pth.tar")
            print(f"\nResuming from {resume_filename}")
            checkpoint = torch.load(resume_filename)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            return checkpoint["epoch"]
        print("\nTraining From Scratch")
        return 0

    def _update_hard_mining_cache(self):
        print(f"\nGenerating cache {self.cache_name} -------->")
        descriptor_database = []
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(self.train_query_loader):
                bev = self._get_bev(batch)
                descriptor = self._forward_descriptors(bev)
                for des_idx in range(descriptor.shape[0]):
                    descriptor_database.append(descriptor[des_idx, :].cpu().detach().numpy())
                torch.cuda.empty_cache()

        cache_path = os.path.join(self.cache_root, f"vec_cache_{self.cache_name}.pickle")
        with open(cache_path, "wb") as file:
            pickle.dump(descriptor_database, file)
        self.train_loader.dataset.update_latent_vectors(cache_path)

    def do_train(self):
        starting_epoch = self._init_weights()
        for epoch in range(starting_epoch + 1, self.params.epochs + 1):
            print(f"\nTraining Process-epoch {epoch}-------->")

            if epoch >= self.hard_mining_from_epoch:
                self._update_hard_mining_cache()

            query_num = np.zeros([1], dtype=np.float32)
            total_loss = np.zeros([1], dtype=np.float32)
            self.model.train()

            for batch_idx, batch in enumerate(tqdm(self.train_loader)):
                self.optimizer.zero_grad()
                bev = self._get_bev(batch)
                descriptors = self._forward_descriptors(bev)
                descriptors = descriptors.view(self.params.batch_size, 12, self.params.output_dim)

                loss = 0.0
                for descriptor in descriptors:
                    o1, o2, o3 = torch.split(descriptor, [1, self.params.pos_num, self.params.neg_num], dim=0)
                    loss += triplet_loss(o1, o2, o3, self.params.loss_margin, lazy=True)
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
