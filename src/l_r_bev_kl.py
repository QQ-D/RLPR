#!/usr/bin/env python3

import os
import pickle
import sys
import random
import numpy as np
from tqdm import tqdm
import faiss
import torch
from scipy.stats import entropy
from sklearn.feature_extraction.image import extract_patches_2d
from torchvision.models import ResNet18_Weights
from config.params_full import ModelParamsFull
from modules.LR_cross_hffc_225_dis import LRCross
from dataset.NuScenes.nuscenes_dataloader import make_dataloader
# from dataset.MulRan.mulran_dataloader import make_dataloader
# from dataset.Boreas.boreas_dataloader import make_dataloader
# from dataset.SnailRadar.snail_dataloader import make_dataloader
from torch.utils.data import Dataset
import cv2

sys.maxsize = 3000
np.set_printoptions(threshold=sys.maxsize)


class Tester:
    def __init__(self, params: ModelParamsFull):
        super(Tester, self).__init__()
        self.params = params
        self.resume = self.params.resume_checkpoint

        self.weights_root = self.params.weights_root
        self.resume_epoch = self.params.resume_epoch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LRCross.create(weights=ResNet18_Weights.DEFAULT)
        n_params = sum([parameter.nelement() for parameter in self.model.parameters()])
        print(f'Number of model parameters: %d' % n_params)
        self.model.to(self.device)
        self.train_query_dataloader, _ = make_dataloader('test', params)

    def do_test(self):
        print("\nEvaluating Process-" + "-------->")

        resume_filename = os.path.join(self.weights_root, str(self.resume_epoch) + ".pth.tar")
        print("\nResuming from %s" % resume_filename)
        checkpoint = torch.load(resume_filename)
        self.model.load_state_dict(checkpoint['state_dict'])

        # fname2 = "/media/octane17/T7ShieldNus/training_unimpr_cross/r_nusc_hfc2_noshv_225/weights/5.pth.tar"
        # checkpoint = torch.load(fname2)
        # self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        # fname4 = "/media/octane17/T7ShieldNus/training_unimpr_cross/l_nusc_hfc2_noshv_225/weights/5.pth.tar"
        # checkpoint2 = torch.load(fname4)
        # self.model.load_state_dict(checkpoint2['state_dict'], strict=False)

        # fname2 = "/media/octane17/T7ShieldNus/training_unimpr_cross/r_mulran_hfc2_225/weights/5.pth.tar"
        # checkpoint = torch.load(fname2)
        # self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        # fname4 = "/media/octane17/T7ShieldNus/training_unimpr_cross/l_mulran_hfc2_225/weights/5.pth.tar"
        # checkpoint2 = torch.load(fname4)
        # self.model.load_state_dict(checkpoint2['state_dict'], strict=False)

        # fname2 = "/media/octane17/T7ShieldNus/training_unimpr_cross/l_snail_hfc2_225/weights/5.pth.tar"
        # checkpoint = torch.load(fname2)
        # self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        # fname4 = "/media/octane17/T7ShieldNus/training_unimpr_cross/r_snail_hfc2_225/weights/5.pth.tar"
        # checkpoint2 = torch.load(fname4)
        # self.model.load_state_dict(checkpoint2['state_dict'], strict=False)

        # kl_lr_total = 0.
        # kl_rl_total = 0.
        # with torch.no_grad():
        #     self.model.eval()
        #     for batch_idx, current_batch in enumerate(tqdm(self.train_query_dataloader)):
        #         l_bev = current_batch['lidar_bev'].numpy()[0, 0, 0, :, :]
        #         r_bev = current_batch['radar_bev'].numpy()[0, 0, 0, :, :]
        #         kl_lr, kl_rl = calculate_bev_channel_kl(l_bev, r_bev)
        #         kl_lr_total += kl_lr
        #         kl_rl_total += kl_rl
        # print("KL_LR: ", kl_lr_total / len(self.train_query_dataloader))
        # print("KL_RL: ", kl_rl_total / len(self.train_query_dataloader))

        lr_total = 0.
        rl_total = 0.
        l_total = 0.
        r_total = 0.
        mutual = 0.
        with torch.no_grad():
            self.model.eval()
            for batch_idx, current_batch in enumerate(tqdm(self.train_query_dataloader)):
                l_bev_cuda = current_batch['lidar_bev'].to(self.device)
                r_bev_cuda = current_batch['radar_bev'].to(self.device)
                descriptors_l, descriptors_r, feat_l, feat_r = self.model(l_bev_cuda, r_bev_cuda)
                feat_l = feat_l.detach().cpu().numpy()
                feat_r = feat_r.detach().cpu().numpy()
                l_given_r, r_en = calc_conditional_entropy(feat_l, feat_r, bins=10)
                r_given_l, l_en = calc_conditional_entropy(feat_r, feat_l, bins=10)
                mutual_inf = calc_mutual_information(feat_l, feat_r, bins=10)
                lr_total += l_given_r
                rl_total += r_given_l
                l_total += l_en
                r_total += r_en
                mutual += mutual_inf
                # torch.cuda.empty_cache()
        print("LR: ", lr_total / len(self.train_query_dataloader))
        print("RL: ", rl_total / len(self.train_query_dataloader))
        print("L: ", l_total / len(self.train_query_dataloader))
        print("R: ", r_total / len(self.train_query_dataloader))
        print("M: ", mutual_inf / len(self.train_query_dataloader))


def calc_spatial_entropy(img):
    # 归一化为空间概率分布
    p = img / (np.sum(img) + 1e-10)
    p = p[p > 0] # 只取非零项
    return -np.sum(p * np.log2(p))


def calc_conditional_entropy(target, condition, bins=16, epsilon=1e-10):
    t_flat = target.flatten()
    c_flat = condition.flatten()
    joint_hist, _, _ = np.histogram2d(t_flat, c_flat, bins=bins)
    p_joint = joint_hist / np.sum(joint_hist)
    p_condition = np.sum(p_joint, axis=0)
    p_condition_safe = p_condition[p_condition > 0]
    h_condition = -np.sum(p_condition_safe * np.log2(p_condition_safe))
    p_joint_safe = p_joint[p_joint > 0]
    h_joint = -np.sum(p_joint_safe * np.log2(p_joint_safe))
    h_conditional = h_joint - h_condition
    return h_conditional, h_condition


def calc_mutual_information(target, condition, bins=100):
    t_flat = target.flatten()
    c_flat = condition.flatten()

    joint_hist, _, _ = np.histogram2d(t_flat, c_flat, bins=bins)
    p_joint = joint_hist / np.sum(joint_hist)

    p_target = np.sum(p_joint, axis=1, keepdims=True)
    p_condition = np.sum(p_joint, axis=0, keepdims=True)

    p_independent = p_target @ p_condition

    mi = np.sum(p_joint * np.log2((p_joint + 1e-10) / (p_independent + 1e-10)))
    return mi


def calc_local_conditional_entropy(target, condition, kernel_size=(3, 3), bins=5):
    t_patches = extract_patches_2d(target, kernel_size)
    c_patches = extract_patches_2d(condition, kernel_size)
    local_entropies = []
    for t_patch, c_patch in zip(t_patches, c_patches):
        if np.std(t_patch) < 1e-6 and np.std(c_patch) < 1e-6:
            local_entropies.append(0.0)
        else:
            h = calc_conditional_entropy(t_patch, c_patch, bins=bins)
            local_entropies.append(h)
    return np.mean(local_entropies)


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    param_class = ModelParamsFull()
    trainer = Tester(param_class)
    trainer.do_test()
