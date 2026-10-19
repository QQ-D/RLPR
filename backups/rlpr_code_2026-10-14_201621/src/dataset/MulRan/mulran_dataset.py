import os
import pickle
import numpy as np
import cv2
import math
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from dataset.MulRan.utils import polar_to_cartesian, cartesian_to_polar
import torch.nn.functional as F
import random
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(self, info_root, bev_dataset_root, w=900, h=200, res=4, measure_range=80.0, trans_threshold=[0.2, 0.2]):
        super(BaseDataset, self).__init__()
        self.info_root = info_root
        self.bev_dataset_root = bev_dataset_root
        self.lidar_bev_root = os.path.join(self.bev_dataset_root, 'lidar')
        self.radar_bev_root = os.path.join(self.bev_dataset_root, 'radar')

        self.infos = self.read_info()
        self.l_bev_w = w
        self.l_bev_h = h
        self.r_bev_w = math.ceil(self.l_bev_w / res)
        self.r_bev_h = math.ceil(self.l_bev_h / res)
        self.trans_threshold = trans_threshold
        self.measure_range = measure_range

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def read_info(self):
        infos = list()
        for cur_info_root in self.info_root:
            with open(cur_info_root, 'rb') as f:
                infos.extend(pickle.load(f))
        return infos

    def load_data(self, index, l_enable=True, r_enable=True):
        if r_enable:
            r_bev = self.load_r_bev(index)
            r_bev = r_bev.to(dtype=torch.float32)
        else:
            r_bev = None

        if l_enable:
            l_bev = self.load_l_bev(index)
            l_bev = l_bev.to(dtype=torch.float32)
        else:
            l_bev = None
        return l_bev, r_bev

    def load_l_bev(self, index):
        cur_info = self.infos[index]
        l_filename = os.path.basename(cur_info['lidar_infos']['LIDAR_TOP']['filename']).split('.')[0] + '.png'
        l_bev_filepath = os.path.join(self.lidar_bev_root, l_filename)
        l_bev = cv2.imread(l_bev_filepath, 0)
        l_bev = (l_bev.astype(np.float32)) / 256
        l_bev = l_bev[np.newaxis, :, :].repeat(3, 0)
        l_bev = torch.from_numpy(l_bev)
        return l_bev

    def load_r_bev(self, index):
        cur_info = self.infos[index]
        r_filename = os.path.basename(cur_info['radar_infos']['RADAR_TOP']['filename']).split('.')[0] + '.png'
        r_bev_filepath = os.path.join(self.radar_bev_root, r_filename)
        r_bev = cv2.imread(r_bev_filepath, 0)
        r_bev = (r_bev.astype(np.float32)) / 256
        r_bev = r_bev[np.newaxis, :, :].repeat(3, 0)
        r_bev = torch.from_numpy(r_bev)
        return r_bev

    def translate_polar_bev(self, l_bev, r_bev, l_enable=True, r_enable=True):
        assert self.trans_threshold is not None
        trans_x = np.random.uniform(self.trans_threshold[0], self.trans_threshold[1])
        trans_y = np.random.uniform(self.trans_threshold[0], self.trans_threshold[1])
        t = np.array([trans_x, trans_y])

        if l_enable:
            l_bev_np_new = np.zeros_like(l_bev[0, :, :])
            l_bev_np = l_bev[0, :, :].numpy()
            valid_points = np.argwhere(l_bev_np > 0)
            for point in valid_points:
                h_ind, w_ind = point
                r = self.measure_range * h_ind / self.l_bev_h
                yaw = np.pi * (2 * w_ind / self.l_bev_w - 1.0)
                x_cart, y_cart = polar_to_cartesian(r, yaw)
                x_cart_tran, y_cart_tran = np.array([x_cart, y_cart]) + t
                r_tran, yaw_tran = cartesian_to_polar(x_cart_tran, y_cart_tran)
                w_ind_tran = np.floor((0.5 * (yaw_tran / np.pi + 1.0)) * self.l_bev_w).astype(int)
                h_ind_tran = np.floor(r_tran / self.measure_range * self.l_bev_h).astype(int)
                if w_ind_tran < 0 or w_ind_tran >= self.l_bev_w or h_ind_tran < 0 or h_ind_tran >= self.l_bev_h:
                    continue
                l_bev_np_new[h_ind_tran, w_ind_tran] = l_bev_np[h_ind, w_ind]
            l_bev_new = l_bev_np_new[np.newaxis, :, :].repeat(3, 0)
            l_bev_new = torch.from_numpy(l_bev_new)
        else:
            l_bev_new = l_bev

        if r_enable:
            r_bev_np_new = np.zeros_like(r_bev[0, :, :])
            r_bev_np = r_bev[0, :, :].numpy()
            valid_points = np.argwhere(r_bev_np > 0)
            for point in valid_points:
                h_ind, w_ind = point
                r = self.measure_range * h_ind / self.r_bev_h
                yaw = np.pi * (2 * w_ind / self.r_bev_w - 1.0)
                x_cart, y_cart = polar_to_cartesian(r, yaw)
                x_cart_tran, y_cart_tran = np.array([x_cart, y_cart]) + t
                r_tran, yaw_tran = cartesian_to_polar(x_cart_tran, y_cart_tran)
                w_ind_tran = np.floor((0.5 * (yaw_tran / np.pi + 1.0)) * self.r_bev_w).astype(int)
                h_ind_tran = np.floor(r_tran / self.measure_range * self.r_bev_h).astype(int)
                if w_ind_tran < 0 or w_ind_tran >= self.r_bev_w or h_ind_tran < 0 or h_ind_tran >= self.r_bev_h:
                    continue
                r_bev_np_new[h_ind_tran, w_ind_tran] = r_bev_np[h_ind, w_ind]
            r_bev_new = r_bev_np_new[np.newaxis, :, :].repeat(3, 0)
            r_bev_new = torch.from_numpy(r_bev_new)
        else:
            r_bev_new = r_bev

        return l_bev_new, r_bev_new

    def rotate_polar_bev(self, l_bev, r_bev, l_enable=True, r_enable=True):
        random_rot_angle = np.random.random() / 50
        if l_enable:
            l_bev_new = torch.zeros_like(l_bev)
            l_bev_trans = np.floor(random_rot_angle * self.l_bev_w).astype(int)
            l_bev_new[:, :, :l_bev_trans] = l_bev[:, :, :l_bev_trans]
            l_bev[:, :, 0:(self.l_bev_w - l_bev_trans)] = l_bev[:, :, l_bev_trans:]
            l_bev[:, :, (self.l_bev_w - l_bev_trans):] = l_bev_new[:, :, :l_bev_trans]
        else:
            l_bev = l_bev

        if r_enable:
            r_bev_new = torch.zeros_like(r_bev)
            r_bev_trans = np.floor(random_rot_angle * self.r_bev_w).astype(int)
            r_bev_new[:, :, :r_bev_trans] = r_bev[:, :, :r_bev_trans]
            r_bev[:, :, 0:(self.r_bev_w - r_bev_trans)] = r_bev[:, :, r_bev_trans:]
            r_bev[:, :, (self.r_bev_w - r_bev_trans):] = r_bev_new[:, :, :r_bev_trans]
        else:
            r_bev = r_bev

        return l_bev, r_bev, random_rot_angle


class TripletDataset(BaseDataset):
    def __init__(self, info_root, bev_dataset_root, database_root_list, query_root_list,
                 n_pos, n_neg, neg_dist_thres, pos_dist_thres, desc_dim, augmentation=False,
                 l_enable=True, r_enable=True):
        super().__init__(info_root, bev_dataset_root, trans_threshold=(-pos_dist_thres / 2.0, pos_dist_thres / 2.0))
        # same elements may exist both in database and query
        self.database_root_list = database_root_list
        self.query_root_list = query_root_list
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.neg_dist_thres = neg_dist_thres
        self.pos_dist_thres = pos_dist_thres
        self.desc_dim = desc_dim
        self.augmentation = augmentation
        self.l_enable = l_enable
        self.r_enable = r_enable

        self.len_per_seq = [0]
        self.query_len_per_seq = [0]
        self.dataset_len_per_seq = [0]
        for cur_info_file in info_root:
            with open(cur_info_file, 'rb') as f:
                self.len_per_seq.append(len(pickle.load(f)))

        assert len(self.database_root_list) != 0, print("Database root list is empty!")
        self.database_index_n_pos = np.load(self.database_root_list[0])
        self.dataset_len_per_seq.append(self.database_index_n_pos.shape[0])
        for i in range(1, len(self.database_root_list)):
            database_index_n_pos = np.load(self.database_root_list[i])
            database_index_n_pos[:, 0] = database_index_n_pos[:, 0] + self.len_per_seq[i]
            self.database_index_n_pos = np.concatenate([self.database_index_n_pos, database_index_n_pos], axis=0)
            self.dataset_len_per_seq.append(self.database_index_n_pos.shape[0])
        assert len(self.query_root_list) != 0, print("Query root list is empty!")
        self.query_index_n_pos = np.load(self.query_root_list[0])
        self.query_len_per_seq.append(self.query_index_n_pos.shape[0])
        for i in range(1, len(query_root_list)):
            query_index_n_pos = np.load(self.query_root_list[i])
            query_index_n_pos[:, 0] = query_index_n_pos[:, 0] + self.len_per_seq[i]
            self.query_index_n_pos = np.concatenate([self.query_index_n_pos, query_index_n_pos], axis=0)
            self.query_len_per_seq.append(self.query_index_n_pos.shape[0])

        self.latent_vectors = np.zeros([len(self.database_index_n_pos) + len(self.query_index_n_pos), 256])
        self.is_batch_hard_mining = False

        self.pos_index = list()
        for i in range(len(self.len_per_seq) - 1):
            knn = NearestNeighbors()
            knn.fit(self.database_index_n_pos[self.dataset_len_per_seq[i]:self.dataset_len_per_seq[i + 1], 1:])
            dist, pos_index_array = knn.radius_neighbors(
                self.query_index_n_pos[self.query_len_per_seq[i]:self.query_len_per_seq[i + 1], 1:],
                radius=self.pos_dist_thres,
                return_distance=True)
            self.pos_index.extend(list(pos_index_array + sum(self.dataset_len_per_seq[:i + 1])))
        for i, posi in enumerate(self.pos_index):
            # pos_index = np.sort(posi[dist[i] != 0.])
            pos_index = np.sort(posi)
            self.pos_index[i] = pos_index

        self.neg_index = list()
        for i in range(len(self.len_per_seq) - 1):
            knn = NearestNeighbors()
            knn.fit(self.database_index_n_pos[self.dataset_len_per_seq[i]:self.dataset_len_per_seq[i + 1], 1:])
            potential_positives = list(knn.radius_neighbors(
                self.query_index_n_pos[self.query_len_per_seq[i]:self.query_len_per_seq[i + 1], 1:],
                radius=self.neg_dist_thres,
                return_distance=False))
            for pos in potential_positives:
                self.neg_index.append(np.setdiff1d(np.arange(self.dataset_len_per_seq[i + 1] - self.dataset_len_per_seq[i]), pos, assume_unique=True) + sum(self.dataset_len_per_seq[:i+1]))

    def __len__(self):
        return len(self.query_index_n_pos)

    def __getitem__(self, index):
        l_bev_list = list()
        r_bev_list = list()

        query_index_in_info = self.query_index_n_pos[index, 0].astype(int)
        query_l_bev, query_r_bev = self.load_data(query_index_in_info, self.l_enable, self.r_enable)
        if self.augmentation:
            # query_l_bev, query_r_bev = self.jitter_polar_bev(query_l_bev, query_r_bev)
            # query_l_bev, query_r_bev = self.drop_polar_bev(query_l_bev, query_r_bev)
            query_l_bev, query_r_bev = self.translate_polar_bev(query_l_bev, query_r_bev, self.l_enable, self.r_enable)
            # query_l_bev = self.erasing(query_l_bev)
            # query_l_bev, query_r_bev = self.rotate_polar_bev(query_l_bev, query_r_bev)
        if self.l_enable:
            l_bev_list.append(query_l_bev)
        if self.r_enable:
            r_bev_list.append(query_r_bev)

        if self.pos_index[index].shape[0] == 0:
            return None
        pos_sample = np.random.choice(self.pos_index[index], self.n_pos).astype(int)
        pos_index_in_info = self.database_index_n_pos[pos_sample, 0].astype(int)
        for i in range(len(pos_index_in_info)):
            l_bev, r_bev = self.load_data(pos_index_in_info[i], self.l_enable, self.r_enable)
            if self.l_enable:
                l_bev_list.append(l_bev)
            if self.r_enable:
                r_bev_list.append(r_bev)

        if self.is_batch_hard_mining:
            query_in_seq_idx = 0
            for i, seq_len in enumerate(self.query_len_per_seq):
                if index < seq_len:
                    query_in_seq_idx = i - 1
            query_index = index + self.database_index_n_pos.shape[0] + sum(self.query_len_per_seq[:query_in_seq_idx])
            query_des = torch.tensor(self.latent_vectors[query_index, :])
            neg_sample = self.latent_vectors[np.array(self.neg_index[index]), :]
            neg_des = torch.tensor(neg_sample)
            dist = - torch.norm(query_des[None, :] - neg_des, dim=1) + 0.5
            result = dist.topk(self.n_neg, largest=True)
            neg_dist, neg_idx_in_sample = result.values, result.indices
            neg_idx = np.array(self.neg_index[index])[neg_idx_in_sample]
        else:
            neg_idx = np.random.choice(self.neg_index[index], self.n_neg).astype(int)
        neg_index_in_info = self.database_index_n_pos[neg_idx, 0].astype(int)
        for i in range(len(neg_index_in_info)):
            l_bev, r_bev = self.load_data(neg_index_in_info[i], self.l_enable, self.r_enable)
            if self.l_enable:
                l_bev_list.append(l_bev)
            if self.r_enable:
                r_bev_list.append(r_bev)

        if self.l_enable:
            l_bev_tensor = torch.stack(l_bev_list, dim=0)
        else:
            l_bev_tensor = torch.tensor(1)
        img_tensor = torch.tensor(1)
        img_metas = torch.tensor(1)
        lidar_tensor = torch.tensor(1)
        if self.r_enable:
            r_bev_tensor = torch.stack(r_bev_list, dim=0)
        else:
            r_bev_tensor = torch.tensor(1)

        res_dict = dict({'image': img_tensor, 'img_metas': img_metas, 'lidar': lidar_tensor,
                         'lidar_bev': l_bev_tensor, 'radar_bev': r_bev_tensor})
        return res_dict

    def update_latent_vectors(self, vecs_filepath):
        latent_vectors = pickle.load(open(vecs_filepath, 'rb'))
        latent_vectors_full = np.zeros([len(self.database_index_n_pos) + len(self.query_index_n_pos), self.desc_dim])
        for i in range(len(latent_vectors)):
            latent_vectors_full[i, :] = latent_vectors[i]
        self.latent_vectors = latent_vectors_full
        self.is_batch_hard_mining = True


class QueryDataset(BaseDataset):
    def __init__(self, info_root, bev_dataset_root, database_root_list, query_root_list, non_triv_pos_dist_thres,
                 l_enable=True, r_enable=True):
        # database and query are separated
        super().__init__(info_root, bev_dataset_root)
        self.database_root_list = database_root_list
        self.query_root_list = query_root_list
        self.positives = None
        self.non_triv_pos_dist_thres = non_triv_pos_dist_thres
        self.l_enable = l_enable
        self.r_enable = r_enable

        self.len_per_seq = [0]
        self.query_len_per_seq = [0]
        self.dataset_len_per_seq = [0]
        for cur_info_file in info_root:
            with open(cur_info_file, 'rb') as f:
                self.len_per_seq.append(len(pickle.load(f)))

        # assert len(self.database_root_list) != 0, print("Database root list is empty!")
        if len(self.database_root_list) != 0:
            self.database_index_n_pos = np.load(self.database_root_list[0])
            self.dataset_len_per_seq.append(self.database_index_n_pos.shape[0])
            for i in range(1, len(self.database_root_list)):
                database_index_n_pos = np.load(self.database_root_list[i])
                database_index_n_pos[:, 0] = database_index_n_pos[:, 0] + self.len_per_seq[i]
                self.database_index_n_pos = np.concatenate([self.database_index_n_pos, database_index_n_pos], axis=0)
                self.dataset_len_per_seq.append(self.database_index_n_pos.shape[0])
        # assert len(self.query_root_list) != 0, print("Query root list is empty!")
        if len(self.query_root_list) != 0:
            self.query_index_n_pos = np.load(self.query_root_list[0])
            self.query_len_per_seq.append(self.query_index_n_pos.shape[0])
            for i in range(1, len(query_root_list)):
                query_index_n_pos = np.load(self.query_root_list[i])
                query_index_n_pos[:, 0] = query_index_n_pos[:, 0] + self.len_per_seq[i]
                self.query_index_n_pos = np.concatenate([self.query_index_n_pos, query_index_n_pos], axis=0)
                self.query_len_per_seq.append(self.query_index_n_pos.shape[0])

        self.dataset_index_n_pos = np.concatenate([self.database_index_n_pos, self.query_index_n_pos], axis=0)
        self.num_db = self.database_index_n_pos.shape[0]
        self.num_query = self.query_index_n_pos.shape[0]

    def __len__(self):
        return len(self.dataset_index_n_pos)

    def __getitem__(self, index):
        l_bev_list = list()
        r_bev_list = list()
        # if index > self.num_db:
        #     if 'fog_0.0163' not in self.lidar_bev_root:
        #         self.lidar_bev_root = self.lidar_bev_root + '_fog_0.0163'
        index_in_info = self.dataset_index_n_pos[index, 0].astype(int)

        l_bev, r_bev = self.load_data(index_in_info, self.l_enable, self.r_enable)
        # l_bev, r_bev, _ = self.rotate_polar_bev(l_bev, r_bev, l_enable=False, r_enable=True)
        # l_bev, r_bev = self.translate_polar_bev(l_bev, r_bev, l_enable=False, r_enable=True)
        if self.l_enable:
            l_bev_list.append(l_bev)
        if self.r_enable:
            r_bev_list.append(r_bev)

        if self.l_enable:
            l_bev_tensor = torch.stack(l_bev_list, dim=0)
        else:
            l_bev_tensor = torch.tensor(1)
        img_tensor = torch.tensor(1)
        img_metas = torch.tensor(1)
        lidar_tensor = torch.tensor(1)
        if self.r_enable:
            r_bev_tensor = torch.stack(r_bev_list, dim=0)
        else:
            r_bev_tensor = torch.tensor(1)

        res_dict = dict({'image': img_tensor, 'img_metas': img_metas, 'lidar': lidar_tensor,
                         'lidar_bev': l_bev_tensor, 'radar_bev': r_bev_tensor})
        return res_dict

    def get_positives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors()
            dataset = np.ascontiguousarray(self.dataset_index_n_pos[:self.num_db, 1:])
            knn.fit(dataset)
            self.positives = list(knn.radius_neighbors(self.dataset_index_n_pos[self.num_db:, 1:],
                                                       radius=self.non_triv_pos_dist_thres,
                                                       return_distance=False))
        return self.positives
