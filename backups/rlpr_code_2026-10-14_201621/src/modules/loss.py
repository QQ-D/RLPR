import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[0]
    query_copies = query.repeat(int(num_pos), 1)  # 重复增加列数，以进行矩阵运算
    diff = ((pos_vecs - query_copies) ** 2).sum(1)

    min_pos, _ = diff.min(0)  # easiest positive
    max_pos, _ = diff.max(0)  # hardest positive
    return min_pos, max_pos


def triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[0]
    query_copies = q_vec.repeat(int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(int(num_neg), 1)

    negative = ((neg_vecs - query_copies) ** 2).sum(1).unsqueeze(1)  # query与所有neg的距离之和

    loss = margin + positive - ((neg_vecs - query_copies) ** 2).sum(1).unsqueeze(1)
    loss = loss.clamp(min=0.0)
    if lazy:
        triplet_loss = loss.max(0)[0]
    else:
        triplet_loss = loss.sum(0) / num_neg
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss
    return triplet_loss


def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):

    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[0]
    query_copies = q_vec.repeat(int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(int(num_neg), 1)

    loss = m1 + positive - ((neg_vecs - query_copies) ** 2).sum(1).unsqueeze(1)
    loss = loss.clamp(min=0.0)
    if lazy:
        triplet_loss = loss.max(0)[0]
    else:
        triplet_loss = loss.sum(0) / num_neg
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss

    other_neg_copies = other_neg.repeat(int(num_neg), 1)
    second_loss = m2 + positive - ((neg_vecs - other_neg_copies) ** 2).sum(1).unsqueeze(1)
    second_loss = second_loss.clamp(min=0.0)
    if lazy:
        second_loss = second_loss.max(0)[0]
    else:
        second_loss = second_loss.sum(0) / num_neg
    if ignore_zero_loss:
        hard_second = torch.gt(second_loss, 1e-16).float()
        num_hard_second = torch.sum(hard_second)
        second_loss = second_loss.sum() / (num_hard_second + 1e-16)
    else:
        second_loss = second_loss

    total_loss = triplet_loss + second_loss

    return total_loss


def structure_aware_loss(old_rep, new_rep):
    with torch.no_grad():
        old_vec = old_rep.unsqueeze(0) - old_rep.unsqueeze(1)  # B x D x D
        norm_old_vec = F.normalize(old_vec, p=2, dim=2)
        old_angles = torch.bmm(norm_old_vec, norm_old_vec.transpose(1, 2)).view(-1)

    new_vec = new_rep.unsqueeze(0) - new_rep.unsqueeze(1)
    norm_new_vec = F.normalize(new_vec, p=2, dim=2)
    new_angles = torch.bmm(norm_new_vec, norm_new_vec.transpose(1, 2)).view(-1)

    loss_incremental = F.smooth_l1_loss(new_angles, old_angles, reduction='none')
    loss_incremental = F.relu(loss_incremental - 0.0002)

    # Remove 0 terms from loss which emerge due to margin
    # Only do if there are any terms where inc. loss is not zero
    if torch.any(loss_incremental > 0):
        loss_incremental = loss_incremental[loss_incremental > 0]

    # loss_incremental = self.weight * loss_incremental.mean()
    loss_incremental = loss_incremental.mean()
    return loss_incremental


def img_token_loss(token_query, token_possible_pos, token_neg,
                   extrinsics_query, extrinsics_pos,
                   loss_margin, lazy, pos_angle_thresh=15, neg_angle_thresh=55):
    # [0:6]是6个query，[6:12]是可能存在正样本的区段，[12:]全是负样本
    loss = 0.
    B = token_query.shape[0]
    valid_pair_num = 0
    for query_idx in range(B):
        cur_query = token_query[query_idx]
        cur_query_extr = extrinsics_query[query_idx]
        pos_idx_in_possible_pos = list()
        neg_idx_in_possible_pos = list()
        for pos_idx in range(B):
            cur_possible_pos_extr = extrinsics_pos[pos_idx]
            R1 = cur_query_extr[:3, :3]
            R2 = cur_possible_pos_extr[:3, :3]
            R_rel = torch.matmul(R2.T, R1)
            trace_R = torch.trace(R_rel)
            angle = torch.acos((trace_R - 1) / 2)
            angle_deg = angle * 180 / torch.pi
            if angle_deg < pos_angle_thresh:
                pos_idx_in_possible_pos.append(pos_idx)
            if angle_deg > neg_angle_thresh:
                neg_idx_in_possible_pos.append(pos_idx)
        pos_idx_in_possible_pos = np.array(pos_idx_in_possible_pos)
        neg_idx_in_possible_pos = np.array(neg_idx_in_possible_pos)
        cur_pos = token_possible_pos[pos_idx_in_possible_pos, :]
        cur_neg = torch.cat([token_possible_pos[neg_idx_in_possible_pos, :], token_neg], dim=0)
        if cur_pos.shape[0] == 0:
            continue
        loss_cur = triplet_loss(cur_query, cur_pos, cur_neg, loss_margin, lazy)
        loss += loss_cur
        valid_pair_num += 1
    if valid_pair_num != 0:
        loss = loss / valid_pair_num
    return loss, valid_pair_num


def feature_alignment_loss(feat_1, feat_2):
    diff = feat_2 - feat_1
    pixel_dist = torch.norm(diff, p=2, dim=1)
    B, H, W = pixel_dist.shape[0], pixel_dist.shape[-2], pixel_dist.shape[-1]
    return torch.sum(pixel_dist) / (B * H * W)


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query, keys):
        query = F.normalize(query, dim=1)
        keys = F.normalize(keys, dim=1)

        logits = torch.matmul(query, keys.T) / self.temperature

        batch_size = query.shape[0]
        labels = torch.arange(batch_size).to(query.device)

        loss = self.criterion(logits, labels)
        return loss
