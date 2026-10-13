import os

import faiss
import numpy as np
import torch
from torchvision.models import ResNet18_Weights
from tqdm import tqdm


class CrossModalTester:
    def __init__(self, params, model_class, make_dataloader_fn):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class.create(weights=ResNet18_Weights.DEFAULT)
        n_params = sum(p.nelement() for p in self.model.parameters())
        print(f"Number of model parameters: {n_params}")
        self.model.to(self.device)
        self.test_loader, self.test_dataset = make_dataloader_fn("test", params)

    def _load_checkpoint(self):
        if not self.params.resume_checkpoint:
            print("\nZero-shot Test")
            return

        if self.params.checkpoint_path:
            resume_filename = self.params.checkpoint_path
        else:
            resume_filename = os.path.join(
                self.params.weights_root, f"{self.params.resume_epoch}.pth.tar"
            )
        print(f"\nResuming from {resume_filename}")
        checkpoint = torch.load(resume_filename)
        self.model.load_state_dict(checkpoint["state_dict"])

    def do_test(self):
        print("\nEvaluating Process -------->")
        self._load_checkpoint()

        descriptor_query = []
        descriptor_database = []
        with torch.no_grad():
            self.model.eval()
            for batch_idx, batch in enumerate(tqdm(self.test_loader)):
                l_bev = batch["lidar_bev"].to(self.device)
                r_bev = batch["radar_bev"].to(self.device)
                descriptors_l, descriptors_r, _, _ = self.model(l_bev, r_bev)
                for des_idx in range(descriptors_l.shape[0]):
                    global_idx = batch_idx * self.params.test_batch_size + des_idx
                    if global_idx < self.test_dataset.num_db:
                        descriptor_database.append(descriptors_l[des_idx, :].cpu().detach().numpy())
                    else:
                        descriptor_query.append(descriptors_r[des_idx, :].cpu().detach().numpy())

        total_query_num = self.test_dataset.num_query
        descriptor_query = np.array(descriptor_query).astype("float32")
        descriptor_database = np.array(descriptor_database).astype("float32")
        gt = self.test_dataset.get_positives()

        faiss_index = faiss.IndexFlatL2(self.params.output_dim)
        faiss_index.add(descriptor_database)
        _, pred_ind = faiss_index.search(descriptor_query, 50)

        no_gt_num = 0
        top_n = [1, 5, 10, 20, max(1, int(total_query_num * 0.01))]
        successful_recall_num = np.zeros(len(top_n))
        for query_ind in range(total_query_num):
            if len(gt[query_ind]) == 0:
                no_gt_num += 1
                continue
            for ind, n in enumerate(top_n):
                if np.any(np.in1d(pred_ind[query_ind, :n], np.array(gt[query_ind]))):
                    successful_recall_num[ind:] += 1
                    break
        total_query_num -= no_gt_num

        recall_rate = successful_recall_num / float(total_query_num) * 100.0
        print(", ".join(f"{num:.2f}" for num in recall_rate))
        print(f"\nRecall@1: {recall_rate[0]:.2f}")
        print(f"Recall@5: {recall_rate[1]:.2f}")
        print(f"Recall@10: {recall_rate[2]:.2f}")
        print(f"Recall@20: {recall_rate[3]:.2f}")
        print(f"Recall@1percent: {recall_rate[4]:.2f}\n")

        self._print_f1(descriptor_query, descriptor_database, gt)
        return recall_rate

    def _print_f1(self, descriptor_query, descriptor_database, gt):
        faiss_index = faiss.IndexFlatL2(self.params.output_dim)
        faiss_index.add(descriptor_database)
        dists, preds = faiss_index.search(descriptor_query, len(descriptor_database))
        dists_min = dists[:, 0].min()
        if dists_min - 0.1 > 0:
            dists_min -= 0.1
        dists_u = np.linspace(dists_min, dists[:, 0].max() + 0.1, 1000)

        recalls = []
        precisions = []
        print("getting pr...")
        for th in tqdm(dists_u, ncols=40):
            tp = fp = fn = tn = 0
            for index_q in range(dists.shape[0]):
                if dists[index_q, 0] < th:
                    if np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                        tp += 1
                    else:
                        fp += 1
                elif np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                    fn += 1
                else:
                    tn += 1
            if tp + fn == 0 or tp + fp == 0:
                continue
            recalls.append(tp / (tp + fn))
            precisions.append(tp / (tp + fp))

        if recalls:
            recalls = np.array(recalls)
            precisions = np.array(precisions)
            order = np.argsort(recalls)
            f1s = 2 * precisions[order] * recalls[order] / (precisions[order] + recalls[order])
            print("f1 score: ", float(np.max(f1s)))
