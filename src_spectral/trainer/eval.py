import os

import faiss
import numpy as np
import torch
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
from utils.pretrain import validate_checkpoint_variant, freeze_radar_branch_cross


def max_f1_score(precisions, recalls):
    """Maximum F1 on the original PR curve, treating a zero/zero point as 0."""
    precisions = np.asarray(precisions, dtype=np.float64)
    recalls = np.asarray(recalls, dtype=np.float64)
    if precisions.shape != recalls.shape:
        raise ValueError('Precision and recall arrays must have matching shapes')
    if precisions.size == 0:
        return 0.
    denominator = precisions + recalls
    scores = np.divide(2 * precisions * recalls, denominator,
                       out=np.zeros_like(denominator), where=denominator > 0)
    return float(scores.max())


def recall_at_k(predictions, ground_truth, top_n):
    """Count each retrieval cutoff independently; cutoffs need not be sorted."""
    if len(predictions) != len(ground_truth):
        raise ValueError('Prediction and ground truth query counts must match')
    successes = np.zeros(len(top_n), dtype=np.float64)
    valid_queries = 0
    for prediction, positives in zip(predictions, ground_truth):
        if len(positives) == 0:
            continue
        valid_queries += 1
        for index, count in enumerate(top_n):
            successes[index] += np.isin(prediction[:count], positives).any()
    if valid_queries == 0:
        raise ValueError('No test queries have ground truth positives')
    return successes / valid_queries * 100.


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
        validate_checkpoint_variant(self.model, checkpoint)
        self.model.load_state_dict(checkpoint["state_dict"])
        if getattr(self.model, 'model_variant', 'rlpr') == 'spectral_word':
            freeze_radar_branch_cross(self.model)

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
        # Keep the author's query-count convention for this legacy metric.
        top_n = [1, 5, 10, 20, max(1, int(total_query_num * 0.01))]
        _, pred_ind = faiss_index.search(descriptor_query, min(len(descriptor_database), max(50, max(top_n))))
        recall_rate = recall_at_k(pred_ind, gt, top_n)
        print(", ".join(f"{num:.2f}" for num in recall_rate))
        print(f"\nRecall@1: {recall_rate[0]:.2f}")
        print(f"Recall@5: {recall_rate[1]:.2f}")
        print(f"Recall@10: {recall_rate[2]:.2f}")
        print(f"Recall@20: {recall_rate[3]:.2f}")
        print(f"Recall@1percent: {recall_rate[4]:.2f} (Top-{top_n[4]}, original query-count convention)\n")

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

        score = max_f1_score(precisions, recalls)
        print("f1 score: ", score)
        return score
