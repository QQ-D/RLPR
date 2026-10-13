"""Shared radar feature separability metrics (reviewer rebuttal analysis)."""

import os
import pickle

import faiss
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

PAIR_KEYS = ["H_R", "silhouette_pair", "fisher_pair", "rho_pair", "AR@1"]
PART_KEYS = ["H_R", "silhouette", "fisher_J", "rho", "AR@1"]


@torch.no_grad()
def extract_radar_features(model, loader, device, num_db, bins_h=10):
    descs = []
    h_r_batches = []
    model.eval()
    for batch in tqdm(loader, desc="extracting R features"):
        l_bev = batch["lidar_bev"].to(device)
        r_bev = batch["radar_bev"].to(device)
        _, descriptors_r, _, feat_r = model(l_bev, r_bev)
        descs.append(descriptors_r.cpu().numpy().astype(np.float32))
        feat_r_np = feat_r.detach().cpu().numpy().flatten()
        hist, _ = np.histogram(feat_r_np, bins=bins_h)
        p = hist.astype(np.float64) / max(hist.sum(), 1)
        p = p[p > 0]
        h_r_batches.append(float(-np.sum(p * np.log2(p))))

    descs = np.concatenate(descs, axis=0).astype(np.float32)
    is_db = np.arange(descs.shape[0]) < num_db
    h_r = float(np.mean(h_r_batches))
    return descs, is_db, h_r


def assign_place_labels(xy, pos_radius=9.0, linkage="complete"):
    if xy.shape[0] <= 1:
        return np.zeros(xy.shape[0], dtype=np.int64)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=pos_radius,
        linkage=linkage,
        metric="euclidean",
    )
    return clustering.fit_predict(xy).astype(np.int64)


def filter_singleton_classes(feats, labels):
    unique, counts = np.unique(labels, return_counts=True)
    keep = set(unique[counts >= 2].tolist())
    mask = np.array([lab in keep for lab in labels])
    return feats[mask], labels[mask], mask


def fisher_discriminant_ratio(feats, labels):
    mu = feats.mean(axis=0)
    sb_trace = 0.0
    sw_trace = 0.0
    for c in np.unique(labels):
        xc = feats[labels == c]
        mu_c = xc.mean(axis=0)
        sb_trace += xc.shape[0] * float(np.sum((mu_c - mu) ** 2))
        sw_trace += float(np.sum((xc - mu_c) ** 2))
    return sb_trace / (sw_trace + 1e-12)


def inter_intra_ratio(feats, labels, max_pairs=400_000, seed=0):
    rng = np.random.default_rng(seed)
    n = feats.shape[0]
    n_draw = int(max_pairs * 1.2) + 1
    i = rng.integers(0, n, size=n_draw)
    j = rng.integers(0, n, size=n_draw)
    mask = i != j
    i, j = i[mask][:max_pairs], j[mask][:max_pairs]
    d = np.linalg.norm(feats[i] - feats[j], axis=1)
    same = labels[i] == labels[j]
    if same.sum() == 0 or (~same).sum() == 0:
        return float("nan"), float("nan"), float("nan")
    d_intra = float(d[same].mean())
    d_inter = float(d[~same].mean())
    return d_inter / (d_intra + 1e-12), d_intra, d_inter


def pairwise_separability(feats, xy, pos_thres=9.0, neg_thres=12.0,
                          max_pairs=800_000, seed=0):
    rng = np.random.default_rng(seed)
    n = feats.shape[0]
    n_draw = int(max_pairs * 8)
    i = rng.integers(0, n, size=n_draw)
    j = rng.integers(0, n, size=n_draw)
    mask = i < j
    i, j = i[mask], j[mask]

    d_sp = np.linalg.norm(xy[i] - xy[j], axis=1)
    d_ft = np.linalg.norm(feats[i] - feats[j], axis=1)

    d_intra = d_ft[d_sp < pos_thres][:max_pairs]
    d_inter = d_ft[d_sp > neg_thres][:max_pairs]

    if d_intra.size == 0 or d_inter.size == 0:
        return {
            "rho_pair": float("nan"),
            "fisher_pair": float("nan"),
            "d_intra_pair": float("nan"),
            "d_inter_pair": float("nan"),
            "n_intra_pairs": int(d_intra.size),
            "n_inter_pairs": int(d_inter.size),
        }

    m_intra = float(d_intra.mean())
    m_inter = float(d_inter.mean())
    s2_intra = float((d_intra ** 2).mean())
    s2_inter = float((d_inter ** 2).mean())
    return {
        "rho_pair": m_inter / (m_intra + 1e-12),
        "fisher_pair": s2_inter / (s2_intra + 1e-12),
        "d_intra_pair": m_intra,
        "d_inter_pair": m_inter,
        "n_intra_pairs": int(d_intra.size),
        "n_inter_pairs": int(d_inter.size),
    }


def per_scan_silhouette(feats, xy, pos_thres=9.0, neg_thres=12.0,
                        n_inter_samples=256, seed=0):
    rng = np.random.default_rng(seed)
    n = feats.shape[0]
    knn = NearestNeighbors(radius=pos_thres, algorithm="ball_tree").fit(xy)
    neigh = knn.radius_neighbors(xy, return_distance=False)

    s_list = []
    for i in range(n):
        idx_intra = neigh[i][neigh[i] != i]
        if idx_intra.size == 0:
            continue
        cand = rng.integers(0, n, size=8 * n_inter_samples)
        cand = cand[cand != i]
        d_sp = np.linalg.norm(xy[cand] - xy[i], axis=1)
        idx_inter = cand[d_sp > neg_thres][:n_inter_samples]
        if idx_inter.size == 0:
            continue
        a = float(np.linalg.norm(feats[idx_intra] - feats[i], axis=1).mean())
        b = float(np.linalg.norm(feats[idx_inter] - feats[i], axis=1).mean())
        denom = max(a, b, 1e-12)
        s_list.append((b - a) / denom)
    if not s_list:
        return float("nan"), 0
    return float(np.mean(s_list)), len(s_list)


def rpr_ar_at_1(feats, is_db, xy, pos_thres=9.0):
    db = feats[is_db].astype("float32")
    qy = feats[~is_db].astype("float32")
    db_xy = xy[is_db]
    qy_xy = xy[~is_db]
    if db.size == 0 or qy.size == 0:
        return float("nan")

    index = faiss.IndexFlatL2(feats.shape[1])
    index.add(db)
    _, pred = index.search(qy, 1)

    hit, evaluated = 0, 0
    for q in range(qy.shape[0]):
        d2 = np.linalg.norm(db_xy - qy_xy[q], axis=1)
        gt = np.where(d2 < pos_thres)[0]
        if gt.size == 0:
            continue
        evaluated += 1
        if pred[q, 0] in gt:
            hit += 1
    return 100.0 * hit / max(evaluated, 1)


def load_pretrained_radar(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded {ckpt_path}")
    print(f"  param keys: {len(state)}  missing={len(missing)}  "
          f"unexpected={len(unexpected)}")


def evaluate_stage(stage_name, model, loader, dataset, device, args):
    print(f"\n=================== {stage_name} ===================")
    feats, is_db, h_r = extract_radar_features(
        model, loader, device, num_db=dataset.num_db, bins_h=args.h_bins)

    xy = dataset.dataset_index_n_pos[:, 1:3].astype(np.float32)

    labels = assign_place_labels(xy, pos_radius=args.pos_radius, linkage=args.linkage)
    feats_f, labels_f, _ = filter_singleton_classes(feats, labels)
    n_classes = int(np.unique(labels_f).size)
    cls_sizes = np.unique(labels_f, return_counts=True)[1] if feats_f.size else np.array([])
    print(f"  scans (total | after singleton filter): "
          f"{feats.shape[0]} | {feats_f.shape[0]}")
    print(f"  place classes (>= 2 scans):              {n_classes}   "
          f"(median size = {int(np.median(cls_sizes)) if cls_sizes.size else 0}, "
          f"max size = {int(cls_sizes.max()) if cls_sizes.size else 0})   "
          f"[linkage={args.linkage}]")

    if feats_f.shape[0] >= 2 and n_classes >= 2:
        sil = float(silhouette_score(feats_f, labels_f, metric="euclidean"))
        j = fisher_discriminant_ratio(feats_f, labels_f)
        rho, d_intra, d_inter = inter_intra_ratio(
            feats_f, labels_f, max_pairs=args.max_pairs, seed=args.seed)
    else:
        sil = j = rho = d_intra = d_inter = float("nan")

    ar1 = rpr_ar_at_1(feats, is_db, xy, pos_thres=args.pos_radius)

    pw = pairwise_separability(
        feats, xy,
        pos_thres=args.pos_radius, neg_thres=args.neg_radius,
        max_pairs=args.max_pairs, seed=args.seed)
    sil_pair, n_sil_scans = per_scan_silhouette(
        feats, xy,
        pos_thres=args.pos_radius, neg_thres=args.neg_radius,
        n_inter_samples=args.n_inter_samples, seed=args.seed)

    print(f"  H(R)                       = {h_r:.4f}")
    print(f"  [partition / {args.linkage:>8s} linkage]")
    print(f"    Silhouette s             = {sil:.4f}")
    print(f"    Fisher J                 = {j:.4f}")
    print(f"    Inter/Intra rho          = {rho:.4f}   "
          f"(d_intra={d_intra:.4f}, d_inter={d_inter:.4f})")
    print(f"  [pair-based / r_pos<{args.pos_radius:.0f}m vs r_neg>{args.neg_radius:.0f}m]")
    print(f"    Silhouette s (per-scan)  = {sil_pair:.4f}   "
          f"({n_sil_scans} / {feats.shape[0]} scans had intra & inter neighbours)")
    print(f"    Fisher J (pair)          = {pw['fisher_pair']:.4f}")
    print(f"    Inter/Intra rho (pair)   = {pw['rho_pair']:.4f}   "
          f"(d_intra={pw['d_intra_pair']:.4f}, d_inter={pw['d_inter_pair']:.4f}; "
          f"n_intra_pairs={pw['n_intra_pairs']}, n_inter_pairs={pw['n_inter_pairs']})")
    print(f"  RPR AR@1                   = {ar1:.2f}")

    out = {
        "H_R": h_r,
        "silhouette": sil,
        "fisher_J": j,
        "rho": rho,
        "d_intra": d_intra,
        "d_inter": d_inter,
        "silhouette_pair": sil_pair,
        "silhouette_pair_n_scans": n_sil_scans,
        "fisher_pair": pw["fisher_pair"],
        "rho_pair": pw["rho_pair"],
        "d_intra_pair": pw["d_intra_pair"],
        "d_inter_pair": pw["d_inter_pair"],
        "n_intra_pairs": pw["n_intra_pairs"],
        "n_inter_pairs": pw["n_inter_pairs"],
        "AR@1": ar1,
        "n_scans_total": int(feats.shape[0]),
        "n_scans_kept": int(feats_f.shape[0]),
        "n_classes": n_classes,
        "linkage": args.linkage,
        "features": feats,
        "labels": labels,
        "is_db": is_db,
        "xy": xy,
    }
    if args.save_tsne:
        try:
            from sklearn.manifold import TSNE
            perp = max(5, min(30, feats.shape[0] // 50))
            print(f"  running t-SNE (perplexity={perp}) ...")
            out["tsne"] = TSNE(
                n_components=2, init="pca", random_state=args.seed, perplexity=perp
            ).fit_transform(feats)
        except Exception as exc:
            print(f"  t-SNE skipped: {exc}")
    return out


def mean_results(results_list, keys):
    out = {}
    for key in keys:
        vals = [
            r[key] for r in results_list
            if r is not None and not np.isnan(r.get(key, float("nan")))
        ]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def print_table(title, subtitle, col_headers, col_keys, stage_rows, avg_rows=None):
    w_stage = 26
    w_col = 12
    print(f"\n{'=' * (w_stage + w_col * len(col_headers))}")
    print(title)
    print(subtitle)
    header = f"{'Stage':<{w_stage}}" + "".join(f"{h:>{w_col}}" for h in col_headers)
    print(header)
    print("-" * len(header))
    for label, row in stage_rows:
        if row is None:
            continue
        line = f"{label:<{w_stage}}"
        for key in col_keys:
            val = row.get(key, float("nan"))
            line += f"{val:>{w_col}.4f}" if key != "AR@1" else f"{val:>{w_col}.2f}"
        print(line)
    if avg_rows:
        print("·" * len(header))
        for label, row in avg_rows:
            if row is None:
                continue
            line = f"{label:<{w_stage}}"
            for key in col_keys:
                val = row.get(key, float("nan"))
                line += f"{val:>{w_col}.4f}" if key != "AR@1" else f"{val:>{w_col}.2f}"
            print(line)


def save_results(all_results, out_pickle):
    out_dir = os.path.dirname(out_pickle) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_pickle, "wb") as handle:
        pickle.dump(all_results, handle)
    print(f"\nSaved all results to {out_pickle}")
