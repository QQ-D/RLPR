#!/usr/bin/env python3
"""
Class-wise radar feature separability analysis (reviewer rebuttal, Table R1).

Unified entry for NuScenes, Snail-Radar, MulRan, and Boreas. Shared metrics live
in `analysis/r_feat_separability_core.py`.

Examples
--------
    python r_feat_separability.py --dataset nusc \\
        --pretrained_r ./outputs/r_nusc/weights/5.pth.tar

    python r_feat_separability.py --dataset snail \\
        --pretrained_r ./outputs/r_snail/weights/5.pth.tar

    python r_feat_separability.py --dataset mulran_boreas \\
        --pretrained_r_mulran ./outputs/r_mulran/weights/5.pth.tar \\
        --pretrained_r_boreas ./outputs/r_mulran/weights/5.pth.tar
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torchvision.models import ResNet18_Weights

from analysis.r_feat_separability_core import (
    PAIR_KEYS,
    PART_KEYS,
    evaluate_stage,
    load_pretrained_radar,
    mean_results,
    print_table,
    save_results,
)
from config.params_full import ModelParamsFull
from dataset.Boreas.boreas_dataloader import make_dataloader as make_dataloader_boreas
from dataset.MulRan.mulran_dataloader import make_dataloader as make_dataloader_mulran
from dataset.NuScenes.nuscenes_dataloader import make_dataloader as make_dataloader_nusc
from dataset.SnailRadar.snail_dataloader import make_dataloader as make_dataloader_snail
from modules.LR_cross_hffc_225_dis import LRCross as LRCrossPolar
from modules.LR_cross_hffcn_225_dis import LRCross as LRCrossCart
from utils.seed import set_global_seed

sys.maxsize = 3000
np.set_printoptions(threshold=sys.maxsize)

_NUSC_SEQ_CFG = {
    "bs": {
        "info": "nuscenes_infos-bs.pkl",
        "bev": "bs_shv_bev",
        "db": "bs_db.npy",
        "test_q": "bs_test_query.npy",
    },
    "son": {
        "info": "nuscenes_infos-son.pkl",
        "bev": "son_bev",
        "db": "son_db.npy",
        "test_q": "son_test_query.npy",
    },
    "sq": {
        "info": "nuscenes_infos-sq.pkl",
        "bev": "sq_bev",
        "db": "sq_db.npy",
        "test_q": "sq_test_query.npy",
    },
}

_MULRAN_SEQ_CFG = {
    "sejong": {
        "info": "mulran_infos-sejong.pkl",
        "db": "sejong_test_db.npy",
        "q": "sejong_test_query.npy",
    },
    "riverside": {
        "info": "mulran_infos-riverside.pkl",
        "db": "riverside_test_db.npy",
        "q": "riverside_test_query.npy",
    },
    "dcc": {
        "info": "mulran_infos-dcc.pkl",
        "db": "dcc_test_db.npy",
        "q": "dcc_test_query.npy",
    },
}

_BOREAS_SEQ_CFG = {
    "bor-snow": {
        "info": "boreas_infos-2021-01-26-11-22.pkl",
        "db": "boreas_infos-2021-01-26-11-22_db.npy",
        "q": "boreas_infos-2021-01-26-11-22_test_query.npy",
    },
    "bor-clear": {
        "info": "boreas_infos-2021-09-14-20-00.pkl",
        "db": "boreas_infos-2021-09-14-20-00_db.npy",
        "q": "boreas_infos-2021-09-14-20-00_test_query.npy",
    },
}


def override_nusc_test_params(params, seq):
    seq = seq.lower()
    if seq not in _NUSC_SEQ_CFG:
        raise ValueError(f"Unknown NuScenes sequence '{seq}'. Valid: {list(_NUSC_SEQ_CFG)}")
    cfg = _NUSC_SEQ_CFG[seq]
    nusc = params.nusc_param
    base = nusc.base_root
    nusc.test_info_root = [os.path.join(base, "info", cfg["info"])]
    nusc.test_bev_dataset_root = os.path.join(base, "bev", cfg["bev"])
    nusc.test_database_index_list = [os.path.join(base, "index", cfg["db"])]
    nusc.test_query_index_list = [os.path.join(base, "index", cfg["test_q"])]
    return params


def override_mulran_test_params(params, seq):
    seq = seq.lower()
    if seq not in _MULRAN_SEQ_CFG:
        raise ValueError(f"Unknown MulRan sequence '{seq}'. Valid: {list(_MULRAN_SEQ_CFG)}")
    cfg = _MULRAN_SEQ_CFG[seq]
    mp = params.mulran_param
    base = mp.base_root
    mp.test_info_root = [os.path.join(base, "info", cfg["info"])]
    mp.test_bev_dataset_root = os.path.join(base, "bev", "mm_bev")
    mp.test_database_index_list = [os.path.join(base, "index", cfg["db"])]
    mp.test_query_index_list = [os.path.join(base, "index", cfg["q"])]
    return params


def override_boreas_test_params(params, seq):
    seq = seq.lower()
    if seq not in _BOREAS_SEQ_CFG:
        raise ValueError(f"Unknown Boreas sequence '{seq}'. Valid: {list(_BOREAS_SEQ_CFG)}")
    cfg = _BOREAS_SEQ_CFG[seq]
    bp = params.boreas_param
    base = bp.base_root
    bp.test_info_root = [os.path.join(base, "info", cfg["info"])]
    bp.test_bev_dataset_root = os.path.join(base, "bev", "mm_bev")
    bp.test_database_index_list = [os.path.join(base, "index", cfg["db"])]
    bp.test_query_index_list = [os.path.join(base, "index", cfg["q"])]
    return params


def _require_checkpoint(path, name):
    if not path:
        raise ValueError(f"{name} is required.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def _add_metric_args(parser):
    parser.add_argument("--pos_radius", type=float, default=9.0)
    parser.add_argument("--neg_radius", type=float, default=12.0)
    parser.add_argument(
        "--linkage", type=str, default="complete", choices=["complete", "single", "average"]
    )
    parser.add_argument("--n_inter_samples", type=int, default=256)
    parser.add_argument("--h_bins", type=int, default=10)
    parser.add_argument("--max_pairs", type=int, default=400_000)
    parser.add_argument("--with_partition", action="store_true")
    parser.add_argument("--save_tsne", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_random_init", action="store_true")


def run_nusc(args, device):
    _require_checkpoint(args.pretrained_r, "--pretrained_r")
    seqs = [s.strip().lower() for s in args.seq.split(",") if s.strip()]

    model_rand = None
    if not args.skip_random_init:
        model_rand = LRCrossPolar.create(weights=ResNet18_Weights.DEFAULT).to(device)

    model_pt = LRCrossPolar.create(weights=ResNet18_Weights.DEFAULT).to(device)
    load_pretrained_radar(model_pt, args.pretrained_r)

    all_results = {}
    for seq in seqs:
        print(f"\n{'#' * 60}\n#  Sequence: {seq.upper()}\n{'#' * 60}")
        params = override_nusc_test_params(ModelParamsFull(), seq)
        loader, dataset = make_dataloader_nusc("test", params)
        print(f"  {seq.upper()} test split: {dataset.num_db} database + "
              f"{dataset.num_query} query scans")

        seq_results = {}
        if model_rand is not None:
            seq_results["random_init"] = evaluate_stage(
                f"[{seq.upper()}] Random init.", model_rand, loader, dataset, device, args)
        seq_results["after_pretrain"] = evaluate_stage(
            f"[{seq.upper()}] After pre-training", model_pt, loader, dataset, device, args)
        all_results[seq] = seq_results

    del model_pt
    if model_rand is not None:
        del model_rand
    torch.cuda.empty_cache()

    save_results(all_results, args.out_pickle)
    _print_nusc_tables(all_results, seqs, args)


def run_snail(args, device):
    _require_checkpoint(args.pretrained_r, "--pretrained_r")

    model_rand = None
    if not args.skip_random_init:
        model_rand = LRCrossCart.create(weights=ResNet18_Weights.DEFAULT).to(device)

    model_pt = LRCrossCart.create(weights=ResNet18_Weights.DEFAULT).to(device)
    load_pretrained_radar(model_pt, args.pretrained_r)

    params = ModelParamsFull()
    loader, dataset = make_dataloader_snail("test", params)
    print(f"\nSnail-Radar test split: {dataset.num_db} database + "
          f"{dataset.num_query} query scans")

    all_results = {}
    if model_rand is not None:
        all_results["random_init"] = evaluate_stage(
            "Random init.", model_rand, loader, dataset, device, args)
    all_results["after_pretrain"] = evaluate_stage(
        "After pre-training", model_pt, loader, dataset, device, args)

    del model_pt
    if model_rand is not None:
        del model_rand
    torch.cuda.empty_cache()

    save_results(all_results, args.out_pickle)
    _print_single_split_tables(all_results, "Snail-Radar test split", args)


def run_mulran_boreas(args, device):
    mulran_seqs = (
        [s.strip().lower() for s in args.mulran_seq.split(",") if s.strip()]
        if not args.skip_mulran else []
    )
    boreas_seqs = (
        [s.strip().lower() for s in args.boreas_seq.split(",") if s.strip()]
        if not args.skip_boreas else []
    )
    if not mulran_seqs and not boreas_seqs:
        raise ValueError("Nothing to evaluate: enable MulRan and/or Boreas sequences.")

    model_rand = None
    if not args.skip_random_init:
        model_rand = LRCrossPolar.create(weights=ResNet18_Weights.DEFAULT).to(device)

    model_pt_mulran = None
    if mulran_seqs:
        _require_checkpoint(args.pretrained_r_mulran, "--pretrained_r_mulran")
        model_pt_mulran = LRCrossPolar.create(weights=ResNet18_Weights.DEFAULT).to(device)
        load_pretrained_radar(model_pt_mulran, args.pretrained_r_mulran)

    model_pt_boreas = None
    if boreas_seqs:
        _require_checkpoint(args.pretrained_r_boreas, "--pretrained_r_boreas")
        if args.pretrained_r_boreas == args.pretrained_r_mulran and model_pt_mulran is not None:
            model_pt_boreas = model_pt_mulran
        else:
            model_pt_boreas = LRCrossPolar.create(weights=ResNet18_Weights.DEFAULT).to(device)
            load_pretrained_radar(model_pt_boreas, args.pretrained_r_boreas)

    all_results = {}
    for seq in mulran_seqs:
        key = f"mulran_{seq}"
        print(f"\n{'#' * 60}\n#  MulRan — {seq.upper()}\n{'#' * 60}")
        params = override_mulran_test_params(ModelParamsFull(), seq)
        loader, dataset = make_dataloader_mulran("test", params)
        print(f"  {seq.upper()} test split: {dataset.num_db} database + "
              f"{dataset.num_query} query scans")

        seq_results = {}
        if model_rand is not None:
            seq_results["random_init"] = evaluate_stage(
                f"[MulRan-{seq.upper()}] Random init.",
                model_rand, loader, dataset, device, args)
        seq_results["after_pretrain"] = evaluate_stage(
            f"[MulRan-{seq.upper()}] After pre-training",
            model_pt_mulran, loader, dataset, device, args)
        all_results[key] = seq_results

    for seq in boreas_seqs:
        key = f"boreas_{seq}"
        print(f"\n{'#' * 60}\n#  Boreas — {seq}\n{'#' * 60}")
        params = override_boreas_test_params(ModelParamsFull(), seq)
        loader, dataset = make_dataloader_boreas("test", params)
        print(f"  {seq} test split: {dataset.num_db} database + "
              f"{dataset.num_query} query scans")

        seq_results = {}
        if model_rand is not None:
            seq_results["random_init"] = evaluate_stage(
                f"[Boreas-{seq}] Random init.",
                model_rand, loader, dataset, device, args)
        seq_results["after_pretrain"] = evaluate_stage(
            f"[Boreas-{seq}] After pre-training",
            model_pt_boreas, loader, dataset, device, args)
        all_results[key] = seq_results

    for model in (model_pt_mulran, model_pt_boreas, model_rand):
        if model is not None:
            del model
    torch.cuda.empty_cache()

    save_results(all_results, args.out_pickle)
    _print_mulran_boreas_tables(all_results, mulran_seqs, boreas_seqs, args)


def _print_nusc_tables(all_results, seqs, args):
    pair_col_h = ["H(R)", "Silhouette*", "Fisher J*", "rho*", "AR@1"]
    part_col_h = ["H(R)", "Silhouette", "Fisher J", "rho", "AR@1"]
    for stage_key, stage_name in [("random_init", "Random init."),
                                  ("after_pretrain", "After pre-training")]:
        pair_rows = [(f"  {seq.upper()}  {stage_name}",
                      all_results.get(seq, {}).get(stage_key)) for seq in seqs]
        valid = [all_results[s][stage_key] for s in seqs if stage_key in all_results.get(s, {})]
        avg_pair = mean_results(valid, PAIR_KEYS) if valid else None
        print_table(
            f"Table R1 — pair-based  [{stage_name}]",
            f"  pair-wise r_pos<{args.pos_radius:.0f}m vs r_neg>{args.neg_radius:.0f}m  (no partition)",
            pair_col_h, PAIR_KEYS, pair_rows, [(f"  AVG  {stage_name}", avg_pair)],
        )
        if args.with_partition:
            part_rows = [(f"  {seq.upper()}  {stage_name}",
                          all_results.get(seq, {}).get(stage_key)) for seq in seqs]
            avg_part = mean_results(valid, PART_KEYS) if valid else None
            print_table(
                f"Table R1 suppl. — partition-based  [{stage_name}]",
                f"  linkage={args.linkage}  r_pos={args.pos_radius:.0f}m",
                part_col_h, PART_KEYS, part_rows, [(f"  AVG  {stage_name}", avg_part)],
            )


def _print_single_split_tables(all_results, title, args):
    pair_col_h = ["H(R)", "Silhouette*", "Fisher J*", "rho*", "AR@1"]
    part_col_h = ["H(R)", "Silhouette", "Fisher J", "rho", "AR@1"]
    stage_defs = [("random_init", "Random init."), ("after_pretrain", "After pre-training")]
    pair_rows = [(name, all_results.get(key)) for key, name in stage_defs]
    print_table(
        "Table R1 — primary: pair-based",
        f"  pair-wise r_pos<{args.pos_radius:.0f}m vs r_neg>{args.neg_radius:.0f}m  (no partition)",
        pair_col_h, PAIR_KEYS, pair_rows,
    )
    if args.with_partition:
        part_rows = [(name, all_results.get(key)) for key, name in stage_defs]
        print_table(
            "Table R1 suppl. — partition-based",
            f"  linkage={args.linkage}  r_pos={args.pos_radius:.0f}m",
            part_col_h, PART_KEYS, part_rows,
        )
    print(f"\n{'=' * 68}\nSUMMARY  —  {title}\n{'=' * 68}")


def _print_mulran_boreas_tables(all_results, mulran_seqs, boreas_seqs, args):
    pair_col_h = ["H(R)", "Silhouette*", "Fisher J*", "rho*", "AR@1"]
    part_col_h = ["H(R)", "Silhouette", "Fisher J", "rho", "AR@1"]
    mulran_keys = [f"mulran_{s}" for s in mulran_seqs]
    boreas_keys = [f"boreas_{s}" for s in boreas_seqs]
    all_keys = mulran_keys + boreas_keys

    for stage_key, stage_name in [("random_init", "Random init."),
                                  ("after_pretrain", "After pre-training")]:
        pair_rows = []
        for key in all_keys:
            tag = key.replace("mulran_", "MulRan-").replace("boreas_", "Boreas-")
            pair_rows.append((f"  {tag}  {stage_name}", all_results.get(key, {}).get(stage_key)))
        avg_rows = []
        for group_keys, group_label in [
            (mulran_keys, "AVG MulRan"),
            (boreas_keys, "AVG Boreas"),
            (all_keys, "AVG ALL"),
        ]:
            valid = [all_results[k][stage_key] for k in group_keys if stage_key in all_results.get(k, {})]
            if valid:
                avg_rows.append((f"  {group_label}  {stage_name}", mean_results(valid, PAIR_KEYS)))
        print_table(
            f"Table R1 — pair-based  [{stage_name}]",
            f"  pair-wise r_pos<{args.pos_radius:.0f}m vs r_neg>{args.neg_radius:.0f}m  (no partition)",
            pair_col_h, PAIR_KEYS, pair_rows, avg_rows,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Radar feature separability analysis (Table R1)."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["nusc", "snail", "mulran_boreas"],
        help="Dataset target: nusc | snail | mulran_boreas",
    )
    _add_metric_args(parser)

    parser.add_argument("--pretrained_r", type=str, default="",
                        help="Unimodal-R checkpoint (nusc / snail).")
    parser.add_argument("--seq", type=str, default="bs,son,sq",
                        help="NuScenes splits (nusc only).")
    parser.add_argument("--pretrained_r_mulran", type=str, default="",
                        help="Unimodal-R MulRan checkpoint (mulran_boreas).")
    parser.add_argument("--pretrained_r_boreas", type=str, default="",
                        help="Unimodal-R Boreas checkpoint (mulran_boreas).")
    parser.add_argument("--mulran_seq", type=str, default="sejong,riverside,dcc")
    parser.add_argument("--boreas_seq", type=str, default="bor-snow,bor-clear")
    parser.add_argument("--skip_mulran", action="store_true")
    parser.add_argument("--skip_boreas", action="store_true")
    parser.add_argument("--out_pickle", type=str, default="")

    args = parser.parse_args()
    if not args.out_pickle:
        defaults = {
            "nusc": "./results/r_separability_nusc.pkl",
            "snail": "./results/r_separability_snail.pkl",
            "mulran_boreas": "./results/r_separability_mulran_boreas.pkl",
        }
        args.out_pickle = defaults[args.dataset]

    set_global_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "nusc":
        run_nusc(args, device)
    elif args.dataset == "snail":
        run_snail(args, device)
    else:
        run_mulran_boreas(args, device)


if __name__ == "__main__":
    main()
