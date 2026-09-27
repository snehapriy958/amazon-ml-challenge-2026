"""Benchmark Retrieval-Rank Prefilter Levels for V3 Full Test Inference Optimization.

CRITICAL CONSTRAINTS:
  - Uses ONLY the FROZEN V3 winning pipeline:
      Model: C_LightGBM_LambdaRank (v3_model_c_lambdarank.joblib)
      Features: 88 V3-B features
      Calibration: Platt calibration
      Policy: Entity Policy 5 (t_us=0.995, t_in=0.995, max_k=6)
  - Does NOT retrain the model, change features, change thresholds, or change Policy 5.
  - Tests candidate retention at:
      Top-50, Top-75, Top-100, Top-125, Top-150 (plus Top-200, Top-250, Top-300, Top-500, Top-750)
  - ALWAYS retains:
      1. All Channel-1 exact candidates (channel_mask & 1 != 0)
      2. Candidates with exact normalized name or address match
      3. Fair S2/S3 interleaving (min_per_source = min(160, K // 3))
  - Evaluates on INTERNAL_CHECK_1000 and FINAL_HOLDOUT_1000 (plus DEV_3000) against the frozen Top-750 baseline.
"""

import heapq
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from code.business_entity_resolution.src.preprocessing import parse_match_ids
from experiments.v3.src.run_v3_d_bakeoff import (
    apply_frozen_policy,
    evaluate_predictions_from_mask,
    prepare_eval_structures,
)

TEST_S1_COUNT = 1_732_544
PREFILTER_LEVELS = [50, 75, 100, 125, 150, 200, 250, 300, 500, 750]


def select_prefilter_mask_for_pool(
    feat_df: pd.DataFrame,
    s1_slices: List[Tuple[int, int]],
    top_k: int,
) -> np.ndarray:
    """Select the exact Top-K retrieval prefilter mask per S1 query.

    Always preserves:
      - All Channel-1 exact candidates (channel_mask & 1 != 0)
      - All candidates with exact_orig_name, exact_norm_name, exact_core_name,
        exact_compact_name, exact_orig_addr, or exact_norm_addr == 1
      - Fair S2/S3 quota: min(160, top_k // 3) per source
      - Remaining slots up to top_k filled by highest retrieval_score
    """
    n_rows = len(feat_df)
    if top_k >= 750:
        return np.ones(n_rows, dtype=bool)

    keep_mask = np.zeros(n_rows, dtype=bool)
    ret_scores = feat_df["retrieval_score"].to_numpy(dtype=np.float64)
    ch_masks = feat_df["channel_mask"].to_numpy(dtype=np.int32)
    cand_ids = feat_df["candidate_entity_id"].tolist()
    is_s2_arr = np.array([cid.startswith("S2-") for cid in cand_ids], dtype=bool)

    exact_feature_hit = (
        (feat_df["exact_orig_name"].to_numpy() > 0.5)
        | (feat_df["exact_norm_name"].to_numpy() > 0.5)
        | (feat_df["exact_core_name"].to_numpy() > 0.5)
        | (feat_df["exact_compact_name"].to_numpy() > 0.5)
        | (feat_df["exact_orig_addr"].to_numpy() > 0.5)
        | (feat_df["exact_norm_addr"].to_numpy() > 0.5)
    )
    is_exact_ch1 = ((ch_masks & 1) != 0) | exact_feature_hit

    min_per_source = min(160, top_k // 3)

    for st, en in s1_slices:
        if st == en:
            continue
        if (en - st) <= top_k:
            keep_mask[st:en] = True
            continue

        selected_local = set()

        # 1. Always preserve all Channel-1 / exact normalized name/address candidates
        for idx in range(st, en):
            if is_exact_ch1[idx]:
                selected_local.add(idx)

        # 2. Fair S2/S3 interleaving quota by retrieval_score
        s2_cands = []
        s3_cands = []
        for idx in range(st, en):
            if idx in selected_local:
                continue
            sc = ret_scores[idx]
            if is_s2_arr[idx]:
                s2_cands.append((sc, idx))
            else:
                s3_cands.append((sc, idx))

        top_s2 = heapq.nlargest(min_per_source, s2_cands)
        top_s3 = heapq.nlargest(min_per_source, s3_cands)

        for sc, idx in top_s2:
            if len(selected_local) < top_k:
                selected_local.add(idx)
        for sc, idx in top_s3:
            if len(selected_local) < top_k:
                selected_local.add(idx)

        # 3. Fill remaining slots up to top_k by global retrieval_score
        rem_slots = top_k - len(selected_local)
        if rem_slots > 0:
            rem_pool = [item for item in (s2_cands + s3_cands) if item[1] not in selected_local]
            for sc, idx in heapq.nlargest(rem_slots, rem_pool):
                selected_local.add(idx)

        for idx in selected_local:
            keep_mask[idx] = True

    return keep_mask


def evaluate_prefilter_level(
    pool_name: str,
    qdf: pd.DataFrame,
    feat_df: pd.DataFrame,
    full_probs: np.ndarray,
    frozen_policy_cfg: dict,
    base_thr: float,
    top_k: int,
    single_core_pairs_per_sec: float,
    num_workers: int,
    index_plus_union_test_hours: float,
) -> dict:
    """Evaluate a single Top-K prefilter level on a validation/holdout pool using the frozen model & policy."""
    eval_struct_full = prepare_eval_structures(qdf, feat_df)
    keep_mask = select_prefilter_mask_for_pool(feat_df, eval_struct_full["s1_row_slices"], top_k)

    sub_df = feat_df[keep_mask].reset_index(drop=True)
    sub_probs = full_probs[keep_mask]
    eval_struct_sub = prepare_eval_structures(qdf, sub_df)

    # Compute candidate recall at this prefilter level
    total_gt = eval_struct_sub["total_gt_links"]
    found_gt = int(eval_struct_sub["label_col"].sum())
    s2_gt = eval_struct_sub["total_s2_gt"]
    s2_found = int(eval_struct_sub["label_col"][eval_struct_sub["is_s2_cand"]].sum())
    s3_gt = eval_struct_sub["total_s3_gt"]
    s3_found = found_gt - s2_found

    entity_recalls_nonempty = []
    cands_per_s1 = []
    for idx, (st, en) in enumerate(eval_struct_sub["s1_row_slices"]):
        cands_per_s1.append(en - st)
        g_len = len(eval_struct_sub["gt_sets"][idx])
        if g_len > 0:
            hits = int(eval_struct_sub["label_col"][st:en].sum()) if en > st else 0
            entity_recalls_nonempty.append(hits / g_len)

    cands_arr = np.array(cands_per_s1, dtype=np.float64)

    # Apply frozen Entity Policy 5 (and frozen threshold-only for reference)
    policy_mask = apply_frozen_policy(sub_probs, sub_df, eval_struct_sub, frozen_policy_cfg)
    policy_metrics = evaluate_predictions_from_mask(policy_mask, eval_struct_sub, f"Top-{top_k}_Policy5")
    thr_metrics = evaluate_predictions_from_mask(sub_probs >= base_thr, eval_struct_sub, f"Top-{top_k}_Thr{base_thr}")

    mean_cands = float(np.mean(cands_arr))
    projected_test_feature_rows = int(round(mean_cands * TEST_S1_COUNT))
    feat_single_core_hours = projected_test_feature_rows / max(1.0, single_core_pairs_per_sec) / 3600.0
    feat_parallel_hours = feat_single_core_hours / max(1.0, num_workers * 0.78)
    total_parallel_test_hours = index_plus_union_test_hours + feat_parallel_hours

    return {
        "pool_name": pool_name,
        "top_k": int(top_k),
        "candidate_recall_micro": float(found_gt / max(1, total_gt)),
        "candidate_recall_macro_nonempty": float(np.mean(entity_recalls_nonempty)),
        "s2_candidate_recall": float(s2_found / max(1, s2_gt)),
        "s3_candidate_recall": float(s3_found / max(1, s3_gt)),
        "found_gt_links": int(found_gt),
        "total_gt_links": int(total_gt),
        "missed_gt_links": int(total_gt - found_gt),
        "candidates_per_s1": {
            "mean": mean_cands,
            "p50": float(np.percentile(cands_arr, 50)),
            "p95": float(np.percentile(cands_arr, 95)),
            "max": int(np.max(cands_arr)),
        },
        "pool_feature_rows": int(len(sub_df)),
        "projected_test_feature_rows": projected_test_feature_rows,
        "projected_test_runtime": {
            "single_core_feature_hours": round(feat_single_core_hours, 2),
            "parallel_feature_hours": round(feat_parallel_hours, 2),
            "index_and_retrieval_hours": round(index_plus_union_test_hours, 2),
            "total_parallel_hours": round(total_parallel_test_hours, 2),
            "parallel_workers": int(num_workers),
        },
        "frozen_policy5_metrics": policy_metrics,
        "frozen_threshold_only_metrics": thr_metrics,
    }


def main():
    t0 = time.time()
    print("=" * 88, flush=True)
    print("BENCHMARKING RETRIEVAL-RANK PREFILTER LEVELS WITH FROZEN V3 MODEL & POLICY", flush=True)
    print("=" * 88, flush=True)

    res_dir = Path("experiments/v3/results")
    data_dir = Path("experiments/v3/data")
    models_dir = Path("experiments/v3/models")

    with open(res_dir / "v3_d_bakeoff_metrics.json", "r", encoding="utf-8") as f:
        bakeoff_data = json.load(f)

    frozen_cfg = bakeoff_data["selected_frozen_pipeline"]
    active_cols = frozen_cfg["active_features"]
    base_thr = float(frozen_cfg["base_threshold"])
    policy_cfg = frozen_cfg["policy_config"]

    cpu_count = os.cpu_count() or 8
    num_workers = max(4, min(12, cpu_count - 2))
    # Measured throughput from task-2080:
    # - Feature computation single-core throughput: ~4,550 pairs/sec (600,000 pairs in 129.2s)
    # - Parallelized retrieval + indexing across workers: ~1.85 hours for 1.73M S1 queries
    single_core_pairs_per_sec = 4550.0
    index_plus_union_test_hours = 1.85

    print(f"Detected {cpu_count} logical CPUs -> parallel worker plan uses {num_workers} workers.", flush=True)
    print(f"Loading frozen C_LightGBM_LambdaRank model + Platt calibrator...", flush=True)
    c_bundle = joblib.load(models_dir / "v3_model_c_lambdarank.joblib")
    ranker = c_bundle["ranker"]
    calibrator = c_bundle["calibrator"]

    pools = [
        ("INTERNAL_CHECK_1000", "experiments/v3/pools/internal_check_1000_s1.tsv", data_dir / "internal_check_1000_v3_features.parquet"),
        ("FINAL_HOLDOUT_1000", "experiments/v3/pools/final_holdout_1000_s1.tsv", data_dir / "final_holdout_1000_v3_features.parquet"),
        ("DEV_3000", "experiments/v3/pools/dev_3000_s1.tsv", data_dir / "dev_3000_v3_features.parquet"),
    ]

    all_results = {
        "frozen_pipeline_config": frozen_cfg,
        "hardware_and_throughput": {
            "logical_cpus": cpu_count,
            "recommended_workers": num_workers,
            "measured_single_core_feature_pairs_per_sec": single_core_pairs_per_sec,
            "projected_parallel_index_and_retrieval_hours": index_plus_union_test_hours,
        },
        "pools": {},
    }

    for pool_name, q_path, f_path in pools:
        print(f"\n--- Benchmarking Prefilter Levels on {pool_name} ---", flush=True)
        qdf = pd.read_csv(q_path, sep="\t", dtype=str)
        feat_df = pd.read_parquet(f_path)
        raw_scores = ranker.predict(feat_df[active_cols])
        full_probs = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

        pool_res = {}
        for k in PREFILTER_LEVELS:
            r = evaluate_prefilter_level(
                pool_name,
                qdf,
                feat_df,
                full_probs,
                policy_cfg,
                base_thr,
                k,
                single_core_pairs_per_sec,
                num_workers,
                index_plus_union_test_hours,
            )
            pool_res[f"Top-{k}"] = r
            m = r["frozen_policy5_metrics"]
            print(
                f"  [Top-{k:3d}] CandRec={r['candidate_recall_micro']*100:5.2f}% "
                f"(S2={r['s2_candidate_recall']*100:5.2f}%, S3={r['s3_candidate_recall']*100:5.2f}%) | "
                f"F0.5={m['macro_f05']:.4f} | Prec={m['macro_precision']*100:5.2f}% | Rec={m['macro_recall']*100:5.2f}% | "
                f"TP={m['tp']:4d} FP={m['fp']:3d} FN={m['fn']:4d} | "
                f"Cands/S1={r['candidates_per_s1']['mean']:6.2f} | "
                f"TestRows={r['projected_test_feature_rows']/1e6:6.1f}M | "
                f"ProjTime={r['projected_test_runtime']['total_parallel_hours']:4.1f}h ({num_workers}w)",
                flush=True,
            )
        all_results["pools"][pool_name] = pool_res

    out_path = res_dir / "v3_prefilter_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved prefilter benchmark metrics to {out_path} in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
