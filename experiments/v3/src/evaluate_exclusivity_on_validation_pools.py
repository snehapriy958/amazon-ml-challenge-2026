"""Evaluate the EXACT V3 Target-Exclusivity Algorithm on Labeled Validation Pools.

Pools evaluated:
  1. FINAL_HOLDOUT_1000
  2. INTERNAL_CHECK_1000
  3. DEV_3000
  (plus COMBINED_5000 = union of all three disjoint pools)

Imports `_extract_record_features` and `_score_owner_evidence` directly from
`experiments/v3/src/run_v3_exclusivity_experiment.py` so the conflict-resolution rule
is 100% identical to the test experiment.
"""

from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from code.business_entity_resolution.src.evaluation import compute_entity_f_beta
from code.business_entity_resolution.src.preprocessing import parse_match_ids
from experiments.v3.src.benchmark_test_prefilter import select_prefilter_mask_for_pool
from experiments.v3.src.run_v3_d_bakeoff import (
    apply_frozen_policy,
    prepare_eval_structures,
)
from experiments.v3.src.run_v3_exclusivity_experiment import (
    _extract_record_features,
    _score_owner_evidence,
)


def evaluate_pool_with_exclusivity(
    pool_name: str,
    qdf: pd.DataFrame,
    feat_df: pd.DataFrame,
    probs: np.ndarray,
    policy_cfg: dict,
    target_raw_lookup: Dict[str, Dict[str, str]],
    top_k: int = 500,
) -> dict:
    eval_full = prepare_eval_structures(qdf, feat_df)
    if top_k < 750:
        keep_mask = select_prefilter_mask_for_pool(feat_df, eval_full["s1_row_slices"], top_k)
        sub_df = feat_df[keep_mask].reset_index(drop=True)
        sub_probs = probs[keep_mask]
    else:
        sub_df = feat_df
        sub_probs = probs

    eval_struct = prepare_eval_structures(qdf, sub_df)
    policy_mask = apply_frozen_policy(sub_probs, sub_df, eval_struct, policy_cfg)

    cand_col = sub_df["candidate_entity_id"].tolist()
    s1_ids = qdf["source1_entity_id"].tolist()
    gt_sets: List[Set[str]] = eval_struct["gt_sets"]
    slices: List[Tuple[int, int]] = eval_struct["s1_row_slices"]

    # Build S1 ordered predictions exactly as written to matching_results.tsv (sorted by prob desc)
    s1_preds_base: List[Tuple[str, List[str], Set[str]]] = []
    target_claims = defaultdict(list)  # tid -> list of (s1_id, rank_idx, s1_idx)

    for s1_idx, (st, en) in enumerate(slices):
        s1_id = s1_ids[s1_idx]
        gt_set = gt_sets[s1_idx]
        if en <= st:
            s1_preds_base.append((s1_id, [], gt_set))
            continue
        loc_mask = policy_mask[st:en]
        if not np.any(loc_mask):
            s1_preds_base.append((s1_id, [], gt_set))
            continue
        loc_indices = np.flatnonzero(loc_mask)
        # Sort selected candidates by descending calibrated probability (exact production order)
        loc_probs = sub_probs[st:en][loc_indices]
        order = np.argsort(-loc_probs, kind="mergesort")
        ordered_mids = [cand_col[st + int(loc_indices[i])] for i in order]
        s1_preds_base.append((s1_id, ordered_mids, gt_set))
        for rank_idx, tid in enumerate(ordered_mids):
            target_claims[tid].append((s1_id, rank_idx, s1_idx))

    # Identify target conflicts (>1 S1 claiming same target)
    conflicted_targets = {tid: claims for tid, claims in target_claims.items() if len(claims) > 1}
    conflicted_s1_ids = {s1_id for claims in conflicted_targets.values() for s1_id, _, _ in claims}

    # Extract raw record features for conflicted S1 and target IDs using exact test function
    s1_rec_feats = {}
    for row in qdf.itertuples(index=False):
        sid = row.source1_entity_id
        if sid in conflicted_s1_ids:
            s1_rec_feats[sid] = _extract_record_features(
                "" if pd.isna(row.business_name) else str(row.business_name),
                "" if pd.isna(row.business_address) else str(row.business_address),
                "" if pd.isna(row.country) else str(row.country),
            )

    tgt_rec_feats = {}
    for tid in conflicted_targets:
        t_raw = target_raw_lookup.get(tid, {})
        tgt_rec_feats[tid] = _extract_record_features(
            t_raw.get("business_name", ""),
            t_raw.get("business_address", ""),
            t_raw.get("country", ""),
        )

    # Resolve each conflicted target using exact _score_owner_evidence
    winning_owner = {}
    for tid, claims in conflicted_targets.items():
        t_rec = tgt_rec_feats[tid]
        best_s1 = None
        best_key = None
        for s1_id, rank_idx, _ in claims:
            s_rec = s1_rec_feats[s1_id]
            score_key = _score_owner_evidence(s_rec, t_rec, rank_idx, s1_id)
            if best_key is None or score_key > best_key:
                best_key = score_key
                best_s1 = s1_id
        winning_owner[tid] = best_s1

    # Compute baseline V3 metrics and V3 + exclusivity metrics
    def _compute_metrics_and_audit(use_exclusivity: bool):
        f05_list = []
        prec_list = []
        rec_list = []
        tp_total = 0
        fp_total = 0
        fn_total = 0
        pred_links_total = 0
        empty_s1_count = 0

        removed_count = 0
        removed_gt_pos = 0
        removed_gt_neg = 0
        retained_gt_pos_all = 0
        retained_gt_pos_on_conflicts = 0
        retained_gt_neg_on_conflicts = 0

        for s1_id, mids, gt_set in s1_preds_base:
            if use_exclusivity:
                kept = []
                for tid in mids:
                    if tid not in winning_owner or winning_owner[tid] == s1_id:
                        kept.append(tid)
                        if tid in gt_set:
                            retained_gt_pos_all += 1
                            if tid in winning_owner:
                                retained_gt_pos_on_conflicts += 1
                        else:
                            if tid in winning_owner:
                                retained_gt_neg_on_conflicts += 1
                    else:
                        removed_count += 1
                        if tid in gt_set:
                            removed_gt_pos += 1
                        else:
                            removed_gt_neg += 1
            else:
                kept = mids
                for tid in kept:
                    if tid in gt_set:
                        retained_gt_pos_all += 1

            pred_set = set(kept)
            pred_links_total += len(pred_set)
            if not pred_set:
                empty_s1_count += 1

            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            tp_total += tp
            fp_total += fp
            fn_total += fn

            if len(gt_set) == 0:
                if len(pred_set) == 0:
                    p_e, r_e, f_e = 1.0, 1.0, 1.0
                else:
                    p_e, r_e, f_e = 0.0, 0.0, 0.0
            else:
                if len(pred_set) == 0:
                    p_e, r_e, f_e = 0.0, 0.0, 0.0
                else:
                    p_e = tp / len(pred_set)
                    r_e = tp / len(gt_set)
                    f_e = (1.25 * p_e * r_e) / (0.25 * p_e + r_e) if (p_e + r_e) > 0 else 0.0
            prec_list.append(p_e)
            rec_list.append(r_e)
            f05_list.append(f_e)

        return {
            "macro_f05": round(float(np.mean(f05_list)), 6),
            "macro_precision": round(float(np.mean(prec_list)), 6),
            "macro_recall": round(float(np.mean(rec_list)), 6),
            "TP": int(tp_total),
            "FP": int(fp_total),
            "FN": int(fn_total),
            "predicted_links": int(pred_links_total),
            "empty_s1": int(empty_s1_count),
            "non_empty_s1": int(len(s1_preds_base) - empty_s1_count),
            "removed_links": int(removed_count),
            "removed_gt_positives": int(removed_gt_pos),
            "removed_gt_negatives": int(removed_gt_neg),
            "retained_gt_positives_total": int(retained_gt_pos_all),
            "retained_gt_positives_on_conflicts": int(retained_gt_pos_on_conflicts),
            "retained_gt_negatives_on_conflicts": int(retained_gt_neg_on_conflicts),
        }

    base_res = _compute_metrics_and_audit(use_exclusivity=False)
    excl_res = _compute_metrics_and_audit(use_exclusivity=True)

    return {
        "pool_name": pool_name,
        "top_k": top_k,
        "num_s1": len(s1_preds_base),
        "target_conflicts_before": len(conflicted_targets),
        "excess_assignments_removed": excl_res["removed_links"],
        "removed_links_gt_positives": excl_res["removed_gt_positives"],
        "removed_links_gt_negatives": excl_res["removed_gt_negatives"],
        "retained_links_gt_positives_total": excl_res["retained_gt_positives_total"],
        "retained_links_gt_positives_on_conflicted_targets": excl_res["retained_gt_positives_on_conflicts"],
        "retained_links_gt_negatives_on_conflicted_targets": excl_res["retained_gt_negatives_on_conflicts"],
        "delta_macro_f05": round(excl_res["macro_f05"] - base_res["macro_f05"], 6),
        "baseline_v3": base_res,
        "v3_plus_exclusivity": excl_res,
    }


def main():
    t0 = time.time()
    res_dir = Path("experiments/v3/results")
    data_dir = Path("experiments/v3/data")
    models_dir = Path("experiments/v3/models")
    ret_dir = Path("experiments/v3/retrieval")

    with open(res_dir / "v3_d_bakeoff_metrics.json", "r", encoding="utf-8") as f:
        bakeoff_data = json.load(f)

    frozen_cfg = bakeoff_data["selected_frozen_pipeline"]
    active_cols = frozen_cfg["active_features"]
    policy_cfg = frozen_cfg["policy_config"]

    print("Loading frozen C_LightGBM_LambdaRank model & Platt calibrator...")
    c_bundle = joblib.load(models_dir / "v3_model_c_lambdarank.joblib")
    ranker = c_bundle["ranker"]
    calibrator = c_bundle["calibrator"]

    print("Loading cached target records from retrieval bundles...")
    b_2000 = joblib.load(ret_dir / "check_and_holdout_2000_retrieval_bundle.joblib")
    b_3000 = joblib.load(ret_dir / "dev_3000_retrieval_bundle.joblib")
    target_raw_lookup = {}
    target_raw_lookup.update(b_3000.get("target_records_all") or b_3000.get("target_records") or {})
    target_raw_lookup.update(b_2000.get("target_records_all") or b_2000.get("target_records") or {})

    pools = [
        ("FINAL_HOLDOUT_1000", "experiments/v3/pools/final_holdout_1000_s1.tsv", data_dir / "final_holdout_1000_v3_features.parquet"),
        ("INTERNAL_CHECK_1000", "experiments/v3/pools/internal_check_1000_s1.tsv", data_dir / "internal_check_1000_v3_features.parquet"),
        ("DEV_3000", "experiments/v3/pools/dev_3000_s1.tsv", data_dir / "dev_3000_v3_features.parquet"),
    ]

    results_top500 = {}
    results_top750 = {}
    qdfs_all = []
    fdfs_all = []
    probs_all = []

    for p_name, q_path, f_path in pools:
        print(f"Scoring {p_name}...")
        qdf = pd.read_csv(q_path, sep="\t", dtype=str)
        fdf = pd.read_parquet(f_path)
        raw_s = ranker.predict(fdf[active_cols])
        probs = calibrator.predict_proba(raw_s.reshape(-1, 1))[:, 1]

        results_top500[p_name] = evaluate_pool_with_exclusivity(
            p_name, qdf, fdf, probs, policy_cfg, target_raw_lookup, top_k=500
        )
        results_top750[p_name] = evaluate_pool_with_exclusivity(
            p_name, qdf, fdf, probs, policy_cfg, target_raw_lookup, top_k=750
        )
        qdfs_all.append(qdf)
        fdfs_all.append(fdf)
        probs_all.append(probs)

    # Also evaluate on COMBINED_5000 (union of all 3 disjoint validation pools)
    comb_qdf = pd.concat(qdfs_all, ignore_index=True)
    comb_fdf = pd.concat(fdfs_all, ignore_index=True)
    comb_probs = np.concatenate(probs_all)
    results_top500["COMBINED_5000"] = evaluate_pool_with_exclusivity(
        "COMBINED_5000", comb_qdf, comb_fdf, comb_probs, policy_cfg, target_raw_lookup, top_k=500
    )
    results_top750["COMBINED_5000"] = evaluate_pool_with_exclusivity(
        "COMBINED_5000", comb_qdf, comb_fdf, comb_probs, policy_cfg, target_raw_lookup, top_k=750
    )

    out_summary = {
        "top_500_production_config": results_top500,
        "top_750_full_config": results_top750,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_path = res_dir / "v4_exclusivity_validation_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)

    print("\n" + "=" * 95)
    print("VALIDATION AUDIT RESULTS (TOP-500 PRODUCTION CONFIG & TOP-750)")
    print("=" * 95)
    print(json.dumps(out_summary, indent=2))


if __name__ == "__main__":
    main()
