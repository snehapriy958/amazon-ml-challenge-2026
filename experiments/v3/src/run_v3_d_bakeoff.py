"""Phase V3-D: Controlled 4-Model Bakeoff & Entity-Level Decision Evaluation on DEV_3000.

Compares on the exact same V3-C training population and feature set:
  0. Reference: Frozen V2-C XGBoost (11 locked features) on DEV_3000 Top-750
  1. Model A: V3 XGBoost Binary Classifier
  2. Model B: V3 LightGBM Binary Classifier
  3. Model C: V3 LightGBM LambdaRank (grouped by source1_entity_id)
  4. Model D: V3 LightGBM rank_xendcg (grouped by source1_entity_id)

Also performs:
  - Phonetic feature ablation (85 core features vs 88 core+phonetic features) on DEV_3000
  - Full threshold sweep across [0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.9125, 0.925, 0.95, 0.975, 0.99]
  - Entity-level decision policies (Policies 1..5 + Per-Country US/India calibration) on DEV_3000
  - Saves all models, DEV predictions, and machine-readable metrics to experiments/v3/.
"""

import gc
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Set, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.insert(0, str(Path.cwd()))
from code.business_entity_resolution.src.feature_engineering import LOCKED_FEATURE_COLUMNS
from code.business_entity_resolution.src.preprocessing import parse_match_ids
from experiments.v3.src.features_v3 import (
    V3_ALL_FEATURE_COLUMNS,
    V3_CORE_FEATURE_COLUMNS,
    V3_PHONETIC_FEATURES,
)

SWEEP_THRESHOLDS = [
    0.70, 0.75, 0.80, 0.825, 0.85, 0.875,
    0.90, 0.9125, 0.925, 0.95, 0.975, 0.99,
]


def prepare_eval_structures(dev_qdf: pd.DataFrame, dev_feat_df: pd.DataFrame) -> dict:
    """Build fast numpy/indexed structures for rapid threshold & entity-level policy evaluation."""
    s1_Order = dev_qdf["source1_entity_id"].tolist()
    s1_to_idx = {sid: idx for idx, sid in enumerate(s1_Order)}

    gt_sets: List[Set[str]] = []
    countries: List[str] = []
    is_empty_s1 = np.zeros(len(s1_Order), dtype=bool)
    is_single_s1 = np.zeros(len(s1_Order), dtype=bool)
    is_multi_s1 = np.zeros(len(s1_Order), dtype=bool)

    total_s2_gt = 0
    total_s3_gt = 0

    for idx, row in enumerate(dev_qdf.itertuples(index=False)):
        g_set = set(parse_match_ids("" if pd.isna(row.matched_entity_ids) else str(row.matched_entity_ids)))
        gt_sets.append(g_set)
        c_str = "" if pd.isna(row.country) else str(row.country).strip().lower()
        countries.append(c_str)
        k = len(g_set)
        if k == 0:
            is_empty_s1[idx] = True
        elif k == 1:
            is_single_s1[idx] = True
        else:
            is_multi_s1[idx] = True
        for cid in g_set:
            if cid.startswith("S2-"):
                total_s2_gt += 1
            else:
                total_s3_gt += 1

    # Group row indices in dev_feat_df by s1_idx
    s1_col = dev_feat_df["source1_entity_id"].tolist()
    cand_col = dev_feat_df["candidate_entity_id"].tolist()
    label_col = dev_feat_df["label"].to_numpy(dtype=np.int32)
    is_s2_cand = np.array([cid.startswith("S2-") for cid in cand_col], dtype=bool)

    s1_row_slices: List[Tuple[int, int]] = [(0, 0)] * len(s1_Order)
    # Verify dev_feat_df is grouped by source1_entity_id in s1_Order
    cur_sid = None
    start_i = 0
    slice_map = {}
    for i, sid in enumerate(s1_col):
        if sid != cur_sid:
            if cur_sid is not None:
                slice_map[cur_sid] = (start_i, i)
            cur_sid = sid
            start_i = i
    if cur_sid is not None:
        slice_map[cur_sid] = (start_i, len(s1_col))

    for idx, sid in enumerate(s1_Order):
        s1_row_slices[idx] = slice_map.get(sid, (0, 0))

    return {
        "s1_order": s1_Order,
        "gt_sets": gt_sets,
        "countries": np.array(countries, dtype=object),
        "is_empty_s1": is_empty_s1,
        "is_single_s1": is_single_s1,
        "is_multi_s1": is_multi_s1,
        "total_gt_links": int(total_s2_gt + total_s3_gt),
        "total_s2_gt": int(total_s2_gt),
        "total_s3_gt": int(total_s3_gt),
        "cand_col": cand_col,
        "label_col": label_col,
        "is_s2_cand": is_s2_cand,
        "s1_row_slices": s1_row_slices,
    }


def evaluate_predictions_from_mask(
    selected_pair_mask: np.ndarray,
    eval_struct: dict,
    threshold_label: object = 0.90,
) -> dict:
    """Compute exact Macro F0.5, Precision, Recall, S2/S3, Country, and Empty-S1 metrics from a boolean mask over dev_feat_df."""
    n_s1 = len(eval_struct["s1_order"])
    gt_sets = eval_struct["gt_sets"]
    s1_slices = eval_struct["s1_row_slices"]
    label_col = eval_struct["label_col"]
    is_s2_cand = eval_struct["is_s2_cand"]
    countries = eval_struct["countries"]
    is_empty_s1 = eval_struct["is_empty_s1"]

    f05_arr = np.zeros(n_s1, dtype=np.float64)
    prec_arr = np.zeros(n_s1, dtype=np.float64)
    rec_arr = np.zeros(n_s1, dtype=np.float64)
    pred_empty_arr = np.zeros(n_s1, dtype=bool)

    total_tp = 0
    total_fp = 0
    s2_tp = 0
    s2_fp = 0
    s3_tp = 0
    s3_fp = 0

    beta2 = 0.25  # F0.5 -> beta^2 = 0.25

    for idx in range(n_s1):
        st, en = s1_slices[idx]
        g_len = len(gt_sets[idx])
        if st == en:
            n_pred = 0
            tp = 0
        else:
            sel = selected_pair_mask[st:en]
            n_pred = int(sel.sum())
            if n_pred == 0:
                tp = 0
            else:
                lbls = label_col[st:en][sel]
                s2_flags = is_s2_cand[st:en][sel]
                tp = int(lbls.sum())
                fp = n_pred - tp
                total_tp += tp
                total_fp += fp

                tp_s2 = int(lbls[s2_flags].sum())
                fp_s2 = int(s2_flags.sum()) - tp_s2
                s2_tp += tp_s2
                s2_fp += fp_s2
                s3_tp += (tp - tp_s2)
                s3_fp += (fp - fp_s2)

        if n_pred == 0:
            pred_empty_arr[idx] = True

        if g_len == 0 and n_pred == 0:
            f05_arr[idx] = 1.0
            prec_arr[idx] = 1.0
            rec_arr[idx] = 1.0
        elif g_len == 0 or n_pred == 0 or tp == 0:
            f05_arr[idx] = 0.0
            prec_arr[idx] = 0.0
            rec_arr[idx] = 0.0
        else:
            p = tp / n_pred
            r = tp / g_len
            prec_arr[idx] = p
            rec_arr[idx] = r
            f05_arr[idx] = (1.0 + beta2) * p * r / (beta2 * p + r)

    total_fn = eval_struct["total_gt_links"] - total_tp
    s2_fn = eval_struct["total_s2_gt"] - s2_tp
    s3_fn = eval_struct["total_s3_gt"] - s3_tp

    empty_s1_acc = float(f05_arr[is_empty_s1].mean()) if is_empty_s1.any() else 1.0
    singleton_fp_rate = float(1.0 - empty_s1_acc)

    by_country = {}
    for c_val in sorted(set(countries)):
        c_mask = countries == c_val
        by_country[str(c_val)] = {
            "s1_count": int(c_mask.sum()),
            "macro_f05": float(f05_arr[c_mask].mean()),
            "macro_prec": float(prec_arr[c_mask].mean()),
            "macro_rec": float(rec_arr[c_mask].mean()),
        }

    s2_prec = float(s2_tp / max(1, s2_tp + s2_fp))
    s2_rec = float(s2_tp / max(1, eval_struct["total_s2_gt"]))
    s2_f05 = float((1.25 * s2_prec * s2_rec) / max(1e-12, 0.25 * s2_prec + s2_rec))

    s3_prec = float(s3_tp / max(1, s3_tp + s3_fp))
    s3_rec = float(s3_tp / max(1, eval_struct["total_s3_gt"]))
    s3_f05 = float((1.25 * s3_prec * s3_rec) / max(1e-12, 0.25 * s3_prec + s3_rec))

    return {
        "threshold": threshold_label,
        "macro_f05": float(f05_arr.mean()),
        "macro_precision": float(prec_arr.mean()),
        "macro_recall": float(rec_arr.mean()),
        "predicted_pairs": int(total_tp + total_fp),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "predicted_empty_s1_count": int(pred_empty_arr.sum()),
        "empty_s1_accuracy": empty_s1_acc,
        "singleton_fp_rate": singleton_fp_rate,
        "s2_metrics": {
            "tp": int(s2_tp),
            "fp": int(s2_fp),
            "fn": int(s2_fn),
            "micro_prec": s2_prec,
            "micro_rec": s2_rec,
            "micro_f05": s2_f05,
        },
        "s3_metrics": {
            "tp": int(s3_tp),
            "fp": int(s3_fp),
            "fn": int(s3_fn),
            "micro_prec": s3_prec,
            "micro_rec": s3_rec,
            "micro_f05": s3_f05,
        },
        "by_country": by_country,
        "by_cardinality": {
            "empty_s1_f05": empty_s1_acc,
            "single_match_s1_f05": float(f05_arr[eval_struct["is_single_s1"]].mean()),
            "multi_match_s1_f05": float(f05_arr[eval_struct["is_multi_s1"]].mean()),
        },
    }


def summarize_score_distribution(scores: np.ndarray, labels: np.ndarray) -> dict:
    def _quantiles(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {}
        return {
            "min": float(np.min(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

    return {
        "all_pairs": _quantiles(scores),
        "positives_gt1": _quantiles(scores[labels == 1]),
        "negatives_gt0": _quantiles(scores[labels == 0]),
    }


def evaluate_entity_level_policies(
    scores: np.ndarray,
    dev_feat_df: pd.DataFrame,
    eval_struct: dict,
    best_base_thr: float,
) -> dict:
    """Evaluate the 5 required Entity-Level Decision Policies + Per-Country Calibration on DEV_3000."""
    n_rows = len(scores)
    n_s1 = len(eval_struct["s1_order"])
    s1_slices = eval_struct["s1_row_slices"]
    countries = eval_struct["countries"]

    # Extract key evidence columns as fast numpy arrays
    exact_ev = (
        (dev_feat_df["exact_orig_name"].to_numpy() > 0.5)
        | (dev_feat_df["exact_norm_name"].to_numpy() > 0.5)
        | (dev_feat_df["exact_core_name"].to_numpy() > 0.5)
        | (dev_feat_df["exact_norm_addr"].to_numpy() > 0.5)
    )
    bldg_match = dev_feat_df["building_number_match"].to_numpy()
    bldg_conflict = dev_feat_df["building_number_conflict"].to_numpy() > 0.5
    postal_match = dev_feat_df["postal_match"].to_numpy() > 0.5
    postal_conflict = dev_feat_df["postal_conflict"].to_numpy() > 0.5
    core_name_set = dev_feat_df["core_name_token_set"].to_numpy()
    norm_addr_set = dev_feat_df["norm_addr_token_set"].to_numpy()
    high_name_low_addr = dev_feat_df["high_name_low_addr"].to_numpy() > 0.5

    num_agree = (bldg_match >= 0.9) | postal_match
    num_conflict = bldg_conflict | postal_conflict

    # Compute per-S1 rank, top1, top2, top3, and margins
    rank_in_s1 = np.zeros(n_rows, dtype=np.int32)
    top1_per_row = np.zeros(n_rows, dtype=np.float64)
    top2_per_row = np.zeros(n_rows, dtype=np.float64)
    margin_from_top1 = np.zeros(n_rows, dtype=np.float64)
    s1_top1_top2_margin = np.zeros(n_rows, dtype=np.float64)
    s1_country_arr = np.empty(n_rows, dtype=object)

    top1_scores_s1 = np.zeros(n_s1, dtype=np.float64)
    top2_scores_s1 = np.zeros(n_s1, dtype=np.float64)
    top3_scores_s1 = np.zeros(n_s1, dtype=np.float64)
    num_above_050_s1 = np.zeros(n_s1, dtype=np.int32)
    num_above_070_s1 = np.zeros(n_s1, dtype=np.int32)
    num_above_080_s1 = np.zeros(n_s1, dtype=np.int32)
    num_above_090_s1 = np.zeros(n_s1, dtype=np.int32)

    for idx in range(n_s1):
        st, en = s1_slices[idx]
        if st == en:
            continue
        sc_sub = scores[st:en]
        order = np.argsort(-sc_sub)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(sc_sub) + 1)
        rank_in_s1[st:en] = ranks

        t1 = float(sc_sub[order[0]])
        t2 = float(sc_sub[order[1]]) if len(sc_sub) >= 2 else 0.0
        t3 = float(sc_sub[order[2]]) if len(sc_sub) >= 3 else 0.0

        top1_scores_s1[idx] = t1
        top2_scores_s1[idx] = t2
        top3_scores_s1[idx] = t3
        num_above_050_s1[idx] = int((sc_sub >= 0.50).sum())
        num_above_070_s1[idx] = int((sc_sub >= 0.70).sum())
        num_above_080_s1[idx] = int((sc_sub >= 0.80).sum())
        num_above_090_s1[idx] = int((sc_sub >= 0.90).sum())

        top1_per_row[st:en] = t1
        top2_per_row[st:en] = t2
        margin_from_top1[st:en] = t1 - sc_sub
        s1_top1_top2_margin[st:en] = t1 - t2
        s1_country_arr[st:en] = countries[idx]

    policies_out = {}

    # 1. Policy 1: Threshold-only (baseline at best_base_thr)
    mask_p1 = scores >= best_base_thr
    policies_out["1_threshold_only"] = evaluate_predictions_from_mask(mask_p1, eval_struct, f"thr={best_base_thr}")

    # 2. Policy 2: Top-score + Margin rule (rescue clear unique top-1 winner when margin >= 0.25, prune weak tail)
    best_p2 = None
    for rescue_floor in [best_base_thr - 0.05, best_base_thr - 0.03]:
        for min_margin in [0.18, 0.25, 0.35]:
            for max_drop in [0.15, 0.22, 1.0]:
                m_p2 = (
                    (scores >= best_base_thr) & ((rank_in_s1 == 1) | (margin_from_top1 <= max_drop) | (scores >= best_base_thr + 0.04))
                ) | (
                    (rank_in_s1 == 1) & (scores >= rescue_floor) & (s1_top1_top2_margin >= min_margin)
                )
                res = evaluate_predictions_from_mask(
                    m_p2, eval_struct, f"thr={best_base_thr},rescue={rescue_floor:.3f},margin={min_margin},drop={max_drop}"
                )
                if best_p2 is None or res["macro_f05"] > best_p2["macro_f05"]:
                    best_p2 = res
    policies_out["2_top_score_plus_margin"] = best_p2

    # 3. Policy 3: Score + Structural Evidence (reward exact + numeric agreement, penalize numeric conflict & branch confusion)
    best_p3 = None
    p3_thrs = sorted(set([
        min(0.996, max(0.50, best_base_thr - 0.015)),
        min(0.996, best_base_thr),
        min(0.994, best_base_thr + 0.004),
    ]))
    for bonus_agree in [0.0, 0.005, 0.015]:
        for pen_conflict in [0.0, 0.005, 0.015, 0.03]:
            for pen_branch in [0.0, 0.01, 0.03]:
                adj_scores = (
                    scores
                    + bonus_agree * ((exact_ev | (num_agree & (core_name_set >= 0.85))).astype(np.float64))
                    - pen_conflict * (num_conflict.astype(np.float64))
                    - pen_branch * (high_name_low_addr.astype(np.float64))
                )
                for thr_try in p3_thrs:
                    m_p3 = adj_scores >= thr_try
                    res = evaluate_predictions_from_mask(
                        m_p3, eval_struct, f"thr={thr_try:.4f},bonus={bonus_agree},pen_num={pen_conflict},pen_br={pen_branch}"
                    )
                    res["_params"] = (thr_try, bonus_agree, pen_conflict, pen_branch)
                    if best_p3 is None or res["macro_f05"] > best_p3["macro_f05"]:
                        best_p3 = res
    p3_params = best_p3.pop("_params")
    policies_out["3_top_score_plus_evidence"] = best_p3

    # 4. Policy 4: Multi-match independent support + Empty-S1 protection
    best_p4 = None
    for max_k in [6, 8, 10, 15]:
        for tail_thr_bump in [0.0, 0.004, 0.01]:
            for empty_guard_thr in [best_base_thr, min(0.995, best_base_thr + 0.004)]:
                m_p4 = (
                    (scores >= best_base_thr)
                    & (rank_in_s1 <= max_k)
                    & ((rank_in_s1 <= 3) | (scores >= min(0.996, best_base_thr + tail_thr_bump)))
                    & ((top1_per_row >= empty_guard_thr) | exact_ev | num_agree)
                )
                res = evaluate_predictions_from_mask(
                    m_p4, eval_struct, f"thr={best_base_thr},max_k={max_k},tail_bump={tail_thr_bump},empty_guard={empty_guard_thr:.4f}"
                )
                if best_p4 is None or res["macro_f05"] > best_p4["macro_f05"]:
                    best_p4 = res
    policies_out["4_multi_match_and_empty_guard"] = best_p4

    # 5. Policy 5: Combined Evidence + Top-1 Margin Rescue + Per-Country Calibration (US vs India) + Multi-Match Cap
    thr_p3, bonus_agree, pen_conflict, pen_branch = p3_params
    adj_scores_best = (
        scores
        + bonus_agree * ((exact_ev | (num_agree & (core_name_set >= 0.85))).astype(np.float64))
        - pen_conflict * (num_conflict.astype(np.float64))
        - pen_branch * (high_name_low_addr.astype(np.float64))
    )

    best_p5 = None
    best_p5_config = None
    is_us_row = s1_country_arr == "us"
    is_in_row = s1_country_arr == "india"

    cand_thrs = sorted(set([
        round(min(0.996, max(0.50, best_base_thr - 0.015)), 4),
        round(min(0.996, max(0.50, best_base_thr - 0.005)), 4),
        round(min(0.996, best_base_thr), 4),
        round(min(0.993, best_base_thr + 0.003), 4),
        round(min(0.995, best_base_thr + 0.005), 4),
    ]))
    for t_us in cand_thrs:
        for t_in in cand_thrs:
            for max_k in [6, 8, 12]:
                for rescue_margin in [1.0, 0.18]:
                    rescue_floor = max(0.50, min(t_us, t_in) - 0.03)
                    base_pass = (
                        ((is_us_row & (adj_scores_best >= t_us)) | (is_in_row & (adj_scores_best >= t_in)))
                        & (rank_in_s1 <= max_k)
                    )
                    if rescue_margin < 1.0:
                        m_p5 = base_pass | (
                            (rank_in_s1 == 1)
                            & (adj_scores_best >= rescue_floor)
                            & (s1_top1_top2_margin >= rescue_margin)
                        )
                    else:
                        m_p5 = base_pass
                    res = evaluate_predictions_from_mask(
                        m_p5,
                        eval_struct,
                        f"t_us={t_us:.4f},t_in={t_in:.4f},bonus={bonus_agree},pen_num={pen_conflict},pen_br={pen_branch},max_k={max_k},rescue_margin={rescue_margin}",
                    )
                    if best_p5 is None or res["macro_f05"] > best_p5["macro_f05"]:
                        best_p5 = res
                        best_p5_config = {
                            "t_us": float(t_us),
                            "t_in": float(t_in),
                            "t_default": float(max(t_us, t_in)),
                            "bonus_agree": float(bonus_agree),
                            "pen_conflict": float(pen_conflict),
                            "pen_branch": float(pen_branch),
                            "max_k": int(max_k),
                            "rescue_floor": float(rescue_floor),
                            "rescue_margin": float(rescue_margin),
                        }
    best_p5["optimal_policy_config"] = best_p5_config
    policies_out["5_combined_country_evidence_margin"] = best_p5

    policies_out["entity_score_summary"] = {
        "mean_top1_score": float(np.mean(top1_scores_s1)),
        "mean_top2_score": float(np.mean(top2_scores_s1)),
        "mean_top3_score": float(np.mean(top3_scores_s1)),
        "mean_top1_top2_margin": float(np.mean(top1_scores_s1 - top2_scores_s1)),
        "mean_cands_above_050": float(np.mean(num_above_050_s1)),
        "mean_cands_above_070": float(np.mean(num_above_070_s1)),
        "mean_cands_above_080": float(np.mean(num_above_080_s1)),
        "mean_cands_above_090": float(np.mean(num_above_090_s1)),
    }
    return policies_out


def apply_frozen_policy(
    scores: np.ndarray,
    feat_df: pd.DataFrame,
    eval_struct: dict,
    policy_cfg: dict,
) -> np.ndarray:
    """Apply the frozen decision policy deterministically to an unseen validation/holdout pool."""
    n_rows = len(scores)
    n_s1 = len(eval_struct["s1_order"])
    s1_slices = eval_struct["s1_row_slices"]
    countries = eval_struct["countries"]

    exact_ev = (
        (feat_df["exact_orig_name"].to_numpy() > 0.5)
        | (feat_df["exact_norm_name"].to_numpy() > 0.5)
        | (feat_df["exact_core_name"].to_numpy() > 0.5)
        | (feat_df["exact_norm_addr"].to_numpy() > 0.5)
    )
    bldg_match = feat_df["building_number_match"].to_numpy()
    bldg_conflict = feat_df["building_number_conflict"].to_numpy() > 0.5
    postal_match = feat_df["postal_match"].to_numpy() > 0.5
    postal_conflict = feat_df["postal_conflict"].to_numpy() > 0.5
    core_name_set = feat_df["core_name_token_set"].to_numpy()
    high_name_low_addr = feat_df["high_name_low_addr"].to_numpy() > 0.5

    num_agree = (bldg_match >= 0.9) | postal_match
    num_conflict = bldg_conflict | postal_conflict

    rank_in_s1 = np.zeros(n_rows, dtype=np.int32)
    s1_top1_top2_margin = np.zeros(n_rows, dtype=np.float64)
    s1_country_arr = np.empty(n_rows, dtype=object)

    for idx in range(n_s1):
        st, en = s1_slices[idx]
        if st == en:
            continue
        sc_sub = scores[st:en]
        order = np.argsort(-sc_sub)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(sc_sub) + 1)
        rank_in_s1[st:en] = ranks
        t1 = float(sc_sub[order[0]])
        t2 = float(sc_sub[order[1]]) if len(sc_sub) >= 2 else 0.0
        s1_top1_top2_margin[st:en] = t1 - t2
        s1_country_arr[st:en] = countries[idx]

    adj_scores = (
        scores
        + policy_cfg["bonus_agree"] * ((exact_ev | (num_agree & (core_name_set >= 0.85))).astype(np.float64))
        - policy_cfg["pen_conflict"] * (num_conflict.astype(np.float64))
        - policy_cfg["pen_branch"] * (high_name_low_addr.astype(np.float64))
    )

    is_us = s1_country_arr == "us"
    is_in = s1_country_arr == "india"
    is_other = (~is_us) & (~is_in)

    base_pass = (
        (
            (is_us & (adj_scores >= policy_cfg["t_us"]))
            | (is_in & (adj_scores >= policy_cfg["t_in"]))
            | (is_other & (adj_scores >= policy_cfg["t_default"]))
        )
        & (rank_in_s1 <= policy_cfg["max_k"])
    )
    rescue_margin = float(policy_cfg.get("rescue_margin", 1.0))
    rescue_floor = float(policy_cfg.get("rescue_floor", 0.96))
    if rescue_margin < 1.0:
        mask = base_pass | (
            (rank_in_s1 == 1)
            & (adj_scores >= rescue_floor)
            & (s1_top1_top2_margin >= rescue_margin)
        )
    else:
        mask = base_pass
    return mask


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("PHASE V3-D: CONTROLLED 4-MODEL BAKEOFF ON DEV_3000", flush=True)
    print("=" * 80, flush=True)

    models_dir = Path("experiments/v3/models")
    res_dir = Path("experiments/v3/results")
    models_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    recompute_only = "--recompute-policies" in sys.argv and (res_dir / "dev_3000_all_model_scores.npz").exists()

    if recompute_only:
        print("Fast path: Loading cached DEV_3000 model scores from dev_3000_all_model_scores.npz...", flush=True)
        dev_feat_df = pd.read_parquet("experiments/v3/data/dev_3000_v3_features.parquet")
        dev_qdf = pd.read_csv("experiments/v3/pools/dev_3000_s1.tsv", sep="\t", dtype=str)
        eval_struct_dev = prepare_eval_structures(dev_qdf, dev_feat_df)
        y_dev = eval_struct_dev["label_col"]
        npz = np.load(res_dir / "dev_3000_all_model_scores.npz")
        v2c_dev_probs = npz["v2c_probs"]
        probs_a_xgb = npz["probs_a_xgb"]
        probs_b_lgb_bin = npz["probs_b_lgb_bin"]
        probs_c_lambdarank = npz["probs_c_lambdarank"]
        raw_c_dev = npz["raw_c_dev"]
        probs_d_xendcg = npz["probs_d_xendcg"]
        raw_d_dev = npz["raw_d_dev"]
        with open(res_dir / "v3_d_bakeoff_metrics.json", "r", encoding="utf-8") as f:
            prev_json = json.load(f)
        v2c_sweep = prev_json["v2c_reference_on_dev3000"]["threshold_sweep"]
        v2c_best = prev_json["v2c_reference_on_dev3000"]["best"]
        phon_ab = prev_json["phonetic_ablation"]
        ACTIVE_FEATURE_COLS = phon_ab["active_features"]
        phonetic_retained = phon_ab["phonetic_retained"]
        best_85 = {"macro_f05": phon_ab["core_85_best_f05"], "threshold": phon_ab["core_85_best_thr"]}
        best_88 = {"macro_f05": phon_ab["all_88_best_f05"], "threshold": phon_ab["all_88_best_thr"]}
        model_a_xgb = joblib.load(models_dir / "v3_model_a_xgboost.joblib")
        model_b_lgb_bin = joblib.load(models_dir / "v3_model_b_lgbm_binary.joblib")
    else:
        # Load V3-C training data and DEV_3000 feature matrix
        print("Loading v3_c_train_features.parquet and dev_3000_v3_features.parquet...", flush=True)
        train_df = pd.read_parquet("experiments/v3/data/v3_c_train_features.parquet")
        dev_feat_df = pd.read_parquet("experiments/v3/data/dev_3000_v3_features.parquet")
        dev_qdf = pd.read_csv("experiments/v3/pools/dev_3000_s1.tsv", sep="\t", dtype=str)

        # Ensure train_df is sorted by source1_entity_id for ranking models
        train_df = train_df.sort_values(["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)
        y_train = train_df["label"].to_numpy(dtype=np.int32)
        train_group_sizes = train_df.groupby("source1_entity_id", sort=False).size().to_numpy()

        eval_struct_dev = prepare_eval_structures(dev_qdf, dev_feat_df)
        y_dev = eval_struct_dev["label_col"]
        print(
            f"Loaded Train: {len(train_df):,} rows ({len(train_group_sizes):,} S1 groups) | "
            f"DEV_3000: {len(dev_feat_df):,} rows ({len(dev_qdf):,} S1 queries, {eval_struct_dev['total_gt_links']:,} GT links)",
            flush=True,
        )

        # =========================================================================
        # STEP 0: Reference V2-C Baseline on DEV_3000 Top-750
        # =========================================================================
        print("\n--- Evaluating Frozen V2-C Model (11 Locked Features) on DEV_3000 ---", flush=True)
        v2c_model = joblib.load("experiments/v2/models/xgboost_matcher_v2_c.joblib")
        v2c_dev_probs = v2c_model.predict_proba(dev_feat_df[LOCKED_FEATURE_COLUMNS])[:, 1]
        v2c_sweep = [
            evaluate_predictions_from_mask(v2c_dev_probs >= thr, eval_struct_dev, thr)
            for thr in SWEEP_THRESHOLDS
        ]
        v2c_best = max(v2c_sweep, key=lambda x: x["macro_f05"])
        print(
            f"  Frozen V2-C on DEV_3000: Best Macro F0.5 = {v2c_best['macro_f05']:.4f} @ thr={v2c_best['threshold']} "
            f"(Prec={v2c_best['macro_precision']*100:.2f}%, Rec={v2c_best['macro_recall']*100:.2f}%, "
            f"S2_rec={v2c_best['s2_metrics']['micro_rec']*100:.2f}%, S3_rec={v2c_best['s3_metrics']['micro_rec']*100:.2f}%)",
            flush=True,
        )

        # =========================================================================
        # STEP 1: Phonetic Feature Ablation Check on DEV_3000 (85 vs 88 features)
        # =========================================================================
        print("\n--- Phonetic Feature Ablation Check (85 Core vs 88 Core+Phonetic Features) ---", flush=True)
        xgb_ablation_params = dict(
            n_estimators=600,
            max_depth=7,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            tree_method="hist",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        t_ab0 = time.time()
        xgb_85 = XGBClassifier(**xgb_ablation_params)
        xgb_85.fit(train_df[V3_CORE_FEATURE_COLUMNS], y_train)
        p_85 = xgb_85.predict_proba(dev_feat_df[V3_CORE_FEATURE_COLUMNS])[:, 1]
        best_85 = max(
            (evaluate_predictions_from_mask(p_85 >= thr, eval_struct_dev, thr) for thr in SWEEP_THRESHOLDS),
            key=lambda x: x["macro_f05"],
        )

        xgb_88 = XGBClassifier(**xgb_ablation_params)
        xgb_88.fit(train_df[V3_ALL_FEATURE_COLUMNS], y_train)
        p_88 = xgb_88.predict_proba(dev_feat_df[V3_ALL_FEATURE_COLUMNS])[:, 1]
        best_88 = max(
            (evaluate_predictions_from_mask(p_88 >= thr, eval_struct_dev, thr) for thr in SWEEP_THRESHOLDS),
            key=lambda x: x["macro_f05"],
        )

        phonetic_retained = best_88["macro_f05"] >= best_85["macro_f05"]
        ACTIVE_FEATURE_COLS = V3_ALL_FEATURE_COLUMNS if phonetic_retained else V3_CORE_FEATURE_COLUMNS
        print(
            f"  85 Core Features Best DEV F0.5 = {best_85['macro_f05']:.4f} @ {best_85['threshold']} | "
            f"88 Core+Phonetic Best DEV F0.5 = {best_88['macro_f05']:.4f} @ {best_88['threshold']} "
            f"-> Phonetic Retained: {phonetic_retained} ({len(ACTIVE_FEATURE_COLS)} features active)",
            flush=True,
        )

        X_tr = train_df[ACTIVE_FEATURE_COLS]
        X_dv = dev_feat_df[ACTIVE_FEATURE_COLS]

        # Reuse whichever XGBoost model corresponds to ACTIVE_FEATURE_COLS as Model A!
        model_a_xgb = xgb_88 if phonetic_retained else xgb_85
        probs_a_xgb = p_88 if phonetic_retained else p_85
        joblib.dump(model_a_xgb, models_dir / "v3_model_a_xgboost.joblib")

        # =========================================================================
        # STEP 2: Train Models B, C, D on Exact Same X_tr and y_train
        # =========================================================================

        # Model B: LightGBM Binary Classifier
        print("\n--- Training Model B: LightGBM Binary Classifier ---", flush=True)
        t_b0 = time.time()
        model_b_lgb_bin = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=700,
            num_leaves=95,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=40,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model_b_lgb_bin.fit(X_tr, y_train)
        probs_b_lgb_bin = model_b_lgb_bin.predict_proba(X_dv)[:, 1]
        joblib.dump(model_b_lgb_bin, models_dir / "v3_model_b_lgbm_binary.joblib")
        print(f"  Model B trained & scored in {time.time()-t_b0:.1f}s", flush=True)

        # Model C: LightGBM LambdaRank
        print("\n--- Training Model C: LightGBM LambdaRank ---", flush=True)
        t_c0 = time.time()
        model_c_lambdarank = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[1, 3, 5, 10],
            n_estimators=600,
            num_leaves=95,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=40,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model_c_lambdarank.fit(X_tr, y_train, group=train_group_sizes)
        raw_c_train = model_c_lambdarank.predict(X_tr)
        raw_c_dev = model_c_lambdarank.predict(X_dv)
        platt_c = LogisticRegression()
        platt_c.fit(raw_c_train.reshape(-1, 1), y_train)
        probs_c_lambdarank = platt_c.predict_proba(raw_c_dev.reshape(-1, 1))[:, 1]
        joblib.dump({"ranker": model_c_lambdarank, "calibrator": platt_c}, models_dir / "v3_model_c_lambdarank.joblib")
        print(f"  Model C trained & scored in {time.time()-t_c0:.1f}s", flush=True)

        # Model D: LightGBM rank_xendcg
        print("\n--- Training Model D: LightGBM rank_xendcg ---", flush=True)
        t_d0 = time.time()
        model_d_xendcg = lgb.LGBMRanker(
            objective="rank_xendcg",
            metric="ndcg",
            ndcg_eval_at=[1, 3, 5, 10],
            n_estimators=600,
            num_leaves=95,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=40,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model_d_xendcg.fit(X_tr, y_train, group=train_group_sizes)
        raw_d_train = model_d_xendcg.predict(X_tr)
        raw_d_dev = model_d_xendcg.predict(X_dv)
        platt_d = LogisticRegression()
        platt_d.fit(raw_d_train.reshape(-1, 1), y_train)
        probs_d_xendcg = platt_d.predict_proba(raw_d_dev.reshape(-1, 1))[:, 1]
        joblib.dump({"ranker": model_d_xendcg, "calibrator": platt_d}, models_dir / "v3_model_d_rank_xendcg.joblib")
        print(f"  Model D trained & scored in {time.time()-t_d0:.1f}s", flush=True)

    # =========================================================================
    # STEP 3: Controlled Evaluation of All 4 Models on DEV_3000
    # =========================================================================
    bakeoff_models = [
        ("A_XGBoost_Binary", probs_a_xgb, None),
        ("B_LightGBM_Binary", probs_b_lgb_bin, None),
        ("C_LightGBM_LambdaRank", probs_c_lambdarank, raw_c_dev),
        ("D_LightGBM_rank_xendcg", probs_d_xendcg, raw_d_dev),
    ]

    bakeoff_results = {
        "phonetic_ablation": {
            "core_85_best_f05": best_85["macro_f05"],
            "core_85_best_thr": best_85["threshold"],
            "all_88_best_f05": best_88["macro_f05"],
            "all_88_best_thr": best_88["threshold"],
            "phonetic_retained": bool(phonetic_retained),
            "active_feature_count": len(ACTIVE_FEATURE_COLS),
            "active_features": ACTIVE_FEATURE_COLS,
        },
        "v2c_reference_on_dev3000": {
            "best": v2c_best,
            "threshold_sweep": v2c_sweep,
        },
        "models": {},
    }

    best_overall_model_name = None
    best_overall_f05 = -1.0
    best_overall_policy_cfg = None
    best_overall_base_thr = None

    for m_name, m_probs, m_raw in bakeoff_models:
        print(f"\nEvaluating {m_name} across thresholds & entity-level policies on DEV_3000...", flush=True)
        sweep = [
            evaluate_predictions_from_mask(m_probs >= thr, eval_struct_dev, thr)
            for thr in SWEEP_THRESHOLDS
        ]
        best_thr_entry = max(sweep, key=lambda x: x["macro_f05"])
        best_thr = float(best_thr_entry["threshold"])

        entity_policies = evaluate_entity_level_policies(m_probs, dev_feat_df, eval_struct_dev, best_thr)
        best_policy_entry = entity_policies["5_combined_country_evidence_margin"]

        # If ranking model, also evaluate raw score quantile/margin grid directly
        raw_ranking_eval = None
        if m_raw is not None:
            raw_q_thrs = [float(np.percentile(m_raw, q)) for q in [98.0, 98.5, 99.0, 99.2, 99.4, 99.5, 99.6, 99.7]]
            raw_sweep = [
                evaluate_predictions_from_mask(m_raw >= r_thr, eval_struct_dev, round(r_thr, 4))
                for r_thr in raw_q_thrs
            ]
            raw_ranking_eval = {
                "raw_score_distribution": summarize_score_distribution(m_raw, y_dev),
                "raw_threshold_sweep": raw_sweep,
                "best_raw_threshold": max(raw_sweep, key=lambda x: x["macro_f05"]),
            }

        m_summary = {
            "best_threshold_only": best_thr_entry,
            "best_entity_level_policy": best_policy_entry,
            "threshold_sweep": sweep,
            "entity_level_policies": entity_policies,
            "calibrated_score_distribution": summarize_score_distribution(m_probs, y_dev),
            "raw_ranking_evaluation": raw_ranking_eval,
        }
        bakeoff_results["models"][m_name] = m_summary

        print(
            f"  [{m_name}] Best Threshold-Only F0.5 = {best_thr_entry['macro_f05']:.4f} @ {best_thr} "
            f"(Prec={best_thr_entry['macro_precision']*100:.2f}%, Rec={best_thr_entry['macro_recall']*100:.2f}%, "
            f"S2_rec={best_thr_entry['s2_metrics']['micro_rec']*100:.2f}%, S3_rec={best_thr_entry['s3_metrics']['micro_rec']*100:.2f}%) | "
            f"Entity-Policy F0.5 = {best_policy_entry['macro_f05']:.4f}",
            flush=True,
        )

        # Track best candidate pipeline by DEV Macro F0.5
        cand_peak_f05 = max(best_thr_entry["macro_f05"], best_policy_entry["macro_f05"])
        if cand_peak_f05 > best_overall_f05:
            best_overall_f05 = cand_peak_f05
            best_overall_model_name = m_name
            best_overall_base_thr = best_thr
            best_overall_policy_cfg = best_policy_entry["optimal_policy_config"]

    # Also save top-25 feature importances for XGBoost and LightGBM Binary
    xgb_imp = sorted(
        zip(ACTIVE_FEATURE_COLS, [float(x) for x in model_a_xgb.feature_importances_]),
        key=lambda x: x[1],
        reverse=True,
    )[:25]
    lgb_imp = sorted(
        zip(ACTIVE_FEATURE_COLS, [float(x) for x in model_b_lgb_bin.feature_importances_]),
        key=lambda x: x[1],
        reverse=True,
    )[:25]

    bakeoff_results["feature_importances_top25"] = {
        "A_XGBoost_Binary": xgb_imp,
        "B_LightGBM_Binary": lgb_imp,
    }
    bakeoff_results["selected_frozen_pipeline"] = {
        "selected_model_name": best_overall_model_name,
        "dev_best_f05": float(best_overall_f05),
        "base_threshold": float(best_overall_base_thr),
        "policy_config": best_overall_policy_cfg,
        "active_features": ACTIVE_FEATURE_COLS,
        "total_bakeoff_runtime_s": round(time.time() - t0, 2),
    }

    with open(res_dir / "v3_d_bakeoff_metrics.json", "w", encoding="utf-8") as f:
        json.dump(bakeoff_results, f, indent=2)

    # Save DEV scores of all 4 models so post-analysis or ensemble check is instant
    np.savez_compressed(
        res_dir / "dev_3000_all_model_scores.npz",
        v2c_probs=v2c_dev_probs,
        probs_a_xgb=probs_a_xgb,
        probs_b_lgb_bin=probs_b_lgb_bin,
        probs_c_lambdarank=probs_c_lambdarank,
        raw_c_dev=raw_c_dev,
        probs_d_xendcg=probs_d_xendcg,
        raw_d_dev=raw_d_dev,
    )

    print("\n" + "=" * 80, flush=True)
    print(f"SELECTED FROZEN V3 PIPELINE: {best_overall_model_name} (DEV Macro F0.5 = {best_overall_f05:.4f})", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
