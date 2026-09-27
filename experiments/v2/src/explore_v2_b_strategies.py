"""Explore V2-B hard negative strategies and evaluate on the 1,000-S1 validation set.
"""

from collections import defaultdict
import gc
import json
from math import ceil
from pathlib import Path
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
import psutil
from rapidfuzz import fuzz, process
from xgboost import XGBClassifier

sys.path.insert(0, str(Path.cwd()))
from code.business_entity_resolution.src.evaluation import evaluate_macro_f_beta
from code.business_entity_resolution.src.feature_engineering import LOCKED_FEATURE_COLUMNS
from code.business_entity_resolution.src.preprocessing import (
    create_train_val_split,
    load_source_tsv,
    parse_match_ids,
)

MIN_RAM_THRESHOLD_GB = 2.0
PROB_CUTOFF = 0.80
BATCH_SIZE = 500_000
THRESHOLDS = [0.85, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.985, 0.990, 0.995]


def get_mem_stats():
    proc = psutil.Process()
    rss = proc.memory_info().rss / (1024 ** 3)
    peak = proc.memory_info().peak_wset / (1024 ** 3)
    avail = psutil.virtual_memory().available / (1024 ** 3)
    return rss, peak, avail


def candidate_pair_generator(s1_candidates, batch_size=BATCH_SIZE):
    current_s1 = []
    current_cand = []
    for s1_id, cands in s1_candidates:
        for cid in cands:
            current_s1.append(s1_id)
            current_cand.append(cid)
            if len(current_s1) >= batch_size:
                yield current_s1, current_cand
                current_s1 = []
                current_cand = []
    if current_s1:
        yield current_s1, current_cand


def evaluate_model_on_val(model, bundle, gt_dict, val_ids, total_gt_matches):
    val_s1 = bundle["val_s1"]
    s1_candidates = bundle["s1_candidates"]
    cand_records = bundle["cand_records"]
    s1_pair_counts = bundle["s1_pair_counts"]
    total_pairs = sum(s1_pair_counts)

    s1_lookup = {}
    for r in val_s1.itertuples(index=False):
        s1_lookup[r.entity_id] = (
            str(r.business_name) if pd.notna(r.business_name) else "",
            str(r.business_address) if pd.notna(r.business_address) else "",
            str(r.country).strip().lower() if pd.notna(r.country) else "",
        )

    n_batches = ceil(total_pairs / BATCH_SIZE)
    high_s1 = []
    high_cand = []
    high_prob = []

    for b_s1, b_cand in candidate_pair_generator(s1_candidates, batch_size=BATCH_SIZE):
        name1 = [s1_lookup[sid][0] for sid in b_s1]
        addr1 = [s1_lookup[sid][1] for sid in b_s1]
        country1 = [s1_lookup[sid][2] for sid in b_s1]

        name2 = [cand_records[cid][0] for cid in b_cand]
        addr2 = [cand_records[cid][1] for cid in b_cand]
        country2 = [cand_records[cid][2] for cid in b_cand]

        name_ratio = process.cpdist(name1, name2, scorer=fuzz.ratio, workers=-1, dtype=np.float64) / 100.0
        name_token_sort = process.cpdist(name1, name2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64) / 100.0
        name_token_set = process.cpdist(name1, name2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64) / 100.0
        address_ratio = process.cpdist(addr1, addr2, scorer=fuzz.ratio, workers=-1, dtype=np.float64) / 100.0
        address_token_sort = process.cpdist(addr1, addr2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64) / 100.0
        address_token_set = process.cpdist(addr1, addr2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64) / 100.0
        country_same = (np.array(country1) == np.array(country2)).astype(np.int64)
        name_len_1 = np.array([len(x) for x in name1], dtype=np.int64)
        name_len_2 = np.array([len(x) for x in name2], dtype=np.int64)
        address_len_1 = np.array([len(x) for x in addr1], dtype=np.int64)
        address_len_2 = np.array([len(x) for x in addr2], dtype=np.int64)

        X_batch = pd.DataFrame({
            "name_ratio": name_ratio,
            "name_token_sort": name_token_sort,
            "name_token_set": name_token_set,
            "address_ratio": address_ratio,
            "address_token_sort": address_token_sort,
            "address_token_set": address_token_set,
            "country_same": country_same,
            "name_len_1": name_len_1,
            "name_len_2": name_len_2,
            "address_len_1": address_len_1,
            "address_len_2": address_len_2,
        })

        probs = model.predict_proba(X_batch[LOCKED_FEATURE_COLUMNS])[:, 1]
        mask = np.where(probs >= PROB_CUTOFF)[0]
        for idx in mask:
            high_s1.append(b_s1[idx])
            high_cand.append(b_cand[idx])
            high_prob.append(float(probs[idx]))

        del name1, addr1, country1, name2, addr2, country2
        del name_ratio, name_token_sort, name_token_set, address_ratio, address_token_sort, address_token_set
        del country_same, name_len_1, name_len_2, address_len_1, address_len_2, X_batch, probs, mask
        gc.collect()

    arr_s1 = np.array(high_s1, dtype=object)
    arr_cand = np.array(high_cand, dtype=object)
    arr_prob = np.array(high_prob, dtype=np.float32)

    results = []
    for thresh in THRESHOLDS:
        mask = arr_prob >= thresh
        sel_s1 = arr_s1[mask]
        sel_cand = arr_cand[mask]

        pred_dict = {sid: set() for sid in val_ids}
        for s1, cid in zip(sel_s1, sel_cand):
            pred_dict[s1].add(cid)

        metrics = evaluate_macro_f_beta(gt_dict, pred_dict, beta=0.5)

        tp_count = 0
        fp_count = 0
        for sid, pset in pred_dict.items():
            tset = gt_dict.get(sid, set())
            tp_count += len(pset & tset)
            fp_count += len(pset - tset)
        fn_count = total_gt_matches - tp_count
        pred_singletons = sum(1 for pset in pred_dict.values() if len(pset) == 0)

        res = {
            "threshold": float(thresh),
            "macro_f05": float(metrics["macro_f_beta"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "singleton_accuracy": float(metrics["singleton_accuracy"]),
            "total_predicted_matches": int(len(sel_s1)),
            "predicted_singletons": int(pred_singletons),
            "tp": int(tp_count),
            "fp": int(fp_count),
            "fn": int(fn_count),
        }
        results.append(res)

    best_res = max(results, key=lambda x: x["macro_f05"])
    return results, best_res


def main():
    print("=" * 80)
    print("EXPLORING V2-B MODEL-HARD NEGATIVE STRATEGIES")
    print("=" * 80)

    # 1. Load clean train features and baseline model
    clean_df = pd.read_parquet("experiments/v2/data/v2_a_train_features.parquet")
    base_model = joblib.load("models/xgboost_matcher.joblib")
    
    print("Scoring clean training negatives with frozen baseline model...")
    t0 = time.time()
    neg_mask = (clean_df["label"] == 0).values
    pos_mask = ~neg_mask

    neg_probs = base_model.predict_proba(clean_df.loc[neg_mask, LOCKED_FEATURE_COLUMNS])[:, 1]
    print(f"Scoring complete in {time.time() - t0:.2f}s.")
    print(f"Negatives breakdown:")
    print(f"  Hard (p >= 0.50): {np.sum(neg_probs >= 0.50):,} ({np.mean(neg_probs >= 0.50)*100:.2f}%)")
    print(f"  Semi-hard (0.10 <= p < 0.50): {np.sum((neg_probs >= 0.10) & (neg_probs < 0.50)):,} ({np.mean((neg_probs >= 0.10) & (neg_probs < 0.50))*100:.2f}%)")
    print(f"  Easy (p < 0.10): {np.sum(neg_probs < 0.10):,} ({np.mean(neg_probs < 0.10)*100:.2f}%)")

    # Load validation bundle once
    bundle_path = Path(r"C:\Users\Sneha Singh\.gemini\antigravity\brain\47892a4a-c3be-4331-990a-46831df8639d\scratch\val_1000_candidates_bundle.joblib")
    bundle = joblib.load(bundle_path)
    val_ids = bundle["val_ids"]

    gt = load_source_tsv("dataset/train/train_ground_truth.tsv")
    gt = create_train_val_split(gt)
    val_gt = gt[gt["split"] == "validation"].copy()
    val_gt = val_gt[val_gt["source1_entity_id"].isin(set(val_ids))].copy()

    gt_dict = {}
    for r in val_gt.itertuples(index=False):
        gt_dict[r.source1_entity_id] = set(parse_match_ids(r.matched_entity_ids))
    del gt, val_gt
    gc.collect()

    total_gt_matches = sum(len(m) for m in gt_dict.values())

    xgb_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "enable_categorical": True,
        "n_jobs": -1,
    }

    # Strategy 1: Moderate Sample Weights on Negatives
    print("\n--- Strategy 1: Moderate Sample Weighting (Hard=2.0, Semi=1.2, Easy=1.0) ---")
    weights_s1 = np.ones(len(clean_df), dtype=np.float32)
    neg_indices = np.where(neg_mask)[0]
    weights_s1[neg_indices[neg_probs >= 0.50]] = 2.0
    weights_s1[neg_indices[(neg_probs >= 0.10) & (neg_probs < 0.50)]] = 1.2
    weights_s1[neg_indices[neg_probs < 0.10]] = 1.0

    m1 = XGBClassifier(**xgb_params)
    m1.fit(clean_df[LOCKED_FEATURE_COLUMNS], clean_df["label"].values, sample_weight=weights_s1)
    results1, best1 = evaluate_model_on_val(m1, bundle, gt_dict, val_ids, total_gt_matches)
    print(f"Strategy 1 Peak: Macro F0.5 = {best1['macro_f05']:.4f} at threshold {best1['threshold']} (Prec: {best1['macro_precision']*100:.2f}%, Rec: {best1['macro_recall']*100:.2f}%, TP: {best1['tp']}, FP: {best1['fp']})")

    # Strategy 2: Strong Sample Weighting (Hard=3.0, Semi=1.5, Easy=0.5)
    print("\n--- Strategy 2: Strong Sample Weighting (Hard=3.0, Semi=1.5, Easy=0.5) ---")
    weights_s2 = np.ones(len(clean_df), dtype=np.float32)
    weights_s2[neg_indices[neg_probs >= 0.50]] = 3.0
    weights_s2[neg_indices[(neg_probs >= 0.10) & (neg_probs < 0.50)]] = 1.5
    weights_s2[neg_indices[neg_probs < 0.10]] = 0.5

    m2 = XGBClassifier(**xgb_params)
    m2.fit(clean_df[LOCKED_FEATURE_COLUMNS], clean_df["label"].values, sample_weight=weights_s2)
    results2, best2 = evaluate_model_on_val(m2, bundle, gt_dict, val_ids, total_gt_matches)
    print(f"Strategy 2 Peak: Macro F0.5 = {best2['macro_f05']:.4f} at threshold {best2['threshold']} (Prec: {best2['macro_precision']*100:.2f}%, Rec: {best2['macro_recall']*100:.2f}%, TP: {best2['tp']}, FP: {best2['fp']})")

    # Strategy 3: Hard Negative Oversampling (Duplicating hard negatives 2x)
    print("\n--- Strategy 3: Hard Negative Oversampling (Duplicates of Hard Negatives) ---")
    hard_neg_indices = neg_indices[neg_probs >= 0.50]
    hard_neg_df = clean_df.iloc[hard_neg_indices].copy()
    oversampled_df = pd.concat([clean_df, hard_neg_df], ignore_index=True)

    m3 = XGBClassifier(**xgb_params)
    m3.fit(oversampled_df[LOCKED_FEATURE_COLUMNS], oversampled_df["label"].values)
    results3, best3 = evaluate_model_on_val(m3, bundle, gt_dict, val_ids, total_gt_matches)
    print(f"Strategy 3 Peak: Macro F0.5 = {best3['macro_f05']:.4f} at threshold {best3['threshold']} (Prec: {best3['macro_precision']*100:.2f}%, Rec: {best3['macro_recall']*100:.2f}%, TP: {best3['tp']}, FP: {best3['fp']})")

    # Compare all strategies against V2-A (0.5250)
    print("\n" + "=" * 90)
    print("COMPARISON OF ALL STRATEGIES AGAINST V2-A CLEAN BASELINE (0.5250)")
    print("=" * 90)
    print(f"{'Model':<35} | {'Peak Thresh':<12} | {'Macro F0.5':<12} | {'Delta vs V2-A':<15} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 90)
    print(f"{'V2-A Clean Baseline':<35} | {'0.985':<12} | {'0.5250':<12} | {'+0.0000':<15} | {'60.10%':<10} | {'45.34%':<10}")
    for name, r, b in [("V2-B Strategy 1 (Moderate Weights)", results1, best1),
                       ("V2-B Strategy 2 (Strong Weights)", results2, best2),
                       ("V2-B Strategy 3 (Oversampling)", results3, best3)]:
        delta = b['macro_f05'] - 0.5250
        print(f"{name:<35} | {b['threshold']:<12.3f} | {b['macro_f05']:<12.4f} | {delta:>+14.4f} | {b['macro_precision']*100:>9.2f}% | {b['macro_recall']*100:>9.2f}%")
    print("=" * 90)

    # Pick the best strategy and save as official V2-B model and metrics
    candidates = [
        ("v2_b_strategy_1", m1, results1, best1, weights_s1, clean_df),
        ("v2_b_strategy_2", m2, results2, best2, weights_s2, clean_df),
        ("v2_b_strategy_3", m3, results3, best3, None, oversampled_df),
    ]
    best_cand = max(candidates, key=lambda x: x[3]["macro_f05"])
    best_name, best_model, best_results, best_info, best_weights, best_df = best_cand

    print(f"\nSelecting best V2-B model: {best_name} with F0.5 = {best_info['macro_f05']:.4f}")
    
    # Save V2-B training dataset
    v2_b_data_path = Path("experiments/v2/data/v2_b_train_features.parquet")
    best_df_to_save = best_df.copy()
    if best_weights is not None:
        best_df_to_save["sample_weight"] = best_weights
    best_df_to_save.to_parquet(v2_b_data_path, index=False)
    print(f"Saved V2-B training data to {v2_b_data_path} ({v2_b_data_path.stat().st_size / (1024**2):.2f} MB)")

    # Save V2-B model
    v2_b_model_path = Path("experiments/v2/models/xgboost_matcher_v2_b.joblib")
    joblib.dump(best_model, v2_b_model_path)
    print(f"Saved V2-B model to {v2_b_model_path} ({v2_b_model_path.stat().st_size / (1024**2):.2f} MB)")

    # Save V2-B validation metrics
    v2_b_metrics_path = Path("experiments/v2/results/v2_b_val_metrics.json")
    with open(v2_b_metrics_path, "w", encoding="utf-8") as f:
        json.dump({"strategy": best_name, "results": best_results, "best": best_info}, f, indent=2)
    print(f"Saved V2-B validation metrics to {v2_b_metrics_path}")


if __name__ == "__main__":
    main()
