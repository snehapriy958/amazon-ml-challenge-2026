"""V2-A Clean Baseline: Data Extraction, Model Training, and Validation Evaluation.

Strict safety constraints:
- Reads training_features_chunks/ in read-only mode.
- Filters strictly by split == 'train'.
- Ensures ZERO validation entity leakage.
- Saves clean data to experiments/v2/data/v2_a_train_features.parquet.
- Trains XGBoost with identical hyperparameters to baseline.
- Saves model to experiments/v2/models/xgboost_matcher_v2_a.joblib.
- Evaluates on the official 1,000-S1 validation set across threshold grid [0.85 to 0.995].
- Saves metrics to experiments/v2/results/v2_a_val_metrics.json.
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


def check_ram(stage: str):
    avail = psutil.virtual_memory().available / (1024 ** 3)
    if avail < MIN_RAM_THRESHOLD_GB:
        raise MemoryError(f"RAM safety limit hit at {stage}: {avail:.2f} GB available.")
    return avail


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


def step_1_extract_clean_data(val_ids_set):
    print("=" * 80)
    print("STEP 1: Extracting Clean Train Subset (Zero Validation Leakage)")
    print("=" * 80)
    
    gt = load_source_tsv("dataset/train/train_ground_truth.tsv")
    gt = create_train_val_split(gt)
    train_gt = gt[gt["split"] == "train"]
    train_s1_set = set(train_gt["source1_entity_id"].unique())
    print(f"Total train-split S1 entities in GT: {len(train_s1_set):,}")
    del gt, train_gt
    gc.collect()

    chunk_files = sorted(Path("training_features_chunks").glob("train_features_chunk_*.parquet"))
    if not chunk_files:
        raise FileNotFoundError("No chunk files found in training_features_chunks/")

    print(f"Reading {len(chunk_files)} chunk files...")
    clean_dfs = []
    total_raw_rows = 0
    
    for cf in chunk_files:
        df = pd.read_parquet(cf)
        total_raw_rows += len(df)
        cdf = df[df["source1_entity_id"].isin(train_s1_set)].copy()
        clean_dfs.append(cdf)
        print(f"  {cf.name}: {len(df):,} total -> {len(cdf):,} clean train pairs")

    clean_df = pd.concat(clean_dfs, ignore_index=True)
    del clean_dfs
    gc.collect()

    clean_s1 = set(clean_df["source1_entity_id"].unique())
    overlap = clean_s1 & val_ids_set
    print(f"\nTotal raw pairs: {total_raw_rows:,}")
    print(f"Clean train pairs: {len(clean_df):,}")
    print(f"Clean unique S1 entities: {len(clean_s1):,}")
    print(f"Label distribution: {clean_df['label'].value_counts().to_dict()}")
    print(f"Validation overlap check: {len(overlap)} entities overlapping")
    
    assert len(overlap) == 0, f"FATAL: Found {len(overlap)} overlapping validation entities!"
    assert len(clean_df) == 667814, f"Expected 667,814 clean rows, got {len(clean_df)}"

    out_data_path = Path("experiments/v2/data/v2_a_train_features.parquet")
    out_data_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(out_data_path, index=False)
    print(f"Saved clean train features to {out_data_path} ({out_data_path.stat().st_size / (1024**2):.2f} MB)")
    return clean_df


def step_2_train_model(clean_df):
    print("\n" + "=" * 80)
    print("STEP 2: Training V2-A XGBoost Model (Identical Hyperparameters)")
    print("=" * 80)

    X = clean_df[LOCKED_FEATURE_COLUMNS]
    y = clean_df["label"].values
    print(f"Training dataset: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Class ratio: {np.sum(y == 1):,} positives ({np.mean(y == 1)*100:.1f}%), {np.sum(y == 0):,} negatives")

    params = {
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
    print(f"Model parameters: {params}")
    t0 = time.time()
    model = XGBClassifier(**params)
    model.fit(X, y)
    t_train = time.time() - t0
    print(f"Model training complete in {t_train:.2f} seconds.")

    out_model_path = Path("experiments/v2/models/xgboost_matcher_v2_a.joblib")
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model_path)
    print(f"Saved model to {out_model_path} ({out_model_path.stat().st_size / (1024**2):.2f} MB)")
    return model


def step_3_evaluate(model, bundle_path):
    print("\n" + "=" * 80)
    print("STEP 3: Evaluating V2-A Model on Official 1,000-S1 Validation Set")
    print("=" * 80)

    bundle = joblib.load(bundle_path)
    val_ids = bundle["val_ids"]
    val_s1 = bundle["val_s1"]
    s1_candidates = bundle["s1_candidates"]
    cand_records = bundle["cand_records"]
    s1_pair_counts = bundle["s1_pair_counts"]
    total_pairs = sum(s1_pair_counts)
    print(f"Loaded {len(val_ids):,} entities with {total_pairs:,} candidate pairs.")

    s1_lookup = {}
    for r in val_s1.itertuples(index=False):
        s1_lookup[r.entity_id] = (
            str(r.business_name) if pd.notna(r.business_name) else "",
            str(r.business_address) if pd.notna(r.business_address) else "",
            str(r.country).strip().lower() if pd.notna(r.country) else "",
        )

    print("Loading official validation ground truth...")
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
    print(f"Ground-truth positive links across 1,000 entities: {total_gt_matches:,}")

    n_batches = ceil(total_pairs / BATCH_SIZE)
    print(f"Scoring {total_pairs:,} pairs in {n_batches} batches...")

    high_s1 = []
    high_cand = []
    high_prob = []

    total_feat_time = 0.0
    total_score_time = 0.0
    processed_pairs = 0
    batch_idx = 0

    for b_s1, b_cand in candidate_pair_generator(s1_candidates, batch_size=BATCH_SIZE):
        batch_idx += 1
        b_len = len(b_s1)
        check_ram(f"Batch {batch_idx}/{n_batches} start")

        name1 = [s1_lookup[sid][0] for sid in b_s1]
        addr1 = [s1_lookup[sid][1] for sid in b_s1]
        country1 = [s1_lookup[sid][2] for sid in b_s1]

        name2 = [cand_records[cid][0] for cid in b_cand]
        addr2 = [cand_records[cid][1] for cid in b_cand]
        country2 = [cand_records[cid][2] for cid in b_cand]

        t_f0 = time.time()
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
        b_feat_time = time.time() - t_f0
        total_feat_time += b_feat_time

        t_s0 = time.time()
        probs = model.predict_proba(X_batch[LOCKED_FEATURE_COLUMNS])[:, 1]
        b_score_time = time.time() - t_s0
        total_score_time += b_score_time

        mask = np.where(probs >= PROB_CUTOFF)[0]
        for idx in mask:
            high_s1.append(b_s1[idx])
            high_cand.append(b_cand[idx])
            high_prob.append(float(probs[idx]))

        processed_pairs += b_len

        del name1, addr1, country1, name2, addr2, country2
        del name_ratio, name_token_sort, name_token_set, address_ratio, address_token_sort, address_token_set
        del country_same, name_len_1, name_len_2, address_len_1, address_len_2, X_batch, probs, mask
        gc.collect()

        if batch_idx % 4 == 0 or batch_idx == n_batches:
            _, _, b_avail = get_mem_stats()
            print(f"  Batch {batch_idx:>2}/{n_batches} | Retained (>= {PROB_CUTOFF}): {len(high_prob):>6,d} | Avail RAM: {b_avail:.2f} GB", flush=True)

    print(f"\nScoring complete: {processed_pairs:,} pairs in {total_feat_time:.1f}s feat + {total_score_time:.1f}s score.")
    print(f"Retained {len(high_prob):,} pairs with prob >= {PROB_CUTOFF}.")

    del bundle, s1_candidates, cand_records, s1_lookup
    gc.collect()

    arr_s1 = np.array(high_s1, dtype=object)
    arr_cand = np.array(high_cand, dtype=object)
    arr_prob = np.array(high_prob, dtype=np.float32)

    results = []
    print("\n" + "=" * 115)
    print(f"{'Threshold':>10} | {'Macro F0.5':>10} | {'Macro Prec':>10} | {'Macro Rec':>10} | {'Pred Pairs':>10} | {'Singletons':>10} | {'TP':>7} | {'FP':>7} | {'FN':>7}")
    print("-" * 115)

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
        print(
            f"{thresh:>10.3f} | {res['macro_f05']:>10.4f} | {res['macro_precision']*100:>9.2f}% | "
            f"{res['macro_recall']*100:>9.2f}% | {res['total_predicted_matches']:>10,d} | "
            f"{res['predicted_singletons']:>10,d} | {res['tp']:>7,d} | {res['fp']:>7,d} | {res['fn']:>7,d}"
        )
    print("=" * 115)

    best_res = max(results, key=lambda x: x["macro_f05"])
    print(f"\nBest V2-A Result: Macro F0.5 = {best_res['macro_f05']:.4f} at Threshold {best_res['threshold']}")
    
    out_metrics_path = Path("experiments/v2/results/v2_a_val_metrics.json")
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best_res}, f, indent=2)
    print(f"Saved validation metrics to {out_metrics_path}")

    return results, best_res


def main():
    bundle_path = Path(r"C:\Users\Sneha Singh\.gemini\antigravity\brain\47892a4a-c3be-4331-990a-46831df8639d\scratch\val_1000_candidates_bundle.joblib")
    bundle = joblib.load(bundle_path)
    val_ids_set = set(bundle["val_ids"])
    del bundle
    gc.collect()

    clean_df = step_1_extract_clean_data(val_ids_set)
    model = step_2_train_model(clean_df)
    del clean_df
    gc.collect()
    step_3_evaluate(model, bundle_path)


if __name__ == "__main__":
    main()
