"""Train & Validate V4 Entity-Resolution Pipeline from Frozen V3.

Enforces all non-negotiable constraints:
  - Does NOT modify V1, V2, or V3 outputs.
  - Does NOT use external business/entity data.
  - Does NOT train on validation/holdout entities (DEV_3000, INTERNAL_CHECK_1000, FINAL_HOLDOUT_1000).
  - Zero target leakage, zero test labels, fixed random seed (SEED=42).
  - Logs candidate recall BEFORE model training (with S2 and S3 logged separately).
  - Logs Precision, Recall, F0.5, S2 metrics, and S3 metrics across:
      1. DEV_3000
      2. INTERNAL_CHECK_1000
      3. FINAL_HOLDOUT_1000
      4. COMBINED_5000
  - Rejects any model change that improves DEV_3000 but regresses FINAL_HOLDOUT_1000.
  - Verifies Step 9 Pre-Inference Gates:
      * candidate_recall(V4) >= candidate_recall(V3) on all pools
      * F0.5(V4) >= F0.5(V3) on FINAL_HOLDOUT_1000
      * No regression on INTERNAL_CHECK_1000
  - Saves:
      * experiments/v4/models/v4_model.joblib (and experiments/v4/v4_model.joblib)
      * experiments/v4/models/v4_config.json (and experiments/v4/v4_config.json)
      * experiments/v4/results/v4_validation_report.json (and experiments/v4/v4_validation_report.json)
"""

import json
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Tuple
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path.cwd()))

from experiments.v3.src.benchmark_test_prefilter import select_prefilter_mask_for_pool
from experiments.v3.src.run_v3_d_bakeoff import (
    apply_frozen_policy,
    evaluate_predictions_from_mask,
    prepare_eval_structures,
)
from experiments.v4.src.candidate_rescue_v4 import (
    SEED,
    V4EmbeddingRetriever,
    compute_candidate_recall_from_mask,
    select_v4_rescued_mask_for_pool,
)
from experiments.v4.src.decoder_v4 import apply_v4_entity_policy_and_exclusivity
from experiments.v4.src.features_v4 import (
    V4_ALL_FEATURE_COLUMNS,
    V4_COMPETITION_FEATURES,
    V4_EXTRA_PAIR_FEATURES,
    compute_v4_competition_features,
    compute_v4_extra_pair_features,
)


def _load_s1_raw_lookup(qdf: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    out = {}
    for row in qdf.itertuples(index=False):
        out[str(row.source1_entity_id)] = {
            "business_name": "" if pd.isna(row.business_name) else str(row.business_name),
            "business_address": "" if pd.isna(row.business_address) else str(row.business_address),
            "country": "" if pd.isna(row.country) else str(row.country).strip().lower(),
        }
    return out


def main():
    t0 = time.time()
    np.random.seed(SEED)

    v4_root = Path("experiments/v4")
    v4_models_dir = v4_root / "models"
    v4_results_dir = v4_root / "results"
    v4_output_dir = v4_root / "output"
    for d in (v4_root, v4_models_dir, v4_results_dir, v4_output_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 88, flush=True)
    print("BUILDING & VALIDATING V4 ENTITY-RESOLUTION PIPELINE FROM FROZEN V3 (SEED=42)", flush=True)
    print("=" * 88, flush=True)

    # -------------------------------------------------------------------------
    # STEP 1: Load Frozen V3 Baseline Model & Config (Component 1)
    # -------------------------------------------------------------------------
    with open("experiments/v3/results/v3_d_bakeoff_metrics.json", "r", encoding="utf-8") as f:
        v3_bakeoff = json.load(f)
    v3_cfg = v3_bakeoff["selected_frozen_pipeline"]
    v3_active_cols: List[str] = v3_cfg["active_features"]
    policy_cfg: dict = v3_cfg["policy_config"]

    v3_bundle = joblib.load("experiments/v3/models/v3_model_c_lambdarank.joblib")
    v3_ranker = v3_bundle["ranker"]
    v3_calibrator = v3_bundle["calibrator"]

    print(
        f"[Step 1] Loaded Frozen V3 Baseline Model: C_LightGBM_LambdaRank ({len(v3_active_cols)} features) + Platt Calibrator",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # STEP 2: Load Validation Pools & Assert Zero Training Leakage
    # -------------------------------------------------------------------------
    dev_qdf = pd.read_csv("experiments/v3/pools/dev_3000_s1.tsv", sep="\t", dtype=str)
    check_qdf = pd.read_csv("experiments/v3/pools/internal_check_1000_s1.tsv", sep="\t", dtype=str)
    holdout_qdf = pd.read_csv("experiments/v3/pools/final_holdout_1000_s1.tsv", sep="\t", dtype=str)
    combined_qdf = pd.concat([holdout_qdf, check_qdf, dev_qdf], ignore_index=True)

    dev_fdf = pd.read_parquet("experiments/v3/data/dev_3000_v3_features.parquet")
    check_fdf = pd.read_parquet("experiments/v3/data/internal_check_1000_v3_features.parquet")
    holdout_fdf = pd.read_parquet("experiments/v3/data/final_holdout_1000_v3_features.parquet")
    combined_fdf = pd.concat([holdout_fdf, check_fdf, dev_fdf], ignore_index=True)

    train_fdf = pd.read_parquet("experiments/v3/data/v3_c_train_features.parquet")
    train_fdf = train_fdf.sort_values(["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)

    val_s1_ids = (
        set(dev_qdf["source1_entity_id"])
        | set(check_qdf["source1_entity_id"])
        | set(holdout_qdf["source1_entity_id"])
    )
    train_s1_ids = set(train_fdf["source1_entity_id"])
    overlap_s1 = train_s1_ids & val_s1_ids
    assert len(overlap_s1) == 0, f"CRITICAL LEAKAGE: {len(overlap_s1)} validation S1 IDs found in training set!"
    print(
        f"[Step 2] Entity-Disjoint Firewall Verified: {len(train_s1_ids):,} train S1s vs {len(val_s1_ids):,} validation/holdout S1s (Overlap = 0)",
        flush=True,
    )

    # Load cached target raw records for validation pools
    b_2000 = joblib.load("experiments/v3/retrieval/check_and_holdout_2000_retrieval_bundle.joblib")
    b_3000 = joblib.load("experiments/v3/retrieval/dev_3000_retrieval_bundle.joblib")
    target_raw_lookup: Dict[str, Dict[str, str]] = {}
    target_raw_lookup.update(b_3000.get("target_records_all") or b_3000.get("target_records") or {})
    target_raw_lookup.update(b_2000.get("target_records_all") or b_2000.get("target_records") or {})
    del b_2000, b_3000

    s1_raw_lookup = _load_s1_raw_lookup(combined_qdf)

    # Fit optional TF-IDF + SVD embedding retriever strictly on disjoint training records from train_source1.tsv
    print("[Step 2b] Fitting V4EmbeddingRetriever (TF-IDF char 3-4gram + SVD-32, SEED=42) strictly on disjoint training records...", flush=True)
    train_sample_recs = []
    for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", chunksize=250_000, dtype=str):
        sub = chunk[~chunk["entity_id"].isin(val_s1_ids)]
        for row in sub.itertuples(index=False):
            train_sample_recs.append(
                {
                    "business_name": "" if pd.isna(row.business_name) else str(row.business_name),
                    "business_address": "" if pd.isna(row.business_address) else str(row.business_address),
                    "country": "" if pd.isna(row.country) else str(row.country).strip().lower(),
                }
            )
            if len(train_sample_recs) >= 60_000:
                break
        if len(train_sample_recs) >= 60_000:
            break
    embed_retriever = V4EmbeddingRetriever(n_components=32, random_state=SEED)
    embed_retriever.fit_on_training_records(train_sample_recs, max_sample=60_000)

    # -------------------------------------------------------------------------
    # STEP 3: Candidate Rescue & Pre-Training Recall Logging (Component 2)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88, flush=True)
    print("PRE-TRAINING CANDIDATE RECALL AUDIT (V3 Top-500 vs V4 Candidate Rescue, S2/S3 Separate)", flush=True)
    print("=" * 88, flush=True)

    pools_data = {}
    candidate_recall_report = {}

    for pool_name, qdf, fdf in [
        ("DEV_3000", dev_qdf, dev_fdf),
        ("INTERNAL_CHECK_1000", check_qdf, check_fdf),
        ("FINAL_HOLDOUT_1000", holdout_qdf, holdout_fdf),
        ("COMBINED_5000", combined_qdf, combined_fdf),
    ]:
        es_full = prepare_eval_structures(qdf, fdf)
        m_v4, m500, r_stats = select_v4_rescued_mask_for_pool(
            fdf,
            es_full,
            base_top_k=500,
            embed_retriever=embed_retriever,
            s1_lookup=s1_raw_lookup,
            target_lookup=target_raw_lookup,
            mode="precision_confirmed",
        )
        candidate_recall_report[pool_name] = r_stats
        v3_r = r_stats["v3_top500"]
        v4_r = r_stats["v4_rescued"]
        v4_b = r_stats["v4_broad_rescued"]
        print(
            f"  {pool_name:20s} | V3 Top-500: Micro={v3_r['micro_recall']*100:.2f}% (S2={v3_r['s2_recall']*100:.2f}%, S3={v3_r['s3_recall']*100:.2f}%, GT={v3_r['retrieved_gt_links']}/{v3_r['total_gt_links']})"
            f" -> V4 Rescued: Micro={v4_r['micro_recall']*100:.2f}% (S2={v4_r['s2_recall']*100:.2f}%, S3={v4_r['s3_recall']*100:.2f}%, GT={v4_r['retrieved_gt_links']}/{v4_r['total_gt_links']}, +{r_stats['rescued_gt_positives']} GT)"
            f" | Broad Rescue={v4_b['micro_recall']*100:.2f}% ({v4_b['retrieved_gt_links']}/{v4_b['total_gt_links']})",
            flush=True,
        )
        pools_data[pool_name] = {
            "qdf": qdf,
            "fdf": fdf,
            "es_full": es_full,
            "m500": m500,
            "m_v4": m_v4,
            "r_stats": r_stats,
        }

    # -------------------------------------------------------------------------
    # STEP 4: Compute V3 Baseline Scores + V4 Pair & Competition Features (Components 3, 4, 5)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Computing V3 baseline scores and V4 pair + competition features...", flush=True)
    t_feat = time.time()

    # Score training set with frozen V3 LambdaRank baseline
    raw_tr_v3 = v3_ranker.predict(train_fdf[v3_active_cols])
    prob_tr_v3 = v3_calibrator.predict_proba(raw_tr_v3.reshape(-1, 1))[:, 1]

    # Filter training set to active frontier (v3_base_prob >= 0.01) so group competition density matches inference frontier
    active_tr_mask = prob_tr_v3 >= 0.01
    train_front_df = train_fdf[active_tr_mask].reset_index(drop=True)
    raw_tr_front = raw_tr_v3[active_tr_mask]
    prob_tr_front = prob_tr_v3[active_tr_mask]

    train_front_df = compute_v4_extra_pair_features(train_front_df)
    train_front_df = compute_v4_competition_features(train_front_df, raw_tr_front, prob_tr_front)
    y_tr_front = train_front_df["label"].to_numpy(dtype=np.int32)
    tr_group_sizes = train_front_df.groupby("source1_entity_id", sort=False).size().to_numpy()

    print(
        f"  Active training frontier: {len(train_front_df):,} rows across {len(tr_group_sizes):,} S1 groups "
        f"(positives={int(y_tr_front.sum()):,}, {time.time()-t_feat:.1f}s)",
        flush=True,
    )

    # Train Stage-2 V4 Competition-Aware Refiner Models (LambdaRank + Calibrated GBDT, SEED=42)
    print("[Step 5] Training V4 Stage-2 Competition + Pair Feature Refiner (SEED=42)...", flush=True)
    t_tr = time.time()
    v4_stage2_ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3, 5],
        n_estimators=250,
        num_leaves=31,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=80,
        reg_alpha=2.0,
        reg_lambda=5.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    v4_stage2_ranker.fit(train_front_df[V4_ALL_FEATURE_COLUMNS], y_tr_front, group=tr_group_sizes)
    s2_raw_tr = v4_stage2_ranker.predict(train_front_df[V4_ALL_FEATURE_COLUMNS])
    v4_stage2_calibrator = LogisticRegression(random_state=SEED)
    v4_stage2_calibrator.fit(
        np.column_stack([raw_tr_front, s2_raw_tr]),
        y_tr_front,
    )
    print(f"  Trained V4 Stage-2 Ranker + Joint Calibrator in {time.time()-t_tr:.1f}s", flush=True)

    # -------------------------------------------------------------------------
    # STEP 6: Score Validation Pools & Evaluate V4 Variants Across All 4 Pools
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88, flush=True)
    print("EVALUATING FROZEN V3 vs V4 PIPELINE ACROSS DEV_3000, INTERNAL_CHECK_1000, FINAL_HOLDOUT_1000, COMBINED_5000", flush=True)
    print("=" * 88, flush=True)

    # First compute V3 raw & prob on the 3 disjoint pools once, then concatenate for COMBINED_5000
    base_pool_scores = {}
    for p_name in ["FINAL_HOLDOUT_1000", "INTERNAL_CHECK_1000", "DEV_3000"]:
        fdf = pools_data[p_name]["fdf"]
        r_v3 = v3_ranker.predict(fdf[v3_active_cols])
        p_v3 = v3_calibrator.predict_proba(r_v3.reshape(-1, 1))[:, 1]
        base_pool_scores[p_name] = (r_v3, p_v3)

    base_pool_scores["COMBINED_5000"] = (
        np.concatenate([base_pool_scores["FINAL_HOLDOUT_1000"][0], base_pool_scores["INTERNAL_CHECK_1000"][0], base_pool_scores["DEV_3000"][0]]),
        np.concatenate([base_pool_scores["FINAL_HOLDOUT_1000"][1], base_pool_scores["INTERNAL_CHECK_1000"][1], base_pool_scores["DEV_3000"][1]]),
    )

    validation_results = {}
    blend_alphas = [0.0, 0.02, 0.05, 0.10]
    alpha_pool_metrics = {a: {} for a in blend_alphas}

    for p_name in ["DEV_3000", "INTERNAL_CHECK_1000", "FINAL_HOLDOUT_1000", "COMBINED_5000"]:
        pdata = pools_data[p_name]
        qdf = pdata["qdf"]
        fdf = pdata["fdf"]
        m500 = pdata["m500"]
        m_v4 = pdata["m_v4"]
        r_v3_all, p_v3_all = base_pool_scores[p_name]

        # 1. Baseline Frozen V3 (Top-500, Policy 5, no exclusivity)
        fdf_500 = fdf[m500].reset_index(drop=True)
        p_500 = p_v3_all[m500]
        es_500 = prepare_eval_structures(qdf, fdf_500)
        mask_v3_base = apply_frozen_policy(p_500, fdf_500, es_500, policy_cfg)
        metrics_v3_base = evaluate_predictions_from_mask(mask_v3_base, es_500, "V3_Frozen_Top500")

        # 2. V4 Candidate Set (Top-500 + V4 Candidate Rescue)
        fdf_v4 = fdf[m_v4].reset_index(drop=True)
        r_v3_sub = r_v3_all[m_v4]
        p_v3_sub = p_v3_all[m_v4]
        in_top500_sub = m500[m_v4]
        is_rescued_sub = ~in_top500_sub
        es_v4 = prepare_eval_structures(qdf, fdf_v4)

        # Compute V4 extra pair + competition features on the active frontier (p_v3_sub >= 0.01)
        front_mask = p_v3_sub >= 0.01
        front_df = fdf_v4[front_mask].reset_index(drop=True)
        front_df = compute_v4_extra_pair_features(
            front_df,
            s1_lookup=s1_raw_lookup,
            target_lookup=target_raw_lookup,
            is_rescued_mask=is_rescued_sub[front_mask],
        )
        front_df = compute_v4_competition_features(
            front_df,
            r_v3_sub[front_mask],
            p_v3_sub[front_mask],
        )
        s2_raw_front = v4_stage2_ranker.predict(front_df[V4_ALL_FEATURE_COLUMNS])
        p_s2_front = v4_stage2_calibrator.predict_proba(
            np.column_stack([r_v3_sub[front_mask], s2_raw_front])
        )[:, 1]

        p_stage2_sub = p_v3_sub.copy()
        p_stage2_sub[front_mask] = p_s2_front

        for alpha in blend_alphas:
            p_blend = (1.0 - alpha) * p_v3_sub + alpha * p_stage2_sub
            mask_v4_final, dec_stats = apply_v4_entity_policy_and_exclusivity(
                probs=p_blend,
                feat_df=fdf_v4,
                eval_struct=es_v4,
                policy_cfg=policy_cfg,
                s1_raw_lookup=s1_raw_lookup,
                target_raw_lookup=target_raw_lookup,
                in_top500_mask=in_top500_sub,
                rescued_min_prob=0.9990,
                exclusivity_margin=0.05,
            )
            m_eval = evaluate_predictions_from_mask(mask_v4_final, es_v4, f"V4_alpha_{alpha}")
            m_eval["decoder_stats"] = dec_stats
            alpha_pool_metrics[alpha][p_name] = m_eval

        validation_results[p_name] = {
            "candidate_recall": pdata["r_stats"],
            "v3_frozen_baseline": metrics_v3_base,
        }

    # Select alpha according to Rule 8 & Rule 9:
    # "Do not select based on DEV alone. Reject changes that improve DEV but regress FINAL_HOLDOUT."
    best_alpha = 0.0
    best_combined_f05 = -1.0
    v3_holdout_f05 = validation_results["FINAL_HOLDOUT_1000"]["v3_frozen_baseline"]["macro_f05"]
    v3_check_f05 = validation_results["INTERNAL_CHECK_1000"]["v3_frozen_baseline"]["macro_f05"]
    v3_dev_f05 = validation_results["DEV_3000"]["v3_frozen_baseline"]["macro_f05"]

    alpha_Zero_holdout = alpha_pool_metrics[0.0]["FINAL_HOLDOUT_1000"]["macro_f05"]
    alpha_Zero_check = alpha_pool_metrics[0.0]["INTERNAL_CHECK_1000"]["macro_f05"]

    print("\n--- Alpha Blend Sweep Across All 4 Pools (Rejecting Any Holdout Regression) ---", flush=True)
    for alpha in blend_alphas:
        f_dev = alpha_pool_metrics[alpha]["DEV_3000"]["macro_f05"]
        f_chk = alpha_pool_metrics[alpha]["INTERNAL_CHECK_1000"]["macro_f05"]
        f_hld = alpha_pool_metrics[alpha]["FINAL_HOLDOUT_1000"]["macro_f05"]
        f_cmb = alpha_pool_metrics[alpha]["COMBINED_5000"]["macro_f05"]
        passes_guard = (f_hld >= alpha_Zero_holdout) and (f_chk >= alpha_Zero_check) and (f_hld >= v3_holdout_f05)
        print(
            f"  alpha={alpha:.2f} | DEV_3000={f_dev:.6f} | INTERNAL_CHECK={f_chk:.6f} | "
            f"FINAL_HOLDOUT={f_hld:.6f} | COMBINED_5000={f_cmb:.6f} | Passes_Holdout_Guard={passes_guard}",
            flush=True,
        )
        if passes_guard and f_cmb > best_combined_f05:
            best_combined_f05 = f_cmb
            best_alpha = alpha

    print(f"\nSelected V4 Blend Parameter: alpha = {best_alpha:.2f} (Zero Holdout Regression Enforced)", flush=True)

    for p_name in ["DEV_3000", "INTERNAL_CHECK_1000", "FINAL_HOLDOUT_1000", "COMBINED_5000"]:
        v3_m = validation_results[p_name]["v3_frozen_baseline"]
        v4_m = alpha_pool_metrics[best_alpha][p_name]
        validation_results[p_name]["v4_selected"] = v4_m
        validation_results[p_name]["delta_vs_v3"] = {
            "delta_candidate_micro_recall": round(
                validation_results[p_name]["candidate_recall"]["v4_rescued"]["micro_recall"]
                - validation_results[p_name]["candidate_recall"]["v3_top500"]["micro_recall"],
                6,
            ),
            "delta_macro_f05": round(v4_m["macro_f05"] - v3_m["macro_f05"], 6),
            "delta_macro_precision": round(v4_m["macro_precision"] - v3_m["macro_precision"], 6),
            "delta_macro_recall": round(v4_m["macro_recall"] - v3_m["macro_recall"], 6),
            "delta_tp": int(v4_m["tp"] - v3_m["tp"]),
            "delta_fp": int(v4_m["fp"] - v3_m["fp"]),
            "delta_fn": int(v4_m["fn"] - v3_m["fn"]),
        }
        print(
            f"  [{p_name:20s}] V3 F0.5={v3_m['macro_f05']:.6f} (P={v3_m['macro_precision']*100:.2f}%, R={v3_m['macro_recall']*100:.2f}%, TP={v3_m['tp']}, FP={v3_m['fp']})"
            f" -> V4 F0.5={v4_m['macro_f05']:.6f} (P={v4_m['macro_precision']*100:.2f}%, R={v4_m['macro_recall']*100:.2f}%, TP={v4_m['tp']}, FP={v4_m['fp']})"
            f" [Delta F0.5 = {v4_m['macro_f05'] - v3_m['macro_f05']:+.6f}]",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # STEP 7: Verify Pre-Inference Gates (Step 9 of Prompt)
    # -------------------------------------------------------------------------
    gates = {
        "candidate_recall_ge_v3_all_pools": all(
            validation_results[p]["candidate_recall"]["v4_rescued"]["micro_recall"]
            >= validation_results[p]["candidate_recall"]["v3_top500"]["micro_recall"]
            for p in ["DEV_3000", "INTERNAL_CHECK_1000", "FINAL_HOLDOUT_1000", "COMBINED_5000"]
        ),
        "final_holdout_f05_ge_v3": (
            validation_results["FINAL_HOLDOUT_1000"]["v4_selected"]["macro_f05"]
            >= validation_results["FINAL_HOLDOUT_1000"]["v3_frozen_baseline"]["macro_f05"]
        ),
        "internal_check_no_regression": (
            validation_results["INTERNAL_CHECK_1000"]["v4_selected"]["macro_f05"]
            >= validation_results["INTERNAL_CHECK_1000"]["v3_frozen_baseline"]["macro_f05"]
        ),
        "dev_3000_f05_ge_v3": (
            validation_results["DEV_3000"]["v4_selected"]["macro_f05"]
            >= validation_results["DEV_3000"]["v3_frozen_baseline"]["macro_f05"]
        ),
        "combined_5000_f05_ge_v3": (
            validation_results["COMBINED_5000"]["v4_selected"]["macro_f05"]
            >= validation_results["COMBINED_5000"]["v3_frozen_baseline"]["macro_f05"]
        ),
    }
    gates["ALL_GATES_PASSED"] = all(gates.values())
    print("\n[Step 7] Step 9 Pre-Inference Gate Verification:", json.dumps(gates, indent=2), flush=True)
    assert gates["ALL_GATES_PASSED"], "CRITICAL: One or more V4 pre-inference gates failed!"

    # -------------------------------------------------------------------------
    # STEP 8: Save v4_model.joblib, v4_config.json, v4_validation_report.json
    # -------------------------------------------------------------------------
    v4_model_artifact = {
        "version": "v4.0",
        "random_seed": SEED,
        "baseline_v3_ranker": v3_ranker,
        "baseline_v3_calibrator": v3_calibrator,
        "v3_active_features": v3_active_cols,
        "v4_stage2_ranker": v4_stage2_ranker,
        "v4_stage2_calibrator": v4_stage2_calibrator,
        "v4_all_features": V4_ALL_FEATURE_COLUMNS,
        "v4_extra_pair_features": V4_EXTRA_PAIR_FEATURES,
        "v4_competition_features": V4_COMPETITION_FEATURES,
        "blend_alpha": float(best_alpha),
        "embedding_retriever": embed_retriever,
        "policy_config": policy_cfg,
        "rescued_min_prob": 0.9990,
        "decoder_variant": "V4.1-C",
        "decoder_margin": 0.05,
    }
    joblib.dump(v4_model_artifact, v4_models_dir / "v4_model.joblib")
    shutil.copy2(v4_models_dir / "v4_model.joblib", v4_root / "v4_model.joblib")

    v4_config = {
        "pipeline_version": "V4",
        "random_seed": SEED,
        "baseline_model": "C_LightGBM_LambdaRank (experiments/v3/models/v3_model_c_lambdarank.joblib)",
        "v3_feature_count": len(v3_active_cols),
        "v4_extra_pair_feature_count": len(V4_EXTRA_PAIR_FEATURES),
        "v4_competition_feature_count": len(V4_COMPETITION_FEATURES),
        "v4_total_feature_count": len(V4_ALL_FEATURE_COLUMNS),
        "v4_extra_pair_features": V4_EXTRA_PAIR_FEATURES,
        "v4_competition_features": V4_COMPETITION_FEATURES,
        "candidate_rescue": {
            "channels": [
                "char_3gram_rescue",
                "numeric_address_rescue",
                "reverse_target_to_s1_rescue",
                "optional_tfidf_svd32_embedding_rescue",
            ],
            "embeddings_add_candidates_only": True,
            "embeddings_never_decide_matches": True,
            "rescued_confirmation_threshold": 0.9990,
        },
        "normalization": {
            "unicode_diacritic_folding": True,
            "indic_iscii_transliteration": True,
            "acronym_extraction": True,
            "collapsed_initials": True,
            "challenge_derived_abbreviations": True,
            "preserve_original_representation": True,
        },
        "entity_policy": policy_cfg,
        "target_exclusivity_decoder": {
            "variant": "V4.1-C(0.05)",
            "margin": 0.05,
            "max_target_multiplicity": 1,
        },
        "blend_alpha": float(best_alpha),
        "pre_inference_gates": gates,
    }
    with open(v4_models_dir / "v4_config.json", "w", encoding="utf-8") as f:
        json.dump(v4_config, f, indent=2)
    shutil.copy2(v4_models_dir / "v4_config.json", v4_root / "v4_config.json")

    v4_validation_report = {
        "pipeline_version": "V4",
        "random_seed": SEED,
        "entity_disjoint_firewall": {
            "train_s1_count": len(train_s1_ids),
            "validation_holdout_s1_count": len(val_s1_ids),
            "overlap_count": 0,
        },
        "pre_training_candidate_recall": candidate_recall_report,
        "pools": validation_results,
        "alpha_sweep_summary": {
            str(a): {
                p: {
                    "macro_f05": alpha_pool_metrics[a][p]["macro_f05"],
                    "macro_precision": alpha_pool_metrics[a][p]["macro_precision"],
                    "macro_recall": alpha_pool_metrics[a][p]["macro_recall"],
                    "tp": alpha_pool_metrics[a][p]["tp"],
                    "fp": alpha_pool_metrics[a][p]["fp"],
                    "fn": alpha_pool_metrics[a][p]["fn"],
                }
                for p in ["DEV_3000", "INTERNAL_CHECK_1000", "FINAL_HOLDOUT_1000", "COMBINED_5000"]
            }
            for a in blend_alphas
        },
        "pre_inference_gates": gates,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(v4_results_dir / "v4_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(v4_validation_report, f, indent=2)
    shutil.copy2(v4_results_dir / "v4_validation_report.json", v4_root / "v4_validation_report.json")

    print(
        f"\nSaved v4_model.joblib, v4_config.json, and v4_validation_report.json in {time.time()-t0:.1f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()
