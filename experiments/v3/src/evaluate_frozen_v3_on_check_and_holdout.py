"""Single-Shot Post-Bakeoff Evaluation of Frozen V3 Pipeline on INTERNAL_CHECK_1000 and FINAL_HOLDOUT_1000.

CRITICAL FIREWALL PROTOCOL:
  - Loads the frozen winning V3 model, threshold, and entity-level decision policy
    selected strictly on DEV_3000 from `experiments/v3/results/v3_d_bakeoff_metrics.json`.
  - Performs ZERO tuning, ZERO threshold adjustment, and ZERO feature changes on
    INTERNAL_CHECK_1000 or FINAL_HOLDOUT_1000.
  - Evaluates INTERNAL_CHECK_1000 once and FINAL_HOLDOUT_1000 once.
"""

import gc
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from code.business_entity_resolution.src.feature_engineering import LOCKED_FEATURE_COLUMNS
from experiments.v3.src.features_v3 import compute_v3_features
from experiments.v3.src.retrieval_v3 import (
    evaluate_retrieval_recall,
    retrieve_candidates_v3_for_queries,
)
from experiments.v3.src.run_v3_d_bakeoff import (
    apply_frozen_policy,
    evaluate_predictions_from_mask,
    prepare_eval_structures,
)
from code.business_entity_resolution.src.preprocessing import parse_match_ids


def build_feature_df_for_pool(
    pool_name: str,
    qdf: pd.DataFrame,
    cands_by_s1: Dict[str, List[Tuple[str, float, int]]],
    target_records: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Build pairwise DataFrame and compute all 88 V3 features for a validation pool."""
    s1_lookup = {}
    gt_by_s1 = {}
    for row in qdf.itertuples(index=False):
        sid = row.source1_entity_id
        s1_lookup[sid] = {
            "business_name": "" if pd.isna(row.business_name) else str(row.business_name),
            "business_address": "" if pd.isna(row.business_address) else str(row.business_address),
            "country": "" if pd.isna(row.country) else str(row.country).strip().lower(),
        }
        m_str = "" if pd.isna(row.matched_entity_ids) else str(row.matched_entity_ids)
        gt_by_s1[sid] = set(parse_match_ids(m_str))

    rows_s1 = []
    rows_cand = []
    rows_label = []
    rows_ret_score = []
    rows_ch_mask = []

    for sid in qdf["source1_entity_id"]:
        g_set = gt_by_s1.get(sid, set())
        for cid, r_sc, m_mask in cands_by_s1.get(sid, []):
            if cid not in target_records:
                continue
            rows_s1.append(sid)
            rows_cand.append(cid)
            rows_label.append(1 if cid in g_set else 0)
            rows_ret_score.append(float(r_sc))
            rows_ch_mask.append(int(m_mask))

    pairs_df = pd.DataFrame(
        {
            "source1_entity_id": rows_s1,
            "candidate_entity_id": rows_cand,
            "label": np.array(rows_label, dtype=np.int8),
            "retrieval_score": np.array(rows_ret_score, dtype=np.float32),
            "channel_mask": np.array(rows_ch_mask, dtype=np.int8),
        }
    )
    print(f"Computing V3 features for {pool_name} ({len(pairs_df):,} candidate pairs)...", flush=True)
    feat_df = compute_v3_features(
        pairs_df,
        s1_lookup=s1_lookup,
        target_lookup=target_records,
        batch_size=150_000,
    )
    return pd.concat([pairs_df, feat_df], axis=1)


def score_with_frozen_model(model_name: str, feat_df: pd.DataFrame, active_cols: List[str]) -> np.ndarray:
    """Score candidate pairs using the frozen winning model artifact."""
    models_dir = Path("experiments/v3/models")
    X = feat_df[active_cols]
    if model_name == "A_XGBoost_Binary":
        model = joblib.load(models_dir / "v3_model_a_xgboost.joblib")
        return model.predict_proba(X)[:, 1]
    elif model_name == "B_LightGBM_Binary":
        model = joblib.load(models_dir / "v3_model_b_lgbm_binary.joblib")
        return model.predict_proba(X)[:, 1]
    elif model_name == "C_LightGBM_LambdaRank":
        bundle = joblib.load(models_dir / "v3_model_c_lambdarank.joblib")
        raw = bundle["ranker"].predict(X)
        return bundle["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]
    elif model_name == "D_LightGBM_rank_xendcg":
        bundle = joblib.load(models_dir / "v3_model_d_rank_xendcg.joblib")
        raw = bundle["ranker"].predict(X)
        return bundle["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def evaluate_pool_single_shot(
    pool_name: str,
    qdf: pd.DataFrame,
    feat_df: pd.DataFrame,
    ret_eval: dict,
    bakeoff_data: dict,
    v2c_model,
) -> dict:
    """Run single-shot evaluation on a firewall pool (zero tuning)."""
    eval_struct = prepare_eval_structures(qdf, feat_df)
    frozen_cfg = bakeoff_data["selected_frozen_pipeline"]
    active_cols = frozen_cfg["active_features"]
    model_name = frozen_cfg["selected_model_name"]
    base_thr = float(frozen_cfg["base_threshold"])
    policy_cfg = frozen_cfg["policy_config"]

    # 1. Score with primary frozen V3 model
    v3_probs = score_with_frozen_model(model_name, feat_df, active_cols)

    # Evaluate frozen threshold-only
    v3_thr_only_metrics = evaluate_predictions_from_mask(
        v3_probs >= base_thr, eval_struct, base_thr
    )

    # Evaluate frozen entity-level decision policy
    v3_policy_mask = apply_frozen_policy(v3_probs, feat_df, eval_struct, policy_cfg)
    v3_policy_metrics = evaluate_predictions_from_mask(
        v3_policy_mask, eval_struct, f"frozen_policy({model_name})"
    )

    use_entity_policy = bool(frozen_cfg.get("use_entity_policy", True))
    primary_v3_metrics = v3_policy_metrics if use_entity_policy else v3_thr_only_metrics

    # Also record frozen single-shot evaluation of all 4 bakeoff models (using their frozen DEV_3000 configs)
    all_frozen_models = {}
    for m_key, m_info in bakeoff_data["models"].items():
        m_probs = v3_probs if m_key == model_name else score_with_frozen_model(m_key, feat_df, active_cols)
        m_thr = float(m_info["best_threshold_only"]["threshold"])
        m_pol_cfg = m_info["best_entity_level_policy"]["optimal_policy_config"]
        m_thr_res = evaluate_predictions_from_mask(m_probs >= m_thr, eval_struct, m_thr)
        m_pol_mask = apply_frozen_policy(m_probs, feat_df, eval_struct, m_pol_cfg)
        m_pol_res = evaluate_predictions_from_mask(m_pol_mask, eval_struct, f"frozen_policy({m_key})")
        all_frozen_models[m_key] = {
            "frozen_threshold_only": m_thr_res,
            "frozen_entity_policy": m_pol_res,
        }

    # 2. Also score with frozen V2-C reference model at its locked threshold 0.900
    v2c_probs = v2c_model.predict_proba(feat_df[LOCKED_FEATURE_COLUMNS])[:, 1]
    v2c_ref_metrics = evaluate_predictions_from_mask(v2c_probs >= 0.900, eval_struct, 0.900)

    print(
        f"\n[{pool_name}] Single-Shot Firewall Evaluation Results:\n"
        f"  Candidate Recall (Top-750): Micro = {ret_eval['micro_link_recall']*100:.2f}% | "
        f"Macro Non-Empty = {ret_eval['macro_entity_recall_nonempty']*100:.2f}% "
        f"(S2={ret_eval['s2_link_recall']*100:.2f}%, S3={ret_eval['s3_link_recall']*100:.2f}%)\n"
        f"  Frozen V2-C (@0.900):       Macro F0.5 = {v2c_ref_metrics['macro_f05']:.4f} "
        f"(Prec={v2c_ref_metrics['macro_precision']*100:.2f}%, Rec={v2c_ref_metrics['macro_recall']*100:.2f}%)\n"
        f"  Frozen Primary ({model_name}) Threshold-Only: Macro F0.5 = {v3_thr_only_metrics['macro_f05']:.4f} "
        f"(Prec={v3_thr_only_metrics['macro_precision']*100:.2f}%, Rec={v3_thr_only_metrics['macro_recall']*100:.2f}%, "
        f"S2_rec={v3_thr_only_metrics['s2_metrics']['micro_rec']*100:.2f}%, S3_rec={v3_thr_only_metrics['s3_metrics']['micro_rec']*100:.2f}%)\n"
        f"  Frozen Primary ({model_name}) Entity-Policy:  Macro F0.5 = {v3_policy_metrics['macro_f05']:.4f} "
        f"(Prec={v3_policy_metrics['macro_precision']*100:.2f}%, Rec={v3_policy_metrics['macro_recall']*100:.2f}%, "
        f"S2_rec={v3_policy_metrics['s2_metrics']['micro_rec']*100:.2f}%, S3_rec={v3_policy_metrics['s3_metrics']['micro_rec']*100:.2f}%)",
        flush=True,
    )
    for m_key, m_res in all_frozen_models.items():
        t_r = m_res["frozen_threshold_only"]
        p_r = m_res["frozen_entity_policy"]
        print(
            f"    -> [{m_key}] Thr-Only F0.5={t_r['macro_f05']:.4f} | Entity-Policy F0.5={p_r['macro_f05']:.4f} "
            f"(Prec={p_r['macro_precision']*100:.2f}%, Rec={p_r['macro_recall']*100:.2f}%, "
            f"S2_rec={p_r['s2_metrics']['micro_rec']*100:.2f}%, S3_rec={p_r['s3_metrics']['micro_rec']*100:.2f}%)",
            flush=True,
        )

    return {
        "pool_name": pool_name,
        "retrieval": ret_eval,
        "frozen_v3_primary": primary_v3_metrics,
        "frozen_v3_threshold_only": v3_thr_only_metrics,
        "frozen_v3_entity_policy": v3_policy_metrics,
        "all_frozen_models": all_frozen_models,
        "frozen_v2c_reference_at_0900": v2c_ref_metrics,
    }


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("SINGLE-SHOT FIREWALL EVALUATION: INTERNAL_CHECK_1000 & FINAL_HOLDOUT_1000", flush=True)
    print("=" * 80, flush=True)

    res_dir = Path("experiments/v3/results")
    ret_dir = Path("experiments/v3/retrieval")
    data_dir = Path("experiments/v3/data")

    with open(res_dir / "v3_d_bakeoff_metrics.json", "r", encoding="utf-8") as f:
        bakeoff_data = json.load(f)

    frozen_cfg = bakeoff_data["selected_frozen_pipeline"]
    win_model = frozen_cfg["selected_model_name"]
    win_m_summary = bakeoff_data["models"][win_model]
    use_entity_policy = (
        win_m_summary["best_entity_level_policy"]["macro_f05"]
        >= win_m_summary["best_threshold_only"]["macro_f05"]
    )
    frozen_cfg["use_entity_policy"] = use_entity_policy

    print(
        f"Loaded Frozen V3 Pipeline from DEV_3000:\n"
        f"  Model:             {win_model}\n"
        f"  DEV Best F0.5:     {frozen_cfg['dev_best_f05']:.4f}\n"
        f"  Base Threshold:    {frozen_cfg['base_threshold']}\n"
        f"  Use Entity Policy: {use_entity_policy}\n"
        f"  Policy Config:     {frozen_cfg['policy_config']}",
        flush=True,
    )

    check_df = pd.read_csv("experiments/v3/pools/internal_check_1000_s1.tsv", sep="\t", dtype=str)
    holdout_df = pd.read_csv("experiments/v3/pools/final_holdout_1000_s1.tsv", sep="\t", dtype=str)

    check_ids = set(check_df["source1_entity_id"])
    holdout_ids = set(holdout_df["source1_entity_id"])
    assert len(check_ids & holdout_ids) == 0, "Overlap between INTERNAL_CHECK_1000 and FINAL_HOLDOUT_1000!"

    # Retrieve Top-750 candidates for both pools in a single pass over S2/S3 (per-query independent)
    bundle_path = ret_dir / "check_and_holdout_2000_retrieval_bundle.joblib"
    if bundle_path.exists():
        print(f"Loading cached {bundle_path}...", flush=True)
        b_data = joblib.load(bundle_path)
        cands_all = b_data["cands_all"]
        target_records_all = b_data["target_records_all"]
        ret_meta = b_data["ret_meta"]
    else:
        combined_qdf = pd.concat([check_df, holdout_df], ignore_index=True)
        cands_all, target_records_all, ret_meta = retrieve_candidates_v3_for_queries(
            combined_qdf, top_k=750, run_tfidf_benchmark=False
        )
        joblib.dump(
            {"cands_all": cands_all, "target_records_all": target_records_all, "ret_meta": ret_meta},
            bundle_path,
            compress=3,
        )
        print(f"Saved {bundle_path}", flush=True)

    cands_check = {sid: cands_all.get(sid, []) for sid in check_df["source1_entity_id"]}
    cands_holdout = {sid: cands_all.get(sid, []) for sid in holdout_df["source1_entity_id"]}

    ret_eval_check = evaluate_retrieval_recall(check_df, cands_check, pool_name="INTERNAL_CHECK_1000")
    ret_eval_holdout = evaluate_retrieval_recall(holdout_df, cands_holdout, pool_name="FINAL_HOLDOUT_1000")

    # Compute 88 V3 features for each pool
    feat_check_df = build_feature_df_for_pool("INTERNAL_CHECK_1000", check_df, cands_check, target_records_all)
    feat_check_df.to_parquet(data_dir / "internal_check_1000_v3_features.parquet", index=False)

    feat_holdout_df = build_feature_df_for_pool("FINAL_HOLDOUT_1000", holdout_df, cands_holdout, target_records_all)
    feat_holdout_df.to_parquet(data_dir / "final_holdout_1000_v3_features.parquet", index=False)

    v2c_model = joblib.load("experiments/v2/models/xgboost_matcher_v2_c.joblib")

    # Evaluate once on INTERNAL_CHECK_1000
    eval_check = evaluate_pool_single_shot(
        "INTERNAL_CHECK_1000", check_df, feat_check_df, ret_eval_check, bakeoff_data, v2c_model
    )

    # Evaluate once on FINAL_HOLDOUT_1000
    eval_holdout = evaluate_pool_single_shot(
        "FINAL_HOLDOUT_1000", holdout_df, feat_holdout_df, ret_eval_holdout, bakeoff_data, v2c_model
    )

    out_payload = {
        "frozen_pipeline_config": frozen_cfg,
        "retrieval_meta": ret_meta,
        "INTERNAL_CHECK_1000": eval_check,
        "FINAL_HOLDOUT_1000": eval_holdout,
        "elapsed_s": round(time.time() - t0, 2),
    }

    with open(res_dir / "v3_internal_and_holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nSaved single-shot firewall evaluation metrics to {res_dir / 'v3_internal_and_holdout_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
