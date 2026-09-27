"""V5 Offline Prototype & Feature/Retrieval Ablation Runner.

Strictly Read-Only on V1, V2, V3, and V4 artifacts (NO full-test inference):
  - Evaluates only on DEV_3000, INTERNAL_CHECK_1000, FINAL_HOLDOUT_1000, and COMBINED_5000.
  - Tunes strictly on DEV_3000 + INTERNAL_CHECK_1000; FINAL_HOLDOUT_1000 is an untouched evaluation gate.
  - Evaluates V5 5-Channel Targeted Candidate Rescue on top of unchanged V3/V4 Top-500 (`m500`),
    measuring S2 and S3 separately for:
      * previously unretrieved GT positives recovered (marginal & incremental)
      * candidate count added (S2, S3, total)
      * candidate recall (micro, macro, S2, S3)
      * runtime
  - Runs the 5-way Feature Ablation + Full V5 Rescue:
      A. V4 baseline
      B. V4 + building features (`V5_BUILDING_FEATURES`)
      C. V4 + abbreviation features (`V5_ABBREVIATION_FEATURES`)
      D. V4 + transliteration features (`V5_TRANSLITERATION_FEATURES`)
      E. V4 + all three (`V5_ALL_NEW_FEATURES`)
      F. V5 Full Prototype (`E + V5 5-Channel Retrieval Rescue`)
  - Reports for every pool and ablation:
      * candidate recall
      * F0.5, precision, recall
      * TP / FP / FN
      * S2 / S3 separately
      * India / US separately
      * incremental GT positives recovered by each retrieval channel
      * runtime & memory
      * feature importance / group ablation
      * strict gate pass/fail status (`FINAL_HOLDOUT_1000` >= +0.003 or `COMBINED_5000` >= +0.003)
"""

import json
import os
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Dict, List, Optional, Tuple
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path.cwd()))

from code.business_entity_resolution.src.run_v3_d_bakeoff import (
    apply_frozen_policy,
    evaluate_predictions_from_mask,
    prepare_eval_structures,
)
from code.business_entity_resolution.src.decoder_v4 import apply_v4_entity_policy_and_exclusivity
from code.business_entity_resolution.src.features_v5 import (
    V5_ABBREVIATION_FEATURES,
    V5_ALL_NEW_FEATURES,
    V5_BUILDING_FEATURES,
    V5_TRANSLITERATION_FEATURES,
    compute_v5_features_for_df,
    mine_or_load_challenge_lexicons,
    prepare_v5_entity_cache,
)
from code.business_entity_resolution.src.retrieval_v5 import (
    V5_RESCUE_CHANNEL_NAMES,
    evaluate_v5_targeted_rescue_for_pool,
)

SEED = 42
V5_ROOT = Path("experiments/v5")
V5_CACHE_DIR = V5_ROOT / "cache"
V5_MODELS_DIR = V5_ROOT / "models"
V5_RESULTS_DIR = V5_ROOT / "results"


def _load_s1_raw_lookup(qdf: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    out = {}
    for row in qdf.itertuples(index=False):
        out[str(row.source1_entity_id)] = {
            "business_name": "" if pd.isna(row.business_name) else str(row.business_name),
            "business_address": "" if pd.isna(row.business_address) else str(row.business_address),
            "country": "" if pd.isna(row.country) else str(row.country).strip().lower(),
        }
    return out


def compute_v5_group_logit_delta(
    df: pd.DataFrame,
    r_v4: np.ndarray,
    p_v4: np.ndarray,
    s1_slices: list,
    mode: str,
) -> np.ndarray:
    """Compute interpretable, feature-group-isolated logit delta for ablation modes A, B, C, D, E.
    Each mode uses ONLY its corresponding V5 feature group columns (+ existing V3/V4 base columns).
    Tuned strictly on DEV_3000 + INTERNAL_CHECK_1000 prior to FINAL_HOLDOUT_1000 gate evaluation.
    """
    n = len(df)
    delta_z = np.zeros(n, dtype=np.float64)
    if mode == "A":
        return delta_z

    use_bldg = mode in ("B", "E")
    use_abbr = mode in ("C", "E")
    use_tran = mode in ("D", "E")

    v3_bldg_conf = df["building_number_conflict"].to_numpy()
    v3_bldg_match = df["building_number_match"].to_numpy()
    v3_post_conf = df["postal_conflict"].to_numpy()
    v3_post_match = df["postal_match"].to_numpy()
    v3_unit_conf = df["unit_number_conflict"].to_numpy()
    v3_name_tset = df["core_name_token_set"].to_numpy()
    v3_addr_tset = df["norm_addr_token_set"].to_numpy()

    s1_rank = np.ones(n, dtype=np.int32)
    for st, en in s1_slices:
        if st < en:
            p_sl = p_v4[st:en]
            order = np.argsort(-p_sl, kind="mergesort")
            ranks = np.empty(en - st, dtype=np.int32)
            ranks[order] = np.arange(1, en - st + 1)
            s1_rank[st:en] = ranks

    if use_bldg:
        b_res = df["v5_bldg_conflict_resolved"].to_numpy()
        b_ind_res = df["v5_bldg_india_noisy_num_resolved"].to_numpy()
        b_min_c = df["v5_bldg_num_min_containment"].to_numpy()
        b_norm_ag = df["v5_bldg_norm_num_agreement"].to_numpy()
        b_loc_sim = df["v5_bldg_street_locality_sim"].to_numpy()
        b_ctx_ag = df["v5_bldg_context_agreement"].to_numpy()
        b_type_m = df["v5_bldg_type_match"].to_numpy()
        b_sev = df["v5_bldg_conflict_severity"].to_numpy()
        b_sev_fl = df["v5_bldg_severe_conflict_flag"].to_numpy()

        # 1. Positive boost for resolved false building conflicts (recovering building-conflict FNs)
        resolved_fn_mask = (
            (b_res > 0.5)
            & (p_v4 >= 0.980)
            & (b_loc_sim >= 0.72)
            & (v3_name_tset >= 0.88)
            & (v3_post_conf < 0.5)
            & (v3_unit_conf < 0.5)
        )
        delta_z[resolved_fn_mask] += (
            2.20
            + 0.65 * b_min_c[resolved_fn_mask]
            + 0.45 * b_loc_sim[resolved_fn_mask]
            + 0.30 * b_type_m[resolved_fn_mask]
            + 0.25 * b_ind_res[resolved_fn_mask]
        )

        # 2. Positive boost for strong typed slot & number agreement near threshold (no conflict)
        strong_bldg_near_thr = (
            (v3_bldg_conf < 0.5)
            & (v3_post_conf < 0.5)
            & (v3_unit_conf < 0.5)
            & (p_v4 >= 0.988)
            & (b_norm_ag >= 0.90)
            & (b_loc_sim >= 0.80)
            & (v3_name_tset >= 0.92)
        )
        delta_z[strong_bldg_near_thr] += (
            1.65 + 0.40 * b_type_m[strong_bldg_near_thr] + 0.30 * b_ctx_ag[strong_bldg_near_thr]
        )

        # 3. Penalty for severe unresolved building conflict on secondary branch candidates
        severe_bldg_fp_mask = (
            (b_sev_fl > 0.5)
            & (b_res < 0.5)
            & (b_min_c == 0.0)
            & (b_loc_sim < 0.30)
            & (p_v4 < 0.9996)
            & ((s1_rank >= 2) | (v3_post_match < 0.5))
        )
        delta_z[severe_bldg_fp_mask] -= 2.50 + 1.20 * b_sev[severe_bldg_fp_mask]

    if use_abbr:
        ab_n_jacc = df["v5_abbr_name_token_jaccard"].to_numpy()
        ab_n_tset = df["v5_abbr_name_token_set"].to_numpy()
        ab_a_jacc = df["v5_abbr_addr_token_jaccard"].to_numpy()
        ab_a_tset = df["v5_abbr_addr_token_set"].to_numpy()
        ab_acro = df["v5_abbr_acronym_compat"].to_numpy()
        ab_init = df["v5_abbr_initial_token_compat"].to_numpy()
        ab_br_jacc = df["v5_abbr_distinctive_brand_jaccard"].to_numpy()
        ab_br_conf = df["v5_abbr_distinctive_brand_conflict"].to_numpy()
        ab_gen_conf = df["v5_abbr_generic_name_diff_branch_conflict"].to_numpy()

        # 1. Positive boost for abbreviation-expanded name + address match near threshold
        abbr_near_thr_mask = (
            (p_v4 >= 0.988)
            & (v3_bldg_conf < 0.5)
            & (v3_post_conf < 0.5)
            & (v3_unit_conf < 0.5)
            & (ab_n_tset >= 0.95)
            & (ab_a_tset >= 0.88)
            & (ab_br_conf < 0.5)
        )
        delta_z[abbr_near_thr_mask] += (
            1.95
            + 0.45 * ab_n_jacc[abbr_near_thr_mask]
            + 0.45 * ab_a_jacc[abbr_near_thr_mask]
            + 0.30 * ab_br_jacc[abbr_near_thr_mask]
        )

        # 2. Positive boost for acronym / initial-token expansion with strong address match
        acro_rescue_mask = (
            (p_v4 >= 0.920)
            & (v3_bldg_conf < 0.5)
            & (v3_post_conf < 0.5)
            & (v3_unit_conf < 0.5)
            & ((ab_acro >= 0.90) | ((ab_init >= 0.85) & (ab_n_tset > v3_name_tset + 0.08)))
            & (ab_a_tset >= 0.88)
            & ((v3_bldg_match > 0.5) | (v3_post_match > 0.5))
        )
        delta_z[acro_rescue_mask] += 2.60 + 0.60 * ab_acro[acro_rescue_mask] + 0.40 * ab_a_tset[acro_rescue_mask]

        # 3. Penalty for generic/city-only name on conflicting branch address (wrong branch / same name FP)
        gen_branch_fp_mask = (
            (ab_gen_conf > 0.5)
            & (p_v4 < 0.9997)
            & ((s1_rank >= 2) | (ab_a_tset < 0.45))
        )
        delta_z[gen_branch_fp_mask] -= 3.20

        # 4. Penalty for distinctive brand conflict when address similarity is weak
        brand_conf_fp_mask = (
            (ab_br_conf > 0.5)
            & (ab_a_tset < 0.55)
            & (v3_bldg_match < 0.5)
            & (p_v4 < 0.9996)
        )
        delta_z[brand_conf_fp_mask] -= 2.80

    if use_tran:
        tr_na = df["v5_translit_has_non_ascii"].to_numpy()
        tr_n_jacc = df["v5_translit_name_token_jaccard"].to_numpy()
        tr_n_tset = df["v5_translit_name_token_set"].to_numpy()
        tr_ph_jacc = df["v5_translit_phonetic_token_jaccard"].to_numpy()
        tr_ph_edit = df["v5_translit_phonetic_edit_ratio"].to_numpy()
        tr_a_tset = df["v5_translit_addr_token_set"].to_numpy()
        tr_conf = df["v5_translit_brand_or_state_conflict"].to_numpy()

        # 1. Positive boost for Indic Unicode / transliterated true matches (where ASCII folding broke name_sim)
        indic_match_mask = (
            (tr_na > 0.5)
            & (tr_conf < 0.5)
            & (v3_bldg_conf < 0.5)
            & (v3_post_conf < 0.5)
            & (v3_unit_conf < 0.5)
            & (tr_n_tset >= 0.90)
            & (tr_n_jacc >= 0.65)
            & (tr_a_tset >= 0.85)
            & ((v3_bldg_match > 0.5) | ((v3_post_match > 0.5) & (tr_a_tset >= 0.92)))
            & (p_v4 >= 0.90)
        )
        delta_z[indic_match_mask] += (
            3.60
            + 0.80 * tr_n_tset[indic_match_mask]
            + 0.60 * tr_ph_edit[indic_match_mask]
            + 0.40 * tr_a_tset[indic_match_mask]
        )

        # 2. Positive boost for Romanized phonetic/transliteration equivalents near threshold
        roman_translit_mask = (
            (tr_na < 0.5)
            & (tr_conf < 0.5)
            & (v3_bldg_conf < 0.5)
            & (v3_post_conf < 0.5)
            & (v3_unit_conf < 0.5)
            & (p_v4 >= 0.985)
            & (
                (tr_n_tset > v3_name_tset + 0.04)
                | (tr_ph_edit > v3_name_tset + 0.08)
                | (tr_a_tset > v3_addr_tset + 0.06)
            )
            & (tr_ph_edit >= 0.90)
            & (tr_a_tset >= 0.85)
            & ((v3_bldg_match > 0.5) | (v3_post_match > 0.5))
        )
        delta_z[roman_translit_mask] += 2.10 + 0.45 * tr_ph_jacc[roman_translit_mask]

        # 3. Penalty for Indic brand conflict or explicit state/city mismatch
        translit_fp_mask = (
            (tr_conf > 0.5)
            & (v3_bldg_match < 0.5)
            & (p_v4 < 0.9998)
        )
        delta_z[translit_fp_mask] -= 3.50

    return delta_z


def evaluate_with_s2_s3_and_country_breakdown(
    pred_mask: np.ndarray,
    eval_struct: dict,
    qdf: pd.DataFrame,
    model_name: str,
) -> dict:
    """Compute overall Macro F0.5/P/R, TP/FP/FN, S2/S3 breakdown, AND India/US country breakdown."""
    base_eval = evaluate_predictions_from_mask(pred_mask, eval_struct, model_name)

    # Compute India and US macro F0.5 / P / R and TP / FP / FN separately
    s1_country_map = {}
    for row in qdf.itertuples(index=False):
        s1_country_map[str(row.source1_entity_id)] = (
            "" if pd.isna(row.country) else str(row.country).strip().lower()
        )

    s1_order = eval_struct["s1_order"]
    s1_slices = eval_struct["s1_row_slices"]
    s1_gt_sets = eval_struct["gt_sets"]
    label_col = eval_struct["label_col"]

    country_stats = {}
    for c_target in ("india", "us"):
        f05_list = []
        p_list = []
        r_list = []
        tp_c = 0
        fp_c = 0
        fn_c = 0
        n_s1_c = 0
        for s1_i, sid in enumerate(s1_order):
            if s1_country_map.get(sid) != c_target:
                continue
            n_s1_c += 1
            st, en = s1_slices[s1_i]
            gt_set = s1_gt_sets[s1_i]
            n_gt = len(gt_set)

            if st < en:
                p_sub = pred_mask[st:en]
                l_sub = label_col[st:en]
                tp_i = int((p_sub & (l_sub == 1)).sum())
                fp_i = int((p_sub & (l_sub == 0)).sum())
            else:
                tp_i = 0
                fp_i = 0

            fn_i = n_gt - tp_i
            tp_c += tp_i
            fp_c += fp_i
            fn_c += fn_i

            n_pred = tp_i + fp_i
            if n_gt == 0 and n_pred == 0:
                p_i, r_i, f05_i = 1.0, 1.0, 1.0
            elif n_gt == 0 and n_pred > 0:
                p_i, r_i, f05_i = 0.0, 1.0, 0.0
            elif n_gt > 0 and n_pred == 0:
                p_i, r_i, f05_i = 1.0, 0.0, 0.0
            else:
                p_i = tp_i / float(n_pred)
                r_i = tp_i / float(n_gt)
                denom = 0.25 * p_i + r_i
                f05_i = (1.25 * p_i * r_i / denom) if denom > 0 else 0.0

            p_list.append(p_i)
            r_list.append(r_i)
            f05_list.append(f05_i)

        country_stats[c_target] = {
            "s1_count": n_s1_c,
            "macro_f05": round(float(np.mean(f05_list)) if f05_list else 0.0, 6),
            "macro_precision": round(float(np.mean(p_list)) if p_list else 0.0, 6),
            "macro_recall": round(float(np.mean(r_list)) if r_list else 0.0, 6),
            "tp": int(tp_c),
            "fp": int(fp_c),
            "fn": int(fn_c),
        }

    base_eval["by_country"] = country_stats
    return base_eval


def main():
    t_start = time.perf_counter()
    tracemalloc.start()
    np.random.seed(SEED)

    for d in (V5_ROOT, V5_CACHE_DIR, V5_MODELS_DIR, V5_RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 96, flush=True)
    print("V5 OFFLINE PROTOTYPE & FEATURE/RETRIEVAL ABLATION (ISOLATED IN experiments/v5/, NO TEST INFERENCE)", flush=True)
    print("=" * 96, flush=True)

    # 1. Load Frozen V4 Model & Validation Pools
    v4_model = joblib.load("experiments/v4/models/v4_model.joblib")
    v3_ranker = v4_model["baseline_v3_ranker"]
    v3_calibrator = v4_model["baseline_v3_calibrator"]
    v3_active_cols: List[str] = v4_model["v3_active_features"]
    embed_retriever = v4_model["embedding_retriever"]
    policy_cfg: dict = v4_model["policy_config"]

    dev_qdf = pd.read_csv("experiments/v3/pools/dev_3000_s1.tsv", sep="\t", dtype=str)
    check_qdf = pd.read_csv("experiments/v3/pools/internal_check_1000_s1.tsv", sep="\t", dtype=str)
    holdout_qdf = pd.read_csv("experiments/v3/pools/final_holdout_1000_s1.tsv", sep="\t", dtype=str)
    combined_qdf = pd.concat([holdout_qdf, check_qdf, dev_qdf], ignore_index=True)

    dev_fdf = pd.read_parquet("experiments/v3/data/dev_3000_v3_features.parquet")
    check_fdf = pd.read_parquet("experiments/v3/data/internal_check_1000_v3_features.parquet")
    holdout_fdf = pd.read_parquet("experiments/v3/data/final_holdout_1000_v3_features.parquet")

    val_s1_ids = (
        set(dev_qdf["source1_entity_id"])
        | set(check_qdf["source1_entity_id"])
        | set(holdout_qdf["source1_entity_id"])
    )
    lex_info = mine_or_load_challenge_lexicons(forbidden_s1_ids=val_s1_ids)
    print(
        f"[Step 1] Verified entity-disjoint challenge lexicons: "
        f"{len(lex_info['name_abbrev'])} name abbrevs, {len(lex_info['addr_abbrev'])} addr abbrevs, "
        f"{len(lex_info['translit_map'])} transliteration mappings (0 validation entities touched).",
        flush=True,
    )

    b_2000 = joblib.load("experiments/v3/retrieval/check_and_holdout_2000_retrieval_bundle.joblib")
    b_3000 = joblib.load("experiments/v3/retrieval/dev_3000_retrieval_bundle.joblib")
    target_raw_lookup: Dict[str, Dict[str, str]] = {}
    target_raw_lookup.update(b_3000.get("target_records_all") or b_3000.get("target_records") or {})
    target_raw_lookup.update(b_2000.get("target_records_all") or b_2000.get("target_records") or {})
    del b_2000, b_3000

    s1_raw_lookup = _load_s1_raw_lookup(combined_qdf)

    # 2. Build or Load V5 Retrieval + Feature Bundles for DEV_3000, INTERNAL_CHECK_1000, FINAL_HOLDOUT_1000
    print("\n[Step 2] Running V5 5-Channel Retrieval Rescue & Extracting 32 V5 Features on Validation Pools...", flush=True)
    s1_cache_all: Optional[Dict[str, dict]] = None

    pool_bundles = {}
    for p_name, qdf, fdf in [
        ("DEV_3000", dev_qdf, dev_fdf),
        ("INTERNAL_CHECK_1000", check_qdf, check_fdf),
        ("FINAL_HOLDOUT_1000", holdout_qdf, holdout_fdf),
    ]:
        t_p0 = time.perf_counter()
        cache_path = V5_CACHE_DIR / f"{p_name}_v5_bundle.joblib"
        es_full = prepare_eval_structures(qdf, fdf)
        r_all = v3_ranker.predict(fdf[v3_active_cols])
        p_all = v3_calibrator.predict_proba(r_all.reshape(-1, 1))[:, 1]

        if cache_path.exists():
            cb = joblib.load(cache_path)
            m_v5 = cb["m_v5"]
            m_v4 = cb["m_v4"]
            m500 = cb["m500"]
            v5_ret_stats = cb["v5_ret_stats"]
            fdf_v5 = cb["fdf_v5"]
        else:
            if s1_cache_all is None:
                s1_cache_all = prepare_v5_entity_cache(list(s1_raw_lookup.keys()), s1_raw_lookup)
            m_v5, m_v4, m500, v5_ret_stats = evaluate_v5_targeted_rescue_for_pool(
                fdf,
                es_full,
                s1_lookup=s1_raw_lookup,
                target_lookup=target_raw_lookup,
                embed_retriever=embed_retriever,
                base_top_k=500,
                s1_cache=s1_cache_all,
            )
            fdf_v5 = fdf[m_v5].reset_index(drop=True)
            p_v5_base = p_all[m_v5]
            in_top500_v5 = m500[m_v5]
            front_mask_v5 = (p_v5_base >= 0.01) | (~in_top500_v5)
            front_idx_v5 = np.flatnonzero(front_mask_v5)
            target_cache_v5 = prepare_v5_entity_cache(
                fdf_v5.iloc[front_idx_v5]["candidate_entity_id"].tolist(),
                target_raw_lookup,
            )
            front_df_v5 = compute_v5_features_for_df(
                fdf_v5.iloc[front_idx_v5].reset_index(drop=True),
                s1_lookup=s1_raw_lookup,
                target_lookup=target_raw_lookup,
                s1_cache=s1_cache_all,
                target_cache=target_cache_v5,
            )
            for col in V5_ALL_NEW_FEATURES:
                arr = np.zeros(len(fdf_v5), dtype=np.float32)
                arr[front_idx_v5] = front_df_v5[col].to_numpy(dtype=np.float32)
                fdf_v5[col] = arr
            joblib.dump(
                {
                    "m_v5": m_v5,
                    "m_v4": m_v4,
                    "m500": m500,
                    "v5_ret_stats": v5_ret_stats,
                    "fdf_v5": fdf_v5,
                },
                cache_path,
            )

        pool_bundles[p_name] = {
            "qdf": qdf,
            "fdf": fdf,
            "es_full": es_full,
            "r_all": r_all,
            "p_all": p_all,
            "m_v5": m_v5,
            "m_v4": m_v4,
            "m500": m500,
            "v5_ret_stats": v5_ret_stats,
            "fdf_v5": fdf_v5,
            "prep_runtime_sec": round(time.perf_counter() - t_p0, 3),
        }
        v3_r = v5_ret_stats["v3_top500_baseline"]
        v4_r = v5_ret_stats["v4_rescued_baseline"]
        v5_r = v5_ret_stats["v5_rescued"]
        inc_top = v5_ret_stats["v5_vs_top500"]
        inc_v4 = v5_ret_stats["v5_vs_v4_incremental"]
        print(
            f"  [{p_name:20s}] Top-500={v3_r['micro_recall']*100:.2f}% (S2={v3_r['s2_recall']*100:.2f}%, S3={v3_r['s3_recall']*100:.2f}%)"
            f" -> V4={v4_r['micro_recall']*100:.2f}% -> V5={v5_r['micro_recall']*100:.2f}% (S2={v5_r['s2_recall']*100:.2f}%, S3={v5_r['s3_recall']*100:.2f}%)"
            f" | +GT vs Top500={inc_top['gt_positives_recovered_total']} (S2=+{inc_top['gt_positives_recovered_s2']}, S3=+{inc_top['gt_positives_recovered_s3']})"
            f" | +GT vs V4={inc_v4['gt_positives_recovered_total']} (S2=+{inc_v4['gt_positives_recovered_s2']}, S3=+{inc_v4['gt_positives_recovered_s3']})",
            flush=True,
        )

    # Assemble COMBINED_5000 bundle from the 3 disjoint pools (order: FINAL_HOLDOUT_1000, INTERNAL_CHECK_1000, DEV_3000)
    comb_order = ["FINAL_HOLDOUT_1000", "INTERNAL_CHECK_1000", "DEV_3000"]
    combined_fdf = pd.concat([pool_bundles[p]["fdf"] for p in comb_order], ignore_index=True)
    combined_es_full = prepare_eval_structures(combined_qdf, combined_fdf)
    combined_r_all = np.concatenate([pool_bundles[p]["r_all"] for p in comb_order])
    combined_p_all = np.concatenate([pool_bundles[p]["p_all"] for p in comb_order])
    combined_m_v5 = np.concatenate([pool_bundles[p]["m_v5"] for p in comb_order])
    combined_m_v4 = np.concatenate([pool_bundles[p]["m_v4"] for p in comb_order])
    combined_m500 = np.concatenate([pool_bundles[p]["m500"] for p in comb_order])
    combined_fdf_v5 = pd.concat([pool_bundles[p]["fdf_v5"] for p in comb_order], ignore_index=True)

    # Aggregate COMBINED_5000 retrieval stats across the 3 pools
    from code.business_entity_resolution.src.candidate_rescue_v4 import compute_candidate_recall_from_mask
    comb_v3_rec = compute_candidate_recall_from_mask(combined_m500, combined_es_full)
    comb_v4_rec = compute_candidate_recall_from_mask(combined_m_v4, combined_es_full)
    comb_v5_rec = compute_candidate_recall_from_mask(combined_m_v5, combined_es_full)
    comb_marg_ch = {}
    comb_inc_ch = {}
    for ch in V5_RESCUE_CHANNEL_NAMES:
        m_list = [pool_bundles[p]["v5_ret_stats"]["marginal_by_channel_vs_top500"][ch] for p in comb_order]
        i_list = [pool_bundles[p]["v5_ret_stats"]["incremental_by_channel_sequence"][ch] for p in comb_order]
        comb_marg_ch[ch] = {
            "candidates_added_total": sum(x["candidates_added_total"] for x in m_list),
            "candidates_added_s2": sum(x["candidates_added_s2"] for x in m_list),
            "candidates_added_s3": sum(x["candidates_added_s3"] for x in m_list),
            "gt_positives_recovered_total": sum(x["gt_positives_recovered_total"] for x in m_list),
            "gt_positives_recovered_s2": sum(x["gt_positives_recovered_s2"] for x in m_list),
            "gt_positives_recovered_s3": sum(x["gt_positives_recovered_s3"] for x in m_list),
            "runtime_sec": round(sum(x["runtime_sec"] for x in m_list), 4),
        }
        comb_inc_ch[ch] = {
            "incremental_candidates_added_total": sum(x["incremental_candidates_added_total"] for x in i_list),
            "incremental_candidates_added_s2": sum(x["incremental_candidates_added_s2"] for x in i_list),
            "incremental_candidates_added_s3": sum(x["incremental_candidates_added_s3"] for x in i_list),
            "incremental_gt_positives_recovered_total": sum(x["incremental_gt_positives_recovered_total"] for x in i_list),
            "incremental_gt_positives_recovered_s2": sum(x["incremental_gt_positives_recovered_s2"] for x in i_list),
            "incremental_gt_positives_recovered_s3": sum(x["incremental_gt_positives_recovered_s3"] for x in i_list),
        }

    comb_ret_stats = {
        "v3_top500_baseline": comb_v3_rec,
        "v4_rescued_baseline": comb_v4_rec,
        "v5_rescued": comb_v5_rec,
        "v5_vs_top500": {
            "candidates_added_total": sum(pool_bundles[p]["v5_ret_stats"]["v5_vs_top500"]["candidates_added_total"] for p in comb_order),
            "candidates_added_s2": sum(pool_bundles[p]["v5_ret_stats"]["v5_vs_top500"]["candidates_added_s2"] for p in comb_order),
            "candidates_added_s3": sum(pool_bundles[p]["v5_ret_stats"]["v5_vs_top500"]["candidates_added_s3"] for p in comb_order),
            "gt_positives_recovered_total": int(comb_v5_rec["retrieved_gt_links"] - comb_v3_rec["retrieved_gt_links"]),
            "gt_positives_recovered_s2": int(comb_v5_rec["s2_retrieved_gt"] - comb_v3_rec["s2_retrieved_gt"]),
            "gt_positives_recovered_s3": int(comb_v5_rec["s3_retrieved_gt"] - comb_v3_rec["s3_retrieved_gt"]),
        },
        "v5_vs_v4_incremental": {
            "candidates_added_total": sum(pool_bundles[p]["v5_ret_stats"]["v5_vs_v4_incremental"]["candidates_added_total"] for p in comb_order),
            "candidates_added_s2": sum(pool_bundles[p]["v5_ret_stats"]["v5_vs_v4_incremental"]["candidates_added_s2"] for p in comb_order),
            "candidates_added_s3": sum(pool_bundles[p]["v5_ret_stats"]["v5_vs_v4_incremental"]["candidates_added_s3"] for p in comb_order),
            "gt_positives_recovered_total": int(comb_v5_rec["retrieved_gt_links"] - comb_v4_rec["retrieved_gt_links"]),
            "gt_positives_recovered_s2": int(comb_v5_rec["s2_retrieved_gt"] - comb_v4_rec["s2_retrieved_gt"]),
            "gt_positives_recovered_s3": int(comb_v5_rec["s3_retrieved_gt"] - comb_v4_rec["s3_retrieved_gt"]),
        },
        "marginal_by_channel_vs_top500": comb_marg_ch,
        "incremental_by_channel_sequence": comb_inc_ch,
        "total_rescue_runtime_sec": round(sum(pool_bundles[p]["v5_ret_stats"]["total_rescue_runtime_sec"] for p in comb_order), 3),
    }

    pool_bundles["COMBINED_5000"] = {
        "qdf": combined_qdf,
        "fdf": combined_fdf,
        "es_full": combined_es_full,
        "r_all": combined_r_all,
        "p_all": combined_p_all,
        "m_v5": combined_m_v5,
        "m_v4": combined_m_v4,
        "m500": combined_m500,
        "v5_ret_stats": comb_ret_stats,
        "fdf_v5": combined_fdf_v5,
        "prep_runtime_sec": round(sum(pool_bundles[p]["prep_runtime_sec"] for p in comb_order), 3),
    }

    # 3. Fit Lightweight V5 GBDT Residual Auditor on Allowed Tuning Frontier (DEV_3000 + INTERNAL_CHECK_1000)
    #    Strictly excluding FINAL_HOLDOUT_1000 to measure tree feature importances (Gain & Split) without holdout leakage.
    print("\n[Step 3] Fitting Lightweight V5 Feature-Importance Auditor (LightGBM on DEV_3000 + INTERNAL_CHECK_1000 active frontier)...", flush=True)
    tune_dfs = []
    for p_tune in ("DEV_3000", "INTERNAL_CHECK_1000"):
        pb = pool_bundles[p_tune]
        in_v4 = pb["m_v4"][pb["m_v5"]]
        df_t = pb["fdf_v5"][in_v4].copy()
        p_t = pb["p_all"][pb["m_v4"]]
        df_t = df_t[p_t >= 0.01].reset_index(drop=True)
        tune_dfs.append(df_t)
    tune_front_df = pd.concat(tune_dfs, ignore_index=True)

    v5_lgb_auditor = lgb.LGBMClassifier(
        n_estimators=120,
        num_leaves=15,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=40,
        reg_alpha=2.0,
        reg_lambda=5.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    v5_lgb_auditor.fit(
        tune_front_df[V5_ALL_NEW_FEATURES],
        tune_front_df["label"].to_numpy(dtype=np.int32),
    )
    gains = v5_lgb_auditor.booster_.feature_importance(importance_type="gain")
    splits = v5_lgb_auditor.booster_.feature_importance(importance_type="split")
    total_gain = float(np.sum(gains)) if np.sum(gains) > 0 else 1.0

    feature_importance_list = []
    for col, g, s in sorted(zip(V5_ALL_NEW_FEATURES, gains, splits), key=lambda x: -x[1]):
        if col in V5_BUILDING_FEATURES:
            grp = "1_building_number"
        elif col in V5_ABBREVIATION_FEATURES:
            grp = "2_abbreviation_brand"
        else:
            grp = "3_transliteration"
        feature_importance_list.append(
            {
                "feature": col,
                "group": grp,
                "gain": round(float(g), 2),
                "gain_pct": round(100.0 * float(g) / total_gain, 2),
                "splits": int(s),
            }
        )

    group_gain_summary = {}
    for grp_name, cols in [
        ("1_building_number", V5_BUILDING_FEATURES),
        ("2_abbreviation_brand", V5_ABBREVIATION_FEATURES),
        ("3_transliteration", V5_TRANSLITERATION_FEATURES),
    ]:
        g_sum = sum(x["gain"] for x in feature_importance_list if x["group"] == grp_name)
        s_sum = sum(x["splits"] for x in feature_importance_list if x["group"] == grp_name)
        group_gain_summary[grp_name] = {
            "num_features": len(cols),
            "total_gain": round(g_sum, 2),
            "gain_share_pct": round(100.0 * g_sum / total_gain, 2),
            "total_splits": int(s_sum),
        }

    # 4. Run Full Feature Ablation (A..E) + V5 Retrieval Rescue (F) Across All 4 Pools
    print("\n" + "=" * 96, flush=True)
    print("STEP 4: V5 RANKER FEATURE ABLATION (A..E) + V5 TARGETED RETRIEVAL RESCUE (F) ACROSS ALL 4 POOLS", flush=True)
    print("=" * 96, flush=True)

    ablation_configs = [
        ("A_V4_Baseline", "A", False, "A. V4 Baseline"),
        ("B_V4_plus_Building", "B", False, "B. V4 + Building Features"),
        ("C_V4_plus_Abbreviation", "C", False, "C. V4 + Abbreviation Features"),
        ("D_V4_plus_Transliteration", "D", False, "D. V4 + Transliteration Features"),
        ("E_V4_plus_All_Three", "E", False, "E. V4 + All Three (B+C+D)"),
        ("F_V5_Full_Prototype_with_Rescue", "E", True, "F. V5 Full (E + V5 5-Ch Rescue)"),
    ]

    all_pool_results = {}

    for p_name in ["DEV_3000", "INTERNAL_CHECK_1000", "FINAL_HOLDOUT_1000", "COMBINED_5000"]:
        pb = pool_bundles[p_name]
        qdf = pb["qdf"]
        r_all = pb["r_all"]
        p_all = pb["p_all"]
        m_v5 = pb["m_v5"]
        m_v4 = pb["m_v4"]
        m500 = pb["m500"]
        fdf_v5 = pb["fdf_v5"]

        in_v4_v5 = m_v4[m_v5]
        fdf_v4 = fdf_v5[in_v4_v5].reset_index(drop=True)
        r_v4 = r_all[m_v4]
        p_v4 = p_all[m_v4]
        in_top500_v4 = m500[m_v4]
        es_v4 = prepare_eval_structures(qdf, fdf_v4)

        r_v5_base = r_all[m_v5]
        p_v5_base = p_all[m_v5]
        in_top500_v5 = m500[m_v5]
        es_v5 = prepare_eval_structures(qdf, fdf_v5)

        print(f"\n--- Pool: {p_name} (S1={len(qdf):,}) ---", flush=True)
        pool_ablations = {}
        base_metrics = None

        for key_name, mode, use_v5_rescue, label in ablation_configs:
            t_ab = time.perf_counter()
            if not use_v5_rescue:
                dz = compute_v5_group_logit_delta(fdf_v4, r_v4, p_v4, es_v4["s1_row_slices"], mode)
                r_mod = r_v4 + dz
                p_mod = v3_calibrator.predict_proba(r_mod.reshape(-1, 1))[:, 1]
                mask_pred, dec_stats = apply_v4_entity_policy_and_exclusivity(
                    probs=p_mod,
                    feat_df=fdf_v4,
                    eval_struct=es_v4,
                    policy_cfg=policy_cfg,
                    s1_raw_lookup=s1_raw_lookup,
                    target_raw_lookup=target_raw_lookup,
                    in_top500_mask=in_top500_v4,
                    rescued_min_prob=0.9990,
                    exclusivity_margin=0.05,
                )
                ev = evaluate_with_s2_s3_and_country_breakdown(mask_pred, es_v4, qdf, label)
            else:
                dz_v5 = compute_v5_group_logit_delta(fdf_v5, r_v5_base, p_v5_base, es_v5["s1_row_slices"], mode)
                new_v5_tail = ~in_v4_v5
                strict_tail_confirm = (
                    new_v5_tail
                    & (p_v5_base >= 0.960)
                    & (fdf_v5["building_number_conflict"].to_numpy() < 0.5)
                    & (fdf_v5["postal_conflict"].to_numpy() < 0.5)
                    & (fdf_v5["unit_number_conflict"].to_numpy() < 0.5)
                    & (fdf_v5["v5_bldg_norm_num_agreement"].to_numpy() >= 0.90)
                    & (fdf_v5["v5_bldg_street_locality_sim"].to_numpy() >= 0.80)
                    & (
                        (fdf_v5["v5_abbr_name_token_set"].to_numpy() >= 0.92)
                        | (fdf_v5["v5_translit_name_token_set"].to_numpy() >= 0.92)
                        | (fdf_v5["v5_abbr_acronym_compat"].to_numpy() >= 0.95)
                    )
                )
                dz_v5[strict_tail_confirm] += 3.20
                r_v5_mod = r_v5_base + dz_v5
                p_v5_mod = v3_calibrator.predict_proba(r_v5_mod.reshape(-1, 1))[:, 1]
                mask_pred, dec_stats = apply_v4_entity_policy_and_exclusivity(
                    probs=p_v5_mod,
                    feat_df=fdf_v5,
                    eval_struct=es_v5,
                    policy_cfg=policy_cfg,
                    s1_raw_lookup=s1_raw_lookup,
                    target_raw_lookup=target_raw_lookup,
                    in_top500_mask=in_top500_v5,
                    rescued_min_prob=0.9990,
                    exclusivity_margin=0.05,
                )
                ev = evaluate_with_s2_s3_and_country_breakdown(mask_pred, es_v5, qdf, label)

            ev["eval_runtime_sec"] = round(time.perf_counter() - t_ab, 3)
            ev["decoder_stats"] = dec_stats

            if base_metrics is None:
                base_metrics = ev

            delta_f05 = round(ev["macro_f05"] - base_metrics["macro_f05"], 6)
            delta_p = round(ev["macro_precision"] - base_metrics["macro_precision"], 6)
            delta_r = round(ev["macro_recall"] - base_metrics["macro_recall"], 6)
            delta_tp = int(ev["tp"] - base_metrics["tp"])
            delta_fp = int(ev["fp"] - base_metrics["fp"])
            delta_fn = int(ev["fn"] - base_metrics["fn"])

            ev["delta_vs_v4_baseline"] = {
                "delta_macro_f05": delta_f05,
                "delta_macro_precision": delta_p,
                "delta_macro_recall": delta_r,
                "delta_tp": delta_tp,
                "delta_fp": delta_fp,
                "delta_fn": delta_fn,
                "delta_s2_f05": round(ev["s2_metrics"]["micro_f05"] - base_metrics["s2_metrics"]["micro_f05"], 6),
                "delta_s3_f05": round(ev["s3_metrics"]["micro_f05"] - base_metrics["s3_metrics"]["micro_f05"], 6),
                "delta_india_f05": round(
                    ev["by_country"]["india"]["macro_f05"] - base_metrics["by_country"]["india"]["macro_f05"], 6
                ),
                "delta_us_f05": round(
                    ev["by_country"]["us"]["macro_f05"] - base_metrics["by_country"]["us"]["macro_f05"], 6
                ),
            }
            pool_ablations[key_name] = ev

            print(
                f"  {label:34s} | F0.5={ev['macro_f05']:.6f} ({delta_f05:+.6f}) | "
                f"P={ev['macro_precision']*100:.2f}% | R={ev['macro_recall']*100:.2f}% | "
                f"TP={ev['tp']} ({delta_tp:+d}) FP={ev['fp']} ({delta_fp:+d}) FN={ev['fn']} ({delta_fn:+d}) | "
                f"S2_F0.5={ev['s2_metrics']['micro_f05']:.5f} S3_F0.5={ev['s3_metrics']['micro_f05']:.5f} | "
                f"IN_F0.5={ev['by_country']['india']['macro_f05']:.5f} US_F0.5={ev['by_country']['us']['macro_f05']:.5f}",
                flush=True,
            )

        all_pool_results[p_name] = {
            "retrieval": pb["v5_ret_stats"],
            "ablations": pool_ablations,
        }

    # 5. Evaluate Strict Gates (Requirement 6)
    v4_hld = all_pool_results["FINAL_HOLDOUT_1000"]["ablations"]["A_V4_Baseline"]
    v5_hld = all_pool_results["FINAL_HOLDOUT_1000"]["ablations"]["F_V5_Full_Prototype_with_Rescue"]
    v4_chk = all_pool_results["INTERNAL_CHECK_1000"]["ablations"]["A_V4_Baseline"]
    v5_chk = all_pool_results["INTERNAL_CHECK_1000"]["ablations"]["F_V5_Full_Prototype_with_Rescue"]
    v4_dev = all_pool_results["DEV_3000"]["ablations"]["A_V4_Baseline"]
    v5_dev = all_pool_results["DEV_3000"]["ablations"]["F_V5_Full_Prototype_with_Rescue"]
    v4_cmb = all_pool_results["COMBINED_5000"]["ablations"]["A_V4_Baseline"]
    v5_cmb = all_pool_results["COMBINED_5000"]["ablations"]["F_V5_Full_Prototype_with_Rescue"]

    d_hld = round(v5_hld["macro_f05"] - v4_hld["macro_f05"], 6)
    d_chk = round(v5_chk["macro_f05"] - v4_chk["macro_f05"], 6)
    d_dev = round(v5_dev["macro_f05"] - v4_dev["macro_f05"], 6)
    d_cmb = round(v5_cmb["macro_f05"] - v4_cmb["macro_f05"], 6)

    strict_gates = {
        "no_final_holdout_f05_loss": bool(d_hld >= 0.0),
        "no_internal_check_f05_loss": bool(d_chk >= 0.0),
        "no_substantial_precision_loss": bool(
            (v5_hld["macro_precision"] >= v4_hld["macro_precision"] - 0.002)
            and (v5_chk["macro_precision"] >= v4_chk["macro_precision"] - 0.002)
            and (v5_cmb["macro_precision"] >= v4_cmb["macro_precision"] - 0.002)
        ),
        "not_dev_only_gain": bool(d_hld > 0.0 and d_chk > 0.0),
        "target_plus_0003_final_holdout": bool(d_hld >= 0.003),
        "target_plus_0003_combined_5000": bool(d_cmb >= 0.003),
        "delta_final_holdout_f05": d_hld,
        "delta_internal_check_f05": d_chk,
        "delta_dev_3000_f05": d_dev,
        "delta_combined_5000_f05": d_cmb,
    }
    strict_gates["OVERALL_V5_GATE_PASS"] = bool(
        strict_gates["no_final_holdout_f05_loss"]
        and strict_gates["no_internal_check_f05_loss"]
        and strict_gates["no_substantial_precision_loss"]
        and strict_gates["not_dev_only_gain"]
        and (strict_gates["target_plus_0003_final_holdout"] or strict_gates["target_plus_0003_combined_5000"])
    )

    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_runtime = round(time.perf_counter() - t_start, 2)

    print("\n" + "=" * 96, flush=True)
    print("STEP 5: STRICT GATE AUDIT (FINAL_HOLDOUT_1000 & COMBINED_5000)", flush=True)
    print("=" * 96, flush=True)
    for k, v in strict_gates.items():
        print(f"  {k:38s}: {v}", flush=True)
    print(f"  Total Runtime: {total_runtime:.2f}s | Peak Python Heap Allocation: {peak_mem / (1024**2):.2f} MB", flush=True)

    # Save V5 Prototype Model & Report
    joblib.dump(
        {
            "version": "v5_offline_prototype",
            "seed": SEED,
            "v5_feature_groups": {
                "building": V5_BUILDING_FEATURES,
                "abbreviation": V5_ABBREVIATION_FEATURES,
                "transliteration": V5_TRANSLITERATION_FEATURES,
            },
            "lgb_auditor": v5_lgb_auditor,
            "strict_gates": strict_gates,
        },
        V5_MODELS_DIR / "v5_prototype_model.joblib",
    )

    report_payload = {
        "experiment": "V5_Offline_Prototype_and_Ablation",
        "frozen_v4_leaderboard_baseline": 0.917,
        "seed": SEED,
        "total_runtime_sec": total_runtime,
        "peak_memory_mb": round(peak_mem / (1024**2), 2),
        "strict_gates": strict_gates,
        "feature_importance": {
            "by_group": group_gain_summary,
            "all_32_v5_features": feature_importance_list,
        },
        "pools": all_pool_results,
    }
    report_path = V5_RESULTS_DIR / "v5_prototype_ablation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\nSaved complete V5 prototype report to: {report_path}", flush=True)


if __name__ == "__main__":
    main()
