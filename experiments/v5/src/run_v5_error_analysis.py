"""V5 Comprehensive Error Analysis & Decoder-Only Improvement Exploration.

Strictly Read-Only on V1, V2, V3, and V4 artifacts:
  - Reconstructs V4 predictions exactly on DEV_3000, INTERNAL_CHECK_1000, FINAL_HOLDOUT_1000, and COMBINED_5000.
  - Identifies every TP, FP, and FN (both in-pool ranking/threshold FNs and unretrieved blocking FNs).
  - Computes all 18 required diagnostic signals for every FP and FN.
  - Categorizes all FPs and FNs into the 12 interpretable error categories and extracts the Top 20 FP and FN patterns.
  - Separately analyzes:
      1. Singleton / empty S1 predictions
      2. Predictions near the probability threshold ([0.90, 1.00))
      3. Target conflicts before V4 decoding (within-pool and full-catalog)
      4. Conflicts where V4.1-C(0.05) selected the wrong owner
      5. FPs where another candidate for the same S1 has stronger structural evidence
  - Tests cheap decoder-only alternatives tuned strictly on DEV_3000 + INTERNAL_CHECK_1000
    (rejecting any change that improves DEV_3000 but regresses INTERNAL_CHECK_1000),
    using FINAL_HOLDOUT_1000 only as a final gate.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set, Tuple
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path.cwd()))

from code.business_entity_resolution.src.preprocessing import parse_match_ids
from experiments.v3.src.run_v3_d_bakeoff import (
    apply_frozen_policy,
    evaluate_predictions_from_mask,
    prepare_eval_structures,
)
from experiments.v4.src.candidate_rescue_v4 import select_v4_rescued_mask_for_pool
from experiments.v4.src.decoder_v4 import (
    _logit_norm,
    apply_v4_entity_policy_and_exclusivity,
    compute_v41c_assignment_score,
    extract_v4_owner_record,
    resolve_v41c_conflict,
)
from experiments.v4.src.features_v4 import (
    _POPCOUNT_TABLE,
    compute_v4_competition_features,
    compute_v4_extra_pair_features,
)

V5_ROOT = Path("experiments/v5")
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


def classify_fp_category(row_dict: dict) -> str:
    """Assign each FP to its primary interpretable error category (out of the 12 required categories)."""
    if row_dict["target_mult_before_v4"] > 1 or row_dict["has_true_owner_in_full_catalog"]:
        # Also note underlying structural conflict if present
        pass
    if row_dict["building_number_conflict"] > 0.5:
        return "building-number conflict"
    if row_dict["postal_conflict"] > 0.5:
        return "postal conflict"
    if row_dict["unit_number_conflict"] > 0.5:
        return "unit conflict"
    if row_dict["v4_numeric_token_conflict"] > 0.5:
        return "numeric conflict"
    if row_dict["core_name_token_set"] >= 0.90 and row_dict["norm_addr_token_set"] < 0.70:
        return "wrong branch / same name"
    if row_dict["norm_addr_token_set"] < 0.60:
        return "wrong address"
    if row_dict["core_name_token_set"] < 0.65 and row_dict["norm_addr_token_set"] >= 0.85:
        return "name collision"
    if row_dict["v4_exact_acronym"] > 0.5 or (
        row_dict["v4_abbrev_name_token_jaccard"] > row_dict["core_name_token_jaccard"] + 0.05
    ):
        return "abbreviation"
    if row_dict["phonetic_name_token_jaccard"] > row_dict["core_name_token_jaccard"] + 0.10:
        return "multilingual/transliteration"
    if row_dict["has_true_owner_in_full_catalog"] or row_dict["target_competitor_count"] > 1:
        return "target competition"
    return "other"


def classify_fn_category(row_dict: Optional[dict], s1_rec: dict, tgt_rec: dict, removed_by_excl: bool) -> str:
    """Assign each FN to its primary interpretable error category."""
    if row_dict is None:
        return "retrieval failure"
    if removed_by_excl:
        return "target competition"
    if row_dict["building_number_conflict"] > 0.5:
        return "building-number conflict"
    if row_dict["postal_conflict"] > 0.5:
        return "postal conflict"
    if row_dict["unit_number_conflict"] > 0.5:
        return "unit conflict"
    if row_dict["v4_numeric_token_conflict"] > 0.5:
        return "numeric conflict"
    if row_dict["phonetic_name_token_jaccard"] > row_dict["core_name_token_jaccard"] + 0.12 or (
        row_dict["norm_name_ratio"] > row_dict["name_ratio"] + 0.15
    ):
        return "multilingual/transliteration"
    if (
        row_dict["v4_exact_acronym"] > 0.5
        or row_dict["v4_abbrev_name_token_jaccard"] > row_dict["core_name_token_jaccard"] + 0.05
        or row_dict["core_name_min_containment"] >= 0.80
        and row_dict["core_name_token_jaccard"] < 0.50
    ):
        return "abbreviation"
    if row_dict["core_name_token_set"] >= 0.85 and row_dict["norm_addr_token_set"] < 0.60:
        return "wrong address"
    if row_dict["core_name_token_set"] < 0.60 and row_dict["norm_addr_token_set"] >= 0.80:
        return "name collision"
    if row_dict["s1_rank"] > 6:
        return "wrong branch / same name"
    if row_dict["target_competitor_count"] > 1:
        return "target competition"
    return "other"


def main():
    t0 = time.time()
    V5_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88, flush=True)
    print("V5 ERROR ANALYSIS ON FROZEN V4 BASELINE (READ-ONLY, NO FULL TEST INFERENCE)", flush=True)
    print("=" * 88, flush=True)

    # 1. Load frozen V4 model & validation pools
    v4_model = joblib.load("experiments/v4/models/v4_model.joblib")
    v3_ranker = v4_model["baseline_v3_ranker"]
    v3_calibrator = v4_model["baseline_v3_calibrator"]
    v3_active_cols: List[str] = v4_model["v3_active_features"]
    embed_retriever = v4_model["embedding_retriever"]
    policy_cfg: dict = v4_model["policy_config"]
    rescued_min_prob = float(v4_model.get("rescued_min_prob", 0.9990))
    exclusivity_margin = float(v4_model.get("decoder_margin", 0.05))

    dev_qdf = pd.read_csv("experiments/v3/pools/dev_3000_s1.tsv", sep="\t", dtype=str)
    check_qdf = pd.read_csv("experiments/v3/pools/internal_check_1000_s1.tsv", sep="\t", dtype=str)
    holdout_qdf = pd.read_csv("experiments/v3/pools/final_holdout_1000_s1.tsv", sep="\t", dtype=str)
    combined_qdf = pd.concat([holdout_qdf, check_qdf, dev_qdf], ignore_index=True)

    dev_fdf = pd.read_parquet("experiments/v3/data/dev_3000_v3_features.parquet")
    check_fdf = pd.read_parquet("experiments/v3/data/internal_check_1000_v3_features.parquet")
    holdout_fdf = pd.read_parquet("experiments/v3/data/final_holdout_1000_v3_features.parquet")

    b_2000 = joblib.load("experiments/v3/retrieval/check_and_holdout_2000_retrieval_bundle.joblib")
    b_3000 = joblib.load("experiments/v3/retrieval/dev_3000_retrieval_bundle.joblib")
    target_raw_lookup: Dict[str, Dict[str, str]] = {}
    target_raw_lookup.update(b_3000.get("target_records_all") or b_3000.get("target_records") or {})
    target_raw_lookup.update(b_2000.get("target_records_all") or b_2000.get("target_records") or {})
    del b_2000, b_3000

    s1_raw_lookup = _load_s1_raw_lookup(combined_qdf)

    # Load full-catalog ground truth target -> true S1 owner map for FP diagnostic
    print("[Step 1] Loading train_ground_truth.tsv target->true_S1 map for full-catalog FP analysis...", flush=True)
    target_to_true_s1: Dict[str, str] = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8", buffering=8 << 20) as f:
        next(f, None)
        for line in f:
            s1_id, _, rest = line.rstrip("\r\n").partition("\t")
            if rest:
                for tid in rest.split(","):
                    target_to_true_s1[tid] = s1_id

    # Score the 3 pools with V3 ranker + calibrator once
    pool_raw_prob = {}
    for p_name, fdf in [
        ("FINAL_HOLDOUT_1000", holdout_fdf),
        ("INTERNAL_CHECK_1000", check_fdf),
        ("DEV_3000", dev_fdf),
    ]:
        r_v3 = v3_ranker.predict(fdf[v3_active_cols])
        p_v3 = v3_calibrator.predict_proba(r_v3.reshape(-1, 1))[:, 1]
        pool_raw_prob[p_name] = (r_v3, p_v3)

    combined_fdf = pd.concat([holdout_fdf, check_fdf, dev_fdf], ignore_index=True)
    pool_raw_prob["COMBINED_5000"] = (
        np.concatenate([pool_raw_prob["FINAL_HOLDOUT_1000"][0], pool_raw_prob["INTERNAL_CHECK_1000"][0], pool_raw_prob["DEV_3000"][0]]),
        np.concatenate([pool_raw_prob["FINAL_HOLDOUT_1000"][1], pool_raw_prob["INTERNAL_CHECK_1000"][1], pool_raw_prob["DEV_3000"][1]]),
    )

    # Reconstruct V4 predictions on all 4 pools & collect detailed TP/FP/FN records
    print("[Step 2] Reconstructing exact V4 predictions & analyzing every TP, FP, and FN...", flush=True)

    pools_reconstructed = {}
    pool_error_summaries = {}

    owner_rec_cache: Dict[str, dict] = {}

    def _get_owner_rec(eid: str, is_s1: bool) -> dict:
        rec = owner_rec_cache.get(eid)
        if rec is None:
            raw = (s1_raw_lookup if is_s1 else target_raw_lookup).get(eid) or {}
            rec = extract_v4_owner_record(
                raw.get("business_name", ""),
                raw.get("business_address", ""),
                raw.get("country", ""),
            )
            owner_rec_cache[eid] = rec
        return rec

    for p_name, qdf, fdf in [
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
        r_all, p_all = pool_raw_prob[p_name]

        fdf_v4 = fdf[m_v4].reset_index(drop=True)
        r_v4 = r_all[m_v4]
        p_v4 = p_all[m_v4]
        in_top500 = m500[m_v4]
        is_rescued = ~in_top500
        es_v4 = prepare_eval_structures(qdf, fdf_v4)

        # Compute V4 extra pair & competition features on active candidates (p_v4 >= 0.01) or predicted/GT rows
        active_or_gt = (p_v4 >= 0.01) | (fdf_v4["label"].to_numpy() == 1)
        sub_idx = np.flatnonzero(active_or_gt)
        sub_df = fdf_v4.iloc[sub_idx].reset_index(drop=True)
        sub_df = compute_v4_extra_pair_features(
            sub_df,
            s1_lookup=s1_raw_lookup,
            target_lookup=target_raw_lookup,
            is_rescued_mask=is_rescued[sub_idx],
        )
        sub_df = compute_v4_competition_features(sub_df, r_v4[sub_idx], p_v4[sub_idx])
        row_to_sub_idx = {int(orig_i): sub_i for sub_i, orig_i in enumerate(sub_idx)}

        # Reconstruct exact V4 predictions
        base_policy_mask = apply_frozen_policy(p_v4, fdf_v4, es_v4, policy_cfg)
        pre_excl_mask = base_policy_mask & (in_top500 | (p_v4 >= rescued_min_prob))
        final_mask, dec_stats = apply_v4_entity_policy_and_exclusivity(
            probs=p_v4,
            feat_df=fdf_v4,
            eval_struct=es_v4,
            policy_cfg=policy_cfg,
            s1_raw_lookup=s1_raw_lookup,
            target_raw_lookup=target_raw_lookup,
            in_top500_mask=in_top500,
            rescued_min_prob=rescued_min_prob,
            exclusivity_margin=exclusivity_margin,
        )
        eval_v4 = evaluate_predictions_from_mask(final_mask, es_v4, "V4_Baseline")
        print(
            f"  [{p_name:20s}] Reconstructed V4 F0.5={eval_v4['macro_f05']:.6f} | "
            f"P={eval_v4['macro_precision']*100:.2f}% | R={eval_v4['macro_recall']*100:.2f}% | "
            f"TP={eval_v4['tp']} FP={eval_v4['fp']} FN={eval_v4['fn']}",
            flush=True,
        )

        # Precompute target multiplicity before decoding (within-pool)
        cand_col = fdf_v4["candidate_entity_id"].tolist()
        s1_col = fdf_v4["source1_entity_id"].tolist()
        label_col = es_v4["label_col"]
        pre_excl_target_mult: Counter = Counter()
        for r_i in np.flatnonzero(pre_excl_mask):
            pre_excl_target_mult[cand_col[int(r_i)]] += 1

        # Precompute per-S1 strongest structural evidence among predicted / top candidates
        s1_slices = es_v4["s1_row_slices"]
        s1_max_struct_ev = np.zeros(len(es_v4["s1_order"]), dtype=np.float32)
        s1_has_exact_bldg_postal = np.zeros(len(es_v4["s1_order"]), dtype=bool)
        bldg_m_arr = fdf_v4["building_number_match"].to_numpy()
        post_m_arr = fdf_v4["postal_match"].to_numpy()
        addr_s_arr = fdf_v4["norm_addr_token_set"].to_numpy()
        name_s_arr = fdf_v4["core_name_token_set"].to_numpy()
        struct_ev_arr = 0.35 * name_s_arr + 0.35 * addr_s_arr + 0.15 * bldg_m_arr + 0.15 * post_m_arr

        for s1_i, (st, en) in enumerate(s1_slices):
            if st < en:
                top_m = p_v4[st:en] >= 0.90
                if top_m.any():
                    s1_max_struct_ev[s1_i] = float(np.max(struct_ev_arr[st:en][top_m]))
                    s1_has_exact_bldg_postal[s1_i] = bool(
                        ((bldg_m_arr[st:en][top_m] > 0.5) & (post_m_arr[st:en][top_m] > 0.5)).any()
                    )

        # Map row_idx -> s1_idx
        row_to_s1_idx = np.zeros(len(fdf_v4), dtype=np.int32)
        for s1_i, (st, en) in enumerate(s1_slices):
            row_to_s1_idx[st:en] = s1_i

        # Helper to build rich diagnostic dict for any row in fdf_v4
        def _build_row_diag(r_i: int) -> dict:
            s_i = row_to_sub_idx[r_i]
            srow = sub_df.iloc[s_i]
            sid = s1_col[r_i]
            tid = cand_col[r_i]
            s1_idx = int(row_to_s1_idx[r_i])
            s1_rec = _get_owner_rec(sid, True)
            tgt_rec = _get_owner_rec(tid, False)
            comp_info = compute_v41c_assignment_score(
                s1_rec,
                tgt_rec,
                float(p_v4[r_i]),
                int(srow["s1_rank"]) - 1,
                sid,
            )
            true_owner = target_to_true_s1.get(tid)
            has_other_true_owner = (true_owner is not None) and (true_owner != sid)
            other_s1_cand_stronger = bool(
                (s1_max_struct_ev[s1_idx] - float(struct_ev_arr[r_i]) >= 0.12)
                or (
                    s1_has_exact_bldg_postal[s1_idx]
                    and (float(srow["building_number_conflict"]) > 0.5 or float(srow["postal_conflict"]) > 0.5)
                )
            )
            return {
                "s1_id": sid,
                "target_id": tid,
                "country": str(es_v4["countries"][s1_idx]),
                "source": "S2" if tid.startswith("S2-") else "S3",
                "model_probability": float(p_v4[r_i]),
                "raw_model_score": float(r_v4[r_i]),
                "owner_score": float(comp_info["owner_score"]),
                "score_v41b": float(comp_info["score_v41b"]),
                "s1_rank": int(srow["s1_rank"]),
                "s1_margin_to_best_prob": float(srow["s1_margin_to_best_prob"]),
                "s1_margin_to_second_prob": float(srow["s1_margin_to_second_prob"]),
                "target_rank": int(srow["target_rank"]),
                "target_competitor_count": int(srow["target_competitor_count"]),
                "core_name_token_set": float(srow["core_name_token_set"]),
                "core_name_token_jaccard": float(srow["core_name_token_jaccard"]),
                "norm_name_ratio": float(srow["norm_name_ratio"]),
                "name_ratio": float(srow["name_ratio"]),
                "name_char_3gram_jaccard": float(srow["name_char_3gram_jaccard"]),
                "phonetic_name_token_jaccard": float(srow["phonetic_name_token_jaccard"]),
                "exact_core_name": float(srow["exact_core_name"]),
                "v4_exact_acronym": float(srow["v4_exact_acronym"]),
                "v4_abbrev_name_token_jaccard": float(srow["v4_abbrev_name_token_jaccard"]),
                "core_name_min_containment": float(srow["core_name_min_containment"]),
                "norm_addr_token_set": float(srow["norm_addr_token_set"]),
                "norm_addr_ratio": float(srow["norm_addr_ratio"]),
                "addr_char_3gram_jaccard": float(srow["addr_char_3gram_jaccard"]),
                "exact_norm_addr": float(srow["exact_norm_addr"]),
                "building_number_match": float(srow["building_number_match"]),
                "building_number_conflict": float(srow["building_number_conflict"]),
                "unit_number_match": float(srow["unit_number_match"]),
                "unit_number_conflict": float(srow["unit_number_conflict"]),
                "postal_match": float(srow["postal_match"]),
                "postal_conflict": float(srow["postal_conflict"]),
                "numeric_token_jaccard": float(srow["numeric_token_jaccard"]),
                "numeric_token_overlap": float(srow["numeric_token_overlap"]),
                "v4_numeric_token_conflict": float(srow["v4_numeric_token_conflict"]),
                "high_name_low_addr": float(srow["high_name_low_addr"]),
                "channel_mask": int(srow["channel_mask"]) if "channel_mask" in srow else 0,
                "retrieval_source_count": int(srow["v4_retrieval_source_count"]),
                "is_rescued": bool(is_rescued[r_i]),
                "target_mult_before_v4": int(pre_excl_target_mult[tid]),
                "has_true_owner_in_full_catalog": bool(has_other_true_owner),
                "true_owner_s1_id": true_owner or "",
                "s1_has_other_cand_with_stronger_evidence": other_s1_cand_stronger,
                "struct_ev": float(struct_ev_arr[r_i]),
                "s1_max_struct_ev": float(s1_max_struct_ev[s1_idx]),
            }

        tp_rows = np.flatnonzero(final_mask & (label_col == 1))
        fp_rows = np.flatnonzero(final_mask & (label_col == 0))
        fn_retrieved_rows = np.flatnonzero((~final_mask) & (label_col == 1))

        fp_records = []
        for r_i in fp_rows:
            d = _build_row_diag(int(r_i))
            d["error_category"] = classify_fp_category(d)
            # Build fine-grained pattern signature
            b_flag = "bldg_conflict" if d["building_number_conflict"] > 0.5 else ("bldg_match" if d["building_number_match"] > 0.5 else "bldg_none")
            p_flag = "postal_conflict" if d["postal_conflict"] > 0.5 else ("postal_match" if d["postal_match"] > 0.5 else "postal_none")
            n_flag = "num_conflict" if d["v4_numeric_token_conflict"] > 0.5 else ("num_agree" if d["numeric_token_overlap"] > 0 else "num_none")
            o_flag = "has_other_true_S1_owner" if d["has_true_owner_in_full_catalog"] else "unowned_target_in_GT"
            d["pattern_signature"] = f"{d['country'].upper()} | {d['source']} | {d['error_category']} | {b_flag} | {p_flag} | {n_flag} | {o_flag}"
            fp_records.append(d)

        fn_records = []
        # 1. Retrieved FNs (in m_v4 but not selected by final_mask)
        for r_i in fn_retrieved_rows:
            d = _build_row_diag(int(r_i))
            removed_by_excl = bool(pre_excl_mask[int(r_i)] and not final_mask[int(r_i)])
            d["fn_stage"] = (
                "removed_by_exclusivity"
                if removed_by_excl
                else ("capped_by_max_k_6" if (d["model_probability"] >= 0.995 and d["s1_rank"] > 6) else "below_policy_threshold")
            )
            d["error_category"] = classify_fn_category(
                d,
                _get_owner_rec(d["s1_id"], True),
                _get_owner_rec(d["target_id"], False),
                removed_by_excl,
            )
            p_bucket = (
                "p>=0.995(max_k/pen)"
                if d["model_probability"] >= 0.995
                else (
                    "p in [0.990,0.995)"
                    if d["model_probability"] >= 0.990
                    else (
                        "p in [0.980,0.990)"
                        if d["model_probability"] >= 0.980
                        else ("p in [0.900,0.980)" if d["model_probability"] >= 0.900 else "p < 0.900")
                    )
                )
            )
            d["pattern_signature"] = f"{d['country'].upper()} | {d['source']} | {d['error_category']} | {d['fn_stage']} | {p_bucket}"
            fn_records.append(d)

        # 2. Unretrieved FNs (not in m_v4)
        for s1_idx, (st, en) in enumerate(s1_slices):
            sid = es_v4["s1_order"][s1_idx]
            c_country = str(es_v4["countries"][s1_idx])
            ret_tids = set(cand_col[st:en])
            for gt_tid in es_v4["gt_sets"][s1_idx]:
                if gt_tid not in ret_tids:
                    src_str = "S2" if gt_tid.startswith("S2-") else "S3"
                    fn_records.append(
                        {
                            "s1_id": sid,
                            "target_id": gt_tid,
                            "country": c_country,
                            "source": src_str,
                            "model_probability": 0.0,
                            "raw_model_score": -99.0,
                            "owner_score": 0.0,
                            "s1_rank": 999,
                            "target_rank": 999,
                            "target_competitor_count": 0,
                            "fn_stage": "unretrieved_by_v4_blocker",
                            "error_category": "retrieval failure",
                            "pattern_signature": f"{c_country.upper()} | {src_str} | retrieval failure | unretrieved_by_v4_blocker | unretrieved",
                        }
                    )

        # Summarize FP/FN categories, top 20 patterns, singleton S1 analysis, near-threshold analysis
        fp_cat_counts = Counter(x["error_category"] for x in fp_records)
        fn_cat_counts = Counter(x["error_category"] for x in fn_records)
        fp_top20_patterns = Counter(x["pattern_signature"] for x in fp_records).most_common(20)
        fn_top20_patterns = Counter(x["pattern_signature"] for x in fn_records).most_common(20)

        # Singleton / Empty S1 analysis
        true_empty_s1 = int(es_v4["is_empty_s1"].sum())
        pred_empty_s1 = 0
        true_empty_correct = 0
        singleton_fp_s1 = 0
        false_empty_s1 = 0
        for s1_idx, (st, en) in enumerate(s1_slices):
            n_p = int(final_mask[st:en].sum()) if st < en else 0
            g_empty = bool(es_v4["is_empty_s1"][s1_idx])
            if n_p == 0:
                pred_empty_s1 += 1
                if g_empty:
                    true_empty_correct += 1
                else:
                    false_empty_s1 += 1
            else:
                if g_empty:
                    singleton_fp_s1 += 1

        # Near-threshold bands analysis
        thr_bands = {}
        for lo, hi, b_label in [
            (0.900, 0.950, "[0.900, 0.950)"),
            (0.950, 0.980, "[0.950, 0.980)"),
            (0.980, 0.990, "[0.980, 0.990)"),
            (0.990, 0.993, "[0.990, 0.993)"),
            (0.993, 0.995, "[0.993, 0.995)"),
            (0.995, 0.997, "[0.995, 0.997)"),
            (0.997, 0.999, "[0.997, 0.999)"),
            (0.999, 1.001, "[0.999, 1.000]"),
        ]:
            bm = (p_v4 >= lo) & (p_v4 < hi)
            pos_cnt = int((bm & (label_col == 1)).sum())
            neg_cnt = int((bm & (label_col == 0)).sum())
            pred_cnt = int((bm & final_mask).sum())
            thr_bands[b_label] = {
                "total_pairs": int(bm.sum()),
                "gt_positives": pos_cnt,
                "gt_negatives": neg_cnt,
                "empirical_precision": round(pos_cnt / max(1, pos_cnt + neg_cnt), 4),
                "predicted_positive_by_v4": pred_cnt,
            }

        # Summary means over FP and retrieved FN
        def _mean_field(recs: List[dict], key: str) -> float:
            vals = [float(r[key]) for r in recs if key in r and r[key] != -99.0]
            return round(float(np.mean(vals)), 4) if vals else 0.0

        pool_error_summaries[p_name] = {
            "v4_metrics": eval_v4,
            "decoder_stats": dec_stats,
            "fp_total": len(fp_records),
            "fn_total": len(fn_records),
            "fn_retrieved": len(fn_retrieved_rows),
            "fn_unretrieved": len(fn_records) - len(fn_retrieved_rows),
            "fp_by_country": dict(Counter(r["country"] for r in fp_records)),
            "fp_by_source": dict(Counter(r["source"] for r in fp_records)),
            "fn_by_country": dict(Counter(r["country"] for r in fn_records)),
            "fn_by_source": dict(Counter(r["source"] for r in fn_records)),
            "fp_categories": dict(fp_cat_counts.most_common()),
            "fn_categories": dict(fn_cat_counts.most_common()),
            "top_20_fp_patterns": [{"pattern": pat, "count": cnt} for pat, cnt in fp_top20_patterns],
            "top_20_fn_patterns": [{"pattern": pat, "count": cnt} for pat, cnt in fn_top20_patterns],
            "fp_feature_means": {
                "model_probability": _mean_field(fp_records, "model_probability"),
                "raw_model_score": _mean_field(fp_records, "raw_model_score"),
                "owner_score": _mean_field(fp_records, "owner_score"),
                "s1_rank": _mean_field(fp_records, "s1_rank"),
                "target_rank": _mean_field(fp_records, "target_rank"),
                "target_competitor_count": _mean_field(fp_records, "target_competitor_count"),
                "core_name_token_set": _mean_field(fp_records, "core_name_token_set"),
                "norm_addr_token_set": _mean_field(fp_records, "norm_addr_token_set"),
                "building_number_match_rate": _mean_field(fp_records, "building_number_match"),
                "building_number_conflict_rate": _mean_field(fp_records, "building_number_conflict"),
                "unit_number_conflict_rate": _mean_field(fp_records, "unit_number_conflict"),
                "postal_match_rate": _mean_field(fp_records, "postal_match"),
                "postal_conflict_rate": _mean_field(fp_records, "postal_conflict"),
                "numeric_token_jaccard": _mean_field(fp_records, "numeric_token_jaccard"),
                "v4_numeric_token_conflict_rate": _mean_field(fp_records, "v4_numeric_token_conflict"),
                "retrieval_source_count": _mean_field(fp_records, "retrieval_source_count"),
                "rescued_rate": _mean_field(fp_records, "is_rescued"),
                "has_true_owner_in_full_catalog_rate": _mean_field(fp_records, "has_true_owner_in_full_catalog"),
                "s1_has_other_cand_with_stronger_evidence_rate": _mean_field(
                    fp_records, "s1_has_other_cand_with_stronger_evidence"
                ),
            },
            "fn_retrieved_feature_means": {
                "model_probability": _mean_field(fn_records[: len(fn_retrieved_rows)], "model_probability"),
                "raw_model_score": _mean_field(fn_records[: len(fn_retrieved_rows)], "raw_model_score"),
                "owner_score": _mean_field(fn_records[: len(fn_retrieved_rows)], "owner_score"),
                "s1_rank": _mean_field(fn_records[: len(fn_retrieved_rows)], "s1_rank"),
                "core_name_token_set": _mean_field(fn_records[: len(fn_retrieved_rows)], "core_name_token_set"),
                "norm_addr_token_set": _mean_field(fn_records[: len(fn_retrieved_rows)], "norm_addr_token_set"),
                "building_number_match_rate": _mean_field(fn_records[: len(fn_retrieved_rows)], "building_number_match"),
                "building_number_conflict_rate": _mean_field(fn_records[: len(fn_retrieved_rows)], "building_number_conflict"),
                "postal_match_rate": _mean_field(fn_records[: len(fn_retrieved_rows)], "postal_match"),
                "postal_conflict_rate": _mean_field(fn_records[: len(fn_retrieved_rows)], "postal_conflict"),
                "numeric_token_jaccard": _mean_field(fn_records[: len(fn_retrieved_rows)], "numeric_token_jaccard"),
                "retrieval_source_count": _mean_field(fn_records[: len(fn_retrieved_rows)], "retrieval_source_count"),
            },
            "singleton_empty_s1_analysis": {
                "total_s1": len(es_v4["s1_order"]),
                "gt_empty_s1": true_empty_s1,
                "pred_empty_s1": pred_empty_s1,
                "true_empty_correctly_predicted_empty": true_empty_correct,
                "singleton_fp_s1_count": singleton_fp_s1,
                "false_empty_s1_count": false_empty_s1,
            },
            "near_threshold_bands": thr_bands,
            "fps_with_stronger_s1_sibling": sum(
                1 for r in fp_records if r["s1_has_other_cand_with_stronger_evidence"]
            ),
        }

        pools_reconstructed[p_name] = {
            "qdf": qdf,
            "fdf_v4": fdf_v4,
            "r_v4": r_v4,
            "p_v4": p_v4,
            "in_top500": in_top500,
            "es_v4": es_v4,
            "final_mask": final_mask,
            "fp_records": fp_records,
            "fn_records": fn_records,
            "struct_ev_arr": struct_ev_arr,
        }

    # -------------------------------------------------------------------------
    # STEP 3: Full-Catalog Conflict Analysis & Why V4.1-C(0.05) Picked Wrong Owner on 44/435 Cases
    # -------------------------------------------------------------------------
    print("\n[Step 3] Analyzing full-catalog target conflicts & testing V4.1-C / prob / logit conflict decoders...", flush=True)
    comb_fps = pools_reconstructed["COMBINED_5000"]["fp_records"]
    owned_fps = [r for r in comb_fps if r["has_true_owner_in_full_catalog"]]
    needed_true_owners = {r["true_owner_s1_id"] for r in owned_fps}

    true_owner_raw_lookup: Dict[str, Dict[str, str]] = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8", buffering=8 << 20) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in needed_true_owners:
                true_owner_raw_lookup[parts[0]] = {
                    "business_name": parts[1] if len(parts) > 1 else "",
                    "business_address": parts[2] if len(parts) > 2 else "",
                    "country": (parts[3] if len(parts) > 3 else "").strip().lower(),
                }

    # Evaluate conflict decoder formulas across all owned_fps (where FP S1 competes against true S1* owner)
    conflict_pairs_for_diag = []
    for r in owned_fps:
        tid = r["target_id"]
        fp_sid = r["s1_id"]
        true_sid = r["true_owner_s1_id"]
        fp_rec = _get_owner_rec(fp_sid, True)
        tgt_rec = _get_owner_rec(tid, False)
        tr_raw = true_owner_raw_lookup.get(true_sid) or {}
        true_rec = extract_v4_owner_record(
            tr_raw.get("business_name", ""),
            tr_raw.get("business_address", ""),
            tr_raw.get("country", ""),
        )
        c_fp = compute_v41c_assignment_score(fp_rec, tgt_rec, r["model_probability"], r["s1_rank"] - 1, fp_sid)
        c_true = compute_v41c_assignment_score(true_rec, tgt_rec, 0.9990, 0, true_sid)
        conflict_pairs_for_diag.append((r, c_fp, c_true))

    wrong_owner_cases_v41c_005 = []
    for r, c_fp, c_true in conflict_pairs_for_diag:
        win = resolve_v41c_conflict([c_fp, c_true], margin_param=0.05)
        if win != c_true["s1_id"]:
            wrong_owner_cases_v41c_005.append(
                {
                    "fp_s1_id": r["s1_id"],
                    "true_s1_id": c_true["s1_id"],
                    "target_id": r["target_id"],
                    "country": r["country"],
                    "fp_owner_score": round(c_fp["owner_score"], 4),
                    "true_owner_score": round(c_true["owner_score"], 4),
                    "owner_score_gap_fp_minus_true": round(c_fp["owner_score"] - c_true["owner_score"], 4),
                    "fp_prob": round(r["model_probability"], 6),
                }
            )

    conflict_decoder_sweep = {}
    # 1. V4.1-C margins 0.01 .. 0.10
    for m_val in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
        wins = sum(
            1
            for _, c_fp, c_true in conflict_pairs_for_diag
            if resolve_v41c_conflict([c_fp, c_true], margin_param=m_val) == c_true["s1_id"]
        )
        conflict_decoder_sweep[f"V4.1-C(m={m_val:.2f})"] = {
            "true_owner_wins": wins,
            "total_conflicts": len(conflict_pairs_for_diag),
            "win_rate_pct": round(100.0 * wins / max(1, len(conflict_pairs_for_diag)), 2),
        }

    # 2. owner_score + alpha * probability
    for alpha_p in [0.05, 0.10, 0.20, 0.35, 0.50]:
        wins = sum(
            1
            for _, c_fp, c_true in conflict_pairs_for_diag
            if (c_true["owner_score"] + alpha_p * c_true["model_probability"])
            >= (c_fp["owner_score"] + alpha_p * c_fp["model_probability"])
        )
        conflict_decoder_sweep[f"owner_score + {alpha_p:.2f}*prob"] = {
            "true_owner_wins": wins,
            "total_conflicts": len(conflict_pairs_for_diag),
            "win_rate_pct": round(100.0 * wins / max(1, len(conflict_pairs_for_diag)), 2),
        }

    # 3. owner_score + alpha * logit(probability)
    for alpha_l in [0.05, 0.10, 0.15, 0.18, 0.25, 0.35]:
        wins = sum(
            1
            for _, c_fp, c_true in conflict_pairs_for_diag
            if (c_true["owner_score"] + alpha_l * c_true["prob_norm"])
            >= (c_fp["owner_score"] + alpha_l * c_fp["prob_norm"])
        )
        conflict_decoder_sweep[f"owner_score + {alpha_l:.2f}*logit_norm(prob)"] = {
            "true_owner_wins": wins,
            "total_conflicts": len(conflict_pairs_for_diag),
            "win_rate_pct": round(100.0 * wins / max(1, len(conflict_pairs_for_diag)), 2),
        }

    # -------------------------------------------------------------------------
    # STEP 4: Test Cheap Decoder-Only Alternatives on DEV_3000 + INTERNAL_CHECK_1000 (Gate: FINAL_HOLDOUT_1000)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Evaluating cheap decoder-only alternatives (Tuned on DEV_3000 + INTERNAL_CHECK_1000, Gate on FINAL_HOLDOUT_1000)...", flush=True)

    def evaluate_decoder_candidate_on_pool(p_name: str, cand_cfg: dict) -> dict:
        pr = pools_reconstructed[p_name]
        fdf_v4 = pr["fdf_v4"]
        p_v4 = pr["p_v4"]
        in_top500 = pr["in_top500"]
        es_v4 = pr["es_v4"]

        # Start from Policy 5 with custom policy_cfg overrides if specified
        pol = dict(policy_cfg)
        pol.update(cand_cfg.get("policy_overrides", {}))

        base_m = apply_frozen_policy(p_v4, fdf_v4, es_v4, pol)
        r_min_p = float(cand_cfg.get("rescued_min_prob", rescued_min_prob))
        pre_m = base_m & (in_top500 | (p_v4 >= r_min_p))

        # Optional conservative contradiction pruning rule:
        if cand_cfg.get("prune_contradictions"):
            p_max_contradict = float(cand_cfg.get("contradict_max_prob", 0.9975))
            bldg_c = fdf_v4["building_number_conflict"].to_numpy() > 0.5
            post_c = fdf_v4["postal_conflict"].to_numpy() > 0.5
            unit_c = fdf_v4["unit_number_conflict"].to_numpy() > 0.5
            addr_s = fdf_v4["norm_addr_token_set"].to_numpy()
            name_s = fdf_v4["core_name_token_set"].to_numpy()
            num_ov = fdf_v4["numeric_token_overlap"].to_numpy()

            rule_type = cand_cfg["prune_contradictions"]
            if rule_type == "severe_bldg_postal_conflict_near_thr":
                contradict = (p_v4 < p_max_contradict) & (
                    (bldg_c & post_c)
                    | (post_c & (addr_s < 0.65))
                    | (bldg_c & (addr_s < 0.60) & (num_ov == 0))
                )
                pre_m = pre_m & (~contradict)
            elif rule_type == "intra_s1_sibling_anchor_conflict":
                # If S1 has a strong anchor match (prob >= 0.999, building_match=1, postal_match=1),
                # prune secondary predictions (prob < p_max_contradict) that have explicit BOTH building_conflict AND low addr similarity
                bldg_m = fdf_v4["building_number_match"].to_numpy() > 0.5
                post_m = fdf_v4["postal_match"].to_numpy() > 0.5
                drop_m = np.zeros(len(fdf_v4), dtype=bool)
                for st, en in es_v4["s1_row_slices"]:
                    if en - st <= 1:
                        continue
                    sub_pre = pre_m[st:en]
                    if sub_pre.sum() <= 1:
                        continue
                    has_strong_anchor = (
                        sub_pre & (p_v4[st:en] >= 0.9990) & bldg_m[st:en] & post_m[st:en]
                    ).any()
                    if has_strong_anchor:
                        bad_sec = (
                            sub_pre
                            & (p_v4[st:en] < p_max_contradict)
                            & ((bldg_c[st:en] & (addr_s[st:en] < 0.68)) | (post_c[st:en] & (addr_s[st:en] < 0.68)))
                        )
                        drop_m[st:en] = bad_sec
                pre_m = pre_m & (~drop_m)

        # Optional conservative structural FN rescue rule:
        if cand_cfg.get("structural_fn_rescue"):
            r_floor = float(cand_cfg.get("rescue_prob_floor", 0.9920))
            bldg_m = fdf_v4["building_number_match"].to_numpy() > 0.5
            bldg_c = fdf_v4["building_number_conflict"].to_numpy() > 0.5
            post_m = fdf_v4["postal_match"].to_numpy() > 0.5
            post_c = fdf_v4["postal_conflict"].to_numpy() > 0.5
            unit_c = fdf_v4["unit_number_conflict"].to_numpy() > 0.5
            addr_s = fdf_v4["norm_addr_token_set"].to_numpy()
            name_s = fdf_v4["core_name_token_set"].to_numpy()
            num_j = fdf_v4["numeric_token_jaccard"].to_numpy()
            ex_name = fdf_v4["exact_core_name"].to_numpy() > 0.5
            ex_addr = fdf_v4["exact_norm_addr"].to_numpy() > 0.5

            resc_mode = cand_cfg["structural_fn_rescue"]
            if resc_mode == "exact_anchor_and_bldg_postal":
                add_m = (
                    in_top500
                    & (p_v4 >= r_floor)
                    & (~bldg_c)
                    & (~post_c)
                    & (~unit_c)
                    & bldg_m
                    & post_m
                    & (num_j >= 0.75)
                    & (addr_s >= 0.92)
                    & (name_s >= 0.92)
                )
            elif resc_mode == "strict_exact_core_name_and_bldg_postal":
                add_m = (
                    in_top500
                    & (p_v4 >= r_floor)
                    & (~bldg_c)
                    & (~post_c)
                    & (~unit_c)
                    & bldg_m
                    & post_m
                    & (num_j >= 0.85)
                    & ((ex_name & (addr_s >= 0.92)) | (ex_addr & (name_s >= 0.92)))
                )
            else:
                add_m = np.zeros(len(fdf_v4), dtype=bool)

            # Enforce max_k=6 per S1 when adding rescued pairs
            if add_m.any():
                for st, en in es_v4["s1_row_slices"]:
                    if st < en and add_m[st:en].any():
                        comb = pre_m[st:en] | add_m[st:en]
                        if comb.sum() > 6:
                            idx_c = np.flatnonzero(comb)
                            top6 = idx_c[np.argsort(-p_v4[st:en][idx_c])[:6]]
                            new_sub = np.zeros(en - st, dtype=bool)
                            new_sub[top6] = True
                            pre_m[st:en] = new_sub
                        else:
                            pre_m[st:en] = comb

        # Apply target exclusivity decoding on pre_m
        m_excl = float(cand_cfg.get("exclusivity_margin", exclusivity_margin))
        cand_col = fdf_v4["candidate_entity_id"].tolist()
        s1_ids = es_v4["s1_order"]
        target_claims = defaultdict(list)
        for s1_idx, (st, en) in enumerate(es_v4["s1_row_slices"]):
            if st >= en:
                continue
            sel_rel = np.flatnonzero(pre_m[st:en])
            if len(sel_rel) == 0:
                continue
            sel_rows = [st + int(r) for r in sel_rel]
            sel_rows.sort(key=lambda r: (-float(p_v4[r]), cand_col[r]))
            for rank_idx, r_idx in enumerate(sel_rows):
                target_claims[cand_col[r_idx]].append((r_idx, s1_ids[s1_idx], rank_idx, float(p_v4[r_idx])))

        final_m = pre_m.copy()
        for tid, claims in target_claims.items():
            if len(claims) > 1:
                tgt_rec = _get_owner_rec(tid, False)
                c_infos = []
                for r_idx, sid, r_k, pv in claims:
                    s1_rec = _get_owner_rec(sid, True)
                    comp = compute_v41c_assignment_score(s1_rec, tgt_rec, pv, r_k, sid)
                    comp["row_idx"] = r_idx
                    c_infos.append(comp)
                win_s1 = resolve_v41c_conflict(c_infos, margin_param=m_excl)
                for comp in c_infos:
                    if comp["s1_id"] != win_s1:
                        final_m[comp["row_idx"]] = False

        return evaluate_predictions_from_mask(final_m, es_v4, cand_cfg["name"])

    decoder_candidates = [
        {"name": "0_V4_Frozen_Baseline", "exclusivity_margin": 0.05},
        {"name": "1_V4.1-C(m=0.03)", "exclusivity_margin": 0.03},
        {"name": "2_V4.1-C(m=0.07)", "exclusivity_margin": 0.07},
        {"name": "3_V4.1-C(m=0.10)", "exclusivity_margin": 0.10},
        {"name": "4_Country_Thr(t_us=0.9945,t_in=0.9950)", "policy_overrides": {"t_us": 0.9945, "t_in": 0.9950}},
        {"name": "5_Country_Thr(t_us=0.9950,t_in=0.9945)", "policy_overrides": {"t_us": 0.9950, "t_in": 0.9945}},
        {"name": "6_Country_Thr(t_us=0.9955,t_in=0.9950)", "policy_overrides": {"t_us": 0.9955, "t_in": 0.9950}},
        {"name": "7_Country_Thr(t_us=0.9950,t_in=0.9955)", "policy_overrides": {"t_us": 0.9950, "t_in": 0.9955}},
        {
            "name": "8_Prune_Severe_BldgPostal_Conflict(p<0.9975)",
            "prune_contradictions": "severe_bldg_postal_conflict_near_thr",
            "contradict_max_prob": 0.9975,
        },
        {
            "name": "9_Prune_Sibling_Anchor_Conflict(p<0.9980)",
            "prune_contradictions": "intra_s1_sibling_anchor_conflict",
            "contradict_max_prob": 0.9980,
        },
        {
            "name": "10_Rescue_StrictExact_BldgPostal(p>=0.9930)",
            "structural_fn_rescue": "strict_exact_core_name_and_bldg_postal",
            "rescue_prob_floor": 0.9930,
        },
        {
            "name": "11_Rescue_StrictExact_BldgPostal(p>=0.9910)",
            "structural_fn_rescue": "strict_exact_core_name_and_bldg_postal",
            "rescue_prob_floor": 0.9910,
        },
        {
            "name": "12_Combined_Prune8_plus_Rescue10",
            "prune_contradictions": "severe_bldg_postal_conflict_near_thr",
            "contradict_max_prob": 0.9975,
            "structural_fn_rescue": "strict_exact_core_name_and_bldg_postal",
            "rescue_prob_floor": 0.9930,
        },
        {
            "name": "13_Combined_Prune9_plus_Rescue10",
            "prune_contradictions": "intra_s1_sibling_anchor_conflict",
            "contradict_max_prob": 0.9980,
            "structural_fn_rescue": "strict_exact_core_name_and_bldg_postal",
            "rescue_prob_floor": 0.9930,
        },
        {
            "name": "14_Combined_Prune8_plus_Prune9_plus_Rescue11",
            "prune_contradictions": "severe_bldg_postal_conflict_near_thr",
            "contradict_max_prob": 0.9975,
            "structural_fn_rescue": "strict_exact_core_name_and_bldg_postal",
            "rescue_prob_floor": 0.9910,
        },
    ]

    decoder_eval_table = []
    v4_dev = pool_error_summaries["DEV_3000"]["v4_metrics"]["macro_f05"]
    v4_int = pool_error_summaries["INTERNAL_CHECK_1000"]["v4_metrics"]["macro_f05"]
    v4_fin = pool_error_summaries["FINAL_HOLDOUT_1000"]["v4_metrics"]["macro_f05"]
    v4_cmb = pool_error_summaries["COMBINED_5000"]["v4_metrics"]["macro_f05"]

    for cfg in decoder_candidates:
        m_dev = evaluate_decoder_candidate_on_pool("DEV_3000", cfg)
        m_int = evaluate_decoder_candidate_on_pool("INTERNAL_CHECK_1000", cfg)
        m_fin = evaluate_decoder_candidate_on_pool("FINAL_HOLDOUT_1000", cfg)
        m_cmb = evaluate_decoder_candidate_on_pool("COMBINED_5000", cfg)

        passes_tuning_rule = (m_dev["macro_f05"] >= v4_dev) and (m_int["macro_f05"] >= v4_int)
        passes_final_gate = passes_tuning_rule and (m_fin["macro_f05"] >= v4_fin)

        entry = {
            "name": cfg["name"],
            "DEV_F0.5": round(m_dev["macro_f05"], 6),
            "DEV_delta": round(m_dev["macro_f05"] - v4_dev, 6),
            "DEV_TP_FP_FN": (m_dev["tp"], m_dev["fp"], m_dev["fn"]),
            "INTERNAL_F0.5": round(m_int["macro_f05"], 6),
            "INTERNAL_delta": round(m_int["macro_f05"] - v4_int, 6),
            "INTERNAL_TP_FP_FN": (m_int["tp"], m_int["fp"], m_int["fn"]),
            "FINAL_F0.5": round(m_fin["macro_f05"], 6),
            "FINAL_delta": round(m_fin["macro_f05"] - v4_fin, 6),
            "FINAL_TP_FP_FN": (m_fin["tp"], m_fin["fp"], m_fin["fn"]),
            "COMBINED_F0.5": round(m_cmb["macro_f05"], 6),
            "COMBINED_delta": round(m_cmb["macro_f05"] - v4_cmb, 6),
            "COMBINED_TP_FP_FN": (m_cmb["tp"], m_cmb["fp"], m_cmb["fn"]),
            "passes_dev_and_internal_tuning": passes_tuning_rule,
            "passes_final_holdout_gate": passes_final_gate,
        }
        decoder_eval_table.append(entry)
        print(
            f"  {cfg['name']:44s} | DEV={entry['DEV_F0.5']:.6f} ({entry['DEV_delta']:+.6f}) | "
            f"INT={entry['INTERNAL_F0.5']:.6f} ({entry['INTERNAL_delta']:+.6f}) | "
            f"FINAL={entry['FINAL_F0.5']:.6f} ({entry['FINAL_delta']:+.6f}) | "
            f"COMB={entry['COMBINED_F0.5']:.6f} ({entry['COMBINED_delta']:+.6f}) | "
            f"Gate={passes_final_gate}",
            flush=True,
        )

    full_report = {
        "v4_public_leaderboard_score": 0.917,
        "pools": pool_error_summaries,
        "full_catalog_conflict_analysis": {
            "total_v4_fps_combined_5000": len(comb_fps),
            "fps_with_true_s1_owner_in_full_catalog": len(owned_fps),
            "fps_with_true_s1_owner_pct": round(100.0 * len(owned_fps) / max(1, len(comb_fps)), 2),
            "wrong_owner_cases_count_at_m005": len(wrong_owner_cases_v41c_005),
            "wrong_owner_sample": wrong_owner_cases_v41c_005[:20],
            "conflict_decoder_sweep": conflict_decoder_sweep,
        },
        "decoder_only_candidates": decoder_eval_table,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    out_path = V5_RESULTS_DIR / "v5_error_analysis_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nSaved complete V5 error analysis to {out_path} in {time.time()-t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
