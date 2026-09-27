"""V5 Targeted Candidate Rescue & Retrieval Module (Requirement 4).

Keeps V4 Top-500 (`m500`) unchanged as the baseline and adds 5 targeted rescue channels:
  1. `abbreviation_token_retrieval`
  2. `char_3_4_gram_retrieval`
  3. `numeric_address_retrieval`
  4. `reverse_target_to_s1_retrieval`
  5. `transliteration_retrieval`

Measures separately for S2 and S3:
  - number of previously unretrieved GT positives recovered (marginal & incremental)
  - candidate count added (S2, S3, total)
  - candidate recall (micro, macro, S2, S3)
  - runtime
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from experiments.v3.src.benchmark_test_prefilter import select_prefilter_mask_for_pool
from experiments.v4.src.candidate_rescue_v4 import (
    V4EmbeddingRetriever,
    compute_candidate_recall_from_mask,
    select_v4_rescued_mask_for_pool,
)
from experiments.v5.src.features_v5 import (
    compute_v5_features_for_df,
    prepare_v5_entity_cache,
)

V5_RESCUE_CHANNEL_NAMES = [
    "abbreviation_token_retrieval",
    "char_3_4_gram_retrieval",
    "numeric_address_retrieval",
    "reverse_target_to_s1_retrieval",
    "transliteration_retrieval",
]


def evaluate_v5_targeted_rescue_for_pool(
    feat_df: pd.DataFrame,
    eval_struct: dict,
    s1_lookup: Dict[str, Dict[str, str]],
    target_lookup: Dict[str, Dict[str, str]],
    embed_retriever: Optional[V4EmbeddingRetriever] = None,
    base_top_k: int = 500,
    s1_cache: Optional[Dict[str, dict]] = None,
    target_cache: Optional[Dict[str, dict]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run V5 targeted candidate rescue on top of the unchanged Top-500 baseline (`m500`),
    measuring each of the 5 rescue channels separately for S2 and S3.

    Returns:
      - `m_v5`: boolean mask of candidates retained in V5 (`m_v4 | v5_rescued_tail`)
      - `m_v4`: boolean mask of frozen V4 candidates (`m500 | v4_rescued_tail`)
      - `m500`: boolean mask of baseline Top-500 candidates
      - `report`: detailed S2/S3 breakdown of recovered GT positives, candidate counts added,
                  candidate recall, and runtime per channel.
    """
    t0 = time.time()

    # 1. Baseline Top-500 & Frozen V4 Rescued masks
    m_v4, m500, v4_stats = select_v4_rescued_mask_for_pool(
        feat_df,
        eval_struct,
        base_top_k=base_top_k,
        embed_retriever=embed_retriever,
        s1_lookup=s1_lookup,
        target_lookup=target_lookup,
        mode="precision_confirmed",
    )
    tail = ~m500

    # Fast coarse filter on tail (~m500) to avoid parsing all 1.15M unpromising tail pairs
    cand_max_ret = (
        feat_df.groupby("candidate_entity_id", sort=False)["retrieval_score"]
        .transform("max")
        .to_numpy()
    )
    coarse_tail = tail & (
        (feat_df["building_number_match"].to_numpy() > 0.5)
        | (feat_df["numeric_token_overlap"].to_numpy() >= 1.0)
        | (feat_df["norm_addr_token_set"].to_numpy() >= 0.58)
        | (feat_df["core_name_token_set"].to_numpy() >= 0.68)
        | (feat_df["name_char_3gram_jaccard"].to_numpy() >= 0.22)
        | (feat_df["retrieval_score"].to_numpy() >= 0.78 * np.maximum(cand_max_ret, 1e-6))
    )

    coarse_idx = np.flatnonzero(coarse_tail)
    sub_df = feat_df.iloc[coarse_idx].reset_index(drop=True)
    sub_v5 = compute_v5_features_for_df(
        sub_df,
        s1_lookup=s1_lookup,
        target_lookup=target_lookup,
        s1_cache=s1_cache,
        target_cache=target_cache,
    )

    n_total = len(feat_df)
    ch_masks: Dict[str, np.ndarray] = {
        ch: np.zeros(n_total, dtype=bool) for ch in V5_RESCUE_CHANNEL_NAMES
    }
    ch_runtimes: Dict[str, float] = {}

    # Extract arrays on coarse_idx
    bldg_m = sub_v5["building_number_match"].to_numpy() > 0.5
    bldg_c = sub_v5["building_number_conflict"].to_numpy() > 0.5
    post_c = sub_v5["postal_conflict"].to_numpy() > 0.5
    num_jacc = sub_v5["numeric_token_jaccard"].to_numpy()
    addr_tset = sub_v5["norm_addr_token_set"].to_numpy()
    name_3g = sub_v5["name_char_3gram_jaccard"].to_numpy()
    name_4g = sub_v5["name_char_4gram_jaccard"].to_numpy()
    addr_3g = sub_v5["addr_char_3gram_jaccard"].to_numpy()
    ret_sc = sub_v5["retrieval_score"].to_numpy()
    sub_cand_max = cand_max_ret[coarse_idx]

    v5_bldg_norm_ag = sub_v5["v5_bldg_norm_num_agreement"].to_numpy()
    v5_bldg_min_cont = sub_v5["v5_bldg_num_min_containment"].to_numpy()
    v5_bldg_loc_sim = sub_v5["v5_bldg_street_locality_sim"].to_numpy()
    v5_bldg_resolved = sub_v5["v5_bldg_conflict_resolved"].to_numpy() > 0.5
    v5_bldg_sev_flag = sub_v5["v5_bldg_severe_conflict_flag"].to_numpy() > 0.5

    v5_abbr_name_jacc = sub_v5["v5_abbr_name_token_jaccard"].to_numpy()
    v5_abbr_name_tset = sub_v5["v5_abbr_name_token_set"].to_numpy()
    v5_abbr_name_3g = sub_v5["v5_abbr_name_char_3gram_jaccard"].to_numpy()
    v5_abbr_addr_tset = sub_v5["v5_abbr_addr_token_set"].to_numpy()
    v5_abbr_acro = sub_v5["v5_abbr_acronym_compat"].to_numpy()
    v5_abbr_init = sub_v5["v5_abbr_initial_token_compat"].to_numpy()
    v5_abbr_brand_conf = sub_v5["v5_abbr_distinctive_brand_conflict"].to_numpy() > 0.5

    v5_tr_non_ascii = sub_v5["v5_translit_has_non_ascii"].to_numpy() > 0.5
    v5_tr_name_tset = sub_v5["v5_translit_name_token_set"].to_numpy()
    v5_tr_phon_edit = sub_v5["v5_translit_phonetic_edit_ratio"].to_numpy()
    v5_tr_addr_tset = sub_v5["v5_translit_addr_token_set"].to_numpy()
    v5_tr_conf = sub_v5["v5_translit_brand_or_state_conflict"].to_numpy() > 0.5

    # Structural address gate for high-precision candidate rescue
    no_hard_conflict = (~post_c) & (~v5_bldg_sev_flag) & (~v5_abbr_brand_conf) & (~v5_tr_conf)
    strong_addr_anchor = (
        ((bldg_m | (v5_bldg_norm_ag >= 0.75) | (v5_bldg_min_cont >= 0.99)) & ((~bldg_c) | v5_bldg_resolved))
        & ((addr_tset >= 0.72) | (v5_bldg_loc_sim >= 0.70) | (v5_abbr_addr_tset >= 0.75))
    )

    # Channel 1: Abbreviation token retrieval
    tc0 = time.time()
    c1_sub = (
        no_hard_conflict
        & strong_addr_anchor
        & (
            (v5_abbr_name_jacc >= 0.35)
            | (v5_abbr_name_tset >= 0.75)
            | (v5_abbr_acro >= 0.75)
            | (v5_abbr_init >= 0.70)
        )
    )
    ch_masks["abbreviation_token_retrieval"][coarse_idx] = c1_sub
    ch_runtimes["abbreviation_token_retrieval"] = round(time.time() - tc0, 4)

    # Channel 2: Char 3/4-gram retrieval
    tc0 = time.time()
    c2_sub = (
        no_hard_conflict
        & strong_addr_anchor
        & (
            (name_3g >= 0.24)
            | (name_4g >= 0.18)
            | (addr_3g >= 0.28)
            | (v5_abbr_name_3g >= 0.24)
        )
    )
    ch_masks["char_3_4_gram_retrieval"][coarse_idx] = c2_sub
    ch_runtimes["char_3_4_gram_retrieval"] = round(time.time() - tc0, 4)

    # Channel 3: Numeric / address retrieval (including V5 building-set containment & resolved noisy numbers)
    tc0 = time.time()
    c3_sub = (
        no_hard_conflict
        & (
            (bldg_m & (~bldg_c) & (num_jacc >= 0.50) & (addr_tset >= 0.80))
            | (
                (v5_bldg_min_cont >= 0.99)
                & (v5_bldg_norm_ag >= 0.80)
                & (v5_bldg_loc_sim >= 0.72)
                & ((v5_abbr_name_tset >= 0.72) | (v5_tr_name_tset >= 0.72))
            )
        )
    )
    ch_masks["numeric_address_retrieval"][coarse_idx] = c3_sub
    ch_runtimes["numeric_address_retrieval"] = round(time.time() - tc0, 4)

    # Channel 4: Reverse target->S1 retrieval
    tc0 = time.time()
    c4_sub = (
        no_hard_conflict
        & strong_addr_anchor
        & (ret_sc >= 0.80 * np.maximum(sub_cand_max, 1e-6))
    )
    ch_masks["reverse_target_to_s1_retrieval"][coarse_idx] = c4_sub
    ch_runtimes["reverse_target_to_s1_retrieval"] = round(time.time() - tc0, 4)

    # Channel 5: Transliteration retrieval (Indic script & phonetic skeleton rescue)
    tc0 = time.time()
    c5_sub = (
        no_hard_conflict
        & strong_addr_anchor
        & (
            (v5_tr_non_ascii & ((v5_tr_name_tset >= 0.65) | (v5_tr_phon_edit >= 0.68)))
            | ((v5_tr_phon_edit >= 0.78) & (v5_tr_addr_tset >= 0.78))
        )
    )
    ch_masks["transliteration_retrieval"][coarse_idx] = c5_sub
    ch_runtimes["transliteration_retrieval"] = round(time.time() - tc0, 4)

    # Combine V4 rescue with V5 5-channel rescue so candidate_recall(V5) >= candidate_recall(V4) >= candidate_recall(V3)
    v5_rescued_tail = tail & (
        (m_v4 & tail)
        | ch_masks["abbreviation_token_retrieval"]
        | ch_masks["char_3_4_gram_retrieval"]
        | ch_masks["numeric_address_retrieval"]
        | ch_masks["reverse_target_to_s1_retrieval"]
        | ch_masks["transliteration_retrieval"]
    )
    m_v5 = m500 | v5_rescued_tail
    total_elapsed = round(time.time() - t0, 3)

    # Measure S2 and S3 candidate counts, recovered GT positives, and recall per channel
    label_col = eval_struct["label_col"]
    is_s2 = eval_struct["is_s2_cand"]
    is_s3 = ~is_s2

    marginal_channel_report = {}
    incremental_channel_report = {}
    running_mask = m500.copy()

    for ch_name in V5_RESCUE_CHANNEL_NAMES:
        cm = tail & ch_masks[ch_name]
        # Marginal over Top-500 baseline
        marg_cands_s2 = int((cm & is_s2).sum())
        marg_cands_s3 = int((cm & is_s3).sum())
        marg_gt_s2 = int(label_col[cm & is_s2].sum())
        marg_gt_s3 = int(label_col[cm & is_s3].sum())
        rec_marg = compute_candidate_recall_from_mask(m500 | cm, eval_struct)

        marginal_channel_report[ch_name] = {
            "candidates_added_total": marg_cands_s2 + marg_cands_s3,
            "candidates_added_s2": marg_cands_s2,
            "candidates_added_s3": marg_cands_s3,
            "gt_positives_recovered_total": marg_gt_s2 + marg_gt_s3,
            "gt_positives_recovered_s2": marg_gt_s2,
            "gt_positives_recovered_s3": marg_gt_s3,
            "micro_recall_with_channel": rec_marg["micro_recall"],
            "s2_recall_with_channel": rec_marg["s2_recall"],
            "s3_recall_with_channel": rec_marg["s3_recall"],
            "runtime_sec": ch_runtimes[ch_name],
        }

        # Incremental over previous channels
        new_cm = cm & (~running_mask)
        inc_cands_s2 = int((new_cm & is_s2).sum())
        inc_cands_s3 = int((new_cm & is_s3).sum())
        inc_gt_s2 = int(label_col[new_cm & is_s2].sum())
        inc_gt_s3 = int(label_col[new_cm & is_s3].sum())
        running_mask = running_mask | new_cm
        rec_inc = compute_candidate_recall_from_mask(running_mask, eval_struct)

        incremental_channel_report[ch_name] = {
            "incremental_candidates_added_total": inc_cands_s2 + inc_cands_s3,
            "incremental_candidates_added_s2": inc_cands_s2,
            "incremental_candidates_added_s3": inc_cands_s3,
            "incremental_gt_positives_recovered_total": inc_gt_s2 + inc_gt_s3,
            "incremental_gt_positives_recovered_s2": inc_gt_s2,
            "incremental_gt_positives_recovered_s3": inc_gt_s3,
            "cumulative_micro_recall": rec_inc["micro_recall"],
            "cumulative_s2_recall": rec_inc["s2_recall"],
            "cumulative_s3_recall": rec_inc["s3_recall"],
        }

    v3_rec = v4_stats["v3_top500"]
    v4_rec = v4_stats["v4_rescued"]
    v5_rec = compute_candidate_recall_from_mask(m_v5, eval_struct)

    v5_only_new = m_v5 & (~m_v4)
    report = {
        "v3_top500_baseline": v3_rec,
        "v4_rescued_baseline": v4_rec,
        "v5_rescued": v5_rec,
        "v5_vs_top500": {
            "candidates_added_total": int(v5_rescued_tail.sum()),
            "candidates_added_s2": int((v5_rescued_tail & is_s2).sum()),
            "candidates_added_s3": int((v5_rescued_tail & is_s3).sum()),
            "gt_positives_recovered_total": int(v5_rec["retrieved_gt_links"] - v3_rec["retrieved_gt_links"]),
            "gt_positives_recovered_s2": int(v5_rec["s2_retrieved_gt"] - v3_rec["s2_retrieved_gt"]),
            "gt_positives_recovered_s3": int(v5_rec["s3_retrieved_gt"] - v3_rec["s3_retrieved_gt"]),
        },
        "v5_vs_v4_incremental": {
            "candidates_added_total": int(v5_only_new.sum()),
            "candidates_added_s2": int((v5_only_new & is_s2).sum()),
            "candidates_added_s3": int((v5_only_new & is_s3).sum()),
            "gt_positives_recovered_total": int(v5_rec["retrieved_gt_links"] - v4_rec["retrieved_gt_links"]),
            "gt_positives_recovered_s2": int(v5_rec["s2_retrieved_gt"] - v4_rec["s2_retrieved_gt"]),
            "gt_positives_recovered_s3": int(v5_rec["s3_retrieved_gt"] - v4_rec["s3_retrieved_gt"]),
        },
        "marginal_by_channel_vs_top500": marginal_channel_report,
        "incremental_by_channel_sequence": incremental_channel_report,
        "total_rescue_runtime_sec": total_elapsed,
    }
    return m_v5, m_v4, m500, report
