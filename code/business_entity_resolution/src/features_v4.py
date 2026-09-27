"""V4 Pair & Competition Feature Engineering Module.

Implements Components 4 & 5 of V4:
  Component 4 (Pair Features):
    - exact name/address (including V4 acronym, collapsed initials, and abbreviation-expanded exact matches)
    - char similarity (3/4/5-gram Jaccard on normalized & abbreviation-expanded text)
    - token similarity (token Jaccard, containment, IDF overlap, abbreviation-expanded token Jaccard)
    - phonetic (phonetic Soundex/skeleton ratio, token Jaccard, exact phonetic core)
    - building number (match, conflict, extraction confidence)
    - unit number (match, conflict)
    - postal code (exact match, conflict, 3-digit prefix match, extraction confidence)
    - numeric-token agreement/conflict (Jaccard, overlap, count diff, explicit disjoint conflict)
    - country (country_same)
    - retrieval source count (popcount of retrieval channels, exact channel flag, rescued candidate flag)

  Component 5 (Competition Features):
    - S1 rank (`s1_rank`, `s1_rank_inv`)
    - S1 margin (`s1_margin_to_best_prob`, `s1_margin_to_second_prob`, `s1_margin_to_best_raw`, `s1_margin_to_second_raw`)
    - target rank (`target_rank`, `target_rank_inv`)
    - target margin (`target_margin_to_best_prob`, `target_margin_to_second_prob`)
    - number of competitors (`target_competitor_count`, `target_competitor_count_995`)
    - number above probability thresholds (`s1_count_above_050`, `s1_count_above_090`, `s1_count_above_098`, `s1_count_above_995`)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from code.business_entity_resolution.src.features_v3 import (
    V3_ALL_FEATURE_COLUMNS,
    _char_ngrams_set,
    _set_jaccard,
    compute_v3_features,
)
from code.business_entity_resolution.src.normalization_v4 import normalize_name_v4, parse_address_v4

V4_EXTRA_PAIR_FEATURES: List[str] = [
    "v4_exact_abbrev_name",
    "v4_exact_acronym",
    "v4_exact_collapsed_name",
    "v4_exact_abbrev_addr",
    "v4_abbrev_name_token_jaccard",
    "v4_abbrev_addr_token_jaccard",
    "v4_abbrev_name_char_3gram_jaccard",
    "v4_postal_prefix3_match",
    "v4_numeric_token_conflict",
    "v4_retrieval_source_count",
    "v4_has_exact_channel",
    "v4_is_rescued_candidate",
]

V4_COMPETITION_FEATURES: List[str] = [
    "v3_raw_score",
    "v3_base_prob",
    "s1_rank",
    "s1_rank_inv",
    "s1_margin_to_best_prob",
    "s1_margin_to_second_prob",
    "s1_margin_to_best_raw",
    "s1_margin_to_second_raw",
    "target_rank",
    "target_rank_inv",
    "target_margin_to_best_prob",
    "target_margin_to_second_prob",
    "target_competitor_count",
    "target_competitor_count_995",
    "s1_count_above_050",
    "s1_count_above_090",
    "s1_count_above_098",
    "s1_count_above_995",
]

V4_ALL_FEATURE_COLUMNS: List[str] = (
    V3_ALL_FEATURE_COLUMNS + V4_EXTRA_PAIR_FEATURES + V4_COMPETITION_FEATURES
)

# Precompute popcount table for channel_mask (0..255)
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.float32)


def _prepare_v4_norm_cache(
    entity_ids: List[str],
    records_lookup: Dict[str, Dict[str, str]],
) -> Dict[str, dict]:
    """Pre-parse unique entity IDs with V4 normalization once."""
    cache: Dict[str, dict] = {}
    for eid in set(entity_ids):
        rec = records_lookup.get(eid)
        if not rec:
            n_raw, a_raw, c_raw = "", "", ""
        else:
            n_raw = str(rec.get("business_name", "") or "")
            a_raw = str(rec.get("business_address", "") or "")
            c_raw = str(rec.get("country", "") or "").strip().lower()
        n4 = normalize_name_v4(n_raw)
        a4 = parse_address_v4(a_raw, c_raw)
        cache[eid] = {
            "n4": n4,
            "a4": a4,
            "abbrev_name_toks": frozenset(n4["abbrev_tokens"]),
            "abbrev_addr_toks": frozenset(a4["abbrev_tokens"]),
            "abbrev_name_3g": _char_ngrams_set(n4["abbrev_compact"], 3),
            "has_num_tokens": len(a4["numeric_tokens"]) > 0,
        }
    return cache


def compute_v4_extra_pair_features(
    df: pd.DataFrame,
    s1_lookup: Optional[Dict[str, Dict[str, str]]] = None,
    target_lookup: Optional[Dict[str, Dict[str, str]]] = None,
    is_rescued_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute the 12 V4 normalization & pair features and append them to `df`."""
    n = len(df)
    out = df.copy()

    # 1. Retrieval source count & exact channel flag
    if "channel_mask" in out.columns:
        ch_arr = np.clip(out["channel_mask"].fillna(0).to_numpy(dtype=np.int32), 0, 255)
        ret_src_cnt = _POPCOUNT_TABLE[ch_arr]
        has_exact_ch = ((ch_arr & 1) != 0).astype(np.float32)
    else:
        # Derive exact channel & channel count deterministically from the 4 V3 blocker criteria
        c1 = (
            (out["exact_orig_name"].to_numpy() > 0.5)
            | (out["exact_norm_name"].to_numpy() > 0.5)
            | (out["exact_core_name"].to_numpy() > 0.5)
            | (out["exact_compact_name"].to_numpy() > 0.5)
            | (out["exact_norm_addr"].to_numpy() > 0.5)
        ).astype(np.float32)
        c2 = (
            (out["name_common_token_count"].to_numpy() >= 1.0)
            | (out["addr_common_token_count"].to_numpy() >= 1.0)
            | (out["phonetic_name_token_jaccard"].to_numpy() > 0.0)
        ).astype(np.float32)
        c3 = (
            (out["name_char_4gram_jaccard"].to_numpy() > 0.0)
            | (out["name_char_5gram_jaccard"].to_numpy() > 0.0)
        ).astype(np.float32)
        c4 = (
            (out["building_number_match"].to_numpy() > 0.5)
            | (out["postal_match"].to_numpy() > 0.5)
        ).astype(np.float32)
        ret_src_cnt = np.maximum(1.0, c1 + c2 + c3 + c4)
        has_exact_ch = c1

    out["v4_retrieval_source_count"] = ret_src_cnt
    out["v4_has_exact_channel"] = has_exact_ch
    out["v4_is_rescued_candidate"] = (
        is_rescued_mask.astype(np.float32)
        if is_rescued_mask is not None
        else np.zeros(n, dtype=np.float32)
    )

    # 2. If raw lookups are provided, compute exact V4 normalization features directly from text
    if s1_lookup is not None and target_lookup is not None:
        s1_ids = out["source1_entity_id"].tolist()
        cand_ids = out["candidate_entity_id"].tolist()
        s1_cache = _prepare_v4_norm_cache(s1_ids, s1_lookup)
        tgt_cache = _prepare_v4_norm_cache(cand_ids, target_lookup)

        exact_abbrev_name = np.zeros(n, dtype=np.float32)
        exact_acronym = np.zeros(n, dtype=np.float32)
        exact_collapsed_name = np.zeros(n, dtype=np.float32)
        exact_abbrev_addr = np.zeros(n, dtype=np.float32)
        abbrev_name_jacc = np.zeros(n, dtype=np.float32)
        abbrev_addr_jacc = np.zeros(n, dtype=np.float32)
        abbrev_name_3g_jacc = np.zeros(n, dtype=np.float32)
        postal_pfx3 = np.zeros(n, dtype=np.float32)
        num_tok_conflict = np.zeros(n, dtype=np.float32)

        num_overlap_arr = out["numeric_token_overlap"].to_numpy()

        for i, (sid, cid) in enumerate(zip(s1_ids, cand_ids)):
            e1 = s1_cache[sid]
            e2 = tgt_cache[cid]
            n1, a1 = e1["n4"], e1["a4"]
            n2, a2 = e2["n4"], e2["a4"]

            if n1["abbrev_norm"] and n1["abbrev_norm"] == n2["abbrev_norm"]:
                exact_abbrev_name[i] = 1.0
            if n1["collapsed_norm"] and n1["collapsed_norm"] == n2["collapsed_norm"]:
                exact_collapsed_name[i] = 1.0
            if a1["abbrev_norm"] and a1["abbrev_norm"] == a2["abbrev_norm"]:
                exact_abbrev_addr[i] = 1.0

            ac1, ac2 = n1["acronym"], n2["acronym"]
            if ac1 and ac2 and len(ac1) >= 2 and ac1 == ac2:
                exact_acronym[i] = 1.0

            abbrev_name_jacc[i] = _set_jaccard(e1["abbrev_name_toks"], e2["abbrev_name_toks"])
            abbrev_addr_jacc[i] = _set_jaccard(e1["abbrev_addr_toks"], e2["abbrev_addr_toks"])
            abbrev_name_3g_jacc[i] = _set_jaccard(e1["abbrev_name_3g"], e2["abbrev_name_3g"])

            p1, p2 = a1["postal_prefix3"], a2["postal_prefix3"]
            if p1 and p2 and p1 == p2:
                postal_pfx3[i] = 1.0

            if e1["has_num_tokens"] and e2["has_num_tokens"] and num_overlap_arr[i] == 0:
                num_tok_conflict[i] = 1.0

        out["v4_exact_abbrev_name"] = exact_abbrev_name
        out["v4_exact_acronym"] = exact_acronym
        out["v4_exact_collapsed_name"] = exact_collapsed_name
        out["v4_exact_abbrev_addr"] = exact_abbrev_addr
        out["v4_abbrev_name_token_jaccard"] = abbrev_name_jacc
        out["v4_abbrev_addr_token_jaccard"] = abbrev_addr_jacc
        out["v4_abbrev_name_char_3gram_jaccard"] = abbrev_name_3g_jacc
        out["v4_postal_prefix3_match"] = postal_pfx3
        out["v4_numeric_token_conflict"] = num_tok_conflict
    else:
        # Fast fallback from existing V3 normalized columns if raw lookup is omitted
        out["v4_exact_abbrev_name"] = out["exact_core_name"].astype(np.float32)
        out["v4_exact_acronym"] = (
            (out["first_core_token_exact"].to_numpy() > 0.5)
            & (out["core_name_min_containment"].to_numpy() >= 0.5)
        ).astype(np.float32)
        out["v4_exact_collapsed_name"] = out["exact_compact_name"].astype(np.float32)
        out["v4_exact_abbrev_addr"] = out["exact_norm_addr"].astype(np.float32)
        out["v4_abbrev_name_token_jaccard"] = out["core_name_token_jaccard"].astype(np.float32)
        out["v4_abbrev_addr_token_jaccard"] = out["addr_token_jaccard"].astype(np.float32)
        out["v4_abbrev_name_char_3gram_jaccard"] = out["name_char_3gram_jaccard"].astype(np.float32)
        out["v4_postal_prefix3_match"] = out["postal_match"].astype(np.float32)
        out["v4_numeric_token_conflict"] = (
            (out["building_number_conflict"].to_numpy() > 0.5)
            | (out["postal_conflict"].to_numpy() > 0.5)
        ).astype(np.float32)

    return out


def compute_v4_competition_features(
    df: pd.DataFrame,
    v3_raw_scores: np.ndarray,
    v3_base_probs: np.ndarray,
    s1_slices: Optional[List[Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """Compute all 18 V4 competition features from V3 baseline scores (`v3_raw_scores`, `v3_base_probs`).

    Includes:
      - S1 rank (`s1_rank`, `s1_rank_inv`)
      - S1 margin (`s1_margin_to_best_prob`, `s1_margin_to_second_prob`, `s1_margin_to_best_raw`, `s1_margin_to_second_raw`)
      - Target rank (`target_rank`, `target_rank_inv`)
      - Target margin (`target_margin_to_best_prob`, `target_margin_to_second_prob`)
      - Number of competitors (`target_competitor_count`, `target_competitor_count_995`)
      - Number above probability thresholds (`s1_count_above_050`, `s1_count_above_090`, `s1_count_above_098`, `s1_count_above_995`)
    """
    n = len(df)
    out = df.copy()
    raw_arr = np.asarray(v3_raw_scores, dtype=np.float32)
    prob_arr = np.asarray(v3_base_probs, dtype=np.float32)

    out["v3_raw_score"] = raw_arr
    out["v3_base_prob"] = prob_arr

    s1_rank = np.ones(n, dtype=np.float32)
    s1_margin_best_p = np.zeros(n, dtype=np.float32)
    s1_margin_2nd_p = np.zeros(n, dtype=np.float32)
    s1_margin_best_r = np.zeros(n, dtype=np.float32)
    s1_margin_2nd_r = np.zeros(n, dtype=np.float32)
    s1_cnt_050 = np.zeros(n, dtype=np.float32)
    s1_cnt_090 = np.zeros(n, dtype=np.float32)
    s1_cnt_098 = np.zeros(n, dtype=np.float32)
    s1_cnt_995 = np.zeros(n, dtype=np.float32)

    # Build contiguous S1 slices if not provided
    if s1_slices is None:
        s1_col = out["source1_entity_id"].tolist()
        s1_slices = []
        if n > 0:
            cur = s1_col[0]
            st = 0
            for i, sid in enumerate(s1_col):
                if sid != cur:
                    s1_slices.append((st, i))
                    cur = sid
                    st = i
            s1_slices.append((st, n))

    for st, en in s1_slices:
        if st >= en:
            continue
        k = en - st
        r_sub = raw_arr[st:en]
        p_sub = prob_arr[st:en]
        order = np.argsort(-r_sub)
        ranks = np.empty(k, dtype=np.float32)
        ranks[order] = np.arange(1, k + 1, dtype=np.float32)
        s1_rank[st:en] = ranks

        best_p = float(p_sub[order[0]])
        sec_p = float(p_sub[order[1]]) if k >= 2 else 0.0
        best_r = float(r_sub[order[0]])
        sec_r = float(r_sub[order[1]]) if k >= 2 else (best_r - 5.0)

        # For rank 1, margin_to_second is positive (best - second); for rank >= 2, it is (score - best)
        m_best_p = p_sub - best_p
        m_2nd_p = p_sub - sec_p
        m_best_r = r_sub - best_r
        m_2nd_r = r_sub - sec_r

        s1_margin_best_p[st:en] = m_best_p
        s1_margin_2nd_p[st:en] = m_2nd_p
        s1_margin_best_r[st:en] = m_best_r
        s1_margin_2nd_r[st:en] = m_2nd_r

        s1_cnt_050[st:en] = float((p_sub >= 0.50).sum())
        s1_cnt_090[st:en] = float((p_sub >= 0.90).sum())
        s1_cnt_098[st:en] = float((p_sub >= 0.98).sum())
        s1_cnt_995[st:en] = float((p_sub >= 0.995).sum())

    # Target-side competition features across active claimants (prob >= 0.50)
    target_rank = np.ones(n, dtype=np.float32)
    target_margin_best_p = np.zeros(n, dtype=np.float32)
    target_margin_2nd_p = np.zeros(n, dtype=np.float32)
    target_comp_cnt = np.ones(n, dtype=np.float32)
    target_comp_995 = np.zeros(n, dtype=np.float32)

    cand_col = out["candidate_entity_id"].tolist()
    active_idx = np.flatnonzero(prob_arr >= 0.50)
    target_groups: Dict[str, List[int]] = defaultdict(list)
    for idx in active_idx:
        target_groups[cand_col[idx]].append(int(idx))

    for cid, idxs in target_groups.items():
        m = len(idxs)
        if m == 1:
            i0 = idxs[0]
            p0 = float(prob_arr[i0])
            target_rank[i0] = 1.0
            target_margin_best_p[i0] = 0.0
            target_margin_2nd_p[i0] = p0
            target_comp_cnt[i0] = 1.0
            target_comp_995[i0] = 1.0 if p0 >= 0.995 else 0.0
        else:
            idx_arr = np.array(idxs, dtype=np.int32)
            p_sub = prob_arr[idx_arr]
            r_sub = raw_arr[idx_arr]
            order = np.argsort(-r_sub)
            ranks = np.empty(m, dtype=np.float32)
            ranks[order] = np.arange(1, m + 1, dtype=np.float32)
            best_p = float(p_sub[order[0]])
            sec_p = float(p_sub[order[1]])
            cnt_995 = float((p_sub >= 0.995).sum())

            target_rank[idx_arr] = ranks
            target_margin_best_p[idx_arr] = p_sub - best_p
            target_margin_2nd_p[idx_arr] = p_sub - sec_p
            target_comp_cnt[idx_arr] = float(m)
            target_comp_995[idx_arr] = cnt_995

    out["s1_rank"] = s1_rank
    out["s1_rank_inv"] = (1.0 / np.maximum(1.0, s1_rank)).astype(np.float32)
    out["s1_margin_to_best_prob"] = s1_margin_best_p
    out["s1_margin_to_second_prob"] = s1_margin_2nd_p
    out["s1_margin_to_best_raw"] = s1_margin_best_r
    out["s1_margin_to_second_raw"] = s1_margin_2nd_r
    out["target_rank"] = target_rank
    out["target_rank_inv"] = (1.0 / np.maximum(1.0, target_rank)).astype(np.float32)
    out["target_margin_to_best_prob"] = target_margin_best_p
    out["target_margin_to_second_prob"] = target_margin_2nd_p
    out["target_competitor_count"] = target_comp_cnt
    out["target_competitor_count_995"] = target_comp_995
    out["s1_count_above_050"] = s1_cnt_050
    out["s1_count_above_090"] = s1_cnt_090
    out["s1_count_above_098"] = s1_cnt_098
    out["s1_count_above_995"] = s1_cnt_995

    return out
