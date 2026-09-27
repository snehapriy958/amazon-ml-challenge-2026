"""Phase V3-B: Normalized, Country-Aware, Structural, Numeric & Phonetic Feature Module.

All fuzzy string calculations operate on normalized representations while preserving
the 11 baseline features for direct comparison.

Total Features: 73 pairwise features (70 core + 3 phonetic features for DEV ablation).
Strictly pairwise and deterministic (zero train-vs-inference group-size skew).
"""

from collections import Counter
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path.cwd()))
from experiments.v3.src.normalization_v3 import normalize_name_v3, parse_address_v3


BASE_11_FEATURES: List[str] = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "country_same",
    "name_len_1",
    "name_len_2",
    "address_len_1",
    "address_len_2",
]

V3_NAME_FEATURES: List[str] = [
    "norm_name_ratio",
    "norm_name_token_sort",
    "norm_name_token_set",
    "norm_name_partial_ratio",
    "core_name_ratio",
    "core_name_token_sort",
    "core_name_token_set",
    "core_name_partial_ratio",
    "compact_name_ratio",
    "name_char_3gram_jaccard",
    "name_char_4gram_jaccard",
    "name_char_5gram_jaccard",
    "name_token_jaccard",
    "core_name_token_jaccard",
    "core_name_min_containment",
    "core_name_max_containment",
    "name_common_token_count",
    "name_rare_token_idf_overlap",
    "name_rare_token_idf_ratio",
    "name_len_ratio",
    "core_name_len_ratio",
    "name_token_count_min",
    "name_token_count_diff",
    "exact_orig_name",
    "exact_norm_name",
    "exact_core_name",
    "exact_compact_name",
    "business_suffix_agreement",
    "first_core_token_exact",
    "first_core_token_ratio",
]

V3_ADDR_FEATURES: List[str] = [
    "norm_addr_ratio",
    "norm_addr_token_sort",
    "norm_addr_token_set",
    "norm_addr_partial_ratio",
    "addr_char_3gram_jaccard",
    "addr_char_4gram_jaccard",
    "addr_char_5gram_jaccard",
    "addr_token_jaccard",
    "addr_min_containment",
    "addr_max_containment",
    "addr_common_token_count",
    "addr_rare_token_idf_overlap",
    "addr_rare_token_idf_ratio",
    "exact_orig_addr",
    "exact_norm_addr",
    "addr_len_ratio",
    "addr_both_present",
    "addr_either_missing",
    "state_region_match",
    "state_region_conflict",
]

V3_NUMERIC_FEATURES: List[str] = [
    "building_number_match",
    "building_number_conflict",
    "building_extraction_conf_min",
    "postal_match",
    "postal_conflict",
    "postal_extraction_conf_min",
    "numeric_token_jaccard",
    "numeric_token_overlap",
    "numeric_count_difference",
    "unit_number_match",
    "unit_number_conflict",
]

V3_INTERACTION_FEATURES: List[str] = [
    "name_x_addr_sim",
    "core_name_x_norm_addr_set",
    "name_addr_agreement",
    "name_addr_conflict",
    "high_name_low_addr",
    "high_name_high_addr",
    "exact_name_plus_addr_ev",
    "exact_addr_plus_name_ev",
    "num_agree_x_name_sim",
    "num_conflict_x_name_sim",
    "postal_agree_x_name_sim",
    "postal_conflict_x_name_sim",
    "is_s2_candidate",
]

V3_PHONETIC_FEATURES: List[str] = [
    "phonetic_name_ratio",
    "phonetic_name_token_jaccard",
    "phonetic_exact_core_name",
]

V3_CORE_FEATURE_COLUMNS: List[str] = (
    BASE_11_FEATURES
    + V3_NAME_FEATURES
    + V3_ADDR_FEATURES
    + V3_NUMERIC_FEATURES
    + V3_INTERACTION_FEATURES
)

V3_ALL_FEATURE_COLUMNS: List[str] = V3_CORE_FEATURE_COLUMNS + V3_PHONETIC_FEATURES


def _char_ngrams_set(s: str, n: int) -> frozenset:
    if not s:
        return frozenset()
    if len(s) < n:
        return frozenset([s])
    return frozenset(s[i : i + n] for i in range(len(s) - n + 1))


def _set_jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def _prepare_entity_cache(
    entity_ids: List[str],
    records_lookup: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, dict], Counter, Counter]:
    """Pre-parse unique entities once and compute local token DF counters for fast IDF."""
    cache: Dict[str, dict] = {}
    name_df_counter: Counter = Counter()
    addr_df_counter: Counter = Counter()

    for eid in set(entity_ids):
        rec = records_lookup.get(eid)
        if rec is None:
            n_raw, a_raw, c_raw = "", "", ""
        else:
            n_raw = str(rec.get("business_name", "") or "")
            a_raw = str(rec.get("business_address", "") or "")
            c_raw = str(rec.get("country", "") or "").strip().lower()

        n_info = normalize_name_v3(n_raw)
        a_info = parse_address_v3(a_raw, c_raw)

        n_comp = n_info["compact_norm"]
        a_norm = a_info["lower_norm"]

        n_tok_set = frozenset(n_info["tokens"])
        n_core_set = frozenset(n_info["core_tokens"])
        n_phon_set = frozenset(n_info["phonetic_tokens"])
        a_tok_set = frozenset(a_info["tokens"])

        for t in n_core_set:
            name_df_counter[t] += 1
        for t in a_tok_set:
            addr_df_counter[t] += 1

        cache[eid] = {
            "raw_name": n_raw,
            "raw_addr": a_raw,
            "country": c_raw,
            "n_info": n_info,
            "a_info": a_info,
            "n_tok_set": n_tok_set,
            "n_core_set": n_core_set,
            "n_phon_set": n_phon_set,
            "a_tok_set": a_tok_set,
            "n_c3": _char_ngrams_set(n_comp, 3),
            "n_c4": _char_ngrams_set(n_comp, 4),
            "n_c5": _char_ngrams_set(n_comp, 5),
            "a_c3": _char_ngrams_set(a_norm, 3),
            "a_c4": _char_ngrams_set(a_norm, 4),
            "a_c5": _char_ngrams_set(a_norm, 5),
            "first_core_tok": n_info["core_tokens"][0] if n_info["core_tokens"] else "",
        }

    return cache, name_df_counter, addr_df_counter


def compute_v3_features(
    pairs_df: pd.DataFrame,
    s1_lookup: Dict[str, Dict[str, str]],
    target_lookup: Dict[str, Dict[str, str]],
    batch_size: int = 150_000,
) -> pd.DataFrame:
    """Compute all 88 V3 pairwise features in memory-safe vectorized batches."""
    if pairs_df.empty:
        return pd.DataFrame(columns=V3_ALL_FEATURE_COLUMNS)

    import time
    t_f0 = time.time()
    s1_ids_all = pairs_df["source1_entity_id"].tolist()
    cand_ids_all = pairs_df["candidate_entity_id"].tolist()

    # Pre-parse unique S1 entities once (small bounded set)
    s1_cache, name_df_cnt, addr_df_cnt = _prepare_entity_cache(
        list(set(s1_ids_all)), s1_lookup
    )

    # Precompute token IDF weights from S1 corpus
    name_idf = {t: 1.0 / math.log10(10.0 + df * 10) for t, df in name_df_cnt.items()}
    addr_idf = {t: 1.0 / math.log10(10.0 + df * 10) for t, df in addr_df_cnt.items()}

    n_total = len(pairs_df)
    out_chunks: List[pd.DataFrame] = []

    for start in range(0, n_total, batch_size):
        t_b0 = time.time()
        end = min(start + batch_size, n_total)
        b_s1 = s1_ids_all[start:end]
        b_cid = cand_ids_all[start:end]
        b_len = end - start

        cand_cache, _, _ = _prepare_entity_cache(b_cid, target_lookup)

        e1_list = [s1_cache[sid] for sid in b_s1]
        e2_list = [cand_cache[cid] for cid in b_cid]

        # String arrays for batched C++ RapidFuzz cpdist
        raw_n1 = [e["raw_name"] for e in e1_list]
        raw_n2 = [e["raw_name"] for e in e2_list]
        raw_a1 = [e["raw_addr"] for e in e1_list]
        raw_a2 = [e["raw_addr"] for e in e2_list]

        norm_n1 = [e["n_info"]["lower_norm"] for e in e1_list]
        norm_n2 = [e["n_info"]["lower_norm"] for e in e2_list]
        core_n1 = [e["n_info"]["core_name_norm"] for e in e1_list]
        core_n2 = [e["n_info"]["core_name_norm"] for e in e2_list]
        comp_n1 = [e["n_info"]["compact_norm"] for e in e1_list]
        comp_n2 = [e["n_info"]["compact_norm"] for e in e2_list]
        phon_n1 = [e["n_info"]["phonetic_str"] for e in e1_list]
        phon_n2 = [e["n_info"]["phonetic_str"] for e in e2_list]
        first_t1 = [e["first_core_tok"] for e in e1_list]
        first_t2 = [e["first_core_tok"] for e in e2_list]

        norm_a1 = [e["a_info"]["lower_norm"] for e in e1_list]
        norm_a2 = [e["a_info"]["lower_norm"] for e in e2_list]

        # 1. Baseline 11 raw similarities
        name_ratio = process.cpdist(raw_n1, raw_n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100.0
        name_token_sort = process.cpdist(raw_n1, raw_n2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100.0
        name_token_set = process.cpdist(raw_n1, raw_n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0
        address_ratio = process.cpdist(raw_a1, raw_a2, scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100.0
        address_token_sort = process.cpdist(raw_a1, raw_a2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100.0
        address_token_set = process.cpdist(raw_a1, raw_a2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0

        # 2. V3 Normalized Name & Address RapidFuzz similarities
        norm_name_ratio = process.cpdist(norm_n1, norm_n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100.0
        norm_name_token_sort = process.cpdist(norm_n1, norm_n2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100.0
        norm_name_token_set = process.cpdist(norm_n1, norm_n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0
        norm_name_partial_ratio = process.cpdist(norm_n1, norm_n2, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32) / 100.0

        core_name_ratio = process.cpdist(core_n1, core_n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100.0
        core_name_token_sort = process.cpdist(core_n1, core_n2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100.0
        core_name_token_set = process.cpdist(core_n1, core_n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0
        core_name_partial_ratio = process.cpdist(core_n1, core_n2, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32) / 100.0

        compact_name_ratio = process.cpdist(comp_n1, comp_n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100.0
        phonetic_name_ratio = process.cpdist(phon_n1, phon_n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100.0
        first_core_token_ratio = process.cpdist(first_t1, first_t2, scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100.0

        norm_addr_ratio = process.cpdist(norm_a1, norm_a2, scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100.0
        norm_addr_token_sort = process.cpdist(norm_a1, norm_a2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100.0
        norm_addr_token_set = process.cpdist(norm_a1, norm_a2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0
        norm_addr_partial_ratio = process.cpdist(norm_a1, norm_a2, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32) / 100.0

        # Allocate numpy arrays for structural, token, char-gram, and numeric features
        country_same = np.zeros(b_len, dtype=np.float32)
        name_len_1 = np.zeros(b_len, dtype=np.float32)
        name_len_2 = np.zeros(b_len, dtype=np.float32)
        address_len_1 = np.zeros(b_len, dtype=np.float32)
        address_len_2 = np.zeros(b_len, dtype=np.float32)

        name_char_3gram_jaccard = np.zeros(b_len, dtype=np.float32)
        name_char_4gram_jaccard = np.zeros(b_len, dtype=np.float32)
        name_char_5gram_jaccard = np.zeros(b_len, dtype=np.float32)
        name_token_jaccard = np.zeros(b_len, dtype=np.float32)
        core_name_token_jaccard = np.zeros(b_len, dtype=np.float32)
        core_name_min_containment = np.zeros(b_len, dtype=np.float32)
        core_name_max_containment = np.zeros(b_len, dtype=np.float32)
        name_common_token_count = np.zeros(b_len, dtype=np.float32)
        name_rare_token_idf_overlap = np.zeros(b_len, dtype=np.float32)
        name_rare_token_idf_ratio = np.zeros(b_len, dtype=np.float32)
        name_len_ratio = np.zeros(b_len, dtype=np.float32)
        core_name_len_ratio = np.zeros(b_len, dtype=np.float32)
        name_token_count_min = np.zeros(b_len, dtype=np.float32)
        name_token_count_diff = np.zeros(b_len, dtype=np.float32)
        exact_orig_name = np.zeros(b_len, dtype=np.float32)
        exact_norm_name = np.zeros(b_len, dtype=np.float32)
        exact_core_name = np.zeros(b_len, dtype=np.float32)
        exact_compact_name = np.zeros(b_len, dtype=np.float32)
        business_suffix_agreement = np.zeros(b_len, dtype=np.float32)
        first_core_token_exact = np.zeros(b_len, dtype=np.float32)

        addr_char_3gram_jaccard = np.zeros(b_len, dtype=np.float32)
        addr_char_4gram_jaccard = np.zeros(b_len, dtype=np.float32)
        addr_char_5gram_jaccard = np.zeros(b_len, dtype=np.float32)
        addr_token_jaccard = np.zeros(b_len, dtype=np.float32)
        addr_min_containment = np.zeros(b_len, dtype=np.float32)
        addr_max_containment = np.zeros(b_len, dtype=np.float32)
        addr_common_token_count = np.zeros(b_len, dtype=np.float32)
        addr_rare_token_idf_overlap = np.zeros(b_len, dtype=np.float32)
        addr_rare_token_idf_ratio = np.zeros(b_len, dtype=np.float32)
        exact_orig_addr = np.zeros(b_len, dtype=np.float32)
        exact_norm_addr = np.zeros(b_len, dtype=np.float32)
        addr_len_ratio = np.zeros(b_len, dtype=np.float32)
        addr_both_present = np.zeros(b_len, dtype=np.float32)
        addr_either_missing = np.zeros(b_len, dtype=np.float32)
        state_region_match = np.zeros(b_len, dtype=np.float32)
        state_region_conflict = np.zeros(b_len, dtype=np.float32)

        building_number_match = np.zeros(b_len, dtype=np.float32)
        building_number_conflict = np.zeros(b_len, dtype=np.float32)
        building_extraction_conf_min = np.zeros(b_len, dtype=np.float32)
        postal_match = np.zeros(b_len, dtype=np.float32)
        postal_conflict = np.zeros(b_len, dtype=np.float32)
        postal_extraction_conf_min = np.zeros(b_len, dtype=np.float32)
        numeric_token_jaccard = np.zeros(b_len, dtype=np.float32)
        numeric_token_overlap = np.zeros(b_len, dtype=np.float32)
        numeric_count_difference = np.zeros(b_len, dtype=np.float32)
        unit_number_match = np.zeros(b_len, dtype=np.float32)
        unit_number_conflict = np.zeros(b_len, dtype=np.float32)

        phonetic_name_token_jaccard = np.zeros(b_len, dtype=np.float32)
        phonetic_exact_core_name = np.zeros(b_len, dtype=np.float32)
        is_s2_candidate = np.zeros(b_len, dtype=np.float32)

        for i in range(b_len):
            e1 = e1_list[i]
            e2 = e2_list[i]
            n1 = e1["n_info"]
            n2 = e2["n_info"]
            a1 = e1["a_info"]
            a2 = e2["a_info"]

            country_same[i] = 1.0 if (e1["country"] and e1["country"] == e2["country"]) else 0.0
            is_s2_candidate[i] = 1.0 if b_cid[i].startswith("S2-") else 0.0

            nl1 = len(e1["raw_name"])
            nl2 = len(e2["raw_name"])
            al1 = len(e1["raw_addr"])
            al2 = len(e2["raw_addr"])
            name_len_1[i] = nl1
            name_len_2[i] = nl2
            address_len_1[i] = al1
            address_len_2[i] = al2

            # Name length ratios & exact flags
            name_len_ratio[i] = min(nl1, nl2) / max(1, max(nl1, nl2))
            cnl1 = len(n1["core_name_norm"])
            cnl2 = len(n2["core_name_norm"])
            core_name_len_ratio[i] = min(cnl1, cnl2) / max(1, max(cnl1, cnl2))

            exact_orig_name[i] = 1.0 if (n1["orig_norm"] and n1["orig_norm"] == n2["orig_norm"]) else 0.0
            exact_norm_name[i] = 1.0 if (n1["lower_norm"] and n1["lower_norm"] == n2["lower_norm"]) else 0.0
            exact_core_name[i] = 1.0 if (cnl1 >= 3 and n1["core_name_norm"] == n2["core_name_norm"]) else 0.0
            exact_compact_name[i] = 1.0 if (len(n1["compact_norm"]) >= 3 and n1["compact_norm"] == n2["compact_norm"]) else 0.0
            first_core_token_exact[i] = 1.0 if (e1["first_core_tok"] and e1["first_core_tok"] == e2["first_core_tok"]) else 0.0

            # Suffix agreement
            sc1 = n1["suffix_class"]
            sc2 = n2["suffix_class"]
            if sc1 and sc2:
                business_suffix_agreement[i] = 1.0 if sc1 == sc2 else -1.0

            # Name char n-grams
            name_char_3gram_jaccard[i] = _set_jaccard(e1["n_c3"], e2["n_c3"])
            name_char_4gram_jaccard[i] = _set_jaccard(e1["n_c4"], e2["n_c4"])
            name_char_5gram_jaccard[i] = _set_jaccard(e1["n_c5"], e2["n_c5"])

            # Name token sets
            nt1 = e1["n_tok_set"]
            nt2 = e2["n_tok_set"]
            name_token_jaccard[i] = _set_jaccard(nt1, nt2)

            ct1 = e1["n_core_set"]
            ct2 = e2["n_core_set"]
            c_inter = ct1 & ct2
            n_c_inter = len(c_inter)
            l_ct1 = len(ct1)
            l_ct2 = len(ct2)
            name_common_token_count[i] = n_c_inter
            name_token_count_min[i] = min(l_ct1, l_ct2)
            name_token_count_diff[i] = abs(l_ct1 - l_ct2)
            if l_ct1 and l_ct2:
                core_name_token_jaccard[i] = n_c_inter / (l_ct1 + l_ct2 - n_c_inter)
                core_name_min_containment[i] = n_c_inter / min(l_ct1, l_ct2)
                core_name_max_containment[i] = n_c_inter / max(l_ct1, l_ct2)
                idf_inter = sum(name_idf.get(t, 0.5) for t in c_inter)
                idf_union = sum(name_idf.get(t, 0.5) for t in (ct1 | ct2))
                name_rare_token_idf_overlap[i] = idf_inter
                name_rare_token_idf_ratio[i] = idf_inter / max(1e-6, idf_union)

            # Phonetic features
            phonetic_name_token_jaccard[i] = _set_jaccard(e1["n_phon_set"], e2["n_phon_set"])
            phonetic_exact_core_name[i] = 1.0 if (len(n1["phonetic_str"]) >= 3 and n1["phonetic_str"] == n2["phonetic_str"]) else 0.0

            # Address presence & exact flags
            has_a1 = len(a1["lower_norm"]) > 0
            has_a2 = len(a2["lower_norm"]) > 0
            if has_a1 and has_a2:
                addr_both_present[i] = 1.0
                addr_len_ratio[i] = min(len(a1["lower_norm"]), len(a2["lower_norm"])) / max(1, max(len(a1["lower_norm"]), len(a2["lower_norm"])))
            else:
                addr_either_missing[i] = 1.0

            exact_orig_addr[i] = 1.0 if (a1["orig_norm"] and a1["orig_norm"] == a2["orig_norm"]) else 0.0
            exact_norm_addr[i] = 1.0 if (has_a1 and a1["lower_norm"] == a2["lower_norm"]) else 0.0

            # Address char n-grams & tokens
            if has_a1 and has_a2:
                addr_char_3gram_jaccard[i] = _set_jaccard(e1["a_c3"], e2["a_c3"])
                addr_char_4gram_jaccard[i] = _set_jaccard(e1["a_c4"], e2["a_c4"])
                addr_char_5gram_jaccard[i] = _set_jaccard(e1["a_c5"], e2["a_c5"])

                at1 = e1["a_tok_set"]
                at2 = e2["a_tok_set"]
                a_inter = at1 & at2
                n_a_inter = len(a_inter)
                l_at1 = len(at1)
                l_at2 = len(at2)
                addr_common_token_count[i] = n_a_inter
                if l_at1 and l_at2:
                    addr_token_jaccard[i] = n_a_inter / (l_at1 + l_at2 - n_a_inter)
                    addr_min_containment[i] = n_a_inter / min(l_at1, l_at2)
                    addr_max_containment[i] = n_a_inter / max(l_at1, l_at2)
                    a_idf_inter = sum(addr_idf.get(t, 0.5) for t in a_inter)
                    a_idf_union = sum(addr_idf.get(t, 0.5) for t in (at1 | at2))
                    addr_rare_token_idf_overlap[i] = a_idf_inter
                    addr_rare_token_idf_ratio[i] = a_idf_inter / max(1e-6, a_idf_union)

            # State / region agreement
            st1 = a1["state_code"]
            st2 = a2["state_code"]
            if st1 and st2:
                if st1 == st2:
                    state_region_match[i] = 1.0
                else:
                    state_region_conflict[i] = 1.0

            # Numeric & Building / Postal / Unit features
            b1 = a1["building_num"]
            b2 = a2["building_num"]
            building_extraction_conf_min[i] = min(a1["building_conf"], a2["building_conf"])
            if b1 and b2:
                if b1 == b2:
                    building_number_match[i] = 1.0
                else:
                    # Check if primary numeric stem matches (e.g. '1839/7' vs '1839')
                    b1_parts = b1.split("/")
                    b2_parts = b2.split("/")
                    if b1_parts[0] == b2_parts[0] and len(b1_parts[0]) >= 2:
                        building_number_match[i] = 0.5
                    else:
                        building_number_conflict[i] = 1.0

            p1 = a1["postal_code"]
            p2 = a2["postal_code"]
            postal_extraction_conf_min[i] = min(a1["postal_conf"], a2["postal_conf"])
            if p1 and p2:
                if p1 == p2:
                    postal_match[i] = 1.0
                else:
                    postal_conflict[i] = 1.0

            u1 = a1["unit_num"]
            u2 = a2["unit_num"]
            if u1 and u2:
                if u1 == u2:
                    unit_number_match[i] = 1.0
                else:
                    unit_number_conflict[i] = 1.0

            num1 = a1["numeric_tokens"]
            num2 = a2["numeric_tokens"]
            numeric_count_difference[i] = abs(len(num1) - len(num2))
            if num1 and num2:
                num_inter = len(num1 & num2)
                numeric_token_overlap[i] = num_inter
                numeric_token_jaccard[i] = num_inter / (len(num1) + len(num2) - num_inter)

        # Interaction features (vectorized on batch arrays)
        name_x_addr_sim = norm_name_token_set * norm_addr_token_set
        core_name_x_norm_addr_set = core_name_token_set * norm_addr_token_set
        name_addr_agreement = 0.5 * (core_name_ratio + norm_addr_token_set)
        name_addr_conflict = core_name_token_set - norm_addr_token_set
        high_name_low_addr = ((core_name_token_set >= 0.90) & (addr_both_present > 0.5) & (norm_addr_token_set < 0.62)).astype(np.float32)
        high_name_high_addr = ((core_name_token_set >= 0.85) & (norm_addr_token_set >= 0.80)).astype(np.float32)
        exact_name_plus_addr_ev = exact_core_name * (norm_addr_token_set + 0.5 * building_number_match)
        exact_addr_plus_name_ev = exact_norm_addr * core_name_token_set
        num_agree_x_name_sim = building_number_match * core_name_token_set
        num_conflict_x_name_sim = building_number_conflict * core_name_token_set
        postal_agree_x_name_sim = postal_match * core_name_token_set
        postal_conflict_x_name_sim = postal_conflict * core_name_token_set

        batch_df = pd.DataFrame({
            # Base 11
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
            # V3 Name (30)
            "norm_name_ratio": norm_name_ratio,
            "norm_name_token_sort": norm_name_token_sort,
            "norm_name_token_set": norm_name_token_set,
            "norm_name_partial_ratio": norm_name_partial_ratio,
            "core_name_ratio": core_name_ratio,
            "core_name_token_sort": core_name_token_sort,
            "core_name_token_set": core_name_token_set,
            "core_name_partial_ratio": core_name_partial_ratio,
            "compact_name_ratio": compact_name_ratio,
            "name_char_3gram_jaccard": name_char_3gram_jaccard,
            "name_char_4gram_jaccard": name_char_4gram_jaccard,
            "name_char_5gram_jaccard": name_char_5gram_jaccard,
            "name_token_jaccard": name_token_jaccard,
            "core_name_token_jaccard": core_name_token_jaccard,
            "core_name_min_containment": core_name_min_containment,
            "core_name_max_containment": core_name_max_containment,
            "name_common_token_count": name_common_token_count,
            "name_rare_token_idf_overlap": name_rare_token_idf_overlap,
            "name_rare_token_idf_ratio": name_rare_token_idf_ratio,
            "name_len_ratio": name_len_ratio,
            "core_name_len_ratio": core_name_len_ratio,
            "name_token_count_min": name_token_count_min,
            "name_token_count_diff": name_token_count_diff,
            "exact_orig_name": exact_orig_name,
            "exact_norm_name": exact_norm_name,
            "exact_core_name": exact_core_name,
            "exact_compact_name": exact_compact_name,
            "business_suffix_agreement": business_suffix_agreement,
            "first_core_token_exact": first_core_token_exact,
            "first_core_token_ratio": first_core_token_ratio,
            # V3 Address (20)
            "norm_addr_ratio": norm_addr_ratio,
            "norm_addr_token_sort": norm_addr_token_sort,
            "norm_addr_token_set": norm_addr_token_set,
            "norm_addr_partial_ratio": norm_addr_partial_ratio,
            "addr_char_3gram_jaccard": addr_char_3gram_jaccard,
            "addr_char_4gram_jaccard": addr_char_4gram_jaccard,
            "addr_char_5gram_jaccard": addr_char_5gram_jaccard,
            "addr_token_jaccard": addr_token_jaccard,
            "addr_min_containment": addr_min_containment,
            "addr_max_containment": addr_max_containment,
            "addr_common_token_count": addr_common_token_count,
            "addr_rare_token_idf_overlap": addr_rare_token_idf_overlap,
            "addr_rare_token_idf_ratio": addr_rare_token_idf_ratio,
            "exact_orig_addr": exact_orig_addr,
            "exact_norm_addr": exact_norm_addr,
            "addr_len_ratio": addr_len_ratio,
            "addr_both_present": addr_both_present,
            "addr_either_missing": addr_either_missing,
            "state_region_match": state_region_match,
            "state_region_conflict": state_region_conflict,
            # V3 Numeric (11)
            "building_number_match": building_number_match,
            "building_number_conflict": building_number_conflict,
            "building_extraction_conf_min": building_extraction_conf_min,
            "postal_match": postal_match,
            "postal_conflict": postal_conflict,
            "postal_extraction_conf_min": postal_extraction_conf_min,
            "numeric_token_jaccard": numeric_token_jaccard,
            "numeric_token_overlap": numeric_token_overlap,
            "numeric_count_difference": numeric_count_difference,
            "unit_number_match": unit_number_match,
            "unit_number_conflict": unit_number_conflict,
            # V3 Interaction (13)
            "name_x_addr_sim": name_x_addr_sim,
            "core_name_x_norm_addr_set": core_name_x_norm_addr_set,
            "name_addr_agreement": name_addr_agreement,
            "name_addr_conflict": name_addr_conflict,
            "high_name_low_addr": high_name_low_addr,
            "high_name_high_addr": high_name_high_addr,
            "exact_name_plus_addr_ev": exact_name_plus_addr_ev,
            "exact_addr_plus_name_ev": exact_addr_plus_name_ev,
            "num_agree_x_name_sim": num_agree_x_name_sim,
            "num_conflict_x_name_sim": num_conflict_x_name_sim,
            "postal_agree_x_name_sim": postal_agree_x_name_sim,
            "postal_conflict_x_name_sim": postal_conflict_x_name_sim,
            "is_s2_candidate": is_s2_candidate,
            # V3 Phonetic (3)
            "phonetic_name_ratio": phonetic_name_ratio,
            "phonetic_name_token_jaccard": phonetic_name_token_jaccard,
            "phonetic_exact_core_name": phonetic_exact_core_name,
        })
        out_chunks.append(batch_df)
        del cand_cache
        if n_total >= 10_000:
            el_b = time.time() - t_b0
            el_tot = time.time() - t_f0
            print(
                f"  [Features {end:,}/{n_total:,}] batch={el_b:.1f}s | total={el_tot:.1f}s ({end/max(1e-6, el_tot):,.0f} pairs/s)",
                flush=True,
            )

    return pd.concat(out_chunks, ignore_index=True) if len(out_chunks) > 1 else out_chunks[0]


if __name__ == "__main__":
    # Explicit unit test required by prompt:
    # "31 Eagle Park Circle" vs "46 Eagle Park Circle" must generate a strong conflict signal.
    test_pairs = pd.DataFrame({
        "source1_entity_id": ["S1-975065352", "S1-883582340"],
        "candidate_entity_id": ["S3-888442365", "S2-11552053"],
    })
    s1_lkp = {
        "S1-975065352": {
            "business_name": "Sterling Restaurant Corp",
            "business_address": "Banner Elk, 31 Eagle Park Circle, NC",
            "country": "us",
        },
        "S1-883582340": {
            "business_name": "Bakratunda Foods Private Limited",
            "business_address": "15 Dholakia Mansionghoghari Mohalla 1St Cross Lane Phydhonie, Mumbai, Maharashtra",
            "country": "india",
        },
    }
    tgt_lkp = {
        "S3-888442365": {
            "business_name": "Sterling Restaurant Co",
            "business_address": "Banner Elk, 46 Eagle Park Circle, North Carolina",
            "country": "us",
        },
        "S2-11552053": {
            "business_name": "Bakratunda Foods Private",
            "business_address": "15 DHOLAKIA MANSIONGHOGHARI MOHALLA 1ST CROSS LANE PHYDHONIE, MUMBAI, Maharashtra",
            "country": "india",
        },
    }
    df_feat = compute_v3_features(test_pairs, s1_lkp, tgt_lkp)
    print("Total V3 features:", len(df_feat.columns))
    print("\nPair 1 (31 Eagle Park Circle vs 46 Eagle Park Circle):")
    print("  building_number_conflict:", df_feat.loc[0, "building_number_conflict"])
    print("  building_number_match:   ", df_feat.loc[0, "building_number_match"])
    print("  numeric_token_jaccard:   ", df_feat.loc[0, "numeric_token_jaccard"])
    print("  num_conflict_x_name_sim: ", df_feat.loc[0, "num_conflict_x_name_sim"])
    assert df_feat.loc[0, "building_number_conflict"] == 1.0
    assert df_feat.loc[0, "building_number_match"] == 0.0
    assert df_feat.loc[0, "numeric_token_jaccard"] == 0.0

    print("\nPair 2 (S2 UPPERCASE address vs S1 Title Case address):")
    print("  raw address_ratio:       ", df_feat.loc[1, "address_ratio"])
    print("  V3 norm_addr_ratio:      ", df_feat.loc[1, "norm_addr_ratio"])
    print("  V3 exact_norm_addr:      ", df_feat.loc[1, "exact_norm_addr"])
    print("  V3 building_number_match:", df_feat.loc[1, "building_number_match"])
    assert df_feat.loc[1, "norm_addr_ratio"] == 1.0
    assert df_feat.loc[1, "exact_norm_addr"] == 1.0
    assert df_feat.loc[1, "building_number_match"] == 1.0
    print("\nALL V3-B UNIT TESTS PASSED!")
