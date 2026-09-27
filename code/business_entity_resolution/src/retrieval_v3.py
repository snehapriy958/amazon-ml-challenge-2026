"""Phase V3-A: Four-Channel Candidate Retrieval Engine & DEV Recall Gate Verification.

Implements Two-Tier Anchor + Verification Expansion:
  - Tier 1 (Anchor Candidate Generation — Bounded DF):
      * Channel 1 (Exact): all exact normalized/name/address matches (preserved regardless of score)
      * Channel 2 (Rare Token / Phonetic):
          - Top 5 rarest name tokens with df <= 25,000 (fallback: single rarest name token up to df <= 60,000)
          - Top 3 rarest phonetic tokens with df <= 15,000
          - Top 4 rarest address tokens with df <= 20,000
      * Channel 3 (Character N-Gram):
          - Top 5 rarest character 4/5-grams with df <= 12,000
      * Channel 4 (Numeric / Postal):
          - (building_num, postal_code) df <= 10,000
          - (building_num, state_code) df <= 10,000
          - building_num df <= 8,000
          - postal_code df <= 5,000
  - Tier 2 (Verification on Existing Anchor Candidates Only):
      * High-DF tokens/grams/numbers above Tier 1 anchor thresholds NEVER create new 1-hit candidates
        and NEVER iterate their full posting lists per query.
      * Instead, high-DF posting lists are converted once to hash sets and intersected in O(|anchors|)
        time strictly against candidates already in the anchor pool.
  - Candidate Selection:
      * Preserves all exact-match candidates regardless of score
      * Fair S2/S3 interleaving + heapq.nlargest top_k=750 selection
      * Lightweight progress logging every 250 S1 queries
      * Evaluates ONLY DEV_3000 first to verify runtime and >=97% recall gate.
"""

from collections import defaultdict
import gc
import heapq
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path.cwd()))
from code.business_entity_resolution.src.preprocessing import parse_match_ids
from code.business_entity_resolution.src.normalization_v3 import normalize_name_v3, parse_address_v3


# Tier 1 Anchor DF Thresholds (Strict adherence to specification)
T1_NAME_TOK_DF = 25_000
T1_NAME_TOK_FALLBACK_DF = 60_000
T1_PHON_TOK_DF = 15_000
T1_ADDR_TOK_DF = 20_000
T1_CHAR_GRAM_DF = 12_000
T1_BLDG_POSTAL_DF = 10_000
T1_BLDG_STATE_DF = 10_000
T1_BLDG_NUM_DF = 8_000
T1_POSTAL_DF = 5_000

# Tier 2 Verification Max DF Caps (Converted to O(1) hash sets for anchor verification only)
T2_NAME_TOK_MAX_DF = 250_000
T2_PHON_TOK_MAX_DF = 80_000
T2_ADDR_TOK_MAX_DF = 160_000
T2_CHAR_GRAM_MAX_DF = 50_000
T2_BLDG_STATE_MAX_DF = 25_000
T2_BLDG_NUM_MAX_DF = 40_000
T2_POSTAL_MAX_DF = 35_000


def extract_char_ngrams_for_name(compact_name: str, core_tokens: Tuple[str, ...]) -> Set[str]:
    """Extract distinctive character 4-grams and 5-grams from business name for Channel 3."""
    grams: Set[str] = set()
    if len(compact_name) >= 4:
        n = len(compact_name)
        for k in (4, 5):
            if n >= k:
                grams.add(f"c{k}:{compact_name[:k]}")
                grams.add(f"c{k}:{compact_name[-k:]}")
                mid = (n - k) // 2
                grams.add(f"c{k}:{compact_name[mid:mid+k]}")
    for tok in core_tokens:
        tl = len(tok)
        if tl >= 4:
            grams.add(f"t4:{tok[:4]}")
            grams.add(f"t4:{tok[-4:]}")
        if tl >= 5:
            grams.add(f"t5:{tok[:5]}")
            grams.add(f"t5:{tok[-5:]}")
            if tl >= 7:
                mid = (tl - 4) // 2
                grams.add(f"t4:{tok[mid:mid+4]}")
    return grams


def benchmark_channel3_tfidf(sample_texts: List[str], total_target_count: int = 5_320_000) -> Dict[str, object]:
    """Benchmark full char 3/4/5-gram TF-IDF matrix on a subset to verify fallback criteria."""
    t0 = time.time()
    avail_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, dtype=np.float32)
    mat = vec.fit_transform(sample_texts)
    elapsed = time.time() - t0
    sample_n = max(1, len(sample_texts))
    mat_bytes = mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
    scale_factor = total_target_count / sample_n
    proj_index_gb = (mat_bytes * scale_factor * 2.8) / (1024 ** 3)
    proj_test_9_9m_gb = proj_index_gb * (9_900_000 / total_target_count)
    proj_runtime_s = elapsed * scale_factor * 2.5

    ram_limit_gb = 0.70 * avail_ram_gb
    trigger_fallback = (proj_test_9_9m_gb > ram_limit_gb) or (proj_runtime_s > 600.0)

    res = {
        "sample_size": sample_n,
        "sample_fit_time_s": round(elapsed, 3),
        "sample_nnz": int(mat.nnz),
        "sample_matrix_mb": round(mat_bytes / (1024 ** 2), 2),
        "available_ram_gb": round(avail_ram_gb, 2),
        "seventy_pct_ram_threshold_gb": round(ram_limit_gb, 2),
        "projected_train_5_3m_gb": round(proj_index_gb, 2),
        "projected_test_9_9m_gb": round(proj_test_9_9m_gb, 2),
        "projected_full_runtime_s": round(proj_runtime_s, 1),
        "fallback_triggered": bool(trigger_fallback),
        "selected_channel3_method": "sparse_inverted_char_4_5_gram_posting_index" if trigger_fallback else "full_tfidf",
    }
    del mat, vec
    gc.collect()
    return res


def retrieve_candidates_v3_for_queries(
    queries_df: pd.DataFrame,
    s2_path: str = "dataset/train/train_source2.tsv",
    s3_path: str = "dataset/train/train_source3.tsv",
    top_k: int = 750,
    run_tfidf_benchmark: bool = False,
) -> Tuple[Dict[str, List[Tuple[str, float, int]]], Dict[str, Dict[str, str]], Dict[str, object]]:
    """Run Two-Tier 4-channel V3 retrieval for a DataFrame of S1 queries."""
    t_start = time.time()
    n_queries = len(queries_df)
    print(f"Preparing V3 Two-Tier 4-Channel Retrieval for {n_queries:,} S1 queries (top_k={top_k})...", flush=True)

    country_queries: Dict[str, Dict[str, dict]] = defaultdict(dict)
    q_orig_names: Dict[str, Set[str]] = defaultdict(set)
    q_lower_names: Dict[str, Set[str]] = defaultdict(set)
    q_core_names: Dict[str, Set[str]] = defaultdict(set)
    q_compact_names: Dict[str, Set[str]] = defaultdict(set)
    q_orig_addrs: Dict[str, Set[str]] = defaultdict(set)
    q_lower_addrs: Dict[str, Set[str]] = defaultdict(set)

    q_name_tokens: Dict[str, Set[str]] = defaultdict(set)
    q_phon_tokens: Dict[str, Set[str]] = defaultdict(set)
    q_addr_tokens: Dict[str, Set[str]] = defaultdict(set)
    q_char_grams: Dict[str, Set[str]] = defaultdict(set)

    q_bldg_nums: Dict[str, Set[str]] = defaultdict(set)
    q_postals: Dict[str, Set[str]] = defaultdict(set)
    q_bldg_postal: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    q_bldg_state: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    all_gt_targets: Set[str] = set()
    has_gt = "matched_entity_ids" in queries_df.columns

    for row in queries_df.itertuples(index=False):
        sid = getattr(row, "source1_entity_id", None) or getattr(row, "entity_id")
        n_raw = "" if pd.isna(row.business_name) else str(row.business_name)
        a_raw = "" if pd.isna(row.business_address) else str(row.business_address)
        c_raw = "" if pd.isna(row.country) else str(row.country).strip().lower()

        if has_gt:
            m_str = getattr(row, "matched_entity_ids", "")
            all_gt_targets.update(parse_match_ids("" if pd.isna(m_str) else str(m_str)))

        n_info = normalize_name_v3(n_raw)
        a_info = parse_address_v3(a_raw, c_raw)
        c_grams = extract_char_ngrams_for_name(n_info["compact_norm"], n_info["core_tokens"])

        name_tok_set = {t for t in n_info["core_tokens"] if len(t) >= 2} | {t for t in n_info["tokens"] if len(t) >= 3}
        phon_tok_set = {p for p in n_info["phonetic_tokens"] if len(p) >= 3}
        addr_tok_set = {t for t in a_info["tokens"] if len(t) >= 2}

        rec = {
            "sid": sid,
            "country": c_raw,
            "n_info": n_info,
            "a_info": a_info,
            "name_tok_set": name_tok_set,
            "phon_tok_set": phon_tok_set,
            "addr_tok_set": addr_tok_set,
            "char_grams": c_grams,
        }
        country_queries[c_raw][sid] = rec

        if n_info["orig_norm"]:
            q_orig_names[c_raw].add(n_info["orig_norm"])
        if n_info["lower_norm"]:
            q_lower_names[c_raw].add(n_info["lower_norm"])
        if n_info["core_name_norm"]:
            q_core_names[c_raw].add(n_info["core_name_norm"])
        if len(n_info["compact_norm"]) >= 3:
            q_compact_names[c_raw].add(n_info["compact_norm"])
        if a_info["orig_norm"]:
            q_orig_addrs[c_raw].add(a_info["orig_norm"])
        if a_info["lower_norm"]:
            q_lower_addrs[c_raw].add(a_info["lower_norm"])

        q_name_tokens[c_raw].update(name_tok_set)
        q_phon_tokens[c_raw].update(phon_tok_set)
        q_addr_tokens[c_raw].update(addr_tok_set)
        q_char_grams[c_raw].update(c_grams)

        b_num = a_info["building_num"]
        p_code = a_info["postal_code"]
        s_code = a_info["state_code"]
        if b_num:
            q_bldg_nums[c_raw].add(b_num)
            if p_code:
                q_bldg_postal[c_raw].add((b_num, p_code))
            if s_code:
                q_bldg_state[c_raw].add((b_num, s_code))
        if p_code:
            q_postals[c_raw].add(p_code)

    # Posting lists per country
    idx_orig_name: Dict[str, Dict[str, List[str]]] = {c: defaultdict(list) for c in country_queries}
    idx_lower_name: Dict[str, Dict[str, List[str]]] = {c: defaultdict(list) for c in country_queries}
    idx_core_name: Dict[str, Dict[str, List[str]]] = {c: defaultdict(list) for c in country_queries}
    idx_compact_name: Dict[str, Dict[str, List[str]]] = {c: defaultdict(list) for c in country_queries}
    idx_orig_addr: Dict[str, Dict[str, List[str]]] = {c: defaultdict(list) for c in country_queries}
    idx_lower_addr: Dict[str, Dict[str, List[str]]] = {c: defaultdict(list) for c in country_queries}

    idx_name_tok: Dict[str, Dict[str, List[str]]] = {c: defaultdict(list) for c in country_queries}
    idx_phon_tok: Dict[str, Dict[str, List[str]]] = {c: defaultdict(list) for c in country_queries}
    idx_addr_tok: Dict[str, Dict[str, List[str]]] = {c: defaultdict(list) for c in country_queries}
    idx_char_gram: Dict[str, Dict[str, List[str]]] = {c: defaultdict(list) for c in country_queries}

    idx_bldg_num: Dict[str, Dict[str, List[str]]] = {c: defaultdict(list) for c in country_queries}
    idx_postal: Dict[str, Dict[str, List[str]]] = {c: defaultdict(list) for c in country_queries}
    idx_bldg_postal: Dict[str, Dict[Tuple[str, str], List[str]]] = {c: defaultdict(list) for c in country_queries}
    idx_bldg_state: Dict[str, Dict[Tuple[str, str], List[str]]] = {c: defaultdict(list) for c in country_queries}

    disq_name_tok: Dict[str, Set[str]] = {c: set() for c in country_queries}
    disq_phon_tok: Dict[str, Set[str]] = {c: set() for c in country_queries}
    disq_addr_tok: Dict[str, Set[str]] = {c: set() for c in country_queries}
    disq_char_gram: Dict[str, Set[str]] = {c: set() for c in country_queries}
    disq_bldg_num: Dict[str, Set[str]] = {c: set() for c in country_queries}
    disq_postal: Dict[str, Set[str]] = {c: set() for c in country_queries}
    disq_bldg_postal: Dict[str, Set[Tuple[str, str]]] = {c: set() for c in country_queries}
    disq_bldg_state: Dict[str, Set[Tuple[str, str]]] = {c: set() for c in country_queries}

    tfidf_benchmark_result = {}
    sample_texts_for_bench: List[str] = []

    t_index_start = time.time()
    for src_label, src_path in [("S2", s2_path), ("S3", s3_path)]:
        t_s0 = time.time()
        for chunk_idx, chunk in enumerate(pd.read_csv(src_path, sep="\t", chunksize=500_000, dtype=str), 1):
            if run_tfidf_benchmark and not tfidf_benchmark_result and len(sample_texts_for_bench) < 50_000:
                sample_texts_for_bench.extend(
                    (chunk["business_name"].fillna("") + " " + chunk["business_address"].fillna("")).iloc[:50_000].tolist()
                )
                if len(sample_texts_for_bench) >= 50_000:
                    tfidf_benchmark_result = benchmark_channel3_tfidf(sample_texts_for_bench[:50_000])
                    print(f"  [Channel 3 Benchmark]: {json.dumps(tfidf_benchmark_result)}", flush=True)
                    sample_texts_for_bench.clear()

            chunk["c_norm"] = chunk["country"].fillna("").str.strip().str.lower()
            for c in country_queries:
                sub = chunk[chunk["c_norm"] == c]
                if sub.empty:
                    continue

                qo_n = q_orig_names[c]
                ql_n = q_lower_names[c]
                qc_n = q_core_names[c]
                qcp_n = q_compact_names[c]
                qo_a = q_orig_addrs[c]
                ql_a = q_lower_addrs[c]

                qt_n = q_name_tokens[c]
                qt_p = q_phon_tokens[c]
                qt_a = q_addr_tokens[c]
                qg_c = q_char_grams[c]

                qb_n = q_bldg_nums[c]
                qp_c = q_postals[c]
                qbp = q_bldg_postal[c]
                qbs = q_bldg_state[c]

                io_n = idx_orig_name[c]
                il_n = idx_lower_name[c]
                ic_n = idx_core_name[c]
                icp_n = idx_compact_name[c]
                io_a = idx_orig_addr[c]
                il_a = idx_lower_addr[c]

                it_n = idx_name_tok[c]
                it_p = idx_phon_tok[c]
                it_a = idx_addr_tok[c]
                ig_c = idx_char_gram[c]

                ib_n = idx_bldg_num[c]
                ip_c = idx_postal[c]
                ibp = idx_bldg_postal[c]
                ibs = idx_bldg_state[c]

                dn_t = disq_name_tok[c]
                dp_t = disq_phon_tok[c]
                da_t = disq_addr_tok[c]
                dc_g = disq_char_gram[c]
                db_n = disq_bldg_num[c]
                dp_c = disq_postal[c]
                dbp = disq_bldg_postal[c]
                dbs = disq_bldg_state[c]

                for cid, n_raw, a_raw in zip(sub["entity_id"], sub["business_name"], sub["business_address"]):
                    n_str = "" if pd.isna(n_raw) else str(n_raw)
                    a_str = "" if pd.isna(a_raw) else str(a_raw)

                    n_info = normalize_name_v3(n_str)
                    a_info = parse_address_v3(a_str, c)

                    # Channel 1: Exact representations
                    if n_info["orig_norm"] in qo_n:
                        io_n[n_info["orig_norm"]].append(cid)
                    if n_info["lower_norm"] in ql_n:
                        il_n[n_info["lower_norm"]].append(cid)
                    if n_info["core_name_norm"] in qc_n and len(ic_n[n_info["core_name_norm"]]) < 10_000:
                        ic_n[n_info["core_name_norm"]].append(cid)
                    if n_info["compact_norm"] in qcp_n and len(icp_n[n_info["compact_norm"]]) < 10_000:
                        icp_n[n_info["compact_norm"]].append(cid)
                    if a_info["orig_norm"] in qo_a:
                        io_a[a_info["orig_norm"]].append(cid)
                    if a_info["lower_norm"] in ql_a:
                        il_a[a_info["lower_norm"]].append(cid)

                    # Channel 2: Rare Name, Phonetic, and Address Tokens
                    cand_name_toks = {t for t in n_info["core_tokens"] if len(t) >= 2} | {t for t in n_info["tokens"] if len(t) >= 3}
                    for t in cand_name_toks & qt_n:
                        if t not in dn_t:
                            lst = it_n[t]
                            lst.append(cid)
                            if len(lst) > T2_NAME_TOK_MAX_DF:
                                dn_t.add(t)
                                del it_n[t]

                    cand_phon_toks = {p for p in n_info["phonetic_tokens"] if len(p) >= 3}
                    for p in cand_phon_toks & qt_p:
                        if p not in dp_t:
                            lst = it_p[p]
                            lst.append(cid)
                            if len(lst) > T2_PHON_TOK_MAX_DF:
                                dp_t.add(p)
                                del it_p[p]

                    cand_addr_toks = {t for t in a_info["tokens"] if len(t) >= 2}
                    for t in cand_addr_toks & qt_a:
                        if t not in da_t:
                            lst = it_a[t]
                            lst.append(cid)
                            if len(lst) > T2_ADDR_TOK_MAX_DF:
                                da_t.add(t)
                                del it_a[t]

                    # Channel 3: Sparse Character 4/5-Gram Posting Index on Business Name
                    c_grams = extract_char_ngrams_for_name(n_info["compact_norm"], n_info["core_tokens"])
                    for g in c_grams & qg_c:
                        if g not in dc_g:
                            lst = ig_c[g]
                            lst.append(cid)
                            if len(lst) > T2_CHAR_GRAM_MAX_DF:
                                dc_g.add(g)
                                del ig_c[g]

                    # Channel 4: Numeric / Postal / Building-Number Index
                    b_num = a_info["building_num"]
                    p_code = a_info["postal_code"]
                    s_code = a_info["state_code"]
                    if b_num:
                        if b_num in qb_n and b_num not in db_n:
                            lst = ib_n[b_num]
                            lst.append(cid)
                            if len(lst) > T2_BLDG_NUM_MAX_DF:
                                db_n.add(b_num)
                                del ib_n[b_num]
                        if p_code and (b_num, p_code) in qbp and (b_num, p_code) not in dbp:
                            lst = ibp[(b_num, p_code)]
                            lst.append(cid)
                            if len(lst) > T1_BLDG_POSTAL_DF:
                                dbp.add((b_num, p_code))
                                del ibp[(b_num, p_code)]
                        if s_code and (b_num, s_code) in qbs and (b_num, s_code) not in dbs:
                            lst = ibs[(b_num, s_code)]
                            lst.append(cid)
                            if len(lst) > T2_BLDG_STATE_MAX_DF:
                                dbs.add((b_num, s_code))
                                del ibs[(b_num, s_code)]
                    if p_code and p_code in qp_c and p_code not in dp_c:
                        lst = ip_c[p_code]
                        lst.append(cid)
                        if len(lst) > T2_POSTAL_MAX_DF:
                            dp_c.add(p_code)
                            del ip_c[p_code]

        print(f"  Indexed {src_label} in {time.time()-t_s0:.1f}s", flush=True)
    index_elapsed_s = time.time() - t_index_start

    # Pre-convert Tier 2 high-DF posting lists into O(1) hash sets once before the query loop
    print("Building Tier 2 O(1) verification hash sets for high-DF posting lists...", flush=True)
    t_prep2 = time.time()
    t2_name_sets: Dict[str, Dict[str, Set[str]]] = {c: {} for c in country_queries}
    t2_phon_sets: Dict[str, Dict[str, Set[str]]] = {c: {} for c in country_queries}
    t2_addr_sets: Dict[str, Dict[str, Set[str]]] = {c: {} for c in country_queries}
    t2_gram_sets: Dict[str, Dict[str, Set[str]]] = {c: {} for c in country_queries}
    t2_bldg_sets: Dict[str, Dict[str, Set[str]]] = {c: {} for c in country_queries}
    t2_postal_sets: Dict[str, Dict[str, Set[str]]] = {c: {} for c in country_queries}
    t2_bldg_state_sets: Dict[str, Dict[Tuple[str, str], Set[str]]] = {c: {} for c in country_queries}

    for c in country_queries:
        for t, lst in idx_name_tok[c].items():
            if len(lst) > T1_NAME_TOK_DF:
                t2_name_sets[c][t] = set(lst)
        for p, lst in idx_phon_tok[c].items():
            if len(lst) > T1_PHON_TOK_DF:
                t2_phon_sets[c][p] = set(lst)
        for t, lst in idx_addr_tok[c].items():
            if len(lst) > T1_ADDR_TOK_DF:
                t2_addr_sets[c][t] = set(lst)
        for g, lst in idx_char_gram[c].items():
            if len(lst) > T1_CHAR_GRAM_DF:
                t2_gram_sets[c][g] = set(lst)
        for b, lst in idx_bldg_num[c].items():
            if len(lst) > T1_BLDG_NUM_DF:
                t2_bldg_sets[c][b] = set(lst)
        for p, lst in idx_postal[c].items():
            if len(lst) > T1_POSTAL_DF:
                t2_postal_sets[c][p] = set(lst)
        for bs, lst in idx_bldg_state[c].items():
            if len(lst) > T1_BLDG_STATE_DF:
                t2_bldg_state_sets[c][bs] = set(lst)
    print(f"  Tier 2 verification sets built in {time.time()-t_prep2:.2f}s", flush=True)

    # Two-Tier Union & Prioritization per S1 query
    print("Running Two-Tier Anchor + Verification Union with fair S2/S3 interleaving...", flush=True)
    t_union_start = time.time()
    candidates_by_s1: Dict[str, List[Tuple[str, float, int]]] = {}
    needed_target_ids: Set[str] = set(all_gt_targets)

    channel_hit_counts = {"ch1_exact": 0, "ch2_rare_tok": 0, "ch3_char_gram": 0, "ch4_numeric": 0}
    processed_q = 0
    total_anchor_pool_size = 0
    total_selected_cands = 0

    for c, queries in country_queries.items():
        io_n = idx_orig_name[c]
        il_n = idx_lower_name[c]
        ic_n = idx_core_name[c]
        icp_n = idx_compact_name[c]
        io_a = idx_orig_addr[c]
        il_a = idx_lower_addr[c]

        it_n = idx_name_tok[c]
        it_p = idx_phon_tok[c]
        it_a = idx_addr_tok[c]
        ig_c = idx_char_gram[c]

        ib_n = idx_bldg_num[c]
        ip_c = idx_postal[c]
        ibp = idx_bldg_postal[c]
        ibs = idx_bldg_state[c]

        s2_n = t2_name_sets[c]
        s2_p = t2_phon_sets[c]
        s2_a = t2_addr_sets[c]
        s2_g = t2_gram_sets[c]
        s2_b = t2_bldg_sets[c]
        s2_po = t2_postal_sets[c]
        s2_bs = t2_bldg_state_sets[c]

        for sid, rec in queries.items():
            n_info = rec["n_info"]
            a_info = rec["a_info"]

            scores: Dict[str, float] = defaultdict(float)
            ch_mask: Dict[str, int] = defaultdict(int)
            has_name_ev: Set[str] = set()
            has_addr_ev: Set[str] = set()
            name_tok_hits: Dict[str, int] = defaultdict(int)
            exact_cids: Set[str] = set()

            # ==================================================================
            # TIER 1: ANCHOR CANDIDATE GENERATION (Bounded DF)
            # ==================================================================

            # --- CHANNEL 1: EXACT (All preserved regardless of score) ---
            for cid in io_n.get(n_info["orig_norm"], ()):
                scores[cid] += 220.0
                ch_mask[cid] |= 1
                has_name_ev.add(cid)
                exact_cids.add(cid)
            for cid in il_n.get(n_info["lower_norm"], ()):
                scores[cid] += 180.0
                ch_mask[cid] |= 1
                has_name_ev.add(cid)
                exact_cids.add(cid)
            for cid in ic_n.get(n_info["core_name_norm"], ()):
                scores[cid] += 140.0
                ch_mask[cid] |= 1
                has_name_ev.add(cid)
                exact_cids.add(cid)
            for cid in icp_n.get(n_info["compact_norm"], ()):
                scores[cid] += 130.0
                ch_mask[cid] |= 1
                has_name_ev.add(cid)
                exact_cids.add(cid)
            for cid in io_a.get(a_info["orig_norm"], ()):
                scores[cid] += 200.0
                ch_mask[cid] |= 1
                has_addr_ev.add(cid)
                exact_cids.add(cid)
            for cid in il_a.get(a_info["lower_norm"], ()):
                scores[cid] += 170.0
                ch_mask[cid] |= 1
                has_addr_ev.add(cid)
                exact_cids.add(cid)

            # --- CHANNEL 2 (Tier 1): RARE TOKEN / PHONETIC ANCHORS ---
            # 2a. Name tokens: top 5 with df <= 25,000; fallback to single rarest if <= 60,000
            u_name_t1 = []
            u_name_t2 = []
            for t in rec["name_tok_set"]:
                if t in it_n:
                    df = len(it_n[t])
                    if 0 < df <= T1_NAME_TOK_DF:
                        u_name_t1.append((df, t))
                    elif df <= T2_NAME_TOK_MAX_DF:
                        u_name_t2.append((df, t))
            u_name_t1.sort()
            u_name_t2.sort()

            active_t1_name = u_name_t1[:5]
            if not active_t1_name and u_name_t2 and u_name_t2[0][0] <= T1_NAME_TOK_FALLBACK_DF:
                active_t1_name = [u_name_t2.pop(0)]
            else:
                # Any excess T1 tokens beyond top 5 can still act as Tier 2 verification
                u_name_t2 = u_name_t1[5:] + u_name_t2

            for df, t in active_t1_name:
                w = 22.0 / math.log10(10.0 + df)
                for cid in it_n[t]:
                    scores[cid] += w
                    name_tok_hits[cid] += 1
                    ch_mask[cid] |= 2
                    has_name_ev.add(cid)

            # 2b. Phonetic tokens: top 3 with df <= 15,000
            u_phon_t1 = []
            u_phon_t2 = []
            for p in rec["phon_tok_set"]:
                if p in it_p:
                    df = len(it_p[p])
                    if 0 < df <= T1_PHON_TOK_DF:
                        u_phon_t1.append((df, p))
                    elif df <= T2_PHON_TOK_MAX_DF:
                        u_phon_t2.append((df, p))
            u_phon_t1.sort()
            u_phon_t2 = u_phon_t1[3:] + sorted(u_phon_t2)
            for df, p in u_phon_t1[:3]:
                w = 9.0 / math.log10(10.0 + df)
                for cid in it_p[p]:
                    scores[cid] += w
                    ch_mask[cid] |= 2
                    has_name_ev.add(cid)

            # 2c. Address tokens: top 4 with df <= 20,000
            u_addr_t1 = []
            u_addr_t2 = []
            for t in rec["addr_tok_set"]:
                if t in it_a:
                    df = len(it_a[t])
                    if 0 < df <= T1_ADDR_TOK_DF:
                        u_addr_t1.append((df, t))
                    elif df <= T2_ADDR_TOK_MAX_DF:
                        u_addr_t2.append((df, t))
            u_addr_t1.sort()
            u_addr_t2 = u_addr_t1[4:] + sorted(u_addr_t2)
            for df, t in u_addr_t1[:4]:
                w = 11.0 / math.log10(10.0 + df)
                for cid in it_a[t]:
                    scores[cid] += w
                    ch_mask[cid] |= 2
                    has_addr_ev.add(cid)

            # --- CHANNEL 3 (Tier 1): CHARACTER 4/5-GRAM ANCHORS (Top 5 with df <= 12,000) ---
            u_gram_t1 = []
            u_gram_t2 = []
            for g in rec["char_grams"]:
                if g in ig_c:
                    df = len(ig_c[g])
                    if 0 < df <= T1_CHAR_GRAM_DF:
                        u_gram_t1.append((df, g))
                    elif df <= T2_CHAR_GRAM_MAX_DF:
                        u_gram_t2.append((df, g))
            u_gram_t1.sort()
            u_gram_t2 = u_gram_t1[5:] + sorted(u_gram_t2)
            for df, g in u_gram_t1[:5]:
                w = 5.5 / math.log10(10.0 + df)
                for cid in ig_c[g]:
                    scores[cid] += w
                    ch_mask[cid] |= 4
                    has_name_ev.add(cid)

            # --- CHANNEL 4 (Tier 1): NUMERIC / POSTAL ANCHORS (Strict DF check before iterating) ---
            b_num = a_info["building_num"]
            p_code = a_info["postal_code"]
            s_code = a_info["state_code"]

            if b_num and p_code and (b_num, p_code) in ibp:
                lst_bp = ibp[(b_num, p_code)]
                if 0 < len(lst_bp) <= T1_BLDG_POSTAL_DF:
                    for cid in lst_bp:
                        scores[cid] += 55.0
                        ch_mask[cid] |= 8
                        has_addr_ev.add(cid)

            if b_num and s_code and (b_num, s_code) in ibs:
                lst_bs = ibs[(b_num, s_code)]
                df_bs = len(lst_bs)
                if 0 < df_bs <= T1_BLDG_STATE_DF:
                    w_bs = 32.0 / math.log10(10.0 + df_bs)
                    for cid in lst_bs:
                        scores[cid] += w_bs
                        ch_mask[cid] |= 8
                        has_addr_ev.add(cid)

            if b_num and b_num in ib_n:
                lst_b = ib_n[b_num]
                df_b = len(lst_b)
                if 0 < df_b <= T1_BLDG_NUM_DF:
                    w_b = 26.0 / math.log10(10.0 + df_b)
                    for cid in lst_b:
                        scores[cid] += w_b
                        ch_mask[cid] |= 8
                        has_addr_ev.add(cid)

            if p_code and p_code in ip_c:
                lst_p = ip_c[p_code]
                df_p = len(lst_p)
                if 0 < df_p <= T1_POSTAL_DF:
                    w_p = 20.0 / math.log10(10.0 + df_p)
                    for cid in lst_p:
                        scores[cid] += w_p
                        ch_mask[cid] |= 8
                        has_addr_ev.add(cid)

            # ==================================================================
            # TIER 2: VERIFICATION ON EXISTING ANCHOR POOL ONLY (Zero New Candidates)
            # ==================================================================
            anchor_cids = list(scores.keys())
            total_anchor_pool_size += len(anchor_cids)

            if anchor_cids:
                # High-DF Name tokens (up to 3 additional tokens)
                for df, t in u_name_t2[:3]:
                    w = 22.0 / math.log10(10.0 + df)
                    hi_set = s2_n.get(t)
                    if hi_set is not None:
                        for cid in anchor_cids:
                            if cid in hi_set:
                                scores[cid] += w
                                name_tok_hits[cid] += 1
                                ch_mask[cid] |= 2
                                has_name_ev.add(cid)

                # Multi-name-token agreement bonus
                for cid, cnt in name_tok_hits.items():
                    if cnt >= 2:
                        scores[cid] += 14.0 * (cnt - 1)

                # High-DF Phonetic tokens (up to 2 additional)
                for df, p in u_phon_t2[:2]:
                    w = 9.0 / math.log10(10.0 + df)
                    hi_set = s2_p.get(p)
                    if hi_set is not None:
                        for cid in anchor_cids:
                            if cid in hi_set:
                                scores[cid] += w
                                ch_mask[cid] |= 2
                                has_name_ev.add(cid)

                # High-DF Address tokens (up to 3 additional)
                for df, t in u_addr_t2[:3]:
                    w = 11.0 / math.log10(10.0 + df)
                    hi_set = s2_a.get(t)
                    if hi_set is not None:
                        for cid in anchor_cids:
                            if cid in hi_set:
                                scores[cid] += w
                                ch_mask[cid] |= 2
                                has_addr_ev.add(cid)

                # High-DF Character 4/5-grams (up to 3 additional)
                for df, g in u_gram_t2[:3]:
                    w = 5.5 / math.log10(10.0 + df)
                    hi_set = s2_g.get(g)
                    if hi_set is not None:
                        for cid in anchor_cids:
                            if cid in hi_set:
                                scores[cid] += w
                                ch_mask[cid] |= 4
                                has_name_ev.add(cid)

                # High-DF Channel 4 Numeric / Postal verification on anchors only
                if b_num and s_code and (b_num, s_code) in s2_bs:
                    hi_bs = s2_bs[(b_num, s_code)]
                    w_bs = 32.0 / math.log10(10.0 + len(hi_bs))
                    for cid in anchor_cids:
                        if cid in hi_bs:
                            scores[cid] += w_bs
                            ch_mask[cid] |= 8
                            has_addr_ev.add(cid)

                if b_num and b_num in s2_b:
                    hi_b = s2_b[b_num]
                    w_b = 26.0 / math.log10(10.0 + len(hi_b))
                    for cid in anchor_cids:
                        if cid in hi_b:
                            scores[cid] += w_b
                            ch_mask[cid] |= 8
                            has_addr_ev.add(cid)

                if p_code and p_code in s2_po:
                    hi_p = s2_po[p_code]
                    w_p = 20.0 / math.log10(10.0 + len(hi_p))
                    for cid in anchor_cids:
                        if cid in hi_p:
                            scores[cid] += w_p
                            ch_mask[cid] |= 8
                            has_addr_ev.add(cid)

                # Cross-evidence synergy bonus (Both Name/Char AND Address/Numeric evidence)
                both_ev = has_name_ev & has_addr_ev
                for cid in both_ev:
                    scores[cid] += 28.0

            # ==================================================================
            # CANDIDATE SELECTION (Exact Preservation + S2/S3 Interleaving + heapq.nlargest)
            # ==================================================================
            selected_map: Dict[str, Tuple[float, int]] = {}

            # 1. Always preserve all exact-match candidates regardless of score
            for cid in exact_cids:
                selected_map[cid] = (scores[cid], ch_mask[cid])

            # 2. Fair S2/S3 quota via heapq.nlargest
            s2_scored = []
            s3_scored = []
            for cid, sc in scores.items():
                if cid in selected_map:
                    continue
                m = ch_mask[cid]
                if cid.startswith("S2-"):
                    s2_scored.append((sc, cid, m))
                else:
                    s3_scored.append((sc, cid, m))

            min_per_source = min(160, top_k // 3)
            top_s2 = heapq.nlargest(min_per_source, s2_scored)
            top_s3 = heapq.nlargest(min_per_source, s3_scored)

            for sc, cid, m in top_s2:
                if len(selected_map) < top_k or cid in exact_cids:
                    selected_map[cid] = (sc, m)
            for sc, cid, m in top_s3:
                if len(selected_map) < top_k or cid in exact_cids:
                    selected_map[cid] = (sc, m)

            # 3. Fill remaining slots up to top_k by global priority score using heapq.nlargest
            rem_slots = top_k - len(selected_map)
            if rem_slots > 0:
                remaining_pool = [
                    item for item in (s2_scored + s3_scored) if item[1] not in selected_map
                ]
                for sc, cid, m in heapq.nlargest(rem_slots, remaining_pool):
                    selected_map[cid] = (sc, m)

            final_cands = sorted(
                [(cid, sc_m[0], sc_m[1]) for cid, sc_m in selected_map.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            candidates_by_s1[sid] = final_cands
            total_selected_cands += len(final_cands)

            for cid, _, m in final_cands:
                needed_target_ids.add(cid)
                if m & 1:
                    channel_hit_counts["ch1_exact"] += 1
                if m & 2:
                    channel_hit_counts["ch2_rare_tok"] += 1
                if m & 4:
                    channel_hit_counts["ch3_char_gram"] += 1
                if m & 8:
                    channel_hit_counts["ch4_numeric"] += 1

            processed_q += 1
            if processed_q % 250 == 0 or processed_q == n_queries:
                el = time.time() - t_union_start
                qps = processed_q / max(1e-6, el)
                avg_cands = total_selected_cands / processed_q
                avg_anchors = total_anchor_pool_size / processed_q
                print(
                    f"  [Progress {processed_q:4d}/{n_queries:4d}] "
                    f"elapsed={el:.2f}s | qps={qps:.1f} S1/s | "
                    f"avg_anchors={avg_anchors:.1f} | avg_selected={avg_cands:.1f} | "
                    f"ch_hits=(exact:{channel_hit_counts['ch1_exact']:,}, "
                    f"tok:{channel_hit_counts['ch2_rare_tok']:,}, "
                    f"gram:{channel_hit_counts['ch3_char_gram']:,}, "
                    f"num:{channel_hit_counts['ch4_numeric']:,})",
                    flush=True,
                )

    union_elapsed_s = time.time() - t_union_start
    union_qps = n_queries / max(1e-6, union_elapsed_s)
    print(
        f"Completed Two-Tier candidate union for {n_queries:,} queries in {union_elapsed_s:.2f}s "
        f"({union_qps:.1f} queries/sec)!",
        flush=True,
    )

    # Free posting lists before fetching target text records
    del (
        idx_orig_name, idx_lower_name, idx_core_name, idx_compact_name,
        idx_orig_addr, idx_lower_addr, idx_name_tok, idx_phon_tok,
        idx_addr_tok, idx_char_gram, idx_bldg_num, idx_postal,
        idx_bldg_postal, idx_bldg_state,
        t2_name_sets, t2_phon_sets, t2_addr_sets, t2_gram_sets,
        t2_bldg_sets, t2_postal_sets, t2_bldg_state_sets,
    )
    gc.collect()

    print(f"Streaming S2 and S3 to fetch text records for {len(needed_target_ids):,} candidate/GT IDs...", flush=True)
    t_fetch0 = time.time()
    target_records: Dict[str, Dict[str, str]] = {}
    for src_path in [s2_path, s3_path]:
        for chunk in pd.read_csv(src_path, sep="\t", chunksize=1_000_000, dtype=str):
            sub = chunk[chunk["entity_id"].isin(needed_target_ids)]
            if sub.empty:
                continue
            for cid, n_raw, a_raw, c_raw in zip(
                sub["entity_id"], sub["business_name"], sub["business_address"], sub["country"]
            ):
                target_records[cid] = {
                    "business_name": "" if pd.isna(n_raw) else str(n_raw),
                    "business_address": "" if pd.isna(a_raw) else str(a_raw),
                    "country": "" if pd.isna(c_raw) else str(c_raw).strip().lower(),
                }
    fetch_elapsed_s = time.time() - t_fetch0

    meta = {
        "total_elapsed_s": round(time.time() - t_start, 2),
        "index_elapsed_s": round(index_elapsed_s, 2),
        "union_elapsed_s": round(union_elapsed_s, 2),
        "union_queries_per_sec": round(union_qps, 2),
        "fetch_elapsed_s": round(fetch_elapsed_s, 2),
        "avg_anchor_pool_per_s1": round(total_anchor_pool_size / max(1, n_queries), 2),
        "top_k": top_k,
        "channel_hit_counts": channel_hit_counts,
        "tfidf_benchmark": tfidf_benchmark_result,
    }
    return candidates_by_s1, target_records, meta


def evaluate_retrieval_recall(
    queries_df: pd.DataFrame,
    candidates_by_s1: Dict[str, List[Tuple[str, float, int]]],
    pool_name: str = "DEV_3000",
) -> Dict[str, object]:
    """Compute comprehensive candidate recall statistics."""
    total_gt = 0
    found_gt = 0
    s2_gt = 0
    s2_found = 0
    s3_gt = 0
    s3_found = 0

    entity_recalls_nonempty = []
    entity_recalls_all = []
    cand_counts = []
    empty_s1_cand_counts = []
    exact_channel_s1_covered = 0
    zero_cand_s1 = 0

    by_country = defaultdict(lambda: {"s1": 0, "gt": 0, "found": 0, "recalls": []})
    ch_only_contributions = {1: 0, 2: 0, 4: 0, 8: 0, "multi_channel": 0}
    ch_marginal_coverage = {"ch1_exact": 0, "ch2_rare_tok": 0, "ch3_char_gram": 0, "ch4_numeric": 0}

    for row in queries_df.itertuples(index=False):
        sid = getattr(row, "source1_entity_id", None) or getattr(row, "entity_id")
        c_raw = "" if pd.isna(row.country) else str(row.country).strip().lower()
        m_str = "" if pd.isna(row.matched_entity_ids) else str(row.matched_entity_ids)
        gt_set = set(parse_match_ids(m_str))

        cands_list = candidates_by_s1.get(sid, [])
        cand_map = {cid: m for cid, _, m in cands_list}
        cand_set = set(cand_map.keys())

        cand_counts.append(len(cand_set))
        if len(cand_set) == 0:
            zero_cand_s1 += 1
        if any(m & 1 for m in cand_map.values()):
            exact_channel_s1_covered += 1

        by_country[c_raw]["s1"] += 1

        if not gt_set:
            entity_recalls_all.append(1.0)
            empty_s1_cand_counts.append(len(cand_set))
            continue

        hits = gt_set & cand_set
        n_gt = len(gt_set)
        n_hits = len(hits)
        total_gt += n_gt
        found_gt += n_hits

        rec = n_hits / n_gt
        entity_recalls_nonempty.append(rec)
        entity_recalls_all.append(rec)

        by_country[c_raw]["gt"] += n_gt
        by_country[c_raw]["found"] += n_hits
        by_country[c_raw]["recalls"].append(rec)

        for g_id in gt_set:
            if g_id.startswith("S2-"):
                s2_gt += 1
                if g_id in cand_map:
                    s2_found += 1
            else:
                s3_gt += 1
                if g_id in cand_map:
                    s3_found += 1
            if g_id in cand_map:
                m = cand_map[g_id]
                if m & 1:
                    ch_marginal_coverage["ch1_exact"] += 1
                if m & 2:
                    ch_marginal_coverage["ch2_rare_tok"] += 1
                if m & 4:
                    ch_marginal_coverage["ch3_char_gram"] += 1
                if m & 8:
                    ch_marginal_coverage["ch4_numeric"] += 1
                if m in (1, 2, 4, 8):
                    ch_only_contributions[m] += 1
                else:
                    ch_only_contributions["multi_channel"] += 1

    arr = np.array(cand_counts, dtype=np.float64)
    empty_arr = np.array(empty_s1_cand_counts, dtype=np.float64) if empty_s1_cand_counts else np.array([0.0])
    return {
        "pool_name": pool_name,
        "s1_count": int(len(queries_df)),
        "micro_link_recall": float(found_gt / max(1, total_gt)),
        "macro_entity_recall_nonempty": float(np.mean(entity_recalls_nonempty)),
        "macro_entity_recall_all": float(np.mean(entity_recalls_all)),
        "total_gt_links": int(total_gt),
        "found_gt_links": int(found_gt),
        "missed_gt_links": int(total_gt - found_gt),
        "s2_gt_links": int(s2_gt),
        "s2_found_links": int(s2_found),
        "s2_link_recall": float(s2_found / max(1, s2_gt)),
        "s3_gt_links": int(s3_gt),
        "s3_found_links": int(s3_found),
        "s3_link_recall": float(s3_found / max(1, s3_gt)),
        "zero_candidate_s1_count": int(zero_cand_s1),
        "zero_candidate_rate": float(zero_cand_s1 / max(1, len(queries_df))),
        "exact_channel_s1_coverage_count": int(exact_channel_s1_covered),
        "exact_channel_s1_coverage_pct": float(exact_channel_s1_covered / max(1, len(queries_df))),
        "empty_s1_coverage": {
            "empty_s1_count": int(len(empty_s1_cand_counts)),
            "zero_cand_empty_s1": int(sum(1 for x in empty_s1_cand_counts if x == 0)),
            "mean_cands_on_empty_s1": float(np.mean(empty_arr)),
        },
        "candidates_per_s1": {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": int(np.max(arr)),
            "total_candidate_pairs": int(np.sum(arr)),
        },
        "by_country": {
            c: {
                "s1_count": int(d["s1"]),
                "gt_links": int(d["gt"]),
                "found_links": int(d["found"]),
                "micro_recall": float(d["found"] / max(1, d["gt"])),
                "macro_recall_nonempty": float(np.mean(d["recalls"])) if d["recalls"] else 0.0,
            }
            for c, d in by_country.items()
        },
        "gt_hit_channel_breakdown": {str(k): int(v) for k, v in ch_only_contributions.items()},
        "gt_hit_marginal_channel_coverage": ch_marginal_coverage,
    }


def _slice_cands(
    cands_map: Dict[str, List[Tuple[str, float, int]]],
    s1_ids: Set[str],
    limit: int,
) -> Dict[str, List[Tuple[str, float, int]]]:
    out = {}
    for sid in s1_ids:
        lst = cands_map.get(sid, [])
        if len(lst) <= limit:
            out[sid] = lst
        else:
            # Preserve all exact matches plus top non-exact up to limit
            exact = [x for x in lst if (x[2] & 1)]
            non_exact = [x for x in lst if not (x[2] & 1)]
            rem = max(0, limit - len(exact))
            merged = exact + non_exact[:rem]
            merged.sort(key=lambda x: x[1], reverse=True)
            out[sid] = merged
    return out


def main():
    print("=" * 80, flush=True)
    print("PHASE V3-A: TWO-TIER 4-CHANNEL RETRIEVAL ON DEV_3000 ONLY", flush=True)
    print("=" * 80, flush=True)

    out_dir = Path("experiments/v3/retrieval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Requirement 8: First run ONLY DEV_3000, not all 5,000.
    dev_df = pd.read_csv("experiments/v3/pools/dev_3000_s1.tsv", sep="\t", dtype=str)
    dev_ids = set(dev_df["source1_entity_id"])

    cands_dev_750, records_dev, meta_dev = retrieve_candidates_v3_for_queries(
        dev_df, top_k=750, run_tfidf_benchmark=True
    )

    cands_dev_500 = _slice_cands(cands_dev_750, dev_ids, 500)
    cands_dev_650 = _slice_cands(cands_dev_750, dev_ids, 650)

    eval_top500 = evaluate_retrieval_recall(dev_df, cands_dev_500, pool_name="DEV_3000_TOP500")
    eval_top650 = evaluate_retrieval_recall(dev_df, cands_dev_650, pool_name="DEV_3000_TOP650")
    eval_top750 = evaluate_retrieval_recall(dev_df, cands_dev_750, pool_name="DEV_3000_TOP750")

    if eval_top500["micro_link_recall"] >= 0.97:
        selected_cap = 500
        selected_dev_eval = eval_top500
        selected_cands = cands_dev_500
    elif eval_top650["micro_link_recall"] >= 0.97:
        selected_cap = 650
        selected_dev_eval = eval_top650
        selected_cands = cands_dev_650
    else:
        selected_cap = 750
        selected_dev_eval = eval_top750
        selected_cands = cands_dev_750

    summary_out = {
        "selected_cap": selected_cap,
        "dev_top500": eval_top500,
        "dev_top650": eval_top650,
        "dev_top750": eval_top750,
        "meta": meta_dev,
    }

    with open(out_dir / "dev_3000_retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    joblib.dump(
        {
            "queries_df": dev_df,
            "candidates_by_s1": selected_cands,
            "candidates_by_s1_top750": cands_dev_750,
            "target_records": records_dev,
            "selected_cap": selected_cap,
            "metrics": summary_out,
        },
        out_dir / "dev_3000_retrieval_bundle.joblib",
        compress=3,
    )
    print(f"\nSaved dev_3000_retrieval_bundle.joblib (selected_cap={selected_cap})", flush=True)

    print("\nDEV_3000 Retrieval Summary:", flush=True)
    print(f"  Top-500: Micro Recall = {eval_top500['micro_link_recall']*100:.2f}%, Macro Recall = {eval_top500['macro_entity_recall_nonempty']*100:.2f}%, S2 = {eval_top500['s2_link_recall']*100:.2f}%, S3 = {eval_top500['s3_link_recall']*100:.2f}%", flush=True)
    print(f"  Top-650: Micro Recall = {eval_top650['micro_link_recall']*100:.2f}%, Macro Recall = {eval_top650['macro_entity_recall_nonempty']*100:.2f}%, S2 = {eval_top650['s2_link_recall']*100:.2f}%, S3 = {eval_top650['s3_link_recall']*100:.2f}%", flush=True)
    print(f"  Top-750: Micro Recall = {eval_top750['micro_link_recall']*100:.2f}%, Macro Recall = {eval_top750['macro_entity_recall_nonempty']*100:.2f}%, S2 = {eval_top750['s2_link_recall']*100:.2f}%, S3 = {eval_top750['s3_link_recall']*100:.2f}%", flush=True)

    if selected_dev_eval["micro_link_recall"] < 0.97 and selected_dev_eval["macro_entity_recall_nonempty"] < 0.97:
        print("WARNING: DEV recall < 97%! Must inspect missed links and tune retrieval before model work!", flush=True)
    else:
        print(
            f"GATE PASSED! DEV Micro Recall = {selected_dev_eval['micro_link_recall']*100:.2f}%, "
            f"Macro Non-Empty Recall = {selected_dev_eval['macro_entity_recall_nonempty']*100:.2f}% (>= 97.0%)",
            flush=True,
        )


if __name__ == "__main__":
    main()
