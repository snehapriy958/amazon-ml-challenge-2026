"""V2-C Scaled Model: Candidate Generation, Model-Hard Negative Mining, Training, and Evaluation.

Constraints:
- 50,000 S1 entities strictly from dataset/train/train_source1.tsv (split == 'train').
- Zero overlap with official validation set.
- All ground-truth positives preserved (label = 1).
- Exact production blocker candidate generation (exact name, exact address, top-3 rare tokens, same country).
- Complete exclusion of ground truth from negative candidates.
- Score candidate negatives with frozen V1 baseline model (models/xgboost_matcher.joblib).
- Mine hard negatives (p >= 0.50), semi-hard (0.10 <= p < 0.50), and easy anchor (p < 0.10).
- Strict pair deduplication and zero negative/GT overlap assertion.
- Train XGBoost with locked hyperparameters and 11 features.
- Evaluate on the exact same 1,000-S1 validation bundle across [0.850 to 0.995].
- Save all artifacts under experiments/v2/.
- Stop after validation and provide GO/NO-GO recommendation.
"""

from collections import Counter, defaultdict
import gc
import json
from math import ceil
from pathlib import Path
import random
import re
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
    add_normalized_columns,
    create_train_val_split,
    load_source_tsv,
    parse_match_ids,
)

MIN_AVAIL_RAM_GB = 1.5
BATCH_SIZE = 500_000
MAX_CANDS_PER_QUERY = 45
RANDOM_SEED = 42
THRESHOLDS = [0.85, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.985, 0.990, 0.995]


def check_ram(stage: str):
    avail = psutil.virtual_memory().available / (1024 ** 3)
    if avail < MIN_AVAIL_RAM_GB:
        gc.collect()
        avail = psutil.virtual_memory().available / (1024 ** 3)
        if avail < MIN_AVAIL_RAM_GB:
            raise MemoryError(f"RAM limit exceeded at {stage}: {avail:.2f} GB available.")
    return avail


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


def step_1_sample_50k_clean():
    print("=" * 80, flush=True)
    print("STEP 1: Sampling 50,000 Clean Training S1 Entities (Zero Validation Leakage)", flush=True)
    print("=" * 80, flush=True)

    gt = load_source_tsv("dataset/train/train_ground_truth.tsv")
    gt = create_train_val_split(gt)
    train_gt = gt[gt["split"] == "train"].copy()
    del gt
    gc.collect()

    bundle_path = Path(r"C:\Users\Sneha Singh\.gemini\antigravity\brain\47892a4a-c3be-4331-990a-46831df8639d\scratch\val_1000_candidates_bundle.joblib")
    bundle = joblib.load(bundle_path)
    val_ids = set(bundle["val_ids"])
    del bundle
    gc.collect()

    # Build complete GT dictionary across ALL train S1 entities for strict negative exclusion
    full_train_gt_dict = {}
    for r in train_gt.itertuples(index=False):
        full_train_gt_dict[r.source1_entity_id] = set(parse_match_ids(r.matched_entity_ids))

    clean_train_gt = train_gt[~train_gt["source1_entity_id"].isin(val_ids)].copy()
    del train_gt
    gc.collect()

    print(f"Total clean train-split entities available: {len(clean_train_gt):,}", flush=True)
    sample_gt = clean_train_gt.sample(n=50_000, random_state=RANDOM_SEED).copy()
    del clean_train_gt
    gc.collect()

    sampled_s1_set = set(sample_gt["source1_entity_id"])
    overlap = sampled_s1_set & val_ids
    assert len(overlap) == 0, f"FATAL: {len(overlap)} validation entities found in sample!"
    assert len(sample_gt) == 50_000, f"Expected 50,000 entities, got {len(sample_gt)}"

    gt_dict = {}
    gt_pos_pairs = []
    all_gt_target_ids = set()
    for r in sample_gt.itertuples(index=False):
        matched = full_train_gt_dict[r.source1_entity_id]
        gt_dict[r.source1_entity_id] = matched
        for m in matched:
            gt_pos_pairs.append((r.source1_entity_id, m))
            all_gt_target_ids.add(m)
    del sample_gt
    gc.collect()

    print(f"Sampled {len(sampled_s1_set):,} clean S1 entities.", flush=True)
    print(f"Total GT positive links for 50k sample: {len(gt_pos_pairs):,} pairs across {len(all_gt_target_ids):,} unique targets.", flush=True)
    print(f"Validation overlap check: {len(overlap)} entities overlapping (PASSED).", flush=True)

    print("Loading Source 1 records for sampled entities...", flush=True)
    s1_df = load_source_tsv("dataset/train/train_source1.tsv")
    sample_s1_df = s1_df[s1_df["entity_id"].isin(sampled_s1_set)].copy()
    sample_s1_df = add_normalized_columns(sample_s1_df)
    del s1_df
    gc.collect()

    re_tok = re.compile(r"[a-z0-9]+")
    s1_lookup = {}
    for r in sample_s1_df.itertuples(index=False):
        n_norm = str(r.name_norm) if pd.notna(r.name_norm) else ""
        a_norm = str(r.address_norm) if pd.notna(r.address_norm) else ""
        toks = set(t for t in re_tok.findall(f"{n_norm} {a_norm}") if len(t) >= 3)
        s1_lookup[r.entity_id] = {
            "name": str(r.business_name) if pd.notna(r.business_name) else "",
            "addr": str(r.business_address) if pd.notna(r.business_address) else "",
            "country": str(r.country).strip().lower() if pd.notna(r.country) else "",
            "name_norm": n_norm,
            "addr_norm": a_norm,
            "tokens": toks,
        }

    return s1_lookup, gt_dict, full_train_gt_dict, gt_pos_pairs, all_gt_target_ids, val_ids


def step_2_mine_hard_negatives(s1_lookup, gt_dict, all_gt_target_ids):
    print("\n" + "=" * 80, flush=True)
    print("STEP 2: Single-Pass Country-Partitioned Indexing & Hard-Negative Mining", flush=True)
    print("=" * 80, flush=True)

    model_path = Path("models/xgboost_matcher.joblib")
    print(f"Loading frozen baseline model from {model_path}...", flush=True)
    baseline_model = joblib.load(model_path)

    # Group S1 queries by country
    country_queries = defaultdict(dict)
    query_names = defaultdict(set)
    query_addrs = defaultdict(set)
    query_tokens = defaultdict(set)

    for sid, rec in s1_lookup.items():
        c = rec["country"]
        country_queries[c][sid] = rec
        if rec["name_norm"]:
            query_names[c].add(rec["name_norm"])
        if rec["addr_norm"]:
            query_addrs[c].add(rec["addr_norm"])
        query_tokens[c].update(rec["tokens"])

    for c in country_queries:
        print(f"  [{c.upper()}]: {len(country_queries[c]):,} queries | {len(query_names[c]):,} names | {len(query_addrs[c]):,} addrs | {len(query_tokens[c]):,} tokens", flush=True)

    # Build targeted indexes across all countries in a single pass over S2 and S3
    cand_name_ids = {c: defaultdict(list) for c in country_queries}
    cand_addr_ids = {c: defaultdict(list) for c in country_queries}
    cand_token_ids = {c: defaultdict(list) for c in country_queries}
    disqualified_tokens = {c: set() for c in country_queries}

    re_tok = re.compile(r"[a-z0-9]+")
    t_idx0 = time.time()

    for src_idx, src_path in enumerate(["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"], 1):
        print(f"  Indexing [{src_idx}/2] {src_path}...", flush=True)
        chunk_no = 0
        for chunk in pd.read_csv(src_path, sep="\t", chunksize=1_000_000, dtype=str):
            chunk_no += 1
            t_ch0 = time.time()
            chunk["c_norm"] = chunk["country"].fillna("").str.strip().str.lower()
            for c in country_queries:
                sub = chunk[chunk["c_norm"] == c]
                if sub.empty:
                    continue
                q_names_c = query_names[c]
                q_addrs_c = query_addrs[c]
                q_toks_c = query_tokens[c]
                disq_c = disqualified_tokens[c]
                c_name_map = cand_name_ids[c]
                c_addr_map = cand_addr_ids[c]
                c_tok_map = cand_token_ids[c]

                for cid, n_raw, a_raw in zip(sub["entity_id"], sub["business_name"], sub["business_address"]):
                    n_str = "" if pd.isna(n_raw) else str(n_raw).lower()
                    a_str = "" if pd.isna(a_raw) else str(a_raw).lower()
                    n_toks = re_tok.findall(n_str)
                    a_toks = re_tok.findall(a_str)
                    n_norm = " ".join(n_toks)
                    a_norm = " ".join(a_toks)

                    if n_norm in q_names_c:
                        c_name_map[n_norm].append(cid)
                    if a_norm in q_addrs_c:
                        c_addr_map[a_norm].append(cid)

                    rec_toks = {t for t in n_toks + a_toks if len(t) >= 3}
                    for t in rec_toks & q_toks_c:
                        if t not in disq_c:
                            lst = c_tok_map[t]
                            lst.append(cid)
                            if len(lst) > 100_000:
                                disq_c.add(t)
                                del c_tok_map[t]
            print(f"    Chunk {chunk_no} processed in {time.time()-t_ch0:.1f}s | Avail RAM: {check_ram('Index chunk'):.2f} GB", flush=True)

    print(f"Single-pass targeted index built in {time.time()-t_idx0:.1f}s.", flush=True)

    # Gather candidate negative pairs for all 50,000 queries
    print("Gathering blocker candidates and excluding all GT matches...", flush=True)
    all_cand_neg_pairs = []
    needed_cids_for_all = set(all_gt_target_ids)

    for c, queries in country_queries.items():
        c_name_map = cand_name_ids[c]
        c_addr_map = cand_addr_ids[c]
        c_tok_map = cand_token_ids[c]
        c_pairs = 0

        for sid, rec in queries.items():
            true_matches = gt_dict.get(sid, set())
            cands = []
            seen = set()

            # 1. Exact normalized name
            n_norm = rec["name_norm"]
            if n_norm and n_norm in c_name_map:
                for cid in c_name_map[n_norm]:
                    if cid not in seen and cid not in true_matches:
                        seen.add(cid)
                        cands.append(cid)

            # 2. Exact normalized address
            a_norm = rec["addr_norm"]
            if a_norm and a_norm in c_addr_map:
                for cid in c_addr_map[a_norm]:
                    if cid not in seen and cid not in true_matches:
                        seen.add(cid)
                        cands.append(cid)

            # 3. Top-3 rarest tokens
            usable = []
            for t in rec["tokens"]:
                if t in c_tok_map:
                    freq = len(c_tok_map[t])
                    if 0 < freq <= 100_000:
                        usable.append((freq, t))
            usable.sort()

            for _, t in usable[:3]:
                for cid in c_tok_map[t]:
                    if cid not in seen and cid not in true_matches:
                        seen.add(cid)
                        cands.append(cid)
                    if len(cands) >= MAX_CANDS_PER_QUERY:
                        break
                if len(cands) >= MAX_CANDS_PER_QUERY:
                    break

            for cid in cands:
                all_cand_neg_pairs.append((sid, cid))
                needed_cids_for_all.add(cid)
            c_pairs += len(cands)

        print(f"  [{c.upper()}]: Gathered {c_pairs:,} candidate negative pairs.", flush=True)

    # Free all index structures before fetching text records
    del cand_name_ids, cand_addr_ids, cand_token_ids, disqualified_tokens
    del query_names, query_addrs, query_tokens, country_queries
    gc.collect()
    print(f"Total candidate negative pairs: {len(all_cand_neg_pairs):,} | Needed target records: {len(needed_cids_for_all):,} | Avail RAM: {check_ram('Post-gather'):.2f} GB", flush=True)

    # Fetch text records for needed candidate IDs
    print("Streaming S2 and S3 to fetch text records for needed candidate IDs...", flush=True)
    t_fetch0 = time.time()
    all_cand_records = {}
    for src_path in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        for chunk in pd.read_csv(src_path, sep="\t", chunksize=1_000_000, dtype=str):
            sub = chunk[chunk["entity_id"].isin(needed_cids_for_all)]
            if sub.empty:
                continue
            for cid, n_raw, a_raw, c_raw in zip(sub["entity_id"], sub["business_name"], sub["business_address"], sub["country"]):
                all_cand_records[cid] = (
                    "" if pd.isna(n_raw) else str(n_raw),
                    "" if pd.isna(a_raw) else str(a_raw),
                    "" if pd.isna(c_raw) else str(c_raw).strip().lower(),
                )
    print(f"Fetched {len(all_cand_records):,} candidate records in {time.time()-t_fetch0:.1f}s | Avail RAM: {check_ram('Records fetched'):.2f} GB", flush=True)

    # Score candidate negatives in batches and mine immediately per batch to keep RAM minimal
    print(f"\nScoring {len(all_cand_neg_pairs):,} candidate negatives in batches of {BATCH_SIZE:,}...", flush=True)
    t_score0 = time.time()
    n_batches = ceil(len(all_cand_neg_pairs) / BATCH_SIZE)

    mined_chunks = []
    total_hard = 0
    total_semi = 0
    total_easy = 0

    for b_idx in range(n_batches):
        b_start = b_idx * BATCH_SIZE
        b_end = min(b_start + BATCH_SIZE, len(all_cand_neg_pairs))
        batch_pairs = all_cand_neg_pairs[b_start:b_end]

        b_s1 = [p[0] for p in batch_pairs]
        b_cid = [p[1] for p in batch_pairs]

        name1 = [s1_lookup[sid]["name"] for sid in b_s1]
        addr1 = [s1_lookup[sid]["addr"] for sid in b_s1]
        country1 = [s1_lookup[sid]["country"] for sid in b_s1]

        name2 = [all_cand_records[cid][0] if cid in all_cand_records else "" for cid in b_cid]
        addr2 = [all_cand_records[cid][1] if cid in all_cand_records else "" for cid in b_cid]
        country2 = [all_cand_records[cid][2] if cid in all_cand_records else "" for cid in b_cid]

        X_batch = pd.DataFrame({
            "source1_entity_id": b_s1,
            "candidate_entity_id": b_cid,
            "name_ratio": process.cpdist(name1, name2, scorer=fuzz.ratio, workers=-1, dtype=np.float64) / 100.0,
            "name_token_sort": process.cpdist(name1, name2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64) / 100.0,
            "name_token_set": process.cpdist(name1, name2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64) / 100.0,
            "address_ratio": process.cpdist(addr1, addr2, scorer=fuzz.ratio, workers=-1, dtype=np.float64) / 100.0,
            "address_token_sort": process.cpdist(addr1, addr2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64) / 100.0,
            "address_token_set": process.cpdist(addr1, addr2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64) / 100.0,
            "country_same": (np.array(country1) == np.array(country2)).astype(np.int64),
            "name_len_1": np.array([len(x) for x in name1], dtype=np.int64),
            "name_len_2": np.array([len(x) for x in name2], dtype=np.int64),
            "address_len_1": np.array([len(x) for x in addr1], dtype=np.int64),
            "address_len_2": np.array([len(x) for x in addr2], dtype=np.int64),
        })

        probs = baseline_model.predict_proba(X_batch[LOCKED_FEATURE_COLUMNS])[:, 1]
        X_batch["base_prob"] = probs
        X_batch["label"] = 0

        # Stratify:
        # 1. All hard negatives (p >= 0.50)
        # 2. All semi-hard negatives (0.10 <= p < 0.50)
        # 3. Easy negative anchor (p < 0.10): sample 35% to achieve ~1.4M - 1.5M total pairs
        hard_mask = probs >= 0.50
        semi_mask = (probs >= 0.10) & (probs < 0.50)
        easy_mask = probs < 0.10

        n_h = int(np.sum(hard_mask))
        n_s = int(np.sum(semi_mask))
        total_hard += n_h
        total_semi += n_s

        hard_df = X_batch[hard_mask]
        semi_df = X_batch[semi_mask]
        easy_df = X_batch[easy_mask].sample(frac=0.35, random_state=RANDOM_SEED + b_idx)
        total_easy += len(easy_df)

        kept_df = pd.concat([hard_df, semi_df, easy_df], ignore_index=True)
        mined_chunks.append(kept_df)

        del name1, addr1, country1, name2, addr2, country2, X_batch, probs, hard_df, semi_df, easy_df, kept_df
        gc.collect()

        print(f"  Batch {b_idx+1}/{n_batches}: Hard={n_h:,}, Semi={n_s:,} | Elapsed: {time.time()-t_score0:.1f}s | Avail RAM: {check_ram('Score batch'):.2f} GB", flush=True)

    mined_neg_df = pd.concat(mined_chunks, ignore_index=True)
    del mined_chunks
    gc.collect()

    stats = {
        "total_candidate_negs": len(all_cand_neg_pairs),
        "hard_count": total_hard,
        "semi_hard_count": total_semi,
        "easy_count": total_easy,
    }
    del all_cand_neg_pairs
    gc.collect()

    return mined_neg_df, all_cand_records, stats


def step_3_assemble_dataset(s1_lookup, full_train_gt_dict, gt_pos_pairs, all_cand_records, mined_neg_df, stats, val_ids):
    print("\n" + "=" * 80, flush=True)
    print("STEP 3: Computing Positive Features, Deduplicating & Assembling V2-C Dataset", flush=True)
    print("=" * 80, flush=True)

    print(f"Computing RapidFuzz features for {len(gt_pos_pairs):,} ground-truth positive pairs...", flush=True)
    t_pos0 = time.time()
    pos_s1 = [p[0] for p in gt_pos_pairs]
    pos_cid = [p[1] for p in gt_pos_pairs]

    name1 = [s1_lookup[sid]["name"] for sid in pos_s1]
    addr1 = [s1_lookup[sid]["addr"] for sid in pos_s1]
    country1 = [s1_lookup[sid]["country"] for sid in pos_s1]

    name2 = [all_cand_records[cid][0] if cid in all_cand_records else "" for cid in pos_cid]
    addr2 = [all_cand_records[cid][1] if cid in all_cand_records else "" for cid in pos_cid]
    country2 = [all_cand_records[cid][2] if cid in all_cand_records else "" for cid in pos_cid]

    pos_df = pd.DataFrame({
        "source1_entity_id": pos_s1,
        "candidate_entity_id": pos_cid,
        "name_ratio": process.cpdist(name1, name2, scorer=fuzz.ratio, workers=-1, dtype=np.float64) / 100.0,
        "name_token_sort": process.cpdist(name1, name2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64) / 100.0,
        "name_token_set": process.cpdist(name1, name2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64) / 100.0,
        "address_ratio": process.cpdist(addr1, addr2, scorer=fuzz.ratio, workers=-1, dtype=np.float64) / 100.0,
        "address_token_sort": process.cpdist(addr1, addr2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64) / 100.0,
        "address_token_set": process.cpdist(addr1, addr2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64) / 100.0,
        "country_same": (np.array(country1) == np.array(country2)).astype(np.int64),
        "name_len_1": np.array([len(x) for x in name1], dtype=np.int64),
        "name_len_2": np.array([len(x) for x in name2], dtype=np.int64),
        "address_len_1": np.array([len(x) for x in addr1], dtype=np.int64),
        "address_len_2": np.array([len(x) for x in addr2], dtype=np.int64),
        "base_prob": np.ones(len(gt_pos_pairs), dtype=np.float64),
        "label": np.ones(len(gt_pos_pairs), dtype=np.int64),
    })
    print(f"Positive features computed in {time.time()-t_pos0:.1f}s.", flush=True)

    del name1, addr1, country1, name2, addr2, country2, all_cand_records
    gc.collect()

    # Combine with existing clean V2-A dataset so we retain full clean positive coverage + new 50k mined hard negatives
    print("Loading clean V2-A baseline pairs to merge with newly mined 50k hard-negative pool...", flush=True)
    v2_a_df = pd.read_parquet("experiments/v2/data/v2_a_train_features.parquet")
    base_model = joblib.load("models/xgboost_matcher.joblib")
    v2_a_probs = np.ones(len(v2_a_df), dtype=np.float64)
    v2_a_neg_mask = (v2_a_df["label"] == 0).values
    v2_a_probs[v2_a_neg_mask] = base_model.predict_proba(v2_a_df.loc[v2_a_neg_mask, LOCKED_FEATURE_COLUMNS])[:, 1]
    v2_a_df["base_prob"] = v2_a_probs

    # Combine and strictly deduplicate on (source1_entity_id, candidate_entity_id)
    cols = ["source1_entity_id", "candidate_entity_id"] + LOCKED_FEATURE_COLUMNS + ["base_prob", "label"]
    combined_df = pd.concat([pos_df[cols], mined_neg_df[cols], v2_a_df[cols]], ignore_index=True)
    del pos_df, mined_neg_df, v2_a_df
    gc.collect()

    before_dedup = len(combined_df)
    combined_df.sort_values(by=["label"], ascending=False, inplace=True)  # Prioritize label=1 if any collision
    combined_df.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"], keep="first", inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    after_dedup = len(combined_df)
    print(f"Deduplicated pairs: {before_dedup:,} -> {after_dedup:,} unique pairs.", flush=True)

    # Strict verification: zero negative/GT overlap across full train GT
    neg_rows = combined_df[combined_df["label"] == 0]
    overlap_gt_count = 0
    for sid, cid in zip(neg_rows["source1_entity_id"], neg_rows["candidate_entity_id"]):
        if cid in full_train_gt_dict.get(sid, ()):
            overlap_gt_count += 1
    assert overlap_gt_count == 0, f"FATAL: Found {overlap_gt_count} negative pairs overlapping with GT!"
    print(f"Negative/GT overlap verification: {overlap_gt_count} overlapping pairs (PASSED).", flush=True)

    # Strict verification: zero validation S1 overlap
    s1_in_combined = set(combined_df["source1_entity_id"].unique())
    val_overlap = s1_in_combined & val_ids
    assert len(val_overlap) == 0, f"FATAL: Found {len(val_overlap)} validation entities in V2-C!"
    print(f"Validation S1 overlap check: {len(val_overlap)} overlapping entities (PASSED).", flush=True)

    # Compute final counts
    n_pos = int((combined_df["label"] == 1).sum())
    n_neg = int((combined_df["label"] == 0).sum())
    neg_probs = combined_df.loc[combined_df["label"] == 0, "base_prob"].values
    final_hard = int(np.sum(neg_probs >= 0.50))
    final_semi = int(np.sum((neg_probs >= 0.10) & (neg_probs < 0.50)))
    final_easy = int(np.sum(neg_probs < 0.10))

    # Assign sample_weight using the V2-B hard-negative boost
    weights = np.ones(len(combined_df), dtype=np.float32)
    is_neg = (combined_df["label"] == 0).values
    b_prob = combined_df["base_prob"].values
    weights[is_neg & (b_prob >= 0.50)] = 2.0
    weights[is_neg & (b_prob >= 0.10) & (b_prob < 0.50)] = 1.25
    combined_df["sample_weight"] = weights

    out_data_path = Path("experiments/v2/data/v2_c_train_features.parquet")
    out_data_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(out_data_path, index=False)
    disk_mb = out_data_path.stat().st_size / (1024 ** 2)

    print("\n" + "=" * 80, flush=True)
    print("V2-C DATASET SUMMARY REPORT", flush=True)
    print("=" * 80, flush=True)
    print(f"Newly Mined S1 Count:         50,000 entities (Total unique clean train S1s: {len(s1_in_combined):,})", flush=True)
    print(f"Positive Count:               {n_pos:,} (50k sample GT positives: {len(gt_pos_pairs):,})", flush=True)
    print(f"Total Candidate Negs Scored:  {stats['total_candidate_negs']:,} (from 50k blocker run)", flush=True)
    print(f"Final Hard Negatives (>=0.5): {final_hard:,} (50k mined: {stats['hard_count']:,})", flush=True)
    print(f"Final Semi-Hard (0.1 to 0.5): {final_semi:,} (50k mined: {stats['semi_hard_count']:,})", flush=True)
    print(f"Final Easy Negatives (<0.1):  {final_easy:,} (50k mined: {stats['easy_count']:,})", flush=True)
    print(f"Final Training Pair Count:    {len(combined_df):,}", flush=True)
    print(f"Class Balance:                {n_pos:,} pos ({n_pos/len(combined_df)*100:.2f}%) / {n_neg:,} neg ({n_neg/len(combined_df)*100:.2f}%)", flush=True)
    print(f"Disk Usage:                   {disk_mb:.2f} MB ({out_data_path})", flush=True)
    print("=" * 80, flush=True)

    full_stats = {
        "sampled_train_s1_count": 50_000,
        "total_unique_clean_s1_count": len(s1_in_combined),
        "sampled_50k_positive_count": len(gt_pos_pairs),
        "final_positive_count": n_pos,
        "total_candidate_negatives_50k": stats["total_candidate_negs"],
        "mined_50k_hard_count": stats["hard_count"],
        "mined_50k_semi_hard_count": stats["semi_hard_count"],
        "mined_50k_easy_count": stats["easy_count"],
        "final_hard_count": final_hard,
        "final_semi_hard_count": final_semi,
        "final_easy_count": final_easy,
        "final_training_pair_count": len(combined_df),
        "positive_pct": float(n_pos / len(combined_df) * 100),
        "negative_pct": float(n_neg / len(combined_df) * 100),
        "disk_usage_mb": float(disk_mb),
    }
    return combined_df, full_stats


def evaluate_model_on_val_bundle(model, bundle, gt_dict, val_ids, total_gt_matches):
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

        X_batch = pd.DataFrame({
            "name_ratio": process.cpdist(name1, name2, scorer=fuzz.ratio, workers=-1, dtype=np.float64) / 100.0,
            "name_token_sort": process.cpdist(name1, name2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64) / 100.0,
            "name_token_set": process.cpdist(name1, name2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64) / 100.0,
            "address_ratio": process.cpdist(addr1, addr2, scorer=fuzz.ratio, workers=-1, dtype=np.float64) / 100.0,
            "address_token_sort": process.cpdist(addr1, addr2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64) / 100.0,
            "address_token_set": process.cpdist(addr1, addr2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64) / 100.0,
            "country_same": (np.array(country1) == np.array(country2)).astype(np.int64),
            "name_len_1": np.array([len(x) for x in name1], dtype=np.int64),
            "name_len_2": np.array([len(x) for x in name2], dtype=np.int64),
            "address_len_1": np.array([len(x) for x in addr1], dtype=np.int64),
            "address_len_2": np.array([len(x) for x in addr2], dtype=np.int64),
        })

        probs = model.predict_proba(X_batch[LOCKED_FEATURE_COLUMNS])[:, 1]
        mask = np.where(probs >= 0.80)[0]
        for idx in mask:
            high_s1.append(b_s1[idx])
            high_cand.append(b_cand[idx])
            high_prob.append(float(probs[idx]))

        del name1, addr1, country1, name2, addr2, country2, X_batch, probs, mask
        gc.collect()

    arr_s1 = np.array(high_s1, dtype=object)
    arr_cand = np.array(high_cand, dtype=object)
    arr_prob = np.array(high_prob, dtype=np.float32)

    with open("experiments/v2/results/v2_b_val_metrics.json") as f:
        v2_b_data = json.load(f)
    v2_b_by_thresh = {r["threshold"]: r for r in v2_b_data["results"]}

    with open("experiments/v2/results/v2_a_val_metrics.json") as f:
        v2_a_data = json.load(f)
    v2_a_by_thresh = {r["threshold"]: r for r in v2_a_data["results"]}

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

        f05_c = float(metrics["macro_f_beta"])
        f05_b = float(v2_b_by_thresh.get(thresh, {}).get("macro_f05", 0.0))
        f05_a = float(v2_a_by_thresh.get(thresh, {}).get("macro_f05", 0.0))

        res = {
            "threshold": float(thresh),
            "macro_f05": f05_c,
            "v2_b_macro_f05": f05_b,
            "v2_a_macro_f05": f05_a,
            "delta_vs_v2_b": float(f05_c - f05_b),
            "delta_vs_v2_a": float(f05_c - f05_a),
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


def step_4_and_5_train_and_eval(combined_df, full_stats):
    print("\n" + "=" * 80, flush=True)
    print("STEP 4 & 5: Training V2-C Models & Evaluating on Official 1,000-S1 Bundle", flush=True)
    print("=" * 80, flush=True)

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "enable_categorical": True,
        "n_jobs": -1,
    }

    bundle_path = Path(r"C:\Users\Sneha Singh\.gemini\antigravity\brain\47892a4a-c3be-4331-990a-46831df8639d\scratch\val_1000_candidates_bundle.joblib")
    bundle = joblib.load(bundle_path)
    val_ids = bundle["val_ids"]

    gt = load_source_tsv("dataset/train/train_ground_truth.tsv")
    gt = create_train_val_split(gt)
    val_gt = gt[gt["split"] == "validation"].copy()
    val_gt = val_gt[val_gt["source1_entity_id"].isin(set(val_ids))].copy()
    gt_dict = {r.source1_entity_id: set(parse_match_ids(r.matched_entity_ids)) for r in val_gt.itertuples(index=False)}
    del gt, val_gt
    gc.collect()
    total_gt_matches = sum(len(m) for m in gt_dict.values())

    X = combined_df[LOCKED_FEATURE_COLUMNS]
    y = combined_df["label"].values
    w = combined_df["sample_weight"].values

    # Variant A: Unweighted deduplicated V2-C
    print("\nTraining Variant 1 (Deduplicated Unweighted V2-C)...", flush=True)
    m_unw = XGBClassifier(**params)
    m_unw.fit(X, y)
    res_unw, best_unw = evaluate_model_on_val_bundle(m_unw, bundle, gt_dict, val_ids, total_gt_matches)
    print(f"Variant 1 Peak: Macro F0.5 = {best_unw['macro_f05']:.4f} at Thresh {best_unw['threshold']} (Prec: {best_unw['macro_precision']*100:.2f}%, Rec: {best_unw['macro_recall']*100:.2f}%)", flush=True)

    # Variant B: Sample-Weighted deduplicated V2-C (Hard=2.0, Semi=1.25)
    print("\nTraining Variant 2 (Deduplicated Sample-Weighted V2-C)...", flush=True)
    m_w = XGBClassifier(**params)
    m_w.fit(X, y, sample_weight=w)
    res_w, best_w = evaluate_model_on_val_bundle(m_w, bundle, gt_dict, val_ids, total_gt_matches)
    print(f"Variant 2 Peak: Macro F0.5 = {best_w['macro_f05']:.4f} at Thresh {best_w['threshold']} (Prec: {best_w['macro_precision']*100:.2f}%, Rec: {best_w['macro_recall']*100:.2f}%)", flush=True)

    # Select best V2-C variant
    if best_w["macro_f05"] >= best_unw["macro_f05"]:
        best_model, best_results, best_info, variant_name = m_w, res_w, best_w, "weighted_dedup"
    else:
        best_model, best_results, best_info, variant_name = m_unw, res_unw, best_unw, "unweighted_dedup"

    print(f"\nSelected V2-C Model ({variant_name}): Peak Macro F0.5 = {best_info['macro_f05']:.4f} at Threshold {best_info['threshold']}", flush=True)

    out_model_path = Path("experiments/v2/models/xgboost_matcher_v2_c.joblib")
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_model_path)
    print(f"Saved V2-C model to {out_model_path} ({out_model_path.stat().st_size / (1024**2):.2f} MB)", flush=True)

    print("\n" + "=" * 130, flush=True)
    print(f"{'Thresh':>7} | {'V2-C F0.5':>10} | {'V2-B F0.5':>10} | {'Delta vs B':>10} | {'Precision':>10} | {'Recall':>9} | {'Pred Pairs':>10} | {'Singletons':>10} | {'TP':>6} | {'FP':>6} | {'FN':>6}", flush=True)
    print("-" * 130, flush=True)
    for r in best_results:
        print(
            f"{r['threshold']:>7.3f} | {r['macro_f05']:>10.4f} | {r['v2_b_macro_f05']:>10.4f} | {r['delta_vs_v2_b']:>+10.4f} | "
            f"{r['macro_precision']*100:>9.2f}% | {r['macro_recall']*100:>8.2f}% | "
            f"{r['total_predicted_matches']:>10,d} | {r['predicted_singletons']:>10,d} | "
            f"{r['tp']:>6,d} | {r['fp']:>6,d} | {r['fn']:>6,d}",
            flush=True,
        )
    print("=" * 130, flush=True)

    out_metrics_path = Path("experiments/v2/results/v2_c_val_metrics.json")
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "variant": variant_name,
            "dataset_stats": full_stats,
            "results": best_results,
            "best": best_info,
            "unweighted_best": best_unw,
            "weighted_best": best_w,
        }, f, indent=2)
    print(f"Saved V2-C validation metrics to {out_metrics_path}", flush=True)


def main():
    t_start = time.time()
    s1_lookup, gt_dict, full_train_gt_dict, gt_pos_pairs, all_gt_target_ids, val_ids = step_1_sample_50k_clean()
    mined_neg_df, all_cand_records, stats = step_2_mine_hard_negatives(s1_lookup, gt_dict, all_gt_target_ids)
    combined_df, full_stats = step_3_assemble_dataset(
        s1_lookup, full_train_gt_dict, gt_pos_pairs, all_cand_records, mined_neg_df, stats, val_ids
    )
    del s1_lookup, gt_dict, full_train_gt_dict, gt_pos_pairs
    gc.collect()

    step_4_and_5_train_and_eval(combined_df, full_stats)
    print(f"\nTotal V2-C pipeline runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.2f} min).", flush=True)


if __name__ == "__main__":
    main()
