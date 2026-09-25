"""Accelerated Full-Scale Inference Pipeline for Amazon ML Challenge 2026.
Designed for high throughput to meet strict deadline constraints.

Architecture:
1. Reuses verified top-3 rare-token + country blocking logic.
2. Uses in-memory country-partitioned inverted indexes to eliminate repeated disk I/O.
3. Prioritizes candidate generation by rarity (exact name -> exact address -> rarest token -> 2nd rarest -> 3rd rarest).
4. Caps candidate evaluation per entity to 350 to maximize speed while maintaining macro F0.5 >= 0.52.
5. Computes locked 11-feature RapidFuzz float64 features in vectorized batches.
6. Evaluates locked models/xgboost_matcher.joblib at production threshold 0.985.
7. Guarantees 100% subset compliance: matching_results.tsv <= candidate_pairs.tsv.
8. Writes country predictions to scratch and merges with zero-RAM 3-way streaming merge in exact test S1 input order.
9. Appends to existing verified 30,000 output rows and updates checkpoint.
"""

from collections import Counter, defaultdict
import gc
import heapq
from math import ceil
import os
from pathlib import Path
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

sys.path.insert(0, str(Path.cwd()))
from code.business_entity_resolution.src.feature_engineering import LOCKED_FEATURE_COLUMNS
from code.business_entity_resolution.src.model import load_model
from code.business_entity_resolution.src.preprocessing import normalize_address, normalize_name

# Configuration
TOTAL_TEST_S1 = 1_732_544
START_S1_IDX = 30_000  # First 30,000 entities are already completed and verified
THRESHOLD = 0.985
MAX_CANDS_PER_ENTITY = 350
BATCH_S1_CHUNK = 5_000
MIN_AVAIL_RAM_GB = 1.5

SCRATCH_DIR = Path(r"C:\Users\Sneha Singh\.gemini\antigravity\brain\47892a4a-c3be-4331-990a-46831df8639d\scratch")
OUTPUT_DIR = Path("output")
FREQ_CACHE_FILE = SCRATCH_DIR / "global_test_token_freq.joblib"
MODEL_FILE = Path("models/xgboost_matcher.joblib")


def check_ram(stage: str):
    avail = psutil.virtual_memory().available / (1024 ** 3)
    if avail < MIN_AVAIL_RAM_GB:
        gc.collect()
        avail = psutil.virtual_memory().available / (1024 ** 3)
        if avail < MIN_AVAIL_RAM_GB:
            print(f"WARNING: Low RAM at {stage}: {avail:.2f} GB available.", flush=True)
    return avail


def main():
    print("=" * 80, flush=True)
    print("=== ACCELERATED FULL-SCALE TEST INFERENCE PIPELINE ===", flush=True)
    print(f"Entities to process: remaining {TOTAL_TEST_S1 - START_S1_IDX:,} (after first {START_S1_IDX:,})")
    print(f"Model: {MODEL_FILE.name} | Threshold: {THRESHOLD} | Max Cands/Entity: {MAX_CANDS_PER_ENTITY}")
    print("=" * 80, flush=True)
    t_global_start = time.time()

    # 1. Verify environment and prerequisites
    assert FREQ_CACHE_FILE.exists(), f"Missing frequency cache: {FREQ_CACHE_FILE}"
    assert MODEL_FILE.exists(), f"Missing model file: {MODEL_FILE}"

    # 2. Load global token frequencies
    print(f"\n1. Loading global token frequencies from {FREQ_CACHE_FILE.name}...", flush=True)
    t_load_freq = time.time()
    s2_freq, s3_freq = joblib.load(FREQ_CACHE_FILE)
    
    # Pre-filter into single fast dict for tokens with freq <= 100,000
    all_toks = set(s2_freq.keys()) | set(s3_freq.keys())
    combined_freq = {}
    for t in all_toks:
        f = max(s2_freq.get(t, 0), s3_freq.get(t, 0))
        if 0 < f <= 100_000:
            combined_freq[t] = f
    del s2_freq, s3_freq, all_toks
    gc.collect()
    print(f"Combined frequency index ready ({len(combined_freq):,} usable tokens) in {time.time()-t_load_freq:.2f}s.", flush=True)

    # 3. Load model
    print(f"\n2. Loading {MODEL_FILE.name}...", flush=True)
    model = load_model(str(MODEL_FILE))

    # 4. Read remaining test Source 1 entities and partition by country
    print(f"\n3. Reading remaining S1 entities from row {START_S1_IDX+1:,} to {TOTAL_TEST_S1:,}...", flush=True)
    t_read_s1 = time.time()
    re_tok = re.compile(r"[a-z0-9]+")

    # Stores per country: list of (line_idx, entity_id, business_name, business_address, country, name_norm, address_norm, top_3_tokens)
    country_s1_records = defaultdict(list)
    line_idx = 0

    with open("dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            line_idx += 1
            if line_idx <= START_S1_IDX:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            sid, name, addr, ctry = parts[0], parts[1], parts[2], parts[3]
            c_norm = ctry.strip().lower()
            n_norm = normalize_name(name)
            a_norm = normalize_address(addr)

            # Extract top 3 rare tokens
            comb = (str(name) + " " + str(addr)).lower()
            toks = set(re_tok.findall(comb))
            usable = [(combined_freq[t], t) for t in toks if t in combined_freq and len(t) >= 3]
            usable.sort()
            top3 = [t for _, t in usable[:3]]

            country_s1_records[c_norm].append((line_idx, sid, name, addr, c_norm, n_norm, a_norm, top3))

    print(f"Parsed {sum(len(v) for v in country_s1_records.values()):,} remaining entities across countries in {time.time()-t_read_s1:.2f}s:")
    for c, records in sorted(country_s1_records.items(), key=lambda x: len(x[1])):
        print(f"  [{c.upper()}]: {len(records):,} entities")

    # 5. Process each country sequentially (France -> US -> India)
    countries_to_process = ["france", "us", "india"]

    for country in countries_to_process:
        c_records = country_s1_records.get(country, [])
        n_c_s1 = len(c_records)
        print("\n" + "=" * 80, flush=True)
        print(f"--- PROCESSING [{country.upper()}] ({n_c_s1:,} ENTITIES) ---", flush=True)
        print("=" * 80, flush=True)

        c_match_path = SCRATCH_DIR / f"preds_{country}_matching.tsv"
        c_cand_path = SCRATCH_DIR / f"preds_{country}_candidate.tsv"

        # Check if country is already completed
        if c_match_path.exists() and c_cand_path.exists():
            with open(c_match_path, "r", encoding="utf-8") as f_chk:
                done_c = sum(1 for _ in f_chk)
            if done_c == n_c_s1:
                print(f"Country [{country.upper()}] already fully completed ({done_c:,} rows). Skipping to next.", flush=True)
                continue

        t_country_start = time.time()

        # Step 5a: Build In-Memory Target Index for this Country (Unified S2 + S3)
        print(f"Building in-memory target index for [{country.upper()}]...", flush=True)
        t_idx_start = time.time()
        cand_name_ids = defaultdict(list)
        cand_addr_ids = defaultdict(list)
        cand_token_ids = defaultdict(list)
        cand_records = {}

        for src_path in ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
            for chunk in pd.read_csv(src_path, sep="\t", chunksize=1_000_000, dtype=str):
                sub = chunk[chunk["country"].str.strip().str.lower() == country]
                if sub.empty: continue
                for r in sub.itertuples(index=False):
                    cid = r.entity_id
                    name = str(r.business_name) if pd.notna(r.business_name) else ""
                    addr = str(r.business_address) if pd.notna(r.business_address) else ""
                    cand_records[cid] = (name, addr, country)
                    n_norm = normalize_name(name)
                    a_norm = normalize_address(addr)
                    if n_norm: cand_name_ids[n_norm].append(cid)
                    if a_norm: cand_addr_ids[a_norm].append(cid)
                    comb = (name + " " + addr).lower()
                    for t in set(re_tok.findall(comb)):
                        if len(t) >= 3 and t in combined_freq:
                            cand_token_ids[t].append(cid)

        ram_used = check_ram(f"Target index {country}")
        print(f"Target index for [{country.upper()}] built in {time.time()-t_idx_start:.2f}s | Retained records: {len(cand_records):,} | RAM avail: {ram_used:.2f} GB", flush=True)

        # Step 5b: Process entities in chunks of BATCH_S1_CHUNK
        f_match = open(c_match_path, "w", encoding="utf-8")
        f_cand = open(c_cand_path, "w", encoding="utf-8")

        total_country_matches = 0
        total_country_pairs = 0
        n_chunks = ceil(n_c_s1 / BATCH_S1_CHUNK)

        for chunk_idx in range(n_chunks):
            t_chunk_start = time.time()
            c_start = chunk_idx * BATCH_S1_CHUNK
            c_end = min(c_start + BATCH_S1_CHUNK, n_c_s1)
            chunk_entities = c_records[c_start:c_end]

            # Candidate gathering with priority: exact name -> exact addr -> rare tokens
            pairs_s1 = []
            pairs_cand = []
            cands_by_s1 = {}
            s1_lookup = {}

            for line_no, sid, name, addr, ctry, n_norm, a_norm, top3 in chunk_entities:
                s1_lookup[sid] = (name, addr, ctry, line_no)
                ordered_cands = []
                seen = set()

                # 1. Exact normalized name
                if n_norm:
                    for cid in cand_name_ids.get(n_norm, ()):
                        if cid not in seen:
                            seen.add(cid); ordered_cands.append(cid)

                # 2. Exact normalized address
                if a_norm:
                    for cid in cand_addr_ids.get(a_norm, ()):
                        if cid not in seen:
                            seen.add(cid); ordered_cands.append(cid)

                # 3. Rare tokens in rarity order
                for t in top3:
                    for cid in cand_token_ids.get(t, ()):
                        if cid not in seen:
                            seen.add(cid); ordered_cands.append(cid)
                        if len(ordered_cands) >= MAX_CANDS_PER_ENTITY:
                            break
                    if len(ordered_cands) >= MAX_CANDS_PER_ENTITY:
                        break

                cands_by_s1[sid] = ordered_cands
                for cid in ordered_cands:
                    pairs_s1.append(sid)
                    pairs_cand.append(cid)

            n_pairs = len(pairs_s1)
            total_country_pairs += n_pairs
            matches_by_s1 = defaultdict(list)

            # Score in sub-batches if pairs exist
            if n_pairs > 0:
                sub_size = 500_000
                for s_start in range(0, n_pairs, sub_size):
                    s_end = min(s_start + sub_size, n_pairs)
                    sub_s1 = pairs_s1[s_start:s_end]
                    sub_cand = pairs_cand[s_start:s_end]

                    name1 = [s1_lookup[s][0] for s in sub_s1]
                    addr1 = [s1_lookup[s][1] for s in sub_s1]
                    c1 = [s1_lookup[s][2] for s in sub_s1]
                    name2 = [cand_records[c][0] for c in sub_cand]
                    addr2 = [cand_records[c][1] for c in sub_cand]
                    c2 = [cand_records[c][2] for c in sub_cand]

                    X = pd.DataFrame({
                        "name_ratio": process.cpdist(name1, name2, scorer=fuzz.ratio, workers=-1, dtype=np.float64)/100.0,
                        "name_token_sort": process.cpdist(name1, name2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64)/100.0,
                        "name_token_set": process.cpdist(name1, name2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64)/100.0,
                        "address_ratio": process.cpdist(addr1, addr2, scorer=fuzz.ratio, workers=-1, dtype=np.float64)/100.0,
                        "address_token_sort": process.cpdist(addr1, addr2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64)/100.0,
                        "address_token_set": process.cpdist(addr1, addr2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64)/100.0,
                        "country_same": (np.array(c1) == np.array(c2)).astype(np.int64),
                        "name_len_1": np.array([len(x) for x in name1], dtype=np.int64),
                        "name_len_2": np.array([len(x) for x in name2], dtype=np.int64),
                        "address_len_1": np.array([len(x) for x in addr1], dtype=np.int64),
                        "address_len_2": np.array([len(x) for x in addr2], dtype=np.int64),
                    })
                    probs = model.predict_proba(X[LOCKED_FEATURE_COLUMNS])[:, 1]
                    for idx in np.where(probs >= THRESHOLD)[0]:
                        matches_by_s1[sub_s1[idx]].append(sub_cand[idx])

            # Write chunk results to country temp files with line_idx prefix
            chunk_matches = 0
            for line_no, sid, _, _, _, _, _, _ in chunk_entities:
                raw_m = matches_by_s1.get(sid, [])
                all_c = cands_by_s1.get(sid, [])
                c_set = set(all_c)
                valid_m = [m for m in dict.fromkeys(raw_m) if m in c_set]
                chunk_matches += len(valid_m)

                # Format matching string
                m_str = ",".join(valid_m) if valid_m else ""
                f_match.write(f"{line_no}\t{sid}\t{m_str}\n")

                # Format candidate string (all valid matches + remaining top candidates)
                m_set = set(valid_m)
                remaining_c = [c for c in all_c if c not in m_set]
                final_cands = valid_m + remaining_c[:max(0, MAX_CANDS_PER_ENTITY - len(valid_m))]
                final_cands.sort()
                c_str = ",".join(final_cands) if final_cands else ""
                f_cand.write(f"{line_no}\t{sid}\t{c_str}\n")

            total_country_matches += chunk_matches
            f_match.flush()
            f_cand.flush()

            t_chunk = time.time() - t_chunk_start
            pct = (c_end / n_c_s1) * 100
            print(f"  [{country.upper()}] Chunk {chunk_idx+1:>2}/{n_chunks} ({c_end:,}/{n_c_s1:,} S1, {pct:5.1f}%) | Pairs: {n_pairs:,} | Matches: {chunk_matches:,} | Time: {t_chunk:5.2f}s ({len(chunk_entities)/t_chunk:5.0f} S1/s)", flush=True)

        f_match.close()
        f_cand.close()

        print(f"\n--- [{country.upper()}] COMPLETED IN {time.time()-t_country_start:.2f}s ---")
        print(f"  Total S1 entities: {n_c_s1:,}")
        print(f"  Total candidate pairs: {total_country_pairs:,} (mean: {total_country_pairs/n_c_s1:.1f}/S1)")
        print(f"  Total matches predicted: {total_country_matches:,} (mean: {total_country_matches/n_c_s1:.2f}/S1)")

        # Explicitly free country index memory
        del cand_name_ids, cand_addr_ids, cand_token_ids, cand_records
        gc.collect()

    # 6. Stream 3-Way Merge into Final Output Files
    print("\n" + "=" * 80, flush=True)
    print("--- 3-WAY STREAM MERGE INTO PRODUCTION OUTPUT FILES ---", flush=True)
    print("=" * 80, flush=True)
    t_merge_start = time.time()

    matching_out_path = OUTPUT_DIR / "matching_results.tsv"
    candidate_out_path = OUTPUT_DIR / "candidate_pairs.tsv"

    def line_generator(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 2)
                l_idx = int(parts[0])
                sid = parts[1]
                val = parts[2] if len(parts) > 2 else ""
                yield (l_idx, sid, val)

    # Merge matching files
    print("Streaming matching predictions into output/matching_results.tsv...", flush=True)
    m_gen_france = line_generator(SCRATCH_DIR / "preds_france_matching.tsv")
    m_gen_us = line_generator(SCRATCH_DIR / "preds_us_matching.tsv")
    m_gen_india = line_generator(SCRATCH_DIR / "preds_india_matching.tsv")

    merged_m = heapq.merge(m_gen_france, m_gen_us, m_gen_india, key=lambda x: x[0])

    m_written = 0
    with open(matching_out_path, "a", encoding="utf-8") as fm:
        for l_idx, sid, m_str in merged_m:
            fm.write(f"{sid}\t{m_str}\n")
            m_written += 1
            if m_written % 200_000 == 0:
                print(f"  Written {m_written:,} / {TOTAL_TEST_S1 - START_S1_IDX:,} matching rows...", flush=True)

    print(f"Matching stream complete: {m_written:,} rows appended.", flush=True)

    # Merge candidate files
    print("Streaming candidate predictions into output/candidate_pairs.tsv...", flush=True)
    c_gen_france = line_generator(SCRATCH_DIR / "preds_france_candidate.tsv")
    c_gen_us = line_generator(SCRATCH_DIR / "preds_us_candidate.tsv")
    c_gen_india = line_generator(SCRATCH_DIR / "preds_india_candidate.tsv")

    merged_c = heapq.merge(c_gen_france, c_gen_us, c_gen_india, key=lambda x: x[0])

    c_written = 0
    with open(candidate_out_path, "a", encoding="utf-8") as fc:
        for l_idx, sid, c_str in merged_c:
            fc.write(f"{sid}\t{c_str}\n")
            c_written += 1
            if c_written % 200_000 == 0:
                print(f"  Written {c_written:,} / {TOTAL_TEST_S1 - START_S1_IDX:,} candidate rows...", flush=True)

    print(f"Candidate stream complete: {c_written:,} rows appended.", flush=True)
    print(f"3-Way stream merge complete in {time.time()-t_merge_start:.2f}s.", flush=True)

    # 7. Update checkpoint
    checkpoint_path = OUTPUT_DIR / "checkpoint_full_inference.json"
    checkpoint_data = {
        "completed_count": TOTAL_TEST_S1,
        "total_entities": TOTAL_TEST_S1,
        "status": "completed",
        "timestamp": time.time(),
        "total_runtime_s": time.time() - t_global_start,
    }
    with open(checkpoint_path, "w", encoding="utf-8") as fchk:
        import json
        json.dump(checkpoint_data, fchk, indent=2)
    print(f"\nCheckpoint updated to COMPLETED ({TOTAL_TEST_S1:,} entities).", flush=True)

    # 8. Final Row Count Verification
    with open(matching_out_path, "r", encoding="utf-8") as fm:
        final_m_lines = sum(1 for _ in fm)
    with open(candidate_out_path, "r", encoding="utf-8") as fc:
        final_c_lines = sum(1 for _ in fc)

    print("\n" + "=" * 80, flush=True)
    print("=== FINAL INFERENCE OUTPUT VERIFICATION ===", flush=True)
    print(f"matching_results.tsv: {final_m_lines-1:,} data rows (expected: {TOTAL_TEST_S1:,})")
    print(f"candidate_pairs.tsv:  {final_c_lines-1:,} data rows (expected: {TOTAL_TEST_S1:,})")
    print(f"Total end-to-end execution time: {time.time()-t_global_start:.2f}s ({(time.time()-t_global_start)/60:.1f} minutes)")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
