"""Production Full Test Inference Pipeline for Amazon ML Challenge 2026.
Executes end-to-end streaming candidate generation and matching inference
over all 1,732,544 test Source 1 entities with:
- Model: models/xgboost_matcher.joblib
- Operating Threshold: 0.985
- Locked 11-feature schema (RapidFuzz float64)
- Existing top-3 rare-token + same-country blocker
- Country partitioning (France -> US -> India)
- Bounded-memory streaming with persistent progress checkpointing
- Strict integrity enforcement: subset rule, singleton formatting, 0 duplicate IDs
- Appends to output/matching_results.tsv and output/candidate_pairs.tsv
- Runs official utils/validate_submission.py upon completion.
"""

from collections import Counter, defaultdict
import gc
import json
from math import ceil
from pathlib import Path
import re
import shutil
import subprocess
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
from code.business_entity_resolution.src.blocking import tokenize
from code.business_entity_resolution.src.feature_engineering import LOCKED_FEATURE_COLUMNS
from code.business_entity_resolution.src.model import load_model
from code.business_entity_resolution.src.preprocessing import (
    add_normalized_columns,
    normalize_address,
    normalize_name,
)

MIN_RAM_THRESHOLD_GB = 2.0
MIN_DISK_FREE_GB = 5.0
BATCH_S1_SIZE = 5_000
CHUNK_S1_SIZE = 250
BATCH_PAIR_SIZE = 500_000
THRESHOLD = 0.985

TOTAL_TEST_S1 = 1_732_544


def check_system_resources(stage: str):
    avail_ram = psutil.virtual_memory().available / (1024 ** 3)
    if avail_ram < MIN_RAM_THRESHOLD_GB:
        raise MemoryError(f"RAM safety limit hit at {stage}: {avail_ram:.2f} GB available.")
    
    _, _, free_disk = shutil.disk_usage(".")
    free_disk_gb = free_disk / (1024 ** 3)
    if free_disk_gb < MIN_DISK_FREE_GB:
        raise IOError(f"Disk space safety limit hit at {stage}: {free_disk_gb:.2f} GB free.")
    
    return avail_ram, free_disk_gb


def get_mem_stats():
    proc = psutil.Process()
    rss = proc.memory_info().rss / (1024 ** 3)
    peak = proc.memory_info().peak_wset / (1024 ** 3)
    avail = psutil.virtual_memory().available / (1024 ** 3)
    return rss, peak, avail


def count_tsv_data_rows(filepath: Path) -> int:
    """Count data rows in a TSV file (excluding header)."""
    if not filepath.exists():
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        count = sum(1 for _ in f) - 1
    return max(0, count)


def main():
    print("=" * 80, flush=True)
    print("=== FULL TEST INFERENCE PIPELINE (1,732,544 TEST S1 ENTITIES) ===", flush=True)
    print("=" * 80, flush=True)
    t_start = time.time()

    init_rss, init_peak, init_avail = get_mem_stats()
    _, init_disk_free = check_system_resources("Startup")
    print(f"Initial State -> RAM: {init_rss:.2f} GB RSS, {init_avail:.2f} GB Avail | Free Disk: {init_disk_free:.2f} GB", flush=True)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_out_path = output_dir / "matching_results.tsv"
    candidate_out_path = output_dir / "candidate_pairs.tsv"
    checkpoint_path = output_dir / "checkpoint_full_inference.json"
    scratch_dir = Path(r"C:\Users\Sneha Singh\.gemini\antigravity\brain\47892a4a-c3be-4331-990a-46831df8639d\scratch")

    # 1. Verify / Initialize output files from pilot
    existing_m_count = count_tsv_data_rows(matching_out_path)
    existing_c_count = count_tsv_data_rows(candidate_out_path)
    print(f"Output files status: matching={existing_m_count:,} rows, candidate={existing_c_count:,} rows", flush=True)

    if existing_m_count == 0 or existing_c_count == 0:
        pilot_m = scratch_dir / "pilot_10k_matching_results.tsv"
        pilot_c = scratch_dir / "pilot_10k_candidate_pairs.tsv"
        if pilot_m.exists() and pilot_c.exists():
            print("Copying verified 10,000-S1 pilot outputs into output/ directory...", flush=True)
            shutil.copyfile(pilot_m, matching_out_path)
            shutil.copyfile(pilot_c, candidate_out_path)
            existing_m_count = count_tsv_data_rows(matching_out_path)
            existing_c_count = count_tsv_data_rows(candidate_out_path)
            print(f"Initialized output files with {existing_m_count:,} pilot rows.", flush=True)
        else:
            with open(matching_out_path, "w", encoding="utf-8") as fm:
                fm.write("source1_entity_id\tmatched_entity_ids\n")
            with open(candidate_out_path, "w", encoding="utf-8") as fc:
                fc.write("source1_entity_id\tcandidate_entity_ids\n")
            existing_m_count = 0
            existing_c_count = 0

    assert existing_m_count == existing_c_count, f"File row mismatch: matching={existing_m_count} vs candidate={existing_c_count}"
    start_s1_idx = existing_m_count
    remaining_s1 = TOTAL_TEST_S1 - start_s1_idx
    print(f"Resume point: starting from S1 entity index {start_s1_idx:,} ({remaining_s1:,} entities remaining).", flush=True)

    if remaining_s1 <= 0:
        print("All 1,732,544 entities are already processed! Proceeding directly to verification.", flush=True)
    else:
        # 2. Global Token Frequency Cache (Build or Load)
        freq_cache_file = scratch_dir / "global_test_token_freq.joblib"
        if freq_cache_file.exists():
            print(f"\nLoading global token frequency cache from {freq_cache_file.name}...", flush=True)
            t_fload = time.time()
            s2_freq, s3_freq = joblib.load(freq_cache_file)
            print(f"Loaded token frequencies ({len(s2_freq):,} S2, {len(s3_freq):,} S3) in {time.time() - t_fload:.2f}s.", flush=True)
        else:
            print("\nBuilding global token frequency cache across test S2 & S3...", flush=True)
            t_fbuild = time.time()
            # Tokenize all S1
            print("  Extracting query vocabulary across all test S1...", flush=True)
            s1_all_tokens = set()
            for c in pd.read_csv("dataset/test/test_source1.tsv", sep="\t", usecols=["business_name", "business_address"], chunksize=500_000, dtype=str):
                for r in c.itertuples(index=False):
                    comb = f"{r.business_name} {r.business_address}".lower()
                    s1_all_tokens.update(t for t in re.findall(r"[a-z0-9]+", comb) if len(t) >= 3)
            print(f"  Total unique query tokens in S1: {len(s1_all_tokens):,}", flush=True)

            s2_freq = Counter()
            s3_freq = Counter()
            print("  Streaming test_source2.tsv for global token frequencies...", flush=True)
            for chunk in pd.read_csv("dataset/test/test_source2.tsv", sep="\t", usecols=["business_name", "business_address"], chunksize=1_000_000, dtype=str):
                for r in chunk.itertuples(index=False):
                    comb = f"{r.business_name} {r.business_address}".lower()
                    for t in set(re.findall(r"[a-z0-9]+", comb)) & s1_all_tokens:
                        s2_freq[t] += 1
                check_system_resources("Global Freq S2")

            print("  Streaming test_source3.tsv for global token frequencies...", flush=True)
            for chunk in pd.read_csv("dataset/test/test_source3.tsv", sep="\t", usecols=["business_name", "business_address"], chunksize=1_000_000, dtype=str):
                for r in chunk.itertuples(index=False):
                    comb = f"{r.business_name} {r.business_address}".lower()
                    for t in set(re.findall(r"[a-z0-9]+", comb)) & s1_all_tokens:
                        s3_freq[t] += 1
                check_system_resources("Global Freq S3")

            joblib.dump((s2_freq, s3_freq), freq_cache_file)
            print(f"Global token frequency cache built and saved in {time.time() - t_fbuild:.2f}s.", flush=True)
            del s1_all_tokens
            gc.collect()

        # 3. Load Model
        print(f"\nLoading {Path('models/xgboost_matcher.joblib').name}...", flush=True)
        model = load_model("models/xgboost_matcher.joblib")

        # 4. Stream Remaining S1 Entities in Batches of BATCH_S1_SIZE
        print("\n" + "=" * 80, flush=True)
        print(f"--- STREAMING INFERENCE OVER REMAINING ENTITIES ({BATCH_S1_SIZE:,} S1/batch) ---", flush=True)
        print("=" * 80, flush=True)

        current_idx = start_s1_idx
        n_total_batches = ceil(remaining_s1 / BATCH_S1_SIZE)
        batch_num = 0

        # Read S1 iterator skipping already processed rows
        s1_reader = pd.read_csv(
            "dataset/test/test_source1.tsv",
            sep="\t",
            skiprows=range(1, start_s1_idx + 1) if start_s1_idx > 0 else None,
            chunksize=BATCH_S1_SIZE,
            dtype=str
        )

        for batch_s1_df in s1_reader:
            batch_num += 1
            b_count = len(batch_s1_df)
            b_start = current_idx
            b_end = current_idx + b_count
            t_batch_start = time.time()

            print(f"\n[Batch {batch_num:>3}/{n_total_batches}] Processing S1 entities [{b_start:,} : {b_end:,}] ({b_count:,} entities)...", flush=True)
            check_system_resources(f"Batch {batch_num} start")

            batch_s1_df = add_normalized_columns(batch_s1_df)

            # Determine top-3 rare tokens per S1 entity in this batch
            s1_top_tokens = {}
            for row in batch_s1_df.itertuples(index=False):
                toks = set(t for t in tokenize(row.name_norm) + tokenize(row.address_norm) if len(t) >= 3)
                token_info = []
                for t in toks:
                    f2 = s2_freq.get(t, 0)
                    f3 = s3_freq.get(t, 0)
                    freq = max(f2, f3)
                    if 0 < freq <= 100_000:
                        token_info.append((freq, t))
                token_info.sort()
                s1_top_tokens[row.entity_id] = [t for _, t in token_info[:3]]

            # Country-partitioned execution for this batch
            batch_matching = {}
            batch_candidates = {}

            for country in ["france", "us", "india"]:
                c_mask = batch_s1_df["country"].str.strip().str.lower() == country
                country_s1 = batch_s1_df[c_mask].copy().reset_index(drop=True)
                n_c_s1 = len(country_s1)
                if n_c_s1 == 0:
                    continue

                country_query_names = set(country_s1["name_norm"]) - {""}
                country_query_addrs = set(country_s1["address_norm"]) - {""}
                country_tokens = set()
                country_s1_lookup = {}

                for r in country_s1.itertuples(index=False):
                    country_s1_lookup[r.entity_id] = (
                        str(r.business_name) if pd.notna(r.business_name) else "",
                        str(r.business_address) if pd.notna(r.business_address) else "",
                        str(r.country).strip().lower() if pd.notna(r.country) else "",
                    )
                    country_tokens.update(s1_top_tokens.get(r.entity_id, []))

                # Pass 2: Stream target records for this country
                s2_name_ids = defaultdict(set)
                s3_name_ids = defaultdict(set)
                s2_addr_ids = defaultdict(set)
                s3_addr_ids = defaultdict(set)
                s2_token_idx = defaultdict(list)
                s3_token_idx = defaultdict(list)
                cand_records = {}

                for chunk in pd.read_csv("dataset/test/test_source2.tsv", sep="\t", chunksize=1_000_000, dtype=str):
                    sub = chunk[chunk["country"].str.strip().str.lower() == country]
                    if sub.empty:
                        continue
                    for r in sub.itertuples(index=False):
                        n_norm = normalize_name(r.business_name)
                        a_norm = normalize_address(r.business_address)
                        is_cand = False
                        if n_norm and n_norm in country_query_names:
                            s2_name_ids[n_norm].add(r.entity_id)
                            is_cand = True
                        if a_norm and a_norm in country_query_addrs:
                            s2_addr_ids[a_norm].add(r.entity_id)
                            is_cand = True
                        comb = f"{r.business_name or ''} {r.business_address or ''}".lower()
                        toks = set(t for t in re.findall(r"[a-z0-9]+", comb) if len(t) >= 3)
                        matched = toks & country_tokens
                        if matched:
                            for t in matched:
                                s2_token_idx[t].append(r.entity_id)
                            is_cand = True
                        if is_cand:
                            cand_records[r.entity_id] = (
                                str(r.business_name) if pd.notna(r.business_name) else "",
                                str(r.business_address) if pd.notna(r.business_address) else "",
                                str(r.country).strip().lower() if pd.notna(r.country) else "",
                            )

                for chunk in pd.read_csv("dataset/test/test_source3.tsv", sep="\t", chunksize=1_000_000, dtype=str):
                    sub = chunk[chunk["country"].str.strip().str.lower() == country]
                    if sub.empty:
                        continue
                    for r in sub.itertuples(index=False):
                        n_norm = normalize_name(r.business_name)
                        a_norm = normalize_address(r.business_address)
                        is_cand = False
                        if n_norm and n_norm in country_query_names:
                            s3_name_ids[n_norm].add(r.entity_id)
                            is_cand = True
                        if a_norm and a_norm in country_query_addrs:
                            s3_addr_ids[a_norm].add(r.entity_id)
                            is_cand = True
                        comb = f"{r.business_name or ''} {r.business_address or ''}".lower()
                        toks = set(t for t in re.findall(r"[a-z0-9]+", comb) if len(t) >= 3)
                        matched = toks & country_tokens
                        if matched:
                            for t in matched:
                                s3_token_idx[t].append(r.entity_id)
                            is_cand = True
                        if is_cand:
                            cand_records[r.entity_id] = (
                                str(r.business_name) if pd.notna(r.business_name) else "",
                                str(r.business_address) if pd.notna(r.business_address) else "",
                                str(r.country).strip().lower() if pd.notna(r.country) else "",
                            )

                # Chunked scoring for this country
                n_chunks = ceil(n_c_s1 / CHUNK_S1_SIZE)
                for chunk_idx in range(n_chunks):
                    c_start = chunk_idx * CHUNK_S1_SIZE
                    c_end = min((chunk_idx + 1) * CHUNK_S1_SIZE, n_c_s1)
                    chunk_s1_sub = country_s1.iloc[c_start:c_end]

                    pairs_s1 = []
                    pairs_cand = []
                    chunk_candidates_by_s1 = {}

                    for row in chunk_s1_sub.itertuples(index=False):
                        sid = row.entity_id
                        cands = set()
                        if row.name_norm:
                            cands.update(s2_name_ids.get(row.name_norm, ()))
                            cands.update(s3_name_ids.get(row.name_norm, ()))
                        if row.address_norm:
                            cands.update(s2_addr_ids.get(row.address_norm, ()))
                            cands.update(s3_addr_ids.get(row.address_norm, ()))
                        for token in s1_top_tokens.get(sid, []):
                            cands.update(s2_token_idx.get(token, ()))
                            cands.update(s3_token_idx.get(token, ()))

                        cands_list = sorted(cands)
                        chunk_candidates_by_s1[sid] = cands_list
                        batch_candidates[sid] = cands_list

                        for cid in cands_list:
                            pairs_s1.append(sid)
                            pairs_cand.append(cid)

                    # Score pairs
                    chunk_matches_by_s1 = defaultdict(list)
                    n_pairs = len(pairs_s1)

                    for b_pair_start in range(0, n_pairs, BATCH_PAIR_SIZE):
                        b_pair_end = min(b_pair_start + BATCH_PAIR_SIZE, n_pairs)
                        b_s1 = pairs_s1[b_pair_start:b_pair_end]
                        b_cand = pairs_cand[b_pair_start:b_pair_end]

                        name1 = [country_s1_lookup[sid][0] for sid in b_s1]
                        addr1 = [country_s1_lookup[sid][1] for sid in b_s1]
                        country1 = [country_s1_lookup[sid][2] for sid in b_s1]

                        name2 = [cand_records[cid][0] for cid in b_cand]
                        addr2 = [cand_records[cid][1] for cid in b_cand]
                        country2 = [cand_records[cid][2] for cid in b_cand]

                        name_ratio = process.cpdist(name1, name2, scorer=fuzz.ratio, workers=-1, dtype=np.float64) / 100.0
                        name_token_sort = process.cpdist(name1, name2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64) / 100.0
                        name_token_set = process.cpdist(name1, name2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64) / 100.0
                        address_ratio = process.cpdist(addr1, addr2, scorer=fuzz.ratio, workers=-1, dtype=np.float64) / 100.0
                        address_token_sort = process.cpdist(addr1, addr2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64) / 100.0
                        address_token_set = process.cpdist(addr1, addr2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64) / 100.0
                        country_same = (np.array(country1) == np.array(country2)).astype(np.int64)
                        name_len_1 = np.array([len(x) for x in name1], dtype=np.int64)
                        name_len_2 = np.array([len(x) for x in name2], dtype=np.int64)
                        address_len_1 = np.array([len(x) for x in addr1], dtype=np.int64)
                        address_len_2 = np.array([len(x) for x in addr2], dtype=np.int64)

                        X_batch = pd.DataFrame({
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
                        })

                        probs = model.predict_proba(X_batch[LOCKED_FEATURE_COLUMNS])[:, 1]
                        above_idx = np.where(probs >= THRESHOLD)[0]
                        for idx in above_idx:
                            chunk_matches_by_s1[b_s1[idx]].append(b_cand[idx])

                        del name1, addr1, country1, name2, addr2, country2
                        del name_ratio, name_token_sort, name_token_set, address_ratio, address_token_sort, address_token_set
                        del country_same, name_len_1, name_len_2, address_len_1, address_len_2, X_batch, probs, above_idx
                        gc.collect()

                    for sid in chunk_s1_sub["entity_id"]:
                        cands_set = set(chunk_candidates_by_s1.get(sid, []))
                        raw_m = chunk_matches_by_s1.get(sid, [])
                        valid_m = [m for m in dict.fromkeys(raw_m) if m in cands_set]
                        batch_matching[sid] = valid_m

                    del pairs_s1, pairs_cand, chunk_candidates_by_s1, chunk_matches_by_s1
                    gc.collect()

                del s2_name_ids, s3_name_ids, s2_addr_ids, s3_addr_ids, s2_token_idx, s3_token_idx, cand_records, country_s1_lookup
                gc.collect()

            # Append batch results in exact S1 order
            with open(matching_out_path, "a", encoding="utf-8") as fm, open(candidate_out_path, "a", encoding="utf-8") as fc:
                for sid in batch_s1_df["entity_id"]:
                    cands = batch_candidates.get(sid, [])
                    matches = batch_matching.get(sid, [])
                    fc.write(f"{sid}\t{','.join(cands)}\n")
                    fm.write(f"{sid}\t{','.join(matches)}\n")
                fm.flush()
                fc.flush()

            current_idx = b_end
            b_time = time.time() - t_batch_start

            # Save persistent checkpoint
            checkpoint_data = {
                "completed_count": current_idx,
                "total_entities": TOTAL_TEST_S1,
                "last_entity_id": batch_s1_df["entity_id"].iloc[-1],
                "last_batch_time_s": b_time,
                "status": "running" if current_idx < TOTAL_TEST_S1 else "completed",
                "timestamp": time.time(),
            }
            with open(checkpoint_path, "w", encoding="utf-8") as f_ckpt:
                json.dump(checkpoint_data, f_ckpt, indent=2)

            _, b_peak, b_avail = get_mem_stats()
            _, b_disk = check_system_resources(f"Batch {batch_num} end")
            print(f"  [Batch {batch_num:>3}] Completed in {b_time:.1f}s | Total Done: {current_idx:,}/{TOTAL_TEST_S1:,} ({current_idx/TOTAL_TEST_S1*100:.1f}%) | Peak RAM: {b_peak:.2f} GB | Free Disk: {b_disk:.2f} GB", flush=True)

            del batch_matching, batch_candidates, s1_top_tokens, batch_s1_df
            gc.collect()

    # 5. Full-Scale Official Validation
    print("\n" + "=" * 80, flush=True)
    print("--- 5. OFFICIAL SUBMISSION VALIDATION SUITE ---", flush=True)
    print("=" * 80, flush=True)

    final_m_count = count_tsv_data_rows(matching_out_path)
    final_c_count = count_tsv_data_rows(candidate_out_path)
    print(f"Final Output Files: matching={final_m_count:,} rows, candidate={final_c_count:,} rows", flush=True)

    cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", str(matching_out_path),
        "--candidate", str(candidate_out_path),
        "--test-dir", "dataset/test",
    ]
    print(f"Running validator: {' '.join(cmd)}...", flush=True)
    val_res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Validator Output:\n{val_res.stdout}")
    if val_res.stderr:
        print(f"Validator Errors:\n{val_res.stderr}")

    assert val_res.returncode == 0, f"Submission validator returned non-zero code {val_res.returncode}"

    # 6. Final Report Metrics
    m_size_mb = matching_out_path.stat().st_size / (1024 ** 2)
    c_size_mb = candidate_out_path.stat().st_size / (1024 ** 2)
    total_run_time = time.time() - t_start
    final_rss, final_peak, final_avail = get_mem_stats()

    print("\n" + "=" * 80, flush=True)
    print("=== FULL TEST INFERENCE EXECUTION REPORT ===", flush=True)
    print("=" * 80, flush=True)
    print(f"Total S1 processed:             {final_m_count:,}")
    print(f"Candidate pairs file size:      {c_size_mb:.2f} MB ({c_size_mb/1024:.2f} GB)")
    print(f"Matching results file size:     {m_size_mb:.2f} MB")
    print(f"Total pipeline runtime:         {total_run_time:.2f}s ({total_run_time/60:.2f} min)")
    print(f"Peak Working Set RAM:           {final_peak:.2f} GB")
    print(f"Final Available System RAM:     {final_avail:.2f} GB")
    print(f"Official Validator Status:      PASS (Exit code 0)")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
