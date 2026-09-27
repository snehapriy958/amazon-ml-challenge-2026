"""Full 1,732,544-Entity Test Inference Runner & Output Auditor for Frozen V3 Pipeline.

Strictly executes the verified task-2471 SharedMemory 8-Worker V3 Engine:
  - Frozen Model: C_LightGBM_LambdaRank (experiments/v3/models/v3_model_c_lambdarank.joblib)
  - Frozen Features: 88 V3-B normalized/numeric/phonetic features
  - Frozen Calibration: Platt probability calibration
  - Frozen Policy: Entity Policy 5 (t_us=0.995, t_in=0.995, t_default=0.995, max_k=6)
  - Validated Top-500 Retrieval Prefilter (preserving all Channel-1 exact matches + fair S2/S3 quota)

Outputs are written strictly to experiments/v3/output/ (never touching V1 output/ or V2-C artifacts):
  - experiments/v3/output/matching_results.tsv
  - experiments/v3/output/candidate_pairs.tsv
  - experiments/v3/output/checkpoint_v3_test_inference.json
  - experiments/v3/output/v3_test_output_audit.json
"""

import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path.cwd()))
from experiments.v3.src.fast_inference_engine_v3 import (
    CountryIndexV3,
    init_inference_worker,
)

TEST_S1_PATH = Path("dataset/test/test_source1.tsv")
TEST_S2_PATH = Path("dataset/test/test_source2.tsv")
TEST_S3_PATH = Path("dataset/test/test_source3.tsv")

MODEL_PATH = Path("experiments/v3/models/v3_model_c_lambdarank.joblib")
METRICS_CFG_PATH = Path("experiments/v3/results/v3_d_bakeoff_metrics.json")

V3_OUT_DIR = Path("experiments/v3/output")
CHUNKS_DIR = V3_OUT_DIR / "chunks"
CHECKPOINT_PATH = V3_OUT_DIR / "checkpoint_v3_test_inference.json"
FINAL_MATCH_PATH = V3_OUT_DIR / "matching_results.tsv"
FINAL_CANDS_PATH = V3_OUT_DIR / "candidate_pairs.tsv"
AUDIT_JSON_PATH = V3_OUT_DIR / "v3_test_output_audit.json"

# Frozen V1 artifacts that must NEVER be modified
V1_MATCH_PATH = Path("output/matching_results.tsv")
V1_CANDS_PATH = Path("output/candidate_pairs.tsv")

TOP_K = 500
CHUNK_SIZE_S1 = 10_000
NUM_WORKERS = 8
NUM_SHARDS_PER_CHUNK = 16


def _save_checkpoint_atomic(state: dict) -> None:
    tmp_path = CHECKPOINT_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, CHECKPOINT_PATH)


def _load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "completed_chunks": {},
        "country_stats": {},
        "processed_s1_total": 0,
        "total_s1": 1_732_544,
        "total_candidate_pairs": 0,
        "total_predicted_matches": 0,
        "non_empty_s1_count": 0,
        "cumulative_index_sec": 0.0,
        "cumulative_retrieval_sec": 0.0,
        "cumulative_scoring_sec": 0.0,
        "cumulative_query_sec": 0.0,
        "history": [],
    }


def merge_country_streams_to_final_outputs(
    total_s1: int,
    countries: List[str],
    chunk_counts: Dict[str, int],
) -> None:
    """Perform an O(1)-RAM multi-way merge across country chunk files ordered by orig_row_idx (0..N-1)."""
    t0 = time.time()
    print("\n" + "=" * 80, flush=True)
    print(f"ASSEMBLING FINAL SUBMISSION FILES IN EXACT TEST_SOURCE1 ORDER ({total_s1:,} ROWS)", flush=True)
    print("=" * 80, flush=True)

    # Open per-country chunk generators (within each country, chunk_000..chunk_K are strictly increasing in orig_row_idx)
    def _country_row_iter(prefix: str, country: str, n_chunks: int):
        for c_idx in range(n_chunks):
            p = CHUNKS_DIR / f"{prefix}_{country}_{c_idx:03d}.tsv"
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    yield int(parts[0]), parts[1], (parts[2] if len(parts) > 2 else "")

    # 1. Merge matching_results.tsv
    tmp_match = FINAL_MATCH_PATH.with_suffix(".tsv.tmp")
    iters_m = {c: _country_row_iter("match", c, chunk_counts[c]) for c in countries if chunk_counts.get(c, 0) > 0}
    heads_m: Dict[str, Tuple[int, str, str]] = {}
    for c, it in list(iters_m.items()):
        try:
            heads_m[c] = next(it)
        except StopIteration:
            del iters_m[c]

    written_m = 0
    with open(tmp_match, "w", encoding="utf-8", newline="\n", buffering=16 * 1024 * 1024) as out_m:
        out_m.write("source1_entity_id\tmatched_entity_ids\n")
        for expected_idx in range(total_s1):
            # Find which country stream has expected_idx at its head
            chosen_c = None
            for c, head in heads_m.items():
                if head[0] == expected_idx:
                    chosen_c = c
                    break
            if chosen_c is None:
                raise RuntimeError(f"Missing orig_row_idx={expected_idx} during matching_results.tsv merge!")
            _, sid, m_str = heads_m[chosen_c]
            out_m.write(f"{sid}\t{m_str}\n")
            written_m += 1
            try:
                heads_m[chosen_c] = next(iters_m[chosen_c])
            except StopIteration:
                del heads_m[chosen_c]
                del iters_m[chosen_c]

    os.replace(tmp_match, FINAL_MATCH_PATH)
    print(f"  [1/2] Wrote {FINAL_MATCH_PATH} ({written_m:,} rows) in {time.time()-t0:.1f}s", flush=True)

    # 2. Merge candidate_pairs.tsv
    t1 = time.time()
    tmp_cands = FINAL_CANDS_PATH.with_suffix(".tsv.tmp")
    iters_c = {c: _country_row_iter("cands", c, chunk_counts[c]) for c in countries if chunk_counts.get(c, 0) > 0}
    heads_c: Dict[str, Tuple[int, str, str]] = {}
    for c, it in list(iters_c.items()):
        try:
            heads_c[c] = next(it)
        except StopIteration:
            del iters_c[c]

    written_c = 0
    with open(tmp_cands, "w", encoding="utf-8", newline="\n", buffering=32 * 1024 * 1024) as out_c:
        out_c.write("source1_entity_id\tcandidate_entity_ids\n")
        for expected_idx in range(total_s1):
            chosen_c = None
            for c, head in heads_c.items():
                if head[0] == expected_idx:
                    chosen_c = c
                    break
            if chosen_c is None:
                raise RuntimeError(f"Missing orig_row_idx={expected_idx} during candidate_pairs.tsv merge!")
            _, sid, c_str = heads_c[chosen_c]
            out_c.write(f"{sid}\t{c_str}\n")
            written_c += 1
            try:
                heads_c[chosen_c] = next(iters_c[chosen_c])
            except StopIteration:
                del heads_c[chosen_c]
                del iters_c[chosen_c]

    os.replace(tmp_cands, FINAL_CANDS_PATH)
    print(f"  [2/2] Wrote {FINAL_CANDS_PATH} ({written_c:,} rows) in {time.time()-t1:.1f}s", flush=True)


def run_comprehensive_output_audit(expected_s1_ids: List[str], v1_pre_stat: dict) -> dict:
    """Audit matching_results.tsv and candidate_pairs.tsv against all challenge and V3 invariants."""
    t0 = time.time()
    print("\n" + "=" * 80, flush=True)
    print("PERFORMING FULL READ-ONLY OUTPUT AUDIT ON V3 SUBMISSION FILES", flush=True)
    print("=" * 80, flush=True)

    n_expected = len(expected_s1_ids)
    m_rows = 0
    c_rows = 0
    id_order_mismatches = 0
    subset_violations = 0
    dup_match_violations = 0
    dup_cand_violations = 0
    max_k_violations = 0
    zero_cand_rows = 0
    non_empty_match_rows = 0
    total_matches = 0
    total_cands = 0
    s2_matches = 0
    s3_matches = 0
    match_count_hist: Dict[int, int] = {}

    with (
        open(FINAL_MATCH_PATH, "r", encoding="utf-8") as fm,
        open(FINAL_CANDS_PATH, "r", encoding="utf-8") as fc,
    ):
        hm = fm.readline().rstrip("\r\n")
        hc = fc.readline().rstrip("\r\n")
        assert hm == "source1_entity_id\tmatched_entity_ids", f"Invalid matching header: {hm!r}"
        assert hc == "source1_entity_id\tcandidate_entity_ids", f"Invalid candidate header: {hc!r}"

        for idx in range(n_expected):
            lm = fm.readline()
            lc = fc.readline()
            if not lm or not lc:
                raise RuntimeError(f"Premature EOF at row {idx}!")
            m_rows += 1
            c_rows += 1

            pm = lm.rstrip("\r\n").split("\t")
            pc = lc.rstrip("\r\n").split("\t")
            sid_m = pm[0]
            sid_c = pc[0]
            exp_sid = expected_s1_ids[idx]

            if sid_m != exp_sid or sid_c != exp_sid:
                id_order_mismatches += 1

            m_str = pm[1] if len(pm) > 1 else ""
            c_str = pc[1] if len(pc) > 1 else ""

            c_list = c_str.split(",") if c_str else []
            m_list = m_str.split(",") if m_str else []

            if not c_list:
                zero_cand_rows += 1
            c_set = set(c_list)
            if len(c_set) != len(c_list):
                dup_cand_violations += 1
            total_cands += len(c_list)

            if m_list:
                non_empty_match_rows += 1
                m_set = set(m_list)
                if len(m_set) != len(m_list):
                    dup_match_violations += 1
                if not m_set.issubset(c_set):
                    subset_violations += 1
                if len(m_list) > 6:
                    max_k_violations += 1
                total_matches += len(m_list)
                for mid in m_list:
                    if mid.startswith("S2-"):
                        s2_matches += 1
                    elif mid.startswith("S3-"):
                        s3_matches += 1
            k_len = len(m_list)
            match_count_hist[k_len] = match_count_hist.get(k_len, 0) + 1

        extra_m = fm.readline()
        extra_c = fc.readline()
        assert extra_m == "", f"Extra trailing lines in matching_results.tsv: {extra_m!r}"
        assert extra_c == "", f"Extra trailing lines in candidate_pairs.tsv: {extra_c!r}"

    # Verify V1 frozen files were NOT touched
    v1_post_stat = {
        "v1_matching_size": V1_MATCH_PATH.stat().st_size if V1_MATCH_PATH.exists() else None,
        "v1_matching_mtime": V1_MATCH_PATH.stat().st_mtime if V1_MATCH_PATH.exists() else None,
        "v1_cands_size": V1_CANDS_PATH.stat().st_size if V1_CANDS_PATH.exists() else None,
        "v1_cands_mtime": V1_CANDS_PATH.stat().st_mtime if V1_CANDS_PATH.exists() else None,
    }
    v1_untouched = (v1_pre_stat == v1_post_stat)

    audit_res = {
        "status": "PASS" if (
            m_rows == n_expected
            and c_rows == n_expected
            and id_order_mismatches == 0
            and subset_violations == 0
            and dup_match_violations == 0
            and dup_cand_violations == 0
            and max_k_violations == 0
            and v1_untouched
        ) else "FAIL",
        "matching_results_path": str(FINAL_MATCH_PATH),
        "matching_results_size_mb": round(FINAL_MATCH_PATH.stat().st_size / (1024 ** 2), 2),
        "candidate_pairs_path": str(FINAL_CANDS_PATH),
        "candidate_pairs_size_mb": round(FINAL_CANDS_PATH.stat().st_size / (1024 ** 2), 2),
        "expected_s1_rows": n_expected,
        "matching_rows": m_rows,
        "candidate_rows": c_rows,
        "id_order_mismatches": id_order_mismatches,
        "subset_violations": subset_violations,
        "dup_match_violations": dup_match_violations,
        "dup_cand_violations": dup_cand_violations,
        "max_k_violations": max_k_violations,
        "zero_candidate_rows": zero_cand_rows,
        "zero_candidate_rate_pct": round(zero_cand_rows / n_expected * 100.0, 4),
        "avg_candidates_per_s1": round(total_cands / n_expected, 2),
        "total_candidate_pairs": total_cands,
        "non_empty_s1_predictions": non_empty_match_rows,
        "non_empty_s1_rate_pct": round(non_empty_match_rows / n_expected * 100.0, 2),
        "singleton_s1_predictions": n_expected - non_empty_match_rows,
        "singleton_s1_rate_pct": round((n_expected - non_empty_match_rows) / n_expected * 100.0, 2),
        "total_predicted_matched_ids": total_matches,
        "avg_matches_per_s1_all": round(total_matches / n_expected, 3),
        "avg_matches_per_non_empty_s1": round(total_matches / max(1, non_empty_match_rows), 3),
        "s2_predicted_matches": s2_matches,
        "s3_predicted_matches": s3_matches,
        "s2_share_pct": round(s2_matches / max(1, total_matches) * 100.0, 2),
        "s3_share_pct": round(s3_matches / max(1, total_matches) * 100.0, 2),
        "match_count_distribution": {str(k): match_count_hist[k] for k in sorted(match_count_hist.keys())},
        "v1_frozen_artifacts_untouched": v1_untouched,
        "audit_elapsed_sec": round(time.time() - t0, 2),
    }

    with open(AUDIT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_res, f, indent=2)

    print(json.dumps(audit_res, indent=2), flush=True)
    return audit_res


def main() -> None:
    t_run_start = time.time()
    V3_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Record V1 frozen artifact stats prior to execution
    v1_pre_stat = {
        "v1_matching_size": V1_MATCH_PATH.stat().st_size if V1_MATCH_PATH.exists() else None,
        "v1_matching_mtime": V1_MATCH_PATH.stat().st_mtime if V1_MATCH_PATH.exists() else None,
        "v1_cands_size": V1_CANDS_PATH.stat().st_size if V1_CANDS_PATH.exists() else None,
        "v1_cands_mtime": V1_CANDS_PATH.stat().st_mtime if V1_CANDS_PATH.exists() else None,
    }

    with open(METRICS_CFG_PATH, "r", encoding="utf-8") as f:
        frozen_cfg = json.load(f)["selected_frozen_pipeline"]
    policy_cfg = frozen_cfg["policy_config"]

    print("=" * 80, flush=True)
    print("STARTING FULL 1,732,544-ENTITY TEST INFERENCE (FROZEN V3 PIPELINE, TOP-500)", flush=True)
    print(f"  Model:      {MODEL_PATH}", flush=True)
    print(f"  Policy 5:   {policy_cfg}", flush=True)
    print(f"  Top-K:      {TOP_K} | Chunk Size: {CHUNK_SIZE_S1:,} S1 | Workers: {NUM_WORKERS}", flush=True)
    print(f"  Output Dir: {V3_OUT_DIR} (V1 output/ and V2-C strictly untouched)", flush=True)
    print("=" * 80, flush=True)

    # Load test_source1.tsv and partition by country while preserving orig_row_idx
    t_load0 = time.time()
    s1_df = pd.read_csv(TEST_S1_PATH, sep="\t", dtype=str)
    total_s1 = len(s1_df)
    assert total_s1 == 1_732_544, f"Expected 1,732,544 S1 rows, got {total_s1}"

    expected_s1_ids: List[str] = list(s1_df["entity_id"])
    records_by_country: Dict[str, List[Tuple[int, str, str, str]]] = {
        "france": [],
        "us": [],
        "india": [],
    }

    for row_idx, (sid, bname, baddr, ctry_raw) in enumerate(
        zip(s1_df["entity_id"], s1_df["business_name"], s1_df["business_address"], s1_df["country"])
    ):
        c_norm = "" if pd.isna(ctry_raw) else str(ctry_raw).strip().lower()
        n_str = "" if pd.isna(bname) else str(bname)
        a_str = "" if pd.isna(baddr) else str(baddr)
        if c_norm not in records_by_country:
            records_by_country[c_norm] = []
        records_by_country[c_norm].append((row_idx, sid, n_str, a_str))

    del s1_df
    gc.collect()

    countries_order = ["france", "us", "india"]
    for c in list(records_by_country.keys()):
        if c not in countries_order and len(records_by_country[c]) > 0:
            countries_order.append(c)

    print(
        f"Loaded {total_s1:,} test S1 entities in {time.time()-t_load0:.1f}s: "
        + ", ".join(f"{c.upper()}={len(records_by_country[c]):,}" for c in countries_order),
        flush=True,
    )

    state = _load_checkpoint()
    chunk_counts: Dict[str, int] = {}
    for c in countries_order:
        n_c = len(records_by_country[c])
        chunk_counts[c] = (n_c + CHUNK_SIZE_S1 - 1) // CHUNK_SIZE_S1

    total_chunks_all = sum(chunk_counts.values())

    print(f"Starting persistent worker pool ({NUM_WORKERS} workers)...", flush=True)
    pool = mp.Pool(
        processes=NUM_WORKERS,
        initializer=init_inference_worker,
        initargs=(str(MODEL_PATH), policy_cfg, 1),
    )

    country_index = None
    try:
        for ctry in countries_order:
            c_recs = records_by_country[ctry]
            n_c = len(c_recs)
            if n_c == 0:
                continue
            n_chunks = chunk_counts[ctry]

            # Check if all chunks for this country are already completed on disk
            all_c_done = all(
                state["completed_chunks"].get(f"{ctry}_{chunk_idx:03d}", False)
                and (CHUNKS_DIR / f"match_{ctry}_{chunk_idx:03d}.tsv").exists()
                and (CHUNKS_DIR / f"cands_{ctry}_{chunk_idx:03d}.tsv").exists()
                for chunk_idx in range(n_chunks)
            )
            if all_c_done:
                print(f"[{ctry.upper()}] All {n_chunks} chunks ({n_c:,} S1) already checkpointed; skipping.", flush=True)
                continue

            print("\n" + "-" * 80, flush=True)
            print(f"[{ctry.upper()}] Building SharedMemory Index for {n_c:,} S1 queries ({n_chunks} chunks)...", flush=True)
            t_idx0 = time.time()
            country_index = CountryIndexV3(ctry)
            country_index.build(
                c_recs,
                str(TEST_S2_PATH),
                str(TEST_S3_PATH),
                worker_pool=pool,
                num_workers=NUM_WORKERS,
            )
            idx_el = time.time() - t_idx0
            state["cumulative_index_sec"] = round(float(state.get("cumulative_index_sec", 0.0)) + idx_el, 2)
            _save_checkpoint_atomic(state)

            for chunk_idx in range(n_chunks):
                chunk_key = f"{ctry}_{chunk_idx:03d}"
                match_chunk_path = CHUNKS_DIR / f"match_{ctry}_{chunk_idx:03d}.tsv"
                cands_chunk_path = CHUNKS_DIR / f"cands_{ctry}_{chunk_idx:03d}.tsv"

                if (
                    state["completed_chunks"].get(chunk_key, False)
                    and match_chunk_path.exists()
                    and cands_chunk_path.exists()
                ):
                    print(f"  [{ctry.upper()} Chunk {chunk_idx+1}/{n_chunks}] Already completed; skipping.", flush=True)
                    continue

                st = chunk_idx * CHUNK_SIZE_S1
                en = min(st + CHUNK_SIZE_S1, n_c)
                sub_recs = c_recs[st:en]
                n_sub = len(sub_recs)

                t_c0 = time.time()
                cands_c, preds_c, n_pairs_c, el_r, el_s = country_index.retrieve_and_score_chunk(
                    sub_recs,
                    worker_pool=pool,
                    top_k=TOP_K,
                    num_shards=NUM_SHARDS_PER_CHUNK,
                )

                # Write chunk files atomically
                tmp_m = match_chunk_path.with_suffix(".tsv.tmp")
                tmp_c = cands_chunk_path.with_suffix(".tsv.tmp")

                chunk_matches_cnt = 0
                chunk_non_empty_cnt = 0

                with (
                    open(tmp_m, "w", encoding="utf-8", newline="\n", buffering=4 * 1024 * 1024) as fm,
                    open(tmp_c, "w", encoding="utf-8", newline="\n", buffering=8 * 1024 * 1024) as fc,
                ):
                    for (orig_i, sid, _, _), c_ids, p_ids in zip(sub_recs, cands_c, preds_c):
                        if p_ids:
                            chunk_non_empty_cnt += 1
                            chunk_matches_cnt += len(p_ids)
                        fm.write(f"{orig_i}\t{sid}\t{','.join(p_ids)}\n")
                        fc.write(f"{orig_i}\t{sid}\t{','.join(c_ids)}\n")

                os.replace(tmp_m, match_chunk_path)
                os.replace(tmp_c, cands_chunk_path)

                el_chunk = time.time() - t_c0
                vm = psutil.virtual_memory()
                avail_gb = vm.available / (1024 ** 3)
                used_gb = vm.used / (1024 ** 3)

                state["completed_chunks"][chunk_key] = True
                state["processed_s1_total"] = int(state.get("processed_s1_total", 0)) + n_sub
                state["total_candidate_pairs"] = int(state.get("total_candidate_pairs", 0)) + int(n_pairs_c)
                state["total_predicted_matches"] = int(state.get("total_predicted_matches", 0)) + chunk_matches_cnt
                state["non_empty_s1_count"] = int(state.get("non_empty_s1_count", 0)) + chunk_non_empty_cnt
                state["cumulative_retrieval_sec"] = round(float(state.get("cumulative_retrieval_sec", 0.0)) + el_r, 2)
                state["cumulative_scoring_sec"] = round(float(state.get("cumulative_scoring_sec", 0.0)) + el_s, 2)
                state["cumulative_query_sec"] = round(float(state.get("cumulative_query_sec", 0.0)) + el_chunk, 2)

                proc_tot = state["processed_s1_total"]
                cum_q_sec = max(1e-6, float(state["cumulative_query_sec"]))
                obs_s1_s = proc_tot / cum_q_sec
                obs_pairs_s = state["total_candidate_pairs"] / cum_q_sec
                rem_s1 = total_s1 - proc_tot
                eta_hr = (rem_s1 / max(1e-6, obs_s1_s)) / 3600.0

                done_chunks_cnt = len(state["completed_chunks"])
                chunk_stat = {
                    "chunk_key": chunk_key,
                    "country": ctry,
                    "chunk_idx": chunk_idx + 1,
                    "country_chunks": n_chunks,
                    "global_chunk_idx": done_chunks_cnt,
                    "global_chunks_total": total_chunks_all,
                    "chunk_s1": n_sub,
                    "processed_s1_total": proc_tot,
                    "pct_complete": round(proc_tot / total_s1 * 100.0, 2),
                    "chunk_pairs": int(n_pairs_c),
                    "chunk_sec": round(el_chunk, 2),
                    "chunk_ret_sec": round(el_r, 2),
                    "chunk_score_sec": round(el_s, 2),
                    "chunk_s1_per_sec": round(n_sub / max(1e-6, el_chunk), 1),
                    "chunk_pairs_per_sec": round(n_pairs_c / max(1e-6, el_chunk), 0),
                    "cumulative_s1_per_sec": round(obs_s1_s, 1),
                    "cumulative_pairs_per_sec": round(obs_pairs_s, 0),
                    "avail_ram_gb": round(avail_gb, 2),
                    "used_ram_gb": round(used_gb, 2),
                    "remaining_eta_hours": round(eta_hr, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                state["latest_status"] = chunk_stat
                state["history"].append(chunk_stat)
                _save_checkpoint_atomic(state)

                print(
                    f"  [{ctry.upper()} {chunk_idx+1:02d}/{n_chunks:02d} | Global {done_chunks_cnt:02d}/{total_chunks_all:02d}] "
                    f"S1={proc_tot:,}/{total_s1:,} ({proc_tot/total_s1*100:.1f}%) | "
                    f"Chunk={el_chunk:.1f}s ({n_sub/el_chunk:.1f} S1/s, {n_pairs_c/el_chunk:,.0f} pairs/s) | "
                    f"Avg={obs_s1_s:.1f} S1/s | AvailRAM={avail_gb:.2f}GB | ETA={eta_hr:.2f}h",
                    flush=True,
                )

                del cands_c, preds_c, sub_recs

            country_index.close_and_cleanup(worker_pool=pool, num_workers=NUM_WORKERS)
            country_index = None
            records_by_country[ctry].clear()
            gc.collect()
    finally:
        if country_index is not None:
            try:
                country_index.close_and_cleanup(worker_pool=pool, num_workers=NUM_WORKERS)
            except Exception:
                pass
        pool.close()
        pool.join()

    # Assemble final matching_results.tsv and candidate_pairs.tsv in exact test_source1.tsv order
    merge_country_streams_to_final_outputs(
        total_s1=total_s1,
        countries=countries_order,
        chunk_counts=chunk_counts,
    )

    # Perform comprehensive read-only output audit
    audit_res = run_comprehensive_output_audit(expected_s1_ids, v1_pre_stat)

    # Clean up temporary per-chunk candidate files only AFTER audit passes
    if audit_res["status"] == "PASS":
        for p in CHUNKS_DIR.glob("cands_*.tsv"):
            try:
                p.unlink()
            except Exception:
                pass

    tot_wall_hr = (time.time() - t_run_start) / 3600.0
    print("\n" + "=" * 80, flush=True)
    print(f"FULL V3 TEST INFERENCE & OUTPUT AUDIT COMPLETE IN {tot_wall_hr:.2f} HOURS (STATUS={audit_res['status']})", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
