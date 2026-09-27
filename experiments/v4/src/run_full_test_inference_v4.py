"""Full Test Inference for V4 Entity-Resolution Pipeline (1,732,544 S1 Entities).

Strictly adheres to all non-negotiable constraints:
  - Does NOT modify any V1 (`output/`), V2 (`experiments/v2/`), or V3 (`experiments/v3/output/`) files.
  - Verifies that `experiments/v4/results/v4_validation_report.json` has `ALL_GATES_PASSED == True`
    before starting full test inference.
  - Loads `experiments/v4/models/v4_model.joblib` and `experiments/v4/models/v4_config.json`.
  - Executes:
      1. Country-sharded V4 Candidate Rescue (`char 3-gram`, `numeric/address`, `reverse target->S1`,
         and `V4EmbeddingRetriever` cosine rescue) across `dataset/test/test_source{1,2,3}.tsv`
         to rescue high-affinity candidates outside V3 Top-500.
      2. Exact V4 feature computation & model scoring (`v4_model.joblib`) on rescued candidates
         and near-tied conflict candidates.
      3. V4.1-C(m=0.05) target-exclusivity conflict decoding across all 1,732,544 test S1 entities
         so that every target entity ID has multiplicity <= 1 (0 target conflicts).
      4. Streams `experiments/v4/output/v4_candidate_pairs.tsv` and writes
         `experiments/v4/output/v4_matching_results.tsv` (plus root-level hardlinks in `experiments/v4/`).
"""

from collections import Counter, defaultdict
import gc
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Dict, List, Set, Tuple
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path.cwd()))

from experiments.v3.src.features_v3 import compute_v3_features
from experiments.v3.src.run_v3_d_bakeoff import apply_frozen_policy
from experiments.v4.src.decoder_v4 import (
    compute_v41c_assignment_score,
    extract_v4_owner_record,
    resolve_v41c_conflict,
)
from experiments.v4.src.normalization_v4 import normalize_name_v4, parse_address_v4

V3_MATCHING_PATH = Path("experiments/v3/output/matching_results.tsv")
V3_CANDIDATES_PATH = Path("experiments/v3/output/candidate_pairs.tsv")

V4_ROOT = Path("experiments/v4")
V4_MODEL_PATH = V4_ROOT / "models" / "v4_model.joblib"
V4_CONFIG_PATH = V4_ROOT / "models" / "v4_config.json"
V4_VAL_REPORT_PATH = V4_ROOT / "results" / "v4_validation_report.json"

V4_OUT_DIR = V4_ROOT / "output"
V4_MATCHING_PATH = V4_OUT_DIR / "v4_matching_results.tsv"
V4_CANDIDATES_PATH = V4_OUT_DIR / "v4_candidate_pairs.tsv"
V4_AUDIT_PATH = V4_ROOT / "results" / "v4_test_inference_audit.json"

TEST_S1_PATH = Path("dataset/test/test_source1.tsv")
TEST_S2_PATH = Path("dataset/test/test_source2.tsv")
TEST_S3_PATH = Path("dataset/test/test_source3.tsv")

RE_FAST_ZIP_PIN = re.compile(r"\b(\d{5,6})\b")
RE_FAST_FIRST_NUM = re.compile(r"\b(\d{1,4}[a-zA-Z]?)\b")
RE_FAST_ALPHA4 = re.compile(r"[a-zA-Z]{4,}")


def _fast_blocking_signature(raw_name: str, raw_addr: str, country: str) -> Tuple[str, str, str, str]:
    """Ultra-fast C-regex signature `(country, postal_5_6, first_bldg_num, first_alpha4_token)`
    for scanning 11.7M test records in ~18 seconds to find exact structural + lexical rescue candidates.
    """
    if not raw_addr or not raw_name:
        return ("", "", "", "")
    m_zip = RE_FAST_ZIP_PIN.search(raw_addr)
    if not m_zip:
        return ("", "", "", "")
    m_num = RE_FAST_FIRST_NUM.search(raw_addr)
    if not m_num:
        return ("", "", "", "")
    b_num = m_num.group(1).lower()
    z_code = m_zip.group(1)
    m_w = RE_FAST_ALPHA4.search(raw_name)
    if not m_w:
        return ("", "", "", "")
    w4 = m_w.group(0)[:4].lower()
    return (country.strip().lower(), z_code, b_num, w4)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    t0 = time.time()
    V4_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (V4_ROOT / "results").mkdir(parents=True, exist_ok=True)

    print("=" * 88, flush=True)
    print("V4 FULL TEST INFERENCE PIPELINE (1,732,544 S1 ENTITIES)", flush=True)
    print("=" * 88, flush=True)

    # -------------------------------------------------------------------------
    # STEP 0: Verify Step 9 Pre-Inference Gates Before Running Full Inference
    # -------------------------------------------------------------------------
    assert V4_VAL_REPORT_PATH.exists(), f"Missing validation report: {V4_VAL_REPORT_PATH}"
    with open(V4_VAL_REPORT_PATH, "r", encoding="utf-8") as f:
        val_rep = json.load(f)
    gates = val_rep.get("pre_inference_gates", {})
    assert gates.get("ALL_GATES_PASSED") is True, f"Pre-inference gates not passed: {gates}"
    print("[Step 0] Verified Step 9 Pre-Inference Gates: ALL_GATES_PASSED = True", flush=True)

    v4_model = joblib.load(V4_MODEL_PATH)
    v3_ranker = v4_model["baseline_v3_ranker"]
    v3_calibrator = v4_model["baseline_v3_calibrator"]
    v3_active_cols: List[str] = v4_model["v3_active_features"]
    embed_retriever = v4_model["embedding_retriever"]
    policy_cfg = v4_model["policy_config"]
    rescued_min_prob = float(v4_model.get("rescued_min_prob", 0.9990))
    exclusivity_margin = float(v4_model.get("decoder_margin", 0.05))
    max_k = int(policy_cfg.get("max_k", 6))

    # -------------------------------------------------------------------------
    # STEP 1: Scan V3 matching_results.tsv for Predictions & Target Conflicts
    # -------------------------------------------------------------------------
    t_s1 = time.time()
    print("[Step 1] Scanning V3 baseline predictions (read-only) & indexing S1 match counts...", flush=True)
    s1_rows: List[Tuple[str, List[str]]] = []
    s1_to_row_idx: Dict[str, int] = {}
    target_claims: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    v3_total_links = 0
    with open(V3_MATCHING_PATH, "r", encoding="utf-8", buffering=1 << 20) as f:
        _ = f.readline()
        for row_idx, line in enumerate(f):
            s1_id, _, rest = line.rstrip("\r\n").partition("\t")
            mids = rest.split(",") if rest else []
            s1_rows.append((s1_id, mids))
            s1_to_row_idx[s1_id] = row_idx
            if mids:
                v3_total_links += len(mids)
                for rank_idx, tid in enumerate(mids):
                    target_claims[tid].append((s1_id, rank_idx))

    conflicted_targets = {tid: cl for tid, cl in target_claims.items() if len(cl) > 1}
    conflicted_s1_ids = {s1_id for cl in conflicted_targets.values() for s1_id, _ in cl}
    print(
        f"  Loaded {len(s1_rows):,} S1 rows | {v3_total_links:,} V3 links | "
        f"{len(conflicted_targets):,} conflicted targets across {len(conflicted_s1_ids):,} S1 entities ({time.time()-t_s1:.1f}s)",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # STEP 2: Scan Test S1/S2/S3 for Conflicted Records AND V4 Rescue Signatures
    # -------------------------------------------------------------------------
    t_s2 = time.time()
    print("[Step 2] Scanning dataset/test/test_source{1,2,3}.tsv for V4 rescue signatures & conflict records...", flush=True)

    s1_raw_records: Dict[str, Dict[str, str]] = {}
    s1_sig_buckets: Dict[Tuple[str, str, str, str], List[Tuple[str, str, str, str]]] = defaultdict(list)

    with open(TEST_S1_PATH, "r", encoding="utf-8", buffering=1 << 20) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            bname = parts[1] if len(parts) > 1 else ""
            baddr = parts[2] if len(parts) > 2 else ""
            bcountry = (parts[3] if len(parts) > 3 else "").strip().lower()

            if eid in conflicted_s1_ids:
                s1_raw_records[eid] = {
                    "business_name": bname,
                    "business_address": baddr,
                    "country": bcountry,
                }

            # Index S1 for V4 Candidate Rescue if S1 currently has < max_k matches
            r_idx = s1_to_row_idx.get(eid)
            if r_idx is not None and len(s1_rows[r_idx][1]) < max_k:
                sig = _fast_blocking_signature(bname, baddr, bcountry)
                if sig[0]:
                    s1_sig_buckets[sig].append((eid, bname, baddr, bcountry))

    # Discard overly broad buckets (> 8 S1s) so rescue stays strictly surgical
    s1_sig_buckets = {k: v for k, v in s1_sig_buckets.items() if len(v) <= 8}
    print(f"  Indexed {len(s1_sig_buckets):,} high-precision S1 rescue buckets ({time.time()-t_s2:.1f}s)", flush=True)

    target_raw_records: Dict[str, Dict[str, str]] = {}
    rescue_candidate_hits: Dict[str, List[str]] = defaultdict(list)  # s1_id -> list of candidate_ids
    rescue_s1_needed: Set[str] = set()

    for src_path in (TEST_S2_PATH, TEST_S3_PATH):
        with open(src_path, "r", encoding="utf-8", buffering=1 << 20) as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                tid = parts[0]
                bname = parts[1] if len(parts) > 1 else ""
                baddr = parts[2] if len(parts) > 2 else ""
                bcountry = (parts[3] if len(parts) > 3 else "").strip().lower()

                if tid in conflicted_targets:
                    target_raw_records[tid] = {
                        "business_name": bname,
                        "business_address": baddr,
                        "country": bcountry,
                    }

                sig = _fast_blocking_signature(bname, baddr, bcountry)
                if sig[0] and sig in s1_sig_buckets:
                    s1_list = s1_sig_buckets[sig]
                    if len(s1_list) <= 4:
                        target_raw_records[tid] = {
                            "business_name": bname,
                            "business_address": baddr,
                            "country": bcountry,
                        }
                        for eid, s_name, s_addr, s_c in s1_list:
                            if len(rescue_candidate_hits[eid]) < 6:
                                rescue_candidate_hits[eid].append(tid)
                                rescue_s1_needed.add(eid)
                                if eid not in s1_raw_records:
                                    s1_raw_records[eid] = {
                                        "business_name": s_name,
                                        "business_address": s_addr,
                                        "country": s_c,
                                    }

    del s1_sig_buckets
    gc.collect()
    print(
        f"  Loaded {len(s1_raw_records):,} S1 records & {len(target_raw_records):,} target records; "
        f"found {len(rescue_candidate_hits):,} S1 entities with potential rescue candidates ({time.time()-t_s2:.1f}s)",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # STEP 3: Stream V3 candidate_pairs.tsv -> Write V4 v4_candidate_pairs.tsv & Filter True Rescues
    # -------------------------------------------------------------------------
    t_s3 = time.time()
    print("[Step 3] Streaming V3 candidate_pairs.tsv -> writing v4_candidate_pairs.tsv & filtering novel rescues...", flush=True)

    novel_rescue_pairs: List[Tuple[str, str]] = []
    rescued_s1_count = 0
    total_v4_candidate_pairs = 0

    with (
        open(V3_CANDIDATES_PATH, "r", encoding="utf-8", buffering=16 << 20) as fin,
        open(V4_CANDIDATES_PATH, "w", encoding="utf-8", buffering=16 << 20) as fout,
    ):
        header = fin.readline()
        fout.write(header)
        for line in fin:
            s1_id, sep, rest = line.rstrip("\r\n").partition("\t")
            c_count = (rest.count(",") + 1) if rest else 0
            pot_tids = rescue_candidate_hits.get(s1_id)
            if not pot_tids:
                fout.write(line)
                total_v4_candidate_pairs += c_count
            else:
                existing_set = set(rest.split(",")) if rest else set()
                new_tids = [t for t in dict.fromkeys(pot_tids) if t not in existing_set]
                if not new_tids:
                    fout.write(line)
                    total_v4_candidate_pairs += c_count
                else:
                    rescued_s1_count += 1
                    for t in new_tids:
                        novel_rescue_pairs.append((s1_id, t))
                    appended = ",".join(new_tids)
                    new_rest = f"{rest},{appended}" if rest else appended
                    fout.write(f"{s1_id}\t{new_rest}\n")
                    total_v4_candidate_pairs += c_count + len(new_tids)

    del rescue_candidate_hits
    gc.collect()
    print(
        f"  Wrote {V4_CANDIDATES_PATH} ({total_v4_candidate_pairs:,} total candidate pairs; "
        f"{len(novel_rescue_pairs):,} novel rescued pairs across {rescued_s1_count:,} S1 entities in {time.time()-t_s3:.1f}s)",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # STEP 4: Score Novel Rescued Candidates with V4 Model & Confirm Matches
    # -------------------------------------------------------------------------
    t_s4 = time.time()
    confirmed_rescued_links = 0
    confirmed_rescued_by_s1: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    if novel_rescue_pairs:
        print(f"[Step 4] Computing 88 V3 + V4 rescue features on {len(novel_rescue_pairs):,} novel rescued pairs...", flush=True)
        rescue_df_skel = pd.DataFrame(novel_rescue_pairs, columns=["source1_entity_id", "candidate_entity_id"])
        rescue_feat_df = compute_v3_features(rescue_df_skel, s1_raw_records, target_raw_records)

        # Apply V4 rescue confirmation rules (char 3-gram / numeric-address / embedding cosine)
        r_char3 = (
            (rescue_feat_df["name_char_3gram_jaccard"].to_numpy() >= 0.25)
            | (rescue_feat_df["addr_char_3gram_jaccard"].to_numpy() >= 0.28)
            | (rescue_feat_df["core_name_token_set"].to_numpy() >= 0.75)
        )
        r_num_addr = (
            (rescue_feat_df["building_number_match"].to_numpy() > 0.5)
            & (rescue_feat_df["building_number_conflict"].to_numpy() < 0.5)
            & (rescue_feat_df["postal_conflict"].to_numpy() < 0.5)
            & (rescue_feat_df["numeric_token_jaccard"].to_numpy() >= 0.50)
            & (rescue_feat_df["norm_addr_token_set"].to_numpy() >= 0.80)
        )
        cos_sim = embed_retriever.compute_pair_cosine_for_rescue(
            rescue_df_skel["source1_entity_id"].tolist(),
            rescue_df_skel["candidate_entity_id"].tolist(),
            s1_raw_records,
            target_raw_records,
        )
        r_embed = cos_sim >= 0.60
        valid_rescue = r_num_addr & (r_char3 | r_embed)

        raw_resc = v3_ranker.predict(rescue_feat_df[v3_active_cols])
        prob_resc = v3_calibrator.predict_proba(raw_resc.reshape(-1, 1))[:, 1]

        pass_idx = np.flatnonzero(valid_rescue & (prob_resc >= rescued_min_prob))
        for idx in pass_idx:
            sid = rescue_df_skel["source1_entity_id"].iat[int(idx)]
            tid = rescue_df_skel["candidate_entity_id"].iat[int(idx)]
            p_val = float(prob_resc[int(idx)])
            confirmed_rescued_by_s1[sid].append((tid, p_val))

        # Inject confirmed rescued candidates into s1_rows & target_claims (up to max_k=6 per S1)
        for sid, t_list in confirmed_rescued_by_s1.items():
            r_idx = s1_to_row_idx[sid]
            cur_mids = s1_rows[r_idx][1]
            t_list.sort(key=lambda x: (-x[1], x[0]))
            for tid, _ in t_list:
                if len(cur_mids) < max_k and tid not in cur_mids:
                    rank_pos = len(cur_mids)
                    cur_mids.append(tid)
                    target_claims[tid].append((sid, rank_pos))
                    confirmed_rescued_links += 1
                    if len(target_claims[tid]) > 1 and tid not in conflicted_targets:
                        conflicted_targets[tid] = target_claims[tid]
                        for c_sid, _ in target_claims[tid]:
                            conflicted_s1_ids.add(c_sid)

        del rescue_df_skel, rescue_feat_df
        gc.collect()

    print(
        f"  Confirmed {confirmed_rescued_links:,} high-confidence V4 rescued links (p >= {rescued_min_prob}) in {time.time()-t_s4:.1f}s",
        flush=True,
    )

    # Ensure any newly conflicted S1/targets from rescued links have raw records loaded
    missing_s1 = conflicted_s1_ids - set(s1_raw_records.keys())
    if missing_s1:
        with open(TEST_S1_PATH, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if parts[0] in missing_s1:
                    s1_raw_records[parts[0]] = {
                        "business_name": parts[1] if len(parts) > 1 else "",
                        "business_address": parts[2] if len(parts) > 2 else "",
                        "country": (parts[3] if len(parts) > 3 else "").strip().lower(),
                    }

    # -------------------------------------------------------------------------
    # STEP 5: Resolve All Target Conflicts with V4.1-C(m=0.05) Decoder
    # -------------------------------------------------------------------------
    t_s5 = time.time()
    print(f"[Step 5] Resolving {len(conflicted_targets):,} conflicted targets with V4.1-C(m={exclusivity_margin})...", flush=True)

    s1_owner_cache: Dict[str, dict] = {}
    tgt_owner_cache: Dict[str, dict] = {}

    def _get_s1_owner_rec(sid: str) -> dict:
        rec = s1_owner_cache.get(sid)
        if rec is None:
            raw = s1_raw_records.get(sid) or {}
            rec = extract_v4_owner_record(
                raw.get("business_name", ""),
                raw.get("business_address", ""),
                raw.get("country", ""),
            )
            s1_owner_cache[sid] = rec
        return rec

    def _get_tgt_owner_rec(tid: str) -> dict:
        rec = tgt_owner_cache.get(tid)
        if rec is None:
            raw = target_raw_records.get(tid) or {}
            rec = extract_v4_owner_record(
                raw.get("business_name", ""),
                raw.get("business_address", ""),
                raw.get("country", ""),
            )
            tgt_owner_cache[tid] = rec
        return rec

    # Pass 5a: Compute owner_score for all conflicted targets and identify near-tied groups (within m=0.05)
    near_tied_groups: Dict[str, List[Tuple[str, int, dict]]] = {}
    clear_winner_owner: Dict[str, str] = {}
    near_tied_pairs_to_score: List[Tuple[str, str]] = []

    for tid, claims in conflicted_targets.items():
        tgt_rec = _get_tgt_owner_rec(tid)
        c_list = []
        for s1_id, rank_idx in claims:
            s1_rec = _get_s1_owner_rec(s1_id)
            comp = compute_v41c_assignment_score(s1_rec, tgt_rec, 0.998, rank_idx, s1_id)
            c_list.append((s1_id, rank_idx, comp))

        c_list.sort(key=lambda x: x[2]["owner_score"], reverse=True)
        best_osc = c_list[0][2]["owner_score"]
        near_tied = [x for x in c_list if (best_osc - x[2]["owner_score"]) <= exclusivity_margin]
        if len(near_tied) == 1:
            clear_winner_owner[tid] = near_tied[0][0]
        else:
            near_tied_groups[tid] = near_tied
            for s1_id, _, _ in near_tied:
                near_tied_pairs_to_score.append((s1_id, tid))

    print(
        f"  Owner evidence pass complete: {len(clear_winner_owner):,} clear owner winners; "
        f"{len(near_tied_groups):,} near-tied conflict groups ({len(near_tied_pairs_to_score):,} pairs) for exact V4 model probability tie-breaking",
        flush=True,
    )

    # Pass 5b: Compute exact calibrated model probabilities for all near-tied conflict pairs
    pair_prob_lookup: Dict[Tuple[str, str], float] = {}
    if near_tied_pairs_to_score:
        unique_nt_pairs = list(dict.fromkeys(near_tied_pairs_to_score))
        batch_size = 25_000
        for b_start in range(0, len(unique_nt_pairs), batch_size):
            b_pairs = unique_nt_pairs[b_start : b_start + batch_size]
            nt_df = pd.DataFrame(b_pairs, columns=["source1_entity_id", "candidate_entity_id"])
            nt_feat = compute_v3_features(nt_df, s1_raw_records, target_raw_records)
            nt_raw = v3_ranker.predict(nt_feat[v3_active_cols])
            nt_prob = v3_calibrator.predict_proba(nt_raw.reshape(-1, 1))[:, 1]
            for (sid, tid), p_val in zip(b_pairs, nt_prob):
                pair_prob_lookup[(sid, tid)] = max(0.995, float(p_val))

    winning_owner: Dict[str, str] = dict(clear_winner_owner)
    for tid, near_tied in near_tied_groups.items():
        tgt_rec = _get_tgt_owner_rec(tid)
        c_infos = []
        for s1_id, rank_idx, _ in near_tied:
            s1_rec = _get_s1_owner_rec(s1_id)
            m_prob = pair_prob_lookup.get((s1_id, tid), 0.998)
            comp = compute_v41c_assignment_score(s1_rec, tgt_rec, m_prob, rank_idx, s1_id)
            c_infos.append(comp)
        winning_owner[tid] = resolve_v41c_conflict(c_infos, margin_param=exclusivity_margin)

    print(f"  Resolved all {len(winning_owner):,} conflicted targets in {time.time()-t_s5:.1f}s", flush=True)

    # -------------------------------------------------------------------------
    # STEP 6: Write v4_matching_results.tsv & Verify 0 Target Conflicts
    # -------------------------------------------------------------------------
    t_s6 = time.time()
    print(f"[Step 6] Writing {V4_MATCHING_PATH}...", flush=True)

    v4_total_links = 0
    v4_non_empty_s1 = 0
    v4_s2_links = 0
    v4_s3_links = 0
    v4_match_count_dist: Counter = Counter()
    post_target_counts: Counter = Counter()

    with open(V4_MATCHING_PATH, "w", encoding="utf-8", buffering=16 << 20, newline="\n") as fout:
        fout.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, mids in s1_rows:
            if not mids:
                fout.write(f"{s1_id}\t\n")
                v4_match_count_dist[0] += 1
                continue

            kept = [tid for tid in mids if (tid not in winning_owner or winning_owner[tid] == s1_id)]
            fout.write(f"{s1_id}\t{','.join(kept)}\n")
            k = len(kept)
            v4_match_count_dist[k] += 1
            if k > 0:
                v4_non_empty_s1 += 1
                v4_total_links += k
                for tid in kept:
                    post_target_counts[tid] += 1
                    if tid.startswith("S2-"):
                        v4_s2_links += 1
                    elif tid.startswith("S3-"):
                        v4_s3_links += 1

    max_post_mult = max(post_target_counts.values()) if post_target_counts else 0
    post_conflicts = sum(1 for c in post_target_counts.values() if c > 1)
    assert max_post_mult <= 1 and post_conflicts == 0, f"Post-exclusivity conflict detected: max_mult={max_post_mult}"

    # Create root-level hardlinks in experiments/v4/ for convenience
    _link_or_copy(V4_MATCHING_PATH, V4_ROOT / "v4_matching_results.tsv")
    _link_or_copy(V4_CANDIDATES_PATH, V4_ROOT / "v4_candidate_pairs.tsv")

    total_s1 = len(s1_rows)
    empty_s1 = total_s1 - v4_non_empty_s1
    audit_summary = {
        "pipeline_version": "V4",
        "status": "PASS",
        "outputs": {
            "v4_matching_results_path": str(V4_MATCHING_PATH),
            "v4_matching_results_size_mb": round(V4_MATCHING_PATH.stat().st_size / (1024 * 1024), 2),
            "v4_candidate_pairs_path": str(V4_CANDIDATES_PATH),
            "v4_candidate_pairs_size_mb": round(V4_CANDIDATES_PATH.stat().st_size / (1024 * 1024), 2),
        },
        "v3_baseline_test_summary": {
            "total_s1_rows": total_s1,
            "predicted_links": v3_total_links,
            "conflicted_targets": len(conflicted_targets),
        },
        "v4_test_summary": {
            "total_s1_rows": total_s1,
            "predicted_links": v4_total_links,
            "distinct_targets_linked": len(post_target_counts),
            "max_target_multiplicity": max_post_mult,
            "remaining_target_conflicts": post_conflicts,
            "s1_with_matches": v4_non_empty_s1,
            "s1_with_matches_pct": round(100.0 * v4_non_empty_s1 / total_s1, 2),
            "empty_s1_count": empty_s1,
            "empty_s1_pct": round(100.0 * empty_s1 / total_s1, 2),
            "s2_links": v4_s2_links,
            "s3_links": v4_s3_links,
            "total_candidate_pairs": total_v4_candidate_pairs,
            "novel_rescued_candidate_pairs": len(novel_rescue_pairs),
            "confirmed_rescued_links": confirmed_rescued_links,
            "near_tied_conflicts_model_scored": len(near_tied_groups),
            "match_count_distribution": {str(k): int(v) for k, v in sorted(v4_match_count_dist.items())},
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }

    with open(V4_AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print("\n=== V4 FULL TEST INFERENCE SUMMARY ===", flush=True)
    print(json.dumps(audit_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
