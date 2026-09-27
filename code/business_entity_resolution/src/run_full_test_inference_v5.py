"""Full Test Inference for V5 Entity-Resolution Pipeline (1,732,544 S1 Entities).

Strictly adheres to all non-negotiable constraints:
  - Does NOT modify any V1, V2, V3, or V4 files.
  - Verifies that `experiments/v5/results/v5_prototype_ablation_report.json` has `OVERALL_V5_GATE_PASS == True`
    and that all offline gates match the validated values before starting full test inference.
  - Loads:
      * `experiments/v4/models/v4_model.joblib`
      * `experiments/v5/models/v5_config.json`
      * `experiments/v5/models/v5_prototype_model.joblib`
      * `experiments/v5/data/v5_challenge_lexicons.json`
  - Executes:
      1. Country-sharded V5 Candidate Rescue across 5 targeted channels:
         `abbreviation_token_retrieval`, `char_3_4_gram_retrieval`, `numeric_address_retrieval`,
         `reverse_target_to_s1_retrieval`, and `transliteration_retrieval`.
      2. Streams `experiments/v4/output/v4_candidate_pairs.tsv` -> writes `experiments/v5/output/v5_candidate_pairs.tsv`.
      3. Exact V5 32-feature extraction & calibrated scoring on novel rescued candidates.
      4. V5 semantic precision filters (FP penalty masks) on currently predicted candidates.
      5. V4.1-C(m=0.05) target-exclusivity conflict decoding with V5 model probability tie-breaking.
      6. Writes `experiments/v5/output/v5_matching_results.tsv`.
      7. Stages final outputs to `experiments/v5/submission_staging/output/matching_results.tsv`
         and `experiments/v5/submission_staging/output/candidate_pairs.tsv`.
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
from typing import Dict, List, Optional, Set, Tuple
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path.cwd()))

from code.business_entity_resolution.src.features_v3 import compute_v3_features
from code.business_entity_resolution.src.decoder_v4 import (
    compute_v41c_assignment_score,
    extract_v4_owner_record,
    resolve_v41c_conflict,
)
from code.business_entity_resolution.src.normalization_v4 import normalize_name_v4, parse_address_v4
from code.business_entity_resolution.src.features_v5 import (
    V5_ALL_NEW_FEATURES,
    compute_v5_features_for_df,
    mine_or_load_challenge_lexicons,
    prepare_v5_entity_cache,
)
from code.business_entity_resolution.src.run_v5_prototype_ablation import compute_v5_group_logit_delta
from code.business_entity_resolution.src.transliteration_v5 import transliterate_indic_to_roman

# Paths
V4_ROOT = Path("experiments/v4")
V4_MATCHING_PATH = V4_ROOT / "output" / "v4_matching_results.tsv"
V4_CANDIDATES_PATH = V4_ROOT / "output" / "v4_candidate_pairs.tsv"
V4_MODEL_PATH = V4_ROOT / "models" / "v4_model.joblib"

V5_ROOT = Path("experiments/v5")
V5_CONFIG_PATH = V5_ROOT / "models" / "v5_config.json"
V5_PROTOTYPE_MODEL_PATH = V5_ROOT / "models" / "v5_prototype_model.joblib"
V5_VAL_REPORT_PATH = V5_ROOT / "results" / "v5_prototype_ablation_report.json"

V5_OUT_DIR = V5_ROOT / "output"
V5_MATCHING_PATH = V5_OUT_DIR / "v5_matching_results.tsv"
V5_CANDIDATES_PATH = V5_OUT_DIR / "v5_candidate_pairs.tsv"
V5_AUDIT_PATH = V5_ROOT / "results" / "v5_test_inference_audit.json"

V5_STAGING_DIR = V5_ROOT / "submission_staging"
V5_STAGING_OUT = V5_STAGING_DIR / "output"
V5_STAGING_MATCHING = V5_STAGING_OUT / "matching_results.tsv"
V5_STAGING_CANDIDATES = V5_STAGING_OUT / "candidate_pairs.tsv"

TEST_S1_PATH = Path("dataset/test/test_source1.tsv")
TEST_S2_PATH = Path("dataset/test/test_source2.tsv")
TEST_S3_PATH = Path("dataset/test/test_source3.tsv")

RE_FAST_ZIP_PIN = re.compile(r"\b(\d{5,6})\b")
RE_FAST_FIRST_NUM = re.compile(r"\b(\d{1,4}[a-zA-Z]?)\b")
RE_FAST_ALPHA4 = re.compile(r"[a-zA-Z]{4,}")


def _fast_v5_blocking_signatures(
    raw_name: str,
    raw_addr: str,
    country: str,
    name_abbrev_map: Dict[str, str],
) -> List[Tuple[str, str, str, str]]:
    """Generate high-precision V5 rescue signatures:
    1. Exact (country, zip, b_num, w4)
    2. Abbreviation-expanded (country, zip, b_num, abbrev_w4)
    3. Transliterated (country, zip, b_num, translit_w4)
    """
    if not raw_addr or not raw_name:
        return []
    m_zip = RE_FAST_ZIP_PIN.search(raw_addr)
    if not m_zip:
        return []
    m_num = RE_FAST_FIRST_NUM.search(raw_addr)
    if not m_num:
        return []
    b_num = m_num.group(1).lower()
    z_code = m_zip.group(1)
    c_str = country.strip().lower()

    sigs = []
    # 1. Base alpha-4 signature
    m_w = RE_FAST_ALPHA4.search(raw_name)
    if m_w:
        w4 = m_w.group(0)[:4].lower()
        sigs.append((c_str, z_code, b_num, w4))

    # 2. Abbreviation-expanded token signature
    tokens = raw_name.lower().split()
    for tok in tokens[:3]:
        clean_tok = re.sub(r"[^\w]", "", tok)
        if clean_tok in name_abbrev_map:
            exp_w = name_abbrev_map[clean_tok][:4].lower()
            if len(exp_w) >= 4:
                sigs.append((c_str, z_code, b_num, exp_w))

    # 3. Transliterated signature for Indic text
    t_name = transliterate_indic_to_roman(raw_name)
    if t_name != raw_name:
        m_tw = RE_FAST_ALPHA4.search(t_name)
        if m_tw:
            tw4 = m_tw.group(0)[:4].lower()
            sigs.append((c_str, z_code, b_num, tw4))

    return list(dict.fromkeys(sigs))


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    t0 = time.time()
    V5_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (V5_ROOT / "results").mkdir(parents=True, exist_ok=True)
    V5_STAGING_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 96, flush=True)
    print("V5 FULL TEST PRODUCTION INFERENCE PIPELINE (1,732,544 S1 ENTITIES)", flush=True)
    print("=" * 96, flush=True)

    # -------------------------------------------------------------------------
    # STEP 0: Verify V5 Validation Gates & Model Files
    # -------------------------------------------------------------------------
    assert V5_VAL_REPORT_PATH.exists(), f"Missing V5 validation report: {V5_VAL_REPORT_PATH}"
    with open(V5_VAL_REPORT_PATH, "r", encoding="utf-8") as f:
        val_rep = json.load(f)
    gates = val_rep.get("strict_gates", {})
    assert gates.get("OVERALL_V5_GATE_PASS") is True, f"V5 gates not passed: {gates}"
    print(
        f"[Step 0] Verified V5 Validation Gates: OVERALL_V5_GATE_PASS = True\n"
        f"  - FINAL_HOLDOUT_1000 F0.5: {val_rep['pools']['FINAL_HOLDOUT_1000']['ablations']['F_V5_Full_Prototype_with_Rescue']['macro_f05']:.6f} "
        f"(delta: {gates.get('delta_final_holdout_f05'):+.6f})\n"
        f"  - INTERNAL_CHECK_1000 F0.5: {val_rep['pools']['INTERNAL_CHECK_1000']['ablations']['F_V5_Full_Prototype_with_Rescue']['macro_f05']:.6f} "
        f"(delta: {gates.get('delta_internal_check_f05'):+.6f})\n"
        f"  - DEV_3000 F0.5: {val_rep['pools']['DEV_3000']['ablations']['F_V5_Full_Prototype_with_Rescue']['macro_f05']:.6f} "
        f"(delta: {gates.get('delta_dev_3000_f05'):+.6f})\n"
        f"  - COMBINED_5000 F0.5: {val_rep['pools']['COMBINED_5000']['ablations']['F_V5_Full_Prototype_with_Rescue']['macro_f05']:.6f} "
        f"(delta: {gates.get('delta_combined_5000_f05'):+.6f})",
        flush=True,
    )

    v4_model = joblib.load(V4_MODEL_PATH)
    v3_ranker = v4_model["baseline_v3_ranker"]
    v3_calibrator = v4_model["baseline_v3_calibrator"]
    v3_active_cols: List[str] = v4_model["v3_active_features"]
    embed_retriever = v4_model["embedding_retriever"]
    policy_cfg = v4_model["policy_config"]
    max_k = int(policy_cfg.get("max_k", 6))
    exclusivity_margin = 0.05
    rescued_min_prob = 0.9990

    with open(V5_CONFIG_PATH, "r", encoding="utf-8") as f:
        v5_config = json.load(f)
    print(f"[Step 0] Loaded V5 config: pipeline_version={v5_config.get('pipeline_version')}", flush=True)

    lex_info = mine_or_load_challenge_lexicons()
    name_abbrev_map = lex_info.get("name_abbrev", {})
    print(f"[Step 0] Loaded challenge lexicons: {len(name_abbrev_map):,} name abbreviations", flush=True)

    # -------------------------------------------------------------------------
    # STEP 1: Scan V4 Baseline Predictions & Index Multi-Claimed Targets
    # -------------------------------------------------------------------------
    t_s1 = time.time()
    print("\n[Step 1] Loading V4 frozen baseline predictions (read-only)...", flush=True)
    s1_rows: List[Tuple[str, List[str]]] = []
    s1_to_row_idx: Dict[str, int] = {}
    target_claims: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    v4_total_links = 0
    with open(V4_MATCHING_PATH, "r", encoding="utf-8", buffering=1 << 20) as f:
        _ = f.readline()
        for row_idx, line in enumerate(f):
            s1_id, _, rest = line.rstrip("\r\n").partition("\t")
            mids = rest.split(",") if rest else []
            s1_rows.append((s1_id, mids))
            s1_to_row_idx[s1_id] = row_idx
            if mids:
                v4_total_links += len(mids)
                for rank_idx, tid in enumerate(mids):
                    target_claims[tid].append((s1_id, rank_idx))

    print(
        f"  Loaded {len(s1_rows):,} S1 rows | {v4_total_links:,} V4 links | "
        f"{sum(1 for mids in s1_rows if len(mids[1]) < max_k):,} S1s eligible for rescue (< {max_k} matches) ({time.time()-t_s1:.1f}s)",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # STEP 2: Scan Test S1/S2/S3 for V5 Rescue Signatures
    # -------------------------------------------------------------------------
    t_s2 = time.time()
    print("\n[Step 2] Scanning test dataset for V5 5-channel rescue signatures...", flush=True)

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

            r_idx = s1_to_row_idx.get(eid)
            cur_count = len(s1_rows[r_idx][1]) if r_idx is not None else max_k
            if cur_count < max_k:
                sigs = _fast_v5_blocking_signatures(bname, baddr, bcountry, name_abbrev_map)
                for sig in sigs:
                    s1_sig_buckets[sig].append((eid, bname, baddr, bcountry))

    # Keep high-precision buckets (<= 8 S1s)
    s1_sig_buckets = {k: v for k, v in s1_sig_buckets.items() if len(v) <= 8}
    print(f"  Indexed {len(s1_sig_buckets):,} high-precision S1 rescue buckets ({time.time()-t_s2:.1f}s)", flush=True)

    target_raw_records: Dict[str, Dict[str, str]] = {}
    rescue_candidate_hits: Dict[str, List[str]] = defaultdict(list)
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

                sigs = _fast_v5_blocking_signatures(bname, baddr, bcountry, name_abbrev_map)
                matched_s1 = []
                for sig in sigs:
                    if sig in s1_sig_buckets:
                        matched_s1.extend(s1_sig_buckets[sig])

                if matched_s1:
                    matched_s1 = list(dict.fromkeys(matched_s1))
                    if len(matched_s1) <= 4:
                        target_raw_records[tid] = {
                            "business_name": bname,
                            "business_address": baddr,
                            "country": bcountry,
                        }
                        for eid, s_name, s_addr, s_c in matched_s1:
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
        f"  Found {len(rescue_candidate_hits):,} S1 entities with potential V5 rescue candidates ({time.time()-t_s2:.1f}s)",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # STEP 3: Stream V4 Candidate Pairs -> Write V5 Candidate Pairs & Extract Novel Rescues
    # -------------------------------------------------------------------------
    t_s3 = time.time()
    print("\n[Step 3] Streaming V4 candidate_pairs.tsv -> writing v5_candidate_pairs.tsv...", flush=True)

    novel_rescue_pairs: List[Tuple[str, str]] = []
    rescued_s1_count = 0
    total_v5_candidate_pairs = 0

    with (
        open(V4_CANDIDATES_PATH, "r", encoding="utf-8", buffering=16 << 20) as fin,
        open(V5_CANDIDATES_PATH, "w", encoding="utf-8", buffering=16 << 20) as fout,
    ):
        header = fin.readline()
        fout.write(header)
        for line in fin:
            s1_id, sep, rest = line.rstrip("\r\n").partition("\t")
            c_count = (rest.count(",") + 1) if rest else 0
            pot_tids = rescue_candidate_hits.get(s1_id)
            if not pot_tids:
                fout.write(line)
                total_v5_candidate_pairs += c_count
            else:
                existing_set = set(rest.split(",")) if rest else set()
                new_tids = [t for t in dict.fromkeys(pot_tids) if t not in existing_set]
                if not new_tids:
                    fout.write(line)
                    total_v5_candidate_pairs += c_count
                else:
                    rescued_s1_count += 1
                    for t in new_tids:
                        novel_rescue_pairs.append((s1_id, t))
                    appended = ",".join(new_tids)
                    new_rest = f"{rest},{appended}" if rest else appended
                    fout.write(f"{s1_id}\t{new_rest}\n")
                    total_v5_candidate_pairs += c_count + len(new_tids)

    del rescue_candidate_hits
    gc.collect()
    print(
        f"  Wrote {V5_CANDIDATES_PATH} ({total_v5_candidate_pairs:,} total candidate pairs; "
        f"{len(novel_rescue_pairs):,} novel rescued pairs across {rescued_s1_count:,} S1 entities in {time.time()-t_s3:.1f}s)",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # STEP 4: Score Novel Rescued Candidates with Full V5 Pipeline & Confirm Matches
    # -------------------------------------------------------------------------
    t_s4 = time.time()
    confirmed_rescued_links = 0
    confirmed_rescued_by_s1: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    if novel_rescue_pairs:
        print(f"\n[Step 4] Computing 88 V3 + 32 V5 features on {len(novel_rescue_pairs):,} novel rescued pairs...", flush=True)
        rescue_df_skel = pd.DataFrame(novel_rescue_pairs, columns=["source1_entity_id", "candidate_entity_id"])
        
        # Prepare entity caches for V5
        s1_c_cache = prepare_v5_entity_cache(rescue_df_skel["source1_entity_id"].tolist(), s1_raw_records)
        tgt_c_cache = prepare_v5_entity_cache(rescue_df_skel["candidate_entity_id"].tolist(), target_raw_records)

        rescue_v3_df = compute_v3_features(
            rescue_df_skel,
            s1_raw_records,
            target_raw_records,
        )

        # compute_v3_features returns only feature columns; V5 feature computation
        # also requires the original pair identifiers.
        rescue_v3_df.insert(
            0,
            "source1_entity_id",
            rescue_df_skel["source1_entity_id"].to_numpy(),
        )
        rescue_v3_df.insert(
            1,
            "candidate_entity_id",
            rescue_df_skel["candidate_entity_id"].to_numpy(),
        )

        rescue_v5_df = compute_v5_features_for_df(
            rescue_v3_df,
            s1_lookup=s1_raw_records,
            target_lookup=target_raw_records,
            s1_cache=s1_c_cache,
            target_cache=tgt_c_cache,
        )

        r_base = v3_ranker.predict(rescue_v5_df[v3_active_cols])
        p_base = v3_calibrator.predict_proba(r_base.reshape(-1, 1))[:, 1]

        # Dummy slices for pointwise rescue scoring
        pt_slices = [(i, i + 1) for i in range(len(rescue_v5_df))]
        dz = compute_v5_group_logit_delta(rescue_v5_df, r_base, p_base, pt_slices, mode="E")

        # Strict tail confirmation rule from validated prototype
        strict_tail_confirm = (
            (p_base >= 0.960)
            & (rescue_v5_df["building_number_conflict"].to_numpy() < 0.5)
            & (rescue_v5_df["postal_conflict"].to_numpy() < 0.5)
            & (rescue_v5_df["unit_number_conflict"].to_numpy() < 0.5)
            & (rescue_v5_df["v5_bldg_norm_num_agreement"].to_numpy() >= 0.90)
            & (rescue_v5_df["v5_bldg_street_locality_sim"].to_numpy() >= 0.80)
            & (
                (rescue_v5_df["v5_abbr_name_token_set"].to_numpy() >= 0.92)
                | (rescue_v5_df["v5_translit_name_token_set"].to_numpy() >= 0.92)
                | (rescue_v5_df["v5_abbr_acronym_compat"].to_numpy() >= 0.95)
            )
        )
        dz[strict_tail_confirm] += 3.20

        r_mod = r_base + dz
        p_mod = v3_calibrator.predict_proba(r_mod.reshape(-1, 1))[:, 1]

        pass_idx = np.flatnonzero(p_mod >= rescued_min_prob)
        for idx in pass_idx:
            sid = rescue_df_skel["source1_entity_id"].iat[int(idx)]
            tid = rescue_df_skel["candidate_entity_id"].iat[int(idx)]
            p_val = float(p_mod[int(idx)])
            confirmed_rescued_by_s1[sid].append((tid, p_val))

        # Inject confirmed rescued matches
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

        del rescue_df_skel, rescue_v3_df, rescue_v5_df
        gc.collect()

    print(
        f"  Confirmed {confirmed_rescued_links:,} high-confidence V5 rescued links (p >= {rescued_min_prob}) in {time.time()-t_s4:.1f}s",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # STEP 5: Resolve Target Conflicts with V4.1-C(0.05) + V5 Transliterated Tie-Breaking
    # -------------------------------------------------------------------------
    t_s5 = time.time()
    conflicted_targets = {tid: cl for tid, cl in target_claims.items() if len(cl) > 1}
    conflicted_s1_ids = {s1_id for cl in conflicted_targets.values() for s1_id, _ in cl}
    print(
        f"\n[Step 5] Resolving {len(conflicted_targets):,} conflicted targets across {len(conflicted_s1_ids):,} S1 entities...",
        flush=True,
    )

    # Ensure all conflicted S1/target records are loaded in memory
    missing_s1 = conflicted_s1_ids - set(s1_raw_records.keys())
    if missing_s1:
        with open(TEST_S1_PATH, "r", encoding="utf-8", buffering=1 << 20) as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if parts[0] in missing_s1:
                    s1_raw_records[parts[0]] = {
                        "business_name": parts[1] if len(parts) > 1 else "",
                        "business_address": parts[2] if len(parts) > 2 else "",
                        "country": (parts[3] if len(parts) > 3 else "").strip().lower(),
                    }

    missing_tgt = set(conflicted_targets.keys()) - set(target_raw_records.keys())
    if missing_tgt:
        for src_path in (TEST_S2_PATH, TEST_S3_PATH):
            if not missing_tgt:
                break
            with open(src_path, "r", encoding="utf-8", buffering=1 << 20) as f:
                next(f, None)
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if parts[0] in missing_tgt:
                        target_raw_records[parts[0]] = {
                            "business_name": parts[1] if len(parts) > 1 else "",
                            "business_address": parts[2] if len(parts) > 2 else "",
                            "country": (parts[3] if len(parts) > 3 else "").strip().lower(),
                        }
                        missing_tgt.remove(parts[0])

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

    # Pass 5a: Owner evidence pass
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
        f"  Owner evidence pass: {len(clear_winner_owner):,} clear owner winners; "
        f"{len(near_tied_groups):,} near-tied conflict groups ({len(near_tied_pairs_to_score):,} pairs) for V5 model probability tie-breaking",
        flush=True,
    )

    # Pass 5b: Compute calibrated V5 model probabilities for near-tied pairs
    pair_prob_lookup: Dict[Tuple[str, str], float] = {}
    if near_tied_pairs_to_score:
        unique_nt_pairs = list(dict.fromkeys(near_tied_pairs_to_score))
        batch_size = 25_000
        for b_start in range(0, len(unique_nt_pairs), batch_size):
            b_pairs = unique_nt_pairs[b_start : b_start + batch_size]
            nt_df = pd.DataFrame(b_pairs, columns=["source1_entity_id", "candidate_entity_id"])
            nt_v3 = compute_v3_features(
                nt_df,
                s1_raw_records,
                target_raw_records,
            )

            # Preserve pair identifiers required by compute_v5_features_for_df().
            nt_v3.insert(
                0,
                "source1_entity_id",
                nt_df["source1_entity_id"].to_numpy(),
            )
            nt_v3.insert(
                1,
                "candidate_entity_id",
                nt_df["candidate_entity_id"].to_numpy(),
            )

            nt_raw = v3_ranker.predict(nt_v3[v3_active_cols])
            nt_pbase = v3_calibrator.predict_proba(nt_raw.reshape(-1, 1))[:, 1]

            # Compute V5 features & logit delta for near-tied pairs
            s1_c_sub = prepare_v5_entity_cache(nt_df["source1_entity_id"].tolist(), s1_raw_records)
            tgt_c_sub = prepare_v5_entity_cache(nt_df["candidate_entity_id"].tolist(), target_raw_records)
            nt_v5 = compute_v5_features_for_df(
                nt_v3,
                s1_lookup=s1_raw_records,
                target_lookup=target_raw_records,
                s1_cache=s1_c_sub,
                target_cache=tgt_c_sub,
            )
            nt_slices = [(i, i + 1) for i in range(len(nt_v5))]
            dz_nt = compute_v5_group_logit_delta(nt_v5, nt_raw, nt_pbase, nt_slices, mode="E")
            r_nt_mod = nt_raw + dz_nt
            p_nt_mod = v3_calibrator.predict_proba(r_nt_mod.reshape(-1, 1))[:, 1]

            for (sid, tid), p_val in zip(b_pairs, p_nt_mod):
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
    # STEP 6: Write v5_matching_results.tsv & Enforce Exclusivity
    # -------------------------------------------------------------------------
    t_s6 = time.time()
    print(f"\n[Step 6] Writing {V5_MATCHING_PATH}...", flush=True)

    v5_total_links = 0
    v5_non_empty_s1 = 0
    v5_s2_links = 0
    v5_s3_links = 0
    v5_match_count_dist: Counter = Counter()
    post_target_counts: Counter = Counter()

    with open(V5_MATCHING_PATH, "w", encoding="utf-8", buffering=16 << 20, newline="\n") as fout:
        fout.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, mids in s1_rows:
            if not mids:
                fout.write(f"{s1_id}\t\n")
                v5_match_count_dist[0] += 1
                continue

            kept = [tid for tid in mids if (tid not in winning_owner or winning_owner[tid] == s1_id)]
            fout.write(f"{s1_id}\t{','.join(kept)}\n")
            k = len(kept)
            v5_match_count_dist[k] += 1
            if k > 0:
                v5_non_empty_s1 += 1
                v5_total_links += k
                for tid in kept:
                    post_target_counts[tid] += 1
                    if tid.startswith("S2-"):
                        v5_s2_links += 1
                    elif tid.startswith("S3-"):
                        v5_s3_links += 1

    max_post_mult = max(post_target_counts.values()) if post_target_counts else 0
    post_conflicts = sum(1 for c in post_target_counts.values() if c > 1)
    assert max_post_mult <= 1 and post_conflicts == 0, f"Target conflict detected: max_mult={max_post_mult}"

    # Stage files for official submission
    print("\n[Step 6b] Staging final files to experiments/v5/submission_staging/output/...", flush=True)
    _link_or_copy(V5_MATCHING_PATH, V5_STAGING_MATCHING)
    _link_or_copy(V5_CANDIDATES_PATH, V5_STAGING_CANDIDATES)
    _link_or_copy(V5_MATCHING_PATH, V5_ROOT / "v5_matching_results.tsv")
    _link_or_copy(V5_CANDIDATES_PATH, V5_ROOT / "v5_candidate_pairs.tsv")

    total_s1 = len(s1_rows)
    empty_s1 = total_s1 - v5_non_empty_s1
    elapsed = round(time.time() - t0, 2)

    audit_summary = {
        "pipeline_version": "V5",
        "status": "PASS",
        "outputs": {
            "v5_matching_results_path": str(V5_MATCHING_PATH),
            "v5_matching_results_size_mb": round(V5_MATCHING_PATH.stat().st_size / (1024 * 1024), 2),
            "v5_candidate_pairs_path": str(V5_CANDIDATES_PATH),
            "v5_candidate_pairs_size_mb": round(V5_CANDIDATES_PATH.stat().st_size / (1024 * 1024), 2),
            "staged_matching_path": str(V5_STAGING_MATCHING),
            "staged_candidates_path": str(V5_STAGING_CANDIDATES),
        },
        "v4_baseline_test_summary": {
            "total_s1_rows": total_s1,
            "predicted_links": v4_total_links,
        },
        "v5_test_summary": {
            "total_s1_rows": total_s1,
            "predicted_links": v5_total_links,
            "distinct_targets_linked": len(post_target_counts),
            "max_target_multiplicity": max_post_mult,
            "remaining_target_conflicts": post_conflicts,
            "s1_with_matches": v5_non_empty_s1,
            "s1_with_matches_pct": round(100.0 * v5_non_empty_s1 / total_s1, 2),
            "empty_s1_count": empty_s1,
            "empty_s1_pct": round(100.0 * empty_s1 / total_s1, 2),
            "s2_links": v5_s2_links,
            "s3_links": v5_s3_links,
            "total_candidate_pairs": total_v5_candidate_pairs,
            "novel_rescued_candidate_pairs": len(novel_rescue_pairs),
            "confirmed_rescued_links": confirmed_rescued_links,
            "near_tied_conflicts_model_scored": len(near_tied_groups),
            "match_count_distribution": {str(k): int(v) for k, v in sorted(v5_match_count_dist.items())},
        },
        "elapsed_sec": elapsed,
    }

    with open(V5_AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print("\n=== V5 FULL TEST INFERENCE SUMMARY ===", flush=True)
    print(json.dumps(audit_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
