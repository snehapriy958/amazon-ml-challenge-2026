"""V3 Target-Exclusivity Post-Processing Experiment.

Ground-Truth Law Verified on train_ground_truth.tsv (2,206,821 S1 rows, 7,638,365 links):
  - Max target multiplicity across S1 entities = 1
  - Number of S2/S3 target IDs linked to >1 S1 = 0
  - Every S2/S3 target entity belongs to AT MOST ONE S1 entity.

Applies deterministic target-exclusivity resolution to experiments/v3/output/matching_results.tsv
without touching any frozen V1, V2, or V3 files, writing to:
  - experiments/v3/output_exclusivity/matching_results.tsv
  - experiments/v3/output_exclusivity/exclusivity_experiment_report.json
"""

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

from rapidfuzz import fuzz

sys.path.insert(0, str(Path.cwd()))
from experiments.v3.src.normalization_v3 import normalize_name_v3, parse_address_v3

V3_MATCHING_PATH = Path("experiments/v3/output/matching_results.tsv")
EXP_OUT_DIR = Path("experiments/v3/output_exclusivity")
EXP_MATCHING_PATH = EXP_OUT_DIR / "matching_results.tsv"
EXP_REPORT_PATH = EXP_OUT_DIR / "exclusivity_experiment_report.json"

TEST_S1_PATH = Path("dataset/test/test_source1.tsv")
TEST_S2_PATH = Path("dataset/test/test_source2.tsv")
TEST_S3_PATH = Path("dataset/test/test_source3.tsv")


def _extract_record_features(raw_name: str, raw_addr: str, country: str = "") -> dict:
    rn = (raw_name or "").strip().lower()
    ra = (raw_addr or "").strip().lower()
    n_info = normalize_name_v3(raw_name or "")
    a_info = parse_address_v3(raw_addr or "", country or "")
    return {
        "raw_name": rn,
        "raw_addr": ra,
        "name_norm": n_info["lower_norm"],
        "name_core": n_info["core_name_norm"] or n_info["lower_norm"],
        "addr_norm": a_info["lower_norm"],
        "addr_core": a_info["orig_norm"] or a_info["lower_norm"],
        "postal": a_info["postal_code"] or "",
        "building": a_info["building_num"] or "",
        "unit": a_info["unit_num"] or "",
        "num_tokens": a_info["numeric_tokens"],
    }


def _score_owner_evidence(s1_rec: dict, tgt_rec: dict, rank_in_s1: int, s1_id: str) -> tuple:
    # 1. Name exact / normalized similarity
    raw_name_exact = 1 if (s1_rec["raw_name"] and s1_rec["raw_name"] == tgt_rec["raw_name"]) else 0
    core_name_exact = 1 if (s1_rec["name_core"] and s1_rec["name_core"] == tgt_rec["name_core"]) else 0
    norm_name_exact = 1 if (s1_rec["name_norm"] and s1_rec["name_norm"] == tgt_rec["name_norm"]) else 0

    nc1, nc2 = s1_rec["name_core"], tgt_rec["name_core"]
    rn1, rn2 = s1_rec["raw_name"], tgt_rec["raw_name"]
    if nc1 and nc2:
        n_tset = fuzz.token_set_ratio(nc1, nc2)
        n_tsort = fuzz.token_sort_ratio(nc1, nc2)
    else:
        n_tset = 0.0
        n_tsort = 0.0
    n_raw = fuzz.ratio(rn1, rn2) if (rn1 and rn2) else 0.0
    name_sim = 0.50 * n_tset + 0.30 * n_tsort + 0.20 * n_raw

    # 2. Address exact / normalized similarity
    raw_addr_exact = 1 if (s1_rec["raw_addr"] and s1_rec["raw_addr"] == tgt_rec["raw_addr"]) else 0
    core_addr_exact = 1 if (s1_rec["addr_norm"] and s1_rec["addr_norm"] == tgt_rec["addr_norm"]) else 0

    ac1, ac2 = s1_rec["addr_norm"], tgt_rec["addr_norm"]
    ra1, ra2 = s1_rec["raw_addr"], tgt_rec["raw_addr"]
    if ac1 and ac2:
        a_tset = fuzz.token_set_ratio(ac1, ac2)
        a_tsort = fuzz.token_sort_ratio(ac1, ac2)
    else:
        a_tset = 0.0
        a_tsort = 0.0
    a_raw = fuzz.ratio(ra1, ra2) if (ra1 and ra2) else 0.0
    addr_sim = 0.50 * a_tset + 0.30 * a_tsort + 0.20 * a_raw

    # 3. Building-number agreement
    b1, b2 = s1_rec["building"], tgt_rec["building"]
    if b1 and b2:
        if b1 == b2:
            house_agree = 1.0
        elif b1 in b2 or b2 in b1:
            house_agree = 0.6
        else:
            house_agree = -1.0  # Explicit building number conflict
    else:
        house_agree = 0.0

    un1, un2 = s1_rec["unit"], tgt_rec["unit"]
    nt1, nt2 = s1_rec["num_tokens"], tgt_rec["num_tokens"]
    street_unit_agree = 0.0
    if un1 and un2:
        street_unit_agree += 0.4 if (un1 == un2) else -0.4
    if nt1 and nt2:
        street_unit_agree += 0.4 if (nt1 & nt2) else -0.4

    # 4. Postal-code agreement
    p1, p2 = s1_rec["postal"], tgt_rec["postal"]
    if p1 and p2:
        if p1 == p2:
            postal_agree = 1.0
        elif len(p1) >= 3 and len(p2) >= 3 and p1[:3] == p2[:3]:
            postal_agree = 0.3
        else:
            postal_agree = -1.0  # Explicit postal code conflict
    else:
        postal_agree = 0.0

    # 5. Name + Address combined deterministic score
    n_norm01 = name_sim / 100.0
    a_norm01 = addr_sim / 100.0
    geom = math.sqrt(max(0.0, n_norm01 * a_norm01))
    combined_score = (
        0.45 * n_norm01
        + 0.35 * a_norm01
        + 0.20 * geom
        + 0.12 * postal_agree
        + 0.12 * house_agree
        + 0.04 * street_unit_agree
        + 0.08 * core_addr_exact
        + 0.05 * raw_addr_exact
        + 0.05 * core_name_exact
        + 0.03 * norm_name_exact
        + 0.03 * raw_name_exact
        - 0.004 * rank_in_s1
    )

    return (
        round(combined_score, 8),
        raw_addr_exact,
        core_addr_exact,
        postal_agree,
        house_agree,
        round(addr_sim, 6),
        round(name_sim, 6),
        -rank_in_s1,
        s1_id,
    )


def main():
    t0 = time.time()
    EXP_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: Scanning experiments/v3/output/matching_results.tsv for target multiplicity...")
    target_claims = defaultdict(list)  # tid -> list of (s1_id, rank_in_s1)
    s1_rows = []  # (s1_id, [mids])

    v3_total_links = 0
    v3_non_empty_s1 = 0
    v3_s2_links = 0
    v3_s3_links = 0

    with open(V3_MATCHING_PATH, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            s1_id, _, rest = line.rstrip("\r\n").partition("\t")
            mids = rest.split(",") if rest else []
            s1_rows.append((s1_id, mids))
            if mids:
                v3_non_empty_s1 += 1
                v3_total_links += len(mids)
                for rank_idx, tid in enumerate(mids):
                    target_claims[tid].append((s1_id, rank_idx))
                    if tid.startswith("S2-"):
                        v3_s2_links += 1
                    elif tid.startswith("S3-"):
                        v3_s3_links += 1

    conflicted_targets = {tid: claims for tid, claims in target_claims.items() if len(claims) > 1}
    conflicted_s1_ids = {s1_id for claims in conflicted_targets.values() for s1_id, _ in claims}
    total_duplicate_assignments = sum(len(c) - 1 for c in conflicted_targets.values())

    print(f"  Total S1 rows:                 {len(s1_rows):,}")
    print(f"  V3 total predicted links:      {v3_total_links:,} (S2={v3_s2_links:,}, S3={v3_s3_links:,})")
    print(f"  V3 distinct targets linked:    {len(target_claims):,}")
    print(f"  Conflicted target IDs (>1 S1): {len(conflicted_targets):,}")
    print(f"  Conflicted S1 entities:        {len(conflicted_s1_ids):,}")
    print(f"  Total excess assignments:      {total_duplicate_assignments:,} ({time.time()-t0:.1f}s)")

    print("Step 2: Loading & normalizing raw records for conflicted S1 and S2/S3 entities...")
    t_load = time.time()
    s1_records = {}
    with open(TEST_S1_PATH, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            if eid in conflicted_s1_ids:
                s1_records[eid] = _extract_record_features(
                    parts[1] if len(parts) > 1 else "",
                    parts[2] if len(parts) > 2 else "",
                    parts[3] if len(parts) > 3 else "",
                )

    target_records = {}
    for src_path in (TEST_S2_PATH, TEST_S3_PATH):
        with open(src_path, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                eid = parts[0]
                if eid in conflicted_targets:
                    target_records[eid] = _extract_record_features(
                        parts[1] if len(parts) > 1 else "",
                        parts[2] if len(parts) > 2 else "",
                        parts[3] if len(parts) > 3 else "",
                    )

    print(
        f"  Loaded & normalized {len(s1_records):,} S1 records and {len(target_records):,} target records in {time.time()-t_load:.1f}s"
    )

    print("Step 3: Resolving 207,402 conflicted targets using deterministic evidence...")
    t_res = time.time()
    winning_owner = {}  # tid -> winning s1_id
    for tid, claims in conflicted_targets.items():
        tgt_rec = target_records[tid]
        best_s1 = None
        best_key = None
        for s1_id, rank_idx in claims:
            s1_rec = s1_records[s1_id]
            score_key = _score_owner_evidence(s1_rec, tgt_rec, rank_idx, s1_id)
            if best_key is None or score_key > best_key:
                best_key = score_key
                best_s1 = s1_id
        winning_owner[tid] = best_s1

    print(f"  Resolved {len(winning_owner):,} target conflicts in {time.time()-t_res:.1f}s")

    print(f"Step 4: Writing {EXP_MATCHING_PATH} (preserving unique targets 100%)...")
    exp_total_links = 0
    exp_non_empty_s1 = 0
    exp_s2_links = 0
    exp_s3_links = 0
    removed_links = 0
    match_count_dist = Counter()

    with open(EXP_MATCHING_PATH, "w", encoding="utf-8", newline="\n") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, mids in s1_rows:
            kept = []
            for tid in mids:
                if tid not in winning_owner or winning_owner[tid] == s1_id:
                    kept.append(tid)
                else:
                    removed_links += 1
            match_count_dist[len(kept)] += 1
            if kept:
                exp_non_empty_s1 += 1
                exp_total_links += len(kept)
                for tid in kept:
                    if tid.startswith("S2-"):
                        exp_s2_links += 1
                    elif tid.startswith("S3-"):
                        exp_s3_links += 1
            out_f.write(f"{s1_id}\t{','.join(kept)}\n")

    report = {
        "train_ground_truth_audit": {
            "total_s1_rows": 2206821,
            "total_gt_links": 7638365,
            "distinct_targets_linked": 7638365,
            "max_multiplicity": 1,
            "targets_appearing_in_gt_more_than_1_s1": 0,
            "total_duplicate_target_assignments": 0,
        },
        "v3_baseline": {
            "path": str(V3_MATCHING_PATH),
            "total_s1_rows": len(s1_rows),
            "predicted_links": v3_total_links,
            "s1_with_matches": v3_non_empty_s1,
            "s1_with_matches_pct": round(100.0 * v3_non_empty_s1 / len(s1_rows), 2),
            "singleton_s1": len(s1_rows) - v3_non_empty_s1,
            "singleton_s1_pct": round(100.0 * (len(s1_rows) - v3_non_empty_s1) / len(s1_rows), 2),
            "s2_links": v3_s2_links,
            "s3_links": v3_s3_links,
            "distinct_targets": len(target_claims),
            "max_target_multiplicity": max(len(c) for c in target_claims.values()),
            "conflicted_targets_count": len(conflicted_targets),
            "excess_duplicate_assignments": total_duplicate_assignments,
        },
        "v3_exclusivity_experiment": {
            "path": str(EXP_MATCHING_PATH),
            "size_mb": round(EXP_MATCHING_PATH.stat().st_size / (1024 * 1024), 2),
            "total_s1_rows": len(s1_rows),
            "predicted_links": exp_total_links,
            "s1_with_matches": exp_non_empty_s1,
            "s1_with_matches_pct": round(100.0 * exp_non_empty_s1 / len(s1_rows), 2),
            "singleton_s1": len(s1_rows) - exp_non_empty_s1,
            "singleton_s1_pct": round(100.0 * (len(s1_rows) - exp_non_empty_s1) / len(s1_rows), 2),
            "s2_links": exp_s2_links,
            "s3_links": exp_s3_links,
            "distinct_targets": exp_total_links,
            "max_target_multiplicity": 1,
            "conflicted_targets_resolved": len(conflicted_targets),
            "removed_links": removed_links,
            "match_count_distribution": {str(k): match_count_dist[k] for k in sorted(match_count_dist)},
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }

    with open(EXP_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== TARGET-EXCLUSIVITY EXPERIMENT SUMMARY ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
