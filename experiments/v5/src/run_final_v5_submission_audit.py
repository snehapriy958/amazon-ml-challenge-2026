"""Full 12-Point Submission Audit for V5 Entity Resolution Pipeline.

Performs the comprehensive audit on:
  - `experiments/v5/output/v5_matching_results.tsv`
  - `experiments/v5/output/v5_candidate_pairs.tsv`
  - `experiments/v5/submission_staging/output/matching_results.tsv`
  - `experiments/v5/submission_staging/output/candidate_pairs.tsv`

Checks:
  1. Exactly 1,732,544 S1 rows.
  2. Every S1 appears exactly once.
  3. No duplicate S1 rows.
  4. No duplicate predictions within an S1 row.
  5. No self-matches (no S1- prefix or S1 == target).
  6. All target IDs are valid (verified against test_source2.tsv and test_source3.tsv).
  7. Every predicted pair exists in candidate_pairs.tsv.
  8. Original S1 ordering is preserved (matches test_source1.tsv exact order).
  9. No malformed/truncated output.
 10. Target exclusivity statistics (0 target conflicts, max target multiplicity <= 1).
 11. S2/S3 prediction counts.
 12. Empty/nonempty S1 counts.
Then runs the official validator:
  python utils/validate_submission.py --matching ... --candidate NONE --test-dir dataset/test --check-ids
"""

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Set

sys.stdout.reconfigure(encoding="utf-8")

V5_ROOT = Path("experiments/v5")
V5_MATCHING = V5_ROOT / "output" / "v5_matching_results.tsv"
V5_CANDIDATES = V5_ROOT / "output" / "v5_candidate_pairs.tsv"
V5_STAGING_MATCHING = V5_ROOT / "submission_staging" / "output" / "matching_results.tsv"
V5_STAGING_CANDIDATES = V5_ROOT / "submission_staging" / "output" / "candidate_pairs.tsv"

TEST_DIR = Path("dataset/test")
TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

AUDIT_OUTPUT_JSON = V5_ROOT / "results" / "v5_final_submission_audit.json"


def run_v5_full_audit(matching_path: Path, candidate_path: Path) -> dict:
    t0 = time.time()
    print("=" * 96, flush=True)
    print(f"RUNNING V5 12-POINT FULL AUDIT ON: {matching_path}", flush=True)
    print("=" * 96, flush=True)

    assert matching_path.exists(), f"Missing matching results file: {matching_path}"
    assert candidate_path.exists(), f"Missing candidate pairs file: {candidate_path}"

    # Step 1: Load expected S1 order and valid target IDs
    print("[Audit Step 1] Loading test_source1.tsv S1 entity order & test_source2/3 target IDs...", flush=True)
    t_load = time.time()
    expected_s1_order = []
    with open(TEST_S1, "r", encoding="utf-8", buffering=4 << 20) as f:
        next(f, None)
        for line in f:
            if line.strip():
                expected_s1_order.append(line.split("\t", 1)[0].strip())
    expected_s1_set = set(expected_s1_order)
    total_expected_s1 = len(expected_s1_order)

    valid_targets = set()
    for src in (TEST_S2, TEST_S3):
        with open(src, "r", encoding="utf-8", buffering=8 << 20) as f:
            next(f, None)
            for line in f:
                if line.strip():
                    valid_targets.add(line.split("\t", 1)[0].strip())

    print(
        f"  Loaded {total_expected_s1:,} expected S1 IDs and {len(valid_targets):,} valid S2/S3 target IDs in {time.time()-t_load:.1f}s",
        flush=True,
    )

    # Step 2: Stream matching_results.tsv and candidate_pairs.tsv simultaneously
    print("\n[Audit Step 2] Streaming simultaneous audit of matching_results and candidate_pairs...", flush=True)
    t_stream = time.time()

    matching_rows = 0
    candidate_rows = 0
    seen_match_s1 = set()
    dup_s1_in_matching = 0
    dup_s1_in_candidate = 0
    s1_order_mismatches_vs_test_s1 = 0
    s1_order_mismatches_match_vs_cand = 0

    intra_row_dup_matching = 0
    self_matches_matching = 0
    invalid_target_ids_matching = 0
    subset_violations = 0
    malformed_rows_matching = 0
    malformed_rows_candidate = 0

    target_multiplicity = Counter()
    total_predicted_links = 0
    s2_links = 0
    s3_links = 0
    empty_s1 = 0
    non_empty_s1 = 0
    match_count_dist = Counter()

    with (
        open(matching_path, "r", encoding="utf-8", buffering=16 << 20) as f_match,
        open(candidate_path, "r", encoding="utf-8", buffering=16 << 20) as f_cand,
    ):
        m_hdr = f_match.readline().rstrip("\r\n").split("\t")
        c_hdr = f_cand.readline().rstrip("\r\n").split("\t")
        assert m_hdr == ["source1_entity_id", "matched_entity_ids"], f"Bad matching header: {m_hdr}"
        assert c_hdr == ["source1_entity_id", "candidate_entity_ids"], f"Bad candidate header: {c_hdr}"

        for line_idx, (m_line, c_line) in enumerate(zip(f_match, f_cand)):
            # Matching row parse
            m_parts = m_line.rstrip("\r\n").split("\t")
            if len(m_parts) < 1:
                malformed_rows_matching += 1
                continue
            m_s1 = m_parts[0]
            m_rest = m_parts[1] if len(m_parts) > 1 else ""

            # Candidate row parse
            c_parts = c_line.rstrip("\r\n").split("\t")
            if len(c_parts) < 1:
                malformed_rows_candidate += 1
                continue
            c_s1 = c_parts[0]
            c_rest = c_parts[1] if len(c_parts) > 1 else ""

            matching_rows += 1
            candidate_rows += 1

            if m_s1 != c_s1:
                s1_order_mismatches_match_vs_cand += 1
            if line_idx < total_expected_s1 and m_s1 != expected_s1_order[line_idx]:
                s1_order_mismatches_vs_test_s1 += 1

            if m_s1 in seen_match_s1:
                dup_s1_in_matching += 1
            seen_match_s1.add(m_s1)

            m_ids = m_rest.split(",") if m_rest else []
            k = len(m_ids)
            match_count_dist[k] += 1
            if k == 0:
                empty_s1 += 1
            else:
                non_empty_s1 += 1
                total_predicted_links += k

                if len(m_ids) != len(set(m_ids)):
                    intra_row_dup_matching += 1

                c_set = set(c_rest.split(",")) if c_rest else set()

                for tid in m_ids:
                    if tid.startswith("S1-") or tid == m_s1:
                        self_matches_matching += 1
                    if tid not in valid_targets:
                        invalid_target_ids_matching += 1
                    if tid not in c_set:
                        subset_violations += 1

                    target_multiplicity[tid] += 1
                    if tid.startswith("S2-"):
                        s2_links += 1
                    elif tid.startswith("S3-"):
                        s3_links += 1

            if (line_idx + 1) % 500_000 == 0:
                print(f"    ... processed {line_idx+1:,} rows ({time.time()-t_stream:.1f}s)", flush=True)

    target_conflicts = sum(1 for c in target_multiplicity.values() if c > 1)
    max_target_mult = max(target_multiplicity.values()) if target_multiplicity else 0

    print(f"  Streaming audit finished in {time.time()-t_stream:.1f}s", flush=True)

    # Step 3: Run official validate_submission.py
    print("\n[Audit Step 3] Running official utils/validate_submission.py --check-ids...", flush=True)
    cmd_val = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching",
        str(matching_path),
        "--candidate",
        "NONE",
        "--test-dir",
        str(TEST_DIR),
        "--check-ids",
    ]
    proc = subprocess.run(cmd_val, capture_output=True, text=True)
    print(proc.stdout.strip(), flush=True)
    if proc.stderr.strip():
        print("Validator stderr:", proc.stderr.strip(), flush=True)

    # Compile 12 audit verification results
    v1_s1_row_count = matching_rows == 1_732_544
    v2_every_s1_once = len(seen_match_s1) == 1_732_544 and (seen_match_s1 == expected_s1_set)
    v3_no_dup_s1_rows = dup_s1_in_matching == 0
    v4_no_intra_row_dups = intra_row_dup_matching == 0
    v5_no_self_matches = self_matches_matching == 0
    v6_all_targets_valid = invalid_target_ids_matching == 0
    v7_all_in_candidates = subset_violations == 0
    v8_order_preserved = (s1_order_mismatches_vs_test_s1 == 0) and (s1_order_mismatches_match_vs_cand == 0)
    v9_no_malformed = (malformed_rows_matching == 0) and (malformed_rows_candidate == 0)
    v10_exclusivity_verified = (target_conflicts == 0) and (max_target_mult <= 1)
    v11_s2_s3_counts = {"s2_links": s2_links, "s3_links": s3_links, "total_links": total_predicted_links}
    v12_empty_counts = {"empty_s1": empty_s1, "non_empty_s1": non_empty_s1, "total_s1": matching_rows}
    v13_official_validator_pass = proc.returncode == 0

    all_passed = (
        v1_s1_row_count
        and v2_every_s1_once
        and v3_no_dup_s1_rows
        and v4_no_intra_row_dups
        and v5_no_self_matches
        and v6_all_targets_valid
        and v7_all_in_candidates
        and v8_order_preserved
        and v9_no_malformed
        and v10_exclusivity_verified
        and v13_official_validator_pass
    )

    audit_summary = {
        "status": "PASS" if all_passed else "FAIL",
        "official_validator_exit_code": proc.returncode,
        "official_validator_pass": v13_official_validator_pass,
        "official_validator_output": proc.stdout.strip(),
        "audit_points": {
            "1_exact_1732544_s1_rows": {"passed": v1_s1_row_count, "count": matching_rows},
            "2_every_s1_appears_exactly_once": {"passed": v2_every_s1_once, "distinct_s1": len(seen_match_s1)},
            "3_no_duplicate_s1_rows": {"passed": v3_no_dup_s1_rows, "duplicates": dup_s1_in_matching},
            "4_no_duplicate_predictions_in_row": {"passed": v4_no_intra_row_dups, "intra_row_dups": intra_row_dup_matching},
            "5_no_self_matches": {"passed": v5_no_self_matches, "self_matches": self_matches_matching},
            "6_all_target_ids_valid": {"passed": v6_all_targets_valid, "invalid_ids": invalid_target_ids_matching},
            "7_every_prediction_in_candidates": {"passed": v7_all_in_candidates, "subset_violations": subset_violations},
            "8_s1_order_preserved": {"passed": v8_order_preserved, "order_mismatches": s1_order_mismatches_vs_test_s1},
            "9_no_malformed_output": {"passed": v9_no_malformed, "malformed_rows": malformed_rows_matching},
            "10_target_exclusivity": {
                "passed": v10_exclusivity_verified,
                "target_conflicts": target_conflicts,
                "max_target_multiplicity": max_target_mult,
                "distinct_targets_linked": len(target_multiplicity),
            },
            "11_s2_s3_prediction_counts": v11_s2_s3_counts,
            "12_empty_nonempty_s1_counts": v12_empty_counts,
        },
        "match_count_distribution": {str(k): int(v) for k, v in sorted(match_count_dist.items())},
        "elapsed_sec": round(time.time() - t0, 2),
    }

    with open(AUDIT_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print("\n" + "=" * 96, flush=True)
    print(f"AUDIT RESULT: {'PASS (ALL CHECKS PASSED)' if all_passed else 'FAIL'}", flush=True)
    print("=" * 96, flush=True)
    print(json.dumps(audit_summary, indent=2), flush=True)
    return audit_summary


if __name__ == "__main__":
    match_file = V5_STAGING_MATCHING if V5_STAGING_MATCHING.exists() else V5_MATCHING
    cand_file = V5_STAGING_CANDIDATES if V5_STAGING_CANDIDATES.exists() else V5_CANDIDATES
    res = run_v5_full_audit(match_file, cand_file)
    if res["status"] != "PASS":
        sys.exit(1)
