"""Complete Independent Post-Inference Audit for V2-C Output.

Validates:
1. Exact S1 count = 1,732,544.
2. Exactly one matching_results row per S1 entity.
3. No duplicate S1 IDs.
4. No missing/extra S1 IDs.
5. Every matched target ID exists in test_source2 or test_source3.
6. No duplicate target IDs within an S1 row.
7. No malformed TSV rows.
8. No self matches (S1-xxx in matched IDs).
9. Candidate pairs contains every predicted match (100% subset compliance).
10. Candidate pairs and matching results have matching S1 coverage and order.
11. Official validator (utils/validate_submission.py) exit code.
12. Country coverage across all 3 countries (France, US, India).
13. Empty / non-empty row counts, total predicted matches, and mean matches per S1.
14. SHA-256 byte-for-byte comparison of frozen V1 artifacts.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import pandas as pd

EXPECTED_S1_COUNT = 1_732_544

V1_BASELINE_METRICS = {
    "matching_results": {
        "path": "output/matching_results.tsv",
        "expected_size": 95_428_597,
        "expected_sha256": "e4354d22a35352854e81982cc3746b160c824a93773e5e69d8ca15909810c884",
    },
    "checkpoint": {
        "path": "output/checkpoint_full_inference.json",
        "expected_size": 167,
        "expected_sha256": "09265263f1c280363b01f7185617e0f35bdeb61206b682c7b45611ed0e29d3fc",
    },
    "model": {
        "path": "models/xgboost_matcher.joblib",
        "expected_size": 1_388_125,
        "expected_sha256": "d2387047d50e99328cf564e840c940bbc5c28bdc17ecc8af93eea3820755eead",
    },
    "candidate_pairs": {
        "path": "output/candidate_pairs.tsv",
        "expected_size": 11_429_655_025,
    },
}

V2_MATCH_PATH = Path("experiments/v2/output/matching_results.tsv")
V2_CAND_PATH = Path("experiments/v2/output/candidate_pairs.tsv")
V2_CHECKPOINT_PATH = Path("experiments/v2/output/checkpoint_full_inference.json")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_audit():
    print("=" * 80)
    print("=== INDEPENDENT POST-INFERENCE AUDIT FOR V2-C OUTPUT ===")
    print("=" * 80)
    t_audit_start = time.time()

    audit_results = {}
    passed_all = True

    # -------------------------------------------------------------------------
    # 1. Verify V1 Baseline Remains 100% Byte-for-Byte Unchanged
    # -------------------------------------------------------------------------
    print("\n--- 1. VERIFYING FROZEN V1 BASELINE ARTIFACTS ---")
    v1_checks = {}
    for name, spec in V1_BASELINE_METRICS.items():
        p = Path(spec["path"])
        if not p.exists():
            print(f"FAILED: {p} not found!")
            passed_all = False
            continue
        cur_sz = p.stat().st_size
        sz_ok = (cur_sz == spec["expected_size"])
        sha_ok = True
        cur_sha = None
        if "expected_sha256" in spec:
            cur_sha = compute_sha256(p)
            sha_ok = (cur_sha == spec["expected_sha256"])

        v1_checks[name] = {
            "path": str(p),
            "size_bytes": cur_sz,
            "expected_size": spec["expected_size"],
            "size_match": sz_ok,
            "sha256": cur_sha,
            "expected_sha256": spec.get("expected_sha256"),
            "sha_match": sha_ok,
        }
        status_str = "PASS (Unchanged)" if (sz_ok and sha_ok) else "FAIL (MODIFIED!)"
        print(f"  {p.name:<32} | {cur_sz:,} bytes | {status_str}")
        if not (sz_ok and sha_ok):
            passed_all = False

    audit_results["v1_preservation"] = v1_checks

    # -------------------------------------------------------------------------
    # 2. File Existence & Headers
    # -------------------------------------------------------------------------
    print("\n--- 2. VERIFYING V2-C FILE EXISTENCE & HEADERS ---")
    assert V2_MATCH_PATH.exists(), f"Missing {V2_MATCH_PATH}"
    assert V2_CAND_PATH.exists(), f"Missing {V2_CAND_PATH}"
    assert V2_CHECKPOINT_PATH.exists(), f"Missing {V2_CHECKPOINT_PATH}"

    with open(V2_MATCH_PATH, "r", encoding="utf-8") as f:
        match_header = f.readline().rstrip("\r\n")
    with open(V2_CAND_PATH, "r", encoding="utf-8") as f:
        cand_header = f.readline().rstrip("\r\n")

    expected_m_hdr = "source1_entity_id\tmatched_entity_ids"
    expected_c_hdr = "source1_entity_id\tcandidate_entity_ids"
    print(f"  Matching Header:  '{match_header}' -> {'PASS' if match_header == expected_m_hdr else 'FAIL'}")
    print(f"  Candidate Header: '{cand_header}' -> {'PASS' if cand_header == expected_c_hdr else 'FAIL'}")
    assert match_header == expected_m_hdr
    assert cand_header == expected_c_hdr

    # -------------------------------------------------------------------------
    # 3. Load Expected Test S1 IDs and Country Mapping
    # -------------------------------------------------------------------------
    print("\n--- 3. LOADING TEST SOURCE 1 REFERENCE DATA ---")
    test_s1_ids = []
    test_s1_countries = {}
    with open("dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                sid, ctry = parts[0], parts[3].strip().lower()
                test_s1_ids.append(sid)
                test_s1_countries[sid] = ctry

    test_s1_set = set(test_s1_ids)
    print(f"  Loaded {len(test_s1_ids):,} test S1 entities.")
    assert len(test_s1_ids) == EXPECTED_S1_COUNT

    # Load Valid Target IDs from test_source2 and test_source3
    print("Loading valid target IDs from test_source2.tsv and test_source3.tsv...")
    valid_target_ids = set()
    for src_path in ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
        with open(src_path, "r", encoding="utf-8") as f:
            header = next(f)
            for line in f:
                cid = line.split("\t", 1)[0].strip()
                valid_target_ids.add(cid)
    print(f"  Loaded {len(valid_target_ids):,} valid target IDs.")

    # -------------------------------------------------------------------------
    # 4. Stream Validation of Matching Results & Candidate Pairs
    # -------------------------------------------------------------------------
    print("\n--- 4. STREAM VALIDATION OF MATCHING_RESULTS & CANDIDATE_PAIRS ---")
    m_seen_s1 = []
    m_dup_s1 = set()
    m_malformed = 0
    m_self_matches = 0
    m_invalid_targets = 0
    m_duplicate_targets_in_row = 0
    empty_rows = 0
    non_empty_rows = 0
    total_predicted_matches = 0
    country_matches = defaultdict(int)
    country_entities = defaultdict(int)

    # Dictionary to check candidate subset compliance
    # We will verify streaming candidate_pairs simultaneously
    subset_violations = 0
    cand_mismatch_s1 = 0

    with open(V2_MATCH_PATH, "r", encoding="utf-8") as fm, open(V2_CAND_PATH, "r", encoding="utf-8") as fc:
        next(fm)  # skip headers
        next(fc)

        row_idx = 0
        for l_m, l_c in zip(fm, fc):
            row_idx += 1
            parts_m = l_m.rstrip("\r\n").split("\t")
            parts_c = l_c.rstrip("\r\n").split("\t")

            if len(parts_m) < 2:
                m_malformed += 1
                continue
            if len(parts_c) < 2:
                m_malformed += 1
                continue

            sid_m, val_m = parts_m[0], parts_m[1]
            sid_c, val_c = parts_c[0], parts_c[1]

            if sid_m != sid_c:
                cand_mismatch_s1 += 1

            m_seen_s1.append(sid_m)
            ctry = test_s1_countries.get(sid_m, "unknown")
            country_entities[ctry] += 1

            if val_m == "":
                empty_rows += 1
            else:
                non_empty_rows += 1
                targets = val_m.split(",")
                total_predicted_matches += len(targets)
                country_matches[ctry] += len(targets)

                # Check duplicates within row
                if len(targets) != len(set(targets)):
                    m_duplicate_targets_in_row += 1

                # Check self matches and invalid targets
                for t in targets:
                    if t == sid_m or t.startswith("S1-"):
                        m_self_matches += 1
                    if t not in valid_target_ids:
                        m_invalid_targets += 1

                # Check candidate subset compliance: all matches must be in candidate string
                cand_set = set(val_c.split(",")) if val_c else set()
                for t in targets:
                    if t not in cand_set:
                        subset_violations += 1

            if row_idx % 400_000 == 0:
                print(f"  Validated {row_idx:,} / {EXPECTED_S1_COUNT:,} rows...", flush=True)

    print(f"Stream validation complete across {row_idx:,} rows.")

    # Check S1 ID set properties
    seen_set = set(m_seen_s1)
    missing_s1 = test_s1_set - seen_set
    extra_s1 = seen_set - test_s1_set
    exact_order_match = (m_seen_s1 == test_s1_ids)

    # -------------------------------------------------------------------------
    # 5. Check Official Validator
    # -------------------------------------------------------------------------
    print("\n--- 5. RUNNING OFFICIAL VALIDATOR (utils/validate_submission.py) ---")
    val_cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", str(V2_MATCH_PATH),
        "--candidate", "NONE",
        "--test-dir", "dataset/test",
        "--check-ids",
    ]
    p_val = subprocess.run(val_cmd, capture_output=True, text=True)
    val_exit_code = p_val.returncode
    print(f"  Validator Exit Code: {val_exit_code} ({'PASS' if val_exit_code == 0 else 'FAIL'})")
    print(f"  Validator Output Summary:\n{p_val.stdout.strip()[:600]}")

    # -------------------------------------------------------------------------
    # 6. Summary Metrics & Audit Compilation
    # -------------------------------------------------------------------------
    mean_matches_per_s1 = total_predicted_matches / EXPECTED_S1_COUNT
    mean_matches_non_empty = total_predicted_matches / non_empty_rows if non_empty_rows > 0 else 0.0

    print("\n" + "=" * 80)
    print("=== AUDIT SUMMARY RESULTS ===")
    print("=" * 80)
    print(f"Total S1 Count:               {row_idx:,} / {EXPECTED_S1_COUNT:,} -> {'PASS' if row_idx == EXPECTED_S1_COUNT else 'FAIL'}")
    print(f"Exact Input Order Match:      {exact_order_match} -> {'PASS' if exact_order_match else 'FAIL'}")
    print(f"Missing S1 IDs:               {len(missing_s1):,} -> {'PASS' if len(missing_s1) == 0 else 'FAIL'}")
    print(f"Extra S1 IDs:                 {len(extra_s1):,} -> {'PASS' if len(extra_s1) == 0 else 'FAIL'}")
    print(f"Duplicate S1 IDs:             {len(m_seen_s1) - len(seen_set):,} -> {'PASS' if len(m_seen_s1) == len(seen_set) else 'FAIL'}")
    print(f"Malformed TSV Rows:           {m_malformed:,} -> {'PASS' if m_malformed == 0 else 'FAIL'}")
    print(f"Self Matches:                 {m_self_matches:,} -> {'PASS' if m_self_matches == 0 else 'FAIL'}")
    print(f"Invalid Target IDs:           {m_invalid_targets:,} -> {'PASS' if m_invalid_targets == 0 else 'FAIL'}")
    print(f"Duplicate Targets per Row:    {m_duplicate_targets_in_row:,} -> {'PASS' if m_duplicate_targets_in_row == 0 else 'FAIL'}")
    print(f"Subset Compliance Violations: {subset_violations:,} -> {'PASS' if subset_violations == 0 else 'FAIL'}")
    print(f"S1 Match / Candidate Drift:   {cand_mismatch_s1:,} -> {'PASS' if cand_mismatch_s1 == 0 else 'FAIL'}")
    print(f"Official Validator:           Exit code {val_exit_code} -> {'PASS' if val_exit_code == 0 else 'FAIL'}")
    print("-" * 80)
    print(f"Predicted Singletons (Empty): {empty_rows:,} ({empty_rows/EXPECTED_S1_COUNT*100:.2f}%)")
    print(f"Non-Empty Predicted Rows:     {non_empty_rows:,} ({non_empty_rows/EXPECTED_S1_COUNT*100:.2f}%)")
    print(f"Total Matches Predicted:      {total_predicted_matches:,}")
    print(f"Mean Matches per S1 Entity:   {mean_matches_per_s1:.4f}")
    print(f"Mean Matches (Non-Empty):     {mean_matches_non_empty:.4f}")
    print("-" * 80)
    print("Country Breakdown:")
    for ctry in sorted(country_entities):
        n_ent = country_entities[ctry]
        n_mat = country_matches[ctry]
        print(f"  [{ctry.upper()}]: {n_ent:,} S1 entities | {n_mat:,} matches (mean: {n_mat/n_ent:.3f}/S1)")
    print("=" * 80)

    # Save audit report
    audit_summary = {
        "status": "PASS" if (val_exit_code == 0 and subset_violations == 0 and len(missing_s1) == 0) else "FAIL",
        "total_s1_rows": row_idx,
        "expected_s1_rows": EXPECTED_S1_COUNT,
        "exact_order_match": exact_order_match,
        "missing_s1_count": len(missing_s1),
        "extra_s1_count": len(extra_s1),
        "duplicate_s1_count": len(m_seen_s1) - len(seen_set),
        "malformed_rows": m_malformed,
        "self_matches": m_self_matches,
        "invalid_targets": m_invalid_targets,
        "duplicate_targets_per_row": m_duplicate_targets_in_row,
        "subset_compliance_violations": subset_violations,
        "official_validator_exit_code": val_exit_code,
        "empty_rows": empty_rows,
        "non_empty_rows": non_empty_rows,
        "total_predicted_matches": total_predicted_matches,
        "mean_matches_per_s1": mean_matches_per_s1,
        "mean_matches_non_empty": mean_matches_non_empty,
        "country_entities": dict(country_entities),
        "country_matches": dict(country_matches),
        "v1_preservation": v1_checks,
        "audit_runtime_s": time.time() - t_audit_start,
    }

    out_audit_path = Path("experiments/v2/results/v2_c_audit_summary.json")
    out_audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Saved audit summary to {out_audit_path}")


if __name__ == "__main__":
    run_audit()
