"""Final V4 Submission Audit & Official Submission ZIP Builder.

Performs the 10-point Read-Only Audit on the already-generated V4 artifacts:
  - experiments/v4/output/v4_matching_results.tsv
  - experiments/v4/output/v4_candidate_pairs.tsv
  - experiments/v4/v4_model.joblib
  - experiments/v4/v4_config.json
  - experiments/v4/v4_validation_report.json

Checks:
  1. Verify matching_results.tsv has exactly 1,732,544 S1 rows.
  2. Verify every S1 appears exactly once.
  3. Verify no duplicate matched target IDs within an S1 row.
  4. Verify no self matches (no S1- prefix or S1==target).
  5. Verify every predicted target is a valid test S2/S3 ID (against dataset/test/test_source2/3.tsv).
  6. Verify every predicted match exists in candidate_pairs.tsv.
  7. Verify candidate_pairs.tsv has exactly the same S1 row order (and matches test_source1.tsv order).
  8. Verify no malformed/truncated rows in either file.
  9. Run official utils/validate_submission.py validator.
 10. Report all metrics and, if PASS, stage & build `experiments/v4/v4_submission.zip`
     following the exact official Amazon ML Challenge 2026 submission structure
     without modifying any V1, V2, V3, or V4 prediction/model files.
"""

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

sys.stdout.reconfigure(encoding="utf-8")

V4_ROOT = Path("experiments/v4")
V4_MATCHING = V4_ROOT / "output" / "v4_matching_results.tsv"
V4_CANDIDATE = V4_ROOT / "output" / "v4_candidate_pairs.tsv"
V4_MODEL = V4_ROOT / "v4_model.joblib"
V4_CONFIG = V4_ROOT / "v4_config.json"
V4_VAL_REPORT = V4_ROOT / "v4_validation_report.json"

TEST_DIR = Path("dataset/test")
TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

FINAL_AUDIT_JSON = V4_ROOT / "results" / "v4_final_submission_audit.json"
STAGING_DIR = V4_ROOT / "submission_staging"
SUBMISSION_ZIP = V4_ROOT / "v4_submission.zip"


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    t0 = time.time()
    print("=" * 88, flush=True)
    print("FINAL V4 SUBMISSION AUDIT (READ-ONLY ON GENERATED V4 ARTIFACTS)", flush=True)
    print("=" * 88, flush=True)

    # Verify all 5 required V4 artifacts exist
    for p in (V4_MATCHING, V4_CANDIDATE, V4_MODEL, V4_CONFIG, V4_VAL_REPORT):
        assert p.exists(), f"Missing required V4 artifact: {p}"

    file_sizes = {
        "v4_matching_results_bytes": V4_MATCHING.stat().st_size,
        "v4_matching_results_mb": round(V4_MATCHING.stat().st_size / (1024 * 1024), 2),
        "v4_candidate_pairs_bytes": V4_CANDIDATE.stat().st_size,
        "v4_candidate_pairs_mb": round(V4_CANDIDATE.stat().st_size / (1024 * 1024), 2),
        "v4_model_joblib_mb": round(V4_MODEL.stat().st_size / (1024 * 1024), 2),
        "v4_config_json_bytes": V4_CONFIG.stat().st_size,
        "v4_validation_report_json_bytes": V4_VAL_REPORT.stat().st_size,
    }
    print("[Artifact Sizes]", json.dumps(file_sizes, indent=2), flush=True)

    # 1. Load required S1 order from dataset/test/test_source1.tsv
    print("\n[Step 1] Loading test_source1.tsv S1 entity order & valid S2/S3 target IDs...", flush=True)
    t_load = time.time()
    expected_s1_order = []
    with open(TEST_S1, "r", encoding="utf-8", buffering=4 << 20) as f:
        next(f, None)
        for line in f:
            if line.strip():
                expected_s1_order.append(line.split("\t", 1)[0].strip())
    expected_s1_set = set(expected_s1_order)

    valid_targets = set()
    for src in (TEST_S2, TEST_S3):
        with open(src, "r", encoding="utf-8", buffering=8 << 20) as f:
            next(f, None)
            for line in f:
                if line.strip():
                    valid_targets.add(line.split("\t", 1)[0].strip())

    print(
        f"  Loaded {len(expected_s1_order):,} test S1 entities and {len(valid_targets):,} valid S2/S3 target IDs ({time.time()-t_load:.1f}s)",
        flush=True,
    )

    # 2. Simultaneous streaming audit of v4_matching_results.tsv and v4_candidate_pairs.tsv
    print("\n[Step 2] Streaming audit of v4_matching_results.tsv & v4_candidate_pairs.tsv...", flush=True)
    t_stream = time.time()

    matching_rows = 0
    candidate_rows = 0
    seen_match_s1 = set()
    seen_cand_s1 = set()
    dup_s1_in_matching = 0
    dup_s1_in_candidate = 0
    s1_order_mismatches_vs_test_s1 = 0
    s1_order_mismatches_match_vs_cand = 0

    intra_row_dup_matching = 0
    intra_row_dup_candidate = 0
    self_matches_matching = 0
    self_matches_candidate = 0
    invalid_target_ids_matching = 0
    subset_violations = 0
    malformed_rows_matching = 0
    malformed_rows_candidate = 0
    max_k_violations = 0
    zero_candidate_rows = 0

    predicted_links = 0
    s2_predicted_links = 0
    s3_predicted_links = 0
    total_candidate_pairs = 0
    non_empty_s1 = 0
    empty_s1 = 0

    target_multiplicity: Counter = Counter()
    match_count_dist: Counter = Counter()

    with (
        open(V4_MATCHING, "r", encoding="utf-8", buffering=16 << 20) as fm,
        open(V4_CANDIDATE, "r", encoding="utf-8", buffering=16 << 20) as fc,
    ):
        hm = fm.readline()
        hc = fc.readline()
        assert hm == "source1_entity_id\tmatched_entity_ids\n", f"Bad matching header: {hm!r}"
        assert hc == "source1_entity_id\tcandidate_entity_ids\n", f"Bad candidate header: {hc!r}"

        while True:
            lm = fm.readline()
            lc = fc.readline()
            if not lm and not lc:
                break
            if not lm or not lc:
                if lm:
                    matching_rows += 1
                    malformed_rows_candidate += 1
                if lc:
                    candidate_rows += 1
                    malformed_rows_matching += 1
                continue

            if not lm.endswith("\n"):
                malformed_rows_matching += 1
            if not lc.endswith("\n"):
                malformed_rows_candidate += 1

            sm, tab_m, rm = lm.rstrip("\r\n").partition("\t")
            sc, tab_c, rc = lc.rstrip("\r\n").partition("\t")

            if not tab_m or not sm:
                malformed_rows_matching += 1
            if not tab_c or not sc:
                malformed_rows_candidate += 1

            idx = matching_rows
            matching_rows += 1
            candidate_rows += 1

            if sm in seen_match_s1:
                dup_s1_in_matching += 1
            seen_match_s1.add(sm)

            if sc in seen_cand_s1:
                dup_s1_in_candidate += 1
            seen_cand_s1.add(sc)

            if sm != sc:
                s1_order_mismatches_match_vs_cand += 1
            if idx < len(expected_s1_order) and sm != expected_s1_order[idx]:
                s1_order_mismatches_vs_test_s1 += 1

            cids = rc.split(",") if rc else []
            if not cids:
                zero_candidate_rows += 1
                cset = set()
            else:
                cset = set(cids)
                total_candidate_pairs += len(cids)
                if len(cset) != len(cids):
                    intra_row_dup_candidate += 1
                for cid in cids:
                    if cid.startswith("S1-") or cid == sc:
                        self_matches_candidate += 1

            mids = rm.split(",") if rm else []
            k = len(mids)
            match_count_dist[k] += 1
            if k == 0:
                empty_s1 += 1
            else:
                non_empty_s1 += 1
                predicted_links += k
                if k > 6:
                    max_k_violations += 1
                mset = set(mids)
                if len(mset) != k:
                    intra_row_dup_matching += 1
                for mid in mids:
                    target_multiplicity[mid] += 1
                    if mid.startswith("S1-") or mid == sm:
                        self_matches_matching += 1
                    elif mid not in valid_targets:
                        invalid_target_ids_matching += 1
                    if mid not in cset:
                        subset_violations += 1
                    if mid.startswith("S2-"):
                        s2_predicted_links += 1
                    elif mid.startswith("S3-"):
                        s3_predicted_links += 1

    missing_s1 = len(expected_s1_set - seen_match_s1)
    extra_s1 = len(seen_match_s1 - expected_s1_set)
    distinct_targets = len(target_multiplicity)
    targets_with_mult_gt_1 = sum(1 for c in target_multiplicity.values() if c > 1)
    duplicate_target_assignments = sum(c - 1 for c in target_multiplicity.values() if c > 1)
    max_target_mult = max(target_multiplicity.values()) if target_multiplicity else 0

    print(f"  Completed streaming 10-point audit in {time.time()-t_stream:.1f}s", flush=True)

    # 3. Run official utils/validate_submission.py
    print("\n[Step 3] Running official utils/validate_submission.py --check-ids on v4_matching_results.tsv...", flush=True)
    cmd_check_ids = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching",
        str(V4_MATCHING),
        "--candidate",
        "NONE",
        "--test-dir",
        str(TEST_DIR),
        "--check-ids",
    ]
    proc_check_ids = subprocess.run(cmd_check_ids, capture_output=True, text=True)
    print(proc_check_ids.stdout, flush=True)

    errors_list = []
    if matching_rows != 1_732_544:
        errors_list.append(f"matching_rows={matching_rows} != 1732544")
    if candidate_rows != 1_732_544:
        errors_list.append(f"candidate_rows={candidate_rows} != 1732544")
    if missing_s1 != 0 or extra_s1 != 0 or dup_s1_in_matching != 0 or dup_s1_in_candidate != 0:
        errors_list.append("S1 coverage/uniqueness mismatch")
    if intra_row_dup_matching != 0 or intra_row_dup_candidate != 0:
        errors_list.append("Intra-row duplicate target IDs detected")
    if self_matches_matching != 0 or self_matches_candidate != 0:
        errors_list.append("Self-matches detected")
    if invalid_target_ids_matching != 0:
        errors_list.append(f"Invalid S2/S3 target IDs: {invalid_target_ids_matching}")
    if subset_violations != 0:
        errors_list.append(f"Subset violations: {subset_violations}")
    if s1_order_mismatches_match_vs_cand != 0 or s1_order_mismatches_vs_test_s1 != 0:
        errors_list.append("S1 row-order mismatch")
    if malformed_rows_matching != 0 or malformed_rows_candidate != 0:
        errors_list.append("Malformed/truncated rows detected")
    if duplicate_target_assignments != 0:
        errors_list.append(f"Duplicate target assignments across S1s: {duplicate_target_assignments}")
    if proc_check_ids.returncode != 0:
        errors_list.append(f"validate_submission.py returned non-zero exit code: {proc_check_ids.returncode}")

    audit_report = {
        "validator_result": "PASS" if len(errors_list) == 0 else "FAIL",
        "validator_exit_code": int(proc_check_ids.returncode),
        "validator_stdout": proc_check_ids.stdout.strip(),
        "matching_row_count": matching_rows,
        "candidate_row_count": candidate_rows,
        "every_s1_appears_once": (
            matching_rows == 1_732_544
            and missing_s1 == 0
            and extra_s1 == 0
            and dup_s1_in_matching == 0
            and dup_s1_in_candidate == 0
        ),
        "intra_row_duplicate_targets_matching": intra_row_dup_matching,
        "intra_row_duplicate_targets_candidate": intra_row_dup_candidate,
        "self_matches_matching": self_matches_matching,
        "self_matches_candidate": self_matches_candidate,
        "invalid_test_s2_s3_ids": invalid_target_ids_matching,
        "subset_violations_not_in_candidate_pairs": subset_violations,
        "s1_row_order_mismatches_match_vs_cand": s1_order_mismatches_match_vs_cand,
        "s1_row_order_mismatches_vs_test_source1": s1_order_mismatches_vs_test_s1,
        "malformed_or_truncated_rows_matching": malformed_rows_matching,
        "malformed_or_truncated_rows_candidate": malformed_rows_candidate,
        "max_k_violations": max_k_violations,
        "zero_candidate_rows": zero_candidate_rows,
        "predicted_links": predicted_links,
        "s2_predicted_links": s2_predicted_links,
        "s3_predicted_links": s3_predicted_links,
        "distinct_target_count": distinct_targets,
        "max_target_multiplicity": max_target_mult,
        "targets_claimed_by_multiple_s1": targets_with_mult_gt_1,
        "duplicate_target_assignments": duplicate_target_assignments,
        "non_empty_s1_count": non_empty_s1,
        "empty_s1_count": empty_s1,
        "total_candidate_pairs": total_candidate_pairs,
        "match_count_distribution": {str(k): int(v) for k, v in sorted(match_count_dist.items())},
        "file_sizes": file_sizes,
        "errors": errors_list,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    with open(FINAL_AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print("\n=== 10-POINT FINAL V4 AUDIT SUMMARY ===", flush=True)
    print(json.dumps(audit_report, indent=2), flush=True)
    assert len(errors_list) == 0, f"Audit failed with errors: {errors_list}"

    # 4. Stage & Build Final Official Submission ZIP (experiments/v4/v4_submission.zip)
    print("\n[Step 4] Preparing official submission staging directory & building experiments/v4/v4_submission.zip...", flush=True)
    t_zip = time.time()

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    (STAGING_DIR / "output").mkdir(parents=True, exist_ok=True)
    (STAGING_DIR / "code" / "business_entity_resolution" / "src").mkdir(parents=True, exist_ok=True)
    (STAGING_DIR / "code" / "business_entity_resolution" / "models").mkdir(parents=True, exist_ok=True)

    # Link output/matching_results.tsv and output/candidate_pairs.tsv
    _hardlink_or_copy(V4_MATCHING, STAGING_DIR / "output" / "matching_results.tsv")
    _hardlink_or_copy(V4_CANDIDATE, STAGING_DIR / "output" / "candidate_pairs.tsv")

    # Copy base code package, V3 baseline modules, and V4 modules + models
    _hardlink_or_copy(Path("code/__init__.py"), STAGING_DIR / "code" / "__init__.py")
    _hardlink_or_copy(
        Path("code/business_entity_resolution/__init__.py"),
        STAGING_DIR / "code" / "business_entity_resolution" / "__init__.py",
    )
    _hardlink_or_copy(
        Path("code/business_entity_resolution/requirements.txt"),
        STAGING_DIR / "code" / "business_entity_resolution" / "requirements.txt",
    )
    _hardlink_or_copy(
        Path("experiments/v3/submission_staging/code/business_entity_resolution/README.md"),
        STAGING_DIR / "code" / "business_entity_resolution" / "README.md",
    )
    _hardlink_or_copy(
        Path("experiments/v3/submission_staging/Documentation_template.md"),
        STAGING_DIR / "Documentation_template.md",
    )

    # Copy all src files from V3 staging + V4 src
    v3_stg_src = Path("experiments/v3/submission_staging/code/business_entity_resolution/src")
    for py_file in v3_stg_src.glob("*.py"):
        _hardlink_or_copy(py_file, STAGING_DIR / "code" / "business_entity_resolution" / "src" / py_file.name)
    for py_file in (V4_ROOT / "src").glob("*.py"):
        _hardlink_or_copy(py_file, STAGING_DIR / "code" / "business_entity_resolution" / "src" / py_file.name)

    # Copy V4 model & config (plus V3 baseline ranker) into models/
    _hardlink_or_copy(V4_MODEL, STAGING_DIR / "code" / "business_entity_resolution" / "models" / "v4_model.joblib")
    _hardlink_or_copy(V4_CONFIG, STAGING_DIR / "code" / "business_entity_resolution" / "models" / "v4_config.json")
    _hardlink_or_copy(
        V4_VAL_REPORT,
        STAGING_DIR / "code" / "business_entity_resolution" / "models" / "v4_validation_report.json",
    )
    _hardlink_or_copy(
        Path("experiments/v3/models/v3_model_c_lambdarank.joblib"),
        STAGING_DIR / "code" / "business_entity_resolution" / "models" / "v3_model_c_lambdarank.joblib",
    )

    if SUBMISSION_ZIP.exists():
        SUBMISSION_ZIP.unlink()

    with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True, compresslevel=1) as zf:
        for file_path in sorted(STAGING_DIR.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(STAGING_DIR).as_posix()
                print(f"  Adding {arcname} ({file_path.stat().st_size / (1024*1024):.2f} MB)...", flush=True)
                zf.write(file_path, arcname=arcname)

    zip_mb = round(SUBMISSION_ZIP.stat().st_size / (1024 * 1024), 2)
    audit_report["submission_zip"] = {
        "path": str(SUBMISSION_ZIP),
        "size_mb": zip_mb,
        "size_gb": round(zip_mb / 1024.0, 3),
        "zip_elapsed_sec": round(time.time() - t_zip, 2),
    }
    with open(FINAL_AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print(f"\nCreated {SUBMISSION_ZIP} ({zip_mb:.2f} MB / {zip_mb/1024.0:.3f} GB) in {time.time()-t_zip:.1f}s", flush=True)


if __name__ == "__main__":
    main()
