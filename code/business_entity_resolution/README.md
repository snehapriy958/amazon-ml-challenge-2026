# Amazon ML Challenge 2026 — Business Entity Resolution

This directory contains the source code used for the V5 entity-resolution solution.

## Pipeline

1. Candidate generation / retrieval
2. Feature engineering
3. Entity matching / ranking
4. Full-test inference
5. Submission validation and audit

## Main source files

- `retrieval_v5.py` — V5 candidate retrieval and blocking logic
- `features_v5.py` — V5 feature engineering
- `run_full_test_inference_v5.py` — full-test inference pipeline
- `run_final_v5_submission_audit.py` — final submission checks
- `transliteration_v5.py` — transliteration utilities
- `run_v5_error_analysis.py` — error-analysis tooling

The submitted prediction file is `output/matching_results.tsv`.
