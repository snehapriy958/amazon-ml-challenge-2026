# Amazon ML Challenge 2026 — Business Entity Resolution

## 1. Problem

The task is business entity resolution: identifying corresponding entity records across datasets while handling noisy names, addresses, transliteration, and other record-level variations.

## 2. Methodology

The V5 solution uses a multi-stage entity-resolution pipeline:

1. Entity normalization and preprocessing.
2. High-recall candidate generation and retrieval.
3. Feature engineering for candidate pairs.
4. Candidate ranking / matching.
5. Final inference and submission validation.

## 3. Candidate Generation / Blocking

The retrieval stage reduces the possible entity pairs to a tractable candidate set while aiming to preserve high recall.

The V5 retrieval implementation uses normalized entity information and blocking/retrieval signals. Transliteration support is included to handle variations across writing systems.

## 4. Feature Engineering

The V5 feature pipeline computes multiple matching signals for candidate entity pairs, including normalized textual and record-level similarity features.

The implementation is provided in `features_v5.py`.

## 5. Model and Inference

The V5 inference pipeline ranks candidate entity pairs and applies the matching logic used to generate the final predictions.

The full-test inference implementation is provided in `run_full_test_inference_v5.py`.

## 6. Validation and Audit

The solution includes final submission validation and error-analysis tooling.

- `run_final_v5_submission_audit.py`
- `run_v5_error_analysis.py`

## 7. Submitted Output

The frozen V5 prediction file is:

`output/matching_results.tsv`

The source code in this repository corresponds to the V5 entity-resolution pipeline used for the submitted solution.
