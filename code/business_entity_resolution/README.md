# Business Entity Resolution Pipeline

This package provides a modular, reproducible, and compliant solution for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

---

## 1. Environment & Setup

- **Python:** 3.10+ (tested on Python 3.13)
- **Key Dependencies:** Listed in `requirements.txt`

To set up the environment:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 2. Architecture & Modules

The pipeline lives under `code/business_entity_resolution/src/`:

1. **`preprocessing.py`**:
   - Tab-separated TSV loader with strict string types.
   - Verified name & address normalizers (`normalize_name`, `normalize_address`).
   - Deterministic 80/20 train/validation split by hashing `source1_entity_id`.

2. **`blocking.py`**:
   - Tokenizer (`[a-z0-9]+`).
   - Exact normalized name and address index structures.
   - Inverted token indexes with global frequency counts.
   - Top-3 rarest token candidate retrieval with open-set same-country filtering.
   - Hard negative sampler strictly excluding all ground truth matches.

3. **`feature_engineering.py`**:
   - Exact 11-feature locked schema:
     - `name_ratio`
     - `name_token_sort`
     - `name_token_set`
     - `address_ratio`
     - `address_token_sort`
     - `address_token_set`
     - `country_same`
     - `name_len_1`
     - `name_len_2`
     - `address_len_1`
     - `address_len_2`
   - High-throughput pairwise similarity calculation via batched `RapidFuzz.process.cpdist` (`dtype=float64`, multi-threaded).

4. **`model.py`**:
   - XGBoost binary classifier (Apache 2.0 license, compliant with the $\le 8\text{B}$ parameter constraint).
   - Training on pre-generated parquet chunks.
   - Serialization and probability prediction.

5. **`evaluation.py`**:
   - Macro-averaged $F_{0.5}$ metric computation per Source 1 entity.
   - Explicit scoring for singletons (correct empty list = 1.0, incorrect match = 0.0).

6. **`inference.py`**:
   - Candidate pair generation and batched scoring.
   - Generation of official submission files: `matching_results.tsv` and `candidate_pairs.tsv`.
   - Strict format compliance: one row per S1 entity, subset invariant, comma-separated lists.

7. **`run_accelerated_full_inference.py`**:
   - Production high-throughput full test inference engine.
   - In-memory country partitioning (France $\rightarrow$ US $\rightarrow$ India) eliminating repeated disk scans.
   - Rarity-prioritized candidate bounding ($350$ pairs/entity).
   - Multi-threaded RapidFuzz float64 feature calculation + XGBoost scoring at threshold `0.985`.
   - Zero-RAM 3-way stream merge ensuring exact sequential order matching `test_source1.tsv`.

---

## 3. Running the Pipeline

From the project root:

### Model Training
```bash
python -m code.business_entity_resolution.src.main train \
    --chunks-dir training_features_chunks \
    --output-model models/xgboost_matcher.joblib
```

### Full-Scale Accelerated Test Inference (1,732,544 Entities)
```bash
python code/business_entity_resolution/src/run_accelerated_full_inference.py
```

### Submission Validation
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --test-dir dataset/test
```
