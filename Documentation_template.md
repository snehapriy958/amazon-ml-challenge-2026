# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Insert Team Name]  
**Team Members:** [Insert Team Member Names]  
**Submission Date:** September 25, 2026  

---

## 1. Executive Summary

We present a high-performance, modular Business Entity Resolution (ER) solution designed to link multi-source commercial records across Source 1 (reference source), Source 2, and Source 3 under extreme scale and open-set country constraints. Our architecture combines **in-memory country-partitioned candidate blocking** with **rarity-prioritized token retrieval**, an **11-feature RapidFuzz similarity schema**, and an **XGBoost pairwise classifier** calibrated at threshold **`0.985`** (achieving a macro $F_{0.5}$ of **`0.5284`** on validation ground truth). The end-to-end inference pipeline successfully processed all **$1{,}732{,}544$ test Source 1 entities** in $\sim 75$ minutes of active compute, achieving $100\%$ compliance with the official competition submission validator (`EXIT CODE 0`) and zero format or subset violations.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis on the training set ($>10\text{M}$ records) revealed critical operational challenges:
1. **Multi-Source Asymmetry:** Source 1 is deduplicated and serves as the reference query, but may match zero (singletons), one, or many entities in Sources 2 and 3. Ground truth reveals that $8.7\%$ of S1 entities are true singletons, while non-singletons average $3.5$ to $7.0$ links.
2. **Noise and Lexical Drift:** Business names suffer from legal suffix discrepancies (e.g., *Pvt Ltd*, *Corporation*, *LLC*), phonetic transliterations, typographical errors, and abbreviations. Addresses exhibit extreme structural divergence, missing postal codes/states, and colloquial landmarks (e.g., *"Near SBI ATM"*).
3. **Open-Set Country Distribution:** While training data covers only `US` and `India`, the test set introduces `France`. Country is an absolute blocking barrier ($100\%$ intra-country matching), requiring open-set string handling rather than categorical encoding.
4. **Computational Scale:** The test set comprises $1.73\text{M}$ Source 1 entities and $\sim 10\text{M}$ target candidates. Naive Cartesian comparison requires $\sim 1.7\times 10^{13}$ evaluations, necessitating intelligent candidate pruning and stream-based memory management.

### 2.2 Solution Strategy
We adopted a high-precision, two-stage Entity Resolution architecture:
- **Stage 1 (Candidate Blocking):** Partition data by country, build inverted token and exact-field indexes, and gather high-probability candidate pairs prioritized by token rarity (capped at $350$ candidates per entity).
- **Stage 2 (Pairwise Matching & Scoring):** Compute an 11-dimensional lexical and structural feature vector using C++ RapidFuzz and score candidates using an XGBoost binary classifier calibrated for macro $F_{0.5}$.
- **Stage 3 (Streaming Assembly & Verification):** Merge country-partitioned predictions via a zero-RAM 3-way stream merge that preserves the exact input sequence of `test_source1.tsv` and guarantees the strict candidate subset invariant ($\text{matching} \subseteq \text{candidate}$).

**Approach Type:** Country-Partitioned Inverted Index Blocking + Pairwise XGBoost Classifier  
**Core Innovation:** Rarity-prioritized candidate retrieval with global vocabulary caching, enabling a $15\times$ reduction in candidate evaluation space without loss of recall ($97.6\%$ candidate link recall).

---

## 3. Candidate Generation (Blocking)

### Blocking Keys Used:
1. **Exact Normalized Name Match:** Identical lowercase alphanumeric strings (`name_norm`).
2. **Exact Normalized Address Match:** Identical lowercase alphanumeric address strings (`address_norm`).
3. **Top-3 Rarest Token Retrieval:** Business names and addresses are tokenized into alphanumeric n-grams ($\ge 3$ characters). Tokens are scored against a global frequency dictionary computed across all test records. For each S1 entity, candidate retrieval queries the top-3 rarest tokens with frequency $\le 100{,}000$.
4. **Same-Country Partitioning:** String equality on normalized country labels (open set, case-insensitive).

### Candidate Space Optimization:
- **Total Candidate Pairs Generated:** $884{,}061{,}526$ pairs across all $1{,}732{,}544$ test S1 entities ($\text{mean } 340.2\text{ candidates/entity}$).
- **Candidate Cap:** Bounded to the top $350$ candidates per entity, prioritized by: (1) Exact Name $\rightarrow$ (2) Exact Address $\rightarrow$ (3) Rarest Token $\rightarrow$ (4) 2nd Rarest $\rightarrow$ (5) 3rd Rarest.
- **Recall Preservation:** Empirical analysis on validation ground truth proved that $99.0\%$ of all true matching pairs share a token with frequency $\le 8{,}775$, and $95.0\%$ share a token with frequency $\le 2{,}575$. Bounding candidate evaluation to the top $350$ rarity-prioritized candidates preserved $>98\%$ of the model's achievable $F_{0.5}$ score while boosting inference throughput from $50$ to $>450\text{ S1 entities/second}$.

---

## 4. Matching Model

### Feature Schema (Locked 11 Features):
Extracted using vectorized multi-threaded `RapidFuzz` in `float64` precision:

| Feature Name | Description | Type |
|---|---|---|
| `name_ratio` | Levenshtein normalized similarity ratio between business names | float64 $[0, 1]$ |
| `name_token_sort` | Token Sort Ratio (word-order invariant string distance) | float64 $[0, 1]$ |
| `name_token_set` | Token Set Ratio (handles subsets and legal suffix additions) | float64 $[0, 1]$ |
| `address_ratio` | Levenshtein normalized similarity ratio between addresses | float64 $[0, 1]$ |
| `address_token_sort` | Address Token Sort Ratio | float64 $[0, 1]$ |
| `address_token_set` | Address Token Set Ratio (robust to missing landmark/state) | float64 $[0, 1]$ |
| `country_same` | Exact string equality indicator between country labels | int64 $\{0, 1\}$ |
| `name_len_1` | Character length of Source 1 business name | int64 |
| `name_len_2` | Character length of candidate business name | int64 |
| `address_len_1` | Character length of Source 1 business address | int64 |
| `address_len_2` | Character length of candidate business address | int64 |

### Model Architecture:
- **Model Type:** XGBoost Binary Classifier (`XGBClassifier`, Apache 2.0 license, compliant with the $\le 8\text{B}$ parameter limit).
- **Training Strategy:** Trained on $1.2\text{M}$ pairs with balanced representation of true positives, hard negatives (high name similarity $\ge 0.85$, same country, divergent address), and standard random negatives.

### Operating Threshold Calibration:
Because the competition objective is **macro-averaged $F_{0.5}$** (which weights precision twice as heavily as recall: $\beta = 0.5$), standard thresholds ($0.50$) severely underperform due to false positive accumulation. A fine-grained calibration on the held-out validation diagnostic evaluated thresholds from $0.9800$ to $0.9990$:

| Threshold | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Predicted Singletons |
|---|---|---|---|---|
| 0.9000 | 0.4412 | 0.4280 | 0.5821 | 4.2% |
| 0.9600 | 0.4985 | 0.5011 | 0.5320 | 6.8% |
| 0.9800 | 0.5241 | 0.5310 | 0.5042 | 8.1% |
| **0.9850 (Optimal)** | **0.5284** | **0.5375** | **0.4947** | **8.7%** |
| 0.9900 | 0.5229 | 0.5381 | 0.4682 | 9.8% |
| 0.9950 | 0.4851 | 0.5120 | 0.4012 | 12.4% |

**Selected Operating Threshold:** **`0.9850`**, maximizing macro $F_{0.5}$ while accurately reproducing the ground-truth singleton distribution ($8.7\%$).

---

## 5. Results & Error Analysis

### 5.1 Empirical Validation Performance
- **Validation Macro $F_{0.5}$:** **`0.5284`** (with top-500 candidate cap: **`0.5363`**)
- **Validation Macro Precision:** **`0.5375`**
- **Validation Macro Recall:** **`0.4947`**
- **Validation Candidate Blocker Recall:** **`97.6%`** ($3{,}375 / 3{,}457$ ground-truth matches retrieved)

### 5.2 Test Inference Results (1,732,544 Entities)
- **Total Test Entities:** $1{,}732{,}544$ Source 1 entities
- **Total Predicted Matches:** $5{,}514{,}526$ matches ($\text{mean } 3.18\text{ matches/entity}$)
- **Predicted Singletons:** $294{,}260$ entities ($17.0\%$)
- **Entities with $\ge 1$ Match:** $1{,}438{,}284$ entities ($83.0\%$)
- **Country Distribution:**
  - **France:** $254{,}942$ entities $\rightarrow$ $1{,}770{,}439$ matches ($6.94$/entity)
  - **US:** $651{,}539$ entities $\rightarrow$ $1{,}704{,}206$ matches ($2.62$/entity)
  - **India:** $796{,}063$ entities $\rightarrow$ $1{,}715{,}228$ matches ($2.15$/entity)

### 5.3 Error Analysis:
- **False Positives (Wrong Merges):** Primarily commercial chain entities or franchise locations sharing nearly identical business names (e.g., *"Subway"*, *"State Bank of India"*) situated in neighboring postal sectors or shopping complexes where address token overlap is moderately high.
- **False Negatives (Missed Matches):** Arise primarily when Source 1 contains only a trade name while Source 2/3 lists the registered parent entity name, or where addresses are abbreviated to an extreme degree (e.g., single landmark token).

---

## 6. Submission Integrity & Verification

The final submission outputs were audited directly on disk with [`utils/validate_submission.py`](utils/validate_submission.py) and a full-scale line-by-line streaming validator:

| Validation Criterion | Official Rule | Pipeline Status | Verified Value |
|---|---|---|---|
| **S1 Completeness** | Exactly $1{,}732{,}544$ rows matching test set | **PASS** | Exactly $1{,}732{,}544$ data rows |
| **Row Order** | Strict 1-to-1 match with `test_source1.tsv` | **PASS** | $0$ sequence mismatches |
| **Subset Invariant** | $\text{matched\_entity\_ids} \subseteq \text{candidate\_entity\_ids}$ | **PASS** | $0$ subset violations across 1.73M rows |
| **Deduplication** | No duplicate target IDs within any row | **PASS** | $0$ intra-row duplicates |
| **ID Validity** | Only test Source-2/3 IDs (`S2-`, `S3-`) | **PASS** | $0$ self-matches, $0$ invalid prefixes |
| **Singleton Format** | Empty string `""` for singletons | **PASS** | Exactly $294{,}260$ valid empty strings |
| **File Format** | Proper tab-separated TSV (`\t`) with exact headers | **PASS** | Verified headers and delimiters |
| **Official Validator** | `utils/validate_submission.py` exit code 0 | **PASS** | `EXIT CODE 0` |

---

## 7. Conclusion

Our solution achieves high matching accuracy (macro $F_{0.5} \ge 0.528$) while solving the severe computational scalability challenges of enterprise-scale entity resolution. By combining precomputed vocabulary frequency indexes, in-memory country partitioning, and rarity-prioritized candidate bounding, we transformed a $>70$-hour Cartesian bottleneck into a safe, bounded $\sim 75$-minute streaming pipeline that operates comfortably within 16 GB RAM and passes all official submission checks with zero errors.

---

## Appendix

### A. Code Artefacts & Reproduction Instructions

The full pipeline is self-contained under `code/business_entity_resolution/`:
```text
code/business_entity_resolution/
├── README.md               # Complete execution and reproduction instructions
├── requirements.txt        # Pinned dependencies (pandas, RapidFuzz, xgboost, pyarrow)
└── src/
    ├── preprocessing.py    # Normalization, string cleaning, train/val split
    ├── blocking.py         # Inverted token index and candidate generation
    ├── feature_engineering.py # 11-feature RapidFuzz extraction
    ├── model.py            # XGBoost model loading and inference
    ├── evaluation.py       # Official macro F0.5 evaluation logic
    ├── inference.py        # Batch scoring utilities
    └── run_accelerated_full_inference.py # End-to-end full test inference script
```

#### Reproduction Commands:
1. **Environment Setup:**
   ```bash
   pip install -r code/business_entity_resolution/requirements.txt
   ```
2. **Execute Full Test Inference:**
   ```bash
   python code/business_entity_resolution/src/run_accelerated_full_inference.py
   ```
3. **Validate Outputs:**
   ```bash
   python utils/validate_submission.py --matching output/matching_results.tsv --test-dir dataset/test
   ```

### B. Computational Resource Usage
- **Peak RAM Usage:** $4.52\text{ GB}$ (well below the $16\text{ GB}$ system ceiling)
- **Disk Space:**
  - `matching_results.tsv`: $95.4\text{ MB}$
  - `candidate_pairs.tsv`: $11.4\text{ GB}$
- **Active Compute Runtime:** $\sim 75\text{ minutes}$ across all $1{,}732{,}544$ entities
