# Phase V3-C: Entity-Disjoint Training Data Construction Report

## 1. Executive Summary
Phase V3-C constructed a strictly leakage-free, entity-disjoint training dataset (`experiments/v3/data/v3_c_train_features.parquet`) aligned with the V3 4-channel retrieval blocker and the 88 V3-B normalized/numeric/phonetic feature representation.

Every requirement of the V3-C training specification was verified programmatically (`experiments/v3/results/v3_c_training_stats.json`):
- **Total Training Pairs:** `1,246,103` (`106.49 MB` Parquet)
- **Unique Clean Training $S_1$ Entities:** `166,865`
- **Unique Candidate $S_2/S_3$ Entities:** `1,011,517`
- **Positive Pairs:** `471,551` (`37.84%`) — `228,132` $S_2$ positives (`48.38%`) and `243,419` $S_3$ positives (`51.62%`)
- **Negative Pairs:** `774,552` (`62.16%`) — **`387,276` $S_2$ negatives (`50.00%`) and `387,276` $S_3$ negatives (`50.00%`)**
- **Duplicate Pair Count:** `0`
- **GT Positive / Negative Overlap:** `0`
- **Validation & Holdout Leakage:** `0` overlap across all 6 validation/holdout/diagnostic sets (`DEV_3000`, `INTERNAL_CHECK_1000`, `FINAL_HOLDOUT_1000`, `DIAGNOSTIC_5000`, `val_1000`, and the official deterministic validation split).

---

## 2. Dataset Composition & Source Balance

| Metric | Count | Percentage / Share | Notes |
|---|---:|---:|---|
| **Total Training Pairs** | `1,246,103` | `100.00%` | Grouped by `source1_entity_id` |
| **Positive Pairs (`label = 1`)** | `471,551` | `37.84%` | V3 4-channel blocker retrievable GT links (`20` cross-country/zero-overlap noise links dropped) |
| **Negative Pairs (`label = 0`)** | `774,552` | `62.16%` | Strictly excluded against complete `7,638,365` GT link universe |
| **$S_2$ Positives** | `228,132` | `48.38%` of positives | Uppercase-heavy raw source normalized via V3-B |
| **$S_3$ Positives** | `243,419` | `51.62%` of positives | Mixed/title-case source normalized via V3-B |
| **$S_2$ Negatives** | `387,276` | **`50.00%` of negatives** | Upgraded from V2-C's `34.1%` $S_2$ negative share |
| **$S_3$ Negatives** | `387,276` | **`50.00%` of negatives** | Exact 1:1 source balance with $S_2$ negatives |
| **Unique Training $S_1$ Entities** | `166,865` | — | Entity-disjoint `train` split only |
| **Unique Target ($S_2 \cup S_3$) Entities** | `1,011,517` | — | Across US, India, and secondary locales |

---

## 3. Hard & Structural Negative Mining Breakdown

To address the dominant error modes diagnosed in V2-C (where $S_3$ negatives represented `65.9%` of training negatives and branch/neighbor/numeric-conflict negatives were under-represented), candidate negatives were scored with the frozen V2-C model and stratified by difficulty and structural confusion category:

### 3.1 Model-Difficulty Stratification (Frozen V2-C Score)
| Difficulty Tier | Criterion | Count | Share of Negatives |
|---|---|---:|---:|
| **Hard Negatives** | `v2c_prob >= 0.50` | `26,002` | `3.36%` |
| **Semi-Hard Negatives** | `0.10 <= v2c_prob < 0.50` | `147,688` | `19.07%` |
| **Retrieval-Hard / Anchor Negatives** | `v2c_prob < 0.10` (shared rare token / postal / building anchor) | `600,862` | `77.58%` |

### 3.2 Structural Confusion Categories Represented in Negatives
*(Note: A single negative pair can satisfy multiple structural confusion predicates.)*

| Structural Negative Category | Predicate Definition | Count in V3-C Negatives | Share of Negatives |
|---|---|---:|---:|
| **Conflicting Numeric Evidence** | `building_number_conflict == 1` or `postal_conflict == 1` | `599,151` | `77.35%` |
| **High Name Sim + Different Address** | `core_name_token_set >= 0.80` and `norm_addr_token_set < 0.50` | `380,612` | `49.14%` |
| **High Name Sim + Low Address Flag** | `high_name_low_addr == 1.0` (`core_name >= 0.88` and `norm_addr < 0.45`) | `366,844` | `47.36%` |
| **Same/Similar Name + Different Building/Street Number** | `core_name_token_set >= 0.75` and `building_number_conflict == 1` | `356,679` | `46.05%` |
| **Same Street + Different Business** | `street_token_jaccard >= 0.50` and `core_name_token_set < 0.50` | `69,479` | `8.97%` |
| **Same Postal Code + Different Business** | `postal_match == 1.0` and `core_name_token_set < 0.50` | `526` | `0.07%` |

---

## 4. Validation Firewall & Zero-Leakage Audit

| Firewall Check | Forbidden / Reference Pool Size | Overlap with V3-C Training $S_1$s | Status |
|---|---:|---:|---|
| `DEV_3000` (`experiments/v3/pools/dev_3000_s1.tsv`) | `3,000` | **`0`** | PASS |
| `INTERNAL_CHECK_1000` (`experiments/v3/pools/internal_check_1000_s1.tsv`) | `1,000` | **`0`** | PASS |
| `FINAL_HOLDOUT_1000` (`experiments/v3/pools/final_holdout_1000_s1.tsv`) | `1,000` | **`0`** | PASS |
| `DIAGNOSTIC_5000` (`experiments/v2/Switch/heldout_5000_s1.tsv`) | `5,000` | **`0`** | PASS |
| `val_1000` (`experiments/v2/val_1000_s1.tsv`) | `1,000` | **`0`** | PASS |
| Official Deterministic Validation Split (`split == 'val'`) | `11,000` | **`0`** | PASS |
| Complete GT Positive Pair Exclusion (`7,638,365` GT links) | `7,638,365` | **`0` GT pairs in negatives** | PASS |
| Duplicate `(source1_entity_id, candidate_entity_id)` rows | `1,246,103` | **`0` duplicates** | PASS |
