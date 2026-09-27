# Amazon ML Challenge 2026 — Business Entity Resolution V3 Master Report

## 1. Executive Summary & Progression (`V1` $\to$ `V2-C` $\to$ `V3`)

| Pipeline Stage | Public Leaderboard | `DEV_3000` Macro F0.5 | `INTERNAL_CHECK_1000` Macro F0.5 | `FINAL_HOLDOUT_1000` Macro F0.5 | Candidate Recall (Micro) | $S_2$ Link Recall | $S_3$ Link Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| **V1 Baseline (Frozen)** | `0.517000` | — | — | — | `~66%` | — | — |
| **V2-C Scaled (Frozen)** | `0.562685` | `0.5711` (`@0.900`) | `0.5660` (`@0.900`) | `0.5481` (`@0.900`) | `66.15%` | `20.53%–22.63%` | `64.75%–67.75%` |
| **V3 Production Pipeline (Frozen)** | *(Not yet run on test)* | **`0.9070`** | **`0.9215`** | **`0.9140`** | **`97.16%–97.70%`** | **`82.70%–84.21%`** | **`82.78%–83.94%`** |

### Why V3 Jump-Shifted Macro F0.5 from `~0.56` to `0.9140`:
1. **Eliminated the 66.15% Candidate Retrieval Ceiling (Phase V3-A):**
   - Replaced V2-C's 3-token blocker (`66.15%` recall) with a **Two-Tier 4-Channel Anchor + Verification Blocker** (`Top-750` with fair $S_2/S_3$ interleaving), lifting candidate recall to **`97.16%`** on `DEV_3000`, **`97.18%`** on `INTERNAL_CHECK_1000`, and **`97.70%`** on `FINAL_HOLDOUT_1000` at **`10.1–10.9 queries/sec`**.
2. **Eliminated the $S_2$ Case & Script Blindness (Phase V3-B):**
   - V2-C computed `RapidFuzz.WRatio` on raw strings where $S_2$ was predominantly uppercase (`"31 EAGLE PARK CIRCLE"`) while $S_1$ was title-case (`"31 Eagle Park Cir"`), causing $S_2$ link recall to collapse to **`20.53%–22.63%`**.
   - V3-B introduced `normalization_v3.py` (case-folding, Latin + 9-script ISCII Indic transliteration, legal suffix stripping, country-aware US/India/France postal/building/unit/state extraction) and `features_v3.py` (`88` features including `name_x_addr_sim`, `core_name_x_norm_addr_set`, `building_number_conflict`, `state_region_conflict`, `name_rare_token_idf_overlap`, and phonetic features), raising $S_2$ link recall to **`82.70%–84.21%`** at **`97.0%–98.7%` precision**!
3. **Eliminated $S_3$ Negative Imbalance & Leakage (Phase V3-C):**
   - Constructed `1,246,103` entity-disjoint training pairs (`471,551` V3-retrievable GT positives + `774,552` negatives with **exact `50.0%` $S_2$ / `50.0%` $S_3$ balance** and `599,151` numeric-conflict / `380,612` branch-confusion structural negatives) and verified **`0` overlap** with all validation and holdout pools.
4. **Group-Aware Ranking + Entity-Level Decision Policy (Phase V3-D):**
   - `C_LightGBM_LambdaRank` grouped by `source1_entity_id` with Platt calibration and Entity-Level Policy 5 (`t_us = 0.995, t_in = 0.995, max_k = 6`) achieved **`0.9070`** on `DEV_3000`, **`0.9215`** on `INTERNAL_CHECK_1000`, and **`0.9140`** on `FINAL_HOLDOUT_1000`.

---

## 2. Validation Firewall & Artifact Integrity Audit

- **Untouched Frozen Baselines Verified:**
  - `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `output/checkpoint_full_inference.json` — **UNTOUCHED**
  - `models/xgboost_matcher.joblib`, `training_features_chunks/` — **UNTOUCHED**
  - `experiments/v2/models/xgboost_matcher_v2_c.joblib`, `experiments/v2/output/*` — **UNTOUCHED**
- **V3 Artifacts Created (`experiments/v3/`):**
  - `experiments/v3/pools/`: `dev_3000_s1.tsv`, `internal_check_1000_s1.tsv`, `final_holdout_1000_s1.tsv`, `pools_manifest.json`
  - `experiments/v3/src/`: `create_validation_pools.py`, `normalization_v3.py`, `features_v3.py`, `retrieval_v3.py`, `build_v3_c_training_data.py`, `run_v3_d_bakeoff.py`, `evaluate_frozen_v3_on_check_and_holdout.py`
  - `experiments/v3/retrieval/`: `dev_3000_retrieval_metrics.json`, `dev_3000_retrieval_bundle.joblib`, `check_and_holdout_2000_retrieval_bundle.joblib`
  - `experiments/v3/data/`: `v3_c_train_features.parquet`, `dev_3000_v3_features.parquet`, `internal_check_1000_v3_features.parquet`, `final_holdout_1000_v3_features.parquet`
  - `experiments/v3/models/`: `v3_model_a_xgboost.joblib`, `v3_model_b_lgbm_binary.joblib`, `v3_model_c_lambdarank.joblib`, `v3_model_d_rank_xendcg.joblib`
  - `experiments/v3/results/`: `v3_c_training_stats.json`, `v3_d_bakeoff_metrics.json`, `v3_internal_and_holdout_metrics.json`, `V3_C_TRAINING_REPORT.md`, `V3_MODEL_BAKEOFF.md`, `V3_MASTER_REPORT.md`

---

## 3. Main Remaining Bottleneck & GO / NO-GO Recommendation

### 3.1 Main Remaining Bottleneck
1. **End-to-End Inference Throughput on the `1.73M` Test Set:**
   - Accuracy is no longer the bottleneck: `FINAL_HOLDOUT_1000` Macro F0.5 is **`0.9140`** (with `A_XGBoost_Binary = 0.9106` and `B_LightGBM_Binary = 0.9120`), well above the `0.80` target and `0.90` stretch target.
   - The remaining engineering bottleneck for full test inference (`1,732,544` $S_1$ queries) is **feature extraction volume** if all `750` candidates per $S_1$ (`1.30B` pairs) were featurized with all 88 features.
   - **Recommended Two-Stage Test Inference Acceleration (when approved for test inference):**
     - **Stage 1 (Retrieval Rank Cutoff / Cheap Pre-Filter):** In `DEV_3000`, `99.1%` of true positives predicted by the model rank in the **Top-120** of the V3 retrieval priority score (or have an exact channel hit). Restricting full 88-feature extraction at test time to `Top-100..150` retrieval candidates + all Channel-1 exact matches (or using `multiprocessing` across 8–12 workers) reduces test pair volume by **`6x–7.5x`** without sacrificing precision.

### 3.2 GO / NO-GO Recommendation
- **RECOMMENDATION: STRONG GO FOR V3.**
  - Every validation gate (`DEV_3000 = 0.9070`, `INTERNAL_CHECK_1000 = 0.9215`, `FINAL_HOLDOUT_1000 = 0.9140`) has been passed by a wide margin (`+0.35` to `+0.37` Macro F0.5 over V2-C).
