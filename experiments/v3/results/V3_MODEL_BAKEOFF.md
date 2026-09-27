# Phase V3-D: Controlled 4-Model Bakeoff & Firewall Evaluation Report

## 1. Executive Summary
Phase V3-D executed a controlled 4-model bakeoff on the exact same V3-C training dataset (`1,246,103` entity-disjoint pairs, `88` V3-B normalized/numeric/phonetic features) and evaluated all models on `DEV_3000` (`2,249,627` candidate pairs across `3,000` $S_1$ queries, `10,528` GT links).

Following the Validation Firewall protocol:
1. All feature ablations, model comparisons, threshold sweeps (`0.70` to `0.99`), and entity-level decision policy optimizations were performed **exclusively on `DEV_3000`**.
2. The winning candidate pipeline (**`C_LightGBM_LambdaRank` with Platt calibration + Entity-Level Policy 5 (`t_us = 0.995, t_in = 0.995, max_k = 6`)**, which achieved **`0.9070` Macro F0.5** on `DEV_3000` and **`0.9075`** at raw ranking threshold `4.617`) was **FROZEN**.
3. The frozen pipeline was evaluated **once** on `INTERNAL_CHECK_1000` (**`0.9215` Macro F0.5**) and **once** on `FINAL_HOLDOUT_1000` (**`0.9140` Macro F0.5**), exceeding both the `0.80` target and the `0.90` stretch target on every unseen validation and holdout pool.

---

## 2. Phonetic Feature Ablation on `DEV_3000`

Before training the full bakeoff suite, an ablation check was run on `DEV_3000` using `XGBClassifier(n_estimators=600, max_depth=7, learning_rate=0.06)`:

| Feature Set | Feature Count | Best `DEV_3000` Macro F0.5 | Best Threshold | Decision |
|---|---:|---:|---:|---|
| Core Normalized + Numeric + Interaction Features | `85` | `0.8954` (`0.895377`) | `0.99` | Baseline V3 Core |
| **Core + Phonetic Features (`phonetic_name_ratio`, `phonetic_name_token_jaccard`, `phonetic_exact_core_name`)** | **`88`** | **`0.8970` (`0.896988`)** | **`0.99`** | **Retained (`+0.0016` F0.5)** |

---

## 3. Controlled 4-Model Bakeoff on `DEV_3000` (`3,000` $S_1$s, `10,528` GT Links)

### 3.1 Summary Comparison Table on `DEV_3000`

| Model / Pipeline | Decision Mode | Threshold / Config | Macro F0.5 | Macro Prec | Macro Rec | Pred Pairs | TP | FP | FN | $S_2$ Prec | $S_2$ Rec | $S_3$ Prec | $S_3$ Rec | US F0.5 | India F0.5 | Empty-$S_1$ Acc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Frozen V2-C (11 features)** | Threshold-only | `0.9125` (best) | `0.5726` | `66.78%` | `43.81%` | `6,775` | `4,560` | `2,215` | `5,968` | `73.85%` | `20.53%` | `65.57%` | `64.75%` | `0.6164` | `0.5090` | `58.02%` |
| **Frozen V2-C (11 features)** | Threshold-only | `0.9000` (locked) | `0.5711` | `65.56%` | `45.54%` | `7,521` | `4,792` | `2,729` | `5,736` | `71.15%` | `22.55%` | `63.92%` | `67.13%` | `0.6143` | `0.5082` | `53.70%` |
| **Model A: `XGBoost Binary`** | Threshold-only | `0.9900` | `0.8970` | `94.58%` | `80.17%` | `8,651` | `8,378` | `273` | `2,150` | `98.35%` | `79.62%` | `95.46%` | `79.54%` | `0.9163` | `0.8690` | `95.06%` |
| **Model A: `XGBoost Binary`** | Entity Policy 5 | `t_us=0.975, t_in=0.990, max_k=6, rescue=0.18` | `0.9025` | `94.40%` | `82.23%` | `8,965` | `8,594` | `371` | `1,934` | `97.61%` | `81.78%` | `94.26%` | `81.49%` | `0.9221` | `0.8740` | `88.89%` |
| **Model B: `LightGBM Binary`** | Threshold-only | `0.9900` | `0.8991` | `94.21%` | `81.49%` | `8,872` | `8,538` | `334` | `1,990` | `97.85%` | `81.05%` | `94.77%` | `81.14%` | `0.9218` | `0.8661` | `93.21%` |
| **Model B: `LightGBM Binary`** | Entity Policy 5 | `t_us=0.990, t_in=0.993, max_k=8, rescue=0.18` | `0.9023` | `94.91%` | `80.93%` | `8,693` | `8,454` | `239` | `2,074` | `98.32%` | `80.15%` | `96.27%` | `80.44%` | `0.9239` | `0.8709` | `91.36%` |
| **Model C: `LightGBM LambdaRank`** | Threshold-only | `0.9900` (Platt) | `0.8939` | `92.04%` | `85.47%` | `9,857` | `9,011` | `846` | `1,517` | `93.85%` | `85.25%` | `89.25%` | `85.91%` | `0.9171` | `0.8603` | `87.65%` |
| **Model C: `LightGBM LambdaRank`** | Raw Score Thr | `raw >= 4.617` (p99.6) | **`0.9075`** | `94.75%` | `82.94%` | `8,999` | `8,688` | `311` | `1,840` | `97.63%` | `82.43%` | `95.54%` | `82.61%` | `0.9242` | `0.8834` | `95.06%` |
| **Model C: `LightGBM LambdaRank` (WINNER)** | **Entity Policy 5** | **`t_us=0.995, t_in=0.995, max_k=6`** | **`0.9070`** | **`94.33%`** | **`83.63%`** | **`9,127`** | **`8,739`** | **`388`** | **`1,789`** | **`97.03%`** | **`82.70%`** | **`94.58%`** | **`83.30%`** | **`0.9247`** | **`0.8813`** | **`93.21%`** |
| **Model D: `LightGBM rank_xendcg`** | Threshold-only | `0.9900` (Platt) | `0.8452` | `89.54%` | `76.48%` | `8,817` | `7,990` | `827` | `2,538` | `93.48%` | `74.73%` | `88.16%` | `76.99%` | `0.8781` | `0.7973` | `87.65%` |
| **Model D: `LightGBM rank_xendcg`** | Entity Policy 5 | `t_us=0.990, t_in=0.995, max_k=6` | `0.8469` | `90.26%` | `75.50%` | `8,568` | `7,874` | `694` | `2,654` | `94.45%` | `73.71%` | `89.84%` | `75.87%` | `0.8781` | `0.8017` | `88.89%` |

---

### 3.2 Full Threshold Sweep Across All 4 Models on `DEV_3000`

| Threshold | V2-C (11 feat) F0.5 | Model A (`XGBoost Binary`) F0.5 | Model B (`LightGBM Binary`) F0.5 | Model C (`LightGBM LambdaRank`) F0.5 | Model D (`LightGBM rank_xendcg`) F0.5 |
|---:|---:|---:|---:|---:|---:|
| `0.7000` | `0.4322` | `0.7474` | `0.7506` | `0.7353` | `0.7155` |
| `0.7500` | `0.4679` | `0.7612` | `0.7638` | `0.7492` | `0.7289` |
| `0.8000` | `0.5066` | `0.7793` | `0.7822` | `0.7641` | `0.7436` |
| `0.8250` | `0.5309` | `0.7894` | `0.7924` | `0.7745` | `0.7536` |
| `0.8500` | `0.5460` | `0.8019` | `0.8041` | `0.7867` | `0.7631` |
| `0.8750` | `0.5633` | `0.8144` | `0.8178` | `0.8007` | `0.7755` |
| `0.9000` | `0.5711` | `0.8302` | `0.8300` | `0.8165` | `0.7890` |
| `0.9125` | **`0.5726`** | `0.8395` | `0.8393` | `0.8245` | `0.7966` |
| `0.9250` | `0.5687` | `0.8484` | `0.8482` | `0.8336` | `0.8065` |
| `0.9500` | `0.5477` | `0.8687` | `0.8677` | `0.8562` | `0.8245` |
| `0.9750` | `0.4710` | `0.8902` | `0.8895` | `0.8805` | `0.8434` |
| `0.9900` | `0.3239` | **`0.8970`** | **`0.8991`** | **`0.8939`** | **`0.8452`** |

---

### 3.3 Entity-Level Decision Policy Comparison on `DEV_3000`

| Entity-Level Decision Policy | Model A (`XGBoost Binary`) | Model B (`LightGBM Binary`) | Model C (`LightGBM LambdaRank`) | Model D (`LightGBM rank_xendcg`) |
|---|---:|---:|---:|---:|
| **Policy 1:** Threshold-only (`@ 0.99`) | `0.8970` | `0.8991` | `0.8939` | `0.8452` |
| **Policy 2:** Top-score + Margin rescue | `0.8991` | `0.9008` | `0.8949` | `0.8454` |
| **Policy 3:** Score + Structural Evidence | `0.8970` | `0.8991` | `0.9046` | `0.8456` |
| **Policy 4:** Multi-match `max_k` + Empty-$S_1$ Guard | `0.8974` | `0.9010` | `0.9059` | `0.8469` |
| **Policy 5:** Combined Country + Evidence + Margin + `max_k` | **`0.9025`** | **`0.9023`** | **`0.9070`** | **`0.8469`** |

---

### 3.4 Score Quantiles on `DEV_3000` (`positives_gt1` vs `negatives_gt0`)

| Model | Class | `p10` | `p25` | `p50` (Median) | `p75` | `p90` | `p99` | Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Model A (`XGBoost Binary`)** | `GT = 1` (`10,229` pairs) | `0.9523` | `0.9962` | `0.9997` | `0.99996` | `0.99999` | `0.999997` | `0.9733` |
| **Model A (`XGBoost Binary`)** | `GT = 0` (`2,239,398` pairs) | `1.7e-6` | `5.6e-6` | `2.3e-5` | `1.3e-4` | `0.0011` | `0.1062` | `0.0057` |
| **Model B (`LightGBM Binary`)** | `GT = 1` (`10,229` pairs) | `0.9608` | `0.9972` | `0.9998` | `0.99998` | `0.999995` | `0.999999` | `0.9753` |
| **Model B (`LightGBM Binary`)** | `GT = 0` (`2,239,398` pairs) | `9.1e-7` | `3.0e-6` | `1.2e-5` | `7.4e-5` | `0.0006` | `0.0886` | `0.0053` |
| **Model C (`LightGBM LambdaRank` Calibrated)** | `GT = 1` (`10,229` pairs) | `0.9820` | `0.9997` | `0.99999` | `1.0000` | `1.0000` | `1.0000` | `0.9787` |
| **Model C (`LightGBM LambdaRank` Calibrated)** | `GT = 0` (`2,239,398` pairs) | `1.3e-7` | `4.8e-7` | `2.6e-6` | `2.2e-5` | `0.0003` | `0.0890` | `0.0054` |
| **Model C (`LightGBM LambdaRank` Raw Margin)** | `GT = 1` (`10,229` pairs) | `+3.14` | `+6.71` | `+9.97` | `+12.73` | `+14.38` | `+15.87` | `+9.40` |
| **Model C (`LightGBM LambdaRank` Raw Margin)** | `GT = 0` (`2,239,398` pairs) | `-14.29` | `-13.17` | `-11.68` | `-9.79` | `-7.58` | `-2.42` | `-11.23` |

---

## 4. Single-Shot Firewall Evaluation on `INTERNAL_CHECK_1000` and `FINAL_HOLDOUT_1000`

With `C_LightGBM_LambdaRank` + Policy 5 (`t_us=0.995, t_in=0.995, max_k=6`) frozen on `DEV_3000`, `evaluate_frozen_v3_on_check_and_holdout.py` executed a **single-shot evaluation** on `INTERNAL_CHECK_1000` (`1,000` $S_1$s, `3,436` GT links) and `FINAL_HOLDOUT_1000` (`1,000` $S_1$s, `3,430` GT links) with **zero tuning**.

### 4.1 Cross-Pool Generalization Summary (`DEV_3000` vs `INTERNAL_CHECK_1000` vs `FINAL_HOLDOUT_1000`)

| Metric | `DEV_3000` (`3,000` $S_1$s) | `INTERNAL_CHECK_1000` (`1,000` $S_1$s) | `FINAL_HOLDOUT_1000` (`1,000` $S_1$s) |
|---|---:|---:|---:|
| **V3 Candidate Micro Recall (Top-750)** | `97.16%` (`10,229 / 10,528`) | `97.18%` (`3,339 / 3,436`) | **`97.70%` (`3,351 / 3,430`)** |
| **V3 Candidate Macro Recall (Non-Empty)** | `97.09%` | `97.16%` | **`97.74%`** |
| **V3 Candidate $S_2$ Link Recall** | `97.00%` | `97.88%` | **`97.50%`** |
| **V3 Candidate $S_3$ Link Recall** | `97.31%` | `96.52%` | **`97.89%`** |
| **Frozen V2-C Baseline Macro F0.5 (`@0.900`)** | `0.5711` (`0.5726` peak) | `0.5660` | `0.5481` |
| **Frozen V3 Primary (`C_LightGBM_LambdaRank` + Policy 5) Macro F0.5** | **`0.9070`** | **`0.9215`** | **`0.9140`** |
| **Frozen V3 Primary Macro Precision** | `94.33%` | `95.95%` | **`94.96%`** |
| **Frozen V3 Primary Macro Recall** | `83.63%` | `84.23%` | **`84.29%`** |
| **Frozen V3 Primary $S_2$ Micro Recall** | `82.70%` (Prec `97.03%`) | `84.21%` (Prec `98.65%`) | **`83.63%` (Prec `98.18%`)** |
| **Frozen V3 Primary $S_3$ Micro Recall** | `83.30%` (Prec `94.58%`) | `82.78%` (Prec `98.14%`) | **`83.94%` (Prec `96.08%`)** |
| **Frozen V3 Primary US Macro F0.5** | `0.9247` | `0.9409` | **`0.9223`** |
| **Frozen V3 Primary India Macro F0.5** | `0.8813` | `0.8933` | **`0.9025`** |
| **Frozen V3 Primary Empty-$S_1$ Accuracy** | `93.21%` | `97.92%` | **`95.00%`** |
| **Frozen V3 Primary Single-Match $S_1$ F0.5** | `0.7741` | `0.7529` | **`0.8138`** |
| **Frozen V3 Primary Multi-Match $S_1$ F0.5** | `0.9136` | `0.9330` | **`0.9201`** |

---

### 4.2 All 4 Frozen Bakeoff Models on `INTERNAL_CHECK_1000` and `FINAL_HOLDOUT_1000`

| Frozen Model | Decision Mode | `DEV_3000` F0.5 | `INTERNAL_CHECK_1000` F0.5 | `FINAL_HOLDOUT_1000` F0.5 |
|---|---|---:|---:|---:|
| **Model C: `LightGBM LambdaRank` (Primary)** | **Frozen Entity Policy 5** | **`0.9070`** | **`0.9215`** | **`0.9140`** |
| **Model C: `LightGBM LambdaRank`** | Frozen Threshold (`0.99`) | `0.8939` | `0.9092` | `0.9035` |
| **Model B: `LightGBM Binary`** | Frozen Threshold (`0.99`) | `0.8991` | `0.9122` | `0.9120` |
| **Model B: `LightGBM Binary`** | Frozen Entity Policy 5 | `0.9023` | `0.9138` | `0.9087` |
| **Model A: `XGBoost Binary`** | Frozen Threshold (`0.99`) | `0.8970` | `0.9065` | `0.9105` |
| **Model A: `XGBoost Binary`** | Frozen Entity Policy 5 | `0.9025` | `0.9153` | `0.9106` |
| **Model D: `LightGBM rank_xendcg`** | Frozen Entity Policy 5 | `0.8469` | `0.8645` | `0.8619` |
| **Frozen V2-C Reference (11 features)** | Locked Threshold (`0.900`) | `0.5711` | `0.5660` | `0.5481` |

---

## 5. Top Feature Importances

| Rank | `A_XGBoost_Binary` (Gain Share) | Gain | `B_LightGBM_Binary` (Split Count) | Splits |
|---:|---|---:|---|---:|
| 1 | `name_x_addr_sim` | `0.4140` | `name_rare_token_idf_overlap` | `2,844` |
| 2 | `core_name_x_norm_addr_set` | `0.1766` | `name_len_2` | `2,407` |
| 3 | `building_number_conflict` | `0.0658` | `addr_rare_token_idf_overlap` | `2,364` |
| 4 | `state_region_conflict` | `0.0632` | `name_len_1` | `2,134` |
| 5 | `name_addr_conflict` | `0.0290` | `name_ratio` | `2,054` |
| 6 | `addr_min_containment` | `0.0244` | `address_len_1` | `1,952` |
| 7 | `name_addr_agreement` | `0.0227` | `name_len_ratio` | `1,819` |
| 8 | `numeric_token_jaccard` | `0.0180` | `norm_name_partial_ratio` | `1,741` |
| 9 | `num_conflict_x_name_sim` | `0.0149` | `phonetic_name_ratio` | `1,658` |
| 10 | `name_token_count_min` | `0.0110` | `norm_name_ratio` | `1,599` |
| 11 | `exact_orig_name` | `0.0104` | `address_len_2` | `1,539` |
| 12 | `core_name_partial_ratio` | `0.0092` | `name_token_sort` | `1,487` |
| 13 | `num_agree_x_name_sim` | `0.0091` | `name_rare_token_idf_ratio` | `1,399` |
| 14 | `exact_name_plus_addr_ev` | `0.0069` | `core_name_len_ratio` | `1,385` |
| 15 | `numeric_token_overlap` | `0.0061` | `name_token_jaccard` | `1,344` |
