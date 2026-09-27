# V3 Diagnostic Report — Amazon ML Challenge 2026 Business Entity Resolution

**Generated:** 2026-09-25  
**Branch:** `experiments/entity-resolution-v2` (HEAD `f36569b`, tag `baseline-v1-0.517`)  
**Mode:** Diagnostics Only (Zero modifications to V1 or V2-C artifacts; no test inference; no submission)

---

## 1. Current V2-C Baseline & Artifact Audit

### Frozen Baseline Status
All V1 and V2-C production artifacts were audited and verified completely untouched:

| Generation | Artifact | Path | Size / Status | Key Metric / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **V1 (Frozen)** | Test Submission | `output/matching_results.tsv` | `37,001,203 B` | Public LB = **`0.517`** (`1,732,544` rows, threshold `0.985`) |
| **V1 (Frozen)** | Candidate Pairs | `output/candidate_pairs.tsv` | `833,281,646 B` | Synchronized with V1 test run |
| **V1 (Frozen)** | Checkpoint | `output/checkpoint_full_inference.json` | `204 B` | `347/347` chunks completed |
| **V1 (Frozen)** | Matcher Model | `models/xgboost_matcher.joblib` | `1,139,309 B` | 11 locked features (`300` trees, depth `6`, lr `0.08`) |
| **V2-C (Frozen)** | Training Parquet | `experiments/v2/data/v2_c_train_features.parquet` | `40,860,693 B` | `2,107,713` rows across `166,865` clean train $S_1$ entities |
| **V2-C (Frozen)** | Matcher Model | `experiments/v2/models/xgboost_matcher_v2_c.joblib` | `1,105,669 B` | Trained with V1-mined hard + semi-hard negatives |
| **V2-C (Frozen)** | Local Validation | `experiments/v2/results/v2_c_threshold_sweep.json` | `3,716 B` | `val_1000` Macro $F_{0.5} = \mathbf{0.5900}$ at threshold `0.900` |
| **V2-C (Frozen)** | Test Submission | `experiments/v2/output/matching_results.tsv` | `57,790,586 B` | Public LB = **`0.562685`** (Rank `1272`, `2,797,545` predicted matches) |
| **V2-C (Frozen)** | Test Candidates | `experiments/v2/output/candidate_pairs.tsv` | `99,000,159 B` | `3,962,740` pairs (`p >= 0.50` logged) |
| **V2-C (Frozen)** | Test Checkpoint | `experiments/v2/output/checkpoint_full_inference.json` | `460 B` | `1,732,544` $S_1$ entities, `548,269,487` scored pairs (`1025.62s`) |

### Executive Summary of the `0.5900` (Local) $\rightarrow$ `0.562685` (Leaderboard) Gap
Our diagnostics across Phases 2–6 isolated **two primary root causes** for the `273 bps` validation-to-leaderboard gap, plus **two massive model/feature bottlenecks** that explain why absolute score is capped below `0.65+`:
1. **Test-Inference `MAX_CANDS_PER_ENTITY = 350` S2-First Truncation (`-420 bps` Macro $F_{0.5}$ loss on held-out validation):**
   - `val_1000` evaluated V2-C on **uncapped** blocker candidates (`96.27%` candidate link recall).
   - `run_v2_c_full_inference.py` capped candidates at `350` per $S_1$ while iterating through `cand_token_ids[t]` where `S2` was appended before `S3`. Whenever the 1st rare token had $\ge 350$ matches in `S2`, the loop terminated early—discarding `S3` candidates and 2nd/3rd rare-token candidates.
   - On our 5,000 held-out $S_1$ validation set, this exact `350`-cap drops candidate link recall from **`96.27%` to `79.11%`** (`S3` candidate recall collapses from `95.72%` to `73.12%`, losing `2,927` true GT links) and drops V2-C Macro $F_{0.5}$ at threshold `0.900` from **`0.6189` (uncapped) to `0.5769` (capped-350)**.
2. **Country Composition Shift & Unseen France Domain (`0.5769` $\rightarrow$ `0.562685`):**
   - `train_source1.tsv` contains **only US (`60.0%`) and India (`40.0%`)**—zero France records.
   - India achieves ~`7.2` to `9.0` points lower Macro $F_{0.5}$ than US (`0.5230` vs `0.6132` capped at `0.900`).
   - `test_source1.tsv` has a higher proportion of India (`46.8%` vs `40.0%`) plus **`259,452` unseen France entities (`15.0%`)**, where V2-C over-predicted **`3.026` matches/$S_1$** (vs `1.609` in US and `1.168` in India).
3. **Critical V3 Bottleneck #1 — Case-Sensitive RapidFuzz Features + 3.2:1 S2 Negative Bias Crushing $S_2$ Recall to `22.10%`:**
   - In `feature_engineering.py`, `process.cpdist` is called on raw un-lowercased strings. In RapidFuzz 3.x, `fuzz.ratio`, `fuzz.token_sort_ratio`, and `fuzz.token_set_ratio` are **case-sensitive** by default (`'A' != 'a'`).
   - Because `S2` frequently stores addresses in `ALL CAPS` while `S1` uses `Title Case`, identical addresses (e.g., `"15 Dholakia Mansion... Mumbai"` vs `"15 DHOLAKIA MANSION... MUMBAI"`) receive `address_ratio = 0.407` and `address_token_set = 0.407`!
   - Combined with `v2_c_train_features.parquet` having a **3.2:1 $S_2$ negative bias** (`1,248,866` $S_2$ negatives vs `387,276` $S_3$ negatives due to `S2`-first negative mining), V2-C's recall on $S_2$ ground-truth links at threshold `0.900` (even uncapped) is only **`22.10%` (`1,841 / 8,330`)**, compared to **`67.73%` (`5,912 / 8,729`)** on $S_3$ links!
4. **Critical V3 Bottleneck #2 — Zero Digit / Street-Number Features Causing `63.8%` of False Positives:**
   - Of the `2,172` false positives at threshold `0.900`, **`1,386` (`63.8%`)** have completely mismatched street/building numbers (e.g., `"4802 126th Street, Omaha, NE"` vs `"5655 F Street, Omaha, NE"`), because the 11 locked features contain zero numeric/digit overlap or postal-code agreement signals.

---

## 2. Training Population Verification (Phase 2)

Direct inspection of `experiments/v2/data/v2_c_train_features.parquet` confirmed exact counts and resolved why `166,865` unique $S_1$ entities appear:

| Metric | Exact Value | Notes |
| :--- | :--- | :--- |
| **Total rows** | `2,107,713` | Strictly deduplicated on `(source1_entity_id, candidate_entity_id)` |
| **Unique `source1_entity_id`** | `166,865` | `100.0%` (`2,107,713` rows) have prefix `S1-` |
| **Unique `candidate_entity_id`** | `1,523,285` | `1,477,010` rows (`70.1%`) `S2-` \| `630,703` rows (`29.9%`) `S3-` |
| **Positive rows (`label = 1`)** | `471,571` (`22.37%`) | `S2-`: `228,144` (`48.4%`) \| `S3-`: `243,427` (`51.6%`) |
| **Negative rows (`label = 0`)** | `1,636,142` (`77.63%`) | **`S2-`: `1,248,866` (`76.3%`)** \| **`S3-`: `387,276` (`23.7%`)** |
| **Unique $S_1$ among positives** | `157,639` | `24` $S_1$ entities have only positive rows (`0` mined negatives) |
| **Unique $S_1$ among negatives** | `166,841` | `9,226` $S_1$ entities are true singletons (only negative rows, `0` GT positives) |
| **Rows per $S_1$ distribution** | `mean = 12.63` | `min=1`, `p10=5`, `p25=6`, `median=6.0`, `p75=20`, `p90=28`, `p95=32`, `p99=69`, `max=395` |

### Why `166,865` Unique $S_1$ Entities When `50,000` Were Sampled in V2-C:
- Not an ID-prefix corruption (`100%` of `source1_entity_id` are valid `S1-` entities with `split == 'train'`, and `0` overlap with validation).
- In `experiments/v2/src/mine_and_train_v2_c.py` (Step 3, lines 441–454), the newly mined pairs from the **`50,000` sampled clean train $S_1$ entities** (`random_state=42`) were **concatenated with `v2_a_train_features.parquet`** (`120,285` clean train $S_1$ entities derived from V1 training chunks after removing the `29,714` leaked validation $S_1$s).
- **Exact Set Union Verification:**
  $$120{,}285 \text{ (V2-A clean train } S_1\text{)} + 50{,}000 \text{ (V2-C sampled train } S_1\text{)} - 3{,}420 \text{ (overlap)} = \mathbf{166{,}865 \text{ unique } S_1\text{ entities}}$$

---

## 3. Genuinely Held-Out 5,000-$S_1$ Validation Definition (Phase 3)

We constructed a strictly disjoint **5,000-$S_1$ held-out validation set** (`experiments/v3/diagnostics/heldout_5000_s1.tsv`, `random_state=2026`) from `dataset/train/train_ground_truth.tsv`:
- **Exclusion Set (`197,504` $S_1$ IDs):** Union of all $S_1$ IDs in `training_features_chunks/` (`149,999`), `v2_a_train_features.parquet` (`120,285`), `v2_b_train_features.parquet` (`120,285`), `v2_c_train_features.parquet` (`166,865`), and `val_1000` (`1,000`).
- **Eligible Pool:** `1,600,309` completely unseen `split == 'train'` $S_1$ entities (**zero overlap** verified by assertion).
- **Held-Out 5,000-$S_1$ Ground-Truth Profile:**
  - **Total $S_1$ entities:** `5,000` (**US:** `2,985` [`59.7%`], **India:** `2,015` [`40.3%`]).
  - **Total Ground-Truth Links:** `17,059` (`mean = 3.4118` links/$S_1$; **$S_2$ links:** `8,330` [`48.8%`], **$S_3$ links:** `8,729` [`51.2%`]).
  - **Cardinality Breakdown:**
    - `empty_s1` (`0` GT matches / true singletons): **`311`** (`6.22%`)
    - `single_match_s1` (`1` GT match): **`294`** (`5.88%`, `294` links)
    - `multi_match_s1` ($\ge 2$ GT matches): **`4,395`** (`87.90%`, `16,765` links)

---

## 4. Candidate Recall Diagnostic Results (Phase 4)

We evaluated **four blocking regimes** on the 5,000 held-out $S_1$ entities against the full `train_source2.tsv` (`2.66M` rows) and `train_source3.tsv` (`2.66M` rows) corpora:

| Blocking Regime | Micro Link Recall | Macro Entity Recall (Non-Empty) | Found / Total GT Links | Missed GT Links | $S_2$ Link Recall (`8,330`) | $S_3$ Link Recall (`8,729`) | Zero-Cand $S_1$s | Candidate Count per $S_1$ (`mean` / `median` / `p90` / `p95` / `p99`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Uncapped Top-3 Rare Tokens** *(V2-C `val_1000` style)* | **`96.27%`** | **`96.35%`** | `16,422 / 17,059` | `637` | `96.84%` (`263` missed) | `95.72%` (`374` missed) | `0` (`0.0%`) | `7,996.8` / `3,662.5` / `20,853` / `29,477` / `54,904` |
| **2. Capped-350 S2-First** *(Actual V2-C Test Inference)* | **`79.11%`** | **`79.09%`** | `13,495 / 17,059` | **`3,564`** | `85.38%` (`1,218` missed) | **`73.12%`** (`2,346` missed) | `0` (`0.0%`) | `334.9` / `350.0` / `350` / `350` / `350` |
| **3. Multi-Token + Rarity Top-500** *(Proposed V3 Fast Blocker)* | **`95.52%`** | **`95.59%`** | `16,295 / 17,059` | `764` | `96.21%` (`316` missed) | **`94.87%`** (`448` missed) | `0` (`0.0%`) | `499.7` / `500.0` / `500` / `500` / `500` |
| **4. Uncapped Top-5 Rare Tokens** *(Upper Bound Reference)* | **`98.62%`** | **`98.70%`** | `16,823 / 17,059` | `236` | `98.82%` (`98` missed) | `98.42%` (`138` missed) | `0` (`0.0%`) | `41,829.9` / `30,112.5` / `94,121` / `128,744` / `205,947` |

### Key Takeaways from Candidate Recall:
1. **The 350-Cap S2-First Bug Discarded `2,927` True Links (`17.16%` of all GT):** Because `run_v2_c_full_inference.py` appended `S2` before `S3` in each token's posting list and stopped as soon as `len(cands) >= 350`, `1,972` true $S_3$ links and `955` true $S_2$ links that were present in the top-3 rare tokens were truncated before scoring.
2. **Why Uncapped Top-3 Missed `637` Links:**
   - `571 / 637` (`89.6%`) shared valid rare tokens ($\le 100\text{k}$ frequency), but those tokens ranked **4th or 5th rarest** in the $S_1$ query (e.g., when an $S_1$ address had 3 rare sub-locality/typo tokens that didn't appear in an abbreviated $S_3$ target like `"203, Hyderabad, TG"`). Expanding from Top-3 to Top-5 rare tokens recovers `401` of those `571` links (`98.62%` micro recall).
   - `61 / 637` (`9.6%`) only shared common tokens with corpus frequency $> 100{,}000$.
   - `5 / 637` (`0.8%`) had zero shared tokens of length $\ge 3$ (cross-script transliteration or severe abbreviation).
3. **Multi-Token + Rarity Prioritized Top-500 Restores `95.52%` Recall at $16\times$ Lower Volume than Uncapped:** By accumulating candidate hits across the query's rare tokens and ranking candidates by `(num_matched_query_tokens, sum_idf_weight)` with interleaved $S_2/S_3$ IDs before capping at `500`, we recover **`+16.41%` micro link recall** over V2-C's test blocker (`95.52%` vs `79.11%`, and `+21.75%` on $S_3$ links) while keeping candidate volume strictly bounded (`500` per $S_1$).

---

## 5. Threshold Robustness Table on Held-Out 5,000-$S_1$ Set (Phase 5)

Using the frozen V2-C model (`experiments/v2/models/xgboost_matcher_v2_c.joblib`), we swept thresholds `[0.850, 0.875, 0.900, 0.9125, 0.925, 0.9375, 0.950, 0.9625, 0.975, 0.985, 0.990]` across all three candidate regimes:

### A. Actual V2-C Test-Inference Blocker Regime (`Capped-350 S2-First`)
*(Explains why `0.900` translated to `0.562685` on the public leaderboard)*

| Threshold | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton Acc (`311`) | TP | FP | FN | Pred Pairs | Empty $S_1$ Preds | US $F_{0.5}$ | India $F_{0.5}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.8500`** | **`0.6101`** | `72.81%` | `43.80%` | `78.14%` | `7,123` | `786` | `9,936` | `7,909` | `1,248` | `0.6364` | `0.5712` |
| **`0.8750`** | `0.5979` | `72.38%` | `41.74%` | `81.99%` | `6,675` | `588` | `10,384` | `7,263` | `1,355` | `0.6300` | `0.5503` |
| **`0.9000`** *(V2-C Sub)* | **`0.5769`** | `70.80%` | `39.14%` | `85.21%` | `6,178` | `406` | `10,881` | `6,584` | `1,518` | `0.6132` | `0.5230` |
| **`0.9125`** | `0.5652` | `69.90%` | `37.84%` | `88.10%` | `5,908` | `300` | `11,151` | `6,208` | `1,621` | `0.6026` | `0.5096` |
| **`0.9250`** | `0.5490` | `68.61%` | `36.12%` | `90.03%` | `5,571` | `243` | `11,488` | `5,814` | `1,718` | `0.5870` | `0.4928` |
| **`0.9375`** | `0.5307` | `67.07%` | `34.31%` | `94.53%` | `5,205` | `163` | `11,854` | `5,368` | `1,852` | `0.5692` | `0.4738` |
| **`0.9500`** | `0.5064` | `64.71%` | `32.24%` | `97.43%` | `4,801` | `103` | `12,258` | `4,904` | `2,013` | `0.5446` | `0.4498` |
| **`0.9625`** | `0.4725` | `61.23%` | `29.53%` | `99.04%` | `4,289` | `52` | `12,770` | `4,341` | `2,216` | `0.5092` | `0.4181` |
| **`0.9750`** | `0.4233` | `55.82%` | `26.01%` | `100.00%` | `3,617` | `20` | `13,442` | `3,637` | `2,507` | `0.4580` | `0.3720` |
| **`0.9850`** | `0.3519` | `47.53%` | `21.19%` | `100.00%` | `2,746` | `4` | `14,313` | `2,750` | `2,932` | `0.3814` | `0.3081` |
| **`0.9900`** | `0.2950` | `40.26%` | `17.80%` | `100.00%` | `2,142` | `3` | `14,917` | `2,145` | `3,296` | `0.3175` | `0.2618` |

### B. Multi-Token + Rarity Prioritized Top-500 Blocker Regime (and Uncapped Reference)

| Threshold | Prioritized-500 Macro $F_{0.5}$ | Uncapped Macro $F_{0.5}$ | Prioritized-500 Precision | Prioritized-500 Recall | Singleton Acc (`311`) | TP (`P-500`) | FP (`P-500`) | FN (`P-500`) | Pred Pairs | US $F_{0.5}$ (`P-500`) | India $F_{0.5}$ (`P-500`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.8500`** | `0.6173` | `0.6150` | `69.74%` | `51.37%` | `51.13%` | `8,811` | `4,145` | `8,248` | `12,956` | `0.6293` | **`0.5995`** |
| **`0.8750`** | **`0.6224`** | **`0.6216`** | `71.68%` | `49.03%` | `57.88%` | `8,276` | `2,960` | `8,783` | `11,236` | `0.6429` | `0.5921` |
| **`0.9000`** | `0.6185` | `0.6189` | `72.82%` | `45.96%` | `64.95%` | `7,654` | `1,999` | `9,405` | `9,653` | **`0.6494`** | `0.5726` |
| **`0.9125`** | `0.6138` | `0.6152` | `73.17%` | `44.36%` | `70.42%` | `7,305` | `1,590` | `9,754` | `8,895` | `0.6484` | `0.5625` |
| **`0.9250`** | `0.6046` | `0.6074` | `73.10%` | `42.41%` | `74.92%` | `6,896` | `1,250` | `10,163` | `8,146` | `0.6420` | `0.5492` |
| **`0.9375`** | `0.5917` | `0.5947` | `72.61%` | `40.19%` | `82.32%` | `6,422` | `873` | `10,637` | `7,295` | `0.6328` | `0.5308` |
| **`0.9500`** | `0.5709` | `0.5740` | `71.23%` | `37.65%` | `89.39%` | `5,885` | `583` | `11,174` | `6,468` | `0.6127` | `0.5089` |
| **`0.9625`** | `0.5392` | `0.5427` | `68.57%` | `34.48%` | `94.86%` | `5,245` | `295` | `11,814` | `5,540` | `0.5814` | `0.4766` |
| **`0.9750`** | `0.4870` | `0.4897` | `63.44%` | `30.26%` | `98.71%` | `4,411` | `142` | `12,648` | `4,553` | `0.5275` | `0.4271` |
| **`0.9850`** | `0.4011` | `0.4030` | `53.89%` | `24.11%` | `99.36%` | `3,291` | `54` | `13,768` | `3,345` | `0.4323` | `0.3547` |
| **`0.9900`** | `0.3325` | `0.3340` | `45.38%` | `19.86%` | `99.68%` | `2,529` | `18` | `14,530` | `2,547` | `0.3569` | `0.2964` |

---

## 6. Country, Cardinality & Source ($S_2$ vs $S_3$) Breakdown (Phase 5)

At threshold `0.900` on the 5,000 held-out $S_1$ entities (Prioritized Top-500 / Uncapped):

1. **Huge Asymmetry Between $S_2$ and $S_3$ Recall:**
   - **$S_3$ Link Performance (`8,729` GT links):** `TP = 5,912`, `FP = 1,750`, `FN = 2,817` $\rightarrow$ **Micro Precision = `77.16%`, Micro Recall = `67.73%`**.
   - **$S_2$ Link Performance (`8,330` GT links):** `TP = 1,841`, `FP = 422`, `FN = 6,489` $\rightarrow$ **Micro Precision = `81.35%`, Micro Recall = `22.10%`**.
   - **Root Cause:** As uncovered in Phase 6, `S2` frequently stores addresses in `UPPERCASE` while `S1` uses `Title Case`. Because `feature_engineering.py` called `process.cpdist` on raw strings without `.lower()`, RapidFuzz penalized `S2` address similarity by `-0.40` to `-0.60`. Worse still, `v2_c_train_features.parquet` contained `1,248,866` `S2` negatives vs `387,276` `S3` negatives (`3.2:1` negative bias), teaching the model to heavily suppress `S2` predictions!
2. **Cardinality Breakdown (at `0.900` Prioritized-500):**
   - `multi_match_s1` (`4,395` $S_1$s): Macro $F_{0.5} = \mathbf{0.6339}$ (Precision `75.91%`, Recall `44.96%`).
   - `single_match_s1` (`294` $S_1$s): Macro $F_{0.5} = \mathbf{0.3551}$ (Precision `34.92%`, Recall `40.82%`) — dragged down when extra false-positive links are predicted alongside or instead of the single true link.
   - `empty_s1` (`311` true singletons): Macro $F_{0.5} = \mathbf{0.6495}$ (`202 / 311` correctly left empty; `109` had $\ge 1$ false positive above `0.900`).
3. **Country Breakdown (at `0.900` Prioritized-500):**
   - **US (`2,985` $S_1$s):** Macro $F_{0.5} = \mathbf{0.6494}$ (Precision `75.03%`, Recall `50.10%`).
   - **India (`2,015` $S_1$s):** Macro $F_{0.5} = \mathbf{0.5726}$ (Precision `69.55%`, Recall `39.84%`) — lower recall due to verbose Indian multi-locality addresses and missing pin-code/house-number features.

---

## 7. Dominant Error Categories (Phase 6)

### A. False Positive Analysis (`2,172` FPs at `p >= 0.900` on Held-Out 5,000 $S_1$s)
| Failure Mode Category | Count (of `2,172` FPs) | Share of FPs | Mechanism & Concrete Example |
| :--- | :---: | :---: | :--- |
| **1. Address Digit / House / Street Number Mismatch** | **`1,386`** | **`63.8%`** | Same city/street or business name, but completely different building/street number: e.g., `S1-975065352` (`"31 Eagle Park Circle, Banner Elk, NC"`) vs `S3-888442365` (`"46 Eagle Park Circle, Banner Elk, NC"`, `p = 0.9924`); `S1-240443369` (`"4802 126th Street, Omaha, NE"`) vs `S3-68783673` (`"5655 F Street, Omaha, NE"`, `p = 0.9943`). |
| **2. High Name Sim (`>= 0.90`) + Low Address Sim (`< 0.65`) — Different Branch** | **`1,222`** | **`56.3%`** | Multi-branch chains or common trade names in the same country/city: e.g., `S1-239591027` (`"Pediatric Dental Group"`, `"4166 Webster Avenue, Cincinnati, OH"`) vs `S3-167086755` (`"Pediatric Group Dental"`, `"945 Locust Lane, Cincinnati, Ohio"`, `p = 0.9948`). |
| **3. Short / Truncated Business Name ($\le 10$ chars)** | **`214`** | **`9.9%`** | Single-word candidate name gets `name_token_set = 1.00` against a multi-word query: e.g., `"Environmental Committee Inc"` vs `"Environmental"` (`p = 0.9935`), `"Young Advanced Plains"` vs `"Advanced"` (`p = 0.9929`). |
| **4. Ambiguous Near-Duplicate / Transliteration Noise** | **`176`** | **`8.1%`** | High name + address similarity where one character in the name differs (`"Gangesh International Pvt Ltd"` vs `"Gangosh International Pvt Ltd"` at `"48.7.66 Rama Talkies Road"`, `p = 0.9989`). |
| **5. True Singleton $S_1$ False Alarms** | **`170`** | **`7.8%`** | True singletons (`311` $S_1$s) where a rearrangeable name (`"Rev It Up Hair Studio"` vs `"Rev Hair Up It Studio"`, `addr_token_set = 0.40`, `p = 0.9925`) triggers a match despite zero address overlap. |
| **6. Explicit 5/6-Digit Postal / PIN Code Mismatch** | **`68`** | **`3.1%`** | Both addresses contain explicit postal codes that disagree. |
| **7. Exact Normalized Name / Address Precision Check** | `18` / `1` | `0.8%` / `0.05%` | At `p >= 0.900`, exact normalized name has **`98.94%` precision** (`1,679 TP / 18 FP`), and exact normalized address has **`99.85%` precision** (`684 TP / 1 FP`). |

### B. False Negative Analysis (`9,306` Total FNs = `637` Blocked + `8,669` Scored $< 0.900$)
| Failure Mode Category | Count | Mechanism & Concrete Example |
| :--- | :---: | :--- |
| **1. Case-Sensitive RapidFuzz Bug & Address Formatting Variation (`name_token_set >= 0.85`)** | **`3,004`** | **Smoking Gun:** `S2` stores uppercase addresses (`"15 DHOLAKIA MANSION... MUMBAI, Maharashtra"` vs `"15 Dholakia Mansion... Mumbai, Maharashtra"`, `S1-883582340` $\leftrightarrow$ `S2-11552053`), yielding raw case-sensitive `addr_ratio = 0.407`, `addr_token_set = 0.407` and `p = 0.8999` (missed at `0.900`!). Also affects abbreviated `S3` Indian addresses (`"1839/7, Bhiwandi, Kalyan, MH"` vs full 12-token address, `p = 0.8998`). |
| **2. Near-Miss Threshold Band (`0.80 <= p < 0.90`)** | **`1,977`** | Nearly `2,000` true positives sit just below `0.900` (`0.80–0.90`), mostly `S2` uppercase-address pairs and `S3` abbreviated-address pairs where building/plot numbers (`1839/7`, `2200`, `1910`) match 100%. |
| **3. Moderate Score Band (`0.50 <= p < 0.80`)** | **`3,085`** | Heavy `S2` penalties where both name and address have case differences or legal-suffix variations (`Pvt Ltd` vs `Private Limited`). |
| **4. Low Score Band (`p < 0.50`)** | **`3,795`** | Domain-name handles (`"Venkata Advisors"` vs `"venkataadvisors.com"`), non-Latin regional scripts (`Telugu` / `Hindi` / `Gujarati` names with matching plot numbers `"203, Hyderabad"`), or DBA/trade-name aliases at identical addresses (`addr_ratio = 1.0`). |

---

## 8. Recommended Single Clean V3 Experiment Design (Phase 7)

Based on the evidence above, **both candidate retrieval AND pairwise feature engineering/scoring are high-leverage bottlenecks**, and they interact directly (fixing candidate truncation recovers `+16.4%` link recall, while fixing case-insensitivity + digit features + balanced S2/S3 training fixes the `22.1%` $S_2$ recall and eliminates `63.8%` of FPs).

We recommend **ONE unified, disciplined V3 experiment (`experiments/v3/`)** combining three clean upgrades:

### Upgrade 1 — Multi-Token + Rarity Prioritized Top-500 Blocker (Interleaved $S_2/S_3$)
- **Change:** Expand query rare-token selection from **Top-3 to Top-5 rarest tokens** ($\text{freq} \le 100{,}000$) plus exact normalized name and exact normalized address. Instead of iterating in raw $S_2$-then-$S_3$ file order and truncating at 350, score every candidate hit in the posting lists by:
  $$\text{priority}(c) = 100 \times (\text{exact\_name} + \text{exact\_addr}) + 10 \times \text{matched\_token\_count}(c) + \sum_{t \in \text{matched}} \frac{1}{\log_{10}(10 + \text{freq}(t))}$$
  and retain the **Top-400 to Top-500 candidates per $S_1$** (with $S_2$ and $S_3$ treated symmetrically).
- **Verified Impact:** Raises candidate recall from `79.11%` (`73.12%` on $S_3$) to **`95.52%+` (`98.62%` uncapped pool before top-500 prioritization)** while keeping test inference candidate volume at $\le 500$ pairs/$S_1$.

### Upgrade 2 — Case-Normalized + Structural & Digit-Aware Feature Set (`25` Features)
- **Keep all 11 locked features** (for backward compatibility/monotonic ensemble if needed) **AND add 14 fast vectorized structural features** computed on **lowercased, normalized strings**:
  1. **Lowercased / Normalized RapidFuzz Similarities (`6` features):**
     - `norm_name_ratio`, `norm_name_token_sort`, `norm_name_token_set` (computed on lowercased alnum-normalized name, stripping legal suffixes `pvt`, `ltd`, `private`, `limited`, `inc`, `llc`, `corp`, `llp`, `co`).
     - `norm_addr_ratio`, `norm_addr_token_sort`, `norm_addr_token_set` (computed on lowercased normalized address — **immediately fixes the `S2` uppercase bug!**).
  2. **Address Digit & Building/Street Number Agreement (`4` features):**
     - `addr_digit_jaccard`: Jaccard similarity of numeric tokens (`\d+`) between `addr1` and `addr2` (`1.0` when street/plot numbers match, `0.0` when `"4802"` vs `"5655"`, `-1.0` when neither has digits).
     - `addr_first_num_match`: `1` if first numeric token (house/building number) matches (`"1375"` == `"01375"` after `lstrip('0')`), `-1` if both have numbers and they disagree (`"31"` != `"46"`), `0` if one/both lack numbers.
     - `postal_code_match`: `1` if both have 5/6-digit postal/PIN codes and they match, `-1` if both have postal codes and they disagree, `0` otherwise.
     - `shared_digit_count`: Count of shared numeric tokens across name + address.
  3. **Token Specificity, Length & Interaction Features (`4` features):**
     - `core_name_token_jaccard` & `min_name_token_containment`: Token Jaccard and containment ratio $\frac{|T_1 \cap T_2|}{\min(|T_1|, |T_2|)}$ on core name tokens, paired with `name_token_count_min` and `name_len_ratio = \min(L_1, L_2)/\max(L_1, L_2)$ (prevents 1-word `"Advanced"` from scoring `1.0` against `"Young Advanced Plains"`).
     - `name_x_addr_norm_sim`: `norm_name_token_set * norm_addr_token_set` + `exact_norm_name` + `exact_norm_addr` binary flags.
     - `is_s2_candidate`: Binary indicator (`1` for `S2-`, `0` for `S3-`) so trees can calibrate source-specific address verbosity.

### Upgrade 3 — Balanced $S_2/S_3$ Hard-Negative Training & Country/Margin Calibration
- **Training Set (`experiments/v3/data/v3_train_features.parquet`):**
  - Use the `166,865` clean `split == 'train'` $S_1$ entities (zero overlap with `val_1000` and zero overlap with `heldout_5000_s1.tsv`).
  - Mine candidates using the **Prioritized Top-500 Blocker** (so $S_2$ and $S_3$ negatives are balanced ~`50/50` instead of `76/24`).
  - Include all `471,571` GT positives + balanced hard/semi-hard/random negatives (`~1.5M` negatives, `50%` $S_2$ / `50%` $S_3$).
- **Model & Post-Processing:**
  - Train XGBoost (`600` trees, `max_depth=7`, `learning_rate=0.06`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`).
  - Evaluate on **`heldout_5000_s1.tsv`** across thresholds AND calibrate a lightweight **top-k / probability-gap margin rule** (e.g., cap max predictions per $S_1$ or require higher threshold when `addr_first_num_match == -1`) to protect precision on India, singletons, and unseen France entities.

---

## 9. Expected Computational Cost

| V3 Pipeline Stage | Scope | Estimated Wall-Clock Time | Peak RAM |
| :--- | :--- | :---: | :---: |
| **1. Build V3 Training Set & Balanced Negative Mining** | `166,865` clean train $S_1$s (reusing `v2_c` pair IDs + balanced $S_3$ prioritized negatives, computing 25 features in vectorized batches) | `~12 – 15 min` | `< 6.5 GB` |
| **2. Train V3 XGBoost Matcher** | `~2.1M` rows $\times$ `25` features (`tree_method="hist"`, `n_jobs=-1`) | `~1.5 – 2.5 min` | `< 4.0 GB` |
| **3. Held-Out 5,000-$S_1$ Validation & Ablation** | Score `2.50M` prioritized-500 candidate pairs + threshold/post-processing sweep | `~3.0 – 4.0 min` | `< 4.5 GB` |
| **Total V3 Experiment + Validation Gate** | End-to-end before any test inference decision | **`~17 – 22 min`** | **`< 6.5 GB`** |
| *(If Gate Passed)* **Full Test Inference (`1.73M` $S_1$s)** | Prioritized Top-400/500 blocker + 25 features in 25k-entity chunks | `~22 – 28 min` | `< 8.0 GB` |

---

## 10. Exact Next Command / Prompt to Execute

To proceed to V3 training and validation on the 5,000 held-out $S_1$ benchmark (while keeping V1 and V2-C completely frozen and without running test inference yet), issue the following prompt or run the V3 training script:

```powershell
.venv\Scripts\python.exe experiments/v3/src/train_and_evaluate_v3.py
```

**Recommended User Prompt to Proceed:**
> *"PROCEED WITH V3 TRAINING AND HELD-OUT 5,000-S1 VALIDATION ONLY. Keep V1 and V2-C frozen. Implement the Prioritized Top-500 blocker (interleaved S2/S3, top-5 rare tokens), the 25 case-normalized + digit/structural features, and balanced S2/S3 hard-negative training under `experiments/v3/`. Evaluate on `experiments/v3/diagnostics/heldout_5000_s1.tsv` and report the comparison against V2-C (`0.5769` capped / `0.6185` prioritized) before any test inference."*
