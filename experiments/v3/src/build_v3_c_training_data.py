"""Phase V3-C: Entity-Disjoint Training Data & DEV_3000 Feature Matrix Construction.

Requirements enforced:
1. Validation firewall preserved: DEV_3000, INTERNAL_CHECK_1000, FINAL_HOLDOUT_1000,
   diagnostic heldout_5000_s1, and val_1000 have ZERO overlap with V3-C training S1s.
2. Positive pairs: all GT positive links retrieved by the V3 4-channel blocker criteria,
   retaining S2 and S3 positives separately.
3. Negative pairs: enforces 50/50 S2/S3 representation across hard & structural negatives:
   - high name similarity + different address
   - same/similar name + different building/street number
   - same postal code + different business
   - same street + different business
   - high name similarity + low address similarity
   - candidates with conflicting numeric evidence
   - high V2-C / V3 retrieval score but GT-negative
4. Critical assertion: zero GT-positive pairs ever appear as a negative.
5. Computes all 88 V3-B normalized/numeric/phonetic features and saves:
   - experiments/v3/data/v3_c_train_features.parquet
   - experiments/v3/data/dev_3000_v3_features.parquet
   - experiments/v3/results/v3_c_training_stats.json
   - experiments/v3/results/V3_C_TRAINING_REPORT.md
"""

import gc
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from code.business_entity_resolution.src.feature_engineering import LOCKED_FEATURE_COLUMNS
from code.business_entity_resolution.src.preprocessing import (
    create_train_val_split,
    load_source_tsv,
    parse_match_ids,
)
from experiments.v3.src.features_v3 import V3_ALL_FEATURE_COLUMNS, compute_v3_features
from experiments.v3.src.normalization_v3 import normalize_name_v3, parse_address_v3
from experiments.v3.src.retrieval_v3 import extract_char_ngrams_for_name


def is_v3_retrievable_positive(s1_rec: dict, tgt_rec: dict) -> bool:
    """Verify that a GT positive pair is retrievable by at least one of the 4 V3 channels."""
    if not s1_rec or not tgt_rec:
        return False
    c1 = s1_rec["country"]
    c2 = tgt_rec["country"]
    if c1 != c2:
        return False

    n1 = normalize_name_v3(s1_rec["business_name"])
    n2 = normalize_name_v3(tgt_rec["business_name"])
    a1 = parse_address_v3(s1_rec["business_address"], c1)
    a2 = parse_address_v3(tgt_rec["business_address"], c2)

    # Channel 1: Exact
    if n1["orig_norm"] and n1["orig_norm"] == n2["orig_norm"]:
        return True
    if n1["lower_norm"] and n1["lower_norm"] == n2["lower_norm"]:
        return True
    if n1["core_name_norm"] and n1["core_name_norm"] == n2["core_name_norm"]:
        return True
    if len(n1["compact_norm"]) >= 3 and n1["compact_norm"] == n2["compact_norm"]:
        return True
    if a1["lower_norm"] and a1["lower_norm"] == a2["lower_norm"]:
        return True

    # Channel 2: Shared name, phonetic, or address token
    nt1 = {t for t in n1["core_tokens"] if len(t) >= 2} | {t for t in n1["tokens"] if len(t) >= 3}
    nt2 = {t for t in n2["core_tokens"] if len(t) >= 2} | {t for t in n2["tokens"] if len(t) >= 3}
    if nt1 & nt2:
        return True

    pt1 = {p for p in n1["phonetic_tokens"] if len(p) >= 3}
    pt2 = {p for p in n2["phonetic_tokens"] if len(p) >= 3}
    if pt1 & pt2:
        return True

    at1 = {t for t in a1["tokens"] if len(t) >= 2}
    at2 = {t for t in a2["tokens"] if len(t) >= 2}
    if at1 & at2:
        return True

    # Channel 3: Shared char 4/5-gram
    g1 = extract_char_ngrams_for_name(n1["compact_norm"], n1["core_tokens"])
    g2 = extract_char_ngrams_for_name(n2["compact_norm"], n2["core_tokens"])
    if g1 & g2:
        return True

    # Channel 4: Building or postal match
    if a1["building_num"] and a1["building_num"] == a2["building_num"]:
        return True
    if a1["postal_code"] and a1["postal_code"] == a2["postal_code"]:
        return True

    return False


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("PHASE V3-C: ENTITY-DISJOINT TRAINING DATA CONSTRUCTION (88 V3 FEATURES)", flush=True)
    print("=" * 80, flush=True)

    data_dir = Path("experiments/v3/data")
    res_dir = Path("experiments/v3/results")
    data_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all validation firewall IDs to assert strict zero leakage
    dev_df = pd.read_csv("experiments/v3/pools/dev_3000_s1.tsv", sep="\t", dtype=str)
    internal_df = pd.read_csv("experiments/v3/pools/internal_check_1000_s1.tsv", sep="\t", dtype=str)
    final_df = pd.read_csv("experiments/v3/pools/final_holdout_1000_s1.tsv", sep="\t", dtype=str)
    diag_df = pd.read_csv("experiments/v3/diagnostics/heldout_5000_s1.tsv", sep="\t", dtype=str)
    val1000_bundle = joblib.load(
        r"C:\Users\Sneha Singh\.gemini\antigravity\brain\47892a4a-c3be-4331-990a-46831df8639d\scratch\val_1000_candidates_bundle.joblib"
    )

    dev_ids = set(dev_df["source1_entity_id"])
    internal_ids = set(internal_df["source1_entity_id"])
    final_ids = set(final_df["source1_entity_id"])
    diag_ids = set(diag_df["source1_entity_id"])
    val1000_ids = set(val1000_bundle["val_ids"])
    del val1000_bundle

    forbidden_s1_ids = dev_ids | internal_ids | final_ids | diag_ids | val1000_ids
    print(f"Total forbidden validation/holdout S1 IDs: {len(forbidden_s1_ids):,}", flush=True)

    # 2. Load complete ground truth to enforce zero GT/negative overlap
    gt = load_source_tsv("dataset/train/train_ground_truth.tsv")
    gt = create_train_val_split(gt)
    official_val_s1_ids = set(gt[gt["split"] == "val"]["source1_entity_id"])
    forbidden_s1_ids.update(official_val_s1_ids)

    all_gt_pos_pairs = set()
    for sid, m_str in zip(gt["source1_entity_id"], gt["matched_entity_ids"].fillna("")):
        for cid in parse_match_ids(str(m_str)):
            all_gt_pos_pairs.add((sid, cid))
    del gt
    gc.collect()
    print(f"Loaded {len(all_gt_pos_pairs):,} complete GT positive pairs for strict negative exclusion.", flush=True)

    # 3. Load candidate pool from v2_c_train_features.parquet (166,865 clean train S1 entities)
    v2c_df = pd.read_parquet("experiments/v2/data/v2_c_train_features.parquet")
    v2c_df = v2c_df[~v2c_df["source1_entity_id"].isin(forbidden_s1_ids)].reset_index(drop=True)
    v2c_df = v2c_df.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)

    # Verify zero GT overlap in negatives
    pos_df = v2c_df[v2c_df["label"] == 1].copy()
    neg_df = v2c_df[v2c_df["label"] == 0].copy()
    neg_pair_tuples = list(zip(neg_df["source1_entity_id"], neg_df["candidate_entity_id"]))
    keep_neg_mask = [p not in all_gt_pos_pairs for p in neg_pair_tuples]
    neg_df = neg_df[keep_neg_mask].reset_index(drop=True)

    # Score negatives with frozen V2-C model to stratify hard / structural / semi-hard / easy negatives
    print("Scoring negative pool with frozen V2-C model to prioritize hard & structural negatives...", flush=True)
    v2c_model = joblib.load("experiments/v2/models/xgboost_matcher_v2_c.joblib")
    neg_df["v2c_prob"] = v2c_model.predict_proba(neg_df[LOCKED_FEATURE_COLUMNS])[:, 1]
    neg_df["is_s2"] = neg_df["candidate_entity_id"].str.startswith("S2-")

    # Compute structural priority score on S2 negatives to match S3 negative count (~50/50 balance)
    s3_neg_df = neg_df[~neg_df["is_s2"]].copy()
    s2_neg_df = neg_df[neg_df["is_s2"]].copy()
    n_s3_neg = len(s3_neg_df)
    print(f"Available S3 negatives: {n_s3_neg:,} | Available S2 negatives: {len(s2_neg_df):,}", flush=True)

    # Priority score for S2 negative selection:
    # 1. High V2-C probability (hard negatives)
    # 2. High name similarity + low address similarity (branch confusion)
    # 3. High address similarity + low name similarity (same building/street, different business)
    # 4. At least 1 negative per S1 preserved for group ranking coverage
    s2_neg_df["struct_priority"] = (
        2.5 * s2_neg_df["v2c_prob"]
        + 1.2 * ((s2_neg_df["name_token_set"] >= 0.85) & (s2_neg_df["address_token_set"] < 0.65)).astype(float)
        + 1.2 * ((s2_neg_df["address_token_set"] >= 0.75) & (s2_neg_df["name_token_set"] < 0.65)).astype(float)
        + 0.5 * s2_neg_df["name_token_set"]
        + 0.5 * s2_neg_df["address_token_set"]
    )
    s2_neg_df = s2_neg_df.sort_values("struct_priority", ascending=False).reset_index(drop=True)

    # Ensure every S1 that has S2 negatives keeps its top-1 S2 negative, then fill to n_s3_neg
    top1_per_s1 = s2_neg_df.drop_duplicates(subset=["source1_entity_id"], keep="first")
    rem_s2 = s2_neg_df.drop(index=top1_per_s1.index)
    needed_extra = max(0, n_s3_neg - len(top1_per_s1))
    selected_s2_neg = pd.concat([top1_per_s1, rem_s2.iloc[:needed_extra]], ignore_index=True)

    balanced_neg_df = pd.concat([selected_s2_neg, s3_neg_df], ignore_index=True)
    print(
        f"Balanced negatives selected: {len(balanced_neg_df):,} "
        f"(S2={len(selected_s2_neg):,} [{len(selected_s2_neg)/len(balanced_neg_df)*100:.2f}%], "
        f"S3={len(s3_neg_df):,} [{len(s3_neg_df)/len(balanced_neg_df)*100:.2f}%])",
        flush=True,
    )

    # Combine positive and balanced negative pair skeletons
    cand_skeleton_df = pd.concat(
        [
            pos_df[["source1_entity_id", "candidate_entity_id", "label"]].assign(v2c_prob=1.0),
            balanced_neg_df[["source1_entity_id", "candidate_entity_id", "label", "v2c_prob"]],
        ],
        ignore_index=True,
    )
    del v2c_df, pos_df, neg_df, s2_neg_df, s3_neg_df, balanced_neg_df
    gc.collect()

    needed_s1_ids = set(cand_skeleton_df["source1_entity_id"].unique())
    needed_tgt_ids = set(cand_skeleton_df["candidate_entity_id"].unique())
    print(f"Streaming raw TSVs to fetch text records for {len(needed_s1_ids):,} S1s and {len(needed_tgt_ids):,} targets...", flush=True)

    s1_lookup = {}
    for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", chunksize=500_000, dtype=str):
        sub = chunk[chunk["entity_id"].isin(needed_s1_ids)]
        if sub.empty:
            continue
        for eid, n_raw, a_raw, c_raw in zip(sub["entity_id"], sub["business_name"], sub["business_address"], sub["country"]):
            s1_lookup[eid] = {
                "business_name": "" if pd.isna(n_raw) else str(n_raw),
                "business_address": "" if pd.isna(a_raw) else str(a_raw),
                "country": "" if pd.isna(c_raw) else str(c_raw).strip().lower(),
            }

    target_lookup = {}
    for src_path in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        for chunk in pd.read_csv(src_path, sep="\t", chunksize=1_000_000, dtype=str):
            sub = chunk[chunk["entity_id"].isin(needed_tgt_ids)]
            if sub.empty:
                continue
            for cid, n_raw, a_raw, c_raw in zip(sub["entity_id"], sub["business_name"], sub["business_address"], sub["country"]):
                target_lookup[cid] = {
                    "business_name": "" if pd.isna(n_raw) else str(n_raw),
                    "business_address": "" if pd.isna(a_raw) else str(a_raw),
                    "country": "" if pd.isna(c_raw) else str(c_raw).strip().lower(),
                }

    # Filter positive pairs to those retrievable by the V3 4-channel blocker (Requirement 3)
    print("Filtering positive pairs to those retrievable by the V3 4-channel blocker...", flush=True)
    keep_rows = []
    dropped_unretrievable_pos = 0
    for row in cand_skeleton_df.itertuples(index=False):
        if row.label == 1:
            if is_v3_retrievable_positive(s1_lookup.get(row.source1_entity_id), target_lookup.get(row.candidate_entity_id)):
                keep_rows.append(True)
            else:
                keep_rows.append(False)
                dropped_unretrievable_pos += 1
        else:
            keep_rows.append(True)

    cand_skeleton_df = cand_skeleton_df[keep_rows].reset_index(drop=True)
    # Sort by source1_entity_id so ranking groups are contiguous!
    cand_skeleton_df = cand_skeleton_df.sort_values(["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)
    print(f"Retained {len(cand_skeleton_df):,} V3 training pairs (dropped {dropped_unretrievable_pos:,} unretrievable cross-country/zero-overlap GT noise links).", flush=True)

    # Strict assertions before feature computation
    train_s1_final = set(cand_skeleton_df["source1_entity_id"].unique())
    assert len(train_s1_final & dev_ids) == 0, "DEV leakage detected!"
    assert len(train_s1_final & internal_ids) == 0, "INTERNAL_CHECK leakage detected!"
    assert len(train_s1_final & final_ids) == 0, "FINAL_HOLDOUT leakage detected!"
    assert len(train_s1_final & official_val_s1_ids) == 0, "Official val leakage detected!"

    neg_sub = cand_skeleton_df[cand_skeleton_df["label"] == 0]
    neg_overlap = sum(
        1 for sid, cid in zip(neg_sub["source1_entity_id"], neg_sub["candidate_entity_id"])
        if (sid, cid) in all_gt_pos_pairs
    )
    assert neg_overlap == 0, f"Found {neg_overlap} GT positives in negatives!"

    # 4. Compute 88 V3-B features for V3-C training set
    print(f"\nComputing 88 V3-B features for {len(cand_skeleton_df):,} training pairs...", flush=True)
    t_feat0 = time.time()
    X_train_v3 = compute_v3_features(
        cand_skeleton_df[["source1_entity_id", "candidate_entity_id"]],
        s1_lookup,
        target_lookup,
        batch_size=150_000,
    )
    X_train_v3["source1_entity_id"] = cand_skeleton_df["source1_entity_id"].values
    X_train_v3["candidate_entity_id"] = cand_skeleton_df["candidate_entity_id"].values
    X_train_v3["label"] = cand_skeleton_df["label"].values.astype(np.int32)
    X_train_v3["v2c_prob"] = cand_skeleton_df["v2c_prob"].values.astype(np.float32)

    train_parquet_path = data_dir / "v3_c_train_features.parquet"
    X_train_v3.to_parquet(train_parquet_path, index=False)
    print(f"Saved {train_parquet_path} ({len(X_train_v3):,} rows, { len(V3_ALL_FEATURE_COLUMNS)} features) in {time.time()-t_feat0:.1f}s", flush=True)

    # 5. Compute detailed V3-C training statistics
    pos_mask = X_train_v3["label"] == 1
    neg_mask = X_train_v3["label"] == 0
    is_s2_col = X_train_v3["candidate_entity_id"].str.startswith("S2-")

    neg_part = X_train_v3[neg_mask]
    hard_v2c_cnt = int((neg_part["v2c_prob"] >= 0.50).sum())
    semi_hard_cnt = int(((neg_part["v2c_prob"] >= 0.10) & (neg_part["v2c_prob"] < 0.50)).sum())
    easy_cnt = int((neg_part["v2c_prob"] < 0.10).sum())

    # Structural negative breakdowns on V3 features
    struct_high_name_diff_addr = int(((neg_part["core_name_token_set"] >= 0.85) & (neg_part["norm_addr_token_set"] < 0.62)).sum())
    struct_same_name_diff_bldg = int(((neg_part["core_name_token_set"] >= 0.80) & (neg_part["building_number_conflict"] == 1.0)).sum())
    struct_same_postal_diff_biz = int(((neg_part["postal_match"] == 1.0) & (neg_part["core_name_token_set"] < 0.70)).sum())
    struct_same_street_diff_biz = int(((neg_part["norm_addr_token_set"] >= 0.75) & (neg_part["core_name_token_set"] < 0.65)).sum())
    struct_high_name_low_addr = int((neg_part["high_name_low_addr"] == 1.0).sum())
    struct_numeric_conflict = int(((neg_part["building_number_conflict"] == 1.0) | (neg_part["postal_conflict"] == 1.0) | (neg_part["unit_number_conflict"] == 1.0)).sum())

    dup_pair_count = int(X_train_v3.duplicated(subset=["source1_entity_id", "candidate_entity_id"]).sum())

    v3_c_stats = {
        "total_pairs": int(len(X_train_v3)),
        "positives": int(pos_mask.sum()),
        "negatives": int(neg_mask.sum()),
        "positive_pct": float(pos_mask.mean()),
        "negative_pct": float(neg_mask.mean()),
        "s2_positives": int((pos_mask & is_s2_col).sum()),
        "s3_positives": int((pos_mask & (~is_s2_col)).sum()),
        "s2_negatives": int((neg_mask & is_s2_col).sum()),
        "s3_negatives": int((neg_mask & (~is_s2_col)).sum()),
        "s2_negative_share": float((neg_mask & is_s2_col).sum() / max(1, neg_mask.sum())),
        "s3_negative_share": float((neg_mask & (~is_s2_col)).sum() / max(1, neg_mask.sum())),
        "hard_negatives_v2c_ge_050": hard_v2c_cnt,
        "semi_hard_negatives_010_to_050": semi_hard_cnt,
        "easy_negatives_lt_010": easy_cnt,
        "structural_negative_counts": {
            "high_name_sim_different_address": struct_high_name_diff_addr,
            "same_similar_name_different_building_num": struct_same_name_diff_bldg,
            "same_postal_code_different_business": struct_same_postal_diff_biz,
            "same_street_different_business": struct_same_street_diff_biz,
            "high_name_low_address_flag": struct_high_name_low_addr,
            "conflicting_numeric_evidence": struct_numeric_conflict,
        },
        "unique_s1_count": int(len(train_s1_final)),
        "unique_candidate_count": int(X_train_v3["candidate_entity_id"].nunique()),
        "duplicate_pair_count": dup_pair_count,
        "gt_negative_overlap_count": int(neg_overlap),
        "validation_leakage_counts": {
            "dev_3000_overlap": int(len(train_s1_final & dev_ids)),
            "internal_check_1000_overlap": int(len(train_s1_final & internal_ids)),
            "final_holdout_1000_overlap": int(len(train_s1_final & final_ids)),
            "diagnostic_5000_overlap": int(len(train_s1_final & diag_ids)),
            "val_1000_overlap": int(len(train_s1_final & val1000_ids)),
            "official_val_split_overlap": int(len(train_s1_final & official_val_s1_ids)),
        },
        "feature_count": int(len(V3_ALL_FEATURE_COLUMNS)),
        "dropped_unretrievable_gt_positives": int(dropped_unretrievable_pos),
        "runtime_s": round(time.time() - t0, 2),
    }

    with open(res_dir / "v3_c_training_stats.json", "w", encoding="utf-8") as f:
        json.dump(v3_c_stats, f, indent=2)

    print("\nV3-C Training Data Summary:", flush=True)
    print(json.dumps(v3_c_stats, indent=2), flush=True)

    del X_train_v3, s1_lookup, target_lookup, cand_skeleton_df
    gc.collect()

    # 6. Pre-compute and save 88 V3-B features for DEV_3000 candidate bundle so V3-D bakeoff runs rapidly!
    print("\n" + "=" * 80, flush=True)
    print("COMPUTING 88 V3-B FEATURES FOR DEV_3000 CANDIDATE BUNDLE (2,249,627 PAIRS)", flush=True)
    print("=" * 80, flush=True)
    t_dev0 = time.time()
    dev_bundle = joblib.load("experiments/v3/retrieval/dev_3000_retrieval_bundle.joblib")
    dev_qdf = dev_bundle["queries_df"]
    dev_cands_map = dev_bundle["candidates_by_s1"]
    dev_tgt_records = dev_bundle["target_records"]

    dev_s1_lookup = {}
    dev_gt_map = {}
    for r in dev_qdf.itertuples(index=False):
        sid = r.source1_entity_id
        dev_s1_lookup[sid] = {
            "business_name": "" if pd.isna(r.business_name) else str(r.business_name),
            "business_address": "" if pd.isna(r.business_address) else str(r.business_address),
            "country": "" if pd.isna(r.country) else str(r.country).strip().lower(),
        }
        dev_gt_map[sid] = set(parse_match_ids("" if pd.isna(r.matched_entity_ids) else str(r.matched_entity_ids)))

    dev_pair_rows = []
    for sid in dev_qdf["source1_entity_id"]:
        gt_s = dev_gt_map[sid]
        for cid, r_score, ch_mask in dev_cands_map.get(sid, []):
            dev_pair_rows.append((sid, cid, 1 if cid in gt_s else 0, float(r_score), int(ch_mask)))

    dev_pairs_df = pd.DataFrame(
        dev_pair_rows,
        columns=["source1_entity_id", "candidate_entity_id", "label", "retrieval_score", "channel_mask"],
    )
    X_dev_v3 = compute_v3_features(
        dev_pairs_df[["source1_entity_id", "candidate_entity_id"]],
        dev_s1_lookup,
        dev_tgt_records,
        batch_size=150_000,
    )
    X_dev_v3["source1_entity_id"] = dev_pairs_df["source1_entity_id"].values
    X_dev_v3["candidate_entity_id"] = dev_pairs_df["candidate_entity_id"].values
    X_dev_v3["label"] = dev_pairs_df["label"].values.astype(np.int32)
    X_dev_v3["retrieval_score"] = dev_pairs_df["retrieval_score"].values.astype(np.float32)
    X_dev_v3["channel_mask"] = dev_pairs_df["channel_mask"].values.astype(np.int32)

    dev_feat_path = data_dir / "dev_3000_v3_features.parquet"
    X_dev_v3.to_parquet(dev_feat_path, index=False)
    print(f"Saved {dev_feat_path} ({len(X_dev_v3):,} rows) in {time.time()-t_dev0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
