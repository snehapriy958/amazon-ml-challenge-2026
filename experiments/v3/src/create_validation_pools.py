"""Phase V3 Validation Firewall: Create Deterministic Entity-Disjoint Pools.

Creates three strictly entity-disjoint validation pools from official training data:
  A. DEV / TUNING POOL:       3,000 S1 entities (experiments/v3/pools/dev_3000_s1.tsv)
  B. INTERNAL CHECK POOL:     1,000 S1 entities (experiments/v3/pools/internal_check_1000_s1.tsv)
  C. FINAL HOLDOUT POOL:      1,000 S1 entities (experiments/v3/pools/final_holdout_1000_s1.tsv)

Strict Disjointness Guarantees:
- Zero overlap with V1 training_features_chunks/ (149,999 S1s)
- Zero overlap with V2-A, V2-B, V2-C training sets (166,865 S1s)
- Zero overlap with V1/V2 val_1000 bundle (1,000 S1s)
- Zero overlap with diagnostic heldout_5000_s1.tsv (5,000 S1s)
- Zero overlap with each other (DEV, INTERNAL CHECK, FINAL HOLDOUT are mutually disjoint)
- Preserves all GT matches and empty S1 entities
- Records full country and cardinality distributions in experiments/v3/pools/pools_manifest.json
"""

import gc
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from code.business_entity_resolution.src.preprocessing import (
    create_train_val_split,
    load_source_tsv,
    parse_match_ids,
)


def summarize_pool(df: pd.DataFrame, name: str) -> dict:
    empty_s1 = 0
    single_s1 = 0
    multi_s1 = 0
    total_links = 0
    s2_links = 0
    s3_links = 0

    for m_str in df["matched_entity_ids"].fillna(""):
        m_set = parse_match_ids(m_str)
        k = len(m_set)
        total_links += k
        for cid in m_set:
            if cid.startswith("S2-"):
                s2_links += 1
            elif cid.startswith("S3-"):
                s3_links += 1
        if k == 0:
            empty_s1 += 1
        elif k == 1:
            single_s1 += 1
        else:
            multi_s1 += 1

    countries = df["country"].fillna("").str.strip().str.lower().value_counts().to_dict()
    return {
        "pool_name": name,
        "s1_count": int(len(df)),
        "total_gt_links": int(total_links),
        "s2_gt_links": int(s2_links),
        "s3_gt_links": int(s3_links),
        "mean_gt_links_per_s1": float(total_links / max(1, len(df))),
        "empty_s1_count": int(empty_s1),
        "empty_s1_pct": float(empty_s1 / max(1, len(df))),
        "single_match_s1_count": int(single_s1),
        "single_match_s1_pct": float(single_s1 / max(1, len(df))),
        "multi_match_s1_count": int(multi_s1),
        "multi_match_s1_pct": float(multi_s1 / max(1, len(df))),
        "countries": {str(k): int(v) for k, v in countries.items()},
    }


def main():
    print("=" * 80, flush=True)
    print("BUILDING V3 VALIDATION FIREWALL (DEV 3000 / INTERNAL 1000 / FINAL HOLDOUT 1000)", flush=True)
    print("=" * 80, flush=True)

    pools_dir = Path("experiments/v3/pools")
    pools_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect all historical S1 IDs to exclude
    excluded_s1 = set()

    # V1 chunks
    chunks_dir = Path("training_features_chunks")
    v1_s1 = set()
    if chunks_dir.exists():
        for f in sorted(chunks_dir.glob("*.parquet")):
            s = pd.read_parquet(f, columns=["source1_entity_id"])
            v1_s1.update(s["source1_entity_id"].unique())
    excluded_s1.update(v1_s1)
    print(f"Excluded V1 chunk S1s: {len(v1_s1):,}", flush=True)

    # V2 parquets
    v2_s1 = set()
    for p_name in [
        "experiments/v2/data/v2_a_train_features.parquet",
        "experiments/v2/data/v2_b_train_features.parquet",
        "experiments/v2/data/v2_c_train_features.parquet",
    ]:
        p = Path(p_name)
        if p.exists():
            s = pd.read_parquet(p, columns=["source1_entity_id"])
            v2_s1.update(s["source1_entity_id"].unique())
    excluded_s1.update(v2_s1)
    print(f"Excluded V2 training S1s: {len(v2_s1):,}", flush=True)

    # val_1000
    bundle_path = Path(
        r"C:\Users\Sneha Singh\.gemini\antigravity\brain\47892a4a-c3be-4331-990a-46831df8639d\scratch\val_1000_candidates_bundle.joblib"
    )
    val_1000_s1 = set()
    if bundle_path.exists():
        b = joblib.load(bundle_path)
        val_1000_s1.update(b["val_ids"])
        del b
    excluded_s1.update(val_1000_s1)
    print(f"Excluded val_1000 S1s: {len(val_1000_s1):,}", flush=True)

    # Diagnostic heldout_5000_s1.tsv
    diag_5000_path = Path("experiments/v3/diagnostics/heldout_5000_s1.tsv")
    diag_5000_s1 = set()
    if diag_5000_path.exists():
        d_df = pd.read_csv(diag_5000_path, sep="\t", usecols=["source1_entity_id"], dtype=str)
        diag_5000_s1.update(d_df["source1_entity_id"].unique())
    excluded_s1.update(diag_5000_s1)
    print(f"Excluded diagnostic 5000 S1s: {len(diag_5000_s1):,}", flush=True)
    print(f"Total historically touched S1s excluded: {len(excluded_s1):,}", flush=True)

    # 2. Load official ground truth and filter to untouched split == 'train' S1s
    gt = load_source_tsv("dataset/train/train_ground_truth.tsv")
    gt = create_train_val_split(gt)
    eligible_gt = gt[(gt["split"] == "train") & (~gt["source1_entity_id"].isin(excluded_s1))].copy()
    eligible_gt = eligible_gt.sort_values("source1_entity_id").reset_index(drop=True)
    print(f"Eligible untouched train S1 entities: {len(eligible_gt):,}", flush=True)

    # Sample 5,000 entities deterministically with seed 302026
    sampled_5000 = eligible_gt.sample(n=5000, random_state=302026).reset_index(drop=True)
    dev_ids = set(sampled_5000.iloc[:3000]["source1_entity_id"])
    internal_ids = set(sampled_5000.iloc[3000:4000]["source1_entity_id"])
    final_holdout_ids = set(sampled_5000.iloc[4000:5000]["source1_entity_id"])
    all_5000_ids = dev_ids | internal_ids | final_holdout_ids

    # Strict assertions
    assert len(dev_ids) == 3000
    assert len(internal_ids) == 1000
    assert len(final_holdout_ids) == 1000
    assert len(all_5000_ids) == 5000
    assert len(dev_ids & internal_ids) == 0
    assert len(dev_ids & final_holdout_ids) == 0
    assert len(internal_ids & final_holdout_ids) == 0
    assert len(all_5000_ids & excluded_s1) == 0

    del gt, eligible_gt
    gc.collect()

    # 3. Fetch full S1 text records from dataset/train/train_source1.tsv
    s1_rows = []
    for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", chunksize=500_000, dtype=str):
        sub = chunk[chunk["entity_id"].isin(all_5000_ids)]
        if not sub.empty:
            s1_rows.append(sub)
    s1_df = pd.concat(s1_rows, ignore_index=True)

    merged = sampled_5000.merge(s1_df, left_on="source1_entity_id", right_on="entity_id", how="inner")
    assert len(merged) == 5000, f"Expected 5000 merged rows, got {len(merged)}"

    dev_df = merged[merged["source1_entity_id"].isin(dev_ids)].sort_values("source1_entity_id").reset_index(drop=True)
    internal_df = merged[merged["source1_entity_id"].isin(internal_ids)].sort_values("source1_entity_id").reset_index(drop=True)
    final_df = merged[merged["source1_entity_id"].isin(final_holdout_ids)].sort_values("source1_entity_id").reset_index(drop=True)

    dev_path = pools_dir / "dev_3000_s1.tsv"
    internal_path = pools_dir / "internal_check_1000_s1.tsv"
    final_path = pools_dir / "final_holdout_1000_s1.tsv"

    dev_df.to_csv(dev_path, sep="\t", index=False)
    internal_df.to_csv(internal_path, sep="\t", index=False)
    final_df.to_csv(final_path, sep="\t", index=False)

    manifest = {
        "firewall_seed": 302026,
        "disjointness_assertions": {
            "dev_vs_internal_overlap": len(dev_ids & internal_ids),
            "dev_vs_final_holdout_overlap": len(dev_ids & final_holdout_ids),
            "internal_vs_final_holdout_overlap": len(internal_ids & final_holdout_ids),
            "overlap_with_v1_train_chunks": len(all_5000_ids & v1_s1),
            "overlap_with_v2_train_parquets": len(all_5000_ids & v2_s1),
            "overlap_with_val_1000": len(all_5000_ids & val_1000_s1),
            "overlap_with_diagnostic_5000": len(all_5000_ids & diag_5000_s1),
        },
        "pools": {
            "dev_3000": summarize_pool(dev_df, "DEV_3000"),
            "internal_check_1000": summarize_pool(internal_df, "INTERNAL_CHECK_1000"),
            "final_holdout_1000": summarize_pool(final_df, "FINAL_HOLDOUT_1000"),
        },
    }

    manifest_path = pools_dir / "pools_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nValidation Firewall Manifest:", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"\nSaved pools to {pools_dir}", flush=True)


if __name__ == "__main__":
    main()
