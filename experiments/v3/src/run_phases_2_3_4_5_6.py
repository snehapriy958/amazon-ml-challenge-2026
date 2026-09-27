"""V3 Comprehensive Diagnostics Pipeline (Phases 2, 3, 4, 5, 6).

Strictly READ-ONLY with respect to V1 and V2-C artifacts.
Writes compact diagnostic outputs ONLY to experiments/v3/diagnostics/.
"""

from collections import Counter, defaultdict
import gc
import json
from math import ceil
from pathlib import Path
import re
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
import psutil
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path.cwd()))
from code.business_entity_resolution.src.evaluation import compute_entity_f_beta, evaluate_macro_f_beta
from code.business_entity_resolution.src.feature_engineering import LOCKED_FEATURE_COLUMNS
from code.business_entity_resolution.src.preprocessing import (
    add_normalized_columns,
    create_train_val_split,
    load_source_tsv,
    parse_match_ids,
)

DIAG_DIR = Path("experiments/v3/diagnostics")
DIAG_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.850, 0.875, 0.900, 0.9125, 0.925, 0.9375, 0.950, 0.9625, 0.975, 0.985, 0.990]
BATCH_SIZE = 500_000


def check_ram(stage: str) -> float:
    avail = psutil.virtual_memory().available / (1024 ** 3)
    if avail < 1.2:
        gc.collect()
        avail = psutil.virtual_memory().available / (1024 ** 3)
    return avail


def extract_digits(s: str) -> set:
    return set(re.findall(r"\d+", s))


def extract_postal(s: str) -> set:
    # US 5-digit zip or India 6-digit PIN or France 5-digit postal
    return set(re.findall(r"\b\d{5,6}\b", s))


def main():
    t_global_start = time.time()
    print("=" * 85, flush=True)
    print("=== V3 DIAGNOSTICS PIPELINE (PHASES 2, 3, 4, 5, 6) ===", flush=True)
    print("=" * 85, flush=True)

    # =========================================================================
    # PHASE 2: VERIFY THE V2-C TRAINING POPULATION
    # =========================================================================
    print("\n" + "=" * 85, flush=True)
    print("PHASE 2: VERIFYING V2-C TRAINING POPULATION", flush=True)
    print("=" * 85, flush=True)

    v2c_df = pd.read_parquet("experiments/v2/data/v2_c_train_features.parquet")
    v2a_df = pd.read_parquet("experiments/v2/data/v2_a_train_features.parquet", columns=["source1_entity_id", "candidate_entity_id", "label"])

    gt_all = load_source_tsv("dataset/train/train_ground_truth.tsv")
    gt_all = create_train_val_split(gt_all)

    bundle_path = Path(r"C:\Users\Sneha Singh\.gemini\antigravity\brain\47892a4a-c3be-4331-990a-46831df8639d\scratch\val_1000_candidates_bundle.joblib")
    bundle_1000 = joblib.load(bundle_path)
    val_1000_ids = set(bundle_1000["val_ids"])
    del bundle_1000
    gc.collect()

    train_gt_all = gt_all[gt_all["split"] == "train"].copy()
    clean_train_gt = train_gt_all[~train_gt_all["source1_entity_id"].isin(val_1000_ids)].copy()
    sampled_50k_gt = clean_train_gt.sample(n=50_000, random_state=42)
    sampled_50k_ids = set(sampled_50k_gt["source1_entity_id"])

    v2a_s1_ids = set(v2a_df["source1_entity_id"].unique())
    v2c_s1_ids = set(v2c_df["source1_entity_id"].unique())

    pos_rows = v2c_df[v2c_df["label"] == 1]
    neg_rows = v2c_df[v2c_df["label"] == 0]
    rows_per_s1 = v2c_df.groupby("source1_entity_id").size()
    pos_per_s1 = pos_rows.groupby("source1_entity_id").size()
    neg_per_s1 = neg_rows.groupby("source1_entity_id").size()

    s1_in_50k_with_pairs = sampled_50k_ids & v2c_s1_ids
    s1_in_50k_no_pairs = sampled_50k_ids - v2c_s1_ids
    overlap_50k_v2a = sampled_50k_ids & v2a_s1_ids

    phase2_stats = {
        "total_rows": int(len(v2c_df)),
        "unique_s1_count": int(len(v2c_s1_ids)),
        "unique_candidate_id_count": int(v2c_df["candidate_entity_id"].nunique()),
        "positive_rows": int(len(pos_rows)),
        "negative_rows": int(len(neg_rows)),
        "unique_s1_among_positives": int(pos_rows["source1_entity_id"].nunique()),
        "unique_s1_among_negatives": int(neg_rows["source1_entity_id"].nunique()),
        "s1_only_positives": int(len(set(pos_rows["source1_entity_id"]) - set(neg_rows["source1_entity_id"]))),
        "s1_only_negatives": int(len(set(neg_rows["source1_entity_id"]) - set(pos_rows["source1_entity_id"]))),
        "s1_id_prefixes": v2c_df["source1_entity_id"].str.slice(0, 3).value_counts().to_dict(),
        "candidate_id_prefixes": v2c_df["candidate_entity_id"].str.slice(0, 3).value_counts().to_dict(),
        "candidate_id_prefixes_pos": pos_rows["candidate_entity_id"].str.slice(0, 3).value_counts().to_dict(),
        "candidate_id_prefixes_neg": neg_rows["candidate_entity_id"].str.slice(0, 3).value_counts().to_dict(),
        "rows_per_s1_quantiles": {
            "min": int(rows_per_s1.min()),
            "p10": float(np.percentile(rows_per_s1, 10)),
            "p25": float(np.percentile(rows_per_s1, 25)),
            "median": float(rows_per_s1.median()),
            "mean": float(rows_per_s1.mean()),
            "p75": float(np.percentile(rows_per_s1, 75)),
            "p90": float(np.percentile(rows_per_s1, 90)),
            "p95": float(np.percentile(rows_per_s1, 95)),
            "p99": float(np.percentile(rows_per_s1, 99)),
            "max": int(rows_per_s1.max()),
        },
        "explanation_of_166865": {
            "v2_a_unique_clean_s1": int(len(v2a_s1_ids)),
            "v2_c_sampled_s1_target": 50000,
            "sampled_50k_with_at_least_1_pair": int(len(s1_in_50k_with_pairs)),
            "sampled_50k_with_0_pairs_empty_gt_and_no_cands": int(len(s1_in_50k_no_pairs)),
            "overlap_between_v2a_and_50k_with_pairs": int(len(s1_in_50k_with_pairs & v2a_s1_ids)),
            "newly_added_s1_from_50k": int(len(s1_in_50k_with_pairs - v2a_s1_ids)),
            "exact_union_check": int(len(v2a_s1_ids | s1_in_50k_with_pairs)),
        },
    }
    print(json.dumps(phase2_stats, indent=2), flush=True)

    del v2c_df, v2a_df, pos_rows, neg_rows
    gc.collect()

    # =========================================================================
    # PHASE 3: CONSTRUCT GENUINELY HELD-OUT 5,000-S1 VALIDATION POOL
    # =========================================================================
    print("\n" + "=" * 85, flush=True)
    print("PHASE 3: CONSTRUCTING GENUINELY HELD-OUT 5,000-S1 POOL", flush=True)
    print("=" * 85, flush=True)

    # Also gather any S1 IDs from V1 training_features_chunks just in case
    v1_chunk_s1 = set()
    for cf in Path("training_features_chunks").glob("train_features_chunk_*.parquet"):
        cdf = pd.read_parquet(cf, columns=["source1_entity_id"])
        v1_chunk_s1.update(cdf["source1_entity_id"].unique())
        del cdf

    excluded_s1_ids = v2c_s1_ids | sampled_50k_ids | v2a_s1_ids | v1_chunk_s1 | val_1000_ids
    print(f"Total excluded S1 IDs (V1 + V2-A + V2-B + V2-C + val_1000): {len(excluded_s1_ids):,}", flush=True)

    eligible_heldout_gt = train_gt_all[~train_gt_all["source1_entity_id"].isin(excluded_s1_ids)].copy()
    print(f"Eligible unseen train-split S1 entities: {len(eligible_heldout_gt):,}", flush=True)

    heldout_5000_gt = eligible_heldout_gt.sample(n=5_000, random_state=2026).copy()
    heldout_s1_ids = heldout_5000_gt["source1_entity_id"].tolist()
    heldout_s1_set = set(heldout_s1_ids)

    assert len(heldout_s1_set & v2c_s1_ids) == 0, "Leakage with V2-C!"
    assert len(heldout_s1_set & sampled_50k_ids) == 0, "Leakage with 50k sample!"
    assert len(heldout_s1_set & v1_chunk_s1) == 0, "Leakage with V1 chunks!"
    assert len(heldout_s1_set & val_1000_ids) == 0, "Overlap with val_1000!"

    # Save compact held-out S1 list
    heldout_tsv_path = DIAG_DIR / "heldout_5000_s1.tsv"
    heldout_5000_gt[["source1_entity_id", "matched_entity_ids"]].to_csv(heldout_tsv_path, sep="\t", index=False)
    print(f"Saved 5,000 held-out S1 entities to {heldout_tsv_path} ({heldout_tsv_path.stat().st_size / 1024:.1f} KB)", flush=True)

    gt_dict = {}
    all_gt_targets = set()
    n_empty_s1 = 0
    n_single_match_s1 = 0
    n_multi_match_s1 = 0
    s2_gt_links = 0
    s3_gt_links = 0

    for r in heldout_5000_gt.itertuples(index=False):
        m_set = set(parse_match_ids(r.matched_entity_ids))
        gt_dict[r.source1_entity_id] = m_set
        all_gt_targets.update(m_set)
        if len(m_set) == 0:
            n_empty_s1 += 1
        elif len(m_set) == 1:
            n_single_match_s1 += 1
        else:
            n_multi_match_s1 += 1
        for m in m_set:
            if m.startswith("S2-"):
                s2_gt_links += 1
            elif m.startswith("S3-"):
                s3_gt_links += 1

    total_gt_links = s2_gt_links + s3_gt_links
    del gt_all, train_gt_all, clean_train_gt, eligible_heldout_gt, heldout_5000_gt
    gc.collect()

    # Load S1 records for the 5,000 held-out entities + check countries in full train_source1
    print("Loading train_source1.tsv...", flush=True)
    s1_df_full = load_source_tsv("dataset/train/train_source1.tsv")
    train_s1_countries_all = s1_df_full["country"].fillna("").str.strip().str.lower().value_counts().to_dict()
    heldout_s1_df = s1_df_full[s1_df_full["entity_id"].isin(heldout_s1_set)].copy()
    heldout_s1_df = add_normalized_columns(heldout_s1_df)
    del s1_df_full
    gc.collect()

    re_tok = re.compile(r"[a-z0-9]+")
    s1_lookup = {}
    for r in heldout_s1_df.itertuples(index=False):
        n_norm = str(r.name_norm) if pd.notna(r.name_norm) else ""
        a_norm = str(r.address_norm) if pd.notna(r.address_norm) else ""
        n_toks = set(t for t in re_tok.findall(n_norm) if len(t) >= 3)
        a_toks = set(t for t in re_tok.findall(a_norm) if len(t) >= 3)
        s1_lookup[r.entity_id] = {
            "name": str(r.business_name) if pd.notna(r.business_name) else "",
            "addr": str(r.business_address) if pd.notna(r.business_address) else "",
            "country": str(r.country).strip().lower() if pd.notna(r.country) else "",
            "name_norm": n_norm,
            "addr_norm": a_norm,
            "tokens": n_toks | a_toks,
            "name_tokens": n_toks,
            "addr_tokens": a_toks,
        }

    heldout_countries = Counter(rec["country"] for rec in s1_lookup.values())
    phase3_stats = {
        "heldout_s1_count": len(heldout_s1_ids),
        "total_gt_links": total_gt_links,
        "s2_gt_links": s2_gt_links,
        "s3_gt_links": s3_gt_links,
        "mean_gt_links_per_s1": total_gt_links / len(heldout_s1_ids),
        "empty_s1_count": n_empty_s1,
        "single_match_s1_count": n_single_match_s1,
        "multi_match_s1_count": n_multi_match_s1,
        "heldout_countries": dict(heldout_countries),
        "full_train_s1_countries": train_s1_countries_all,
    }
    print(json.dumps(phase3_stats, indent=2), flush=True)

    # =========================================================================
    # PHASE 4: CANDIDATE RECALL DIAGNOSTIC
    # =========================================================================
    print("\n" + "=" * 85, flush=True)
    print("PHASE 4: CANDIDATE RECALL DIAGNOSTIC (UNCAPPED vs CAPPED-350 vs BALANCED)", flush=True)
    print("=" * 85, flush=True)

    # Group query vocab by country
    country_queries = defaultdict(dict)
    q_names = defaultdict(set)
    q_addrs = defaultdict(set)
    q_tokens = defaultdict(set)

    for sid, rec in s1_lookup.items():
        c = rec["country"]
        country_queries[c][sid] = rec
        if rec["name_norm"]:
            q_names[c].add(rec["name_norm"])
        if rec["addr_norm"]:
            q_addrs[c].add(rec["addr_norm"])
        q_tokens[c].update(rec["tokens"])

    # Pass 1 over train_source2 and train_source3:
    # Build exact name/addr indexes and token indexes for the 5,000 queries,
    # PLUS global token frequency for the 5,000 queries' tokens (to also test global vs country freq),
    # AND retain text records for all GT targets so we can inspect missed GT links!
    cand_name_ids = {c: defaultdict(list) for c in country_queries}
    cand_addr_ids = {c: defaultdict(list) for c in country_queries}
    s2_tok_ids = {c: defaultdict(list) for c in country_queries}
    s3_tok_ids = {c: defaultdict(list) for c in country_queries}
    s2_tok_freq_global = Counter()
    s3_tok_freq_global = Counter()
    all_5k_tokens = set().union(*q_tokens.values())

    gt_target_records = {}

    t_idx0 = time.time()
    for src_idx, src_path in enumerate(["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"], 1):
        print(f"  Scanning [{src_idx}/2] {src_path}...", flush=True)
        for chunk in pd.read_csv(src_path, sep="\t", chunksize=1_000_000, dtype=str):
            # Capture GT target text records
            gt_sub = chunk[chunk["entity_id"].isin(all_gt_targets)]
            if not gt_sub.empty:
                for cid, n_r, a_r, c_r in zip(gt_sub["entity_id"], gt_sub["business_name"], gt_sub["business_address"], gt_sub["country"]):
                    gt_target_records[cid] = (
                        "" if pd.isna(n_r) else str(n_r),
                        "" if pd.isna(a_r) else str(a_r),
                        "" if pd.isna(c_r) else str(c_r).strip().lower(),
                    )

            chunk["c_norm"] = chunk["country"].fillna("").str.strip().str.lower()
            for c in country_queries:
                sub = chunk[chunk["c_norm"] == c]
                if sub.empty:
                    continue
                qn_c = q_names[c]
                qa_c = q_addrs[c]
                qt_c = q_tokens[c]
                cname_m = cand_name_ids[c]
                caddr_m = cand_addr_ids[c]
                ctok_m = s2_tok_ids[c] if src_idx == 1 else s3_tok_ids[c]
                gfreq_m = s2_tok_freq_global if src_idx == 1 else s3_tok_freq_global

                for cid, n_raw, a_raw in zip(sub["entity_id"], sub["business_name"], sub["business_address"]):
                    n_str = "" if pd.isna(n_raw) else str(n_raw).lower()
                    a_str = "" if pd.isna(a_raw) else str(a_raw).lower()
                    n_t = re_tok.findall(n_str)
                    a_t = re_tok.findall(a_str)
                    n_norm = " ".join(n_t)
                    a_norm = " ".join(a_t)

                    if n_norm in qn_c:
                        cname_m[n_norm].append(cid)
                    if a_norm in qa_c:
                        caddr_m[a_norm].append(cid)

                    rec_toks = {t for t in n_t + a_t if len(t) >= 3}
                    common_global = rec_toks & all_5k_tokens
                    for t in common_global:
                        gfreq_m[t] += 1
                    for t in rec_toks & qt_c:
                        ctok_m[t].append(cid)

    print(f"  Index scan completed in {time.time()-t_idx0:.1f}s | Avail RAM: {check_ram('Index scan'):.2f} GB", flush=True)

    # Evaluate Candidate Generation Regimes for all 5,000 Held-Out S1s:
    # Regime 1: UNCAPPED Blocker (Top-3 rare tokens with max(f2, f3) <= 100,000 + exact name/addr + same country)
    # Regime 2: CAPPED-350 File-Order Blocker (Exact logic of run_v2_c_full_inference.py: S2 first then S3, truncated at 350)
    # Regime 3: MULTI-TOKEN + RARITY PRIORITIZED CAPPED-350 / CAPPED-600 (Ranks candidates by token overlap count + rarity, interleaving S2/S3)
    # Regime 4: TOP-5 RARE TOKENS (or Top-3 Name + Top-2 Addr tokens) UNCAPPED

    def eval_blocker_regime(regime_name, cands_per_s1_dict):
        total_links = 0
        found_links = 0
        s2_total, s2_found = 0, 0
        s3_total, s3_found = 0, 0
        entity_recalls_nonempty = []
        entity_recalls_all = []
        cand_counts = []
        zero_cand_s1 = 0

        by_country = defaultdict(lambda: {"s1": 0, "gt_links": 0, "found_links": 0, "entity_recalls": []})
        by_card = {
            "empty_s1": {"s1": 0, "zero_cands": 0, "cand_counts": []},
            "single_match_s1": {"s1": 0, "gt_links": 0, "found_links": 0, "entity_recalls": []},
            "multi_match_s1": {"s1": 0, "gt_links": 0, "found_links": 0, "entity_recalls": []},
        }
        missed_links_list = []

        for sid in heldout_s1_ids:
            cands = cands_per_s1_dict[sid]
            cand_set = set(cands)
            gt_set = gt_dict[sid]
            ctry = s1_lookup[sid]["country"]

            cand_counts.append(len(cand_set))
            if len(cand_set) == 0:
                zero_cand_s1 += 1

            by_country[ctry]["s1"] += 1

            if len(gt_set) == 0:
                entity_recalls_all.append(1.0)
                by_card["empty_s1"]["s1"] += 1
                by_card["empty_s1"]["cand_counts"].append(len(cand_set))
                if len(cand_set) == 0:
                    by_card["empty_s1"]["zero_cands"] += 1
            else:
                hits = len(gt_set & cand_set)
                rec_val = hits / len(gt_set)
                entity_recalls_nonempty.append(rec_val)
                entity_recalls_all.append(rec_val)
                total_links += len(gt_set)
                found_links += hits

                by_country[ctry]["gt_links"] += len(gt_set)
                by_country[ctry]["found_links"] += hits
                by_country[ctry]["entity_recalls"].append(rec_val)

                bucket = "single_match_s1" if len(gt_set) == 1 else "multi_match_s1"
                by_card[bucket]["s1"] += 1
                by_card[bucket]["gt_links"] += len(gt_set)
                by_card[bucket]["found_links"] += hits
                by_card[bucket]["entity_recalls"].append(rec_val)

                for m in gt_set:
                    if m.startswith("S2-"):
                        s2_total += 1
                        if m in cand_set:
                            s2_found += 1
                    else:
                        s3_total += 1
                        if m in cand_set:
                            s3_found += 1
                    if m not in cand_set:
                        missed_links_list.append((sid, m))

        cand_arr = np.array(cand_counts)
        summary = {
            "regime": regime_name,
            "micro_link_recall": float(found_links / total_links) if total_links > 0 else 0.0,
            "macro_entity_recall_nonempty": float(np.mean(entity_recalls_nonempty)),
            "macro_entity_recall_all_including_empty": float(np.mean(entity_recalls_all)),
            "total_gt_links": int(total_links),
            "found_gt_links": int(found_links),
            "missed_gt_links": int(total_links - found_links),
            "s2_link_recall": float(s2_found / s2_total) if s2_total > 0 else 0.0,
            "s2_missed": int(s2_total - s2_found),
            "s3_link_recall": float(s3_found / s3_total) if s3_total > 0 else 0.0,
            "s3_missed": int(s3_total - s3_found),
            "zero_candidate_s1_count": int(zero_cand_s1),
            "zero_candidate_s1_pct": float(zero_cand_s1 / len(heldout_s1_ids) * 100),
            "candidate_count_dist": {
                "mean": float(np.mean(cand_arr)),
                "median": float(np.median(cand_arr)),
                "p90": float(np.percentile(cand_arr, 90)),
                "p95": float(np.percentile(cand_arr, 95)),
                "p99": float(np.percentile(cand_arr, 99)),
                "max": int(np.max(cand_arr)),
                "total_pairs": int(np.sum(cand_arr)),
            },
            "by_country": {
                c: {
                    "s1_count": d["s1"],
                    "gt_links": d["gt_links"],
                    "found_links": d["found_links"],
                    "missed_links": d["gt_links"] - d["found_links"],
                    "micro_recall": float(d["found_links"] / d["gt_links"]) if d["gt_links"] > 0 else 0.0,
                    "macro_recall_nonempty": float(np.mean(d["entity_recalls"])) if d["entity_recalls"] else 0.0,
                }
                for c, d in by_country.items()
            },
            "by_cardinality": {
                "empty_s1": {
                    "s1_count": by_card["empty_s1"]["s1"],
                    "zero_cands": by_card["empty_s1"]["zero_cands"],
                    "mean_cands": float(np.mean(by_card["empty_s1"]["cand_counts"])) if by_card["empty_s1"]["cand_counts"] else 0.0,
                },
                "single_match_s1": {
                    "s1_count": by_card["single_match_s1"]["s1"],
                    "gt_links": by_card["single_match_s1"]["gt_links"],
                    "found_links": by_card["single_match_s1"]["found_links"],
                    "micro_recall": float(by_card["single_match_s1"]["found_links"] / by_card["single_match_s1"]["gt_links"]),
                    "macro_recall": float(np.mean(by_card["single_match_s1"]["entity_recalls"])),
                },
                "multi_match_s1": {
                    "s1_count": by_card["multi_match_s1"]["s1"],
                    "gt_links": by_card["multi_match_s1"]["gt_links"],
                    "found_links": by_card["multi_match_s1"]["found_links"],
                    "micro_recall": float(by_card["multi_match_s1"]["found_links"] / by_card["multi_match_s1"]["gt_links"]),
                    "macro_recall": float(np.mean(by_card["multi_match_s1"]["entity_recalls"])),
                },
            },
        }
        return summary, missed_links_list

    # Build candidate lists for each regime
    cands_uncapped = {}
    cands_capped350_fileorder = {}
    cands_prioritized_500 = {}
    cands_top5_uncapped = {}

    # We will also collect the union of candidates to score in Phase 5:
    # (All Capped-350 candidates + all Prioritized-600 candidates) so we can evaluate V2-C on BOTH!
    cands_to_score_by_s1 = {}
    needed_cids_for_scoring = set(all_gt_targets)

    for sid in heldout_s1_ids:
        rec = s1_lookup[sid]
        c = rec["country"]
        cname_m = cand_name_ids[c]
        caddr_m = cand_addr_ids[c]
        s2_tm = s2_tok_ids[c]
        s3_tm = s3_tok_ids[c]

        # Sort tokens by max(f2, f3) <= 100,000
        tok_info = []
        for t in rec["tokens"]:
            f2 = s2_tok_freq_global.get(t, 0)
            f3 = s3_tok_freq_global.get(t, 0)
            freq = max(f2, f3)
            if 0 < freq <= 100_000:
                tok_info.append((freq, t))
        tok_info.sort()
        top3_toks = [t for _, t in tok_info[:3]]
        top5_toks = [t for _, t in tok_info[:5]]

        # 1. Exact name & address matches
        exact_cands = []
        seen_exact = set()
        if rec["name_norm"] and rec["name_norm"] in cname_m:
            for cid in cname_m[rec["name_norm"]]:
                if cid not in seen_exact:
                    seen_exact.add(cid)
                    exact_cands.append(cid)
        if rec["addr_norm"] and rec["addr_norm"] in caddr_m:
            for cid in caddr_m[rec["addr_norm"]]:
                if cid not in seen_exact:
                    seen_exact.add(cid)
                    exact_cands.append(cid)

        # Regime 1: Uncapped Top-3
        uncapped_set = set(exact_cands)
        for t in top3_toks:
            uncapped_set.update(s2_tm.get(t, ()))
            uncapped_set.update(s3_tm.get(t, ()))
        cands_uncapped[sid] = list(uncapped_set)

        # Regime 2: Capped-350 in S2-then-S3 file order (exact replica of run_v2_c_full_inference.py)
        ordered_350 = list(exact_cands[:350])
        seen_350 = set(ordered_350)
        if len(ordered_350) < 350:
            for t in top3_toks:
                # In run_v2_c_full_inference.py, S2 was indexed first then S3
                combined_list = s2_tm.get(t, []) + s3_tm.get(t, [])
                for cid in combined_list:
                    if cid not in seen_350:
                        seen_350.add(cid)
                        ordered_350.append(cid)
                    if len(ordered_350) >= 350:
                        break
                if len(ordered_350) >= 350:
                    break
        cands_capped350_fileorder[sid] = ordered_350

        # Regime 3: Multi-token + Rarity prioritized (score each candidate by how many of the query tokens it shares!)
        cand_tok_hits = Counter()
        for freq, t in tok_info[:6]:  # check top-6 rarest tokens
            weight = 1.0 + (1.0 / (1.0 + np.log10(max(1, freq))))
            for cid in s2_tm.get(t, ()):
                cand_tok_hits[cid] += weight
            for cid in s3_tm.get(t, ()):
                cand_tok_hits[cid] += weight
        for cid in exact_cands:
            cand_tok_hits[cid] += 100.0

        prioritized_all = [cid for cid, _ in cand_tok_hits.most_common(600)]
        cands_prioritized_500[sid] = prioritized_all[:500]

        # Regime 4: Uncapped Top-5
        uncapped5_set = set(exact_cands)
        for t in top5_toks:
            uncapped5_set.update(s2_tm.get(t, ()))
            uncapped5_set.update(s3_tm.get(t, ()))
        cands_top5_uncapped[sid] = list(uncapped5_set)

        # Union for Phase 5 model scoring (Capped-350 + Prioritized-600 + any found GT links in Uncapped Top-3)
        found_gt_in_uncapped = gt_dict[sid] & uncapped_set
        to_score = list(dict.fromkeys(ordered_350 + prioritized_all + list(found_gt_in_uncapped)))
        cands_to_score_by_s1[sid] = to_score
        needed_cids_for_scoring.update(to_score)

    sum_uncapped, missed_uncapped = eval_blocker_regime("1_uncapped_top3_v2c_blocker", cands_uncapped)
    sum_cap350, missed_cap350 = eval_blocker_regime("2_capped350_s2first_test_inference_blocker", cands_capped350_fileorder)
    sum_prio500, _ = eval_blocker_regime("3_multi_token_rarity_prioritized_top500", cands_prioritized_500)
    sum_top5, _ = eval_blocker_regime("4_uncapped_top5_rare_tokens", cands_top5_uncapped)

    # Diagnose WHY missed_uncapped links were missed by the top-3 blocker
    missed_reasons = Counter()
    missed_examples = []
    for sid, mid in missed_uncapped:
        s1_rec = s1_lookup[sid]
        t_rec = gt_target_records.get(mid, ("", "", ""))
        t_name, t_addr, t_ctry = t_rec
        t_ntoks = set(t for t in re_tok.findall(t_name.lower()) if len(t) >= 3)
        t_atoks = set(t for t in re_tok.findall(t_addr.lower()) if len(t) >= 3)
        t_all_toks = t_ntoks | t_atoks

        shared_all = s1_rec["tokens"] & t_all_toks
        if s1_rec["country"] != t_ctry:
            reason = "country_mismatch"
        elif len(shared_all) == 0:
            reason = "zero_shared_tokens_len_ge_3_transliteration_or_abbrev"
        else:
            # Shared tokens exist, but weren't in top-3 rarest tokens (or had freq > 100k)
            usable_s1 = sorted([(max(s2_tok_freq_global.get(t, 0), s3_tok_freq_global.get(t, 0)), t) for t in s1_rec["tokens"] if 0 < max(s2_tok_freq_global.get(t, 0), s3_tok_freq_global.get(t, 0)) <= 100_000])
            top3_set = {t for _, t in usable_s1[:3]}
            shared_usable = [t for t in shared_all if max(s2_tok_freq_global.get(t, 0), s3_tok_freq_global.get(t, 0)) <= 100_000]
            if not shared_usable:
                reason = "all_shared_tokens_exceeded_100k_freq"
            else:
                reason = "shared_tokens_ranked_4th_or_higher_outside_top3"
        missed_reasons[reason] += 1
        if len(missed_examples) < 15:
            missed_examples.append({
                "s1_id": sid,
                "target_id": mid,
                "reason": reason,
                "s1_name": s1_rec["name"],
                "target_name": t_name,
                "s1_addr": s1_rec["addr"],
                "target_addr": t_addr,
                "shared_tokens": sorted(list(shared_all)),
            })

    phase4_results = {
        "uncapped_top3_blocker": sum_uncapped,
        "capped350_test_inference_blocker": sum_cap350,
        "prioritized_top500_blocker": sum_prio500,
        "uncapped_top5_blocker": sum_top5,
        "missed_by_uncapped_reasons": dict(missed_reasons),
        "missed_by_uncapped_examples": missed_examples,
        "extra_links_lost_purely_to_350_cap": int(sum_uncapped["found_gt_links"] - sum_cap350["found_gt_links"]),
        "s3_links_lost_purely_to_350_cap": int(sum_uncapped["s3_missed"] - sum_cap350["s3_missed"]) * -1,
    }
    print("\n--- PHASE 4 SUMMARY ---", flush=True)
    print(f"Uncapped Top-3 Blocker:     Micro Recall = {sum_uncapped['micro_link_recall']*100:.2f}% | Macro (non-empty) = {sum_uncapped['macro_entity_recall_nonempty']*100:.2f}% | S2 Recall = {sum_uncapped['s2_link_recall']*100:.2f}% | S3 Recall = {sum_uncapped['s3_link_recall']*100:.2f}% | Missed = {sum_uncapped['missed_gt_links']}", flush=True)
    print(f"Capped-350 (Test Logic):    Micro Recall = {sum_cap350['micro_link_recall']*100:.2f}% | Macro (non-empty) = {sum_cap350['macro_entity_recall_nonempty']*100:.2f}% | S2 Recall = {sum_cap350['s2_link_recall']*100:.2f}% | S3 Recall = {sum_cap350['s3_link_recall']*100:.2f}% | Missed = {sum_cap350['missed_gt_links']}", flush=True)
    print(f"Prioritized-500 Blocker:    Micro Recall = {sum_prio500['micro_link_recall']*100:.2f}% | Macro (non-empty) = {sum_prio500['macro_entity_recall_nonempty']*100:.2f}% | S2 Recall = {sum_prio500['s2_link_recall']*100:.2f}% | S3 Recall = {sum_prio500['s3_link_recall']*100:.2f}% | Missed = {sum_prio500['missed_gt_links']}", flush=True)
    print(f"Uncapped Top-5 Blocker:     Micro Recall = {sum_top5['micro_link_recall']*100:.2f}% | Macro (non-empty) = {sum_top5['macro_entity_recall_nonempty']*100:.2f}% | Missed = {sum_top5['missed_gt_links']}", flush=True)
    print("Reasons for Uncapped Missed GT Links:", dict(missed_reasons), flush=True)

    # Free index structures before Phase 5 scoring
    del cand_name_ids, cand_addr_ids, s2_tok_ids, s3_tok_ids, cands_uncapped, cands_top5_uncapped
    gc.collect()

    # =========================================================================
    # PHASE 5: VALIDATION ROBUSTNESS ON 5,000 HELD-OUT POOL
    # =========================================================================
    print("\n" + "=" * 85, flush=True)
    print("PHASE 5: VALIDATION ROBUSTNESS ON 5,000 HELD-OUT POOL", flush=True)
    print("=" * 85, flush=True)

    print(f"Fetching text records for {len(needed_cids_for_scoring):,} candidate IDs...", flush=True)
    t_f0 = time.time()
    cand_records = dict(gt_target_records)
    remaining_needed = needed_cids_for_scoring - set(cand_records.keys())

    for src_path in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        for chunk in pd.read_csv(src_path, sep="\t", chunksize=1_000_000, dtype=str):
            sub = chunk[chunk["entity_id"].isin(remaining_needed)]
            if sub.empty:
                continue
            for cid, n_r, a_r, c_r in zip(sub["entity_id"], sub["business_name"], sub["business_address"], sub["country"]):
                cand_records[cid] = (
                    "" if pd.isna(n_r) else str(n_r),
                    "" if pd.isna(a_r) else str(a_r),
                    "" if pd.isna(c_r) else str(c_r).strip().lower(),
                )
    print(f"Fetched {len(cand_records):,} candidate records in {time.time()-t_f0:.1f}s | Avail RAM: {check_ram('Phase 5 fetch'):.2f} GB", flush=True)

    # Build flat pairs to score
    all_pairs_s1 = []
    all_pairs_cid = []
    for sid in heldout_s1_ids:
        for cid in cands_to_score_by_s1[sid]:
            all_pairs_s1.append(sid)
            all_pairs_cid.append(cid)

    total_eval_pairs = len(all_pairs_s1)
    print(f"Scoring {total_eval_pairs:,} candidate pairs across 5,000 held-out S1 entities with V2-C model...", flush=True)
    v2c_model = joblib.load("experiments/v2/models/xgboost_matcher_v2_c.joblib")

    # Store scored pairs with prob >= 0.50 AND all GT pairs for error analysis
    scored_high_rows = []
    pair_prob_map = {}  # (sid, cid) -> prob for prob >= 0.80 or cid in gt_dict[sid]

    n_batches = ceil(total_eval_pairs / BATCH_SIZE)
    t_sc0 = time.time()
    for b_idx in range(n_batches):
        bs = b_idx * BATCH_SIZE
        be = min(bs + BATCH_SIZE, total_eval_pairs)
        b_s1 = all_pairs_s1[bs:be]
        b_cid = all_pairs_cid[bs:be]

        name1 = [s1_lookup[s]["name"] for s in b_s1]
        addr1 = [s1_lookup[s]["addr"] for s in b_s1]
        c1 = [s1_lookup[s]["country"] for s in b_s1]

        name2 = [cand_records[c][0] for c in b_cid]
        addr2 = [cand_records[c][1] for c in b_cid]
        c2 = [cand_records[c][2] for c in b_cid]

        X_batch = pd.DataFrame({
            "name_ratio": process.cpdist(name1, name2, scorer=fuzz.ratio, workers=-1, dtype=np.float64) / 100.0,
            "name_token_sort": process.cpdist(name1, name2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64) / 100.0,
            "name_token_set": process.cpdist(name1, name2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64) / 100.0,
            "address_ratio": process.cpdist(addr1, addr2, scorer=fuzz.ratio, workers=-1, dtype=np.float64) / 100.0,
            "address_token_sort": process.cpdist(addr1, addr2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float64) / 100.0,
            "address_token_set": process.cpdist(addr1, addr2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64) / 100.0,
            "country_same": (np.array(c1) == np.array(c2)).astype(np.int64),
            "name_len_1": np.array([len(x) for x in name1], dtype=np.int64),
            "name_len_2": np.array([len(x) for x in name2], dtype=np.int64),
            "address_len_1": np.array([len(x) for x in addr1], dtype=np.int64),
            "address_len_2": np.array([len(x) for x in addr2], dtype=np.int64),
        })

        probs = v2c_model.predict_proba(X_batch[LOCKED_FEATURE_COLUMNS])[:, 1]
        is_gt = np.array([cid in gt_dict[sid] for sid, cid in zip(b_s1, b_cid)], dtype=bool)
        keep_mask = (probs >= 0.50) | is_gt

        for idx in np.where(keep_mask)[0]:
            sid = b_s1[idx]
            cid = b_cid[idx]
            pr = float(probs[idx])
            pair_prob_map[(sid, cid)] = pr
            scored_high_rows.append({
                "s1_id": sid,
                "cand_id": cid,
                "prob": pr,
                "is_gt": bool(is_gt[idx]),
                "name_ratio": float(X_batch["name_ratio"].iat[idx]),
                "name_token_sort": float(X_batch["name_token_sort"].iat[idx]),
                "name_token_set": float(X_batch["name_token_set"].iat[idx]),
                "address_ratio": float(X_batch["address_ratio"].iat[idx]),
                "address_token_sort": float(X_batch["address_token_sort"].iat[idx]),
                "address_token_set": float(X_batch["address_token_set"].iat[idx]),
                "name_len_1": int(X_batch["name_len_1"].iat[idx]),
                "name_len_2": int(X_batch["name_len_2"].iat[idx]),
                "address_len_1": int(X_batch["address_len_1"].iat[idx]),
                "address_len_2": int(X_batch["address_len_2"].iat[idx]),
            })

        del name1, addr1, c1, name2, addr2, c2, X_batch, probs, is_gt, keep_mask
        gc.collect()
        print(f"  Scored batch {b_idx+1}/{n_batches} in {time.time()-t_sc0:.1f}s", flush=True)

    scored_df = pd.DataFrame(scored_high_rows)
    del scored_high_rows
    gc.collect()

    # Tag each row with whether it was in Capped-350 (test inference blocker) and Prioritized-500
    cap350_sets = {sid: set(cands) for sid, cands in cands_capped350_fileorder.items()}
    prio500_sets = {sid: set(cands) for sid, cands in cands_prioritized_500.items()}

    scored_df["in_cap350"] = [cid in cap350_sets[sid] for sid, cid in zip(scored_df["s1_id"], scored_df["cand_id"])]
    scored_df["in_prio500"] = [cid in prio500_sets[sid] for sid, cid in zip(scored_df["s1_id"], scored_df["cand_id"])]

    def evaluate_threshold_sweep(df_sub, label_desc):
        sweep_results = []
        high_df = df_sub[df_sub["prob"] >= 0.80]
        arr_s1 = high_df["s1_id"].values
        arr_cid = high_df["cand_id"].values
        arr_prob = high_df["prob"].values

        for th in THRESHOLDS:
            mask = arr_prob >= th
            sel_s1 = arr_s1[mask]
            sel_cid = arr_cid[mask]

            pred_dict = {sid: set() for sid in heldout_s1_ids}
            for s, c in zip(sel_s1, sel_cid):
                pred_dict[s].add(c)

            metrics = evaluate_macro_f_beta(gt_dict, pred_dict, beta=0.5)

            tp, fp = 0, 0
            s2_tp, s2_fp = 0, 0
            s3_tp, s3_fp = 0, 0
            for sid, pset in pred_dict.items():
                tset = gt_dict[sid]
                tp += len(pset & tset)
                fp += len(pset - tset)
                for m in pset:
                    if m.startswith("S2-"):
                        if m in tset:
                            s2_tp += 1
                        else:
                            s2_fp += 1
                    else:
                        if m in tset:
                            s3_tp += 1
                        else:
                            s3_fp += 1
            fn = total_gt_links - tp
            s2_fn = s2_gt_links - s2_tp
            s3_fn = s3_gt_links - s3_tp
            pred_singletons = sum(1 for pset in pred_dict.values() if len(pset) == 0)

            # Stratify by country and cardinality at this threshold
            ctry_f05 = defaultdict(list)
            ctry_prec = defaultdict(list)
            ctry_rec = defaultdict(list)
            card_f05 = defaultdict(list)
            card_prec = defaultdict(list)
            card_rec = defaultdict(list)

            for sid in heldout_s1_ids:
                tset = gt_dict[sid]
                pset = pred_dict[sid]
                p_e, r_e, f_e = compute_entity_f_beta(tset, pset, beta=0.5)
                c = s1_lookup[sid]["country"]
                ctry_f05[c].append(f_e)
                ctry_prec[c].append(p_e)
                ctry_rec[c].append(r_e)

                cb = "empty_s1" if len(tset) == 0 else ("single_match_s1" if len(tset) == 1 else "multi_match_s1")
                card_f05[cb].append(f_e)
                card_prec[cb].append(p_e)
                card_rec[cb].append(r_e)

            sweep_results.append({
                "threshold": float(th),
                "macro_f05": float(metrics["macro_f_beta"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "singleton_accuracy": float(metrics["singleton_accuracy"]),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "predicted_pairs": int(len(sel_s1)),
                "predicted_singletons_empty_count": int(pred_singletons),
                "s2_metrics": {
                    "tp": int(s2_tp),
                    "fp": int(s2_fp),
                    "fn": int(s2_fn),
                    "micro_prec": float(s2_tp / (s2_tp + s2_fp)) if (s2_tp + s2_fp) > 0 else 0.0,
                    "micro_rec": float(s2_tp / s2_gt_links) if s2_gt_links > 0 else 0.0,
                },
                "s3_metrics": {
                    "tp": int(s3_tp),
                    "fp": int(s3_fp),
                    "fn": int(s3_fn),
                    "micro_prec": float(s3_tp / (s3_tp + s3_fp)) if (s3_tp + s3_fp) > 0 else 0.0,
                    "micro_rec": float(s3_tp / s3_gt_links) if s3_gt_links > 0 else 0.0,
                },
                "by_country": {
                    c: {
                        "macro_f05": float(np.mean(ctry_f05[c])),
                        "macro_prec": float(np.mean(ctry_prec[c])),
                        "macro_rec": float(np.mean(ctry_rec[c])),
                    }
                    for c in ctry_f05
                },
                "by_cardinality": {
                    cb: {
                        "macro_f05": float(np.mean(card_f05[cb])),
                        "macro_prec": float(np.mean(card_prec[cb])),
                        "macro_rec": float(np.mean(card_rec[cb])),
                    }
                    for cb in card_f05
                },
            })
        return sweep_results

    sweep_cap350 = evaluate_threshold_sweep(scored_df[scored_df["in_cap350"]], "Capped-350 (Test Inference Blocker)")
    sweep_uncapped = evaluate_threshold_sweep(scored_df, "Uncapped / Full Found Pool")
    sweep_prio500 = evaluate_threshold_sweep(scored_df[scored_df["in_prio500"]], "Prioritized-500 Blocker")

    print("\n--- PHASE 5A: HELD-OUT 5,000 S1 THRESHOLD SWEEP (TEST-INFERENCE CAPPED-350 BLOCKER) ---", flush=True)
    print(f"{'Thresh':>7} | {'Macro F0.5':>10} | {'Precision':>9} | {'Recall':>8} | {'Pred Pairs':>10} | {'Empty/Sing':>10} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'US F0.5':>8} | {'IND F0.5':>8}", flush=True)
    print("-" * 115, flush=True)
    for r in sweep_cap350:
        print(
            f"{r['threshold']:>7.4f} | {r['macro_f05']:>10.4f} | {r['macro_precision']*100:>8.2f}% | {r['macro_recall']*100:>7.2f}% | "
            f"{r['predicted_pairs']:>10,d} | {r['predicted_singletons_empty_count']:>10,d} | "
            f"{r['tp']:>6,d} | {r['fp']:>6,d} | {r['fn']:>6,d} | "
            f"{r['by_country']['us']['macro_f05']:>8.4f} | {r['by_country']['india']['macro_f05']:>8.4f}",
            flush=True,
        )

    print("\n--- PHASE 5B: HELD-OUT 5,000 S1 THRESHOLD SWEEP (UNCAPPED / PRIORITIZED BLOCKER) ---", flush=True)
    print(f"{'Thresh':>7} | {'Macro F0.5':>10} | {'Precision':>9} | {'Recall':>8} | {'Pred Pairs':>10} | {'Empty/Sing':>10} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'US F0.5':>8} | {'IND F0.5':>8}", flush=True)
    print("-" * 115, flush=True)
    for r in sweep_uncapped:
        print(
            f"{r['threshold']:>7.4f} | {r['macro_f05']:>10.4f} | {r['macro_precision']*100:>8.2f}% | {r['macro_recall']*100:>7.2f}% | "
            f"{r['predicted_pairs']:>10,d} | {r['predicted_singletons_empty_count']:>10,d} | "
            f"{r['tp']:>6,d} | {r['fp']:>6,d} | {r['fn']:>6,d} | "
            f"{r['by_country']['us']['macro_f05']:>8.4f} | {r['by_country']['india']['macro_f05']:>8.4f}",
            flush=True,
        )

    # =========================================================================
    # PHASE 6: DEEP ERROR ANALYSIS ON 5,000 HELD-OUT POOL
    # =========================================================================
    print("\n" + "=" * 85, flush=True)
    print("PHASE 6: DEEP ERROR ANALYSIS AT OPERATING THRESHOLD 0.900", flush=True)
    print("=" * 85, flush=True)

    # Analyze False Positives (is_gt == False and prob >= 0.900)
    fp_df = scored_df[(~scored_df["is_gt"]) & (scored_df["prob"] >= 0.900)].copy()
    tp_df = scored_df[(scored_df["is_gt"]) & (scored_df["prob"] >= 0.900)].copy()
    fn_scored_df = scored_df[(scored_df["is_gt"]) & (scored_df["prob"] < 0.900)].copy()

    fp_categories = Counter()
    fp_examples = []

    # Also check exact normalized name and exact normalized address precision across all scored pairs
    exact_name_tp, exact_name_fp = 0, 0
    exact_addr_tp, exact_addr_fp = 0, 0

    for r in scored_df[scored_df["prob"] >= 0.900].itertuples(index=False):
        s1_rec = s1_lookup[r.s1_id]
        t_name, t_addr, _ = cand_records.get(r.cand_id, ("", "", ""))
        t_n_norm = " ".join(re_tok.findall(t_name.lower()))
        t_a_norm = " ".join(re_tok.findall(t_addr.lower()))

        if s1_rec["name_norm"] and s1_rec["name_norm"] == t_n_norm:
            if r.is_gt:
                exact_name_tp += 1
            else:
                exact_name_fp += 1
        if s1_rec["addr_norm"] and s1_rec["addr_norm"] == t_a_norm:
            if r.is_gt:
                exact_addr_tp += 1
            else:
                exact_addr_fp += 1

    for r in fp_df.sort_values(by="prob", ascending=False).itertuples(index=False):
        s1_rec = s1_lookup[r.s1_id]
        t_name, t_addr, _ = cand_records.get(r.cand_id, ("", "", ""))
        t_n_norm = " ".join(re_tok.findall(t_name.lower()))
        t_a_norm = " ".join(re_tok.findall(t_addr.lower()))

        d1 = extract_digits(s1_rec["addr_norm"])
        d2 = extract_digits(t_a_norm)
        p1 = extract_postal(s1_rec["addr_norm"])
        p2 = extract_postal(t_a_norm)

        cats = []
        if len(gt_dict[r.s1_id]) == 0:
            cats.append("singleton_s1_false_alarm")
        if s1_rec["name_norm"] == t_n_norm and s1_rec["name_norm"] != "":
            cats.append("exact_normalized_name_match")
        if s1_rec["addr_norm"] == t_a_norm and s1_rec["addr_norm"] != "":
            cats.append("exact_normalized_address_match")
        if r.name_token_set >= 0.90 and r.address_token_set < 0.65:
            cats.append("high_name_sim_low_addr_sim_different_branch")
        if r.address_token_set >= 0.90 and r.name_token_set < 0.65:
            cats.append("high_addr_sim_low_name_sim_same_building_diff_business")
        if p1 and p2 and len(p1 & p2) == 0:
            cats.append("postal_pin_code_mismatch")
        if d1 and d2 and len(d1 & d2) == 0:
            cats.append("all_address_digits_mismatch_house_or_street_no")
        if r.name_len_1 <= 10 or r.name_len_2 <= 10:
            cats.append("short_business_name_le_10_chars")
        if not cats:
            cats.append("partial_name_and_address_overlap_ambiguous")

        for c in cats:
            fp_categories[c] += 1

        if len(fp_examples) < 15:
            fp_examples.append({
                "s1_id": r.s1_id,
                "cand_id": r.cand_id,
                "prob": round(r.prob, 4),
                "categories": cats,
                "s1_name": s1_rec["name"],
                "cand_name": t_name,
                "s1_addr": s1_rec["addr"],
                "cand_addr": t_addr,
                "name_ratio": round(r.name_ratio, 3),
                "name_token_set": round(r.name_token_set, 3),
                "addr_ratio": round(r.address_ratio, 3),
                "addr_token_set": round(r.address_token_set, 3),
            })

    # Analyze False Negatives (scored < 0.900)
    fn_categories = Counter()
    fn_examples = []
    for r in fn_scored_df.sort_values(by="prob", ascending=False).itertuples(index=False):
        s1_rec = s1_lookup[r.s1_id]
        t_name, t_addr, _ = cand_records.get(r.cand_id, ("", "", ""))
        cats = []
        if r.prob >= 0.80:
            cats.append("near_miss_prob_0.80_to_0.90")
        elif r.prob >= 0.50:
            cats.append("moderate_score_0.50_to_0.80")
        else:
            cats.append("low_score_below_0.50")

        if r.name_token_set < 0.70 and r.address_token_set >= 0.80:
            cats.append("name_variation_or_acronym_with_good_address")
        if r.address_token_set < 0.65 and r.name_token_set >= 0.85:
            cats.append("address_formatting_or_locality_variation_with_good_name")
        if r.name_len_1 <= 10 or r.name_len_2 <= 10:
            cats.append("short_name_penalty")

        for c in cats:
            fn_categories[c] += 1

        if len(fn_examples) < 15:
            fn_examples.append({
                "s1_id": r.s1_id,
                "cand_id": r.cand_id,
                "prob": round(r.prob, 4),
                "categories": cats,
                "s1_name": s1_rec["name"],
                "cand_name": t_name,
                "s1_addr": s1_rec["addr"],
                "cand_addr": t_addr,
                "name_ratio": round(r.name_ratio, 3),
                "name_token_set": round(r.name_token_set, 3),
                "addr_ratio": round(r.address_ratio, 3),
                "addr_token_set": round(r.address_token_set, 3),
            })

    phase6_results = {
        "total_fp_at_0900_uncapped": int(len(fp_df)),
        "total_tp_at_0900_uncapped": int(len(tp_df)),
        "total_fn_scored_below_0900": int(len(fn_scored_df)),
        "exact_normalized_name_at_0900": {"tp": exact_name_tp, "fp": exact_name_fp, "precision": float(exact_name_tp / max(1, exact_name_tp + exact_name_fp))},
        "exact_normalized_addr_at_0900": {"tp": exact_addr_tp, "fp": exact_addr_fp, "precision": float(exact_addr_tp / max(1, exact_addr_tp + exact_addr_fp))},
        "fp_failure_mode_counts": dict(fp_categories.most_common()),
        "fn_failure_mode_counts": dict(fn_categories.most_common()),
        "top_fp_examples": fp_examples,
        "top_fn_examples": fn_examples,
    }
    print("FP Failure Modes:", json.dumps(dict(fp_categories.most_common()), indent=2), flush=True)
    print("FN Failure Modes:", json.dumps(dict(fn_categories.most_common()), indent=2), flush=True)

    # Save all diagnostic results to JSON
    all_diag = {
        "phase2_training_population": phase2_stats,
        "phase3_heldout_definition": phase3_stats,
        "phase4_candidate_recall": phase4_results,
        "phase5_threshold_sweeps": {
            "capped350_test_inference_regime": sweep_cap350,
            "uncapped_regime": sweep_uncapped,
            "prioritized500_regime": sweep_prio500,
        },
        "phase6_error_analysis": phase6_results,
        "total_runtime_s": time.time() - t_global_start,
    }
    out_json = DIAG_DIR / "v3_diagnostics_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_diag, f, indent=2)
    print(f"\nSaved complete diagnostics summary to {out_json}", flush=True)


if __name__ == "__main__":
    main()
