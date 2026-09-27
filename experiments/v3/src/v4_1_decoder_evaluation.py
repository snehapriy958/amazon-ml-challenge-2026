"""V4.1 Conflict Decoder Evaluation & Full-Catalog Head-to-Head Diagnostic.

Evaluates V4.1 decoder variants using EXISTING V3 predictions/features only:
  - V4.1-A: Existing exclusivity resolver (_score_owner_evidence from run_v3_exclusivity_experiment.py)
  - V4.1-B: Model probability + enhanced owner evidence (numeric-token Jaccard/conflict,
            phonetic overlap, state agreement, building/street/unit agreement, postal agreement,
            country agreement, name+address combined evidence)
  - V4.1-C(m): Model probability + owner evidence + margin-aware tie-breaking (m in [0, 0.001, 0.005, 0.01, 0.02, 0.05])
  - V4.1-D(m): Conservative margin abstention — if owner_margin < m, drop ambiguous target assignment
               from all competing S1s rather than forcing an owner (m in [0, 0.001, 0.005, 0.01, 0.02, 0.05])

Evaluates all variants on:
  1. FINAL_HOLDOUT_1000
  2. INTERNAL_CHECK_1000
  3. DEV_3000
  4. COMBINED_5000
  5. Full-Catalog FP Head-to-Head Diagnostic across all 435 V3 FPs that have a true S1 owner in train_source1.tsv.
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path.cwd()))

from experiments.v3.src.benchmark_test_prefilter import select_prefilter_mask_for_pool
from experiments.v3.src.normalization_v3 import normalize_name_v3, parse_address_v3
from experiments.v3.src.run_v3_d_bakeoff import (
    apply_frozen_policy,
    prepare_eval_structures,
)
from experiments.v3.src.run_v3_exclusivity_experiment import (
    _extract_record_features as _extract_v41a_features,
    _score_owner_evidence as _score_v41a_evidence,
)

MARGIN_SWEEP = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]


def extract_v41_features(raw_name: str, raw_addr: str, country: str = "") -> dict:
    """Extract comprehensive raw + V3 normalized record features for V4.1 conflict decoding."""
    rn = (raw_name or "").strip().lower()
    ra = (raw_addr or "").strip().lower()
    rc = (country or "").strip().lower()
    n_info = normalize_name_v3(raw_name or "")
    a_info = parse_address_v3(raw_addr or "", rc)
    return {
        "raw_name": rn,
        "raw_addr": ra,
        "country": rc,
        "name_norm": n_info["lower_norm"],
        "name_core": n_info["core_name_norm"] or n_info["lower_norm"],
        "name_compact": n_info["compact_norm"],
        "core_tokens": set(n_info["core_tokens"]),
        "phon_tokens": set(n_info["phonetic_tokens"]),
        "suffix_class": n_info["suffix_class"],
        "addr_norm": a_info["lower_norm"],
        "addr_orig": a_info["orig_norm"] or a_info["lower_norm"],
        "addr_tokens": set(a_info["tokens"]),
        "postal": a_info["postal_code"] or "",
        "building": a_info["building_num"] or "",
        "unit": a_info["unit_num"] or "",
        "state": a_info["state_code"] or "",
        "num_tokens": set(a_info["numeric_tokens"]),
    }


def _logit_norm(prob: float) -> float:
    """Map calibrated probability in [0.995, 1.0) onto a smooth [0, 1] log-odds scale."""
    p = min(max(float(prob), 1e-6), 1.0 - 1e-7)
    lo = math.log(p / (1.0 - p))
    lo_min = math.log(0.995 / 0.005)  # ~5.2933
    lo_max = math.log(0.99999 / 0.00001)  # ~11.5129
    return min(1.0, max(0.0, (lo - lo_min) / (lo_max - lo_min)))


def compute_assignment_components(
    s1_rec: dict,
    tgt_rec: dict,
    model_prob: float,
    model_rank: int,
    s1_id: str,
) -> dict:
    """Compute all required V4.1 components (A, B, C) for a candidate assignment (S1, target)."""
    # 1. Name similarity & exact anchors
    raw_name_exact = 1.0 if (s1_rec["raw_name"] and s1_rec["raw_name"] == tgt_rec["raw_name"]) else 0.0
    core_name_exact = 1.0 if (s1_rec["name_core"] and s1_rec["name_core"] == tgt_rec["name_core"]) else 0.0
    norm_name_exact = 1.0 if (s1_rec["name_norm"] and s1_rec["name_norm"] == tgt_rec["name_norm"]) else 0.0
    compact_name_exact = 1.0 if (s1_rec["name_compact"] and s1_rec["name_compact"] == tgt_rec["name_compact"]) else 0.0

    nc1, nc2 = s1_rec["name_core"], tgt_rec["name_core"]
    rn1, rn2 = s1_rec["raw_name"], tgt_rec["raw_name"]
    n_tset = (fuzz.token_set_ratio(nc1, nc2) / 100.0) if (nc1 and nc2) else 0.0
    n_tsort = (fuzz.token_sort_ratio(nc1, nc2) / 100.0) if (nc1 and nc2) else 0.0
    n_raw = (fuzz.ratio(rn1, rn2) / 100.0) if (rn1 and rn2) else 0.0

    ct1, ct2 = s1_rec["core_tokens"], tgt_rec["core_tokens"]
    core_tok_jacc = (len(ct1 & ct2) / len(ct1 | ct2)) if (ct1 and ct2) else 0.0

    pt1, pt2 = s1_rec["phon_tokens"], tgt_rec["phon_tokens"]
    phon_jacc = (len(pt1 & pt2) / len(pt1 | pt2)) if (pt1 and pt2) else 0.0

    name_sim = (
        0.35 * n_tset
        + 0.25 * n_tsort
        + 0.15 * n_raw
        + 0.15 * core_tok_jacc
        + 0.10 * phon_jacc
    )

    # 2. Address similarity & exact anchors
    raw_addr_exact = 1.0 if (s1_rec["raw_addr"] and s1_rec["raw_addr"] == tgt_rec["raw_addr"]) else 0.0
    core_addr_exact = 1.0 if (s1_rec["addr_norm"] and s1_rec["addr_norm"] == tgt_rec["addr_norm"]) else 0.0

    ac1, ac2 = s1_rec["addr_norm"], tgt_rec["addr_norm"]
    ra1, ra2 = s1_rec["raw_addr"], tgt_rec["raw_addr"]
    a_tset = (fuzz.token_set_ratio(ac1, ac2) / 100.0) if (ac1 and ac2) else 0.0
    a_tsort = (fuzz.token_sort_ratio(ac1, ac2) / 100.0) if (ac1 and ac2) else 0.0
    a_raw = (fuzz.ratio(ra1, ra2) / 100.0) if (ra1 and ra2) else 0.0

    at1, at2 = s1_rec["addr_tokens"], tgt_rec["addr_tokens"]
    addr_tok_jacc = (len(at1 & at2) / len(at1 | at2)) if (at1 and at2) else 0.0
    addr_tok_contain = (len(at1 & at2) / min(len(at1), len(at2))) if (at1 and at2) else 0.0

    addr_sim = (
        0.35 * a_tset
        + 0.20 * a_tsort
        + 0.15 * a_raw
        + 0.15 * addr_tok_jacc
        + 0.15 * addr_tok_contain
    )

    # 3. Building / street / unit number agreement
    b1, b2 = s1_rec["building"], tgt_rec["building"]
    if b1 and b2:
        if b1 == b2:
            bldg_agree = 1.0
        elif b1 in b2 or b2 in b1:
            bldg_agree = 0.6
        else:
            bldg_agree = -1.0
    else:
        bldg_agree = 0.0

    u1, u2 = s1_rec["unit"], tgt_rec["unit"]
    if u1 and u2:
        unit_agree = 0.5 if (u1 == u2) else -0.5
    else:
        unit_agree = 0.0

    # 4. Postal code & state agreement
    p1, p2 = s1_rec["postal"], tgt_rec["postal"]
    if p1 and p2:
        if p1 == p2:
            postal_agree = 1.0
        elif len(p1) >= 3 and len(p2) >= 3 and p1[:3] == p2[:3]:
            postal_agree = 0.3
        else:
            postal_agree = -1.0
    else:
        postal_agree = 0.0

    st1, st2 = s1_rec["state"], tgt_rec["state"]
    if st1 and st2:
        state_agree = 0.3 if (st1 == st2) else -0.5
    else:
        state_agree = 0.0

    # 5. Country agreement
    c1, c2 = s1_rec["country"], tgt_rec["country"]
    if c1 and c2:
        country_agree = 1.0 if (c1 == c2) else -1.0
    else:
        country_agree = 0.0

    # 6. Numeric-token agreement (critical for multi-branch disambiguation)
    nt1, nt2 = s1_rec["num_tokens"], tgt_rec["num_tokens"]
    if nt1 and nt2:
        inter = len(nt1 & nt2)
        union = len(nt1 | nt2)
        if inter > 0:
            num_tok_agree = float(inter / union)
        else:
            num_tok_agree = -0.85  # Both addresses have numbers, zero numbers shared
    else:
        num_tok_agree = 0.0

    # 7. Name + Address combined evidence (owner_score)
    geom_na = math.sqrt(max(0.0, name_sim * addr_sim))
    owner_score = (
        0.40 * name_sim
        + 0.40 * addr_sim
        + 0.20 * geom_na
        + 0.14 * postal_agree
        + 0.14 * bldg_agree
        + 0.10 * num_tok_agree
        + 0.04 * unit_agree
        + 0.03 * state_agree
        + 0.04 * country_agree
        + 0.08 * core_addr_exact
        + 0.05 * raw_addr_exact
        + 0.05 * core_name_exact
        + 0.03 * compact_name_exact
        + 0.03 * raw_name_exact
    )

    # Model probability contribution (calibrated probability + logit-scaled margin + rank)
    prob_norm = _logit_norm(model_prob)
    rank_bonus = -0.006 * model_rank

    # V4.1-B composite score: owner_score + model_probability evidence
    score_v41b = owner_score + 0.18 * prob_norm + rank_bonus

    return {
        "s1_id": s1_id,
        "owner_score": float(owner_score),
        "model_probability": float(model_prob),
        "prob_norm": float(prob_norm),
        "model_rank": int(model_rank),
        "score_v41b": float(score_v41b),
        "name_sim": float(name_sim),
        "addr_sim": float(addr_sim),
        "bldg_agree": float(bldg_agree),
        "postal_agree": float(postal_agree),
        "country_agree": float(country_agree),
        "num_tok_agree": float(num_tok_agree),
        "raw_addr_exact": float(raw_addr_exact),
        "core_addr_exact": float(core_addr_exact),
    }


def resolve_conflict_group(
    variant: str,
    margin_param: float,
    candidates_info: List[Tuple[dict, tuple]],
) -> Set[str]:
    """Given competing assignments for a single target `tid`, return the set of winning `s1_id`s (size 0 or 1).

    Each item in `candidates_info` is `(comp_dict, v41a_tuple)`.
    """
    if len(candidates_info) == 1:
        return {candidates_info[0][0]["s1_id"]}

    if variant == "V4.1-A":
        # Exact existing exclusivity resolver
        best_item = max(candidates_info, key=lambda x: x[1])
        return {best_item[0]["s1_id"]}

    elif variant == "V4.1-B":
        # Model probability + enhanced owner evidence (always keeps top-1 owner)
        best_item = max(
            candidates_info,
            key=lambda x: (
                round(x[0]["score_v41b"], 8),
                x[0]["raw_addr_exact"],
                x[0]["core_addr_exact"],
                x[0]["postal_agree"],
                x[0]["bldg_agree"],
                x[0]["num_tok_agree"],
                round(x[0]["model_probability"], 8),
                -x[0]["model_rank"],
                x[0]["s1_id"],
            ),
        )
        return {best_item[0]["s1_id"]}

    elif variant == "V4.1-C":
        # Model probability + owner evidence + margin:
        # Rank primarily by owner_score; if two candidates are within `margin_param` on owner_score,
        # break the near-tie decisively using model_probability (prob_norm) and model_rank.
        sorted_by_owner = sorted(
            candidates_info,
            key=lambda x: x[0]["owner_score"],
            reverse=True,
        )
        best_owner_sc = sorted_by_owner[0][0]["owner_score"]
        near_tied = [
            x for x in sorted_by_owner
            if (best_owner_sc - x[0]["owner_score"]) <= margin_param
        ]
        if len(near_tied) == 1:
            return {near_tied[0][0]["s1_id"]}
        # Among near-tied owner_score candidates, select by highest model_probability + score_v41b
        winner = max(
            near_tied,
            key=lambda x: (
                round(x[0]["model_probability"], 8),
                -x[0]["model_rank"],
                round(x[0]["score_v41b"], 8),
                x[0]["s1_id"],
            ),
        )
        return {winner[0]["s1_id"]}

    elif variant == "V4.1-D":
        # Conservative margin abstention:
        # Sort by V4.1-B composite score (score_v41b). Compute owner_margin = best_score - second_best_score.
        # If owner_margin < margin_param, remove the ambiguous target assignment from ALL competing S1s.
        ranked = sorted(
            candidates_info,
            key=lambda x: (
                round(x[0]["score_v41b"], 8),
                round(x[0]["model_probability"], 8),
                -x[0]["model_rank"],
                x[0]["s1_id"],
            ),
            reverse=True,
        )
        best_score = ranked[0][0]["score_v41b"]
        second_best_score = ranked[1][0]["score_v41b"]
        owner_margin = best_score - second_best_score
        if owner_margin < margin_param:
            return set()  # Ambiguous conflict -> remove from all competing S1s
        return {ranked[0][0]["s1_id"]}

    else:
        raise ValueError(f"Unknown variant: {variant}")


def evaluate_all_variants_on_pool(
    pool_name: str,
    qdf: pd.DataFrame,
    feat_df: pd.DataFrame,
    probs: np.ndarray,
    policy_cfg: dict,
    target_raw_lookup: Dict[str, Dict[str, str]],
    top_k: int = 500,
) -> dict:
    eval_full = prepare_eval_structures(qdf, feat_df)
    if top_k < 750:
        keep_mask = select_prefilter_mask_for_pool(feat_df, eval_full["s1_row_slices"], top_k)
        sub_df = feat_df[keep_mask].reset_index(drop=True)
        sub_probs = probs[keep_mask]
    else:
        sub_df = feat_df
        sub_probs = probs

    eval_struct = prepare_eval_structures(qdf, sub_df)
    policy_mask = apply_frozen_policy(sub_probs, sub_df, eval_struct, policy_cfg)

    cand_col = sub_df["candidate_entity_id"].tolist()
    s1_ids = qdf["source1_entity_id"].tolist()
    gt_sets: List[Set[str]] = eval_struct["gt_sets"]
    slices: List[Tuple[int, int]] = eval_struct["s1_row_slices"]

    s1_preds_base: List[Tuple[str, List[Tuple[str, float]], Set[str]]] = []
    target_claims = defaultdict(list)  # tid -> list of (s1_id, rank_idx, prob)

    for s1_idx, (st, en) in enumerate(slices):
        s1_id = s1_ids[s1_idx]
        gt_set = gt_sets[s1_idx]
        if en <= st:
            s1_preds_base.append((s1_id, [], gt_set))
            continue
        loc_mask = policy_mask[st:en]
        if not np.any(loc_mask):
            s1_preds_base.append((s1_id, [], gt_set))
            continue
        loc_indices = np.flatnonzero(loc_mask)
        loc_probs = sub_probs[st:en][loc_indices]
        order = np.argsort(-loc_probs, kind="mergesort")
        ordered_items = [
            (cand_col[st + int(loc_indices[i])], float(loc_probs[i]))
            for i in order
        ]
        s1_preds_base.append((s1_id, ordered_items, gt_set))
        for rank_idx, (tid, p_val) in enumerate(ordered_items):
            target_claims[tid].append((s1_id, rank_idx, p_val))

    conflicted_targets = {tid: claims for tid, claims in target_claims.items() if len(claims) > 1}
    conflicted_s1_ids = {s1_id for claims in conflicted_targets.values() for s1_id, _, _ in claims}

    # Precompute record features for conflicted S1 and targets
    s1_v41a = {}
    s1_v41 = {}
    for row in qdf.itertuples(index=False):
        sid = row.source1_entity_id
        if sid in conflicted_s1_ids:
            bn = "" if pd.isna(row.business_name) else str(row.business_name)
            ba = "" if pd.isna(row.business_address) else str(row.business_address)
            co = "" if pd.isna(row.country) else str(row.country)
            s1_v41a[sid] = _extract_v41a_features(bn, ba, co)
            s1_v41[sid] = extract_v41_features(bn, ba, co)

    tgt_v41a = {}
    tgt_v41 = {}
    for tid in conflicted_targets:
        t_raw = target_raw_lookup.get(tid, {})
        bn = t_raw.get("business_name", "")
        ba = t_raw.get("business_address", "")
        co = t_raw.get("country", "")
        tgt_v41a[tid] = _extract_v41a_features(bn, ba, co)
        tgt_v41[tid] = extract_v41_features(bn, ba, co)

    # Precompute assignment components for each conflicted target
    conflict_candidates_info = {}
    conflict_diagnostics = []
    for tid, claims in conflicted_targets.items():
        c_list = []
        for s1_id, rank_idx, p_val in claims:
            comp = compute_assignment_components(
                s1_v41[s1_id], tgt_v41[tid], p_val, rank_idx, s1_id
            )
            v41a_tup = _score_v41a_evidence(s1_v41a[s1_id], tgt_v41a[tid], rank_idx, s1_id)
            c_list.append((comp, v41a_tup))
        conflict_candidates_info[tid] = c_list

        # Log best_score, second_best_score, owner_margin
        scores_sorted = sorted([x[0]["score_v41b"] for x in c_list], reverse=True)
        conflict_diagnostics.append({
            "tid": tid,
            "num_competing_s1": len(c_list),
            "best_score": round(scores_sorted[0], 6),
            "second_best_score": round(scores_sorted[1], 6),
            "owner_margin": round(scores_sorted[0] - scores_sorted[1], 6),
        })

    def _eval_decoder(variant_label: str, winning_owners: Dict[str, Set[str]] | None):
        f05_list, prec_list, rec_list = [], [], []
        tp_total = fp_total = fn_total = pred_links_total = empty_s1_count = 0
        removed_gt_pos = removed_gt_neg = 0

        for s1_id, items, gt_set in s1_preds_base:
            if winning_owners is None:
                kept = [tid for tid, _ in items]
            else:
                kept = []
                for tid, _ in items:
                    if tid not in winning_owners or s1_id in winning_owners[tid]:
                        kept.append(tid)
                    else:
                        if tid in gt_set:
                            removed_gt_pos += 1
                        else:
                            removed_gt_neg += 1

            pred_set = set(kept)
            pred_links_total += len(pred_set)
            if not pred_set:
                empty_s1_count += 1

            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            tp_total += tp
            fp_total += fp
            fn_total += fn

            if len(gt_set) == 0:
                p_e, r_e, f_e = (1.0, 1.0, 1.0) if len(pred_set) == 0 else (0.0, 0.0, 0.0)
            else:
                if len(pred_set) == 0:
                    p_e, r_e, f_e = (0.0, 0.0, 0.0)
                else:
                    p_e = tp / len(pred_set)
                    r_e = tp / len(gt_set)
                    f_e = (1.25 * p_e * r_e) / (0.25 * p_e + r_e) if (p_e + r_e) > 0 else 0.0
            prec_list.append(p_e)
            rec_list.append(r_e)
            f05_list.append(f_e)

        return {
            "variant": variant_label,
            "F0.5": round(float(np.mean(f05_list)), 6),
            "precision": round(float(np.mean(prec_list)), 6),
            "recall": round(float(np.mean(rec_list)), 6),
            "TP": int(tp_total),
            "FP": int(fp_total),
            "FN": int(fn_total),
            "predicted_links": int(pred_links_total),
            "empty_s1": int(empty_s1_count),
            "target_conflicts": int(len(conflicted_targets)) if winning_owners is None else 0,
            "gt_positives_removed": int(removed_gt_pos),
            "gt_negatives_removed": int(removed_gt_neg),
        }

    variants_out = {}
    variants_out["Baseline_V3"] = _eval_decoder("Baseline_V3", None)

    # V4.1-A
    wo_a = {tid: resolve_conflict_group("V4.1-A", 0.0, c_info) for tid, c_info in conflict_candidates_info.items()}
    variants_out["V4.1-A"] = _eval_decoder("V4.1-A", wo_a)

    # V4.1-B
    wo_b = {tid: resolve_conflict_group("V4.1-B", 0.0, c_info) for tid, c_info in conflict_candidates_info.items()}
    variants_out["V4.1-B"] = _eval_decoder("V4.1-B", wo_b)

    # V4.1-C across margin sweep
    for m in MARGIN_SWEEP:
        lbl = f"V4.1-C(m={m})"
        wo_c = {tid: resolve_conflict_group("V4.1-C", m, c_info) for tid, c_info in conflict_candidates_info.items()}
        variants_out[lbl] = _eval_decoder(lbl, wo_c)

    # V4.1-D across margin sweep
    for m in MARGIN_SWEEP:
        lbl = f"V4.1-D(m={m})"
        wo_d = {tid: resolve_conflict_group("V4.1-D", m, c_info) for tid, c_info in conflict_candidates_info.items()}
        variants_out[lbl] = _eval_decoder(lbl, wo_d)

    return {
        "pool_name": pool_name,
        "num_s1": len(s1_preds_base),
        "target_conflicts_before": len(conflicted_targets),
        "conflict_diagnostics": conflict_diagnostics,
        "variants": variants_out,
    }


def run_full_catalog_fp_diagnostic(
    pool_preds_by_name: Dict[str, Tuple[pd.DataFrame, List[Tuple[str, List[Tuple[str, float]], Set[str]]]]],
    gt_owner: Dict[str, str],
    s1_raw_catalog: Dict[str, Tuple[str, str, str]],
    target_raw_lookup: Dict[str, Dict[str, str]],
) -> dict:
    """Evaluate every V4.1 decoder variant on the 435 full-catalog FP conflicts across FINAL/INTERNAL/DEV."""
    diag_results = {}

    variant_specs = [
        ("V4.1-A", "V4.1-A", 0.0),
        ("V4.1-B", "V4.1-B", 0.0),
    ]
    for m in MARGIN_SWEEP:
        variant_specs.append((f"V4.1-C(m={m})", "V4.1-C", m))
    for m in MARGIN_SWEEP:
        variant_specs.append((f"V4.1-D(m={m})", "V4.1-D", m))

    for p_name, (qdf, preds) in pool_preds_by_name.items():
        fp_conflicts = []
        for s1_id, items, gt_set in preds:
            for r_idx, (tid, prob_fp) in enumerate(items):
                if tid not in gt_set:
                    true_s1 = gt_owner.get(tid)
                    if true_s1 and true_s1 in s1_raw_catalog and s1_id in s1_raw_catalog:
                        t_raw = target_raw_lookup.get(tid, {})
                        bn_t, ba_t, co_t = t_raw.get("business_name", ""), t_raw.get("business_address", ""), t_raw.get("country", "")
                        t_v41a = _extract_v41a_features(bn_t, ba_t, co_t)
                        t_v41 = extract_v41_features(bn_t, ba_t, co_t)

                        fp_v41a = _extract_v41a_features(*s1_raw_catalog[s1_id])
                        fp_v41 = extract_v41_features(*s1_raw_catalog[s1_id])

                        tp_v41a = _extract_v41a_features(*s1_raw_catalog[true_s1])
                        tp_v41 = extract_v41_features(*s1_raw_catalog[true_s1])

                        # Note: when true_s1 is outside the validation pool, its exact ranker probability is unknown
                        # unless scored on its full 500-candidate group; we test both equal-prob prior (prob_tp = prob_fp)
                        # so owner evidence + rank decides, and also log the margin distribution.
                        comp_fp = compute_assignment_components(fp_v41, t_v41, prob_fp, r_idx, s1_id)
                        comp_tp = compute_assignment_components(tp_v41, t_v41, prob_fp, 0, true_s1)

                        tup_fp = _score_v41a_evidence(fp_v41a, t_v41a, r_idx, s1_id)
                        tup_tp = _score_v41a_evidence(tp_v41a, t_v41a, 0, true_s1)

                        fp_conflicts.append((s1_id, true_s1, [(comp_fp, tup_fp), (comp_tp, tup_tp)]))

        pool_var_summary = {}
        for lbl, v_code, m_val in variant_specs:
            fp_removed = 0
            true_owner_kept = 0
            abstained_both = 0
            fp_wrongly_kept = 0
            for fp_s1, tp_s1, c_info in fp_conflicts:
                winners = resolve_conflict_group(v_code, m_val, c_info)
                if fp_s1 not in winners:
                    fp_removed += 1
                    if tp_s1 in winners:
                        true_owner_kept += 1
                    else:
                        abstained_both += 1
                else:
                    fp_wrongly_kept += 1
            n_conf = max(1, len(fp_conflicts))
            pool_var_summary[lbl] = {
                "fp_conflicts_total": len(fp_conflicts),
                "fp_eliminated": fp_removed,
                "fp_eliminated_pct": round(100.0 * fp_removed / n_conf, 2),
                "true_owner_selected": true_owner_kept,
                "true_owner_selected_pct": round(100.0 * true_owner_kept / n_conf, 2),
                "abstained_both": abstained_both,
                "fp_wrongly_kept": fp_wrongly_kept,
            }
        diag_results[p_name] = pool_var_summary

    return diag_results


def main():
    t0 = time.time()
    res_dir = Path("experiments/v3/results")
    data_dir = Path("experiments/v3/data")
    models_dir = Path("experiments/v3/models")
    ret_dir = Path("experiments/v3/retrieval")

    with open(res_dir / "v3_d_bakeoff_metrics.json", "r", encoding="utf-8") as f:
        bakeoff_data = json.load(f)

    frozen_cfg = bakeoff_data["selected_frozen_pipeline"]
    active_cols = frozen_cfg["active_features"]
    policy_cfg = frozen_cfg["policy_config"]

    print("Loading frozen C_LightGBM_LambdaRank model & Platt calibrator...")
    c_bundle = joblib.load(models_dir / "v3_model_c_lambdarank.joblib")
    ranker, calibrator = c_bundle["ranker"], c_bundle["calibrator"]

    print("Loading cached target records from retrieval bundles...")
    b_2000 = joblib.load(ret_dir / "check_and_holdout_2000_retrieval_bundle.joblib")
    b_3000 = joblib.load(ret_dir / "dev_3000_retrieval_bundle.joblib")
    target_raw_lookup = {}
    target_raw_lookup.update(b_3000.get("target_records_all") or b_3000.get("target_records") or {})
    target_raw_lookup.update(b_2000.get("target_records_all") or b_2000.get("target_records") or {})

    pools = [
        ("FINAL_HOLDOUT_1000", "experiments/v3/pools/final_holdout_1000_s1.tsv", data_dir / "final_holdout_1000_v3_features.parquet"),
        ("INTERNAL_CHECK_1000", "experiments/v3/pools/internal_check_1000_s1.tsv", data_dir / "internal_check_1000_v3_features.parquet"),
        ("DEV_3000", "experiments/v3/pools/dev_3000_s1.tsv", data_dir / "dev_3000_v3_features.parquet"),
    ]

    pool_evals = {}
    qdfs_all, fdfs_all, probs_all = [], [], []
    pool_preds_for_diag = {}
    needed_s1_ids = set()

    # Load GT target -> true S1 owner map
    gt_owner = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            s1_id, _, rest = line.rstrip("\r\n").partition("\t")
            if rest:
                for tid in rest.split(","):
                    tid = tid.strip()
                    if tid:
                        gt_owner[tid] = s1_id

    for p_name, q_path, f_path in pools:
        print(f"Evaluating V4.1 decoder variants on {p_name}...")
        qdf = pd.read_csv(q_path, sep="\t", dtype=str)
        fdf = pd.read_parquet(f_path)
        probs = calibrator.predict_proba(ranker.predict(fdf[active_cols]).reshape(-1, 1))[:, 1]

        pool_evals[p_name] = evaluate_all_variants_on_pool(
            p_name, qdf, fdf, probs, policy_cfg, target_raw_lookup, top_k=500
        )
        qdfs_all.append(qdf)
        fdfs_all.append(fdf)
        probs_all.append(probs)

        # Also collect predictions for full-catalog FP diagnostic
        ef = prepare_eval_structures(qdf, fdf)
        km = select_prefilter_mask_for_pool(fdf, ef["s1_row_slices"], 500)
        sub_df = fdf[km].reset_index(drop=True)
        sub_probs = probs[km]
        es = prepare_eval_structures(qdf, sub_df)
        pm = apply_frozen_policy(sub_probs, sub_df, es, policy_cfg)
        cand_col = sub_df["candidate_entity_id"].tolist()
        s1_ids = qdf["source1_entity_id"].tolist()
        preds = []
        for i, (st, en) in enumerate(es["s1_row_slices"]):
            loc = np.flatnonzero(pm[st:en])
            order = np.argsort(-sub_probs[st:en][loc], kind="mergesort")
            items = [(cand_col[st + int(loc[j])], float(sub_probs[st + int(loc[j])])) for j in order]
            preds.append((s1_ids[i], items, es["gt_sets"][i]))
            needed_s1_ids.add(s1_ids[i])
            for tid, _ in items:
                if tid in gt_owner:
                    needed_s1_ids.add(gt_owner[tid])
        pool_preds_for_diag[p_name] = (qdf, preds)

    # COMBINED_5000 evaluation
    comb_qdf = pd.concat(qdfs_all, ignore_index=True)
    comb_fdf = pd.concat(fdfs_all, ignore_index=True)
    comb_probs = np.concatenate(probs_all)
    pool_evals["COMBINED_5000"] = evaluate_all_variants_on_pool(
        "COMBINED_5000", comb_qdf, comb_fdf, comb_probs, policy_cfg, target_raw_lookup, top_k=500
    )
    comb_preds = []
    for p_name in ("FINAL_HOLDOUT_1000", "INTERNAL_CHECK_1000", "DEV_3000"):
        comb_preds.extend(pool_preds_for_diag[p_name][1])
    pool_preds_for_diag["COMBINED_5000"] = (comb_qdf, comb_preds)

    # Load raw S1 records for full-catalog FP head-to-head diagnostic
    s1_raw_catalog = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in needed_s1_ids:
                s1_raw_catalog[parts[0]] = (
                    parts[1] if len(parts) > 1 else "",
                    parts[2] if len(parts) > 2 else "",
                    parts[3] if len(parts) > 3 else "",
                )

    print("Running full-catalog FP head-to-head diagnostic across all decoder variants...")
    fp_diag = run_full_catalog_fp_diagnostic(
        pool_preds_for_diag, gt_owner, s1_raw_catalog, target_raw_lookup
    )

    out_report = {
        "within_pool_evaluations": pool_evals,
        "full_catalog_fp_head_to_head_diagnostic": fp_diag,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_path = res_dir / "v4_1_decoder_evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_report, f, indent=2)

    print(f"\nSaved complete V4.1 decoder evaluation to {out_path} in {time.time()-t0:.1f}s.")
    print("\n=== COMBINED_5000 WITHIN-POOL SUMMARY ===")
    for k, v in pool_evals["COMBINED_5000"]["variants"].items():
        print(
            f"  {k:18s}: F0.5={v['F0.5']:.6f} | Prec={v['precision']*100:.4f}% | Rec={v['recall']*100:.4f}% | "
            f"TP={v['TP']} FP={v['FP']} FN={v['FN']} | Links={v['predicted_links']} EmptyS1={v['empty_s1']} | "
            f"GT_Pos_Removed={v['gt_positives_removed']} GT_Neg_Removed={v['gt_negatives_removed']}"
        )

    print("\n=== FULL-CATALOG FP HEAD-TO-HEAD DIAGNOSTIC (COMBINED_5000: 435 FPs with True S1 Owner) ===")
    for k, v in fp_diag["COMBINED_5000"].items():
        print(
            f"  {k:18s}: FP_Eliminated={v['fp_eliminated']}/{v['fp_conflicts_total']} ({v['fp_eliminated_pct']:.2f}%) | "
            f"True_Owner_Selected={v['true_owner_selected']}/{v['fp_conflicts_total']} ({v['true_owner_selected_pct']:.2f}%) | "
            f"Abstained_Both={v['abstained_both']} | FP_Wrongly_Kept={v['fp_wrongly_kept']}"
        )


if __name__ == "__main__":
    main()
