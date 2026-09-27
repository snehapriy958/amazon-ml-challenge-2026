"""V4 Entity Policy & V4.1-C(0.05) Target-Exclusivity Conflict Decoder Module.

Implements Component 6 of V4:
  - Baseline decoder: V4.1-C(m=0.05) target-exclusivity resolver
  - Enforces the verified ground-truth structural invariant:
    every S2/S3 target entity belongs to AT MOST ONE S1 entity (max target multiplicity = 1).
  - Ranks competing S1 claims for any multi-claimed target `tid` by deterministic owner evidence
    (`owner_score` combining normalized/abbreviated name, acronym, address, building, unit, postal,
    state, country, and numeric-token agreement/conflict); if competing claims are within margin `m=0.05`
    on `owner_score`, breaks near-ties decisively using calibrated model probability and rank.
"""

from collections import defaultdict
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from code.business_entity_resolution.src.run_v3_d_bakeoff import (
    apply_frozen_policy,
    evaluate_predictions_from_mask,
)
from code.business_entity_resolution.src.normalization_v4 import normalize_name_v4, parse_address_v4


def extract_v4_owner_record(raw_name: str, raw_addr: str, country: str = "") -> dict:
    """Extract V4 normalized record representation for deterministic conflict resolution."""
    rn = (raw_name or "").strip().lower()
    ra = (raw_addr or "").strip().lower()
    rc = (country or "").strip().lower()
    n4 = normalize_name_v4(raw_name or "")
    a4 = parse_address_v4(raw_addr or "", rc)
    return {
        "raw_name": rn,
        "raw_addr": ra,
        "country": rc,
        "name_norm": n4["lower_norm"],
        "name_core": n4["core_name_norm"] or n4["lower_norm"],
        "name_compact": n4["compact_norm"],
        "name_abbrev": n4["abbrev_norm"] or n4["core_name_norm"],
        "acronym": n4["acronym"],
        "core_tokens": set(n4["core_tokens"]),
        "abbrev_tokens": set(n4["abbrev_tokens"]),
        "phon_tokens": set(n4["phonetic_tokens"]),
        "addr_norm": a4["lower_norm"],
        "addr_abbrev": a4["abbrev_norm"] or a4["lower_norm"],
        "addr_tokens": set(a4["tokens"]),
        "postal": a4["postal_code"] or "",
        "postal_prefix3": a4["postal_prefix3"] or "",
        "building": a4["building_num"] or "",
        "unit": a4["unit_num"] or "",
        "state": a4["state_code"] or "",
        "num_tokens": set(a4["numeric_tokens"]),
    }


def _logit_norm(prob: float) -> float:
    p = min(max(float(prob), 1e-6), 1.0 - 1e-7)
    lo = math.log(p / (1.0 - p))
    lo_min = math.log(0.995 / 0.005)
    lo_max = math.log(0.99999 / 0.00001)
    return min(1.0, max(0.0, (lo - lo_min) / (lo_max - lo_min)))


def compute_v41c_assignment_score(
    s1_rec: dict,
    tgt_rec: dict,
    model_prob: float,
    model_rank: int,
    s1_id: str,
) -> dict:
    """Compute V4.1-C owner evidence and model probability tie-breaking components."""
    raw_name_exact = 1.0 if (s1_rec["raw_name"] and s1_rec["raw_name"] == tgt_rec["raw_name"]) else 0.0
    core_name_exact = 1.0 if (s1_rec["name_core"] and s1_rec["name_core"] == tgt_rec["name_core"]) else 0.0
    abbrev_name_exact = 1.0 if (s1_rec["name_abbrev"] and s1_rec["name_abbrev"] == tgt_rec["name_abbrev"]) else 0.0
    compact_name_exact = 1.0 if (s1_rec["name_compact"] and s1_rec["name_compact"] == tgt_rec["name_compact"]) else 0.0
    acronym_exact = (
        1.0
        if (s1_rec["acronym"] and len(s1_rec["acronym"]) >= 2 and s1_rec["acronym"] == tgt_rec["acronym"])
        else 0.0
    )

    nc1, nc2 = s1_rec["name_abbrev"], tgt_rec["name_abbrev"]
    rn1, rn2 = s1_rec["raw_name"], tgt_rec["raw_name"]
    n_tset = (fuzz.token_set_ratio(nc1, nc2) / 100.0) if (nc1 and nc2) else 0.0
    n_tsort = (fuzz.token_sort_ratio(nc1, nc2) / 100.0) if (nc1 and nc2) else 0.0
    n_raw = (fuzz.ratio(rn1, rn2) / 100.0) if (rn1 and rn2) else 0.0

    ct1, ct2 = s1_rec["abbrev_tokens"], tgt_rec["abbrev_tokens"]
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

    raw_addr_exact = 1.0 if (s1_rec["raw_addr"] and s1_rec["raw_addr"] == tgt_rec["raw_addr"]) else 0.0
    core_addr_exact = 1.0 if (s1_rec["addr_abbrev"] and s1_rec["addr_abbrev"] == tgt_rec["addr_abbrev"]) else 0.0

    ac1, ac2 = s1_rec["addr_abbrev"], tgt_rec["addr_abbrev"]
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

    c1, c2 = s1_rec["country"], tgt_rec["country"]
    if c1 and c2:
        country_agree = 1.0 if (c1 == c2) else -1.0
    else:
        country_agree = 0.0

    nt1, nt2 = s1_rec["num_tokens"], tgt_rec["num_tokens"]
    if nt1 and nt2:
        inter = len(nt1 & nt2)
        union = len(nt1 | nt2)
        num_tok_agree = float(inter / union) if inter > 0 else -0.85
    else:
        num_tok_agree = 0.0

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
        + 0.05 * max(core_name_exact, abbrev_name_exact)
        + 0.03 * compact_name_exact
        + 0.03 * raw_name_exact
        + 0.02 * acronym_exact
    )

    prob_norm = _logit_norm(model_prob)
    rank_bonus = -0.006 * model_rank
    score_v41b = owner_score + 0.18 * prob_norm + rank_bonus

    return {
        "s1_id": s1_id,
        "owner_score": float(owner_score),
        "model_probability": float(model_prob),
        "prob_norm": float(prob_norm),
        "model_rank": int(model_rank),
        "score_v41b": float(score_v41b),
    }


def resolve_v41c_conflict(candidates_info: List[dict], margin_param: float = 0.05) -> str:
    """Resolve a target conflict group using V4.1-C(margin_param=0.05) and return the winning `s1_id`."""
    if len(candidates_info) == 1:
        return candidates_info[0]["s1_id"]

    sorted_by_owner = sorted(
        candidates_info,
        key=lambda x: x["owner_score"],
        reverse=True,
    )
    best_owner_sc = sorted_by_owner[0]["owner_score"]
    near_tied = [
        x for x in sorted_by_owner
        if (best_owner_sc - x["owner_score"]) <= margin_param
    ]
    if len(near_tied) == 1:
        return near_tied[0]["s1_id"]

    winner = max(
        near_tied,
        key=lambda x: (
            round(x["model_probability"], 8),
            -x["model_rank"],
            round(x["score_v41b"], 8),
            x["s1_id"],
        ),
    )
    return winner["s1_id"]


def apply_v4_entity_policy_and_exclusivity(
    probs: np.ndarray,
    feat_df: pd.DataFrame,
    eval_struct: dict,
    policy_cfg: dict,
    s1_raw_lookup: Dict[str, Dict[str, str]],
    target_raw_lookup: Dict[str, Dict[str, str]],
    in_top500_mask: Optional[np.ndarray] = None,
    rescued_min_prob: float = 0.9990,
    exclusivity_margin: float = 0.05,
) -> Tuple[np.ndarray, dict]:
    """Apply Policy 5 + rescued-candidate confirmation + V4.1-C(0.05) target exclusivity decoding.

    Returns:
      - `final_mask`: boolean array over `feat_df` of retained predictions after V4.1-C(0.05)
      - `decoder_stats`: conflict resolution statistics
    """
    base_policy_mask = apply_frozen_policy(probs, feat_df, eval_struct, policy_cfg)
    if in_top500_mask is not None:
        pre_excl_mask = base_policy_mask & (in_top500_mask | (probs >= rescued_min_prob))
    else:
        pre_excl_mask = base_policy_mask.copy()

    cand_col = feat_df["candidate_entity_id"].tolist()
    s1_ids = eval_struct["s1_order"]
    slices = eval_struct["s1_row_slices"]

    # Map each target_id -> list of (row_idx, s1_id, rank_in_s1, prob)
    target_claims: Dict[str, List[Tuple[int, str, int, float]]] = defaultdict(list)

    for s1_idx, (st, en) in enumerate(slices):
        if st >= en:
            continue
        s1_id = s1_ids[s1_idx]
        sel_rel = np.flatnonzero(pre_excl_mask[st:en])
        if len(sel_rel) == 0:
            continue
        sel_rows = [st + int(r) for r in sel_rel]
        sel_rows.sort(key=lambda r: (-float(probs[r]), cand_col[r]))
        for rank_idx, r_idx in enumerate(sel_rows):
            tid = cand_col[r_idx]
            target_claims[tid].append((r_idx, s1_id, rank_idx, float(probs[r_idx])))

    conflicted_targets = {tid: cl for tid, cl in target_claims.items() if len(cl) > 1}
    final_mask = pre_excl_mask.copy()

    if conflicted_targets:
        s1_rec_cache: Dict[str, dict] = {}
        tgt_rec_cache: Dict[str, dict] = {}

        for tid, claims in conflicted_targets.items():
            if tid not in tgt_rec_cache:
                tr = target_raw_lookup.get(tid) or {}
                tgt_rec_cache[tid] = extract_v4_owner_record(
                    tr.get("business_name", ""),
                    tr.get("business_address", ""),
                    tr.get("country", ""),
                )
            tgt_rec = tgt_rec_cache[tid]

            c_infos = []
            for r_idx, s1_id, rank_idx, prob_val in claims:
                if s1_id not in s1_rec_cache:
                    sr = s1_raw_lookup.get(s1_id) or {}
                    s1_rec_cache[s1_id] = extract_v4_owner_record(
                        sr.get("business_name", ""),
                        sr.get("business_address", ""),
                        sr.get("country", ""),
                    )
                s1_rec = s1_rec_cache[s1_id]
                comp = compute_v41c_assignment_score(s1_rec, tgt_rec, prob_val, rank_idx, s1_id)
                comp["row_idx"] = r_idx
                c_infos.append(comp)

            win_s1 = resolve_v41c_conflict(c_infos, margin_param=exclusivity_margin)
            for comp in c_infos:
                if comp["s1_id"] != win_s1:
                    final_mask[comp["row_idx"]] = False

    removed_mask = pre_excl_mask & (~final_mask)
    label_col = eval_struct["label_col"]
    decoder_stats = {
        "decoder_variant": f"V4.1-C(m={exclusivity_margin})",
        "target_conflicts_before": int(len(conflicted_targets)),
        "excess_assignments_removed": int(removed_mask.sum()),
        "gt_positives_removed": int(label_col[removed_mask].sum()),
        "gt_negatives_removed": int((label_col[removed_mask] == 0).sum()),
    }
    return final_mask, decoder_stats
