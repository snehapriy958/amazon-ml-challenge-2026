"""Candidate scoring, threshold filtering, and submission generation.
Ensures strict compliance with submission formatting rules:
- Exactly one row per Source 1 entity
- matched_entity_ids is a subset of candidate_entity_ids
- Empty string for predicted singletons
- No duplicate IDs, comma-separated, tab-delimited
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

try:
    from .blocking import CandidateBlocker
    from .feature_engineering import LOCKED_FEATURE_COLUMNS, build_features_fast
    from .model import predict_match_probabilities
except ImportError:
    from blocking import CandidateBlocker
    from feature_engineering import LOCKED_FEATURE_COLUMNS, build_features_fast
    from model import predict_match_probabilities



def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    blocker: CandidateBlocker,
    top_k: int = 3,
    max_freq: int = 100_000
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Generate candidate pairs for all S1 entities in s1_df.
    
    Returns:
    --------
    pairs_df: DataFrame of ['source1_entity_id', 'candidate_entity_id']
    candidates_by_s1: Mapping of s1_id -> list of candidate IDs
    """
    pairs: List[Tuple[str, str]] = []
    candidates_by_s1: Dict[str, List[str]] = {}

    for row in s1_df.itertuples(index=False):
        s1_id = row.entity_id
        cands = blocker.get_candidates(
            name_norm=getattr(row, "name_norm", ""),
            address_norm=getattr(row, "address_norm", ""),
            country=getattr(row, "country", ""),
            top_k=top_k,
            max_freq=max_freq
        )
        cand_list = sorted(cands)
        candidates_by_s1[s1_id] = cand_list
        for cid in cand_list:
            pairs.append((s1_id, cid))

    pairs_df = pd.DataFrame(
        pairs,
        columns=["source1_entity_id", "candidate_entity_id"]
    )
    return pairs_df, candidates_by_s1


def score_candidate_pairs_batched(
    model: Any,
    pairs_df: pd.DataFrame,
    s1_lookup: Any,
    s2_lookup: Any,
    s3_lookup: Any,
    batch_size: int = 50_000
) -> pd.DataFrame:
    """Score candidate pairs in batches to minimize memory footprint.
    
    Returns:
    --------
    DataFrame with ['source1_entity_id', 'candidate_entity_id', 'match_prob']
    """
    if pairs_df.empty:
        result = pairs_df.copy()
        result["match_prob"] = np.array([], dtype=float)
        return result

    scores: List[np.ndarray] = []
    total_pairs = len(pairs_df)

    for i in range(0, total_pairs, batch_size):
        batch = pairs_df.iloc[i : i + batch_size].copy()
        feat_df = build_features_fast(batch, s1_lookup, s2_lookup, s3_lookup)
        probs = predict_match_probabilities(model, feat_df)
        scores.append(probs)

    result = pairs_df.copy()
    result["match_prob"] = np.concatenate(scores) if scores else np.array([])
    return result


def build_submission_dataframes(
    all_s1_ids: List[str],
    candidates_by_s1: Dict[str, List[str]],
    scored_pairs_df: pd.DataFrame,
    threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct matching_results.tsv and candidate_pairs.tsv DataFrames.
    
    Guarantees:
    - Every S1 entity has exactly one row in both DataFrames
    - matched_entity_ids is empty for singletons
    - matched_entity_ids is a strict subset of candidate_entity_ids
    - Comma-separated, no duplicate IDs
    """
    # Group matched IDs above threshold by s1_id
    matches_by_s1: Dict[str, List[str]] = defaultdict(list)
    if not scored_pairs_df.empty:
        above_thresh = scored_pairs_df[scored_pairs_df["match_prob"] >= threshold]
        for s1_id, cid in zip(above_thresh["source1_entity_id"], above_thresh["candidate_entity_id"]):
            matches_by_s1[s1_id].append(cid)

    matching_rows = []
    candidate_rows = []

    for s1_id in all_s1_ids:
        # Candidate list
        cands = candidates_by_s1.get(s1_id, [])
        cand_str = ",".join(cands)
        candidate_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": cand_str
        })

        # Matches must be a subset of candidates
        matches = matches_by_s1.get(s1_id, [])
        cands_set = set(cands)
        valid_matches = [m for m in dict.fromkeys(matches) if m in cands_set]
        match_str = ",".join(valid_matches)

        matching_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": match_str
        })

    matching_df = pd.DataFrame(matching_rows)
    candidate_df = pd.DataFrame(candidate_rows)

    return matching_df, candidate_df


def save_submission_files(
    matching_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    output_dir: Union[str, Path]
) -> Tuple[Path, Path]:
    """Save matching_results.tsv and candidate_pairs.tsv with sep='\\t'."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    matching_file = out_path / "matching_results.tsv"
    candidate_file = out_path / "candidate_pairs.tsv"

    matching_df.to_csv(matching_file, sep="\t", index=False)
    candidate_df.to_csv(candidate_file, sep="\t", index=False)

    return matching_file, candidate_file
