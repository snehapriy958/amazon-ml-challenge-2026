"""Blocking and candidate generation for Business Entity Resolution.
Implements the verified top-3 rare-token + same-country blocking strategy.
"""

from collections import Counter, defaultdict
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import pandas as pd


def tokenize(text: Any) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens.
    
    Verified exact logic from project inspection.
    """
    if pd.isna(text):
        return []
    text = str(text).lower()
    return re.findall(r"[a-z0-9]+", text)


def build_field_index(df: pd.DataFrame, column: str) -> Dict[str, Set[str]]:
    """Index entity IDs by exact normalized field value (e.g. name_norm or address_norm)."""
    index: Dict[str, Set[str]] = defaultdict(set)
    for entity_id, val in zip(df["entity_id"], df[column]):
        if val:
            index[val].add(entity_id)
    return dict(index)


def build_token_index(
    df: pd.DataFrame,
    min_len: int = 3
) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """Build inverted token index and token frequency dictionary from business name and address.
    
    Only tokens with length >= min_len are included.
    """
    token_index: Dict[str, Set[str]] = defaultdict(set)
    token_freq: Counter = Counter()

    for row in df.itertuples(index=False):
        name_val = getattr(row, "business_name", "")
        addr_val = getattr(row, "business_address", "")
        tokens = set(
            t for t in tokenize(name_val) + tokenize(addr_val)
            if len(t) >= min_len
        )
        for token in tokens:
            token_index[token].add(row.entity_id)
            token_freq[token] += 1

    return dict(token_index), dict(token_freq)


class CandidateBlocker:
    """Manages inverted indexes and generates candidate matches for S1 entities.
    
    Implements:
    1. Exact normalized name matching
    2. Exact normalized address matching
    3. Top-k rarest usable tokens (frequency <= max_freq)
    4. Same-country filtering (open-set string equality)
    """

    def __init__(
        self,
        s2_name_ids: Dict[str, Set[str]],
        s3_name_ids: Dict[str, Set[str]],
        s2_address_ids: Dict[str, Set[str]],
        s3_address_ids: Dict[str, Set[str]],
        s2_token_index: Dict[str, Set[str]],
        s3_token_index: Dict[str, Set[str]],
        s2_token_freq: Dict[str, int],
        s3_token_freq: Dict[str, int],
        s2_country: Dict[str, str],
        s3_country: Dict[str, str],
    ):
        self.s2_name_ids = s2_name_ids
        self.s3_name_ids = s3_name_ids
        self.s2_address_ids = s2_address_ids
        self.s3_address_ids = s3_address_ids
        self.s2_token_index = s2_token_index
        self.s3_token_index = s3_token_index
        self.s2_token_freq = s2_token_freq
        self.s3_token_freq = s3_token_freq
        self.s2_country = s2_country
        self.s3_country = s3_country

    @classmethod
    def from_dataframes(cls, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> "CandidateBlocker":
        """Build all blocking structures from preprocessed Source 2 and Source 3 DataFrames."""
        s2_name_ids = build_field_index(s2_df, "name_norm")
        s3_name_ids = build_field_index(s3_df, "name_norm")
        s2_address_ids = build_field_index(s2_df, "address_norm")
        s3_address_ids = build_field_index(s3_df, "address_norm")

        s2_token_index, s2_token_freq = build_token_index(s2_df)
        s3_token_index, s3_token_freq = build_token_index(s3_df)

        s2_country = dict(zip(s2_df["entity_id"], s2_df["country"].fillna("").astype(str)))
        s3_country = dict(zip(s3_df["entity_id"], s3_df["country"].fillna("").astype(str)))

        return cls(
            s2_name_ids=s2_name_ids,
            s3_name_ids=s3_name_ids,
            s2_address_ids=s2_address_ids,
            s3_address_ids=s3_address_ids,
            s2_token_index=s2_token_index,
            s3_token_index=s3_token_index,
            s2_token_freq=s2_token_freq,
            s3_token_freq=s3_token_freq,
            s2_country=s2_country,
            s3_country=s3_country,
        )

    def get_candidates(
        self,
        name_norm: str,
        address_norm: str,
        country: str,
        top_k: int = 3,
        max_freq: int = 100_000
    ) -> Set[str]:
        """Generate candidate S2/S3 entity IDs for a Source 1 entity."""
        candidates: Set[str] = set()
        country_norm = str(country).strip().lower()

        # 1. Exact normalized matches
        if name_norm:
            candidates.update(self.s2_name_ids.get(name_norm, ()))
            candidates.update(self.s3_name_ids.get(name_norm, ()))

        if address_norm:
            candidates.update(self.s2_address_ids.get(address_norm, ()))
            candidates.update(self.s3_address_ids.get(address_norm, ()))

        # 2. Rare-token candidates
        tokens = set(
            t for t in tokenize(name_norm) + tokenize(address_norm)
            if len(t) >= 3
        )

        token_info = []
        for token in tokens:
            f2 = self.s2_token_freq.get(token, 0)
            f3 = self.s3_token_freq.get(token, 0)
            freq = max(f2, f3)
            if 0 < freq <= max_freq:
                token_info.append((freq, token))

        token_info.sort()

        for _, token in token_info[:top_k]:
            candidates.update(self.s2_token_index.get(token, ()))
            candidates.update(self.s3_token_index.get(token, ()))

        # 3. Same-country filtering (open set, case-insensitive)
        if country_norm:
            candidates = {
                cid for cid in candidates
                if str(self.s2_country.get(cid, self.s3_country.get(cid, ""))).strip().lower() == country_norm
            }

        return candidates

    def get_training_negatives(
        self,
        name_norm: str,
        address_norm: str,
        country: str,
        true_matches: Set[str],
        max_negatives: int = 3,
        top_k: int = 3,
        max_freq: int = 100_000
    ) -> List[str]:
        """Sample hard negatives for training, strictly excluding all known GT matches."""
        negatives: Set[str] = set()
        country_norm = str(country).strip().lower()

        tokens = set(
            t for t in tokenize(name_norm) + tokenize(address_norm)
            if len(t) >= 3
        )

        token_info = []
        for token in tokens:
            f2 = self.s2_token_freq.get(token, 0)
            f3 = self.s3_token_freq.get(token, 0)
            freq = max(f2, f3)
            if 0 < freq <= max_freq:
                token_info.append((freq, token))

        token_info.sort()

        for _, token in token_info[:top_k]:
            # S2 postings
            for cid in self.s2_token_index.get(token, ()):
                if cid in true_matches or cid in negatives:
                    continue
                if str(self.s2_country.get(cid, "")).strip().lower() == country_norm:
                    negatives.add(cid)
                    if len(negatives) >= max_negatives:
                        return list(negatives)

            # S3 postings
            for cid in self.s3_token_index.get(token, ()):
                if cid in true_matches or cid in negatives:
                    continue
                if str(self.s3_country.get(cid, "")).strip().lower() == country_norm:
                    negatives.add(cid)
                    if len(negatives) >= max_negatives:
                        return list(negatives)

        return list(negatives)
