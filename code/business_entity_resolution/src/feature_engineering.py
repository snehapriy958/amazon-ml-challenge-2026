"""Feature engineering for Business Entity Resolution.
Computes the locked 11-feature schema using verified float64 batched RapidFuzz cpdist.
"""

from typing import Dict, List, Union
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

LOCKED_FEATURE_COLUMNS: List[str] = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "country_same",
    "name_len_1",
    "name_len_2",
    "address_len_1",
    "address_len_2",
]


def build_features_fast(
    pairs_df: pd.DataFrame,
    s1_lookup: Union[pd.DataFrame, Dict[str, Dict[str, str]]],
    s2_lookup: Union[pd.DataFrame, Dict[str, Dict[str, str]]],
    s3_lookup: Union[pd.DataFrame, Dict[str, Dict[str, str]]],
) -> pd.DataFrame:
    """Compute the locked 11 pairwise features using RapidFuzz process.cpdist at float64 precision.
    
    Verified exact logic with 0.0 max absolute difference against the row-by-row reference.
    
    Parameters:
    -----------
    pairs_df: DataFrame containing ['source1_entity_id', 'candidate_entity_id']
    s1_lookup: S1 DataFrame indexed by entity_id, or dictionary mapping entity_id -> row dict
    s2_lookup: S2 DataFrame indexed by entity_id, or dictionary mapping entity_id -> row dict
    s3_lookup: S3 DataFrame indexed by entity_id, or dictionary mapping entity_id -> row dict
    
    Returns:
    --------
    DataFrame with exactly LOCKED_FEATURE_COLUMNS
    """
    if pairs_df.empty:
        return pd.DataFrame(columns=LOCKED_FEATURE_COLUMNS)

    s1_ids = pairs_df["source1_entity_id"].tolist()
    candidate_ids = pairs_df["candidate_entity_id"].tolist()

    # Extract S1 records
    if isinstance(s1_lookup, pd.DataFrame):
        s1_rows = s1_lookup.loc[s1_ids]
        name1 = s1_rows["business_name"].fillna("").astype(str).to_numpy()
        addr1 = s1_rows["business_address"].fillna("").astype(str).to_numpy()
        country1 = s1_rows["country"].fillna("").astype(str).str.lower().to_numpy()
    else:
        name1 = np.array([s1_lookup.get(sid, {}).get("business_name", "") for sid in s1_ids], dtype=object)
        addr1 = np.array([s1_lookup.get(sid, {}).get("business_address", "") for sid in s1_ids], dtype=object)
        country1 = np.array([str(s1_lookup.get(sid, {}).get("country", "")).lower() for sid in s1_ids], dtype=object)

    # Extract Candidate records
    name2_list: List[str] = []
    addr2_list: List[str] = []
    country2_list: List[str] = []

    for cid in candidate_ids:
        if cid.startswith("S2-"):
            rec = s2_lookup.loc[cid] if isinstance(s2_lookup, pd.DataFrame) else s2_lookup.get(cid, {})
        else:
            rec = s3_lookup.loc[cid] if isinstance(s3_lookup, pd.DataFrame) else s3_lookup.get(cid, {})

        name2_list.append(str(rec.get("business_name", "") if isinstance(rec, dict) else rec["business_name"]) if rec is not None else "")
        addr2_list.append(str(rec.get("business_address", "") if isinstance(rec, dict) else rec["business_address"]) if rec is not None else "")
        country2_list.append(str(rec.get("country", "") if isinstance(rec, dict) else rec["country"]).lower() if rec is not None else "")

    name2 = np.array(name2_list, dtype=object)
    addr2 = np.array(addr2_list, dtype=object)
    country2 = np.array(country2_list, dtype=object)

    # Batched pairwise similarities (cpdist) with float64 precision
    name_ratio = process.cpdist(
        name1, name2,
        scorer=fuzz.ratio,
        workers=-1,
        dtype=np.float64
    ) / 100.0

    name_token_sort = process.cpdist(
        name1, name2,
        scorer=fuzz.token_sort_ratio,
        workers=-1,
        dtype=np.float64
    ) / 100.0

    name_token_set = process.cpdist(
        name1, name2,
        scorer=fuzz.token_set_ratio,
        workers=-1,
        dtype=np.float64
    ) / 100.0

    address_ratio = process.cpdist(
        addr1, addr2,
        scorer=fuzz.ratio,
        workers=-1,
        dtype=np.float64
    ) / 100.0

    address_token_sort = process.cpdist(
        addr1, addr2,
        scorer=fuzz.token_sort_ratio,
        workers=-1,
        dtype=np.float64
    ) / 100.0

    address_token_set = process.cpdist(
        addr1, addr2,
        scorer=fuzz.token_set_ratio,
        workers=-1,
        dtype=np.float64
    ) / 100.0

    country_same = (country1 == country2).astype(np.int64)

    return pd.DataFrame({
        "name_ratio": name_ratio,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "address_ratio": address_ratio,
        "address_token_sort": address_token_sort,
        "address_token_set": address_token_set,
        "country_same": country_same,
        "name_len_1": np.array([len(x) for x in name1], dtype=np.int64),
        "name_len_2": np.array([len(x) for x in name2], dtype=np.int64),
        "address_len_1": np.array([len(x) for x in addr1], dtype=np.int64),
        "address_len_2": np.array([len(x) for x in addr2], dtype=np.int64),
    })
