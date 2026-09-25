"""Data loading, normalization, and splitting utilities for Business Entity Resolution.
Preserves exact verified normalization and splitting logic.
"""

from pathlib import Path
import re
from typing import Any, Iterable, List, Optional, Set, Union
import numpy as np
import pandas as pd


def normalize_name(x: Any) -> str:
    """Normalize business name by lowercasing and replacing non-alphanumeric chars with spaces.
    
    Verified exact logic from project inspection.
    """
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def normalize_address(x: Any) -> str:
    """Normalize business address by lowercasing and replacing non-alphanumeric chars with spaces.
    
    Verified exact logic from project inspection.
    """
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def load_source_tsv(
    path: Union[str, Path],
    usecols: Optional[List[str]] = None,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """Load a tab-separated source file (*_source1.tsv, *_source2.tsv, etc.) with explicit sep='\\t'.
    
    Ensures all columns are read as strings to avoid numeric coercion.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=usecols,
        nrows=nrows
    )


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add name_norm and address_norm columns to a source DataFrame.
    
    Expects 'business_name' and 'business_address' in df.columns.
    """
    if "business_name" not in df.columns or "business_address" not in df.columns:
        raise ValueError("DataFrame must contain 'business_name' and 'business_address'.")

    df = df.copy()
    df["name_norm"] = df["business_name"].map(normalize_name)
    df["address_norm"] = df["business_address"].map(normalize_address)
    return df


def parse_match_ids(value: Any) -> List[str]:
    """Parse comma-separated match IDs into a deduplicated list of valid S2-/S3- entity IDs.
    
    Ignores empty values, literal 'nan', and non-S2/S3 IDs.
    Preserves first-seen order.
    """
    if pd.isna(value):
        return []

    value = str(value).strip()
    if not value or value.lower() == "nan":
        return []

    ids: List[str] = []
    for x in value.split(","):
        x = x.strip()
        if not x or x.lower() == "nan":
            continue
        if x.startswith("S2-") or x.startswith("S3-"):
            ids.append(x)

    return list(dict.fromkeys(ids))


def create_train_val_split(
    gt_df: pd.DataFrame,
    train_ratio: float = 0.8
) -> pd.DataFrame:
    """Deterministic hash-based train/validation split by S1 entity ID.
    
    Guarantees no S1 entity leaks across splits (80% train / 20% validation).
    """
    df = gt_df.copy()
    if "split" in df.columns:
        return df
    s1_ids = df["source1_entity_id"].astype(str).str.strip()

    hash_values = pd.util.hash_pandas_object(s1_ids, index=False)
    threshold = int(train_ratio * 10)
    df["split"] = np.where(hash_values % 10 < threshold, "train", "validation")
    return df
