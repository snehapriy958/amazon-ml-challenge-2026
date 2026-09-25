"""Model training, serialization, and scoring for Business Entity Resolution.
Complies with challenge constraints: MIT/Apache 2.0 license, <= 8B parameters.
Uses XGBoost (Apache 2.0).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

try:
    from .feature_engineering import LOCKED_FEATURE_COLUMNS
except ImportError:
    from feature_engineering import LOCKED_FEATURE_COLUMNS



def load_training_chunks(
    chunk_paths: List[Union[str, Path]]
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load training parquet chunks and extract features X and labels y.
    
    Guarantees:
    - X contains ONLY the locked 11 feature columns
    - Entity IDs are strictly EXCLUDED from feature space
    - y contains integer binary labels {0, 1}
    """
    dfs: List[pd.DataFrame] = []
    for path in chunk_paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Training chunk not found: {path}")
        df = pd.read_parquet(path)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    missing = set(LOCKED_FEATURE_COLUMNS) - set(combined.columns)
    if missing:
        raise ValueError(f"Parquet chunks missing required features: {missing}")

    X = combined[LOCKED_FEATURE_COLUMNS].copy()
    y = combined["label"].astype(int).to_numpy()

    return X, y


def get_model(
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.08,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any
) -> XGBClassifier:
    """Instantiate an XGBoost classifier with settings tuned for precision and tabular speed.
    
    License: Apache 2.0 (< 8B parameters), fully compliant with challenge constraints.
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
        **kwargs
    )


def train_model(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    eval_set: Optional[List[Tuple[pd.DataFrame, np.ndarray]]] = None,
    verbose: bool = True
) -> XGBClassifier:
    """Train the classifier on feature matrix X and labels y."""
    model.fit(
        X,
        y,
        eval_set=eval_set,
        verbose=verbose
    )
    return model


def predict_match_probabilities(
    model: XGBClassifier,
    X: pd.DataFrame
) -> np.ndarray:
    """Return predicted probability of positive match (class 1) for candidate pairs."""
    return model.predict_proba(X[LOCKED_FEATURE_COLUMNS])[:, 1]


def save_model(
    model: XGBClassifier,
    output_path: Union[str, Path]
) -> None:
    """Save trained model to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(
    model_path: Union[str, Path]
) -> XGBClassifier:
    """Load trained model from disk."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
