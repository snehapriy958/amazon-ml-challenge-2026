"""Main entry point for Amazon ML Challenge 2026 Business Entity Resolution.
Supports:
- model training from existing parquet chunks
- validation & threshold tuning on held-out split (macro F_0.5)
- full test inference producing matching_results.tsv and candidate_pairs.tsv
"""

import argparse
from pathlib import Path
import sys
import time
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from .blocking import CandidateBlocker
    from .evaluation import evaluate_macro_f_beta
    from .feature_engineering import LOCKED_FEATURE_COLUMNS
    from .inference import (
        build_submission_dataframes,
        generate_candidate_pairs,
        save_submission_files,
        score_candidate_pairs_batched,
    )
    from .model import (
        get_model,
        load_model,
        load_training_chunks,
        predict_match_probabilities,
        save_model,
        train_model,
    )
    from .preprocessing import (
        add_normalized_columns,
        create_train_val_split,
        load_source_tsv,
        parse_match_ids,
    )
except ImportError:
    from blocking import CandidateBlocker
    from evaluation import evaluate_macro_f_beta
    from feature_engineering import LOCKED_FEATURE_COLUMNS
    from inference import (
        build_submission_dataframes,
        generate_candidate_pairs,
        save_submission_files,
        score_candidate_pairs_batched,
    )
    from model import (
        get_model,
        load_model,
        load_training_chunks,
        predict_match_probabilities,
        save_model,
        train_model,
    )
    from preprocessing import (
        add_normalized_columns,
        create_train_val_split,
        load_source_tsv,
        parse_match_ids,
    )



def run_train(
    chunks_dir: Path,
    model_output_path: Path,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.08
) -> None:
    """Train XGBoost model on pre-generated parquet chunks."""
    chunk_files = sorted(list(chunks_dir.glob("train_features_chunk_*.parquet")))
    if not chunk_files:
        raise FileNotFoundError(f"No parquet chunks found in {chunks_dir}")

    print(f"Loading {len(chunk_files)} training chunks from {chunks_dir}...")
    X, y = load_training_chunks(chunk_files)
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: positive={(y == 1).sum():,}, negative={(y == 0).sum():,}")

    model = get_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )

    print("Fitting model...")
    t0 = time.time()
    train_model(model, X, y, verbose=True)
    print(f"Model trained in {time.time() - t0:.2f}s")

    save_model(model, model_output_path)
    print(f"Model saved to {model_output_path}")


def run_validation(
    data_dir: Path,
    model_path: Path,
    sample_size: Optional[int] = 10_000,
    thresholds: Optional[List[float]] = None
) -> None:
    """Run validation evaluation on held-out split using official macro F_0.5 metric."""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]

    print("Loading datasets for validation...")
    gt = load_source_tsv(data_dir / "train" / "train_ground_truth.tsv")
    gt = create_train_val_split(gt)
    val_gt = gt[gt["split"] == "validation"].copy()

    s1 = load_source_tsv(data_dir / "train" / "train_source1.tsv")
    s2 = load_source_tsv(data_dir / "train" / "train_source2.tsv")
    s3 = load_source_tsv(data_dir / "train" / "train_source3.tsv")

    print("Normalizing records...")
    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)
    s3 = add_normalized_columns(s3)

    # Filter to validation S1 entities
    val_s1 = s1[s1["entity_id"].isin(set(val_gt["source1_entity_id"]))].copy()
    if sample_size and len(val_s1) > sample_size:
        print(f"Sampling {sample_size:,} validation entities for evaluation...")
        val_s1 = val_s1.sample(n=sample_size, random_state=42).copy()

    val_s1_ids = val_s1["entity_id"].tolist()
    gt_map = {
        row.source1_entity_id: set(parse_match_ids(row.matched_entity_ids))
        for row in val_gt[val_gt["source1_entity_id"].isin(val_s1_ids)].itertuples(index=False)
    }

    print("Building candidate blocker on train S2/S3...")
    blocker = CandidateBlocker.from_dataframes(s2, s3)

    print("Generating candidate pairs...")
    pairs_df, candidates_by_s1 = generate_candidate_pairs(val_s1, blocker, top_k=3)
    print(f"Generated {len(pairs_df):,} candidate pairs across {len(val_s1):,} S1 entities")

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    print("Scoring candidate pairs...")
    s1_lookup = val_s1.set_index("entity_id")
    s2_lookup = s2.set_index("entity_id")
    s3_lookup = s3.set_index("entity_id")

    scored_pairs = score_candidate_pairs_batched(model, pairs_df, s1_lookup, s2_lookup, s3_lookup)

    print("\n" + "=" * 80)
    print(f"{'Threshold':>9} | {'Macro F_0.5':>11} | {'Precision':>9} | {'Recall':>8} | {'Pred Singletons':>15}")
    print("-" * 80)
    best_f = -1.0
    best_thresh = 0.5
    best_res = {}

    for thresh in thresholds:
        matching_df, _ = build_submission_dataframes(val_s1_ids, candidates_by_s1, scored_pairs, thresh)
        pred_singletons = int((matching_df["matched_entity_ids"] == "").sum())
        pred_map = {
            row.source1_entity_id: set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set()
            for row in matching_df.itertuples(index=False)
        }
        res = evaluate_macro_f_beta(gt_map, pred_map, beta=0.5)
        print(
            f"{thresh:9.2f} | "
            f"{res['macro_f_beta']:11.4f} | "
            f"{res['macro_precision']:9.4f} | "
            f"{res['macro_recall']:8.4f} | "
            f"{pred_singletons:15,d}"
        )
        if res["macro_f_beta"] > best_f:
            best_f = res["macro_f_beta"]
            best_thresh = thresh
            best_res = {**res, "predicted_singletons": pred_singletons}

    print("=" * 80)
    print(f"Selected Validation Threshold: {best_thresh:.2f} (Macro F_0.5 = {best_f:.4f})")
    return best_thresh, best_res



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Business Entity Resolution")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model from parquet chunks")
    train_parser.add_argument("--chunks-dir", type=Path, default=Path("training_features_chunks"))
    train_parser.add_argument("--output-model", type=Path, default=Path("model.joblib"))

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Run validation evaluation")
    val_parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    val_parser.add_argument("--model-path", type=Path, default=Path("model.joblib"))
    val_parser.add_argument("--sample-size", type=int, default=10000)

    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Generate test predictions")
    pred_parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    pred_parser.add_argument("--model-path", type=Path, default=Path("model.joblib"))
    pred_parser.add_argument("--output-dir", type=Path, default=Path("output"))
    pred_parser.add_argument("--threshold", type=float, default=0.7)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args.chunks_dir, args.output_model)
    elif args.command == "validate":
        run_validation(args.data_dir, args.model_path, sample_size=args.sample_size)
    elif args.command == "predict":
        print("Predict command reserved for Phase 4.")
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
