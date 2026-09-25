"""Official evaluation metric implementation for Amazon ML Challenge 2026.
Macro-averaged F_0.5 score per Source 1 entity, including singletons.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np


def compute_entity_f_beta(
    true_set: Set[str],
    pred_set: Set[str],
    beta: float = 0.5
) -> Tuple[float, float, float]:
    """Compute (precision, recall, f_beta) for a single Source 1 entity.
    
    Includes singleton scoring rules:
    - True singleton (empty true_set):
        - Pred empty -> precision=1.0, recall=1.0, f_beta=1.0
        - Pred non-empty -> precision=0.0, recall=0.0, f_beta=0.0
    - True non-singleton (non-empty true_set):
        - Pred empty -> precision=0.0, recall=0.0, f_beta=0.0
        - Pred non-empty -> standard precision, recall, f_beta
    """
    beta_sq = beta ** 2

    # Case 1: True singleton (no true matches)
    if len(true_set) == 0:
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    # Case 2: True matches exist, but nothing predicted
    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0

    # Case 3: Both non-empty
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    if tp == 0:
        return 0.0, 0.0, 0.0

    denominator = (beta_sq * precision) + recall
    if denominator == 0:
        return 0.0, 0.0, 0.0

    f_beta = ((1.0 + beta_sq) * precision * recall) / denominator
    return precision, recall, f_beta


def evaluate_macro_f_beta(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5
) -> Dict[str, float]:
    """Compute macro-averaged precision, recall, and F_beta across all Source 1 entities.
    
    Parameters:
    -----------
    ground_truth: mapping of {s1_entity_id: set_of_true_matched_ids}
    predictions: mapping of {s1_entity_id: set_of_predicted_matched_ids}
    beta: F-score weight (default 0.5 for precision-heavy evaluation)
    
    Returns:
    --------
    Dictionary containing macro_f05, macro_precision, macro_recall,
    singleton_accuracy, and entity count.
    """
    all_s1_ids = list(ground_truth.keys())
    if not all_s1_ids:
        return {
            "macro_f_beta": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_accuracy": 0.0,
            "total_entities": 0,
        }

    f_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    singleton_correct: int = 0
    singleton_total: int = 0

    for s1_id in all_s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        p, r, f = compute_entity_f_beta(true_set, pred_set, beta=beta)
        precisions.append(p)
        recalls.append(r)
        f_scores.append(f)

        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1

    return {
        "macro_f_beta": float(np.mean(f_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "singleton_accuracy": float(singleton_correct / singleton_total) if singleton_total > 0 else 1.0,
        "total_entities": len(all_s1_ids),
        "total_singletons": singleton_total,
    }
