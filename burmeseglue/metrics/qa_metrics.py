"""
Question Answering metrics for BurmeseGLUE.

Implements Exact Match and token-level F1 (SQuAD-style).
"""
import re
import string
from collections import Counter
from typing import Dict, List


def normalize_answer(s: str) -> str:
    """Normalize answer text: lowercase, strip punctuation and extra whitespace."""
    s = s.lower()
    # Remove punctuation
    exclude = set(string.punctuation + "။၊")
    s = "".join(ch for ch in s if ch not in exclude)
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def get_tokens(s: str) -> List[str]:
    """Tokenize (character-level for Burmese, word-split for mixed)."""
    return normalize_answer(s).split() if normalize_answer(s) else []


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = get_tokens(prediction)
    truth_tokens = get_tokens(ground_truth)

    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_qa_metrics(
    predictions: List[str],
    labels: List[List[str]],
) -> Dict[str, float]:
    """
    Compute SQuAD-style Exact Match and F1 for QA.

    Args:
        predictions: List of predicted answer strings
        labels: List of lists of acceptable answer strings (gold answers)

    Returns:
        Dictionary with exact_match and f1
    """
    assert len(predictions) == len(labels), "Predictions and labels must have same length"

    exact_scores = []
    f1_scores = []

    for pred, golds in zip(predictions, labels):
        # Take max over all gold answers
        em = max(compute_exact_match(pred, g) for g in golds)
        f1 = max(compute_f1(pred, g) for g in golds)
        exact_scores.append(em)
        f1_scores.append(f1)

    return {
        "exact_match": round(float(sum(exact_scores) / len(exact_scores)), 4),
        "f1": round(float(sum(f1_scores) / len(f1_scores)), 4),
    }
