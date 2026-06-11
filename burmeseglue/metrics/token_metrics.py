"""
Token classification (NER) metrics for BurmeseGLUE.

Uses seqeval for span-level F1 evaluation with BIO tagging.
"""
from typing import Dict, List


def compute_token_metrics(
    predictions: List[List[str]],
    labels: List[List[str]],
) -> Dict[str, float]:
    """
    Compute span-level NER metrics using seqeval.

    Args:
        predictions: List of predicted tag sequences
        labels: List of true tag sequences

    Returns:
        Dictionary with precision, recall, f1 (span-level)
    """
    from seqeval.metrics import (
        f1_score,
        precision_score,
        recall_score,
        classification_report,
    )

    # seqeval expects lists of lists of strings
    p = precision_score(labels, predictions, zero_division=0)
    r = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "span_f1": round(f1, 4),
    }


def compute_token_classification_report(
    predictions: List[List[str]],
    labels: List[List[str]],
) -> str:
    """
    Generate a detailed per-entity-type classification report.

    Args:
        predictions: List of predicted tag sequences
        labels: List of true tag sequences

    Returns:
        String classification report
    """
    from seqeval.metrics import classification_report

    return classification_report(labels, predictions, zero_division=0)
