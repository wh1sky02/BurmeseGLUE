"""
Classification metrics for BurmeseGLUE.

Implements accuracy, F1, precision, and recall for classification tasks.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from typing import Dict, List, Optional, Union


def compute_classification_metrics(
    predictions: Union[np.ndarray, List],
    labels: Union[np.ndarray, List],
    average: str = "macro",
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Predicted labels or logits
        labels: True labels
        average: Averaging strategy ('macro', 'micro', 'weighted', 'binary')
        label_names: Optional label names for detailed report

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # Handle logits (multi-dimensional predictions)
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    # Flatten if needed
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
    }

    # Determine if binary or multi-class
    num_classes = len(np.unique(labels))

    if num_classes == 2 and average == "macro":
        # For binary classification, use binary metrics
        metrics["f1"] = f1_score(labels, predictions, average="binary", zero_division=0)
        metrics["precision"] = precision_score(labels, predictions, average="binary", zero_division=0)
        metrics["recall"] = recall_score(labels, predictions, average="binary", zero_division=0)
    else:
        # For multi-class
        metrics["f1"] = f1_score(labels, predictions, average=average, zero_division=0)
        metrics["precision"] = precision_score(labels, predictions, average=average, zero_division=0)
        metrics["recall"] = recall_score(labels, predictions, average=average, zero_division=0)

    # Add macro variants for multi-class
    if num_classes > 2:
        metrics["f1_macro"] = f1_score(labels, predictions, average="macro", zero_division=0)
        metrics["f1_micro"] = f1_score(labels, predictions, average="micro", zero_division=0)
        metrics["f1_weighted"] = f1_score(labels, predictions, average="weighted", zero_division=0)

    return metrics


def compute_classification_report(
    predictions: Union[np.ndarray, List],
    labels: Union[np.ndarray, List],
    label_names: Optional[List[str]] = None,
) -> str:
    """
    Generate detailed classification report.

    Args:
        predictions: Predicted labels
        labels: True labels
        label_names: Optional label names

    Returns:
        Classification report string
    """
    # Convert to numpy arrays
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # Handle logits
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    # Flatten
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Generate report
    report = classification_report(
        labels,
        predictions,
        target_names=label_names,
        zero_division=0,
    )

    return report


def compute_confusion_matrix(
    predictions: Union[np.ndarray, List],
    labels: Union[np.ndarray, List],
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted labels
        labels: True labels

    Returns:
        Confusion matrix
    """
    # Convert to numpy arrays
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # Handle logits
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    # Flatten
    predictions = predictions.flatten()
    labels = labels.flatten()

    return confusion_matrix(labels, predictions)


def compute_metrics_for_transformers(eval_pred):
    """
    Compute metrics in HuggingFace Trainer format.

    Args:
        eval_pred: EvalPrediction object with predictions and label_ids

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    return compute_classification_metrics(predictions, labels)
