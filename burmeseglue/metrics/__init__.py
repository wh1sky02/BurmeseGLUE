"""
Evaluation metrics for BurmeseGLUE tasks.

Available metrics:
- Classification: accuracy, F1, precision, recall
- Token classification: span-level F1
- Regression: Pearson, Spearman correlation
- QA: exact match, F1
"""

from burmeseglue.metrics.classification import (
    compute_classification_metrics,
    compute_classification_report,
    compute_confusion_matrix,
    compute_metrics_for_transformers,
)

# Metric implementations will be imported as they're implemented
# from burmeseglue.metrics.token_metrics import compute_token_metrics
# from burmeseglue.metrics.regression import compute_regression_metrics
# from burmeseglue.metrics.qa_metrics import compute_qa_metrics

__all__ = [
    "compute_classification_metrics",
    "compute_classification_report",
    "compute_confusion_matrix",
    "compute_metrics_for_transformers",
    # "compute_token_metrics",
    # "compute_regression_metrics",
    # "compute_qa_metrics",
]
