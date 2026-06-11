"""
Regression metrics for BurmeseGLUE sentence similarity task.

Uses Pearson and Spearman correlation.
"""
import numpy as np
from typing import Dict, List, Union


def compute_regression_metrics(
    predictions: Union[List[float], np.ndarray],
    labels: Union[List[float], np.ndarray],
) -> Dict[str, float]:
    """
    Compute regression metrics: Pearson and Spearman correlation.

    Args:
        predictions: Predicted similarity scores
        labels: True similarity scores

    Returns:
        Dictionary with pearson, spearman, mse, rmse
    """
    from scipy.stats import pearsonr, spearmanr

    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()

    pearson, _ = pearsonr(predictions, labels)
    spearman, _ = spearmanr(predictions, labels)
    mse = float(np.mean((predictions - labels) ** 2))
    rmse = float(np.sqrt(mse))

    return {
        "pearson": round(float(pearson), 4),
        "spearman": round(float(spearman), 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
    }
