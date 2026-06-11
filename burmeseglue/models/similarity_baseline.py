"""
TF-IDF + Ridge Regression baseline for Burmese sentence similarity.
"""
import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack
from typing import Dict, List, Optional, Union


class SimilarityBaselineModel:
    """
    Sentence similarity baseline using TF-IDF features + Ridge regression.

    Features: [v1, v2, |v1-v2|, v1*v2] concatenated dense vectors.
    """

    def __init__(self, alpha: float = 1.0, max_features: int = 5000):
        self.alpha = alpha
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.regressor = Ridge(alpha=alpha)
        self.is_fitted = False

    def _make_features(self, sentences1: List[str], sentences2: List[str],
                       fit: bool = False) -> np.ndarray:
        all_sentences = sentences1 + sentences2

        if fit:
            self.vectorizer.fit(all_sentences)

        v1 = self.vectorizer.transform(sentences1).toarray()
        v2 = self.vectorizer.transform(sentences2).toarray()

        diff = np.abs(v1 - v2)
        prod = v1 * v2

        return np.hstack([v1, v2, diff, prod])

    def fit(
        self,
        sentences1: List[str],
        sentences2: List[str],
        scores: Union[List[float], np.ndarray],
        verbose: bool = True,
    ) -> "SimilarityBaselineModel":
        if verbose:
            print(f"Building TF-IDF features for {len(sentences1)} pairs...")

        X = self._make_features(sentences1, sentences2, fit=True)
        y = np.array(scores)

        if verbose:
            print(f"Feature matrix shape: {X.shape}")
            print("Training Ridge regression model...")

        self.regressor.fit(X, y)
        self.is_fitted = True

        if verbose:
            print("Training complete!")

        return self

    def predict(self, sentences1: List[str], sentences2: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X = self._make_features(sentences1, sentences2, fit=False)
        preds = self.regressor.predict(X)
        # Clip to [0, 5] range
        return np.clip(preds, 0.0, 5.0)

    def evaluate(
        self,
        sentences1: List[str],
        sentences2: List[str],
        scores: Union[List[float], np.ndarray],
        compute_metrics_fn=None,
    ) -> Dict[str, float]:
        predictions = self.predict(sentences1, sentences2)

        if compute_metrics_fn is None:
            from burmeseglue.metrics.regression_metrics import compute_regression_metrics
            compute_metrics_fn = compute_regression_metrics

        return compute_metrics_fn(predictions, scores)

    def save(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, output_path / "vectorizer.pkl")
        joblib.dump(self.regressor, output_path / "regressor.pkl")
        joblib.dump({"alpha": self.alpha, "max_features": self.max_features},
                    output_path / "config.pkl")
        print(f"Similarity model saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "SimilarityBaselineModel":
        model_path = Path(model_dir)
        config = joblib.load(model_path / "config.pkl")
        instance = cls(**config)
        instance.vectorizer = joblib.load(model_path / "vectorizer.pkl")
        instance.regressor = joblib.load(model_path / "regressor.pkl")
        instance.is_fitted = True
        return instance

    def __repr__(self):
        return f"SimilarityBaselineModel(alpha={self.alpha}, fitted={self.is_fitted})"
