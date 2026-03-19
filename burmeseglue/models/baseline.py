"""
Baseline models for BurmeseGLUE using TF-IDF + linear models.

Simple but effective baselines using scikit-learn.
"""
import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from typing import Any, Dict, List, Optional, Union


class BaselineClassifier:
    """
    Baseline classifier using TF-IDF + linear model.

    Supports Logistic Regression and Linear SVM.
    """

    def __init__(
        self,
        model_type: str = "logistic",
        vectorizer_params: Optional[Dict] = None,
        model_params: Optional[Dict] = None,
    ):
        """
        Initialize baseline classifier.

        Args:
            model_type: Type of classifier ("logistic" or "svm")
            vectorizer_params: Parameters for TfidfVectorizer
            model_params: Parameters for the classifier
        """
        self.model_type = model_type

        # Default vectorizer parameters
        default_vectorizer_params = {
            "max_features": 10000,
            "min_df": 2,
            "max_df": 0.95,
            "ngram_range": (1, 2),
            "sublinear_tf": True,
        }
        self.vectorizer_params = {
            **default_vectorizer_params,
            **(vectorizer_params or {})
        }

        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)

        # Default model parameters
        if model_type == "logistic":
            default_model_params = {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
            }
            self.model_params = {
                **default_model_params,
                **(model_params or {})
            }
            self.model = LogisticRegression(**self.model_params)

        elif model_type == "svm":
            default_model_params = {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
            }
            self.model_params = {
                **default_model_params,
                **(model_params or {})
            }
            self.model = LinearSVC(**self.model_params)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.is_fitted = False

    def fit(
        self,
        texts: List[str],
        labels: Union[List[int], np.ndarray],
        verbose: bool = True,
    ) -> "BaselineClassifier":
        """
        Train the baseline model.

        Args:
            texts: List of text strings
            labels: List or array of labels
            verbose: Whether to print training info

        Returns:
            Self (for method chaining)
        """
        if verbose:
            print(f"Vectorizing {len(texts)} training examples...")

        # Fit and transform texts
        X = self.vectorizer.fit_transform(texts)

        if verbose:
            print(f"Feature matrix shape: {X.shape}")
            print(f"Training {self.model_type} model...")

        # Train model
        self.model.fit(X, labels)
        self.is_fitted = True

        if verbose:
            print("Training complete!")

        return self

    def predict(
        self,
        texts: List[str],
        return_proba: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """
        Make predictions on texts.

        Args:
            texts: List of text strings
            return_proba: Whether to return class probabilities

        Returns:
            Predictions (and probabilities if return_proba=True)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Transform texts
        X = self.vectorizer.transform(texts)

        # Predict
        predictions = self.model.predict(X)

        if return_proba:
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(X)
            elif hasattr(self.model, "decision_function"):
                # For SVM, use decision function
                decision = self.model.decision_function(X)
                # Convert to pseudo-probabilities
                probas = self._decision_to_proba(decision)
            else:
                probas = None

            return predictions, probas

        return predictions

    def _decision_to_proba(self, decision: np.ndarray) -> np.ndarray:
        """Convert SVM decision function to pseudo-probabilities."""
        if len(decision.shape) == 1:
            # Binary classification
            exp_scores = np.exp(decision)
            probas = np.vstack([1 / (1 + exp_scores), exp_scores / (1 + exp_scores)]).T
        else:
            # Multi-class: softmax
            exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probas

    def evaluate(
        self,
        texts: List[str],
        labels: Union[List[int], np.ndarray],
        compute_metrics_fn: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            texts: List of text strings
            labels: True labels
            compute_metrics_fn: Function to compute metrics

        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(texts)

        if compute_metrics_fn is None:
            from burmeseglue.metrics import compute_classification_metrics
            compute_metrics_fn = compute_classification_metrics

        metrics = compute_metrics_fn(predictions, labels)

        return metrics

    def save(self, output_dir: str) -> None:
        """
        Save model and vectorizer.

        Args:
            output_dir: Directory to save model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save vectorizer
        joblib.dump(self.vectorizer, output_path / "vectorizer.pkl")

        # Save model
        joblib.dump(self.model, output_path / "model.pkl")

        # Save config
        config = {
            "model_type": self.model_type,
            "vectorizer_params": self.vectorizer_params,
            "model_params": self.model_params,
        }
        joblib.dump(config, output_path / "config.pkl")

        print(f"Model saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "BaselineClassifier":
        """
        Load model from directory.

        Args:
            model_dir: Directory containing saved model

        Returns:
            Loaded model instance
        """
        model_path = Path(model_dir)

        # Load config
        config = joblib.load(model_path / "config.pkl")

        # Create instance
        instance = cls(
            model_type=config["model_type"],
            vectorizer_params=config["vectorizer_params"],
            model_params=config["model_params"],
        )

        # Load vectorizer and model
        instance.vectorizer = joblib.load(model_path / "vectorizer.pkl")
        instance.model = joblib.load(model_path / "model.pkl")
        instance.is_fitted = True

        print(f"Model loaded from {model_dir}")

        return instance

    def __repr__(self) -> str:
        return f"BaselineClassifier(model_type={self.model_type}, fitted={self.is_fitted})"
