"""
CRF-based NER baseline model for BurmeseGLUE.

Uses sklearn-crfsuite with hand-crafted features for Burmese NER.
"""
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def _word_features(tokens: List[str], i: int) -> Dict:
    """Extract features for a single token at position i."""
    word = tokens[i]

    features = {
        "bias": 1.0,
        "word": word,
        "word.lower": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:3]": word[:3],
        "word[:2]": word[:2],
        "word.isdigit": word.isdigit(),
        "word.isupper": word.isupper(),
        "word.len": len(word),
    }

    if i > 0:
        prev = tokens[i - 1]
        features.update({
            "-1:word": prev,
            "-1:word.lower": prev.lower(),
            "-1:word[-2:]": prev[-2:],
        })
    else:
        features["BOS"] = True

    if i > 1:
        prev2 = tokens[i - 2]
        features["-2:word"] = prev2

    if i < len(tokens) - 1:
        nxt = tokens[i + 1]
        features.update({
            "+1:word": nxt,
            "+1:word.lower": nxt.lower(),
            "+1:word[-2:]": nxt[-2:],
        })
    else:
        features["EOS"] = True

    if i < len(tokens) - 2:
        nxt2 = tokens[i + 2]
        features["+2:word"] = nxt2

    return features


def _sentence_features(tokens: List[str]) -> List[Dict]:
    return [_word_features(tokens, i) for i in range(len(tokens))]


class NERBaselineModel:
    """
    CRF-based NER baseline using sklearn-crfsuite.
    """

    def __init__(self, algorithm: str = "lbfgs", c1: float = 0.1, c2: float = 0.1,
                 max_iterations: int = 100):
        import sklearn_crfsuite
        self.algorithm = algorithm
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.model = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )
        self.is_fitted = False

    def fit(
        self,
        token_sequences: List[List[str]],
        tag_sequences: List[List[str]],
        verbose: bool = True,
    ) -> "NERBaselineModel":
        if verbose:
            print(f"Extracting features for {len(token_sequences)} sequences...")

        X = [_sentence_features(tokens) for tokens in token_sequences]
        y = tag_sequences

        if verbose:
            print("Training CRF model...")

        self.model.fit(X, y)
        self.is_fitted = True

        if verbose:
            print("Training complete!")

        return self

    def predict(self, token_sequences: List[List[str]]) -> List[List[str]]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = [_sentence_features(tokens) for tokens in token_sequences]
        return self.model.predict(X)

    def evaluate(
        self,
        token_sequences: List[List[str]],
        tag_sequences: List[List[str]],
        compute_metrics_fn=None,
    ) -> Dict[str, float]:
        predictions = self.predict(token_sequences)

        if compute_metrics_fn is None:
            from burmeseglue.metrics.token_metrics import compute_token_metrics
            compute_metrics_fn = compute_token_metrics

        return compute_metrics_fn(predictions, tag_sequences)

    def save(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, output_path / "crf_model.pkl")
        config = {
            "algorithm": self.algorithm,
            "c1": self.c1,
            "c2": self.c2,
            "max_iterations": self.max_iterations,
        }
        joblib.dump(config, output_path / "config.pkl")
        print(f"NER model saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "NERBaselineModel":
        model_path = Path(model_dir)
        config = joblib.load(model_path / "config.pkl")
        instance = cls(**config)
        instance.model = joblib.load(model_path / "crf_model.pkl")
        instance.is_fitted = True
        return instance

    def __repr__(self):
        return f"NERBaselineModel(algorithm={self.algorithm}, fitted={self.is_fitted})"
