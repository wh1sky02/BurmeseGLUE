"""
TF-IDF based extractive QA baseline for BurmeseGLUE.

Finds the best matching sentence in the context using TF-IDF similarity to the question,
then returns it as the answer span.
"""
import re
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional


def _split_sentences(text: str) -> List[str]:
    """Split context into sentences on Burmese and ASCII sentence endings."""
    parts = re.split(r'[။\.\?!]', text)
    return [p.strip() for p in parts if p.strip()]


class QABaselineModel:
    """
    Extractive QA baseline using TF-IDF sentence retrieval.

    For each (context, question) pair:
    1. Split context into candidate spans (sentences / n-gram windows).
    2. Rank by TF-IDF cosine similarity to the question.
    3. Return the top-ranked span as the predicted answer.
    """

    def __init__(self, max_features: int = 5000, window_size: int = 5):
        self.max_features = max_features
        self.window_size = window_size
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            analyzer="char_wb",
        )
        self.is_fitted = True  # QA baseline fits on-the-fly per example

    def _get_candidates(self, context: str) -> List[str]:
        """Generate candidate answer spans from context."""
        # Sentence-level candidates
        sentences = _split_sentences(context)
        if not sentences:
            return [context]

        candidates = sentences[:]

        # Add character-level windows (word-level approximation)
        words = context.split()
        for size in [1, 2, 3]:
            for start in range(len(words)):
                span = " ".join(words[start:start + size])
                if span and span not in candidates:
                    candidates.append(span)

        return candidates

    def predict_single(self, context: str, question: str) -> str:
        """Predict answer for a single (context, question) pair."""
        candidates = self._get_candidates(context)
        if not candidates:
            return ""

        all_texts = [question] + candidates
        try:
            tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                analyzer="char_wb",
            )
            tfidf.fit(all_texts)
            q_vec = tfidf.transform([question])
            c_vecs = tfidf.transform(candidates)
            sims = cosine_similarity(q_vec, c_vecs).flatten()
            best_idx = int(np.argmax(sims))
            return candidates[best_idx]
        except Exception:
            return candidates[0] if candidates else ""

    def predict(self, contexts: List[str], questions: List[str]) -> List[str]:
        """Predict answers for a batch of (context, question) pairs."""
        return [
            self.predict_single(ctx, q)
            for ctx, q in zip(contexts, questions)
        ]

    def evaluate(
        self,
        contexts: List[str],
        questions: List[str],
        gold_answers: List[List[str]],
        compute_metrics_fn=None,
    ) -> Dict[str, float]:
        predictions = self.predict(contexts, questions)

        if compute_metrics_fn is None:
            from burmeseglue.metrics.qa_metrics import compute_qa_metrics
            compute_metrics_fn = compute_qa_metrics

        return compute_metrics_fn(predictions, gold_answers)

    def save(self, output_dir: str) -> None:
        import joblib
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"max_features": self.max_features,
                     "window_size": self.window_size},
                    output_path / "config.pkl")
        print(f"QA model saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "QABaselineModel":
        import joblib
        config = joblib.load(Path(model_dir) / "config.pkl")
        return cls(**config)

    def __repr__(self):
        return f"QABaselineModel(max_features={self.max_features})"
