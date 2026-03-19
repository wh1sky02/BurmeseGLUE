"""
Dataset loaders for BurmeseGLUE tasks.

Available datasets:
- SentimentDataset: Sentiment classification
- NERDataset: Named entity recognition
- SimilarityDataset: Sentence similarity
- QADataset: Question answering
"""

from burmeseglue.datasets.base import BurmeseGLUEDataset

# Task-specific datasets will be imported as they're implemented
# from burmeseglue.datasets.sentiment import SentimentDataset
# from burmeseglue.datasets.ner import NERDataset
# from burmeseglue.datasets.similarity import SimilarityDataset
# from burmeseglue.datasets.qa import QADataset

__all__ = [
    "BurmeseGLUEDataset",
    # "SentimentDataset",
    # "NERDataset",
    # "SimilarityDataset",
    # "QADataset",
]
