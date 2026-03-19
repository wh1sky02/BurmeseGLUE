"""
BurmeseGLUE: A standardized evaluation benchmark for Burmese natural language understanding.

This package provides:
- Dataset loaders for Burmese NLU tasks (sentiment, NER, similarity, QA)
- Task processors and metrics
- Baseline and transformer model implementations
- Training and evaluation pipelines
- Reproducible benchmarking tools
"""

__version__ = "0.1.0"
__author__ = "BurmeseGLUE Contributors"

from burmeseglue.datasets import *
from burmeseglue.tasks import *
from burmeseglue.models import *
from burmeseglue.metrics import *

__all__ = [
    "datasets",
    "tasks",
    "models",
    "metrics",
    "training",
    "evaluation",
]
