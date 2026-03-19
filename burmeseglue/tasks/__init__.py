"""
Task processors for BurmeseGLUE.

Available tasks:
- ClassificationTask: Sentiment classification
- TokenClassificationTask: Named entity recognition
- RegressionTask: Sentence similarity
- SpanExtractionTask: Question answering
"""

from burmeseglue.tasks.base import Task
from burmeseglue.tasks.classification import ClassificationTask

# Task-specific implementations will be imported as they're implemented
# from burmeseglue.tasks.token_classification import TokenClassificationTask
# from burmeseglue.tasks.regression import RegressionTask
# from burmeseglue.tasks.span_extraction import SpanExtractionTask

__all__ = [
    "Task",
    "ClassificationTask",
    # "TokenClassificationTask",
    # "RegressionTask",
    # "SpanExtractionTask",
]
