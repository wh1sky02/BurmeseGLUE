"""
Span extraction (QA) task processor for BurmeseGLUE.
"""
from typing import Any, Dict, List, Optional

from burmeseglue.tasks.base import Task


class SpanExtractionTask(Task):
    """
    Extractive question answering task processor.
    """

    def __init__(self, dataset, max_length: int = 512, tokenizer=None):
        super().__init__(dataset, max_length, tokenizer)

    def get_train_examples(self) -> List[Dict]:
        return self.dataset.load_split("train")

    def get_dev_examples(self) -> List[Dict]:
        return self.dataset.load_split("validation")

    def get_test_examples(self) -> List[Dict]:
        return self.dataset.load_split("test")

    def get_labels(self) -> List:
        return []

    def get_metrics(self) -> List[str]:
        return ["exact_match", "f1"]

    def format_for_model(self, examples: List[Dict], stage: str = "train") -> Dict:
        return {
            "contexts": [ex["context"] for ex in examples],
            "questions": [ex["question"] for ex in examples],
            "answers": [ex["answers"] for ex in examples],
        }

    def format_single_example(self, example: Dict) -> Dict:
        return {
            "context": example["context"],
            "question": example["question"],
        }

    def decode_predictions(self, predictions: Any, examples: List[Dict]) -> List[Dict]:
        results = []
        for example, pred_answer in zip(examples, predictions):
            results.append({
                "id": example["id"],
                "context": example["context"],
                "question": example["question"],
                "true_answers": example.get("answers", {}).get("text", []),
                "pred_answer": pred_answer,
            })
        return results
