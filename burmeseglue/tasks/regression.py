"""
Regression task processor for BurmeseGLUE sentence similarity.
"""
from typing import Any, Dict, List, Optional

from burmeseglue.tasks.base import Task


class RegressionTask(Task):
    """
    Sentence similarity (regression) task processor.
    """

    def __init__(self, dataset, max_length: int = 128, tokenizer=None):
        super().__init__(dataset, max_length, tokenizer)

    def get_train_examples(self) -> List[Dict]:
        return self.dataset.load_split("train")

    def get_dev_examples(self) -> List[Dict]:
        return self.dataset.load_split("validation")

    def get_test_examples(self) -> List[Dict]:
        return self.dataset.load_split("test")

    def get_labels(self) -> List:
        return self.dataset.get_labels()

    def get_metrics(self) -> List[str]:
        return ["pearson", "spearman", "mse", "rmse"]

    def format_for_model(self, examples: List[Dict], stage: str = "train") -> Dict:
        return {
            "sentences1": [ex["sentence1"] for ex in examples],
            "sentences2": [ex["sentence2"] for ex in examples],
            "scores": [ex["score"] for ex in examples],
        }

    def format_single_example(self, example: Dict) -> Dict:
        return {
            "sentence1": example["sentence1"],
            "sentence2": example["sentence2"],
        }

    def decode_predictions(self, predictions: Any, examples: List[Dict]) -> List[Dict]:
        results = []
        for example, pred_score in zip(examples, predictions):
            results.append({
                "id": example["id"],
                "sentence1": example["sentence1"],
                "sentence2": example["sentence2"],
                "true_score": example.get("score"),
                "pred_score": float(pred_score),
            })
        return results
