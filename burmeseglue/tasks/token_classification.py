"""
Token classification (NER) task processor for BurmeseGLUE.
"""
from typing import Any, Dict, List, Optional

from burmeseglue.tasks.base import Task


class TokenClassificationTask(Task):
    """
    Token classification task processor (e.g., NER with BIO tagging).
    """

    def __init__(self, dataset, max_length: int = 128, tokenizer=None):
        super().__init__(dataset, max_length, tokenizer)

    def get_train_examples(self) -> List[Dict]:
        return self.dataset.load_split("train")

    def get_dev_examples(self) -> List[Dict]:
        return self.dataset.load_split("validation")

    def get_test_examples(self) -> List[Dict]:
        return self.dataset.load_split("test")

    def get_labels(self) -> List[str]:
        return self.dataset.get_labels()

    def get_num_labels(self) -> int:
        return len(self.get_labels())

    def get_metrics(self) -> List[str]:
        return ["span_f1", "precision", "recall"]

    def format_for_model(self, examples: List[Dict], stage: str = "train") -> Dict:
        return {
            "token_sequences": [ex["tokens"] for ex in examples],
            "tag_sequences": [ex["tags"] for ex in examples],
        }

    def format_single_example(self, example: Dict) -> Dict:
        return {"tokens": example["tokens"]}

    def decode_predictions(self, predictions: Any, examples: List[Dict]) -> List[Dict]:
        results = []
        for example, pred_tags in zip(examples, predictions):
            results.append({
                "id": example["id"],
                "tokens": example["tokens"],
                "true_tags": example.get("tags", []),
                "pred_tags": pred_tags,
            })
        return results
