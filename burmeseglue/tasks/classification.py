"""
Classification task processor for BurmeseGLUE.

Handles sentiment classification and other text classification tasks.
"""
import torch
from typing import Any, Dict, List, Optional

from burmeseglue.tasks.base import Task


class ClassificationTask(Task):
    """
    Text classification task processor.

    Supports binary and multi-class classification.
    """

    def __init__(
        self,
        dataset,
        max_length: int = 128,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize classification task.

        Args:
            dataset: Dataset instance (e.g., SentimentDataset)
            max_length: Maximum sequence length
            tokenizer: Tokenizer for encoding text
        """
        super().__init__(dataset, max_length, tokenizer)

    def get_train_examples(self) -> List[Dict]:
        """Get training examples."""
        return self.dataset.load_split("train")

    def get_dev_examples(self) -> List[Dict]:
        """Get validation examples."""
        return self.dataset.load_split("validation")

    def get_test_examples(self) -> List[Dict]:
        """Get test examples."""
        return self.dataset.load_split("test")

    def get_labels(self) -> List:
        """Get label schema."""
        return self.dataset.get_labels()

    def get_metrics(self) -> List[str]:
        """Get evaluation metrics for classification."""
        return ["accuracy", "f1", "precision", "recall"]

    def format_for_model(
        self,
        examples: List[Dict],
        stage: str = "train"
    ) -> Dict[str, Any]:
        """
        Format examples for model input.

        Args:
            examples: Raw examples from dataset
            stage: One of "train", "eval", "test"

        Returns:
            Dictionary with model inputs
        """
        if self.tokenizer is None:
            # For baseline models without tokenizer
            return {
                "texts": [ex["text"] for ex in examples],
                "labels": [ex["label"] for ex in examples],
            }

        # For transformer models
        texts = [ex["text"] for ex in examples]
        labels = [ex["label"] for ex in examples]

        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Add labels
        encodings["labels"] = torch.tensor(labels)

        return encodings

    def format_single_example(self, example: Dict) -> Dict[str, Any]:
        """
        Format a single example for inference.

        Args:
            example: Single example dictionary

        Returns:
            Formatted inputs
        """
        if self.tokenizer is None:
            return {"text": example["text"]}

        encoding = self.encode_text(example["text"])
        return encoding

    def decode_predictions(
        self,
        predictions: Any,
        examples: List[Dict]
    ) -> List[Dict]:
        """
        Decode model predictions to human-readable format.

        Args:
            predictions: Model predictions (logits or class indices)
            examples: Original examples

        Returns:
            List of dictionaries with predictions
        """
        import numpy as np

        # Handle different prediction formats
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Logits: take argmax
            pred_labels = np.argmax(predictions, axis=1)
        else:
            # Already class indices
            pred_labels = predictions.flatten()

        # Get label names if available
        if hasattr(self.dataset, "get_label_names"):
            label_names = self.dataset.get_label_names()
        else:
            label_names = [str(i) for i in self.get_labels()]

        results = []
        for i, (example, pred_label) in enumerate(zip(examples, pred_labels)):
            result = {
                "id": example["id"],
                "text": example["text"],
                "true_label": example.get("label", None),
                "pred_label": int(pred_label),
                "pred_label_name": label_names[int(pred_label)],
            }

            # Add true label name if label is present
            if "label" in example:
                result["true_label_name"] = label_names[example["label"]]

            results.append(result)

        return results
