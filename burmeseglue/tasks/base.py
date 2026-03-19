"""
Base task class for BurmeseGLUE tasks.

All task-specific processors inherit from this base class.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from transformers import PreTrainedTokenizer


class Task(ABC):
    """
    Abstract base class for BurmeseGLUE tasks.

    Each task defines:
    - How to load examples from datasets
    - How to format data for models
    - Which metrics to use for evaluation
    """

    def __init__(
        self,
        dataset,
        max_length: int = 128,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """
        Initialize task.

        Args:
            dataset: Dataset instance for this task
            max_length: Maximum sequence length for tokenization
            tokenizer: Tokenizer for encoding text (optional for baseline models)
        """
        self.dataset = dataset
        self.max_length = max_length
        self.tokenizer = tokenizer

    @abstractmethod
    def get_train_examples(self) -> List[Dict]:
        """
        Get training examples.

        Returns:
            List of training examples
        """
        pass

    @abstractmethod
    def get_dev_examples(self) -> List[Dict]:
        """
        Get development/validation examples.

        Returns:
            List of validation examples
        """
        pass

    @abstractmethod
    def get_test_examples(self) -> List[Dict]:
        """
        Get test examples.

        Returns:
            List of test examples
        """
        pass

    @abstractmethod
    def get_labels(self) -> List:
        """
        Get label schema.

        Returns:
            List of possible labels
        """
        pass

    @abstractmethod
    def get_metrics(self) -> List[str]:
        """
        Get metrics for evaluation.

        Returns:
            List of metric names (e.g., ["accuracy", "f1"])
        """
        pass

    @abstractmethod
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
            Dictionary with model inputs (e.g., input_ids, attention_mask, labels)
        """
        pass

    def get_num_labels(self) -> int:
        """
        Get number of labels.

        Returns:
            Number of unique labels
        """
        labels = self.get_labels()
        return len(labels) if isinstance(labels, list) else 1

    def encode_text(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Encode text using tokenizer.

        Args:
            text: Primary text to encode
            text_pair: Optional second text (for sentence pairs)
            max_length: Override default max length

        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Initialize task with a tokenizer.")

        max_len = max_length or self.max_length

        encoding = self.tokenizer(
            text,
            text_pair=text_pair,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {k: v.squeeze(0) for k, v in encoding.items()}

    def get_data_collator(self) -> Optional[Callable]:
        """
        Get data collator for batching.

        Returns:
            Data collator function or None for default behavior
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset.task_name}, max_length={self.max_length})"
