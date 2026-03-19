"""
Transformer model wrapper for BurmeseGLUE.

Provides a simple interface for fine-tuning HuggingFace transformers.
"""
import torch
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from typing import Any, Dict, List, Optional, Union
import numpy as np


class TransformerClassifier:
    """
    Wrapper for transformer-based sequence classification.

    Supports any HuggingFace model (mBERT, XLM-RoBERTa, etc.).
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_labels: int = 2,
        max_length: int = 128,
        device: Optional[str] = None,
    ):
        """
        Initialize transformer classifier.

        Args:
            model_name: HuggingFace model name
            num_labels: Number of classification labels
            max_length: Maximum sequence length
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.model.to(self.device)

        self.trainer = None

    def prepare_dataset(
        self,
        texts: List[str],
        labels: Optional[Union[List[int], np.ndarray]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare dataset for training/evaluation.

        Args:
            texts: List of text strings
            labels: Optional labels

        Returns:
            Dictionary with encoded inputs
        """
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if labels is not None:
            encodings["labels"] = torch.tensor(labels)

        return encodings

    def train(
        self,
        train_texts: List[str],
        train_labels: Union[List[int], np.ndarray],
        eval_texts: Optional[List[str]] = None,
        eval_labels: Optional[Union[List[int], np.ndarray]] = None,
        output_dir: str = "./outputs",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        eval_steps: int = 500,
        save_steps: int = 500,
        logging_steps: int = 100,
        early_stopping_patience: int = 3,
        compute_metrics_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_texts: Training texts
            train_labels: Training labels
            eval_texts: Validation texts
            eval_labels: Validation labels
            output_dir: Directory for outputs
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            eval_steps: Evaluation frequency
            save_steps: Checkpoint frequency
            logging_steps: Logging frequency
            early_stopping_patience: Early stopping patience
            compute_metrics_fn: Function to compute metrics

        Returns:
            Training results
        """
        from torch.utils.data import Dataset as TorchDataset

        # Prepare datasets
        class SimpleDataset(TorchDataset):
            def __init__(self, encodings):
                self.encodings = encodings

            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}

            def __len__(self):
                return len(self.encodings["input_ids"])

        train_encodings = self.prepare_dataset(train_texts, train_labels)
        train_dataset = SimpleDataset(train_encodings)

        eval_dataset = None
        if eval_texts is not None and eval_labels is not None:
            eval_encodings = self.prepare_dataset(eval_texts, eval_labels)
            eval_dataset = SimpleDataset(eval_encodings)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=eval_steps if eval_dataset else None,
            save_strategy="steps",
            save_steps=save_steps,
            logging_steps=logging_steps,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            save_total_limit=3,
            report_to=[],
        )

        # Metrics function
        if compute_metrics_fn is None:
            from burmeseglue.metrics import compute_metrics_for_transformers
            compute_metrics_fn = compute_metrics_for_transformers

        # Create trainer
        callbacks = []
        if eval_dataset and early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_fn,
            callbacks=callbacks,
        )

        # Train
        print(f"Training {self.model_name} for {num_epochs} epochs...")
        train_result = self.trainer.train()

        return train_result

    def predict(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Make predictions on texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for inference

        Returns:
            Predictions (logits)
        """
        self.model.eval()

        all_logits = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encodings = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**encodings)
                logits = outputs.logits.cpu().numpy()
                all_logits.append(logits)

        return np.vstack(all_logits)

    def evaluate(
        self,
        texts: List[str],
        labels: Union[List[int], np.ndarray],
        compute_metrics_fn: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            texts: List of text strings
            labels: True labels
            compute_metrics_fn: Function to compute metrics

        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(texts)

        if compute_metrics_fn is None:
            from burmeseglue.metrics import compute_classification_metrics
            compute_metrics_fn = compute_classification_metrics

        metrics = compute_metrics_fn(predictions, labels)

        return metrics

    def save(self, output_dir: str) -> None:
        """
        Save model and tokenizer.

        Args:
            output_dir: Directory to save model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        print(f"Model saved to {output_dir}")

    @classmethod
    def load(
        cls,
        model_dir: str,
        device: Optional[str] = None,
    ) -> "TransformerClassifier":
        """
        Load model from directory.

        Args:
            model_dir: Directory containing saved model
            device: Device to use

        Returns:
            Loaded model instance
        """
        model_path = Path(model_dir)

        # Load config to get num_labels
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)

        # Create instance
        instance = cls(
            model_name=str(model_path),
            num_labels=config.num_labels,
            device=device,
        )

        print(f"Model loaded from {model_dir}")

        return instance

    def __repr__(self) -> str:
        return f"TransformerClassifier(model={self.model_name}, num_labels={self.num_labels}, device={self.device})"
