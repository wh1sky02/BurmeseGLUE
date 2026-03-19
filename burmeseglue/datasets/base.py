"""
Base dataset class for BurmeseGLUE datasets.

All task-specific dataset loaders inherit from this base class.
"""
import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
from tqdm import tqdm


class BurmeseGLUEDataset(ABC):
    """
    Abstract base class for BurmeseGLUE datasets.

    All task-specific datasets should inherit from this class and implement:
    - download(): Download raw data from source
    - load_split(split): Load train/val/test split
    - get_labels(): Return label schema
    - preprocess(): Burmese text normalization
    """

    def __init__(
        self,
        data_dir: str,
        task_name: str,
        cache_dir: Optional[str] = None,
        download: bool = True,
        seed: int = 42,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory for datasets
            task_name: Name of the task (e.g., "sentiment", "ner")
            cache_dir: Directory for caching processed data
            download: Whether to download data if not present
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.task_name = task_name
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.seed = seed

        # Create directories
        self.task_dir = self.data_dir / task_name
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

        # Download data if needed
        if download and not self._is_downloaded():
            print(f"Downloading {task_name} dataset...")
            self.download()
            print(f"Download complete!")

    @abstractmethod
    def download(self) -> None:
        """
        Download raw data from source.

        Implementation should:
        1. Download data files
        2. Extract if needed
        3. Save to self.task_dir
        """
        pass

    @abstractmethod
    def load_split(self, split: str) -> List[Dict]:
        """
        Load a specific data split.

        Args:
            split: One of "train", "validation", "test"

        Returns:
            List of examples as dictionaries
        """
        pass

    @abstractmethod
    def get_labels(self) -> List:
        """
        Get label schema for this dataset.

        Returns:
            List of labels (e.g., [0, 1] for binary classification)
        """
        pass

    def preprocess_text(self, text: str) -> str:
        """
        Normalize Burmese text.

        Basic preprocessing for Burmese unicode text:
        - Strip whitespace
        - Normalize unicode (NFD to NFC)
        - Remove zero-width characters

        Args:
            text: Raw text string

        Returns:
            Preprocessed text
        """
        import unicodedata

        # Strip whitespace
        text = text.strip()

        # Normalize unicode to NFC form
        text = unicodedata.normalize("NFC", text)

        # Remove zero-width characters
        text = text.replace("\u200b", "")  # Zero-width space
        text = text.replace("\ufeff", "")  # Zero-width no-break space

        return text

    def create_splits(
        self,
        examples: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create train/val/test splits from examples.

        Args:
            examples: List of examples
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing

        Returns:
            Tuple of (train, val, test) splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        # Shuffle examples
        random.shuffle(examples)

        # Calculate split indices
        n = len(examples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = examples[:train_end]
        val = examples[train_end:val_end]
        test = examples[val_end:]

        return train, val, test

    def save_split(self, examples: List[Dict], split: str) -> None:
        """
        Save a split to JSON Lines format.

        Args:
            examples: List of examples
            split: Split name ("train", "validation", "test")
        """
        output_path = self.task_dir / f"{split}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Saved {len(examples)} examples to {output_path}")

    def load_jsonl(self, file_path: str) -> List[Dict]:
        """
        Load data from JSON Lines file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of examples
        """
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))
        return examples

    def _is_downloaded(self) -> bool:
        """
        Check if dataset has been downloaded.

        Returns:
            True if all split files exist
        """
        required_files = ["train.jsonl", "validation.jsonl", "test.jsonl"]
        return all((self.task_dir / f).exists() for f in required_files)

    def download_file(
        self,
        url: str,
        output_path: str,
        description: Optional[str] = None
    ) -> None:
        """
        Download a file with progress bar.

        Args:
            url: URL to download from
            output_path: Local path to save to
            description: Description for progress bar
        """
        class TqdmUpTo(tqdm):
            def update_to(self, blocks=1, block_size=1, total_size=None):
                if total_size is not None:
                    self.total = total_size
                self.update(blocks * block_size - self.n)

        with TqdmUpTo(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=description or "Downloading",
        ) as t:
            urlretrieve(url, output_path, reporthook=t.update_to)

    def get_statistics(self, split: str) -> Dict:
        """
        Get statistics for a data split.

        Args:
            split: Split name

        Returns:
            Dictionary with statistics
        """
        examples = self.load_split(split)

        stats = {
            "num_examples": len(examples),
            "task": self.task_name,
            "split": split,
        }

        # Add task-specific statistics
        if examples:
            stats["example_keys"] = list(examples[0].keys())

        return stats

    def __len__(self) -> int:
        """Return total number of examples across all splits."""
        try:
            train = self.load_split("train")
            val = self.load_split("validation")
            test = self.load_split("test")
            return len(train) + len(val) + len(test)
        except Exception:
            return 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task_name}, data_dir={self.data_dir})"
