"""
Sentiment classification dataset loader for BurmeseGLUE.

This loader supports Burmese sentiment datasets with binary or multi-class labels.
For testing, it can generate synthetic data if no real dataset is available.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from burmeseglue.datasets.base import BurmeseGLUEDataset


class SentimentDataset(BurmeseGLUEDataset):
    """
    Sentiment classification dataset for Burmese text.

    Data format:
    {
        "id": "sent_001",
        "text": "ဤရုပ်ရှင်သည် အလွန်ကောင်းမွန်ပါသည်။",
        "label": 1  # 0=negative, 1=positive
    }
    """

    def __init__(
        self,
        data_dir: str,
        num_labels: int = 2,
        use_synthetic: bool = False,
        cache_dir: Optional[str] = None,
        download: bool = True,
        seed: int = 42,
    ):
        """
        Initialize sentiment dataset.

        Args:
            data_dir: Root directory for datasets
            num_labels: Number of sentiment classes (2 for binary, 3+ for multi-class)
            use_synthetic: Use synthetic data for testing
            cache_dir: Directory for caching processed data
            download: Whether to download data if not present
            seed: Random seed for reproducibility
        """
        self.num_labels = num_labels
        self.use_synthetic = use_synthetic
        super().__init__(
            data_dir=data_dir,
            task_name="sentiment",
            cache_dir=cache_dir,
            download=download,
            seed=seed,
        )

    def download(self) -> None:
        """
        Download sentiment dataset.

        For testing, generates synthetic Burmese sentiment data.
        In production, this should download from a real data source.
        """
        if self.use_synthetic:
            print("Generating synthetic sentiment data for testing...")
            self._generate_synthetic_data()
        else:
            # TODO: Replace with actual dataset download
            print("No real dataset configured. Using synthetic data...")
            self.use_synthetic = True
            self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> None:
        """
        Generate synthetic sentiment data for testing.

        Creates balanced dataset with Burmese text samples.
        """
        import random

        # Sample Burmese texts (positive and negative)
        positive_samples = [
            "ဤရုပ်ရှင်သည် အလွန်ကောင်းမွန်ပါသည်။",
            "ကောင်းမွန်သော ဝန်ဆောင်မှု ဖြစ်ပါသည်။",
            "အရသာ အလွန်ကောင်းပါတယ်။",
            "ဒီစာအုပ်က တကယ့်ကို စိတ်ဝင်စားစရာ ကောင်းတယ်။",
            "ကျွန်တော် ဒီထုတ်ကုန်ကို အကြိုက်ဆုံးပါ။",
            "အရည်အသွေး အလွန်မြင့်မားပါသည်။",
            "ဝန်ထမ်းတွေ အလွန် ဖြူစင်ပါတယ်။",
            "စျေးနှုန်းလည်း သင့်တော်ပါတယ်။",
            "ဒီနေရာက တကယ် လှပါတယ်။",
            "အလွန် ကောင်းမွန်တဲ့ အတွေ့အကြုံ ဖြစ်ခဲ့ပါတယ်။",
        ]

        negative_samples = [
            "ဤထုတ်ကုန်သည် အရည်အသွေး ညံ့ဖျင်းပါသည်။",
            "ဝန်ဆောင်မှု အလွန် ညံ့ပါတယ်။",
            "စျေးနှုန်း အလွန်မြင့်လွန်းပါတယ်။",
            "အရသာ မကောင်းပါဘူး။",
            "ကျွန်တော် စိတ်ပျက်ခဲ့ပါတယ်။",
            "အရည်အသွေး မျှော်လင့်ချက်နဲ့ မကိုက်ညီပါဘူး။",
            "ဝန်ထမ်းတွေ မဖြူစင်ပါဘူး။",
            "အချိန်ကုန်တာ ဖြစ်ခဲ့ပါတယ်။",
            "အလွန် စိတ်ထိခိုက်ဖွယ် ဖြစ်ခဲ့ပါတယ်။",
            "နောက်တစ်ခါ မသွားတော့ပါဘူး။",
        ]

        # Generate examples
        examples = []
        id_counter = 0

        # Generate 200 examples (100 per class for binary)
        samples_per_class = 100

        for label in range(self.num_labels):
            if label == 0:
                source = negative_samples
            elif label == 1:
                source = positive_samples
            else:
                # For multi-class, duplicate some samples
                source = positive_samples if label % 2 == 1 else negative_samples

            for i in range(samples_per_class):
                text = random.choice(source)
                examples.append({
                    "id": f"sent_{id_counter:04d}",
                    "text": self.preprocess_text(text),
                    "label": label,
                })
                id_counter += 1

        # Shuffle examples
        random.shuffle(examples)

        # Create splits (70/15/15)
        train, val, test = self.create_splits(examples)

        # Save splits
        self.save_split(train, "train")
        self.save_split(val, "validation")
        self.save_split(test, "test")

        print(f"Generated {len(examples)} synthetic sentiment examples")
        print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    def load_split(self, split: str) -> List[Dict]:
        """
        Load a specific data split.

        Args:
            split: One of "train", "validation", "test"

        Returns:
            List of examples as dictionaries
        """
        file_path = self.task_dir / f"{split}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}")

        return self.load_jsonl(str(file_path))

    def get_labels(self) -> List[int]:
        """
        Get label schema for sentiment classification.

        Returns:
            List of label indices (e.g., [0, 1] for binary)
        """
        return list(range(self.num_labels))

    def get_label_names(self) -> List[str]:
        """
        Get human-readable label names.

        Returns:
            List of label names
        """
        if self.num_labels == 2:
            return ["negative", "positive"]
        elif self.num_labels == 3:
            return ["negative", "neutral", "positive"]
        else:
            return [f"class_{i}" for i in range(self.num_labels)]
