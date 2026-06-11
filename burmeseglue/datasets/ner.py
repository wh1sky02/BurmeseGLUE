"""
Named Entity Recognition dataset loader for BurmeseGLUE.

Supports BIO-tagged sequences with PER, ORG, LOC entity types.
Generates synthetic Burmese NER data for testing.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from burmeseglue.datasets.base import BurmeseGLUEDataset


class NERDataset(BurmeseGLUEDataset):
    """
    Named Entity Recognition dataset for Burmese text.

    Data format:
    {
        "id": "ner_0001",
        "tokens": ["သမ္မတ", "အောင်ဆန်းစုကြည်", "မြန်မာ", "နိုင်ငံ", "မှ", "ဖြစ်", "သည်"],
        "tags":   ["O",    "B-PER",             "B-LOC",  "I-LOC",  "O",  "O",    "O"]
    }
    """

    LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def __init__(
        self,
        data_dir: str,
        use_synthetic: bool = False,
        cache_dir: Optional[str] = None,
        download: bool = True,
        seed: int = 42,
    ):
        self.use_synthetic = use_synthetic
        super().__init__(
            data_dir=data_dir,
            task_name="ner",
            cache_dir=cache_dir,
            download=download,
            seed=seed,
        )

    def download(self) -> None:
        print("Generating synthetic NER data for testing...")
        self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> None:
        random.seed(self.seed)

        persons = [
            ["အောင်ဆန်းစုကြည်"],
            ["ဦးနုပါ"],
            ["မင်းအောင်လှိုင်"],
            ["ကျော်ဇင်"],
            ["သူရိန်"],
            ["နန်းခင်ထားမူ"],
            ["ဒေါ်အောင်ဆန်းစုကြည်"],
        ]

        orgs = [
            ["အမျိုးသား", "ဒီမိုကရေစီ", "အဖွဲ့ချုပ်"],
            ["မြန်မာ", "အမျိုးသား", "တပ်မတော်"],
            ["ကုလသမဂ္ဂ"],
            ["ASEAN", "အဖွဲ့"],
            ["ဗဟို", "ဘဏ်"],
        ]

        locations = [
            ["နေပြည်တော်"],
            ["ရန်ကုန်", "မြို့"],
            ["မန္တလေး"],
            ["မြန်မာ", "နိုင်ငံ"],
            ["ဧရာဝတီ", "တိုင်းဒေသကြီး"],
        ]

        sentence_templates = [
            ("O_words", "PER", "O_words"),
            ("O_words", "ORG", "O_words"),
            ("O_words", "LOC", "O_words"),
            ("O_words", "PER", "O_words", "ORG", "O_words"),
            ("O_words", "PER", "O_words", "LOC", "O_words"),
            ("O_words", "ORG", "O_words", "LOC", "O_words"),
        ]

        o_words_pool = [
            ["က", "ဆိုင်ရာ"],
            ["မှ", "ထွက်ပြေး"],
            ["သည်", "ဖြစ်သည်"],
            ["တွင်", "ရောက်ရှိ"],
            ["၏", "ခေါင်းဆောင်"],
            ["အား", "ဖိတ်ကြား"],
            ["နှင့်", "တွေ့ဆုံ"],
            ["ကို", "ကြေငြာ"],
        ]

        examples = []
        id_counter = 0

        for _ in range(300):
            template = random.choice(sentence_templates)
            tokens = []
            tags = []

            for part in template:
                if part == "O_words":
                    words = random.choice(o_words_pool)
                    tokens.extend(words)
                    tags.extend(["O"] * len(words))
                elif part == "PER":
                    person = random.choice(persons)
                    tokens.extend(person)
                    tags.append("B-PER")
                    tags.extend(["I-PER"] * (len(person) - 1))
                elif part == "ORG":
                    org = random.choice(orgs)
                    tokens.extend(org)
                    tags.append("B-ORG")
                    tags.extend(["I-ORG"] * (len(org) - 1))
                elif part == "LOC":
                    loc = random.choice(locations)
                    tokens.extend(loc)
                    tags.append("B-LOC")
                    tags.extend(["I-LOC"] * (len(loc) - 1))

            examples.append({
                "id": f"ner_{id_counter:04d}",
                "tokens": tokens,
                "tags": tags,
            })
            id_counter += 1

        random.shuffle(examples)
        train, val, test = self.create_splits(examples)
        self.save_split(train, "train")
        self.save_split(val, "validation")
        self.save_split(test, "test")

        print(f"Generated {len(examples)} synthetic NER examples")
        print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    def load_split(self, split: str) -> List[Dict]:
        file_path = self.task_dir / f"{split}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}")
        return self.load_jsonl(str(file_path))

    def get_labels(self) -> List[str]:
        return self.LABEL_LIST

    def get_label_names(self) -> List[str]:
        return self.LABEL_LIST
