"""
Sentence Similarity dataset loader for BurmeseGLUE.

Pairs of Burmese sentences with human-annotated similarity scores (0-5).
"""
import random
from typing import Dict, List, Optional

from burmeseglue.datasets.base import BurmeseGLUEDataset


class SimilarityDataset(BurmeseGLUEDataset):
    """
    Sentence similarity dataset for Burmese text.

    Data format:
    {
        "id": "sim_0001",
        "sentence1": "ဤစာအုပ်သည် ကောင်းမွန်ပါသည်",
        "sentence2": "ဤစာအုပ်မှာ အလွန် ကောင်းပါသည်",
        "score": 4.5   # 0 = completely different, 5 = identical meaning
    }
    """

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
            task_name="similarity",
            cache_dir=cache_dir,
            download=download,
            seed=seed,
        )

    def download(self) -> None:
        print("Generating synthetic similarity data for testing...")
        self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> None:
        random.seed(self.seed)

        # Near-identical pairs (score 4-5)
        high_pairs = [
            ("ဤရုပ်ရှင်သည် အလွန်ကောင်းမွန်ပါသည်", "ဒီဇာတ်ကားမှာ တကယ်ကောင်းတယ်", 4.5),
            ("မြန်မာနိုင်ငံ၏ မြို့တော်မှာ နေပြည်တော် ဖြစ်ပါသည်", "မြန်မာရဲ့ မြို့တော်က နေပြည်တော် ပါ", 4.8),
            ("ကောင်းမွန်သော ဝန်ဆောင်မှု ဖြစ်ပါသည်", "ဝန်ဆောင်မှု ကောင်းပါတယ်", 4.2),
            ("အရသာ အလွန်ကောင်းပါတယ်", "စားရတာ အရမ်းကောင်းတယ်", 4.0),
            ("ဒီစာအုပ်က တကယ့်ကို စိတ်ဝင်စားစရာ ကောင်းတယ်", "ဒီစာအုပ်ဟာ ဖတ်ရတာ ဝင်ဆောင်တယ်", 4.3),
            ("ကျောင်းသားများ ကျောင်းသွားကြသည်", "ကလေးများ ကျောင်းသို့ သွားကြသည်", 4.1),
            ("ဆေးရုံကို လာရောက်ကုသသည်", "ဆေးရုံသို့ ကုသမှုခံယူရန် ရောက်ရှိသည်", 4.6),
        ]

        # Related but different pairs (score 2-3)
        mid_pairs = [
            ("ရုပ်ရှင်ကြည့်သည်", "တီဗွီကြည့်သည်", 2.5),
            ("မြန်မာဘာသာ သင်ကြားသည်", "အင်္ဂလိပ်ဘာသာ လေ့လာသည်", 2.0),
            ("ဆေးရုံသွားသည်", "ဆေးဆိုင်ကို ရောက်သည်", 3.0),
            ("ထမင်းစားသည်", "ရေသောက်သည်", 2.0),
            ("ကားမောင်းသည်", "ဆိုင်ကယ်စီးသည်", 2.5),
            ("ဖတ်စာ ပြင်ဆင်သည်", "စာမေးပွဲ အတွက် ကြိုးစားသည်", 3.2),
            ("မိုးရွာသည်", "မိုး မဲနေသည်", 3.0),
        ]

        # Unrelated pairs (score 0-1)
        low_pairs = [
            ("ကြက်ကောင်ငယ် ကစားသည်", "ကွန်ပျူတာ ပြင်သည်", 0.5),
            ("ပန်းချီပန်းဆွဲသည်", "ရေကူးသည်", 0.8),
            ("ချောင်းဆိုး ဖျားနာသည်", "ကားဝယ်သည်", 0.2),
            ("တောင်တက်သည်", "ငါးဖမ်းသည်", 0.5),
            ("ဂစ်တာတီးသည်", "ထမင်းချက်သည်", 0.3),
            ("ကုန်ပစ္စည်း ဝယ်သည်", "ဘောလုံးကန်သည်", 0.4),
            ("ကဗျာ ရေးသည်", "ကပ်ပြားချိတ်သည်", 0.1),
        ]

        examples = []
        id_counter = 0

        # Oversample to get ~200 examples
        all_templates = high_pairs * 10 + mid_pairs * 10 + low_pairs * 10
        random.shuffle(all_templates)

        for s1, s2, base_score in all_templates[:200]:
            # Add slight noise to scores
            noise = random.uniform(-0.3, 0.3)
            score = max(0.0, min(5.0, base_score + noise))
            examples.append({
                "id": f"sim_{id_counter:04d}",
                "sentence1": self.preprocess_text(s1),
                "sentence2": self.preprocess_text(s2),
                "score": round(score, 2),
            })
            id_counter += 1

        random.shuffle(examples)
        train, val, test = self.create_splits(examples)
        self.save_split(train, "train")
        self.save_split(val, "validation")
        self.save_split(test, "test")

        print(f"Generated {len(examples)} synthetic similarity examples")
        print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    def load_split(self, split: str) -> List[Dict]:
        file_path = self.task_dir / f"{split}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}")
        return self.load_jsonl(str(file_path))

    def get_labels(self) -> List[float]:
        # Regression task — no fixed label set, return score range
        return [0.0, 5.0]
