"""
Question Answering dataset loader for BurmeseGLUE.

Extractive QA: given context + question, find the answer span in the context.
"""
import random
from typing import Dict, List, Optional

from burmeseglue.datasets.base import BurmeseGLUEDataset


class QADataset(BurmeseGLUEDataset):
    """
    Extractive Question Answering dataset for Burmese text.

    Data format:
    {
        "id": "qa_0001",
        "context": "မြန်မာနိုင်ငံ၏ မြို့တော်မှာ နေပြည်တော် ဖြစ်ပါသည်",
        "question": "မြန်မာနိုင်ငံ၏ မြို့တော်က ဘယ်မြို့လဲ",
        "answers": {
            "text": ["နေပြည်တော်"],
            "answer_start": [14]
        }
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
            task_name="qa",
            cache_dir=cache_dir,
            download=download,
            seed=seed,
        )

    def download(self) -> None:
        print("Generating synthetic QA data for testing...")
        self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> None:
        random.seed(self.seed)

        # (context, question, answer) triples
        qa_templates = [
            (
                "မြန်မာနိုင်ငံ၏ မြို့တော်မှာ နေပြည်တော် ဖြစ်ပါသည်။ နေပြည်တော်ကို ၂၀၀၅ ခုနှစ်တွင် မြို့တော်အဖြစ် သတ်မှတ်ခဲ့သည်။",
                "မြန်မာနိုင်ငံ၏ မြို့တော်က ဘာလဲ",
                "နေပြည်တော်",
            ),
            (
                "ရန်ကုန်မြို့သည် မြန်မာနိုင်ငံ၏ စီးပွားရေး မြို့တော်ကြီး ဖြစ်သည်။ လူဦးရေ သန်း ၅ ဦးကျော် နေထိုင်သည်။",
                "ရန်ကုန်မြို့တွင် လူဦးရေ ဘယ်လောက်ရှိသလဲ",
                "သန်း ၅ ဦးကျော်",
            ),
            (
                "မြန်မာနိုင်ငံတွင် ရာသီဥတု သုံးမျိုး ရှိသည်။ မိုးရာသီ၊ ဆောင်းရာသီနှင့် နွေရာသီ ဖြစ်သည်။",
                "မြန်မာနိုင်ငံတွင် ရာသီဥတု ဘယ်နှမျိုး ရှိသလဲ",
                "သုံးမျိုး",
            ),
            (
                "ဧရာဝတီမြစ်သည် မြန်မာနိုင်ငံ၏ အဓိက မြစ်ကြီး ဖြစ်သည်။ မြစ်ရှည် မိုင် ၁,၃၅၀ ရှိသည်။",
                "ဧရာဝတီမြစ်ရဲ့ အရှည်ကဘယ်လောက်လဲ",
                "မိုင် ၁,၃၅၀",
            ),
            (
                "မြန်မာနိုင်ငံ လွတ်လပ်ရေးကို ၁၉၄၈ ခုနှစ် ဇန်နဝါရီ ၄ ရက်နေ့တွင် ရရှိခဲ့သည်။ ဦးနုပါ ပထမဆုံး ဝန်ကြီးချုပ် ဖြစ်သည်။",
                "မြန်မာနိုင်ငံ လွတ်လပ်ရေး ဘယ်နှစ်တွင် ရခဲ့သလဲ",
                "၁၉၄၈ ခုနှစ်",
            ),
            (
                "ဘာသာပေါင်းစုံ ရွှင်ပျသောနှစ်ဦးစုံတို့ ပျော်ရွှင်ကြ၏။ မြန်မာနိုင်ငံတွင် ဘာသာ ၁၃၅ မျိုး ရှိသည်။",
                "မြန်မာနိုင်ငံတွင် ဘာသာ ဘယ်နှမျိုး ရှိသလဲ",
                "၁၃၅ မျိုး",
            ),
            (
                "မြန်မာနိုင်ငံ၏ ငွေကြေး ယူနစ်မှာ ကျပ် ဖြစ်သည်။ မြန်မာ ငွေကြေးနှင့် ဘဏ္ဍာရေး ဝန်ဆောင်မှုများ ဗဟိုဘဏ်မှ ထိန်းချုပ်သည်။",
                "မြန်မာနိုင်ငံ၏ ငွေကြေး ယူနစ်မှာ ဘာလဲ",
                "ကျပ်",
            ),
            (
                "သရက်ရည်သည် မြန်မာ့ ရိုးရာ အဖျော်ယမကာ တစ်မျိုး ဖြစ်သည်။ သရက်သီး စစ်ကာ ကြပ်ကာ ပြုလုပ်သည်။",
                "သရက်ရည်ကို ဘာနဲ့ ပြုလုပ်သလဲ",
                "သရက်သီး",
            ),
            (
                "နောင်တောင်းမြို့သည် ရှမ်းပြည်နယ်တွင် တည်ရှိသည်။ မြို့ကလေးသည် တောင်ကုန်းများ ဝန်းရံသော နေရာ ဖြစ်သည်။",
                "နောင်တောင်းမြို့က ဘယ်ပြည်နယ်မှာ တည်ရှိသလဲ",
                "ရှမ်းပြည်နယ်",
            ),
            (
                "မြန်မာ့ ရိုးရာ ကဒ်ဂိမ်းတွင် မြင်းဂေါက်ကဒ် ဘောင်းဘီဆတ်ကဒ် ကြက်ကဒ် စသော မျိုးစိတ်များ ပါဝင်သည်။",
                "မြန်မာ့ ရိုးရာ ကဒ်ဂိမ်းတွင် ဘယ်လို ကဒ်မျိုးစိတ်များ ပါဝင်သလဲ",
                "မြင်းဂေါက်ကဒ် ဘောင်းဘီဆတ်ကဒ် ကြက်ကဒ်",
            ),
        ]

        examples = []
        id_counter = 0

        # Oversample to ~200 examples
        extended = qa_templates * 20
        random.shuffle(extended)

        for context, question, answer in extended[:200]:
            # Find answer start position in context
            answer_start = context.find(answer)
            if answer_start == -1:
                # Skip if answer not found literally (shouldn't happen with our templates)
                continue

            examples.append({
                "id": f"qa_{id_counter:04d}",
                "context": self.preprocess_text(context),
                "question": self.preprocess_text(question),
                "answers": {
                    "text": [answer],
                    "answer_start": [answer_start],
                },
            })
            id_counter += 1

        random.shuffle(examples)
        train, val, test = self.create_splits(examples)
        self.save_split(train, "train")
        self.save_split(val, "validation")
        self.save_split(test, "test")

        print(f"Generated {len(examples)} synthetic QA examples")
        print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    def load_split(self, split: str) -> List[Dict]:
        file_path = self.task_dir / f"{split}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}")
        return self.load_jsonl(str(file_path))

    def get_labels(self) -> List:
        return []  # Extractive QA has no fixed label set
