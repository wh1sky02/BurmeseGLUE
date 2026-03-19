# BurmeseGLUE

**A Standardized Evaluation Benchmark for Burmese Natural Language Understanding**

BurmeseGLUE is a multi-task benchmark for evaluating Burmese NLP models across diverse natural language understanding tasks. Inspired by the English GLUE benchmark, BurmeseGLUE provides a unified framework for reproducible evaluation of Burmese language models.

## Overview

BurmeseGLUE aggregates multiple curated Burmese datasets into a single benchmark with:

- **4 Core NLU Tasks**: Sentiment Classification, Named Entity Recognition, Sentence Similarity, Question Answering
- **Standardized Data Format**: Consistent JSON Lines format with fixed train/val/test splits (70/15/15)
- **Task-Specific Metrics**: Accuracy/F1 for classification, span-level F1 for NER/QA, Pearson/Spearman for similarity
- **Baseline Implementations**: TF-IDF + linear models and transformer-based models (mBERT, XLM-RoBERTa)
- **Evaluation Pipeline**: Reproducible evaluation with unified reporting

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/burmeseglue.git
cd burmeseglue

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Download Datasets

```bash
# Download all datasets
python scripts/download_data.py --tasks all --data_dir ./data

# Or download specific tasks
python scripts/download_data.py --tasks sentiment,ner --data_dir ./data
```

### Train a Model

```bash
# Train a baseline model
python scripts/train.py \
  --task sentiment \
  --model baseline \
  --output_dir ./runs/sentiment_baseline

# Fine-tune a transformer model
python scripts/train.py \
  --task sentiment \
  --model xlm-roberta-base \
  --config configs/transformer.yaml \
  --output_dir ./runs/sentiment_xlmr \
  --epochs 5
```

### Evaluate a Model

```bash
# Evaluate on test set
python scripts/evaluate.py \
  --task sentiment \
  --model_path ./runs/sentiment_xlmr \
  --split test

# Output: metrics (accuracy, F1, etc.) and predictions
```

### Run Full Benchmark

```bash
# Evaluate model on all tasks
python scripts/benchmark.py \
  --model xlm-roberta-base \
  --output benchmark_results.json

# View aggregated scores across all tasks
```

## Tasks

### 1. Sentiment Classification

Binary or multi-class sentiment classification of Burmese text.

**Example:**
```json
{
  "id": "sent_001",
  "text": "ဤရုပ်ရှင်သည် အလွန်ကောင်းမွန်ပါသည်။",
  "label": 1
}
```

**Metrics:** Accuracy, Macro F1, Precision, Recall

### 2. Named Entity Recognition (NER)

Token-level sequence labeling for named entities in Burmese text.

**Example:**
```json
{
  "id": "ner_001",
  "tokens": ["မြန်မာနိုင်ငံ", "တွင်", "..."],
  "ner_tags": [1, 0, 0]
}
```

**Metrics:** Span-level F1, Precision, Recall (using seqeval)

### 3. Sentence Similarity

Regression task to predict semantic similarity between sentence pairs.

**Example:**
```json
{
  "id": "sim_001",
  "text1": "ဤနေ့သည် ရာသီဥတုကောင်းသည်။",
  "text2": "ယနေ့ရာသီဥတုမှာ ကောင်းမွန်ပါတယ်။",
  "score": 4.5
}
```

**Metrics:** Pearson Correlation, Spearman Correlation

### 4. Question Answering

Extractive QA where the answer is a span within the context.

**Example:**
```json
{
  "id": "qa_001",
  "context": "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။",
  "question": "မြန်မာနိုင်ငံသည် ဘယ်နေရာတွင် ရှိသနည်း။",
  "answers": {
    "text": ["အရှေ့တောင်အာရှတွင်"],
    "answer_start": [18]
  }
}
```

**Metrics:** Exact Match (EM), F1

## Project Structure

```
BurmeseGLUE/
├── burmeseglue/              # Main package
│   ├── datasets/             # Data loaders
│   ├── tasks/                # Task processors
│   ├── models/               # Model implementations
│   ├── metrics/              # Evaluation metrics
│   ├── training/             # Training utilities
│   ├── evaluation/           # Evaluation pipeline
│   └── utils/                # Common utilities
├── scripts/                  # CLI scripts
│   ├── download_data.py      # Download datasets
│   ├── train.py              # Train models
│   ├── evaluate.py           # Evaluate models
│   └── benchmark.py          # Run full benchmark
├── configs/                  # Configuration files
│   ├── baseline.yaml         # Baseline configs
│   └── transformer.yaml      # Transformer configs
├── examples/                 # Example notebooks
├── tests/                    # Unit tests
└── data/                     # Data directory (gitignored)
```

## Baseline Models

### TF-IDF + Linear Models

Simple but effective baselines using TF-IDF features with:
- Logistic Regression (classification)
- Ridge Regression (similarity)
- CRF (NER)

### Transformer Models

Fine-tuning multilingual transformers:
- **mBERT** (bert-base-multilingual-cased)
- **XLM-RoBERTa** (xlm-roberta-base)

## Configuration

Models are configured via YAML files. See `configs/` directory for examples.

**Example transformer config:**
```yaml
model:
  name: xlm-roberta-base
  num_labels: 2

training:
  learning_rate: 2e-5
  batch_size: 16
  epochs: 5
  warmup_steps: 500
```

## Results & Leaderboard

### Baseline Results (Coming Soon)

| Model | Sentiment | NER | Similarity | QA | Average |
|-------|-----------|-----|------------|----|------------|
| TF-IDF + LR | - | - | - | - | - |
| mBERT | - | - | - | - | - |
| XLM-RoBERTa | - | - | - | - | - |

**Submit your results!** Open a PR to add your model to the leaderboard.

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Check code style
black burmeseglue/ scripts/ tests/
flake8 burmeseglue/ scripts/ tests/
```

### Adding a New Task

1. Create dataset loader in `burmeseglue/datasets/your_task.py`
2. Implement task processor in `burmeseglue/tasks/your_task.py`
3. Add metrics in `burmeseglue/metrics/your_metrics.py`
4. Update configuration files
5. Add tests

See existing implementations for reference.

## Dataset Sources & Licenses

- **Sentiment**: [Link to dataset and license]
- **NER**: [Link to dataset and license]
- **Similarity**: [Link to dataset and license]
- **QA**: [Link to dataset and license]

Please cite the original dataset papers when using BurmeseGLUE.

## Citation

If you use BurmeseGLUE in your research, please cite:

```bibtex
@misc{burmeseglue2026,
  title={BurmeseGLUE: A Multi-Task Benchmark for Burmese Natural Language Understanding},
  author={BurmeseGLUE Contributors},
  year={2026},
  url={https://github.com/yourusername/burmeseglue}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas for contribution:
- New tasks and datasets
- Improved baseline implementations
- Bug fixes and documentation
- Leaderboard submissions

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contact

- **GitHub Issues**: [Report bugs or feature requests](https://github.com/yourusername/burmeseglue/issues)
- **Email**: [your-email@example.com]

## Acknowledgments

- Inspired by the English GLUE benchmark
- Built with PyTorch and Hugging Face Transformers
- Thanks to all dataset contributors and the Burmese NLP community

---

**Status:** 🚧 Under active development. Contributions welcome!
