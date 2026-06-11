# BurmeseGLUE

A standardized evaluation benchmark for Burmese Natural Language Understanding (NLU), inspired by the English GLUE benchmark. Provides a unified framework for reproducible evaluation of Burmese language models across multiple NLP tasks.

## Project Overview

**Tasks supported:**
- Sentiment Classification
- Named Entity Recognition (NER)
- Sentence Similarity
- Question Answering (QA)

**Model types:**
- Baseline: TF-IDF + Logistic Regression / CRF
- Transformers: mBERT, XLM-RoBERTa

## Tech Stack

- **Language:** Python 3.12
- **ML Frameworks:** PyTorch 2.x, HuggingFace Transformers, Datasets
- **NLP Utilities:** tokenizers, sentencepiece, seqeval, sklearn-crfsuite
- **Data Processing:** pandas, numpy, scipy, scikit-learn
- **Configuration:** PyYAML

## Project Structure

```
burmeseglue/        # Core library package
  datasets/         # Data loading and formatting
  tasks/            # Task-specific processing
  models/           # Model implementations
  metrics/          # Evaluation metrics
  training/         # Training utilities
  evaluation/       # Evaluation pipeline
  utils/            # Shared utilities
scripts/            # CLI entry points
  download_data.py  # Download datasets
  train.py          # Train models
  evaluate.py       # Evaluate models
  benchmark.py      # Run full benchmark
  test_installation.py  # Verify installation
configs/            # YAML config files
  baseline.yaml     # Baseline model config
  transformer.yaml  # Transformer model config
data/               # Local dataset storage (gitignored)
examples/           # Example scripts and notebooks
```

## Getting Started

### Verify Installation
```bash
python scripts/test_installation.py
```

### Download Data
```bash
burmeseglue-download --task sentiment
```

### Train a Baseline Model
```bash
burmeseglue-train --config configs/baseline.yaml --task sentiment
```

### Run Benchmark
```bash
burmeseglue-benchmark --config configs/baseline.yaml
```

## Workflow

The **"Start application"** workflow runs `python scripts/test_installation.py` to verify the environment is set up correctly.

## User Preferences

- Use Python 3.12
- Follow existing project structure and conventions
