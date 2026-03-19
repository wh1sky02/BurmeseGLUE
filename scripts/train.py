"""
Training script for BurmeseGLUE models.

Usage:
    # Train baseline model
    python scripts/train.py --task sentiment --model baseline --output_dir ./runs/baseline

    # Train transformer model
    python scripts/train.py --task sentiment --model xlm-roberta-base --output_dir ./runs/xlmr --epochs 3
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from burmeseglue.datasets import SentimentDataset
from burmeseglue.tasks import ClassificationTask
from burmeseglue.models import BaselineClassifier, TransformerClassifier
from burmeseglue.metrics import compute_classification_metrics


def main():
    parser = argparse.ArgumentParser(description="Train BurmeseGLUE models")

    # Task and data arguments
    parser.add_argument("--task", type=str, default="sentiment",
                        choices=["sentiment"], help="Task name")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Data directory")

    # Model arguments
    parser.add_argument("--model", type=str, default="baseline",
                        help="Model name (baseline, xlm-roberta-base, bert-base-multilingual-cased, etc.)")
    parser.add_argument("--output_dir", type=str, default="./runs/default",
                        help="Output directory for trained model")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate (for transformers)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic data for testing")

    args = parser.parse_args()

    print("=" * 80)
    print(f"BurmeseGLUE Training Script")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    if args.task == "sentiment":
        dataset = SentimentDataset(
            data_dir=args.data_dir,
            use_synthetic=args.use_synthetic,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    print(f"Dataset loaded: {dataset}")
    print(f"  Train examples: {len(dataset.load_split('train'))}")
    print(f"  Val examples: {len(dataset.load_split('validation'))}")
    print(f"  Test examples: {len(dataset.load_split('test'))}")

    # Load task processor
    print("\n[2/5] Setting up task processor...")
    task = ClassificationTask(dataset=dataset, max_length=args.max_length)
    print(f"Task: {task}")
    print(f"  Num labels: {task.get_num_labels()}")
    print(f"  Metrics: {task.get_metrics()}")

    # Prepare data
    print("\n[3/5] Preparing data...")
    train_examples = task.get_train_examples()
    val_examples = task.get_dev_examples()
    test_examples = task.get_test_examples()

    train_texts = [ex["text"] for ex in train_examples]
    train_labels = [ex["label"] for ex in train_examples]

    val_texts = [ex["text"] for ex in val_examples]
    val_labels = [ex["label"] for ex in val_examples]

    test_texts = [ex["text"] for ex in test_examples]
    test_labels = [ex["label"] for ex in test_examples]

    print(f"Data prepared:")
    print(f"  Train: {len(train_texts)} examples")
    print(f"  Val: {len(val_texts)} examples")
    print(f"  Test: {len(test_texts)} examples")

    # Train model
    print("\n[4/5] Training model...")

    if args.model == "baseline":
        # Train baseline model
        model = BaselineClassifier(model_type="logistic")
        model.fit(train_texts, train_labels, verbose=True)

    else:
        # Train transformer model
        model = TransformerClassifier(
            model_name=args.model,
            num_labels=task.get_num_labels(),
            max_length=args.max_length,
        )

        model.train(
            train_texts=train_texts,
            train_labels=train_labels,
            eval_texts=val_texts,
            eval_labels=val_labels,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    # Evaluate model
    print("\n[5/5] Evaluating model...")

    # Evaluate on validation set
    print("\nValidation set:")
    val_metrics = model.evaluate(val_texts, val_labels, compute_classification_metrics)
    for metric_name, value in val_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Evaluate on test set
    print("\nTest set:")
    test_metrics = model.evaluate(test_texts, test_labels, compute_classification_metrics)
    for metric_name, value in test_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Save model
    print(f"\nSaving model to {args.output_dir}...")
    model.save(args.output_dir)

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
