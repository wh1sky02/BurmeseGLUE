"""
Evaluation script for BurmeseGLUE models.

Usage:
    # Evaluate baseline model
    python scripts/evaluate.py --task sentiment --model_path ./runs/baseline --split test

    # Evaluate transformer model
    python scripts/evaluate.py --task sentiment --model_path ./runs/xlmr --split test --model_type transformer
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from burmeseglue.datasets import SentimentDataset
from burmeseglue.tasks import ClassificationTask
from burmeseglue.models import BaselineClassifier, TransformerClassifier
from burmeseglue.metrics import (
    compute_classification_metrics,
    compute_classification_report,
    compute_confusion_matrix,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BurmeseGLUE models")

    # Task and data arguments
    parser.add_argument("--task", type=str, default="sentiment",
                        choices=["sentiment"], help="Task name")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Data directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"],
                        help="Data split to evaluate on")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--model_type", type=str, default="baseline",
                        choices=["baseline", "transformer"],
                        help="Type of model")

    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for predictions (JSON)")

    # Other arguments
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic data for testing")

    args = parser.parse_args()

    print("=" * 80)
    print(f"BurmeseGLUE Evaluation Script")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Model: {args.model_path}")
    print(f"Split: {args.split}")
    print("=" * 80)

    # Load dataset
    print("\n[1/4] Loading dataset...")
    if args.task == "sentiment":
        dataset = SentimentDataset(
            data_dir=args.data_dir,
            use_synthetic=args.use_synthetic,
            download=False,  # Assume data is already downloaded
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    print(f"Dataset loaded: {dataset}")

    # Load task processor
    print("\n[2/4] Setting up task processor...")
    task = ClassificationTask(dataset=dataset)
    print(f"Task: {task}")

    # Load model
    print(f"\n[3/4] Loading model from {args.model_path}...")
    if args.model_type == "baseline":
        model = BaselineClassifier.load(args.model_path)
    else:
        model = TransformerClassifier.load(args.model_path)

    print(f"Model loaded: {model}")

    # Prepare data
    print(f"\n[4/4] Evaluating on {args.split} set...")
    if args.split == "train":
        examples = task.get_train_examples()
    elif args.split == "validation":
        examples = task.get_dev_examples()
    else:
        examples = task.get_test_examples()

    texts = [ex["text"] for ex in examples]
    labels = [ex["label"] for ex in examples]

    print(f"  {len(examples)} examples")

    # Make predictions
    predictions = model.predict(texts)

    # Compute metrics
    print("\n" + "=" * 80)
    print("Metrics:")
    print("=" * 80)
    metrics = compute_classification_metrics(predictions, labels)
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Detailed report
    print("\n" + "=" * 80)
    print("Classification Report:")
    print("=" * 80)
    label_names = dataset.get_label_names() if hasattr(dataset, "get_label_names") else None
    report = compute_classification_report(predictions, labels, label_names)
    print(report)

    # Confusion matrix
    print("\n" + "=" * 80)
    print("Confusion Matrix:")
    print("=" * 80)
    cm = compute_confusion_matrix(predictions, labels)
    print(cm)

    # Save predictions if requested
    if args.output_file:
        print(f"\nSaving predictions to {args.output_file}...")

        decoded_predictions = task.decode_predictions(predictions, examples)

        output_data = {
            "task": args.task,
            "split": args.split,
            "model_path": args.model_path,
            "metrics": metrics,
            "predictions": decoded_predictions,
        }

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Predictions saved!")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
