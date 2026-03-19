"""
Quick Start Example for BurmeseGLUE

This script demonstrates the basic usage of BurmeseGLUE for sentiment classification.
"""
from burmeseglue.datasets import SentimentDataset
from burmeseglue.tasks import ClassificationTask
from burmeseglue.models import BaselineClassifier
from burmeseglue.metrics import compute_classification_metrics

def main():
    print("BurmeseGLUE Quick Start Example")
    print("=" * 80)

    # 1. Load dataset
    print("\n[1] Loading dataset...")
    dataset = SentimentDataset(
        data_dir="./data",
        use_synthetic=True,  # Use synthetic data for demo
        download=True,
    )
    print(f"Dataset loaded with {len(dataset)} total examples")

    # 2. Create task processor
    print("\n[2] Setting up task processor...")
    task = ClassificationTask(dataset=dataset, max_length=128)
    print(f"Task: {task}")
    print(f"Number of labels: {task.get_num_labels()}")

    # 3. Prepare data
    print("\n[3] Preparing data...")
    train_examples = task.get_train_examples()
    test_examples = task.get_test_examples()

    train_texts = [ex["text"] for ex in train_examples]
    train_labels = [ex["label"] for ex in train_examples]

    test_texts = [ex["text"] for ex in test_examples]
    test_labels = [ex["label"] for ex in test_examples]

    print(f"Training examples: {len(train_texts)}")
    print(f"Test examples: {len(test_texts)}")

    # 4. Train baseline model
    print("\n[4] Training baseline model...")
    model = BaselineClassifier(model_type="logistic")
    model.fit(train_texts, train_labels, verbose=True)

    # 5. Make predictions
    print("\n[5] Making predictions...")
    predictions = model.predict(test_texts)
    print(f"Predictions shape: {predictions.shape}")

    # 6. Evaluate
    print("\n[6] Evaluating model...")
    metrics = compute_classification_metrics(predictions, test_labels)

    print("\nResults:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # 7. Try some custom predictions
    print("\n[7] Testing custom predictions...")
    custom_texts = [
        "ဤရုပ်ရှင်သည် အလွန်ကောင်းမွန်ပါသည်။",  # Positive
        "ဝန်ဆောင်မှု အလွန် ညံ့ပါတယ်။",  # Negative
    ]

    custom_predictions = model.predict(custom_texts)
    label_names = dataset.get_label_names()

    print("\nCustom predictions:")
    for i, (text, pred) in enumerate(zip(custom_texts, custom_predictions)):
        # Use safe encoding for Windows console
        try:
            print(f"  Text: {text}")
        except UnicodeEncodeError:
            print(f"  Text: [Burmese text {i+1}]")
        print(f"  Predicted: {label_names[pred]} ({pred})")
        print()

    print("=" * 80)
    print("Quick start complete!")


if __name__ == "__main__":
    main()
