"""
Quick test script to verify BurmeseGLUE installation and basic functionality.

Run this after installation to ensure everything works.
"""
import sys
from pathlib import Path

# Add parent directory if running from scripts/
if Path(__file__).parent.name == 'scripts':
    sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import burmeseglue
        from burmeseglue.datasets import SentimentDataset
        from burmeseglue.tasks import ClassificationTask
        from burmeseglue.models import BaselineClassifier, TransformerClassifier
        from burmeseglue.metrics import compute_classification_metrics
        print("  [OK] All imports successful")
        return True
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_dataset():
    """Test dataset creation and loading."""
    print("\nTesting dataset...")
    try:
        from burmeseglue.datasets import SentimentDataset

        dataset = SentimentDataset(
            data_dir="./data",
            use_synthetic=True,
            download=True,
        )

        train = dataset.load_split("train")
        val = dataset.load_split("validation")
        test = dataset.load_split("test")

        assert len(train) > 0, "Training set is empty"
        assert len(val) > 0, "Validation set is empty"
        assert len(test) > 0, "Test set is empty"

        print(f"  [OK] Dataset created: {len(train)} train, {len(val)} val, {len(test)} test")
        return True
    except Exception as e:
        print(f"  [FAIL] Dataset error: {e}")
        return False


def test_baseline_model():
    """Test baseline model training and prediction."""
    print("\nTesting baseline model...")
    try:
        from burmeseglue.datasets import SentimentDataset
        from burmeseglue.models import BaselineClassifier

        # Create small dataset
        dataset = SentimentDataset(data_dir="./data", use_synthetic=True, download=False)
        train = dataset.load_split("train")[:20]  # Use only 20 examples

        texts = [ex["text"] for ex in train]
        labels = [ex["label"] for ex in train]

        # Train model
        model = BaselineClassifier()
        model.fit(texts, labels, verbose=False)

        # Predict
        predictions = model.predict(texts[:5])

        assert len(predictions) == 5, "Wrong number of predictions"

        print(f"  [OK] Baseline model trained and predictions made")
        return True
    except Exception as e:
        print(f"  [FAIL] Baseline model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation."""
    print("\nTesting metrics...")
    try:
        from burmeseglue.metrics import compute_classification_metrics
        import numpy as np

        # Create dummy predictions and labels
        predictions = np.array([0, 1, 1, 0, 1])
        labels = np.array([0, 1, 0, 0, 1])

        metrics = compute_classification_metrics(predictions, labels)

        assert "accuracy" in metrics, "accuracy metric missing"
        assert "f1" in metrics, "f1 metric missing"
        assert 0 <= metrics["accuracy"] <= 1, "accuracy out of range"

        print(f"  [OK] Metrics computed: accuracy={metrics['accuracy']:.2f}, f1={metrics['f1']:.2f}")
        return True
    except Exception as e:
        print(f"  [FAIL] Metrics error: {e}")
        return False


def main():
    print("=" * 80)
    print("BurmeseGLUE Installation Test")
    print("=" * 80)

    tests = [
        ("Imports", test_imports),
        ("Dataset", test_dataset),
        ("Baseline Model", test_baseline_model),
        ("Metrics", test_metrics),
    ]

    results = []
    for test_name, test_fn in tests:
        try:
            results.append(test_fn())
        except Exception as e:
            print(f"\n[FAIL] {test_name} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for (test_name, _), result in zip(tests, results):
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test_name}")

    total = len(results)
    passed = sum(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed! BurmeseGLUE is ready to use.")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
