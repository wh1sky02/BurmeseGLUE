"""
BurmeseGLUE Full Benchmark Script.

Runs all 4 tasks with baseline models and reports a unified benchmark score.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --data_dir ./data --output_dir ./runs/benchmark
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SEPARATOR = "=" * 80
THIN_SEP  = "-" * 80


def print_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def print_metrics(metrics: dict, indent: int = 4) -> None:
    pad = " " * indent
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{pad}{k:20s}: {v:.4f}")
        else:
            print(f"{pad}{k:20s}: {v}")


# ─────────────────────────────────────────────────────────────────────────────
# Task runners
# ─────────────────────────────────────────────────────────────────────────────

def run_sentiment(data_dir: str, output_dir: str) -> dict:
    print_section("TASK 1 / 4 — Sentiment Classification (Baseline: TF-IDF + LogReg)")

    from burmeseglue.datasets import SentimentDataset
    from burmeseglue.tasks import ClassificationTask
    from burmeseglue.models import BaselineClassifier
    from burmeseglue.metrics import (
        compute_classification_metrics,
        compute_classification_report,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset …")
    dataset = SentimentDataset(data_dir=data_dir, download=True, seed=42)
    task    = ClassificationTask(dataset=dataset)

    train_ex = task.get_train_examples()
    val_ex   = task.get_dev_examples()
    test_ex  = task.get_test_examples()

    print(f"      Train: {len(train_ex)}  Val: {len(val_ex)}  Test: {len(test_ex)}")
    print(f"      Labels: {dataset.get_label_names()}")

    train_texts  = [e["text"]  for e in train_ex]
    train_labels = [e["label"] for e in train_ex]
    val_texts    = [e["text"]  for e in val_ex]
    val_labels   = [e["label"] for e in val_ex]
    test_texts   = [e["text"]  for e in test_ex]
    test_labels  = [e["label"] for e in test_ex]

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[2/4] Training …")
    model = BaselineClassifier(model_type="logistic")
    model.fit(train_texts, train_labels, verbose=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[3/4] Evaluating …")
    val_metrics  = model.evaluate(val_texts,  val_labels,  compute_classification_metrics)
    test_metrics = model.evaluate(test_texts, test_labels, compute_classification_metrics)

    print("\n  Validation:")
    print_metrics(val_metrics)
    print("\n  Test:")
    print_metrics(test_metrics)

    # Detailed report on test
    test_preds = model.predict(test_texts)
    print("\n  Classification Report (test):")
    print(compute_classification_report(test_preds, test_labels,
                                        dataset.get_label_names()))

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[4/4] Saving model …")
    model.save(str(Path(output_dir) / "sentiment"))

    return {"val": val_metrics, "test": test_metrics}


def run_ner(data_dir: str, output_dir: str) -> dict:
    print_section("TASK 2 / 4 — Named Entity Recognition  (Baseline: CRF)")

    from burmeseglue.datasets import NERDataset
    from burmeseglue.tasks import TokenClassificationTask
    from burmeseglue.models import NERBaselineModel
    from burmeseglue.metrics import (
        compute_token_metrics,
        compute_token_classification_report,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset …")
    dataset = NERDataset(data_dir=data_dir, download=True, seed=42)
    task    = TokenClassificationTask(dataset=dataset)

    train_ex = task.get_train_examples()
    val_ex   = task.get_dev_examples()
    test_ex  = task.get_test_examples()

    print(f"      Train: {len(train_ex)}  Val: {len(val_ex)}  Test: {len(test_ex)}")
    print(f"      Labels: {dataset.get_labels()}")

    train_tokens = [e["tokens"] for e in train_ex]
    train_tags   = [e["tags"]   for e in train_ex]
    val_tokens   = [e["tokens"] for e in val_ex]
    val_tags     = [e["tags"]   for e in val_ex]
    test_tokens  = [e["tokens"] for e in test_ex]
    test_tags    = [e["tags"]   for e in test_ex]

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[2/4] Training CRF …")
    model = NERBaselineModel(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100)
    model.fit(train_tokens, train_tags, verbose=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[3/4] Evaluating …")
    val_metrics  = model.evaluate(val_tokens,  val_tags,  compute_token_metrics)
    test_metrics = model.evaluate(test_tokens, test_tags, compute_token_metrics)

    print("\n  Validation:")
    print_metrics(val_metrics)
    print("\n  Test:")
    print_metrics(test_metrics)

    # Per-entity report
    test_preds = model.predict(test_tokens)
    print("\n  Entity-level Report (test):")
    print(compute_token_classification_report(test_preds, test_tags))

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[4/4] Saving model …")
    model.save(str(Path(output_dir) / "ner"))

    return {"val": val_metrics, "test": test_metrics}


def run_similarity(data_dir: str, output_dir: str) -> dict:
    print_section("TASK 3 / 4 — Sentence Similarity  (Baseline: TF-IDF + Ridge)")

    from burmeseglue.datasets import SimilarityDataset
    from burmeseglue.tasks import RegressionTask
    from burmeseglue.models import SimilarityBaselineModel
    from burmeseglue.metrics import compute_regression_metrics

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset …")
    dataset = SimilarityDataset(data_dir=data_dir, download=True, seed=42)
    task    = RegressionTask(dataset=dataset)

    train_ex = task.get_train_examples()
    val_ex   = task.get_dev_examples()
    test_ex  = task.get_test_examples()

    print(f"      Train: {len(train_ex)}  Val: {len(val_ex)}  Test: {len(test_ex)}")

    def unpack(examples):
        s1 = [e["sentence1"] for e in examples]
        s2 = [e["sentence2"] for e in examples]
        sc = [e["score"]     for e in examples]
        return s1, s2, sc

    tr_s1, tr_s2, tr_sc = unpack(train_ex)
    va_s1, va_s2, va_sc = unpack(val_ex)
    te_s1, te_s2, te_sc = unpack(test_ex)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[2/4] Training Ridge regression …")
    model = SimilarityBaselineModel(alpha=1.0)
    model.fit(tr_s1, tr_s2, tr_sc, verbose=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[3/4] Evaluating …")
    val_metrics  = model.evaluate(va_s1, va_s2, va_sc, compute_regression_metrics)
    test_metrics = model.evaluate(te_s1, te_s2, te_sc, compute_regression_metrics)

    print("\n  Validation:")
    print_metrics(val_metrics)
    print("\n  Test:")
    print_metrics(test_metrics)

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[4/4] Saving model …")
    model.save(str(Path(output_dir) / "similarity"))

    return {"val": val_metrics, "test": test_metrics}


def run_qa(data_dir: str, output_dir: str) -> dict:
    print_section("TASK 4 / 4 — Question Answering  (Baseline: TF-IDF Span Retrieval)")

    from burmeseglue.datasets import QADataset
    from burmeseglue.tasks import SpanExtractionTask
    from burmeseglue.models import QABaselineModel
    from burmeseglue.metrics import compute_qa_metrics

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset …")
    dataset = QADataset(data_dir=data_dir, download=True, seed=42)
    task    = SpanExtractionTask(dataset=dataset)

    train_ex = task.get_train_examples()
    val_ex   = task.get_dev_examples()
    test_ex  = task.get_test_examples()

    print(f"      Train: {len(train_ex)}  Val: {len(val_ex)}  Test: {len(test_ex)}")

    def unpack(examples):
        ctxs = [e["context"]  for e in examples]
        qs   = [e["question"] for e in examples]
        ans  = [e["answers"]["text"] for e in examples]
        return ctxs, qs, ans

    tr_ctx, tr_q, tr_ans = unpack(train_ex)
    va_ctx, va_q, va_ans = unpack(val_ex)
    te_ctx, te_q, te_ans = unpack(test_ex)

    # ── "Train" (no-op for retrieval baseline) ────────────────────────────────
    print("\n[2/4] Initialising QA model (retrieval-based, no training) …")
    model = QABaselineModel()

    # Show a sample prediction to illustrate
    print("\n  Sample prediction:")
    sample_ctx = te_ctx[0]
    sample_q   = te_q[0]
    sample_ans = te_ans[0]
    sample_pred = model.predict_single(sample_ctx, sample_q)
    print(f"    Context : {sample_ctx[:80]}…")
    print(f"    Question: {sample_q}")
    print(f"    Gold    : {sample_ans[0]}")
    print(f"    Pred    : {sample_pred}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[3/4] Evaluating …")
    val_metrics  = model.evaluate(va_ctx, va_q, va_ans, compute_qa_metrics)
    test_metrics = model.evaluate(te_ctx, te_q, te_ans, compute_qa_metrics)

    print("\n  Validation:")
    print_metrics(val_metrics)
    print("\n  Test:")
    print_metrics(test_metrics)

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[4/4] Saving model config …")
    model.save(str(Path(output_dir) / "qa"))

    return {"val": val_metrics, "test": test_metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Final score aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_burmeseglue_score(results: dict) -> float:
    """
    Aggregate BurmeseGLUE score (macro-average of primary metrics).

    Primary metric per task:
      sentiment  → test accuracy
      ner        → test span_f1
      similarity → test pearson
      qa         → test f1
    """
    primaries = {
        "sentiment":  results["sentiment"]["test"].get("accuracy", 0),
        "ner":        results["ner"]["test"].get("span_f1",  0),
        "similarity": results["similarity"]["test"].get("pearson",   0),
        "qa":         results["qa"]["test"].get("f1",        0),
    }
    score = sum(primaries.values()) / len(primaries)
    return primaries, score


def print_final_report(results: dict, elapsed: float) -> None:
    primaries, score = compute_burmeseglue_score(results)

    print(f"\n{SEPARATOR}")
    print("  BURMESEGLUE BENCHMARK — FINAL RESULTS")
    print(SEPARATOR)
    print(f"\n  {'Task':<20} {'Primary Metric':<22} {'Val':>8}  {'Test':>8}")
    print(f"  {THIN_SEP[:60]}")

    task_info = [
        ("sentiment",  "accuracy",  "Sentiment"),
        ("ner",        "span_f1",   "NER"),
        ("similarity", "pearson",   "Similarity"),
        ("qa",         "f1",        "QA"),
    ]

    for task_key, metric_key, label in task_info:
        val_v  = results[task_key]["val"].get(metric_key,  0)
        test_v = results[task_key]["test"].get(metric_key, 0)
        marker = f"({metric_key})"
        print(f"  {label:<20} {marker:<22} {val_v:>8.4f}  {test_v:>8.4f}")

    print(f"\n  {THIN_SEP[:60]}")
    print(f"  {'BurmeseGLUE Score':<20} {'(macro avg)':22} {'':>8}  {score:>8.4f}")
    print(f"\n  All task results (test):")
    print(f"  {THIN_SEP[:60]}")

    for task_key, _, label in task_info:
        print(f"\n  [{label}]")
        print_metrics(results[task_key]["test"], indent=6)

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(SEPARATOR)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run BurmeseGLUE full benchmark")
    parser.add_argument("--data_dir",   default="./data",            help="Data root directory")
    parser.add_argument("--output_dir", default="./runs/benchmark",  help="Model output directory")
    parser.add_argument("--tasks",      default="all",
                        help="Comma-separated list of tasks or 'all' "
                             "(sentiment,ner,similarity,qa)")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    requested = (
        ["sentiment", "ner", "similarity", "qa"]
        if args.tasks == "all"
        else [t.strip() for t in args.tasks.split(",")]
    )

    print(SEPARATOR)
    print("  BurmeseGLUE Benchmark Runner")
    print(SEPARATOR)
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Tasks      : {', '.join(requested)}")

    start = time.time()
    results = {}

    runners = {
        "sentiment":  run_sentiment,
        "ner":        run_ner,
        "similarity": run_similarity,
        "qa":         run_qa,
    }

    for task in requested:
        if task not in runners:
            print(f"  [WARN] Unknown task '{task}' — skipping")
            continue
        try:
            results[task] = runners[task](args.data_dir, args.output_dir)
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] Task '{task}' failed: {exc}")
            traceback.print_exc()
            results[task] = {"val": {}, "test": {}}

    elapsed = time.time() - start

    if len(results) == 4:
        print_final_report(results, elapsed)

        # Save results JSON
        out_file = Path(args.output_dir) / "benchmark_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  Results saved to: {out_file}")
    else:
        print(f"\n  Completed {len(results)}/{len(requested)} tasks in {elapsed:.1f}s")

    print(f"\n{SEPARATOR}")
    print("  Benchmark complete!")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
