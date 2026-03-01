"""Benchmark helpers for HealthBench dataset.

HealthBench is an open-generation benchmark graded against physician rubrics.
Full evaluation requires an LLM-as-judge against each rubric criterion.
This stub uses ideal_completion substring overlap as a lightweight proxy.

For production rubric grading, see OpenAI's simple-evals:
  https://github.com/openai/simple-evals
"""
from typing import Callable
import pandas as pd
from pipeline import format_item


def score(prediction: str, ground_truth: str) -> float:
    """Proxy score: fraction of ground-truth key phrases found in prediction.

    HealthBench is rubric-graded (not exact-match). This uses a lightweight
    heuristic: split ideal_completion into sentences, check what fraction
    appear (case-insensitive substring) in the prediction.

    For rigorous evaluation, replace this with an LLM-as-judge rubric scorer.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not ground_truth.strip():
        return 0.0
    sentences = [s.strip() for s in ground_truth.split(".") if len(s.strip()) > 20]
    if not sentences:
        return 1.0 if ground_truth.strip().lower() in prediction.strip().lower() else 0.0
    pred_lower = prediction.strip().lower()
    hits = sum(1 for s in sentences if s.lower() in pred_lower)
    return hits / len(sentences)


def run_benchmark(model_fn: Callable[[dict], str], dataset: pd.DataFrame, n: int = 100) -> dict:
    """Run model_fn on n samples and return aggregated results.

    Args:
        model_fn: Callable that takes format_item() output and returns prediction string.
        dataset:  DataFrame from load_dataset().
        n:        Number of samples to run.

    Returns:
        dict with keys: mean_score, n, results (list of per-item dicts)
    """
    samples = dataset.sample(n=min(n, len(dataset)), random_state=42)
    results = []
    for _, row in samples.iterrows():
        item = format_item(row.to_dict())
        prediction = model_fn(item)
        s = score(prediction, item["answer"])
        results.append({"id": item["id"], "score": s, "prediction": prediction})
    scores = [r["score"] for r in results]
    return {"mean_score": sum(scores) / len(scores), "n": len(scores), "results": results}


def compare_models(results_a: dict, results_b: dict) -> dict:
    """Compare two run_benchmark result dicts.

    Returns:
        dict with keys: a_score, b_score, delta (b - a)
    """
    return {
        "a_score": results_a["mean_score"],
        "b_score": results_b["mean_score"],
        "delta": results_b["mean_score"] - results_a["mean_score"],
    }
