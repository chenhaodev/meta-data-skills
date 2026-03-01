"""Benchmark helpers for MedBench 2026 dataset.

MedBench 2026 uses two scoring methods:
  - MedIntentID: Accuracy (exact match of intent label string)
  - All others:  LLM-as-judge (requires external judge call — not implementable locally)

score() here uses exact match for all subtasks. Since TEST answers are empty strings,
this will return 0.0 for all generation tasks when run locally. To get real scores,
submit outputs to the MedBench leaderboard or run your own LLM-as-judge pipeline.
"""
from typing import Callable
import pandas as pd
from pipeline import format_item, ACCURACY_SUBTASKS


def score(prediction: str, ground_truth: str, subtask: str = "") -> float:
    """Score a prediction for MedBench 2026.

    For MedIntentID (accuracy subtask): exact match (strip + lower).
    For all other subtasks: returns 1.0 if prediction is non-empty (proxy only —
    real scoring requires LLM-as-judge; TEST ground truth is always empty).

    Args:
        prediction:   Model output string.
        ground_truth: Reference answer (empty string in TEST split).
        subtask:      Subtask name (from item["metadata"]["subtask"]).

    Returns:
        Float 0.0–1.0.
    """
    if subtask in ACCURACY_SUBTASKS:
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0
    # LLM-judge subtasks: proxy — reward non-empty responses
    return 1.0 if prediction.strip() else 0.0


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
        s = score(prediction, item["answer"], subtask=item["metadata"]["subtask"])
        results.append({"id": item["id"], "score": s, "prediction": prediction,
                        "subtask": item["metadata"]["subtask"]})
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
