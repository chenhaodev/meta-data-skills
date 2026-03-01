"""Benchmark helpers for NEJM Clinical Annotations dataset.

NEJM answers are short clinical phrases (e.g., "Cholesterol embolization").
score() uses substring match: prediction is correct if it contains the ground-truth
phrase (case-insensitive). This handles common variations like "Cholesterol embolization
syndrome" matching "Cholesterol embolization".

For NLI use-case (classifying model answer as correct/incorrect), pass the
item["metadata"]["correct_clean"] label as ground_truth instead.
"""
from typing import Callable
import pandas as pd
from pipeline import format_item


def score(prediction: str, ground_truth: str) -> float:
    """Score a prediction against the NEJM ground-truth answer.

    Uses case-insensitive substring match: ground_truth phrase must appear
    somewhere in prediction. This gracefully handles answer elaborations.

    Returns:
        1.0 if ground_truth (stripped, lowercased) is found in prediction, else 0.0.
    """
    if not ground_truth.strip():
        return 0.0
    return 1.0 if ground_truth.strip().lower() in prediction.strip().lower() else 0.0


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
