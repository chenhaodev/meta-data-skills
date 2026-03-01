"""Benchmark helpers for MedXpertQA dataset.

MedXpertQA is a single-label MCQ task scored by exact match of the option letter.
model_fn should return a single uppercase letter (e.g., "A", "B", ..., "J").

Random baselines:
  Text subset (10 options A-J): ~10%
  MM   subset  (5 options A-E): ~20%

SOTA: o1 ~50% overall; top text-only models ~37-38%.
"""
from typing import Callable
import pandas as pd
from pipeline import format_item


def score(prediction: str, ground_truth: str) -> float:
    """Exact match on the option letter (case-insensitive, stripped).

    model_fn should return the letter of the chosen option ('A', 'B', ...).
    If the model returns a full answer string, only the first letter is used.

    Returns:
        1.0 if prediction letter matches ground_truth letter, else 0.0.
    """
    pred = prediction.strip().upper()
    gt = ground_truth.strip().upper()
    if not pred or not gt:
        return 0.0
    # Accept bare letter or "A." / "A)" style prefixes
    pred_letter = pred[0] if pred else ""
    return 1.0 if pred_letter == gt else 0.0


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
