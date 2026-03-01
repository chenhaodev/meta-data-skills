"""Benchmark helpers for Hospital@Home dataset.

Hospital@Home is a patient simulation dataset with no ground-truth outcome label.
score() implements keyword-coverage heuristic: checks whether the model response
addresses clinically expected topics (fall risk, medications, conditions, frailty).

For rigorous evaluation, use an LLM-as-judge rubric tailored to nursing assessment
quality criteria (assessment completeness, safety prioritisation, communication).
"""
from typing import Callable
import pandas as pd
from pipeline import format_item

# Clinical keywords expected in a good nursing assessment response
ASSESSMENT_KEYWORDS = [
    "fall", "medication", "frailty", "assessment", "vitals",
    "condition", "risk", "monitor", "safety", "home",
]


def score(prediction: str, ground_truth: str) -> float:
    """Keyword-coverage proxy score for patient simulation responses.

    Since there is no ground-truth answer in Hospital@Home, this heuristic
    measures whether the prediction addresses core nursing assessment topics.

    Returns:
        Float 0.0–1.0: fraction of ASSESSMENT_KEYWORDS present in prediction.
    """
    if not prediction.strip():
        return 0.0
    pred_lower = prediction.strip().lower()
    hits = sum(1 for kw in ASSESSMENT_KEYWORDS if kw in pred_lower)
    return hits / len(ASSESSMENT_KEYWORDS)


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
