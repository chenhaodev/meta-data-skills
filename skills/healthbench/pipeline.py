"""Data access layer for HealthBench dataset.

HealthBench is an open-ended rubric-graded medical benchmark from OpenAI (2025).
5,000 multi-turn conversations with 48,562 physician-written rubric criteria.
HuggingFace: openai/healthbench

Schema note: the HuggingFace dataset has schema drift across JSONL files.
We load via the `datasets` library which normalises this automatically.
"""
import pandas as pd
from datasets import load_dataset as hf_load
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "healthbench"


def load_dataset(split: str = "test") -> pd.DataFrame:
    """Load HealthBench from HuggingFace.

    Args:
        split: Only 'test' is available (HealthBench has no train split).

    Returns:
        DataFrame with one row per conversation example.
        Key columns: prompt, prompt_id, canary, example_tags,
                     ideal_completions_data, rubrics.
    """
    ds = hf_load("openai/healthbench", split=split)
    return ds.to_pandas()


def sample(n: int, split: str = "test", seed: int = 42) -> pd.DataFrame:
    """Sample n rows from the dataset reproducibly."""
    df = load_dataset(split)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def format_item(item: dict) -> dict:
    """Normalize a HealthBench row to the common med-data schema.

    HealthBench prompt is a list of {"role": ..., "content": ...} dicts.
    The final user turn is extracted as the question.
    ideal_completions_data.ideal_completion is the ground-truth answer proxy.

    Returns:
        dict with keys: id, question, context, choices, answer, metadata
    """
    prompt_turns = item.get("prompt", [])
    # Extract last user message as the question
    user_turns = [t for t in prompt_turns if t.get("role") == "user"]
    question = user_turns[-1]["content"] if user_turns else ""

    # Prior turns (system + earlier user/assistant) form context
    context_turns = prompt_turns[:-1] if user_turns else prompt_turns
    context = "\n".join(
        f"[{t['role']}] {t['content']}" for t in context_turns
    )

    ideal = item.get("ideal_completions_data") or {}
    answer = ideal.get("ideal_completion", "")

    return {
        "id": str(item.get("prompt_id", "")),
        "question": question,
        "context": context,
        "choices": [],  # open-generation task — no MCQ choices
        "answer": answer,
        "metadata": {
            "example_tags": item.get("example_tags", []),
            "rubrics": item.get("rubrics", []),
            "canary": item.get("canary", ""),
        },
    }
