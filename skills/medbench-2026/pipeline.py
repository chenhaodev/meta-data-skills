"""Data access layer for MedBench 2026 dataset.

Chinese-language agentic medical benchmark, 14 subtasks × 30 items = 420 total.
Local JSONL files: data/MedBench-2026/MedBench_Agent/TEST/<subtask>.jsonl

Each JSONL row: {"question": str, "answer": str, "other": {"id": int, "source": str}}
Note: "answer" is an empty string in the TEST split — no reference answers provided.

Subtasks: MedCOT, MedCallAPI, MedCollab, MedDBOps, MedDecomp, MedDefend,
          MedIntentID, MedLongConv, MedLongQA, MedPathPlan, MedReflect,
          MedRetAPI, MedRoleAdapt, MedShield
"""
import json
import pandas as pd
from pathlib import Path
from typing import Optional

# Note: directory is MedBench-2026 (capital B, hyphen), not medbench-2026
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "MedBench-2026"
JSONL_DIR = DATA_DIR / "MedBench_Agent" / "TEST"

SUBTASKS = [
    "MedCOT", "MedCallAPI", "MedCollab", "MedDBOps", "MedDecomp", "MedDefend",
    "MedIntentID", "MedLongConv", "MedLongQA", "MedPathPlan", "MedReflect",
    "MedRetAPI", "MedRoleAdapt", "MedShield",
]

# Subtasks scored by Accuracy (exact match); all others use LLM-as-judge
ACCURACY_SUBTASKS = {"MedIntentID"}


def load_dataset(split: str = "test", subtask: Optional[str] = None) -> pd.DataFrame:
    """Load MedBench 2026 from local JSONL files.

    Args:
        split:   Ignored — only TEST split is available.
        subtask: If given, load only that subtask (e.g., "MedCOT").
                 If None, load all 14 subtasks (420 rows total).

    Returns:
        DataFrame with columns: question, answer, other, _subtask.
        'answer' is always an empty string in the TEST split.
    """
    if subtask is not None:
        files = [JSONL_DIR / f"{subtask}.jsonl"]
    else:
        files = sorted(JSONL_DIR.glob("*.jsonl"))

    rows = []
    for f in files:
        task_name = f.stem
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_subtask"] = task_name
            rows.append(item)
    return pd.DataFrame(rows)


def sample(n: int, split: str = "test", seed: int = 42,
           subtask: Optional[str] = None) -> pd.DataFrame:
    """Sample n rows from the dataset reproducibly."""
    df = load_dataset(split, subtask=subtask)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def format_item(item: dict) -> dict:
    """Normalize a MedBench 2026 row to the common med-data schema.

    Returns:
        dict with keys: id, question, context, choices, answer, metadata
    """
    other = item.get("other") or {}
    if isinstance(other, str):
        try:
            other = json.loads(other)
        except Exception:
            other = {}

    subtask = item.get("_subtask", other.get("source", "unknown"))

    return {
        "id": str(other.get("id", "")),
        "question": item.get("question", ""),
        "context": "",  # questions are self-contained; no separate context field
        "choices": [],  # open-generation tasks; MedIntentID labels are embedded in question
        "answer": item.get("answer", ""),  # empty string in TEST split
        "metadata": {
            "subtask": subtask,
            "source": other.get("source", ""),
            "metric": "accuracy" if subtask in ACCURACY_SUBTASKS else "llm-judge",
        },
    }
