"""Data access layer for MedXpertQA dataset.

Expert-level medical MCQ benchmark from TsinghuaC3I (2025).
HuggingFace: TsinghuaC3I/MedXpertQA

Two subsets:
  Text — 2,460 questions, 10 options (A-J), text-only
  MM   — 2,010 questions,  5 options (A-E), multimodal (text + images)

Each subset has a 'dev' split (5 items) and a 'test' split (~2,000 items).

Schema: id, question, options (dict), label (answer letter),
        images (list, MM only), medical_task, body_system, question_type
"""
import pandas as pd
from datasets import load_dataset as hf_load
from typing import Literal

SUBSETS = ("Text", "MM")
SPLITS = ("dev", "test")


def load_dataset(
    split: str = "test",
    subset: Literal["Text", "MM"] = "Text",
) -> pd.DataFrame:
    """Load MedXpertQA from HuggingFace.

    Args:
        split:  'dev' (5 items, sanity check) or 'test' (~2,000 items).
        subset: 'Text' (10-option A-J) or 'MM' (5-option A-E, multimodal).

    Returns:
        DataFrame with columns: id, question, options, label, images,
        medical_task, body_system, question_type.
        '_subset' column is added for downstream grouping.
    """
    ds = hf_load("TsinghuaC3I/MedXpertQA", subset, split=split)
    df = ds.to_pandas()
    df["_subset"] = subset
    return df


def sample(
    n: int,
    split: str = "test",
    subset: Literal["Text", "MM"] = "Text",
    seed: int = 42,
) -> pd.DataFrame:
    """Sample n rows from the dataset reproducibly."""
    df = load_dataset(split=split, subset=subset)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def format_item(item: dict) -> dict:
    """Normalize a MedXpertQA row to the common med-data schema.

    'choices' is a list of option strings ordered by letter (A, B, C, ...).
    'answer' is the single correct option letter (e.g., 'A', 'B', ...).
    'context' encodes body_system + medical_task + question_type metadata.

    Returns:
        dict with keys: id, question, context, choices, answer, metadata
    """
    options: dict = item.get("options") or {}
    # Sort by key to get consistent A, B, C, ... ordering
    choices = [options[k] for k in sorted(options.keys())]

    context = (
        f"Body system: {item.get('body_system', '')} | "
        f"Medical task: {item.get('medical_task', '')} | "
        f"Question type: {item.get('question_type', '')}"
    )

    images = item.get("images") or []
    # images may be PIL objects (when loaded with HF datasets) or filenames
    image_refs = [
        img.filename if hasattr(img, "filename") else str(img)
        for img in images
    ]

    return {
        "id": str(item.get("id", "")),
        "question": str(item.get("question", "")),
        "context": context,
        "choices": choices,
        "answer": str(item.get("label", "")),
        "metadata": {
            "subset":        item.get("_subset", ""),
            "body_system":   item.get("body_system", ""),
            "medical_task":  item.get("medical_task", ""),
            "question_type": item.get("question_type", ""),
            "images":        image_refs,
            "n_options":     len(choices),
        },
    }
