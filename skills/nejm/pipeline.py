"""Data access layer for NEJM Clinical Annotations dataset.

101 NEJM clinical vignettes answered by an LLM, annotated by two physician experts.
Local Excel: data/NEJM/41746_2024_1010_MOESM9_ESM.xlsx (single sheet, 104 rows).

Paper: npj Digital Medicine 7, 103 (2024). DOI: 10.1038/s41746-024-01010-1

Columns:
  index, question, answer, Prompt Style, Model answer, rationale,
  Correct?, Reviewer #1 Number of errors, Explaining the errors found,
  Reviewer #2 Number of errors, Explaining the errors found.1

Data quality notes:
  - 3 rows have NaN index (summary/metadata rows) — dropped by load_dataset()
  - 'Correct?' contains dirty values: 'Y', 'N', 'f', 'false arguments', integers
  - Error count columns contain fractional values in ~2 rows (data entry artefacts)
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "NEJM"
XLSX_FILE = DATA_DIR / "41746_2024_1010_MOESM9_ESM.xlsx"


def _parse_correct(val) -> str:
    """Normalise the dirty 'Correct?' column to 'Y', 'N', or 'unknown'."""
    if pd.isna(val):
        return "unknown"
    s = str(val).strip().upper()
    if s in ("Y", "YES", "TRUE"):
        return "Y"
    if s in ("N", "NO", "FALSE", "F", "FALSE ARGUMENTS"):
        return "N"
    return "unknown"


def _parse_errors(val) -> int:
    """Normalise error count to int; return -1 for non-integer/artefact values."""
    if pd.isna(val):
        return 0
    try:
        f = float(val)
        if f == int(f) and 0 <= f <= 10:
            return int(f)
        return -1  # fractional or out-of-range — data artefact
    except (ValueError, TypeError):
        return -1


def load_dataset(split: str = "test") -> pd.DataFrame:
    """Load NEJM Clinical Annotations from the local Excel file.

    Args:
        split: Ignored — no train/val/test split. All 101 valid rows returned.

    Returns:
        DataFrame with 101 rows and cleaned columns:
          question, answer, model_answer, rationale, correct_clean,
          r1_errors_clean, r1_explanation, r2_errors_clean, r2_explanation,
          prompt_style, row_index.
    """
    raw = pd.read_excel(XLSX_FILE)
    # Drop 3 non-data rows (NaN index)
    raw = raw[raw["index"].notna()].reset_index(drop=True)

    df = pd.DataFrame({
        "row_index":       raw["index"].astype(int).astype(str),
        "question":        raw["question"].fillna(""),
        "answer":          raw["answer"].fillna(""),
        "prompt_style":    raw["Prompt Style"].fillna("DR"),
        "model_answer":    raw["Model answer"].fillna(""),
        "rationale":       raw["rationale"].fillna(""),
        "correct_clean":   raw["Correct?"].apply(_parse_correct),
        "r1_errors_clean": raw["Reviewer # 1 Number of errors"].apply(_parse_errors),
        "r1_explanation":  raw["Explaining the errors found"].fillna(""),
        "r2_errors_clean": raw["Reviewer # 2Number of errors"].apply(_parse_errors),
        "r2_explanation":  raw["Explaining the errors found.1"].fillna(""),
    })
    return df


def sample(n: int, split: str = "test", seed: int = 42) -> pd.DataFrame:
    """Sample n rows from the dataset reproducibly."""
    df = load_dataset(split)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def format_item(item: dict) -> dict:
    """Normalize a NEJM row to the common med-data schema.

    'question' is the clinical vignette.
    'answer' is the expert ground-truth answer (short clinical phrase).
    'context' contains the embedded model answer + rationale (for NLI tasks).

    Returns:
        dict with keys: id, question, context, choices, answer, metadata
    """
    context = ""
    if item.get("model_answer") or item.get("rationale"):
        context = (
            f"[Model answer] {item.get('model_answer', '')}\n"
            f"[Rationale] {item.get('rationale', '')}"
        )

    return {
        "id": str(item.get("row_index", "")),
        "question": str(item.get("question", "")),
        "context": context,
        "choices": [],  # open-ended — no MCQ options
        "answer": str(item.get("answer", "")),
        "metadata": {
            "prompt_style":    item.get("prompt_style", "DR"),
            "correct_clean":   item.get("correct_clean", "unknown"),
            "r1_errors_clean": item.get("r1_errors_clean", 0),
            "r1_explanation":  item.get("r1_explanation", ""),
            "r2_errors_clean": item.get("r2_errors_clean", 0),
            "r2_explanation":  item.get("r2_explanation", ""),
        },
    }
