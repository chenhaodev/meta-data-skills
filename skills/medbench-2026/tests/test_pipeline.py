"""Smoke tests for MedBench 2026 pipeline.

Run: pytest skills/medbench-2026/tests/ -v
Note: Tests load local JSONL files (no network needed) but are marked @pytest.mark.slow
      to allow skipping in fast CI runs.
      Run without slow tests: pytest skills/medbench-2026/tests/ -v -m "not slow"
      Requires: data/MedBench-2026/MedBench_Agent/TEST/*.jsonl to be present.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import load_dataset, sample, format_item, SUBTASKS


REQUIRED_SCHEMA_KEYS = {"id", "question", "context", "choices", "answer", "metadata"}


@pytest.mark.slow
def test_load_dataset_returns_dataframe():
    """Load all subtasks — 420 rows total."""
    df = load_dataset()
    assert len(df) == 420


@pytest.mark.slow
def test_load_single_subtask():
    """Load a single subtask returns 30 rows."""
    df = load_dataset(subtask="MedCOT")
    assert len(df) == 30
    assert (df["_subtask"] == "MedCOT").all()


@pytest.mark.slow
def test_all_subtasks_present():
    """All 14 subtasks are loaded."""
    df = load_dataset()
    assert set(df["_subtask"].unique()) == set(SUBTASKS)


@pytest.mark.slow
def test_sample_returns_n_rows():
    """sample() returns exactly n rows."""
    df = sample(n=5, seed=42)
    assert len(df) <= 5
    assert len(df) > 0


@pytest.mark.slow
def test_sample_is_reproducible():
    """Same seed always returns the same rows."""
    df1 = sample(n=10, seed=42)
    df2 = sample(n=10, seed=42)
    assert df1.equals(df2)


@pytest.mark.slow
def test_format_item_schema():
    """format_item() returns dict with all required keys."""
    df = sample(n=1, seed=42)
    item = format_item(df.iloc[0].to_dict())
    assert REQUIRED_SCHEMA_KEYS.issubset(item.keys()), (
        f"Missing keys: {REQUIRED_SCHEMA_KEYS - item.keys()}"
    )


@pytest.mark.slow
def test_format_item_types():
    """format_item() returns correct types for each field."""
    df = sample(n=1, seed=42)
    item = format_item(df.iloc[0].to_dict())
    assert isinstance(item["id"], str)
    assert isinstance(item["question"], str)
    assert isinstance(item["context"], str)
    assert isinstance(item["choices"], list)
    assert isinstance(item["answer"], str)
    assert isinstance(item["metadata"], dict)
    assert "subtask" in item["metadata"]
    assert "metric" in item["metadata"]
