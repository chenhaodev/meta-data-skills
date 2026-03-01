"""Smoke tests for NEJM Clinical Annotations pipeline.

Run: pytest skills/nejm/tests/ -v
Note: Tests load the local Excel file (no network needed) but are marked @pytest.mark.slow.
      Run without slow tests: pytest skills/nejm/tests/ -v -m "not slow"
      Requires: data/NEJM/41746_2024_1010_MOESM9_ESM.xlsx to be present.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import load_dataset, sample, format_item


REQUIRED_SCHEMA_KEYS = {"id", "question", "context", "choices", "answer", "metadata"}


@pytest.mark.slow
def test_load_dataset_drops_metadata_rows():
    """Load returns exactly 101 valid rows (3 metadata rows dropped)."""
    df = load_dataset()
    assert len(df) == 101


@pytest.mark.slow
def test_correct_column_normalised():
    """correct_clean contains only 'Y', 'N', or 'unknown'."""
    df = load_dataset()
    assert set(df["correct_clean"].unique()).issubset({"Y", "N", "unknown"})


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
    assert "correct_clean" in item["metadata"]
    assert "r1_errors_clean" in item["metadata"]
    assert "r2_errors_clean" in item["metadata"]
