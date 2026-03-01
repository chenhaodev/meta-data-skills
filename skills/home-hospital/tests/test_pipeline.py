"""Smoke tests for Hospital@Home pipeline.

Run: pytest skills/home-hospital/tests/ -v
Note: Tests load the local Excel file (no network needed) but are marked @pytest.mark.slow
      because openpyxl parsing of 574-row × 140-col file takes ~1–2 seconds.
      Run without slow tests: pytest skills/home-hospital/tests/ -v -m "not slow"
      For CI: ensure data/HomeHospital/Hospital@Home-db.clinic.xlsx is present.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import load_dataset, sample, format_item


REQUIRED_SCHEMA_KEYS = {"id", "question", "context", "choices", "answer", "metadata"}


@pytest.mark.slow
def test_load_dataset_returns_dataframe():
    """Load succeeds and returns non-empty DataFrame."""
    df = load_dataset(split="test")
    assert df is not None
    assert len(df) > 0


@pytest.mark.slow
def test_sample_returns_n_rows():
    """sample() returns exactly n rows (or dataset size if smaller)."""
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
