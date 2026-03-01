"""Data access layer for Hospital@Home dataset.

574 real Hospital@Home patients from the Levine et al. 2020 RCT (BWH/BWFH).
Local Excel file: data/HomeHospital/Hospital@Home-db.clinic.xlsx
Header is at row 8 (0-indexed) — do not use default header=0.

Columns: 140 total covering admission, conditions, medications, demographics,
functional status (ADL/IADL), HRQoL (EQ-5D), frailty (PRISMA), cognition (AD8),
mood (PHQ-2), and social support.
"""
import pandas as pd
from pathlib import Path

# The actual file lives in data/HomeHospital/ (note capitalisation)
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "HomeHospital"
XLSX_FILE = DATA_DIR / "Hospital@Home-db.clinic.xlsx"
EXCEL_HEADER_ROW = 8  # metadata rows precede the actual column headers

MED_COLS = [f"Current_Meds{i}" for i in range(1, 27)]
CONDITION_COLS = ["Condition_1", "Condition_2", "Condition_3", "Condition_4",
                  "Condition_5", "Condition_Extra"]

# Numeric code → label maps (from Excel metadata rows 3–6)
GENDER_MAP = {0: "female", 1: "male"}
RACE_MAP = {1: "white", 2: "black", 3: "latino", 4: "asian", 5: "multi/other"}
INSURANCE_MAP = {1: "private", 1.0: "private", 2: "medicare", 2.0: "medicare",
                 3: "medicaid", 3.0: "medicaid", 4: "medicare+medicaid", 4.0: "medicare+medicaid",
                 5: "other", 5.0: "other"}


def load_dataset(split: str = "test") -> pd.DataFrame:
    """Load Hospital@Home from the local Excel file.

    Args:
        split: Ignored — no train/val/test split exists; all 574 patients returned.

    Returns:
        DataFrame with 574 rows and 140 columns. Key columns:
          Biofourmis_ID, Age, Gender, Race_ethnicity, Condition_1–5,
          Current_Meds1–26, Morse_Fall_Risk_Score, PRISMA_tot, AD8_tot,
          EQ_A_VAS, ADL_A_tot, IADL_A_tot, PHQ_A_tot.
    """
    return pd.read_excel(XLSX_FILE, header=EXCEL_HEADER_ROW)


def sample(n: int, split: str = "test", seed: int = 42) -> pd.DataFrame:
    """Sample n rows from the dataset reproducibly."""
    df = load_dataset(split)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def format_item(item: dict) -> dict:
    """Normalize a Hospital@Home row to the common med-data schema.

    Constructs a natural-language patient simulation prompt as 'question',
    with the full clinical profile as 'context'. There is no ground-truth
    answer — this dataset is used for patient simulation, not Q&A.

    Returns:
        dict with keys: id, question, context, choices, answer, metadata
    """
    # Collect conditions (drop NaN)
    conditions = [
        str(item[c]) for c in CONDITION_COLS
        if item.get(c) and str(item[c]) != "nan"
    ]
    conditions_str = ", ".join(conditions) if conditions else "unspecified"

    # Collect medications (drop NaN)
    meds = [
        str(item[c]) for c in MED_COLS
        if item.get(c) and str(item[c]) != "nan"
    ]

    age = item.get("Age", "unknown")
    gender = GENDER_MAP.get(item.get("Gender"), str(item.get("Gender", "unknown")))
    race = RACE_MAP.get(item.get("Race_ethnicity"), str(item.get("Race_ethnicity", "unknown")))
    insurance = INSURANCE_MAP.get(item.get("Insurance"), str(item.get("Insurance", "unknown")))

    question = (
        f"You are simulating a Hospital@Home patient. "
        f"The patient is a {age}-year-old {gender} ({race}) with {insurance} insurance "
        f"presenting with: {conditions_str}. "
        f"They are on {len(meds)} outpatient medications. "
        f"How would you assess and manage this patient in a home hospital setting?"
    )

    context = (
        f"Chronic conditions: {item.get('Condition_Chronic_Count', '?')} | "
        f"Comorbidities: {item.get('Comorbidities_Count', '?')} | "
        f"Morse fall score: {item.get('Morse_Fall_Risk_Score', '?')} | "
        f"PRISMA frailty: {item.get('PRISMA_tot', '?')} | "
        f"AD8 cognition: {item.get('AD8_tot', '?')} | "
        f"EQ-VAS HRQoL: {item.get('EQ_A_VAS', '?')} | "
        f"ADL (admission): {item.get('ADL_A_tot', '?')} | "
        f"IADL (admission): {item.get('IADL_A_tot', '?')} | "
        f"PHQ-2 depression: {item.get('PHQ_A_tot', '?')} | "
        f"Lives alone: {item.get('Lives_alone', '?')} | "
        f"Education: {item.get('Education', '?')}"
    )

    return {
        "id": str(int(item.get("Biofourmis_ID", 0))),
        "question": question,
        "context": context,
        "choices": [],  # patient simulation — no MCQ options
        "answer": "",   # no ground-truth outcome label in this dataset
        "metadata": {
            "age": age,
            "gender": gender,
            "race_ethnicity": race,
            "conditions": conditions,
            "medications": meds,
            "morse_fall_score": item.get("Morse_Fall_Risk_Score"),
            "prisma_tot": item.get("PRISMA_tot"),
            "ad8_tot": item.get("AD8_tot"),
            "eq_vas": item.get("EQ_A_VAS"),
            "adl_admission": item.get("ADL_A_tot"),
            "iadl_admission": item.get("IADL_A_tot"),
            "adl_delta": item.get("Delta_ADL_MINUS30toA"),
            "phq2_tot": item.get("PHQ_A_tot"),
        },
    }
