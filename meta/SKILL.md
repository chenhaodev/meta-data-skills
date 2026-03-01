---
name: meddata-build
description: Build a new med-data agent-skill. Usage: /meddata:build <slug> (e.g., healthbench, home-hospital, medbench-2026, medxpertqa, nejm)
---

# Med-Data Skill Builder

You are building a specialized Claude Code agent-skill for a medical dataset.
Project root: /Users/chenhao/MyOpenCode/meta-data-skills

## Step 1: Resolve Dataset

Read `TARGET.md` from the project root. Find the `##` section whose `slug:` field matches `$ARGUMENTS`.

Extract: name, modality, tasks, access, hf_repo (if present), local_file or local_dir (if present), license, difficulty.

If no entry is found, stop and print:
```
ERROR: No TARGET.md entry for '<slug>'. Add a YAML block to TARGET.md first.
Available slugs: (list all slug: values from TARGET.md)
```

## Step 2: Scaffold

Run:
```bash
cd /Users/chenhao/MyOpenCode/meta-data-skills
python meta/scaffold.py $ARGUMENTS
```

Confirm: `skills/$ARGUMENTS/` exists with SKILL.md, pipeline.py, stubs.py, tests/test_pipeline.py.

## Step 3: Gather Sources (Two-Phase)

**Phase 1:** Read `data/$ARGUMENTS/README.md` (if it exists). Extract:
- Dataset description and purpose
- Paper links and DOI
- HuggingFace repo links
- Dataset size and splits
- Task types and subtasks

**Phase 2 (if Phase 1 yielded fewer than 3 concrete facts):**
Web search: `"<name> dataset" site:huggingface.co OR site:arxiv.org`
Extract from top 3 results: dataset card info, paper abstract, known subtasks, known issues.

## Step 4: Fill SKILL.md

Open `skills/$ARGUMENTS/SKILL.md`. Replace each `<!-- TODO -->` section:

- **Dataset Overview**: Full name, creator institution, year, total examples, modality, languages, paper citation
- **Clinical Domain & AI Relevance**: Medical specialties, disease categories, task definitions, current SOTA scores on this benchmark, why it matters for LLM evaluation today
- **Access & Licensing**: HuggingFace repo, local path `data/$ARGUMENTS/`, license, citation requirement, any access restrictions
- **Key Splits & Subsets**: Markdown table of split sizes; for agentic datasets (e.g., medbench-2026), list all subtasks with sample counts and README links
- **Recommended Use Cases**: Specific, actionable guidance for (a) benchmarking, (b) fine-tuning data selection, (c) few-shot prompt selection, (d) agentic pipeline evaluation
- **Pitfalls & Gotchas**: 3-5 specific known issues with this dataset (contamination risk, annotation noise, label skew, access restrictions, multimodal caveats)
- **Code Patterns**: 3 complete, runnable code snippets using pipeline.py functions

## Step 5: Fill pipeline.py

Open `skills/$ARGUMENTS/pipeline.py`. Implement all three functions:

### load_dataset()

Choose implementation based on `access` field:

**HuggingFace access:**
```python
from datasets import load_dataset as hf_load

def load_dataset(split: str = "test") -> pd.DataFrame:
    ds = hf_load("<hf_repo>", split=split)
    return ds.to_pandas()
```

**Local xlsx:**
```python
def load_dataset(split: str = "test") -> pd.DataFrame:
    return pd.read_excel(DATA_DIR / "<filename>.xlsx")
```

**Local jsonl (single file):**
```python
import json

def load_dataset(split: str = "test") -> pd.DataFrame:
    rows = [json.loads(line) for line in (DATA_DIR / "<task>.jsonl").open()]
    return pd.DataFrame(rows)
```

**Local jsonl (multiple subtasks — e.g., medbench-2026):**
```python
import json

def load_dataset(split: str = "test", subtask: str = None) -> pd.DataFrame:
    jsonl_dir = DATA_DIR / "MedBench_Agent" / "TEST"
    files = [jsonl_dir / f"{subtask}.jsonl"] if subtask else sorted(jsonl_dir.glob("*.jsonl"))
    rows = []
    for f in files:
        task_name = f.stem
        for line in f.open():
            item = json.loads(line)
            item["_subtask"] = task_name
            rows.append(item)
    return pd.DataFrame(rows)
```

### format_item()

Map dataset-specific field names to the common schema. Inspect 2-3 example rows first to identify field names. Return:
```python
{
    "id": str,       # unique identifier
    "question": str, # question or prompt
    "context": str,  # background context (empty string if none)
    "choices": list, # MCQ options (empty list for open-ended)
    "answer": str,   # ground truth answer
    "metadata": dict # extra fields (source, difficulty, subtask, etc.)
}
```

## Step 6: Fill stubs.py

Open `skills/$ARGUMENTS/stubs.py`. Implement `score()` based on task type:

- **MCQ (exact match):**
  ```python
  def score(prediction: str, ground_truth: str) -> float:
      return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0
  ```

- **Open generation (substring match):**
  ```python
  def score(prediction: str, ground_truth: str) -> float:
      return 1.0 if ground_truth.strip().lower() in prediction.strip().lower() else 0.0
  ```

Add at top of file:
```python
from pipeline import format_item
```

`run_benchmark()` and `compare_models()` are already implemented in the template — do not change them.

## Step 7: Update tests/test_pipeline.py

Review the generated `skills/$ARGUMENTS/tests/test_pipeline.py`. If load_dataset() requires network access (HuggingFace), ensure all tests have `@pytest.mark.slow` and add a comment explaining how to run with mocked data for CI.

## Step 8: Update TARGET.md Status

In TARGET.md, change `status: pending` to `status: generated` for `$ARGUMENTS`.

## Step 9: Confirm

Print:
```
✓ Skill built: skills/$ARGUMENTS/
  Dataset:  <name>
  Access:   <method>
  Tasks:    <task list>
  Invoke:   /meddata:<slug>
  Test:     pytest skills/$ARGUMENTS/tests/ -v -m "not slow"
```
