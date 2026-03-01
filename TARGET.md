# Med-Data Dataset Registry

Each dataset has a YAML metadata block followed by prose notes.
To add a new dataset: add a new `##` section with YAML block, then run `/meddata:build <slug>`.

---

## HealthBench

```yaml
slug: healthbench
name: HealthBench
created_by: OpenAI
year: 2025
modality: text
tasks: [MCQ, open-generation]
access: huggingface
hf_repo: openai/healthbench
license: MIT
difficulty: hard
status: generated
```

Comprehensive medical QA benchmark. Covers physician-level questions across 26 specialties.
Links: https://huggingface.co/datasets/openai/healthbench

---

## HomeHospital

```yaml
slug: home-hospital
name: Hospital@Home
created_by: Levine et al. (Ann Intern Med 2020)
year: 2020
modality: text
tasks: [clinical-prediction, patient-simulation]
access: local
local_file: Hospital@Home-db.clinic.xlsx
license: research-only
difficulty: medium
status: generated
```

574 real Hospital@Home patients with demographics, chronic conditions, medications, HRQoL.
Local file: data/HomeHospital/Hospital@Home-db.clinic.xlsx

---

## MedBench2026

```yaml
slug: medbench-2026
name: MedBench 2026
created_by: MedBench consortium
year: 2026
modality: text
tasks: [agentic, MCQ, NLI, generation, tool-use]
access: local
local_dir: MedBench_Agent/TEST/
license: research-only
difficulty: hard
status: generated
```

14 agentic medical subtasks: MedCOT, MedCallAPI, MedCollab, MedDBOps, MedDecomp,
MedEthics, MedIntentID, MedLongConv, MedLongQA, MedPathPlan, MedReflect,
MedRetAPI, MedRoleAdapt, MedShield. Each subtask has a .jsonl test file and README.

---

## MedXpertQA

```yaml
slug: medxpertqa
name: MedXpertQA
created_by: TsinghuaC3I
year: 2025
modality: multimodal
tasks: [MCQ, expert-reasoning]
access: huggingface
hf_repo: TsinghuaC3I/MedXpertQA
license: CC-BY-4.0
difficulty: hard
status: pending
```

Expert-level medical QA designed to challenge frontier models. Multimodal (text + images).
Links: https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA

---

## NEJM

```yaml
slug: nejm
name: NEJM Clinical Annotations
created_by: NEJM / Nature npj Digital Medicine
year: 2024
modality: text
tasks: [annotation, NLI, expert-comparison]
access: local
local_file: 41746_2024_1010_MOESM9_ESM.xlsx
license: research-only
difficulty: medium
status: generated
```

Two human expert annotation sets for NEJM clinical cases.
DOI: https://www.nature.com/articles/s41746-024-01010-1
Local file: data/NEJM/41746_2024_1010_MOESM9_ESM.xlsx
