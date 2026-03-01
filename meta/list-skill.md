---
name: meddata-list
description: List all available med-data datasets and their skill generation status.
---

# Med-Data Dataset Listing

Read `TARGET.md` from /Users/chenhao/MyOpenCode/meta-data-skills.

Parse all `##` sections. For each section, extract: slug, name, tasks, access, difficulty, status.

For each dataset, check whether `skills/<slug>/SKILL.md` exists on disk.
If it exists, mark as "generated" regardless of the TARGET.md status field.

Print a formatted table:

```
Available Med-Data Datasets
===========================

SLUG             NAME                  STATUS      ACCESS        TASKS                      DIFFICULTY
healthbench      HealthBench           generated   huggingface   MCQ, generation            hard
home-hospital    Hospital@Home         pending     local         clinical-prediction        medium
medbench-2026    MedBench 2026         pending     local         agentic (14 subtasks)      hard
medxpertqa       MedXpertQA            pending     huggingface   MCQ, expert-reasoning      hard
nejm             NEJM Annotations      pending     local         annotation, NLI            medium

Total: 5 datasets (1 generated, 4 pending)

To build a skill:  /meddata:build <slug>
To use a skill:    /meddata:<slug>
```
