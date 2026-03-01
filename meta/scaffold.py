#!/usr/bin/env python3
"""
scaffold.py — Med-data skill scaffolding CLI.

Usage:
    python meta/scaffold.py <slug>

Examples:
    python meta/scaffold.py healthbench
    python meta/scaffold.py home-hospital
    python meta/scaffold.py medbench-2026

Creates skills/<slug>/ with:
    SKILL.md, pipeline.py, stubs.py, tests/test_pipeline.py, tests/__init__.py
"""

import sys
from pathlib import Path
from string import Template

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = Path(__file__).parent / "templates"
SKILLS_DIR = REPO_ROOT / "skills"

TEMPLATE_MAP = {
    "SKILL.md.tmpl": "SKILL.md",
    "pipeline.py.tmpl": "pipeline.py",
    "stubs.py.tmpl": "stubs.py",
    "test_pipeline.py.tmpl": "tests/test_pipeline.py",
}


def scaffold(slug: str) -> Path:
    """
    Scaffold a new med-data skill folder.

    Args:
        slug: Dataset slug (kebab-case). e.g., "healthbench", "home-hospital"

    Returns:
        Path to created skill folder.

    Raises:
        SystemExit: If skill folder already exists.
    """
    skill_dir = SKILLS_DIR / slug

    if skill_dir.exists():
        print(f"Error: skill '{slug}' already exists at {skill_dir}", file=sys.stderr)
        sys.exit(1)

    name = slug.replace("-", " ").title()
    substitutions = {"slug": slug, "name": name}

    skill_dir.mkdir(parents=True)
    (skill_dir / "tests").mkdir()
    (skill_dir / "tests" / "__init__.py").touch()

    for tmpl_name, output_name in TEMPLATE_MAP.items():
        tmpl_path = TEMPLATES_DIR / tmpl_name
        if not tmpl_path.exists():
            print(f"Warning: template {tmpl_name} not found, skipping", file=sys.stderr)
            continue
        raw = tmpl_path.read_text()
        rendered = Template(raw).safe_substitute(substitutions)
        output_path = skill_dir / output_name
        output_path.write_text(rendered)

    print(f"Scaffolded: {skill_dir}")
    print(f"Next: /meddata:build {slug} to fill domain content")
    return skill_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python meta/scaffold.py <slug>", file=sys.stderr)
        sys.exit(1)
    scaffold(sys.argv[1])
