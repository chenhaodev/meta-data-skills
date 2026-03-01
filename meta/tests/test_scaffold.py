"""Tests for meta/scaffold.py — run before implementing scaffold.py."""
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
SCAFFOLD = REPO_ROOT / "meta" / "scaffold.py"
SKILLS_DIR = REPO_ROOT / "skills"


def _cleanup(slug: str) -> None:
    shutil.rmtree(SKILLS_DIR / slug, ignore_errors=True)


def setup_function():
    _cleanup("test-dataset")
    _cleanup("test-multi-word")


def teardown_function():
    _cleanup("test-dataset")
    _cleanup("test-multi-word")


def test_scaffold_creates_folder():
    result = subprocess.run(
        [sys.executable, str(SCAFFOLD), "test-dataset"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert (SKILLS_DIR / "test-dataset").is_dir()


def test_scaffold_creates_skill_md():
    subprocess.run([sys.executable, str(SCAFFOLD), "test-dataset"],
                   check=True, cwd=REPO_ROOT)
    skill_md = SKILLS_DIR / "test-dataset" / "SKILL.md"
    assert skill_md.exists()
    content = skill_md.read_text()
    assert "test-dataset" in content
    assert "TODO" in content


def test_scaffold_creates_pipeline_py():
    subprocess.run([sys.executable, str(SCAFFOLD), "test-dataset"],
                   check=True, cwd=REPO_ROOT)
    assert (SKILLS_DIR / "test-dataset" / "pipeline.py").exists()


def test_scaffold_creates_stubs_py():
    subprocess.run([sys.executable, str(SCAFFOLD), "test-dataset"],
                   check=True, cwd=REPO_ROOT)
    assert (SKILLS_DIR / "test-dataset" / "stubs.py").exists()


def test_scaffold_creates_tests_dir_with_file():
    subprocess.run([sys.executable, str(SCAFFOLD), "test-dataset"],
                   check=True, cwd=REPO_ROOT)
    tests_dir = SKILLS_DIR / "test-dataset" / "tests"
    assert tests_dir.is_dir()
    assert (tests_dir / "test_pipeline.py").exists()


def test_scaffold_rejects_duplicate():
    subprocess.run([sys.executable, str(SCAFFOLD), "test-dataset"],
                   check=True, cwd=REPO_ROOT)
    result = subprocess.run(
        [sys.executable, str(SCAFFOLD), "test-dataset"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode != 0


def test_scaffold_kebab_slug_preserved():
    subprocess.run([sys.executable, str(SCAFFOLD), "test-multi-word"],
                   check=True, cwd=REPO_ROOT)
    assert (SKILLS_DIR / "test-multi-word").is_dir()
    content = (SKILLS_DIR / "test-multi-word" / "SKILL.md").read_text()
    assert "test-multi-word" in content
