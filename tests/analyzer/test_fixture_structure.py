"""Structural validation of the Claude Code fixture corpus.

These tests run without any LLM or network access. They validate that
the hand-authored fixtures under
``tests/analyzer/fixtures/claude_code/`` are internally consistent:

- Every skill directory has a parseable ``SKILL.md``.
- Every ``expected/<name>.json`` deserialises as a ``SkillExposure``.
- The skill directory name, the ``name`` field in the SKILL.md
  frontmatter, and the ``skill_name`` in the expected file all agree.
- Every finding in an expected file references a ``file_path`` that
  exists in the corresponding skill directory and a ``snippet`` that
  appears as a substring of that file.

These tests catch authoring mistakes (typos, renames, stale expected
files) at the earliest possible moment — before the analyzer harness
is built, and without burning any LLM calls. They do NOT validate
that the analyzer would actually produce the expected findings; that
is the job of the end-to-end harness introduced in a later phase.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from limenex.analyzer.schema import SkillExposure


FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "claude_code"
SKILLS_DIR = FIXTURES_ROOT / "skills"
EXPECTED_DIR = FIXTURES_ROOT / "expected"


def _skill_dirs() -> list[Path]:
    """Return all skill directories under fixtures/claude_code/skills/."""
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())


def _parse_skill_md(skill_md_path: Path) -> tuple[dict, str]:
    """Parse a SKILL.md file into (frontmatter_dict, body_markdown)."""
    text = skill_md_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{skill_md_path} does not start with YAML frontmatter")
    # Split off the YAML block between the first two '---' lines.
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{skill_md_path} has malformed frontmatter")
    frontmatter = yaml.safe_load(parts[1]) or {}
    body = parts[2].lstrip("\n")
    return frontmatter, body


@pytest.fixture(scope="module")
def skill_dirs() -> list[Path]:
    dirs = _skill_dirs()
    if not dirs:
        pytest.skip("No fixtures under tests/analyzer/fixtures/claude_code/skills/")
    return dirs


def test_skills_directory_exists():
    assert SKILLS_DIR.is_dir(), f"Missing fixtures directory: {SKILLS_DIR}"


def test_expected_directory_exists():
    assert EXPECTED_DIR.is_dir(), f"Missing expected directory: {EXPECTED_DIR}"


def test_every_skill_has_skill_md(skill_dirs):
    for skill_dir in skill_dirs:
        skill_md = skill_dir / "SKILL.md"
        assert skill_md.is_file(), f"Missing SKILL.md in {skill_dir}"


def test_skill_md_frontmatter_parses(skill_dirs):
    for skill_dir in skill_dirs:
        frontmatter, _ = _parse_skill_md(skill_dir / "SKILL.md")
        assert isinstance(frontmatter, dict)
        assert "name" in frontmatter, f"{skill_dir}/SKILL.md missing 'name'"
        assert "description" in frontmatter, f"{skill_dir}/SKILL.md missing 'description'"


def test_skill_name_matches_directory(skill_dirs):
    for skill_dir in skill_dirs:
        frontmatter, _ = _parse_skill_md(skill_dir / "SKILL.md")
        assert frontmatter["name"] == skill_dir.name, (
            f"SKILL.md name {frontmatter['name']!r} does not match "
            f"directory name {skill_dir.name!r}"
        )


def test_every_skill_has_expected_json(skill_dirs):
    for skill_dir in skill_dirs:
        expected_path = EXPECTED_DIR / f"{skill_dir.name}.json"
        assert expected_path.is_file(), (
            f"Missing expected file for skill {skill_dir.name}: {expected_path}"
        )


def test_every_expected_json_has_corresponding_skill():
    if not EXPECTED_DIR.is_dir():
        pytest.skip("No expected directory")
    for expected_path in EXPECTED_DIR.glob("*.json"):
        skill_name = expected_path.stem
        skill_dir = SKILLS_DIR / skill_name
        assert skill_dir.is_dir(), (
            f"Expected file {expected_path.name} references non-existent "
            f"skill directory {skill_dir}"
        )


def test_expected_json_deserialises(skill_dirs):
    for skill_dir in skill_dirs:
        expected_path = EXPECTED_DIR / f"{skill_dir.name}.json"
        data = json.loads(expected_path.read_text(encoding="utf-8"))
        # Round-trip through the schema dataclass. Raises on any
        # structural problem.
        exposure = SkillExposure.from_dict(data)
        assert exposure.skill_name == skill_dir.name


def test_expected_declared_purpose_matches_skill_description(skill_dirs):
    for skill_dir in skill_dirs:
        frontmatter, _ = _parse_skill_md(skill_dir / "SKILL.md")
        expected_path = EXPECTED_DIR / f"{skill_dir.name}.json"
        data = json.loads(expected_path.read_text(encoding="utf-8"))
        assert data["declared_purpose"] == frontmatter["description"], (
            f"{skill_dir.name}: expected.declared_purpose differs from "
            f"SKILL.md description. Keep them identical so the analyzer "
            f"harness can trust either source."
        )


def test_finding_evidence_references_existing_files(skill_dirs):
    for skill_dir in skill_dirs:
        expected_path = EXPECTED_DIR / f"{skill_dir.name}.json"
        data = json.loads(expected_path.read_text(encoding="utf-8"))
        for i, finding in enumerate(data["findings"]):
            evidence = finding["evidence"]
            file_path = evidence["file_path"]
            # file_path is relative to the skill directory.
            referenced = skill_dir / file_path
            assert referenced.is_file(), (
                f"{skill_dir.name} finding[{i}] evidence.file_path "
                f"{file_path!r} does not resolve to an existing file "
                f"(looked for {referenced})"
            )


def test_finding_snippet_appears_in_referenced_file(skill_dirs):
    for skill_dir in skill_dirs:
        expected_path = EXPECTED_DIR / f"{skill_dir.name}.json"
        data = json.loads(expected_path.read_text(encoding="utf-8"))
        for i, finding in enumerate(data["findings"]):
            evidence = finding["evidence"]
            file_path = evidence["file_path"]
            snippet = evidence["snippet"]
            referenced = skill_dir / file_path
            content = referenced.read_text(encoding="utf-8")

            def _normalise(text: str) -> str:
                text = text.replace("\r\n", "\n")
                return "\n".join(line.lstrip() for line in text.split("\n"))

            content_n = _normalise(content)
            snippet_n = _normalise(snippet)

            assert snippet_n in content_n, (
                f"{skill_dir.name} finding[{i}] evidence.snippet is not "
                f"a substring of {file_path} (after leading-whitespace "
                f"normalisation). Either the snippet is stale or the "
                f"script has drifted."
            )


def test_affected_components_reference_existing_files(skill_dirs):
    for skill_dir in skill_dirs:
        expected_path = EXPECTED_DIR / f"{skill_dir.name}.json"
        data = json.loads(expected_path.read_text(encoding="utf-8"))
        for i, finding in enumerate(data["findings"]):
            for component in finding["affected_components"]:
                # affected_components are paths relative to the skill
                # directory, same convention as evidence.file_path.
                referenced = skill_dir / component
                assert referenced.is_file(), (
                    f"{skill_dir.name} finding[{i}] affected_component "
                    f"{component!r} does not resolve to an existing file"
                )
