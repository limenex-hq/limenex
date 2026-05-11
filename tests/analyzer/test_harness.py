"""Tests for the analyzer harness: discovery and analysis."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from limenex.analyzer.harness import Skill, discover_skills


# ---------- symlink privilege probe ----------

def _can_symlink(tmp_path: Path) -> bool:
    """Probe whether this process can create directory symlinks in ``tmp_path``."""
    probe_target = tmp_path / "_probe_target"
    probe_link = tmp_path / "_probe_link"
    probe_target.mkdir()
    try:
        os.symlink(probe_target, probe_link, target_is_directory=True)
        probe_link.unlink()
        probe_target.rmdir()
        return True
    except (OSError, NotImplementedError):
        probe_target.rmdir()
        return False


@pytest.fixture
def symlink_or_skip(tmp_path):
    """Skip the test if this environment cannot create symlinks."""
    if not _can_symlink(tmp_path):
        pytest.skip("No symlink privilege on this platform/setup.")


# ---------- helpers ----------

def _make_skill(
    root: Path, name: str, extra_files: dict[str, str] | None = None
) -> Path:
    """Create a skill dir at ``root/name`` with a SKILL.md and extra files.

    Returns the skill's root directory.
    """
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\n---\n# {name}\n", encoding="utf-8"
    )
    for rel_path, content in (extra_files or {}).items():
        full = skill_dir / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
    return skill_dir


# ---------- basic discovery ----------

def test_empty_directory_returns_no_skills(tmp_path):
    assert discover_skills(tmp_path) == []


def test_single_skill_flat_layout(tmp_path):
    _make_skill(tmp_path, "foo", {"scripts/main.py": "print('hi')\n"})
    skills = discover_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "foo"
    assert skills[0].skill_md_path == (tmp_path / "foo" / "SKILL.md").resolve()
    assert skills[0].root == (tmp_path / "foo").resolve()


def test_nested_claude_code_layout(tmp_path):
    base = tmp_path / "project" / ".claude" / "skills"
    _make_skill(base, "bar", {"scripts/b.py": "x = 1\n"})
    skills = discover_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "bar"


def test_multiple_skills_discovered(tmp_path):
    for n in ("alpha", "beta", "gamma"):
        _make_skill(tmp_path, n, {"scripts/s.py": f"# {n}\n"})
    skills = discover_skills(tmp_path)
    assert {s.name for s in skills} == {"alpha", "beta", "gamma"}
    assert len(skills) == 3


# ---------- file ordering contract ----------

def test_manifest_is_first_in_files(tmp_path):
    _make_skill(
        tmp_path,
        "foo",
        {
            "scripts/a.py": "a\n",
            "scripts/z.py": "z\n",
            "README.md": "readme\n",
        },
    )
    (skill,) = discover_skills(tmp_path)
    assert skill.files[0] == skill.skill_md_path


def test_non_manifest_files_sorted_alphabetically(tmp_path):
    _make_skill(
        tmp_path,
        "foo",
        {
            "scripts/z.py": "z\n",
            "scripts/a.py": "a\n",
            "README.md": "readme\n",
            "docs/guide.md": "guide\n",
        },
    )
    (skill,) = discover_skills(tmp_path)
    rel = [p.relative_to(skill.root).as_posix() for p in skill.files[1:]]
    assert rel == sorted(rel)


# ---------- UTF-8 vs binary ----------

def test_utf8_files_of_various_extensions_included(tmp_path):
    _make_skill(
        tmp_path,
        "foo",
        {
            "scripts/main.py": "py\n",
            "config.yaml": "yaml\n",
            "notes.txt": "txt\n",
            "no_extension": "plain\n",
        },
    )
    (skill,) = discover_skills(tmp_path)
    rel = {p.relative_to(skill.root).as_posix() for p in skill.files}
    assert rel == {
        "SKILL.md",
        "scripts/main.py",
        "config.yaml",
        "notes.txt",
        "no_extension",
    }
    assert skill.skipped_binaries == []


def test_binary_files_go_to_skipped_binaries(tmp_path):
    skill_dir = _make_skill(tmp_path, "foo", {"scripts/main.py": "ok\n"})
    (skill_dir / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
    (skill_dir / "blob.bin").write_bytes(b"\xff\xfe\xfd\xfc")

    (skill,) = discover_skills(tmp_path)
    rel_text = {p.relative_to(skill.root).as_posix() for p in skill.files}
    rel_bin = {p.relative_to(skill.root).as_posix() for p in skill.skipped_binaries}
    assert rel_text == {"SKILL.md", "scripts/main.py"}
    assert rel_bin == {"logo.png", "blob.bin"}


def test_skill_md_itself_not_utf8_raises(tmp_path):
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_bytes(b"\xff\xfe\xfd not utf-8")
    with pytest.raises(UnicodeDecodeError, match="not valid UTF-8"):
        discover_skills(tmp_path)


# ---------- path validation ----------

def test_nonexistent_path_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_skills(tmp_path / "does-not-exist")


def test_path_is_file_not_dir_raises_notadirectory(tmp_path):
    f = tmp_path / "not-a-dir.txt"
    f.write_text("hi", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        discover_skills(f)


# ---------- symlinks ----------

def test_symlink_to_directory_is_followed(tmp_path, symlink_or_skip):
    real = tmp_path / "real"
    _make_skill(real, "foo", {"scripts/s.py": "x\n"})
    link = tmp_path / "link"
    os.symlink(real, link, target_is_directory=True)

    skills = discover_skills(link)
    assert len(skills) == 1
    assert skills[0].name == "foo"


def test_aliased_symlink_skill_reported_once(tmp_path, symlink_or_skip):
    real_skills = tmp_path / "real-skills"
    real_skills.mkdir()
    _make_skill(real_skills, "foo", {"scripts/s.py": "x\n"})
    alias = tmp_path / "aliased-skills"
    os.symlink(real_skills, alias, target_is_directory=True)

    skills = discover_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "foo"


# ---------- duplicate names ----------

def test_duplicate_skill_names_allowed(tmp_path):
    _make_skill(tmp_path / "group-a", "foo", {"scripts/s.py": "a\n"})
    _make_skill(tmp_path / "group-b", "foo", {"scripts/s.py": "b\n"})
    skills = discover_skills(tmp_path)
    assert len(skills) == 2
    assert {s.name for s in skills} == {"foo"}
    assert len({s.root for s in skills}) == 2


# ---------- subdirectory walking ----------

def test_subdirectories_of_skill_root_are_walked(tmp_path):
    _make_skill(
        tmp_path,
        "foo",
        {
            "scripts/main.py": "x\n",
            "references/data.json": '{"k": 1}\n',
            "templates/email.txt": "hello\n",
            "nested/deep/inside.py": "y\n",
        },
    )
    (skill,) = discover_skills(tmp_path)
    rel = {p.relative_to(skill.root).as_posix() for p in skill.files}
    assert rel == {
        "SKILL.md",
        "scripts/main.py",
        "references/data.json",
        "templates/email.txt",
        "nested/deep/inside.py",
    }


# ---------- custom manifest name ----------

def test_custom_manifest_name(tmp_path):
    skill_dir = tmp_path / "codex-skill"
    skill_dir.mkdir()
    (skill_dir / "AGENT.md").write_text("---\nname: codex\n---\n", encoding="utf-8")
    (skill_dir / "main.py").write_text("x\n", encoding="utf-8")

    skills = discover_skills(tmp_path, manifest_name="AGENT.md")
    assert len(skills) == 1
    assert skills[0].name == "codex-skill"
    assert skills[0].skill_md_path.name == "AGENT.md"


def test_default_manifest_name_does_not_find_other_conventions(tmp_path):
    skill_dir = tmp_path / "codex-skill"
    skill_dir.mkdir()
    (skill_dir / "AGENT.md").write_text("x\n", encoding="utf-8")
    skills = discover_skills(tmp_path)
    assert skills == []


# ---------- Skill dataclass sanity ----------

def test_skill_all_paths_are_absolute(tmp_path):
    _make_skill(tmp_path, "foo", {"scripts/s.py": "x\n"})
    (skill,) = discover_skills(tmp_path)
    assert skill.skill_md_path.is_absolute()
    assert skill.root.is_absolute()
    for p in skill.files:
        assert p.is_absolute()
    for p in skill.skipped_binaries:
        assert p.is_absolute()