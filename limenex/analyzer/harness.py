"""Skill discovery and analysis orchestration for the analyzer.

``discover_skills(path, manifest_name)`` walks a directory for
plugin-style skill frameworks that follow the one-manifest-per-directory
convention (Claude Code ``SKILL.md``, and the same shape is expected
for Codex / OpenCode / LangGraph skills with their respective manifest
filenames). Returns ``Skill`` objects the harness feeds to an
``LLMProvider``.

Frameworks with fundamentally different shapes (e.g. MCP, where one
JSON manifest enumerates many tools) need their own discovery
function; this one is only appropriate for the per-directory-plugin
pattern.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    """A discovered skill plugin ready for analysis.

    All paths are absolute and resolved (symlinks dereferenced).

    Attributes
    ----------
    name
        Parent directory name of the manifest.
    skill_md_path
        Absolute resolved path to the skill's manifest file.
    root
        Absolute resolved path to the skill's root directory
        (``skill_md_path.parent``).
    files
        All UTF-8-decodable files under ``root``. ``skill_md_path`` is
        always first (payload format requires manifest first); remaining
        files are sorted alphabetically by path relative to ``root``.
    skipped_binaries
        Files under ``root`` that failed UTF-8 decoding. Not included
        in the analyzer payload.
    """

    name: str
    skill_md_path: Path
    root: Path
    files: list[Path]
    skipped_binaries: list[Path]


def _is_utf8_text(path: Path) -> bool:
    """Return True if ``path`` is decodable as UTF-8."""
    try:
        path.read_text(encoding="utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _collect_skill_files(
    root: Path, skill_md_path: Path
) -> tuple[list[Path], list[Path]]:
    """Partition files under ``root`` into UTF-8 vs binary.

    Returns ``(utf8_files, binary_files)``. ``utf8_files`` has
    ``skill_md_path`` first; remaining files sorted alphabetically by
    path relative to ``root``.
    """
    utf8_files: list[Path] = []
    binary_files: list[Path] = []

    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for fname in filenames:
            entry = Path(dirpath, fname).resolve()
            if _is_utf8_text(entry):
                utf8_files.append(entry)
            else:
                binary_files.append(entry)

    # Partition out the manifest so it can be placed first.
    non_manifest = [p for p in utf8_files if p != skill_md_path]
    non_manifest.sort(key=lambda p: p.relative_to(root).as_posix())
    binary_files.sort(key=lambda p: p.relative_to(root).as_posix())

    return [skill_md_path, *non_manifest], binary_files


def discover_skills(
    path: Path, manifest_name: str = "SKILL.md"
) -> list[Skill]:
    """Discover every skill plugin under ``path``.

    Walks ``path`` recursively looking for directories containing
    ``manifest_name`` (default ``"SKILL.md"``, the Claude Code
    convention). Each such directory is treated as one skill plugin
    whose root is the directory containing the manifest. Symlinks to
    directories are followed; skills reached via multiple paths
    (symlink cycles, aliased trees) are reported once.

    Parameters
    ----------
    path
        Directory to scan.
    manifest_name
        Filename that marks a skill root. Defaults to ``"SKILL.md"``.

    Returns
    -------
    list[Skill]
        One per discovered manifest, in traversal order.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    NotADirectoryError
        If ``path`` is not a directory.
    UnicodeDecodeError
        If a discovered manifest cannot be decoded as UTF-8. A skill
        without a readable manifest is unanalyzable; failure is hard
        rather than recorded as a skipped binary.
    """
    if not path.exists():
        raise FileNotFoundError(f"Discovery path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Discovery path is not a directory: {path}")

    seen_roots: set[Path] = set()
    skills: list[Skill] = []

    for dirpath, _, filenames in os.walk(path, followlinks=True):
        if manifest_name not in filenames:
            continue
        manifest = Path(dirpath, manifest_name).resolve()
        root = manifest.parent
        if root in seen_roots:
            continue
        seen_roots.add(root)

        if not _is_utf8_text(manifest):
            raise UnicodeDecodeError(
                "utf-8",
                b"",
                0,
                1,
                f"{manifest_name} is not valid UTF-8: {manifest}",
            )

        files, binaries = _collect_skill_files(root, manifest)
        skills.append(
            Skill(
                name=root.name,
                skill_md_path=manifest,
                root=root,
                files=files,
                skipped_binaries=binaries,
            )
        )

    return skills