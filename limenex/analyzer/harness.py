"""Skill discovery and analysis orchestration for the analyzer.

``discover_skills(path, manifest_name)`` walks a directory for
plugin-style skill frameworks that follow the one-manifest-per-directory
convention (Claude Code ``SKILL.md``, and the same shape is expected
for Codex / OpenCode / LangGraph skills with their respective manifest
filenames). Returns ``Skill`` objects the harness feeds to an
``LLMProvider``.

``analyze_skill(skill, provider, *, framework)`` analyses one
discovered skill against one provider and returns a ``SkillExposure``.
Raises ``AnalyzeSkillError`` on any failure.

Frameworks with fundamentally different shapes (e.g. MCP, where one
JSON manifest enumerates many tools) need their own discovery
function; this one is only appropriate for the per-directory-plugin
pattern.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from limenex.analyzer.prompts import build_system_prompt, build_user_prompt
from limenex.analyzer.providers import LLMProvider, LLMProviderError
from limenex.analyzer.schema import Evidence, Finding, SkillExposure


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


class AnalyzeSkillError(Exception):
    """Raised by ``analyze_skill`` on any per-skill failure.

    Covers payload build, LLM transport, JSON parse, and response
    shape failures. Carries ``skill_name`` so aggregating callers can
    record which skill failed without re-parsing the message.
    """

    def __init__(self, skill_name: str, message: str) -> None:
        self.skill_name = skill_name
        self.reason = message
        super().__init__(f"{skill_name}: {message}")


# Maps model-emitted category strings to rule_id prefixes.
_CATEGORY_TO_PREFIX: dict[str, str] = {
    "finance": "FIN",
    "filesystem": "FS",
    "comm": "COMM",
    "web": "WEB",
    "injection": "INJ",
    "flow": "FLOW",
    "mismatch": "MISMATCH",
}


def _build_payload(skill: Skill) -> str:
    """Concatenate a skill's files into the BEGIN/END payload format.

    Files are emitted in ``Skill.files`` order (manifest first, rest
    alphabetical). Each file is prefixed with a
    ``=== FILE: <relpath> ===`` line. Relative paths use POSIX
    separators so payload hashes are stable across host OSes.

    Size-cap enforcement and BEGIN/END marker-collision detection are
    delegated to ``build_user_prompt``, which raises ``ValueError`` on
    either condition.
    """
    parts: list[str] = []
    for path in skill.files:
        relpath = path.relative_to(skill.root).as_posix()
        content = path.read_text(encoding="utf-8")
        parts.append(f"=== FILE: {relpath} ===\n{content}")
    return "\n\n".join(parts)


async def _complete_with_retry(
    provider: LLMProvider, messages: list[dict[str, str]]
) -> str:
    """Call ``provider.complete`` with a single retry on ``LLMProviderError``.

    Fixed 2-second sleep between attempts. Other exceptions bubble
    unchanged.
    """
    try:
        return await provider.complete(
            messages, response_format={"type": "json_object"}
        )
    except LLMProviderError:
        await asyncio.sleep(2.0)
        return await provider.complete(
            messages, response_format={"type": "json_object"}
        )


def _reshape_finding(
    raw: dict[str, Any], skill_name: str, counters: dict[str, int]
) -> Finding:
    """Convert a model-emitted finding dict into a schema ``Finding``.

    The model emits ``category`` and ``reasoning`` at finding-level;
    the schema nests both inside ``Evidence``. This function performs
    the re-nesting and stamps ``rule_id`` (per-skill sequential,
    format ``<PREFIX>###``) and ``affected_components``
    (``[skill_name]``). ``policy_scaffold`` is always ``None``.

    Raises ``KeyError`` on missing required fields, ``ValueError`` on
    unknown category.
    """
    category = raw["category"]
    try:
        prefix = _CATEGORY_TO_PREFIX[category]
    except KeyError as e:
        raise ValueError(f"unknown finding category {category!r}") from e

    counters[prefix] = counters.get(prefix, 0) + 1
    rule_id = f"{prefix}{counters[prefix]:03d}"

    raw_evidence = raw["evidence"]
    line_range_raw = raw_evidence.get("line_range")
    line_range: tuple[int, int] | None = None
    if line_range_raw is not None:
        line_range = (int(line_range_raw[0]), int(line_range_raw[1]))

    evidence = Evidence(
        category=category,
        snippet=raw_evidence["snippet"],
        file_path=raw_evidence["file_path"],
        reasoning=raw["reasoning"],
        line_range=line_range,
    )

    return Finding(
        rule_id=rule_id,
        type=raw["type"],
        severity=raw["severity"],
        confidence=raw["confidence"],
        affected_components=[skill_name],
        evidence=evidence,
        recommendation=raw["recommendation"],
        policy_scaffold=None,
    )


def _parse_response(raw_json: str, skill: Skill) -> SkillExposure:
    """Parse the model's JSON output into a ``SkillExposure``.

    Raises ``AnalyzeSkillError`` tagged ``parse_failed`` on JSON
    decode failure, or ``invalid_response_shape`` on missing/wrong
    fields. The offending content is truncated to 500 chars in the
    exception message.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        excerpt = raw_json[:500]
        raise AnalyzeSkillError(
            skill.name,
            f"parse_failed: model response was not valid JSON ({e.msg}); "
            f"excerpt: {excerpt!r}",
        ) from e

    if not isinstance(data, dict):
        raise AnalyzeSkillError(
            skill.name,
            f"invalid_response_shape: top-level JSON is {type(data).__name__}, "
            f"expected object",
        )

    try:
        declared_purpose = data["declared_purpose"]
        raw_findings = data["findings"]
        counters: dict[str, int] = {}
        findings = [_reshape_finding(f, skill.name, counters) for f in raw_findings]
    except (KeyError, TypeError, ValueError) as e:
        raise AnalyzeSkillError(
            skill.name, f"invalid_response_shape: {e}"
        ) from e

    return SkillExposure(
        skill_name=skill.name,
        skill_path=str(skill.root),
        declared_purpose=declared_purpose,
        findings=findings,
    )


async def analyze_skill(
    skill: Skill,
    provider: LLMProvider,
    *,
    framework: str = "claude_code",
) -> SkillExposure:
    """Analyze a single discovered skill against a provider.

    Orchestrates payload construction, prompt rendering, the LLM call
    (with a single retry on transport failure), JSON parse, and
    reshape into a ``SkillExposure``.

    Parameters
    ----------
    skill
        A ``Skill`` produced by ``discover_skills``.
    provider
        An ``LLMProvider`` implementation. The caller owns its
        lifecycle (e.g. entering / exiting its async context manager).
    framework
        Skill framework identifier passed to prompt rendering.
        Default ``"claude_code"``. Must be a key known to
        ``build_system_prompt``.

    Returns
    -------
    SkillExposure
        Structured analysis of the skill.

    Raises
    ------
    AnalyzeSkillError
        On payload build failure (size cap, marker collision), LLM
        transport failure after retry, JSON decode failure, or
        response shape error. Other exceptions (e.g. unknown
        framework) propagate unchanged.
    """
    try:
        payload = _build_payload(skill)
        system_prompt = build_system_prompt(framework)
        user_prompt = build_user_prompt(payload)
    except ValueError as e:
        raise AnalyzeSkillError(skill.name, f"payload_rejected: {e}") from e

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_response = await _complete_with_retry(provider, messages)
    except LLMProviderError as e:
        raise AnalyzeSkillError(skill.name, f"provider_failed: {e}") from e

    return _parse_response(raw_response, skill)