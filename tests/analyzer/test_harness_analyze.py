"""Tests for the analyzer harness: analyze_skill and analyze_directory.

Uses an in-file ``FakeLLMProvider`` that implements the ``LLMProvider``
protocol. Promote to ``conftest.py`` if a second test file ever needs it.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from limenex.analyzer.harness import (
    AnalyzeSkillError,
    Skill,
    analyze_directory,
    analyze_skill,
    discover_skills,
)
from limenex.analyzer.prompts import MAX_PAYLOAD_BYTES, system_prompt_hash
from limenex.analyzer.providers import LLMProviderError, ProviderFingerprint
from limenex.analyzer.schema import SkillExposure

# ---------- FakeLLMProvider ----------

ResponderResult = str | BaseException
Responder = Callable[[str], ResponderResult]


class FakeLLMProvider:
    """Protocol-compatible fake ``LLMProvider`` for harness tests.

    The ``responder`` callable receives the user message content and
    returns either a response string or an exception instance (which
    is raised). Return an ``LLMProviderError`` to exercise the retry
    path; return another exception to exercise propagation.

    Tracks ``call_count`` and ``max_in_flight`` so concurrency tests
    can assert semaphore behaviour.
    """

    def __init__(
        self,
        responder: Responder,
        *,
        model: str = "fake-model",
        temperature: float = 0.0,
    ) -> None:
        self._responder = responder
        self._model = model
        self._temperature = temperature
        self.call_count = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = asyncio.Lock()

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        async with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            self.call_count += 1
            user_msg = next(m["content"] for m in messages if m["role"] == "user")
            result = self._responder(user_msg)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, BaseException):
                raise result
            return result
        finally:
            async with self._lock:
                self.in_flight -= 1

    def fingerprint(self) -> ProviderFingerprint:
        return ProviderFingerprint(model=self._model, temperature=self._temperature)


# ---------- helpers ----------


def _make_skill_dir(
    root: Path, name: str, extra_files: dict[str, str] | None = None
) -> Path:
    """Create a skill dir at ``root/name`` with a SKILL.md and extra files."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill {name}.\n---\n# {name}\n",
        encoding="utf-8",
    )
    for rel_path, content in (extra_files or {}).items():
        full = skill_dir / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
    return skill_dir


def _discover_one(tmp_path: Path) -> Skill:
    """Discover exactly one skill under ``tmp_path``. Fail if not exactly one."""
    skills = discover_skills(tmp_path)
    assert len(skills) == 1, f"expected 1 skill, got {len(skills)}"
    return skills[0]


def _valid_response(
    *,
    declared_purpose: str = "Test skill.",
    findings: list[dict[str, Any]] | None = None,
) -> str:
    """Build a valid model response JSON string."""
    return json.dumps(
        {"declared_purpose": declared_purpose, "findings": findings or []}
    )


def _finding(
    category: str,
    *,
    snippet: str = "x = 1",
    file_path: str = "scripts/main.py",
    severity: str = "medium",
    confidence: str = "high",
    type_: str = "capability_risk",
) -> dict[str, Any]:
    """Build a model-emitted finding dict (pre-harness-reshape)."""
    return {
        "category": category,
        "type": type_,
        "severity": severity,
        "confidence": confidence,
        "evidence": {
            "snippet": snippet,
            "file_path": file_path,
        },
        "reasoning": f"Detected {category} capability.",
        "recommendation": f"Review {category} usage.",
    }


# ---------- analyze_skill ----------


async def test_analyze_skill_happy_path_returns_exposure(tmp_path):
    _make_skill_dir(tmp_path, "foo", {"scripts/main.py": "x = 1\n"})
    skill = _discover_one(tmp_path)
    provider = FakeLLMProvider(
        lambda _msg: _valid_response(
            declared_purpose="Foo skill.",
            findings=[_finding("finance")],
        )
    )

    exposure = await analyze_skill(skill, provider, framework="claude_code")

    assert isinstance(exposure, SkillExposure)
    assert exposure.skill_name == "foo"
    assert exposure.declared_purpose == "Foo skill."
    assert len(exposure.findings) == 1
    f = exposure.findings[0]
    assert f.rule_id == "FIN001"
    assert f.affected_components == ["foo"]
    assert f.policy_scaffold is None
    assert f.evidence.category == "finance"
    assert provider.call_count == 1


async def test_analyze_skill_rule_id_numbering_per_category(tmp_path):
    _make_skill_dir(tmp_path, "foo", {"scripts/main.py": "x = 1\n"})
    skill = _discover_one(tmp_path)
    provider = FakeLLMProvider(
        lambda _msg: _valid_response(
            findings=[
                _finding("finance"),
                _finding("finance"),
                _finding("filesystem"),
            ]
        )
    )

    exposure = await analyze_skill(skill, provider)

    assert [f.rule_id for f in exposure.findings] == ["FIN001", "FIN002", "FS001"]


async def test_analyze_skill_malformed_json_raises_parse_failed(tmp_path):
    _make_skill_dir(tmp_path, "foo", {"scripts/main.py": "x = 1\n"})
    skill = _discover_one(tmp_path)
    provider = FakeLLMProvider(lambda _msg: "not json {")

    with pytest.raises(AnalyzeSkillError) as exc_info:
        await analyze_skill(skill, provider)
    assert "parse_failed" in exc_info.value.reason
    assert exc_info.value.skill_name == "foo"


async def test_analyze_skill_missing_declared_purpose_raises_shape_error(tmp_path):
    _make_skill_dir(tmp_path, "foo", {"scripts/main.py": "x = 1\n"})
    skill = _discover_one(tmp_path)
    provider = FakeLLMProvider(lambda _msg: json.dumps({"findings": []}))

    with pytest.raises(AnalyzeSkillError) as exc_info:
        await analyze_skill(skill, provider)
    assert "invalid_response_shape" in exc_info.value.reason


async def test_analyze_skill_unknown_category_raises_shape_error(tmp_path):
    _make_skill_dir(tmp_path, "foo", {"scripts/main.py": "x = 1\n"})
    skill = _discover_one(tmp_path)
    provider = FakeLLMProvider(
        lambda _msg: _valid_response(findings=[_finding("bogus")])
    )

    with pytest.raises(AnalyzeSkillError) as exc_info:
        await analyze_skill(skill, provider)
    assert "invalid_response_shape" in exc_info.value.reason


async def test_analyze_skill_retries_once_on_provider_error(tmp_path):
    _make_skill_dir(tmp_path, "foo", {"scripts/main.py": "x = 1\n"})
    skill = _discover_one(tmp_path)
    calls = {"n": 0}

    def responder(_msg: str) -> ResponderResult:
        calls["n"] += 1
        if calls["n"] == 1:
            return LLMProviderError("transient network blip")
        return _valid_response()

    provider = FakeLLMProvider(responder)

    exposure = await analyze_skill(skill, provider)

    assert isinstance(exposure, SkillExposure)
    assert provider.call_count == 2


async def test_analyze_skill_provider_fails_twice_raises_provider_failed(tmp_path):
    _make_skill_dir(tmp_path, "foo", {"scripts/main.py": "x = 1\n"})
    skill = _discover_one(tmp_path)
    provider = FakeLLMProvider(lambda _msg: LLMProviderError("persistent failure"))

    with pytest.raises(AnalyzeSkillError) as exc_info:
        await analyze_skill(skill, provider)
    assert "provider_failed" in exc_info.value.reason
    assert provider.call_count == 2


async def test_analyze_skill_oversized_payload_raises_payload_rejected(tmp_path):
    big_content = "x" * (MAX_PAYLOAD_BYTES + 1)
    _make_skill_dir(tmp_path, "foo", {"scripts/big.py": big_content})
    skill = _discover_one(tmp_path)
    provider = FakeLLMProvider(lambda _msg: _valid_response())

    with pytest.raises(AnalyzeSkillError) as exc_info:
        await analyze_skill(skill, provider)
    assert "payload_rejected" in exc_info.value.reason
    assert provider.call_count == 0


# ---------- analyze_directory ----------


async def test_analyze_directory_empty_dir_returns_empty_report(tmp_path):
    provider = FakeLLMProvider(lambda _msg: _valid_response())

    report = await analyze_directory(tmp_path, provider)

    assert report.skills == []
    assert report.errors == []
    assert report.evaluator_fingerprint.model == "fake-model"
    assert provider.call_count == 0


async def test_analyze_directory_single_skill_happy_path(tmp_path):
    _make_skill_dir(tmp_path, "foo", {"scripts/main.py": "x = 1\n"})
    provider = FakeLLMProvider(lambda _msg: _valid_response())

    report = await analyze_directory(tmp_path, provider)

    assert len(report.skills) == 1
    assert report.skills[0].skill_name == "foo"
    assert report.errors == []


async def test_analyze_directory_multi_skill_all_succeed(tmp_path):
    for n in ("alpha", "beta", "gamma"):
        _make_skill_dir(tmp_path, n, {"scripts/s.py": f"# {n}\n"})
    provider = FakeLLMProvider(lambda _msg: _valid_response())

    report = await analyze_directory(tmp_path, provider)

    assert {s.skill_name for s in report.skills} == {"alpha", "beta", "gamma"}
    assert report.errors == []
    assert provider.call_count == 3


async def test_analyze_directory_mixed_success_and_failure(tmp_path):
    for n in ("alpha", "beta", "gamma"):
        _make_skill_dir(tmp_path, n, {"scripts/s.py": f"# {n}\n"})

    def responder(msg: str) -> ResponderResult:
        if "# beta" in msg:
            return "not json {"
        return _valid_response()

    provider = FakeLLMProvider(responder)

    report = await analyze_directory(tmp_path, provider)

    assert {s.skill_name for s in report.skills} == {"alpha", "gamma"}
    assert len(report.errors) == 1
    assert report.errors[0].skill_name == "beta"
    assert "parse_failed" in report.errors[0].message


async def test_analyze_directory_progress_callback_fires_once_per_skill(tmp_path):
    for n in ("a", "b", "c", "d", "e"):
        _make_skill_dir(tmp_path, n, {"scripts/s.py": f"# {n}\n"})
    provider = FakeLLMProvider(lambda _msg: _valid_response())
    events: list[tuple[int, int]] = []

    await analyze_directory(
        tmp_path, provider, progress=lambda done, total: events.append((done, total))
    )

    assert len(events) == 5
    assert [e[0] for e in events] == [1, 2, 3, 4, 5]
    assert all(e[1] == 5 for e in events)


async def test_analyze_directory_concurrency_limit_respected(tmp_path):
    for n in range(10):
        _make_skill_dir(tmp_path, f"skill-{n:02d}", {"scripts/s.py": f"# {n}\n"})

    async def slow_response(_msg: str) -> str:
        await asyncio.sleep(0.02)
        return _valid_response()

    provider = FakeLLMProvider(lambda msg: slow_response(msg))

    report = await analyze_directory(tmp_path, provider, concurrency=3)

    assert len(report.skills) == 10
    assert provider.max_in_flight <= 3
    assert provider.max_in_flight >= 2  # sanity: concurrency actually happened


async def test_analyze_directory_concurrency_zero_raises(tmp_path):
    _make_skill_dir(tmp_path, "foo", {"scripts/s.py": "x\n"})
    provider = FakeLLMProvider(lambda _msg: _valid_response())

    with pytest.raises(ValueError, match="concurrency"):
        await analyze_directory(tmp_path, provider, concurrency=0)
    assert provider.call_count == 0


async def test_analyze_directory_unknown_framework_raises(tmp_path):
    _make_skill_dir(tmp_path, "foo", {"scripts/s.py": "x\n"})
    provider = FakeLLMProvider(lambda _msg: _valid_response())

    with pytest.raises(ValueError, match="unknown framework"):
        await analyze_directory(tmp_path, provider, framework="nonsense")
    assert provider.call_count == 0


async def test_analyze_directory_explicit_manifest_name_bypasses_framework_table(
    tmp_path,
):
    skill_dir = tmp_path / "codex-skill"
    skill_dir.mkdir()
    (skill_dir / "AGENT.md").write_text(
        "---\nname: codex\n---\n", encoding="utf-8"
    )
    (skill_dir / "main.py").write_text("x\n", encoding="utf-8")

    provider = FakeLLMProvider(lambda _msg: _valid_response())

    # Unknown framework is tolerated when manifest_name is explicit,
    # up until build_system_prompt is called inside analyze_skill.
    # We use "claude_code" here to exercise the manifest-override path
    # cleanly without tripping prompt rendering.
    report = await analyze_directory(
        tmp_path, provider, framework="claude_code", manifest_name="AGENT.md"
    )

    assert len(report.skills) == 1
    assert report.skills[0].skill_name == "codex-skill"


async def test_analyze_directory_fingerprint_populated(tmp_path):
    _make_skill_dir(tmp_path, "foo", {"scripts/s.py": "x\n"})
    provider = FakeLLMProvider(
        lambda _msg: _valid_response(), model="test-model-7b", temperature=0.0
    )

    report = await analyze_directory(tmp_path, provider, framework="claude_code")

    fp = report.evaluator_fingerprint
    assert fp.model == "test-model-7b"
    assert fp.temperature == 0.0
    assert fp.framework == "claude_code"
    assert fp.prompt_template_hash == system_prompt_hash("claude_code")


async def test_analyze_directory_against_real_fixtures_smoke():
    fixtures_dir = (
        Path(__file__).parent / "fixtures" / "claude_code" / "skills"
    )
    if not fixtures_dir.exists():
        pytest.skip(f"fixtures directory not present: {fixtures_dir}")

    provider = FakeLLMProvider(lambda _msg: _valid_response())

    report = await analyze_directory(fixtures_dir, provider)

    assert len(report.skills) > 0
    assert report.errors == []
    assert provider.call_count == len(report.skills)
