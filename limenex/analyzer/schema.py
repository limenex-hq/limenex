"""Analyzer output schema (v1).

Dataclasses for the structured report produced by the Limenex Skill Risk
Analyzer. Importable without the ``analyzer`` optional extras — schema
consumers (policy authoring flows, managed cloud dashboard, CI
integrations) can validate analyzer output on a core ``pip install limenex``.

Consumers of this module
------------------------
The primary downstream consumer is Limenex's own policy authoring flow
(a future ``limenex draft-policies`` command). ``AnalysisReport.to_dict``
output is consumed by humans and by that flow — never by the runtime
enforcement engine. The engine reads only ``policies.yaml`` and the
``StateStore``.

Schema versioning
-----------------
Schema version is currently implicit (v1). An explicit ``schema_version``
field will be added to the report payload the day a breaking schema
change is introduced.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from limenex import __version__ as _LIMENEX_VERSION

Severity = Literal["critical", "high", "medium", "low"]
Confidence = Literal["high", "medium", "low"]
FindingType = Literal["compromise", "capability_risk"]
Verdict = Literal["BLOCK", "ESCALATE"]
PolicyType = Literal["deterministic", "semantic"]


@dataclass(frozen=True)
class EvaluatorFingerprint:
    """Identifies the LLM configuration that produced an analysis.

    Two reports are meaningfully comparable only when fingerprints match.
    ``temperature`` should be 0 for reproducible analysis; the provider
    defaults it to 0 and the docs recommend leaving it there.
    """

    model: str
    temperature: float
    prompt_template_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "prompt_template_hash": self.prompt_template_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluatorFingerprint:
        return cls(
            model=data["model"],
            temperature=float(data["temperature"]),
            prompt_template_hash=data["prompt_template_hash"],
        )


@dataclass(frozen=True)
class Evidence:
    """Evidence for a single finding.

    ``snippet`` is always populated (quoted excerpt from the skill file).
    ``line_range`` is optional — populate when the finding has a concrete
    textual anchor, omit for meaning-level findings where line numbers
    would be misleading.
    """

    category: str
    snippet: str
    file_path: str
    reasoning: str
    line_range: Optional[tuple[int, int]] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "category": self.category,
            "snippet": self.snippet,
            "file_path": self.file_path,
            "reasoning": self.reasoning,
        }
        if self.line_range is not None:
            out["line_range"] = [self.line_range[0], self.line_range[1]]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        line_range_raw = data.get("line_range")
        line_range: Optional[tuple[int, int]] = None
        if line_range_raw is not None:
            line_range = (int(line_range_raw[0]), int(line_range_raw[1]))
        return cls(
            category=data["category"],
            snippet=data["snippet"],
            file_path=data["file_path"],
            reasoning=data["reasoning"],
            line_range=line_range,
        )


@dataclass(frozen=True)
class PolicyHint:
    """One element of a policy scaffold.

    Deliberately NOT a runtime ``PolicyConfig``. PolicyHint is consumed by
    the ``limenex draft-policies`` flow, which fills in specifics (skill
    IDs, agent IDs, exact numeric thresholds) before emitting YAML that
    the runtime engine can load.
    """

    type: PolicyType
    verdict: Verdict
    dimension: Optional[str] = None
    operator: Optional[str] = None
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type, "verdict": self.verdict}
        if self.dimension is not None:
            out["dimension"] = self.dimension
        if self.operator is not None:
            out["operator"] = self.operator
        if self.value is not None:
            out["value"] = self.value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyHint:
        return cls(
            type=data["type"],
            verdict=data["verdict"],
            dimension=data.get("dimension"),
            operator=data.get("operator"),
            value=data.get("value"),
        )


@dataclass(frozen=True)
class PolicyScaffold:
    """Structured policy suggestion attached to a finding.

    Optional on ``Finding`` — emitted only when the analyzer can produce
    a concrete scaffold. When absent, consumers fall back to the
    finding's ``recommendation`` free-text field.
    """

    skill_id_hint: str
    policies: list[PolicyHint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id_hint": self.skill_id_hint,
            "policies": [p.to_dict() for p in self.policies],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyScaffold:
        return cls(
            skill_id_hint=data["skill_id_hint"],
            policies=[PolicyHint.from_dict(p) for p in data["policies"]],
        )


@dataclass(frozen=True)
class Finding:
    """A single analyzer finding on a skill.

    ``rule_id`` follows prefix-namespaced grammar: ``FIN###`` / ``FS###``
    / ``COMM###`` / ``WEB###`` for capability findings, ``INJ###`` /
    ``POISON###`` / ``SHADOW###`` / ``DRIFT###`` for prompt-injection
    family findings (only ``INJ`` populated in v1), ``FLOW###`` for
    single-skill capability-combination findings, ``MISMATCH###`` for
    declared-purpose-vs-detected-capability findings.

    ``affected_components`` is always length 1 in v1. The array shape is
    forward-compatibility for v1.1+ cross-component findings.
    """

    rule_id: str
    type: FindingType
    severity: Severity
    confidence: Confidence
    affected_components: list[str]
    evidence: Evidence
    recommendation: str
    policy_scaffold: Optional[PolicyScaffold] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "rule_id": self.rule_id,
            "type": self.type,
            "severity": self.severity,
            "confidence": self.confidence,
            "affected_components": list(self.affected_components),
            "evidence": self.evidence.to_dict(),
            "recommendation": self.recommendation,
        }
        if self.policy_scaffold is not None:
            out["policy_scaffold"] = self.policy_scaffold.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        scaffold_raw = data.get("policy_scaffold")
        return cls(
            rule_id=data["rule_id"],
            type=data["type"],
            severity=data["severity"],
            confidence=data["confidence"],
            affected_components=list(data["affected_components"]),
            evidence=Evidence.from_dict(data["evidence"]),
            recommendation=data["recommendation"],
            policy_scaffold=(
                PolicyScaffold.from_dict(scaffold_raw) if scaffold_raw else None
            ),
        )


@dataclass(frozen=True)
class SkillExposure:
    """Analyzer output for a single skill."""

    skill_name: str
    skill_path: str
    declared_purpose: str
    findings: list[Finding]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "skill_path": self.skill_path,
            "declared_purpose": self.declared_purpose,
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillExposure:
        return cls(
            skill_name=data["skill_name"],
            skill_path=data["skill_path"],
            declared_purpose=data["declared_purpose"],
            findings=[Finding.from_dict(f) for f in data["findings"]],
        )


@dataclass(frozen=True)
class SkillAnalysisError:
    """A per-skill failure captured during ``analyze_directory``.

    Single-skill failures do not abort the run — they are
    collected here so the report is complete and the user knows which
    skills were not analyzed.
    """

    skill_name: str
    skill_path: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "skill_path": self.skill_path,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillAnalysisError:
        return cls(
            skill_name=data["skill_name"],
            skill_path=data["skill_path"],
            message=data["message"],
        )


@dataclass(frozen=True)
class AnalysisReport:
    """Top-level analyzer report.

    Primary downstream consumer is the Limenex policy authoring flow.
    The runtime enforcement engine never reads this structure.
    """

    analyzer_version: str
    analyzed_at: datetime
    evaluator_fingerprint: EvaluatorFingerprint
    skills: list[SkillExposure]
    errors: list[SkillAnalysisError] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        evaluator_fingerprint: EvaluatorFingerprint,
        skills: list[SkillExposure],
        errors: Optional[list[SkillAnalysisError]] = None,
        analyzed_at: Optional[datetime] = None,
    ) -> AnalysisReport:
        """Construct a report with analyzer_version and timestamp filled in."""
        return cls(
            analyzer_version=_LIMENEX_VERSION,
            analyzed_at=analyzed_at or datetime.now(timezone.utc),
            evaluator_fingerprint=evaluator_fingerprint,
            skills=skills,
            errors=errors or [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyzer_version": self.analyzer_version,
            "analyzed_at": self.analyzed_at.isoformat(),
            "evaluator_fingerprint": self.evaluator_fingerprint.to_dict(),
            "skills": [s.to_dict() for s in self.skills],
            "errors": [e.to_dict() for e in self.errors],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisReport:
        return cls(
            analyzer_version=data["analyzer_version"],
            analyzed_at=datetime.fromisoformat(data["analyzed_at"]),
            evaluator_fingerprint=EvaluatorFingerprint.from_dict(
                data["evaluator_fingerprint"]
            ),
            skills=[SkillExposure.from_dict(s) for s in data["skills"]],
            errors=[SkillAnalysisError.from_dict(e) for e in data.get("errors", [])],
        )


_SCHEMA_PATH = Path(__file__).parent / "schemas" / "v1.json"


def _load_schema() -> dict[str, Any]:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


_SCHEMA_CACHE: Optional[dict[str, Any]] = None
_VALIDATOR_CACHE: Optional[Draft202012Validator] = None


def _get_validator() -> Draft202012Validator:
    global _SCHEMA_CACHE, _VALIDATOR_CACHE
    if _VALIDATOR_CACHE is None:
        _SCHEMA_CACHE = _load_schema()
        _VALIDATOR_CACHE = Draft202012Validator(_SCHEMA_CACHE)
    return _VALIDATOR_CACHE


class ReportValidationError(ValueError):
    """Raised when a dict fails validation against the analyzer report schema."""


def validate_report(data: dict[str, Any]) -> AnalysisReport:
    """Validate a raw dict against the analyzer report schema and return a typed report.

    Parameters
    ----------
    data
        Raw dictionary, typically from ``json.loads``.

    Returns
    -------
    AnalysisReport
        Typed report if validation succeeds.

    Raises
    ------
    ReportValidationError
        If the dict does not conform to the schema. The exception message
        contains the validator's path into the document and a human-readable
        description of the failure.
    """
    validator = _get_validator()
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        path = "/".join(str(p) for p in first.absolute_path) or "<root>"
        raise ReportValidationError(
            f"Invalid analyzer report at {path}: {first.message}"
        )
    return AnalysisReport.from_dict(data)
