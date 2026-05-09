"""Tests for the analyzer output schema and validator."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from limenex.analyzer.schema import (
    AnalysisReport,
    Evidence,
    EvaluatorFingerprint,
    Finding,
    PolicyHint,
    PolicyScaffold,
    ReportValidationError,
    SkillAnalysisError,
    SkillExposure,
    validate_report,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_report.json"


@pytest.fixture
def canonical_data() -> dict:
    """Fresh deep copy of the canonical fixture per test (tests may mutate)."""
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_construct_to_dict_validate_from_dict():
    """Dataclasses → to_dict → validate_report → typed report, losslessly."""
    report = AnalysisReport.create(
        evaluator_fingerprint=EvaluatorFingerprint(
            model="gpt-4o-mini",
            temperature=0.0,
            prompt_template_hash="hash",
            framework="claude_code",
        ),
        skills=[
            SkillExposure(
                skill_name="example",
                skill_path="example.md",
                declared_purpose="Example skill.",
                findings=[
                    Finding(
                        rule_id="FIN001",
                        type="capability_risk",
                        severity="medium",
                        confidence="high",
                        affected_components=["example"],
                        evidence=Evidence(
                            category="finance",
                            snippet="charges money",
                            file_path="example.md",
                            reasoning="money movement",
                        ),
                        recommendation="constrain",
                        policy_scaffold=PolicyScaffold(
                            skill_id_hint="finance.charge",
                            policies=[
                                PolicyHint(
                                    type="deterministic",
                                    verdict="ESCALATE",
                                    dimension="usd",
                                    operator="gt",
                                    value=50,
                                )
                            ],
                        ),
                    )
                ],
            )
        ],
    )

    data = report.to_dict()
    typed = validate_report(data)

    assert typed.analyzer_version == report.analyzer_version
    assert typed.analyzed_at == report.analyzed_at
    assert typed.evaluator_fingerprint == report.evaluator_fingerprint
    assert typed.skills == report.skills
    assert typed.errors == report.errors


def test_round_trip_preserves_optional_omissions():
    """Optional fields omitted on construction stay omitted through the round trip."""
    report = AnalysisReport.create(
        evaluator_fingerprint=EvaluatorFingerprint(
            model="m", temperature=0.0, prompt_template_hash="h", framework="claude_code"
        ),
        skills=[
            SkillExposure(
                skill_name="s",
                skill_path="s.md",
                declared_purpose="p",
                findings=[
                    Finding(
                        rule_id="FIN001",
                        type="capability_risk",
                        severity="low",
                        confidence="low",
                        affected_components=["s"],
                        evidence=Evidence(
                            category="finance",
                            snippet="x",
                            file_path="s.md",
                            reasoning="y",
                            # line_range omitted
                        ),
                        recommendation="r",
                        # policy_scaffold omitted
                    )
                ],
            )
        ],
    )

    data = report.to_dict()
    assert "line_range" not in data["skills"][0]["findings"][0]["evidence"]
    assert "policy_scaffold" not in data["skills"][0]["findings"][0]

    typed = validate_report(data)
    assert typed.skills[0].findings[0].evidence.line_range is None
    assert typed.skills[0].findings[0].policy_scaffold is None


# ---------------------------------------------------------------------------
# Canonical fixture
# ---------------------------------------------------------------------------


def test_canonical_fixture_validates(canonical_data):
    typed = validate_report(canonical_data)
    assert typed.analyzer_version == "0.2.0.dev0"
    assert len(typed.skills) == 1
    assert len(typed.skills[0].findings) == 2
    assert len(typed.errors) == 1


def test_canonical_fixture_preserves_policy_scaffold(canonical_data):
    typed = validate_report(canonical_data)
    first_finding = typed.skills[0].findings[0]
    assert first_finding.policy_scaffold is not None
    assert first_finding.policy_scaffold.skill_id_hint == "finance.charge"
    assert first_finding.policy_scaffold.policies[0].value == 100


def test_canonical_fixture_preserves_line_range(canonical_data):
    typed = validate_report(canonical_data)
    ev = typed.skills[0].findings[0].evidence
    assert ev.line_range == (12, 14)


# ---------------------------------------------------------------------------
# Malformed input rejection
# ---------------------------------------------------------------------------


def test_rejects_empty_dict():
    with pytest.raises(ReportValidationError) as exc:
        validate_report({})
    assert "analyzer_version" in str(exc.value)


@pytest.mark.parametrize(
    "missing_field",
    ["analyzer_version", "analyzed_at", "evaluator_fingerprint", "skills", "errors"],
)
def test_rejects_missing_top_level_required(canonical_data, missing_field):
    del canonical_data[missing_field]
    with pytest.raises(ReportValidationError) as exc:
        validate_report(canonical_data)
    assert missing_field in str(exc.value)


def test_rejects_unknown_top_level_key(canonical_data):
    canonical_data["unexpected_field"] = "nope"
    with pytest.raises(ReportValidationError):
        validate_report(canonical_data)


def test_rejects_wrong_type_on_analyzer_version(canonical_data):
    canonical_data["analyzer_version"] = 123
    with pytest.raises(ReportValidationError):
        validate_report(canonical_data)


def test_rejects_malformed_rule_id(canonical_data):
    canonical_data["skills"][0]["findings"][0]["rule_id"] = "BOGUS001"
    with pytest.raises(ReportValidationError) as exc:
        validate_report(canonical_data)
    assert "rule_id" in str(exc.value) or "pattern" in str(exc.value).lower()


def test_rejects_invalid_severity_enum(canonical_data):
    canonical_data["skills"][0]["findings"][0]["severity"] = "catastrophic"
    with pytest.raises(ReportValidationError):
        validate_report(canonical_data)


def test_rejects_invalid_confidence_enum(canonical_data):
    canonical_data["skills"][0]["findings"][0]["confidence"] = "pretty sure"
    with pytest.raises(ReportValidationError):
        validate_report(canonical_data)


def test_rejects_invalid_finding_type_enum(canonical_data):
    canonical_data["skills"][0]["findings"][0]["type"] = "other"
    with pytest.raises(ReportValidationError):
        validate_report(canonical_data)


def test_rejects_invalid_verdict_enum(canonical_data):
    canonical_data["skills"][0]["findings"][0]["policy_scaffold"]["policies"][0][
        "verdict"
    ] = "ALLOW"
    with pytest.raises(ReportValidationError):
        validate_report(canonical_data)


def test_rejects_empty_affected_components(canonical_data):
    canonical_data["skills"][0]["findings"][0]["affected_components"] = []
    with pytest.raises(ReportValidationError):
        validate_report(canonical_data)


def test_rejects_line_range_wrong_length(canonical_data):
    canonical_data["skills"][0]["findings"][0]["evidence"]["line_range"] = [1, 2, 3]
    with pytest.raises(ReportValidationError):
        validate_report(canonical_data)


def test_rejects_missing_evidence_field(canonical_data):
    del canonical_data["skills"][0]["findings"][0]["evidence"]["snippet"]
    with pytest.raises(ReportValidationError) as exc:
        validate_report(canonical_data)
    assert "snippet" in str(exc.value)


def test_rejects_negative_temperature(canonical_data):
    canonical_data["evaluator_fingerprint"]["temperature"] = -0.5
    with pytest.raises(ReportValidationError):
        validate_report(canonical_data)