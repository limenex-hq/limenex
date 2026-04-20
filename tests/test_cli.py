from __future__ import annotations

from pathlib import Path

import pytest

from limenex.__main__ import main


def _write_policies(tmp_path: Path, body: str) -> Path:
    policies = tmp_path / ".limenex" / "policies.yaml"
    policies.parent.mkdir(parents=True, exist_ok=True)
    policies.write_text(body)
    return policies


def test_validate_valid_file(tmp_path, capsys):
    policies = _write_policies(
        tmp_path,
        """
finance.spend:
  policies:
    - type: deterministic
      dimension: spend_usd
      operator: lt
      value: 50.0
      param: amount_usd
      breach_verdict: ESCALATE
    - type: semantic
      rule: "Is this payment to a known vendor?"
      verdict_ceiling: ESCALATE
filesystem.write:
  policies:
    - type: deterministic
      dimension: allowed_paths
      operator: in
      values: ["/tmp", "/var/tmp"]
      param: path
      breach_verdict: BLOCK
""",
    )
    assert main(["validate", str(policies)]) == 0
    out = capsys.readouterr().out
    assert "✓ Valid" in out
    assert "2 skills" in out
    assert "3 policies" in out
    assert "2 deterministic" in out
    assert "1 semantic" in out


def test_validate_missing_file(tmp_path, capsys):
    missing = tmp_path / "does" / "not" / "exist.yaml"
    assert main(["validate", str(missing)]) == 1
    err = capsys.readouterr().err
    assert "✗ Invalid" in err
    assert "not found" in err
    # The actionable hint should be present so developers know how to fix it.
    assert "python -m limenex validate" in err


def test_validate_invalid_yaml(tmp_path, capsys):
    policies = _write_policies(
        tmp_path,
        """
finance.spend:
  policies:
    - type: semantic
      rule: "Is this a known vendor?"
      verdict_ceiling: ALLOW
""",
    )
    assert main(["validate", str(policies)]) == 1
    err = capsys.readouterr().err
    assert "✗ Invalid" in err
    assert "verdict_ceiling" in err
    assert "finance.spend" in err


def test_validate_singular_pluralisation(tmp_path, capsys):
    policies = _write_policies(
        tmp_path,
        """
finance.spend:
  policies:
    - type: deterministic
      dimension: spend_usd
      operator: lt
      value: 50.0
      param: amount_usd
      breach_verdict: ESCALATE
""",
    )
    assert main(["validate", str(policies)]) == 0
    out = capsys.readouterr().out
    assert "1 skill," in out
    assert "1 policy" in out
    assert "1 deterministic" in out


def test_validate_uses_default_path(tmp_path, monkeypatch, capsys):
    _write_policies(
        tmp_path,
        """
finance.spend:
  policies:
    - type: semantic
      rule: "Known vendor?"
      verdict_ceiling: ESCALATE
""",
    )
    monkeypatch.chdir(tmp_path)
    assert main(["validate"]) == 0
    out = capsys.readouterr().out
    assert "✓ Valid" in out


def test_cli_entry_point_requires_subcommand(capsys):
    with pytest.raises(SystemExit):
        main([])
