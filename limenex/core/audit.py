"""
audit.py — Audit logging for Limenex policy evaluations.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import EvaluationResult
from .policy import DeterministicPolicy, SemanticPolicy

__all__ = ["LocalAuditLogger"]


class LocalAuditLogger:
    """Appends structured JSON audit entries to a local log file (JSONL).

    Each line is a self-contained JSON object:
        {
          "timestamp": "2026-03-25T12:00:00+00:00",
          "agent_id":  "agent_007",
          "skill_id":  "charge_card",
          "kwargs":    {"amount": 120.0, "merchant": "AWS"},
          "verdict":   "ALLOW",
          "triggered_by": null
        }

    Non-JSON-serialisable kwargs values are safely repr()-ed rather
    than raising — the audit log must never crash a skill call.

    Args:
        path: Path to the audit log file. Parent directories are created
              automatically. Defaults to .limenex/audit.log.
    """

    def __init__(self, path: str | Path = ".limenex/audit.log") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    @staticmethod
    def _serialise_policy(
        policy: DeterministicPolicy | SemanticPolicy | None,
    ) -> dict | None:
        if policy is None:
            return None
        if isinstance(policy, DeterministicPolicy):
            return {
                "type": "deterministic",
                "dimension": policy.dimension,
                "operator": policy.operator,
                "value": policy.value,
                "breach_verdict": policy.breach_verdict,
            }
        return {
            "type": "semantic",
            "rule": policy.rule,
            "verdict_ceiling": policy.verdict_ceiling,
        }

    def log(self, result: EvaluationResult, kwargs: dict[str, Any]) -> None:
        """Append a single audit entry as a JSONL line."""
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": result.agent_id,
                "skill_id": result.skill_id,
                "kwargs": json.loads(json.dumps(kwargs, default=repr)),
                "verdict": result.verdict,
                "triggered_by": self._serialise_policy(result.triggered_by),
            }
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            traceback.print_exc(
                file=sys.stderr
            )  # Audit failure must never propagate to the skill call
