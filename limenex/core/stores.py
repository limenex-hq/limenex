"""
stores.py — Local file-backed implementation of StateStore.
"""

from __future__ import annotations

import os
import json
import tempfile
import threading
import warnings
from pathlib import Path

from .policy import StateStore

__all__ = ["LocalFileStateStore"]


class LocalFileStateStore:
    """File-backed implementation of the StateStore protocol.

    State is persisted as JSON with the layout:
        { dimension: { agent_id: value } }

    Dimension-first layout means new agents appear automatically as new
    keys under a dimension when first recorded — no pre-registration
    required. Agents are zero-initialised implicitly on first read.

    Thread-safe via an in-process lock. Not safe for concurrent
    multi-process access — use a cloud StateStore for that.

    Args:
        path: Path to the JSON state file. Created automatically if absent.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists() and self._path.is_dir():
            raise ValueError(
                f"State store path '{self._path}' is a directory, not a file."
            )
        if not self._path.exists():
            self._path.write_text("{}", encoding="utf-8")

    def _read(self) -> dict[str, dict[str, float]]:
        text = self._path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            warnings.warn(
                f"State file '{self._path}' contains invalid JSON and could not be read. "
                f"Returning empty state. Manual inspection required.",
                stacklevel=3,
            )
            return {}

    def _write(self, data: dict[str, dict[str, float]]) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._path.parent,
            delete=False,
            suffix=".tmp",
        ) as f:
            tmp_path = f.name
            try:
                json.dump(data, f, indent=2)
            except Exception:
                os.unlink(tmp_path)
                raise
        os.replace(tmp_path, self._path)  # atomic on POSIX, best-effort on Windows

    def get(self, agent_id: str, dimension: str) -> float:
        """Return the current accumulated value for (agent_id, dimension).

        Returns 0.0 if no entry exists.
        """
        with self._lock:
            data = self._read()
            return data.get(dimension, {}).get(agent_id, 0.0)

    def record(self, agent_id: str, dimension: str, value: float) -> None:
        """Increment the stored value for (agent_id, dimension) by value."""
        with self._lock:
            data = self._read()
            if dimension not in data:
                data[dimension] = {}
            data[dimension][agent_id] = data[dimension].get(agent_id, 0.0) + value
            self._write(data)

    # ------------------------------------------------------------------
    # Convenience helpers — not part of the StateStore protocol.
    # Intended for org-managed reset jobs (e.g. end-of-day cron).
    # ------------------------------------------------------------------

    def reset_dimension(self, dimension: str) -> None:
        """Zero all agent values for a given dimension.

        Typical use: end-of-day reset for time-windowed limits
        (e.g. finance.today_spend). Zeros all agents atomically.
        """
        with self._lock:
            data = self._read()
            if dimension not in data:
                warnings.warn(
                    f"reset_dimension called on unknown dimension '{dimension}'. "
                    f"No state was modified.",
                    stacklevel=2,
                )
                return
            data[dimension] = {agent: 0.0 for agent in data[dimension]}
            self._write(data)

    def reset_agent(self, agent_id: str) -> None:
        """Zero all dimension values for a given agent.

        Typical use: decommissioning an agent or resetting a
        specific agent's state across all dimensions.
        """
        with self._lock:
            data = self._read()
            found = any(agent_id in agents for agents in data.values())
            if not found:
                warnings.warn(
                    f"reset_agent called on unknown agent_id '{agent_id}'. "
                    f"No state was modified.",
                    stacklevel=2,
                )
                return
            for dimension in data:
                if agent_id in data[dimension]:
                    data[dimension][agent_id] = 0.0
            self._write(data)
