"""
policy_store.py — Local file-backed implementation of PolicyStore.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

from .policy import (
    DeterministicPolicy,
    PolicyConfig,
    PolicyStore,
    SemanticPolicy,
    UnregisteredSkillError,
)

__all__ = ["LocalFilePolicyStore"]

_EXTENDS_KEY = "_extends"


class LocalFilePolicyStore:
    """YAML file-backed implementation of the PolicyStore protocol.

    Reads one or more policies.yaml files, resolves single-level inheritance
    via the _extends key, and deserialises into PolicyConfig objects.

    Policies are loaded and cached at instantiation time. Changes to the
    YAML file after startup require a new LocalFilePolicyStore instance.

    Inheritance rules:
        - A child file may declare _extends: <relative or absolute path>
        - Parent policies for a skill are prepended to child policies
        - Skills defined only in parent or only in child are taken as-is
        - Multi-level inheritance is not supported in Phase 1

    Raises UnregisteredSkillError for unknown skill_ids. Emits a warning
    for skills with an empty policies list.

    Args:
        path: Path to the primary (child) policies YAML file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"Policy file not found: '{self._path}'. "
                f"Ensure the file exists before instantiating LocalFilePolicyStore."
            )
        if self._path.is_dir():
            raise ValueError(
                f"Policy store path '{self._path}' is a directory, not a file."
            )
        self._configs: dict[str, PolicyConfig] = self._load()

    # ------------------------------------------------------------------
    # PolicyStore protocol
    # ------------------------------------------------------------------

    def get(self, skill_id: str) -> PolicyConfig:
        """Return the PolicyConfig for skill_id.

        Raises:
            UnregisteredSkillError: skill_id not found in loaded configs.
        """
        if skill_id not in self._configs:
            raise UnregisteredSkillError(
                f"No policy configuration found for skill_id '{skill_id}'. "
                f"Ensure it is declared in your policies YAML file."
            )
        return self._configs[skill_id]

    # ------------------------------------------------------------------
    # Internal loading and deserialisation
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, PolicyConfig]:
        """Load, resolve inheritance, and deserialise all skill configs."""
        child_raw = self._read_yaml(self._path)

        # Resolve _extends if present
        parent_raw: dict[str, Any] = {}
        extends_value = child_raw.pop(_EXTENDS_KEY, None)
        if extends_value is not None:
            parent_path = Path(extends_value)
            if not parent_path.is_absolute():
                parent_path = self._path.parent / parent_path
            if not parent_path.exists():
                raise FileNotFoundError(
                    f"Policy file '{self._path}' declares _extends: '{extends_value}' "
                    f"but the referenced file was not found at '{parent_path}'."
                )
            parent_raw = self._read_yaml(parent_path)
            # Disallow chained inheritance in Phase 1
            if _EXTENDS_KEY in parent_raw:
                raise ValueError(
                    f"Multi-level policy inheritance is not supported. "
                    f"'{parent_path}' itself declares _extends. "
                    f"Flatten your policy hierarchy to a single level."
                )

        merged = self._merge(parent_raw, child_raw)
        return {
            skill_id: self._deserialise(skill_id, skill_data)
            for skill_id, skill_data in merged.items()
        }

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        """Read and parse a YAML file. Returns an empty dict for empty files."""
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(
                f"Policy file '{path}' must be a YAML mapping at the top level, "
                f"got {type(data).__name__}."
            )
        return data

    def _merge(
        self,
        parent: dict[str, Any],
        child: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge parent and child skill configs.

        For skills present in both: parent policies prepended to child policies.
        For skills present in only one: taken as-is.
        """
        merged: dict[str, Any] = {}

        all_skills = sorted(set(parent) | set(child))
        for skill_id in all_skills:
            if skill_id in parent and skill_id in child:
                parent_policies = parent[skill_id].get("policies", [])
                child_policies = child[skill_id].get("policies", [])
                merged[skill_id] = {"policies": parent_policies + child_policies}
            elif skill_id in parent:
                merged[skill_id] = parent[skill_id]
            else:
                merged[skill_id] = child[skill_id]

        return merged

    def _deserialise(self, skill_id: str, data: dict[str, Any]) -> PolicyConfig:
        """Deserialise a single skill's raw YAML dict into a PolicyConfig."""
        raw_policies = data.get("policies", [])

        if not raw_policies:
            warnings.warn(
                f"Skill '{skill_id}' has an empty policies list. "
                f"All calls will be ALLOW by default.",
                stacklevel=4,
            )

        policies = []
        for i, raw in enumerate(raw_policies):
            policy_type = raw.get("type")

            if policy_type == "deterministic":
                try:
                    policies.append(
                        DeterministicPolicy(
                            dimension=raw["dimension"],
                            operator=raw["operator"],
                            value=float(raw["value"]),
                            param=raw.get("param"),
                            breach_verdict=raw["breach_verdict"],
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid deterministic policy at index {i} for skill '{skill_id}': {exc}. "
                        f"Required fields: dimension, operator, value, breach_verdict."
                    ) from exc
            elif policy_type == "semantic":
                try:
                    policies.append(
                        SemanticPolicy(
                            rule=raw["rule"],
                            verdict_ceiling=raw["verdict_ceiling"],
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid semantic policy at index {i} for skill '{skill_id}': {exc}. "
                        f"Required fields: rule, verdict_ceiling."
                    ) from exc
            else:
                raise ValueError(
                    f"Unknown policy type '{policy_type}' in skill '{skill_id}' "
                    f"at index {i}. Expected 'deterministic' or 'semantic'."
                )

        return PolicyConfig(policies=policies)
