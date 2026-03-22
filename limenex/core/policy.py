"""
policy.py — Core policy definitions for Limenex.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Literal, Protocol, get_args, runtime_checkable

__all__ = [
    "Verdict",
    "BreachVerdict",
    "Operator",
    "DeterministicPolicy",
    "SemanticPolicy",
    "PolicyConfig",
    "StateStore",
    "AsyncStateStore",
    "PolicyStore",
    "AsyncPolicyStore",
    "UnregisteredSkillError",
    "LimenexConfigWarning",
]

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class LimenexConfigWarning(UserWarning):
    """Warnings emitted for Limenex policy configuration issues.

    Filter or suppress independently of other UserWarnings:
        warnings.filterwarnings("ignore", category=LimenexConfigWarning)
    """


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

Verdict = Literal["ALLOW", "BLOCK", "ESCALATE"]
BreachVerdict = Literal["BLOCK", "ESCALATE"]

_VALID_BREACH_VERDICTS: frozenset[str] = frozenset(get_args(BreachVerdict))

# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

Operator = Literal["lt", "lte", "gt", "gte", "eq", "neq"]

# Private — used internally by engine.py only.
_OPERATOR_FNS: MappingProxyType[str, Callable[[float, float], bool]] = MappingProxyType(
    {
        "lt": lambda current, val: current < val,
        "lte": lambda current, val: current <= val,
        "gt": lambda current, val: current > val,
        "gte": lambda current, val: current >= val,
        "eq": lambda current, val: current == val,
        "neq": lambda current, val: current != val,
    }
)

if set(get_args(Operator)) != set(_OPERATOR_FNS.keys()):
    raise RuntimeError(
        "Operator Literal and _OPERATOR_FNS keys are out of sync. "
        "Update both together when adding or removing operators."
    )

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnregisteredSkillError(Exception):
    """Raised when a skill_id is not registered in the PolicyStore.

    Attributes:
        skill_id: The unregistered skill identifier.
    """

    def __init__(self, skill_id: str) -> None:
        self.skill_id = skill_id
        super().__init__(f"Skill '{skill_id}' is not registered in the PolicyStore.")


# ---------------------------------------------------------------------------
# DeterministicPolicy
# ---------------------------------------------------------------------------


@dataclass
class DeterministicPolicy:
    """A serializable, rule-based governance constraint.

    Evaluates a state dimension from the StateStore against an operator +
    value condition. Returns breach_verdict if the condition is not satisfied.

    Projective check: when param is set, the engine evaluates
    (current_value + proposed_value) where proposed_value is extracted from
    the named function argument. The same value is passed to StateStore.record()
    after successful execution. When param is None, the check is non-projective
    and StateStore.record() is called with value=1.0 (count increment).

    eq/neq use exact float comparison — only use with integer-equivalent
    values (e.g. 1.0, 0.0). Use lt/lte/gt/gte for fractional thresholds.

    Args:
        dimension:      StateStore key to query. User-defined, any string is valid.
        operator:       One of: "lt", "lte", "gt", "gte", "eq", "neq".
        value:          Finite threshold to compare against.
        param:          Function argument name mapping to this dimension.
                        None for count-based (non-projective) checks.
        breach_verdict: Verdict when condition is not satisfied.
                        Must be "BLOCK" or "ESCALATE". Default: "ESCALATE".

    Examples:
        DeterministicPolicy(                      # projective spend check
            dimension="4h_finance_spend",
            operator="lt", value=50.0, param="amount",
            breach_verdict="BLOCK",
        )
        DeterministicPolicy(                      # count check
            dimension="daily_api_calls",
            operator="lt", value=100,
        )
        DeterministicPolicy(                      # approval gate
            dimension="top_up_aws_approved",
            operator="eq", value=1.0,
        )
    """

    dimension: str
    operator: Operator
    value: float
    param: str | None = None
    breach_verdict: BreachVerdict = "ESCALATE"

    def __post_init__(self) -> None:
        self.dimension = self.dimension.strip()
        if not self.dimension:
            raise ValueError("dimension must be a non-empty string.")
        if self.operator not in _OPERATOR_FNS:
            raise ValueError(
                f"Invalid operator '{self.operator}'. "
                f"Must be one of: {list(_OPERATOR_FNS.keys())}."
            )
        if not math.isfinite(self.value):
            raise ValueError(f"value must be a finite number, got {self.value!r}.")
        if self.breach_verdict not in _VALID_BREACH_VERDICTS:
            raise ValueError(
                f"Invalid breach_verdict '{self.breach_verdict}'. "
                f"Must be one of: {sorted(_VALID_BREACH_VERDICTS)}."
            )
        if self.param is not None:
            self.param = self.param.strip()
            if not self.param:
                raise ValueError("param must be a non-empty string or None.")
        if self.operator in ("eq", "neq") and self.value != math.floor(self.value):
            warnings.warn(
                f"operator '{self.operator}' uses exact float comparison. "
                f"Non-integer value {self.value!r} may be unreliable. "
                f"Use lt/lte/gt/gte for fractional thresholds.",
                LimenexConfigWarning,
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# SemanticPolicy
# ---------------------------------------------------------------------------


@dataclass
class SemanticPolicy:
    """A natural language governance rule evaluated by an LLM.

    The rule is passed to the engine's llm_evaluator alongside the action
    intent. verdict_ceiling caps the maximum verdict this policy can produce —
    if the evaluator returns a more severe verdict, verdict_ceiling is used.
    verdict_ceiling is also used as fallback for unexpected evaluator responses.

    Args:
        rule:            Governance rule in plain language. Be specific.
        verdict_ceiling: Maximum verdict this policy can produce.
                         Must be "BLOCK" or "ESCALATE". Default: "ESCALATE".

    Example:
        SemanticPolicy(
            rule="Do not approve actions that transfer customer PII externally.",
            verdict_ceiling="BLOCK",
        )
    """

    rule: str
    verdict_ceiling: BreachVerdict = "ESCALATE"

    def __post_init__(self) -> None:
        self.rule = self.rule.strip()
        if not self.rule:
            raise ValueError("SemanticPolicy.rule must not be empty.")
        if self.verdict_ceiling not in _VALID_BREACH_VERDICTS:
            raise ValueError(
                f"Invalid verdict_ceiling '{self.verdict_ceiling}'. "
                f"Must be one of: {sorted(_VALID_BREACH_VERDICTS)}."
            )


# ---------------------------------------------------------------------------
# PolicyConfig
# ---------------------------------------------------------------------------


@dataclass
class PolicyConfig:
    """All governance policies for a single skill.

    Loaded at runtime by a PolicyStore — not instantiated directly in
    application code.

    policies is a unified ordered list of DeterministicPolicy and/or
    SemanticPolicy entries. The engine evaluates them in sequence and
    short-circuits on the first non-ALLOW verdict. Order is significant —
    place BLOCK policies before ESCALATE policies.

    Args:
        policies: Ordered list of DeterministicPolicy and/or SemanticPolicy.

    Example:
        PolicyConfig(policies=[
            DeterministicPolicy(
                dimension="4h_finance_spend",
                operator="lt", value=50.0, param="amount",
                breach_verdict="BLOCK",
            ),
            SemanticPolicy(
                rule="Do not approve duplicate transactions.",
            ),
        ])
    """

    policies: list[DeterministicPolicy | SemanticPolicy] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.policies, list):
            raise TypeError("policies must be a list.")
        for i, policy in enumerate(self.policies):
            if not isinstance(policy, (DeterministicPolicy, SemanticPolicy)):
                raise TypeError(
                    f"policies[{i}] must be a DeterministicPolicy or "
                    f"SemanticPolicy instance, got {type(policy).__name__}."
                )
        if not self.policies:
            warnings.warn(
                "PolicyConfig has no policies — all actions will be ALLOWED "
                "unconditionally. Add at least one policy.",
                LimenexConfigWarning,
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# StateStore
# ---------------------------------------------------------------------------


@runtime_checkable
class StateStore(Protocol):
    """Synchronous interface for resolving dimension values at runtime."""

    def get(self, agent_id: str, dimension: str) -> float:
        """Return current value of dimension for agent_id. Returns 0.0 if unset."""
        ...

    def record(self, agent_id: str, dimension: str, value: float) -> None:
        """Persist a completed action value against a dimension."""
        ...


@runtime_checkable
class AsyncStateStore(Protocol):
    """Asynchronous counterpart to StateStore. The engine accepts either.

    Note: isinstance() checks verify method existence only, not that methods
    are async. Implementors must define all methods as coroutines.
    """

    async def get(self, agent_id: str, dimension: str) -> float:
        """Return current value of dimension for agent_id. Returns 0.0 if unset."""
        ...

    async def record(self, agent_id: str, dimension: str, value: float) -> None:
        """Persist a completed action value against a dimension."""
        ...


# ---------------------------------------------------------------------------
# PolicyStore
# ---------------------------------------------------------------------------


@runtime_checkable
class PolicyStore(Protocol):
    """Synchronous interface for loading PolicyConfig by skill_id at runtime.

    Raises UnregisteredSkillError for unknown skill IDs — never fails open.
    Implementations should warn when a registered skill has an empty PolicyConfig.
    """

    def get(self, skill_id: str) -> PolicyConfig:
        """Return PolicyConfig for skill_id.

        Raises:
            UnregisteredSkillError: If skill_id is not registered.
        """
        ...


@runtime_checkable
class AsyncPolicyStore(Protocol):
    """Asynchronous counterpart to PolicyStore. The engine accepts either.

    Note: isinstance() checks verify method existence only, not that methods
    are async. Implementors must define all methods as coroutines.
    """

    async def get(self, skill_id: str) -> PolicyConfig:
        """Return PolicyConfig for skill_id.

        Raises:
            UnregisteredSkillError: If skill_id is not registered.
        """
        ...
