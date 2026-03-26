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

Operator = Literal["lt", "lte", "gt", "gte", "eq", "neq", "in", "not_in"]

# Private — used internally by engine.py only.
_NUMERIC_OPERATOR_FNS: MappingProxyType[str, Callable[[float, float], bool]] = (
    MappingProxyType(
        {
            "lt": lambda current, val: current < val,
            "lte": lambda current, val: current <= val,
            "gt": lambda current, val: current > val,
            "gte": lambda current, val: current >= val,
            "eq": lambda current, val: current == val,
            "neq": lambda current, val: current != val,
        }
    )
)

_SET_OPERATOR_FNS: MappingProxyType[str, Callable[[str, frozenset[str]], bool]] = (
    MappingProxyType(
        {
            "in": lambda val, allowed: val in allowed,
            "not_in": lambda val, excluded: val not in excluded,
        }
    )
)

if set(get_args(Operator)) != (
    set(_NUMERIC_OPERATOR_FNS.keys()) | set(_SET_OPERATOR_FNS.keys())
):
    raise RuntimeError(
        "Operator Literal and operator function maps are out of sync. "
        "Update _NUMERIC_OPERATOR_FNS, _SET_OPERATOR_FNS, and the Operator "
        "Literal together when adding or removing operators."
    )

if not set(_NUMERIC_OPERATOR_FNS.keys()).isdisjoint(set(_SET_OPERATOR_FNS.keys())):
    raise RuntimeError(
        "_NUMERIC_OPERATOR_FNS and _SET_OPERATOR_FNS share overlapping keys. "
        "Operator names must be unique across both maps."
    )

# Module-level constant — computed once after guards confirm correctness.
_ALL_OPERATORS: frozenset[str] = frozenset(
    _NUMERIC_OPERATOR_FNS.keys() | _SET_OPERATOR_FNS.keys()
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

    Supports two operator families:

    **Numeric operators** (lt, lte, gt, gte, eq, neq):
        Evaluates a state dimension from the StateStore against a numeric
        threshold. `value` is required; `values` must be None.

        Projective check: when `param` is set, the engine evaluates
        (current_value + proposed_value) where proposed_value is extracted
        from the named function argument. The same value is passed to
        StateStore.record() after successful execution. When `param` is None,
        the check is non-projective and StateStore.record() is called with
        value=1.0 (count increment).

        eq/neq use exact float comparison — only use with integer-equivalent
        values (e.g. 1.0, 0.0). Use lt/lte/gt/gte for fractional thresholds.

    **Set membership operators** (in, not_in):
        Evaluates whether kwargs[param] is a member of the defined string set.
        `values` is required; `value` must be None; `param` is mandatory.

        Matching is exact and case-sensitive: "/Workspace" is NOT considered
        in {"/workspace", "/tmp"}. Domain or suffix extraction is out of scope
        for this operator — use exact full strings (e.g. full email addresses,
        full file paths, full URLs).

        `dimension` acts as a human-readable label only for set-operator
        policies. The StateStore is never read or written — there is no state
        accumulation for set membership checks.

        Empty set behaviour is deterministic:
            in     frozenset() → always breaches (nothing satisfies membership)
            not_in frozenset() → always passes (nothing to exclude)
        Both are almost certainly misconfigurations; a LimenexConfigWarning
        is emitted at construction time when `values` is empty.

    Args:
        dimension:      StateStore key (numeric operators) or human-readable
                        label (set operators). User-defined, any string is valid.
        operator:       One of: "lt", "lte", "gt", "gte", "eq", "neq",
                        "in", "not_in".
        value:          Finite numeric threshold. Required for numeric operators;
                        must be None for set operators.
        param:          Function argument name mapped to this dimension.
                        Required for set operators (hard error if absent).
                        Optional (None) for count-based numeric checks.
        breach_verdict: Verdict when condition is not satisfied.
                        Must be "BLOCK" or "ESCALATE". Default: "ESCALATE".
        values:         Frozenset of exact strings for membership evaluation.
                        Required for set operators; must be None for numeric
                        operators. Must be a frozenset — lists and mutable sets
                        are rejected at construction time.

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
        DeterministicPolicy(                      # path allowlist
            dimension="allowed_filepaths",
            operator="in",
            values=frozenset({"/workspace", "/tmp"}),
            param="filepath",
            breach_verdict="BLOCK",
        )
        DeterministicPolicy(                      # recipient blocklist
            dimension="blocked_recipients",
            operator="not_in",
            values=frozenset({"alice@baddomain.com", "bob@baddomain.com"}),
            param="recipient",
            breach_verdict="BLOCK",
        )
    """

    dimension: str
    operator: Operator
    value: float | None = None
    param: str | None = None
    breach_verdict: BreachVerdict = "ESCALATE"
    values: frozenset[str] | None = None

    def __post_init__(self) -> None:
        # --- dimension ---
        self.dimension = self.dimension.strip()
        if not self.dimension:
            raise ValueError("dimension must be a non-empty string.")

        # --- operator ---
        if self.operator not in _ALL_OPERATORS:
            raise ValueError(
                f"Invalid operator '{self.operator}'. "
                f"Must be one of: {sorted(_ALL_OPERATORS)}."
            )

        # --- breach_verdict ---
        if self.breach_verdict not in _VALID_BREACH_VERDICTS:
            raise ValueError(
                f"Invalid breach_verdict '{self.breach_verdict}'. "
                f"Must be one of: {sorted(_VALID_BREACH_VERDICTS)}."
            )

        # --- param (strip and validate if provided; set operators enforce
        #     presence in the family-specific block below) ---
        if self.param is not None:
            self.param = self.param.strip()
            if not self.param:
                raise ValueError("param must be a non-empty string or None.")

        # --- operator-family-specific validation ---
        if self.operator in _NUMERIC_OPERATOR_FNS:
            # Numeric branch
            if self.values is not None:
                raise ValueError(
                    f"Numeric operator '{self.operator}' does not accept 'values'. "
                    f"Provide 'value' (a finite float) and leave 'values' as None."
                )
            if self.value is None:
                raise ValueError(
                    f"Numeric operator '{self.operator}' requires 'value' to be a "
                    f"finite float. Got None."
                )
            if not math.isfinite(self.value):
                raise ValueError(f"value must be a finite number, got {self.value!r}.")
            # Preserve existing eq/neq fractional-float warning
            if self.operator in ("eq", "neq") and self.value != math.floor(self.value):
                warnings.warn(
                    f"operator '{self.operator}' uses exact float comparison. "
                    f"Non-integer value {self.value!r} may be unreliable. "
                    f"Use lt/lte/gt/gte for fractional thresholds.",
                    LimenexConfigWarning,
                    stacklevel=3,
                )

        else:
            # Set membership branch (in, not_in)
            if self.value is not None:
                raise ValueError(
                    f"Set operator '{self.operator}' does not accept 'value'. "
                    f"Provide 'values' (a frozenset of strings) and leave 'value' as None."
                )
            if self.param is None:
                raise ValueError(
                    f"Set operator '{self.operator}' requires 'param' to be set. "
                    f"'param' identifies the skill argument to evaluate against 'values'."
                )
            if self.values is None:
                raise ValueError(
                    f"Set operator '{self.operator}' requires 'values' to be a "
                    f"frozenset of strings. Got None."
                )
            if not isinstance(self.values, frozenset):
                raise TypeError(
                    f"Set operator '{self.operator}' requires 'values' to be a frozenset. "
                    f"Got {type(self.values).__name__!r}. Wrap your values: "
                    f"frozenset({{...}}) or frozenset(your_iterable)."
                )
            if not all(isinstance(v, str) for v in self.values):
                non_strings = [v for v in self.values if not isinstance(v, str)]
                raise ValueError(
                    f"All items in 'values' must be strings. "
                    f"Got non-string items: {non_strings!r}."
                )
            if len(self.values) == 0:
                warnings.warn(
                    f"DeterministicPolicy for dimension '{self.dimension}' has an "
                    f"empty 'values' set with operator '{self.operator}'. "
                    f"'in' with an empty set always breaches; "
                    f"'not_in' with an empty set always passes. "
                    f"This is almost certainly a misconfiguration.",
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
