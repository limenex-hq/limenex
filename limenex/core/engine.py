"""
engine.py — Policy evaluation engine for Limenex.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, cast, get_args

from .policy import (
    _OPERATOR_FNS,
    AsyncPolicyStore,
    AsyncStateStore,
    DeterministicPolicy,
    PolicyStore,
    SemanticPolicy,
    StateStore,
    UnregisteredSkillError,
    Verdict,
)

__all__ = [
    "LimenexConfigError",
    "BlockedError",
    "EscalationRequired",
    "EvaluationResult",
    "PolicyEngine",
    "UnregisteredSkillError",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERDICT_SEVERITY: dict[str, int] = {"ALLOW": 0, "ESCALATE": 1, "BLOCK": 2}

if set(get_args(Verdict)) != set(_VERDICT_SEVERITY.keys()):
    raise RuntimeError(
        "Verdict and _VERDICT_SEVERITY are out of sync. "
        "Update both together when adding or removing verdicts."
    )

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LimenexConfigError(Exception):
    """Raised when a misconfiguration is detected at evaluation time.

    Distinct from UnregisteredSkillError (unknown skill_id) — this covers
    engine-level configuration errors: missing llm_evaluator for a
    SemanticPolicy, param absent from kwargs, non-numeric param value, or
    agent_id_param absent from decorated skill arguments.
    """


class BlockedError(Exception):
    """Raised by @governed when a skill call is hard-stopped by policy.

    The skill function was never executed.

    Attributes:
        result: The full EvaluationResult, including triggered_by.
    """

    def __init__(self, result: EvaluationResult) -> None:
        self.result = result
        super().__init__(
            f"Skill '{result.skill_id}' blocked by policy: {result.triggered_by}"
        )


class EscalationRequired(Exception):
    """Raised by @governed when a skill call must be routed to a human approver.

    The skill function was never executed.

    Attributes:
        result: The full EvaluationResult, including triggered_by.
    """

    def __init__(self, result: EvaluationResult) -> None:
        self.result = result
        super().__init__(
            f"Skill '{result.skill_id}' requires escalation: {result.triggered_by}"
        )


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """The outcome of a single policy evaluation pass.

    Attributes:
        verdict:         Final governance verdict for this skill call.
        skill_id:        The skill that was evaluated.
        agent_id:        The agent that initiated the skill call.
        triggered_by:    The policy that caused a non-ALLOW verdict.
                         None when verdict is ALLOW; always set otherwise.
        _record_targets: Internal list of (dimension, value) pairs to be
                         persisted by engine.record() after successful
                         execution. Always empty when verdict is not ALLOW.
                         Underscore-prefixed — not part of the public contract.
    """

    verdict: Verdict
    skill_id: str
    agent_id: str
    triggered_by: DeterministicPolicy | SemanticPolicy | None
    _record_targets: list[tuple[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.verdict != "ALLOW" and self.triggered_by is None:
            raise ValueError("triggered_by must be set when verdict is not ALLOW.")
        if self.verdict == "ALLOW" and self.triggered_by is not None:
            raise ValueError("triggered_by must be None when verdict is ALLOW.")
        if self.verdict != "ALLOW" and self._record_targets:
            raise ValueError("_record_targets must be empty when verdict is not ALLOW.")


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------


class PolicyEngine:
    """Evaluates skill calls against registered policies.

    Instantiated once at application startup and reused across every skill
    call. Holds policy_store, state_store, and llm_evaluator as configured
    dependencies. Async/sync detection for stores is resolved once at
    instantiation and cached — not re-evaluated per policy or per call.

    evaluate() is always async. The decorator handles the sync surface for
    non-async skill functions.

    Args:
        policy_store:   Sync or async store for fetching PolicyConfig by skill_id.
        state_store:    Sync or async store for reading and recording dimension state.
        llm_evaluator:  Optional callable for SemanticPolicy evaluation.
                        Signature: fn(action_intent: str, rule: str) -> Verdict.
                        Sync and async callables are both supported.
                        Required only when policies include SemanticPolicy entries.
    """

    def __init__(
        self,
        policy_store: PolicyStore | AsyncPolicyStore,
        state_store: StateStore | AsyncStateStore,
        llm_evaluator: (
            Callable[[str, str], Verdict]
            | Callable[[str, str], Awaitable[Verdict]]
            | None
        ) = None,
    ) -> None:
        self._policy_store = policy_store
        self._state_store = state_store
        self._llm_evaluator = llm_evaluator

        # Cached once at instantiation — store types are immutable after init.
        self._policy_store_is_async = asyncio.iscoroutinefunction(policy_store.get)
        self._state_store_get_is_async = asyncio.iscoroutinefunction(state_store.get)
        self._state_store_record_is_async = asyncio.iscoroutinefunction(
            state_store.record
        )

    async def evaluate(
        self,
        skill_id: str,
        agent_id: str,
        kwargs: dict[str, Any],
    ) -> EvaluationResult:
        """Evaluate a skill call against all registered policies.

        Fetches PolicyConfig for skill_id, iterates the unified policy list
        in order, and short-circuits on the first non-ALLOW verdict.

        Args:
            skill_id:  The registered skill identifier.
            agent_id:  The agent initiating the call, extracted by the decorator.
            kwargs:    The full keyword arguments the skill was called with.

        Returns:
            EvaluationResult with verdict, triggered_by, and _record_targets.

        Raises:
            UnregisteredSkillError:  skill_id not found in policy_store.
            LimenexConfigError:      SemanticPolicy present but llm_evaluator
                                     is None; declared param absent from kwargs;
                                     or param value not castable to float.
        """
        if not skill_id:
            raise LimenexConfigError("skill_id must be a non-empty string.")
        if not agent_id:
            raise LimenexConfigError("agent_id must be a non-empty string.")

        if self._policy_store_is_async:
            config = await cast(AsyncPolicyStore, self._policy_store).get(skill_id)
        else:
            config = cast(PolicyStore, self._policy_store).get(skill_id)

        record_targets: list[tuple[str, float]] = []

        for policy in config.policies:

            if isinstance(policy, DeterministicPolicy):
                if policy.param is not None and policy.param not in kwargs:
                    raise LimenexConfigError(
                        f"DeterministicPolicy for skill '{skill_id}' declares "
                        f"param='{policy.param}' but it was not found in kwargs. "
                        f"Ensure the skill function accepts this argument."
                    )

                proposed: float | None = None
                if policy.param is not None:
                    try:
                        proposed = float(kwargs[policy.param])
                    except (TypeError, ValueError) as exc:
                        raise LimenexConfigError(
                            f"DeterministicPolicy for skill '{skill_id}' declares "
                            f"param='{policy.param}' but its value "
                            f"{kwargs[policy.param]!r} could not be cast to float."
                        ) from exc

                if self._state_store_get_is_async:
                    current = await cast(AsyncStateStore, self._state_store).get(
                        agent_id, policy.dimension
                    )
                else:
                    current = cast(StateStore, self._state_store).get(
                        agent_id, policy.dimension
                    )

                check_value = (current + proposed) if proposed is not None else current

                if not _OPERATOR_FNS[policy.operator](check_value, policy.value):
                    return EvaluationResult(
                        verdict=policy.breach_verdict,
                        skill_id=skill_id,
                        agent_id=agent_id,
                        triggered_by=policy,
                    )

                record_targets.append(
                    (policy.dimension, proposed if proposed is not None else 1.0)
                )

            elif isinstance(policy, SemanticPolicy):
                if self._llm_evaluator is None:
                    raise LimenexConfigError(
                        f"SemanticPolicy in skill '{skill_id}' requires an "
                        f"llm_evaluator, but none was provided to PolicyEngine."
                    )

                action_intent = f"Skill: {skill_id}\nArguments: {kwargs}"
                raw = self._llm_evaluator(action_intent, policy.rule)
                if asyncio.iscoroutine(raw):
                    raw = await raw

                if (
                    raw not in _VERDICT_SEVERITY
                    or _VERDICT_SEVERITY[raw]
                    > _VERDICT_SEVERITY[policy.verdict_ceiling]
                ):
                    final_verdict: Verdict = policy.verdict_ceiling
                else:
                    final_verdict = raw  # type: ignore[assignment]

                if final_verdict != "ALLOW":
                    return EvaluationResult(
                        verdict=final_verdict,
                        skill_id=skill_id,
                        agent_id=agent_id,
                        triggered_by=policy,
                    )

        return EvaluationResult(
            verdict="ALLOW",
            skill_id=skill_id,
            agent_id=agent_id,
            triggered_by=None,
            _record_targets=record_targets,
        )

    async def record(self, result: EvaluationResult) -> None:
        """Persist state for all DeterministicPolicy dimensions that passed.

        Called by the decorator after successful skill execution. Never called
        on BLOCK or ESCALATE — the invariants on EvaluationResult enforce this
        structurally (_record_targets is always empty for non-ALLOW results).

        Args:
            result: The EvaluationResult returned by evaluate().
        """
        for dimension, value in result._record_targets:
            if self._state_store_record_is_async:
                await cast(AsyncStateStore, self._state_store).record(
                    result.agent_id, dimension, value
                )
            else:
                cast(StateStore, self._state_store).record(
                    result.agent_id, dimension, value
                )

    def governed(
        self,
        skill_id: str,
        agent_id_param: str = "agent_id",
    ) -> Callable:
        """Decorator that intercepts a skill call and evaluates it against policy.

        Fetches the agent_id from the decorated function's arguments at call
        time using agent_id_param. Raises BlockedError or EscalationRequired
        on non-ALLOW verdicts — the skill function is never executed in either
        case. Calls engine.record() only after the skill returns successfully;
        if the skill itself raises, the exception propagates and record() is
        not called.

        Transparently supports both def and async def skill functions.
        Sync skills wrapped by @governed run evaluate() and record() via
        asyncio.run() — do not call sync governed skills from within a
        running event loop. Use async def for skills called in async contexts.

        Usage:
            @engine.governed("charge_card", agent_id_param="agent_id")
            async def charge_card(agent_id: str, amount: float) -> str: ...

        Args:
            skill_id:        The registered skill identifier. Must match a
                             skill_id known to the engine's policy_store.
            agent_id_param:  Name of the function argument that carries the
                             agent identifier. Defaults to "agent_id".

        Raises:
            BlockedError:           Verdict is BLOCK — skill was not executed.
            EscalationRequired:     Verdict is ESCALATE — skill was not executed.
            LimenexConfigError:     agent_id_param not found in skill arguments.
            UnregisteredSkillError: skill_id not registered in policy_store.
        """

        def decorator(
            fn: Callable,
        ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            sig = inspect.signature(fn)

            if asyncio.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    if agent_id_param not in bound.arguments:
                        raise LimenexConfigError(
                            f"@governed could not find agent_id_param='{agent_id_param}' "
                            f"in the arguments of skill '{fn.__name__}'. "
                            f"Ensure the skill function accepts this argument."
                        )
                    agent_id = bound.arguments[agent_id_param]
                    result = await self.evaluate(
                        skill_id, agent_id, dict(bound.arguments)
                    )
                    if result.verdict == "BLOCK":
                        raise BlockedError(result)
                    if result.verdict == "ESCALATE":
                        raise EscalationRequired(result)
                    ret = await fn(*args, **kwargs)
                    await self.record(result)
                    return ret

                return async_wrapper

            else:

                @functools.wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    if agent_id_param not in bound.arguments:
                        raise LimenexConfigError(
                            f"@governed could not find agent_id_param='{agent_id_param}' "
                            f"in the arguments of skill '{fn.__name__}'. "
                            f"Ensure the skill function accepts this argument."
                        )
                    agent_id = bound.arguments[agent_id_param]

                    async def _run() -> Any:
                        result = await self.evaluate(
                            skill_id, agent_id, dict(bound.arguments)
                        )
                        if result.verdict == "BLOCK":
                            raise BlockedError(result)
                        if result.verdict == "ESCALATE":
                            raise EscalationRequired(result)
                        ret = fn(*args, **kwargs)
                        await self.record(result)
                        return ret

                    return asyncio.run(_run())

                return sync_wrapper

        return decorator
