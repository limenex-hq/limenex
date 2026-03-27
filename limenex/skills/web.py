"""
web.py — Web skills for Limenex.

Governed execution functions for outbound HTTP actions. Each skill is
obtained via a factory that binds it to a PolicyEngine instance at
application startup.

Skill IDs (reference these in .limenex/policies.yaml):
    web.post — send an HTTP POST request to a named destination

Policy guidance:
    The destination parameter supports exact-string DeterministicPolicy
    checks using in/not_in operators. This enables destination allowlists
    and blocklists without routing to SemanticPolicy.

    Matching is exact and case-sensitive. The destination string is a
    developer-defined key (e.g. "ibkr", "yahoo") that maps to an executor
    in the registry supplied at factory time. The actual URL is an internal
    concern of the executor — it is never exposed to the agent or the
    governance layer.

    For rules about what may be sent in the payload (e.g. "do not transmit
    PII externally"), use SemanticPolicy targeting the payload parameter.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from limenex.core.engine import PolicyEngine
from limenex.skills._exceptions import UnregisteredExecutorError
from limenex.skills._types import ReturnT

__all__ = [
    "POST_SKILL_ID",
    "make_post",
]

POST_SKILL_ID: str = "web.post"


def make_post(engine: PolicyEngine, registry: dict[str, Callable]) -> Callable:
    """Return a governed HTTP POST skill bound to engine.

    Call once at application startup. The returned callable is safe to reuse
    across concurrent async tasks — no shared mutable state.

    Args:
        engine:   The PolicyEngine instance to bind this skill to.
        registry: Mapping of destination name to executor callable.
                  Keys must match the destination strings the agent will pass
                  at call time (e.g. {"ibkr": post_to_ibkr,
                  "yahoo": post_to_yahoo}).
                  Executors receive (payload=payload). agent_id and
                  destination are never forwarded. The target URL is an
                  internal concern of each executor.
                  Sync and async callables are both supported.

    Returns:
        An async callable with signature:
        post(agent_id, destination, payload) -> ReturnT
    """

    @engine.governed(POST_SKILL_ID, agent_id_param="agent_id")
    async def _governed(
        agent_id: str, destination: str, payload: dict[str, Any]
    ) -> None:
        pass

    async def post(
        agent_id: str,
        destination: str,
        payload: dict[str, Any],
    ) -> ReturnT:
        """Governed skill: send an HTTP POST request to a named destination.

        Evaluates all policies registered under POST_SKILL_ID before
        dispatching to the executor registered for destination. The executor
        is never called on BLOCK or ESCALATE verdicts.

        Policy dimensions:
            destination (str): Supports exact-string DeterministicPolicy
                checks via in/not_in operators. Example: restrict outbound
                POSTs to an approved destination allowlist using
                DeterministicPolicy(
                    operator="in",
                    values=frozenset({"ibkr", "yahoo"}),
                    param="destination",
                    breach_verdict="BLOCK",
                ).
                Matching is exact and case-sensitive. The destination key
                is developer-defined — the actual URL is owned by the
                executor and never appears in the governance surface.

            payload (dict): Not a good fit for DeterministicPolicy. Use
                SemanticPolicy for content-based rules (e.g. "do not
                transmit PII or credentials in outbound requests").

            frequency / count: Use a numeric DeterministicPolicy with
                param=None to govern how often post is called
                (e.g. daily outbound request count).

        Governance timing: state is recorded after governance passes but before
        the executor runs. Executor failure does not roll back recorded state —
        governance tracks authorisation, not execution outcome.

        Args:
            agent_id:     The agent initiating this POST. Used by the engine
                          to resolve and record policy state.
            destination:  Named target (e.g. "ibkr", "yahoo"). Must match a
                          key in the registry supplied to make_post. Supports
                          exact-string in/not_in policy checks. Never
                          forwarded to the executor.
            payload:      Request body as a dict. Forwarded to the executor.
                          Use SemanticPolicy for content-based governance.

        Returns:
            Whatever the executor returns.

        Raises:
            UnregisteredExecutorError: destination has no entry in the
                                       registry supplied to make_post.
                                       Executor was not called and no state
                                       was recorded.
            BlockedError:              Policy verdict is BLOCK. Executor was not called.
            EscalationRequired:        Policy verdict is ESCALATE. Executor was not called.
        """
        if destination not in registry:
            raise UnregisteredExecutorError(POST_SKILL_ID, destination)
        executor = registry[destination]
        await _governed(agent_id=agent_id, destination=destination, payload=payload)
        if asyncio.iscoroutinefunction(executor):
            return await executor(payload=payload)
        return executor(payload=payload)

    return post
