"""
web.py — Web skills for Limenex.

Governed execution functions for outbound HTTP actions. Each skill is obtained
via a factory that binds it to a PolicyEngine instance at application startup.

http_get is intentionally excluded — read-only HTTP requests are too broad
to attach a bounded policy to and fail the narrow-scope principle.

Skill IDs (reference these in .limenex/policies.yaml):
    web.post  —  perform an outbound HTTP POST request
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from limenex.core.engine import PolicyEngine
from limenex.skills._types import ReturnT

__all__ = [
    "POST_SKILL_ID",
    "make_post",
]

POST_SKILL_ID: str = "web.post"


def make_post(engine: PolicyEngine) -> Callable:
    """Return a governed post skill bound to engine.

    Call once at application startup. The returned callable is safe to reuse
    across concurrent async tasks — no shared mutable state.

    Args:
        engine: The PolicyEngine instance to bind this skill to.

    Returns:
        An async callable with signature:
        post(agent_id, url, payload, executor) -> ReturnT
    """

    @engine.governed(POST_SKILL_ID, agent_id_param="agent_id")
    async def _governed(agent_id: str, url: str, payload: dict[str, Any]) -> None:
        pass

    async def post(
        agent_id: str,
        url: str,
        payload: dict[str, Any],
        executor: Callable[..., ReturnT],
    ) -> ReturnT:
        """Governed skill: perform an outbound HTTP POST request on behalf of an agent.

        Evaluates all policies registered under POST_SKILL_ID before executing
        the injected executor. The executor is never called on BLOCK or
        ESCALATE verdicts.

        Policy dimensions:
            url (str): Cannot be used as DeterministicPolicy.param — string
                values are not numeric. Use SemanticPolicy for URL-based rules
                (e.g. "Do not allow POST requests to external domains").
            payload (dict): Cannot be used as DeterministicPolicy.param.
                Use SemanticPolicy for payload-based rules (e.g. "Do not
                send requests containing customer PII").
            Request frequency/velocity: Use DeterministicPolicy without param
                — non-projective count check (e.g. max N POST requests per hour).

        Governance timing: state is recorded after governance passes but before
        the executor runs. Executor failure does not roll back recorded state —
        governance tracks authorisation, not execution outcome.

        Args:
            agent_id: The agent initiating this request. Used by the engine
                to resolve and record policy state.
            url: The target URL. Forwarded to the executor; govern via
                SemanticPolicy if URL-based rules are required.
            payload: Request body as a dict. Serialisation to JSON or another
                format is the executor's responsibility. Forwarded to the
                executor; govern via SemanticPolicy if payload inspection
                is required.
            executor: Developer-injected callable that performs the actual HTTP
                POST. Receives (url=url, payload=payload). agent_id is never
                forwarded. Sync and async callables are both supported.

        Returns:
            Whatever the executor returns.

        Raises:
            BlockedError: Policy verdict is BLOCK. Executor was not called.
            EscalationRequired: Policy verdict is ESCALATE. Executor was not called.
        """
        await _governed(agent_id=agent_id, url=url, payload=payload)
        if asyncio.iscoroutinefunction(executor):
            return await executor(url=url, payload=payload)
        return executor(url=url, payload=payload)

    return post
