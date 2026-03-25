"""
comms.py — Comms skills for Limenex.

Governed execution functions for communications actions. Each skill is obtained
via a factory that binds it to a PolicyEngine instance at application startup.

Skill IDs (reference these in .limenex/policies.yaml):
    comms.send  —  send a message to a recipient via a channel
"""

from __future__ import annotations

import asyncio
from typing import Callable

from limenex.core.engine import PolicyEngine
from limenex.skills._types import ReturnT

__all__ = [
    "SEND_SKILL_ID",
    "make_send",
]

SEND_SKILL_ID: str = "comms.send"


def make_send(engine: PolicyEngine) -> Callable:
    """Return a governed send skill bound to engine.

    Call once at application startup. The returned callable is safe to reuse
    across concurrent async tasks — no shared mutable state.

    Args:
        engine: The PolicyEngine instance to bind this skill to.

    Returns:
        An async callable with signature:
        send(agent_id, channel, recipient, text, executor) -> ReturnT
    """

    @engine.governed(SEND_SKILL_ID, agent_id_param="agent_id")
    async def _governed(agent_id: str, channel: str, recipient: str, text: str) -> None:
        pass

    async def send(
        agent_id: str,
        channel: str,
        recipient: str,
        text: str,
        executor: Callable[..., ReturnT],
    ) -> ReturnT:
        """Governed skill: send a message to a recipient on behalf of an agent.

        Evaluates all policies registered under SEND_SKILL_ID before executing
        the injected executor. The executor is never called on BLOCK or
        ESCALATE verdicts.

        Policy dimensions:
            channel (str, dual-purpose): Serves as both a routing hint to the
                executor (e.g. "slack", "email", "teams") and a governable
                parameter. Cannot be used as DeterministicPolicy.param — string
                values are not numeric. Use SemanticPolicy for channel-based
                rules (e.g. "Do not send messages to external channels").
            recipient (str): Cannot be used as DeterministicPolicy.param.
                Use SemanticPolicy for recipient-based rules (e.g. "Do not
                send messages to addresses outside the organisation domain").
            text (str): Cannot be used as DeterministicPolicy.param.
                Use SemanticPolicy for content-based rules (e.g. "Do not
                send messages containing customer PII").
            Message frequency/velocity: Use DeterministicPolicy without param
                — non-projective count check (e.g. max N messages per hour).

        Governance timing: state is recorded after governance passes but before
        the executor runs. Executor failure does not roll back recorded state —
        governance tracks authorisation, not execution outcome.

        Args:
            agent_id: The agent initiating this send. Used by the engine
                to resolve and record policy state.
            channel: Delivery channel (e.g. "slack", "email", "teams").
                Forwarded to the executor for routing; govern via SemanticPolicy
                if channel-based rules are required.
            recipient: Message recipient — channel address, email, or user ID
                depending on the executor. Forwarded to the executor.
                agent_id is never forwarded.
            text: Message body. Forwarded to the executor.
            executor: Developer-injected callable that performs the actual
                message send. Receives (channel=channel, recipient=recipient,
                text=text). Sync and async callables are both supported.

        Returns:
            Whatever the executor returns.

        Raises:
            BlockedError: Policy verdict is BLOCK. Executor was not called.
            EscalationRequired: Policy verdict is ESCALATE. Executor was not called.
        """
        await _governed(
            agent_id=agent_id, channel=channel, recipient=recipient, text=text
        )
        if asyncio.iscoroutinefunction(executor):
            return await executor(channel=channel, recipient=recipient, text=text)
        return executor(channel=channel, recipient=recipient, text=text)

    return send
