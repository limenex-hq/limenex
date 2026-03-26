"""
comms.py — Communications skills for Limenex.

Governed execution functions for outbound messaging actions. Each skill is
obtained via a factory that binds it to a PolicyEngine instance at
application startup.

Skill IDs (reference these in .limenex/policies.yaml):
    comms.send  —  send a message to a recipient over a channel

Policy guidance:
    String parameters (channel, recipient) now support exact-string
    DeterministicPolicy checks using in/not_in operators. This enables
    recipient blocklists, channel allowlists, and similar membership-based
    controls without routing to SemanticPolicy.

    Matching is exact and case-sensitive. Limenex does not parse, normalise,
    or extract components from recipient addresses or channel identifiers
    before comparison. A recipient policy targeting "alice@baddomain.com"
    matches only that exact address — domain-level blocking (matching any
    address at baddomain.com) requires SemanticPolicy or pre-processing the
    recipient before calling the skill.

    For content-based rules (e.g. "do not transmit PII"), use SemanticPolicy
    targeting the text parameter. For structural matching such as domain-level
    blocking, pre-process the value before calling the skill (e.g. extract the
    domain from the recipient address) and use in/not_in on the extracted value.

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
        """Governed skill: send a message to a recipient over a channel.

        Evaluates all policies registered under SEND_SKILL_ID before
        executing the injected executor. The executor is never called on
        BLOCK or ESCALATE verdicts.

        Policy dimensions:
            recipient (str): Supports exact-string DeterministicPolicy checks
                via in/not_in operators. Example: block sending to a known
                set of bad actors using DeterministicPolicy(
                    operator="not_in",
                    values=frozenset({"alice@baddomain.com", "bob@baddomain.com"}),
                    param="recipient",
                    breach_verdict="BLOCK",
                ).
                Matching is exact and case-sensitive — "alice@baddomain.com"
                does NOT match "baddomain.com" in the values set. For
                domain-level blocking, extract the domain before calling the
                skill (e.g. domain = recipient.split("@")[-1]) and pass the
                extracted value as the governed parameter instead.

            channel (str): Supports exact-string DeterministicPolicy checks
                via in/not_in operators. Example: restrict sending to approved
                channels using DeterministicPolicy(
                    operator="in",
                    values=frozenset({"email", "slack"}),
                    param="channel",
                    breach_verdict="BLOCK",
                ).
                Matching is exact and case-sensitive.

            text (str): Not a good fit for DeterministicPolicy. Use
                SemanticPolicy for content-based rules (e.g. "do not
                transmit PII or credentials in outbound messages").

            frequency / count: Use a numeric DeterministicPolicy with
                param=None to govern how often send is called
                (e.g. daily outbound message count).

        Args:
            agent_id:   The agent initiating this send. Used by the engine
                        to resolve and record policy state.
            channel:    Delivery channel (e.g. "email", "slack", "sms").
                        Supports exact-string in/not_in policy checks.
                        Forwarded to the executor.
            recipient:  Destination address or identifier
                        (e.g. "alice@example.com", "ops-team"). Supports
                        exact-string in/not_in policy checks. Forwarded to
                        the executor. Matching is against the exact string
                        as provided — no domain extraction is performed.
            text:       Message body. Forwarded to the executor. Use
                        SemanticPolicy for content-based governance.
            executor:   Developer-injected callable that performs the actual
                        message delivery. Receives (channel=channel,
                        recipient=recipient, text=text). agent_id is never
                        forwarded. Sync and async callables are both supported.

        Returns:
            Whatever the executor returns.

        Raises:
            BlockedError:        Policy verdict is BLOCK. Executor was not called.
            EscalationRequired:  Policy verdict is ESCALATE. Executor was not called.
        """
        await _governed(
            agent_id=agent_id, channel=channel, recipient=recipient, text=text
        )
        if asyncio.iscoroutinefunction(executor):
            return await executor(channel=channel, recipient=recipient, text=text)
        return executor(channel=channel, recipient=recipient, text=text)

    return send
