"""
web.py — Web skills for Limenex.

Governed execution functions for outbound HTTP actions. Each skill is
obtained via a factory that binds it to a PolicyEngine instance at
application startup.

Skill IDs (reference these in .limenex/policies.yaml):
    web.post  —  send an HTTP POST request to a URL

Policy guidance:
    The url parameter now supports exact-string DeterministicPolicy checks
    using in/not_in operators. This enables URL allowlists and blocklists
    without routing to SemanticPolicy.

    Matching is exact and case-sensitive. Limenex does not parse, normalise,
    or canonicalise URLs before comparison. "https://api.example.com/v1/send"
    and "https://api.example.com/v1/send/" are treated as distinct strings.
    Scheme, host, path, and query string are all part of the comparison.
    For host-only or prefix-based URL governance, extract the relevant
    component before calling the skill (e.g. parse just the hostname) and
    use in/not_in on the extracted value.

    For rules about what may be sent in the payload (e.g. "do not transmit
    PII externally"), use SemanticPolicy targeting the payload parameter.
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
    """Return a governed HTTP POST skill bound to engine.

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
        """Governed skill: send an HTTP POST request to a URL.

        Evaluates all policies registered under POST_SKILL_ID before
        executing the injected executor. The executor is never called on
        BLOCK or ESCALATE verdicts.

        Policy dimensions:
            url (str): Supports exact-string DeterministicPolicy checks via
                in/not_in operators. Example: restrict outbound POSTs to an
                approved URL allowlist using DeterministicPolicy(
                    operator="in",
                    values=frozenset({
                        "https://api.example.com/v1/send",
                        "https://api.example.com/v1/submit",
                    }),
                    param="url",
                    breach_verdict="BLOCK",
                ).
                Matching is exact and case-sensitive. URLs are compared as
                provided with no normalisation — trailing slashes, casing,
                and query strings are all significant. For host-only or
                prefix-based URL rules, extract the relevant component
                before calling the skill (e.g. parse just the hostname)
                and use in/not_in on the extracted value.

            payload (dict): Not a good fit for DeterministicPolicy. Use
                SemanticPolicy for content-based rules (e.g. "do not
                transmit PII or credentials in outbound requests").

            frequency / count: Use a numeric DeterministicPolicy with
                param=None to govern how often post is called
                (e.g. daily outbound request count).

        Args:
            agent_id:  The agent initiating this POST. Used by the engine
                       to resolve and record policy state.
            url:       The target URL. Supports exact-string in/not_in
                       policy checks. Compared as provided with no
                       normalisation. Forwarded to the executor.
            payload:   Request body as a dict. Forwarded to the executor.
                       Use SemanticPolicy for content-based governance.
            executor:  Developer-injected callable that performs the actual
                       HTTP request. Receives (url=url, payload=payload).
                       agent_id is never forwarded. Sync and async callables
                       are both supported.

        Returns:
            Whatever the executor returns.

        Raises:
            BlockedError:        Policy verdict is BLOCK. Executor was not called.
            EscalationRequired:  Policy verdict is ESCALATE. Executor was not called.
        """
        await _governed(agent_id=agent_id, url=url, payload=payload)
        if asyncio.iscoroutinefunction(executor):
            return await executor(url=url, payload=payload)
        return executor(url=url, payload=payload)

    return post
