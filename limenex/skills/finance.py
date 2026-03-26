"""
finance.py — Finance skills for Limenex.

Governed execution functions for financial actions. Each skill is obtained
via a factory that binds it to a PolicyEngine instance at application startup.

Skill IDs (reference these in .limenex/policies.yaml):
    finance.charge  —  charge a payment against a third party
    finance.spend   —  outbound spend from the operator's own budget

Policy guidance:
    Numeric parameters (amount, amount_usd) support projective
    DeterministicPolicy checks — the engine accumulates proposed values
    against recorded state before evaluating the threshold.

    String parameters (currency, service) support exact-string
    DeterministicPolicy checks using in/not_in operators (e.g. restrict
    to an approved currency set, or block spend on specific services).
    For broader content- or intent-based rules on string parameters,
    use SemanticPolicy.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from limenex.core.engine import PolicyEngine
from limenex.skills._types import ReturnT

__all__ = [
    "CHARGE_SKILL_ID",
    "SPEND_SKILL_ID",
    "make_charge",
    "make_spend",
]

CHARGE_SKILL_ID: str = "finance.charge"
SPEND_SKILL_ID: str = "finance.spend"


def make_charge(engine: PolicyEngine) -> Callable:
    """Return a governed charge skill bound to engine.

    Call once at application startup. The returned callable is safe to reuse
    across concurrent async tasks — no shared mutable state.

    Args:
        engine: The PolicyEngine instance to bind this skill to.

    Returns:
        An async callable with signature:
        charge(agent_id, amount, currency, executor) -> ReturnT
    """

    @engine.governed(CHARGE_SKILL_ID, agent_id_param="agent_id")
    async def _governed(agent_id: str, amount: float, currency: str) -> None:
        pass

    async def charge(
        agent_id: str,
        amount: float,
        currency: str,
        executor: Callable[..., ReturnT],
    ) -> ReturnT:
        """Governed skill: charge a payment on behalf of an agent.

        Evaluates all policies registered under CHARGE_SKILL_ID before
        executing the injected executor. The executor is never called on
        BLOCK or ESCALATE verdicts.

        Policy dimensions:
            amount (float, projective): Use DeterministicPolicy(param="amount")
                to enforce cumulative spend limits. The engine adds the proposed
                amount to the agent's recorded state before evaluating the policy.

            currency (str): Supports exact-string DeterministicPolicy checks
                via in/not_in operators. Example: restrict to an approved
                currency set using DeterministicPolicy(operator="in",
                values=frozenset({"USD", "GBP"}), param="currency").
                Matching is exact and case-sensitive. For broader rules
                (e.g. "only allow stable fiat currencies"), use SemanticPolicy.

        Governance timing: state is recorded after governance passes but before
        the executor runs. Executor failure does not roll back recorded state —
        governance tracks authorisation, not execution outcome.

        Args:
            agent_id:  The agent initiating this charge. Used by the engine
                       to resolve and record policy state.
            amount:    Payment amount. Maps to DeterministicPolicy.param="amount"
                       for projective spend limit checks.
            currency:  ISO 4217 currency code (e.g. "USD", "GBP"). Supports
                       exact-string in/not_in policy checks. Forwarded to the
                       executor as a routing hint.
            executor:  Developer-injected callable that performs the actual
                       payment. Receives (amount=amount, currency=currency).
                       agent_id is never forwarded. Sync and async callables
                       are both supported.

        Returns:
            Whatever the executor returns.

        Raises:
            BlockedError:        Policy verdict is BLOCK. Executor was not called.
            EscalationRequired:  Policy verdict is ESCALATE. Executor was not called.
        """
        await _governed(agent_id=agent_id, amount=amount, currency=currency)
        if asyncio.iscoroutinefunction(executor):
            return await executor(amount=amount, currency=currency)
        return executor(amount=amount, currency=currency)

    return charge


def make_spend(engine: PolicyEngine) -> Callable:
    """Return a governed spend skill bound to engine.

    Call once at application startup. The returned callable is safe to reuse
    across concurrent async tasks — no shared mutable state.

    Args:
        engine: The PolicyEngine instance to bind this skill to.

    Returns:
        An async callable with signature:
        spend(agent_id, service, amount_usd, executor) -> ReturnT
    """

    @engine.governed(SPEND_SKILL_ID, agent_id_param="agent_id")
    async def _governed(agent_id: str, service: str, amount_usd: float) -> None:
        pass

    async def spend(
        agent_id: str,
        service: str,
        amount_usd: float,
        executor: Callable[..., ReturnT],
    ) -> ReturnT:
        """Governed skill: outbound spend from operator budget on a service.

        Covers all cases where an agent authorises spend from the operator's
        own budget — API credit top-ups, data purchases, compute provisioning,
        or any other service expenditure. The executor is never called on
        BLOCK or ESCALATE verdicts.

        Policy dimensions:
            amount_usd (float, projective): Use DeterministicPolicy(param="amount_usd")
                to enforce cumulative outbound budget limits. The engine adds the
                proposed amount to the agent's recorded state before evaluating
                the policy.

            service (str): Supports exact-string DeterministicPolicy checks
                via in/not_in operators. Example: restrict to an approved
                service set using DeterministicPolicy(operator="in",
                values=frozenset({"openai", "aws"}), param="service"), or
                block specific services using DeterministicPolicy(operator="not_in",
                values=frozenset({"unapproved-vendor"}), param="service").
                Matching is exact and case-sensitive. For broader rules
                (e.g. "do not approve spend on unapproved services"), use
                SemanticPolicy.

        Governance timing: state is recorded after governance passes but before
        the executor runs. Executor failure does not roll back recorded state —
        governance tracks authorisation, not execution outcome.

        Args:
            agent_id:    The agent initiating this spend. Used by the engine
                         to resolve and record policy state.
            service:     The target service (e.g. "openai", "aws"). Supports
                         exact-string in/not_in policy checks. Forwarded to
                         the executor as a routing hint.
            amount_usd:  Spend amount in USD. Maps to
                         DeterministicPolicy.param="amount_usd" for projective
                         budget limit checks.
            executor:    Developer-injected callable that performs the actual
                         spend action. Receives (service=service,
                         amount_usd=amount_usd). agent_id is never forwarded.
                         Sync and async callables are both supported.

        Returns:
            Whatever the executor returns.

        Raises:
            BlockedError:        Policy verdict is BLOCK. Executor was not called.
            EscalationRequired:  Policy verdict is ESCALATE. Executor was not called.
        """
        await _governed(agent_id=agent_id, service=service, amount_usd=amount_usd)
        if asyncio.iscoroutinefunction(executor):
            return await executor(service=service, amount_usd=amount_usd)
        return executor(service=service, amount_usd=amount_usd)

    return spend
