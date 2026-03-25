"""
tests/test_decorator.py — Unit tests for the @governed decorator.

Covers:
    - Async skill executes on ALLOW
    - Sync skill executes on ALLOW
    - BlockedError raised on BLOCK — skill never executes
    - EscalationRequired raised on ESCALATE — skill never executes
    - record() called after successful async skill execution
    - record() called after successful sync skill execution
    - record() not called when skill itself raises
    - agent_id_param missing from skill signature raises LimenexConfigError
    - agent_id extracted correctly from positional argument
    - audit_logger.log() called on ALLOW
    - audit_logger.log() called on BLOCK
    - audit_logger.log() called on ESCALATE
    - Custom agent_id_param name resolved correctly
"""

from __future__ import annotations

import pytest

from limenex.core.engine import (
    BlockedError,
    EscalationRequired,
    LimenexConfigError,
    PolicyEngine,
)
from limenex.core.policy import (
    DeterministicPolicy,
    PolicyConfig,
    UnregisteredSkillError,
)

# ---------------------------------------------------------------------------
# Fakes (same pattern as test_engine.py)
# ---------------------------------------------------------------------------


class FakePolicyStore:
    def __init__(self, configs: dict[str, PolicyConfig]) -> None:
        self._configs = configs

    def get(self, skill_id: str) -> PolicyConfig:
        if skill_id not in self._configs:
            raise UnregisteredSkillError(skill_id)
        return self._configs[skill_id]


class FakeStateStore:
    def __init__(self, state: dict[str, dict[str, float]] | None = None) -> None:
        self._state: dict[str, dict[str, float]] = state or {}
        self.recorded: list[tuple[str, str, float]] = []

    def get(self, agent_id: str, dimension: str) -> float:
        return self._state.get(dimension, {}).get(agent_id, 0.0)

    def record(self, agent_id: str, dimension: str, value: float) -> None:
        self.recorded.append((agent_id, dimension, value))


class FakeAuditLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def log(self, result, kwargs: dict) -> None:
        self.calls.append((result.verdict, kwargs))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_engine(
    policies: list,
    state: dict | None = None,
    llm_evaluator=None,
    audit_logger=None,
) -> tuple[PolicyEngine, FakeStateStore]:
    store = FakeStateStore(state)
    engine = PolicyEngine(
        policy_store=FakePolicyStore({"test_skill": PolicyConfig(policies=policies)}),
        state_store=store,
        llm_evaluator=llm_evaluator,
        audit_logger=audit_logger,
    )
    return engine, store


def allow_policy() -> DeterministicPolicy:
    return DeterministicPolicy(
        dimension="count", operator="lte", value=100.0, breach_verdict="BLOCK"
    )


def block_policy() -> DeterministicPolicy:
    return DeterministicPolicy(
        dimension="count",
        operator="lte",
        value=-1.0,
        breach_verdict="BLOCK",
    )


def escalate_policy() -> DeterministicPolicy:
    return DeterministicPolicy(
        dimension="count",
        operator="lte",
        value=-1.0,
        breach_verdict="ESCALATE",
    )


# ---------------------------------------------------------------------------
# Async skill — ALLOW / BLOCK / ESCALATE
# ---------------------------------------------------------------------------


async def test_async_skill_executes_on_allow():
    engine, _ = make_engine(policies=[allow_policy()])
    executed = []

    @engine.governed("test_skill")
    async def my_skill(agent_id: str) -> str:
        executed.append(True)
        return "done"

    result = await my_skill(agent_id="agent_1")
    assert result == "done"
    assert executed == [True]


async def test_async_skill_raises_blocked_error():
    engine, _ = make_engine(policies=[block_policy()])
    executed = []

    @engine.governed("test_skill")
    async def my_skill(agent_id: str) -> str:
        executed.append(True)
        return "done"

    with pytest.raises(BlockedError):
        await my_skill(agent_id="agent_1")
    assert executed == []  # skill never ran


async def test_async_skill_raises_escalation_required():
    engine, _ = make_engine(policies=[escalate_policy()])
    executed = []

    @engine.governed("test_skill")
    async def my_skill(agent_id: str) -> str:
        executed.append(True)
        return "done"

    with pytest.raises(EscalationRequired):
        await my_skill(agent_id="agent_1")
    assert executed == []  # skill never ran


# ---------------------------------------------------------------------------
# Sync skill — ALLOW / BLOCK / ESCALATE
# ---------------------------------------------------------------------------


def test_sync_skill_executes_on_allow():
    engine, _ = make_engine(policies=[allow_policy()])
    executed = []

    @engine.governed("test_skill")
    def my_skill(agent_id: str) -> str:
        executed.append(True)
        return "done"

    result = my_skill(agent_id="agent_1")
    assert result == "done"
    assert executed == [True]


def test_sync_skill_raises_blocked_error():
    engine, _ = make_engine(policies=[block_policy()])
    executed = []

    @engine.governed("test_skill")
    def my_skill(agent_id: str) -> str:
        executed.append(True)
        return "done"

    with pytest.raises(BlockedError):
        my_skill(agent_id="agent_1")
    assert executed == []


def test_sync_skill_raises_escalation_required():
    engine, _ = make_engine(policies=[escalate_policy()])
    executed = []

    @engine.governed("test_skill")
    def my_skill(agent_id: str) -> str:
        executed.append(True)
        return "done"

    with pytest.raises(EscalationRequired):
        my_skill(agent_id="agent_1")
    assert executed == []


# ---------------------------------------------------------------------------
# record() behaviour
# ---------------------------------------------------------------------------


async def test_record_called_after_successful_async_skill():
    engine, store = make_engine(policies=[allow_policy()])

    @engine.governed("test_skill")
    async def my_skill(agent_id: str) -> str:
        return "done"

    await my_skill(agent_id="agent_1")
    assert ("agent_1", "count", 1.0) in store.recorded


def test_record_called_after_successful_sync_skill():
    engine, store = make_engine(policies=[allow_policy()])

    @engine.governed("test_skill")
    def my_skill(agent_id: str) -> str:
        return "done"

    my_skill(agent_id="agent_1")
    assert ("agent_1", "count", 1.0) in store.recorded


async def test_record_not_called_when_skill_raises():
    engine, store = make_engine(policies=[allow_policy()])

    @engine.governed("test_skill")
    async def my_skill(agent_id: str) -> str:
        raise RuntimeError("skill failed")

    with pytest.raises(RuntimeError, match="skill failed"):
        await my_skill(agent_id="agent_1")
    # record() must not be called if the skill raises — state should not advance
    assert store.recorded == []


# ---------------------------------------------------------------------------
# agent_id_param
# ---------------------------------------------------------------------------


async def test_missing_agent_id_param_raises_config_error():
    engine, _ = make_engine(policies=[allow_policy()])

    @engine.governed("test_skill", agent_id_param="agent_id")
    async def my_skill(name: str) -> str:  # no agent_id argument
        return "done"

    with pytest.raises(LimenexConfigError, match="agent_id_param"):
        await my_skill(name="test")


async def test_agent_id_extracted_from_positional_arg():
    engine, store = make_engine(policies=[allow_policy()])

    @engine.governed("test_skill")
    async def my_skill(agent_id: str) -> str:
        return "done"

    # Call with positional, not keyword
    await my_skill("agent_1")
    assert ("agent_1", "count", 1.0) in store.recorded


async def test_custom_agent_id_param_name():
    engine, store = make_engine(policies=[allow_policy()])

    @engine.governed("test_skill", agent_id_param="caller_id")
    async def my_skill(caller_id: str) -> str:
        return "done"

    await my_skill(caller_id="agent_1")
    assert ("agent_1", "count", 1.0) in store.recorded


# ---------------------------------------------------------------------------
# audit_logger
# ---------------------------------------------------------------------------


async def test_audit_logger_called_on_allow():
    logger = FakeAuditLogger()
    engine, _ = make_engine(policies=[allow_policy()], audit_logger=logger)

    @engine.governed("test_skill")
    async def my_skill(agent_id: str) -> str:
        return "done"

    await my_skill(agent_id="agent_1")
    assert len(logger.calls) == 1
    assert logger.calls[0][0] == "ALLOW"


async def test_audit_logger_called_on_block():
    logger = FakeAuditLogger()
    engine, _ = make_engine(policies=[block_policy()], audit_logger=logger)

    @engine.governed("test_skill")
    async def my_skill(agent_id: str) -> str:
        return "done"

    with pytest.raises(BlockedError):
        await my_skill(agent_id="agent_1")
    assert len(logger.calls) == 1
    assert logger.calls[0][0] == "BLOCK"


async def test_audit_logger_called_on_escalate():
    logger = FakeAuditLogger()
    engine, _ = make_engine(policies=[escalate_policy()], audit_logger=logger)

    @engine.governed("test_skill")
    async def my_skill(agent_id: str) -> str:
        return "done"

    with pytest.raises(EscalationRequired):
        await my_skill(agent_id="agent_1")
    assert len(logger.calls) == 1
    assert logger.calls[0][0] == "ESCALATE"
