"""
tests/test_engine.py — Unit tests for PolicyEngine evaluation logic.

Covers:
    - Auto-allow: all policies pass
    - Auto-block: deterministic breach with BLOCK verdict
    - Escalation: deterministic breach with ESCALATE verdict
    - Projective checks: proposed value added to current state
    - Non-projective checks: current state only, records 1.0
    - Short-circuit: engine stops at first non-ALLOW verdict
    - SemanticPolicy: allow, block, escalate, verdict capping, unknown verdict
    - Missing llm_evaluator with SemanticPolicy
    - Async llm_evaluator
    - Async policy_store and state_store
    - Missing param in kwargs
    - Non-numeric param value
    - Empty skill_id / agent_id
    - UnregisteredSkillError propagation
    - record() persists correct dimensions and values
    - record() is a no-op on empty _record_targets
"""

from __future__ import annotations

import pytest

from limenex.core.engine import (
    LimenexConfigError,
    PolicyEngine,
)
from limenex.core.policy import (
    DeterministicPolicy,
    PolicyConfig,
    SemanticPolicy,
    UnregisteredSkillError,
)

# ---------------------------------------------------------------------------
# Minimal in-memory fakes
# ---------------------------------------------------------------------------


class FakePolicyStore:
    def __init__(self, configs: dict[str, PolicyConfig]) -> None:
        self._configs = configs

    def get(self, skill_id: str) -> PolicyConfig:
        if skill_id not in self._configs:
            raise UnregisteredSkillError(skill_id)
        return self._configs[skill_id]


class FakeAsyncPolicyStore:
    def __init__(self, configs: dict[str, PolicyConfig]) -> None:
        self._configs = configs

    async def get(self, skill_id: str) -> PolicyConfig:
        if skill_id not in self._configs:
            raise UnregisteredSkillError(skill_id)
        return self._configs[skill_id]


class FakeStateStore:
    def __init__(self, state: dict[str, dict[str, float]] | None = None) -> None:
        # Layout: dimension -> agent_id -> value
        self._state: dict[str, dict[str, float]] = state or {}
        self.recorded: list[tuple[str, str, float]] = []

    def get(self, agent_id: str, dimension: str) -> float:
        return self._state.get(dimension, {}).get(agent_id, 0.0)

    def record(self, agent_id: str, dimension: str, value: float) -> None:
        self.recorded.append((agent_id, dimension, value))


class FakeAsyncStateStore:
    def __init__(self, state: dict[str, dict[str, float]] | None = None) -> None:
        self._state: dict[str, dict[str, float]] = state or {}
        self.recorded: list[tuple[str, str, float]] = []

    async def get(self, agent_id: str, dimension: str) -> float:
        return self._state.get(dimension, {}).get(agent_id, 0.0)

    async def record(self, agent_id: str, dimension: str, value: float) -> None:
        self.recorded.append((agent_id, dimension, value))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_engine(
    policies: list,
    state: dict | None = None,
    llm_evaluator=None,
) -> tuple[PolicyEngine, FakeStateStore]:
    store = FakeStateStore(state)
    engine = PolicyEngine(
        policy_store=FakePolicyStore({"test_skill": PolicyConfig(policies=policies)}),
        state_store=store,
        llm_evaluator=llm_evaluator,
    )
    return engine, store


# ---------------------------------------------------------------------------
# Auto-allow
# ---------------------------------------------------------------------------


async def test_allow_no_policies():
    engine, store = make_engine(policies=[])
    result = await engine.evaluate("test_skill", "agent_1", {})
    assert result.verdict == "ALLOW"
    assert result.triggered_by is None
    assert result._record_targets == []


async def test_allow_deterministic_within_limit():
    policy = DeterministicPolicy(
        dimension="spend",
        operator="lte",
        value=100.0,
        param="amount",
        breach_verdict="BLOCK",
    )
    engine, store = make_engine(policies=[policy])
    result = await engine.evaluate("test_skill", "agent_1", {"amount": 50.0})
    assert result.verdict == "ALLOW"
    assert result._record_targets == [("spend", 50.0)]


# ---------------------------------------------------------------------------
# Auto-block
# ---------------------------------------------------------------------------


async def test_block_deterministic_exceeds_limit():
    policy = DeterministicPolicy(
        dimension="spend",
        operator="lte",
        value=100.0,
        param="amount",
        breach_verdict="BLOCK",
    )
    engine, _ = make_engine(policies=[policy])
    result = await engine.evaluate("test_skill", "agent_1", {"amount": 150.0})
    assert result.verdict == "BLOCK"
    assert result.triggered_by is policy
    assert result._record_targets == []


async def test_block_deterministic_with_accumulated_state():
    policy = DeterministicPolicy(
        dimension="spend",
        operator="lte",
        value=100.0,
        param="amount",
        breach_verdict="BLOCK",
    )
    # agent already has 80.0 accumulated
    engine, _ = make_engine(
        policies=[policy],
        state={"spend": {"agent_1": 80.0}},
    )
    result = await engine.evaluate("test_skill", "agent_1", {"amount": 30.0})
    assert result.verdict == "BLOCK"  # 80 + 30 = 110 > 100


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


async def test_escalate_deterministic():
    policy = DeterministicPolicy(
        dimension="count",
        operator="lte",
        value=5.0,
        breach_verdict="ESCALATE",
    )
    engine, _ = make_engine(
        policies=[policy],
        state={"count": {"agent_1": 6.0}},
    )
    result = await engine.evaluate("test_skill", "agent_1", {})
    assert result.verdict == "ESCALATE"
    assert result.triggered_by is policy


# ---------------------------------------------------------------------------
# Projective vs non-projective
# ---------------------------------------------------------------------------


async def test_projective_check_adds_proposed_to_current():
    policy = DeterministicPolicy(
        dimension="spend",
        operator="lte",
        value=100.0,
        param="amount",
        breach_verdict="BLOCK",
    )
    engine, store = make_engine(
        policies=[policy],
        state={"spend": {"agent_1": 60.0}},
    )
    result = await engine.evaluate("test_skill", "agent_1", {"amount": 30.0})
    assert result.verdict == "ALLOW"  # 60 + 30 = 90 <= 100
    assert result._record_targets == [("spend", 30.0)]


async def test_non_projective_check_uses_current_state_only():
    policy = DeterministicPolicy(
        dimension="count",
        operator="lte",
        value=5.0,
        breach_verdict="ESCALATE",
    )
    engine, store = make_engine(
        policies=[policy],
        state={"count": {"agent_1": 3.0}},
    )
    result = await engine.evaluate("test_skill", "agent_1", {})
    assert result.verdict == "ALLOW"
    # Non-projective records 1.0
    assert result._record_targets == [("count", 1.0)]


# ---------------------------------------------------------------------------
# Short-circuit
# ---------------------------------------------------------------------------


async def test_short_circuit_on_first_breach():
    p1 = DeterministicPolicy(
        dimension="spend",
        operator="lte",
        value=100.0,
        param="amount",
        breach_verdict="BLOCK",
    )
    p2 = DeterministicPolicy(
        dimension="count",
        operator="lte",
        value=5.0,
        breach_verdict="ESCALATE",
    )
    engine, _ = make_engine(policies=[p1, p2])
    result = await engine.evaluate("test_skill", "agent_1", {"amount": 200.0})
    assert result.verdict == "BLOCK"
    assert result.triggered_by is p1  # p2 never evaluated


# ---------------------------------------------------------------------------
# SemanticPolicy
# ---------------------------------------------------------------------------


async def test_semantic_allow():
    policy = SemanticPolicy(rule="Do not approve X.", verdict_ceiling="BLOCK")
    engine, _ = make_engine(
        policies=[policy],
        llm_evaluator=lambda intent, rule: "ALLOW",
    )
    result = await engine.evaluate("test_skill", "agent_1", {})
    assert result.verdict == "ALLOW"


async def test_semantic_block():
    policy = SemanticPolicy(rule="Do not approve X.", verdict_ceiling="BLOCK")
    engine, _ = make_engine(
        policies=[policy],
        llm_evaluator=lambda intent, rule: "BLOCK",
    )
    result = await engine.evaluate("test_skill", "agent_1", {})
    assert result.verdict == "BLOCK"
    assert result.triggered_by is policy


async def test_semantic_escalate():
    policy = SemanticPolicy(rule="Do not approve X.", verdict_ceiling="ESCALATE")
    engine, _ = make_engine(
        policies=[policy],
        llm_evaluator=lambda intent, rule: "ESCALATE",
    )
    result = await engine.evaluate("test_skill", "agent_1", {})
    assert result.verdict == "ESCALATE"
    assert result.triggered_by is policy


async def test_semantic_verdict_capped_to_ceiling():
    # LLM returns BLOCK but ceiling is ESCALATE — should be capped to ESCALATE
    policy = SemanticPolicy(rule="Do not approve X.", verdict_ceiling="ESCALATE")
    engine, _ = make_engine(
        policies=[policy],
        llm_evaluator=lambda intent, rule: "BLOCK",
    )
    result = await engine.evaluate("test_skill", "agent_1", {})
    assert result.verdict == "ESCALATE"


async def test_semantic_unknown_verdict_falls_back_to_ceiling():
    policy = SemanticPolicy(rule="Do not approve X.", verdict_ceiling="BLOCK")
    engine, _ = make_engine(
        policies=[policy],
        llm_evaluator=lambda intent, rule: "GIBBERISH",
    )
    result = await engine.evaluate("test_skill", "agent_1", {})
    assert result.verdict == "BLOCK"


async def test_semantic_async_llm_evaluator():
    async def async_evaluator(intent: str, rule: str) -> str:
        return "BLOCK"

    policy = SemanticPolicy(rule="Do not approve X.", verdict_ceiling="BLOCK")
    engine, _ = make_engine(policies=[policy], llm_evaluator=async_evaluator)
    result = await engine.evaluate("test_skill", "agent_1", {})
    assert result.verdict == "BLOCK"


async def test_semantic_missing_llm_evaluator_raises():
    policy = SemanticPolicy(rule="Do not approve X.", verdict_ceiling="BLOCK")
    engine, _ = make_engine(policies=[policy], llm_evaluator=None)
    with pytest.raises(LimenexConfigError, match="llm_evaluator"):
        await engine.evaluate("test_skill", "agent_1", {})


# ---------------------------------------------------------------------------
# Async stores
# ---------------------------------------------------------------------------


async def test_async_policy_store():
    policy = DeterministicPolicy(
        dimension="spend",
        operator="lte",
        value=100.0,
        param="amount",
        breach_verdict="BLOCK",
    )
    async_store = FakeAsyncPolicyStore({"test_skill": PolicyConfig(policies=[policy])})
    state_store = FakeStateStore()
    engine = PolicyEngine(policy_store=async_store, state_store=state_store)
    result = await engine.evaluate("test_skill", "agent_1", {"amount": 50.0})
    assert result.verdict == "ALLOW"


async def test_async_state_store():
    policy = DeterministicPolicy(
        dimension="spend",
        operator="lte",
        value=100.0,
        param="amount",
        breach_verdict="BLOCK",
    )
    policy_store = FakePolicyStore({"test_skill": PolicyConfig(policies=[policy])})
    async_state = FakeAsyncStateStore()
    engine = PolicyEngine(policy_store=policy_store, state_store=async_state)
    result = await engine.evaluate("test_skill", "agent_1", {"amount": 50.0})
    assert result.verdict == "ALLOW"


# ---------------------------------------------------------------------------
# Config errors
# ---------------------------------------------------------------------------


async def test_missing_param_in_kwargs_raises():
    policy = DeterministicPolicy(
        dimension="spend",
        operator="lte",
        value=100.0,
        param="amount",
        breach_verdict="BLOCK",
    )
    engine, _ = make_engine(policies=[policy])
    with pytest.raises(LimenexConfigError, match="amount"):
        await engine.evaluate("test_skill", "agent_1", {})


async def test_non_numeric_param_raises():
    policy = DeterministicPolicy(
        dimension="spend",
        operator="lte",
        value=100.0,
        param="amount",
        breach_verdict="BLOCK",
    )
    engine, _ = make_engine(policies=[policy])
    with pytest.raises(LimenexConfigError, match="could not be cast to float"):
        await engine.evaluate("test_skill", "agent_1", {"amount": "not_a_number"})


async def test_empty_skill_id_raises():
    engine, _ = make_engine(policies=[])
    with pytest.raises(LimenexConfigError, match="skill_id"):
        await engine.evaluate("", "agent_1", {})


async def test_empty_agent_id_raises():
    engine, _ = make_engine(policies=[])
    with pytest.raises(LimenexConfigError, match="agent_id"):
        await engine.evaluate("test_skill", "", {})


async def test_unregistered_skill_raises():
    engine, _ = make_engine(policies=[])
    with pytest.raises(UnregisteredSkillError):
        await engine.evaluate("unknown_skill", "agent_1", {})


# ---------------------------------------------------------------------------
# record()
# ---------------------------------------------------------------------------


async def test_record_persists_projective_value():
    policy = DeterministicPolicy(
        dimension="spend",
        operator="lte",
        value=100.0,
        param="amount",
        breach_verdict="BLOCK",
    )
    engine, store = make_engine(policies=[policy])
    result = await engine.evaluate("test_skill", "agent_1", {"amount": 40.0})
    assert result.verdict == "ALLOW"
    await engine.record(result)
    assert ("agent_1", "spend", 40.0) in store.recorded


async def test_record_persists_non_projective_as_one():
    policy = DeterministicPolicy(
        dimension="count",
        operator="lte",
        value=10.0,
        breach_verdict="ESCALATE",
    )
    engine, store = make_engine(policies=[policy])
    result = await engine.evaluate("test_skill", "agent_1", {})
    assert result.verdict == "ALLOW"
    await engine.record(result)
    assert ("agent_1", "count", 1.0) in store.recorded


async def test_record_noop_on_empty_targets():
    engine, store = make_engine(policies=[])
    result = await engine.evaluate("test_skill", "agent_1", {})
    await engine.record(result)
    assert store.recorded == []


async def test_record_uses_async_state_store():
    policy = DeterministicPolicy(
        dimension="spend",
        operator="lte",
        value=100.0,
        param="amount",
        breach_verdict="BLOCK",
    )
    policy_store = FakePolicyStore({"test_skill": PolicyConfig(policies=[policy])})
    async_state = FakeAsyncStateStore()
    engine = PolicyEngine(policy_store=policy_store, state_store=async_state)
    result = await engine.evaluate("test_skill", "agent_1", {"amount": 25.0})
    await engine.record(result)
    assert ("agent_1", "spend", 25.0) in async_state.recorded


async def test_multiple_passing_policies_accumulate_record_targets():
    p1 = DeterministicPolicy(
        dimension="spend",
        operator="lte",
        value=100.0,
        param="amount",
        breach_verdict="BLOCK",
    )
    p2 = DeterministicPolicy(
        dimension="count",
        operator="lte",
        value=10.0,
        breach_verdict="ESCALATE",
    )
    engine, _ = make_engine(policies=[p1, p2])
    result = await engine.evaluate("test_skill", "agent_1", {"amount": 40.0})
    assert result.verdict == "ALLOW"
    assert ("spend", 40.0) in result._record_targets
    assert ("count", 1.0) in result._record_targets
