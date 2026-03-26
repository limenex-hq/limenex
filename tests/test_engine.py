import math
import warnings
from typing import Any

import pytest

from limenex.core.engine import EvaluationResult, LimenexConfigError, PolicyEngine
from limenex.core.policy import (
    DeterministicPolicy,
    LimenexConfigWarning,
    PolicyConfig,
    SemanticPolicy,
    UnregisteredSkillError,
)


class InMemoryPolicyStore:
    def __init__(self, configs: dict[str, PolicyConfig]) -> None:
        self.configs = configs

    def get(self, skill_id: str) -> PolicyConfig:
        if skill_id not in self.configs:
            raise UnregisteredSkillError(skill_id)
        return self.configs[skill_id]


class AsyncInMemoryPolicyStore:
    def __init__(self, configs: dict[str, PolicyConfig]) -> None:
        self.configs = configs

    async def get(self, skill_id: str) -> PolicyConfig:
        if skill_id not in self.configs:
            raise UnregisteredSkillError(skill_id)
        return self.configs[skill_id]


class SpyStateStore:
    def __init__(self, initial: dict[tuple[str, str], float] | None = None) -> None:
        self._state = dict(initial or {})
        self.get_calls: list[tuple[str, str]] = []
        self.record_calls: list[tuple[str, str, float]] = []

    def get(self, agent_id: str, dimension: str) -> float:
        self.get_calls.append((agent_id, dimension))
        return self._state.get((agent_id, dimension), 0.0)

    def record(self, agent_id: str, dimension: str, value: float) -> None:
        self.record_calls.append((agent_id, dimension, value))
        self._state[(agent_id, dimension)] = (
            self._state.get((agent_id, dimension), 0.0) + value
        )


class AsyncSpyStateStore:
    def __init__(self, initial: dict[tuple[str, str], float] | None = None) -> None:
        self._state = dict(initial or {})
        self.get_calls: list[tuple[str, str]] = []
        self.record_calls: list[tuple[str, str, float]] = []

    async def get(self, agent_id: str, dimension: str) -> float:
        self.get_calls.append((agent_id, dimension))
        return self._state.get((agent_id, dimension), 0.0)

    async def record(self, agent_id: str, dimension: str, value: float) -> None:
        self.record_calls.append((agent_id, dimension, value))
        self._state[(agent_id, dimension)] = (
            self._state.get((agent_id, dimension), 0.0) + value
        )


# ---------------------------------------------------------------------------
# DeterministicPolicy construction & validation
# ---------------------------------------------------------------------------


def test_deterministic_policy_accepts_set_operator_in() -> None:
    policy = DeterministicPolicy(
        dimension="allowed_filepaths",
        operator="in",
        values=frozenset({"/tmp", "/workspace"}),
        param="filepath",
        breach_verdict="BLOCK",
    )

    assert policy.operator == "in"
    assert policy.values == frozenset({"/tmp", "/workspace"})
    assert policy.value is None
    assert policy.param == "filepath"


def test_deterministic_policy_accepts_set_operator_not_in() -> None:
    policy = DeterministicPolicy(
        dimension="blocked_recipients",
        operator="not_in",
        values=frozenset({"alice@example.com"}),
        param="recipient",
        breach_verdict="ESCALATE",
    )

    assert policy.operator == "not_in"
    assert policy.values == frozenset({"alice@example.com"})
    assert policy.value is None
    assert policy.param == "recipient"


def test_deterministic_policy_rejects_values_on_numeric_operator() -> None:
    with pytest.raises(ValueError, match="does not accept 'values'"):
        DeterministicPolicy(
            dimension="spend",
            operator="lt",
            value=50.0,
            values=frozenset({"oops"}),
            param="amount",
            breach_verdict="BLOCK",
        )


def test_deterministic_policy_rejects_value_on_set_operator() -> None:
    with pytest.raises(ValueError, match="does not accept 'value'"):
        DeterministicPolicy(
            dimension="allowed_filepaths",
            operator="in",
            value=0.0,
            values=frozenset({"/tmp"}),
            param="filepath",
            breach_verdict="BLOCK",
        )


def test_deterministic_policy_rejects_missing_param_for_set_operator() -> None:
    with pytest.raises(ValueError, match="requires 'param' to be set"):
        DeterministicPolicy(
            dimension="allowed_filepaths",
            operator="in",
            values=frozenset({"/tmp"}),
            breach_verdict="BLOCK",
        )


def test_deterministic_policy_rejects_missing_values_for_set_operator() -> None:
    with pytest.raises(ValueError, match="requires 'values'"):
        DeterministicPolicy(
            dimension="allowed_filepaths",
            operator="in",
            param="filepath",
            breach_verdict="BLOCK",
        )


def test_deterministic_policy_rejects_non_frozenset_values() -> None:
    with pytest.raises(TypeError, match="requires 'values' to be a frozenset"):
        DeterministicPolicy(
            dimension="allowed_filepaths",
            operator="in",
            values=["/tmp", "/workspace"],  # type: ignore[arg-type]
            param="filepath",
            breach_verdict="BLOCK",
        )


def test_deterministic_policy_rejects_non_string_items_in_values() -> None:
    with pytest.raises(ValueError, match="All items in 'values' must be strings"):
        DeterministicPolicy(
            dimension="allowed_filepaths",
            operator="in",
            values=frozenset({"/tmp", 42}),  # type: ignore[arg-type]
            param="filepath",
            breach_verdict="BLOCK",
        )


def test_deterministic_policy_warns_on_empty_values_set() -> None:
    with pytest.warns(LimenexConfigWarning, match="empty 'values' set"):
        DeterministicPolicy(
            dimension="allowed_filepaths",
            operator="in",
            values=frozenset(),
            param="filepath",
            breach_verdict="BLOCK",
        )


def test_deterministic_policy_case_sensitive_exact_matching() -> None:
    policy = DeterministicPolicy(
        dimension="allowed_filepaths",
        operator="in",
        values=frozenset({"/workspace"}),
        param="filepath",
        breach_verdict="BLOCK",
    )

    assert "/workspace" in policy.values
    assert "/Workspace" not in policy.values


def test_deterministic_policy_rejects_non_finite_numeric_value() -> None:
    with pytest.raises(ValueError, match="finite number"):
        DeterministicPolicy(
            dimension="spend",
            operator="lt",
            value=math.inf,
            param="amount",
            breach_verdict="BLOCK",
        )


def test_deterministic_policy_warns_on_fractional_eq_value() -> None:
    with pytest.warns(LimenexConfigWarning, match="exact float comparison"):
        DeterministicPolicy(
            dimension="approval_flag",
            operator="eq",
            value=1.5,
            breach_verdict="BLOCK",
        )


# ---------------------------------------------------------------------------
# Engine: numeric deterministic evaluation
# ---------------------------------------------------------------------------


async def test_engine_allows_when_policy_config_empty() -> None:
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore({"finance.charge": PolicyConfig()}),
        state_store=SpyStateStore(),
    )

    result = await engine.evaluate(
        skill_id="finance.charge",
        agent_id="agent-1",
        kwargs={"amount": 10.0},
    )

    assert result.verdict == "ALLOW"
    assert result.triggered_by is None
    assert result._record_targets == []


async def test_engine_projective_numeric_policy_allows_and_prepares_record_target() -> (
    None
):
    store = SpyStateStore({("agent-1", "4h_spend"): 10.0})
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "finance.charge": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="4h_spend",
                            operator="lt",
                            value=50.0,
                            param="amount",
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=store,
    )

    result = await engine.evaluate(
        skill_id="finance.charge",
        agent_id="agent-1",
        kwargs={"amount": 20.0},
    )

    assert result.verdict == "ALLOW"
    assert result.triggered_by is None
    assert result._record_targets == [("4h_spend", 20.0)]
    assert store.get_calls == [("agent-1", "4h_spend")]


async def test_engine_projective_numeric_policy_blocks_on_breach() -> None:
    store = SpyStateStore({("agent-1", "4h_spend"): 40.0})
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "finance.charge": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="4h_spend",
                            operator="lt",
                            value=50.0,
                            param="amount",
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=store,
    )

    result = await engine.evaluate(
        skill_id="finance.charge",
        agent_id="agent-1",
        kwargs={"amount": 15.0},
    )

    assert result.verdict == "BLOCK"
    assert isinstance(result.triggered_by, DeterministicPolicy)
    assert result._record_targets == []
    assert store.get_calls == [("agent-1", "4h_spend")]


async def test_engine_non_projective_numeric_policy_records_count_increment() -> None:
    store = SpyStateStore({("agent-1", "daily_calls"): 4.0})
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "comms.send": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="daily_calls",
                            operator="lt",
                            value=10.0,
                            breach_verdict="ESCALATE",
                        )
                    ]
                )
            }
        ),
        state_store=store,
    )

    result = await engine.evaluate(
        skill_id="comms.send",
        agent_id="agent-1",
        kwargs={"recipient": "alice@example.com"},
    )

    assert result.verdict == "ALLOW"
    assert result._record_targets == [("daily_calls", 1.0)]


async def test_engine_numeric_policy_missing_param_raises_config_error() -> None:
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "finance.charge": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="4h_spend",
                            operator="lt",
                            value=50.0,
                            param="amount",
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=SpyStateStore(),
    )

    with pytest.raises(LimenexConfigError, match="param='amount'"):
        await engine.evaluate(
            skill_id="finance.charge",
            agent_id="agent-1",
            kwargs={"currency": "USD"},
        )


async def test_engine_numeric_policy_non_numeric_param_raises_config_error() -> None:
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "finance.charge": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="4h_spend",
                            operator="lt",
                            value=50.0,
                            param="amount",
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=SpyStateStore(),
    )

    with pytest.raises(LimenexConfigError, match="could not be cast to float"):
        await engine.evaluate(
            skill_id="finance.charge",
            agent_id="agent-1",
            kwargs={"amount": "not-a-number"},
        )


async def test_engine_record_persists_numeric_targets() -> None:
    store = SpyStateStore()
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore({}),
        state_store=store,
    )

    result = EvaluationResult(
        verdict="ALLOW",
        skill_id="finance.charge",
        agent_id="agent-1",
        triggered_by=None,
        _record_targets=[("4h_spend", 12.5), ("daily_calls", 1.0)],
    )

    await engine.record(result)

    assert store.record_calls == [
        ("agent-1", "4h_spend", 12.5),
        ("agent-1", "daily_calls", 1.0),
    ]


# ---------------------------------------------------------------------------
# Engine: set-operator evaluation
# ---------------------------------------------------------------------------


async def test_engine_set_operator_in_allows_when_value_is_in_set() -> None:
    store = SpyStateStore()
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "filesystem.delete": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="allowed_filepaths",
                            operator="in",
                            values=frozenset({"/tmp", "/workspace"}),
                            param="filepath",
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=store,
    )

    result = await engine.evaluate(
        skill_id="filesystem.delete",
        agent_id="agent-1",
        kwargs={"filepath": "/tmp"},
    )

    assert result.verdict == "ALLOW"
    assert result.triggered_by is None
    assert result._record_targets == []
    assert store.get_calls == []
    await engine.record(result)
    assert store.record_calls == []


async def test_engine_set_operator_in_blocks_when_value_not_in_set() -> None:
    store = SpyStateStore()
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "filesystem.delete": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="allowed_filepaths",
                            operator="in",
                            values=frozenset({"/tmp", "/workspace"}),
                            param="filepath",
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=store,
    )

    result = await engine.evaluate(
        skill_id="filesystem.delete",
        agent_id="agent-1",
        kwargs={"filepath": "/etc/passwd"},
    )

    assert result.verdict == "BLOCK"
    assert isinstance(result.triggered_by, DeterministicPolicy)
    assert result._record_targets == []
    assert store.get_calls == []


async def test_engine_set_operator_in_escalates_when_value_not_in_set() -> None:
    store = SpyStateStore()
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "filesystem.delete": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="allowed_filepaths",
                            operator="in",
                            values=frozenset({"/tmp", "/workspace"}),
                            param="filepath",
                            breach_verdict="ESCALATE",
                        )
                    ]
                )
            }
        ),
        state_store=store,
    )

    result = await engine.evaluate(
        skill_id="filesystem.delete",
        agent_id="agent-1",
        kwargs={"filepath": "/etc/passwd"},
    )

    assert result.verdict == "ESCALATE"
    assert isinstance(result.triggered_by, DeterministicPolicy)
    assert result._record_targets == []
    assert store.get_calls == []


async def test_engine_set_operator_not_in_allows_when_value_not_in_excluded_set() -> (
    None
):
    store = SpyStateStore()
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "comms.send": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="blocked_recipients",
                            operator="not_in",
                            values=frozenset({"alice@blocked.com"}),
                            param="recipient",
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=store,
    )

    result = await engine.evaluate(
        skill_id="comms.send",
        agent_id="agent-1",
        kwargs={"recipient": "bob@example.com"},
    )

    assert result.verdict == "ALLOW"
    assert result.triggered_by is None
    assert result._record_targets == []
    assert store.get_calls == []


async def test_engine_set_operator_not_in_blocks_when_value_is_in_excluded_set() -> (
    None
):
    store = SpyStateStore()
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "comms.send": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="blocked_recipients",
                            operator="not_in",
                            values=frozenset({"alice@blocked.com"}),
                            param="recipient",
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=store,
    )

    result = await engine.evaluate(
        skill_id="comms.send",
        agent_id="agent-1",
        kwargs={"recipient": "alice@blocked.com"},
    )

    assert result.verdict == "BLOCK"
    assert isinstance(result.triggered_by, DeterministicPolicy)
    assert result._record_targets == []
    assert store.get_calls == []


async def test_engine_set_operator_not_in_escalates_when_value_is_in_excluded_set() -> (
    None
):
    store = SpyStateStore()
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "comms.send": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="blocked_recipients",
                            operator="not_in",
                            values=frozenset({"alice@blocked.com"}),
                            param="recipient",
                            breach_verdict="ESCALATE",
                        )
                    ]
                )
            }
        ),
        state_store=store,
    )

    result = await engine.evaluate(
        skill_id="comms.send",
        agent_id="agent-1",
        kwargs={"recipient": "alice@blocked.com"},
    )

    assert result.verdict == "ESCALATE"
    assert isinstance(result.triggered_by, DeterministicPolicy)
    assert result._record_targets == []
    assert store.get_calls == []


async def test_engine_set_operator_missing_param_raises_config_error() -> None:
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "web.post": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="approved_urls",
                            operator="in",
                            values=frozenset({"https://api.example.com"}),
                            param="url",
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=SpyStateStore(),
    )

    with pytest.raises(LimenexConfigError, match="uses a set operator"):
        await engine.evaluate(
            skill_id="web.post",
            agent_id="agent-1",
            kwargs={"payload": {"x": 1}},
        )


async def test_engine_set_operator_non_string_param_raises_config_error() -> None:
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "filesystem.delete": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="allowed_filepaths",
                            operator="in",
                            values=frozenset({"/tmp"}),
                            param="filepath",
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=SpyStateStore(),
    )

    with pytest.raises(LimenexConfigError, match="is not a string"):
        await engine.evaluate(
            skill_id="filesystem.delete",
            agent_id="agent-1",
            kwargs={"filepath": 42},
        )


async def test_engine_empty_values_in_policy_behaves_deterministically_for_in_operator() -> (
    None
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LimenexConfigWarning)
        policy = DeterministicPolicy(
            dimension="allowed_filepaths",
            operator="in",
            values=frozenset(),
            param="filepath",
            breach_verdict="BLOCK",
        )
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {"filesystem.delete": PolicyConfig(policies=[policy])}
        ),
        state_store=SpyStateStore(),
    )

    result = await engine.evaluate(
        skill_id="filesystem.delete",
        agent_id="agent-1",
        kwargs={"filepath": "/tmp"},
    )

    assert result.verdict == "BLOCK"


async def test_engine_empty_values_in_policy_behaves_deterministically_for_not_in_operator() -> (
    None
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LimenexConfigWarning)
        policy = DeterministicPolicy(
            dimension="blocked_recipients",
            operator="not_in",
            values=frozenset(),
            param="recipient",
            breach_verdict="BLOCK",
        )
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {"comms.send": PolicyConfig(policies=[policy])}
        ),
        state_store=SpyStateStore(),
    )

    result = await engine.evaluate(
        skill_id="comms.send",
        agent_id="agent-1",
        kwargs={"recipient": "alice@example.com"},
    )

    assert result.verdict == "ALLOW"


# ---------------------------------------------------------------------------
# Engine: semantic policies
# ---------------------------------------------------------------------------


async def test_engine_semantic_policy_requires_llm_evaluator() -> None:
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "web.post": PolicyConfig(
                    policies=[
                        SemanticPolicy(
                            rule="Do not send secrets externally.",
                            verdict_ceiling="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=SpyStateStore(),
        llm_evaluator=None,
    )

    with pytest.raises(LimenexConfigError, match="requires an llm_evaluator"):
        await engine.evaluate(
            skill_id="web.post",
            agent_id="agent-1",
            kwargs={"url": "https://example.com"},
        )


async def test_engine_semantic_policy_allows_when_evaluator_returns_allow() -> None:
    def evaluator(action_intent: str, rule: str) -> str:
        assert "web.post" in action_intent
        assert "Do not send secrets externally." == rule
        return "ALLOW"

    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "web.post": PolicyConfig(
                    policies=[
                        SemanticPolicy(
                            rule="Do not send secrets externally.",
                            verdict_ceiling="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=SpyStateStore(),
        llm_evaluator=evaluator,
    )

    result = await engine.evaluate(
        skill_id="web.post",
        agent_id="agent-1",
        kwargs={"url": "https://example.com"},
    )

    assert result.verdict == "ALLOW"


async def test_engine_semantic_policy_applies_verdict_ceiling_to_more_severe_response() -> (
    None
):
    def evaluator(action_intent: str, rule: str) -> str:
        return "BLOCK"

    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "web.post": PolicyConfig(
                    policies=[
                        SemanticPolicy(
                            rule="Escalate suspicious outbound requests.",
                            verdict_ceiling="ESCALATE",
                        )
                    ]
                )
            }
        ),
        state_store=SpyStateStore(),
        llm_evaluator=evaluator,
    )

    result = await engine.evaluate(
        skill_id="web.post",
        agent_id="agent-1",
        kwargs={"url": "https://example.com"},
    )

    assert result.verdict == "ESCALATE"


async def test_engine_semantic_policy_falls_back_to_verdict_ceiling_on_invalid_response() -> (
    None
):
    def evaluator(action_intent: str, rule: str) -> Any:
        return "SOMETHING_INVALID"

    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "web.post": PolicyConfig(
                    policies=[
                        SemanticPolicy(
                            rule="Escalate suspicious outbound requests.",
                            verdict_ceiling="ESCALATE",
                        )
                    ]
                )
            }
        ),
        state_store=SpyStateStore(),
        llm_evaluator=evaluator,
    )

    result = await engine.evaluate(
        skill_id="web.post",
        agent_id="agent-1",
        kwargs={"url": "https://example.com"},
    )

    assert result.verdict == "ESCALATE"


# ---------------------------------------------------------------------------
# Engine: ordering, short-circuiting, and async stores
# ---------------------------------------------------------------------------


async def test_engine_short_circuits_on_first_non_allow_policy() -> None:
    store = SpyStateStore({("agent-1", "4h_spend"): 60.0})

    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "finance.charge": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="4h_spend",
                            operator="lt",
                            value=50.0,
                            param="amount",
                            breach_verdict="BLOCK",
                        ),
                        SemanticPolicy(
                            rule="This should never run.",
                            verdict_ceiling="ESCALATE",
                        ),
                    ]
                )
            }
        ),
        state_store=store,
        llm_evaluator=lambda action_intent, rule: "ALLOW",
    )

    result = await engine.evaluate(
        skill_id="finance.charge",
        agent_id="agent-1",
        kwargs={"amount": 1.0},
    )

    assert result.verdict == "BLOCK"
    assert isinstance(result.triggered_by, DeterministicPolicy)


async def test_engine_supports_async_policy_store_and_async_state_store() -> None:
    store = AsyncSpyStateStore({("agent-1", "4h_spend"): 5.0})
    engine = PolicyEngine(
        policy_store=AsyncInMemoryPolicyStore(
            {
                "finance.charge": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="4h_spend",
                            operator="lt",
                            value=50.0,
                            param="amount",
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=store,
    )

    result = await engine.evaluate(
        skill_id="finance.charge",
        agent_id="agent-1",
        kwargs={"amount": 10.0},
    )

    assert result.verdict == "ALLOW"
    assert result._record_targets == [("4h_spend", 10.0)]

    await engine.record(result)

    assert store.get_calls == [("agent-1", "4h_spend")]
    assert store.record_calls == [("agent-1", "4h_spend", 10.0)]


# ---------------------------------------------------------------------------
# EvaluationResult invariants
# ---------------------------------------------------------------------------


def test_evaluation_result_requires_triggered_by_when_not_allow() -> None:
    with pytest.raises(ValueError, match="triggered_by must be set"):
        EvaluationResult(
            verdict="BLOCK",
            skill_id="finance.charge",
            agent_id="agent-1",
            triggered_by=None,
        )


def test_evaluation_result_requires_triggered_by_none_when_allow() -> None:
    policy = DeterministicPolicy(
        dimension="4h_spend",
        operator="lt",
        value=50.0,
        param="amount",
        breach_verdict="BLOCK",
    )

    with pytest.raises(ValueError, match="triggered_by must be None"):
        EvaluationResult(
            verdict="ALLOW",
            skill_id="finance.charge",
            agent_id="agent-1",
            triggered_by=policy,
        )


def test_evaluation_result_disallows_record_targets_on_non_allow() -> None:
    policy = DeterministicPolicy(
        dimension="4h_spend",
        operator="lt",
        value=50.0,
        param="amount",
        breach_verdict="BLOCK",
    )

    with pytest.raises(ValueError, match="_record_targets must be empty"):
        EvaluationResult(
            verdict="ESCALATE",
            skill_id="finance.charge",
            agent_id="agent-1",
            triggered_by=policy,
            _record_targets=[("4h_spend", 10.0)],
        )
