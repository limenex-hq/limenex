"""
tests/test_skills.py — Unit tests for the Limenex skills library.

Covers per-skill:
- ALLOW: executor called with correct named kwargs, returns result (executor-injected)
- ALLOW: filesystem operation performed (filesystem skills)
- BLOCK: BlockedError raised, executor/operation never invoked
- ESCALATE: EscalationRequired raised, executor/operation never invoked
- Executor strip validation: executor absent from kwargs seen by engine (executor-injected)
- Sync and async executor dispatch (executor-injected)

All executors are Mock or AsyncMock — no live external API calls.
Filesystem tests use tmp_path — no real filesystem writes outside temp dir.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from limenex.core.engine import BlockedError, EscalationRequired, PolicyEngine
from limenex.core.policy import (
    DeterministicPolicy,
    PolicyConfig,
    UnregisteredSkillError,
)
from limenex.skills.comms import SEND_SKILL_ID, make_send
from limenex.skills.filesystem import (
    DELETE_SKILL_ID,
    MOVE_SKILL_ID,
    WRITE_SKILL_ID,
    make_delete,
    make_move,
    make_write,
)
from limenex.skills.finance import (
    CHARGE_SKILL_ID,
    SPEND_SKILL_ID,
    make_charge,
    make_spend,
)
from limenex.skills.web import POST_SKILL_ID, make_post

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakePolicyStore:
    def __init__(self, configs: dict[str, PolicyConfig]) -> None:
        self._configs = configs

    def get(self, skill_id: str) -> PolicyConfig:
        if skill_id not in self._configs:
            raise UnregisteredSkillError(skill_id)
        return self._configs[skill_id]


class FakeStateStore:
    def __init__(self) -> None:
        self._state: dict[str, dict[str, float]] = {}
        self.recorded: list[tuple[str, str, float]] = []

    def get(self, agent_id: str, dimension: str) -> float:
        return self._state.get(dimension, {}).get(agent_id, 0.0)

    def record(self, agent_id: str, dimension: str, value: float) -> None:
        self.recorded.append((agent_id, dimension, value))


class CapturingPolicyEngine(PolicyEngine):
    """PolicyEngine subclass that captures kwargs passed to evaluate().

    Used to verify executor is never present in the kwargs the engine sees.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.captured_evaluate_kwargs: list[dict] = []

    async def evaluate(self, skill_id, agent_id, kwargs):
        self.captured_evaluate_kwargs.append(dict(kwargs))
        return await super().evaluate(skill_id, agent_id, kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def allow_policy() -> DeterministicPolicy:
    return DeterministicPolicy(
        dimension="count", operator="lte", value=100.0, breach_verdict="BLOCK"
    )


def block_policy() -> DeterministicPolicy:
    return DeterministicPolicy(
        dimension="count", operator="lte", value=-1.0, breach_verdict="BLOCK"
    )


def escalate_policy() -> DeterministicPolicy:
    return DeterministicPolicy(
        dimension="count", operator="lte", value=-1.0, breach_verdict="ESCALATE"
    )


def make_engine(skill_id: str, policies: list) -> tuple[PolicyEngine, FakeStateStore]:
    store = FakeStateStore()
    engine = PolicyEngine(
        policy_store=FakePolicyStore({skill_id: PolicyConfig(policies=policies)}),
        state_store=store,
    )
    return engine, store


def make_capturing_engine(
    skill_id: str, policies: list
) -> tuple[CapturingPolicyEngine, FakeStateStore]:
    store = FakeStateStore()
    engine = CapturingPolicyEngine(
        policy_store=FakePolicyStore({skill_id: PolicyConfig(policies=policies)}),
        state_store=store,
    )
    return engine, store


# ---------------------------------------------------------------------------
# Finance — charge
# ---------------------------------------------------------------------------


async def test_charge_allow_sync_executor():
    engine, _ = make_engine(CHARGE_SKILL_ID, [allow_policy()])
    executor = Mock(return_value="receipt_001")
    charge = make_charge(engine)

    result = await charge(
        agent_id="agent_1", amount=50.0, currency="USD", executor=executor
    )

    executor.assert_called_once_with(amount=50.0, currency="USD")
    assert result == "receipt_001"


async def test_charge_allow_async_executor():
    engine, _ = make_engine(CHARGE_SKILL_ID, [allow_policy()])
    executor = AsyncMock(return_value="receipt_002")
    charge = make_charge(engine)

    result = await charge(
        agent_id="agent_1", amount=30.0, currency="GBP", executor=executor
    )

    executor.assert_called_once_with(amount=30.0, currency="GBP")
    assert result == "receipt_002"


async def test_charge_block_executor_not_called():
    engine, _ = make_engine(CHARGE_SKILL_ID, [block_policy()])
    executor = Mock()
    charge = make_charge(engine)

    with pytest.raises(BlockedError):
        await charge(agent_id="agent_1", amount=50.0, currency="USD", executor=executor)

    executor.assert_not_called()


async def test_charge_escalate_executor_not_called():
    engine, _ = make_engine(CHARGE_SKILL_ID, [escalate_policy()])
    executor = Mock()
    charge = make_charge(engine)

    with pytest.raises(EscalationRequired):
        await charge(agent_id="agent_1", amount=50.0, currency="USD", executor=executor)

    executor.assert_not_called()


async def test_charge_executor_not_in_engine_kwargs():
    engine, _ = make_capturing_engine(CHARGE_SKILL_ID, [allow_policy()])
    charge = make_charge(engine)

    await charge(
        agent_id="agent_1",
        amount=10.0,
        currency="USD",
        executor=Mock(return_value="ok"),
    )

    assert all("executor" not in kw for kw in engine.captured_evaluate_kwargs)


# ---------------------------------------------------------------------------
# Finance — spend
# ---------------------------------------------------------------------------


async def test_spend_allow_sync_executor():
    engine, _ = make_engine(SPEND_SKILL_ID, [allow_policy()])
    executor = Mock(return_value="confirmed")
    spend = make_spend(engine)

    result = await spend(
        agent_id="agent_1", service="openai", amount_usd=20.0, executor=executor
    )

    executor.assert_called_once_with(service="openai", amount_usd=20.0)
    assert result == "confirmed"


async def test_spend_allow_async_executor():
    engine, _ = make_engine(SPEND_SKILL_ID, [allow_policy()])
    executor = AsyncMock(return_value="confirmed_async")
    spend = make_spend(engine)

    result = await spend(
        agent_id="agent_1", service="aws", amount_usd=100.0, executor=executor
    )

    executor.assert_called_once_with(service="aws", amount_usd=100.0)
    assert result == "confirmed_async"


async def test_spend_block_executor_not_called():
    engine, _ = make_engine(SPEND_SKILL_ID, [block_policy()])
    executor = Mock()
    spend = make_spend(engine)

    with pytest.raises(BlockedError):
        await spend(
            agent_id="agent_1", service="openai", amount_usd=20.0, executor=executor
        )

    executor.assert_not_called()


async def test_spend_escalate_executor_not_called():
    engine, _ = make_engine(SPEND_SKILL_ID, [escalate_policy()])
    executor = Mock()
    spend = make_spend(engine)

    with pytest.raises(EscalationRequired):
        await spend(
            agent_id="agent_1", service="openai", amount_usd=20.0, executor=executor
        )

    executor.assert_not_called()


async def test_spend_executor_not_in_engine_kwargs():
    engine, _ = make_capturing_engine(SPEND_SKILL_ID, [allow_policy()])
    spend = make_spend(engine)

    await spend(
        agent_id="agent_1",
        service="openai",
        amount_usd=10.0,
        executor=Mock(return_value="ok"),
    )

    assert all("executor" not in kw for kw in engine.captured_evaluate_kwargs)


# ---------------------------------------------------------------------------
# Filesystem — delete
# ---------------------------------------------------------------------------


def test_delete_allow(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("content")
    engine, _ = make_engine(DELETE_SKILL_ID, [allow_policy()])
    delete = make_delete(engine)

    delete(agent_id="agent_1", filepath=str(target))

    assert not target.exists()


def test_delete_block(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("content")
    engine, _ = make_engine(DELETE_SKILL_ID, [block_policy()])
    delete = make_delete(engine)

    with pytest.raises(BlockedError):
        delete(agent_id="agent_1", filepath=str(target))

    assert target.exists()


def test_delete_escalate(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("content")
    engine, _ = make_engine(DELETE_SKILL_ID, [escalate_policy()])
    delete = make_delete(engine)

    with pytest.raises(EscalationRequired):
        delete(agent_id="agent_1", filepath=str(target))

    assert target.exists()


# ---------------------------------------------------------------------------
# Filesystem — write
# ---------------------------------------------------------------------------


def test_write_allow(tmp_path):
    target = tmp_path / "output.txt"
    engine, _ = make_engine(WRITE_SKILL_ID, [allow_policy()])
    write = make_write(engine)

    write(agent_id="agent_1", filepath=str(target), content="hello limenex")

    assert target.read_text(encoding="utf-8") == "hello limenex"


def test_write_block(tmp_path):
    target = tmp_path / "output.txt"
    engine, _ = make_engine(WRITE_SKILL_ID, [block_policy()])
    write = make_write(engine)

    with pytest.raises(BlockedError):
        write(agent_id="agent_1", filepath=str(target), content="should not appear")

    assert not target.exists()


def test_write_escalate(tmp_path):
    target = tmp_path / "output.txt"
    engine, _ = make_engine(WRITE_SKILL_ID, [escalate_policy()])
    write = make_write(engine)

    with pytest.raises(EscalationRequired):
        write(agent_id="agent_1", filepath=str(target), content="should not appear")

    assert not target.exists()


# ---------------------------------------------------------------------------
# Filesystem — move
# ---------------------------------------------------------------------------


def test_move_allow(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("moveme")
    engine, _ = make_engine(MOVE_SKILL_ID, [allow_policy()])
    move = make_move(engine)

    move(agent_id="agent_1", src=str(src), dst=str(dst))

    assert not src.exists()
    assert dst.read_text(encoding="utf-8") == "moveme"


def test_move_block(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("moveme")
    engine, _ = make_engine(MOVE_SKILL_ID, [block_policy()])
    move = make_move(engine)

    with pytest.raises(BlockedError):
        move(agent_id="agent_1", src=str(src), dst=str(dst))

    assert src.exists()
    assert not dst.exists()


def test_move_escalate(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("moveme")
    engine, _ = make_engine(MOVE_SKILL_ID, [escalate_policy()])
    move = make_move(engine)

    with pytest.raises(EscalationRequired):
        move(agent_id="agent_1", src=str(src), dst=str(dst))

    assert src.exists()
    assert not dst.exists()


# ---------------------------------------------------------------------------
# Comms — send
# ---------------------------------------------------------------------------


async def test_send_allow_sync_executor():
    engine, _ = make_engine(SEND_SKILL_ID, [allow_policy()])
    executor = Mock(return_value="msg_001")
    send = make_send(engine)

    result = await send(
        agent_id="agent_1",
        channel="slack",
        recipient="U12345",
        text="hello",
        executor=executor,
    )

    executor.assert_called_once_with(channel="slack", recipient="U12345", text="hello")
    assert result == "msg_001"


async def test_send_allow_async_executor():
    engine, _ = make_engine(SEND_SKILL_ID, [allow_policy()])
    executor = AsyncMock(return_value="msg_002")
    send = make_send(engine)

    result = await send(
        agent_id="agent_1",
        channel="email",
        recipient="user@example.com",
        text="hi",
        executor=executor,
    )

    executor.assert_called_once_with(
        channel="email", recipient="user@example.com", text="hi"
    )
    assert result == "msg_002"


async def test_send_block_executor_not_called():
    engine, _ = make_engine(SEND_SKILL_ID, [block_policy()])
    executor = Mock()
    send = make_send(engine)

    with pytest.raises(BlockedError):
        await send(
            agent_id="agent_1",
            channel="slack",
            recipient="U12345",
            text="hello",
            executor=executor,
        )

    executor.assert_not_called()


async def test_send_escalate_executor_not_called():
    engine, _ = make_engine(SEND_SKILL_ID, [escalate_policy()])
    executor = Mock()
    send = make_send(engine)

    with pytest.raises(EscalationRequired):
        await send(
            agent_id="agent_1",
            channel="slack",
            recipient="U12345",
            text="hello",
            executor=executor,
        )

    executor.assert_not_called()


async def test_send_executor_not_in_engine_kwargs():
    engine, _ = make_capturing_engine(SEND_SKILL_ID, [allow_policy()])
    send = make_send(engine)

    await send(
        agent_id="agent_1",
        channel="slack",
        recipient="U12345",
        text="hello",
        executor=Mock(return_value="ok"),
    )

    assert all("executor" not in kw for kw in engine.captured_evaluate_kwargs)


# ---------------------------------------------------------------------------
# Web — post
# ---------------------------------------------------------------------------


async def test_post_allow_sync_executor():
    engine, _ = make_engine(POST_SKILL_ID, [allow_policy()])
    executor = Mock(return_value={"status": 200})
    post = make_post(engine)

    result = await post(
        agent_id="agent_1",
        url="https://api.example.com/data",
        payload={"key": "value"},
        executor=executor,
    )

    executor.assert_called_once_with(
        url="https://api.example.com/data", payload={"key": "value"}
    )
    assert result == {"status": 200}


async def test_post_allow_async_executor():
    engine, _ = make_engine(POST_SKILL_ID, [allow_policy()])
    executor = AsyncMock(return_value={"status": 201})
    post = make_post(engine)

    result = await post(
        agent_id="agent_1",
        url="https://api.example.com/items",
        payload={"name": "test"},
        executor=executor,
    )

    executor.assert_called_once_with(
        url="https://api.example.com/items", payload={"name": "test"}
    )
    assert result == {"status": 201}


async def test_post_block_executor_not_called():
    engine, _ = make_engine(POST_SKILL_ID, [block_policy()])
    executor = Mock()
    post = make_post(engine)

    with pytest.raises(BlockedError):
        await post(
            agent_id="agent_1",
            url="https://api.example.com/data",
            payload={},
            executor=executor,
        )

    executor.assert_not_called()


async def test_post_escalate_executor_not_called():
    engine, _ = make_engine(POST_SKILL_ID, [escalate_policy()])
    executor = Mock()
    post = make_post(engine)

    with pytest.raises(EscalationRequired):
        await post(
            agent_id="agent_1",
            url="https://api.example.com/data",
            payload={},
            executor=executor,
        )

    executor.assert_not_called()


async def test_post_executor_not_in_engine_kwargs():
    engine, _ = make_capturing_engine(POST_SKILL_ID, [allow_policy()])
    post = make_post(engine)

    await post(
        agent_id="agent_1",
        url="https://api.example.com/data",
        payload={"x": 1},
        executor=Mock(return_value={"status": 200}),
    )

    assert all("executor" not in kw for kw in engine.captured_evaluate_kwargs)
