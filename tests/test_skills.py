from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from limenex.core.engine import BlockedError, EscalationRequired, PolicyEngine
from limenex.core.policy import (
    DeterministicPolicy,
    PolicyConfig,
    UnregisteredSkillError,
)
from limenex.skills import UnregisteredExecutorError
from limenex.skills.comms import make_send
from limenex.skills.filesystem import make_delete, make_move, make_write
from limenex.skills.finance import make_charge, make_spend
from limenex.skills.web import make_post


class InMemoryPolicyStore:
    def __init__(self, configs: dict[str, PolicyConfig]) -> None:
        self.configs = configs

    def get(self, skill_id: str) -> PolicyConfig:
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


class CapturingPolicyEngine(PolicyEngine):
    """PolicyEngine subclass that captures kwargs seen by evaluate().

    Used to assert that executor is never present in the kwargs the engine
    receives — regardless of whether it was a call-time parameter or not.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.captured_kwargs: list[dict] = []

    async def evaluate(self, skill_id: str, agent_id: str, kwargs: dict) -> object:
        self.captured_kwargs.append(dict(kwargs))
        return await super().evaluate(skill_id, agent_id, kwargs)


def make_engine(configs: dict[str, PolicyConfig]) -> PolicyEngine:
    return PolicyEngine(
        policy_store=InMemoryPolicyStore(configs),
        state_store=SpyStateStore(),
    )


def make_capturing_engine(configs: dict[str, PolicyConfig]) -> CapturingPolicyEngine:
    return CapturingPolicyEngine(
        policy_store=InMemoryPolicyStore(configs),
        state_store=SpyStateStore(),
    )


# ---------------------------------------------------------------------------
# Finance skills
# ---------------------------------------------------------------------------


async def test_charge_allow_calls_sync_executor_and_returns_value() -> None:
    engine = make_engine(
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
    )
    executor = Mock(return_value="charge-ok")
    charge = make_charge(engine, registry={"stripe": executor})

    result = await charge(
        agent_id="agent-1",
        provider="stripe",
        amount=10.0,
        currency="USD",
    )

    assert result == "charge-ok"
    executor.assert_called_once_with(amount=10.0, currency="USD")


async def test_charge_allow_calls_async_executor_and_returns_value() -> None:
    engine = make_engine(
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
    )
    executor = AsyncMock(return_value="charge-ok-async")
    charge = make_charge(engine, registry={"stripe": executor})

    result = await charge(
        agent_id="agent-1",
        provider="stripe",
        amount=10.0,
        currency="USD",
    )

    assert result == "charge-ok-async"
    executor.assert_awaited_once_with(amount=10.0, currency="USD")


async def test_charge_block_does_not_call_executor() -> None:
    engine = make_engine(
        {
            "finance.charge": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="4h_spend",
                        operator="lt",
                        value=5.0,
                        param="amount",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    executor = Mock()
    charge = make_charge(engine, registry={"stripe": executor})

    with pytest.raises(BlockedError):
        await charge(
            agent_id="agent-1",
            provider="stripe",
            amount=10.0,
            currency="USD",
        )

    executor.assert_not_called()


async def test_charge_escalate_does_not_call_executor() -> None:
    engine = make_engine(
        {
            "finance.charge": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="4h_spend",
                        operator="lt",
                        value=5.0,
                        param="amount",
                        breach_verdict="ESCALATE",
                    )
                ]
            )
        }
    )
    executor = Mock()
    charge = make_charge(engine, registry={"stripe": executor})

    with pytest.raises(EscalationRequired):
        await charge(
            agent_id="agent-1",
            provider="stripe",
            amount=10.0,
            currency="USD",
        )

    executor.assert_not_called()


async def test_charge_unregistered_provider_raises_before_governance() -> None:
    state_store = SpyStateStore()
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
        state_store=state_store,
    )
    charge = make_charge(engine, registry={"stripe": Mock()})

    with pytest.raises(UnregisteredExecutorError) as exc_info:
        await charge(
            agent_id="agent-1",
            provider="square",
            amount=10.0,
            currency="USD",
        )

    assert exc_info.value.skill_id == "finance.charge"
    assert exc_info.value.key == "square"
    assert state_store.record_calls == []


async def test_charge_executor_not_in_engine_kwargs() -> None:
    engine = make_capturing_engine(
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
    )
    executor = Mock(return_value="ok")
    charge = make_charge(engine, registry={"stripe": executor})

    await charge(agent_id="agent-1", provider="stripe", amount=10.0, currency="USD")

    assert len(engine.captured_kwargs) == 1
    assert "executor" not in engine.captured_kwargs[0]


async def test_spend_allow_calls_sync_executor_and_returns_value() -> None:
    engine = make_engine(
        {
            "finance.spend": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="daily_spend_usd",
                        operator="lt",
                        value=100.0,
                        param="amount_usd",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    executor = Mock(return_value={"ok": True})
    spend = make_spend(engine, registry={"aws": executor})

    result = await spend(
        agent_id="agent-1",
        service="aws",
        amount_usd=25.0,
    )

    assert result == {"ok": True}
    executor.assert_called_once_with(amount_usd=25.0)


async def test_spend_allow_calls_async_executor_and_returns_value() -> None:
    engine = make_engine(
        {
            "finance.spend": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="daily_spend_usd",
                        operator="lt",
                        value=100.0,
                        param="amount_usd",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    executor = AsyncMock(return_value={"ok": True})
    spend = make_spend(engine, registry={"aws": executor})

    result = await spend(
        agent_id="agent-1",
        service="aws",
        amount_usd=25.0,
    )

    assert result == {"ok": True}
    executor.assert_awaited_once_with(amount_usd=25.0)


async def test_spend_block_does_not_call_executor() -> None:
    engine = make_engine(
        {
            "finance.spend": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="daily_spend_usd",
                        operator="lt",
                        value=10.0,
                        param="amount_usd",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    executor = AsyncMock()
    spend = make_spend(engine, registry={"aws": executor})

    with pytest.raises(BlockedError):
        await spend(
            agent_id="agent-1",
            service="aws",
            amount_usd=25.0,
        )

    executor.assert_not_called()


async def test_spend_escalate_does_not_call_executor() -> None:
    engine = make_engine(
        {
            "finance.spend": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="daily_spend_usd",
                        operator="lt",
                        value=10.0,
                        param="amount_usd",
                        breach_verdict="ESCALATE",
                    )
                ]
            )
        }
    )
    executor = AsyncMock()
    spend = make_spend(engine, registry={"openai": executor})

    with pytest.raises(EscalationRequired):
        await spend(
            agent_id="agent-1",
            service="openai",
            amount_usd=25.0,
        )

    executor.assert_not_called()


async def test_spend_unregistered_service_raises_before_governance() -> None:
    state_store = SpyStateStore()
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "finance.spend": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="daily_spend_usd",
                            operator="lt",
                            value=100.0,
                            param="amount_usd",
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=state_store,
    )
    spend = make_spend(engine, registry={"aws": AsyncMock()})

    with pytest.raises(UnregisteredExecutorError) as exc_info:
        await spend(
            agent_id="agent-1",
            service="gcp",
            amount_usd=25.0,
        )

    assert exc_info.value.skill_id == "finance.spend"
    assert exc_info.value.key == "gcp"
    assert state_store.record_calls == []


async def test_spend_executor_not_in_engine_kwargs() -> None:
    engine = make_capturing_engine(
        {
            "finance.spend": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="daily_spend_usd",
                        operator="lt",
                        value=100.0,
                        param="amount_usd",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    executor = AsyncMock(return_value={"ok": True})
    spend = make_spend(engine, registry={"aws": executor})

    await spend(agent_id="agent-1", service="aws", amount_usd=25.0)

    assert len(engine.captured_kwargs) == 1
    assert "executor" not in engine.captured_kwargs[0]


# ---------------------------------------------------------------------------
# Filesystem skills
# ---------------------------------------------------------------------------


def test_filesystem_write_allowlist_allows_exact_filepath(tmp_path) -> None:
    allowed_path = tmp_path / "allowed.txt"
    blocked_path = tmp_path / "blocked.txt"

    engine = make_engine(
        {
            "filesystem.write": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="allowed_filepaths",
                        operator="in",
                        values=frozenset({str(allowed_path)}),
                        param="filepath",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    write = make_write(engine)

    write(agent_id="agent-1", filepath=str(allowed_path), content="hello")

    assert allowed_path.read_text(encoding="utf-8") == "hello"
    assert not blocked_path.exists()


def test_filesystem_write_allowlist_blocks_non_member_filepath(tmp_path) -> None:
    allowed_path = tmp_path / "allowed.txt"
    blocked_path = tmp_path / "blocked.txt"

    engine = make_engine(
        {
            "filesystem.write": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="allowed_filepaths",
                        operator="in",
                        values=frozenset({str(allowed_path)}),
                        param="filepath",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    write = make_write(engine)

    with pytest.raises(BlockedError):
        write(agent_id="agent-1", filepath=str(blocked_path), content="nope")

    assert not blocked_path.exists()


def test_filesystem_delete_allowlist_blocks_non_member_filepath(tmp_path) -> None:
    allowed_path = tmp_path / "allowed.txt"
    blocked_path = tmp_path / "blocked.txt"
    blocked_path.write_text("secret", encoding="utf-8")

    engine = make_engine(
        {
            "filesystem.delete": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="allowed_filepaths",
                        operator="in",
                        values=frozenset({str(allowed_path)}),
                        param="filepath",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    delete = make_delete(engine)

    with pytest.raises(BlockedError):
        delete(agent_id="agent-1", filepath=str(blocked_path))

    assert blocked_path.exists()
    assert blocked_path.read_text(encoding="utf-8") == "secret"


def test_filesystem_move_allowlist_allows_exact_src_path(tmp_path) -> None:
    src = tmp_path / "from.txt"
    dst = tmp_path / "to.txt"
    src.write_text("payload", encoding="utf-8")

    engine = make_engine(
        {
            "filesystem.move": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="allowed_move_sources",
                        operator="in",
                        values=frozenset({str(src)}),
                        param="src",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    move = make_move(engine)

    move(agent_id="agent-1", src=str(src), dst=str(dst))

    assert not src.exists()
    assert dst.read_text(encoding="utf-8") == "payload"


# ---------------------------------------------------------------------------
# Comms skills
# ---------------------------------------------------------------------------


async def test_comms_send_allow_calls_sync_executor() -> None:
    engine = make_engine(
        {
            "comms.send": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="daily_messages",
                        operator="lt",
                        value=10.0,
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    executor = Mock(return_value="sent")
    send = make_send(engine, registry={"email": executor})

    result = await send(
        agent_id="agent-1",
        channel="email",
        recipient="alice@example.com",
        text="hello",
    )

    assert result == "sent"
    executor.assert_called_once_with(
        recipient="alice@example.com",
        text="hello",
    )


async def test_comms_send_recipient_blocklist_blocks_exact_match_and_never_calls_executor() -> (
    None
):
    engine = make_engine(
        {
            "comms.send": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="blocked_recipients",
                        operator="not_in",
                        values=frozenset({"alice@baddomain.com"}),
                        param="recipient",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    executor = AsyncMock()
    send = make_send(engine, registry={"email": executor})

    with pytest.raises(BlockedError):
        await send(
            agent_id="agent-1",
            channel="email",
            recipient="alice@baddomain.com",
            text="should not send",
        )

    executor.assert_not_called()


async def test_comms_send_recipient_blocklist_uses_exact_string_not_domain_matching() -> (
    None
):
    engine = make_engine(
        {
            "comms.send": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="blocked_recipients",
                        operator="not_in",
                        values=frozenset({"alice@baddomain.com"}),
                        param="recipient",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    executor = AsyncMock(return_value="sent")
    send = make_send(engine, registry={"email": executor})

    result = await send(
        agent_id="agent-1",
        channel="email",
        recipient="bob@baddomain.com",
        text="allowed because exact recipient is different",
    )

    assert result == "sent"
    executor.assert_awaited_once_with(
        recipient="bob@baddomain.com",
        text="allowed because exact recipient is different",
    )


async def test_comms_send_escalate_does_not_call_executor() -> None:
    engine = make_engine(
        {
            "comms.send": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="daily_messages",
                        operator="lt",
                        value=0.0,
                        breach_verdict="ESCALATE",
                    )
                ]
            )
        }
    )
    executor = AsyncMock()
    send = make_send(engine, registry={"slack": executor})

    with pytest.raises(EscalationRequired):
        await send(
            agent_id="agent-1",
            channel="slack",
            recipient="ops",
            text="check this",
        )

    executor.assert_not_called()


async def test_comms_send_unregistered_channel_raises_before_governance() -> None:
    state_store = SpyStateStore()
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "comms.send": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="daily_messages",
                            operator="lt",
                            value=10.0,
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=state_store,
    )
    send = make_send(engine, registry={"email": Mock()})

    with pytest.raises(UnregisteredExecutorError) as exc_info:
        await send(
            agent_id="agent-1",
            channel="whatsapp",
            recipient="+85291234567",
            text="hello",
        )

    assert exc_info.value.skill_id == "comms.send"
    assert exc_info.value.key == "whatsapp"
    assert state_store.record_calls == []


async def test_comms_send_executor_not_in_engine_kwargs() -> None:
    engine = make_capturing_engine(
        {
            "comms.send": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="daily_messages",
                        operator="lt",
                        value=10.0,
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    executor = Mock(return_value="sent")
    send = make_send(engine, registry={"email": executor})

    await send(
        agent_id="agent-1",
        channel="email",
        recipient="alice@example.com",
        text="hello",
    )

    assert len(engine.captured_kwargs) == 1
    assert "executor" not in engine.captured_kwargs[0]


# ---------------------------------------------------------------------------
# Web skills
# NOTE: URL-string governance (DeterministicPolicy param="url") has been
# replaced by named-destination governance (param="destination"). This is a
# deliberate feature replacement — destination allowlists are strictly cleaner
# than URL string matching. The old URL allowlist tests are removed accordingly.
# ---------------------------------------------------------------------------


async def test_web_post_allow_calls_async_executor() -> None:
    engine = make_engine(
        {
            "web.post": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="approved_destinations",
                        operator="in",
                        values=frozenset({"ibkr", "yahoo"}),
                        param="destination",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    executor = AsyncMock(return_value={"status": "ok"})
    post = make_post(engine, registry={"ibkr": executor})

    result = await post(
        agent_id="agent-1",
        destination="ibkr",
        payload={"hello": "world"},
    )

    assert result == {"status": "ok"}
    executor.assert_awaited_once_with(payload={"hello": "world"})


async def test_web_post_allow_calls_sync_executor() -> None:
    engine = make_engine(
        {
            "web.post": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="approved_destinations",
                        operator="in",
                        values=frozenset({"ibkr", "yahoo"}),
                        param="destination",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    executor = Mock(return_value={"status": "ok"})
    post = make_post(engine, registry={"ibkr": executor})

    result = await post(
        agent_id="agent-1",
        destination="ibkr",
        payload={"hello": "world"},
    )

    assert result == {"status": "ok"}
    executor.assert_called_once_with(payload={"hello": "world"})


async def test_web_post_destination_allowlist_blocks_unapproved_destination() -> None:
    engine = make_engine(
        {
            "web.post": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="approved_destinations",
                        operator="in",
                        values=frozenset({"ibkr", "yahoo"}),
                        param="destination",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    executor = AsyncMock()
    post = make_post(engine, registry={"ibkr": executor, "evil": executor})

    with pytest.raises(BlockedError):
        await post(
            agent_id="agent-1",
            destination="evil",
            payload={"hello": "world"},
        )

    executor.assert_not_called()


async def test_web_post_escalate_does_not_call_executor() -> None:
    engine = make_engine(
        {
            "web.post": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="approved_destinations",
                        operator="in",
                        values=frozenset({"ibkr", "yahoo"}),
                        param="destination",
                        breach_verdict="ESCALATE",
                    )
                ]
            )
        }
    )
    executor = AsyncMock()
    post = make_post(engine, registry={"ibkr": executor, "evil": executor})

    with pytest.raises(EscalationRequired):
        await post(
            agent_id="agent-1",
            destination="evil",
            payload={"hello": "world"},
        )

    executor.assert_not_called()


async def test_web_post_unregistered_destination_raises_before_governance() -> None:
    state_store = SpyStateStore()
    engine = PolicyEngine(
        policy_store=InMemoryPolicyStore(
            {
                "web.post": PolicyConfig(
                    policies=[
                        DeterministicPolicy(
                            dimension="approved_destinations",
                            operator="in",
                            values=frozenset({"ibkr", "yahoo"}),
                            param="destination",
                            breach_verdict="BLOCK",
                        )
                    ]
                )
            }
        ),
        state_store=state_store,
    )
    post = make_post(engine, registry={"ibkr": AsyncMock()})

    with pytest.raises(UnregisteredExecutorError) as exc_info:
        await post(
            agent_id="agent-1",
            destination="yahoo",
            payload={"hello": "world"},
        )

    assert exc_info.value.skill_id == "web.post"
    assert exc_info.value.key == "yahoo"
    assert state_store.record_calls == []


async def test_web_post_executor_not_in_engine_kwargs() -> None:
    engine = make_capturing_engine(
        {
            "web.post": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="approved_destinations",
                        operator="in",
                        values=frozenset({"ibkr", "yahoo"}),
                        param="destination",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    executor = AsyncMock(return_value={"status": "ok"})
    post = make_post(engine, registry={"ibkr": executor})

    await post(
        agent_id="agent-1",
        destination="ibkr",
        payload={"hello": "world"},
    )

    assert len(engine.captured_kwargs) == 1
    assert "executor" not in engine.captured_kwargs[0]
    assert "url" not in engine.captured_kwargs[0]
    assert "destination" in engine.captured_kwargs[0]
