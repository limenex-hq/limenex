from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from limenex.core.engine import BlockedError, EscalationRequired, PolicyEngine
from limenex.core.policy import (
    DeterministicPolicy,
    PolicyConfig,
    UnregisteredSkillError,
)
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


def make_engine(configs: dict[str, PolicyConfig]) -> PolicyEngine:
    return PolicyEngine(
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
    charge = make_charge(engine)
    executor = Mock(return_value="charge-ok")

    result = await charge(
        agent_id="agent-1",
        amount=10.0,
        currency="USD",
        executor=executor,
    )

    assert result == "charge-ok"
    executor.assert_called_once_with(amount=10.0, currency="USD")


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
    charge = make_charge(engine)
    executor = Mock()

    with pytest.raises(BlockedError):
        await charge(
            agent_id="agent-1",
            amount=10.0,
            currency="USD",
            executor=executor,
        )

    executor.assert_not_called()


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
    spend = make_spend(engine)
    executor = AsyncMock(return_value={"ok": True})

    result = await spend(
        agent_id="agent-1",
        service="aws",
        amount_usd=25.0,
        executor=executor,
    )

    assert result == {"ok": True}
    executor.assert_awaited_once_with(service="aws", amount_usd=25.0)


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
    spend = make_spend(engine)
    executor = AsyncMock()

    with pytest.raises(EscalationRequired):
        await spend(
            agent_id="agent-1",
            service="openai",
            amount_usd=25.0,
            executor=executor,
        )

    executor.assert_not_called()


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
    send = make_send(engine)
    executor = Mock(return_value="sent")

    result = await send(
        agent_id="agent-1",
        channel="email",
        recipient="alice@example.com",
        text="hello",
        executor=executor,
    )

    assert result == "sent"
    executor.assert_called_once_with(
        channel="email",
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
    send = make_send(engine)
    executor = AsyncMock()

    with pytest.raises(BlockedError):
        await send(
            agent_id="agent-1",
            channel="email",
            recipient="alice@baddomain.com",
            text="should not send",
            executor=executor,
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
    send = make_send(engine)
    executor = AsyncMock(return_value="sent")

    result = await send(
        agent_id="agent-1",
        channel="email",
        recipient="bob@baddomain.com",
        text="allowed because exact recipient is different",
        executor=executor,
    )

    assert result == "sent"
    executor.assert_awaited_once_with(
        channel="email",
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
    send = make_send(engine)
    executor = AsyncMock()

    with pytest.raises(EscalationRequired):
        await send(
            agent_id="agent-1",
            channel="slack",
            recipient="ops",
            text="check this",
            executor=executor,
        )

    executor.assert_not_called()


# ---------------------------------------------------------------------------
# Web skills
# ---------------------------------------------------------------------------


async def test_web_post_allow_calls_async_executor() -> None:
    engine = make_engine(
        {
            "web.post": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="approved_urls",
                        operator="in",
                        values=frozenset({"https://api.example.com/v1/send"}),
                        param="url",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    post = make_post(engine)
    executor = AsyncMock(return_value={"status": "ok"})

    result = await post(
        agent_id="agent-1",
        url="https://api.example.com/v1/send",
        payload={"hello": "world"},
        executor=executor,
    )

    assert result == {"status": "ok"}
    executor.assert_awaited_once_with(
        url="https://api.example.com/v1/send",
        payload={"hello": "world"},
    )


async def test_web_post_url_allowlist_blocks_unapproved_url_and_never_calls_executor() -> (
    None
):
    engine = make_engine(
        {
            "web.post": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="approved_urls",
                        operator="in",
                        values=frozenset({"https://api.example.com/v1/send"}),
                        param="url",
                        breach_verdict="BLOCK",
                    )
                ]
            )
        }
    )
    post = make_post(engine)
    executor = AsyncMock()

    with pytest.raises(BlockedError):
        await post(
            agent_id="agent-1",
            url="https://evil.example.com/collect",
            payload={"hello": "world"},
            executor=executor,
        )

    executor.assert_not_called()


async def test_web_post_escalate_does_not_call_executor() -> None:
    engine = make_engine(
        {
            "web.post": PolicyConfig(
                policies=[
                    DeterministicPolicy(
                        dimension="approved_urls",
                        operator="in",
                        values=frozenset({"https://api.example.com/v1/send"}),
                        param="url",
                        breach_verdict="ESCALATE",
                    )
                ]
            )
        }
    )
    post = make_post(engine)
    executor = AsyncMock()

    with pytest.raises(EscalationRequired):
        await post(
            agent_id="agent-1",
            url="https://evil.example.com/collect",
            payload={"hello": "world"},
            executor=executor,
        )

    executor.assert_not_called()
