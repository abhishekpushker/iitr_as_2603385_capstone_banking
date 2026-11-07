"""Session memory (§ARCH 7.10): SessionStore (LRU + idle TTL + reset + per-session lock) and the
RunnableWithMessageHistory turn wrapper."""

from __future__ import annotations

import anyio
import anyio.lowlevel
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from credagent.memory.session import ConversationMemory, SessionStore


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def store(
    max_sessions: int = 1000, ttl_s: float = 3600.0, clock: FakeClock | None = None
) -> SessionStore:
    return SessionStore(max_sessions=max_sessions, ttl_s=ttl_s, clock=clock or FakeClock())


# ------------------------------------------------------------------ store
def test_get_creates_once_and_returns_the_same_history() -> None:
    s = store()
    h = s.get("a")
    assert s.get("a") is h
    assert len(s) == 1


def test_inspection_never_creates_a_session() -> None:
    s = store()
    assert s.history_length("nope") == 0
    assert s.messages("nope") == []
    assert "nope" not in s and len(s) == 0


def test_lru_cap_evicts_the_least_recently_used_session() -> None:
    s = store(max_sessions=3)
    for sid in ("a", "b", "c"):
        s.get(sid).add_user_message(sid)
    s.get("a")  # touch: b is now the least recently used
    s.get("d")
    assert sorted(s.session_ids()) == ["a", "c", "d"]


def test_idle_sessions_expire_after_the_ttl() -> None:
    clock = FakeClock()
    s = store(ttl_s=3600, clock=clock)
    s.get("old").add_user_message("x")
    clock.now += 1800
    s.get("young")
    clock.now += 1800  # "old" idle exactly 3600 s: expired; "young" 1800 s: kept
    assert s.history_length("old") == 0
    assert "old" not in s and "young" in s


def test_access_refreshes_the_ttl() -> None:
    clock = FakeClock()
    s = store(ttl_s=10, clock=clock)
    s.get("a").add_user_message("x")
    clock.now += 9
    s.get("a")
    clock.now += 9
    assert s.history_length("a") == 1


def test_reset_deletes_the_history() -> None:
    s = store()
    s.get("a").add_user_message("x")
    assert s.reset("a") is True
    assert s.history_length("a") == 0 and "a" not in s
    assert s.reset("a") is False


def test_lock_is_per_session_and_stable() -> None:
    s = store()
    assert s.lock("a") is s.lock("a")
    assert s.lock("a") is not s.lock("b")


def test_from_settings_uses_the_documented_limits() -> None:
    s = SessionStore.from_settings()
    assert (s.max_sessions, s.ttl_s) == (1000, 3600.0)


@pytest.mark.parametrize(("max_sessions", "ttl_s"), [(0, 1.0), (1, 0.0)])
def test_invalid_limits_are_rejected(max_sessions: int, ttl_s: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        SessionStore(max_sessions=max_sessions, ttl_s=ttl_s)


# ------------------------------------------------------------------ the Runnable writes history
def echo_turn(record_id: str | None = None):  # type: ignore[no-untyped-def]
    async def turn(masked: str, history: list[BaseMessage]) -> AIMessage:
        return AIMessage(
            content=f"seen={len(history)}:{masked}",
            additional_kwargs={"record_id": record_id, "category": None},
        )

    return turn


@pytest.mark.anyio
async def test_a_turn_sees_the_history_and_the_runnable_appends_both_messages() -> None:
    memory = ConversationMemory(store())
    first = await memory.run_turn("s", "hello", echo_turn("LA-0007"))
    second = await memory.run_turn("s", "again", echo_turn())
    assert (first.content, second.content) == ("seen=0:hello", "seen=2:again")
    msgs = memory.store.messages("s")
    assert [(type(m), m.content) for m in msgs] == [
        (HumanMessage, "hello"), (AIMessage, "seen=0:hello"),
        (HumanMessage, "again"), (AIMessage, "seen=2:again"),
    ]  # fmt: skip
    assert msgs[1].additional_kwargs == {"record_id": "LA-0007", "category": None}


@pytest.mark.anyio
async def test_a_failed_turn_is_not_stored_and_releases_the_lock() -> None:
    memory = ConversationMemory(store())

    async def boom(masked: str, history: list[BaseMessage]) -> AIMessage:
        raise TimeoutError("crew too slow")

    with pytest.raises(TimeoutError):
        await memory.run_turn("s", "hello", boom)
    assert memory.store.history_length("s") == 0
    assert not memory.store.lock("s").locked()


@pytest.mark.anyio
async def test_turns_in_one_session_are_serialised() -> None:
    memory = ConversationMemory(store())
    release = anyio.Event()
    seen: list[int] = []

    async def slow(masked: str, history: list[BaseMessage]) -> AIMessage:
        seen.append(len(history))
        if masked == "first":
            await release.wait()
        return AIMessage(content=masked)

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(memory.run_turn, "s", "first", slow)
            await anyio.wait_all_tasks_blocked()
            tg.start_soon(memory.run_turn, "s", "second", slow)
            await anyio.wait_all_tasks_blocked()
            assert seen == [0]  # the second turn waits for the first
            release.set()
    assert seen == [0, 2]


@pytest.mark.anyio
async def test_different_sessions_run_in_parallel() -> None:
    memory = ConversationMemory(store())
    a_started = anyio.Event()

    async def a_turn(masked: str, history: list[BaseMessage]) -> AIMessage:
        a_started.set()
        await b_done.wait()
        return AIMessage(content="a")

    b_done = anyio.Event()

    async def b_turn(masked: str, history: list[BaseMessage]) -> AIMessage:
        await a_started.wait()
        b_done.set()
        return AIMessage(content="b")

    with anyio.fail_after(5):  # a deadlock here would mean sessions share a lock
        async with anyio.create_task_group() as tg:
            tg.start_soon(memory.run_turn, "a", "x", a_turn)
            tg.start_soon(memory.run_turn, "b", "y", b_turn)


@pytest.mark.anyio
async def test_ten_concurrent_sessions_stay_isolated() -> None:
    memory = ConversationMemory(store())

    async def turn(masked: str, history: list[BaseMessage]) -> AIMessage:
        await anyio.lowlevel.checkpoint()  # yield, so the sessions interleave
        session = masked.split(":", maxsplit=1)[0]
        assert all(str(m.content).startswith(session + ":") for m in history)
        return AIMessage(content=f"{masked}:answer", additional_kwargs={"record_id": None})

    async def converse(i: int) -> None:
        for t in range(3):
            await memory.run_turn(f"s{i}", f"s{i}:turn{t}", turn)

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            for i in range(10):
                tg.start_soon(converse, i)
    for i in range(10):
        humans = [m.content for m in memory.store.messages(f"s{i}") if isinstance(m, HumanMessage)]
        assert humans == [f"s{i}:turn{t}" for t in range(3)]
        assert memory.store.history_length(f"s{i}") == 6


@pytest.mark.anyio
async def test_busy_sessions_and_the_one_being_accessed_are_never_evicted() -> None:
    s = store(max_sessions=1)
    async with s.lock("busy"):
        s.get("new").add_user_message("x")  # over the cap, but "busy" is locked
        assert sorted(s.session_ids()) == ["busy", "new"]
    s.get("third")
    assert s.session_ids() == ["third"]


def test_the_turn_wrapper_is_a_runnable_with_message_history_over_the_store() -> None:
    from langchain_core.runnables.history import RunnableWithMessageHistory

    s = store()
    memory = ConversationMemory(s)
    assert isinstance(memory.runnable, RunnableWithMessageHistory)
    assert memory.runnable.get_session_history == s.get
    assert (memory.runnable.input_messages_key, memory.runnable.history_messages_key) == (
        "input", "history",
    )  # fmt: skip
