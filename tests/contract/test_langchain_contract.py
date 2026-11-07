"""langchain-core 1.7.x session memory behaviour (§ARCH 2.4, 7.10, P19)."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import langchain_core
import pytest
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory


def test_version_is_below_two() -> None:
    """P19: InMemoryChatMessageHistory is removed in 2.0."""
    major, minor, *_ = (int(p) for p in langchain_core.__version__.split("."))
    assert (major, minor) >= (1, 7)
    assert major < 2


_WARNINGS_PROBE = """
import json, warnings
warnings.simplefilter("always")
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
seen = []
for _ in range(2):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h = InMemoryChatMessageHistory()
        RunnableWithMessageHistory(RunnableLambda(lambda d: "x"), get_session_history=lambda s: h,
                                   input_messages_key="input", history_messages_key="history")
    seen.append([[w.category.__name__, str(w.message)] for w in caught])
print(json.dumps(seen))
"""


def test_both_classes_emit_their_deprecation_warnings() -> None:
    """Expected and NOT silenced (§ARCH 2.4). Checked in a fresh interpreter because
    InMemoryChatMessageHistory warns once per process (decorator flag), while
    RunnableWithMessageHistory warns on every construction."""
    proc = subprocess.run(
        [sys.executable, "-c", _WARNINGS_PROBE],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    first, second = json.loads(proc.stdout)
    assert [category for category, _ in first] == ["LangChainDeprecationWarning"] * 2
    assert "InMemoryChatMessageHistory" in first[0][1]
    assert "removed in 2.0" in first[0][1]
    assert "RunnableWithMessageHistory is deprecated" in first[1][1]
    assert [category for category, _ in second] == ["LangChainDeprecationWarning"]
    assert "RunnableWithMessageHistory is deprecated" in second[0][1]


def test_deprecation_warning_class_is_a_deprecation_warning() -> None:
    assert issubclass(LangChainDeprecationWarning, DeprecationWarning)


def _wrap(store: dict[str, InMemoryChatMessageHistory]) -> RunnableWithMessageHistory:
    def core_turn(inputs: dict[str, Any]) -> AIMessage:
        history: list[BaseMessage] = inputs["history"]
        return AIMessage(
            content=f"seen={len(history)}:{inputs['input']}",
            additional_kwargs={"record_id": "LA-0007"},
        )

    return RunnableWithMessageHistory(
        RunnableLambda(core_turn),
        get_session_history=store.__getitem__,
        input_messages_key="input",
        history_messages_key="history",
    )


@pytest.fixture
def store() -> dict[str, InMemoryChatMessageHistory]:
    return {"s1": InMemoryChatMessageHistory(), "s2": InMemoryChatMessageHistory()}


def _ask(chain: RunnableWithMessageHistory, session: str, text: str) -> Any:
    return chain.invoke({"input": text}, config={"configurable": {"session_id": session}})


def test_history_is_written_by_the_runnable(store: dict[str, InMemoryChatMessageHistory]) -> None:
    chain = _wrap(store)
    assert _ask(chain, "s1", "one").content == "seen=0:one"
    assert _ask(chain, "s1", "two").content == "seen=2:two"
    msgs = store["s1"].messages
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage, HumanMessage, AIMessage]
    assert msgs[1].additional_kwargs == {"record_id": "LA-0007"}


def test_sessions_are_isolated(store: dict[str, InMemoryChatMessageHistory]) -> None:
    chain = _wrap(store)
    _ask(chain, "s1", "one")
    _ask(chain, "s1", "two")
    assert _ask(chain, "s2", "fresh").content == "seen=0:fresh"
    assert (len(store["s1"].messages), len(store["s2"].messages)) == (4, 2)


def test_sessions_are_isolated_under_threads(store: dict[str, InMemoryChatMessageHistory]) -> None:
    chain = _wrap(store)

    def worker(session: str) -> None:
        for i in range(5):
            _ask(chain, session, f"{session}-{i}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, ["s1", "s2"]))
    for session in ("s1", "s2"):
        humans = [m.content for m in store[session].messages if isinstance(m, HumanMessage)]
        assert humans == [f"{session}-{i}" for i in range(5)]


# ------------------------------------------------------------------ Phase 7: what memory relies on
def _async_chain(store: dict[str, InMemoryChatMessageHistory]) -> RunnableWithMessageHistory:
    async def core_turn(inputs: dict[str, Any]) -> AIMessage:
        if inputs["input"] == "boom":
            raise RuntimeError("turn failed")
        return AIMessage(
            content=f"seen={len(inputs['history'])}:{inputs['input']}:{inputs['extra']}",
            additional_kwargs={"record_id": "LA-0007", "category": "Home"},
        )

    return RunnableWithMessageHistory(
        RunnableLambda(core_turn),
        get_session_history=store.__getitem__,
        input_messages_key="input",
        history_messages_key="history",
    )


async def _aask(chain: RunnableWithMessageHistory, session: str, text: str) -> Any:
    return await chain.ainvoke(
        {"input": text, "extra": "x"}, config={"configurable": {"session_id": session}}
    )


@pytest.mark.anyio
async def test_async_path_writes_history_and_passes_extra_input_keys(
    store: dict[str, InMemoryChatMessageHistory],
) -> None:
    chain = _async_chain(store)
    assert (await _aask(chain, "s1", "one")).content == "seen=0:one:x"
    assert (await _aask(chain, "s1", "two")).content == "seen=2:two:x"
    msgs = store["s1"].messages
    assert [(type(m), m.content) for m in msgs[::2]] == [
        (HumanMessage, "one"),
        (HumanMessage, "two"),
    ]
    assert msgs[3].additional_kwargs == {"record_id": "LA-0007", "category": "Home"}


@pytest.mark.anyio
async def test_a_failed_turn_writes_nothing_to_history(
    store: dict[str, InMemoryChatMessageHistory],
) -> None:
    """The orchestrator relies on this: a timeout or budget overrun inside the turn must not
    leave a half-written exchange in memory."""
    chain = _async_chain(store)
    await _aask(chain, "s1", "one")
    with pytest.raises(RuntimeError, match="turn failed"):
        await _aask(chain, "s1", "boom")
    assert len(store["s1"].messages) == 2


def test_a_failed_sync_turn_writes_nothing_to_history(
    store: dict[str, InMemoryChatMessageHistory],
) -> None:
    def core_turn(inputs: dict[str, Any]) -> AIMessage:
        raise RuntimeError("turn failed")

    chain = RunnableWithMessageHistory(
        RunnableLambda(core_turn), get_session_history=store.__getitem__,
        input_messages_key="input", history_messages_key="history",
    )  # fmt: skip
    with pytest.raises(RuntimeError):
        _ask(chain, "s1", "x")
    assert store["s1"].messages == []
