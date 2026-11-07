"""Session memory (§ARCH 7.10, Task 8).

`SessionStore`: `session_id → InMemoryChatMessageHistory` behind a `threading.Lock`, with an
`asyncio.Lock` per session so two requests in one session run one after the other while
different sessions run in parallel. LRU cap (`SESSION_MAX`, 1000) and idle TTL (`SESSION_TTL_S`,
1 h), both enforced on access; a session whose lock is held is never evicted. `reset` deletes a
session's history. Inspection (`history_length`, `messages`) never creates a session.

`ConversationMemory.run_turn(session_id, masked_text, turn)` runs one turn through
`RunnableWithMessageHistory(RunnableLambda(…), get_session_history=store.get,
input_messages_key="input", history_messages_key="history")` under the session lock. `turn`
receives the masked text and the history so far and returns the `AIMessage` (with
`additional_kwargs={"record_id", "category"}`); appending `HumanMessage(masked)` and that message
is the Runnable's job, and it does so only when the turn succeeds (contract-tested). Only masked
text ever reaches this module.

langchain-core 1.7 marks both classes deprecated (removed in 2.0, pinned `<2.0`, P19). The
warnings are expected and deliberately not silenced (§ARCH 2.4).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from credagent.settings import get_settings

Turn = Callable[[str, list[BaseMessage]], Awaitable[AIMessage]]


@dataclass
class _Session:
    history: InMemoryChatMessageHistory
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = 0.0


class SessionStore:
    def __init__(
        self, *, max_sessions: int, ttl_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be ≥ 1")
        if ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")
        self.max_sessions = max_sessions
        self.ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, _Session] = OrderedDict()

    @classmethod
    def from_settings(cls) -> SessionStore:
        s = get_settings()
        return cls(max_sessions=s.session_max, ttl_s=s.session_ttl_s)

    # ------------------------------------------------------------------ access (creates)
    def _session(self, session_id: str) -> _Session:
        with self._lock:
            now = self._clock()
            self._evict(now)
            session = self._sessions.get(session_id)
            if session is None:
                session = _Session(InMemoryChatMessageHistory(), last_used=now)
                self._sessions[session_id] = session
            session.last_used = now
            self._sessions.move_to_end(session_id)
            self._evict(now, keep=session_id)  # the cap, now that this session is most recent
            return session

    def get(self, session_id: str) -> InMemoryChatMessageHistory:
        """The session's history, created if needed (`get_session_history` for the Runnable)."""
        return self._session(session_id).history

    def lock(self, session_id: str) -> asyncio.Lock:
        return self._session(session_id).lock

    def _evict(self, now: float, keep: str | None = None) -> None:
        """Drop idle sessions, then the least recently used beyond the cap; never a busy one,
        nor `keep` (the session being accessed)."""
        for sid in [s for s, v in self._sessions.items() if now - v.last_used >= self.ttl_s]:
            if not self._sessions[sid].lock.locked():
                del self._sessions[sid]
        idle = (s for s, v in list(self._sessions.items()) if not v.lock.locked() and s != keep)
        while len(self._sessions) > self.max_sessions:
            victim = next(idle, None)
            if victim is None:
                break
            del self._sessions[victim]

    # ------------------------------------------------------------------ inspection (never creates)
    def _peek(self, session_id: str) -> _Session | None:
        with self._lock:
            self._evict(self._clock())
            return self._sessions.get(session_id)

    def history_length(self, session_id: str) -> int:
        session = self._peek(session_id)
        return 0 if session is None else len(session.history.messages)

    def messages(self, session_id: str) -> list[BaseMessage]:
        session = self._peek(session_id)
        return [] if session is None else list(session.history.messages)

    def session_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions)

    def reset(self, session_id: str) -> bool:
        """Delete the session's history; `False` if there was none."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.history.clear()
        return True

    def __contains__(self, session_id: object) -> bool:
        with self._lock:
            return session_id in self._sessions

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


async def _dispatch(inputs: dict[str, Any]) -> AIMessage:
    turn: Turn = inputs["turn"]
    return await turn(inputs["input"], list(inputs["history"]))


class ConversationMemory:
    def __init__(self, store: SessionStore) -> None:
        self.store = store
        self._chain = RunnableWithMessageHistory(
            RunnableLambda(_dispatch),
            get_session_history=store.get,
            input_messages_key="input",
            history_messages_key="history",
        )

    @property
    def runnable(self) -> RunnableWithMessageHistory:
        return self._chain

    async def run_turn(self, session_id: str, masked_text: str, turn: Turn) -> AIMessage:
        async with self.store.lock(session_id):
            result = await self._chain.ainvoke(
                {"input": masked_text, "turn": turn},
                config={"configurable": {"session_id": session_id}},
            )
        if not isinstance(result, AIMessage):
            raise TypeError("a turn must return an AIMessage")
        return result
