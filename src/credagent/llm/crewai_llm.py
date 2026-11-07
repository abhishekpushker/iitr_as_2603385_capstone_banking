"""`MockCrewLLM(BaseLLM)` — the CrewAI adapter over `MockEngine` (§ARCH 7.8.1, pitfalls P1–P3,
P18). One instance per agent per request: `(persona, tools, ctx)`.

`call()` algorithm:
1. Charge `ceil(chars/4)` tokens to `ctx.budget`. Over budget → the meter's `exceeded` flag is
   set and a schema-valid `budget_exceeded` refusal is returned. Never raises (CrewAI may swallow
   and retry); the orchestrator checks the flag after kickoff.
2. Our own previous action carries `⟦mock-action:{trace_id}:{n}⟧`. The observation is read
   **only** from the assistant message with that marker: the text after its first
   `"\\nObservation:"`, up to the end of that message. The system template's
   "Observation: the result of the action" (P1) and the later `post_tool_reasoning` nudge (P3)
   are never read.
3. Inputs come only from the `"Current Task:"` user message, between our delimiters
   (`task_inputs`). The record-id block comes first and the query runs to the **last**
   `<<<END_QUERY>>>`, so delimiter text inside a query can neither forge an id nor cut the query.
   The Composer finds the prior task outputs in its context by their JSON `kind`.
4. Tool arguments come from each tool's `args_schema.model_fields` through the resolver
   `{query, record_id}`; a tool is callable only if every required field resolves. The tool name
   is copied, never interpreted (P2).
5. We never emit `Observation:` (a CrewAI stop word): any occurrence inside our JSON is written
   as `Observation\\u003a`, which decodes to the same string. More than `MAX_STEPS` calls → an
   `agent_error` Final Answer. Any protocol surprise also fails closed as `agent_error`.

Retrieval, Lookup and protocol-error refusals are all `CrewResponse` JSON, so the Composer can
pass an upstream refusal straight through.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr, ValidationError

from credagent.context import RequestContext, estimate_tokens
from credagent.llm.engine import MockEngine
from credagent.schemas import (
    LOAN_STATUS_OUTPUT,
    LOOKUP_TASK_OUTPUT,
    RAG_TOOL_OUTPUT,
    CrewResponse,
    LookupNotFound,
    RefusalReason,
    RetrievalTaskOutput,
)

Persona = Literal["retrieval", "lookup", "composer"]

QUERY_OPEN: Final = "<<<QUERY>>>"
QUERY_CLOSE: Final = "<<<END_QUERY>>>"
RECORD_ID_OPEN: Final = "<<<RECORD_ID>>>"
RECORD_ID_CLOSE: Final = "<<<END_RECORD_ID>>>"
NO_RECORD_ID: Final = "none"
DELIMITERS: Final = (QUERY_OPEN, QUERY_CLOSE, RECORD_ID_OPEN, RECORD_ID_CLOSE)

CURRENT_TASK: Final = "Current Task:"
CONTEXT_HEADER: Final = "This is the context you're working with:"  # D-014.4
OBSERVATION: Final = "\nObservation:"
MARKER_PREFIX: Final = "⟦mock-action:"
MAX_STEPS: Final = 3
CONTEXT_WINDOW: Final = 32768


def task_inputs(query: str, record_id: str | None) -> str:
    """The delimited inputs block every task description embeds (record id first)."""
    rid = NO_RECORD_ID if record_id is None else record_id
    return f"{RECORD_ID_OPEN}{rid}{RECORD_ID_CLOSE}\n{QUERY_OPEN}{query}{QUERY_CLOSE}"


class _ProtocolError(Exception):
    """CrewAI handed us something our protocol does not expect; answer `agent_error`."""


def _text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _messages(messages: str | Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    if isinstance(messages, str):
        return [("user", messages)]
    return [(str(m.get("role", "")), _text(m.get("content", ""))) for m in messages]


def _no_observation(text: str) -> str:
    return text.replace("Observation:", "Observation\\u003a")


def _final(model: BaseModel) -> str:
    payload = _no_observation(model.model_dump_json())
    return f"Thought: I now know the final answer\nFinal Answer: {payload}"


def _json_objects(text: str) -> list[dict[str, Any]]:
    """Every top-level JSON object embedded in `text`, in order."""
    decoder, out, i = json.JSONDecoder(), [], text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        out.append(obj)  # decoding from "{" only ever yields an object
        i = text.find("{", end)
    return out


class MockCrewLLM(BaseLLM):
    response_format: type[BaseModel] | None = None  # §ARCH 7.11: the Composer's CrewResponse

    _persona: Persona = PrivateAttr()
    _tools: tuple[BaseTool, ...] = PrivateAttr()
    _ctx: RequestContext = PrivateAttr()
    _engine: MockEngine = PrivateAttr()
    _calls: int = PrivateAttr(default=0)
    _actions: int = PrivateAttr(default=0)

    def __init__(
        self,
        *,
        persona: Persona,
        tools: Sequence[BaseTool],
        ctx: RequestContext,
        engine: MockEngine | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> None:
        super().__init__(model="mock-llm", response_format=response_format)
        self._persona = persona
        self._tools = tuple(tools)
        self._ctx = ctx
        self._engine = MockEngine() if engine is None else engine
        self._marker = re.compile(re.escape(f"{MARKER_PREFIX}{ctx.trace_id}:") + r"\d+⟧")

    _marker: re.Pattern[str] = PrivateAttr()

    @property
    def persona(self) -> Persona:
        return self._persona

    @property
    def calls(self) -> int:
        return self._calls

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return CONTEXT_WINDOW

    def call(  # noqa: PLR0917  (mirrors BaseLLM.call)
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        msgs = _messages(messages)
        self._calls += 1
        tokens = estimate_tokens("".join(content for _, content in msgs))
        if self._ctx.budget.charge(self._persona, tokens):
            out = self._refuse("budget_exceeded")
        elif self._calls > MAX_STEPS:
            out = self._refuse("agent_error")
        else:
            try:
                out = self._step(msgs)
            except _ProtocolError:
                out = self._refuse("agent_error")
        self._ctx.budget.record_completion(self._persona, estimate_tokens(out))
        return out

    # ------------------------------------------------------------------ protocol
    def _refuse(self, reason: RefusalReason) -> str:
        return _final(self._engine.refusal(reason))

    def _step(self, msgs: list[tuple[str, str]]) -> str:
        task = next((c for r, c in msgs if r == "user" and CURRENT_TASK in c), None)
        if task is None:
            raise _ProtocolError("no Current Task message")
        record_id = self._between(task, RECORD_ID_OPEN, RECORD_ID_CLOSE, last=False)
        query = self._between(task, QUERY_OPEN, QUERY_CLOSE, last=True)
        rid = None if record_id.strip() == NO_RECORD_ID else record_id.strip()
        if self._persona == "composer":
            return self._compose(task, query, rid)
        observation = self._observation(msgs)
        if self._persona == "retrieval":
            return self._retrieve(query, rid, observation)
        return self._lookup(query, rid, observation)

    @staticmethod
    def _between(text: str, start: str, end: str, *, last: bool) -> str:
        a = text.find(start)
        b = text.rfind(end) if last else text.find(end, a + len(start))
        if a == -1 or b < a + len(start):
            raise _ProtocolError(f"missing {start}…{end}")
        return text[a + len(start) : b]

    def _observation(self, msgs: list[tuple[str, str]]) -> Any:
        ours = [c for r, c in msgs if r == "assistant" and self._marker.search(c)]
        if not ours:
            return None
        _, sep, observation = ours[-1].partition(OBSERVATION)
        if not sep:
            raise _ProtocolError("our action has no observation")
        try:
            return json.loads(observation)
        except json.JSONDecodeError as exc:
            raise _ProtocolError("observation is not JSON") from exc

    def _act(self, query: str, record_id: str | None) -> str:
        resolver = {"query": query, "record_id": record_id}
        for tool in self._tools:
            fields = tool.args_schema.model_fields
            if all(resolver.get(n) is not None for n, f in fields.items() if f.is_required()):
                args = {n: resolver[n] for n in fields if resolver.get(n) is not None}
                self._actions += 1
                marker = f"{MARKER_PREFIX}{self._ctx.trace_id}:{self._actions}⟧"
                return _no_observation(
                    f"Thought: {marker} I will call my tool with the arguments its schema "
                    f"declares.\nAction: {tool.name}\nAction Input: "
                    f"{json.dumps(args, ensure_ascii=False)}"
                )
        raise _ProtocolError("no tool's required arguments resolve")

    def _retrieve(self, query: str, record_id: str | None, observation: Any) -> str:
        if self._engine.classify_intent(query, record_id) in {"status", "clarify"}:
            return _final(RetrievalTaskOutput(kind="retrieval_result", rag=None))
        if observation is None:
            return self._act(query, record_id)
        try:
            rag = RAG_TOOL_OUTPUT.validate_python(observation)
        except ValidationError as exc:
            raise _ProtocolError("observation is not a rag_lookup output") from exc
        return _final(RetrievalTaskOutput(kind="retrieval_result", rag=rag))

    def _lookup(self, query: str, record_id: str | None, observation: Any) -> str:
        if record_id is None:
            return _final(LookupNotFound(kind="lookup_result", found=False, error="no_record_id"))
        if observation is None:
            return self._act(query, record_id)
        try:
            result = LOAN_STATUS_OUTPUT.validate_python(observation)
        except ValidationError as exc:
            raise _ProtocolError("observation is not a lookup output") from exc
        return _final(
            LOOKUP_TASK_OUTPUT.validate_python({"kind": "lookup_result", **result.model_dump()})
        )

    def _compose(self, task: str, query: str, record_id: str | None) -> str:
        context = task.partition(CONTEXT_HEADER)[2]
        by_kind: dict[str, dict[str, Any]] = {}
        for obj in _json_objects(context):
            by_kind.setdefault(str(obj.get("kind")), obj)
        upstream = by_kind.get("crew_response")
        try:
            if upstream is not None:
                refusal = CrewResponse.model_validate(upstream)
                if refusal.refused:
                    return _final(refusal)
            retrieval = RetrievalTaskOutput.model_validate(by_kind["retrieval_result"])
            lookup = LOOKUP_TASK_OUTPUT.validate_python(by_kind["lookup_result"])
        except (KeyError, ValidationError) as exc:
            raise _ProtocolError("task context is missing or invalid") from exc
        return _final(self._engine.compose(retrieval, lookup, query=query, record_id=record_id))
