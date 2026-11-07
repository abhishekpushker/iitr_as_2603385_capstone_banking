"""`MockAutogenClient(ChatCompletionClient)` — the Autogen adapter over `MockEngine` (§ARCH 7.8.2,
P11, P12). One instance per agent per request: `(persona, ctx)`.

* All nine abstract members are implemented; `model_info` declares `json_output` and
  `structured_output`, which `AssistantAgent(output_content_type=ReviewVerdict)` requires.
* Every `create` charges `ceil(chars/4)` of its input messages to `ctx.budget` under the persona
  (`reviewer` / `editor`). Over budget only sets the meter's flag; the pipeline's post-check
  raises (never raise for budget, §ARCH 7.8.1).
* **reviewer** reads only the task message (`source == "user"`): the draft between `<<<DRAFT>>>`
  and `<<<CONTEXT>>>`, and the context items after `<<<CONTEXT>>>`, separated by `<<<ITEM>>>`.
  It replies with the draft between `<<<DRAFT>>>`/`<<<END_DRAFT>>>`, then `MockEngine.review`,
  whose last line is `REVIEW_JSON: {...}`.
* **editor** reads only the reviewer's message (`source == "Policy_Compliance_Reviewer"`): the
  draft block and the `REVIEW_JSON` line, and replies with `MockEngine.edit`'s `ReviewVerdict`
  JSON.

Any message it cannot read raises `ReviewProtocolError`; `team.run()` then fails and the review
stage fails closed. The question (`<<<QUESTION>>>`) is part of the task for a real model; the
mock's support rule does not use it.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any, Final, Literal

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel

from credagent.context import RequestContext, estimate_tokens
from credagent.llm.engine import REVIEW_JSON_PREFIX, MockEngine

Persona = Literal["reviewer", "editor"]

REVIEWER_NAME: Final = "Policy_Compliance_Reviewer"
EDITOR_NAME: Final = "Final_Editor"
TASK_SOURCE: Final = "user"
QUESTION_OPEN: Final = "<<<QUESTION>>>"
DRAFT_OPEN: Final = "<<<DRAFT>>>"
DRAFT_CLOSE: Final = "<<<END_DRAFT>>>"
CONTEXT_OPEN: Final = "<<<CONTEXT>>>"
CONTEXT_ITEM: Final = "<<<ITEM>>>"
CONTEXT_WINDOW: Final = 32768


class ReviewProtocolError(ValueError):
    """A message the mock cannot read (wrong source, missing delimiter, bad REVIEW_JSON)."""


def _text(message: LLMMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else "".join(str(part) for part in content)


def _between(text: str, start: str, end: str | None) -> str:
    head, found, rest = text.partition(start)
    if not found:
        raise ReviewProtocolError(f"missing {start}")
    if end is None:
        return rest.strip("\n")
    body, found, _ = rest.partition(end)
    if not found:
        raise ReviewProtocolError(f"missing {end}")
    return body.strip("\n")


def _only_from(messages: Sequence[LLMMessage], source: str) -> str:
    found = [_text(m) for m in messages if getattr(m, "source", None) == source]
    if len(found) != 1:
        raise ReviewProtocolError(f"expected one message from {source!r}, got {len(found)}")
    return found[0]


class MockAutogenClient(ChatCompletionClient):
    def __init__(
        self, persona: Persona, ctx: RequestContext, engine: MockEngine | None = None
    ) -> None:
        self._persona: Persona = persona
        self._ctx = ctx
        self._engine = MockEngine() if engine is None else engine
        self._last = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._total = RequestUsage(prompt_tokens=0, completion_tokens=0)

    # ------------------------------------------------------------------ the two personas
    def _reply(self, messages: Sequence[LLMMessage]) -> str:
        if self._persona == "reviewer":
            task = _only_from(messages, TASK_SOURCE)
            draft = _between(task, DRAFT_OPEN, CONTEXT_OPEN)
            context = [
                c.strip("\n") for c in _between(task, CONTEXT_OPEN, None).split(CONTEXT_ITEM)
            ]
            report = self._engine.review(draft, [c for c in context if c])
            return f"{DRAFT_OPEN}\n{draft}\n{DRAFT_CLOSE}\n{report}"
        review = _only_from(messages, REVIEWER_NAME)
        draft = _between(review, DRAFT_OPEN, DRAFT_CLOSE)
        last = review.rsplit("\n", 1)[-1]
        if not last.startswith(REVIEW_JSON_PREFIX):
            raise ReviewProtocolError("the reviewer's last line is not REVIEW_JSON")
        verdict = self._engine.edit(last.removeprefix(REVIEW_JSON_PREFIX), draft)
        return verdict.model_dump_json()

    # ------------------------------------------------------------------ ChatCompletionClient
    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = (),
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},  # noqa: B006  (mirrors the library signature)
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        prompt_tokens = self.count_tokens(messages)
        self._ctx.budget.charge(self._persona, prompt_tokens)
        content = self._reply(messages)
        self._last = RequestUsage(
            prompt_tokens=prompt_tokens, completion_tokens=estimate_tokens(content)
        )
        self._ctx.budget.record_completion(self._persona, self._last.completion_tokens)
        self._total = RequestUsage(
            prompt_tokens=self._total.prompt_tokens + self._last.prompt_tokens,
            completion_tokens=self._total.completion_tokens + self._last.completion_tokens,
        )
        return CreateResult(finish_reason="stop", content=content, usage=self._last, cached=False)

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = (),
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},  # noqa: B006
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        """Not used by the team; yields the one final result."""
        yield await self.create(messages, json_output=json_output)

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._last

    def total_usage(self) -> RequestUsage:
        return self._total

    def count_tokens(
        self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = ()
    ) -> int:
        return estimate_tokens("".join(_text(m) for m in messages))

    def remaining_tokens(
        self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = ()
    ) -> int:
        return CONTEXT_WINDOW - self.count_tokens(messages)

    @property
    def capabilities(self) -> ModelCapabilities:  # deprecated upstream, still abstract
        return ModelCapabilities(vision=False, function_calling=False, json_output=True)

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False, function_calling=False, json_output=True, structured_output=True,
            family="unknown",
        )  # fmt: skip
