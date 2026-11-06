"""autogen-agentchat/core 0.7.5 behaviour the review stage depends on (§ARCH 2.3, P11, P12)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import StructuredMessage, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    ModelInfo,
    RequestUsage,
)
from pydantic import BaseModel

import credagent  # noqa: F401


class Verdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


class ScriptedClient(ChatCompletionClient):
    """Minimal client implementing every abstract member; replies with a fixed string."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},  # noqa: B006  (mirrors the library signature)
        cancellation_token: Any = None,
    ) -> CreateResult:
        self.calls.append({"messages": list(messages), "json_output": json_output})
        return CreateResult(
            finish_reason="stop",
            content=self.reply,
            usage=RequestUsage(prompt_tokens=1, completion_tokens=1),
            cached=False,
        )

    async def create_stream(
        self, messages: Sequence[LLMMessage], **kwargs: Any
    ) -> AsyncGenerator[str | CreateResult, None]:
        yield await self.create(messages, **kwargs)

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        return 0

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        return 32768

    @property
    def capabilities(self) -> ModelCapabilities:  # deprecated upstream, still abstract
        return ModelCapabilities(vision=False, function_calling=False, json_output=True)

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False,
            function_calling=False,
            json_output=True,
            structured_output=True,
            family="unknown",
        )


VERDICT = {"approved": True, "final_answer": "draft", "reason": "All 1 sentences supported."}


def _team(with_custom_types: bool) -> tuple[RoundRobinGroupChat, ScriptedClient, ScriptedClient]:
    reviewer_client = ScriptedClient('notes\nREVIEW_JSON: {"unsupported": []}')
    editor_client = ScriptedClient(json.dumps(VERDICT))
    reviewer = AssistantAgent(
        "Policy_Compliance_Reviewer", model_client=reviewer_client, system_message="review"
    )
    editor = AssistantAgent(
        "Final_Editor",
        model_client=editor_client,
        system_message="edit",
        output_content_type=Verdict,
    )
    kwargs: dict[str, Any] = {}
    if with_custom_types:
        kwargs["custom_message_types"] = [StructuredMessage[Verdict]]
    team = RoundRobinGroupChat([reviewer, editor], max_turns=2, **kwargs)
    return team, reviewer_client, editor_client


@pytest.fixture(scope="module")
def run() -> tuple[TaskResult, ScriptedClient, ScriptedClient]:
    team, rc, ec = _team(with_custom_types=True)
    result = asyncio.run(team.run(task="<<<DRAFT>>>draft<<<CONTEXT>>>ctx"))
    return result, rc, ec


def test_chat_completion_client_abstract_members() -> None:
    assert ChatCompletionClient.__abstractmethods__ == frozenset(
        {
            "create", "create_stream", "close", "actual_usage", "total_usage",
            "count_tokens", "remaining_tokens", "capabilities", "model_info",
        }
    )  # fmt: skip


def test_round_robin_signature_has_the_knobs_we_use() -> None:
    import inspect

    params = inspect.signature(RoundRobinGroupChat.__init__).parameters
    assert {"participants", "termination_condition", "max_turns", "custom_message_types"} <= set(
        params
    )


def test_max_turns_two_yields_task_reviewer_editor(
    run: tuple[TaskResult, ScriptedClient, ScriptedClient],
) -> None:
    """P12: both agents speak; the editor's turn is a StructuredMessage[Verdict]."""
    result, _, _ = run
    shape = [(type(m), m.source) for m in result.messages]
    assert shape == [
        (TextMessage, "user"),
        (TextMessage, "Policy_Compliance_Reviewer"),
        (StructuredMessage[Verdict], "Final_Editor"),
    ]
    assert result.messages[-1].content == Verdict(**VERDICT)


def test_stop_reason_names_the_turn_limit(
    run: tuple[TaskResult, ScriptedClient, ScriptedClient],
) -> None:
    assert run[0].stop_reason == "Maximum number of turns 2 reached."


def test_editor_create_receives_the_verdict_type_as_json_output(
    run: tuple[TaskResult, ScriptedClient, ScriptedClient],
) -> None:
    _, reviewer_client, editor_client = run
    assert [c["json_output"] for c in reviewer_client.calls] == [None]
    assert [c["json_output"] for c in editor_client.calls] == [Verdict]


def test_editor_sees_the_reviewer_message_by_source(
    run: tuple[TaskResult, ScriptedClient, ScriptedClient],
) -> None:
    _, _, editor_client = run
    sources = [getattr(m, "source", None) for m in editor_client.calls[0]["messages"]]
    assert sources == [None, "user", "Policy_Compliance_Reviewer"]


def test_missing_custom_message_types_fails_the_run() -> None:
    """P11. Observed: the registry's ValueError surfaces from team.run() as a RuntimeError
    carrying the ValueError text (§ARCH 2.3.4 says ValueError — see docs/DECISIONS.md)."""
    team, _, _ = _team(with_custom_types=False)
    not_registered = r"ValueError: Message type .*StructuredMessage\[Verdict\].* is not registered"
    with pytest.raises(RuntimeError, match=not_registered):
        asyncio.run(team.run(task="t"))
