"""The Autogen review stage (§ARCH 7.13, Task 14) — pipeline stage 7.

`build_review_team(ctx)` is a new `RoundRobinGroupChat([Policy_Compliance_Reviewer, Final_Editor],
max_turns=2, custom_message_types=[StructuredMessage[ReviewVerdict]])` per request (no shared
state), each agent with its own `MockAutogenClient` bound to the request's budget meter. Without
`custom_message_types` the run raises (P11); `max_turns=2`, not a message-count termination,
makes both agents speak (P12).

`run_review(draft, context, ctx, inject_claim=None)` renders the task
(`<<<QUESTION>>>…<<<DRAFT>>>…<<<CONTEXT>>>…`), runs the team and checks the shape: exactly three
messages from `user`, the reviewer and the editor, the last a `StructuredMessage[ReviewVerdict]`
whose content re-validates. Any deviation, or any error during the run, fails closed: an
unapproved verdict carrying the `agent_error` refusal text, with `error` set (logged as
`review_error`).

`inject_claim` is a test-only hook (transcript 21, tests): it appends one sentence to the draft
before review. It exists only on this function; no pipeline or API input carries it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from pydantic import ValidationError

from credagent.context import RequestContext
from credagent.llm.autogen_client import (
    CONTEXT_ITEM,
    CONTEXT_OPEN,
    DRAFT_OPEN,
    EDITOR_NAME,
    QUESTION_OPEN,
    REVIEWER_NAME,
    TASK_SOURCE,
    MockAutogenClient,
)
from credagent.llm.engine import REFUSAL_ANSWERS
from credagent.schemas import ReviewVerdict

MAX_TURNS: Final = 2
EXPECTED_SOURCES: Final = (TASK_SOURCE, REVIEWER_NAME, EDITOR_NAME)
REVIEWER_SYSTEM: Final = (
    "You are Cred's policy compliance reviewer. Check every sentence of the DRAFT answer against "
    "the CONTEXT (retrieved policy text and the application record). A sentence is supported only "
    "if the context states it and every number in it appears in the context. List each sentence "
    'as supported or UNSUPPORTED, then end with one line: REVIEW_JSON: {"unsupported": [...], '
    '"supported_count": n, "total": m}.'
)
EDITOR_SYSTEM: Final = (
    "You are Cred's final editor. Read only the reviewer's report. If nothing is unsupported, "
    "approve the draft and return it unchanged. Otherwise remove the unsupported sentences, keep "
    "the rest in order, and do not add anything; if nothing remains, return the safe fallback. "
    "Reply with a ReviewVerdict JSON object: approved, final_answer, reason."
)


@dataclass(frozen=True)
class ReviewContext:
    """What the reviewer checks the draft against: the (masked, standalone) question and the
    context documents (retrieved chunk texts, the lookup record)."""

    question: str
    documents: tuple[str, ...]


@dataclass(frozen=True)
class ReviewOutcome:
    verdict: ReviewVerdict
    draft: str  # the draft that was reviewed (with the injected claim, if any)
    messages: tuple[tuple[str, str, str], ...]  # (source, message type, content) per message
    error: str | None  # set when the review failed closed


class ReviewShapeError(ValueError):
    pass


def build_review_team(ctx: RequestContext) -> RoundRobinGroupChat:
    reviewer = AssistantAgent(
        REVIEWER_NAME, model_client=MockAutogenClient("reviewer", ctx),
        system_message=REVIEWER_SYSTEM,
    )  # fmt: skip
    editor = AssistantAgent(
        EDITOR_NAME, model_client=MockAutogenClient("editor", ctx),
        system_message=EDITOR_SYSTEM, output_content_type=ReviewVerdict,
    )  # fmt: skip
    return RoundRobinGroupChat(
        [reviewer, editor], max_turns=MAX_TURNS,
        custom_message_types=[StructuredMessage[ReviewVerdict]],
    )  # fmt: skip


def render_task(question: str, draft: str, documents: Sequence[str]) -> str:
    items = f"\n{CONTEXT_ITEM}\n".join(documents)
    return f"{QUESTION_OPEN}\n{question}\n{DRAFT_OPEN}\n{draft}\n{CONTEXT_OPEN}\n{items}"


def verdict_from(result: TaskResult) -> ReviewVerdict:
    """The editor's verdict, if the run has exactly the expected shape (§ARCH 7.13)."""
    messages = result.messages
    sources = tuple(getattr(m, "source", None) for m in messages)
    if sources != EXPECTED_SOURCES:
        raise ReviewShapeError(f"expected messages from {EXPECTED_SOURCES}, got {sources}")
    last = messages[-1]
    if not isinstance(last, StructuredMessage) or not isinstance(last.content, ReviewVerdict):
        raise ReviewShapeError(f"the editor's message is a {type(last).__name__}")
    try:
        return ReviewVerdict.model_validate(last.content.model_dump())
    except ValidationError as exc:
        raise ReviewShapeError("the editor's verdict does not re-validate") from exc


def _transcript(result: TaskResult) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (getattr(m, "source", "?"), type(m).__name__, m.to_text()) for m in result.messages
    )


async def run_review(
    draft: str, context: ReviewContext, ctx: RequestContext, inject_claim: str | None = None
) -> ReviewOutcome:
    if inject_claim is not None:
        draft = f"{draft} {inject_claim}"
    task = render_task(context.question, draft, context.documents)
    result: TaskResult | None = None
    try:
        result = await build_review_team(ctx).run(task=task)
        verdict = verdict_from(result)
    except Exception as exc:  # noqa: BLE001  (fail closed on anything, §ARCH 7.13)
        failed = ReviewVerdict(
            approved=False, final_answer=REFUSAL_ANSWERS["agent_error"],
            reason=f"review_error: {type(exc).__name__}",
        )  # fmt: skip
        messages = () if result is None else _transcript(result)
        return ReviewOutcome(failed, draft, messages, error=f"{type(exc).__name__}: {exc}")
    return ReviewOutcome(verdict, draft, _transcript(result), error=None)
