"""The Autogen review stage (§ARCH 7.8.2, 7.13): approve, revise, fallback, fail closed, budget,
and the test-only `inject_claim` hook staying out of every input model."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import pytest
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage
from pydantic import BaseModel

import credagent
from credagent.context import RequestContext
from credagent.llm.engine import REFUSAL_ANSWERS
from credagent.pipeline.orchestrator import AutogenReview, Orchestrator
from credagent.review import team
from credagent.review.team import ReviewContext, run_review
from credagent.schemas import FALLBACK_ANSWER
from tests.unit.test_engine_compose import found
from tests.unit.test_orchestrator import make

KB08 = (
    "Floating-rate home loans taken by individuals carry no prepayment penalty. Fixed-rate "
    "personal loans carry a prepayment charge of 2% of the principal outstanding."
)
CONTEXT = ReviewContext(question="Is there a prepayment penalty?", documents=(KB08,))
CLAIM = "Cred also offers a 0% interest rate on all personal loans for life."  # §ARCH 7.13


def ctx() -> RequestContext:
    return RequestContext.new(session_id="s", max_tokens=8000)


@pytest.mark.anyio
async def test_approve_path_returns_the_draft_byte_for_byte() -> None:
    outcome = await run_review(KB08, CONTEXT, ctx())
    assert outcome.error is None and outcome.verdict.approved
    assert outcome.verdict.final_answer.encode() == KB08.encode()


@pytest.mark.anyio
async def test_revise_path_removes_the_injected_claim_and_keeps_the_rest_in_order() -> None:
    outcome = await run_review(KB08, CONTEXT, ctx(), inject_claim=CLAIM)
    assert outcome.draft == f"{KB08} {CLAIM}"
    assert (outcome.error, outcome.verdict.approved) == (None, False)
    assert outcome.verdict.final_answer == KB08
    assert CLAIM in outcome.verdict.reason


@pytest.mark.anyio
async def test_every_sentence_unsupported_gives_the_fallback() -> None:
    outcome = await run_review("Cred waives every fee forever. Loans are free.", CONTEXT, ctx())
    assert (outcome.verdict.approved, outcome.verdict.final_answer) == (False, FALLBACK_ANSWER)


RECORD = found()  # LA-0007, Home, Under Review, ₹25,00,000, 12 days, score 0.24 < 0.5
STATUS = (
    "Application LA-0007 (Home Loan) is currently Under Review. The loan amount is ₹25,00,000 and "
    "the application was created 12 days ago. No escalation is needed (escalation score 0.24 < "
    "threshold 0.5)."
)
WRONG_STATUS = "Application LA-0007 (Home Loan) is currently Approved."
WRONG_AMOUNT = "The loan amount is ₹52,00,000 and the application was created 12 days ago."


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("draft", "wrong"),
    [
        (STATUS, None),
        (STATUS.replace("Under Review", "Approved"), WRONG_STATUS),
        (STATUS.replace("25,00,000", "52,00,000"), WRONG_AMOUNT),
    ],
    ids=["correct", "wrong-status", "wrong-amount"],
)
async def test_the_status_sentence_is_rebuilt_from_the_record_json_never_from_the_draft(
    draft: str, wrong: str | None
) -> None:
    """D-039: the reviewer builds the record's status sentences from the record JSON with the
    shared template; a draft sentence about the record must equal one of them."""
    context = AutogenReview().context_for("Status of LA-0007?", None, RECORD)
    assert context.documents == (RECORD.model_dump_json(exclude={"kind"}),)  # JSON only
    outcome = await run_review(draft, context, ctx())
    assert outcome.error is None
    assert outcome.verdict.approved is (wrong is None)
    if wrong is not None:
        assert wrong in outcome.verdict.reason
        assert wrong not in outcome.verdict.final_answer


class MalformedTeam:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages

    async def run(self, *, task: str) -> TaskResult:
        return TaskResult(messages=self.messages)


def _text(source: str) -> TextMessage:
    return TextMessage(source=source, content="x")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "messages",
    [
        [_text("user"), _text("Policy_Compliance_Reviewer")],  # the editor never spoke
        [_text("user"), _text("Policy_Compliance_Reviewer"), _text("Final_Editor")],  # not typed
    ],
    ids=["two-messages", "text-not-structured"],
)
async def test_malformed_team_output_fails_closed(
    messages: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(team, "build_review_team", lambda _ctx: MalformedTeam(messages))
    orch, _, sink, _ = make()
    r = await orch.handle("Is there a prepayment penalty on a home loan?", "s1")
    assert (r.refused, r.refusal_reason, r.answer) == (
        True, "agent_error", REFUSAL_ANSWERS["agent_error"],
    )  # fmt: skip
    assert r.review is not None and not r.review.approved
    record = sink.records[0]
    assert record.review_error is not None and record.review_error.startswith("ReviewShapeError")
    assert record.guard_reason is None  # nothing unreviewed reached the gate


@pytest.mark.anyio
async def test_autogen_calls_are_charged_to_the_budget_meter() -> None:
    c = ctx()
    await run_review(KB08, CONTEXT, c)
    tokens = c.budget.tokens_by_persona()
    assert c.budget.calls() == {"editor": 1, "reviewer": 1}
    assert tokens["editor"] > tokens["reviewer"] > 0  # the editor also reads the reviewer
    assert c.budget.used_tokens == sum(tokens.values())


def _models() -> list[type[BaseModel]]:
    for info in pkgutil.walk_packages(credagent.__path__, "credagent."):
        importlib.import_module(info.name)
    seen: list[type[BaseModel]] = []
    stack = list(BaseModel.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls.__module__.startswith("credagent."):
            seen.append(cls)
    return seen


def test_inject_claim_is_not_reachable_from_any_pipeline_or_api_input() -> None:
    assert "inject_claim" in inspect.signature(run_review).parameters  # the hook exists …
    models = _models()
    assert models  # … and the scan is not vacuous
    assert [m.__name__ for m in models if "inject_claim" in m.model_fields] == []
    for entry in (Orchestrator.handle, Orchestrator.__init__, AutogenReview.review,
                  AutogenReview.__init__):  # fmt: skip
        assert "inject_claim" not in inspect.signature(entry).parameters, entry
