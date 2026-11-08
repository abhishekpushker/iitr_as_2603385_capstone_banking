"""`MockEngine.resolve_followup` (§ARCH 7.10): follow-up state comes only from history messages."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from credagent.llm.engine import MockEngine

ENGINE = MockEngine()
T1 = "What is the status of application LA-0007?"
T2 = "What are the eligibility criteria for it?"
T3 = "And does it carry a prepayment penalty?"


def ai(record_id: Any = None, category: Any = None, text: str = "answer") -> AIMessage:
    return AIMessage(content=text, additional_kwargs={"record_id": record_id, "category": category})


AFTER_T1: list[BaseMessage] = [HumanMessage(content=T1), ai("LA-0007", "Home")]


def test_no_reference_passes_the_query_through() -> None:
    r = ENGINE.resolve_followup(T1, [])
    assert (r.query, r.record_id, r.category, r.resolved, r.clarify) == (
        T1, "LA-0007", None, False, False,
    )  # fmt: skip


def test_a_named_loan_category_is_recorded() -> None:
    r = ENGINE.resolve_followup("Is there a prepayment penalty on a home loan?", [])
    assert (r.category, r.resolved) == ("Home", False)


def test_fresh_session_reference_asks_for_clarification() -> None:
    r = ENGINE.resolve_followup(T2, [])
    assert r.clarify and not r.resolved
    assert r.record_id is None and r.category is None


def test_turn_two_resolves_the_pronoun_to_the_loan_category() -> None:
    r = ENGINE.resolve_followup(T2, AFTER_T1)
    assert r.query == "What are the eligibility criteria for a Home loan?"
    assert (r.record_id, r.category, r.resolved, r.clarify) == (None, "Home", True, False)


def test_turn_three_uses_the_most_recent_category_even_without_a_record_id() -> None:
    history = [*AFTER_T1, HumanMessage(content="…"), ai(None, "Home")]
    r = ENGINE.resolve_followup(T3, history)
    assert r.query == "And does a Home loan carry a prepayment penalty?"
    assert r.category == "Home"


@pytest.mark.parametrize(
    ("question", "standalone"),
    [
        ("Has it been approved?", "Has application LA-0007 been approved?"),
        ("What is its status?", "What is application LA-0007's status?"),
        ("Any update on my application?", "Any update on application LA-0007?"),
        ("What is the status of the same one?", "What is the status of application LA-0007?"),
    ],
)
def test_status_follow_ups_carry_the_record_id(question: str, standalone: str) -> None:
    r = ENGINE.resolve_followup(question, AFTER_T1)
    assert (r.query, r.record_id, r.resolved) == (standalone, "LA-0007", True)


def test_status_follow_up_with_only_a_category_in_history_asks_for_the_id() -> None:
    r = ENGINE.resolve_followup(
        "Has it been approved?", [HumanMessage(content="x"), ai(None, "Home")]
    )
    assert r.clarify


def test_the_most_recent_record_wins() -> None:
    history = [*AFTER_T1, HumanMessage(content="x"), ai("LA-0012", "Auto")]
    r = ENGINE.resolve_followup("Does it carry a prepayment penalty?", history)
    assert r.query == "Does an Auto loan carry a prepayment penalty?"
    assert ENGINE.resolve_followup("Has it been approved?", history).record_id == "LA-0012"


def test_state_is_read_only_from_ai_messages_with_valid_values() -> None:
    history: list[BaseMessage] = [
        HumanMessage(content="x", additional_kwargs={"record_id": "LA-0001", "category": "Home"}),
        ai("not-an-id", "Gold"),
        ai(42, ["Home"]),
    ]
    assert ENGINE.resolve_followup("Has it been approved?", history).clarify
    assert ENGINE.resolve_followup(T2, history).clarify


def test_an_explicit_record_id_beats_history() -> None:
    r = ENGINE.resolve_followup("Has LA-0009 been approved, and what is its status?", AFTER_T1)
    assert (r.record_id, r.resolved) == ("LA-0009", False)


def test_expletive_it_is_not_treated_as_a_reference() -> None:
    """Documented limit (D-036): "is it" is never a reference, so a fresh-session policy question
    like this one is not turned into a clarification."""
    r = ENGINE.resolve_followup("Is it possible to close a joint account online?", [])
    assert not r.clarify and not r.resolved


@pytest.mark.parametrize("history", [[], AFTER_T1], ids=["fresh", "after-t1"])
@pytest.mark.parametrize(
    "question",
    [
        "What fee applies if I close my account in its first year?",
        "Is the annual fee on the card waived in its first year?",
    ],
)
def test_its_with_an_antecedent_in_the_same_sentence_is_not_a_reference(
    question: str, history: list[BaseMessage]
) -> None:
    """BUG-003: "its" bound to a noun phrase earlier in the same sentence ("my account … its
    first year") refers to that phrase, not to an earlier turn: no clarification, no rewrite."""
    r = ENGINE.resolve_followup(question, history)
    assert (r.query, r.resolved, r.clarify) == (question, False, False)


@pytest.mark.parametrize("history", [[], AFTER_T1], ids=["fresh", "after-t1"])
@pytest.mark.parametrize(
    "question",
    [
        "Which documents does the bank need, and can it accept a passport?",
        "What does the closure fee cost and can it be refunded?",
        "How long does it take to get a replacement debit card?",
        "How much does it cost to issue a duplicate statement?",
    ],
)
def test_it_bound_in_its_own_sentence_or_extraposed_is_not_a_reference(
    question: str, history: list[BaseMessage]
) -> None:
    """BUG-006: "it" in a later clause than a noun phrase of the same sentence refers to that
    phrase; "does it take/cost … to <verb>" is an expletive. Neither asks for clarification."""
    r = ENGINE.resolve_followup(question, history)
    assert (r.query, r.resolved, r.clarify) == (question, False, False)


def test_it_without_a_clause_break_or_infinitive_is_still_a_reference() -> None:
    """BUG-006 guard: turn 2/3 stay references; "does it cost" without a to-clause too."""
    assert ENGINE.resolve_followup(T2, []).clarify
    assert ENGINE.resolve_followup(T3, AFTER_T1).resolved
    r = ENGINE.resolve_followup("Does it cost anything extra?", AFTER_T1)
    assert r.query == "Does a Home loan cost anything extra?"


def test_its_after_a_sentence_boundary_is_still_a_reference() -> None:
    """The antecedent search stops at the sentence boundary."""
    r = ENGINE.resolve_followup("Thanks for the answer. What is its status?", AFTER_T1)
    assert r.resolved and not r.clarify and "its" not in r.query


def test_a_reference_at_the_start_is_capitalised() -> None:
    r = ENGINE.resolve_followup("Its prepayment penalty?", AFTER_T1)
    assert r.query == "A Home loan's prepayment penalty?"


def test_resolution_is_pure() -> None:
    history = list(AFTER_T1)
    first = ENGINE.resolve_followup(T2, history)
    assert ENGINE.resolve_followup(T2, history) == first
    assert history == AFTER_T1
