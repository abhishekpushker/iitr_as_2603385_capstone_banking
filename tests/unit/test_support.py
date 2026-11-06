"""Support rule (§ARCH 7.8): a sentence S is supported by context C iff token_recall(S, best
context sentence) ≥ 0.6 and every number in S appears among the numbers of C."""

from __future__ import annotations

import pytest

from credagent.llm.engine import SUPPORT_RECALL, MockEngine, token_recall

CONTEXT = [
    "Floating-rate home loans taken by individuals carry no prepayment penalty. Fixed-rate "
    "personal loans charge a prepayment penalty of 4% of the prepaid principal during the "
    "first 12 EMIs and 2% after that.",
    "Personal loan interest ranges from 10.5% to 24% p.a. by credit score band.",
]

engine = MockEngine()


def test_threshold_is_the_architecture_value() -> None:
    assert SUPPORT_RECALL == 0.6


def test_verbatim_context_sentence_is_supported() -> None:
    sentence = "Floating-rate home loans taken by individuals carry no prepayment penalty."
    report = engine.support_report(sentence, CONTEXT)
    assert report.total == 1 and report.supported_count == 1 and report.ratio == 1.0
    (s,) = report.sentences
    assert s.supported and s.recall == 1.0 and s.numbers_missing == ()
    assert s.best_match == sentence


def test_fabricated_zero_percent_interest_is_unsupported() -> None:
    report = engine.support_report("Personal loans have 0% interest for life.", CONTEXT)
    (s,) = report.sentences
    assert not s.supported
    assert s.numbers_missing == ("0%",)


def test_high_recall_with_a_changed_number_is_unsupported() -> None:
    """Rule (b) on its own: recall clears 0.6 but the number 3% is not in the context."""
    sentence = (
        "Fixed-rate personal loans charge a prepayment penalty of 3% of the prepaid principal "
        "during the first 12 EMIs."
    )
    (s,) = engine.support_report(sentence, CONTEXT).sentences
    assert s.recall >= SUPPORT_RECALL
    assert s.numbers_missing == ("3%",)
    assert not s.supported


def test_paraphrase_with_the_same_numbers_is_supported_when_recall_is_high() -> None:
    sentence = "During the first 12 EMIs, fixed-rate personal loans charge a 4% prepayment penalty."
    (s,) = engine.support_report(sentence, CONTEXT).sentences
    assert s.recall >= SUPPORT_RECALL and s.numbers_missing == ()
    assert s.supported


def test_paraphrase_with_low_recall_is_unsupported() -> None:
    (s,) = engine.support_report("Borrowers pay 4% if they settle early.", CONTEXT).sentences
    assert s.recall < SUPPORT_RECALL
    assert not s.supported


def test_numbers_may_come_from_any_context_sentence() -> None:
    """Rule (b) is over the numbers of the whole context C, not just the best match."""
    sentence = "Personal loan interest ranges from 10.5% to 24% p.a., and the penalty is 4%."
    (s,) = engine.support_report(sentence, CONTEXT).sentences
    assert s.numbers_missing == ()


def test_numbers_are_compared_canonically() -> None:
    ctx = ["The annual fee is waived when card spend reaches ₹1.5 lakh."]
    (s,) = engine.support_report("The annual fee is waived at a spend of ₹150,000.", ctx).sentences
    assert s.numbers_missing == ()


def test_report_counts_each_sentence() -> None:
    answer = (
        "Floating-rate home loans taken by individuals carry no prepayment penalty. "
        "Personal loans have 0% interest for life."
    )
    report = engine.support_report(answer, CONTEXT)
    assert (report.supported_count, report.total) == (1, 2)
    assert report.ratio == 0.5
    assert not report.all_supported


def test_empty_answer_has_no_claims() -> None:
    report = engine.support_report("", CONTEXT)
    assert report.total == 0 and report.ratio == 1.0 and report.all_supported


def test_empty_context_supports_nothing_with_content() -> None:
    (s,) = engine.support_report("Prepayment is free.", []).sentences
    assert s.best_match is None and s.recall == 0.0 and not s.supported


@pytest.mark.parametrize(
    ("sentence", "reference", "expected"),
    [
        ("prepayment penalty applies", "a prepayment penalty applies here", 1.0),
        ("prepayment penalty applies", "penalty", 1 / 3),
        ("the and of", "anything", 1.0),  # no content tokens → no claim to check
        ("no prepayment penalty", "prepayment penalty", 2 / 3),  # negation is a token
    ],
)
def test_token_recall(sentence: str, reference: str, expected: float) -> None:
    assert token_recall(sentence, reference) == pytest.approx(expected)
