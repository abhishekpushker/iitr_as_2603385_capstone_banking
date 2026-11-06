"""text.py primitives (§ARCH 7.3, 7.8): normalisation, tokens, numbers, sentence splitter (P15)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from credagent import text

# ------------------------------------------------------------------ normalisation


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello World", "hello world"),
        ("STRASSE Straße", "strasse strasse"),  # casefold, not lower
        ("ｆｕｌｌ　ｗｉｄｔｈ", "full width"),  # NFKC folds full-width forms and spaces
        ("zero\u200bwidth‌‍⁠﻿join", "zerowidthjoin"),
        ("soft­hyphen", "softhyphen"),
        ("  many \n\t spaces  ", "many spaces"),
        ("₹10,000 at 10.5%", "₹10,000 at 10.5%"),
        ("", ""),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert text.normalize(raw) == expected


def test_clean_keeps_case_but_applies_nfkc_and_strips_zero_width() -> None:
    assert text.clean("Ｐａｎ\u200b Card  No.") == "Pan Card No."


@given(st.text())
def test_normalize_is_idempotent(s: str) -> None:
    once = text.normalize(s)
    assert text.normalize(once) == once


# ------------------------------------------------------------------ tokens


def test_content_tokens_keep_numbers_percent_and_rupee_amounts() -> None:
    tokens = text.content_tokens(
        "Personal loans cost 10.5% p.a. and the fee is ₹10,000 for 24 months"
    )
    assert "10.5%" in tokens
    assert "₹10,000" in tokens
    assert "24" in tokens
    assert {"personal", "loans", "cost", "fee", "months"} <= set(tokens)


def test_content_tokens_drop_stopwords_and_single_letters() -> None:
    tokens = text.content_tokens("Is there a penalty for the loan? It is p.a. based")
    assert tokens == ["penalty", "loan", "based"]


def test_content_tokens_keep_negations() -> None:
    """'no prepayment penalty' must not look like 'prepayment penalty' to the support rule."""
    assert text.content_tokens("There is no penalty, not ever") == ["no", "penalty", "not", "ever"]


def test_content_tokens_are_normalised() -> None:
    assert text.content_tokens("PREPAYMENT Penalty\u200b") == ["prepayment", "penalty"]


def test_content_tokens_keep_hyphenated_words() -> None:
    assert "either-or-survivor" in text.content_tokens("Either-or-survivor mode")


# ------------------------------------------------------------------ numbers


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("₹1.5 lakh", ["150000"]),
        ("Rs 1.5 lakhs", ["150000"]),
        ("1.5 lac", ["150000"]),
        ("₹40 lakh", ["4000000"]),
        ("₹2 crore", ["20000000"]),
        ("₹1.25 crores", ["12500000"]),
        ("₹10,000", ["10000"]),
        ("10,000", ["10000"]),
        ("₹10,00,000", ["1000000"]),  # Indian digit grouping
        ("Rs. 500", ["500"]),
        ("a fee of ₹500.", ["500"]),  # sentence period is not a decimal point
        ("2%", ["2%"]),
        ("3.5 %", ["3.5%"]),
        ("10.5%", ["10.5%"]),
        ("10.50%", ["10.5%"]),
        ("1.0", ["1"]),
        ("12.5% p.a.", ["12.5%"]),
        ("10.5–12% for 750–799", ["10.5", "12%", "750", "799"]),
        ("every 2/8/10 years", ["2", "8", "10"]),
        ("24 months and 7 days", ["24", "7"]),
        ("the 5th of the month", ["5"]),
        ("no numbers here", []),
    ],
)
def test_extract_numbers_canonicalises(raw: str, expected: list[str]) -> None:
    assert text.extract_numbers(raw) == expected


def test_extract_numbers_equates_lakh_and_digit_forms() -> None:
    assert text.extract_numbers("₹1.5 lakh") == text.extract_numbers("₹1,50,000")


# ------------------------------------------------------------------ sentence splitter (P15)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "Personal loans cost 10.5% p.a. for top scores. Fees apply.",
            ["Personal loans cost 10.5% p.a. for top scores.", "Fees apply."],
        ),
        (
            "Rates start at 10.5% p.a. Fees apply.",
            ["Rates start at 10.5% p.a.", "Fees apply."],
        ),
        ("The rate is 10.5% p.a.", ["The rate is 10.5% p.a."]),
        (
            "The fee is waived on ₹1.5 lakh of spend. Ask us.",
            ["The fee is waived on ₹1.5 lakh of spend.", "Ask us."],
        ),
        (
            "Submit one proof, e.g. Aadhaar or a passport. Video KYC works.",
            ["Submit one proof, e.g. Aadhaar or a passport.", "Video KYC works."],
        ),
        (
            "Use one ID, i.e. Your PAN card. Done.",
            ["Use one ID, i.e. Your PAN card.", "Done."],
        ),
        (
            "The fee is Rs. 500 per year. It is waived.",
            ["The fee is Rs. 500 per year.", "It is waived."],
        ),
        (
            "Quote account No. 12345 in the form. Thanks.",
            ["Quote account No. 12345 in the form.", "Thanks."],
        ),
        ("Wait... then apply. Next step.", ["Wait... then apply.", "Next step."]),
        ("It depends… Ask us later.", ["It depends…", "Ask us later."]),
        ("It depends... Ask us later.", ["It depends...", "Ask us later."]),
        ("No trailing period here", ["No trailing period here"]),
        ("First one. Second without period", ["First one.", "Second without period"]),
        ("Is it free? Yes! Really.", ["Is it free?", "Yes!", "Really."]),
        ('He said "Done." Then he left.', ['He said "Done."', "Then he left."]),
        ("The rate is 10.5%. The fee is 2.5%.", ["The rate is 10.5%.", "The fee is 2.5%."]),
        ("Version 1.5 is out.", ["Version 1.5 is out."]),
        ("Pay ₹500. 3 documents are needed.", ["Pay ₹500.", "3 documents are needed."]),
        ("One.\n\nTwo.", ["One.", "Two."]),
        ("Heading line\n\nBody text.", ["Heading line", "Body text."]),
        ("", []),
        ("   \n ", []),
    ],
)
def test_split_sentences(raw: str, expected: list[str]) -> None:
    assert text.split_sentences(raw) == expected


def test_abbreviation_list_covers_the_documented_cases() -> None:
    assert {"p.a", "e.g", "i.e", "rs", "no"} <= text.ABBREVIATIONS
    assert text.SENTENCE_FINAL_ABBREVIATIONS <= text.ABBREVIATIONS


def test_sentence_spans_slice_back_to_the_sentences() -> None:
    raw = "  Rates are 10.5% p.a. Fees apply.  Done"
    spans = text.sentence_spans(raw)
    assert [raw[a:b] for a, b in spans] == text.split_sentences(raw)
    assert spans[0][0] == 2


_WORDS = st.sampled_from(
    ["loan", "Fee", "p.a.", "e.g.", "Rs.", "10.5%", "₹1.5", "lakh", "No.", "end.", "Why?",
     "Yes!", "wait...", "…", "OK", "i.e.", "3", "x"]
)  # fmt: skip


@given(st.lists(st.tuples(_WORDS, st.sampled_from([" ", "  ", "\n", "\n\n"])), max_size=40))
def test_sentence_spans_partition_the_non_whitespace_text(parts: list[tuple[str, str]]) -> None:
    raw = "".join(w + sep for w, sep in parts)
    spans = text.sentence_spans(raw)
    previous_end = 0
    for start, end in spans:
        piece = raw[start:end]
        assert piece and piece == piece.strip()
        assert start >= previous_end
        assert raw[previous_end:start].strip() == ""  # nothing but whitespace is dropped
        previous_end = end
    assert raw[previous_end:].strip() == ""
