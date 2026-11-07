"""PII masking (§ARCH 7.12.1): five rules in precedence order, typed tokens, counts only."""

from __future__ import annotations

import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

from credagent.guardrails.pii import RULES, TOKENS, MaskResult, PiiMasker

masker = PiiMasker()
PAN, AADHAAR, BANK = TOKENS["PAN"], TOKENS["AADHAAR"], TOKENS["BANK_ACCOUNT"]


def _counts(pan: int = 0, aadhaar: int = 0, bank: int = 0) -> dict[str, int]:
    return {"PAN": pan, "AADHAAR": aadhaar, "BANK_ACCOUNT": bank}


# ------------------------------------------------------------------ positives
@pytest.mark.parametrize(
    ("raw", "expected", "counts"),
    [
        ("My PAN is ABCPE1234F.", f"My PAN is {PAN}.", _counts(pan=1)),
        ("pan abcpe1234f please", f"pan {PAN} please", _counts(pan=1)),
        ("PAN: ABCPE 1234 F", f"PAN: {PAN}", _counts(pan=1)),
        ("PAN ABCPE-1234-F ok", f"PAN {PAN} ok", _counts(pan=1)),
        ("id_ABCPE1234F_x", f"id_{PAN}_x", _counts(pan=1)),  # underscores do not hide it
        ("PAN ABC\u200bPE1234F", f"PAN {PAN}", _counts(pan=1)),  # zero-width inside
        ("PAN ＡＢＣＰＥ１２３４Ｆ", f"PAN {PAN}", _counts(pan=1)),  # full-width (NFKC)
        ("Aadhaar 2345 6789 0123", f"Aadhaar {AADHAAR}", _counts(aadhaar=1)),
        ("Aadhaar 2345-6789-0123.", f"Aadhaar {AADHAAR}.", _counts(aadhaar=1)),
        ("Aadhaar 2345 6789 0123", f"Aadhaar {AADHAAR}", _counts(aadhaar=1)),
        ("uid 234567890123", f"uid {AADHAAR}", _counts(aadhaar=1)),
        ("account number 123456789", f"account number {BANK}", _counts(bank=1)),
        ("a/c no. 12345678901", f"a/c no. {BANK}", _counts(bank=1)),
        ("Acct: 123456789012345678", f"Acct: {BANK}", _counts(bank=1)),
        ("A/C#000123456789", f"A/C#{BANK}", _counts(bank=1)),
        ("call me on 9876543210", f"call me on {BANK}", _counts(bank=1)),  # phones over-masked
        ("ref 1234567890123456789012", f"ref {BANK}", _counts(bank=1)),  # > 18 digits too
        (
            "PAN ABCPE1234F, Aadhaar 2345 6789 0123, a/c 123456789012",
            f"PAN {PAN}, Aadhaar {AADHAAR}, a/c {BANK}",
            _counts(pan=1, aadhaar=1, bank=1),
        ),
        ("two: 234567890123 and 345678901234", f"two: {AADHAAR} and {AADHAAR}", _counts(aadhaar=2)),
    ],
)
def test_masks_each_pii_type_with_its_token(
    raw: str, expected: str, counts: dict[str, int]
) -> None:
    result = masker.mask(raw)
    assert result.text == expected
    assert result.counts_by_type == counts


# ------------------------------------------------------------------ negatives
@pytest.mark.parametrize(
    "raw",
    [
        "What is the status of LA-0007?",
        "A fee of ₹25,000 applies.",
        "The rate is 10.5% p.a.",
        "Created on 2024-11-07 and 07/11/2024.",
        "A loan of 12345678 rupees",  # 8 digits: amounts never reach 9 (§7.1)
        "₹2,50,00,000 is the cap",
        "ABCDE12345 is not a PAN",  # 5 letters + 5 digits
        "ABCXE1234F has an invalid 4th letter",
        "1234 5678 9012 starts with 1",  # Aadhaar never starts with 0 or 1
        "small 5000 a month",  # the separated PAN form needs capitals
        "EMI of ₹10,000 for 24 months at 9.5%",
    ],
)
def test_leaves_non_pii_alone(raw: str) -> None:
    result = masker.mask(raw)
    assert result.text == raw
    assert result.counts_by_type == _counts()


# ------------------------------------------------------------------ precedence
@pytest.mark.parametrize(
    ("raw", "expected", "counts"),
    [
        # rule 3 (keyword-anchored account) beats rule 4 (contiguous Aadhaar)
        ("account 234567890123", f"account {BANK}", _counts(bank=1)),
        ("my account number: 234567890123", f"my account number: {BANK}", _counts(bank=1)),
        # grouped 12 digits are Aadhaar (rule 2) even after an account keyword
        ("account 2345 6789 0123", f"account {AADHAAR}", _counts(aadhaar=1)),
        # bare 12 digits starting 2–9: Aadhaar (rule 4) before bare account (rule 5)
        ("234567890123", AADHAAR, _counts(aadhaar=1)),
        # bare 12 digits starting with 1: not Aadhaar, so rule 5
        ("123456789012", BANK, _counts(bank=1)),
    ],
)
def test_precedence(raw: str, expected: str, counts: dict[str, int]) -> None:
    result = masker.mask(raw)
    assert (result.text, result.counts_by_type) == (expected, counts)


def test_rules_are_in_the_documented_order() -> None:
    assert [(r.name, r.pii_type) for r in RULES] == [
        ("pan", "PAN"),
        ("aadhaar_grouped", "AADHAAR"),
        ("bank_account_keyword", "BANK_ACCOUNT"),
        ("aadhaar_contiguous", "AADHAAR"),
        ("bank_account_bare", "BANK_ACCOUNT"),
    ]


def test_mask_result_carries_counts_never_values() -> None:
    assert set(MaskResult.model_fields) == {"text", "counts_by_type"}
    result = masker.mask("PAN ABCPE1234F")
    assert "ABCPE1234F" not in result.model_dump_json()
    assert result.total == 1


def test_masking_is_idempotent_and_tokens_are_never_rematched() -> None:
    once = masker.mask("PAN ABCPE1234F, a/c 123456789012, 2345 6789 0123")
    twice = masker.mask(once.text)
    assert twice.text == once.text
    assert twice.counts_by_type == _counts()


def test_mask_result_is_frozen() -> None:
    result = masker.mask("x")
    with pytest.raises(ValueError, match="frozen"):
        result.text = "y"  # type: ignore[misc]


# ------------------------------------------------------------------ property: raw never survives
_PAN_4TH = "ABCFGHLJPT"
_upper = st.sampled_from(string.ascii_uppercase)
pans = st.tuples(
    st.text(string.ascii_uppercase, min_size=3, max_size=3),
    st.sampled_from(_PAN_4TH),
    _upper,
    st.text(string.digits, min_size=4, max_size=4),
    _upper,
).map("".join)
aadhaar_digits = st.tuples(
    st.sampled_from("23456789"), st.text(string.digits, min_size=11, max_size=11)
).map("".join)
aadhaar_values = st.one_of(
    aadhaar_digits,
    st.tuples(aadhaar_digits, st.sampled_from([" ", "-"])).map(
        lambda t: t[1].join([t[0][:4], t[0][4:8], t[0][8:]])
    ),
)
accounts = st.text(string.digits, min_size=9, max_size=18)
account_mentions = st.tuples(
    st.sampled_from(["account", "a/c", "A/C no.", "acct #", "Account number:", ""]), accounts
)
# Filler is realistic prose: words, short numbers and punctuation, joined by separators.
words = st.one_of(
    st.text(string.ascii_letters, min_size=1, max_size=9),
    st.text(string.digits, min_size=1, max_size=4),
    st.sampled_from(["₹25,000", "10.5%", "LA-0007", "p.a.", "—", "(", ")", ",", "."]),
)
filler = st.lists(words, max_size=8).map(" ".join)
separators = st.sampled_from([" ", "\n", ", ", ": ", " (", ") ", " - ", "_"])


@given(filler, separators, st.tuples(pans, st.booleans()), separators, filler)
def test_property_pan_never_survives(
    pre: str, s1: str, pan_case: tuple[str, bool], s2: str, post: str
) -> None:
    pan, lower = pan_case
    value = pan.lower() if lower else pan
    masked = masker.mask(f"{pre}{s1}{value}{s2}{post}").text
    assert value not in masked
    assert value.upper() not in masked.upper()


@given(filler, separators, aadhaar_values, separators, filler)
def test_property_aadhaar_never_survives(pre: str, s1: str, value: str, s2: str, post: str) -> None:
    masked = masker.mask(f"{pre}{s1}{value}{s2}{post}").text
    assert value not in masked


@given(filler, separators, account_mentions, separators, filler)
def test_property_bank_account_never_survives(
    pre: str, s1: str, mention: tuple[str, str], s2: str, post: str
) -> None:
    keyword, number = mention
    masked = masker.mask(f"{pre}{s1}{keyword} {number}{s2}{post}").text
    assert number not in masked
