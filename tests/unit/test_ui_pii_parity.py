"""The chat UI keeps conversation history in browser storage, so it masks the user's text with a
JavaScript port of `PiiMasker` before storing it (D-058). This test pins the shared case list
(`ui/tests/pii_cases.json`) to the server's masker; `ui/tests/pii.test.js` pins the same list to
the browser's `maskPii`. A rule change on either side fails one of the two."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from credagent.guardrails.pii import TOKENS, PiiMasker

CASES_PATH = Path(__file__).resolve().parents[2] / "ui" / "tests" / "pii_cases.json"
CASES: list[dict[str, Any]] = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_the_server_masker_produces_the_shared_expectation(case: dict[str, Any]) -> None:
    assert PiiMasker().mask(case["input"]).text == case["masked"]


def test_the_case_list_covers_every_token_type() -> None:
    masked = " ".join(c["masked"] for c in CASES)
    assert all(token in masked for token in TOKENS.values())


def test_the_ui_uses_the_server_tokens() -> None:
    js = (CASES_PATH.parents[1] / "public" / "js" / "pii.js").read_text(encoding="utf-8")
    assert all(json.dumps(token) in js for token in TOKENS.values())
