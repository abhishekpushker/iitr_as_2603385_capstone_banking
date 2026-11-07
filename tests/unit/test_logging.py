"""Structured logging (§ARCH 7.17): JSON lines, a PII scrub filter on both handlers, the
`LogEntry` schema and the 2,000-char text cap."""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from credagent.guardrails.pii import PiiMasker
from credagent.observability.logging import (
    TEXT_LIMIT,
    JsonLinesFormatter,
    LogEntry,
    PiiScrubFilter,
    configure_logging,
    truncate,
)

MASKER = PiiMasker()
PAN, AADHAAR = "ABCPE1234F", "2345 6789 0123"


def scrub(text: str) -> str:
    return MASKER.mask(text).text


def _fields(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "trace_id": "0f0f0f0f-0000", "session_id": "session-1", "transport": "http",
        "method": "POST", "path": "/v1/ask", "status_code": 200, "outcome": "answered",
        "request_text_masked": "hello", "truncated": False, "pii_masked": {"PAN": 0},
        "injection_score": 0.0, "intent": "policy", "cache_hit": False, "tools_invoked": [],
        "review_approved": True, "tokens_estimated": 10, "cost_estimated_usd": 0.00002,
        "llm_usage": [[8, 2]], "history_length": 2, "latency_ms": 1.5,
        "timings_ms": {"total": 1.5}, "error_code": None,
    }  # fmt: skip
    return {**base, **over}


def test_formatter_writes_one_json_object_with_utc_millisecond_ts() -> None:
    record = logging.LogRecord("credagent", logging.INFO, __file__, 1, "startup", None, None)
    record.fields = {"pid": 1, "note": "ünïcode"}
    line = JsonLinesFormatter().format(record)
    assert "\n" not in line and "ünïcode" in line  # ensure_ascii=False
    data = json.loads(line)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", data["ts"])
    assert (data["level"], data["event"], data["pid"]) == ("INFO", "startup", 1)


def test_scrub_filter_masks_the_message_and_every_nested_string() -> None:
    record = logging.LogRecord("credagent", logging.INFO, __file__, 1, f"m {PAN}", None, None)
    record.fields = {"a": f"x {AADHAAR}", "b": [{"c": PAN}, 3], "n": 7}
    assert PiiScrubFilter(scrub).filter(record) is True
    line = JsonLinesFormatter().format(record)
    assert PAN not in line and AADHAAR not in line
    assert json.loads(line)["n"] == 7


def test_truncate_caps_the_masked_text() -> None:
    assert truncate("short") == ("short", False)
    text, cut = truncate("x" * (TEXT_LIMIT + 5))
    assert (len(text), cut) == (TEXT_LIMIT, True)


def test_configure_writes_scrubbed_lines_to_file_and_stream(tmp_path: Path) -> None:
    stream = io.StringIO()
    log = configure_logging(tmp_path / "logs" / "requests.jsonl", scrub, stream=stream)
    handlers = logging.getLogger("credagent").handlers
    assert len(handlers) == 2
    assert all(any(isinstance(f, PiiScrubFilter) for f in h.filters) for h in handlers)
    log.request(_fields(request_text_masked=f"pan {PAN}"))
    log.event("ws_disconnect", session_id="s1", detail=AADHAAR)
    lines = (tmp_path / "logs" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines == stream.getvalue().splitlines() and len(lines) == 2
    entry = LogEntry.model_validate_json(lines[0])
    assert entry.event == "request" and PAN not in lines[0] and AADHAAR not in lines[1]
    configure_logging(tmp_path / "again.jsonl", scrub, stream=stream)  # replaces, never stacks
    assert len(logging.getLogger("credagent").handlers) == 2


def test_request_fields_are_validated_against_the_schema(tmp_path: Path) -> None:
    log = configure_logging(tmp_path / "r.jsonl", scrub, stream=io.StringIO())
    with pytest.raises(ValidationError):
        log.request(_fields(outcome="whatever"))
    with pytest.raises(ValidationError):
        log.request(_fields(llm_usage=[["composer", 3]]))  # numbers only
