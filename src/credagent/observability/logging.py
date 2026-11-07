"""Structured JSON-lines logging (§ARCH 7.17).

Stdlib `logging` on the `credagent` logger with two handlers, a `WatchedFileHandler(LOG_PATH)`
and a stream (stdout), both formatted by `JsonLinesFormatter` and both filtered by
`PiiScrubFilter`, which re-applies the PII masker to the message and every string field. The
masker is injected (`scrub`): this module is a leaf every layer may use (D-024), so it imports
no guardrail.

`EventLog.request(fields)` writes the one `event="request"` line per HTTP request or WS frame;
the fields are validated against `LogEntry` first. `EventLog.event(name, **fields)` writes the
other events (`startup`, `ready`, `ws_disconnect`, `ws_disconnect_mid_turn`, `error`, …), which
are never counted as requests.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field

LOGGER_NAME: Final = "credagent"
TEXT_LIMIT: Final = 2000

Scrub = Callable[[str], str]
LogOutcome = Literal[
    "answered", "refused", "clarify", "blocked_injection", "budget_rejected", "validation_error",
    "unauthorized", "timeout", "error",
]  # fmt: skip


class LogEntry(BaseModel):
    """One `event="request"` line (§ARCH 7.17). `llm_usage` is `[prompt, completion]` tokens per
    mock-LLM call: numbers only."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ts: str
    level: str
    event: Literal["request"]
    trace_id: str
    session_id: str | None
    transport: Literal["http", "ws"]
    method: str
    path: str
    status_code: int
    outcome: LogOutcome
    request_text_masked: str = Field(max_length=TEXT_LIMIT)
    truncated: bool
    pii_masked: dict[str, int]
    injection_score: float
    intent: str | None
    cache_hit: bool
    tools_invoked: list[str]
    review_approved: bool | None
    tokens_estimated: int
    cost_estimated_usd: float
    llm_usage: list[list[int]]
    history_length: int | None
    latency_ms: float
    timings_ms: dict[str, float]
    error_code: str | None


def truncate(text: str) -> tuple[str, bool]:
    return (text, False) if len(text) <= TEXT_LIMIT else (text[:TEXT_LIMIT], True)


def _ts(created: float) -> str:
    return datetime.fromtimestamp(created, UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": _ts(record.created), "level": record.levelname, "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }  # fmt: skip
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _scrubbed(value: Any, scrub: Scrub) -> Any:
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {k: _scrubbed(v, scrub) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_scrubbed(v, scrub) for v in value]
    return value


class PiiScrubFilter(logging.Filter):
    """Last line of defence: whatever reaches a handler is masked again (idempotent)."""

    def __init__(self, scrub: Scrub) -> None:
        super().__init__()
        self._scrub = scrub

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = self._scrub(record.getMessage()), None
        record.fields = _scrubbed(getattr(record, "fields", {}), self._scrub)
        return True


class EventLog:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def request(self, fields: dict[str, Any]) -> None:
        entry = LogEntry.model_validate(
            {"ts": _ts(datetime.now(UTC).timestamp()), "level": "INFO", "event": "request",
             **fields}
        )  # fmt: skip
        body = entry.model_dump(mode="json", exclude={"ts", "level", "event"})
        self._logger.info("request", extra={"fields": body})

    def event(self, name: str, *, level: int = logging.INFO, **fields: Any) -> None:
        self._logger.log(level, name, extra={"fields": fields})


def configure_logging(path: Path, scrub: Scrub, *, stream: TextIO | None = None) -> EventLog:
    """(Re)configure the `credagent` logger: previous handlers are closed and replaced."""
    logger = logging.getLogger(LOGGER_NAME)
    for old in list(logger.handlers):
        logger.removeHandler(old)
        old.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.handlers.WatchedFileHandler(path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout if stream is None else stream),
    ]
    for handler in handlers:
        handler.setFormatter(JsonLinesFormatter())
        handler.addFilter(PiiScrubFilter(scrub))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return EventLog(logger)
