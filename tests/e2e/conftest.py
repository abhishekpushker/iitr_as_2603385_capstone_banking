"""E2E wiring (§ARCH 3.1, 7.16): every test talks to `api-offline` over the backend network.

The log file is the shared `logs` volume (read-only here). Tests find their own lines by the
trace ids they sent (`X-Trace-Id`) or received (WS frames), so other runs' lines never matter.
Documents a test adds are deleted from both collections afterwards, straight in Chroma.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from credagent.api.middleware import new_trace_id
from credagent.guardrails.pii import PiiMasker
from credagent.rag.retriever import collections_from_settings
from credagent.rag.store import VectorStore, connect_http
from credagent.settings import get_settings

S = get_settings()
BASE = "http://api-offline:8000"
WS_URL = "ws://api-offline:8000/v1/ws/chat"
USER = {"X-API-Key": S.api_key.get_secret_value()}
ADMIN = {"X-API-Key": S.admin_api_key.get_secret_value()}
# §ARCH 11.6: fixed fabricated canaries (PAN, Aadhaar grouped/contiguous, bank account).
CANARIES = ("ABCPE1234F", "2345 6789 0123", "234567890123", "123456789012345")
CANARY_TEXT = (
    f"My PAN is {CANARIES[0]}, Aadhaar {CANARIES[1]} (or {CANARIES[2]}), account no "
    f"{CANARIES[3]}. What is the status of application LA-0007?"
)
MASKER = PiiMasker()


def trace() -> str:
    return new_trace_id(MASKER)


def unique_doc_id(prefix: str) -> str:
    return f"kb-{prefix}-{uuid.uuid4().hex[:12]}"


@pytest.fixture(scope="session")
def client() -> Iterator[httpx.Client]:
    with httpx.Client(base_url=BASE, timeout=180) as c:
        ready = c.get("/readyz")  # compose waits for this healthcheck before tests start
        assert ready.status_code == 200, ready.text
        yield c


def log_lines() -> list[dict[str, Any]]:
    with S.log_path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def wait_for_log(predicate: Callable[[dict[str, Any]], bool], timeout_s: float = 60) -> bool:
    """Only for side effects that happen after the client is gone (e.g. a mid-turn disconnect
    is logged when the server's pipeline finishes): no response can signal them."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if any(predicate(x) for x in log_lines()):
            return True
        time.sleep(0.2)
    return False


@pytest.fixture(scope="session")
def vectors() -> VectorStore:
    return VectorStore(connect_http(S.chroma_host, S.chroma_port, attempts=3, backoff_cap_s=1))


@pytest.fixture
def added_docs(vectors: VectorStore) -> Iterator[list[str]]:
    """Doc ids the test adds; their chunks are removed from both collections afterwards."""
    ids: list[str] = []
    yield ids
    for name in collections_from_settings(S).values():
        for doc_id in ids:
            vectors.delete(name, vectors.ids_for_doc(name, doc_id))
