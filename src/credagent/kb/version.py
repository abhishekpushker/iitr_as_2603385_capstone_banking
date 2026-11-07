"""`kb_version` (§ARCH 3.5): sha256 over the KB files and the chunk parameters.

Any edit to a KB file, an added or removed file, or a change to the chunking parameters or the
splitter's abbreviation list yields a new version. Ingests it into chunk metadata and `kb_meta`.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from credagent.kb.chunking import chunk_params
from credagent.kb.loader import kb_files


def kb_version(kb_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in kb_files(kb_dir):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    digest.update(json.dumps(chunk_params(), sort_keys=True).encode())
    return digest.hexdigest()


def next_kb_version(current: str, doc_id: str, body: str) -> str:
    """The version after adding `doc_id` with `body` (§ARCH 7.16): chained, deterministic."""
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return hashlib.sha256(f"{current}|{doc_id}|{body_hash}".encode()).hexdigest()


class KbVersion:
    """The live `kb_version` in the cache key (§ARCH 7.15, P13). It starts at the indexed version
    and `/v1/documents` advances it, so no key computed before an add can be hit after it.
    In-process only, like the cache it invalidates (D-043)."""

    def __init__(self, initial: str) -> None:
        self._lock = threading.Lock()
        self._value = initial

    @property
    def current(self) -> str:
        with self._lock:
            return self._value

    def advance(self, doc_id: str, body: str) -> str:
        with self._lock:
            self._value = next_kb_version(self._value, doc_id, body)
            return self._value
