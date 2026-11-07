"""`POST /v1/documents` behind the API (§ARCH 7.16, P13).

`prepare` masks the title, topic and text and scores injection (pure, never raises), so the
request log has the masked fields whatever happens next. `add` then, in order: refuses an
injection (422), a body outside 2–5 sentences (422), a doc id already indexed unless `overwrite`
(409); indexes the masked body into **both** collections; advances the live `kb_version` and
clears the response cache. Adds are serialised by one lock, so the duplicate check and the write
cannot interleave.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol

from credagent.guardrails.injection import InjectionDetector
from credagent.guardrails.pii import PiiMasker, PiiType
from credagent.kb.chunking import Strategy
from credagent.kb.version import KbVersion
from credagent.pipeline.orchestrator import PipelineError
from credagent.rag.embedder import Encoder
from credagent.rag.indexing import index_document
from credagent.rag.store import VectorStore
from credagent.text import clean, split_sentences

MIN_SENTENCES: Final = 2
MAX_SENTENCES: Final = 5


class DuplicateDocument(PipelineError):
    error_code = "DUPLICATE_DOCUMENT"


class DocumentInjection(PipelineError):
    error_code = "INJECTION_DETECTED"


class DocumentInvalid(PipelineError):
    error_code = "VALIDATION_ERROR"


class Clearable(Protocol):
    """The response cache, as far as adding a document needs it."""

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class PreparedDocument:
    """Masked fields only; the raw request text is not kept."""

    doc_id: str
    title: str
    topic: str
    body: str
    pii_counts: dict[PiiType, int]
    injection_score: float
    injection_blocked: bool
    sentences: int


@dataclass(frozen=True)
class AddedDocument:
    doc_id: str
    chunks: dict[Strategy, int]
    kb_version: str


class DocumentService:
    def __init__(
        self,
        *,
        vectors: VectorStore,
        encoder: Encoder,
        collections: Mapping[Strategy, str],
        kb: KbVersion,
        cache: Clearable,
        masker: PiiMasker | None = None,
        detector: InjectionDetector | None = None,
    ) -> None:
        self._vectors = vectors
        self._encoder = encoder
        self._collections = dict(collections)
        self._kb = kb
        self._cache = cache
        self._masker = PiiMasker() if masker is None else masker
        self._detector = InjectionDetector() if detector is None else detector
        self._lock = threading.Lock()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def prepare(self, *, doc_id: str, title: str, topic: str, text: str) -> PreparedDocument:
        parts = [self._masker.mask(v) for v in (title, topic, text)]
        counts: dict[PiiType, int] = {}
        for part in parts:
            for pii_type, n in part.counts_by_type.items():
                counts[pii_type] = counts.get(pii_type, 0) + n
        title_m, topic_m, body = (clean(p.text) for p in parts)
        injection = self._detector.detect(f"{title_m}\n{topic_m}\n{body}")
        return PreparedDocument(
            doc_id=doc_id, title=title_m, topic=topic_m, body=body, pii_counts=counts,
            injection_score=injection.score, injection_blocked=injection.blocked,
            sentences=len(split_sentences(body)),
        )  # fmt: skip

    def add(self, doc: PreparedDocument, *, overwrite: bool) -> AddedDocument:
        if doc.injection_blocked:
            raise DocumentInjection("the document contains a prompt-injection pattern")
        if not MIN_SENTENCES <= doc.sentences <= MAX_SENTENCES:
            raise DocumentInvalid(f"the text must have {MIN_SENTENCES}–{MAX_SENTENCES} sentences")
        with self._lock:
            exists = any(
                self._vectors.ids_for_doc(n, doc.doc_id) for n in self._collections.values()
            )
            if exists and not overwrite:
                raise DuplicateDocument("a document with this doc_id is already indexed")
            version = self._kb.advance(doc.doc_id, doc.body)
            chunks = index_document(self._vectors, self._encoder, self._collections, doc, version)
            self._cache.clear()
        return AddedDocument(doc_id=doc.doc_id, chunks=chunks, kb_version=version)
