"""Mock honesty, first version (§ARCH 11.4, N7).

Ten paraphrased policy questions that appear nowhere else in the repository: they are not
calibration, demo, P/R or eval queries, and the mock engine has never seen them. Through the real
retriever + MockEngine, at least 8/10 must give a grounded, non-fallback answer whose top-1
source is the right document, and every emitted sentence must be supported by the context.
"""

from __future__ import annotations

import pytest

from credagent.llm.engine import MockEngine
from credagent.observability.metrics import Metrics
from credagent.rag import calibration as cal
from credagent.rag.embedder import get_embedder
from credagent.rag.generator import GroundedGenerator
from credagent.rag.retriever import Retriever, collections_from_settings
from credagent.rag.store import VectorStore, connect_http
from credagent.schemas import FALLBACK_ANSWER
from credagent.settings import get_settings
from scripts import indexer

# (paraphrased question, the document that answers it)
HELD_OUT: tuple[tuple[str, str], ...] = (
    ("At what age can I apply for a personal loan?", "kb-01-loan-eligibility"),
    ("On which date is my first instalment due?", "kb-02-emi-calculation"),
    ("How much does withdrawing cash on the card cost?", "kb-03-credit-card-fees"),
    ("How often do low-risk customers need to redo KYC?", "kb-04-kyc-documents"),
    ("Will I get a temporary credit while my dispute is investigated?", "kb-05-fraud-dispute"),
    ("When will the credit bureau be informed that my account is closed?",
     "kb-06-account-closure"),
    ("What is the floating rate for housing loans?", "kb-07-interest-rate-slabs"),
    ("When am I allowed to foreclose my loan completely?", "kb-08-prepayment-penalty"),
    ("Are salary accounts required to maintain a minimum balance?", "kb-09-minimum-balance"),
    ("Can NRIs take a personal loan?", "kb-12-nri-eligibility"),
)  # fmt: skip
REQUIRED_CORRECT = 8


@pytest.fixture(scope="module")
def generator() -> GroundedGenerator:
    assert indexer.main([]) == 0
    s = get_settings()
    client = connect_http(
        s.chroma_host, s.chroma_port, attempts=s.chroma_connect_attempts,
        backoff_cap_s=s.chroma_backoff_cap_s,
    )  # fmt: skip
    retriever = Retriever(
        VectorStore(client), get_embedder(), collections_from_settings(s), top_k=s.top_k,
        metrics=Metrics(),
    )  # fmt: skip
    config = cal.load_retrieval_config(s.retrieval_config_path)
    strategy = config.gated_strategy
    return GroundedGenerator(
        retriever, MockEngine(), strategy=strategy, threshold=config.threshold_for(strategy),
        top_k=config.top_k, metrics=Metrics(),
    )  # fmt: skip


def test_held_out_paraphrases_are_answered_from_the_right_document(
    generator: GroundedGenerator,
) -> None:
    engine = MockEngine()
    outcomes: list[tuple[str, bool]] = []
    for query, expected_doc in HELD_OUT:
        retrieval = generator.retrieve(query)
        answer = generator.generate(query, retrieval)
        ok = (
            answer.grounded
            and answer.answer != FALLBACK_ANSWER
            and retrieval.chunks[0].doc_id == expected_doc
            and answer.sources[0].doc_id == expected_doc
            and engine.support_report(
                answer.answer, [c.text for c in retrieval.chunks]
            ).all_supported
        )
        outcomes.append((query, ok))
    correct = sum(ok for _, ok in outcomes)
    assert correct >= REQUIRED_CORRECT, f"{correct}/10 held-out correct: {outcomes}"


def test_held_out_queries_are_not_reused_anywhere() -> None:
    from tests.unit.test_query_sets import other_query_sets

    held = {q for q, _ in HELD_OUT}
    for name, queries in other_query_sets(include_held_out=False).items():
        assert held.isdisjoint(queries), f"held-out queries reused in {name}"
