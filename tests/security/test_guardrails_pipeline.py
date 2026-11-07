"""Guardrails at the pipeline boundary (§ARCH 7.12, 9): every corpus attack is refused before the
crew and leaves no trace in memory; every benign question reaches the crew; no PII value
reaches any sink."""

from __future__ import annotations

import pytest

from credagent.guardrails.injection import load_corpus
from credagent.settings import get_settings
from tests.unit.test_orchestrator import CANARIES, RAW_PII, make

CORPUS = load_corpus(get_settings().injection_corpus_path)


@pytest.mark.anyio
async def test_every_attack_is_refused_without_a_crew_call_and_not_stored() -> None:
    orch, crew, sink, store = make()
    for item in CORPUS.attacks:
        r = await orch.handle(item.text, f"attack-{item.id}")
        assert (r.refused, r.refusal_reason) == (True, "injection_detected"), item.id
        assert store.history_length(f"attack-{item.id}") == 0, item.id
    assert crew.queries == []
    assert {rec.outcome for rec in sink.records} == {"blocked_injection"}


@pytest.mark.anyio
async def test_every_benign_question_reaches_the_crew_or_a_clarification() -> None:
    orch, crew, sink, _ = make()
    for item in CORPUS.benign:
        await orch.handle(item.text, f"benign-{item.id}")
    assert "blocked_injection" not in {rec.outcome for rec in sink.records}
    assert len(crew.queries) + sum(rec.outcome == "clarify" for rec in sink.records) == 25


@pytest.mark.anyio
async def test_pii_canaries_reach_no_sink() -> None:
    orch, crew, sink, store = make()
    r = await orch.handle(RAW_PII, "canary")
    sinks = [r.model_dump_json(), repr(sink.records), repr(store.messages("canary")), *crew.queries]
    for canary in CANARIES:
        assert all(canary not in s for s in sinks), canary
