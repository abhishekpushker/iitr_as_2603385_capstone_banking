"""`build_orchestrator` wiring (no kickoff here; the real crew runs in tests/integration)."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from credagent.memory.session import SessionStore
from credagent.pipeline.factory import build_orchestrator
from credagent.pipeline.orchestrator import Orchestrator
from credagent.settings import Settings, get_settings


def _client() -> chromadb.api.ClientAPI:
    return chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))


def test_builds_an_orchestrator_on_the_given_client_and_store() -> None:
    store = SessionStore(max_sessions=5, ttl_s=60)
    orch = build_orchestrator(client=_client(), store=store)
    assert isinstance(orch, Orchestrator)
    assert orch.memory.store is store


def test_refuses_a_config_without_a_deployed_strategy(tmp_path: Path) -> None:
    config = json.loads(get_settings().retrieval_config_path.read_text(encoding="utf-8"))
    config["deployed_strategy"] = None
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="deployed_strategy"):
        build_orchestrator(Settings(retrieval_config_path=path), client=_client())
