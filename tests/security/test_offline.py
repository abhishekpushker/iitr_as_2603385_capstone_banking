"""The `tests` container sits on the internal `backend` network: no route out (§ARCH 3.3, 11.7)."""

from __future__ import annotations

import json
import socket
import subprocess
import sys

import pytest

from credagent.settings import Settings
from scripts import make_transcripts
from scripts.transcript_kit import Transcript


def test_tcp_to_public_ip_fails() -> None:
    with pytest.raises(OSError):
        socket.create_connection(("1.1.1.1", 443), timeout=3).close()


def test_public_dns_does_not_resolve() -> None:
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("pypi.org", 443)


def test_offline_probe_script_reports_offline() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.offline_probe"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["offline"] is True
    assert {p["target"] for p in report["probes"]} == {"tcp://1.1.1.1:443", "dns://pypi.org"}
    assert all(p["reachable"] is False for p in report["probes"])


def _offline_proof(network_facts: str) -> Transcript:
    t = Transcript("25_offline_proof", "Offline proof", command="c",
                   settings=Settings(network_facts=network_facts))  # fmt: skip
    make_transcripts.offline_proof(t)
    return t


INTERNAL = json.dumps({svc: [{"name": "credagent_backend", "internal": True}]
                       for svc in ("runner", "tests", "api-offline")})  # fmt: skip


def test_offline_proof_transcript_records_probes_flags_and_networks() -> None:
    t = _offline_proof(INTERNAL)
    text = t.render()
    assert t.failed == []
    assert "tcp://1.1.1.1:443" in text and "dns://pypi.org" in text
    for flag in ("CREWAI_DISABLE_TELEMETRY", "OTEL_SDK_DISABLED", "CREWAI_TRACING_ENABLED",
                 "ANONYMIZED_TELEMETRY", "HF_HUB_OFFLINE", "MOCK_LLM"):  # fmt: skip
        assert f"`{flag}`" in text
    assert "credagent_backend" in text


@pytest.mark.parametrize("facts", [
    "",  # not passed in from the host
    json.dumps({"runner": [{"name": "credagent_edge", "internal": False}],
                "tests": [], "api-offline": []}),
])  # fmt: skip
def test_offline_proof_fails_without_internal_only_membership(facts: str) -> None:
    assert _offline_proof(facts).failed
