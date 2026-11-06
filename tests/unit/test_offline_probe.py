"""`scripts/offline_probe.py`: the egress probe must be able to tell open from closed (§ARCH 11.7).

The real "no internet from the backend network" assertion lives in tests/security/test_offline.py;
here we prove the probe is not trivially "always unreachable".
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator

import pytest

from scripts import offline_probe
from scripts.offline_probe import ProbeResult, probe_dns, probe_tcp


@pytest.fixture
def listening_port() -> Iterator[int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        yield srv.getsockname()[1]


@pytest.mark.enable_socket
def test_tcp_probe_reports_an_open_port_as_reachable(listening_port: int) -> None:
    result = probe_tcp("127.0.0.1", listening_port, timeout_s=2.0)
    assert result.reachable is True
    assert result.target == f"tcp://127.0.0.1:{listening_port}"
    assert result.error is None


@pytest.mark.enable_socket
def test_tcp_probe_reports_a_closed_port_as_unreachable() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        closed = s.getsockname()[1]  # bound but never listening → connection refused
    result = probe_tcp("127.0.0.1", closed, timeout_s=2.0)
    assert result.reachable is False
    assert result.error


def test_dns_probe_reports_resolution() -> None:
    def resolver(host: str, port: int) -> list[object]:
        return [("AF_INET", "SOCK_STREAM", 6, "", ("203.0.113.7", port))]

    result = probe_dns("pypi.org", resolver=resolver)
    assert result.reachable is True
    assert result.target == "dns://pypi.org"


def test_dns_probe_reports_failure() -> None:
    def resolver(host: str, port: int) -> list[object]:
        raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")

    result = probe_dns("pypi.org", resolver=resolver)
    assert result.reachable is False
    assert "name resolution" in (result.error or "")


def test_non_network_errors_are_not_mistaken_for_offline() -> None:
    """Only OSError means 'unreachable'; anything else is a probe bug and must surface."""

    def resolver(host: str, port: int) -> list[object]:
        raise ValueError("bug")

    with pytest.raises(ValueError, match="bug"):
        probe_dns("pypi.org", resolver=resolver)


def _fake(reachable_tcp: bool, reachable_dns: bool) -> tuple[ProbeResult, ProbeResult]:
    return (
        ProbeResult("tcp://1.1.1.1:443", reachable_tcp, None if reachable_tcp else "timed out"),
        ProbeResult("dns://pypi.org", reachable_dns, None if reachable_dns else "EAI_AGAIN"),
    )


@pytest.mark.parametrize(
    ("tcp", "dns", "code"),
    [(False, False, 0), (True, False, 1), (False, True, 1), (True, True, 1)],
)
def test_main_exit_code_is_zero_only_when_everything_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tcp: bool,
    dns: bool,
    code: int,
) -> None:
    tcp_result, dns_result = _fake(tcp, dns)
    monkeypatch.setattr(offline_probe, "probe_tcp", lambda *a, **k: tcp_result)
    monkeypatch.setattr(offline_probe, "probe_dns", lambda *a, **k: dns_result)
    assert offline_probe.main() == code
    report = json.loads(capsys.readouterr().out)
    assert report["offline"] is (code == 0)
    assert [p["target"] for p in report["probes"]] == ["tcp://1.1.1.1:443", "dns://pypi.org"]


def test_main_refuses_to_run_without_mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from credagent.settings import get_settings

    monkeypatch.setenv("MOCK_LLM", "false")
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit, match="MOCK_LLM"):
            offline_probe.main()
    finally:
        get_settings.cache_clear()
