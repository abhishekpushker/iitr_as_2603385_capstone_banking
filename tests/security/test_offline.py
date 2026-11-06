"""The `tests` container sits on the internal `backend` network: no route out (§ARCH 3.3, 11.7)."""

from __future__ import annotations

import json
import socket
import subprocess
import sys

import pytest


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
