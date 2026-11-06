"""Egress probe (§ARCH 11.7): exit 0 only if the internet is unreachable from this container.

Prints a JSON report. Run with `python -m scripts.offline_probe` on the backend network.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from credagent.settings import get_settings

Resolver = Callable[[str, int], Sequence[object]]

TCP_TARGET = ("1.1.1.1", 443)
DNS_TARGET = "pypi.org"


@dataclass(frozen=True)
class ProbeResult:
    target: str
    reachable: bool
    error: str | None


def probe_tcp(host: str, port: int, timeout_s: float = 3.0) -> ProbeResult:
    target = f"tcp://{host}:{port}"
    try:
        socket.create_connection((host, port), timeout=timeout_s).close()
    except OSError as exc:  # refused, unreachable, timed out: all mean "no route"
        return ProbeResult(target, False, f"{type(exc).__name__}: {exc}")
    return ProbeResult(target, True, None)


def probe_dns(name: str, resolver: Resolver = socket.getaddrinfo) -> ProbeResult:
    target = f"dns://{name}"
    try:
        resolver(name, 443)
    except OSError as exc:  # socket.gaierror is an OSError
        return ProbeResult(target, False, f"{type(exc).__name__}: {exc}")
    return ProbeResult(target, True, None)


def main() -> int:
    try:
        get_settings()
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise SystemExit(f"refusing to run: {exc}") from exc
    probes = [probe_tcp(*TCP_TARGET), probe_dns(DNS_TARGET)]
    offline = not any(p.reachable for p in probes)
    print(json.dumps({"offline": offline, "probes": [asdict(p) for p in probes]}, indent=2))
    return 0 if offline else 1


if __name__ == "__main__":
    raise SystemExit(main())
