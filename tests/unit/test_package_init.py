"""`import credagent` sets the egress-off defaults before anything can import crewai (§ARCH 5)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

EGRESS_OFF = {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "ANONYMIZED_TELEMETRY": "False",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
}


def _env_after_import(env: dict[str, str]) -> dict[str, str | None]:
    code = (
        "import json, os, credagent; "
        f"print(json.dumps({{k: os.environ.get(k) for k in {sorted(EGRESS_OFF)!r}}}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)  # type: ignore[no-any-return]


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in EGRESS_OFF}


def test_import_sets_every_egress_flag_when_env_is_empty() -> None:
    assert _env_after_import(_clean_env()) == EGRESS_OFF


def test_import_does_not_override_explicit_values() -> None:
    """setdefault semantics: an explicit value survives; Settings is what rejects unsafe ones."""
    env = _clean_env() | {"OMP_NUM_THREADS": "2"}
    assert _env_after_import(env)["OMP_NUM_THREADS"] == "2"


def test_package_module_exports_the_defaults_table() -> None:
    import credagent

    assert dict(credagent.EGRESS_OFF_DEFAULTS) == EGRESS_OFF


@pytest.mark.parametrize("key", sorted(EGRESS_OFF))
def test_current_process_has_flag(key: str) -> None:
    import credagent  # noqa: F401  (import side effect under test)

    assert key in os.environ
