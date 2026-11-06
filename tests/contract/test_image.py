"""Image and container contract (§ARCH 3.2, 3.3, 5): user, read-only fs, model, offline env."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

MODEL_DIR = Path("/models/all-MiniLM-L6-v2")


def test_runs_as_uid_10001() -> None:
    assert (os.getuid(), os.getgid()) == (10001, 10001)


@pytest.mark.parametrize("path", ["/", "/app", "/usr/local/lib", "/models"])
def test_root_filesystem_is_read_only(path: str) -> None:
    denied = "Read-only file system|Permission denied"
    with pytest.raises(OSError, match=denied), tempfile.TemporaryFile(dir=path):
        pass


@pytest.mark.parametrize("path", ["/tmp", "/home/app"])
def test_scratch_dirs_are_writable(path: str) -> None:
    with tempfile.TemporaryFile(dir=path) as fh:
        fh.write(b"ok")


def test_embedding_model_is_baked_in_at_the_pinned_revision() -> None:
    assert (MODEL_DIR / "config.json").is_file()
    assert any(MODEL_DIR.glob("**/*.safetensors"))
    revision = (MODEL_DIR / "REVISION").read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert revision == os.environ["EMBED_MODEL_REVISION"]


def test_every_model_file_is_readable_by_the_runtime_user() -> None:
    """BUG-001: `SentenceTransformer.save()` (run as root in the `model` stage) wrote
    `model.safetensors` with mode 0600, so uid 10001 could not read the weights. transformers
    reported that as a misleading "FileNotFoundError: No such file or directory"."""
    files = [p for p in MODEL_DIR.rglob("*") if p.is_file()]
    assert files
    unreadable = [str(p.relative_to(MODEL_DIR)) for p in files if not os.access(p, os.R_OK)]
    assert unreadable == []
    with (MODEL_DIR / "model.safetensors").open("rb") as fh:
        assert fh.read(8)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("MOCK_LLM", "true"),
        ("LLM_PROVIDER", "mock"),
        ("CREWAI_DISABLE_TELEMETRY", "true"),
        ("OTEL_SDK_DISABLED", "true"),
        ("CREWAI_TRACING_ENABLED", "false"),
        ("ANONYMIZED_TELEMETRY", "False"),
        ("HF_HUB_OFFLINE", "1"),
        ("TRANSFORMERS_OFFLINE", "1"),
        ("HF_HUB_DISABLE_TELEMETRY", "1"),
        ("TOKENIZERS_PARALLELISM", "false"),
        ("OMP_NUM_THREADS", "1"),
        ("EMBED_MODEL_PATH", str(MODEL_DIR)),
        ("CREWAI_STORAGE_DIR", "/tmp/crewai"),
        ("HOME", "/home/app"),
    ],
)
def test_runtime_env_is_set_by_the_image(key: str, value: str) -> None:
    assert os.environ.get(key) == value


# Runs in a fresh interpreter: every non-unix socket is blocked (pytest-socket) and every
# name lookup is recorded and refused. Prints a JSON report of what the import attempted.
_GUARDED_IMPORT = """
import json, socket, sys, traceback
import pytest_socket

created, lookups = [], []
_blocked_new = None

def _record_creation():
    frame = traceback.extract_stack()[-4]
    created.append(frame.filename.split("site-packages/")[-1] + ":" + frame.name)

class _Err(pytest_socket.SocketBlockedError):
    def __init__(self, *a, **k):
        _record_creation()
        super().__init__(*a, **k)

pytest_socket.SocketBlockedError = _Err
pytest_socket.disable_socket(allow_unix_socket=True)

def _no_dns(host, *a, **k):
    lookups.append(str(host))
    raise socket.gaierror(socket.EAI_NONAME, "blocked by test")

socket.getaddrinfo = _no_dns
socket.gethostbyname = _no_dns

import credagent  # egress-off defaults first, as every entry point does
import {module}
print(json.dumps({{"created": sorted(set(created)), "lookups": lookups}}))
"""

# The one socket a library may create at import: urllib3 binds an AF_INET6 socket to ::1 to
# detect IPv6 support (loopback, no traffic). It handles the refusal and the import succeeds.
ALLOWED_IMPORT_SOCKETS = {"urllib3/util/connection.py:_has_ipv6"}


@pytest.mark.parametrize(
    "module",
    ["crewai", "chromadb", "autogen_agentchat.teams", "sentence_transformers", "langchain_core"],
)
def test_import_works_offline_and_attempts_no_network(module: str) -> None:
    """P5/P6/P20: with sockets blocked the import still succeeds, makes zero DNS lookups,
    and nothing but the known loopback probe even tries to create a socket."""
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _GUARDED_IMPORT.format(module=module)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["lookups"] == []
    assert set(report["created"]) <= ALLOWED_IMPORT_SOCKETS, report["created"]
