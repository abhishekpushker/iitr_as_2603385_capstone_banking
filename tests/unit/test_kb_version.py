"""kb_version (§ARCH 3.5): stable for the same inputs, sensitive to every input."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from credagent.kb import chunking
from credagent.kb.version import kb_version

KB_DIR = Path(__file__).resolve().parents[2] / "kb"


@pytest.fixture
def kb_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "kb"
    shutil.copytree(KB_DIR, dst)
    return dst


def test_version_is_a_stable_sha256(kb_copy: Path) -> None:
    v = kb_version(KB_DIR)
    assert re.fullmatch(r"[0-9a-f]{64}", v)
    assert kb_version(KB_DIR) == v == kb_version(kb_copy)  # independent of the directory path


def test_version_changes_when_a_file_changes(kb_copy: Path) -> None:
    before = kb_version(kb_copy)
    path = kb_copy / "kb-06-account-closure.md"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert kb_version(kb_copy) != before


def test_version_changes_when_a_file_is_removed_or_renamed(kb_copy: Path) -> None:
    before = kb_version(kb_copy)
    (kb_copy / "kb-06-account-closure.md").rename(kb_copy / "kb-06-closure.md")
    renamed = kb_version(kb_copy)
    (kb_copy / "kb-06-closure.md").unlink()
    assert len({before, renamed, kb_version(kb_copy)}) == 3


def test_version_changes_with_chunk_params(monkeypatch: pytest.MonkeyPatch) -> None:
    before = kb_version(KB_DIR)
    monkeypatch.setattr(chunking, "FIXED_PARAMS", chunking.FixedParams(size=200))
    assert kb_version(KB_DIR) != before


def test_live_version_advances_deterministically_per_added_document() -> None:
    """P13: `/v1/documents` advances the live kb_version, so every cache key changes."""
    from credagent.kb.version import KbVersion

    a, b = KbVersion("v0"), KbVersion("v0")
    assert a.current == "v0"
    first = a.advance("kb-new-doc", "Body one.")
    assert re.fullmatch(r"[0-9a-f]{64}", first) and a.current == first
    assert b.advance("kb-new-doc", "Body one.") == first  # same inputs, same version
    assert a.advance("kb-new-doc", "Body one.") != first  # a re-add still moves it on
