"""Repository hygiene (Phase 12, §PLAN 5 "Submission"): what git tracks.

The tests container has no git binary, and reading `.git/index` through the bind mount races with
host git commands that replace it. `make test-unit` / `test-unit-contract` therefore write
`git ls-files` to `artifacts/tracked-files.txt` (in place) just before the container starts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TRACKED_LIST = REPO / "artifacts" / "tracked-files.txt"
MEDIA = frozenset({
    "png", "jpg", "jpeg", "gif", "bmp", "svg", "webp", "ico", "tif", "tiff", "heic", "psd", "ai",
    "eps", "pdf", "ppt", "pptx", "pps", "ppsx", "odp", "key", "mp3", "mp4", "wav", "avi", "mov",
    "mkv", "webm", "flac", "ogg", "m4a",
})  # fmt: skip
# Shapes of real credentials; the placeholders in .env.example match none of them.
SECRET_SHAPES = (
    re.compile(r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"xox[abpr]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
)
PLACEHOLDER = re.compile(r"^[A-Z_]*(?:KEY|SECRET|TOKEN|PASSWORD)[A-Z_]*=(.*)$", re.MULTILINE)


@pytest.fixture(scope="module")
def tracked() -> list[str]:
    assert TRACKED_LIST.is_file(), "run through `make test-unit` (it writes the git ls-files list)"
    return [p for p in TRACKED_LIST.read_text(encoding="utf-8").splitlines() if p]


def test_the_list_holds_the_tracked_paths(tracked: list[str]) -> None:
    assert {"dataset.py", "CLAUDE.md", ".env.example", "docker/Dockerfile"} <= set(tracked)


def test_no_image_pdf_slide_or_media_files_are_tracked(tracked: list[str]) -> None:
    media = [p for p in tracked if p.rsplit(".", 1)[-1].lower() in MEDIA]
    assert media == []


def test_no_env_file_is_tracked_only_the_example(tracked: list[str]) -> None:
    envs = [p for p in tracked if re.fullmatch(r"(?:.*/)?\.env(?:\..+)?", p)]
    assert envs == [".env.example"]


def test_env_example_holds_only_dev_placeholders() -> None:
    values = PLACEHOLDER.findall((REPO / ".env.example").read_text(encoding="utf-8"))
    assert values and all(v.endswith("-not-a-secret") for v in values), values


@pytest.mark.parametrize("pattern", SECRET_SHAPES, ids=lambda p: p.pattern[:12])
def test_no_tracked_file_contains_a_credential(
    tracked: list[str], pattern: re.Pattern[str]
) -> None:
    hits = []
    for path in tracked:
        file = REPO / path
        if not file.is_file():
            continue
        if pattern.search(file.read_text(encoding="utf-8", errors="ignore")):
            hits.append(path)
    assert hits == []
