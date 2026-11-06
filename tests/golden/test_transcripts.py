"""Golden determinism (§ARCH 11.8): regenerate every transcript and diff against the committed one.

Only volatile header values, UUIDs, timestamps and `*_ms` timings are normalised away. Any other
difference fails; intentional changes are re-blessed with `make bless` and reviewed in the diff.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

from scripts import make_transcripts
from scripts.transcript_kit import normalise

COMMITTED = Path(__file__).resolve().parents[2] / "transcripts"


def test_committed_transcripts_match_the_generators() -> None:
    committed = sorted(p.stem for p in COMMITTED.glob("*.md"))
    assert committed == sorted(make_transcripts.GENERATORS), "run `make bless` and review"


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("transcripts")
    assert make_transcripts.main(["--out", str(out)]) == 0
    return out


@pytest.mark.parametrize("name", sorted(make_transcripts.GENERATORS))
def test_transcript_is_byte_identical_after_normalisation(regenerated: Path, name: str) -> None:
    expected = normalise((COMMITTED / f"{name}.md").read_text(encoding="utf-8"))
    actual = normalise((regenerated / f"{name}.md").read_text(encoding="utf-8"))
    diff = "\n".join(
        difflib.unified_diff(
            expected.splitlines(), actual.splitlines(), "committed", "regenerated", lineterm=""
        )
    )
    assert actual == expected, f"{name} drifted:\n{diff}"
