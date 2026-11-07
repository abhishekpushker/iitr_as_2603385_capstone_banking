"""The transcript CLI writes every registered transcript against the real index (§ARCH 7.19).

BUG-002: this test used to live in the unit layer, but transcripts 03–06 query Chroma. It passed
only while a previous make target had left `chroma` running; on a fresh `make gate`, step 3
(unit + contract) runs before any service is up, and it failed with "chroma unreachable".
It now runs in the integration layer and indexes first, so a cold volume cannot leave it
reading an empty index.
"""

from __future__ import annotations

from pathlib import Path

from scripts import indexer, make_transcripts


def test_cli_writes_every_registered_transcript(tmp_path: Path) -> None:
    """BUG-002 regression: needs chroma + an index, so it belongs to the integration layer."""
    assert indexer.main([]) == 0
    assert make_transcripts.main(["--out", str(tmp_path)]) == 0
    written = sorted(p.stem for p in tmp_path.glob("*.md"))
    assert written == sorted(make_transcripts.GENERATORS)
    assert "01_dataset_report" in written
