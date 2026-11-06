"""Golden determinism for generated config (§ARCH 11.8): `config/retrieval.json` must equal a fresh
calibration byte for byte. Intentional changes are re-blessed with `make bless`."""

from __future__ import annotations

import difflib
from pathlib import Path

from scripts import calibrate_threshold

COMMITTED = Path(__file__).resolve().parents[2] / "config" / "retrieval.json"


def test_retrieval_config_matches_a_fresh_calibration(tmp_path: Path) -> None:
    out = tmp_path / "retrieval.json"
    assert calibrate_threshold.main(["--out", str(out)]) == 0
    expected = COMMITTED.read_text(encoding="utf-8")
    actual = out.read_text(encoding="utf-8")
    diff = "\n".join(
        difflib.unified_diff(
            expected.splitlines(), actual.splitlines(), "committed", "regenerated", lineterm=""
        )
    )
    assert actual == expected, f"config/retrieval.json drifted:\n{diff}"
