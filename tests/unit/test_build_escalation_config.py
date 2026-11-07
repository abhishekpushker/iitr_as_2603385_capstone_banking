"""`scripts/build_escalation_config.py` writes `config/escalation.json` from the dataset (§7.7)."""

from __future__ import annotations

from pathlib import Path

import pytest

import dataset
from credagent.tools import escalation as esc
from scripts import build_escalation_config


def test_writes_the_config_built_from_the_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "nested" / "escalation.json"
    assert build_escalation_config.main(["--out", str(out)]) == 0
    expected = esc.build_config(dataset.LOAN_APPLICATIONS)
    assert out.read_text(encoding="utf-8") == esc.dumps_config(expected)
    printed = capsys.readouterr().out
    assert f"P80 = {expected.p80_days_since_created}" in printed
    assert f"T = {expected.threshold}" in printed


def test_refuses_an_invalid_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset, "LOAN_APPLICATIONS", dataset.LOAN_APPLICATIONS[:10])
    out = tmp_path / "escalation.json"
    assert build_escalation_config.main(["--out", str(out)]) == 1
    assert not out.exists()
