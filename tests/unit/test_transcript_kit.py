"""The transcript framework (§ARCH 7.19): header block, ✔/✘ checks, normalisation, CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from credagent.settings import Settings
from scripts import make_transcripts
from scripts.transcript_kit import Transcript, normalise

GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
IMAGE_ID = "sha256:" + "ab" * 32
DATASET_SHA = "cd" * 32


def _settings() -> Settings:
    return Settings(git_sha=GIT_SHA, image_id=IMAGE_ID)


def _transcript() -> Transcript:
    return Transcript("99_example", "Example", command="make transcripts", settings=_settings())


def test_header_block_has_the_provenance_fields() -> None:
    text = _transcript().render()
    assert text.startswith("# 99 — Example\n")
    assert "- **Command:** `make transcripts`" in text
    assert f"- **Image:** `{IMAGE_ID}`" in text
    assert f"- **Git SHA:** `{GIT_SHA}`" in text
    assert "- **MOCK_LLM:** `true`" in text
    for flag in (
        "CREWAI_DISABLE_TELEMETRY=true",
        "OTEL_SDK_DISABLED=true",
        "CREWAI_TRACING_ENABLED=false",
        "ANONYMIZED_TELEMETRY=False",
        "HF_HUB_OFFLINE=1",
    ):
        assert flag in text
    assert "- **Generated (UTC):** `" in text


def test_checks_render_tick_and_cross() -> None:
    t = _transcript()
    assert t.check("one is one", True) is True
    assert t.check("one is two", False) is False
    text = t.render()
    assert "- ✔ one is one" in text
    assert "- ✘ one is two" in text
    assert t.failed == ["one is two"]


def test_a_transcript_without_checks_fails() -> None:
    """Every transcript must assert something; an empty one is a broken generator."""
    t = _transcript()
    t.para("no checks here")
    assert t.failed == ["(no checks recorded)"]


def test_body_helpers_render_markdown() -> None:
    t = _transcript()
    t.section("Inputs")
    t.para("Some text.")
    t.table(["a", "b"], [[1, "x"], [2, "y"]])
    t.code("print(1)", lang="python")
    text = t.render()
    assert "\n## Inputs\n" in text
    assert "Some text." in text
    assert "| a | b |\n|---|---|\n| 1 | x |\n| 2 | y |" in text
    assert "```python\nprint(1)\n```" in text


def test_normalise_masks_volatile_values_only() -> None:
    t = _transcript()
    t.para(f"dataset sha256 `{DATASET_SHA}`")
    t.para("trace 123e4567-e89b-12d3-a456-426614174000 at 2026-11-07T10:11:12Z")
    t.code('{"total_ms": 12.5, "count": 3}')
    t.check("ok", True)
    out = normalise(t.render())
    assert GIT_SHA not in out
    assert IMAGE_ID not in out
    assert "- **Image:** <id>" in out
    assert "- **Git SHA:** <id>" in out
    assert "- **Generated (UTC):** <ts>" in out
    assert "trace <uuid> at <ts>" in out
    assert '"total_ms": <ms>' in out
    assert '"count": 3' in out
    assert DATASET_SHA in out  # golden content is NOT normalised away


def test_normalise_is_idempotent() -> None:
    once = normalise(_transcript().render())
    assert normalise(once) == once


def test_two_renders_differ_only_in_volatile_fields() -> None:
    a = Transcript("99_x", "X", command="c", settings=Settings(git_sha="a" * 40, image_id="i1"))
    b = Transcript("99_x", "X", command="c", settings=Settings(git_sha="b" * 40, image_id="i2"))
    for t in (a, b):
        t.check("same", True)
    assert normalise(a.render()) == normalise(b.render())


# The all-generators CLI run needs Chroma: tests/integration/test_transcripts_cli.py (BUG-002).


def test_cli_only_selects_one(tmp_path: Path) -> None:
    assert make_transcripts.main(["--out", str(tmp_path), "--only", "01_dataset_report"]) == 0
    assert [p.name for p in tmp_path.iterdir()] == ["01_dataset_report.md"]


def test_cli_rejects_an_unknown_transcript(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        make_transcripts.main(["--out", str(tmp_path), "--only", "77_nope"])


def test_cli_exits_non_zero_on_a_failed_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing(t: Transcript) -> None:
        t.check("this must fail", False)

    monkeypatch.setitem(make_transcripts.GENERATORS, "98_failing", ("Failing", failing))
    assert make_transcripts.main(["--out", str(tmp_path), "--only", "98_failing"]) == 1
    assert "✘ this must fail" in (tmp_path / "98_failing.md").read_text(encoding="utf-8")
    assert "98_failing" in capsys.readouterr().err


def test_dataset_transcript_content(tmp_path: Path) -> None:
    import dataset

    make_transcripts.main(["--out", str(tmp_path), "--only", "01_dataset_report"])
    text = (tmp_path / "01_dataset_report.md").read_text(encoding="utf-8")
    assert f"`SEED = {dataset.SEED}`" in text
    assert "## Seed history" in text
    assert dataset.report(dataset.LOAN_APPLICATIONS)["sha256"] in text
    for r in dataset.LOAN_APPLICATIONS:
        assert f"| {r['record_id']} |" in text
    assert "✘" not in text
    assert text.count("✔") >= 10
