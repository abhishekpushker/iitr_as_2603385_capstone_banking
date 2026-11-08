"""README numbers come from generated artifacts (Phase 12, §ARCH 12).

* A block after `<!-- source: PATH -->` (up to the next blank line) is copied: each of its lines
  must occur verbatim as a line of PATH.
* The calibration, escalation, budget and evaluation numbers are parsed from the README and
  compared with `config/*.json` and `eval/results/eval_report.json`.
Intentional changes: `make bless`, then update the README from the regenerated artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

import dataset

REPO = Path(__file__).resolve().parents[2]
README = (REPO / "README.md").read_text(encoding="utf-8")
RETRIEVAL = json.loads((REPO / "config" / "retrieval.json").read_text(encoding="utf-8"))
ESCALATION = json.loads((REPO / "config" / "escalation.json").read_text(encoding="utf-8"))
BUDGET = json.loads((REPO / "config" / "budget.json").read_text(encoding="utf-8"))
EVAL = json.loads((REPO / "eval" / "results" / "eval_report.json").read_text(encoding="utf-8"))
SOURCE = re.compile(r"^<!-- source: (\S+) -->\n((?:.+\n)+)", re.MULTILINE)


def _row(first_cells: str) -> list[str]:
    """The cells of the README table row that starts with `first_cells`."""
    rows = [line for line in README.splitlines() if line.startswith(f"| {first_cells} |")]
    assert len(rows) == 1, f"expected one README row starting with {first_cells!r}, got {rows}"
    return [c.strip() for c in rows[0].strip("|").split("|")]


def _one(pattern: str) -> str:
    found = re.findall(pattern, README)
    assert len(found) == 1, f"{pattern!r}: {found}"
    return str(found[0])


def test_every_sourced_block_is_copied_verbatim() -> None:
    blocks = SOURCE.findall(README)
    assert len(blocks) >= 10
    for path, block in blocks:
        source = set((REPO / path).read_text(encoding="utf-8").splitlines())
        missing = [line for line in block.splitlines() if line not in source]
        assert missing == [], f"README lines not in {path}: {missing}"


@pytest.mark.parametrize("strategy", ["fixed", "sentence"])
def test_every_calibration_similarity_gap_and_threshold(strategy: str) -> None:
    col = RETRIEVAL["collections"][strategy]
    index = {"fixed": 3, "sentence": 4}[strategy]
    measured = [*col["in_scope"], *col["out_of_scope"]]
    readme_rows = re.findall(r"^\| \d+ \| (.+?) \| (?:in-scope|out-of-scope) \|", README, re.M)
    assert sorted(readme_rows) == sorted(q["query"] for q in measured)
    for q in measured:
        row = [r for r in README.splitlines() if f"| {q['query']} |" in r][0]
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells[index] == f"{q['top1_similarity']:.4f}", q["query"]
    cells = _row(f"`{col['collection']}`")
    assert cells[1:5] == [f"{col['min_in']:.4f}", f"{col['max_oos']:.4f}", f"{col['gap']:.4f}",
                          str(col["threshold"])]  # fmt: skip
    assert cells[5] == ("yes" if col["meets_min_gap"] else "no")


def test_deployed_strategy_and_minimum_gap() -> None:
    deployed = RETRIEVAL["deployed_strategy"]
    threshold = RETRIEVAL["collections"][deployed]["threshold"]
    assert f"**Deployed: `{deployed}`, threshold {threshold}.**" in README
    assert _one(r"`gap ≥ ([\d.]+)`") == str(RETRIEVAL["min_gap"])


def test_escalation_formula_p80_and_threshold() -> None:
    e = ESCALATION
    assert int(_one(r"\*\*P80 = (\d+)\*\*")) == e["p80_days_since_created"]
    assert float(_one(r"\*\*T = ([\d.]+)\*\*")) == e["threshold"]
    weights = re.search(r"round\(([\d.]+) · flagged \+ ([\d.]+) · recency", README)
    assert weights is not None
    assert (float(weights[1]), float(weights[2])) == (e["flag_weight"], e["recency_weight"])
    assert int(_one(r"\*\*unflagged\*\* application escalates from \*\*day (\d+)\*\*")) == \
        e["unflagged_min_days"]  # fmt: skip
    assert int(_one(r"\*\*flagged\*\* application starts at [\d.]+ and escalates from "
                    r"\*\*day (\d+)\*\*")) == e["flagged_min_days"]  # fmt: skip


BUDGET_ROWS: dict[str, Any] = {
    "`PROMPT_OVERHEAD_TOKENS`": BUDGET["prompt_overhead_tokens"],
    "`MAX_OUTPUT_TOKENS`": BUDGET["max_output_tokens"],
    "`N_LLM_CALLS`": BUDGET["n_llm_calls"],
    "`BUDGET_MAX_TOKENS`": BUDGET["budget_max_tokens"],
    "`PRICE_PER_1K_TOKENS_USD`": BUDGET["price_per_1k_tokens_usd"],
    "`BUDGET_MAX_COST_USD`": BUDGET["budget_max_cost_usd"],
    "requests measured": BUDGET["measured"]["requests"],
    "LLM calls measured": BUDGET["measured"]["llm_calls"],
    "max actual tokens per request": BUDGET["measured"]["max_actual_total_tokens"],
    "max preflight projection (normal query)": BUDGET["measured"]["max_normal_projection_tokens"],
}


@pytest.mark.parametrize(("label", "value"), BUDGET_ROWS.items())
def test_budget_numbers(label: str, value: Any) -> None:
    assert _row(label)[1] == str(value)


def test_eval_scores_and_averages() -> None:
    dims = ("accuracy", "grounding", "completeness", "safety")
    assert len(EVAL["items"]) == EVAL["n_items"] == 15
    for item in EVAL["items"]:
        assert _row(item["id"])[5:9] == [f"{item['scores'][d]:.3f}" for d in dims], item["id"]
    for d in dims:
        assert _row(d)[1] == f"{EVAL['averages'][d]:.3f}"


def test_dataset_inputs() -> None:
    assert int(_one(r"`SEED = (\d+)`")) == dataset.SEED
    assert int(_one(r"`N_RECORDS = (\d+)`")) == dataset.N_RECORDS
    assert float(_one(r"Bernoulli, p = ([\d.]+)")) == dataset.FRAUD_FLAG_PROBABILITY


def test_offline_statement_and_referenced_files() -> None:
    statement = (  # §ARCH 12, compared with line breaks collapsed
        "All graded transcripts were produced with `MOCK_LLM=true`, zero API keys, "
        "`CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`, "
        "`CREWAI_TRACING_ENABLED=false`, `ANONYMIZED_TELEMETRY=False`, `HF_HUB_OFFLINE=1`, "
        "inside a Docker network with no internet route."
    )
    assert statement in " ".join(README.split())
    assert README.startswith("**Track: Cred (Banking & FinTech).**")
    links = re.findall(r"\]\(((?:transcripts|docs)/[^)#]+)\)", README)
    paths = re.findall(r"`((?:tests|scripts)/[\w/]+\.py)`", README)
    missing = [p for p in {*links, *paths} if not (REPO / p).is_file()]
    assert missing == []
