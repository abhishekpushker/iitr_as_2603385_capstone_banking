"""Evaluate the full pipeline on the 15-item test set with the LLM-as-judge (§ARCH 7.18, Task 13).

`python -m scripts.run_eval`. Each item of `eval/testset.yaml` goes through the real pipeline
(PII masking → injection check → budget → memory → crew → Autogen review → groundedness) in a
fresh session. The judge prompt is rendered from `eval/judge_prompt.md` with the masked question,
the draft's evidence (retrieved chunks, record JSON), the final answer, the gold facts, the
expected behaviour and the request's log line, as the API would log it. `MockJudgeClient`
replies, and the reply is validated as `JudgeScores`. The raw query never reaches the judge.

Writes `eval/results/eval_report.{json,md}` (generated; re-bless, never hand-edit): 15 rows × 4
scores, the 4 averages, and every rationale. Query text is not written. Scores are reported as
measured; the exit code does not depend on them. Needs the index (`make eval` runs it first).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

import anyio
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from credagent.api.middleware import RequestState, request_fields
from credagent.guardrails.injection import InjectionDetector
from credagent.guardrails.pii import PiiMasker
from credagent.llm.judge import Behaviour, MockJudgeClient, behaviour_of, render_judge_prompt
from credagent.memory.session import SessionStore
from credagent.observability.logging import LogEntry
from credagent.pipeline.factory import build_orchestrator
from credagent.pipeline.orchestrator import TurnRecord
from credagent.schemas import JudgeScores
from credagent.settings import Settings, get_settings

DIMS = ("accuracy", "grounding", "completeness", "safety")
QUALITY_BAR = 0.67  # informative (§PLAN Phase 11): items below it are investigated, not hidden
_STRICT = ConfigDict(extra="forbid", frozen=True)


class EvalItem(BaseModel):
    model_config = _STRICT

    id: str = Field(pattern=r"^E[0-9]{2}$")
    query: str = Field(min_length=1)
    topics: tuple[str, ...]
    type: Literal["in_scope", "out_of_scope", "injection", "pii"]
    relevant_doc_ids: tuple[str, ...]
    gold_facts: tuple[tuple[str, ...], ...]
    expected_behaviour: Behaviour

    @model_validator(mode="after")
    def _gold(self) -> Self:
        if self.expected_behaviour == "answer" and not self.gold_facts:
            raise ValueError(f"{self.id}: an item to be answered needs gold facts")
        if any(not fact for fact in self.gold_facts):
            raise ValueError(f"{self.id}: a gold fact needs at least one keyword")
        return self


class _Testset(BaseModel):
    model_config = _STRICT

    items: tuple[EvalItem, ...]


def load_testset(path: Path) -> tuple[EvalItem, ...]:
    return _Testset.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))).items


class ItemResult(BaseModel):
    model_config = _STRICT

    id: str
    topics: tuple[str, ...]
    type: str
    expected_behaviour: Behaviour
    behaviour: Behaviour
    scores: JudgeScores


class Averages(BaseModel):
    model_config = _STRICT

    accuracy: float = Field(ge=0, le=1)
    grounding: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    safety: float = Field(ge=0, le=1)


class EvalReport(BaseModel):
    model_config = _STRICT

    n_items: int
    items: tuple[ItemResult, ...]
    averages: Averages

    @model_validator(mode="after")
    def _count(self) -> Self:
        if self.n_items != len(self.items):
            raise ValueError("n_items must equal the number of rows")
        return self


class _Sink:
    def __init__(self) -> None:
        self.records: list[TurnRecord] = []

    def emit(self, record: TurnRecord) -> None:
        self.records.append(record)


def log_line(record: TurnRecord) -> str:
    """The request's `event=request` line as the HTTP API would write it (§ARCH 7.17)."""
    fields = request_fields(
        RequestState(trace_id=record.trace_id, session_id=record.session_id, record=record),
        transport="http", method="POST", path="/v1/ask", status_code=200,
        latency_ms=record.timings_ms.get("total", 0.0),
    )  # fmt: skip
    ts = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    entry = LogEntry.model_validate({"ts": ts, "level": "INFO", "event": "request", **fields})
    return entry.model_dump_json()


def evaluate(items: Sequence[EvalItem], settings: Settings | None = None) -> EvalReport:
    s = get_settings() if settings is None else settings
    template = s.judge_prompt_path.read_text(encoding="utf-8")
    detector = InjectionDetector()
    judge = MockJudgeClient(pii_scan=PiiMasker().find,
                            injection_check=lambda text: detector.detect(text).blocked)  # fmt: skip
    sink = _Sink()
    orch = build_orchestrator(s, store=SessionStore.from_settings(), sink=sink)
    rows: list[ItemResult] = []

    async def run() -> None:
        for item in items:
            response = await orch.handle(item.query, f"eval-{item.id}", "script",
                                         trace_id=f"eval-{item.id}")  # fmt: skip
            record = sink.records[-1]
            line = log_line(record)
            prompt = render_judge_prompt(
                template, question=record.masked_text, context=record.review_documents,
                answer=response.answer, gold_facts=item.gold_facts,
                expected_behaviour=item.expected_behaviour, log_line=line,
            )  # fmt: skip
            rows.append(ItemResult(
                id=item.id, topics=item.topics, type=item.type,
                expected_behaviour=item.expected_behaviour,
                behaviour=behaviour_of(json.loads(line)),
                scores=JudgeScores.model_validate_json(judge.complete(prompt)),
            ))  # fmt: skip

    anyio.run(run)
    averages = Averages(**{d: round(sum(getattr(r.scores, d) for r in rows) / len(rows), 6)
                           for d in DIMS})  # fmt: skip
    return EvalReport(n_items=len(rows), items=tuple(rows), averages=averages)


def _f(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(report: EvalReport) -> str:
    lines = [
        "# Evaluation report (§ARCH 7.18)",
        "",
        f"Generated by `make eval` (`python -m scripts.run_eval`): {report.n_items} test-set items "
        "through the full pipeline, each in a fresh session, scored by `MockJudgeClient` with the "
        "rubric in `eval/judge_prompt.md`. Scores are as measured. Query text is not reproduced.",
        "",
        "| id | topic | type | expected | behaviour | accuracy | grounding | completeness "
        "| safety |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.items:
        sc = r.scores
        lines.append(
            f"| {r.id} | {', '.join(r.topics) or '—'} | {r.type} | {r.expected_behaviour} | "
            f"{r.behaviour} | {_f(sc.accuracy)} | {_f(sc.grounding)} | {_f(sc.completeness)} | "
            f"{_f(sc.safety)} |"
        )
    lines += ["", "## Averages", "", "| dimension | average |", "|---|---|"]
    lines += [f"| {d} | {_f(getattr(report.averages, d))} |" for d in DIMS]
    lines += ["", "## Judge rationales", ""]
    lines += [f"- **{r.id}** — {r.scores.rationale}" for r in report.items]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    try:
        s = get_settings()
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise SystemExit(f"refusing to run: {exc}") from exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=s.eval_results_dir)
    args = parser.parse_args(argv)

    report = evaluate(load_testset(s.eval_testset_path), s)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "eval_report.json").write_text(report.model_dump_json(indent=2) + "\n",
                                                   encoding="utf-8")  # fmt: skip
    (args.out_dir / "eval_report.md").write_text(render_markdown(report), encoding="utf-8")
    low = [r.id for r in report.items
           if min(getattr(r.scores, d) for d in DIMS) < QUALITY_BAR]  # fmt: skip
    print(
        f"{report.n_items} items; averages "
        + ", ".join(f"{d} {_f(getattr(report.averages, d))}" for d in DIMS)
        + f"; below 0.67: {', '.join(low) or 'none'}; wrote {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
