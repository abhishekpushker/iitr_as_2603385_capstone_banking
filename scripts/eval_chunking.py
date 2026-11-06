"""Evaluate both chunking strategies by document-level P/R (§ARCH 7.6, Task 5).

`python -m scripts.eval_chunking`. Runs the P/R queries through the production retriever on both
collections, prints and writes the report (`eval/results/chunking_pr.md`), and sets
`deployed_strategy` in `config/retrieval.json`. Only strategies whose calibration gap meets
`min_gap` compete (D-028). It changes nothing and exits 1 if the config was calibrated on another
KB version, or if no strategy meets the gap (never deploy a threshold without the margin).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from credagent.kb.version import kb_version
from credagent.observability.metrics import Metrics
from credagent.rag import calibration as cal
from credagent.rag import evaluation as ev
from credagent.rag.embedder import get_embedder
from credagent.rag.retriever import Retriever, collections_from_settings
from credagent.rag.store import VectorStore, connect_http
from credagent.settings import get_settings


def main(argv: Sequence[str] | None = None) -> int:
    try:
        s = get_settings()
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise SystemExit(f"refusing to run: {exc}") from exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=s.retrieval_config_path)
    parser.add_argument("--report", type=Path, default=s.chunking_report_path)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    current = kb_version(s.kb_dir)
    if config["kb_version"] != current:
        print(
            f"{args.config} was calibrated on kb_version {config['kb_version']}, but kb/ is "
            f"{current}: run `make calibrate` first",
            file=sys.stderr,
        )
        return 1

    client = connect_http(
        s.chroma_host, s.chroma_port, attempts=s.chroma_connect_attempts,
        backoff_cap_s=s.chroma_backoff_cap_s,
    )  # fmt: skip
    retriever = Retriever(
        VectorStore(client), get_embedder(), collections_from_settings(s), top_k=s.top_k,
        metrics=Metrics(),
    )  # fmt: skip
    top_k = int(config["top_k"])
    results = ev.evaluate(retriever, top_k=top_k)
    calibrations = {st: ev.calibration_of(config, st) for st in results}
    try:
        choice = ev.choose(results, gated=config["gated_strategy"], calibrations=calibrations)
    except ev.NoEligibleStrategyError as exc:
        for c in results.values():
            print(f"[{c.strategy}] {c.macro_precision_arithmetic}; {c.macro_recall_arithmetic}")
        print(f"{exc}; {args.config} and {args.report} left unchanged", file=sys.stderr)
        return 1
    rec = ev.recommendation(
        results, choice, calibrations, min_gap=float(config["min_gap"]), top_k=top_k
    )
    report = ev.render_report(results, choice, rec, top_k=top_k)
    print(report)

    try:
        deployed = ev.deploy(config, choice.winner)
    except ev.NotDeployableError as exc:
        print(f"{exc}; {args.config} and {args.report} left unchanged", file=sys.stderr)
        return 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    args.config.write_text(cal.dumps_config(deployed), encoding="utf-8")
    print(f"wrote {args.report}; deployed_strategy = {choice.winner!r} in {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
