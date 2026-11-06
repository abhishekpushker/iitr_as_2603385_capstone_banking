"""Calibrate the similarity threshold (§ARCH 7.5, Task 4). `python -m scripts.calibrate_threshold`.

Measures the top-1 cosine of every calibration query on both collections through the production
retriever, prints the table, and writes `config/retrieval.json`. Exits 1 without writing if the
gated collection's gap is below MIN_GAP: a threshold is never chosen inside an overlap.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from credagent.kb.chunking import chunk_params
from credagent.kb.version import kb_version
from credagent.observability.metrics import Metrics
from credagent.rag import calibration as cal
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
    parser.add_argument("--out", type=Path, default=s.retrieval_config_path)
    args = parser.parse_args(argv)

    client = connect_http(
        s.chroma_host, s.chroma_port, attempts=s.chroma_connect_attempts,
        backoff_cap_s=s.chroma_backoff_cap_s,
    )  # fmt: skip
    retriever = Retriever(
        VectorStore(client), get_embedder(), collections_from_settings(s), top_k=s.top_k,
        metrics=Metrics(),
    )  # fmt: skip
    results = cal.calibrate(retriever, top_k=s.top_k)
    for strategy, c in results.items():
        print(f"[{strategy}] {c.collection}")
        for kind, rows in (("in ", c.in_scope), ("oos", c.out_of_scope)):
            for m in rows:
                print(f"  {kind} {m.top1_similarity:.3f}  {m.top1_doc_id}  {m.query}")
        print(
            f"  max_oos={c.max_oos:.3f} min_in={c.min_in:.3f} gap={c.gap:.3f} "
            f"threshold={c.threshold} meets_min_gap={c.meets_min_gap}"
        )

    gated = results[cal.GATED_STRATEGY]
    if not gated.meets_min_gap:
        print(
            f"gap {gated.gap:.3f} < {cal.MIN_GAP} on {cal.GATED_STRATEGY}: not writing "
            f"{args.out} (revise KB wording or query design, §ARCH 7.5)",
            file=sys.stderr,
        )
        return 1
    config = cal.build_config(
        results, kb_version=kb_version(s.kb_dir), top_k=s.top_k, chunk_params=chunk_params(),
        embed_model=s.embed_model_path.name,
    )  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(cal.dumps_config(config), encoding="utf-8")
    print(f"wrote {args.out} (threshold {gated.threshold} on {cal.GATED_STRATEGY})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
