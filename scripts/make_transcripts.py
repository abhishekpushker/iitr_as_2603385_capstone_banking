"""Regenerate the graded transcripts (§ARCH 7.19). Run with `python -m scripts.make_transcripts`.

Each generator fills a `Transcript` with inputs, outputs and ✔/✘ checks from real assertions.
Every transcript is written even when a check fails; the exit code is then 1.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError

import dataset
from credagent import text
from credagent.cache.response_cache import normalize_query
from credagent.kb.chunking import FIXED_PARAMS, STRATEGIES, Chunk, chunk_document, chunk_params
from credagent.kb.loader import DISCLAIMER, KbDocument, load_kb
from credagent.kb.version import kb_version
from credagent.llm.engine import MockEngine
from credagent.observability.metrics import Metrics
from credagent.rag import calibration as cal
from credagent.rag import evaluation as ev
from credagent.rag.embedder import get_embedder
from credagent.rag.generator import GroundedGenerator
from credagent.rag.retriever import Retriever, collections_from_settings
from credagent.rag.store import VectorStore, connect_http
from credagent.schemas import FALLBACK_ANSWER, LOAN_STATUS_OUTPUT, LoanCategory, LoanStatus
from credagent.settings import Settings, get_settings
from credagent.tools import escalation as esc
from credagent.tools.ledger import ToolCallLedger
from credagent.tools.lookup_tool import LoanStatusTool
from scripts import validate_dataset
from scripts.indexer import CollectionNames
from scripts.transcript_kit import Transcript

COMMAND = "make transcripts  (python -m scripts.make_transcripts)"

Generator = Callable[[Transcript], None]


def dataset_report(t: Transcript) -> None:
    records = dataset.LOAN_APPLICATIONS
    rep = dataset.report(records)
    n = len(records)

    t.section("Inputs")
    t.para(
        f"`SEED = {dataset.SEED}` · `N_RECORDS = {dataset.N_RECORDS}` · "
        f"`FRAUD_FLAG_PROBABILITY = {dataset.FRAUD_FLAG_PROBABILITY}` (Bernoulli per record). "
        "Each record draws, from a local `random.Random(SEED)` and in this order: category, "
        "status, amount (log-uniform within the category range, rounded to the nearest "
        f"₹{dataset.AMOUNT_ROUNDING_INR:,}), `days_since_created = randint(0, "
        f"{dataset.MAX_DAYS_SINCE_CREATED})`, fraud flag."
    )
    t.table(
        ["Category", "Weight", "Amount range"],
        [
            [c, w, f"{validate_dataset.inr(lo)} – {validate_dataset.inr(hi)}"]
            for (c, w), (lo, hi) in zip(
                dataset.CATEGORY_WEIGHTS.items(), dataset.AMOUNT_RANGES_INR.values(), strict=True
            )
        ],
    )
    t.table(["Status", "Weight"], list(dataset.STATUS_WEIGHTS.items()))

    t.section("Seed history")
    history = validate_dataset.seed_history(validate_dataset.SEED_PROCEDURE_START, dataset.SEED)
    t.para(
        f"Procedure (§ARCH 7.1): start at {validate_dataset.SEED_PROCEDURE_START}, increment "
        "until `validate` returns no problems. Records are never edited. Recomputed here:"
    )
    t.table(
        ["Seed", "Result"],
        [[s, "valid" if not p else "invalid: " + "; ".join(p)] for s, p in history],
    )

    t.raw(validate_dataset.render_markdown(rep))

    t.section("Records")
    t.table(
        list(dataset.FIELDS),
        [
            [r["record_id"], r["category"], r["status"], r["loan_amount_inr"],
             r["days_since_created"], r["flagged_for_fraud_review"]]
            for r in records
        ],
    )  # fmt: skip

    categories = Counter(r["category"] for r in records)
    statuses = Counter(r["status"] for r in records)
    share = sum(r["flagged_for_fraud_review"] for r in records) / n
    t.check("`validate(LOAN_APPLICATIONS)` returns no problems", dataset.validate(records) == [])
    t.check(f"record count {n} ≥ 40", n >= 40)
    t.check(
        f"every category has ≥ 3 records (min {min(categories.values())})",
        set(categories) == set(dataset.CATEGORY_WEIGHTS) and min(categories.values()) >= 3,
    )
    t.check(
        f"every status has ≥ 1 record ({len(statuses)}/{len(dataset.STATUS_WEIGHTS)} present)",
        set(statuses) == set(dataset.STATUS_WEIGHTS),
    )
    t.check(f"fraud share {share:.2%} within [10%, 30%]", 0.10 <= share <= 0.30)
    ids = [r["record_id"] for r in records]
    t.check(
        "record ids unique and match `^LA-\\d{4}$`",
        len(set(ids)) == n and all(re.fullmatch(r"LA-\d{4}", i) for i in ids),
    )
    t.check(
        "field set and order fixed in every record",
        all(tuple(r) == dataset.FIELDS for r in records),
    )
    t.check(
        "every amount within its category range and ≤ 8 digits",
        all(
            dataset.AMOUNT_RANGES_INR[r["category"]][0]
            <= r["loan_amount_inr"]
            <= dataset.AMOUNT_RANGES_INR[r["category"]][1]
            and len(str(r["loan_amount_inr"])) <= 8
            for r in records
        ),
    )
    t.check(
        "every `days_since_created` is an int in [0, 30]",
        all(
            type(r["days_since_created"]) is int and 0 <= r["days_since_created"] <= 30
            for r in records
        ),
    )
    t.check(
        "`generate(SEED)` twice is identical and equals `LOAN_APPLICATIONS`",
        dataset.generate(dataset.SEED) == dataset.generate(dataset.SEED) == records,
    )
    t.check("`generate(SEED + 1)` differs", dataset.generate(dataset.SEED + 1) != records)
    t.check(
        f"SEED {dataset.SEED} is the first valid seed from {validate_dataset.SEED_PROCEDURE_START}",
        history[-1] == (dataset.SEED, []) and all(p for _, p in history[:-1]),
    )
    t.check(
        "schema literals `LoanCategory` / `LoanStatus` equal the dataset vocabularies",
        get_args(LoanCategory) == tuple(dataset.CATEGORY_WEIGHTS)
        and get_args(LoanStatus) == tuple(dataset.STATUS_WEIGHTS),
    )


# The brief's 12 required KB topics (Task 2), restated independently of the tests.
REQUIRED_TOPICS = (
    "Loan eligibility by type", "EMI rules", "Credit-card fee structure", "KYC requirements",
    "Fraud-dispute resolution", "Account closure", "Interest-rate slabs", "Prepayment penalties",
    "Minimum balance", "Credit-score impact", "Joint-account rules", "NRI eligibility",
)  # fmt: skip
SAMPLE_QUERY = "Is there a penalty for prepaying a personal loan?"
SAMPLE_EXPECTED_DOC = "kb-08-prepayment-penalty"


def kb_inventory(t: Transcript) -> None:
    s = t.settings
    docs = load_kb(s.kb_dir)
    sentences = {d.doc_id: text.split_sentences(d.body) for d in docs}

    t.section("Inputs")
    t.para(
        f"`{len(docs)}` Markdown files in `kb/`, loaded by `credagent.kb.loader.load_kb` "
        "(YAML `safe_load` front matter → frozen `KbDocument`, NFKC-normalised). Sentences are "
        "counted with the project splitter `credagent.text.split_sentences`. "
        f"`kb_version = {kb_version(s.kb_dir)}`."
    )

    t.section("Inventory")
    t.table(
        ["doc_id", "title", "topic", "sentences", "chars"],
        [[f"`{d.doc_id}`", d.title, d.topic, len(sentences[d.doc_id]), len(d.body)] for d in docs],
    )

    t.section("Required topic → document coverage")
    by_topic = {d.topic: d.doc_id for d in docs}
    t.table(
        ["#", "Required topic", "doc_id", "Covered"],
        [
            [i, topic, f"`{by_topic[topic]}`" if topic in by_topic else "—",
             "✔" if topic in by_topic else "✘"]
            for i, topic in enumerate(REQUIRED_TOPICS, start=1)
        ],
    )  # fmt: skip
    extra = [d for d in docs if d.topic not in REQUIRED_TOPICS]
    t.para("Additional document: " + ", ".join(f"`{d.doc_id}` ({d.topic})" for d in extra) + ".")

    t.section("Sentence split of each document")
    for d in docs:
        t.para(f"**`{d.doc_id}`**")
        t.raw("\n".join(f"{i}. {sent}" for i, sent in enumerate(sentences[d.doc_id], start=1)))

    counts = [len(v) for v in sentences.values()]
    ids = [d.doc_id for d in docs]
    t.check(f"exactly 13 documents ({len(docs)})", len(docs) == 13)
    t.check(
        f"all 12 required topics covered ({sum(x in by_topic for x in REQUIRED_TOPICS)}/12)",
        all(x in by_topic for x in REQUIRED_TOPICS),
    )
    t.check("each topic maps to exactly one document", len(by_topic) == len(docs))
    t.check(
        f"every body has 2–5 sentences (min {min(counts)}, max {max(counts)})",
        all(2 <= c <= 5 for c in counts),
    )
    t.check("doc_ids unique and sorted", ids == sorted(set(ids)))
    t.check("every document carries the disclaimer", all(d.disclaimer == DISCLAIMER for d in docs))
    t.check("every document is version 1", all(d.version == 1 for d in docs))


def _chunk_rows(chunks: Sequence[Chunk]) -> list[list[object]]:
    return [
        [f"`{c.chunk_id}`", f"{c.char_start}–{c.char_end}", len(c.text), c.text] for c in chunks
    ]


def _fixed_invariants_hold(doc: KbDocument, chunks: Sequence[Chunk]) -> bool:
    body = doc.body
    covered = {i for c in chunks for i in range(c.char_start, c.char_end)}
    return (
        all(body[c.char_start : c.char_end] == c.text and c.text.strip() == c.text for c in chunks)
        and all(i in covered or body[i].isspace() for i in range(len(body)))
        and all(b.char_start < a.char_end for a, b in zip(chunks, chunks[1:], strict=False))
        and all(
            (c.char_start == 0 or body[c.char_start - 1].isspace())
            and (c.char_end == len(body) or body[c.char_end].isspace())
            for c in chunks
        )
    )


def chunking_and_index(t: Transcript) -> None:
    s = t.settings
    docs = load_kb(s.kb_dir)
    version = kb_version(s.kb_dir)
    names = CollectionNames.from_settings(s)
    chunks = {st: [c for d in docs for c in chunk_document(d, st)] for st in STRATEGIES}

    t.section("Inputs")
    t.para(
        f"Fixed-size: `size={FIXED_PARAMS.size}`, `overlap={FIXED_PARAMS.overlap}`, word-boundary "
        f"snapping capped at `size+{FIXED_PARAMS.max_extend}`. Sentence-based: one chunk per "
        "sentence of `credagent.text.split_sentences` (abbreviations: "
        + ", ".join(f"`{a}.`" for a in chunk_params()["sentence"]["abbreviations"])
        + f"). Chunk ids `{{doc_id}}::{{strategy}}::{{index:03d}}`. `kb_version = {version}`."
    )

    t.section("Chunk counts")
    per_doc = {st: {d.doc_id: 0 for d in docs} for st in STRATEGIES}
    for st in STRATEGIES:
        for c in chunks[st]:
            per_doc[st][c.doc_id] += 1
    t.table(
        ["doc_id", "fixed", "sentence"],
        [[f"`{d.doc_id}`", per_doc["fixed"][d.doc_id], per_doc["sentence"][d.doc_id]] for d in docs]
        + [["**total**", f"**{len(chunks['fixed'])}**", f"**{len(chunks['sentence'])}**"]],
    )

    for st in STRATEGIES:
        t.section(f"Example chunks — {st}")
        t.table(["chunk_id", "span", "chars", "text"], _chunk_rows(chunks[st][:3]))

    client = connect_http(
        s.chroma_host, s.chroma_port, attempts=s.chroma_connect_attempts,
        backoff_cap_s=s.chroma_backoff_cap_s,
    )  # fmt: skip
    store = VectorStore(client)
    present = set(store.list_collections())
    stored = {st: store.ids(names.for_strategy(st)) for st in STRATEGIES}
    meta = store.read_meta(names.meta) or {}

    t.section("Collections read back from Chroma")
    t.table(
        ["collection", "strategy", "count"],
        [
            [f"`{names.for_strategy(st)}`", st, store.count(names.for_strategy(st))]
            for st in STRATEGIES
        ],
    )
    t.table(
        ["`" + names.meta + "` field", "value"], [[k, f"`{v}`"] for k, v in sorted(meta.items())]
    )

    t.section("Sample query")
    t.para(f"Query: **{SAMPLE_QUERY}** (embedded with `all-MiniLM-L6-v2`, top-3, cosine).")
    (vector,) = get_embedder().encode([SAMPLE_QUERY])
    results = {st: store.query(names.for_strategy(st), vector, k=3) for st in STRATEGIES}
    for st in STRATEGIES:
        t.para(f"**`{names.for_strategy(st)}`**")
        t.table(
            ["rank", "similarity", "doc_id", "chunk_id", "text"],
            [[r.rank, f"{r.similarity:.3f}", f"`{r.doc_id}`", f"`{r.chunk_id}`", r.text]
             for r in results[st]],
        )  # fmt: skip

    by_id = {d.doc_id: d for d in docs}
    fixed_by_doc = {d.doc_id: [c for c in chunks["fixed"] if c.doc_id == d.doc_id] for d in docs}
    for st in STRATEGIES:
        name = names.for_strategy(st)
        t.check(f"collection `{name}` exists", name in present)
        t.check(
            f"`{name}` holds exactly the {len(chunks[st])} chunk ids the chunker produces",
            stored[st] == {c.chunk_id for c in chunks[st]},
        )
    t.check(f"collection `{names.meta}` exists", names.meta in present)
    t.check(
        "`kb_meta.kb_version` equals the version computed from `kb/`",
        meta.get("kb_version") == version,
    )
    t.check(
        "`kb_meta` counts equal the collection counts",
        meta.get("count_fixed") == len(chunks["fixed"])
        and meta.get("count_sentence") == len(chunks["sentence"]),
    )
    t.check(
        "chunk ids unique per strategy",
        all(len({c.chunk_id for c in chunks[st]}) == len(chunks[st]) for st in STRATEGIES),
    )
    t.check(
        "fixed chunks: full coverage, consecutive overlap, no split words, exact spans",
        all(_fixed_invariants_hold(by_id[k], v) for k, v in fixed_by_doc.items()),
    )
    t.check(
        f"fixed chunks ≤ size+{FIXED_PARAMS.max_extend} chars",
        all(len(c.text) <= FIXED_PARAMS.size + FIXED_PARAMS.max_extend for c in chunks["fixed"]),
    )
    t.check(
        "sentence chunks equal `split_sentences(body)` and their spans reconstruct the body",
        all(
            [c.text for c in chunks["sentence"] if c.doc_id == d.doc_id]
            == text.split_sentences(d.body)
            and all(
                d.body[c.char_start : c.char_end] == c.text
                for c in chunks["sentence"]
                if c.doc_id == d.doc_id
            )
            for d in docs
        ),
    )
    for st in STRATEGIES:
        top = results[st]
        t.check(
            f"`{names.for_strategy(st)}`: sample-query top-1 doc is `{SAMPLE_EXPECTED_DOC}` "
            f"(similarity {top[0].similarity:.3f})",
            top[0].doc_id == SAMPLE_EXPECTED_DOC,
        )
        t.check(
            f"`{names.for_strategy(st)}`: 3 results sorted by similarity, descending",
            len(top) == 3
            and all(a.similarity >= b.similarity for a, b in zip(top, top[1:], strict=False)),
        )


# Task-4 demo set (§ARCH 7.5), disjoint from the calibration queries (test_query_sets.py).
DEMO_IN_SCOPE: tuple[tuple[str, str], ...] = (
    ("What is the annual fee on the credit card and when is it waived?", "kb-03-credit-card-fees"),
    ("Within how many days must I report an unauthorised transaction for zero liability?",
     "kb-05-fraud-dispute"),
    ("Can I close my account if I still have dues?", "kb-06-account-closure"),
    ("What interest rate does a personal loan carry for a credit score above 800?",
     "kb-07-interest-rate-slabs"),
    ("Is there a penalty for prepaying a home loan?", "kb-08-prepayment-penalty"),
)  # fmt: skip
DEMO_OUT_OF_SCOPE = "Can I get travel insurance with my card?"
# §ARCH 7.12.3: Phase 3 verifies that this near-domain question falls below the threshold.
GROUNDEDNESS_DEMO_QUERY = "What is the gold-loan late payment penalty?"


def _retriever(s: Settings) -> Retriever:
    client = connect_http(
        s.chroma_host, s.chroma_port, attempts=s.chroma_connect_attempts,
        backoff_cap_s=s.chroma_backoff_cap_s,
    )  # fmt: skip
    return Retriever(
        VectorStore(client), get_embedder(), collections_from_settings(s), top_k=s.top_k,
        metrics=Metrics(),
    )  # fmt: skip


def threshold_calibration(t: Transcript) -> None:
    s = t.settings
    retriever = _retriever(s)
    results = cal.calibrate(retriever, top_k=s.top_k)
    gated = results[cal.GATED_STRATEGY]

    t.section("Inputs")
    t.para(
        f"{len(cal.IN_SCOPE_QUERIES)} in-scope and {len(cal.OUT_OF_SCOPE_QUERIES)} out-of-scope "
        "calibration queries (`credagent.rag.calibration`), each embedded with "
        f"`{s.embed_model_path.name}` and sent through the production `Retriever` "
        f"(top-{s.top_k}, cosine) against **both** collections. Per collection: "
        "`max_oos` = highest out-of-scope top-1, `min_in` = lowest in-scope top-1, "
        "`gap = min_in − max_oos`, `threshold = round((max_oos + min_in) / 2, 3)`. "
        f"Gate: `gap ≥ {cal.MIN_GAP}` on `{cal.GATED_STRATEGY}` (the collection expected to be "
        f"deployed; Phase 4 confirms it). `kb_version = {kb_version(s.kb_dir)}`."
    )

    for strategy, c in results.items():
        t.section(f"Measured top-1 similarity — `{c.collection}` ({strategy})")
        rows: list[list[str]] = [
            ["in-scope", m.query, f"{m.top1_similarity:.3f}", f"`{m.top1_doc_id}`",
             "✔" if m.top1_doc_id == m.expected_doc_id else f"✘ (expected `{m.expected_doc_id}`)"]
            for m in sorted(c.in_scope, key=lambda m: -m.top1_similarity)
        ] + [
            ["out-of-scope", m.query, f"{m.top1_similarity:.3f}", f"`{m.top1_doc_id}`", "—"]
            for m in sorted(c.out_of_scope, key=lambda m: -m.top1_similarity)
        ]  # fmt: skip
        t.table(["kind", "query", "top-1 sim", "top-1 doc", "doc correct"], rows)

    t.section("Gap and threshold")
    t.table(
        ["collection", "max_oos", "min_in", "gap", "threshold", f"gap ≥ {cal.MIN_GAP}"],
        [
            [
                f"`{c.collection}`",
                f"{c.max_oos:.3f}",
                f"{c.min_in:.3f}",
                f"{c.gap:.3f}",
                "—" if c.threshold is None else f"{c.threshold:.3f}",
                "✔" if c.meets_min_gap else "✘",
            ]
            for c in results.values()
        ],
    )
    t.para(
        f"Deployed threshold: **{gated.threshold}** on `{gated.collection}`. A query whose top-1 "
        f'similarity is below it is answered with "{FALLBACK_ANSWER}"'
    )

    t.section("Near-domain verification for the groundedness demo (§ARCH 7.12.3)")
    probe = retriever.retrieve(GROUNDEDNESS_DEMO_QUERY, cal.GATED_STRATEGY)
    probe_top = probe.chunks[0].doc_id if probe.chunks else None
    margin = (gated.threshold or 0.0) - probe.top1_similarity
    t.para(
        f"**{GROUNDEDNESS_DEMO_QUERY}** → top-1 {probe.top1_similarity:.3f} (`{probe_top}`) "
        f"vs threshold {gated.threshold}, margin {margin:.3f}."
    )

    committed = s.retrieval_config_path.read_text(encoding="utf-8")
    rendered = cal.dumps_config(
        cal.build_config(
            results,
            kb_version=kb_version(s.kb_dir),
            top_k=s.top_k,
            chunk_params=chunk_params(),
            embed_model=s.embed_model_path.name,
        )
    )
    t.section("`config/retrieval.json`")
    t.code(committed, "json")

    demo = {normalize_query(q) for q, _ in DEMO_IN_SCOPE} | {
        normalize_query(q) for q in (DEMO_OUT_OF_SCOPE, GROUNDEDNESS_DEMO_QUERY, SAMPLE_QUERY)
    }
    calib = {normalize_query(q.query) for q in cal.IN_SCOPE_QUERIES} | {
        normalize_query(q) for q in cal.OUT_OF_SCOPE_QUERIES
    }
    t.check(
        f"≥ 5 in-scope calibration queries ({len(cal.IN_SCOPE_QUERIES)})",
        len(cal.IN_SCOPE_QUERIES) >= 5,
    )
    t.check(
        f"≥ 5 out-of-scope calibration queries ({len(cal.OUT_OF_SCOPE_QUERIES)})",
        len(cal.OUT_OF_SCOPE_QUERIES) >= 5,
    )
    t.check(
        "both collections measured for every query",
        all(
            len(c.in_scope) == len(cal.IN_SCOPE_QUERIES)
            and len(c.out_of_scope) == len(cal.OUT_OF_SCOPE_QUERIES)
            for c in results.values()
        ),
    )
    t.check(
        f"`{gated.collection}`: max_oos {gated.max_oos:.3f} < threshold {gated.threshold} "
        f"< min_in {gated.min_in:.3f}",
        gated.threshold is not None and gated.max_oos < gated.threshold < gated.min_in,
    )
    t.check(f"`{gated.collection}`: gap {gated.gap:.3f} ≥ 0.05", gated.gap >= 0.05)
    t.check(
        f"`{gated.collection}`: every in-scope query retrieves its document first "
        f"({gated.in_scope_top1_correct}/{len(gated.in_scope)})",
        gated.in_scope_top1_correct == len(gated.in_scope),
    )
    t.check(
        "calibration queries are disjoint from the demo and sample queries", calib.isdisjoint(demo)
    )
    # `deployed_strategy` is owned by scripts/eval_chunking.py (transcript 06, D-027).
    deployed = json.loads(committed).get("deployed_strategy")
    t.check(
        "`config/retrieval.json` equals a fresh calibration byte for byte, apart from "
        f"`deployed_strategy` (`{deployed}`, set by `eval_chunking`, transcript 06)",
        committed == cal.dumps_config({**json.loads(rendered), "deployed_strategy": deployed}),
    )
    t.check(
        f"groundedness demo query is below the threshold ({probe.top1_similarity:.3f} < "
        f"{gated.threshold})",
        gated.threshold is not None and probe.top1_similarity < gated.threshold,
    )


def grounded_generation(t: Transcript) -> None:
    s = t.settings
    config = cal.load_retrieval_config(s.retrieval_config_path)
    strategy = config.gated_strategy
    generator = GroundedGenerator(
        _retriever(s),
        MockEngine(),
        strategy=strategy,
        threshold=config.threshold_for(strategy),
        top_k=config.top_k,
        metrics=Metrics(),
    )
    engine = MockEngine()

    t.section("Inputs")
    t.para(
        f"Collection `{strategy}`, threshold **{generator.threshold}** and top-{generator.top_k} "
        "from `config/retrieval.json`. `GroundedGenerator` → `MockEngine.generate_grounded` "
        "(§ARCH 7.5): below the threshold → fallback; otherwise the retrieved chunks are split "
        "into sentences and deduplicated, each sentence is scored "
        "`0.7·query-token recall + 0.3·chunk similarity`, and the best 2 with score ≥ 0.3 are "
        "emitted verbatim in document order. The support rule (§ARCH 7.8: token recall ≥ 0.6 "
        "against the best context sentence and every number present in the context) is "
        "re-applied to every answer."
    )

    cases = [(q, d, True) for q, d in DEMO_IN_SCOPE] + [(DEMO_OUT_OF_SCOPE, None, False)]
    for n, (query, expected_doc, in_scope) in enumerate(cases, start=1):
        retrieval = generator.retrieve(query)
        answer = generator.generate(query, retrieval)
        context = [c.text for c in retrieval.chunks]
        support = engine.support_report(answer.answer, context)
        label = "in-scope" if in_scope else "out-of-scope"

        t.section(f"Q{n} ({label}): {query}")
        t.table(
            ["rank", "similarity", "doc_id", "chunk_id", "text"],
            [
                [c.rank, f"{c.similarity:.3f}", f"`{c.doc_id}`", f"`{c.chunk_id}`", c.text]
                for c in retrieval.chunks
            ],
        )
        t.para(
            f"top-1 {answer.top1_similarity:.3f} "
            f"{'≥' if answer.top1_similarity >= generator.threshold else '<'} threshold "
            f"{generator.threshold} → `grounded={answer.grounded}`"
        )
        t.para(f"**Answer:** {answer.answer}")
        t.para(
            "**Sources:** "
            + (", ".join(f"`{r.chunk_id}` ({r.similarity:.3f})" for r in answer.sources) or "none")
        )
        if in_scope:
            t.table(
                ["answer sentence", "best-match recall", "numbers missing", "supported"],
                [
                    [
                        x.sentence,
                        f"{x.recall:.2f}",
                        ", ".join(x.numbers_missing) or "—",
                        "✔" if x.supported else "✘",
                    ]
                    for x in support.sentences
                ],
            )
            sentences = {x for c in context for x in text.split_sentences(c)}
            t.check(
                f"Q{n}: grounded, non-fallback answer",
                answer.grounded and answer.answer != FALLBACK_ANSWER,
            )
            t.check(
                f"Q{n}: top-1 document is `{expected_doc}`",
                bool(retrieval.chunks) and retrieval.chunks[0].doc_id == expected_doc,
            )
            t.check(
                f"Q{n}: every answer sentence is verbatim from the retrieved context",
                set(text.split_sentences(answer.answer)) <= sentences,
            )
            t.check(
                f"Q{n}: no answer sentence repeats",
                len(set(text.split_sentences(answer.answer)))
                == len(text.split_sentences(answer.answer)),
            )
            t.check(
                f"Q{n}: support rule holds for {support.supported_count}/{support.total} sentences",
                support.total > 0 and support.all_supported,
            )
            t.check(
                f"Q{n}: sources are retrieved chunks",
                bool(answer.sources)
                and {r.chunk_id for r in answer.sources} <= {c.chunk_id for c in retrieval.chunks},
            )
        else:
            t.check(
                f"Q{n}: top-1 {answer.top1_similarity:.3f} is below the threshold",
                answer.top1_similarity < generator.threshold,
            )
            t.check(
                f'Q{n}: answer is exactly "{FALLBACK_ANSWER}"', answer.answer == FALLBACK_ANSWER
            )
            t.check(
                f"Q{n}: `grounded=False` and no sources",
                not answer.grounded and answer.sources == (),
            )


_DECIMAL = re.compile(r"\d+\.\d+")


def chunking_precision_recall(t: Transcript) -> None:
    s = t.settings
    committed = s.retrieval_config_path.read_text(encoding="utf-8")
    config = json.loads(committed)
    top_k = int(config["top_k"])
    results = ev.evaluate(_retriever(s), top_k=top_k)
    calibrations = {st: ev.calibration_of(config, st) for st in results}
    choice = ev.choose(results, gated=config["gated_strategy"], calibrations=calibrations)
    calib = calibrations[choice.winner]
    min_gap = float(config["min_gap"])
    rec = ev.recommendation(results, choice, calibrations, min_gap=min_gap, top_k=top_k)
    report = ev.render_report(results, choice, rec, top_k=top_k)

    t.section("Inputs")
    t.para(
        f"{len(ev.PR_QUERIES)} in-scope queries with gold `relevant_doc_ids` "
        "(`credagent.rag.evaluation.PR_QUERIES`, disjoint from the calibration set), run through "
        f"the production `Retriever` (top-{top_k}, cosine) on **both** collections. Per query: "
        "top-k chunks → `doc_id` → deduplicated in rank order → `P = |rel ∩ ret| / |ret|`, "
        "`R = |rel ∩ ret| / |rel|` (`P := 0` and a warning when nothing is retrieved). Macro "
        "averages per collection. Only strategies whose calibration gap meets `min_gap` compete "
        "(D-028); among those: higher macro F1, then higher mean top-1 similarity, then fewer "
        "documents retrieved per query, then the gated strategy (D-027)."
    )
    t.table(
        ["#", "query", "relevant_doc_ids"],
        [[i, q.query, ", ".join(f"`{d}`" for d in q.relevant_doc_ids)]
         for i, q in enumerate(ev.PR_QUERIES, start=1)],
    )  # fmt: skip

    for st in sorted(results):
        c = results[st]
        t.section(f"Per-query retrieval — `{c.collection}` ({st})")
        for i, qe in enumerate(c.queries, start=1):
            t.para(f"**Q{i}: {qe.query.query}**")
            t.table(
                ["rank", "similarity", "doc_id", "chunk_id", "relevant"],
                [[ch.rank, f"{ch.similarity:.3f}", f"`{ch.doc_id}`", f"`{ch.chunk_id}`",
                  "✔" if ch.doc_id in qe.query.relevant_doc_ids else "—"]
                 for ch in qe.chunks],
            )  # fmt: skip
            t.para(
                f"retrieved docs (dedup): {', '.join(qe.score.retrieved) or '—'} → "
                f"`{qe.score.precision_arithmetic}`, `{qe.score.recall_arithmetic}`"
            )
        t.section(f"P/R table — `{c.collection}` ({st})")
        t.table(ev.PR_HEADERS, ev.pr_rows(c))
        t.para(f"`{c.macro_precision_arithmetic}`  \n`{c.macro_recall_arithmetic}`")

    t.section("Macro averages")
    t.table(ev.AVERAGE_HEADERS, ev.average_rows(results))
    t.table(
        ["strategy", "calibration gap", f"≥ min_gap {min_gap}", "competes"],
        [[f"`{st}`", f"{c.gap:.3f}", "✔" if c.meets_min_gap else "✘",
          "no" if st in choice.excluded else "yes"]
         for st, c in sorted(calibrations.items())],
    )  # fmt: skip
    t.para(
        f"Winner: **`{choice.winner}`** (decided by `{choice.decided_by}`); other collection "
        f"`{choice.runner_up}`."
    )

    t.section("Recommendation")
    t.para(rec)

    t.section("Deployment")
    threshold = config["collections"][choice.winner]["threshold"]
    t.para(
        f"`config/retrieval.json`: `deployed_strategy = {json.dumps(config['deployed_strategy'])}`"
        f" → collection `{results[choice.winner].collection}`, threshold `{threshold}` "
        f"(calibration gap {calib.gap:.3f}, `meets_min_gap = {str(calib.meets_min_gap).lower()}`)."
    )

    calib_queries = {normalize_query(q.query) for q in cal.IN_SCOPE_QUERIES} | {
        normalize_query(q) for q in cal.OUT_OF_SCOPE_QUERIES
    }
    pr_queries = {normalize_query(q.query) for q in ev.PR_QUERIES}
    t.check(f"exactly 5 P/R queries ({len(ev.PR_QUERIES)})", len(pr_queries) == 5)
    two = sum(len(q.relevant_doc_ids) == 2 for q in ev.PR_QUERIES)
    t.check(f"≥ 2 queries have two relevant documents ({two})", two >= 2)
    t.check("P/R queries are disjoint from the calibration queries",
            pr_queries.isdisjoint(calib_queries))  # fmt: skip
    for st, c in sorted(results.items()):
        t.check(
            f"`{c.collection}`: every in-scope query retrieves ≥ 1 document (no P := 0 case)",
            all(qe.score.retrieved for qe in c.queries),
        )
        t.check(
            f"`{c.collection}`: every P and R recomputed from the chunk table matches",
            all(
                qe.score.retrieved == tuple(dict.fromkeys(ch.doc_id for ch in qe.chunks))
                and qe.score.precision
                == Fraction(
                    len(set(qe.score.retrieved) & set(qe.query.relevant_doc_ids)),
                    len(qe.score.retrieved),
                )
                and qe.score.recall
                == Fraction(
                    len(set(qe.score.retrieved) & set(qe.query.relevant_doc_ids)),
                    len(qe.query.relevant_doc_ids),
                )
                for qe in c.queries
            ),
        )
        t.check(
            f"`{c.collection}` ({st}): macro P {float(c.macro_precision):.3f} and macro R "
            f"{float(c.macro_recall):.3f} are the means of the per-query values",
            c.macro_precision == sum(q.score.precision for q in c.queries) / len(c.queries)
            and c.macro_recall == sum(q.score.recall for q in c.queries) / len(c.queries),
        )
    w, lo = results[choice.winner], results[choice.runner_up]
    numbers = _DECIMAL.findall(rec)
    t.check(
        f"recommendation has 2–3 sentences ({len(text.split_sentences(rec))})",
        2 <= len(text.split_sentences(rec)) <= 3,
    )
    t.check(
        "numbers parsed back from the recommendation equal the measured macro P/R and gap",
        numbers[:4]
        == [f"{float(x):.3f}" for x in
            (w.macro_precision, w.macro_recall, lo.macro_precision, lo.macro_recall)]
        and f"{calib.gap:.3f}" in numbers,
    )  # fmt: skip
    below = tuple(sorted(st for st, c in calibrations.items() if not c.meets_min_gap))
    t.check(
        "exactly the strategies below the calibration gap were excluded before the comparison ("
        + (", ".join(f"`{st}` gap {calibrations[st].gap:.3f}" for st in below) or "none")
        + ")",
        choice.excluded == below and choice.winner not in below,
    )
    t.check(
        f"winner `{choice.winner}` meets the calibration gap ({calib.gap:.3f} ≥ {min_gap})",
        calib.meets_min_gap and calib.gap >= min_gap,
    )
    t.check(
        f"`config/retrieval.json` deploys the winner (`deployed_strategy = "
        f"{json.dumps(config['deployed_strategy'])}`)",
        config["deployed_strategy"] == choice.winner,
    )
    t.check(
        "`eval/results/chunking_pr.md` equals the freshly rendered report byte for byte",
        s.chunking_report_path.is_file()
        and s.chunking_report_path.read_text(encoding="utf-8") == report,
    )


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _valid_lookup_output(out: str) -> bool:
    try:
        LOAN_STATUS_OUTPUT.validate_json(out)
    except ValidationError:
        return False
    return True


def lookup_and_escalation(t: Transcript) -> None:
    s = t.settings
    records = dataset.LOAN_APPLICATIONS
    committed_text = s.escalation_config_path.read_text(encoding="utf-8")
    cfg = esc.load_escalation_config(s.escalation_config_path)
    fresh = esc.build_config(records)
    days = sorted(r["days_since_created"] for r in records)
    n = len(days)
    rank = math.ceil(Fraction(cfg.percentile, 100) * n)
    p80, threshold = cfg.p80_days_since_created, cfg.threshold
    flagged_cut, unflagged_cut = cfg.flagged_min_days, cfg.unflagged_min_days
    formula_cut = 30 * (threshold - cfg.flag_weight) / cfg.recency_weight

    t.section("Formula")
    t.code(
        "recency          = min(max(days_since_created, 0), 30) / 30\n"
        f"escalation_score = round({cfg.flag_weight} · flagged + {cfg.recency_weight} · recency, "
        "4)        ∈ [0, 1]\n"
        f"T                = round({cfg.recency_weight} · P80(days_since_created) / 30, 4)\n"
        "escalate         ⇔ escalation_score ≥ T"
    )
    t.para(
        "P80 is the **nearest-rank** 80th percentile, `sorted(days)[ceil(0.8 · n) − 1]`, taken "
        "over the generated dataset (`dataset.LOAN_APPLICATIONS`) by "
        "`scripts/build_escalation_config.py`, which writes `config/escalation.json`. The lookup "
        "tool reads the weights and T back from that file."
    )

    t.section("P80 and T from the dataset")
    t.para(f"`days_since_created`, sorted (n = {n}):")
    t.code(", ".join(str(d) for d in days))
    t.para(
        f"rank = ceil(0.8 × {n}) = {rank} → P80 = sorted(days)[{rank - 1}] = **{p80} days**  \n"
        f"T = round({cfg.recency_weight} × {p80} / 30, 4) = **{threshold}**"
    )
    t.code(committed_text, "json")

    t.section("What T means")
    older = sum(d >= p80 for d in days)
    t.para(
        f"- **Unflagged** applications score `0.6 · days/30` and escalate from **day "
        f"{unflagged_cut}** — the P80 age. Those are the oldest applications in the queue: "
        f"{older} of {n} ({older / n:.1%}) are at least {p80} days old (ties at {p80} days push "
        "this above 20%). This is the stale-queue rule.\n"
        f"- **Flagged** applications start at {cfg.flag_weight} and escalate once "
        f"`days ≥ 30 · (T − {cfg.flag_weight}) / {cfg.recency_weight}` = "
        f"30 × ({threshold} − {cfg.flag_weight}) / {cfg.recency_weight} = {formula_cut:.1f}, "
        f"i.e. from **day {flagged_cut}**, so a fraud flag is reviewed within days.\n"
        f"- It is a weighted blend, not a boolean OR: a brand-new flagged application scores "
        f"{cfg.score(0, True)} < T = {threshold} and does **not** auto-escalate.\n"
        "- Known limitation: the score ignores status, so a flagged, already disbursed loan still "
        "escalates on age alone."
    )
    t.table(
        ["days", "unflagged score", "escalates", "flagged score", "escalates"],
        [
            [d, cfg.score(d, False), _yes(cfg.score(d, False) >= threshold),
             cfg.score(d, True), _yes(cfg.score(d, True) >= threshold)]
            for d in sorted({0, 1, flagged_cut or 0, (flagged_cut or 0) + 1, 10, 15, 20,
                             p80 - 1, p80, 30})
        ],
    )  # fmt: skip
    escalated = [
        r
        for r in records
        if cfg.score(r["days_since_created"], r["flagged_for_fraud_review"]) >= threshold
    ]
    esc_flagged = sum(r["flagged_for_fraud_review"] for r in escalated)
    n_flagged = sum(r["flagged_for_fraud_review"] for r in records)
    t.para(
        f"Across the dataset {len(escalated)} of {n} applications are recommended for escalation: "
        f"{esc_flagged} of {n_flagged} flagged and {len(escalated) - esc_flagged} of "
        f"{n - n_flagged} unflagged."
    )

    _sample_lookups(t, cfg, records)

    t.check(
        f"P80 = sorted(days)[ceil(0.8·{n}) − 1] = sorted(days)[{rank - 1}] = {days[rank - 1]} "
        f"equals config p80 {p80}",
        days[rank - 1] == p80 == esc.percentile_nearest_rank(days, 80),
    )
    t.check(
        f"T = round(0.6 × {p80} / 30, 4) = {round(0.6 * p80 / 30, 4)} equals config T {threshold}",
        round(0.6 * p80 / 30, 4) == threshold,
    )
    t.check(
        "`config/escalation.json` equals a fresh build from the dataset byte for byte",
        committed_text == esc.dumps_config(fresh),
    )
    t.check(
        f"config dataset sha256 matches the dataset ({cfg.dataset_sha256[:12]}…)",
        cfg.dataset_sha256 == dataset.sha256(records),
    )
    t.check(
        f"unflagged cut-off day {unflagged_cut} equals P80 {p80}",
        unflagged_cut == p80 and cfg.score(p80, False) >= threshold > cfg.score(p80 - 1, False),
    )
    t.check(
        f"flagged cut-off day {flagged_cut} = ceil(30·(T − 0.4)/0.6) = ceil({formula_cut:.4f})",
        flagged_cut == max(0, math.ceil(round(formula_cut, 9))),
    )
    grid = [(d, f, cfg.score(d, f)) for d in range(31) for f in (False, True)]
    t.check(
        "every score over 0–30 days × {unflagged, flagged} is within [0, 1]",
        all(0.0 <= sc <= 1.0 for *_, sc in grid),
    )
    t.check(
        "score strictly increases with days and with the flag",
        all(cfg.score(d + 1, f) > cfg.score(d, f) for d in range(30) for f in (False, True))
        and all(cfg.score(d, True) > cfg.score(d, False) for d in range(31)),
    )
    t.check(
        f"a brand-new flagged application ({cfg.score(0, True)}) does not escalate",
        cfg.score(0, True) < threshold,
    )


def _sample_lookups(
    t: Transcript, cfg: esc.EscalationConfig, records: Sequence[dict[str, Any]]
) -> None:
    """Five lookups chosen by rule from the dataset, run through the owner-bound tool."""
    threshold, p80 = cfg.threshold, cfg.p80_days_since_created
    flagged_cut = cfg.flagged_min_days or 0
    t.section("Sample lookups")
    first_flagged_at_cut = next(r for r in records if r["flagged_for_fraud_review"]
                                and r["days_since_created"] == flagged_cut)  # fmt: skip
    first_flagged_young = next(r for r in records if r["flagged_for_fraud_review"]
                               and r["days_since_created"] < flagged_cut)  # fmt: skip
    first_unflagged_at_p80 = next(r for r in records if not r["flagged_for_fraud_review"]
                                  and r["days_since_created"] == p80)  # fmt: skip
    samples: list[tuple[str, str, Callable[[dict[str, object]], bool]]] = [
        (first_flagged_at_cut["record_id"],
         f"flagged, {flagged_cut} days: exactly at the flagged cut-off → escalated",
         lambda o: o["found"] is True and o["escalation_recommended"] is True
         and o["escalation_score"] == threshold),
        (f"  {first_flagged_young['record_id'].lower()} ",
         f"lower-case, padded; flagged but only {first_flagged_young['days_since_created']} days "
         "old → not escalated",
         lambda o: o["found"] is True and o["record_id"] == first_flagged_young["record_id"]
         and o["escalation_recommended"] is False),
        (first_unflagged_at_p80["record_id"],
         f"unflagged, {p80} days (the P80 age) → escalated by the stale-queue rule",
         lambda o: o["found"] is True and o["escalation_recommended"] is True
         and o["flagged_for_fraud_review"] is False),
        ("LA-9999", "well-formed but unknown → not_found",
         lambda o: o == {"found": False, "error": "not_found"}),
        ("LA-0001' OR '1'='1", "SQL-style injection string → invalid_record_id",
         lambda o: o == {"found": False, "error": "invalid_record_id"}),
    ]  # fmt: skip
    ledger = ToolCallLedger()
    tool = LoanStatusTool(owner="lookup_agent", ledger=ledger, config=cfg)
    outputs = []
    for i, (raw, why, _) in enumerate(samples, start=1):
        out = tool._run(raw)
        outputs.append(out)
        t.para(f"**L{i}** — input `{raw!r}` — {why}")
        t.code(json.dumps(json.loads(out), indent=2), "json")
    t.para("Tool-call ledger (`ToolCallLedger`):")
    t.table(
        ["#", "agent", "tool", "args", "cache_hit"],
        [[i, e.agent, e.tool, f"`{json.dumps(e.args)}`", e.cache_hit]
         for i, e in enumerate(ledger.entries(), start=1)],
    )  # fmt: skip
    for i, ((_, why, expect), out) in enumerate(zip(samples, outputs, strict=True), start=1):
        t.check(
            f"L{i}: {why} (output validates against the output model)",
            _valid_lookup_output(out) and expect(json.loads(out)),
        )
    t.check(
        "the ledger records all 5 calls by `lookup_agent` on `check_loan_application_status`",
        [(e.agent, e.tool, e.args["record_id"]) for e in ledger.entries()]
        == [("lookup_agent", "check_loan_application_status", raw) for raw, *_ in samples],
    )


GENERATORS: dict[str, tuple[str, Generator]] = {
    "01_dataset_report": ("Dataset report (Task 1)", dataset_report),
    "02_kb_inventory": ("Knowledge-base inventory (Task 2)", kb_inventory),
    "03_chunking_and_index": ("Chunking and Chroma index (Task 3)", chunking_and_index),
    "04_threshold_calibration": ("Similarity-threshold calibration (Task 4)",
                                 threshold_calibration),
    "05_grounded_generation": ("Grounded generation with fallback (Task 4)",
                               grounded_generation),
    "06_chunking_precision_recall": ("Chunking evaluation: precision / recall (Task 5)",
                                     chunking_precision_recall),
    "07_lookup_and_escalation": ("Loan lookup tool and escalation threshold (Task 6)",
                                 lookup_and_escalation),
}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("transcripts"))
    parser.add_argument("--only", action="append", choices=sorted(GENERATORS), default=None)
    args = parser.parse_args(argv)
    try:
        settings = get_settings()
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise SystemExit(f"refusing to run: {exc}") from exc

    args.out.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for name in args.only or sorted(GENERATORS):
        title, generate = GENERATORS[name]
        transcript = Transcript(name, title, command=COMMAND, settings=settings)
        generate(transcript)
        (args.out / f"{name}.md").write_text(transcript.render(), encoding="utf-8")
        failed += [f"{name}: ✘ {label}" for label in transcript.failed]
        print(f"{name}: {'FAILED' if transcript.failed else 'ok'}")
    for line in failed:
        print(line, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
