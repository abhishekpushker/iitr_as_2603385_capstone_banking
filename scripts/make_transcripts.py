"""Regenerate the graded transcripts (§ARCH 7.19). Run with `python -m scripts.make_transcripts`.

Each generator fills a `Transcript` with inputs, outputs and ✔/✘ checks from real assertions.
Every transcript is written even when a check fails; the exit code is then 1.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import get_args

import dataset
from credagent import text
from credagent.cache.response_cache import normalize_query
from credagent.kb.chunking import FIXED_PARAMS, STRATEGIES, Chunk, chunk_document, chunk_params
from credagent.kb.loader import DISCLAIMER, KbDocument, load_kb
from credagent.kb.version import kb_version
from credagent.llm.engine import MockEngine
from credagent.observability.metrics import Metrics
from credagent.rag import calibration as cal
from credagent.rag.embedder import get_embedder
from credagent.rag.generator import GroundedGenerator
from credagent.rag.retriever import Retriever, collections_from_settings
from credagent.rag.store import VectorStore, connect_http
from credagent.schemas import FALLBACK_ANSWER, LoanCategory, LoanStatus
from credagent.settings import Settings, get_settings
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
    t.check(
        "`config/retrieval.json` equals a fresh calibration byte for byte", committed == rendered
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


GENERATORS: dict[str, tuple[str, Generator]] = {
    "01_dataset_report": ("Dataset report (Task 1)", dataset_report),
    "02_kb_inventory": ("Knowledge-base inventory (Task 2)", kb_inventory),
    "03_chunking_and_index": ("Chunking and Chroma index (Task 3)", chunking_and_index),
    "04_threshold_calibration": ("Similarity-threshold calibration (Task 4)",
                                 threshold_calibration),
    "05_grounded_generation": ("Grounded generation with fallback (Task 4)",
                               grounded_generation),
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
