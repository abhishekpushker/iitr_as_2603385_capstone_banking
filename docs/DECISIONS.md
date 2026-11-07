# Decisions

Each entry: what was decided, why, and where it is enforced. Newest last. Library facts are
pinned by contract tests in `tests/contract/`; a failing contract test means re-read the entry.

## Phase 0 — foundation, toolchain, dependency contracts (2026-11-07)

### D-001 Docs live in `docs/`
The architecture and plan arrived at the repo root, but every reference (§ARCH 4, CLAUDE.md)
points to `docs/`. Moved `ARCHITECTURE.md` and `IMPLEMENTATION_PLAN.md` into `docs/`;
`CLAUDE.md` stays at the root.

### D-002 Pinned base image
`python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df`
(multi-arch index digest; Python 3.12.16). Pinned in `docker/Dockerfile` (`PYTHON_IMAGE`) and the
`make lock` builder.

### D-003 Pinned embedding model revision
`sentence-transformers/all-MiniLM-L6-v2` at commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
(HF `main` on 2026-11-07, last modified 2026-06-01). Downloaded only in the `model` build stage,
`save()`d to `/models/all-MiniLM-L6-v2`, with the SHA written to `REVISION` there and exported
as `EMBED_MODEL_REVISION`; `tests/contract/test_image.py` asserts both agree.

### D-004 Chroma server image and healthcheck
`chromadb/chroma:1.5.9@sha256:1e0b73a187a28757c572acba508c46f48c9e8b0acaf5c20e6d95cdedce1acdf6`.
The image has no curl, wget, python or nc, but ships `bash`, so the compose healthcheck is an
HTTP `GET /api/v2/heartbeat` over bash's `/dev/tcp`. Dependants use `service_healthy`
(§ARCH 3.5's `service_started` fallback is not needed). The server persists to `/data`
(`/config.yaml: persist_path`), mounted from the `chroma-data` volume.

### D-005 Lockfiles
* `uv 0.13.7` in the pinned python image compiles both locks with `--generate-hashes
  --emit-index-url --index-strategy unsafe-best-match --python-platform x86_64-manylinux_2_28`.
  `unsafe-best-match` is required because the PyTorch CPU index (extra index) also mirrors
  common packages such as numpy; first-index resolution would pin those to its older copies.
* torch is pinned as `2.15.0+cpu`: no CUDA/nvidia/triton wheels in the lock (checked).
* The lock targets linux/amd64 (the dev host is x86_64). An arm64 host would need a second lock.
* `requirements-dev.in` is constrained with `-c requirements.lock`, so shared packages never
  diverge between the runtime and dev images.
* `setuptools` is in the runtime lock so `pip install --no-deps --no-build-isolation .` builds
  the package from hash-checked inputs instead of fetching an unpinned build backend.
* Resolved: crewai 1.15.36, chromadb 1.5.9, autogen-agentchat/core 0.7.5, langchain-core 1.7.6,
  sentence-transformers 6.2.0, torch 2.15.0+cpu, transformers 5.22.2, fastapi 0.162.0,
  starlette 1.10.0, uvicorn 0.54.0, websockets 17.4, pydantic 2.12.5, pydantic-settings 2.16.2,
  numpy 2.5.4, anyio 4.16.0. `litellm` is not pulled (as §ARCH 2.1 says); `openai`, `onnxruntime`,
  `lancedb`, `pyarrow` and `kubernetes` are, transitively.

### D-006 Image size
Target < 2.5 GB (§PLAN Phase 0). Measured (see `make image-size` and the gate log):
filesystem content of `credagent:runtime` = **2,307 MB** (`du -smx /`), after removing
`torch/include` and `torch/test` (≈ 170 MB, unused: no C++ extension builds) — **meets** the
target. Docker Desktop's containerd image store reports a larger "Size", which also counts the
compressed layer blobs: runtime **3.19 GB**, dev **3.33 GB** (3.41 / 3.55 GB before the strip).
By that metric the target is **not** met.
The bulk is the transitive closure: torch CPU 809 MB, lancedb 184 MB + pyarrow 170 MB (crewai),
transformers 120 MB, scipy 109 MB, kubernetes 83 MB (chromadb), onnxruntime 68 MB (chromadb),
~400 MB of bytecode (kept: the root fs is read-only, so .pyc cannot be written at runtime).
Removing transitive packages from a `--no-deps` lock would break `pip check` and is not done.

### D-007 `dataset.py` placeholder
`pyproject.toml` declares `py-modules=["dataset"]` (Phase 0) but the dataset is Phase 1 work, so
`dataset.py` is a docstring-only module until then. The layering test already asserts it
imports nothing from `credagent`.

### D-008 The `tests` service mounts the repo read-only at `/app`
`.:/app:ro` plus `PYTHONPATH=/app/src:/app` in the dev image: the mounted sources win over the
copy installed in site-packages, so edit → test needs no rebuild. Caches (`pytest`, `mypy`,
`ruff`, coverage) go to `/tmp` (tmpfs). `make format` is the only writer and uses a plain
`docker run --network none` with the host uid, outside compose.

### D-009 Layers with no tests yet
`ci/run_layer.sh` reports a layer as `N/A` (exit 0) only when *no* test file carries it — no
`tests/<layer>/test_*.py` and no explicit `pytest.mark.<layer>`. Once a layer has a test file,
pytest's exit code is passed through unchanged, including 5 ("no tests collected"). In Phase 0
this applies to `e2e` and `golden`. This is not skipping: there is nothing to run.

### D-010 Services that land later
`indexer` (`scripts.indexer`, Phase 2) and `api`/`api-offline` (`credagent.api.app`, Phase 10)
are defined in `compose.yaml` now but cannot start yet. Phase 0 layers start only `chroma`.
Only `api` (runtime) and `tests` (dev) have a `build:` section; the other runtime services use
`image: credagent:runtime` with `pull_policy: never`, so one `make build` exports each image once.

### D-011 Egress flags: `setdefault` in the package, enforcement in Settings
`credagent/__init__.py` `setdefault`s the telemetry/offline flags (§ARCH 5), so importing the
package before crewai is enough. An explicit unsafe value (e.g. `CREWAI_TRACING_ENABLED=true`)
survives the `setdefault`, and `Settings()` then refuses to construct; every script calls
`get_settings()` first. Tested in `tests/unit/test_package_init.py` and `test_settings.py`.

### D-012 Settings defaults for API keys
`API_KEY` / `ADMIN_API_KEY` default to the same dev placeholders as `.env.example`
(`SecretStr`, never in `repr`). There is no secret of value anywhere (§ARCH 9).

### D-013 Library facts confirmed as written in §ARCH 2
Pinned by contract tests: BaseLLM's single abstract `call` and its signature; Agent defaults
(`cache=True`, `allow_delegation=False`, `max_iter=25`, `max_retry_limit=2`); Crew defaults
(`cache/memory/planning=False`); the ReAct layout after one tool call (template "Observation: the
result of the action" in the system message; our assistant message carries `\nObservation:
<tool result>`; then a user message equal to `post_tool_reasoning`); `output_pydantic` filled in
exactly 2 calls; kickoff with stdin closed in < 20 s; ceil(total_chars/4) token estimate (logged
as a warning); the Autogen abstract members, RR(max_turns=2) message shape, stop reason and
`json_output=Verdict`; Chroma cosine distance = 1 − cos (numpy-checked); `embedding_function=None`
never falls back to ONNX; langchain-core < 2.0 with both deprecation warnings.

### D-014 Library facts that refine or differ from §ARCH 2 (§2 amended with human approval, 2026-11-07)
1. **`supports_function_calling` (§2.2.1).** `BaseLLM` in crewai 1.15.36 does **not** define
   `supports_function_calling()`. CrewAI uses native tool calling only when the LLM has a
   callable one returning True (`utilities/agent_utils.py`), so a custom LLM is on the text
   ReAct path either way. But the output **converter** calls `llm.supports_function_calling()`
   unconditionally. The adapter must define it (returning `False`), as §7.8.1 already requires.
2. **Converter re-ask (§2.2.7).** With that method defined, a Final Answer that is not schema
   JSON triggers **3 converter re-asks** (`Converter.max_attempts`). That makes 4 LLM calls in
   total, each re-ask laid out as `[system="Format your final answer according to…", user=<raw
   output>]`. CrewAI then logs "Converter could not convert…", keeps `raw`, and sets
   `pydantic=None`. Consequences for Phase 6: the mock must always emit valid schema JSON; the
   exact call-count assertions catch a regression; the orchestrator's own
   `CrewResponse.model_validate_json(result.raw)` stays the real gate.
3. **`response_model` (§2.2.1, §7.11).** A tool-less agent with `output_pydantic` (the
   Composer) receives `response_model=<the model>` in `call()`. A tool-using agent receives
   `None`.
4. **Task context hand-off.** A task's `context` reaches the next agent inside its "Current Task"
   user message, after `This is the context you're working with:\n`, as the prior raw output.
   That supports §7.8.1 step 4 (find prior outputs by JSON `kind`).
5. **Autogen without `custom_message_types` (§2.3.4).** `team.run()` raises **`RuntimeError`**,
   whose message carries the registry's `ValueError: Message type …StructuredMessage[Verdict]…
   is not registered.` A bare `ValueError` is not what reaches the caller. The design (always pass
   `custom_message_types`) is unchanged; only the wording of §2.3.4 needs amending.
6. **WebSocket send after disconnect (§2.4).** uvicorn `ws="auto"` resolves to
   `WebSocketsSansIOProtocol`. Receive after a client close raises
   `WebSocketDisconnect(code=1000)`. The **first** send raises `WebSocketDisconnect(code=1006)`.
   Any **later** send raises `starlette.websockets.WebSocketDisconnected`, a `RuntimeError`
   subclass. The WS loop (Phase 10) must catch both.
7. **LangChain deprecation warnings.** `InMemoryChatMessageHistory` warns **once per process**
   (its decorator keeps a "warned" flag). `RunnableWithMessageHistory` warns on **every**
   construction. The contract test therefore checks them in a fresh interpreter.
8. **Import-time sockets.** No library makes a DNS lookup or a connection at import. With every
   socket blocked, all imports still succeed. The only code that even creates a socket is
   urllib3's IPv6 capability probe (`_has_ipv6`: binds an AF_INET6 socket to `::1`, loopback, no
   traffic). The contract test allows exactly that one site.

### D-015 Chroma server version is not comparable to the client version
The pinned image `chromadb/chroma:1.5.9` reports `1.0.0` from `/api/v2/version`
(`HttpClient.get_version()`), and its CLI says `chroma 1.4.4`. The Rust server is versioned
independently of the Python package, so `test_server_version_equals_client_version` (P8, §2.1)
cannot pass as written. It was left failing (no assertion weakened) and escalated per the stop rule.
**Decided by the human (2026-11-07): round-trip only.** The test is replaced by
`test_client_and_server_interoperate_over_http`. It exercises every call the vector store makes
against the real server: heartbeat, cosine collection with `embedding_function=None`,
configuration read-back, idempotent upsert, query distances vs numpy, metadata, delete, reopen.
No version strings are compared. The client stays pinned at 1.5.9 (`test_client_version_is_pinned`)
and the server image by digest (D-004). §ARCH 2.1 and P8 are amended accordingly.

## Phase 1 — dataset (2026-11-07)

### D-016 Generator details within §ARCH 7.1
* Per record, a local `random.Random(seed)` draws in this fixed order: category
  (`rng.choices` with `CATEGORY_WEIGHTS`), status (`STATUS_WEIGHTS`), amount
  (`exp(uniform(ln lo, ln hi))` rounded to the nearest ₹5,000, then clamped to `[lo, hi]`; the
  clamp never changes a value because both bounds are multiples of 5,000), `randint(0, 30)` days,
  then the fraud flag (`rng.random() < 0.18`). Changing this order changes the dataset and its
  golden hash.
* Stdlib only, as §7.1 says. Besides the modules listed there, `math` (log-uniform), `re` (id
  pattern), `sys` (exit code) and `typing` are used. A unit test asserts every import is in
  `sys.stdlib_module_names`.
* `validate` checks more than the §7.1 minimum because the same rules guard Phase 5's lookup
  tool: exact field order (`dataset.FIELDS`), exact types (`bool` is not accepted as `int`),
  known vocabularies, days in [0, 30], and an id pattern of `^LA-\d{4}$`.
* `report` adds the sha256 fingerprint, per-category min/median/max amounts and a
  category × status table. `python -m dataset` prints it as JSON. `python -m
  scripts.validate_dataset` prints it as Markdown. Both exit 1 if the dataset is invalid.
* The seed procedure is code (`scripts/validate_dataset.py::seed_history`). Transcript 01
  recomputes it, and a unit test asserts that `SEED` is the first valid seed from 2026.

### D-017 `LoanCategory` / `LoanStatus` land in Phase 1
§PLAN Phase 1 tests that the schema literals equal the dataset vocabularies, so
`credagent/schemas.py` is created now with just those two `Literal`s. The models in §7.11 come
in Phase 6.

### D-018 Transcript framework
`scripts/transcript_kit.py` (`Transcript`: header block, Markdown helpers, ✔/✘ checks,
`normalise`) and `scripts/make_transcripts.py` (generator registry and CLI; exit 1 on any ✘
or on a transcript with no checks). The header provenance, `GIT_SHA` and `IMAGE_ID`, is two new
`Settings` fields (default `unknown`) that `make transcripts` passes in. The image cannot see
`.git`, and only settings may read the environment. `make transcripts` depends on `make build`
because the `runner` runs the code baked into `credagent:runtime`. `make bless` is
`transcripts` for now, plus a `git status` of the artifact directories. The golden layer
(`tests/golden/test_transcripts.py`) regenerates into a tmp dir in the `tests` container and
diffs after normalisation. It also checks that committed transcripts and registered generators
are the same set. `make offline-proof` now also runs `scripts.validate_dataset` and
`scripts.make_transcripts --out /tmp/transcripts` in the runner on the backend network.

## Seed history (Phase 1)
Procedure (§ARCH 7.1): start at `SEED = 2026` and increment until `validate` returns no
problems. Records are never edited. Every attempt:

| Attempt | Seed | `validate` result |
|---|---|---|
| 1 | 2026 | valid: no problems |

**Chosen seed: 2026**, the first attempt. Resulting dataset: 60 records. Categories: Personal
12, Home 14, Auto 19, Education 6, Business 9. Statuses: Submitted 13, Under Review 15,
Approved 10, Rejected 13, Disbursed 9. Fraud-flagged 10/60 = 16.67%. sha256
`92fd6e6696e4437ed8fefe7b2333a9700dfd46ddfcf411f1744231724d39946e`, pinned in
`tests/unit/test_dataset.py::GOLDEN_SHA256`. Transcript 01 recomputes this table on every run,
and `test_seed_is_the_first_valid_seed_from_the_start` enforces it.

## Phase 2 — knowledge base, chunking, embeddings, indexing (2026-11-07)

### D-019 The metadata collection is `kb_meta`, not `_meta`
chromadb 1.5.9 rejects `_meta`: collection names must be 3–512 characters of `[a-zA-Z0-9._-]`
and must **start and end** with `[a-zA-Z0-9]`. Pinned by
`tests/contract/test_chroma_contract.py::test_collection_names_must_start_and_end_alphanumeric`.
Escalated per the stop rule. **Decided by the human (2026-11-07): `kb_meta`**, configured as
`Settings.collection_meta` (`COLLECTION_META`). The collection holds one record, id `kb_meta`,
whose metadata carries `kb_version`, both collection names and counts, the doc count and the
chunk params as JSON. Chroma requires a document and an embedding on every record (also pinned
by a contract test), so the record has the document `kb_meta` and the 1-dimensional embedding
`[1.0]`. §ARCH 3.5 and 7.4 and §PLAN Phase 2 are amended.

### D-020 Indexer readiness: healthcheck plus a bounded connect retry
The Chroma image has a working healthcheck (bash `/dev/tcp`, D-004), so `indexer` keeps
`depends_on: chroma: service_healthy` and `api`/`api-offline` keep
`indexer: service_completed_successfully`. §ARCH 3.5's bounded retry is still implemented, as
`credagent.rag.store.connect_http` (10 attempts, sleeps 0.5, 1, 2, 4, then 5 s; settings
`CHROMA_CONNECT_ATTEMPTS`, `CHROMA_BACKOFF_CAP_S`). The indexer and the transcripts use it now;
the API will use it in Phase 10. It is a startup guard, not a test retry: unit tests drive it
with an injected `sleep` and never wait.

### D-021 Embedding determinism: bit-identical per batch, ~1e-7 across batch compositions
The same texts in the same batch give bit-identical vectors (single torch thread). The same text
embedded alone and inside a larger batch differs by up to ~6e-8, because padding changes the
float32 arithmetic. Pinned by `tests/contract/test_sentence_transformers_contract.py`. Across
processes the plan's `allclose 1e-6` applies. The indexer always embeds each strategy's chunks
in one call, in chunk order, so the stored vectors are reproducible. Similarities are printed to
3 decimals, and §ARCH 7.19's ≥ 0.05 threshold margin absorbs this drift.

### D-022 Phase 2 implementation details within §ARCH 7.2–7.4
* **KB files**: front matter has exactly `doc_id, title, topic, version: 1, disclaimer`. The body
  is one prose paragraph. The loader collapses whitespace and NFKC-normalises, so char spans
  refer to the normalised single-line body. `doc_id` must equal the file stem, and the
  `topic` strings are the brief's 12 topic names plus `Status meanings & SLAs`.
* **Splitter** (`text.sentence_spans`): a terminator (`.`, `!`, `?`, `...`, `…`, plus closing
  quotes/brackets) ends a sentence only when followed by the end of text or by whitespace and an
  uppercase letter, digit, `₹` or opening quote. A period after an abbreviation in
  `ABBREVIATIONS` never ends a sentence, except for `SENTENCE_FINAL_ABBREVIATIONS`
  (`p.a`, `etc`, `a.m`, `p.m`), which do when an uppercase letter follows (`… 9.9% p.a. Auto
  loans …` is two sentences). A blank line is a hard break.
* **Tokens**: negations (`no`, `not`) are not stopwords, so "no prepayment penalty" never
  matches "prepayment penalty" on tokens alone. Single letters are dropped.
* **Numbers**: lakh/lac/crore (singular or plural) multiply the preceding number; digit grouping
  (Western or Indian) is removed; a `%` directly after or one space after the number is kept.
* **Fixed chunking**: the next window starts at the last word start at or before
  `end − overlap` inside the current chunk, else at the chunk's last word start. Overlap is
  therefore 40 characters plus up to one word. A single word longer than `size + max_extend`
  stays whole as its own chunk (never split). Such a chunk cannot overlap its successor, which
  is the only exception to the overlap invariant; the Hypothesis test states that precondition.
* **kb_version** (`credagent.kb.version`): sha256 over each KB file's name and bytes in name
  order, then the JSON of `chunk_params()` (fixed params + the splitter's abbreviation lists).
* **Chunk metadata**: `doc_id, strategy, index, char_start, char_end, kb_version, title`.
  Stale ids are computed per collection as stored ids − produced ids and deleted after the upsert.
* **Settings** added: `KB_DIR` (`/app/kb`), `EMBED_BATCH_SIZE` (32), `COLLECTION_META`,
  `CHROMA_CONNECT_ATTEMPTS`, `CHROMA_BACKOFF_CAP_S`. The chunk parameters are design constants
  in `kb/chunking.py`, not environment settings; they reach `config/retrieval.json` in Phase 3.
* **Layering**: besides rank order, `kb` may not import `rag`/`llm`/`crew`/`pipeline`/`api`, and
  `rag` may not import `llm`/`crew`/`pipeline`/`api` (`test_static_rules.py`). Scripts read the
  environment only through settings.
* **Make/CI wiring**: `make index` depends on `build`, because the service runs the code baked
  into the image. `make transcripts` depends on `index`, since transcript 03 reads the index back.
  The golden layer runs as `ci/run_layer.sh golden chroma indexer`; `run_layer.sh` runs
  `indexer` to completion with `compose run` instead of `up --wait`. `make offline-proof` also
  runs the indexer on the backend network.
* **Hypothesis**: profile `credagent` (no example database because the mount is read-only;
  `deadline=None` so results never depend on machine load; 200 examples), seeded by
  pytest-randomly.

### D-023 Integration test for the server/client version (P8) stays replaced
§PLAN Phase 2 and the Phase 2 brief list "Chroma server version == client version (1.5.9)".
Per D-015 that comparison cannot hold (the server reports `1.0.0`), and the human chose the
HttpClient round-trip test instead. Phase 2 adds no version-string test.
`test_client_and_server_interoperate_over_http` keeps covering P8 in the integration layer.

### D-024 Phase 3 implementation details within §ARCH 7.5, 7.8, 7.15
* **Layering**: `observability` (metrics now, the JSON log formatter in Phase 10) is a rank-0
  leaf, because `rag` (retrieval/embedding counters) and `cache` both increment counters. The
  Phase 10 `PiiScrubFilter` needs `PiiMasker` (guardrails, rank 2), so the masker will be
  injected into the formatter, not imported.
* **LLMClient**: `rag` may not import `llm`, so `rag/generator.py` defines the `LLMClient`
  protocol and `GroundedGenerator`. Its mock is `llm.engine.MockEngine`, which callers inject.
  `generate_grounded(query, chunks, *, threshold)` takes the top-1 from the chunks (−1.0 when
  there are none). Every client call, fallback included, counts one `generation_calls`.
* **Sentence selection**: `MIN_SENTENCE_SCORE = SIMILARITY_WEIGHT = 0.3`, so similarity alone
  (≤ 1) can never select a sentence: it must share at least one query token. If no sentence
  qualifies, the answer is the fallback with `grounded=False`. Dedupe is by normalised text (the
  higher-ranked copy wins), plus containment: a fixed-chunk fragment contained in another
  context sentence is dropped (P16). "Document order" is (order of the document's first
  appearance by rank, chunk `char_start` + sentence offset), so `RetrievedChunk` now carries
  `char_start` from the chunk metadata. Similarities are clamped to [−1, 1] (float32 distance).
* **Support rule**: token recall is computed against each context sentence and the best one is
  kept. A sentence with no content tokens makes no claim (recall 1.0), and an empty answer has
  ratio 1.0. Numbers are compared canonically (`₹1.5 lakh` = `₹150,000`) over all of C.
* **Calibration**: 10 in-scope and 8 OOS queries (near-domain traps: share price, mutual fund,
  USD/INR rate, fixed-deposit interest, a poem about loans). The gate applies to
  `GATED_STRATEGY = "sentence"`. If Phase 4 deploys `fixed`, its own gap must reach 0.05 first
  (measured 0.033). A collection with gap ≤ 0 gets `threshold: null`. Similarities are stored
  to 4 decimals. `deployed_strategy` stays `null` until Phase 4. The script refuses to write
  the file when the gated gap is below `MIN_GAP`. `RETRIEVAL_CONFIG_PATH` is a new setting
  (`/app/config/retrieval.json`), and the runner mounts `./config` so `make calibrate` can
  write it.
* **Cache**: `normalize_query` = `text.normalize` → typographic quotes to ASCII → strip edge
  whitespace/Unicode punctuation (Hypothesis proves idempotence). Values are computed outside
  the lock; concurrent misses may compute twice (deterministic, harmless). Exceptions propagate
  and are never stored. TTL expiry is inclusive at `put_time + ttl`. Not yet wired into
  `RagTool` (Phase 6).
* **Coverage gate**: `fail_under = 90` (combined) plus `scripts/coverage_gate.py`, which
  checks line ≥ 90 % and branch ≥ 85 % separately from `coverage json` in the same container.

### D-025 Calibration revision 1: kb-04 wording (the only revision)
First measurement: the sentence collection had min_in 0.497 ("Which documents are required for
KYC?") against max_oos 0.480 (fixed-deposit trap), a gap of 0.017. kb-04's sentences never
said "KYC documents" (only the title did, and the title is not embedded). It was reworded,
keeping every fact: "For KYC, a PAN is mandatory …" and "one address proof as a KYC document".
The query sets were written before measuring and were not changed. Result on `kb_sentence_v1`:
max_oos 0.480, min_in 0.654, gap 0.175, threshold 0.567. `kb_fixed_v1` measured gap 0.033
(below 0.05; reported, not gated).

### D-026 Groundedness demo query (§ARCH 7.12.3) verified, with a thin margin
"What is the gold-loan late payment penalty?" has top-1 0.555 (kb-08) on `kb_sentence_v1`,
below the 0.567 threshold. It stays the transcript-15 case, and transcript 04 checks it. The
margin is only 0.012, smaller than the 0.05 calibration margin, so any KB or threshold change
can flip it. Transcript 04's check will then fail loudly, and the case must be re-chosen.
Held-out honesty set (§ARCH 11.4): 8/10 grounded with the right top-1 doc. Top-1 doc is
correct on 10/10. The two misses ("first instalment due" 0.469, "withdrawing cash on the card"
0.505) fall below the threshold and get the fallback, not a wrong answer.

### D-027 Phase 4 chunking evaluation (§ARCH 7.6) — method (deployment resolved by D-028)
* **Queries** (`evaluation.PR_QUERIES`, written before measuring, never tuned): the two §ARCH
  7.6 two-document examples (kb-11+kb-12, kb-07+kb-08) plus three single-document queries
  (kb-06, kb-09, kb-13). Registered in `tests/unit/test_query_sets.py`, so they are checked as
  disjoint from the calibration set and absent from `src/credagent/llm/`.
* **Math**: P and R are exact `Fraction`s; `P := 0` with a warning on an empty retrieval (the
  query text is not logged). Gold ids must be non-empty and unique.
* **Choice**: macro F1 (from macro P and macro R), then mean top-1 similarity rounded to 4
  decimals, then fewer distinct docs per query, then the gated strategy. A winner whose
  calibration `meets_min_gap` is false is reported but never deployed (`NotDeployableError`;
  the script exits 1 and leaves both files unchanged).
* **Ownership**: `make calibrate` writes `deployed_strategy: null`, and `make eval-chunking` sets
  it, so `make bless` runs calibrate → eval-chunking → transcripts. Transcript 04's
  byte-equality check now excludes `deployed_strategy`. The full byte equality moved to
  `tests/golden/test_config.py` (calibrate + eval), and transcript 06 checks the deployment.
  New setting `CHUNKING_REPORT_PATH` (`/app/eval/results/chunking_pr.md`).
* **Measured (top-3)**: both collections give per-query P/R that are identical, macro P = macro R
  = macro F1 = 0.900, and mean docs 1.400. Mean top-1 similarity is 0.655 (fixed) vs 0.654
  (sentence), so the rule picks **fixed**. Its calibration gap is 0.033 < 0.05, so it was not
  deployed: a stop-and-ask condition. Escalated to the human and resolved by D-028.

### D-028 Only strategies that meet the calibration gap compete for deployment (human-approved)
**Context:** under D-027's rule, the P/R comparison on the 5 queries tied exactly. Both
collections had macro P = macro R = macro F1 = 0.900 and mean docs 1.400, so the result rested
on mean top-1 similarity, 0.655 vs 0.654. That 0.001 margin is noise next to the 0.05 margin
the project requires everywhere else (§ARCH 7.19). It selected `fixed`, whose calibration gap
(0.033) is below `min_gap` (0.05), so `fixed` cannot be deployed.
**Options escalated:** (a) keep the rule and revise the KB until `fixed` reaches the gap;
(b) only strategies whose calibration meets `min_gap` compete; (c) treat sub-tolerance top-1
differences as ties.
**Decision (the user approved option b, 2026-11-07):** `choose(..., calibrations=…)` first
excludes every strategy whose `meets_min_gap` is false (a missing calibration counts as false).
The D-027 chain (macro F1 → mean top-1 → fewer docs → gated) then runs unchanged over the
eligible set. With exactly one eligible strategy, `decided_by = "only_eligible"`. With none,
`NoEligibleStrategyError` is raised: the script exits 1, and the config (with
`deployed_strategy: null`) and the report stay untouched. Excluded collections are still
measured and reported, and the recommendation states the exclusion with its measured gap.
**Result:** `fixed` is excluded (gap 0.033 < 0.05), and `sentence` (gap 0.175, threshold 0.567)
is deployed: `deployed_strategy: "sentence"`.

## Phase 5 — lookup tool, escalation, ACL core (2026-11-07)

### D-029 `cache` and `governance` share the tools rank in the layering test (approved with an amendment, D-031)
**Context:** §ARCH 4 ranks `cache` and `governance` above `tools`, `rag` and `llm`. Three other
sections need the opposite direction: §7.9 (`RagTool._run` → `ResponseCache.get_or_compute`),
§7.14.1 (`tools/acl.py` checks `TOOL_ACL` from `governance/policy.py`) and, from Phase 6, §7.14.3
(the mock LLM in `llm` charges `governance`'s `BudgetMeter`). Under the §4 ranks, each of these
would be an upward import.
**Decision:** `LAYER_RANK` in `tests/unit/test_static_rules.py` moves `cache` and `governance` to
rank 1, next to `tools`/`rag`/`llm`. They depend only on rank 0 (`settings`, `text`,
`observability`). `FORBIDDEN_IMPORTS` keeps both as leaves of that rank: neither may import
`tools`, `rag`, `llm` or the other. `pipeline` still sits above all of them. Dependency injection
was rejected: the ACL signature `assert_allowed(agent_key, tool)` is fixed by §7.14.1 and has no
slot for the table.
**Resolved (2026-11-07):** approved, amended by D-031 (the BudgetMeter does not live in `governance`). §4 is updated.

### D-030 Phase 5 implementation details within §ARCH 7.7, 7.9, 7.14.1
* **Escalation:** P80 uses integer arithmetic, `ceil(p·n/100) = -(-p·n // 100)`. A Hypothesis
  property compares it with an exact `Fraction`, so no float `0.8·n` can round across an integer.
  The cut-offs (`unflagged_min_days`, `flagged_min_days`) come from evaluating the rounded score
  itself, not from the closed form. `EscalationConfig` re-derives T and the cut-offs on load, so a
  hand-edited `config/escalation.json` is rejected. Measured: n = 60, **P80 = 25 days, T = 0.5**;
  unflagged applications escalate from day 25 and flagged ones from day 5; 16/60 escalate (8/10
  flagged, 8/50 unflagged).
* **Config ownership:** `scripts/build_escalation_config.py` (`make escalation-config`, first
  step of `make bless`; also run by `offline-proof`). New setting `ESCALATION_CONFIG_PATH`.
  Golden byte-equality is in `tests/golden/test_config.py`.
* **Record id:** the regex is `^LA-[0-9]{4}$`, ASCII only. `\d` would accept Arabic-Indic and
  other Unicode digits. Non-string input is `invalid_record_id`. `RecordIdArg` deliberately has
  no pattern: a malformed id must reach the function and come back as JSON, not as a CrewAI
  argument-validation error.
* **Owner binding:** both tools are owner-bound (`RagTool` to `retrieval_agent`, symmetric with
  `TOOL_ACL`). Each checks `assert_allowed(owner, self)` at construction and again on every
  `_run`. `assert_allowed` also rejects a tool whose `owner` differs from the agent receiving it.
  `LeastAutonomyViolation` (a `PermissionError`) is the only exception a tool raises; when it
  does, nothing is recorded.
* **Crew audit:** fail-closed on unknown roles (`AGENT_ROLES` maps the §7.9 roles to agent
  keys), on `allow_delegation` not `False`, and on a task with tools but no agent. Uses duck
  typing, so it needs no CrewAI import.
* **Tool outputs** (`schemas.py`): `LoanStatusFound | LoanStatusNotFound` and
  `RetrievalToolOutput | RagToolError`, exposed as `TypeAdapter`s. A retrieval or generation
  failure becomes `{"kind":"rag_error","error":"retrieval_unavailable"}` with no exception text
  and is never cached. Structured logging of the swallowed error lands with §7.17 in Phase 10.
* **Ledger:** `LedgerEntry(agent, tool, args, duration_ms, cache_hit)`, where `agent` is the
  owner key the ACL checks (not the display role). The lookup tool has no cache, so its
  `cache_hit` is always `False`.
* **Deployment binding:** `rag_tool_for_deployment(config, …)` takes the strategy from
  `deployed_strategy`, plus its calibrated threshold, `top_k` and `kb_version`, from
  `config/retrieval.json`. With no deployed strategy or no threshold it raises at wiring time.

## Phase 6 — mock CrewAI LLM, crew, structured output (2026-11-07)

### D-031 Per-request state lives in `credagent/context.py` (human-approved amendment of D-029)
**Context:** D-029 put the `BudgetMeter` in `governance`, so `llm` would import `governance`.
**Decision:** new bottom-layer module `credagent/context.py` (rank 0, beside `schemas`) holds
`RequestContext`, `BudgetMeter` and `ToolCallLedger`; `tools/ledger.py` moved there unchanged.
`governance/` keeps policy only (`TOOL_ACL` now, `BudgetPolicy` in Phase 9). The meter is
charge-only: `charge(persona, tokens)` adds `ceil(chars/4)` tokens, counts calls per persona and
sets a sticky `exceeded` flag once the total passes `max_tokens` (exactly at the limit is within
it). It never raises; the orchestrator checks the flag (§6 step 9). `RequestContext.new` takes an
optional fixed `trace_id` so transcripts are reproducible. §4 (diagram, layout, rule) is amended;
`tests/unit/test_static_rules.py` gains `context` at rank 0 and a test that these classes are
defined only in `context.py`.

### D-032 Retrieval skips its tool for status-only questions (refines §ARCH 7.8.1 step 5; human-approved 2026-11-07)
**Context:** §7.8.1 step 5 has the Retrieval persona always call `rag_lookup`. Then a pure status
question ("What is the status of application LA-0007?") would also get whatever policy text
retrieval finds, and nothing could tell `status` from `mixed`.
**Decision:** `MockEngine.classify_intent(query, record_id)` removes the record id and checks the
remaining content tokens against `STATUS_TERMS` (generic status words, plus the `LoanStatus` and
`LoanCategory` vocabularies taken from the schema `Literal`s). All status words → `status` with an
id, `clarify` without one when a status cue (`status`, `update`, `approved`, …) is present.
Otherwise `mixed` with an id, `policy` without. Retrieval skips the tool for `status` and
`clarify` and answers `{"kind":"retrieval_result","rag":null}`, mirroring the Lookup persona's
`no_record_id` skip. The Composer re-derives the intent and fails closed (`agent_error`) when the
task outputs contradict it. No query string is special-cased; the rule only reads tokens.
**Consequence:** exact LLM calls (retrieval / lookup / composer): policy and out-of-scope 2/1/1,
status and unknown id 1/2/1, mixed 2/2/1, clarify 1/1/1. The §7.14.3 worst case (crew 5 calls)
is unchanged. Reverting to "always retrieve" is a one-line change in `_retrieve`.

### D-033 Phase 6 implementation details within §ARCH 7.8.1, 7.9, 7.11
* **Task outputs:** `RetrievalTaskOutput{kind:"retrieval_result", rag: RetrievalToolOutput |
  RagToolError | null}` (a wrapper, because the tool output already has `kind:"rag_result"`);
  the Lookup output is the tool output plus `kind:"lookup_result"` (`LookupFound` /
  `LookupNotFound`, error adds `no_record_id`). `kind` is required on both.
* **CrewResponse:** `RefusalReason = budget_exceeded | agent_error | retrieval_unavailable` for
  now (Phase 7 adds its own). Validators beyond §7.11: `refusal_reason` is set **exactly** when
  `refused`; `intent == "refused"` exactly when `refused`; a refusal is never grounded; an
  `escalation` needs an `application`. `sources` is a tuple (a JSON array).
* **Unknown or malformed id** (human-approved 2026-11-07; §7.11 amended): answered as `intent="clarify"` ("I couldn't find a loan
  application with id LA-9999. Please check the id (format LA-0000)."), because §7.11 makes
  `status` require an application. `text.find_record_id` accepts `LA-` + 1–8 ASCII digits, any
  case, so a malformed id reaches the tool and comes back as `invalid_record_id`.
* **Answers:** the status sentence gives status, amount (Indian grouping), age and the
  escalation decision with score and threshold. It never mentions the fraud flag (the
  `application` field still carries it). A mixed answer is the status sentence followed by the
  policy answer; `grounded` then reflects the policy part. Phase 7's groundedness guard must
  treat the status sentence as grounded in the record, not in the retrieved chunks.
* **Refusals everywhere are `CrewResponse` JSON.** Retrieval and Lookup emit the same refusal
  shape on budget overrun or protocol error, and the Composer passes an upstream refusal
  through, so the crew output always validates (budget overrun: 1/1/1 calls, no tool runs).
* **Delimiters** live in `llm/crewai_llm.py` (`task_inputs`), not `crew/templates.py`, because
  `llm` may not import `crew`. The record-id block precedes the query, and the query runs to the
  **last** `<<<END_QUERY>>>`, so delimiter text inside a query can neither forge an id nor cut
  the query short (P17 blocking itself is Phase 7).
* **"Observation:"** inside our JSON (a KB sentence or a query could contain it) is written
  `Observation:`: the same string after JSON decoding, but never the CrewAI stop word.
* **Factory:** `build_crew(ctx, query, deps)`. The strategy comes from
  `config/retrieval.json` (`deployed_strategy`), the record id from the masked query.
  `run_crew` kicks off without `inputs` (CrewAI then skips interpolation, so braces in a query
  stay literal; tested) and narrows to `CrewOutput`. `parse_crew_output` raises
  `SchemaViolation` (§6 step 6). Role strings come from `governance.policy.AGENT_ROLES`.
* **Measured prompt sizes** (fixed trace ids, transcripts 08/09): policy 3,605 tokens, status
  3,325, mixed 4,551, unknown id 3,219, against `BUDGET_MAX_TOKENS = 8000`. The mixed crew alone
  is already 57 % of the budget, above the §7.14.3 "≤ 50 % per eval query" invariant, before the
  reviewer and editor run. Phase 9 calibrates the budget, so it must expect this.
* **Transcripts** print floats rounded to 3 decimals (the risk register's float-drift
  mitigation); the checks use the unrounded values.

## Phase 7 — pipeline, guardrails, memory (2026-11-07)

### D-034 A Claude Code hook enforces "Docker only" for host Python
**Context:** CLAUDE.md forbids `pip install` or running Python on the host; until now only the
rule text enforced it.
**Decision:** `.claude/settings.json` registers a `PreToolUse` hook on `Bash` that runs
`.claude/hooks/deny-host-python.sh` (bash + `jq`, no Python). It drops heredoc bodies and
quoted strings (data, not commands), splits the rest on `;`, `&&`, `||`, `|` and newlines, skips
leading `VAR=value` assignments and `env`/`command`/`exec`/`sudo`/`time`/`nohup`, and rejects
(exit 2, reason on stderr) a segment whose first word's basename matches `^(python|pip)[0-9.]*$`.
`make …` and `docker compose run … python …` pass, because the first word is `make`/`docker`.
**Verified once (2026-11-07):** 10 deny cases (`python3 -c 1`, `pip install x`, `pip3 list`,
`python -V`, `/usr/bin/python3.12 x`, `cd src && python -m x`, `FOO=1 python x`,
`env python x`, `ls | pip3 freeze`, `python f` after a heredoc) exit 2; 9 allow cases
(`make gate-fast`, `docker compose run --rm tests python -m pytest`, `git log`, `echo python`,
`grep pip README`, `pipx run x`, `ruff check`, a quoted `"a | python b"`, a heredoc whose body
holds `python -V`) exit 0; a live `python3 --version` tool call was blocked by Claude Code. The
first version also split heredoc bodies and blocked this very entry; fixed before commit.
**Limits:** it guards Claude Code's Bash tool only, not a human's shell; `bash -c '…'` and `eval`
are not parsed. A guard rail, not a sandbox.

### D-035 PII masking details within §ARCH 7.12.1
* **Input:** `text.strip_invisible` first (zero-width and bidi controls removed, then NFKC), so
  `ABC​PE1234F` and full-width digits are caught. The masked text *is* that stripped text
  with the replacements; everything downstream sees it.
* **Spans:** rules run in the §7.12.1 order; a match is kept only if it overlaps no span already
  taken (within one rule, longest first). Tokens contain no digits, so masking is idempotent.
* **Boundaries:** "not a letter/digit" lookarounds instead of `\b`, so `_` cannot glue a value to
  a neighbour (`id_ABCPE1234F_x` is masked).
* **PAN:** compact form in any case; the space/hyphen-separated form only in capitals, because
  case-insensitive separators turned ordinary phrases into PANs ("small 5000 a" → `SMALL 5000 A`).
* **Bank account:** the keyword rule also accepts `#` before the digits (`A/C#…`). Rule 5 is
  `\d{9,}` (fail-safe: runs longer than 18 digits are masked too, not left whole).
* **`MaskResult{text, counts_by_type}`**, counts for all three types always present (0 allowed);
  no values or spans. `total` is a property.

### D-036 Phase 7 pipeline, guard and memory details within §ARCH 6, 7.10, 7.12
* **Injection (§7.12.2):** runs on the PII-masked text. Normalisation order: strip invisible +
  NFKC → casefold → join runs of ≥ 3 single characters ("i g n o r e") → collapse whitespace →
  de-leet. 18 rules; strong families weigh 1.0, ambiguous words less ("act as" 0.5, "you are
  now" 0.6, the name "Dan" 0.4). `delimiter_smuggling` (any `<<<`/`>>>`, `⟦`/`⟧`, `</system>`,
  `<|im_start|>`, `[INST]`, `## system`) weighs the full 1.0. Base64 runs ≥ 40 chars weigh 0.6
  and are decoded and re-scanned; a decoded payload that would block adds 1.0. The 25 + 25 corpus
  is source data in `eval/injection_corpus.yaml` (`INJECTION_CORPUS_PATH`): 0 FN, 0 FP.
* **Groundedness (§7.12.3):** the gate also covers `out_of_scope` (a policy question whose
  retrieval fell short), so the "I don't know" fallback becomes the documented `ungrounded`
  refusal and the transcript-15 demo actually fires the gate. A policy-type answer without a
  retrieval trace (RAG tool error) is refused as `retrieval_unavailable`. Sentences are judged
  policy-first (support rule against the source chunks' texts, read back with
  `VectorStore.documents`); a sentence that fails it is a *record* sentence only if it is anchored
  to the application (record id, a loan status, "application"/"escalation", or a number) and the
  intent has a record, and it must then agree with the record (numbers, id, status, category).
  Anything else is `unsupported_sentence`.
* **Follow-ups (§7.10):** references are "its", "this/that/my/the same application|loan", "the
  same one", and "it" only after a preposition or auxiliary ("for it", "does it", "has it").
  "is it" is never a reference ("Is it possible …" is an expletive; known limit). A status-only
  remainder takes the most recent record id; anything else takes the most recent category ("a
  Business loan"). Nothing to resolve → `clarify`, answered **without** the crew and without
  review (`review=None`, 0 tokens); clarifications are stored like answers.
* **Consequence to flag (needs a human view before Phase 11):** masking leaves words like `pan`,
  `redacted`, `no` in a status question, so D-032 classifies "My PAN is … What is the status of
  LA-0007?" as `mixed`; its policy half is far below the threshold (0.334) and §7.12.3 then
  refuses the whole answer. PII-bearing demo/eval queries therefore use a policy question
  (transcript 13: KYC, top-1 0.583). A possible amendment: drop redaction tokens and their
  keyword ("PAN", "a/c no") before `classify_intent`.
* **Timeout:** `REQUEST_TIMEOUT_S` bounds the crew kickoff (§7.9), not the whole pipeline; §5's
  row is amended. `anyio.to_thread.run_sync(..., abandon_on_cancel=True)` under `fail_after`;
  the abandoned thread finishes on its own.
* **Budget:** checked right after kickoff (no review runs on an overrun) and again at stage 9
  (a later review overrun). Both raise `BudgetExceeded(stage="runtime")`; the preflight raises
  `stage="preflight"` with zero crew calls. Stage 3 is `PassThroughPreflight` (estimate only,
  never rejects) until Phase 9.
* **Interfaces for later phases:** `ReviewStage` (stage 7, `PassThroughReview` approves the draft
  unchanged), `BudgetPreflight` (stage 3), `RequestSink` + `TurnRecord` (stage 12: one record per
  request in `finally`, masked text and counts only; Phase 10 writes it as JSON lines).
  `AskResponse`, `ReviewVerdict`, `ToolCall`, `Usage` live in `schemas.py`, because `pipeline`
  may not import `api`; `AskResponse` reuses `CrewResponse`'s consistency validator.
* **Session store:** a session whose lock is held, and the one being accessed, are never evicted;
  `reset` removes the session.
* **Regenerated artifacts:** adding `ungrounded` and `injection_detected` to `RefusalReason`
  lengthens the `CrewResponse` schema in the Composer's prompt by 15 tokens, so transcripts
  08/09/12 changed only in those counts: policy 3,620, status 3,340, mixed 4,566, unknown id
  3,234 (D-033 measured 3,605 / 3,325 / 4,551 / 3,219).
* **Groundedness demo re-verified:** "What is the gold-loan late payment penalty?" top-1 0.555 <
  0.567 on `kb_sentence_v1` (margin 0.012, as in D-026); it stays the case.

## Phase 8 — Autogen review stage (2026-11-07)

### D-037 Phase 8 pre-steps: intent ignores PII redactions; demo margin; timeouts; `is None`
* **Intent (amends D-032/D-036):** `MockEngine.classify_intent` and `is_status_only` drop PII
  redaction tokens (`[PAN_REDACTED]`, `[AADHAAR_REDACTED]`, `[BANK_ACCOUNT_REDACTED]`) and, when
  one was present, the words that announce them (`pan`, `aadhaar`, `bank`, `account`, `acct`,
  `no`, `number`). "My PAN is ABCPE1234F, status of LA-0007?" is now `status` and answered, not a
  `mixed` answer refused for its empty policy half. Without a redaction token nothing changes.
  Test: `test_orchestrator.py::test_a_redaction_token_does_not_turn_a_status_question_into_a_mixed_one`.
* **Groundedness demo margin:** the integration demo test now also fails when
  `threshold − top1 < 0.005` (today 0.012, D-026), so a KB or embedding change cannot leave
  the demo balanced on the threshold unnoticed.
* **Timeouts:** pytest-timeout stays `timeout = 60` for unit/contract/security/golden;
  `tests/conftest.py` gives every `integration` and `e2e` test 300 s unless it sets its own.
* **CLAUDE.md rule:** injected dependencies default with `is None`, never `x or Default()`
  (an empty `SessionStore` is falsy). The one remaining `crew or FakeCrew()` in the tests is fixed.

### D-038 Phase 8 review-stage details within §ARCH 7.8.2, 7.13
* **Interfaces:** `ReviewStage.review(ctx, query, draft, trace, lookup) -> ReviewOutcome`
  (Phase 7's `(ctx, draft, trace) -> ReviewVerdict` had no question and no record).
  `run_review(draft, context, ctx, inject_claim=None)` keeps the brief's signature; `context` is
  `ReviewContext(question, documents)`. `ReviewOutcome = (verdict, draft, messages, error)`.
  `ReviewVerdict` stays in `schemas.py` (D-036), so there is no `review/models.py`.
  `AutogenReview` (the default stage 7, in `pipeline/orchestrator.py`) replaces
  `PassThroughReview`, which is removed.
* **Context documents:** the retrieved chunk texts and, for a found record, its JSON (without
  `kind`) **plus** `MockEngine.record_statement(record)`. The JSON alone is not enough for the
  token-based support rule: "₹25,00,000", "created today", "1 day ago" share no surface tokens
  with `"loan_amount_inr": 2500000, "days_since_created": 0`, so correct status sentences would
  be flagged. Numbers are still checked against the record.
* **Delimiters:** task = `<<<QUESTION>>>`, `<<<DRAFT>>>`, `<<<CONTEXT>>>` (items separated by
  `<<<ITEM>>>`). The reviewer echoes the draft between `<<<DRAFT>>>`/`<<<END_DRAFT>>>` before
  its report, so the editor, which reads only the reviewer's message, can return the draft byte
  for byte. User text cannot forge any of these (`<<<` always blocks, P17).
* **Not reviewed:** refusal and clarification drafts claim nothing (the gate already passes
  them as `not_applicable`), so stage 7 is skipped and `AskResponse.review` is `None`, as for a
  memory clarification (D-036).
* **Fail closed:** any exception in the run or any shape deviation (not exactly
  `user → Policy_Compliance_Reviewer → Final_Editor`, last not a re-validating
  `StructuredMessage[ReviewVerdict]`) → the verdict `approved=false` with the `agent_error`
  refusal text, `reason="review_error: <type>"`; the pipeline sends the `agent_error` refusal,
  skips the gate, and logs `TurnRecord.review_error = "<type>: <message>"`. No new
  `RefusalReason`.
* **Budget:** each `create` charges `ceil(chars/4)` of its input messages under `reviewer` /
  `editor` (like `MockCrewLLM`, output is not charged); over budget only sets the flag and the
  stage-9 post-check raises. The review runs in the event loop and is not under
  `REQUEST_TIMEOUT_S` (which bounds the crew kickoff, D-036).
* **Measured (transcript 20, review included):** policy 6 calls / 4,229 tokens (review 609),
  status 6 / 4,087 (747), mixed 7 / 5,667 (1,101). **Flag for Phase 9:** against today's
  `BUDGET_MAX_TOKENS = 8000` that is 53 % / 51 % / 71 %, above the §7.14.3 "≤ 50 % of budget"
  invariant. Phase 9 sets the budget from measured overhead, so the values are not changed here.
* **Artifacts:** transcript 10's tokens per turn rose by 747 / 718 / 605 (the two review calls)
  and stage-7 wording changed in 10 and 15; 08, 09, 11–14 changed only in their headers.

## Phase 9 — Governance: budget cap, least autonomy, risk (2026-11-07)

### D-039 The reviewer rebuilds a record's status sentences from the record JSON
* **Problem:** D-038 put `MockEngine.record_statement(record)` into the review context as prose,
  and draft sentences were checked against it by token recall (≥ 0.6). A status sentence with a
  wrong status ("… is currently Approved." for an Under Review record) still matched 5 of 6
  tokens, so the reviewer approved it.
* **Rule:** the context carries only the record JSON (`model_dump_json(exclude={"kind"})`). The
  reviewer (`MockEngine.review`) recognises a context item that validates as `LoanStatusFound`,
  builds that record's sentences with the Composer's own template (`_status_sentence`), and a
  draft sentence is supported by the record **only if it equals one of them**
  (`support_report(…, exact=…)`). The draft's wording of the record is never trusted. Other
  sentences are checked against the remaining documents (the KB chunks) as before.
  `record_statement` and `AutogenReview`'s unused `engine` parameter are removed.
* Test: `test_review.py::test_the_status_sentence_is_rebuilt_from_the_record_json_never_from_the_draft`
  (correct / wrong status / wrong amount). Status and mixed reviews now cost 113 fewer tokens
  (transcript 20).

### D-040 `MAX_MESSAGE_CHARS` 20,000 → 16,000 (amends §ARCH 5, 7.14.3, 7.16; human-approved)
The P10 ordering test requires `MAX_MESSAGE_CHARS × 4` (worst-case UTF-8 bytes per char) to be
below the 64 KiB body cap. 20,000 × 4 = 80,000 > 65,536. The human chose to lower the message
limit and keep the body cap: 16,000 × 4 = 64,000 < 65,536. The ≈ 12,000-char oversized demo
stays under it. New setting `MAX_BODY_BYTES = 65536` (the Phase 10 middleware reads it).

### D-041 Budget measurement and wiring within §ARCH 7.14.3
* **Per-call log:** `BudgetMeter.usage()` keeps one `CallUsage(persona, prompt_tokens,
  completion_tokens)` per mock call. Both adapters call `record_completion` after they produce
  output. This is measured only, never charged: the actual total stays input-only, as before.
  `TurnRecord.llm_usage` carries the log; its length is the "mock LLM call counter" of the
  zero-call tests.
* **Measured set (22 requests):** transcript 03 sample, transcript 05 demo (5 + 1), P/R (5),
  crew 08/09 (policy, status, mixed, unknown, clarify), the PII pipeline query (13), the
  groundedness query (15), and the three memory turns (10) in one session. Duplicates are
  dropped. Phase 11 adds `eval/testset.yaml` and re-runs the measurement. Fixed trace ids make the
  run reproducible; a golden test compares `config/budget.json` to a fresh run.
* **Overhead definition:** per call, prompt tokens − `ceil(len(masked text)/4)`. Max 1,644 →
  `PROMPT_OVERHEAD_TOKENS = 1700`; `MAX_OUTPUT_TOKENS = 298` (max completion); max actual 5,552
  (mixed); max projection 14,217 → `BUDGET_MAX_TOKENS = ceil1000(max(11,104, 17,060.4)) =
  18,000`, `BUDGET_MAX_COST_USD = 0.036`. The smallest rejected input is 2,293 chars; the
  12,000-char demo projects 34,986 tokens.
* **Settings:** a `before` validator fills `budget_max_tokens`, `budget_max_cost_usd`,
  `prompt_overhead_tokens` and `max_output_tokens` from `BUDGET_CONFIG_PATH`
  (`/app/config/budget.json`). An environment variable still wins (tests force tiny budgets
  that way or by constructor). Without the file, the §ARCH 5 initial values apply, so the
  measuring script can run cold.
* **Costs** are compared rounded to 8 decimals (`cost_usd`), like every reported cost, so a
  projection exactly at the cap is not over it.
* **Stage 3:** `BudgetPolicy(max_tokens=<the orchestrator's>)` is the default preflight;
  `PassThroughPreflight` is removed. `PreflightResult` lives in `governance/budget.py`.
  `build_orchestrator` takes `preflight` and `max_tokens` (the measurement and the forced
  runtime-overrun demo use them).
* **`make budget`** runs after `eval-chunking` and before `transcripts` in `make bless`, and in
  `make offline-proof`.
* **Least-autonomy tests (a)–(d)** already existed (Phases 5–6) and are reused, not duplicated:
  (a) `test_crew_factory.py::test_audit_passes_on_the_built_crew_and_maps_the_tools`, (b)
  `test_crew_factory.py::test_make_agent_refuses_a_disallowed_tool_and_an_unknown_agent`, (c)
  `test_acl.py::test_audit_catches_a_tool_smuggled_onto_the_composer`, (d)
  `test_lookup_tool.py::test_owner_tamper_by_assignment_raises_on_run_and_records_nothing`.
