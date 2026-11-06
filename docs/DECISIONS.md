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
