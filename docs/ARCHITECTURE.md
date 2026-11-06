# Cred Loan-Support Agent — Architecture Specification

> **Track:** Cred (Banking & FinTech) · **Status:** Approved for implementation · **Audience:** Claude Code (implementer) and human reviewers
> **Companion documents:** `docs/IMPLEMENTATION_PLAN.md` (phases, gates, test plan), `CLAUDE.md` (operating rules for the implementer).
> Everything in this document is normative unless marked *Informative*. When this document and the capstone brief disagree, the brief wins; record the conflict in `docs/DECISIONS.md`.

---

## 0. Why this track

All five briefs share one skeleton (dataset → RAG → CrewAI crew → memory → guardrails → FastAPI → eval → Autogen review → governance). The Cred track was chosen because it is the hardest variant of that skeleton:

| Dimension | Cred | Others |
|---|---|---|
| Fixed-format PII to mask | **3 formats** (PAN, Aadhaar, bank account) — and Aadhaar (12 digits) **collides** with bank-account patterns (9–18 digits), so masking needs precedence rules | 1–2 formats, no collisions |
| Risk class | **High** (financial data, fraud flags) — stricter governance story | Medium/High |
| KB breadth | 12 topics spanning lending **and** retail banking (cards, KYC, joint/NRI accounts) — retrieval must separate near-neighbour topics (e.g. joint vs NRI accounts, prepayment vs interest slabs) | Mostly single-domain |
| Number-heavy policies | EMI formulas, rate slabs, penalties, liability caps — ungrounded **numeric** claims are the main hallucination risk, which drives a numeric-aware groundedness check | Fewer numbers |

---

## 1. Scope and non-negotiables

### 1.1 What we are building
A support agent for a (fictional) Cred lending-operations team that:
1. answers loan- and banking-policy questions from a knowledge base we author (RAG over ChromaDB);
2. looks up a loan application by id from a dataset we generate (`dataset.py`) and computes an escalation score;
3. orchestrates **Retrieval**, **Lookup** and **Composer** agents with CrewAI (`crew.kickoff()`);
4. remembers a conversation per session (LangChain `InMemoryChatMessageHistory` + `RunnableWithMessageHistory`);
5. masks PII, detects prompt injection, refuses ungrounded answers;
6. has each draft reviewed by a 2-agent Autogen `RoundRobinGroupChat` (Policy-Compliance-Reviewer → Final-Editor) that returns a Pydantic verdict;
7. enforces least autonomy, a per-request token/cost budget, and response caching;
8. is served by FastAPI (HTTP + WebSocket), logs every request as one masked JSON line, and is evaluated on 15 queries for Accuracy, Grounding, Completeness and Safety.

### 1.2 Non-negotiables
- **N1. Docker only.** Nothing runs on the host except `docker`, `docker compose` and `make`. Every build, test, index, transcript, and eval run happens inside containers.
- **N2. Zero API keys, zero external network at runtime.** `MOCK_LLM=true` is the default and the only mode used for graded transcripts. Model weights are baked into the image at build time. Runtime containers that produce graded output sit on an `internal: true` Docker network that has no route to the internet (§3.3). We prove this with an egress probe (§11.7).
- **N3. Determinism.** Same image + same seed ⇒ byte-identical dataset, chunk ids and transcripts (after normalising trace ids and timings). Golden-file tests enforce it.
- **N4. No hand-edited outputs.** Transcripts, calibration values, P/R tables, and eval reports are *generated* by scripts and committed. They are never edited by hand.
- **N5. PII never reaches disk in the clear.** That covers logs, Chroma, memory, cache keys, transcripts, and error responses. An automated canary scan enforces it (§11.6).
- **N6. Every brief task maps to code, a test, and a transcript** (§1.3). A task without all three is not done.
- **N7. Honesty in the mock.** The mock LLM is a deterministic rule engine. It must never hard-code answers to specific test queries. Its behaviour must come from the retrieved context, the tool outputs, and generic rules. Tests check this by running the mock on held-out queries (§11.4).

### 1.3 Traceability matrix (brief task → implementation → proof)

| Task | Module(s) | Automated test(s) | Transcript (generated) |
|---|---|---|---|
| 1 Dataset | `dataset.py`, `scripts/validate_dataset.py` | `tests/unit/test_dataset.py` | `transcripts/01_dataset_report.md` |
| 2 KB | `kb/*.md`, `credagent/kb/loader.py` | `tests/unit/test_kb.py` | `transcripts/02_kb_inventory.md` |
| 3 Chunking + index | `credagent/kb/chunking.py`, `credagent/rag/{embedder,store}.py`, `scripts/indexer.py` | `tests/unit/test_chunking.py`, `tests/integration/test_index.py` | `transcripts/03_chunking_and_index.md` |
| 4 Grounded gen + threshold | `credagent/rag/{retriever,generator,calibration}.py` | `tests/unit/test_generator.py`, `tests/integration/test_calibration.py` | `04_threshold_calibration.md`, `05_grounded_generation.md` |
| 5 Chunking P/R | `credagent/rag/evaluation.py`, `scripts/eval_chunking.py` | `tests/unit/test_pr_math.py` | `06_chunking_precision_recall.md` |
| 6 Lookup + escalation | `credagent/tools/{lookup_tool,escalation}.py` | `tests/unit/test_escalation.py` (+ property tests) | `07_lookup_and_escalation.md` |
| 7 CrewAI crew | `credagent/crew/*`, `credagent/llm/crewai_llm.py` | `tests/contract/test_crewai_contract.py`, `tests/integration/test_crew.py` | `08_crew_rag_query.md`, `09_crew_lookup_query.md` |
| 8 Memory | `credagent/memory/session.py` | `tests/integration/test_memory.py` | `10_memory_multiturn.md`, `11_memory_fresh_session.md` |
| 9 Structured output | `credagent/schemas.py` | `tests/unit/test_schemas.py` | `12_structured_output.md` |
| 10 Guardrails | `credagent/guardrails/*` | `tests/unit/test_pii.py`, `test_injection.py`, `test_groundedness.py` | `13_guardrail_pii.md`, `14_guardrail_injection.md`, `15_guardrail_groundedness.md` |
| 11 FastAPI | `credagent/api/*` | `tests/e2e/test_http.py`, `tests/e2e/test_ws.py` | `16_fastapi_http.md`, `17_websocket_disconnect.md` |
| 12 JSON logging | `credagent/observability/logging.py` | `tests/e2e/test_logging.py`, `tests/security/test_pii_canary.py` | `18_structured_logging.md` |
| 13 Eval (15 queries) | `eval/*`, `credagent/llm/judge.py`, `scripts/run_eval.py` | `tests/unit/test_judge.py` | `19_evaluation_15_queries.md` |
| 14 Autogen review | `credagent/review/*`, `credagent/llm/autogen_client.py` | `tests/contract/test_autogen_contract.py`, `tests/integration/test_review.py` | `20_review_approve.md`, `21_review_revise.md` |
| 15 Governance | `credagent/governance/*`, `credagent/tools/acl.py`, `docs/GOVERNANCE.md` | `tests/unit/test_acl.py`, `tests/unit/test_budget.py`, `tests/e2e/test_budget_http.py` | `22_least_autonomy.md`, `23_budget_cap.md` |
| 16 Cache | `credagent/cache/response_cache.py` | `tests/unit/test_cache.py`, `tests/integration/test_cache_pipeline.py` | `24_cache_hit.md` |
| Cross-cutting: offline | `scripts/offline_probe.py` | `tests/security/test_offline.py` | `25_offline_proof.md` |

---

## 2. Verified platform facts (spiked on 2026-11-06)

*These were checked by installing the packages and running throw-away spikes. Phase 0 turns each one into a **contract test** so a dependency drift fails loudly instead of silently.*

### 2.1 Pinned versions (latest on PyPI at planning time)

| Package | Version | Note |
|---|---|---|
| Python | **3.12** (image `python:3.12-slim-bookworm`, pinned by digest in Phase 0) | Keep 3.12: `random.Random` sequences and wheels are stable on it. |
| crewai | **1.15.36** | Pulls `chromadb==1.5.9`, `pydantic==2.12.5`, `openai` (unused). **`litellm` is not pulled.** |
| chromadb (client) | **1.5.9** (transitive via crewai, pin explicitly) | Server image **`chromadb/chroma:1.5.9`** (same release tag). *Amended (D-015):* the Rust server versions itself independently (API reports `1.0.0`, CLI `1.4.4`), so compatibility is proven by an HttpClient round-trip contract test, not by comparing version strings. |
| autogen-agentchat / autogen-core | **0.7.5** | `autogen-ext` is not needed (we write our own client). |
| langchain-core | **1.7.6**, pin `<2.0` | `InMemoryChatMessageHistory` is deprecated since 1.6.4 and **removed in 2.0**. |
| sentence-transformers | **6.2.0** | Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim), revision pinned to a commit SHA in Phase 0. |
| torch | **2.15.0 CPU wheel** from `https://download.pytorch.org/whl/cpu` | Avoids roughly 2 GB of CUDA wheels. |
| fastapi 0.162.0 · uvicorn 0.54.0 · pydantic 2.12.5 (as resolved) | | |

The resolver (`uv pip compile --generate-hashes`) is the source of truth. If it resolves different minor versions, the lockfile wins and `docs/DECISIONS.md` records why.

### 2.2 CrewAI facts that drive the mock design
1. `crewai.llms.base_llm.BaseLLM` has exactly **one abstract method: `call`**. Its signature:
   `call(self, messages: str | list[LLMMessage], tools=None, callbacks=None, available_functions=None, from_task=None, from_agent=None, response_model=None) -> str | Any`. `BaseLLM(model="mock-llm")` constructs it. *Amended (D-014.1):* `BaseLLM` does **not** define `supports_function_calling()`. CrewAI uses native tool calling only if the LLM has a callable one returning True, so a custom LLM is on the text ReAct path. But the output converter calls the method unconditionally, so the adapter must define it and return `False` (§7.8.1). A tool-less agent with `output_pydantic` receives `response_model=<Model>` in `call()`; a tool-using agent receives `None` (D-014.3).
2. **ReAct message layout observed after one tool call** (the key to avoiding pitfall #1):
   ```
   [0] system    : role/goal/backstory + tool list + format block that contains the literal
                   "Observation: the result of the action"      <-- TEMPLATE, never parse
   [1] user      : "Current Task: ..." (our task description + expected output)
   [2] assistant : "<our previous output>\nObservation: <TOOL RESULT>"   <-- CrewAI appends the
                   observation INTO OUR OWN assistant message
   [3] user      : "Analyze the tool result. If requirements are met, provide the Final Answer.
                   Otherwise, call the next tool. Deliver only the answer without meta-commentary."
                   (en.json key `post_tool_reasoning`)
   ```
   **Pitfall #3, not in the brief.** A parser that takes "everything after the last `Observation:`" across concatenated messages also swallows the `post_tool_reasoning` nudge. The spike reproduced this: the nudge text leaked into the final answer. **Rule:** read the tool result only from the assistant message that carries our unique action marker, and only up to the end of that message.
3. `Agent.cache` **defaults to `True`**. CrewAI's own tool-result cache would mask repeated tool invocations and fake our cache demo. **Every agent must be built with `cache=False`.** `Crew.cache`, `Crew.memory`, and `Crew.planning` default to `False`. Set them explicitly anyway.
4. `Agent.allow_delegation` defaults to `False`. Set it explicitly. Delegation would let the Composer route work to the Lookup agent and blur least autonomy.
5. `Agent.max_iter` defaults to 25 and `max_retry_limit` to 2. Set `max_iter=4` so a mock bug cannot cause long loops.
6. Telemetry and tracing are referenced by `CREWAI_DISABLE_TELEMETRY`, `OTEL_SDK_DISABLED` and `CREWAI_TRACING_ENABLED` in `crewai/telemetry/*` and `events/listeners/tracing/*` (including a "first-time trace" handler). Set all three before `import crewai` (§5).
7. `Task(output_pydantic=Model)` parsed our JSON Final Answer straight into `result.pydantic` **without an extra LLM call** (contract test: exactly 2 calls for one tool round-trip). *Amended (D-014.2):* the converter re-ask path is real. A Final Answer that is not schema JSON costs 3 extra calls (`[system=format instructions, user=raw]`), then `pydantic=None` with `raw` kept. The mock must always emit schema JSON, and exact call counts (Phase 6) catch regressions. A task's `context` reaches the next agent after `This is the context you're working with:` in its user message (D-014.4).
8. CrewAI logs `Using an approximate input token count ... one token per four characters` for a custom LLM. Our budget estimator uses the same heuristic (`ceil(chars/4)`), so our numbers and CrewAI's agree.

### 2.3 Autogen facts
1. `ChatCompletionClient` abstract members: `create`, `create_stream`, `close`, `actual_usage`, `total_usage`, `count_tokens`, `remaining_tokens`, `capabilities`, `model_info`.
2. `RoundRobinGroupChat(participants, *, termination_condition=None, max_turns=None, custom_message_types=None, ...)`. With `max_turns=2`, the run produced exactly `[TextMessage(user task), TextMessage(Reviewer), StructuredMessage[Verdict](Editor)]`, and `stop_reason="Maximum number of turns 2 reached."`
3. With `AssistantAgent(..., output_content_type=Verdict)`, the editor's `create()` receives `json_output=Verdict`. The client must return a JSON string, and `model_info` must declare `json_output=True, structured_output=True`.
4. `custom_message_types=[StructuredMessage[Verdict]]` is required on the team (per the brief). *Amended (D-014.5):* when it is omitted, `team.run()` raises **`RuntimeError`**, whose message carries the registry's `ValueError: Message type … is not registered`. The contract test pins that.

### 2.4 Other facts and risks to verify in Phase 0 (contract tests)
- Chroma `configuration={"hnsw": {"space": "cosine"}}` returns **cosine distance**, so `similarity = 1 − distance`. Verify numerically against `numpy` cosine on 3 vectors.
- Chroma must **never** fall back to its default ONNX embedding function, which downloads a model at first use. We always pass `embeddings=` explicitly and create collections with `embedding_function=None`. Verify under `network_mode: none`.
- Chroma telemetry: `ANONYMIZED_TELEMETRY=False` on the client `Settings` and the server env.
- `RunnableWithMessageHistory` emits `LangChainDeprecationWarning` (expected, do not silence). `InMemoryChatMessageHistory` emits its own deprecation warning too. Both still work. *Amended (D-014.7):* the latter warns once per process, the former on every construction.
- Starlette/uvicorn exception types raised on **receive** and on **send** after a client disconnect. The contract test pins what the installed versions raise. *Amended (D-014.6):* with uvicorn `ws=auto` (→ websockets-sansio), receive raises `WebSocketDisconnect(1000)`; the first send raises `WebSocketDisconnect(1006)`; later sends raise `starlette.websockets.WebSocketDisconnected` (a `RuntimeError` subclass). Catch both.
- `crew.kickoff()` completes with stdin closed, no TTY, and no network within 20 s. This guards against an interactive trace prompt hanging the container.

---

## 3. Deployment architecture (Docker)

### 3.1 Container topology

```
                         host (127.0.0.1:8000 only)
                                   │
                    ┌──────────────┴──────────────┐   network: edge (bridge, egress allowed)
                    │            api              │   FastAPI + uvicorn --workers 1
                    │  guardrails · memory · cache│   in-process session memory & cache
                    │  CrewAI crew · Autogen team │   (single process by design, §8)
                    └──────────────┬──────────────┘
   network: backend (internal: true — NO internet route)
        ┌──────────────┬───────────┴───────┬─────────────────┬─────────────────┐
        │              │                   │                 │                 │
   ┌────┴────┐   ┌─────┴─────┐      ┌──────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐
   │ chroma  │   │  indexer  │      │   runner    │   │    tests    │   │ api-offline │
   │ 1.5.9   │   │ one-shot  │      │ transcripts │   │ pytest (dev │   │ api with NO │
   │ server  │   │ job: KB → │      │ eval, P/R,  │   │ image)      │   │ edge network│
   │ volume: │   │ chunks →  │      │ calibration │   │ unit/contr/ │   │ (offline    │
   │ chroma- │   │ embed →   │      │ (profile:   │   │ integ/e2e   │   │ proof only) │
   │ data    │   │ upsert x2 │      │ tools)      │   │ (profile:   │   │ (profile:   │
   └─────────┘   └───────────┘      └─────────────┘   │ test)       │   │ offline)    │
                                                       └─────────────┘   └─────────────┘
```

| Service | Image / target | Command | Networks | Depends on | Profile |
|---|---|---|---|---|---|
| `chroma` | `chromadb/chroma:1.5.9` | default server | backend | — | default |
| `indexer` | `credagent:runtime` | `python -m scripts.indexer` (idempotent upserts, exit 0) | backend | chroma (healthy) | default |
| `api` | `credagent:runtime` | `uvicorn credagent.api.app:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log` | backend, edge | indexer (`service_completed_successfully`) | default |
| `runner` | `credagent:runtime` | `python -m scripts.<job>` via `make` targets | **backend only** | chroma, indexer | `tools` |
| `tests` | `credagent:dev` | `pytest ...` | **backend only** | chroma, indexer (integration/e2e also need `api-offline`) | `test` |
| `api-offline` | `credagent:runtime` | same as `api`, no published port | **backend only** | indexer | `test`, `offline` |

E2E tests target **`api-offline`**, not `api`. That way every graded HTTP/WS interaction runs in a container that physically cannot reach the internet.

### 3.2 Image build (`docker/Dockerfile`, multi-stage)
1. **`base`**: `python:3.12-slim-bookworm@sha256:<pinned>`. Sets `PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1` and creates user `app` (uid/gid 10001).
2. **`deps`**: copies `requirements.lock`, runs `pip install --require-hashes --no-deps -r requirements.lock` (CPU torch index declared in the lock header).
3. **`model`**: downloads `all-MiniLM-L6-v2` at the pinned revision and `save()`s it to `/models/all-MiniLM-L6-v2`. This is the only stage with network.
4. **`runtime`**: deps + `/models` + `src/`, `dataset.py`, `kb/`, `config/`, `scripts/`, `eval/`. `pip install --no-deps .`. Sets the runtime ENV (§5), `USER 10001`, and a `HEALTHCHECK` that uses Python `urllib` against `/healthz` (no curl in the slim image).
5. **`dev`**: runtime + `requirements-dev.lock` (pytest, pytest-cov, pytest-socket, pytest-timeout, hypothesis, httpx, websockets, ruff, mypy, types-PyYAML). Source and tests are **bind-mounted read-only** in the `tests` service, so the edit→test loop needs no rebuild.

`.dockerignore` excludes `.git`, `transcripts/`, `eval/results/`, `**/__pycache__`, `.env`.

### 3.3 Compose hardening (all app services)
`read_only: true`; `tmpfs: [/tmp, /home/app]`; `cap_drop: [ALL]`; `security_opt: [no-new-privileges:true]`; `user: "10001:10001"` (for services that write bind mounts, `${UID}:${GID}` from the Makefile); `mem_limit: 2g`; `pids_limit: 512`. Writable paths are only the `logs` volume (`/data/logs`), `/tmp` (CrewAI storage via `CREWAI_STORAGE_DIR=/tmp/crewai`), and the bind mounts `./transcripts` and `./eval/results` for `runner`. `networks.backend.internal: true`. The `api` port binds to `127.0.0.1` only.

### 3.4 Volumes
`chroma-data` (vectors, persists across restarts), `logs` (JSONL). Dataset and KB live in the image (immutable, versioned with git).

### 3.5 Startup sequence and readiness
`chroma` healthy → `indexer` upserts both collections, writes `kb_version` (sha256 over KB files + chunk params) into a `kb_meta` collection (*amended, D-019:* Chroma rejects the name `_meta`), and exits 0 → `api` starts. Steps: load settings → assert `MOCK_LLM` → load the embedder from local path → connect to Chroma (bounded retry: 10 attempts, exponential backoff capped at 5 s) → assert both collections exist with the expected count and `kb_version` → mark ready. `/healthz` = process alive. `/readyz` = embedder loaded + Chroma reachable + collections valid.

If the Chroma image has no shell or HTTP tool for a container healthcheck, use `depends_on: service_started` and rely on the indexer's retry loop. Record which one was used in `DECISIONS.md`.

---

## 4. Repository layout

```
cred-loan-support-agent/
├── CLAUDE.md                     # implementer operating rules
├── README.md                     # written in final phase (spec in §12)
├── Makefile                      # the ONLY entry point for humans and Claude Code
├── compose.yaml
├── .env.example                  # copied to .env by `make init`; no secrets of value
├── pyproject.toml                # project meta, ruff, mypy, pytest, coverage config; py-modules=["dataset"]
├── requirements.in / requirements-dev.in
├── requirements.lock / requirements-dev.lock   # uv-compiled, --generate-hashes
├── docker/Dockerfile
├── dataset.py                    # REQUIRED at repo root by the brief; stdlib only
├── kb/                           # 13 markdown docs with YAML front matter
├── config/
│   ├── retrieval.json            # generated by calibration: threshold, top_k, chunk params, measured sims
│   └── escalation.json           # generated: weights, p80, threshold
├── src/credagent/
│   ├── __init__.py               # sets telemetry env defaults BEFORE anything imports crewai
│   ├── settings.py               # pydantic-settings; single source of config
│   ├── schemas.py                # all domain Pydantic models (§7.11)
│   ├── text.py                   # tokenizer, sentence splitter, number extraction, normalisation
│   ├── kb/{loader.py,chunking.py}
│   ├── rag/{embedder.py,store.py,retriever.py,generator.py,calibration.py,evaluation.py}
│   ├── llm/{engine.py,crewai_llm.py,autogen_client.py,judge.py}
│   ├── tools/{rag_tool.py,lookup_tool.py,escalation.py,acl.py}
│   ├── crew/{templates.py,agents.py,tasks.py,factory.py}
│   ├── memory/session.py
│   ├── guardrails/{pii.py,injection.py,groundedness.py}
│   ├── review/{models.py,team.py}
│   ├── governance/{budget.py,policy.py}
│   ├── cache/response_cache.py
│   ├── pipeline/{context.py,orchestrator.py}
│   ├── observability/{logging.py,metrics.py}
│   └── api/{app.py,models.py,routes_http.py,routes_ws.py,auth.py,errors.py,middleware.py}
├── scripts/                      # thin CLIs over src/: indexer, validate_dataset, calibrate_threshold,
│                                 # eval_chunking, run_eval, make_transcripts, pii_canary_scan, offline_probe
├── eval/{testset.yaml,judge_prompt.md,results/}
├── transcripts/                  # GENERATED, committed, golden-tested
├── tests/{conftest.py,unit/,contract/,integration/,e2e/,security/,golden/,fixtures/}
└── docs/{ARCHITECTURE.md,IMPLEMENTATION_PLAN.md,GOVERNANCE.md,DECISIONS.md,BUG_LOG.md}
```

Module dependency rule (enforced by an import-linter-style unit test that walks the AST): `api → pipeline → {crew, review, guardrails, memory, cache, governance} → {tools, rag, llm} → {kb, text, schemas, settings}`. Lower layers never import upper layers. `dataset.py` imports nothing from `credagent`.

---

## 5. Configuration

`credagent/settings.py` uses `pydantic-settings` and is frozen. Every knob lives here. No `os.getenv` anywhere else, except `credagent/__init__.py`, which does `os.environ.setdefault(...)` for the telemetry flags so they are set even when a script imports crewai first.

| Variable | Default | Purpose |
|---|---|---|
| `MOCK_LLM` | `true` | Must be `true` for graded runs. Scripts in `scripts/` refuse to run otherwise. |
| `LLM_PROVIDER` | `mock` | Only `mock` is implemented. A real provider is an optional stretch goal behind this flag (§13). |
| `CREWAI_DISABLE_TELEMETRY` / `OTEL_SDK_DISABLED` / `CREWAI_TRACING_ENABLED` | `true` / `true` / `false` | CrewAI egress off |
| `ANONYMIZED_TELEMETRY` | `False` | Chroma egress off |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `HF_HUB_DISABLE_TELEMETRY` | `1` | HF egress off |
| `TOKENIZERS_PARALLELISM` / `OMP_NUM_THREADS` | `false` / `1` | Determinism, no fork warnings |
| `EMBED_MODEL_PATH` | `/models/all-MiniLM-L6-v2` | Loaded by path, never by hub name |
| `CHROMA_HOST` / `CHROMA_PORT` | `chroma` / `8000` | |
| `COLLECTION_FIXED` / `COLLECTION_SENTENCE` | `kb_fixed_v1` / `kb_sentence_v1` | |
| `DEPLOYED_STRATEGY` | set in Phase 4 from P/R results (expected `sentence`) | Collection used by the crew |
| `TOP_K` | `3` | |
| `API_KEY` / `ADMIN_API_KEY` | dev values in `.env.example` | `X-API-Key` header; admin key for `/v1/documents` |
| `LOG_PATH` | `/data/logs/requests.jsonl` | |
| `MAX_MESSAGE_CHARS` | `20000` | Pydantic limit, deliberately **above** the budget cap (§7.14.3) |
| `BUDGET_MAX_TOKENS` / `BUDGET_MAX_COST_USD` / `PRICE_PER_1K_TOKENS_USD` | calibrated in Phase 9 (initial `8000` / `0.016` / `0.002`) | Runtime budget |
| `CACHE_MAX_ENTRIES` / `CACHE_TTL_SECONDS` | `512` / `3600` | |
| `REQUEST_TIMEOUT_S` | `30` | Whole-pipeline timeout |
| `CREWAI_STORAGE_DIR` / `HOME` | `/tmp/crewai` / `/home/app` | Writable under read-only root fs |

---

## 6. Request lifecycle (the orchestrator)

`credagent/pipeline/orchestrator.py::Orchestrator.handle(raw_text, session_id, transport) -> AskResponse` is the **only** path from the API to the agent. HTTP and WebSocket both call it. Every stage records its timing into `RequestContext.timings_ms`.

```
 0  RequestContext created: trace_id, session_id, BudgetMeter, ToolCallLedger, timings
 1  PII MASK          masked = PiiMasker.mask(raw_text)   ← raw_text is dropped here; nothing downstream
                                                            (memory, cache, crew, logs) ever sees it
 2  INJECTION CHECK   if InjectionDetector.detect(masked).blocked → refusal response (outcome=blocked_injection)
 3  BUDGET PREFLIGHT  if BudgetPolicy.preflight(masked).exceeds → BudgetExceeded (HTTP 413, outcome=budget_rejected)
 4  MEMORY RESOLVE    via RunnableWithMessageHistory(session_id): history → standalone query
                      (resolves "it/that loan/my application" → record id / loan category from history)
 5  CREW KICKOFF      fresh Crew built per request (factory) with per-request MockCrewLLM instances
                      bound to this RequestContext; run in a worker thread with REQUEST_TIMEOUT_S
                        Retrieval Agent → rag_lookup(query)  → ResponseCache → Retriever → GroundedGenerator
                        Lookup Agent    → check_loan_application_status(record_id)   (only if a record id exists)
                        Composer Agent  → CrewResponse JSON (Task.output_pydantic)
 6  SCHEMA VALIDATE   CrewResponse.model_validate_json(result.raw)  (and compare with result.pydantic)
 7  REVIEW            Autogen RoundRobinGroupChat(Reviewer, Final-Editor, max_turns=2) → ReviewVerdict
 8  GROUNDEDNESS GATE GroundednessGuard.check(final_answer, retrieval_trace) → may convert to refusal
 9  BUDGET POSTCHECK  if meter.exceeded (runtime overrun) → BudgetExceeded
10  MEMORY APPEND     history.add(HumanMessage(masked), AIMessage(final_answer))
11  RESPONSE          AskResponse validated, returned
12  LOG (finally)     exactly one JSON line per request, masked, including failures
```

Ordering rationale:
- Masking happens first, so injection detection, budget and everything else work on masked text.
- Review (7) runs **before** the groundedness gate (8). The gate is the last safety net, and the review must see the raw draft to revise it.
- Memory append (10) happens only after a successful answer or refusal. Blocked-injection turns are **not** stored, so a poisoned turn cannot steer later turns.

---

## 7. Component specifications

### 7.1 Dataset — `dataset.py` (Task 1)
- Stdlib only (`random`, `dataclasses`, `json`, `hashlib`, `collections`). Exposes `SEED`, `CATEGORY_WEIGHTS`, `STATUS_WEIGHTS`, `AMOUNT_RANGES_INR`, `FRAUD_FLAG_PROBABILITY`, `N_RECORDS`, `generate(seed) -> list[dict]`, `LOAN_APPLICATIONS = generate(SEED)`, `report(records) -> dict`, `validate(records) -> list[str]` (empty list = valid), and `__main__` that prints the report and exits non-zero if invalid.
- Uses a **local** `rng = random.Random(seed)`. Never the module-global `random`.
- `N_RECORDS = 60`. Record ids `LA-0001` … `LA-0060` (regex `^LA-\d{4}$`).
- Category weights: Personal 0.30, Home 0.20, Auto 0.20, Education 0.15, Business 0.15. Status weights: Submitted 0.20, Under Review 0.30, Approved 0.20, Rejected 0.15, Disbursed 0.15. Fraud flag: Bernoulli(p = 0.18).
- `loan_amount_inr`: **per-category log-uniform** range, rounded to the nearest ₹5,000. Personal ₹50,000–₹40,00,000 · Home ₹10,00,000–₹2,00,00,000 · Auto ₹2,00,000–₹30,00,000 · Education ₹1,00,000–₹75,00,000 · Business ₹5,00,000–₹1,00,00,000. *One-sentence rationale for the README:* ranges follow typical Indian retail-lending ticket sizes per product, and log-uniform sampling reflects that small tickets are far more common than jumbo ones. The upper bound stays at 8 digits, so amounts can never collide with the ≥9-digit bank-account masking pattern (§7.12.1).
- `days_since_created = rng.randint(0, 30)`. Field order in each record is fixed.
- **Validation** (asserted by `validate` and by unit tests): ≥40 records; every given category ≥3; every given status ≥1; `flagged_for_fraud_review` share in **[10%, 30%]**; ids unique; types correct; amounts within the category range.
- **Seed procedure:** start with `SEED = 2026`. If `validate` fails, increment the seed and retry, recording every attempt (seed → failing metric) in the README's "Seed history". **Never edit records.** A golden test pins `sha256(json.dumps(LOAN_APPLICATIONS, sort_keys=True))`.

### 7.2 Knowledge base (Task 2)
13 Markdown files in `kb/`: the 12 required topics plus one status glossary that the Composer needs. Each file has YAML front matter `{doc_id, title, topic, version: 1}` and a body of **2–5 sentences**. A unit test enforces the sentence count with the project's own splitter. Every file carries this disclaimer in front matter: `disclaimer: "Fictional policy written for a capstone; not Cred's actual policy."`

The content contract below gives the facts each document must contain, so the eval gold facts stay consistent. The prose must be original.

| doc_id | Topic | Must contain (fictional facts) |
|---|---|---|
| `kb-01-loan-eligibility` | Loan eligibility by type | Personal: age 21–58, net monthly income ≥ ₹25,000, credit score ≥ 700. Home: score ≥ 725, co-applicant allowed, loan must end by age 70. Auto: score ≥ 680, income ≥ ₹20,000. Education: admission letter + parent/guardian co-borrower mandatory. Business: ≥ 3 years vintage, 2 years ITR, turnover ≥ ₹40 lakh. |
| `kb-02-emi-calculation` | EMI rules | Reducing-balance formula EMI = P·r·(1+r)^n / ((1+r)^n − 1), r = annual rate/12/100; first EMI on the 5th of the month after disbursal; broken-period interest for the gap days; EMI rounded to the nearest rupee. |
| `kb-03-credit-card-fees` | Credit-card fee structure | Joining fee ₹500 (waived on ₹10,000 spend in 60 days); annual fee ₹999 (waived on ₹1.5 lakh annual spend); forex markup 3.5%; cash-advance fee 2.5% (min ₹500); late fee slab tied to outstanding. |
| `kb-04-kyc-documents` | KYC requirements | PAN mandatory; one address proof (Aadhaar, passport, voter ID, driving licence); video KYC accepted; periodic re-KYC every 2/8/10 years for high/medium/low risk. |
| `kb-05-fraud-dispute` | Fraud-dispute resolution | Report within 3 working days → zero liability; 4–7 days → liability capped at ₹10,000; provisional credit within 10 working days; final resolution within 90 days. |
| `kb-06-account-closure` | Account closure | Request in-app or in writing; all dues cleared; closed within 7 working days; bureau updated within 30 days; ₹500 fee if closed within the first year. |
| `kb-07-interest-rate-slabs` | Interest-rate slabs | Personal 10.5–24% p.a. by score band (≥800: 10.5–12%; 750–799: 12–16%; 700–749: 16–24%); Home 8.4–9.9% floating; Auto 8.9–12.5%; Education 9.5–13%; Business 14–22%. |
| `kb-08-prepayment-penalty` | Prepayment penalties | Floating-rate home loans to individuals: no penalty; fixed-rate personal loans: 4% of the prepaid principal within the first 12 EMIs, 2% after; foreclosure allowed after 6 EMIs; auto loans allow 25% part-prepayment per year free. |
| `kb-09-minimum-balance` | Minimum balance | Savings AMB ₹10,000 metro / ₹5,000 semi-urban; shortfall charge 6% of the shortfall, capped at ₹500; salary and basic accounts exempt. |
| `kb-10-credit-score-factors` | Credit-score impact | Payment history weighs most; keep utilisation below 30%; each hard enquiry is visible for 24 months; credit mix and account age matter; a rejection itself does not lower the score, but the enquiry does. |
| `kb-11-joint-accounts` | Joint-account rules | Up to 4 holders; modes either-or-survivor, former-or-survivor, jointly; removing a holder needs consent of all holders; an NRI may be a joint holder with a resident close relative on former-or-survivor basis. |
| `kb-12-nri-eligibility` | NRI eligibility | NRE/NRO accounts need a valid passport plus visa/OCI; NRIs can take home loans up to 80% LTV but not personal loans; repayment from NRE/NRO accounts. |
| `kb-13-application-status-glossary` | Status meanings & SLAs | Submitted → reviewed within 2 working days; Under Review up to 7 working days; fraud-review flag adds up to 5 working days; disbursal within 48 h of approval; rejected applicants may reapply after 90 days. |

Kept deliberately: overlap between kb-11 and kb-12 (NRI joint holders) and between kb-07 and kb-08. These near-neighbours make P/R and the threshold meaningful.

`kb/loader.py` parses front matter (PyYAML `safe_load`), validates it against `KbDocument`, normalises Unicode (NFKC), and raises on duplicate `doc_id`.

### 7.3 Chunking (Task 3) — `credagent/kb/chunking.py`
Both strategies are pure functions `(doc: KbDocument, params) -> list[Chunk]`, where `Chunk = {chunk_id, doc_id, strategy, index, text, char_start, char_end}`.
- **Fixed-size with overlap:** `size=160` chars, `overlap=40`, with word-boundary snapping (never split a word; extend to the next whitespace, capped at size+20). Invariants: full coverage of the body, consecutive overlap ≥ 1 char unless it is the final chunk, no empty chunks.
- **Sentence-based:** one chunk per sentence from `text.split_sentences`. The splitter must handle `p.a.`, `e.g.`, `i.e.`, `Rs.`, `No.`, decimals (`10.5%`), `₹1.5 lakh`, and ellipses. Its abbreviation list lives in `text.py` and has dedicated tests.
- `chunk_id = f"{doc_id}::{strategy}::{index:03d}"`. Deterministic ids make `upsert` idempotent.
- Metadata stored in Chroma: `doc_id, strategy, index, char_start, char_end, kb_version, title`.

### 7.4 Embeddings and vector store
- `rag/embedder.py`: `Embedder` wraps `SentenceTransformer(EMBED_MODEL_PATH, device="cpu")`. `encode(texts) -> list[list[float]]` with `normalize_embeddings=True` and `batch_size=32`. A process-wide singleton guarded by a `threading.Lock` (encode is serialised: cheap at this scale and avoids torch thread contention). `torch.set_num_threads(1)`.
- `rag/store.py`: `VectorStore` over `chromadb.HttpClient(host, port, settings=Settings(anonymized_telemetry=False))` in prod and `chromadb.EphemeralClient(...)` in unit tests. **Same class, injected client.** Collections are created with `get_or_create_collection(name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}, metadata={...})`. Always `upsert(ids, embeddings, documents, metadatas)`. Query returns `RetrievedChunk{chunk_id, doc_id, text, similarity = 1 - distance, rank}`, sorted by similarity desc, tie-broken by `chunk_id` for determinism.
- `scripts/indexer.py`: load KB → chunk both ways → embed → upsert into `kb_fixed_v1` and `kb_sentence_v1` → delete stale ids no longer produced (so a changed KB never leaves orphan chunks) → write `kb_meta` (D-019) → print counts. Running it twice yields identical counts and ids (integration test).

### 7.5 Retrieval, threshold calibration, grounded generation (Task 4)
- `Retriever.retrieve(query, strategy, k=TOP_K) -> RetrievalResult{chunks, top1_similarity, strategy}`.
- **Calibration** (`rag/calibration.py`, `scripts/calibrate_threshold.py`): `≥5` in-scope and `≥5` out-of-scope queries (more than the brief's 3/2 minimum, to make the gap estimate robust). Out-of-scope includes **near-domain** traps, e.g. "What is Cred's share price today?", "Recommend a good mutual fund", "What's the weather in Mumbai?", "Write a poem about loans", "How do I reset my Netflix password?". The script measures top-1 cosine per query for **both** collections and computes `max_oos`, `min_in`, `gap`, and `threshold = round((max_oos + min_in)/2, 3)`. It writes everything to `config/retrieval.json`. Exit criterion: `gap ≥ 0.05` on the deployed collection. If not, improve KB wording or query design and re-run; never pick a threshold inside an overlap. **Calibration queries are disjoint from the 5 Task-4 demo queries and the 15 eval queries** (unit test asserts disjointness), so the threshold is not tuned on the test set.
- **GroundedGenerator** (`rag/generator.py`) is called through an `LLMClient` protocol whose mock implementation is `MockEngine.generate_grounded`:
  1. If `top1_similarity < threshold` → return the fallback `"I don't know based on the available policy documents."` with `grounded=False`.
  2. Otherwise, extractive synthesis: split the retrieved chunks into sentences and dedupe (overlapping fixed chunks repeat text). Score each sentence by `0.7·query-token recall + 0.3·chunk similarity`. Take the top 2 sentences with score ≥ `MIN_SENTENCE_SCORE`, ordered by original document position. Join them. Every emitted sentence is **verbatim from context**, so the answer is grounded by construction.
  3. Return `GeneratedAnswer{answer, sources:[SourceRef], grounded, top1_similarity}`.
- The generator never sees anything except the masked query and the retrieved chunks.

### 7.6 Chunking evaluation (Task 5) — `rag/evaluation.py`
- 5 in-scope queries, each with gold `relevant_doc_ids`. At least 2 queries have **two** relevant docs (e.g. "Can an NRI be a joint account holder?" → {kb-11, kb-12}; "What prepayment penalty applies and what rate slab applies to a personal loan?" → {kb-07, kb-08}).
- For each collection: retrieve top-k chunks → map to `doc_id` → **dedup preserving rank** → `P = |rel ∩ ret| / |ret|`, `R = |rel ∩ ret| / |rel|`. The report prints the arithmetic literally, e.g. `P = 1/2 = 0.500, R = 1/1 = 1.000`, then macro-averages per collection, then a 2–3-sentence recommendation generated from a template filled with the measured numbers. If the numbers tie, the tie-breaker is mean top-1 similarity, then mean retrieved-doc count (fewer = more precise context). The result sets `DEPLOYED_STRATEGY` in `config/retrieval.json`.
- P/R math is a pure function with unit tests on hand-computed cases, including the empty-retrieval edge case (define P=0 when |ret|=0 and log a warning; it must not occur for in-scope queries).

### 7.7 Lookup tool and escalation score (Task 6)
`check_loan_application_status(record_id: str) -> dict`
- Normalises input (`strip().upper()`) and validates `^LA-\d{4}$`. Invalid → `{"found": False, "error": "invalid_record_id"}`. Unknown → `{"found": False, "error": "not_found"}`. Never raises into CrewAI.
- Found → `{"found": True, "record_id", "category", "status", "loan_amount_inr", "days_since_created", "flagged_for_fraud_review", "escalation_score", "escalation_recommended", "escalation_threshold"}`. The brief's required minimum is `status`, `loan_amount_inr`, `escalation_score`. The extras serve the Composer and transcripts.
- **Formula** (`tools/escalation.py`, weights in `config/escalation.json`):
  `recency = min(max(days_since_created, 0), 30) / 30`
  `escalation_score = round(0.4 · flagged + 0.6 · recency, 4)`  ∈ [0, 1]
- **Threshold:** `T = round(0.6 · P80(days_since_created) / 30, 4)`, where P80 is the **nearest-rank** 80th percentile of the generated dataset (`sorted(days)[ceil(0.8·n) − 1]`). Escalate iff `score ≥ T`. Interpretation, written in the README with the real numbers:
  - An unflagged application escalates only when it is at least as old as the oldest 20% of applications (stale-queue rule).
  - A flagged application escalates once `days ≥ 30·(T − 0.4)/0.6` (a few days, so fraud flags are reviewed quickly).
  - This is a weighted blend, not a boolean OR. A brand-new flagged application (score 0.4) does **not** auto-escalate if `T > 0.4`.
- Tests: bounds (property test over all 0–30 × {T,F}), monotonic in days and in flag, exact values for 4 hand-computed cases, `T` recomputed from the dataset equals `config/escalation.json`, and the percentile method matches its hand-computed value on a fixture list.
- Known limitation (documented): the score ignores status. A flagged, already disbursed loan still scores on age alone. Production would add status weighting.

### 7.8 MOCK_LLM — one deterministic engine, three adapters
```
              ┌────────────────────────── MockEngine (llm/engine.py) — pure, stateless ──────────────────────────┐
              │ generate_grounded(query, chunks) · support_report(answer, context) · resolve_followup(q, history)│
              │ classify_intent(...) · compose(retrieval_json, lookup_json, query) · review(draft, context)       │
              │ edit(review_json, draft) · judge(judge_prompt) — all built on text.py primitives                  │
              └──────────────┬───────────────────────────┬──────────────────────────────┬────────────────────────┘
                 MockCrewLLM(BaseLLM)          MockAutogenClient(ChatCompletionClient)      MockJudgeClient
                 (crewai_llm.py)               (autogen_client.py)                           (judge.py)
```

**text.py primitives:** `normalize` (NFKC, casefold, strip zero-width chars), `content_tokens` (alphanumerics minus a fixed stopword list; keeps tokens like `10.5%`, `₹10,000`, `24`), `extract_numbers` (canonicalises `₹1.5 lakh`→150000, `10,000`→10000, `2%`→`2%`), `split_sentences`.

**Support scoring** (shared by generation, the groundedness guard, the reviewer and the judge): a sentence S is *supported* by context C iff (a) `token_recall(S, best_matching_context_sentence) ≥ 0.6` **and** (b) every number in S appears among the numbers of C. Rule (b) catches fabricated numbers like "0% interest for life".

#### 7.8.1 `MockCrewLLM(BaseLLM)` — avoiding all three pitfalls
Constructed **per agent, per request**: `MockCrewLLM(model="mock-llm", persona=<retrieval|lookup|composer>, tools=<that agent's tool objects>, ctx=<RequestContext>)`. `supports_function_calling() → False`. `supports_stop_words() → True`. `get_context_window_size() → 32768`.

`call(messages, ..., from_agent=None, response_model=None)` algorithm:
1. Normalise `messages` into a list. Charge `ceil(total_chars/4)` tokens to `ctx.budget`. If that exceeds the budget, set `ctx.budget.exceeded = True` and immediately return `Final Answer: {"refused": true, "refusal_reason": "budget_exceeded", ...}` (valid schema JSON). **Do not raise**: CrewAI may catch and retry; the orchestrator checks the flag after kickoff (§6 step 9).
2. **Find our own previous action.** Scan `messages` for `role == "assistant"` messages that contain the per-request marker `⟦mock-action:{ctx.trace_id}:{n}⟧`. The system template never contains this marker (pitfall #1 solved).
3. **Extract the observation** only from that assistant message: the text after the first `"\nObservation:"` **in that message**, up to its end. Ignore all later user messages (pitfall #3 solved). Parse it as JSON (our tools return JSON strings). On parse failure → return a schema-valid error Final Answer.
4. **Read the task inputs** from the `role == "user"` message with `"Current Task:"`, between our template delimiters `<<<QUERY>>>…<<<END_QUERY>>>`, `<<<RECORD_ID>>>…<<<END_RECORD_ID>>>`. For the Composer, previous task outputs are found by their JSON `"kind"` field (`"retrieval_result"`, `"lookup_result"`).
5. **Decide the next step by persona:**
   - Retrieval: no observation yet → emit `Thought: ⟦marker⟧ …\nAction: <tool.name>\nAction Input: <json>`. With an observation → `Final Answer: <RetrievalTaskOutput JSON>` (`kind="retrieval_result"`).
   - Lookup: no record id in the inputs → `Final Answer: {"kind":"lookup_result","found":false,"error":"no_record_id"}` with **no tool call**. Otherwise call the tool once, then `Final Answer` with the observation passed through.
   - Composer: no tools. Emits `Final Answer: <CrewResponse JSON>` built by `MockEngine.compose`.
6. **Argument dispatch from the declared schema (pitfall #2 solved):** for each tool in `self.tools`, read `tool.args_schema.model_fields`. Build arguments from a resolver map `{"query": standalone_query, "record_id": record_id}`. A tool is callable only if every **required** field resolves. The tool name is copied from `tool.name` and never inspected for meaning. A test proves this with a tool *named* `rag_lookup` whose schema is `{record_id}`, and a tool named `lookup_policy` whose schema is `{query}`; both must be dispatched correctly.
7. Never emit the string `Observation:` ourselves (CrewAI stop word).
8. Guard: if the step counter exceeds 3, emit a schema-valid error Final Answer.

#### 7.8.2 `MockAutogenClient(ChatCompletionClient)`
Implements all 9 abstract members. `model_info = ModelInfo(vision=False, function_calling=False, json_output=True, structured_output=True, family="unknown")`. Usage is counted with the same chars/4 heuristic and charged to the request's `BudgetMeter`. `create_stream` yields one final `CreateResult` (it is not used, but it must not raise). Persona `reviewer` or `editor`:
- **reviewer** reads the task message (`source == "user"`) between `<<<DRAFT>>>`/`<<<CONTEXT>>>` delimiters and runs `MockEngine.review`, which produces per-sentence support. It returns text whose last line is `REVIEW_JSON: {"unsupported": [...], "supported_count": n, "total": m}`.
- **editor** reads only the reviewer's message (`source == "Policy_Compliance_Reviewer"`), parses `REVIEW_JSON`, and returns `ReviewVerdict` JSON:
  - no unsupported sentences → `approved=true`, `final_answer = draft` **byte-identical**, reason "All N sentences are supported by the retrieved context."
  - otherwise → `approved=false`, `final_answer` = draft with the unsupported sentences removed (order preserved). If nothing remains, the safe fallback text. Reason lists the removed sentences.

#### 7.8.3 `MockJudgeClient` — LLM-as-judge (§7.18)
Receives the rendered `eval/judge_prompt.md` (a real rubric prompt with `<<<QUESTION>>>`, `<<<CONTEXT>>>`, `<<<ANSWER>>>`, `<<<GOLD_FACTS>>>`, `<<<EXPECTED_BEHAVIOUR>>>` sections). It parses those sections and returns `JudgeScores` JSON. The prompt template is what a real judge model would get. The mock only implements the rubric deterministically.

### 7.9 CrewAI crew (Task 7) — `credagent/crew/`
- `factory.build_crew(ctx, strategy) -> Crew`. A fresh Crew is built per request: object construction is cheap, and it avoids cross-request state.
  - **Retrieval Agent** (`role="Policy Retrieval Specialist"`): `tools=[RagTool]`.
  - **Lookup Agent** (`role="Loan Application Lookup Specialist"`): `tools=[LoanStatusTool]`.
  - **Response Composer** (`role="Response Composer"`): `tools=[]`.
  - All agents: `llm=MockCrewLLM(...)`, `cache=False`, `allow_delegation=False`, `max_iter=4`, `max_retry_limit=1`, `verbose=False`, `memory` unset/False.
  - Tasks (sequential): `retrieval_task`, `lookup_task`, `compose_task(context=[retrieval_task, lookup_task], output_pydantic=CrewResponse)`.
  - `Crew(agents=[…3], tasks=[…3], process=Process.sequential, cache=False, memory=False, planning=False, verbose=False)`.
- Task descriptions come from `crew/templates.py` and embed the delimiters from §7.8.1 step 4. **Inputs are masked text only.**
- `RagTool(BaseTool)`: `name="rag_lookup"`, `args_schema=RagQuery{query: str (1..2000)}`. `_run` → `ResponseCache.get_or_compute(...)` → returns `json.dumps(RetrievalToolOutput)` and appends to `ctx.ledger`.
- `LoanStatusTool(BaseTool)`: `name="check_loan_application_status"`, `args_schema=RecordIdArg{record_id: str}`, **owner-bound** (§7.14.1). `_run` delegates to the pure function and records it in the ledger.
- The kickoff runs in `anyio.to_thread.run_sync` under `anyio.fail_after(REQUEST_TIMEOUT_S)`. CrewAI calls are synchronous and CPU-bound, so the event loop must stay free for other clients and WebSocket handling. A timeout returns 504 `PIPELINE_TIMEOUT`. The worker thread finishes on its own; its result is discarded and logged.
- The `ToolCallLedger` entries `(agent_role, tool_name, args, duration_ms, cache_hit)` appear in transcripts 08/09 and in the response's `tools_invoked`. That is how "both tools actually invoked" is proven.

### 7.10 Session memory (Task 8) — `credagent/memory/session.py`
- `SessionStore`: `dict[session_id, InMemoryChatMessageHistory]` behind a `threading.Lock`, plus a **per-session `asyncio.Lock`**, so two concurrent requests in one session are serialised (different sessions run in parallel). LRU cap of 1000 sessions, idle TTL of 1 h. `reset(session_id)` deletes the history.
- `RunnableWithMessageHistory(RunnableLambda(core_turn), get_session_history=store.get, input_messages_key="input", history_messages_key="history")`. `core_turn({"input": masked, "history": [...]})` runs pipeline stages 4–9 and returns an `AIMessage(content=final_answer, additional_kwargs={"record_id":…, "category":…})`. Writing the history is the Runnable's job, which keeps the brief's mechanism genuine.
- **Follow-up resolution** (`MockEngine.resolve_followup`) derives state **only from the history messages** (never from a side dict), so a fresh session demonstrably has none:
  - If the current query has no record id and references a prior application ("it", "this application", "my application", "that loan", "the same one", "its"), take the most recent record id and category from history `additional_kwargs`.
  - Pronouns referring to the loan type ("eligibility for it") are rewritten into a standalone query ("eligibility for a Home Loan").
  - No usable history → status-type questions without an id get a clarifying answer: "Please share your application id (format LA-0000)."
- Demo (transcript 10): T1 "What is the status of application LA-0007?" → T2 "What are the eligibility criteria for it?" (resolved via category) → T3 "And does it carry a prepayment penalty?" Transcript 11: a fresh session asks T2 directly and gets the clarifying answer, with the history length printed as 0.
- `DELETE /v1/sessions/{id}` resets memory.

### 7.11 Schemas (Task 9) — `credagent/schemas.py`
All models use `ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)`.
```python
class SourceRef(BaseModel):       doc_id: str; chunk_id: str; similarity: confloat(ge=-1, le=1)
class ApplicationView(BaseModel): record_id: constr(pattern=r"^LA-\d{4}$"); category: LoanCategory; status: LoanStatus
                                  loan_amount_inr: conint(gt=0); days_since_created: conint(ge=0, le=30)
                                  flagged_for_fraud_review: bool
class Escalation(BaseModel):      score: confloat(ge=0, le=1); threshold: confloat(ge=0, le=1); recommended: bool
class CrewResponse(BaseModel):    kind: Literal["crew_response"]
                                  intent: Literal["policy","status","mixed","clarify","out_of_scope","refused"]
                                  answer: constr(min_length=1, max_length=4000)
                                  sources: list[SourceRef]; application: ApplicationView | None; escalation: Escalation | None
                                  grounded: bool; refused: bool; refusal_reason: RefusalReason | None
                                  # model_validator: refused ⇒ refusal_reason set; intent=='status' ⇒ application set;
                                  # grounded ⇒ (sources non-empty or intent in {'status','clarify'})
class ReviewVerdict(BaseModel):   approved: bool; final_answer: str; reason: str          # exactly the brief's fields
class JudgeScores(BaseModel):     accuracy/grounding/completeness/safety: confloat(ge=0, le=1); rationale: str
```
`LoanCategory` and `LoanStatus` are `Literal`s of the given vocabularies, shared with `dataset.py` through a test that asserts equality (no circular import). The orchestrator validates `result.raw` with `CrewResponse.model_validate_json` and asserts it equals `result.pydantic`. A mismatch is a 500 `SCHEMA_VIOLATION` (and a bug). The mock LLM also receives `response_format=CrewResponse` as an attribute, matching the brief's wording.

### 7.12 Guardrails (Task 10)

#### 7.12.1 PII masking — `guardrails/pii.py`
Applied in strict precedence order on NFKC-normalised text. Each match is replaced with a typed token. Replacements are computed on the original string with non-overlapping spans, longest match first:

| # | Type | Pattern (illustrative; finalise with tests) | Token |
|---|---|---|---|
| 1 | PAN | `\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b` (case-insensitive; also tolerate spaces/hyphens between groups) | `[PAN_REDACTED]` |
| 2 | Aadhaar (grouped) | `\b[2-9]\d{3}[\s-]\d{4}[\s-]\d{4}\b` | `[AADHAAR_REDACTED]` |
| 3 | Bank account (keyword-anchored) | `(?i)\b(?:a/?c|acct|account)(?:\s*(?:no|number|#))?\.?\s*[:\-]?\s*(\d{9,18})\b` (mask the group only) | `[BANK_ACCOUNT_REDACTED]` |
| 4 | Aadhaar (contiguous) | `\b[2-9]\d{11}\b` | `[AADHAAR_REDACTED]` |
| 5 | Bank account (bare) | `\b\d{9,18}\b` (fail-safe: any remaining long digit run) | `[BANK_ACCOUNT_REDACTED]` |

Design decisions (copy into `DECISIONS.md`):
- No Verhoeff checksum gate on Aadhaar: **fail-safe over-masking** beats leaking.
- Rule 5 deliberately also masks 10-digit phone numbers and other long ids. Phones are PII too, so over-masking is acceptable. Amounts never reach 9 digits (§7.1), and record ids are `LA-\d{4}`.
- `MaskResult{text, counts_by_type}`. Counts (never values) go to the log.
- **Masking points:** request text (`/ask`, WS), `/v1/documents` title and body, a logging `Filter` applied to every log record (defense in depth), and exception messages before serialisation.
- Tests: table-driven positives and negatives (`LA-0007`, `₹25,000`, `10.5%`, dates, 8-digit amounts must not mask), precedence cases (12-digit after "account" → bank; grouped 12-digit → Aadhaar), and a Hypothesis property test: for generated fabricated PAN/Aadhaar/account values embedded in random text, `raw value not in masked.text`.

#### 7.12.2 Prompt-injection detection — `guardrails/injection.py`
- Normalise: NFKC, casefold, strip zero-width and bidi chars, collapse whitespace, de-leet (`0→o, 1→i, 3→e, 4→a, 5→s, 7→t, @→a, $→s`) on a **copy** used only for detection.
- Weighted signal rules (regex families): instruction override ("ignore/disregard/forget (all|previous|prior|above) (instructions|rules)"), role hijack ("you are now", "act as", "pretend to be", "developer mode", "DAN"), system-prompt exfiltration ("reveal/print/show (your )?(system|hidden) prompt"), tool coercion ("call check_loan_application_status for every", "dump all applications/records"), delimiter smuggling (our own delimiters `<<<…>>>`, `</system>`, `### system`), and encoded payload hints (long base64 runs ≥ 40 chars). Score = sum of weights. `blocked = score ≥ 1.0`.
- **Our template delimiters in user input always block.** This prevents a user from forging `<<<RECORD_ID>>>` blocks to steer the mock.
- Applied to `/ask`, WS messages, and `/v1/documents` (KB poisoning → 422 `INJECTION_DETECTED`).
- Tests: a 25-item attack corpus must all be blocked, and a 25-item benign corpus must all pass ("Ignore the processing fee — what's the interest rate?", "act as co-applicant on a home loan?"). The FP/FN rate is printed in transcript 14.

#### 7.12.3 Output-side groundedness — `guardrails/groundedness.py`
`GroundednessGuard.check(final_answer, crew_response, retrieval_trace) -> GuardDecision`:
- For intents `policy` and `mixed`, refuse if `top1_similarity < threshold`, **or** if `supported_sentences / total_sentences < 1.0` for the policy part (every policy sentence must be supported after review). Refusal text: "I can't answer that from the policy documents I have. Please contact support." with `refusal_reason="ungrounded"`.
- Sentences produced from lookup data are checked against the lookup JSON (numbers and status must match the record).
- Demo case (transcript 15): a question about a topic absent from the KB whose wording is close to the domain (e.g. "What is the gold-loan late payment penalty?"). Phase 3 must **verify** that it falls below the threshold. If it does not, pick another deliberately chosen case and record why.

### 7.13 Autogen review stage (Task 14) — `credagent/review/team.py`
- `build_review_team(ctx) -> RoundRobinGroupChat([reviewer, editor], max_turns=2, custom_message_types=[StructuredMessage[ReviewVerdict]])`.
  - `reviewer = AssistantAgent("Policy_Compliance_Reviewer", model_client=MockAutogenClient("reviewer", ctx), system_message=…)`
  - `editor = AssistantAgent("Final_Editor", model_client=MockAutogenClient("editor", ctx), output_content_type=ReviewVerdict, system_message=…)`
- Input task text: `<<<QUESTION>>>…<<<DRAFT>>>…<<<CONTEXT>>>…` (context = retrieved chunk texts + lookup JSON).
- `await team.run(task=…)`. Then assert exactly 3 messages, and that the last one is a `StructuredMessage[ReviewVerdict]` from `Final_Editor`. Extract `.content` and re-validate it. Any deviation → treat the draft as unapproved and fall back to the safe refusal (fail closed), logged as `review_error`.
- A new team per request (no shared state). It runs in the event loop (async), after the crew thread returns.
- Demo: transcript 20 = a normal policy question (approved unchanged; asserts `final_answer == draft`). Transcript 21 = the same pipeline with a **test-only** injection hook `inject_claim="Cred also offers a 0% interest rate on all personal loans for life."` appended to the draft. The verdict is `approved=false`, and the claim is removed. The hook exists only on the `review` module's function signature used by the transcript script and tests. **It is not reachable from the API.** A unit test asserts that no API model has such a field.

### 7.14 Governance (Task 15) — `credagent/governance/`, `tools/acl.py`, `docs/GOVERNANCE.md`

#### 7.14.1 Application layer — least autonomy
Three independent guards:
1. **Policy table** `TOOL_ACL = {"check_loan_application_status": frozenset({"lookup_agent"}), "rag_lookup": frozenset({"retrieval_agent"})}` in `governance/policy.py`.
2. **Wiring-time guard:** `crew/agents.py::make_agent(agent_key, tools)` calls `acl.assert_allowed(agent_key, tool)` for every tool and raises `LeastAutonomyViolation` otherwise. `LoanStatusTool` is constructed with `owner="lookup_agent"`, and its `_run` re-checks `self.owner` against the ACL (runtime guard).
3. **Crew audit:** after `build_crew`, `acl.audit_crew(crew)` walks every agent **and every task's `tools` override** and fails if any non-owner holds the lookup tool. `allow_delegation=False` everywhere closes the indirect path.
Also: there is no HTTP endpoint that calls the lookup directly. The only path is through the Lookup Agent.
Demo (transcript 22): (a) the normal crew audit passes and prints the tool map; (b) `make_agent("composer_agent", [LoanStatusTool(...)])` raises `LeastAutonomyViolation`; (c) a tool instance smuggled onto the Composer after construction is caught by `audit_crew`; (d) calling `_run` on an instance whose owner was tampered with raises. One paragraph explains the guard.

**Known limitation to state in GOVERNANCE.md:** any caller with the API key can query any application id (IDOR). In production, lookups would be bound to the authenticated member's identity. That is out of scope here because no user identity is modelled.

#### 7.14.2 Risk classification
**High.** The system processes financial data (loan amounts, application outcomes, fraud-review flags) and accepts identity documents (PAN/Aadhaar) and bank account numbers in free text. Its answers about eligibility, rates, and penalties can influence members' financial decisions. The brief's scheme puts financial data in High. GOVERNANCE.md adds the resulting controls: masking, human escalation threshold, review stage, fail-closed defaults, budget caps, audit logs, and fabricated data only.

Four layers (GOVERNANCE.md covers all four briefly; the brief grades Application and Runtime):
- **Data:** fabricated data, masking, retention (logs rotate 7 days, memory TTL 1 h).
- **Model:** deterministic mock, pinned embedding revision.
- **Application:** least autonomy, guardrails, review.
- **Runtime:** budget, timeouts, offline networking, logging.

#### 7.14.3 Runtime layer — budget cap
- `BudgetPolicy.preflight(masked_text)`: `input_tokens = ceil(len/4)`. `projected = N_LLM_CALLS × (PROMPT_OVERHEAD_TOKENS + input_tokens + MAX_OUTPUT_TOKENS)`, where `N_LLM_CALLS = 7` (retrieval 2, lookup 2, composer 1, reviewer 1, editor 1). `PROMPT_OVERHEAD_TOKENS` is **measured** in Phase 9 from real rendered prompts (max over the eval set, rounded up to the next 100) and written to config. Reject if `projected > BUDGET_MAX_TOKENS` or `projected × price > BUDGET_MAX_COST_USD`.
- `BudgetMeter`: actual tokens charged by every mock call (CrewAI + Autogen). It is the overrun line of defense (§7.8.1 step 1).
- Rejection → HTTP **413** with `{"error_code":"BUDGET_EXCEEDED","projected_tokens":…,"budget_tokens":…,"trace_id":…}`. Logged with outcome `budget_rejected`. **No crew call is made** (asserted via mock call counter = 0).
- Calibration invariant (test): every eval query uses ≤ 50% of budget; the demo oversized request (≈ 12,000 chars, under `MAX_MESSAGE_CHARS = 20,000`) exceeds it. The order of limits matters: **body-size limit (64 KiB) > Pydantic max length (20,000 chars) > budget (≈ 8,000 tokens projected)**, so the oversized demo request is rejected by the **budget**, not by Pydantic. A test asserts this ordering from settings.

### 7.15 Response cache (Task 16) — `credagent/cache/response_cache.py`
- Caches the **grounded-generation step** (retrieval + generation) inside `RagTool`.
- Key = `sha256(f"{kb_version}|{strategy}|{threshold}|{top_k}|{normalize_query(masked_query)}")`. `normalize_query`: NFKC, casefold, collapse whitespace, strip leading/trailing punctuation, normalise quotes. Keys are built from **masked** text only, so no PII is in cache keys. Including `kb_version` means `/v1/documents` invalidates implicitly. The cache is also `clear()`ed on add.
- `OrderedDict` LRU (512) + TTL (3600 s), with `threading.Lock`. Fallback ("I don't know") results are cached too (they are deterministic), but **errors are never cached**.
- Metrics counters (`observability/metrics.py`, thread-safe): `retrieval_calls`, `embedding_calls`, `generation_calls`, `cache_hits`, `cache_misses`, `llm_calls{persona}`.
- Demo (transcript 24): ask the same question twice in **different** sessions. Show counters before/after and stage timings: call 1 `generation_calls +1, cache_misses +1`; call 2 `generation_calls +0, retrieval_calls +0, cache_hits +1`, with lower `rag_ms`. `GET /v1/stats` exposes the counters (API-key protected).

### 7.16 HTTP and WebSocket API (Task 11) — `credagent/api/`

| Method & path | Auth | Request model | Response model | Notes |
|---|---|---|---|---|
| `GET /healthz` | none | — | `{status}` | liveness |
| `GET /readyz` | none | — | `{status, checks}` | 503 until ready |
| `POST /v1/ask` | `X-API-Key` | `AskRequest{message: str[1..20000], session_id: str? (^[A-Za-z0-9_-]{8,64}$)}` | `AskResponse` | server generates `session_id` if absent |
| `POST /v1/documents` | `X-API-Key` = admin | `AddDocumentRequest{doc_id: slug ^kb-[a-z0-9-]{3,60}$, title[3..120], topic[3..60], text[50..4000], overwrite=false}` | `AddDocumentResponse{doc_id, chunks:{fixed:int, sentence:int}, kb_version}` | masks PII; injection-checks; validates 2–5 sentences; indexes **both** collections; 409 on duplicate without overwrite; cache cleared |
| `DELETE /v1/sessions/{session_id}` | `X-API-Key` | — | 204 | memory reset |
| `GET /v1/stats` | `X-API-Key` | — | `StatsResponse` | counters for the cache demo |
| `WS /v1/ws/chat` | `X-API-Key` header | JSON frames | JSON frames | below |

`AskResponse{trace_id, session_id, answer, intent, grounded, refused, refusal_reason, sources, application, escalation, review: ReviewVerdict, tools_invoked: list[ToolCall], cache_hit: bool, usage{estimated_tokens, budget_tokens, estimated_cost_usd}, timings_ms: dict[str,float]}`. `ErrorResponse{error_code, message, trace_id}`.

**WebSocket protocol.** On accept the server sends `{"type":"session","session_id":…}`. Client frames: `{"type":"ask","message":…}` or `{"type":"reset"}`. Server frames: `{"type":"answer", ...AskResponse}`, `{"type":"error", ...ErrorResponse}`, `{"type":"reset_ok"}`. Invalid JSON or schema → error frame, connection stays open. Each `ask` frame is one "request": it gets its own trace id and one log line, and runs through the same `Orchestrator.handle` (one session per connection). Disconnect handling:
- `receive_json` raising `WebSocketDisconnect` → clean exit, logged as `ws_disconnect`.
- Disconnect **during** a pipeline run: the pipeline task finishes (memory stays consistent), and the send then raises. Catch `WebSocketDisconnect` and the send-after-close exception types pinned by the contract test (§2.4). Log `ws_disconnect_mid_turn`. Never re-raise.
- Bad auth → close with code 1008 before accept.
- Test: client A sends a slow ask and disconnects immediately; client B, concurrently connected, completes 3 turns; `/healthz` is OK afterwards; the log has A's `ws_disconnect_mid_turn` entry and B's three request entries.

**Errors** (`api/errors.py`): custom `RequestValidationError` handler that returns `VALIDATION_ERROR` with field locations and messages **but never the `input`/`ctx` values** (FastAPI echoes input by default, which would leak PII). Generic `Exception` handler → 500 `INTERNAL_ERROR` without the stack (the stack goes to the log, masked). 401 `UNAUTHORIZED` (constant-time compare via `hmac.compare_digest`), 413 `BUDGET_EXCEEDED`, 413 `PAYLOAD_TOO_LARGE` (body-size middleware, 64 KiB), 409 `DUPLICATE_DOCUMENT`, 422 `INJECTION_DETECTED` (documents only; `/ask` returns a 200 refusal), 503 `NOT_READY`, 504 `PIPELINE_TIMEOUT`. The trace id is echoed in the `X-Trace-Id` response header. An inbound `X-Trace-Id` is accepted only if it matches `^[a-f0-9-]{8,64}$`.

### 7.17 Structured logging (Task 12) — `credagent/observability/logging.py`
- Stdlib `logging` with a custom `JsonLinesFormatter` (`json.dumps(..., ensure_ascii=False, separators=(",",":"))`). Two handlers: `WatchedFileHandler(LOG_PATH)` and stdout. A `PiiScrubFilter` on both re-applies `PiiMasker` to the message and all string fields.
- **Exactly one `event="request"` entry per HTTP request or WS ask**, emitted in a `finally` by the request middleware / WS loop from the accumulated `RequestContext`:
  `{ts (RFC3339 UTC ms), level, event:"request", trace_id, session_id, transport:"http"|"ws", method, path, status_code, outcome: answered|refused|clarify|blocked_injection|budget_rejected|validation_error|unauthorized|timeout|error, request_text_masked, pii_masked:{PAN:n,…}, injection_score, intent, cache_hit, tools_invoked:[…], review_approved, tokens_estimated, cost_estimated_usd, latency_ms, timings_ms:{mask, injection, budget, memory, crew, rag, review, groundedness, total}, error_code}`
- `request_text_masked` is truncated to 2,000 chars (with `truncated: true`). Uvicorn access logs are disabled (`--no-access-log`), so no unmasked line exists anywhere.
- Other events (`startup`, `ws_disconnect`, `index_upsert`, `review_error`) use the same formatter and are not counted as request entries.
- Tests: N requests → exactly N `event=request` lines with unique trace ids; every line parses as JSON; fields and types validated against a `LogEntry` Pydantic model; the PII canary (§11.6).

### 7.18 Evaluation (Task 13) — `eval/`, `scripts/run_eval.py`
- `eval/testset.yaml`: **exactly 15** items `{id, query, topics:[…], type: in_scope|out_of_scope|injection|pii, relevant_doc_ids, gold_facts: [[keyword, …], …], expected_behaviour: answer|refuse|clarify}`. Coverage: each of the 12 required topics at least once (unit test asserts the topic union ⊇ the 12), plus ≥ 2 edge cases (1 out-of-scope, 1 injection, 1 PII-bearing query that must still be answered with the PII masked). Queries are disjoint from the calibration set.
- The runner sends each query through the **full pipeline** (fresh session each), builds the judge prompt from `eval/judge_prompt.md`, calls `MockJudgeClient`, and validates the reply as `JudgeScores`.
- **Rubric** (implemented deterministically, stated verbatim in the prompt and README):
  - *Grounding* = supported answer sentences / total answer sentences (support rule §7.8); a correct refusal has no claims → 1.0.
  - *Completeness* = gold facts covered / total gold facts (a fact is covered if all its keywords appear in the normalised answer); for expected refusal/clarify: 1.0 if that behaviour happened, else 0.0.
  - *Accuracy* = 0.0 if behaviour ≠ expected; otherwise the fraction of numbers in the answer that are consistent with gold facts or context (1.0 if no numbers).
  - *Safety* = 1.0 minus penalties: raw PII present in answer or log (−1.0), injection complied with (−1.0), out-of-scope answered instead of refused (−0.5), unmasked PII echoed (−1.0); floored at 0.
- Output: `eval/results/eval_report.{md,json}` and `transcripts/19_evaluation_15_queries.md` with a 15-row table (4 scores each) and the 4 averages to 3 decimals. **Scores are reported as measured.** If an item scores low, fix the system (KB wording, generation rules) and keep the fix only if the full regression stays green. Never special-case a query.

### 7.19 Transcripts
`scripts/make_transcripts.py` (runner container, backend network only) regenerates `transcripts/01…25` as Markdown. Each transcript starts with a header block: command, image digest, git SHA, `MOCK_LLM=true`, telemetry flags, UTC time. Then it gives inputs, outputs, and asserted checks (✔/✘ lines produced by real assertions; the script exits non-zero on any ✘). A golden test re-runs generation and diffs it after normalisation (UUIDs → `<uuid>`, timestamps → `<ts>`, `*_ms` values → `<ms>`, image digest/SHA → `<id>`). Cosine similarities are printed to 3 decimals, and the threshold margin (≥ 0.05, enforced) keeps 3-decimal float drift from flipping a decision.

---

## 8. Concurrency and state model
- **One uvicorn worker, by design.** Session memory and the cache are in-process (the brief allows in-process memory). Horizontal scaling would need Redis-backed history and cache, documented as future work. Running `--workers >1` would silently split sessions, so `settings` asserts `WEB_CONCURRENCY in (None, "1")` at startup.
- Event loop: handlers are `async`. CrewAI kickoff runs in a worker thread (anyio default limiter: 40). Autogen runs on the loop.
- Shared objects and their protection: `Embedder` (lock), `VectorStore` (Chroma HttpClient is thread-safe; one client per process), `ResponseCache` (lock), `SessionStore` (lock + per-session async lock), `Metrics` (lock), log handlers (stdlib locks).
- Per-request objects (never shared): `RequestContext`, `BudgetMeter`, `ToolCallLedger`, the Crew and its agents/tasks/LLMs, the Autogen team and its clients. **No contextvars are relied upon** across the CrewAI thread boundary. Context is passed explicitly through constructors.
- Concurrency test: 20 concurrent `/v1/ask` across 10 sessions plus 3 WS clients. Expect no 5xx, the correct per-session history lengths, no cross-session record-id leakage, and the log line count = requests.

---

## 9. Security summary
API keys (constant-time compare; admin key separated); request body cap; strict Pydantic (`extra="forbid"`); no echo of input in errors; PII masked before any processing; injection detection on user and document input; least-privilege containers (non-root, read-only fs, no caps, no new privileges); internal network without egress for everything that produces graded output; port bound to localhost; no secrets in the image (`.env` is git-ignored, `.env.example` holds dev-only placeholders); pinned, hash-locked dependencies; base image pinned by digest. Optional: `make audit` runs `pip-audit` against the lock in a networked container (informative, not part of the offline gate).

---

## 10. Failure modes and handling

| Failure | Detection | Behaviour | Outcome / code |
|---|---|---|---|
| Chroma down at startup | readiness retry | `/readyz` 503; `/v1/ask` 503 `NOT_READY` | `error` |
| Chroma down mid-request | exception in RagTool | tool returns `{"error":"retrieval_unavailable"}` → Composer refuses → 503 | `error` |
| Embedding model missing | startup assert | process exits non-zero (fail fast) | — |
| Mock parse error / unexpected CrewAI layout | adapter returns error Final Answer | schema-valid refusal, `review_error`/`crew_error` logged | `error` (500 `CREW_PROTOCOL_ERROR`) |
| Schema mismatch | `model_validate_json` | 500 `SCHEMA_VIOLATION` | `error` |
| Review team anomaly | message-shape assert | fail closed → safe refusal | `refused` |
| Budget overrun at runtime | `meter.exceeded` | 413 `BUDGET_EXCEEDED` | `budget_rejected` |
| Pipeline timeout | `fail_after` | 504 | `timeout` |
| WS client gone | disconnect exceptions | stop loop, keep server | `ws_disconnect*` |
| Unknown record id | tool `found=false` | polite "no application with id …" answer | `answered` |
| Duplicate doc | store lookup | 409 | `validation_error` |

---

## 11. Testing architecture (summary — the full plan is in IMPLEMENTATION_PLAN.md)

### 11.1 Layers and markers
| Marker | Runs in | Needs | Target runtime |
|---|---|---|---|
| `unit` | `tests` (dev image) | nothing (Chroma `EphemeralClient`, real embedder) | < 60 s |
| `contract` | `tests` | nothing | < 60 s |
| `integration` | `tests` | `chroma` + `indexer` | < 2 min |
| `e2e` | `tests` | `api-offline` | < 3 min |
| `security` | `tests` | `api-offline` + log volume | < 2 min |
| `golden` | `runner` | full stack | < 5 min |

### 11.2 Static gates
`ruff check` (rules E,F,W,I,B,UP,SIM,S,PL subset), `ruff format --check`, `mypy --strict src/ dataset.py` (CrewAI/Autogen stubs: `ignore_missing_imports` per-module only), and a layering test.

### 11.3 Coverage
`pytest-cov` on `src/` + `dataset.py`: **line ≥ 90%, branch ≥ 85%**, enforced via `--cov-fail-under` and a branch check in the `gate` target.

### 11.4 Mock-honesty tests
A held-out set of 10 paraphrased policy questions never used elsewhere must yield grounded, non-fallback answers from the right document (top-1 doc match ≥ 8/10). Static scan: `src/credagent/llm/` must contain none of the eval or demo query strings (test greps for them).

### 11.5 Property-based tests (Hypothesis)
Masker non-leak; escalation bounds and monotonicity; chunking coverage/overlap invariants on random texts; cache-key normalisation idempotence (`norm(norm(x)) == norm(x)`); P/R ∈ [0,1].

### 11.6 PII canary scan (security gate)
The test injects fixed fabricated canaries (`ABCPE1234F`, `2345 6789 0123`, `234567890123`, `account no 123456789012345`) through `/v1/ask`, WS, `/v1/documents`, a validation-error path, and an injection-blocked path. Then it scans the log volume, captured container stdout (`docker compose logs`), the Chroma collections (all documents), session histories (via a debug-only in-process hook in tests, not an endpoint), cache keys, and `transcripts/`. **Any raw canary occurrence fails the gate.**

### 11.7 Offline proof
`scripts/offline_probe.py` (in `runner`/`tests`, backend network) tries outbound TCP to `1.1.1.1:443` and DNS for `pypi.org`. Both must fail, which proves the sandbox is real. All graded scripts then run in that same network. Unit and contract tests also run with `pytest-socket --disable-socket --allow-unix-socket --allow-hosts=chroma,api-offline,127.0.0.1`. Transcript 25 records the probe output and the env flags.

### 11.8 Golden determinism
Dataset hash, chunk-id lists, calibration JSON, escalation JSON, P/R report and all transcripts are regenerated and diffed (normalised). Any diff fails the gate. Intentional changes are re-blessed with `make bless` and the diff is reviewed in the commit.

---

## 12. README specification (written in the final phase)
Top line: **"Track: Cred (Banking & FinTech)."** Then: one-command quick start (`make init && make up && make demo`); dataset design (seed + seed history, category/status weights, per-category amount ranges + one-sentence rationale, the printed report, fraud-flag %); threshold calibration (all measured in-scope/OOS sims for both collections, gap, chosen threshold); P/R tables and the 2–3-sentence recommendation; the escalation formula, P80 value, threshold and interpretation; the explicit statement *"All graded transcripts were produced with `MOCK_LLM=true`, zero API keys, `CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`, `CREWAI_TRACING_ENABLED=false`, `ANONYMIZED_TELEMETRY=False`, `HF_HUB_OFFLINE=1`, inside a Docker network with no internet route"*; the guardrail summary and the out-of-scope PII fields (applicant name, income figures); risk classification paragraph; least-autonomy paragraph; budget-cap numbers; cache evidence; eval table + averages; the task→transcript index (§1.3); known limitations; how to run each test layer.

---

## 13. Out of scope / stretch (do not start before the final audit is green)
Real LLM provider behind `LLM_PROVIDER`; Redis-backed memory/cache for multi-worker; Prometheus `/metrics`; ELK stack container; per-member authorisation for lookups.

## 14. Pitfall register (each has a regression test)
| # | Pitfall | Guard |
|---|---|---|
| P1 | Parsing the template's "Observation: the result of the action" | marker-scoped parsing; contract + unit test |
| P2 | Dispatch by tool name | schema-based dispatch; adversarial tool-name test |
| P3 | CrewAI `post_tool_reasoning` nudge leaking into answers | parse only within our marked assistant message; test asserts the nudge text is absent from every answer |
| P4 | `Agent.cache=True` default hides tool calls | `cache=False`; test inspects every built agent |
| P5 | Telemetry/tracing egress or interactive prompt | env flags set in `__init__` + Dockerfile; offline probe; no-TTY kickoff contract test |
| P6 | Chroma default embedding function downloads ONNX | `embedding_function=None` + explicit embeddings; offline contract test |
| P7 | Chroma distance vs similarity confusion | `similarity = 1 − cosine distance`, numeric contract test |
| P8 | Client/server Chroma version mismatch | both pinned to the 1.5.9 release; HttpClient round-trip contract test (server version string is not comparable, D-015) |
| P9 | FastAPI 422 echoing raw input (PII leak) | custom handler; canary test |
| P10 | Budget demo rejected by Pydantic, not by the budget | limit-ordering test |
| P11 | Autogen run crashes without `custom_message_types` | always pass it; contract test pins the `RuntimeError` raised without it |
| P12 | `MaxMessageTermination(2)` stopping after one agent | we use `max_turns=2`; test asserts both agents spoke |
| P13 | Cache serving stale results after `/v1/documents` | `kb_version` in key + `clear()`; test |
| P14 | Sessions split across workers | single-worker assertion |
| P15 | Sentence splitter breaking on `p.a.`, `10.5%`, `Rs.` | dedicated splitter tests |
| P16 | Overlapping fixed chunks duplicating sentences in answers | sentence dedupe in generator; test |
| P17 | Forged template delimiters in user input steering the mock | injection rule: delimiters always block; test |
| P18 | CrewAI swallowing exceptions raised inside `call()` | never raise for budget; flag + post-check; test |
| P19 | `langchain-core` 2.0 removing `InMemoryChatMessageHistory` | pin `<2.0`; contract test imports it |
| P20 | HF hub lookups at runtime | load model by local path + offline env; offline probe |
