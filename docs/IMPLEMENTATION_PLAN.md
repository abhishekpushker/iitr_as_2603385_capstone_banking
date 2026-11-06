# Cred Loan-Support Agent — Implementation & Verification Plan

> Read `docs/ARCHITECTURE.md` first; section references (§) point there. This plan says **in what order** to build, **how each step is proven**, and **what blocks progress**. Nothing here relaxes the architecture.

---

## 1. Operating model

### 1.1 The loop for every unit of work
```
1. Write/extend the failing test(s) that express the requirement   (red)
2. Implement the smallest change that makes them pass              (green)
3. Refactor; keep tests green
4. make gate-fast            (lint + types + unit + contract, < 3 min)
5. At the end of each phase: make gate   (EVERYTHING, §3) — must be fully green
6. Commit (Conventional Commits), update docs/DECISIONS.md if a decision was made
```
No phase starts until the previous phase's **exit gate** is green. "Green" means every check passed, with none skipped, xfailed or quarantined. `pytest -p no:randomly` is not allowed either. Tests run in random order (`pytest-randomly`, seed printed) to catch hidden coupling.

### 1.2 Bug protocol (zero-bug discipline)
1. **Reproduce first:** write a test that fails because of the bug (mark it with the bug id in its docstring, e.g. `BUG-007`).
2. Log it in `docs/BUG_LOG.md`: id, phase found, symptom, root cause, fix commit, regression test name.
3. Fix the root cause, not the symptom. **Never** weaken an assertion, widen a tolerance, add a `sleep`, special-case a query string, or hand-edit a generated artifact to make a test pass.
4. Run `make gate` (full). The new test stays in the suite permanently.
5. If the root cause is a library behaviour, add or extend a **contract test** in `tests/contract/` so the same surprise can never recur silently.
6. Flaky test = bug. Find the source of nondeterminism. Never retry-until-pass.

### 1.3 Stop rules (escalate to the human instead of improvising)
- A contract test reveals a library behaviour that contradicts the architecture (e.g. CrewAI message layout changed). Stop, describe the diff, and propose an architecture amendment.
- The calibration gap is < 0.05 after two honest KB/query revisions.
- An acceptance criterion cannot be met under `MOCK_LLM` as designed.
- Any need for external network at runtime.

### 1.4 Commit and branch conventions
Work on `main` with small commits, or use one branch per phase (`phase-03-retrieval`) merged after the gate. Each phase ends with a tag: `phase-NN-green`. Commit messages follow `feat(rag): …`, `test(guardrails): …`, `fix(crew): … (BUG-004)`. Generated artifacts (transcripts, config JSONs, eval results) are committed in **separate** commits titled `chore(artifacts): regenerate …` so their diffs are reviewable.

---

## 2. Make targets (the only interface)

| Target | What it does |
|---|---|
| `make init` | copy `.env.example` → `.env` if missing; create dirs |
| `make lock` | `uv pip compile --generate-hashes` both lockfiles inside a builder container (needs network; the only networked dev step besides `build`) |
| `make build` | `docker compose build` (runtime + dev targets) |
| `make up` / `make down` / `make logs` | default stack (chroma, indexer, api) |
| `make index` | run the indexer one-shot |
| `make lint` | ruff check + ruff format --check (dev container) |
| `make typecheck` | mypy --strict |
| `make test-unit` / `test-contract` / `test-integration` / `test-e2e` / `test-security` | the pytest layers, inside `tests` on the backend network |
| `make golden` | regenerate deterministic artifacts into a temp dir and diff against committed ones |
| `make transcripts` / `make eval` / `make calibrate` / `make eval-chunking` | generators (runner container) |
| `make bless` | regenerate and overwrite committed artifacts (human reviews the diff) |
| `make pii-scan` | canary scan (§ARCH 11.6) |
| `make offline-proof` | egress probe + all graded scripts in the internal network |
| `make gate-fast` | lint + typecheck + unit + contract |
| `make gate` | build → gate-fast → integration → e2e → security → golden → pii-scan → offline-proof → coverage thresholds |
| `make demo` | up + transcripts (human-friendly) |
| `make clean` | down -v, prune project images |

`make gate` must be runnable from a clean clone with only Docker installed, and it must exit non-zero on the first failure.

---

## 3. The regression gate (run at the end of **every** phase and before every tag)

```
[1]  docker compose build                           (cached; fails on lock/hash mismatch)
[2]  ruff check · ruff format --check · mypy --strict · layering test
[3]  pytest -m "unit or contract"  --randomly       coverage collected
[4]  pytest -m integration                          (chroma + indexer up)
[5]  pytest -m e2e                                  (api-offline up)
[6]  pytest -m security                             (canary, injection corpus, auth)
[7]  golden diff (dataset hash, chunk ids, configs, P/R, transcripts present so far)
[8]  offline probe (must FAIL to reach the internet) + graded scripts re-run inside backend net
[9]  coverage: line ≥ 90 %, branch ≥ 85 % (from phase 3 onward; before that ≥ 80 %)
[10] docker compose down -v ; second run of [4]–[5] from a cold volume (idempotency)
```
Step 10 catches "works only on a warm volume" bugs (stale Chroma state, leftover logs). Gate output is saved to `artifacts/gate-<phase>.log` (git-ignored) and summarised in the phase's tag message.

---

## 4. Phases

Each phase lists **Goal → Build → Tests written first → Exit gate**. Estimates assume one focused implementer; the brief's 14 days are budgeted so testing is never squeezed.

### Phase 0 — Foundation, toolchain, dependency contracts (Day 1–2)
**Goal:** a reproducible Docker toolchain and machine-checked proof of every library assumption in §ARCH 2.

Build:
- `pyproject.toml` (package `credagent` under `src/`, `py-modules=["dataset"]`, ruff/mypy/pytest/coverage config, pytest markers), `requirements.in` (crewai==1.15.36, chromadb==1.5.9, autogen-agentchat==0.7.5, autogen-core==0.7.5, langchain-core>=1.7.6,<2.0, sentence-transformers==6.2.0, torch==2.15.0 (cpu), fastapi, uvicorn[standard], pydantic-settings, pyyaml, anyio, numpy), `requirements-dev.in`, both locks with hashes.
- `docker/Dockerfile` (stages base/deps/model/runtime/dev, §ARCH 3.2), `compose.yaml` (all services, networks, profiles, hardening), `.env.example`, `.dockerignore`, `.gitignore`, `Makefile`.
- Pin the base image digest and the MiniLM revision SHA; record both in `DECISIONS.md`.
- `src/credagent/__init__.py` (telemetry env defaults), `settings.py` skeleton, `scripts/offline_probe.py`.
- Empty docs: `GOVERNANCE.md`, `DECISIONS.md`, `BUG_LOG.md`.

Tests written first (`tests/contract/`):
- `test_crewai_contract.py`: BaseLLM abstract set == {"call"}; `call` signature has `from_agent`, `response_model`; a recording mock + 1 tool + `output_pydantic` crew proves (a) the system message contains the literal "Observation: the result of the action", (b) after the tool, the observation is inside **our** assistant message, (c) the next user message equals `post_tool_reasoning`, (d) `result.pydantic` is filled with **exactly 2** LLM calls (no converter re-ask), (e) `Agent().cache` default is True (to justify our override). Plus kickoff with stdin closed, under `pytest-timeout=20`.
- `test_autogen_contract.py`: abstract-member set; RR(max_turns=2) message shape and types; `stop_reason`; missing `custom_message_types` → `ValueError`.
- `test_chroma_contract.py`: cosine-distance semantics vs numpy; `embedding_function=None` + explicit embeddings work with sockets disabled; server/client version equality (integration-marked).
- `test_langchain_contract.py`: `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` per-session isolation; deprecation warning is emitted (assert, don't silence).
- `test_starlette_ws_contract.py`: exception types on receive/send after disconnect.
- `test_offline.py`: probe fails to reach `1.1.1.1:443` and to resolve `pypi.org` from the `tests` container.
- `test_image.py`: runs as uid 10001, root fs read-only, model path exists, `HF_HUB_OFFLINE=1`, telemetry flags set, `import crewai` triggers no socket (pytest-socket).

**Exit gate:** `make build` from clean; `make gate-fast` green; contract + offline tests green; image size recorded (target < 2.5 GB). Tag `phase-00-green`.

### Phase 1 — Dataset (Task 1) (Day 2)
Build: `dataset.py` exactly per §ARCH 7.1, `scripts/validate_dataset.py` (prints the report as Markdown; non-zero exit if invalid).
Tests first (`tests/unit/test_dataset.py`): count ≥ 40 (=60); every category ≥ 3; every status ≥ 1; fraud share in [0.10, 0.30]; ids unique and pattern-valid; field set and order; amount within category range and ≤ 8 digits; `days` ∈ [0,30] ints; `generate(SEED)` called twice → identical; a different seed → different; golden sha256 of the dataset; no import of `credagent`; the `LoanCategory`/`LoanStatus` literals equal dataset vocabularies.
Seed procedure per §ARCH 7.1; record the seed history in `DECISIONS.md` (later copied to the README).
**Exit gate:** full `make gate`; transcript `01_dataset_report.md` generated by `make transcripts` (the transcript generator framework is built here with its header block and ✔/✘ assertions). Tag.

### Phase 2 — Knowledge base, chunking, embeddings, indexing (Tasks 2–3) (Day 3–4)
Build: the 13 KB docs (§ARCH 7.2 content contract), `kb/loader.py`, `text.py` (normalise, tokens, numbers, sentence splitter), `kb/chunking.py`, `rag/embedder.py`, `rag/store.py`, `scripts/indexer.py`, the `indexer` service wiring.
Tests first:
- `test_text.py`: splitter golden cases (`10.5% p.a.`, `₹1.5 lakh`, `e.g.`, `Rs. 500`, ellipsis, trailing no-period); number canonicalisation table; tokenizer keeps `10.5%`, `₹10,000`.
- `test_kb.py`: 13 docs; the 12 required topics present (topic→doc map); each body 2–5 sentences; front matter schema; unique ids; disclaimer present; the facts from the content contract present (keyword checks per doc).
- `test_chunking.py`: deterministic ids; fixed-size coverage/overlap/word-boundary invariants (+ Hypothesis); one sentence per sentence-chunk; char spans reconstruct the source.
- `test_embedder.py`: dim 384; normalised (‖v‖≈1); deterministic across two calls (exact equality on the same process, `allclose 1e-6` across processes).
- `tests/integration/test_index.py`: both collections exist with the expected counts; re-run of the indexer → same ids/counts (idempotent); orphan cleanup after a doc is removed in a temp KB; `kb_meta.kb_version` stable (D-019); a sample query returns a sensible top doc in **both** collections.
**Exit gate:** full gate incl. cold-volume rerun; transcripts 02 and 03. Tag.

### Phase 3 — Retrieval, calibration, grounded generation, cache core (Tasks 4, 16-core) (Day 4–5)
Build: `rag/retriever.py`, `llm/engine.py` (generate_grounded, support_report), `rag/generator.py` (LLMClient protocol + mock), `rag/calibration.py`, `scripts/calibrate_threshold.py`, `config/retrieval.json`, `cache/response_cache.py`, `observability/metrics.py`.
Tests first:
- `test_support.py`: support rule incl. numeric rule ("0% interest" unsupported; verbatim context sentence supported; paraphrase with the same numbers supported if recall ≥ 0.6).
- `test_generator.py`: below threshold → fallback; above → every answer sentence verbatim from context; dedupe of overlapping chunks; deterministic; uses only the given chunks (no global state).
- `tests/integration/test_calibration.py`: re-measure → identical JSON; `max_oos < threshold < min_in`; `gap ≥ 0.05`; calibration queries disjoint from demo/eval/P-R queries.
- `test_cache.py`: hit/miss; normalisation idempotence (Hypothesis); key includes `kb_version`/strategy/threshold/top_k; LRU eviction; TTL expiry (injected clock, no sleeps); errors not cached; thread-safety under 8 threads; keys contain no raw canary PII.
- Mock-honesty held-out set (§ARCH 11.4) — first version.
Demo set: 5 in-scope + 1 OOS → transcripts 04, 05.
**Exit gate:** full gate. Tag.

### Phase 4 — Chunking evaluation (Task 5) (Day 5)
Build: `rag/evaluation.py`, `scripts/eval_chunking.py`; it writes the P/R report and sets `deployed_strategy` in `config/retrieval.json`.
Tests first: `test_pr_math.py` (hand-computed cases incl. two-relevant-doc queries, duplicate-chunk dedupe, empty retrieval); report shows literal arithmetic; recommendation sentences reference the measured numbers (the test parses them back out of the text).
**Exit gate:** full gate; transcript 06. Tag.

### Phase 5 — Lookup tool, escalation, ACL core (Tasks 6, 15-ACL) (Day 6)
Build: `tools/escalation.py`, `config/escalation.json` (generated by a script from the dataset), `tools/lookup_tool.py` (pure fn + `LoanStatusTool` owner-bound), `governance/policy.py`, `tools/acl.py`, `tools/rag_tool.py` (wraps cache → retriever → generator, records the ledger).
Tests first: §ARCH 7.7 test list; tool input validation (lower-case id, whitespace, `LA-07`, SQL-ish/injection strings → `invalid_record_id`); ACL: allowed/denied matrix, owner tamper → raises, tool JSON outputs validate against output models.
**Exit gate:** full gate; transcript 07. Tag.

### Phase 6 — Mock CrewAI LLM, crew, structured output (Tasks 7, 9) (Day 7–8) — *highest-risk phase*
Build: `llm/crewai_llm.py` (§ARCH 7.8.1 exactly), `crew/{templates,agents,tasks,factory}.py`, `schemas.py` (CrewResponse with validators), `MockEngine.compose`, `RequestContext`, `BudgetMeter` (charging only; policy in Phase 9).
Tests first:
- Unit (no crew): the adapter fed **recorded real CrewAI message lists** (captured in Phase 0 into `tests/fixtures/crewai_messages/*.json`) → correct Action / Final Answer; the template "Observation:" is ignored; the `post_tool_reasoning` nudge never appears in the output (P1, P3); schema dispatch with adversarial tool names (P2); a step-limit guard; budget exceed → schema-valid refusal without raising (P18); never emits "Observation:".
- `test_schemas.py`: validators (refused ⇒ reason; status ⇒ application; extra fields forbidden); JSON round-trip.
- `tests/integration/test_crew.py` (real kickoff, Chroma up): policy query → ledger shows `rag_lookup` by Retrieval only, no lookup call; status query "LA-00xx" → ledger shows `check_loan_application_status` by Lookup; mixed query → both; unknown id → polite not-found; every `result.raw` validates and equals `result.pydantic`; all agents have `cache=False`, `allow_delegation=False` (P4); exact LLM-call counts per query type (detects hidden retries); `audit_crew` passes.
**Exit gate:** full gate; transcripts 08, 09, 12. Tag.

### Phase 7 — Pipeline, guardrails, memory (Tasks 8, 10) (Day 8–9)
Build: `guardrails/{pii,injection,groundedness}.py`, `memory/session.py`, `pipeline/orchestrator.py` (stages 0–6, 8–12; review stubbed as pass-through until Phase 8, behind an interface), `MockEngine.resolve_followup`.
Tests first: §ARCH 7.12 test lists (tables, precedence, Hypothesis non-leak, 25/25 attack/benign corpora, delimiter forgery P17); groundedness: below-threshold refusal, unsupported-sentence refusal, lookup-sentence numeric check; memory: 3-turn carry-over, fresh-session absence (history length 0 + clarify intent), reset, per-session isolation under concurrency, blocked turns not stored, history contains masked text only; orchestrator: stage order (spy), raw text unreachable after stage 1 (the context object holds no raw field), timeouts → 504 path.
**Exit gate:** full gate; transcripts 10, 11, 13, 14, 15. Tag.

### Phase 8 — Autogen review stage (Task 14) (Day 10)
Build: `llm/autogen_client.py`, `review/{models,team}.py`, `MockEngine.review/edit`; wire into the orchestrator stage 7; the test-only `inject_claim` parameter on `review.run_review(...)`.
Tests first: client implements all 9 members; reviewer/editor parse only their own expected sources; approve path returns a byte-identical draft; revise path removes exactly the injected sentence and keeps the rest in order; all-unsupported → fallback text; malformed team output → fail closed; 3-message shape asserted; `ReviewVerdict` fields exactly `{approved, final_answer, reason}`; no API model exposes `inject_claim`; budget charged by Autogen calls.
**Exit gate:** full gate; transcripts 20, 21; re-run of transcripts 08–15 (now passing through the review) with golden diffs reviewed and blessed. Tag.

### Phase 9 — Governance: budget cap, least-autonomy demo, risk doc (Task 15) (Day 10–11)
Build: `governance/budget.py` (preflight + meter policy), measure `PROMPT_OVERHEAD_TOKENS` with a script over the eval + demo queries, set the budget values, `docs/GOVERNANCE.md` (risk = High + justification paragraph, least-autonomy paragraph, four layers, IDOR limitation).
Tests first: preflight math; limit ordering body > Pydantic > budget (P10); oversized request rejected with **zero** LLM calls; every eval/demo query ≤ 50% budget; runtime overrun via a forced tiny budget → 413 after kickoff without a crash; least-autonomy demo assertions (a)–(d) from §ARCH 7.14.1.
**Exit gate:** full gate; transcripts 22, 23. Tag.

### Phase 10 — FastAPI HTTP + WebSocket + structured logging (Tasks 11, 12, 16-demo) (Day 11–12)
Build: `api/*`, `observability/logging.py`, middleware (trace id, body cap, timing, single log emission), `/v1/stats`, WS loop, `api-offline` service.
Tests first (`tests/e2e/`, via httpx/websockets against `api-offline`):
- HTTP: every endpoint's happy path and every error code in §ARCH 7.16; OpenAPI schema contains the Pydantic models; 422 bodies never contain input values; auth (missing/wrong/admin-only); `X-Trace-Id` echo and validation; `/v1/documents` → new doc retrievable in both collections, duplicates → 409, injection → 422, PII masked in the stored text, cache invalidated (P13).
- WS: session frame on connect; multi-turn memory over one connection; bad JSON → error frame and the connection survives; bad auth → 1008; **disconnect mid-turn** scenario from §ARCH 7.16 with a concurrent second client; 50 connect/disconnect cycles → no leaked tasks (task count stable) and the server healthy.
- Logging: exactly one `event=request` line per request (HTTP and WS), unique trace ids, schema-valid, timings present, outcomes correct for every error path.
- Cache demo over HTTP: counters before/after via `/v1/stats`.
- Concurrency test from §ARCH 8; single-worker assertion (P14).
**Exit gate:** full gate incl. `make pii-scan` across all endpoints; transcripts 16, 17, 18, 24. Tag.

### Phase 11 — Evaluation at scale (Task 13) (Day 12–13)
Build: `eval/testset.yaml` (15 items, coverage per §ARCH 7.18), `eval/judge_prompt.md`, `llm/judge.py`, `scripts/run_eval.py`.
Tests first: `test_judge.py` (each rubric dimension on hand-built cases: perfect, ungrounded number, missing fact, wrong refusal, PII leak, injection compliance); test-set coverage (12 topics + ≥ 2 edge, exactly 15, disjoint from calibration); report has 15 rows × 4 scores + 4 averages; averages recomputed independently in the test equal the report; judge output validated by `JudgeScores`.
Quality bar (informative, honest): investigate any score < 0.67 and fix only by improving generic behaviour or KB wording, with a full gate after each fix. Record the before/after in `DECISIONS.md`.
**Exit gate:** full gate; transcript 19. Tag.

### Phase 12 — Hardening, documentation, final audit, release (Day 13–14)
Build: README per §ARCH 12; transcript 25 (offline proof); final `make bless` + review of every artifact diff; remove dead code (`ruff` + `vulture` report, informative).
Verification activities:
1. **Clean-room run:** fresh `git clone` into a new directory → `make init && make gate` → green. No cached images (`docker builder prune` first).
2. **Soak:** 15 minutes of mixed load (HTTP + WS, disconnects every 10th connection) via a `tests/soak/` script. Expect no 5xx except the intended ones, RSS growth < 10%, log lines = requests.
3. **Fault injection:** stop `chroma` mid-soak → 503s, no crash; restart → recovery without restarting `api`.
4. **Acceptance audit:** walk the checklist in §5 and tick each item against concrete evidence (file + line or transcript section). A second, independent reviewer pass (a fresh Claude Code session given only the brief and the repo) re-derives the checklist and reports gaps.
5. **Security review:** canary scan, injection corpus, container hardening checks (`docker inspect` assertions in `test_image.py`).
**Exit gate:** everything green twice in a row (two consecutive full gates, the second from cold). Tag `v1.0.0`.

---

## 5. Final acceptance checklist (evidence must be cited for each)

**Part 1**
- [ ] `dataset.py` ≥ 40 records; category ≥ 3 each; status ≥ 1 each; fraud % in band; design choices in README → `01_dataset_report.md`, `test_dataset.py`
- [ ] ≥ 12 KB docs covering all 12 topics, 2–5 sentences → `02_kb_inventory.md`, `test_kb.py`
- [ ] Two strategies, two Chroma collections, both sensible on a sample query → `03_chunking_and_index.md`
- [ ] Threshold calibrated (≥ 3 in / ≥ 2 OOS measured, values in README); ≥ 5 in-scope + 1 OOS fallback → `04`, `05`
- [ ] P/R for both collections, per-query arithmetic, numbers-cited recommendation → `06`

**Part 2**
- [ ] Lookup + designed, justified escalation score (not a boolean OR) → `07`, `test_escalation.py`
- [ ] Crew ≥ 3 agents via `.kickoff()`; RAG and lookup tools each invoked on different queries → `08`, `09`
- [ ] Multi-turn memory + separate fresh-session transcript → `10`, `11`
- [ ] Every crew response validated against the Pydantic schema → `12`, orchestrator check
- [ ] PII masking (PAN, Aadhaar, bank account), injection detection, groundedness refusal, each firing → `13`, `14`, `15`

**Part 3**
- [ ] ≥ 2 HTTP endpoints + 1 WS with Pydantic models; WS survives a mid-conversation disconnect → `16`, `17`
- [ ] One JSONL entry per request with trace id + timings; no raw PII on disk → `18`, `make pii-scan`
- [ ] 15 queries × 4 scores + 4 averages under MOCK_LLM → `19`

**Part 4**
- [ ] Autogen RR team (max_turns=2, `output_content_type`, `custom_message_types`): approve-unchanged and revise cases with Pydantic verdicts → `20`, `21`
- [ ] Least autonomy demonstrated + paragraph; risk = High + paragraph; budget cap rejects an oversized request → `22`, `23`, `GOVERNANCE.md`
- [ ] Cache hit with before/after evidence → `24`

**Submission**
- [ ] README: track line, seed/weights/ranges, threshold values, telemetry flags confirmed, MOCK_LLM-only statement, zero API keys
- [ ] No images, PDFs, slides, or media anywhere in the repo (a test asserts no files with those extensions)
- [ ] Public GitHub repo, one link

---

## 6. Test inventory targets (minimums)
| Layer | Approx. test count | Notes |
|---|---|---|
| unit | 250+ | incl. ~15 Hypothesis properties |
| contract | 25+ | one per §ARCH 2 fact |
| integration | 40+ | real Chroma, real kickoff |
| e2e | 50+ | HTTP + WS against api-offline |
| security | 20+ | canary, corpora, auth, hardening |
| golden | 1 per artifact | ~32 artifacts |

## 7. Risk register (plan-level)
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CrewAI internal prompt/flow differs in edge paths (retries, converter) | Medium | High | Phase 0 contract tests on recorded messages; exact LLM-call-count assertions; adapter fails closed |
| Threshold gap too small with short policy sentences | Medium | Medium | more OOS/in-scope samples; KB wording; sentence strategy; stop rule at 2 revisions |
| Torch image bloat / slow builds | High | Low | CPU wheels, layer ordering (deps before src), BuildKit cache mounts |
| Chroma server image lacks healthcheck tooling | Medium | Low | indexer retry loop; documented in DECISIONS |
| Float drift flipping golden outputs across machines | Low | Medium | 1 thread, 3-decimal printing, ≥ 0.05 margin, normalised diffs |
| PII leak via an overlooked path | Medium | High | mask-first design, logging filter, canary scan over every sink |
| Schedule overrun in Phase 6 | Medium | High | Phase 6 has 2 days + the Phase 0 spike already de-risked the mock protocol |
