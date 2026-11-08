# Independent audit against docs/BRIEF.md

Audited: commit `b5ad3cd` (main = origin/main), 2026-11-08. Source of truth: `docs/BRIEF.md` only;
the architecture, plan, decisions, acceptance docs and CLAUDE.md were not read before the
verdicts below were fixed. Evidence is cited from the code, `transcripts/`, `README.md`,
`eval/results/` and `config/`. Read-only audit: no file other than this report was changed.

Verdicts: **PASS** = met, with evidence a grader can check · **WEAK** = arguably met, but a strict
grader could deduct · **FAIL** = the brief's literal requirement is not met.

## Gate run

`make gate` was run once (2026-11-08, 07:43–07:55 UTC). **Result: FAILED (exit 1) at the last step.**

| Step | Result |
|---|---|
| [1] build, [2] lint + mypy | green ("All checks passed!") |
| [3] unit + contract, coverage | 1456 passed (coverage gate line ≥ 90 %, branch ≥ 85 % met) |
| [4] integration (warm) | 59 passed |
| [5] e2e | 22 passed |
| [6] security + pii-scan | 11 passed |
| [7] golden: regenerates every artifact and diffs against the committed ones, including README numbers | 48 passed |
| [8] offline proof: egress probe, then every graded script and all 25 transcripts on the internal network | offline = true, 25/25 transcripts "ok" |
| [10] cold volume (`down -v`, rerun integration + e2e) | **1 failed, 58 passed**: `tests/integration/test_chunking_eval.py::test_script_deploys_nothing_when_no_strategy_meets_the_gap` |

**What this confirms.** Steps [7] and [8] regenerate the calibration, P/R, escalation, budget and eval artifacts and the transcripts, then compare them with the committed files. Both passed, so the numbers quoted in README and the transcripts (thresholds 0.567/0.49, gaps 0.1746/0.0329, P/R 0.900, P80 = 25 and T = 0.5, budget 18,000, eval 1.000 ×60, cache deltas) are real and reproducible.

**What failed.** In the cold-volume rerun (pytest-randomly seed `2424900888`), that test ran first in the session, before anything had indexed the fresh Chroma volume. `calibrate_threshold.main` measured top-1 = −1.000 / `None` for all 36 queries on both collections and returned 1, so `assert ... == 0` failed at `tests/integration/test_chunking_eval.py:101`. The same test passed in the warm step [4]. This is a **test-isolation / ordering defect**: the test depends on collections populated by another test or an earlier step. The product is not broken, but the repository's own full gate is red, and a grader who runs `make gate` may hit it depending on the random seed.

**Fix needed.** Give that test (or the integration session) a fixture that indexes the KB into the cold volume before calibration runs, reproduce with `--randomly-seed=2424900888` after `down -v`, and log it as a bug.

## Tasks

| # | Item | Verdict | Evidence | Fix needed |
|---|---|---|---|---|
| T1 | `dataset.py`, seeded, deterministic, `LOAN_APPLICATIONS` ≥ 40 | PASS | `dataset.py` (`SEED=2026`, `N_RECORDS=60`, local `random.Random`, `LOAN_APPLICATIONS = generate(SEED)`); transcript 01 checks "generate(SEED) twice is identical" | — |
| T1 | Use every **given** category value (`Personal Loan`, `Home Loan`, `Auto Loan`, `Education Loan`, `Business Loan`) | **FAIL** | `dataset.py` `CATEGORY_WEIGHTS` uses `"Personal"`, `"Home"`, `"Auto"`, `"Education"`, `"Business"`. `grep "Personal Loan"` over `src/`, `dataset.py`, `README.md` finds nothing. A grader running `Counter(r["category"])` and looking for `"Personal Loan"` gets 0 for every given value. | Rename the five category values to the brief's exact strings (or add a `category` field that uses them), regenerate, `make bless`. Amount-range keys, schema literals, answer templates ("(Business loan)") and transcripts 01/07/09/10/16/18 change with it. |
| T1 | Every given status ≥ 1 | PASS | Transcript 01 "By status": Submitted 13, Under Review 15, Approved 10, Rejected 13, Disbursed 9 (exact brief strings) | — |
| T1 | Required fields, `days_since_created` int 0–30, bool flag | PASS | `LoanApplication` dataclass in `dataset.py`; transcript 01 checks | — |
| T1 | `loan_amount_inr` range with one-sentence reasoning | PASS | README "Weights and amount ranges" (italic rationale sentence); per-category ranges table | — |
| T1 | Report count per category (each ≥ 3), per status, fraud % in 10–30 % | PASS | Transcript 01 and README "Dataset report": min category 6, fraud 10/60 = 16.67 % | — |
| T1 | Seed/weights changed rather than hand-editing | PASS | Seed history table (2026 valid on first draw); `validate()` in `dataset.py` | — |
| T2 | ≥ 12 docs, 2–5 sentences, every required topic | PASS | `kb/kb-01…kb-13`; transcript 02 inventory (3–5 sentences each) and topic→doc coverage table (12/12 + glossary) | — |
| T3 | Fixed-size-with-overlap and sentence chunking | PASS | `src/credagent/kb/chunking.py`; transcript 03 (160 chars / 40 overlap; one chunk per sentence) | — |
| T3 | Local SentenceTransformers embeddings | PASS | `src/credagent/rag/embedder.py:34` `SentenceTransformer(..., local_files_only=True)`, `all-MiniLM-L6-v2` | — |
| T3 | Two separate Chroma collections via `collection.upsert()` | PASS | `src/credagent/rag/store.py:112`; transcript 03 "Collections read back" (`kb_fixed_v1` 43, `kb_sentence_v1` 50) and sample query on both | — |
| T4 | Retrieve top-k, answer only from retrieved context | PASS | `src/credagent/rag/generator.py`; transcript 05 (every answer sentence verbatim from retrieved chunks) | — |
| T4 | Empirical calibration: ≥ 3 in-scope, ≥ 2 OOS top-1 cosine | PASS | 10 in-scope + 8 OOS per collection; transcript 04, `config/retrieval.json`, README table | — |
| T4 | Threshold sits between the measured clusters; values in README | PASS | `kb_sentence_v1`: max OOS 0.4797 < **0.567** < min in-scope 0.6543 (gap 0.1746). `kb_fixed_v1`: 0.4734 < 0.49 < 0.5063. All 36 values + thresholds are in README "Calibration". Not a tutorial preset. | — |
| T4 | Demo ≥ 5 in-scope + 1 OOS fallback | PASS (minor WEAK) | Transcript 05 Q1–Q5 grounded, Q6 (0.361) → "I don't know…". Q4 ("personal loan, score above 800") also emits an off-topic auto-loan sentence; a strict grader may question answer quality. | Optional: tighten the sentence-selection score so a second sentence must also match the query's loan type. |
| T5 | Document-level P/R, chunks → parent docs, dedup | PASS | `src/credagent/rag/evaluation.py`; `eval/results/chunking_pr.md` | — |
| T5 | Per-query arithmetic shown for **both** collections | PASS | `eval/results/chunking_pr.md` and transcript 06 show `P = a/b`, `R = a/b` per query for `kb_fixed_v1` **and** `kb_sentence_v1`, plus macro arithmetic. (README shows only row 1 of the sentence table, see S-README.) | — |
| T5 | P/R on **the same ≥ 5 queries from Task 4** | **FAIL** | Task 4 demo queries (transcript 05: annual fee, unauthorised transaction, close with dues, rate above 800, prepay home loan) and the P/R queries (transcript 06: NRI joint holder, prepayment + rate slab, closure fee year 1, savings shortfall, Under Review) are different sets; transcript 06 even says they are "disjoint". | Run P/R on the five transcript-05 queries (label their relevant docs), or make transcript 05 demo the five P/R queries. Keep the current set as extra evidence if wanted. |
| T5 | 2–3 sentence numbers-cited recommendation | PASS | `chunking_pr.md` "Recommendation" (2 sentences, cites 0.900/0.900 and gap 0.033 vs 0.175). P/R ties, so the decision rests on calibration gap, which is stated honestly. | — |
| T6 | `check_loan_application_status(record_id: str) -> dict` returning status, amount, score | PASS (minor WEAK) | `src/credagent/tools/lookup_tool.py:57`. Signature is `record_id: object, *, config=None, records=None) -> dict[str, Any]`, not the brief's `record_id: str`; the extra keyword-only args are injection seams. | Optional: annotate `record_id: str` (still accept anything at runtime). |
| T6 | Escalation score is a designed blend, not a boolean OR | PASS | `src/credagent/tools/escalation.py`: `0.4·flagged + 0.6·min(days,30)/30`; transcript 07 shows a flagged day-0 application scores 0.4 < T and does not escalate | — |
| T6 | Formula + threshold justified from own distribution | PASS | T = 0.6·P80/30 with P80 = 25 days (nearest rank, sorted list printed) → T = 0.5; `config/escalation.json`; README "Loan lookup and escalation" | — |
| T7 | ≥ 3 agents (Retrieval, Lookup, Composer) via `.kickoff()` | PASS | `src/credagent/crew/factory.py`, `agents.py`; transcript 08/09 wiring tables | — |
| T7 | RAG tool and lookup tool invoked on different queries | PASS | Transcript 08 ledger: only `rag_lookup`; transcript 09 run 1 ledger: only `check_loan_application_status` | — |
| T8 | LangChain `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` | PASS | `src/credagent/memory/session.py:31-34,145` | — |
| T8 | Multi-turn transcript showing state carried | PASS (WEAK on content) | Transcript 10: "it" → "a Business loan" resolved in turns 2–3, history 2→4→6. But turn 2's answer opens with a **Personal** loan sentence and turn 3's answer is about home/personal loans, not Business loans (the transcript admits this). A grader skimming the answers may conclude memory "didn't work". | Choose a follow-up chain the KB answers squarely for the remembered entity (e.g. LA-xxxx is a Home loan → "what credit score does it need?" → "is there a prepayment penalty on it?"), or improve sentence selection to prefer sentences naming the resolved loan type. |
| T8 | **Separate** fresh-session transcript showing state absent | PASS | Transcript 11 (separate file): fresh session, history length 0, "it" not resolved → clarify; other session's LA-0007 not leaked; reset also shown | — |
| T9 | Pydantic `BaseModel` for every crew response, `response_format`, validated in code | PASS | `CrewResponse` in `src/credagent/schemas.py`; `response_format=CrewResponse` on the Composer's LLM (`crew/factory.py:74`) and `output_pydantic` on the task; `parse_crew_output` validation; transcript 12 (7 cases + a rejected malformed one) | — |
| T10 | PII masking fires on PAN / Aadhaar / bank account | PASS | `src/credagent/guardrails/pii.py`; transcript 13 (all three types, plus through-pipeline sinks) | — |
| T10 | Prompt-injection detection fires | PASS | Transcript 14: 25/25 attacks blocked, 0/25 benign blocked, pipeline forgery case | — |
| T10 | Output groundedness refuses unsupported answers | PASS (fragile) | Transcript 15 case A: 0.555 vs 0.567 (margin 0.012), plus C/D/E unsupported-sentence refusals | A larger-margin below-threshold example would be more convincing (the 0.012 margin invites "lucky case" comments). |
| T11 | ≥ 2 HTTP endpoints with Pydantic models | PASS | `src/credagent/api/routes_http.py`, `models.py`; transcript 16 (`/v1/ask`, `/v1/documents`, `/v1/stats`, `DELETE /v1/sessions/{id}`, health) | — |
| T11 | `@app.websocket` endpoint, catches `WebSocketDisconnect`, survives | PASS | `src/credagent/api/routes_ws.py:116,128`; transcript 17 (A disconnects mid-turn, B completes 3 turns, `/healthz` 200) | — |
| T12 | One JSON-Lines entry per request with trace id + timing | PASS | `src/credagent/observability/logging.py`; transcript 18 (9 requests → 9 lines, `trace_id`, `latency_ms`, `timings_ms`) | — |
| T12 | Logged text masked like model input | PASS | Transcript 18 `request_text_masked` with typed tokens; "no raw PII canary anywhere in the log file" | — |
| T13 | 15 queries, every KB topic + ≥ 2 OOS/edge | PASS | `eval/testset.yaml` (E01–E12 one per topic, E13 OOS, E14 injection, E15 PII) | — |
| T13 | LLM-as-judge prompt under MOCK_LLM | PASS | `eval/judge_prompt.md`, `src/credagent/llm/judge.py` (`MockJudgeClient`) | — |
| T13 | 4 scores × 15 queries + 4 averages | PASS (WEAK on verifiability) | Transcript 19, `eval/results/eval_report.md/json`, README: full 15×4 table + averages. But **neither the queries nor the agent's answers are printed anywhere** ("query text is not printed"), and every one of the 60 cells is 1.000. A grader cannot check a single score against an answer and may read uniform 1.000 as a rigged judge. | Print each item's masked query and final answer in transcript 19 (masking already prevents PII leakage; E15 can show the masked form). Consider keeping the D-049 caveat prominent. |
| T14 | Autogen `RoundRobinGroupChat`, 2 agents, `max_turns=2` | PASS | `src/credagent/review/team.py` `build_review_team`: `RoundRobinGroupChat([reviewer, editor], max_turns=2, ...)` | — |
| T14 | `output_content_type` on editor + `custom_message_types=[StructuredMessage[...]]` on team | PASS | Same function: `output_content_type=ReviewVerdict`, `custom_message_types=[StructuredMessage[ReviewVerdict]]` | — |
| T14 | Verdict fields `approved`, `final_answer`, `reason`; input = draft + retrieved context | PASS | Transcripts 20/21 message 1 carries `<<<DRAFT>>>` and `<<<CONTEXT>>>`; verdict JSON has exactly the three fields | — |
| T14 | One approve-unchanged case, one revise case | PASS | Transcript 20 (`approved: true`, final == draft byte for byte); transcript 21 (injected "0% for life" claim removed, `approved: false`) | — |
| T15 | Least autonomy demonstrated + one-paragraph explanation | PASS | `governance/policy.py` `TOOL_ACL`, `tools/acl.py`, owner-bound `LoanStatusTool`; transcript 22 (b) wiring refused, (c) audit catches smuggled tool, (d) tampered owner refuses to run | — |
| T15 | Risk classification Low/Medium/High with justification | PASS | README "Governance": **High** (financial data), one paragraph; `docs/GOVERNANCE.md` | — |
| T15 | Per-request budget cap rejects an oversized request | PASS | `governance/budget.py`, `config/budget.json`; transcript 23: 12,000-char request projected 34,986 > 18,000 → `BudgetExceeded` at preflight, 0 LLM / 0 tool calls; HTTP 413 in transcripts 16/18 | — |
| T16 | In-memory cache keyed by normalized query for grounded generation | PASS | `src/credagent/cache/response_cache.py` (`normalize_query`; key also includes kb_version/strategy/threshold/top_k) | — |
| T16 | Cache hit with before/after evidence | PASS (note) | Transcript 24 / README: `generation_calls` +1 then +0, `retrieval_calls` +1 then +0, `cache_hits` +1, rag stage 25.7 ms → 0.1 ms. Note: the crew agents' mock-LLM calls still run on the hit (persona counters double), so "avoids a redundant LLM call" holds for the grounded-generation step only. | Optional: one sentence in README saying the cached step is the grounded-generation LLM call; the crew's own calls are not cached by design (`cache=False`). |

## Acceptance criteria

| Part | Criterion | Verdict | Evidence / note |
|---|---|---|---|
| 1 | Dataset meets thresholds; design choices in README | **WEAK** | Thresholds met (T1), but the category values are not the given vocabulary strings (see T1 FAIL). |
| 1 | KB ≥ 12 docs covering every topic | PASS | Transcript 02 |
| 1 | Two strategies, two collections, sensible retrieval on a sample query | PASS | Transcript 03 sample query top-1 = kb-08 on both |
| 1 | Grounded generation on ≥ 5 in-scope + 1 OOS fallback | PASS | Transcript 05 |
| 1 | P/R for both collections on the same ≥ 5 queries, per-query arithmetic, recommendation | **WEAK** | Both collections share the same 5 queries with full arithmetic, but they are not the Task 4 queries the task text names (T5 FAIL). |
| 2 | Lookup correct; designed, justified escalation score (not a boolean OR) | PASS | Transcript 07 |
| 2 | ≥ 3 agents; both tools invoked on different queries via `.kickoff()` | PASS | Transcripts 08, 09 |
| 2 | Multi-turn memory + separate fresh transcript | PASS | Transcripts 10, 11 (answer quality caveat under T8) |
| 2 | Every crew response validates against Pydantic schema | PASS | Transcript 12 |
| 2 | Both guardrail families fire on a deliberate case each | PASS | Transcripts 13, 14, 15 |
| 3 | ≥ 2 HTTP + 1 WS endpoint surviving disconnect, Pydantic models | PASS | Transcripts 16, 17 |
| 3 | One JSON-Lines entry per request with trace id | PASS | Transcript 18 |
| 3 | 4 scores per query × 15 + 4 averages under MOCK_LLM | PASS (WEAK on verifiability) | Transcript 19; answers not shown, all 1.000 |
| 4 | Autogen approves one, revises one, structured verdicts | PASS | Transcripts 20, 21 |
| 4 | Least autonomy; risk classification; oversized request rejected | PASS | Transcripts 22, 23; README Governance |
| 4 | Cache hit with before/after evidence | PASS | Transcript 24 |

## Submission guidelines

| Item | Verdict | Evidence / note |
|---|---|---|
| One public GitHub repo | Not verifiable here | `origin = github.com/abhishekpushker/iitr_as_2603385_capstone_banking`, `main` = `origin/main`. Visibility could not be checked (no `gh` CLI, no browsing). **Confirm the repo is public.** |
| README top states Cred (Banking & FinTech) track | PASS | README line 1 |
| README states seed, category/status weights, amount range | PASS | README "Dataset (Task 1)" |
| No images/PDF/slides/video/audio | PASS | `git ls-files` has no png/jpg/gif/svg/pdf/pptx/docx/mp3/mp4/wav |
| No paid account; MOCK_LLM, zero keys, zero network | PASS | README "Offline, mock-only operation"; transcript 25 (egress fails, 0 provider credentials, internal network) |
| Mock extends `crewai.llms.base_llm.BaseLLM`; both pitfalls handled | PASS | `src/credagent/llm/crewai_llm.py:35,111` (`MockCrewLLM(BaseLLM)`); parses only its own marked message, dispatches on `args_schema.model_fields` (line 222), never emits `Observation:` |
| `CREWAI_DISABLE_TELEMETRY` / `OTEL_SDK_DISABLED` set and confirmed in README | PASS | README "Offline…" bold paragraph; transcript 25 env table |
| README confirms everything runs under MOCK_LLM with zero API keys | PASS | README intro + "Offline…" section |
| Brief naming: `dataset.py`, `LOAN_APPLICATIONS`, `check_loan_application_status` | PASS | Root `dataset.py`; `LOAN_APPLICATIONS` module global; function and tool name `check_loan_application_status` |
| Brief naming: `/ask`, `/add-document`-style endpoints | PASS (minor) | `POST /v1/ask`, `POST /v1/documents` (admin key required). The brief says "e.g.", so acceptable; a grader trying `POST /add-document` with the user key gets 404 / 403. |
| No secrets committed | PASS | Only `.env.example` with `dev-api-key-not-a-secret` placeholders; no key-like strings found by pattern scan. `eval/testset.yaml` E15 holds an obviously fabricated PAN canary (`ABCPE1234F`), as the brief allows. |

### Grader-visible details (S-items)

| Item | Verdict | Note | Fix needed |
|---|---|---|---|
| S-README: sentence-collection P/R in README | WEAK | README shows the full fixed table but only row 1 of the sentence table ("same documents… different order for query 1 only"). Full arithmetic exists in `eval/results/chunking_pr.md`, but a grader reading only README sees one collection fully. | Paste the full 5-row `kb_sentence_v1` table into README. |
| S-SHA: transcript headers say `Git SHA: 35e671a…+dirty` | WEAK | Transcripts were generated from a dirty tree two commits before HEAD. A grader can read that as "outputs not from the submitted code". | Regenerate transcripts from a clean checkout of the final commit (`make bless`), commit as `chore(artifacts)`. |
| S-THRESH: deployed threshold refuses two P/R questions | WEAK | On `kb_sentence_v1`, P/R Q4 top-1 = 0.523 and Q5 = 0.515, both below the deployed 0.567, so the live agent would answer "I don't know" to two in-scope questions used to recommend that very collection. The calibration set's min in-scope (0.654) does not generalise. | Acknowledge in README, or add these near-boundary in-scope queries to calibration and re-derive the threshold honestly (gap may shrink). |
| S-PHONE: bare 10-digit number masked as bank account | Note | Transcript 13 "Call me on ⟨9999999999⟩" → `[BANK_ACCOUNT_REDACTED]`. Over-masking is safe, but a grader may flag it as a false positive. | Optional: mention in README limitations. |
| S-UNTRACKED: `.vscode/` | Note | Untracked; not in the repo. | Add to `.gitignore` or leave uncommitted. |

## Top risks (most mark-costly first)

0. **`make gate` is red from a cold volume (gate step [10]).** An order-dependent integration test fails when it runs before the KB is indexed. Every artifact check passed, but "the full gate fails" is the first thing a grader who runs it will see. Fix the test isolation before submitting.
1. **Category vocabulary mismatch (T1, FAIL).** The brief gives `Personal Loan` … `Business Loan` and says to use every value. The dataset uses `Personal` … `Business`. This is the easiest deduction for a grader to apply mechanically, and it touches Part 1's first acceptance criterion.
2. **P/R not run on the Task 4 queries (T5, FAIL).** The brief says "for the same ≥5 queries from Task 4". The repo deliberately uses a disjoint set. The arithmetic is exemplary, but the literal requirement is missed.
3. **Eval is unverifiable and uniformly perfect (T13, WEAK).** 60/60 cells at 1.000 with no query text or answers shown. Expect a grader to doubt the judge. Printing masked queries and answers is cheap and fixes this.
4. **Memory transcript answers look off-topic (T8, WEAK).** The standalone queries are right, but the answers talk about Personal/home loans after "it" resolved to a Business loan. This invites "memory not demonstrated".
5. **Threshold vs P/R queries (S-THRESH).** Two in-scope P/R questions fall below the deployed threshold. A careful grader cross-reading transcripts 04 and 06 will notice that calibration over-fits its own 10 questions.
6. **Dirty-tree transcript SHAs (S-SHA)** and **README showing only one full P/R table (S-README).** Both are presentation issues, quick to fix.
7. **Repo visibility not confirmed.** The brief requires a public repo; check before submitting.

## Resolution (branch `fix/audit-findings`, 2026-11-08)

Fixed test-first, then `make bless`; `make gate` is **GREEN**, including the cold-volume rerun.

| Finding | Resolution |
|---|---|
| Gate step 10 red | BUG-009: `test_chunking_eval.py` gets a module `indexed` fixture; reproduced and fixed on a fresh volume. |
| T1 category vocabulary (FAIL) | D-057: categories are `Personal Loan` … `Business Loan`; same draws, new sha256 `01a4304a…`. |
| T5 P/R not on the Task 4 queries (FAIL) | D-055 (Option A): one 10-query set for transcript 05 and transcript 06; P/R macro 0.850 / 0.950 on both collections. |
| T13 eval unverifiable | Transcript 19 and `eval_report.md` print each masked question and the scored answer. |
| T8 memory answers off-topic | D-056: answers keep to the loan type asked; transcripts 10/11 use LA-0037 (Home Loan). The same rule removes the off-topic auto-loan sentence from the old T4 demo answer. |
| S-THRESH | Option (b): threshold unchanged; README "The cost of the threshold" and Known limitations state the two declined in-scope questions. |
| S-README | README shows the full per-query P/R table for both collections. |
| S-SHA | Artifacts regenerated from a clean tree after the code commits. |
| T16 note, S-PHONE | Stated in README (cache scope; phone numbers masked as bank accounts). |
| Not changed | `record_id: object` signature (tests pass non-strings on purpose); the 0.012-margin groundedness example (D-026); repo visibility still to be confirmed by the owner. |
