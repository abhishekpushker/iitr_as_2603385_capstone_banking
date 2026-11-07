# CLAUDE.md — operating rules for this repository

Project: **Cred (Banking & FinTech) Loan-Support Agent** — CrewAI crew + RAG (ChromaDB) + Autogen review + FastAPI, fully in Docker, `MOCK_LLM` only.

## Read before doing anything
1. `docs/ARCHITECTURE.md` — the normative design (what to build). Section refs (§) point there.
2. `docs/IMPLEMENTATION_PLAN.md` — the phase order, tests-first lists, and exit gates (how and when).
3. `docs/DECISIONS.md` and `docs/BUG_LOG.md` — what has already been decided or fixed.

Find the current phase from the latest `phase-NN-green` git tag. Work only on the next phase. Do not start a phase before the previous gate is green.

## Hard rules
- **Docker only.** Run everything through `make` targets. Never `pip install` or run Python on the host.
- **No external network at runtime.** Graded scripts and tests run on the `backend` (internal) network. Only `make lock` and `make build` may use the network.
- **`MOCK_LLM=true` always.** No API keys. Telemetry flags per §ARCH 5 must be set before `import crewai`.
- **Tests first.** For every requirement and every bug, write the failing test first, then the code.
- **Never** weaken an assertion, skip/xfail a test, add sleeps or retries to hide flakiness, special-case a query string in the mock, or hand-edit generated artifacts (`transcripts/`, `config/*.json`, `eval/results/`). Regenerate with `make bless` and review the diff.
- **Mask first.** Raw user text must not reach memory, cache, Chroma, the crew, logs or error bodies (§ARCH 6, 7.12).
- **Least autonomy.** Only the Lookup Agent may hold `check_loan_application_status`. All agents: `cache=False`, `allow_delegation=False`, `max_iter=4`.
- **Mock adapter rules (§ARCH 7.8.1):** parse tool observations only inside our own marked assistant message; dispatch tool arguments from `args_schema`, never from tool names; never emit `Observation:`; never raise from `call()` for budget, set the flag instead.
- Keep the layering in §ARCH 4. All config goes through `credagent/settings.py`.
- **Injected dependencies default with `is None`, never `x or Default()`** (an empty store or a falsy test double would be silently replaced). D-037.

## The loop
```
red test → minimal code → green → refactor → make gate-fast
end of phase → make gate (full, incl. cold-volume rerun) → commit → tag phase-NN-green
```
Bug found? Reproduce it with a test (`BUG-NNN` in the docstring), log it in `docs/BUG_LOG.md`, fix the root cause, run the full gate. Library surprise? Add a contract test in `tests/contract/`.

## Stop and ask the human when
A contract test contradicts the architecture; the calibration gap stays < 0.05 after two honest revisions; an acceptance criterion seems unreachable under MOCK_LLM; anything appears to need internet at runtime.

## Commands
`make init` · `make build` · `make up` · `make gate-fast` · `make gate` · `make transcripts` · `make eval` · `make bless` · `make pii-scan` · `make offline-proof`

## Commits
Conventional Commits (`feat(rag): …`, `test(crew): …`, `fix(api): … (BUG-007)`). Generated-artifact updates go in separate `chore(artifacts): …` commits.
