# Governance — Cred Loan-Support Agent (Task 15)

## Risk classification: **High**

The agent handles financial data: loan amounts, application outcomes and fraud-review flags.
Members also type identity documents (PAN, Aadhaar) and bank account numbers into free text.
Its answers on eligibility, interest rates and prepayment penalties can shape a member's
borrowing decisions, and a wrong or invented answer can cost them money. The brief's scheme puts
financial data in the High tier. So the system runs with these controls: PII masking before
anything else sees the text, a human-escalation threshold on applications, an independent review
of every answer, fail-closed defaults, a per-request budget cap, one masked audit log line per
request, and fabricated data only.

## Least autonomy (Application layer)

Only the Lookup Agent may hold `check_loan_application_status`, and three independent guards
enforce this. **(1) Policy table.** `TOOL_ACL` in `governance/policy.py` lists the only agents
allowed to hold each tool; any tool not listed is denied to everyone. **(2) Wiring-time and
runtime guard.** `crew/agents.py::make_agent` checks every tool against the table and raises
`LeastAutonomyViolation` if a tool is not allowed. The lookup tool is bound to
`owner="lookup_agent"` and re-checks that owner on every `_run`, so an instance whose owner was
tampered with refuses to run. **(3) Crew audit.** After every `build_crew`, `audit_crew` walks
every agent *and every task's `tools` override*, and the request fails if any non-owner holds the
tool. All agents set `allow_delegation=False`, so one agent cannot ask the Lookup Agent to act
for it. There is no HTTP endpoint that calls the lookup directly: the only way to read a record
is a question answered by the crew's Lookup Agent. Transcript 22 demonstrates all four cases
(normal audit, refused wiring, smuggled tool, tampered owner).

## Four layers

| Layer | Controls |
|---|---|
| **Data** | Fabricated dataset and KB only; PII (PAN, Aadhaar, bank account) is masked at stage 1, before memory, cache, Chroma, the crew or logs; retention: logs rotate after 7 days, session memory has a 1 h idle TTL. |
| **Model** | Deterministic `MOCK_LLM` engine (no API keys, no external calls); pinned embedding-model revision loaded from a local path. |
| **Application** | Least autonomy (above); prompt-injection detection before the crew; grounded generation with a calibrated similarity threshold; Autogen reviewer/editor stage; output groundedness gate; schema-validated structured output; failures end in a safe refusal. |
| **Runtime** | Budget cap (below); crew-kickoff timeout; offline networking (internal Docker network, telemetry off); one structured, masked log line per request. |

### Runtime budget cap

Before any LLM call, `BudgetPolicy.preflight` projects the request's worst case:
`7 × (PROMPT_OVERHEAD_TOKENS + ceil(len/4) + MAX_OUTPUT_TOKENS)`. It rejects the request (HTTP 413,
`BUDGET_EXCEEDED`, zero LLM calls) if the projection exceeds `BUDGET_MAX_TOKENS` or its cost
exceeds `BUDGET_MAX_COST_USD`. The constants are measured by `scripts/measure_budget.py` from
real rendered prompts and written to `config/budget.json` (generated, never hand-edited):
overhead 1,700, max output 241, budget 17,000 tokens / $0.034. During the run, the `BudgetMeter`
charges every mock call. If a request overruns at runtime, it ends with `BudgetExceeded` after
kickoff, and nothing is written to memory. The limits are ordered so that the budget, not the
body cap or Pydantic, rejects long input: body cap 64 KiB > `MAX_MESSAGE_CHARS` 16,000 × 4 bytes
> the smallest rejected input (≈ 2,300 chars). Transcript 23 shows the numbers.

## Known limitation: IDOR

Any caller with the API key can ask about any application id. This is an insecure direct object
reference: no member identity is modelled, so nothing ties a lookup to the person asking. In
production, lookups would be bound to the authenticated member's identity (the record's owner
must match the caller), and the API key would only identify the client application. This is out
of scope here because the brief models no user identity.
