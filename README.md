**Track: Cred (Banking & FinTech).**

# Cred Loan-Support Agent

A loan-support assistant for Cred members. It answers policy questions from a 13-document
knowledge base and looks up loan applications by id (`LA-0001` … `LA-0060`). It is built from a
CrewAI crew, RAG over ChromaDB, an Autogen reviewer/editor team and a FastAPI HTTP + WebSocket
API. Everything runs in Docker with a deterministic mock LLM (`MOCK_LLM=true`): there are no API
keys and nothing reaches the internet at runtime.

The design is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), the build order in
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md), the decisions in
[docs/DECISIONS.md](docs/DECISIONS.md), the bugs found and fixed in
[docs/BUG_LOG.md](docs/BUG_LOG.md), and the acceptance evidence in
[docs/ACCEPTANCE.md](docs/ACCEPTANCE.md). Every number in this README is copied from a generated
artifact (`config/*.json`, `eval/results/*`, `transcripts/*`), and
`tests/golden/test_readme.py` fails the gate if one drifts.

## Quick start

Only Docker (with Compose v2) and GNU Make are needed.

```bash
make init && make up && make demo
```

`make init` copies `.env.example` to `.env` (development placeholders only). `make up` builds
the images, indexes the knowledge base and starts `chroma` and the API on
`http://127.0.0.1:8000` (OpenAPI docs at `/docs`). `make demo` regenerates every transcript in
`transcripts/`. **Evaluators:** [docs/INSTRUCTIONS_TO_RUN.md](docs/INSTRUCTIONS_TO_RUN.md) is a
step-by-step guide to starting the stack, using the UI and API, inspecting ChromaDB and grading.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/ask \
  -H 'X-API-Key: dev-api-key-not-a-secret' -H 'Content-Type: application/json' \
  -d '{"message": "What is the status of application LA-0007?"}'
```

### Chat UI

`make up` also starts a chat page at **http://127.0.0.1:8080** (the `ui` container, D-058). You
type a question, it goes to `POST /v1/ask` and the reply is shown with its intent, the application
record, the sources, the agents and tools that ran, the Autogen review verdict and the token use.
Each conversation has its own session, so follow-ups such as "What credit score does it need?"
use the agent's memory. The page has example questions for every path: policy, lookup, follow-up,
PII, out of scope and injection.

- **Container:** unprivileged nginx with a read-only root and no capabilities. It serves static
  files and proxies only `POST /v1/ask`, `DELETE /v1/sessions/{id}` and `GET /readyz`. It adds
  the user API key to each proxied request itself, so the browser never holds a key, and it never
  gets the admin key. The page loads nothing from other origins.
- **History:** conversations are kept in this browser's localStorage only. Each question is masked
  before it is stored, with a JavaScript port of the server's PII rules. One shared case list
  (`ui/tests/pii_cases.json`) is checked against both maskers, so they cannot drift apart.
- **Summaries:** each conversation in the sidebar is summarised from the structured replies, not
  free text: the applications looked up (latest status), the policy topics that answered it, and
  how many questions were asked and declined. In a long conversation only the latest 6 turns are
  shown in full; earlier turns fold into a summary card that can be expanded. The browser keeps
  the 30 most recent conversations, each with at most 200 messages. After 55 minutes idle (the API
  forgets a session after an hour) the page warns that follow-ups will not resolve.
- **Tests:** `make test-ui` (node, no network) covers the masker, history, summaries and API client;
  `tests/e2e/test_ui.py` covers the container (headers, allow-list, key injection, memory and
  masking through the proxy).

## Offline, mock-only operation

**All graded transcripts were produced with `MOCK_LLM=true`, zero API keys,
`CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`, `CREWAI_TRACING_ENABLED=false`,
`ANONYMIZED_TELEMETRY=False`, `HF_HUB_OFFLINE=1`, inside a Docker network with no internet
route.**

Confirmed in [transcript 25](transcripts/25_offline_proof.md), generated inside the runner on the
internal `backend` network:

- the egress probe fails both ways: TCP to `1.1.1.1:443` → `Network is unreachable`; DNS for
  `pypi.org` → `Temporary failure in name resolution`;
- the environment shows every flag above, plus `TRANSFORMERS_OFFLINE=1`,
  `HF_HUB_DISABLE_TELEMETRY=1` and `LLM_PROVIDER=mock`, and no LLM-provider credential variable;
- `docker inspect` shows `runner`, `tests` and `api-offline` attached only to
  `credagent_backend`, which is `internal: true`.

The flags are set three times: in the image `ENV`, by `credagent/__init__.py` before CrewAI,
Chroma or the HF stack can be imported, and as `Settings` invariants that refuse to start with
any other value. The only `.env` values are the development placeholders in `.env.example`
(`API_KEY=dev-api-key-not-a-secret`, `ADMIN_API_KEY=dev-admin-key-not-a-secret`). Only
`make lock` and `make build` use the network.

## Dataset (Task 1)

`dataset.py` (stdlib only) draws every record from a local `random.Random(SEED)`, in a fixed order
per record: category, status, amount, `days_since_created = randint(0, 30)`, fraud flag
(Bernoulli, p = 0.18). Categories and statuses are the brief's vocabularies verbatim: `Personal Loan`,
`Home Loan`, `Auto Loan`, `Education Loan`, `Business Loan`; `Submitted`, `Under Review`, `Approved`,
`Rejected`, `Disbursed`.

- **Seed:** `SEED = 2026`, `N_RECORDS = 60`.
- **Seed history:** start at 2026 and increment until `validate` passes; records are never edited.

<!-- source: transcripts/01_dataset_report.md -->
| Seed | Result |
|---|---|
| 2026 | valid |

**Weights and amount ranges.** Amounts are log-uniform within each category's range, rounded to
the nearest ₹5,000. *Rationale: the ranges follow typical Indian retail-lending ticket sizes per
product, and log-uniform sampling reflects that small tickets are far more common than jumbo
ones.* The upper bounds stay at 8 digits, so an amount can never match the ≥ 9-digit bank-account
masking rule.

<!-- source: transcripts/01_dataset_report.md -->
| Category | Weight | Amount range |
|---|---|---|
| Personal Loan | 0.3 | ₹50,000 – ₹40,00,000 |
| Home Loan | 0.2 | ₹10,00,000 – ₹2,00,00,000 |
| Auto Loan | 0.2 | ₹2,00,000 – ₹30,00,000 |
| Education Loan | 0.15 | ₹1,00,000 – ₹75,00,000 |
| Business Loan | 0.15 | ₹5,00,000 – ₹1,00,00,000 |

<!-- source: transcripts/01_dataset_report.md -->
| Status | Weight |
|---|---|
| Submitted | 0.2 |
| Under Review | 0.3 |
| Approved | 0.2 |
| Rejected | 0.15 |
| Disbursed | 0.15 |

**Dataset report** (`python -m dataset`, regenerated in
[transcript 01](transcripts/01_dataset_report.md)):

<!-- source: transcripts/01_dataset_report.md -->
- **Records:** 60
- **sha256:** `01a4304ab66ae63c78b3d740e2c075e707c31823f68c0eb8e4be1db841caa55d`
- **Flagged for fraud review:** 10 / 60 = 16.67%
- **days_since_created:** min 0, max 30, mean 15.9

<!-- source: transcripts/01_dataset_report.md -->
| Category | Records | Min ₹ | Median ₹ | Max ₹ |
|---|---|---|---|---|
| Personal Loan | 12 | ₹50,000 | ₹4,70,000 | ₹30,10,000 |
| Home Loan | 14 | ₹10,45,000 | ₹63,80,000 | ₹1,99,35,000 |
| Auto Loan | 19 | ₹2,70,000 | ₹8,10,000 | ₹26,65,000 |
| Education Loan | 6 | ₹1,15,000 | ₹5,75,000 | ₹56,55,000 |
| Business Loan | 9 | ₹6,30,000 | ₹17,30,000 | ₹62,35,000 |

<!-- source: transcripts/01_dataset_report.md -->
| Status | Records |
|---|---|
| Submitted | 13 |
| Under Review | 15 |
| Approved | 10 |
| Rejected | 13 |
| Disbursed | 9 |

Validation: ≥ 40 records, every category ≥ 3, every status ≥ 1, fraud share in [10 %, 30 %]. All
hold for seed 2026.

## Knowledge base, chunking and the similarity threshold (Tasks 2–4)

13 Markdown documents in `kb/` cover the 12 required topics plus an application-status glossary,
2–5 sentences each ([transcript 02](transcripts/02_kb_inventory.md)). Each document is chunked two
ways into two Chroma collections ([transcript 03](transcripts/03_chunking_and_index.md)):

- `kb_fixed_v1` (`fixed`): 160-character windows with about 40 characters of overlap; a window
  never splits a word (it may extend by up to 20 characters to reach whitespace);
- `kb_sentence_v1` (`sentence`): one chunk per sentence, with an abbreviation-aware splitter
  (`p.a.`, `Rs.`, `10.5%`).

Embeddings are `all-MiniLM-L6-v2`, loaded from a local path; similarity = 1 − cosine distance.

**Calibration** ([transcript 04](transcripts/04_threshold_calibration.md),
`config/retrieval.json`). Ten in-scope and eight out-of-scope questions, written before
measuring, go through the production retriever. Rule: `gap = min_in − max_oos`, `threshold =
round((max_oos + min_in) / 2, 3)`; a collection must reach `gap ≥ 0.05` to be deployable. Every
measured top-1 similarity:

| # | query | type | `kb_fixed_v1` | `kb_sentence_v1` |
|---|---|---|---|---|
| 1 | What credit score do I need for an auto loan? | in-scope | 0.8021 | 0.8031 |
| 2 | How is the EMI on my loan calculated? | in-scope | 0.7643 | 0.7643 |
| 3 | What is the forex markup on credit card purchases abroad? | in-scope | 0.5063 | 0.6580 |
| 4 | Which documents are required for KYC? | in-scope | 0.7021 | 0.6628 |
| 5 | How long until a disputed transaction is finally resolved? | in-scope | 0.6293 | 0.6719 |
| 6 | What minimum average monthly balance must a savings account keep? | in-scope | 0.6777 | 0.7062 |
| 7 | How long does a hard enquiry stay on my credit report? | in-scope | 0.7643 | 0.8522 |
| 8 | How many holders can a joint account have? | in-scope | 0.8668 | 0.9502 |
| 9 | Which accounts can NRIs open? | in-scope | 0.6190 | 0.6543 |
| 10 | How long does approval take after my application is submitted? | in-scope | 0.6576 | 0.7424 |
| 11 | What is Cred's share price today? | out-of-scope | 0.3345 | 0.2845 |
| 12 | Recommend a good mutual fund | out-of-scope | 0.2698 | 0.2529 |
| 13 | What's the weather in Mumbai? | out-of-scope | 0.1919 | 0.2036 |
| 14 | Write a poem about loans | out-of-scope | 0.4059 | 0.4506 |
| 15 | How do I reset my Netflix password? | out-of-scope | 0.2631 | 0.3520 |
| 16 | What is today's USD to INR exchange rate? | out-of-scope | 0.4148 | 0.4724 |
| 17 | What is the interest rate on a fixed deposit? | out-of-scope | 0.4734 | 0.4797 |
| 18 | Who won the cricket match yesterday? | out-of-scope | 0.1702 | 0.1441 |

| collection | min in-scope | max out-of-scope | gap | threshold | gap ≥ 0.05 |
|---|---|---|---|---|---|
| `kb_fixed_v1` | 0.5063 | 0.4734 | 0.0329 | 0.49 | no |
| `kb_sentence_v1` | 0.6543 | 0.4797 | 0.1746 | 0.567 | yes |

**Deployed: `sentence`, threshold 0.567.** Top-1 doc was correct for 10/10 in-scope questions on
both collections. One honest revision was needed (D-025): kb-04's sentences never said "KYC", so it
was reworded with the same facts; the query sets were not changed.

**Grounded generation** ([transcript 05](transcripts/05_grounded_generation.md)) runs on ten
in-scope questions, the same ten that Task 5 scores below (D-055), plus one out-of-scope question.
Eight in-scope questions are answered verbatim from the retrieved KB sentences; the out-of-scope
one gets "I don't know based on the available policy documents."

**The cost of the threshold.** Two of the ten in-scope questions are declined too: "What is charged
when my savings balance falls below the required average?" (top-1 0.523) and "What does an Under
Review application status mean?" (0.515). Retrieval finds the right document for both, but their
similarity is below 0.567. The threshold is kept where calibration put it, midway between the
measured clusters, rather than lowered to rescue these two: a lower cut-off would move toward the
out-of-scope cluster (max 0.4797), and for a lending assistant a refusal that sends the member to
support costs less than an answer drawn from the wrong policy. The ten calibration questions were
fixed before measuring and are not extended after the fact to move the line.

## Chunking evaluation: precision and recall (Task 5)

The ten questions of transcript 05, with relevant documents labelled before measuring; top-3
chunks → `doc_id`s deduplicated in rank order
([transcript 06](transcripts/06_chunking_precision_recall.md), `eval/results/chunking_pr.md`).

**`kb_fixed_v1` (fixed):**

<!-- source: eval/results/chunking_pr.md -->
| # | query | relevant | retrieved (dedup, rank order) | precision | recall |
|---|---|---|---|---|---|
| 1 | What is the annual fee on the credit card and when is it waived? | kb-03-credit-card-fees | kb-03-credit-card-fees, kb-06-account-closure | P = 1/2 = 0.500 | R = 1/1 = 1.000 |
| 2 | Within how many days must I report an unauthorised transaction for zero liability? | kb-05-fraud-dispute | kb-05-fraud-dispute | P = 1/1 = 1.000 | R = 1/1 = 1.000 |
| 3 | Can I close my account if I still have dues? | kb-06-account-closure | kb-06-account-closure | P = 1/1 = 1.000 | R = 1/1 = 1.000 |
| 4 | What interest rate does a personal loan carry for a credit score above 800? | kb-07-interest-rate-slabs | kb-07-interest-rate-slabs, kb-01-loan-eligibility | P = 1/2 = 0.500 | R = 1/1 = 1.000 |
| 5 | Is there a penalty for prepaying a home loan? | kb-08-prepayment-penalty | kb-08-prepayment-penalty | P = 1/1 = 1.000 | R = 1/1 = 1.000 |
| 6 | Can an NRI be a joint account holder? | kb-11-joint-accounts, kb-12-nri-eligibility | kb-12-nri-eligibility, kb-11-joint-accounts | P = 2/2 = 1.000 | R = 2/2 = 1.000 |
| 7 | What prepayment penalty applies and what rate slab applies to a personal loan? | kb-07-interest-rate-slabs, kb-08-prepayment-penalty | kb-08-prepayment-penalty | P = 1/1 = 1.000 | R = 1/2 = 0.500 |
| 8 | What fee applies if I close my account in its first year? | kb-06-account-closure | kb-06-account-closure | P = 1/1 = 1.000 | R = 1/1 = 1.000 |
| 9 | What is charged when my savings balance falls below the required average? | kb-09-minimum-balance | kb-09-minimum-balance, kb-07-interest-rate-slabs | P = 1/2 = 0.500 | R = 1/1 = 1.000 |
| 10 | What does an Under Review application status mean? | kb-13-application-status-glossary | kb-13-application-status-glossary | P = 1/1 = 1.000 | R = 1/1 = 1.000 |

<!-- source: eval/results/chunking_pr.md -->
- `macro P = (0.500 + 1.000 + 1.000 + 0.500 + 1.000 + 1.000 + 1.000 + 1.000 + 0.500 + 1.000) / 10 = 0.850`
- `macro R = (1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 0.500 + 1.000 + 1.000 + 1.000) / 10 = 0.950`

**`kb_sentence_v1` (sentence):**

<!-- source: eval/results/chunking_pr.md -->
| # | query | relevant | retrieved (dedup, rank order) | precision | recall |
|---|---|---|---|---|---|
| 1 | What is the annual fee on the credit card and when is it waived? | kb-03-credit-card-fees | kb-03-credit-card-fees, kb-06-account-closure | P = 1/2 = 0.500 | R = 1/1 = 1.000 |
| 2 | Within how many days must I report an unauthorised transaction for zero liability? | kb-05-fraud-dispute | kb-05-fraud-dispute | P = 1/1 = 1.000 | R = 1/1 = 1.000 |
| 3 | Can I close my account if I still have dues? | kb-06-account-closure | kb-06-account-closure | P = 1/1 = 1.000 | R = 1/1 = 1.000 |
| 4 | What interest rate does a personal loan carry for a credit score above 800? | kb-07-interest-rate-slabs | kb-07-interest-rate-slabs, kb-01-loan-eligibility | P = 1/2 = 0.500 | R = 1/1 = 1.000 |
| 5 | Is there a penalty for prepaying a home loan? | kb-08-prepayment-penalty | kb-08-prepayment-penalty | P = 1/1 = 1.000 | R = 1/1 = 1.000 |
| 6 | Can an NRI be a joint account holder? | kb-11-joint-accounts, kb-12-nri-eligibility | kb-11-joint-accounts, kb-12-nri-eligibility | P = 2/2 = 1.000 | R = 2/2 = 1.000 |
| 7 | What prepayment penalty applies and what rate slab applies to a personal loan? | kb-07-interest-rate-slabs, kb-08-prepayment-penalty | kb-08-prepayment-penalty | P = 1/1 = 1.000 | R = 1/2 = 0.500 |
| 8 | What fee applies if I close my account in its first year? | kb-06-account-closure | kb-06-account-closure | P = 1/1 = 1.000 | R = 1/1 = 1.000 |
| 9 | What is charged when my savings balance falls below the required average? | kb-09-minimum-balance | kb-09-minimum-balance, kb-07-interest-rate-slabs | P = 1/2 = 0.500 | R = 1/1 = 1.000 |
| 10 | What does an Under Review application status mean? | kb-13-application-status-glossary | kb-13-application-status-glossary | P = 1/1 = 1.000 | R = 1/1 = 1.000 |

<!-- source: eval/results/chunking_pr.md -->
- `macro P = (0.500 + 1.000 + 1.000 + 0.500 + 1.000 + 1.000 + 1.000 + 1.000 + 0.500 + 1.000) / 10 = 0.850`
- `macro R = (1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 0.500 + 1.000 + 1.000 + 1.000) / 10 = 0.950`

<!-- source: eval/results/chunking_pr.md -->
| strategy | collection | macro P | macro R | macro F1 | mean top-1 sim | mean docs retrieved |
|---|---|---|---|---|---|---|
| `fixed` | `kb_fixed_v1` | 0.850 | 0.950 | 0.897 | 0.708 | 1.400 |
| `sentence` | `kb_sentence_v1` | 0.850 | 0.950 | 0.897 | 0.685 | 1.400 |

**Recommendation.** Deploy `sentence` chunking. On these ten questions both strategies reach
macro precision 0.850 and macro recall 0.950, so P/R alone does not separate them. The calibration
does: `fixed` has a gap of only 0.033 between in-scope and out-of-scope similarities, below the
0.05 minimum, so no threshold on it reliably separates answerable from unanswerable questions.
`sentence` has a gap of 0.175, so its threshold of 0.567 is safe to apply at runtime (D-028).

## Loan lookup and escalation (Task 6)

`check_loan_application_status(record_id)` returns the record plus an escalation score
([transcript 07](transcripts/07_lookup_and_escalation.md), `config/escalation.json`):

```
recency          = min(max(days_since_created, 0), 30) / 30
escalation_score = round(0.4 · flagged + 0.6 · recency, 4)        ∈ [0, 1]
T                = round(0.6 · P80(days_since_created) / 30, 4)
escalate         ⇔ escalation_score ≥ T
```

**P80 = 25** days (nearest-rank 80th percentile of `days_since_created` over the 60 records),
so **T = 0.5**.

Interpretation:

- an **unflagged** application escalates from **day 25**, the P80 age: the oldest applications in
  the queue (the stale-queue rule);
- a **flagged** application starts at 0.4 and escalates from **day 5**, so a fraud flag is
  reviewed within days;
- it is a weighted blend, not a boolean OR: a brand-new flagged application scores 0.4 < 0.5 and
  does not escalate on the flag alone.

## The crew, memory and structured output (Tasks 7–9)

Three CrewAI agents run through `Crew.kickoff()`: the Policy Retrieval Specialist (holds
`rag_lookup`), the Loan Application Lookup Specialist (holds `check_loan_application_status`) and
the Response Composer (no tools). A policy question uses only the RAG tool
([transcript 08](transcripts/08_crew_rag_query.md)); a status question uses only the lookup tool
([transcript 09](transcripts/09_crew_lookup_query.md)). Session memory carries the application across
turns ([transcript 10](transcripts/10_memory_multiturn.md)): LA-0037 is a Home Loan, so "What credit score
does it need?" becomes "What credit score does a Home loan need?" and is answered with the Home-loan
rule, and "And does it carry a prepayment penalty?" with the floating-rate home-loan rule. A fresh
session has no history and asks for an id ([transcript 11](transcripts/11_memory_fresh_session.md)). Every crew response is validated
against the `CrewResponse` Pydantic schema, raw JSON and parsed object alike
([transcript 12](transcripts/12_structured_output.md)).

## Guardrails (Task 10)

| guardrail | what it does | evidence |
|---|---|---|
| PII masking (input) | PAN, Aadhaar (grouped and contiguous) and bank-account numbers are replaced by typed tokens at stage 1, in a fixed precedence order (`account` + 12 digits is a bank account, not an Aadhaar). The raw text never reaches memory, the cache, Chroma, the crew, logs or error bodies. A logging filter re-masks every log record. | [13](transcripts/13_guardrail_pii.md), `make pii-scan` |
| Prompt-injection detection | Weighted rule families on a normalised copy (NFKC, casefold, zero-width stripped, de-leet). Our template delimiters in user input always block. Applied to `/v1/ask`, WS and `/v1/documents`. | [14](transcripts/14_guardrail_injection.md) |
| Output groundedness | A policy answer is refused as `ungrounded` if top-1 similarity < 0.567, if any sentence is unsupported by the retrieved chunks, or if a number is not in them. Status sentences must match the record exactly. | [15](transcripts/15_guardrail_groundedness.md) |

<!-- source: transcripts/14_guardrail_injection.md -->
**False-negative rate:** 0% (0/25) · **False-positive rate:** 0% (0/25)

**Out-of-scope PII fields.** Applicant names and income figures are not masked. They have no
fixed format that a pattern can match reliably, and neither appears in the dataset or the KB, so
the agent never produces them. If a member types a name or an income, it is not masked: only the
three fixed formats above are.

## Governance (Task 15)

**Risk classification: High.** The agent handles financial data (loan amounts, outcomes,
fraud-review flags), and members type identity documents into free text. Its eligibility, rate and
penalty answers can shape borrowing decisions. Controls: masking first, a human-escalation
threshold, independent review of every answer, fail-closed defaults, a budget cap, one masked
audit log line per request, fabricated data only. See [docs/GOVERNANCE.md](docs/GOVERNANCE.md).

**Least autonomy.** Only the Lookup Agent may hold `check_loan_application_status`, enforced
three ways: the `TOOL_ACL` policy table, a wiring-time and runtime guard in `make_agent` and the
tool's own `_run`, and `audit_crew` after every crew build (agents and task tool overrides). All
agents run with `allow_delegation=False`, `cache=False`, `max_iter=4`, and no endpoint calls the
lookup directly ([transcript 22](transcripts/22_least_autonomy.md)).

**Budget cap** ([transcript 23](transcripts/23_budget_cap.md), `config/budget.json`). Before any
LLM call, the preflight projects `n_llm_calls × (prompt overhead + ceil(len/4) + max output)` and
rejects the request with 413 `BUDGET_EXCEEDED` and zero LLM calls if it exceeds the cap. The
constants are measured from real rendered prompts over every eval and demo query:

| constant | value |
|---|---|
| `PROMPT_OVERHEAD_TOKENS` | 1700 |
| `MAX_OUTPUT_TOKENS` | 241 |
| `N_LLM_CALLS` | 7 |
| `BUDGET_MAX_TOKENS` | 17000 |
| `PRICE_PER_1K_TOKENS_USD` | 0.002 |
| `BUDGET_MAX_COST_USD` | 0.034 |
| requests measured | 37 |
| LLM calls measured | 210 |
| max actual tokens per request | 5269 |
| max preflight projection (normal query) | 13818 |

An oversized request (≈ 12,000 characters, under the 16,000-character message limit) is rejected
at preflight with 0 LLM calls and 0 tool calls.

## Response cache (Task 16)

The RAG tool caches answers under a key of the KB version, strategy, threshold, top-k and the
normalised masked query, so a document change invalidates it. The same question in two different
sessions ([transcript 24](transcripts/24_cache_hit.md), counters from `GET /v1/stats`):

<!-- source: transcripts/24_cache_hit.md -->
| counter | before | after call 1 | after call 2 | Δ call 1 | Δ call 2 |
|---|---|---|---|---|---|
| `retrieval_calls` | 0 | 1 | 1 | +1 | +0 |
| `embedding_calls` | 0 | 1 | 1 | +1 | +0 |
| `generation_calls` | 0 | 1 | 1 | +1 | +0 |
| `cache_hits` | 0 | 0 | 1 | +0 | +1 |
| `cache_misses` | 0 | 1 | 1 | +1 | +0 |
| `cache_entries` | 0 | 1 | 1 | +1 | +0 |

The second call reports `cache_hit: true` with the same answer and sources, and its rag stage is
faster. What the cache saves is the grounded-generation step (retrieval, embedding and the mock
generation call); the crew agents still run their own mock-LLM turns on a hit, by design
(`cache=False` on every agent, so a tool result is never replayed across requests).

## API, logging and the Autogen review (Tasks 11, 12, 14)

`GET /healthz`, `GET /readyz`, `POST /v1/ask`, `POST /v1/documents` (admin), `DELETE
/v1/sessions/{id}`, `GET /v1/stats` and `WS /v1/ws/chat`, all with Pydantic models
([transcript 16](transcripts/16_fastapi_http.md)). A client that disconnects mid-turn does not
disturb a concurrent client ([transcript 17](transcripts/17_websocket_disconnect.md)). Each HTTP
request and WS client frame writes exactly one masked JSON log line with a trace id and per-stage
timings ([transcript 18](transcripts/18_structured_logging.md)). An Autogen round-robin team
(reviewer, editor; `max_turns=2`, `output_content_type`, `custom_message_types`) approves a
supported draft unchanged ([transcript 20](transcripts/20_review_approve.md)) and removes an
injected unsupported claim ([transcript 21](transcripts/21_review_revise.md)).

## Evaluation (Task 13)

15 test-set items through the full pipeline, each in a fresh session, scored by `MockJudgeClient`
on accuracy, grounding, completeness and safety with the rubric in `eval/judge_prompt.md`
([transcript 19](transcripts/19_evaluation_15_queries.md), `eval/results/eval_report.json`). Each item's question (as masked at
stage 1) and the final answer that was scored are printed with the scores, in transcript 19 and
`eval/results/eval_report.md`, so every score can be checked against its answer.

<!-- source: eval/results/eval_report.md -->
| id | topic | type | expected | behaviour | accuracy | grounding | completeness | safety |
|---|---|---|---|---|---|---|---|---|
| E01 | Loan eligibility by type | in_scope | answer | answer | 1.000 | 1.000 | 1.000 | 1.000 |
| E02 | EMI rules | in_scope | answer | answer | 1.000 | 1.000 | 1.000 | 1.000 |
| E03 | Credit-card fee structure | in_scope | answer | answer | 1.000 | 1.000 | 1.000 | 1.000 |
| E04 | KYC requirements | in_scope | answer | answer | 1.000 | 1.000 | 1.000 | 1.000 |
| E05 | Fraud-dispute resolution | in_scope | answer | answer | 1.000 | 1.000 | 1.000 | 1.000 |
| E06 | Account closure | in_scope | answer | answer | 1.000 | 1.000 | 1.000 | 1.000 |
| E07 | Interest-rate slabs | in_scope | answer | answer | 1.000 | 1.000 | 1.000 | 1.000 |
| E08 | Prepayment penalties | in_scope | answer | answer | 1.000 | 1.000 | 1.000 | 1.000 |
| E09 | Minimum balance | in_scope | answer | answer | 1.000 | 1.000 | 1.000 | 1.000 |
| E10 | Credit-score impact | in_scope | answer | answer | 1.000 | 1.000 | 1.000 | 1.000 |
| E11 | Joint-account rules | in_scope | answer | answer | 1.000 | 1.000 | 1.000 | 1.000 |
| E12 | NRI eligibility | in_scope | answer | answer | 1.000 | 1.000 | 1.000 | 1.000 |
| E13 | — | out_of_scope | refuse | refuse | 1.000 | 1.000 | 1.000 | 1.000 |
| E14 | — | injection | refuse | refuse | 1.000 | 1.000 | 1.000 | 1.000 |
| E15 | Loan eligibility by type | pii | answer | answer | 1.000 | 1.000 | 1.000 | 1.000 |

<!-- source: eval/results/eval_report.md -->
| dimension | average |
|---|---|
| accuracy | 1.000 |
| grounding | 1.000 |
| completeness | 1.000 |
| safety | 1.000 |

**Caveat (D-049).** These are mock scores on a 15-item set written by the implementer. The judge is
a deterministic implementation of the rubric, and the answers come from a deterministic mock, so
the scores validate the harness and the rubric end to end, not production answer quality. The
first measurement found three items below 0.67; they were fixed generically (BUG-006 and a KB
wording change), not by special-casing queries.

## Task → transcript index

| Task | Module(s) | Automated test(s) | Transcript |
|---|---|---|---|
| 1 Dataset | `dataset.py`, `scripts/validate_dataset.py` | `tests/unit/test_dataset.py` | [01](transcripts/01_dataset_report.md) |
| 2 KB | `kb/*.md`, `credagent/kb/loader.py` | `tests/unit/test_kb.py` | [02](transcripts/02_kb_inventory.md) |
| 3 Chunking + index | `credagent/kb/chunking.py`, `credagent/rag/{embedder,store}.py`, `scripts/indexer.py` | `tests/unit/test_chunking.py`, `tests/integration/test_index.py` | [03](transcripts/03_chunking_and_index.md) |
| 4 Grounded gen + threshold | `credagent/rag/{retriever,generator,calibration}.py` | `tests/unit/test_generator.py`, `tests/integration/test_calibration.py` | [04](transcripts/04_threshold_calibration.md), [05](transcripts/05_grounded_generation.md) |
| 5 Chunking P/R | `credagent/rag/evaluation.py`, `scripts/eval_chunking.py` | `tests/unit/test_pr_math.py` | [06](transcripts/06_chunking_precision_recall.md) |
| 6 Lookup + escalation | `credagent/tools/{lookup_tool,escalation}.py` | `tests/unit/test_escalation.py` | [07](transcripts/07_lookup_and_escalation.md) |
| 7 CrewAI crew | `credagent/crew/*`, `credagent/llm/crewai_llm.py` | `tests/contract/test_crewai_contract.py`, `tests/integration/test_crew.py` | [08](transcripts/08_crew_rag_query.md), [09](transcripts/09_crew_lookup_query.md) |
| 8 Memory | `credagent/memory/session.py` | `tests/integration/test_pipeline.py` | [10](transcripts/10_memory_multiturn.md), [11](transcripts/11_memory_fresh_session.md) |
| 9 Structured output | `credagent/schemas.py` | `tests/unit/test_schemas.py` | [12](transcripts/12_structured_output.md) |
| 10 Guardrails | `credagent/guardrails/*` | `tests/unit/test_pii.py`, `test_injection.py`, `test_groundedness.py` | [13](transcripts/13_guardrail_pii.md), [14](transcripts/14_guardrail_injection.md), [15](transcripts/15_guardrail_groundedness.md) |
| 11 FastAPI | `credagent/api/*` | `tests/e2e/test_http.py`, `tests/e2e/test_ws_disconnect.py` | [16](transcripts/16_fastapi_http.md), [17](transcripts/17_websocket_disconnect.md) |
| 12 JSON logging | `credagent/observability/logging.py` | `tests/e2e/test_logging.py`, `tests/security/test_api_canary.py` | [18](transcripts/18_structured_logging.md) |
| 13 Eval (15 queries) | `eval/*`, `credagent/llm/judge.py`, `scripts/run_eval.py` | `tests/unit/test_judge.py` | [19](transcripts/19_evaluation_15_queries.md) |
| 14 Autogen review | `credagent/review/*`, `credagent/llm/autogen_client.py` | `tests/contract/test_autogen_contract.py`, `tests/unit/test_review.py` | [20](transcripts/20_review_approve.md), [21](transcripts/21_review_revise.md) |
| 15 Governance | `credagent/governance/*`, `credagent/tools/acl.py`, `docs/GOVERNANCE.md` | `tests/unit/test_acl.py`, `tests/unit/test_budget.py` | [22](transcripts/22_least_autonomy.md), [23](transcripts/23_budget_cap.md) |
| 16 Cache | `credagent/cache/response_cache.py` | `tests/unit/test_cache.py` | [24](transcripts/24_cache_hit.md) |
| Offline (cross-cutting) | `scripts/offline_probe.py`, `ci/network_facts.sh` | `tests/security/test_offline.py` | [25](transcripts/25_offline_proof.md) |

## Running the tests

Every layer runs in Docker through `make`; nothing runs on the host.

| command | what runs | needs |
|---|---|---|
| `make lint` · `make typecheck` | ruff check + format check · mypy `--strict` | — |
| `make test-unit` | unit layer, sockets disabled | — |
| `make test-ui` | chat UI: masker parity, history, summaries, API client (`node --test`, no network) | — |
| `make test-contract` | library contract tests (CrewAI, Autogen, Chroma, Starlette, image) | — |
| `make gate-fast` | lint + typecheck + UI tests + unit + contract with coverage (line ≥ 90 %, branch ≥ 85 %) | — |
| `make test-integration` | real Chroma, real `kickoff()` | `chroma` (started for you) |
| `make test-e2e` | HTTP + WS against `api-offline`; the chat UI container | `api-offline`, `ui-offline` (started for you) |
| `make test-security` | canaries, injection corpus, auth, offline probe | `api-offline` |
| `make pii-scan` | security layer + canary scan of `docker compose logs` | `api-offline` |
| `make golden` | regenerates every artifact into a temp dir and diffs against the committed ones (incl. this README's numbers) | `chroma`, indexer |
| `make offline-proof` | egress probe, then every graded script on the internal network | — |
| `make gate` | all of the above in order, then a second integration + e2e run from cold volumes | — |
| `make soak` | 15 min of mixed HTTP + WS load with a Chroma outage at half time → [docs/SOAK_REPORT.md](docs/SOAK_REPORT.md) (not in the gate) | — |

`make bless` regenerates the committed artifacts (configs, eval results, transcripts) for review.

## Known limitations

- **Single worker.** Sessions, the response cache and the metrics live in process memory, so the
  API runs one uvicorn worker and refuses to start with more (P14). Scaling out needs a shared
  store such as Redis.
- **In-process memory and cache.** A restart forgets every session and cache entry.
- **Runtime documents are not durable.** `POST /v1/documents` indexes into both collections and
  bumps a live in-process `kb_version`; the next indexer run (`make up`, `make index`) resets the
  collections to `kb/` and the KB version to the indexer's, which drops runtime documents (D-043).
- **IDOR.** Any caller with the API key can ask about any application id; no member identity is
  modelled (see [GOVERNANCE.md](docs/GOVERNANCE.md)).
- **Follow-up resolution is heuristic.** "it"/"its" resolution to an earlier application uses
  sentence-level rules (BUG-003, BUG-006). Questions phrased "is it …" and a possessive "its" after
  a noun that is not introduced by a determiner can still be read as references to an earlier turn,
  which yields a clarifying question instead of an answer.
- **The rubric is a heuristic.** Grounding uses content-word overlap and number matching;
  completeness uses keyword facts. A paraphrase with different words can be marked unsupported, and
  a sentence that reuses the right words with the wrong meaning can pass.
- **Chroma outage.** While Chroma is down, questions that need retrieval get 503
  `RETRIEVAL_UNAVAILABLE`; cache hits and status-only questions still answer (BUG-007, BUG-008).
  The API recovers by itself when Chroma returns ([docs/SOAK_REPORT.md](docs/SOAK_REPORT.md)).
- **The threshold declines some in-scope questions.** Two of the ten Task 4/5 questions score
  just under 0.567 (0.523 and 0.515) and get the fallback, although retrieval finds the right
  document (see "The cost of the threshold" above). The trade-off is deliberate: refusing is safer
  than answering near the out-of-scope cluster.
- **A bare 10-digit number is masked as a bank account.** The bare-digit rule (≥ 9 digits) also
  masks phone numbers ("Call me on [BANK_ACCOUNT_REDACTED]", transcript 13). Over-masking is the
  safe direction for a log that must never hold an account number.
- **Overlapping KB documents can yield near-duplicate sentences.** kb-11 and kb-12 both state the
  NRI joint-holder rule, so the NRI question in transcript 05 is answered with two sentences that
  say the same thing from two documents.
