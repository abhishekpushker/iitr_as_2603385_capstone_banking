# The ONLY entry point (docs/IMPLEMENTATION_PLAN.md §2). Everything runs in Docker.
# GNU Make 3.81 compatible (macOS default): no .SHELLFLAGS, so multi-step logic lives in ci/*.sh.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON_IMAGE := python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
UV_VERSION   := 0.13.7

HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)
export HOST_UID HOST_GID

COMPOSE := docker compose
# The tests container: dev image, backend (internal) network only, repo bind-mounted read-only.
RUN_TESTS := $(COMPOSE) --profile test run --rm --no-deps -T tests
RUN_TOOLS := $(COMPOSE) --profile tools run --rm --no-deps -T runner

# pytest-socket: unit + contract layers may not open network sockets (§ARCH 11.7).
SOCKET_OFF := --disable-socket --allow-unix-socket --allow-hosts=chroma,api-offline,127.0.0.1

PHASE ?= $(shell ci/next_phase.sh)

.PHONY: help init lock build up down logs index lint format typecheck \
        test-unit test-contract test-unit-contract test-integration test-e2e test-security \
        golden transcripts eval calibrate eval-chunking escalation-config budget bless pii-scan \
        offline-proof \
        gate-fast gate demo clean clean-volumes capture-crewai-fixtures image-size

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-24s %s\n",$$1,$$2}'

init: ## create .env from .env.example and the git-ignored work dirs
	@test -f .env || cp .env.example .env
	@mkdir -p artifacts transcripts eval/results tests/fixtures/crewai_messages

lock: ## compile both hash-locked lockfiles (networked builder container)
	docker run --rm --user $(HOST_UID):$(HOST_GID) -e HOME=/tmp -e UV_CACHE_DIR=/tmp/uv-cache \
	  -v "$(CURDIR)":/work -w /work $(PYTHON_IMAGE) sh -euc '\
	    pip install --quiet --user uv==$(UV_VERSION); UV=/tmp/.local/bin/uv; \
	    $$UV pip compile requirements.in --generate-hashes --emit-index-url \
	      --index-strategy unsafe-best-match --python-version 3.12 \
	      --python-platform x86_64-manylinux_2_28 --custom-compile-command "make lock" \
	      -o requirements.lock; \
	    $$UV pip compile requirements-dev.in --generate-hashes --emit-index-url \
	      --index-strategy unsafe-best-match --python-version 3.12 \
	      --python-platform x86_64-manylinux_2_28 --custom-compile-command "make lock" \
	      -o requirements-dev.lock'

build: init ## build the runtime and dev images
	$(COMPOSE) --profile test --profile tools build

up: init ## start the default stack (chroma, indexer, api)
	$(COMPOSE) up -d --wait

down: ## stop the stack (keeps volumes)
	$(COMPOSE) --profile test --profile tools --profile offline down --remove-orphans

logs: ## follow logs
	$(COMPOSE) logs -f

index: build ## run the indexer one-shot (rebuilds first: the service runs the code baked into the image)
	$(COMPOSE) run --rm -T indexer

lint: ## ruff check + ruff format --check
	$(RUN_TESTS) sh -c 'ruff check . && ruff format --check .'

format: ## apply ruff format + safe fixes (writes to the repo; not part of the gate)
	docker run --rm --network none --user $(HOST_UID):$(HOST_GID) -e RUFF_CACHE_DIR=/tmp/.ruff_cache \
	  -v "$(CURDIR)":/app -w /app credagent:dev sh -c 'ruff format . && ruff check --fix .'

typecheck: ## mypy --strict
	$(RUN_TESTS) mypy

test-unit: ## unit layer (sockets disabled)
	$(RUN_TESTS) pytest -m unit $(SOCKET_OFF)

test-contract: ## contract layer (sockets disabled)
	$(RUN_TESTS) pytest -m contract $(SOCKET_OFF)

test-unit-contract: ## unit + contract with coverage; line ≥ 90 % and branch ≥ 85 % (gate steps 3 + 9)
	$(RUN_TESTS) sh -c 'pytest -m "unit or contract" $(SOCKET_OFF) --cov --cov-report=term-missing:skip-covered \
	  && coverage json -q -o /tmp/coverage.json && python -m scripts.coverage_gate /tmp/coverage.json'

test-integration: ## integration layer (chroma up)
	ci/run_layer.sh integration chroma

test-e2e: ## e2e layer (api-offline up)
	ci/run_layer.sh e2e api-offline

test-security: ## security layer (canary, injection corpus, auth, offline) against api-offline
	ci/run_layer.sh security api-offline

golden: ## regenerate deterministic artifacts into a temp dir and diff (from Phase 1)
	ci/run_layer.sh golden chroma indexer

pii-scan: ## PII canary scan over every sink and endpoint (§ARCH 11.6) + docker compose logs
	ci/pii_scan.sh

offline-proof: ## egress probe from the backend network (+ graded scripts as they appear)
	$(RUN_TESTS) python -m scripts.offline_probe
	$(RUN_TOOLS) python -m scripts.offline_probe
	$(COMPOSE) run --rm -T indexer
	$(RUN_TOOLS) python -m scripts.validate_dataset
	$(RUN_TOOLS) python -m scripts.build_escalation_config --out /tmp/escalation.json
	$(RUN_TOOLS) sh -c 'python -m scripts.calibrate_threshold --out /tmp/retrieval.json \
	  && python -m scripts.eval_chunking --config /tmp/retrieval.json --report /tmp/chunking_pr.md'
	$(RUN_TOOLS) python -m scripts.measure_budget --out /tmp/budget.json
	$(RUN_TOOLS) python -m scripts.make_transcripts --out /tmp/transcripts

image-size: ## print image sizes (recorded in docs/DECISIONS.md)
	@docker image ls --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep -E '^credagent:'

capture-crewai-fixtures: ## (re)record real CrewAI message lists into tests/fixtures/crewai_messages
	$(COMPOSE) --profile test run --rm --no-deps -T \
	  -v "$(CURDIR)/tests/fixtures/crewai_messages":/app/tests/fixtures/crewai_messages \
	  tests python -m tests.contract.crewai_recorder /app/tests/fixtures/crewai_messages

gate-fast: ## lint + typecheck + unit + contract (+ coverage)
	ci/gate_fast.sh

gate: ## the full regression gate (§PLAN 3); log in artifacts/gate-<phase>.log
	ci/gate.sh $(PHASE)

# The runner executes the code baked into credagent:runtime, so rebuild first (cached), and
# transcripts 03+ read the index back, so `index` (which depends on build) runs first.
# Provenance for the transcript header (§ARCH 7.19) is read after the build.
transcripts: index ## regenerate transcripts/ in the runner (backend network only)
	$(COMPOSE) --profile tools run --rm --no-deps -T \
	  -e GIT_SHA="$$(git rev-parse HEAD)$$(git diff --quiet HEAD -- || echo +dirty)" \
	  -e IMAGE_ID="$$(docker image inspect --format '{{.Id}}' credagent:runtime)" \
	  runner python -m scripts.make_transcripts

calibrate: index ## measure calibration similarities on both collections → config/retrieval.json
	$(RUN_TOOLS) python -m scripts.calibrate_threshold

escalation-config: build ## P80 of days_since_created → threshold T → config/escalation.json
	$(RUN_TOOLS) python -m scripts.build_escalation_config

# `calibrate` rewrites config/retrieval.json with deployed_strategy = null; `eval-chunking` then
# sets it from the P/R results (D-027), so bless runs them in this order, before the transcripts.
eval-chunking: index ## P/R of both chunking strategies → eval/results/chunking_pr.md + deployed_strategy
	$(RUN_TOOLS) python -m scripts.eval_chunking

# The pipeline (crew + review) on every eval/demo query, with an unlimited meter → config/budget.json.
# It runs after eval-chunking (the crew reads deployed_strategy) and before the transcripts (23).
budget: index ## measure PROMPT_OVERHEAD_TOKENS, MAX_OUTPUT_TOKENS, the budget cap → config/budget.json
	$(RUN_TOOLS) python -m scripts.measure_budget

bless: ## regenerate and overwrite every committed artifact (review the diff!)
	$(MAKE) --no-print-directory escalation-config
	$(MAKE) --no-print-directory calibrate
	$(MAKE) --no-print-directory eval-chunking
	$(MAKE) --no-print-directory budget
	$(MAKE) --no-print-directory transcripts
	@git status --short -- transcripts config eval/results

eval demo: ## generators (land in Phases 10–11)
	@echo "make $@: not implemented yet (see docs/IMPLEMENTATION_PLAN.md §4)"; exit 1

clean-volumes: ## down -v (fresh chroma/log volumes), keep images — the gate's cold rerun
	$(COMPOSE) --profile test --profile tools --profile offline down -v --remove-orphans

clean: ## down -v and remove project images
	$(COMPOSE) --profile test --profile tools --profile offline down -v --remove-orphans
	-docker image rm credagent:runtime credagent:dev
