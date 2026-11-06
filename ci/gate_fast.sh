#!/usr/bin/env bash
# lint + typecheck + unit + contract (with coverage) — §PLAN 3 steps [2]–[3] (+ [9] threshold).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== lint";      make --no-print-directory lint
echo "== typecheck"; make --no-print-directory typecheck
echo "== unit + contract (random order, sockets off, coverage)"
make --no-print-directory test-unit-contract
