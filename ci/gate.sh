#!/usr/bin/env bash
# The full regression gate (docs/IMPLEMENTATION_PLAN.md §3). Stops at the first failure.
# Usage: ci/gate.sh <phase>   — output is also saved to artifacts/gate-<phase>.log
set -euo pipefail
cd "$(dirname "$0")/.."
phase=${1:?phase number}
mkdir -p artifacts
log="artifacts/gate-${phase}.log"
exec > >(tee "$log") 2>&1

step() { printf '\n==================== [%s] %s (%s)\n' "$1" "$2" "$(date -u +%H:%M:%SZ)"; }
trap 'rc=$?; echo; echo "GATE phase ${phase}: FAILED (exit ${rc})"; exit $rc' ERR

echo "GATE phase ${phase} — $(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty') — $(date -u +%FT%TZ)"
step 1  "docker compose build";                 make --no-print-directory build
step 2-3 "lint · typecheck · unit+contract";     ci/gate_fast.sh
step 4  "integration";                          ci/run_layer.sh integration chroma
step 5  "e2e";                                  ci/run_layer.sh e2e api-offline
step 6  "security";                             ci/run_layer.sh security
step 7  "golden";                               ci/run_layer.sh golden chroma indexer
step 8  "offline proof";                        make --no-print-directory offline-proof
step 9  "coverage threshold";                   echo "enforced in step 3 (line ≥ 90 %, branch ≥ 85 %: scripts/coverage_gate.py)"
step 10 "cold volume: down -v, rerun [4]–[5]"
make --no-print-directory clean-volumes
ci/run_layer.sh integration chroma
ci/run_layer.sh e2e api-offline
make --no-print-directory down
trap - ERR
echo; echo "GATE phase ${phase}: GREEN"
