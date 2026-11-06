#!/usr/bin/env bash
# Run one pytest layer in the `tests` container: ci/run_layer.sh <layer> [service ...]
# Services are started (and waited for) first. A layer is "not started yet" only when NO test
# file carries it (no tests/<layer>/test_*.py and no explicit pytest.mark.<layer> anywhere);
# that prints N/A and exits 0. Otherwise pytest's exit code is returned as-is — including 5
# ("no tests collected"), which then fails the gate.
set -euo pipefail
cd "$(dirname "$0")/.."
layer=$1; shift
services=("$@")

if ! compgen -G "tests/${layer}/test_*.py" >/dev/null \
   && ! grep -rqs --include='test_*.py' "pytest.mark.${layer}\b" tests; then
  echo "layer ${layer}: N/A — no tests in this layer yet"
  exit 0
fi

if ((${#services[@]})); then
  docker compose --profile test up -d --wait "${services[@]}"
fi
docker compose --profile test run --rm --no-deps -T tests pytest -m "${layer}"
