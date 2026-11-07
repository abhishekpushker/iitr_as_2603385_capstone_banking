#!/usr/bin/env bash
# PII canary scan over every sink (§ARCH 11.6). The security layer sends the canaries through
# every endpoint of api-offline and scans the log volume, both Chroma collections, transcripts/
# and (in-process) session histories and cache keys. Container stdout is only visible from the
# host, so `docker compose logs` is scanned here afterwards. Any raw canary fails.
set -euo pipefail
cd "$(dirname "$0")/.."
canaries=("ABCPE1234F" "2345 6789 0123" "234567890123" "123456789012345")  # = tests/e2e/conftest.py

ci/run_layer.sh security api-offline

logs=$(docker compose --profile test --profile tools logs --no-color api-offline api indexer)
lines=$(printf '%s\n' "$logs" | wc -l | tr -d ' ')
leaked=0
for c in "${canaries[@]}"; do
  if printf '%s\n' "$logs" | grep -qF -- "$c"; then
    echo "pii-scan: raw canary '$c' in docker compose logs"; leaked=1
  fi
done
((leaked == 0)) || exit 1
echo "pii-scan: docker compose logs clean (${lines} lines, ${#canaries[@]} canaries)"
