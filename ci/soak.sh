#!/usr/bin/env bash
# Soak + fault injection (docs/IMPLEMENTATION_PLAN.md Phase 12, verification 2–3). Not in the gate.
# Usage: ci/soak.sh   (SOAK_DURATION_S=900, SOAK_FAULT_HOLD_S=60; the fault starts at half time)
#
# A fresh api-offline serves the load (so RSS starts cold and no earlier cache entries exist).
# The host samples the API's VmRSS every ~10 s and stops/starts chroma mid-run, writing both to
# artifacts/soak/, which the tests container reads (repo bind mount) when it builds the report.
# The report goes to docs/SOAK_REPORT.md; the exit code is the soak's pass/fail.
set -euo pipefail
cd "$(dirname "$0")/.."
duration=${SOAK_DURATION_S:-900}
hold=${SOAK_FAULT_HOLD_S:-60}
dir=artifacts/soak
rm -rf "$dir"; mkdir -p "$dir"

docker compose --profile test up -d --wait api-offline
docker compose --profile test up -d --wait --force-recreate --no-deps api-offline
api=$(docker compose --profile test ps -q api-offline)

# ---- in-container probe: prints "<status> <error_code|->" for GET /readyz or POST /v1/ask
probe() {
  docker compose exec -T api-offline python -c '
import json, os, sys, urllib.error, urllib.request
path, msg = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
req = urllib.request.Request("http://127.0.0.1:8000" + path,
    data=None if msg is None else json.dumps({"message": msg}).encode(),
    headers={"X-API-Key": os.environ.get("API_KEY", "dev-api-key-not-a-secret"),
             "Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=60); print(r.status, "-")
except urllib.error.HTTPError as e:
    print(e.code, json.loads(e.read() or b"{}").get("error_code", "-"))
' "$@"
}

sample_rss() {
  while :; do
    kb=$(docker compose exec -T api-offline sh -c "grep VmRSS /proc/1/status" | awk '{print $2}')
    echo "$(date +%s) ${kb:-0}" >> "$dir/rss.tsv"
    sleep 10
  done
}

inject_fault() {
  sleep $((duration / 2))  # the fault is scheduled at half time, by design
  local before t_stop rd ask running t_start t_ready after ask_after after_started restarts
  before=$(docker inspect -f '{{.State.StartedAt}}' "$api")
  docker compose stop chroma >&2
  t_stop=$(date +%s)
  rd=$(probe /readyz)
  ask=$(probe /v1/ask "Is there a fee for closing a savings account in the first year?")
  running=$(docker inspect -f '{{.State.Running}}' "$api")
  sleep "$hold"
  docker compose start chroma >&2
  t_start=$(date +%s)
  after="503 -"
  for _ in $(seq 1 120); do  # bounded readiness wait (≤ 120 probes)
    after=$(probe /readyz)
    [[ $after == 200* ]] && break
    sleep 1
  done
  t_ready=$(date +%s)
  ask_after=$(probe /v1/ask "Is there a fee for closing a savings account in the first year?")
  after_started=$(docker inspect -f '{{.State.StartedAt}}' "$api")
  restarts=$(docker inspect -f '{{.RestartCount}}' "$api")
  restarted=false
  [[ $before != "$after_started" || $restarts != 0 ]] && restarted=true
  cat > "$dir/fault.json" <<EOF
{"t_stop": $t_stop, "t_start": $t_start, "t_ready": $t_ready,
 "readyz_during": ${rd%% *}, "ask_during": ${ask%% *}, "ask_during_code": "${ask#* }",
 "api_running_during": $running, "readyz_after": ${after%% *}, "ask_after": ${ask_after%% *},
 "api_restarted": $restarted}
EOF
}

sample_rss & sampler=$!
inject_fault & injector=$!
trap 'kill $sampler $injector 2>/dev/null || true; docker compose start chroma >/dev/null 2>&1 || true' EXIT

rc=0
docker compose --profile test run --rm --no-deps -T \
  -e GIT_SHA="$(git rev-parse HEAD)$(git diff --quiet HEAD -- || echo +dirty)" tests \
  python -m tests.soak.soak --duration "$duration" > "$dir/report.md" || rc=$?
cp "$dir/report.md" docs/SOAK_REPORT.md
echo "soak: exit $rc — report in docs/SOAK_REPORT.md"
exit $rc
