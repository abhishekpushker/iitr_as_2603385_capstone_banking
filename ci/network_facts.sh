#!/usr/bin/env bash
# Network membership from `docker inspect` (§ARCH 11.7, transcript 25), printed as one JSON line.
# A container cannot inspect itself, so the host starts a throwaway container of each service that
# runs graded code, reads the networks it is attached to and whether each is `internal`, and
# removes it. Callers pass the line in as NETWORK_FACTS (compose.yaml: runner, tests).
set -euo pipefail
cd "$(dirname "$0")/.."

members() {  # <service> <profile> → {"service": [{"name": …, "internal": …}, …]}
  local id names out="" sep=""
  id=$(docker compose --profile "$2" run -d --no-deps "$1" sleep 60)
  names=$(docker inspect --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$id")
  docker rm -f "$id" >/dev/null
  for net in $names; do
    out+="${sep}{\"name\": \"${net}\", \"internal\": $(docker network inspect --format '{{.Internal}}' "$net")}"
    sep=", "
  done
  printf '"%s": [%s]' "$1" "$out"
}

printf '{%s, %s, %s}\n' "$(members runner tools)" "$(members tests test)" "$(members api-offline test)"
