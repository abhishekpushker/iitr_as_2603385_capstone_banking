#!/usr/bin/env bash
# Print the two-digit number of the phase being worked on: latest phase-NN-green tag + 1.
set -euo pipefail
last=$(git tag --list 'phase-*-green' | sed -E 's/phase-([0-9]+)-green/\1/' | sort -n | tail -1)
if [[ -z "$last" ]]; then echo "00"; else printf '%02d\n' $((10#$last + 1)); fi
