#!/usr/bin/env bash
# PreToolUse(Bash) hook: CLAUDE.md "Docker only" — reject host commands that start with python,
# python3, pip or pip3 (any version suffix or path). Heredoc bodies and quoted strings are data,
# so they are removed first; then every segment of the compound command (`;`, `&&`, `||`, `|`,
# newline) is checked, after leading `VAR=value` assignments and `env`, `command`, `exec`, `sudo`,
# `time` and `nohup` prefixes. docker/make commands pass, including a `python` that runs inside a
# container (`docker compose run … python -m …`, `make …`).
# Exit 2 blocks the call and shows stderr to Claude; exit 0 allows it.
set -uo pipefail

cmd=$(jq -r '.tool_input.command // empty')
[[ -z $cmd ]] && exit 0

# Drop heredoc bodies (<<EOF, <<'EOF', <<-"EOF" … up to the terminator line).
code="" terminator=""
while IFS= read -r line || [[ -n $line ]]; do
  if [[ -n $terminator ]]; then
    [[ ${line#"${line%%[![:space:]]*}"} == "$terminator" ]] && terminator=""
    continue
  fi
  code+="$line"$'\n'
  if [[ $line =~ \<\<-?[[:space:]]*[\'\"]?([A-Za-z_][A-Za-z0-9_]*)[\'\"]? ]]; then
    terminator=${BASH_REMATCH[1]}
  fi
done <<<"$cmd"

deny_re='^(python|pip)[0-9.]*$'
while IFS= read -r segment; do
  read -ra words <<<"$segment"
  for word in "${words[@]+"${words[@]}"}"; do
    word=${word#[({]}
    case $word in
      *=*|env|command|exec|sudo|time|nohup|'') continue ;;
    esac
    if [[ ${word##*/} =~ $deny_re ]]; then
      echo "Blocked by .claude/hooks/deny-host-python.sh: '${word}' on the host. CLAUDE.md: Docker only — run Python through make targets (make gate-fast, make test-unit, …) or docker compose." >&2
      exit 2
    fi
    break
  done
done < <(printf '%s' "$code" | sed -E "s/'[^']*'//g; s/\"[^\"]*\"//g; s/(\&\&|\|\||;|\|)/\n/g")
exit 0
