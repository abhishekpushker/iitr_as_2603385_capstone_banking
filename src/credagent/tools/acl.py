"""Least-autonomy guards (§ARCH 7.14.1) over the policy table in `governance/policy.py`.

* `assert_allowed(agent_key, tool)`: wiring-time guard (and the tools' runtime re-check). The
  tool's name must list `agent_key` in `TOOL_ACL`, and an owner-bound tool must be owned by that
  same agent. Anything not explicitly allowed is denied.
* `audit_crew(crew)`: walks every agent **and every task's `tools` override**, fails on any
  disallowed tool, unknown role or `allow_delegation=True`, and returns the tool map.

It reads only `role`, `tools`, `allow_delegation` (agents) and `agent`, `tools` (tasks), so it
needs no CrewAI import.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from credagent.governance.policy import AGENT_ROLES, TOOL_ACL


class LeastAutonomyViolation(PermissionError):
    """An agent holds, or a tool runs for, an agent the policy does not allow."""


def _tool_name(tool: Any) -> str:
    return tool if isinstance(tool, str) else str(getattr(tool, "name", tool))


def assert_allowed(agent_key: str, tool: Any) -> None:
    name = _tool_name(tool)
    if agent_key not in TOOL_ACL.get(name, frozenset()):
        raise LeastAutonomyViolation(f"agent {agent_key!r} may not hold tool {name!r}")
    owner = getattr(tool, "owner", None)
    if owner is not None and owner != agent_key:
        raise LeastAutonomyViolation(
            f"tool {name!r} is bound to owner {owner!r}, not to agent {agent_key!r}"
        )


def _agent_key(agent: Any) -> str:
    role = getattr(agent, "role", None)
    key = AGENT_ROLES.get(role) if isinstance(role, str) else None
    if key is None:
        raise LeastAutonomyViolation(f"unknown agent role {role!r}: not in the policy")
    return key


def _check_all(agent_key: str, tools: Iterable[Any] | None, where: str) -> tuple[str, ...]:
    names = []
    for tool in tools or ():
        try:
            assert_allowed(agent_key, tool)
        except LeastAutonomyViolation as exc:
            raise LeastAutonomyViolation(f"{where}: {exc}") from None
        names.append(_tool_name(tool))
    return tuple(names)


def audit_crew(crew: Any) -> dict[str, tuple[str, ...]]:
    """Fail closed on any least-autonomy breach; return `{agent_key: tool names}`."""
    tool_map: dict[str, tuple[str, ...]] = {}
    for agent in crew.agents:
        key = _agent_key(agent)
        if getattr(agent, "allow_delegation", True) is not False:
            raise LeastAutonomyViolation(f"agent {key!r} allows delegation")
        tool_map[key] = _check_all(key, agent.tools, f"agent {key!r}")
    for index, task in enumerate(crew.tasks):
        tools = list(getattr(task, "tools", None) or ())
        if task.agent is None:
            if tools:
                raise LeastAutonomyViolation(f"task {index} has tools but no agent")
            continue
        key = _agent_key(task.agent)
        _check_all(key, tools, f"task {index} tools override for agent {key!r}")
    return tool_map
