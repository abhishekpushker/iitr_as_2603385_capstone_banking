"""Agent construction with the wiring-time least-autonomy guard (§ARCH 7.9, 7.14.1 guard 2).

`make_agent(agent_key, tools, llm=…)` checks every tool with `acl.assert_allowed` before the
Agent exists, then builds it with the overrides §ARCH 2.2.3–5 requires: `cache=False` (P4: the
default `True` would hide repeated tool calls), `allow_delegation=False`, `max_iter=4`,
`max_retry_limit=1`, `verbose=False`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from crewai import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool

from credagent.crew import templates
from credagent.llm.crewai_llm import Persona
from credagent.tools.acl import LeastAutonomyViolation, assert_allowed

PERSONAS: Final[dict[str, Persona]] = {
    "retrieval_agent": "retrieval",
    "lookup_agent": "lookup",
    "composer_agent": "composer",
}
MAX_ITER: Final = 4
MAX_RETRY_LIMIT: Final = 1


def make_agent(agent_key: str, tools: Sequence[BaseTool], *, llm: BaseLLM) -> Agent:
    if agent_key not in templates.ROLES:
        raise LeastAutonomyViolation(f"unknown agent {agent_key!r}: not in the policy")
    for tool in tools:
        assert_allowed(agent_key, tool)
    return Agent(
        role=templates.ROLES[agent_key],
        goal=templates.GOALS[agent_key],
        backstory=templates.BACKSTORIES[agent_key],
        tools=list(tools),
        llm=llm,
        cache=False,
        allow_delegation=False,
        max_iter=MAX_ITER,
        max_retry_limit=MAX_RETRY_LIMIT,
        verbose=False,
    )
