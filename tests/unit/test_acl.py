"""Least autonomy (§ARCH 7.14.1): the policy table, the wiring-time guard and the crew audit.

The crew does not exist until Phase 6, so `audit_crew` is exercised on duck-typed stand-ins with
the attributes it reads from a CrewAI `Crew` (`agents[].role/tools/allow_delegation`,
`tasks[].agent/tools`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import pytest

import dataset
from credagent.context import ToolCallLedger
from credagent.governance.policy import AGENT_ROLES, TOOL_ACL
from credagent.tools import escalation as esc
from credagent.tools.acl import LeastAutonomyViolation, assert_allowed, audit_crew
from credagent.tools.lookup_tool import LoanStatusTool

AGENTS = ("retrieval_agent", "lookup_agent", "composer_agent", "admin_agent", "")
TOOLS = ("rag_lookup", "check_loan_application_status", "shell", "")
ALLOWED = {("retrieval_agent", "rag_lookup"), ("lookup_agent", "check_loan_application_status")}


def test_policy_table_is_the_architecture_table() -> None:
    assert dict(TOOL_ACL) == {
        "check_loan_application_status": frozenset({"lookup_agent"}),
        "rag_lookup": frozenset({"retrieval_agent"}),
    }


def test_policy_table_is_immutable() -> None:
    with pytest.raises(TypeError):
        TOOL_ACL["check_loan_application_status"] = frozenset({"composer_agent"})  # type: ignore[index]


def test_agent_roles_are_the_architecture_roles() -> None:
    assert dict(AGENT_ROLES) == {
        "Policy Retrieval Specialist": "retrieval_agent",
        "Loan Application Lookup Specialist": "lookup_agent",
        "Response Composer": "composer_agent",
    }


@pytest.mark.parametrize(("agent", "tool"), list(itertools.product(AGENTS, TOOLS)))
def test_allow_deny_matrix(agent: str, tool: str) -> None:
    if (agent, tool) in ALLOWED:
        assert_allowed(agent, tool)
    else:
        with pytest.raises(LeastAutonomyViolation):
            assert_allowed(agent, tool)


@dataclass
class NamedTool:
    name: str
    owner: str | None = None


def test_tool_objects_are_checked_by_name() -> None:
    assert_allowed("retrieval_agent", NamedTool("rag_lookup"))
    with pytest.raises(LeastAutonomyViolation, match="composer_agent.*rag_lookup"):
        assert_allowed("composer_agent", NamedTool("rag_lookup"))


def test_an_owner_bound_tool_must_be_held_by_its_owner() -> None:
    """A lookup tool bound to someone else cannot be wired to the Lookup Agent either."""
    with pytest.raises(LeastAutonomyViolation, match="owner"):
        assert_allowed("lookup_agent", NamedTool("check_loan_application_status", "intruder"))
    assert_allowed("lookup_agent", NamedTool("check_loan_application_status", "lookup_agent"))


def test_violation_is_a_permission_error() -> None:
    assert issubclass(LeastAutonomyViolation, PermissionError)


# ------------------------------------------------------------------ audit_crew
@dataclass
class FakeAgent:
    role: str
    tools: list[Any] = field(default_factory=list)
    allow_delegation: bool = False


@dataclass
class FakeTask:
    agent: FakeAgent | None
    tools: list[Any] | None = None


@dataclass
class FakeCrew:
    agents: list[FakeAgent]
    tasks: list[FakeTask]


def lookup_tool() -> LoanStatusTool:
    return LoanStatusTool(
        owner="lookup_agent", ledger=ToolCallLedger(),
        config=esc.build_config(dataset.LOAN_APPLICATIONS),
    )  # fmt: skip


def normal_crew() -> FakeCrew:
    retrieval = FakeAgent(
        "Policy Retrieval Specialist", [NamedTool("rag_lookup", "retrieval_agent")]
    )
    lookup = FakeAgent("Loan Application Lookup Specialist", [lookup_tool()])
    composer = FakeAgent("Response Composer")
    return FakeCrew(
        [retrieval, lookup, composer],
        [FakeTask(retrieval), FakeTask(lookup), FakeTask(composer, tools=[])],
    )


def test_audit_passes_on_the_normal_crew_and_returns_the_tool_map() -> None:
    assert audit_crew(normal_crew()) == {
        "retrieval_agent": ("rag_lookup",),
        "lookup_agent": ("check_loan_application_status",),
        "composer_agent": (),
    }


def test_audit_catches_a_tool_smuggled_onto_the_composer() -> None:
    crew = normal_crew()
    crew.agents[2].tools.append(lookup_tool())
    with pytest.raises(LeastAutonomyViolation, match="composer_agent"):
        audit_crew(crew)


def test_audit_catches_a_task_tools_override() -> None:
    crew = normal_crew()
    crew.tasks[2].tools = [lookup_tool()]
    with pytest.raises(LeastAutonomyViolation, match="task"):
        audit_crew(crew)


def test_audit_catches_a_tool_on_a_task_without_an_agent() -> None:
    crew = normal_crew()
    crew.tasks.append(FakeTask(None, tools=[lookup_tool()]))
    with pytest.raises(LeastAutonomyViolation, match="no agent"):
        audit_crew(crew)


def test_audit_allows_a_toolless_task_without_an_agent() -> None:
    crew = normal_crew()
    crew.tasks.append(FakeTask(None))
    audit_crew(crew)


def test_audit_rejects_an_unknown_role() -> None:
    crew = normal_crew()
    crew.agents.append(FakeAgent("Helpful Generalist"))
    with pytest.raises(LeastAutonomyViolation, match="Helpful Generalist"):
        audit_crew(crew)


def test_audit_rejects_delegation() -> None:
    crew = normal_crew()
    crew.agents[2].allow_delegation = True
    with pytest.raises(LeastAutonomyViolation, match="delegation"):
        audit_crew(crew)


def test_audit_catches_an_owner_tampered_tool_on_the_right_agent() -> None:
    crew = normal_crew()
    object.__setattr__(crew.agents[1].tools[0], "owner", "composer_agent")
    with pytest.raises(LeastAutonomyViolation, match="owner"):
        audit_crew(crew)
