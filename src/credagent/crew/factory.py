"""`build_crew` (§ARCH 7.9): a fresh Crew per request, so no state crosses requests.

Per request: owner-bound tools recording into `ctx.ledger`, one `MockCrewLLM` per agent bound to
`ctx`, three agents (via the ACL-checked `make_agent`), three sequential tasks, and a Crew with
`cache`, `memory` and `planning` off. `audit_crew` runs on the built crew before it is returned
(§ARCH 7.14.1 guard 3). The record id is read from the (masked) query.

`run_crew` kicks off without `inputs` (so CrewAI never interpolates the masked text) and
returns the non-streaming `CrewOutput`. `parse_crew_output` is §ARCH 6 step 6: `result.raw`
must validate as `CrewResponse` and equal `result.pydantic`; anything else is a
`SchemaViolation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crewai import Crew, Process
from crewai.crews.crew_output import CrewOutput
from crewai.tools import BaseTool
from pydantic import ValidationError

from credagent.cache.response_cache import ResponseCache
from credagent.context import RequestContext
from credagent.crew.agents import PERSONAS, make_agent
from credagent.crew.tasks import make_tasks
from credagent.llm.crewai_llm import MockCrewLLM
from credagent.llm.engine import MockEngine
from credagent.observability.metrics import Metrics
from credagent.rag.calibration import RetrievalConfig
from credagent.rag.retriever import SupportsRetrieve
from credagent.schemas import CrewResponse, GeneratedAnswer
from credagent.text import find_record_id
from credagent.tools.acl import audit_crew
from credagent.tools.escalation import EscalationConfig
from credagent.tools.lookup_tool import LoanStatusTool
from credagent.tools.rag_tool import rag_tool_for_deployment


class SchemaViolation(ValueError):
    """The crew's raw output is not a valid `CrewResponse`, or differs from `result.pydantic`."""


@dataclass(frozen=True)
class CrewDeps:
    """Process-wide collaborators; everything per-request comes from the `RequestContext`."""

    retrieval_config: RetrievalConfig
    retriever: SupportsRetrieve
    cache: ResponseCache[GeneratedAnswer]
    metrics: Metrics
    engine: MockEngine = field(default_factory=MockEngine)
    escalation_config: EscalationConfig | None = None


def build_crew(ctx: RequestContext, query: str, deps: CrewDeps) -> Crew:
    rag = rag_tool_for_deployment(
        deps.retrieval_config, retriever=deps.retriever, client=deps.engine, cache=deps.cache,
        ledger=ctx.ledger, metrics=deps.metrics,
    )  # fmt: skip
    lookup = LoanStatusTool(owner="lookup_agent", ledger=ctx.ledger, config=deps.escalation_config)
    tools: dict[str, list[BaseTool]] = {
        "retrieval_agent": [rag], "lookup_agent": [lookup], "composer_agent": [],
    }  # fmt: skip
    agents = {
        key: make_agent(
            key, tools[key],
            llm=MockCrewLLM(
                persona=persona, tools=tools[key], ctx=ctx, engine=deps.engine,
                response_format=CrewResponse if persona == "composer" else None,
            ),
        )
        for key, persona in PERSONAS.items()
    }  # fmt: skip
    tasks = make_tasks(
        agents["retrieval_agent"], agents["lookup_agent"], agents["composer_agent"],
        query=query, record_id=find_record_id(query),
    )  # fmt: skip
    crew = Crew(
        agents=list(agents.values()),
        tasks=list(tasks),
        process=Process.sequential,
        cache=False,
        memory=False,
        planning=False,
        verbose=False,
    )
    audit_crew(crew)
    return crew


def run_crew(crew: Crew) -> CrewOutput:
    result = crew.kickoff()
    if not isinstance(result, CrewOutput):
        raise TypeError("expected a CrewOutput: crew streaming must stay off")
    return result


def parse_crew_output(result: Any) -> CrewResponse:
    try:
        response = CrewResponse.model_validate_json(result.raw)
    except ValidationError as exc:
        raise SchemaViolation("crew output is not a valid CrewResponse") from exc
    if result.pydantic != response:
        raise SchemaViolation("result.pydantic differs from the validated raw output")
    return response
