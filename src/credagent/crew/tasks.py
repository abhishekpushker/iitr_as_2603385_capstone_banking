"""The three sequential tasks (§ARCH 7.9). The Composer's task takes both earlier tasks as
`context` and parses its Final Answer into `CrewResponse` (`output_pydantic`)."""

from __future__ import annotations

from crewai import Agent, Task

from credagent.crew import templates
from credagent.schemas import CrewResponse


def make_tasks(
    retrieval: Agent, lookup: Agent, composer: Agent, *, query: str, record_id: str | None
) -> tuple[Task, Task, Task]:
    retrieval_task = Task(
        description=templates.retrieval_description(query, record_id),
        expected_output=templates.RETRIEVAL_EXPECTED,
        agent=retrieval,
    )
    lookup_task = Task(
        description=templates.lookup_description(query, record_id),
        expected_output=templates.LOOKUP_EXPECTED,
        agent=lookup,
    )
    compose_task = Task(
        description=templates.composer_description(query, record_id),
        expected_output=templates.COMPOSER_EXPECTED,
        agent=composer,
        context=[retrieval_task, lookup_task],
        output_pydantic=CrewResponse,
    )
    return retrieval_task, lookup_task, compose_task
