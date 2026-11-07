"""Agent and task texts for the crew (§ARCH 7.9). Each task description ends with the delimited
inputs block (`llm.crewai_llm.task_inputs`), which is the only place the mock reads inputs from.

Inputs are masked text only. Descriptions are built by concatenation and the crew is kicked off
without `inputs`, so CrewAI never interpolates them: braces in a query stay literal.
"""

from __future__ import annotations

from typing import Final

from credagent.governance.policy import AGENT_ROLES
from credagent.llm.crewai_llm import task_inputs

# agent key → CrewAI role; the policy table is the single source of the role strings.
ROLES: Final = {key: role for role, key in AGENT_ROLES.items()}
GOALS: Final = {
    "retrieval_agent": "Find what Cred's policy documents say about the member's question.",
    "lookup_agent": "Report the current state of the member's loan application.",
    "composer_agent": "Write one accurate, grounded answer for the member.",
}
BACKSTORIES: Final = {
    "retrieval_agent": (
        "You answer only from Cred's knowledge base and say 'I don't know' when it is silent."
    ),
    "lookup_agent": (
        "You read loan applications by id and never guess an application's state."
    ),
    "composer_agent": (
        "You combine the policy findings and the application record into the final answer, "
        "adding nothing that is not in them."
    ),
}  # fmt: skip

_RETRIEVAL: Final = (
    "Look up the member's question in the policy knowledge base with your tool, once, using the "
    "question as written. Skip the tool when the question only asks for an application's status."
)
_LOOKUP: Final = (
    "If a record id is given, look up that loan application with your tool, once. If the record "
    "id is 'none', do not call the tool."
)
_COMPOSER: Final = (
    "Compose the final answer to the member's question from the retrieval result and the lookup "
    "result in your context. Use only facts found there."
)
RETRIEVAL_EXPECTED: Final = (
    'One JSON object with "kind": "retrieval_result" and "rag": the tool output, or null when '
    "the tool was skipped."
)
LOOKUP_EXPECTED: Final = (
    'One JSON object with "kind": "lookup_result": the tool output, or found=false with '
    'error "no_record_id" when no id was given.'
)
COMPOSER_EXPECTED: Final = 'One JSON object with "kind": "crew_response".'


def retrieval_description(query: str, record_id: str | None) -> str:
    return f"{_RETRIEVAL}\n\n{task_inputs(query, record_id)}"


def lookup_description(query: str, record_id: str | None) -> str:
    return f"{_LOOKUP}\n\n{task_inputs(query, record_id)}"


def composer_description(query: str, record_id: str | None) -> str:
    return f"{_COMPOSER}\n\n{task_inputs(query, record_id)}"
