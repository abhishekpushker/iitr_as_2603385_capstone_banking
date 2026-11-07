"""Least-autonomy policy table (§ARCH 7.14.1, guard 1). Read-only at runtime.

`TOOL_ACL` maps each tool name to the only agent keys that may hold it; a tool missing from the
table is denied to everyone. `AGENT_ROLES` maps the CrewAI `role` strings (§ARCH 7.9) to agent
keys, so the crew audit can identify an agent; an unknown role fails the audit.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

TOOL_ACL: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "check_loan_application_status": frozenset({"lookup_agent"}),
        "rag_lookup": frozenset({"retrieval_agent"}),
    }
)

AGENT_ROLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "Policy Retrieval Specialist": "retrieval_agent",
        "Loan Application Lookup Specialist": "lookup_agent",
        "Response Composer": "composer_agent",
    }
)
