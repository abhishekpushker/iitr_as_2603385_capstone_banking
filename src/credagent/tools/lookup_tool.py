"""Loan-application lookup (§ARCH 7.7, 7.9, 7.14.1).

`check_loan_application_status(record_id)` is pure: it normalises the id (`strip().upper()`),
validates `^LA-[0-9]{4}$` (ASCII digits only), and returns a dict that validates against
`LoanStatusFound` or `LoanStatusNotFound`. It never raises for any input: a malformed id
(including non-strings and injection strings) is `invalid_record_id`, a well-formed unknown id is
`not_found`.

`LoanStatusTool` is the CrewAI wrapper. It is **owner-bound**: constructed only for
`owner="lookup_agent"` and re-checking `self.owner` against the ACL on every `_run`. The only
exception it raises is `LeastAutonomyViolation`, for a tampered owner; nothing is recorded then.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Final

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

import dataset
from credagent.schemas import RECORD_ID_PATTERN, LoanStatusFound, LoanStatusNotFound
from credagent.settings import get_settings
from credagent.tools.acl import assert_allowed
from credagent.tools.escalation import EscalationConfig, load_escalation_config, should_escalate
from credagent.tools.ledger import ToolCallLedger

TOOL_NAME: Final = "check_loan_application_status"
_RECORD_ID: Final = re.compile(RECORD_ID_PATTERN)

Record = Mapping[str, Any]
RECORDS: Final[Mapping[str, Record]] = MappingProxyType(
    {r["record_id"]: r for r in dataset.LOAN_APPLICATIONS}
)


@lru_cache(maxsize=1)
def default_config() -> EscalationConfig:
    """`config/escalation.json`, loaded once per process."""
    return load_escalation_config(get_settings().escalation_config_path)


def normalize_record_id(raw: object) -> str | None:
    """The canonical id, or `None` if `raw` is not a well-formed record id."""
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().upper()
    return candidate if _RECORD_ID.fullmatch(candidate) else None


def check_loan_application_status(
    record_id: object,
    *,
    config: EscalationConfig | None = None,
    records: Mapping[str, Record] | None = None,
) -> dict[str, Any]:
    rid = normalize_record_id(record_id)
    if rid is None:
        return LoanStatusNotFound(found=False, error="invalid_record_id").model_dump()
    record = (RECORDS if records is None else records).get(rid)
    if record is None:
        return LoanStatusNotFound(found=False, error="not_found").model_dump()
    cfg = default_config() if config is None else config
    score = cfg.score(record["days_since_created"], record["flagged_for_fraud_review"])
    return LoanStatusFound(
        found=True,
        record_id=rid,
        category=record["category"],
        status=record["status"],
        loan_amount_inr=record["loan_amount_inr"],
        days_since_created=record["days_since_created"],
        flagged_for_fraud_review=record["flagged_for_fraud_review"],
        escalation_score=score,
        escalation_recommended=should_escalate(score, cfg.threshold),
        escalation_threshold=cfg.threshold,
    ).model_dump()


class RecordIdArg(BaseModel):
    """No pattern here on purpose: a malformed id must reach the function and come back as
    `invalid_record_id` JSON, not as a CrewAI argument-validation error."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(description="The loan application id, e.g. LA-0042.")


class LoanStatusTool(BaseTool):
    name: str = TOOL_NAME
    description: str = (
        "Look up one loan application by its id (format LA-0000). Returns JSON with the "
        "application's status, loan amount and escalation score, or found=false with an error."
    )
    args_schema: type[BaseModel] = RecordIdArg
    owner: str

    _ledger: ToolCallLedger = PrivateAttr()
    _config: EscalationConfig = PrivateAttr()
    _records: Mapping[str, Record] = PrivateAttr()

    def __init__(
        self,
        *,
        owner: str,
        ledger: ToolCallLedger,
        config: EscalationConfig | None = None,
        records: Mapping[str, Record] | None = None,
        **data: Any,
    ) -> None:
        super().__init__(owner=owner, **data)
        assert_allowed(owner, self)  # wiring-time guard: only the Lookup Agent may own it
        self._ledger = ledger
        self._config = default_config() if config is None else config
        self._records = RECORDS if records is None else records

    @property
    def ledger(self) -> ToolCallLedger:
        return self._ledger

    def _run(self, record_id: str) -> str:
        assert_allowed(self.owner, self)  # runtime guard against a tampered owner
        start = time.perf_counter()
        result = check_loan_application_status(
            record_id, config=self._config, records=self._records
        )
        self._ledger.record(
            agent=self.owner, tool=self.name, args={"record_id": record_id},
            duration_ms=(time.perf_counter() - start) * 1000, cache_hit=False,
        )  # fmt: skip
        return json.dumps(result)
