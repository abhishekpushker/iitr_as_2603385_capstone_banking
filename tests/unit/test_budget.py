"""The runtime budget policy (§ARCH 7.14.3) with the measured constants from config/budget.json."""

from __future__ import annotations

from credagent.governance.budget import BudgetPolicy
from credagent.settings import get_settings


def test_limit_ordering_body_cap_above_pydantic_above_budget() -> None:
    """P10: a request is rejected by the budget long before Pydantic's length limit, and a
    maximal-length message still fits under the body cap (≤ 4 UTF-8 bytes per char)."""
    s = get_settings()
    policy = BudgetPolicy()
    n = policy.smallest_rejected_length()
    assert policy.preflight("x" * n).exceeds
    assert not policy.preflight("x" * (n - 1)).exceeds
    assert n < s.max_message_chars
    assert s.max_message_chars * 4 < s.max_body_bytes == 64 * 1024
