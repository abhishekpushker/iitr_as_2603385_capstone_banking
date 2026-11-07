"""Runtime budget policy (§ARCH 7.14.3). Policy only: the meter that charges every mock call is
`context.BudgetMeter`; the orchestrator turns a rejection into `BudgetExceeded` (stage 3).

`preflight(masked_text)` projects a request's worst case before any LLM call:

    projected = N_LLM_CALLS × (PROMPT_OVERHEAD_TOKENS + ceil(len/4) + MAX_OUTPUT_TOKENS)

and rejects it if `projected > BUDGET_MAX_TOKENS` or its cost exceeds `BUDGET_MAX_COST_USD`.
The two constants and the caps are measured by `scripts/measure_budget.py` into
`config/budget.json` and read through `settings`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from credagent.context import estimate_tokens
from credagent.settings import get_settings

N_LLM_CALLS: Final = 7  # retrieval 2, lookup 2, composer 1, reviewer 1, editor 1


@dataclass(frozen=True)
class PreflightResult:
    exceeds: bool
    projected_tokens: int
    budget_tokens: int


def cost_usd(tokens: int, price_per_1k_tokens_usd: float) -> float:
    """Rounded like every reported cost, so a projection exactly at the cap is not over it."""
    return round(tokens / 1000 * price_per_1k_tokens_usd, 8)


class BudgetPolicy:
    def __init__(
        self,
        *,
        prompt_overhead_tokens: int | None = None,
        max_output_tokens: int | None = None,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
        price_per_1k_tokens_usd: float | None = None,
        n_llm_calls: int = N_LLM_CALLS,
    ) -> None:
        s = get_settings()
        self.prompt_overhead_tokens = (
            s.prompt_overhead_tokens if prompt_overhead_tokens is None else prompt_overhead_tokens
        )
        self.max_output_tokens = (
            s.max_output_tokens if max_output_tokens is None else max_output_tokens
        )
        self.max_tokens = s.budget_max_tokens if max_tokens is None else max_tokens
        self.max_cost_usd = s.budget_max_cost_usd if max_cost_usd is None else max_cost_usd
        self.price = (
            s.price_per_1k_tokens_usd
            if price_per_1k_tokens_usd is None
            else price_per_1k_tokens_usd
        )
        self.n_llm_calls = n_llm_calls

    def project(self, masked_text: str) -> int:
        return self.project_tokens(estimate_tokens(masked_text))

    def project_tokens(self, input_tokens: int) -> int:
        per_call = self.prompt_overhead_tokens + input_tokens + self.max_output_tokens
        return self.n_llm_calls * per_call

    def _exceeds(self, projected: int) -> bool:
        return projected > self.max_tokens or cost_usd(projected, self.price) > self.max_cost_usd

    def preflight(self, masked_text: str) -> PreflightResult:
        projected = self.project(masked_text)
        return PreflightResult(
            exceeds=self._exceeds(projected), projected_tokens=projected,
            budget_tokens=self.max_tokens,
        )  # fmt: skip

    def smallest_rejected_length(self) -> int:
        """The fewest characters preflight rejects (projection is monotone in the length)."""
        lo, hi = 0, 1
        while not self._exceeds(self.project("x" * hi)):
            lo, hi = hi, hi * 2
        while hi - lo > 1:  # invariant: lo passes (or is 0), hi is rejected
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if not self._exceeds(self.project("x" * mid)) else (lo, mid)
        return hi
