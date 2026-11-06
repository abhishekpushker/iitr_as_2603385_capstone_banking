"""Domain Pydantic models (§ARCH 7.11).

`LoanCategory` and `LoanStatus` mirror `dataset.CATEGORY_WEIGHTS` / `STATUS_WEIGHTS`. They are
shared through a unit test that asserts equality, not an import (dataset.py is stdlib-only).
Phase 3 adds the grounded-generation result (`SourceRef`, `GeneratedAnswer`).
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

LoanCategory = Literal["Personal", "Home", "Auto", "Education", "Business"]
LoanStatus = Literal["Submitted", "Under Review", "Approved", "Rejected", "Disbursed"]

# §ARCH 7.5 step 1: the answer when retrieval is below the calibrated threshold.
FALLBACK_ANSWER: Final = "I don't know based on the available policy documents."

_MODEL = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SourceRef(BaseModel):
    model_config = _MODEL

    doc_id: str
    chunk_id: str
    similarity: float = Field(ge=-1, le=1)


class GeneratedAnswer(BaseModel):
    model_config = _MODEL

    answer: str = Field(min_length=1)
    sources: tuple[SourceRef, ...]
    grounded: bool
    top1_similarity: float = Field(ge=-1, le=1)
