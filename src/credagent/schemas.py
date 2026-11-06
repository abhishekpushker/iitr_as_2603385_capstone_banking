"""Domain Pydantic models (§ARCH 7.11). Phase 1 adds only the dataset vocabularies.

`LoanCategory` and `LoanStatus` mirror `dataset.CATEGORY_WEIGHTS` / `STATUS_WEIGHTS`. They are
shared through a unit test that asserts equality, not an import (dataset.py is stdlib-only).
"""

from __future__ import annotations

from typing import Literal

LoanCategory = Literal["Personal", "Home", "Auto", "Education", "Business"]
LoanStatus = Literal["Submitted", "Under Review", "Approved", "Rejected", "Disbursed"]
