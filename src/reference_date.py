"""Single source of truth for the frozen reference date.

Every date-math op in the pipeline (recency, tenure, honeypot interval checks,
last-active decay) anchors here so Stage-3 reproduction is wall-clock-independent.

CI lint (added Phase 4) greps the codebase for `date.today()` / `datetime.now()`
and fails on any hit outside this file.
"""

from datetime import date

REFERENCE_DATE: date = date(2026, 6, 1)

__all__ = ["REFERENCE_DATE"]
