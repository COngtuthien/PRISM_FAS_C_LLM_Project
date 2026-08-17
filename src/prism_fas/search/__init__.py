"""Bounded source-only search: the frozen envelope and the engine that walks it.

`plan` declares what may be searched; `coordinate` executes exactly that. The
split is deliberate — a plan is written and hashed before the first candidate
runs, so "what was this search allowed to do" is answerable from an artifact
rather than from the code that happened to run.
"""
from __future__ import annotations

from prism_fas.search.coordinate import (EnvelopeExhausted, SearchError,
                                         SearchInterrupted, SearchOutcome, Trial,
                                         TrialResult, coordinate_search, rank_key)
from prism_fas.search.plan import (CANONICAL_TIE_BREAK, Coordinate, SearchPlan,
                                   SearchPlanError, canonical_config_sha256,
                                   detector_search_plan, gpat_search_plan)

__all__ = ["CANONICAL_TIE_BREAK", "Coordinate", "SearchPlan", "SearchPlanError",
           "canonical_config_sha256", "detector_search_plan", "gpat_search_plan",
           "EnvelopeExhausted", "SearchError", "SearchInterrupted", "SearchOutcome",
           "Trial", "TrialResult", "coordinate_search", "rank_key"]
