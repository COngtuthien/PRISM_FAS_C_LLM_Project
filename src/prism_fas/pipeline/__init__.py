"""The v1.5 execution layer (Appendix L).

Profiles, dual status, stage declarations, pipeline state, the master run index,
identity-aware resume and operational budget. `train.py` is the human-facing
entrypoint; everything it does lives here, because L.4 requires the root script
to delegate rather than accumulate logic.

Nothing in this package holds a scientific constant. Thresholds, quotas,
schedules, seed families and selection rules come from the v1.5 spec and the
immutable locks; this package only decides what runs, records what happened and
refuses what the profile does not permit.
"""
from __future__ import annotations

from prism_fas.pipeline.profiles import (PROFILE_NAMES, ExecutionProfile, ProfileError,
                                         load_profile)
from prism_fas.pipeline.stages import STAGE_IDS, STAGES, Stage, get_stage, stage_slice
from prism_fas.pipeline.status import (ENGINEERING_STATUS, SCIENTIFIC_STATUS, DualStatus,
                                       StatusError, scientifically_complete)

__all__ = ["PROFILE_NAMES", "ExecutionProfile", "ProfileError", "load_profile",
           "STAGES", "STAGE_IDS", "Stage", "get_stage", "stage_slice",
           "ENGINEERING_STATUS", "SCIENTIFIC_STATUS", "DualStatus", "StatusError",
           "scientifically_complete"]
