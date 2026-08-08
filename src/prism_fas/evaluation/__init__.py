"""M10: the experiment matrix, the blind target evaluation and the report.

Module boundary (spec section 12.1 / Table 41): **nothing in this package imports
modal**, exactly as `prism_fas.detector` does not. The Modal wrapper orchestrates;
it never holds evaluation logic.

Boundary (spec Table 54): the evaluator never updates a model, an optimizer or a
calibration from target labels, and the reporter never recomputes a prediction.
"""
from __future__ import annotations

from .contracts import (M10ContractError, MetricNotAvailable, PredictionLockError,
                        ScoringRefusal, TargetLabelFirewallViolation, is_not_applicable,
                        may_carry_statistical_claim, not_applicable, stable_identity)
from .firewall import FirewallConfig, TargetLabelFirewall, load_firewall_config

__all__ = ["M10ContractError", "MetricNotAvailable", "PredictionLockError", "ScoringRefusal",
           "TargetLabelFirewallViolation", "is_not_applicable", "may_carry_statistical_claim",
           "not_applicable", "stable_identity", "FirewallConfig", "TargetLabelFirewall",
           "load_firewall_config"]
