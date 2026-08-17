"""Canonical execution profiles (v1.5 Appendix L.2, L.5, L.12).

A profile is loaded from ``configs/execution/<name>.yaml`` and carries its own
identity over the file bytes, so "which profile did this artifact run under" is
answerable from the artifact alone rather than from the command line that has
since scrolled away.

The invariants L.2 states about profiles are enforced here, at load time,
against the file's own declarations. That ordering matters: a profile config
that had been edited to make validate scientifically eligible would be rejected
when it is read, before any stage could write an artifact under it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: `rehearsal` is CPU_FULL_REHEARSAL — a full traversal of the implementation on
#: a host with no suitable GPU. It sits beside smoke rather than replacing it:
#: smoke proves the control flow cheaply, rehearsal proves the real training
#: path. Neither is scientifically eligible, and L.2's "exactly one eligible
#: profile" invariant below is what keeps that true no matter how many
#: non-eligible profiles exist.
PROFILE_NAMES: tuple[str, ...] = ("validate", "smoke", "rehearsal", "full")

#: L.2: exactly one profile is scientifically eligible, and exactly the same one
#: may select a winner. Held here as the expectation the loaded file is checked
#: against, so a config edit cannot quietly grant eligibility.
_ELIGIBLE_PROFILE = "full"

CONFIG_DIR = Path("configs") / "execution"


class ProfileError(ValueError):
    """The profile is unknown, unreadable, or contradicts L.2."""


@dataclass(frozen=True)
class ComputePolicy:
    """What a profile is permitted to reach. Checked before a stage runs."""

    scientific_training: bool
    gpu: str
    modal: Any
    live_provider: Any
    live_provider_permitted_for_stages: tuple[str, ...]
    target_label_access: Any
    target_metric_access: Any
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def forbids_live_provider(self) -> bool:
        return self.live_provider is False

    @property
    def forbids_target_labels(self) -> bool:
        return self.target_label_access is False

    def as_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class ExecutionProfile:
    """One loaded profile, plus the identity of the bytes it came from."""

    name: str
    purpose: str
    scientific_eligible: bool
    may_select_scientific_winner: bool
    winner_selection_rule: str | None
    reports_namespace: str
    runs_namespace: str | None
    compute_policy: ComputePolicy
    engineering_budget: dict[str, Any] | None
    reduction_permitted: bool
    preserved_under_reduction: tuple[str, ...]
    phases: tuple[str, ...]
    config_path: str
    profile_identity: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def reports_dir(self, repo: Path) -> Path:
        return repo / self.reports_namespace

    def runs_dir(self, repo: Path) -> Path | None:
        return repo / self.runs_namespace if self.runs_namespace else None

    def stamp(self) -> dict[str, Any]:
        """The two fields L.2 requires on every artifact, plus provenance.

        Every writer in the pipeline package starts from this, which is why no
        artifact can be written without declaring its profile and eligibility.
        """
        return {
            "execution_profile": self.name,
            "scientific_eligible": self.scientific_eligible,
            "profile_identity": self.profile_identity,
            "profile_config": self.config_path,
        }


def _require(payload: dict[str, Any], key: str, path: Path) -> Any:
    if key not in payload:
        raise ProfileError(f"{path.name} is missing the required key {key!r}")
    return payload[key]


def load_profile(name: str, *, repo: Path) -> ExecutionProfile:
    """Load and validate one profile.

    Validation is against L.2 rather than against a schema of our own: the
    profile's declared eligibility must agree with the single profile the spec
    makes eligible, and the namespaces must be distinct per profile so evidence
    from two profiles can never land in one directory.
    """
    if name not in PROFILE_NAMES:
        raise ProfileError(f"unknown execution profile {name!r}; expected one of {PROFILE_NAMES}")

    path = repo / CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise ProfileError(f"execution profile config not found: {path}")

    raw_bytes = path.read_bytes()
    payload = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ProfileError(f"{path.name} does not parse to a mapping")

    declared = _require(payload, "profile", path)
    if declared != name:
        raise ProfileError(
            f"{path.name} declares profile {declared!r} but was loaded as {name!r}")

    eligible = bool(_require(payload, "scientific_eligible", path))
    may_select = bool(_require(payload, "may_select_scientific_winner", path))
    if eligible != (name == _ELIGIBLE_PROFILE):
        raise ProfileError(
            f"{path.name} declares scientific_eligible={eligible} for profile {name!r}; "
            f"L.2 makes exactly {_ELIGIBLE_PROFILE!r} eligible and no other")
    if may_select != (name == _ELIGIBLE_PROFILE):
        raise ProfileError(
            f"{path.name} declares may_select_scientific_winner={may_select} for profile "
            f"{name!r}; L.2 permits winner selection only under {_ELIGIBLE_PROFILE!r}")

    reports_namespace = str(_require(payload, "reports_namespace", path))
    if reports_namespace != f"reports/{name}":
        raise ProfileError(
            f"{path.name} declares reports_namespace {reports_namespace!r}; L.2 gives "
            f"profile {name!r} the namespace 'reports/{name}'")

    runs_namespace = payload.get("runs_namespace")
    if runs_namespace is not None and str(runs_namespace) != f"runs/{name}":
        raise ProfileError(
            f"{path.name} declares runs_namespace {runs_namespace!r}, which would let "
            f"{name!r} evidence share a run tree with another profile")

    policy_raw = dict(_require(payload, "compute_policy", path))
    policy = ComputePolicy(
        scientific_training=bool(policy_raw.get("scientific_training", False)),
        gpu=str(policy_raw.get("gpu", "none")),
        modal=policy_raw.get("modal", False),
        live_provider=policy_raw.get("live_provider", False),
        live_provider_permitted_for_stages=tuple(
            policy_raw.get("live_provider_permitted_for_stages") or ()),
        target_label_access=policy_raw.get("target_label_access", False),
        target_metric_access=policy_raw.get("target_metric_access", False),
        raw=policy_raw,
    )
    if not eligible and policy.scientific_training:
        raise ProfileError(
            f"{path.name} permits scientific training under a profile that cannot produce "
            "scientifically eligible evidence")
    if not eligible and policy.live_provider is not False:
        raise ProfileError(
            f"{path.name} permits a live provider under non-eligible profile {name!r}")
    if not eligible and policy.target_label_access is not False:
        raise ProfileError(
            f"{path.name} permits target label access under non-eligible profile {name!r}")

    budget = payload.get("engineering_budget")
    if budget is not None and eligible:
        raise ProfileError(
            f"{path.name} declares an engineering budget for the full profile; L.12 forbids "
            "shrinking a scientific budget through operational configuration")

    return ExecutionProfile(
        name=name,
        purpose=str(payload.get("purpose", "")),
        scientific_eligible=eligible,
        may_select_scientific_winner=may_select,
        winner_selection_rule=payload.get("winner_selection_rule"),
        reports_namespace=reports_namespace,
        runs_namespace=str(runs_namespace) if runs_namespace else None,
        compute_policy=policy,
        engineering_budget=dict(budget) if isinstance(budget, dict) else None,
        reduction_permitted=bool(payload.get("reduction_permitted", False)),
        preserved_under_reduction=tuple(payload.get("preserved_under_reduction") or ()),
        phases=tuple(payload.get("phases") or ()),
        config_path=f"{CONFIG_DIR.as_posix()}/{name}.yaml",
        profile_identity=hashlib.sha256(raw_bytes).hexdigest(),
        raw=payload,
    )


__all__ = ["PROFILE_NAMES", "CONFIG_DIR", "ProfileError", "ComputePolicy",
           "ExecutionProfile", "load_profile"]
