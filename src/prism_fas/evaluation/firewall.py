"""The structural target-label firewall.

Enforcement is a RESOLVABLE-PATH PERMISSION CHECK plus a capability object, not a
string scan. This is the M8/M9 lesson, recorded twice already and applying here
without change:

    the required evidence of isolation is literally a set of keys named after the
    forbidden things (`target_labels_opened: false`), so a blanket token match
    flags the proof of isolation as a leak.

So: declaration keys are skipped structurally, paths are compared after
`Path.resolve()` (so `..`, a symlink or a relative spelling cannot slip past), and
a value is only a violation when it names a forbidden thing OUTSIDE a declaration.

    TRAIN  may not resolve the target feature root or the target label root
    G7     may resolve the target feature root, may not resolve the label root
    G8     may resolve the label root and the prediction root, and may not write
           any model state (`.pt`, optimizer, calibration, checkpoints/)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import yaml
from .contracts import M10ContractError, TargetLabelFirewallViolation

FIREWALL_SCHEMA_VERSION = "m10-firewall-v1"
STAGES = ("TRAIN", "G7", "G8")
ROOT_NAMES = ("source_package_root", "target_feature_root", "target_label_root", "prediction_root")
PERMISSIONS = ("read", "write", "deny")

# Keys whose presence IS the evidence of isolation. Their subtrees are skipped by
# the structural scan; they are data, never instructions and never a leak.
ISOLATION_DECLARATION_KEYS = frozenset({
    "target_labels_revealed", "target_labels_opened", "target_test_opened",
    "selection_uses_target", "uses_target", "used_target", "used_source_dev",
    "forbidden_splits", "forbidden_columns", "forbidden_in_training_config",
    "forbidden_target_columns", "target_paths_in_config", "isolation",
    "target_attack_taxonomy_in_config", "permissions", "roots", "mounts_by_stage",
    "attack_taxonomy_forbidden_outside_g8", "isolation_declaration_keys",
    "g8_forbidden_write_patterns", "reject_policy_reason", "reason",
    "blocked_reason", "blocked_reasons", "not_claimed", "disclaimer",
    "target_label_artifact", "opened_splits_by_stage", "attack_family_stems",
    "private_evaluation_fields", "expected_counts", "table_59_text", "question",
    "statement", "description", "docstring", "note", "notes"})

# G8 may never produce model state, whatever root it is pointed at.
G8_FORBIDDEN_WRITE_PATTERNS = (".pt", ".pth", ".ckpt", ".safetensors",
                               "optimizer", "calibration/", "checkpoints/")


@dataclass(frozen=True)
class FirewallConfig:
    """The declared roots and the per-stage permission table."""
    roots: dict[str, Path]
    permissions: dict[str, dict[str, str]]
    attack_taxonomy: tuple[str, ...] = ()
    declaration_keys: frozenset[str] = field(default_factory=lambda: ISOLATION_DECLARATION_KEYS)
    g8_forbidden_write_patterns: tuple[str, ...] = G8_FORBIDDEN_WRITE_PATTERNS

    def validate(self) -> "FirewallConfig":
        missing = [name for name in ROOT_NAMES if name not in self.roots]
        if missing: raise M10ContractError(f"the firewall config is missing roots {missing}")
        for stage in STAGES:
            if stage not in self.permissions:
                raise M10ContractError(f"the firewall config declares no permissions for {stage}")
            for name in ROOT_NAMES:
                value = self.permissions[stage].get(name, "deny")
                if value not in PERMISSIONS:
                    raise M10ContractError(f"{stage}.{name} permission {value!r} is not one of {list(PERMISSIONS)}")
        # The three non-negotiable rules, asserted rather than trusted to the YAML.
        if self.permissions["TRAIN"].get("target_label_root", "deny") != "deny":
            raise M10ContractError("TRAIN may never read target labels")
        if self.permissions["TRAIN"].get("target_feature_root", "deny") != "deny":
            raise M10ContractError("TRAIN may never read the target feature root")
        if self.permissions["G7"].get("target_label_root", "deny") != "deny":
            raise M10ContractError("G7 may never read target labels")
        if self.permissions["G8"].get("target_label_root", "deny") != "read":
            raise M10ContractError("G8 must be the stage that reads the evaluation-only labels")
        return self


def load_firewall_config(path: Path) -> FirewallConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise M10ContractError(f"{Path(path).name} is not a mapping")
    roots = {name: Path(str(payload["roots"][name])) for name in ROOT_NAMES}
    isolation = payload.get("isolation") or {}
    declared = payload.get("isolation", {}).get("isolation_declaration_keys") or []
    return FirewallConfig(
        roots=roots,
        permissions={stage: dict(payload["permissions"][stage]) for stage in STAGES},
        attack_taxonomy=tuple(str(name) for name in isolation.get("attack_taxonomy_forbidden_outside_g8", ())),
        declaration_keys=ISOLATION_DECLARATION_KEYS | {str(key) for key in declared},
        g8_forbidden_write_patterns=tuple(str(item) for item in
                                          (payload.get("g8_forbidden_write_patterns") or G8_FORBIDDEN_WRITE_PATTERNS)),
    ).validate()


def _resolved(path: Path) -> Path:
    """`resolve(strict=False)` so a not-yet-built root is still comparable."""
    return Path(path).expanduser().resolve()


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class TargetLabelFirewall:
    """The capability object every M10 stage passes its paths through."""
    config: FirewallConfig
    project_root: Path = Path(".")

    def root_path(self, name: str) -> Path:
        if name not in self.config.roots: raise M10ContractError(f"unknown root {name!r}")
        declared = Path(self.config.roots[name])
        return _resolved(declared if declared.is_absolute() else Path(self.project_root) / declared)

    def classify(self, path: Path) -> str | None:
        """Which declared root a resolved path belongs to, longest match first."""
        candidate = _resolved(path)
        matches = [(len(str(self.root_path(name))), name) for name in ROOT_NAMES
                   if _is_within(candidate, self.root_path(name))]
        return max(matches)[1] if matches else None

    def permission(self, stage: str, root_name: str) -> str:
        if stage not in STAGES: raise M10ContractError(f"unknown firewall stage {stage!r}")
        return self.config.permissions[stage].get(root_name, "deny")

    def check_read(self, stage: str, path: Path) -> Path:
        """Permit a read, or raise. An unclassified path is outside the declared
        roots and is left to the caller's own contracts."""
        root_name = self.classify(path)
        if root_name is None: return _resolved(path)
        if self.permission(stage, root_name) not in ("read", "write"):
            raise TargetLabelFirewallViolation(
                f"{stage} may not read {root_name}; refused {Path(path).name!r}")
        return _resolved(path)

    def check_write(self, stage: str, path: Path) -> Path:
        # The model-state rule is checked FIRST and applies whatever root the path
        # is aimed at, including a path outside every declared root. Checking the
        # root permission first would hide the stronger reason behind a weaker one.
        if stage == "G8":
            text = str(_resolved(path)).replace("\\", "/").lower()
            hit = [pattern for pattern in self.config.g8_forbidden_write_patterns
                   if text.endswith(pattern.lower()) or pattern.lower().rstrip("/") + "/" in text]
            if hit:
                raise TargetLabelFirewallViolation(
                    f"G8 may never write model state; refused {Path(path).name!r} (matched {hit})")
        root_name = self.classify(path)
        if root_name is not None and self.permission(stage, root_name) != "write":
            raise TargetLabelFirewallViolation(
                f"{stage} may not write {root_name}; refused {Path(path).name!r}")
        return _resolved(path)

    def assert_cannot_resolve_labels(self, stage: str) -> dict[str, Any]:
        """The positive proof TRAIN and G7 need to record."""
        label_root = self.root_path("target_label_root")
        permitted = self.permission(stage, "target_label_root") == "read"
        if permitted and stage != "G8":
            raise TargetLabelFirewallViolation(f"{stage} must not be able to resolve the label root")
        return {"stage": stage, "label_root_permission": self.permission(stage, "target_label_root"),
                "label_root_declared": True, "label_root_exists": label_root.exists(),
                "target_labels_opened": False}

    # --- structural config scans -------------------------------------------
    def _walk(self, node: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
        """Depth-first walk that SKIPS declaration subtrees entirely."""
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key) in self.config.declaration_keys: continue
                yield from self._walk(value, path + (str(key),))
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                yield from self._walk(value, path + (str(index),))
        else:
            yield path, node

    def assert_no_attack_taxonomy(self, payload: dict[str, Any], *, where: str = "config") -> dict[str, Any]:
        """No target attack family may appear outside a declaration.

        Structural: a declaration key's subtree is never scanned, so
        `attack_taxonomy_forbidden_outside_g8: [Replay, ...]` — the list that makes
        this check possible — is not itself a violation.
        """
        families = {name.lower() for name in self.config.attack_taxonomy}
        if not families: raise M10ContractError("no attack taxonomy is declared; the check would be vacuous")
        hits = []
        for location, value in self._walk(payload):
            if isinstance(value, str) and value.lower() in families:
                hits.append({"path": "/".join(location), "value": value})
        if hits:
            raise TargetLabelFirewallViolation(
                f"the target attack taxonomy appears in {where} outside a declaration: {hits[:5]}")
        return {"checked": where, "families_declared": len(families), "violations": 0}

    def assert_no_target_paths(self, payload: dict[str, Any], *, where: str = "config") -> dict[str, Any]:
        """No target root may be reachable from a training config."""
        needles = ("target_test", "siw_mv2", "siw-mv2", "evaluation_only")
        hits = []
        for location, value in self._walk(payload):
            if isinstance(value, str) and any(needle in value.lower() for needle in needles):
                hits.append({"path": "/".join(location), "value": value})
        if hits:
            raise TargetLabelFirewallViolation(
                f"a target path appears in {where} outside a declaration: {hits[:5]}")
        return {"checked": where, "violations": 0}

    def report(self) -> dict[str, Any]:
        return {"firewall_schema_version": FIREWALL_SCHEMA_VERSION,
                "roots": {name: str(self.root_path(name)) for name in ROOT_NAMES},
                "permissions": {stage: dict(self.config.permissions[stage]) for stage in STAGES},
                "train_can_read_target_labels": self.permission("TRAIN", "target_label_root") == "read",
                "g7_can_read_target_labels": self.permission("G7", "target_label_root") == "read",
                "g8_can_read_target_labels": self.permission("G8", "target_label_root") == "read",
                "g8_can_write_model_state": False,
                "enforcement": "resolvable_path_permission_check_not_string_scan"}
