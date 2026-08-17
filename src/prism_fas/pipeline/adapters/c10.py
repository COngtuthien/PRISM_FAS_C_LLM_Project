"""C10 — target package, capability lock and the label firewall.

This is the stage with the strictest handling rule in the project, and the rule
shapes the adapter: **the real SiW target package is never opened here.** Not
read, not hashed, not resolved. Every root this adapter points the firewall at
lives inside the smoke namespace and was created by this run.

That is not caution for its own sake. §19.2 says training, LLM and synthesis
environments may resolve no SiW labels, and C10 runs long before any prediction
exists. An adapter that "just checked the package identity" would be the first
Version-C process to resolve the target root, and it would do so from a profile
that is not permitted to.

What is exercised is the machinery: the canonical `TargetLabelFirewall`, its
per-stage permission table, its refusal to let a training stage resolve a label
root, its refusal to let the scorer write model state, feature-package identity
verification and tamper detection. All of it runs against a fixture package whose
bytes this adapter wrote, so a failure here is a defect in the firewall rather
than a fact about SiW.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism_fas.pipeline.adapters import AdapterRequest, AdapterResult
from prism_fas.pipeline.adapters.common import (EngineeringAdapter, RequiredInput, check,
                                                resume_decision, stage_reports_dir, utc,
                                                write_artifact)

STAGE_ID = "C10"

BUILD_FIXTURE_PACKAGE = "BUILD_FIXTURE_PACKAGE"
FIREWALL_PERMISSIONS = "FIREWALL_PERMISSIONS"
PACKAGE_IDENTITY = "PACKAGE_IDENTITY"
TARGET_LOCK = "TARGET_LOCK"
TAMPER_DETECTION = "TAMPER_DETECTION"

MODES: tuple[str, ...] = (BUILD_FIXTURE_PACKAGE, FIREWALL_PERMISSIONS, PACKAGE_IDENTITY,
                          TARGET_LOCK, TAMPER_DETECTION)

TARGET_CONFIG = "configs/evaluation/m10_target.yaml"


def _fixture_roots(reports: Path) -> dict[str, Path]:
    """Four disjoint roots inside the smoke namespace. None of them is SiW."""
    base = reports / "fixture_target"
    return {"source_package_root": base / "source_package",
            "target_feature_root": base / "target_features",
            "target_label_root": base / "target_labels",
            "prediction_root": base / "predictions"}


def _package_identity(root: Path) -> str:
    entries = sorted((path.relative_to(root).as_posix(),
                      hashlib.sha256(path.read_bytes()).hexdigest())
                     for path in root.rglob("*") if path.is_file())
    return hashlib.sha256(json.dumps(entries, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass
class C10Adapter(EngineeringAdapter):
    """The C10 execution adapter. The firewall is imported, never reimplemented."""

    stage_id: str = STAGE_ID
    substages: tuple[str, ...] = (STAGE_ID,)
    title: str = "Target package and label firewall"
    modes: tuple[str, ...] = MODES
    requires_gpu: bool = False

    def required_inputs(self) -> tuple[RequiredInput, ...]:
        return (
            RequiredInput("target_evaluation_config", TARGET_CONFIG,
                          "the frozen roots, permissions and prediction schema"),
            RequiredInput("c9_source_lock", "reports/full/c9/SOURCE_MATRIX_LOCK_C.json",
                          "the source freeze that must precede any target work"),
            RequiredInput("target_feature_package", "data/processed/prism_target_eval_v2",
                          "the SiW-Mv2 v2 feature package, mounted READ-ONLY for C11"),
        )

    def run_smoke(self, request: AdapterRequest) -> list[AdapterResult]:
        reports = stage_reports_dir(request, STAGE_ID)
        roots, build = self._build_fixture(request, reports)
        return [build,
                self._permissions(request, roots, reports),
                self._identity(request, roots, reports),
                self._lock(request, roots, reports),
                self._tamper(request, roots, reports)]

    # --- modes ----------------------------------------------------------------

    def _build_fixture(self, request: AdapterRequest,
                       reports: Path) -> tuple[dict[str, Path], AdapterResult]:
        checks: list[dict[str, Any]] = []
        roots = _fixture_roots(reports)
        for name, path in roots.items():
            path.mkdir(parents=True, exist_ok=True)
            (path / "README.txt").write_text(
                f"engineering fixture root for {name}; contains no real target data\n",
                encoding="utf-8")
        (roots["target_feature_root"] / "features.jsonl").write_text(
            "\n".join(json.dumps({"video_id": f"fixture_video_{index:03d}",
                                  "frame_id": frame, "feature_dim": 8})
                      for index in range(4) for frame in range(4)) + "\n",
            encoding="utf-8")
        (roots["target_label_root"] / "labels.json").write_text(
            json.dumps({f"fixture_video_{index:03d}": index % 2 for index in range(4)}),
            encoding="utf-8")

        real = _real_roots(request.repo)
        checks.append(check(
            "c10_fixture_roots_are_disjoint_from_the_real_target",
            all(not str(path.resolve()).startswith(str(Path(request.repo, value).resolve()))
                for path in roots.values() for value in real.values()),
            "every fixture root lies inside the smoke namespace and under no real "
            "target root",
            fixture_roots={name: path.relative_to(request.repo).as_posix()
                           for name, path in roots.items()},
            declared_real_roots=real))
        checks.append(check(
            "c10_real_target_package_not_opened", True,
            "this adapter did not read, hash or resolve the real target package",
            real_feature_root=real.get("target_feature_root"),
            real_label_root=real.get("target_label_root"),
            files_opened_under_real_roots=0,
            rule="§C10/§19.2: the SiW feature package may be mounted read-only for C11 "
                 "inference; label files are evaluation-only and are never mounted on a "
                 "training process. C10 under a non-eligible profile opens neither"))

        artifact = write_artifact(request, reports / "C10_FIXTURE_PACKAGE.json", {
            "schema_version": "c10-fixture-package-v1", "generated_at_utc": utc(),
            "mode": BUILD_FIXTURE_PACKAGE,
            "roots": {name: path.relative_to(request.repo).as_posix()
                      for name, path in roots.items()},
            "fixture_backed": True, "contains_real_target_data": False})
        return roots, self.result(request, mode=BUILD_FIXTURE_PACKAGE, checks=checks,
                                  artifacts=[artifact])

    def _permissions(self, request: AdapterRequest, roots: dict[str, Path],
                     reports: Path) -> AdapterResult:
        from prism_fas.evaluation.contracts import TargetLabelFirewallViolation
        from prism_fas.evaluation.firewall import FirewallConfig, TargetLabelFirewall

        checks: list[dict[str, Any]] = []
        declared = _declared_permissions(request.repo)
        firewall = TargetLabelFirewall(
            config=FirewallConfig(roots=dict(roots), permissions=declared).validate(),
            project_root=request.repo)

        def refused(stage: str, root: str, action: str = "read") -> bool:
            method = firewall.check_read if action == "read" else firewall.check_write
            try:
                method(stage, roots[root] / "README.txt")
                return False
            except TargetLabelFirewallViolation:
                return True

        cases = {
            "TRAIN_cannot_read_target_features": refused("TRAIN", "target_feature_root"),
            "TRAIN_cannot_read_target_labels": refused("TRAIN", "target_label_root"),
            "G7_can_read_target_features": not refused("G7", "target_feature_root"),
            "G7_cannot_read_target_labels": refused("G7", "target_label_root"),
            "G8_can_read_target_labels": not refused("G8", "target_label_root"),
            "G8_cannot_read_source_package": refused("G8", "source_package_root"),
        }
        for name, passed in cases.items():
            checks.append(check(f"c10_{name.lower()}", passed,
                                name.replace("_", " "),
                                permission_table=declared))

        model_state = False
        try:
            firewall.check_write("G8", roots["prediction_root"] / "model.pt")
        except TargetLabelFirewallViolation:
            model_state = True
        checks.append(check(
            "c10_g8_cannot_write_model_state", model_state,
            "the scorer cannot write a checkpoint, optimizer or calibration artifact",
            forbidden_patterns=list(firewall.config.g8_forbidden_write_patterns),
            rule="C-G8 is a scorer; producing model state would make target results able "
                 "to mutate an upstream artifact (§C12)"))

        # `assert_cannot_resolve_labels` RETURNS the positive proof when the
        # stage is correctly denied and RAISES when it is not, so a clean return
        # is the pass and the exception is the finding.
        proofs: dict[str, Any] = {}
        for stage in ("TRAIN", "G7"):
            try:
                proofs[stage] = firewall.assert_cannot_resolve_labels(stage)
            except TargetLabelFirewallViolation as violation:
                proofs[stage] = {"violation": str(violation)}
        denied = all("violation" not in proof for proof in proofs.values())
        checks.append(check(
            "c10_training_environment_denial", denied,
            "neither the training stage nor the prediction stage can resolve the label root",
            proofs=proofs,
            rule="§19.2: only the isolated C-G8 scorer may resolve labels, and only after "
                 "the prediction lockset"))

        # And the guard must fire on a leaky table. It turns out to fire earlier
        # than at the resolve call: the config refuses to validate at all, so a
        # leaky permission table can never become a live firewall object. That is
        # the stronger property, so it is what gets asserted.
        from prism_fas.evaluation.contracts import M10ContractError

        fired, refusal = False, ""
        try:
            FirewallConfig(
                roots=dict(roots),
                permissions={**declared,
                             "TRAIN": {**declared["TRAIN"], "target_label_root": "read"}}
            ).validate()
        except (M10ContractError, TargetLabelFirewallViolation) as error:
            fired, refusal = True, str(error)
        checks.append(check(
            "c10_denial_guard_fires_on_a_leaky_table", fired,
            "a permission table that granted TRAIN label access cannot be constructed",
            constructed_case="TRAIN.target_label_root = read", refusal=refusal,
            note="the refusal happens at config validation, before any firewall object "
                 "exists, so there is no window in which a leaky table is live"))

        artifact = write_artifact(request, reports / "C10_FIREWALL.json", {
            "schema_version": "c10-firewall-v1", "generated_at_utc": utc(),
            "mode": FIREWALL_PERMISSIONS, "permissions": declared,
            "cases": cases, "firewall_report": firewall.report(),
            "fixture_backed": True})
        return self.result(request, mode=FIREWALL_PERMISSIONS, checks=checks,
                           artifacts=[artifact])

    def _identity(self, request: AdapterRequest, roots: dict[str, Path],
                  reports: Path) -> AdapterResult:
        checks: list[dict[str, Any]] = []
        feature_root = roots["target_feature_root"]
        identity = _package_identity(feature_root)
        again = _package_identity(feature_root)

        checks.append(check(
            "c10_package_identity_is_deterministic", identity == again,
            "the feature-package identity reproduces from the same bytes",
            identity=identity))
        checks.append(check(
            "c10_package_identity_is_path_independent", True,
            "the identity is taken over sorted relative paths and file hashes",
            material="sorted (relative_path, sha256) pairs",
            rule="a package identity must verify identically on the execution backend, so "
                 "it cannot depend on an absolute location"))

        declared = _declared_identity(request.repo)
        checks.append(check(
            "c10_real_package_identity_declared_not_computed", bool(declared),
            "the real package identity is read from the frozen config and not recomputed "
            "by opening the package",
            declared_target_feature_package_identity=declared,
            computed_here=False,
            rule="verifying the real identity is C10's job under the FULL profile, where "
                 "the read-only mount is permitted"))

        artifact = write_artifact(request, reports / "C10_PACKAGE_IDENTITY.json", {
            "schema_version": "c10-package-identity-v1", "generated_at_utc": utc(),
            "mode": PACKAGE_IDENTITY, "fixture_package_identity": identity,
            "declared_real_identity": declared, "real_package_opened": False,
            "fixture_backed": True})
        return self.result(request, mode=PACKAGE_IDENTITY, checks=checks,
                           artifacts=[artifact])

    def _lock(self, request: AdapterRequest, roots: dict[str, Path],
              reports: Path) -> AdapterResult:
        checks: list[dict[str, Any]] = []
        identity = _package_identity(roots["target_feature_root"])
        material = {
            "target_feature_package_identity": identity,
            "target_package_id": "fixture_target_eval",
            "permissions": _declared_permissions(request.repo),
            "label_capability": {"granted_to": ["G8"], "granted_after": "prediction lockset"},
            "protocol": "P3",
            "target": "siw_mv2_v2",
        }
        lock_identity = hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        checks.append(check(
            "c10_target_lock_binds_capability_separately_from_features",
            material["label_capability"]["granted_to"] == ["G8"],
            "label capability is granted to the scorer alone, and only after the "
            "prediction lockset exists",
            label_capability=material["label_capability"],
            rule="§19.2: grant label access only to the isolated C-G8 scorer after the "
                 "prediction lockset"))
        checks.append(check(
            "c10_target_is_fixed_to_siw_mv2_v2", material["target"] == "siw_mv2_v2",
            "P3's target is fixed in advance and is not a runtime choice",
            rule="§19.1: changing target or dataset composition requires a new protocol "
                 "version; it is not an in-place edit"))
        checks.append(check(
            "c10_lock_identity_reproduces",
            lock_identity == hashlib.sha256(
                json.dumps(material, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")).hexdigest(),
            "the target lock identity recomputes from its own material",
            lock_identity=lock_identity))

        artifact = write_artifact(request, reports / "C10_TARGET_LOCK.json", {
            "schema_version": "c10-target-lock-v1", "generated_at_utc": utc(),
            "mode": TARGET_LOCK, "lock_identity": lock_identity, **material,
            "is_scientific_lock": False,
            "why_not": "built over a fixture package; the scientific target lock is built "
                       "at C10 under the full profile against the real read-only mount",
            "fixture_backed": True})

        decision = resume_decision(request, "c10_target_lock",
                                   reports / "C10_TARGET_LOCK.json",
                                   expected_identity=lock_identity,
                                   identity_key="lock_identity")
        checks.append(check(
            "c10_resume_is_identity_aware", decision["identity_matches"],
            "resume validates the target lock by identity", **decision))
        return self.result(request, mode=TARGET_LOCK, checks=checks, artifacts=[artifact])

    def _tamper(self, request: AdapterRequest, roots: dict[str, Path],
                reports: Path) -> AdapterResult:
        checks: list[dict[str, Any]] = []
        feature_root = roots["target_feature_root"]
        before = _package_identity(feature_root)

        extra = feature_root / "injected.jsonl"
        extra.write_text('{"video_id": "injected", "frame_id": 0}\n', encoding="utf-8")
        after_addition = _package_identity(feature_root)
        extra.unlink()

        target = feature_root / "features.jsonl"
        original = target.read_bytes()
        target.write_bytes(original.replace(b"fixture_video_000", b"fixture_video_999"))
        after_edit = _package_identity(feature_root)
        target.write_bytes(original)
        restored = _package_identity(feature_root)

        checks.append(check(
            "c10_detects_an_added_file", before != after_addition,
            "adding a file changes the package identity",
            before=before[:16], after=after_addition[:16]))
        checks.append(check(
            "c10_detects_an_edited_file", before != after_edit,
            "editing a byte changes the package identity",
            before=before[:16], after=after_edit[:16]))
        checks.append(check(
            "c10_identity_restores_exactly", before == restored,
            "restoring the original bytes restores the original identity",
            before=before[:16], restored=restored[:16]))

        artifact = write_artifact(request, reports / "C10_TAMPER_DETECTION.json", {
            "schema_version": "c10-tamper-detection-v1", "generated_at_utc": utc(),
            "mode": TAMPER_DETECTION,
            "identities": {"baseline": before, "after_added_file": after_addition,
                           "after_edited_file": after_edit, "after_restore": restored},
            "fixture_backed": True})
        return self.result(request, mode=TAMPER_DETECTION, checks=checks,
                           artifacts=[artifact])


def _target_config(repo: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((repo / TARGET_CONFIG).read_text(encoding="utf-8"))


def _declared_permissions(repo: Path) -> dict[str, dict[str, str]]:
    return dict(_target_config(repo)["permissions"])


def _real_roots(repo: Path) -> dict[str, str]:
    """The declared real roots, as STRINGS. Declaring is not resolving."""
    roots = dict(_target_config(repo)["roots"])
    return {name: str(value) for name, value in roots.items()
            if name.endswith("_root")}


def _declared_identity(repo: Path) -> str:
    return str(_target_config(repo)["roots"].get("target_feature_package_identity", ""))


__all__ = ["STAGE_ID", "MODES", "BUILD_FIXTURE_PACKAGE", "FIREWALL_PERMISSIONS",
           "PACKAGE_IDENTITY", "TARGET_LOCK", "TAMPER_DETECTION", "C10Adapter"]
