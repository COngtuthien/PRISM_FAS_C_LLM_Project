"""Where a stage's inputs come from, decided by the context rather than the stage.

This is the one seam where a rehearsal and a scientific run genuinely differ.
Everything downstream — scheduling, identity, checkpointing, resume, selection,
acceptance, the writers — is the same code under both; what changes is whether
the tensors entering it came from a deterministic fixture or from the frozen
source packages.

Keeping that difference in one module is the point. Scattered `if smoke:`
branches inside ten adapters would be ten places for the two paths to diverge,
and the divergence would only ever be discovered on the machine that runs the
scientific pass.

Two rules hold here:

**A fixture never reaches a scientific context.** Every entry point asks the
context first, and the scientific branch has no fixture fallback to fall into.
If the real input is unreadable the call raises `SourceUnavailable`, which the
orchestrator reports as a data problem — never as a missing implementation.

**The real branch delegates.** Loading a source package, building a pair plan and
resolving a sealed target package are all implemented elsewhere, canonically.
This module calls those implementations; it does not contain a second copy of
any of them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from prism_fas.pipeline.adapters import AdapterError
from prism_fas.pipeline.preparation import (DERIVED_PACKAGES, M7_RECIPE_BANK,
                                            PAIR_PLAN_PACKAGE)

#: The canonical scientific inputs, taken from the module that PRODUCES them
#: rather than spelled again here.
#:
#: Both were wrong, and in different ways. `SOURCE_PACKAGE_ROOT` was
#: `data/packages` — the parent directory — while `SampleStore.open` reads
#: `<package_root>/manifests/source_train.parquet`, a file that has never existed
#: one level up. And the recipe bank was `assets/recipe_banks/c3`, the container
#: of the three C3 treatment banks, which is not an M7 frozen-bank root at all.
#:
#: The bank is not a preference. `m8_pipeline.build_batch` does
#: `recipes[pair["recipe_id"]]` over the bank it is handed, so the support batch
#: must resolve the SAME bank the pair plan drew its recipe ids from — anything
#: else is a KeyError at best and a silent mismatch at worst. Preparation binds
#: the plan to M3B + M7 from frozen Version-B evidence; this consumer reads that
#: binding rather than restating it, so the two cannot drift apart.
SOURCE_PACKAGE_ROOT = DERIVED_PACKAGES[PAIR_PLAN_PACKAGE]
GPAT_PAIR_ROOT = DERIVED_PACKAGES["gpat_pairs"]
RECIPE_BANK_ROOT = M7_RECIPE_BANK

#: C3 stays exactly where it was: the rehearsal fixture's conditioning source and
#: nothing else. `adapters/c4.SUPPORT_RECIPE_SOURCE` names one arm's recipes so a
#: rehearsal exercises a real conditioning vector; a scientific run never reaches
#: that branch, because `fixtures_permitted` is `not scientific_eligible`.
REHEARSAL_CONDITIONING_SOURCE = "assets/recipe_banks/c3"


class SourceUnavailable(AdapterError):
    """A real input a scientific run needs could not be read.

    Deliberately distinct from `CanonicalImplementationUnavailable`: this means
    the data is absent or malformed on this machine, which is a legitimate
    runtime block. It never means the code path was not written.
    """

    #: Reported so the orchestrator can classify the block without parsing prose.
    reason_code = "MISSING_DATA"


def support_batch(repo: Path, size: int, context: Any, *,
                  seed: int = 20260806) -> tuple[Any, dict[str, Any]]:
    """The GPAT support batch: a fixture under rehearsal, the pair plan under science.

    Returns the batch and a provenance block naming which branch produced it, so
    the artifact records its own origin rather than relying on the profile stamp
    alone.
    """
    if context.fixtures_permitted:
        from prism_fas.pipeline.adapters.c4 import _fixture_batch

        return _fixture_batch(repo, size, seed=seed), {
            "source": "deterministic_fixture",
            "fixture_backed": True,
            "seed": seed,
            "conditioning_source": REHEARSAL_CONDITIONING_SOURCE,
            "note": "conditioned on the frozen C3 recipes; the imagery is synthetic "
                    "and carries no source-package content. C3 conditions the "
                    "REHEARSAL only: the scientific branch resolves the frozen M7 "
                    "bank the pair plan is bound to, and cannot reach this branch "
                    "because fixtures_permitted is `not scientific_eligible`."}

    return _real_support_batch(repo, size, seed=seed)


class SupportIdentityMismatch(SourceUnavailable):
    """The scientific inputs are all present and each disagrees with the others.

    Distinct from plain absence because the operator's response differs: nothing
    is missing, so nothing here can be fetched or rebuilt to fix it. Something
    was rebuilt out from under the pair plan, or the wrong root was resolved.
    """

    reason_code = "IDENTITY_MISMATCH"


def verify_support_inputs(repo: Path) -> dict[str, Any]:
    """Prove the pair plan, the M3B package and the M7 bank are the same three.

    Every pair id in the plan is a digest over the package identity, the bank
    identity and a recipe id from that bank, and `build_batch` looks each recipe
    id up in whatever bank it is handed. So three artifacts have to agree before
    a single tensor is built, and this refuses rather than discovers it later:

    * the pair-plan lock and both pair manifests are present;
    * the M3B `PACKAGE_LOCK.json` is present and records its own validation as
      passed — the full re-derivation is M3A/M3B's own build-time step, not
      something to repeat per batch;
    * the package content identity equals the plan's `package_identity`;
    * the M7 bank passes the canonical frozen-bank gate — every `BANK_FILES`
      entry, `validate_bank`, `status == frozen`, the bank id and the pinned
      content identity; and
    * that bank identity equals the plan's `recipe_bank_identity`.

    Reads three lock files. Opens no manifest, no image and no target.
    """
    import json

    from prism_fas.pipeline.preparation import PreparationError, validate_recipe_bank

    pair_root = repo / GPAT_PAIR_ROOT
    package_root = repo / SOURCE_PACKAGE_ROOT

    required = {
        "the source-only GPAT pair plan lock": pair_root / "PAIR_PLAN_LOCK.json",
        "the GPAT training pair manifest": pair_root / "pair_manifest_train.parquet",
        "the GPAT validation pair manifest": pair_root / "pair_manifest_validation.parquet",
        "the frozen M3B source package lock": package_root / "PACKAGE_LOCK.json",
    }
    for label, path in required.items():
        if not path.is_file():
            raise SourceUnavailable(
                f"{path.relative_to(repo).as_posix()} is absent; it holds {label}, "
                "which a scientific support batch is built from and for which "
                "there is no substitute on this path")

    plan = json.loads((pair_root / "PAIR_PLAN_LOCK.json").read_text(encoding="utf-8"))
    package = json.loads((package_root / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))

    if package.get("status") != "validated":
        raise SourceUnavailable(
            f"{SOURCE_PACKAGE_ROOT} reports status {package.get('status')!r}; a "
            "scientific run trains only against a package its own validator passed")
    validation = package.get("package_validation") or {}
    if validation.get("status") != "passed":
        raise SourceUnavailable(
            f"{SOURCE_PACKAGE_ROOT} records package_validation "
            f"{validation.get('status')!r}; the package never completed validation")

    # Raises PreparationError(RECIPE_BANK_INVALID) on any of the five bank rules.
    try:
        bank = validate_recipe_bank(repo)
    except PreparationError as error:
        raise SourceUnavailable(
            f"the frozen recipe bank is not usable: {error}") from error

    package_identity = str(package.get("content_identity_sha256"))
    bank_identity = str(bank["bank_content_identity_sha256"])
    disagreements: list[str] = []
    if plan.get("package_identity") != package_identity:
        disagreements.append(
            f"the pair plan was built against package {plan.get('package_identity')} "
            f"but {SOURCE_PACKAGE_ROOT} is {package_identity}")
    if plan.get("recipe_bank_identity") != bank_identity:
        disagreements.append(
            f"the pair plan was built against recipe bank "
            f"{plan.get('recipe_bank_identity')} but {RECIPE_BANK_ROOT} is "
            f"{bank_identity}")
    if disagreements:
        raise SupportIdentityMismatch(
            "the scientific support inputs do not agree: " + "; ".join(disagreements)
            + ". Every pair id is a digest over both identities, so a batch built "
            "from these would not be the plan that was locked.")

    return {"package_root": SOURCE_PACKAGE_ROOT, "package_identity": package_identity,
            "bank_root": RECIPE_BANK_ROOT, "bank_id": bank["bank_id"],
            "bank_status": bank["status"], "bank_identity": bank_identity,
            "bank_recipe_count": bank["recipe_count"],
            "pair_root": GPAT_PAIR_ROOT,
            "pair_plan_identity": str(plan.get("pair_plan_identity_sha256")),
            "train_pairs": plan.get("train_pairs"),
            "validation_pairs": plan.get("validation_pairs"),
            "identities_agree": True,
            "verified_by": "prism_fas.pipeline.adapters.sources.verify_support_inputs"}


def _real_support_batch(repo: Path, size: int, *,
                        seed: int) -> tuple[Any, dict[str, Any]]:
    """The scientific support batch, assembled by the canonical M8 pipeline.

    `m8_pipeline.build_batch` is what `GPATTrainer` itself calls, so the batch a
    scientific C4 trains on is built by the same code that has always built it —
    the sample store, the resolved recipe bank and the pinned AdaFace embedder.
    Nothing about the batch is constructed here.
    """
    from prism_fas.pipeline.adapters.common import import_canonical

    identities = verify_support_inputs(repo)
    pair_root = repo / GPAT_PAIR_ROOT
    package_root = repo / SOURCE_PACKAGE_ROOT
    bank_root = repo / RECIPE_BANK_ROOT

    m8 = import_canonical("prism_fas.synthesis.m8_pipeline",
                          "the GPAT sample store and batch builder")
    pair_plan = import_canonical("prism_fas.synthesis.pair_plan", "the GPAT pair plan")
    quality = import_canonical("prism_fas.synthesis.quality_models",
                               "the pinned identity model registry")

    audit = m8.SourceOnlyAudit()
    store = m8.SampleStore.open(package_root, audit)
    bank = m8.resolve_bank(bank_root)
    pairs = m8.load_pairs(pair_root, "train")
    if len(pairs) < size:
        raise SourceUnavailable(
            f"the pair plan holds {len(pairs)} training pairs but the support batch "
            f"needs {size}; a scientific run does not shrink to fit its input (L.12)")

    # The lock identities agreed; this is the row-level consequence of that, and
    # it is what would otherwise surface as a bare KeyError inside `build_batch`.
    known = {recipe.recipe_id for recipe in bank["recipes"]}
    unknown = sorted({pair["recipe_id"] for pair in pairs[:size]} - known)
    if unknown:
        raise SupportIdentityMismatch(
            f"{len(unknown)} recipe id(s) in the pair plan are not in "
            f"{RECIPE_BANK_ROOT}, starting at {unknown[:3]}")

    registry = quality.QualityModelRegistry.resolve(repo / "weights",
                                                    roles=("identity",))
    identity_model = registry.adaface("cpu")
    batch = m8.build_batch(store, pairs[:size], bank, identity_model, device="cpu")
    return batch, {
        "source": "frozen_source_only_pair_plan",
        "fixture_backed": False,
        "seed": seed,
        "builder": "prism_fas.synthesis.m8_pipeline.build_batch (the same builder "
                   "GPATTrainer uses; not reimplemented here)",
        "pair_plan_identity": pair_plan.pair_plan_identity(pair_root),
        "adaface_weight_sha256": registry.verified["identity"],
        "pairs_available": len(pairs),
        "pairs_used": size,
        "inputs": identities,
        # Measured by the store itself as it opened, not asserted here.
        "source_only_audit": audit.report()}


#: Where a scientific C5 wrote its rendered candidates. The C6 matched banks name
#: candidate ids; the bytes for those ids live here, and C7/C8 resolve them
#: through `detector.c6_bank` rather than by scanning the directory.
C5_CANDIDATES_ROOT = "runs/full/c5/scientific/candidates"

#: The pinned pretrained weight root. Weights are never vendored into git; this
#: is where the GPU host materializes them.
WEIGHT_ROOT = "weights"


class DetectorInputsUnavailable(SourceUnavailable):
    """A scientific detector run's frozen inputs are not usable on this host."""

    reason_code = "MISSING_DETECTOR_INPUT"


def verify_detector_inputs(repo: Path, *, arms: Sequence[str] = ()) -> dict[str, Any]:
    """Prove every frozen input a scientific C7/C8 detector run consumes.

    Five artifacts have to agree before one tensor is built, and each is checked
    by the module that OWNS it rather than by a second opinion here:

    * the frozen C6 closure and its three arm BANK_LOCKs
      (`evaluation.c6_evidence.verify_c6_evidence`, which applies the same strict
      rules C6's own VERIFY_C6_LOCKS applies);
    * the M3B source package, present and recording its own validation as passed;
    * the frozen M7 recipe bank, through the canonical frozen-bank gate;
    * the pinned SigLIP2 tower and ConvNeXt V2 Atto weight, SHA-verified by
      `detector.pretrained` -- never a shape-exact stub and never a download;
    * the C5 candidate tree the matched banks address.

    Reads locks, a package lock and weight file hashes. Opens no image, no
    manifest row and no target.
    """
    import json

    from prism_fas.evaluation import c6_evidence
    from prism_fas.pipeline.preparation import PreparationError, validate_recipe_bank

    repo = Path(repo)
    package_root = repo / SOURCE_PACKAGE_ROOT

    package_lock = package_root / "PACKAGE_LOCK.json"
    if not package_lock.is_file():
        raise DetectorInputsUnavailable(
            f"{SOURCE_PACKAGE_ROOT}/PACKAGE_LOCK.json is absent; a scientific detector "
            "run trains on the frozen M3B package or it does not train")
    package = json.loads(package_lock.read_text(encoding="utf-8"))
    if package.get("status") != "validated" or (
            package.get("package_validation") or {}).get("status") != "passed":
        raise DetectorInputsUnavailable(
            f"{SOURCE_PACKAGE_ROOT} reports status {package.get('status')!r} and "
            f"validation {(package.get('package_validation') or {}).get('status')!r}; "
            "a scientific run trains only against a package its own validator passed")

    try:
        bank = validate_recipe_bank(repo)
    except PreparationError as error:
        raise DetectorInputsUnavailable(
            f"the frozen M7 recipe bank is not usable: {error}") from error

    try:
        evidence = c6_evidence.verify_c6_evidence(repo)
    except c6_evidence.C6EvidenceError as error:
        raise DetectorInputsUnavailable(
            f"the frozen C6 closure does not verify: {error}") from error

    requested = tuple(arms) or evidence.arms
    unknown = sorted(set(requested) - set(evidence.arms))
    if unknown:
        raise DetectorInputsUnavailable(
            f"C6 froze no matched bank for arm(s) {unknown}; it froze {evidence.arms}")

    candidates_root = repo / C5_CANDIDATES_ROOT
    if not candidates_root.is_dir():
        raise DetectorInputsUnavailable(
            f"{C5_CANDIDATES_ROOT} is absent; the C6 banks name candidate ids whose "
            "rendered bytes live there, and there is no substitute for them")

    weights = _pinned_detector_weights(repo)

    return {
        "package_root": SOURCE_PACKAGE_ROOT,
        "package_identity": str(package.get("content_identity_sha256")),
        "recipe_bank_root": RECIPE_BANK_ROOT,
        "recipe_bank_id": bank["bank_id"],
        "recipe_bank_identity": str(bank["bank_content_identity_sha256"]),
        "recipe_bank_recipe_count": bank["recipe_count"],
        "candidates_root": C5_CANDIDATES_ROOT,
        "weight_root": WEIGHT_ROOT,
        "pretrained": weights,
        "c6": evidence.as_dict(),
        "c6_arms": list(requested),
        "identities_agree": True,
        "target_paths_resolved": 0,
        "target_labels_resolved": 0,
        "verified_by": "prism_fas.pipeline.adapters.sources.verify_detector_inputs",
    }


def _pinned_detector_weights(repo: Path) -> dict[str, Any]:
    """The two pinned backbones, resolved and SHA-verified, or a refusal.

    `SigLIP2Artifacts.resolve` and `resolve_convnext_weight` verify every pinned
    file hash themselves. A missing or altered weight raises `PretrainedError`,
    which is translated rather than swallowed: a scientific detector may not fall
    back to a randomly initialized tower, and the audit fixture model that C7
    readiness builds is not a substitute for one.
    """
    from prism_fas.detector.pretrained import (PretrainedError, SigLIP2Artifacts,
                                               resolve_convnext_weight, sha256_file)

    repo = Path(repo)
    root = repo / WEIGHT_ROOT
    try:
        siglip = SigLIP2Artifacts.resolve(root)
        convnext = resolve_convnext_weight(root)
    except (PretrainedError, OSError) as error:
        raise DetectorInputsUnavailable(
            f"the pinned detector backbones are not resolvable under {WEIGHT_ROOT}: "
            f"{error}. A scientific detector binds the frozen SigLIP2 tower and the "
            "pinned ConvNeXt V2 Atto weight, never a shape-exact stub and never a "
            "silent download") from error

    return {
        "global_tower": {
            "role": "frozen_global_tower",
            "component": "siglip",
            "path": _relative_to(siglip.root, repo),
            "identity_sha256": siglip.identity(),
            "frozen": True,
        },
        "local_backbone": {
            "role": "trainable_local_branch",
            "component": "convnext",
            "path": _relative_to(convnext, repo),
            "weight_sha256": sha256_file(convnext),
            "frozen": False,
        },
        "resolved_by": "prism_fas.detector.pretrained (SHA-verified pins)",
        "stub_substituted": False,
        "downloaded_during_run": False,
    }


def _relative_to(path: Path, repo: Path) -> str:
    try:
        return Path(path).relative_to(Path(repo)).as_posix()
    except ValueError:
        return Path(path).as_posix()


def target_roots(repo: Path, reports: Path, context: Any) -> tuple[dict[str, Path],
                                                                   dict[str, Any]]:
    """The C10 target package roots.

    Under rehearsal these are a synthetic package written inside the rehearsal
    reports tree — four fixture videos with invented labels, which is why a
    rehearsal cannot produce a Version-C P3 number even in principle. Under
    science they are the sealed real package, resolved through the target
    firewall, which is the only component in this file permitted to name SiW.
    """
    if context.fixtures_permitted:
        from prism_fas.pipeline.adapters.c10 import _fixture_roots

        return _fixture_roots(reports), {
            "source": "synthetic_fixture_package",
            "fixture_backed": True,
            "real_target_resolved": False,
            "note": "four synthetic videos with invented labels; structurally "
                    "incapable of yielding a target metric"}

    return _real_target_roots(repo)


def _real_target_roots(repo: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """The sealed scientific target package, resolved through the canonical firewall.

    `prism_fas.data.target_eval` owns the layout, the inventory audit and the
    label sealing; this function locates the package and confirms the firewall's
    own two assertions hold before any stage is handed the roots. It does not
    open a label — `seal_evaluation_labels` did that once, and only C12's scorer
    may reverse it.
    """
    from prism_fas.pipeline.adapters.common import import_canonical

    firewall = import_canonical("prism_fas.data.target_eval",
                                "the target firewall and package resolver")

    layout_path = repo / "configs/data/target_layout.yaml"
    package_root = repo / "data/packages/target_eval"
    if not layout_path.is_file():
        raise SourceUnavailable(
            "configs/data/target_layout.yaml is absent; it declares where the "
            "held-out target lives and the firewall will not guess")
    if not package_root.is_dir():
        raise SourceUnavailable(
            "data/packages/target_eval is absent; the sealed target package is "
            "built by the C10 preparation step and is not rebuilt implicitly here")

    layout = firewall.load_target_layout(layout_path)
    # Both are the firewall's own assertions, raising on violation. Calling them
    # before handing out roots means a package that leaked an identity or a
    # label into its feature rows fails here rather than at C11.
    label_free = firewall.assert_features_label_free(
        firewall.evaluation_label_rows(
            firewall.audit_inventory(package_root, layout)))
    no_identity = firewall.assert_no_target_identity(package_root)

    roots = {"package": package_root,
             "features": package_root / "features",
             "predictions": package_root / "predictions",
             "labels": package_root / "sealed_labels"}
    return roots, {
        "source": "sealed_scientific_target_package",
        "fixture_backed": False,
        "real_target_resolved": True,
        "labels_sealed": True,
        "features_label_free": label_free,
        "no_target_identity": no_identity,
        "note": "§19.2: features are label-free until the scorer unlocks them at C12"}


__all__ = ["SourceUnavailable", "support_batch", "target_roots",
           "SupportIdentityMismatch", "verify_support_inputs",
           "DetectorInputsUnavailable", "verify_detector_inputs",
           "SOURCE_PACKAGE_ROOT", "GPAT_PAIR_ROOT", "RECIPE_BANK_ROOT",
           "C5_CANDIDATES_ROOT", "WEIGHT_ROOT",
           "REHEARSAL_CONDITIONING_SOURCE"]
