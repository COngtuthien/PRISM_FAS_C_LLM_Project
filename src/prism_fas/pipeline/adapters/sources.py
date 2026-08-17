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
from typing import Any

from prism_fas.pipeline.adapters import AdapterError

#: Where a portable bundle keeps its derived packages, relative to the project.
SOURCE_PACKAGE_ROOT = "data/packages"
GPAT_PAIR_ROOT = "data/packages/gpat_pairs"


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
            "note": "conditioned on the frozen C3 recipes; the imagery is synthetic "
                    "and carries no source-package content"}

    return _real_support_batch(repo, size, seed=seed)


def _real_support_batch(repo: Path, size: int, *,
                        seed: int) -> tuple[Any, dict[str, Any]]:
    """The scientific support batch, assembled by the canonical M8 pipeline.

    `m8_pipeline.build_batch` is what `GPATTrainer` itself calls, so the batch a
    scientific C4 trains on is built by the same code that has always built it —
    the sample store, the resolved recipe bank and the pinned AdaFace embedder.
    Nothing about the batch is constructed here.
    """
    from prism_fas.pipeline.adapters.common import import_canonical

    pair_root = repo / GPAT_PAIR_ROOT
    package_root = repo / SOURCE_PACKAGE_ROOT
    bank_root = repo / "assets/recipe_banks/c3"
    for label, path in (("the source-only GPAT pair plan", pair_root),
                        ("the preprocessed source packages", package_root),
                        ("the frozen C3 recipe banks", bank_root)):
        if not path.is_dir():
            raise SourceUnavailable(
                f"{path.relative_to(repo).as_posix()} is absent; it holds {label}, "
                "which a scientific support batch is built from and for which "
                "there is no substitute on this path")

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
        "pairs_used": size}


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
           "SOURCE_PACKAGE_ROOT", "GPAT_PAIR_ROOT"]
