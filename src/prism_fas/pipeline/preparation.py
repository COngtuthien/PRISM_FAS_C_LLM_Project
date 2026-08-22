"""Rebuilding the derived data trees, automatically, before C4.

The preprocessed frames and the built packages are deterministic derivations of
the raw datasets. They are large, so they are not copied between machines; they
are also required by every scientific stage from C4 onward. Previously that
combination was recorded as "does not travel in the folder", which quietly made
the collaborator responsible for a manual preparation step.

This module removes that step. If a derived tree is absent and its raw source is
present, `python train.py` builds it, validates it, and continues into C4 — no
second command, no flags.

Five properties matter here, and each is delegated rather than reimplemented:

**One contract between producer and consumer.** M2 writes the canonical parquet
manifests and M3A reads them, and both resolve that location through
`m2_output_root`. They used to resolve it separately — M2 through the legacy
`m2a` CLI helper, M3A through `paths.processed_root` — which is a mismatch no
unit test could see and which cost a real GPU host hours of SCRFD work before
failing on a file nothing had ever written.

**Deterministic.** Every builder is the one the project already uses
(`run_preprocessing` under the `full_preprocessing` profile,
`prism_fas.data.package`, `m3b`, `prism_fas.synthesis.pair_plan`). None of them
is re-derived here; this module decides *whether* to call them and in what order.

**Resumable.** Completed M2 records are discovered from the canonical manifests
and are not walked again; the packages skip work whose hash still validates. An
interrupted preparation continues from what survived, and nothing is deleted to
make room for a rebuild.

**Validated by content, not by presence.** "The directory has something in it"
is true of a tree that died halfway through its first dataset. M2 completion is
the canonical `validate_full_profile` plus full record coverage plus a marker
written last; a package is its own validator and lock.

**Honest about absence.** With no raw source there is nothing to derive from, and
this module reports MISSING_RAW_DATA rather than fabricating a smaller package.

Source-only, always. SiW-Mv2 is the held-out target and appears in no list here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "prism-preparation-v1"

MISSING_RAW_DATA = "MISSING_RAW_DATA"
PREPARATION_FAILED = "PREPARATION_FAILED"
M2_INCOMPLETE = "M2_INCOMPLETE"
TARGET_IN_SOURCE_TREE = "TARGET_IN_SOURCE_TREE"
RECIPE_BANK_INVALID = "RECIPE_BANK_INVALID"
PACKAGE_NOT_VALIDATED = "PACKAGE_NOT_VALIDATED"

#: The two states a derived package lock passes through. `building` is written by
#: the builder itself; only `finalize_lock` promotes it, and only after the
#: package validated. "A lock exists" says nothing about which of the two it is.
PACKAGE_STATUS_BUILDING = "building"
PACKAGE_STATUS_VALIDATED = "validated"

#: Built in this order; each consumes the previous one's output.
STEPS = ("m2_preprocess", "m3a_package", "m3b_priors", "gpat_pairs")

#: The M2 run profile the scientific path uses, and the only one this module
#: will ever name. `small_acceptance` (namespace `m2a`) is the frozen
#: three-record acceptance namespace, whose artifacts are JSONL results rather
#: than the canonical parquet manifests M3A consumes; `target_eval_v2` is a
#: post-C12 target namespace that must never run before C4.
M2_RUN_PROFILE = "full_preprocessing"

#: Preparation before C4 is source-only. SiW-Mv2 is the held-out target: it is
#: never preprocessed here, and it appears in no list below.
SOURCE_DATASETS = ("casia_fasd", "msu_mfsd")

#: The five manifests the M2 repository maintains and `build_package` reads.
M2_MANIFESTS = ("source_frames", "source_crops", "target_frames", "target_crops",
                "preprocessing_failures")

#: Every config this module hands to a builder, by project-relative path.
#:
#: Declared rather than spelled out at each call site so that one test can walk
#: them against the shipped folder. `configs/models/model_priors.yaml` was named
#: here for a file that has never existed — the canonical M3B config is
#: `m3b_priors.yaml`, which is what the CLI, the docs and the M3B test suite all
#: use — and the fixture project wrote a placeholder under the wrong name, so
#: nothing compared the string to the repository until a real GPU run did, at
#: M3B, after M2 and M3A had already completed.
PREPARATION_CONFIGS = {
    "m2_preprocess": "configs/data/preprocess_m2.yaml",
    "m2_run_profiles": "configs/data/m2_run_profiles.yaml",
    "m3a_package": "configs/data/package_m3a.yaml",
    "m3b_model_priors": "configs/models/m3b_priors.yaml",
}

#: Per-dataset adapter definitions, resolved by name from `SOURCE_DATASETS`.
DATASET_CONFIG_TEMPLATE = "configs/data/{dataset}.yaml"

#: The derived package roots this module builds and chains, by project-relative
#: path. Declared for the same reason as the configs above: a root spelled at
#: four call sites is a root nothing can check.
DERIVED_PACKAGES = {
    "m3a": "data/packages/prism_data_v1_m3a",
    "m3b": "data/packages/prism_data_v1_m3b",
    "gpat_pairs": "data/packages/gpat_pairs",
}

#: The FROZEN M7 recipe bank, and the only recipe bank M8 has ever been bound to.
#:
#: `assets/recipe_banks/c3/` is a different contract and was named here by
#: mistake: it is a CONTAINER of the three C3 scientific banks (`det/`, `llm/`,
#: `rnd/`), each holding `C3_BANK.json` + `recipes.jsonl`, and it carries none of
#: the seven files `recipes.bank.load_bank` requires. The two are kept explicitly
#: distinct — `PORTABLE_ASSET_MANIFEST.json` lists the C3 container under its own
#: logical name `c3_scientific_recipe_banks`, and nothing converts between them.
M7_RECIPE_BANK = "assets/recipe_banks/prism_recipe_bank_m7_v1"

#: What the frozen bank must hash to. Recorded in Version-B evidence
#: (`docs/c0/C0_VERSION_B_INTEGRITY.md` §2.2) and in the frozen pair-plan lock
#: `reports/m8/pairs/PAIR_PLAN_LOCK.json` on the immutable Version-B repository.
#:
#: `validate_bank` re-derives every hash in BANK_LOCK.json from the files beside
#: it, which catches a tampered bank but not a SUBSTITUTED one — a different bank
#: that is internally consistent passes. Pinning the identity is what makes the
#: check refuse the wrong bank, which is the defect class this whole gate exists
#: for. This records a frozen fact; it does not choose one.
M7_BANK_CONTENT_IDENTITY = (
    "fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb")
M7_BANK_ID = "prism_recipe_bank_m7_v1"

#: The package the GPAT pair plan is bound to. Frozen Version-B evidence: the
#: pair-plan lock records `package_identity` = the M3B package identity
#: b1cf29b6…9dc6, and both production callsites — `modal_m8.py` (REMOTE_PACKAGE)
#: and `cli/main.py::_m8_defaults` — pass the M3B root. M3A is structurally
#: loadable here too, which is exactly the hazard: it would produce a plan whose
#: `package_identity` is stamped into every `pair_id` and into the pair-plan
#: identity, and it would be the wrong one, silently.
PAIR_PLAN_PACKAGE = "m3b"

#: Written under the M2 output root after a full pass that validated. Its absence
#: is what separates an interrupted preprocessing run from a finished one — the
#: manifests exist from the first flush onward, so their presence proves nothing.
M2_COMPLETION_MARKER = "state/M2_PREPARATION_COMPLETE.json"
M2_MARKER_SCHEMA_VERSION = "prism-m2-preparation-v1"


class PreparationError(RuntimeError):
    def __init__(self, reason: str, message: str, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.reason = reason
        self.detail = detail or {}


@dataclass
class StepOutcome:
    name: str
    action: str          # BUILT | FINALIZED | REUSED_VALID |
                         # SKIPPED_NOT_NEEDED | FAILED
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"step": self.name, "action": self.action, "summary": self.summary,
                "seconds": round(self.seconds, 2), **self.detail}


#: The file each derived package writes last. Its absence means the build was
#: interrupted, which a directory listing cannot distinguish from a finished one.
COMPLETION_MARKERS = {
    "packages": ("prism_data_v1_m3a/PACKAGE_LOCK.json",
                 "prism_data_v1_m3b/PACKAGE_LOCK.json"),
    "gpat_pairs": ("PAIR_PLAN_LOCK.json",),
}


def _incomplete(repo: Path, name: str) -> bool:
    """True when a derived tree exists but is not finished scientific input.

    Presence used to be `the directory has something in it`, which is true of a
    package that died halfway through writing. Then it became `the marker its
    builder writes last is there` — which is true of a package whose lock still
    says `building`, because the builder writes that lock itself.

    So this asks the artifact what state it is in. A package must be `validated`
    with its own validation passed; a pair plan must still be bound to the
    package and bank identities that exist right now. Either of those failing
    puts the tree back on the to-do list, which is what lets `prepare` finalize
    an unfinalized package and rebuild a stale plan rather than reporting
    NOTHING_TO_DO over the top of both.
    """
    from prism_fas.pipeline import portable_paths

    root = repo / portable_paths.DERIVED_ROOTS[name]
    if not root.is_dir():
        return False
    if any(not (root / marker).is_file()
           for marker in COMPLETION_MARKERS.get(name, ())):
        return True
    if name == "packages":
        return any(not package_status(repo / DERIVED_PACKAGES[key])[
            "reusable_as_scientific_input"] for key in ("m3a", "m3b"))
    if name == "gpat_pairs":
        return not _pair_plan_diagnosis(repo)["reusable"]
    return False


# --- the ONE canonical M2 contract -------------------------------------------
#
# The defect this section exists to close: `_step_m2` used to call the legacy
# `m2_runner.run`, which writes JSONL results and crops under
# `<work_root>/m2/<version>/<config_hash>/m2a/`, while `_step_m3a` called
# `build_package(paths.processed_root, ...)`, whose `load_m2_samples` reads
# `<input_root>/manifests/source_frames.parquet`. Nothing wrote that file, so the
# real GPU run spent hours on SCRFD and then died with FileNotFoundError.
#
# Producer and consumer now resolve their root through `m2_output_root` — one
# function, so the two cannot drift apart again, and a test asserts they are the
# same path.


def _m2_config(repo: Path) -> Any:
    from prism_fas.data.preprocess_m2 import load_m2_config

    return load_m2_config(Path(repo) / PREPARATION_CONFIGS["m2_preprocess"])


def _m2_profile(repo: Path) -> Any:
    from prism_fas.data.run_profiles import load_profiles

    return load_profiles(Path(repo) / PREPARATION_CONFIGS["m2_run_profiles"])[
        M2_RUN_PROFILE]


def m2_output_root(repo: Path) -> Path:
    """The canonical M2 output root: what `_step_m2` writes and `_step_m3a` reads.

    `profile_root` is the project's own constructor for this path and it refuses
    a root outside `<work_root>/m2/<version>/<config_hash>/`. The directory is
    named for the profile, which matters beyond tidiness: `build_lock` records
    `source_m2_namespace = input_root.name` beside a hard-coded
    `source_m2_validation_profile = "full_preprocessing"`, so any other basename
    produces an M3A lock that contradicts itself.
    """
    from prism_fas.data.run_profiles import profile_root

    paths, config = _paths(repo), _m2_config(repo)
    return profile_root(paths.work_root, config.preprocessing_version,
                        config.config_hash, _m2_profile(repo))


def _manifest_column(path: Path, column: str) -> list[Any]:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=[column]).column(column).to_pylist()


def _manifest_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.read_metadata(path).num_rows)


def _records(repo: Path, dataset: str) -> list[Any]:
    """The canonical record list for one dataset, via the canonical adapter.

    Asking the adapter is the only way to say "all of them" without hard-coding a
    number that would rot the first time a dataset changed.
    """
    import yaml

    from prism_fas.data.adapters import adapter_for
    from prism_fas.config.models import DatasetDefinition

    paths = _paths(repo)
    definition = DatasetDefinition.model_validate(
        yaml.safe_load((Path(repo) / DATASET_CONFIG_TEMPLATE.format(dataset=dataset))
                       .read_text(encoding="utf-8")))
    return list(adapter_for(definition, getattr(paths.raw_datasets, dataset)).records())


def _record_count(repo: Path, dataset: str) -> int:
    """How many records the dataset actually has, via the canonical adapter."""
    return len(_records(repo, dataset))


def _settled_counts(manifests: Path) -> dict[tuple[str, str], int]:
    """Rows already recorded per (dataset, source_record_id), successes + failures.

    A failure row counts: a frame that was walked and routed to
    `preprocessing_failures` is finished work, not missing work, and re-walking
    it every resume would never terminate on a corpus with an undecodable file.
    """
    counts: dict[tuple[str, str], int] = {}
    for name in ("source_frames", "preprocessing_failures"):
        path = manifests / f"{name}.parquet"
        if not path.is_file():
            continue
        datasets = _manifest_column(path, "dataset")
        records = _manifest_column(path, "source_record_id")
        for dataset, record in zip(datasets, records):
            counts[(str(dataset), str(record))] = counts.get((str(dataset), str(record)), 0) + 1
    return counts


def _outstanding(repo: Path, manifests: Path, frames_per_video: int) -> dict[str, list[Any]]:
    """The records this host still has to walk, per source dataset.

    A record is settled once it has contributed at least `frames_per_video` rows.
    A video so short that `uniform_indices` yields fewer than that is re-walked on
    every resume; that is deterministic, idempotent and bounded to a handful of
    frames, and the alternative — trusting a count we cannot verify without
    opening the media — would let a genuinely truncated record be called done.
    """
    settled = _settled_counts(manifests)
    return {dataset: [record for record in _records(repo, dataset)
                      if settled.get((dataset, str(record.video_id)), 0) < frames_per_video]
            for dataset in SOURCE_DATASETS}


def m2_status(repo: Path, *, deep: bool = False,
              require_marker: bool = True) -> dict[str, Any]:
    """Whether the canonical M2 tree is COMPLETE, measured from its content.

    The rule this replaces was `data/processed exists and is non-empty`, which is
    true of a tree that died halfway through its first dataset. Completion here
    means, in order:

    1.  all five canonical manifests exist under `<m2_root>/manifests/`;
    2.  no target rows are present — preparation before C4 is source-only, and a
        SiW row in this tree is a firewall breach, not an incomplete build;
    3.  every canonical source record has been walked;
    4.  the completion marker matches the current config hash, detector hash and
        record counts;
    5.  with `deep=True`, the canonical `validate_full_profile` passes — crop
        SHA-256s, decodability, dimensions, orphans, temporaries, target isolation.

    `deep` is off for the preflight summary, which must stay cheap, and on before
    M3A, which must not consume an unvalidated tree. `require_marker` is off for
    exactly one caller: the check `_step_m2` runs immediately after preprocessing,
    to decide whether it has earned the right to write that marker.
    """
    repo = Path(repo)
    root = m2_output_root(repo)
    manifests = root / "manifests"
    config = _m2_config(repo)
    status: dict[str, Any] = {
        "profile": M2_RUN_PROFILE,
        "root": root.as_posix(),
        "manifests_root": manifests.as_posix(),
        "manifests_present": {name: (manifests / f"{name}.parquet").is_file()
                              for name in M2_MANIFESTS},
        "counts": {}, "outstanding_records": {}, "marker": None,
        "validation": None, "complete": False, "reason": "MANIFESTS_ABSENT",
    }
    if not all(status["manifests_present"].values()):
        return status

    status["counts"] = {name: _manifest_rows(manifests / f"{name}.parquet")
                        for name in M2_MANIFESTS}
    if status["counts"]["target_frames"] or status["counts"]["target_crops"]:
        # Not "incomplete". The pre-C4 tree may not contain the held-out target
        # at all, so this stops the run rather than being repaired by more work.
        status["reason"] = TARGET_IN_SOURCE_TREE
        return status
    if status["counts"]["source_frames"] != status["counts"]["source_crops"]:
        status["reason"] = "FRAME_CROP_COUNTS_DIFFER"
        return status
    if not status["counts"]["source_frames"]:
        status["reason"] = "NO_SOURCE_SAMPLES"
        return status

    outstanding = _outstanding(repo, manifests, int(config.frames_per_video))
    status["outstanding_records"] = {name: len(items) for name, items in outstanding.items()}
    if any(outstanding.values()):
        status["reason"] = "RECORDS_OUTSTANDING"
        return status

    marker_path = root / M2_COMPLETION_MARKER
    if require_marker:
        if not marker_path.is_file():
            status["reason"] = "COMPLETION_MARKER_ABSENT"
            return status
        import json

        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        status["marker"] = marker
        expected = _m2_marker_identity(repo, config)
        if {key: marker.get(key) for key in expected} != expected:
            status["reason"] = "COMPLETION_MARKER_STALE"
            return status

    if deep:
        from prism_fas.data.m2_validation import validate_full_profile

        report = validate_full_profile(_paths(repo), config, root)
        status["validation"] = {
            "validation_version": report["validation_version"],
            "passed": bool(report["passed"]),
            "errors": [item["check_id"] for item in report["errors"]],
            "crop_integrity_passed": bool(report["crop_integrity"]["passed"]),
            "target_isolation_passed": bool(report["target_isolation"]["passed"]),
            "crops_on_disk": report["crops_on_disk"]}
        if not report["passed"]:
            status["reason"] = "VALIDATION_FAILED"
            return status

    status["complete"] = True
    status["reason"] = None
    return status


def _m2_marker_identity(repo: Path, config: Any) -> dict[str, Any]:
    """What the completion marker must agree with to still be believed."""
    from prism_fas.data.preprocess_m2 import resolve_detector_path
    from prism_fas.utils.core import sha256_file

    return {
        "schema_version": M2_MARKER_SCHEMA_VERSION,
        "profile": M2_RUN_PROFILE,
        "preprocessing_version": config.preprocessing_version,
        "preprocessing_config_hash": config.config_hash,
        "detector_model_sha256": sha256_file(
            resolve_detector_path(config.scrfd_model_path)),
        "record_counts": {dataset: _record_count(repo, dataset)
                          for dataset in SOURCE_DATASETS},
    }


def what_is_needed(repo: Path) -> dict[str, Any]:
    """Which derived trees are absent or unfinished, and whether they can be rebuilt.

    Reported without building anything, so the preflight can print it before a
    long run rather than discovering it at C4.
    """
    from prism_fas.pipeline import portable_paths

    repo = Path(repo)
    resolution = portable_paths.resolve(repo)
    raw_present = {name: root.present for name, root in resolution.raw.items()}
    # SiW is the held-out target and is not an input to any derived source tree.
    buildable = raw_present.get("casia_fasd", False) and raw_present.get("msu_mfsd", False)

    missing = [name for name, root in resolution.derived.items()
               if name != "processed" and (not root.present or _incomplete(repo, name))]
    # `processed` is the logical name of the preprocessed source tree. Its state
    # is the M2 contract's, not a directory listing's — the M2 tree lives in the
    # profile namespace under `data/work`, and `data/processed` never held its
    # manifests under any convention this project has used.
    m2 = _m2_status_or_absent(repo, buildable)
    if not m2["complete"]:
        missing.insert(0, "processed")

    return {
        "missing_derived": missing,
        "raw_present": raw_present,
        "m2": m2,
        "can_rebuild": bool(missing) and buildable,
        "nothing_to_do": not missing,
        "blocked": bool(missing) and not buildable,
        "reason": (None if not missing else
                   None if buildable else
                   "the CASIA and MSU raw datasets are required to derive the "
                   "processed and package trees, and at least one is absent"),
    }


def _m2_status_or_absent(repo: Path, buildable: bool) -> dict[str, Any]:
    """`m2_status`, degraded to a reportable answer when it cannot be measured.

    With no raw corpus there is no canonical record list to compare against, and
    with no paths config there is no work root. Neither is an error here: the
    caller already reports MISSING_RAW_DATA, and a preflight summary must not
    raise.
    """
    try:
        if not buildable:
            root = m2_output_root(repo)
            return {"profile": M2_RUN_PROFILE, "root": root.as_posix(),
                    "complete": False, "reason": "RAW_DATA_ABSENT",
                    "manifests_present": {}, "counts": {},
                    "outstanding_records": {}, "marker": None, "validation": None}
        return m2_status(repo)
    except Exception as error:                               # noqa: BLE001 - reported
        return {"profile": M2_RUN_PROFILE, "root": None, "complete": False,
                "reason": f"UNMEASURABLE: {type(error).__name__}: {error}",
                "manifests_present": {}, "counts": {},
                "outstanding_records": {}, "marker": None, "validation": None}


def prepare(repo: Path, *, resume: bool = True,
            dry_run: bool = False) -> dict[str, Any]:
    """Build whatever derived tree is missing, in dependency order.

    Returns a report. Raises `PreparationError` only when a build was needed,
    attempted and failed — an absent raw source is reported as a blocked outcome
    rather than an exception, because the caller has a better message for it.
    """
    from prism_fas.pipeline import portable_paths

    started = time.time()
    repo = Path(repo).resolve()
    # Resolved before anything reads it. `what_is_needed` now measures the M2
    # tree, which means resolving the work root, which means this file must
    # already exist — and the report must record whether it had to be written
    # rather than observing its own side effect and calling it REUSED.
    paths_config = portable_paths.ensure_local_paths(repo)
    needed = what_is_needed(repo)
    outcomes: list[StepOutcome] = []

    if needed["nothing_to_do"]:
        return _report(started, outcomes, needed,
                       outcome="NOTHING_TO_DO",
                       summary="every derived tree is present")

    if needed["blocked"]:
        return _report(started, outcomes, needed, outcome="BLOCKED",
                       summary=needed["reason"] or "raw data is absent",
                       reason_code=MISSING_RAW_DATA)

    if dry_run:
        return _report(started, outcomes, needed, outcome="WOULD_BUILD",
                       summary=f"would build {', '.join(needed['missing_derived'])}")

    for step in STEPS:
        try:
            outcome = _run_step(repo, step, resume=resume)
        except PreparationError:
            raise
        except Exception as error:                           # noqa: BLE001
            raise PreparationError(
                PREPARATION_FAILED,
                f"derived-data preparation failed at {step}: "
                f"{type(error).__name__}: {error}",
                {"step": step, "completed": [item.name for item in outcomes]}) from error
        outcomes.append(outcome)

    after = what_is_needed(repo)
    if after["missing_derived"]:
        raise PreparationError(
            PREPARATION_FAILED,
            "preparation completed but these derived trees are still absent: "
            + ", ".join(after["missing_derived"]),
            {"still_missing": after["missing_derived"]})

    counted = {action: len([item for item in outcomes if item.action == action])
               for action in ("BUILT", "FINALIZED", "REUSED_VALID")}
    return _report(started, outcomes, needed, outcome="PREPARED",
                   summary=f"{counted['BUILT']} tree(s) built, "
                           f"{counted['FINALIZED']} finalized in place, "
                           f"{counted['REUSED_VALID']} reused",
                   paths_config={"action": paths_config["action"],
                                 "reason": paths_config["reason"]})


def _run_step(repo: Path, step: str, *, resume: bool) -> StepOutcome:
    """Dispatch one preparation step to its canonical builder."""
    started = time.time()
    handler = {
        "m2_preprocess": _step_m2,
        "m3a_package": _step_m3a,
        "m3b_priors": _step_m3b,
        "gpat_pairs": _step_pairs,
    }[step]
    outcome = handler(repo, resume=resume)
    outcome.seconds = time.time() - started
    return outcome


def _paths_config(repo: Path) -> Path:
    """The paths config the canonical builders take, guaranteed to describe this folder.

    Every builder below is handed this path rather than a hard-coded
    `configs/paths.local.yaml`. That file is Git-ignored, so a clone has none, and
    a copied folder carries one naming the machine it left — both of which used to
    fail here rather than at the operator, which is exactly the one-folder promise
    this module exists to keep.
    """
    from prism_fas.pipeline import portable_paths

    return Path(portable_paths.ensure_local_paths(repo)["path"])


def _paths(repo: Path) -> Any:
    from prism_fas.config.models import load_paths

    return load_paths(_paths_config(repo))


def _step_m2(repo: Path, *, resume: bool) -> StepOutcome:
    """Preprocessing: raw source video to the canonical M2 parquet manifests.

    This drives the production `full_preprocessing` path — `PreprocessingRunContext`
    plus `run_preprocessing`, the same pair `prism data preprocess run
    --run-profile full_preprocessing` drives — and not the legacy `m2_runner.run`
    CLI helper, which writes JSONL results into the frozen `m2a` acceptance
    namespace that M3A cannot read.

    SiW-Mv2 is absent from `SOURCE_DATASETS` and is never constructed here, so
    the target is not preprocessed and no target label is resolved.
    """
    from prism_fas.data.m2_runner import run_preprocessing
    from prism_fas.data.manifests.repository import ManifestRepository
    from prism_fas.data.preprocess_m2 import SCRFDDetector, resolve_detector_path
    from prism_fas.data.run_context import build_preprocessing_run_context

    # Structural first, then the canonical validator only if the structure says
    # this tree could be finished. The deep pass hashes and decodes every crop,
    # which is minutes of work there is no reason to spend on a tree we already
    # know is half-built.
    before = m2_status(repo)
    if before["complete"]:
        before = m2_status(repo, deep=True)
        if before["complete"]:
            return StepOutcome("m2_preprocess", "REUSED_VALID",
                               "the canonical M2 tree is complete and validates; "
                               "nothing was reprocessed",
                               {"m2_root": before["root"], "counts": before["counts"],
                                "validation": before["validation"]})
    if before["reason"] == TARGET_IN_SOURCE_TREE:
        raise PreparationError(
            TARGET_IN_SOURCE_TREE,
            "the source M2 tree carries target rows. Preparation before C4 is "
            "source-only; this tree cannot be repaired by preprocessing more of "
            "it and must be investigated rather than extended.",
            {"m2_root": before["root"], "counts": before["counts"]})

    paths, config = _paths(repo), _m2_config(repo)
    profile = _m2_profile(repo)
    root = m2_output_root(repo)
    detector_path = resolve_detector_path(config.scrfd_model_path)
    # One session for both datasets: loading SCRFD twice buys nothing and the
    # provider must be the same for every row, because it is recorded per crop.
    detector = SCRFDDetector(detector_path, config.scrfd_input_size,
                             config.detector.get("provider", "CPUExecutionProvider"))

    manifests = root / "manifests"
    outstanding = (_outstanding(repo, manifests, int(config.frames_per_video))
                   if resume and manifests.is_dir() else
                   {dataset: _records(repo, dataset) for dataset in SOURCE_DATASETS})
    per_dataset: dict[str, Any] = {}
    for dataset in SOURCE_DATASETS:
        total = _record_count(repo, dataset)
        pending = outstanding[dataset]
        if not pending:
            per_dataset[dataset] = {"records_total": total, "records_walked": 0,
                                    "action": "ALREADY_WALKED"}
            continue
        context = build_preprocessing_run_context(
            paths, config, profile, dataset, f"preparation-{dataset}",
            all_records=True, limit_records=None, limit_samples=None,
            resume=resume, dry_run=False, partial=False, root=root)
        result = run_preprocessing(context, _Progress(pending, dataset, total),
                                   detector=detector,
                                   repository_factory=ManifestRepository)
        per_dataset[dataset] = {
            "records_total": total, "records_walked": len(pending),
            "records_reused": total - len(pending),
            "samples_successful": result.samples_successful,
            "samples_failed": result.samples_failed,
            "crops_written": result.crops_written,
            "failures_by_code": result.failures_by_code,
            "action": "PREPROCESSED"}

    after = m2_status(repo, deep=True, require_marker=False)
    if not after["complete"]:
        # Two different situations, and the operator needs to tell them apart.
        # More work will finish an under-covered tree; nothing this step can do
        # will repair a crop whose bytes no longer match the hash beside them.
        remedy = ("the canonical validator refused this tree, so more preprocessing "
                  "will not repair it. Read the failed checks below and decide; "
                  "nothing here deletes an artifact."
                  if after["reason"] == "VALIDATION_FAILED" else
                  "the partial work is preserved and a rerun continues from it; "
                  "nothing was deleted.")
        raise PreparationError(
            M2_INCOMPLETE,
            f"M2 preprocessing ran but the tree is still not complete: "
            f"{after['reason']} under {after['root']}. {remedy}",
            {"m2": after, "per_dataset": per_dataset,
             "failed_checks": (after.get("validation") or {}).get("errors", [])})

    _write_m2_marker(repo, root, config, per_dataset, after)
    return StepOutcome("m2_preprocess", "BUILT",
                       "preprocessed the CASIA and MSU source corpora into the "
                       "canonical M2 manifests",
                       {"m2_root": after["root"], "per_dataset": per_dataset,
                        "counts": after["counts"], "validation": after["validation"]})


class _Progress(list):
    """Canonical records that report progress as the runner consumes them.

    Preprocessing a full corpus is hours of work. Without this the operator
    watches a silent terminal and cannot tell a slow run from a hung one — which
    matters more than usual here, because the last attempt spent those hours and
    then threw them away. Aggregate counts only: never a path, a filename or any
    canonical metadata field.
    """

    def __init__(self, records: list[Any], dataset: str, total: int, every: int = 25):
        super().__init__(records)
        self.dataset, self.total, self.every = dataset, total, every
        self.started = time.monotonic()

    def __iter__(self) -> Any:
        pending = len(self)
        print(f"      {self.dataset:12s} {pending} record(s) to walk "
              f"({self.total - pending} already in the manifests)", flush=True)
        for position, record in enumerate(super().__iter__(), 1):
            if position == 1 or position % self.every == 0 or position == pending:
                elapsed = time.monotonic() - self.started
                rate = position / elapsed if elapsed > 0 else 0.0
                remaining = (pending - position) / rate if rate > 0 else 0.0
                print(f"      {self.dataset:12s} {position}/{pending}  "
                      f"{elapsed / 60:.1f} min elapsed, ~{remaining / 60:.0f} min left",
                      flush=True)
            yield record


def _write_m2_marker(repo: Path, root: Path, config: Any,
                     per_dataset: dict[str, Any], status: dict[str, Any]) -> Path:
    """Record that a full, validated pass finished. Written last, on purpose."""
    from prism_fas.pipeline.state import atomic_write_json

    path = root / M2_COMPLETION_MARKER
    atomic_write_json(path, {
        **_m2_marker_identity(repo, config),
        "datasets": per_dataset,
        "manifest_counts": status["counts"],
        "source_only": True,
        "target_datasets_preprocessed": [],
        "validation": status["validation"],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_eligible": False,
    })
    return path


def _step_m3a(repo: Path, *, resume: bool) -> StepOutcome:
    """The M3A package foundation, plus its validator and lock."""
    from prism_fas.data.package import (build_package, finalize_lock,
                                        load_package_config, validate_package)

    root = repo / DERIVED_PACKAGES["m3a"]
    if (root / "PACKAGE_LOCK.json").is_file():
        # Same lifecycle as M3B: a `validated` package is strict-validated and
        # reused, a `building` one is finalized in place, and neither is accepted
        # on the strength of the lock file merely existing.
        state = ensure_package_validated(root, label="the M3A source package")
        return StepOutcome("m3a_package", state["action"],
                           "the M3A package validates; not rebuilt"
                           if not state["finalized"] else
                           "the M3A package was already built; its lock was "
                           "validated and finalized in place",
                           {"package_root": root.name, "package": state})

    # M3A reads `<input_root>/manifests/*.parquet` and resolves each crop as
    # `<input_root>/<crop_relative_path>`, where that path was recorded relative
    # to the M2 output root. So the input root is the M2 output root — resolved
    # by the same function the producer used, never `paths.processed_root`, which
    # holds no M2 manifest under any convention this project has used.
    m2 = m2_status(repo, deep=True)
    if not m2["complete"]:
        raise PreparationError(
            M2_INCOMPLETE,
            f"the M3A package cannot be built from an M2 tree that is not "
            f"complete: {m2['reason']}. Expected the canonical manifests under "
            f"{m2['manifests_root'] if m2.get('manifests_root') else m2['root']}.",
            {"m2": m2})

    config = load_package_config(repo / PREPARATION_CONFIGS["m3a_package"])
    result = build_package(m2_output_root(repo), root, config, resume=resume)

    # Validate, finalize, validate again — the order the canonical CLI uses.
    # `build_package` writes the lock with status "building", so a strict
    # validation before `finalize_lock` fails on `lock.status` every time. The
    # laptop only ever took the dry-run branch, so this never ran here either.
    pre = validate_package(root, require_validated_status=False)
    if not pre.get("passed"):
        raise PreparationError(
            PREPARATION_FAILED,
            "the M3A package was built but did not validate: " + _failed_checks(pre),
            {"failed_checks": _failed_checks(pre), "validation": pre})
    # finalize_lock(package_root, report). The third argument this used to pass
    # was a TypeError on every real M3A build.
    lock = finalize_lock(root, pre)
    report = validate_package(root)
    if not report.get("passed"):
        raise PreparationError(
            PREPARATION_FAILED,
            "the M3A package did not validate after its lock was finalized: "
            + _failed_checks(report),
            {"failed_checks": _failed_checks(report), "validation": report})
    return StepOutcome("m3a_package", "BUILT",
                       "built and validated the M3A source package",
                       {"status": lock.get("status"),
                        "m2_root": m2_output_root(repo).as_posix(),
                        "samples": (result or {}).get("samples")})


def _failed_checks(report: dict[str, Any]) -> str:
    """The check ids that actually failed, so the message names the problem."""
    return ", ".join(item["check_id"] for item in report.get("checks", [])
                     if not item.get("passed")
                     and item.get("severity", "error") == "error") or "no check reported"


def package_status(root: Path) -> dict[str, Any]:
    """What a derived package's own lock says about itself. Read-only."""
    import json

    lock_path = Path(root) / "PACKAGE_LOCK.json"
    if not lock_path.is_file():
        return {"present": Path(root).is_dir(), "locked": False, "status": None,
                "package_validation": None, "content_identity_sha256": None,
                "reusable_as_scientific_input": False,
                "why": "no PACKAGE_LOCK.json"}
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except ValueError as error:
        return {"present": True, "locked": True, "status": None,
                "package_validation": None, "content_identity_sha256": None,
                "reusable_as_scientific_input": False,
                "why": f"PACKAGE_LOCK.json does not parse: {error}"}
    status = lock.get("status")
    validation = (lock.get("package_validation") or {}).get("status")
    reusable = status == PACKAGE_STATUS_VALIDATED and validation == "passed"
    return {
        "present": True, "locked": True, "status": status,
        "package_validation": validation,
        "content_identity_sha256": lock.get("content_identity_sha256"),
        "reusable_as_scientific_input": reusable,
        "why": None if reusable else
               (f"lock status is {status!r}, not {PACKAGE_STATUS_VALIDATED!r}: the "
                "package was built but never validated and finalized, so its "
                "content identity is still the pre-finalization one"
                if status == PACKAGE_STATUS_BUILDING else
                f"lock status {status!r} / package_validation {validation!r}"),
    }


def ensure_package_validated(root: Path, *, parent: Path | None = None,
                             label: str) -> dict[str, Any]:
    """Bring an existing derived package to `validated`, or refuse it.

    The defect this closes: `_step_m3b` treated "the directory exists and holds a
    PACKAGE_LOCK.json" as REUSED_VALID. `build_m3b_package` writes that lock with
    `status: "building"`, and preparation never ran the validate → finalize →
    validate sequence the canonical CLI runs, so a fully built M3B sat there
    unfinalized. C4 refused it, correctly, three steps later.

    "Locked" is not "validated". The lock exists from the first write; `status`
    is what says whether anything checked it.

    A `building` package is finalized IN PLACE. `finalize_lock` rewrites
    `PACKAGE_LOCK.json` and nothing else, so the model priors — hours of frozen
    tower inference — are never recomputed to obtain a status change.

    Finalization is not free of consequence: it promotes `status`, fills in
    `target_isolation` and `package_validation`, and RECOMPUTES
    `content_identity_sha256` over the promoted lock. Anything bound to the
    pre-finalization identity is therefore stale, which is what
    `_pair_plan_is_current` exists to catch.
    """
    from prism_fas.data.package import finalize_lock, validate_package

    state = package_status(root)
    if not state["locked"]:
        raise PreparationError(
            PACKAGE_NOT_VALIDATED,
            f"{label} has no PACKAGE_LOCK.json at {Path(root).as_posix()}",
            {"package": state})

    if state["status"] == PACKAGE_STATUS_VALIDATED:
        report = validate_package(root, parent_package=parent)
        if not report.get("passed"):
            raise PreparationError(
                PACKAGE_NOT_VALIDATED,
                f"{label} reports status validated but does not validate: "
                + _failed_checks(report),
                {"failed_checks": _failed_checks(report), "package": state})
        return {"action": "REUSED_VALID", "finalized": False, **package_status(root),
                "checks_passed": sum(1 for item in report["checks"] if item["passed"]),
                "checks_total": len(report["checks"])}

    if state["status"] != PACKAGE_STATUS_BUILDING:
        raise PreparationError(
            PACKAGE_NOT_VALIDATED,
            f"{label} carries an unrecognized lock status {state['status']!r}; "
            f"only {PACKAGE_STATUS_BUILDING!r} and {PACKAGE_STATUS_VALIDATED!r} "
            "are part of the package lifecycle",
            {"package": state})

    identity_before = state["content_identity_sha256"]
    pre = validate_package(root, require_validated_status=False, parent_package=parent)
    if not pre.get("passed"):
        raise PreparationError(
            PACKAGE_NOT_VALIDATED,
            f"{label} is still building and does not validate, so it cannot be "
            f"finalized: {_failed_checks(pre)}. Nothing was rewritten.",
            {"failed_checks": _failed_checks(pre), "validation": pre, "package": state})

    finalize_lock(root, pre)
    report = validate_package(root, parent_package=parent)
    if not report.get("passed"):
        raise PreparationError(
            PACKAGE_NOT_VALIDATED,
            f"{label} did not validate after its lock was finalized: "
            + _failed_checks(report),
            {"failed_checks": _failed_checks(report), "validation": report})
    return {"action": "FINALIZED", "finalized": True, **package_status(root),
            "content_identity_before_finalization": identity_before,
            "priors_rebuilt": False,
            "checks_passed": sum(1 for item in report["checks"] if item["passed"]),
            "checks_total": len(report["checks"])}


def _step_m3b(repo: Path, *, resume: bool) -> StepOutcome:
    """Model priors: the pinned towers' features over the packaged frames."""
    from prism_fas.data.package.m3b import build_m3b_package

    source = repo / DERIVED_PACKAGES["m3a"]
    target = repo / DERIVED_PACKAGES["m3b"]
    if (target / "PACKAGE_LOCK.json").is_file():
        # "Present and locked" was the defect. The builder writes the lock as
        # `building`; a package is only scientific input once it has validated
        # and been finalized. A `building` package with sound content is
        # finalized in place — the priors are hours of frozen-tower inference and
        # are never recomputed to change a status field.
        state = ensure_package_validated(target, parent=source,
                                         label="the M3B prior package")
        return StepOutcome("m3b_priors", state["action"],
                           "the M3B prior package validates; not rebuilt"
                           if not state["finalized"] else
                           "the M3B priors were already built; the lock was "
                           "validated and finalized in place, without recomputing "
                           "a single prior",
                           {"package_root": target.name, "package": state})

    from prism_fas.pipeline import portable_paths

    resolution = portable_paths.resolve(repo)
    weight_root = resolution.weights.path if resolution.weights else None
    if weight_root is None:
        raise PreparationError(
            MISSING_RAW_DATA,
            "the pinned model weights could not be resolved; M3B priors are "
            "produced by the frozen towers and there is no substitute")

    result = build_m3b_package(source, target,
                               repo / PREPARATION_CONFIGS["m3b_model_priors"],
                               weight_root=weight_root, resume=resume)
    # The builder leaves the lock at `building`. Validate against the M3A parent,
    # finalize, then strict-validate — the sequence `prism data priors model-build`
    # runs, and the sequence whose absence here left a fully built M3B that C4
    # refused three steps later.
    state = ensure_package_validated(target, parent=source,
                                     label="the M3B prior package")
    return StepOutcome("m3b_priors", "BUILT",
                       "built, validated and finalized the M3B model-prior package",
                       {"samples": (result or {}).get("samples"),
                        "package_root": target.name, "package": state})


def recipe_bank_root(repo: Path) -> Path:
    """The ONE frozen recipe bank the pair plan may be built from."""
    return Path(repo) / M7_RECIPE_BANK


def validate_recipe_bank(repo: Path) -> dict[str, Any]:
    """Prove the frozen M7 bank is present, self-consistent and the right bank.

    Four separate things, because passing three of them is not enough:

    1.  every file in `recipes.bank.BANK_FILES` exists — the check whose failure
        produced `BankError: ... missing [...]` on the GPU host when this step
        was pointed at the C3 container;
    2.  the canonical `validate_bank` re-derives every hash in BANK_LOCK.json
        from the files beside it, so a tampered bank fails;
    3.  the lock status is `frozen`, not a bank still being built; and
    4.  the content identity equals the frozen project contract. Steps 2 and 3
        are satisfied by ANY internally consistent frozen bank, so only this one
        refuses a substituted bank — which is the defect this gate exists for.

    Raises `PreparationError(RECIPE_BANK_INVALID)`; never repairs, never writes.
    """
    from prism_fas.recipes.bank import (BANK_FILES, BANK_STATUS_FROZEN, BankError,
                                        load_bank, validate_bank)

    root = recipe_bank_root(repo)
    missing = [name for name in BANK_FILES if not (root / name).is_file()]
    if missing:
        raise PreparationError(
            RECIPE_BANK_INVALID,
            f"{root.as_posix()} is not a frozen recipe bank; missing {missing}. "
            "The M8 pair plan is bound to the frozen M7 bank; nothing here "
            "creates, converts or substitutes one.",
            {"bank_root": root.as_posix(), "missing": missing})
    try:
        bank = load_bank(root)
        report = validate_bank(root)
    except BankError as error:
        raise PreparationError(
            RECIPE_BANK_INVALID,
            f"the frozen recipe bank at {root.as_posix()} did not load: {error}",
            {"bank_root": root.as_posix()}) from error

    lock = bank["lock"]
    status, bank_id = lock.get("status"), str(lock.get("bank_id"))
    identity = str(lock.get("bank_content_identity_sha256"))
    failures: list[str] = []
    if not report.get("passed"):
        failures.append(f"validate_bank: {report.get('errors')}")
    if status != BANK_STATUS_FROZEN:
        failures.append(f"status {status!r} != {BANK_STATUS_FROZEN!r}")
    if bank_id != M7_BANK_ID:
        failures.append(f"bank_id {bank_id!r} != {M7_BANK_ID!r}")
    if identity != M7_BANK_CONTENT_IDENTITY:
        failures.append(f"content identity {identity} != the frozen contract "
                        f"{M7_BANK_CONTENT_IDENTITY}")
    if failures:
        raise PreparationError(
            RECIPE_BANK_INVALID,
            "the recipe bank does not satisfy the frozen project contract: "
            + "; ".join(failures),
            {"bank_root": root.as_posix(), "failures": failures,
             "observed_identity": identity, "expected_identity": M7_BANK_CONTENT_IDENTITY})

    return {"bank_root": root.as_posix(), "bank_id": bank_id, "status": status,
            "recipe_count": len(bank["recipes"]),
            "bank_content_identity_sha256": identity,
            "validated_by": "prism_fas.recipes.bank.validate_bank"}


def _pair_plan_is_current(output: Path, package_root: Path, bank_identity: str) -> bool:
    """Whether an existing plan was built from THESE inputs and finished writing.

    `PAIR_PLAN_LOCK.json` is written last, so its presence means the write
    completed — but not that it completed against the package and bank this run
    resolved. Both identities are stamped into every `pair_id`, so a plan built
    from different inputs is a different plan and must not be reused.
    """
    import json

    lock_path = output / "PAIR_PLAN_LOCK.json"
    manifests = ("pair_manifest_train.parquet", "pair_manifest_validation.parquet")
    if not lock_path.is_file() or any(not (output / name).is_file() for name in manifests):
        return False
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        package_lock = json.loads(
            (package_root / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (lock.get("recipe_bank_identity") == bank_identity
            and lock.get("package_identity") == package_lock.get("content_identity_sha256"))


def _step_pairs(repo: Path, *, resume: bool) -> StepOutcome:
    """The source-only GPAT pair plan C4 trains against.

    Two roots, both bound by frozen Version-B evidence rather than chosen here:
    the M3B package (`PAIR_PLAN_LOCK.package_identity` = b1cf29b6…9dc6) and the
    frozen M7 recipe bank (`recipe_bank_identity` = fa989938…10cb). The plan
    opens `manifests/source_train.parquet` and nothing else — `source_dev` and
    `target_test` are never read.
    """
    from prism_fas.synthesis import pair_plan

    bank = validate_recipe_bank(repo)
    output = repo / DERIVED_PACKAGES["gpat_pairs"]
    package_root = repo / DERIVED_PACKAGES[PAIR_PLAN_PACKAGE]

    if _pair_plan_is_current(output, package_root, bank["bank_content_identity_sha256"]):
        return StepOutcome("gpat_pairs", "REUSED_VALID",
                           "the GPAT pair plan is present, complete and built "
                           "from this package and this frozen bank",
                           {"pair_plan_identity": pair_plan.pair_plan_identity(output),
                            "recipe_bank": bank})

    result = pair_plan.write_pair_plan(package_root, recipe_bank_root(repo), output)
    return StepOutcome("gpat_pairs", "BUILT",
                       "built the source-only GPAT pair plan",
                       {"pair_plan_identity": pair_plan.pair_plan_identity(output),
                        "package_root": DERIVED_PACKAGES[PAIR_PLAN_PACKAGE],
                        "recipe_bank": bank,
                        "train_pairs": (result or {}).get("lock", {}).get("train_pairs"),
                        "validation_pairs": (result or {}).get("lock", {}).get("validation_pairs"),
                        "source_dev_opened": (result or {}).get("summary", {}).get("source_dev_opened"),
                        "target_test_opened": (result or {}).get("summary", {}).get("target_test_opened")})


def diagnose(repo: Path) -> dict[str, Any]:
    """Read-only forensics over whatever derived state this machine actually has.

    Written for a host whose partial layout nobody can assume: it names the
    legacy `m2a` work root separately from the canonical `full_preprocessing`
    root, counts what each holds, and says plainly whether the legacy tree can be
    adopted. It opens no dataset, resolves no target and writes nothing.
    """
    repo = Path(repo)
    report: dict[str, Any] = {"schema_version": "prism-preparation-diagnosis-v1",
                              "repo": repo.as_posix(), "scientific_eligible": False}
    try:
        paths, config = _paths(repo), _m2_config(repo)
    except Exception as error:                               # noqa: BLE001 - reported
        report["error"] = f"{type(error).__name__}: {error}"
        return report

    base = Path(paths.work_root) / "m2" / config.preprocessing_version / config.config_hash
    report["m2_config_root"] = base.as_posix()
    report["namespaces"] = sorted(item.name for item in base.iterdir()) if base.is_dir() else []

    legacy = base / "m2a"
    report["legacy_m2a"] = {
        "root": legacy.as_posix(),
        "present": legacy.is_dir(),
        "crops": _count_files(legacy / "crops", "*.jpg"),
        "result_files": sorted(item.name for item in (legacy / "results").glob("*.jsonl"))
                        if (legacy / "results").is_dir() else [],
        "reusable_as_m2_input": False,
        "why": "the legacy namespace holds JSONL results, not the canonical "
               "parquet manifests M3A reads, and `migrate_m2a` is contract-locked "
               "to the frozen 24/24/12/12/0 acceptance counts, so it cannot carry "
               "a full corpus across. It is left exactly where it is: nothing "
               "here deletes it, and nothing adopts it.",
    }

    root = m2_output_root(repo)
    manifests = root / "manifests"
    report["full_preprocessing"] = {
        "root": root.as_posix(),
        "present": root.is_dir(),
        "manifests_root": manifests.as_posix(),
        "manifests": {name: (_manifest_rows(manifests / f"{name}.parquet")
                             if (manifests / f"{name}.parquet").is_file() else None)
                      for name in M2_MANIFESTS},
        "crops_on_disk": _count_files(root / "crops", f"*.{config.output_image_format}"),
        "completion_marker": (root / M2_COMPLETION_MARKER).is_file(),
    }
    report["m2_status"] = _m2_status_or_absent(repo, True)

    # "locked" alone is what hid the M3B lifecycle defect: a package whose lock
    # says `building` is present, locked, and not scientific input. Every field
    # the operator needs to tell those apart is reported.
    report["packages"] = {
        key: {**package_status(repo / relative), "root": relative,
              "files": _count_files(repo / relative, "*")}
        for key, relative in (("m3a", DERIVED_PACKAGES["m3a"]),
                              ("m3b", DERIVED_PACKAGES["m3b"]))}
    report["gpat_pairs"] = _pair_plan_diagnosis(repo)
    processed = repo / "data" / "processed"
    report["data_processed"] = {
        "path": processed.as_posix(), "present": processed.is_dir(),
        "entries": sorted(item.name for item in processed.iterdir())[:20]
                   if processed.is_dir() else [],
        "role": "not an M2 root. Under the inherited convention it holds built "
                "PACKAGES; Version-C preparation writes packages to data/packages. "
                "No M2 manifest has ever lived here."}
    return report


def _pair_plan_diagnosis(repo: Path) -> dict[str, Any]:
    """Whether the pair plan on disk still belongs to the package beside it.

    Reported because M3B finalization RECOMPUTES the package content identity,
    and a plan built before that promotion is bound to an identity that no longer
    exists. It is rebuilt automatically; this is so the operator can see why.
    """
    import json

    output = repo / DERIVED_PACKAGES["gpat_pairs"]
    package = repo / DERIVED_PACKAGES[PAIR_PLAN_PACKAGE]
    state: dict[str, Any] = {
        "root": DERIVED_PACKAGES["gpat_pairs"],
        "present": output.is_dir(),
        "locked": (output / "PAIR_PLAN_LOCK.json").is_file(),
        "manifests_present": all((output / name).is_file() for name in
                                 ("pair_manifest_train.parquet",
                                  "pair_manifest_validation.parquet")),
        "package_identity": None, "recipe_bank_identity": None,
        "bound_to_current_package": False, "bound_to_frozen_bank": False,
        "reusable": False, "why": None,
    }
    if not state["locked"]:
        state["why"] = "no PAIR_PLAN_LOCK.json"
        return state
    try:
        lock = json.loads((output / "PAIR_PLAN_LOCK.json").read_text(encoding="utf-8"))
    except ValueError as error:
        state["why"] = f"PAIR_PLAN_LOCK.json does not parse: {error}"
        return state

    state["package_identity"] = lock.get("package_identity")
    state["recipe_bank_identity"] = lock.get("recipe_bank_identity")
    state["train_pairs"] = lock.get("train_pairs")
    state["validation_pairs"] = lock.get("validation_pairs")
    current = package_status(package)["content_identity_sha256"]
    state["current_package_identity"] = current
    state["bound_to_current_package"] = bool(current) and lock.get("package_identity") == current
    state["bound_to_frozen_bank"] = lock.get("recipe_bank_identity") == M7_BANK_CONTENT_IDENTITY
    state["reusable"] = bool(state["manifests_present"]
                             and state["bound_to_current_package"]
                             and state["bound_to_frozen_bank"])
    if not state["reusable"]:
        state["why"] = (
            "the pair manifests are incomplete" if not state["manifests_present"] else
            "it was built against a different recipe bank"
            if not state["bound_to_frozen_bank"] else
            "it was built against a different package identity — most likely the "
            "pre-finalization one, since finalizing M3B recomputes that identity. "
            "It is rebuilt automatically; no manual deletion is needed.")
    return state


def _count_files(root: Path, pattern: str) -> int:
    return sum(1 for item in root.rglob(pattern) if item.is_file()) if root.is_dir() else 0


def _report(started: float, outcomes: list[StepOutcome], needed: dict[str, Any], *,
            outcome: str, summary: str, reason_code: str | None = None,
            paths_config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "outcome": outcome,
        "summary": summary,
        "reason_code": reason_code,
        "steps": [item.as_dict() for item in outcomes],
        "needed": needed,
        # Operational provenance: which paths config the builders were handed, and
        # whether it had to be derived from this folder's location. Never an input
        # to a scientific identity.
        "paths_config": paths_config,
        "elapsed_seconds": round(time.time() - started, 2),
        "scientific_eligible": False,
        "meaning": "derived data preparation is deterministic engineering. It "
                   "produces no scientific evidence and selects nothing.",
    }


def write_report(repo: Path, report: dict[str, Any]) -> Path:
    from prism_fas.pipeline.state import atomic_write_json

    path = Path(repo) / "reports" / "preflight" / "DERIVED_DATA_PREPARATION.json"
    atomic_write_json(path, report)
    return path


__all__ = ["prepare", "what_is_needed", "write_report", "diagnose", "m2_status",
           "m2_output_root", "recipe_bank_root", "validate_recipe_bank",
           "package_status", "ensure_package_validated",
           "PACKAGE_STATUS_BUILDING", "PACKAGE_STATUS_VALIDATED",
           "PACKAGE_NOT_VALIDATED",
           "PreparationError", "SCHEMA_VERSION", "STEPS",
           "SOURCE_DATASETS", "M2_RUN_PROFILE", "M2_COMPLETION_MARKER",
           "PREPARATION_CONFIGS", "DERIVED_PACKAGES", "M7_RECIPE_BANK",
           "M7_BANK_CONTENT_IDENTITY", "M7_BANK_ID", "PAIR_PLAN_PACKAGE",
           "MISSING_RAW_DATA", "PREPARATION_FAILED", "M2_INCOMPLETE",
           "TARGET_IN_SOURCE_TREE", "RECIPE_BANK_INVALID"]
