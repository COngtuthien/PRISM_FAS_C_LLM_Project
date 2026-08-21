"""The M2 producer and the M3A consumer, run for real against each other.

The defect this file exists to stop from recurring: `_step_m2` called the legacy
`m2_runner.run`, which writes JSONL results and crops under
`<work_root>/m2/<version>/<hash>/m2a/`, while `_step_m3a` called
`build_package(paths.processed_root, ...)`, whose `load_m2_samples` reads
`<input_root>/manifests/source_frames.parquet`. Nothing wrote that file. A real
RTX 5090 host spent hours on SCRFD and then died with

    FileNotFoundError: <project>/data/processed/manifests/source_frames.parquet

The existing preparation tests could not see it, because their stub `m2_run`
created `data/processed/<dataset>/` — the stub wrote where the consumer read, so
producer and consumer agreed only inside the test. Nothing here stubs the write
location. `run_preprocessing`, `ManifestRepository`, the routing converters and
`load_m2_samples` are all the real code; only the SCRFD session is replaced,
because a face detector is not what this contract is about.

Everything is source-only: the fixture datasets are CASIA and MSU, SiW-Mv2 is
never constructed, and a test asserts the target never enters the tree.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prism_fas.pipeline import preparation

pytest.importorskip("cv2")


# --- a source corpus small enough to preprocess for real ---------------------

FRAMES_PER_VIDEO = 4


def _frame(index: int) -> np.ndarray:
    return np.full((240, 320, 3), 40 + index * 10, dtype=np.uint8)


def _image_sequence(root: Path, video: str, count: int = 6) -> Path:
    """A CASIA-style PNG sequence the real `ImageSequenceReader` accepts.

    Its anchor filename grammar is `[bf]?s<digits>v<id>f<digits>.png`, so the
    fixture writes files the production reader will actually open rather than a
    shape the reader would refuse.
    """
    root.mkdir(parents=True, exist_ok=True)
    import cv2

    for index in range(1, count + 1):
        cv2.imwrite(str(root / f"s1v{video}f{index}.png"), _frame(index))
    return root / f"s1v{video}f1.png"


def _video_file(path: Path, count: int = 6) -> Path:
    """An MSU-style video the real `OpenCVVideoDecoder` can seek and decode."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    if not writer.isOpened():                # pragma: no cover - codec-dependent
        pytest.skip("this OpenCV build cannot write an MJPG test video")
    for index in range(1, count + 1):
        writer.write(_frame(index))
    writer.release()
    return path


def _record(dataset: str, video: str, path: Path, label: str) -> Any:
    from prism_fas.data.schemas.records import CanonicalVideoRecord

    return CanonicalVideoRecord(
        dataset=dataset, subject_id=video.split("_")[0], video_id=video,
        source_path=path, official_split="train", label=label,
        adapter_version="1.0", source_fingerprint=f"fp-{video}",
        metadata_provenance="fixture")


class _FixedFaceDetector:
    """A deterministic stand-in for the SCRFD session.

    The M2 contract is about where manifests and crops land, not about whether a
    face was found. One face, always, in the same place, so every crop is
    reproducible and the sample accounting is exact.
    """

    name = "scrfd"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.model_sha256 = "f" * 64
        self.provider = "CPUExecutionProvider"

    def detect(self, image: np.ndarray) -> list[Any]:
        from prism_fas.data.preprocess_m2 import Detection

        return [Detection(bbox=(80.0, 60.0, 240.0, 200.0), score=0.99,
                          landmarks=[(120.0, 100.0), (200.0, 100.0), (160.0, 140.0),
                                     (130.0, 170.0), (190.0, 170.0)])]


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A repo whose source records are real media and whose detector is fixed."""
    from prism_fas.data import preprocess_m2

    # A throwaway project that carries the real `configs/`, so every config the
    # preparation path reads is the project's own and nothing is written into the
    # developer's tree.
    source_repo = Path(__file__).resolve().parents[2]
    project = tmp_path / "p"   # short: the M2 root nests a 64-char config hash
    (project / "data").mkdir(parents=True)
    shutil.copytree(source_repo / "configs", project / "configs")
    repo = project

    records: dict[str, list[Any]] = {
        # CASIA is an image sequence and MSU is a video file, exactly as
        # `run_preprocessing` decides from the dataset name. Both readers are the
        # production ones; the fixture writes media they can really open.
        "casia_fasd": [
            _record("casia_fasd", f"casia{index}",
                    _image_sequence(project / "raw" / "casia_fasd" / f"v{index}",
                                    f"casia{index}"),
                    "live" if index % 2 == 0 else "spoof")
            for index in range(2)],
        "msu_mfsd": [
            _record("msu_mfsd", f"msu{index}",
                    _video_file(project / "raw" / "msu_mfsd" / f"v{index}.avi"),
                    "live" if index % 2 == 0 else "spoof")
            for index in range(2)],
    }

    paths_config = project / "paths.yaml"
    paths_config.write_text(json.dumps({
        "workspace_root": project.as_posix(), "project_root": project.as_posix(),
        "raw_datasets": {name: (project / "raw" / name).as_posix()
                         for name in ("casia_fasd", "msu_mfsd", "siw_mv2")},
        "model_cache": (project / "weights").as_posix(),
        "work_root": (project / "data" / "work").as_posix(),
        "processed_root": (project / "data" / "processed").as_posix(),
        "package_root": (project / "data" / "packages").as_posix(),
        "runs_root": (project / "runs").as_posix(),
        "reports_root": (project / "reports").as_posix()}), encoding="utf-8")

    monkeypatch.setattr(preparation, "_paths_config", lambda _repo: paths_config)
    monkeypatch.setattr(preparation, "_records",
                        lambda _repo, dataset: list(records[dataset]))
    # Only the ONNX session is replaced. The real frozen SCRFD file is still
    # resolved and hashed, because `validate_full_profile` checks that hash
    # against the pinned digest and a fake detector file would fail it — which is
    # exactly the guarantee we want left switched on.
    monkeypatch.setattr(preprocess_m2, "SCRFDDetector", _FixedFaceDetector)
    return {"repo": repo, "project": project, "records": records,
            "config_path": paths_config}


def _prepare_m2(corpus: dict[str, Any], *, resume: bool = True) -> Any:
    return preparation._step_m2(corpus["repo"], resume=resume)


# --- 1-2. the producer writes where the consumer reads -----------------------

def test_m2_writes_the_manifests_m3a_reads(corpus) -> None:
    """The whole defect, in one assertion pair."""
    from prism_fas.data.package.builder import load_m2_samples

    outcome = _prepare_m2(corpus)
    root = preparation.m2_output_root(corpus["repo"])

    assert outcome.action == "BUILT"
    assert (root / "manifests" / "source_frames.parquet").is_file()
    assert (root / "manifests" / "source_crops.parquet").is_file()
    # The consumer's own loader, unmodified, against the producer's own root.
    samples = load_m2_samples(root)
    assert len(samples) == 4 * FRAMES_PER_VIDEO
    assert {sample["dataset"] for sample in samples} == set(preparation.SOURCE_DATASETS)


def test_the_crop_paths_resolve_from_the_root_m3a_is_given(corpus) -> None:
    """`crop_relative_path` is relative to the M2 output root, which is why the
    consumer must be handed that root and not some other directory."""
    from prism_fas.data.package.builder import load_m2_samples

    _prepare_m2(corpus)
    root = preparation.m2_output_root(corpus["repo"])

    for sample in load_m2_samples(root):
        crop = root / sample["crop_relative_path"]
        assert crop.is_file(), sample["crop_relative_path"]
        assert not Path(sample["crop_relative_path"]).is_absolute()


def test_the_producer_and_the_consumer_resolve_the_same_root(
        corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    """`_step_m3a` must hand `build_package` the root `_step_m2` wrote to."""
    from prism_fas.data import package

    _prepare_m2(corpus)
    seen: dict[str, Path] = {}

    def spy(input_root: Path, package_root: Path, config: Any, **kwargs: Any) -> dict:
        seen["input_root"] = Path(input_root)
        raise RuntimeError("stop after the argument that matters")

    monkeypatch.setattr(package, "build_package", spy)
    monkeypatch.setattr(package, "load_package_config", lambda path: {"path": str(path)})
    with pytest.raises(RuntimeError, match="stop after"):
        preparation._step_m3a(corpus["repo"], resume=True)

    assert seen["input_root"] == preparation.m2_output_root(corpus["repo"])
    assert seen["input_root"].name == preparation.M2_RUN_PROFILE


def test_m2_then_m3a_runs_end_to_end_with_neither_side_stubbed(corpus) -> None:
    """The chain that failed on the RTX 5090, executed for real from raw media to
    a validated M3A package. No stub stands between the producer and consumer."""
    outcome_m2 = _prepare_m2(corpus)
    outcome_m3a = preparation._step_m3a(corpus["repo"], resume=True)

    package_root = corpus["repo"] / "data" / "packages" / "prism_data_v1_m3a"
    assert outcome_m2.action == "BUILT"
    assert outcome_m3a.action == "BUILT"
    assert outcome_m3a.detail["status"] == "validated"
    assert len(outcome_m3a.detail["samples"]) == 4 * FRAMES_PER_VIDEO

    lock = json.loads((package_root / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))
    # The lock hard-codes `full_preprocessing` as its validation profile, so the
    # namespace it records has to be that same directory — which is only true
    # when M3A was handed the M2 profile root.
    assert lock["source_m2_namespace"] == preparation.M2_RUN_PROFILE
    assert lock["source_m2_validation_profile"] == "full_preprocessing"
    assert set(lock["source_m2_manifest_sha256"]) >= {"source_frames", "source_crops"}
    assert lock["per_split_counts"].get("target_test") is None, (
        "a source-only preparation cannot package a target split")


def test_the_canonical_root_is_never_data_processed(corpus) -> None:
    """The old consumer path, named so a regression to it is unmistakable."""
    root = preparation.m2_output_root(corpus["repo"])
    processed = Path(preparation._paths(corpus["repo"]).processed_root)

    assert root != processed
    assert "m2a" not in root.parts
    assert root.name == "full_preprocessing"


# --- 3-6. incomplete trees are not complete trees ----------------------------

def test_a_non_empty_but_incomplete_tree_is_not_reused(corpus) -> None:
    """The rule this replaces was `the directory is non-empty`."""
    root = preparation.m2_output_root(corpus["repo"])
    (root / "crops" / "casia_fasd").mkdir(parents=True)
    (root / "crops" / "casia_fasd" / "stray.jpg").write_bytes(b"junk")

    status = preparation.m2_status(corpus["repo"])
    assert status["complete"] is False
    assert status["reason"] == "MANIFESTS_ABSENT"


@pytest.mark.parametrize("manifest", ["source_frames", "source_crops"])
def test_a_missing_manifest_is_never_complete(corpus, manifest: str) -> None:
    _prepare_m2(corpus)
    root = preparation.m2_output_root(corpus["repo"])
    (root / "manifests" / f"{manifest}.parquet").unlink()

    status = preparation.m2_status(corpus["repo"])
    assert status["complete"] is False
    assert status["reason"] == "MANIFESTS_ABSENT"
    assert status["manifests_present"][manifest] is False


def test_an_interrupted_run_is_incomplete_and_says_which_records_remain(
        corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    """One dataset preprocessed, the process died before the second."""
    only_casia = dict(corpus["records"])
    monkeypatch.setattr(preparation, "_records",
                        lambda _repo, dataset: list(only_casia[dataset])
                        if dataset == "casia_fasd" else [])
    _prepare_m2(corpus)

    monkeypatch.setattr(preparation, "_records",
                        lambda _repo, dataset: list(corpus["records"][dataset]))
    status = preparation.m2_status(corpus["repo"])

    assert status["complete"] is False
    assert status["reason"] in ("RECORDS_OUTSTANDING", "COMPLETION_MARKER_STALE")
    assert status["outstanding_records"]["msu_mfsd"] == 2


def test_a_stale_completion_marker_is_not_believed(corpus) -> None:
    """The corpus grew; the marker describes a smaller one."""
    _prepare_m2(corpus)
    root = preparation.m2_output_root(corpus["repo"])
    marker_path = root / preparation.M2_COMPLETION_MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["record_counts"]["casia_fasd"] += 1
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    status = preparation.m2_status(corpus["repo"])
    assert status["complete"] is False
    assert status["reason"] == "COMPLETION_MARKER_STALE"


def test_a_tree_that_fails_the_canonical_validator_is_not_complete(corpus) -> None:
    """A crop whose bytes changed under a manifest that still names its hash."""
    _prepare_m2(corpus)
    root = preparation.m2_output_root(corpus["repo"])
    crop = next(iter((root / "crops").rglob("*.jpg")))
    crop.write_bytes(crop.read_bytes() + b"tampered")

    status = preparation.m2_status(corpus["repo"], deep=True)
    assert status["complete"] is False
    assert status["reason"] == "VALIDATION_FAILED"
    assert "crop.sha" in status["validation"]["errors"]


# --- 7-9. resume keeps the work that survived --------------------------------

def test_completed_records_are_reused_not_reprocessed(
        corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    """The RTX 5090 case: hours of SCRFD already done must not be redone."""
    monkeypatch.setattr(preparation, "_records",
                        lambda _repo, dataset: list(corpus["records"][dataset])
                        if dataset == "casia_fasd" else [])
    _prepare_m2(corpus)
    first = preparation.m2_status(corpus["repo"])["counts"]["source_crops"]

    monkeypatch.setattr(preparation, "_records",
                        lambda _repo, dataset: list(corpus["records"][dataset]))
    outcome = _prepare_m2(corpus, resume=True)

    assert outcome.action == "BUILT"
    per_dataset = outcome.detail["per_dataset"]
    assert per_dataset["casia_fasd"]["records_walked"] == 0, (
        "records already in the manifests must not be walked again")
    assert per_dataset["msu_mfsd"]["records_walked"] == 2
    after = preparation.m2_status(corpus["repo"], deep=True)
    assert after["complete"] is True
    assert after["counts"]["source_crops"] == first + 2 * FRAMES_PER_VIDEO


def test_partial_work_survives_a_failed_run(corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing deletes a partial tree to make room for a rebuild."""
    monkeypatch.setattr(preparation, "_records",
                        lambda _repo, dataset: list(corpus["records"][dataset])
                        if dataset == "casia_fasd" else [])
    _prepare_m2(corpus)
    root = preparation.m2_output_root(corpus["repo"])
    crops_before = sorted(path.name for path in (root / "crops").rglob("*.jpg"))

    def explode(_repo: Path, dataset: str) -> list[Any]:
        if dataset == "msu_mfsd":
            raise RuntimeError("synthetic adapter failure")
        return list(corpus["records"][dataset])

    monkeypatch.setattr(preparation, "_records", explode)
    with pytest.raises(RuntimeError, match="synthetic adapter failure"):
        _prepare_m2(corpus)

    assert sorted(path.name for path in (root / "crops").rglob("*.jpg")) == crops_before


def test_a_tree_the_validator_refuses_stops_and_deletes_nothing(corpus) -> None:
    """More preprocessing cannot repair a crop whose bytes no longer match its
    hash, so the step says so instead of quietly rebuilding over it."""
    _prepare_m2(corpus)
    root = preparation.m2_output_root(corpus["repo"])
    crop = next(iter((root / "crops").rglob("*.jpg")))
    crop.write_bytes(crop.read_bytes() + b"tampered")
    before = sorted(path.name for path in (root / "crops").rglob("*.jpg"))

    with pytest.raises(preparation.PreparationError) as raised:
        _prepare_m2(corpus)

    assert raised.value.reason == preparation.M2_INCOMPLETE
    assert "VALIDATION_FAILED" in str(raised.value)
    assert "will not repair it" in str(raised.value)
    assert "crop.sha" in raised.value.detail["failed_checks"]
    assert sorted(path.name for path in (root / "crops").rglob("*.jpg")) == before, (
        "a refused tree must not be deleted to make room for a rebuild")


def test_a_complete_validated_tree_is_reused(corpus) -> None:
    _prepare_m2(corpus)
    outcome = _prepare_m2(corpus)

    assert outcome.action == "REUSED_VALID"
    assert outcome.detail["validation"]["passed"] is True


def test_a_second_pass_changes_nothing(corpus) -> None:
    """Idempotent: re-running over a complete tree rewrites no crop."""
    _prepare_m2(corpus)
    root = preparation.m2_output_root(corpus["repo"])
    before = {path: path.read_bytes() for path in sorted((root / "crops").rglob("*.jpg"))}

    _prepare_m2(corpus)

    assert {path: path.read_bytes() for path in sorted((root / "crops").rglob("*.jpg"))} == before


# --- 10. M3A starts only after M2 validates ----------------------------------

def test_m3a_refuses_an_incomplete_m2_tree(corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    from prism_fas.data import package

    monkeypatch.setattr(package, "build_package",
                        lambda *args, **kwargs: pytest.fail(
                            "M3A must not build from an unvalidated M2 tree"))
    with pytest.raises(preparation.PreparationError) as raised:
        preparation._step_m3a(corpus["repo"], resume=True)

    assert raised.value.reason == preparation.M2_INCOMPLETE
    assert "MANIFESTS_ABSENT" in str(raised.value)


def test_m3a_refuses_a_tree_whose_crops_were_tampered_with(
        corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    from prism_fas.data import package

    _prepare_m2(corpus)
    crop = next(iter((preparation.m2_output_root(corpus["repo"]) / "crops").rglob("*.jpg")))
    crop.write_bytes(b"")

    monkeypatch.setattr(package, "build_package",
                        lambda *args, **kwargs: pytest.fail("must not reach the builder"))
    with pytest.raises(preparation.PreparationError) as raised:
        preparation._step_m3a(corpus["repo"], resume=True)

    assert raised.value.reason == preparation.M2_INCOMPLETE


# --- 11-13. the target firewall and fail-closed ------------------------------

def test_no_target_dataset_is_ever_preprocessed(corpus) -> None:
    _prepare_m2(corpus)
    status = preparation.m2_status(corpus["repo"], deep=True)

    assert preparation.SOURCE_DATASETS == ("casia_fasd", "msu_mfsd")
    assert "siw_mv2" not in preparation.SOURCE_DATASETS
    assert status["counts"]["target_frames"] == 0
    assert status["counts"]["target_crops"] == 0
    assert status["validation"]["target_isolation_passed"] is True


def test_no_siw_artifact_appears_anywhere_in_the_m2_tree(corpus) -> None:
    _prepare_m2(corpus)
    root = preparation.m2_output_root(corpus["repo"])

    offenders = [path for path in root.rglob("*") if "siw" in path.name.lower()]
    assert offenders == []


def test_target_rows_in_the_source_tree_stop_the_run(
        corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    """Not repairable by preprocessing more; it stops with its own reason code."""
    _prepare_m2(corpus)
    monkeypatch.setattr(preparation, "m2_status",
                        lambda *args, **kwargs: {
                            "complete": False, "reason": preparation.TARGET_IN_SOURCE_TREE,
                            "root": "x", "counts": {"target_frames": 1}})
    with pytest.raises(preparation.PreparationError) as raised:
        _prepare_m2(corpus)

    assert raised.value.reason == preparation.TARGET_IN_SOURCE_TREE


def test_the_marker_records_that_the_pass_was_source_only(corpus) -> None:
    _prepare_m2(corpus)
    marker = json.loads(
        (preparation.m2_output_root(corpus["repo"]) / preparation.M2_COMPLETION_MARKER)
        .read_text(encoding="utf-8"))

    assert marker["source_only"] is True
    assert marker["target_datasets_preprocessed"] == []
    assert marker["scientific_eligible"] is False
    assert sorted(marker["datasets"]) == sorted(preparation.SOURCE_DATASETS)


# --- forensics ---------------------------------------------------------------

def test_the_diagnosis_separates_the_two_namespaces(corpus) -> None:
    _prepare_m2(corpus)
    report = preparation.diagnose(corpus["repo"])

    assert report["full_preprocessing"]["present"] is True
    assert report["full_preprocessing"]["manifests"]["source_crops"] == 4 * FRAMES_PER_VIDEO
    assert report["full_preprocessing"]["completion_marker"] is True
    assert report["legacy_m2a"]["reusable_as_m2_input"] is False
    assert report["m2_status"]["complete"] is True
    assert report["scientific_eligible"] is False


def test_the_diagnosis_reports_a_legacy_tree_without_adopting_it(corpus) -> None:
    """The remote's existing `m2a` work: named, counted, and left alone."""
    root = preparation.m2_output_root(corpus["repo"])
    legacy = root.parent / "m2a"
    (legacy / "crops" / "casia_fasd").mkdir(parents=True)
    (legacy / "crops" / "casia_fasd" / "a.jpg").write_bytes(b"legacy")
    (legacy / "results").mkdir(parents=True)
    (legacy / "results" / "casia_fasd.jsonl").write_text("{}\n", encoding="utf-8")

    report = preparation.diagnose(corpus["repo"])

    assert report["legacy_m2a"]["present"] is True
    assert report["legacy_m2a"]["crops"] == 1
    assert report["legacy_m2a"]["result_files"] == ["casia_fasd.jsonl"]
    assert report["legacy_m2a"]["reusable_as_m2_input"] is False
    assert (legacy / "crops" / "casia_fasd" / "a.jpg").is_file(), (
        "the diagnosis is read-only; it must never remove a legacy artifact")
    # The legacy namespace is a sibling, so it cannot contaminate the profile root.
    assert "m2a" not in preparation.m2_output_root(corpus["repo"]).parts
