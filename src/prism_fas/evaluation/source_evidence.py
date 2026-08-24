"""Real C8 row evidence, read back off disk for the C9 source freeze.

C9's refusal logic is a pure function of the evidence SHAPE, which is why the
rehearsal can exercise it against synthetic rows and learn something real. What
the rehearsal cannot do is tell C9 what actually ran — and a scientific
SOURCE_MATRIX_LOCK_C built over constructed rows would be a freeze over an
experiment nobody performed.

So this module is the only way a scientific C9 obtains evidence. It reads what
C8 wrote and nothing else:

* `reports/full/c8/C8_ACCEPTANCE.json`, which must exist and record acceptance;
* one `run_manifest.json` per PLANNED row, at the deterministic path C8's own
  scheduler addresses;
* the checkpoint and calibration each manifest names, re-hashed rather than
  trusted.

Three refusals are structural rather than advisory. A manifest that declares
itself `fixture_backed` is refused, so a rehearsal artifact copied into the
scientific tree cannot become evidence. A checkpoint whose bytes no longer hash
to what its manifest recorded is refused, because the frozen thing must be the
thing on disk. And a manifest whose `run_identity` disagrees with the plan's is
refused, because it describes a different run.

Nothing here fabricates a row. A planned row with no manifest is simply absent
from the returned evidence, and `source_lock.audit` reports it as missing —
which is the correct behaviour, and is what a synthesized placeholder would have
hidden.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from prism_fas.evaluation.source_lock import RowEvidence

SCHEMA_VERSION = "prism-c8-source-evidence-v1"

#: Where a scientific C8 writes. Not configurable: evidence C9 freezes lives in
#: the scientific namespace or it is not the evidence C9 means.
C8_RUNS = "runs/full/c8"
C8_REPORTS = "reports/full/c8"
ACCEPTANCE = "C8_ACCEPTANCE.json"
RUN_MANIFEST = "run_manifest.json"


class SourceEvidenceError(RuntimeError):
    """C8's evidence is absent, fixture-backed or does not match its own record."""

    reason_code = "C8_EVIDENCE_INVALID"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def row_directory(runs_root: Path, row: Any) -> Path:
    """The deterministic path C8's scheduler addresses this row by.

    Derived the same way in both modules on purpose: if C8 wrote somewhere else,
    C9 must fail to find the row rather than discover it by scanning, because a
    scan would also find rows the plan never declared.
    """
    return (Path(runs_root) / row.protocol / row.experiment_id /
            row.config_identity[:12] / str(row.seed))


def acceptance_report(repo: Path, *, reports_root: str = C8_REPORTS) -> dict[str, Any]:
    """C8's own verdict over its completed matrix, or a refusal naming its absence."""
    path = Path(repo) / reports_root / ACCEPTANCE
    if not path.is_file():
        raise SourceEvidenceError(
            f"{path.as_posix()} is absent; C9 freezes a matrix C8 accepted, and "
            "there is no substitute for C8's own acceptance verdict")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SourceEvidenceError(f"{path.as_posix()} is not readable JSON: {error}") from error
    if payload.get("fixture_backed") is not False:
        raise SourceEvidenceError(
            f"{path.as_posix()} does not declare fixture_backed=false; a rehearsal's "
            "acceptance may never govern a scientific freeze")
    if payload.get("scientific_eligible") is not True:
        raise SourceEvidenceError(
            f"{path.as_posix()} is not scientifically eligible "
            f"(execution_profile={payload.get('execution_profile')!r})")
    return payload


def load_row_evidence(repo: Path, plan: Any, *, runs_root: str = C8_RUNS,
                      verify_bytes: bool = True) -> tuple[list[RowEvidence],
                                                          list[dict[str, Any]]]:
    """Every planned row's real evidence, plus the problems found reading it.

    Returns `(evidence, problems)` rather than raising, because C9's job is to
    report every reason the freeze is refused in one pass. A row that could not
    be read contributes no `RowEvidence`, so `source_lock.audit` reports it as
    missing and the freeze is refused for the right reason.
    """
    root = Path(repo) / runs_root
    evidence: list[RowEvidence] = []
    problems: list[dict[str, Any]] = []

    for row in plan.rows:
        directory = row_directory(root, row)
        path = directory / RUN_MANIFEST
        if not path.is_file():
            problems.append({"row_id": row.row_id, "problem": "MANIFEST_ABSENT",
                             "path": path.relative_to(Path(repo)).as_posix()})
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            problems.append({"row_id": row.row_id, "problem": "MANIFEST_UNREADABLE",
                             "detail": str(error)})
            continue

        row_problems = _manifest_problems(repo, row, manifest, directory,
                                          verify_bytes=verify_bytes)
        if row_problems:
            problems.extend({"row_id": row.row_id, **item} for item in row_problems)
            continue

        checkpoint = dict(manifest.get("checkpoint") or {})
        calibration = dict(manifest.get("calibration") or {})
        evidence.append(RowEvidence(
            row_id=str(manifest["row_id"]),
            run_identity=str(manifest["run_identity"]),
            config_identity=str(manifest["config_identity"]),
            status=str(manifest["status"]),
            checkpoint_sha256=checkpoint.get("sha256"),
            calibration_sha256=calibration.get("calibration_hash"),
            calibration_hash=calibration.get("calibration_hash"),
            decision_logit_name=str(manifest.get("decision_logit_name", "")),
            decision_score_name=str(manifest.get("decision_score_name", "")),
            parent_identities={key: str(value) for key, value
                               in dict(manifest.get("parent_identities") or {}).items()},
            metrics=dict((manifest.get("metrics") or {}).get("source_dev", {}).get(
                "ranking_tuple", {})),
            notes=str(manifest.get("reason", ""))))
    return evidence, problems


def _manifest_problems(repo: Path, row: Any, manifest: dict[str, Any], directory: Path,
                       *, verify_bytes: bool) -> list[dict[str, Any]]:
    """Every way one manifest can fail to be evidence for the row it claims."""
    problems: list[dict[str, Any]] = []

    if manifest.get("fixture_backed") is not False:
        problems.append({"problem": "FIXTURE_BACKED",
                         "detail": "the manifest does not declare fixture_backed=false"})
    if manifest.get("row_id") != row.row_id:
        problems.append({"problem": "ROW_ID_MISMATCH",
                         "detail": f"manifest names {manifest.get('row_id')!r}"})
    if manifest.get("run_identity") != row.run_identity:
        problems.append({"problem": "RUN_IDENTITY_MISMATCH",
                         "detail": f"manifest records {manifest.get('run_identity')!r}, "
                                   f"the plan declares {row.run_identity!r}"})
    if manifest.get("config_identity") != row.config_identity:
        problems.append({"problem": "CONFIG_IDENTITY_MISMATCH",
                         "detail": f"manifest records {manifest.get('config_identity')!r}"})

    status = str(manifest.get("status"))
    if status != "PASS":
        # Not a defect in the manifest — a real failed row is evidence and is
        # returned as such. `source_lock.audit` is what refuses to freeze over it.
        return problems

    checkpoint = dict(manifest.get("checkpoint") or {})
    calibration = dict(manifest.get("calibration") or {})
    if not checkpoint.get("sha256"):
        problems.append({"problem": "CHECKPOINT_IDENTITY_ABSENT"})
    if not calibration.get("calibration_hash"):
        problems.append({"problem": "CALIBRATION_HASH_ABSENT"})
    if calibration.get("split") != "source_dev":
        problems.append({"problem": "CALIBRATION_NOT_SOURCE_DEV",
                         "detail": f"split={calibration.get('split')!r}"})
    if int(manifest.get("target_labels_resolved", -1)) != 0:
        problems.append({"problem": "TARGET_LABELS_RESOLVED",
                         "detail": str(manifest.get("target_labels_resolved"))})

    if verify_bytes and checkpoint.get("path") and checkpoint.get("sha256"):
        path = Path(repo) / str(checkpoint["path"])
        if not path.is_file():
            problems.append({"problem": "CHECKPOINT_MISSING",
                             "detail": str(checkpoint["path"])})
        elif _sha256_file(path) != checkpoint["sha256"]:
            problems.append({"problem": "CHECKPOINT_MOVED",
                             "detail": "the checkpoint on disk no longer hashes to what "
                                       "its manifest recorded"})
    calibration_file = directory / "calibration.json"
    if not calibration_file.is_file():
        problems.append({"problem": "CALIBRATION_ARTIFACT_MISSING",
                         "detail": calibration_file.name})
    return problems


def evidence_report(repo: Path, plan: Any, *, runs_root: str = C8_RUNS,
                    reports_root: str = C8_REPORTS) -> dict[str, Any]:
    """Everything a scientific C9 needs, or a named refusal. Never partial silently."""
    try:
        acceptance = acceptance_report(repo, reports_root=reports_root)
    except SourceEvidenceError as error:
        return {"available": False, "reason_code": error.reason_code,
                "error": str(error), "acceptance": None, "evidence": [],
                "problems": [], "rows_found": 0, "rows_planned": len(plan.rows)}

    evidence, problems = load_row_evidence(repo, plan, runs_root=runs_root)
    return {
        "available": True, "reason_code": "", "error": "",
        "schema_version": SCHEMA_VERSION,
        "acceptance": {key: acceptance.get(key) for key in
                       ("accepted", "matrix_identity", "rows_declared", "rows_terminal",
                        "rows_passed", "rows_failed", "hidden_rows", "missing_rows",
                        "c7_detector_config_sha256", "c6_selector_identity_sha256",
                        "source_package_identity", "target_access")},
        "acceptance_accepted": acceptance.get("accepted") is True,
        "evidence": evidence,
        "problems": problems,
        "rows_found": len(evidence),
        "rows_planned": len(plan.rows),
        "runs_root": runs_root,
        "loaded_by": "prism_fas.evaluation.source_evidence.load_row_evidence",
    }


__all__ = ["SCHEMA_VERSION", "C8_RUNS", "C8_REPORTS", "ACCEPTANCE", "RUN_MANIFEST",
           "SourceEvidenceError", "row_directory", "acceptance_report",
           "load_row_evidence", "evidence_report"]
