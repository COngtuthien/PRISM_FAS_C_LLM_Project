"""Build the runtime hotfix package for the deployed Linux RTX 5090 host.

The project folder on that machine is ~34 GB of raw datasets and frozen weights,
plus whatever partial preprocessing survived the failed run. Four source files
changed, so this assembles the minimum needed to carry that change: the runtime
files in their project-relative positions, a manifest recording what they
replace, and instructions.

Git is the preferred delivery — `git fetch && git checkout <NEW_HEAD>` moves
exactly these four files and touches no data, no weights and no `.venv`. The
copies here exist only as the fallback for a host with no route to the remote.

Only files the RUNNING pipeline reads go in. Tests, docs, PROJECT_STATE, data,
weights and evidence stay out, and are listed as deliberately excluded so each
omission is a recorded decision rather than a gap.

    python scripts/build_m2_m3a_contract_hotfix.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "reports" / "handoff" / "LINUX_RTX5090_M2_M3A_CONTRACT_HOTFIX"

#: The commit the deployed copy on the GPU host was taken from.
DEPLOYED_COMMIT = "787b9b2"

RUNTIME_FILES: list[dict[str, object]] = [
    {
        "path": "src/prism_fas/pipeline/preparation.py",
        "reason": "the fix itself: `_step_m2` now drives the production "
                  "full_preprocessing path (PreprocessingRunContext + "
                  "run_preprocessing) instead of the legacy m2a CLI helper; "
                  "`_step_m3a` is handed the same root through `m2_output_root`; "
                  "M2 completion is measured by content via the canonical "
                  "validate_full_profile plus record coverage plus a marker "
                  "written last; completed records are resumed rather than "
                  "reprocessed; and `_step_m3a` validates, finalizes, then "
                  "re-validates in the order the canonical CLI uses",
        "required": True,
        "scientific_identity_affected": False,
    },
    {
        "path": "src/prism_fas/data/run_context.py",
        "reason": "`build_preprocessing_run_context` moved here from cli/main.py "
                  "so the pre-C4 preparation path can reach the one canonical "
                  "context constructor without importing the CLI. That it could "
                  "not is why it drifted onto the legacy runner in the first place",
        "required": True,
        "scientific_identity_affected": False,
    },
    {
        "path": "src/prism_fas/cli/main.py",
        "reason": "imports `build_preprocessing_run_context` from its new home; "
                  "the function body is unchanged, so `prism data preprocess run "
                  "--run-profile full_preprocessing` behaves identically",
        "required": True,
        "scientific_identity_affected": False,
    },
    {
        "path": "train.py",
        "reason": "adds `--diagnose-data`, a read-only forensic view of the M2 "
                  "namespaces, manifest row counts, crop counts and packages on "
                  "this machine. Nothing else in the entrypoint changed",
        "required": True,
        "scientific_identity_affected": False,
    },
]

NOT_SHIPPED: list[dict[str, str]] = [
    {"path": "tests/pipeline/test_m2_m3a_contract.py",
     "category": "test",
     "why_not_shipped": "the unstubbed producer/consumer integration suite for "
                        "this fix; the GPU host runs the pipeline, not pytest. It "
                        "arrives with the git checkout anyway and costs nothing"},
    {"path": "tests/pipeline/test_preparation.py",
     "category": "test",
     "why_not_shipped": "the orchestration suite, rewired to the new producer; "
                        "same reason"},
    {"path": "tests/pipeline/test_checks_and_firewall.py",
     "category": "test",
     "why_not_shipped": "allows the new --diagnose-data helper in train.py; same reason"},
    {"path": "tests/pipeline/test_gpu_hotfix_package.py",
     "category": "test",
     "why_not_shipped": "compares a delivered handoff package against the commit "
                        "it names rather than the working tree; same reason"},
    {"path": "scripts/build_m2_m3a_contract_hotfix.py",
     "category": "tooling",
     "why_not_shipped": "builds this package; nothing on the GPU host calls it"},
    {"path": "docs/PROJECT_STATE.md",
     "category": "state",
     "why_not_shipped": "derived handoff state, not runtime input"},
]

MUST_NOT_BE_TOUCHED = ["data/", "weights/", "assets/", "runs/", "reports/",
                       "state/", ".venv/", ".git/"]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()


def _previous_bytes(path: str) -> bytes | None:
    result = subprocess.run(["git", "show", f"{DEPLOYED_COMMIT}:{path}"],
                            cwd=REPO, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def build() -> dict[str, object]:
    shipped_root = PACKAGE / "src"
    if shipped_root.exists():
        for existing in sorted(shipped_root.rglob("*"), reverse=True):
            existing.unlink() if existing.is_file() else existing.rmdir()
        shipped_root.rmdir()
    (PACKAGE / "train.py").unlink(missing_ok=True)
    PACKAGE.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for declared in RUNTIME_FILES:
        relative = str(declared["path"])
        # The target host is Linux. Ship LF exactly as Git stores it, so a manual
        # copy lands byte-identical to what the git checkout would write.
        shipped = (REPO / relative).read_bytes().replace(b"\r\n", b"\n")
        destination = PACKAGE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(shipped)

        previous = _previous_bytes(relative)
        entries.append({
            "destination": relative,
            "sha256": _sha256(shipped),
            "bytes": len(shipped),
            "reason": declared["reason"],
            "required_on_gpu_host": declared["required"],
            "previous_sha256_lf": _sha256(previous) if previous is not None else None,
            "previous_bytes": len(previous) if previous is not None else None,
            "previous_sha256_source":
                f"git {DEPLOYED_COMMIT}:{relative}, the commit the deployed copy "
                "was taken from. Both hashes are over LF bytes, which is what a "
                "Linux checkout writes and what you would measure on that host.",
            "scientific_identity_affected": declared["scientific_identity_affected"],
        })

    manifest: dict[str, object] = {
        "schema_version": "prism-m2-m3a-contract-hotfix-v1",
        "title": "Linux RTX 5090 M2 to M3A derived-data contract hotfix",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "authoritative_commit": _git("rev-parse", "HEAD"),
        "runtime_files_introduced_in_commit": _git("rev-parse", "HEAD"),
        "previous_base_commit": _git("rev-parse", DEPLOYED_COMMIT),
        "previous_base_commit_short": DEPLOYED_COMMIT,
        "file_count": len(entries),
        "line_endings": "LF, as a Linux checkout writes them. Each `sha256` is "
                        "over the exact bytes shipped.",
        "preferred_deployment": {
            "method": "git",
            "commands": ["cd /home/sparc/workdir/longnm/PRISM_FAS_C_LLM_Project",
                         "git fetch origin",
                         "git checkout <authoritative_commit>",
                         "/usr/bin/python3 train.py"],
            "why": "moves exactly the changed files and leaves data, weights, "
                   ".venv, caches and partial preprocessing untouched",
        },
        "full_project_recopy_required": False,
        "dataset_recopy_required": False,
        "weights_recopy_required": False,
        "venv_recopy_required": False,
        "cache_recopy_required": False,
        "manual_cleanup_required": False,
        "defects_fixed": [
            {
                "id": "M2_M3A_PRODUCER_CONSUMER_PATH_MISMATCH",
                "symptom": "[PREPARATION_FAILED] derived-data preparation failed "
                           "at m3a_package: FileNotFoundError: "
                           "<project>/data/processed/manifests/source_frames.parquet",
                "producer_before": "preparation._step_m2 -> m2_runner.run(...), "
                                   "which writes JSONL results and crops under "
                                   "<work_root>/m2/<version>/<config_hash>/m2a/ "
                                   "and never writes a parquet manifest",
                "consumer_before": "preparation._step_m3a -> "
                                   "build_package(paths.processed_root, ...) -> "
                                   "load_m2_samples reads "
                                   "<input_root>/manifests/{source,target}_"
                                   "{frames,crops}.parquet",
                "root_cause": "two independent path expressions for one artifact. "
                              "data/processed has never held an M2 manifest under "
                              "any convention this project has used: in the "
                              "inherited Version-B layout it holds built PACKAGES, "
                              "and Version-C preparation writes packages to "
                              "data/packages",
                "fix": "both sides now resolve the location through the single "
                       "function preparation.m2_output_root(repo), which delegates "
                       "to the project's own run_profiles.profile_root under the "
                       "full_preprocessing profile. The producer is the production "
                       "run_preprocessing + PreprocessingRunContext pair",
                "why_tests_missed_it": "the preparation suite stubbed "
                                       "m2_runner.run with a fake that created "
                                       "data/processed/<dataset>/ — the stub wrote "
                                       "where the consumer read, so producer and "
                                       "consumer agreed only inside the test. "
                                       "build_package was stubbed too, so "
                                       "load_m2_samples never ran",
            },
            {
                "id": "M2_COMPLETENESS_WAS_DIRECTORY_PRESENCE",
                "symptom": "latent; a partial or interrupted tree would have been "
                           "reported REUSED_VALID and consumed by M3A",
                "root_cause": "_step_m2 returned REUSED_VALID when data/processed "
                              "existed and was non-empty, which is equally true of "
                              "a tree that died halfway through its first dataset",
                "fix": "completion is now measured: all five canonical manifests "
                       "present, no target rows, frame/crop counts equal, every "
                       "canonical source record walked, a completion marker whose "
                       "config hash / detector hash / record counts still match, "
                       "and the canonical validate_full_profile passing",
            },
            {
                "id": "M3A_VALIDATED_BEFORE_ITS_LOCK_WAS_FINALIZED",
                "symptom": "found by running the chain end to end for the first "
                           "time: the M3A build failed on check `lock.status`",
                "root_cause": "build_package writes PACKAGE_LOCK.json with "
                              "status 'building'; _step_m3a then called "
                              "validate_package(root) with require_validated_status "
                              "defaulting to True, which can never pass before "
                              "finalize_lock runs",
                "fix": "validate loose, finalize, validate strict — the order the "
                       "canonical CLI uses",
            },
        ],
        "m2_contract": {
            "run_profile": "full_preprocessing",
            "output_root": "<work_root>/m2/<preprocessing_version>/"
                           "<config_hash>/full_preprocessing",
            "resolved_by": "prism_fas.pipeline.preparation.m2_output_root",
            "producer": "prism_fas.data.m2_runner.run_preprocessing",
            "consumer": "prism_fas.data.package.builder.build_package",
            "validator": "prism_fas.data.m2_validation.validate_full_profile "
                         "(m2f1a-full-v1)",
            "completion_marker": "state/M2_PREPARATION_COMPLETE.json",
            "source_datasets": ["casia_fasd", "msu_mfsd"],
            "target_datasets": [],
        },
        "legacy_m2a": {
            "reusable_as_m2_input": False,
            "why": "the legacy namespace holds JSONL results, not the canonical "
                   "parquet manifests M3A reads, and migrate_m2a is contract-locked "
                   "to the frozen 24/24/12/12/0 acceptance counts, so it cannot "
                   "carry a full corpus across. Nothing deletes it and nothing "
                   "adopts it; `python3 train.py --diagnose-data` reports it.",
            "consequence": "the SCRFD detection the failed run performed into the "
                           "m2a namespace is not reused. The full profile redoes "
                           "it into its own namespace, and from then on resumes.",
        },
        "scientific_impact": {
            "protocols_or_constants_changed": False,
            "scientific_identity_affected": False,
            "frozen_configs_changed": False,
            "c3_banks_changed": False,
            "scrfd_policy_changed": False,
            "datasets_changed": False,
            "c4_to_c13_scientific_status": "NOT_RUN",
            "scientific_execution": "NOT_RUN",
            "target_access": 0,
        },
        "must_not_be_touched": MUST_NOT_BE_TOUCHED,
        "not_shipped": NOT_SHIPPED,
        "package_own_files": ["HOTFIX_MANIFEST.json", "README_APPLY_HOTFIX.md"],
        "expected_next_command": "/usr/bin/python3 train.py",
        "files": entries,
    }

    manifest["handoff_files"] = [
        {"path": path.relative_to(PACKAGE).as_posix(),
         "sha256": _sha256(path.read_bytes()),
         "bytes": path.stat().st_size}
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file() and path.name != "HOTFIX_MANIFEST.json"
    ]
    manifest["handoff_files_note"] = (
        "HOTFIX_MANIFEST.json is absent from this list because a manifest cannot "
        "carry its own hash. Its identity is the Git blob at authoritative_commit.")

    (PACKAGE / "HOTFIX_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    manifest = build()
    print(f"{PACKAGE.relative_to(REPO).as_posix()}: {manifest['file_count']} runtime file(s)")
    for entry in manifest["files"]:                          # type: ignore[index]
        print(f"  {entry['destination']:44s} {entry['sha256'][:16]}  {entry['bytes']} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
