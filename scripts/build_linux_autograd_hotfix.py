"""Build the runtime hotfix package for the deployed Linux RTX 5090 host.

The project folder on that machine is ~34 GB of raw datasets and frozen weights,
already copied. Two source files changed, so this assembles the minimum needed to
carry that change: the runtime files themselves in their project-relative
positions, a manifest that records what they replace, and instructions.

Git is the preferred delivery — `git fetch && git checkout <NEW_HEAD>` on that
host moves exactly these two files and nothing else. The copies here exist only
as the fallback for a host with no route to the remote.

Only files the RUNNING pipeline reads go in. Tests, docs, PROJECT_STATE, data,
weights and evidence stay out, and are listed in the manifest as deliberately
excluded so each omission is a recorded decision rather than a gap.

    python scripts/build_linux_autograd_hotfix.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "reports" / "handoff" / "LINUX_RTX5090_AUTOGRAD_HOTFIX"

#: The commit the deployed copy on the GPU host was taken from.
DEPLOYED_COMMIT = "1f2e24f"

#: Every runtime file the fix changes, with why it must travel.
RUNTIME_FILES: list[dict[str, object]] = [
    {
        "path": "src/prism_fas/pipeline/gpu_preflight.py",
        "reason": "the fix itself: the autograd probe now builds a real "
                  "DetectorBatch through batch_contract_for/audit_batch instead "
                  "of handing the detector a bare image tensor, runs the real "
                  "loss graph, and verifies scalar loss, per-parameter finite "
                  "gradients and the selected CUDA device before C4",
        "required": True,
        "scientific_identity_affected": False,
    },
    {
        "path": "src/prism_fas/evaluation/variant_audit.py",
        "reason": "the second, latent CUDA defect: the audit stub tower drew its "
                  "tokens on the host and returned them unmoved, so the region "
                  "path would have mixed CPU and CUDA tensors on the first real "
                  "forward. It now follows the input image's device. CPU values "
                  "are bit-identical — the draw still uses the same seeded CPU "
                  "generator and .to('cpu') is a no-op",
        "required": True,
        "scientific_identity_affected": False,
    },
]

#: Changed or added on this branch but deliberately NOT shipped, with the reason.
NOT_SHIPPED: list[dict[str, str]] = [
    {"path": "tests/pipeline/test_gpu_preflight_autograd.py",
     "category": "test",
     "why_not_shipped": "the regression suite for this fix; the GPU host runs the "
                        "pipeline, not pytest. It arrives with the git checkout "
                        "anyway and costs nothing there"},
    {"path": "tests/pipeline/test_preparation.py",
     "category": "test",
     "why_not_shipped": "carries the AUTOGRAD_FAILED stop-before-C4 test; same reason"},
    {"path": "scripts/build_linux_autograd_hotfix.py",
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
    """The file's content at the commit the GPU host was deployed from."""
    result = subprocess.run(["git", "show", f"{DEPLOYED_COMMIT}:{path}"],
                            cwd=REPO, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def build() -> dict[str, object]:
    # Rebuild the shipped copies from scratch; README_APPLY_HOTFIX.md is written
    # by hand and is not generated here, so it survives.
    shipped_root = PACKAGE / "src"
    if shipped_root.exists():
        for existing in sorted(shipped_root.rglob("*"), reverse=True):
            existing.unlink() if existing.is_file() else existing.rmdir()
        shipped_root.rmdir()
    PACKAGE.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for declared in RUNTIME_FILES:
        relative = str(declared["path"])
        source = REPO / relative
        # The target host is Linux. Ship LF exactly as Git stores it, so a manual
        # copy lands byte-identical to what the git checkout would write there.
        shipped = source.read_bytes().replace(b"\r\n", b"\n")
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
        "schema_version": "prism-linux-autograd-hotfix-v1",
        "title": "Linux RTX 5090 autograd preflight input-contract hotfix",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # The commit that introduced the fixed runtime files. This package is
        # built AFTER it and committed on top, so the branch head may be one
        # commit further along; both carry these exact bytes, which is what the
        # per-file sha256 below is for.
        "authoritative_commit": _git("rev-parse", "HEAD"),
        "runtime_files_introduced_in_commit": _git("rev-parse", "HEAD"),
        "previous_base_commit": _git("rev-parse", DEPLOYED_COMMIT),
        "previous_base_commit_short": DEPLOYED_COMMIT,
        "file_count": len(entries),
        "line_endings": "LF, as a Linux checkout writes them. Each `sha256` is "
                        "over the exact bytes shipped.",
        "preferred_deployment": {
            "method": "git",
            "commands": ["git fetch origin",
                         "git checkout <authoritative_commit>"],
            "why": "moves exactly the changed files and leaves data, weights, "
                   ".venv and caches untouched",
        },
        "full_project_recopy_required": False,
        "dataset_recopy_required": False,
        "weights_recopy_required": False,
        "venv_recopy_required": False,
        "cache_recopy_required": False,
        "defects_fixed": [
            {
                "id": "AUTOGRAD_PROBE_WRONG_INPUT_CONTRACT",
                "symptom": "[AUTOGRAD_FAILED] the representative model could not "
                           "complete a forward/backward step: AttributeError: "
                           "'Tensor' object has no attribute 'image'",
                "root_cause": "the preflight built the real PRISMDetector and then "
                              "called it with torch.randn(2,3,224,224). "
                              "PRISMDetector.forward takes a DetectorBatch and "
                              "reads batch.image and batch.region_priors, so the "
                              "probe had invented an input contract the trainer "
                              "does not use",
                "fix": "the probe now builds its batch through the real contract "
                       "— batch_contract_for('G5', M9TrainingConfig(...)) then "
                       "audit_batch(variant, contract).to(device) — runs the real "
                       "loss graph via compute_losses, and asserts a scalar finite "
                       "loss, a finite gradient on every trainable parameter, and "
                       "that batch, output, loss, parameters and gradients all sit "
                       "on the selected CUDA device",
                "scientific_model_changed": False,
            },
            {
                "id": "AUDIT_STUB_TOWER_STAYED_ON_THE_HOST",
                "symptom": "latent; it would have been the next failure once the "
                           "input contract was correct",
                "root_cause": "_StubTower drew its tokens with a seeded CPU "
                              "generator and returned them unmoved, so a model on "
                              "CUDA would have concatenated host tensors with "
                              "device tensors inside region_embeddings",
                "fix": "the draw still uses the same seeded CPU generator, then "
                       "moves to pixel_values.device. CPU values are unchanged",
                "scientific_model_changed": False,
            },
        ],
        "preflight_contract": {
            "runs_before_stage": "C4",
            "on_failure": "AUTOGRAD_FAILED, then 'Stopped BEFORE C4. No scientific "
                          "work was started.', exit BLOCKED",
            "proves": ["forward on the selected CUDA device",
                       "a scalar finite loss from the real loss graph",
                       "backward completes",
                       "every trainable parameter receives a gradient",
                       "every gradient is finite",
                       "batch, output, loss, parameters and gradients are on the "
                       "selected CUDA device",
                       "no silent CPU fallback"],
            "writes": "nothing. The checkpoint round-trip probe writes one "
                      "temporary file under state/preflight and deletes it; the "
                      "autograd probe writes nothing at all",
            "opens_dataset": False,
            "resolves_target": False,
        },
        "scientific_impact": {
            "protocols_or_constants_changed": False,
            "scientific_identity_affected": False,
            "frozen_configs_changed": False,
            "c3_banks_changed": False,
            "c4_to_c13_scientific_status": "NOT_RUN",
            "scientific_execution": "NOT_RUN",
            "target_access": 0,
        },
        "must_not_be_touched": MUST_NOT_BE_TOUCHED,
        "not_shipped": NOT_SHIPPED,
        "package_own_files": ["HOTFIX_MANIFEST.json", "README_APPLY_HOTFIX.md"],
        "expected_next_command": "python train.py",
        "files": entries,
    }

    # Every file this package ships, hashed, including its own documentation.
    # The manifest is excluded because it cannot contain its own hash; Git carries
    # its identity instead.
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
