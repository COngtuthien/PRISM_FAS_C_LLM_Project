"""C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 closure CLI — read-only status and
interpretation registration over an ALREADY, GENUINELY, scientifically
executed V2 diagnostics result.

    python -m prism_fas.evaluation.post_failure_diagnostics_v2_closure --repo . --status
    python -m prism_fas.evaluation.post_failure_diagnostics_v2_closure --repo . --register-interpretation

This is NOT a diagnostics runner: it never loads a checkpoint, never
forwards an image, and never recomputes a diagnostic metric.
`--register-interpretation` runs
`post_failure_diagnostics_v2.validate_existing_diagnostics_result` first and
refuses (BLOCKED) if that validation fails; it then derives a bounded,
arithmetic-only interpretation from the RECORDED `per_test` values via
`post_failure_diagnostics_v2_interpretation.derive_full_interpretation`,
hashes the four already-validated result artifacts (real `sha256` of bytes
already on disk — never fabricated), and writes
`DIAGNOSTICS_INTERPRETATION.json`/`.md` exactly once. A second call is
idempotent on an exact match and BLOCKS on any conflicting existing
interpretation — it never overwrites one.

Exit codes: 0 = OK (interpretation registered or already valid/registered),
2 = BLOCKED, 3 = USAGE error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

EXIT_PASS, EXIT_BLOCKED, EXIT_USAGE = 0, 2, 3

from prism_fas.evaluation.post_failure_diagnostics_v2 import DIAGNOSTICS_DIR  # noqa: E402

INTERPRETATION_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_INTERPRETATION.json"
INTERPRETATION_MD_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_INTERPRETATION.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism_fas.evaluation.post_failure_diagnostics_v2_closure",
        description="Read-only status and interpretation registration over an already "
                    "scientifically executed C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 result. "
                    "Never loads a checkpoint, forwards an image, or recomputes a metric.")
    parser.add_argument("--repo", default=".", type=Path,
                        help="repository root (default: current directory)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--register-interpretation", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- Markdown rendering -------------------------------------------------

def _render_markdown_impl(document: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 — interpretation\n")
    lines.append(f"- diagnostics protocol identity: `{document.get('diagnostics_protocol_identity')}`")
    lines.append(f"- code commit: `{document.get('code_commit')}`")
    lines.append(f"- c8 matrix identity: `{document.get('c8_matrix_identity')}`")
    lines.append(f"- checkpoint binding identity: `{document.get('checkpoint_binding_identity')}`")
    lines.append(f"- population binding identity: `{document.get('population_binding_identity')}`")
    lines.append(f"- source package identity: `{document.get('source_package_identity')}`")
    lines.append(f"- c6 bank identities: `{document.get('c6_bank_identities')}`")
    lines.append(f"- overall diagnostics verdict: **{document.get('overall_diagnostics_verdict')}**")
    lines.append(f"- ba_sep observed verdict: `{document.get('ba_sep_observed_verdict')}`")
    lines.append(f"- detector reliability lock c observed overall: "
                 f"`{document.get('detector_reliability_lock_c_observed_overall')}`")
    lines.append(f"- c9_may_close: `{document.get('c9_may_close')}`")
    lines.append(f"- target_access: `{document.get('target_access')}`")
    lines.append("\n## Result file SHA256\n")
    for name, digest in sorted((document.get("result_file_sha256") or {}).items()):
        lines.append(f"- `{name}`: `{digest}`")

    interpretation = document.get("interpretation") or {}
    lines.append("\n## Per-test interpretation\n")
    for test_id, test_doc in sorted((interpretation.get("tests") or {}).items()):
        lines.append(f"\n### `{test_id}`\n")
        lines.append(f"- classification: `{test_doc.get('classification')}`")
        if "test_verdict" in test_doc:
            lines.append(f"- test verdict: **{test_doc.get('test_verdict')}**")
        for arm, arm_doc in sorted((test_doc.get("per_arm") or {}).items()):
            observed = arm_doc.get("observed", {})
            derived = arm_doc.get("derived_arithmetic", {})
            lines.append(f"\n**{arm}** — OBSERVED: {observed}")
            lines.append(f"  DERIVED_ARITHMETIC: {derived}")
            lines.append(f"  criterion_failed: `{arm_doc.get('criterion_failed')}`")
        if test_doc.get("observed") and "per_arm" not in test_doc:
            lines.append(f"- OBSERVED: {test_doc.get('observed')}")
        lines.append("\nINTERPRETATION:")
        for statement in test_doc.get("interpretation") or []:
            lines.append(f"- {statement}")
        lines.append("\nNOT_SUPPORTED:")
        for statement in test_doc.get("not_supported") or []:
            lines.append(f"- {statement}")

    lines.append("\n## Global interpretation\n")
    for statement in interpretation.get("global_interpretation") or []:
        lines.append(f"- {statement}")
    lines.append("\n## Global NOT_SUPPORTED\n")
    for statement in interpretation.get("global_not_supported") or []:
        lines.append(f"- {statement}")
    lines.append("")
    return "\n".join(lines)


# ==============================================================================
# --status
# ==============================================================================

def _status(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import post_failure_diagnostics_v2 as diag2

    validation = diag2.validate_existing_diagnostics_result(repo)
    report: dict[str, Any] = {
        "diagnostics_result_valid": validation["valid"],
        "diagnostics_result_problems": validation["problems"],
        "interpretation_registered": (Path(repo) / INTERPRETATION_PATH).is_file(),
        "checkpoint_weights_loaded": False, "images_forwarded": False,
        "diagnostic_metric_recomputed": False,
        "c9_may_close": False, "target_access": 0,
    }
    if not validation["valid"]:
        report["reason"] = "NO_VALID_DIAGNOSTICS_RESULT"
        return EXIT_BLOCKED, report
    if not report["interpretation_registered"]:
        report["reason"] = "INTERPRETATION_NOT_YET_REGISTERED"
        return EXIT_BLOCKED, report
    return EXIT_PASS, report


# ==============================================================================
# --register-interpretation
# ==============================================================================

def _register_interpretation(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import post_failure_diagnostics_v2 as diag2
    from prism_fas.evaluation import post_failure_diagnostics_v2_interpretation as interp
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {
        "registered": False, "checkpoint_weights_loaded": False, "images_forwarded": False,
        "diagnostic_metric_recomputed": False, "target_access": 0,
    }

    validation = diag2.validate_existing_diagnostics_result(repo)
    if not validation["valid"]:
        report.update({"error": "EXISTING_RESULT_FAILED_VALIDATION",
                      "problems": validation["problems"]})
        return EXIT_BLOCKED, report

    docs = validation["docs"]
    per_test_doc, provenance, verdict_doc = docs["per_test"], docs["provenance"], docs["verdict"]

    try:
        result_file_sha256 = {
            name: hashlib.sha256((Path(repo) / relative).read_bytes()).hexdigest()
            for name, relative in diag2.RESULT_ARTIFACT_PATHS.items()}
    except OSError as error:
        report["error"] = f"could not hash a validated result artifact: {error}"
        return EXIT_BLOCKED, report

    interpretation_body = interp.derive_full_interpretation(per_test_doc.get("per_test") or {})

    document = {
        "schema_version": "c9-post-failure-diagnostics-v2-interpretation-v1",
        "diagnostics_protocol_identity": provenance.get("protocol_identity"),
        "code_commit": provenance.get("code_commit"),
        "c8_matrix_identity": provenance.get("c8_matrix_identity"),
        "checkpoint_binding_identity": provenance.get("checkpoint_binding_identity"),
        "population_binding_identity": provenance.get("population_binding_identity"),
        "source_package_identity": provenance.get("source_package_identity"),
        "c6_bank_identities": provenance.get("c6_bank_identities"),
        "result_file_sha256": result_file_sha256,
        "per_test": per_test_doc.get("per_test"),
        "overall_diagnostics_verdict": verdict_doc.get("overall_diagnostics_verdict"),
        "ba_sep_observed_verdict": "FAIL",
        "detector_reliability_lock_c_observed_overall": "FAILED",
        "c9_may_close": False,
        "target_access": 0,
        "interpretation": interpretation_body,
    }

    path = Path(repo) / INTERPRETATION_PATH
    existing = _read_json(path)
    if existing is not None:
        if existing != document:
            report["error"] = ("an existing DIAGNOSTICS_INTERPRETATION.json differs from the "
                               "one just derived from the current, validated result; refusing "
                               "to overwrite a prior registration")
            return EXIT_BLOCKED, report
        report.update({"registered": True, "reused": True, "written": False,
                      "interpretation_path": INTERPRETATION_PATH})
        return EXIT_PASS, report

    atomic_write_json(path, document)
    md_path = Path(repo) / INTERPRETATION_MD_PATH
    md_path.write_text(_render_markdown_impl(document), encoding="utf-8")
    report.update({"registered": True, "reused": False, "written": True,
                  "interpretation_path": INTERPRETATION_PATH,
                  "interpretation_md_path": INTERPRETATION_MD_PATH})
    return EXIT_PASS, report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.status:
        exit_code, payload = _status(args.repo)
    else:
        exit_code, payload = _register_interpretation(args.repo)

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
