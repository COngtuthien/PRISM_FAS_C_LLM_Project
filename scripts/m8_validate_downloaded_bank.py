"""Validate a downloaded M8 synthetic bank on Windows.

Extracts the single deterministic transport archive (M6 showed a Windows
directory download of thousands of small files is not reliable), then runs the
same validator the remote build ran, against the real local `source_train`
package and the frozen M7 recipe bank.

Source-only: `source_dev`, `target_test`, target labels and raw dataset paths are
never opened.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from prism_fas.synthesis.synthetic_export import extract_archive
from prism_fas.synthesis.synthetic_validation import validate_bank, write_validation_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=None, help="downloaded <bank_id>.tar")
    parser.add_argument("--destination", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--bank-root", type=Path, default=None, help="skip extraction and validate this directory")
    parser.add_argument("--package-root", type=Path, default=ROOT / "data" / "processed" / "prism_data_v1_m3b")
    parser.add_argument("--recipe-bank", type=Path, default=ROOT / "assets" / "recipe_banks" / "prism_recipe_bank_m7_v1")
    parser.add_argument("--expected-identity", type=str, default=None,
                        help="the remote bank_content_identity_sha256; the local bank must equal it")
    parser.add_argument("--expected-archive-sha256", type=str, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "m8" / "local_downloaded_bank_validation.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.archive is None and args.bank_root is None:
        parser.error("either --archive or --bank-root is required")

    archive_report: dict = {"status": "skipped"}
    if args.archive is not None:
        import hashlib
        digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
        archive_report = {"archive_name": args.archive.name, "archive_bytes": args.archive.stat().st_size,
                          "archive_sha256": digest,
                          "archive_sha256_matches": (args.expected_archive_sha256 is None
                                                     or digest == args.expected_archive_sha256)}
        if not archive_report["archive_sha256_matches"]:
            print(json.dumps({"passed": False, "reason": "downloaded archive SHA-256 mismatch",
                              **archive_report}, indent=2))
            return 1
        if args.dry_run:
            print(json.dumps({"status": "dry_run", "written": [], **archive_report}, indent=2))
            return 0
        archive_report["extraction"] = extract_archive(args.archive, args.destination)

    bank_root = Path(args.bank_root or archive_report["extraction"]["bank_root"])
    report = validate_bank(bank_root, package_root=args.package_root, recipe_bank_root=args.recipe_bank)
    matches = args.expected_identity is None or report["bank_content_identity_sha256"] == args.expected_identity
    payload = {**report, "transport": archive_report, "expected_identity": args.expected_identity,
               "local_identity_equals_remote": bool(matches),
               "local_bank_root_name": bank_root.name}
    if not args.dry_run:
        write_validation_report(args.output, payload)
    print(json.dumps({"passed": bool(report["passed"] and matches), "bank_id": report["bank_id"],
                      "bank_content_identity_sha256": report["bank_content_identity_sha256"],
                      "local_identity_equals_remote": bool(matches), "counts": report["counts"],
                      "operational_minimums_passed": report["operational_minimums"]["passed"],
                      "shards_passed": report["shard_report"]["passed"],
                      "error_count": report["error_count"], "errors": report["errors"][:10],
                      "written": [] if args.dry_run else [str(args.output.name)]}, indent=2))
    return 0 if (report["passed"] and matches) else 1


if __name__ == "__main__":
    raise SystemExit(main())
