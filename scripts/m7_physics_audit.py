"""Run the real M7 source_train-live physics preview and determinism audit.

CPU only. No Modal, no GPU, no SSH, no network. `source_dev` and `target_test`
are never opened. Equivalent to:

    python -m prism_fas.cli.main synthesis physics-audit \
      --package-root data/processed/prism_data_v1_m3b \
      --bank assets/recipe_banks/prism_recipe_bank_m7_v1 \
      --config configs/synthesis/physics_m7.yaml --output reports/m7
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from prism_fas.synthesis.audit import run_audit

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=ROOT / "data" / "processed" / "prism_data_v1_m3b")
    parser.add_argument("--bank", type=Path, default=ROOT / "assets" / "recipe_banks" / "prism_recipe_bank_m7_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "synthesis" / "physics_m7.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "m7")
    parser.add_argument("--limit", type=int, default=None, help="development smoke only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_audit(args.package_root, args.bank, args.config, args.output, limit=args.limit,
                       dry_run=args.dry_run, progress=lambda payload: print(json.dumps({"progress": payload}), flush=True))
    if result["status"] == "dry_run":
        print(json.dumps({"status": "dry_run", "plan": result["plan"], "selected_samples": result["selected_samples"],
                          "planned_pairs": result["planned_pairs"]}, indent=2))
        return 0
    print(json.dumps({"status": result["status"], "preview_rows": result["preview_rows"],
                      "physics": result["physics"]["checks"],
                      "outside_mask_max_abs_error": result["physics"]["outside_mask_max_abs_error"],
                      "determinism_passed": result["determinism"]["passed"],
                      "source_isolation_passed": result["source_isolation"]["passed"],
                      "written": result["written"]}, indent=2))
    return 0 if (result["physics"]["passed"] and result["determinism"]["passed"]
                 and result["source_isolation"]["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
