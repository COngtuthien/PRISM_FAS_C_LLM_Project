from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Any
from prism_fas.config.models import DatasetDefinition
from prism_fas.data.adapters.adapters import adapter_for
from prism_fas.utils.core import atomic_json_write, sha256_file, stable_json_hash
def audit_dataset(definition: DatasetDefinition, root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"dataset": definition.dataset, "root": str(root), "exists": root.exists(), "errors": []}
    if not root.is_dir(): report["errors"].append("raw root missing"); return report
    files=sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p:p.as_posix())
    report.update(file_count=len(files), extensions=dict(Counter(p.suffix.lower() or "<none>" for p in files)), duplicate_paths=0)
    try:
        records=adapter_for(definition, root).records(); report["record_count"]=len(records)
        if definition.dataset == "siw_mv2": report.update(target_label_isolation=True, label_distribution="withheld")
        else: report["label_distribution"]=dict(Counter(r.label for r in records))
        keys=[(r.official_split, r.video_id) for r in records]; report["split_overlap"] = len(keys) != len(set(keys)); report["video_overlap"] = len([r.video_id for r in records]) != len({r.video_id for r in records})
        if definition.dataset != "siw_mv2":
            subjects: dict[str, set[str]] = {}
            for record in records:
                if record.subject_id is not None: subjects.setdefault(record.subject_id, set()).add(record.official_split or "<none>")
            report["subject_overlap"] = any(len(splits) > 1 for splits in subjects.values()); report["subject_count"] = len(subjects)
    except (ValueError, FileNotFoundError) as exc: report["errors"].append(str(exc)); report["record_count"]=0
    report["dataset_fingerprint"]=stable_json_hash([(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in files])
    return report
def write_audits(reports_root: Path, reports: list[dict[str, Any]]) -> Path:
    target=reports_root / "raw_audit"; target.mkdir(parents=True, exist_ok=True)
    for report in reports: atomic_json_write(target / f"{report['dataset']}.json", report)
    summary={"datasets": reports, "error_count": sum(len(r["errors"]) for r in reports)}; atomic_json_write(target / "summary.json", summary)
    (target / "summary.md").write_text("# Raw audit\n\n" + "\n".join(f"- {r['dataset']}: {r['file_count']} files; errors={len(r['errors'])}" for r in reports) + "\n", encoding="utf-8")
    (target / "errors.jsonl").write_text("".join(f'{{"dataset":"{r["dataset"]}","error":{e!r}}}\n' for r in reports for e in r["errors"]), encoding="utf-8")
    return target
