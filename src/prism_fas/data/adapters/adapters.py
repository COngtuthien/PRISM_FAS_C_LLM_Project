from __future__ import annotations
import hashlib, re
from pathlib import Path
from prism_fas.config.models import DatasetDefinition
from prism_fas.data.adapters.base import DatasetAdapter
from prism_fas.data.schemas.records import CanonicalVideoRecord, TargetInferenceRecord
from prism_fas.utils.core import sha256_file, stable_json_hash
class ExplicitRuleAdapter(DatasetAdapter):
    def _files(self):
        base = self.root / self.definition.root_relative
        if not base.is_dir(): raise FileNotFoundError(f"Dataset layout root does not exist: {base}")
        found = set()
        for glob in self.definition.include_globs:
            found.update(p for p in base.glob(glob) if p.is_file())
        return base, sorted(found, key=lambda p: p.as_posix())
class CasiaFasdAdapter(ExplicitRuleAdapter):
    def records(self) -> list[CanonicalVideoRecord]:
        base, files = self._files(); pattern = re.compile(self.definition.path_pattern)
        grouped: dict[tuple[str,str,str,str], list[Path]] = {}
        for file in files:
            match = pattern.fullmatch(file.relative_to(base).as_posix())
            if not match: raise ValueError(f"CASIA path does not match explicit rule: {file}")
            g = match.groupdict(); key = (g["split"], g["label"], g["subject_id"], g["video_id"]); grouped.setdefault(key, []).append(file)
        # The same s<subject>v<video> stem exists in both official splits, so the
        # split is part of the canonical record identity (as it is for MSU).
        records=[CanonicalVideoRecord(dataset="casia_fasd", subject_id=k[2], video_id=f"{self.definition.splits[k[0]]}_s{k[2]}v{k[3]}", source_path=paths[0], official_split=self.definition.splits[k[0]], label=self.definition.labels[k[1]], adapter_version=self.definition.adapter_version, source_fingerprint=stable_json_hash([str(p) for p in paths]), metadata_provenance="explicit YAML path_pattern") for k, paths in sorted(grouped.items())]
        seen=set()
        for record in records:
            if record.video_id in seen: raise ValueError(f"Duplicate CASIA video id: {record.video_id}")
            seen.add(record.video_id)
        return records
class MsuMfsdAdapter(ExplicitRuleAdapter):
    def records(self) -> list[CanonicalVideoRecord]:
        base, files=self._files(); required={"train","test"}
        if set(self.definition.protocol_files) != required: raise ValueError("MSU requires explicit train/test protocol files")
        subject_splits: dict[str,str]={}
        for split, rel in self.definition.protocol_files.items():
            path=base/rel
            if not path.is_file(): raise FileNotFoundError(f"MSU protocol file missing: {path}")
            for subject in path.read_text(encoding="utf-8").split():
                if not re.fullmatch(r"\d{2}", subject): raise ValueError(f"Invalid MSU subject in protocol {path}: {subject!r}")
                subject=subject.zfill(3)
                if subject in subject_splits: raise ValueError(f"MSU split overlap for subject {subject}")
                subject_splits[subject]=split
        patterns={label:re.compile(rule) for label, rule in self.definition.filename_patterns.items()}
        records=[]; seen=set()
        for file in files:
            rel=file.relative_to(base).as_posix(); matches=[(label, pat.fullmatch(rel)) for label, pat in patterns.items()]; matches=[x for x in matches if x[1]]
            if len(matches) != 1: raise ValueError(f"MSU file does not match exactly one explicit filename rule: {file}")
            kind, match=matches[0]; data=match.groupdict(); subject=data["subject_id"]
            if subject not in subject_splits: raise ValueError(f"MSU subject missing from official protocol: {subject}")
            video_id=file.stem
            if video_id in seen: raise ValueError(f"Duplicate MSU video id: {video_id}")
            seen.add(video_id); capture={k:v for k,v in data.items() if k not in {"subject_id"}}
            records.append(CanonicalVideoRecord(dataset="msu_mfsd", subject_id=subject, video_id=video_id, source_path=file, official_split=self.definition.splits[subject_splits[subject]], label=self.definition.labels[kind], capture_metadata=capture, adapter_version=self.definition.adapter_version, protocol_version="MSU-MFSD README train_sub_list/test_sub_list", source_fingerprint=stable_json_hash([str(file.relative_to(base)), file.stat().st_size, file.stat().st_mtime_ns]), metadata_provenance="MSU README.txt naming protocol + train_sub_list.txt/test_sub_list.txt; YAML provenance"))
        return sorted(records, key=lambda r:r.video_id)
def opaque_record_id(dataset: str, relative_path: str) -> str:
    """Deterministic, root-independent identifier for a target record.

    Derived from the dataset-root-relative POSIX path so it is stable across
    runs, processes and machines, and carries no class, subject or filename
    information.  hashlib is used deliberately: hash() is not process-stable.
    """
    digest=hashlib.sha256(dataset.encode()+b"\0"+relative_path.encode()).hexdigest()
    return f"{dataset.split('_')[0]}_{digest[:16]}"
class SiWMv2Adapter(ExplicitRuleAdapter):
    def _opaque(self, file: Path, base: Path, seen: dict[str,str]) -> str:
        relative=file.relative_to(base).as_posix(); identifier=opaque_record_id("siw_mv2",relative)
        if seen.setdefault(identifier,relative) != relative: raise ValueError(f"Duplicate SiW-Mv2 opaque record id: {identifier}")
        return identifier
    def inference_records(self) -> list[TargetInferenceRecord]:
        base, files = self._files(); pattern = re.compile(self.definition.path_pattern)
        records=[]; seen: dict[str,str]={}
        for file in files:
            match=pattern.fullmatch(file.relative_to(base).as_posix())
            if not match: raise ValueError(f"SiW-Mv2 path does not match explicit rule: {file}")
            records.append(TargetInferenceRecord(dataset="siw_mv2", video_id=self._opaque(file,base,seen), source_path=file, official_split=self.definition.official_split or "target_test", adapter_version=self.definition.adapter_version, source_fingerprint=sha256_file(file), metadata_provenance="explicit YAML path_pattern; labels withheld"))
        return records
    def private_evaluation_records(self) -> list[CanonicalVideoRecord]:
        base, files=self._files(); pattern=re.compile(self.definition.path_pattern); records=[]; seen: dict[str,str]={}
        for file in files:
            match=pattern.fullmatch(file.relative_to(base).as_posix()); assert match is not None
            # Same opaque id as the inference record so evaluators can join
            # predictions without ever handling the private filename.
            records.append(CanonicalVideoRecord(dataset="siw_mv2", subject_id=None, video_id=self._opaque(file,base,seen), source_path=file, official_split=self.definition.official_split, label="live" if match["private_label"] == "Live" else "spoof", adapter_version=self.definition.adapter_version, source_fingerprint=sha256_file(file), metadata_provenance="PRIVATE evaluator-only YAML path_pattern"))
        return records
    def records(self): return self.inference_records()
def adapter_for(definition: DatasetDefinition, root: Path) -> DatasetAdapter:
    return {"casia_fasd": CasiaFasdAdapter, "msu_mfsd": MsuMfsdAdapter, "siw_mv2": SiWMv2Adapter}[definition.dataset](root, definition)
