from __future__ import annotations
import hashlib, json, os, platform, subprocess, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""): digest.update(chunk)
    return digest.hexdigest()
def stable_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
def _atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle: handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, path)
    except Exception:
        try: os.unlink(temp)
        except FileNotFoundError: pass
        raise
def atomic_json_write(path: Path, value: Any) -> None: _atomic(path, json.dumps(value, sort_keys=True, indent=2, default=str) + "\n")
def atomic_yaml_write(path: Path, value: Any) -> None: _atomic(path, yaml.safe_dump(value, sort_keys=True, allow_unicode=True))
def utc_timestamp() -> str: return datetime.now(timezone.utc).isoformat()
def git_commit(cwd: Path) -> str | None:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError): return None
def environment_fingerprint(cwd: Path) -> dict[str, Any]: return {"python": platform.python_version(), "platform": platform.platform(), "git_commit": git_commit(cwd)}
class RunMetadata(BaseModel): run_id: str; created_at: str; config_hash: str; environment: dict[str, Any]
