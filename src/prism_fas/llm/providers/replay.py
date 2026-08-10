"""Replay provider: re-serve archived raw responses, never call a model.

This exists because a hosted model is not reproducible. Asking Gemini the same
question twice may return different text, so the reproducible scientific artifact
is the archived response, not the request. Replay lets C2/C3 raw responses be
re-parsed, re-validated, re-canonicalized and re-hashed at any later date and
produce byte-identical recipes.

The archive is keyed by slot id and attempt. A slot that was never archived is a
hard error: replay never invents a response and never falls back to a network
call.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import ErrorClass, GenerationRequest, ProviderError, ProviderGenerationResult
from .base import RecipeProvider


@dataclass(frozen=True)
class ReplayEntry:
    slot_id: str
    attempt: int
    raw_text: str
    provider: str = "gemini"
    model_id: str = ""
    model_version: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    provider_request_id: str | None = None
    provider_seed: int | None = None
    sdk_version: str | None = None
    api_surface: str | None = None
    request_sha256: str | None = None

    @property
    def raw_response_sha256(self) -> str:
        return hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()


class ReplayArchive:
    """An immutable set of archived raw responses."""

    def __init__(self, entries: list[ReplayEntry]) -> None:
        self._entries: dict[tuple[str, int], ReplayEntry] = {}
        for entry in entries:
            key = (entry.slot_id, int(entry.attempt))
            if key in self._entries:
                raise ValueError(f"duplicate archived response for slot {entry.slot_id!r} "
                                 f"attempt {entry.attempt}")
            self._entries[key] = entry

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def slot_ids(self) -> list[str]:
        return sorted({slot for slot, _ in self._entries})

    def get(self, slot_id: str, attempt: int) -> ReplayEntry | None:
        return self._entries.get((slot_id, int(attempt)))

    def identity(self) -> str:
        """SHA-256 over every archived response hash, in slot/attempt order."""
        material = [{"slot_id": slot, "attempt": attempt,
                     "raw_response_sha256": entry.raw_response_sha256}
                    for (slot, attempt), entry in sorted(self._entries.items())]
        text = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "ReplayArchive":
        return cls([ReplayEntry(**record) for record in records])

    @classmethod
    def from_jsonl(cls, path: Path) -> "ReplayArchive":
        """One JSON record per line. The archive file is provenance and is never
        rewritten once C3 begins."""
        path = Path(path)
        records: list[dict[str, Any]] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} is not a JSON object: {exc}") from exc
        return cls.from_records(records)


class ReplayRecipeProvider(RecipeProvider):
    """Serves archived responses. Structurally incapable of a network call: it
    imports no client and holds no credential."""

    name = "replay"

    def __init__(self, archive: ReplayArchive, *, strict: bool = True) -> None:
        self._archive = archive
        self._strict = strict
        self.calls: list[tuple[str, int]] = []

    @property
    def archive(self) -> ReplayArchive:
        return self._archive

    def _generate(self, request: GenerationRequest, *, attempt: int) -> ProviderGenerationResult:
        self.calls.append((request.slot_id, attempt))
        entry = self._archive.get(request.slot_id, attempt)
        if entry is None:
            message = (f"no archived response for slot {request.slot_id!r} attempt {attempt}; "
                       "replay never invents a response and never calls a provider")
            if self._strict:
                raise KeyError(message)
            return ProviderGenerationResult(
                slot_id=request.slot_id, attempt=attempt, provider=self.name,
                model_id=request.model_id, raw_text=None, parsed=None,
                error=ProviderError(ErrorClass.LOCAL_ERROR, message), api_surface="replay")

        if entry.request_sha256 is not None and entry.request_sha256 != request.request_sha256:
            # The archived response was produced under a different prompt, schema,
            # model, ontology or thinking level. Replaying it under the current
            # request would silently mix two scientific identities.
            raise ValueError(
                f"archived response for slot {request.slot_id!r} attempt {attempt} was produced "
                f"under request identity {entry.request_sha256[:16]}..., but the current request "
                f"identity is {request.request_sha256[:16]}...; refusing to replay across identities")

        return ProviderGenerationResult(
            slot_id=request.slot_id,
            attempt=attempt,
            provider=entry.provider,
            model_id=entry.model_id or request.model_id,
            raw_text=entry.raw_text,
            parsed=None,
            finish_reason=entry.finish_reason,
            latency_seconds=None,
            usage=dict(entry.usage or {}),
            provider_request_id=entry.provider_request_id,
            model_version=entry.model_version,
            provider_seed=entry.provider_seed,
            error=None,
            sdk_version=entry.sdk_version,
            api_surface=entry.api_surface or "replay",
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "archived_responses": len(self._archive),
                "archive_identity": self._archive.identity(), "network": False}
