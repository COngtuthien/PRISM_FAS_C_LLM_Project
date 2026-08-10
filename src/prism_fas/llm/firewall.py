"""Fail-closed request firewall and secret redaction.

Two independent jobs:

1. Nothing that could constitute target adaptation may reach the provider. The
   LLM's contribution must be a text-only prior derived from a generic ontology,
   not visual or metric feedback from SiW-Mv2 (spec section 7.1, 7.5).
2. No credential may reach a log, an exception message, a report or a
   provenance record (spec section 7.2).

Both are enforced structurally. Developer discipline is not the control.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

# --- target-leakage vocabulary ---------------------------------------------
# Dataset and split names. The LLM receives a generic ontology; naming a corpus
# at all is outside its allowed input.
_DATASET_TOKENS: tuple[str, ...] = (
    "siw", "siw-mv2", "siw_mv2", "siwmv2", "spoof in the wild",
    "casia", "casia-fasd", "casia_fasd", "msu", "msu-mfsd", "msu_mfsd",
    "celeba", "celeba-spoof", "oulu", "oulu-npu", "replay-attack", "wmca",
)

# Target-metric vocabulary. Feeding any of these back into the planner would
# turn a source-only prior into target tuning.
_METRIC_TOKENS: tuple[str, ...] = (
    "acer", "apcer", "bpcer", "hter", "roc-auc", "roc_auc", "auroc", "eer",
    "attack-wise", "attack_wise", "target metric", "target_metric",
    "target score", "target_score", "video acer", "false reject", "false accept",
)

# Split / label / protocol vocabulary that only exists on the evaluation side.
_TARGET_STRUCTURE_TOKENS: tuple[str, ...] = (
    "target_test", "target_dev", "target_label", "target_labels", "ground_truth",
    "ground truth", "evaluation_only", "attack_family", "attack family",
    "prediction_lock", "source_matrix_lock", "held-out label", "held_out_label",
    "subject_id", "official_split", "label_live_spoof",
)

# Feedback-shaped instructions: the phrasings that would make the planner chase
# a target result even without naming a dataset or a metric.
_FEEDBACK_PATTERNS: tuple[str, ...] = (
    r"\bbeat\s+the\s+(target|benchmark|baseline)\b",
    r"\b(version[\s_-]?b|previous\s+version)\b.{0,40}\b(fail|miss|weak|error|wrong)",
    r"\b(fail|miss|weak|error)\w*\b.{0,40}\b(target|benchmark|held[\s_-]?out)\b",
    r"\boptimi[sz]e\b.{0,30}\b(acer|apcer|bpcer|target|benchmark)\b",
    r"\bimprove\b.{0,30}\b(target|benchmark|held[\s_-]?out)\s+(score|result|performance|metric)",
    r"\bfocus\s+on\b.{0,40}\b(attacks?\s+that|target|held[\s_-]?out)\b",
    r"\bthe\s+(test|target|evaluation)\s+set\s+contains\b",
    r"\bwhich\s+attacks?\s+(are|were)\s+(in|present\s+in)\s+the\s+(test|target)\b",
)

# Binary/media payloads. The scientific path is TEXT ONLY, so an inline image is
# a structural violation regardless of its content.
_BINARY_PATTERNS: tuple[str, ...] = (
    r"data:image/[a-z0-9.+-]+;base64,",
    r"data:video/[a-z0-9.+-]+;base64,",
    r"data:audio/[a-z0-9.+-]+;base64,",
    r"\.(?:png|jpe?g|bmp|tiff?|webp|mp4|avi|mov|mkv|wav|npz|npy|parquet|pt|pth|ckpt|safetensors)\b",
)

# Keys that must never appear in a request payload, whatever their value.
_FORBIDDEN_PAYLOAD_KEYS: frozenset[str] = frozenset({
    "image", "images", "image_bytes", "image_url", "inline_data", "inlinedata",
    "file_data", "filedata", "file_uri", "blob", "media", "video", "audio",
    "parts", "attachments", "label", "labels", "ground_truth", "attack_family",
    "target_metrics", "predictions", "acer", "apcer", "bpcer",
})

# Words that legitimately appear in ontology text and must not trip the scan.
# "context" is one of the nine semantic regions; "blur" is an artifact.
_ALLOWLIST_PHRASES: tuple[str, ...] = ()

_SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    # Google API keys: "AIza" + 35 chars.
    (r"AIza[0-9A-Za-z_\-]{35}", "GOOGLE_API_KEY"),
    (r"\bya29\.[0-9A-Za-z_\-]+", "GOOGLE_OAUTH_TOKEN"),
    (r"\bsk-[0-9A-Za-z]{20,}", "GENERIC_SECRET_KEY"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "PRIVATE_KEY"),
    (r"(?i)\b(api[_-]?key|apikey|authorization|bearer|access[_-]?token|secret)\b\s*[:=]\s*[\"']?[^\s\"',}]{8,}",
     "CREDENTIAL_ASSIGNMENT"),
)

REDACTION = "[REDACTED]"


class RequestFirewallError(RuntimeError):
    """A request was refused before it could reach a provider."""

    def __init__(self, violations: list["FirewallViolation"]) -> None:
        self.violations = violations
        super().__init__("; ".join(violation.text() for violation in violations))


@dataclass(frozen=True)
class FirewallViolation:
    category: str
    location: str
    detail: str

    def text(self) -> str:
        return f"[{self.category}][{self.location}] {self.detail}"

    def as_dict(self) -> dict[str, str]:
        return {"category": self.category, "location": self.location, "detail": self.detail}


def redact_secrets(text: str) -> str:
    """Replace anything that looks like a credential.

    Applied to every message that could be logged, raised or serialized. It is
    deliberately eager: a false positive costs a redacted log line, a false
    negative leaks a key.
    """
    if not isinstance(text, str) or not text:
        return text
    result = text
    for pattern, _name in _SECRET_PATTERNS:
        result = re.sub(pattern, REDACTION, result)
    return result


def _iter_strings(value: Any, location: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield location, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(key, f"{location}.<key>")
            yield from _iter_strings(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_strings(item, f"{location}[{index}]")


def _forbidden_keys(value: Any, location: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.strip().lower() in _FORBIDDEN_PAYLOAD_KEYS:
                yield f"{location}.{key}", key
            yield from _forbidden_keys(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _forbidden_keys(item, f"{location}[{index}]")


def _word_hit(haystack: str, token: str) -> bool:
    """Token match on word boundaries, so 'context' does not match inside a
    longer word and 'msu' does not fire on 'consumed'."""
    return re.search(rf"(?<![0-9A-Za-z_]){re.escape(token)}(?![0-9A-Za-z_])", haystack) is not None


def scan_for_target_leakage(payload: Any) -> list[FirewallViolation]:
    """Return every violation found. Never raises; the caller decides."""
    violations: list[FirewallViolation] = []

    for location, key in _forbidden_keys(payload):
        violations.append(FirewallViolation(
            "forbidden_payload_key", location,
            f"key {key!r} can carry target or binary content and is not part of the text-only contract"))

    for location, text in _iter_strings(payload):
        lowered = text.lower()
        for token in _DATASET_TOKENS:
            if _word_hit(lowered, token):
                violations.append(FirewallViolation(
                    "dataset_reference", location, f"names a dataset/corpus: {token!r}"))
        for token in _METRIC_TOKENS:
            if _word_hit(lowered, token):
                violations.append(FirewallViolation(
                    "target_metric", location, f"names a target metric: {token!r}"))
        for token in _TARGET_STRUCTURE_TOKENS:
            if _word_hit(lowered, token):
                violations.append(FirewallViolation(
                    "target_structure", location, f"names target-side structure: {token!r}"))
        for pattern in _FEEDBACK_PATTERNS:
            if re.search(pattern, lowered):
                violations.append(FirewallViolation(
                    "target_feedback", location,
                    f"reads as target-performance feedback (pattern {pattern!r})"))
        for pattern in _BINARY_PATTERNS:
            if re.search(pattern, lowered):
                violations.append(FirewallViolation(
                    "binary_payload", location,
                    f"carries non-text media content (pattern {pattern!r})"))

    # Deduplicate while keeping a deterministic order for reports.
    seen: set[tuple[str, str, str]] = set()
    unique: list[FirewallViolation] = []
    for violation in violations:
        key = (violation.category, violation.location, violation.detail)
        if key not in seen:
            seen.add(key)
            unique.append(violation)
    return sorted(unique, key=lambda item: (item.category, item.location, item.detail))


def assert_request_is_target_free(payload: Any) -> None:
    """Fail closed. Called by every provider before a request leaves the process,
    including the mock and replay providers, so a leak cannot hide behind a test
    double."""
    violations = scan_for_target_leakage(payload)
    if violations:
        raise RequestFirewallError(violations)
