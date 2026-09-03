"""Shared primitives for the PRISM-FAS-C EXT-Q1Q2 additive extension.

This module is deliberately dependency-light: it imports only the Python
standard library and (optionally) NumPy.  It must never import ``pydantic``,
``pandas``, ``pyarrow``, ``torch`` or any project module that would pull a
scientific stack in, because the E0-E4 laptop phase has to run under a bare
interpreter.

Scope of this file:

* ``assert_ext_write_path`` - the single reusable guard that every extension
  write must pass through.  It rejects any path that resolves into a frozen
  Flow-1 / Flow-2 namespace and only permits the four ``c_ext_q1q2_v1``
  namespaces.
* deterministic identity helpers (canonical JSON, SHA-256).
* small pure-Python statistics used by E1/E2/E3 (sample SD with ddof=1,
  Shannon / normalized entropy, Jensen-Shannon divergence, standardized mean
  difference, Gower-style mixed distance).

Nothing in this file performs science; it only provides the plumbing the
milestone modules share.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# --------------------------------------------------------------------------- #
# Namespace constants
# --------------------------------------------------------------------------- #

EXT_ID = "c_ext_q1q2_v1"

# Output roots the extension is allowed to write into (repo-relative, POSIX).
EXT_ALLOWED_ROOTS: tuple[str, ...] = (
    f"configs/{EXT_ID}",
    f"reports/{EXT_ID}",
    f"runs/{EXT_ID}",
    f"state/{EXT_ID}",
)

# Frozen Flow-1 / Flow-2 scientific namespaces.  No extension write may resolve
# into any of these, ever.
FROZEN_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "reports/full",
    "reports/flow2_counterfactual_assumed_pass",
    "runs/full",
    "runs/flow2_counterfactual_assumed_pass",
    "runs/exploratory_target_v3",
    "reports/c0", "reports/c1", "reports/c2", "reports/c2b", "reports/c2c",
    "reports/c3", "reports/c4", "reports/c5", "reports/c6", "reports/c7",
    "reports/c8", "reports/c9", "reports/c10", "reports/c11", "reports/c12",
    "reports/c13",
    "reports/m10",
    "assets/recipe_banks",
    "data/evaluation_only",
    "data/processed/prism_target_eval_v2",
    # frozen top-level state artifacts belonging to Flow-1 / Flow-2
    "state/PIPELINE_STATE.json",
    "state/MASTER_RUN_INDEX.json",
    "state/ENVIRONMENT_MANIFEST.json",
    "state/flow2_counterfactual_assumed_pass",
    "state/preflight",
)


class ExtPathSafetyError(RuntimeError):
    """A write path resolves outside the extension namespace or into frozen evidence."""


# --------------------------------------------------------------------------- #
# Repo-root resolution
# --------------------------------------------------------------------------- #

def repo_root() -> Path:
    """Return the repository root (the directory that contains ``src/prism_fas``)."""
    here = Path(__file__).resolve()
    # src/prism_fas/evaluation/c_ext_common.py  ->  parents[3] == repo root
    root = here.parents[3]
    if not (root / "src" / "prism_fas").is_dir():
        # Fall back to a .git walk for unusual layouts.
        cur = here
        for parent in cur.parents:
            if (parent / ".git").exists():
                return parent
    return root


def _as_repo_relative_posix(path: os.PathLike | str, root: Path) -> str:
    """Resolve *path* against *root* and return a normalized repo-relative POSIX string.

    Raises ``ExtPathSafetyError`` if the resolved path escapes *root*.
    """
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    # Normalize '..' / '.' lexically without requiring the path to exist.
    resolved = Path(os.path.normpath(str(p)))
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:  # path is outside the repo
        raise ExtPathSafetyError(
            f"path {os.fspath(path)!r} resolves to {resolved} which is outside the "
            f"repository root {root}"
        ) from exc
    return rel.as_posix()


def _has_prefix(rel_posix: str, prefix: str) -> bool:
    """True if *rel_posix* equals *prefix* or lies underneath it."""
    return rel_posix == prefix or rel_posix.startswith(prefix.rstrip("/") + "/")


def assert_ext_write_path(
    path: os.PathLike | str,
    *,
    root: Path | None = None,
    must_be_under: str | None = None,
) -> str:
    """Guard every extension write.

    Returns the normalized repo-relative POSIX path on success.

    Rejects, with :class:`ExtPathSafetyError`:

    * anything outside the repository root (including ``..`` traversal);
    * anything under a frozen Flow-1 / Flow-2 namespace
      (:data:`FROZEN_FORBIDDEN_PREFIXES`);
    * anything not under one of :data:`EXT_ALLOWED_ROOTS`;
    * (optionally) anything not under *must_be_under* (a further restriction,
      e.g. ``reports/c_ext_q1q2_v1/e1_recipe_analysis``).
    """
    root = root or repo_root()
    rel = _as_repo_relative_posix(path, root)

    for forbidden in FROZEN_FORBIDDEN_PREFIXES:
        if _has_prefix(rel, forbidden):
            raise ExtPathSafetyError(
                f"REFUSED: {rel!r} resolves into frozen Flow-1/Flow-2 namespace "
                f"{forbidden!r}; extension writes are forbidden there"
            )

    if not any(_has_prefix(rel, allowed) for allowed in EXT_ALLOWED_ROOTS):
        raise ExtPathSafetyError(
            f"REFUSED: {rel!r} is not under any allowed extension root "
            f"{EXT_ALLOWED_ROOTS!r}"
        )

    if must_be_under is not None and not _has_prefix(rel, must_be_under):
        raise ExtPathSafetyError(
            f"REFUSED: {rel!r} is not under the required subtree {must_be_under!r}"
        )

    return rel


def is_ext_write_path_ok(path: os.PathLike | str, *, root: Path | None = None) -> bool:
    """Boolean wrapper around :func:`assert_ext_write_path` (no raise)."""
    try:
        assert_ext_write_path(path, root=root)
        return True
    except ExtPathSafetyError:
        return False


# --------------------------------------------------------------------------- #
# Deterministic identity helpers
# --------------------------------------------------------------------------- #

def canonical_json_bytes(obj) -> bytes:
    """Canonical JSON serialization used for every identity hash in the extension.

    Sorted keys, compact separators, ``ensure_ascii=True``, UTF-8.  ``NaN`` /
    ``Infinity`` are rejected (``allow_nan=False``) so an identity can never
    depend on a non-finite float.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike | str, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def identity_hash(material: Mapping) -> str:
    """SHA-256 over the canonical JSON of an identity-material mapping."""
    return sha256_json(dict(material))


# --------------------------------------------------------------------------- #
# Atomic, guarded writes
# --------------------------------------------------------------------------- #

def write_text_atomic(path: os.PathLike | str, text: str, *, root: Path | None = None) -> str:
    """Guarded, atomic text write.  Returns the repo-relative path written."""
    rel = assert_ext_write_path(path, root=root)
    root = root or repo_root()
    abs_path = root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = abs_path.with_suffix(abs_path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, abs_path)
    return rel


def write_json_atomic(path: os.PathLike | str, obj, *, root: Path | None = None,
                      indent: int = 2, sort_keys: bool = True) -> str:
    text = json.dumps(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=True,
                      allow_nan=False) + "\n"
    return write_text_atomic(path, text, root=root)


# --------------------------------------------------------------------------- #
# Pure-Python statistics (extension convention: sample SD, ddof=1)
# --------------------------------------------------------------------------- #

def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    if not xs:
        raise ValueError("mean of empty sequence")
    return math.fsum(xs) / len(xs)


def sample_sd(xs: Sequence[float]) -> float:
    """Sample standard deviation, ddof=1.  Returns 0.0 for a single value."""
    xs = list(xs)
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (n - 1))


def summarize(xs: Sequence[float]) -> dict:
    """Standard descriptive summary used across E1/E2/E3 (sample SD, ddof=1)."""
    xs = sorted(float(x) for x in xs)
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "sd_ddof1": None, "median": None,
                "q1": None, "q3": None, "min": None, "max": None}

    def _quantile(q: float) -> float:
        if n == 1:
            return xs[0]
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    return {
        "n": n,
        "mean": mean(xs),
        "sd_ddof1": sample_sd(xs),
        "median": _quantile(0.5),
        "q1": _quantile(0.25),
        "q3": _quantile(0.75),
        "min": xs[0],
        "max": xs[-1],
    }


def shannon_entropy(counts: Iterable[float]) -> float:
    """H(X) = -sum p_k ln p_k  (natural log).  Empty / zero-total -> 0.0."""
    counts = [float(c) for c in counts if float(c) > 0.0]
    total = math.fsum(counts)
    if total <= 0.0:
        return 0.0
    h = 0.0
    for c in counts:
        p = c / total
        h -= p * math.log(p)
    return h


def normalized_entropy(counts: Iterable[float], *, k: int | None = None) -> float:
    """H_norm = H(X) / ln(K).

    ``K`` defaults to the number of strictly-positive categories.  Per the
    EXT spec: report 0.0 when ``K <= 1``.
    """
    counts = [float(c) for c in counts]
    positive = [c for c in counts if c > 0.0]
    kk = k if k is not None else len(positive)
    if kk is None or kk <= 1:
        return 0.0
    return shannon_entropy(counts) / math.log(kk)


def _as_prob(vec: Sequence[float], eps: float) -> list[float]:
    v = [max(float(x), 0.0) + eps for x in vec]
    s = math.fsum(v)
    return [x / s for x in v]


def kl_divergence(p: Sequence[float], q: Sequence[float], *, eps: float = 1e-12) -> float:
    pp = _as_prob(p, eps)
    qq = _as_prob(q, eps)
    return math.fsum(pi * math.log(pi / qi) for pi, qi in zip(pp, qq))


def js_divergence(p: Sequence[float], q: Sequence[float], *, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (natural log, base-e).  Fixed eps smoothing = 1e-12.

    JS(P,Q) = 1/2 KL(P||M) + 1/2 KL(Q||M),  M = (P+Q)/2.
    Symmetric, non-negative, bounded above by ln(2).
    """
    if len(p) != len(q):
        raise ValueError("js_divergence: length mismatch")
    pp = _as_prob(p, eps)
    qq = _as_prob(q, eps)
    m = [(a + b) / 2.0 for a, b in zip(pp, qq)]
    return 0.5 * kl_divergence(pp, m, eps=0.0) + 0.5 * kl_divergence(qq, m, eps=0.0)


def standardized_mean_difference(a: Sequence[float], b: Sequence[float]) -> dict:
    """SMD(A,B) = (mean(A) - mean(B)) / s_pooled, with s_pooled the ddof=1 pooled SD.

    Returns a dict with the components so callers can serialize the full trace.
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("standardized_mean_difference needs >=2 observations per group")
    ma, mb = mean(a), mean(b)
    va = math.fsum((x - ma) ** 2 for x in a) / (na - 1)
    vb = math.fsum((x - mb) ** 2 for x in b) / (nb - 1)
    s_pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    smd = (ma - mb) / s_pooled if s_pooled > 0.0 else 0.0
    return {
        "mean_a": ma, "mean_b": mb, "n_a": na, "n_b": nb,
        "var_a_ddof1": va, "var_b_ddof1": vb,
        "s_pooled": s_pooled, "smd": smd,
    }


def gower_mixed_distance(
    rec_a: Mapping[str, object],
    rec_b: Mapping[str, object],
    *,
    categorical_fields: Sequence[str],
    continuous_fields: Mapping[str, tuple[float, float]],
) -> float:
    """Gower-style mixed distance between two flat field mappings.

    * categorical: 0 if equal, 1 if different (missing on both -> 0; missing on
      one -> 1).
    * continuous: ``|a - b| / range`` where ``range = hi - lo`` from
      ``continuous_fields[field] = (lo, hi)``.

    The result is the unweighted mean over all fields that are comparable
    (i.e. present or range-defined).
    """
    terms: list[float] = []
    for f in categorical_fields:
        av, bv = rec_a.get(f), rec_b.get(f)
        if av is None and bv is None:
            continue
        terms.append(0.0 if av == bv else 1.0)
    for f, (lo, hi) in continuous_fields.items():
        av, bv = rec_a.get(f), rec_b.get(f)
        if av is None or bv is None:
            continue
        span = float(hi) - float(lo)
        if span <= 0.0:
            terms.append(0.0)
        else:
            terms.append(min(abs(float(av) - float(bv)) / span, 1.0))
    if not terms:
        return 0.0
    return math.fsum(terms) / len(terms)


# --------------------------------------------------------------------------- #
# Small JSONL / JSON readers (stdlib only)
# --------------------------------------------------------------------------- #

def read_jsonl(path: os.PathLike | str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_json(path: os.PathLike | str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
