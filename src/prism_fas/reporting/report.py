"""The self-contained final report and the bundle manifest beside it.

`report.html` is the artifact a reader opens with no context: it must explain what
was run, on what, with which frozen inputs, and what the results were — including
the failures. §31's phrase for the requirement is that no conversational memory
may be needed, and that is the test this module is written against.

Three properties are enforced rather than assumed.

**Self-contained.** One HTML file with inline CSS and no external request. Figures
are embedded as `data:` URIs rather than linked, so the file survives being
mailed, moved or read from a clone where the regenerable PNGs are absent. A
figure that cannot be read is named as missing instead of rendering as a broken
image.

**Never fabricates a target number.** A section with no evidence says so. The one
failure mode that would actually matter here is printing a plausible ACER for a
run that never happened, so absent evidence renders as an explicit "not run"
rather than as a blank cell or a zero.

**Preserves the negatives.** Failed, blocked and losing rows get their own
section. A report that showed only the winners would be the winner-only cleanup
L.8 forbids, performed at the presentation layer instead of on disk.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "prism-final-report-v1"
BUNDLE_SCHEMA_VERSION = "prism-final-bundle-v1"

_CSS = """
:root { --ink:#12161c; --muted:#5b6675; --line:#dde3ea; --bg:#ffffff;
        --accent:#1f5fa9; --warn:#8a5a00; --bad:#a3282d; --good:#1d6b3f; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#e8edf3; --muted:#9aa7b6; --line:#2a323d; --bg:#0f1318;
          --accent:#6fa8e8; --warn:#e0a340; --bad:#e4767b; --good:#5fc98d; }
}
* { box-sizing:border-box; }
body { margin:0; padding:0 0 4rem; background:var(--bg); color:var(--ink);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1100px; margin:0 auto; padding:0 1.5rem; }
header { border-bottom:1px solid var(--line); padding:2.5rem 0 1.5rem; margin-bottom:2rem; }
h1 { margin:0 0 .3rem; font-size:1.75rem; letter-spacing:-.02em; }
h2 { margin:2.5rem 0 .75rem; font-size:1.2rem; padding-bottom:.4rem;
     border-bottom:1px solid var(--line); }
h3 { margin:1.5rem 0 .5rem; font-size:1rem; color:var(--muted); }
.sub { color:var(--muted); font-size:.9rem; }
nav { margin:1rem 0; }
nav a { display:inline-block; margin:0 .8rem .4rem 0; color:var(--accent);
        text-decoration:none; font-size:.88rem; }
nav a:hover { text-decoration:underline; }
table { border-collapse:collapse; width:100%; margin:.75rem 0; font-size:.85rem; }
th,td { text-align:left; padding:.4rem .6rem; border-bottom:1px solid var(--line);
        vertical-align:top; }
th { color:var(--muted); font-weight:600; white-space:nowrap; }
code,.mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.82em; }
.scroll { overflow-x:auto; }
.tag { display:inline-block; padding:.1rem .45rem; border-radius:4px; font-size:.75rem;
       font-weight:600; }
.ok { background:rgba(29,107,63,.15); color:var(--good); }
.warn { background:rgba(138,90,0,.15); color:var(--warn); }
.bad { background:rgba(163,40,45,.15); color:var(--bad); }
.absent { color:var(--muted); font-style:italic; }
figure { margin:1rem 0; }
figure img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:6px; }
figcaption { color:var(--muted); font-size:.82rem; margin-top:.35rem; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:1rem; }
.note { border-left:3px solid var(--accent); padding:.5rem .9rem; margin:1rem 0;
        background:rgba(31,95,169,.06); font-size:.88rem; }
"""


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _kv_table(payload: dict[str, Any]) -> str:
    if not payload:
        return '<p class="absent">not recorded</p>'
    rows = "".join(
        f"<tr><th>{_escape(key)}</th><td class='mono'>{_escape(value)}</td></tr>"
        for key, value in payload.items())
    return f'<div class="scroll"><table>{rows}</table></div>'


def _rows_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not rows:
        return '<p class="absent">no rows — this evidence does not exist yet</p>'
    fields = columns or sorted({key for row in rows for key in row})
    head = "".join(f"<th>{_escape(name)}</th>" for name in fields)
    body = "".join(
        "<tr>" + "".join(f"<td class='mono'>{_escape(row.get(name, ''))}</td>"
                         for name in fields) + "</tr>"
        for row in rows)
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def _section(anchor: str, title: str, body: str) -> str:
    return f'<section id="{anchor}"><h2>{_escape(title)}</h2>{body}</section>'


def _absent(reason: str) -> str:
    return f'<p class="absent">{_escape(reason)}</p>'


def _figure_card(plots_root: Path | None, name: str) -> str:
    """Embed one figure, or say plainly that its PNG could not be read.

    Embedding is what makes the page self-contained. The PNGs are regenerable
    from the stored history and are not committed, so a reader opening the
    report from a fresh clone would otherwise get four broken images.
    """
    source = (plots_root / name) if plots_root is not None else None
    if source is None or not source.is_file():
        return (f'<figure>{_absent(f"figure not embedded: {name} was not readable")}'
                f'<figcaption>{_escape(name)}</figcaption></figure>')
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return (f'<figure><img src="data:image/png;base64,{encoded}" '
            f'alt="{_escape(name)}"><figcaption>{_escape(name)}</figcaption></figure>')


def render(evidence: dict[str, Any], *, plots_root: Path | None = None) -> str:
    """Build the whole page from a mapping of already-stored evidence."""
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    identity = evidence.get("identity") or {}
    scientific = bool(evidence.get("scientific"))
    intent = evidence.get("execution_intent", "UNKNOWN")

    sections = [
        ("identity", "Project and run identity", _kv_table(identity)),
        ("environment", "Environment and compute",
         _kv_table(evidence.get("environment") or {})),
        ("inputs", "Data and weight identities",
         _rows_table(evidence.get("assets") or [],
                     ["logical_name", "expected_path", "identity", "present",
                      "required_stage"])),
        ("c3", "C3 frozen recipe banks", _kv_table(evidence.get("c3") or {})),
        ("lr", "Approved learning-rate decision", _kv_table(evidence.get("lr") or {})),
        ("source", "C4-C9 source pipeline",
         _rows_table(evidence.get("source_stages") or [],
                     ["stage", "engineering_status", "scientific_status", "outcome",
                      "checks_run", "checks_failed"])),
        ("target", "C10-C12 target protocol",
         _kv_table(evidence.get("target") or {}) if evidence.get("target")
         else _absent("no target protocol evidence: C10-C12 have not run "
                      "scientifically")),
        ("results", "Primary and secondary results",
         _rows_table(evidence.get("main_results") or [])
         if evidence.get("main_results")
         else _absent("no scientific results exist. Nothing is shown here rather than "
                      "a placeholder number")),
        ("per_seed", "Per-seed values, mean and standard deviation",
         _rows_table(evidence.get("per_seed") or [])),
        ("stats", "Bootstrap and Holm-corrected hypotheses",
         _rows_table(evidence.get("hypotheses") or [])),
        ("complexity", "Model complexity",
         _rows_table(evidence.get("complexity") or [],
                     ["model", "total_parameters", "trainable_parameters",
                      "parameter_megabytes", "macs", "flops", "status"])),
        ("compute", "Training and inference performance",
         _rows_table(evidence.get("compute") or [])),
        ("locks", "Locks and identities", _rows_table(evidence.get("locks") or [])),
        ("negatives", "Failed, blocked and negative results",
         _rows_table(evidence.get("negatives") or [],
                     ["run_id", "stage_id", "status", "execution_profile", "notes"])),
        ("firewall", "Target-firewall evidence",
         _kv_table(evidence.get("firewall") or {})),
    ]

    figures = evidence.get("figures") or []
    if figures:
        cards = "".join(_figure_card(plots_root, name) for name in figures)
        sections.insert(9, ("plots", "Plots", f'<div class="grid">{cards}</div>'))
    else:
        sections.insert(9, ("plots", "Plots",
                            _absent("no figures were produced: the evidence they draw "
                                    "from does not exist yet")))

    tables = evidence.get("tables") or []
    if tables:
        links = "".join(
            f'<tr><td class="mono">{_escape(item.get("table"))}</td>'
            f'<td>{_escape(item.get("rows", 0))}</td>'
            f'<td class="mono"><a href="../tables/{_escape(item.get("csv"))}">'
            f'{_escape(item.get("csv"))}</a></td></tr>' for item in tables)
        sections.insert(10, ("tables", "Tables",
                             f'<div class="scroll"><table>'
                             f'<tr><th>table</th><th>rows</th><th>file</th></tr>'
                             f'{links}</table></div>'))

    banner = ('<span class="tag ok">SCIENTIFIC</span>' if scientific else
              '<span class="tag warn">REHEARSAL — NOT SCIENTIFIC EVIDENCE</span>')
    caveat = "" if scientific else (
        '<div class="note"><strong>This is a CPU rehearsal.</strong> It proves the '
        'implementation executes. It completes no milestone, selects no winner, opens '
        'no real target label and produces no scientifically eligible evidence. Every '
        'number below is fixture-derived.</div>')

    nav = "".join(f'<a href="#{anchor}">{_escape(title)}</a>'
                  for anchor, title, _body in sections)
    body = "".join(_section(anchor, title, content) for anchor, title, content in sections)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRISM-FAS-C-LLM final report</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<header>
  <h1>PRISM-FAS-C-LLM — final report</h1>
  <p class="sub">{banner} &nbsp; execution intent <code>{_escape(intent)}</code>
     &nbsp; generated {_escape(generated)}</p>
  {caveat}
  <nav>{nav}</nav>
</header>
{body}
<footer class="sub" style="margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line)">
  Every value on this page derives from a stored artifact. Sections with no evidence
  say so rather than showing a placeholder.
</footer>
</div></body></html>
"""


def write_report(path: Path, evidence: dict[str, Any],
                 plots_root: Path | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The report always lands at <reports_root>/final/report.html, so the sibling
    # plots directory is derivable; an explicit root still wins for tests.
    if plots_root is None:
        plots_root = path.parent.parent / "plots"
    path.write_text(render(evidence, plots_root=plots_root), encoding="utf-8")
    return path


def build_bundle(repo: Path, *, reports_root: Path, runs_root: Path,
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The navigation manifest: references and hashes, never duplicated payload.

    Checkpoints are recorded by path and size, and hashed only when small enough
    that hashing is cheap. Copying a multi-gigabyte checkpoint into a manifest
    would double the folder for no benefit — the point is to be able to find and
    verify it, not to carry it twice.
    """
    repo, reports_root, runs_root = Path(repo), Path(reports_root), Path(runs_root)
    HASH_LIMIT = 64 * 1024 * 1024

    def describe(path: Path) -> dict[str, Any]:
        size = path.stat().st_size
        record: dict[str, Any] = {
            "path": path.relative_to(repo).as_posix(), "size_bytes": size}
        if size <= HASH_LIMIT:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(chunk)
            record["sha256"] = digest.hexdigest()
        else:
            record["sha256"] = None
            record["hash_skipped"] = f"larger than {HASH_LIMIT} bytes; referenced by path"
        return record

    def collect(root: Path, patterns: tuple[str, ...]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if not root.exists():
            return found
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                if path.is_file():
                    found.append(describe(path))
        return found

    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        # Which profile produced this bundle. Without it the manifest asserted
        # `scientific_eligible: false` while giving no way to tell WHICH
        # non-scientific run wrote it.
        "execution_profile": reports_root.name,
        "reports_root": reports_root.relative_to(repo).as_posix()
        if reports_root.is_relative_to(repo) else str(reports_root),
        "runs_root": runs_root.relative_to(repo).as_posix()
        if runs_root.is_relative_to(repo) else str(runs_root),
        "reports": collect(reports_root, ("*.json", "*.html")),
        "plots": collect(reports_root / "plots", ("*.png",)),
        "tables": collect(reports_root / "tables", ("*.csv", "*.json")),
        "run_manifests": collect(runs_root, ("run_manifest.json",)),
        "configs": collect(runs_root, ("config.json",)),
        "training_history": collect(runs_root, ("train_history.jsonl",)),
        "checkpoints": collect(runs_root, ("*.pt", "*.safetensors")),
        "state": collect(repo / "state", ("*.json",)),
        "policy": {
            "references_not_copies": True,
            "large_files_referenced_by_path": True,
            "hash_limit_bytes": HASH_LIMIT,
            "losing_and_failed_runs_included": True,
        },
        **(extra or {}),
    }
    bundle["file_count"] = sum(len(value) for value in bundle.values()
                               if isinstance(value, list))
    return bundle


__all__ = ["SCHEMA_VERSION", "BUNDLE_SCHEMA_VERSION", "render", "write_report",
           "build_bundle"]
