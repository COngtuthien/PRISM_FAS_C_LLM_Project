"""Deterministic figures, drawn only from stored evidence.

Every function here takes rows that already exist on disk and returns a path. No
plotting call may load a model, touch a dataset or trigger training — if a figure
cannot be drawn from the recorded artifacts, the artifact is missing and the
honest output is a skip with a reason, not a plot with invented data.

Two mechanical decisions keep the figures reproducible:

* the Agg backend is selected before pyplot is imported, so a headless GPU host
  behaves identically to a laptop;
* no timestamp, hostname or random colour is drawn into a figure, so the same
  evidence produces the same bytes.

`generate_all` reports what it drew AND what it skipped, with the reason. A
reporting layer that silently produced eleven of nineteen figures would be
indistinguishable from one that had nothing to say.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

SCHEMA_VERSION = "prism-plots-v1"

#: Every figure the closure contract asks for, with the evidence it needs. A
#: figure whose evidence is absent is skipped by name rather than omitted.
DECLARED_FIGURES: tuple[tuple[str, str], ...] = (
    ("training_loss", "train_history.jsonl total_loss"),
    ("loss_components", "train_history.jsonl losses.*"),
    ("learning_rate", "train_history.jsonl learning_rates.*"),
    ("source_dev_metrics", "train_history.jsonl source_dev.*"),
    ("recipe_bank_coverage", "C3 bank coverage per arm"),
    ("recipe_bank_diversity", "C3 bank diversity per arm"),
    ("quality_gate_acceptance", "C6 gate decisions"),
    ("q_distribution", "C6 quality weights"),
    ("synthetic_vs_real_separability", "C6 reliability report"),
    ("track_g_per_seed", "C8/C12 per-seed metrics, Track G"),
    ("track_g_mean_std", "C12 seed summary, Track G"),
    ("track_r_llm_vs_det", "C12 seed summary, Track R"),
    ("prompthead_ablation", "C12 C-R-NOPROMPT vs C-R-LLM"),
    ("forest_hypotheses", "C12 bootstrap intervals for C-H1/C-H2/C-H3/C-H5"),
    ("ch4_mechanism", "C12 C-H4 mechanism evidence"),
    ("frame_vs_video", "C12 frame and video metrics"),
    ("roc_curves", "C12 scored predictions"),
    ("calibration_reliability", "C12 calibrated scores"),
    ("threshold_metrics", "C12 threshold sweep"),
)


def _pyplot() -> Any:
    """Import pyplot with a headless backend, or explain why it is unavailable."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _finish(figure: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path.name


def training_curves(rows: list[dict[str, Any]], out: Path) -> list[str]:
    """Total loss, per-component losses, and per-group learning rates."""
    from prism_fas.reporting.history import series

    if not rows:
        return []
    plt = _pyplot()
    written: list[str] = []

    x, y = series(rows, "total_loss")
    if y:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(x, y, linewidth=1.5)
        axis.set_xlabel("step"); axis.set_ylabel("total loss")
        axis.set_title("Training loss"); axis.grid(alpha=0.3)
        written.append(_finish(figure, out / "training_loss.png"))

    components = sorted({key for row in rows for key in (row.get("losses") or {})})
    if components:
        figure, axis = plt.subplots(figsize=(7, 4))
        for name in components:
            cx, cy = series(rows, name, nested="losses")
            if cy:
                axis.plot(cx, cy, linewidth=1.2, label=name)
        axis.set_xlabel("step"); axis.set_ylabel("loss")
        axis.set_title("Active loss components"); axis.grid(alpha=0.3)
        axis.legend(fontsize=7, ncol=2)
        written.append(_finish(figure, out / "loss_components.png"))

    groups = sorted({key for row in rows for key in (row.get("learning_rates") or {})})
    if groups:
        figure, axis = plt.subplots(figsize=(7, 4))
        for name in groups:
            gx, gy = series(rows, name, nested="learning_rates")
            if gy:
                axis.plot(gx, gy, linewidth=1.2, label=name)
        axis.set_xlabel("step"); axis.set_ylabel("learning rate")
        axis.set_yscale("log")
        axis.set_title("Learning rate by optimizer group"); axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        written.append(_finish(figure, out / "learning_rate.png"))

    metrics = sorted({key for row in rows for key in (row.get("source_dev") or {})})
    if metrics:
        figure, axis = plt.subplots(figsize=(7, 4))
        for name in metrics:
            mx, my = series(rows, name, nested="source_dev")
            if my:
                axis.plot(mx, my, linewidth=1.2, marker="o", markersize=3, label=name)
        axis.set_xlabel("step"); axis.set_ylabel("value")
        axis.set_title("source_dev metrics"); axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        written.append(_finish(figure, out / "source_dev_metrics.png"))
    return written


def bank_coverage(banks: dict[str, dict[str, Any]], out: Path) -> list[str]:
    """Recipe-bank coverage and diversity across RND, DET and LLM."""
    if not banks:
        return []
    plt = _pyplot()
    arms = sorted(banks)
    written: list[str] = []

    axes_names = sorted({name for bank in banks.values()
                         for name in (bank.get("coverage") or {})})
    if axes_names:
        figure, axis = plt.subplots(figsize=(max(7, len(axes_names) * 0.6), 4))
        width = 0.8 / max(len(arms), 1)
        for index, arm in enumerate(arms):
            coverage = banks[arm].get("coverage") or {}
            values = [float(coverage.get(name, 0)) for name in axes_names]
            positions = [pos + index * width for pos in range(len(axes_names))]
            axis.bar(positions, values, width=width, label=arm)
        axis.set_xticks([pos + 0.4 - width / 2 for pos in range(len(axes_names))])
        axis.set_xticklabels(axes_names, rotation=60, ha="right", fontsize=7)
        axis.set_ylabel("count"); axis.set_title("Recipe bank coverage by arm")
        axis.legend(); axis.grid(alpha=0.3, axis="y")
        written.append(_finish(figure, out / "recipe_bank_coverage.png"))

    diversity = {arm: banks[arm].get("diversity") for arm in arms
                 if banks[arm].get("diversity") is not None}
    if diversity:
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.bar(list(diversity), [float(value) for value in diversity.values()])
        axis.set_ylabel("diversity"); axis.set_title("Recipe bank diversity by arm")
        axis.grid(alpha=0.3, axis="y")
        written.append(_finish(figure, out / "recipe_bank_diversity.png"))
    return written


def quality_gate(summaries: dict[str, Any], q_values: dict[str, list[float]],
                 out: Path) -> list[str]:
    """Acceptance per arm and the q distribution behind it."""
    plt = _pyplot()
    written: list[str] = []
    if summaries:
        arms = sorted(summaries)
        accepted = [float(summaries[arm].get("accepted", 0)) for arm in arms]
        rejected = [float(summaries[arm].get("rejected", 0)) for arm in arms]
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.bar(arms, accepted, label="accepted")
        axis.bar(arms, rejected, bottom=accepted, label="rejected")
        axis.set_ylabel("candidates"); axis.set_title("Quality-gate outcome by arm")
        axis.legend(); axis.grid(alpha=0.3, axis="y")
        written.append(_finish(figure, out / "quality_gate_acceptance.png"))
    if q_values:
        figure, axis = plt.subplots(figsize=(6, 4))
        for arm, values in sorted(q_values.items()):
            if values:
                axis.hist(values, bins=20, alpha=0.55, label=arm)
        axis.set_xlabel("q"); axis.set_ylabel("count")
        axis.set_title("Quality weight q distribution")
        axis.legend(); axis.grid(alpha=0.3)
        written.append(_finish(figure, out / "q_distribution.png"))
    return written


def seed_comparison(rows: dict[str, dict[str, Any]], out: Path, *,
                    metric: str = "acer", filename: str = "track_g_per_seed.png",
                    title: str = "Per-seed metric by arm") -> list[str]:
    """Per-seed values with a mean +/- std overlay. No best-seed reporting."""
    if not rows:
        return []
    plt = _pyplot()
    arms = sorted(rows)
    figure, axis = plt.subplots(figsize=(6.5, 4))
    for index, arm in enumerate(arms):
        values = [float(value) for value in (rows[arm].get("per_seed") or [])]
        if not values:
            continue
        axis.scatter([index] * len(values), values, s=28, alpha=0.75, zorder=3)
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            axis.errorbar(index, mean, yerr=variance ** 0.5, fmt="_", markersize=22,
                          capsize=6, linewidth=1.5, zorder=2)
        else:
            axis.plot([index], [mean], "_", markersize=22)
    axis.set_xticks(range(len(arms))); axis.set_xticklabels(arms)
    axis.set_ylabel(metric); axis.set_title(title); axis.grid(alpha=0.3, axis="y")
    return [_finish(figure, out / filename)]


def forest(hypotheses: dict[str, dict[str, Any]], out: Path,
           filename: str = "forest_hypotheses.png") -> list[str]:
    """Effect sizes with bootstrap intervals, one row per hypothesis."""
    if not hypotheses:
        return []
    plt = _pyplot()
    names = sorted(hypotheses)
    figure, axis = plt.subplots(figsize=(7, 0.6 * len(names) + 2))
    for index, name in enumerate(names):
        row = hypotheses[name]
        point = row.get("effect")
        low, high = row.get("ci_low"), row.get("ci_high")
        if point is None:
            continue
        axis.plot([float(point)], [index], "o", markersize=6, zorder=3)
        if low is not None and high is not None:
            axis.plot([float(low), float(high)], [index, index], linewidth=2, zorder=2)
    axis.axvline(0.0, linestyle="--", linewidth=1, alpha=0.6)
    axis.set_yticks(range(len(names))); axis.set_yticklabels(names)
    axis.set_xlabel("effect (paired bootstrap)")
    axis.set_title("Hypothesis effects with bootstrap confidence intervals")
    axis.grid(alpha=0.3, axis="x")
    return [_finish(figure, out / filename)]


def roc_and_calibration(scored: dict[str, dict[str, Any]], out: Path) -> list[str]:
    """ROC curves, a reliability diagram and a threshold sweep."""
    if not scored:
        return []
    plt = _pyplot()
    written: list[str] = []

    figure, axis = plt.subplots(figsize=(5.5, 5))
    drew = False
    for name, block in sorted(scored.items()):
        curve = block.get("roc") or {}
        fpr, tpr = curve.get("fpr"), curve.get("tpr")
        if fpr and tpr:
            axis.plot(fpr, tpr, linewidth=1.4,
                      label=f"{name} (AUC={block.get('roc_auc', float('nan')):.3f})")
            drew = True
    if drew:
        axis.plot([0, 1], [0, 1], "--", linewidth=1, alpha=0.5)
        axis.set_xlabel("FPR"); axis.set_ylabel("TPR"); axis.set_title("ROC")
        axis.legend(fontsize=8); axis.grid(alpha=0.3)
        written.append(_finish(figure, out / "roc_curves.png"))
    else:
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 5))
    drew = False
    for name, block in sorted(scored.items()):
        curve = block.get("reliability") or {}
        confidence, accuracy = curve.get("confidence"), curve.get("accuracy")
        if confidence and accuracy:
            axis.plot(confidence, accuracy, marker="o", markersize=3, linewidth=1.2,
                      label=name)
            drew = True
    if drew:
        axis.plot([0, 1], [0, 1], "--", linewidth=1, alpha=0.5)
        axis.set_xlabel("confidence"); axis.set_ylabel("empirical accuracy")
        axis.set_title("Calibration reliability"); axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
        written.append(_finish(figure, out / "calibration_reliability.png"))
    else:
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 4))
    drew = False
    for name, block in sorted(scored.items()):
        sweep = block.get("threshold_sweep") or {}
        thresholds, values = sweep.get("threshold"), sweep.get("acer")
        if thresholds and values:
            axis.plot(thresholds, values, linewidth=1.3, label=name)
            drew = True
    if drew:
        axis.set_xlabel("threshold"); axis.set_ylabel("ACER")
        axis.set_title("Threshold sweep"); axis.legend(fontsize=8); axis.grid(alpha=0.3)
        written.append(_finish(figure, out / "threshold_metrics.png"))
    else:
        plt.close(figure)
    return written


def generate_all(evidence: dict[str, Any], out: Path) -> dict[str, Any]:
    """Draw every figure the evidence supports; name every one it does not.

    `evidence` is a plain mapping assembled by the caller from stored artifacts.
    Passing data rather than paths keeps this module unable to read a model or a
    dataset even by accident.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    errors: list[dict[str, str]] = []

    def attempt(name: str, function: Any, *args: Any, **kwargs: Any) -> None:
        try:
            written.extend(function(*args, **kwargs))
        except Exception as error:                           # noqa: BLE001 - reported
            errors.append({"figure": name, "error": f"{type(error).__name__}: {error}"})

    attempt("training_curves", training_curves, evidence.get("history") or [], out)
    attempt("bank_coverage", bank_coverage, evidence.get("banks") or {}, out)
    attempt("quality_gate", quality_gate, evidence.get("gate_summaries") or {},
            evidence.get("q_values") or {}, out)
    attempt("track_g_per_seed", seed_comparison, evidence.get("track_g_seeds") or {}, out,
            metric="video_ACER", filename="track_g_per_seed.png",
            title="Track G per-seed video ACER by arm")
    attempt("track_g_mean_std", seed_comparison, evidence.get("track_g_seeds") or {}, out,
            metric="video_ACER", filename="track_g_mean_std.png",
            title="Track G mean +/- std by arm")
    attempt("track_r_llm_vs_det", seed_comparison, evidence.get("track_r_seeds") or {},
            out, metric="video_ACER", filename="track_r_llm_vs_det.png",
            title="Track R: LLM versus DET")
    attempt("prompthead_ablation", seed_comparison,
            evidence.get("prompthead_seeds") or {}, out, metric="video_ACER",
            filename="prompthead_ablation.png", title="PromptHead ablation (C-H5)")
    attempt("frame_vs_video", seed_comparison, evidence.get("frame_vs_video") or {}, out,
            metric="ACER", filename="frame_vs_video.png",
            title="Frame-level versus video-level")
    attempt("forest_hypotheses", forest, evidence.get("hypotheses") or {}, out)
    attempt("ch4_mechanism", forest, evidence.get("ch4") or {}, out,
            filename="ch4_mechanism.png")
    attempt("roc_and_calibration", roc_and_calibration, evidence.get("scored") or {}, out)

    drawn = {name.replace(".png", "") for name in written}
    skipped = [{"figure": name, "needs": needs} for name, needs in DECLARED_FIGURES
               if name not in drawn]
    return {
        "schema_version": SCHEMA_VERSION,
        "output_dir": out.as_posix(),
        "declared": [name for name, _needs in DECLARED_FIGURES],
        "written": sorted(written),
        "written_count": len(written),
        "skipped": skipped,
        "skipped_count": len(skipped),
        "errors": errors,
        "source": "stored evidence only; no plot triggers training or opens a dataset",
        "deterministic": "Agg backend, no timestamps, no random colours",
    }


__all__ = ["SCHEMA_VERSION", "DECLARED_FIGURES", "training_curves", "bank_coverage",
           "quality_gate", "seed_comparison", "forest", "roc_and_calibration",
           "generate_all"]
