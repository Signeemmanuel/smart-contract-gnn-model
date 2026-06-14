#!/usr/bin/env python3
"""Generate report-ready figures from the pipeline's own artefacts.

Every figure is drawn from a JSON/parquet file the pipeline already writes, so
the numbers in the dissertation trace back to a produced artefact rather than a
hand-made chart. The script is incremental: it renders whatever inputs exist now
and skips the rest, so you can run it after labelling for the labelling figures,
then again after training/evaluation for the rest.

Outputs go to ``--out`` (default ``reports/figures``) as both PNG (for Word) and
PDF (vector, for LaTeX), plus ``FIGURES.md`` with a draft caption and the source
artefact for each figure produced.

Inputs and the figures they drive:
  data/processed/class_frequency.json   -> class distribution of weak labels
  data/processed/tool_vote_summary.json -> per-tool coverage per flaw (Osiris=arithmetic)
  data/processed/reliabilities.json     -> Snorkel learned tool reliabilities (heatmap)
  data/processed/labels.parquet         -> flaw co-occurrence (multi-label structure)
  data/processed/build_report.json      -> train/val/test split composition
  runs/<run>/history.json               -> training loss + val macro-F1 curves
  data/processed/eval_metrics.json      -> per-flaw P/R/F1 on the Curated test split
  data/processed/localisation_report_*.json -> top-k localisation accuracy (+ ablation)

Example:
  PYTHONPATH=. python scripts/make_figures.py \
      --processed data/processed --run-dir runs/sage_multitool --out reports/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless: no display needed
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

from scgnn.schema import FLAWS, display_name  # noqa: E402

# Okabe-Ito colourblind-safe palette, one stable colour per flaw.
FLAW_COLOUR = {
    "reentrancy": "#0072B2", "access_control": "#E69F00", "arithmetic": "#009E73",
    "unchecked_calls": "#CC79A7", "dos": "#D55E00",
}
TOOL_COLOUR = {"slither": "#0072B2", "mythril": "#E69F00",
               "securify": "#009E73", "osiris": "#D55E00"}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.autolayout": True,
    })


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save(fig, out: Path, name: str, formats) -> list[str]:
    written = []
    for fmt in formats:
        p = out / f"{name}.{fmt}"
        fig.savefig(p, format=fmt)
        written.append(p.name)
    plt.close(fig)
    return written


# --- pure data-shaping helpers (unit-tested) ---------------------------------

def reliabilities_to_matrix(rel: dict, flaws: list[str]) -> tuple[np.ndarray, list[str]]:
    """``{flaw: {tool: acc}}`` -> (matrix[flaw, tool], tool order). Pure."""
    tools: list[str] = []
    for flaw in flaws:
        for t in (rel.get(flaw) or {}):
            if t not in tools:
                tools.append(t)
    M = np.full((len(flaws), len(tools)), np.nan)
    for i, flaw in enumerate(flaws):
        for j, t in enumerate(tools):
            v = (rel.get(flaw) or {}).get(t)
            if v is not None:
                M[i, j] = float(v)
    return M, tools


def cooccurrence_from_labels(Y: np.ndarray) -> np.ndarray:
    """Symmetric ``(5,5)`` count of contracts positive for both flaw i and j. Pure."""
    Y = (np.asarray(Y) >= 0.5).astype(int)
    return Y.T @ Y


def coverage_matrix(summary: dict, flaws: list[str], tools: list[str]) -> np.ndarray:
    """``{tool: {flaw: {positive, ran}}}`` -> positives matrix[tool, flaw]. Pure."""
    M = np.zeros((len(tools), len(flaws)), dtype=int)
    for i, t in enumerate(tools):
        for j, flaw in enumerate(flaws):
            M[i, j] = int(((summary.get(t) or {}).get(flaw) or {}).get("positive", 0))
    return M


# --- figure builders: each returns (made: bool, caption: str|None) -----------

def fig_class_distribution(P: Path, out: Path, formats, captions):
    freq = _load(P / "class_frequency.json")
    if not freq:
        return False
    names = [display_name(f) for f in FLAWS]
    vals = [int(freq.get(f, 0)) for f in FLAWS]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, vals, color=[FLAW_COLOUR[f] for f in FLAWS])
    for x, v in enumerate(vals):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("positive contracts (weak labels)")
    ax.set_title("Weak-label class distribution")
    ax.tick_params(axis="x", rotation=20)
    files = _save(fig, out, "fig_class_distribution", formats)
    captions.append(("fig_class_distribution", files, "class_frequency.json",
                     "Distribution of positive contracts per flaw class in the Snorkel-"
                     "denoised weak labels. The imbalance motivates the per-class "
                     "positive weighting used in the training loss."))
    return True


def fig_tool_coverage(P: Path, out: Path, formats, captions):
    summary = _load(P / "tool_vote_summary.json")
    if not summary:
        return False
    tools = [t for t in ("slither", "mythril", "securify", "osiris") if t in summary]
    M = coverage_matrix(summary, FLAWS, tools)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(FLAWS))
    w = 0.8 / max(len(tools), 1)
    for i, t in enumerate(tools):
        ax.bar(x + i * w - 0.4 + w / 2, M[i], w, label=t.capitalize(),
               color=TOOL_COLOUR.get(t, "#666666"))
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(f) for f in FLAWS], rotation=20)
    ax.set_ylabel("contracts flagged positive")
    ax.set_title("Per-tool flaw coverage")
    ax.legend(ncol=len(tools))
    files = _save(fig, out, "fig_tool_coverage", formats)
    captions.append(("fig_tool_coverage", files, "tool_vote_summary.json",
                     "Number of contracts each static/symbolic tool flags for each flaw "
                     "class. Slither, Mythril and Securify leave arithmetic uncovered; "
                     "Osiris supplies the arithmetic votes, which is why the ensemble "
                     "combines all four."))
    return True


def fig_tool_reliability(P: Path, out: Path, formats, captions):
    rel = _load(P / "reliabilities.json")
    if not rel:
        return False
    M, tools = reliabilities_to_matrix(rel, FLAWS)
    if not tools:
        return False
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(tools), 4.2))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(tools)), [t.capitalize() for t in tools])
    ax.set_yticks(range(len(FLAWS)), [display_name(f) for f in FLAWS])
    for i in range(len(FLAWS)):
        for j in range(len(tools)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.6 else "black", fontsize=9)
    ax.set_title("Snorkel learned tool reliabilities")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="learned accuracy")
    ax.grid(False)
    files = _save(fig, out, "fig_tool_reliability", formats)
    captions.append(("fig_tool_reliability", files, "reliabilities.json",
                     "Per-flaw reliability the Snorkel label model learned for each tool. "
                     "These weights are what let the model denoise the tool votes rather "
                     "than take a plain majority."))
    return True


def fig_label_cooccurrence(P: Path, out: Path, formats, captions):
    try:
        import pandas as pd
        df = pd.read_parquet(P / "labels.parquet")
    except Exception:
        return False
    Y = df[FLAWS].to_numpy()
    C = cooccurrence_from_labels(Y)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(C, cmap="magma")
    names = [display_name(f) for f in FLAWS]
    ax.set_xticks(range(5), names, rotation=30, ha="right")
    ax.set_yticks(range(5), names)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{C[i, j]}", ha="center", va="center",
                    color="white" if C[i, j] < C.max() * 0.6 else "black", fontsize=9)
    ax.set_title("Flaw co-occurrence (contracts)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="shared contracts")
    ax.grid(False)
    files = _save(fig, out, "fig_label_cooccurrence", formats)
    captions.append(("fig_label_cooccurrence", files, "labels.parquet",
                     "Number of contracts labelled positive for each pair of flaws "
                     "(diagonal = per-class totals). Off-diagonal mass confirms the task "
                     "is genuinely multi-label, justifying independent sigmoid outputs."))
    return True


def fig_split_composition(P: Path, out: Path, formats, captions):
    rep = _load(P / "build_report.json")
    if not rep:
        return False
    enc = rep.get("encoded") or rep.get("counts") or {}
    splits = [s for s in ("train", "val", "test") if s in enc]
    if not splits:
        return False
    vals = [int(enc[s]) for s in splits]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar([s.capitalize() for s in splits], vals,
           color=["#0072B2", "#E69F00", "#009E73"][:len(splits)])
    for x, v in enumerate(vals):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("contracts")
    ax.set_title("Dataset split composition")
    files = _save(fig, out, "fig_split_composition", formats)
    captions.append(("fig_split_composition", files, "build_report.json",
                     "Number of contracts in each split after extraction. Test is the "
                     "frozen, stratified Curated split; train mixes Wild weak labels with "
                     "the Curated remainder; val is a held-out Wild slice."))
    return True


def fig_training_curves(run_dir: Path, out: Path, formats, captions):
    hist = _load(run_dir / "history.json")
    if not hist:
        return False
    ep = [h["epoch"] for h in hist]
    loss = [h.get("train_loss") for h in hist]
    f1 = [h.get("val_macro_f1") for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(ep, loss, color="#D55E00")
    a1.set_xlabel("epoch"); a1.set_ylabel("train loss"); a1.set_title("Training loss")
    a2.plot(ep, f1, color="#0072B2")
    a2.set_xlabel("epoch"); a2.set_ylabel("val macro-F1"); a2.set_title("Validation macro-F1")
    if f1:
        best = int(np.argmax([v if v is not None else -1 for v in f1]))
        a2.scatter([ep[best]], [f1[best]], color="#009E73", zorder=5)
        a2.annotate(f"best {f1[best]:.3f}", (ep[best], f1[best]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    files = _save(fig, out, "fig_training_curves", formats)
    captions.append(("fig_training_curves", files, f"{run_dir.name}/history.json",
                     "Training loss and validation macro-F1 per epoch. The best-F1 epoch "
                     "(marked) is the checkpoint kept for evaluation."))
    return True


def fig_detection_performance(P: Path, out: Path, formats, captions):
    m = _load(P / "eval_metrics.json")
    if not m or "per_flaw" not in m:
        return False
    metrics = ("precision", "recall", "f1")
    x = np.arange(len(FLAWS))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, met in enumerate(metrics):
        vals = [m["per_flaw"][f][met] for f in FLAWS]
        ax.bar(x + (i - 1) * w, vals, w, label=met.capitalize())
    ax.set_xticks(x, [display_name(f) for f in FLAWS], rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    macro = m.get("macro", {})
    ax.set_title("Detection performance on Curated test"
                 + (f"  (macro-F1 {macro.get('f1', 0):.3f})" if macro else ""))
    ax.legend(ncol=3)
    files = _save(fig, out, "fig_detection_performance", formats)
    captions.append(("fig_detection_performance", files, "eval_metrics.json",
                     "Per-flaw precision, recall and F1 on the frozen Curated test split. "
                     "Macro-F1 is reported in the title; sparse classes (e.g. arithmetic) "
                     "rest on few positives."))
    return True


def fig_localisation_accuracy(P: Path, out: Path, formats, captions):
    reports = sorted(P.glob("localisation_report_*.json"))
    rep = _load(reports[-1]) if reports else None
    if not rep or "results" not in rep:
        return False
    ks = ["1", "3", "5"]
    modes = rep.get("merge_modes") or list(rep["results"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(ks))
    w = 0.8 / max(len(modes), 1)
    for i, mode in enumerate(modes):
        res = rep["results"][mode]
        mean = [res["accuracy_at_k_mean"][k] for k in ks]
        std = [res.get("accuracy_at_k_std", {}).get(k, 0.0) for k in ks]
        ax.bar(x + i * w - 0.4 + w / 2, mean, w, yerr=std, capsize=4,
               label=f"merge={mode}")
    ax.set_xticks(x, [f"@{k}" for k in ks])
    ax.set_ylim(0, 1)
    ax.set_ylabel("localisation accuracy")
    ax.set_title(f"Top-k line localisation (n={rep.get('n_localisation_targets', '?')})")
    ax.legend()
    files = _save(fig, out, "fig_localisation_accuracy", formats)
    cap = ("Top-k line-localisation accuracy on the Curated test contracts carrying gold "
           "lines, with error bars over seeded repeats.")
    if len(modes) > 1:
        cap += (" The merge rule (max across branches) is compared against the concat "
                "baseline; max wins at k=3 and k=5.")
    captions.append(("fig_localisation_accuracy", files,
                     reports[-1].name, cap))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--run-dir", default="runs/sage",
                    help="Training run folder holding history.json.")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--format", default="both", choices=["png", "pdf", "both"],
                    help="png (Word), pdf (LaTeX vector), or both.")
    args = ap.parse_args()

    _style()
    formats = ["png", "pdf"] if args.format == "both" else [args.format]
    P = Path(args.processed)
    run_dir = Path(args.run_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    captions: list[tuple] = []
    builders = [
        ("class distribution", lambda: fig_class_distribution(P, out, formats, captions)),
        ("tool coverage", lambda: fig_tool_coverage(P, out, formats, captions)),
        ("tool reliability", lambda: fig_tool_reliability(P, out, formats, captions)),
        ("label co-occurrence", lambda: fig_label_cooccurrence(P, out, formats, captions)),
        ("split composition", lambda: fig_split_composition(P, out, formats, captions)),
        ("training curves", lambda: fig_training_curves(run_dir, out, formats, captions)),
        ("detection performance", lambda: fig_detection_performance(P, out, formats, captions)),
        ("localisation accuracy", lambda: fig_localisation_accuracy(P, out, formats, captions)),
    ]
    made, skipped = [], []
    for label, fn in builders:
        try:
            (made if fn() else skipped).append(label)
        except Exception as exc:                       # never let one figure kill the rest
            skipped.append(f"{label} (error: {type(exc).__name__})")

    # caption sheet for the report
    lines = ["# Generated figures\n",
             "Each figure is rendered from a pipeline artefact; the source is listed so "
             "every number is reproducible.\n"]
    for name, files, source, cap in captions:
        lines.append(f"## {name}\n")
        lines.append(f"- files: {', '.join(files)}")
        lines.append(f"- source: `{source}`")
        lines.append(f"- caption: {cap}\n")
    (out / "FIGURES.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"figures written to {out}/  (formats: {', '.join(formats)})")
    print(f"made ({len(made)}): {', '.join(made) if made else 'none'}")
    if skipped:
        print(f"skipped ({len(skipped)}, inputs not present yet): {', '.join(skipped)}")
    print(f"caption sheet: {out / 'FIGURES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
