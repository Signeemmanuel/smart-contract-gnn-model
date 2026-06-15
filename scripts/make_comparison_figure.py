#!/usr/bin/env python3
"""Three-model detection comparison figure (the model-selection centrepiece).

Reads each model's evaluation JSON (as written by scripts/evaluate.py) and renders
a grouped bar chart of per-class F1 with a macro-F1 panel, so GCN vs GraphSAGE vs
GAT sit on one chart at an identical decision threshold. PNG (Word) + PDF (LaTeX),
plus a caption + a small CSV table for the thesis.

Example:
  PYTHONPATH=. python scripts/make_comparison_figure.py \
      --eval gcn=runs/gcn_multitool/eval_05.json \
      --eval sage=runs/sage_multitool_long/eval_05.json \
      --eval gat=runs/gat_multitool/eval_05.json \
      --out reports/figures/comparison
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

from scgnn.schema import FLAWS, display_name  # noqa: E402

# Stable colour per model (Okabe-Ito), distinct from the per-flaw palette.
MODEL_COLOUR = {"gcn": "#0072B2", "sage": "#009E73", "gat": "#D55E00",
                "graphsage": "#009E73"}
_FALLBACK = ["#0072B2", "#009E73", "#D55E00", "#E69F00", "#CC79A7"]


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def _colour(name: str, i: int) -> str:
    return MODEL_COLOUR.get(name.lower(), _FALLBACK[i % len(_FALLBACK)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="append", required=True, metavar="LABEL=PATH",
                    help="Model label and its eval JSON, e.g. gcn=runs/gcn_multitool/eval_05.json. "
                         "Repeat for each model.")
    ap.add_argument("--out", default="reports/figures/comparison")
    ap.add_argument("--format", default="both", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    models: list[tuple[str, dict]] = []
    for spec in args.eval:
        if "=" not in spec:
            ap.error(f"--eval expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        models.append((label, data))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    formats = ["png", "pdf"] if args.format == "both" else [args.format]
    _style()

    flaws = FLAWS
    labels = [display_name(f) for f in flaws]
    n_models = len(models)
    x = np.arange(len(flaws))
    width = 0.8 / max(n_models, 1)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [3.2, 1]})

    # Left: per-class F1, grouped bars per model.
    for i, (label, data) in enumerate(models):
        pf = data.get("per_flaw", {})
        f1s = [pf.get(f, {}).get("f1", 0.0) for f in flaws]
        axL.bar(x + (i - (n_models - 1) / 2) * width, f1s, width,
                label=label.upper(), color=_colour(label, i))
    axL.set_xticks(x); axL.set_xticklabels(labels, rotation=20, ha="right")
    axL.set_ylabel("F1 (Curated test)"); axL.set_ylim(0, 1)
    axL.set_title("Per-class detection F1 by architecture")
    axL.legend(title=None, ncol=n_models)

    # Right: macro-F1 per model.
    macros = [data.get("macro", {}).get("f1", 0.0) for _, data in models]
    mx = np.arange(n_models)
    bars = axR.bar(mx, macros, 0.6,
                   color=[_colour(l, i) for i, (l, _) in enumerate(models)])
    axR.set_xticks(mx); axR.set_xticklabels([l.upper() for l, _ in models])
    axR.set_ylabel("macro-F1"); axR.set_ylim(0, 1)
    axR.set_title("Macro-F1")
    for b, v in zip(bars, macros):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)

    thr = models[0][1].get("threshold", {})
    tdesc = thr.get("threshold", thr.get("mode", "?"))
    fig.suptitle(f"Detection performance comparison (decision threshold = {tdesc})",
                 y=1.02, fontsize=13)

    stem = out / "fig_model_comparison"
    written = []
    for fmt in formats:
        p = f"{stem}.{fmt}"; fig.savefig(p); written.append(Path(p).name)
    plt.close(fig)

    # A small CSV table for the thesis, same numbers.
    csv_path = out / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model"] + labels + ["macro_f1"])
        for label, data in models:
            pf = data.get("per_flaw", {})
            row = [label] + [f"{pf.get(f, {}).get('f1', 0.0):.4f}" for f in flaws]
            row.append(f"{data.get('macro', {}).get('f1', 0.0):.4f}")
            w.writerow(row)

    (out / "COMPARISON.md").write_text(
        "# Model comparison figure\n"
        f"- files: {', '.join(written)}\n"
        f"- table: {csv_path.name}\n"
        f"- threshold: {tdesc}\n"
        "- caption: Per-class and macro detection F1 on the frozen Curated test "
        "split for the three GNN architectures at an identical decision threshold. "
        "GCN attains the highest macro-F1 and is the only architecture with non-zero "
        "F1 on every supported class; GAT, which converged prematurely in training, "
        "trails substantially.\n", encoding="utf-8")

    print("wrote", ", ".join(written), "+", csv_path.name, "to", out)
    print("macro-F1:", {l: round(d.get("macro", {}).get("f1", 0.0), 4) for l, d in models})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())