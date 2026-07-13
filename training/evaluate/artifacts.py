"""Emit the dissertation artifacts: LaTeX tables and figures (Workstreams 8, 9).

House rules enforced here (non-negotiable #6):
  * British English; no em dashes anywhere in generated text.
  * No italics in prose.
  * Table captions ABOVE tables; figure captions BELOW figures.
  * booktabs tables; class names with escaped underscores.
  * matplotlib only (no seaborn), one chart per figure, colour-blind-safe
    palette, PNG at 300 dpi plus PDF.

Everything reads from the ``results.json`` structure produced by the evaluation,
so tables and figures can be regenerated off-server at any time without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

from scgnn.schema import FLAWS

# Okabe-Ito: colour-blind-safe, print-safe.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]


def tex_escape(s: str) -> str:
    """Escape a class name for LaTeX (underscores are the only hazard here)."""
    return str(s).replace("_", r"\_")


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


# --------------------------------- tables ---------------------------------

def table_dual_benchmark(results: dict, model: str, *, label: str = "tab:dual") -> str:
    """T1: per-class F1 + macro-F1 with CIs, Test A vs Test B, for one model.

    Caption above the table, booktabs, no italics. Pure (returns LaTeX text).
    """
    a = results[model]["test_a"]
    b = results[model]["test_b"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Per-class and macro F1 of {tex_escape(model)} on the two frozen "
        rf"test sets, with bootstrap 95 per cent confidence intervals. Test A is "
        rf"tool-labelled; Test B is expert-labelled.}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Class & Test A F1 & Test A 95\% CI & Test B F1 & Test B 95\% CI \\",
        r"\midrule",
    ]
    for f in FLAWS:
        fa = a["per_flaw"][f]["f1"]
        fb = b["per_flaw"][f]["f1"]
        ca = a.get("ci", {}).get("per_flaw", {}).get(f, {})
        cb = b.get("ci", {}).get("per_flaw", {}).get(f, {})
        lines.append(
            f"{tex_escape(f)} & {_fmt(fa)} & "
            f"[{_fmt(ca.get('lo'))}, {_fmt(ca.get('hi'))}] & {_fmt(fb)} & "
            f"[{_fmt(cb.get('lo'))}, {_fmt(cb.get('hi'))}] \\\\")
    lines += [
        r"\midrule",
        (f"Macro & {_fmt(a['macro']['f1'])} & "
         f"[{_fmt(a.get('ci',{}).get('macro_f1',{}).get('lo'))}, "
         f"{_fmt(a.get('ci',{}).get('macro_f1',{}).get('hi'))}] & "
         f"{_fmt(b['macro']['f1'])} & "
         f"[{_fmt(b.get('ci',{}).get('macro_f1',{}).get('lo'))}, "
         f"{_fmt(b.get('ci',{}).get('macro_f1',{}).get('hi'))}] \\\\"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table_model_matrix(results: dict, *, label: str = "tab:matrix") -> str:
    """T2: macro-F1 of every run on val / Test A / Test B."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Macro F1 of every trained configuration on the validation "
        r"split and the two frozen test sets.}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Run & Validation & Test A & Test B \\",
        r"\midrule",
    ]
    for model in sorted(results):
        r = results[model]
        lines.append(
            f"{tex_escape(model)} & {_fmt(r.get('val',{}).get('macro',{}).get('f1'))} & "
            f"{_fmt(r.get('test_a',{}).get('macro',{}).get('f1'))} & "
            f"{_fmt(r.get('test_b',{}).get('macro',{}).get('f1'))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_ablation(results: dict, encoders: list[str], *,
                   label: str = "tab:ablation") -> str:
    """T3: with vs without data-flow edges, per encoder, on both test sets.

    Expects run keys ``<encoder>`` (without) and ``<encoder>_df`` (with).
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Effect of adding data-flow edges to the control-flow graph. "
        r"Macro F1 with and without the data-flow representation, per encoder.}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Encoder & Test A without DF & Test A with DF & Test B without DF & Test B with DF \\",
        r"\midrule",
    ]
    for e in encoders:
        wo, wi = results.get(e, {}), results.get(f"{e}_df", {})
        lines.append(
            f"{tex_escape(e)} & "
            f"{_fmt(wo.get('test_a',{}).get('macro',{}).get('f1'))} & "
            f"{_fmt(wi.get('test_a',{}).get('macro',{}).get('f1'))} & "
            f"{_fmt(wo.get('test_b',{}).get('macro',{}).get('f1'))} & "
            f"{_fmt(wi.get('test_b',{}).get('macro',{}).get('f1'))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_durieux(matrix: dict, *, label: str = "tab:durieux") -> str:
    """T5: the tool-vs-model matrix on Test B (per-class recall + false warnings)."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-class recall of each analysis tool, the union of the four "
        r"tools, and the trained models, evaluated on the expert test set. The "
        r"final column reports the fraction of flaw-free contracts that each row "
        r"flags, the false-warning analogue of Durieux et al. (2020).}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "c" * (len(FLAWS) + 2) + "}",
        r"\toprule",
        "Detector & " + " & ".join(tex_escape(f) for f in FLAWS)
        + r" & Macro F1 & False warnings \\",
        r"\midrule",
    ]
    for name in matrix:
        row = matrix[name]
        rec = " & ".join(_fmt(row["per_flaw"][f]["recall"]) for f in FLAWS)
        lines.append(f"{tex_escape(name)} & {rec} & {_fmt(row['macro']['f1'])} & "
                     f"{_fmt(row['false_warning_rate'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------- figures ---------------------------------

def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")


def fig_model_comparison(results: dict, out_dir: Path,
                         name: str = "F4.6_model_comparison") -> None:
    """F4.6 (headline): macro-F1 on val / Test A / Test B for every run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = sorted(results)
    splits = [("val", "Validation"), ("test_a", "Test A (tool-labelled)"),
              ("test_b", "Test B (expert)")]
    x = np.arange(len(models))
    w = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (key, lab) in enumerate(splits):
        vals = [results[m].get(key, {}).get("macro", {}).get("f1", 0) for m in models]
        ax.bar(x + i * w, vals, w, label=lab, color=PALETTE[i], edgecolor="white")
    ax.set_xticks(x + w * (len(splits) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ") for m in models], rotation=20, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_ablation(results: dict, encoders: list[str], out_dir: Path,
                 name: str = "F4.8_dataflow_ablation") -> None:
    """F4.8: paired bars, with and without data-flow edges, per encoder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(encoders))
    w = 0.38
    wo = [results.get(e, {}).get("test_b", {}).get("macro", {}).get("f1", 0)
          for e in encoders]
    wi = [results.get(f"{e}_df", {}).get("test_b", {}).get("macro", {}).get("f1", 0)
          for e in encoders]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, wo, w, label="Without data-flow edges", color=PALETTE[0],
           edgecolor="white")
    ax.bar(x + w / 2, wi, w, label="With data-flow edges", color=PALETTE[1],
           edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(encoders)
    ax.set_ylabel("Macro F1 on the expert test set")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_ab_gap(results: dict, out_dir: Path, name: str = "F4.12_ab_gap") -> None:
    """F4.12: the Test A vs Test B gap per model, visualising label-provenance cost."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = sorted(results)
    x = np.arange(len(models))
    w = 0.38
    a = [results[m].get("test_a", {}).get("macro", {}).get("f1", 0) for m in models]
    b = [results[m].get("test_b", {}).get("macro", {}).get("f1", 0) for m in models]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, a, w, label="Test A (tool-labelled)", color=PALETTE[2],
           edgecolor="white")
    ax.bar(x + w / 2, b, w, label="Test B (expert)", color=PALETTE[3],
           edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in models], rotation=20, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, name)
    plt.close(fig)


def emit_all(results_path: str | Path, out_root: str | Path,
             encoders: list[str] | None = None,
             durieux: dict | None = None) -> dict:
    """Emit every table and figure from a results.json. Returns what was written."""
    encoders = encoders or ["gcn", "sage", "gatv2"]
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    out = Path(out_root)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    figs = out / "figures"

    best = max(results, key=lambda m: results[m].get("test_b", {})
               .get("macro", {}).get("f1", 0))
    written = {"tables": [], "figures": [], "best_model": best}

    tables = {
        "T1_dual_benchmark.tex": table_dual_benchmark(results, best),
        "T2_model_matrix.tex": table_model_matrix(results),
        "T3_ablation.tex": table_ablation(results, encoders),
    }
    if durieux:
        tables["T5_durieux.tex"] = table_durieux(durieux)
    for fn, tex in tables.items():
        (out / "tables" / fn).write_text(tex, encoding="utf-8")
        written["tables"].append(fn)

    fig_model_comparison(results, figs)
    fig_ablation(results, encoders, figs)
    fig_ab_gap(results, figs)
    written["figures"] = ["F4.6_model_comparison", "F4.8_dataflow_ablation",
                          "F4.12_ab_gap"]
    return written