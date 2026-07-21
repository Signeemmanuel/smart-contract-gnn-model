"""Emit the dissertation artifacts: LaTeX tables and figures (Workstreams 8, 9).

House rules enforced here (non-negotiable #6):
  * British English; no em dashes anywhere in generated text.
  * No italics in prose.
  * Table captions ABOVE tables; figure captions BELOW figures.
  * booktabs tables; class names with escaped underscores.
  * matplotlib only (no seaborn), one chart per figure, colour-blind-safe
    palette, PNG at 300 dpi plus PDF.

Everything reads from the ``results.json`` structure produced by the evaluation
plus the small JSON reports the pipeline leaves behind (class frequencies, tool
votes, test-set summary, build reports, training histories, the localisation
benchmark), so tables and figures can be regenerated off-server at any time
without a GPU. Every emitter is guarded: a missing input skips that artifact
with a note instead of failing the stage.

Coverage:
  Tables   T1 dual benchmark, T2 model matrix, T3 data-flow ablation,
           T4 ensemble vs best single, T5 Durieux-style matrix,
           T6 dataset and labelling scale, T7 before/after v1 vs v2.
  Figures  F3.1-F3.9 methodology schematics (pipeline, labelling, graph views,
           node features, splits and firewall, architecture, ensemble,
           localisation, evaluation framework);
           F4.1-F4.14 results (distributions, co-occurrence, splits, tool
           findings, training curves, headline bars, per-class heatmaps,
           ablation, ensemble deltas, confusions, Durieux, A-vs-B gap,
           before/after, localisation).
  Plus     per-run training-curve CSVs and RESULTS_SUMMARY.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from scgnn.schema import FLAWS

# Okabe-Ito: colour-blind-safe, print-safe.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]

# The v1 (predefence) record, frozen for the before/after comparison (T7, F4.13).
V1_RECORD = {
    "dataset_contracts": 2488,
    "val_macro": {"gcn": 0.725, "sage": 0.736, "gat": 0.635},
    "expert_macro": {"gcn": 0.578, "sage": 0.454, "gat": 0.174},
    "selected": "gcn",
    "localisation_at": {1: 0.043, 3: 0.119, 5: 0.157},
}


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


def _load_json(path) -> dict | list | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


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


def table_ensemble(results: dict, *, label: str = "tab:ensemble") -> str:
    """T4: ensemble against the best single model, per arm, both test sets."""
    def best_single(arm_suffix: str) -> str | None:
        cands = [m for m in results
                 if "ensemble" not in m and m.endswith(arm_suffix)
                 and (arm_suffix or "_df" not in m)]
        if not cands:
            return None
        return max(cands, key=lambda m: results[m].get("test_b", {})
                   .get("macro", {}).get("f1", 0))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{The per-class mean ensemble against the best single model in "
        r"each data-flow arm. Macro F1 on both frozen test sets; the max policy "
        r"is reported for the record.}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Arm & Model & Test A & Test B & Test B (max policy) \\",
        r"\midrule",
    ]
    for arm, ens_name, suffix in (("without DF", "ensemble", ""),
                                  ("with DF", "ensemble_df", "_df")):
        bs = best_single(suffix)
        for name, role in ((bs, "best single"), (ens_name, "ensemble")):
            if name is None or name not in results:
                continue
            r = results[name]
            mx = r.get("test_b", {}).get("max_policy_macro_f1")
            lines.append(
                f"{arm} & {tex_escape(name)} ({role}) & "
                f"{_fmt(r.get('test_a',{}).get('macro',{}).get('f1'))} & "
                f"{_fmt(r.get('test_b',{}).get('macro',{}).get('f1'))} & "
                f"{_fmt(mx) if mx is not None else 'n/a'} \\\\")
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


def table_dataset_scale(class_freq: dict | None, testsets_summary: dict | None,
                        build_report: dict | None, tool_votes: dict | None, *,
                        label: str = "tab:scale") -> str:
    """T6: dataset and labelling scale, v2."""
    rows: list[tuple[str, str]] = []
    if build_report:
        c = build_report.get("counts", {})
        rows += [("Labelled Wild pool (train + val source)", str(c.get("wild_pool", "n/a"))),
                 ("Training contracts (encoded)",
                  str(build_report.get("encoded", {}).get("train", "n/a"))),
                 ("Validation contracts (encoded)",
                  str(build_report.get("encoded", {}).get("val", "n/a"))),
                 ("Test A contracts (frozen / encoded)",
                  f"{c.get('test_a', 'n/a')} / "
                  f"{build_report.get('encoded', {}).get('test_a', 'n/a')}"),
                 ("Test B contracts (frozen / encoded)",
                  f"{c.get('test_b', 'n/a')} / "
                  f"{build_report.get('encoded', {}).get('test_b', 'n/a')}"),
                 ("Reserved (firewalled) content hashes",
                  str(c.get("reserved_hashes", "n/a"))),
                 ("Data-flow edges (with-DF build)",
                  str(build_report.get("data_flow_edges", "n/a")))]
    if class_freq:
        for f in FLAWS:
            if f in class_freq:
                rows.append((f"Positives: {f}", str(class_freq[f])))
    if tool_votes:
        for tool, per in tool_votes.items():
            ran = None
            for f in FLAWS:
                cell = per.get(f) if isinstance(per, dict) else None
                if isinstance(cell, dict) and "ran" in cell:
                    ran = max(ran or 0, cell["ran"])
            if ran is not None:
                rows.append((f"Contracts analysed by {tool}", str(ran)))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Scale of the v2 dataset and labelling effort. Positives are "
        r"union-rule labels over the full labelled pool; encoded counts are the "
        r"contracts Slither could represent as graphs.}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
    ]
    for k, v in rows:
        lines.append(f"{tex_escape(k)} & {tex_escape(v)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_before_after(results: dict, best: str, class_freq: dict | None,
                       localisation: dict | None, *,
                       label: str = "tab:beforeafter") -> str:
    """T7: v1 (predefence) against v2, the headline before/after."""
    v2_expert = results.get(best, {}).get("test_b", {}).get("macro", {}).get("f1")
    v2_pool = sum(1 for _ in []) or None
    n_labelled = None
    if class_freq is not None:
        n_labelled = class_freq.get("_n_contracts")
    loc_cells = ("n/a", "n/a", "n/a")
    if localisation:
        models = localisation.get("models", {})
        if best in models:
            m = models[best]["results_by_tolerance"]["0"]["accuracy_at_k_mean"]
            loc_cells = (_fmt(m.get("1")), _fmt(m.get("3")), _fmt(m.get("5")))
    v1 = V1_RECORD
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Before and after: the v1 predefence system against v2. The v1 "
        r"figures are the frozen predefence record; v2 figures come from the "
        r"eight-run matrix on the frozen test sets.}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Quantity & v1 (predefence) & v2 \\",
        r"\midrule",
        f"Labelled contracts & {v1['dataset_contracts']} & "
        f"{n_labelled if n_labelled else 'see Table T6'} \\\\",
        f"Selected model & {tex_escape(v1['selected'])} & {tex_escape(best)} \\\\",
        f"Expert-set macro F1 (selected) & {_fmt(v1['expert_macro']['gcn'])} & "
        f"{_fmt(v2_expert)} \\\\",
        f"Expert-set macro F1 (GAT / GATv2) & {_fmt(v1['expert_macro']['gat'])} & "
        f"{_fmt(results.get('gatv2_df', results.get('gatv2', {})).get('test_b', {}).get('macro', {}).get('f1'))} \\\\",
        f"Localisation accuracy at 1 / 3 / 5 & "
        f"{_fmt(v1['localisation_at'][1])} / {_fmt(v1['localisation_at'][3])} / "
        f"{_fmt(v1['localisation_at'][5])} & "
        f"{loc_cells[0]} / {loc_cells[1]} / {loc_cells[2]} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ------------------------------ figure helpers ------------------------------

def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _box(ax, x, y, w, h, text, colour="#0072B2", fs=9):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                linewidth=1.2, edgecolor=colour,
                                facecolor=colour + "22"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            wrap=True)


def _arrow(ax, x0, y0, x1, y1, colour="#444444", style="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=colour, lw=1.4,
                                linestyle=style))


def _schematic_axes(figsize=(11, 5)):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return plt, fig, ax


# ---------------------------- F3.x methodology ----------------------------

def fig_pipeline(out_dir: Path, name: str = "F3.1_pipeline") -> None:
    """F3.1: the updated end-to-end v2 pipeline."""
    plt, fig, ax = _schematic_axes((12, 5))
    _box(ax, 1, 62, 13, 20, "SmartBugs Wild\n47k contracts", PALETTE[0])
    _box(ax, 17, 62, 14, 20, "Dedup by\ncontent hash", PALETTE[0])
    _box(ax, 34, 62, 16, 20, "Four tools via\nSmartBugs\n(Slither, Mythril,\nSecurify, Osiris)", PALETTE[1])
    _box(ax, 53, 62, 13, 20, "Union rule\nlabels", PALETTE[1])
    _box(ax, 69, 62, 14, 20, "Freeze Test A\nand Test B", PALETTE[3])
    _box(ax, 86, 62, 13, 20, "Firewalled\ntrain / val", PALETTE[3])
    _box(ax, 10, 20, 22, 22, "Build: AST + CFG(+DF)\ngraphs, CodeBERT + PCA\nnode features", PALETTE[2])
    _box(ax, 38, 20, 24, 22, "Eight-run matrix\n(GCN, SAGE, GATv2, hybrid\nx with/without DF)\n+ two ensembles", PALETTE[2])
    _box(ax, 68, 20, 14, 22, "Evaluate on\nTest A and B\n(CIs, Durieux)", PALETTE[4])
    _box(ax, 85, 20, 14, 22, "Explain and\nlocalise\n(GNNExplainer)", PALETTE[4])
    for x0, x1 in ((14, 17), (31, 34), (50, 53), (66, 69), (83, 86)):
        _arrow(ax, x0, 72, x1, 72)
    _arrow(ax, 92, 62, 21, 42)
    _arrow(ax, 32, 31, 38, 31)
    _arrow(ax, 62, 31, 68, 31)
    _arrow(ax, 82, 31, 85, 31)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_labelling_pipeline(out_dir: Path, name: str = "F3.2_labelling_pipeline") -> None:
    """F3.2: the scaled, resumable labelling orchestrator."""
    plt, fig, ax = _schematic_axes((11, 5))
    _box(ax, 2, 66, 16, 20, "Wild pool\n(shuffled,\nhash-deduped)", PALETTE[0])
    _box(ax, 24, 66, 18, 20, "SQLite ledger\nper (contract, tool)\npending / ok /\ntimeout / crash /\nno output", PALETTE[3])
    _box(ax, 48, 66, 18, 20, "Worker pool\none SmartBugs\ninvocation per task,\nper-task timeout", PALETTE[1])
    _box(ax, 72, 66, 24, 20, "Results tree\ntool / run / contract /\nresult.json\n(verified on disk)", PALETTE[2])
    _box(ax, 24, 18, 18, 18, "Resume:\nre-claim pending\nand retryable", PALETTE[3])
    _box(ax, 48, 18, 18, 18, "Union labels\n(labels.parquet)", PALETTE[1])
    _box(ax, 72, 18, 24, 18, "Off-box sync\n(ledger snapshots,\narchived tree)", PALETTE[4])
    _arrow(ax, 18, 76, 24, 76)
    _arrow(ax, 42, 76, 48, 76)
    _arrow(ax, 66, 76, 72, 76)
    _arrow(ax, 33, 66, 33, 36)
    _arrow(ax, 33, 27, 48, 27)
    _arrow(ax, 84, 66, 84, 36)
    _arrow(ax, 66, 27, 72, 27)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_graph_views(out_dir: Path, name: str = "F3.3_graph_views") -> None:
    """F3.3: AST vs CFG vs CFG with data-flow edges, one small contract."""
    plt, fig, ax = _schematic_axes((12, 5.5))
    ax.text(2, 96, "function withdraw() { uint a = bal[msg.sender]; "
                   "msg.sender.call.value(a)(); bal[msg.sender] = 0; }",
            fontsize=8.5, family="monospace")

    def node(x, y, label, colour):
        ax.scatter([x], [y], s=760, color=colour + "33", edgecolor=colour, zorder=3)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.4, zorder=4)

    # AST cluster
    ax.text(14, 84, "AST view", fontsize=10)
    node(14, 70, "func", PALETTE[0])
    for i, (lab, x) in enumerate((("decl a", 5), ("call", 14), ("assign 0", 23))):
        node(x, 48, lab, PALETTE[0])
        _arrow(ax, 14, 65, x, 53)
    node(5, 28, "bal[..]", PALETTE[0]); _arrow(ax, 5, 43, 5, 33)
    node(14, 28, "value(a)", PALETTE[0]); _arrow(ax, 14, 43, 14, 33)

    # CFG cluster
    ax.text(48, 84, "CFG view", fontsize=10)
    node(48, 70, "entry", PALETTE[2])
    node(48, 50, "a = bal[s]", PALETTE[2])
    node(48, 30, "call(a)", PALETTE[2])
    node(48, 10, "bal[s] = 0", PALETTE[2])
    _arrow(ax, 48, 65, 48, 55); _arrow(ax, 48, 45, 48, 35); _arrow(ax, 48, 25, 48, 15)

    # CFG + DF cluster
    ax.text(78, 84, "CFG + data-flow view", fontsize=10)
    node(78, 70, "entry", PALETTE[2])
    node(78, 50, "a = bal[s]", PALETTE[2])
    node(78, 30, "call(a)", PALETTE[2])
    node(78, 10, "bal[s] = 0", PALETTE[2])
    _arrow(ax, 78, 65, 78, 55); _arrow(ax, 78, 45, 78, 35); _arrow(ax, 78, 25, 78, 15)
    _arrow(ax, 84, 48, 84, 33, colour=PALETTE[3], style="--")   # def-use a
    ax.text(90, 41, "data-flow\n(def-use of a)", fontsize=8, color=PALETTE[3])
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_node_features(out_dir: Path, name: str = "F3.4_node_features") -> None:
    """F3.4: composition of the 129-dimensional node feature vector."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(10, 2.6))
    segs = [("Node-type one-hot (61)", 61, PALETTE[0]),
            ("Structural (4)", 4, PALETTE[3]),
            ("CodeBERT snippet embedding, PCA 768 to 64 (64)", 64, PALETTE[2])]
    left = 0
    for lab, wdt, col in segs:
        ax.barh([0], [wdt], left=left, color=col + "55", edgecolor=col, height=0.6)
        ax.text(left + wdt / 2, 0, lab, ha="center", va="center", fontsize=9)
        left += wdt
    ax.set_xlim(0, 129)
    ax.set_yticks([])
    ax.set_xlabel("Feature dimension (in dim = 129)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_splits_firewall(build_report: dict | None, out_dir: Path,
                        name: str = "F3.5_splits_firewall") -> None:
    """F3.5: the split layout with the content-hash firewall and both test sets."""
    plt, fig, ax = _schematic_axes((11, 5))
    c = (build_report or {}).get("counts", {})
    _box(ax, 2, 60, 24, 26, f"Labelled Wild pool\n{c.get('wild_pool', '~43k')} contracts",
         PALETTE[0])
    _box(ax, 36, 72, 20, 16, f"Train\n{c.get('train', '')}", PALETTE[2])
    _box(ax, 36, 50, 20, 16, f"Validation\n{c.get('val', '')}", PALETTE[2])
    _box(ax, 70, 72, 26, 16, f"Test A (frozen)\ntool-labelled, {c.get('test_a', 1500)}",
         PALETTE[3])
    _box(ax, 70, 50, 26, 16, f"Test B (frozen)\nexpert Curated, {c.get('test_b', 142)}",
         PALETTE[3])
    ax.plot([64, 64], [44, 94], color=PALETTE[3], lw=3, linestyle="--")
    ax.text(64, 40, "content-hash firewall\n(comment-stripped, whitespace-collapsed "
                    "SHA-256;\nleak fails the build)", ha="center", fontsize=9,
            color=PALETTE[3])
    _arrow(ax, 26, 80, 36, 80)
    _arrow(ax, 26, 66, 36, 58)
    _box(ax, 2, 12, 30, 16, "SmartBugs Curated\n143 files, 142 contracts", PALETTE[4])
    _arrow(ax, 32, 20, 70, 52)
    ax.text(50, 18, "expert labels + gold lines", fontsize=8.5)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_architecture(out_dir: Path, name: str = "F3.6_architecture") -> None:
    """F3.6: the DualGNN with the swappable encoders, GATv2 and hybrid included."""
    plt, fig, ax = _schematic_axes((11, 5.5))
    _box(ax, 2, 68, 16, 18, "AST graph\n(129-d nodes)", PALETTE[0])
    _box(ax, 2, 22, 16, 18, "CFG(+DF) graph\n(129-d nodes)", PALETTE[0])
    _box(ax, 24, 68, 24, 18, "Encoder (3 layers,\nhidden 128, mean pool)", PALETTE[2])
    _box(ax, 24, 22, 24, 18, "Encoder (3 layers,\nhidden 128, mean pool)", PALETTE[2])
    _box(ax, 55, 45, 12, 18, "Concat\n(256)", PALETTE[1])
    _box(ax, 71, 45, 12, 18, "MLP head\n(128)", PALETTE[1])
    _box(ax, 87, 45, 11, 18, "5 sigmoid\nlogits", PALETTE[3])
    _arrow(ax, 18, 77, 24, 77); _arrow(ax, 18, 31, 24, 31)
    _arrow(ax, 48, 77, 57, 63); _arrow(ax, 48, 31, 57, 45)
    _arrow(ax, 67, 54, 71, 54); _arrow(ax, 83, 54, 87, 54)
    ax.text(36, 8, "Swappable encoder: GCN | GraphSAGE | GATv2 (4 heads) | "
                   "hybrid (2 x SAGE then GATv2, two-stage, after Lee et al. 2025)",
            fontsize=9)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_ensemble_mechanism(out_dir: Path, name: str = "F3.7_ensemble") -> None:
    """F3.7: the per-class mean ensemble."""
    plt, fig, ax = _schematic_axes((10, 4.6))
    for i, m in enumerate(("GCN", "GraphSAGE", "GATv2")):
        _box(ax, 4, 70 - 26 * i, 18, 18, m, PALETTE[i])
        _arrow(ax, 22, 79 - 26 * i, 34, 56)
    _box(ax, 34, 44, 26, 20, "Per-class mean of\nsigmoid probabilities\n(max policy reported)",
         PALETTE[1])
    _box(ax, 66, 44, 18, 20, "Per-class\nthresholds\n(validation only)", PALETTE[3])
    _box(ax, 88, 44, 10, 20, "5 labels", PALETTE[4])
    _arrow(ax, 60, 54, 66, 54); _arrow(ax, 84, 54, 88, 54)
    ax.text(4, 14, "Membership pre-specified (Workstream E); the hybrid run is "
                   "reported as a single and never joins.", fontsize=9)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_localisation_flow(out_dir: Path, name: str = "F3.8_localisation_flow") -> None:
    """F3.8: from trained model to ranked lines."""
    plt, fig, ax = _schematic_axes((11, 4.6))
    _box(ax, 2, 60, 18, 24, "Trained model\n+ contract graphs", PALETTE[0])
    _box(ax, 25, 60, 20, 24, "GNNExplainer\nper annotated flaw\n(seeded, repeated)", PALETTE[1])
    _box(ax, 50, 60, 18, 24, "Node importance\nscores (AST, CFG)", PALETTE[2])
    _box(ax, 73, 60, 25, 24, "Node-to-line map,\nbranch merge (max),\nranked lines", PALETTE[2])
    _box(ax, 25, 12, 20, 22, "Expert gold lines\n(Test B)", PALETTE[3])
    _box(ax, 55, 12, 34, 22, "accuracy at k in {1, 3, 5, 10},\ntolerance 0 / 1 / 2,\nmean and std over seeds", PALETTE[4])
    _arrow(ax, 20, 72, 25, 72); _arrow(ax, 45, 72, 50, 72); _arrow(ax, 68, 72, 73, 72)
    _arrow(ax, 85, 60, 75, 34)
    _arrow(ax, 45, 23, 55, 23)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_eval_framework(out_dir: Path, name: str = "F3.9_evaluation_framework") -> None:
    """F3.9: the whole evaluation framework on one page."""
    plt, fig, ax = _schematic_axes((11, 5))
    _box(ax, 2, 64, 22, 22, "Eight runs\n+ two ensembles", PALETTE[0])
    _box(ax, 30, 76, 30, 14, "Test A (tool-labelled, frozen)", PALETTE[3])
    _box(ax, 30, 56, 30, 14, "Test B (expert, frozen)", PALETTE[3])
    _box(ax, 66, 64, 32, 26, "Per-class P / R / F1 / accuracy,\nmacro and micro F1,\nsubset accuracy,\nbootstrap 95 per cent CIs,\nper-class confusions", PALETTE[2])
    _box(ax, 30, 14, 30, 22, "Durieux-style matrix:\nfour tools, union,\nmodels, false warnings", PALETTE[1])
    _box(ax, 66, 14, 32, 22, "Localisation benchmark:\nevery single model,\ntop-1/3/5/10, tolerances", PALETTE[4])
    _arrow(ax, 24, 78, 30, 82); _arrow(ax, 24, 70, 30, 62)
    _arrow(ax, 60, 82, 66, 80); _arrow(ax, 60, 62, 66, 70)
    _arrow(ax, 45, 56, 45, 36); _arrow(ax, 24, 68, 70, 36)
    _save(fig, out_dir, name)
    plt.close(fig)


# ------------------------------ F4.x results ------------------------------

def fig_class_distribution(class_freq: dict, out_dir: Path,
                           name: str = "F4.1_class_distribution") -> None:
    plt = _mpl()
    vals = [class_freq.get(f, 0) for f in FLAWS]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(range(len(FLAWS)), vals, color=PALETTE[: len(FLAWS)], edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(FLAWS)))
    ax.set_xticklabels([f.replace("_", "\n") for f in FLAWS])
    ax.set_ylabel("Positive contracts (union labels)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_cooccurrence(labels_df, out_dir: Path,
                     name: str = "F4.2_cooccurrence") -> None:
    plt = _mpl()
    import numpy as np
    Y = labels_df[FLAWS].to_numpy(dtype=int)
    C = (Y.T @ Y).astype(float)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(C, cmap="Blues")
    ax.set_xticks(range(len(FLAWS)))
    ax.set_xticklabels([f.replace("_", "\n") for f in FLAWS], fontsize=8)
    ax.set_yticks(range(len(FLAWS)))
    ax.set_yticklabels([f.replace("_", " ") for f in FLAWS], fontsize=8)
    for i in range(len(FLAWS)):
        for j in range(len(FLAWS)):
            ax.text(j, i, f"{int(C[i, j]):,}", ha="center", va="center", fontsize=8,
                    color="white" if C[i, j] > C.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Contracts sharing both labels")
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_split_composition(class_freq: dict | None, testsets_summary: dict | None,
                          out_dir: Path, name: str = "F4.3_split_composition") -> None:
    plt = _mpl()
    import numpy as np
    groups = []
    if class_freq:
        groups.append(("Labelled pool", [class_freq.get(f, 0) for f in FLAWS]))
    if testsets_summary:
        groups.append(("Test A", [testsets_summary.get("test_a", {})
                                  .get("positives_per_class", {}).get(f, 0)
                                  for f in FLAWS]))
        groups.append(("Test B", [testsets_summary.get("test_b", {})
                                  .get("positives_per_class", {}).get(f, 0)
                                  for f in FLAWS]))
    if not groups:
        raise FileNotFoundError("no class frequency or test-set summary inputs")
    x = np.arange(len(FLAWS))
    w = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (lab, vals) in enumerate(groups):
        ax.bar(x + i * w, vals, w, label=lab, color=PALETTE[i], edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x + w * (len(groups) - 1) / 2)
    ax.set_xticklabels([f.replace("_", "\n") for f in FLAWS])
    ax.set_ylabel("Positive contracts (log scale)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_tool_findings(tool_votes: dict, out_dir: Path,
                      name: str = "F4.4_tool_findings") -> None:
    plt = _mpl()
    import numpy as np
    tools = list(tool_votes)
    M = np.zeros((len(tools), len(FLAWS)))
    for i, t in enumerate(tools):
        for j, f in enumerate(FLAWS):
            cell = tool_votes[t].get(f, {})
            M[i, j] = cell.get("positive", 0) if isinstance(cell, dict) else 0
    fig, ax = plt.subplots(figsize=(8, 4.2))
    im = ax.imshow(M, cmap="Oranges", aspect="auto")
    ax.set_xticks(range(len(FLAWS)))
    ax.set_xticklabels([f.replace("_", "\n") for f in FLAWS], fontsize=8)
    ax.set_yticks(range(len(tools)))
    ax.set_yticklabels(tools)
    for i in range(len(tools)):
        for j in range(len(FLAWS)):
            ax.text(j, i, f"{int(M[i, j]):,}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > M.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Contracts flagged positive")
    _save(fig, out_dir, name)
    plt.close(fig)


def _load_history(run_dir: Path):
    """Tolerant loader for history.json: list-of-dicts or dict-of-lists."""
    h = _load_json(run_dir / "history.json")
    if h is None:
        return None
    if isinstance(h, dict):
        rows = None
        for v in h.values():
            if isinstance(v, list):
                rows = [{k: h[k][i] for k in h if isinstance(h[k], list)
                         and i < len(h[k])} for i in range(len(v))]
                break
        h = rows
    if not isinstance(h, list) or not h or not isinstance(h[0], dict):
        return None
    return h


def _hist_key(row: dict, *needles: str) -> str | None:
    for k in row:
        lk = k.lower()
        if all(n in lk for n in needles):
            return k
    return None


def fig_training_curves(runs_dir: Path, results: dict, out_dir: Path,
                        csv_dir: Path, name: str = "F4.5_training_curves") -> list[str]:
    """F4.5: validation macro-F1 per epoch for every run; also emits the CSVs."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    styles = ["-", "--", "-.", ":"]
    written = []
    plotted = 0
    for i, run in enumerate(sorted(m for m in results if "ensemble" not in m)):
        hist = _load_history(runs_dir / run)
        if not hist:
            continue
        fkey = _hist_key(hist[0], "val", "f1") or _hist_key(hist[0], "macro")
        if fkey is None:
            continue
        ys = [row.get(fkey) for row in hist if row.get(fkey) is not None]
        ax.plot(range(1, len(ys) + 1), ys, label=run.replace("_", " "),
                color=PALETTE[i % len(PALETTE)], linestyle=styles[i // len(PALETTE)])
        plotted += 1
        # CSV emission (training-curve CSVs are a required artifact)
        csv_dir.mkdir(parents=True, exist_ok=True)
        cols = sorted(hist[0])
        lines = [",".join(cols)]
        for row in hist:
            lines.append(",".join(str(row.get(c, "")) for c in cols))
        (csv_dir / f"{run}.csv").write_text("\n".join(lines), encoding="utf-8")
        written.append(f"{run}.csv")
    if plotted == 0:
        plt.close(fig)
        raise FileNotFoundError("no run histories found")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro F1")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, name)
    plt.close(fig)
    return written


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


def fig_per_class_heatmap(results: dict, split: str, out_dir: Path, name: str) -> None:
    """F4.7: models x classes F1 heatmap for one split (one chart per figure)."""
    plt = _mpl()
    import numpy as np
    models = sorted(results)
    M = np.array([[results[m].get(split, {}).get("per_flaw", {})
                   .get(f, {}).get("f1", np.nan) for f in FLAWS] for m in models])
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(models) + 1.8))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FLAWS)))
    ax.set_xticklabels([f.replace("_", "\n") for f in FLAWS], fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace("_", " ") for m in models], fontsize=8)
    for i in range(len(models)):
        for j in range(len(FLAWS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < 0.55 else "black")
    fig.colorbar(im, ax=ax, label="F1")
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


def fig_ensemble_deltas(results: dict, out_dir: Path,
                        name: str = "F4.9_ensemble_deltas") -> None:
    """F4.9: ensemble minus best member, per class, on Test B, both arms."""
    plt = _mpl()
    import numpy as np
    arms = []
    for ens, suffix, lab in (("ensemble", "", "without DF"),
                             ("ensemble_df", "_df", "with DF")):
        if ens not in results:
            continue
        members = [m for m in results
                   if "ensemble" not in m
                   and (m.endswith("_df") if suffix else not m.endswith("_df"))]
        if not members:
            continue
        best = max(members, key=lambda m: results[m].get("test_b", {})
                   .get("macro", {}).get("f1", 0))
        d = [results[ens]["test_b"]["per_flaw"][f]["f1"]
             - results[best]["test_b"]["per_flaw"][f]["f1"] for f in FLAWS]
        arms.append((lab, best, d))
    if not arms:
        raise FileNotFoundError("no ensembles in results")
    x = np.arange(len(FLAWS))
    w = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (lab, best, d) in enumerate(arms):
        ax.bar(x + i * w, d, w, label=f"{lab} (vs {best.replace('_', ' ')})",
               color=PALETTE[i], edgecolor="white")
    ax.axhline(0, color="#444444", lw=1)
    ax.set_xticks(x + w * (len(arms) - 1) / 2)
    ax.set_xticklabels([f.replace("_", "\n") for f in FLAWS])
    ax.set_ylabel("Ensemble F1 minus best single F1 (Test B)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_confusions(results: dict, model: str, split: str, out_dir: Path,
                   name: str) -> None:
    """F4.10: the winner's five per-class 2x2 confusion matrices, one split."""
    plt = _mpl()
    import numpy as np
    conf = results[model].get(split, {}).get("confusion")
    if not conf:
        raise FileNotFoundError("no confusion matrices in results")
    fig, axes = plt.subplots(1, len(FLAWS), figsize=(2.3 * len(FLAWS), 2.9))
    for ax, f in zip(np.atleast_1d(axes), FLAWS):
        c = conf[f]
        M = np.array([[c.get("tn", 0), c.get("fp", 0)],
                      [c.get("fn", 0), c.get("tp", 0)]], float)
        ax.imshow(M, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if M[i, j] > M.max() * 0.55 else "black")
        ax.set_title(f.replace("_", "\n"), fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"], fontsize=7)
    fig.suptitle(f"{model.replace('_', ' ')} on {split.replace('_', ' ')}", fontsize=10)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_durieux(matrix: dict, out_dir: Path, name: str = "F4.11_durieux") -> None:
    """F4.11: detectors x classes recall heatmap with the false-warning column."""
    plt = _mpl()
    import numpy as np
    names = list(matrix)
    cols = FLAWS + ["false_warnings"]
    M = np.zeros((len(names), len(cols)))
    for i, n in enumerate(names):
        for j, f in enumerate(FLAWS):
            M[i, j] = matrix[n]["per_flaw"][f]["recall"]
        M[i, -1] = matrix[n].get("false_warning_rate", 0)
    fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(names) + 1.8))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols], fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=8)
    for i in range(len(names)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] < 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Recall (final column: false-warning rate)")
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


def fig_before_after(results: dict, best: str, out_dir: Path,
                     name: str = "F4.13_before_after") -> None:
    """F4.13: v1 predefence against v2 on the expert test set."""
    plt = _mpl()
    import numpy as np
    pairs = [
        ("GCN", V1_RECORD["expert_macro"]["gcn"],
         results.get("gcn_df", results.get("gcn", {})).get("test_b", {})
         .get("macro", {}).get("f1", 0)),
        ("GraphSAGE", V1_RECORD["expert_macro"]["sage"],
         results.get("sage_df", results.get("sage", {})).get("test_b", {})
         .get("macro", {}).get("f1", 0)),
        ("GAT / GATv2", V1_RECORD["expert_macro"]["gat"],
         results.get("gatv2_df", results.get("gatv2", {})).get("test_b", {})
         .get("macro", {}).get("f1", 0)),
        ("Selected / winner", V1_RECORD["expert_macro"][V1_RECORD["selected"]],
         results.get(best, {}).get("test_b", {}).get("macro", {}).get("f1", 0)),
    ]
    x = np.arange(len(pairs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.bar(x - w / 2, [p[1] for p in pairs], w, label="v1 (predefence, 2,488 contracts)",
           color=PALETTE[0], edgecolor="white")
    ax.bar(x + w / 2, [p[2] for p in pairs], w, label="v2 (full-scale matrix)",
           color=PALETTE[1], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs])
    ax.set_ylabel("Macro F1 on the expert test set")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_localisation(bench: dict, out_dir: Path,
                     name: str = "F4.14_localisation") -> None:
    """F4.14: accuracy@k per model (exact match), the localisation benchmark."""
    plt = _mpl()
    import numpy as np
    models = sorted(bench.get("models", {}))
    if not models:
        raise FileNotFoundError("empty localisation benchmark")
    ks = [str(k) for k in bench.get("ks", [1, 3, 5, 10])]
    x = np.arange(len(models))
    w = 0.8 / len(ks)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, k in enumerate(ks):
        vals = [float(bench["models"][m]["results_by_tolerance"]["0"]
                      ["accuracy_at_k_mean"].get(k, 0)) for m in models]
        ax.bar(x + i * w, vals, w, label=f"top-{k}", color=PALETTE[i],
               edgecolor="white")
    ax.set_xticks(x + w * (len(ks) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ") for m in models], rotation=20, ha="right")
    ax.set_ylabel("Exact-line accuracy at k (Test B)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, name)
    plt.close(fig)


# ---------------------------- summary + emit_all ----------------------------

def results_summary_md(results: dict, best: str, encoders: list[str],
                       durieux: dict | None, localisation: dict | None,
                       build_report: dict | None) -> str:
    """RESULTS_SUMMARY.md: one page, British English, no em dashes."""
    b = results[best]
    def m(r, s):
        return r.get(s, {}).get("macro", {}).get("f1")
    def ci(r, s):
        c = r.get(s, {}).get("ci", {}).get("macro_f1", {})
        return f"[{_fmt(c.get('lo'))}, {_fmt(c.get('hi'))}]"

    deltas = []
    for e in encoders:
        wo, wi = results.get(e), results.get(f"{e}_df")
        if wo and wi:
            deltas.append(m(wi, "test_b") - m(wo, "test_b"))
    df_verdict = ("data-flow edges helped on average "
                  f"(+{sum(deltas)/len(deltas):.3f} macro F1 on Test B)"
                  if deltas and sum(deltas) > 0 else
                  ("data-flow edges did not help on average "
                   f"({sum(deltas)/len(deltas):+.3f} macro F1 on Test B)"
                   if deltas else "ablation incomplete"))

    singles = {k: v for k, v in results.items() if "ensemble" not in k}
    best_single = max(singles, key=lambda k: m(singles[k], "test_b")) if singles else None
    ens = results.get("ensemble_df") or results.get("ensemble")
    ens_verdict = "no ensemble results"
    if ens and best_single:
        d = m(ens, "test_b") - m(results[best_single], "test_b")
        ens_verdict = (f"the ensemble {'beats' if d > 0 else 'does not beat'} the "
                       f"best single ({best_single}) by {d:+.3f} macro F1 on Test B")

    v1_sel = V1_RECORD["expert_macro"][V1_RECORD["selected"]]
    lines = [
        "# v2 results summary",
        "",
        f"Winner (macro F1 on the expert test set, Test B): **{best}**.",
        "",
        f"* Test B macro F1: {_fmt(m(b, 'test_b'))} {ci(b, 'test_b')}",
        f"* Test A macro F1: {_fmt(m(b, 'test_a'))} {ci(b, 'test_a')}",
        f"* Validation macro F1: {_fmt(m(b, 'val'))}",
        "",
        f"Ablation verdict: {df_verdict}.",
        "",
        f"Ensemble verdict: {ens_verdict}.",
        "",
        f"Before and after: v1 selected {V1_RECORD['selected']} at {_fmt(v1_sel)} "
        f"expert macro F1 on {V1_RECORD['dataset_contracts']:,} contracts; v2's "
        f"winner reaches {_fmt(m(b, 'test_b'))} on the frozen expert set after "
        f"scaling the corpus (see Table T6 for the full scale).",
    ]
    if build_report:
        enc = build_report.get("encoded", {})
        c = build_report.get("counts", {})
        lines += ["",
                  f"Evaluation coverage: Test A {enc.get('test_a', '?')} of "
                  f"{c.get('test_a', '?')} manifest contracts encoded; Test B "
                  f"{enc.get('test_b', '?')} of {c.get('test_b', '?')}. The "
                  f"remainder could not be represented as graphs and is reported, "
                  f"not silently dropped."]
    if durieux:
        try:
            union = durieux.get("union", {})
            lines += ["",
                      f"Durieux-style comparison: the four-tool union reaches macro "
                      f"F1 {_fmt(union.get('macro', {}).get('f1'))} on Test B with a "
                      f"false-warning rate of "
                      f"{_fmt(union.get('false_warning_rate'))}; the model rows sit "
                      f"in Table T5."]
        except Exception:
            pass
    if localisation and best in localisation.get("models", {}):
        a = localisation["models"][best]["results_by_tolerance"]["0"]["accuracy_at_k_mean"]
        lines += ["",
                  f"Localisation ({best}, exact line): top-1 {_fmt(a.get('1'))}, "
                  f"top-3 {_fmt(a.get('3'))}, top-5 {_fmt(a.get('5'))}, "
                  f"top-10 {_fmt(a.get('10'))} "
                  f"(v1: {_fmt(V1_RECORD['localisation_at'][1])} / "
                  f"{_fmt(V1_RECORD['localisation_at'][3])} / "
                  f"{_fmt(V1_RECORD['localisation_at'][5])} at 1 / 3 / 5)."]
    lines.append("")
    return "\n".join(lines)


def emit_all(results_path: str | Path, out_root: str | Path,
             encoders: list[str] | None = None,
             durieux: dict | None = None,
             *,
             labels_path: str = "data/processed/labels.parquet",
             class_freq_path: str = "data/processed/class_frequency.json",
             tool_votes_path: str = "data/processed/tool_vote_summary.json",
             testsets_summary_path: str = "data/testsets/testsets_summary.json",
             build_report_path: str = "data/processed_df/build_report.json",
             runs_dir: str | Path | None = None,
             localisation_path: str | Path | None = None) -> dict:
    """Emit every table and figure from a results.json. Returns what was written.

    Optional inputs (class frequencies, tool votes, test-set summary, build
    report, run histories, localisation benchmark) enrich T6/T7 and F3.5/F4.x;
    any that are missing skip their artifact with a note in ``skipped``.
    """
    encoders = encoders or ["gcn", "sage", "gatv2"]
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    out = Path(out_root)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    runs = Path(runs_dir) if runs_dir else Path(results_path).parent
    loc_path = Path(localisation_path) if localisation_path \
        else out / "localisation_benchmark.json"

    class_freq = _load_json(class_freq_path)
    tool_votes = _load_json(tool_votes_path)
    testsets_summary = _load_json(testsets_summary_path)
    build_report = _load_json(build_report_path)
    localisation = _load_json(loc_path)

    best = max(results, key=lambda m: results[m].get("test_b", {})
               .get("macro", {}).get("f1", 0))
    written = {"tables": [], "figures": [], "csv": [], "skipped": [],
               "best_model": best}

    # ---- tables ----
    tables = {
        "T1_dual_benchmark.tex": lambda: table_dual_benchmark(results, best),
        "T2_model_matrix.tex": lambda: table_model_matrix(results),
        "T3_ablation.tex": lambda: table_ablation(results, encoders),
        "T4_ensemble.tex": lambda: table_ensemble(results),
        "T6_dataset_scale.tex": lambda: table_dataset_scale(
            class_freq, testsets_summary, build_report, tool_votes),
        "T7_before_after.tex": lambda: table_before_after(
            results, best, class_freq, localisation),
    }
    if durieux:
        tables["T5_durieux.tex"] = lambda: table_durieux(durieux)
    else:
        written["skipped"].append("T5 (no Durieux matrix yet)")
    for fn, make in tables.items():
        try:
            (out / "tables" / fn).write_text(make(), encoding="utf-8")
            written["tables"].append(fn)
        except Exception as e:
            written["skipped"].append(f"{fn} ({e})")

    # ---- F3.x methodology schematics (no data inputs needed) ----
    schematics = [
        ("F3.1_pipeline", lambda: fig_pipeline(figs)),
        ("F3.2_labelling_pipeline", lambda: fig_labelling_pipeline(figs)),
        ("F3.3_graph_views", lambda: fig_graph_views(figs)),
        ("F3.4_node_features", lambda: fig_node_features(figs)),
        ("F3.5_splits_firewall", lambda: fig_splits_firewall(build_report, figs)),
        ("F3.6_architecture", lambda: fig_architecture(figs)),
        ("F3.7_ensemble", lambda: fig_ensemble_mechanism(figs)),
        ("F3.8_localisation_flow", lambda: fig_localisation_flow(figs)),
        ("F3.9_evaluation_framework", lambda: fig_eval_framework(figs)),
    ]

    # ---- F4.x results figures ----
    def _labels_df():
        import pandas as pd
        return pd.read_parquet(labels_path)

    figures = schematics + [
        ("F4.1_class_distribution",
         lambda: fig_class_distribution(class_freq or {}, figs)
         if class_freq else (_ for _ in ()).throw(FileNotFoundError(class_freq_path))),
        ("F4.2_cooccurrence", lambda: fig_cooccurrence(_labels_df(), figs)),
        ("F4.3_split_composition",
         lambda: fig_split_composition(class_freq, testsets_summary, figs)),
        ("F4.4_tool_findings",
         lambda: fig_tool_findings(tool_votes or {}, figs)
         if tool_votes else (_ for _ in ()).throw(FileNotFoundError(tool_votes_path))),
        ("F4.5_training_curves",
         lambda: written["csv"].extend(
             fig_training_curves(runs, results, figs, out / "training_curves"))),
        ("F4.6_model_comparison", lambda: fig_model_comparison(results, figs)),
        ("F4.7a_per_class_f1_test_a",
         lambda: fig_per_class_heatmap(results, "test_a", figs,
                                       "F4.7a_per_class_f1_test_a")),
        ("F4.7b_per_class_f1_test_b",
         lambda: fig_per_class_heatmap(results, "test_b", figs,
                                       "F4.7b_per_class_f1_test_b")),
        ("F4.8_dataflow_ablation", lambda: fig_ablation(results, encoders, figs)),
        ("F4.9_ensemble_deltas", lambda: fig_ensemble_deltas(results, figs)),
        ("F4.10a_confusions_test_b",
         lambda: fig_confusions(results, best, "test_b", figs,
                                "F4.10a_confusions_test_b")),
        ("F4.10b_confusions_test_a",
         lambda: fig_confusions(results, best, "test_a", figs,
                                "F4.10b_confusions_test_a")),
        ("F4.11_durieux",
         lambda: fig_durieux(durieux, figs)
         if durieux else (_ for _ in ()).throw(FileNotFoundError("durieux matrix"))),
        ("F4.12_ab_gap", lambda: fig_ab_gap(results, figs)),
        ("F4.13_before_after", lambda: fig_before_after(results, best, figs)),
        ("F4.14_localisation",
         lambda: fig_localisation(localisation or {}, figs)
         if localisation else (_ for _ in ()).throw(FileNotFoundError(str(loc_path)))),
    ]
    for name, make in figures:
        try:
            make()
            written["figures"].append(name)
        except Exception as e:
            written["skipped"].append(f"{name} ({e})")

    # ---- one-page summary ----
    try:
        (out / "RESULTS_SUMMARY.md").write_text(
            results_summary_md(results, best, encoders, durieux, localisation,
                               build_report), encoding="utf-8")
        written["summary"] = "RESULTS_SUMMARY.md"
    except Exception as e:
        written["skipped"].append(f"RESULTS_SUMMARY.md ({e})")

    if written["skipped"]:
        print("emit_all skipped:", "; ".join(written["skipped"]))
    return written