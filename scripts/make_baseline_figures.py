#!/usr/bin/env python3
"""Generate the learned-baseline benchmarking figures from results.json.

Every number is READ from the result files produced by scripts/train_baselines.py
and scripts/train_v2.py. Nothing is hardcoded: re-run a baseline and the figures
regenerate with the new values. Missing inputs are skipped with a note rather
than crashing (same tolerance as the baseline suite).

Figures:
  F4.15_baseline_ladder     macro-F1 of every baseline + the model on Test Set B,
                            with the four-tool union oracle as a reference line.
  F4.16_structure_vs_text   sequence (flat text) vs the model across val/A/B.
  F4.17_truncation_rate     fraction of contracts exceeding 512 tokens (why flat
                            text is handicapped), from the sequence run_info.
  F4.18_votes_artifact      votes macro-F1 on tool-labelled splits vs expert
                            truth (the union rule is recoverable only on tools).

Usage
-----
    PYTHONPATH=. python scripts/make_baseline_figures.py \
        --baselines runs/baselines/results.json \
        --baselines-testb runs/baselines_testb/results.json \
        --sequence runs/baselines_seq/results.json \
        --model runs/v2/results.json \
        --model-name sage_df \
        --out artifacts/v2/figures

The four-tool union Test B macro-F1 (the reference line) is read from
artifacts/v2/durieux_matrix.json by default, or pass --union-oracle to override.
Omit both to draw the figures without the reference line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
})
INK, MUTED, ACCENT, TOOL, TEXT, FLOOR = (
    "#1a1a2e", "#8d99ae", "#2a9d8f", "#e76f51", "#457b9d", "#adb5bd")


def load(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  note: {p} not found; figures needing it are skipped")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def macro_f1(results, model, split):
    try:
        return float(results[model][split]["macro"]["f1"])
    except (KeyError, TypeError):
        return None


def union_oracle_value(args):
    if args.union_oracle is not None:
        return args.union_oracle
    dur = load(args.durieux)
    if dur and "union_of_tools" in dur:
        try:
            return float(dur["union_of_tools"]["macro"]["f1"])
        except (KeyError, TypeError):
            pass
    return None


def save(fig, name, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def fig_ladder(base, testb, seq, model, model_name, oracle, out):
    entries = []

    def add(results, key, label, colour):
        if results is None:
            return
        v = macro_f1(results, key, "test_b")
        if v is not None:
            entries.append((label, v, colour))

    add(base, "trivial_majority", "Majority", FLOOR)
    add(base, "trivial_all_positive", "All-positive", FLOOR)
    add(seq, "sequence_codebert", "Sequence\n(CodeBERT)", TEXT)
    votes_src = testb if (testb and macro_f1(testb, "votes_logreg", "test_b")) else base
    add(votes_src, "votes_logreg", "Votes\n(tool-mimicry)", TOOL)
    if model is not None:
        v = macro_f1(model, model_name, "test_b")
        if v is not None:
            entries.append((f"{model_name}\n(our GNN)", v, ACCENT))

    if not entries:
        print("  ladder skipped: no Test B values found")
        return
    entries.sort(key=lambda e: e[1])
    labels, vals, colours = zip(*entries)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, vals, color=colours, edgecolor="white", width=0.66,
                  zorder=3)
    if oracle is not None:
        ax.axhline(oracle, color=INK, ls="--", lw=1.2, zorder=2)
        ax.text(len(labels) - 0.5, oracle,
                f"  four-tool union\n  oracle {oracle:.3f}",
                va="center", ha="left", fontsize=8.5, color=INK)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylabel("Macro-F1 (expert Test Set B)")
    top = max(list(vals) + ([oracle] if oracle else []))
    ax.set_ylim(0, top * 1.2 + 0.03)
    ax.set_xlim(-0.7, len(labels) + 0.4)
    ax.set_title("Baseline ladder on the expert test set", fontsize=12,
                 color=INK, pad=10)
    save(fig, "F4.15_baseline_ladder", out)


def fig_structure_vs_text(seq, model, model_name, out):
    if seq is None or model is None:
        print("  structure-vs-text skipped: needs both --sequence and --model")
        return
    splits = [("val", "Validation"), ("test_a", "Test A\n(tool-labelled)"),
              ("test_b", "Test B\n(expert)")]
    seq_v = [macro_f1(seq, "sequence_codebert", s) for s, _ in splits]
    gnn_v = [macro_f1(model, model_name, s) for s, _ in splits]
    if any(v is None for v in seq_v + gnn_v):
        print("  structure-vs-text skipped: missing a split value")
        return
    labels = [lbl for _, lbl in splits]
    x = range(len(splits))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    b1 = ax.bar([i - w / 2 for i in x], seq_v, w,
                label="Sequence (CodeBERT, flat text)", color=TEXT,
                edgecolor="white", zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], gnn_v, w,
                label=f"{model_name} (dual-graph GNN)", color=ACCENT,
                edgecolor="white", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    f"{b.get_height():.3f}", ha="center", va="bottom",
                    fontsize=9, color=INK)
    for i, (s, g) in enumerate(zip(seq_v, gnn_v)):
        ax.annotate(f"+{g - s:.3f}", xy=(i, max(s, g) + 0.05), ha="center",
                    fontsize=8.5, color=ACCENT, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    ax.set_title("Structure versus flat text under a matched protocol",
                 fontsize=12, color=INK, pad=10)
    save(fig, "F4.16_structure_vs_text", out)


def fig_truncation(seq, out):
    if seq is None:
        print("  truncation skipped: no sequence results")
        return
    info = seq.get("sequence_codebert", {}).get("run_info")
    if not info or "train_truncation_rate" not in info:
        print("  truncation skipped: no run_info in sequence results")
        return
    rates = [("Training set", info.get("train_truncation_rate")),
             ("Validation set", info.get("val_truncation_rate"))]
    rates = [(k, v) for k, v in rates if v is not None]
    labels, vals = zip(*rates)
    vals = [v * 100 for v in vals]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    bars = ax.bar(labels, vals, color=TEXT, edgecolor="white", width=0.5,
                  zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=11, color=INK)
    ax.set_ylabel("Contracts exceeding 512 tokens (%)")
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_title("Token-length pressure on the flat-text baseline\n"
                 f"(mode: {info.get('mode', 'sliding')} window)",
                 fontsize=11.5, color=INK, pad=10)
    save(fig, "F4.17_truncation_rate", out)


def fig_votes_artifact(base, testb, oracle, out):
    votes_src = testb if (testb and macro_f1(testb, "votes_logreg", "test_b")) else base
    if votes_src is None:
        print("  votes-artifact skipped: no votes results")
        return
    splits = [("val", "Validation\n(tool labels)"),
              ("test_a", "Test A\n(tool labels)"),
              ("test_b", "Test B\n(expert truth)")]
    vals = [macro_f1(votes_src, "votes_logreg", s) for s, _ in splits]
    if any(v is None for v in vals):
        print("  votes-artifact skipped: missing a split value")
        return
    labels = [lbl for _, lbl in splits]
    colours = [MUTED, MUTED, TOOL]

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    bars = ax.bar(labels, vals, color=colours, edgecolor="white", width=0.6,
                  zorder=3)
    if oracle is not None:
        ax.axhline(oracle, color=INK, ls="--", lw=1.2, zorder=2)
        ax.text(len(labels) - 0.55, oracle, f"  union oracle\n  {oracle:.3f}",
                va="center", ha="left", fontsize=8.5, color=INK)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylabel("Macro-F1 (votes-only logistic regression)")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.6, len(labels) - 0.2)
    ax.set_title("Votes baseline: the union rule is recoverable on tool labels,\n"
                 "not on expert truth", fontsize=11.5, color=INK, pad=10)
    save(fig, "F4.18_votes_artifact", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", default="runs/baselines/results.json")
    ap.add_argument("--baselines-testb", default="runs/baselines_testb/results.json")
    ap.add_argument("--sequence", default="runs/baselines_seq/results.json")
    ap.add_argument("--model", default="runs/v2/results.json")
    ap.add_argument("--model-name", default="sage_df")
    ap.add_argument("--durieux", default="artifacts/v2/durieux_matrix.json")
    ap.add_argument("--union-oracle", type=float, default=None)
    ap.add_argument("--out", default="artifacts/v2/figures")
    args = ap.parse_args()

    base = load(args.baselines)
    testb = load(args.baselines_testb)
    seq = load(args.sequence)
    model = load(args.model)
    oracle = union_oracle_value(args)
    if oracle is not None:
        print(f"  union oracle reference: {oracle:.3f}")

    print("generating baseline figures ...")
    fig_ladder(base, testb, seq, model, args.model_name, oracle, args.out)
    fig_structure_vs_text(seq, model, args.model_name, args.out)
    fig_truncation(seq, args.out)
    fig_votes_artifact(base, testb, oracle, args.out)
    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())