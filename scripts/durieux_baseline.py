#!/usr/bin/env python3
"""The Durieux-style tool-vs-model baseline on Test Set B (Workstream G, T5/F4.11).

Runs the four analysis tools directly on the EXPERT test set, maps their findings
to our five classes with the existing map_dasp mapping, and puts them in one
matrix beside the trained models, all scored against the same expert ground
truth. This is the direct analogue of Durieux et al. (2020), whose headline
figures sit beside ours in the dissertation:

    best single tool detected 27 per cent of the vulnerabilities;
    all tools combined detected 42 per cent;
    97 per cent of contracts were flagged by at least one tool.

Our matrix reports, per class: recall, precision, F1 and accuracy for each tool
alone, for the UNION of the four tools (which is exactly our labelling rule, so
the table shows what the labelling oracle itself achieves against expert truth),
and for each trained model. The final column is the false-warning analogue: the
fraction of flaw-FREE Test B contracts that each row flags.

Why this matters for the thesis: the union rule generates our TRAINING labels. If
the union scores poorly against expert ground truth, that is the ceiling our
models are learning towards, and it explains the Test A / Test B gap far better
than any model-side argument.

Pipeline position:
    freeze_testsets.py -> [THIS, needs only Test B + SmartBugs] -> artifacts

Usage
-----
    # 1. run the four tools over the Test B contracts (resumable, reuses the
    #    orchestrator's ledger machinery)
    PYTHONPATH=. python scripts/durieux_baseline.py run \
        --testsets data/testsets --results data/sb_testb --workers 16

    # 2. build the matrix (tools + models) once model probabilities exist
    PYTHONPATH=. python scripts/durieux_baseline.py matrix \
        --testsets data/testsets --results data/sb_testb \
        --model-probs runs/v2/test_b_probs.json \
        --out artifacts/v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scgnn.schema import FLAWS, N_FLAWS

TOOLS = ["slither", "mythril", "securify", "osiris"]


# ------------------------------------------------------------------ run tools

def cmd_run(args: argparse.Namespace) -> int:
    """Run the four tools over the Test B contracts, resumably."""
    sys.path.insert(0, str(Path(__file__).parent))
    from label_orchestrator import Ledger, TOOLS as ORCH_TOOLS, hash_file, run_task
    import concurrent.futures as cf
    import time

    from training.data.testsets import read_manifest

    rows = read_manifest(Path(args.testsets) / "test_b.csv")
    print(f"Test B: {len(rows)} contracts x {len(ORCH_TOOLS)} tools")

    led = Ledger(args.ledger)
    work = [(r["contract_id"], r["path"], r["chash"]) for r in rows]
    nc, nt = led.enrol(work)
    print(f"enrolled {nc} contracts, {nt} tasks")

    tasks = led.claim_pending(args.max_attempts)
    if not tasks:
        print("all tasks complete.")
        return 0
    print(f"{len(tasks)} tasks to run (workers={args.workers})")

    Path(args.results).mkdir(parents=True, exist_ok=True)
    sb_cmd = args.sb_cmd.split()
    t0 = time.time()

    def worker(t):
        cid, tool, path = t
        status, secs = run_task(sb_cmd, tool, path, args.results, args.timeout)
        led.update(cid, tool, status, secs)
        return status

    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in cf.as_completed([ex.submit(worker, t) for t in tasks]):
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min")
    s = led.summary()
    print("per-tool status:", {t: r for t, r in s["per_tool"].items()})
    return 0


# --------------------------------------------------------------- build matrix

def tool_predictions(results_dir: str | Path,
                     contract_ids: list[str]) -> dict[str, np.ndarray]:
    """Per-tool (n, 5) binary predictions on the Test B contracts.

    Reads the SmartBugs results tree with the project's own ``collect_votes`` and
    maps findings via ``map_dasp``, so the tools are scored through EXACTLY the
    same mapping used to build the training labels. A tool that did not run, or
    that crashed, on a contract predicts 0 for every class there (it raised no
    warning), which is the honest reading for a detection baseline.
    """
    from training.labelling.run_tools import collect_votes
    from scgnn.schema import FLAW_INDEX

    votes = collect_votes(results_dir)          # {contract: {tool: set(flaws)}}
    preds = {t: np.zeros((len(contract_ids), N_FLAWS), dtype=int) for t in TOOLS}
    for i, cid in enumerate(contract_ids):
        per_tool = votes.get(cid, {})
        for tool in TOOLS:
            for flaw in per_tool.get(tool, set()) or set():
                preds[tool][i, FLAW_INDEX[flaw]] = 1
    return preds


def cmd_matrix(args: argparse.Namespace) -> int:
    from training.data.testsets import read_manifest
    from training.evaluate.metrics import apply_thresholds, tool_baseline_matrix

    rows = read_manifest(Path(args.testsets) / "test_b.csv")
    cids = [r["contract_id"] for r in rows]
    y_true = np.array([[int(r[f]) for f in FLAWS] for r in rows], dtype=int)
    print(f"Test B ground truth: {len(cids)} contracts, "
          f"positives per class: {dict(zip(FLAWS, y_true.sum(axis=0).tolist()))}")

    tools = tool_predictions(args.results, cids)
    for t in TOOLS:
        ran = int(tools[t].any(axis=1).sum())
        print(f"  {t:<10} flagged {ran} contracts")

    # model predictions: {model: {"probs": [[...]], "thresholds": [...]}} keyed by
    # the SAME contract order as the manifest.
    models: dict[str, np.ndarray] = {}
    if args.model_probs:
        mp = json.loads(Path(args.model_probs).read_text(encoding="utf-8"))
        for name, blob in mp.items():
            probs = np.asarray(blob["probs"], dtype=float)
            thr = blob.get("thresholds", [0.5] * N_FLAWS)
            if probs.shape[0] != len(cids):
                sys.exit(f"ERROR: {name} has {probs.shape[0]} rows but Test B has "
                         f"{len(cids)} contracts; the order must match the manifest.")
            models[name] = apply_thresholds(probs, thr)
            print(f"  model {name:<12} flagged "
                  f"{int(models[name].any(axis=1).sum())} contracts")

    matrix = tool_baseline_matrix(y_true, tools, models)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "durieux_matrix.json").write_text(json.dumps(matrix, indent=2),
                                             encoding="utf-8")

    # LaTeX table T5
    from training.evaluate.artifacts import table_durieux
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tables" / "T5_durieux.tex").write_text(table_durieux(matrix),
                                                   encoding="utf-8")

    # Figure F4.11
    _fig_durieux(matrix, out / "figures")

    # --- the comparison the dissertation actually makes ---
    print("\n" + "=" * 74)
    print(f"{'detector':<16} {'macro F1':>9} {'macro recall':>13} {'false warnings':>15}")
    print("-" * 74)
    for name in matrix:
        r = matrix[name]
        rec = float(np.mean([r["per_flaw"][f]["recall"] for f in FLAWS]))
        print(f"{name:<16} {r['macro']['f1']:>9.3f} {rec:>13.3f} "
              f"{r['false_warning_rate']:>15.3f}")
    print("=" * 74)
    print("Durieux et al. (2020) reported: best single tool 27 per cent detection, "
          "all tools combined 42 per cent, 97 per cent of contracts flagged.")
    print(f"\nwrote {out/'durieux_matrix.json'}, {out/'tables'/'T5_durieux.tex'}")
    return 0


def _fig_durieux(matrix: dict, out_dir: Path) -> None:
    """F4.11: per-class recall of tools, their union, and the models; precision
    as a second panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from training.evaluate.artifacts import PALETTE, _save

    names = list(matrix)
    x = np.arange(len(FLAWS))
    w = 0.8 / max(1, len(names))

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for metric, ax in zip(("recall", "precision"), axes):
        for i, name in enumerate(names):
            vals = [matrix[name]["per_flaw"][f][metric] for f in FLAWS]
            ax.bar(x + i * w, vals, w, label=name.replace("_", " "),
                   color=PALETTE[i % len(PALETTE)], edgecolor="white")
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].set_xticks(x + w * (len(names) - 1) / 2)
    axes[1].set_xticklabels([f.replace("_", " ") for f in FLAWS])
    _save(fig, out_dir, "F4.11_durieux_comparison")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run the four tools over Test B.")
    r.add_argument("--testsets", default="data/testsets")
    r.add_argument("--results", default="data/sb_testb")
    r.add_argument("--ledger", default="data/testb_ledger.sqlite")
    r.add_argument("--sb-cmd", default="sb")
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--timeout", type=float, default=300,
                   help="Per-task timeout. Higher than the corpus run: Test B is "
                        "small, and a tool timing out here would understate it.")
    r.add_argument("--max-attempts", type=int, default=2)
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("matrix", help="Build the tool-vs-model matrix + T5 + F4.11.")
    m.add_argument("--testsets", default="data/testsets")
    m.add_argument("--results", default="data/sb_testb")
    m.add_argument("--model-probs", default=None,
                   help="JSON: {model: {probs: [[...]], thresholds: [...]}} in the "
                        "manifest's contract order.")
    m.add_argument("--out", default="artifacts/v2")
    m.set_defaults(func=cmd_matrix)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())