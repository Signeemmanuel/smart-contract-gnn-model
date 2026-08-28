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

Alignment: the trained models can only predict for contracts the build could
represent as graphs (141 of the 142 in the frozen manifest). The matrix is
therefore evaluated on the ENCODED subset for every row, tools included, so all
detectors share one denominator; the excluded contract is counted and printed,
never silently dropped.

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
    """Run the four tools over the Test B contracts, resumably.

    Uses the corpus orchestrator's machinery (ledger, results template, on-disk
    output verification), so the same guarantees hold here: exit 0 without a
    result.json is recorded as ``no_output`` and retried, never trusted. Worker
    exceptions are surfaced, not swallowed: a run that does nothing dies loudly.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from label_orchestrator import Ledger, run_task
    import concurrent.futures as cf
    import time

    from training.data.testsets import read_manifest

    rows = read_manifest(Path(args.testsets) / "test_b.csv")
    print(f"Test B: {len(rows)} contracts x {len(TOOLS)} tools")

    led = Ledger(args.ledger)
    work = [(r["contract_id"], r["path"], r["chash"]) for r in rows]
    nc, nt = led.enrol(work)
    print(f"enrolled {nc} contracts, {nt} tasks")

    tasks = led.claim_pending(args.max_attempts)
    if not tasks:
        print("all tasks complete.")
        return 0
    print(f"{len(tasks)} tasks to run (workers={args.workers}, "
          f"timeout={args.timeout:g}s)")

    Path(args.results).mkdir(parents=True, exist_ok=True)
    sb_cmd = args.sb_cmd.split()
    sb_timeout = max(30, int(args.timeout) - 30)
    t0 = time.time()

    def worker(t):
        cid, tool, path = t
        status, secs = run_task(sb_cmd, tool, path, args.results,
                                args.timeout, sb_timeout)
        led.update(cid, tool, status, secs)
        return status

    done = 0
    counts: dict[str, int] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in cf.as_completed([ex.submit(worker, t) for t in tasks]):
            status = fut.result()          # surface worker exceptions loudly
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}  {counts}", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min")
    s = led.summary()
    print("per-tool status:", {t: r for t, r in s["per_tool"].items()})
    ok = sum(r["statuses"].get("ok", 0) for r in s["per_tool"].values())
    if ok == 0:
        sys.exit("ERROR: zero successful tool runs; the results tree is empty. "
                 "Check SmartBugs (SB_CMD/PYTHONPATH) before building the matrix.")
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

    # --- align everything to the ENCODED contracts (shared denominator) ---
    # Model probabilities exist only for contracts the build could represent as
    # graphs, in the order of the processed test_b index. Restrict the manifest
    # (and therefore the tools) to that subset so every row of the matrix is
    # scored on the same contracts; count and print the exclusions.
    index_path = Path(args.model_index)
    if index_path.exists():
        encoded_ids = [e["id"] for e in
                       json.loads(index_path.read_text(encoding="utf-8"))]
        by_id = {r["contract_id"]: r for r in rows}
        missing = [i for i in encoded_ids if i not in by_id]
        if missing:
            sys.exit(f"ERROR: {len(missing)} encoded ids absent from the manifest "
                     f"(e.g. {missing[0]}); index and manifest disagree.")
        excluded = [r["contract_id"] for r in rows if r["contract_id"] not in
                    set(encoded_ids)]
        rows = [by_id[i] for i in encoded_ids]   # manifest data, index ORDER
        print(f"evaluating on {len(rows)} encoded contracts "
              f"({len(excluded)} manifest contract(s) not representable as "
              f"graphs, excluded from every row: {', '.join(excluded) or 'none'})")
    else:
        print(f"note: {index_path} not found; using the full manifest order "
              f"(model rows must then match it exactly)")

    cids = [r["contract_id"] for r in rows]
    y_true = np.array([[int(r[f]) for f in FLAWS] for r in rows], dtype=int)
    print(f"Test B ground truth: {len(cids)} contracts, "
          f"positives per class: {dict(zip(FLAWS, y_true.sum(axis=0).tolist()))}")

    tools = tool_predictions(args.results, cids)
    for t in TOOLS:
        ran = int(tools[t].any(axis=1).sum())
        print(f"  {t:<10} flagged {ran} contracts")
    if all(int(tools[t].sum()) == 0 for t in TOOLS):
        sys.exit("ERROR: every tool flagged nothing; the results tree at "
                 f"{args.results} is empty or mislaid. Run the 'run' subcommand "
                 "first and check its per-tool status.")

    # model predictions: {model: {"probs": [[...]], "thresholds": [...]}} in the
    # processed test_b index order (which is exactly ``cids`` after alignment).
    models: dict[str, np.ndarray] = {}
    if args.model_probs:
        mp = json.loads(Path(args.model_probs).read_text(encoding="utf-8"))
        for name, blob in mp.items():
            probs = np.asarray(blob["probs"], dtype=float)
            thr = blob.get("thresholds", [0.5] * N_FLAWS)
            if probs.shape[0] != len(cids):
                sys.exit(f"ERROR: {name} has {probs.shape[0]} rows but the "
                         f"evaluation set has {len(cids)} contracts; pass the "
                         f"matching --model-index for the build these "
                         f"probabilities came from.")
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
                        "processed test_b index order.")
    m.add_argument("--model-index", default="data/processed_df/test_b_index.json",
                   help="The processed test_b index the model probabilities follow; "
                        "the matrix is evaluated on exactly these contracts.")
    m.add_argument("--out", default="artifacts/v2")
    m.set_defaults(func=cmd_matrix)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())