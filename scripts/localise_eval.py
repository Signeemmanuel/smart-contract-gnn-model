#!/usr/bin/env python3
"""Top-k line-localisation accuracy over the frozen expert test set.

For every test contract that carries expert gold lines, this runs the real
explainer for the flaw(s) the contract is annotated with, then scores the
ranked predicted lines against the gold lines with the proposal's metric:

    accuracy@k = fraction of flawed contracts whose top-k predicted lines
                 include at least one expert-marked line.

k in {1, 3, 5, 10}. GNNExplainer is stochastic (random mask initialisation),
so every run is seeded (pass r uses seed+r) and `--repeats N` reports
`mean +/- std` — the figure to cite. `--merge` selects how the two branches'
line scores are combined:

    max     line takes its best score across branches, ranked globally (default)
    concat  every AST line before any CFG line (the pre-fix behaviour)
    both    run a paired ablation: one explainer run per contract feeds BOTH
            rules, so the arms differ only in the combination, never in the
            importances — the honest before/after.

Reuses the build caches (`records/<hash>.pt`, `raw/<hash>.json`); nothing is
re-extracted or re-embedded. The v2 test index (``test_b_index.json``) is used
when present, falling back to v1's ``test_index.json``.

Benchmark mode (v2)
-------------------
``--all-models --runs-dir runs/v2`` localises EVERY trained single model, not
just the winner: each run directory containing a ``best_model.pt`` is loaded
from its own ``provenance.json`` (resolved config + the processed data dir it
trained on), evaluated with the same seeded passes, and scored at every
tolerance in {0, 1, 2} FROM THE SAME PASSES (tolerance affects scoring only,
so re-running the explainer per tolerance would triple the cost for nothing).
Ensemble directories carry no checkpoint and are skipped by construction: an
ensemble has no single computational graph for GNNExplainer to perturb, so it
is excluded from localisation and the write-up says so. Emits one report per
model beside its checkpoint plus a combined ``localisation_benchmark.json``.

Examples:
    # v2 benchmark: every single model, top-1/3/5/10, tolerances 0/1/2
    PYTHONPATH=. python scripts/localise_eval.py --all-models \
        --runs-dir runs/v2 --repeats 5 --device cuda \
        --benchmark-out artifacts/v2/localisation_benchmark.json

    # single reproducible run, merge-max (v1-compatible)
    PYTHONPATH=. python scripts/localise_eval.py \
        --checkpoint runs/sage/best_model.pt --config configs/sage.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

KS = (1, 3, 5, 10)
TOLS = (0, 1, 2)


def _dedup_keep_order(lines: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch so a pass is reproducible."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _test_index_path(processed: Path) -> Path:
    """v2 builds name the expert split ``test_b``; v1 named it ``test``."""
    p = processed / "test_b_index.json"
    return p if p.exists() else processed / "test_index.json"


def _run_pass(model, targets, gold_map, processed: Path, device: str,
              k: int, epochs: int, modes: list[str], verbose: bool, tol: int = 0,
              method: str = "gnnexplainer"):
    """One pass over all targets. Returns ``{mode: rows}``.

    The explainer runs ONCE per contract per flaw; every requested combination
    rule is applied to that single run's scores, so the arms are perfectly
    paired and `both` costs the same as one arm.
    """
    import torch
    from torch_geometric.data import Data

    from scgnn.explain.explainer import explain_line_scores
    from scgnn.explain.attention import attention_lines
    from scgnn.explain.localise import concat_line_scores, merge_line_scores
    from scgnn.extraction.graph_types import RawGraph
    from scgnn.schema import FLAWS, display_name

    combine = {"max": merge_line_scores, "concat": concat_line_scores}
    out: dict[str, list[dict]] = {m: [] for m in modes}
    primary = "max" if "max" in modes else modes[-1]
    total = len(targets)

    for n, entry in enumerate(targets, 1):
        cid = entry["id"]
        gold = sorted(set(gold_map[cid]))
        h = Path(entry["path"]).stem
        rec_path = processed / "records" / f"{h}.pt"
        rawj = processed / "raw" / f"{h}.json"
        if not rec_path.exists() or not rawj.exists():
            if verbose:
                print(f"  [{n}/{total}] {cid}: SKIP (missing record/raw cache for {h})")
            continue
        rec = torch.load(rec_path, weights_only=True)
        d = json.loads(rawj.read_text(encoding="utf-8"))
        ast_raw = RawGraph.from_dict(d["ast"])
        cfg_raw = RawGraph.from_dict(d["cfg"])
        ast_data = Data(x=rec["ast_x"], edge_index=rec["ast_edge_index"]).to(device)
        cfg_data = Data(x=rec["cfg_x"], edge_index=rec["cfg_edge_index"]).to(device)

        positive = [j for j, v in enumerate(rec["y"].tolist()) if v >= 0.5]
        if not positive:
            continue

        merged: dict[str, list[int]] = {m: [] for m in modes}
        if method == "attention":
            # One forward pass populates each attention-bearing encoder's
            # last_attention; the proposal's secondary signal. Branch lines are
            # merged like the explainer's, so the same merge modes and metric
            # apply.
            with torch.no_grad():
                model(ast_data, cfg_data)
            ast_lines, _ua = attention_lines(model.ast, ast_raw.node_lines, k=k)
            cfg_lines, _uc = attention_lines(model.cfg, cfg_raw.node_lines, k=k)
            for m in modes:
                if m == "concat":
                    merged[m].extend((ast_lines + cfg_lines)[: 2 * k])
                else:  # max: interleave, AST first
                    inter = [x for pair in zip(ast_lines, cfg_lines) for x in pair]
                    inter += ast_lines[len(cfg_lines):] + cfg_lines[len(ast_lines):]
                    merged[m].extend(inter[: 2 * k])
        else:
            for j in positive:
                ast_s, cfg_s, _unmapped = explain_line_scores(
                    model, ast_data, cfg_data, j,
                    ast_raw.node_lines, cfg_raw.node_lines, epochs=epochs)
                for m in modes:
                    merged[m].extend(combine[m](ast_s, cfg_s, k=k))

        flaws = [FLAWS[j] for j in positive]
        for m in modes:
            out[m].append({"id": cid, "flaws": flaws, "gold": gold,
                           "pred": _dedup_keep_order(merged[m])})
        if verbose:
            pred = out[primary][-1]["pred"]
            mark = "\u2713" if set(pred[:k]) & _expand(set(gold), tol) else "\u00b7"
            fl = "/".join(display_name(f) for f in flaws)
            tag = f" [{primary}]" if len(modes) > 1 else ""
            print(f"  [{n}/{total}] {mark} {cid} [{fl}]  gold={gold}  "
                  f"pred@{k}{tag}={pred[:k]}", flush=True)
    return out


def _expand(gold: set[int], tol: int) -> set[int]:
    """Widen each gold line into a +/-tol window. tol=0 is exact match."""
    if tol <= 0:
        return set(gold)
    out: set[int] = set()
    for g in gold:
        out.update(range(g - tol, g + tol + 1))
    return out


def _accuracy(rows: list[dict], ks=KS, tol: int = 0) -> dict[int, float]:
    """Accuracy@k via the canonical metric, to avoid a second definition.

    A predicted line counts as a hit if it lies within +/-tol of any gold line,
    implemented by widening the gold set before scoring (tol=0 == exact match).
    """
    from training.evaluate.localisation import top_k_localisation

    pred_lines = [r["pred"] for r in rows]
    gold_lines = [_expand(set(r["gold"]), tol) for r in rows]
    return top_k_localisation(pred_lines, gold_lines, ks=ks)


# --------------------------------------------------------------- benchmark mode

def _benchmark_model(tag: str, run_dir: Path, *, k: int, epochs: int,
                     repeats: int, base_seed: int, device: str,
                     merge: str) -> dict | None:
    """Localise ONE trained run from its provenance. Returns its report.

    Loads the resolved config and the processed data dir the run trained on
    from ``provenance.json``, so nodf runs score against ``processed_nodf``
    records and df runs against ``processed_df``. All tolerances are scored
    from the same seeded passes.
    """
    import torch

    from scgnn.models.dual_gnn import build_model

    ckpt = run_dir / "best_model.pt"
    prov_p = run_dir / "provenance.json"
    if not ckpt.exists() or not prov_p.exists():
        return None                       # ensembles / incomplete runs
    prov = json.loads(prov_p.read_text(encoding="utf-8"))
    config = prov["config"]
    processed = Path(prov["data_dir"])

    model = build_model(config)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()

    test_index = json.loads(_test_index_path(processed).read_text(encoding="utf-8"))
    gold_map = json.loads((processed / "curated_gold_lines.json")
                          .read_text(encoding="utf-8"))
    targets = [e for e in test_index if gold_map.get(e["id"])]
    modes = [merge]

    print(f"\n=== localise {tag}  (conv={config.get('conv')}, data={processed.name}, "
          f"{len(targets)} targets, {repeats} pass(es)) ===", flush=True)

    acc_samples = {t: {kk: [] for kk in KS} for t in TOLS}
    per_pass: list[dict] = []
    n_rows = 0
    for r in range(repeats):
        seed = base_seed + r
        _seed_everything(seed)
        out = _run_pass(model, targets, gold_map, processed, device,
                        k, epochs, modes, verbose=False, method="gnnexplainer")
        rows = out[merge]
        n_rows = len(rows)
        accs = {t: _accuracy(rows, tol=t) for t in TOLS}
        per_pass.append({"seed": seed,
                         "accuracy_at_k": {str(t): {str(kk): round(accs[t][kk], 4)
                                                    for kk in KS} for t in TOLS}})
        for t in TOLS:
            for kk in KS:
                acc_samples[t][kk].append(accs[t][kk])
        a0 = accs[0]
        print(f"  pass {r + 1}/{repeats} (seed={seed}) tol=0: "
              + "  ".join(f"@{kk}={a0[kk]:.3f}" for kk in KS), flush=True)

    mean = {t: {kk: statistics.mean(acc_samples[t][kk]) for kk in KS} for t in TOLS}
    std = {t: {kk: (statistics.stdev(acc_samples[t][kk]) if repeats > 1 else 0.0)
               for kk in KS} for t in TOLS}
    report = {
        "model": tag,
        "conv": config.get("conv"),
        "data_dir": str(processed),
        "n_localisation_targets": n_rows,
        "repeats": repeats,
        "base_seed": base_seed,
        "epochs": epochs,
        "k": k,
        "merge": merge,
        "method": "gnnexplainer",
        "device": device,
        "results_by_tolerance": {
            str(t): {"accuracy_at_k_mean": {str(kk): round(mean[t][kk], 4) for kk in KS},
                     "accuracy_at_k_std": {str(kk): round(std[t][kk], 4) for kk in KS}}
            for t in TOLS},
        "passes": per_pass,
    }
    (run_dir / "localisation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_benchmark(runs_dir: Path, *, k: int, epochs: int, repeats: int,
                  base_seed: int, device: str, merge: str,
                  benchmark_out: str | None) -> int:
    """Localise every trained single run under ``runs_dir``; write the combined
    benchmark JSON (model -> accuracy@k mean/std per tolerance)."""
    run_dirs = sorted(p for p in runs_dir.iterdir()
                      if p.is_dir() and (p / "best_model.pt").exists())
    if not run_dirs:
        raise SystemExit(f"no run directories with best_model.pt under {runs_dir}")
    skipped = sorted(p.name for p in runs_dir.iterdir()
                     if p.is_dir() and not (p / "best_model.pt").exists())
    if skipped:
        print(f"skipping (no single checkpoint to explain): {', '.join(skipped)}")

    combined: dict[str, dict] = {}
    for rd in run_dirs:
        rep = _benchmark_model(rd.name, rd, k=k, epochs=epochs, repeats=repeats,
                               base_seed=base_seed, device=device, merge=merge)
        if rep is not None:
            combined[rd.name] = {"conv": rep["conv"],
                                 "n_targets": rep["n_localisation_targets"],
                                 "results_by_tolerance": rep["results_by_tolerance"]}

    print("\n=== localisation benchmark (tol=0, mean over "
          f"{repeats} seeded pass(es)) ===")
    header = f"  {'model':<14}" + "".join(f"{'@' + str(kk):>9}" for kk in KS)
    print(header)
    for tag in sorted(combined):
        m = combined[tag]["results_by_tolerance"]["0"]["accuracy_at_k_mean"]
        print(f"  {tag:<14}" + "".join(f"{float(m[str(kk)]):>9.3f}" for kk in KS))

    payload = {"ks": list(KS), "tolerances": list(TOLS), "repeats": repeats,
               "base_seed": base_seed, "epochs": epochs, "merge": merge,
               "models": combined}
    outp = Path(benchmark_out) if benchmark_out \
        else runs_dir / "localisation_benchmark.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}  ({len(combined)} models)")
    return 0


# ------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-models", action="store_true",
                    help="Benchmark mode: localise EVERY trained single run under "
                         "--runs-dir (each from its own provenance.json), scoring "
                         "all tolerances from the same passes.")
    ap.add_argument("--runs-dir", default="runs/v2",
                    help="Run matrix directory for --all-models.")
    ap.add_argument("--benchmark-out", default=None,
                    help="Combined benchmark JSON path (--all-models). Defaults to "
                         "<runs-dir>/localisation_benchmark.json")
    ap.add_argument("--checkpoint", default="runs/sage/best_model.pt")
    ap.add_argument("--config", default="configs/sage.yaml")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--feature-config", default=None,
                    help="Defaults to <processed>/feature_config.json (used only for in_dim).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, default=10,
                    help="Lines requested per branch / final (>= max(KS) so @10 is scoreable).")
    ap.add_argument("--epochs", type=int, default=150, help="GNNExplainer steps per branch.")
    ap.add_argument("--repeats", type=int, default=5,
                    help="Seeded passes; reports mean +/- std when >1.")
    ap.add_argument("--seed", type=int, default=0, help="Base seed; pass r uses seed+r.")
    ap.add_argument("--merge", choices=["max", "concat", "both"], default="max",
                    help="Branch combination rule, or 'both' for a paired ablation.")
    ap.add_argument("--method", choices=["gnnexplainer", "attention"], default="gnnexplainer",
                    help="Localisation method. 'gnnexplainer' (default) runs GNNExplainer "
                         "per flaw on any model. 'attention' reads the attention layer's "
                         "weights (requires an attention-bearing checkpoint: gat, gatv2 "
                         "or hybrid) as the proposal's secondary explanation signal.")
    ap.add_argument("--tolerance", type=int, default=0,
                    help="Line tolerance: a prediction within +/-tolerance of a gold "
                         "line counts as a hit. 0 (default) = exact match. Try 1 or 2 "
                         "to report neighbourhood localisation alongside exact. "
                         "(--all-models always scores tolerances 0/1/2.)")
    ap.add_argument("--out", default=None,
                    help="Defaults to <processed>/localisation_report_<merge>.json")
    args = ap.parse_args()

    if args.all_models:
        return run_benchmark(Path(args.runs_dir), k=args.k, epochs=args.epochs,
                             repeats=args.repeats, base_seed=args.seed,
                             device=args.device, merge=("max" if args.merge == "both"
                                                        else args.merge),
                             benchmark_out=args.benchmark_out)

    import torch

    from scgnn.extraction.features import FeatureConfig
    from scgnn.models.dual_gnn import build_model
    from training.config import load_config

    processed = Path(args.processed)
    fc_path = args.feature_config or str(processed / "feature_config.json")
    feat_cfg = FeatureConfig.from_json(fc_path)

    config = load_config(args.config)
    config["in_dim"] = feat_cfg.in_dim
    model = build_model(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    model.to(args.device).eval()
    print(f"loaded {config.get('conv')} model (in_dim={feat_cfg.in_dim}) on {args.device}")

    if args.method == "attention" and config.get("conv") not in ("gat", "gatv2", "hybrid"):
        raise SystemExit(
            f"--method attention requires an attention-bearing checkpoint (gat, gatv2 "
            f"or hybrid), but this model is '{config.get('conv')}'. Point "
            f"--checkpoint/--config at an attention run.")

    test_index = json.loads(_test_index_path(processed).read_text(encoding="utf-8"))
    gold_map = json.loads((processed / "curated_gold_lines.json").read_text(encoding="utf-8"))
    targets = [e for e in test_index if gold_map.get(e["id"])]
    modes = ["concat", "max"] if args.merge == "both" else [args.merge]
    print(f"{len(test_index)} test contracts; {len(targets)} carry gold lines "
          f"(localisation targets)")
    print(f"merge={args.merge}  {args.repeats} seeded pass(es), base seed {args.seed}, "
          f"{args.epochs} explainer epochs\n")

    acc_samples = {m: {k: [] for k in KS} for m in modes}
    hit_counts = {m: {} for m in modes}
    example = {m: None for m in modes}
    per_pass: list[dict] = []

    for r in range(args.repeats):
        seed = args.seed + r
        _seed_everything(seed)
        verbose = (r == 0)
        if args.repeats > 1 and verbose:
            print(f"--- pass 1/{args.repeats} (seed={seed}); per-contract detail ---")
        out = _run_pass(model, targets, gold_map, processed, args.device,
                        args.k, args.epochs, modes, verbose, tol=args.tolerance,
                        method=args.method)
        accs = {m: _accuracy(out[m], tol=args.tolerance) for m in modes}
        per_pass.append({"seed": seed,
                         "accuracy_at_k": {m: {str(k): round(accs[m][k], 4) for k in KS}
                                           for m in modes}})
        for m in modes:
            for k in KS:
                acc_samples[m][k].append(accs[m][k])
            for row in out[m]:
                hc = hit_counts[m].setdefault(row["id"], {k: 0 for k in KS})
                gold_win = _expand(set(row["gold"]), args.tolerance)
                for k in KS:
                    if set(row["pred"][:k]) & gold_win:
                        hc[k] += 1
            if example[m] is None:
                example[m] = out[m]
        summary = "  |  ".join(
            f"{m}: " + " ".join(f"@{k}={accs[m][k]:.3f}" for k in KS)
            for m in modes)
        print(f"pass {r + 1}/{args.repeats} (seed={seed}):  {summary}", flush=True)

    n = len(example[modes[0]]) if example[modes[0]] else 0
    mean = {m: {k: statistics.mean(acc_samples[m][k]) for k in KS} for m in modes}
    std = {m: {k: (statistics.stdev(acc_samples[m][k]) if args.repeats > 1 else 0.0)
               for k in KS} for m in modes}

    print("\n=== localisation accuracy (expert test split, n="
          f"{n}) ===")
    if args.merge == "both":
        print(f"  {'merge':<8}" + "".join(f"{'@' + str(k):<16}" for k in KS))
        for m in modes:
            cells = "".join(f"{mean[m][k]:.3f} \u00b1 {std[m][k]:.3f}   " for k in KS)
            print(f"  {m:<8}{cells}")
        dl = {k: mean['max'][k] - mean['concat'][k] for k in KS}
        delta_label = "\u0394(max-concat)"
        print(f"  {delta_label}:  " + "  ".join(f"@{k}={dl[k]:+.3f}" for k in KS))
    else:
        m = modes[0]
        for k in KS:
            if args.repeats > 1:
                print(f"  accuracy@{k}: {mean[m][k]:.3f} \u00b1 {std[m][k]:.3f}   "
                      f"(mean of {args.repeats} seeded runs)")
            else:
                print(f"  accuracy@{k}: {mean[m][k]:.3f}   ({round(mean[m][k] * n)}/{n})")

    per_contract = []
    base_rows = example[modes[0]] or []
    by_id = {m: {r["id"]: r for r in (example[m] or [])} for m in modes}
    for row in base_rows:
        cid = row["id"]
        entry = {"id": cid, "flaws": row["flaws"], "gold": row["gold"]}
        for m in modes:
            entry[f"hit_at_k_over_runs__{m}"] = {str(k): hit_counts[m][cid][k] for k in KS}
            entry[f"example_pred__{m}"] = by_id[m][cid]["pred"][:args.k]
        per_contract.append(entry)

    report = {
        "n_localisation_targets": n,
        "repeats": args.repeats,
        "base_seed": args.seed,
        "epochs": args.epochs,
        "tolerance": args.tolerance,
        "method": args.method,
        "device": args.device,
        "merge_modes": modes,
        "results": {m: {"accuracy_at_k_mean": {str(k): round(mean[m][k], 4) for k in KS},
                        "accuracy_at_k_std": {str(k): round(std[m][k], 4) for k in KS}}
                    for m in modes},
        "passes": per_pass,
        "per_contract": per_contract,
    }
    if args.merge == "both":
        report["delta_max_minus_concat"] = {
            str(k): round(mean["max"][k] - mean["concat"][k], 4) for k in KS}
    suffix = args.merge if args.method == "gnnexplainer" else f"{args.method}_{args.merge}"
    outp = args.out or str(processed / f"localisation_report_{suffix}.json")
    Path(outp).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())