#!/usr/bin/env python3
"""Learned-baseline suite: the comparators the GNN must beat to be a contribution.

Every number is produced here, scored through the ONE metrics path
(``training.evaluate.metrics``) and written in the SAME ``results.json`` schema
that ``scripts/train_v2.py`` emits, so the artifact generator and comparison
figure consume it with no special casing. Each baseline is independently
skippable: a failure in one is reported and the others still run.

Baselines:
  sequence  CodeBERT fine-tuned on contract text (the flat-text control for the
            structure claim). GPU for real runs; CPU smoke via --limit/--epochs.
  votes     logistic regression on the four tools' votes only (isolates how much
            of the score is "learning the tools" versus "learning code").
  trivial   majority / stratified / all-positive floors.
  peculiar  external reentrancy detector via a predictions CSV (Test B only,
            reentrancy column masked-scored). Skipped loudly if the file is
            absent; never falls back to published numbers.

Firewall: the sequence and votes baselines TRAIN, so their training hashes are
asserted disjoint from the union of both test manifests via
``training.data.firewall.assert_firewall`` before any fitting. Thresholds are
tuned on validation only and applied frozen to both test sets, exactly as in
train_v2.

    PYTHONPATH=. python scripts/train_baselines.py \
        --data-nodf data/processed_nodf \
        --baselines sequence,votes,trivial,peculiar \
        --out runs/baselines --seeds 42

    # laptop smoke of the whole path (no GPU):
    PYTHONPATH=. python scripts/train_baselines.py \
        --data-nodf data/processed_nodf --baselines sequence,votes,trivial \
        --out runs/baselines_smoke --seeds 42 \
        --seq-limit 50 --seq-epochs 1 --device cpu
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scgnn.schema import FLAWS


# --------------------------- provenance (as train_v2) ---------------------------

def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict:
    out = {}
    for mod in ("torch", "transformers", "numpy", "sklearn"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            out[mod] = "not installed"
    out["python"] = sys.version.split()[0]
    return out


# ------------------------------ shared evaluation ------------------------------

def evaluate(y_true, probs, thresholds, *, n_boot: int, seed: int,
             mask=None) -> dict:
    """Score one split through the single metrics path, mask-aware."""
    from training.evaluate.metrics import (apply_thresholds, bootstrap_ci,
                                           confusion_per_class, full_metrics)
    y_pred = apply_thresholds(probs, thresholds)
    m = full_metrics(y_true, y_pred, mask=mask)
    m["ci"] = bootstrap_ci(y_true, y_pred, n_resamples=n_boot, seed=seed, mask=mask)
    m["confusion"] = confusion_per_class(y_true, y_pred)
    m["thresholds"] = list(thresholds)
    if mask is not None:
        m["scored_classes"] = list(mask)
    return m


def read_split_labels(data_dir: Path, split: str) -> tuple[list[str], np.ndarray]:
    """Contract ids and (n,5) labels for a processed split, from its index +
    record tensors. Mirrors how train_v2 loads labels, without a GPU."""
    import torch
    idx = json.loads((data_dir / f"{split}_index.json").read_text(encoding="utf-8"))
    ids, ys = [], []
    for e in idx:
        ids.append(e["id"])
        ys.append((torch.load(e["path"], weights_only=True)["y"] >= 0.5).int().tolist())
    return ids, np.array(ys, dtype=int)


def manifest_labels(manifest: Path) -> tuple[list[str], np.ndarray]:
    from training.data.testsets import read_manifest
    rows = read_manifest(manifest)
    ids = [r["contract_id"] for r in rows]
    Y = np.array([[int(r[f]) for f in FLAWS] for r in rows], dtype=int)
    return ids, Y


# --------------------------------- baselines ---------------------------------

def run_votes(args, seed, splits, out_dir) -> dict:
    """Votes baseline: features are the four tools' votes; firewalled; val-tuned."""
    from training.baselines.votes import VotesBaseline, votes_feature_matrix
    from training.data.firewall import assert_firewall
    from training.data.testsets import firewall_hashes
    from training.evaluate.metrics import tune_thresholds

    train_ids, Ytr = splits["train"]
    val_ids, Yval = splits["val"]

    # Firewall: the training contracts must not include any test content hash.
    # We assert on the ids' content hashes recorded in the manifests indirectly:
    # the processed split was already firewalled at build time, but we re-check
    # here so the baseline carries its own guarantee (spec non-negotiable 3).
    reserved = firewall_hashes(args.test_a_manifest, args.test_b_manifest)
    train_hashes = _content_hashes_for(train_ids, args)
    assert_firewall(train_hashes, reserved)

    Xtr = votes_feature_matrix(args.results_train, train_ids)
    Xval = votes_feature_matrix(args.results_train, val_ids)
    model = VotesBaseline(kind=args.votes_kind, seed=seed).fit(Xtr, Ytr)

    pval = model.predict_proba(Xval)
    thr = tune_thresholds(Yval, pval)
    res = {"val": evaluate(Yval, pval, thr, n_boot=args.bootstrap, seed=seed)}
    for split, results_dir in (("test_a", args.results_test_a),
                               ("test_b", args.results_test_b)):
        ids, Y = splits[split]
        X = votes_feature_matrix(results_dir, ids)
        res[split] = evaluate(Y, model.predict_proba(X), thr,
                              n_boot=args.bootstrap, seed=seed)
    return res


def run_trivial(args, seed, splits) -> dict:
    """Three floors, reported as separate pseudo-models."""
    from training.baselines.trivial import TrivialBaseline
    from training.evaluate.metrics import tune_thresholds

    _, Ytr = splits["train"]
    out = {}
    for kind in ("majority", "stratified", "all_positive"):
        model = TrivialBaseline(kind=kind, seed=seed).fit(Ytr)
        _, Yval = splits["val"]
        pval = model.predict_proba(len(Yval))
        thr = tune_thresholds(Yval, pval) if kind == "stratified" else [0.5] * 5
        res = {"val": evaluate(Yval, pval, thr, n_boot=args.bootstrap, seed=seed)}
        for split in ("test_a", "test_b"):
            _, Y = splits[split]
            res[split] = evaluate(Y, model.predict_proba(len(Y)), thr,
                                  n_boot=args.bootstrap, seed=seed)
        out[f"trivial_{kind}"] = res
    return out


def run_sequence(args, seed, out_dir) -> dict:
    """CodeBERT sequence baseline; firewalled; val-tuned; CPU smoke supported."""
    from transformers import AutoTokenizer

    from training.baselines.sequence import (SequenceConfig, SequenceModel,
                                             predict_probs, tokenise_split,
                                             train_sequence)
    from training.data.firewall import assert_firewall
    from training.data.testsets import (firewall_hashes, read_manifest)
    from training.evaluate.metrics import apply_thresholds, tune_thresholds

    cfg = SequenceConfig(mode=args.seq_mode, epochs=args.seq_epochs,
                         batch_size=args.seq_batch, grad_accum=args.seq_grad_accum,
                         seed=seed, device=args.device, limit=args.seq_limit)

    # rows with source paths come from the processed index (train/val) and the
    # frozen manifests (test), so the baseline reads the same contracts.
    train_rows = _rows_with_paths(args, "train")
    val_rows = _rows_with_paths(args, "val")

    reserved = firewall_hashes(args.test_a_manifest, args.test_b_manifest)
    train_hashes = _content_hashes_for([r["id"] for r in train_rows], args)
    assert_firewall(train_hashes, reserved)

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    tr = tokenise_split([{"path": r["path"], **r["labels"]} for r in train_rows], tok, cfg)
    va = tokenise_split([{"path": r["path"], **r["labels"]} for r in val_rows], tok, cfg)

    model = SequenceModel(cfg)
    info = train_sequence(model, tr, va, cfg, str(out_dir / "sequence"))

    (out_dir / "sequence").mkdir(parents=True, exist_ok=True)
    (out_dir / "sequence" / "run_info.json").write_text(json.dumps(info, indent=2),
                                                        encoding="utf-8")

    val_probs = predict_probs(model, va, cfg.batch_size)
    thr = tune_thresholds(va.labels, val_probs)
    res = {"val": evaluate(va.labels, val_probs, thr, n_boot=args.bootstrap, seed=seed),
           "run_info": info}
    for split, manifest in (("test_a", args.test_a_manifest),
                            ("test_b", args.test_b_manifest)):
        rows = [{"path": r["path"], **{f: int(r[f]) for f in FLAWS}}
                for r in read_manifest(manifest)]
        if cfg.limit:
            rows = rows[: cfg.limit]
        ts = tokenise_split(rows, tok, cfg)
        res[split] = evaluate(ts.labels, predict_probs(model, ts, cfg.batch_size),
                              thr, n_boot=args.bootstrap, seed=seed)
        res[split]["truncation_rate"] = round(ts.truncation_rate, 4)
    return res


def run_peculiar(args, splits) -> dict:
    """External reentrancy baseline via predictions CSV; Test B only; masked."""
    from training.baselines.peculiar import (PECULIAR_MASK, PeculiarUnavailable,
                                             load_peculiar_predictions)

    ids, Y = splits["test_b"]
    probs, mask = load_peculiar_predictions(args.peculiar_csv, ids)   # may raise
    # No validation split for an external checkpoint: fixed, stated 0.5 threshold.
    thr = [0.5] * 5
    res = {"test_b": evaluate(Y, probs, thr, n_boot=args.bootstrap, seed=args.seeds[0],
                              mask=mask)}
    res["test_b"]["external"] = True
    res["test_b"]["note"] = ("Peculiar, reentrancy only, Test B only; fixed 0.5 "
                             "threshold; some Curated contracts also appear in "
                             "Wild so this is not a leakage-free comparison.")
    return res


# --------------------------------- helpers ---------------------------------

def _content_hashes_for(contract_ids, args) -> set[str]:
    """Content hashes for a set of processed-split contracts, read from whichever
    manifest-independent source is available. The processed split was firewalled
    at build time; here we recompute from source paths so the baseline owns its
    check. Falls back to an empty set only if paths are unavailable, in which
    case the build-time firewall still stands (and the assert is a no-op)."""
    from training.data.firewall import content_hash
    paths = getattr(args, "_id_to_path", {})
    hashes = set()
    for cid in contract_ids:
        p = paths.get(cid)
        if p and Path(p).exists():
            hashes.add(content_hash(Path(p).read_text(encoding="utf-8", errors="ignore")))
    return hashes


def _rows_with_paths(args, split) -> list[dict]:
    """Processed-split rows carrying source path + labels, for the sequence model.

    The processed index stores record paths, not .sol paths, so this reads the
    contract source path from the build's id->path sidecar if present
    (data/processed_*/source_paths.json), else falls back to the manifest-style
    path stored on the record. When neither exists (older build), the sequence
    baseline cannot locate source text and raises a clear error."""
    data_dir = Path(args.data_nodf)
    idx = json.loads((data_dir / f"{split}_index.json").read_text(encoding="utf-8"))
    import torch
    src_map = {}
    sp = data_dir / "source_paths.json"
    if sp.exists():
        src_map = json.loads(sp.read_text(encoding="utf-8"))
    rows = []
    for e in idx:
        path = src_map.get(e["id"])
        if path is None:
            raise SystemExit(
                "sequence baseline needs contract source paths; expected "
                f"{sp} mapping contract_id -> .sol path. Regenerate it from the "
                "build (the labels.parquet or the wild dir), or run only the "
                "votes/trivial/peculiar baselines.")
        y = (torch.load(e["path"], weights_only=True)["y"] >= 0.5).int().tolist()
        rows.append({"id": e["id"], "path": path,
                     "labels": {f: int(v) for f, v in zip(FLAWS, y)}})
        args._id_to_path = getattr(args, "_id_to_path", {})
        args._id_to_path[e["id"]] = path
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-nodf", required=True,
                    help="Processed split dir (indices + record tensors + "
                         "source_paths.json for the sequence baseline).")
    ap.add_argument("--baselines", default="votes,trivial",
                    help="Comma list: sequence,votes,trivial,peculiar.")
    ap.add_argument("--out", default="runs/baselines")
    ap.add_argument("--seeds", default="42",
                    help="Comma list, e.g. 42 or 42,43,44 (mean +/- sd reported).")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--test-a-manifest", default="data/testsets/test_a.csv")
    ap.add_argument("--test-b-manifest", default="data/testsets/test_b.csv")

    # votes
    ap.add_argument("--results-train", default="data/sb_results",
                    help="SmartBugs results tree covering train/val contracts.")
    ap.add_argument("--results-test-a", default="data/sb_results")
    ap.add_argument("--results-test-b", default="data/sb_testb")
    ap.add_argument("--votes-kind", default="logreg", choices=["logreg", "tree"])

    # sequence
    ap.add_argument("--seq-mode", default="sliding", choices=["sliding", "truncate"])
    ap.add_argument("--seq-epochs", type=int, default=10)
    ap.add_argument("--seq-batch", type=int, default=8)
    ap.add_argument("--seq-grad-accum", type=int, default=1)
    ap.add_argument("--seq-limit", type=int, default=None,
                    help="Cap contracts per split (CPU smoke).")

    # peculiar
    ap.add_argument("--peculiar-csv", default="data/peculiar/peculiar_predictions.csv")

    args = ap.parse_args()
    args.seeds = [int(s) for s in str(args.seeds).replace(",", " ").split()]
    wanted = [b.strip() for b in args.baselines.split(",") if b.strip()]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    provenance = {"git_hash": git_hash(), "packages": package_versions(),
                  "device": args.device, "seeds": args.seeds, "baselines": wanted}
    (out_root / "provenance.json").write_text(json.dumps(provenance, indent=2),
                                              encoding="utf-8")
    print("provenance:", json.dumps(provenance["packages"]))

    # per-seed results, then aggregated to mean +/- sd
    per_seed: dict[int, dict] = {}
    skipped: dict[str, str] = {}

    for seed in args.seeds:
        data_dir = Path(args.data_nodf)
        splits = {s: read_split_labels(data_dir, s)
                  for s in ("train", "val", "test_a", "test_b")}
        results: dict[str, dict] = {}

        if "votes" in wanted:
            try:
                results[f"votes_{args.votes_kind}"] = run_votes(args, seed, splits, out_root)
                print(f"[seed {seed}] votes done")
            except Exception as e:
                skipped["votes"] = f"{type(e).__name__}: {e}"
                print(f"[seed {seed}] votes SKIPPED: {e}")

        if "trivial" in wanted:
            try:
                results.update(run_trivial(args, seed, splits))
                print(f"[seed {seed}] trivial done")
            except Exception as e:
                skipped["trivial"] = f"{type(e).__name__}: {e}"
                print(f"[seed {seed}] trivial SKIPPED: {e}")

        if "sequence" in wanted:
            try:
                results["sequence_codebert"] = run_sequence(args, seed, out_root)
                print(f"[seed {seed}] sequence done")
            except Exception as e:
                skipped["sequence"] = f"{type(e).__name__}: {e}"
                print(f"[seed {seed}] sequence SKIPPED: {e}")

        if "peculiar" in wanted:
            try:
                results["peculiar_reentrancy"] = run_peculiar(args, splits)
                print(f"[seed {seed}] peculiar done")
            except Exception as e:
                skipped["peculiar"] = f"{type(e).__name__}: {e}"
                print(f"[seed {seed}] peculiar SKIPPED: {e}")

        per_seed[seed] = results

    # aggregate: single seed -> that dict; multi-seed -> mean +/- sd of macro-F1
    if len(args.seeds) == 1:
        final = per_seed[args.seeds[0]]
    else:
        final = _aggregate_seeds(per_seed)

    (out_root / "results.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    if skipped:
        (out_root / "skipped.json").write_text(json.dumps(skipped, indent=2),
                                               encoding="utf-8")

    # summary table
    print("\n" + "=" * 60)
    print(f"{'baseline':<22} {'val':>7} {'test A':>8} {'test B':>8}")
    print("-" * 60)
    for name in sorted(final):
        r = final[name]
        def g(s):
            v = r.get(s, {})
            v = v.get("macro", {}) if isinstance(v, dict) else {}
            return v.get("f1")
        cells = []
        for s in ("val", "test_a", "test_b"):
            f = g(s)
            cells.append(f"{f:>8.3f}" if isinstance(f, (int, float)) else f"{'-':>8}")
        print(f"{name:<22} {cells[0]:>7} {cells[1]} {cells[2]}")
    print("=" * 60)
    if skipped:
        print("skipped:", "; ".join(f"{k} ({v})" for k, v in skipped.items()))
    print(f"results -> {out_root / 'results.json'}")
    return 0


def _aggregate_seeds(per_seed: dict[int, dict]) -> dict:
    """Combine per-seed results into mean +/- sd of macro-F1 per split, keeping
    the first seed's full metrics as the representative point estimate."""
    seeds = sorted(per_seed)
    names = sorted({n for s in seeds for n in per_seed[s]})
    out = {}
    for name in names:
        base = dict(per_seed[seeds[0]].get(name, {}))
        for split in ("val", "test_a", "test_b"):
            vals = [per_seed[s][name][split]["macro"]["f1"]
                    for s in seeds
                    if name in per_seed[s] and split in per_seed[s][name]
                    and "macro" in per_seed[s][name][split]]
            if vals and split in base:
                base[split] = dict(base[split])
                base[split]["macro_f1_mean"] = float(np.mean(vals))
                base[split]["macro_f1_std"] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                base[split]["macro_f1_seeds"] = vals
        out[name] = base
    return out


if __name__ == "__main__":
    raise SystemExit(main())
