"""Build the processed dataset: extract -> fit (train-only) -> encode -> index.

This is the middle of the pipeline: it turns raw contracts plus labels into the
per-contract records and the split index files the training loop consumes, in
the only correct order. The node-type vocabulary and the PCA are fitted on the
TRAIN split only, then every split is encoded with those frozen artefacts.

v1 split policy (retained, unchanged, for the before/after table)
----------------------------------------------------------------
``plan_splits`` derives a stratified Curated test split at build time:
  test  = frozen stratified Curated split; train = Wild + Curated remainder;
  val   = held-out slice of Wild.

v2 split policy (Workstream B, non-negotiable #3 and #4)
-------------------------------------------------------
``plan_splits_v2`` READS two frozen manifests instead of deriving anything, so
the benchmarks cannot drift between runs:
  test_a = tool-labelled Wild contracts (frozen manifest);
  test_b = expert Curated contracts (frozen manifest, line annotations kept);
  train  = the labelled Wild pool MINUS every hash in either manifest;
  val    = a stratified slice of that pool.
The firewall is enforced against the UNION of both test sets and fails the build
on any leak.

Data-flow edges (Workstream C)
------------------------------
Extraction adds typed data-flow edges to the CFG view. The ablation's "without"
arm does NOT re-extract: ``materialise(..., with_data_flow=False)`` strips the
data-flow edges from the SAME cached graphs via ``RawGraph.without_data_flow()``,
so both arms share identical node features and extraction is paid for once.

Memory
------
At full-corpus scale the train split can hold millions of unique snippets;
embedding all of them to fit a 64-dim PCA exhausts RAM. ``pca_fit_sample``
bounds how many snippets are embedded for the fit (the projection is stable far
below the full set); the sample is drawn deterministically from ``seed``.

Status: the planners are pure and unit-tested. ``materialise`` needs the full
stack (solc/Slither, CodeBERT, torch). Extraction is cached to
``out/raw/<hash>.json`` so a rerun or the second pass never re-runs Slither.
"""

from __future__ import annotations

import json
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scgnn.extraction.features import (
    FeatureConfig, FeatureEncoder, fit_feature_config, fit_pca,
)
from scgnn.extraction.graph_types import RawGraph
from scgnn.schema import FLAWS, N_FLAWS
from training.data.firewall import (
    assert_firewall, content_hash, dedup_wild_against_curated,
    stratified_multilabel_split,
)

SPLITS_V1 = ("train", "val", "test")
SPLITS_V2 = ("train", "val", "test_a", "test_b")


class ExtractTimeout(Exception):
    """Raised when one contract's extraction exceeds the time budget."""


@contextmanager
def _time_limit(seconds: float):
    """Abort the wrapped block after ``seconds`` via SIGALRM (Unix, main thread).

    Slither's Python API and the solc subprocess have no timeout of their own, so
    a single pathological contract can hang the whole build with the CPU idle
    (blocked on a stuck compile). This raises :class:`ExtractTimeout`, which the
    extraction loop catches and logs as a per-contract failure. A no-op where
    SIGALRM is unavailable (non-Unix, or not the main thread).
    """
    usable = seconds and seconds > 0 and hasattr(signal, "SIGALRM")
    if not usable:
        yield
        return

    def _on_alarm(signum, frame):
        raise ExtractTimeout(f"extraction exceeded {seconds:g}s")

    try:
        old = signal.signal(signal.SIGALRM, _on_alarm)
    except ValueError:                 # not the main thread -> cannot use SIGALRM
        yield
        return
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


@dataclass
class Item:
    cid: str
    path: str
    y: list[int]
    split: str               # v1: train|val|test   v2: train|val|test_a|test_b
    provenance: str          # "wild" | "curated"
    gold_lines: list[int] = field(default_factory=list)


@dataclass
class SplitPlan:
    items: list[Item]
    counts: dict


def _read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


# ------------------------------------------------------------------ v1 planner

def plan_splits(wild_paths: dict[str, str], wild_labels: dict[str, list[int]],
                curated: dict[str, dict], *, val_frac: float = 0.2,
                test_frac: float = 0.3, seed: int = 42,
                max_wild: int | None = None) -> SplitPlan:
    """v1 planner, UNCHANGED, so the v1 results stay reproducible.

    ``wild_paths``: {cid: path}; ``wild_labels``: {cid: [5]};
    ``curated``: {cid: {"path", "y", "lines"}}. Pure.
    """
    wild_sources = {cid: _read(p) for cid, p in wild_paths.items()}
    curated_sources = {cid: _read(v["path"]) for cid, v in curated.items()}
    kept, _removed, n_removed = dedup_wild_against_curated(wild_sources, curated_sources)
    kept = [c for c in sorted(kept) if c in wild_labels]   # must have a label
    if max_wild is not None:
        kept = kept[:max_wild]

    if kept:
        Yw = np.array([wild_labels[c] for c in kept], dtype=int)
        wtr, wva = stratified_multilabel_split(Yw, test_frac=val_frac, seed=seed)
    else:
        wtr, wva = np.array([], int), np.array([], int)
    wild_train = [kept[i] for i in wtr]
    wild_val = [kept[i] for i in wva]

    cur_ids = sorted(curated)
    if cur_ids:
        Yc = np.array([curated[c]["y"] for c in cur_ids], dtype=int)
        ctr, cte = stratified_multilabel_split(Yc, test_frac=test_frac, seed=seed)
    else:
        ctr, cte = np.array([], int), np.array([], int)
    cur_remainder = [cur_ids[i] for i in ctr]
    cur_test = [cur_ids[i] for i in cte]

    items: list[Item] = []
    for c in wild_train:
        items.append(Item(c, wild_paths[c], list(wild_labels[c]), "train", "wild"))
    for c in cur_remainder:
        items.append(Item(c, curated[c]["path"], list(curated[c]["y"]), "train",
                          "curated", list(curated[c].get("lines", []))))
    for c in wild_val:
        items.append(Item(c, wild_paths[c], list(wild_labels[c]), "val", "wild"))
    for c in cur_test:
        items.append(Item(c, curated[c]["path"], list(curated[c]["y"]), "test",
                          "curated", list(curated[c].get("lines", []))))

    train_hashes = {content_hash(_read(it.path)) for it in items if it.split == "train"}
    test_hashes = {content_hash(_read(it.path)) for it in items if it.split == "test"}
    assert_firewall(train_hashes, test_hashes)

    counts = {
        "train": sum(it.split == "train" for it in items),
        "val": sum(it.split == "val" for it in items),
        "test": sum(it.split == "test" for it in items),
        "wild_dedup_removed": n_removed,
    }
    return SplitPlan(items=items, counts=counts)


# ------------------------------------------------------------------ v2 planner

def plan_splits_v2(
    wild_paths: dict[str, str],
    wild_labels: dict[str, list[int]],
    *,
    test_a_manifest: str | Path,
    test_b_manifest: str | Path,
    curated_lines: dict[str, list[int]] | None = None,
    val_frac: float = 0.1,
    seed: int = 42,
    max_wild: int | None = None,
) -> SplitPlan:
    """v2 planner: read the two FROZEN manifests; firewall train/val against both.

    ``wild_paths``  : cid -> .sol path for the labelled Wild pool.
    ``wild_labels`` : cid -> [5] union labels (from scripts/label.py).
    ``curated_lines``: cid -> gold line numbers, carried into Test B for
                      localisation.

    Train/val come from the Wild pool AFTER removing every contract whose content
    hash appears in either manifest, so no test contract (nor a whitespace-variant
    copy of one) can ever be trained on. ``assert_firewall`` fails the build on a
    leak. Pure apart from reading the manifests and the contract sources.
    """
    from training.data.testsets import firewall_hashes, read_manifest

    curated_lines = curated_lines or {}

    rows_a = read_manifest(test_a_manifest)
    rows_b = read_manifest(test_b_manifest)
    reserved = firewall_hashes(test_a_manifest, test_b_manifest)

    items: list[Item] = []
    for r in rows_a:
        items.append(Item(r["contract_id"], r["path"],
                          [int(r[f]) for f in FLAWS], "test_a", "wild"))
    for r in rows_b:
        items.append(Item(r["contract_id"], r["path"],
                          [int(r[f]) for f in FLAWS], "test_b", "curated",
                          list(curated_lines.get(r["contract_id"], []))))

    pool: list[str] = []
    for cid in sorted(wild_paths):
        if cid not in wild_labels:
            continue                       # unlabelled -> not usable for training
        try:
            h = content_hash(_read(wild_paths[cid]))
        except OSError:
            continue
        if h in reserved:
            continue                       # firewalled: reserved by a test manifest
        pool.append(cid)
    if max_wild is not None:
        pool = pool[:max_wild]

    if pool:
        Y = np.array([wild_labels[c] for c in pool], dtype=int)
        tr_idx, va_idx = stratified_multilabel_split(Y, test_frac=val_frac, seed=seed)
    else:
        tr_idx, va_idx = np.array([], int), np.array([], int)

    for i in tr_idx:
        c = pool[i]
        items.append(Item(c, wild_paths[c], list(wild_labels[c]), "train", "wild"))
    for i in va_idx:
        c = pool[i]
        items.append(Item(c, wild_paths[c], list(wild_labels[c]), "val", "wild"))

    trainval_hashes = {content_hash(_read(it.path))
                       for it in items if it.split in ("train", "val")}
    assert_firewall(trainval_hashes, reserved)

    counts = {s: sum(it.split == s for it in items) for s in SPLITS_V2}
    counts["reserved_hashes"] = len(reserved)
    counts["wild_pool"] = len(pool)
    return SplitPlan(items=items, counts=counts)


# ---------------------------------------------------------------- materialise

def materialise(plan: SplitPlan, extract_fn, embedder, out_dir: str, *,
                embed_dim: int = 64, seed: int = 42, extract_timeout: float = 120,
                embed_batch: int = 128, pca_fit_sample: int = 200_000,
                with_data_flow: bool = True) -> dict:
    """Extract, fit train-only artefacts, encode every split, write indices.

    ``extract_fn(path) -> (ast RawGraph, cfg RawGraph)``; ``embedder`` exposes
    ``embed_many(snippets, batch_size)``. Needs torch + the extraction stack.

    ``with_data_flow=False`` is the ablation's "without" arm: the SAME cached
    graphs are used, with data-flow edges stripped, so the two arms differ only
    in the data-flow representation and extraction is not repeated.

    ``pca_fit_sample`` bounds the snippets embedded to FIT the PCA (0 = all).
    At full-corpus scale, embedding every unique snippet exhausts RAM; a 64-dim
    projection is stable far below the full set.
    """
    import joblib
    import torch

    out = Path(out_dir)
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    splits = SPLITS_V2 if any(it.split in ("test_a", "test_b") for it in plan.items) \
        else SPLITS_V1

    # Pass 1 - extract every contract (cached), tolerating per-contract failures
    # and hangs (a stuck Slither/solc on one contract is timed out and skipped).
    raws: dict[str, tuple[RawGraph, RawGraph]] = {}
    failed: list[tuple[str, str]] = []
    n = len(plan.items)
    for i, it in enumerate(plan.items):
        h = content_hash(_read(it.path))
        cache = out / "raw" / f"{h}.json"
        try:
            ast = cfg = None
            if cache.exists():
                try:
                    d = json.loads(cache.read_text(encoding="utf-8"))
                    ast, cfg = RawGraph.from_dict(d["ast"]), RawGraph.from_dict(d["cfg"])
                except Exception:
                    ast = cfg = None    # partial/corrupt cache (e.g. Ctrl-C) -> re-extract
            if ast is None:
                with _time_limit(extract_timeout):
                    ast, cfg = extract_fn(it.path)
                tmp = cache.with_name(cache.name + ".tmp")   # atomic -> safe to interrupt
                tmp.write_text(json.dumps({"ast": ast.to_dict(), "cfg": cfg.to_dict()}),
                               encoding="utf-8")
                tmp.replace(cache)
            raws[it.cid] = (ast, cfg)
        except Exception as exc:               # skip & log; never guess
            failed.append((it.cid, str(exc)))
        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"  extracted {i + 1}/{n} (ok {len(raws)}, failed {len(failed)})",
                  flush=True)
    print(f"pass 1 done: {len(raws)}/{n} extracted, {len(failed)} failed", flush=True)

    # The ablation: strip data-flow edges from the cached CFGs. Node features and
    # the node->lines map are untouched, so the two arms differ only in the edges
    # (and the degree features those edges induce - state this in the write-up).
    if not with_data_flow:
        raws = {cid: (ast, cfg.without_data_flow()) for cid, (ast, cfg) in raws.items()}
        print("ablation: data-flow edges stripped from every CFG", flush=True)

    n_df = sum(cfg.n_data_flow_edges for _, cfg in raws.values())
    n_degraded = sum(1 for _, cfg in raws.values() if cfg.degraded)
    print(f"  data-flow edges: {n_df}; degraded CFGs: {n_degraded}", flush=True)

    # Fit on TRAIN only.
    print("fitting train-only feature vocab + PCA ...", flush=True)
    train_ids = [it.cid for it in plan.items if it.split == "train" and it.cid in raws]
    train_graphs = [g for cid in train_ids for g in raws[cid]]

    all_snips = sorted({s for cid in train_ids for g in raws[cid] for s in g.snippets})
    n_all = len(all_snips)
    if pca_fit_sample and n_all > pca_fit_sample:
        rng = np.random.default_rng(seed)
        pick = rng.choice(n_all, size=pca_fit_sample, replace=False)
        train_snippets = [all_snips[i] for i in sorted(pick)]
        print(f"  PCA fit: sampling {len(train_snippets):,} of {n_all:,} unique "
              f"train snippets (memory-safe)", flush=True)
    else:
        train_snippets = all_snips
        print(f"  PCA fit: using all {n_all:,} unique train snippets", flush=True)

    print(f"  embedding {len(train_snippets):,} snippets (batch {embed_batch}) ...",
          flush=True)
    emb = (embedder.embed_many(train_snippets, batch_size=embed_batch)
           if train_snippets else np.zeros((0, 768), np.float32))
    k = int(min(embed_dim, emb.shape[0] or embed_dim, emb.shape[1] or embed_dim))
    feat_cfg = fit_feature_config(train_graphs, embed_dim=k)
    pca = fit_pca(emb, embed_dim=k, seed=seed) if emb.shape[0] >= k and k > 0 else None
    del emb                                   # free before pass 2
    encoder = FeatureEncoder(feat_cfg, embedder if pca is not None else None, pca)

    # Pass 2 - encode every split and write records + indices.
    print("encoding all splits -> records ...", flush=True)
    indices: dict[str, list[dict]] = {s: [] for s in splits}
    for it in plan.items:
        if it.cid not in raws:
            continue
        ast, cfg = raws[it.cid]
        ax, ai = encoder.encode_array(ast)
        cx, ci = encoder.encode_array(cfg)
        record = {
            "ast_x": torch.from_numpy(ax), "ast_edge_index": torch.from_numpy(ai),
            "cfg_x": torch.from_numpy(cx), "cfg_edge_index": torch.from_numpy(ci),
            "y": torch.tensor(it.y, dtype=torch.float),
        }
        rp = out / "records" / f"{content_hash(_read(it.path))}.pt"
        torch.save(record, rp)
        indices[it.split].append({"id": it.cid, "path": str(rp)})

    feat_cfg.to_json(str(out / "feature_config.json"))
    if pca is not None:
        joblib.dump(pca, out / "pca.joblib")
    for split in splits:
        (out / f"{split}_index.json").write_text(
            json.dumps(indices[split], indent=2), encoding="utf-8")

    # Gold lines for localisation: Test B in v2, the Curated test split in v1.
    gold_split = "test_b" if "test_b" in splits else "test"
    gold = {it.cid: it.gold_lines for it in plan.items
            if it.split == gold_split and it.gold_lines}
    (out / "curated_gold_lines.json").write_text(json.dumps(gold, indent=2),
                                                 encoding="utf-8")

    report = {
        "counts": plan.counts,
        "encoded": {k2: len(v) for k2, v in indices.items()},
        "failed": failed,
        "in_dim": feat_cfg.in_dim,
        "embed_dim": k,
        "with_data_flow": bool(with_data_flow),
        "data_flow_edges": int(n_df),
        "degraded_cfgs": int(n_degraded),
        "pca_fit_snippets": len(train_snippets),
        "pca_fit_snippets_available": n_all,
    }
    (out / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report