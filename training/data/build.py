"""Build the processed dataset: extract -> fit (train-only) -> encode -> index.

This is the missing middle of the pipeline: it turns raw contracts plus labels
into the per-contract records and the train/val/test index files that the
training loop consumes, and it does so in the only correct order — the node-type
vocabulary and the PCA are fitted on the TRAIN split *only*, then every split is
encoded with those frozen artefacts.

Split policy (the deployed/headline arrangement, Condition C):
* test  = the frozen, stratified Curated split (expert labels);
* train = Wild (auto-labelled) + the Curated remainder (expert labels);
* val   = a held-out slice of Wild, for early stopping.
The A/B/C experiment later subsets these same indices.

Status: ``plan_splits`` is pure and unit-tested. ``materialise`` needs the full
stack (solc/Slither, CodeBERT, torch) and runs on the Studio. Extraction is
cached to ``out/raw/<hash>.json`` so a rerun or the second pass never re-runs
Slither.
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
from scgnn.schema import N_FLAWS
from training.data.firewall import (
    assert_firewall, content_hash, dedup_wild_against_curated,
    stratified_multilabel_split,
)


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
    split: str               # "train" | "val" | "test"
    provenance: str          # "wild" | "curated"
    gold_lines: list[int] = field(default_factory=list)


@dataclass
class SplitPlan:
    items: list[Item]
    counts: dict


def _read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def plan_splits(wild_paths: dict[str, str], wild_labels: dict[str, list[int]],
                curated: dict[str, dict], *, val_frac: float = 0.2,
                test_frac: float = 0.3, seed: int = 42,
                max_wild: int | None = None) -> SplitPlan:
    """Assign every contract to train/val/test, with de-dup + firewall. Pure.

    ``wild_paths``: {cid: path}; ``wild_labels``: {cid: [5]};
    ``curated``: {cid: {"path", "y", "lines"}}.
    """
    wild_sources = {cid: _read(p) for cid, p in wild_paths.items()}
    curated_sources = {cid: _read(v["path"]) for cid, v in curated.items()}
    kept, _removed, n_removed = dedup_wild_against_curated(wild_sources, curated_sources)
    kept = [c for c in sorted(kept) if c in wild_labels]   # must have a label
    if max_wild is not None:
        kept = kept[:max_wild]

    # Wild -> train/val (stratified on its weak labels).
    if kept:
        Yw = np.array([wild_labels[c] for c in kept], dtype=int)
        wtr, wva = stratified_multilabel_split(Yw, test_frac=val_frac, seed=seed)
    else:
        wtr, wva = np.array([], int), np.array([], int)
    wild_train = [kept[i] for i in wtr]
    wild_val = [kept[i] for i in wva]

    # Curated -> frozen test split + remainder (stratified on expert labels).
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


def materialise(plan: SplitPlan, extract_fn, embedder, out_dir: str, *,
                embed_dim: int = 64, seed: int = 42, extract_timeout: float = 120,
                embed_batch: int = 128) -> dict:
    """Extract, fit train-only artefacts, encode every split, write indices.

    ``extract_fn(path) -> (ast RawGraph, cfg RawGraph)``;
    ``embedder`` has ``.embed(snippet) -> np.ndarray``. Needs torch + the
    extraction stack; run on the Studio.
    """
    import joblib
    import torch

    out = Path(out_dir)
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    # Pass 1 — extract every contract (cached), tolerating per-contract failures
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
                    ast = cfg = None        # partial/corrupt cache (e.g. Ctrl-C) -> re-extract
            if ast is None:
                with _time_limit(extract_timeout):
                    ast, cfg = extract_fn(it.path)
                tmp = cache.with_name(cache.name + ".tmp")   # atomic write -> safe to interrupt
                tmp.write_text(json.dumps({"ast": ast.to_dict(), "cfg": cfg.to_dict()}),
                               encoding="utf-8")
                tmp.replace(cache)
            raws[it.cid] = (ast, cfg)
        except Exception as exc:               # skip & log; never guess
            failed.append((it.cid, str(exc)))
        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"  extracted {i + 1}/{n} (ok {len(raws)}, failed {len(failed)})", flush=True)
    print(f"pass 1 done: {len(raws)}/{n} extracted, {len(failed)} failed", flush=True)

    # Fit on TRAIN only.
    print("fitting train-only feature vocab + PCA ...", flush=True)
    train_ids = [it.cid for it in plan.items if it.split == "train" and it.cid in raws]
    train_graphs = [g for cid in train_ids for g in raws[cid]]
    train_snippets = sorted({s for cid in train_ids for g in raws[cid] for s in g.snippets})
    print(f"  embedding {len(train_snippets)} unique train snippets (batch {embed_batch}) ...",
          flush=True)
    emb = (embedder.embed_many(train_snippets, batch_size=embed_batch)
           if train_snippets else np.zeros((0, 768), np.float32))
    # Clamp PCA dims for small (e.g. --max-wild smoke) runs.
    k = int(min(embed_dim, emb.shape[0] or embed_dim, emb.shape[1] or embed_dim))
    feat_cfg = fit_feature_config(train_graphs, embed_dim=k)
    pca = fit_pca(emb, embed_dim=k, seed=seed) if emb.shape[0] >= k and k > 0 else None
    encoder = FeatureEncoder(feat_cfg, embedder if pca is not None else None, pca)

    # Pass 2 — encode every split and write records + indices.
    print("encoding all splits -> records ...", flush=True)
    indices: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
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
    for split in ("train", "val", "test"):
        (out / f"{split}_index.json").write_text(json.dumps(indices[split], indent=2),
                                                 encoding="utf-8")
    gold = {it.cid: it.gold_lines for it in plan.items
            if it.split == "test" and it.gold_lines}
    (out / "curated_gold_lines.json").write_text(json.dumps(gold, indent=2), encoding="utf-8")
    report = {"counts": plan.counts, "encoded": {k2: len(v) for k2, v in indices.items()},
              "failed": failed, "in_dim": feat_cfg.in_dim, "embed_dim": k}
    (out / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
