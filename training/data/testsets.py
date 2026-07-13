"""Freeze and enforce the two test sets (Workstream B + non-negotiable #4).

Two frozen benchmarks, each committed as a CSV manifest with a content-hash
ledger so train/val can be firewalled against BOTH, and so the exact test
contracts are reproducible for the dissertation:

  * Test Set A - tool-labelled internal benchmark. Drawn FIRST from freshly
    labelled Wild contracts (labels.parquet), stratified for rare-class
    coverage, firewalled against Curated. Measures agreement with the four-tool
    union oracle (comparable in kind to how BugSweeper/ReVulDL are scored).
  * Test Set B - expert external benchmark. The SmartBugs Curated contracts
    (via load_curated), with line annotations kept for localisation. The real
    cross-distribution generalisation benchmark.

Frozen manifest schema (CSV): contract_id, path, chash, <5 label columns>.
Once written, a manifest is immutable input: build/train read it, never rewrite
it. ``firewall_hashes`` returns the union of both sets' hashes so the split
excludes every test contract (and all Curated provenance) from train/val.

Pure except for file IO; the selection logic is unit-tested.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scgnn.schema import FLAWS, N_FLAWS
from training.data.firewall import content_hash, stratified_multilabel_split

MANIFEST_COLUMNS = ["contract_id", "path", "chash", *FLAWS]


def _read_source(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


# --------------------------- Test Set A (tool-labelled) ---------------------------

def select_test_a(
    labels: dict[str, list[int]],
    paths: dict[str, str],
    *,
    exclude_hashes: set[str],
    min_per_class: int = 60,
    target_per_class: int = 100,
    neg_ratio: float = 2.0,
    seed: int = 42,
) -> list[dict]:
    """Select Test Set A from freshly labelled Wild contracts.

    ``labels``: cid -> [5] union labels; ``paths``: cid -> .sol path.
    ``exclude_hashes``: content hashes already reserved (Curated / any prior
    split) - such contracts are never eligible for Test A.

    Stratified so each class gets up to ``target_per_class`` positives (at least
    ``min_per_class`` where the pool allows), plus negatives at roughly
    ``neg_ratio`` per positive contract. Deterministic given ``seed``.

    Returns a list of manifest rows (dicts with MANIFEST_COLUMNS keys). Pure.
    """
    rng = np.random.default_rng(seed)

    # eligible = has a path, readable, not excluded by hash
    eligible: list[str] = []
    chash: dict[str, str] = {}
    for cid, y in labels.items():
        p = paths.get(cid)
        if not p:
            continue
        try:
            h = content_hash(_read_source(p))
        except OSError:
            continue
        if h in exclude_hashes:
            continue
        eligible.append(cid)
        chash[cid] = h

    chosen: set[str] = set()
    # rarest class first so scarce positives (dos) are allocated before commoner
    # classes consume the budget.
    counts = {j: sum(labels[c][j] for c in eligible) for j in range(N_FLAWS)}
    for j in sorted(range(N_FLAWS), key=lambda k: counts[k]):
        pos = [c for c in eligible if labels[c][j] == 1 and c not in chosen]
        rng.shuffle(pos)
        take = pos[:target_per_class]
        chosen.update(take)

    # negatives: contracts with an all-zero label, up to neg_ratio x positives
    n_pos = len(chosen)
    negs = [c for c in eligible if sum(labels[c]) == 0 and c not in chosen]
    rng.shuffle(negs)
    chosen.update(negs[: int(round(neg_ratio * n_pos))])

    rows = []
    for cid in sorted(chosen):
        row = {"contract_id": cid, "path": paths[cid], "chash": chash[cid]}
        for j, f in enumerate(FLAWS):
            row[f] = int(labels[cid][j])
        rows.append(row)
    return rows


# --------------------------- Test Set B (expert Curated) ---------------------------

def select_test_b(curated: dict[str, dict]) -> tuple[list[dict], dict]:
    """Build Test Set B rows from load_curated output, plus an exclusion report.

    ``curated``: cid -> {"path","y","lines"} (already DASP-mapped to our five).
    Contracts whose only signal is out-of-scope map to all-zero and are KEPT as
    negatives (curated provenance must stay out of train regardless). Returns
    (rows, report) where report counts in-scope positives per class. Pure.
    """
    rows = []
    per_class = {f: 0 for f in FLAWS}
    for cid in sorted(curated):
        rec = curated[cid]
        try:
            h = content_hash(_read_source(rec["path"]))
        except OSError:
            continue
        row = {"contract_id": cid, "path": rec["path"], "chash": h}
        for j, f in enumerate(FLAWS):
            v = int(rec["y"][j])
            row[f] = v
            per_class[f] += v
        rows.append(row)
    report = {"contracts": len(rows), "positives_per_class": per_class}
    return rows, report


# ------------------------------- manifest IO -------------------------------

def write_manifest(rows: list[dict], path: str | Path) -> None:
    """Write a frozen manifest CSV. Idempotent given identical rows."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in MANIFEST_COLUMNS})


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def manifest_hashes(path: str | Path) -> set[str]:
    """Content hashes of every contract in a frozen manifest."""
    return {r["chash"] for r in read_manifest(path)}


def firewall_hashes(*manifest_paths: str | Path) -> set[str]:
    """Union of content hashes across the given manifests.

    Pass Test A and Test B manifests: the returned set is what train/val must
    exclude so no test contract (nor any Curated provenance) can leak in.
    """
    out: set[str] = set()
    for mp in manifest_paths:
        if Path(mp).exists():
            out |= manifest_hashes(mp)
    return out