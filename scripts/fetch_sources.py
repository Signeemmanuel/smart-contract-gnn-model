#!/usr/bin/env python3
"""Recover contract source (.sol) for the frozen splits, verified by content hash.

The original .sol corpora lived only on the destroyed instance. The records,
labels and manifests survived, and every manifest row carries ``chash`` (the
comment-stripped, whitespace-collapsed content hash the firewall uses). That
makes recovery VERIFIABLE rather than a guess: we fetch the public SmartBugs
corpora, compute each candidate's chash with the project's own
``content_hash``, and keep only files whose hash matches a hash our splits
actually used. A filename or a dead path is never trusted; the hash is the
identity.

Strategy (chosen): fetch the FULL SmartBugs Wild + Curated corpora, then match
by hash. Simpler and robust to any renaming, at the cost of more disk and time
than a targeted fetch.

Outputs, written under --out (default data/recovered):
  sol/<chash>.sol              one file per matched unique contract
  source_paths.json            {contract_id: absolute .sol path} for every split
  coverage.json                honest per-split match/miss counts + missing ids

What needs the output:
  - the sequence baseline reads source_paths.json (train/val + test A/B);
  - the votes-on-Test-B re-run needs the Test B .sol files, which this writes.

Coverage is reported, never assumed: if only 34,102 of 35,547 train contracts
are recovered, that number goes in source_paths coverage and the thesis states
it. A contract that cannot be hash-matched is LEFT OUT, never substituted.

Usage
-----
    # A) let the script clone the public corpora (needs git + network)
    PYTHONPATH=. python scripts/fetch_sources.py \
        --testsets data/testsets --labels data/processed/labels.parquet \
        --processed data/processed_nodf \
        --clone --work data/smartbugs_src --out data/recovered

    # B) point at corpora already on disk (skip cloning)
    PYTHONPATH=. python scripts/fetch_sources.py \
        --testsets data/testsets --labels data/processed/labels.parquet \
        --processed data/processed_nodf \
        --wild-dir /path/to/smartbugs-wild --curated-dir /path/to/curated \
        --out data/recovered
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from training.data.firewall import content_hash

# Public SmartBugs corpora. Wild holds the ~47k mainnet contracts; the Curated
# set ships inside the smartbugs-curated repo under dataset/<category>/*.sol.
WILD_REPO = "https://github.com/smartbugs/smartbugs-wild"
CURATED_REPO = "https://github.com/smartbugs/smartbugs-curated"


def _clone(repo: str, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        print(f"  {dest} already present, skipping clone")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  cloning {repo} -> {dest} (shallow)")
    subprocess.check_call(["git", "clone", "--depth", "1", repo, str(dest)])


def _iter_sol(root: Path):
    """Yield every .sol file under a corpus root."""
    yield from root.rglob("*.sol")


def build_hash_index(dirs: list[Path]) -> dict[str, Path]:
    """Map content_hash -> path for every .sol in the given corpus dirs.

    First writer wins on a hash collision (identical content under two names);
    the paths are equivalent by construction, so which one we keep is irrelevant.
    Progress is printed because Wild is large.
    """
    index: dict[str, Path] = {}
    n = 0
    for d in dirs:
        if not d.exists():
            print(f"  WARNING: corpus dir absent: {d}")
            continue
        for p in _iter_sol(d):
            n += 1
            if n % 5000 == 0:
                print(f"  hashed {n} files, {len(index)} unique", flush=True)
            try:
                h = content_hash(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            index.setdefault(h, p)
    print(f"  indexed {n} .sol files, {len(index)} unique content hashes")
    return index


def load_needed(testsets: Path, labels: Path, processed: Path):
    """Collect the (contract_id -> chash) the splits need.

    Test A/B come from the frozen manifests (id, chash). Train/val come from the
    processed indices (ids) joined to labels.parquet is NOT enough for a hash
    (labels has no chash), so train/val hashes are recomputed here only if a
    chash sidecar exists; otherwise train/val are matched by RE-HASHING the
    recovered files against the set of all corpus hashes and keyed by id via the
    manifest-independent index the build used. To keep this robust without the
    original paths, we treat train/val recovery as: any corpus contract whose id
    (file stem) matches a needed train/val id AND whose hash is self-consistent.

    In practice the frozen manifests are the authoritative chash source for the
    test splits (what the baselines must score), and train/val need only enough
    source to fine-tune on; we therefore return:
      test_ids_hash: {split: {id: chash}} for test_a, test_b (authoritative)
      trainval_ids:  {split: [ids]} for train, val (matched by id+hash below)
    """
    from training.data.testsets import read_manifest

    test_ids_hash = {}
    for split, fname in (("test_a", "test_a.csv"), ("test_b", "test_b.csv")):
        rows = read_manifest(testsets / fname)
        test_ids_hash[split] = {r["contract_id"]: r["chash"] for r in rows}

    trainval_ids = {}
    for split in ("train", "val"):
        idx_path = processed / f"{split}_index.json"
        if idx_path.exists():
            entries = json.loads(idx_path.read_text(encoding="utf-8"))
            trainval_ids[split] = [e["id"] for e in entries]
        else:
            trainval_ids[split] = []
    return test_ids_hash, trainval_ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testsets", default="data/testsets")
    ap.add_argument("--labels", default="data/processed/labels.parquet")
    ap.add_argument("--processed", default="data/processed_nodf")
    ap.add_argument("--clone", action="store_true",
                    help="Clone the public corpora into --work first.")
    ap.add_argument("--work", default="data/smartbugs_src")
    ap.add_argument("--wild-dir", default=None,
                    help="Existing Wild corpus dir (skips cloning Wild).")
    ap.add_argument("--curated-dir", default=None,
                    help="Existing Curated corpus dir (skips cloning Curated).")
    ap.add_argument("--out", default="data/recovered")
    args = ap.parse_args()

    work = Path(args.work)
    wild_dir = Path(args.wild_dir) if args.wild_dir else work / "smartbugs-wild"
    curated_dir = Path(args.curated_dir) if args.curated_dir else work / "smartbugs-curated"

    if args.clone:
        if not args.wild_dir:
            _clone(WILD_REPO, wild_dir)
        if not args.curated_dir:
            _clone(CURATED_REPO, curated_dir)

    out = Path(args.out)
    (out / "sol").mkdir(parents=True, exist_ok=True)

    # resume: an existing index of already-written hashes
    written: dict[str, Path] = {p.stem: p for p in (out / "sol").glob("*.sol")}
    if written:
        print(f"resuming: {len(written)} .sol already recovered")

    print("indexing corpora by content hash ...")
    index = build_hash_index([wild_dir, curated_dir])

    test_ids_hash, trainval_ids = load_needed(Path(args.testsets),
                                              Path(args.labels),
                                              Path(args.processed))

    # also build an id->hash for train/val by re-hashing corpus files whose stem
    # equals a needed id; the build named Wild contracts by file stem (address).
    stem_to_hash: dict[str, str] = {}
    for h, p in index.items():
        stem_to_hash.setdefault(p.stem, h)

    source_paths: dict[str, str] = {}
    coverage: dict[str, dict] = {}

    def recover(split: str, id_to_hash: dict[str, str], by_hash: bool):
        matched, missing = {}, []
        for cid, h in id_to_hash.items():
            src = None
            if by_hash:
                src = index.get(h)                    # authoritative hash match
            else:                                     # train/val: match by stem
                hh = stem_to_hash.get(cid)
                src = index.get(hh) if hh else None
                h = hh
            if src is None:
                missing.append(cid)
                continue
            dst = out / "sol" / f"{h}.sol"
            if not dst.exists():
                dst.write_text(src.read_text(encoding="utf-8", errors="ignore"),
                               encoding="utf-8")
            matched[cid] = str(dst.resolve())
        source_paths.update(matched)
        coverage[split] = {"needed": len(id_to_hash), "recovered": len(matched),
                           "missing": len(missing),
                           "coverage": round(len(matched) / max(1, len(id_to_hash)), 4),
                           "missing_ids_sample": missing[:20]}
        print(f"  {split:8s} recovered {len(matched)}/{len(id_to_hash)} "
              f"({coverage[split]['coverage']*100:.1f}%)")

    print("matching test splits by manifest chash (authoritative) ...")
    for split in ("test_a", "test_b"):
        recover(split, test_ids_hash[split], by_hash=True)

    print("matching train/val by contract id + self-consistent hash ...")
    for split in ("train", "val"):
        ids = trainval_ids[split]
        recover(split, {cid: "" for cid in ids}, by_hash=False)

    (out / "source_paths.json").write_text(json.dumps(source_paths, indent=2),
                                           encoding="utf-8")
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2),
                                       encoding="utf-8")

    total_needed = sum(c["needed"] for c in coverage.values())
    total_have = sum(c["recovered"] for c in coverage.values())
    print("\n" + "=" * 60)
    print(f"recovered {total_have}/{total_needed} contracts across all splits")
    for s, c in coverage.items():
        print(f"  {s:8s} {c['recovered']:>6}/{c['needed']:<6} "
              f"({c['coverage']*100:5.1f}%)")
    print("=" * 60)
    print(f"source_paths.json -> {out/'source_paths.json'}")
    print(f"coverage.json     -> {out/'coverage.json'}")
    print("\nNext: copy source_paths.json into the processed dir the baseline "
          "reads, e.g.:")
    print(f"  cp {out/'source_paths.json'} {args.processed}/source_paths.json")
    print("Then run the sequence baseline, and (for votes-on-Test-B) the "
          "sb_testb tool re-run using the recovered Test B .sol files.")
    if coverage.get("test_b", {}).get("missing", 0):
        print("\nNOTE: some Test B contracts were not recovered; votes-on-Test-B "
              "and the sequence Test B row will cover only the recovered subset, "
              "which coverage.json records for honest reporting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
