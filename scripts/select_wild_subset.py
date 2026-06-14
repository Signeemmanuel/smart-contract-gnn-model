#!/usr/bin/env python3
"""Select a class-balanced Wild subset for the four-tool labelling run.

We never label all 47k contracts. Instead we pick ~N that are *dense in positives
for every class*, because the smoke model's real defect was empty/sparse classes
(arithmetic, dos), not size. Selection uses cheap source-text signatures only to
BIAS which contracts we pick — the Slither/Mythril/Securify/Osiris ensemble then
assigns the authoritative labels. So this is candidate selection, not labelling.

Signatures (heuristic, deliberately over-inclusive):
  reentrancy   low-level value call: .call.value(... / .call{value: ...
  unchecked    low-level call/send:  .call( / .send(
  access       selfdestruct / suicide / tx.origin / delegatecall
  dos          a loop AND an external call (.send/.transfer/.call)
  arithmetic   pre-0.8 pragma (no built-in overflow checks) AND arithmetic ops
               — Osiris confirms the real positives within this candidate pool

Selected files are copied (not symlinked — Docker mounts need real files) into
--subset-dir, ready for:
  python -m sb -t slither mythril securify osiris -f '<subset-dir>/*.sol' ...

Example:
  PYTHONPATH=. python scripts/select_wild_subset.py \
    --wild-dir data/raw/wild --curated-dir data/raw/curated \
    --n 2500 --subset-dir data/raw/wild_subset
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from training.data.firewall import content_hash
from training.labelling.run_slither import pragma_minor

# --- signature patterns (case-insensitive) -----------------------------------
_RE_VALUE_CALL = re.compile(r"\.call\.value\s*\(|\.call\s*\{[^}]*value\s*:|\.call\.gas", re.I)
_RE_LOWLEVEL = re.compile(r"\.call\s*[\({]|\.send\s*\(", re.I)
_RE_ACCESS = re.compile(r"\bselfdestruct\b|\bsuicide\b|tx\.origin|\.?delegatecall\s*\(", re.I)
_RE_LOOP = re.compile(r"\bfor\s*\(|\bwhile\s*\(", re.I)
_RE_EXT_CALL = re.compile(r"\.send\s*\(|\.transfer\s*\(|\.call\s*[\({]", re.I)
_RE_ARITH_OPS = re.compile(r"\+\+|--|[+\-*]=|\bSafeMath\b|\w\s*[+\-*]\s*\w", re.I)

CLASSES = ("reentrancy", "access_control", "arithmetic", "unchecked_calls", "dos")
# Selection quotas as fractions of N (sum > 1 because contracts overlap classes;
# rare classes first so their scarce candidates are allocated before the budget
# fills). Tuned to lift the classes the smoke model starved.
QUOTA_FRAC = {
    "arithmetic": 0.28, "dos": 0.18, "access_control": 0.18,
    "reentrancy": 0.20, "unchecked_calls": 0.20,
}


def _minor_lt_0_8(minor: str | None) -> bool:
    """True for pragmas < 0.8 (no built-in overflow checks)."""
    if not minor:
        return False
    try:
        major, mn = (int(x) for x in minor.split("."))
    except ValueError:
        return False
    return (major, mn) < (0, 8)


def is_collectable(source: str) -> bool:
    """True iff SmartBugs can assign a solc to this contract.

    Every tool in the ensemble (Slither, Mythril, Securify, Osiris) compiles the
    contract, so a file with no resolvable ``pragma solidity`` cannot be analysed
    by any of them — and SmartBugs aborts the *whole batch* at task collection if
    even one such file is present. We therefore exclude them at selection time;
    they are raw on-chain sources missing the pragma line, not usable labels.
    """
    return pragma_minor(source) is not None


def detect_signals(source: str) -> set[str]:
    """Heuristic candidate classes a contract's source suggests. Pure."""
    sig: set[str] = set()
    if _RE_VALUE_CALL.search(source):
        sig.add("reentrancy")
    if _RE_LOWLEVEL.search(source):
        sig.add("unchecked_calls")
    if _RE_ACCESS.search(source):
        sig.add("access_control")
    if _RE_LOOP.search(source) and _RE_EXT_CALL.search(source):
        sig.add("dos")
    if _minor_lt_0_8(pragma_minor(source)) and _RE_ARITH_OPS.search(source):
        sig.add("arithmetic")
    return sig


def balanced_select(signals_by_id: dict[str, set[str]], n: int, seed: int = 42) -> list[str]:
    """Pick ``n`` ids, filling per-class quotas (rarest first) then background.

    Deterministic given ``seed``. A contract counts toward every class it
    signals, but is selected once; the background fill brings the total to ``n``.
    """
    import random

    rng = random.Random(seed)
    chosen: set[str] = set()
    quotas = {c: max(1, int(round(QUOTA_FRAC[c] * n))) for c in CLASSES}
    # rarest class (fewest candidates) first
    by_scarcity = sorted(
        CLASSES, key=lambda c: sum(1 for s in signals_by_id.values() if c in s))
    for c in by_scarcity:
        if len(chosen) >= n:
            break
        cands = [i for i, s in signals_by_id.items() if c in s and i not in chosen]
        rng.shuffle(cands)
        take = min(quotas[c], n - len(chosen), len(cands))
        chosen.update(cands[:take])
    if len(chosen) < n:                       # background fill from the rest
        rest = [i for i in signals_by_id if i not in chosen]
        rng.shuffle(rest)
        chosen.update(rest[: n - len(chosen)])
    return sorted(chosen)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wild-dir", default="data/raw/wild")
    ap.add_argument("--curated-dir", default="data/raw/curated")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--subset-dir", default="data/raw/wild_subset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    curated_hashes = {
        content_hash(_read(p)) for p in Path(args.curated_dir).rglob("*.sol")}
    print(f"curated contracts hashed for firewall: {len(curated_hashes)}")

    wild = sorted(Path(args.wild_dir).rglob("*.sol"))
    print(f"scanning {len(wild)} Wild contracts ...")
    signals_by_id: dict[str, set[str]] = {}
    path_by_id: dict[str, Path] = {}
    removed = 0
    no_pragma = 0
    sig_counts: Counter = Counter()
    for i, p in enumerate(wild, 1):
        if i % 5000 == 0:
            print(f"  {i}/{len(wild)}", flush=True)
        try:
            src = _read(p)
        except Exception:
            continue
        if content_hash(src) in curated_hashes:    # firewall: drop curated dupes
            removed += 1
            continue
        if not is_collectable(src):                 # no resolvable pragma -> SmartBugs can't analyse it
            no_pragma += 1
            continue
        sig = detect_signals(src)
        signals_by_id[p.stem] = sig
        path_by_id[p.stem] = p
        for c in sig:
            sig_counts[c] += 1

    print(f"deduped against curated: removed {removed}")
    print(f"excluded (no resolvable pragma, SmartBugs-uncollectable): {no_pragma}")
    print("candidate signal counts (a contract may signal several):")
    for c in CLASSES:
        print(f"  {c:<16} {sig_counts[c]}")

    chosen = balanced_select(signals_by_id, min(args.n, len(signals_by_id)), args.seed)
    chosen_sig = Counter()
    for cid in chosen:
        for c in signals_by_id[cid]:
            chosen_sig[c] += 1
    n_background = sum(1 for cid in chosen if not signals_by_id[cid])
    print(f"\nselected {len(chosen)} contracts:")
    for c in CLASSES:
        print(f"  {c:<16} candidates in subset: {chosen_sig[c]}")
    print(f"  background (no signal):          {n_background}")

    out = Path(args.subset_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.sol"):          # clear a prior selection so it can't linger
        stale.unlink()
    for cid in chosen:
        shutil.copy2(path_by_id[cid], out / path_by_id[cid].name)
    manifest = {
        "n": len(chosen), "seed": args.seed, "wild_dir": args.wild_dir,
        "removed_curated_dupes": removed,
        "excluded_no_pragma": no_pragma,
        "candidate_signal_counts": {c: sig_counts[c] for c in CLASSES},
        "selected_signal_counts": {c: chosen_sig[c] for c in CLASSES},
        "background": n_background,
        "contracts": {cid: sorted(signals_by_id[cid]) for cid in chosen},
    }
    (out.parent / "wild_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\ncopied to {out}/  (manifest: {out.parent / 'wild_subset_manifest.json'})")
    print(f"next: python -m sb -t slither mythril securify osiris "
          f"-f '{out}/*.sol' --processes 8 --mem-limit 8g --timeout 300 --json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
