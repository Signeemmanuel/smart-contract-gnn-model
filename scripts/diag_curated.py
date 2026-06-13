#!/usr/bin/env python3
"""Throwaway: show the real cause of each failed extraction + the crytic AST API.

Run on the Studio:  PYTHONPATH=. python scripts/diag_curated.py
Paste the whole output back. Safe to delete afterwards.
"""
from __future__ import annotations

import re
import subprocess
import traceback
from pathlib import Path

from scgnn.extraction.solc import installed_solc_binaries, solc_for_file

FAILED = [
    "data/raw/curated/dataset/arithmetic/overflow_simple_add.sol",
    "data/raw/curated/dataset/access_control/parity_wallet_bug_1.sol",
    "data/raw/curated/dataset/denial_of_service/send_loop.sol",
    "data/raw/curated/dataset/unchecked_low_level_calls/unchecked_return_value.sol",
    "data/raw/curated/dataset/reentrancy/reentrancy_bonus.sol",
    "data/raw/curated/dataset/reentrancy/reentrancy_cross_function.sol",
    "0x89c1b3807d4c67df034fffb62f3509561218d30b",   # wild; resolved below
]


def resolve(p: str) -> str | None:
    if Path(p).exists():
        return p
    stem = Path(p).name
    if not stem.endswith(".sol"):
        stem += ".sol"
    for hit in Path("data/raw").rglob(stem):
        return str(hit)
    return None


def tail(n: int = 1600) -> str:
    return traceback.format_exc().strip()[-n:]


def main() -> int:
    bins = installed_solc_binaries()
    print("installed solc binaries:", {k: Path(v).name for k, v in bins.items()})

    for raw in FAILED:
        p = resolve(raw)
        print("\n" + "=" * 72)
        print(raw, "->", p)
        if not p:
            print("  NOT FOUND on disk")
            continue
        head = Path(p).read_text(encoding="utf-8", errors="ignore")[:400]
        m = re.search(r"pragma solidity[^;]*;", head)
        print("  pragma:", m.group(0) if m else "NONE")
        sb = solc_for_file(p, bins)
        print("  resolved solc:", Path(sb).name if sb else None)

        # 1) raw solc AST call — show the ACTUAL stderr (CalledProcessError hides it)
        if sb:
            r = subprocess.run([sb, "--ast-compact-json", p],
                               capture_output=True, text=True)
            print("  [raw solc --ast-compact-json] returncode:", r.returncode)
            if r.returncode != 0:
                print("  solc stderr:\n   ", (r.stderr or "").strip()[:900].replace("\n", "\n    "))

        # 2) our CFG path — full traceback if it raises (the empty-error mode)
        try:
            from scgnn.extraction.slither_cfg import extract_cfg
            g = extract_cfg(p, solc_binary=sb)
            print(f"  [extract_cfg] ok: {g.n_nodes} nodes, {len(g.edges)} edges")
        except Exception:
            print("  [extract_cfg] RAISED:\n   ", tail().replace("\n", "\n    "))

    # 3) crytic-compile AST API shape on THIS install (for the single-pass fix)
    print("\n" + "=" * 72)
    print("crytic-compile AST API probe:")
    probe = resolve("data/raw/curated/dataset/reentrancy/reentrancy_bonus.sol")
    try:
        from slither import Slither
        sl = Slither(probe, solc=solc_for_file(probe, bins))
        cc = sl.crytic_compile
        print("  CryticCompile has compilation_units:", hasattr(cc, "compilation_units"))
        for uid, unit in cc.compilation_units.items():
            srcattr = "source_units" if hasattr(unit, "source_units") else (
                "_source_units" if hasattr(unit, "_source_units") else None)
            print("  unit:", uid, "| source attr:", srcattr,
                  "| has .asts:", hasattr(cc, "asts"))
            sus = getattr(unit, "source_units", None) or getattr(unit, "_source_units", None)
            if sus:
                for fn, su in sus.items():
                    print("    source_unit:", Path(str(fn)).name,
                          "| ast attr type:", type(getattr(su, "ast", None)).__name__,
                          "| ast keys:", list(getattr(su, "ast", {}) or {})[:6])
                    break
            break
    except Exception:
        print("  probe RAISED:\n   ", tail().replace("\n", "\n    "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())