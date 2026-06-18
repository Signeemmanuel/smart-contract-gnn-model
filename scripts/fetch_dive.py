#!/usr/bin/env python3
"""Fetch the DIVE dataset from Zenodo and report its structure.

Purpose
-------
DIVE's labels are keyed by an integer ``contractID`` while this project keys
contracts by file stem / content hash. Before a loader can be written, we must
know *how* ``contractID`` in ``DIVE_Labels.csv`` joins to the actual ``.sol``
source files. This script downloads the dataset, unpacks the parts we need, and
prints the exact facts required to settle that join — it does not assume any
layout.

What it does
------------
1. Resolves the DIVE Zenodo record (DOI 10.5281/zenodo.18519253) via the Zenodo
   API and lists its files (names + sizes), so you can see what is actually
   published.
2. Downloads the label + raw-source archives (skippable / selectable).
3. Extracts them and prints:
     * the directory tree (top levels),
     * the header + first rows of every CSV it finds (esp. DIVE_Labels.csv and
       Code-based.csv — the likely join key),
     * how the .sol source files are named/organised (a sample listing),
     * a first attempt to match a few contractIDs to source files, reporting
       which join strategy works.

Usage
-----
    # just look at what's in the record (no big download):
    python scripts/fetch_dive.py --list-only

    # download labels + raw source, extract, and report (the usual run):
    python scripts/fetch_dive.py --out data/raw/dive

    # if you already downloaded the zips yourself, point at them and only report:
    python scripts/fetch_dive.py --out data/raw/dive --report-only

Notes
-----
* Network: needs access to zenodo.org. If your shell is behind the restricted
  proxy, run this on a host that can reach Zenodo (gpu-01 should be fine).
* Size: the raw archive is large (tens of thousands of contracts). Use
  --skip-raw to fetch only labels first if you just want to see the CSV schema.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD_ID = "18519253"            # from DOI 10.5281/zenodo.18519253
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Archives we care about (names per the paper's Table 6; matched case-insensitively
# and by substring, so minor naming differences still resolve).
WANTED = {
    "labels": ["dive_labels", "labels"],
    "raw": ["dive_raw_data", "raw_data", "raw"],
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "scgnn-dive-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def resolve_record() -> dict:
    print(f"resolving Zenodo record {ZENODO_RECORD_ID} ...")
    try:
        meta = json.loads(_get(ZENODO_API).decode("utf-8"))
    except Exception as exc:
        sys.exit(f"ERROR: could not reach Zenodo API ({exc}).\n"
                 f"If this host can't reach zenodo.org, run on one that can, or "
                 f"download manually from https://doi.org/10.5281/zenodo.18519253")
    return meta


def list_files(meta: dict) -> list[dict]:
    files = meta.get("files", [])
    title = meta.get("metadata", {}).get("title", "?")
    print(f"\nrecord: {title}")
    print(f"files ({len(files)}):")
    out = []
    for f in files:
        key = f.get("key") or f.get("filename") or "?"
        size = f.get("size") or f.get("filesize") or 0
        link = (f.get("links", {}) or {}).get("self") or f.get("download") or ""
        out.append({"key": key, "size": size, "url": link})
        print(f"  - {key}  ({size/1e6:.1f} MB)")
    return out


def pick(files: list[dict], needles: list[str]) -> dict | None:
    for f in files:
        name = f["key"].lower()
        if any(n in name for n in needles):
            return f
    return None


def download(f: dict, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {dest.name} ({dest.stat().st_size/1e6:.1f} MB), skipping download")
        return dest
    print(f"  downloading {f['key']} -> {dest} ...")
    data = _get(f["url"])
    dest.write_bytes(data)
    print(f"    saved {len(data)/1e6:.1f} MB")
    return dest


def unzip(path: Path, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {path.name} -> {into} ...")
    with zipfile.ZipFile(path) as z:
        z.extractall(into)
    return into


def tree(root: Path, max_depth: int = 2, max_entries: int = 30) -> None:
    print(f"\n=== directory tree under {root} (depth {max_depth}) ===")
    root = Path(root)
    shown = 0
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if len(rel.parts) > max_depth:
            continue
        shown += 1
        if shown > max_entries:
            print("   ... (truncated)")
            break
        indent = "  " * (len(rel.parts) - 1)
        tag = "/" if p.is_dir() else ""
        size = "" if p.is_dir() else f"  ({p.stat().st_size} B)"
        print(f"  {indent}{rel.parts[-1]}{tag}{size}")


def show_csv_head(path: Path, n: int = 5) -> list[str]:
    print(f"\n=== CSV: {path} ===")
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
            reader = csv.reader(fh)
            rows = []
            for i, row in enumerate(reader):
                rows.append(row)
                if i >= n:
                    break
        if not rows:
            print("  (empty)")
            return []
        header = rows[0]
        print(f"  header ({len(header)} cols): {header}")
        for r in rows[1:n + 1]:
            print(f"  {r}")
        return header
    except Exception as exc:
        print(f"  could not read: {exc}")
        return []


def sample_sol(root: Path, k: int = 10) -> list[Path]:
    sols = list(root.rglob("*.sol"))
    print(f"\n=== .sol source files under {root} ===")
    print(f"  total .sol found: {len(sols)}")
    if not sols:
        print("  (no .sol files — source may be inside a CSV/JSONL column instead)")
        # look for jsonl that might hold source
        for j in list(root.rglob("*.jsonl"))[:3]:
            print(f"  found JSONL (may hold source): {j.relative_to(root)}")
        return []
    print("  sample names + parent folders:")
    for p in sols[:k]:
        print(f"    {p.relative_to(root)}")
    # report the naming pattern of the stems
    stems = [p.stem for p in sols[:200]]
    numeric = sum(s.isdigit() for s in stems)
    hexlike = sum(s.lower().startswith("0x") for s in stems)
    print(f"  stem pattern (first {len(stems)}): numeric={numeric}, "
          f"0x-address={hexlike}, other={len(stems) - numeric - hexlike}")
    return sols


def try_join(labels_csv: Path, sol_root: Path) -> None:
    """Attempt to match the first few contractIDs to source files; report which works."""
    print("\n=== join probe: contractID -> .sol file ===")
    try:
        with labels_csv.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh)
            ids = []
            for i, row in enumerate(reader):
                ids.append(row.get("contractID") or row.get("contractAddress")
                           or list(row.values())[0])
                if i >= 9:
                    break
    except Exception as exc:
        print(f"  could not read labels CSV: {exc}")
        return

    sols = {p.stem: p for p in sol_root.rglob("*.sol")}
    sols_lower = {k.lower(): v for k, v in sols.items()}
    print(f"  first contractIDs: {ids}")

    # Strategy A: <id>.sol by stem
    a = sum(str(i) in sols for i in ids)
    # Strategy B: 0x-prefixed / address-style (id is an address)
    b = sum(str(i).lower() in sols_lower for i in ids)
    print(f"  match as '<contractID>.sol' (stem == id): {a}/{len(ids)}")
    print(f"  match as address stem (id == file stem, any case): {b}/{len(ids)}")
    if a == len(ids):
        print("  => JOIN: source files are named '<contractID>.sol'. Direct join works.")
    elif b == len(ids):
        print("  => JOIN: source files are named by the id/address directly.")
    else:
        print("  => JOIN UNCLEAR: contractID does NOT directly name the .sol files.\n"
              "     There must be a mapping column (likely in Code-based.csv) linking\n"
              "     contractID to ContractName/address. Paste Code-based.csv's header\n"
              "     and a couple of rows so the loader can join through it.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/dive", help="Where to download + extract.")
    ap.add_argument("--list-only", action="store_true",
                    help="Only list the Zenodo record's files; download nothing.")
    ap.add_argument("--report-only", action="store_true",
                    help="Skip download; just inspect what's already under --out.")
    ap.add_argument("--skip-raw", action="store_true",
                    help="Download only the labels archive (fast; to see the CSV schema).")
    args = ap.parse_args()

    out = Path(args.out)

    if not args.report_only:
        meta = resolve_record()
        files = list_files(meta)
        if args.list_only:
            return 0

        labels_f = pick(files, WANTED["labels"])
        raw_f = pick(files, WANTED["raw"])
        if not labels_f:
            print("WARNING: could not identify a labels archive by name; "
                  "see the file list above and download manually.")
        if labels_f:
            lp = download(labels_f, out / labels_f["key"])
            if lp.suffix == ".zip":
                unzip(lp, out / "labels")
        if raw_f and not args.skip_raw:
            rp = download(raw_f, out / raw_f["key"])
            if rp.suffix == ".zip":
                unzip(rp, out / "raw")
        elif args.skip_raw:
            print("  (--skip-raw: not downloading the raw source archive)")

    # ---- report whatever is present under --out ----
    tree(out, max_depth=2)

    # find and show every CSV (esp. labels + code-based)
    csvs = list(out.rglob("*.csv"))
    labels_csv = None
    for c in csvs:
        header = show_csv_head(c, n=5)
        if "contractID" in header or "Reentrancy" in header:
            labels_csv = c

    # show how source is organised
    sol_root = (out / "raw") if (out / "raw").exists() else out
    sols = sample_sol(sol_root)

    # probe the join
    if labels_csv and sols:
        try_join(labels_csv, sol_root)
    elif labels_csv and not sols:
        print("\n=== join probe skipped: no .sol files found yet ===")
        print("  Run without --skip-raw to fetch the source archive, or paste the\n"
              "  Code-based.csv header so the loader can resolve source another way.")

    print("\nDONE. Paste this script's output back so the DIVE loader can be written\n"
          "to match the real layout (esp. the join between contractID and .sol files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())