"""Load the BIT smartcontract-benchmark as additional expert test contracts.

Source: github.com/bit-smartcontract-analysis/smartcontract-benchmark

Layout (same shape as SmartBugs Curated, which is why this is nearly a drop-in):
    <root>/small_dataset/dataset/<category>/<contract>.sol      (110 contracts)
    <root>/small_dataset/vulnerabilities.json                  (gold lines)
    <root>/labelled-dataset/...                                (389 contracts)

Each contract carries the same annotation convention as Curated:
    /* ... @vulnerable_at_lines: 30 */
    // <yes> <report> unsafe delegatecall

So this loader returns the SAME structure as training.data.curated.load_curated
-> ``{cid: {"path", "y", "lines"}}`` and merges into the test pool identically.

BIT's taxonomy is FINER than our five DASP classes, so several BIT folders fold
into one of ours; the rest are dropped (-> all-zero negatives), exactly as
curated.py treats out-of-scope folders:

    reentrancy            -> reentrancy
    arithmetic            -> arithmetic
    unchecked_send        -> unchecked_calls
    unsafe_delegatecall   -> access_control   (DASP-2)
    unsafe_suicide        -> access_control   (unprotected SELFDESTRUCT)
    tx_origin             -> access_control   (tx.origin auth)
    gasless_send          -> dos              (failed/gasless external call)
    TOD                   -> (drop: front-running, out of scope)
    time_manipulation     -> (drop)
    bad_randomness        -> (drop)
    safecontract          -> (kept as all-zero NEGATIVE; useful true negatives)

Note on `dos`: BIT's `gasless_send` is the insufficient-gas/failed-send DoS
subtype specifically — a narrower notion than the broad DoS our DIVE training
labels use. Mapping it to `dos` is defensible (it is a DoS pattern) but the test
`dos` cases are predominantly this subtype; state that in the write-up.
"""
from __future__ import annotations

import re
from pathlib import Path

from scgnn.schema import FLAW_INDEX, N_FLAWS

# BIT category folder -> our flaw code. Folders not listed map to nothing
# (-> all-zero negative). 'safecontract' is intentionally absent: its contracts
# are negatives, which is exactly an all-zero label, so we keep them.
BIT_CATEGORY_MAP: dict[str, str] = {
    "reentrancy": "reentrancy",
    "arithmetic": "arithmetic",
    "unchecked_send": "unchecked_calls",
    "unsafe_delegatecall": "access_control",
    "unsafe_suicide": "access_control",
    "tx_origin": "access_control",
    "gasless_send": "dos",
    # dropped (out of our five): TOD, time_manipulation, bad_randomness
    # kept as negatives: safecontract
}

# Folders we explicitly recognise as in-scope-or-negative, so an UNEXPECTED new
# folder name is reported rather than silently treated as a negative.
KNOWN_FOLDERS = set(BIT_CATEGORY_MAP) | {
    "tod", "time_manipulation", "bad_randomness", "safecontract",
}

_VULN_LINES_RE = re.compile(r"@vulnerable_at_lines:\s*([0-9,\s]+)", re.IGNORECASE)
_REPORT_RE = re.compile(r"<yes>\s*<report>\s*([A-Za-z_ ]+)")

# Inline report-tag text -> our flaw, for the cross-check / multi-flaw files.
# Tag text mirrors the folder names but may contain spaces (e.g. "unsafe delegatecall").
_TAG_TO_FLAW = {
    "reentrancy": "reentrancy",
    "arithmetic": "arithmetic",
    "unchecked_send": "unchecked_calls",
    "unchecked send": "unchecked_calls",
    "unsafe_delegatecall": "access_control",
    "unsafe delegatecall": "access_control",
    "unsafe_suicide": "access_control",
    "unsafe suicidal": "access_control",
    "unsafe suicide": "access_control",
    "tx_origin": "access_control",
    "tx.origin": "access_control",
    "gasless_send": "dos",
    "gasless send": "dos",
}


def parse_vulnerable_lines(source: str) -> list[int]:
    lines: list[int] = []
    for m in _VULN_LINES_RE.finditer(source):
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if tok.isdigit():
                lines.append(int(tok))
    return sorted(set(lines))


def report_tag_flaws(source: str) -> set[str]:
    out: set[str] = set()
    for m in _REPORT_RE.finditer(source):
        key = m.group(1).strip().lower().rstrip()
        flaw = _TAG_TO_FLAW.get(key)
        if flaw:
            out.add(flaw)
    return out


def _dataset_dirs(root: Path) -> list[Path]:
    """Find the category-folder parent dir(s): small_dataset and/or labelled-dataset."""
    candidates: list[Path] = []
    for sub in ("small_dataset/dataset", "labelled-dataset/dataset",
                "labelled-dataset", "small_dataset"):
        p = root / sub
        if p.is_dir() and any(c.is_dir() for c in p.iterdir()):
            # only accept if it actually holds category folders with .sol inside
            if any(p.rglob("*.sol")):
                candidates.append(p)
    # de-dup nested (prefer the .../dataset level)
    out = []
    for c in candidates:
        if not any(c != o and str(c).startswith(str(o)) for o in candidates):
            out.append(c)
    return out or [root]


def load_bit(bit_root: str | Path, *, include_safe: bool = True) -> dict[str, dict]:
    """Load BIT benchmark contracts -> ``{cid: {"path", "y", "lines"}}``.

    Labels come from the category folder (folded to our five) with the inline
    ``<yes> <report>`` tag as a cross-check; gold lines from ``@vulnerable_at_lines``.
    cid is prefixed ``bit_`` so it never collides with Curated/DIVE/SWC stems.
    Set ``include_safe=False`` to drop the explicit safe-contract negatives.
    """
    root = Path(bit_root)
    unknown: set[str] = set()
    out: dict[str, dict] = {}

    for ds in _dataset_dirs(root):
        for sol in sorted(ds.rglob("*.sol")):
            if "__MACOSX" in sol.parts or sol.name.startswith("._"):
                continue
            category = sol.parent.name.lower()
            if category not in KNOWN_FOLDERS:
                unknown.add(category)

            folder_flaw = BIT_CATEGORY_MAP.get(category)
            if category == "safecontract" and not include_safe:
                continue

            try:
                source = sol.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            flaws: set[str] = set()
            if folder_flaw:
                flaws.add(folder_flaw)
            flaws |= report_tag_flaws(source)   # catches multi-flaw / confirms

            y = [0] * N_FLAWS
            for f in flaws:
                y[FLAW_INDEX[f]] = 1
            gold = parse_vulnerable_lines(source) if any(y) else []

            cid = f"bit_{category}_{sol.stem}".lower()
            out[cid] = {"path": str(sol), "y": y, "lines": gold}

    if unknown:
        print(f"  [load_bit] NOTE: unrecognised category folders treated as negatives: "
              f"{sorted(unknown)}")
    return out