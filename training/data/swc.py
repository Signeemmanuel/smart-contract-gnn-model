"""Load SWC Registry test cases as additional expert test contracts.

The SWC Registry (github.com/SmartContractSecurity/SWC-registry) ships, under
``test_cases/<swc-id>/``, hand-authored Solidity contracts annotated in the SAME
convention SmartBugs Curated uses:

    /*
     * @source: ...
     * @vulnerable_at_lines: 17
     */
    ...
    // <yes> <report> REENTRANCY
    ...

This loader parses those annotations and returns the EXACT same structure as
``training.data.curated.load_curated`` -> ``{cid: {"path", "y", "lines"}}`` so
the two sets merge into one test pool with no special-casing downstream.

Mapping to our five flaws is done two ways, both consistent with curated.py's
philosophy (out-of-scope -> all-zero negative):
  * by the SWC id of the folder (e.g. SWC-107 -> reentrancy), and
  * by the inline ``<yes> <report> CATEGORY`` tag (DASP-style), as a cross-check
    and to catch multi-flaw files.

A contract is positive for a flaw if EITHER signal maps to it. Files whose only
signals are out-of-scope (bad randomness, front-running, etc.) become honest
all-zero negatives — useful true negatives for the test set. ``*_fixed`` /
non-vulnerable variants (no positive signal) are kept as negatives too.

Pure parsing helpers are unit-testable; ``load_swc`` walks a checkout.
"""
from __future__ import annotations

import re
from pathlib import Path

from scgnn.schema import FLAW_INDEX, N_FLAWS

# SWC id -> our flaw code. Only in-scope ids are listed; everything else maps to
# nothing (-> all-zero negative), exactly as curated.py drops out-of-scope folders.
SWC_TO_FLAW: dict[str, str] = {
    "SWC-107": "reentrancy",
    "SWC-101": "arithmetic",
    "SWC-104": "unchecked_calls",
    # access control family
    "SWC-105": "access_control",   # unprotected ether withdrawal
    "SWC-106": "access_control",   # unprotected SELFDESTRUCT
    "SWC-115": "access_control",   # authorization through tx.origin
    "SWC-118": "access_control",   # incorrect constructor name
    # denial of service family
    "SWC-113": "dos",              # DoS with failed call
    "SWC-128": "dos",              # DoS with block gas limit
}

# Inline ``<yes> <report> CATEGORY`` tag text -> our flaw code. These are the
# DASP-style category names that appear in the annotations; keep this aligned
# with curated.py's CATEGORY_MAP semantics (out-of-scope -> nothing).
REPORT_TAG_TO_FLAW: dict[str, str] = {
    "reentrancy": "reentrancy",
    "access_control": "access_control",
    "accesscontrol": "access_control",
    "arithmetic": "arithmetic",
    "integer_overflow": "arithmetic",
    "integer_underflow": "arithmetic",
    "unchecked_low_level_calls": "unchecked_calls",
    "unchecked_call": "unchecked_calls",
    "unchecked_return_value": "unchecked_calls",
    "denial_of_service": "dos",
    "dos": "dos",
}

_VULN_LINES_RE = re.compile(r"@vulnerable_at_lines:\s*([0-9,\s]+)", re.IGNORECASE)
_REPORT_RE = re.compile(r"<yes>\s*<report>\s*([A-Za-z_]+)")
_SWC_ID_RE = re.compile(r"(SWC-\d{3})", re.IGNORECASE)


def parse_vulnerable_lines(source: str) -> list[int]:
    """Collect every line number in the file's ``@vulnerable_at_lines`` headers."""
    lines: list[int] = []
    for m in _VULN_LINES_RE.finditer(source):
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if tok.isdigit():
                lines.append(int(tok))
    return sorted(set(lines))


def report_tag_flaws(source: str) -> set[str]:
    """Flaw codes implied by inline ``<yes> <report> CATEGORY`` tags (in-scope only)."""
    out: set[str] = set()
    for m in _REPORT_RE.finditer(source):
        flaw = REPORT_TAG_TO_FLAW.get(m.group(1).strip().lower())
        if flaw:
            out.add(flaw)
    return out


def swc_id_from_path(path: Path) -> str | None:
    """Pull an ``SWC-NNN`` id from a path component (the test_cases/<id>/ folder)."""
    for part in path.parts:
        m = _SWC_ID_RE.search(part)
        if m:
            return m.group(1).upper()
    return None


def labels_for_file(path: Path, source: str) -> tuple[list[int], list[int]]:
    """Return ``(y[5], gold_lines)`` for one SWC contract.

    Positive for a flaw if EITHER the folder's SWC id OR an inline report tag
    maps to it. Out-of-scope-only files are all-zero negatives. Pure.
    """
    y = [0] * N_FLAWS
    flaws: set[str] = set()

    swc_id = swc_id_from_path(path)
    folder_flaw = SWC_TO_FLAW.get(swc_id) if swc_id else None
    if folder_flaw:
        flaws.add(folder_flaw)
    flaws |= report_tag_flaws(source)

    for f in flaws:
        y[FLAW_INDEX[f]] = 1
    gold = parse_vulnerable_lines(source) if any(y) else []
    return y, gold


def load_swc(swc_root: str | Path, *, test_cases_subdir: str = "test_cases") -> dict[str, dict]:
    """Load SWC test-case contracts -> ``{cid: {"path", "y", "lines"}}``.

    ``swc_root`` is a checkout of the SWC-registry repo (or any tree containing
    ``test_cases/<SWC-id>/*.sol``). cid is prefixed ``swc_`` so ids never collide
    with Curated/DIVE stems. Mirrors ``load_curated``'s return shape exactly.
    """
    root = Path(swc_root)
    base = root / test_cases_subdir
    search_root = base if base.exists() else root

    out: dict[str, dict] = {}
    for sol in sorted(search_root.rglob("*.sol")):
        if "__MACOSX" in sol.parts or sol.name.startswith("._"):
            continue
        try:
            source = sol.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        y, gold = labels_for_file(sol, source)
        cid = f"swc_{swc_id_from_path(sol) or 'x'}_{sol.stem}".lower()
        out[cid] = {"path": str(sol), "y": y, "lines": gold}
    return out