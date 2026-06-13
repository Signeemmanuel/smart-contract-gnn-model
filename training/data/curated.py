"""Load the SmartBugs Curated set: gold labels + vulnerable line numbers.

The label for each contract is taken from its DASP **category folder**
(``dataset/<category>/<addr>.sol``), which is canonical, rather than from the
``category`` string in ``vulnerabilities.json`` (whose casing follows the
in-source report tags). The annotation file is used only for the gold line
numbers, which feed the localisation metric. The category mapping and the
JSON assembler are pure and unit-tested; ``load_curated`` walks a checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

from scgnn.schema import FLAW_INDEX, N_FLAWS

# SmartBugs Curated category folder -> our flaw code. Folders outside our five
# (bad_randomness, front_running, time_manipulation, short_addresses, other)
# map to nothing, so such contracts are honest negatives for our classes.
CATEGORY_MAP: dict[str, str] = {
    "reentrancy": "reentrancy",
    "access_control": "access_control",
    "arithmetic": "arithmetic",
    "unchecked_low_level_calls": "unchecked_calls",
    "denial_of_service": "dos",
}


def _entry_fields(entry: dict) -> tuple[str, list[int]]:
    """Pull (category, lines) from one annotation entry, casing-normalised."""
    category = str(entry.get("category") or entry.get("vulnerability") or "").lower()
    lines = entry.get("lines") or entry.get("line") or []
    if isinstance(lines, int):
        lines = [lines]
    return category, [int(x) for x in lines]


def parse_vulnerabilities(records: list[dict]) -> dict[str, dict]:
    """Assemble ``cid -> {"y": [5], "lines": [...]}`` from annotation records.

    Each record may carry a single category or a nested ``vulnerabilities`` list.
    ``y`` is set from in-scope categories; ``lines`` collects those categories'
    annotated lines. Pure; unit-tested.
    """
    out: dict[str, dict] = {}
    for rec in records:
        path = str(rec.get("path") or rec.get("name") or "")
        cid = Path(path).stem or path
        y = [0] * N_FLAWS
        lines: list[int] = []
        nested = rec.get("vulnerabilities")
        entries = nested if isinstance(nested, list) and nested else [rec]
        for entry in entries:
            category, ls = _entry_fields(entry)
            flaw = CATEGORY_MAP.get(category)
            if flaw:
                y[FLAW_INDEX[flaw]] = 1
                lines.extend(ls)
        out[cid] = {"y": y, "lines": sorted(set(lines))}
    return out


def load_curated(curated_root: str | Path) -> dict[str, dict]:
    """Load curated contracts -> ``{cid: {"path", "y", "lines"}}``.

    Labels come from the ``dataset/<category>/`` folder; gold lines come from
    the root ``vulnerabilities.json`` (matched by contract stem) if present.
    """
    root = Path(curated_root)
    gold: dict[str, dict] = {}
    vp = root / "vulnerabilities.json"
    if vp.exists():
        ann = json.loads(vp.read_text(encoding="utf-8"))
        records = ann if isinstance(ann, list) else ann.get("dataset", [])
        gold = parse_vulnerabilities(records)

    out: dict[str, dict] = {}
    for sol in root.glob("dataset/*/*.sol"):
        category = sol.parent.name
        flaw = CATEGORY_MAP.get(category)
        y = [0] * N_FLAWS
        if flaw:
            y[FLAW_INDEX[flaw]] = 1
        out[sol.stem] = {
            "path": str(sol),
            "y": y,
            "lines": gold.get(sol.stem, {}).get("lines", []),
        }
    return out
