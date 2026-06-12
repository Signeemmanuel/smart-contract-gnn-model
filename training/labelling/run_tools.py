"""Parse normalised SmartBugs tool output into per-flaw label matrices.

Status: depends on the exact SmartBugs 2.0 result layout; validate on the Studio
against a few real result files first. The matrix assembler is straightforward
and the mapping is unit-tested separately.

Convention for each (contract, tool, flaw) vote:
    1   tool ran and reported this flaw
    0   tool ran and did not report this flaw
   -1   tool abstained (timed out, crashed, or does not cover this flaw)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scgnn.schema import FLAW_INDEX, FLAWS
from training.labelling.map_dasp import TOOL_MAPS, map_finding

ABSTAIN = -1
TOOLS = ["slither", "mythril", "securify"]


def parse_smartbugs_result(path: str | Path) -> tuple[str, str, set[str]]:
    """Read one SmartBugs result file -> (contract_id, tool, set_of_flaw_codes).

    Expects a JSON object exposing the tool name and a list of findings, each
    finding carrying an identifier under one of: ``check``, ``name``,
    ``swc-id``, ``id``. Adjust the field names here if your SmartBugs version
    differs.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tool = str(data.get("tool", data.get("analysis", ""))).lower()
    contract = str(data.get("contract", data.get("filename", Path(path).stem)))
    flaws: set[str] = set()
    for finding in data.get("findings", data.get("errors", []) or []):
        if isinstance(finding, dict):
            ident = (finding.get("check") or finding.get("name")
                     or finding.get("swc-id") or finding.get("id") or "")
        else:
            ident = str(finding)
        code = map_finding(tool, str(ident))
        if code:
            flaws.add(code)
    return contract, tool, flaws


def build_label_matrices(
    votes: dict[str, dict[str, set[str] | None]],
    contract_ids: list[str],
) -> dict[str, np.ndarray]:
    """Assemble one ``(n_contracts, n_tools)`` matrix per flaw.

    ``votes[contract][tool]`` is the set of flaw codes that tool reported, or
    ``None`` if the tool abstained on that contract.
    """
    n = len(contract_ids)
    mats = {flaw: np.full((n, len(TOOLS)), ABSTAIN, dtype=np.int8) for flaw in FLAWS}
    for i, cid in enumerate(contract_ids):
        per_tool = votes.get(cid, {})
        for j, tool in enumerate(TOOLS):
            reported = per_tool.get(tool)
            if reported is None:
                continue  # abstain stays -1
            for flaw in FLAWS:
                mats[flaw][i, j] = 1 if flaw in reported else 0
    return mats
