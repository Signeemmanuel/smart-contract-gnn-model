"""Parse SmartBugs' normalised tool output into per-flaw label matrices.

SmartBugs writes two JSON files per contract per tool: ``smartbugs.json`` is the
run metadata (cfg.TASK_LOG), and ``result.json`` is the parsed, normalised
findings (cfg.PARSER_OUTPUT) — uniform across tools, each finding carrying a
detector ``name`` (e.g. Slither ``reentrancy-eth``) and sometimes a ``swc-id``
(Mythril). We read ``result.json`` and map those identifiers via ``map_dasp``;
tool/contract identity comes from the results path:

    results/<tool-id>/<runid>/<contract>/result.json

(If a run was done without ``--json``, generate the parsed output with
SmartBugs' ``reparse`` first.)

Convention for each (contract, tool, flaw) vote:
    1   tool ran and reported this flaw
    0   tool ran and did not report this flaw
   -1   tool abstained (timed out, crashed, or does not cover this flaw)

``parse_smartbugs_result`` (an older normalised-format reader) is retained for
``training/evaluate/baselines.py`` but is not used by ``collect_votes``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from scgnn.schema import FLAW_INDEX, FLAWS  # noqa: F401  (FLAW_INDEX re-exported)
from training.labelling.map_dasp import TOOL_COVERAGE, TOOL_MAPS, map_finding  # noqa: F401

ABSTAIN = -1
TOOLS = ["slither", "mythril", "securify", "osiris"]
PARSER_OUTPUT = "result.json"   # SmartBugs' normalised findings (cfg.PARSER_OUTPUT)


def mythril_finding_ids(data: dict) -> list[str]:
    """Normalised SWC ids from Mythril's native JSON (``{"issues": [...]}``).

    Mythril emits the bare number under ``swc-id`` (e.g. ``"107"``); we prefix
    ``SWC-`` so it matches ``MYTHRIL_SWC``. Pure; order-preserving. Retained as a
    validated cross-check of the SWC vocabulary; the pipeline reads the parsed
    ``result.json`` below.
    """
    ids: list[str] = []
    for issue in data.get("issues", []) or []:
        if not isinstance(issue, dict):
            continue
        swc = str(issue.get("swc-id", "")).strip()
        if swc:
            ids.append(swc if swc.upper().startswith("SWC-") else f"SWC-{swc}")
    return ids


def parse_mythril_issues(data: dict) -> set[str]:
    """Map Mythril's native JSON to the set of flaw codes it reported. Pure."""
    flaws: set[str] = set()
    for rid in mythril_finding_ids(data):
        code = map_finding("mythril", rid)
        if code:
            flaws.add(code)
    return flaws


_SWC_IN_TEXT = re.compile(r"SWC[\s\-_]*0*(\d{1,4})", re.IGNORECASE)


def finding_identifiers(finding: dict) -> list[str]:
    """Candidate ids for one parsed finding.

    Yields the detector ``name`` (Slither's ``reentrancy-eth`` etc.) and any SWC
    id we can recover: a ``swc-id`` field if present, and — for Mythril, whose
    parsed ``name`` is a title with the SWC embedded, e.g. "Unchecked return
    value from external call. (SWC 104)" — the number parsed out of the text,
    normalised to ``SWC-104``.
    """
    ids: list[str] = []
    name = str(finding.get("name", "")).strip()
    if name:
        ids.append(name)
    swc = str(finding.get("swc-id", finding.get("swc_id", ""))).strip()
    if swc:
        ids.append(swc if swc.upper().startswith("SWC-") else f"SWC-{swc}")
    for m in _SWC_IN_TEXT.findall(name):       # SWC embedded in the title
        ids.append(f"SWC-{m}")
    return ids


def parser_output_flaws(tool: str, data: dict) -> set[str] | None:
    """Map a SmartBugs ``result.json`` (PARSER_OUTPUT) to flaw codes.

    A parsed result always carries a ``findings`` list (possibly empty = a clean
    verdict). Its absence means no parsed output was produced for this tool/
    contract -> ``None`` (abstain), so a crash is never counted as 'clean'.
    """
    findings = data.get("findings")
    if findings is None:
        return None
    flaws: set[str] = set()
    for f in findings:
        idents = finding_identifiers(f) if isinstance(f, dict) else [str(f)]
        for ident in idents:
            code = map_finding(tool, ident)
            if code:
                flaws.add(code)
    return flaws


def collect_votes(results_dir: str | Path) -> dict[str, dict[str, set[str]]]:
    """Walk a SmartBugs results tree into ``votes[contract][tool] = flaw set``.

    Reads the normalised ``result.json`` per ``<tool>/<runid>/<contract>/``.
    Tools with no parsed output for a contract (missing/unparseable/failed) are
    left absent, which ``build_label_matrices`` reads as abstain.
    """
    votes: dict[str, dict[str, set[str]]] = {}
    root = Path(results_dir)
    for pj in root.rglob(PARSER_OUTPUT):
        contract = Path(pj.parent.name).stem                        # <contract> dir -> stem (drop .sol)
        tool = pj.parent.parent.parent.name.split("-")[0].lower()   # <tool-id> -> tool
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except Exception:
            continue                                                # unparseable -> abstain
        flaws = parser_output_flaws(tool, data)
        if flaws is None:
            continue                                                # abstain
        votes.setdefault(contract, {})[tool] = flaws
    return votes


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
                continue  # tool didn't run -> abstain (-1)
            covers = TOOL_COVERAGE.get(tool, set())
            for flaw in FLAWS:
                if flaw not in covers:
                    continue  # tool is not a detector for this flaw -> abstain (-1)
                mats[flaw][i, j] = 1 if flaw in reported else 0
    return mats
