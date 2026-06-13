"""Weak-label contracts by running Slither directly (no SmartBugs/Docker).

Slither covers four of the five flaws via ``SLITHER_MAP`` (no arithmetic detector,
so arithmetic stays 0 here — Curated supplies arithmetic positives). We run
Slither's DEFAULT detectors and filter the output, because detector ids change
between versions and one unknown ``--detect`` name rejects the whole run.

Compiler selection (the crucial part for a multi-version corpus) lives in the
shipped :mod:`scgnn.extraction.solc` helper and is re-exported here for the
labeller and its tests: each run is handed the exact ``--solc <binary>`` matching
the contract's pragma, since the ``solc`` on PATH may be a fixed compiler.
"""

from __future__ import annotations

import json
import os
import subprocess

# Canonical solc helpers live in the shipped package; re-exported for the
# labeller and tests/test_slither_label.py.
from scgnn.extraction.solc import (  # noqa: F401
    choose_solc, installed_solc_binaries, installed_solc_by_minor, pragma_minor,
)
from training.labelling.map_dasp import SLITHER_MAP, map_finding

MAPPED_DETECTORS = sorted(SLITHER_MAP)   # reference/diagnostics only


def parse_slither_detectors(data: dict) -> set[str]:
    """Map a Slither ``--json`` payload's detector checks to flaw codes. Pure."""
    flaws: set[str] = set()
    detectors = (data.get("results", {}) or {}).get("detectors", []) or []
    for det in detectors:
        flaw = map_finding("slither", str(det.get("check", "")))
        if flaw:
            flaws.add(flaw)
    return flaws


def run_slither(path: str, solc_binary: str | None = None,
                solc_version: str | None = None, timeout: int = 120) -> set[str] | None:
    """Run Slither (default detectors) on one contract -> flaw codes, or None.

    Prefers an explicit ``--solc <binary>`` (bypasses PATH/env entirely); falls
    back to the process-local ``SOLC_VERSION`` env var. Returns None on a missing
    Slither, timeout, compile failure, or unparseable output. Slither exits
    non-zero when it finds issues, so the return code is ignored.
    """
    cmd = ["slither", path, "--json", "-"]
    env = os.environ.copy()
    if solc_binary:
        cmd += ["--solc", solc_binary]
    elif solc_version:
        env["SOLC_VERSION"] = solc_version
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    if data.get("success") is False:
        return None
    return parse_slither_detectors(data)
