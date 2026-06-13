"""Per-contract solc selection — the canonical helper for the whole project.

Most real-world corpora span many Solidity versions, and the ``solc`` on PATH is
often a single fixed compiler (e.g. a conda build) that is wrong for, and refuses
to compile, the majority of contracts. The reliable fix is to hand the exact
compiler binary to each tool per contract: ``solc <binary> --ast-compact-json``
for the AST and ``Slither(path, solc=<binary>)`` for the CFG.

These helpers discover the binaries installed by solc-select (under
``~/.solc-select/artifacts``) and pick the one whose ``major.minor`` matches a
contract's pragma. Everything here is pure / filesystem-only (no solc, no torch),
so it imports cheaply and is unit-testable.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_PRAGMA = re.compile(r"pragma\s+solidity\s+([^;]+);")
_VER = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def pragma_minor(source: str) -> str | None:
    """Extract the ``major.minor`` from a contract's solidity pragma. Pure."""
    m = _PRAGMA.search(source)
    if not m:
        return None
    v = _VER.search(m.group(1))
    return f"{v.group(1)}.{v.group(2)}" if v else None


def pragma_exact(source: str) -> str | None:
    """Return the pinned ``major.minor.patch`` iff the pragma is an exact pin.

    ``pragma solidity 0.4.25;`` (optionally a leading ``=``) is exact and must be
    compiled with that precise solc; ``^0.4.24``, ``~0.4.0`` or a ``>= <`` range
    are not exact (return None -> fall back to the minor's newest patch). Pure.
    """
    m = _PRAGMA.search(source)
    if not m:
        return None
    spec = m.group(1).strip()
    pin = re.fullmatch(r"=?\s*(\d+\.\d+\.\d+)", spec)
    return pin.group(1) if pin else None


def choose_solc(source: str, mapping: dict[str, str]) -> str | None:
    """Look up an entry (version string or binary path) for the pragma minor. Pure."""
    minor = pragma_minor(source)
    return mapping.get(minor) if minor else None


def installed_solc_binaries(base: str | Path | None = None) -> dict[str, str]:
    """Map ``major.minor`` -> path to the highest installed solc binary.

    Reads ``~/.solc-select/artifacts/solc-<ver>/solc-<ver>`` (the standard
    layout), falling back to any file in the version directory.
    """
    root = Path(base) if base else Path.home() / ".solc-select" / "artifacts"
    best: dict[str, tuple[tuple[int, int, int], str]] = {}
    if root.is_dir():
        for d in sorted(root.glob("solc-*")):
            m = re.match(r"solc-(\d+)\.(\d+)\.(\d+)$", d.name)
            if not m or not d.is_dir():
                continue
            ver = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            key = f"{m.group(1)}.{m.group(2)}"
            binp = d / d.name
            if not binp.exists():
                files = [p for p in d.iterdir() if p.is_file()]
                binp = files[0] if files else None
            if binp and binp.exists() and (key not in best or ver > best[key][0]):
                best[key] = (ver, str(binp))
    return {k: v[1] for k, v in best.items()}


def installed_solc_full(base: str | Path | None = None) -> dict[str, str]:
    """Map exact ``major.minor.patch`` -> binary path for every installed solc."""
    root = Path(base) if base else Path.home() / ".solc-select" / "artifacts"
    out: dict[str, str] = {}
    if root.is_dir():
        for d in sorted(root.glob("solc-*")):
            m = re.match(r"solc-(\d+\.\d+\.\d+)$", d.name)
            if not m or not d.is_dir():
                continue
            binp = d / d.name
            if not binp.exists():
                files = [p for p in d.iterdir() if p.is_file()]
                binp = files[0] if files else None
            if binp and binp.exists():
                out[m.group(1)] = str(binp)
    return out


def installed_solc_by_minor() -> dict[str, str]:
    """Map ``major.minor`` -> highest installed patch (version strings), via solc-select."""
    try:
        out = subprocess.run(["solc-select", "versions"], capture_output=True,
                             text=True, timeout=30).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    best: dict[str, tuple[int, int, int]] = {}
    for a, b, c in _VER.findall(out):
        key, ver = f"{a}.{b}", (int(a), int(b), int(c))
        if key not in best or ver > best[key]:
            best[key] = ver
    return {k: f"{v[0]}.{v[1]}.{v[2]}" for k, v in best.items()}


def solc_for_file(path: str | Path, binaries: dict[str, str],
                  full: dict[str, str] | None = None, head_bytes: int = 4096) -> str | None:
    """Resolve the solc binary for one contract by reading its pragma.

    An exact pin (``pragma solidity 0.4.25;``) must use that precise version, so
    if ``full`` (an exact ``installed_solc_full`` map) is given and contains the
    pinned version, it wins. Otherwise we fall back to the newest patch of the
    pragma's minor (``binaries``). Returns None when the file is unreadable, has
    no pragma, or no suitable solc is installed — callers skip or fall back.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            head = fh.read(head_bytes)
    except OSError:
        return None
    if full:
        exact = pragma_exact(head)
        if exact:
            return full.get(exact)        # exact pin: that version or None (never a wrong patch)
    return choose_solc(head, binaries)
