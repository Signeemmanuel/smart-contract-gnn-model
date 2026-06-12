"""AST extraction to a :class:`RawGraph`.

Status: the pure assembler ``ast_from_compact_json`` is unit-tested; the
``extract_ast`` driver shells out to ``solc`` and must be validated on the
Studio (it needs ``solc``/``solc-select`` matching each contract's pragma).

The compact AST (``solc --ast-compact-json``) gives each node a ``nodeType`` and
a ``src`` field of the form ``"start:length:fileIndex"``. We walk it depth-first,
recording the node type, depth, child count, the snippet, and the line span, and
add parent->child edges.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from scgnn.extraction.graph_types import RawGraph, byte_offset_to_line


def _children(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Yield child AST dict-nodes of a compact-AST node, order-stable."""
    kids: list[dict[str, Any]] = []
    for value in node.values():
        if isinstance(value, dict) and "nodeType" in value:
            kids.append(value)
        elif isinstance(value, list):
            kids.extend(v for v in value if isinstance(v, dict) and "nodeType" in v)
    return kids


def ast_from_compact_json(ast_json: dict[str, Any], source: str) -> RawGraph:
    """Build a :class:`RawGraph` from a compact-AST root node. Pure; testable."""
    src_bytes = source.encode("utf-8")
    g = RawGraph(view="ast")
    index: dict[int, int] = {}

    def visit(node: dict[str, Any], depth: int) -> int:
        nid = g.n_nodes
        index[id(node)] = nid
        g.node_types.append(str(node.get("nodeType", "Unknown")))
        start, length = _parse_src(node.get("src", "0:0:0"))
        snippet = src_bytes[start : start + length].decode("utf-8", "ignore")
        g.snippets.append(snippet)
        first = byte_offset_to_line(src_bytes, start)
        last = byte_offset_to_line(src_bytes, start + max(length - 1, 0))
        g.node_lines[nid] = list(range(first, last + 1))
        kids = _children(node)
        g.depths.append(depth)
        g.n_children.append(len(kids))
        for kid in kids:
            cid = visit(kid, depth + 1)
            g.edges.append((nid, cid))
        return nid

    visit(ast_json, 0)
    return g


def _parse_src(src: str) -> tuple[int, int]:
    parts = src.split(":")
    start = int(parts[0]) if parts and parts[0] else 0
    length = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    return start, length


def extract_ast(path: str, solc_binary: str = "solc") -> RawGraph:
    """Compile ``path`` with solc and assemble its AST graph.

    Needs a solc matching the contract pragma on PATH (use solc-select). Raises
    on compilation failure; the caller should skip and log such contracts.
    """
    out = subprocess.run(
        [solc_binary, "--ast-compact-json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    brace = out.find("{")
    if brace < 0:
        raise ValueError(f"no AST JSON in solc output for {path}")
    ast_json = json.loads(out[brace:])
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        source = fh.read()
    return ast_from_compact_json(ast_json, source)
