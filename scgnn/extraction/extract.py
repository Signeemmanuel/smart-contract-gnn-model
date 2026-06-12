"""Orchestrator: a Solidity file/source -> (AST RawGraph, CFG RawGraph).

Status: needs solc + Slither; validate on the Studio.
"""

from __future__ import annotations

from scgnn.extraction.graph_types import RawGraph
from scgnn.extraction.slither_ast import extract_ast
from scgnn.extraction.slither_cfg import extract_cfg


def extract_contract(path: str, solc_binary: str = "solc") -> tuple[RawGraph, RawGraph]:
    """Extract both views for one contract, keeping them aligned by contract."""
    ast = extract_ast(path, solc_binary=solc_binary)
    cfg = extract_cfg(path)
    return ast, cfg
