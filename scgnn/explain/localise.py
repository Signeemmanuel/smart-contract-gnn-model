"""Map influential graph nodes back to ranked, de-duplicated source lines.

Pure module (no torch); unit-tested. ``rank_unique`` preserves influence order
(most influential first) while removing repeats, matching the schema contract.
Nodes that carry no source line are recorded in ``unmapped`` rather than guessed
at, so a broken node-to-lines map shows up as missing evidence, not silent noise.
"""

from __future__ import annotations


def rank_unique(lines: list[int]) -> list[int]:
    """De-duplicate while preserving first-seen (highest-rank) order."""
    out: list[int] = []
    seen: set[int] = set()
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out


def nodes_to_lines(ranked_nodes: list[int], node_lines: dict[int, list[int]]) -> tuple[list[int], list[int]]:
    """Expand ranked node indices to ranked unique lines.

    Returns ``(lines, unmapped_nodes)`` where ``unmapped_nodes`` lists node
    indices that mapped to no source line at all.
    """
    lines: list[int] = []
    unmapped: list[int] = []
    for n in ranked_nodes:
        ls = node_lines.get(n, [])
        if not ls:
            unmapped.append(n)
            continue
        lines.extend(ls)
    return rank_unique(lines), unmapped
