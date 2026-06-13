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


def normalise(importance: list[float]) -> list[float]:
    """Scale per-node importances into ``[0, 1]`` by the branch maximum.

    Each branch's GNNExplainer mask lives on its own scale, so before lines from
    the AST and CFG branches can be compared they must be put on a common one.
    A non-positive maximum (degenerate mask) maps everything to ``0``.
    """
    mx = max(importance) if importance else 0.0
    if mx <= 0:
        return [0.0 for _ in importance]
    return [v / mx for v in importance]


def line_scores_from_importance(
    importance: list[float],
    node_lines: dict[int, list[int]],
    unmapped_top: int = 10,
) -> tuple[dict[int, float], list[int]]:
    """Turn per-node importances into per-line normalised scores for one branch.

    A line's score is the **maximum** normalised importance of any node that
    maps to it. Returns ``(line_scores, unmapped)`` where ``unmapped`` lists the
    nodes among the ``unmapped_top`` most influential that carry no source line
    (a diagnostic for a broken node-to-lines map, not silent guessing).
    """
    norm = normalise(list(importance))
    scores: dict[int, float] = {}
    for nid, s in enumerate(norm):
        for ln in node_lines.get(nid, []):
            if s > scores.get(ln, 0.0):
                scores[ln] = s
    order = sorted(range(len(norm)), key=lambda i: norm[i], reverse=True)
    unmapped = [nid for nid in order[:unmapped_top] if not node_lines.get(nid)]
    return scores, unmapped


def merge_line_scores(*score_dicts: dict[int, float], k: int = 5) -> list[int]:
    """Merge per-branch line-score maps and rank globally.

    A line's merged score is the maximum it achieves in any branch, so a line
    the CFG branch localises strongly is no longer buried beneath every AST
    line. Ranked by score descending, ties broken by line number ascending for
    determinism; returns at most ``k`` lines.
    """
    merged: dict[int, float] = {}
    for d in score_dicts:
        for ln, s in d.items():
            if s > merged.get(ln, 0.0):
                merged[ln] = s
    ranked = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))
    return [ln for ln, _ in ranked[:k]]
