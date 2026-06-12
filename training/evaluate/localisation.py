"""Top-k localisation accuracy. Pure; unit-tested.

Accuracy@k = fraction of flawed test contracts for which at least one of the top
k predicted lines matches an expert-marked flawed line.
"""

from __future__ import annotations


def top_k_localisation(
    pred_lines: list[list[int]], gold_lines: list[set[int]], ks=(1, 3, 5)
) -> dict[int, float]:
    """``pred_lines[i]`` is one flawed contract's ranked lines; ``gold_lines[i]``
    is its expert-marked set. Returns ``{k: accuracy}``."""
    if len(pred_lines) != len(gold_lines):
        raise ValueError("pred_lines and gold_lines must be the same length")
    n = len(pred_lines)
    out: dict[int, float] = {}
    for k in ks:
        if n == 0:
            out[k] = 0.0
            continue
        hits = sum(1 for pred, gold in zip(pred_lines, gold_lines)
                   if gold and set(pred[:k]) & gold)
        out[k] = hits / n
    return out
