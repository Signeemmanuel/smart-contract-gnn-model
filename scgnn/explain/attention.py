"""GAT first-layer attention as a cheap secondary explanation signal.

Status: needs torch; py_compiled here. Ranks edges by attention weight, maps the
incident nodes to lines, and returns a ranked unique line list. Use only when the
selected model is GAT (the encoder exposes ``last_attention`` after a forward).
"""

from __future__ import annotations

from scgnn.explain.localise import nodes_to_lines


def attention_lines(encoder, node_lines: dict[int, list[int]], k: int = 5):
    """Return ``(lines, unmapped)`` from the encoder's first-layer attention.

    ``encoder.last_attention`` is the ``(edge_index, alpha)`` tuple captured by
    the GAT first layer during the forward pass.
    """
    if getattr(encoder, "last_attention", None) is None:
        return [], []
    import torch

    edge_index, alpha = encoder.last_attention
    score = alpha.mean(dim=1) if alpha.dim() > 1 else alpha  # average over heads
    top = torch.topk(score, k=min(k, score.numel())).indices.tolist()
    ranked_nodes: list[int] = []
    seen: set[int] = set()
    for e in top:
        for node in (int(edge_index[0, e]), int(edge_index[1, e])):
            if node not in seen:
                seen.add(node)
                ranked_nodes.append(node)
    return nodes_to_lines(ranked_nodes, node_lines)
