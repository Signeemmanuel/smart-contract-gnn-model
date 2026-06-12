"""GNNExplainer over the dual-graph model, one flaw class and one branch at a time.

Status: needs torch + torch_geometric; py_compiled here, validated on the Studio.
This is the fiddliest module in the repo, so the design is explicit:

* The model is multi-label, but PyG's Explainer has no multi-label mode, so we
  wrap it to expose a SINGLE class logit and explain it as binary classification
  (the agreed implementation fix).
* PyG's Explainer expects one ``(x, edge_index)``, but the model consumes two
  graphs. So we explain EACH branch separately: when explaining the AST, the CFG
  branch is run once and its pooled vector held fixed (detached); the explainer
  then perturbs only AST nodes/edges. Symmetric for the CFG branch. We union the
  per-branch line attributions. This is the agreed dual-graph handling.
* Nodes that map to no source line are logged via ``localise.nodes_to_lines``.
"""

from __future__ import annotations

from scgnn.explain.localise import rank_unique


def _branch_explainer(model, branch: str, fixed_other_pooled, class_idx: int):
    import torch
    import torch.nn as nn

    class BranchModel(nn.Module):
        """Single-graph view of the dual model for the chosen branch/class."""

        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x, edge_index, batch=None, **kwargs):
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            enc = self.model.ast if branch == "ast" else self.model.cfg
            pooled = enc(x, edge_index, batch)
            if branch == "ast":
                h = torch.cat([pooled, fixed_other_pooled], dim=1)
            else:
                h = torch.cat([fixed_other_pooled, pooled], dim=1)
            logit = self.model.head(h)[:, class_idx]
            return logit.unsqueeze(-1)  # (batch, 1) -> binary task

    return BranchModel()


def explain_branch(model, this_data, other_pooled, branch, class_idx,
                   node_lines, k=5, epochs=200):
    """Top-k influential lines for one branch and one class. Needs torch/PyG."""
    import torch
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.explain.config import ModelConfig

    bm = _branch_explainer(model, branch, other_pooled, class_idx)
    explainer = Explainer(
        model=bm,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(mode="binary_classification",
                                 task_level="graph", return_type="raw"),
    )
    expl = explainer(this_data.x, this_data.edge_index)
    importance = expl.node_mask.sum(dim=1)
    top = torch.topk(importance, k=min(k, importance.numel())).indices.tolist()
    from scgnn.explain.localise import nodes_to_lines
    return nodes_to_lines(top, node_lines)


def explain_lines(model, ast_data, cfg_data, class_idx, ast_lines, cfg_lines,
                  k=5, epochs=200):
    """Union of AST- and CFG-branch line attributions for one flaw class.

    Returns ``(lines, unmapped)`` where ``unmapped`` aggregates nodes (across
    both branches) that had no source line.
    """
    import torch

    model.eval()
    with torch.no_grad():
        batch_ast = torch.zeros(ast_data.x.size(0), dtype=torch.long)
        batch_cfg = torch.zeros(cfg_data.x.size(0), dtype=torch.long)
        cfg_pooled = model.cfg(cfg_data.x, cfg_data.edge_index, batch_cfg)
        ast_pooled = model.ast(ast_data.x, ast_data.edge_index, batch_ast)

    ast_l, ast_un = explain_branch(model, ast_data, cfg_pooled, "ast", class_idx, ast_lines, k, epochs)
    cfg_l, cfg_un = explain_branch(model, cfg_data, ast_pooled, "cfg", class_idx, cfg_lines, k, epochs)
    return rank_unique(ast_l + cfg_l), (ast_un + cfg_un)
