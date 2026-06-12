"""GNN model definitions (shipped to the back end)."""

from scgnn.models.dual_gnn import DualGNN, build_model
from scgnn.models.encoders import CONV, Encoder

__all__ = ["DualGNN", "build_model", "Encoder", "CONV"]
