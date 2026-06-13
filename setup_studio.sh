#!/usr/bin/env bash
# One-shot environment setup for a Lightning Studio L4 (torch 2.8.0+cu128 preinstalled).
# Run from the repo root:  bash setup_studio.sh
set -euo pipefail

C="constraints.txt"   # keeps torch pinned during every install

echo "==> tier 1: GNN train/inference env"
# Base torch_geometric is enough for this repo's layers (GCN/SAGE/GAT, mean pool,
# Explainer); the compiled extras (torch_scatter/sparse) are optional.
pip install -c "$C" torch_geometric
pip install -c "$C" transformers huggingface_hub scikit-learn numpy joblib safetensors pyyaml pandas pyarrow
pip install -c "$C" slither-analyzer solc-select
solc-select install 0.8.19 && solc-select use 0.8.19   # add other pragma versions as your contracts need

echo "==> tier 2: weak-supervision labelling (snorkel; torch stays pinned via -c)"
pip install -c "$C" snorkel || echo "!! snorkel reported issues — paste the output back to Claude"

echo "==> dev tooling + editable package (no-deps so torch is never re-resolved)"
pip install -c "$C" pytest
pip install -e . --no-deps

echo "==> sanity: is torch still the CUDA build?"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.is_available())"

echo "==> unit tests (expect 25 passed)"
PYTHONPATH=. pytest -q

echo "==> GPU smoke test: a real DualGNN forward pass on the L4"
python - <<'PY'
import torch
from torch_geometric.data import Data, Batch
from scgnn.models.dual_gnn import DualGNN

dev = "cuda"
model = DualGNN(in_dim=16, hid=32, conv="sage").to(dev).eval()
def g(n):
    return Data(x=torch.randn(n, 16), edge_index=torch.randint(0, n, (2, n * 2)))
ast = Batch.from_data_list([g(5), g(7)]).to(dev)
cfg = Batch.from_data_list([g(6), g(4)]).to(dev)
with torch.no_grad():
    out = model(ast, cfg)
print("forward OK on", torch.cuda.get_device_name(0), "| logits shape", tuple(out.shape), "(expect (2, 5))")
PY

echo "==> freeze the committed lockfile"
pip freeze > requirements-lock.txt
echo "DONE. Commit requirements-lock.txt."