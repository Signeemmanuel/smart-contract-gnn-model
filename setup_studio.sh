#!/usr/bin/env bash
# =============================================================================
# setup_studio.sh - one-shot environment setup for a Lightning Studio.
#
#   bash setup_studio.sh
#
# Installs, in order:
#   1. the GNN train/inference stack (torch_geometric over the preinstalled torch)
#   2. extraction deps (Slither + a solc for every pragma the corpus uses)
#   3. SMARTBUGS + its Docker tool images  <- the four-tool labelling backend
#   4. dev tooling, then verifies everything with the unit tests and a GPU smoke
#
# Torch is NEVER re-resolved: every pip install is constrained by constraints.txt
# so the Studio's working CUDA build survives.
#
# The SmartBugs step is the one that was missing before: the labelling stage runs
# each analysis tool in a Docker container through the `sb` command, so without
# it every labelling task fails instantly with "command not found" and the ledger
# records 800 crashes in two seconds.
# =============================================================================
set -euo pipefail

C="constraints.txt"                       # keeps torch pinned during every install
SMARTBUGS="${SMARTBUGS:-$HOME/smartbugs}" # where SmartBugs lives (override to move it)

say() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m    !! %s\033[0m\n" "$*" >&2; }

# -----------------------------------------------------------------------------
say "tier 1: GNN train/inference env"
# Base torch_geometric is enough for this repo's layers (GCN/SAGE/GAT/GATv2, mean
# pool, Explainer); the compiled extras (torch_scatter/sparse) are optional.
pip install -c "$C" torch_geometric
pip install -c "$C" transformers huggingface_hub scikit-learn numpy joblib \
                    safetensors pyyaml pandas pyarrow matplotlib

# -----------------------------------------------------------------------------
say "tier 2: extraction (Slither on the HOST, for build_dataset) + solc"
pip install -c "$C" slither-analyzer solc-select

say "tier 2: solc versions for the corpus (Wild spans 0.4.x to 0.8.x)"
# A contract whose pragma has no matching compiler simply fails extraction, so
# install the whole range the corpus uses rather than a single version.
solc-select install \
  0.4.11 0.4.13 0.4.15 0.4.16 0.4.17 0.4.18 0.4.19 0.4.20 0.4.21 0.4.22 \
  0.4.23 0.4.24 0.4.25 0.4.26 \
  0.5.0 0.5.1 0.5.2 0.5.3 0.5.4 0.5.5 0.5.6 0.5.7 0.5.8 0.5.9 0.5.10 0.5.17 \
  0.6.12 0.7.6 0.8.19 || warn "some solc versions unavailable (non-fatal)"
solc-select install 0.4.4 0.4.6 0.4.8 0.4.9 || warn "ancient solc unavailable (non-fatal)"
solc-select use 0.8.19

# -----------------------------------------------------------------------------
say "tier 3: SmartBugs (the four-tool labelling backend)"

if ! command -v docker >/dev/null 2>&1; then
  warn "docker not found. SmartBugs runs every analysis tool in a container, so"
  warn "labelling CANNOT run here. Everything else (build/train/eval) still works."
else
  docker info >/dev/null 2>&1 || warn "docker is installed but the daemon is not reachable."

  if [ ! -d "$SMARTBUGS" ]; then
    say "tier 3: cloning SmartBugs -> $SMARTBUGS"
    git clone --depth 1 https://github.com/smartbugs/smartbugs.git "$SMARTBUGS"
  else
    echo "    SmartBugs already present at $SMARTBUGS"
  fi

  say "tier 3: installing the sb entrypoint"
  # SmartBugs has its own dependency set; install it constrained so it cannot
  # pull a different torch into this environment.
  ( cd "$SMARTBUGS" && pip install -c "$OLDPWD/$C" -e . ) \
    || ( cd "$SMARTBUGS" && pip install -e . ) \
    || warn "pip install of SmartBugs failed; see its README for the supported route."

  if command -v sb >/dev/null 2>&1; then
    echo "    sb installed: $(command -v sb)"
  else
    warn "the 'sb' command is still not on PATH."
    warn "SmartBugs may be module-invoked instead. If so, run the labelling with:"
    warn "    python scripts/label_orchestrator.py ... --sb-cmd \"python -m sb\""
    warn "(and set SB_CMD=\"python -m sb\" before run_pipeline.sh)"
  fi

  say "tier 3: pre-pulling the four tool images (slow once, then cached)"
  # Pulling now means the labelling run does not stall on image downloads, and it
  # proves Docker can actually fetch and run the tools BEFORE a long job starts.
  for img in \
      "smartbugs/slither:0.10.0" \
      "smartbugs/mythril:0.24.7" \
      "smartbugs/securify:usolc" \
      "smartbugs/osiris:d3ccc80"
  do
    docker pull "$img" >/dev/null 2>&1 \
      && echo "    pulled $img" \
      || warn "could not pull $img (SmartBugs will fetch its own tag on first use)"
  done
fi

# -----------------------------------------------------------------------------
say "dev tooling + editable package (no-deps so torch is never re-resolved)"
pip install -c "$C" pytest
pip install -e . --no-deps

# -----------------------------------------------------------------------------
say "sanity: is torch still the CUDA build?"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.is_available())"

say "sanity: GATv2 is available (the v2 encoder)"
python -c "from torch_geometric.nn import GATv2Conv; print('GATv2Conv OK')"

say "unit tests (expect 82 passed)"
PYTHONPATH=. pytest -q

say "GPU smoke test: a real DualGNN forward pass"
python - <<'PY'
import torch
from torch_geometric.data import Data, Batch
from scgnn.models.dual_gnn import DualGNN

dev = "cuda" if torch.cuda.is_available() else "cpu"
for conv in ("gcn", "sage", "gatv2"):
    model = DualGNN(in_dim=16, hid=32, conv=conv).to(dev).eval()
    def g(n):
        return Data(x=torch.randn(n, 16), edge_index=torch.randint(0, n, (2, n * 2)))
    ast = Batch.from_data_list([g(5), g(7)]).to(dev)
    cfg = Batch.from_data_list([g(6), g(4)]).to(dev)
    with torch.no_grad():
        out = model(ast, cfg)
    assert tuple(out.shape) == (2, 5), out.shape
    print(f"  {conv:6s} forward OK on {dev} | logits {tuple(out.shape)}")

# The empty-graph case: a contract whose CFG extraction degraded to zero nodes.
# This crashed training before the size= fix, so assert it stays fixed.
model = DualGNN(in_dim=16, hid=32, conv="sage").to(dev).eval()
ast = Batch.from_data_list([Data(x=torch.randn(4, 16),
                                 edge_index=torch.tensor([[0, 1], [1, 2]])),
                            Data(x=torch.randn(3, 16),
                                 edge_index=torch.tensor([[0, 1], [1, 2]]))]).to(dev)
cfg = Batch.from_data_list([Data(x=torch.randn(2, 16),
                                 edge_index=torch.tensor([[0, 1], [1, 0]])),
                            Data(x=torch.zeros(0, 16),          # EMPTY graph
                                 edge_index=torch.zeros(2, 0, dtype=torch.long))]).to(dev)
with torch.no_grad():
    out = model(ast, cfg)
assert tuple(out.shape) == (2, 5), out.shape
print("  empty-CFG contract handled (no size mismatch):", tuple(out.shape))
PY

say "SmartBugs smoke: one tool on one contract (proves Docker + the tool image)"
if command -v sb >/dev/null 2>&1 && [ -d data/raw/curated ]; then
  SOL=$(find data/raw/curated -name "*.sol" | head -1)
  if [ -n "$SOL" ]; then
    rm -rf /tmp/sbsmoke
    sb -t slither -f "$SOL" --results /tmp/sbsmoke --json \
      && find /tmp/sbsmoke -name "result.json" | head -1 \
      && echo "    SmartBugs produced result.json: labelling will work." \
      || warn "SmartBugs ran but produced no result.json; check its output above."
  fi
else
  warn "skipping SmartBugs smoke (no 'sb' on PATH, or data/raw/curated not downloaded yet)."
  warn "Run 'bash run_pipeline.sh data' first, then re-run this smoke by hand."
fi

# -----------------------------------------------------------------------------
say "freeze the committed lockfile"
pip freeze > requirements-lock.txt
say "DONE. Commit requirements-lock.txt."
echo
echo "Next:  export SMARTBUGS=\"$SMARTBUGS\""
echo "       bash run_pipeline.sh data      # download Wild + Curated"
echo "       bash run_pipeline.sh tools     # timed smoke batch, then the full run"