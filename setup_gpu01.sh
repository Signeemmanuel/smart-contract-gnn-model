#!/usr/bin/env bash
# =============================================================================
# setup_gpu01.sh - one-shot environment setup for a RENTED GPU server.
#
#   bash setup_gpu01.sh
#
# Named for the original gpu-01 box; it now targets any rented Linux GPU server
# (Vast.ai, DataBaseMart, CloudClusters). It handles BOTH cases:
#
#   * torch already present (a preconfigured image): it is NEVER touched.
#     Re-installing torch risks pulling a build without support for the card's
#     compute capability, which breaks the GPU. Every install below avoids
#     dragging torch in, and torch is re-asserted at the end so a mistake is
#     caught immediately.
#   * torch absent (a bare Ubuntu VM): a CUDA build is installed for the card
#     detected by nvidia-smi.
#
# Covers everything the pipeline needs on one box:
#   - labelling        SmartBugs + Docker + the four tool images  <- the long stage
#   - extraction/build Slither + a solc for every pragma in the corpus
#   - training/eval    torch_geometric, transformers, sklearn, safetensors
#   - artifacts        matplotlib; tests
# =============================================================================
set -uo pipefail   # not -e: report failures, do not abort the whole script

SMARTBUGS="${SMARTBUGS:-$HOME/smartbugs}"

say()  { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m    !! %s\033[0m\n" "$*" >&2; }
ok()   { printf "\033[1;32m    %s\033[0m\n" "$*"; }

# -----------------------------------------------------------------------------
say "0. inspect the machine"
echo "    cores:  $(nproc)"
echo "    memory: $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GB"}' || echo '?')"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    gpu:    /'
else
  warn "nvidia-smi not found: no GPU visible. Build/train will fall back to CPU."
fi

TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "MISSING")
echo "    torch:  ${TORCH_BEFORE}"

# -----------------------------------------------------------------------------
if [ "${TORCH_BEFORE}" = "MISSING" ]; then
  say "1. torch is absent: installing a CUDA build"
  # A bare VM (the Ubuntu template) has no torch. Install the current CUDA wheel;
  # if the card is very new and this build lacks its compute capability, the GPU
  # check in step 8 will say so and you can swap the index URL.
  pip install torch --index-url https://download.pytorch.org/whl/cu121 \
    || pip install torch \
    || warn "torch install failed; the GPU stages will not run."
else
  say "1. torch ${TORCH_BEFORE} already present: it will NOT be touched"
fi

# -----------------------------------------------------------------------------
say "2. labelling-parse deps"
# snorkel is deliberately NOT installed: the Snorkel label model was evaluated
# and rejected (it collapsed on these low-overlap tools). The union rule is used.
pip install "numpy" "pandas" "pyarrow"

say "3. extraction deps: Slither (Python API, used on the HOST by build_dataset) + solc-select"
# Pin Slither to the validated version so CFG and data-flow extraction match.
pip install "slither-analyzer==0.11.5" "solc-select"

say "4. feature/embedding + model-side deps (CodeBERT, PCA, bundle I/O)"
# torch_geometric is pure python over torch and will NOT pull a different torch.
pip install "transformers" "huggingface_hub" "scikit-learn" "joblib" "safetensors" \
            "torch_geometric"

say "5. figures + tests"
pip install "matplotlib" "pytest"

# -----------------------------------------------------------------------------
say "6. solc compilers the corpus needs (idempotent; skips existing)"
# Caret pragmas use the newest patch of a minor; EXACT pins need that precise
# solc. A contract with no matching compiler simply fails extraction, so install
# the whole range the corpus uses rather than one version.
solc-select install \
  0.4.10 0.4.11 0.4.12 0.4.13 0.4.14 0.4.15 0.4.16 0.4.17 0.4.18 0.4.19 \
  0.4.20 0.4.21 0.4.22 0.4.23 0.4.24 0.4.25 0.4.26 \
  0.5.0 0.5.1 0.5.2 0.5.3 0.5.4 0.5.5 0.5.6 0.5.7 0.5.8 0.5.9 0.5.10 0.5.17 \
  0.6.12 0.7.6 0.8.19 || warn "some solc versions failed (non-fatal)"
solc-select install 0.4.4 0.4.6 0.4.8 0.4.9 || warn "ancient solc unavailable (non-fatal)"
solc-select use 0.8.19

# -----------------------------------------------------------------------------
say "7. SmartBugs + Docker (the four-tool labelling backend)"
# This is the step whose absence made every labelling task crash instantly:
# without the `sb` command, each task fails with "command not found" and the
# ledger fills with crashes in seconds.

if ! command -v docker >/dev/null 2>&1; then
  warn "docker is NOT installed. SmartBugs runs each analysis tool in a container,"
  warn "so LABELLING CANNOT RUN on this box. Install it first:"
  warn "    sudo apt-get update && sudo apt-get install -y docker.io"
  warn "    sudo usermod -aG docker \$USER && newgrp docker"
  warn "(Everything else - build, train, evaluate - still works without Docker.)"
else
  if ! docker info >/dev/null 2>&1; then
    warn "docker is installed but the daemon is unreachable. Try:"
    warn "    sudo systemctl start docker    (or add yourself to the docker group)"
  else
    ok "docker OK: $(docker --version)"

    if [ ! -d "$SMARTBUGS" ]; then
      say "7. cloning SmartBugs -> $SMARTBUGS"
      git clone --depth 1 https://github.com/smartbugs/smartbugs.git "$SMARTBUGS"
    else
      ok "SmartBugs already present at $SMARTBUGS"
    fi

    say "7. installing the sb entrypoint"
    ( cd "$SMARTBUGS" && pip install -e . ) \
      || warn "pip install of SmartBugs failed; see its README."

    if command -v sb >/dev/null 2>&1; then
      ok "sb installed: $(command -v sb)"
    else
      warn "'sb' is not on PATH. SmartBugs may be module-invoked. If so, run:"
      warn "    export SB_CMD=\"python -m sb\""
      warn "    python scripts/label_orchestrator.py ... --sb-cmd \"\$SB_CMD\""
    fi

    say "7. pre-pulling the four tool images (slow once, then cached)"
    # Pulling now means the labelling run never stalls on downloads, and it proves
    # Docker can actually fetch and run the tools BEFORE a long job starts.
    for img in "smartbugs/slither" "smartbugs/mythril" \
               "smartbugs/securify" "smartbugs/osiris"; do
      docker pull "$img" >/dev/null 2>&1 && ok "pulled $img" \
        || warn "could not pull $img (SmartBugs will fetch its own tag on first use)"
    done
  fi
fi

# -----------------------------------------------------------------------------
say "8. editable install of the scgnn package (no-deps so torch is never re-resolved)"
pip install -e . --no-deps

# -----------------------------------------------------------------------------
say "9. VERIFY torch is untouched and the GPU works"
TORCH_AFTER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "MISSING")
echo "    torch before: ${TORCH_BEFORE}"
echo "    torch after : ${TORCH_AFTER}"
if [ "${TORCH_BEFORE}" != "MISSING" ] && [ "${TORCH_BEFORE}" != "${TORCH_AFTER}" ]; then
  warn "torch version CHANGED. Something pulled a new torch."
  warn "If the GPU check below fails, reinstall the build this box shipped with:"
  warn "    pip install torch==${TORCH_BEFORE} --index-url <the wheel index you used>"
fi

python - <<'PY'
import torch
avail = torch.cuda.is_available()
print("    cuda available:", avail,
      "| device:", torch.cuda.get_device_name(0) if avail else "none")
dev = "cuda" if avail else "cpu"

from torch_geometric.data import Data, Batch
from scgnn.models.dual_gnn import DualGNN

# All three v2 encoders, including GATv2 (new in v2).
for conv in ("gcn", "sage", "gatv2"):
    m = DualGNN(in_dim=16, hid=32, conv=conv).to(dev).eval()
    g = lambda n: Data(x=torch.randn(n, 16), edge_index=torch.randint(0, n, (2, n * 2)))
    ast = Batch.from_data_list([g(5), g(7)]).to(dev)
    cfg = Batch.from_data_list([g(6), g(4)]).to(dev)
    with torch.no_grad():
        out = m(ast, cfg)
    assert tuple(out.shape) == (2, 5), out.shape
    print(f"    {conv:6s} forward OK on {dev} | logits {tuple(out.shape)}")

# The empty-CFG case: a contract whose CFG extraction degraded to zero nodes.
# This crashed training before the pooling fix, so assert it stays fixed here
# rather than discovering it hours into a run.
m = DualGNN(in_dim=16, hid=32, conv="sage").to(dev).eval()
ast = Batch.from_data_list([
    Data(x=torch.randn(4, 16), edge_index=torch.tensor([[0, 1], [1, 2]])),
    Data(x=torch.randn(3, 16), edge_index=torch.tensor([[0, 1], [1, 2]]))]).to(dev)
cfg = Batch.from_data_list([
    Data(x=torch.randn(2, 16), edge_index=torch.tensor([[0, 1], [1, 0]])),
    Data(x=torch.zeros(0, 16), edge_index=torch.zeros(2, 0, dtype=torch.long))]).to(dev)
with torch.no_grad():
    out = m(ast, cfg)
assert tuple(out.shape) == (2, 5), out.shape
print("    empty-CFG contract handled (no size mismatch):", tuple(out.shape))
PY

# -----------------------------------------------------------------------------
say "10. verify Slither produces a REAL CFG with DATA-FLOW edges"
PYTHONPATH=. python - <<'PY'
import glob
try:
    from scgnn.extraction.extract import extract_contract
    cands = sorted(glob.glob("data/raw/wild/**/*.sol", recursive=True)) \
        or sorted(glob.glob("data/raw/curated/**/*.sol", recursive=True))
    if not cands:
        print("    skipped: no contracts on disk yet (run 'run_pipeline.sh data' first)")
    else:
        ast, cfg = extract_contract(cands[0], with_data_flow=True)
        print(f"    AST nodes {ast.n_nodes} | CFG nodes {cfg.n_nodes} "
              f"| real CFG: {cfg.n_nodes > 1}")
        print(f"    data-flow edges: {cfg.n_data_flow_edges} "
              f"| degraded: {cfg.degraded}")
        if cfg.n_data_flow_edges == 0 and cfg.n_nodes > 1:
            print("    !! no data-flow edges on a real CFG: check Slither's "
                  "variables_read/variables_written on this version.")
except Exception as e:
    print("    CFG check skipped/failed:", type(e).__name__, e)
PY

# -----------------------------------------------------------------------------
say "11. run the test suite (expect 82 passed)"
PYTHONPATH=. python -m pytest -q tests/ || warn "tests reported failures - inspect above"

# -----------------------------------------------------------------------------
say "12. freeze the committed lockfile for THIS box"
pip freeze > requirements-lock.txt

say "DONE."
echo
echo "Review the checks above, especially:"
echo "  * step 7: is 'sb' on PATH and did the tool images pull?  (labelling needs this)"
echo "  * step 9: does the GPU forward pass work?"
echo "  * step 10: are data-flow edges being produced?"
echo
echo "Next:"
echo "  export SMARTBUGS=\"$SMARTBUGS\""
echo "  bash run_pipeline.sh tag data     # tag v1, download Wild + Curated"
echo "  bash run_pipeline.sh tools        # timed smoke batch, then the full run"