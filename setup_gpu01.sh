#!/usr/bin/env bash
# One-shot environment setup for gpu-01 (RTX PRO 4000 Blackwell, sm_120).
#
# Run ONCE from the repo root:  bash setup_gpu01.sh
#
# Unlike setup_studio.sh (which pins the L4's torch 2.8.0+cu128), this script
# DOES NOT touch torch: gpu-01 already has a working, verified Blackwell build
# (torch 2.12.0+cu130). Re-installing torch here risks pulling a build without
# sm_120 support and breaking the GPU. Every install below uses --no-deps-style
# care or plain installs that won't drag torch in, and we re-assert torch at the
# end so a mistake is caught immediately.
#
# Covers all three phases on one box:
#   - labelling parse  (numpy, pandas, pyarrow, snorkel)
#   - extraction+build (slither-analyzer + solc-select, transformers, sklearn)
#   - training/eval    (torch_geometric, safetensors, joblib) + figures + tests
set -uo pipefail   # not -e: we want to report failures, not abort the whole script

echo "==> 0. record the torch we must NOT disturb"
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "MISSING")
echo "    torch currently: ${TORCH_BEFORE}"
if [ "${TORCH_BEFORE}" = "MISSING" ]; then
  echo "    !! torch is not installed. This script assumes the working Blackwell"
  echo "       torch is already present. Install it first, then re-run."
fi

echo "==> 1. labelling-parse deps"
pip install "numpy" "pandas" "pyarrow" "snorkel"

echo "==> 2. extraction deps: Slither (Python API, used on the HOST by build_dataset) + solc-select"
# Pin Slither to the version validated on the Studio so CFG extraction matches.
pip install "slither-analyzer==0.11.5" "solc-select"

echo "==> 3. feature/embedding + model-side deps (CodeBERT, PCA, bundle I/O)"
# torch_geometric is pure-python over torch and will NOT pull a different torch.
pip install "transformers" "huggingface_hub" "scikit-learn" "joblib" "safetensors" \
            "torch_geometric"

echo "==> 4. figures + tests"
pip install "matplotlib" "pytest"

echo "==> 5. install solc compilers the corpus needs (idempotent; skips existing)"
# Caret pragmas use the newest patch of a minor; EXACT pins need that precise solc.
# This superset covers the curated + wild exact pins seen in this project.
solc-select install \
  0.4.10 0.4.11 0.4.12 0.4.13 0.4.14 0.4.15 0.4.16 0.4.17 0.4.18 0.4.19 \
  0.4.20 0.4.21 0.4.22 0.4.23 0.4.24 0.4.25 0.4.26 \
  0.5.0 0.5.1 0.5.2 0.5.3 0.5.4 0.5.5 0.5.6 0.5.7 0.5.8 0.5.9 0.5.10 0.5.17 \
  0.6.12 0.7.6 0.8.19 || echo "    (some versions may have failed; non-fatal)"
solc-select install 0.4.4 0.4.6 0.4.8 0.4.9 || echo "    (ancient solc unavailable; non-fatal)"

echo "==> 6. editable install of the scgnn package (no-deps so torch is never re-resolved)"
pip install -e . --no-deps

echo
echo "==> 7. VERIFY torch is untouched and the GPU still works"
TORCH_AFTER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "MISSING")
echo "    torch before: ${TORCH_BEFORE}"
echo "    torch after : ${TORCH_AFTER}"
if [ "${TORCH_BEFORE}" != "${TORCH_AFTER}" ] && [ "${TORCH_BEFORE}" != "MISSING" ]; then
  echo "    !! WARNING: torch version CHANGED. Something pulled a new torch."
  echo "       If the GPU check below fails, reinstall the Blackwell build:"
  echo "       pip install torch==${TORCH_BEFORE} --index-url <the cu130 wheel index you used>"
fi
python - <<'PY'
import torch
print("    cuda available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
from torch_geometric.data import Data, Batch
from scgnn.models.dual_gnn import DualGNN
m = DualGNN(in_dim=16, hid=32, conv="sage").to("cuda").eval()
g = lambda n: Data(x=torch.randn(n, 16), edge_index=torch.randint(0, n, (2, n * 2)))
ast = Batch.from_data_list([g(5), g(7)]).to("cuda")
cfg = Batch.from_data_list([g(6), g(4)]).to("cuda")
with torch.no_grad():
    out = m(ast, cfg)
print("    GPU forward OK:", tuple(out.shape), "(expect (2, 5))")
PY

echo "==> 8. verify Slither's Python API produces a REAL CFG (not the 1-node placeholder)"
PYTHONPATH=. python - <<'PY'
import glob
try:
    from scgnn.extraction.extract import extract_contract
    f = sorted(glob.glob("data/raw/wild/contracts/*.sol"))[0]
    ast, cfg = extract_contract(f)
    print(f"    AST nodes {ast.n_nodes} | CFG nodes {cfg.n_nodes} | real CFG: {cfg.n_nodes > 1}")
except Exception as e:
    print("    CFG check skipped/failed:", type(e).__name__, e)
PY

echo "==> 9. run the test suite"
PYTHONPATH=. python -m pytest -q tests/ || echo "    (tests reported failures — inspect above)"

echo
echo "==> 10. freeze the committed lockfile for THIS box (cu130/Blackwell)"
pip freeze > requirements-lock.txt
echo "DONE. Review the VERIFY/CFG checks above; commit requirements-lock.txt."