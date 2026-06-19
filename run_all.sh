#!/usr/bin/env bash
# run_all.sh — build -> train (gcn,sage,gat) -> evaluate, in one go.
#
# Safe by construction:
#   set -e        : abort the whole run if any command fails (no training on a
#                   broken build, no eval on a model that didn't finish).
#   set -o pipefail: a failure anywhere in a pipe still aborts.
#   tee -> log    : full output is mirrored to a timestamped log file so you can
#                   read what happened even after the terminal scrolls/closes.
#
# Usage:
#   tmux new -s run            # so an SSH drop doesn't kill the job
#   bash run_all.sh
#   # detach: Ctrl-b then d  ;  reattach: tmux attach -t run
#
# Resume note: the build reuses the extraction cache, and each per-model step is
# independent, so re-running this after a fix re-does only what is needed.

set -euo pipefail

LOG="run_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG"

# Send everything below to BOTH the screen and the log file.
exec > >(tee -a "$LOG") 2>&1

echo "===================================================================="
echo "STAGE 1/3 — BUILD DATASET   ($(date))"
echo "===================================================================="
PYTHONPATH=. python scripts/build_dataset.py \
  --wild-dir data/raw/dive_sources \
  --wild-labels data/processed/dive_labels.parquet \
  --curated-dir data/raw/curated \
  --bit-dir data/raw/bit-benchmark \
  --out data/processed --device cuda

echo
echo "===================================================================="
echo "STAGE 2/3 — TRAIN gcn, sage, gat   ($(date))"
echo "===================================================================="
for m in gcn sage gat; do
  echo "--- training $m ---"
  PYTHONPATH=. python scripts/train.py --config "configs/$m.yaml" \
    --train-index data/processed/train_index.json \
    --val-index data/processed/val_index.json \
    --out "runs/${m}_multitool"
done

echo
echo "===================================================================="
echo "STAGE 3/3 — EVALUATE gcn, sage, gat   ($(date))"
echo "===================================================================="
for m in gcn sage gat; do
  echo "--- evaluating $m ---"
  PYTHONPATH=. python scripts/evaluate.py --config "configs/$m.yaml" \
    --checkpoint "runs/${m}_multitool/best_model.pt" \
    --test-index data/processed/test_index.json \
    --val-index data/processed/val_index.json \
    --tune-threshold per_class \
    --out "runs/${m}_multitool/test_metrics.json"
done

echo
echo "===================================================================="
echo "ALL DONE   ($(date))"
echo "===================================================================="
echo "Build report : data/processed/build_report.json"
echo "Test metrics :"
for m in gcn sage gat; do
  echo "  runs/${m}_multitool/test_metrics.json"
done
echo
echo "Quick peek at test macro-F1 + dos support per model:"
for m in gcn sage gat; do
  echo "=== $m ==="
  python - "$m" <<'PY'
import json, sys
m = sys.argv[1]
try:
    d = json.load(open(f"runs/{m}_multitool/test_metrics.json"))
    macro = d.get("macro", {}).get("f1")
    dos = d.get("per_flaw", {}).get("dos", {})
    print(f"  macro-F1 = {macro}")
    print(f"  dos: f1={dos.get('f1')}  support={dos.get('support')}")
except Exception as e:
    print(f"  (could not read metrics: {e})")
PY
done