#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — scgnn-model end-to-end pipeline, install through release.
#
# Reproduces every stage that produced the reported results, in order, with the
# settled default parameters (baked into configs/base.yaml and each script).
#
#   bash run_pipeline.sh <stage> [<stage> ...]
#   bash run_pipeline.sh all          # run the whole pipeline
#
# Stages (each depends on the previous; run individually to resume):
#   setup      install the gpu-01 environment (delegates to setup_gpu01.sh)
#   data       download SmartBugs Curated + Wild
#   select     pick the class-balanced Wild subset for labelling
#   tools      run the four-tool SmartBugs labelling (Slither/Mythril/Securify/Osiris)
#   label      parse tool results -> union weak labels
#   build      extract graphs + features -> train/val/test dataset
#   train      train all three architectures (GCN, GraphSAGE, GAT)
#   evaluate   score all three on the Curated test split (fixed threshold 0.5)
#   localise   run localisation on the selected model (GCN) + GAT attention
#   figures    render all per-model figures + the three-model comparison
#   release    build the deployable bundle for the selected model (GCN)
#
# Notes
#   * 'tools' needs SmartBugs installed at $SMARTBUGS (Docker). It is the only
#     stage that depends on something outside this repo; see README.
#   * Long stages (tools/build/train) should run under tmux.
#   * The selected model is GCN (see README, model comparison). Change SELECTED
#     below only if a re-run picks a different architecture.
# =============================================================================
set -uo pipefail

# ---- configuration (override via environment) -------------------------------
REPO="${REPO:-$HOME/smart-contract-gnn-model}"
SMARTBUGS="${SMARTBUGS:-$HOME/smartbugs}"
PROCESSED="${PROCESSED:-data/processed}"
SUBSET_DIR="${SUBSET_DIR:-$HOME/wild_subset}"
SELECTED="${SELECTED:-gcn}"                 # architecture chosen for release/localisation
MODELS=("gcn" "sage" "gat")
declare -A RUN=( [gcn]="runs/gcn_multitool" [sage]="runs/sage_multitool_long" [gat]="runs/gat_multitool" )
declare -A CFG=( [gcn]="configs/gcn.yaml"  [sage]="configs/sage.yaml"          [gat]="configs/gat.yaml" )
export PYTHONPATH="${PYTHONPATH:-.}"

cd "$REPO"
say() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
die() { printf "\n\033[1;31mERROR: %s\033[0m\n" "$*" >&2; exit 1; }

stage_setup() {
  say "SETUP — environment (delegates to setup_gpu01.sh; preserves working torch)"
  [ -f setup_gpu01.sh ] || die "setup_gpu01.sh not found in repo root."
  bash setup_gpu01.sh
}

stage_data() {
  say "DATA — download SmartBugs Curated + Wild into data/raw"
  python scripts/download_data.py
}

stage_select() {
  say "SELECT — class-balanced Wild subset -> $SUBSET_DIR"
  python scripts/select_wild_subset.py --wild-dir data/raw/wild --out "$SUBSET_DIR"
}

stage_tools() {
  say "TOOLS — four-tool SmartBugs labelling run (CPU/Docker; long)"
  [ -d "$SMARTBUGS" ] || die "SmartBugs not found at $SMARTBUGS (set \$SMARTBUGS)."
  ( cd "$SMARTBUGS" && PYTHONPATH=. python -m sb \
      -t slither mythril securify osiris \
      -f "$SUBSET_DIR/*.sol" --processes 6 --mem-limit 4g --timeout 300 \
      --json --continue-on-errors )
  say "TOOLS — audit raw votes (sanity: arithmetic should come from Osiris)"
  python scripts/validate_labels.py --results "$SMARTBUGS/results"
}

stage_label() {
  say "LABEL — parse tool results -> union weak labels (+ Snorkel for comparison)"
  python scripts/label.py --results "$SMARTBUGS/results" --out "$PROCESSED"
  python scripts/label.py --results "$SMARTBUGS/results" --out "${PROCESSED}_snorkel" --method snorkel
}

stage_build() {
  say "BUILD — extract graphs + features -> dataset (val-frac 0.1, embed-batch 256)"
  python scripts/build_dataset.py \
    --wild-dir data/raw/wild --wild-labels "$PROCESSED/labels.parquet" \
    --curated-dir data/raw/curated --out "$PROCESSED" --device cuda
}

stage_train() {
  for m in "${MODELS[@]}"; do
    say "TRAIN — $m -> ${RUN[$m]} (epochs/patience/batch from config)"
    python scripts/train.py --config "${CFG[$m]}" \
      --train-index "$PROCESSED/train_index.json" \
      --val-index "$PROCESSED/val_index.json" \
      --out "${RUN[$m]}"
  done
}

stage_evaluate() {
  for m in "${MODELS[@]}"; do
    say "EVALUATE — $m on Curated test (threshold 0.5 from config)"
    python scripts/evaluate.py --config "${CFG[$m]}" \
      --checkpoint "${RUN[$m]}/best_model.pt" \
      --test-index "$PROCESSED/test_index.json" \
      --out "${RUN[$m]}/eval_05.json"
  done
}

stage_localise() {
  say "LOCALISE — GNNExplainer on selected model ($SELECTED), exact + tolerance"
  for t in 0 1 2; do
    python scripts/localise_eval.py --merge max --tolerance "$t" \
      --checkpoint "${RUN[$SELECTED]}/best_model.pt" --config "${CFG[$SELECTED]}" \
      --out "${RUN[$SELECTED]}/localisation_tol${t}.json"
  done
  say "LOCALISE — GAT attention (proposal's secondary signal), exact + tolerance"
  for t in 0 1 2; do
    python scripts/localise_eval.py --method attention --merge max --tolerance "$t" \
      --checkpoint "${RUN[gat]}/best_model.pt" --config "${CFG[gat]}" \
      --out "${RUN[gat]}/localisation_attention_tol${t}.json"
  done
}

stage_figures() {
  for m in "${MODELS[@]}"; do
    say "FIGURES — per-model set for $m"
    cp "${RUN[$m]}/eval_05.json" "$PROCESSED/eval_metrics.json"
    python scripts/make_figures.py --processed "$PROCESSED" \
      --run-dir "${RUN[$m]}" --out "reports/figures/$m"
  done
  say "FIGURES — three-model comparison (the model-selection centrepiece)"
  python scripts/make_comparison_figure.py \
    --eval gcn="${RUN[gcn]}/eval_05.json" \
    --eval sage="${RUN[sage]}/eval_05.json" \
    --eval gat="${RUN[gat]}/eval_05.json" \
    --out reports/figures/comparison
}

stage_release() {
  say "RELEASE — deployable bundle for selected model ($SELECTED)"
  python scripts/build_release_bundle.py \
    --checkpoint "${RUN[$SELECTED]}/best_model.pt" \
    --config "${RUN[$SELECTED]}/config.json" \
    --feature-config "$PROCESSED/feature_config.json" \
    --pca "$PROCESSED/pca.joblib" \
    --out "release/${SELECTED}_v1"
  say "RELEASE — bundle ready at release/${SELECTED}_v1"
  echo "Publish with: python scripts/publish_model.py --bundle release/${SELECTED}_v1 --repo-id <id> --version v1"
}

ALL=(setup data select tools label build train evaluate localise figures release)
[ $# -eq 0 ] && { echo "usage: bash run_pipeline.sh <stage|all> [...]"; echo "stages: ${ALL[*]}"; exit 1; }
[ "$1" = "all" ] && set -- "${ALL[@]}"

for stage in "$@"; do
  case "$stage" in
    setup) stage_setup ;; data) stage_data ;; select) stage_select ;;
    tools) stage_tools ;; label) stage_label ;; build) stage_build ;;
    train) stage_train ;; evaluate) stage_evaluate ;; localise) stage_localise ;;
    figures) stage_figures ;; release) stage_release ;;
    *) die "unknown stage '$stage'. Valid: ${ALL[*]} (or 'all')." ;;
  esac
done
say "DONE: $*"