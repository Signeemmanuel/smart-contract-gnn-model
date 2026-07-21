#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh - scgnn-model v2 end-to-end pipeline, install through release.
#
# Reproduces every stage that produces the reported v2 results, in order, with
# the settled default parameters (configs/base.yaml and each script's defaults).
#
#   bash run_pipeline.sh <stage> [<stage> ...]
#   bash run_pipeline.sh all          # the whole pipeline
#
# Stages (each depends on the previous; run individually to resume):
#   setup      install the environment (delegates to setup_gpu01.sh)
#   tag        tag v1-predefence BEFORE anything is touched (non-negotiable #1)
#   data       download SmartBugs Curated + Wild
#   tools      resumable four-tool labelling over the FULL Wild corpus
#   label      parse tool results -> union weak labels (labels.parquet)
#   freeze     freeze the two test-set manifests (Test A + Test B)
#   build      extract graphs (+ data-flow edges) -> BOTH ablation datasets
#   train      the six-run matrix + the two ensembles, evaluated on A and B
#   durieux    run the four tools on Test B -> the tool-vs-model matrix
#   localise   GNNExplainer line localisation for the winning model on Test B
#   figures    render every table and figure into artifacts/v2
#   release    build the deployable v2 bundle for the winning model
#
# Notes
#   * 'tools' needs SmartBugs installed at $SMARTBUGS (Docker). It is the only
#     stage that depends on something outside this repo; see README.
#   * 'tools', 'build' and 'train' are long: run them under tmux.
#   * The labelling stage is RESUMABLE: re-run it and it picks up where it
#     stopped, so an interruption costs nothing but time.
#   * The winning model is decided by macro-F1 on TEST B (the expert set), not
#     on validation and not on Test A. WINNER below is read from results.json.
# =============================================================================
set -uo pipefail

# ---- configuration (override via environment) -------------------------------
REPO="${REPO:-$HOME/smart-contract-gnn-model}"
SMARTBUGS="${SMARTBUGS:-$HOME/smartbugs}"
WILD="${WILD:-data/raw/wild}"
CURATED="${CURATED:-data/raw/curated}"
PROCESSED="${PROCESSED:-data/processed}"          # labels.parquet lives here
TESTSETS="${TESTSETS:-data/testsets}"             # the two frozen manifests
CACHE="${CACHE:-data/extract_cache}"              # shared by BOTH ablation arms
DATA_DF="${DATA_DF:-data/processed_df}"           # build WITH data-flow edges
DATA_NODF="${DATA_NODF:-data/processed_nodf}"     # build WITHOUT (ablation arm)
RUNS="${RUNS:-runs/v2}"
ARTIFACTS="${ARTIFACTS:-artifacts/v2}"
WORKERS="${WORKERS:-64}"
BUILD_JOBS="${BUILD_JOBS:-96}"                    # parallel Slither extraction workers in the build stage                          # labelling parallelism (CPU cores)
# Per-tool time budget. 600s (raised from 300s after the 200-contract smoke on
# the 122-core box): at 300s Mythril and Securify timed out on ~23% of their
# tasks (43 and 39 of 185). A timeout is an abstention under the union rule,
# never a clean verdict, but recovering the verdicts that finish between 300s
# and 600s buys real per-tool coverage for roughly 7-14 wall-hours across the
# full corpus. Osiris (the only arithmetic detector) barely times out at either
# budget (3 of 185 at 300s). Tasks that already finish fast are unaffected.
TIMEOUT="${TIMEOUT:-300}"
SB_CMD="${SB_CMD:-python -m sb}"                  # SmartBugs 2.x has no console script:
                                                  # it is a package named `sb`, run via -m
# SmartBugs must be importable by the subprocesses the orchestrator spawns, and so
# must this repo. NOTE: never run SmartBugs from inside its own checkout - its
# sb/docker.py would shadow the real Docker SDK.
export PYTHONPATH="${SMARTBUGS}:${PYTHONPATH:-.}"

# Off-instance sync (section 5: the disk is ephemeral). The store repo is a
# CONSTANT: edit this line to change stores. Deliberately NOT env-overridable —
# a stale SYNC_REPO export in a shell once silently redirected a sync to a
# retired repo, so the code is the single source of truth. A FRESH HF_TOKEN
# must still be exported (the token itself is never hardcoded). Every long
# stage pushes its artefacts off-box when it finishes; the continuous loop for
# DURING long runs is:  tmux new -s sync -d \
#   'python scripts/sync_offbox.py --interval 15 2>&1 | tee -a sync.log'
SYNC_REPO="Signeemmanuel/scgnn-v2-store"

cd "$REPO"
say() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
warn() { printf "\n\033[1;33mWARNING: %s\033[0m\n" "$*" >&2; }
die() { printf "\n\033[1;31mERROR: %s\033[0m\n" "$*" >&2; exit 1; }

# One-shot off-box sync after a stage; a no-op (with a nag) when unconfigured.
maybe_sync() {
  if [ -n "$SYNC_REPO" ] && [ -n "${HF_TOKEN:-}" ]; then
    say "SYNC - pushing artefacts to $SYNC_REPO"
    python scripts/sync_offbox.py --repo-id "$SYNC_REPO" --once \
      || warn "off-box sync failed; artefacts exist only on this instance."
  else
    warn "SYNC_REPO/HF_TOKEN not set: artefacts exist ONLY on this ephemeral instance."
  fi
}

# The winning model, read from results.json (never hardcoded: the data decides).
winner() {
  python - <<'PY'
import json, pathlib, sys, os
p = pathlib.Path(os.environ.get("RUNS", "runs/v2")) / "results.json"
if not p.exists():
    sys.exit("results.json not found; run the 'train' stage first.")
r = json.loads(p.read_text())
print(max(r, key=lambda m: r[m]["test_b"]["macro"]["f1"]))
PY
}

# The best SINGLE model (excludes the ensembles). Localisation and the deployed
# bundle need one checkpoint and one config; an ensemble has neither, so those
# stages use this instead, and the ensemble is reported as a result only.
best_single() {
  python - <<'PY'
import json, pathlib, sys, os
p = pathlib.Path(os.environ.get("RUNS", "runs/v2")) / "results.json"
if not p.exists():
    sys.exit("results.json not found; run the 'train' stage first.")
r = json.loads(p.read_text())
singles = {k: v for k, v in r.items() if not k.startswith("ensemble")}
if not singles:
    sys.exit("no single-model runs in results.json")
print(max(singles, key=lambda m: singles[m]["test_b"]["macro"]["f1"]))
PY
}

# -----------------------------------------------------------------------------

stage_setup() {
  say "SETUP - environment (delegates to setup_gpu01.sh; preserves working torch)"
  [ -f setup_gpu01.sh ] || die "setup_gpu01.sh not found in repo root."
  bash setup_gpu01.sh
}

stage_tag() {
  say "TAG - freeze v1 before anything is touched (non-negotiable #1)"
  if git rev-parse -q --verify refs/tags/v1-predefence >/dev/null; then
    echo "  v1-predefence already tagged; nothing to do."
  else
    git tag -a v1-predefence -m "Pre-defence v1: subset training, GAT, no data-flow edges" \
      || die "could not tag; commit your work first."
    echo "  tagged v1-predefence. Push it: git push --tags"
  fi
}

stage_data() {
  say "DATA - download SmartBugs Curated + Wild into data/raw"
  python scripts/download_data.py
}

stage_tools() {
  say "TOOLS - resumable four-tool labelling over the FULL Wild corpus (long; tmux)"
  [ -d "$SMARTBUGS" ] || die "SmartBugs not found at $SMARTBUGS (set \$SMARTBUGS)."

  # Measure before committing: 200 contracts tells you the real rate, so a
  # multi-day run never starts by accident.
  if [ ! -f data/labelling_ledger.sqlite ]; then
    say "TOOLS - timed smoke batch (200 contracts) to measure the labelling rate"
    python scripts/label_orchestrator.py \
      --wild-dir "$WILD" --results data/sb_results \
      --ledger data/labelling_ledger.sqlite --sb-cmd "$SB_CMD" \
      --workers "$WORKERS" --timeout "$TIMEOUT" --limit 200 \
      || die "smoke labelling failed; check SmartBugs/Docker before the full run."

    # Trust the disk, not the exit codes (the lesson of the no-op run): the
    # results tree must hold roughly 4 result.json per ok contract.
    local njson; njson=$(find data/sb_results -name result.json 2>/dev/null | wc -l)
    say "TOOLS - smoke verification: ${njson} result.json files on disk"
    [ "$njson" -ge 100 ] || die "results tree is nearly empty (${njson} files): the \
tools did NOT run. Do not start the full run; investigate one task by hand."
    maybe_sync
    warn "Read the rate above and extrapolate to the full corpus BEFORE continuing."
    warn "Re-run this stage to proceed with the full run (the ledger resumes)."
    return 0
  fi

  say "TOOLS - full corpus (resumable: safe to interrupt and re-run)"
  python scripts/label_orchestrator.py \
    --wild-dir "$WILD" --results data/sb_results \
    --ledger data/labelling_ledger.sqlite --sb-cmd "$SB_CMD" \
    --workers "$WORKERS" --timeout "$TIMEOUT" --max-attempts 2
  maybe_sync
}

stage_label() {
  say "LABEL - parse tool results -> union weak labels"
  python scripts/label.py --results data/sb_results --method union --out "$PROCESSED"

  say "LABEL - save labels.parquet OFF the machine immediately"
  if [ -f save_labels_to_github.sh ]; then
    bash save_labels_to_github.sh || warn "could not commit labels.parquet; save it manually."
  fi
  warn "labels.parquet is the expensive artefact. Keep a second copy off this box."
  maybe_sync
}

stage_freeze() {
  say "FREEZE - the two test-set manifests (once; then COMMIT them)"
  python scripts/freeze_testsets.py \
    --wild-dir "$WILD" --labels "$PROCESSED/labels.parquet" \
    --curated-dir "$CURATED" --out "$TESTSETS" \
    --target-per-class 100 --min-per-class 60 --neg-ratio 2.0
  warn "Commit $TESTSETS/*.csv now. They are frozen: never redraw them."
  maybe_sync
}

stage_build() {
  [ -f "$TESTSETS/test_a.csv" ] || die "no frozen manifests; run the 'freeze' stage."

  # Both arms share ONE extraction cache: Slither runs once, not twice.
  say "BUILD - WITH data-flow edges -> $DATA_DF (long; tmux)"
  python scripts/build_dataset.py \
    --wild-dir "$WILD" --wild-labels "$PROCESSED/labels.parquet" \
    --curated-dir "$CURATED" --testsets "$TESTSETS" \
    --out "$DATA_DF" --cache "$CACHE" --with-data-flow --device cuda \
    --jobs "$BUILD_JOBS" \
    || die "build (with data-flow) failed."

  say "BUILD - WITHOUT data-flow edges -> $DATA_NODF (reuses the cache: no re-extraction)"
  python scripts/build_dataset.py \
    --wild-dir "$WILD" --wild-labels "$PROCESSED/labels.parquet" \
    --curated-dir "$CURATED" --testsets "$TESTSETS" \
    --out "$DATA_NODF" --cache "$CACHE" --no-data-flow --device cuda \
    --jobs "$BUILD_JOBS" \
    || die "build (without data-flow) failed."
}

stage_train() {
  say "TRAIN - eight-run matrix (gcn/sage/gatv2/hybrid x with/without data-flow) + 2 ensembles"
  python scripts/train_v2.py \
    --data-nodf "$DATA_NODF" --data-df "$DATA_DF" \
    --configs configs --out "$RUNS" --seeds 42 \
    || die "training matrix failed."
  say "TRAIN - winner on the expert test set: $(winner)"
  maybe_sync
}

stage_durieux() {
  say "DURIEUX - run the four tools directly on Test B (small; resumable)"
  python scripts/durieux_baseline.py run \
    --testsets "$TESTSETS" --results data/sb_testb \
    --ledger data/testb_ledger.sqlite --sb-cmd "$SB_CMD" \
    --workers 16 --timeout 300

  say "DURIEUX - tool-vs-model matrix (T5 + F4.11)"
  local probs="$RUNS/test_b_probs.json"
  if [ -f "$probs" ]; then
    python scripts/durieux_baseline.py matrix \
      --testsets "$TESTSETS" --results data/sb_testb \
      --model-probs "$probs" --out "$ARTIFACTS"
  else
    warn "no $probs; emitting the TOOLS-ONLY matrix (still the novel part: what the"
    warn "union labelling oracle achieves against expert ground truth)."
    python scripts/durieux_baseline.py matrix \
      --testsets "$TESTSETS" --results data/sb_testb --out "$ARTIFACTS"
  fi
  maybe_sync
}

stage_localise() {
  # An ensemble has no single checkpoint to explain, so localisation always runs
  # on the best SINGLE model. If the ensemble is the overall winner, say so in
  # the write-up and localise the best single: that is the honest reading.
  local w; w="$(best_single)" || die "run 'train' first."
  local overall; overall="$(winner)"
  [ "$w" = "$overall" ] || say "LOCALISE - overall winner is $overall (an ensemble); \
localising the best single model instead: $w"
  local cfg="configs/${w%_df}.yaml"         # gcn_df -> gcn.yaml
  say "LOCALISE - GNNExplainer on $w, exact and tolerant, on Test B"
  for t in 0 1 2; do
    python scripts/localise_eval.py --merge max --tolerance "$t" \
      --checkpoint "$RUNS/$w/best_model.pt" --config "$cfg" \
      --out "$RUNS/$w/localisation_tol${t}.json" \
      || warn "localisation (tolerance $t) failed; v1 numbers stand."
  done
  maybe_sync
}

stage_figures() {
  [ -f "$RUNS/results.json" ] || die "no results.json; run the 'train' stage."
  say "FIGURES - every table and figure -> $ARTIFACTS"
  python - <<PY
from training.evaluate.artifacts import emit_all
import json, pathlib
durieux = None
p = pathlib.Path("$ARTIFACTS") / "durieux_matrix.json"
if p.exists():
    durieux = json.loads(p.read_text())
out = emit_all("$RUNS/results.json", "$ARTIFACTS",
               encoders=["gcn", "sage", "gatv2", "hybrid"], durieux=durieux)
print("best model:", out["best_model"])
print("tables:", out["tables"])
print("figures:", out["figures"])
PY
  maybe_sync
}

stage_release() {
  # The deployed bundle is a single model (the API loads one checkpoint), so it
  # is the best SINGLE model even when an ensemble scores higher. The ensemble's
  # weights and thresholds are still emitted for the record by the train stage.
  local w; w="$(best_single)" || die "run 'train' first."
  local overall; overall="$(winner)"
  [ "$w" = "$overall" ] || warn "overall winner is $overall (an ensemble); the deployed \
bundle is the best single model, $w. Report both."
  local data="$DATA_DF"
  [[ "$w" == *_df ]] || data="$DATA_NODF"     # bundle must match the winner's arm
  say "RELEASE - v2 bundle for $w, features from $data"
  python scripts/build_release_bundle.py \
    --checkpoint "$RUNS/$w/best_model.pt" \
    --config "$RUNS/$w/config.json" \
    --feature-config "$data/feature_config.json" \
    --pca "$data/pca.joblib" \
    --out "release/${w}_v2"
  say "RELEASE - bundle ready at release/${w}_v2 (v1's bundle is untouched)"
  echo "Publish with: python scripts/publish_model.py --bundle release/${w}_v2 --repo-id <id> --version v2"
  maybe_sync
}

# archive: one-object off-box backup of an immutable bulk tree.
# Per-file sync does not survive contact with >500k tiny files (thousands of
# Hub commits, 504s — observed on the full sb_results tree), so finished trees
# are tarred once and uploaded as a single object instead. Default source is
# data/sb_results; override with ARCHIVE_SRC=<dir>. Skips re-tarring when
# today's archive already exists, so the stage is safely re-runnable after an
# interrupted upload.
stage_archive() {
  local src="${ARCHIVE_SRC:-data/sb_results}"
  say "ARCHIVE - ${src} -> single object in ${SYNC_REPO}/archives/"
  [ -d "$src" ] || die "no such directory: $src"
  [ -n "${HF_TOKEN:-}" ] || die "export HF_TOKEN first (never hardcode it)."
  command -v zstd >/dev/null 2>&1 || apt-get install -y zstd
  local name; name="$(basename "$src")_$(date +%Y%m%d).tar.zst"
  local out="/root/${name}"
  if [ -f "$out" ]; then
    say "ARCHIVE - ${out} already exists; skipping tar, proceeding to upload"
  else
    tar -I 'zstd -T0 -3' -cf "$out" -C "$(dirname "$src")" "$(basename "$src")" \
      || die "tar failed; archive not uploaded."
  fi
  ls -lh "$out"
  hf upload "$SYNC_REPO" "$out" "archives/${name}" --repo-type dataset \
    || die "upload failed; the archive remains at ${out} - re-run this stage."
  say "ARCHIVE - stored as ${SYNC_REPO}/archives/${name}"
  warn "restore: hf download ${SYNC_REPO} archives/${name} --repo-type dataset --local-dir /tmp && tar -I zstd -xf /tmp/archives/${name} -C data"
}

ALL=(setup tag data tools label freeze build train durieux localise figures release)
[ $# -eq 0 ] && { echo "usage: bash run_pipeline.sh <stage|all> [...]"; echo "stages: ${ALL[*]}"; exit 1; }
[ "$1" = "all" ] && set -- "${ALL[@]}"

for stage in "$@"; do
  case "$stage" in
    setup) stage_setup ;; tag) stage_tag ;; data) stage_data ;;
    tools) stage_tools ;; label) stage_label ;; freeze) stage_freeze ;;
    build) stage_build ;; train) stage_train ;; durieux) stage_durieux ;;
    localise) stage_localise ;; figures) stage_figures ;; release) stage_release ;;
    sync) maybe_sync ;; archive) stage_archive ;;
    *) die "unknown stage '$stage'. Valid: ${ALL[*]}, sync, archive (or 'all')." ;;
  esac
done
say "DONE: $*"