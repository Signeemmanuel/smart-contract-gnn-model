#!/usr/bin/env python3
"""Continuous off-instance sync (section 5: the vast.ai disk is EPHEMERAL).

Every ``--interval`` minutes, uploads the expensive artefacts to a PRIVATE
Hugging Face dataset repo, so an instance loss costs at most one interval of
work:

  * the labelling ledger (snapshotted via SQLite's backup API, so a mid-write
    upload can never capture a torn file)
  * data/processed/            (labels.parquet + labelling reports)
  * data/testsets/             (the frozen manifests, once they exist)
  * runs/                      (checkpoints, configs, histories, results.json)
  * artifacts/                 (tables, figures, summaries)

data/sb_results is deliberately NOT in the default set: at full-corpus scale it
is >500k tiny files, and per-file sync at that scale produces thousands of Hub
commits and 504s (observed). Once labelling completes, the tree is immutable:
archive it ONCE and upload the archive as a single object instead:

    tar -I 'zstd -T0 -3' -cf sb_results_final.tar.zst -C data sb_results
    hf upload <repo-id> sb_results_final.tar.zst archives/sb_results_final.tar.zst --repo-type dataset

During a labelling run, the ledger snapshot (always synced) plus orchestrator
resume bounds the loss window to re-runnable work.

Auth: the ``HF_TOKEN`` environment variable (export it in the shell or via a
non-committed .env; NEVER commit the token). The repo is created private on
first run. The destination repo is the DEFAULT_REPO constant below — edit the
constant to change stores; it is deliberately not read from the environment.

Usage
-----
    # one-shot
    python scripts/sync_offbox.py --once

    # continuous, under tmux, while labelling/training runs elsewhere
    tmux new -s sync -d 'python scripts/sync_offbox.py --interval 15 2>&1 | tee -a sync.log'
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

# The store repo is a CONSTANT: edit this line to change stores. Deliberately
# not read from the environment — a stale SYNC_REPO export once silently
# redirected a sync to a retired repo. --repo-id remains for exceptional
# one-off use (e.g. restoring from an old store).
DEFAULT_REPO = "Signeemmanuel/scgnn-v2-store"
DEFAULT_PATHS = ["data/processed", "data/testsets", "runs", "artifacts"]


def snapshot_sqlite(src: Path, dst: Path) -> None:
    """Consistent copy of a live WAL-mode SQLite database via the backup API."""
    con = sqlite3.connect(str(src))
    try:
        bck = sqlite3.connect(str(dst))
        with bck:
            con.backup(bck)
        bck.close()
    finally:
        con.close()


def sync_once(api, repo_id: str, paths: list[str], ledger: str | None) -> None:
    from huggingface_hub import CommitOperationAdd  # noqa: F401  (import check)

    if ledger and Path(ledger).exists():
        with tempfile.TemporaryDirectory() as td:
            snap = Path(td) / Path(ledger).name
            snapshot_sqlite(Path(ledger), snap)
            api.upload_file(path_or_fileobj=str(snap),
                            path_in_repo=f"ledger/{snap.name}",
                            repo_id=repo_id, repo_type="dataset")
            print(f"  ledger snapshot -> {repo_id}/ledger/{snap.name}", flush=True)

    for p in paths:
        d = Path(p)
        if not d.exists():
            continue
        if d.is_dir():
            api.upload_folder(folder_path=str(d), path_in_repo=d.as_posix(),
                              repo_id=repo_id, repo_type="dataset",
                              ignore_patterns=["*.pyc", "__pycache__/*", "*.tmp"])
        else:
            api.upload_file(path_or_fileobj=str(d), path_in_repo=d.as_posix(),
                            repo_id=repo_id, repo_type="dataset")
        print(f"  synced {p}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO,
                    help=f"Private HF dataset repo (default: {DEFAULT_REPO}; "
                         "override with this flag or the SYNC_REPO env var).")
    ap.add_argument("--paths", nargs="+", default=DEFAULT_PATHS,
                    help=f"Paths to sync (default: {DEFAULT_PATHS}).")
    ap.add_argument("--ledger", default="data/labelling_ledger.sqlite",
                    help="Live SQLite ledger; snapshotted safely before upload.")
    ap.add_argument("--interval", type=float, default=15,
                    help="Minutes between syncs (ignored with --once).")
    ap.add_argument("--once", action="store_true", help="Sync once and exit.")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("ERROR: export HF_TOKEN first (a fresh token; the old "
                         "one committed in .env must be treated as revoked).")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    private=True, exist_ok=True)
    print(f"syncing to private dataset repo: {args.repo_id}")

    while True:
        t0 = time.time()
        try:
            sync_once(api, args.repo_id, args.paths, args.ledger)
            print(f"sync complete in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:                                  # noqa: BLE001
            print(f"WARNING sync failed ({e}); retrying next interval", flush=True)
        if args.once:
            return 0
        time.sleep(max(60.0, args.interval * 60))


if __name__ == "__main__":
    raise SystemExit(main())