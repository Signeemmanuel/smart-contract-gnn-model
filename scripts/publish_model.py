#!/usr/bin/env python3
"""Publish a trained model artefact bundle to the Hugging Face Hub.

This is the model-side half of cross-repository handoff Point 2 from the
implementation guide: ``scgnn-model`` produces a versioned model artefact that
``scgnn-api`` downloads, pinned to an *immutable commit revision*, so the
deployed service always pairs a known checkpoint with a known code version.

The artefact **bundle** is a single local folder holding everything the back
end needs to reproduce inference exactly — not just the weights::

    bundle/
      model.safetensors     # DualGNN state_dict (weights)
      config.json           # arch: conv type, hidden size, in_dim, n_classes, threshold, scgnn version
      pca.joblib            # the PCA fitted on the TRAIN split only (768 -> 64)
      feature_config.json   # node-type vocabulary + feature layout, so one-hot encoding matches training
      README.md             # the model card (a starter card is written here if absent)

Why bundle the PCA and the feature config, not only the weights: the back end
must encode an uploaded contract's nodes with the *same* node-type vocabulary
and the *same* fitted PCA used at training time, or its predictions silently
drift from the evaluated ones. The weights alone are not enough.

Authentication uses a write token. Prefer the ``HF_TOKEN`` environment variable
(safe for CI); ``--token`` is accepted as a fallback. Create one at
https://huggingface.co/settings/tokens with write permission.

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/publish_model.py \
        --bundle data/processed/release_v0.1.0 \
        --repo-id <user-or-org>/scgnn-smartcontract \
        --version v0.1.0

On success it prints the commit SHA to pin in ``scgnn-api``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Core files the back end cannot run inference without. Missing one of these is
# an error; the model card is only a warning.
REQUIRED = ("config.json", "feature_config.json", "pca.joblib")
WEIGHTS = ("model.safetensors", "model.pt", "pytorch_model.bin")

MODEL_CARD_TEMPLATE = """\
---
license: apache-2.0
library_name: scgnn
tags:
  - smart-contracts
  - solidity
  - vulnerability-detection
  - graph-neural-network
---

# {repo_id}

Dual-graph (AST + CFG) graph neural network that detects five DASP-aligned
security flaws in Ethereum smart contracts — reentrancy, access control,
arithmetic (integer overflow/underflow), unchecked low-level calls, and Denial
of Service — and localises each to source lines.

Produced by the `scgnn-model` pipeline. Load it with the matching `scgnn`
package version recorded in `config.json`; the back end runs the identical
extraction, feature-encoding and model code that produced these weights.

## Intended use

Pre-deployment screening and auditing of Solidity contracts. It is a research
tool, not a guarantee of security, and a clean result is not a proof of safety.

## Training data and labelling

Trained on SmartBugs Wild, auto-labelled by combining Slither, Mythril and
Securify through SmartBugs and denoising their votes per flaw with a Snorkel
`LabelModel`. Labels are therefore weak (noisy), not expert-verified. Evaluated
on the expert-verified SmartBugs Curated set.

## Evaluation

<!-- Fill in with MEASURED numbers once evaluation has been run. Do not invent
results. Report per-flaw and macro precision/recall/F1 on the Curated test
split, plus top-k localisation accuracy (k = 1, 3, 5). -->

## Limitations

- Labels are weakly supervised, so systematic blind spots of the three tools
  can propagate into the model.
- Arithmetic is rare (the Solidity 0.8 compiler checks overflow by default), so
  its metrics rest on few positives.
- Localisation depends on the fidelity of Slither's source mappings.

## Licence

Apache-2.0. Bundles a fitted PCA and a node-type vocabulary; uses CodeBERT at
inference (downloaded separately, not redistributed here).
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish a model bundle to the Hugging Face Hub.")
    p.add_argument("--bundle", required=True, type=Path, help="Local folder containing the artefact bundle.")
    p.add_argument("--repo-id", required=True, help="Target repo, e.g. user-or-org/scgnn-smartcontract.")
    p.add_argument("--version", required=True, help="Version tag to create, e.g. v0.1.0.")
    p.add_argument("--private", action="store_true", help="Create the repo as private (default: public).")
    p.add_argument("--token", default=None, help="HF write token; falls back to the HF_TOKEN env var.")
    return p.parse_args(argv)


def check_bundle(bundle: Path) -> None:
    """Fail loudly before touching the network if the bundle is incomplete."""
    if not bundle.is_dir():
        sys.exit(f"error: bundle folder not found: {bundle}")
    present = {f.name for f in bundle.iterdir() if f.is_file()}
    missing = [f for f in REQUIRED if f not in present]
    if not any(w in present for w in WEIGHTS):
        missing.append(f"one of {WEIGHTS}")
    if missing:
        sys.exit("error: bundle is missing required files: " + ", ".join(missing))
    if "README.md" not in present:
        print("note: no README.md in bundle; writing a starter model card.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("error: set HF_TOKEN or pass --token (write permission required).")

    check_bundle(args.bundle)

    # Import here so --help works without huggingface_hub installed.
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    card = args.bundle / "README.md"
    if not card.exists():
        card.write_text(MODEL_CARD_TEMPLATE.format(repo_id=args.repo_id), encoding="utf-8")

    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    commit = api.upload_folder(
        folder_path=str(args.bundle),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Publish {args.version}",
    )
    sha = commit.oid  # immutable; this is what scgnn-api pins against

    # A human-friendly tag too — but note tags can be re-pointed, so the SHA is
    # the reproducible pin, not the tag.
    try:
        api.create_tag(repo_id=args.repo_id, tag=args.version, revision=sha, repo_type="model")
    except Exception as exc:  # tag may already exist; not fatal
        print(f"note: could not create tag {args.version!r}: {exc}", file=sys.stderr)

    print(f"\nPublished to https://huggingface.co/{args.repo_id}")
    print(f"commit SHA (pin this): {sha}")
    print(f"version tag:           {args.version}")
    print("\nIn scgnn-api, pin the immutable revision, e.g.:")
    print(f'    REPO_ID  = "{args.repo_id}"')
    print(f'    REVISION = "{sha}"   # not the tag')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
