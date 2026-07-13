#!/usr/bin/env python3
"""Publish a trained model artefact bundle to the Hugging Face Hub.

This is the model-side half of the cross-repository handoff: ``scgnn-model``
produces a versioned model artefact that ``scgnn-api`` downloads, pinned to an
*immutable commit revision*, so the deployed service always pairs a known
checkpoint with a known code version.

The artefact bundle is a single local folder holding everything the back end
needs to reproduce inference exactly, not just the weights::

    bundle/
      model.safetensors     # DualGNN state_dict (weights)
      config.json           # arch: conv type, hidden size, in_dim, n_classes
      manifest.json         # v2: PER-CLASS thresholds, edge schema, provenance
      pca.joblib            # the PCA fitted on the TRAIN split only (768 -> 64)
      feature_config.json   # node-type vocabulary + feature layout
      README.md             # the model card (a starter card is written if absent)

Why the manifest is required for a v2 bundle: v2 tunes one decision threshold
PER CLASS on the validation split. A bundle published without it would leave the
API applying a single global threshold, which silently suppresses the rare
classes (a global 0.7 can miss every flaw a per-class rule catches). The check
below therefore FAILS a v2 publish that has no manifest, rather than shipping a
model that quietly behaves worse than the reported numbers.

Why bundle the PCA and the feature config, not only the weights: the back end
must encode an uploaded contract's nodes with the same node-type vocabulary and
the same fitted PCA used at training time, or its predictions silently drift
from the evaluated ones.

Authentication uses a write token. Prefer the ``HF_TOKEN`` environment variable
(safe for CI); ``--token`` is accepted as a fallback.

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/publish_model.py \
        --bundle release/sage_df_v2 \
        --repo-id <user-or-org>/scgnn-smartcontract \
        --version v2

On success it prints the commit SHA to pin in ``scgnn-api``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Core files the back end cannot run inference without.
REQUIRED = ("config.json", "feature_config.json", "pca.joblib")
# Required for a v2 bundle: without it the API cannot apply the per-class rule.
REQUIRED_V2 = ("manifest.json",)
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
security flaws in Ethereum smart contracts (reentrancy, access control,
arithmetic, unchecked low-level calls, and denial of service) and localises each
to source lines.

Produced by the `scgnn-model` pipeline. Load it with the matching `scgnn`
package version recorded in `config.json`; the back end runs the identical
extraction, feature-encoding and model code that produced these weights.

## Bundle contents

| File | Purpose |
| --- | --- |
| `model.safetensors` | model weights |
| `config.json` | architecture |
| `manifest.json` | per-class decision thresholds, edge schema, provenance |
| `feature_config.json` | node-type vocabulary and feature layout |
| `pca.joblib` | PCA fitted on the training split only |

`manifest.json` carries one decision threshold PER CLASS, tuned on the
validation split. Apply those thresholds, not a single global cut-off: the rare
classes need lower thresholds and a global rule suppresses them.

## Intended use

Pre-deployment screening and auditing of Solidity contracts. This is a research
tool, not a guarantee of security, and a clean result is not a proof of safety.

## Training data and labelling

Trained on SmartBugs Wild. Labels are produced by running four analysis tools
(Slither, Mythril, Securify and Osiris) through SmartBugs and combining their
findings with a UNION rule: a contract is positive for a flaw if any tool that
covers that flaw reports it. A Snorkel label model was evaluated and rejected,
because it collapsed on these low-overlap tools.

Labels are therefore weak and tool-derived, not expert-verified. The model is
evaluated on two frozen benchmarks: a tool-labelled test set drawn from the same
distribution, and the expert-labelled SmartBugs Curated set.

## Evaluation

{evaluation}

## Limitations

- Labels are weakly supervised, so the systematic blind spots of the four tools
  can propagate into the model.
- Performance on the expert-labelled benchmark is substantially lower than on
  the tool-labelled one. The model learns the tools' union rule, which is itself
  an imperfect approximation of expert judgement.
- Denial of service is rare in the corpus and poorly covered by the four tools,
  so its metrics rest on few positives and should be read with caution.
- Localisation depends on the fidelity of Slither's source mappings.

## Licence

Apache-2.0. Bundles a fitted PCA and a node-type vocabulary; uses CodeBERT at
inference (downloaded separately, not redistributed here).
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publish a model bundle to the Hugging Face Hub.")
    p.add_argument("--bundle", required=True, type=Path,
                   help="Local folder containing the artefact bundle.")
    p.add_argument("--repo-id", required=True,
                   help="Target repo, e.g. user-or-org/scgnn-smartcontract.")
    p.add_argument("--version", required=True, help="Version tag to create, e.g. v2.")
    p.add_argument("--private", action="store_true",
                   help="Create the repo as private (default: public).")
    p.add_argument("--token", default=None,
                   help="HF write token; falls back to the HF_TOKEN env var.")
    return p.parse_args(argv)


def check_bundle(bundle: Path, version: str) -> dict:
    """Fail loudly before touching the network if the bundle is incomplete.

    A v2 bundle without ``manifest.json`` is REJECTED: publishing it would leave
    the API applying a single global threshold instead of the validation-tuned
    per-class rule, silently degrading the deployed model relative to the
    reported numbers. Returns the manifest (empty dict for a v1 bundle).
    """
    if not bundle.is_dir():
        sys.exit(f"error: bundle folder not found: {bundle}")
    present = {f.name for f in bundle.iterdir() if f.is_file()}

    missing = [f for f in REQUIRED if f not in present]
    if not any(w in present for w in WEIGHTS):
        missing.append(f"one of {WEIGHTS}")

    is_v2 = str(version).lstrip("v").startswith("2") or "manifest.json" in present
    if is_v2:
        missing += [f for f in REQUIRED_V2 if f not in present]

    if missing:
        sys.exit("error: bundle is missing required files: " + ", ".join(missing)
                 + ("\n  A v2 bundle MUST carry manifest.json: it holds the "
                    "per-class thresholds the API needs. Rebuild the bundle with "
                    "scripts/build_release_bundle.py." if is_v2 else ""))

    manifest: dict = {}
    if "manifest.json" in present:
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        thr = manifest.get("thresholds") or []
        names = manifest.get("class_names") or []
        if len(thr) != len(names) or not thr:
            sys.exit(f"error: manifest.json has {len(thr)} thresholds for "
                     f"{len(names)} classes; refusing to publish a mismatched "
                     f"decision rule.")
        print(f"manifest: {manifest.get('architecture')} "
              f"in_dim={manifest.get('in_dim')} "
              f"edges={manifest.get('edge_schema')} "
              f"thresholds={[round(float(t), 2) for t in thr]}")

    if "README.md" not in present:
        print("note: no README.md in bundle; writing a starter model card.",
              file=sys.stderr)
    return manifest


def _evaluation_section(manifest: dict) -> str:
    """Model-card evaluation text. Never invents numbers: if none are supplied,
    it says so plainly rather than shipping a hollow placeholder."""
    if not manifest:
        return ("Not yet filled in. Report per-class and macro precision, recall "
                "and F1 on both frozen test sets, plus top-k localisation "
                "accuracy, from the measured results.")
    bits = []
    if manifest.get("training_set_size"):
        bits.append(f"- Training contracts: {manifest['training_set_size']:,}")
    if manifest.get("edge_schema"):
        bits.append(f"- Graph edges: {manifest['edge_schema']}")
    if manifest.get("architecture"):
        bits.append(f"- Encoder: {manifest['architecture']}")
    bits.append("")
    bits.append("Measured metrics are reported in the accompanying dissertation. "
                "Fill in the per-class and macro figures here from results.json; "
                "do not state numbers that have not been measured.")
    return "\n".join(bits)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("error: set HF_TOKEN or pass --token (write permission required).")

    manifest = check_bundle(args.bundle, args.version)

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    card = args.bundle / "README.md"
    if not card.exists():
        card.write_text(
            MODEL_CARD_TEMPLATE.format(repo_id=args.repo_id,
                                       evaluation=_evaluation_section(manifest)),
            encoding="utf-8")

    api.create_repo(repo_id=args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)

    # upload_folder sweeps the whole directory, so manifest.json travels with the
    # weights. The check above guarantees it is there for a v2 bundle.
    commit = api.upload_folder(
        folder_path=str(args.bundle),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Publish {args.version}",
    )
    sha = commit.oid  # immutable; this is what scgnn-api pins against

    try:
        api.create_tag(repo_id=args.repo_id, tag=args.version, revision=sha,
                       repo_type="model")
    except Exception as exc:  # tag may already exist; not fatal
        print(f"note: could not create tag {args.version!r}: {exc}", file=sys.stderr)

    print(f"\nPublished to https://huggingface.co/{args.repo_id}")
    print(f"commit SHA (pin this): {sha}")
    print(f"version tag:           {args.version}")
    print("\nIn scgnn-api, pin the immutable revision, e.g.:")
    print(f'    REPO_ID  = "{args.repo_id}"')
    print(f'    REVISION = "{sha}"   # not the tag')
    if manifest:
        print(f"\nThe API will read manifest.json and apply the per-class "
              f"thresholds {[round(float(t), 2) for t in manifest['thresholds']]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())