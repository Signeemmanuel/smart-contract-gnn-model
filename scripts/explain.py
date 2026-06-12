#!/usr/bin/env python3
"""Analyse one .sol file end to end and print the schema result. Needs the stack."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("contract", help="Path to a .sol file.")
    ap.add_argument("--repo-id", required=True, help="Hub model repo id.")
    ap.add_argument("--revision", required=True, help="Immutable commit SHA to pin.")
    args = ap.parse_args()

    from scgnn.inference import analyze_source, load_model

    loaded = load_model(args.repo_id, args.revision)
    src = open(args.contract, encoding="utf-8", errors="ignore").read()
    print(json.dumps(analyze_source(loaded, src), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
