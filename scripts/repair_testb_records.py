#!/usr/bin/env python3
"""Surgical repair of Test B records corrupted by the cid collision.

Background: build keyed graphs by bare contract id, and Curated and Wild both
name contracts by address. Six Test B contracts share an address with a
DIFFERENT-content Wild twin in train; the twin's hash overwrote theirs, so
those Test B rows were encoded from the WRONG graphs with the WRONG labels.
``training/data/build.py`` is fixed (per-item keys); this script repairs the
existing builds without rebuilding anything else:

  1. detects every test_b index row whose record label vector differs from the
     frozen manifest (the authoritative expert truth);
  2. re-extracts each affected contract from its true Curated source (AST +
     CFG with data-flow), writing the shared cache under the CORRECT hash;
  3. re-encodes it per processed arm with that arm's FROZEN featurisers
     (feature_config.json + pca.joblib), stripping data-flow edges for the
     nodf arm, and writes the record under the correct hash;
  4. rewrites the test_b index entries to point at the corrected records;
  5. verifies record truth == manifest truth for every test_b row, and asserts
     test_a is clean (it is structurally immune, but we check, not assume).

Nothing about training, validation, checkpoints or thresholds is touched: the
repair only makes the expert test rows be the expert contracts. Re-run
inference afterwards (scripts/reeval_testb.py).

Usage
-----
    PYTHONPATH=. python scripts/repair_testb_records.py \
        --manifest data/testsets/test_b.csv \
        --processed data/processed_df data/processed_nodf \
        --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scgnn.schema import FLAWS


def load_manifest(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        return {r["contract_id"]: r for r in csv.DictReader(fh)}


def detect_mismatches(processed: Path, manifest: dict[str, dict],
                      split: str) -> list[str]:
    import torch

    idx_path = processed / f"{split}_index.json"
    entries = json.loads(idx_path.read_text(encoding="utf-8"))
    bad = []
    for e in entries:
        row = manifest.get(e["id"])
        if row is None:
            continue
        y_rec = (torch.load(e["path"], weights_only=True)["y"] >= 0.5).int().tolist()
        y_man = [int(row[f]) for f in FLAWS]
        if y_rec != y_man:
            bad.append(e["id"])
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/testsets/test_b.csv")
    ap.add_argument("--manifest-a", default="data/testsets/test_a.csv")
    ap.add_argument("--processed", nargs="+",
                    default=["data/processed_df", "data/processed_nodf"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extract-timeout", type=float, default=300)
    args = ap.parse_args()

    import joblib
    import torch

    from scgnn.extraction.extract import extract_contract
    from scgnn.extraction.features import (CodeBERTEmbedder, FeatureConfig,
                                           FeatureEncoder)
    from scgnn.extraction.solc import installed_solc_binaries, installed_solc_full
    from training.data.build import _time_limit
    from training.data.firewall import content_hash

    manifest = load_manifest(Path(args.manifest))
    manifest_a = load_manifest(Path(args.manifest_a)) if Path(args.manifest_a).exists() else {}

    # 1. detect (union across arms; they were built from the same collided keys)
    bad_ids: set[str] = set()
    for p in args.processed:
        found = detect_mismatches(Path(p), manifest, "test_b")
        print(f"{p}: {len(found)} corrupted test_b record(s)")
        bad_ids.update(found)
        if manifest_a:
            bad_a = detect_mismatches(Path(p), manifest_a, "test_a")
            if bad_a:
                raise SystemExit(f"UNEXPECTED: test_a mismatches in {p}: {bad_a}; "
                                 f"investigate before repairing.")
    print(f"test_a verified clean; repairing {len(bad_ids)} test_b contract(s): "
          f"{sorted(bad_ids)}")
    if not bad_ids:
        print("nothing to repair.")
        return 0

    # 2. re-extract each affected contract from its TRUE Curated source
    binaries = installed_solc_binaries()
    full_binaries = installed_solc_full()
    graphs: dict[str, tuple] = {}
    for cid in sorted(bad_ids):
        src_path = manifest[cid]["path"]
        h = content_hash(Path(src_path).read_text(encoding="utf-8", errors="ignore"))
        assert h == manifest[cid]["chash"], (
            f"{cid}: source hash {h[:12]} != manifest chash "
            f"{manifest[cid]['chash'][:12]}; the source file moved or changed.")
        with _time_limit(args.extract_timeout):
            ast, cfg = extract_contract(src_path, solc_binary=None,
                                        binaries=binaries,
                                        full_binaries=full_binaries,
                                        with_data_flow=True)
        graphs[cid] = (ast, cfg, h)
        print(f"  extracted {cid} -> {h[:12]} "
              f"(cfg data-flow edges: {cfg.n_data_flow_edges})")

    # write the shared cache under the correct hashes (raw/ symlinks to it)
    cache_dir = Path(args.processed[0]) / "raw"
    for cid, (ast, cfg, h) in graphs.items():
        (cache_dir / f"{h}.json").write_text(
            json.dumps({"ast": ast.to_dict(), "cfg": cfg.to_dict()}),
            encoding="utf-8")

    # 3-4. re-encode per arm with the FROZEN featurisers; fix the index
    embedder = CodeBERTEmbedder(device=args.device)
    for p in args.processed:
        processed = Path(p)
        feat_cfg = FeatureConfig.from_json(str(processed / "feature_config.json"))
        pca_path = processed / "pca.joblib"
        pca = joblib.load(pca_path) if pca_path.exists() else None
        encoder = FeatureEncoder(feat_cfg, embedder if pca is not None else None, pca)
        report = json.loads((processed / "build_report.json").read_text(encoding="utf-8"))
        with_df = bool(report.get("with_data_flow", True))

        idx_path = processed / "test_b_index.json"
        entries = json.loads(idx_path.read_text(encoding="utf-8"))
        fixed = 0
        for e in entries:
            if e["id"] not in graphs:
                continue
            ast, cfg, h = graphs[e["id"]]
            cfg_use = cfg if with_df else cfg.without_data_flow()
            ax, ai = encoder.encode_array(ast)
            cx, ci = encoder.encode_array(cfg_use)
            y = torch.tensor([float(manifest[e["id"]][f]) for f in FLAWS])
            rp = processed / "records" / f"{h}.pt"
            torch.save({"ast_x": torch.from_numpy(ax),
                        "ast_edge_index": torch.from_numpy(ai),
                        "cfg_x": torch.from_numpy(cx),
                        "cfg_edge_index": torch.from_numpy(ci),
                        "y": y}, rp)
            e["path"] = str(rp)
            fixed += 1
        idx_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        print(f"{p}: re-encoded {fixed} record(s) "
              f"({'with' if with_df else 'without'} data-flow edges)")

        # 5. verify the whole split is now truth-consistent
        residue = detect_mismatches(processed, manifest, "test_b")
        if residue:
            raise SystemExit(f"REPAIR INCOMPLETE in {p}: {residue}")
        print(f"{p}: test_b record truth == manifest truth for all "
              f"{len(entries)} rows \u2713")

    print("\nrepair complete. Next: PYTHONPATH=. python scripts/reeval_testb.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())