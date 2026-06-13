"""plan_splits: de-dup, frozen curated test, firewall, rare-class presence."""
import numpy as np
from pathlib import Path
from training.data.build import plan_splits
from training.data.firewall import content_hash


def _write(d: Path, name: str, body: str) -> str:
    p = d / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_plan_splits_partitions_with_firewall(tmp_path: Path):
    # 8 wild contracts (one a duplicate of a curated one), 6 curated.
    wild_paths, wild_labels = {}, {}
    for i in range(8):
        cid = f"w{i}"
        wild_paths[cid] = _write(tmp_path, f"{cid}.sol", f"contract W{i} {{ uint x{i}; }}")
        wild_labels[cid] = [int(i % 2 == 0), 0, 0, int(i % 3 == 0), 0]
    # Make w0 a content-duplicate of curated c0 so de-dup must drop it.
    Path(wild_paths["w0"]).write_text("contract Dup {}", encoding="utf-8")

    curated = {}
    for i in range(6):
        cid = f"c{i}"
        body = "contract Dup {}" if i == 0 else f"contract C{i} {{ uint y{i}; }}"
        # ensure arithmetic (rare) appears so the stratified test split keeps it
        y = [0, 0, int(i in (0, 1)), 0, 0]
        curated[cid] = {"path": _write(tmp_path, f"{cid}.sol", body), "y": y, "lines": [i + 1]}

    plan = plan_splits(wild_paths, wild_labels, curated,
                       val_frac=0.25, test_frac=0.34, seed=1)

    by_split = {s: [it.cid for it in plan.items if it.split == s] for s in ("train", "val", "test")}
    # w0 dropped as a curated duplicate.
    assert "w0" not in {it.cid for it in plan.items}
    assert plan.counts["wild_dedup_removed"] == 1
    # test set is curated-only and non-empty; train carries the curated remainder + wild.
    assert by_split["test"] and all(c.startswith("c") for c in by_split["test"])
    assert any(c.startswith("c") for c in by_split["train"])  # curated remainder in train
    assert any(c.startswith("w") for c in by_split["train"])  # wild in train
    # firewall holds: no train contract shares content with any test contract.
    tr = {content_hash(Path(it.path).read_text()) for it in plan.items if it.split == "train"}
    te = {content_hash(Path(it.path).read_text()) for it in plan.items if it.split == "test"}
    assert tr.isdisjoint(te)
    # gold lines preserved on curated test items.
    assert all(it.gold_lines for it in plan.items if it.split == "test")
