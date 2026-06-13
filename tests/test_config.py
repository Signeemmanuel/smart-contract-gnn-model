"""Config `extends` inheritance is resolved (the silent-defaults defect)."""
from pathlib import Path
from training.config import load_config


def test_extends_merges_base_with_child_override(tmp_path: Path):
    (tmp_path / "base.yaml").write_text("seed: 42\nhid: 128\nconv: sage\n")
    child = tmp_path / "gat.yaml"
    child.write_text("extends: base.yaml\nconv: gat\n")
    cfg = load_config(child)
    assert cfg["conv"] == "gat"      # child overrides
    assert cfg["hid"] == 128         # inherited from base
    assert cfg["seed"] == 42
    assert "extends" not in cfg


def test_plain_config_loads_unchanged(tmp_path: Path):
    p = tmp_path / "plain.yaml"
    p.write_text("a: 1\nb: two\n")
    assert load_config(p) == {"a": 1, "b": "two"}
