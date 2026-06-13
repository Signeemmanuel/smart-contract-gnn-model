"""Curated category->flaw mapping and gold-line assembly."""
from training.data.curated import CATEGORY_MAP, parse_vulnerabilities
from scgnn.schema import FLAW_INDEX, N_FLAWS


def test_category_map_targets_only_canonical_flaws():
    from scgnn.schema import FLAWS
    assert set(CATEGORY_MAP.values()) <= set(FLAWS)


def test_parse_single_category_entry():
    recs = [{"path": "dataset/reentrancy/simple_dao.sol",
             "category": "reentrancy", "lines": [25, 27]}]
    out = parse_vulnerabilities(recs)
    item = out["simple_dao"]
    assert item["y"][FLAW_INDEX["reentrancy"]] == 1
    assert sum(item["y"]) == 1
    assert item["lines"] == [25, 27]


def test_parse_nested_multi_category_and_unknown():
    recs = [{
        "name": "multi.sol",
        "vulnerabilities": [
            {"category": "access_control", "lines": [5]},
            {"category": "arithmetic", "lines": [9, 9]},
            {"category": "front_running", "lines": [12]},  # outside our five
        ],
    }]
    out = parse_vulnerabilities(recs)["multi"]
    assert out["y"][FLAW_INDEX["access_control"]] == 1
    assert out["y"][FLAW_INDEX["arithmetic"]] == 1
    assert sum(out["y"]) == 2                 # front_running contributes nothing
    assert out["lines"] == [5, 9]             # de-duped + sorted


def test_load_curated_labels_from_folder_and_lines_from_json(tmp_path):
    """load_curated takes the label from the category folder, lines from JSON."""
    import json as _json
    from training.data.curated import load_curated
    from scgnn.schema import FLAW_INDEX

    ds = tmp_path / "dataset"
    (ds / "reentrancy").mkdir(parents=True)
    (ds / "bad_randomness").mkdir(parents=True)        # outside our five
    (ds / "reentrancy" / "dao.sol").write_text("contract D {}", encoding="utf-8")
    (ds / "bad_randomness" / "lotto.sol").write_text("contract L {}", encoding="utf-8")
    (tmp_path / "vulnerabilities.json").write_text(_json.dumps([
        {"name": "dao.sol", "path": "dataset/reentrancy/dao.sol",
         "vulnerabilities": [{"category": "reentrancy", "lines": [25, 27]}]},
    ]), encoding="utf-8")

    cur = load_curated(tmp_path)
    assert cur["dao"]["y"][FLAW_INDEX["reentrancy"]] == 1
    assert cur["dao"]["lines"] == [25, 27]
    # a contract whose folder is outside our five is an all-negative example
    assert sum(cur["lotto"]["y"]) == 0
