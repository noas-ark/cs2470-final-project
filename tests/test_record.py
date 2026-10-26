import json

import pytest

from s1.record import RunRecord, summarize


def test_summarize_keys_and_values():
    m = summarize("score", [float(x) for x in range(101)])
    assert m == {
        "score": 50.0,
        "score_std": pytest.approx(29.15, abs=0.01),
        "score_p90": 90.0,
        "score_p95": 95.0,
        "score_p99": 99.0,
    }


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize("score", [])


def test_roundtrip_matches_layout(tmp_path):
    rec = RunRecord("r1", {"gpu": "a100"}, {"model": "openjev1"}, {"b": {"x": 1.0}})
    path = rec.save(tmp_path)
    assert set(json.loads(path.read_text())["r1"]) == {"invariants", "params", "results", "meta"}
    assert RunRecord.load(path) == rec


def test_grid_is_written_only_when_non_empty(tmp_path):
    plain = RunRecord("r1", {"gpu": "a100"}, {"model": "m"}, {"b": {"x": 1.0}}, {"k": 1})
    path = plain.save(tmp_path)
    assert "grid" not in json.loads(path.read_text())["r1"]
    assert RunRecord.load(tmp_path / "r1.json").grid == {}

    gridded = RunRecord("r2", {}, {"model": "m"}, grid={"state_tokens": [32, 64]})
    path = gridded.save(tmp_path)
    assert json.loads(path.read_text())["r2"]["grid"] == {"state_tokens": [32, 64]}
    assert RunRecord.load(path) == gridded
