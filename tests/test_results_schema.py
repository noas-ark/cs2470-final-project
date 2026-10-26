"""Every committed results/*.json must look like something the runner wrote, byte for byte.

These checks make hand-written or hand-edited records fail. They cannot prove a record came from a real
run: someone determined could forge every field.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from s1 import config
from s1.record import RunRecord

ROOT = Path(__file__).resolve().parent.parent
RECORDS = sorted((ROOT / "results").glob("*.json"))
RUN_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{8}")
META_KEYS = ("started", "git_commit", "host", "duration_s", "load_s")


def benchmarks(record: RunRecord) -> list[str]:
    """Benchmark names; auxiliary entries like "jeval/nvidia_smi" carry a slash."""
    return [name for name in record.results if "/" not in name]


def test_records_exist() -> None:
    assert RECORDS, f"no run records found in {ROOT / 'results'}"


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_record_has_required_fields(path: Path) -> None:
    record = RunRecord.load(path)
    assert record.params.get("model") not in (None, ""), "params.model missing"
    assert record.invariants.get("gpu") not in (None, ""), "invariants.gpu missing"
    assert benchmarks(record), "no benchmark in results"
    assert record.meta.get("git_commit") not in (None, ""), "meta.git_commit missing"


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_record_is_canonical(path: Path) -> None:
    """Re-serializing exactly as RunRecord.save does must reproduce the file (catches hand edits)."""
    record = RunRecord.load(path)
    assert path.read_text() == json.dumps(record.to_json(), indent=4) + "\n"


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_run_id_matches_filename_and_format(path: Path) -> None:
    record = RunRecord.load(path)
    assert record.run_id == path.stem
    assert RUN_ID.fullmatch(record.run_id), f"run id {record.run_id!r} is not YYYYMMDD-HHMMSS-<8 hex>"


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_gpu_was_resolved(path: Path) -> None:
    assert RunRecord.load(path).invariants["gpu"] != "auto"


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_meta_written_by_runner(path: Path) -> None:
    record = RunRecord.load(path)
    for key in META_KEYS:
        assert key in record.meta, f"meta.{key} missing"
    for name in benchmarks(record):
        assert f"{name}_s" in record.meta, f"meta.{name}_s missing"


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_gpu_runs_have_nvidia_smi(path: Path) -> None:
    record = RunRecord.load(path)
    if record.invariants["gpu"] == "cpu":
        return
    for name in benchmarks(record):
        assert f"{name}/nvidia_smi" in record.results, f"{name}/nvidia_smi missing"


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_config_exists_and_covers_benchmarks(path: Path) -> None:
    record = RunRecord.load(path)
    cfg = record.meta.get("config")
    if cfg is None:
        return
    cfg_path = ROOT / str(cfg)
    assert cfg_path.is_file(), f"meta.config {cfg} does not exist"
    declared = {b for spec in config.load(cfg_path) for b in spec.benchmarks}
    assert set(benchmarks(record)) <= declared, f"{cfg} does not list {benchmarks(record)}"
