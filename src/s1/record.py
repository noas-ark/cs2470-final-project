"""Run records: one JSON file per run, keyed by run id.

    {"<run_id>": {"invariants": {...}, "params": {...}, "grid": {...},
                  "results": {"<bench>": {...}}, "meta": {...}}}

`invariants` describe the fixed setup (hardware, shapes), `params` what the experiment varies, `grid`
(only when non-empty) the axes a benchmark sweeps inside the run, `results` the metrics each benchmark
produced, `meta` bookkeeping (timing, git commit, slurm job).
"""

from __future__ import annotations

import json
import math
import statistics
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

type Scalar = str | int | float | bool
type Metrics = dict[str, float]

RESULTS_DIR = Path("results")


def new_run_id() -> str:
    """Sortable by start time, unique across concurrent jobs: 20261021-153012-a1b2c3d4."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"


@dataclass
class RunRecord:
    run_id: str
    invariants: dict[str, Scalar]
    params: dict[str, Scalar]
    results: dict[str, Metrics] = field(default_factory=dict)
    meta: dict[str, Scalar] = field(default_factory=dict)
    grid: dict[str, list[Scalar]] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        body = asdict(self)
        grid = body.pop("grid")
        if grid:  # absent when empty so records without a grid serialize as before
            body = {k: v for k, v in body.items() if k != "meta"} | {"grid": grid, "meta": body["meta"]}
        return {body.pop("run_id"): body}

    def save(self, results_dir: Path = RESULTS_DIR) -> Path:
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_json(), indent=4) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> RunRecord:
        ((run_id, body),) = json.loads(path.read_text()).items()
        return cls(
            run_id=run_id,
            invariants=body["invariants"],
            params=body["params"],
            results=body.get("results", {}),
            meta=body.get("meta", {}),
            grid=body.get("grid", {}),
        )


def load_all(results_dir: Path = RESULTS_DIR) -> list[RunRecord]:
    return [RunRecord.load(p) for p in sorted(results_dir.glob("*.json"))]


def _percentile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    pos = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def summarize(prefix: str, samples: Sequence[float]) -> Metrics:
    """Per-sample values -> {prefix, prefix_std, prefix_p90, prefix_p95, prefix_p99}."""
    if not samples:
        raise ValueError(f"no samples to summarize for {prefix!r}")
    xs = sorted(samples)
    return {
        prefix: statistics.fmean(xs),
        f"{prefix}_std": statistics.pstdev(xs),
        f"{prefix}_p90": _percentile(xs, 0.90),
        f"{prefix}_p95": _percentile(xs, 0.95),
        f"{prefix}_p99": _percentile(xs, 0.99),
    }
