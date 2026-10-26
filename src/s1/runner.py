"""Execute a RunSpec: build the model, run each benchmark, write the record."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from . import models, registry
from .config import RunSpec
from .gpu_stats import GpuSampler
from .record import RESULTS_DIR, RunRecord, Scalar, new_run_id


def detect_gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "cpu"
    names = sorted({line.strip() for line in out.splitlines() if line.strip()})
    return ",".join(names) or "cpu"


def _git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def grid_settings(grid: dict[str, list[Scalar]]) -> dict[str, Scalar]:
    """Expose each grid axis to benchmarks as `grid_<axis>` = comma-joined values ("32,128,512")."""
    return {f"grid_{k}": ",".join(str(v) for v in vals) for k, vals in grid.items()}


def run(
    spec: RunSpec,
    *,
    config_path: Path | None = None,
    results_dir: Path = RESULTS_DIR,
    log: Callable[[str], None] = print,
    gpu_sampler: Callable[[], GpuSampler] = GpuSampler,
) -> RunRecord:
    invariants = dict(spec.invariants)
    if invariants.get("gpu", "auto") == "auto":
        invariants["gpu"] = detect_gpu()

    meta: dict[str, Scalar] = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "host": os.uname().nodename,
    }
    if config_path is not None:
        meta["config"] = str(config_path)
    if (job := os.environ.get("SLURM_JOB_ID")) is not None:
        meta["slurm_job_id"] = job

    rec = RunRecord(
        run_id=new_run_id(),
        invariants=invariants,
        params=dict(spec.params),
        meta=meta,
        grid={k: list(v) for k, v in spec.grid.items()},
    )
    log(f"[{rec.run_id}] params={rec.params}")

    settings = rec.invariants | rec.params | grid_settings(rec.grid)
    t0 = time.perf_counter()
    model = registry.get_model(str(spec.params["model"]))(settings)
    for key, val in models.spec_params(str(spec.params["model"])).items():
        rec.params.setdefault(key, val)  # resolved spec values (revision, options); config wins
    meta["load_s"] = round(time.perf_counter() - t0, 3)
    try:
        for name in spec.benchmarks:
            t = time.perf_counter()
            with gpu_sampler() as smi:
                rec.results[name] = registry.get_benchmark(name)(model, settings)
            meta[f"{name}_s"] = round(time.perf_counter() - t, 3)
            if gpu := smi.metrics():
                rec.results[f"{name}/nvidia_smi"] = gpu
            log(f"[{rec.run_id}] {name}: {rec.results[name]}")
    finally:
        if isinstance(model, registry.Closeable):
            model.close()

    meta["duration_s"] = round(time.perf_counter() - t0, 3)
    path = rec.save(results_dir)
    log(f"[{rec.run_id}] wrote {path}")
    return rec
