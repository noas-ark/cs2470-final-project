"""Experiment configs (TOML) and their expansion into individual runs.

benchmarks = ["jev_bench"]
[invariants]   # fixed for every run
[params]       # fixed for every run
[sweep]        # each key is a list; one run per element of the cartesian product, merged into params
[grid]         # each key is a list; NOT expanded: the run carries them all and one benchmark
               # (e.g. shape_sweep) walks the grid inside a single model load
"""

from __future__ import annotations

import itertools
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .record import Scalar


@dataclass
class RunSpec:
    invariants: dict[str, Scalar]
    params: dict[str, Scalar]
    benchmarks: list[str]
    grid: dict[str, list[Scalar]] = field(default_factory=dict)


def _scalars(table: object, where: str) -> dict[str, Scalar]:
    if not isinstance(table, dict):
        raise TypeError(f"[{where}] must be a table")
    out: dict[str, Scalar] = {}
    for k, v in table.items():
        if not isinstance(v, str | int | float | bool):
            raise TypeError(f"[{where}] {k} must be a scalar, got {type(v).__name__}")
        out[str(k)] = v
    return out


def _axes(table: object, where: str, path: Path) -> dict[str, list[Scalar]]:
    if not isinstance(table, dict):
        raise TypeError(f"{path}: [{where}] must be a table")
    out: dict[str, list[Scalar]] = {}
    for k, vals in table.items():
        if not isinstance(vals, list) or not vals:
            raise ValueError(f"{path}: [{where}] {k} must be a non-empty list")
        out[str(k)] = list(_scalars(dict(enumerate(vals)), f"{where}.{k}").values())
    return out


def load(path: Path) -> list[RunSpec]:
    raw = tomllib.loads(path.read_text())
    benchmarks = raw.get("benchmarks", [])
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError(f"{path}: `benchmarks` must be a non-empty list")
    invariants = _scalars(raw.get("invariants", {}), "invariants")
    params = _scalars(raw.get("params", {}), "params")

    sweep = _axes(raw.get("sweep", {}), "sweep", path)
    axes = list(sweep.items())
    grid = _axes(raw.get("grid", {}), "grid", path)

    if "model" not in params and "model" not in sweep:
        raise ValueError(f"{path}: set `model` in [params] or [sweep]")

    combos = itertools.product(*(vals for _, vals in axes))
    return [
        RunSpec(
            invariants=dict(invariants),
            params=params | {k: v for (k, _), v in zip(axes, combo, strict=True)},
            benchmarks=[str(b) for b in benchmarks],
            grid={k: list(v) for k, v in grid.items()},
        )
        for combo in combos
    ]
