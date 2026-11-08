"""Experiment configs (TOML) and their expansion into individual runs.

benchmarks = ["jev_bench"]
[invariants]   # fixed for every run
[params]       # fixed for every run
[sweep]        # each key is a list; one run per element of the cartesian product, merged into params
[grid]         # each key is a list; NOT expanded: the run carries them all and one benchmark
               # (e.g. shape_sweep) walks the grid inside a single model load
[launch]       # where `s1 submit` queues it (Slurm or Modal), see s1/launch/base.py. Read only by
               # load_launch: never part of a RunSpec, so it changes no params or invariants
"""

from __future__ import annotations

import itertools
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from .launch.base import LaunchTarget, Resources
from .record import Scalar

ORCHESTRATORS: tuple[LaunchTarget, ...] = ("slurm", "modal")


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


def target(value: object, where: str) -> LaunchTarget:
    """Validate an orchestrator name ([launch].orchestrator or `s1 submit --on`)."""
    for name in ORCHESTRATORS:
        if value == name:
            return name
    raise ValueError(f"{where}: orchestrator must be one of {', '.join(ORCHESTRATORS)}, got {value!r}")


_LAUNCH_INTS = ("gpus", "cpus")
_LAUNCH_STRS = ("gpu", "mem", "time")


def load_launch(path: Path, orchestrator: LaunchTarget | None = None) -> Resources | None:
    """The config's [launch] table as `Resources`, or None when it has none.

    `orchestrator` overrides [launch].orchestrator (`s1 submit --on`). Extras come from the
    [launch.<orchestrator>] table of whichever orchestrator wins, so one config can carry both."""
    raw = tomllib.loads(path.read_text())
    if "launch" not in raw:
        return None
    table = raw["launch"]
    if not isinstance(table, dict):
        raise TypeError(f"{path}: [launch] must be a table")
    if "orchestrator" not in table:
        raise ValueError(f"{path}: [launch] needs `orchestrator` ({' or '.join(ORCHESTRATORS)})")
    declared = target(table["orchestrator"], f"{path}: [launch]")
    chosen = orchestrator or declared
    res = Resources(orchestrator=chosen)
    for key, value in table.items():
        if key == "orchestrator":
            continue
        if key in ORCHESTRATORS:
            extras = _scalars(value, f"launch.{key}")
            if key == chosen:
                res = replace(res, extras=extras)
        elif key in _LAUNCH_INTS:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TypeError(f"{path}: [launch] {key} must be a non-negative integer, got {value!r}")
            res = replace(res, **{key: value})
        elif key in _LAUNCH_STRS:
            if not isinstance(value, str) or not value:
                raise TypeError(f"{path}: [launch] {key} must be a non-empty string, got {value!r}")
            res = replace(res, **{key: value})
        else:
            known = ", ".join(("orchestrator", *_LAUNCH_INTS, *_LAUNCH_STRS, *ORCHESTRATORS))
            raise ValueError(f"{path}: [launch] unknown key `{key}` (known: {known})")
    return res
