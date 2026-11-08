"""The launcher contract: what `s1 submit` hands to Slurm or Modal. Frozen for milestones M1-M2.

`[launch]` in a config (parsed by s1.launch, workstream B) becomes a `Resources`; a `Launcher` for that
orchestrator (slurm.py, workstream B; modal_app.py, workstream C) queues `s1 <s1_args>` on those resources.

    [launch]
    orchestrator = "slurm"     # or "modal": the one-line switch
    gpu = "H100"               # "L4", "A10G", "A100-80GB", "H100", "H200", "B200"; omit for CPU
    gpus = 1
    cpus = 8
    mem = "32G"
    time = "01:00:00"
    [launch.slurm]             # extras for one orchestrator only (partition, gpu_type, ...)
    partition = "gpu-spec"

Changing these names or signatures needs the coordinator: other workstreams build against them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from ..record import Scalar

type LaunchTarget = Literal["slurm", "modal"]


@dataclass(frozen=True)
class Resources:
    orchestrator: LaunchTarget
    gpu: str | None = None  # portable GPU name; None = CPU only
    gpus: int = 1
    cpus: int = 8
    mem: str = "32G"  # Slurm syntax ("28G", "160G"); Modal converts to MiB
    time: str = "01:00:00"  # HH:MM:SS; Modal converts to a timeout in seconds
    extras: Mapping[str, Scalar] = field(default_factory=dict)  # the [launch.<orchestrator>] table


class Launcher(Protocol):
    def submit(
        self, cfg: Path, resources: Resources, s1_args: Sequence[str], *, dry_run: bool = False
    ) -> str:
        """Queue `s1 <s1_args>` (e.g. ["run", "configs/x.toml"]) and return a handle: the Slurm job id or
        the Modal function-call id. With dry_run, print what would run and return "" without queueing."""
        ...
