"""Queue `s1 <args>` on the HUIT Slurm cluster with `sbatch`.

The command is exactly the one `s1 submit` built before launchers existed (tests/test_launch.py pins it).
Slurm reads `gpus`, `cpus`, `mem` and `time` from `Resources` and these extras from [launch.slurm]:

    partition   -p; omitted means the cluster default
    gpu_type    GRES type, e.g. "l4" (gpu-spec types its GPUs): --gres=gpu:l4:N

`Resources.gpu` (the portable name) is not used here: Slurm GRES labels are unreliable on this cluster
(AGENTS.md), so name the partition/type explicitly and set `require_gpu` in [invariants].
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .base import Resources

EXTRAS = ("partition", "gpu_type")


def sbatch_command(cfg: Path, resources: Resources, s1_args: Sequence[str]) -> list[str]:
    """The argv of the `sbatch` call that runs `uv run --extra openjev s1 <s1_args>`."""
    unknown = sorted(set(resources.extras) - set(EXTRAS))
    if unknown:
        raise ValueError(f"[launch.slurm] unknown key `{unknown[0]}` (known: {', '.join(EXTRAS)})")
    partition = resources.extras.get("partition")
    gpu_type = resources.extras.get("gpu_type")
    cmd = ["sbatch", "-J", f"s1-{cfg.stem}", "-c", str(resources.cpus), f"--mem={resources.mem}"]
    cmd += ["-t", resources.time, "-o", "logs/%j.out"]
    if partition is not None:
        cmd += ["-p", str(partition)]
    if resources.gpus:
        cmd += [f"--gres=gpu:{'' if gpu_type is None else f'{gpu_type}:'}{resources.gpus}"]
    cmd += ["--wrap", shlex.join(["uv", "run", "--extra", "openjev", "s1", *s1_args])]
    return cmd


class SlurmLauncher:
    def submit(
        self, cfg: Path, resources: Resources, s1_args: Sequence[str], *, dry_run: bool = False
    ) -> str:
        cmd = sbatch_command(cfg, resources, s1_args)
        print(shlex.join(cmd))
        if dry_run:
            return ""
        Path("logs").mkdir(exist_ok=True)
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        print(out, end="")
        found = re.search(r"Submitted batch job (\d+)", out)
        return str(found.group(1)) if found else out.strip()
