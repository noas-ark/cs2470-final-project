"""Where a run executes: job id, orchestrator, git commit, log and model directories.

One place for every environment lookup that used to assume the HUIT Slurm cluster, so the same code
runs under Slurm, on Modal, or locally. Torch-free and cheap: safe to import anywhere.

Environment variables read:
    SLURM_JOB_ID      set by Slurm inside a job
    MODAL_TASK_ID     set by Modal inside a container
    S1_GIT_COMMIT     commit to record when there is no .git (a Modal image); the launcher sets it
                      from the local checkout, "-dirty" suffix included
    S1_LOG_DIR        where server logs go (vLLM, shim, ncu); default ./logs
    MODELS_DIR        model weights; default /shared/home/$USER/models (the cluster layout)

File paths in configs (tokenizer files and the like) go through `resolve_path`, so a config can write
`$MODELS_DIR/<model>/tokenizer.json` and run unchanged on Slurm and Modal.

Record fields (names are fixed here so every workstream writes the same keys):
    meta[META_ORCHESTRATOR]       "slurm" | "modal" | "local", on every new record
    meta["slurm_job_id"]          unchanged: Slurm runs only
    meta[META_MODAL_TASK_ID]      Modal runs only
    meta[META_CPU_MODEL]          host CPU model string, when it can be read
    meta[META_CPU_COUNT]          CPUs this process can see (os.sched_getaffinity). On Modal this is more than
                                  requested: `cpu=` is a guaranteed share, not a cap
    meta[META_CPUS_REQUESTED]     CPUs the job asked for: $SLURM_CPUS_PER_TASK, else $S1_CPUS_REQUESTED
                                  (set by the Modal launcher from [launch] cpus)
    meta[META_GPU_DRIVER]         NVIDIA driver version, when nvidia-smi is present
    meta[META_GPU_POWER_LIMIT_W]  enforced power limit of GPU 0 in watts, when present
    meta[META_NVIDIA_SMI_Q_LOG]   path of the saved `nvidia-smi -q` output (the text stays out of the record)
    invariants[INV_PLATFORM]      "modal" on Modal runs only; absent means the HUIT cluster, so Slurm
                                  runs keep exactly the invariants they had before
Hardware description goes in `meta`, not `invariants`: it describes the machine, it is not an experiment knob,
and putting it in invariants would split new Slurm runs from the existing ones in the report.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Literal

type Orchestrator = Literal["slurm", "modal", "local"]

META_ORCHESTRATOR = "orchestrator"
META_MODAL_TASK_ID = "modal_task_id"
META_CPU_MODEL = "cpu_model"
META_CPU_COUNT = "cpu_count"
META_CPUS_REQUESTED = "cpus_requested"
META_GPU_DRIVER = "gpu_driver"
META_GPU_POWER_LIMIT_W = "gpu_power_limit_w"
META_NVIDIA_SMI_Q_LOG = "nvidia_smi_q_log"
INV_PLATFORM = "platform"


def orchestrator() -> Orchestrator:
    """Which scheduler this process runs under."""
    if os.environ.get("SLURM_JOB_ID", ""):
        return "slurm"
    if os.environ.get("MODAL_TASK_ID", ""):
        return "modal"
    return "local"


def job_id() -> str:
    """A short id for log names: the Slurm job id, else the Modal task id, else "local"."""
    return os.environ.get("SLURM_JOB_ID", "") or os.environ.get("MODAL_TASK_ID", "") or "local"


def git_commit() -> str:
    """$S1_GIT_COMMIT if set, else `<short sha>[-dirty]` of the checkout, else "unknown"."""
    if pinned := os.environ.get("S1_GIT_COMMIT", ""):
        return pinned
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


def log_dir() -> Path:
    """$S1_LOG_DIR, default ./logs (not created here)."""
    return Path(os.environ.get("S1_LOG_DIR", "logs"))


def models_dir() -> Path:
    """$MODELS_DIR, default /shared/home/$USER/models."""
    return Path(os.environ.get("MODELS_DIR", f"/shared/home/{os.environ.get('USER', '')}/models"))


_MODELS_VAR = re.compile(r"\$(?:MODELS_DIR\b|\{MODELS_DIR\})")


def resolve_path(value: str) -> Path:
    """A path from a config: `$MODELS_DIR` / `${MODELS_DIR}` become `models_dir()` (set or not), then
    other `$VARS` and `~` are expanded. Write model files as `$MODELS_DIR/...` so one config runs on every
    orchestrator."""
    portable = _MODELS_VAR.sub(lambda _: str(models_dir()), value)
    return Path(os.path.expandvars(portable)).expanduser()


def cpus_requested() -> int | None:
    """CPUs the job asked for: $SLURM_CPUS_PER_TASK under Slurm, $S1_CPUS_REQUESTED on Modal, else None."""
    for var in ("SLURM_CPUS_PER_TASK", "S1_CPUS_REQUESTED"):
        raw = os.environ.get(var, "")
        if raw.strip().isdigit():
            return int(raw)
    return None
