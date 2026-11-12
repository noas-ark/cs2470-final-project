"""CPU vs GPU share of a decision's latency from Nsight Systems (`s1 nsys`), and the API-name rules that the
in-process torch.profiler trace (benchmarks/batch_sweep.py) shares with it.

`s1 nsys CONFIG` runs each model of a `batch` config under

    nsys profile --trace=cuda,osrt --sample=<sample> --capture-range=cudaProfilerApi --capture-range-end=stop

with `batch_nsys = N` set, so the benchmark answers N batch-1 decisions inside one CUDA profiler range
(cudaProfilerStart/Stop) and records that region's host wall time. `nsys stats` then writes the
`cuda_api_sum` and `cuda_gpu_kern_sum` reports as CSV, and `summarize` turns them into, per decision:

    batch_nsys_kernel_ms        summed GPU kernel time (what the GPU spent computing)
    batch_nsys_kernels          kernels launched (a CUDA graph's kernels count one by one)
    batch_nsys_launch_api_ms    host time in kernel and graph launch calls
    batch_nsys_sync_ms          host time blocked in synchronize calls (waiting for the GPU)
    batch_nsys_gpu_share        kernel time / wall time: the GPU's share of the latency
    batch_nsys_cpu_share        1 - gpu_share: host work, launch gaps and idle GPU between kernels

Kernel time on one stream never overlaps, so gpu_share <= 1 for these single-stream models. nsys adds a little
host time to every API call it traces, so wall time under nsys is slightly above an untraced run's.

Torch-free: only parses `nsys stats --format csv` output.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence

from .record import Metrics

REPORTS = ("cuda_api_sum", "cuda_gpu_kern_sum")
SAMPLES = ("process-tree", "none")  # nsys --sample: CPU IP/backtrace sampling, or off
META_SAMPLE = "nsys_sample"
LAUNCH_HINTS = ("LaunchKernel", "GraphLaunch", "LaunchCooperativeKernel")
SYNC_HINT = "Synchronize"


def is_launch(name: str) -> bool:
    """A CUDA runtime/driver call that launches GPU work (one kernel, or a whole CUDA graph)."""
    return any(h in name for h in LAUNCH_HINTS)


def is_sync(name: str) -> bool:
    """A call that blocks the host until the GPU catches up (stream/event/device synchronize)."""
    return SYNC_HINT in name


def profile_args(report: str, sample: str = "process-tree") -> tuple[str, ...]:
    """`nsys profile` arguments (before the command): CUDA + OS runtime trace, CPU sampling, and only the
    cudaProfilerStart/Stop range, written to `report` (.nsys-rep is appended by nsys)."""
    if sample not in SAMPLES:
        raise ValueError(f"nsys sample must be one of {SAMPLES}, not {sample!r}")
    return (
        "profile",
        "--trace=cuda,osrt",
        "--cuda-graph-trace=node",  # a graph's kernels one by one; the default traces a graph as one range
        f"--sample={sample}",
        f"--cpuctxsw={'process-tree' if sample != 'none' else 'none'}",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        "-o",
        report,
    )


def stats_args(report_file: str, out_base: str) -> tuple[str, ...]:
    """`nsys stats` arguments: both reports as CSV files `<out_base>_<report>.csv`."""
    reports: list[str] = []
    for r in REPORTS:
        reports += ["--report", r]
    return ("stats", *reports, "--format", "csv", "--force-export=true", "--output", out_base, report_file)


def _col(header: Sequence[str], *names: str) -> int:
    for i, h in enumerate(header):
        if any(n.lower() in h.lower() for n in names):
            return i
    raise ValueError(f"nsys CSV has none of {names} in its header {list(header)}")


def parse(text: str) -> list[tuple[str, float, int]]:
    """(name, total ns, calls or instances) per row of a `cuda_api_sum` or `cuda_gpu_kern_sum` CSV. Lines
    before the header (nsys notices) are skipped; an empty report (nothing in the range) is []."""
    start = text.find('"Time')
    if start < 0:
        start = text.find("Time (%)")
    if start < 0:
        return []
    rows = list(csv.reader(io.StringIO(text[start:])))
    header = rows[0]
    name, total, count = (
        _col(header, "Name"),
        _col(header, "Total Time"),
        _col(header, "Num Calls", "Instances", "Count"),
    )
    out = []
    for r in rows[1:]:
        if len(r) != len(header):
            continue
        try:
            out.append((r[name], float(r[total].replace(",", "")), int(float(r[count].replace(",", "")))))
        except ValueError:
            continue
    return out


def summarize(
    api: Sequence[tuple[str, float, int]], kernels: Sequence[tuple[str, float, int]], run: Mapping[str, float]
) -> Metrics:
    """Per-decision CPU/GPU split of the profiled region; `run` holds the benchmark's `batch_nsys_n` and
    `batch_nsys_wall_ms` (the region's host wall time)."""
    n = run.get("batch_nsys_n", 0.0)
    wall = run.get("batch_nsys_wall_ms", 0.0)
    if n <= 0 or wall <= 0:
        raise ValueError("the run recorded no nsys region (batch_nsys_n / batch_nsys_wall_ms)")
    kernel_ms = sum(t for _, t, _ in kernels) / 1e6
    launches = sum(c for name, _, c in api if is_launch(name))
    launch_ms = sum(t for name, t, _ in api if is_launch(name)) / 1e6
    sync_ms = sum(t for name, t, _ in api if is_sync(name)) / 1e6
    api_ms = sum(t for _, t, _ in api) / 1e6
    share = kernel_ms / wall
    return {
        "batch_nsys_cpu_share": 1.0 - share,
        "batch_nsys_gpu_share": share,
        "batch_nsys_wall_ms_per_decision": wall / n,
        "batch_nsys_kernel_ms": kernel_ms / n,
        "batch_nsys_kernels": sum(c for _, _, c in kernels) / n,
        "batch_nsys_launches": launches / n,
        "batch_nsys_launch_api_ms": launch_ms / n,
        "batch_nsys_sync_ms": sync_ms / n,
        "batch_nsys_api_ms": api_ms / n,
        "batch_nsys_launch_share": launch_ms / wall,
        "batch_nsys_sync_share": sync_ms / wall,
    }


def top_kernels(kernels: Sequence[tuple[str, float, int]], top: int = 8) -> str:
    """The `top` kernels by summed time, as '<share> <name>' (for `meta`)."""
    whole = sum(t for _, t, _ in kernels)
    ranked = sorted(kernels, key=lambda k: k[1], reverse=True)[:top]
    return "; ".join(f"{t / whole:.1%} x{c} {name[:100]}" for name, t, c in ranked) if whole > 0 else ""
