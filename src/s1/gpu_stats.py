"""Sample `nvidia-smi` in the background while a benchmark runs; summarize into metrics.

    with GpuSampler() as smi:
        ...
    smi.metrics()  # {"gpu_util_pct": ..., "gpu_util_pct_p90": ..., "gpu_mem_used_mb_max": ..., ...}

No GPU / no nvidia-smi -> empty metrics, so the same code runs on CPU-only machines.
"""

from __future__ import annotations

import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import TracebackType

from .record import Metrics, summarize

QUERY = "index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"


@dataclass(frozen=True)
class Sample:
    gpu: int
    util_pct: float
    mem_used_mb: float
    mem_total_mb: float
    power_w: float
    temp_c: float


def parse_line(line: str) -> Sample | None:
    """One `--format=csv,noheader,nounits` row; None for blank lines or `[N/A]` fields."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        return None
    try:
        util, mem, total, power, temp = (float(v) for v in parts[1:])
        return Sample(int(parts[0]), util, mem, total, power, temp)
    except ValueError:
        return None


def summarize_samples(samples: Sequence[Sample], duration_s: float) -> Metrics:
    if not samples:
        return {}
    gpus = sorted({s.gpu for s in samples})
    # each GPU's mean draw over the window, summed across GPUs
    mean_power = sum(statistics.fmean(s.power_w for s in samples if s.gpu == g) for g in gpus)
    return summarize("gpu_util_pct", [s.util_pct for s in samples]) | {
        "gpu_count": float(len(gpus)),
        "gpu_mem_used_mb_max": max(s.mem_used_mb for s in samples),
        "gpu_mem_used_mb_mean": statistics.fmean(s.mem_used_mb for s in samples),
        "gpu_mem_total_mb": max(s.mem_total_mb for s in samples),
        "gpu_power_w_total_mean": mean_power,
        "gpu_power_w_max": max(s.power_w for s in samples),
        "gpu_energy_wh": mean_power * duration_s / 3600,
        "gpu_temp_c_max": max(s.temp_c for s in samples),
        "gpu_samples": float(len(samples)),
        "gpu_sampled_s": duration_s,
    }


class GpuSampler:
    def __init__(self, interval_ms: int = 500, cmd: Sequence[str] | None = None) -> None:
        self.cmd = (
            list(cmd)
            if cmd is not None
            else [
                "nvidia-smi",
                f"--query-gpu={QUERY}",
                "--format=csv,noheader,nounits",
                f"-lms={interval_ms}",
            ]
        )
        self.samples: list[Sample] = []
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._t0 = self._t1 = 0.0

    def __enter__(self) -> GpuSampler:
        self._t0 = time.perf_counter()
        try:
            self._proc = subprocess.Popen(
                self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
        except OSError:
            return self
        self._thread = threading.Thread(target=self._read, args=(self._proc,), daemon=True)
        self._thread.start()
        return self

    def _read(self, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            if (s := parse_line(line)) is not None:
                self.samples.append(s)

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._t1 = time.perf_counter()
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def metrics(self) -> Metrics:
        return summarize_samples(self.samples, self._t1 - self._t0)


@dataclass
class PointMetrics:
    """Filled in when the `sample_point` block exits; empty if nvidia-smi is unavailable."""

    metrics: Metrics = field(default_factory=dict)


@contextmanager
def sample_point(
    prefix: str,
    keys: Sequence[str] | None = None,
    sampler: Callable[[], GpuSampler] = GpuSampler,
) -> Generator[PointMetrics]:
    """Sample one sweep point with its own `GpuSampler`; `point.metrics` is keyed `<prefix><metric>`.

        with sample_point("t128_q4_", keys=("gpu_util_pct",)) as point:
            ...  # run the point
        point.metrics  # {"t128_q4_gpu_util_pct": ...}

    `keys` restricts which of `summarize_samples`' metrics are kept (default: all).

    Caveats when reading per-point numbers:
      * `nvidia-smi` is polled every 500 ms (`GpuSampler(interval_ms=500)`), so a point needs to run for
        several seconds to have a meaningful number of samples. Check `<prefix>gpu_samples`.
      * Samples from every visible GPU are pooled: `gpu_util_pct` is the mean over all GPUs and all
        ticks, so a TP=4 run with one busy GPU reads as 25% and gpu_count says how many were pooled.
    """
    point = PointMetrics()
    with sampler() as smi:
        yield point
    point.metrics = {f"{prefix}{k}": v for k, v in smi.metrics().items() if keys is None or k in keys}
