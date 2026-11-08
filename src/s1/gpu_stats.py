"""Sample `nvidia-smi` in the background while a benchmark runs; summarize into metrics.

    with GpuSampler() as smi:
        ...
    smi.metrics()  # {"gpu_util_pct": ..., "gpu_util_pct_p90": ..., "gpu_mem_used_mb_max": ..., ...}

No GPU / no nvidia-smi -> empty metrics, so the same code runs on CPU-only machines.

Two metric sets, chosen by the run's `gpu_metrics` setting (the runner applies it with `metrics_mode`):

  "basic" (default)  the pooled keys below, unchanged since the first records. Nothing else is added, so
                     basic runs write exactly what they always did.
  "detailed"         the basic keys PLUS the schema-1 keys below. Opt-in. Records made before it exist
                     simply do not have them; that is intentional, and `gpu_metrics_schema` (recorded in
                     `invariants`) says which key set a record was run with. Bump `SCHEMA` when the
                     detailed key set changes.

Basic keys (pooled over all GPUs and ticks), every record:
    gpu_util_pct, gpu_util_pct_{std,p90,p95,p99}, gpu_count, gpu_mem_used_mb_{max,mean},
    gpu_mem_total_mb, gpu_power_w_total_mean, gpu_power_w_max, gpu_energy_wh, gpu_temp_c_max,
    gpu_samples, gpu_sampled_s

Detailed keys, schema 1 (`g` is the GPU index nvidia-smi reports; a key is absent when its source is
missing or `[N/A]`, never zero-filled):
    per GPU, from `nvidia-smi --query-gpu` every 500 ms:
        gpu{g}_util_pct, gpu{g}_util_pct_max            SM-busy % (utilization.gpu)
        gpu{g}_mem_util_pct                              memory-controller busy % (utilization.memory)
        gpu{g}_mem_used_mb_max
        gpu{g}_power_w, gpu{g}_power_w_max
        gpu{g}_clock_sm_mhz, gpu{g}_clock_mem_mhz        mean clocks (clocks.sm, clocks.mem)
        gpu{g}_throttle_frac                             share of samples with a performance-limiting
                                                         throttle reason active (power cap, HW/SW thermal,
                                                         HW slowdown, power brake; idle is not counted)
    per GPU, from `nvidia-smi dmon -s t` every 1 s (PCIe throughput, MB/s):
        gpu{g}_pcie_rx_mb_s, gpu{g}_pcie_rx_mb_s_max, gpu{g}_pcie_tx_mb_s, gpu{g}_pcie_tx_mb_s_max
    pooled:
        gpu_mem_util_pct, gpu_clock_sm_mhz, gpu_clock_sm_mhz_min, gpu_clock_mem_mhz, gpu_throttle_frac,
        gpu_throttle_mask_seen      OR of all throttle bitmasks seen (decode: nvidia-smi -q -d PERFORMANCE)
        gpu_util_pct_spread         max - min over GPUs of the per-GPU mean util (imbalance)
        gpu_pcie_rx_mb_s_total_mean, gpu_pcie_tx_mb_s_total_mean   sum over GPUs of the mean rate
        gpu_pcie_samples

The dmon stream is optional: if `nvidia-smi dmon` is unavailable the pcie keys are simply absent.
"""

from __future__ import annotations

import functools
import re
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import TracebackType

from .record import Metrics, summarize

QUERY = "index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"

SCHEMA = 1  # key set of gpu_metrics = "detailed"; bump when it changes
MODES = ("basic", "detailed")

# Optional --query-gpu columns appended in detailed mode, in the order they are requested. The throttle
# column is named `clocks_event_reasons.active` on recent drivers and `clocks_throttle_reasons.active`
# on older ones; `detail_fields` picks whichever this nvidia-smi lists.
DETAIL_FIELDS = ("utilization.memory", "clocks.sm", "clocks.mem")
THROTTLE_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")
# bits that mean "slower than it could be": SwPowerCap, HwSlowdown, SwThermal, HwThermal, HwPowerBrake
THROTTLE_LIMITING = 0x4 | 0x8 | 0x20 | 0x40 | 0x80

DMON_CMD = ("nvidia-smi", "dmon", "-s", "t", "-d", "1")

_mode: ContextVar[str] = ContextVar("s1_gpu_metrics", default="basic")


def resolve_mode(mode: str, schema: int = SCHEMA) -> str:
    """Validate the `gpu_metrics` / `gpu_metrics_schema` settings; returns the mode."""
    if mode not in MODES:
        raise ValueError(f"gpu_metrics must be one of {MODES}, got {mode!r}")
    if schema != SCHEMA:
        raise ValueError(f"gpu_metrics_schema {schema} is not supported; this code writes schema {SCHEMA}")
    return mode


@contextmanager
def metrics_mode(mode: str) -> Generator[None]:
    """Make every `GpuSampler()` created inside the block (including those in `sample_point`) use `mode`."""
    token = _mode.set(resolve_mode(mode))
    try:
        yield
    finally:
        _mode.reset(token)


@dataclass(frozen=True)
class Sample:
    gpu: int
    util_pct: float
    mem_used_mb: float
    mem_total_mb: float
    power_w: float
    temp_c: float
    # detailed mode only; None when not requested or reported as [N/A]
    mem_util_pct: float | None = None
    clock_sm_mhz: float | None = None
    clock_mem_mhz: float | None = None
    throttle_mask: int | None = None


@dataclass(frozen=True)
class PcieSample:
    gpu: int
    rx_mb_s: float
    tx_mb_s: float


def _opt_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _opt_mask(text: str) -> int | None:
    try:
        return int(text, 16)
    except ValueError:
        return None


def parse_line(line: str, extra: Sequence[str] = ()) -> Sample | None:
    """One `--format=csv,noheader,nounits` row; None for blank lines or `[N/A]` in a basic field.

    `extra` names the columns after the six basic ones (a subset of DETAIL_FIELDS plus a throttle field,
    in query order). An `[N/A]` there leaves just that attribute None.
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6 + len(extra):
        return None
    try:
        util, mem, total, power, temp = (float(v) for v in parts[1:6])
        gpu = int(parts[0])
    except ValueError:
        return None
    mem_util = clock_sm = clock_mem = None
    mask: int | None = None
    for name, text in zip(extra, parts[6:], strict=True):
        if name == "utilization.memory":
            mem_util = _opt_float(text)
        elif name == "clocks.sm":
            clock_sm = _opt_float(text)
        elif name == "clocks.mem":
            clock_mem = _opt_float(text)
        elif name in THROTTLE_FIELDS:
            mask = _opt_mask(text)
    return Sample(gpu, util, mem, total, power, temp, mem_util, clock_sm, clock_mem, mask)


def parse_dmon_line(line: str) -> PcieSample | None:
    """One `nvidia-smi dmon -s t` row: `<gpu> <rxpci MB/s> <txpci MB/s>`. Headers and `-` give None."""
    parts = line.split()
    if line.lstrip().startswith("#") or len(parts) < 3:
        return None
    try:
        return PcieSample(int(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None


def parse_help_query_fields(text: str) -> frozenset[str]:
    """Field names listed by `nvidia-smi --help-query-gpu` (each appears double-quoted)."""
    return frozenset(str(n) for n in re.findall(r'"([a-z0-9_.]+)"', text))


@functools.cache
def listed_fields() -> frozenset[str]:
    """Query fields this nvidia-smi supports; empty if it cannot be run."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--help-query-gpu"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset[str]()
    return parse_help_query_fields(out)


def detail_fields(listed: frozenset[str] | None = None) -> tuple[str, ...]:
    """The optional columns to request. One unsupported field makes nvidia-smi fail the whole query,
    so only fields it lists are asked for."""
    listed = listed_fields() if listed is None else listed
    fields = [f for f in DETAIL_FIELDS if f in listed]
    fields += [next((f for f in THROTTLE_FIELDS if f in listed), "")]
    return tuple(f for f in fields if f)


def _mean(values: Iterable[float | None]) -> float | None:
    xs = [v for v in values if v is not None]
    return statistics.fmean(xs) if xs else None


def _put(m: Metrics, key: str, value: float | None) -> None:
    if value is not None:
        m[key] = value


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


def _throttled(mask: int | None) -> float | None:
    return None if mask is None else float(bool(mask & THROTTLE_LIMITING))


def summarize_detail(samples: Sequence[Sample], pcie: Sequence[PcieSample]) -> Metrics:
    """The schema-1 detailed keys (see the module docstring). Empty without samples."""
    if not samples:
        return {}
    m: Metrics = {}
    gpus = sorted({s.gpu for s in samples})
    util_means: list[float] = []
    for g in gpus:
        mine = [s for s in samples if s.gpu == g]
        util_means.append(statistics.fmean(s.util_pct for s in mine))
        m[f"gpu{g}_util_pct"] = util_means[-1]
        m[f"gpu{g}_util_pct_max"] = max(s.util_pct for s in mine)
        m[f"gpu{g}_mem_used_mb_max"] = max(s.mem_used_mb for s in mine)
        m[f"gpu{g}_power_w"] = statistics.fmean(s.power_w for s in mine)
        m[f"gpu{g}_power_w_max"] = max(s.power_w for s in mine)
        _put(m, f"gpu{g}_mem_util_pct", _mean(s.mem_util_pct for s in mine))
        _put(m, f"gpu{g}_clock_sm_mhz", _mean(s.clock_sm_mhz for s in mine))
        _put(m, f"gpu{g}_clock_mem_mhz", _mean(s.clock_mem_mhz for s in mine))
        _put(m, f"gpu{g}_throttle_frac", _mean(_throttled(s.throttle_mask) for s in mine))
    m["gpu_util_pct_spread"] = max(util_means) - min(util_means)
    _put(m, "gpu_mem_util_pct", _mean(s.mem_util_pct for s in samples))
    _put(m, "gpu_clock_sm_mhz", _mean(s.clock_sm_mhz for s in samples))
    clocks = [s.clock_sm_mhz for s in samples if s.clock_sm_mhz is not None]
    if clocks:
        m["gpu_clock_sm_mhz_min"] = min(clocks)
    _put(m, "gpu_clock_mem_mhz", _mean(s.clock_mem_mhz for s in samples))
    _put(m, "gpu_throttle_frac", _mean(_throttled(s.throttle_mask) for s in samples))
    masks = [s.throttle_mask for s in samples if s.throttle_mask is not None]
    if masks:
        seen = 0
        for mask in masks:
            seen |= mask
        m["gpu_throttle_mask_seen"] = float(seen)
    if pcie:
        rx_total = tx_total = 0.0
        for g in sorted({p.gpu for p in pcie}):
            mine_p = [p for p in pcie if p.gpu == g]
            rx, tx = statistics.fmean(p.rx_mb_s for p in mine_p), statistics.fmean(p.tx_mb_s for p in mine_p)
            rx_total, tx_total = rx_total + rx, tx_total + tx
            m[f"gpu{g}_pcie_rx_mb_s"] = rx
            m[f"gpu{g}_pcie_rx_mb_s_max"] = max(p.rx_mb_s for p in mine_p)
            m[f"gpu{g}_pcie_tx_mb_s"] = tx
            m[f"gpu{g}_pcie_tx_mb_s_max"] = max(p.tx_mb_s for p in mine_p)
        m["gpu_pcie_rx_mb_s_total_mean"] = rx_total
        m["gpu_pcie_tx_mb_s_total_mean"] = tx_total
        m["gpu_pcie_samples"] = float(len(pcie))
    return m


class GpuSampler:
    """Background nvidia-smi sampling.

    `metrics` is "basic" or "detailed"; None takes the mode set by `metrics_mode` (default basic).
    `cmd` / `pcie_cmd` replace the nvidia-smi commands (tests, wrappers); with a custom `cmd` in detailed
    mode `extra` names its columns after the basic six. `pcie_cmd` defaults to `nvidia-smi dmon -s t`.
    """

    def __init__(
        self,
        interval_ms: int = 500,
        cmd: Sequence[str] | None = None,
        *,
        metrics: str | None = None,
        extra: Sequence[str] | None = None,
        pcie_cmd: Sequence[str] | None = None,
    ) -> None:
        self.detailed = resolve_mode(metrics if metrics is not None else _mode.get()) == "detailed"
        self.extra: tuple[str, ...] = ()
        if self.detailed:
            self.extra = tuple(extra) if extra is not None else (detail_fields() if cmd is None else ())
        query = ",".join([QUERY, *self.extra])
        self.cmd = (
            list(cmd)
            if cmd is not None
            else [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
                f"-lms={interval_ms}",
            ]
        )
        self.pcie_cmd = list(pcie_cmd) if pcie_cmd is not None else list(DMON_CMD) if self.detailed else None
        if not self.detailed:
            self.pcie_cmd = None
        self.samples: list[Sample] = []
        self.pcie_samples: list[PcieSample] = []
        self._procs: list[subprocess.Popen[str]] = []
        self._threads: list[threading.Thread] = []
        self._t0 = self._t1 = 0.0

    def _spawn(self, cmd: Sequence[str], target: Callable[[subprocess.Popen[str]], None]) -> None:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        thread = threading.Thread(target=target, args=(proc,), daemon=True)
        thread.start()
        self._procs.append(proc)
        self._threads.append(thread)

    def __enter__(self) -> GpuSampler:
        self._t0 = time.perf_counter()
        self._spawn(self.cmd, self._read)
        if self.pcie_cmd is not None:
            self._spawn(self.pcie_cmd, self._read_pcie)
        return self

    def _read(self, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            if (s := parse_line(line, self.extra)) is not None:
                self.samples.append(s)

    def _read_pcie(self, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            if (p := parse_dmon_line(line)) is not None:
                self.pcie_samples.append(p)

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._t1 = time.perf_counter()
        for proc in self._procs:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        for thread in self._threads:
            thread.join(timeout=5)

    def metrics(self) -> Metrics:
        """Basic keys, plus the detailed ones in detailed mode."""
        return summarize_samples(self.samples, self._t1 - self._t0) | self.detail_metrics()

    def detail_metrics(self) -> Metrics:
        """Only the schema-1 detailed keys; empty in basic mode or without nvidia-smi samples."""
        return summarize_detail(self.samples, self.pcie_samples) if self.detailed else {}


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

    `keys` restricts which of the BASIC metrics are kept (default: all). The detailed schema-1 keys
    (per-GPU, clocks, pcie; only present when the run's `gpu_metrics` is "detailed") are always kept in
    full, so a sweep that selects its basic keys needs no change to carry them.

    Caveats when reading per-point numbers:
      * `nvidia-smi` is polled every 500 ms (`GpuSampler(interval_ms=500)`), so a point needs to run for
        several seconds to have a meaningful number of samples. Check `<prefix>gpu_samples`. The PCIe
        stream (`dmon`) ticks once per second, so check `<prefix>gpu_pcie_samples` too.
      * The basic keys pool every visible GPU: `gpu_util_pct` is the mean over all GPUs and all ticks,
        so a TP=4 run with one busy GPU reads as 25% and gpu_count says how many were pooled. The
        detailed `gpu<g>_...` keys are the per-GPU view.
    """
    point = PointMetrics()
    with sampler() as smi:
        yield point
    detail = smi.detail_metrics()
    point.metrics = {
        f"{prefix}{k}": v for k, v in smi.metrics().items() if k in detail or keys is None or k in keys
    }
