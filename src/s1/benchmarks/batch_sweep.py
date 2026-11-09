"""batch: batch-size sweep and CPU share of latency (hypothesis H2: small models are launch-bound at batch 1).

B independent requests, each with its own state and one question, run as one batched forward. For every
(state length L, batch size B) point of the grid

    [grid]
    batch_state_tokens = [128, 1024]                 # reference tokens per state, as roofline's length axis
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

the benchmark takes a pool of `batch_pool` seeded JEVal rows whose own state is at least max(L) reference
tokens (`roofline.length_rows`: the same rows at every L, each state cut to its first L tokens), forms
n = max(batch_min_batches, ceil(batch_min_decisions / B)) batches of B rows (batch k takes rows kB.. kB+B-1 of
the pool, wrapping), runs them all once untimed (warm-up: CUDA-graph captures, compile, allocator), then
again timed. Per point, `batch_s<L>_b<B>_*`:

    wall_ms, wall_ms_median   host wall time per batch (tokenize, forward, copy the scores back)
    ms_per_decision           wall_ms / B
    decisions_per_s
    gpu_ms                    forwards' CUDA-event span per batch (start to end on the stream: includes the
                              GPU's idle gaps while it waits for the host to launch the next kernel)
    gpu_busy                  gpu_ms / wall_ms
    tflops                    cost-model FLOPs / gpu_ms (s1.flops, as roofline_tflops)
    tflops_wall               cost-model FLOPs / wall_ms (end-to-end achieved FLOP/s; H2's test)
    intensity, gbs, mb, gflop cost-model FLOP/byte, bytes / gpu_ms, MB and GFLOP per batch
    tokens, padding           tokens charged per batch, share of them that were padding
    kernel_ms, kernels        summed GPU kernel time and kernels per batch    (torch.profiler, `batch_trace`)
    launch_api_ms             host time in kernel/graph launch calls per batch
    kernel_busy, cpu_share    kernel_ms / wall_ms and 1 - that: how much of the latency the GPU computes
    kernels_per_forward
    oom                       1 when the point ran out of GPU memory (its other keys are then absent)

The kernel trace runs after the timed pass on `batch_trace` batches; its wall time is never used (tracing
slows the host). Headline keys (first): the batch-1 CPU share at the shortest length and the FLOP/s gain from
B = 1 to the largest B that ran there. `batch_b1_*` repeat the shortest length's batch-1 point.

`batch_nsys = N` (set by `s1 nsys`, see s1/nsys.py) replaces the sweep: N batch-1 decisions at the shortest
length run inside one CUDA profiler range, after `batch_warmup` untimed ones, and the region's wall time is
recorded as batch_nsys_{n,wall_ms}; `s1 nsys` adds the Nsight Systems CPU/GPU split to the record.

The model implements, besides roofline's `profile_info` / `take_forwards`:
    decide_batch(requests: [(state, {key: question})]) -> {"decisions": int, "real_tokens": int}
        one batched pass over every request's questions, end to end (scores on the host when it returns)
and may implement `after_warmup()` (called after each point's warm-up; kev captures pending CUDA graphs) and
`profiler(on)` (needed for batch_nsys).

Settings (all optional): batch_pool (default max batch size), batch_min_decisions 32, batch_min_batches 4,
batch_seed 0, batch_max_options 16, batch_max_question_tokens 512, batch_tokenizer_file (a tokenizer.json;
words if unset), batch_trace 2 (batches traced per point; 0 = no trace), batch_nsys 0, batch_warmup 8.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .. import env, flops, nsys
from ..record import Metrics
from ..registry import Settings, benchmark, get_int, get_str, note
from . import jeval, roofline
from .shape_sweep import FileTokenCounter, WordCounter, parse_int_list

NAME = "batch"
SIZES = "batch_sizes"
LENGTHS = "batch_state_tokens"

Request = tuple[str, Mapping[str, object]]


class Batched(Protocol):
    def profile_info(self) -> Mapping[str, object]: ...
    def take_forwards(self) -> list[Mapping[str, object]]: ...
    def decide_batch(self, requests: Sequence[Request]) -> Mapping[str, int]: ...


@dataclass(frozen=True)
class Timed:
    wall_ms: float
    cost: flops.Cost
    forwards: int
    real_tokens: int
    decisions: int


@dataclass(frozen=True)
class Trace:
    kernel_us: float
    kernels: int
    launch_us: float
    sync_us: float


def batches(pool: Sequence[jeval.Row], size: int, settings: Settings) -> list[list[jeval.Row]]:
    """n = max(batch_min_batches, ceil(batch_min_decisions / size)) batches of `size` rows; batch k takes pool
    rows k*size .. k*size + size - 1, wrapping, so no batch repeats a row while size <= len(pool)."""
    if size > len(pool):
        raise ValueError(f"batch size {size} is over the pool of {len(pool)} rows (batch_pool)")
    n = max(
        get_int(settings, "batch_min_batches", 4),
        math.ceil(get_int(settings, "batch_min_decisions", 32) / size),
    )
    return [[pool[(k * size + j) % len(pool)] for j in range(size)] for k in range(n)]


def requests(rows: Sequence[jeval.Row]) -> list[Request]:
    return [(r.state, r.questions) for r in rows]


def trace_split(events: Sequence[tuple[str, str, float]]) -> Trace:
    """Kernel time and count, and host time in launch and synchronize calls, of a profiler trace."""
    kernels = [us for _, kind, us in events if kind == "kernel"]
    api = [(name, us) for name, kind, us in events if kind == "api"]
    return Trace(
        kernel_us=sum(kernels),
        kernels=len(kernels),
        launch_us=sum(us for name, us in api if nsys.is_launch(name)),
        sync_us=sum(us for name, us in api if nsys.is_sync(name)),
    )


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


def point(prefix: str, timed: Sequence[Timed], size: int, trace: Trace | None, traced: int) -> Metrics:
    """The metrics of one (length, batch size) point; `trace` covers `traced` batches."""
    n = len(timed)
    cost = flops.ZERO
    for t in timed:
        cost = cost + t.cost
    wall = sum(t.wall_ms for t in timed) / n
    gpu = cost.ms / n
    m: Metrics = {
        f"{prefix}wall_ms": wall,
        f"{prefix}wall_ms_median": statistics.median(t.wall_ms for t in timed),
        f"{prefix}ms_per_decision": wall / size,
        f"{prefix}decisions_per_s": _ratio(
            sum(t.decisions for t in timed), sum(t.wall_ms for t in timed) / 1e3
        ),
        f"{prefix}gpu_ms": gpu,
        f"{prefix}gpu_busy": _ratio(gpu, wall),
        f"{prefix}tflops": _ratio(cost.flops, cost.ms * 1e9),
        f"{prefix}tflops_wall": _ratio(cost.flops, sum(t.wall_ms for t in timed) * 1e9),
        f"{prefix}intensity": _ratio(cost.flops, cost.bytes),
        f"{prefix}gbs": _ratio(cost.bytes, cost.ms * 1e6),
        f"{prefix}mb": cost.bytes / 1e6 / n,
        f"{prefix}gflop": cost.flops / 1e9 / n,
        f"{prefix}tokens": cost.tokens / n,
        f"{prefix}padding": 1 - _ratio(sum(t.real_tokens for t in timed), cost.tokens),
        f"{prefix}forwards": sum(t.forwards for t in timed) / n,
        f"{prefix}batches": float(n),
    }
    if trace is not None and traced:
        kernel_ms = trace.kernel_us / 1e3 / traced
        m |= {
            f"{prefix}kernel_ms": kernel_ms,
            f"{prefix}kernels": trace.kernels / traced,
            f"{prefix}launch_api_ms": trace.launch_us / 1e3 / traced,
            f"{prefix}sync_api_ms": trace.sync_us / 1e3 / traced,
            f"{prefix}kernel_busy": _ratio(kernel_ms, wall),
            f"{prefix}cpu_share": 1 - _ratio(kernel_ms, wall),
            f"{prefix}kernels_per_forward": _ratio(trace.kernels / traced, m[f"{prefix}forwards"]),
        }
    return m


def headline(m: Metrics, lengths: Sequence[int], sizes: Sequence[int]) -> Metrics:
    """batch_b1_cpu_share and batch_tflops_gain (tflops_wall at the largest B that ran / at B = 1) at the
    shortest length, then batch_b1_* copies of that point, first in the dict."""
    short, b1 = f"batch_s{min(lengths)}_", f"batch_s{min(lengths)}_b{min(sizes)}_"
    ran = [b for b in sorted(sizes) if f"{short}b{b}_tflops_wall" in m]
    out: Metrics = {
        "batch_b1_cpu_share": m.get(f"{b1}cpu_share", math.nan),
        "batch_tflops_gain": (
            _ratio(m[f"{short}b{ran[-1]}_tflops_wall"], m[f"{short}b{ran[0]}_tflops_wall"])
            if ran
            else math.nan
        ),
        "batch_max_size_ran": float(ran[-1]) if ran else math.nan,
    }
    out |= {"batch_b1_" + k.removeprefix(b1): v for k, v in m.items() if k.startswith(b1)}
    return out


def _counter(settings: Settings) -> roofline.FittingCounter:
    path = get_str(settings, "batch_tokenizer_file", "")
    return FileTokenCounter(str(env.resolve_path(path))) if path else WordCounter()


def pools(settings: Settings, lengths: Sequence[int], size: int) -> dict[int, list[jeval.Row]]:
    """{length: the pool's rows cut to it}: roofline.length_rows with this benchmark's settings."""
    keys: Settings = {
        "roofline_per_length": get_int(settings, "batch_pool", size),
        "roofline_seed": get_int(settings, "batch_seed", 0),
        "roofline_max_options": get_int(settings, "batch_max_options", 16),
        "roofline_max_question_tokens": get_int(settings, "batch_max_question_tokens", 512),
    }
    out: dict[int, list[jeval.Row]] = {n: [] for n in lengths}
    for n, row in roofline.length_rows(jeval.load_rows(), _counter(settings), keys, lengths):
        out[n].append(row)
    return out


def _model(model: object) -> Batched:
    for attr in ("decide_batch", "profile_info", "take_forwards"):
        if not callable(getattr(model, attr, None)):
            raise TypeError(f"batch needs a model with .{attr}(); got {type(model).__name__}")
    return model  # ty: ignore[invalid-return-type]


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def _free_cuda() -> None:
    import gc  # noqa: PLC0415
    import sys  # noqa: PLC0415

    gc.collect()
    if "torch" in sys.modules:
        sys.modules["torch"].cuda.empty_cache()


@benchmark(NAME)
def run(model: object, settings: Settings) -> Metrics:
    m = _model(model)
    info = m.profile_info()
    config = info["config"]
    if not isinstance(config, Mapping):
        raise TypeError("profile_info()['config'] must be the model's HF config dict")
    arch = flops.arch(config)
    sizes_kw = {
        "head_bytes": int(roofline._num({"head_bytes": flops.BF16_BYTES} | dict(info), "head_bytes", int)),
        "weight_bytes": int(
            roofline._num({"weight_bytes": flops.BF16_BYTES} | dict(info), "weight_bytes", int)
        ),
    }
    lengths, sizes = sorted(parse_int_list(settings, LENGTHS)), sorted(parse_int_list(settings, SIZES))
    if n_nsys := get_int(settings, "batch_nsys", 0):
        return nsys_region(m, pools(settings, lengths[:1], 1)[lengths[0]], n_nsys, settings)
    pool = pools(settings, lengths, max(sizes))
    after_warmup = getattr(m, "after_warmup", None)
    traced_n = get_int(settings, "batch_trace", 2)
    out: Metrics = {}
    for length in lengths:
        for size in sizes:
            prefix = f"batch_s{length}_b{size}_"
            plan = batches(pool[length], size, settings)
            try:
                for b in plan:  # warm-up: every shape once, untimed
                    m.decide_batch(requests(b))
                if callable(after_warmup):
                    after_warmup()
                m.take_forwards()
                timed = [time_batch(m, b, arch, sizes_kw) for b in plan]
                trace = None
                if traced_n:
                    trace = trace_batches(m, plan[:traced_n])
            except Exception as e:
                if not _is_oom(e):
                    raise
                m.take_forwards()
                _free_cuda()
                out[f"{prefix}oom"] = 1.0
                note(f"{prefix}oom", f"{type(e).__name__}: {e}"[:200])
                print(f"batch: L={length} B={size} out of memory", flush=True)
                continue
            out |= point(prefix, timed, size, trace, min(traced_n, len(plan)))
            print(
                f"batch: L={length} B={size} wall {out[f'{prefix}wall_ms']:.2f} ms, "
                f"{out[f'{prefix}tflops_wall']:.2f} TFLOP/s, "
                f"cpu_share {out.get(f'{prefix}cpu_share', math.nan):.2f}",
                flush=True,
            )
    return (
        headline(out, lengths, sizes)
        | {"batch_lengths": float(len(lengths)), "batch_sizes": float(len(sizes))}
        | out
    )


def time_batch(m: Batched, rows: Sequence[jeval.Row], arch: flops.Arch, sizes_kw: Mapping[str, int]) -> Timed:
    t0 = time.perf_counter()
    done = m.decide_batch(requests(rows))
    wall = (time.perf_counter() - t0) * 1e3
    fwd = [roofline.forward(f) for f in m.take_forwards()]
    return Timed(
        wall_ms=wall,
        cost=flops.total(arch, fwd, **sizes_kw),
        forwards=len(fwd),
        real_tokens=int(done.get("real_tokens", 0)),
        decisions=int(done.get("decisions", len(rows))),
    )


def trace_batches(m: Batched, plan: Sequence[Sequence[jeval.Row]]) -> Trace | None:
    """torch.profiler over `plan` (best effort: a trace failure leaves the point's other keys)."""
    from .. import profiling  # noqa: PLC0415 — imports torch lazily inside

    def go() -> None:
        for b in plan:
            m.decide_batch(requests(b))

    try:
        return trace_split(profiling.trace_activity(go))
    except Exception as e:
        if _is_oom(e):
            raise
        note("batch_trace_error", f"{type(e).__name__}: {e}"[:300])
        return None
    finally:
        m.take_forwards()


def nsys_region(m: Batched, pool: Sequence[jeval.Row], n: int, settings: Settings) -> Metrics:
    """`batch_warmup` untimed batch-1 decisions, then `n` inside one CUDA profiler range (cudaProfilerStart/
    Stop), the range Nsight Systems captures under `s1 nsys`."""
    profiler = getattr(m, "profiler", None)
    if not callable(profiler):
        raise TypeError(f"batch_nsys needs a model with .profiler(on); got {type(m).__name__}")
    for i in range(get_int(settings, "batch_warmup", 8)):
        m.decide_batch(requests([pool[i % len(pool)]]))
    m.take_forwards()
    profiler(True)
    t0 = time.perf_counter()
    try:
        for i in range(n):
            m.decide_batch(requests([pool[i % len(pool)]]))
    finally:
        wall = (time.perf_counter() - t0) * 1e3
        profiler(False)
    m.take_forwards()
    lengths = sorted(parse_int_list(settings, LENGTHS))
    return {
        "batch_nsys_n": float(n),
        "batch_nsys_wall_ms": wall,
        "batch_nsys_state_tokens": float(lengths[0]),
    }
