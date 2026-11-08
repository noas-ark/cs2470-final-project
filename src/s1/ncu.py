"""Hardware-counter roofline: run a config's models under Nsight Compute (`s1 ncu`) and sum the counters.

The roofline benchmark (with `roofline_ncu = N`) re-answers its first N questions inside a CUDA profiler
range; ncu, started with `--profile-from-start off --target-processes all` (so it also follows elephant-1's
worker process), replays each kernel in that range and reads:

    FLOPs  = tensor-core ops (bf16/fp16/tf32 sources; an FMA counts 2, checked: a 4096^3 bf16 GEMM reads
             exactly 2 * 4096^3) + CUDA-core fp32 (2 * FFMA + FADD + FMUL) + fp16 (2 * HFMA + HADD + HMUL)
    bytes  = DRAM bytes read + written (real traffic: L2 misses, re-reads included)

so `roofline_hw_intensity` is the measured operational intensity, and `roofline_hw_tflops` divides the
measured FLOPs by the same questions' GPU time from the run's timed pass (ncu's own kernel durations are taken
at the clocks ncu sets, so they are kept separately as `roofline_hw_kernel_ms`). `roofline_hw_flop_ratio` and
`roofline_hw_byte_ratio` compare the counters with the cost model (s1.flops) for the same forwards.

Clock control (`ncu_clock_control`, ncu's `--clock-control`): "base" is ncu's default, which locks clocks to
base (what every Slurm run so far used, and the flag is then left off the command line, so it is unchanged);
"none" leaves the GPU's own clocks. On Modal the default is "none", because clock locking fails on Modal
workers. The value is recorded as `roofline_hw_clocks_locked` (1 = base, 0 = none) next to the kernel times it
qualifies.

Torch-free: only parses ncu's `--csv --page raw` output.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping

from . import env
from .record import Metrics

TENSOR = (
    "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
    "sm__ops_path_tensor_src_fp16_dst_fp16.sum",
    "sm__ops_path_tensor_src_tf32_dst_fp32.sum",
)
CORE = {  # CUDA-core instruction counts and the FLOPs each is worth
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 2,
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum": 2,
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum": 1,
}
DRAM = ("dram__bytes_read.sum", "dram__bytes_write.sum")
TIME = "gpu__time_duration.sum"
METRICS = (*TENSOR, *CORE, *DRAM, TIME)
CLOCK_CONTROLS = ("base", "none")  # ncu --clock-control values that profile (ncu's "reset" just exits)
META_CLOCK_CONTROL = "ncu_clock_control"  # meta field for the resolved value (written by `s1 ncu`)
BASE_ARGS = (
    "--profile-from-start",
    "off",
    "--target-processes",
    "all",
    "--csv",
    "--page",
    "raw",
    "--metrics",
    ",".join(METRICS),
)


def clock_control(explicit: str | None = None) -> str:
    """The `ncu_clock_control` to use: `explicit` if set, else "none" on Modal, else "base" (the default)."""
    value = explicit or ("none" if env.orchestrator() == "modal" else "base")
    if value not in CLOCK_CONTROLS:
        raise ValueError(f"ncu_clock_control must be one of {CLOCK_CONTROLS}, not {value!r}")
    return value


def args(clock: str | None = None) -> tuple[str, ...]:
    """ncu's arguments for `s1 ncu`. "base" adds no flag: the cluster's command line is unchanged."""
    clock = clock_control(clock)
    return BASE_ARGS if clock == "base" else ("--clock-control", clock, *BASE_ARGS)


ARGS = args()  # resolved for this process's orchestrator (with "none" inside a Modal container)


def parse(text: str) -> list[dict[str, float]]:
    """One {metric: value} per profiled kernel from ncu's raw CSV page (log lines before it are skipped,
    as is the units row). Missing or non-numeric values read as 0."""
    start = text.find('"ID",')
    if start < 0:
        raise ValueError("no ncu CSV in the log (did the profiler range run?)")
    rows = list(csv.reader(io.StringIO(text[start:])))
    header, kernels = rows[0], [r for r in rows[2:] if len(r) == len(rows[0])]
    out = []
    for r in kernels:
        values = {}
        for name, val in zip(header, r, strict=True):
            if name in METRICS:
                try:
                    values[name] = float(val.replace(",", ""))
                except ValueError:
                    values[name] = 0.0
        out.append(values)
    return out


def summarize(kernels: list[dict[str, float]], run: Mapping[str, float], clock: str | None = None) -> Metrics:
    """Totals over the profiled kernels, against the run's own `roofline_hw_*` numbers for the same work.
    `clock` is the clock control the kernels ran under (default: this process's, as in `ARGS`)."""
    tensor = sum(k.get(m, 0.0) for k in kernels for m in TENSOR)
    core = sum(k.get(m, 0.0) * w for k in kernels for m, w in CORE.items())
    dram = sum(k.get(m, 0.0) for k in kernels for m in DRAM)
    flops = tensor + core
    theory_flops = run.get("roofline_hw_theory_gflop", 0.0) * 1e9
    theory_bytes = run.get("roofline_hw_theory_mb", 0.0) * 1e6
    timed_ms = run.get("roofline_hw_timed_ms", 0.0)

    def ratio(a: float, b: float) -> float:
        return a / b if b > 0 else float("nan")

    return {
        "roofline_hw_tflops": ratio(flops, timed_ms * 1e9),
        "roofline_hw_intensity": ratio(flops, dram),
        "roofline_hw_gflop": flops / 1e9,
        "roofline_hw_dram_mb": dram / 1e6,
        "roofline_hw_tensor_share": ratio(tensor, flops),
        "roofline_hw_flop_ratio": ratio(flops, theory_flops),
        "roofline_hw_byte_ratio": ratio(dram, theory_bytes),
        "roofline_hw_kernels": float(len(kernels)),
        "roofline_hw_kernel_ms": sum(k.get(TIME, 0.0) for k in kernels) / 1e6,
        "roofline_hw_clocks_locked": 1.0 if clock_control(clock) == "base" else 0.0,
    }
