"""The batch benchmark (batch-size sweep, CPU share of latency), the nsys report parser, and the torch-free
parts of the Laya batch path. No torch."""

from __future__ import annotations

import math
from typing import override

import pytest

from s1 import nsys, profiling, registry, runner
from s1.benchmarks import batch_sweep, jeval
from s1.config import RunSpec
from s1.launch import modal_app
from s1.launch.base import Resources
from s1.models.backends import laya
from tests.test_flops import QWEN3_8B
from tests.test_roofline import _no_gpu, row

# ---- nsys stats CSV (columns as nsys 2026.3 writes them) ----

API_CSV = """\
NOTICE: Existing SQLite export found
"Time (%)","Total Time (ns)","Num Calls","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"
"60.0","6000000","1200","5000.0","4800.0","3000","20000","900.0","cudaLaunchKernel"
"25.0","2500000","10","250000.0","250000.0","200000","300000","1000.0","cudaStreamSynchronize"
"10.0","1000000","20","50000.0","50000.0","40000","60000","100.0","cudaMemcpyAsync"
"5.0","500000","10","50000.0","50000.0","40000","60000","100.0","cudaGraphLaunch_v10000"
"""
KERN_CSV = """\
"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"
"70.0","3500000","500","7000.0","7000.0","5000","9000","100.0","ampere_bf16_s16816gemm_bf16_128x64"
"30.0","1500000","700","2142.9","2000.0","1000","4000","100.0","vectorized_elementwise_kernel"
"""


def test_nsys_parse_skips_notices_and_reads_counts():
    api = nsys.parse(API_CSV)
    assert api[0] == ("cudaLaunchKernel", 6e6, 1200)
    assert len(api) == 4
    assert nsys.parse(KERN_CSV)[1] == ("vectorized_elementwise_kernel", 1.5e6, 700)
    assert nsys.parse("SKIPPED: no CUDA kernel data\n") == []


def test_nsys_summarize_splits_wall_time_per_decision():
    run = {"batch_nsys_n": 10.0, "batch_nsys_wall_ms": 20.0}
    m = nsys.summarize(nsys.parse(API_CSV), nsys.parse(KERN_CSV), run)
    assert next(iter(m)) == "batch_nsys_cpu_share"
    assert m["batch_nsys_kernel_ms"] == pytest.approx(0.5)  # 5 ms of kernels / 10 decisions
    assert m["batch_nsys_gpu_share"] == pytest.approx(0.25)  # 5 of 20 ms
    assert m["batch_nsys_cpu_share"] == pytest.approx(0.75)
    assert m["batch_nsys_kernels"] == 120
    assert m["batch_nsys_launches"] == 121  # cudaLaunchKernel + cudaGraphLaunch, not memcpy
    assert m["batch_nsys_launch_api_ms"] == pytest.approx(0.65)
    assert m["batch_nsys_sync_ms"] == pytest.approx(0.25)
    assert m["batch_nsys_wall_ms_per_decision"] == 2.0
    with pytest.raises(ValueError, match="no nsys region"):
        nsys.summarize([], [], {})
    assert nsys.top_kernels(nsys.parse(KERN_CSV), top=1) == "70.0% x500 ampere_bf16_s16816gemm_bf16_128x64"


def test_nsys_command_lines():
    args = nsys.profile_args("logs/nsys-1-cfg-0")
    assert args[0] == "profile"
    assert "--capture-range=cudaProfilerApi" in args
    assert "--sample=process-tree" in args
    assert args[-2:] == ("-o", "logs/nsys-1-cfg-0")
    assert "--cpuctxsw=none" in nsys.profile_args("x", "none")
    with pytest.raises(ValueError, match="sample"):
        nsys.profile_args("x", "cpu")
    stats = nsys.stats_args("r.nsys-rep", "r")
    assert stats[0] == "stats" and stats[-1] == "r.nsys-rep"
    assert [stats[i + 1] for i, a in enumerate(stats) if a == "--report"] == list(nsys.REPORTS)


def test_launch_and_sync_names():
    for name in ("cudaLaunchKernel", "cuLaunchKernelEx", "cudaLaunchKernelExC_v11060", "cudaGraphLaunch"):
        assert nsys.is_launch(name)
    assert not nsys.is_launch("cudaMemcpyAsync")
    assert nsys.is_sync("cudaStreamSynchronize") and nsys.is_sync("cudaEventSynchronize")
    assert not nsys.is_sync("cudaLaunchKernel")


def test_modal_plan_passes_results_dir_to_nsys(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('benchmarks = ["batch"]\n[params]\nmodel = "laya.english"\n')
    res = Resources(orchestrator="modal", gpu="L4", gpus=1, cpus=8, mem="28G", time="00:30:00")
    p = modal_app.plan(cfg, res, ["nsys", str(cfg), "--decisions", "16"])
    assert p.s1_args[0] == "nsys"
    assert "--results-dir" in p.s1_args
    assert "--set" not in p.s1_args  # s1 nsys takes no --set


# ---- the benchmark's pure parts ----


def test_batches_wrap_the_pool_without_repeating_rows_inside_a_batch():
    pool = [row(i) for i in range(8)]
    plan = batch_sweep.batches(pool, 4, {"batch_min_decisions": 32, "batch_min_batches": 2})
    assert len(plan) == 8  # ceil(32 / 4)
    assert [r.id for r in plan[0]] == ["r0", "r1", "r2", "r3"]
    assert [r.id for r in plan[2]] == ["r0", "r1", "r2", "r3"]  # wraps
    assert all(len({r.id for r in b}) == 4 for b in plan)
    assert len(batch_sweep.batches(pool, 8, {"batch_min_decisions": 8, "batch_min_batches": 3})) == 3
    with pytest.raises(ValueError, match="over the pool"):
        batch_sweep.batches(pool, 16, {})


def test_trace_split_counts_kernels_and_launch_time():
    events = [
        ("gemm", "kernel", 30.0),
        ("elementwise", "kernel", 10.0),
        ("Memcpy DtoH", "copy", 5.0),
        ("cudaLaunchKernel", "api", 4.0),
        ("cudaLaunchKernel", "api", 4.0),
        ("cudaStreamSynchronize", "api", 50.0),
    ]
    t = batch_sweep.trace_split(events)
    assert t == batch_sweep.Trace(kernel_us=40.0, kernels=2, launch_us=8.0, sync_us=50.0)


def _timed(wall, gpu_ms, rows, tokens, decisions):
    from s1 import flops  # noqa: PLC0415

    arch = flops.arch(QWEN3_8B)
    cost = flops.cost(arch, flops.Forward(rows=rows, tokens=tokens, kept=1, ms=gpu_ms))
    return batch_sweep.Timed(wall, cost, 1, rows * tokens - rows, decisions)


def test_point_metrics_and_cpu_share():
    timed = [_timed(10.0, 4.0, 4, 100, 4), _timed(12.0, 4.0, 4, 100, 4)]
    trace = batch_sweep.Trace(kernel_us=6000.0, kernels=400, launch_us=1000.0, sync_us=0.0)
    m = batch_sweep.point("p_", timed, 4, trace, 2)
    assert m["p_wall_ms"] == 11.0
    assert m["p_ms_per_decision"] == 2.75
    assert m["p_decisions_per_s"] == pytest.approx(8 / 0.022)
    assert m["p_gpu_busy"] == pytest.approx(4 / 11)
    assert m["p_tflops_wall"] == pytest.approx(m["p_tflops"] * 4 / 11)
    assert m["p_padding"] == pytest.approx(0.01)
    assert m["p_kernel_ms"] == 3.0  # 6 ms over 2 traced batches
    assert m["p_kernel_busy"] == pytest.approx(3 / 11)
    assert m["p_cpu_share"] == pytest.approx(8 / 11)
    assert m["p_kernels_per_forward"] == 200
    assert "p_cpu_share" not in batch_sweep.point("p_", timed, 4, None, 0)


def test_headline_uses_the_shortest_length_and_largest_batch_that_ran():
    m = {
        "batch_s128_b1_tflops_wall": 1.0,
        "batch_s128_b1_cpu_share": 0.8,
        "batch_s128_b8_tflops_wall": 6.0,
        "batch_s128_b64_oom": 1.0,
        "batch_s1024_b1_tflops_wall": 3.0,
    }
    h = batch_sweep.headline(m, [1024, 128], [1, 8, 64])
    assert list(h)[:2] == ["batch_b1_cpu_share", "batch_tflops_gain"]
    assert h["batch_b1_cpu_share"] == 0.8
    assert h["batch_tflops_gain"] == 6.0
    assert h["batch_max_size_ran"] == 8
    assert h["batch_b1_tflops_wall"] == 1.0
    assert math.isnan(batch_sweep.headline({}, [128], [1])["batch_tflops_gain"])


# ---- end to end with a fake batched model ----


class FakeBatched:
    """Every batch 'runs' one forward of rows x longest state; the scores are on the host at once."""

    def __init__(self, oom_at: int = 0):
        self.pending: list[dict[str, object]] = []
        self.oom_at, self.calls, self.warmups, self.profiled = oom_at, 0, 0, []

    def profile_info(self):
        return {"config": QWEN3_8B, "head_bytes": 2, "ceilings": {}}

    def take_forwards(self):
        out, self.pending = self.pending, []
        return out

    def decide_batch(self, requests):
        if self.oom_at and len(requests) >= self.oom_at:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.calls += 1
        width = max(len(s.split()) for s, _ in requests)
        self.pending.append(
            {"rows": len(requests), "tokens": width, "past": 0, "kept": 1, "cached": False, "ms": 1.0}
        )
        return {"decisions": len(requests), "real_tokens": sum(len(s.split()) for s, _ in requests)}

    def after_warmup(self):
        self.warmups += 1

    def profiler(self, on):
        self.profiled.append(on)


def _run(tmp_path, monkeypatch, model, **params):
    rows = [row(i, state_words=300 + i) for i in range(20)]
    monkeypatch.setattr(jeval, "load_rows", lambda: rows)
    monkeypatch.setattr(profiling, "trace_activity", lambda fn: (fn(), [("k", "kernel", 500.0)])[1])
    registry.MODELS["fake-batched"] = lambda settings: model
    try:
        spec = RunSpec(
            benchmarks=["batch"],
            invariants={"gpu": "NVIDIA L4"},
            params={"model": "fake-batched", "batch_pool": 8, "batch_min_decisions": 8, **params},
            grid={batch_sweep.LENGTHS: [16, 64], batch_sweep.SIZES: [1, 2, 8]},
        )
        return runner.run(spec, results_dir=tmp_path, log=lambda _: None, gpu_sampler=_no_gpu)
    finally:
        registry.MODELS.pop("fake-batched")


def test_benchmark_end_to_end_through_runner(tmp_path, monkeypatch):
    model = FakeBatched()
    rec = _run(tmp_path, monkeypatch, model)
    m = rec.results["batch"]
    assert next(iter(m)) == "batch_b1_cpu_share"
    assert m["batch_s16_b1_batches"] == 8 and m["batch_s16_b8_batches"] == 4  # min_batches = 4
    assert m["batch_s64_b8_tokens"] == 8 * 64
    assert m["batch_s16_b1_kernel_ms"] == pytest.approx(0.25)  # 0.5 ms traced over 2 batches
    assert m["batch_s16_b8_intensity"] > m["batch_s16_b1_intensity"]  # weights shared by more rows
    assert m["batch_tflops_gain"] > 1
    assert model.warmups == 6  # once per point
    assert rec.grid == {batch_sweep.LENGTHS: [16, 64], batch_sweep.SIZES: [1, 2, 8]}


def test_out_of_memory_point_is_recorded_and_the_sweep_goes_on(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, FakeBatched(oom_at=8))
    m = rec.results["batch"]
    assert m["batch_s16_b8_oom"] == 1 and m["batch_s64_b8_oom"] == 1
    assert "batch_s16_b8_wall_ms" not in m
    assert m["batch_s64_b2_wall_ms"] > 0
    assert m["batch_max_size_ran"] == 2
    assert "out of memory" in str(rec.meta["batch_s16_b8_oom"])


def test_other_errors_are_not_swallowed(tmp_path, monkeypatch):
    class Broken(FakeBatched):
        @override
        def decide_batch(self, requests):
            raise ValueError("bad question")

    with pytest.raises(ValueError, match="bad question"):
        _run(tmp_path, monkeypatch, Broken())


def test_nsys_mode_runs_only_the_profiled_region(tmp_path, monkeypatch):
    model = FakeBatched()
    rec = _run(tmp_path, monkeypatch, model, batch_nsys=5, batch_warmup=3)
    m = rec.results["batch"]
    assert m["batch_nsys_n"] == 5 and m["batch_nsys_state_tokens"] == 16
    assert m["batch_nsys_wall_ms"] >= 0
    assert model.calls == 8  # 3 warm-up + 5 profiled, no sweep
    assert model.profiled == [True, False]


def test_benchmark_rejects_models_without_decide_batch():
    with pytest.raises(TypeError, match="decide_batch"):
        batch_sweep.run(object(), {})


# ---- Laya batch path (torch-free parts) ----


def test_padded_width():
    assert laya.padded_width(130, 64) == 192
    assert laya.padded_width(128, 64) == 128
    assert laya.padded_width(7, 1) == 7
    with pytest.raises(ValueError, match="laya_pad_multiple"):
        laya.padded_width(7, 0)


def test_laya_graph_key():
    class T:
        def __init__(self, *shape):
            self.shape = shape

    assert laya.graph_key([T(4, 192), T(4, 192), T(4, 3), T(4, 3), T(4)]) == (4, 192, 3)


def test_laya_modes():
    assert laya.MODES == ("eager", "graphs", "compile", "compile_graphs")
