"""GPU-side instruments for the roofline benchmark: per-forward timing, measured ceilings, counted FLOPs.

Used in-process by the transformers backends and by the elephant runtime's worker
(`runtimes/elephant/worker.py`), which runs under another torch (2.8) and Python (3.11): keep this module
3.11-compatible and import torch only inside functions. Everything returned is plain JSON-able data.

    rec = ForwardRecorder(model)          # hooks the top-level forward of a HF model
    ...                                   # run decisions
    rec.take()                            # [{"rows", "tokens", "past", "kept", "cached", "ms"}, ...]

    measure_ceilings()                    # {"peak_bf16_tflops", "peak_fp32_tflops", "peak_bw_gbs", ...}
    out, flops = count_flops(lambda: model.decide(state, questions))
"""

from __future__ import annotations

import contextlib
import importlib.util
import statistics
import time
from collections.abc import Callable, Generator
from typing import Any

# Each ceiling runs for about this long, so a power-capped GPU (the L4 is 72 W) reaches its sustained clock.
CEILING_SECONDS = 2.0


class ForwardRecorder:
    """Records every top-level forward of a HF model: its shape (rows x new tokens, cached past length, kept
    logit positions, whether it writes a cache) and its GPU time between CUDA events recorded on the current
    stream before and after it. Events are resolved in `take()`, so recording does not synchronize."""

    def __init__(self, model: Any, *, head: bool = True) -> None:  # noqa: ANN401 — any torch.nn.Module
        self.model, self.head = model, head  # head=False: a backbone, no vocab logits (kept = 0)
        self._suspended = 0
        self._pending: list[tuple[dict[str, Any], Any, Any]] = []
        self._open: list[tuple[dict[str, Any], Any]] = []
        self._handles = [
            model.register_forward_pre_hook(self._pre, with_kwargs=True),
            model.register_forward_hook(self._post, with_kwargs=True),
        ]

    def _pre(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:  # noqa: ANN401
        import torch  # noqa: PLC0415

        if self._suspended:
            return
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            ids = kwargs["inputs_embeds"]
        rows, tokens = int(ids.shape[0]), int(ids.shape[1])
        cache = kwargs.get("past_key_values")
        past = int(cache.get_seq_length()) if cache is not None and hasattr(cache, "get_seq_length") else 0
        keep = kwargs.get("logits_to_keep", 0)
        kept = int(keep.numel()) if isinstance(keep, torch.Tensor) else (int(keep) or tokens)
        kept = kept if self.head else 0
        use_cache = kwargs.get("use_cache")
        if use_cache is None:
            use_cache = bool(getattr(getattr(module, "config", None), "use_cache", False))
        shape = {"rows": rows, "tokens": tokens, "past": past, "kept": kept, "cached": bool(use_cache)}
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._open.append((shape, start))

    def _post(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:  # noqa: ANN401
        import torch  # noqa: PLC0415

        if self._suspended:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        shape, start = self._open.pop()
        self._pending.append((shape, start, end))

    @contextlib.contextmanager
    def suspended(self) -> Generator[None]:
        """No recording inside: work that is not a decision's (a CUDA graph capture, which must not record
        events either)."""
        self._suspended += 1
        try:
            yield
        finally:
            self._suspended -= 1

    @contextlib.contextmanager
    def span(self, shape: dict[str, Any]) -> Generator[None]:
        """Record the enclosed GPU work as one forward of `shape`, for passes the hooks cannot see (a CUDA
        graph replay never calls the module). The hooks are off inside, so an eager run counts once."""
        import torch  # noqa: PLC0415

        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        self._suspended += 1
        try:
            yield
        finally:
            self._suspended -= 1
            end.record()
            self._pending.append((shape, start, end))

    def take(self) -> list[dict[str, object]]:
        """The forwards recorded since the last call, with `ms` filled in (waits for them to finish)."""
        out: list[dict[str, object]] = []
        for shape, start, end in self._pending:
            end.synchronize()
            out.append({**shape, "ms": float(start.elapsed_time(end))})
        self._pending.clear()
        return out

    def close(self) -> None:
        for h in self._handles:
            h.remove()


def _median_ms(fn: Callable[[], object], seconds: float) -> float:
    """Median time of `fn` over about `seconds` of back-to-back runs, after a short warm-up."""
    import torch  # noqa: PLC0415

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end or len(times) < 5:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times)


def gemm_tflops(n: int, dtype_name: str, device: str) -> float:
    """Sustained TFLOP/s of an n x n x n matmul in `dtype_name` (float32 runs without TF32)."""
    import torch  # noqa: PLC0415

    dtype = getattr(torch, dtype_name)
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")  # true fp32 when dtype is float32; bf16 is unaffected
    try:
        a = torch.randn(n, n, dtype=dtype, device=device)
        b, c = torch.randn_like(a), torch.empty_like(a)
        ms = _median_ms(lambda: torch.matmul(a, b, out=c), CEILING_SECONDS)
    finally:
        torch.set_float32_matmul_precision(precision)
    return 2 * n**3 / (ms / 1e3) / 1e12


def copy_gbs(mib: int, device: str) -> float:
    """Sustained DRAM GB/s of a device-to-device copy of `mib` MiB (counted as read + write)."""
    import torch  # noqa: PLC0415

    n = mib * 2**20
    src = torch.empty(n, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    ms = _median_ms(lambda: dst.copy_(src), CEILING_SECONDS)
    return 2 * n / (ms / 1e3) / 1e9


def measure_ceilings(
    device: str = "cuda:0", *, gemm: int = 8192, fp32_gemm: int = 4096, copy_mib: int = 1024
) -> dict[str, float]:
    """Sustained (median) throughput of the roofline's ceilings on this GPU, with this torch: a square bf16
    GEMM (tensor cores), a square fp32 GEMM with TF32 off (CUDA cores), and a device-to-device copy.
    Needs about max(3 * gemm^2 * 2 B, 2 * copy_mib MiB) free, so run it before loading weights."""
    import torch  # noqa: PLC0415

    out = {
        "peak_bf16_tflops": gemm_tflops(gemm, "bfloat16", device),
        "peak_fp32_tflops": gemm_tflops(fp32_gemm, "float32", device),
        "peak_bw_gbs": copy_gbs(copy_mib, device),
    }
    torch.cuda.empty_cache()
    out |= {
        "ceiling_gemm_n": float(gemm),
        "ceiling_fp32_gemm_n": float(fp32_gemm),
        "ceiling_copy_mib": float(copy_mib),
    }
    return out


def count_flops(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run `fn` under torch's FlopCounterMode: the FLOPs of the matmul, convolution and attention ops the
    dispatcher sees. Custom Triton/CUDA kernels (fla's delta rule, causal-conv1d) are invisible to it, and
    causal attention is counted as the full square, so this is a cross-check, not ground truth."""
    from torch.utils.flop_counter import FlopCounterMode  # noqa: PLC0415

    with FlopCounterMode(display=False) as counter:
        out = fn()
    return out, float(counter.get_total_flops())


def kernel_env(model: Any = None) -> dict[str, str]:  # noqa: ANN401
    """Which fast kernels this process can use; transformers falls back to plain torch without them."""
    import torch  # noqa: PLC0415

    env = {
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    for pkg in ("transformers", "fla", "causal_conv1d", "mamba_ssm", "flash_attn"):
        if importlib.util.find_spec(pkg) is None:
            env[pkg] = "absent"
        else:
            env[pkg] = str(getattr(importlib.import_module(pkg), "__version__", "present"))
    if model is not None:
        env["attn_implementation"] = str(getattr(model.config, "_attn_implementation", "unknown"))
    return env


ATTENTION_HINTS = ("flash", "fmha", "attention", "attn", "sdpa", "efficient", "cudnn_sdp")


def kernel_shares(events: list[tuple[str, float]], top: int | None = 10) -> list[tuple[str, float]]:
    """(kernel name, GPU time) pairs -> the `top` names (None: all) by summed time, each with its share."""
    totals: dict[str, float] = {}
    for name, t in events:
        totals[name] = totals.get(name, 0.0) + t
    whole = sum(totals.values())
    if whole <= 0:
        return []
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return [(name, t / whole) for name, t in ranked]


def attention_kernel(shares: list[tuple[str, float]]) -> str | None:
    """The busiest kernel whose name looks like attention (flash/fmha/efficient/cudnn sdp...), if any."""
    for name, _ in shares:
        if any(h in name.lower() for h in ATTENTION_HINTS):
            return name
    return None


def profile_kernels(fn: Callable[[], object]) -> list[tuple[str, float]]:
    """Run `fn` once under torch.profiler (CUDA activity only) and return (kernel name, device time in us)
    for every CUDA kernel it launched. Uses CUPTI tracing, not hardware counters, so it needs no special
    permission; run it outside timed regions, since tracing adds overhead."""
    import torch  # noqa: PLC0415

    cuda = torch.autograd.DeviceType.CUDA
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = prof.events() or []
    # kernel launches are the events that ran on the device; their own interval is the kernel's duration
    return [
        (str(e.name), float(e.time_range.elapsed_us()))
        for e in events
        if e.device_type == cuda and e.time_range.elapsed_us() > 0
    ]
