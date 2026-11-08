#!/usr/bin/env python3
"""Model-free test of the profiling method: bf16 square matmuls whose FLOPs and bytes we know exactly.

    python ncu_smoke.py run   --sizes 512,1024,2048,4096         # the workload (what ncu / nsys wrap)
    python ncu_smoke.py time  --sizes 512,1024,2048,4096         # unprofiled TFLOP/s with CUDA events
    python ncu_smoke.py parse ncu.csv --sizes 512,1024,2048,4096 # ncu --csv output -> roofline table
    python ncu_smoke.py sample                                   # idle vs busy GpuSampler(detailed) keys

`run` makes, for every size, fresh operands, one warm-up matmul, then a `spin` marker kernel
(torch.cuda._sleep) and the measured matmul. `parse` finds the marker in ncu's per-kernel rows and takes
the GEMM kernels that follow it, so warm-up and operand-generation kernels never reach the table.

Needs only torch (and, for `sample`, the repo's `s1` package on PYTHONPATH). Never loads a model.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Spec-sheet peaks, dense bf16 tensor FLOP/s and DRAM bytes/s, by substring of the GPU name.
# H100 is the SXM5 80 GB part Modal rents ("NVIDIA H100 80GB HBM3"); the PCIe card is lower (756e12, 2.0e12).
PEAKS = {
    "L4": (121e12, 300e9),
    "A10G": (70e12, 600e9),
    "A100": (312e12, 1555e9),
    "H100": (989e12, 3350e9),
}
GEMM = re.compile(r"gemm|cutlass|xmma|nvjet|cublas|sm\d\d_", re.IGNORECASE)
MARKER = re.compile(r"spin", re.IGNORECASE)
UNIT_TO_S = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
    "second": 1.0,
    "msecond": 1e-3,
    "usecond": 1e-6,
    "nsecond": 1e-9,
}
UNIT_TO_B = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KiByte": 1024.0}
TENSOR_HINT = "tensor"


def parse_sizes(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def cmd_run(sizes: list[int]) -> None:
    torch = importlib.import_module("torch")  # lazy: parse mode runs without torch

    torch.manual_seed(0)
    for n in sizes:
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        torch.matmul(a, b)  # warm-up: same shape, so the same cuBLAS kernel choice
        torch.cuda.synchronize()
        torch.cuda._sleep(1000)  # marker kernel: parse looks for "spin"
        torch.matmul(a, b)
        torch.cuda.synchronize()
    print(f"ran sizes {sizes}")


def cmd_time(sizes: list[int]) -> None:
    torch = importlib.import_module("torch")  # lazy: parse mode runs without torch

    props = torch.cuda.get_device_properties(0)
    print(
        f"device {props.name} cc {props.major}.{props.minor} "
        f"torch {torch.__version__} cuda {torch.version.cuda}"
    )
    for n in sizes:
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        for _ in range(5):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        iters = 50
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        s = start.elapsed_time(end) / 1000 / iters
        print(f"n={n} {s * 1e6:.1f} us  {2 * n**3 / s / 1e12:.2f} TFLOP/s (unprofiled)")


def peak_for(device: str) -> tuple[float, float] | None:
    return next((v for k, v in PEAKS.items() if k in device), None)


def metric_value(raw: str) -> float:
    return float(raw.replace(",", "").replace("'", ""))


@dataclass
class Kernel:
    name: str = ""
    device: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


def read_ncu_csv(path: str) -> list[Kernel]:
    """Group `ncu --csv` rows (one per kernel x metric) into one Kernel per launch, in launch order."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    start = text.find('"ID"')  # skip any "==PROF==" lines before the header
    rows = csv.DictReader(text[start:].splitlines()) if start >= 0 else []
    kernels: dict[int, Kernel] = defaultdict(Kernel)
    for r in rows:
        k = kernels[int(r["ID"])]
        k.name, k.device = r["Kernel Name"], r.get("Device Name", r.get("Device", ""))
        unit = r.get("Metric Unit", "")
        try:
            value = metric_value(r["Metric Value"])
        except ValueError:
            continue
        value *= UNIT_TO_S.get(unit, UNIT_TO_B.get(unit, 1.0))
        k.metrics[r["Metric Name"]] = value
    return [kernels[i] for i in sorted(kernels)]


def measured_kernels(kernels: list[Kernel], n_sizes: int) -> list[list[Kernel]]:
    """For each marker, the run of GEMM-looking kernels right after it."""
    groups: list[list[Kernel]] = []
    i = 0
    while i < len(kernels):
        if MARKER.search(kernels[i].name):
            j, group = i + 1, []
            while j < len(kernels) and GEMM.search(kernels[j].name):
                group.append(kernels[j])
                j += 1
            groups.append(group)
            i = j
        else:
            i += 1
    return groups[:n_sizes]


def cmd_parse(path: str, sizes: list[int], gpu: str = "") -> None:
    kernels = read_ncu_csv(path)
    if not kernels:
        print("no kernel rows in the ncu output (profiling failed? see the ncu log)")
        sys.exit(1)
    device = gpu or kernels[0].device  # the CSV's Device column is only an index; pass --gpu "NVIDIA L4"
    peak = peak_for(device)
    groups = measured_kernels(kernels, len(sizes))
    print(f"device: {device}; {len(kernels)} kernels profiled, {len(groups)}/{len(sizes)} sizes found")
    if peak:
        print(
            f"spec peaks (dense bf16): {peak[0] / 1e12:.0f} TFLOP/s, {peak[1] / 1e9:.0f} GB/s, "
            f"ridge {peak[0] / peak[1]:.0f} FLOP/B"
        )
    nan = float("nan")
    print(
        f"{'n':>5} {'kernel':<38} {'us':>8} {'DRAM MB':>8} {'TFLOP/s':>8} {'%peakF':>6} "
        f"{'AI F/B':>7} {'AI ideal':>8} {'sm%':>5} {'cm%':>5} {'tensor%':>7}"
    )
    for n, group in zip(sizes, groups, strict=False):
        if not group:
            print(f"{n:>5} (no GEMM kernel after the marker)")
            continue
        sec = sum(k.metrics.get("gpu__time_duration.sum", 0.0) for k in group)
        dram = sum(
            k.metrics.get("dram__bytes_read.sum", 0.0) + k.metrics.get("dram__bytes_write.sum", 0.0)
            for k in group
        )
        flops = 2 * n**3
        m = group[-1].metrics
        tensor = next((v for k, v in m.items() if TENSOR_HINT in k and "pct" in k), nan)
        sm = m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", nan)
        cm = m.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", nan)
        ach = flops / sec if sec else nan
        pct = f"{100 * ach / peak[0]:.0f}" if peak else "?"
        ai = flops / dram if dram else nan
        print(
            f"{n:>5} {group[-1].name[:38]:<38} {sec * 1e6:>8.1f} {dram / 1e6:>8.2f} {ach / 1e12:>8.2f} "
            f"{pct:>6} {ai:>7.0f} {n / 3:>8.0f} {sm:>5.0f} {cm:>5.0f} {tensor:>7.0f}"
        )
    print(
        "AI ideal = FLOPs / (read A,B + write C in bf16) = n/3; "
        "measured AI is higher when DRAM traffic is lower"
    )


def cmd_sample() -> None:
    """Idle then busy on this node's GPU(s) with the detailed GpuSampler: which new keys move?"""
    torch = importlib.import_module("torch")
    GpuSampler = importlib.import_module("s1.gpu_stats").GpuSampler

    def run_phase(label: str, busy_s: float) -> dict[str, float]:
        with GpuSampler(metrics="detailed") as smi:
            end = time.time() + busy_s
            if label == "busy":
                a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
                while time.time() < end:
                    torch.matmul(a, a)
                torch.cuda.synchronize()
            else:
                time.sleep(busy_s)
        return smi.metrics()

    idle = run_phase("idle", 5.0)
    busy = run_phase("busy", 8.0)
    keys = sorted(set(idle) | set(busy))
    print(f"{'key':<34} {'idle':>12} {'busy':>12}")
    for k in keys:
        if k.endswith(("_std", "_p90", "_p95", "_p99")):
            continue
        print(f"{k:<34} {idle.get(k, float('nan')):>12.2f} {busy.get(k, float('nan')):>12.2f}")
    print(json.dumps({"idle_keys": len(idle), "busy_keys": len(busy)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "time", "parse", "sample"])
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--sizes", default="512,1024,2048,4096")
    ap.add_argument("--gpu", default="", help="GPU name for the spec-peak lookup (parse mode)")
    args = ap.parse_args()
    sizes = parse_sizes(args.sizes)
    if args.mode == "run":
        cmd_run(sizes)
    elif args.mode == "time":
        cmd_time(sizes)
    elif args.mode == "sample":
        cmd_sample()
    else:
        if not args.csv:
            ap.error("parse needs the ncu csv path")
        cmd_parse(args.csv, sizes, args.gpu)


if __name__ == "__main__":
    main()
