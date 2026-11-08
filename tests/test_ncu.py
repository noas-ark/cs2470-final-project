"""Parsing Nsight Compute's raw CSV and summing it into roofline_hw_* metrics. Values from a real run
(logs/probe3: a 4096^3 bf16 GEMM and a 1024^3 fp32 GEMM on an A10G)."""

import math

import pytest

from s1 import ncu
from s1.cli import _scalar

HEADER = [
    "ID",
    "Kernel Name",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__time_duration.sum",
    "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
]
UNITS = ["", "", "byte", "byte", "ns", "", "inst", "inst"]
BF16_GEMM = ["0", "ampere_bf16_s16816gemm", "248846208", "40539904", "2848832", "137438953472", "0", "0"]
SGEMM = ["1", "ampere_sgemm_128x64_nn", "19724160", "15972864", "149568", "0", "1094713344", "5242880"]


def csv_text(*rows):
    lines = ["==PROF== Connected to process 1", "done NVIDIA A10G"]
    lines += [",".join(f'"{v}"' for v in r) for r in (HEADER, UNITS, *rows)]
    return "\n".join(lines) + "\n"


def test_parse_skips_log_lines_and_units():
    kernels = ncu.parse(csv_text(BF16_GEMM, SGEMM))
    assert len(kernels) == 2
    assert kernels[0]["sm__ops_path_tensor_src_bf16_dst_fp32.sum"] == 2 * 4096**3
    assert "Kernel Name" not in kernels[0]


def test_parse_without_profiled_kernels():
    with pytest.raises(ValueError, match="no ncu CSV"):
        ncu.parse("==PROF== Disconnected\n")


def test_summarize_flops_bytes_and_ratios():
    kernels = ncu.parse(csv_text(BF16_GEMM, SGEMM))
    run = {
        "roofline_hw_theory_gflop": 137.4389 + 2.1475,
        "roofline_hw_theory_mb": 100.0,
        "roofline_hw_timed_ms": 3.0,
    }
    m = ncu.summarize(kernels, run)
    flops = 2 * 4096**3 + 2 * 1094713344 + 5242880
    dram = 248846208 + 40539904 + 19724160 + 15972864
    assert next(iter(m)) == "roofline_hw_tflops"
    assert m["roofline_hw_gflop"] == pytest.approx(flops / 1e9)
    assert m["roofline_hw_dram_mb"] == pytest.approx(dram / 1e6)
    assert m["roofline_hw_intensity"] == pytest.approx(flops / dram)
    assert m["roofline_hw_tflops"] == pytest.approx(flops / 3e9)
    assert m["roofline_hw_tensor_share"] == pytest.approx(2 * 4096**3 / flops)
    # the counters match 2 * n^3 to within the sgemm's alpha/beta epilogue
    assert m["roofline_hw_flop_ratio"] == pytest.approx(1.0, abs=0.02)
    assert m["roofline_hw_byte_ratio"] == pytest.approx(dram / 1e8)
    assert m["roofline_hw_kernels"] == 2
    assert m["roofline_hw_kernel_ms"] == pytest.approx((2848832 + 149568) / 1e6)


def test_summarize_without_run_numbers_is_nan_not_an_error():
    m = ncu.summarize(ncu.parse(csv_text(BF16_GEMM)), {})
    assert math.isnan(m["roofline_hw_tflops"])
    assert math.isnan(m["roofline_hw_flop_ratio"])


def test_ncu_profiles_children_from_a_range():
    for clock in ncu.CLOCK_CONTROLS:
        args = list(ncu.args(clock))
        assert args[args.index("--profile-from-start") + 1] == "off"
        assert args[args.index("--target-processes") + 1] == "all"  # elephant-1's worker is a child


@pytest.fixture
def on_slurm(monkeypatch):
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "4233")


@pytest.fixture
def on_modal(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("MODAL_TASK_ID", "ta-01TEST")


def test_clock_control_default_is_unchanged_off_modal(on_slurm):
    assert ncu.clock_control() == "base"
    assert ncu.args() == ncu.BASE_ARGS  # no --clock-control flag: the cluster command line is as before
    assert "--clock-control" not in ncu.ARGS  # tests import ncu off Modal


def test_clock_control_is_none_on_modal(on_modal):
    assert ncu.clock_control() == "none"
    args = list(ncu.args())
    assert args[args.index("--clock-control") + 1] == "none"
    assert args[2:] == list(ncu.BASE_ARGS)


def test_explicit_clock_control_wins(on_modal):
    assert ncu.clock_control("base") == "base"
    assert ncu.args("none")[:2] == ("--clock-control", "none")


def test_explicit_none_off_modal(on_slurm):
    assert ncu.args("none")[:2] == ("--clock-control", "none")


def test_explicit_base_on_modal_adds_no_flag(on_modal):
    assert ncu.args("base") == ncu.BASE_ARGS


def test_bad_clock_control():
    with pytest.raises(ValueError, match="ncu_clock_control"):
        ncu.clock_control("boost")


def test_clock_control_is_recorded_on_modal(on_modal):
    kernels = ncu.parse(csv_text(BF16_GEMM))
    assert ncu.summarize(kernels, {})["roofline_hw_clocks_locked"] == 0.0  # Modal default: none
    assert ncu.summarize(kernels, {}, clock="base")["roofline_hw_clocks_locked"] == 1.0


def test_clock_control_recorded_as_locked_on_slurm(on_slurm):
    assert ncu.summarize(ncu.parse(csv_text(BF16_GEMM)), {})["roofline_hw_clocks_locked"] == 1.0


@pytest.mark.parametrize(
    ("text", "val"), [("4", 4), ("1.5", 1.5), ("true", True), ("false", False), ("L4", "L4")]
)
def test_set_values(text, val):
    assert _scalar(text) == val
    assert type(_scalar(text)) is type(val)
