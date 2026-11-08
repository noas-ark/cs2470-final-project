import time
from typing import override

import pytest

from s1.gpu_stats import (
    GpuSampler,
    Sample,
    detail_fields,
    metrics_mode,
    parse_dmon_line,
    parse_help_query_fields,
    parse_line,
    resolve_mode,
    sample_point,
    summarize_samples,
)


def test_parse_line():
    s = parse_line("3, 87, 20480, 23034, 71.25, 66\n")
    assert s is not None and (s.gpu, s.util_pct, s.mem_used_mb, s.power_w) == (3, 87.0, 20480.0, 71.25)
    assert parse_line("0, [N/A], 1, 2, 3, 4") is None
    assert parse_line("") is None


def test_sampler_reads_fake_nvidia_smi(fake_nvidia_smi):
    with GpuSampler(cmd=fake_nvidia_smi) as smi:
        deadline = time.monotonic() + 5
        while len(smi.samples) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
    m = smi.metrics()
    assert m["gpu_count"] == 2
    assert m["gpu_samples"] == 4  # the [N/A] row is dropped
    assert m["gpu_util_pct"] == 60.0
    assert m["gpu_util_pct_p99"] == pytest.approx(89.4)
    assert m["gpu_mem_used_mb_max"] == 4000
    assert m["gpu_power_w_total_mean"] == 65.0 + 65.0
    assert m["gpu_temp_c_max"] == 55


def test_energy_uses_window_length():
    m = summarize_samples([Sample(0, 0, 0, 0, 100.0, 0), Sample(1, 0, 0, 0, 200.0, 0)], duration_s=36.0)
    assert m["gpu_energy_wh"] == pytest.approx(3.0)  # 300 W for 36 s


def test_no_nvidia_smi_means_no_metrics():
    with GpuSampler(cmd=["/nonexistent/nvidia-smi"]) as smi:
        pass
    assert smi.metrics() == {}


EXTRA = ("utilization.memory", "clocks.sm", "clocks.mem", "clocks_event_reasons.active")


def wait_for(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.01)


def detailed(cmd, pcie_cmd=None):
    return GpuSampler(cmd=cmd, metrics="detailed", extra=EXTRA, pcie_cmd=pcie_cmd or ["/nonexistent/dmon"])


def test_parse_line_detailed_columns():
    s = parse_line("0, 90, 2000, 23034, 70.0, 50, 40, 1500, 6250, 0x4", EXTRA)
    assert s is not None
    assert (s.mem_util_pct, s.clock_sm_mhz, s.clock_mem_mhz, s.throttle_mask) == (40.0, 1500.0, 6250.0, 4)
    s = parse_line("0, 90, 2000, 23034, 70.0, 50, [N/A], 1500, 6250, [N/A]", EXTRA)
    assert s is not None and s.mem_util_pct is None and s.throttle_mask is None and s.clock_sm_mhz == 1500
    assert parse_line("0, 90, 2000, 23034, 70.0, 50", EXTRA) is None  # column count must match
    assert parse_line("0, 90, 2000, 23034, 70.0, 50, 1, 2, 3, 0x0") is None  # basic parse rejects extras


def test_parse_dmon_line():
    p = parse_dmon_line("    2     100     250\n")
    assert p is not None and (p.gpu, p.rx_mb_s, p.tx_mb_s) == (2, 100.0, 250.0)
    assert parse_dmon_line("# gpu   rxpci   txpci") is None
    assert parse_dmon_line("    1       -       -") is None
    assert parse_dmon_line("") is None


def test_help_query_parsing_and_field_choice():
    text = '"clocks.current.sm" or "clocks.sm"\n"utilization.memory"\n"clocks_throttle_reasons.active"\n'
    listed = parse_help_query_fields(text)
    assert detail_fields(listed) == ("utilization.memory", "clocks.sm", "clocks_throttle_reasons.active")
    both = listed | {"clocks_event_reasons.active"}
    assert detail_fields(both)[-1] == "clocks_event_reasons.active"  # newer name preferred
    assert detail_fields(frozenset()) == ()  # nothing listed -> only the basic query


def test_detailed_per_gpu_summaries(fake_nvidia_smi_detailed, fake_dmon):
    with detailed(fake_nvidia_smi_detailed, fake_dmon) as smi:
        wait_for(lambda: len(smi.samples) >= 4 and len(smi.pcie_samples) >= 3)
    m = smi.metrics()
    assert m["gpu0_util_pct"] == 80.0 and m["gpu1_util_pct"] == 20.0  # busy vs idle GPU
    assert m["gpu_util_pct_spread"] == 60.0
    assert m["gpu0_util_pct_max"] == 90.0
    assert m["gpu0_power_w"] == 80.0 and m["gpu1_power_w"] == 40.0
    assert m["gpu0_mem_util_pct"] == 50.0 and m["gpu1_mem_util_pct"] == 4.0
    assert m["gpu0_clock_sm_mhz"] == 1400.0 and m["gpu_clock_sm_mhz_min"] == 300.0
    assert m["gpu0_clock_mem_mhz"] == 6250.0 and m["gpu1_clock_mem_mhz"] == 810.0
    assert m["gpu0_throttle_frac"] == 0.5  # power cap on 1 of 2 samples
    assert m["gpu1_throttle_frac"] == 0.0  # idle bit alone is not throttling; [N/A] row is ignored
    assert m["gpu_throttle_mask_seen"] == 5.0
    assert m["gpu0_pcie_rx_mb_s"] == 200.0 and m["gpu0_pcie_tx_mb_s_max"] == 400.0
    assert m["gpu1_pcie_rx_mb_s"] == 0.0  # the "-" row is dropped, the 0 row stays
    assert m["gpu_pcie_rx_mb_s_total_mean"] == 200.0 and m["gpu_pcie_samples"] == 3


def test_pooled_keys_unchanged_in_detailed_mode(fake_nvidia_smi_detailed):
    with detailed(fake_nvidia_smi_detailed) as smi:
        wait_for(lambda: len(smi.samples) >= 4)
    plain = summarize_samples(smi.samples, smi.metrics()["gpu_sampled_s"])
    m = smi.metrics()
    assert {k: m[k] for k in plain} == plain  # every basic key keeps its value and name
    assert m["gpu_util_pct"] == 50.0 and m["gpu_count"] == 2


def test_basic_mode_output_is_byte_identical(fake_nvidia_smi):
    with GpuSampler(cmd=fake_nvidia_smi) as smi:
        wait_for(lambda: len(smi.samples) >= 4)
    assert smi.pcie_cmd is None and smi.detail_metrics() == {}
    assert smi.metrics() == summarize_samples(smi.samples, smi._t1 - smi._t0)
    assert not any(k.startswith(("gpu0_", "gpu1_")) or "pcie" in k or "clock" in k for k in smi.metrics())


def test_missing_dmon_and_fields_give_basic_plus_partial(fake_nvidia_smi):
    # detailed mode with plain 6-column output and no dmon: per-GPU keys appear, optional ones do not
    with GpuSampler(cmd=fake_nvidia_smi, metrics="detailed", extra=(), pcie_cmd=["/nonexistent"]) as smi:
        wait_for(lambda: len(smi.samples) >= 4)
    m = smi.metrics()
    assert m["gpu0_util_pct"] == 70.0 and m["gpu_count"] == 2
    assert not any("pcie" in k or "clock" in k or "throttle" in k or "mem_util" in k for k in m)


def test_detailed_without_nvidia_smi_is_empty():
    with GpuSampler(cmd=["/nonexistent/nvidia-smi"], metrics="detailed", extra=EXTRA) as smi:
        pass
    assert smi.metrics() == {}


def test_metrics_mode_sets_the_default_and_validates():
    assert GpuSampler(cmd=["/nonexistent"]).detailed is False
    with metrics_mode("detailed"):
        assert GpuSampler(cmd=["/nonexistent"]).detailed is True
    assert GpuSampler(cmd=["/nonexistent"]).detailed is False
    with pytest.raises(ValueError, match="gpu_metrics"):
        resolve_mode("verbose")
    with pytest.raises(ValueError, match="schema"):
        resolve_mode("detailed", schema=2)


def test_sample_point_carries_detailed_keys_with_basic_selection(fake_nvidia_smi_detailed, fake_dmon):

    class Waiting(GpuSampler):
        @override
        def __enter__(self):
            super().__enter__()
            wait_for(lambda: len(self.samples) >= 4 and len(self.pcie_samples) >= 3)
            return self

    def factory():
        return Waiting(cmd=fake_nvidia_smi_detailed, metrics="detailed", extra=EXTRA, pcie_cmd=fake_dmon)

    with sample_point("p_", ("gpu_util_pct",), factory) as pt:
        pass
    assert pt.metrics["p_gpu_util_pct"] == 50.0
    assert pt.metrics["p_gpu0_util_pct"] == 80.0 and pt.metrics["p_gpu1_pcie_rx_mb_s"] == 0.0
    assert "p_gpu_energy_wh" not in pt.metrics  # basic selection still applies to basic keys
