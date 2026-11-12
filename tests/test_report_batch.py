"""The batch benchmark's report section: series, panels, tables and the two charts."""

from __future__ import annotations

import math

from s1 import report, report_batch
from s1.record import RunRecord


def _rec(run_id, gpu, model, mode=None, *, host="Intel", tflops=(10.0, 20.0, 40.0), busy=0.3, nsys=False):
    m = {}
    for length in (128, 1024):
        for b, v in zip((1, 4, 16), tflops, strict=True):
            m[f"batch_s{length}_b{b}_tflops_wall"] = v * (1.5 if length == 1024 else 1)
        m[f"batch_s{length}_b1_wall_ms"] = 20.0
        m[f"batch_s{length}_b1_kernel_ms"] = 20.0 * busy
        m[f"batch_s{length}_b1_kernel_busy"] = busy
        m[f"batch_s{length}_b1_kernels"] = 1150.0
    if nsys:
        m = {"batch_nsys_n": 64.0, "batch_nsys_wall_ms": 100.0}
    params = {"model": model} | ({"laya_mode": mode} if mode else {})
    inv = {"gpu": gpu, "platform": "modal"}
    return RunRecord(run_id, inv, params, {"batch": m}, meta={"cpu_model": host})


def test_series_uses_the_shortest_length_and_names_laya_modes():
    s = report_batch.series_of(_rec("r1", "NVIDIA L4", "laya.english", "graphs"))
    assert s is not None
    assert s.key == "laya graphs" and s.name == "Laya CUDA graphs"
    assert s.length == 128
    assert s.points == [(1, 10.0), (4, 20.0), (16, 40.0)]
    assert s.b1["kernel_busy"] == 0.3
    assert report_batch.series_of(_rec("r2", "NVIDIA L4", "laya.english", nsys=True)) is None


def test_panels_keep_the_newest_record_and_order_gpus_and_series():
    recs = [
        _rec("20261109-01", "NVIDIA H100 80GB HBM3", "laya.english", "eager", host="old"),
        _rec("20261109-02", "NVIDIA H100 80GB HBM3", "laya.english", "eager", host="new"),
        _rec("20261109-03", "NVIDIA H100 80GB HBM3", "qwen3-8b.zeroshot"),
        _rec("20261109-04", "NVIDIA H100 80GB HBM3", "laya.english", "compile"),
        _rec("20261109-05", "NVIDIA L4", "laya.english", "eager"),
    ]
    ps = report_batch.panels(recs)
    assert [p.gpu for p in ps] == ["NVIDIA L4", "NVIDIA H100 80GB HBM3"]
    h100 = ps[1]
    assert [s.key for s in h100.series] == ["laya eager", "laya compile", "qwen3-8b.zeroshot"]
    assert h100.series[0].host == "new"


def test_split_clips_gpu_bound_points():
    gpu_bound = report_batch.series_of(_rec("r", "NVIDIA L4", "kev-4b.hybrid", busy=1.15))
    assert gpu_bound is not None
    assert report_batch.split(gpu_bound) == (1.0, 0.0)
    s = report_batch.series_of(_rec("r", "NVIDIA L4", "kev-4b.hybrid", busy=0.25))
    assert s is not None
    assert report_batch.split(s) == (0.25, 0.75)


def test_tables():
    ps = report_batch.panels([_rec("r", "NVIDIA B200", "laya.english", "eager", busy=0.25)])
    h, rows = report_batch.b1_table(ps)
    assert h[-2] == "CPU share" and rows[0][-2] == "0.75"
    h, rows = report_batch.curve_table(ps)
    assert h[2:5] == ["B=1", "B=4", "B=16"]
    assert rows[0][2:] == ["10", "20", "40", "4.0x"]


def test_charts_and_report_sections():
    recs = [
        _rec("a", "NVIDIA L4", "laya.english", "eager", busy=0.4),
        _rec("b", "NVIDIA H100 80GB HBM3", "laya.english", "graphs", busy=0.7),
    ]
    ps = report_batch.panels(recs)
    curves, bars = report_batch.curve_charts(ps), report_batch.split_chart(ps)
    assert curves.count("<svg") == 2  # one small multiple per GPU
    assert "var(--bs-0)" in curves and "var(--bs-1)" in curves  # fixed slots: eager 0, graphs 1
    assert bars.count('class="seg"') == 2 and "40%" in bars
    page = report.to_html(recs)
    assert "Where batch-1 latency goes" in page and "--bs-gpu" in page
    md = report.to_markdown(recs)
    assert "### Batch 1: CPU vs GPU share of latency" in md
    assert "### Achieved TFLOP/s vs batch size" in md
    assert report_batch.curve_charts([]) == "" and report_batch.split_chart([]) == ""
    assert math.isnan(report_batch.split(report_batch.Series("x", "r", "h", 128, [], {}))[0])


def test_host_names_and_mixed_hosts_per_row():
    assert report_batch.host_name("GenuineIntel family 6 model 143") == "Sapphire Rapids"
    assert report_batch.host_name("AuthenticAMD family 25 model 1") == "Zen 3"
    assert report_batch.host_name("Apple M3") == "Apple M3"
    same = report_batch.panels(
        [_rec("a", "NVIDIA L4", "laya.english", "eager", host="AuthenticAMD family 25 model 1")]
    )
    assert "host Zen 3" in report_batch.split_chart(same)
    mixed = report_batch.panels(
        [
            _rec("a", "NVIDIA L4", "laya.english", "eager", host="AuthenticAMD family 25 model 1"),
            _rec("b", "NVIDIA L4", "kev-4b.hybrid", host="GenuineIntel family 6 model 85"),
        ]
    )
    bars = report_batch.split_chart(mixed)
    assert "hosts differ per row" in bars and "Kev-4B (fused + graphs) · Skylake/Cascade Lake" in bars
