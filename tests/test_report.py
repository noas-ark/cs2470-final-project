from s1 import report, report_roofline
from s1.record import RunRecord
from s1.report import DASH, TIMES


def _rec(run_id, model, acc, n, util=None):
    results = {"jeval": {"jeval_accuracy": acc, "jeval_ece": 0.05, "jeval_n": n, "jeval_latency_ms": 1500.0}}
    if util is not None:
        results["jeval/nvidia_smi"] = {"gpu_util_pct": util, "gpu_count": 4.0, "gpu_energy_wh": 36.2}
    return RunRecord(
        run_id,
        {"gpu": "NVIDIA A10G", "quantization": "none"},
        {"model": model, "jeval_limit": n},
        results,
        {"git_commit": "abc123"},
    )


RECS = [
    _rec("20260101-000000-aaaa", "openjev", 0.70, 200, util=97.0),
    _rec("20260102-000000-bbbb", "openjev", 0.77, 2000, util=99.9),
    _rec("20260103-000000-cccc", "otherjev", 0.81, 500),
]


def test_leaderboard_takes_each_models_largest_run_sorted_by_headline():
    t = report.leaderboard("jeval", RECS)
    assert t.header[:3] == ["Model", "Accuracy", "ECE"]
    assert [row[0] for row in t.rows] == ["otherjev", "openjev"]
    assert t.rows[1][1] == "77.0%"  # the 2,000-question run, not the 200 one
    assert t.rows[1][-1] == "20260102-000000-bbbb"


def test_all_runs_newest_first_with_gpu_columns():
    t = report.all_runs("jeval", RECS)
    assert [row[0] for row in t.rows] == [
        "20260103-000000-cccc",
        "20260102-000000-bbbb",
        "20260101-000000-aaaa",
    ]
    util = t.header.index("GPU util")
    assert t.rows[1][util] == "99.9%" and t.rows[0][util] == DASH
    assert f"A10G {TIMES}4, none" in t.rows[1]


def test_formatting():
    assert report.fmt("jeval_latency_ms", 12347.0) == "12.3 s"
    assert report.fmt("jeval_latency_ms_p99", 80.0) == "80 ms"
    assert report.fmt("jeval_n", 2000.0) == "2,000"
    assert report.fmt("gpu_mem_used_mb_max", 20567.0) == "20.1 GiB"
    assert report.fmt("jeval_ece", float("nan")) == DASH


def test_write_is_deterministic(tmp_path):
    for r in RECS:
        r.save(tmp_path)
    md, page = report.write(tmp_path)
    first = (md.read_text(), page.read_text())
    report.write(tmp_path)
    assert (md.read_text(), page.read_text()) == first
    assert "| otherjev | 81.0% |" in first[0]
    assert "<svg" in first[1] and "otherjev" in first[1]


# --- roofline section ---------------------------------------------------------------------------


def _roof_rec(run_id, model, metrics, gpu="NVIDIA L4", **params):
    return RunRecord(run_id, {"gpu": gpu}, {"model": model, **params}, {"roofline": metrics})


ROOF = {
    "roofline_tflops": 30.0,
    "roofline_intensity": 400.0,
    "roofline_t512_intensity": 300.0,
    "roofline_t512_tflops": 25.0,
    "roofline_t1024_intensity": 600.0,
    "roofline_t1024_tflops": 35.0,
    "roofline_params_b": 8.95,
    "roofline_active_params_b": 6.9,
    "roofline_attention_layers": 0.25,
    "roofline_peak_bf16_tflops": 90.0,
    "roofline_peak_fp32_tflops": 25.0,
    "roofline_peak_bw_gbs": 270.0,
    "roofline_spec_bf16_tflops": 121.0,
    "roofline_spec_bw_gbs": 300.0,
    "roofline_accuracy": 0.6,
}


def test_roofline_family_comes_from_the_metrics():
    assert report_roofline.family(ROOF) == 0  # dense, 1 in 4 layers attention
    assert report_roofline.family(ROOF | {"roofline_attention_layers": 1.0}) == 1
    assert report_roofline.family(ROOF | {"roofline_active_params_b": 1.1, "roofline_params_b": 6.9}) == 2
    small_encoder = ROOF | {
        "roofline_active_params_b": 0.11,
        "roofline_params_b": 0.31,
        "roofline_attention_layers": 1.0,
    }
    assert (
        report_roofline.family(small_encoder, "laya.multilingual") == 1
    )  # the catalog says encoder, not MoE
    assert report_roofline.family(ROOF, "olmoe-1b-7b.zeroshot") == 2


NCU = ROOF | {"roofline_hw_intensity": 150.0, "roofline_hw_tflops": 29.0}


def test_roofline_points_pair_clean_runs_with_their_ncu_run():
    recs = [
        _roof_rec("20261104-000001-a", "m1", ROOF),
        _roof_rec("20261104-000002-b", "m1", ROOF | {"roofline_tflops": 31.0}),  # latest clean run wins
        _roof_rec("20261104-000003-c", "m1", NCU, roofline_ncu=4),
        _roof_rec("20261104-000004-d", "m2", ROOF, gpu="NVIDIA A10G"),
    ]
    by_gpu = report_roofline.points_by_gpu(recs)
    assert set(by_gpu) == {"NVIDIA L4", "NVIDIA A10G"}
    (p,) = by_gpu["NVIDIA L4"]
    assert p.m["roofline_tflops"] == 31.0
    assert p.hw["roofline_hw_intensity"] == 150.0
    assert by_gpu["NVIDIA A10G"][0].hw == {}


def test_roofline_chart_and_table():
    pts = report_roofline.points_by_gpu([_roof_rec("20261104-000001-a", "jevk5-9b", ROOF)])["NVIDIA L4"]
    svg = report_roofline.chart("NVIDIA L4", pts)
    assert ">jevk5-9b</text>" in svg  # direct label
    assert "measured: 90 TFLOP/s bf16 tensor, 270 GB/s" in svg
    assert "datasheet: 121 TFLOP/s bf16" in svg
    assert 'class="traj"' in svg  # path through the prompt-length bins
    header, rows = report_roofline.table(pts)
    assert len(rows) == 1
    assert len(rows[0]) == len(header)
    assert rows[0][header.index("Accuracy")] == "60.0%"
    assert rows[0][header.index("HW TFLOP/s")] == report_roofline.DASH


def test_report_includes_the_roofline_section(tmp_path):
    _roof_rec("20261104-000001-a", "jevk5-9b", ROOF).save(tmp_path)
    md, page = report.write(tmp_path)
    assert "### Roofline on NVIDIA L4" in md.read_text()
    assert 'class="card chart roof"' in page.read_text()


def test_recipes_list_decision_models_and_the_control_only():
    recs = [
        _roof_rec("20261104-000001-a", "jevk5-9b", ROOF),
        _roof_rec("20261104-000002-b", "kev-4b.hybrid", ROOF | {"roofline_padding": 0.25}),
        _roof_rec("20261104-000003-c", "olmoe-1b-7b.zeroshot", ROOF),  # a reference, no recipe
        _roof_rec("20261104-000004-d", "qwen35-9b.zeroshot", ROOF),  # the control has one
    ]
    header, rows = report_roofline.recipes(report_roofline.points_by_gpu(recs)["NVIDIA L4"])
    assert [r[0] for r in rows] == ["jevk5-9b", "kev-4b.hybrid", "qwen35-9b.zeroshot"]
    assert all(len(r) == len(header) for r in rows)
    assert rows[1][header.index("Padding")] == "25%"
    assert rows[0][header.index("Padding")] == report_roofline.DASH
    assert report_roofline.is_decision_model("kev-4b.hybrid")
    assert not report_roofline.is_decision_model("olmoe-1b-7b.zeroshot")


def test_report_includes_findings_when_written(tmp_path):
    _roof_rec("20261104-000001-a", "jevk5-9b", ROOF).save(tmp_path)
    md, page = report.write(tmp_path)
    assert "### Findings" not in md.read_text()
    (tmp_path / report_roofline.FINDINGS).write_text("Some **bold** claim.\n\n- one `x`\n- two\n  wrapped\n")
    md, page = report.write(tmp_path)
    assert "### Findings" in md.read_text() and "### Decision recipes on NVIDIA L4" in md.read_text()
    assert "<p>Some <strong>bold</strong> claim.</p>" in page.read_text()
    assert "<ul><li>one <code>x</code></li><li>two wrapped</li></ul>" in page.read_text()


def test_findings_nested_lists():
    text = "- a\n  - b\n    wrapped\n    - c\n  - d\n- e\n"
    assert report_roofline.markdown_to_html(text) == (
        "<ul><li>a<ul><li>b wrapped<ul><li>c</li></ul></li><li>d</li></ul></li><li>e</li></ul>"
    )
