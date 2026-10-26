import math
import time
from typing import override

import pytest

from s1 import config, registry, report, runner
from s1.benchmarks import shape_sweep
from s1.benchmarks.shape_sweep import SyntheticStates, WordCounter, check_answers, make_questions, sweep
from s1.gpu_stats import GpuSampler, sample_point
from s1.record import RunRecord, load_all


class FakeModel:
    """Answers every choice question 'a'; records each request's shape."""

    def __init__(self, max_state_tokens=None, fail_over=None):
        self.calls = []
        self.fail_over = fail_over
        if max_state_tokens is not None:
            self.max_state_tokens = max_state_tokens

    def decide(self, state, questions):
        n = len(state.split())
        self.calls.append((n, len(questions)))
        if self.fail_over is not None and n > self.fail_over:
            raise RuntimeError("context too long")
        return {k: {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}} for k in questions}


class Clock:
    """Each call advances 0.25 s, so min_point_s maps to a predictable number of reps."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.25
        return self.t


def test_state_hits_target_and_is_deterministic():
    c = WordCounter()
    for n in (1, 32, 2048):
        s = SyntheticStates(seed=3).state(n, c)
        assert c.count(s) == n and s == SyntheticStates(seed=3).state(n, c)
    assert SyntheticStates(seed=4).state(32, c) != SyntheticStates(seed=3).state(32, c)


def test_state_fits_a_nonlinear_counter():
    class Pairs:  # 2 words per "token", rounded up
        name = "pairs"

        def count(self, text):
            return (len(text.split()) + 1) // 2

    assert Pairs().count(SyntheticStates().state(100, Pairs())) == 100


def test_questions_match_jeval_shape():
    qs = make_questions(3)
    assert list(qs) == ["q0", "q1", "q2"]
    for q in qs.values():
        assert isinstance(q, dict) and q["type"] == "choice" and q["instructions"]
        assert len(q["criteria"]) >= 2


def test_check_answers_flags_missing_and_bad_choices():
    qs = make_questions(3)
    ans = {"q0": {"choice": "a"}, "q1": {"choice": "zzz"}}
    assert check_answers(qs, ans) == 2


def test_parse_int_list():
    assert shape_sweep.parse_int_list({"grid_x": "32,128"}, "x") == [32, 128]
    with pytest.raises(KeyError, match="grid"):
        shape_sweep.parse_int_list({}, "x")
    with pytest.raises(ValueError):
        shape_sweep.parse_int_list({"grid_x": "a,2"}, "x")
    with pytest.raises(ValueError):
        shape_sweep.parse_int_list({"grid_x": "0"}, "x")


def test_sweep_metrics_layout_and_per_question_latency():
    model = FakeModel()
    m = sweep(model, [8, 16], [1, 4], warmup=1, min_point_s=0.0, clock=Clock())
    keys = list(m)
    assert keys[:4] == [
        "shape_sweep_latency_ms",
        "shape_sweep_n",
        "shape_sweep_invalid",
        "shape_sweep_skipped",
    ]
    assert m["shape_sweep_n"] == 4 + 4 + 4 + 4 - 6  # one request per point: 1+4+1+4 = 10 questions
    assert m["shape_sweep_n"] == 10 and m["shape_sweep_invalid"] == 0 and m["shape_sweep_skipped"] == 0
    assert m["shape_sweep_tokenizer_approx_words"] == 1.0
    assert m["shape_sweep_t16_q4_state_tokens_actual"] == 16
    assert m["shape_sweep_t16_q4_request_ms"] == pytest.approx(m["shape_sweep_t16_q4_latency_ms"] * 4)
    assert m["shape_sweep_t8_q1_reps"] == 1
    assert len(model.calls) == 4 + 1  # four points plus one discarded warm-up
    assert model.calls[0] == model.calls[1] == (8, 1)  # warm-up uses the first point's shape
    assert not any(math.isnan(v) for v in m.values())


def test_points_repeat_until_min_point_s():
    # each clock() call is 0.25 s: a rep costs 2 ticks = 0.5 s, so 2 s needs several reps
    m = sweep(FakeModel(), [8], [2], warmup=0, min_point_s=2.0, clock=Clock())
    assert m["shape_sweep_t8_q2_reps"] >= 3
    assert m["shape_sweep_n"] == m["shape_sweep_t8_q2_reps"] * 2


def test_points_above_max_state_tokens_are_skipped_not_invalid():
    model = FakeModel(max_state_tokens=16)
    m = sweep(model, [8, 32], [1, 2], warmup=0, min_point_s=0.0, clock=Clock())
    assert m["shape_sweep_skipped"] == 2 and m["shape_sweep_invalid"] == 0
    assert m["shape_sweep_n"] == 3
    assert m["shape_sweep_t32_q2_skipped"] == 1.0
    assert "shape_sweep_t32_q2_latency_ms" not in m
    assert all(n <= 16 for n, _ in model.calls)


def test_failed_requests_are_invalid_and_excluded_from_latency():
    m = sweep(FakeModel(fail_over=8), [8, 16], [2], warmup=0, min_point_s=0.0, clock=Clock())
    assert m["shape_sweep_invalid"] == 2 and m["shape_sweep_n"] == 4
    assert m["shape_sweep_t16_q2_invalid"] == 2
    assert math.isnan(m["shape_sweep_t16_q2_latency_ms"])
    assert m["shape_sweep_latency_ms"] == m["shape_sweep_t8_q2_latency_ms"]


def test_sweep_records_gpu_metrics_per_point(fake_nvidia_smi):
    samplers: list[GpuSampler] = []

    def factory():
        samplers.append(GpuSampler(cmd=fake_nvidia_smi))
        return samplers[-1]

    class Slow(FakeModel):
        @override
        def decide(self, state, questions):
            deadline = time.monotonic() + 5  # wait for the fake nvidia-smi rows to be read
            while len(samplers[-1].samples) < 4 and time.monotonic() < deadline:
                time.sleep(0.01)
            return super().decide(state, questions)

    m = sweep(Slow(), [8], [1], warmup=0, min_point_s=0.0, gpu_sampler=factory)
    assert m["shape_sweep_t8_q1_gpu_count"] == 2
    assert m["shape_sweep_t8_q1_gpu_samples"] == 4
    assert m["shape_sweep_t8_q1_gpu_util_pct"] == 60.0
    assert "shape_sweep_t8_q1_gpu_energy_wh" not in m  # only the selected keys


def test_sample_point_without_nvidia_smi_is_empty():
    with sample_point("p_", sampler=lambda: GpuSampler(cmd=["/nonexistent/nvidia-smi"])) as point:
        pass
    assert point.metrics == {}


def test_end_to_end_run_records_grid_and_reports(tmp_path):
    registry.MODELS["fake_shape"] = lambda settings: FakeModel()
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        'benchmarks = ["shape_sweep"]\n[invariants]\ngpu = "test"\n'
        '[params]\nmodel = "fake_shape"\nshape_min_point_s = 0.0\nshape_warmup = 0\n'
        "[grid]\nstate_tokens = [8, 16]\nquestions_per_request = [1, 2]\n"
    )
    try:
        (spec,) = config.load(cfg)  # grid axes are not expanded into runs
        assert spec.grid == {"state_tokens": [8, 16], "questions_per_request": [1, 2]}
        runner.run(
            spec,
            results_dir=tmp_path / "out",
            log=lambda _: None,
            gpu_sampler=lambda: GpuSampler(cmd=["/nonexistent"]),
        )
    finally:
        registry.MODELS.pop("fake_shape", None)
    (rec,) = load_all(tmp_path / "out")
    assert rec.grid == spec.grid and "grid_state_tokens" not in rec.params
    assert rec.results["shape_sweep"]["shape_sweep_n"] == 6

    page = report.to_markdown([rec])
    assert "shape_sweep_t16_q2_latency_ms" in page  # per-point keys under Run details
    assert "state_tokens=8|16" in page
    t = report.leaderboard("shape_sweep", [rec])
    assert t.header[:2] == ["Model", "Latency mean"]


def _latency_rec(run_id, model, ms, n=10.0):
    return RunRecord(run_id, {"gpu": "x"}, {"model": model}, {"b": {"b_latency_ms": ms, "b_n": n}})


def test_latency_headline_sorts_ascending():
    recs = [
        _latency_rec("1", "slow", 900.0),
        _latency_rec("2", "fast", 100.0),
        _latency_rec("3", "nan", math.nan),
    ]
    assert [report.model_of(r) for r in report.best_per_model("b", recs)] == ["fast", "slow", "nan"]
    assert "lower is better" in report._bar_chart("b", recs)


def test_report_format_of_counts():
    assert report.fmt("shape_sweep_t8_q1_state_tokens_actual", 2048.0) == "2,048"
    assert report.fmt("shape_sweep_t8_q1_reps", 12.0) == "12"
