"""The roofline benchmark's questions-per-request axis (benchmarks/roofline_q.py), the T5Gemma 2 cost model
and Bongard's pass shapes, and the Q-axis report. No torch."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import override

import pytest

from s1 import flops, registry, report, report_lengths, report_qpr, runner
from s1.benchmarks import jeval, roofline, roofline_q, shape_sweep
from s1.config import RunSpec
from s1.models.backends import bongard
from s1.record import RunRecord
from s1.registry import Settings
from tests.test_roofline import FakeProfiled, _no_gpu, row

# backbone/config.json of AgentBull/bongard-mini at the catalog revision: the fields the cost model reads
_GEMMA3 = {
    "hidden_size": 2560,
    "num_hidden_layers": 34,
    "vocab_size": 262144,
    "intermediate_size": 10240,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "sliding_window": 1024,
    "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 5 + ["sliding_attention"] * 4,
}
T5GEMMA2 = {
    "model_type": "t5gemma2",
    "encoder": {"model_type": "t5gemma2_encoder", "text_config": {"model_type": "t5gemma2_text", **_GEMMA3}},
    "decoder": {"model_type": "t5gemma2_decoder", **_GEMMA3},
}
# backbone/model.safetensors holds 7,508,928,880 parameters; the vision tower (416,866,032) and the multimodal
# projector (2,950,272) never run on text
BONGARD_TEXT_PARAMS = 7_508_928_880 - 416_866_032 - 2_950_272


# --- cost model -----------------------------------------------------------------------------------


def test_t5gemma2_text_parameters_match_the_checkpoint():
    parts = flops.stacks(T5GEMMA2)
    assert set(parts) == {"encoder", "cross_kv", "decoder"}
    assert flops.merged(parts).params == BONGARD_TEXT_PARAMS


def test_t5gemma2_refuses_a_single_arch_and_single_models_have_one_stack():
    with pytest.raises(KeyError, match="several stacks"):
        flops.arch(T5GEMMA2)
    from tests.test_flops import QWEN3_8B  # noqa: PLC0415

    assert flops.stacks(QWEN3_8B) == {"": flops.arch(QWEN3_8B)}


def test_t5gemma2_encoder_is_bidirectional_with_sliding_bands():
    enc = flops.stacks(T5GEMMA2)["encoder"]
    assert not any(ly.causal for ly in enc.layers)
    assert sum(ly.window == 0 for ly in enc.layers) == 5  # full-attention layers
    assert {ly.window for ly in enc.layers} == {0, 1024}


def test_shared_past_reads_the_state_once_for_every_row():
    dec = flops.stacks(T5GEMMA2)["decoder"]
    own = flops.cost(dec, flops.Forward(rows=8, tokens=100, past=2048, kept=0))
    shared = flops.cost(dec, flops.Forward(rows=8, tokens=100, past=2048, kept=0, shared_past=True))
    assert shared.flops == own.flops  # every row still attends to every state key
    kv_bytes = sum(2 * ly.kv_dim * 2048 * 2 for ly in dec.layers)
    assert own.bytes - shared.bytes == pytest.approx(7 * kv_bytes)


def test_stack_cost_routes_each_forward_and_rejects_unknown_stacks():
    parts = flops.stacks(T5GEMMA2)
    enc = bongard.encoder_pass(500) | {"ms": 1.0}
    dec = bongard.decoder_pass([90, 100, 80], 500) | {"ms": 2.0}
    got = roofline_q.stack_cost(parts, [enc, dec])
    want = flops.cost(parts["encoder"], flops.Forward(1, 500, kept=0, ms=1.0)) + flops.cost(
        parts["decoder"], flops.Forward(3, 100, 500, 0, True, 2.0, shared_past=True)
    )
    assert got == want
    with pytest.raises(KeyError, match="stack 'vision'"):
        roofline_q.stack_cost(parts, [enc | {"stack": "vision"}])


def test_bongard_pass_shapes():
    assert bongard.decoder_pass([10, 30, 20], 700) == {
        "stack": "decoder",
        "rows": 3,
        "tokens": 30,
        "past": 700,
        "kept": 0,
        "cached": True,
        "shared_past": True,
        "real_tokens": 60,
    }
    assert bongard.cross_kv_pass(1, 700)["stack"] == "cross_kv"
    assert bongard.encoder_pass(700)["tokens"] == 700


# --- plan -----------------------------------------------------------------------------------------


def _rows(n: int = 60) -> list[jeval.Row]:
    rows = [
        row(i, "noul" if i % 3 == 0 else "choice", n_options=2 + i % 4, state_words=300) for i in range(n)
    ]
    return [dataclasses.replace(r, state=" ".join(f"{r.id}w{j}" for j in range(300))) for r in rows]


BAND: Settings = {"roofline_qpr_min_question_tokens": 1, "roofline_qpr_max_question_tokens": 50}


def test_plan_reuses_the_same_states_and_question_prefixes_at_every_point():
    counter = shape_sweep.WordCounter()
    settings: Settings = {**BAND, "roofline_qpr_requests": 3, "roofline_qpr_warmup": 2}
    points, skipped = roofline_q.plan(_rows(), counter, settings, [32, 128], [1, 4, 9])
    assert skipped == []
    assert [(t[0].length, t[0].q) for _, t in points] == [
        (32, 1),
        (32, 4),
        (32, 9),
        (128, 1),
        (128, 4),
        (128, 9),
    ]
    for warm, timed in points:
        assert len(warm) == 2 and len(timed) == 3
        assert warm[0].state not in {r.state for r in timed}  # warm-ups never on a timed state
        assert all(
            len(r.questions) == r.q and list(r.questions) == [f"q{i}" for i in range(r.q)] for r in timed
        )
    by_len = {n: [[r.state.split()[0] for r in t] for _, t in points if t[0].length == n] for n in (32, 128)}
    assert by_len[32][0] == by_len[32][1] == by_len[32][2] == by_len[128][0]  # the same rows everywhere
    assert {counter.count(r.state) for _, t in points for r in t} == {32, 128}
    q9 = points[2][1][0].questions
    assert dict(list(q9.items())[:4]) == dict(points[1][1][0].questions)  # Q+1 adds, never swaps
    assert all(isinstance(q, dict) and "type" in q for q in q9.values())  # single question objects


def test_plan_caps_q_per_length_and_rejects_bad_axes():
    counter = shape_sweep.WordCounter()
    settings: Settings = {**BAND, "roofline_qpr_requests": 2, "roofline_qpr_max_questions_at_128": 4}
    points, skipped = roofline_q.plan(_rows(), counter, settings, [32, 128], [1, 4, 9])
    assert skipped == [(128, 9)]
    assert len(points) == 5
    with pytest.raises(ValueError, match="positive"):
        roofline_q.plan(_rows(), counter, settings, [32], [0, 4])


def test_question_pool_keeps_the_token_band_and_raises_when_short():
    rows = _rows()
    lens = sorted({len(json.dumps(r.questions).split()) for r in rows})
    lo = lens[1]
    pool = roofline_q.question_pool(
        rows, shape_sweep.WordCounter().count, {"roofline_qpr_min_question_tokens": lo}, 5, set()
    )
    assert all(len(json.dumps({"q": q}).split()) >= lo for q in pool)
    with pytest.raises(ValueError, match="needs 500"):
        roofline_q.question_pool(rows, shape_sweep.WordCounter().count, BAND, 500, set())


def test_invalid_answers_counts_missing_and_malformed():
    assert roofline_q.invalid_answers({"q0": {"type": "noul", "noul": 0.5}, "q1": 3}, ["q0", "q1", "q2"]) == 2
    assert roofline_q.invalid_answers(None, ["q0"]) == 1


def test_fit_and_marginal_cost():
    assert roofline_q.fit([1, 2, 3], [5, 7, 9]) == (2.0, 3.0)
    assert all(math.isnan(v) for v in roofline_q.fit([4, 4], [1, 2]))
    cost = flops.Cost(1e12, 1e9, 10.0, 100)

    def s(q: int, ms: float) -> roofline_q.Sample:
        return roofline_q.Sample(64, q, dataclasses.replace(cost, ms=ms), 1, ms * 2, 0, 100.0)

    m = roofline_q.point_metrics([s(1, 10.0), s(1, 10.0), s(2, 11.0), s(4, 13.0)])
    assert m["roofline_s64_qslope_gpu_ms"] == pytest.approx(1.0)
    assert m["roofline_s64_qmarginal"] == pytest.approx(0.1)  # an added question costs a tenth of a request
    assert m["roofline_s64_q4_gpu_ms_per_question"] == pytest.approx(13.0 / 4)
    assert m["roofline_s64_qcost_ratio"] == pytest.approx(13.0 / 4 / 10.0)
    assert m["roofline_s64_q1_n"] == 2
    assert m["roofline_s64_qmax"] == 4


# --- end to end -----------------------------------------------------------------------------------


class SharedStateProfiled(FakeProfiled):
    """One forward per request: the state's words once plus 10 tokens per question (a shared-state design)."""

    def __init__(self):
        super().__init__()
        self.resets = 0
        self.calls: list[int] = []

    @override
    def decide(self, state, questions):
        self.calls.append(len(questions))
        tokens = len(state.split()) + 10 * len(questions)
        self.pending.append(
            {"rows": 1, "tokens": tokens, "past": 0, "kept": 1, "cached": False, "ms": tokens / 50}
        )
        return {k: {"type": "noul", "noul": 0.5} for k in questions}

    def after_warmup(self):
        self.resets += 1
        self.pending.append({"rows": 1, "tokens": 999, "past": 0, "kept": 1, "cached": False, "ms": 99.0})


def _run(tmp_path, monkeypatch, fake, grid, **params):
    monkeypatch.setattr(jeval, "load_rows", _rows)
    registry.MODELS["fake-q"] = lambda settings: fake
    try:
        spec = RunSpec(
            benchmarks=["roofline"],
            invariants={"gpu": "NVIDIA L4"},
            params={"model": "fake-q", "roofline_qpr_requests": 2, **BAND, **params},
            grid=grid,
        )
        return runner.run(spec, results_dir=tmp_path, log=lambda _: None, gpu_sampler=_no_gpu)
    finally:
        registry.MODELS.pop("fake-q")


GRID = {roofline.LENGTH_AXIS: [32, 128], roofline.QPR_AXIS: [1, 2, 8]}


def test_q_axis_end_to_end(tmp_path, monkeypatch):
    fake = SharedStateProfiled()
    rec = _run(tmp_path, monkeypatch, fake, GRID)
    m = rec.results["roofline"]
    assert m["roofline_n"] == 2 * 2 * (1 + 2 + 8)  # questions: lengths x requests x sum(Q)
    assert m["roofline_requests"] == 12
    assert m["roofline_invalid"] == 0
    assert fake.resets == 12  # before every timed request, and its passes never charged
    assert fake.calls == [q for q in (1, 2, 8) for _ in range(3)] * 2  # warm-up + 2 timed per point
    assert m["roofline_s128_q8_tokens"] == 128 + 80
    assert m["roofline_s128_q8_gpu_ms_median"] == pytest.approx((128 + 80) / 50)
    assert m["roofline_s128_qslope_gpu_ms"] == pytest.approx(10 / 50)
    assert m["roofline_s32_qmarginal"] > m["roofline_s128_qmarginal"]  # a longer shared state amortizes more
    assert m["roofline_s128_qtflops_gain"] > 1  # more tokens per weight read: achieved FLOP/s rises with Q
    assert m["roofline_s128_qcost_ratio"] < 1
    assert "roofline_accuracy" not in m  # answers are never graded on this axis
    assert rec.grid == GRID


def test_q_axis_counts_failures_as_invalid(tmp_path, monkeypatch):
    class Failing(SharedStateProfiled):
        @override
        def decide(self, state, questions):
            if len(questions) == 8:
                raise RuntimeError("out of memory")
            return super().decide(state, questions)

    m = _run(tmp_path, monkeypatch, Failing(), GRID).results["roofline"]
    assert m["roofline_s32_q8_invalid"] == 16
    assert m["roofline_invalid"] == 32


def test_q_axis_needs_the_length_axis(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="roofline_state_tokens too"):
        _run(tmp_path, monkeypatch, SharedStateProfiled(), {roofline.QPR_AXIS: [1, 2]})


# --- report ---------------------------------------------------------------------------------------


def _q_rec(run_id: str, gpu: str, model: str = "bongard-mini") -> RunRecord:
    m = {"roofline_tflops": 5.0}
    for n in (256, 2048):
        for q in (1, 8, 9):
            k = f"roofline_s{n}_q{q}_"
            m |= {k + "gpu_ms_median": n / 100 + q, k + "tflops": q * 1.0, k + "latency_ms": n / 50 + q}
        m[f"roofline_s{n}_qmarginal"] = 100 / n
    return RunRecord(
        run_id,
        {"gpu": gpu},
        {"model": model, "question_mode": "shared_encoder", "bongard_branch_batch": 8},
        {"roofline": m},
        grid={roofline.LENGTH_AXIS: [256, 2048], roofline.QPR_AXIS: [1, 8, 9]},
    )


def test_q_runs_are_drawn_apart_from_length_curves():
    recs = [_q_rec("a", "NVIDIA L4"), _q_rec("b", "NVIDIA H100 80GB HBM3")]
    assert report_lengths.panels(recs) == []
    (mq,) = report_qpr.models(recs)
    assert mq.mode == "shared_encoder, batch 8"
    assert [(ln.gpu_slot, ln.length) for ln in mq.lines] == [(0, 256), (0, 2048), (2, 256), (2, 2048)]
    rel = dict(mq.lines[1].relative())
    assert rel[1] == 1.0 and rel[8] == pytest.approx((20.48 + 8) / 8 / 21.48)
    header, rows = report_qpr.table([mq])
    assert len(rows) == 4 and len(rows[0]) == len(header)
    assert "<svg" in report_qpr.chart(mq)


def test_report_renders_the_q_section(tmp_path):
    recs = [_q_rec("a", "NVIDIA L4")]
    assert "Questions per request" in report.to_markdown(recs, tmp_path)
    assert "Q axis, bongard-mini" in report.to_html(recs, tmp_path)
