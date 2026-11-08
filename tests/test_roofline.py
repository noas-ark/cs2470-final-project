"""The roofline benchmark (subset, metrics, end to end with a fake profiled model) and the elephant backend's
worker protocol (with a fake worker). No torch."""

import json
import math
import sys
import textwrap
from typing import override

import pytest

from s1 import flops, profiling, registry, runner
from s1.benchmarks import jeval, roofline
from s1.config import RunSpec
from s1.models.backends import elephant, kev
from s1.models.catalog import CATALOG
from s1.record import Scalar
from tests.conftest import EchoModel
from tests.test_flops import OLMOE, QWEN3_8B


def row(i, qtype="choice", n_options=3, state_words=5):
    if qtype == "choice":
        q = {"type": "choice", "instructions": "Pick", "criteria": {f"o{j}": "x" for j in range(n_options)}}
        label = "o0"
    else:
        q = {"type": qtype, "instructions": "True?"}
        label = 1
    return jeval.Row(f"r{i}", "src", qtype, " ".join(["w"] * state_words), {"q": q}, label)


def test_subset_keeps_short_choice_and_noul_with_few_options():
    rows = [
        row(0),
        row(1, "noul"),
        row(2, n_options=17),  # too many options
        row(3, "score"),  # not a classification question
        row(4, state_words=2000),  # too long
    ]
    settings: dict[str, Scalar] = {"roofline_max_tokens": 1024}
    got = roofline.subset(rows, roofline.WordCounter(), settings)
    assert sorted(r.id for r in got) == ["r0", "r1"]


def test_subset_is_seeded_and_model_independent():
    rows = [row(i) for i in range(50)]
    settings: dict[str, Scalar] = {"roofline_limit": 10, "roofline_seed": 3}
    a = roofline.subset(rows, roofline.WordCounter(), settings)
    b = roofline.subset(rows, roofline.WordCounter(), settings)
    assert [r.id for r in a] == [r.id for r in b]
    assert len(a) == 10


def question(tokens, ms, correct=True, counted=math.nan):
    a = flops.arch(QWEN3_8B)
    cost = flops.cost(a, flops.Forward(rows=1, tokens=tokens, ms=ms))
    return roofline.Question(cost, 1, jeval.Outcome("choice", True, correct, 0.9, 0.02, ms * 1.25), counted)


def test_metrics_headline_rates_bins_and_ceilings():
    qs = [question(100, 10.0), question(300, 20.0, correct=False), question(900, 50.0)]
    qs[0] = roofline.Question(qs[0].cost, 1, qs[0].outcome, counted_flops=qs[0].cost.flops * 0.8)
    a = flops.arch(QWEN3_8B)
    ceilings = {"peak_bf16_tflops": 100.0, "peak_bw_gbs": 250.0}
    m = roofline.metrics(qs, a, ceilings, "NVIDIA L4", wall_s=0.1)

    assert next(iter(m)) == "roofline_tflops"  # headline first
    total_flops = sum(q.cost.flops for q in qs)
    assert m["roofline_tflops"] == pytest.approx(total_flops / (80.0 * 1e9))
    assert m["roofline_intensity"] == pytest.approx(total_flops / sum(q.cost.bytes for q in qs))
    assert m["roofline_accuracy"] == pytest.approx(2 / 3)
    assert m["roofline_n"] == 3
    assert m["roofline_gpu_share"] == pytest.approx(0.8)
    assert not any("score" in k for k in m)
    assert m["roofline_t128_n"] == 1
    assert m["roofline_t512_n"] == 1
    assert m["roofline_t1024_n"] == 1
    assert "roofline_t256_n" not in m
    assert m["roofline_counted_ratio"] == pytest.approx(0.8)
    assert m["roofline_ridge"] == pytest.approx(400.0)
    assert m["roofline_attainable_tflops"] == pytest.approx(min(100.0, 250.0 * m["roofline_intensity"] / 1e3))
    assert m["roofline_spec_bf16_tflops"] == 121.0
    assert m["roofline_params_b"] == pytest.approx(8.19, abs=0.01)


def test_metrics_without_ceilings_or_known_gpu():
    m = roofline.metrics([question(100, 10.0)], flops.arch(QWEN3_8B), {}, "NVIDIA A10G", wall_s=0.1)
    assert "roofline_ridge" not in m
    assert "roofline_spec_bf16_tflops" not in m


def test_forward_conversion_validates_types():
    f = roofline.forward({"rows": 2, "tokens": 7, "past": 3, "kept": 1, "cached": True, "ms": 1.5})
    assert f == flops.Forward(2, 7, 3, 1, True, 1.5)
    with pytest.raises(TypeError):
        roofline.forward({"rows": 2.5, "tokens": 7, "past": 0, "kept": 1, "cached": False, "ms": 1.0})


class FakeProfiled(EchoModel):
    """EchoModel answers; every decision 'runs' one forward over its state words."""

    def __init__(self):
        self.pending = []

    @override
    def decide(self, state, questions):
        self.pending.append(
            {"rows": 1, "tokens": len(state.split()), "past": 0, "kept": 1, "cached": False, "ms": 2.0}
        )
        return super().decide(state, questions)

    def profile_info(self):
        return {
            "config": OLMOE,
            "head_bytes": 2,
            "ceilings": {"peak_bf16_tflops": 90.0, "peak_bw_gbs": 280.0},
        }

    def take_forwards(self):
        out, self.pending = self.pending, []
        return out

    def count_flops(self, state, questions):
        self.decide(state, questions)
        return 1e9


def test_benchmark_end_to_end_through_runner(tmp_path, monkeypatch):
    rows = [row(i, "noul" if i % 2 else "choice", state_words=10 + i) for i in range(30)]
    monkeypatch.setattr(jeval, "load_rows", lambda: rows)
    registry.MODELS["fake-profiled"] = lambda settings: FakeProfiled()
    try:
        spec = RunSpec(
            benchmarks=["roofline"],
            invariants={"gpu": "NVIDIA L4"},
            params={"model": "fake-profiled", "roofline_limit": 20, "roofline_counted": 2},
        )
        rec = runner.run(spec, results_dir=tmp_path, log=lambda _: None, gpu_sampler=_no_gpu)
    finally:
        registry.MODELS.pop("fake-profiled")
    m = rec.results["roofline"]
    assert m["roofline_n"] == 20
    assert m["roofline_invalid"] == 0
    assert m["roofline_forwards_per_question"] == 1
    assert m["roofline_peak_bf16_tflops"] == 90.0
    assert m["roofline_intensity"] < 200  # a MoE at ~20 tokens: far left of the ridge
    assert 0 < m["roofline_counted_ratio"] < 1


class PaddedProfiled(FakeProfiled):
    """Each decision runs one pass padded to twice its tokens, like a CUDA-graph bucket."""

    def __init__(self):
        super().__init__()
        self.warmed = 0

    @override
    def decide(self, state, questions):
        out = super().decide(state, questions)
        f = self.pending[-1]
        f["real_tokens"], f["tokens"] = f["tokens"], 2 * f["tokens"]
        return out

    def after_warmup(self):
        self.warmed += 1


def test_padding_and_after_warmup(monkeypatch):
    monkeypatch.setattr(jeval, "load_rows", lambda: [row(i, "choice", state_words=10 + i) for i in range(12)])
    model = PaddedProfiled()
    m = roofline.run(model, {"roofline_limit": 10, "roofline_warmup": 3, "roofline_counted": 0})
    assert model.warmed == 1
    assert m["roofline_padding"] == pytest.approx(0.5)


def test_unpadded_forwards_have_no_padding(monkeypatch):
    monkeypatch.setattr(jeval, "load_rows", lambda: [row(i, "choice", state_words=10 + i) for i in range(12)])
    m = roofline.run(FakeProfiled(), {"roofline_limit": 10, "roofline_counted": 0})
    assert m["roofline_padding"] == 0


def test_kev_graph_passes_charge_the_padded_bucket():
    """kev.cuda_graphs uploads: a state pass row is [ids | positions | length], a row pass row
    [ids | positions | row length | state length | bank entry]; padding rows carry length 0."""
    sb = 16
    states = [[0] * (2 * sb) + [5], [0] * (2 * sb) + [9], [0] * (2 * sb) + [0]]
    assert kev.graph_pass(("states", 3, sb), states) == {
        "rows": 3,
        "tokens": sb,
        "past": 0,
        "kept": 0,
        "cached": True,
        "real_tokens": 14,
    }
    lb = 32
    rows = [[0] * (2 * lb) + [7, 100, 0], [0] * (2 * lb) + [0, 0, 0]]
    f = kev.graph_pass(("rows", 2, lb, 128), rows)
    assert (f["rows"], f["tokens"], f["past"], f["kept"], f["real_tokens"]) == (2, lb, 128, 0, 7)
    with pytest.raises(ValueError, match="unknown"):
        kev.graph_pass(("other", 1), [])


def _no_gpu():
    from s1.gpu_stats import GpuSampler  # noqa: PLC0415

    return GpuSampler(cmd=["false"])


def test_benchmark_rejects_unprofiled_models():
    with pytest.raises(TypeError, match="profiled model"):
        roofline.run(EchoModel(), {})


# --- elephant: the harness side of the worker protocol, against a fake worker ---------------------

FAKE_WORKER = textwrap.dedent(
    """\
    import json, sys
    print(json.dumps({"ready": True, "config": {"model_type": "olmoe"}, "head_bytes": 4,
                      "ceilings": {"peak_bw_gbs": 1.0}, "env": {"torch": "2.8.0", "fork_mode": "batched"}}),
          flush=True)
    fwd = {"rows": 1, "tokens": 5, "past": 0, "kept": 1, "cached": True, "ms": 1.0}
    for line in sys.stdin:
        req = json.loads(line)
        if req["op"] == "quit":
            break
        if "boom" in req["state"]:
            print(json.dumps({"error": "ValueError: boom", "forwards": [fwd]}), flush=True)
            continue
        ((key, q),) = req["questions"].items()
        ans = {key: {"type": "noul", "noul": 0.25}}
        print(json.dumps({"answers": ans, "flops": 7.0 if req["count"] else None, "forwards": [fwd, fwd]}),
              flush=True)
    """
)


@pytest.fixture
def fake_elephant(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    (tmp_path / "elephant-1").mkdir()
    spec = next(s for s in CATALOG if s.name == "elephant-1")
    settings: dict[str, Scalar] = {
        "elephant_python": sys.executable,
        "elephant_worker": str(worker),
        "model_dir": str(tmp_path / "elephant-1"),
        "gpu": "NVIDIA L4",
        "require_gpu": "L4",
    }
    model = elephant.build(spec, settings)
    yield model
    model.close()
    assert model.proc.poll() is not None


def test_elephant_protocol(fake_elephant):
    m = fake_elephant
    assert m.profile_info() == {
        "config": {"model_type": "olmoe"},
        "head_bytes": 4,
        "ceilings": {"peak_bw_gbs": 1.0},
    }
    assert m.run_info() == {"env_torch": "2.8.0", "env_fork_mode": "batched"}
    q = {"q": {"type": "noul", "instructions": "?"}}
    assert m.decide("state", q) == {"q": {"type": "noul", "noul": 0.25}}
    assert len(m.take_forwards()) == 2
    assert m.take_forwards() == []
    assert m.count_flops("state", q) == 7.0
    with pytest.raises(RuntimeError, match="boom"):
        m.decide("boom", q)
    assert len(m.take_forwards()) == 3  # the failed request's forwards still count


def test_elephant_refuses_the_wrong_gpu(tmp_path):
    spec = next(s for s in CATALOG if s.name == "elephant-1")
    (tmp_path / "elephant-1").mkdir()
    settings: dict[str, Scalar] = {
        "elephant_python": sys.executable,
        "model_dir": str(tmp_path / "elephant-1"),
        "gpu": "NVIDIA A10G",
        "require_gpu": "L4",
    }
    with pytest.raises(RuntimeError, match="requires"):
        elephant.build(spec, settings)


def test_runner_records_model_run_info(tmp_path, echo_model):
    class WithInfo(EchoModel):
        def run_info(self):
            return {"env_torch": "2.13.0", "gpu": "should not override"}

    registry.MODELS["echo"] = lambda settings: WithInfo()
    spec = RunSpec(benchmarks=[], invariants={"gpu": "NVIDIA L4"}, params={"model": "echo"})
    rec = runner.run(spec, results_dir=tmp_path, log=lambda _: None, gpu_sampler=_no_gpu)
    assert rec.invariants["env_torch"] == "2.13.0"
    assert rec.invariants["gpu"] == "NVIDIA L4"  # the config's value wins
    assert json.loads((tmp_path / f"{rec.run_id}.json").read_text())


def test_kernel_shares_rank_and_attention_pick():

    events = [
        ("ampere_bf16_gemm", 60.0),
        ("flash_fwd_kernel", 30.0),
        ("ampere_bf16_gemm", 30.0),
        ("copy", 0.0),
    ]
    shares = profiling.kernel_shares(events, top=2)
    assert [n for n, _ in shares] == ["ampere_bf16_gemm", "flash_fwd_kernel"]
    assert shares[0][1] == pytest.approx(0.75)
    assert profiling.attention_kernel(shares) == "flash_fwd_kernel"
    assert profiling.attention_kernel([("gemm", 1.0)]) is None
    assert profiling.kernel_shares([]) == []


def _roofline_rec(tmp_path, monkeypatch, **params):
    rows = [row(i, state_words=10 + i) for i in range(12)]
    monkeypatch.setattr(jeval, "load_rows", lambda: rows)
    registry.MODELS["fake-profiled"] = lambda settings: FakeProfiled()
    try:
        spec = RunSpec(
            benchmarks=["roofline"],
            invariants={"gpu": "NVIDIA L4"},
            params={"model": "fake-profiled", "roofline_limit": 10, "roofline_counted": 0, **params},
        )
        return runner.run(spec, results_dir=tmp_path, log=lambda _: None, gpu_sampler=_no_gpu)
    finally:
        registry.MODELS.pop("fake-profiled")


def test_kernel_names_land_in_meta_after_timing(tmp_path, monkeypatch):

    calls = []

    def fake_profile(fn):
        calls.append(fn())  # the profiled question really runs
        return [("sm90_xmma_gemm_bf16", 80.0), ("fmha_cutlassF_bf16", 20.0)]

    monkeypatch.setattr(profiling, "profile_kernels", fake_profile)
    rec = _roofline_rec(tmp_path, monkeypatch)
    assert calls
    assert rec.meta["roofline_kernels_top"] == "80.0% sm90_xmma_gemm_bf16; 20.0% fmha_cutlassF_bf16"
    assert rec.meta["roofline_attention_kernel"] == "fmha_cutlassF_bf16"
    assert rec.meta["roofline_kernels_seen"] == 2
    assert rec.results["roofline"]["roofline_n"] == 10  # the traced question is not a measured one


def test_kernel_profile_failure_is_noted_not_fatal(tmp_path, monkeypatch):

    def broken(fn):
        raise RuntimeError("CUPTI unavailable")

    monkeypatch.setattr(profiling, "profile_kernels", broken)
    rec = _roofline_rec(tmp_path, monkeypatch)
    assert rec.meta["roofline_kernels_error"] == "RuntimeError: CUPTI unavailable"
    assert "roofline" in rec.results


def test_kernel_profile_can_be_turned_off(tmp_path, monkeypatch):

    monkeypatch.setattr(profiling, "profile_kernels", lambda fn: pytest.fail("should not trace"))
    rec = _roofline_rec(tmp_path, monkeypatch, roofline_kernel_profile=0)
    assert not any(k.startswith("roofline_kernels") for k in rec.meta)


def test_note_outside_a_run_is_a_no_op():
    registry.note("x", "y")  # no collector: nothing happens, nothing raises
    with registry.collecting_notes() as notes:
        registry.note("roofline_attention_kernel", "flash")
    assert notes == {"roofline_attention_kernel": "flash"}


def test_attention_kernel_found_below_the_top_ten(tmp_path, monkeypatch):
    """OLMoE on the L4: twelve expert GEMMs outrank attention, which must still be found."""
    gemms = [(f"ampere_bf16_gemm_{i}", 50.0) for i in range(12)]
    monkeypatch.setattr(profiling, "profile_kernels", lambda fn: [*gemms, ("flash_fwd_kernel", 4.0)])
    rec = _roofline_rec(tmp_path, monkeypatch)
    assert "flash" not in str(rec.meta["roofline_kernels_top"])
    assert rec.meta["roofline_attention_kernel"] == "flash_fwd_kernel"
    assert rec.meta["roofline_attention_share"] == round(4.0 / 604.0, 4)
    assert rec.meta["roofline_kernels_seen"] == 13
