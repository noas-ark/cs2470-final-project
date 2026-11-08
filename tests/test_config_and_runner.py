import time
from typing import override

import pytest

from s1 import config, registry, runner
from s1.gpu_stats import GpuSampler
from s1.record import load_all

CFG = """
benchmarks = ["count_answers"]
[invariants]
gpu = "test"
[params]
lr = 0.1
[sweep]
model = ["echo"]
seed = [0, 1, 2]
"""


class WaitingSampler(GpuSampler):
    """Holds the benchmark back until the fake nvidia-smi has printed its 4 valid rows.

    A fixed sleep inside the benchmark raced with process start-up: on a loaded machine the fake had not
    printed anything yet, so the record had no nvidia_smi entry.
    """

    @override
    def __enter__(self):
        super().__enter__()
        deadline = time.monotonic() + 10
        while len(self.samples) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        return self


@pytest.fixture
def count_bench():
    def run(model, settings):
        answers = model.decide("state", {"q": {"type": "noul", "instructions": "?"}})
        return {"p_true": answers["q"]["noul"], "seed": float(settings["seed"])}

    registry.BENCHMARKS["count_answers"] = run
    yield
    registry.BENCHMARKS.pop("count_answers", None)


def test_sweep_expands_cartesian_product(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(CFG)
    specs = config.load(p)
    assert [s.params for s in specs] == [{"lr": 0.1, "model": "echo", "seed": i} for i in range(3)]


def test_missing_model_is_an_error(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('benchmarks = ["jeval"]\n')
    with pytest.raises(ValueError, match="model"):
        config.load(p)


def test_end_to_end_writes_records_and_gpu_stats(tmp_path, echo_model, count_bench, fake_nvidia_smi):
    sampler = lambda: WaitingSampler(cmd=fake_nvidia_smi)  # noqa: E731
    p = tmp_path / "c.toml"
    p.write_text(CFG)
    for spec in config.load(p):
        runner.run(spec, results_dir=tmp_path / "out", log=lambda _: None, gpu_sampler=sampler)
    recs = load_all(tmp_path / "out")
    assert len(recs) == 3
    assert recs[0].invariants["gpu"] == "test"
    assert sorted(r.results["count_answers"]["seed"] for r in recs) == [0.0, 1.0, 2.0]
    assert all(r.results["count_answers"]["p_true"] == 0.8 for r in recs)
    smi = recs[0].results["count_answers/nvidia_smi"]
    assert smi["gpu_count"] == 2 and smi["gpu_mem_used_mb_max"] == 4000
    assert echo_model.closed  # runner closes models that define close()


def test_grid_is_carried_not_expanded(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(CFG + "[grid]\nstate_tokens = [32, 64]\nquestions_per_request = [1, 2, 4]\n")
    specs = config.load(p)
    assert len(specs) == 3  # only [sweep] expands
    assert all(s.grid == {"state_tokens": [32, 64], "questions_per_request": [1, 2, 4]} for s in specs)
    assert config.load(tmp_path / "c.toml")[0].params["model"] == "echo"


def test_configs_without_grid_have_empty_grid(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(CFG)
    assert all(s.grid == {} for s in config.load(p))


@pytest.mark.parametrize("grid", ["x = []", "x = 3", "x = [[1]]", "x = [1, { a = 2 }]"])
def test_bad_grid_is_an_error(tmp_path, grid):
    p = tmp_path / "c.toml"
    p.write_text(CFG + f"[grid]\n{grid}\n")
    with pytest.raises((ValueError, TypeError)):
        config.load(p)


def test_grid_settings_are_comma_joined():
    assert runner.grid_settings({"a": [1, 2], "b": ["x"]}) == {"grid_a": "1,2", "grid_b": "x"}


def test_runner_basic_has_no_schema_and_detailed_records_it(tmp_path, echo_model, count_bench):
    p = tmp_path / "c.toml"
    p.write_text(CFG)
    (spec, *_) = config.load(p)
    basic = runner.run(
        spec, results_dir=tmp_path / "b", log=lambda _: None, gpu_sampler=lambda: GpuSampler(cmd=["/x"])
    )
    assert "gpu_metrics" not in basic.invariants and "gpu_metrics_schema" not in basic.invariants

    p.write_text(CFG.replace('gpu = "test"', 'gpu = "test"\ngpu_metrics = "detailed"'))
    (spec, *_) = config.load(p)
    seen: list[bool] = []

    def factory():
        s = GpuSampler(cmd=["/x"])
        seen.append(s.detailed)
        return s

    rec = runner.run(spec, results_dir=tmp_path / "d", log=lambda _: None, gpu_sampler=factory)
    assert rec.invariants["gpu_metrics"] == "detailed" and rec.invariants["gpu_metrics_schema"] == 1
    assert seen == [True]

    p.write_text(CFG.replace('gpu = "test"', 'gpu = "test"\ngpu_metrics = "bogus"'))
    (spec, *_) = config.load(p)
    with pytest.raises(ValueError, match="gpu_metrics"):
        runner.run(spec, results_dir=tmp_path / "e", log=lambda _: None)


def test_warmup_runs_before_sampling_and_is_recorded(tmp_path, echo_model):
    events = []

    class Sampler(GpuSampler):
        @override
        def __enter__(self):
            events.append("sample")
            return self

        @override
        def __exit__(self, exc_type, exc, tb):
            return None

        @override
        def metrics(self):
            return {}

    @registry.benchmark("warm_bench")
    def bench(model, settings):
        events.append("bench")
        return {"warm_bench_x": 1.0}

    @registry.warmup("warm_bench")
    def warm(model, settings):
        events.append("warm")
        return bool(settings.get("do_warm"))

    try:
        for do_warm in (True, False):
            spec = config.RunSpec(
                invariants={"gpu": "cpu"},
                params={"model": "echo", "do_warm": do_warm},
                benchmarks=["warm_bench"],
            )
            rec = runner.run(spec, results_dir=tmp_path, log=lambda _: None, gpu_sampler=Sampler)
            assert ("warm_bench_warmup_s" in rec.meta) is do_warm
    finally:
        registry.BENCHMARKS.pop("warm_bench", None)
        registry.WARMUPS.pop("warm_bench", None)
    assert events == ["warm", "sample", "bench"] * 2
