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


@pytest.fixture
def count_bench():
    def run(model, settings):
        time.sleep(0.3)  # let the sampler read the fake nvidia-smi output
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


class _PrimedSampler(GpuSampler):
    """The fake benchmark finishes instantly, so wait for the fake nvidia-smi's first rows before returning.
    Without this the sampler can stop before the subprocess prints anything (empty metrics, flaky test)."""

    @override
    def __enter__(self) -> GpuSampler:
        super().__enter__()
        deadline = time.monotonic() + 10
        while len(self.samples) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        return self


def test_end_to_end_writes_records_and_gpu_stats(tmp_path, echo_model, count_bench, fake_nvidia_smi):
    sampler = lambda: _PrimedSampler(cmd=fake_nvidia_smi)  # noqa: E731
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
