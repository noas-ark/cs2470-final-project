"""s1.launch.modal_app without the Modal client: mapping, dry-run plan, write-back, download check."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from s1 import registry
from s1.launch import Resources
from s1.launch import modal_app as ma
from s1.record import RunRecord, load_all
from tests.conftest import EchoModel as TestsEchoModel

REPO = Path(__file__).resolve().parents[1]
CPU_CFG = """
benchmarks = ["modal_count"]
[invariants]
gpu = "auto"
[params]
model = "echo"
"""


def modal_resources(**kw):
    return Resources(orchestrator="modal", **kw)


@pytest.fixture
def no_modal(monkeypatch):
    """Fail the test if anything tries to reach the Modal client."""

    def boom():
        raise AssertionError("contacted Modal")

    monkeypatch.setattr(ma, "_modal", boom)


def test_import_does_not_import_modal():
    code = "import sys, s1, s1.launch.modal_app; assert 'modal' not in sys.modules, 'modal imported'"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


@pytest.mark.parametrize(
    ("gpu", "gpus", "want"),
    [
        (None, 1, None),
        (None, 4, None),
        ("H100", 1, "H100!"),
        ("h100", 2, "H100!:2"),
        ("A100-80GB", 1, "A100-80GB"),
        ("L4", 1, "L4"),
        ("L4", 4, "L4:4"),
        ("A10G", 1, "A10"),
        ("H200", 1, "H200"),
        ("B200", 1, "B200"),
    ],
)
def test_modal_gpu(gpu, gpus, want):
    assert ma.modal_gpu(gpu, gpus) == want


def test_modal_gpu_rejects_unknown_and_zero():
    with pytest.raises(ValueError, match="unknown GPU"):
        ma.modal_gpu("A100")  # ambiguous: no fallback, name the memory size
    with pytest.raises(ValueError, match="gpus"):
        ma.modal_gpu("L4", 0)


def test_no_fallback_lists():
    assert all(":" not in k.modal and "," not in k.modal for k in ma.GPUS.values())


@pytest.mark.parametrize(
    ("mem", "mib"),
    [
        ("28G", 28 * 1024),
        ("160G", 160 * 1024),
        ("512M", 512),
        ("16384", 16384),
        ("1T", 1024**2),
        ("4gb", 4096),
    ],
)
def test_mem_mib(mem, mib):
    assert ma.mem_mib(mem) == mib


@pytest.mark.parametrize("bad", ["", "lots", "28 GiB", "-1G"])
def test_mem_mib_rejects(bad):
    with pytest.raises(ValueError, match="bad mem"):
        ma.mem_mib(bad)


@pytest.mark.parametrize(
    ("time", "seconds"),
    [
        ("02:00:00", 7200),
        ("00:10:00", 600),
        ("30", 1800),
        ("05:30", 330),
        ("1-00", 86400),
        ("0-06:30", 6 * 3600 + 1800),
        ("0-01:00:05", 3605),
    ],
)
def test_timeout_s(time, seconds):
    assert ma.timeout_s(time) == seconds


@pytest.mark.parametrize("bad", ["", "1:2:3:4", "1h", "00:00:00", "25:00:00", "2-00:00:00"])
def test_timeout_s_rejects(bad):
    with pytest.raises(ValueError, match="time"):
        ma.timeout_s(bad)


def test_cpu_plan(tmp_path):
    cfg = tmp_path / "echo.toml"
    cfg.write_text(CPU_CFG)
    res = modal_resources(cpus=2, mem="4G", time="00:10:00", extras={"echo": True})
    p = ma.plan(cfg, res, ["run", str(cfg)], git_commit="abc1234-dirty")
    assert p.gpu is None and not p.openjev and p.echo
    assert (p.cpu, p.memory_mib, p.timeout_s) == (2.0, 4096, 600)
    assert p.cfg_remote == "configs/submitted/echo.toml" and p.cfg_text == CPU_CFG
    assert p.s1_args == ("run", "configs/submitted/echo.toml", "--results-dir", "{run_dir}/results")
    assert p.env == {"S1_GIT_COMMIT": "abc1234-dirty", "S1_CPUS_REQUESTED": "2"}


def test_gpu_plan_sets_require_gpu(tmp_path):
    cfg = tmp_path / "x.toml"
    cfg.write_text(CPU_CFG)
    p = ma.plan(cfg, modal_resources(gpu="H100", mem="64G"), ["run", str(cfg)], git_commit="x")
    assert p.gpu == "H100!" and p.openjev and not p.echo
    assert p.s1_args[-2:] == ("--set", "require_gpu=H100")
    p = ma.plan(
        cfg, modal_resources(gpu="H100", extras={"openjev": False}), ["run", str(cfg)], git_commit="x"
    )
    assert not p.openjev


def test_config_require_gpu_is_kept_or_rejected():
    cfg = REPO / "configs" / "roofline_l4.toml"  # sets require_gpu = "L4"
    p = ma.plan(cfg, modal_resources(gpu="L4"), ["run", str(cfg)], git_commit="x")
    assert p.cfg_remote == "configs/roofline_l4.toml"
    assert "--set" not in p.s1_args
    with pytest.raises(ValueError, match="contradicts"):
        ma.plan(cfg, modal_resources(gpu="H100"), ["run", str(cfg)], git_commit="x")
    with pytest.raises(ValueError, match="no GPU requested"):
        ma.plan(cfg, modal_resources(), ["run", str(cfg)], git_commit="x")


def test_slurm_flags_are_rejected_on_modal(tmp_path):
    cfg = tmp_path / "x.toml"
    cfg.write_text(CPU_CFG)
    with pytest.raises(ValueError, match="partition"):
        ma.plan(cfg, modal_resources(extras={"partition": "gpu-spec"}), ["run", str(cfg)], git_commit="x")


def test_ncu_plan_gets_results_dir_but_no_set(tmp_path):
    cfg = tmp_path / "x.toml"
    cfg.write_text(CPU_CFG)
    p = ma.plan(cfg, modal_resources(gpu="L4"), ["ncu", str(cfg), "--questions", "4"], git_commit="x")
    assert p.s1_args == (
        "ncu",
        "configs/submitted/x.toml",
        "--questions",
        "4",
        "--results-dir",
        "{run_dir}/results",
    )


def test_unknown_extras_rejected(tmp_path):
    cfg = tmp_path / "x.toml"
    cfg.write_text(CPU_CFG)
    with pytest.raises(ValueError, match=r"launch\.modal"):
        ma.plan(cfg, modal_resources(extras={"partition": "gpu"}), ["run", str(cfg)])


def test_dry_run_prints_plan_without_modal(tmp_path, no_modal):
    cfg = tmp_path / "x.toml"
    cfg.write_text(CPU_CFG)
    lines: list[str] = []
    launcher = ma.ModalLauncher(results_dir=tmp_path / "results", log=lines.append)
    handle = launcher.submit(
        cfg, modal_resources(gpu="H100", mem="28G", time="02:00:00"), ["run", str(cfg)], dry_run=True
    )
    assert handle == ""
    text = "\n".join(lines)
    for want in [
        "gpu='H100!'",
        "memory=28672 MiB",
        "timeout=7200 s",
        "--extra openjev",
        "s1-models -> /models",
        "s1-hf-cache -> /hf-cache",
        "s1-runs -> /runs",
        "MODELS_DIR=/models",
        "S1_GIT_COMMIT=",
        "command: s1 run configs/submitted/x.toml --results-dir /runs/CALL_ID/results --set require_gpu=H100",
    ]:
        assert want in text, want
    assert not (tmp_path / "results").exists()


def _record(run_id: str) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        invariants={"gpu": "cpu", "platform": "modal"},
        params={"model": "echo"},
        results={"modal_count": {"p_true": 0.8}},
        meta={"orchestrator": "modal", "git_commit": "abc1234"},
    )


def test_write_back_saves_records_and_logs(tmp_path):
    rec = _record("20261108-120000-deadbeef")
    files = {"fc-1/logs/fc-1.out": b"hello\n", "fc-1/logs/vllm-ta-1-8000.log": b"vllm\n"}
    result = ma.remote_result(
        json.dumps(
            {
                "call_id": "fc-1",
                "returncode": 0,
                "records": {f"{rec.run_id}.json": json.dumps(rec.to_json())},
                "logs": ["logs/fc-1.out", "logs/vllm-ta-1-8000.log"],
            }
        )
    )
    fetched: list[str] = []

    def fetch(path):
        fetched.append(path)
        yield files[path]

    records, logs = ma.write_back(result, fetch, results_dir=tmp_path / "results", logs_dir=tmp_path / "logs")
    assert records == [tmp_path / "results" / f"{rec.run_id}.json"]
    assert load_all(tmp_path / "results") == [rec]
    assert (tmp_path / "logs" / "fc-1.out").read_bytes() == b"hello\n"
    assert (tmp_path / "logs" / "vllm-ta-1-8000.log").read_bytes() == b"vllm\n"
    assert sorted(fetched) == sorted(files)
    assert logs == [tmp_path / "logs" / "fc-1.out", tmp_path / "logs" / "vllm-ta-1-8000.log"]


def test_write_back_refuses_escaping_log_paths(tmp_path):
    result = ma.remote_result(json.dumps({"call_id": "c", "returncode": 0, "records": {}, "logs": ["../x"]}))
    with pytest.raises(ValueError, match="refusing"):
        ma.write_back(result, lambda _: iter([b""]), results_dir=tmp_path, logs_dir=tmp_path / "logs")


def test_remote_result_rejects_malformed():
    with pytest.raises(TypeError):
        ma.remote_result("[]")
    with pytest.raises(TypeError):
        ma.remote_result(json.dumps({"call_id": "c", "returncode": 0, "records": {}, "logs": [1]}))


def test_verify_sha256sums(tmp_path):
    good, readme = b"weights", b"card"
    (tmp_path / "model.safetensors").write_bytes(good)
    (tmp_path / "README.md").write_bytes(b"card, edited later")
    lines = [
        f"{hashlib.sha256(good).hexdigest()}  model.safetensors",
        f"{hashlib.sha256(readme).hexdigest()} *README.md",
        f"{hashlib.sha256(b'gone').hexdigest()}  helper/missing.py",
    ]
    (tmp_path / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    assert ma.verify_sha256sums(tmp_path) == (3, ["README.md", "helper/missing.py"])


def test_download_report_parses():
    text = json.dumps(
        {
            "repo": "o/m",
            "revision": None,
            "dest": "/models/m",
            "sha256_checked": 2,
            "sha256_failed": ["README.md"],
        }
    )
    assert ma.download_report(text)["sha256_failed"] == ["README.md"]


QUESTIONS = [
    {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}},
    {"type": "noul", "instructions": "?"},
    {"type": "score", "criteria": ["low", "mid", "high"]},
]


@pytest.mark.parametrize("q", QUESTIONS)
def test_builtin_echo_matches_tests_echo(q):
    assert ma.EchoModel().decide("s", {"k": q}) == TestsEchoModel().decide("s", {"k": q})


def test_container_entry_runs_echo_config(tmp_path, monkeypatch):
    """`python -m s1.launch.modal_app --echo run ...` as the container runs it, minus Modal."""

    def count(model, settings):
        return {"p_true": model.decide("s", {"q": {"type": "noul"}})["q"]["noul"]}

    cfg = tmp_path / "echo.toml"
    cfg.write_text(CPU_CFG)
    monkeypatch.setitem(registry.BENCHMARKS, "modal_count", count)
    try:
        with pytest.raises(SystemExit) as exit_info:
            ma._main(["--echo", "run", str(cfg), "--results-dir", str(tmp_path / "out")])
        assert exit_info.value.code in (0, None)
    finally:
        registry.MODELS.pop("echo", None)
    (rec,) = load_all(tmp_path / "out")
    assert rec.results["modal_count"]["p_true"] == 0.8
