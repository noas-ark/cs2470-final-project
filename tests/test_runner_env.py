"""The run path is orchestrator-neutral: Slurm, Modal and local runs record the right fields."""

import stat
import textwrap
from pathlib import Path

import pytest

from s1 import config, env, registry, runner
from s1.gpu_stats import GpuSampler
from s1.models.backends import gpu, vllm_shim

CFG = """
benchmarks = ["noop"]
[invariants]
gpu = "test"
[params]
model = "echo"
"""

SMI_Q = "==============NVSMI LOG==============\nDriver Version : 580.95.05\n"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in ("SLURM_JOB_ID", "MODAL_TASK_ID", "S1_GIT_COMMIT", "MODELS_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("S1_LOG_DIR", str(tmp_path / "logs"))


@pytest.fixture
def noop_bench():
    def noop(model: object, settings: registry.Settings) -> dict[str, float]:
        return {"noop_n": 1.0}

    registry.BENCHMARKS["noop"] = noop
    yield
    registry.BENCHMARKS.pop("noop", None)


def _bin(tmp_path: Path, name: str, body: str) -> Path:
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    script = d / name
    script.write_text("#!/bin/sh\n" + textwrap.dedent(body))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return d


@pytest.fixture
def fake_smi_path(tmp_path, monkeypatch):
    """PATH with a fake nvidia-smi answering the driver/power query (2 GPUs) and `-q`."""
    d = _bin(
        tmp_path,
        "nvidia-smi",
        f"""\
        case "$1" in
          -q) printf '%s' '{SMI_Q}' ;;
          --query-gpu=driver_version,power.limit) echo "580.95.05, 72.00"; echo "580.95.05, 70.00" ;;
          *) exit 1 ;;
        esac
        """,
    )
    monkeypatch.setenv("PATH", f"{d}:/usr/bin:/bin")
    return d


@pytest.fixture
def no_smi_path(tmp_path, monkeypatch):
    d = tmp_path / "empty-bin"
    d.mkdir()
    monkeypatch.setenv("PATH", str(d))
    return d


def _run(tmp_path: Path):
    spec = config.load(_cfg(tmp_path))[0]
    return runner.run(
        spec,
        results_dir=tmp_path / "out",
        log=lambda _: None,
        gpu_sampler=lambda: GpuSampler(cmd=["/nonexistent"]),
    )


def _cfg(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.toml"
    p.write_text(CFG)
    return p


def test_slurm_run_keeps_todays_invariants_and_job_id(
    tmp_path, monkeypatch, echo_model, noop_bench, fake_smi_path
):
    monkeypatch.setenv("SLURM_JOB_ID", "4059")
    monkeypatch.setenv("S1_GIT_COMMIT", "abc1234")
    rec = _run(tmp_path)
    assert rec.invariants == {"gpu": "test"}
    assert rec.meta["slurm_job_id"] == "4059"
    assert env.META_MODAL_TASK_ID not in rec.meta
    assert rec.meta[env.META_ORCHESTRATOR] == "slurm"
    assert rec.meta[env.META_GPU_DRIVER] == "580.95.05"
    assert rec.meta[env.META_GPU_POWER_LIMIT_W] == 72.0  # GPU 0
    log = Path(str(rec.meta[env.META_NVIDIA_SMI_Q_LOG]))
    assert log == tmp_path / "logs" / "nvidia-smi-q-4059.txt"
    assert log.read_text() == SMI_Q


def test_modal_run_records_platform_and_task_id(tmp_path, monkeypatch, echo_model, noop_bench, fake_smi_path):
    monkeypatch.setenv("MODAL_TASK_ID", "ta-01ABC")
    rec = _run(tmp_path)
    assert rec.invariants == {"gpu": "test", env.INV_PLATFORM: "modal"}
    assert rec.meta[env.META_MODAL_TASK_ID] == "ta-01ABC"
    assert "slurm_job_id" not in rec.meta
    assert rec.meta[env.META_ORCHESTRATOR] == "modal"
    assert rec.meta[env.META_NVIDIA_SMI_Q_LOG] == str(tmp_path / "logs" / "nvidia-smi-q-ta-01ABC.txt")


def test_local_run_has_no_platform_or_job_ids(tmp_path, echo_model, noop_bench, no_smi_path):
    rec = _run(tmp_path)
    assert rec.invariants == {"gpu": "test"}
    assert rec.meta[env.META_ORCHESTRATOR] == "local"
    assert "slurm_job_id" not in rec.meta and env.META_MODAL_TASK_ID not in rec.meta


def test_missing_nvidia_smi_omits_hardware_fields(tmp_path, monkeypatch, echo_model, noop_bench, no_smi_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    rec = _run(tmp_path)
    for key in (env.META_GPU_DRIVER, env.META_GPU_POWER_LIMIT_W, env.META_NVIDIA_SMI_Q_LOG):
        assert key not in rec.meta
    assert not (tmp_path / "logs").exists()


def test_na_fields_are_omitted(tmp_path, monkeypatch):
    d = _bin(
        tmp_path,
        "nvidia-smi",
        """\
        [ "$1" = -q ] && exit 1
        echo "[N/A], [N/A]"
        """,
    )
    monkeypatch.setenv("PATH", f"{d}:/usr/bin:/bin")
    meta = runner.hardware_meta()
    for key in (env.META_GPU_DRIVER, env.META_GPU_POWER_LIMIT_W, env.META_NVIDIA_SMI_Q_LOG):
        assert key not in meta


def test_cpu_model_from_sysctl_when_no_proc_cpuinfo(tmp_path, monkeypatch):
    if Path("/proc/cpuinfo").exists():
        pytest.skip("Linux host reads /proc/cpuinfo first")
    d = _bin(tmp_path, "sysctl", 'echo "Apple M3 Pro"\n')
    monkeypatch.setenv("PATH", str(d))
    assert runner.cpu_model() == "Apple M3 Pro"
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    assert runner.cpu_model() is None


def test_git_commit_env_lands_in_meta(tmp_path, monkeypatch, echo_model, noop_bench, no_smi_path):
    monkeypatch.setenv("S1_GIT_COMMIT", "deadbee-dirty")
    assert _run(tmp_path).meta["git_commit"] == "deadbee-dirty"


def test_vllm_shim_model_and_log_dirs_follow_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", "/models")
    assert vllm_shim.model_dir({}) == Path("/models/openjev")
    assert vllm_shim.model_dir({"openjev_dir": "/x/oj"}) == Path("/x/oj")
    assert vllm_shim.log_dir() == tmp_path / "logs"
    monkeypatch.setenv("S1_LOG_DIR", "/runs/logs")
    assert vllm_shim.log_dir() == Path("/runs/logs")  # resolved when used, not at import


def test_vllm_shim_log_names_use_job_id(tmp_path, monkeypatch):
    (tmp_path / "helper").mkdir()
    (tmp_path / "helper" / "shim.py").write_text("")
    monkeypatch.setenv("MODAL_TASK_ID", "ta-9")
    logs: list[Path] = []

    class Proc:
        pid = 0

        def poll(self):
            return None

    def fake_spawn(cmd, log, env=None):
        logs.append(log)
        return Proc()

    monkeypatch.setattr(vllm_shim, "_spawn", fake_spawn)
    monkeypatch.setattr(vllm_shim, "_ready", lambda url: True)
    vllm_shim.build({"openjev_dir": str(tmp_path)})
    assert [p.parent for p in logs] == [tmp_path / "logs"] * 2
    assert all(p.name.split("-", 1)[1].startswith("ta-9-") for p in logs)


def test_gpu_models_dir_delegates_to_env(monkeypatch):
    monkeypatch.setenv("MODELS_DIR", "/models")
    assert gpu.models_dir() == Path("/models")


GVISOR_CPUINFO = """processor\t: 0
vendor_id\t: GenuineIntel
cpu family\t: 6
model\t\t: 143
model name\t: unknown
stepping\t: 8

processor\t: 1
vendor_id\t: GenuineIntel
model name\t: unknown
"""


def test_cpu_from_cpuinfo_real_name_and_sandbox_fallback():
    real = "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD EPYC 7R13 Processor\n"
    assert runner.cpu_from_cpuinfo(real) == "AMD EPYC 7R13 Processor"
    assert runner.cpu_from_cpuinfo(GVISOR_CPUINFO) == "GenuineIntel family 6 model 143 stepping 8"
    # what Modal's gVisor really reported on 2026-11-08: the stepping is "unknown" too
    seen = GVISOR_CPUINFO.replace("model\t\t: 143", "model\t\t: 85").replace(
        "stepping\t: 8", "stepping\t: unknown"
    )
    assert runner.cpu_from_cpuinfo(seen) == "GenuineIntel family 6 model 85"
    assert runner.cpu_from_cpuinfo("model name\t: unknown\n") is None
    assert runner.cpu_from_cpuinfo("") is None


def test_hardware_meta_records_cpu_count(monkeypatch):
    monkeypatch.setattr(runner, "_run", lambda cmd: None)
    meta = runner.hardware_meta()
    count = meta[env.META_CPU_COUNT]
    assert isinstance(count, int)
    assert count >= 1


def test_cpus_requested_from_slurm_or_modal_launcher(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("S1_CPUS_REQUESTED", raising=False)
    assert env.cpus_requested() is None
    monkeypatch.setenv("S1_CPUS_REQUESTED", "8")
    assert env.cpus_requested() == 8
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    assert env.cpus_requested() == 32
