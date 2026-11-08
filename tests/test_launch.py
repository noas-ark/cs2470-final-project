"""`[launch]` parsing, `s1 submit` flag precedence, the sbatch command, and orchestrator dispatch.

Nothing here queues a job: sbatch is either --dry-run or a monkeypatched subprocess.run, and the Modal
launcher is a fake module in sys.modules."""

from __future__ import annotations

import subprocess
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from s1 import cli, config, launch
from s1.launch import Resources, get_launcher
from s1.launch.slurm import SlurmLauncher, sbatch_command

ROOT = Path(__file__).resolve().parent.parent
ROOFLINE = "configs/roofline_l4.toml"
SMOKE = "configs/openjev_jeval_smoke.toml"

# `s1 submit ... --dry-run` output recorded before the launcher refactor (commit 0824430).
PINNED: dict[tuple[str, ...], str] = {
    (
        ROOFLINE,
        "-p",
        "gpu-spec",
        "--gpu-type",
        "l4",
        "-g",
        "1",
        "-c",
        "8",
        "--mem",
        "28G",
        "-t",
        "02:00:00",
    ): (
        "sbatch -J s1-roofline_l4 -c 8 --mem=28G -t 02:00:00 -o logs/%j.out -p gpu-spec --gres=gpu:l4:1 "
        "--wrap 'uv run --extra openjev s1 run configs/roofline_l4.toml'"
    ),
    (SMOKE, "-p", "gpu-parallel", "-g", "4", "-c", "32", "--mem", "160G", "-t", "01:00:00"): (
        "sbatch -J s1-openjev_jeval_smoke -c 32 --mem=160G -t 01:00:00 -o logs/%j.out -p gpu-parallel "
        "--gres=gpu:4 --wrap 'uv run --extra openjev s1 run configs/openjev_jeval_smoke.toml'"
    ),
    (
        ROOFLINE,
        *("-p", "gpu-spec", "--gpu-type", "l4", "-g", "1", "-c", "8", "--mem", "28G", "-t", "02:00:00"),
        *("--ncu", "4"),
    ): (
        "sbatch -J s1-roofline_l4 -c 8 --mem=28G -t 02:00:00 -o logs/%j.out -p gpu-spec --gres=gpu:l4:1 "
        "--wrap 'uv run --extra openjev s1 ncu configs/roofline_l4.toml --questions 4'"
    ),
    (SMOKE,): (
        "sbatch -J s1-openjev_jeval_smoke -c 8 --mem=32G -t 01:00:00 -o logs/%j.out --gres=gpu:1 "
        "--wrap 'uv run --extra openjev s1 run configs/openjev_jeval_smoke.toml'"
    ),
    (SMOKE, "-g", "0"): (
        "sbatch -J s1-openjev_jeval_smoke -c 8 --mem=32G -t 01:00:00 -o logs/%j.out "
        "--wrap 'uv run --extra openjev s1 run configs/openjev_jeval_smoke.toml'"
    ),
}

CFG_BODY = 'benchmarks = ["jeval"]\n[params]\nmodel = "echo"\n'


def _cfg(tmp_path: Path, launch_toml: str = "", name: str = "exp") -> Path:
    path = tmp_path / f"{name}.toml"
    path.write_text(CFG_BODY + launch_toml)
    return path


def _submit(*args: str) -> str:
    result = CliRunner().invoke(cli.app, ["submit", *args, "--dry-run"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


# --- pinned sbatch commands ------------------------------------------------------------------------


@pytest.mark.parametrize("args", list(PINNED), ids=lambda a: " ".join(a[1:]) or "defaults")
def test_dry_run_matches_pre_refactor_output(args: tuple[str, ...], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    assert _submit(*args) == PINNED[args] + "\n"


def test_launch_table_reproduces_the_flag_command(tmp_path: Path) -> None:
    flags = _submit(str(_cfg(tmp_path)), "-p", "gpu-spec", "--gpu-type", "l4", "-c", "8", "--mem", "28G")
    table = (
        '[launch]\norchestrator = "slurm"\ngpus = 1\ncpus = 8\nmem = "28G"\n'
        '[launch.slurm]\npartition = "gpu-spec"\ngpu_type = "l4"\n'
    )
    other = tmp_path / "t"
    other.mkdir()
    assert _submit(str(_cfg(other, table))) == flags.replace(str(tmp_path / "exp"), str(other / "exp"))


def test_sbatch_command_is_built_from_resources() -> None:
    res = Resources(
        orchestrator="slurm", gpus=2, cpus=4, mem="8G", time="00:10:00", extras={"partition": "gpu"}
    )
    cmd = sbatch_command(Path("configs/x.toml"), res, ["run", "configs/x.toml"])
    assert cmd == [
        *("sbatch", "-J", "s1-x", "-c", "4", "--mem=8G", "-t", "00:10:00", "-o", "logs/%j.out"),
        *("-p", "gpu", "--gres=gpu:2", "--wrap", "uv run --extra openjev s1 run configs/x.toml"),
    ]


def test_sbatch_rejects_unknown_slurm_extra() -> None:
    res = Resources(orchestrator="slurm", extras={"qos": "high"})
    with pytest.raises(ValueError, match="`qos`"):
        sbatch_command(Path("x.toml"), res, ["run", "x.toml"])


def test_slurm_submit_runs_sbatch_and_returns_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(cmd, 0, stdout="Submitted batch job 4242\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = Resources(orchestrator="slurm")
    job = SlurmLauncher().submit(Path("x.toml"), res, ["run", "x.toml"])
    assert job == "4242"
    assert calls == [sbatch_command(Path("x.toml"), res, ["run", "x.toml"])]
    assert (tmp_path / "logs").is_dir()
    assert "Submitted batch job 4242" in capsys.readouterr().out


def test_dry_run_never_calls_sbatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("sbatch called on a dry run")

    monkeypatch.setattr(subprocess, "run", boom)
    assert (
        SlurmLauncher().submit(Path("x.toml"), Resources(orchestrator="slurm"), ["run"], dry_run=True) == ""
    )


# --- [launch] parsing -------------------------------------------------------------------------------


def test_no_launch_table_is_none(tmp_path: Path) -> None:
    assert config.load_launch(_cfg(tmp_path)) is None


def test_launch_table_parses(tmp_path: Path) -> None:
    path = _cfg(
        tmp_path,
        '[launch]\norchestrator = "modal"\ngpu = "H100"\ngpus = 1\ncpus = 16\n'
        'mem = "64G"\ntime = "00:30:00"\n'
        '[launch.slurm]\npartition = "gpu-spec"\n[launch.modal]\nregion = "us-east"\n',
    )
    assert config.load_launch(path) == Resources(
        orchestrator="modal",
        gpu="H100",
        gpus=1,
        cpus=16,
        mem="64G",
        time="00:30:00",
        extras={"region": "us-east"},
    )
    # --on picks the other orchestrator and its extras; the shared fields stay
    slurm = config.load_launch(path, "slurm")
    assert slurm is not None
    assert (slurm.orchestrator, slurm.extras, slurm.cpus) == ("slurm", {"partition": "gpu-spec"}, 16)


def test_launch_defaults_match_resources(tmp_path: Path) -> None:
    assert config.load_launch(_cfg(tmp_path, '[launch]\norchestrator = "slurm"\n')) == Resources("slurm")


def test_launch_never_reaches_run_spec(tmp_path: Path) -> None:
    path = _cfg(tmp_path, '[launch]\norchestrator = "modal"\ngpu = "H100"\n[launch.modal]\nx = 1\n')
    (spec,) = config.load(path)
    assert spec.params == {"model": "echo"}
    assert spec.invariants == {}


@pytest.mark.parametrize(
    ("body", "error", "match"),
    [
        ("[launch]\ngpus = 1\n", ValueError, "needs `orchestrator`"),
        ('[launch]\norchestrator = "aws"\n', ValueError, "orchestrator must be one of slurm, modal"),
        ('[launch]\norchestrator = "slurm"\ngpus = "4"\n', TypeError, "gpus must be a non-negative integer"),
        ('[launch]\norchestrator = "slurm"\ngpus = true\n', TypeError, "gpus must be"),
        ('[launch]\norchestrator = "slurm"\ncpus = -1\n', TypeError, "cpus must be"),
        ('[launch]\norchestrator = "slurm"\nmem = 28\n', TypeError, "mem must be a non-empty string"),
        ('[launch]\norchestrator = "slurm"\ntime = 3600\n', TypeError, "time must be"),
        ('[launch]\norchestrator = "slurm"\ngpu = 1\n', TypeError, "gpu must be"),
        ('[launch]\norchestrator = "slurm"\npartition = "gpu"\n', ValueError, "unknown key `partition`"),
        ('[launch]\norchestrator = "slurm"\n[launch.aws]\nx = 1\n', ValueError, "unknown key `aws`"),
        ('[launch]\norchestrator = "slurm"\n[launch.slurm]\nx = [1]\n', TypeError, "x must be a scalar"),
    ],
)
def test_launch_validation(tmp_path: Path, body: str, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        config.load_launch(_cfg(tmp_path, body))


def test_launch_must_be_a_table(tmp_path: Path) -> None:
    path = tmp_path / "x.toml"
    path.write_text("launch = 3\n" + CFG_BODY)
    with pytest.raises(TypeError, match=r"\[launch\] must be a table"):
        config.load_launch(path)


def test_bad_on_flag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--on: orchestrator must be"):
        cli.launch_resources(_cfg(tmp_path), on="aws")
    result = CliRunner().invoke(cli.app, ["submit", str(_cfg(tmp_path)), "--on", "aws", "--dry-run"])
    assert result.exit_code == 2
    assert "sbatch" not in result.output


# --- precedence -------------------------------------------------------------------------------------


def test_no_launch_no_flags_is_todays_slurm_default(tmp_path: Path) -> None:
    assert cli.launch_resources(_cfg(tmp_path)) == Resources(orchestrator="slurm")


def test_flags_override_launch_field_by_field(tmp_path: Path) -> None:
    path = _cfg(
        tmp_path,
        '[launch]\norchestrator = "slurm"\ngpus = 4\ncpus = 32\nmem = "160G"\ntime = "06:00:00"\n'
        '[launch.slurm]\npartition = "gpu-parallel"\ngpu_type = "l4"\n',
    )
    res = cli.launch_resources(path, cpus=8, partition="gpu-spec")
    assert res == Resources(
        orchestrator="slurm",
        gpus=4,
        cpus=8,
        mem="160G",
        time="06:00:00",
        extras={"partition": "gpu-spec", "gpu_type": "l4"},
    )
    assert cli.launch_resources(path, gpus=0, mem="1G", time="00:01:00", gpu_type="a10g") == Resources(
        orchestrator="slurm",
        gpus=0,
        cpus=32,
        mem="1G",
        time="00:01:00",
        extras={"partition": "gpu-parallel", "gpu_type": "a10g"},
    )


def test_on_overrides_orchestrator(tmp_path: Path) -> None:
    path = _cfg(tmp_path, '[launch]\norchestrator = "slurm"\ngpu = "L4"\n[launch.slurm]\npartition = "gpu"\n')
    res = cli.launch_resources(path, on="modal")
    assert res == Resources(orchestrator="modal", gpu="L4")  # [launch.slurm] extras do not follow


def test_on_without_launch_table(tmp_path: Path) -> None:
    res = cli.launch_resources(_cfg(tmp_path), on="modal", gpu="H100")
    assert res == Resources(orchestrator="modal", gpu="H100")


# --- dispatch -----------------------------------------------------------------------------------------


class FakeModalLauncher:
    calls: ClassVar[list[tuple[Path, Resources, list[str], bool]]] = []

    def submit(
        self, cfg: Path, resources: Resources, s1_args: Sequence[str], *, dry_run: bool = False
    ) -> str:
        FakeModalLauncher.calls.append((cfg, resources, list(s1_args), dry_run))
        return "fc-123"


@pytest.fixture
def fake_modal(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, Resources, list[str], bool]]:
    module = types.ModuleType(launch.MODAL_MODULE)
    module.__dict__["ModalLauncher"] = FakeModalLauncher
    monkeypatch.setitem(sys.modules, launch.MODAL_MODULE, module)
    FakeModalLauncher.calls = []
    return FakeModalLauncher.calls


def test_get_launcher_slurm() -> None:
    assert isinstance(get_launcher("slurm"), SlurmLauncher)


def test_get_launcher_modal_through_protocol(fake_modal: list[object]) -> None:
    assert isinstance(get_launcher("modal"), FakeModalLauncher)


def test_modal_missing_says_not_built(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, launch.MODAL_MODULE, None)  # import raises ModuleNotFoundError
    with pytest.raises(ModuleNotFoundError, match=r"Modal launcher not built yet \(workstream C\)"):
        get_launcher("modal")


def test_submit_on_modal_dispatches(
    tmp_path: Path, fake_modal: list[tuple[Path, Resources, list[str], bool]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("sbatch called"))
    path = _cfg(tmp_path, '[launch]\norchestrator = "slurm"\n[launch.modal]\nregion = "us"\n')
    out = _submit(str(path), "--on", "modal", "--gpu", "H100", "--ncu", "2")
    assert "sbatch" not in out
    assert fake_modal == [
        (
            path,
            Resources(orchestrator="modal", gpu="H100", extras={"region": "us"}),
            ["ncu", str(path), "--questions", "2"],
            True,
        )
    ]


def test_launch_table_alone_selects_modal(
    tmp_path: Path, fake_modal: list[tuple[Path, Resources, list[str], bool]]
) -> None:
    path = _cfg(tmp_path, '[launch]\norchestrator = "modal"\ngpu = "H100"\n')
    _submit(str(path))
    assert [(r.orchestrator, r.gpu, args) for _, r, args, _ in fake_modal] == [
        ("modal", "H100", ["run", str(path)])
    ]


class _Stop(Exception):
    pass


def test_ncu_log_name_uses_env_job_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("MODAL_TASK_ID", "ta-9")
    cfg = _cfg(tmp_path)
    monkeypatch.chdir(tmp_path)
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        seen.append(cmd)
        raise _Stop

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(_Stop):
        cli.ncu_cmd(cfg, questions=1, results_dir=tmp_path, ncu_bin="ncu")
    log = seen[0][seen[0].index("--log-file") + 1]
    assert log == str(Path("logs") / "ncu-ta-9-exp-0.csv")
