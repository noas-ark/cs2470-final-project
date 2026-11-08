from pathlib import Path

import pytest

from s1 import env
from s1.launch import Resources


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("SLURM_JOB_ID", "MODAL_TASK_ID", "S1_GIT_COMMIT", "S1_LOG_DIR", "MODELS_DIR"):
        monkeypatch.delenv(var, raising=False)


def test_local_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "alice")
    assert env.orchestrator() == "local"
    assert env.job_id() == "local"
    assert env.log_dir() == Path("logs")
    assert env.models_dir() == Path("/shared/home/alice/models")


def test_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "4059")
    monkeypatch.setenv("MODAL_TASK_ID", "ta-1")
    assert env.orchestrator() == "slurm"
    assert env.job_id() == "4059"


def test_modal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODAL_TASK_ID", "ta-01ABC")
    monkeypatch.setenv("S1_LOG_DIR", "/runs/logs")
    monkeypatch.setenv("MODELS_DIR", "/models")
    assert env.orchestrator() == "modal"
    assert env.job_id() == "ta-01ABC"
    assert env.log_dir() == Path("/runs/logs")
    assert env.models_dir() == Path("/models")


def test_git_commit_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S1_GIT_COMMIT", "abc1234-dirty")
    assert env.git_commit() == "abc1234-dirty"


def test_git_commit_from_checkout() -> None:
    commit = env.git_commit()
    assert commit == "unknown" or commit.split("-")[0].isalnum()


def test_resources_defaults() -> None:
    r = Resources(orchestrator="modal", gpu="H100")
    assert (r.gpus, r.cpus, r.mem, r.time, dict(r.extras)) == (1, 8, "32G", "01:00:00", {})


def test_resolve_path_models_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """`$MODELS_DIR` resolves on Modal (variable set) and on the cluster (unset: the default layout)."""
    monkeypatch.setenv("MODELS_DIR", "/models")
    assert env.resolve_path("$MODELS_DIR/openjev/tokenizer.json") == Path("/models/openjev/tokenizer.json")
    assert env.resolve_path("${MODELS_DIR}/x.json") == Path("/models/x.json")
    monkeypatch.delenv("MODELS_DIR")
    monkeypatch.setenv("USER", "noa")
    want = Path("/shared/home/noa/models/openjev/tokenizer.json")
    assert env.resolve_path("$MODELS_DIR/openjev/tokenizer.json") == want


def test_resolve_path_other_vars_and_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/noa")
    monkeypatch.setenv("TOK", "/t")
    assert env.resolve_path("~/models/a.json") == Path("/home/noa/models/a.json")
    assert env.resolve_path("$TOK/a.json") == Path("/t/a.json")
    assert env.resolve_path("$MODELS_DIRX/a.json") == Path("$MODELS_DIRX/a.json")  # not the same variable
