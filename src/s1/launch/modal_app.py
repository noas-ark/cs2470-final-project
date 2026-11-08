"""Modal launcher: run `s1 <args>` in a Modal container and bring the records and logs back.

    [launch]
    orchestrator = "modal"
    gpu = "H100"          # -> Modal "H100!" (no silent H200 upgrade); omit for a CPU-only container
    gpus = 1              # -> "H100!:2" when > 1
    cpus = 8              # -> cpu=8
    mem = "32G"           # -> memory=32768 (MiB)
    time = "01:00:00"     # -> timeout=3600 s (Modal's maximum is 24 h)
    [launch.modal]        # optional, see EXTRAS
    openjev = true        # install the openjev extra (vLLM, transformers); default: true iff gpu is set
    echo = true           # register a built-in fake model "echo" (tests' EchoModel) for a CPU round trip
    hf_secret = "hf"      # a Modal secret holding HF_TOKEN, for gated repos

What runs where:
    image    Python 3.12 + `uv sync --frozen` of this repo's uv.lock (with `--extra openjev` for GPU runs),
             then src/ and configs/ copied in. tests/ is not shipped.
    volumes  s1-models -> /models (MODELS_DIR), s1-hf-cache -> /hf-cache (HF_HOME), s1-runs -> /runs.
    command  the same `s1 <args>` Slurm's --wrap runs; `run` gets `--results-dir /runs/<call>/results`, and
             `--set require_gpu=<name>` so a wrong card fails fast (unless the config already sets it).
    output   /runs/<call>/{results,logs}/ stay on the volume (a copy of everything); the records are written
             into local results/ with RunRecord.save and the logs are downloaded into local logs/.

`modal` is imported only inside the functions that talk to Modal (`uv sync --group modal`), so importing this
module, the tests and the cluster never need it. `download()` fetches a Hub model into the s1-models volume,
like scripts/download_model.sh.

This module is also the container's entry point: `python -m s1.launch.modal_app [--echo] <s1 args>`.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict

from .. import config, env
from ..record import RESULTS_DIR, RunRecord, Scalar
from .base import Resources

APP_NAME = "s1"
PYTHON = "3.12"
UV_VERSION = "0.13.5"  # uv.lock is lock revision 5; pin the uv that reads it
CUDA_DEVEL_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu24.04"  # GPU image base: nvcc for FlashInfer's JIT
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKDIR = "/root/s1"  # container cwd: src/, configs/ and the submitted config live here
VENV_PYTHON = "/.uv/.venv/bin/python"  # where Image.uv_sync puts the environment
MODELS_MOUNT = "/models"
HF_MOUNT = "/hf-cache"
RUNS_MOUNT = "/runs"
VOLUMES: Mapping[str, str] = {MODELS_MOUNT: "s1-models", HF_MOUNT: "s1-hf-cache", RUNS_MOUNT: "s1-runs"}
IMAGE_ENV: Mapping[str, str] = {
    "PYTHONPATH": f"{WORKDIR}/src",
    "MODELS_DIR": MODELS_MOUNT,
    "HF_HOME": HF_MOUNT,
    "PYTHONUNBUFFERED": "1",
}
RUN_DIR = "{run_dir}"  # placeholder in planned args, filled in the container with /runs/<function call id>
MAX_TIMEOUT_S = 24 * 3600
EXTRAS = {"openjev", "echo", "hf_secret"}
DOWNLOAD_CPU, DOWNLOAD_MEMORY_MIB, DOWNLOAD_TIMEOUT_S = 8.0, 16 * 1024, 4 * 3600  # as download_model.sh


@dataclass(frozen=True)
class GpuKind:
    modal: str  # Modal's gpu= string: exact type, never a fallback list
    require: str  # `require_gpu` substring the CUDA device name must contain
    example: str  # a device name nvidia-smi reports for it (to check a config's own require_gpu)


GPUS: Mapping[str, GpuKind] = {
    "T4": GpuKind("T4", "T4", "Tesla T4"),
    "L4": GpuKind("L4", "L4", "NVIDIA L4"),
    "A10G": GpuKind("A10", "A10G", "NVIDIA A10G"),  # Modal calls the A10G "A10"
    "L40S": GpuKind("L40S", "L40S", "NVIDIA L40S"),
    "A100-40GB": GpuKind("A100-40GB", "A100", "NVIDIA A100-SXM4-40GB"),
    "A100-80GB": GpuKind("A100-80GB", "A100", "NVIDIA A100-SXM4-80GB"),
    "H100": GpuKind("H100!", "H100", "NVIDIA H100 80GB HBM3"),  # "!": no automatic upgrade to H200
    "H200": GpuKind("H200", "H200", "NVIDIA H200"),
    "B200": GpuKind("B200", "B200", "NVIDIA B200"),
}


def gpu_kind(name: str) -> GpuKind:
    """The Modal mapping of a portable GPU name (case-insensitive); unknown names are an error."""
    for key, kind in GPUS.items():
        if key.lower() == name.strip().lower():
            return kind
    raise ValueError(f"unknown GPU {name!r} for Modal; use one of {', '.join(GPUS)}")


def modal_gpu(gpu: str | None, gpus: int = 1) -> str | None:
    """`gpu=` for Modal: None (CPU only), "<type>" or "<type>:<n>"."""
    if gpu is None:
        return None
    if gpus < 1:
        raise ValueError(f"gpus must be >= 1 when gpu is set, got {gpus}")
    kind = gpu_kind(gpu).modal
    return kind if gpus == 1 else f"{kind}:{gpus}"


_MEM_UNIT_MIB = {"K": 1 / 1024, "M": 1, "": 1, "G": 1024, "T": 1024 * 1024}


def mem_mib(mem: str) -> int:
    """Slurm memory ("28G", "512M", "1T"; no unit = MB) -> MiB, Modal's `memory=`."""
    m = re.fullmatch(r"(\d+)\s*([KMGT]?)B?", mem.strip(), re.IGNORECASE)
    if m is None:
        raise ValueError(f"bad mem {mem!r}; use Slurm syntax such as '28G'")
    mib = int(m.group(1)) * _MEM_UNIT_MIB[m.group(2).upper()]
    return max(1, round(mib))


def timeout_s(time: str) -> int:
    """Slurm time ("MM", "MM:SS", "HH:MM:SS", "D-HH[:MM[:SS]]") -> seconds, Modal's `timeout=`."""
    days, sep, clock = time.strip().rpartition("-")
    parts = clock.split(":")
    if not ((days.isdigit() or not sep) and 1 <= len(parts) <= 3 and all(p.isdigit() for p in parts)):
        raise ValueError(f"bad time {time!r}; use Slurm syntax such as '02:00:00'")
    nums = [int(x) for x in parts]
    if sep:  # D-HH[:MM[:SS]]
        h, mnt, s = [*nums, 0, 0][:3]
    elif len(nums) == 3:  # HH:MM:SS
        h, mnt, s = nums
    else:  # MM or MM:SS
        h, mnt, s = 0, nums[0], nums[1] if len(nums) == 2 else 0
    seconds = (((int(days) if sep else 0) * 24 + h) * 60 + mnt) * 60 + s
    if not 0 < seconds <= MAX_TIMEOUT_S:
        raise ValueError(f"time {time!r} must be between 1 s and Modal's 24 h limit")
    return seconds


def _norm(name: str) -> str:
    return "".join(name.lower().split())  # the same comparison as backends/gpu.py's gpu_matches


@dataclass(frozen=True)
class ModalPlan:
    """Everything a Modal run needs, computed locally without contacting Modal (the dry run prints it)."""

    gpu: str | None
    cpu: float
    memory_mib: int
    timeout_s: int
    openjev: bool
    echo: bool
    hf_secret: str | None
    cfg_remote: str  # config path relative to WORKDIR
    cfg_text: str
    s1_args: tuple[str, ...]  # contains RUN_DIR, filled in remotely
    env: Mapping[str, str] = field(default_factory=dict)  # passed to the remote command

    def describe(self) -> str:
        extra = "--extra openjev" if self.openjev else "(base dependencies)"
        vols = ", ".join(f"{name} -> {mount}" for mount, name in VOLUMES.items())
        cmd = shlex.join(["s1", *(a.replace(RUN_DIR, f"{RUNS_MOUNT}/CALL_ID") for a in self.s1_args)])
        lines = [
            f"modal app {APP_NAME!r}, image: python {PYTHON} + uv {UV_VERSION} sync --frozen {extra}"
            f" + src/ configs/ at {WORKDIR}",
            f"  gpu={self.gpu!r} cpu={self.cpu:g} memory={self.memory_mib} MiB timeout={self.timeout_s} s",
            f"  volumes: {vols}",
            f"  env: {', '.join(f'{k}={v}' for k, v in {**IMAGE_ENV, **self.env}.items())}",
            f"  config: {self.cfg_remote}" + ("  (+ built-in echo model)" if self.echo else ""),
            f"  command: {cmd}",
        ]
        if self.hf_secret is not None:
            lines.append(f"  secret: {self.hf_secret}")
        return "\n".join(lines)


def _extras(resources: Resources) -> Mapping[str, Scalar]:
    unknown = set(resources.extras) - EXTRAS
    if unknown:
        raise ValueError(f"unknown [launch.modal] keys {sorted(unknown)}; known: {sorted(EXTRAS)}")
    return resources.extras


def _cfg_remote(cfg: Path) -> str:
    """Where the config goes in the container: its repo-relative path, or configs/submitted/<name>."""
    try:
        return cfg.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"configs/submitted/{cfg.name}"


def _require_gpu_args(cfg: Path, kind: GpuKind | None) -> list[str]:
    """`--set require_gpu=...` unless the config sets it already (then it must agree with the request)."""
    found = ((s.invariants | s.params).get("require_gpu", "") for s in config.load(cfg))
    given = {str(r) for r in found if r != ""}
    if kind is None:
        if given:
            raise ValueError(
                f"{cfg}: require_gpu = {sorted(given)} but no GPU requested; pass --gpu or [launch] gpu"
            )
        return []
    for req in sorted(given):
        if _norm(req) not in _norm(kind.example):
            raise ValueError(f"{cfg}: require_gpu = {req!r} contradicts [launch] gpu ({kind.example})")
    return [] if given else ["--set", f"require_gpu={kind.require}"]


def plan(
    cfg: Path, resources: Resources, s1_args: Sequence[str], *, git_commit: str | None = None
) -> ModalPlan:
    """Map Resources and `s1 <args>` to a Modal run (pure: reads the config, never contacts Modal)."""
    extras = _extras(resources)
    kind = None if resources.gpu is None else gpu_kind(resources.gpu)
    cfg_remote = _cfg_remote(cfg)
    args = [cfg_remote if a == str(cfg) else a for a in s1_args]
    if args[:1] in (["run"], ["ncu"]):
        args += ["--results-dir", f"{RUN_DIR}/results"]
    if args[:1] in (["run"], ["ncu"]):
        required = _require_gpu_args(cfg, kind)
        args += required if args[:1] == ["run"] else []  # `s1 ncu` takes no --set
    hf_secret = extras.get("hf_secret")
    return ModalPlan(
        gpu=modal_gpu(resources.gpu, resources.gpus),
        cpu=float(resources.cpus),
        memory_mib=mem_mib(resources.mem),
        timeout_s=timeout_s(resources.time),
        openjev=bool(extras.get("openjev", kind is not None)),
        echo=bool(extras.get("echo", False)),
        hf_secret=None if hf_secret is None else str(hf_secret),
        cfg_remote=cfg_remote,
        cfg_text=cfg.read_text(),
        s1_args=tuple(args),
        env={
            "S1_GIT_COMMIT": env.git_commit() if git_commit is None else git_commit,
            "S1_CPUS_REQUESTED": str(resources.cpus),
        },
    )


class RemoteResult(TypedDict):
    call_id: str
    returncode: int
    records: dict[str, str]  # file name -> record JSON text
    logs: list[str]  # paths relative to /runs/<call id>, e.g. "logs/<call id>.out"


def _str_list(value: object, what: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise TypeError(f"remote {what} must be a list of strings")
    return [str(v) for v in value]


def remote_result(text: str) -> RemoteResult:
    """Parse and check what `_remote_run` returned."""
    raw = json.loads(text)
    records = raw.get("records") if isinstance(raw, dict) else None
    if not isinstance(raw, dict) or not isinstance(records, dict):
        raise TypeError("remote result must be an object with a `records` object")
    return {
        "call_id": str(raw["call_id"]),
        "returncode": int(raw["returncode"]),
        "records": {str(k): str(v) for k, v in records.items()},
        "logs": _str_list(raw.get("logs", []), "logs"),
    }


def write_back(
    result: RemoteResult,
    fetch: Callable[[str], Iterable[bytes]],
    *,
    results_dir: Path = RESULTS_DIR,
    logs_dir: Path | None = None,
) -> tuple[list[Path], list[Path]]:
    """Save the remote records with RunRecord.save and download each log (`fetch(path in s1-runs)`)."""
    logs_dir = env.log_dir() if logs_dir is None else logs_dir
    records: list[Path] = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in sorted(result["records"].items()):
            staged = Path(tmp) / Path(name).name
            staged.write_text(text)
            records.append(RunRecord.load(staged).save(results_dir))
    logs: list[Path] = []
    for rel in result["logs"]:
        parts = Path(rel).parts
        dest = logs_dir.joinpath(*(parts[1:] if parts[:1] == ("logs",) else parts))
        if ".." in parts:
            raise ValueError(f"refusing log path {rel!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as out:
            for chunk in fetch(f"{result['call_id']}/{rel}"):
                out.write(chunk)
        logs.append(dest)
    return records, logs


# ---- talking to Modal (imports modal lazily) ----


def _modal() -> ModuleType:
    try:
        return importlib.import_module("modal")
    except ModuleNotFoundError as e:
        raise RuntimeError("the Modal client is not installed; run: uv sync --group modal") from e


def _check_python() -> None:
    here = f"{sys.version_info.major}.{sys.version_info.minor}"
    if here != PYTHON:  # functions are shipped with cloudpickle, which needs the same Python on both ends
        raise RuntimeError(f"launch from Python {PYTHON} (this is {here}): uv run --python {PYTHON} ...")


def _image(modal: ModuleType, *, openjev: bool) -> object:
    if openjev:
        # GPU runs need a CUDA toolkit at /usr/local/cuda, as on the cluster nodes: FlashInfer JIT-compiles
        # its sampling kernel with nvcc during vLLM's warmup (debian_slim has no nvcc: vLLM failed to start).
        # 13.0 matches torch 2.13.0's CUDA build and the cluster's toolkit. Triton needs gcc for its launcher.
        base = (
            modal.Image.from_registry(CUDA_DEVEL_IMAGE, add_python=PYTHON)
            .apt_install("gcc", "libc6-dev")
            .env({"CUDA_HOME": "/usr/local/cuda"})
        )
    else:
        base = modal.Image.debian_slim(python_version=PYTHON)
    return (
        base.uv_sync(
            str(REPO_ROOT), extras=["openjev"] if openjev else [], frozen=True, uv_version=UV_VERSION
        )
        .env(dict(IMAGE_ENV))
        .add_local_dir(REPO_ROOT / "src", f"{WORKDIR}/src", ignore=["**/__pycache__", "**/*.pyc"])
        .add_local_dir(REPO_ROOT / "configs", f"{WORKDIR}/configs")
    )


def _volumes(modal: ModuleType) -> dict[str, object]:
    return {mount: modal.Volume.from_name(name, create_if_missing=True) for mount, name in VOLUMES.items()}


def _secrets(modal: ModuleType, name: str | None) -> list[object]:
    return [] if name is None else [modal.Secret.from_name(name)]


def build_image(*, openjev: bool = False) -> None:
    """Build (and cache) the run image without running anything: the first, cheapest end-to-end check."""
    _check_python()
    modal = _modal()
    app = modal.App(APP_NAME)
    with modal.enable_output(), app.run():
        image: Any = _image(modal, openjev=openjev)  # a modal.Image (modal is imported lazily)
        image.build(app)  # instance method: Image.build(self, app)


@dataclass
class ModalLauncher:
    """`Launcher` for Modal: an ephemeral app per submit; blocks until the run ends, then writes back."""

    results_dir: Path = RESULTS_DIR
    logs_dir: Path | None = None
    log: Callable[[str], None] = print

    def submit(
        self, cfg: Path, resources: Resources, s1_args: Sequence[str], *, dry_run: bool = False
    ) -> str:
        p = plan(cfg, resources, s1_args)
        self.log(p.describe())
        if dry_run:
            return ""
        _check_python()
        modal = _modal()
        app = modal.App(APP_NAME)
        volumes = _volumes(modal)
        fn = app.function(
            image=_image(modal, openjev=p.openjev),
            gpu=p.gpu,
            cpu=p.cpu,
            memory=p.memory_mib,
            timeout=p.timeout_s,
            volumes=volumes,
            secrets=_secrets(modal, p.hf_secret),
            serialized=True,
            include_source=False,
        )(_remote_run)
        payload = json.dumps(
            {
                "cfg_remote": p.cfg_remote,
                "cfg_text": p.cfg_text,
                "s1_args": list(p.s1_args),
                "env": dict(p.env),
                "echo": p.echo,
            }
        )
        with modal.enable_output(), app.run():
            call = fn.spawn(payload)
            self.log(f"modal function call {call.object_id}")
            result = remote_result(call.get())
        runs = modal.Volume.from_name(VOLUMES[RUNS_MOUNT])
        records, logs = write_back(
            result, runs.read_file, results_dir=self.results_dir, logs_dir=self.logs_dir
        )
        for path in [*records, *logs]:
            self.log(f"wrote {path}")
        if result["returncode"] != 0:
            raise RuntimeError(
                f"remote `s1 {' '.join(p.s1_args[:2])}` exited {result['returncode']}; "
                f"see {logs[0] if logs else 'the s1-runs volume'}"
            )
        return result["call_id"]


class DownloadReport(TypedDict):
    repo: str
    revision: str | None
    dest: str
    sha256_checked: int
    sha256_failed: list[str]


def download_report(text: str) -> DownloadReport:
    """Parse and check what `_remote_download` returned."""
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise TypeError("remote download report must be an object")
    revision = raw.get("revision")
    return {
        "repo": str(raw["repo"]),
        "revision": None if revision is None else str(revision),
        "dest": str(raw["dest"]),
        "sha256_checked": int(raw["sha256_checked"]),
        "sha256_failed": _str_list(raw.get("sha256_failed", []), "sha256_failed"),
    }


def download(
    repo: str, revision: str | None = None, dirname: str | None = None, *, hf_secret: str | None = None
) -> DownloadReport:
    """`s1 download`: fetch a Hub model into the s1-models volume at /models/<dirname> on a CPU container,
    then `sha256sum -c SHA256SUMS` when the repo ships one. Only README.md mismatching is reported, not fatal
    (the openjev card was edited after its sums were generated); any other mismatch raises."""
    _check_python()
    modal = _modal()
    app = modal.App(f"{APP_NAME}-download")
    fn = app.function(
        image=_image(modal, openjev=False),
        cpu=DOWNLOAD_CPU,
        memory=DOWNLOAD_MEMORY_MIB,
        timeout=DOWNLOAD_TIMEOUT_S,
        volumes={k: v for k, v in _volumes(modal).items() if k != RUNS_MOUNT},
        secrets=_secrets(modal, hf_secret),
        serialized=True,
        include_source=False,
    )(_remote_download)
    with modal.enable_output(), app.run():
        report = download_report(fn.remote(repo, revision, dirname or repo.split("/", 1)[-1]))
    failed = report["sha256_failed"]
    print(f"downloaded {repo}{'' if revision is None else '@' + revision} -> s1-models:{report['dest']}")
    if report["sha256_checked"]:
        print(f"sha256: {report['sha256_checked']} file(s) checked, failed: {failed or 'none'}")
    if [f for f in failed if f != "README.md"]:
        raise RuntimeError(f"sha256 mismatch in {report['dest']}: {failed}")
    return report


def verify_sha256sums(directory: Path, sums_file: str = "SHA256SUMS") -> tuple[int, list[str]]:
    """`sha256sum -c`: (number of entries, names that are missing or mismatch)."""
    entries = []
    for line in (directory / sums_file).read_text().splitlines():
        if line.strip():
            digest, _, name = line.strip().partition(" ")
            entries.append((digest.lower(), name.strip().removeprefix("*")))
    failed = []
    for digest, name in entries:
        path = directory / name
        if not path.is_file():
            failed.append(name)
            continue
        with path.open("rb") as f:
            got = hashlib.file_digest(f, "sha256").hexdigest()
        if got != digest:
            failed.append(name)
    return len(entries), failed


# ---- what runs in the container ----


def _python() -> str:
    return VENV_PYTHON if Path(VENV_PYTHON).exists() else sys.executable


def _remote_run(payload_json: str) -> str:
    """Container side of submit: write the config, run `s1 <args>` (teeing output to logs/<call id>.out),
    commit the volumes and return the records as JSON."""
    modal = _modal()
    payload = json.loads(payload_json)
    call_id = modal.current_function_call_id() or "local"
    run_dir = Path(RUNS_MOUNT) / call_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    cfg = Path(WORKDIR) / payload["cfg_remote"]
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(payload["cfg_text"])
    args = [a.replace(RUN_DIR, str(run_dir)) for a in payload["s1_args"]]
    cmd = [_python(), "-m", "s1.launch.modal_app", *(["--echo"] if payload["echo"] else []), *args]
    child_env = os.environ | payload["env"] | {"S1_LOG_DIR": str(run_dir / "logs")}
    out = run_dir / "logs" / f"{call_id}.out"
    with out.open("w") as log:
        log.write(f"$ {shlex.join(['s1', *args])}\n")
        proc = subprocess.Popen(
            cmd, cwd=WORKDIR, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
        log.write(f"EXIT={code}\n")
    records = {p.name: p.read_text() for p in sorted((run_dir / "results").glob("*.json"))}
    logs = [p.relative_to(run_dir).as_posix() for p in sorted((run_dir / "logs").rglob("*")) if p.is_file()]
    for name in (VOLUMES[RUNS_MOUNT], VOLUMES[HF_MOUNT]):
        modal.Volume.from_name(name).commit()
    result: RemoteResult = {"call_id": call_id, "returncode": code, "records": records, "logs": logs}
    return json.dumps(result)


def _remote_download(repo: str, revision: str | None, dirname: str) -> str:
    modal = _modal()
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    dest = Path(MODELS_MOUNT) / dirname
    snapshot_download(repo, revision=revision, local_dir=dest)
    checked, failed = verify_sha256sums(dest) if (dest / "SHA256SUMS").is_file() else (0, [])
    for name in (VOLUMES[MODELS_MOUNT], VOLUMES[HF_MOUNT]):
        modal.Volume.from_name(name).commit()
    report: DownloadReport = {
        "repo": repo,
        "revision": revision,
        "dest": str(dest),
        "sha256_checked": checked,
        "sha256_failed": failed,
    }
    return json.dumps(report)


class EchoModel:
    """The tests' EchoModel (tests/conftest.py), for a CPU round trip: first option / yes / level 0."""

    def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
        del state
        ((key, q),) = questions.items()
        if not isinstance(q, Mapping):
            raise TypeError(f"question {key!r} is not a mapping")
        criteria = q.get("criteria")
        opts = [str(o) for o in criteria] if isinstance(criteria, Iterable) else []
        if q.get("type") == "choice":
            probs = {o: (0.7 if i == 0 else 0.3 / (len(opts) - 1)) for i, o in enumerate(opts)}
            return {key: {"type": "choice", "choice": opts[0], "probabilities": probs}}
        if q.get("type") == "noul":
            return {key: {"type": "noul", "noul": 0.8}}
        levels = {str(i): (1.0 if i == 0 else 0.0) for i in range(len(opts))}
        return {key: {"type": "score", "score": 0.0, "probabilities": levels}}


def _main(argv: Sequence[str]) -> None:
    """Container entry: optionally register the echo model, then run the s1 CLI."""
    args = list(argv)
    if args[:1] == ["--echo"]:
        from .. import registry  # noqa: PLC0415

        registry.register_model("echo", lambda settings: EchoModel())
        args = args[1:]
    from ..cli import app  # noqa: PLC0415

    app(args, prog_name="s1")


if __name__ == "__main__":
    _main(sys.argv[1:])
