"""Shared plumbing for the in-process (transformers) backends: GPU guard, local weights, dtype.

torch is imported inside the functions, so importing this module stays cheap.

Settings (all optional):
    require_gpu    substring the CUDA device name must contain, ignoring case and spaces
                   (e.g. "rtx pro 4500"); "" accepts any CUDA GPU
    dtype          torch dtype for the weights        default "bfloat16"
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ... import env
from ...record import Scalar
from ...registry import get_str

if TYPE_CHECKING:
    import torch


def models_dir() -> Path:
    return env.models_dir()


def download_hint(repo: str, revision: str | None, name: str) -> str:
    rev = revision or '""'  # the script's second argument is the revision; "" means the default branch
    return f"sbatch scripts/download_model.sh {repo} {rev} {name}"


def local_dir(settings: dict[str, Scalar], key: str, name: str, fix: str) -> Path:
    """Setting `key` if given, else $MODELS_DIR/<name>; raises with `fix` (how to get it) if it is missing."""
    path = Path(get_str(settings, key, str(models_dir() / name)))
    if not path.is_dir():
        raise FileNotFoundError(f"{path} missing: {fix}")
    return path


def _norm(name: str) -> str:
    return "".join(name.lower().split())


def gpu_matches(device_name: str, required: str) -> bool:
    return _norm(required) in _norm(device_name)


def cuda_device(settings: dict[str, Scalar], model: str) -> torch.device:
    """cuda:0, after checking there is a GPU and that it is the one `require_gpu` asks for."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError(f"{model} runs on a GPU only; submit it to a GPU partition")
    name = torch.cuda.get_device_name(0)
    if (required := get_str(settings, "require_gpu", "")) and not gpu_matches(name, required):
        raise RuntimeError(f"{model} requires a {required!r} GPU, got {name!r}")
    return torch.device("cuda", 0)


def check_sha256(path: Path, expected: str) -> None:
    """Refuse to run shipped model code that is not the file we reviewed."""
    import hashlib  # noqa: PLC0415

    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != expected:
        raise RuntimeError(
            f"{path} has sha256 {got}, not the reviewed {expected}; re-review before running it"
        )


def check_git_commit(path: Path, expected: str) -> None:
    """Refuse a code checkout that is not at the pinned commit or has local changes."""
    import subprocess  # noqa: PLC0415

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    if (head := git("rev-parse", "HEAD")) != expected:
        raise RuntimeError(f"{path} is at {head}, not the pinned {expected}")
    if git("status", "--porcelain"):
        raise RuntimeError(f"{path} has local changes; the pinned code must be unmodified")


def torch_dtype(settings: dict[str, Scalar]) -> torch.dtype:
    import torch  # noqa: PLC0415

    name = get_str(settings, "dtype", "bfloat16")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unknown dtype {name!r}")
    return dtype
