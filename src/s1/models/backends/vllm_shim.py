"""The `vllm_shim` backend: OpenJev (huggingface.co/openjev/openjev), a 27B Qwen3.5-based decision model.

It answers typed questions through the decision API of its helper (`helper/shim.py`, shipped in the
model repo), which sits in front of a vLLM server. `build` starts both on this node, with the serving
flags and calibration from the model card's serve/SERVE.md, and waits until they answer; `close`
stops them. Set `openjev_endpoint` to use an already-running helper instead.

Settings (all optional):
    openjev_dir             model directory           default <s1.env.models_dir()>/openjev ($MODELS_DIR,
                                                      else the cluster's shared models directory)
    openjev_endpoint        existing helper URL       default "" (start our own)
    tensor_parallel_size    GPUs to shard across      default 1 (use 4 on gpu-parallel's 4x A10G)
    quantization            vLLM --quantization       default "fp8" (the card's measured recipe; needs
                                                      compute capability >= 8.9). "none" serves bf16.
    max_model_len           context limit (tokens)    default 16384
    prefix_caching          vLLM prefix caching       default true. false passes --no-enable-prefix-caching
                                                      (vLLM V1 would otherwise turn it on by default).
                                                      For this hybrid model the cache works in blocks of 784
                                                      tokens (see the vLLM log), so shorter prompts never hit.
    openjev_startup_s       readiness timeout         default 1800 (loading 54 GB over the network is slow)

vLLM and shim logs go to $S1_LOG_DIR (s1.env.log_dir(), default ./logs) as `vllm-<job>-<port>.log` and
`shim-<job>-<port>.log`, where <job> is s1.env.job_id() (Slurm job id, Modal task id, or "local").
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ... import env
from ...record import Scalar
from ...registry import get_bool, get_float, get_int, get_str
from ..spec import ModelSpec

# The model card's calibration; its accuracy numbers only hold with exactly these.
SHIM_ENV = {
    "READOUT_T": "0.85",
    "READOUT_NOUL_T": "1.829074",
    "READOUT_NOUL_BIAS": "0",
    "READOUT_TARGETED": "1",
    "READOUT_INSTR_STYLE": "pyrepr",
    "SHIM_STAGGER": "1",
}
LOG_DIR: Path | None = None  # override for tests; None means env.log_dir(), resolved when used


def log_dir() -> Path:
    return LOG_DIR if LOG_DIR is not None else env.log_dir()


def model_dir(settings: dict[str, Scalar]) -> Path:
    """Setting `openjev_dir` if given, else $MODELS_DIR/openjev."""
    return Path(get_str(settings, "openjev_dir", str(env.models_dir() / "openjev")))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _post(url: str, body: dict[str, object], timeout: float) -> dict[str, object]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    if not isinstance(out, dict):
        raise TypeError(f"expected a JSON object from {url}, got {type(out).__name__}")
    return {str(k): v for k, v in out.items()}


def _ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return bool(resp.status == 200)
    except (urllib.error.URLError, OSError):
        return False


@dataclass
class OpenJev:
    endpoint: str
    request_timeout_s: float = 300.0
    procs: list[subprocess.Popen[bytes]] = field(default_factory=list)

    def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
        """POST /v1/systemone; returns the `answers` map (one typed answer per question key)."""
        out = _post(
            f"{self.endpoint}/v1/systemone",
            {"model": "openjev", "state": state, "questions": dict(questions)},
            self.request_timeout_s,
        )
        answers = out.get("answers")
        if not isinstance(answers, dict):
            raise TypeError(f"response has no answers: {str(out)[:200]}")
        return {str(k): v for k, v in answers.items()}

    def close(self) -> None:
        for p in reversed(self.procs):  # helper first, then vLLM
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in self.procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        self.procs.clear()


def _spawn(cmd: list[str], log: Path, env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as fh:
        # own process group, so close() also takes down vLLM's tensor-parallel workers
        return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def _wait_ready(url: str, proc: subprocess.Popen[bytes], log: Path, deadline: float) -> None:
    while not _ready(url):
        if proc.poll() is not None:
            raise RuntimeError(f"{proc.args!r} exited ({proc.returncode}); see {log}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"{url} not ready in time; see {log}")
        time.sleep(2)


def serve_args(settings: dict[str, Scalar], model_dir: Path, port: int) -> list[str]:
    """The `vllm serve` command line for these settings."""
    quant = get_str(settings, "quantization", "fp8")
    prefix_caching = get_bool(settings, "prefix_caching", True)
    return [
        str(Path(sys.executable).parent / "vllm"), "serve", str(model_dir),
        "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", "qwen",
        "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching",
        "--max-model-len", str(get_int(settings, "max_model_len", 16384)),
        "--gpu-memory-utilization", "0.90",
        "--limit-mm-per-prompt", '{"image":1}',
        "--trust-remote-code",
        "--max-num-seqs", "256",
        "--max-logprobs", "64",
        "--gdn-prefill-backend", "triton",
        "--tensor-parallel-size", str(get_int(settings, "tensor_parallel_size", 1)),
        *(["--quantization", quant] if quant != "none" else []),
    ]  # fmt: skip


def build(settings: dict[str, Scalar]) -> OpenJev:
    if endpoint := get_str(settings, "openjev_endpoint", ""):
        return OpenJev(endpoint.rstrip("/"))

    mdir = model_dir(settings)
    shim = mdir / "helper" / "shim.py"
    if not shim.exists():
        raise FileNotFoundError(f"{shim} missing: sbatch scripts/download_model.sh openjev/openjev")
    vllm_port, shim_port = _free_port(), _free_port()
    tag = f"{env.job_id()}-{vllm_port}"
    logs = log_dir()

    vllm = _spawn(serve_args(settings, mdir, vllm_port), logs / f"vllm-{tag}.log")
    jev = OpenJev(f"http://127.0.0.1:{shim_port}", procs=[vllm])
    try:
        deadline = time.monotonic() + get_float(settings, "openjev_startup_s", 1800)
        _wait_ready(f"http://127.0.0.1:{vllm_port}/health", vllm, logs / f"vllm-{tag}.log", deadline)

        shim_env = (
            os.environ | SHIM_ENV | {"VLLM": f"http://127.0.0.1:{vllm_port}/v1", "TOKENIZER": str(mdir)}
        )
        cmd = [sys.executable, str(shim), "--host", "127.0.0.1", "--port", str(shim_port)]
        jev.procs.append(_spawn(cmd, logs / f"shim-{tag}.log", shim_env))
        _wait_ready(f"{jev.endpoint}/v1/version", jev.procs[-1], logs / f"shim-{tag}.log", deadline)
    except BaseException:
        jev.close()
        raise
    return jev


class VllmShim:
    def build(self, spec: ModelSpec, settings: dict[str, Scalar]) -> OpenJev:
        return build(settings)
