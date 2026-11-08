"""The `elephant` backend: baselabs/elephant-1 on its own serving runtime, in a separate process.

elephant-1 is served by OneJev's qev engine @81ce62f plus the checkpoint's serve.patch (reviewed
2026-11-04: no network, subprocess or pickle use; it adds block-bidirectional attention in the 8
full-attention layers, a shared-prefix KV fork that scores every question of a request in one batched
forward, and the slot_lm readout). Upstream runs it under torch 2.8 / transformers 5.19, which conflict
with this harness, so it lives
in its own uv project (`runtimes/elephant`, built by `sbatch runtimes/elephant/setup.sh`) and runs as
`runtimes/elephant/worker.py`, spoken to over a pipe (one JSON object per line). The worker records every
forward pass and, with `measure_ceilings`, the GPU's ceilings before it loads the weights. MIT.

The LM head runs in fp32 by default (qev's `head_dtype`, against bf16 logit quantization): about 1B weights
(4096 x 248,320) at 4 bytes, so 2 GB more than bf16 and FP32 rather than tensor-core math for that matmul.

Weights: `model_dir`, default $MODELS_DIR/elephant-1 (scripts/download_model.sh at the catalog revision).
Settings (all optional):
    elephant_python   the runtime's interpreter       default runtimes/elephant/.venv/bin/python
    elephant_worker   the worker script                default runtimes/elephant/worker.py
    head_dtype        float32 | bfloat16               default float32 (from the spec, as upstream)
    measure_ceilings  see backends/semif.py            default false
    require_gpu       see backends/gpu.py
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ...record import Scalar
from ...registry import get_bool, get_str
from ..spec import ModelSpec
from . import gpu

RUNTIME = Path(__file__).resolve().parents[4] / "runtimes" / "elephant"


@dataclass
class Elephant:
    proc: subprocess.Popen[str]
    ready: dict[str, object]
    lock: threading.Lock = field(default_factory=threading.Lock)
    forwards: list[dict[str, object]] = field(default_factory=list)

    def _call(self, req: dict[str, object]) -> dict[str, object]:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("elephant worker has no pipes")
        with self.lock:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"elephant worker exited ({self.proc.poll()}); see its stderr in the run log")
        resp = json.loads(line)
        if not isinstance(resp, dict):
            raise TypeError(f"elephant worker sent {line!r}")
        fwd = resp.get("forwards", [])
        if isinstance(fwd, list):
            self.forwards.extend({str(k): v for k, v in f.items()} for f in fwd if isinstance(f, dict))
        return {str(k): v for k, v in resp.items()}

    def _decide(self, state: str, questions: Mapping[str, object], count: bool) -> dict[str, object]:
        resp = self._call({"op": "decide", "state": state, "questions": dict(questions), "count": count})
        if "error" in resp:
            raise RuntimeError(f"elephant-1: {resp['error']}")
        return resp

    def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
        answers = self._decide(state, questions, count=False)["answers"]
        if not isinstance(answers, dict):
            raise TypeError("elephant worker returned no answers")
        return {str(k): v for k, v in answers.items()}

    def profile_info(self) -> dict[str, object]:
        return {k: self.ready[k] for k in ("config", "head_bytes", "ceilings")}

    def take_forwards(self) -> list[dict[str, object]]:
        out, self.forwards = self.forwards, []
        return out

    def count_flops(self, state: str, questions: Mapping[str, object]) -> float:
        flops = self._decide(state, questions, count=True)["flops"]
        if not isinstance(flops, int | float):
            raise TypeError("elephant worker returned no FLOP count")
        return float(flops)

    def profiler(self, on: bool) -> None:
        """Open/close the CUDA profiler range in the worker, where the model's CUDA context is."""
        self._call({"op": "profiler", "on": on})

    def run_info(self) -> dict[str, Scalar]:
        env = self.ready.get("env", {})
        return {f"env_{k}": str(v) for k, v in env.items()} if isinstance(env, dict) else {}

    def close(self) -> None:
        if self.proc.poll() is None:
            with contextlib.suppress(RuntimeError, OSError, ValueError):
                self._call({"op": "quit"})  # quit has no reply; the worker exits
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def build(spec: ModelSpec, settings: dict[str, Scalar]) -> Elephant:
    name = spec.repo.rpartition("/")[2]
    model_dir = gpu.local_dir(settings, "model_dir", name, gpu.download_hint(spec.repo, spec.revision, name))
    python = Path(get_str(settings, "elephant_python", str(RUNTIME / ".venv" / "bin" / "python")))
    if not python.exists():
        raise FileNotFoundError(
            f"{python} missing: build the runtime with `sbatch runtimes/elephant/setup.sh`"
        )
    # The GPU name the runner read from nvidia-smi: no torch here (a CUDA context would take worker memory)
    found = str(settings.get("gpu", ""))
    if (required := get_str(settings, "require_gpu", "")) and not gpu.gpu_matches(found, required):
        raise RuntimeError(f"{spec.name} requires a {required!r} GPU, got {found!r}")
    worker = get_str(settings, "elephant_worker", str(RUNTIME / "worker.py"))
    cmd = [str(python), worker, "--model-dir", str(model_dir)]
    cmd += ["--head-dtype", get_str(settings, "head_dtype", "float32")]
    if get_bool(settings, "measure_ceilings", False):
        cmd.append("--measure-ceilings")
    # stderr is inherited: the worker's logs land in the run log
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    if proc.stdout is None:
        raise RuntimeError("could not start the elephant worker")
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"elephant worker exited during startup ({proc.wait()}); see the run log")
    ready = json.loads(line)
    if not ready.get("ready"):
        proc.kill()
        raise RuntimeError(f"elephant worker did not start: {ready}")
    return Elephant(proc, ready)


class ElephantBackend:
    def build(self, spec: ModelSpec, settings: dict[str, Scalar]) -> Elephant:
        return build(spec, settings)
