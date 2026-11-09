"""The `bongard` backend: Bongard-mini (huggingface.co/AgentBull/bongard-mini), a T5Gemma 2 encoder-decoder
with a judgment head.

Runs upstream's runtime, `bongard.inference.Predictor`, from a checkout of github.com/AgentBull/bongard at the
catalog's code pin (checked before import; the inference path loads only local files). The encoder reads the
JSON-encoded state plus every question's instructions once per request; the decoder reads each question with
its candidates, and the head scores each candidate's end marker against the decision marker. Temperatures come
from the checkpoint's temperatures.json. Requests over 32,768 tokens (state + one question) are rejected, not
truncated, so they count as invalid. Gemma terms.

The spec sets `require_gpu` to the 32 GB RTX Pro 4500 (`gpu-g7-reserved`): 15 GB of bf16 weights leave little
room on a 24 GB card for long states. A config may override it (the Modal fan-out sets one per GPU).

Roofline (benchmarks/roofline.py): every pass of `JudgmentModel.forward` is timed with `recorder.span`, as
a forward of the stack it runs (s1.flops.t5gemma2_stacks): `encode` (the state plus every question's
instructions, one row, bidirectional), `state_kv` (the encoded state projected to K/V once per decoder
layer) and each `decode_batch` (up to `bongard_branch_batch` questions, padded to the longest, attending to
the one shared state: `shared_past`). The judgment head's few small matmuls are not timed. Upstream's default
attention backend computes the decoder's partitioned attention in fp32 (torch.bmm), so its attention FLOPs
are not bf16 ones.

Weights and code:
    bongard_dir       checkpoint     default $MODELS_DIR/bongard-mini    (scripts/download_model.sh)
    bongard_code_dir  runtime code   default $MODELS_DIR/bongard-code    (scripts/fetch_code.sh)
Settings (all optional):
    bongard_branch_batch   questions decoded per pass    default 8 (upstream)
    measure_ceilings       measure the GPU's roofline ceilings before loading   default false
    require_gpu            see backends/gpu.py           from the spec
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ... import profiling
from ...record import Scalar
from ...registry import get_bool, get_int
from ..spec import ModelSpec
from . import gpu


def encoder_pass(tokens: int) -> dict[str, object]:
    """The encoder over one request's state (+ the questions' instructions): one bidirectional row."""
    return {"stack": "encoder", "rows": 1, "tokens": tokens, "past": 0, "kept": 0, "cached": False}


def cross_kv_pass(rows: int, tokens: int) -> dict[str, object]:
    """The encoded state's K/V, projected once per decoder layer (no attention)."""
    return {"stack": "cross_kv", "rows": rows, "tokens": tokens, "past": 0, "kept": 0, "cached": False}


def decoder_pass(lengths: Sequence[int], state: int) -> dict[str, object]:
    """One decoder batch: a row per question, padded to the longest, every row attending to the same `state`
    keys (read once: shared_past) and writing its own K/V to a private cache."""
    return {
        "stack": "decoder",
        "rows": len(lengths),
        "tokens": max(lengths),
        "past": state,
        "kept": 0,
        "cached": True,
        "shared_past": True,
        "real_tokens": sum(lengths),
    }


def instrument(model: object, recorder: profiling.ForwardRecorder) -> None:
    """Time `model`'s (a bongard.model.JudgmentModel) encode, state_kv and decode_batch as forwards."""
    encode: Callable[..., object] = model.encode  # ty: ignore[unresolved-attribute]
    state_kv: Callable[..., object] = model.state_kv  # ty: ignore[unresolved-attribute]
    decode: Callable[..., object] = model.decode_batch  # ty: ignore[unresolved-attribute]

    def timed_encode(tokens: Sequence[int], pixel_values: object = None) -> object:
        with recorder.span(encoder_pass(len(tokens))):
            return encode(tokens, pixel_values)

    def timed_state_kv(state: object) -> object:
        rows, tokens = state.shape[0], state.shape[1]  # ty: ignore[unresolved-attribute]
        with recorder.span(cross_kv_pass(int(rows), int(tokens))):
            return state_kv(state)

    def timed_decode(
        rows: Sequence[Sequence[int]], state: object, cache: object = None, **kw: object
    ) -> object:
        with recorder.span(decoder_pass([len(r) for r in rows], int(state.shape[1]))):  # ty: ignore[unresolved-attribute]
            return decode(rows, state, cache, **kw)

    model.encode = timed_encode  # ty: ignore[unresolved-attribute]
    model.state_kv = timed_state_kv  # ty: ignore[unresolved-attribute]
    model.decode_batch = timed_decode  # ty: ignore[unresolved-attribute]


@dataclass
class Bongard:
    predictor: object
    lock: threading.Lock = field(default_factory=threading.Lock)
    hf_config: dict[str, object] = field(default_factory=dict)  # backbone config.json, for s1.flops
    ceilings: dict[str, float] = field(default_factory=dict)
    recorder: profiling.ForwardRecorder | None = None

    def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
        with self.lock:
            out = self.predictor.predict({"state": state, "questions": dict(questions)})  # ty: ignore[unresolved-attribute]
        return {str(k): v for k, v in out["answers"].items()}

    def _model(self) -> object:
        return self.predictor.model  # ty: ignore[unresolved-attribute]

    def profile_info(self) -> dict[str, object]:
        if self.recorder is None:
            self.recorder = profiling.ForwardRecorder(None, head=False)
            instrument(self._model(), self.recorder)
        return {"config": self.hf_config, "head_bytes": 2, "ceilings": self.ceilings}

    def take_forwards(self) -> list[dict[str, object]]:
        return [] if self.recorder is None else self.recorder.take()

    def count_flops(self, state: str, questions: Mapping[str, object]) -> float:
        return profiling.count_flops(lambda: self.decide(state, questions))[1]

    def profiler(self, on: bool) -> None:
        import torch  # noqa: PLC0415

        torch.cuda.synchronize()
        (torch.cuda.profiler.start if on else torch.cuda.profiler.stop)()

    def run_info(self) -> dict[str, Scalar]:
        backbone = self._model().backbone  # ty: ignore[unresolved-attribute]
        return {f"env_{k}": v for k, v in profiling.kernel_env(backbone).items()}

    def close(self) -> None:
        """Drop the weights and free the GPU, so the next run in this process can load."""
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415

        self.recorder = None
        self.predictor = None
        gc.collect()
        torch.cuda.empty_cache()


def build(spec: ModelSpec, settings: dict[str, Scalar]) -> Bongard:
    if spec.code_repo is None or spec.code_revision is None:
        raise ValueError(f"{spec.name}: bongard needs a pinned code_repo")
    hint = gpu.download_hint(spec.repo, spec.revision, "bongard-mini")
    checkpoint = gpu.local_dir(settings, "bongard_dir", "bongard-mini", hint)
    fetch = f"scripts/fetch_code.sh {spec.code_repo} {spec.code_revision} bongard-code"
    code_dir = gpu.local_dir(settings, "bongard_code_dir", "bongard-code", fetch)
    gpu.check_git_commit(code_dir, spec.code_revision)
    device = gpu.cuda_device(settings, spec.name)
    ceilings = (
        profiling.measure_ceilings(str(device)) if get_bool(settings, "measure_ceilings", False) else {}
    )
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    predictor = importlib.import_module("bongard.inference").Predictor.load(
        checkpoint,
        device=device.type,  # upstream accepts only "cpu" | "mps" | "cuda"; cuda_device gave cuda:0
        temperatures=checkpoint / "temperatures.json",
        branch_batch_size=get_int(settings, "bongard_branch_batch", 8),
    )
    config = json.loads((Path(checkpoint) / "backbone" / "config.json").read_text())
    return Bongard(predictor, hf_config=config, ceilings=ceilings)


class BongardBackend:
    def build(self, spec: ModelSpec, settings: dict[str, Scalar]) -> Bongard:
        return build(spec, settings)
