"""The `d1` backend: Liquid AI's open d1 decision models (huggingface.co/LiquidAI/d1-3B), in-process.

Runs the model repo's own System One code (`modeling_d1.D1Model`, `runner.SystemOne` over `hybrid.py`'s
language model), like laya runs its shipped helper. The six code files are checked against the sha256 we
reviewed before `trust_remote_code` imports them: they make no network calls and read only the checkpoint
directory (tokenizer, processor). A question is rendered as `<|im_start|>user\\n<state>\\n\\nQUESTION:\\n
<question block><|im_end|>\\n<|im_start|>assistant\\n` and answered by a softmax over its options' tokens at
the last position (yes/no forms for a noul, option codes for a choice, digits for a score); nothing is decoded
and nothing is cached. The answers already have the /v1/systemone shape. LFM Open License v1.0.

A one-question request is one plain pass, `model(input_ids, logits_to_keep=1)` (`SystemOne._one_pass`). With
`d1_compile` that pass is `torch.compile(model.forward, mode="reduce-overhead")`, the card's CUDA-graph path:
each new prompt length compiles/captures on first use, so measure warm (the roofline's warm-up answers its
subset once, untimed). A request with several questions runs as one packed tree (`model.answer`: the state
once, the questions after it with no padding).

Roofline (benchmarks/roofline.py): both kinds of pass are timed with `ForwardRecorder.span` around the call,
so the eager and the graphed path are recorded the same way (hooks would miss a compiled forward). The pass
writes no cache (`cached` false) and keeps the logits at 1 position per question (the tied LM head runs
there). A tree is recorded as one row of trunk + all branch tokens, which overstates its attention slightly
(branches do not see each other). FlopCounterMode cannot see a graph replay, so `count_flops` is nan when
compiled. The SigLIP2 tower is loaded but never runs on a text state; the cost model reads the text tower.

Weights and code: `scripts/download_model.sh LiquidAI/d1-3B <revision> d1-3B` (or `s1 download` on Modal):
    model_dir         the repo                         default $MODELS_DIR/<repo name>
Settings (all optional):
    d1_compile        CUDA graphs via torch.compile     from the spec
    max_prompt_tokens longest pass (a tree: state + its longest question) before the answer is invalid
                      default 16384 (the model reads 32,768; 16384 matches openjev's max_model_len)
    measure_ceilings  measure the GPU's roofline ceilings before loading   default false
    dtype, require_gpu see backends/gpu.py
"""

from __future__ import annotations

import math
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ... import profiling
from ...record import Scalar
from ...registry import get_bool, get_int
from ..spec import ModelSpec
from . import gpu

# sha256 of the shipped System One code at d1-3B@05b80fa (reviewed 2026-11-12)
CODE_SHA256 = {
    "api.py": "94e9f82eabfdf92c4abe1153e9a6dc4dd9821f9cc2fec3221e79e70723d8ada0",
    "hybrid.py": "ce579827b952a6978a3560f70e74be9e228f9029de46048996b0c5f58d7f1f39",
    "lfm2_vl.py": "f9e8d3354572a169237865283594b0364209fdb858901121b9f99147328b5d95",
    "modeling_d1.py": "7fc6a001f80fc2d146f2a51128478f3b3bd4f1423a3ac129bc10f3fd0caf1d98",
    "prompt.py": "a20b5f52e41a6d29e694bb8c861a5a4803ce3eb69364edad79511ccf8edb5c31",
    "runner.py": "24d3ae718e9f9717771c0df39c93d324d00aa27bf3e12bab085c0922096cb0dd",
}


def check_code(model_dir: Path) -> None:
    """Refuse a checkout whose System One code is not the reviewed one."""
    for name, digest in CODE_SHA256.items():
        gpu.check_sha256(model_dir / name, digest)


def chain_shape(rows: int, tokens: int, keep: int) -> dict[str, object]:
    """A plain pass: `rows` x `tokens`, logits at `keep` positions per row (0 = all), no cache."""
    return {"rows": rows, "tokens": tokens, "past": 0, "kept": keep or tokens, "cached": False}


def tree_shape(trunk: int, branches: list[int]) -> dict[str, object]:
    """A packed tree: the trunk and every branch on one axis, logits at each branch's last token."""
    return {"rows": 1, "tokens": trunk + sum(branches), "past": 0, "kept": len(branches), "cached": False}


def check_length(tokens: int, max_tokens: int) -> None:
    if tokens > max_tokens:
        raise ValueError(f"{tokens} prompt tokens, over max_prompt_tokens={max_tokens}")


@dataclass
class D1:
    model: object  # modeling_d1.D1Model
    engine: object  # runner.SystemOne, model.engine
    max_tokens: int
    compiled: bool
    hf_config: dict[str, object] = field(
        default_factory=dict
    )  # the VLM config; the cost model reads text_config
    ceilings: dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    recorder: profiling.ForwardRecorder | None = None

    def install(self) -> None:
        """Route the engine's two kinds of pass through the length guard and the recorder."""
        chain: Callable[..., object] = self.engine._one_pass  # ty: ignore[unresolved-attribute]
        tree: Callable[..., object] = self.model.answer  # ty: ignore[unresolved-attribute]

        def one_pass(**kwargs: object) -> object:
            rows, tokens = kwargs["input_ids"].shape  # ty: ignore[unresolved-attribute]
            check_length(int(tokens), self.max_tokens)
            keep = kwargs.get("logits_to_keep", 0)
            return self._timed(chain_shape(int(rows), int(tokens), int(keep)), lambda: chain(**kwargs))  # ty: ignore[invalid-argument-type]

        def answer(trunk: object, questions: object, lengths: object, **vision: object) -> object:
            p, branches = int(trunk.shape[1]), [int(n) for n in lengths.tolist()]  # ty: ignore[unresolved-attribute]
            check_length(p + max(branches), self.max_tokens)
            return self._timed(tree_shape(p, branches), lambda: tree(trunk, questions, lengths, **vision))

        self.engine._one_pass = one_pass  # ty: ignore[unresolved-attribute]
        self.model.answer = answer  # ty: ignore[unresolved-attribute]

    def _timed(self, shape: dict[str, object], run: Callable[[], object]) -> object:
        if self.recorder is None:
            return run()
        with self.recorder.span(shape):
            return run()

    def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
        with self.lock:  # one engine, and CUDA-graph outputs are reused by the next replay
            out = self.engine.system_one(state, dict(questions))  # ty: ignore[unresolved-attribute]
        return {str(k): v for k, v in out["answers"].items()}

    def profile_info(self) -> dict[str, object]:
        if self.recorder is None:
            self.recorder = profiling.ForwardRecorder(self.model)  # its hooks stay off: every pass is a span
        return {"config": self.hf_config, "head_bytes": 2, "ceilings": self.ceilings}

    def take_forwards(self) -> list[dict[str, object]]:
        return [] if self.recorder is None else self.recorder.take()

    def count_flops(self, state: str, questions: Mapping[str, object]) -> float:
        if self.compiled:
            return math.nan
        return profiling.count_flops(lambda: self.decide(state, questions))[1]

    def profiler(self, on: bool) -> None:
        """Open/close the CUDA profiler range Nsight Compute records (`s1 ncu`)."""
        import torch  # noqa: PLC0415

        torch.cuda.synchronize()
        (torch.cuda.profiler.start if on else torch.cuda.profiler.stop)()

    def run_info(self) -> dict[str, Scalar]:
        env: dict[str, Scalar] = {f"env_{k}": v for k, v in profiling.kernel_env(self.model).items()}
        env["env_d1_compiled"] = self.compiled
        return env

    def close(self) -> None:
        """Drop the weights and any compiled graphs, so the next run in this process can load."""
        import gc  # noqa: PLC0415

        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
        self.engine = self.model = None
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            if self.compiled:
                torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()


def build(spec: ModelSpec, settings: dict[str, Scalar]) -> D1:
    name = spec.repo.rpartition("/")[2]
    model_dir = gpu.local_dir(settings, "model_dir", name, gpu.download_hint(spec.repo, spec.revision, name))
    check_code(model_dir)
    device = gpu.cuda_device(settings, spec.name)
    ceilings = (
        profiling.measure_ceilings(str(device)) if get_bool(settings, "measure_ceilings", False) else {}
    )

    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    model = transformers.AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, dtype=gpu.torch_dtype(settings)
    )
    model = model.to(device).eval()
    engine = model.engine  # SystemOne(model=self, tokenizer from model_dir), compile=False
    compiled = get_bool(settings, "d1_compile", False)
    if compiled:  # what SystemOne(compile=True) does
        engine._one_pass = torch.compile(model.forward, mode="reduce-overhead")
    d1 = D1(
        model,
        engine,
        get_int(settings, "max_prompt_tokens", 16384),
        compiled,
        hf_config=dict(model.config.to_dict()),
        ceilings=ceilings,
    )
    d1.install()
    print(f"{spec.name}: {type(model).__name__} on {device}, compiled={compiled}", flush=True)
    return d1


class D1Backend:
    def build(self, spec: ModelSpec, settings: dict[str, Scalar]) -> D1:
        return build(spec, settings)
