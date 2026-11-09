"""The `laya` backend: Laya (huggingface.co/convaiinnovations/laya), an encoder with a decision head.

Runs the model repo's own inference code, `rl_agent_api.RLAgent` (with `rl_common.py`), like openjev runs its
shipped helper. Both files are checked against the sha256 we reviewed before import: they read only the
checkpoint directory and make no network calls when given a local encoder directory, which RLAgent always
does. Each question is one encoder row, `[CLS] <type> question: <instructions> [SEP] ([MASK] option)* [SEP]
state [SEP]`, cut to the checkpoint's `max_len` (512 English, 1024 multilingual) by dropping the end of the
state. The head scores each option's [MASK]; temperatures come from the checkpoint's rl_agent_config.json, and
the probabilities are rounded to 4 decimals, as upstream. All of a request's questions run in one batched
pass. Apache-2.0.

Weights: `scripts/download_model.sh convaiinnovations/laya <revision> laya` holds both checkpoints:
    laya_dir   the repo                         default $MODELS_DIR/laya
    checkpoint subdirectory of the checkpoint    from the spec ("" English, "multilingual")
Settings: require_gpu (see backends/gpu.py). Weights stay fp32 under bf16 autocast, as upstream.
measure_ceilings (default false) measures the GPU's roofline ceilings before loading.

Roofline (benchmarks/roofline.py): the encoder's forwards are recorded by hooks on `model.encoder` (the
ModernBERT backbone; the decision head is a small MLP on the option markers, no vocab head, so kept = 0),
and its weights are fp32 in memory (weight_bytes = 4).

Batch sweep (benchmarks/batch_sweep.py): `decide_batch` encodes each (state, question) with upstream's
`rl_common.build_sequence`, collates them with `collate_items` (one row each, right-padded, as `system_one`
batches a request's questions) and runs the whole DecisionModel once under the same autocast, timed as one
forward (`recorder.span`: encoder and the 2-layer head together; the cost model charges the encoder only). The
scores are copied to the host; the answer assembly (softmax, rounding) is skipped. Settings:
    laya_mode         eager | graphs | compile | compile_graphs                          default eager
                      graphs: one torch.cuda.graph per (rows, width, options) shape, captured on first use
                      (unrecorded) and replayed; compile: torch.compile(dynamic=True), inductor's fused
                      kernels, no graphs; compile_graphs: the same with mode="reduce-overhead" (CUDA graphs).
                      Only decide_batch uses the mode; decide (jeval, roofline) stays upstream's eager path.
    laya_pad_multiple pad each batch's width up to a multiple of this (1 = none)        default 1
                      (bounds the number of graph shapes; the padded width is what the cost model charges)
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ... import profiling
from ...record import Scalar
from ...registry import get_bool, get_int, get_str
from ..spec import ModelSpec
from . import gpu

# sha256 of the shipped inference code at the catalog revision (reviewed 2026-10-30)
CODE_SHA256 = {
    "rl_agent_api.py": "be3b46819c9999c3ef88e0f2ecf6d3ab1cdfed1d9a8b466fc89811e34d44031b",
    "rl_common.py": "8d83611d480c971d640a7b7d3aa2f2219c5e8455e9cc2329fd073681bd8be23e",
}


def import_reviewed(code_dir: Path) -> type:
    """RLAgent from `code_dir`, after checking both files are the reviewed ones."""
    for name, digest in CODE_SHA256.items():
        gpu.check_sha256(code_dir / name, digest)
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    agent = importlib.import_module("rl_agent_api").RLAgent
    if not isinstance(agent, type):
        raise TypeError("rl_agent_api.RLAgent is not a class")
    return agent


MODES = ("eager", "graphs", "compile", "compile_graphs")


def padded_width(width: int, multiple: int) -> int:
    """`width` rounded up to a multiple of `multiple` (>= 1)."""
    if multiple < 1:
        raise ValueError(f"laya_pad_multiple must be >= 1, got {multiple}")
    return -(-width // multiple) * multiple


def graph_key(inputs: Sequence[object]) -> tuple[int, int, int]:
    """A CUDA graph's shape: (rows, padded width, option markers) of decide_batch's inputs."""
    ids, _, markers = inputs[:3]
    return (int(ids.shape[0]), int(ids.shape[1]), int(markers.shape[1]))  # ty: ignore[unresolved-attribute]


@dataclass
class Laya:
    agent: object
    max_state_tokens: int | None = None  # shape_sweep skips longer states
    lock: threading.Lock = field(default_factory=threading.Lock)
    ceilings: dict[str, float] = field(default_factory=dict)
    recorder: profiling.ForwardRecorder | None = None
    mode: str = "eager"
    pad_multiple: int = 1
    _compiled: object = None
    _graphs: dict[tuple[int, int, int], tuple[object, list[object], object]] = field(default_factory=dict)
    _pool: object = None

    def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
        with self.lock:
            out = self.agent.system_one(state, dict(questions))  # ty: ignore[unresolved-attribute]
        answers = out["answers"]
        return {str(k): v for k, v in answers.items()}

    def _encoder(self) -> object:
        return self.agent.model.encoder  # ty: ignore[unresolved-attribute]

    def profile_info(self) -> dict[str, object]:
        if self.recorder is None:
            self.recorder = profiling.ForwardRecorder(self._encoder(), head=False)
            if self.mode != "eager":  # decide_batch times with spans; hooks must not reach compile/capture
                self.recorder.close()
        config = dict(self._encoder().config.to_dict())  # ty: ignore[unresolved-attribute]
        return {"config": config, "head_bytes": 2, "weight_bytes": 4, "ceilings": self.ceilings}

    def take_forwards(self) -> list[dict[str, object]]:
        return [] if self.recorder is None else self.recorder.take()

    def count_flops(self, state: str, questions: Mapping[str, object]) -> float:
        return profiling.count_flops(lambda: self.decide(state, questions))[1]

    def profiler(self, on: bool) -> None:
        import torch  # noqa: PLC0415

        torch.cuda.synchronize()
        (torch.cuda.profiler.start if on else torch.cuda.profiler.stop)()

    def run_info(self) -> dict[str, Scalar]:
        env: dict[str, Scalar] = {f"env_{k}": v for k, v in profiling.kernel_env(self._encoder()).items()}
        env["env_laya_mode"] = self.mode
        env["env_laya_graphs"] = len(self._graphs)
        return env

    def _items(self, requests: Sequence[tuple[str, Mapping[str, object]]]) -> list[dict[str, object]]:
        """One upstream item per (state, question), as system_one builds them."""
        rl = sys.modules["rl_common"]  # imported by rl_agent_api
        agent = self.agent
        cfg = agent.cfg  # ty: ignore[unresolved-attribute]
        items: list[dict[str, object]] = []
        for state, questions in requests:
            for qid, qdef in questions.items():
                q = agent._to_internal(qdef)  # ty: ignore[unresolved-attribute]
                seq, markers = rl.build_sequence(agent.tok, state, q, cfg["max_len"], cfg["head_max_len"])  # ty: ignore[unresolved-attribute]
                if len(markers) != len(rl.render_options(q)):
                    raise ValueError(f"question {qid!r}: options do not fit in head_max_len")
                items.append(
                    {
                        "ids": seq,
                        "markers": markers,
                        "qtype": rl.QTYPES[q["t"]],
                        "target": [0.0] * len(markers),
                        "label": -1,
                        "episode": 0,
                        "ep_step": 0,
                        "ep_len": 1,
                        "src": "api",
                    }
                )
        return items

    def _fn(self) -> Callable[..., object]:
        """The forward decide_batch runs: the DecisionModel under upstream's autocast, compiled per mode."""
        import torch  # noqa: PLC0415

        model, agent = self.agent.model, self.agent  # ty: ignore[unresolved-attribute]
        dtype = agent.dtype  # ty: ignore[unresolved-attribute]

        def fwd(*xs: object) -> object:
            with torch.autocast(device_type="cuda", dtype=dtype, cache_enabled=self.mode == "eager"):
                return model(*xs)

        if self.mode in ("compile", "compile_graphs"):
            if self._compiled is None:
                dyn = torch._dynamo.config
                for knob in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit"):
                    if hasattr(dyn, knob):
                        setattr(dyn, knob, 512)
                mode = "reduce-overhead" if self.mode == "compile_graphs" else "default"
                self._compiled = torch.compile(fwd, mode=mode, dynamic=True)
            return self._compiled  # ty: ignore[invalid-return-type]
        return fwd

    def _replay(self, inputs: list[object]) -> object:
        """Run `inputs` through this shape's CUDA graph, capturing it (unrecorded) the first time."""
        import torch  # noqa: PLC0415

        key = graph_key(inputs)
        if key not in self._graphs:
            fwd = self._fn()
            static = [x.clone() for x in inputs]  # ty: ignore[unresolved-attribute]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):  # warm on a side stream, as torch.cuda.graphs asks
                    fwd(*static)
            torch.cuda.current_stream().wait_stream(stream)
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                out = fwd(*static)
            self._graphs[key] = (graph, static, out)
        graph, static, out = self._graphs[key]
        for dst, src in zip(static, inputs, strict=True):
            dst.copy_(src)  # ty: ignore[unresolved-attribute]
        graph.replay()  # ty: ignore[unresolved-attribute]
        return out

    def decide_batch(self, requests: Sequence[tuple[str, Mapping[str, object]]]) -> dict[str, int]:
        """Every request's questions as rows of one pass (benchmarks/batch_sweep.py); scores to the host."""
        import torch  # noqa: PLC0415

        rl = sys.modules["rl_common"]
        agent = self.agent
        device = agent.device  # ty: ignore[unresolved-attribute]
        with self.lock, torch.no_grad():
            items = self._items(requests)
            b = rl.collate_items([items], agent.tok.pad_token_id)  # ty: ignore[unresolved-attribute]
            ids, att = b["input_ids"], b["attention_mask"]
            rows, width = ids.shape
            pad = padded_width(width, self.pad_multiple) - width
            if pad:
                ids = torch.nn.functional.pad(ids, (0, pad), value=agent.tok.pad_token_id)  # ty: ignore[unresolved-attribute]
                att = torch.nn.functional.pad(att, (0, pad), value=0)
            inputs = [
                t.to(device, non_blocking=True)
                for t in (ids, att, b["marker_pos"], b["marker_mask"], b["qtype"])
            ]
            shape = {
                "rows": int(rows),
                "tokens": int(ids.shape[1]),
                "past": 0,
                "kept": 0,
                "cached": False,
                "real_tokens": int(b["attention_mask"].sum()),
            }
            recorder = self.recorder
            span = recorder.span(shape) if recorder is not None else contextlib.nullcontext()
            if self.mode == "compile_graphs":
                torch.compiler.cudagraph_mark_step_begin()
            if self.mode == "graphs" and graph_key(inputs) not in self._graphs:
                self._replay(inputs)  # capture before the span, so its events time only the replay
            with span:
                if self.mode == "graphs":
                    logits, act = self._replay(inputs)  # ty: ignore[not-iterable]
                else:
                    logits, act = self._fn()(*inputs)  # ty: ignore[not-iterable]
            logits.float().cpu()
            torch.softmax(act.float(), -1).cpu()
        return {"decisions": len(items), "real_tokens": shape["real_tokens"]}


def build(spec: ModelSpec, settings: dict[str, Scalar]) -> Laya:
    code_dir = gpu.local_dir(
        settings, "laya_dir", "laya", gpu.download_hint(spec.repo, spec.revision, "laya")
    )
    checkpoint = code_dir / get_str(settings, "checkpoint", "")
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"{checkpoint}/model.safetensors missing")
    device = gpu.cuda_device(settings, spec.name)
    ceilings = (
        profiling.measure_ceilings(str(device)) if get_bool(settings, "measure_ceilings", False) else {}
    )
    agent = import_reviewed(code_dir)(str(checkpoint), device=str(device))
    mode = get_str(settings, "laya_mode", "eager")
    if mode not in MODES:
        raise ValueError(f"laya_mode must be one of {MODES}, got {mode!r}")
    pad = get_int(settings, "laya_pad_multiple", 1)
    padded_width(1, pad)  # validates
    return Laya(agent, spec.max_state_tokens, ceilings=ceilings, mode=mode, pad_multiple=pad)


class LayaBackend:
    def build(self, spec: ModelSpec, settings: dict[str, Scalar]) -> Laya:
        return build(spec, settings)
