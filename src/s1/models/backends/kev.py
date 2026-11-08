"""The `kev` backend: Kev-4B (github.com/jaredpalmer/kev), a LoRA adapter plus a pointer head on a Qwen base.

Runs upstream's own serving stack in-process, from a checkout of the catalog's code pin (checked before
import): `kev.serve.Server` over `Checkpoint.load` with the options `kev.serve` picks on CUDA. bf16 with the
LoRA merged, and for the hybrid (Qwen3.5) base fused Triton kernels (kev.fused_qwen35) and CUDA graphs
(kev.cuda_graphs). The server's model thread batches whatever requests are waiting, so `jeval_concurrency` > 1
measures batched throughput. Each question is one causal row, `<state> state <q> instr (<opt> option </opt>)*
<decide>`; the pointer head scores each `</opt>` against `<decide>`, divided by the temperature stored in
head.pt. Upstream reads the base by Hub name; here the spec's base is read from its local copy instead.
Apache-2.0.

CUDA graphs are captured lazily (a new shape first runs eagerly), so measure warm: `jeval_warmup = true`.
`after_warmup()` then captures what is pending and empties the state-prefix cache, so no state carries over.

Roofline (benchmarks/roofline.py): eager passes are recorded by hooks on the backbone (no vocab head runs: the
pointer head reads hidden states, so kept = 0); graphed passes never call the module, so `CudaGraphs._replay`
is wrapped to time each pass, replayed or eager, at its bucket's padded shape, with the real token count
beside it. Graph captures run unrecorded. FlopCounterMode cannot see the model thread or a graph replay, so
`count_flops` is nan; `s1 ncu` counts the kernels.

Weights and code:
    kev_dir          adapter + head.pt     default $MODELS_DIR/kev-4b-<variant>    (scripts/download_model.sh)
    base_dir         base model            default $MODELS_DIR/<base repo name>    (scripts/download_model.sh)
    kev_code_dir     upstream code         default $MODELS_DIR/kev-code            (scripts/fetch_code.sh)
Settings (all optional):
    max_prompt_tokens longest row (state + one question) before the answer is invalid   default 16384
                      (upstream serves rows up to 73,728 tokens; 16384 matches openjev's max_model_len)
    kev_fused         fused Qwen3.5 kernels (hybrid base only)                          default true
    kev_cuda_graphs   CUDA graphs (hybrid base only)                                    default true
    measure_ceilings  measure the GPU's roofline ceilings before loading                default false
    require_gpu       see backends/gpu.py
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import sys
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field

from ... import profiling
from ...record import Scalar
from ...registry import get_bool, get_int
from ..spec import ModelSpec
from . import gpu


@dataclass
class Kev:
    server: object  # kev.serve.Server
    serve: object  # the kev.serve module
    max_tokens: int
    hf_config: dict[str, object] = field(default_factory=dict)  # the backbone's, for the roofline cost model
    ceilings: dict[str, float] = field(default_factory=dict)
    recorder: profiling.ForwardRecorder | None = None

    def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
        serve, server = self.serve, self.server
        request = serve.SystemOneRequest  # ty: ignore[unresolved-attribute]
        req = request(state=state, questions=dict(questions))
        rec, meta = serve.to_record(serve.prepare(req))  # ty: ignore[unresolved-attribute]
        # Server.submit, encoded at our limit instead of the 64k serving limits (strict: no truncation)
        enc = server.model.encode(  # ty: ignore[unresolved-attribute]
            server.tok,  # ty: ignore[unresolved-attribute]
            rec,
            max_state=self.max_tokens,
            max_branch=self.max_tokens,
            strict=True,
        )
        done: Future[tuple[list[list[float]], dict[str, object]]] = Future()
        server.queue.put((enc, done))  # ty: ignore[unresolved-attribute]
        probs, _ = done.result()
        return dict(serve.to_answers(probs, meta))  # ty: ignore[unresolved-attribute]

    def after_warmup(self) -> None:
        """Capture the CUDA graphs warm-up left pending, then forget every cached state."""
        server = self.server
        server.wait_idle()  # ty: ignore[unresolved-attribute]
        with server.lock:  # ty: ignore[unresolved-attribute]
            server.prefix_cache.clear()  # ty: ignore[unresolved-attribute]
        if (graphs := getattr(server.model, "graphs", None)) is not None:  # ty: ignore[unresolved-attribute]
            print(f"kev graphs after warm-up: {graphs.stats()}", flush=True)

    def profile_info(self) -> dict[str, object]:
        return {"config": self.hf_config, "head_bytes": 2, "ceilings": self.ceilings}

    def take_forwards(self) -> list[dict[str, object]]:
        return [] if self.recorder is None else self.recorder.take()

    def count_flops(self, state: str, questions: Mapping[str, object]) -> float:
        return math.nan

    def profiler(self, on: bool) -> None:
        """Open/close the CUDA profiler range Nsight Compute records (`s1 ncu`), the model thread idle."""
        import torch  # noqa: PLC0415

        self.server.wait_idle()  # ty: ignore[unresolved-attribute]
        torch.cuda.synchronize()
        (torch.cuda.profiler.start if on else torch.cuda.profiler.stop)()

    def run_info(self) -> dict[str, Scalar]:
        model = self.server.model  # ty: ignore[unresolved-attribute]
        env: dict[str, Scalar] = {f"env_{k}": v for k, v in profiling.kernel_env(model.lm).items()}
        graphs = getattr(model, "graphs", None)
        env["env_kev_graphs"] = graphs is not None
        if graphs is not None:
            env |= {f"env_kev_graphs_{k}": int(v) for k, v in graphs.stats().items()}
        return env

    def close(self) -> None:
        """Stop the model thread and free the GPU, so the next run in this process can load.

        Server.__post_init__ registers close() with atexit; that reference alone would keep the weights and
        the CUDA graph buffers alive."""
        import atexit  # noqa: PLC0415
        import gc  # noqa: PLC0415

        server = self.server
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
        server.close()  # ty: ignore[unresolved-attribute]
        atexit.unregister(server.close)  # ty: ignore[unresolved-attribute]
        server.prefix_cache.clear()  # ty: ignore[unresolved-attribute]
        server.model = server.tok = None  # ty: ignore[unresolved-attribute]
        gc.collect()
        if "torch" in sys.modules:
            sys.modules["torch"].cuda.empty_cache()


def build(spec: ModelSpec, settings: dict[str, Scalar]) -> Kev:
    if spec.base_repo is None or spec.code_repo is None or spec.code_revision is None:
        raise ValueError(f"{spec.name}: kev needs a base_repo and a pinned code_repo")
    variant = spec.name.partition(".")[2]
    name = f"kev-4b-{variant}"
    kev_dir = gpu.local_dir(settings, "kev_dir", name, gpu.download_hint(spec.repo, spec.revision, name))
    base_name = spec.base_repo.rpartition("/")[2]
    base_dir = gpu.local_dir(
        settings, "base_dir", base_name, gpu.download_hint(spec.base_repo, spec.base_revision, base_name)
    )
    fetch = f"scripts/fetch_code.sh {spec.code_repo} {spec.code_revision} kev-code"
    code_dir = gpu.local_dir(settings, "kev_code_dir", "kev-code", fetch)
    gpu.check_git_commit(code_dir, spec.code_revision)
    device = gpu.cuda_device(settings, spec.name)
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    import torch  # noqa: PLC0415

    checkpoint = importlib.import_module("kev.checkpoint")
    serve = importlib.import_module("kev.serve")
    ck = checkpoint.Checkpoint(str(kev_dir))
    if ck.meta.base != spec.base_repo:
        raise ValueError(f"{kev_dir}/head.pt is on {ck.meta.base}, the catalog says {spec.base_repo}")
    ck.meta = dataclasses.replace(
        ck.meta, base=str(base_dir), base_revision=None
    )  # the local copy of the pin
    fused = get_bool(settings, "kev_fused", True)
    if fused and not checkpoint.fused_available():
        raise RuntimeError("kev_fused needs flash-linear-attention at kev.fused_qwen35.FLA_VERSION")
    opts = checkpoint.LoadOptions(  # kev.serve's CUDA defaults
        dtype=torch.bfloat16,
        backend="torch",
        cuda_graphs=get_bool(settings, "kev_cuda_graphs", True),
        fused=fused,
    )
    ceilings = (
        profiling.measure_ceilings(str(device)) if get_bool(settings, "measure_ceilings", False) else {}
    )
    tok, model = ck.load(str(device), opts)
    recorder = profiling.ForwardRecorder(model.lm, head=False)
    if (graphs := getattr(model, "graphs", None)) is not None:
        record_graphs(graphs, recorder)
    print(
        f"{spec.name}: {'hybrid' if model.hybrid else 'attention-only'} base, temperature "
        f"{model.head.temperature}, fused={fused and model.hybrid}, "
        f"graphs={getattr(model, 'graphs', None) is not None}",
        flush=True,
    )
    server = serve.Server(ck, tok, model, str(device), release_date="local")
    return Kev(
        server,
        serve,
        get_int(settings, "max_prompt_tokens", 16384),
        hf_config=dict(model.lm.config.to_dict()),
        ceilings=ceilings,
        recorder=recorder,
    )


def graph_pass(key: tuple[object, ...], rows: list[list[int]]) -> dict[str, object]:
    """The forward a kev.cuda_graphs pass runs: its bucket's padded shape (`key`), and the real tokens in the
    uploaded `rows`. State pass ("states", Nb, Sb): rows [ids | positions | length], new states into the bank.
    Row pass ("rows", Nb, Lb, Sr): rows [ids | positions | row length | state length | entry], each row
    attending to Sr state positions."""
    if key[0] == "states":
        _, n, s = key
        return {
            "rows": n,
            "tokens": s,
            "past": 0,
            "kept": 0,
            "cached": True,
            "real_tokens": sum(r[-1] for r in rows),
        }
    if key[0] == "rows":
        _, n, length, past = key
        real = sum(r[2 * int(length)] for r in rows)  # ty: ignore[invalid-argument-type]
        return {"rows": n, "tokens": length, "past": past, "kept": 0, "cached": True, "real_tokens": real}
    raise ValueError(f"unknown kev.cuda_graphs pass {key!r}")


def record_graphs(graphs: object, recorder: profiling.ForwardRecorder) -> None:
    """Time every graphed pass (replay or eager) as one forward, and keep captures out of the record."""
    replay: Callable[..., None] = graphs._replay  # ty: ignore[unresolved-attribute]
    capture: Callable[..., None] = graphs.capture_pending  # ty: ignore[unresolved-attribute]

    def timed_replay(key: tuple[object, ...], body: object, rows: list[list[int]]) -> None:
        with recorder.span(graph_pass(key, rows)):
            replay(key, body, rows)

    def quiet_capture(*args: object, **kwargs: object) -> None:
        with recorder.suspended():
            capture(*args, **kwargs)

    graphs._replay = timed_replay  # ty: ignore[unresolved-attribute]
    graphs.capture_pending = quiet_capture  # ty: ignore[unresolved-attribute]


class KevBackend:
    def build(self, spec: ModelSpec, settings: dict[str, Scalar]) -> Kev:
        return build(spec, settings)
