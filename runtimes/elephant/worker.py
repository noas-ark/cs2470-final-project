"""elephant-1 decision worker: runs in runtimes/elephant/.venv (Python 3.11, torch 2.8), driven by the
harness's `elephant` backend over stdin/stdout, one JSON object per line.

Serves the model the way elephant-1's engine.py does (OneJev qev @81ce62f + serve.patch): qev's
DecisionEngine, with the checkpoint's readout.json / config overrides (block-bidirectional full attention,
slot_lm readout), one shared-prefix prefill per request and one batched forward over the question suffixes,
and the sequential fork as the fallback when the batched fork overflows 32-bit indexing. Every forward of
the underlying HF model is recorded with s1.profiling (imported from the harness's src/, 3.11-compatible).

Startup (ceilings first, before the weights take the memory), then one line:
    {"ready": true, "config": {...}, "head_bytes": 4, "ceilings": {...}, "env": {...}}
Requests -> responses:
    {"op": "decide", "state": ..., "questions": {...}, "count": false}
        -> {"answers": {...}, "flops": <counted or null>, "forwards": [...]}
        or {"error": "...", "forwards": [...]}
    {"op": "forwards"} -> {"forwards": [...]}
    {"op": "profiler", "on": true} -> {"forwards": []}   (cudaProfilerStart/Stop, for Nsight Compute)
    {"op": "quit"}
Library output goes to stderr; stdout carries only the protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, TextIO

FIELDS = {
    "choice": {"type", "choice", "probabilities"},
    "noul": {"type", "noul"},
    "score": {"type", "score", "probabilities"},
}


def staged(engine_cls: type) -> type:
    """qev's DecisionEngine with the fp32 head upcast staged through host memory. Upstream clones and casts
    on the GPU, which briefly holds the bf16 head, its clone and the fp32 copy next to ~18.6 GiB of weights:
    over an L4's 22 GiB. Casting on the CPU gives the same fp32 weight and hook, and peaks at the final
    size."""

    class Staged(engine_cls):
        def _upcast_head(self, head_dtype: object) -> None:
            head = self.model.get_output_embeddings()
            device = head.weight.device if head is not None and hasattr(head, "weight") else None
            if device is not None:
                head.to("cpu")
            super()._upcast_head(head_dtype)
            if device is not None:
                head.to(device)

    return Staged


class Worker:
    def __init__(self, args: argparse.Namespace, out: TextIO) -> None:
        sys.path.insert(0, args.harness_src)
        from qev.engine import DecisionEngine  # noqa: PLC0415

        from s1 import profiling  # noqa: PLC0415

        self.out, self.profiling = out, profiling
        ceilings = profiling.measure_ceilings(args.device) if args.measure_ceilings else {}
        self.engine = staged(DecisionEngine)(args.model_dir, device=args.device, head_dtype=args.head_dtype)
        self.recorder = profiling.ForwardRecorder(self.engine.model)
        info = self.engine.info
        env = profiling.kernel_env(self.engine.model) | {
            "fork_mode": info.fork_mode,
            "attention": str(info.extra.get("attention")),
            "readout": str(info.extra.get("readout")),
            "head_dtype": args.head_dtype,
        }
        self.send(
            {
                "ready": True,
                "config": self.engine.model.config.to_dict(),
                "head_bytes": 4 if args.head_dtype == "float32" else 2,
                "ceilings": ceilings,
                "env": env,
            }
        )

    def send(self, msg: dict[str, object]) -> None:
        self.out.write(json.dumps(msg) + "\n")
        self.out.flush()

    def decide(self, state: object, questions: dict[str, object]) -> dict[str, object]:
        from qev.schema import SystemOneRequest  # noqa: PLC0415

        engine = self.engine
        request = SystemOneRequest(state=state, questions=questions)
        try:
            resp, _ = engine.decide(request)
        except RuntimeError as e:
            if "32BitIndexMath" not in str(e):
                raise
            mode, engine.info.fork_mode = engine.info.fork_mode, "sequential"
            try:
                resp, _ = engine.decide(request)
            finally:
                engine.info.fork_mode = mode
        return {k: a.model_dump(include=FIELDS[a.type]) for k, a in resp.answers.items()}

    def handle(self, req: dict[str, Any]) -> None:
        if req.get("op") == "forwards":
            self.send({"forwards": self.recorder.take()})
            return
        if req.get("op") == "profiler":
            import torch  # noqa: PLC0415

            torch.cuda.synchronize()
            (torch.cuda.profiler.start if req["on"] else torch.cuda.profiler.stop)()
            self.send({"forwards": []})
            return
        try:
            if req.get("count"):
                answers, flops = self.profiling.count_flops(
                    lambda: self.decide(req["state"], req["questions"])
                )
            else:
                answers, flops = self.decide(req["state"], req["questions"]), None
            self.send({"answers": answers, "flops": flops, "forwards": self.recorder.take()})
        except Exception as e:  # reported to the harness, which counts the answer invalid
            traceback.print_exc()
            self.send({"error": f"{type(e).__name__}: {e}", "forwards": self.recorder.take()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--head-dtype", default="float32")  # upstream DecisionEngine default
    ap.add_argument("--measure-ceilings", action="store_true")
    ap.add_argument("--harness-src", default=str(Path(__file__).resolve().parents[2] / "src"))
    args = ap.parse_args()

    out = sys.stdout
    sys.stdout = sys.stderr  # anything a library prints must not corrupt the protocol
    worker = Worker(args, out)
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("op") == "quit":
            break
        worker.handle(req)


if __name__ == "__main__":
    main()
