"""The `bongard` backend: Bongard-mini (huggingface.co/AgentBull/bongard-mini), a T5Gemma 2 encoder-decoder
with a judgment head.

Runs upstream's runtime, `bongard.inference.Predictor`, from a checkout of github.com/AgentBull/bongard at the
catalog's code pin (checked before import; the inference path loads only local files). The encoder reads the
JSON-encoded state plus every question's instructions once per request; the decoder reads each question with
its candidates, and the head scores each candidate's end marker against the decision marker. Temperatures come
from the checkpoint's temperatures.json. Requests over 32,768 tokens (state + one question) are rejected, not
truncated, so they count as invalid. Gemma terms.

Only for the 32 GB RTX Pro 4500 (`gpu-g7-reserved`): the spec sets `require_gpu`. 15 GB of bf16 weights would
leave too little room on a 24 GB card for long states.

Weights and code:
    bongard_dir       checkpoint     default $MODELS_DIR/bongard-mini    (scripts/download_model.sh)
    bongard_code_dir  runtime code   default $MODELS_DIR/bongard-code    (scripts/fetch_code.sh)
Settings (all optional):
    bongard_branch_batch   questions decoded per pass    default 8 (upstream)
    require_gpu            see backends/gpu.py           from the spec
"""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

from ...record import Scalar
from ...registry import get_int
from ..spec import ModelSpec
from . import gpu


@dataclass
class Bongard:
    predictor: object
    lock: threading.Lock = field(default_factory=threading.Lock)

    def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
        with self.lock:
            out = self.predictor.predict({"state": state, "questions": dict(questions)})  # ty: ignore[unresolved-attribute]
        return {str(k): v for k, v in out["answers"].items()}


def build(spec: ModelSpec, settings: dict[str, Scalar]) -> Bongard:
    if spec.code_repo is None or spec.code_revision is None:
        raise ValueError(f"{spec.name}: bongard needs a pinned code_repo")
    hint = gpu.download_hint(spec.repo, spec.revision, "bongard-mini")
    checkpoint = gpu.local_dir(settings, "bongard_dir", "bongard-mini", hint)
    fetch = f"scripts/fetch_code.sh {spec.code_repo} {spec.code_revision} bongard-code"
    code_dir = gpu.local_dir(settings, "bongard_code_dir", "bongard-code", fetch)
    gpu.check_git_commit(code_dir, spec.code_revision)
    device = gpu.cuda_device(settings, spec.name)
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    predictor = importlib.import_module("bongard.inference").Predictor.load(
        checkpoint,
        device=str(device),
        temperatures=checkpoint / "temperatures.json",
        branch_batch_size=get_int(settings, "bongard_branch_batch", 8),
    )
    return Bongard(predictor)


class BongardBackend:
    def build(self, spec: ModelSpec, settings: dict[str, Scalar]) -> Bongard:
        return build(spec, settings)
