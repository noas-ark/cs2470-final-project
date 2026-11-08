"""The `semif` backend: SemIf (github.com/TheoLeeCJ/SemIf), generic Qwen models read out as decision models.

A control: no decision training, just a readout over off-the-shelf weights. Reimplements the catalog's code
pin (src/semif_phase1/{core,direct,reranker}.py; the question mapping is benchmarks/build_typesafe.py):

- `mode = "direct"` (Qwen3.5-4B): one chat prompt per question, the user turn a JSON object of evidence,
  criterion and lettered options, thinking disabled. The answer is the softmax over the option letters' logits
  at the first generated position.
- `mode = "reranker"` (Qwen3-Reranker-4B): one raw prompt per (question, option) pair, all of a question's
  pairs in one left-padded batch. Each option scores logit("yes") - logit("no"), and the answer is the softmax
  of those log-odds across options (SemIf's rule, not the reranker's).

Options are "<key>: <description>", noul is (true, false) in that order. Upstream has no score questions; here
a score level is the option "<i>: <level text>" (our extension). Upstream caps a row at 16 options (letters
A-P) and rejects over-long prompts rather than truncating; both stay errors, so they count as invalid answers.
Code MIT; weights Apache-2.0.

The same readout serves any causal LM with a chat template, so the catalog also uses it for zero-shot
controls of other architectures (Qwen3, OLMoE, LFM2, Granite-4.0-H) and for JevK5, which was trained on
SemIf's prompt and reads out with a fitted temperature.

For the roofline benchmark the model is profiled (`profile_info`, `take_forwards`, `count_flops`): a forward
recorder is hooked on first use, and with `measure_ceilings` the GPU's ceilings are measured before the
weights load. `close()` frees the GPU so the next model of a sweep can load in the same process.

Weights (scripts/download_model.sh at the catalog revision): `model_dir`, default $MODELS_DIR/<repo name>.
Settings (all optional):
    mode               direct | reranker                                          (from the spec)
    semif_max_options  options per question, at most 26 (A-Z)                    default 16 (upstream)
    semif_temperature  divides the option scores before the softmax               default 1.0 (upstream)
    max_prompt_tokens  longest prompt (one pair, for the reranker)                default 16384
                       (upstream defaults to 4096; 16384 matches openjev's max_model_len)
    measure_ceilings   measure the GPU's roofline ceilings before loading         default false
    dtype, require_gpu see backends/gpu.py
"""

from __future__ import annotations

import inspect
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ... import profiling
from ...record import Scalar
from ...registry import get_bool, get_float, get_int, get_str
from ..answers import choice_answer, noul_from_logits, score_answer
from ..spec import ModelSpec
from . import gpu

if TYPE_CHECKING:
    import torch

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # upstream stops at P (16)
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DECISION_INSTRUCTION = (
    "Given evidence and one possible answer to a question, determine whether the evidence supports that "
    "answer under the question's criterion. Use only the supplied evidence."
)


def _text(v: object) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


@dataclass(frozen=True)
class Row:
    kind: str  # choice | noul | score
    keys: list[str]
    question: str
    options: list[str]  # "<key>: <description>"


def to_row(q: object, max_options: int) -> Row:
    """benchmarks/build_typesafe.py's mapping, plus score levels."""
    if not isinstance(q, Mapping):
        raise TypeError("question must be an object")
    kind, criteria, question = q.get("type"), q.get("criteria"), _text(q.get("instructions", ""))
    if kind == "noul":
        c = criteria if isinstance(criteria, Mapping) else {}
        pairs = [(k, _text(c.get(k, f"The proposition is {k}."))) for k in ("true", "false")]
    elif kind == "choice" and isinstance(criteria, Mapping):
        pairs = [(str(k), _text(v)) for k, v in criteria.items()]
    elif kind == "score" and isinstance(criteria, list):
        pairs = [(str(i), _text(v)) for i, v in enumerate(criteria)]
    else:
        raise ValueError(f"unsupported question: type {kind!r} with criteria {type(criteria).__name__}")
    if not 2 <= len(pairs) <= min(max_options, len(LETTERS)):
        raise ValueError(f"{len(pairs)} options; semif takes 2-{min(max_options, len(LETTERS))}")
    if not question:
        raise ValueError("semif needs non-empty instructions")
    return Row(kind, [k for k, _ in pairs], question, [f"{k}: {d}" for k, d in pairs])


def direct_messages(state: str, row: Row) -> list[dict[str, str]]:
    payload = {
        "evidence": state,
        "criterion": row.question,
        "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(row.options)],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def rerank_text(state: str, row: Row, option: str) -> str:
    body = (
        f"<Instruct>: {DECISION_INSTRUCTION}\n"
        f"<Query>: Question: {row.question}\nCandidate answer: {option}\n"
        f"<Document>: {state}"
    )
    return RERANK_PREFIX + body + RERANK_SUFFIX


def answer(row: Row, scores: Sequence[float]) -> dict[str, object]:
    """Softmax (T = 1, as upstream) of the option scores, in the typed answer shape. Noul is (true, false)."""
    if row.kind == "noul":
        return noul_from_logits(scores[0], scores[1])
    if row.kind == "choice":
        return choice_answer(dict(zip(row.keys, scores, strict=True)))
    return score_answer(scores)


@dataclass
class SemIf:
    tokenizer: object
    model: torch.nn.Module
    mode: str
    device: torch.device
    max_tokens: int
    max_options: int
    letter_ids: list[int]
    yes_no: tuple[int, int] | None  # (no, yes); reranker only
    temperature: float = 1.0
    hf_config: dict[str, object] = field(default_factory=dict)  # what the roofline cost model reads
    ceilings: dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    recorder: profiling.ForwardRecorder | None = None

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))  # ty: ignore[unresolved-attribute]

    def _checked(self, ids: list[int]) -> list[int]:
        if not ids or len(ids) > self.max_tokens:
            raise ValueError(f"{len(ids)} prompt tokens, over max_prompt_tokens={self.max_tokens}")
        return ids

    def _last_logits(self, rows: list[list[int]]) -> torch.Tensor:
        """Last-position logits for a left-padded batch of prompts, one forward pass."""
        import torch  # noqa: PLC0415

        width = max(len(r) for r in rows)
        pad = self.tokenizer.pad_token_id  # ty: ignore[unresolved-attribute]
        ids = torch.tensor([[pad] * (width - len(r)) + r for r in rows], device=self.device)
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], device=self.device)
        kwargs: dict[str, object] = {"input_ids": ids, "attention_mask": mask, "use_cache": False}
        if "logits_to_keep" in inspect.signature(self.model.forward).parameters:
            kwargs["logits_to_keep"] = 1
        with self.lock, torch.inference_mode():
            logits = self.model(**kwargs).logits[:, -1, :].float()
        if not isinstance(logits, torch.Tensor):
            raise TypeError("model returned no logits")
        return logits

    def scores(self, state: str, row: Row) -> list[float]:
        if self.mode == "direct":
            prompt = self.tokenizer.apply_chat_template(  # ty: ignore[unresolved-attribute]
                direct_messages(state, row), tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            logits = self._last_logits([self._checked(self.encode(prompt))])[0]
            return [float(logits[t]) / self.temperature for t in self.letter_ids[: len(row.options)]]
        if self.yes_no is None:
            raise RuntimeError("reranker mode needs the yes/no token ids")
        pairs = [self._checked(self.encode(rerank_text(state, row, o))) for o in row.options]
        no, yes = self.yes_no
        logits = self._last_logits(pairs)
        return [float(x) / self.temperature for x in (logits[:, yes] - logits[:, no]).cpu()]

    def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
        rows = {key: to_row(q, self.max_options) for key, q in questions.items()}
        return {key: answer(row, self.scores(state, row)) for key, row in rows.items()}

    def profile_info(self) -> dict[str, object]:
        if self.recorder is None:
            self.recorder = profiling.ForwardRecorder(self.model)
        return {"config": self.hf_config, "head_bytes": 2, "ceilings": self.ceilings}

    def take_forwards(self) -> list[dict[str, object]]:
        return [] if self.recorder is None else self.recorder.take()

    def count_flops(self, state: str, questions: Mapping[str, object]) -> float:
        return profiling.count_flops(lambda: self.decide(state, questions))[1]

    def profiler(self, on: bool) -> None:
        """Open/close the CUDA profiler range Nsight Compute records (`s1 ncu`)."""
        import torch  # noqa: PLC0415

        torch.cuda.synchronize()
        (torch.cuda.profiler.start if on else torch.cuda.profiler.stop)()

    def run_info(self) -> dict[str, Scalar]:
        return {f"env_{k}": v for k, v in profiling.kernel_env(self.model).items()}

    def close(self) -> None:
        """Drop the weights and free the GPU, so the next run in this process can load."""
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415

        if self.recorder is not None:
            self.recorder.close()
        self.model = self.recorder = None  # ty: ignore[invalid-assignment]
        gc.collect()
        torch.cuda.empty_cache()


def single_token(tokenizer: object, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)  # ty: ignore[unresolved-attribute]
    if len(ids) != 1 or tokenizer.decode(ids) != text:  # ty: ignore[unresolved-attribute]
        raise ValueError(f"{text!r} is not one round-trip token")
    return int(ids[0])


def build(spec: ModelSpec, settings: dict[str, Scalar]) -> SemIf:
    import transformers  # noqa: PLC0415

    mode = get_str(settings, "mode", "direct")
    if mode not in ("direct", "reranker"):
        raise ValueError(f"semif mode must be direct or reranker, got {mode!r}")
    name = spec.repo.rpartition("/")[2]
    model_dir = gpu.local_dir(settings, "model_dir", name, gpu.download_hint(spec.repo, spec.revision, name))
    device = gpu.cuda_device(settings, spec.name)
    ceilings = (
        profiling.measure_ceilings(str(device)) if get_bool(settings, "measure_ceilings", False) else {}
    )
    tok = transformers.AutoTokenizer.from_pretrained(model_dir)
    if tok is None:
        raise RuntimeError(f"no tokenizer in {model_dir}")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    config = transformers.AutoConfig.from_pretrained(model_dir)
    cls = transformers.AutoModelForCausalLM
    if config.model_type in ("qwen3_5", "qwen3_5_text"):  # the VLM checkpoint's text tower, as upstream
        cls, config = transformers.Qwen3_5ForCausalLM, config.get_text_config()
    loaded: object = cls.from_pretrained(
        model_dir,
        config=config,
        dtype=gpu.torch_dtype(settings),
        device_map={"": str(device)},
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple):
        raise TypeError("from_pretrained ignored output_loading_info")
    model, info = loaded
    if bad := {k: v for k, v in info.items() if k in ("missing_keys", "mismatched_keys", "error_msgs") and v}:
        raise RuntimeError(f"{model_dir}: incomplete load {bad}")
    letters = [single_token(tok, c) for c in LETTERS]
    if len(set(letters)) != len(letters):
        raise ValueError("option letters collide")
    return SemIf(
        tokenizer=tok,
        model=model.eval(),
        mode=mode,
        device=device,
        max_tokens=get_int(settings, "max_prompt_tokens", 16384),
        max_options=get_int(settings, "semif_max_options", 16),
        letter_ids=letters,
        yes_no=(single_token(tok, "no"), single_token(tok, "yes")) if mode == "reranker" else None,
        temperature=get_float(settings, "semif_temperature", 1.0),
        hf_config=dict(config.to_dict()),
        ceilings=ceilings,
    )


class SemIfBackend:
    def build(self, spec: ModelSpec, settings: dict[str, Scalar]) -> SemIf:
        return build(spec, settings)
