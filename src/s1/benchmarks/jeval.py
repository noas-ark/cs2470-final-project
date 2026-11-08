"""JEVal (huggingface.co/datasets/carlosxiang/JEVal, arXiv:2610.03935): 11,257 decision questions
from 36 datasets across 10 domains, already in the /v1/systemone request shape (state + questions).

Metrics mirror the paper's Table 1: `jeval_accuracy` over the choice + noul questions with invalid
responses counted wrong, and ECE / raw Brier over the valid ones. The 319 `score` questions are
reported separately (accuracy of the most likely level, and MAE of the expected level).

Settings (all optional):
    jeval_limit        questions to run, 0 = all     default 0
    jeval_seed         seed for the --limit sample   default 0
    jeval_concurrency  requests in flight            default 8
    jeval_warmup       answer the sample once, untimed, before measuring    default false
                       (excludes one-time costs: kernel compilation, CUDA graph capture. The model's
                       `after_warmup()`, if any, then drops whatever it cached, so no answers carry over.)
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..record import Metrics, summarize
from ..registry import Settings, benchmark, get_bool, get_int, warmup

REPO = "carlosxiang/JEVal"
REVISION = "5d0253c34cf5e371b3ae99c21f0a7170d7289f63"  # pinned so results stay comparable
ECE_BINS = 15


class DecisionModel(Protocol):
    def decide(self, state: str, questions: Mapping[str, object]) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class Row:
    id: str
    source: str
    qtype: str  # choice | noul | score
    state: str
    questions: Mapping[str, object]  # exactly one question per row
    label: str | int


@dataclass(frozen=True)
class Outcome:
    qtype: str
    valid: bool
    correct: bool
    confidence: float  # probability given to the predicted answer
    brier: float  # sum over options of (p - onehot(label))^2
    latency_ms: float
    score_abs_err: float = 0.0


def load_rows(path: Path | None = None) -> list[Row]:
    import pyarrow.parquet as pq  # noqa: PLC0415 — heavy, only needed for real runs

    if path is None:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        path = Path(hf_hub_download(REPO, "data/test.parquet", repo_type="dataset", revision=REVISION))
    return [
        Row(
            id=r["id"],
            source=r["source"],
            qtype=r["question_type"],
            state=r["state"],
            questions=json.loads(r["questions_json"]),
            label=json.loads(r["label_json"]),
        )
        for r in pq.read_table(path, use_threads=False).to_pylist()
    ]


def _probs(answer: dict[str, object]) -> dict[str, float]:
    p = answer.get("probabilities")
    if not isinstance(p, dict):
        raise TypeError("answer has no probabilities")
    return {str(k): float(v) for k, v in p.items()}


def grade(row: Row, answer: object, latency_ms: float) -> Outcome:
    """Score one answer against the row's label. Raises on a malformed answer (-> invalid)."""
    if not isinstance(answer, dict):
        raise TypeError(f"{row.id}: answer is not an object")
    if row.qtype == "choice":
        (question,) = row.questions.values()
        if not isinstance(question, dict) or not isinstance(question.get("criteria"), dict):
            raise TypeError(f"{row.id}: choice question without criteria")
        options = [str(k) for k in question["criteria"]]
        probs, pred = _probs(answer), answer.get("choice")
        if pred not in options:
            raise ValueError(f"{row.id}: chose {pred!r}, not one of the options")
        brier = sum((probs.get(o, 0.0) - (o == row.label)) ** 2 for o in options)
        return Outcome("choice", True, pred == row.label, probs[str(pred)], brier, latency_ms)

    if row.qtype == "noul":
        p_true = answer.get("noul")
        if not isinstance(p_true, int | float):
            raise TypeError(f"{row.id}: noul answer without a probability")
        y = int(row.label)
        correct = (p_true >= 0.5) == bool(y)
        return Outcome("noul", True, correct, max(p_true, 1 - p_true), 2 * (p_true - y) ** 2, latency_ms)

    if row.qtype == "score":
        expected, probs = answer.get("score"), _probs(answer)
        if not isinstance(expected, int | float):
            raise TypeError(f"{row.id}: score answer without an expected level")
        y = int(row.label)
        best = int(max(probs, key=lambda k: probs[k]))
        brier = sum((p - (int(k) == y)) ** 2 for k, p in probs.items())
        return Outcome("score", True, best == y, max(probs.values()), brier, latency_ms, abs(expected - y))

    raise ValueError(f"{row.id}: unknown question type {row.qtype!r}")


def ask(model: DecisionModel, row: Row) -> Outcome:
    t = time.perf_counter()
    try:
        answers = model.decide(row.state, row.questions)
        ms = (time.perf_counter() - t) * 1000
        (answer,) = answers.values()
        return grade(row, answer, ms)
    except Exception:  # any failure is an invalid response, counted wrong as in the paper
        return Outcome(row.qtype, False, False, 0.0, 0.0, (time.perf_counter() - t) * 1000)


def ece(outcomes: Sequence[Outcome], bins: int = ECE_BINS) -> float:
    """Expected calibration error of the top-answer confidence, equal-width bins."""
    total = len(outcomes)
    err = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        in_bin = [o for o in outcomes if lo < o.confidence <= hi or (b == 0 and o.confidence == 0)]
        if in_bin:
            acc = sum(o.correct for o in in_bin) / len(in_bin)
            conf = sum(o.confidence for o in in_bin) / len(in_bin)
            err += len(in_bin) / total * abs(acc - conf)
    return err


def metrics(outcomes: Sequence[Outcome], wall_s: float) -> Metrics:
    cls = [o for o in outcomes if o.qtype in ("choice", "noul")]
    cls_valid = [o for o in cls if o.valid]
    scores = [o for o in outcomes if o.qtype == "score"]
    scores_valid = [o for o in scores if o.valid]
    valid = [o for o in outcomes if o.valid]

    def acc(xs: Sequence[Outcome]) -> float:
        return sum(o.correct for o in xs) / len(xs) if xs else float("nan")

    def mean(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    m: Metrics = {
        "jeval_accuracy": acc(cls),  # headline: first key is what `s1 ls` shows
        "jeval_ece": ece(cls_valid) if cls_valid else float("nan"),
        "jeval_brier": mean([o.brier for o in cls_valid]),
        "jeval_choice_accuracy": acc([o for o in cls if o.qtype == "choice"]),
        "jeval_noul_accuracy": acc([o for o in cls if o.qtype == "noul"]),
        "jeval_score_accuracy": acc(scores),
        "jeval_score_mae": mean([o.score_abs_err for o in scores_valid]),
        "jeval_n": float(len(outcomes)),
        "jeval_invalid": float(len(outcomes) - len(valid)),
        "jeval_questions_per_s": len(outcomes) / wall_s if wall_s > 0 else float("nan"),
    }
    if valid:
        m |= summarize("jeval_latency_ms", [o.latency_ms for o in valid])
    return m


def evaluate(model: DecisionModel, rows: Sequence[Row], concurrency: int = 8) -> Metrics:
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(lambda r: ask(model, r), rows))
    return metrics(outcomes, time.perf_counter() - t)


def sample(settings: Settings) -> list[Row]:
    rows = load_rows()
    if limit := get_int(settings, "jeval_limit", 0):
        rows = random.Random(get_int(settings, "jeval_seed", 0)).sample(rows, min(limit, len(rows)))
    return rows


def _decision_model(model: object) -> DecisionModel:
    if not callable(getattr(model, "decide", None)):
        raise TypeError(
            f"jeval needs a decision model with .decide(state, questions); got {type(model).__name__}"
        )
    return model  # ty: ignore[invalid-return-type]


@warmup("jeval")
def warm(model: object, settings: Settings) -> bool:
    if not get_bool(settings, "jeval_warmup", False):
        return False
    evaluate(_decision_model(model), sample(settings), get_int(settings, "jeval_concurrency", 8))
    if callable(after := getattr(model, "after_warmup", None)):
        after()
    return True


@benchmark("jeval")
def run(model: object, settings: Settings) -> Metrics:
    return evaluate(_decision_model(model), sample(settings), get_int(settings, "jeval_concurrency", 8))
