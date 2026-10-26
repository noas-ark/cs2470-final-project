"""Model-agnostic request-shape sweep: cost per question as a function of state length and of how
many questions share one request. The measurement of interest is GPU utilization per point.

For every (state_tokens, questions_per_request) point of the run's `[grid]`, build one request (an
N-token state plus Q choice questions in the JEVal question shape), send it with concurrency 1 and
repeat until `shape_min_point_s` has elapsed, so the GPU sampler (500 ms interval, all GPUs pooled,
see `gpu_stats.sample_point`) sees enough samples. Latency is divided by Q: the unit of cost is the
question. Failed or malformed answers count as invalid, not as skipped.

Grid axes (config `[grid]`, exposed to the benchmark as `grid_<axis>` = comma-joined values):
    state_tokens           target state lengths
    questions_per_request  questions per request

Settings (all optional):
    shape_warmup       warm-up requests at the first runnable point, discarded   default 1
    shape_min_point_s  keep repeating a point until this many seconds passed     default 2.0
    state_tokenizer    name of the token counter (see TOKENIZERS)                default "approx_words"
    state_seed         seed of the synthetic state text                          default 0

The state length is measured by a `TokenCounter`, and the ACTUAL count of every point is recorded
(`..._state_tokens_actual`). Only the dependency-free `approx_words` counter exists so far; which
real tokenizer to use (one reference for all models vs each model's own) is an open design decision,
so a model-specific counter is added to TOKENIZERS later. Models may define `max_state_tokens`;
points above it are skipped (`shape_sweep_skipped`), not counted as wrong answers.

Results: one flat dict. Headline `shape_sweep_latency_ms` (lower is better) is the mean over points
of the per-question latency; then `shape_sweep_{n,invalid,skipped}` and per point
`shape_sweep_t{T}_q{Q}_{latency_ms,request_ms,state_tokens_actual,reps,invalid,questions_per_s,
gpu_util_pct,gpu_samples,...}` with T, Q the requested grid values.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import random
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..gpu_stats import GpuSampler, sample_point
from ..record import Metrics
from ..registry import Settings, benchmark, get_float, get_int, get_str

NAME = "shape_sweep"
OPTIONS = ("a", "b", "c", "d")
GPU_KEYS = ("gpu_util_pct", "gpu_mem_used_mb_max", "gpu_power_w_total_mean", "gpu_count", "gpu_samples")
MAX_FIT_ROUNDS = 8


class DecisionModel(Protocol):
    def decide(self, state: str, questions: Mapping[str, object]) -> Mapping[str, object]: ...


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class StateSource(Protocol):
    def state(self, tokens: int, counter: TokenCounter) -> str:
        """Text whose `counter.count` is `tokens` (or as close as the counter allows)."""
        ...


class WordCounter:
    """Deterministic approximation: one token per whitespace-separated word."""

    name = "approx_words"

    def count(self, text: str) -> int:
        return len(text.split())


TOKENIZERS: dict[str, Callable[[], TokenCounter]] = {WordCounter.name: WordCounter}


@dataclass(frozen=True)
class SyntheticStates:
    """Seeded pseudo-words, grown or trimmed word by word until the counter reads the target."""

    seed: int = 0

    def state(self, tokens: int, counter: TokenCounter) -> str:
        rng = random.Random(f"{self.seed}:{tokens}")
        words: list[str] = []

        def grow(n: int) -> None:
            words.extend(f"w{rng.randrange(10_000)}" for _ in range(n))

        grow(tokens)
        for _ in range(MAX_FIT_ROUNDS):
            got = counter.count(" ".join(words))
            if got == tokens:
                break
            if got < tokens:
                grow(max(tokens - got, 1))
            else:
                del words[-max(got - tokens, 1) :]
        return " ".join(words)


def parse_int_list(settings: Settings, axis: str) -> list[int]:
    """The grid axis `axis` as ints, from the runner's `grid_<axis>` comma-joined setting."""
    raw = settings.get(f"grid_{axis}")
    if not isinstance(raw, str):
        raise KeyError(f"{NAME} needs a [grid] axis {axis!r} (e.g. {axis} = [1, 2, 4])")
    try:
        vals = [int(v) for v in raw.split(",")]
    except ValueError as e:
        raise ValueError(f"[grid] {axis} must be integers, got {raw!r}") from e
    if any(v < 1 for v in vals):
        raise ValueError(f"[grid] {axis} must be positive, got {raw!r}")
    return vals


def make_questions(n: int) -> dict[str, object]:
    """`n` choice questions in the JEVal shape: {id: {type, instructions, criteria}}."""
    return {
        f"q{i}": {
            "type": "choice",
            "instructions": f"Question {i}: which option best fits the situation described in the state?",
            "criteria": {o: f"Option {o}" for o in OPTIONS},
        }
        for i in range(n)
    }


def check_answers(questions: Mapping[str, object], answers: Mapping[str, object]) -> int:
    """Number of invalid answers among `questions` (missing, malformed, or not one of the options)."""
    bad = 0
    for key, q in questions.items():
        a = answers.get(key)
        criteria = q.get("criteria") if isinstance(q, dict) else None
        if not (isinstance(a, dict) and isinstance(criteria, dict) and a.get("choice") in criteria):
            bad += 1
    return bad


def point_key(tokens: int, questions: int) -> str:
    return f"{NAME}_t{tokens}_q{questions}_"


def measure_point(
    model: DecisionModel,
    state: str,
    questions: Mapping[str, object],
    min_s: float,
    clock: Callable[[], float],
) -> tuple[list[float], int, int, float]:
    """(latency in ms of each fully valid request, requests made, invalid question count, wall s)."""
    latencies: list[float] = []
    reps = invalid = 0
    t0 = clock()
    while True:
        t = clock()
        try:
            bad = check_answers(questions, model.decide(state, questions))
        except Exception:  # any failure is an invalid response, as in jeval
            bad = len(questions)
        ms = (clock() - t) * 1000
        reps += 1
        invalid += bad
        if bad == 0:
            latencies.append(ms)
        if clock() - t0 >= min_s:
            return latencies, reps, invalid, clock() - t0


def sweep(
    model: DecisionModel,
    token_grid: Sequence[int],
    question_grid: Sequence[int],
    *,
    counter: TokenCounter | None = None,
    states: StateSource | None = None,
    warmup: int = 1,
    min_point_s: float = 2.0,
    gpu_sampler: Callable[[], GpuSampler] = GpuSampler,
    clock: Callable[[], float] = time.perf_counter,
) -> Metrics:
    counter = counter or WordCounter()
    states = states or SyntheticStates()
    limit = getattr(model, "max_state_tokens", None)
    points = list(itertools.product(token_grid, question_grid))

    m: Metrics = {}
    point_latency: list[float] = []
    n = invalid = skipped = 0
    warm = True
    for tokens, q in points:
        pre = point_key(tokens, q)
        if isinstance(limit, int) and tokens > limit:
            skipped += 1
            m[f"{pre}skipped"] = 1.0
            continue
        state, questions = states.state(tokens, counter), make_questions(q)
        if warm:  # first runnable point only: loads kernels/caches, result discarded
            warm = False
            for _ in range(warmup):
                with contextlib.suppress(Exception):  # a failing model shows up as invalid in the point
                    model.decide(state, questions)
        with sample_point(pre, GPU_KEYS, gpu_sampler) as gpu:
            latencies, reps, bad, wall_s = measure_point(model, state, questions, min_point_s, clock)
        asked = reps * q
        n += asked
        invalid += bad
        m[f"{pre}state_tokens_actual"] = float(counter.count(state))
        m[f"{pre}reps"] = float(reps)
        m[f"{pre}invalid"] = float(bad)
        m[f"{pre}questions_per_s"] = asked / wall_s if wall_s > 0 else math.nan
        if latencies:
            request_ms = statistics.fmean(latencies)
            m[f"{pre}request_ms"] = request_ms
            m[f"{pre}latency_ms"] = request_ms / q
            point_latency.append(request_ms / q)
        else:
            m[f"{pre}request_ms"] = m[f"{pre}latency_ms"] = math.nan
        m |= gpu.metrics

    head: Metrics = {
        f"{NAME}_latency_ms": statistics.fmean(point_latency) if point_latency else math.nan,
        f"{NAME}_n": float(n),
        f"{NAME}_invalid": float(invalid),
        f"{NAME}_skipped": float(skipped),
        f"{NAME}_tokenizer_{counter.name}": 1.0,
    }
    return head | m


@benchmark(NAME)
def run(model: object, settings: Settings) -> Metrics:
    if not callable(getattr(model, "decide", None)):
        raise TypeError(
            f"{NAME} needs a decision model with .decide(state, questions); got {type(model).__name__}"
        )
    tokenizer = get_str(settings, "state_tokenizer", WordCounter.name)
    if tokenizer not in TOKENIZERS:
        raise KeyError(f"unknown state_tokenizer {tokenizer!r}; known: {', '.join(sorted(TOKENIZERS))}")
    return sweep(
        model,  # ty: ignore[invalid-argument-type]
        parse_int_list(settings, "state_tokens"),
        parse_int_list(settings, "questions_per_request"),
        counter=TOKENIZERS[tokenizer](),
        states=SyntheticStates(get_int(settings, "state_seed", 0)),
        warmup=get_int(settings, "shape_warmup", 1),
        min_point_s=get_float(settings, "shape_min_point_s", 2.0),
    )
