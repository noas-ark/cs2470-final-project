"""Model-agnostic request-shape sweep: cost per question as a function of state length and of how
many questions share one request. The measurement of interest is GPU utilization per point.

For every (state_tokens, questions_per_request) point of the run's `[grid]`, build one request (an
N-token state plus Q choice questions in the JEVal question shape), send it with concurrency 1 and
repeat until `shape_min_point_s` has elapsed, so the GPU sampler (500 ms interval, all GPUs pooled,
see `gpu_stats.sample_point`) sees enough samples. Latency is divided by Q: the unit of cost is the
question. Failed or malformed answers count as invalid, not as skipped.

Every repetition gets a DIFFERENT state of the same length (`shape_unique_states`, default true).
Sending the identical request again lets a prefix cache answer it, so only the first repetition would
pay for the state and the rest would measure the cache. (Job 4059 showed exactly that for OpenJev:
vLLM caches this hybrid model's prompts in 784-token blocks.) The Q questions of one request still
share their state, as they do in a real request. Set `shape_unique_states = false` to measure
cache-hit cost on purpose.

Grid axes (config `[grid]`, exposed to the benchmark as `grid_<axis>` = comma-joined values):
    state_tokens           target state lengths
    questions_per_request  questions per request

Settings (all optional):
    shape_warmup       warm-up requests at the first runnable point, discarded   default 1
    shape_min_point_s  keep repeating a point until this many seconds passed     default 2.0
    shape_unique_states  a fresh state for every repetition (see above)          default true
    state_tokenizer    name of the token counter (see TOKENIZERS)                default "approx_words"
    state_tokenizer_file  tokenizer.json for state_tokenizer = "file"; $MODELS_DIR, ~ and $VARS are expanded
    state_seed         seed of the synthetic state text                          default 0

The state length is measured by a `TokenCounter`, and the ACTUAL count of every point is recorded
(`..._state_tokens_actual`, of the first request). `approx_words` (one token per word) is dependency-free
and only for tests: the first run used it and a "2048-token" state was really ~10,000 tokens. Use
`state_tokenizer = "file"` with one reference tokenizer for every model, so a state length means the
same thing across models. The question text and chat template are added on top of the state.
Models may define `max_state_tokens`; points above it are skipped (`shape_sweep_skipped`), not
counted as wrong answers.

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

from .. import env
from ..gpu_stats import GpuSampler, sample_point
from ..record import Metrics
from ..registry import Settings, benchmark, get_bool, get_float, get_int, get_str

NAME = "shape_sweep"
OPTIONS = ("a", "b", "c", "d")
GPU_KEYS = ("gpu_util_pct", "gpu_mem_used_mb_max", "gpu_power_w_total_mean", "gpu_count", "gpu_samples")
MAX_FIT_ROUNDS = 8
# Very common words: almost all are a single token (with a leading space) in common BPE vocabularies, so a
# state of N words is about N tokens and the length fit below converges. Pseudo-words like "w4821" were
# ~5 tokens each (digits are split one by one).
WORDS = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "that",
    "for",
    "it",
    "as",
    "was",
    "with",
    "be",
    "by",
    "on",
    "not",
    "he",
    "this",
    "are",
    "or",
    "his",
    "from",
    "at",
    "which",
    "but",
    "have",
    "an",
    "had",
    "they",
    "you",
    "were",
    "their",
    "one",
    "all",
    "we",
    "can",
    "her",
    "has",
    "there",
    "been",
    "if",
    "more",
    "when",
    "will",
    "would",
    "who",
    "so",
    "no",
    "said",
    "what",
    "up",
    "out",
    "about",
    "into",
    "than",
    "them",
    "some",
    "could",
    "other",
    "then",
    "two",
    "time",
    "these",
]


class DecisionModel(Protocol):
    def decide(self, state: str, questions: Mapping[str, object]) -> Mapping[str, object]: ...


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class StateSource(Protocol):
    def state(self, tokens: int, counter: TokenCounter, salt: int = 0) -> str:
        """Text whose `counter.count` is `tokens` (or as close as the counter allows).

        The same (tokens, salt) always gives the same text; a different salt gives different text.
        """
        ...


class WordCounter:
    """Deterministic approximation: one token per whitespace-separated word."""

    name = "approx_words"

    def count(self, text: str) -> int:
        return len(text.split())


class FileTokenCounter:
    """Token counts from a real tokenizer file (a Hugging Face `tokenizer.json`).

    Use one reference tokenizer for every model so a state length means the same thing across models.
    Needs the `tokenizers` package (installed with `transformers`, so with the openjev extra).
    """

    name = "file"

    def __init__(self, path: str) -> None:
        try:
            from tokenizers import Tokenizer  # noqa: PLC0415 — heavy and optional
        except ImportError as e:
            raise ImportError("state_tokenizer = 'file' needs the `tokenizers` package") from e
        self.path = env.resolve_path(path)
        self._tokenizer = Tokenizer.from_file(str(self.path))

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


def _file_counter(settings: Settings) -> TokenCounter:
    path = get_str(settings, "state_tokenizer_file", "")
    if not path:
        raise KeyError("state_tokenizer = 'file' needs state_tokenizer_file (a tokenizer.json path)")
    return FileTokenCounter(path)


TOKENIZERS: dict[str, Callable[[Settings], TokenCounter]] = {
    WordCounter.name: lambda _settings: WordCounter(),
    FileTokenCounter.name: _file_counter,
}


@dataclass(frozen=True)
class SyntheticStates:
    """Seeded common words, grown or trimmed word by word until the counter reads the target."""

    seed: int = 0

    def state(self, tokens: int, counter: TokenCounter, salt: int = 0) -> str:
        rng = random.Random(f"{self.seed}:{tokens}:{salt}")
        words: list[str] = []

        def grow(n: int) -> None:
            words.extend(rng.choice(WORDS) for _ in range(n))

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
    state: str | Callable[[int], str],
    questions: Mapping[str, object],
    min_s: float,
    clock: Callable[[], float],
) -> tuple[list[float], int, int, float]:
    """(latency in ms of each fully valid request, requests made, invalid question count, wall s).

    `state` is the request's state text, or a function of the repetition number (0, 1, ...) that
    returns it. It is built before the request's clock starts, so building it is not measured.
    """
    latencies: list[float] = []
    reps = invalid = 0
    t0 = clock()
    while True:
        text = state if isinstance(state, str) else state(reps)
        t = clock()
        try:
            bad = check_answers(questions, model.decide(text, questions))
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
    unique_states: bool = True,
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
        questions = make_questions(q)

        def state_for(rep: int, tokens: int = tokens) -> str:
            # salt 0 = the identical request every time; otherwise rep + 1, so warm-up (-1) and every
            # repetition differ and none can be answered from a prefix cache
            return states.state(tokens, counter, salt=rep + 1 if unique_states else 0)

        if warm:  # first runnable point only: loads kernels/caches, result discarded
            warm = False
            for _ in range(warmup):
                with contextlib.suppress(Exception):  # a failing model shows up as invalid in the point
                    model.decide(state_for(-2 if unique_states else 0), questions)
        with sample_point(pre, GPU_KEYS, gpu_sampler) as gpu:
            latencies, reps, bad, wall_s = measure_point(model, state_for, questions, min_point_s, clock)
        asked = reps * q
        n += asked
        invalid += bad
        m[f"{pre}state_tokens_actual"] = float(counter.count(state_for(0)))
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
        f"{NAME}_unique_states": float(unique_states),
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
    counter = TOKENIZERS[tokenizer](settings)
    return sweep(
        model,  # ty: ignore[invalid-argument-type]
        parse_int_list(settings, "state_tokens"),
        parse_int_list(settings, "questions_per_request"),
        counter=counter,
        states=SyntheticStates(get_int(settings, "state_seed", 0)),
        warmup=get_int(settings, "shape_warmup", 1),
        min_point_s=get_float(settings, "shape_min_point_s", 2.0),
        unique_states=get_bool(settings, "shape_unique_states", True),
    )
