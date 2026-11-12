"""The roofline benchmark's questions-per-request (Q) axis: what one shared state costs as more questions ride
on it. Selected by `[grid] roofline_questions_per_request` (with `[grid] roofline_state_tokens`);
`roofline.run` hands the whole run to `run` here.

At every (state length L, Q) point the model gets `roofline_qpr_requests` requests, each ONE `decide(state,
questions)` call with Q questions against one state. The states are the length axis's
(`roofline.length_rows`: real JEVal states at least max(L) reference tokens long, each cut to its first L
tokens), and the SAME states run at every point; the questions are a fixed, seeded pool of `max(Q)` JEVal
questions, and a Q-question request asks the first Q of them. So between two points of one length only Q
differs, and the questions added at Q+1 are always the same one. Questions come from other rows than the
state (a fixed pool is what keeps Q the only variable), so answers are timed and checked for shape, never
graded: accuracy here means nothing.

The pool keeps questions whose JSON is `roofline_qpr_min_question_tokens` to
`roofline_qpr_max_question_tokens` reference tokens long (default 40-79: less than twice the shortest).
Bongard-mini decodes a request's questions in length-sorted batches of up to 8 and starts a new batch early
when a question is over twice the batch's shortest; a narrow band keeps its batches at exactly 8, so any
steps sit at Q = 9, 17, 25.

Each point is warmed with `roofline_qpr_warmup` requests on a state that is not timed (a new shape is a new
CUDA graph for kev, a new kernel plan for an MoE). `after_warmup()`, when the model has one, runs after each
point's warm-up and again before every timed request: kev keeps a state-prefix cache that would otherwise
answer a state it saw at an earlier Q, and its after_warmup empties it (and captures pending CUDA graphs).
Its GPU work is never recorded.

Per point (`roofline_s<L>_q<Q>_*`): achieved TFLOP/s, intensity and bandwidth of the point's forwards
(`roofline.point`), and per request: GPU ms (mean, median), wall latency, forwards, executed tokens,
padding, invalid answers; per question: GPU ms, latency, GFLOP, MB. Per length (`roofline_s<L>_q*`), a
least-squares line through the per-point median GPU ms against Q: `qslope_gpu_ms` (GPU ms per added
question), `qintercept_gpu_ms`, `qmarginal` (slope / the Q = 1 median: 1 means a question costs what a whole
request did, near 0 that the state dominates and is shared), and Q-max over Q-min ratios of TFLOP/s
(`qtflops_gain`) and GPU ms per question (`qcost_ratio`). Wall latency gets its own slope
(`qslope_latency_ms`). The pooled headline keys (`roofline_tflops`, `roofline_intensity`, ...) pool every
point.

Settings (all optional; the length axis's `roofline_seed`, `roofline_max_options`, `roofline_tokenizer_file`
apply too):
    roofline_qpr_requests            timed requests per point                       default 6
    roofline_qpr_warmup              untimed requests per point first              default 1
    roofline_qpr_min_question_tokens reference tokens of a pool question's JSON      default 40
    roofline_qpr_max_question_tokens                                                default 79
    roofline_qpr_max_questions_at_<L>  cap Q at length L (a model that cannot fit max(Q) there); points over
                                     the cap are skipped and noted                   default none
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .. import flops
from ..record import Metrics
from ..registry import Settings, get_int, note
from . import jeval, roofline
from .shape_sweep import parse_int_list

QPR_AXIS = roofline.QPR_AXIS  # [grid] axis: questions per decide() call


@dataclass(frozen=True)
class Request:
    length: int
    q: int
    state: str
    questions: Mapping[str, object]


@dataclass(frozen=True)
class Sample:
    """One timed request."""

    length: int
    q: int
    cost: flops.Cost
    forwards: int
    latency_ms: float
    invalid: int
    real_tokens: float


def question_pool(
    rows: Sequence[jeval.Row], count: Callable[[str], int], settings: Settings, k: int, skip: set[str]
) -> list[object]:
    """`k` single questions (the question objects, not rows), seeded and spread across datasets: choice/noul
    with at most `roofline_max_options` options whose JSON is within the token band, none from a row in
    `skip` (the state rows). Raises if fewer than `k` qualify."""

    lo = get_int(settings, "roofline_qpr_min_question_tokens", 40)
    hi = get_int(settings, "roofline_qpr_max_question_tokens", 79)
    max_options = get_int(settings, "roofline_max_options", 16)
    keep = [
        r
        for r in rows
        if r.id not in skip
        and r.qtype in roofline.TYPES
        and roofline.options(r) <= max_options
        and lo <= count(json.dumps(r.questions, ensure_ascii=False)) <= hi
    ]
    if len(keep) < k:
        raise ValueError(
            f"only {len(keep)} JEVal questions are {lo}-{hi} reference tokens (choice/noul, at most "
            f"{max_options} options); {QPR_AXIS} needs {k}"
        )
    chosen = roofline.spread(keep, k, random.Random(get_int(settings, "roofline_seed", 0) + 1))
    return [next(iter(r.questions.values())) for r in chosen]


def caps(settings: Settings, lengths: Sequence[int]) -> dict[int, int]:
    """Per-length Q caps from `roofline_qpr_max_questions_at_<L>` (absent: no cap)."""
    return {n: get_int(settings, f"roofline_qpr_max_questions_at_{n}", 0) for n in lengths}


def plan(
    rows: Sequence[jeval.Row],
    counter: roofline.FittingCounter,
    settings: Settings,
    lengths: Sequence[int],
    qs: Sequence[int],
) -> tuple[list[tuple[list[Request], list[Request]]], list[tuple[int, int]]]:
    """[(warm-up requests, timed requests) per point], length-major then Q ascending, and the (L, Q) points
    skipped by a cap. The same `roofline_qpr_requests` states (cut to each length) run at every point; one
    more state, never timed, serves every warm-up."""

    if not qs or min(qs) < 1:
        raise ValueError(f"{QPR_AXIS} needs positive question counts, got {list(qs)}")
    per = get_int(settings, "roofline_qpr_requests", 6)
    warm_n = get_int(settings, "roofline_qpr_warmup", 1)
    tagged = roofline.length_rows(
        rows, counter, {**settings, "roofline_per_length": per + 1}, sorted(set(lengths))
    )
    pool = question_pool(rows, counter.count, settings, max(qs), {r.id for _, r in tagged})
    cap = caps(settings, lengths)
    points: list[tuple[list[Request], list[Request]]] = []
    skipped: list[tuple[int, int]] = []
    for n in sorted(set(lengths)):
        states = [r.state for m, r in tagged if m == n]
        for q in sorted(set(qs)):
            if cap[n] and q > cap[n]:
                skipped.append((n, q))
                continue
            questions = {f"q{i}": pool[i] for i in range(q)}
            warm = [Request(n, q, states[0], questions)] * warm_n
            timed = [Request(n, q, s, questions) for s in states[1:]]
            points.append((warm, timed))
    return points, skipped


def invalid_answers(answers: object, keys: Sequence[str]) -> int:
    """Questions without a well-formed answer: missing, or not an object with a "type"."""
    if not isinstance(answers, Mapping):
        return len(keys)
    return sum(not (isinstance(a := answers.get(k), Mapping) and "type" in a) for k in keys)


def stack_cost(
    parts: Mapping[str, flops.Arch], raw: Sequence[Mapping[str, object]], **sizes: int
) -> flops.Cost:
    """The cost of recorded forwards, each charged to the stack it names (`stack`, "" when absent)."""

    total = flops.ZERO
    for f in raw:
        stack = str(f.get("stack", ""))
        if stack not in parts:
            raise KeyError(f"forward names stack {stack!r}; the model has {sorted(parts)}")
        fwd = roofline.forward(f)
        fwd = flops.Forward(
            fwd.rows,
            fwd.tokens,
            fwd.past,
            fwd.kept,
            fwd.cached,
            fwd.ms,
            bool(f.get("shared_past", False)),
            stack,
        )
        total = total + flops.cost(parts[stack], fwd, **sizes)
    return total


def fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Least-squares (slope, intercept); nan for fewer than two distinct x."""
    if len(set(xs)) < 2:
        return math.nan, math.nan
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    return slope, my - slope * mx


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


def point_metrics(samples: Sequence[Sample]) -> Metrics:
    """`roofline_s<L>_q<Q>_*` for every point, and the per-length Q fits."""

    m: Metrics = {}
    by_point: dict[tuple[int, int], list[Sample]] = {}
    for s in samples:
        by_point.setdefault((s.length, s.q), []).append(s)
    medians: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for (n, q), at in sorted(by_point.items()):
        cost = flops.ZERO
        for s in at:
            cost = cost + s.cost
        k = f"roofline_s{n}_q{q}_"
        m |= roofline.point(k, cost, len(at))
        ms_med = statistics.median(s.cost.ms for s in at)
        lat_mean = statistics.fmean(s.latency_ms for s in at)
        m |= {
            k + "questions": float(q),
            k + "gpu_ms": cost.ms / len(at),
            k + "gpu_ms_median": ms_med,
            k + "gpu_ms_per_question": cost.ms / len(at) / q,
            k + "latency_ms": lat_mean,
            k + "latency_ms_per_question": lat_mean / q,
            k + "forwards": statistics.fmean(s.forwards for s in at),
            k + "tokens": cost.tokens / len(at),
            k + "padding": 1 - _ratio(sum(s.real_tokens for s in at), cost.tokens),
            k + "gflop_per_question": cost.flops / 1e9 / len(at) / q,
            k + "mb_per_question": cost.bytes / 1e6 / len(at) / q,
            k + "invalid": float(sum(s.invalid for s in at)),
        }
        tflops = m[k + "tflops"]
        medians.setdefault(n, []).append((q, ms_med, lat_mean, tflops, ms_med / q))
    for n, pts in medians.items():
        pts.sort()
        qs = [float(p[0]) for p in pts]
        slope, intercept = fit(qs, [p[1] for p in pts])
        lat_slope, _ = fit(qs, [p[2] for p in pts])
        base = pts[0][1] if pts[0][0] == 1 else intercept + slope
        k = f"roofline_s{n}_q"
        m |= {
            k + "slope_gpu_ms": slope,
            k + "intercept_gpu_ms": intercept,
            k + "marginal": _ratio(slope, base),
            k + "slope_latency_ms": lat_slope,
            k + "tflops_gain": _ratio(pts[-1][3], pts[0][3]),
            k + "cost_ratio": _ratio(pts[-1][4], pts[0][4]),
            k + "max": float(pts[-1][0]),
        }
    return m


def headline(
    samples: Sequence[Sample],
    arch: flops.Arch,
    ceilings: Mapping[str, float],
    gpu: str,
) -> Metrics:
    """Pooled keys named as in natural mode, the unit being the question where natural mode has one."""

    total = flops.ZERO
    for s in samples:
        total = total + s.cost
    n_q = sum(s.q for s in samples)
    m = roofline.point("roofline_", total, n_q)
    m.pop("roofline_n")
    m |= {
        "roofline_n": float(n_q),
        "roofline_invalid": float(sum(s.invalid for s in samples)),
        "roofline_latency_ms": sum(s.latency_ms for s in samples) / n_q if n_q else math.nan,
        "roofline_requests": float(len(samples)),
        "roofline_gflop_per_question": total.flops / 1e9 / n_q if n_q else math.nan,
        "roofline_mb_per_question": total.bytes / 1e6 / n_q if n_q else math.nan,
        "roofline_padding": 1 - _ratio(sum(s.real_tokens for s in samples), total.tokens),
        "roofline_gpu_share": _ratio(total.ms, sum(s.latency_ms for s in samples)),
        "roofline_params_b": arch.params / 1e9,
        "roofline_active_params_b": arch.active_params / 1e9,
        "roofline_attention_layers": sum(ly.mixer == "attention" for ly in arch.layers) / len(arch.layers),
    }
    m |= {f"roofline_{k}": float(v) for k, v in ceilings.items()}
    peak, bw = ceilings.get("peak_bf16_tflops"), ceilings.get("peak_bw_gbs")
    if peak is not None and bw is not None and peak > 0 and bw > 0:
        attainable = min(peak, bw * m["roofline_intensity"] / 1e3)
        m |= {
            "roofline_ridge": peak * 1e3 / bw,
            "roofline_attainable_tflops": attainable,
            "roofline_efficiency": m["roofline_tflops"] / attainable,
            "roofline_mfu": m["roofline_tflops"] / peak,
        }
    if (spec := roofline.spec_peaks(gpu)) is not None:
        m |= {
            "roofline_spec_bf16_tflops": spec[0],
            "roofline_spec_fp32_tflops": spec[1],
            "roofline_spec_bw_gbs": spec[2],
        }
    return m


def run(
    model: roofline.Profiled,
    settings: Settings,
    config: Mapping[str, object],
    ceilings: Mapping[str, float],
    sizes: Mapping[str, int],
) -> Metrics:
    """The Q-axis run (see the module docstring); roofline.run has read the model's profile_info."""
    parts = flops.stacks(config)
    lengths = (
        parse_int_list(settings, roofline.LENGTH_AXIS) if f"grid_{roofline.LENGTH_AXIS}" in settings else []
    )
    if not lengths:
        raise ValueError(
            f"{QPR_AXIS} needs [grid] {roofline.LENGTH_AXIS} too (the state lengths to hold fixed)"
        )
    points, skipped = plan(
        jeval.load_rows(), roofline._counter(settings), settings, lengths, parse_int_list(settings, QPR_AXIS)
    )
    if skipped:
        note("roofline_qpr_skipped", "; ".join(f"s{n}_q{q}" for n, q in skipped))
    reset = getattr(model, "after_warmup", None)
    reset = reset if callable(reset) else None

    samples: list[Sample] = []
    for warm, timed in points:
        for r in warm:
            with contextlib.suppress(Exception):  # a failing shape shows as invalid in the timed requests
                model.decide(r.state, r.questions)
        for r in timed:
            if reset is not None:
                reset()
            model.take_forwards()
            t = time.perf_counter()
            try:
                bad = invalid_answers(model.decide(r.state, r.questions), list(r.questions))
            except Exception:
                bad = r.q
            ms = (time.perf_counter() - t) * 1000
            raw = model.take_forwards()
            samples.append(
                Sample(
                    r.length,
                    r.q,
                    stack_cost(parts, raw, **sizes),
                    len(raw),
                    ms,
                    bad,
                    roofline.real_tokens(raw),
                )
            )
    m = headline(samples, flops.merged(parts), ceilings, str(settings.get("gpu", "")))
    m |= point_metrics(samples)
    return m
