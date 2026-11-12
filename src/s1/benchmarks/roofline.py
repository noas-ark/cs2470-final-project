"""roofline: where a decision model's forward passes sit on its GPU's roofline, on a fast JEVal subset.

Answers a fixed, model-independent subset of JEVal one question at a time (batch 1: the per-decision regime)
and, for every forward pass the model runs, records its shape and GPU time (`s1.profiling.ForwardRecorder`).
`s1.flops` turns each pass into theoretical FLOPs and compulsory DRAM bytes from the model's config, so

    achieved FLOP/s   = theoretical FLOPs / measured GPU time          (headline: roofline_tflops)
    intensity         = theoretical FLOPs / compulsory bytes           (FLOP/byte, x-axis of the roofline)
    attainable FLOP/s = min(measured bf16 GEMM peak, measured copy bandwidth * intensity)

The ceilings are measured on the same GPU by the backend before it loads weights (`measure_ceilings = true`),
and the datasheet's are added for known GPUs. `roofline_counted_ratio` cross-checks the cost model against
torch's FlopCounterMode on the first `roofline_counted` questions (that counter misses custom Triton/CUDA
kernels, so a ratio under 1 there is expected for hybrid models). Accuracy uses jeval's grading, renamed.

The subset is the same for every model: choice and noul questions with at most `roofline_max_options`
options whose state + question JSON is at most `roofline_max_tokens` tokens of one reference tokenizer
(`roofline_tokenizer_file`, a tokenizer.json; words if unset), then a seeded sample.

**State-length axis** (`[grid] roofline_state_tokens = [32, 128, 256, 512, 768, 1024, 1536, 2048]`): in place
of natural JEVal lengths, `roofline_per_length` seeded questions, spread round-robin across JEVal's datasets,
whose own state is at least the longest length (in reference tokens) run at every length, each state cut to
that length's prefix (`counter.fit`). Every point is a contiguous prefix of one real state and the same
questions run at every length, so only the state length differs between points. It places each model's
forwards across every GPU's ridge (bf16 intensity grows with tokens; natural JEVal at <= 1024 tokens sits left
of all of them). Too few qualifying rows is an error, never a silent fallback. Per-length points are
`roofline_s<L>_{tflops,intensity,bandwidth_gbs,n,tokens,gpu_ms,gpu_ms_median,gpu_ms_max}`; the headline keys
pool all lengths. Accuracy at cut lengths is not meaningful (cut states lose evidence): only timing and the
roofline are. A backend that cuts its input (Laya: 512 / 1024 tokens) is charged for what it processed, so its
long points stop growing. Without the axis the benchmark is exactly the natural mode (Nathan's records stay
comparable).

**Questions-per-request axis** (`[grid] roofline_questions_per_request`, with the length axis): Q questions
against one shared state in a single `decide` call, at fixed state lengths. benchmarks/roofline_q.py runs it.

The model must implement (see backends/semif.py, backends/elephant.py):
    profile_info() -> {"config": HF config dict, "head_bytes": int, "ceilings": {name: float}}
                      (and "weight_bytes": 4 for a model that keeps fp32 weights; default 2)
    take_forwards() -> [{"rows", "tokens", "past", "kept", "cached", "ms"}, ...]  (since the last call)
    count_flops(state, questions) -> float
and may implement `after_warmup()` (called once after the warm-up questions; kev captures its CUDA graphs and
empties its prefix cache there). A forward may carry `real_tokens`, the tokens it holds that are not padding
(a CUDA-graph bucket runs a fixed padded shape); the cost model charges the padded shape, the work the GPU
does, and `roofline_padding` is the share of executed tokens that were padding.

Settings (all optional):
    roofline_limit         questions measured                         default 200
    roofline_seed          sample seed                                default 0
    roofline_max_tokens    reference tokens per question              default 1024
    roofline_max_options   options per question                       default 16
    roofline_warmup        questions answered untimed first           default 8
    roofline_counted       questions re-run under FlopCounterMode     default 8
    roofline_ncu           questions re-run in a CUDA profiler range  default 0 (set by `s1 ncu` runs)
    roofline_kernel_profile  1: note the CUDA kernels of one question in meta (roofline_kernels_top,
                           roofline_attention_kernel), traced after timing; 0: skip     default 1
    roofline_per_length    questions per state length (with the [grid] axis)        default 40
    roofline_max_question_tokens  reference tokens of a length-axis question's JSON  default 512
    roofline_tokenizer_file
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .. import env, flops
from ..record import Metrics, summarize
from ..registry import Settings, benchmark, get_int, get_str, note
from . import jeval
from .shape_sweep import FileTokenCounter, TokenCounter, WordCounter, parse_int_list

LENGTH_AXIS = "roofline_state_tokens"  # [grid] axis: state lengths in reference tokens
QPR_AXIS = "roofline_questions_per_request"  # [grid] axis: questions per decide() call (roofline_q.py)
WARM_PER_LENGTH = 2  # untimed questions at each length before timing

BINS = (128, 256, 512, 1024, 4096)  # prompt tokens a question processed (all its forwards)
TYPES = ("choice", "noul")
# Datasheet peaks: dense bf16 tensor TFLOP/s, fp32 (non-tensor) TFLOP/s, DRAM GB/s. NVIDIA quotes bf16 with
# 2:4 sparsity (2x the dense figure used here). bf16 and bandwidth: the selection doc's ridge-point appendix.
SPEC = {
    # L4: 242 TFLOPS BF16 is sparse, so 121 dense; 30.3 FP32; 300 GB/s GDDR6.
    # https://www.nvidia.com/en-us/data-center/l4/
    "L4": (121.0, 30.3, 300.0),
    # A100 80GB SXM: FP32 19.5 (datasheet), 2,039 GB/s. https://www.nvidia.com/en-us/data-center/a100/
    "A100-80GB": (312.0, 19.5, 2039.0),
    # H100 SXM: FP32 67 (datasheet), 3.35 TB/s. https://www.nvidia.com/en-us/data-center/h100/
    "H100": (989.0, 67.0, 3350.0),
    # H200 SXM: FP32 67 (datasheet), 4.8 TB/s. https://www.nvidia.com/en-us/data-center/h200/
    "H200": (989.0, 67.0, 4800.0),
    # HGX B200 (1,000 W): the datasheet gives no per-GPU FP32, only 600 TFLOPS for the 8-GPU board, so 75.0
    # is that divided by 8 (derived, not quoted). https://www.nvidia.com/en-us/data-center/hgx/
    "B200": (2250.0, 75.0, 7700.0),
}
# nvidia-smi names -> SPEC key. Anchored so lookalikes (A100 40GB/PCIe, H100 PCIe/NVL, H200 NVL, L40S, A10G)
# match nothing.
_SPEC_NAMES = (
    (re.compile(r"^(NVIDIA )?L4$"), "L4"),
    (re.compile(r"^(NVIDIA )?A100-SXM4-80GB$"), "A100-80GB"),
    (re.compile(r"^(NVIDIA )?H100 80GB HBM3$"), "H100"),
    (re.compile(r"^(NVIDIA )?H200$"), "H200"),
    (re.compile(r"^(NVIDIA )?B200$"), "B200"),
)


class Profiled(Protocol):
    def decide(self, state: str, questions: Mapping[str, object]) -> Mapping[str, object]: ...
    def profile_info(self) -> Mapping[str, object]: ...
    def take_forwards(self) -> list[Mapping[str, object]]: ...
    def count_flops(self, state: str, questions: Mapping[str, object]) -> float: ...


@dataclass(frozen=True)
class Question:
    cost: flops.Cost
    forwards: int
    outcome: jeval.Outcome
    counted_flops: float = math.nan  # FlopCounterMode, when this question was re-run under it
    real_tokens: float = math.nan  # tokens that were not padding (nan: the same as cost.tokens)


def options(row: jeval.Row) -> int:
    (q,) = row.questions.values()
    criteria = q.get("criteria") if isinstance(q, Mapping) else None
    return len(criteria) if isinstance(criteria, Mapping | list) else 2


def subset(rows: Sequence[jeval.Row], counter: TokenCounter, settings: Settings) -> list[jeval.Row]:
    max_tokens = get_int(settings, "roofline_max_tokens", 1024)
    max_options = get_int(settings, "roofline_max_options", 16)
    keep = [
        r
        for r in rows
        if r.qtype in TYPES
        and options(r) <= max_options
        and counter.count(r.state + json.dumps(r.questions, ensure_ascii=False)) <= max_tokens
    ]
    limit = get_int(settings, "roofline_limit", 200)
    return random.Random(get_int(settings, "roofline_seed", 0)).sample(keep, min(limit, len(keep)))


class FittingCounter(TokenCounter, Protocol):
    def fit(self, text: str, tokens: int) -> str: ...


def spread(rows: Sequence[jeval.Row], k: int, rng: random.Random) -> list[jeval.Row]:
    """`k` rows taken round-robin across datasets (`source`), each dataset shuffled by `rng` and the datasets
    visited in a seeded order: at most ceil(k / datasets) from each, the rest filled from the datasets that
    still have rows when a small one runs out."""
    groups: dict[str, list[jeval.Row]] = {}
    for r in rows:
        groups.setdefault(r.source, []).append(r)
    order = sorted(groups)
    rng.shuffle(order)
    for name in order:
        rng.shuffle(groups[name])
    out: list[jeval.Row] = []
    while len(out) < k and any(groups.values()):
        for name in order:
            if groups[name] and len(out) < k:
                out.append(groups[name].pop())
    return out


def length_rows(
    rows: Sequence[jeval.Row], counter: FittingCounter, settings: Settings, lengths: Sequence[int]
) -> list[tuple[int, jeval.Row]]:
    """(length, row) for every length in `lengths` x the same `roofline_per_length` seeded rows, states cut
    to its first `length` reference tokens, picked round-robin across datasets (`spread`; the per-dataset
    counts are noted as roofline_length_sources). Rows qualify when they are choice/noul with at most
    `roofline_max_options` options, their question JSON is at most `roofline_max_question_tokens` tokens, and
    their own state is at least max(lengths) tokens, so every point is a prefix of one real state. Ordered
    length-major (all questions at one length, then the next). Raises if too few rows qualify."""
    if not lengths:
        raise ValueError(f"{LENGTH_AXIS} is empty")
    per = get_int(settings, "roofline_per_length", 40)
    max_options = get_int(settings, "roofline_max_options", 16)
    max_question = get_int(settings, "roofline_max_question_tokens", 512)
    longest = max(lengths)
    keep = [
        r
        for r in rows
        if r.qtype in TYPES
        and options(r) <= max_options
        and counter.count(json.dumps(r.questions, ensure_ascii=False)) <= max_question
        and counter.count(r.state) >= longest
    ]
    if len(keep) < per:
        raise ValueError(
            f"only {len(keep)} JEVal rows have a state of at least {longest} reference tokens (and at most "
            f"{max_options} options, {max_question} question tokens); roofline_per_length = {per}"
        )
    chosen = spread(keep, per, random.Random(get_int(settings, "roofline_seed", 0)))
    counts: dict[str, int] = {}
    for r in chosen:
        counts[r.source] = counts.get(r.source, 0) + 1
    note("roofline_length_sources", "; ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    out: list[tuple[int, jeval.Row]] = []
    for n in lengths:
        for r in chosen:
            out.append((n, dataclasses.replace(r, state=counter.fit(r.state, n))))
    return out


def spec_peaks(gpu: str) -> tuple[float, float, float] | None:
    name = gpu.strip()
    return next((SPEC[key] for rx, key in _SPEC_NAMES if rx.match(name)), None)


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


def point(prefix: str, cost: flops.Cost, n: int) -> Metrics:
    """Achieved TFLOP/s, intensity (FLOP/byte) and bandwidth of a set of forwards."""
    return {
        f"{prefix}tflops": _ratio(cost.flops, cost.ms * 1e9),
        f"{prefix}intensity": _ratio(cost.flops, cost.bytes),
        f"{prefix}bandwidth_gbs": _ratio(cost.bytes, cost.ms * 1e6),
        f"{prefix}n": float(n),
    }


def metrics(
    questions: Sequence[Question],
    arch: flops.Arch,
    ceilings: Mapping[str, float],
    gpu: str,
    wall_s: float,
) -> Metrics:
    total = flops.ZERO
    for q in questions:
        total = total + q.cost
    m = point("roofline_", total, len(questions))
    m.pop("roofline_n")
    acc = jeval.metrics([q.outcome for q in questions], wall_s)
    m |= {"roofline" + k.removeprefix("jeval"): v for k, v in acc.items() if "score" not in k}
    m |= {
        "roofline_gflop_per_question": total.flops / 1e9 / len(questions),
        "roofline_mb_per_question": total.bytes / 1e6 / len(questions),
        "roofline_tokens_per_question": total.tokens / len(questions),
        "roofline_padding": 1
        - _ratio(
            sum(q.cost.tokens if math.isnan(q.real_tokens) else q.real_tokens for q in questions),
            total.tokens,
        ),
        "roofline_forwards_per_question": sum(q.forwards for q in questions) / len(questions),
        "roofline_gpu_share": _ratio(total.ms, sum(q.outcome.latency_ms for q in questions)),
        "roofline_params_b": arch.params / 1e9,
        "roofline_active_params_b": arch.active_params / 1e9,
        "roofline_attention_layers": sum(ly.mixer == "attention" for ly in arch.layers) / len(arch.layers),
    }
    m |= summarize("roofline_gpu_ms", [q.cost.ms for q in questions])
    lo = 0
    for hi in BINS:
        in_bin = [q for q in questions if lo < q.cost.tokens <= hi]
        if in_bin:
            cost = flops.ZERO
            for q in in_bin:
                cost = cost + q.cost
            m |= point(f"roofline_t{hi}_", cost, len(in_bin))
        lo = hi
    if counted := [q for q in questions if not math.isnan(q.counted_flops)]:
        m["roofline_counted_ratio"] = sum(q.counted_flops for q in counted) / sum(
            q.cost.flops for q in counted
        )
    m |= {f"roofline_{k}": float(v) for k, v in ceilings.items()}
    peak, bw = ceilings.get("peak_bf16_tflops"), ceilings.get("peak_bw_gbs")
    if peak is not None and bw is not None and peak > 0 and bw > 0:
        intensity = m["roofline_intensity"]
        attainable = min(peak, bw * intensity / 1e3)
        m |= {
            "roofline_ridge": peak * 1e3 / bw,
            "roofline_attainable_tflops": attainable,
            "roofline_efficiency": m["roofline_tflops"] / attainable,
            "roofline_mfu": m["roofline_tflops"] / peak,
        }
    if (spec := spec_peaks(gpu)) is not None:
        m |= {
            "roofline_spec_bf16_tflops": spec[0],
            "roofline_spec_fp32_tflops": spec[1],
            "roofline_spec_bw_gbs": spec[2],
        }
    return m


def select(settings: Settings) -> tuple[list[jeval.Row], list[jeval.Row], list[int]]:
    """(timed rows, warm-up rows, each timed row's state length or [] in natural mode)."""
    counter = _counter(settings)
    warmup = get_int(settings, "roofline_warmup", 8)
    if f"grid_{LENGTH_AXIS}" not in settings:
        rows = subset(jeval.load_rows(), counter, settings)
        return rows, rows[:warmup], []
    lengths = parse_int_list(settings, LENGTH_AXIS)
    tagged = length_rows(jeval.load_rows(), counter, settings, lengths)
    rows = [r for _, r in tagged]
    # every length warmed with its first WARM_PER_LENGTH questions: the first pass at a new shape pays one-off
    # costs (an MoE grouped GEMM picks its kernel per token count; one 512-token OLMoE question took 1.2 s on
    # an H100 on 2026-11-08 against ~45 ms for the rest), then the configured warm-up
    per = max(len(tagged) // len(lengths), 1)
    warm = [r for i, r in enumerate(rows) if i % per < WARM_PER_LENGTH] + rows[:warmup]
    return rows, warm, [n for n, _ in tagged]


def length_points(questions: Sequence[Question], lengths: Sequence[int]) -> Metrics:
    """roofline_s<L>_* for each state length: achieved TFLOP/s, intensity, bandwidth, n, mean prompt tokens
    and mean GPU ms of the questions run at that length (questions[i] ran at lengths[i])."""
    m: Metrics = {"roofline_lengths": float(len(set(lengths)))}
    for n in sorted(set(lengths)):
        at = [q for q, ln in zip(questions, lengths, strict=True) if ln == n]
        cost = flops.ZERO
        for q in at:
            cost = cost + q.cost
        m |= point(f"roofline_s{n}_", cost, len(at))
        m[f"roofline_s{n}_tokens"] = cost.tokens / len(at)
        m[f"roofline_s{n}_gpu_ms"] = cost.ms / len(at)
        ms = sorted(q.cost.ms for q in at)  # a one-off slow pass shows as max >> median
        m[f"roofline_s{n}_gpu_ms_median"] = (
            ms[len(ms) // 2] if len(ms) % 2 else (ms[len(ms) // 2 - 1] + ms[len(ms) // 2]) / 2
        )
        m[f"roofline_s{n}_gpu_ms_max"] = ms[-1]
    return m


def _profiled(model: object) -> Profiled:
    for attr in ("decide", "profile_info", "take_forwards", "count_flops"):
        if not callable(getattr(model, attr, None)):
            raise TypeError(f"roofline needs a profiled model with .{attr}(); got {type(model).__name__}")
    return model  # ty: ignore[invalid-return-type]


def _num(d: Mapping[str, object], key: str, kind: type[int] | type[float]) -> float:
    val = d[key]
    if (
        not isinstance(val, int | float)
        or isinstance(val, bool)
        or (kind is int and not isinstance(val, int))
    ):
        raise TypeError(f"{key!r} must be a{'n integer' if kind is int else ' number'}, got {val!r}")
    return val


def real_tokens(forwards: Sequence[Mapping[str, object]]) -> float:
    """Non-padding tokens of a question's forwards: `real_tokens` where reported, else rows x tokens."""
    return float(
        sum(
            _num(f, "real_tokens", int)
            if "real_tokens" in f
            else _num(f, "rows", int) * _num(f, "tokens", int)
            for f in forwards
        )
    )


def forward(f: Mapping[str, object]) -> flops.Forward:
    """A recorded forward (plain JSON from the recorder, maybe from another process) as a flops.Forward."""
    return flops.Forward(
        rows=int(_num(f, "rows", int)),
        tokens=int(_num(f, "tokens", int)),
        past=int(_num(f, "past", int)),
        kept=int(_num(f, "kept", int)),
        cached=bool(f["cached"]),
        ms=float(_num(f, "ms", float)),
    )


def _counter(settings: Settings) -> FittingCounter:
    path = get_str(settings, "roofline_tokenizer_file", "")
    return FileTokenCounter(str(env.resolve_path(path))) if path else WordCounter()


@benchmark("roofline")
def run(model: object, settings: Settings) -> Metrics:
    import time  # noqa: PLC0415

    prof = _profiled(model)
    info = prof.profile_info()
    config = info["config"]
    if not isinstance(config, Mapping):
        raise TypeError("profile_info()['config'] must be the model's HF config dict")
    head_bytes = int(_num({"head_bytes": flops.BF16_BYTES} | dict(info), "head_bytes", int))
    weight_bytes = int(_num({"weight_bytes": flops.BF16_BYTES} | dict(info), "weight_bytes", int))
    sizes = {"head_bytes": head_bytes, "weight_bytes": weight_bytes}
    raw = info.get("ceilings", {})
    if not isinstance(raw, Mapping):
        raise TypeError("profile_info()['ceilings'] must be a mapping")
    ceilings = {str(k): _num(raw, k, float) for k in raw}
    if f"grid_{QPR_AXIS}" in settings:  # the questions-per-request axis: benchmarks/roofline_q.py
        from . import roofline_q  # noqa: PLC0415 — it imports this module

        return roofline_q.run(prof, settings, config, ceilings, sizes)
    arch = flops.arch(config)

    rows, warm, row_lengths = select(settings)
    for row in warm:
        jeval.ask(prof, row)
    if callable(after_warmup := getattr(prof, "after_warmup", None)):
        after_warmup()
    prof.take_forwards()

    questions: list[Question] = []
    t0 = time.perf_counter()
    for row in rows:
        outcome = jeval.ask(prof, row)
        raw_fwd = prof.take_forwards()
        fwd = [forward(f) for f in raw_fwd]
        cost = flops.total(arch, fwd, **sizes)
        questions.append(Question(cost, len(fwd), outcome, real_tokens=real_tokens(raw_fwd)))
    wall_s = time.perf_counter() - t0

    for i, row in enumerate(rows[: get_int(settings, "roofline_counted", 8)]):
        try:
            counted = prof.count_flops(row.state, row.questions)
        except Exception:
            counted = math.nan
        prof.take_forwards()
        q = questions[i]
        questions[i] = Question(q.cost, q.forwards, q.outcome, counted, q.real_tokens)

    # skipped under `s1 ncu`: torch.profiler's CUPTI tracing and Nsight Compute must not share a process
    if rows and get_int(settings, "roofline_kernel_profile", 1) and not get_int(settings, "roofline_ncu", 0):
        record_kernels(prof, rows[0])

    gpu = str(settings.get("gpu", ""))
    m = metrics(questions, arch, ceilings, gpu, wall_s)
    if row_lengths:
        m |= length_points(questions, row_lengths)
    if n_hw := get_int(settings, "roofline_ncu", 0):
        m |= hardware_region(prof, rows[:n_hw], questions[:n_hw], arch, sizes)
    return m


def record_kernels(prof: Profiled, row: jeval.Row, top: int = 10) -> None:
    """Answer `row` once more under torch.profiler and note which CUDA kernels ran (after the timed passes, so
    tracing overhead never reaches a timing). PyTorch picks attention kernels per GPU, so compare these before
    comparing GPUs. Best effort: a backend whose kernels run in another process (elephant-1) notes
    "none seen"."""
    from .. import profiling  # noqa: PLC0415 — imports torch lazily inside

    try:
        events = profiling.profile_kernels(lambda: jeval.ask(prof, row))
    except Exception as e:  # a profiler failure must not lose the run
        note("roofline_kernels_error", f"{type(e).__name__}: {e}"[:300])
        return
    finally:
        prof.take_forwards()
    every = profiling.kernel_shares(
        events, top=None
    )  # attention can sit below the top 10 (MoE: expert GEMMs)
    note("roofline_kernels_seen", len(every))
    note("roofline_kernels_top", "; ".join(f"{s:.1%} {name[:120]}" for name, s in every[:top]) or "none seen")
    attention = profiling.attention_kernel(every)
    note("roofline_attention_kernel", attention or "none seen")
    if attention is not None:
        note("roofline_attention_share", round(dict(every)[attention], 4))


def hardware_region(
    prof: Profiled,
    rows: Sequence[jeval.Row],
    timed: Sequence[Question],
    arch: flops.Arch,
    sizes: Mapping[str, int],
) -> Metrics:
    """Answer `rows` again inside a CUDA profiler range (cudaProfilerStart/Stop) for `s1 ncu`: Nsight
    Compute, started with --profile-from-start off, reads hardware counters for exactly these kernels.

    Returns the theoretical cost of the profiled passes and the timed-pass GPU time of the same questions,
    which `s1 ncu` pairs with the counters (ncu times kernels at locked base clocks)."""
    profiler = getattr(prof, "profiler", None)
    if not callable(profiler):
        raise TypeError(f"roofline_ncu needs a model with .profiler(on); got {type(prof).__name__}")
    prof.take_forwards()
    profiler(True)
    try:
        for row in rows:
            jeval.ask(prof, row)
    finally:
        profiler(False)
    cost = flops.total(arch, [forward(f) for f in prof.take_forwards()], **sizes)
    return {
        "roofline_hw_questions": float(len(rows)),
        "roofline_hw_theory_gflop": cost.flops / 1e9,
        "roofline_hw_theory_mb": cost.bytes / 1e6,
        "roofline_hw_timed_ms": sum(q.cost.ms for q in timed),
    }
