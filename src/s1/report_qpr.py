"""The roofline benchmark's questions-per-request (Q) axis in the report: one chart per model, one table for
all.

A Q-axis run (`[grid] roofline_questions_per_request`, benchmarks/roofline_q.py) sends Q questions against
one state in a single request, at fixed state lengths, and records `roofline_s<L>_q<Q>_*` per point. The
chart plots GPU ms per question relative to Q = 1 against Q: a model that re-reads the state for every
question stays near 1.0 (the flat reference line), one that reads it once and shares it falls toward 1/Q
(the dotted curve). One line per (GPU, state length), colored by GPU (direct-labelled at the line's end),
solid for the longest length and dashed for the shortest. The table carries the numbers behind every line,
plus the achieved TFLOP/s at Q = 1 and at the largest Q (hypothesis H3 predicts it rises with Q for designs
that share the state).
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from . import report_roofline as rr
from .benchmarks import roofline as bench_roofline
from .record import Metrics, RunRecord

BENCH = "roofline"
_POINT = re.compile(r"^roofline_s(\d+)_q(\d+)_gpu_ms_median$")
GPU_ORDER = ("L4", "A100-80GB", "H100", "H200", "B200")  # fixed hue order (never by rank)
CSS = """
:root{--gpu-0:#2a78d6;--gpu-1:#eb6834;--gpu-2:#1baf7a;--gpu-3:#eda100;--gpu-4:#e87ba4;--gpu-5:#6250d6}
@media (prefers-color-scheme:dark){:root{--gpu-0:#3987e5;--gpu-1:#d95926;--gpu-2:#199e70;--gpu-3:#c98500;
--gpu-4:#d55181;--gpu-5:#9085e9}}
.roof .dash{stroke-dasharray:6 4}
"""
W, H = 820, 420


def is_q_axis(r: RunRecord) -> bool:
    return bench_roofline.QPR_AXIS in r.grid


@dataclass(frozen=True)
class QPoint:
    q: int
    ms_median: float  # GPU ms per request
    tflops: float
    latency_ms: float  # wall ms per request

    @property
    def ms_per_question(self) -> float:
        return self.ms_median / self.q


@dataclass(frozen=True)
class Line:
    gpu: str
    platform: str
    length: int
    run_id: str
    points: list[QPoint]
    marginal: float

    @property
    def gpu_slot(self) -> int:
        key = _spec_key(self.gpu)
        return GPU_ORDER.index(key) if key in GPU_ORDER else len(GPU_ORDER)

    def relative(self) -> list[tuple[int, float]]:
        base = self.points[0].ms_per_question
        return [(p.q, p.ms_per_question / base) for p in self.points if base > 0]


@dataclass(frozen=True)
class ModelQ:
    model: str
    mode: str
    lines: list[Line]


def _spec_key(gpu: str) -> str | None:
    return next((key for rx, key in bench_roofline._SPEC_NAMES if rx.match(gpu.strip())), None)


def points(m: Metrics, length: int) -> list[QPoint]:
    out = []
    for k in m:
        if (g := _POINT.match(k)) and int(g.group(1)) == length:
            q = int(g.group(2))
            p = f"roofline_s{length}_q{q}_"
            if m[k] > 0:
                out.append(QPoint(q, m[k], m.get(p + "tflops", math.nan), m.get(p + "latency_ms", math.nan)))
    return sorted(out, key=lambda p: p.q)


def lengths(m: Metrics) -> list[int]:
    return sorted({int(g.group(1)) for k in m if (g := _POINT.match(k))})


def mode_of(r: RunRecord) -> str:
    """How the model handles several questions (the spec's question_mode, and SemIf's readout mode)."""
    parts = [str(r.params[k]) for k in ("question_mode", "mode") if k in r.params]
    if "bongard_branch_batch" in r.params and str(r.params.get("model", "")).startswith("bongard"):
        parts.append(f"batch {r.params['bongard_branch_batch']}")
    return ", ".join(parts) or "?"


def models(recs: Sequence[RunRecord]) -> list[ModelQ]:
    """Latest clean Q-axis run per (model, platform, GPU), grouped by model."""
    runs = sorted((r for r in recs if BENCH in r.results and is_q_axis(r)), key=lambda r: r.run_id)
    latest: dict[tuple[str, str, str], RunRecord] = {}
    for r in runs:
        if not bool(r.params.get("roofline_ncu", 0)):
            latest[
                (
                    str(r.params.get("model", "?")),
                    rr.platform(r.invariants),
                    str(r.invariants.get("gpu", "?")),
                )
            ] = r
    grouped: dict[str, ModelQ] = {}
    for (model, plat, gpu), r in sorted(latest.items()):
        m = r.results[BENCH]
        mq = grouped.setdefault(model, ModelQ(model, mode_of(r), []))
        for n in lengths(m):
            if len(pts := points(m, n)) >= 2:
                mq.lines.append(
                    Line(gpu, plat, n, r.run_id, pts, m.get(f"roofline_s{n}_qmarginal", math.nan))
                )
    for mq in grouped.values():
        mq.lines.sort(key=lambda ln: (ln.gpu_slot, ln.gpu, ln.platform, ln.length))
    return list(grouped.values())


def _f(v: float, digits: int = 3) -> str:
    return rr.DASH if not math.isfinite(v) else f"{v:.{digits}g}"


def table(mqs: Sequence[ModelQ]) -> tuple[list[str], list[list[str]]]:
    header = [
        "Model", "Questions handled as", "GPU", "State tokens", "Q range",
        "GPU ms/question at Q min", "at Q max", "ratio", "Marginal question (slope / Q=1)",
        "TFLOP/s at Q min", "at Q max",
    ]  # fmt: skip
    rows = []
    for mq in mqs:
        for ln in mq.lines:
            lo, hi = ln.points[0], ln.points[-1]
            rows.append(
                [
                    mq.model,
                    mq.mode,
                    f"{rr.gpu_name(ln.gpu)} ({ln.platform})",
                    str(ln.length),
                    f"{lo.q}-{hi.q}",
                    _f(lo.ms_per_question),
                    _f(hi.ms_per_question),
                    _f(hi.ms_per_question / lo.ms_per_question if lo.ms_per_question > 0 else math.nan),
                    _f(ln.marginal, 2),
                    _f(lo.tflops),
                    _f(hi.tflops),
                ]
            )
    return header, rows


def chart(mq: ModelQ) -> str:
    """GPU ms per question relative to Q min, against Q, one line per (GPU, state length)."""
    rel = [v for ln in mq.lines for _, v in ln.relative() if v > 0 and math.isfinite(v)]
    qs = sorted({p.q for ln in mq.lines for p in ln.points})
    if not rel or len(qs) < 2:
        return ""
    q0, q1 = qs[0], qs[-1]
    y_lo = math.floor(math.log2(min([*rel, q0 / q1])))
    y_hi = max(math.ceil(math.log2(max(rel) * 1.1)), 1)
    left, right, top, bottom = 64, 150, 16, 48

    def x(q: float) -> float:
        return left + (q - q0) / max(q1 - q0, 1e-9) * (W - left - right)

    def y(v: float) -> float:
        return H - bottom - (math.log2(v) - y_lo) / max(y_hi - y_lo, 1e-9) * (H - top - bottom)

    out = []
    for q in qs:
        out.append(f'<line class="grid" x1="{x(q):.1f}" x2="{x(q):.1f}" y1="{top}" y2="{H - bottom}"/>')
        out.append(
            f'<text class="muted" x="{x(q):.1f}" y="{H - bottom + 16}" text-anchor="middle">{q}</text>'
        )
    for e in range(y_lo, y_hi + 1):
        v = 2.0**e
        label = f"{v:g}x" if v >= 1 else f"1/{1 / v:g}"
        out.append(f'<line class="grid" x1="{left}" x2="{W - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        out.append(f'<text class="muted" x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{label}</text>')
    out.append(f'<path class="ceil spec" d="M{left},{y(1):.1f} L{W - right},{y(1):.1f}"/>')
    share = " ".join(
        f"{'M' if i == 0 else 'L'}{x(q):.1f},{y(q0 / q):.1f}" for i, q in enumerate(range(q0, q1 + 1))
    )
    out.append(f'<path class="ceil spec" style="stroke-dasharray:1 4" d="{share}"/>')
    longest = max(ln.length for ln in mq.lines)
    labels = []
    for ln in mq.lines:
        pts = [(q, v) for q, v in ln.relative() if v > 0 and math.isfinite(v)]
        if not pts:
            continue
        color = f"var(--gpu-{ln.gpu_slot})"
        d = " ".join(f"{'M' if i == 0 else 'L'}{x(q):.1f},{y(v):.1f}" for i, (q, v) in enumerate(pts))
        dash = "" if ln.length == longest else " dash"
        out.append(f'<path class="traj curve{dash}" stroke="{color}" d="{d}"/>')
        by_q = {p.q: p for p in ln.points}
        for q, v in pts:
            p = by_q[q]
            tip = (
                f"{mq.model} on {rr.gpu_name(ln.gpu)} ({ln.platform}), state {ln.length} tokens, Q = {q}: "
                f"{p.ms_per_question:.2f} GPU ms/question ({v:.2f}x Q = {q0}), {p.tflops:.2f} TFLOP/s, "
                f"median {p.ms_median:.1f} GPU ms/request (run {ln.run_id})"
            )
            out.append(
                f'<g class="pt"><title>{html.escape(tip)}</title><circle class="dot" cx="{x(q):.1f}" '
                f'cy="{y(v):.1f}" r="4" fill="{color}" stroke="var(--surface)" stroke-width="2"/></g>'
            )
        labels.append((x(pts[-1][0]), y(pts[-1][1]), f"{rr.gpu_name(ln.gpu)} {ln.length}"))
    out.append(
        f'<text class="muted" x="{(left + W - right) / 2}" y="{H - 8}" text-anchor="middle">'
        "Questions per request (Q)</text>"
        f'<text class="muted" transform="translate(16 {(top + H - bottom) / 2}) rotate(-90)" '
        f'text-anchor="middle">GPU ms per question, relative to Q = {q0} (log scale)</text>'
    )
    gpus = sorted({(ln.gpu_slot, rr.gpu_name(ln.gpu)) for ln in mq.lines})
    legend = "".join(
        f'<span class="key"><span class="sw" style="background:var(--gpu-{s})"></span>{html.escape(g)}</span>'
        for s, g in gpus
    )
    legend += (
        f'<span class="key">solid: {longest}-token state · dashed: shorter</span>'
        '<span class="key">- - 1.0: every question re-reads the state</span>'
        '<span class="key">··· 1/Q: the state is read once</span>'
    )
    title = f"Q axis, {mq.model} ({mq.mode}): GPU cost per question as questions share one state"
    return (
        f'<figure class="card chart roof"><figcaption><strong>{html.escape(title)}</strong></figcaption>'
        f'<div class="legend">{legend}</div>'
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">'
        f"{''.join(out)}{rr._labels(labels)}</svg></figure>"
    )
