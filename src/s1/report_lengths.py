"""The roofline benchmark's state-length axis in the report: one curve per model per (platform, GPU), and the
H200's speedup over the H100 at every length.

A length-axis run (`[grid] roofline_state_tokens`) answers the same questions with their state cut to each
length, and records `roofline_s<L>_*` per length. Each model's curve walks (intensity, achieved TFLOP/s)
from the shortest length to the longest, against that panel's roofs. Panels never mix platforms or GPUs,
and natural-mode runs never appear here (`report_roofline.panels` keeps them, this module only length runs).

Achieved TFLOP/s on a curve uses the **median** GPU time per question at that length (the record's
`roofline_s<L>_tflops` scaled by mean/median GPU ms): one-off first passes at a new shape (an MoE grouped
GEMM picking its kernel, a cuDNN plan) can make the mean several times the median at one length. The
tables show the median GPU ms.

The speedup table divides the H100's median GPU ms by the H200's, length by length, for every model with a
length run on both (same platform). Same compute, ~1.4x bandwidth: a memory-bound pass should run ~1.4x
faster on the H200, a compute-bound one ~1.0x, and an overhead-bound one ~1.0x at every length.
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

BENCH = rr.BENCH
_LEN = re.compile(r"^roofline_s(\d+)_gpu_ms_median$")


@dataclass(frozen=True)
class LengthPoint:
    length: int  # state tokens (reference tokenizer)
    tokens: float  # mean prompt tokens the model processed
    intensity: float  # FLOP/byte
    tflops: float  # achieved, at the median GPU time
    ms_median: float
    ms_mean: float
    ms_max: float


@dataclass(frozen=True)
class Curve:
    model: str
    variant: str
    run_id: str
    m: Metrics
    points: list[LengthPoint]

    @property
    def name(self) -> str:
        return f"{self.model} ({self.variant})" if self.variant else self.model

    @property
    def short(self) -> str:
        """A chart label: without ".zeroshot", the variant as "+fused"."""
        return self.model.removesuffix(".zeroshot") + (" +fused" if self.variant else "")

    @property
    def family(self) -> int:
        return rr.family(self.m, self.model)

    def at(self, length: int) -> LengthPoint | None:
        return next((p for p in self.points if p.length == length), None)


def _is_q_axis(r: RunRecord) -> bool:
    """A questions-per-request run also has the length axis in its grid; report_qpr draws it."""
    return bench_roofline.QPR_AXIS in r.grid


def lengths_of(m: Metrics) -> list[int]:
    return sorted(int(g.group(1)) for k in m if (g := _LEN.match(k)))


def length_points(m: Metrics) -> list[LengthPoint]:
    out = []
    for n in lengths_of(m):
        k = f"roofline_s{n}_"
        med, mean = m[k + "gpu_ms_median"], m.get(k + "gpu_ms", math.nan)
        if not med > 0 or not m.get(k + "tokens", 0.0) > 0:
            continue  # every question at this length failed (counted invalid): no timing to show
        scale = mean / med if med > 0 and math.isfinite(mean) else math.nan
        out.append(
            LengthPoint(
                n,
                m.get(k + "tokens", math.nan),
                m.get(k + "intensity", math.nan),
                m.get(k + "tflops", math.nan) * scale,
                med,
                mean,
                m.get(k + "gpu_ms_max", math.nan),
            )
        )
    return out


@dataclass(frozen=True)
class LengthPanel:
    gpu: str
    platform: str
    curves: list[Curve]

    @property
    def label(self) -> str:
        return f"{rr.gpu_name(self.gpu)} ({self.platform})"

    def _as_panel(self) -> rr.Panel:
        return rr.Panel(
            self.gpu, self.platform, [rr.Point(c.model, c.run_id, c.m, {}, c.variant) for c in self.curves]
        )

    @property
    def measured(self) -> rr.Roof:
        return self._as_panel().measured

    @property
    def datasheet(self) -> rr.Roof:
        return self._as_panel().datasheet

    @property
    def lengths(self) -> list[int]:
        return sorted({p.length for c in self.curves for p in c.points})


def panels(recs: Sequence[RunRecord]) -> list[LengthPanel]:
    """Latest length-axis run per (platform, GPU, model, variant), grouped by (GPU, platform) and ordered as
    the natural panels are (datasheet bf16 peak, then name, the cluster first)."""
    runs = sorted(
        (r for r in recs if BENCH in r.results and rr.is_length_axis(r) and not _is_q_axis(r)),
        key=lambda r: r.run_id,
    )
    latest: dict[tuple[str, str, str, str], RunRecord] = {}
    for r in runs:
        if bool(r.params.get("roofline_ncu", 0)):
            continue
        latest[
            (
                str(r.invariants.get("gpu", "?")),
                rr.platform(r.invariants),
                str(r.params.get("model", "?")),
                rr.variant(r),
            )
        ] = r
    grouped: dict[tuple[str, str], list[Curve]] = {}
    for (gpu, plat, model, var), r in sorted(latest.items()):
        m = r.results[BENCH]
        if pts := length_points(m):
            grouped.setdefault((gpu, plat), []).append(Curve(model, var, r.run_id, m, pts))
    out = [LengthPanel(gpu, plat, cs) for (gpu, plat), cs in grouped.items()]

    def order(p: LengthPanel) -> tuple[bool, float, str, bool, str]:
        peak = p.datasheet.bf16
        known = math.isfinite(peak)
        return (not known, peak if known else 0.0, p.gpu, p.platform != "cluster", p.platform)

    return sorted(out, key=order)


def _f(v: float, digits: int = 3) -> str:
    return rr.DASH if not math.isfinite(v) else f"{v:.{digits}g}"


def ms_table(panel: LengthPanel) -> tuple[list[str], list[list[str]]]:
    """Median GPU ms per question at each length (and intensity at the shortest and longest)."""
    lengths = panel.lengths
    header = ["Model", *(f"{n} tok" for n in lengths), "Intensity (first → last)", "Max / median, worst"]
    rows = []
    for c in sorted(panel.curves, key=lambda c: c.name):
        cells = [(_f(p.ms_median) if (p := c.at(n)) else rr.DASH) for n in lengths]
        first, last = c.points[0], c.points[-1]
        worst = max(
            (p.ms_max / p.ms_median for p in c.points if p.ms_median > 0 and math.isfinite(p.ms_max)),
            default=math.nan,
        )
        rows.append([c.name, *cells, f"{_f(first.intensity)} → {_f(last.intensity)}", _f(worst, 2)])
    return header, rows


def _spec_key(gpu: str) -> str | None:
    return next((key for rx, key in bench_roofline._SPEC_NAMES if rx.match(gpu.strip())), None)


@dataclass(frozen=True)
class Speedup:
    platform: str
    base: LengthPanel
    other: LengthPanel
    lengths: list[int]
    rows: list[tuple[str, int, list[float]]]  # (model name, family, base ms / other ms per length)
    shorts: tuple[str, ...] = ()  # chart labels, in row order

    @property
    def bw_ratio(self) -> float:
        """The other GPU's measured DRAM bandwidth over the base's (datasheet when not measured)."""
        a, b = self.base.measured.bw, self.other.measured.bw
        if not (math.isfinite(a) and math.isfinite(b)):
            a, b = self.base.datasheet.bw, self.other.datasheet.bw
        return b / a if a > 0 else math.nan


def speedups(panels_: Sequence[LengthPanel], base: str = "H100", other: str = "H200") -> list[Speedup]:
    """`other`'s speedup over `base` (median GPU ms ratio) per model and length, per platform with both."""
    out = []
    for plat in sorted({p.platform for p in panels_}):
        a = next((p for p in panels_ if p.platform == plat and _spec_key(p.gpu) == base), None)
        b = next((p for p in panels_ if p.platform == plat and _spec_key(p.gpu) == other), None)
        if a is None or b is None:
            continue
        lengths = sorted(set(a.lengths) & set(b.lengths))
        by_name = {c.name: c for c in b.curves}
        rows = []
        shorts: list[str] = []
        for ca in sorted(a.curves, key=lambda c: c.name):
            if (cb := by_name.get(ca.name)) is None:
                continue
            ratios = []
            for n in lengths:
                pa, pb = ca.at(n), cb.at(n)
                ratios.append(pa.ms_median / pb.ms_median if pa and pb and pb.ms_median > 0 else math.nan)
            rows.append((ca.name, ca.family, ratios))
            shorts.append(ca.short)
        if rows:
            out.append(Speedup(plat, a, b, lengths, rows, tuple(shorts)))
    return out


def speedup_table(sp: Speedup) -> tuple[list[str], list[list[str]]]:
    header = ["Model", *(f"{n} tok" for n in sp.lengths), "Mean"]
    rows = []
    for name, _, ratios in sp.rows:
        ok = [v for v in ratios if math.isfinite(v)]
        rows.append([name, *(_f(v, 3) for v in ratios), _f(sum(ok) / len(ok) if ok else math.nan, 3)])
    return header, rows


# ---------------------------------------------------------------- charts

W, H = rr.W, rr.H
LEFT, RIGHT, TOP, BOTTOM = rr.LEFT, 150, rr.TOP, rr.BOTTOM  # room right of the plot for curve labels


def shared_scale(panels_: Sequence[LengthPanel]) -> rr.Scale:
    xs = [p.intensity for lp in panels_ for c in lp.curves for p in c.points]
    ys = [p.tflops for lp in panels_ for c in lp.curves for p in c.points]
    ys += [v for lp in panels_ for v in (lp.measured.bf16, lp.datasheet.bf16)]
    xs, ys = [v for v in xs if v > 0 and math.isfinite(v)], [v for v in ys if v > 0 and math.isfinite(v)]
    if not xs or not ys:
        return rr.Scale(0, 4, -1, 3)
    return rr.Scale(
        math.floor(math.log10(min(xs))),
        max(math.ceil(math.log10(max(xs))), 3),
        math.floor(math.log10(min(ys))),
        math.ceil(math.log10(max(ys) * 1.3)),
    )


def length_chart(panel: LengthPanel, s: rr.Scale | None = None) -> str:
    """Each model's curve from its shortest to its longest state length, against this panel's roofs."""
    s = s or shared_scale([panel])
    meas, spec = panel.measured, panel.datasheet
    roofs = [
        rr._roof(s, spec.bf16, spec.bw, "spec", f"datasheet: {spec.bf16:.0f} TFLOP/s, {spec.bw:.0f} GB/s"),
        rr._roof(s, meas.bf16, meas.bw, "", f"measured: {meas.bf16:.0f} TFLOP/s bf16, {meas.bw:.0f} GB/s"),
    ]
    marks, labels = [], []
    for c in panel.curves:
        pts = [
            p for p in c.points if p.intensity > 0 and p.tflops > 0 and math.isfinite(p.intensity * p.tflops)
        ]
        if not pts:
            continue
        color = f"var(--fam-{c.family})"
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{s.x(p.intensity):.1f},{s.y(p.tflops):.1f}" for i, p in enumerate(pts)
        )
        marks.append(f'<path class="traj curve" stroke="{color}" d="{d}"/>')
        for i, p in enumerate(pts):
            tip = (
                f"{c.name}, state {p.length} tokens ({p.tokens:.0f} processed): {p.intensity:.0f} FLOP/B, "
                f"{p.tflops:.2f} TFLOP/s, median {p.ms_median:.1f} ms (run {c.run_id})"
            )
            r = 4 if i in (0, len(pts) - 1) else 2.5
            marks.append(
                f'<g class="pt"><title>{html.escape(tip)}</title>'
                f'<circle class="dot" cx="{s.x(p.intensity):.1f}" cy="{s.y(p.tflops):.1f}" r="{r}" '
                f'fill="{color}" '
                'stroke="var(--surface)" stroke-width="1"/></g>'
            )
        labels.append((s.x(pts[-1].intensity), s.y(pts[-1].tflops), c.short))
    present = sorted({c.family for c in panel.curves})
    legend = "".join(
        f'<span class="key"><span class="sw" style="background:var(--fam-{f})"></span>{rr.FAMILIES[f]}</span>'
        for f in present
    )
    lo, hi = panel.lengths[0], panel.lengths[-1]
    legend += (
        f'<span class="key">each curve: {lo} → {hi} state tokens (large dots at the ends)</span>'
        '<span class="key">TFLOP/s at the median GPU time per length</span>'
        '<span class="key">— measured roof · - - datasheet</span>'
    )
    title = f"State-length axis on {panel.label}: the same questions, state cut to each length"
    return (
        f'<figure class="card chart roof"><figcaption><strong>{html.escape(title)}</strong></figcaption>'
        f'<div class="legend">{legend}</div>'
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">'
        f"{rr._axes(s)}{''.join(roofs)}{''.join(marks)}{rr._labels(labels)}</svg></figure>"
    )


def speedup_chart(sp: Speedup) -> str:
    """Speedup by state length (log2 x), one line per model, with reference lines at 1.0x (no gain:
    compute- or overhead-bound) and at the bandwidth ratio (fully memory-bound)."""
    vals = [v for _, _, rs in sp.rows for v in rs if math.isfinite(v)]
    bw = sp.bw_ratio
    y0: float = float(min([0.6, *vals])) - 0.05
    y1: float = float(max([1.0, *vals, *([bw] if math.isfinite(bw) else [])])) + 0.1
    height: float = float(H)
    x0, x1 = math.log2(sp.lengths[0]), math.log2(sp.lengths[-1])
    left, right, top, bottom = 56, 190, 14, 46

    def x(n: float) -> float:
        return left + (math.log2(n) - x0) / max(x1 - x0, 1e-9) * (W - left - right)

    def y(v: float) -> float:
        return height - bottom - (v - y0) / (y1 - y0) * (height - top - bottom)

    out = []
    for n in sp.lengths:
        out.append(f'<line class="grid" x1="{x(n):.1f}" x2="{x(n):.1f}" y1="{top}" y2="{H - bottom}"/>')
        out.append(
            f'<text class="muted" x="{x(n):.1f}" y="{H - bottom + 16}" text-anchor="middle">{n}</text>'
        )
    tick = math.ceil(y0 * 10) / 10
    while tick <= y1:
        out.append(f'<line class="grid" x1="{left}" x2="{W - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>')
        out.append(
            f'<text class="muted" x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick:.1f}x</text>'
        )
        tick = round(tick + 0.1, 1)
    refs = [(1.0, "1.0x: no gain (compute- or overhead-bound)")]
    if math.isfinite(bw):
        refs.append((bw, f"{bw:.2f}x: bandwidth ratio (memory-bound)"))
    for v, text in refs:
        out.append(f'<path class="ceil spec" d="M{left},{y(v):.1f} L{W - right},{y(v):.1f}"/>')
        out.append(
            f'<text class="muted" x="{W - right - 4}" y="{y(v) - 6:.1f}" text-anchor="end">'
            f"{html.escape(text)}</text>"
        )
    labels = []
    for i, (name, fam, ratios) in enumerate(sp.rows):
        pts = [(n, v) for n, v in zip(sp.lengths, ratios, strict=True) if math.isfinite(v)]
        if not pts:
            continue
        color = f"var(--fam-{fam})"
        d = " ".join(f"{'M' if i == 0 else 'L'}{x(n):.1f},{y(v):.1f}" for i, (n, v) in enumerate(pts))
        out.append(f'<path class="traj curve" stroke="{color}" d="{d}"/>')
        for n, v in pts:
            tip = f"{name}, state {n} tokens: {v:.2f}x faster on {rr.gpu_name(sp.other.gpu)}"
            out.append(
                f'<g class="pt"><title>{html.escape(tip)}</title>'
                f'<circle class="dot" cx="{x(n):.1f}" cy="{y(v):.1f}" r="3" fill="{color}"/></g>'
            )
        labels.append((x(pts[-1][0]), y(pts[-1][1]), sp.shorts[i] if i < len(sp.shorts) else name))
    out.append(
        f'<text class="muted" x="{(left + W - right) / 2}" y="{H - 8}" text-anchor="middle">'
        "State length (reference tokens, log scale)</text>"
        f'<text class="muted" transform="translate(16 {(top + H - bottom) / 2}) rotate(-90)" '
        f'text-anchor="middle">{html.escape(rr.gpu_name(sp.other.gpu))} speedup (median GPU ms ratio)</text>'
    )
    title = f"{rr.gpu_name(sp.other.gpu)} vs {rr.gpu_name(sp.base.gpu)} ({sp.platform}) by state length"
    return (
        f'<figure class="card chart roof"><figcaption><strong>{html.escape(title)}</strong></figcaption>'
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">'
        f"{''.join(out)}{rr._labels(labels)}</svg></figure>"
    )
