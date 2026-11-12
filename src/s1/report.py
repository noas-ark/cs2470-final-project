"""Summarize every run record into results/REPORT.md (renders on GitHub) and results/report.html
(self-contained page with charts). Output depends only on the records, so regenerating without new
runs changes nothing; the pre-commit hook (scripts/hooks/pre-commit) runs `s1 report` every commit.

Benchmark-agnostic: a benchmark's columns are its first metric (the headline) plus whichever of the
conventional `<bench>_<suffix>` keys it reports (see AGENTS.md "Adding a benchmark"). Every table names the
platform a run was on ("cluster" when `invariants.platform` is absent, else e.g. "Modal"), and run details add
a telemetry digest (`telemetry()`) for records that carry the newer hardware fields.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import report_batch, report_lengths, report_qpr, report_roofline
from .record import RESULTS_DIR, Metrics, RunRecord, load_all

SUFFIXES = ("accuracy", "ece", "brier", "invalid", "n", "latency_ms", "latency_ms_p99", "questions_per_s")
LABELS = {
    "accuracy": "Accuracy",
    "ece": "ECE",
    "brier": "Brier",
    "invalid": "Invalid",
    "n": "N",
    "latency_ms": "Latency mean",
    "latency_ms_p99": "Latency p99",
    "questions_per_s": "q/s",
}
DASH, TIMES = "\u2013", "\u00d7"  # placeholder for missing values; "A10G x4"
GPU_COLS = (
    ("gpu_util_pct", "GPU util"),
    ("gpu_mem_used_mb_max", "GPU mem peak"),
    ("gpu_energy_wh", "Energy"),
)


FORMATS: list[tuple[Callable[[str], bool], Callable[[float], str]]] = [
    (lambda k: k.endswith("accuracy"), lambda v: f"{v:.1%}"),
    (
        lambda k: k.endswith(("_n", "invalid", "count", "samples", "skipped", "reps", "tokens_actual")),
        lambda v: f"{v:,.0f}",
    ),
    (lambda k: "latency_ms" in k, lambda v: f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"),
    (lambda k: k.endswith("_pct"), lambda v: f"{v:.1f}%"),
    (lambda k: k.endswith(("_mb_max", "_mb_mean")), lambda v: f"{v / 1024:.1f} GiB"),
    (lambda k: k.endswith("_wh"), lambda v: f"{v:,.0f} Wh"),
]


def fmt(key: str, val: float) -> str:
    if math.isnan(val):
        return DASH
    return next((f(val) for match, f in FORMATS if match(key)), f"{val:.3g}")


@dataclass(frozen=True)
class Table:
    header: list[str]
    rows: list[list[str]]


def benchmarks(recs: Sequence[RunRecord]) -> list[str]:
    return sorted({b for r in recs for b in r.results if "/" not in b})


def metric_keys(bench: str, recs: Sequence[RunRecord]) -> list[str]:
    present: dict[str, None] = {}
    for r in recs:
        present |= dict.fromkeys(r.results.get(bench, {}))
    first = next(iter(present), None)
    keys = [first] if first is not None else []
    keys += [k for s in SUFFIXES if (k := f"{bench}_{s}") in present and k not in keys]
    return keys


def label(bench: str, key: str) -> str:
    return LABELS.get(key.removeprefix(f"{bench}_"), key.removeprefix(f"{bench}_"))


def model_of(r: RunRecord) -> str:
    return str(r.params.get("model", "?"))


def platform(r: RunRecord) -> str:
    """ "cluster" (no `invariants.platform`: the HUIT cluster) or "Modal"."""
    return report_roofline.platform(r.invariants)


def hardware(r: RunRecord, bench: str) -> str:
    gpu = str(r.invariants.get("gpu", "?")).removeprefix("NVIDIA ")
    count = r.results.get(f"{bench}/nvidia_smi", {}).get("gpu_count")
    hw = f"{gpu} {TIMES}{count:.0f}" if count is not None else gpu
    quant = r.invariants.get("quantization")
    return f"{hw}, {quant}" if quant is not None else hw


def other_params(r: RunRecord) -> str:
    items = [f"{k}={v}" for k, v in sorted(r.params.items()) if k != "model"]
    items += [f"{k}={'|'.join(map(str, v))}" for k, v in sorted(r.grid.items())]
    return ", ".join(items) or DASH


def lower_is_better(head: str) -> bool:
    """Latency-like headlines (key contains `latency`) rank ascending; everything else descending."""
    return "latency" in head


def best_per_model(bench: str, recs: Sequence[RunRecord]) -> list[RunRecord]:
    """Each model's run with the most questions (`<bench>_n`), latest on ties; sorted by headline."""
    keys = metric_keys(bench, recs)
    best: dict[str, RunRecord] = {}
    for r in recs:
        m = r.results.get(bench)
        if m is None:
            continue
        cur = best.get(model_of(r))
        size = (m.get(f"{bench}_n", 0.0), r.run_id)
        if cur is None or size > (cur.results[bench].get(f"{bench}_n", 0.0), cur.run_id):
            best[model_of(r)] = r
    head = keys[0] if keys else ""
    sign = 1 if lower_is_better(head) else -1

    def rank(r: RunRecord) -> float:
        v = r.results[bench].get(head, math.nan)
        return math.inf if math.isnan(v) else sign * v  # missing headline sorts last

    return sorted(best.values(), key=rank)


def leaderboard(bench: str, recs: Sequence[RunRecord]) -> Table:
    keys = metric_keys(bench, recs)
    rows = []
    for r in best_per_model(bench, recs):
        m = r.results[bench]
        rows.append(
            [
                model_of(r),
                *(fmt(k, m[k]) if k in m else DASH for k in keys),
                platform(r),
                hardware(r, bench),
                r.run_id,
            ]
        )
    return Table(["Model", *(label(bench, k) for k in keys), "Platform", "Hardware", "Run"], rows)


def all_runs(bench: str, recs: Sequence[RunRecord]) -> Table:
    keys = metric_keys(bench, recs)
    rows = []
    for r in sorted((r for r in recs if bench in r.results), key=lambda r: r.run_id, reverse=True):
        m, g = r.results[bench], r.results.get(f"{bench}/nvidia_smi", {})
        rows.append(
            [
                r.run_id,
                model_of(r),
                other_params(r),
                *(fmt(k, m[k]) if k in m else DASH for k in keys),
                *(fmt(k, g[k]) if k in g else DASH for k, _ in GPU_COLS),
                platform(r),
                hardware(r, bench),
                str(r.meta.get("git_commit", "?")),
            ]
        )
    header = ["Run", "Model", "Params", *(label(bench, k) for k in keys), *(n for _, n in GPU_COLS)]
    return Table([*header, "Platform", "Hardware", "Commit"], rows)


def run_detail(r: RunRecord) -> list[tuple[str, Metrics]]:
    return sorted(r.results.items())


def _num(v: float, unit: str = "", digits: int = 0) -> str:
    return f"{v:,.{digits}f}{unit}"


def _gpu_line(m: Metrics, g: str) -> str:
    """One GPU's detailed (schema-1) means; `g` is "" for the pooled keys or "<index>" per GPU."""
    k = f"gpu{g}_"
    parts = []
    if (v := m.get(f"{k}util_pct")) is not None and g:
        parts.append(f"util {_num(v, '%', 1)}")
    if (v := m.get(f"{k}mem_util_pct")) is not None:
        parts.append(f"memory controller busy {_num(v, '%', 1)}")
    if (v := m.get(f"{k}clock_sm_mhz")) is not None:
        low = m.get(f"{k}clock_sm_mhz_min")
        parts.append(f"SM {_num(v, ' MHz')}" + (f" (min {_num(low, ' MHz')})" if low is not None else ""))
    if (v := m.get(f"{k}clock_mem_mhz")) is not None:
        parts.append(f"memory {_num(v, ' MHz')}")
    if (v := m.get(f"{k}throttle_frac")) is not None:
        mask = m.get(f"{k}throttle_mask_seen")
        seen = f", reasons 0x{int(mask):x}" if mask is not None and mask > 0 else ""
        parts.append(f"throttled {v:.0%} of samples{seen}")
    total = "_total_mean" if not g else ""
    rx, tx = m.get(f"{k}pcie_rx_mb_s{total}"), m.get(f"{k}pcie_tx_mb_s{total}")
    if rx is not None or tx is not None:
        rate = [DASH if v is None else _num(v, " MB/s") for v in (rx, tx)]
        parts.append(f"PCIe rx {rate[0]}, tx {rate[1]}")
    if not g and (v := m.get("gpu_util_pct_spread")) is not None:
        parts.append(f"util spread across GPUs {_num(v, ' pts', 1)}")
    return "; ".join(parts)


def telemetry(r: RunRecord) -> list[tuple[str, str, bool]]:
    """Readable digest of the hardware telemetry newer records carry: (label, value, value is code).
    Empty for records without any of it, so older records render as before. Covers the CPU (meta
    cpu_model, cpus_requested, cpu_count), the detailed GPU metrics (`gpu_metrics = "detailed"`), and the
    CUDA kernels a roofline run noted (meta roofline_kernels_top / roofline_attention_kernel)."""
    rows: list[tuple[str, str, bool]] = []
    cpu = r.meta.get("cpu_model")
    if cpu is not None and str(cpu) not in ("", "unknown"):  # "unknown": early Modal records' placeholder
        rows.append(("CPU", str(cpu), False))
    cpus = [
        f"{r.meta[k]} {what}" for k, what in (("cpus_requested", "requested"), ("cpu_count", "visible"))
        if k in r.meta
    ]  # fmt: skip
    if cpus:
        rows.append(("CPUs", ", ".join(cpus), False))
    for name, m in run_detail(r):
        if not name.endswith("/nvidia_smi"):
            continue
        where = (
            f" ({name.removesuffix('/nvidia_smi')})"
            if sum(n.endswith("/nvidia_smi") for n in r.results) > 1
            else ""
        )
        indices = sorted(
            {int(i) for k in m if k.startswith("gpu") and (i := k[3:].partition("_")[0]).isdigit()}
        )
        if line := _gpu_line(m, ""):
            rows.append((f"{'GPUs, pooled' if len(indices) > 1 else 'GPU'}{where}", line, False))
        if len(indices) > 1:
            rows += [(f"GPU {i}{where}", line, False) for i in indices if (line := _gpu_line(m, str(i)))]
    if (seen := r.meta.get("roofline_kernels_seen")) is not None:
        rows.append(("CUDA kernels seen", str(seen), False))
    if (top := r.meta.get("roofline_kernels_top")) is not None:
        if str(top) == "none seen":
            rows.append(("Top CUDA kernels", "none seen", False))
        for i, item in enumerate((k for k in str(top).split("; ") if k and str(top) != "none seen"), 1):
            share, _, name = item.partition(" ")  # "41.2% <kernel name>"
            rows.append((f"Kernel {i} ({share} of GPU time)", name, True))
    if (att := r.meta.get("roofline_attention_kernel")) is not None:
        share = r.meta.get("roofline_attention_share")
        rows.append(("Attention kernel", str(att), str(att) != "none seen"))
        if share is not None:
            rows.append(("Attention kernel share", f"{float(share):.1%}", False))
    return rows


# ---------------------------------------------------------------- markdown


def _md_table(t: Table) -> str:
    def esc(s: str) -> str:
        return s.replace("|", "\\|")

    lines = ["| " + " | ".join(t.header) + " |", "|" + "---|" * len(t.header)]
    lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in t.rows]
    return "\n".join(lines)


def _md_lengths(recs: Sequence[RunRecord]) -> list[str]:
    """The state-length axis: median GPU ms by length per panel, then the H200-vs-H100 speedups."""
    lpanels = report_lengths.panels(recs)
    out: list[str] = []
    for sp in report_lengths.speedups(lpanels):
        name = f"{report_roofline.gpu_name(sp.other.gpu)} vs {report_roofline.gpu_name(sp.base.gpu)}"
        out += [
            f"### {name} speedup by state length ({sp.platform})",
            "",
            f"Median GPU ms on the {report_roofline.gpu_name(sp.base.gpu)} over the "
            f"{report_roofline.gpu_name(sp.other.gpu)}, same questions at each length. Memory-bound: "
            f"~{sp.bw_ratio:.2f}x (bandwidth ratio); compute- or overhead-bound: ~1.0x.",
            "",
            _md_table(Table(*report_lengths.speedup_table(sp))),
            "",
        ]
    for lp in lpanels:
        out += [
            f"### State-length axis on {lp.label}: median GPU ms per question",
            "",
            _md_table(Table(*report_lengths.ms_table(lp))),
            "",
        ]
    return out


def _html_lengths(recs: Sequence[RunRecord]) -> list[str]:
    lpanels = report_lengths.panels(recs)
    parts: list[str] = []
    for sp in report_lengths.speedups(lpanels):
        h, rows = report_lengths.speedup_table(sp)
        parts += [report_lengths.speedup_chart(sp), '<div class="card">',
                  _html_table(Table(h, rows), 1, len(h)), "</div>"]  # fmt: skip
    scale = report_lengths.shared_scale(lpanels)
    for lp in lpanels:
        h, rows = report_lengths.ms_table(lp)
        parts += [report_lengths.length_chart(lp, scale), '<div class="card">',
                  _html_table(Table(h, rows), 1, len(h) - 2), "</div>"]  # fmt: skip
    return parts


BATCH_NOTE = (
    "Hypothesis H2 (docs/h2_batch_findings.md): B independent requests per forward, newest record per GPU "
    "and series, `s1 nsys` records left out. TFLOP/s is cost-model FLOPs over wall time; CPU share is "
    "1 - GPU kernel time / wall time at batch 1. The eager batch-1 floor depends on the host CPU, "
    "listed per row."
)


def _md_batch(recs: Sequence[RunRecord]) -> list[str]:
    ps = report_batch.panels(recs)
    if not ps:
        return []
    return [
        BATCH_NOTE,
        "",
        "### Batch 1: CPU vs GPU share of latency",
        "",
        _md_table(Table(*report_batch.b1_table(ps))),
        "",
        "### Achieved TFLOP/s vs batch size",
        "",
        _md_table(Table(*report_batch.curve_table(ps))),
        "",
    ]


def _md_qpr(recs: Sequence[RunRecord]) -> list[str]:
    mqs = report_qpr.models(recs)
    if not mqs:
        return []
    return [
        "### Questions per request: GPU cost per question as questions share one state",
        "",
        _md_table(Table(*report_qpr.table(mqs))),
        "",
    ]


def _html_batch(recs: Sequence[RunRecord]) -> list[str]:
    ps = report_batch.panels(recs)
    if not ps:
        return []
    bh, br = report_batch.b1_table(ps)
    ch, cr = report_batch.curve_table(ps)
    return [
        f'<p class="sub">{html.escape(BATCH_NOTE)}</p>',
        "<h3>Batch 1: CPU vs GPU share of latency</h3>",
        report_batch.split_chart(ps),
        '<div class="card">', _html_table(Table(bh, br), 3, len(bh)), "</div>",
        "<h3>Achieved TFLOP/s vs batch size</h3>",
        report_batch.curve_charts(ps),
        '<div class="card">', _html_table(Table(ch, cr), 2, len(ch)), "</div>",
    ]  # fmt: skip


def _html_qpr(recs: Sequence[RunRecord]) -> list[str]:
    mqs = report_qpr.models(recs)
    if not mqs:
        return []
    parts = [report_qpr.chart(mq) for mq in mqs]
    h, rows = report_qpr.table(mqs)
    return [
        *parts,
        "<h3>Questions per request</h3>",
        '<div class="card">',
        _html_table(Table(h, rows), 5, len(h)),
        "</div>",
    ]


def to_markdown(recs: Sequence[RunRecord], results_dir: Path = RESULTS_DIR) -> str:
    out = [
        "# Results",
        "",
        f"Generated by `uv run s1 report` from {len(recs)} run record(s) in `results/`. Do not edit by hand.",
        "A browsable version with charts is [`report.html`](report.html).",
        "",
    ]
    for bench in benchmarks(recs):
        out += [f"## {bench}", ""]
        if bench == report_roofline.BENCH:
            if text := report_roofline.findings(results_dir):
                out += ["### Findings", "", text, ""]
            panels = report_roofline.panels(recs)
            out += ["### Roofs", "", _md_table(Table(*report_roofline.roofs(panels))), ""]
            if len(panels) > 1:
                out += [
                    "### Achieved TFLOP/s by GPU (% of that GPU's roof)",
                    "",
                    _md_table(Table(*report_roofline.by_gpu(panels))),
                    "",
                ]
            for panel in panels:
                out += [
                    f"### Decision recipes on {panel.label}",
                    "",
                    _md_table(Table(*report_roofline.recipes(panel.points))),
                    "",
                ]
                out += [
                    f"### Roofline on {panel.label}",
                    "",
                    _md_table(Table(*report_roofline.table(panel.points))),
                    "",
                ]
            out += _md_lengths(recs)
            out += _md_qpr(recs)
        if bench == report_batch.BENCH:
            out += _md_batch(recs)
        out += [
            "### Leaderboard (each model's largest run)",
            "",
            _md_table(leaderboard(bench, recs)),
        ]
        out += ["", "### All runs", "", _md_table(all_runs(bench, recs)), ""]
    out += ["## Run details", ""]
    for r in sorted(recs, key=lambda r: r.run_id, reverse=True):
        out += [
            f"<details><summary><code>{r.run_id}</code> · {model_of(r)} · {other_params(r)}</summary>",
            "",
        ]
        out += [
            _md_table(Table(["Setting", "Value"], [[k, str(v)] for k, v in sorted(r.invariants.items())]))
        ]
        if tele := telemetry(r):
            rows = [[k, f"`{v}`" if code else v] for k, v, code in tele]
            out += ["", "**Telemetry**", "", _md_table(Table(["Telemetry", "Value"], rows))]
        for name, metrics in run_detail(r):
            out += [
                "",
                f"**{name}**",
                "",
                _md_table(Table(["Metric", "Value"], [[k, f"{v:.6g}"] for k, v in metrics.items()])),
            ]
        meta = ", ".join(f"{k}={v}" for k, v in r.meta.items())
        out += ["", f"<sub>{meta}</sub>", "", "</details>", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- html

CSS = """
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--rule:#e4e3df;--bar:#2a78d6;
  --track:#efeeea;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink-2:#c3c2b7;
  --rule:#383835;--bar:#3987e5;--track:#262624;color-scheme:dark}}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 8px}
h3{font-size:15px;margin:24px 0 8px;color:var(--ink-2)}
p.sub{color:var(--ink-2);margin:0 0 16px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:16px;margin:8px 0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}
th{color:var(--ink-2);font-weight:600;font-size:12px}
td.num,th.num{text-align:right}
code{font:12px ui-monospace,monospace}
details{margin:6px 0}summary{cursor:pointer}
.chart svg{display:block;width:100%;max-width:820px;height:auto}
.chart text{fill:var(--ink);font:12px system-ui,sans-serif}.chart .muted{fill:var(--ink-2)}
.chart .bar{fill:var(--bar)}.chart .track{fill:var(--track)}
.chart g.mark:hover .bar{opacity:.8}
.prose{line-height:1.6}.prose p,.prose ul{max-width:90ch}.prose li{margin:4px 0}.prose h4{margin:18px 0 6px}
.recipes td{white-space:normal;min-width:7em;vertical-align:top}
"""


def _cell(s: str, numeric: bool) -> str:
    return f'<td class="num">{html.escape(s)}</td>' if numeric else f"<td>{html.escape(s)}</td>"


def _html_table(t: Table, numeric_from: int, numeric_to: int) -> str:
    def is_num(i: int) -> bool:
        return numeric_from <= i < numeric_to

    head = "".join(f'<th class="num">{html.escape(h)}</th>' if is_num(i) else f"<th>{html.escape(h)}</th>"
                   for i, h in enumerate(t.header))  # fmt: skip
    body = "".join(
        "<tr>" + "".join(_cell(c, is_num(i)) for i, c in enumerate(row)) + "</tr>" for row in t.rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _bar_chart(bench: str, recs: Sequence[RunRecord]) -> str:
    """Horizontal bars of the headline metric, one per model (single series: one hue, no legend)."""
    keys = metric_keys(bench, recs)
    runs = best_per_model(bench, recs)
    if not keys or not runs:
        return ""
    head = keys[0]
    vals = [r.results[bench].get(head, 0.0) for r in runs]
    finite = [v for v in vals if math.isfinite(v)]
    vals = [v if math.isfinite(v) else 0.0 for v in vals]
    vmax = 1.0 if head.endswith("accuracy") else max([*finite, 1e-9])
    label_w, chart_w, value_w, row_h, bar_h = 180, 520, 70, 34, 20
    width, height = label_w + chart_w + value_w, row_h * len(runs) + 8
    marks = []
    for i, (r, v) in enumerate(zip(runs, vals, strict=True)):
        y = 4 + i * row_h
        w = max(chart_w * v / vmax, 2)
        n = fmt("_n", r.results[bench].get(f"{bench}_n", float("nan")))
        tip = f"{model_of(r)}: {label(bench, head)} {fmt(head, v)} (n={n}, run {r.run_id})"
        bar_y, end = y + (row_h - bar_h) / 2, max(w - 4, 0)
        marks.append(
            f'<g class="mark"><title>{html.escape(tip)}</title>'
            f'<rect x="0" y="{y}" width="{width}" height="{row_h}" fill="transparent"/>'
            f'<text x="{label_w - 10}" y="{y + row_h / 2 + 4}" text-anchor="end">'
            f"{html.escape(model_of(r))}</text>"
            f'<rect class="track" x="{label_w}" y="{bar_y}" width="{chart_w}" height="{bar_h}" rx="4"/>'
            # 4px rounded data end, square at the baseline
            f'<path class="bar" d="M{label_w},{bar_y}h{end:.1f}a4,4 0 0 1 4,4'
            f'v{bar_h - 8}a4,4 0 0 1 -4,4h{-end:.1f}z"/>'
            f'<text x="{label_w + w + 8:.1f}" y="{y + row_h / 2 + 4}">{html.escape(fmt(head, v))}</text></g>'
        )
    note = ", lower is better" if lower_is_better(head) else ""
    title = f"{label(bench, head)} by model ({bench}, each model's largest run{note})"
    return (
        f'<figure class="card chart"><figcaption><strong>{html.escape(title)}</strong></figcaption>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f"{''.join(marks)}</svg>"
        "</figure>"
    )


def to_html(recs: Sequence[RunRecord], results_dir: Path = RESULTS_DIR) -> str:
    parts = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Decision-model results</title>",
        f"<style>{CSS}{report_roofline.CSS}{report_batch.CSS}{report_qpr.CSS}</style></head><body><main>",
        "<h1>Decision-model results</h1>",
        f'<p class="sub">Generated by <code>uv run s1 report</code> from {len(recs)} run record(s) in '
        "<code>results/</code>. Do not edit by hand.</p>",
    ]
    for bench in benchmarks(recs):
        n_keys = len(metric_keys(bench, recs))
        parts += [f"<h2>{html.escape(bench)}</h2>", _bar_chart(bench, recs)]
        if bench == report_roofline.BENCH:
            if text := report_roofline.findings(results_dir):
                parts += [
                    '<h3>Findings</h3><div class="card prose">',
                    report_roofline.markdown_to_html(text),
                    "</div>",
                ]
            panels = report_roofline.panels(recs)
            scale = report_roofline.shared_scale(panels)
            fh, fr = report_roofline.roofs(panels)
            parts += [
                "<h3>Roofs</h3>",
                '<div class="card">',
                _html_table(Table(fh, fr), 1, len(fh)),
                "</div>",
            ]
            if len(panels) > 1:
                gh, gr = report_roofline.by_gpu(panels)
                parts += ["<h3>Achieved TFLOP/s by GPU (% of that GPU's roof)</h3>", '<div class="card">',
                          _html_table(Table(gh, gr), 1, len(gh)), "</div>"]  # fmt: skip
            for panel in panels:
                rh, rr = report_roofline.recipes(panel.points)
                name = html.escape(panel.label)
                parts += [f"<h3>Decision recipes on {name}</h3>", '<div class="card recipes">',
                          _html_table(Table(rh, rr), 5, len(rh)), "</div>"]  # fmt: skip
                header, rows = report_roofline.table(panel.points)
                parts += [report_roofline.chart(panel, scale), '<div class="card">',
                          _html_table(Table(header, rows), 2, len(header)), "</div>"]  # fmt: skip
            parts += _html_lengths(recs)
            parts += _html_qpr(recs)
        if bench == report_batch.BENCH:
            parts += _html_batch(recs)
        parts += ["<h3>Leaderboard (each model's largest run)</h3>", '<div class="card">',
                  _html_table(leaderboard(bench, recs), 1, 1 + n_keys), "</div>"]  # fmt: skip
        parts += ["<h3>All runs</h3>", '<div class="card">',
                  _html_table(all_runs(bench, recs), 3, 3 + n_keys + len(GPU_COLS)), "</div>"]  # fmt: skip
    parts.append("<h2>Run details</h2>")
    for r in sorted(recs, key=lambda r: r.run_id, reverse=True):
        rows = [[k, str(v)] for k, v in sorted(r.invariants.items())]
        rows += [[f"telemetry · {k}", v] for k, v, _ in telemetry(r)]
        rows += [[f"{name} · {k}", f"{v:.6g}"] for name, m in run_detail(r) for k, v in m.items()]
        rows += [[f"meta · {k}", str(v)] for k, v in r.meta.items()]
        parts += [
            f"<details class=card><summary><code>{html.escape(r.run_id)}</code> · "
            f"{html.escape(model_of(r))} · {html.escape(other_params(r))}</summary>",
            _html_table(Table(["Key", "Value"], rows), 1, 2),
            "</details>",
        ]
    parts.append("</main></body></html>")
    return "\n".join(parts) + "\n"


def write(results_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    recs = load_all(results_dir)
    md, page = results_dir / "REPORT.md", results_dir / "report.html"
    md.write_text(to_markdown(recs, results_dir) + "\n")
    page.write_text(to_html(recs, results_dir))
    return md, page
