"""Summarize every run record into results/REPORT.md (renders on GitHub) and results/report.html
(self-contained page with charts). Output depends only on the records, so regenerating without new
runs changes nothing; the pre-commit hook (scripts/hooks/pre-commit) runs `s1 report` every commit.

Benchmark-agnostic: a benchmark's columns are its first metric (the headline) plus whichever of the
conventional `<bench>_<suffix>` keys it reports (see AGENTS.md "Adding a benchmark").
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

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
            [model_of(r), *(fmt(k, m[k]) if k in m else DASH for k in keys), hardware(r, bench), r.run_id]
        )
    return Table(["Model", *(label(bench, k) for k in keys), "Hardware", "Run"], rows)


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
                hardware(r, bench),
                str(r.meta.get("git_commit", "?")),
            ]
        )
    header = ["Run", "Model", "Params", *(label(bench, k) for k in keys), *(n for _, n in GPU_COLS)]
    return Table([*header, "Hardware", "Commit"], rows)


def run_detail(r: RunRecord) -> list[tuple[str, Metrics]]:
    return sorted(r.results.items())


# ---------------------------------------------------------------- markdown


def _md_table(t: Table) -> str:
    def esc(s: str) -> str:
        return s.replace("|", "\\|")

    lines = ["| " + " | ".join(t.header) + " |", "|" + "---|" * len(t.header)]
    lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in t.rows]
    return "\n".join(lines)


def to_markdown(recs: Sequence[RunRecord]) -> str:
    out = [
        "# Results",
        "",
        f"Generated by `uv run s1 report` from {len(recs)} run record(s) in `results/`. Do not edit by hand.",
        "A browsable version with charts is [`report.html`](report.html).",
        "",
    ]
    for bench in benchmarks(recs):
        out += [
            f"## {bench}",
            "",
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


def to_html(recs: Sequence[RunRecord]) -> str:
    parts = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>Decision-model results</title><style>{CSS}</style></head><body><main>",
        "<h1>Decision-model results</h1>",
        f'<p class="sub">Generated by <code>uv run s1 report</code> from {len(recs)} run record(s) in '
        "<code>results/</code>. Do not edit by hand.</p>",
    ]
    for bench in benchmarks(recs):
        n_keys = len(metric_keys(bench, recs))
        parts += [f"<h2>{html.escape(bench)}</h2>", _bar_chart(bench, recs)]
        parts += ["<h3>Leaderboard (each model's largest run)</h3>", '<div class="card">',
                  _html_table(leaderboard(bench, recs), 1, 1 + n_keys), "</div>"]  # fmt: skip
        parts += ["<h3>All runs</h3>", '<div class="card">',
                  _html_table(all_runs(bench, recs), 3, 3 + n_keys + len(GPU_COLS)), "</div>"]  # fmt: skip
    parts.append("<h2>Run details</h2>")
    for r in sorted(recs, key=lambda r: r.run_id, reverse=True):
        rows = [[k, str(v)] for k, v in sorted(r.invariants.items())]
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
    md.write_text(to_markdown(recs) + "\n")
    page.write_text(to_html(recs))
    return md, page
