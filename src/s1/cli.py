"""`s1` command line: run experiments locally or via sbatch, and inspect results."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import config, report, runner
from .models.catalog import CATALOG
from .record import RESULTS_DIR, load_all

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

ResultsDir = typer.Option(RESULTS_DIR, "--results-dir", "-o", help="Where run records are written/read.")


@app.command()
def run(cfg: Path, results_dir: Path = ResultsDir) -> None:
    """Run every combination in CONFIG in this process (use inside an allocation)."""
    specs = config.load(cfg)
    console.print(f"{cfg}: {len(specs)} run(s)")
    for spec in specs:
        runner.run(spec, config_path=cfg, results_dir=results_dir, log=console.print)


@app.command()
def submit(
    cfg: Path,
    *,
    partition: str | None = typer.Option(None, "-p", "--partition", help="Default: cluster default."),
    gpus: int = typer.Option(1, "-g", "--gpus"),
    cpus: int = typer.Option(8, "-c", "--cpus"),
    mem: str = typer.Option("32G", "--mem"),
    time: str = typer.Option("01:00:00", "-t", "--time"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the sbatch command only."),
) -> None:
    """Queue `s1 run CONFIG` as a batch job; logs go to logs/<jobid>.out."""
    config.load(cfg)  # fail fast on a bad config before queueing
    cmd = ["sbatch", "-J", f"s1-{cfg.stem}", "-c", str(cpus), f"--mem={mem}", "-t", time]
    cmd += ["-o", "logs/%j.out"]
    if partition is not None:
        cmd += ["-p", partition]
    if gpus:
        cmd += [f"--gres=gpu:{gpus}"]
    cmd += ["--wrap", shlex.join(["uv", "run", "--extra", "openjev", "s1", "run", str(cfg)])]
    console.print(shlex.join(cmd))
    if not dry_run:
        Path("logs").mkdir(exist_ok=True)
        subprocess.run(cmd, check=True)


@app.command("ls")
def ls(results_dir: Path = ResultsDir) -> None:
    """Table of runs: params and each benchmark's headline metric."""
    recs = load_all(results_dir)
    if not recs:
        console.print(f"no runs in {results_dir}/")
        return
    param_keys = sorted({k for r in recs for k in r.params})
    benches = sorted({b for r in recs for b in r.results})
    t = Table(header_style="bold")
    for col in ["run_id", "gpu", *param_keys, *benches]:
        t.add_column(col, no_wrap=True)
    for r in recs:
        headline = [_headline(r.results.get(b)) for b in benches]
        params = [str(r.params.get(k, "")) for k in param_keys]
        t.add_row(r.run_id, str(r.invariants.get("gpu", "")), *params, *headline)
    console.print(t)


def _headline(metrics: dict[str, float] | None) -> str:
    """First metric, with its _std alongside when present."""
    if metrics is None or not metrics:
        return ""
    key, val = next(iter(metrics.items()))
    std = metrics.get(f"{key}_std")
    return f"{val:.4g}" + (f" ± {std:.2g}" if std is not None else "")


@app.command("models")
def models_cmd() -> None:
    """Table of the model catalog (pins, sizes, status); reads metadata only."""
    t = Table(header_style="bold")
    for col in ["name", "group", "status", "repo@rev", "GB", "license", "notes"]:
        t.add_column(col, no_wrap=col != "notes")
    for m in CATALOG:
        gb = "" if m.weights_gb is None else f"{m.weights_gb:.1f}"
        t.add_row(
            m.name, m.group, m.status, f"{m.repo}@{(m.revision or 'unpinned')[:8]}", gb, m.license, m.notes
        )
    console.print(t)


@app.command()
def collect(
    results_dir: Path = ResultsDir,
    out: Path | None = typer.Option(None, "--out", help="Write here instead of stdout."),
) -> None:
    """Merge all run records into one {run_id: record} JSON document."""
    merged: dict[str, object] = {}
    for r in load_all(results_dir):
        merged |= r.to_json()
    text = json.dumps(merged, indent=4)
    if out is None:
        print(text)
    else:
        out.write_text(text + "\n")
        console.print(f"wrote {len(merged)} run(s) to {out}")


@app.command("report")
def report_cmd(results_dir: Path = ResultsDir) -> None:
    """Regenerate results/REPORT.md and results/report.html from every run record."""
    md, page = report.write(results_dir)
    console.print(f"wrote {md} and {page}")


def main() -> None:
    app()
