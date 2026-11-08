"""`s1` command line: run experiments locally or via sbatch, and inspect results."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import config, env, ncu, report, runner
from .launch import Resources, get_launcher
from .models.catalog import CATALOG
from .record import RESULTS_DIR, RunRecord, Scalar, load_all

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

ResultsDir = typer.Option(RESULTS_DIR, "--results-dir", "-o", help="Where run records are written/read.")


def _scalar(text: str) -> Scalar:
    """A `--set` value: true/false, an int, a float, else the string."""
    if text in ("true", "false"):
        return text == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


@app.command()
def run(
    cfg: Path,
    results_dir: Path = ResultsDir,
    index: int | None = typer.Option(None, "--index", help="Run only this combination (0-based)."),
    sets: list[str] = typer.Option([], "--set", help="Override a param: key=value (repeatable)."),
) -> None:
    """Run every combination in CONFIG in this process (use inside an allocation)."""
    specs = config.load(cfg)
    if index is not None:
        specs = [specs[index]]
    overrides = {k: _scalar(v) for k, _, v in (s.partition("=") for s in sets)}
    console.print(f"{cfg}: {len(specs)} run(s)")
    for spec in specs:
        spec.params |= overrides
        runner.run(spec, config_path=cfg, results_dir=results_dir, log=console.print)


@app.command("ncu")
def ncu_cmd(
    cfg: Path,
    questions: int = typer.Option(4, "--questions", help="Questions profiled per model."),
    results_dir: Path = ResultsDir,
    ncu_bin: str = typer.Option("/usr/local/cuda/bin/ncu", "--ncu", help="Nsight Compute CLI."),
    clock_control: Annotated[
        str | None,
        typer.Option("--clock-control", help="ncu --clock-control: base (default; none on Modal) or none."),
    ] = None,
) -> None:
    """Run each model of CONFIG under Nsight Compute (use inside an allocation): the roofline benchmark
    re-answers `--questions` questions in a CUDA profiler range, and the summed hardware counters (FLOPs,
    DRAM bytes) are added to that run's record as roofline_hw_*. See s1/ncu.py."""
    specs = config.load(cfg)
    clock = ncu.clock_control(clock_control)
    job = env.job_id()
    Path("logs").mkdir(exist_ok=True)
    for i in range(len(specs)):
        log = Path("logs") / f"ncu-{job}-{cfg.stem}-{i}.csv"
        before = set(results_dir.glob("*.json"))
        child = [sys.executable, "-m", "s1.cli", "run", str(cfg), "--index", str(i)]
        child += ["--results-dir", str(results_dir), "--set", f"roofline_ncu={questions}"]
        subprocess.run([ncu_bin, *ncu.args(clock), "--log-file", str(log), *child], check=True)
        (path,) = set(results_dir.glob("*.json")) - before
        rec = RunRecord.load(path)
        hw = ncu.summarize(ncu.parse(log.read_text()), rec.results["roofline"], clock)
        rec.results["roofline"] |= hw
        rec.meta["ncu_log"] = str(log)
        rec.meta[ncu.META_CLOCK_CONTROL] = clock
        rec.save(results_dir)
        console.print(f"[{rec.run_id}] hardware counters: {hw}")


def launch_resources(
    cfg: Path,
    *,
    on: str | None = None,
    partition: str | None = None,
    gpu: str | None = None,
    gpus: int | None = None,
    gpu_type: str | None = None,
    cpus: int | None = None,
    mem: str | None = None,
    time: str | None = None,
) -> Resources:
    """The config's [launch] table (Slurm defaults when absent), each flag that was given overriding its
    field: `on` the orchestrator, `partition`/`gpu_type` the extras of that orchestrator."""
    target = None if on is None else config.target(on, "--on")
    res = config.load_launch(cfg, target) or Resources(orchestrator=target or "slurm")
    extras = dict(res.extras)
    if partition is not None:
        extras["partition"] = partition
    if gpu_type is not None:
        extras["gpu_type"] = gpu_type
    return replace(
        res,
        gpu=res.gpu if gpu is None else gpu,
        gpus=res.gpus if gpus is None else gpus,
        cpus=res.cpus if cpus is None else cpus,
        mem=res.mem if mem is None else mem,
        time=res.time if time is None else time,
        extras=extras,
    )


@app.command()
def submit(
    cfg: Path,
    *,
    on: str | None = typer.Option(
        None, "--on", help="slurm or modal. Default: [launch] orchestrator, else slurm."
    ),
    partition: str | None = typer.Option(
        None, "-p", "--partition", help="Default: [launch.<on>], else cluster default."
    ),
    gpu: str | None = typer.Option(None, "--gpu", help="Portable GPU name (Modal), e.g. H100, L4."),
    gpus: int | None = typer.Option(None, "-g", "--gpus", help="Default: [launch], else 1."),
    gpu_type: str | None = typer.Option(None, "--gpu-type", help="GRES type, e.g. l4 (gpu-spec types them)."),
    cpus: int | None = typer.Option(None, "-c", "--cpus", help="Default: [launch], else 8."),
    mem: str | None = typer.Option(None, "--mem", help="Default: [launch], else 32G."),
    time: str | None = typer.Option(None, "-t", "--time", help="Default: [launch], else 01:00:00."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be queued only."),
    ncu_questions: int = typer.Option(0, "--ncu", help="Queue `s1 ncu` with this many questions instead."),
) -> None:
    """Queue `s1 run CONFIG` (or `s1 ncu CONFIG`) on Slurm or Modal, per the config's [launch] table and
    the flags (flags win). Without [launch] or --on it is a Slurm batch job; logs go to logs/<jobid>.out."""
    config.load(cfg)  # fail fast on a bad config before queueing
    if on is not None and on not in config.ORCHESTRATORS:
        raise typer.BadParameter(f"must be one of {', '.join(config.ORCHESTRATORS)}", param_hint="--on")
    res = launch_resources(
        cfg, on=on, partition=partition, gpu=gpu, gpus=gpus, gpu_type=gpu_type, cpus=cpus, mem=mem, time=time
    )
    s1 = ["ncu", str(cfg), "--questions", str(ncu_questions)] if ncu_questions else ["run", str(cfg)]
    get_launcher(res.orchestrator).submit(cfg, res, s1, dry_run=dry_run)


@app.command()
def download(
    repo: str,
    revision: str | None = typer.Argument(None, help="Pin a revision (use the catalog's)."),
    dirname: str | None = typer.Argument(None, help="Directory under MODELS_DIR (default: repo name)."),
    hf_secret: str | None = typer.Option(None, "--hf-secret", help="Modal secret with HF_TOKEN."),
) -> None:
    """Download a Hub model into the Modal `s1-models` volume (/models/<dir>) on a CPU container, with the
    SHA256SUMS check of scripts/download_model.sh. For the cluster use `sbatch scripts/download_model.sh`."""
    from .launch import modal_app  # noqa: PLC0415

    modal_app.download(repo, revision, dirname, hf_secret=hf_secret)


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


if __name__ == "__main__":
    main()
