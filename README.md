# cs2470-final-project — System One decision-model experiments

A small harness for benchmarking **Jev-style decision models**. These models take a `state` plus typed
questions (`choice`, yes/no `noul`, ordered `score`) and return one probability per option from a
single forward pass. Each run writes one JSON record you can commit, compare and merge.

> Agents (and humans doing a new run): read [`AGENTS.md`](AGENTS.md). It is the step-by-step
> runbook for this cluster, including every pitfall found so far.

## Results so far

Full tables for every run: [`results/REPORT.md`](results/REPORT.md) (renders on GitHub) and
`results/report.html` (the same with charts; open it locally). Both regenerate automatically on every commit.

OpenJev ([openjev/openjev](https://huggingface.co/openjev/openjev), 27B) on
[JEVal](https://huggingface.co/datasets/carlosxiang/JEVal), bf16, one `gpu-parallel` node (4× A10G):

| Run | Questions | Accuracy | ECE | Brier | Invalid | Latency mean / p99 | GPU util | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [`20261021-233308-d6173a57`](results/20261021-233308-d6173a57.json) | 2,000 | **76.8%** ±1.9 | 0.041 | 0.282 | 49 | 12.3 s / 70.6 s | 99.9% | 380 Wh |
| [`20261021-230256-c16c0fcf`](results/20261021-230256-c16c0fcf.json) | 200 | 77.6% ±5.9 | 0.057 | 0.241 | 6 | 11.7 s / 60.1 s | 96.8% | 36 Wh |

Accuracy covers choice and yes/no questions, with invalid answers counted as wrong. All invalid
answers came from prompts longer than the 16,384-token context. For reference, the JEVal paper's
Table 1 reports hosted Jev at 78.4% and Kev-27b at 77.2% on all 11,257 questions. Latency is far
above the model card's H100 figures because A10Gs are slower and lack NVLink.

## What a run record looks like

```json
{
    "20261021-233308-d6173a57": {
        "invariants": { "gpu": "NVIDIA A10G", "tensor_parallel_size": 4, "quantization": "none", "max_model_len": 16384 },
        "params":     { "model": "openjev", "jeval_limit": 2000, "jeval_seed": 0 },
        "results":    { "jeval":            { "jeval_accuracy": 0.7685, "jeval_ece": 0.0408, "jeval_latency_ms_p99": 70627.7, ... },
                        "jeval/nvidia_smi": { "gpu_util_pct": 99.92, "gpu_mem_used_mb_max": 20567.0, "gpu_energy_wh": 379.8, ... } },
        "meta":       { "started": "2026-10-21T23:33:08", "git_commit": "f5d41d1-dirty", "slurm_job_id": "3241", "duration_s": 3489.0 }
    }
}
```

- **invariants**: the fixed setup (hardware, serving config). `gpu = "auto"` is filled in from `nvidia-smi`.
- **params**: what the experiment varies (model, sample size, seeds, ...).
- **results**: one flat metrics dict per benchmark, plus `<bench>/nvidia_smi`. That one comes from sampling
  `nvidia-smi` every 500 ms during the benchmark: utilization (mean/std/p90/p95/p99), memory, power, energy and temperature.
- **meta**: where and when it ran, the git commit, and timings.

Records go to `results/<run_id>.json`, one file per run, so concurrent jobs never collide. **Commit
them.** Logs go to `logs/`, which is gitignored.

## Quick start

```bash
uv sync                                                    # light deps; tests run anywhere
uv run pytest                                              # no GPU needed (fake model, fake nvidia-smi)

sbatch scripts/download_model.sh openjev/openjev           # once: weights -> /shared/home/$USER/models/openjev
uv run s1 submit configs/openjev_jeval_smoke.toml -p gpu-parallel -g 4 -c 32 --mem 160G -t 01:00:00   # 200 questions
uv run s1 ls                                               # table of every run in results/
uv run s1 collect --out all_runs.json                      # all records merged into one {run_id: record}
```

| Command | What it does |
|---|---|
| `s1 run CONFIG` | Run every combination in CONFIG in this shell (inside a GPU allocation) |
| `s1 submit CONFIG [-p -g -c --mem -t --dry-run]` | Same, as an `sbatch` job; log in `logs/<jobid>.out` |
| `s1 ls` | Table of runs: params and each benchmark's headline metric |
| `s1 collect [--out FILE]` | Merge all records |
| `s1 report` | Regenerate `results/REPORT.md` and `results/report.html` from all records |

### Configs

```toml
benchmarks = ["jeval"]
[invariants]            # fixed setup, recorded as-is
gpu = "auto"
tensor_parallel_size = 4
[params]                # fixed for every run in this file
model = "openjev"
[sweep]                 # optional: lists; one run per combination, merged into params
jeval_seed = [0, 1, 2]
[grid]                  # optional: lists that are NOT expanded; the run carries them (recorded as `grid`)
state_tokens = [32, 128]  # and a benchmark such as shape_sweep walks them inside one model load
```

Models and benchmarks both receive `invariants | params` as their `settings`, plus each grid axis as
`grid_<axis>` = comma-joined values (`grid_state_tokens = "32,128"`).
Shipped configs: `configs/openjev_jeval_smoke.toml` (200 questions, about 17 min including load),
`configs/openjev_jeval_2k.toml` (2,000, about 1 h) and `configs/openjev_jeval.toml` (all 11,257, about 5 h).

## What's wired up

| | Name | Notes |
|---|---|---|
| Model | `openjev` | Starts vLLM plus the model's own `helper/shim.py` (`POST /v1/systemone`) with the model card's calibration, and stops both after the run. `openjev_endpoint` reuses a running helper instead. On A10G it needs `quantization = "none"`, because the card's fp8 recipe requires compute capability 8.9 or higher. |
| Benchmark | `jeval` | JEVal (arXiv:2610.03935), pinned revision, 11,257 questions over 10 domains. It reports accuracy with invalid answers counted wrong, ECE (15 bins) and Brier score as in the paper, plus score-question accuracy/MAE and latency percentiles. Subsample with `jeval_limit` / `jeval_seed`. |
| Benchmark | `shape_sweep` | Model-agnostic cost per question over a `[grid]` of `state_tokens` x `questions_per_request` (synthetic states, JEVal-shaped choice questions, concurrency 1). Each point repeats for `shape_min_point_s` and gets its own GPU sampling; latency is divided by the question count. Headline `shape_sweep_latency_ms` (lower is better), per-point keys `shape_sweep_t{T}_q{Q}_*`. Points above a model's optional `max_state_tokens` are counted as skipped, not wrong. Token counting is a pluggable `state_tokenizer` (only the approximate `approx_words` so far). Example: `configs/openjev_shape_sweep_smoke.toml`. |

To add another model or benchmark, see [`AGENTS.md`](AGENTS.md#adding-a-jev-style-model).

## Layout

```
src/s1/
  cli.py          s1 run / submit / ls / collect
  config.py       TOML -> list of RunSpec (sweep expansion, grid)
  runner.py       build model -> each benchmark under GpuSampler -> close -> write record
  record.py       RunRecord, results/ I/O, summarize() (mean/std/p90/p95/p99)
  registry.py     model / benchmark registries (duplicate names raise), typed settings getters
  gpu_stats.py    nvidia-smi sampler
  report.py       results/REPORT.md + report.html (run by the pre-commit hook)
  models/         spec.py (ModelSpec), catalog.py (the list of specs), answers.py (System One answer
                  assembly), backends/ (lazy table + vllm_shim.py, the OpenJev serving code)
  benchmarks/     jeval.py, shape_sweep.py
configs/          experiment configs
results/          run records (committed)
scripts/          download_model.sh (sbatch), alloc.sh (interactive GPU shell in tmux), hooks/pre-commit
tests/            pytest; fakes in conftest.py
```

## Development

```bash
git config core.hooksPath scripts/hooks      # once per clone: pre-commit regenerates the results report
uv run ruff format && uv run ruff check && uv run ty check && uv run pytest
```
Ruff and ty run in strict mode (annotations required, every ty rule is an error). Keep them passing.

---

### Cluster note (HUIT OnDemand)

| Partition | Slurm GRES | Actual GPU | Notes |
|---|---|---|---|
| `gpu` | `gpu:l4:1` | 1× L4, 24 GB | |
| `gpu-parallel` | `gpu:l4:4` | **4× A10G**, 24 GB each, compute 8.6 | Mislabeled. No FP8. Use this for 27B models (bf16, TP=4). |
| `gpu-g7-reserved` | `gpu:rtxpro4500:1` | 1× RTX Pro 4500, 32 GB | |

Login nodes cap each process at 2 GB of virtual memory and allow 100 processes per user, so
downloads and installs belong on compute nodes. Cloud nodes (`idle~`) take 5–10 min to boot. A
`NODE_FAIL` with `InsufficientInstanceCapacity` means AWS is out of that instance type; retry.
For an interactive GPU shell that survives a dropped terminal: `scripts/alloc.sh gpu-parallel 4 02:00:00`.
