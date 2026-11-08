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

Kev-4B ([jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b), a LoRA adapter on a 4B Qwen base) on the
same JEVal sample of 200 questions, bf16 on **one L4**, through upstream's own server (`kev.serve`), measured
warm (`jeval_warmup`: one untimed pass first, then the prefix cache is emptied):

| Run | Base | Accuracy | ECE | Brier | Invalid | Latency mean / p99 (concurrency 1) | q/s (concurrency 8) | GPU util |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [`20261030-144131-594ef3c9`](results/20261030-144131-594ef3c9.json) / [`…144511…`](results/20261030-144511-1354270f.json) | Qwen3.5-4B (hybrid) | **70.9%** ±6.4 | 0.137 | 0.391 | 6 | 304 ms / 2.5 s | 3.55 | 98% |
| [`20261030-144732-3359fb5e`](results/20261030-144732-3359fb5e.json) / [`…145138…`](results/20261030-145138-ad478289.json) | Qwen3-4B (attention) | 58.7% ±6.9 | 0.132 | 0.514 | 6 | 618 ms / 7.1 s | 1.67 | 99% |

The hybrid runs with upstream's fused Qwen3.5 kernels and CUDA graphs. Its first run, cold and without them, took
1.2 s per question at 41% GPU utilization (`20261030-140509-a01f91d9`). The qwen3 latency is provisional: upstream
serves attention-only bases in a packed form whose explicit attention mask rules out flash attention, and a plain
causal pass measured 379 ms (`20261030-141249-3126d051`, cold). The invalid answers are prompts over 16,384 tokens.

**Decision-model recipes on one L4 (roofline, 2026-11-04).** The same 200 short JEVal questions (≤1,024 tokens,
≤16 options), one per request, every forward timed with CUDA events and costed with `s1.flops`.
The measured L4 ceilings are 59 TFLOP/s bf16 and 231 GB/s, which puts the ridge at about 257 FLOP/B.
The analysis is in [`results/REPORT.md`](results/REPORT.md#findings) (prose: `results/roofline_findings.md`).

| Model | Recipe | Accuracy | GPU ms/q | GB read/q | FLOP/B | % of roof |
|---|---|---:|---:|---:|---:|---:|
| elephant-1 | Qwen3.5-9B full FT, fp32 slot head, state + forked questions (2 passes) | **75.5%** | 225 | 36 | 120 | 70% |
| jevk5-9b | Qwen3.5-9B full FT, letter softmax over the vocab head | 70.5% | 156 | 16 | 291 | 51% |
| qwen35-9b.zeroshot | its zero-shot base (control) | 72.5% | 154 | 16 | 291 | 51% |
| kev-4b.qwen3 | Qwen3-4B LoRA + pointer head, one packed pass | 65.5% | **66** | 7.4 | 236 | 48% |
| kev-4b.hybrid | Qwen3.5-4B LoRA + pointer head, fused kernels + CUDA graphs (2 passes) | 65.0% | 89 | 14.5 | 130 | 70% |
| laya.english | ModernBERT-large encoder + marker head, fp32 | 44.0% | 20 | 1.4 | 105 | 31% |

At batch 1 every recipe is bound by memory, so GPU time follows the bytes read. A vocab-sized readout head
and a two-pass state/question split are the expensive choices; kev's pointer head avoids the first and its
graphs reach the roof. Accuracy is ±6.5 points at this sample size.

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
| Model | `kev-4b.hybrid`, `kev-4b.qwen3` | Backend `kev`: our reimplementation of Kev's inference (LoRA merged into the Qwen3.5-4B-Base / Qwen3-4B-Base backbone, pointer head over `</opt>` and `<decide>` hidden states, head.pt's temperature). One L4. |
| Model | `semif.direct`, `semif.reranker` | Backend `semif`: controls with no decision training. Qwen3.5-4B's option-letter logits, or Qwen3-Reranker-4B's yes/no log-odds per option. Upstream's 16-option cap makes ~20% of JEVal invalid. One L4. |
| Model | `laya.english`, `laya.multilingual` | Backend `laya`: the model repo's own `rl_agent_api.py`, run only if its sha256 matches the reviewed file. Small encoders that cut the state to fit 512 / 1024 tokens. Any GPU. |
| Model | `bongard-mini` | Backend `bongard`: upstream's `Predictor` from a pinned checkout of AgentBull/bongard (`scripts/fetch_code.sh`). RTX Pro 4500 only (`gpu-g7-reserved`); the backend refuses other GPUs. |
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
                  assembly), backends/ (lazy table; vllm_shim.py serves OpenJev, kev/semif/laya/
                  bongard.py run in-process with transformers, gpu.py is their shared GPU guard)
  benchmarks/     jeval.py, shape_sweep.py
configs/          experiment configs
results/          run records (committed)
scripts/          download_model.sh (sbatch; optional revision + dir name), fetch_code.sh (pinned git
                  checkout of a model's runtime code), alloc.sh (interactive GPU shell in tmux), hooks/pre-commit
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
