# H2: is Laya launch-bound at batch 1? (batch sweep + CPU share, Modal, 2026-11-09)

**H2** (model-selection doc): at short states and batch 1, Laya sits far below both roofline ceilings, most
of its latency is CPU and kernel-launch time, and achieved FLOP/s rises with batch size. Rejected if GPU
kernels fill most of the batch-1 latency, or if achieved FLOP/s does not rise with batch size.

**Verdict: H2 holds for Laya on every GPU measured (L4, A100-80GB, H100, H200).** Neither rejection
condition is met. Two qualifications: the launch-bound regime is not special to small models (eager
Qwen3-8B on an H100 is launch-bound at 128 tokens too), and the size of the CPU share depends on the
container's host CPU as much as on the GPU.

Code: `benchmarks/batch_sweep.py` (`batch`), `s1/nsys.py` (`s1 nsys`), `laya_mode` in `backends/laya.py`.
Configs: `configs/batch_sweep_{smoke,laya,ref}.toml`. Records: `results/20261109-18*.json`,
`results/20261109-2*.json` (meta.cpu_model is the host CPU of each).

## How it is measured

- B independent requests (own state, one question each) run as one forward; B = 1..64; states cut to 128
  and 1024 reference tokens (English Laya reads at most 512 tokens per row). 64 seeded JEVal rows.
- **wall**: host time per batch, tokenization through scores on the host. **kernel time**: summed GPU
  kernel durations from a CUPTI trace (torch.profiler) of the same batches, run after the timed pass.
  **CPU share** = 1 - kernel time / wall. **FLOP/s** = cost-model FLOPs (s1.flops) / wall.
- Nsight Systems (`s1 nsys`, 64 batch-1 decisions inside cudaProfilerStart/Stop) is the cross-check. It
  agrees with the CUPTI trace on kernel time for eager and compile (L4 15.0 vs 15.4 ms; H100 4.3 vs 4.4 ms)
  but its own tracing inflates wall time (L4 eager 58 vs 40 ms), so its CPU shares run high. With
  `--cuda-graph-trace=node` (needed to see a graph's kernels at all) it inflates graph-mode runs far more
  (L4 graphs 71 vs 17 ms), so **graph-mode nsys numbers are not used**.

## Laya (laya.english, ModernBERT-large) at batch 1, 128 tokens

| GPU | host CPU | mode | wall ms | kernel ms | CPU share | kernels / decision |
|---|---|---|---|---|---|---|
| L4 | Intel fam 6 model 85 (Skylake-SP) | eager | 39.6 | 15.4 | 0.61 | ~1,150 |
| L4 | AMD fam 25 model 1 (Zen 3) | eager | 24.6 | 15.6 | 0.37 | ~1,150 |
| A100-80GB | AMD fam 23 model 49 (Zen 2) | eager | 30.7 | 9.7 | 0.68 | ~1,150 |
| H100 | Intel fam 6 model 143 (Sapphire Rapids) | eager | 13.9 | 4.4 | 0.68 | ~1,150 |
| H200 | Intel fam 6 model 143 | eager | 15.7 | 4.0 | 0.74 | ~1,150 |
| L4 | Skylake-SP | graphs | 17.1 | 14.4 | 0.16 | (one graph launch) |
| H100 | Sapphire Rapids | graphs | 5.7 | 4.1 | 0.28 | |
| H200 | Sapphire Rapids | graphs | 5.6 | 3.9 | 0.31 | |
| L4 | Skylake-SP | compile | 54.3 | 11.9 | 0.78 | ~440 |
| H100 | Sapphire Rapids | compile | 14.6 | 2.4 | 0.84 | ~440 |
| L4 | Skylake-SP | compile_graphs | 20.5 | 12.2 | 0.40 | |

- **Most batch-1 latency is not GPU kernels**: 61-74% in eager mode on every GPU (37% on the one Zen 3 host).
  Each decision launches ~1,150 kernels for one 28-layer encoder plus the 2-layer head.
- **Removing launch overhead closes most of the gap.** CUDA graphs cut batch-1 latency 2.3x (L4), 2.4x (H100) and 2.8x
  (H200), and bring wall within 1.2-1.4x of kernel time. What is left is host tokenization and the copy
  back. torch.compile alone fuses kernels (1,150 -> 440, kernel time -20 to -45%) but leaves wall time where it
  was (L4 worse, from dynamic-shape guards); compile + graphs is no better than graphs at batch 1.
- **Far below the memory roof even when the GPU is busy.** One batch-1 forward moves ~1.43 GB (fp32 weights
  under bf16 autocast). At spec bandwidth that is 4.8 ms on the L4 and 0.43 ms on the H100, against 40 and
  14 ms of wall: 12% and 3%. The midterm's "weights are about a third of the time" overstates it. Even the
  kernels alone (15 ms L4, 4.4 ms H100) are 3-10x above the memory floor: many small kernels, each far from
  peak bandwidth.

## Achieved FLOP/s vs batch size (TFLOP/s on wall time, B = 1, 2, 4, 8, 16, 32, 64)

| GPU | mode | 128 tokens | 1024 tokens |
|---|---|---|---|
| L4 | eager | 5 9 18 21 19 17 17 | 7 13 18 18 16 15 15 |
| A100-80GB | eager | 6 12 24 48 59 64 66 | 10 17 30 43 47 51 52 |
| H100 | eager | 13 23 46 92 114 124 129 | 19 37 60 84 89 94 94 |
| H200 | eager | 12 20 45 82 115 126 134 | 17 33 54 84 92 93 94 |
| H100 | graphs | 33 54 84 107 119 125 129 | 49 64 78 89 95 98 97 |
| H100 | compile | 13 34 71 122 171 180 190 | 19 44 67 103 104 111 122 |

FLOP/s rises 10-11x from B = 1 to 64 on the A100/H100/H200 and 3.6-4.6x on the L4, which saturates by B = 8
(~20 TFLOP/s, power-capped at 72 W). Graphs raise the B = 1 point 2.5x and the curves meet by B = 16: launch
overhead is a batch-1 cost, which batching amortizes. At large B the remaining host time is tokenization of
64 states per batch, not launches.

## References (H100 only; other GPUs not run, see budget)

| model | 128 tokens, B = 1 | 1024 tokens, B = 1 | FLOP/s at B = 1 -> 64 (128 tokens) |
|---|---|---|---|
| kev-4b.hybrid (fused + CUDA graphs) | 16.0 ms, CPU share 0.10 | 100 ms, CPU share 0.73 (*) | 145 -> 302 |
| qwen3-8b.zeroshot (eager HF) | 49.2 ms, CPU share 0.61 | 53.1 ms, CPU share 0.14 | 111 -> 410 |

- Eager Qwen3-8B is launch-bound at 128 tokens too (0.61, ~1,970 kernels per forward) and GPU-bound at 1024.
  So the batch-1 launch floor is a property of eager PyTorch on a fast GPU, not of Laya's size; Laya stays in it
  at every length because its forward is small.
- Kev with its CUDA graphs is GPU-bound at 128 tokens (0.10).
- (*) Kev at 1024 tokens is unexplained: CUDA-event busy 0.94 but CUPTI kernel time only 27.5 of 100 ms, flat
  ~93 TFLOP/s at every B. Its long-state passes may fall outside its graph buckets, or CUPTI may miss
  Triton/graph kernels there. Check with nsys before using this point.

## Confound: host CPU

Same GPU, same code, eager Laya at B = 1: 39.6 ms on a Skylake-SP host, 24.6 ms on a Zen 3 host (CPU share
0.61 vs 0.37); kernel time is the same (15.4 vs 15.6 ms). The batch-1 floor of an eager model measures the
container's host as much as its GPU, so compare GPUs only between records with the same `meta.cpu_model`.

## Not measured (budget)

Twelve runs died together when the submitting laptop lost its network (fixed: Modal runs now detach); the
finished records were recovered from the s1-runs volume, but the guard charged each killed run in full and the
budget ran out. Missing: B200 entirely; Laya graphs/compile on the A100 and compile_graphs on H100/H200; the
two references on L4, A100, H200, B200; nsys for graph modes (not usable anyway, above). About $8.5 at ledger
estimates.
