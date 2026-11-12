# H2: is Laya launch-bound at batch 1? (batch sweep + CPU share, Modal, 2026-11-09)

**H2** (model-selection doc): at short states and batch 1, Laya sits far below both roofline ceilings, most
of its latency is CPU and kernel-launch time, and achieved FLOP/s rises with batch size. Rejected if GPU
kernels fill most of the batch-1 latency, or if achieved FLOP/s does not rise with batch size.

**Verdict: H2 holds for Laya on all five GPUs (L4, A100-80GB, H100, H200, B200).** Neither rejection
condition is met. Two qualifications: the launch-bound regime is not special to small models (eager
Qwen3-8B is launch-bound at 128 tokens on the H100, H200 and B200 too), and the size of the eager CPU share
depends on the container's host CPU as much as on the GPU. CUDA graphs remove both the overhead and most of
the host dependence.

Code: `benchmarks/batch_sweep.py` (`batch`), `s1/nsys.py` (`s1 nsys`), `laya_mode` in `backends/laya.py`.
Configs: `configs/batch_sweep_{smoke,laya,ref}.toml`. Records: `results/20261109-*.json` with benchmark
`batch` (`meta.cpu_model` is each run's host CPU).

## How it is measured

- B independent requests (own state, one question each) run as one forward; B = 1..64; states cut to 128
  and 1024 reference tokens (English Laya reads at most 512 tokens per row). 64 seeded JEVal rows.
- **wall**: host time per batch, tokenization through scores on the host. **kernel time**: summed GPU
  kernel durations from a CUPTI trace (torch.profiler) of the same batches, run after the timed pass.
  **CPU share** = 1 - kernel time / wall. **FLOP/s** = cost-model FLOPs (s1.flops) / wall.
- For GPU-bound points the traced kernel time can exceed the untraced wall by up to ~20% (L4 and A100
  references), giving a slightly negative CPU share: read those as about 0.
- Nsight Systems (`s1 nsys`, 64 batch-1 decisions inside cudaProfilerStart/Stop) is the cross-check. It
  agrees with the CUPTI trace on kernel time for eager and compile (L4 15.0 vs 15.4 ms; H100 4.3 vs 4.4 ms)
  but its own tracing inflates wall time (L4 eager 58 vs 40 ms), so its CPU shares run high. With
  `--cuda-graph-trace=node` (needed to see a graph's kernels at all) it inflates graph-mode runs far more
  (L4 graphs 71 vs 17 ms), so **graph-mode nsys numbers are not used**.

## Laya (laya.english, ModernBERT-large) at batch 1, 128 tokens

Host CPUs: Skylake-SP = Intel family 6 model 85; Sapphire Rapids = model 143; Emerald Rapids = model 207
(B200); Zen 2 / Zen 3 = AMD family 23 model 49 / family 25 model 1.

| GPU | host | eager ms (CPU share) | graphs ms (CPU share) | compile ms | compile+graphs ms | eager kernel ms |
|---|---|---|---|---|---|---|
| L4 | Skylake-SP | 39.6 (0.61) | 17.1 (0.16) | 54.3 | 20.5 | 15.4 |
| L4 | Zen 3 | 24.6 (0.37) | | | | 15.6 |
| A100-80GB | Sapphire Rapids | 16.1 (0.50) | 9.5 (0.22) | 16.8 | 8.7 | 8.1 |
| A100-80GB | Zen 2 | 30.7 (0.68) | | | | 9.7 |
| H100 | Sapphire Rapids | 13.9 (0.68) | 5.7 (0.28) | 14.6 | | 4.4 |
| H100 | Zen 3 | 35.2 (0.87) | 5.9 (0.30) | 35.1 | 8.3 | 4.4 |
| H200 | Sapphire Rapids | 15.7 / 16.8 (0.74 / 0.76) | 5.6 / 6.0 (0.31 / 0.36) | 20.8 / 18.3 | 6.3 | 4.0 |
| B200 | Emerald Rapids | 17.3 (0.75) | 5.4 (0.28) | 15.0 | 5.6 | 4.4 |

(Two numbers in a cell are two runs on the same host type.)

- **Most batch-1 latency is not GPU kernels.** In eager mode it is 50-87% on every GPU (37% on the one L4 Zen 3
  host). Each decision launches ~1,150 kernels for one 28-layer encoder plus the 2-layer head, and the faster
  the GPU, the more of the decision is launch time: kernel time falls from 15 ms (L4) to 4 ms (H100-B200)
  while eager wall barely moves (14-35 ms).
- **Removing launch overhead closes most of the gap.** CUDA graphs cut batch-1 latency 1.7x (A100) to 6x
  (H100 on Zen 3), to within 1.2-1.6x of kernel time. What is left is host tokenization and the copy back.
  torch.compile alone fuses kernels (~1,150 -> ~440; kernel time down 20-55%) but leaves wall time where it was.
  Compile + graphs is no better than graphs at batch 1 (its dynamic-shape guards cost host time).
- **Graphs also remove the host-CPU confound.** H100 eager: 13.9 ms on Sapphire Rapids, 35.2 ms on Zen 3;
  H100 graphs: 5.7 vs 5.9 ms.
- **Far below the memory roof even when the GPU is busy.** One batch-1 forward moves ~1.43 GB (fp32 weights
  under bf16 autocast). At spec bandwidth that is 4.8 ms on the L4 and 0.43 ms on the H100, against 40 and
  14 ms of wall: 12% and 3%. The midterm's "weights are about a third of the time" overstates it. Even the
  kernels alone (15 ms L4, 4.4 ms H100) are 3-10x above the memory floor: many small kernels, each far from
  peak bandwidth.

## Achieved FLOP/s vs batch size (TFLOP/s on wall time, B = 1, 2, 4, 8, 16, 32, 64)

| GPU (host) | mode | 128 tokens | 1024 tokens |
|---|---|---|---|
| L4 (Skylake-SP) | eager | 5 9 18 21 19 17 17 | 7 13 18 18 16 15 15 |
| A100-80GB (Sapphire Rapids) | eager | 11 21 42 57 64 69 72 | 17 31 44 50 55 57 59 |
| H100 (Sapphire Rapids) | eager | 13 23 46 92 114 124 129 | 19 37 60 84 89 94 94 |
| H200 (Sapphire Rapids) | eager | 11 20 42 82 118 126 136 | 17 31 54 85 92 99 96 |
| B200 (Emerald Rapids) | eager | 11 19 44 78 148 155 156 | 17 30 54 83 106 111 111 |
| H100 (Sapphire Rapids) | graphs | 33 54 84 107 119 125 129 | 49 64 78 89 95 98 97 |
| B200 (Emerald Rapids) | graphs | 34 61 101 133 151 157 164 | 52 73 88 101 114 112 118 |
| B200 (Emerald Rapids) | compile | 12 36 74 118 225 253 266 | 24 42 79 116 147 148 134 |

FLOP/s rises 6.5-14x from B = 1 to 64 on the A100, H100, H200 and B200, and 3.6-4.6x on the L4, which
saturates by B = 8 (~20 TFLOP/s, power-capped at 72 W). Graphs raise the B = 1 point about 3x and the curves
meet by B = 16: launch overhead is a batch-1 cost that batching amortizes. At large B the remaining host time is
tokenizing 64 states per batch, not launches. Compiled kernels reach the highest large-batch FLOP/s (B200 266
at 128 tokens).

## References: Kev-4B and Qwen3-8B, batch 1

| GPU (host) | kev-4b.hybrid 128 / 1024 tokens: ms (CPU share) | qwen3-8b.zeroshot 128 / 1024 tokens: ms (CPU share) |
|---|---|---|
| L4 (Skylake-SP) | 99.9 (~0) / 252 (~0) | 188 (~0) / 456 (~0) |
| A100-80GB (Sapphire Rapids) | 28.8 (~0) / 77.4 (0.14) | 47.7 (~0) / 115 (~0) |
| H100 (Zen 3) | 16.0 (0.10) / 100 (0.73 *) | 49.2 (0.61) / 53.1 (0.14) |
| H200 (Sapphire Rapids) | 14.3 (0.10) / 79.4 (0.68 *) | 31.1 (0.41) / 46.8 (0.07) |
| B200 (Emerald Rapids) | 12.1 (0.03) / 77.3 (0.74 *) | 30.6 (0.52) / 33.3 (0.14) |

- Eager Qwen3-8B is GPU-bound on the L4 and A100 but launch-bound at 128 tokens on the H100, H200 and B200
  (0.41-0.61): once the GPU is fast enough, even an 8B eager forward is shorter than its launches. So the
  batch-1 launch floor is a property of eager PyTorch on a fast GPU. Laya is in it on every GPU because its
  forward is small. At 1024 tokens Qwen3-8B is GPU-bound everywhere.
- Kev with its CUDA graphs is GPU-bound at 128 tokens on every GPU.
- (*) Kev at 1024 tokens on the H100, H200 and B200 is unexplained: CUDA-event busy ~0.94 but CUPTI kernel
  time only a quarter of wall, and FLOP/s flat in B (~93-125 TFLOP/s). On the L4 and A100 the same point is
  GPU-bound. Its long-state passes may fall outside its graph buckets, or CUPTI may miss Triton/graph kernels.
  Check with nsys before using this point.
- Qwen3-8B on the L4 ran out of memory at 1024 tokens for B = 32 and 64 (recorded as `_oom`).

## Confound: host CPU

The eager batch-1 floor measures the container's host as much as its GPU. Kernel time is the same within a GPU
(L4 15.4 vs 15.6 ms; H100 4.4 vs 4.4 ms) while eager wall moves 1.6x on the L4 (Skylake-SP 39.6 vs Zen 3
24.6 ms) and 2.5x on the H100 (Sapphire Rapids 13.9 vs Zen 3 35.2 ms). Compare GPUs only between records with
the same `meta.cpu_model`, or with CUDA graphs.

## Runs

Twelve runs of the first launch died together when the submitting laptop lost its network (fixed: Modal runs now
detach). The finished records were recovered from the s1-runs volume and the rest re-run. Every Laya mode now
has at least one record on every GPU, and the references have one run per GPU. Single runs per point: read differences under ~10% as noise.
