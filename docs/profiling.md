# Profiling on the HUIT cluster: what works

**Every number in this document comes from one GPU type: an NVIDIA L4 (Ada, compute capability 8.9).**
It was probed with `scripts/profiling_probe.sh` (Slurm job 4237, 2026-10-27) on one `gpu`-partition node:
1x L4, 22 GiB, 58 SMs, driver 580.126.20, CUDA toolkit 13.0 (`nvcc` V13.0.88; torch 2.13.0+cu130). No model
was loaded. The roofline peaks (121 dense bf16 TFLOP/s, 300 GB/s) are the L4 spec-sheet values.

| Partition | Slurm GRES says | Actual GPU (AGENTS.md) | Probed here? |
|---|---|---|---|
| `gpu` | `gpu:l4:1` | 1x L4 24 GB (sm_89) | **yes, all results below** |
| `gpu-parallel` | `gpu:l4:4` | 4x **A10G** 24 GB (sm_86), PCIe, no NVLink; the label is wrong | **no** |
| `gpu-g7-reserved` | `gpu:rtxpro4500:1` | 1x RTX Pro 4500 32 GB | no |

The TP=4 sweeps run on `gpu-parallel`, so the A10G is the GPU that matters most for the project and is
still unmeasured. The counter permission (`RmProfilingAdminOnly`), the available tools and the ncu metric
names are driver or architecture specific and may differ there. See "Running the probe on gpu-parallel".

## Summary

| Tool | Works on the L4 (`gpu`) node? | Notes |
|---|---|---|
| `nvidia-smi --query-gpu` fields | yes | `utilization.memory`, `clocks.sm`, `clocks.mem`, `clocks_event_reasons.active` (and the old `clocks_throttle_reasons.active` alias) all return values |
| `nvidia-smi dmon -s t` | yes | rxpci/txpci MB/s, one line per second, streams line by line through a pipe |
| DCGM (`dcgmi dmon`) | **no, as installed** | `dcgmi` 4.5.1 exists, but no host engine runs: `Unable to connect to host engine` (exit 235) |
| Nsight Compute `ncu` | **yes** | 2025.3.1; `RmProfilingAdminOnly: 0`; the matmul roofline below ran as a normal user |
| Nsight Systems `nsys` | **yes** | 2025.3.2; CUDA trace and kernel summary work; GPU-metrics sampling works |

Tools are installed but **not on `PATH`**: `ncu`, `nsys`, `nvcc` are in `/usr/local/cuda/bin` (-> `cuda-13.0`),
`dcgmi` and `nv-hostengine` in `/usr/bin`. Add `export PATH=/usr/local/cuda/bin:$PATH` in the job.

## Permission (the usual ncu blocker)

```
$ grep -i RmProfilingAdminOnly /proc/driver/nvidia/params
RmProfilingAdminOnly: 0
$ sudo -n true      # no passwordless sudo (not needed, and nothing was changed)
```
`0` means non-admins may read performance counters, and `ncu` and `nsys --gpu-metrics-devices` ran as the
normal user. Had it been `1` the error is `ERR_NVGPUCTRPERM` and only a node admin can fix it (no workaround
on the cluster; ask HUIT research computing, use a machine with root, or fall back to nsys timelines plus
analytic FLOP/byte accounting). **Check this line on every new node type.**

## nvidia-smi (what the `gpu_metrics = "detailed"` sampler uses)

All probed fields work; `nvidia-smi --help-query-gpu` lists both `clocks_event_reasons.*` (new) and
`clocks_throttle_reasons.*` (old), and the sampler prefers the new name. Idle vs busy on the same GPU
(`ncu_smoke.py sample`, 5 s idle then 8 s of 4096x4096 bf16 matmuls):

| key | idle | busy |
|---|---|---|
| `gpu0_util_pct` | 0 | 92.0 |
| `gpu0_mem_util_pct` | 0 | 7.2 |
| `gpu0_clock_sm_mhz` | 210 | 1048 |
| `gpu0_clock_mem_mhz` | 405 | 5973 |
| `gpu0_power_w` | 16.2 | 65.3 |
| `gpu0_throttle_frac` | 0 | 0.95 (mask `0x4` = SW power cap; the L4 is a 72 W card) |
| `gpu0_pcie_tx_mb_s` | 0 | 1.6 |
| `gpu_util_pct_spread` | 0 | 0 (one GPU; it needs a multi-GPU node to be non-zero) |

The throttle result is the useful surprise: a busy L4 runs power-capped, which pooled utilization hides.

## DCGM

`dcgmi discovery -l` and `dcgmi dmon -e 1002,1003,1004,1005,1009,1010 -c 5 -d 500` both fail with
`Error: Unable to connect to host engine` because nothing runs `nv-hostengine` on the compute node. We did
**not** try starting one (that is a daemon; ask first). So the Stage-3 DCGM sampler was not built. The
same quantities are available from the nsys GPU-metrics sampler below.

## Nsight Systems

```bash
export PATH=/usr/local/cuda/bin:$PATH
nsys profile --trace=cuda,nvtx --stats=true -o out/nsys_basic python script.py        # timeline + summaries
nsys profile --trace=cuda --gpu-metrics-devices=cuda-visible -o out/nsys_gm python script.py
```
- CUDA API and kernel summaries work (`cuda_api_sum`, `cuda_gpu_kern_sum`). NVTX is empty unless the
  script emits ranges.
- `--gpu-metrics-device=all` is deprecated; use `--gpu-metrics-devices=cuda-visible` (or `all`).
- GPU-metrics sampling works at ~10 kHz. `nsys stats --report gpu_metrics` does **not** exist in this
  version; read the `GPU_METRICS` and `TARGET_INFO_GPU_METRICS` tables of the exported `.sqlite`
  (`metricName`, `value`). Metrics seen on the L4: SMs Active, SM Issue, Tensor Active, Compute Warps in
  Flight, DRAM Read/Write Bandwidth, PCIe RX/TX Throughput, GR Active, clocks. In the matmul run Tensor
  Active peaked at 96%.

## Nsight Compute: roofline method validated on bf16 matmuls

`scripts/ncu_smoke.py run` does, per size, fresh operands, a warm-up matmul, a `spin` marker kernel and the
measured matmul. The parser finds the marker in ncu's CSV and takes the GEMM kernels after it.

```bash
export PATH=/usr/local/cuda/bin:$PATH
ncu --target-processes all --csv \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
  --log-file ncu.csv python scripts/ncu_smoke.py run --sizes 512,1024,2048,4096
python scripts/ncu_smoke.py parse ncu.csv --gpu "NVIDIA L4"
```
Metric names are architecture specific: the tensor-pipe metric above exists on this Ada (sm_89) chip;
`ncu --query-metrics | grep -i tensor` lists the alternatives (newer `_v2` instruction counters and
`sm__ops_path_tensor_src_bf16_dst_fp32`). The CSV `Metric Unit` for durations is `ns`.

Result (L4; spec peaks 121 dense bf16 TFLOP/s, 300 GB/s, ridge 403 FLOP/B):

| n | kernel | us | DRAM MB | TFLOP/s | % peak | AI FLOP/B | ideal AI (n/3) | sm% | mem% | tensor% |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | ampere_bf16_s16816gemm 128x64 | 19.7 | 1.28 | 13.6 | 11 | 210 | 171 | 29 | 28 | 61 |
| 1024 | ampere_bf16_s1688gemm 128x128 | 80.9 | 5.03 | 26.5 | 22 | 427 | 341 | 55 | 45 | 67 |
| 2048 | ampere_bf16_s1688gemm 128x128 | 424.4 | 20.21 | 40.5 | 33 | 850 | 683 | 86 | 56 | 88 |
| 4096 | ampere_bf16_s1688gemm 128x128 | 3123.0 | 164.52 | 44.0 | 36 | 835 | 1365 | 93 | 55 | 95 |

Reading it: SM and tensor-pipe utilization climb with size (29% -> 93% SM, 61% -> 95% tensor), arithmetic
intensity is above the ridge from n=1024 up (compute-bound), and n=512 sits below it with lower
utilization. Measured AI is below the ideal at n=4096 because tiled kernels re-read operands.

Caveats on the numbers:
- **ncu locks clocks to base by default**, so its TFLOP/s is below the unprofiled run. CUDA-event timing
  without a profiler gave 27.6 / 57.3 / 72.6 / 57.3 TFLOP/s for the same sizes. Use `--clock-control none`
  to measure at boost clocks. Use ncu for ratios and bytes, not absolute speed.
- The L4 is power-capped (72 W), which also lowers sustained clocks (see the throttle row above).
- Peaks come from spec sheets in `PEAKS` in `ncu_smoke.py`; check them for other GPUs.
- ncu caches are flushed between replays by default, so DRAM bytes are cold-cache numbers.

## What this means for model profiling

- **ncu replays each kernel** several times (one pass per counter group), so it is practical only for a
  handful of kernels chosen after an nsys run finds the top ones. It is not for a full sweep.
- **vLLM tensor-parallel workers are separate processes.** ncu needs `--target-processes all` and is clumsy
  with TP=4 workers; for nsys also set `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
  `--trace-fork-before-exec=true` and `--cuda-graph-trace=node`.
- **The in-process PyTorch models** (SemIf, Kev, Laya, Bongard, decider) are the better ncu targets: one
  process, one GPU, and `torch.cuda.nvtx` ranges or marker kernels delimit the region.
- A sweep that needs SM, tensor and DRAM activity over time can use the nsys GPU-metrics sampler instead of
  DCGM until a host engine is available.

## Rerunning the probe

```bash
mkdir -p ~/probes/logs ~/probes/pkg/s1 && touch ~/probes/pkg/s1/__init__.py
cp scripts/profiling_probe.sh scripts/ncu_smoke.py ~/probes/
cp src/s1/gpu_stats.py src/s1/record.py ~/probes/pkg/s1/
cd ~/probes && sbatch -p gpu --gres=gpu:1 -c 4 --mem=16G -t 00:10:00 -J profiling-probe \
  -o ~/probes/logs/%j.out ~/probes/profiling_probe.sh
```
Run `sbatch` from the directory holding `ncu_smoke.py`: Slurm runs a copy of the script from its spool
directory. Raw ncu/nsys files land in `~/probes/out/<jobid>/`.

## Modal

The GPU sweep moves to Modal (the course workspace; credits are shared by the whole class, so probes stay at
a few minutes each). `scripts/modal_probe.py` runs the same `profiling_probe.sh` and `ncu_smoke.py` inside a
Modal container. **Status: run on 2026-11-08 on a Modal L4 and a pinned H100 (results below). ncu and nsys
both work as a normal user on both GPUs; DCGM is not installed.**

### Running the probe

`modal` is not a dependency of s1; run the script through `uvx`. Auth is the user's own Modal profile, never
a file in the repo.

```bash
uvx --python 3.12 --from modal==1.6.3 modal run scripts/modal_probe.py --gpu L4
```
```bash
uvx --python 3.12 --from modal==1.6.3 modal run scripts/modal_probe.py --gpu 'H100!'
```
- `--gpu` takes `L4`, `A10`, `L40S`, `A100-80GB` or `H100!`. A plain `H100` is refused: Modal may upgrade it
  to an H200. `--sizes` (default `512,1024,2048,4096`) are the matmul sizes.
- The image (built once, CPU only): `debian_slim` Python 3.12, `torch==2.14.1` (the course starter's), and
  Nsight Compute 2026.3.1.2 and Nsight Systems 2026.3.2.476 from NVIDIA's CUDA redist archive (the pair in
  `redistrib_13.4.3.json`), sha256-checked, unpacked to `/opt/nvidia/{ncu,nsys}` and called through wrappers in
  `/opt/nvidia/bin` (`TOOL_PATH`). DCGM is not installed, so section b should report `dcgmi: MISSING`.
- One function call, `gpu=<--gpu>`, 4 CPUs, 16 GiB, 20 min timeout. In the container the script saves
  `nvidia-smi -q`, prints a summary of it, then runs `profiling_probe.sh` with `MODAL_TASK_ID` set. That skips
  the Slurm guard and runs ncu with `--clock-control none`.
- Output: Modal volume `s1-probe-logs`, directory `<gpu>-<UTC stamp>-<id>/`, downloaded to
  `logs/modal-probe-<gpu>-<stamp>-<id>/`: `probe.log` (the whole probe output), `summary.json` (GPU name,
  driver, compute capability, tool versions, whether ncu ran or hit `ERR_NVGPUCTRPERM`, the
  `RmProfilingAdminOnly` value), `nvidia-smi-q.txt`, `ncu.csv`, `ncu_query_metrics.txt`, the nsys reports.

### What to check per GPU

1. **Which GPU you got.** `nvidia-smi --query-gpu=name,compute_cap` and the `nvidia-smi -q` summary: product
   name, compute capability, SM count (torch line in section 0), memory, driver and CUDA version, power limits
   (current, default, max), persistence and virtualization mode. An `H100!` should be an H100 80GB HBM3 (sm90).
2. **nvidia-smi fields** (section a): whether `utilization.memory`, `clocks.sm`, `clocks.mem` and
   `clocks_event_reasons.active` return values inside the container, and whether `dmon -s t` gives PCIe rx/tx.
   The `gpu_metrics = "detailed"` sampler depends on these.
3. **DCGM** (section b): expected missing; note it if `dcgmi` is somehow present.
4. **ncu** (section c): `RmProfilingAdminOnly` (it may not be readable under Modal's sandbox), whether the
   profiled run produces `ncu.csv` or `ERR_NVGPUCTRPERM`, which tensor-pipe metric exists (Hopper may need a
   different name than Ada's `sm__pipe_tensor_op_hmma_cycles_active`), the roofline table against the spec
   peaks in `ncu_smoke.py` (`PEAKS`: L4 121 TFLOP/s and 300 GB/s; H100 SXM 989 TFLOP/s and 3.35 TB/s), and
   how close ncu's TFLOP/s (unlocked clocks here) is to the unprofiled CUDA-event numbers.
5. **nsys** (section d): CUDA trace and kernel summary, and whether `--gpu-metrics-devices=cuda-visible` works
   (it needs the same counter permission as ncu).
6. **GpuSampler idle vs busy** (section e): which detailed keys move, and the throttle mask under load (the
   cluster L4 ran power-capped, mask `0x4`).

### Results

| | L4 | H100! |
|---|---|---|
| Modal task / date | `ta-01M6XXSCNA1A8WHNHE4DBK1TER`, 2026-11-08 | `ta-01M6XXSCJAW4EXRVMGE80VA1YR`, 2026-11-08 |
| GPU, cc, SMs, memory | NVIDIA L4, 8.9, 58 SMs, 22.1 GiB | NVIDIA H100 80GB HBM3, 9.0, 132 SMs, 79.2 GiB (no H200 swap) |
| driver / CUDA | 610.57.04 / 13.3 | 610.57.04 / 13.3 |
| power limit (enforced / max) | 72 W / 72 W | 700 W / 700 W |
| virtualization | pass-through | none |
| `RmProfilingAdminOnly` | 0 | 0 |
| nvidia-smi detailed fields | all 38 sampler keys present idle and busy | all 38 sampler keys present idle and busy |
| `dmon -s t` | works, streams line by line | works, streams line by line |
| DCGM | `dcgmi` missing | `dcgmi` missing |
| ncu (`--clock-control none`) | runs, no `ERR_NVGPUCTRPERM` | runs, no `ERR_NVGPUCTRPERM` |
| ncu tensor metric | `sm__pipe_tensor_op_hmma_cycles_active` | same name works on sm90 |
| nsys trace / GPU metrics | trace + kernel summary work; GPU-metrics sampling exits 0 (read the `.sqlite`; `--report gpu_metrics` does not exist, as on the cluster) | same |
| 4096 bf16 matmul: ncu TFLOP/s, % peak, AI | 76.9, 64% of 121, 838 FLOP/B | 767, 78% of 989, 766 FLOP/B |
| 4096 bf16 matmul: unprofiled TFLOP/s | 51.7 (2048: 72.8) | 773 |
| throttle under load | 95% of busy samples, mask `0x4` (SW power cap), max 74 W | 88% of busy samples, mask `0x4`, max 719 W, SM clock 1375 MHz mean |

Both GPUs run power-capped under a sustained bf16 matmul, as the cluster L4 did: throttle fractions and clocks
belong next to any cross-GPU latency comparison. Raw files: `logs/modal-probe-{l4,h100}-20261108T155248Z-*`
(gitignored).

### Clock control in `s1 ncu`

`s1.ncu` has an `ncu_clock_control` setting (`base` or `none`, ncu's `--clock-control`). The default is
unchanged off Modal: `base`, ncu's own default, which locks clocks to base, and the flag is left off the command
line. Inside a Modal container (`s1.env.orchestrator() == "modal"`) the default is `none`, because clock
locking fails there. Every `s1 ncu` run records it as `roofline_hw_clocks_locked` in `results["roofline"]`
(1 = base, 0 = none). Kernel times (`roofline_hw_kernel_ms`) from the two settings are not comparable: base
clocks are lower than boost.

## Running the probe on gpu-parallel (4x A10G)

Same scripts, different `sbatch` flags. `sbatch` command-line options override the `#SBATCH` lines in the
script. Nodes are powered down cloud instances, so expect 5-10 minutes to boot (`NODE_FAIL` /
`InsufficientInstanceCapacity` means resubmit). The partition allows 2 days; 15 minutes is plenty.

```bash
cd ~/probes && sbatch -p gpu-parallel --gres=gpu:4 -c 8 --mem=32G -t 00:15:00 -J profiling-probe-a10g \
  -o ~/probes/logs/%j.out ~/probes/profiling_probe.sh
```
Asking for all four GPUs makes the `ncu_smoke.py sample` section informative: it loads only GPU 0, so the
other three stay idle and `gpu_util_pct_spread` and the per-GPU keys (`gpu0_*` vs `gpu1_*`) finally differ,
and `dmon` reports four PCIe rows. Use `--gres=gpu:1` if you only want the permission and ncu answers.

What to read from the log:
- the GPU name must read `NVIDIA A10G`, compute capability 8.6 (the GRES label says L4; trust `nvidia-smi`);
- `RmProfilingAdminOnly` (0 = ncu works; 1 = `ERR_NVGPUCTRPERM`, report it and stop);
- the tensor-pipe metric the probe selected (names differ on sm_86) and whether the matmul table appears;
- whether `dcgmi` and `nsys --gpu-metrics-devices` behave as on the L4.

Known gaps before it is trusted: the `PEAKS` entry for A10G in `scripts/ncu_smoke.py` (70 dense bf16
TFLOP/s, 600 GB/s) is an unverified spec figure, and Ampere has no FP8, so only bf16 is meaningful.
