How the design choices that turn a base model into a decision model use a GPU, and what changes when the GPU
gets bigger. Eleven models answer JEVal questions one at a time (batch 1, one forward pass per decision, no
generation). Each forward is timed with CUDA events; its FLOPs and compulsory DRAM bytes come from the analytic
cost model (`s1.flops`), so **intensity (FLOP/byte) is a property of the model and its prompt, the same on
every GPU**, and only the achieved FLOP/s depends on the GPU and the software.

Two platforms, never on one roof: the HUIT cluster L4 (the midterm runs, 2026-11-04) and Modal (2026-11-08):
L4, A100 80GB (SXM4, and one PCIe card Modal substituted), H100 SXM, H200 and B200, all one GPU each.
Measured ridge points (bf16 GEMM peak over copy bandwidth, measured on each card before loading weights):
cluster L4 ~257, Modal L4 245, A100 SXM 152, A100 PCIe 135, H100 227, H200 160, B200 207 FLOP/byte.
The measured bf16 peaks are 47–85% of the datasheet (L4 47%, B200 59%, H100 and H200 68%, A100 SXM 85%).

Two modes. **Natural:** 200 JEVal questions at their own lengths (<= 1,024 reference tokens), Nathan's
midterm subset. **Length axis:** 40 questions spread over JEVal's 9 datasets whose own state is >= 2,048
reference tokens, each state cut to its first 32, 128, 256, 512, 768, 1,024, 1,536 and 2,048 tokens (320
timed questions per model, the same questions at every length). Length-axis timings use the median GPU time
per length.

### What we found

- **On big GPUs, batch-1 decisions are overhead-bound, not memory- or compute-bound.**
  - On the cluster L4 the models reach 48–74% of the attainable roof (laya and plain-PyTorch granite lower).
    On the H100 and H200 the same models reach 7–14% (kev-4b.hybrid 28–35%, laya and granite under 5%).
  - qwen3-8b on the H100 at natural lengths: ~16 GB of weights stream in ~5 ms at the measured 2,980 GB/s,
    the math needs ~7 ms at the measured 677 TFLOP/s, and the forward takes 62 ms. About 50 ms per question
    is neither: kernel launches and framework work that do not shrink with a faster GPU.
  - The H100 is ~11x the L4's measured compute and ~13x its bandwidth; the models run 1.4–2.6x faster on it
    (kev-4b.hybrid 6.4x), and laya runs 3x slower.
- **The H200-vs-H100 test: no memory-bound regime appears.** The pair has the same compute and 1.41x the
  measured bandwidth. A memory-bound pass should run ~1.4x faster on the H200, a compute-bound one ~1.0x, and
  the speedup should fall from one to the other as the state grows through the band between the two ridges
  (160–227 FLOP/byte).
  - What we measured: ~1.1x at every length for the dense 8–9B models (1.02–1.16), the MoE bases (1.07–1.15),
    elephant-1 (1.07–1.10) and plain-PyTorch granite (1.04–1.05). No 1.4x at short lengths, no fall.
  - The short end is not memory-bound because it is overhead-bound: GPU time barely changes from 32 to 768
    state tokens on either card (qwen3-8b 62 → 71 ms on the H100). The long end does approach compute:
    qwen3-8b reaches 304 TFLOP/s (45% of the measured peak) at 2,048 tokens on the H100, where the H200 is
    1.08x.
  - Some models are slower on the H200: kev-4b.qwen3 0.66–0.93x, granite with fused Mamba kernels
    0.80–1.01x, kev-4b.hybrid 0.95x past 1,024 tokens. A slower result on a strictly larger GPU means the
    hardware is not what sets the time.
  - laya is the only family above 1.2x (1.19–1.49x), and that is a kernel effect, not bandwidth (below).
- **Where the hardware does bind, the roofline predicts.** On the L4, GPU time grows with the state: qwen3-8b
  137 → 796 ms from 32 to 2,048 tokens, kev-4b.qwen3 54 → 519 ms, at 29–43 TFLOP/s of a measured 57. The
  A100 (PCIe, for these models) is in between (qwen3-8b 43 → 221 ms, 155 TFLOP/s of 230 at 2,048 tokens). On the H100, H200 and B200
  the same curves are nearly flat until ~1,024 tokens. The roofline describes a model well when moving its
  bytes and FLOPs takes long next to the fixed per-question overhead: true on an L4, not on HBM GPUs.
- **A length-axis curve that climbs parallel to the memory slope is not evidence of a memory bound.** As the
  state grows, intensity rises ~7x (qwen3-8b 280 → 2,074 FLOP/byte) while GPU time stays flat, so achieved
  FLOP/s rises with the work and each curve runs up and to the right like a bandwidth-bound one. Only the
  controlled H100/H200 pair tells the two apart.
- **Software sets the time on big GPUs.**
  - **CUDA graphs:** kev-4b.hybrid (fused Triton DeltaNet kernels, graphs over padded shape buckets) is the
    fastest model on every big GPU (10–14 ms natural) and the only one that scales with the hardware (6.4x
    L4 → H100). Its time and kev-4b.qwen3's jump ~2.5–4x between 768 and 1,024 state tokens on the H100, H200
    and B200 (H100: 25 → 67 ms and 28 → 68 ms), which looks like the end of the captured shape range (not checked in the kev code).
  - **Fused kernels:** granite-4.0-H-tiny with mamba-ssm and causal-conv1d is 2.5x faster than its
    plain-PyTorch fallback on the H100 natural set (114 vs 276 ms) and ~11x at 2,048 tokens; the
    midterm's 1.7 s per question on the L4 was the fallback, not the architecture.
  - **Attention kernel and input shape (laya):** below its context limit laya sees a new input length on
    every question; at or above it every input is cut to one shape. On the GPUs where PyTorch picks cuDNN's
    SDPA kernel (H100, H200, B200) laya takes 64–122 ms per question below the limit and 13–26 ms at it,
    and the drop sits exactly at each checkpoint's limit (512 tokens English, 1,024 multilingual). On the GPUs
    where it picks the CUTLASS memory-efficient kernel (A100, L4) laya is flat at 27–33 ms at every length.
    The likely cause is a per-shape setup cost in the cuDNN path; forcing the other SDPA backend would test
    it (not run). It also explains laya being 3x slower on the H100 than on the cluster L4 at natural lengths.
- **The B200 is the one big step, for one backbone.** qwen35-9b and jevk5-9b (Qwen3.5, Gated DeltaNet
  hybrid) take 21–24 ms on the B200 at short states against 53–65 ms on the H100, and reach ~600 TFLOP/s at
  2,048 tokens. Dense qwen3-8b gains 1.6x, the MoE bases ~1.5x, elephant-1 and fused granite ~1.0x, and kev-4b.qwen3 is
  slower (0.7x at short states). The Qwen3.5 gain is too large for bandwidth or compute alone (2.15x and 1.96x the H100's measured
  ceilings); a lower-overhead kernel path for its mixer is the likely reason (not profiled).
- **OpenJev (27B, vLLM, TP=1) on four 80 GB GPUs.** Accuracy is 77–79% on every GPU (200 questions, noise
  ±6 points), 6 invalid everywhere (prompts over the 16,384-token limit). Questions/s at 8 in flight and at 1:
  A100 2.49 / 2.11, H100 6.68 / 5.45, H200 6.62 / 5.17, B200 10.31 / 6.49. The H200 equals the H100; the
  B200 is 1.5x the H100 with 8 requests in flight but 1.2x with one. A bigger GPU pays off when it gets work
  in parallel.
- **What still holds from the midterm (L4).** The readout head is a hidden weight read (a 248k-row vocab head
  is 2 GB in bf16 per forward; elephant-1 keeps it in fp32). Splitting state and questions into two passes
  doubles weight traffic. MoE saves FLOPs, not bytes, at decision lengths (intensity 49–60). These are
  properties of the bytes and hold on every GPU. What does not carry over is their cost in time: on HBM GPUs
  the extra bytes take a few milliseconds against tens of milliseconds of overhead, so the attention-only vs
  hybrid ranking flips (kev-4b.qwen3 beat the hybrid on the L4, 66 vs 89 ms; the hybrid wins on the H100,
  14 vs 30 ms) and the MoE bases' L4 advantage over dense qwen3-8b (1.5–1.7x) shrinks to ~1.0x on the H100 (OLMoE 57 ms, LFM2 70 ms, qwen3-8b 62 ms).

### What this suggests for choosing hardware next

- **Fix the software before buying a GPU.** Past an L4, a batch-1 decision model spends most of its time on
  overhead: CUDA graphs, fused kernels, fixed input shapes and several questions per forward move it toward
  its roof, and only then does the roof's memory-vs-compute split decide which GPU helps.
- **More bandwidth alone buys ~1.1x** for these models (H100 → H200). It is not the lever the midterm's L4
  analysis suggested.
- **Cheap GPUs are where the roofline binds.** On the L4 the cost model predicts time well, and its advice
  (bandwidth for two-pass, MoE and encoder recipes) applies there.
- **Measure ceilings and check the card.** Measured peaks are 47–85% of the datasheet, and "A100-80GB" on
  Modal can be a PCIe card with a lower roof.

### Questions per request (hypothesis H3)

H3: sharing one read of the state across many questions is the only real hardware advantage of these designs;
achieved FLOP/s per pass should rise with Q (questions per request), and most for the high-ridge L4. The Q axis
(`benchmarks/roofline_q.py`, 2026-11-09, Modal L4, A100 SXM, H100, H200, B200) sends Q = 1-32 questions against
one state in one request, at 256 and 2,048 reference tokens: the same 5 states at every point and the first Q of
one fixed 32-question pool, 5 timed requests per point, 0 invalid answers in 30 runs.

- **Sharing the state cuts the cost per question about 10x at Q = 32, and it is less work, not better
  utilization.** At 2,048 tokens bongard-mini (encoder once, decoder batches of 8 against one shared K/V copy)
  and both Kev variants (state prefilled once, a row per question) need 0.06-0.08x the FLOPs per question at
  Q = 32 that they need at Q = 1, and on the L4 and A100 their GPU time per question falls by about the same
  factor (bongard 0.10x on the L4, 0.13x on the A100; kev hybrid 0.08x; kev attention-only 0.06x). Achieved
  TFLOP/s stays flat or falls (bongard L4 30 -> 24, A100 82 -> 50, B200 189 -> 109; kev hybrid 40 -> 30 on
  the L4, 276 -> 225 on the B200): the added question rows are small, lower-intensity passes beside a state
  pass that already sat near the roof. Only at 256 tokens, where a one-question pass is far left of the ridge,
  does FLOP/s rise with Q (bongard 0.30x FLOPs per question; kev hybrid 27 -> 35 TFLOP/s on the L4).
- **The controls behave as designs that share nothing should.** semif.direct (a full prompt per question) holds
  0.99-1.01x GPU time per question on the L4, A100 and B200, and 0.99x FLOPs everywhere. semif.reranker batches
  a question's options, not its questions: flat TFLOP/s; its 0.73x FLOPs per question is the pool's mix of option
  counts (its first question has 4 options, the pool averages ~3.2).
- **Laya gains the opposite way.** Each question is its own encoder row with the state (1.00x FLOPs per
  question), but a request's rows run as one batch, so FLOP/s rises with Q on the big GPUs (B200 25 -> 164,
  H100 25 -> 123 at 2,048, cut to 1,024 tokens) and its cost per question falls 0.15-0.25x there, against
  0.74x on the L4, which one row already fills. That is plain batching, available to any design.
- **Bongard's batches of 8 show as steps.** Its forwards per request go 3 -> 4 at Q = 9 (encoder, K/V projection,
  one decoder pass per 8 questions); on the L4, A100 and B200 each new decoder pass adds 30-107 ms at Q = 9, 17
  and 25, against 0-36 ms per question inside a batch. On the H100 and H200 the step at 25 is missing.
- **No ridge-point ordering.** The cost-per-question ratio at Q = 32 is set by the FLOP amortization and is
  about the same on every GPU (kev hybrid 0.07-0.08x on all five); the L4 gains no more than the others.

So H3 holds in its first half (a shared state is a large, real per-question saving, 10-16x at Q = 32 and 2,048
tokens, which re-encoding designs cannot get), but not by its stated mechanism: past a few hundred state tokens
the saving is work avoided, with utilization flat or lower, and it does not grow with the GPU's ridge point.

Caveat for this section: on the sm90+ GPUs (H100, H200, B200) PyTorch's default cuDNN attention builds a plan
per new (query, key) shape on the host. Every pool question is a new shape the first time it appears, and points
run in ascending Q, so a Q that adds several new questions pays more (semif.direct on the H100: 62.7 ms per
question at Q = 8, 4 new, against 38.7 at Q = 9, 1 new), and their cost-per-question curves fall faster than the
work does (semif.direct 0.53-0.71x there). The first point of most sm90+ runs (Q = 1 at 256) is slow the same
way. The statements above rest on the L4 and A100 and on FLOP counts; the H100 and H200 timings are biased in
favor of H3.

### Caveats

- **One run per (model, GPU, mode).** No repeats, so differences of a few percent are noise. Platforms are
  never compared on one roof: the Modal L4 measured 3–18% slower than the cluster L4.
- **Batch 1 only.** One question per request (OpenJev's shape sweep and 8-in-flight runs aside). Batching is
  the measurement this analysis most needs next.
- **The short end of the length axis is not short.** At 32 state tokens each prompt is still ~150–300 tokens
  (instructions and question text), so curves start at 49–280 FLOP/byte, not near zero.
- **The length-axis sample** is 40 questions from 9 datasets, most of them legal texts (the only JEVal states
  of >= 2,048 tokens); accuracy at cut lengths is not meaningful, only timing.
- **Hardware substitutions and failures.** Modal ran the six-model A100 length run on an A100 80GB PCIe (no
  datasheet roof drawn for it). Plain-PyTorch granite on the L4 failed every question at >= 768 state tokens
  and some at 512 (178 of 320 invalid), so its curve stops at 512.
- **Theoretical FLOPs and bytes.** Compulsory traffic is a lower bound; real intensities are if anything lower.
  Hardware-counter (ncu) runs were not repeated on the Modal GPUs.
- **Causes marked "likely" are hypotheses.** The laya cuDNN shape cost, kev's 1,024-token cliff and the B200's
  Qwen3.5 gain are read from timings and kernel names, not profiled.
