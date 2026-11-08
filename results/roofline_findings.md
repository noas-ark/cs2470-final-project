How the design choices that turn a base model into a decision model use one cheap GPU. Runs of 2026-11-04,
one **NVIDIA L4** each (24 GB, 72 W). All models answered the same 200 JEVal questions (choice and yes/no, at
most 16 options, at most 1,024 reference tokens), one question per request at batch 1. Each forward was
timed with CUDA events, and its FLOPs and compulsory DRAM bytes come from the analytic cost model
(`s1.flops`, checked against the Hub's parameter counts). The L4's measured ceilings are about **59 TFLOP/s
bf16** and **231 GB/s** (the datasheet says 121 and 300), so its ridge point is about **257 FLOP/byte**.
Accuracy on 200 questions carries roughly ±6.5 points (95%).

### What we found

- **At batch 1, the recipe decides which roof binds.** Intensities run from 105 to 308 FLOP/byte against
  a ridge of about 257, because a single prefill reads every weight once for only ~200–340 prompt tokens.
  - **Compute-bound, barely:** the single-pass dense 9B models: JevK5-9B and its Qwen3.5 base at 291, and
    zero-shot Qwen3-8B at 308. They reach only 51–56% of the measured bf16 peak, so their gap is serving
    software, not bandwidth.
  - **At the ridge:** kev-4b.qwen3, at 236.
  - **Memory-bound:** the two-pass recipes (elephant-1 at 120, kev-4b.hybrid at 130), laya's encoders (about
    105) and the MoE bases (56–60). For these, GPU time follows the bytes a recipe makes the GPU read:
    0.5 GB per question for laya.multilingual, 14.5 GB for kev-4b.hybrid and 36 GB for elephant-1.
  - The bytes are compulsory traffic, a lower bound, so real intensities are if anything lower. Every point
    would shift toward the memory side.
- **The readout head is a hidden weight read.** A vocab readout (JevK5, the Qwen3.5 control, elephant-1) runs
  a 248k × 4096 LM head, 2.0 GB in bf16, on every forward, only to read a handful of letter or slot logits.
  elephant-1 keeps that head in fp32 for calibration, which makes 4.1 GB per pass and about 8 GB of its
  36 GB per question. kev's pointer head and laya's marker head read hidden states directly and skip it.
- **Splitting the state from the questions doubles weight traffic unless the state is reused.**
  - elephant-1 prefills the state once and forks the questions in a second batched pass. kev-4b.hybrid runs a
    graphed state pass and then a row pass. Both take 2 forwards per question.
  - With one question per request, elephant-1 reads 2.25× JevK5's bytes for 0.93× its FLOPs. Its intensity
    is 120 against 291, and it takes 225 ms per question against 156 ms on the same backbone.
  - This subset is the worst case for that design. The second pass pays off only when several questions
    share a state, or when the prefix cache hits. A questions-per-request sweep is the next measurement.
- **Serving engineering moves a recipe up to its roof.**
  - kev-4b.hybrid (fused Triton DeltaNet kernels, CUDA graphs over 79 padded shape buckets) and elephant-1
    (its own engine) both reach **70% of the attainable FLOP/s** at their intensity. Graph padding costs kev
    10% of executed tokens.
  - Plain-transformers single-pass 9B models reach about 51% of the measured bf16 peak.
  - laya's encoders run eager PyTorch at 31% (English) and 14% (multilingual) of their roof. A 395M model
    taking 20 ms per question, when its fp32 weights stream in about 7 ms, is launch-bound. laya would gain
    most from CUDA graphs and bf16 weights.
- **Same recipe, different backbone: the hybrid mixer does not pay at short prompts.**
  - kev's LoRA + pointer-head recipe scores the same on both 4B bases: Qwen3-4B (attention only) 65.5% and
    Qwen3.5-4B (Gated DeltaNet hybrid) 65.0%.
  - The attention-only model packs all questions into one block-masked pass: 66 ms and 14.7 questions/s,
    against 89 ms and 11.0/s for the hybrid's two passes.
  - The hybrid's linear-time mixer is a long-context advantage. Prompts of 1,024 tokens or fewer never
    exercise it. On full JEVal, long prompts included, the hybrid led 70.9% to 58.7% (README).
- **Accuracy per GPU-millisecond on an L4.**
  - elephant-1 is the most accurate and best calibrated: 75.5%, ECE 0.071, 225 ms.
  - JevK5-9B and its zero-shot base are within noise of each other: 70.5% and 72.5%, both about 155 ms.
    Fine-tuning mostly improved calibration (ECE 0.108 against 0.138).
  - **kev-4b.qwen3 is the value point:** 65.5% at 66 ms, 2.3× the questions/s of the 9B models, in 7.4 GB/q.
  - laya.english is the floor: 44% at 20 ms.
  - elephant-1 buys about 3–5 points over its own base for 1.45× the GPU time.
- **MoE saves FLOPs, not bytes, at decision lengths.**
  - No decision-trained MoE fits one L4: decider-35B-A3B is 69 GB.
  - Zero-shot MoE bases of about 8B show the behavior. OLMoE and LFM2 compute with 1.1–1.4B active
    parameters, but their ~350-token prompts touch nearly every expert. They read 13.7–16.8 GB per question,
    about what a dense 8B reads.
  - Their intensity is therefore 56–60 FLOP/byte, far left of the ridge, at about 70% of the roof. They run
    only 1.5–1.7× faster than dense Qwen3-8B, despite 5× fewer active FLOPs.
  - A decision MoE is a bandwidth-hardware question.
- **Software availability can dominate the architecture.** Granite-4.0-H-tiny's Mamba2 layers fell back to
  plain PyTorch, because mamba-ssm and causal-conv1d are not installed. It ran at 0.5 TFLOP/s and 1.7 s per
  question, so it is not evidence about the architecture.

### What this suggests for choosing hardware next

Batch-1 decisions on an L4 are bound by memory bandwidth for every recipe here except the single-pass dense
9B models. Those sit just past the ridge and lose about half their peak to software. So:

- **Bandwidth-bound recipes** are the two-pass recipes (elephant-1, kev-4b.hybrid), MoE backbones and small
  encoders. They should gain most from GPUs with more GB/s per dollar, such as an A10G (600 GB/s) or HBM parts.
- **Compute-bound recipes** are single-pass dense models, at long prompts or when batched. They gain from
  tensor throughput.
- **Measure the ceilings, not the datasheet.** The power-capped L4 sustained about half its datasheet bf16
  rate, and the roofline must be drawn from measured ceilings.

### Caveats

- **Single questions only.** One question per request and prompts of at most 1,024 tokens. This favors
  single-pass recipes and hides prefix caching, shared-state forks and long-context mixers.
- **Prompt lengths differ by model.** Each model's tokenizer and prompt format changes the tokens it
  processes for the same question (212–350 per question). laya truncates states to its context.
- **Theoretical FLOPs and bytes.** FLOPs and bytes come from the cost model (compulsory traffic, useful
  attention pairs, padded shapes as executed). The Nsight Compute hardware-counter runs (`s1 ncu`) did not
  finish this round. torch's FlopCounterMode agrees within 1% for the dense models, misses grouped-expert
  matmuls on MoEs (ratio 0.3), and cannot see kev's CUDA-graph replays.
- **Not run.** bongard-mini (T5Gemma-2 encoder-decoder) needs its own torch ≥ 2.14 runtime. decider-35B-A3B
  and djev do not fit, or are not available, on one L4.
- **Records marked `-dirty`.** Records whose commit ends in `-dirty` ran with only docs or report code
  uncommitted; the run code matches the named commit. kev ran twice (pre-commit and at 312c12a), and the
  repeats agree within 1%.
