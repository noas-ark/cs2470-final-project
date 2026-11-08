"""Every model the harness knows how to run, with pinned revisions and exact sizes.

Pins and sizes come from read-only Hub/GitHub metadata (`model_info`, file listings, `gh api`);
no weights are downloaded and no model code is run to fill them in. Full 40-char SHAs only.
`size_bytes` sums the weight files (*.safetensors, *.bin, *.pt, adapters and heads; no GGUF).
A "proposed pin" is the default-branch HEAD on `resolved_on`, not something anyone has verified.
Add a spec as "planned" (no backend); flip it to "ready" with a `backend` key once the adapter exists.
"""

from __future__ import annotations

from dataclasses import replace

from .spec import Group, ModelSpec

RESOLVED = "2026-10-25"

OPENJEV = ModelSpec(
    name="openjev",
    group="reference",
    status="ready",
    backend="vllm_shim",
    repo="openjev/openjev",
    license="cc-by-nc-4.0",
    resolved_on=RESOLVED,
    size_bytes=54_713_606_768,
    notes="Unpinned: the local copy's revision is unchecked (Hub HEAD b4a89f58 on 2026-10-25). "
    "Hub card names no base model (tags say qwen3_5).",
)

LAYA_REV = "7b928d828b7b0e022f929d9bd2e44165aa270148"
LAYA_MULTILINGUAL = ModelSpec(
    name="laya.multilingual",
    group="E",
    status="ready",
    backend="laya",
    repo="convaiinnovations/laya",
    revision=LAYA_REV,
    license="apache-2.0",
    resolved_on=RESOLVED,
    size_bytes=643_835_514,  # multilingual/model.safetensors only
    options={"question_mode": "reencode", "checkpoint": "multilingual"},
    notes="Checkpoint in multilingual/ (mmBERT-base, 1024-token sequences). Revision is a proposed pin "
    "(unpinned in our doc); the laya package's own reviewed pin is 55cf4c4e. The repo also ships a "
    "fine-tuned typed-decisions/ checkpoint we don't list.",
)
# Same repo and revision; the root checkpoint is the English one, capped at 512 tokens.
LAYA_ENGLISH = LAYA_MULTILINGUAL.variant(
    "english",
    size_bytes=842_609_210,  # root model.safetensors only
    max_state_tokens=512,
    options={"checkpoint": ""},
    notes="Checkpoint at repo root (ModernBERT-large); 512 tokens for the whole sequence, so the state "
    "gets fewer. Revision is a proposed pin (unpinned in our doc).",
)

KEV_HYBRID = ModelSpec(
    name="kev-4b.hybrid",
    group="D-hybrid",
    status="ready",
    backend="kev",
    repo="jaredpalmer/kev-4b",
    revision="139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
    base_repo="Qwen/Qwen3.5-4B-Base",
    base_revision="1001bb4d826a52d1f399e183466143f4da7b741b",
    code_repo="jaredpalmer/kev",
    code_revision="2074cc44256200bb37540103fa4572901faa10f3",
    license="apache-2.0",
    resolved_on=RESOLVED,
    size_bytes=135_173_823,  # adapter_model.safetensors + head.pt
    base_size_bytes=9_319_828_056,
    options={"question_mode": "row"},
    notes="Adapter + head on the base. code_revision is a proposed pin (kev main HEAD).",
)
KEV_QWEN3 = KEV_HYBRID.variant(
    "qwen3",
    group="D-attn",
    revision="c4bfa11b0dc07691884f2d97f1c4c4c05c92e416",
    base_repo="Qwen/Qwen3-4B-Base",
    base_revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
    size_bytes=137_432_991,
    base_size_bytes=8_044_982_000,
    notes="Hub tag `qwen3`; revision is what the Hub resolves it to (the refs list shows tag target "
    "c113c66d, which resolves to the same c4bfa11b). Base revision is a proposed pin (Qwen3-4B-Base HEAD), "
    "unverified against the adapter. code_revision is a proposed pin (kev main HEAD).",
)

BONGARD_MINI = ModelSpec(
    name="bongard-mini",
    group="ED",
    status="ready",
    backend="bongard",
    repo="AgentBull/bongard-mini",
    revision="c81c5b883a5983669be84be98658eac940489613",
    base_repo="google/t5gemma-2-4b-4b",
    base_revision="487d4acf21a4d70c70bf534265b5263c9424979e",
    code_repo="AgentBull/bongard",
    code_revision="3e6e6030a953dfe8d30d93aca2d0ab92b87b4a5a",
    license="gemma",
    resolved_on=RESOLVED,
    size_bytes=15_020_655_896,  # backbone/model.safetensors + head.safetensors (backbone is bundled)
    options={"question_mode": "shared_encoder", "require_gpu": "rtx pro 4500"},
    notes="Gemma terms. The backbone is bundled, so the gated base repo is not needed at runtime. Upstream "
    "wants Python >=3.12, transformers==5.17.0 and torch>=2.14 (we have 2.13). RTX Pro 4500 (32 GB) only.",
)

DECIDER = ModelSpec(
    name="decider-35b-a3b",
    group="MoE",
    repo="Mapika/decider-35b-a3b",
    revision="91470e65478656eb68fde8c0ccdfe575d69d0309",
    base_repo="Qwen/Qwen3.5-35B-A3B-Base",
    base_revision="0f0813072d2358973511097385626f21fcb6d422",
    license="apache-2.0",
    resolved_on=RESOLVED,
    size_bytes=69_325_579_424,  # 15 safetensors shards
    options={"question_mode": "independent"},
    notes="Hub weights are 69.3 GB; our doc said 65 GB. Both revisions are proposed pins (HEAD). "
    "License from Hub card metadata.",
)

SEMIF_DIRECT = ModelSpec(
    name="semif.direct",
    group="control",
    status="ready",
    backend="semif",
    repo="Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    code_repo="TheoLeeCJ/SemIf",
    code_revision="23cf1f39fc9534fe81437200959b6dfc7106e45a",
    license="MIT (SemIf code); apache-2.0 (Qwen3.5-4B weights)",
    resolved_on=RESOLVED,
    size_bytes=9_319_828_096,
    options={"mode": "direct", "question_mode": "row"},
    notes="code_revision is a proposed pin (SemIf master HEAD).",
)
SEMIF_RERANKER = SEMIF_DIRECT.variant(
    "reranker",
    repo="Qwen/Qwen3-Reranker-4B",
    revision="22e683669bc0f0bd69640a1354a6d0aebcfeede5",
    size_bytes=8_043_614_960,
    options={"mode": "reranker"},
    license="MIT (SemIf code); apache-2.0 (Qwen3-Reranker-4B weights)",
    notes="Reranker revision and code_revision are proposed pins (HEAD).",
)

DJEV = ModelSpec(
    name="djev",
    group="MoE-Diff",
    repo="mmastrac/djev",
    base_repo="google/diffusiongemma-26B-A4B-it",
    base_revision="f7f5b7f5fa82ffc52addd066915886d497f5517b",
    license="unverified",
    resolved_on=RESOLVED,
    notes="Optional. mmastrac/djev returned 401 from the Hub API (private or missing), so no revision or "
    "size. Needs a patched vLLM nightly. Base revision is a proposed pin (HEAD, 51.6 GB, apache-2.0).",
)

# The ~8B L4 set: one parameter band (7-9B, bf16 on one 24 GB L4) varying the token mixer (Gated DeltaNet
# hybrid, plain attention, short conv, Mamba2), dense vs MoE, and the readout. Six run through SemIf's
# letter-softmax readout on the same transformers runtime, so only the architecture differs; elephant-1 runs
# its own optimized engine. Revisions are the default-branch HEAD on the date (proposed pins).
L4_RESOLVED = "2026-11-04"


def zeroshot(name: str, repo: str, *, revision: str, size_bytes: int, group: Group, notes: str) -> ModelSpec:
    """An off-the-shelf model read out zero-shot by SemIf's direct letter softmax (no decision training)."""
    weights = "lfm1.0" if repo.startswith("LiquidAI/") else "apache-2.0"
    return replace(
        SEMIF_DIRECT,
        name=f"{name}.zeroshot",
        group=group,
        repo=repo,
        revision=revision,
        size_bytes=size_bytes,
        license=f"MIT (SemIf code); {weights} ({repo} weights)",
        resolved_on=L4_RESOLVED,
        notes=notes,
    )


QWEN35_9B = zeroshot(
    "qwen35-9b",
    "Qwen/Qwen3.5-9B",
    revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    size_bytes=19_306_310_880,
    group="control",
    notes="Base of jevk5-9b and elephant-1. 24 Gated DeltaNet + 8 gated-attention layers; text tower only "
    "(the vision tower and MTP layer in the size are not loaded).",
)
JEVK5_9B = replace(
    QWEN35_9B,
    name="jevk5-9b",
    group="D-hybrid",
    repo="alibiserikbay/JevK5-9B",
    revision="d6521a18a86999190e9d775c915af3d6d6772fc4",
    size_bytes=17_907_663_008,
    license="MIT (SemIf code); apache-2.0 (JevK5-9B weights)",
    options={**QWEN35_9B.options, "semif_temperature": 1.316},
    notes="Qwen3.5-9B with a merged LoRA trained on SemIf's prompt; readout = SemIf's letter softmax at the "
    "card's fitted temperature (jevk5_config.json; the knockout pass for >16 options is not implemented). "
    "Upstream's runtime (allebee/jevk5) adds CUDA graphs; here it runs on the same eager transformers path.",
)
QWEN3_8B = zeroshot(
    "qwen3-8b",
    "Qwen/Qwen3-8B",
    revision="b968826d9c46dd6066d109eabc6255188de91218",
    size_bytes=16_381_516_776,
    group="control",
    notes="Dense, plain attention (36 layers): the dense control for the MoE models.",
)
OLMOE = zeroshot(
    "olmoe-1b-7b",
    "allenai/OLMoE-1B-7B-0125-Instruct",
    revision="b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e",
    size_bytes=13_838_721_960,
    group="MoE",
    notes="MoE with plain attention: 6.9B total, ~1.1B active (64 experts, top-8, 16 layers). 4k context.",
)
LFM2_MOE = zeroshot(
    "lfm2-8b-a1b",
    "LiquidAI/LFM2-8B-A1B",
    revision="7ed1404fdf502a5446c1b67d4fe1533158d6f2ad",
    size_bytes=16_680_151_504,
    group="MoE",
    notes="MoE with 18 short-conv + 6 attention layers: 8.3B total, ~1.4B active (32 experts, top-4). "
    "LFM Open License v1.0. Short convolutions run in plain torch without causal-conv1d.",
)
GRANITE_H = zeroshot(
    "granite4-h-tiny",
    "ibm-granite/granite-4.0-h-tiny",
    revision="791e0d3d28c86e106c9b6e0b4cecdee0375b6124",
    size_bytes=13_878_142_256,
    group="MoE",
    notes="MoE with 36 Mamba2 + 4 attention layers: 6.9B total, ~1.3B active (64 experts, top-6, plus a "
    "shared expert). Without mamba-ssm, transformers runs the Mamba2 scan in plain torch.",
)
ELEPHANT = ModelSpec(
    name="elephant-1",
    group="D-hybrid",
    status="ready",
    backend="elephant",
    repo="baselabs/elephant-1",
    revision="b3f8a93462fb1051a9de031e94a7ff380e254072",
    base_repo="Qwen/Qwen3.5-9B",
    base_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    code_repo="OmniJev/OneJev",
    code_revision="81ce62f1597c91e46767d4d02d6ac2e18534fe94",
    license="mit",
    resolved_on=L4_RESOLVED,
    size_bytes=18_819_722_392,
    options={"head_dtype": "float32"},
    notes="Full fine-tune of Qwen3.5-9B, served on OneJev qev @81ce62f + the repo's serve.patch in its own "
    "runtime (runtimes/elephant: torch 2.8, transformers 5.19): block-bidirectional full attention, one "
    "shared-prefix prefill per request, slot_lm readout, fp32 LM head. #1 of 6-13B on Decision Index 0.4.1.",
)

CATALOG: list[ModelSpec] = [
    OPENJEV,
    LAYA_MULTILINGUAL,
    LAYA_ENGLISH,
    KEV_HYBRID,
    KEV_QWEN3,
    BONGARD_MINI,
    DECIDER,
    SEMIF_DIRECT,
    SEMIF_RERANKER,
    DJEV,
    QWEN35_9B,
    JEVK5_9B,
    QWEN3_8B,
    OLMOE,
    LFM2_MOE,
    GRANITE_H,
    ELEPHANT,
]
