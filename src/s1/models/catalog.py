"""Every model the harness knows how to run, with pinned revisions and exact sizes.

Pins and sizes come from read-only Hub/GitHub metadata (`model_info`, file listings, `gh api`);
no weights are downloaded and no model code is run to fill them in. Full 40-char SHAs only.
`size_bytes` sums the weight files (*.safetensors, *.bin, *.pt, adapters and heads; no GGUF).
A "proposed pin" is the default-branch HEAD on `resolved_on`, not something anyone has verified.
Add a spec as "planned" (no backend); flip it to "ready" with a `backend` key once the adapter exists.
"""

from __future__ import annotations

from .spec import ModelSpec

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
    repo="convaiinnovations/laya",
    revision=LAYA_REV,
    license="apache-2.0",
    resolved_on=RESOLVED,
    size_bytes=643_835_514,  # multilingual/model.safetensors only
    options={"question_mode": "reencode"},
    notes="Checkpoint in multilingual/. Revision is a proposed pin (unpinned in our doc).",
)
# Same repo and revision; the root checkpoint is the English one, capped at 512 tokens.
LAYA_ENGLISH = LAYA_MULTILINGUAL.variant(
    "english",
    size_bytes=842_609_210,  # root model.safetensors only
    max_state_tokens=512,
    notes="Checkpoint at repo root; 512-token cap. Revision is a proposed pin (unpinned in our doc).",
)

KEV_HYBRID = ModelSpec(
    name="kev-4b.hybrid",
    group="D-hybrid",
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
    repo="AgentBull/bongard-mini",
    revision="c81c5b883a5983669be84be98658eac940489613",
    base_repo="google/t5gemma-2-4b-4b",
    base_revision="487d4acf21a4d70c70bf534265b5263c9424979e",
    code_repo="AgentBull/bongard",
    code_revision="3e6e6030a953dfe8d30d93aca2d0ab92b87b4a5a",
    license="gemma",
    resolved_on=RESOLVED,
    size_bytes=15_020_655_896,  # backbone/model.safetensors + head.safetensors (backbone is bundled)
    options={"question_mode": "shared_encoder"},
    notes="Gemma terms; base repo is gated (manual approval). Needs Python >=3.12 and transformers==5.17.0.",
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
]
