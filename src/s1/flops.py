"""Analytic cost of a decision model's forward passes: FLOPs and compulsory DRAM bytes, from its HF config.

Torch-free, so it is unit-tested on the login node. Python 3.11-compatible syntax: the elephant runtime's
worker (Python 3.11) imports nothing from here, but keep it importable there anyway.

A decision model only runs prefill: every question is one or more forward passes over a prompt, with the
logits read at one (or a few) positions. A `Forward` is one such pass as recorded on the GPU (backends record
them with `s1.profiling.ForwardRecorder`); `cost(arch, fwd)` turns it into

    flops   = 2 * rows * tokens * (weights a token passes through: dense layers + router + top_k experts)
            + attention score/value matmuls over the causal (query, key) pairs, past included
            + recurrent-state updates (Gated DeltaNet, Mamba2): ~6 FLOPs per state element per token
            + LM head at the kept positions only (`logits_to_keep`)
    bytes   = every weight the pass touches, read once (a MoE layer touches the experts its tokens route to:
              E * (1 - (1 - k/E) ** routed_tokens), uniform routing assumed)
            + embedding rows gathered + the residual stream written and read once per layer
            + KV written (when the pass caches) and the past KV read; recurrent states read/written

so `flops / bytes` is the operational ("algorithmic") intensity against compulsory traffic. It is a lower
bound on real traffic: intermediate activations that spill out of L2 are not counted. FLOPs count useful
work: causal attention counts the (n+1)/2 average keys a query sees, not the full masked square a non-flash
kernel computes, and padding rows are counted because the GPU executes them. Elementwise ops (norms, rotary,
activations) are left out; they are well under 1% of the matmul FLOPs at these sizes.

Supported `model_type`s: qwen3_5 / qwen3_5_text (Gated DeltaNet + gated attention), qwen3, olmoe, lfm2_moe
(short conv + attention, MoE), granitemoehybrid (Mamba2 + attention, MoE) and modernbert (a bidirectional
encoder: every query sees every key, or a `local_attention` window on its sliding layers; Laya).
`tests/test_flops.py` checks the derived parameter count of each against the Hub's safetensors total.

An encoder-decoder (t5gemma2: Bongard-mini) has several stacks, each its own `Arch`: `stacks(config)`
returns {"encoder", "cross_kv", "decoder"}, and a recorded forward names its stack (`Forward.stack`). Its
decoder reads the encoded state as `past` keys (merged self + cross attention), one copy shared by every
question row of the batch (`shared_past`). `arch(config)` refuses it: charging every pass to one stack
would be silently wrong.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

BF16_BYTES = 2
STATE_BYTES = 4  # recurrent states are kept in fp32 (mamba_ssm_dtype / fla)
STATE_FLOPS_PER_ELEMENT = 6  # decay, rank-1 (delta) update and readout of each state element per token


@dataclass(frozen=True)
class Layer:
    """One decoder layer. `dense_params` are weights every token passes through (mixer, dense FFN,
    router, shared expert, norms); a MoE layer adds `top_k` of its `experts` routed experts of
    `expert_params` each."""

    mixer: str  # attention | gated_deltanet | mamba2 | short_conv
    dense_params: int
    expert_params: int = 0
    experts: int = 0
    top_k: int = 0
    q_heads: int = 0  # attention: query heads x head_dim for the score/value matmuls
    head_dim: int = 0
    kv_dim: int = 0  # attention: K and V width each (kv_heads * head_dim), cached per token
    state_elements: int = 0  # recurrent layers: state size per row (heads * key_dim * value_dim)
    causal: bool = True  # attention: False for an encoder, where a query sees keys on both sides
    window: int = 0  # attention: keys a query sees at most (sliding window), 0 for all


@dataclass(frozen=True)
class Arch:
    model_type: str
    hidden: int
    vocab: int
    tied: bool
    layers: tuple[Layer, ...]
    extra_params: int  # embedding (+ untied LM head) + final norm

    @property
    def params(self) -> int:
        """Total parameters, all experts included (what the checkpoint stores)."""
        return self.extra_params + sum(ly.dense_params + ly.experts * ly.expert_params for ly in self.layers)

    @property
    def active_params(self) -> int:
        """Parameters one token passes through, excluding the embedding (a lookup) and the LM head."""
        return sum(ly.dense_params + ly.top_k * ly.expert_params for ly in self.layers)


@dataclass(frozen=True)
class Forward:
    """One forward pass as executed: `rows` x `tokens` new tokens (padded width), attending to `past` cached
    tokens each, with logits computed at `kept` positions per row. `ms` is its GPU time (CUDA events)."""

    rows: int
    tokens: int
    past: int = 0
    kept: int = 1
    cached: bool = False  # the pass writes its K/V and states to a cache
    ms: float = math.nan
    shared_past: bool = False  # every row attends to ONE copy of the past K/V (read once, not once per row)
    stack: str = ""  # which of `stacks(config)` ran it ("" for a single-stack model)


@dataclass(frozen=True)
class Cost:
    flops: float
    bytes: float
    ms: float
    tokens: int  # rows * tokens: prompt tokens processed

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            self.flops + other.flops, self.bytes + other.bytes, self.ms + other.ms, self.tokens + other.tokens
        )


ZERO = Cost(0.0, 0.0, 0.0, 0)


def _int(cfg: Mapping[str, object], key: str, default: int | None = None) -> int:
    val = cfg.get(key, default)
    if not isinstance(val, int) or isinstance(val, bool):
        raise KeyError(f"config has no integer {key!r} (got {val!r})")
    return val


def _text_config(cfg: Mapping[str, object]) -> Mapping[str, object]:
    text = cfg.get("text_config")
    return {str(k): v for k, v in text.items()} if isinstance(text, Mapping) else cfg


def _layer_types(cfg: Mapping[str, object], key: str, n: int) -> list[str]:
    types = cfg.get(key)
    if not isinstance(types, list) or len(types) != n or not all(isinstance(t, str) for t in types):
        raise KeyError(f"config needs {key!r}: one layer type per layer ({n})")
    return [str(t) for t in types]


def _attention(cfg: Mapping[str, object], d: int, *, gate: bool, qk_norm: str) -> Layer:
    """q/k/v/o projections; `gate` doubles q_proj (Qwen3.5's output gate); `qk_norm` is "head" (one weight
    per head_dim, Qwen3), "full" (over all q and kv widths, OLMoE) or "" (none)."""
    heads, kv_heads = _int(cfg, "num_attention_heads"), _int(cfg, "num_key_value_heads")
    hd = _int(cfg, "head_dim", d // heads)
    q, kv = heads * hd, kv_heads * hd
    params = d * q * (2 if gate else 1) + 2 * d * kv + q * d
    params += {"head": 2 * hd, "full": q + kv, "": 0}[qk_norm]
    return Layer("attention", params, q_heads=heads, head_dim=hd, kv_dim=kv)


def _swiglu(d: int, inter: int) -> int:
    return 3 * d * inter


def qwen35(cfg: Mapping[str, object]) -> Arch:
    d, n, vocab = _int(cfg, "hidden_size"), _int(cfg, "num_hidden_layers"), _int(cfg, "vocab_size")
    types = cfg.get("layer_types") or [
        "full_attention" if (i + 1) % _int(cfg, "full_attention_interval") == 0 else "linear_attention"
        for i in range(n)
    ]
    types = _layer_types({"layer_types": types}, "layer_types", n)
    hk, hv = _int(cfg, "linear_num_key_heads"), _int(cfg, "linear_num_value_heads")
    dk, dv = _int(cfg, "linear_key_head_dim"), _int(cfg, "linear_value_head_dim")
    key_dim, value_dim = hk * dk, hv * dv
    conv = (2 * key_dim + value_dim) * _int(cfg, "linear_conv_kernel_dim")
    # in_proj_qkv + in_proj_z, in_proj_b + in_proj_a, depthwise conv, dt_bias + A_log, gated norm, out_proj
    deltanet = d * (2 * key_dim + 2 * value_dim) + d * 2 * hv + conv + 2 * hv + dv + value_dim * d
    mlp = _swiglu(d, _int(cfg, "intermediate_size")) + 2 * d  # + input/post-attention norms
    layers: list[Layer] = []
    for t in types:
        if t == "full_attention":
            att = _attention(cfg, d, gate=bool(cfg.get("attn_output_gate", True)), qk_norm="head")
            layers.append(replace(att, dense_params=att.dense_params + mlp))
        elif t == "linear_attention":
            layers.append(Layer("gated_deltanet", deltanet + mlp, state_elements=hv * dk * dv))
        else:
            raise ValueError(f"unknown qwen3.5 layer type {t!r}")
    tied = bool(cfg.get("tie_word_embeddings", False))
    return Arch("qwen3_5", d, vocab, tied, tuple(layers), vocab * d * (1 if tied else 2) + d)


def qwen3(cfg: Mapping[str, object]) -> Arch:
    d, n, vocab = _int(cfg, "hidden_size"), _int(cfg, "num_hidden_layers"), _int(cfg, "vocab_size")
    att = _attention(cfg, d, gate=False, qk_norm="head")
    layer = replace(att, dense_params=att.dense_params + _swiglu(d, _int(cfg, "intermediate_size")) + 2 * d)
    tied = bool(cfg.get("tie_word_embeddings", False))
    return Arch("qwen3", d, vocab, tied, (layer,) * n, vocab * d * (1 if tied else 2) + d)


def olmoe(cfg: Mapping[str, object]) -> Arch:
    d, n, vocab = _int(cfg, "hidden_size"), _int(cfg, "num_hidden_layers"), _int(cfg, "vocab_size")
    experts, k = _int(cfg, "num_experts"), _int(cfg, "num_experts_per_tok")
    att = _attention(cfg, d, gate=False, qk_norm="full")
    layer = replace(
        att,
        dense_params=att.dense_params + d * experts + 2 * d,  # + router, norms
        expert_params=_swiglu(d, _int(cfg, "intermediate_size")),
        experts=experts,
        top_k=k,
    )
    tied = bool(cfg.get("tie_word_embeddings", False))
    return Arch("olmoe", d, vocab, tied, (layer,) * n, vocab * d * (1 if tied else 2) + d)


def lfm2_moe(cfg: Mapping[str, object]) -> Arch:
    d, n, vocab = _int(cfg, "hidden_size"), _int(cfg, "num_hidden_layers"), _int(cfg, "vocab_size")
    experts, k = _int(cfg, "num_experts"), _int(cfg, "num_experts_per_tok")
    dense_layers = _int(cfg, "num_dense_layers", 0)  # (use_expert_bias adds a buffer, not stored weights)
    # in_proj (B, C, x gates), depthwise conv over d channels, out_proj
    short_conv = d * 3 * d + d * _int(cfg, "conv_L_cache") + d * d
    layers: list[Layer] = []
    for i, t in enumerate(_layer_types(cfg, "layer_types", n)):
        if t == "full_attention":
            mixer = _attention(cfg, d, gate=False, qk_norm="head")
        elif t == "conv":
            mixer = Layer("short_conv", short_conv)
        else:
            raise ValueError(f"unknown lfm2 layer type {t!r}")
        norms = 2 * d
        if i < dense_layers:  # the first layers have a dense FFN
            layers.append(
                replace(
                    mixer,
                    dense_params=mixer.dense_params + _swiglu(d, _int(cfg, "intermediate_size")) + norms,
                )
            )
        else:
            layers.append(
                replace(
                    mixer,
                    dense_params=mixer.dense_params + d * experts + norms,  # + router
                    expert_params=_swiglu(d, _int(cfg, "moe_intermediate_size")),
                    experts=experts,
                    top_k=k,
                )
            )
    tied = bool(cfg.get("tie_word_embeddings", True))  # LFM2 ties by default
    return Arch("lfm2_moe", d, vocab, tied, tuple(layers), vocab * d * (1 if tied else 2) + d)


def granitemoehybrid(cfg: Mapping[str, object]) -> Arch:
    d, n, vocab = _int(cfg, "hidden_size"), _int(cfg, "num_hidden_layers"), _int(cfg, "vocab_size")
    experts, k = _int(cfg, "num_local_experts"), _int(cfg, "num_experts_per_tok")
    heads, hd, dstate = _int(cfg, "mamba_n_heads"), _int(cfg, "mamba_d_head"), _int(cfg, "mamba_d_state")
    groups, inner = _int(cfg, "mamba_n_groups"), _int(cfg, "mamba_expand") * d
    conv_dim = inner + 2 * groups * dstate
    conv = conv_dim * _int(cfg, "mamba_d_conv") + (conv_dim if cfg.get("mamba_conv_bias", True) else 0)
    # in_proj (z, x, B, C, dt), conv1d, dt_bias + A_log + D, gated norm, out_proj
    mamba = d * (2 * inner + 2 * groups * dstate + heads) + conv + 3 * heads + inner + inner * d
    shared = _swiglu(d, _int(cfg, "shared_intermediate_size"))
    ffn_dense = d * experts + shared + 2 * d  # router, shared expert, norms
    layers: list[Layer] = []
    for t in _layer_types(cfg, "layer_types", n):
        # the Hub's config.json says attention/mamba; transformers 5.x's to_dict() says full_/linear_attention
        if t in {"attention", "full_attention"}:
            mixer = _attention(cfg, d, gate=False, qk_norm="")
        elif t in {"mamba", "linear_attention"}:
            mixer = Layer("mamba2", mamba, state_elements=heads * hd * dstate)
        else:
            raise ValueError(f"unknown granite layer type {t!r}")
        layers.append(
            replace(
                mixer,
                dense_params=mixer.dense_params + ffn_dense,
                expert_params=_swiglu(d, _int(cfg, "intermediate_size")),
                experts=experts,
                top_k=k,
            )
        )
    tied = bool(cfg.get("tie_word_embeddings", True))
    return Arch("granitemoehybrid", d, vocab, tied, tuple(layers), vocab * d * (1 if tied else 2) + d)


def modernbert(cfg: Mapping[str, object]) -> Arch:
    """ModernBERT encoder (the AutoModel backbone, no MLM head): fused Wqkv, GeGLU MLP, pre-norm LayerNorms
    (layer 0 has no attention norm), global attention every `global_attn_every_n_layers`, the rest a sliding
    window of `local_attention` keys."""
    d, n, vocab = _int(cfg, "hidden_size"), _int(cfg, "num_hidden_layers"), _int(cfg, "vocab_size")
    inter, heads = _int(cfg, "intermediate_size"), _int(cfg, "num_attention_heads")
    norm = d * (2 if cfg.get("norm_bias", False) else 1)
    every, window = _int(cfg, "global_attn_every_n_layers"), _int(cfg, "local_attention")
    layers = []
    for i in range(n):
        params = 4 * d * d + 3 * d * inter + (2 if i else 1) * norm  # Wqkv + Wo, Wi (2 x inter) + Wo, norms
        layers.append(
            Layer(
                "attention",
                params,
                q_heads=heads,
                head_dim=d // heads,
                kv_dim=d,
                causal=False,
                window=0 if i % every == 0 else window,
            )
        )
    return Arch("modernbert", d, vocab, True, tuple(layers), vocab * d + 2 * norm)  # embedding + 2 norms


def _gemma3_layer(cfg: Mapping[str, object], d: int, *, causal: bool, window: int) -> Layer:
    """A Gemma 3 block: q/k/v/o with per-head q/k norms, GeGLU MLP (gate, up, down), four RMSNorms."""
    att = _attention(cfg, d, gate=False, qk_norm="head")
    params = att.dense_params + _swiglu(d, _int(cfg, "intermediate_size")) + 4 * d
    return replace(att, dense_params=params, causal=causal, window=window)


def _section(cfg: Mapping[str, object], key: str) -> Mapping[str, object]:
    sub = cfg.get(key)
    if not isinstance(sub, Mapping):
        raise KeyError(f"config has no {key!r} section")
    return _text_config({str(k): v for k, v in sub.items()})


def t5gemma2_stacks(cfg: Mapping[str, object]) -> dict[str, Arch]:
    """T5Gemma 2 (Bongard-mini's backbone), text path only (the vision tower never runs on text).

    encoder   bidirectional Gemma 3 text stack: sliding layers see a `sliding_window`-wide band, full ones all
    cross_kv  the state's K/V projected once per decoder layer (k_proj + k_norm, v_proj: decoder weights)
    decoder   Gemma 3 stack with merged attention: each question token attends causally to its own tokens and
              to every state token (`past`); no separate cross-attention weights. The sliding window only
              limits the self part, which is far shorter than the window, so it is not applied.
    The decoder shares the encoder's embedding (no decoder.embed_tokens in the checkpoint); there is no vocab
    head (a judgment head reads hidden states), so forwards keep 0 positions."""
    enc, dec = _section(cfg, "encoder"), _section(cfg, "decoder")
    d, vocab = _int(enc, "hidden_size"), _int(enc, "vocab_size")
    n_enc, n_dec = _int(enc, "num_hidden_layers"), _int(dec, "num_hidden_layers")
    window = _int(enc, "sliding_window")
    types = _layer_types(enc, "layer_types", n_enc)
    enc_layers = tuple(
        _gemma3_layer(enc, d, causal=False, window=window if t == "sliding_attention" else 0) for t in types
    )
    dec_layers = tuple(
        _gemma3_layer(dec, _int(dec, "hidden_size"), causal=True, window=0) for _ in range(n_dec)
    )
    kv = tuple(Layer("projection", 2 * d * ly.kv_dim + ly.head_dim) for ly in dec_layers)  # k, v, k_norm
    # embedding + the encoder's eoi embedding + one final norm per stack (vision tower and projector excluded)
    return {
        "encoder": Arch("t5gemma2_encoder", d, vocab, True, enc_layers, vocab * d + d + d),
        "cross_kv": Arch("t5gemma2_cross_kv", d, 0, True, kv, 0),
        "decoder": Arch("t5gemma2_decoder", d, vocab, True, dec_layers, d),
    }


STACKS = {"t5gemma2": t5gemma2_stacks}


def stacks(config: Mapping[str, object]) -> dict[str, Arch]:
    """Every stack of a model, by the name its forwards carry: {"": arch(config)} for a single-stack model."""
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type in STACKS:
        return STACKS[model_type](config)
    return {"": arch(config)}


def merged(parts: Mapping[str, Arch]) -> Arch:
    """One Arch holding every stack's layers and parameters (`params`, `active_params`, layer mix), for
    reporting; a cross_kv stack reuses decoder weights, so its parameters are not counted again."""
    if len(parts) == 1:
        return next(iter(parts.values()))
    counted = [a for name, a in parts.items() if name != "cross_kv"]
    first = counted[0]
    return Arch(
        first.model_type.partition("_")[0],
        first.hidden,
        first.vocab,
        first.tied,
        tuple(ly for a in counted for ly in a.layers),
        sum(a.extra_params for a in counted),
    )


BUILDERS = {
    "qwen3_5": qwen35,
    "qwen3_5_text": qwen35,
    "qwen3": qwen3,
    "olmoe": olmoe,
    "lfm2_moe": lfm2_moe,
    "granitemoehybrid": granitemoehybrid,
    "modernbert": modernbert,
}


def arch(config: Mapping[str, object]) -> Arch:
    """The cost model of a HF `config.json` (a VLM's `text_config` is used when present)."""
    cfg = _text_config(config)
    model_type = cfg.get("model_type")
    if isinstance(model_type, str) and model_type in STACKS:
        raise KeyError(
            f"{model_type} has several stacks ({', '.join(STACKS[model_type](cfg))}); use flops.stacks"
        )
    if not isinstance(model_type, str) or model_type not in BUILDERS:
        raise KeyError(f"no cost model for model_type {model_type!r}; known: {', '.join(sorted(BUILDERS))}")
    return BUILDERS[model_type](cfg)


def experts_touched(experts: int, top_k: int, routed_tokens: int) -> float:
    """Expected distinct experts that `routed_tokens` tokens hit with uniform top-k routing."""
    if experts == 0 or routed_tokens == 0:
        return 0.0
    return experts * (1 - math.pow(1 - top_k / experts, routed_tokens))


def cost(a: Arch, f: Forward, *, weight_bytes: int = BF16_BYTES, head_bytes: int = BF16_BYTES) -> Cost:
    tokens = f.rows * f.tokens
    causal_pairs = f.rows * (f.tokens * f.past + f.tokens * (f.tokens + 1) / 2)  # (query, key) pairs
    flops = 2.0 * tokens * a.active_params + 2.0 * f.rows * f.kept * a.hidden * a.vocab
    nbytes = float(a.vocab * a.hidden * head_bytes if f.kept else 0)  # LM head, read once if it runs at all
    nbytes += tokens * a.hidden * weight_bytes  # embedding rows gathered
    for ly in a.layers:
        if ly.mixer == "attention":
            keys = f.past + f.tokens
            pairs = causal_pairs if ly.causal else f.rows * f.tokens * keys
            if ly.window:
                pairs = min(pairs, f.rows * f.tokens * min(keys, ly.window + 1))
            flops += 4.0 * ly.q_heads * ly.head_dim * pairs  # QK^T and PV
            past_read = f.past if f.shared_past else f.rows * f.past
            kv_tokens = past_read + (tokens if f.cached else 0)  # past read + new written
            nbytes += 2.0 * ly.kv_dim * kv_tokens * weight_bytes
        elif ly.state_elements:
            flops += STATE_FLOPS_PER_ELEMENT * ly.state_elements * tokens
            nbytes += f.rows * ly.state_elements * STATE_BYTES * ((f.past > 0) + f.cached)
        nbytes += ly.dense_params * weight_bytes
        nbytes += experts_touched(ly.experts, ly.top_k, tokens) * ly.expert_params * weight_bytes
        nbytes += 2.0 * tokens * a.hidden * weight_bytes  # residual stream in and out
    return Cost(flops, nbytes, f.ms, tokens)


def total(a: Arch, forwards: Sequence[Forward], **kw: int) -> Cost:
    out = ZERO
    for f in forwards:
        out = out + cost(a, f, **kw)
    return out
