"""The analytic cost model: parameter counts against the Hub, and the FLOP/byte terms."""

import math

import pytest

from s1 import flops

# The config.json fields the cost model reads, at the catalog revisions (2026-11-04), and each checkpoint's
# safetensors parameter total from the Hub. Qwen3.5-9B's checkpoint also holds a vision tower and an MTP
# layer the text model never loads; JevK5-9B is the same text architecture saved without them.
QWEN35_9B = {
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vocab_size": 248320,
        "intermediate_size": 12288,
        "full_attention_interval": 4,
        "attn_output_gate": True,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    },
}
QWEN3_8B = {
    "model_type": "qwen3",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "vocab_size": 151936,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "tie_word_embeddings": False,
}
OLMOE = {
    "model_type": "olmoe",
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "vocab_size": 50304,
    "intermediate_size": 1024,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "tie_word_embeddings": False,
}
LFM2_LAYERS = ["conv", "conv", "full_attention"] + ["conv", "conv", "conv", "full_attention"] * 5 + ["conv"]
LFM2 = {
    "model_type": "lfm2_moe",
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "vocab_size": 65536,
    "intermediate_size": 7168,
    "moe_intermediate_size": 1792,
    "num_dense_layers": 2,
    "num_experts": 32,
    "num_experts_per_tok": 4,
    "use_expert_bias": True,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "conv_L_cache": 3,
    "layer_types": LFM2_LAYERS,
}
GRANITE = {
    "model_type": "granitemoehybrid",
    "hidden_size": 1536,
    "num_hidden_layers": 40,
    "vocab_size": 100352,
    "intermediate_size": 512,
    "shared_intermediate_size": 1024,
    "num_local_experts": 64,
    "num_experts_per_tok": 6,
    "num_attention_heads": 12,
    "num_key_value_heads": 4,
    "mamba_n_heads": 48,
    "mamba_d_head": 64,
    "mamba_d_state": 128,
    "mamba_n_groups": 1,
    "mamba_expand": 2,
    "mamba_d_conv": 4,
    "mamba_conv_bias": True,
    "tie_word_embeddings": True,
    "layer_types": (["mamba"] * 9 + ["attention"]) * 4,
}

MODERNBERT_LARGE = {  # Laya (English): ModernBERT-large
    "model_type": "modernbert",
    "hidden_size": 1024,
    "intermediate_size": 2624,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "vocab_size": 50368,
    "global_attn_every_n_layers": 3,
    "local_attention": 128,
    "norm_bias": False,
}
MMBERT_BASE = MODERNBERT_LARGE | {  # Laya (multilingual): mmBERT-base
    "hidden_size": 768,
    "intermediate_size": 1152,
    "num_attention_heads": 12,
    "num_hidden_layers": 22,
    "vocab_size": 256000,
}


@pytest.mark.parametrize(
    ("config", "hub_params"),
    [
        (QWEN35_9B, 8_953_803_264),  # JevK5-9B's checkpoint: the same text model, nothing else
        (QWEN3_8B, 8_190_735_360),
        (OLMOE, 6_919_161_856),
        (LFM2, 8_339_929_856),
        (GRANITE, 6_939_037_248),
        (MODERNBERT_LARGE, 394_781_696),  # laya@7b928d8 model.safetensors, encoder.* tensors
        (MMBERT_BASE, 306_939_648),  # laya@7b928d8 multilingual/model.safetensors, encoder.*
    ],
)
def test_parameter_count_matches_the_hub(config, hub_params):
    assert flops.arch(config).params == hub_params


def test_granite_reads_transformers_layer_names():
    """GraniteMoeHybridConfig.to_dict() (what the backend passes) renames mamba/attention."""
    renamed = {"mamba": "linear_attention", "attention": "full_attention"}
    cfg = GRANITE | {"layer_types": [renamed[t] for t in (["mamba"] * 9 + ["attention"]) * 4]}
    assert flops.arch(cfg) == flops.arch(GRANITE)


def test_lfm2_layout_has_the_configs_layer_mix():
    assert len(LFM2_LAYERS) == 24
    assert LFM2_LAYERS.count("full_attention") == 6
    mixers = [ly.mixer for ly in flops.arch(LFM2).layers]
    assert mixers.count("attention") == 6
    assert mixers.count("short_conv") == 18


def test_active_params_count_top_k_experts_only():
    a = flops.arch(OLMOE)
    assert a.active_params < a.params / 5  # ~1B of 6.9B
    dense = flops.arch(QWEN3_8B)
    assert dense.active_params == dense.params - 2 * 151936 * 4096 - 4096  # minus embedding, head, norm


def test_dense_intensity_tracks_tokens_and_moe_sits_far_lower():
    """Prefill reads every weight once and does 2 FLOPs per weight per token, so a dense model's intensity
    grows ~linearly with the prompt; a MoE reads all its experts but computes with top-k of them."""
    dense, moe = flops.arch(QWEN3_8B), flops.arch(OLMOE)

    def ai(a, n):
        c = flops.cost(a, flops.Forward(rows=1, tokens=n))
        return c.flops / c.bytes

    assert ai(dense, 1024) == pytest.approx(2 * ai(dense, 512), rel=0.1)
    assert 300 < ai(dense, 512) < 600
    assert ai(moe, 512) < ai(dense, 512) / 4


def test_experts_touched():
    assert flops.experts_touched(64, 8, 0) == 0
    assert flops.experts_touched(64, 8, 1) == pytest.approx(8)
    assert flops.experts_touched(64, 8, 1000) == pytest.approx(64)
    assert flops.experts_touched(0, 0, 100) == 0


def test_attention_term_counts_causal_pairs_and_past():
    a = flops.arch(QWEN3_8B)
    base = flops.cost(a, flops.Forward(rows=1, tokens=1))
    one_more = flops.cost(a, flops.Forward(rows=1, tokens=1, past=10))
    per_pair = 4 * 32 * 128 * 36  # QK^T and PV, every layer
    assert one_more.flops - base.flops == pytest.approx(10 * per_pair)


def test_cache_bytes_only_when_cached_and_head_counted_at_kept_positions():
    a = flops.arch(QWEN35_9B)
    plain = flops.cost(a, flops.Forward(rows=1, tokens=100))
    cached = flops.cost(a, flops.Forward(rows=1, tokens=100, cached=True))
    kv = 8 * 2 * (4 * 256) * 100 * 2  # 8 attention layers, K and V, 100 tokens, bf16
    states = 24 * 32 * 128 * 128 * 4  # 24 Gated DeltaNet layers' fp32 states written
    assert cached.bytes - plain.bytes == pytest.approx(kv + states)
    all_logits = flops.cost(a, flops.Forward(rows=1, tokens=100, kept=100))
    assert all_logits.flops - plain.flops == pytest.approx(99 * 2 * 4096 * 248320)


def test_headless_forward_skips_the_lm_head():
    """A pointer-head model (kev) runs the backbone only: no vocab logits, no head weight read."""
    a = flops.arch(QWEN3_8B)
    with_head = flops.cost(a, flops.Forward(rows=1, tokens=100))
    headless = flops.cost(a, flops.Forward(rows=1, tokens=100, kept=0))
    assert with_head.flops - headless.flops == 2 * 4096 * 151936
    assert with_head.bytes - headless.bytes == 2 * 4096 * 151936


def test_encoder_attention_is_bidirectional_and_windowed():
    """ModernBERT: global layers see all n keys per query (n^2 pairs, not the causal n(n+1)/2); sliding
    layers see at most window + 1."""
    a = flops.arch(MODERNBERT_LARGE)
    glob, local = a.layers[0], a.layers[1]
    assert (glob.causal, glob.window, local.window) == (False, 0, 128)
    assert sum(ly.window == 0 for ly in a.layers) == 10  # every third of 28
    n = 512
    attn = flops.cost(a, flops.Forward(rows=1, tokens=n, kept=0)).flops - 2 * n * a.active_params
    per_pair = 4 * 16 * 64
    assert attn == per_pair * (10 * n * n + 18 * n * 129)


def test_fp32_weights_double_the_weight_bytes():
    a = flops.arch(MODERNBERT_LARGE)
    f = flops.Forward(rows=1, tokens=4, kept=0)
    extra = flops.cost(a, f, weight_bytes=4).bytes - flops.cost(a, f).bytes
    assert extra > 2 * a.active_params  # every weight read at 4 bytes instead of 2


def test_fp32_head_doubles_its_bytes():
    a = flops.arch(QWEN35_9B)
    f = flops.Forward(rows=1, tokens=8)
    assert flops.cost(a, f, head_bytes=4).bytes - flops.cost(a, f).bytes == 2 * 4096 * 248320


def test_total_sums_forwards():
    a = flops.arch(OLMOE)
    fs = [flops.Forward(rows=1, tokens=50, ms=2.0), flops.Forward(rows=3, tokens=10, past=50, ms=1.5)]
    t = flops.total(a, fs)
    assert t.ms == 3.5
    assert t.tokens == 80
    assert t.flops == pytest.approx(sum(flops.cost(a, f).flops for f in fs))


def test_unknown_model_type():
    with pytest.raises(KeyError, match="no cost model"):
        flops.arch({"model_type": "llama"})
    assert math.isnan(flops.Forward(rows=1, tokens=1).ms)
