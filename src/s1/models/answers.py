"""Assemble System One answers (the shapes in AGENTS.md) from per-option scores or logits.

Pure functions, no heavy imports. Scores are raw logits; `temperature` divides them before
the softmax. Invalid input raises, and `jeval` counts a raised exception as an invalid answer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _check_temperature(temperature: float) -> None:
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError(f"temperature must be positive and finite, got {temperature!r}")


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    _check_temperature(temperature)
    if not scores:
        raise ValueError("need at least one score")
    if not all(math.isfinite(s) for s in scores):
        raise ValueError(f"scores must be finite, got {list(scores)!r}")
    scaled = [s / temperature for s in scores]
    peak = max(scaled)
    exps = [math.exp(s - peak) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def choice_answer(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, object]:
    """Pick the highest-scoring option key; the first wins ties."""
    keys = list(scores)
    probs = softmax([scores[k] for k in keys], temperature)
    best = max(range(len(keys)), key=lambda i: probs[i])
    return {"type": "choice", "choice": keys[best], "probabilities": dict(zip(keys, probs, strict=True))}


def noul_answer(p_true: float) -> dict[str, object]:
    if not (math.isfinite(p_true) and 0.0 <= p_true <= 1.0):
        raise ValueError(f"P(true) must be in [0, 1], got {p_true!r}")
    return {"type": "noul", "noul": p_true}


def noul_from_logits(
    true_logit: float, false_logit: float, temperature: float = 1.0, bias: float = 0.0
) -> dict[str, object]:
    """P(true) from the true/false logits; `bias` is added to the true-minus-false margin."""
    p_true, _ = softmax([true_logit + bias, false_logit], temperature)
    return noul_answer(p_true)


def score_answer(level_scores: Sequence[float], temperature: float = 1.0) -> dict[str, object]:
    """Expected 0-based level over the softmax of per-level scores; probability keys are "0", "1", ..."""
    probs = softmax(level_scores, temperature)
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probs)),
        "probabilities": {str(i): p for i, p in enumerate(probs)},
    }
