"""Backends turn a ModelSpec + settings into an object with `decide(state, questions)`.

The table holds lazy "module:Class" strings, imported only when a model using the backend is
built, so listing models (and tests) never import torch or vllm.
"""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

from ...record import Scalar
from ..spec import ModelSpec


@runtime_checkable
class Backend(Protocol):
    """`build` returns the model: `decide(state, questions) -> {key: answer}` (see AGENTS.md),
    optionally `close()`."""

    def build(self, spec: ModelSpec, settings: dict[str, Scalar]) -> object: ...


BACKENDS: dict[str, str] = {
    "vllm_shim": "s1.models.backends.vllm_shim:VllmShim",
}


def load_backend(key: str) -> Backend:
    if key not in BACKENDS:
        raise KeyError(f"unknown backend {key!r}; known: {', '.join(sorted(BACKENDS))}")
    module, _, cls = BACKENDS[key].partition(":")
    backend = getattr(importlib.import_module(module), cls)()
    if not isinstance(backend, Backend):
        raise TypeError(f"{BACKENDS[key]} does not implement Backend.build(spec, settings)")
    return backend
