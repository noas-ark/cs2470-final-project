"""Name -> implementation registries for models and benchmarks.

Both receive `settings`: the run's invariants merged with its params (params win).

Add a model:      add a ModelSpec to s1/models/catalog.py (see AGENTS.md), or for a one-off
                  @model("name")      def build(settings) -> object   (may define close())
Add a benchmark:  @benchmark("name")  def run(model, settings) -> Metrics
Warm one up:      @warmup("name")     def warm(model, settings) -> bool   (optional; runs before GPU sampling;
                                                                          True if it did anything)
Text facts:       note("<bench>_key", "value") inside a benchmark puts a string/scalar into the record's
                  `meta` (e.g. which CUDA kernels ran); metrics stay numbers

Implementations live in s1/models/ and s1/benchmarks/ and are imported by those packages'
__init__.py so registering is just "write the function, add the import".
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol, runtime_checkable

from .record import Metrics, Scalar

type Settings = dict[str, Scalar]


class ModelFactory(Protocol):
    def __call__(self, settings: Settings, /) -> object: ...


class BenchmarkFn(Protocol):
    def __call__(self, model: object, settings: Settings, /) -> Metrics: ...


@runtime_checkable
class Closeable(Protocol):
    def close(self) -> None: ...


class WarmupFn(Protocol):
    def __call__(self, model: object, settings: Settings, /) -> bool: ...


MODELS: dict[str, ModelFactory] = {}
BENCHMARKS: dict[str, BenchmarkFn] = {}
WARMUPS: dict[str, WarmupFn] = {}


def register_model(name: str, fn: ModelFactory) -> ModelFactory:
    if name in MODELS:
        raise ValueError(f"model {name!r} is already registered")
    MODELS[name] = fn
    return fn


def model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    def register(fn: ModelFactory) -> ModelFactory:
        return register_model(name, fn)

    return register


def benchmark(name: str) -> Callable[[BenchmarkFn], BenchmarkFn]:
    def register(fn: BenchmarkFn) -> BenchmarkFn:
        BENCHMARKS[name] = fn
        return fn

    return register


def warmup(name: str) -> Callable[[WarmupFn], WarmupFn]:
    """Untimed work the runner does before a benchmark's GPU sampling and timing start (kernel compilation,
    graph capture). It must leave no answers behind: models clear their caches in `after_warmup()`."""

    def register(fn: WarmupFn) -> WarmupFn:
        WARMUPS[name] = fn
        return fn

    return register


def _lookup[T](table: dict[str, T], kind: str, name: str) -> T:
    from . import benchmarks, models  # noqa: F401, PLC0415 — populate registries

    if name not in table:
        raise KeyError(f"unknown {kind} {name!r}; registered: {', '.join(sorted(table)) or '(none)'}")
    return table[name]


def get_model(name: str) -> ModelFactory:
    return _lookup(MODELS, "model", name)


def get_benchmark(name: str) -> BenchmarkFn:
    return _lookup(BENCHMARKS, "benchmark", name)


def get_warmup(name: str) -> WarmupFn | None:
    get_benchmark(name)  # populates the registries; unknown names raise there
    return WARMUPS.get(name)


def get_str(settings: Settings, key: str, default: str) -> str:
    val = settings.get(key, default)
    if not isinstance(val, str):
        raise TypeError(f"setting {key!r} must be a string, got {val!r}")
    return val


def get_int(settings: Settings, key: str, default: int) -> int:
    val = settings.get(key, default)
    if not isinstance(val, int) or isinstance(val, bool):
        raise TypeError(f"setting {key!r} must be an integer, got {val!r}")
    return val


def get_bool(settings: Settings, key: str, default: bool) -> bool:
    val = settings.get(key, default)
    if not isinstance(val, bool):
        raise TypeError(f"setting {key!r} must be true or false, got {val!r}")
    return val


def get_float(settings: Settings, key: str, default: float) -> float:
    val = settings.get(key, default)
    if not isinstance(val, int | float) or isinstance(val, bool):
        raise TypeError(f"setting {key!r} must be a number, got {val!r}")
    return float(val)


_NOTES: ContextVar[dict[str, Scalar] | None] = ContextVar("s1_notes", default=None)


@contextmanager
def collecting_notes() -> Generator[dict[str, Scalar]]:
    """The runner wraps each benchmark in this; `note()` calls inside land in the yielded dict."""
    notes: dict[str, Scalar] = {}
    token = _NOTES.set(notes)
    try:
        yield notes
    finally:
        _NOTES.reset(token)


def note(key: str, value: Scalar) -> None:
    """Record a non-numeric fact about this run in `meta` (a no-op outside a runner benchmark)."""
    if (notes := _NOTES.get()) is not None:
        notes[key] = value
