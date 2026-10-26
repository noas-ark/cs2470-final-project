"""Name -> implementation registries for models and benchmarks.

Both receive `settings`: the run's invariants merged with its params (params win).

Add a model:      add a ModelSpec to s1/models/catalog.py (see AGENTS.md), or for a one-off
                  @model("name")      def build(settings) -> object   (may define close())
Add a benchmark:  @benchmark("name")  def run(model, settings) -> Metrics

Implementations live in s1/models/ and s1/benchmarks/ and are imported by those packages'
__init__.py so registering is just "write the function, add the import".
"""

from __future__ import annotations

from collections.abc import Callable
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


MODELS: dict[str, ModelFactory] = {}
BENCHMARKS: dict[str, BenchmarkFn] = {}


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


def _lookup[T](table: dict[str, T], kind: str, name: str) -> T:
    from . import benchmarks, models  # noqa: F401, PLC0415 — populate registries

    if name not in table:
        raise KeyError(f"unknown {kind} {name!r}; registered: {', '.join(sorted(table)) or '(none)'}")
    return table[name]


def get_model(name: str) -> ModelFactory:
    return _lookup(MODELS, "model", name)


def get_benchmark(name: str) -> BenchmarkFn:
    return _lookup(BENCHMARKS, "benchmark", name)


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


def get_float(settings: Settings, key: str, default: float) -> float:
    val = settings.get(key, default)
    if not isinstance(val, int | float) or isinstance(val, bool):
        raise TypeError(f"setting {key!r} must be a number, got {val!r}")
    return float(val)
