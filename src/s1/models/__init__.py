"""Model registration: every catalog spec becomes a registered factory."""

from __future__ import annotations

from ..record import Scalar
from ..registry import ModelFactory, Settings, register_model
from .backends import load_backend
from .catalog import CATALOG
from .spec import ModelSpec


def resolve_settings(spec: ModelSpec, settings: Settings) -> dict[str, Scalar]:
    """Spec fields and options, under the run's settings (invariants, then params, which win)."""
    return spec.settings() | settings


def spec_params(name: str) -> dict[str, Scalar]:
    """The catalog spec's scalars for a model name ({} for models registered some other way)."""
    return next((s.settings() for s in CATALOG if s.name == name), {})


def make_factory(spec: ModelSpec) -> ModelFactory:
    def build(settings: Settings) -> object:
        if spec.backend is None:
            raise NotImplementedError(f"no backend yet: {spec.name}")
        return load_backend(spec.backend).build(spec, resolve_settings(spec, settings))

    return build


for _spec in CATALOG:
    register_model(_spec.name, make_factory(_spec))
