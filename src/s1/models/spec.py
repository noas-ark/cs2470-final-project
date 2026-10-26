"""ModelSpec: the data that identifies a model, separate from the code that serves it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ..record import Scalar

type Group = Literal["E", "D-hybrid", "D-attn", "ED", "MoE", "control", "MoE-Diff", "reference"]


type Status = Literal["planned", "ready"]


@dataclass(frozen=True)
class ModelSpec:
    """One benchmarkable model.

    `name` is `model.variant` (e.g. `kev-4b.hybrid`); a bare name like `openjev` is allowed.
    `status` is "planned" (pins and sizes recorded, no backend yet) or "ready" (`backend` keys into
    `s1.models.backends.BACKENDS`). `revision=None` means unpinned. `code_repo`/`code_revision` pin
    runtime code that is not on the Hub. `size_bytes` is the exact sum of this repo's weight files
    (`*.safetensors`, `*.bin`, `*.pt`; adapters and heads included, GGUF and docs excluded) at
    `revision`, `base_size_bytes` the same for a base model that is needed on top (adapter models).
    `resolved_on` is the ISO date the pins and sizes were read from the Hub. `options` are backend knobs
    (question_mode, mode, temperature, ...); they become settings, so configs can override them.
    """

    name: str
    group: Group
    repo: str
    license: str
    resolved_on: str
    status: Status = "planned"
    backend: str | None = None
    revision: str | None = None
    base_repo: str | None = None
    base_revision: str | None = None
    code_repo: str | None = None
    code_revision: str | None = None
    size_bytes: int | None = None
    base_size_bytes: int | None = None
    tokenizer: str | None = None
    max_state_tokens: int | None = None
    options: Mapping[str, Scalar] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status == "ready" and self.backend is None:
            raise ValueError(f"{self.name}: a ready model needs a backend")
        if self.backend is not None and self.status == "planned":
            raise ValueError(f"{self.name}: planned models have no backend; mark it ready")

    @property
    def weights_gb(self) -> float | None:
        """GB to hold in memory: this repo's weights plus the base model's, from the exact byte counts."""
        if self.size_bytes is None:
            return None
        return (self.size_bytes + (self.base_size_bytes or 0)) / 1e9

    def variant(self, tag: str, **changes: Any) -> ModelSpec:  # noqa: ANN401 — field values vary by field
        """A sibling spec named `<model>.<tag>` differing from this one only in `changes`.

        `options` changes are merged into the existing options, so a controlled pair is an
        explicit diff of the knobs that differ.
        """
        if "name" in changes:
            raise ValueError("variant() sets the name from `tag`")
        if "options" in changes:
            changes["options"] = {**self.options, **changes["options"]}
        return replace(self, name=f"{self.name.split('.', 1)[0]}.{tag}", **changes)

    def settings(self) -> dict[str, Scalar]:
        """Scalar view of the spec, to merge under the run's invariants/params (those win).

        Fields are namespaced `spec_*`; `options` keys are used as-is. None fields are omitted.
        """
        fields: dict[str, Scalar | None] = {
            "spec_group": self.group,
            "spec_repo": self.repo,
            "spec_revision": self.revision,
            "spec_base_repo": self.base_repo,
            "spec_base_revision": self.base_revision,
            "spec_code_repo": self.code_repo,
            "spec_code_revision": self.code_revision,
            "spec_license": self.license,
            "spec_weights_gb": self.weights_gb,
            "spec_tokenizer": self.tokenizer,
            "spec_max_state_tokens": self.max_state_tokens,
        }
        return {k: v for k, v in fields.items() if v is not None} | dict(self.options)
