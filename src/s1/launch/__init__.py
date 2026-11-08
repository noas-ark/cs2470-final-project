"""Launchers for `s1 submit`: the [launch] table of a config picks Slurm or Modal (see base.py).

`s1.config.load_launch` parses [launch] into `Resources`; `get_launcher` picks the orchestrator.
The Modal launcher is imported lazily, so the cluster and the tests never need the `modal` package.
"""

from __future__ import annotations

import importlib
from typing import cast

from .base import Launcher, LaunchTarget, Resources
from .slurm import SlurmLauncher

__all__ = ["LaunchTarget", "Launcher", "Resources", "get_launcher"]

MODAL_MODULE = "s1.launch.modal_app"


def get_launcher(target: LaunchTarget) -> Launcher:
    """The `Launcher` for an orchestrator: "slurm" -> SlurmLauncher, "modal" -> modal_app.ModalLauncher."""
    if target == "slurm":
        return SlurmLauncher()
    if target == "modal":
        try:
            module = importlib.import_module(MODAL_MODULE)
        except ModuleNotFoundError as e:
            if e.name != MODAL_MODULE:
                raise
            raise ModuleNotFoundError("Modal launcher not built yet (workstream C)", name=e.name) from e
        return cast("Launcher", module.ModalLauncher())
    raise ValueError(f"unknown orchestrator {target!r} (slurm or modal)")
