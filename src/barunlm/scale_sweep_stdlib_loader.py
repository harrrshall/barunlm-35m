"""Load the scale-sweep runner without executing Torch-bearing package imports.

``barunlm.baselines`` eagerly imports ``mobile_matched`` (which imports Torch).
The production isolation gate must run before that side effect. Loading
``mobile_scale_sweep.py`` by file path under a distinct module name keeps the
stdlib-only preflight reachable without changing the frozen matched-baseline
``package_init`` hash.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_MODULE_NAME = "barunlm_scale_sweep_stdlib_preflight"
_RUNNER_PATH = Path(__file__).resolve().parent / "baselines" / "mobile_scale_sweep.py"


def load_mobile_scale_sweep_stdlib() -> ModuleType:
    """Return the scale-sweep module loaded without ``baselines`` package init."""

    existing = sys.modules.get(_MODULE_NAME)
    if existing is not None:
        return existing
    if "torch" in sys.modules:
        raise RuntimeError(
            "torch is already present in sys.modules before the stdlib scale-sweep loader; "
            "the isolation gate cannot prove a pre-Torch boundary"
        )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load scale-sweep runner from {_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    if "torch" in sys.modules:
        raise RuntimeError(
            "loading the stdlib scale-sweep preflight imported torch; "
            "the runner module is no longer stdlib-first"
        )
    if "barunlm.baselines.mobile_matched" in sys.modules:
        raise RuntimeError("stdlib scale-sweep preflight imported mobile_matched")
    if "barunlm.training.data" in sys.modules:
        raise RuntimeError("stdlib scale-sweep preflight imported training.data")
    return module
