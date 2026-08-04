"""Public BarunLM API with lazy model and quantization imports.

Importing a CPU-only dataset, evaluator, or provenance module must not import Torch,
inspect CUDA, or make model code executable as a side effect of package discovery.
The historical top-level API remains available and is loaded only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import BarunConfig
    from .model import BarunLM
    from .quantization import (
        INT8_FORMAT_VERSION,
        Int8CheckpointInfo,
        QuantizationError,
        export_dynamic_int8_checkpoint,
        load_verified_int8_model,
        verify_int8_checkpoint,
    )

_LAZY_EXPORTS = {
    "BarunConfig": (".config", "BarunConfig"),
    "BarunLM": (".model", "BarunLM"),
    "INT8_FORMAT_VERSION": (".quantization", "INT8_FORMAT_VERSION"),
    "Int8CheckpointInfo": (".quantization", "Int8CheckpointInfo"),
    "QuantizationError": (".quantization", "QuantizationError"),
    "export_dynamic_int8_checkpoint": (".quantization", "export_dynamic_int8_checkpoint"),
    "load_verified_int8_model": (".quantization", "load_verified_int8_model"),
    "verify_int8_checkpoint": (".quantization", "verify_int8_checkpoint"),
}

__all__ = [
    "INT8_FORMAT_VERSION",
    "BarunConfig",
    "BarunLM",
    "Int8CheckpointInfo",
    "QuantizationError",
    "export_dynamic_int8_checkpoint",
    "load_verified_int8_model",
    "verify_int8_checkpoint",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
