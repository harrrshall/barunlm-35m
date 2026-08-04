"""Strict CPU reference decoder for the non-authorizing BarunAction GVS-v1 proposal.

The reference implementation exists to freeze semantics and produce independently
checkable golden vectors.  It never loads weights, chooses a device, authorizes model
or label access, or substitutes for the future optimized remote decoder.  Any optimized
implementation must reproduce this reference's complete candidate vector exactly under
the same bound runtime before it can be considered for a preregistered experiment.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import math
import platform
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import tokenizers
import tokenizers.tokenizers as tokenizers_native
import torch
from torch import Tensor
from torch.nn.modules import module as torch_module

from barunlm import model as barun_model_module
from barunlm.config import BarunConfig
from barunlm.datasets import mobile_actions as mobile_actions_module
from barunlm.datasets.mobile_actions import (
    PROMPT_CONTRACT_VERSION,
    PROMPT_TEMPLATE_SHA256,
    ToolDefinition,
    render_prompt,
)
from barunlm.evaluation import action_ir as action_ir_module
from barunlm.evaluation.action_ir import (
    ActionIRParseError,
    ActionIRValidationError,
    ToolSchema,
    ValueSchema,
    parse_action_ir,
)
from barunlm.model import (
    BarunBlock,
    BarunLM,
    BoundedSwiGLU,
    GroupedAttention,
    PartialRotaryEmbedding,
    ResidualSelector,
    RMSNorm,
)

# Candidate bytes and receipts must not silently change when a process-global
# dependency is replaced after import.  Execute through captured references and
# include both the captured and live origin bindings in the runtime identity.
_DATACLASSES_REPLACE = dataclasses.replace
_HASH_SHA256 = hashlib.sha256
_JSON_DUMPS = json.dumps
_JSON_LOADS = json.loads
_MATH_FSUM = math.fsum
_MATH_ISFINITE = math.isfinite
_TORCH_ARGSORT = torch.argsort
_TORCH_DEVICE = torch.device
_TORCH_INFERENCE_MODE = torch.inference_mode
_TORCH_ISFINITE = torch.isfinite
_TORCH_LOG_SOFTMAX = torch.log_softmax
_TORCH_SET_DEFAULT_DEVICE = torch.set_default_device
_TORCH_SET_DEFAULT_DTYPE = torch.set_default_dtype
_TORCH_SET_FLOAT32_MATMUL_PRECISION = torch.set_float32_matmul_precision
_TORCH_SET_GRAD_ENABLED = torch.set_grad_enabled
_TORCH_SET_NUM_THREADS = torch.set_num_threads
_TORCH_TENSOR = torch.tensor
_TORCH_USE_DETERMINISTIC_ALGORITHMS = torch.use_deterministic_algorithms

GVS_DECODER_SCHEMA_VERSION = "barun-gvs-decoder-reference-v2"
GVS_RUNTIME_SCHEMA_VERSION = "barun-gvs-runtime-identity-v2"
GVS_TRACE_SCHEMA_VERSION = "barun-gvs-candidate-trace-v3"
GVS_ANALYSIS_SCHEMA_VERSION = "barun-gvs-candidate-analysis-v3"
GVS_SELECTION_SCHEMA_VERSION = "barun-gvs-likelihood-selection-v2"
GVS_TRACE_ARTIFACT_SCHEMA_VERSION = "barun-gvs-candidate-trace-artifact-v2"
GVS_ANALYSIS_ARTIFACT_SCHEMA_VERSION = "barun-gvs-analysis-artifact-v2"
GVS_SELECTION_ARTIFACT_SCHEMA_VERSION = "barun-gvs-selection-artifact-v1"

PROPOSED_MAX_NEW_TOKENS = 192
FROZEN_CANDIDATE_COUNT = 8
FROZEN_BEAM_WIDTH = 7

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}")
_MAX_TRACE_BYTES = 2 * 1024 * 1024
_MAX_ANALYSIS_BYTES = 2 * 1024 * 1024
_MAX_SELECTION_BYTES = 64 * 1024
_MAX_PROMPT_UTF8_BYTES = 64 * 1024
_MAX_RAW_OUTPUT_UTF8_BYTES = 16 * 1024
_MAX_SCHEMA_CANONICAL_UTF8_BYTES = 32 * 1024
_MAX_ACTION_CANONICAL_UTF8_BYTES = 32 * 1024
_MAX_JSON_INTEGER_DIGITS = 128
_MAX_JSON_FLOAT_CHARACTERS = 256
_FAILURE_STAGES = frozenset({"native_generate", "beam_search", "decode", "likelihood"})
_ANALYSIS_ERROR_STAGES = frozenset({"generation", "parse", "schema"})
_ADAPTER_MARKERS = (
    "_disable_adapters",
    "active_adapter",
    "active_adapters",
    "adapter",
    "adapter_names",
    "disable_adapter",
    "disable_adapters",
    "ia3",
    "lora",
    "merged_adapters",
    "peft",
    "prefix_encoder",
    "prompt_encoder",
)
_MAX_BEHAVIOR_STATE_DEPTH = 24
_MAX_BEHAVIOR_STATE_NODES = 16_384
_REFERENCE_GENERATION_SETTINGS = {
    "attention_mask": "omitted_for_single_unpadded_prompt",
    "beam_cache": False,
    "beam_width": FROZEN_BEAM_WIDTH,
    "do_sample": False,
    "eos_token_id": "trace.eos_token_id",
    "max_new_tokens": "trace.contract.max_new_tokens",
    "native_batch_size": 1,
    "native_temperature": 0,
    "pad_token_id": "trace.pad_token_id",
    "repair": False,
}


class GVSDecoderError(ValueError):
    """The reference decoder contract or its evidence failed closed."""


class IncompleteNativeGenerationError(GVSDecoderError):
    """Native rank zero stopped without EOS before the exact token limit."""


class NativeGenerationAfterEOSError(GVSDecoderError):
    """Native rank zero returned one or more tokens after its first EOS."""


_RETAINABLE_EXCEPTIONS = (
    GVSDecoderError,
    RuntimeError,
    TypeError,
    ValueError,
    AttributeError,
    IndexError,
    KeyError,
)


class TokenizerLike(Protocol):
    def encode(self, sequence: str, add_special_tokens: bool = False) -> Any: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str: ...

    def token_to_id(self, token: str) -> int | None: ...

    def to_str(self) -> str: ...


def _canonical_json(value: object) -> str:
    try:
        return _JSON_DUMPS(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise GVSDecoderError("decoder evidence is not strict JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return _HASH_SHA256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _source_sha256(value: object) -> str:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as error:
        raise GVSDecoderError("runtime object has no inspectable source") from error
    return _sha256_text(source)


def _file_sha256(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as error:
        raise GVSDecoderError(f"could not hash required source file: {path.name}") from error


def _strict_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise GVSDecoderError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise GVSDecoderError(f"{label} must be a bounded portable identifier")
    return value


def _strict_string(value: object, *, label: str, maximum: int = 4096) -> str:
    if type(value) is not str:
        raise GVSDecoderError(f"{label} must be a bounded UTF-8 string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise GVSDecoderError(f"{label} is not valid UTF-8") from error
    if len(encoded) > maximum:
        raise GVSDecoderError(f"{label} must be a bounded UTF-8 string")
    return value


def _strict_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise GVSDecoderError(f"{label} must be a positive integer")
    return value


def _strict_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise GVSDecoderError(f"{label} must be a non-negative integer")
    return value


def _strict_token_tuple(
    value: object,
    *,
    label: str,
    vocab_size: int,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if type(value) is not tuple or (not value and not allow_empty):
        qualifier = "a tuple" if allow_empty else "a nonempty tuple"
        raise GVSDecoderError(f"{label} must be {qualifier}")
    for token_id in value:
        if type(token_id) is not int or not 0 <= token_id < vocab_size:
            raise GVSDecoderError(f"{label} contains an invalid token ID")
    return value


def _exact_keys(value: object, expected: frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GVSDecoderError(f"{label} must be an exact JSON object")
    actual = set(value)
    if actual != expected:
        raise GVSDecoderError(
            f"{label} fields changed; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    return value


def _detach_json(value: object, *, label: str) -> object:
    """Detach a runtime record into exact built-in JSON values."""

    if value is None or type(value) in {bool, int, float, str}:
        if type(value) is float and not _MATH_ISFINITE(value):
            raise GVSDecoderError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Enum):
        return _detach_json(value.value, label=label)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _detach_json(dataclasses.asdict(value), label=label)
    if hasattr(value, "__dict__") and type(vars(value)) is dict:
        return _detach_json(vars(value), label=label)
    if isinstance(value, Mapping):
        detached: dict[str, object] = {}
        for key, child in value.items():
            if type(key) is not str or key in detached:
                raise GVSDecoderError(f"{label} has an invalid or duplicate key")
            detached[key] = _detach_json(child, label=f"{label}.{key}")
        return detached
    if isinstance(value, tuple | list):
        return [_detach_json(child, label=f"{label}[]") for child in value]
    raise GVSDecoderError(f"{label} contains unsupported type {type(value).__name__}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GVSDecoderError(f"serialized evidence contains duplicate key {key!r}")
        result[key] = value
    return result


def _bounded_json_integer(value: str) -> int:
    digits = value.removeprefix("-")
    if not digits or len(digits) > _MAX_JSON_INTEGER_DIGITS:
        raise GVSDecoderError("serialized evidence contains an oversized integer")
    return int(value)


def _bounded_json_float(value: str) -> float:
    if len(value) > _MAX_JSON_FLOAT_CHARACTERS:
        raise GVSDecoderError("serialized evidence contains an oversized float")
    parsed = float(value)
    if not _MATH_ISFINITE(parsed):
        raise GVSDecoderError("serialized evidence contains a non-finite float")
    return parsed


def _decode_bounded_json(payload: str | bytes, *, maximum: int, label: str) -> object:
    if type(payload) is str:
        try:
            raw = payload.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise GVSDecoderError(f"{label} is not valid UTF-8") from error
    elif type(payload) is bytes:
        raw = payload
    else:
        raise GVSDecoderError(f"{label} must be serialized text or bytes")
    if not raw or len(raw) > maximum:
        raise GVSDecoderError(f"{label} exceeds its strict serialized size bound")
    try:
        text = raw.decode("utf-8", errors="strict")
        return _JSON_LOADS(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_float=_bounded_json_float,
            parse_int=_bounded_json_integer,
            parse_constant=lambda value: (_ for _ in ()).throw(
                GVSDecoderError(f"forbidden JSON constant {value!r}")
            ),
        )
    except GVSDecoderError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise GVSDecoderError(f"{label} is not strict JSON") from error


def _behavior_state_record(value: object, *, label: str) -> object:
    """Encode bounded behavior-bearing state without using address-bearing reprs."""

    seen: set[int] = set()
    nodes = 0

    def visit(current: object, *, path: str, depth: int) -> object:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_BEHAVIOR_STATE_NODES:
            raise GVSDecoderError(f"{label} exceeds the behavior-state node bound")
        if depth > _MAX_BEHAVIOR_STATE_DEPTH:
            raise GVSDecoderError(f"{label} exceeds the behavior-state depth bound")
        if current is None or type(current) in {bool, int, str}:
            if type(current) is str:
                _strict_string(current, label=path, maximum=1_048_576)
            return {"kind": type(current).__name__, "value": current}
        if type(current) is float:
            if not _MATH_ISFINITE(current):
                raise GVSDecoderError(f"{path} contains a non-finite float")
            return {"kind": "float", "value": current}
        if isinstance(current, Enum):
            return {
                "kind": "enum",
                "type": f"{type(current).__module__}.{type(current).__qualname__}",
                "value": visit(current.value, path=f"{path}.value", depth=depth + 1),
            }
        if isinstance(current, Path):
            return {"kind": "path", "value": str(current)}
        if isinstance(current, torch.dtype | torch.device):
            return {"kind": type(current).__name__, "value": str(current)}
        if isinstance(current, Tensor):
            if current.device.type != "cpu" or current.layout is not torch.strided:
                raise GVSDecoderError(f"{path} tensor state must be dense CPU storage")
            detached = current.detach().contiguous()
            return {
                "dtype": str(detached.dtype),
                "kind": "tensor",
                "sha256": _sha256_bytes(detached.view(torch.uint8).numpy().tobytes(order="C")),
                "shape": list(detached.shape),
            }
        if callable(current):
            return {"kind": "callable", "sha256": _callable_sha256(current, label=path)}

        identity = id(current)
        if identity in seen:
            raise GVSDecoderError(f"{path} contains cyclic or aliased mutable state")
        seen.add(identity)
        try:
            if dataclasses.is_dataclass(current) and not isinstance(current, type):
                return {
                    "fields": [
                        {
                            "name": field.name,
                            "value": visit(
                                getattr(current, field.name),
                                path=f"{path}.{field.name}",
                                depth=depth + 1,
                            ),
                        }
                        for field in dataclasses.fields(current)
                    ],
                    "kind": "dataclass",
                    "type": f"{type(current).__module__}.{type(current).__qualname__}",
                }
            if isinstance(current, Mapping):
                entries = [
                    {
                        "key": visit(key, path=f"{path}.key", depth=depth + 1),
                        "value": visit(child, path=f"{path}.value", depth=depth + 1),
                    }
                    for key, child in current.items()
                ]
                entries.sort(key=_canonical_json)
                return {"entries": entries, "kind": "mapping"}
            if isinstance(current, tuple | list):
                return {
                    "items": [
                        visit(child, path=f"{path}[{index}]", depth=depth + 1)
                        for index, child in enumerate(current)
                    ],
                    "kind": type(current).__name__,
                }
            if isinstance(current, set | frozenset):
                items = [visit(child, path=f"{path}[]", depth=depth + 1) for child in current]
                items.sort(key=_canonical_json)
                return {"items": items, "kind": type(current).__name__}
            try:
                attributes = vars(current)
            except TypeError as error:
                raise GVSDecoderError(
                    f"{path} contains unsupported behavior state {type(current).__name__}"
                ) from error
            if type(attributes) is not dict:
                raise GVSDecoderError(f"{path} has a non-exact attribute dictionary")
            return {
                "attributes": [
                    {
                        "name": name,
                        "value": visit(child, path=f"{path}.{name}", depth=depth + 1),
                    }
                    for name, child in sorted(attributes.items())
                    if type(name) is str
                ],
                "kind": "object",
                "type": f"{type(current).__module__}.{type(current).__qualname__}",
            }
        finally:
            seen.remove(identity)

    return visit(value, path=label, depth=0)


def _callable_sha256(value: object, *, label: str) -> str:
    """Bind the live Python code/wrapper chain or exact builtin identity."""

    records: list[dict[str, object]] = []
    current = value.__func__ if inspect.ismethod(value) else value
    visited: set[int] = set()
    for _ in range(16):
        if not callable(current):
            raise GVSDecoderError(f"{label} must be callable")
        identity = id(current)
        if identity in visited:
            raise GVSDecoderError(f"{label} has a cyclic wrapper chain")
        visited.add(identity)
        module = getattr(current, "__module__", type(current).__module__)
        qualname = getattr(current, "__qualname__", type(current).__qualname__)
        record: dict[str, object] = {
            "module": _strict_string(str(module), label=f"{label}.module", maximum=1024),
            "qualname": _strict_string(str(qualname), label=f"{label}.qualname", maximum=2048),
            "type": f"{type(current).__module__}.{type(current).__qualname__}",
        }
        code = getattr(current, "__code__", None)
        if code is not None:
            record["code_sha256"] = _python_code_sha256(code)
            defaults = getattr(current, "__defaults__", None)
            kwdefaults = getattr(current, "__kwdefaults__", None)
            record["defaults_sha256"] = _sha256_text(
                _canonical_json(
                    _behavior_state_record(
                        {"defaults": defaults, "kwdefaults": kwdefaults},
                        label=f"{label}.defaults",
                    )
                )
            )
            closure_rows: list[object] = []
            for index, cell in enumerate(getattr(current, "__closure__", None) or ()):
                try:
                    child = cell.cell_contents
                except ValueError:
                    closure_rows.append({"kind": "empty_cell"})
                    continue
                if callable(child):
                    child_target = child.__func__ if inspect.ismethod(child) else child
                    child_code = getattr(child_target, "__code__", None)
                    closure_rows.append(
                        {
                            "code_sha256": (
                                None if child_code is None else _python_code_sha256(child_code)
                            ),
                            "module": str(
                                getattr(child_target, "__module__", type(child_target).__module__)
                            ),
                            "qualname": str(
                                getattr(
                                    child_target,
                                    "__qualname__",
                                    type(child_target).__qualname__,
                                )
                            ),
                        }
                    )
                else:
                    closure_rows.append(
                        _behavior_state_record(child, label=f"{label}.closure[{index}]")
                    )
            record["closure_sha256"] = _sha256_text(_canonical_json(closure_rows))
        else:
            record["code_sha256"] = None
            record["defaults_sha256"] = None
            record["closure_sha256"] = None
        try:
            record["source_sha256"] = _sha256_text(inspect.getsource(current))
        except (OSError, TypeError):
            record["source_sha256"] = None
        records.append(record)
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is None:
            break
        current = wrapped.__func__ if inspect.ismethod(wrapped) else wrapped
    else:  # pragma: no cover - defensive bound
        raise GVSDecoderError(f"{label} exceeds the callable-wrapper bound")
    return _sha256_text(_canonical_json(records))


def _python_code_sha256(code: object) -> str:
    """Hash stable exposed code fields, excluding CPython's mutable quickening cache."""

    if not inspect.iscode(code):
        raise GVSDecoderError("callable code identity requires a Python code object")

    def constant_record(value: object) -> object:
        if value is None or type(value) in {bool, int, str}:
            return {"kind": type(value).__name__, "value": value}
        if type(value) is float:
            return {"kind": "float", "value": value.hex()}
        if type(value) is complex:
            return {
                "imag": value.imag.hex(),
                "kind": "complex",
                "real": value.real.hex(),
            }
        if type(value) is bytes:
            return {"kind": "bytes", "value": value.hex()}
        if inspect.iscode(value):
            return {"kind": "code", "value": code_record(value)}
        if type(value) is tuple:
            return {"items": [constant_record(item) for item in value], "kind": "tuple"}
        if type(value) is frozenset:
            items = [constant_record(item) for item in value]
            items.sort(key=_canonical_json)
            return {"items": items, "kind": "frozenset"}
        if value is Ellipsis:
            return {"kind": "ellipsis"}
        if value is NotImplemented:
            return {"kind": "not_implemented"}
        raise GVSDecoderError(f"callable code contains unsupported constant {type(value).__name__}")

    def code_record(value: Any) -> dict[str, object]:
        return {
            "argcount": value.co_argcount,
            "cellvars": list(value.co_cellvars),
            "code": value.co_code.hex(),
            "consts": [constant_record(item) for item in value.co_consts],
            "exceptiontable": getattr(value, "co_exceptiontable", b"").hex(),
            "filename": value.co_filename,
            "firstlineno": value.co_firstlineno,
            "flags": value.co_flags,
            "freevars": list(value.co_freevars),
            "kwonlyargcount": value.co_kwonlyargcount,
            "linetable": getattr(value, "co_linetable", value.co_lnotab).hex(),
            "name": value.co_name,
            "names": list(value.co_names),
            "nlocals": value.co_nlocals,
            "posonlyargcount": value.co_posonlyargcount,
            "qualname": getattr(value, "co_qualname", value.co_name),
            "stacksize": value.co_stacksize,
            "varnames": list(value.co_varnames),
        }

    return _sha256_text(_canonical_json(code_record(code)))


def _loaded_python_module_sha256(
    *,
    path: Path,
    namespace: Mapping[str, object],
    label: str,
    extra_callables: Mapping[str, object] | None = None,
) -> str:
    """Bind source bytes to the live functions/classes that those bytes name."""

    try:
        source_bytes = path.read_bytes()
        source = source_bytes.decode("utf-8", errors="strict")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise GVSDecoderError(f"{label} source is not a stable Python module") from error
    rows: list[dict[str, object]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            value = namespace.get(node.name)
            if not callable(value):
                raise GVSDecoderError(f"{label}.{node.name} is not the loaded source callable")
            rows.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "sha256": _callable_sha256(value, label=f"{label}.{node.name}"),
                }
            )
        elif isinstance(node, ast.ClassDef):
            cls = namespace.get(node.name)
            if not inspect.isclass(cls):
                raise GVSDecoderError(f"{label}.{node.name} is not the loaded source class")
            methods: list[dict[str, str]] = []
            for child in node.body:
                if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                raw = vars(cls).get(child.name)
                if isinstance(raw, property):
                    value = raw.fget
                elif isinstance(raw, staticmethod | classmethod):
                    value = raw.__func__
                else:
                    value = getattr(cls, child.name, None)
                if not callable(value):
                    raise GVSDecoderError(
                        f"{label}.{node.name}.{child.name} is not the loaded source callable"
                    )
                methods.append(
                    {
                        "name": child.name,
                        "sha256": _callable_sha256(
                            value,
                            label=f"{label}.{node.name}.{child.name}",
                        ),
                    }
                )
            if issubclass(cls, torch.nn.Module):
                methods.append(
                    {
                        "name": "__call__",
                        "sha256": _callable_sha256(
                            inspect.getattr_static(cls, "__call__"),
                            label=f"{label}.{node.name}.__call__",
                        ),
                    }
                )
            methods.sort(key=lambda row: row["name"])
            rows.append(
                {
                    "kind": "class",
                    "methods": methods,
                    "name": node.name,
                    "type": f"{cls.__module__}.{cls.__qualname__}",
                }
            )
    extras: list[dict[str, object]] = []
    for name, value in sorted((extra_callables or {}).items()):
        if value is None or type(value) in {bool, int, float, str}:
            evidence: object = _behavior_state_record(value, label=f"{label}.{name}")
        elif callable(value):
            evidence = {
                "kind": "callable",
                "sha256": _callable_sha256(value, label=f"{label}.{name}"),
            }
        else:
            raise GVSDecoderError(
                f"{label}.{name} has unsupported loaded dependency {type(value).__name__}"
            )
        extras.append({"evidence": evidence, "name": name})
    return _sha256_text(
        _canonical_json(
            {
                "extras": extras,
                "rows": rows,
                "source_sha256": _sha256_bytes(source_bytes),
            }
        )
    )


def _tensor_algorithm_dependency_bindings() -> dict[str, object]:
    """Bind Tensor methods/operators reached by decoder and BarunLM inference.

    ``torch.__version__`` does not detect a live process monkeypatch.  Hashing the
    exact descriptors used by the reference closes that gap without trying to
    serialize Torch's large, mutable module namespace.
    """

    names = (
        "__add__",
        "__and__",
        "__eq__",
        "__getitem__",
        "__invert__",
        "__le__",
        "__lt__",
        "__mul__",
        "__neg__",
        "__or__",
        "__pow__",
        "__sub__",
        "__truediv__",
        "abs",
        "all",
        "any",
        "argmax",
        "chunk",
        "clamp",
        "clamp_min",
        "clone",
        "contiguous",
        "cos",
        "detach",
        "eq",
        "expand",
        "float",
        "gather",
        "item",
        "masked_fill",
        "mean",
        "numel",
        "repeat_interleave",
        "reshape",
        "scatter",
        "sin",
        "softmax",
        "square",
        "squeeze",
        "storage_offset",
        "stride",
        "sum",
        "to",
        "tolist",
        "transpose",
        "untyped_storage",
        "view",
    )
    bindings: dict[str, object] = {}
    for name in names:
        value = getattr(Tensor, name, None)
        if not callable(value):
            raise GVSDecoderError(f"required torch.Tensor dependency {name!r} is not callable")
        bindings[f"torch.Tensor.{name}"] = value
    return bindings


def _decoder_algorithm_dependency_bindings() -> dict[str, object]:
    """Return live and captured dependencies that can change candidate evidence."""

    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    return {
        "dataclasses.replace.captured": _DATACLASSES_REPLACE,
        "dataclasses.replace.origin": dataclasses.replace,
        "hashlib.sha256.captured": _HASH_SHA256,
        "hashlib.sha256.origin": hashlib.sha256,
        "json.dumps.captured": _JSON_DUMPS,
        "json.dumps.origin": json.dumps,
        "json.loads.captured": _JSON_LOADS,
        "json.loads.origin": json.loads,
        "math.fsum.captured": _MATH_FSUM,
        "math.fsum.origin": math.fsum,
        "math.isfinite.captured": _MATH_ISFINITE,
        "math.isfinite.origin": math.isfinite,
        "torch.Tensor.gather": Tensor.gather,
        "torch.Tensor.to": Tensor.to,
        "torch.Tensor.tolist": Tensor.tolist,
        "torch.are_deterministic_algorithms_enabled": (torch.are_deterministic_algorithms_enabled),
        "torch.argsort.captured": _TORCH_ARGSORT,
        "torch.argsort.origin": torch.argsort,
        "torch.device.captured": _TORCH_DEVICE,
        "torch.device.origin": torch.device,
        "torch.get_autocast_dtype": torch.get_autocast_dtype,
        "torch.get_default_device": torch.get_default_device,
        "torch.get_default_dtype": torch.get_default_dtype,
        "torch.get_float32_matmul_precision": torch.get_float32_matmul_precision,
        "torch.get_num_interop_threads": torch.get_num_interop_threads,
        "torch.get_num_threads": torch.get_num_threads,
        "torch.inference_mode.captured": _TORCH_INFERENCE_MODE,
        "torch.inference_mode.origin": torch.inference_mode,
        "torch.is_autocast_enabled": torch.is_autocast_enabled,
        "torch.is_grad_enabled": torch.is_grad_enabled,
        "torch.is_inference_mode_enabled": torch.is_inference_mode_enabled,
        "torch.is_deterministic_algorithms_warn_only_enabled": (
            torch.is_deterministic_algorithms_warn_only_enabled
        ),
        "torch.isfinite.captured": _TORCH_ISFINITE,
        "torch.isfinite.origin": torch.isfinite,
        "torch.log_softmax.captured": _TORCH_LOG_SOFTMAX,
        "torch.log_softmax.origin": torch.log_softmax,
        "torch.set_default_device.captured": _TORCH_SET_DEFAULT_DEVICE,
        "torch.set_default_device.origin": torch.set_default_device,
        "torch.set_default_dtype.captured": _TORCH_SET_DEFAULT_DTYPE,
        "torch.set_default_dtype.origin": torch.set_default_dtype,
        "torch.set_float32_matmul_precision.captured": _TORCH_SET_FLOAT32_MATMUL_PRECISION,
        "torch.set_float32_matmul_precision.origin": torch.set_float32_matmul_precision,
        "torch.set_grad_enabled.captured": _TORCH_SET_GRAD_ENABLED,
        "torch.set_grad_enabled.origin": torch.set_grad_enabled,
        "torch.set_num_threads.captured": _TORCH_SET_NUM_THREADS,
        "torch.set_num_threads.origin": torch.set_num_threads,
        "torch.nn.Module.buffers": torch.nn.Module.buffers,
        "torch.nn.Module.eval": torch.nn.Module.eval,
        "torch.nn.Module.named_modules": torch.nn.Module.named_modules,
        "torch.nn.Module.parameters": torch.nn.Module.parameters,
        "torch.nn.Module.state_dict": torch.nn.Module.state_dict,
        "torch.nn.Module.train": torch.nn.Module.train,
        "torch.overrides._get_current_function_mode_stack": (_get_current_function_mode_stack),
        "torch.tensor.captured": _TORCH_TENSOR,
        "torch.tensor.origin": torch.tensor,
        "torch.use_deterministic_algorithms.captured": _TORCH_USE_DETERMINISTIC_ALGORITHMS,
        "torch.use_deterministic_algorithms.origin": torch.use_deterministic_algorithms,
        "torch.utils._python_dispatch._get_current_dispatch_mode_stack": (
            _get_current_dispatch_mode_stack
        ),
        **_tensor_algorithm_dependency_bindings(),
    }


def _decoder_callable_sha256() -> str:
    return _loaded_python_module_sha256(
        path=Path(__file__),
        namespace=globals(),
        label="gvs_decoder",
        extra_callables={
            "ActionIRParseError_import": ActionIRParseError,
            "ActionIRValidationError_import": ActionIRValidationError,
            "parse_action_ir_import": parse_action_ir,
            "render_prompt_import": render_prompt,
            "render_prompt_origin": mobile_actions_module.render_prompt,
            "renderer_PROMPT_CONTRACT_VERSION": mobile_actions_module.PROMPT_CONTRACT_VERSION,
            "renderer_PROMPT_TEMPLATE": mobile_actions_module.PROMPT_TEMPLATE,
            "renderer_PROMPT_TEMPLATE_SHA256": mobile_actions_module.PROMPT_TEMPLATE_SHA256,
            "renderer_render_tool": mobile_actions_module._render_tool,
            **_decoder_algorithm_dependency_bindings(),
        },
    )


def _model_dependency_sha256() -> str:
    if (
        barun_model_module.torch is not torch
        or barun_model_module.F is not torch.nn.functional
        or barun_model_module.nn is not torch.nn
        or barun_model_module.Tensor is not Tensor
    ):
        raise GVSDecoderError("BarunLM Torch/functional global aliases changed")
    path = Path(barun_model_module.__file__ or "")
    if not path.is_file():
        raise GVSDecoderError("BarunLM model module source is not inspectable")
    return _loaded_python_module_sha256(
        path=path,
        namespace=vars(barun_model_module),
        label="barun_model",
        extra_callables={
            "F.cross_entropy": torch.nn.functional.cross_entropy,
            "F.dropout": torch.nn.functional.dropout,
            "F.embedding": torch.nn.functional.embedding,
            "F.linear": torch.nn.functional.linear,
            "F.scaled_dot_product_attention": (torch.nn.functional.scaled_dot_product_attention),
            "F.silu": torch.nn.functional.silu,
            "compiled_flex_attention": barun_model_module.compiled_flex_attention,
            "create_block_mask": barun_model_module.create_block_mask,
            "flash_attn_func": barun_model_module.flash_attn_func,
            "flex_attention": barun_model_module.flex_attention,
            "torch.all": torch.all,
            "torch.any": torch.any,
            "torch.arange": torch.arange,
            "torch.cat": torch.cat,
            "torch.compile": torch.compile,
            "torch.compiler.is_compiling": torch.compiler.is_compiling,
            "torch.device": torch.device,
            "torch.full_like": torch.full_like,
            "torch.inference_mode": torch.inference_mode,
            "torch.is_floating_point": torch.is_floating_point,
            "torch.isfinite": torch.isfinite,
            "torch.multinomial": torch.multinomial,
            "torch.no_grad": torch.no_grad,
            "torch.ones": torch.ones,
            "torch.ones_like": torch.ones_like,
            "torch.outer": torch.outer,
            "torch.rsqrt": torch.rsqrt,
            "torch.sigmoid": torch.sigmoid,
            "torch.stack": torch.stack,
            "torch.where": torch.where,
            "torch.zeros": torch.zeros,
            "torch.zeros_like": torch.zeros_like,
            **_tensor_algorithm_dependency_bindings(),
        },
    )


def _clear_barun_model_execution_caches(model: BarunLM) -> None:
    for name in ("_local_bidirectional_mask", "_local_block_mask", "_local_causal_mask"):
        function = getattr(barun_model_module, name, None)
        cache_clear = getattr(function, "cache_clear", None)
        if not callable(cache_clear):
            raise GVSDecoderError(f"BarunLM execution cache {name} is not clearable")
        cache_clear()
    selectors = getattr(model, "selectors", None)
    if type(selectors) is not torch.nn.ModuleList or any(
        type(selector) is not ResidualSelector for selector in selectors
    ):
        raise GVSDecoderError("BarunLM selector state is not canonical")
    for selector in selectors:
        selector.last_mean_weights = None


def _evaluator_runtime_sha256() -> str:
    path = Path(action_ir_module.__file__ or "")
    if not path.is_file():
        raise GVSDecoderError("Action IR evaluator source is not inspectable")
    return _loaded_python_module_sha256(
        path=path,
        namespace=vars(action_ir_module),
        label="action_ir",
        extra_callables={"parse_action_ir_import": parse_action_ir},
    )


def _torch_execution_state_record() -> dict[str, object]:
    try:
        cpu_autocast_enabled = bool(torch.is_autocast_enabled("cpu"))
        cpu_autocast_dtype = str(torch.get_autocast_dtype("cpu"))
    except (RuntimeError, TypeError) as error:  # pragma: no cover - old unsupported Torch
        raise GVSDecoderError("Torch cannot attest CPU autocast state") from error
    try:
        from torch.overrides import _get_current_function_mode_stack
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

        function_modes = tuple(_get_current_function_mode_stack())
        dispatch_modes = tuple(_get_current_dispatch_mode_stack())
    except (AttributeError, RuntimeError, TypeError) as error:  # pragma: no cover - version guard
        raise GVSDecoderError("Torch cannot attest Python execution-mode stacks") from error
    return {
        "cpu_autocast_dtype": cpu_autocast_dtype,
        "cpu_autocast_enabled": cpu_autocast_enabled,
        "default_device": str(torch.get_default_device()),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only_enabled": (torch.is_deterministic_algorithms_warn_only_enabled()),
        "dispatch_modes": [
            f"{type(mode).__module__}.{type(mode).__qualname__}" for mode in dispatch_modes
        ],
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "function_modes": [
            f"{type(mode).__module__}.{type(mode).__qualname__}" for mode in function_modes
        ],
        "grad_enabled": torch.is_grad_enabled(),
        "inference_mode_enabled": torch.is_inference_mode_enabled(),
        "mkldnn_deterministic": bool(torch.backends.mkldnn.deterministic),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "num_interop_threads": torch.get_num_interop_threads(),
        "num_threads": torch.get_num_threads(),
    }


def _torch_execution_state_sha256() -> str:
    return _sha256_text(_canonical_json(_torch_execution_state_record()))


def _require_reference_torch_execution_state() -> dict[str, object]:
    state = _torch_execution_state_record()
    if state["default_device"] != "cpu":
        raise GVSDecoderError("the CPU reference requires Torch default device cpu")
    if state["cpu_autocast_enabled"] is not False:
        raise GVSDecoderError("the CPU reference forbids an ambient autocast context")
    if state["inference_mode_enabled"] is not False or state["grad_enabled"] is not True:
        raise GVSDecoderError("the CPU reference requires a clean outer grad/inference mode")
    if state["dispatch_modes"] or state["function_modes"]:
        raise GVSDecoderError("the CPU reference forbids ambient Torch execution modes")
    return state


@contextmanager
def _guard_torch_execution_state() -> Iterable[None]:
    """Restore mutable globals and reject any model-caused Torch-state drift."""

    expected = _torch_execution_state_record()
    expected_default_dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        actual = _torch_execution_state_record()
        if (
            actual["default_device"] != expected["default_device"]
            or actual["function_modes"] != expected["function_modes"]
        ):
            # All public entry points reject an ambient DeviceContext, so the
            # only admissible outer state is Torch's native CPU default.
            _TORCH_SET_DEFAULT_DEVICE(None)
        if torch.get_default_dtype() is not expected_default_dtype:
            _TORCH_SET_DEFAULT_DTYPE(expected_default_dtype)
        if torch.is_grad_enabled() is not expected["grad_enabled"]:
            _TORCH_SET_GRAD_ENABLED(bool(expected["grad_enabled"]))
        if torch.get_float32_matmul_precision() != expected["float32_matmul_precision"]:
            _TORCH_SET_FLOAT32_MATMUL_PRECISION(str(expected["float32_matmul_precision"]))
        if (
            torch.are_deterministic_algorithms_enabled()
            is not expected["deterministic_algorithms_enabled"]
            or torch.is_deterministic_algorithms_warn_only_enabled()
            is not expected["deterministic_warn_only_enabled"]
        ):
            _TORCH_USE_DETERMINISTIC_ALGORITHMS(
                bool(expected["deterministic_algorithms_enabled"]),
                warn_only=bool(expected["deterministic_warn_only_enabled"]),
            )
        if bool(torch.backends.mkldnn.enabled) is not expected["mkldnn_enabled"]:
            torch.backends.mkldnn.enabled = bool(expected["mkldnn_enabled"])
        if bool(torch.backends.mkldnn.deterministic) is not expected["mkldnn_deterministic"]:
            torch.backends.mkldnn.deterministic = bool(expected["mkldnn_deterministic"])
        if torch.get_num_threads() != expected["num_threads"]:
            _TORCH_SET_NUM_THREADS(int(expected["num_threads"]))
        restored = _torch_execution_state_record()
        if actual != expected:
            if restored != expected:
                raise GVSDecoderError(
                    "model mutated Torch execution state and it could not be restored exactly"
                )
            raise GVSDecoderError("model mutated process-global Torch execution state")


def _tensor_execution_record(
    tensor: Tensor,
    *,
    label: str,
    alias_of: str,
    kind: str,
    persistent: bool,
) -> dict[str, object]:
    if tensor.device.type != "cpu" or tensor.layout is not torch.strided:
        raise GVSDecoderError(f"{label} must be a dense CPU tensor")
    detached = tensor.detach().contiguous()
    return {
        "alias_of": alias_of,
        "bytes_sha256": _sha256_bytes(detached.view(torch.uint8).numpy().tobytes(order="C")),
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "kind": kind,
        "persistent": persistent,
        "requires_grad": bool(tensor.requires_grad),
        "shape": list(tensor.shape),
        "storage_offset": tensor.storage_offset(),
        "stride": list(tensor.stride()),
    }


def _model_execution_tensor_sha256(model: Any) -> str:
    """Hash every direct parameter/buffer, including nonpersistent buffers and aliases."""

    slots: list[tuple[str, str, bool, Tensor | None]] = []
    try:
        named_modules = tuple(model.named_modules())
    except (AttributeError, TypeError, ValueError) as error:
        raise GVSDecoderError("model cannot expose an execution tensor graph") from error
    for module_name, module in named_modules:
        parameters = getattr(module, "_parameters", None)
        buffers = getattr(module, "_buffers", None)
        nonpersistent = getattr(module, "_non_persistent_buffers_set", None)
        if (
            not isinstance(parameters, dict)
            or not isinstance(buffers, dict)
            or not isinstance(nonpersistent, set)
        ):
            raise GVSDecoderError("model has malformed parameter/buffer registries")
        if not nonpersistent.issubset(buffers):
            raise GVSDecoderError("model nonpersistent-buffer registry is inconsistent")
        prefix = f"{module_name}." if module_name else ""
        for name, value in parameters.items():
            if type(name) is not str or (value is not None and not isinstance(value, Tensor)):
                raise GVSDecoderError("model parameter registry has an invalid entry")
            slots.append((f"{prefix}{name}", "parameter", True, value))
        for name, value in buffers.items():
            if type(name) is not str or (value is not None and not isinstance(value, Tensor)):
                raise GVSDecoderError("model buffer registry has an invalid entry")
            slots.append((f"{prefix}{name}", "buffer", name not in nonpersistent, value))
    slots.sort(key=lambda row: (row[0], row[1]))
    first_path_by_identity: dict[int, str] = {}
    rows: list[dict[str, object]] = []
    for path, kind, persistent, tensor in slots:
        if tensor is None:
            rows.append(
                {
                    "kind": kind,
                    "path": path,
                    "persistent": persistent,
                    "tensor": None,
                }
            )
            continue
        alias_of = first_path_by_identity.setdefault(id(tensor), path)
        rows.append(
            {
                "kind": kind,
                "path": path,
                "persistent": persistent,
                "tensor": _tensor_execution_record(
                    tensor,
                    label=path,
                    alias_of=alias_of,
                    kind=kind,
                    persistent=persistent,
                ),
            }
        )
    if not any(row["tensor"] is not None for row in rows):
        raise GVSDecoderError("model execution tensor graph is empty")
    return _sha256_text(_canonical_json(rows))


def _validate_exact_barunlm_topology(model: BarunLM, config: BarunConfig) -> str:
    """Require the unwrapped canonical BarunLM module graph and config aliases."""

    integer_config_fields = (
        "vocab_size",
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "ffn_dim",
        "max_seq_len",
        "local_window",
        "full_attention_every",
        "residual_select_every",
        "mtp_offset",
    )
    float_config_fields = (
        "rope_theta",
        "rope_fraction",
        "activation_clip",
        "dropout",
        "norm_eps",
        "mtp_loss_weight",
    )
    bool_config_fields = ("attention_gate", "qk_norm", "tie_embeddings")
    if any(type(getattr(config, name, None)) is not int for name in integer_config_fields):
        raise GVSDecoderError("exact BarunConfig integer field types changed")
    if any(type(getattr(config, name, None)) is not float for name in float_config_fields):
        raise GVSDecoderError("exact BarunConfig float field types changed")
    if any(not _MATH_ISFINITE(getattr(config, name)) for name in float_config_fields):
        raise GVSDecoderError("exact BarunConfig float fields must be finite")
    if any(type(getattr(config, name, None)) is not bool for name in bool_config_fields):
        raise GVSDecoderError("exact BarunConfig boolean field types changed")
    if (
        min(
            config.vocab_size,
            config.dim,
            config.n_layers,
            config.n_heads,
            config.n_kv_heads,
            config.ffn_dim,
            config.max_seq_len,
            config.local_window,
            config.full_attention_every,
            config.mtp_offset,
        )
        < 1
        or config.residual_select_every < 0
        or config.dim % config.n_heads
        or config.n_heads % config.n_kv_heads
        or not 0.0 <= config.dropout < 1.0
        or config.rope_theta <= 0.0
        or not 0.0 < config.rope_fraction <= 1.0
        or config.activation_clip < 0.0
        or config.norm_eps <= 0.0
        or config.mtp_loss_weight < 0.0
        or config.rope_dim < 2
        or config.rope_dim % 2
    ):
        raise GVSDecoderError("exact BarunConfig values are structurally invalid")
    if model.config is not config:
        raise GVSDecoderError("exact BarunLM root config alias changed")

    expected: dict[str, type[torch.nn.Module]] = {
        "": BarunLM,
        "embedding": torch.nn.Embedding,
        "final_norm": RMSNorm,
        "layers": torch.nn.ModuleList,
        "lm_head": torch.nn.Linear,
        "selectors": torch.nn.ModuleList,
    }
    for index in range(config.n_layers):
        prefix = f"layers.{index}"
        expected.update(
            {
                prefix: BarunBlock,
                f"{prefix}.attn": GroupedAttention,
                f"{prefix}.attn.k_norm": RMSNorm if config.qk_norm else torch.nn.Identity,
                f"{prefix}.attn.k_proj": torch.nn.Linear,
                f"{prefix}.attn.o_proj": torch.nn.Linear,
                f"{prefix}.attn.q_norm": RMSNorm if config.qk_norm else torch.nn.Identity,
                f"{prefix}.attn.q_proj": torch.nn.Linear,
                f"{prefix}.attn.rope": PartialRotaryEmbedding,
                f"{prefix}.attn.v_proj": torch.nn.Linear,
                f"{prefix}.attn_norm": RMSNorm,
                f"{prefix}.ffn": BoundedSwiGLU,
                f"{prefix}.ffn.down": torch.nn.Linear,
                f"{prefix}.ffn.gate_up": torch.nn.Linear,
                f"{prefix}.ffn_norm": RMSNorm,
            }
        )
        if config.attention_gate:
            expected[f"{prefix}.attn.g_proj"] = torch.nn.Linear
    selector_count = (
        config.n_layers // config.residual_select_every if config.residual_select_every else 0
    )
    for index in range(selector_count):
        prefix = f"selectors.{index}"
        expected.update(
            {
                prefix: ResidualSelector,
                f"{prefix}.norm": RMSNorm,
                f"{prefix}.score": torch.nn.Linear,
            }
        )
    if config.mtp_loss_weight > 0:
        expected["mtp_norm"] = RMSNorm
        expected["mtp_proj"] = torch.nn.Linear

    actual_rows = tuple(model.named_modules())
    actual = {name: module for name, module in actual_rows}
    if len(actual) != len(actual_rows):
        raise GVSDecoderError("exact BarunLM topology contains duplicate module names")
    if set(actual) != set(expected):
        raise GVSDecoderError(
            "exact BarunLM topology names changed; "
            f"missing={sorted(set(expected) - set(actual))!r}, "
            f"extra={sorted(set(actual) - set(expected))!r}"
        )
    for name, expected_type in expected.items():
        if type(actual[name]) is not expected_type:
            raise GVSDecoderError(
                f"exact BarunLM module {name!r} must be {expected_type.__qualname__}"
            )

    expected_children: dict[str, tuple[str, ...]] = {
        "": ("embedding", "layers", "selectors", "final_norm", "lm_head"),
        "layers": tuple(str(index) for index in range(config.n_layers)),
        "selectors": tuple(str(index) for index in range(selector_count)),
    }
    if config.mtp_loss_weight > 0:
        expected_children[""] += ("mtp_norm", "mtp_proj")
    for index in range(config.n_layers):
        prefix = f"layers.{index}"
        attention_children = ("q_proj", "k_proj", "v_proj", "o_proj")
        if config.attention_gate:
            attention_children += ("g_proj",)
        expected_children.update(
            {
                prefix: ("attn_norm", "attn", "ffn_norm", "ffn"),
                f"{prefix}.attn": attention_children + ("q_norm", "k_norm", "rope"),
                f"{prefix}.ffn": ("gate_up", "down"),
            }
        )
    for index in range(selector_count):
        expected_children[f"selectors.{index}"] = ("norm", "score")

    expected_linear_dimensions: dict[str, tuple[int, int]] = {
        "lm_head": (config.dim, config.vocab_size),
    }
    expected_norm_dimensions: dict[str, int] = {"final_norm": config.dim}
    if config.mtp_loss_weight > 0:
        expected_linear_dimensions["mtp_proj"] = (config.dim, config.dim)
        expected_norm_dimensions["mtp_norm"] = config.dim
    for index in range(config.n_layers):
        prefix = f"layers.{index}"
        expected_linear_dimensions.update(
            {
                f"{prefix}.attn.q_proj": (config.dim, config.n_heads * config.head_dim),
                f"{prefix}.attn.k_proj": (config.dim, config.n_kv_heads * config.head_dim),
                f"{prefix}.attn.v_proj": (config.dim, config.n_kv_heads * config.head_dim),
                f"{prefix}.attn.o_proj": (config.n_heads * config.head_dim, config.dim),
                f"{prefix}.ffn.gate_up": (config.dim, 2 * config.ffn_dim),
                f"{prefix}.ffn.down": (config.ffn_dim, config.dim),
            }
        )
        if config.attention_gate:
            expected_linear_dimensions[f"{prefix}.attn.g_proj"] = (
                config.dim,
                config.n_heads * config.head_dim,
            )
        expected_norm_dimensions.update(
            {
                f"{prefix}.attn_norm": config.dim,
                f"{prefix}.ffn_norm": config.dim,
            }
        )
        if config.qk_norm:
            expected_norm_dimensions[f"{prefix}.attn.q_norm"] = config.head_dim
            expected_norm_dimensions[f"{prefix}.attn.k_norm"] = config.head_dim
    for index in range(selector_count):
        expected_linear_dimensions[f"selectors.{index}.score"] = (config.dim, 1)
        expected_norm_dimensions[f"selectors.{index}.norm"] = config.dim

    parameter_paths_by_identity: dict[int, list[str]] = {}
    buffer_paths_by_identity: dict[int, list[str]] = {}
    storage_slots: dict[tuple[str, int, int], list[tuple[str, int]]] = {}

    def register_storage(path: str, tensor: Tensor) -> None:
        storage = tensor.untyped_storage()
        expected_bytes = tensor.numel() * tensor.element_size()
        if (
            not tensor.is_contiguous()
            or tensor.storage_offset() != 0
            or storage.nbytes() != expected_bytes
        ):
            raise GVSDecoderError(f"exact BarunLM tensor {path!r} must own full contiguous storage")
        storage_slots.setdefault(
            (str(tensor.device), storage.data_ptr(), storage.nbytes()), []
        ).append((path, id(tensor)))

    topology_rows: list[dict[str, object]] = []
    for name, module in actual_rows:
        expected_parameter_names: tuple[str, ...] = ()
        expected_buffer_names: tuple[str, ...] = ()
        expected_nonpersistent_names: frozenset[str] = frozenset()
        if type(module) is torch.nn.Linear:
            expected_parameter_names = ("weight", "bias")
        elif type(module) is torch.nn.Embedding or (
            type(module) is RMSNorm and not name.startswith("selectors.")
        ):
            expected_parameter_names = ("weight",)
        elif type(module) is PartialRotaryEmbedding:
            expected_buffer_names = ("cos", "sin")
            expected_nonpersistent_names = frozenset(expected_buffer_names)

        if (
            type(module._parameters) is not dict
            or tuple(module._parameters) != expected_parameter_names
            or type(module._buffers) is not dict
            or tuple(module._buffers) != expected_buffer_names
            or type(module._non_persistent_buffers_set) is not set
            or frozenset(module._non_persistent_buffers_set) != expected_nonpersistent_names
            or type(module._modules) is not dict
            or tuple(module._modules) != expected_children.get(name, ())
        ):
            raise GVSDecoderError(f"exact BarunLM module {name!r} registry roster changed")
        if type(module.training) is not bool:
            raise GVSDecoderError(f"exact BarunLM module {name!r} has a non-boolean mode")
        prefix = f"{name}." if name else ""
        for slot_name, parameter in module._parameters.items():
            if slot_name == "bias":
                if parameter is not None:
                    raise GVSDecoderError(f"exact BarunLM linear {name!r} gained a bias")
                continue
            if type(parameter) is not torch.nn.Parameter:
                raise GVSDecoderError(f"exact BarunLM parameter {prefix + slot_name!r} is wrapped")
            path = prefix + slot_name
            parameter_paths_by_identity.setdefault(id(parameter), []).append(path)
            register_storage(path, parameter)
        for slot_name, buffer in module._buffers.items():
            if type(buffer) is not Tensor:
                raise GVSDecoderError(f"exact BarunLM buffer {prefix + slot_name!r} is wrapped")
            path = prefix + slot_name
            buffer_paths_by_identity.setdefault(id(buffer), []).append(path)
            register_storage(path, buffer)

        if type(module) is torch.nn.Linear:
            expected_in, expected_out = expected_linear_dimensions[name]
            if (
                type(module.in_features) is not int
                or module.in_features != expected_in
                or type(module.out_features) is not int
                or module.out_features != expected_out
                or tuple(module.weight.shape) != (expected_out, expected_in)
            ):
                raise GVSDecoderError(f"exact BarunLM linear {name!r} dimensions changed")
        elif type(module) is torch.nn.Embedding:
            if (
                type(module.num_embeddings) is not int
                or module.num_embeddings != config.vocab_size
                or type(module.embedding_dim) is not int
                or module.embedding_dim != config.dim
                or tuple(module.weight.shape) != (config.vocab_size, config.dim)
                or module.padding_idx is not None
                or module.max_norm is not None
                or type(module.norm_type) is not float
                or module.norm_type != 2.0
                or module.scale_grad_by_freq is not False
                or module.sparse is not False
            ):
                raise GVSDecoderError("exact BarunLM embedding behavior or dimensions changed")
        elif type(module) is RMSNorm:
            expected_dimension = expected_norm_dimensions[name]
            should_have_weight = not name.startswith("selectors.")
            if (
                type(module.eps) is not float
                or module.eps != config.norm_eps
                or (module.weight is not None) is not should_have_weight
                or (
                    module.weight is not None
                    and tuple(module.weight.shape) != (expected_dimension,)
                )
            ):
                raise GVSDecoderError(f"exact BarunLM RMSNorm {name!r} behavior changed")
        elif type(module) is PartialRotaryEmbedding:
            expected_shape = (1, 1, config.max_seq_len, config.rope_dim)
            if (
                type(module.dim) is not int
                or module.dim != config.rope_dim
                or tuple(module.cos.shape) != expected_shape
                or tuple(module.sin.shape) != expected_shape
                or module.cos.dtype is not module.sin.dtype
                or module.cos.dtype
                not in {torch.bfloat16, torch.float16, torch.float32, torch.float64}
                or module.cos.requires_grad
                or module.sin.requires_grad
            ):
                raise GVSDecoderError(f"exact BarunLM rotary module {name!r} behavior changed")

        topology_rows.append(
            {
                "buffers": list(expected_buffer_names),
                "children": list(expected_children.get(name, ())),
                "name": name,
                "nonpersistent_buffers": sorted(expected_nonpersistent_names),
                "parameters": list(expected_parameter_names),
                "type": f"{type(module).__module__}.{type(module).__qualname__}",
            }
        )

    allowed_parameter_alias = {"embedding.weight", "lm_head.weight"}
    for paths in parameter_paths_by_identity.values():
        if len(paths) > 1 and (not config.tie_embeddings or set(paths) != allowed_parameter_alias):
            raise GVSDecoderError("exact BarunLM parameter alias topology changed")
    if any(len(paths) > 1 for paths in buffer_paths_by_identity.values()):
        raise GVSDecoderError("exact BarunLM buffer alias topology changed")
    if any(len({identity for _, identity in slots}) > 1 for slots in storage_slots.values()):
        raise GVSDecoderError("exact BarunLM contains distinct tensors sharing one storage")

    if type(model.layers) is not torch.nn.ModuleList or len(model.layers) != config.n_layers:
        raise GVSDecoderError("exact BarunLM layer roster disagrees with config")
    if type(model.selectors) is not torch.nn.ModuleList or len(model.selectors) != selector_count:
        raise GVSDecoderError("exact BarunLM selector roster disagrees with config")
    for index, layer in enumerate(model.layers):
        expected_full = (index + 1) % config.full_attention_every == 0
        if layer.attn.config is not config or layer.attn.is_full is not expected_full:
            raise GVSDecoderError("exact BarunLM attention config/topology alias changed")
        if type(layer.dropout) is not float or layer.dropout != config.dropout:
            raise GVSDecoderError("exact BarunLM block dropout disagrees with config")
        if type(layer.ffn.clip) is not float or layer.ffn.clip != config.activation_clip:
            raise GVSDecoderError("exact BarunLM feed-forward clip disagrees with config")
    if (model.lm_head.weight is model.embedding.weight) is not config.tie_embeddings:
        raise GVSDecoderError("exact BarunLM embedding tie state disagrees with config")
    if (model.mtp_norm is not None) is not (config.mtp_loss_weight > 0) or (
        model.mtp_proj is not None
    ) is not (config.mtp_loss_weight > 0):
        raise GVSDecoderError("exact BarunLM MTP topology disagrees with config")

    return _sha256_text(_canonical_json(topology_rows))


def _encoding_probe_record(encoding: Any, *, label: str, depth: int = 0) -> dict[str, object]:
    if depth > 8:
        raise GVSDecoderError(f"{label} overflow nesting exceeds the probe bound")
    fields: dict[str, object] = {}
    for name in (
        "attention_mask",
        "ids",
        "offsets",
        "special_tokens_mask",
        "tokens",
        "type_ids",
        "word_ids",
    ):
        try:
            value = getattr(encoding, name)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise GVSDecoderError(f"{label}.{name} is not inspectable") from error
        fields[name] = _detach_json(value, label=f"{label}.{name}")
    try:
        overflowing = tuple(encoding.overflowing)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise GVSDecoderError(f"{label}.overflowing is not inspectable") from error
    if len(overflowing) > 64:
        raise GVSDecoderError(f"{label} has too many overflowing probe encodings")
    fields["overflowing"] = [
        _encoding_probe_record(child, label=f"{label}.overflowing[{index}]", depth=depth + 1)
        for index, child in enumerate(overflowing)
    ]
    return fields


def _exact_tokenizer_behavior_sha256(
    tokenizer: tokenizers.Tokenizer,
    reconstructed: tokenizers.Tokenizer,
) -> str:
    live_attributes = vars(tokenizer)
    reconstructed_attributes = vars(reconstructed)
    if type(live_attributes) is not dict or live_attributes:
        raise GVSDecoderError("exact Tokenizer cannot carry Python-side instance attributes")
    if type(reconstructed_attributes) is not dict or reconstructed_attributes:
        raise GVSDecoderError("reconstructed Tokenizer has unexpected Python-side attributes")

    discovered_flags: dict[str, bool] = {}
    for name, descriptor in vars(tokenizers.Tokenizer).items():
        if name.startswith("__") or type(descriptor).__name__ != "getset_descriptor":
            continue
        try:
            value = getattr(tokenizer, name)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise GVSDecoderError(f"exact Tokenizer flag discovery failed for {name}") from error
        if type(value) is bool:
            discovered_flags[name] = value
    if set(discovered_flags) != {"encode_special_tokens"}:
        raise GVSDecoderError(
            f"exact Tokenizer behavior-flag roster changed: {sorted(discovered_flags)!r}"
        )
    if type(tokenizer.encode_special_tokens) is not bool or (
        tokenizer.encode_special_tokens != reconstructed.encode_special_tokens
    ):
        raise GVSDecoderError(
            "exact Tokenizer encode_special_tokens is not preserved by reconstruction"
        )
    for name in ("padding", "truncation"):
        live = _detach_json(getattr(tokenizer, name), label=f"tokenizer.{name}")
        rebuilt = _detach_json(getattr(reconstructed, name), label=f"reconstructed.{name}")
        if live != rebuilt:
            raise GVSDecoderError(f"exact Tokenizer {name} differs after reconstruction")

    probe_texts = (
        "",
        "plain ASCII probe",
        "<eos>",
        "<pad>",
        "prefix<eos>suffix",
        "é e\u0301 नमस्ते",
    )
    probe_rows: list[dict[str, object]] = []
    for text_value in probe_texts:
        for add_special_tokens in (False, True):
            try:
                live_encoding = tokenizer.encode(
                    text_value,
                    add_special_tokens=add_special_tokens,
                )
                rebuilt_encoding = reconstructed.encode(
                    text_value,
                    add_special_tokens=add_special_tokens,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                raise GVSDecoderError("exact Tokenizer failed a reconstruction probe") from error
            live_record = _encoding_probe_record(live_encoding, label="tokenizer.probe")
            rebuilt_record = _encoding_probe_record(
                rebuilt_encoding,
                label="reconstructed.probe",
            )
            if live_record != rebuilt_record:
                raise GVSDecoderError(
                    "exact Tokenizer encode behavior differs after reconstruction"
                )
            probe_rows.append(
                {
                    "add_special_tokens": add_special_tokens,
                    "encoding": live_record,
                    "text_sha256": _sha256_text(text_value),
                }
            )
    for ids in ((), (tokenizer.token_to_id("<eos>"),), (tokenizer.token_to_id("<pad>"),)):
        if any(type(token_id) is not int for token_id in ids):
            raise GVSDecoderError("exact Tokenizer decode probe has an invalid token ID")
        live_decoded = tokenizer.decode(list(ids), skip_special_tokens=False)
        rebuilt_decoded = reconstructed.decode(list(ids), skip_special_tokens=False)
        if type(live_decoded) is not str or live_decoded != rebuilt_decoded:
            raise GVSDecoderError("exact Tokenizer decode behavior differs after reconstruction")
        _strict_string(live_decoded, label="tokenizer.decode_probe", maximum=4096)
        probe_rows.append(
            {
                "decoded_sha256": _sha256_text(live_decoded),
                "ids": list(ids),
                "skip_special_tokens": False,
            }
        )
    return _sha256_text(
        _canonical_json(
            {
                "behavior_flags": discovered_flags,
                "probes": probe_rows,
            }
        )
    )


_NN_MODULE_INTERNAL_FIELDS = frozenset(vars(torch.nn.Module()))


def _instance_state_sha256(
    value: object,
    *,
    label: str,
    excluded_fields: frozenset[str] = frozenset(),
) -> str:
    try:
        attributes = vars(value)
    except TypeError:
        attributes = {}
    if type(attributes) is not dict:
        raise GVSDecoderError(f"{label} must expose an exact attribute dictionary")
    state = {
        name: child
        for name, child in attributes.items()
        if type(name) is str and name not in excluded_fields
    }
    return _sha256_text(
        _canonical_json(_behavior_state_record(state, label=f"{label}.instance_state"))
    )


def _reject_execution_hooks(model: Any) -> None:
    global_hook_names = (
        "_global_backward_pre_hooks",
        "_global_buffer_registration_hooks",
        "_global_forward_hooks",
        "_global_forward_hooks_always_called",
        "_global_forward_hooks_with_kwargs",
        "_global_forward_pre_hooks",
        "_global_backward_hooks",
        "_global_module_registration_hooks",
        "_global_parameter_registration_hooks",
    )
    if any(getattr(torch_module, name, {}) for name in global_hook_names):
        raise GVSDecoderError("global Torch execution hooks are forbidden in the reference runtime")
    try:
        modules = tuple(model.named_modules())
    except (AttributeError, TypeError, ValueError) as error:
        raise GVSDecoderError("model must expose a deterministic named_modules graph") from error
    for module_name, module in modules:
        if type(module_name) is not str or not isinstance(module, torch.nn.Module):
            raise GVSDecoderError("model named_modules graph contains an invalid entry")
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise GVSDecoderError(
                f"model module {module_name!r} has a forbidden compiled call override"
            )
        for attribute in module.__dict__:
            if callable(getattr(type(module), attribute, None)):
                raise GVSDecoderError(
                    f"model module {module_name!r} shadows callable {attribute!r}"
                )
        for hook_name in (
            "_backward_hooks",
            "_backward_pre_hooks",
            "_forward_hooks",
            "_forward_hooks_always_called",
            "_forward_hooks_with_kwargs",
            "_forward_pre_hooks",
            "_forward_pre_hooks_with_kwargs",
            "_load_state_dict_post_hooks",
            "_load_state_dict_pre_hooks",
            "_state_dict_hooks",
            "_state_dict_pre_hooks",
        ):
            if getattr(module, hook_name, {}):
                raise GVSDecoderError(
                    f"model module {module_name!r} has a forbidden execution hook"
                )


def _adapter_inventory(model: Any) -> tuple[str, ...]:
    evidence: set[str] = set()
    try:
        named_modules = tuple(model.named_modules())
        state_keys = tuple(model.state_dict())
    except (AttributeError, TypeError, ValueError) as error:
        raise GVSDecoderError("model cannot provide adapter inventory evidence") from error
    for name, module in named_modules:
        identity = f"{name}:{type(module).__module__}.{type(module).__qualname__}"
        lowered = identity.casefold()
        if any(marker in lowered for marker in _ADAPTER_MARKERS):
            evidence.add(f"module:{identity}")
        for attribute in _ADAPTER_MARKERS:
            if hasattr(module, attribute):
                evidence.add(f"attribute:{name}:{attribute}")
    for key in state_keys:
        if type(key) is not str:
            raise GVSDecoderError("model state contains a non-string adapter inventory key")
        lowered = key.casefold()
        if any(marker in lowered for marker in _ADAPTER_MARKERS):
            evidence.add(f"state:{key}")
    return tuple(sorted(evidence))


def _model_module_graph_sha256(model: Any) -> str:
    rows: list[dict[str, object]] = []
    for name, module in model.named_modules():
        rows.append(
            {
                "call_callable_sha256": _callable_sha256(
                    type(module).__call__, label=f"model.modules[{name!r}].__call__"
                ),
                "forward_callable_sha256": _callable_sha256(
                    module.forward, label=f"model.modules[{name!r}].forward"
                ),
                "name": name,
                "state_sha256": _instance_state_sha256(
                    module,
                    label=f"model.modules[{name!r}]",
                    excluded_fields=_NN_MODULE_INTERNAL_FIELDS | frozenset({"config"}),
                ),
                "training": bool(module.training),
                "type": f"{type(module).__module__}.{type(module).__qualname__}",
            }
        )
    return _sha256_text(_canonical_json(rows))


def _cpu_long_tensor(value: object, *, label: str) -> Tensor:
    tensor = _TORCH_TENSOR(value, dtype=torch.long, device=_TORCH_DEVICE("cpu"))
    if tensor.device.type != "cpu" or tensor.dtype is not torch.long:
        raise GVSDecoderError(f"{label} was not allocated as an exact CPU int64 tensor")
    return tensor


@dataclass(frozen=True, slots=True)
class GVSDecoderContract:
    """Every semantic choice of the CPU reference decoder."""

    max_new_tokens: int = PROPOSED_MAX_NEW_TOKENS
    schema_version: str = GVS_DECODER_SCHEMA_VERSION
    status: str = "reference_only_proposal"
    launch_authorized: bool = False
    optimized_remote_exact_match_required: bool = True
    candidate_count: int = FROZEN_CANDIDATE_COUNT
    greedy_count: int = 1
    greedy_temperature: int = 0
    beam_width: int = FROZEN_BEAM_WIDTH
    beam_groups: int = 1
    diversity_penalty: int = 0
    sampling: bool = False
    repair: bool = False
    token_suppression: bool = False
    beam_cache: bool = False
    min_content_tokens: int = 0
    stop_rule: str = "first_eos_or_exact_max_new_tokens"
    finished_beam_policy: str = "finished_hypotheses_retain_active_width_slots"
    search_score: str = "float64_sum_all_generated_token_log_probabilities"
    final_length_normalization: str = "float64_mean_all_generated_token_log_probabilities"
    include_eos_in_score: bool = True
    include_generated_pad_in_score: bool = True
    filler_mask_semantics: str = "explicit_mask_only_token_id_never_implies_filler"
    reference_filler_mask: str = "all_false"
    failure_policy: str = "retain_all_eight_slots_with_stage_and_code"
    native_completion_rule: str = "eos_must_be_final_or_length_must_equal_max_new_tokens"
    failed_beam_final_order: str = "after_successful_scored_beams_then_failed_beam_rank_asc"
    model_identity: str = (
        "checkpoint_state_dict_plus_runtime_all_parameters_buffers_topology_and_callables"
    )
    tokenizer_identity: str = "to_str_plus_native_flags_reconstruction_probes_and_callables"
    source_identity: str = "sha256_exact_rendered_prompt_utf8"
    runtime_stability: str = "exact_pre_and_post_generation_identity_with_loaded_callables"
    prompt_encoding_rule: str = "mobile_renderer_then_encode_add_special_tokens_false"
    special_token_rule: str = "literal_eos_and_pad_distinct_and_in_vocabulary"
    native_generation_rule: str = "single_unpadded_model_generate_temperature_zero"
    beam_implementation: str = "single_example_full_prefix_no_cache_float64_log_softmax"
    likelihood_recompute_rule: str = "teacher_forced_full_prefix_float64_log_softmax"
    output_decode_rule: str = "tokens_before_eos_decode_skip_special_tokens_false"
    execution_mode: str = (
        "cpu_eval_inference_mode_autocast_off_clean_model_runtime_state_then_restore_training_flag"
    )
    max_prompt_utf8_bytes: int = _MAX_PROMPT_UTF8_BYTES
    max_raw_output_utf8_bytes: int = _MAX_RAW_OUTPUT_UTF8_BYTES
    max_schema_canonical_utf8_bytes: int = _MAX_SCHEMA_CANONICAL_UTF8_BYTES
    max_action_canonical_utf8_bytes: int = _MAX_ACTION_CANONICAL_UTF8_BYTES
    max_trace_artifact_bytes: int = _MAX_TRACE_BYTES
    max_analysis_artifact_bytes: int = _MAX_ANALYSIS_BYTES
    max_selection_artifact_bytes: int = _MAX_SELECTION_BYTES
    expansion_tie_break: tuple[str, ...] = (
        "cumulative_log_likelihood_desc",
        "parent_rank_asc",
        "token_id_asc",
    )
    final_tie_break: tuple[str, ...] = (
        "normalized_log_likelihood_desc",
        "raw_log_likelihood_desc",
        "beam_rank_asc",
    )

    def __post_init__(self) -> None:
        _strict_positive_int(self.max_new_tokens, label="max_new_tokens")
        required_values: dict[str, object] = {
            "schema_version": GVS_DECODER_SCHEMA_VERSION,
            "status": "reference_only_proposal",
            "launch_authorized": False,
            "optimized_remote_exact_match_required": True,
            "candidate_count": 8,
            "greedy_count": 1,
            "greedy_temperature": 0,
            "beam_width": 7,
            "beam_groups": 1,
            "diversity_penalty": 0,
            "sampling": False,
            "repair": False,
            "token_suppression": False,
            "beam_cache": False,
            "min_content_tokens": 0,
            "stop_rule": "first_eos_or_exact_max_new_tokens",
            "finished_beam_policy": "finished_hypotheses_retain_active_width_slots",
            "search_score": "float64_sum_all_generated_token_log_probabilities",
            "final_length_normalization": ("float64_mean_all_generated_token_log_probabilities"),
            "include_eos_in_score": True,
            "include_generated_pad_in_score": True,
            "filler_mask_semantics": "explicit_mask_only_token_id_never_implies_filler",
            "reference_filler_mask": "all_false",
            "failure_policy": "retain_all_eight_slots_with_stage_and_code",
            "native_completion_rule": ("eos_must_be_final_or_length_must_equal_max_new_tokens"),
            "failed_beam_final_order": ("after_successful_scored_beams_then_failed_beam_rank_asc"),
            "model_identity": (
                "checkpoint_state_dict_plus_runtime_all_parameters_buffers_topology_and_callables"
            ),
            "tokenizer_identity": ("to_str_plus_native_flags_reconstruction_probes_and_callables"),
            "source_identity": "sha256_exact_rendered_prompt_utf8",
            "runtime_stability": ("exact_pre_and_post_generation_identity_with_loaded_callables"),
            "prompt_encoding_rule": ("mobile_renderer_then_encode_add_special_tokens_false"),
            "special_token_rule": "literal_eos_and_pad_distinct_and_in_vocabulary",
            "native_generation_rule": "single_unpadded_model_generate_temperature_zero",
            "beam_implementation": ("single_example_full_prefix_no_cache_float64_log_softmax"),
            "likelihood_recompute_rule": "teacher_forced_full_prefix_float64_log_softmax",
            "output_decode_rule": "tokens_before_eos_decode_skip_special_tokens_false",
            "execution_mode": (
                "cpu_eval_inference_mode_autocast_off_clean_model_runtime_state_then_restore_training_flag"
            ),
            "max_prompt_utf8_bytes": _MAX_PROMPT_UTF8_BYTES,
            "max_raw_output_utf8_bytes": _MAX_RAW_OUTPUT_UTF8_BYTES,
            "max_schema_canonical_utf8_bytes": _MAX_SCHEMA_CANONICAL_UTF8_BYTES,
            "max_action_canonical_utf8_bytes": _MAX_ACTION_CANONICAL_UTF8_BYTES,
            "max_trace_artifact_bytes": _MAX_TRACE_BYTES,
            "max_analysis_artifact_bytes": _MAX_ANALYSIS_BYTES,
            "max_selection_artifact_bytes": _MAX_SELECTION_BYTES,
            "expansion_tie_break": (
                "cumulative_log_likelihood_desc",
                "parent_rank_asc",
                "token_id_asc",
            ),
            "final_tie_break": (
                "normalized_log_likelihood_desc",
                "raw_log_likelihood_desc",
                "beam_rank_asc",
            ),
        }
        for name, required in required_values.items():
            actual = getattr(self, name)
            if actual != required or type(actual) is not type(required):
                raise GVSDecoderError(f"{name} is frozen by the GVS-v1 reference")

    def to_record(self) -> dict[str, object]:
        return {
            field.name: (
                list(value) if isinstance(value := getattr(self, field.name), tuple) else value
            )
            for field in dataclasses.fields(self)
        }

    @property
    def sha256(self) -> str:
        return _sha256_text(_canonical_json(self.to_record()))


@dataclass(frozen=True, slots=True)
class GVSRuntimeIdentity:
    schema_version: str
    python_implementation: str
    python_version: str
    torch_version: str
    tokenizers_version: str
    tokenizers_implementation_sha256: str
    platform_system: str
    platform_machine: str
    device_type: str
    default_dtype: str
    torch_default_device: str
    parameter_dtypes: tuple[str, ...]
    buffer_dtypes: tuple[str, ...]
    deterministic_algorithms_enabled: bool
    torch_num_threads: int
    model_class: str
    model_class_source_sha256: str
    model_module_source_sha256: str
    model_config_sha256: str
    model_state_sha256: str
    model_contract: str
    model_callable_sha256: str
    model_dependency_sha256: str
    model_execution_tensor_sha256: str
    model_instance_state_sha256: str
    model_module_graph_sha256: str
    model_topology_sha256: str
    tokenizer_class: str
    tokenizer_state_sha256: str
    tokenizer_contract: str
    tokenizer_behavior_sha256: str
    tokenizer_callable_sha256: str
    tokenizer_instance_state_sha256: str
    generation_settings_sha256: str
    torch_execution_state_sha256: str
    adapter_inventory: tuple[str, ...]
    adapters_enabled: bool
    production_identity_contract_satisfied: bool
    decoder_source_sha256: str
    decoder_callable_sha256: str
    reference_impl_only: bool = True
    optimized_remote_exact_match_required: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != GVS_RUNTIME_SCHEMA_VERSION:
            raise GVSDecoderError("unsupported GVS runtime identity schema")
        for name in (
            "python_implementation",
            "python_version",
            "torch_version",
            "tokenizers_version",
            "platform_system",
            "platform_machine",
            "default_dtype",
            "torch_default_device",
            "model_class",
            "tokenizer_class",
        ):
            _strict_string(getattr(self, name), label=name, maximum=512)
        if self.device_type != "cpu":
            raise GVSDecoderError("the frozen reference runtime must be CPU")
        if self.torch_default_device != "cpu":
            raise GVSDecoderError("the frozen reference runtime requires Torch default device cpu")
        for name in (
            "model_class_source_sha256",
            "model_module_source_sha256",
            "model_config_sha256",
            "model_state_sha256",
            "model_callable_sha256",
            "model_dependency_sha256",
            "model_execution_tensor_sha256",
            "model_instance_state_sha256",
            "model_module_graph_sha256",
            "model_topology_sha256",
            "tokenizer_state_sha256",
            "tokenizer_behavior_sha256",
            "tokenizer_callable_sha256",
            "tokenizer_instance_state_sha256",
            "generation_settings_sha256",
            "torch_execution_state_sha256",
            "tokenizers_implementation_sha256",
            "decoder_source_sha256",
            "decoder_callable_sha256",
        ):
            _strict_sha256(getattr(self, name), label=name)
        for name in ("parameter_dtypes", "buffer_dtypes"):
            values = getattr(self, name)
            if type(values) is not tuple or any(type(value) is not str for value in values):
                raise GVSDecoderError(f"{name} must be a tuple of dtype names")
        if not self.parameter_dtypes and not self.buffer_dtypes:
            raise GVSDecoderError("runtime identity requires at least one model tensor")
        if type(self.deterministic_algorithms_enabled) is not bool:
            raise GVSDecoderError("deterministic_algorithms_enabled must be boolean")
        _strict_positive_int(self.torch_num_threads, label="torch_num_threads")
        if self.model_contract not in {"exact_barunlm_v1", "reference_test_double"}:
            raise GVSDecoderError("runtime model contract is unsupported")
        if self.tokenizer_contract not in {
            "exact_tokenizers_reconstructible_v1",
            "reference_test_double",
        }:
            raise GVSDecoderError("runtime tokenizer contract is unsupported")
        if self.model_contract == "exact_barunlm_v1" and self.model_class != (
            f"{BarunLM.__module__}.{BarunLM.__qualname__}"
        ):
            raise GVSDecoderError("exact BarunLM runtime has the wrong model class")
        if self.tokenizer_contract == "exact_tokenizers_reconstructible_v1" and (
            self.tokenizer_class
            != f"{tokenizers.Tokenizer.__module__}.{tokenizers.Tokenizer.__qualname__}"
        ):
            raise GVSDecoderError("exact Tokenizer runtime has the wrong tokenizer class")
        if type(self.adapter_inventory) is not tuple or any(
            type(value) is not str for value in self.adapter_inventory
        ):
            raise GVSDecoderError("adapter inventory must be an exact string tuple")
        if self.adapter_inventory:
            raise GVSDecoderError("candidate generator must have an empty adapter inventory")
        if self.adapters_enabled is not False:
            raise GVSDecoderError("candidate generator adapters must be explicitly off")
        expected_production_eligibility = (
            self.model_contract == "exact_barunlm_v1"
            and self.tokenizer_contract == "exact_tokenizers_reconstructible_v1"
        )
        if (
            type(self.production_identity_contract_satisfied) is not bool
            or self.production_identity_contract_satisfied is not expected_production_eligibility
        ):
            raise GVSDecoderError("production runtime eligibility disagrees with exact contracts")
        if (
            self.reference_impl_only is not True
            or self.optimized_remote_exact_match_required is not True
        ):
            raise GVSDecoderError("runtime identity cannot authorize or replace remote equivalence")

    def to_record(self) -> dict[str, object]:
        return {
            "adapter_inventory": list(self.adapter_inventory),
            "adapters_enabled": self.adapters_enabled,
            "buffer_dtypes": list(self.buffer_dtypes),
            "decoder_callable_sha256": self.decoder_callable_sha256,
            "decoder_source_sha256": self.decoder_source_sha256,
            "default_dtype": self.default_dtype,
            "deterministic_algorithms_enabled": self.deterministic_algorithms_enabled,
            "device_type": self.device_type,
            "generation_settings_sha256": self.generation_settings_sha256,
            "model_class": self.model_class,
            "model_callable_sha256": self.model_callable_sha256,
            "model_class_source_sha256": self.model_class_source_sha256,
            "model_contract": self.model_contract,
            "model_dependency_sha256": self.model_dependency_sha256,
            "model_execution_tensor_sha256": self.model_execution_tensor_sha256,
            "model_module_source_sha256": self.model_module_source_sha256,
            "model_instance_state_sha256": self.model_instance_state_sha256,
            "model_module_graph_sha256": self.model_module_graph_sha256,
            "model_topology_sha256": self.model_topology_sha256,
            "model_config_sha256": self.model_config_sha256,
            "model_state_sha256": self.model_state_sha256,
            "optimized_remote_exact_match_required": self.optimized_remote_exact_match_required,
            "parameter_dtypes": list(self.parameter_dtypes),
            "platform_machine": self.platform_machine,
            "platform_system": self.platform_system,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "production_identity_contract_satisfied": (self.production_identity_contract_satisfied),
            "reference_impl_only": self.reference_impl_only,
            "schema_version": self.schema_version,
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_behavior_sha256": self.tokenizer_behavior_sha256,
            "tokenizer_callable_sha256": self.tokenizer_callable_sha256,
            "tokenizer_contract": self.tokenizer_contract,
            "tokenizer_instance_state_sha256": self.tokenizer_instance_state_sha256,
            "tokenizer_state_sha256": self.tokenizer_state_sha256,
            "tokenizers_implementation_sha256": self.tokenizers_implementation_sha256,
            "tokenizers_version": self.tokenizers_version,
            "torch_default_device": self.torch_default_device,
            "torch_execution_state_sha256": self.torch_execution_state_sha256,
            "torch_num_threads": self.torch_num_threads,
            "torch_version": self.torch_version,
        }

    @property
    def sha256(self) -> str:
        return _sha256_text(_canonical_json(self.to_record()))


def _require_cpu_model(model: Any) -> tuple[Tensor, ...]:
    try:
        tensors = tuple(model.parameters()) + tuple(model.buffers())
    except (AttributeError, TypeError) as error:
        raise GVSDecoderError("model must expose parameters and buffers") from error
    if not tensors:
        raise GVSDecoderError("reference model must expose at least one tensor")
    if any(not isinstance(tensor, Tensor) for tensor in tensors):
        raise GVSDecoderError("model parameters and buffers must be tensors")
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise GVSDecoderError("GVS reference is CPU-only and never moves a model")
    return tensors


def _snapshot_model_training_modes(model: Any) -> tuple[tuple[torch.nn.Module, bool], ...]:
    """Capture every module mode so temporary eval cannot leak or flatten mixed state."""

    try:
        modules = tuple(model.named_modules())
    except (AttributeError, TypeError, ValueError) as error:
        raise GVSDecoderError("model cannot expose training modes") from error
    if not modules or modules[0][1] is not model:
        raise GVSDecoderError("model training-mode graph has no canonical root")
    snapshot: list[tuple[torch.nn.Module, bool]] = []
    seen: set[int] = set()
    for name, module in modules:
        if type(name) is not str or not isinstance(module, torch.nn.Module) or id(module) in seen:
            raise GVSDecoderError("model training-mode graph contains an invalid entry")
        seen.add(id(module))
        if type(module.training) is not bool:
            raise GVSDecoderError(f"model module {name!r} has a non-boolean training mode")
        snapshot.append((module, module.training))
    return tuple(snapshot)


def _set_model_training_modes(
    snapshot: tuple[tuple[torch.nn.Module, bool], ...],
    mode: bool,
) -> None:
    if type(mode) is not bool:
        raise GVSDecoderError("temporary model mode must be boolean")
    for module, _ in snapshot:
        object.__setattr__(module, "training", mode)


def _require_model_training_mode(
    snapshot: tuple[tuple[torch.nn.Module, bool], ...],
    expected: bool,
) -> None:
    if any(
        type(module.training) is not bool or module.training is not expected
        for module, _ in snapshot
    ):
        raise GVSDecoderError("model mutated its module training modes during reference execution")


def _restore_model_training_modes(
    snapshot: tuple[tuple[torch.nn.Module, bool], ...],
) -> None:
    for module, mode in snapshot:
        object.__setattr__(module, "training", mode)
    if any(
        type(module.training) is not bool or module.training is not mode
        for module, mode in snapshot
    ):
        raise GVSDecoderError("model training modes could not be restored exactly")


def _model_dimensions(model: Any) -> tuple[int, int]:
    config = getattr(model, "config", None)
    return (
        _strict_positive_int(getattr(config, "vocab_size", None), label="model.config.vocab_size"),
        _strict_positive_int(
            getattr(config, "max_seq_len", None), label="model.config.max_seq_len"
        ),
    )


def _model_state_sha256(model: Any) -> str:
    try:
        state = model.state_dict()
    except (AttributeError, TypeError) as error:
        raise GVSDecoderError("model must expose a deterministic state_dict") from error
    if not isinstance(state, Mapping) or not state:
        raise GVSDecoderError("model state_dict must be a nonempty mapping")
    digest = _HASH_SHA256()
    for name, tensor in sorted(state.items()):
        if type(name) is not str or not isinstance(tensor, Tensor):
            raise GVSDecoderError("model state_dict contains an invalid entry")
        if tensor.device.type != "cpu" or tensor.layout is not torch.strided:
            raise GVSDecoderError("reference model state must be dense CPU tensors")
        detached = tensor.detach().contiguous()
        digest.update(len(name.encode("utf-8")).to_bytes(8, "big"))
        digest.update(name.encode("utf-8"))
        metadata = _canonical_json(
            {"dtype": str(detached.dtype), "shape": list(detached.shape)}
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(detached.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def inspect_gvs_runtime(model: Any, tokenizer: TokenizerLike) -> GVSRuntimeIdentity:
    """Snapshot every runtime property that can change reference candidate bytes."""

    decoder_source_path = Path(__file__)
    decoder_source_sha256 = _file_sha256(decoder_source_path)
    torch_execution_state = _require_reference_torch_execution_state()
    default_device = str(torch_execution_state["default_device"])
    if type(model) is BarunLM:
        _clear_barun_model_execution_caches(model)
    _require_cpu_model(model)
    _reject_execution_hooks(model)
    adapter_inventory = _adapter_inventory(model)
    if adapter_inventory:
        raise GVSDecoderError(
            f"candidate generator adapter inventory must be empty: {adapter_inventory!r}"
        )
    config = getattr(model, "config", None)
    if config is None:
        raise GVSDecoderError("model must expose config")
    model_contract = "reference_test_double"
    model_topology_sha256 = _sha256_text(
        _canonical_json(
            [
                {
                    "name": name,
                    "type": f"{type(module).__module__}.{type(module).__qualname__}",
                }
                for name, module in model.named_modules()
            ]
        )
    )
    model_dependency_sha256 = _sha256_text(_canonical_json({"contract": "reference_test_double"}))
    if type(model) is BarunLM:
        if type(config) is not BarunConfig:
            raise GVSDecoderError("exact BarunLM production runtime requires exact BarunConfig")
        model_topology_sha256 = _validate_exact_barunlm_topology(model, config)
        model_dependency_sha256 = _model_dependency_sha256()
        model_contract = "exact_barunlm_v1"
    config_record = _detach_json(config, label="model.config")
    try:
        generation_config = getattr(model, "generation_config", None)
    except (AttributeError, TypeError, ValueError) as error:
        raise GVSDecoderError("model generation configuration is not inspectable") from error
    if generation_config is not None:
        raise GVSDecoderError("reference generation forbids an implicit model generation_config")
    try:
        tokenizer_state = tokenizer.to_str()
    except (AttributeError, TypeError, ValueError) as error:
        raise GVSDecoderError(
            "tokenizer must expose deterministic to_str identity bytes"
        ) from error
    tokenizer_state = _strict_string(
        tokenizer_state, label="tokenizer.to_str", maximum=32 * 1024 * 1024
    )
    tokenizer_contract = "reference_test_double"
    tokenizer_behavior_sha256 = _sha256_text(_canonical_json({"contract": "reference_test_double"}))
    if type(tokenizer) is tokenizers.Tokenizer:
        try:
            reconstructed = tokenizers.Tokenizer.from_str(tokenizer_state)
            reconstructed_state = reconstructed.to_str()
        except (TypeError, ValueError, RuntimeError) as error:
            raise GVSDecoderError("tokenizer state is not exactly reconstructible") from error
        if reconstructed_state != tokenizer_state:
            raise GVSDecoderError("tokenizer state failed byte-exact reconstruction")
        if reconstructed.get_vocab_size(with_added_tokens=True) != getattr(
            config, "vocab_size", None
        ):
            raise GVSDecoderError("reconstructible tokenizer vocabulary differs from model config")
        reconstructed_eos = reconstructed.token_to_id("<eos>")
        reconstructed_pad = reconstructed.token_to_id("<pad>")
        if (
            type(reconstructed_eos) is not int
            or type(reconstructed_pad) is not int
            or reconstructed_eos == reconstructed_pad
        ):
            raise GVSDecoderError(
                "reconstructible tokenizer requires distinct literal EOS and pad tokens"
            )
        tokenizer_behavior_sha256 = _exact_tokenizer_behavior_sha256(
            tokenizer,
            reconstructed,
        )
        tokenizer_contract = "exact_tokenizers_reconstructible_v1"
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    tokenizer_class = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    try:
        model_source_path = Path(inspect.getsourcefile(type(model)) or "")
    except TypeError as error:
        raise GVSDecoderError("model class must come from inspectable Python source") from error
    if not model_source_path.is_file():
        raise GVSDecoderError("model class must come from an inspectable source file")
    model_module_source_sha256 = _file_sha256(model_source_path)
    tokenizers_implementation_path = Path(tokenizers_native.__file__ or "")
    if not tokenizers_implementation_path.is_file():
        raise GVSDecoderError("Tokenizers implementation binary is not inspectable")
    tokenizers_implementation_sha256 = _file_sha256(tokenizers_implementation_path)
    model_callable_record = {
        "call": _callable_sha256(type(model).__call__, label="model.__call__"),
        **{
            name: _callable_sha256(getattr(model, name), label=f"model.{name}")
            for name in (
                "buffers",
                "eval",
                "forward",
                "generate",
                "modules",
                "named_buffers",
                "named_modules",
                "named_parameters",
                "parameters",
                "state_dict",
                "train",
            )
        },
    }
    tokenizer_callable_record = {
        name: _callable_sha256(getattr(tokenizer, name), label=f"tokenizer.{name}")
        for name in ("decode", "encode", "to_str", "token_to_id")
    }
    model_callable_sha256 = _sha256_text(_canonical_json(model_callable_record))
    generation_settings_sha256 = _sha256_text(
        _canonical_json(
            {
                "generate_callable_sha256": model_callable_record["generate"],
                "implicit_generation_config": None,
                "settings": _REFERENCE_GENERATION_SETTINGS,
            }
        )
    )
    model_instance_state_sha256 = _instance_state_sha256(
        model,
        label="model",
        excluded_fields=_NN_MODULE_INTERNAL_FIELDS | frozenset({"config"}),
    )
    if tokenizer_contract == "exact_tokenizers_reconstructible_v1":
        tokenizer_instance_state_sha256 = _sha256_text(
            _canonical_json(
                {
                    "contract": tokenizer_contract,
                    "behavior_sha256": tokenizer_behavior_sha256,
                    "state_sha256": _sha256_text(tokenizer_state),
                }
            )
        )
    else:
        tokenizer_instance_state_sha256 = _instance_state_sha256(tokenizer, label="tokenizer")
    model_execution_tensor_sha256 = _model_execution_tensor_sha256(model)
    model_module_graph_sha256 = _model_module_graph_sha256(model)
    decoder_callable_sha256 = _decoder_callable_sha256()
    torch_execution_state_sha256 = _sha256_text(_canonical_json(torch_execution_state))

    try:
        tokenizer_state_after = tokenizer.to_str()
    except (AttributeError, TypeError, ValueError) as error:
        raise GVSDecoderError("tokenizer identity failed its post-snapshot read") from error
    if tokenizer_state_after != tokenizer_state:
        raise GVSDecoderError("tokenizer state mutated while runtime identity was inspected")
    if _file_sha256(decoder_source_path) != decoder_source_sha256:
        raise GVSDecoderError("decoder source mutated while runtime identity was inspected")
    if _file_sha256(model_source_path) != model_module_source_sha256:
        raise GVSDecoderError("model source mutated while runtime identity was inspected")
    if _file_sha256(tokenizers_implementation_path) != tokenizers_implementation_sha256:
        raise GVSDecoderError(
            "Tokenizers implementation mutated while runtime identity was inspected"
        )
    if _torch_execution_state_sha256() != torch_execution_state_sha256:
        raise GVSDecoderError("Torch execution state mutated while runtime identity was inspected")
    return GVSRuntimeIdentity(
        schema_version=GVS_RUNTIME_SCHEMA_VERSION,
        python_implementation=platform.python_implementation(),
        python_version=".".join(str(value) for value in sys.version_info[:3]),
        torch_version=str(torch.__version__),
        tokenizers_version=str(tokenizers.__version__),
        tokenizers_implementation_sha256=tokenizers_implementation_sha256,
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        device_type="cpu",
        default_dtype=str(torch.get_default_dtype()),
        torch_default_device=default_device,
        parameter_dtypes=tuple(sorted({str(parameter.dtype) for parameter in model.parameters()})),
        buffer_dtypes=tuple(sorted({str(buffer.dtype) for buffer in model.buffers()})),
        deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
        torch_num_threads=torch.get_num_threads(),
        model_class=model_class,
        model_class_source_sha256=_source_sha256(type(model)),
        model_module_source_sha256=model_module_source_sha256,
        model_config_sha256=_sha256_text(_canonical_json(config_record)),
        model_state_sha256=_model_state_sha256(model),
        model_contract=model_contract,
        model_callable_sha256=model_callable_sha256,
        model_dependency_sha256=model_dependency_sha256,
        model_execution_tensor_sha256=model_execution_tensor_sha256,
        model_instance_state_sha256=model_instance_state_sha256,
        model_module_graph_sha256=model_module_graph_sha256,
        model_topology_sha256=model_topology_sha256,
        tokenizer_class=tokenizer_class,
        tokenizer_state_sha256=_sha256_text(tokenizer_state),
        tokenizer_contract=tokenizer_contract,
        tokenizer_behavior_sha256=tokenizer_behavior_sha256,
        tokenizer_callable_sha256=_sha256_text(_canonical_json(tokenizer_callable_record)),
        tokenizer_instance_state_sha256=tokenizer_instance_state_sha256,
        generation_settings_sha256=generation_settings_sha256,
        torch_execution_state_sha256=torch_execution_state_sha256,
        adapter_inventory=adapter_inventory,
        adapters_enabled=False,
        production_identity_contract_satisfied=(
            model_contract == "exact_barunlm_v1"
            and tokenizer_contract == "exact_tokenizers_reconstructible_v1"
        ),
        decoder_source_sha256=decoder_source_sha256,
        decoder_callable_sha256=decoder_callable_sha256,
    )


@dataclass(frozen=True, slots=True)
class LikelihoodTrace:
    generated_token_ids: tuple[int, ...]
    token_log_likelihoods: tuple[float, ...]
    filler_mask: tuple[bool, ...]
    score_included: tuple[bool, ...]
    raw_log_likelihood: float
    normalized_log_likelihood: float
    scored_token_count: int

    def __post_init__(self) -> None:
        length = len(self.generated_token_ids)
        if not length or any(
            len(values) != length
            for values in (self.token_log_likelihoods, self.filler_mask, self.score_included)
        ):
            raise GVSDecoderError("likelihood arrays must have one entry per generated token")
        if type(self.generated_token_ids) is not tuple or any(
            type(token_id) is not int or token_id < 0 for token_id in self.generated_token_ids
        ):
            raise GVSDecoderError("generated token IDs must be non-negative integer tuples")
        if type(self.token_log_likelihoods) is not tuple or any(
            type(value) is not float or not _MATH_ISFINITE(value) or value > 0.0
            for value in self.token_log_likelihoods
        ):
            raise GVSDecoderError("token log likelihoods must be finite non-positive floats")
        if type(self.filler_mask) is not tuple or any(
            type(value) is not bool for value in self.filler_mask
        ):
            raise GVSDecoderError("filler mask must be a boolean tuple")
        expected_included = tuple(not filler for filler in self.filler_mask)
        if self.score_included != expected_included:
            raise GVSDecoderError(
                "score inclusion must be exactly the inverse explicit filler mask"
            )
        if type(self.scored_token_count) is not int or self.scored_token_count != sum(
            expected_included
        ):
            raise GVSDecoderError("scored token count disagrees with explicit filler mask")
        if self.scored_token_count < 1:
            raise GVSDecoderError(
                "a candidate must contain at least one non-filler generated token"
            )
        for name in ("raw_log_likelihood", "normalized_log_likelihood"):
            value = getattr(self, name)
            if type(value) is not float or not _MATH_ISFINITE(value) or value > 0.0:
                raise GVSDecoderError(f"{name} must be an exact finite non-positive float")
        try:
            expected_raw = _MATH_FSUM(
                value
                for value, included in zip(
                    self.token_log_likelihoods,
                    expected_included,
                    strict=True,
                )
                if included
            )
        except OverflowError as error:
            raise GVSDecoderError("included token likelihood sum overflowed") from error
        if not _MATH_ISFINITE(expected_raw):
            raise GVSDecoderError("included token likelihood sum is non-finite")
        if self.raw_log_likelihood != expected_raw:
            raise GVSDecoderError("raw likelihood was not recomputed from included token scores")
        if self.normalized_log_likelihood != expected_raw / self.scored_token_count:
            raise GVSDecoderError("normalized likelihood is not the exact scored-token mean")

    def to_record(self) -> dict[str, object]:
        return {
            "filler_mask": list(self.filler_mask),
            "generated_token_ids": list(self.generated_token_ids),
            "normalized_log_likelihood": self.normalized_log_likelihood,
            "raw_log_likelihood": self.raw_log_likelihood,
            "score_included": list(self.score_included),
            "scored_token_count": self.scored_token_count,
            "token_log_likelihoods": list(self.token_log_likelihoods),
        }


@dataclass(frozen=True, slots=True)
class CandidateTrace:
    attempted_rank: int
    origin: str
    beam_rank: int | None
    generated_token_ids: tuple[int, ...]
    content_token_ids: tuple[int, ...]
    output_present: bool
    raw_output: str | None
    raw_output_sha256: str | None
    eos_emitted: bool
    eos_position: int | None
    truncated: bool
    likelihood: LikelihoodTrace | None
    beam_search_raw_log_likelihood: float | None
    failure_stage: str | None
    failure_code: str | None

    def __post_init__(self) -> None:
        if type(self.attempted_rank) is not int or not 0 <= self.attempted_rank < 8:
            raise GVSDecoderError("attempted rank must be in [0, 8)")
        if self.origin == "greedy":
            if self.attempted_rank != 0 or self.beam_rank is not None:
                raise GVSDecoderError("greedy must occupy rank zero without a beam rank")
        elif self.origin == "beam":
            if (
                self.attempted_rank == 0
                or type(self.beam_rank) is not int
                or not 0 <= self.beam_rank < 7
            ):
                raise GVSDecoderError("beam candidates require a nonzero slot and valid beam rank")
        else:
            raise GVSDecoderError("candidate origin must be greedy or beam")
        if type(self.generated_token_ids) is not tuple or any(
            type(token_id) is not int or token_id < 0 for token_id in self.generated_token_ids
        ):
            raise GVSDecoderError("candidate token IDs must be a non-negative integer tuple")
        if type(self.content_token_ids) is not tuple:
            raise GVSDecoderError("content token IDs must be a tuple")
        if type(self.output_present) is not bool:
            raise GVSDecoderError("output_present must be boolean")
        if self.output_present is not (self.raw_output is not None):
            raise GVSDecoderError("output_present must agree with nullable raw output")
        if self.output_present is not (self.raw_output_sha256 is not None):
            raise GVSDecoderError("output_present must agree with nullable raw output hash")
        if self.raw_output is not None:
            _strict_string(
                self.raw_output,
                label="raw_output",
                maximum=_MAX_RAW_OUTPUT_UTF8_BYTES,
            )
            if self.raw_output_sha256 != _sha256_text(self.raw_output):
                raise GVSDecoderError("raw output hash disagrees with output bytes")
        if type(self.eos_emitted) is not bool or type(self.truncated) is not bool:
            raise GVSDecoderError("EOS and truncation fields must be booleans")
        failed = self.failure_stage is not None
        if self.generated_token_ids and not failed:
            if self.eos_emitted:
                if type(self.eos_position) is not int or self.eos_position < 0:
                    raise GVSDecoderError("EOS-emitted candidate requires an EOS position")
                if len(self.generated_token_ids) != self.eos_position + 1:
                    raise GVSDecoderError("no generated token may follow EOS")
                if self.content_token_ids != self.generated_token_ids[: self.eos_position]:
                    raise GVSDecoderError("content must exclude terminal EOS")
                if self.truncated:
                    raise GVSDecoderError("EOS-emitted candidate cannot be truncated")
            elif (
                self.eos_position is not None or self.content_token_ids != self.generated_token_ids
            ):
                raise GVSDecoderError("non-EOS content must contain the complete continuation")
        elif not self.generated_token_ids and (
            self.eos_emitted
            or self.eos_position is not None
            or self.truncated
            or self.content_token_ids
        ):
            raise GVSDecoderError("empty failed slot cannot claim EOS, content, or truncation")
        if failed is not (self.failure_code is not None):
            raise GVSDecoderError("candidate failure stage and code must be jointly nullable")
        if failed:
            if self.failure_stage not in _FAILURE_STAGES:
                raise GVSDecoderError("candidate has an unsupported failure stage")
            _strict_identifier(self.failure_code, label="failure_code")
        elif not self.output_present or self.likelihood is None or not self.generated_token_ids:
            raise GVSDecoderError("successful candidate requires tokens, output, and likelihood")
        if self.likelihood is not None:
            if type(self.likelihood) is not LikelihoodTrace:
                raise GVSDecoderError("candidate likelihood must be an exact LikelihoodTrace")
            if self.likelihood.generated_token_ids != self.generated_token_ids:
                raise GVSDecoderError("candidate tokens disagree with likelihood trace")
        if self.origin == "greedy" and self.beam_search_raw_log_likelihood is not None:
            raise GVSDecoderError("greedy cannot carry a beam-search score")
        if self.origin == "beam" and self.failure_stage == "beam_search":
            if self.beam_search_raw_log_likelihood is not None:
                raise GVSDecoderError("beam-search failure cannot claim a retained search score")
        elif self.origin == "beam" and (
            type(self.beam_search_raw_log_likelihood) is not float
            or not _MATH_ISFINITE(self.beam_search_raw_log_likelihood)
            or self.beam_search_raw_log_likelihood > 0.0
        ):
            raise GVSDecoderError(
                "every searched beam requires a finite retained search score that is non-positive"
            )

        if not failed:
            return
        if self.failure_stage is None or self.failure_code is None:
            raise GVSDecoderError("failed candidate lost its exact stage/code evidence")
        if self.failure_stage == "beam_search":
            if (
                self.origin != "beam"
                or self.generated_token_ids
                or self.content_token_ids
                or self.output_present
                or self.eos_emitted
                or self.eos_position is not None
                or self.truncated
                or self.likelihood is not None
            ):
                raise GVSDecoderError("beam-search failure must be an exact empty failed slot")
        elif self.failure_stage == "decode":
            if not self.generated_token_ids or self.output_present or self.likelihood is not None:
                raise GVSDecoderError(
                    "decode failure requires retained tokens and no decoded output or likelihood"
                )
        elif self.failure_stage == "likelihood":
            if (
                not self.generated_token_ids
                or not self.output_present
                or self.likelihood is not None
            ):
                raise GVSDecoderError(
                    "likelihood failure requires retained decoded output and no likelihood"
                )
        elif self.failure_stage == "native_generate":
            if self.origin != "greedy":
                raise GVSDecoderError(
                    "native-generation failure is valid only for greedy rank zero"
                )
            if self.output_present:
                if (
                    self.failure_code != IncompleteNativeGenerationError.__name__
                    or not self.generated_token_ids
                    or self.content_token_ids != self.generated_token_ids
                    or self.eos_emitted
                    or self.eos_position is not None
                    or self.truncated
                    or self.likelihood is None
                ):
                    raise GVSDecoderError(
                        "materialized native failure must be one exact incomplete continuation"
                    )
            elif self.generated_token_ids:
                if (
                    self.failure_code != NativeGenerationAfterEOSError.__name__
                    or not self.eos_emitted
                    or type(self.eos_position) is not int
                    or not 0 <= self.eos_position < len(self.generated_token_ids) - 1
                    or self.generated_token_ids[self.eos_position] < 0
                    or self.content_token_ids != self.generated_token_ids[: self.eos_position]
                    or self.truncated
                    or self.likelihood is not None
                ):
                    raise GVSDecoderError(
                        "token-retaining native failure must be an exact post-EOS continuation"
                    )
            elif (
                self.content_token_ids
                or self.eos_emitted
                or self.eos_position is not None
                or self.truncated
                or self.likelihood is not None
            ):
                raise GVSDecoderError("empty native-generation failure has contradictory evidence")

    def _evidence_record(self) -> dict[str, object]:
        return {
            "attempted_rank": self.attempted_rank,
            "beam_rank": self.beam_rank,
            "beam_search_raw_log_likelihood": self.beam_search_raw_log_likelihood,
            "content_token_ids": list(self.content_token_ids),
            "eos_emitted": self.eos_emitted,
            "eos_position": self.eos_position,
            "failure_code": self.failure_code,
            "failure_stage": self.failure_stage,
            "generated_token_ids": list(self.generated_token_ids),
            "likelihood": None if self.likelihood is None else self.likelihood.to_record(),
            "origin": self.origin,
            "output_present": self.output_present,
            "raw_output": self.raw_output,
            "raw_output_sha256": self.raw_output_sha256,
            "truncated": self.truncated,
        }

    @property
    def sha256(self) -> str:
        return _sha256_text(_canonical_json(self._evidence_record()))

    @property
    def candidate_id(self) -> str:
        return f"gvs-slot-{self.attempted_rank}-{self.sha256}"

    def to_record(self) -> dict[str, object]:
        record = self._evidence_record()
        record["candidate_id"] = self.candidate_id
        record["candidate_trace_sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CandidateSetTrace:
    schema_version: str
    contract: GVSDecoderContract
    contract_sha256: str
    runtime: GVSRuntimeIdentity
    runtime_sha256: str
    model_sha256: str
    tokenizer_sha256: str
    source_sha256: str
    prompt_contract_version: str
    prompt_template_sha256: str
    prompt: str
    prompt_sha256: str
    prompt_token_ids: tuple[int, ...]
    eos_token_id: int
    pad_token_id: int
    vocab_size: int
    model_max_seq_len: int
    candidates: tuple[CandidateTrace, ...]

    def __post_init__(self) -> None:
        if self.schema_version != GVS_TRACE_SCHEMA_VERSION:
            raise GVSDecoderError("unsupported GVS trace schema")
        if type(self.contract) is not GVSDecoderContract:
            raise GVSDecoderError("trace contract must be GVSDecoderContract")
        if type(self.runtime) is not GVSRuntimeIdentity:
            raise GVSDecoderError("trace runtime must be GVSRuntimeIdentity")
        for name in (
            "contract_sha256",
            "runtime_sha256",
            "model_sha256",
            "tokenizer_sha256",
            "source_sha256",
            "prompt_template_sha256",
            "prompt_sha256",
        ):
            _strict_sha256(getattr(self, name), label=name)
        if self.contract_sha256 != self.contract.sha256:
            raise GVSDecoderError("trace contract hash disagrees with embedded contract")
        if self.runtime_sha256 != self.runtime.sha256:
            raise GVSDecoderError("trace runtime hash disagrees with embedded runtime")
        if self.model_sha256 != self.runtime.model_state_sha256:
            raise GVSDecoderError("trace model identity is not the exact runtime state hash")
        if self.tokenizer_sha256 != self.runtime.tokenizer_state_sha256:
            raise GVSDecoderError("trace tokenizer identity is not the exact runtime state hash")
        if self.prompt_contract_version != PROMPT_CONTRACT_VERSION:
            raise GVSDecoderError("trace prompt contract version changed")
        if self.prompt_template_sha256 != PROMPT_TEMPLATE_SHA256:
            raise GVSDecoderError("trace prompt template hash changed")
        _strict_string(self.prompt, label="prompt", maximum=_MAX_PROMPT_UTF8_BYTES)
        if self.prompt_sha256 != _sha256_text(self.prompt):
            raise GVSDecoderError("prompt hash disagrees with embedded prompt bytes")
        if self.source_sha256 != self.prompt_sha256:
            raise GVSDecoderError("trace source identity must be the exact rendered prompt hash")
        _strict_positive_int(self.vocab_size, label="vocab_size")
        _strict_positive_int(self.model_max_seq_len, label="model_max_seq_len")
        _strict_token_tuple(
            self.prompt_token_ids, label="prompt_token_ids", vocab_size=self.vocab_size
        )
        for name in ("eos_token_id", "pad_token_id"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < self.vocab_size:
                raise GVSDecoderError(f"{name} must be within the vocabulary")
        if self.eos_token_id == self.pad_token_id:
            raise GVSDecoderError("EOS and pad token IDs must be distinct")
        if len(self.prompt_token_ids) + self.contract.max_new_tokens > self.model_max_seq_len:
            raise GVSDecoderError("trace prompt and generation limit exceed model context")
        if type(self.candidates) is not tuple or len(self.candidates) != FROZEN_CANDIDATE_COUNT:
            raise GVSDecoderError("trace must preserve exactly eight attempted slots")
        if any(type(candidate) is not CandidateTrace for candidate in self.candidates):
            raise GVSDecoderError("trace contains a non-candidate entry")
        if tuple(candidate.attempted_rank for candidate in self.candidates) != tuple(range(8)):
            raise GVSDecoderError("trace candidate slots must be ordered 0 through 7")
        if self.candidates[0].origin != "greedy" or any(
            candidate.origin != "beam" for candidate in self.candidates[1:]
        ):
            raise GVSDecoderError("trace must contain greedy rank zero followed by seven beams")
        beam_ranks = tuple(candidate.beam_rank for candidate in self.candidates[1:])
        if tuple(sorted(beam_ranks)) != tuple(range(7)):
            raise GVSDecoderError("trace beam ranks must occur exactly once across 0 through 6")
        if len({candidate.candidate_id for candidate in self.candidates}) != 8:
            raise GVSDecoderError("slot-bound candidate IDs must be unique")
        for candidate in self.candidates:
            _strict_token_tuple(
                candidate.generated_token_ids,
                label="candidate generated_token_ids",
                vocab_size=self.vocab_size,
                allow_empty=candidate.failure_stage is not None,
            )
            _strict_token_tuple(
                candidate.content_token_ids,
                label="candidate content_token_ids",
                vocab_size=self.vocab_size,
                allow_empty=True,
            )
            if len(candidate.generated_token_ids) > self.contract.max_new_tokens:
                raise GVSDecoderError("candidate exceeds the frozen generation limit")
            if candidate.generated_token_ids and candidate.failure_stage is None:
                if candidate.eos_emitted:
                    if candidate.eos_position is None:
                        raise GVSDecoderError("candidate EOS evidence lost its exact position")
                    if candidate.generated_token_ids[candidate.eos_position] != self.eos_token_id:
                        raise GVSDecoderError("candidate EOS position does not contain bound EOS")
                elif self.eos_token_id in candidate.generated_token_ids:
                    raise GVSDecoderError("non-EOS candidate contains an unrecorded EOS")
                if candidate.truncated is not (
                    not candidate.eos_emitted
                    and len(candidate.generated_token_ids) == self.contract.max_new_tokens
                ):
                    raise GVSDecoderError("candidate truncation disagrees with EOS and exact limit")
            elif candidate.failure_stage == "native_generate" and candidate.generated_token_ids:
                if candidate.failure_code == IncompleteNativeGenerationError.__name__:
                    if (
                        self.eos_token_id in candidate.generated_token_ids
                        or len(candidate.generated_token_ids) >= self.contract.max_new_tokens
                    ):
                        raise GVSDecoderError("incomplete native failure is not short and EOS-free")
                elif candidate.failure_code == NativeGenerationAfterEOSError.__name__:
                    if candidate.eos_position is None:
                        raise GVSDecoderError("native post-EOS failure lost its EOS position")
                    if (
                        candidate.generated_token_ids[candidate.eos_position] != self.eos_token_id
                        or candidate.eos_position
                        != candidate.generated_token_ids.index(self.eos_token_id)
                    ):
                        raise GVSDecoderError(
                            "post-EOS native failure disagrees with the bound EOS token"
                        )
            elif candidate.failure_stage in {"decode", "likelihood"}:
                try:
                    expected_layout = _candidate_layout(
                        candidate.generated_token_ids,
                        eos_token_id=self.eos_token_id,
                        max_new_tokens=self.contract.max_new_tokens,
                    )
                except GVSDecoderError:
                    if (
                        candidate.failure_stage != "decode"
                        or candidate.content_token_ids != candidate.generated_token_ids
                        or candidate.eos_emitted
                        or candidate.eos_position is not None
                        or candidate.truncated
                    ):
                        raise GVSDecoderError(
                            "failed candidate does not preserve its exact layout failure"
                        ) from None
                else:
                    actual_layout = (
                        candidate.content_token_ids,
                        candidate.eos_emitted,
                        candidate.eos_position,
                        candidate.truncated,
                    )
                    if actual_layout != expected_layout:
                        raise GVSDecoderError(
                            "failed candidate layout differs from retained generated tokens"
                        )
            if candidate.likelihood is not None:
                if any(candidate.likelihood.filler_mask):
                    raise GVSDecoderError("CPU reference candidates cannot contain future filler")
                if not all(candidate.likelihood.score_included):
                    raise GVSDecoderError("CPU reference must score every generated token")

    def to_record(self) -> dict[str, object]:
        return {
            "candidates": [candidate.to_record() for candidate in self.candidates],
            "contract": self.contract.to_record(),
            "contract_sha256": self.contract_sha256,
            "eos_token_id": self.eos_token_id,
            "model_max_seq_len": self.model_max_seq_len,
            "model_sha256": self.model_sha256,
            "pad_token_id": self.pad_token_id,
            "prompt": self.prompt,
            "prompt_contract_version": self.prompt_contract_version,
            "prompt_sha256": self.prompt_sha256,
            "prompt_template_sha256": self.prompt_template_sha256,
            "prompt_token_ids": list(self.prompt_token_ids),
            "runtime": self.runtime.to_record(),
            "runtime_sha256": self.runtime_sha256,
            "schema_version": self.schema_version,
            "source_sha256": self.source_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "vocab_size": self.vocab_size,
        }

    @property
    def sha256(self) -> str:
        return _sha256_text(_canonical_json(self.to_record()))

    def golden_vector_record(self) -> dict[str, object]:
        """Runtime-independent algorithm vector used by immutable synthetic goldens."""

        return {
            "candidates": [
                {
                    "attempted_rank": candidate.attempted_rank,
                    "beam_rank": candidate.beam_rank,
                    "beam_search_raw_log_likelihood": candidate.beam_search_raw_log_likelihood,
                    "eos_emitted": candidate.eos_emitted,
                    "failure_code": candidate.failure_code,
                    "failure_stage": candidate.failure_stage,
                    "generated_token_ids": list(candidate.generated_token_ids),
                    "likelihood": (
                        None if candidate.likelihood is None else candidate.likelihood.to_record()
                    ),
                    "origin": candidate.origin,
                    "raw_output": candidate.raw_output,
                    "truncated": candidate.truncated,
                }
                for candidate in self.candidates
            ],
            "contract": self.contract.to_record(),
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "vocab_size": self.vocab_size,
        }

    @property
    def golden_vector_sha256(self) -> str:
        return _sha256_text(_canonical_json(self.golden_vector_record()))

    def to_artifact_record(self) -> dict[str, object]:
        return {
            "artifact_schema_version": GVS_TRACE_ARTIFACT_SCHEMA_VERSION,
            "candidate_set_trace_sha256": self.sha256,
            "trace": self.to_record(),
        }

    def to_json_bytes(self) -> bytes:
        return (_canonical_json(self.to_artifact_record()) + "\n").encode("utf-8")


@dataclass(frozen=True, slots=True)
class CandidateAnalysis:
    attempted_rank: int
    candidate_id: str
    candidate_trace_sha256: str
    raw_output_sha256: str | None
    output_present: bool
    generation_failed: bool
    truncated: bool
    parse_valid: bool
    schema_valid: bool
    canonical_action_json: str | None
    canonical_action_sha256: str | None
    error_stage: str | None
    error_code: str | None
    error_path: str | None
    normalized_log_likelihood: float | None
    raw_log_likelihood: float | None

    def __post_init__(self) -> None:
        if type(self.attempted_rank) is not int or not 0 <= self.attempted_rank < 8:
            raise GVSDecoderError("analysis attempted rank is invalid")
        _strict_identifier(self.candidate_id, label="candidate_id")
        _strict_sha256(self.candidate_trace_sha256, label="candidate_trace_sha256")
        if (
            type(self.output_present) is not bool
            or type(self.generation_failed) is not bool
            or type(self.truncated) is not bool
        ):
            raise GVSDecoderError("analysis output/truncation fields must be booleans")
        if type(self.parse_valid) is not bool or type(self.schema_valid) is not bool:
            raise GVSDecoderError("analysis validity fields must be booleans")
        if self.output_present is not (self.raw_output_sha256 is not None):
            raise GVSDecoderError("analysis output presence must agree with raw hash")
        if not self.output_present and not self.generation_failed:
            raise GVSDecoderError("missing analysis output must be an explicit generation failure")
        if self.generation_failed and (self.parse_valid or self.schema_valid):
            raise GVSDecoderError("generation-failed candidate cannot be parse/schema valid")
        if self.raw_output_sha256 is not None:
            _strict_sha256(self.raw_output_sha256, label="raw_output_sha256")
        if self.schema_valid and not self.parse_valid:
            raise GVSDecoderError("schema-valid output must be parse-valid")
        if self.schema_valid is not (self.canonical_action_json is not None):
            raise GVSDecoderError("schema validity must agree with canonical Action IR")
        if self.schema_valid is not (self.canonical_action_sha256 is not None):
            raise GVSDecoderError("schema validity must agree with canonical Action IR hash")
        if self.canonical_action_json is not None and self.canonical_action_sha256 != _sha256_text(
            self.canonical_action_json
        ):
            raise GVSDecoderError("canonical Action IR hash disagrees with bytes")
        if self.canonical_action_json is not None:
            _strict_string(
                self.canonical_action_json,
                label="canonical_action_json",
                maximum=_MAX_ACTION_CANONICAL_UTF8_BYTES,
            )
        has_error = self.error_stage is not None
        if has_error is not (self.error_code is not None) or has_error is not (
            self.error_path is not None
        ):
            raise GVSDecoderError("analysis error fields must be jointly nullable")
        expected_stage: str | None
        if self.generation_failed:
            expected_stage = "generation"
        elif not self.parse_valid:
            expected_stage = "parse"
        elif not self.schema_valid:
            expected_stage = "schema"
        else:
            expected_stage = None
        if self.error_stage != expected_stage:
            raise GVSDecoderError("analysis error stage disagrees with exact validity state")
        if has_error:
            if self.error_stage not in _ANALYSIS_ERROR_STAGES:
                raise GVSDecoderError("analysis contains an unsupported error stage")
            _strict_identifier(self.error_code, label="error_code")
            _strict_string(self.error_path, label="error_path", maximum=1024)
        likelihood_present = self.normalized_log_likelihood is not None
        if likelihood_present is not (self.raw_log_likelihood is not None):
            raise GVSDecoderError("analysis likelihood fields must be jointly nullable")
        for value in (self.normalized_log_likelihood, self.raw_log_likelihood):
            if value is not None and (
                type(value) is not float or not _MATH_ISFINITE(value) or value > 0.0
            ):
                raise GVSDecoderError("analysis likelihoods must be finite non-positive floats")

    @property
    def selection_eligible(self) -> bool:
        return (
            self.schema_valid
            and not self.truncated
            and self.normalized_log_likelihood is not None
            and self.raw_log_likelihood is not None
        )

    def to_record(self) -> dict[str, object]:
        return {
            "attempted_rank": self.attempted_rank,
            "candidate_id": self.candidate_id,
            "candidate_trace_sha256": self.candidate_trace_sha256,
            "canonical_action_json": self.canonical_action_json,
            "canonical_action_sha256": self.canonical_action_sha256,
            "error_code": self.error_code,
            "error_path": self.error_path,
            "error_stage": self.error_stage,
            "generation_failed": self.generation_failed,
            "normalized_log_likelihood": self.normalized_log_likelihood,
            "output_present": self.output_present,
            "parse_valid": self.parse_valid,
            "raw_log_likelihood": self.raw_log_likelihood,
            "raw_output_sha256": self.raw_output_sha256,
            "schema_valid": self.schema_valid,
            "selection_eligible": self.selection_eligible,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class CandidateSetAnalysis:
    schema_version: str
    candidate_set_trace_sha256: str
    schema_canonical_json: str
    schema_sha256: str
    evaluator_source_sha256: str
    evaluator_runtime_sha256: str
    candidates: tuple[CandidateAnalysis, ...]

    def __post_init__(self) -> None:
        if self.schema_version != GVS_ANALYSIS_SCHEMA_VERSION:
            raise GVSDecoderError("unsupported GVS analysis schema")
        for name in (
            "candidate_set_trace_sha256",
            "schema_sha256",
            "evaluator_source_sha256",
            "evaluator_runtime_sha256",
        ):
            _strict_sha256(getattr(self, name), label=name)
        _strict_string(
            self.schema_canonical_json,
            label="schema_canonical_json",
            maximum=_MAX_SCHEMA_CANONICAL_UTF8_BYTES,
        )
        if self.schema_sha256 != _sha256_text(self.schema_canonical_json):
            raise GVSDecoderError("schema hash disagrees with canonical schema bytes")
        if type(self.candidates) is not tuple or len(self.candidates) != 8:
            raise GVSDecoderError("analysis must preserve all eight attempted candidates")
        if any(type(candidate) is not CandidateAnalysis for candidate in self.candidates):
            raise GVSDecoderError("analysis contains a non-candidate entry")
        if tuple(candidate.attempted_rank for candidate in self.candidates) != tuple(range(8)):
            raise GVSDecoderError("analysis ranks must be ordered 0 through 7")
        if len({candidate.candidate_id for candidate in self.candidates}) != 8:
            raise GVSDecoderError("analysis candidate IDs must be unique")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_set_trace_sha256": self.candidate_set_trace_sha256,
            "candidates": [candidate.to_record() for candidate in self.candidates],
            "evaluator_runtime_sha256": self.evaluator_runtime_sha256,
            "evaluator_source_sha256": self.evaluator_source_sha256,
            "schema_canonical_json": self.schema_canonical_json,
            "schema_sha256": self.schema_sha256,
            "schema_version": self.schema_version,
        }

    @property
    def sha256(self) -> str:
        return _sha256_text(_canonical_json(self.to_record()))

    def to_artifact_record(self) -> dict[str, object]:
        return {
            "analysis": self.to_record(),
            "analysis_sha256": self.sha256,
            "artifact_schema_version": GVS_ANALYSIS_ARTIFACT_SCHEMA_VERSION,
        }

    def to_json_bytes(self) -> bytes:
        return (_canonical_json(self.to_artifact_record()) + "\n").encode("utf-8")


@dataclass(frozen=True, slots=True)
class LikelihoodSelection:
    schema_version: str
    analysis_sha256: str
    selected_rank: int
    selected_candidate_id: str
    selected_candidate_trace_sha256: str
    selected_canonical_action_sha256: str
    eligible_ranks_in_order: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema_version != GVS_SELECTION_SCHEMA_VERSION:
            raise GVSDecoderError("unsupported GVS selection schema")
        for name in (
            "analysis_sha256",
            "selected_candidate_trace_sha256",
            "selected_canonical_action_sha256",
        ):
            _strict_sha256(getattr(self, name), label=name)
        if type(self.selected_rank) is not int or not 0 <= self.selected_rank < 8:
            raise GVSDecoderError("selected rank must be an exact candidate slot")
        _strict_identifier(self.selected_candidate_id, label="selected_candidate_id")
        if type(self.eligible_ranks_in_order) is not tuple or not self.eligible_ranks_in_order:
            raise GVSDecoderError("selection must bind a nonempty eligible ordering")
        if any(type(rank) is not int or not 0 <= rank < 8 for rank in self.eligible_ranks_in_order):
            raise GVSDecoderError("eligible ranks must be exact candidate slots")
        if len(set(self.eligible_ranks_in_order)) != len(self.eligible_ranks_in_order):
            raise GVSDecoderError("eligible ranks must be unique")
        if self.selected_rank != self.eligible_ranks_in_order[0]:
            raise GVSDecoderError("selected rank must be first in eligible ordering")

    def to_record(self) -> dict[str, object]:
        return {
            "analysis_sha256": self.analysis_sha256,
            "eligible_ranks_in_order": list(self.eligible_ranks_in_order),
            "schema_version": self.schema_version,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_trace_sha256": self.selected_candidate_trace_sha256,
            "selected_canonical_action_sha256": self.selected_canonical_action_sha256,
            "selected_rank": self.selected_rank,
        }

    @property
    def sha256(self) -> str:
        return _sha256_text(_canonical_json(self.to_record()))

    def to_artifact_record(self) -> dict[str, object]:
        return {
            "artifact_schema_version": GVS_SELECTION_ARTIFACT_SCHEMA_VERSION,
            "selection": self.to_record(),
            "selection_sha256": self.sha256,
        }

    def to_json_bytes(self) -> bytes:
        return (_canonical_json(self.to_artifact_record()) + "\n").encode("utf-8")


@dataclass(frozen=True, slots=True)
class _Beam:
    token_ids: tuple[int, ...]
    search_score: float
    finished: bool
    rank: int


def _model_logits(model: Any, input_ids: Tensor, *, vocab_size: int) -> Tensor:
    with _guard_torch_execution_state():
        output = model(input_ids)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor) or logits.shape != (
        input_ids.shape[0],
        input_ids.shape[1],
        vocab_size,
    ):
        raise GVSDecoderError("model forward must return exact rank-three vocabulary logits")
    if (
        logits.device.type != "cpu"
        or logits.layout is not torch.strided
        or not torch.is_floating_point(logits)
    ):
        raise GVSDecoderError("reference model forward must return dense floating CPU logits")
    last = logits[:, -1].to(dtype=torch.float64)
    if not _TORCH_ISFINITE(last).all().item():
        raise GVSDecoderError("model produced non-finite next-token logits")
    return last


def recompute_likelihood(
    model: Any,
    *,
    prompt_token_ids: tuple[int, ...],
    generated_token_ids: tuple[int, ...],
    filler_mask: tuple[bool, ...] | None = None,
) -> LikelihoodTrace:
    """Teacher-force all generated tokens; only an explicit mask can mark filler."""

    _require_reference_torch_execution_state()
    _require_cpu_model(model)
    vocab_size, max_seq_len = _model_dimensions(model)
    prompt_token_ids = _strict_token_tuple(
        prompt_token_ids, label="prompt_token_ids", vocab_size=vocab_size
    )
    generated_token_ids = _strict_token_tuple(
        generated_token_ids, label="generated_token_ids", vocab_size=vocab_size
    )
    if filler_mask is None:
        filler_mask = (False,) * len(generated_token_ids)
    if (
        type(filler_mask) is not tuple
        or len(filler_mask) != len(generated_token_ids)
        or any(type(value) is not bool for value in filler_mask)
    ):
        raise GVSDecoderError("filler_mask must have one explicit boolean per generated token")
    if len(prompt_token_ids) + len(generated_token_ids) > max_seq_len:
        raise GVSDecoderError("candidate exceeds the model context length")

    full_ids = prompt_token_ids + generated_token_ids
    input_ids = _cpu_long_tensor([full_ids], label="likelihood input_ids")
    training_modes = _snapshot_model_training_modes(model)
    try:
        _set_model_training_modes(training_modes, False)
        with _TORCH_INFERENCE_MODE():
            with _guard_torch_execution_state():
                output = model(input_ids)
            logits = getattr(output, "logits", None)
            if not isinstance(logits, Tensor) or logits.shape != (1, len(full_ids), vocab_size):
                raise GVSDecoderError(
                    "model logits shape disagrees during likelihood recomputation"
                )
            if (
                logits.device.type != "cpu"
                or logits.layout is not torch.strided
                or not torch.is_floating_point(logits)
            ):
                raise GVSDecoderError(
                    "likelihood recomputation must return dense floating CPU logits"
                )
            prediction_logits = logits[0, len(prompt_token_ids) - 1 : len(full_ids) - 1].to(
                dtype=torch.float64
            )
            if not _TORCH_ISFINITE(prediction_logits).all().item():
                raise GVSDecoderError("model produced non-finite likelihood logits")
            log_probs = _TORCH_LOG_SOFTMAX(prediction_logits, dim=-1)
            targets = _cpu_long_tensor(generated_token_ids, label="likelihood targets")[:, None]
            values = log_probs.gather(dim=1, index=targets).squeeze(1)
            _require_model_training_mode(training_modes, False)
    finally:
        _restore_model_training_modes(training_modes)
    token_scores = tuple(float(value) for value in values.tolist())
    included = tuple(not filler for filler in filler_mask)
    count = sum(included)
    if count < 1:
        raise GVSDecoderError("candidate contains no scoreable token after explicit filler masking")
    try:
        raw = float(
            _MATH_FSUM(value for value, keep in zip(token_scores, included, strict=True) if keep)
        )
    except OverflowError as error:
        raise GVSDecoderError("recomputed token likelihood sum overflowed") from error
    if not _MATH_ISFINITE(raw):
        raise GVSDecoderError("recomputed token likelihood sum is non-finite")
    return LikelihoodTrace(
        generated_token_ids=generated_token_ids,
        token_log_likelihoods=token_scores,
        filler_mask=filler_mask,
        score_included=included,
        raw_log_likelihood=raw,
        normalized_log_likelihood=float(raw / count),
        scored_token_count=count,
    )


def _ordinary_beam(
    model: Any,
    *,
    prompt_token_ids: tuple[int, ...],
    eos_token_id: int,
    vocab_size: int,
    max_new_tokens: int,
) -> tuple[_Beam, ...]:
    """Run the exact fixed-width full-prefix reference beam."""

    beams = (_Beam(token_ids=(), search_score=0.0, finished=False, rank=0),)
    for _ in range(max_new_tokens):
        expanded: list[tuple[float, int, int, tuple[int, ...], bool]] = []
        for parent_rank, beam in enumerate(beams):
            if beam.finished:
                expanded.append((beam.search_score, parent_rank, -1, beam.token_ids, True))
                continue
            input_ids = _cpu_long_tensor(
                [prompt_token_ids + beam.token_ids], label="beam input_ids"
            )
            logits = _model_logits(model, input_ids, vocab_size=vocab_size)[0]
            log_probs = _TORCH_LOG_SOFTMAX(logits, dim=-1)
            token_order = _TORCH_ARGSORT(log_probs, descending=True, stable=True)[:7].tolist()
            for token_id in token_order:
                score = float(beam.search_score + float(log_probs[token_id].item()))
                expanded.append(
                    (
                        score,
                        parent_rank,
                        token_id,
                        beam.token_ids + (token_id,),
                        token_id == eos_token_id,
                    )
                )
        expanded.sort(key=lambda row: (-row[0], row[1], row[2]))
        chosen = expanded[:7]
        if len(chosen) != 7:
            raise GVSDecoderError("ordinary beam could not preserve seven attempted hypotheses")
        beams = tuple(
            _Beam(token_ids=row[3], search_score=row[0], finished=row[4], rank=rank)
            for rank, row in enumerate(chosen)
        )
        if all(beam.finished for beam in beams):
            break
    return beams


def _candidate_layout(
    generated_token_ids: tuple[int, ...], *, eos_token_id: int, max_new_tokens: int
) -> tuple[tuple[int, ...], bool, int | None, bool]:
    eos_positions = [
        index for index, token_id in enumerate(generated_token_ids) if token_id == eos_token_id
    ]
    if eos_positions:
        eos_position = eos_positions[0]
        if eos_position != len(generated_token_ids) - 1:
            raise GVSDecoderError("generated candidate contains tokens after EOS")
        return generated_token_ids[:eos_position], True, eos_position, False
    return (
        generated_token_ids,
        False,
        None,
        len(generated_token_ids) == max_new_tokens,
    )


def _failure_code(error: Exception) -> str:
    return _strict_identifier(type(error).__name__, label="failure_code")


def _failed_slot(
    *,
    attempted_rank: int,
    origin: str,
    beam_rank: int | None,
    stage: str,
    error: Exception,
    generated_token_ids: tuple[int, ...] = (),
    content_token_ids: tuple[int, ...] = (),
    output_present: bool = False,
    raw_output: str | None = None,
    eos_emitted: bool = False,
    eos_position: int | None = None,
    truncated: bool = False,
    likelihood: LikelihoodTrace | None = None,
    beam_search_raw_log_likelihood: float | None = None,
) -> CandidateTrace:
    return CandidateTrace(
        attempted_rank=attempted_rank,
        origin=origin,
        beam_rank=beam_rank,
        generated_token_ids=generated_token_ids,
        content_token_ids=content_token_ids,
        output_present=output_present,
        raw_output=raw_output,
        raw_output_sha256=None if raw_output is None else _sha256_text(raw_output),
        eos_emitted=eos_emitted,
        eos_position=eos_position,
        truncated=truncated,
        likelihood=likelihood,
        beam_search_raw_log_likelihood=beam_search_raw_log_likelihood,
        failure_stage=stage,
        failure_code=_failure_code(error),
    )


def _materialize_candidate(
    model: Any,
    tokenizer: TokenizerLike,
    *,
    attempted_rank: int,
    origin: str,
    beam_rank: int | None,
    generated_token_ids: tuple[int, ...],
    beam_search_raw_log_likelihood: float | None,
    prompt_token_ids: tuple[int, ...],
    eos_token_id: int,
    max_new_tokens: int,
    precomputed_likelihood: LikelihoodTrace | None = None,
    forced_likelihood_error: Exception | None = None,
    retained_failure: tuple[str, Exception] | None = None,
) -> CandidateTrace:
    if precomputed_likelihood is not None and forced_likelihood_error is not None:
        raise GVSDecoderError("candidate cannot have both a likelihood and likelihood failure")
    try:
        content_ids, eos_emitted, eos_position, truncated = _candidate_layout(
            generated_token_ids,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        )
    except _RETAINABLE_EXCEPTIONS as error:
        return _failed_slot(
            attempted_rank=attempted_rank,
            origin=origin,
            beam_rank=beam_rank,
            stage="decode",
            error=error,
            generated_token_ids=generated_token_ids,
            content_token_ids=generated_token_ids,
            beam_search_raw_log_likelihood=beam_search_raw_log_likelihood,
        )
    try:
        raw_output = tokenizer.decode(list(content_ids), skip_special_tokens=False)
        if type(raw_output) is not str:
            raise GVSDecoderError("tokenizer.decode must return a string")
        _strict_string(
            raw_output,
            label="raw_output",
            maximum=_MAX_RAW_OUTPUT_UTF8_BYTES,
        )
    except _RETAINABLE_EXCEPTIONS as error:
        return _failed_slot(
            attempted_rank=attempted_rank,
            origin=origin,
            beam_rank=beam_rank,
            stage="decode",
            error=error,
            generated_token_ids=generated_token_ids,
            content_token_ids=content_ids,
            eos_emitted=eos_emitted,
            eos_position=eos_position,
            truncated=truncated,
            beam_search_raw_log_likelihood=beam_search_raw_log_likelihood,
        )
    if forced_likelihood_error is not None:
        return _failed_slot(
            attempted_rank=attempted_rank,
            origin=origin,
            beam_rank=beam_rank,
            stage="likelihood",
            error=forced_likelihood_error,
            generated_token_ids=generated_token_ids,
            content_token_ids=content_ids,
            output_present=True,
            raw_output=raw_output,
            eos_emitted=eos_emitted,
            eos_position=eos_position,
            truncated=truncated,
            beam_search_raw_log_likelihood=beam_search_raw_log_likelihood,
        )
    if precomputed_likelihood is None:
        try:
            likelihood = recompute_likelihood(
                model,
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=generated_token_ids,
                filler_mask=(False,) * len(generated_token_ids),
            )
        except _RETAINABLE_EXCEPTIONS as error:
            return _failed_slot(
                attempted_rank=attempted_rank,
                origin=origin,
                beam_rank=beam_rank,
                stage="likelihood",
                error=error,
                generated_token_ids=generated_token_ids,
                content_token_ids=content_ids,
                output_present=True,
                raw_output=raw_output,
                eos_emitted=eos_emitted,
                eos_position=eos_position,
                truncated=truncated,
                beam_search_raw_log_likelihood=beam_search_raw_log_likelihood,
            )
    else:
        likelihood = precomputed_likelihood
    failure_stage = None if retained_failure is None else retained_failure[0]
    failure_code = None if retained_failure is None else _failure_code(retained_failure[1])
    return CandidateTrace(
        attempted_rank=attempted_rank,
        origin=origin,
        beam_rank=beam_rank,
        generated_token_ids=generated_token_ids,
        content_token_ids=content_ids,
        output_present=True,
        raw_output=raw_output,
        raw_output_sha256=_sha256_text(raw_output),
        eos_emitted=eos_emitted,
        eos_position=eos_position,
        truncated=truncated,
        likelihood=likelihood,
        beam_search_raw_log_likelihood=beam_search_raw_log_likelihood,
        failure_stage=failure_stage,
        failure_code=failure_code,
    )


def _native_greedy_tokens(
    model: Any,
    input_ids: Tensor,
    *,
    prompt_token_ids: tuple[int, ...],
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
    vocab_size: int,
) -> tuple[int, ...]:
    with _guard_torch_execution_state():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    if (
        not isinstance(output, Tensor)
        or output.ndim != 2
        or output.shape[0] != 1
        or output.device.type != "cpu"
        or output.layout is not torch.strided
        or output.dtype is not torch.int64
    ):
        raise GVSDecoderError(
            "model.generate must return one dense rank-two CPU int64 token tensor"
        )
    all_ids = tuple(int(value) for value in output[0].tolist())
    if all_ids[: len(prompt_token_ids)] != prompt_token_ids:
        raise GVSDecoderError("model.generate changed the prompt prefix")
    generated = all_ids[len(prompt_token_ids) :]
    _strict_token_tuple(generated, label="greedy generated tokens", vocab_size=vocab_size)
    if len(generated) > max_new_tokens:
        raise GVSDecoderError("greedy generation exceeded max_new_tokens")
    return generated


def _native_completion_error(
    generated_token_ids: tuple[int, ...], *, eos_token_id: int, max_new_tokens: int
) -> Exception | None:
    eos_positions = [
        index for index, token_id in enumerate(generated_token_ids) if token_id == eos_token_id
    ]
    if eos_positions:
        if eos_positions[0] != len(generated_token_ids) - 1:
            return NativeGenerationAfterEOSError(
                "native rank zero returned tokens after its first EOS"
            )
        return None
    if len(generated_token_ids) != max_new_tokens:
        return IncompleteNativeGenerationError(
            "native rank zero stopped without EOS before max_new_tokens"
        )
    return None


def generate_gvs_trace(
    model: Any,
    tokenizer: TokenizerLike,
    *,
    now: str,
    tools: Sequence[ToolDefinition],
    user_text: str,
    model_sha256: str,
    tokenizer_sha256: str,
    source_sha256: str,
    expected_runtime_sha256: str,
    contract: GVSDecoderContract | None = None,
) -> CandidateSetTrace:
    """Generate one complete, non-authorizing reference trace with eight retained slots."""

    contract = GVSDecoderContract() if contract is None else contract
    if type(contract) is not GVSDecoderContract:
        raise GVSDecoderError("contract must be GVSDecoderContract")
    if type(model) is BarunLM:
        _clear_barun_model_execution_caches(model)
    model_sha256 = _strict_sha256(model_sha256, label="model_sha256")
    tokenizer_sha256 = _strict_sha256(tokenizer_sha256, label="tokenizer_sha256")
    source_sha256 = _strict_sha256(source_sha256, label="source_sha256")
    expected_runtime_sha256 = _strict_sha256(
        expected_runtime_sha256, label="expected_runtime_sha256"
    )
    runtime = inspect_gvs_runtime(model, tokenizer)
    if runtime.sha256 != expected_runtime_sha256:
        raise GVSDecoderError(
            "actual model/tokenizer/device/dtype runtime disagrees with expectation"
        )
    if model_sha256 != runtime.model_state_sha256:
        raise GVSDecoderError("model_sha256 must equal the actual runtime state hash")
    if tokenizer_sha256 != runtime.tokenizer_state_sha256:
        raise GVSDecoderError("tokenizer_sha256 must equal the actual tokenizer state hash")
    vocab_size, max_seq_len = _model_dimensions(model)
    if vocab_size < 7:
        raise GVSDecoderError("model vocabulary is too small for beam width seven")
    prompt = render_prompt(now=now, tools=tools, user_text=user_text)
    _strict_string(prompt, label="prompt", maximum=_MAX_PROMPT_UTF8_BYTES)
    prompt_sha256 = _sha256_text(prompt)
    if source_sha256 != prompt_sha256:
        raise GVSDecoderError("source_sha256 must equal the exact rendered prompt hash")
    try:
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_token_ids = tuple(encoded.ids)
    except (AttributeError, TypeError, ValueError) as error:
        raise GVSDecoderError("tokenizer failed strict prompt encoding") from error
    prompt_token_ids = _strict_token_tuple(
        prompt_token_ids, label="prompt_token_ids", vocab_size=vocab_size
    )
    try:
        eos_token_id = tokenizer.token_to_id("<eos>")
        pad_token_id = tokenizer.token_to_id("<pad>")
    except (AttributeError, TypeError, ValueError) as error:
        raise GVSDecoderError("tokenizer failed special-token lookup") from error
    for label, token_id in (("eos_token_id", eos_token_id), ("pad_token_id", pad_token_id)):
        if type(token_id) is not int or not 0 <= token_id < vocab_size:
            raise GVSDecoderError(f"tokenizer {label} must be within model vocabulary")
    if eos_token_id == pad_token_id:
        raise GVSDecoderError("tokenizer must define distinct EOS and pad tokens")
    if len(prompt_token_ids) + contract.max_new_tokens > max_seq_len:
        raise GVSDecoderError("prompt plus max_new_tokens exceeds model context length")

    input_ids = _cpu_long_tensor([prompt_token_ids], label="native input_ids")
    training_modes = _snapshot_model_training_modes(model)
    greedy_error: Exception | None = None
    greedy_completion_error: Exception | None = None
    beam_error: Exception | None = None
    greedy_tokens: tuple[int, ...] = ()
    beams: tuple[_Beam, ...] = ()
    try:
        _set_model_training_modes(training_modes, False)
        with _TORCH_INFERENCE_MODE():
            try:
                greedy_tokens = _native_greedy_tokens(
                    model,
                    input_ids,
                    prompt_token_ids=prompt_token_ids,
                    max_new_tokens=contract.max_new_tokens,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    vocab_size=vocab_size,
                )
                greedy_completion_error = _native_completion_error(
                    greedy_tokens,
                    eos_token_id=eos_token_id,
                    max_new_tokens=contract.max_new_tokens,
                )
            except _RETAINABLE_EXCEPTIONS as error:
                greedy_error = error
            _require_model_training_mode(training_modes, False)
            try:
                beams = _ordinary_beam(
                    model,
                    prompt_token_ids=prompt_token_ids,
                    eos_token_id=eos_token_id,
                    vocab_size=vocab_size,
                    max_new_tokens=contract.max_new_tokens,
                )
            except _RETAINABLE_EXCEPTIONS as error:
                beam_error = error
            _require_model_training_mode(training_modes, False)
    finally:
        _restore_model_training_modes(training_modes)

    if isinstance(greedy_completion_error, NativeGenerationAfterEOSError):
        eos_position = greedy_tokens.index(eos_token_id)
        greedy = _failed_slot(
            attempted_rank=0,
            origin="greedy",
            beam_rank=None,
            stage="native_generate",
            error=greedy_completion_error,
            generated_token_ids=greedy_tokens,
            content_token_ids=greedy_tokens[:eos_position],
            eos_emitted=True,
            eos_position=eos_position,
        )
    elif greedy_error is None:
        greedy = _materialize_candidate(
            model,
            tokenizer,
            attempted_rank=0,
            origin="greedy",
            beam_rank=None,
            generated_token_ids=greedy_tokens,
            beam_search_raw_log_likelihood=None,
            prompt_token_ids=prompt_token_ids,
            eos_token_id=eos_token_id,
            max_new_tokens=contract.max_new_tokens,
            retained_failure=(
                None
                if greedy_completion_error is None
                else ("native_generate", greedy_completion_error)
            ),
        )
    else:
        greedy = _failed_slot(
            attempted_rank=0,
            origin="greedy",
            beam_rank=None,
            stage="native_generate",
            error=greedy_error,
        )

    beam_candidates: list[CandidateTrace] = []
    if beam_error is not None:
        for beam_rank in range(7):
            beam_candidates.append(
                _failed_slot(
                    attempted_rank=beam_rank + 1,
                    origin="beam",
                    beam_rank=beam_rank,
                    stage="beam_search",
                    error=beam_error,
                )
            )
    else:
        rescored: list[tuple[_Beam, LikelihoodTrace]] = []
        rescore_failures: list[tuple[_Beam, Exception]] = []
        for beam in beams:
            try:
                rescored.append(
                    (
                        beam,
                        recompute_likelihood(
                            model,
                            prompt_token_ids=prompt_token_ids,
                            generated_token_ids=beam.token_ids,
                            filler_mask=(False,) * len(beam.token_ids),
                        ),
                    )
                )
            except _RETAINABLE_EXCEPTIONS as error:
                rescore_failures.append((beam, error))
        rescored.sort(
            key=lambda row: (
                -row[1].normalized_log_likelihood,
                -row[1].raw_log_likelihood,
                row[0].rank,
            )
        )
        rescore_failures.sort(key=lambda row: row[0].rank)
        ordered_beams: list[tuple[_Beam, LikelihoodTrace | None, Exception | None]] = [
            (beam, likelihood, None) for beam, likelihood in rescored
        ]
        ordered_beams.extend((beam, None, error) for beam, error in rescore_failures)
        materialized_candidates: list[CandidateTrace] = []
        for provisional_rank, (beam, likelihood, error) in enumerate(ordered_beams, start=1):
            materialized_candidates.append(
                _materialize_candidate(
                    model,
                    tokenizer,
                    attempted_rank=provisional_rank,
                    origin="beam",
                    beam_rank=beam.rank,
                    generated_token_ids=beam.token_ids,
                    beam_search_raw_log_likelihood=beam.search_score,
                    prompt_token_ids=prompt_token_ids,
                    eos_token_id=eos_token_id,
                    max_new_tokens=contract.max_new_tokens,
                    precomputed_likelihood=likelihood,
                    forced_likelihood_error=error,
                )
            )
        successful_candidates = [
            candidate for candidate in materialized_candidates if candidate.failure_stage is None
        ]
        failed_candidates = sorted(
            (
                candidate
                for candidate in materialized_candidates
                if candidate.failure_stage is not None
            ),
            key=lambda candidate: int(candidate.beam_rank),
        )
        beam_candidates = [
            _DATACLASSES_REPLACE(candidate, attempted_rank=attempted_rank)
            for attempted_rank, candidate in enumerate(
                (*successful_candidates, *failed_candidates),
                start=1,
            )
        ]

    candidates = (greedy, *beam_candidates)
    if type(model) is BarunLM:
        _clear_barun_model_execution_caches(model)
    runtime_after = inspect_gvs_runtime(model, tokenizer)
    if runtime_after != runtime:
        raise GVSDecoderError("model/tokenizer/runtime identity mutated during reference decoding")

    return CandidateSetTrace(
        schema_version=GVS_TRACE_SCHEMA_VERSION,
        contract=contract,
        contract_sha256=contract.sha256,
        runtime=runtime,
        runtime_sha256=runtime.sha256,
        model_sha256=model_sha256,
        tokenizer_sha256=tokenizer_sha256,
        source_sha256=source_sha256,
        prompt_contract_version=PROMPT_CONTRACT_VERSION,
        prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
        prompt=prompt,
        prompt_sha256=prompt_sha256,
        prompt_token_ids=prompt_token_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        vocab_size=vocab_size,
        model_max_seq_len=max_seq_len,
        candidates=candidates,
    )


def _value_schema_record(schema: ValueSchema) -> dict[str, object]:
    if not isinstance(schema, ValueSchema):
        raise GVSDecoderError("schema value must be ValueSchema")
    return {
        "additional_properties": schema.additional_properties,
        "enum": None if schema.enum is None else list(schema.enum),
        "items": None if schema.items is None else _value_schema_record(schema.items),
        "kind": schema.kind.value,
        "properties": {
            name: _value_schema_record(child) for name, child in sorted(schema.properties.items())
        },
        "required": sorted(schema.required),
        "set_semantics": schema.set_semantics,
    }


def _schema_canonical_json(
    schemas: Mapping[str, ToolSchema] | Iterable[ToolSchema],
) -> tuple[str, Mapping[str, ToolSchema] | tuple[ToolSchema, ...]]:
    if isinstance(schemas, Mapping):
        snapshot: Mapping[str, ToolSchema] | tuple[ToolSchema, ...] = dict(schemas)
        rows = list(snapshot.values())
        for key, schema in snapshot.items():
            if not isinstance(schema, ToolSchema) or key != schema.name:
                raise GVSDecoderError("schema mapping key/value identity is invalid")
    else:
        snapshot = tuple(schemas)
        rows = list(snapshot)
    if any(not isinstance(schema, ToolSchema) for schema in rows):
        raise GVSDecoderError("all schemas must be ToolSchema")
    if len({schema.name for schema in rows}) != len(rows):
        raise GVSDecoderError("tool schemas must have unique names")
    record = [
        {
            "additional_arguments": schema.additional_arguments,
            "arguments": {
                name: _value_schema_record(value)
                for name, value in sorted(schema.arguments.items())
            },
            "name": schema.name,
            "required": sorted(schema.required),
            "side_effecting": schema.side_effecting,
        }
        for schema in sorted(rows, key=lambda value: value.name)
    ]
    return _canonical_json(record), snapshot


def analyze_gvs_trace(
    trace: CandidateSetTrace,
    schemas: Mapping[str, ToolSchema] | Iterable[ToolSchema],
) -> CandidateSetAnalysis:
    """Analyze all slots under an explicitly hashed evaluator/schema snapshot."""

    if type(trace) is not CandidateSetTrace:
        raise GVSDecoderError("trace must be CandidateSetTrace")
    evaluator_path = Path(inspect.getsourcefile(action_ir_module) or "")
    evaluator_source_sha256 = _file_sha256(evaluator_path)
    evaluator_runtime_sha256 = _evaluator_runtime_sha256()
    schema_json, schema_snapshot = _schema_canonical_json(schemas)
    analyses: list[CandidateAnalysis] = []
    for candidate in trace.candidates:
        parse_valid = False
        schema_valid = False
        canonical: str | None = None
        canonical_sha256: str | None = None
        error_stage: str | None = None
        error_code: str | None = None
        error_path: str | None = None
        if candidate.failure_stage is not None:
            error_stage = "generation"
            error_code = candidate.failure_code
            error_path = "$"
        else:
            if candidate.raw_output is None:
                raise GVSDecoderError("successful candidate lost its decoded output")
            try:
                action = parse_action_ir(candidate.raw_output, schema_snapshot)
            except ActionIRParseError as error:
                error_stage = "parse"
                error_code = error.code
                error_path = error.path
            except ActionIRValidationError as error:
                parse_valid = True
                error_stage = "schema"
                error_code = error.code
                error_path = error.path
            else:
                parse_valid = True
                schema_valid = True
                canonical = action.canonical_json()
                canonical_sha256 = _sha256_text(canonical)
        likelihood = candidate.likelihood
        analyses.append(
            CandidateAnalysis(
                attempted_rank=candidate.attempted_rank,
                candidate_id=candidate.candidate_id,
                candidate_trace_sha256=candidate.sha256,
                raw_output_sha256=candidate.raw_output_sha256,
                output_present=candidate.output_present,
                generation_failed=candidate.failure_stage is not None,
                truncated=candidate.truncated,
                parse_valid=parse_valid,
                schema_valid=schema_valid,
                canonical_action_json=canonical,
                canonical_action_sha256=canonical_sha256,
                error_stage=error_stage,
                error_code=error_code,
                error_path=error_path,
                normalized_log_likelihood=(
                    None if likelihood is None else likelihood.normalized_log_likelihood
                ),
                raw_log_likelihood=None if likelihood is None else likelihood.raw_log_likelihood,
            )
        )
    if _file_sha256(evaluator_path) != evaluator_source_sha256:
        raise GVSDecoderError("Action IR evaluator source mutated during analysis")
    if _evaluator_runtime_sha256() != evaluator_runtime_sha256:
        raise GVSDecoderError("Action IR evaluator callables mutated during analysis")
    return CandidateSetAnalysis(
        schema_version=GVS_ANALYSIS_SCHEMA_VERSION,
        candidate_set_trace_sha256=trace.sha256,
        schema_canonical_json=schema_json,
        schema_sha256=_sha256_text(schema_json),
        evaluator_source_sha256=evaluator_source_sha256,
        evaluator_runtime_sha256=evaluator_runtime_sha256,
        candidates=tuple(analyses),
    )


def select_schema_valid_by_likelihood(analysis: CandidateSetAnalysis) -> LikelihoodSelection:
    if type(analysis) is not CandidateSetAnalysis:
        raise GVSDecoderError("analysis must be CandidateSetAnalysis")
    eligible = [candidate for candidate in analysis.candidates if candidate.selection_eligible]
    if not eligible:
        raise GVSDecoderError("no complete schema-valid candidate is eligible for selection")
    eligible.sort(
        key=lambda candidate: (
            -float(candidate.normalized_log_likelihood),
            -float(candidate.raw_log_likelihood),
            candidate.attempted_rank,
        )
    )
    selected = eligible[0]
    if selected.canonical_action_sha256 is None:
        raise GVSDecoderError("eligible candidate lost its canonical Action IR identity")
    return LikelihoodSelection(
        schema_version=GVS_SELECTION_SCHEMA_VERSION,
        analysis_sha256=analysis.sha256,
        selected_rank=selected.attempted_rank,
        selected_candidate_id=selected.candidate_id,
        selected_candidate_trace_sha256=selected.candidate_trace_sha256,
        selected_canonical_action_sha256=selected.canonical_action_sha256,
        eligible_ranks_in_order=tuple(candidate.attempted_rank for candidate in eligible),
    )


def _parse_contract(value: object) -> GVSDecoderContract:
    expected_fields = frozenset(GVSDecoderContract().to_record())
    row = _exact_keys(value, expected_fields, label="decoder contract")
    values = dict(row)
    for name in ("expansion_tie_break", "final_tie_break"):
        raw = values[name]
        if type(raw) is not list or any(type(item) is not str for item in raw):
            raise GVSDecoderError(f"contract {name} must be a string array")
        values[name] = tuple(raw)
    return GVSDecoderContract(**values)


_RUNTIME_FIELDS = frozenset(
    {
        "adapter_inventory",
        "adapters_enabled",
        "buffer_dtypes",
        "decoder_callable_sha256",
        "decoder_source_sha256",
        "default_dtype",
        "deterministic_algorithms_enabled",
        "device_type",
        "generation_settings_sha256",
        "model_class",
        "model_callable_sha256",
        "model_class_source_sha256",
        "model_contract",
        "model_dependency_sha256",
        "model_execution_tensor_sha256",
        "model_module_source_sha256",
        "model_instance_state_sha256",
        "model_module_graph_sha256",
        "model_topology_sha256",
        "model_config_sha256",
        "model_state_sha256",
        "optimized_remote_exact_match_required",
        "parameter_dtypes",
        "platform_machine",
        "platform_system",
        "python_implementation",
        "python_version",
        "production_identity_contract_satisfied",
        "reference_impl_only",
        "schema_version",
        "tokenizer_class",
        "tokenizer_behavior_sha256",
        "tokenizer_callable_sha256",
        "tokenizer_contract",
        "tokenizer_instance_state_sha256",
        "tokenizer_state_sha256",
        "tokenizers_implementation_sha256",
        "tokenizers_version",
        "torch_num_threads",
        "torch_version",
        "torch_default_device",
        "torch_execution_state_sha256",
    }
)


def _parse_runtime(value: object) -> GVSRuntimeIdentity:
    row = _exact_keys(value, _RUNTIME_FIELDS, label="runtime identity")
    dtypes: dict[str, tuple[str, ...]] = {}
    for name in ("parameter_dtypes", "buffer_dtypes"):
        raw = row[name]
        if type(raw) is not list or any(type(item) is not str for item in raw):
            raise GVSDecoderError(f"runtime {name} must be a string array")
        dtypes[name] = tuple(raw)
    raw_adapter_inventory = row["adapter_inventory"]
    if type(raw_adapter_inventory) is not list or any(
        type(item) is not str for item in raw_adapter_inventory
    ):
        raise GVSDecoderError("runtime adapter_inventory must be a string array")
    return GVSRuntimeIdentity(
        schema_version=row["schema_version"],
        python_implementation=row["python_implementation"],
        python_version=row["python_version"],
        torch_version=row["torch_version"],
        tokenizers_version=row["tokenizers_version"],
        tokenizers_implementation_sha256=row["tokenizers_implementation_sha256"],
        platform_system=row["platform_system"],
        platform_machine=row["platform_machine"],
        device_type=row["device_type"],
        default_dtype=row["default_dtype"],
        torch_default_device=row["torch_default_device"],
        parameter_dtypes=dtypes["parameter_dtypes"],
        buffer_dtypes=dtypes["buffer_dtypes"],
        deterministic_algorithms_enabled=row["deterministic_algorithms_enabled"],
        torch_num_threads=row["torch_num_threads"],
        model_class=row["model_class"],
        model_class_source_sha256=row["model_class_source_sha256"],
        model_module_source_sha256=row["model_module_source_sha256"],
        model_config_sha256=row["model_config_sha256"],
        model_state_sha256=row["model_state_sha256"],
        model_contract=row["model_contract"],
        model_callable_sha256=row["model_callable_sha256"],
        model_dependency_sha256=row["model_dependency_sha256"],
        model_execution_tensor_sha256=row["model_execution_tensor_sha256"],
        model_instance_state_sha256=row["model_instance_state_sha256"],
        model_module_graph_sha256=row["model_module_graph_sha256"],
        model_topology_sha256=row["model_topology_sha256"],
        tokenizer_class=row["tokenizer_class"],
        tokenizer_state_sha256=row["tokenizer_state_sha256"],
        tokenizer_contract=row["tokenizer_contract"],
        tokenizer_behavior_sha256=row["tokenizer_behavior_sha256"],
        tokenizer_callable_sha256=row["tokenizer_callable_sha256"],
        tokenizer_instance_state_sha256=row["tokenizer_instance_state_sha256"],
        generation_settings_sha256=row["generation_settings_sha256"],
        torch_execution_state_sha256=row["torch_execution_state_sha256"],
        adapter_inventory=tuple(raw_adapter_inventory),
        adapters_enabled=row["adapters_enabled"],
        production_identity_contract_satisfied=row["production_identity_contract_satisfied"],
        decoder_source_sha256=row["decoder_source_sha256"],
        decoder_callable_sha256=row["decoder_callable_sha256"],
        reference_impl_only=row["reference_impl_only"],
        optimized_remote_exact_match_required=row["optimized_remote_exact_match_required"],
    )


_LIKELIHOOD_FIELDS = frozenset(
    {
        "filler_mask",
        "generated_token_ids",
        "normalized_log_likelihood",
        "raw_log_likelihood",
        "score_included",
        "scored_token_count",
        "token_log_likelihoods",
    }
)


def _exact_array(value: object, *, label: str) -> list[Any]:
    if type(value) is not list:
        raise GVSDecoderError(f"{label} must be an exact JSON array")
    return value


def _parse_likelihood(value: object) -> LikelihoodTrace:
    row = _exact_keys(value, _LIKELIHOOD_FIELDS, label="likelihood trace")
    return LikelihoodTrace(
        generated_token_ids=tuple(
            _exact_array(row["generated_token_ids"], label="generated_token_ids")
        ),
        token_log_likelihoods=tuple(
            _exact_array(row["token_log_likelihoods"], label="token_log_likelihoods")
        ),
        filler_mask=tuple(_exact_array(row["filler_mask"], label="filler_mask")),
        score_included=tuple(_exact_array(row["score_included"], label="score_included")),
        raw_log_likelihood=row["raw_log_likelihood"],
        normalized_log_likelihood=row["normalized_log_likelihood"],
        scored_token_count=row["scored_token_count"],
    )


_CANDIDATE_FIELDS = frozenset(
    {
        "attempted_rank",
        "beam_rank",
        "beam_search_raw_log_likelihood",
        "candidate_id",
        "candidate_trace_sha256",
        "content_token_ids",
        "eos_emitted",
        "eos_position",
        "failure_code",
        "failure_stage",
        "generated_token_ids",
        "likelihood",
        "origin",
        "output_present",
        "raw_output",
        "raw_output_sha256",
        "truncated",
    }
)


def _parse_candidate(value: object) -> CandidateTrace:
    row = _exact_keys(value, _CANDIDATE_FIELDS, label="candidate trace")
    likelihood = None if row["likelihood"] is None else _parse_likelihood(row["likelihood"])
    candidate = CandidateTrace(
        attempted_rank=row["attempted_rank"],
        origin=row["origin"],
        beam_rank=row["beam_rank"],
        generated_token_ids=tuple(
            _exact_array(row["generated_token_ids"], label="generated_token_ids")
        ),
        content_token_ids=tuple(_exact_array(row["content_token_ids"], label="content_token_ids")),
        output_present=row["output_present"],
        raw_output=row["raw_output"],
        raw_output_sha256=row["raw_output_sha256"],
        eos_emitted=row["eos_emitted"],
        eos_position=row["eos_position"],
        truncated=row["truncated"],
        likelihood=likelihood,
        beam_search_raw_log_likelihood=row["beam_search_raw_log_likelihood"],
        failure_stage=row["failure_stage"],
        failure_code=row["failure_code"],
    )
    if row["candidate_trace_sha256"] != candidate.sha256:
        raise GVSDecoderError("candidate trace hash failed recomputation")
    if row["candidate_id"] != candidate.candidate_id:
        raise GVSDecoderError("candidate ID failed exact derivation")
    return candidate


_TRACE_FIELDS = frozenset(
    {
        "candidates",
        "contract",
        "contract_sha256",
        "eos_token_id",
        "model_max_seq_len",
        "model_sha256",
        "pad_token_id",
        "prompt",
        "prompt_contract_version",
        "prompt_sha256",
        "prompt_template_sha256",
        "prompt_token_ids",
        "runtime",
        "runtime_sha256",
        "schema_version",
        "source_sha256",
        "tokenizer_sha256",
        "vocab_size",
    }
)


def parse_candidate_set_trace_json(
    payload: str | bytes,
    *,
    expected_trace_sha256: str,
    expected_runtime_sha256: str,
) -> CandidateSetTrace:
    """Strictly reconstruct and hash-check one bounded serialized trace artifact."""

    expected_trace_sha256 = _strict_sha256(expected_trace_sha256, label="expected_trace_sha256")
    expected_runtime_sha256 = _strict_sha256(
        expected_runtime_sha256, label="expected_runtime_sha256"
    )
    artifact = _exact_keys(
        _decode_bounded_json(payload, maximum=_MAX_TRACE_BYTES, label="trace artifact"),
        frozenset({"artifact_schema_version", "candidate_set_trace_sha256", "trace"}),
        label="trace artifact",
    )
    if artifact["artifact_schema_version"] != GVS_TRACE_ARTIFACT_SCHEMA_VERSION:
        raise GVSDecoderError("unsupported trace artifact schema")
    row = _exact_keys(artifact["trace"], _TRACE_FIELDS, label="trace")
    runtime = _parse_runtime(row["runtime"])
    if row["runtime_sha256"] != runtime.sha256 or runtime.sha256 != expected_runtime_sha256:
        raise GVSDecoderError("serialized runtime identity failed exact binding")
    if runtime.decoder_source_sha256 != _file_sha256(Path(__file__)):
        raise GVSDecoderError("serialized trace was not produced by the current decoder source")
    if runtime.decoder_callable_sha256 != _decoder_callable_sha256():
        raise GVSDecoderError("serialized trace was not produced by the loaded decoder callables")
    tokenizers_implementation_path = Path(tokenizers_native.__file__ or "")
    if runtime.tokenizers_implementation_sha256 != _file_sha256(tokenizers_implementation_path):
        raise GVSDecoderError(
            "serialized trace was not produced by the current Tokenizers implementation"
        )
    if runtime.model_contract == "exact_barunlm_v1":
        if runtime.model_class != f"{BarunLM.__module__}.{BarunLM.__qualname__}":
            raise GVSDecoderError("serialized production trace has the wrong model class")
        if runtime.model_dependency_sha256 != _model_dependency_sha256():
            raise GVSDecoderError("serialized production trace has stale model dependencies")
    if runtime.tokenizer_contract == "exact_tokenizers_reconstructible_v1" and (
        runtime.tokenizer_class
        != f"{tokenizers.Tokenizer.__module__}.{tokenizers.Tokenizer.__qualname__}"
    ):
        raise GVSDecoderError("serialized production trace has the wrong Tokenizer class")
    candidates = tuple(
        _parse_candidate(value)
        for value in _exact_array(row["candidates"], label="trace candidates")
    )
    trace = CandidateSetTrace(
        schema_version=row["schema_version"],
        contract=_parse_contract(row["contract"]),
        contract_sha256=row["contract_sha256"],
        runtime=runtime,
        runtime_sha256=row["runtime_sha256"],
        model_sha256=row["model_sha256"],
        tokenizer_sha256=row["tokenizer_sha256"],
        source_sha256=row["source_sha256"],
        prompt_contract_version=row["prompt_contract_version"],
        prompt_template_sha256=row["prompt_template_sha256"],
        prompt=row["prompt"],
        prompt_sha256=row["prompt_sha256"],
        prompt_token_ids=tuple(_exact_array(row["prompt_token_ids"], label="prompt_token_ids")),
        eos_token_id=row["eos_token_id"],
        pad_token_id=row["pad_token_id"],
        vocab_size=row["vocab_size"],
        model_max_seq_len=row["model_max_seq_len"],
        candidates=candidates,
    )
    if artifact["candidate_set_trace_sha256"] != trace.sha256:
        raise GVSDecoderError("embedded candidate-set trace hash failed recomputation")
    if trace.sha256 != expected_trace_sha256:
        raise GVSDecoderError("candidate-set trace differs from caller commitment")
    return trace


def parse_candidate_set_analysis_json(
    payload: str | bytes,
    *,
    trace: CandidateSetTrace,
    schemas: Mapping[str, ToolSchema] | Iterable[ToolSchema],
    expected_analysis_sha256: str,
) -> CandidateSetAnalysis:
    """Recompute analysis from trace/schema and require byte-exact serialized evidence."""

    expected_analysis_sha256 = _strict_sha256(
        expected_analysis_sha256, label="expected_analysis_sha256"
    )
    artifact = _exact_keys(
        _decode_bounded_json(payload, maximum=_MAX_ANALYSIS_BYTES, label="analysis artifact"),
        frozenset({"analysis", "analysis_sha256", "artifact_schema_version"}),
        label="analysis artifact",
    )
    if artifact["artifact_schema_version"] != GVS_ANALYSIS_ARTIFACT_SCHEMA_VERSION:
        raise GVSDecoderError("unsupported analysis artifact schema")
    recomputed = analyze_gvs_trace(trace, schemas)
    if _canonical_json(artifact["analysis"]) != _canonical_json(recomputed.to_record()):
        raise GVSDecoderError("serialized analysis differs from evaluator recomputation")
    if artifact["analysis_sha256"] != recomputed.sha256:
        raise GVSDecoderError("embedded analysis hash failed recomputation")
    if recomputed.sha256 != expected_analysis_sha256:
        raise GVSDecoderError("analysis differs from caller commitment")
    return recomputed


def parse_likelihood_selection_json(
    payload: str | bytes,
    *,
    analysis: CandidateSetAnalysis,
    expected_selection_sha256: str,
) -> LikelihoodSelection:
    """Recompute likelihood selection and require an exact bounded artifact."""

    expected_selection_sha256 = _strict_sha256(
        expected_selection_sha256, label="expected_selection_sha256"
    )
    artifact = _exact_keys(
        _decode_bounded_json(payload, maximum=_MAX_SELECTION_BYTES, label="selection artifact"),
        frozenset({"artifact_schema_version", "selection", "selection_sha256"}),
        label="selection artifact",
    )
    if artifact["artifact_schema_version"] != GVS_SELECTION_ARTIFACT_SCHEMA_VERSION:
        raise GVSDecoderError("unsupported selection artifact schema")
    recomputed = select_schema_valid_by_likelihood(analysis)
    if _canonical_json(artifact["selection"]) != _canonical_json(recomputed.to_record()):
        raise GVSDecoderError("serialized selection differs from analysis recomputation")
    if artifact["selection_sha256"] != recomputed.sha256:
        raise GVSDecoderError("embedded selection hash failed recomputation")
    if recomputed.sha256 != expected_selection_sha256:
        raise GVSDecoderError("selection differs from caller commitment")
    return recomputed


__all__ = [
    "FROZEN_BEAM_WIDTH",
    "FROZEN_CANDIDATE_COUNT",
    "GVS_ANALYSIS_SCHEMA_VERSION",
    "GVS_DECODER_SCHEMA_VERSION",
    "GVS_RUNTIME_SCHEMA_VERSION",
    "GVS_SELECTION_SCHEMA_VERSION",
    "GVS_TRACE_SCHEMA_VERSION",
    "PROPOSED_MAX_NEW_TOKENS",
    "CandidateAnalysis",
    "CandidateSetAnalysis",
    "CandidateSetTrace",
    "CandidateTrace",
    "GVSDecoderContract",
    "GVSDecoderError",
    "GVSRuntimeIdentity",
    "LikelihoodSelection",
    "LikelihoodTrace",
    "analyze_gvs_trace",
    "generate_gvs_trace",
    "inspect_gvs_runtime",
    "parse_candidate_set_analysis_json",
    "parse_candidate_set_trace_json",
    "parse_likelihood_selection_json",
    "recompute_likelihood",
    "select_schema_valid_by_likelihood",
]
