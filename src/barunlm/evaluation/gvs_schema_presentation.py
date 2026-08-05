"""Strict renamed-schema presentation and lowering for BarunAction GVS-v1.

The model may be shown a frozen presentation in which tool and top-level argument
names differ from the simulator's canonical names.  This module proves that the
presentation is a total bijective *name-only* transformation, builds the exact
model-facing validation schemas, and lowers a schema-valid presented Action IR back
to canonical names without changing decisions, modes, ordering, or argument values.

Descriptions remain model-visible text and are hash-bound, but this module does not
claim they are shortcut-free or semantically audited.  It does not authenticate
population custody and never authorizes a model, labels, CUDA, training, or Jarvis.
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
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

from barunlm.datasets import mobile_actions as mobile_actions_module
from barunlm.datasets.mobile_actions import ToolArgument, ToolDefinition

from . import action_ir as action_ir_module
from .action_ir import (
    ActionIR,
    ActionIRError,
    CallMode,
    CanonicalJSONValue,
    Decision,
    JSONScalar,
    JSONType,
    ToolSchema,
    ValueSchema,
    action_ir_equal,
    parse_action_ir,
    validate_action_ir,
)

GVS_SCHEMA_PRESENTATION_VERSION = "barun-gvs-schema-presentation-v2"
GVS_SCHEMA_PRESENTATION_RECEIPT_VERSION = "barun-gvs-schema-presentation-receipt-v2"
GVS_SCHEMA_PRESENTATION_ARTIFACT_VERSION = "barun-gvs-schema-presentation-artifact-v1"

IDENTITY_PRESENTATION = "identity"
RENAMED_PRESENTATION = "renamed"
SUPPORTED_PRESENTATION_MODES = frozenset({IDENTITY_PRESENTATION, RENAMED_PRESENTATION})

_IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_DESCRIPTION_LENGTH = 2_048
_MAX_DESCRIPTION_UTF8_BYTES = 8_192
_MAX_SCHEMA_SCALAR_UTF8_BYTES = 64 * 1_024
_MAX_TOOLS = 256
_MAX_ARGUMENTS_PER_TOOL = 256
_MAX_PROPERTIES_PER_NODE = 256
_MAX_ENUM_VALUES = 1_024
_MAX_SCHEMA_NODES_PER_TOOL = 4_096
_MAX_ARTIFACT_BYTES = 4 * 1_024 * 1_024
_MAX_JSON_INTEGER_DIGITS = 20
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))


class GVSSchemaPresentationError(ValueError):
    """A schema presentation is malformed, lossy, or internally inconsistent."""


def _file_sha256(path: Path) -> str:
    try:
        before = path.stat()
        if not path.is_file() or before.st_size > 8 * 1_024 * 1_024:
            raise GVSSchemaPresentationError(f"source path {path} is absent or oversized")
        payload = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise GVSSchemaPresentationError(f"source path {path} is not readable") from error
    if (
        len(payload) != before.st_size
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise GVSSchemaPresentationError(f"source path {path} changed while hashing")
    return hashlib.sha256(payload).hexdigest()


def _module_path(module: object, *, label: str) -> Path:
    raw = getattr(module, "__file__", None)
    if type(raw) is not str:
        raise GVSSchemaPresentationError(f"{label} has no exact source path")
    return Path(raw)


def _python_code_sha256(code: object) -> str:
    """Hash exposed code fields without CPython's mutable quickening cache."""

    if not inspect.iscode(code):
        raise GVSSchemaPresentationError("runtime identity requires a Python code object")

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
        raise GVSSchemaPresentationError(
            f"runtime callable contains unsupported constant {type(value).__name__}"
        )

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

    return _sha256_json(code_record(code))


def _behavior_state(value: object, *, label: str, depth: int = 0) -> object:
    if depth > 16:
        raise GVSSchemaPresentationError(f"{label} behavior state exceeds the depth bound")
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, Enum):
        return {
            "enum_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _behavior_state(value.value, label=label, depth=depth + 1),
        }
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _behavior_state(
                    getattr(value, field.name),
                    label=f"{label}.{field.name}",
                    depth=depth + 1,
                )
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(value, tuple | list):
        return [_behavior_state(child, label=f"{label}[]", depth=depth + 1) for child in value]
    if isinstance(value, frozenset | set):
        children = [_behavior_state(child, label=f"{label}[]", depth=depth + 1) for child in value]
        return sorted(children, key=_canonical_json)
    if isinstance(value, Mapping):
        rows: dict[str, object] = {}
        for key, child in value.items():
            if type(key) is not str or key in rows:
                raise GVSSchemaPresentationError(f"{label} mapping state has an invalid key")
            rows[key] = _behavior_state(child, label=f"{label}.{key}", depth=depth + 1)
        return rows
    if inspect.isfunction(value) or inspect.ismethod(value):
        return {
            "callable": f"{value.__module__}.{value.__qualname__}",
            "code_sha256": _python_code_sha256(value.__code__),
        }
    raise GVSSchemaPresentationError(
        f"{label} has unsupported behavior state {type(value).__name__}"
    )


def _callable_sha256(value: object, *, label: str) -> str:
    if isinstance(value, property):
        value = value.fget
    elif isinstance(value, staticmethod | classmethod) or inspect.ismethod(value):
        value = value.__func__
    if not inspect.isfunction(value):
        raise GVSSchemaPresentationError(f"{label} is not an inspectable Python function")
    code = value.__code__
    try:
        code_sha256 = _python_code_sha256(code)
        defaults = _behavior_state(value.__defaults__, label=f"{label}.defaults")
        keyword_defaults = _behavior_state(
            value.__kwdefaults__,
            label=f"{label}.keyword_defaults",
        )
        closure = _behavior_state(
            None
            if value.__closure__ is None
            else tuple(cell.cell_contents for cell in value.__closure__),
            label=f"{label}.closure",
        )
    except (TypeError, ValueError, GVSSchemaPresentationError) as error:
        raise GVSSchemaPresentationError(f"{label} has unhashable behavior state") from error
    return _sha256_json(
        {
            "closure": closure,
            "code_sha256": code_sha256,
            "defaults": defaults,
            "keyword_defaults": keyword_defaults,
            "module": value.__module__,
            "qualname": value.__qualname__,
        }
    )


def _loaded_module_sha256(module: object, *, label: str) -> str:
    path = _module_path(module, label=label)
    namespace = getattr(module, "__dict__", None)
    if type(namespace) is not dict:
        raise GVSSchemaPresentationError(f"{label} has no exact runtime namespace")
    try:
        source = path.read_text(encoding="utf-8", errors="strict")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise GVSSchemaPresentationError(f"{label} source cannot be inspected") from error
    rows: list[dict[str, object]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            value = namespace.get(node.name)
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
                raise GVSSchemaPresentationError(f"{label}.{node.name} is not a live class")
            methods: list[dict[str, str]] = []
            for child in node.body:
                if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                raw = vars(cls).get(child.name)
                methods.append(
                    {
                        "name": child.name,
                        "sha256": _callable_sha256(
                            raw,
                            label=f"{label}.{node.name}.{child.name}",
                        ),
                    }
                )
            rows.append(
                {
                    "kind": "class",
                    "methods": methods,
                    "name": node.name,
                    "type": f"{cls.__module__}.{cls.__qualname__}",
                }
            )
    return _sha256_json({"source_sha256": _file_sha256(path), "symbols": rows})


def _source_files_record() -> dict[str, str]:
    return {
        "action_ir": _file_sha256(_module_path(action_ir_module, label="action_ir")),
        "gvs_schema_presentation": _file_sha256(Path(__file__)),
        "mobile_actions": _file_sha256(_module_path(mobile_actions_module, label="mobile_actions")),
    }


def _assert_import_aliases() -> None:
    expected = {
        "ActionIR": (ActionIR, action_ir_module.ActionIR),
        "ActionIRError": (ActionIRError, action_ir_module.ActionIRError),
        "CallMode": (CallMode, action_ir_module.CallMode),
        "Decision": (Decision, action_ir_module.Decision),
        "JSONType": (JSONType, action_ir_module.JSONType),
        "ToolArgument": (ToolArgument, mobile_actions_module.ToolArgument),
        "ToolDefinition": (ToolDefinition, mobile_actions_module.ToolDefinition),
        "ToolSchema": (ToolSchema, action_ir_module.ToolSchema),
        "ValueSchema": (ValueSchema, action_ir_module.ValueSchema),
        "action_ir_equal": (action_ir_equal, action_ir_module.action_ir_equal),
        "parse_action_ir": (parse_action_ir, action_ir_module.parse_action_ir),
        "validate_action_ir": (validate_action_ir, action_ir_module.validate_action_ir),
    }
    for name, (imported, live) in expected.items():
        if imported is not live:
            raise GVSSchemaPresentationError(f"imported runtime alias {name} changed identity")


def _runtime_identity_record() -> dict[str, object]:
    _assert_import_aliases()
    return {
        "bounds": {
            "artifact_bytes": _MAX_ARTIFACT_BYTES,
            "arguments_per_tool": _MAX_ARGUMENTS_PER_TOOL,
            "description_characters": _MAX_DESCRIPTION_LENGTH,
            "description_utf8_bytes": _MAX_DESCRIPTION_UTF8_BYTES,
            "enum_values": _MAX_ENUM_VALUES,
            "identifier_pattern": _IDENTIFIER_RE.pattern,
            "json_integer_digits": _MAX_JSON_INTEGER_DIGITS,
            "properties_per_node": _MAX_PROPERTIES_PER_NODE,
            "schema_nodes_per_tool": _MAX_SCHEMA_NODES_PER_TOOL,
            "schema_scalar_utf8_bytes": _MAX_SCHEMA_SCALAR_UTF8_BYTES,
            "tools": _MAX_TOOLS,
        },
        "contract": {
            "artifact_version": GVS_SCHEMA_PRESENTATION_ARTIFACT_VERSION,
            "identity_mode": IDENTITY_PRESENTATION,
            "presentation_version": GVS_SCHEMA_PRESENTATION_VERSION,
            "receipt_version": GVS_SCHEMA_PRESENTATION_RECEIPT_VERSION,
            "renamed_mode": RENAMED_PRESENTATION,
            "supported_modes": sorted(SUPPORTED_PRESENTATION_MODES),
        },
        "loaded_module_sha256s": {
            "action_ir": _loaded_module_sha256(action_ir_module, label="action_ir"),
            "gvs_schema_presentation": _loaded_module_sha256(
                sys.modules[__name__],
                label="gvs_schema_presentation",
            ),
            "mobile_actions": _loaded_module_sha256(
                mobile_actions_module,
                label="mobile_actions",
            ),
        },
        "python_cache_tag": sys.implementation.cache_tag,
        "python_implementation": platform.python_implementation(),
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "schema_version": "barun-gvs-schema-presentation-runtime-v1",
        "source_files_sha256": _sha256_json(_source_files_record()),
    }


def _runtime_identity_sha256() -> str:
    return _sha256_json(_runtime_identity_record())


def _assert_live_runtime() -> tuple[str, str]:
    source_sha256 = _sha256_json(_source_files_record())
    runtime_sha256 = _runtime_identity_sha256()
    if source_sha256 != _INITIAL_SOURCE_FILES_SHA256:
        raise GVSSchemaPresentationError(
            "schema-presentation source differs from the import-time baseline"
        )
    if runtime_sha256 != _INITIAL_RUNTIME_IDENTITY_SHA256:
        raise GVSSchemaPresentationError(
            "schema-presentation loaded runtime differs from the import-time baseline"
        )
    return source_sha256, runtime_sha256


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise GVSSchemaPresentationError("schema presentation is not strict JSON") from error


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_object(
    value: object,
    fields: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise GVSSchemaPresentationError(f"{label} must be an exact JSON object")
    missing = sorted(fields.difference(value))
    extra = sorted(set(value).difference(fields))
    if missing or extra:
        raise GVSSchemaPresentationError(
            f"{label} fields changed; missing={missing!r}, extra={extra!r}"
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GVSSchemaPresentationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise GVSSchemaPresentationError(f"non-finite JSON constant {value!r}")


def _bounded_json_integer(value: str) -> int:
    digits = value.removeprefix("-")
    if not digits or len(digits) > _MAX_JSON_INTEGER_DIGITS:
        raise GVSSchemaPresentationError("JSON integer exceeds the presentation bound")
    return int(value)


def _bounded_json_float(value: str) -> float:
    if len(value) > 64:
        raise GVSSchemaPresentationError("JSON float exceeds the presentation bound")
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as error:
        raise GVSSchemaPresentationError("JSON float is malformed") from error
    if not math.isfinite(parsed):
        raise GVSSchemaPresentationError("JSON float must be finite")
    return parsed


def _strict_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise GVSSchemaPresentationError(f"{label} must be a bounded ASCII identifier")
    return value


def _strict_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise GVSSchemaPresentationError(f"{label} must be lowercase SHA-256")
    return value


def _strict_description(value: object, *, label: str) -> str:
    if type(value) is not str or not value or len(value) > _MAX_DESCRIPTION_LENGTH:
        raise GVSSchemaPresentationError(f"{label} must be nonempty and bounded")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise GVSSchemaPresentationError(f"{label} is not valid UTF-8") from error
    if len(encoded) > _MAX_DESCRIPTION_UTF8_BYTES:
        raise GVSSchemaPresentationError(f"{label} exceeds the UTF-8 byte bound")
    if value != unicodedata.normalize("NFC", value):
        raise GVSSchemaPresentationError(f"{label} must be NFC-normalized")
    if any(
        character in "\r\n" or unicodedata.category(character).startswith("C")
        for character in value
    ):
        raise GVSSchemaPresentationError(
            f"{label} contains a newline, control, or formatting character"
        )
    return value


def _json_scalar(value: object, *, label: str) -> JSONScalar:
    if value is None or type(value) in {bool, int, float, str}:
        try:
            _canonical_json(value)
        except GVSSchemaPresentationError as error:
            raise GVSSchemaPresentationError(f"{label} is not a finite JSON scalar") from error
        if type(value) is str:
            try:
                encoded = value.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise GVSSchemaPresentationError(f"{label} is not valid UTF-8") from error
            if len(encoded) > _MAX_SCHEMA_SCALAR_UTF8_BYTES:
                raise GVSSchemaPresentationError(f"{label} exceeds the schema scalar byte bound")
            if value != unicodedata.normalize("NFC", value):
                raise GVSSchemaPresentationError(f"{label} must be NFC-normalized")
        elif type(value) is int and len(str(value).removeprefix("-")) > _MAX_JSON_INTEGER_DIGITS:
            raise GVSSchemaPresentationError(f"{label} exceeds the integer scalar bound")
        elif type(value) is float and len(_canonical_json(value).encode("utf-8")) > 64:
            raise GVSSchemaPresentationError(f"{label} exceeds the numeric scalar bound")
        return value
    raise GVSSchemaPresentationError(f"{label} must be an exact JSON scalar")


def _value_schema_record(
    schema: ValueSchema,
    *,
    path: str,
    depth: int = 0,
    _nodes: list[int] | None = None,
) -> dict[str, object]:
    nodes = [0] if _nodes is None else _nodes
    nodes[0] += 1
    if nodes[0] > _MAX_SCHEMA_NODES_PER_TOOL:
        raise GVSSchemaPresentationError("value schema exceeds the presentation node bound")
    if type(schema) is not ValueSchema:
        raise GVSSchemaPresentationError(f"{path} must be an exact ValueSchema")
    if depth > 32:
        raise GVSSchemaPresentationError("value schema nesting exceeds the presentation bound")
    if type(schema.kind) is not JSONType:
        raise GVSSchemaPresentationError(f"{path}.kind must be an exact JSONType")
    if type(schema.properties) is not _MAPPING_PROXY_TYPE:
        raise GVSSchemaPresentationError(f"{path}.properties must remain immutable")
    if type(schema.required) is not frozenset:
        raise GVSSchemaPresentationError(f"{path}.required must remain a frozenset")
    if type(schema.additional_properties) is not bool or type(schema.set_semantics) is not bool:
        raise GVSSchemaPresentationError(f"{path} boolean schema flags changed type")
    if schema.enum is not None and type(schema.enum) is not tuple:
        raise GVSSchemaPresentationError(f"{path}.enum must remain an exact tuple")
    if len(schema.properties) > _MAX_PROPERTIES_PER_NODE:
        raise GVSSchemaPresentationError(f"{path}.properties exceeds the presentation bound")
    if schema.enum is not None and len(schema.enum) > _MAX_ENUM_VALUES:
        raise GVSSchemaPresentationError(f"{path}.enum exceeds the presentation bound")
    for name in schema.properties:
        _strict_identifier(name, label=f"{path}.property name")
    for name in schema.required:
        _strict_identifier(name, label=f"{path}.required name")
    unknown_required = schema.required.difference(schema.properties)
    if unknown_required:
        raise GVSSchemaPresentationError(
            f"{path}.required contains undeclared properties: {sorted(unknown_required)!r}"
        )
    properties = {
        name: _value_schema_record(
            child,
            path=f"{path}.properties.{name}",
            depth=depth + 1,
            _nodes=nodes,
        )
        for name, child in sorted(schema.properties.items())
    }
    enum = (
        None
        if schema.enum is None
        else [
            _json_scalar(value, label=f"{path}.enum[{index}]")
            for index, value in enumerate(schema.enum)
        ]
    )
    return {
        "additional_properties": schema.additional_properties,
        "enum": enum,
        "items": (
            None
            if schema.items is None
            else _value_schema_record(
                schema.items,
                path=f"{path}.items",
                depth=depth + 1,
                _nodes=nodes,
            )
        ),
        "kind": schema.kind.value,
        "properties": properties,
        "required": sorted(schema.required),
        "set_semantics": schema.set_semantics,
    }


def _value_schema_from_record(
    record: Mapping[str, object],
    *,
    path: str = "value_schema",
    depth: int = 0,
    _nodes: list[int] | None = None,
) -> ValueSchema:
    nodes = [0] if _nodes is None else _nodes
    nodes[0] += 1
    if depth > 32 or nodes[0] > _MAX_SCHEMA_NODES_PER_TOOL:
        raise GVSSchemaPresentationError("serialized value schema exceeds the nesting/node bound")
    row = _exact_object(
        record,
        frozenset(
            {
                "additional_properties",
                "enum",
                "items",
                "kind",
                "properties",
                "required",
                "set_semantics",
            }
        ),
        label=path,
    )
    properties = row["properties"]
    if type(properties) is not dict:
        raise GVSSchemaPresentationError("internal value-schema properties are malformed")
    if len(properties) > _MAX_PROPERTIES_PER_NODE:
        raise GVSSchemaPresentationError("serialized value-schema properties exceed the bound")
    required = row["required"]
    enum = row["enum"]
    if type(required) is not list or any(type(value) is not str for value in required):
        raise GVSSchemaPresentationError("serialized value-schema required must be a string array")
    if required != sorted(set(required)):
        raise GVSSchemaPresentationError("serialized value-schema required is not canonical")
    if enum is not None and (type(enum) is not list or len(enum) > _MAX_ENUM_VALUES):
        raise GVSSchemaPresentationError("serialized value-schema enum exceeds the bound")
    for index, value in enumerate(enum or []):
        _json_scalar(value, label=f"{path}.enum[{index}]")
    items = row["items"]
    if items is not None and type(items) is not dict:
        raise GVSSchemaPresentationError("serialized value-schema items must be an object")
    if type(row["additional_properties"]) is not bool or type(row["set_semantics"]) is not bool:
        raise GVSSchemaPresentationError("serialized value-schema flags must be booleans")
    try:
        return ValueSchema(
            kind=JSONType(row["kind"]),
            items=(
                None
                if items is None
                else _value_schema_from_record(
                    items,
                    path=f"{path}.items",
                    depth=depth + 1,
                    _nodes=nodes,
                )
            ),
            properties={
                _strict_identifier(name, label=f"{path}.property name"): _value_schema_from_record(
                    child,
                    path=f"{path}.properties.{name}",
                    depth=depth + 1,
                    _nodes=nodes,
                )
                for name, child in properties.items()
            },
            required=frozenset(required),
            additional_properties=row["additional_properties"],
            set_semantics=row["set_semantics"],
            enum=None if enum is None else tuple(enum),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GVSSchemaPresentationError("serialized value schema is invalid") from error


def _tool_schema_record(schema: ToolSchema, *, path: str) -> dict[str, object]:
    if type(schema) is not ToolSchema:
        raise GVSSchemaPresentationError(f"{path} must be an exact ToolSchema")
    _strict_identifier(schema.name, label=f"{path}.name")
    if type(schema.arguments) is not _MAPPING_PROXY_TYPE:
        raise GVSSchemaPresentationError(f"{path}.arguments must remain immutable")
    if type(schema.required) is not frozenset:
        raise GVSSchemaPresentationError(f"{path}.required must remain a frozenset")
    if type(schema.additional_arguments) is not bool or type(schema.side_effecting) is not bool:
        raise GVSSchemaPresentationError(f"{path} boolean tool flags changed type")
    if schema.additional_arguments:
        raise GVSSchemaPresentationError(
            f"{path} permits unmapped additional arguments and cannot be losslessly renamed"
        )
    if len(schema.arguments) > _MAX_ARGUMENTS_PER_TOOL:
        raise GVSSchemaPresentationError(f"{path}.arguments exceeds the presentation bound")
    for name in schema.arguments:
        _strict_identifier(name, label=f"{path}.argument name")
    for name in schema.required:
        _strict_identifier(name, label=f"{path}.required name")
    unknown_required = schema.required.difference(schema.arguments)
    if unknown_required:
        raise GVSSchemaPresentationError(
            f"{path}.required contains undeclared arguments: {sorted(unknown_required)!r}"
        )
    nodes = [0]
    return {
        "additional_arguments": schema.additional_arguments,
        "arguments": {
            name: _value_schema_record(
                child,
                path=f"{path}.arguments.{name}",
                _nodes=nodes,
            )
            for name, child in sorted(schema.arguments.items())
        },
        "name": schema.name,
        "required": sorted(schema.required),
        "side_effecting": schema.side_effecting,
    }


def _tool_schema_from_record(record: object, *, path: str) -> ToolSchema:
    row = _exact_object(
        record,
        frozenset(
            {
                "additional_arguments",
                "arguments",
                "name",
                "required",
                "side_effecting",
            }
        ),
        label=path,
    )
    arguments = row["arguments"]
    required = row["required"]
    if type(arguments) is not dict or len(arguments) > _MAX_ARGUMENTS_PER_TOOL:
        raise GVSSchemaPresentationError(f"{path}.arguments must be a bounded exact object")
    if type(required) is not list or any(type(value) is not str for value in required):
        raise GVSSchemaPresentationError(f"{path}.required must be a string array")
    if required != sorted(set(required)):
        raise GVSSchemaPresentationError(f"{path}.required must be canonical and unique")
    if type(row["additional_arguments"]) is not bool or type(row["side_effecting"]) is not bool:
        raise GVSSchemaPresentationError(f"{path} flags must be exact booleans")
    nodes = [0]
    try:
        return ToolSchema(
            name=_strict_identifier(row["name"], label=f"{path}.name"),
            arguments={
                _strict_identifier(name, label=f"{path}.argument name"): _value_schema_from_record(
                    child,
                    path=f"{path}.arguments.{name}",
                    _nodes=nodes,
                )
                for name, child in arguments.items()
            },
            required=frozenset(required),
            additional_arguments=row["additional_arguments"],
            side_effecting=row["side_effecting"],
        )
    except (TypeError, ValueError) as error:
        raise GVSSchemaPresentationError(f"{path} is not a valid tool schema") from error


def _schema_snapshot(
    schemas: tuple[ToolSchema, ...],
    *,
    label: str,
    sort_by_name: bool,
) -> tuple[tuple[ToolSchema, ...], list[dict[str, object]]]:
    if type(schemas) is not tuple:
        raise GVSSchemaPresentationError(f"{label} must be an exact tuple")
    if not schemas or len(schemas) > _MAX_TOOLS:
        raise GVSSchemaPresentationError(f"{label} must be nonempty and bounded")
    records = [
        _tool_schema_record(schema, path=f"{label}[{index}]")
        for index, schema in enumerate(schemas)
    ]
    names = [record["name"] for record in records]
    if len(names) != len(set(names)):
        raise GVSSchemaPresentationError(f"{label} names must be unique")
    if sort_by_name:
        records.sort(key=lambda value: value["name"])
    detached = tuple(
        _tool_schema_from_record(record, path=f"detached_{label}[{index}]")
        for index, record in enumerate(records)
    )
    return detached, records


def _canonical_schema_snapshot(
    schemas: tuple[ToolSchema, ...],
) -> tuple[tuple[ToolSchema, ...], list[dict[str, object]]]:
    return _schema_snapshot(
        schemas,
        label="canonical schemas",
        sort_by_name=True,
    )


def _schemas_from_records(value: object, *, label: str) -> tuple[ToolSchema, ...]:
    if type(value) is not list or not value or len(value) > _MAX_TOOLS:
        raise GVSSchemaPresentationError(f"{label} must be a nonempty bounded array")
    schemas = tuple(
        _tool_schema_from_record(record, path=f"{label}[{index}]")
        for index, record in enumerate(value)
    )
    _, canonical_records = _schema_snapshot(
        schemas,
        label=label,
        sort_by_name=False,
    )
    if canonical_records != value:
        raise GVSSchemaPresentationError(f"{label} is not canonically serialized")
    return schemas


@dataclass(frozen=True, slots=True)
class ArgumentNameMap:
    canonical_name: str
    presented_name: str

    def __post_init__(self) -> None:
        _strict_identifier(self.canonical_name, label="canonical argument name")
        _strict_identifier(self.presented_name, label="presented argument name")

    def to_record(self) -> dict[str, str]:
        return {
            "canonical_name": self.canonical_name,
            "presented_name": self.presented_name,
        }


@dataclass(frozen=True, slots=True)
class ToolNameMap:
    canonical_name: str
    presented_name: str
    arguments: tuple[ArgumentNameMap, ...]

    def __post_init__(self) -> None:
        _strict_identifier(self.canonical_name, label="canonical tool name")
        _strict_identifier(self.presented_name, label="presented tool name")
        if type(self.arguments) is not tuple or any(
            type(value) is not ArgumentNameMap for value in self.arguments
        ):
            raise GVSSchemaPresentationError("argument name maps must be an exact tuple")
        if len(self.arguments) > _MAX_ARGUMENTS_PER_TOOL:
            raise GVSSchemaPresentationError("argument name maps exceed the presentation bound")
        canonical = [value.canonical_name for value in self.arguments]
        presented = [value.presented_name for value in self.arguments]
        if canonical != sorted(set(canonical)):
            raise GVSSchemaPresentationError("argument maps must have unique canonical-name order")
        if len(presented) != len(set(presented)):
            raise GVSSchemaPresentationError("presented argument names must be unique per tool")

    def to_record(self) -> dict[str, object]:
        return {
            "arguments": [value.to_record() for value in self.arguments],
            "canonical_name": self.canonical_name,
            "presented_name": self.presented_name,
        }


@dataclass(frozen=True, slots=True)
class SchemaPresentation:
    presentation_id: str
    mode: str
    tools: tuple[ToolNameMap, ...]
    schema_version: str = GVS_SCHEMA_PRESENTATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GVS_SCHEMA_PRESENTATION_VERSION:
            raise GVSSchemaPresentationError("schema presentation version changed")
        _strict_identifier(self.presentation_id, label="presentation_id")
        if type(self.mode) is not str or self.mode not in SUPPORTED_PRESENTATION_MODES:
            raise GVSSchemaPresentationError("presentation mode must be identity or renamed")
        if (
            type(self.tools) is not tuple
            or not self.tools
            or any(type(value) is not ToolNameMap for value in self.tools)
        ):
            raise GVSSchemaPresentationError("tool name maps must be a nonempty exact tuple")
        if len(self.tools) > _MAX_TOOLS:
            raise GVSSchemaPresentationError("tool name maps exceed the presentation bound")
        canonical = [value.canonical_name for value in self.tools]
        presented = [value.presented_name for value in self.tools]
        if canonical != sorted(set(canonical)):
            raise GVSSchemaPresentationError("tool maps must have unique canonical-name order")
        if len(presented) != len(set(presented)):
            raise GVSSchemaPresentationError("presented tool names must be globally unique")
        any_renamed = any(
            tool.canonical_name != tool.presented_name
            or any(arg.canonical_name != arg.presented_name for arg in tool.arguments)
            for tool in self.tools
        )
        if self.mode == IDENTITY_PRESENTATION and any_renamed:
            raise GVSSchemaPresentationError("identity presentation cannot rename a field")
        if self.mode == RENAMED_PRESENTATION and not any_renamed:
            raise GVSSchemaPresentationError("renamed presentation must rename at least one field")

    def to_record(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "presentation_id": self.presentation_id,
            "schema_version": self.schema_version,
            "tools": [value.to_record() for value in self.tools],
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_record())


def _presentation_from_record(value: object) -> SchemaPresentation:
    row = _exact_object(
        value,
        frozenset({"mode", "presentation_id", "schema_version", "tools"}),
        label="schema presentation",
    )
    tools = row["tools"]
    if type(tools) is not list or not tools or len(tools) > _MAX_TOOLS:
        raise GVSSchemaPresentationError("serialized tool maps must be a bounded array")
    parsed_tools: list[ToolNameMap] = []
    for tool_index, value_tool in enumerate(tools):
        tool = _exact_object(
            value_tool,
            frozenset({"arguments", "canonical_name", "presented_name"}),
            label=f"schema presentation tool {tool_index}",
        )
        arguments = tool["arguments"]
        if type(arguments) is not list or len(arguments) > _MAX_ARGUMENTS_PER_TOOL:
            raise GVSSchemaPresentationError("serialized argument maps must be bounded arrays")
        parsed_arguments: list[ArgumentNameMap] = []
        for argument_index, value_argument in enumerate(arguments):
            argument = _exact_object(
                value_argument,
                frozenset({"canonical_name", "presented_name"}),
                label=f"schema presentation argument {tool_index}:{argument_index}",
            )
            parsed_arguments.append(
                ArgumentNameMap(
                    canonical_name=argument["canonical_name"],
                    presented_name=argument["presented_name"],
                )
            )
        parsed_tools.append(
            ToolNameMap(
                canonical_name=tool["canonical_name"],
                presented_name=tool["presented_name"],
                arguments=tuple(parsed_arguments),
            )
        )
    return SchemaPresentation(
        presentation_id=row["presentation_id"],
        mode=row["mode"],
        tools=tuple(parsed_tools),
        schema_version=row["schema_version"],
    )


def _stable_presentation(presentation: SchemaPresentation) -> SchemaPresentation:
    if type(presentation) is not SchemaPresentation:
        raise GVSSchemaPresentationError("presentation must be exact SchemaPresentation")
    first = presentation.to_record()
    detached = _presentation_from_record(first)
    second = presentation.to_record()
    if first != second:
        raise GVSSchemaPresentationError("presentation mapping changed during stable snapshot")
    if detached.to_record() != first:
        raise GVSSchemaPresentationError("presentation mapping is not canonical")
    return detached


def _prompt_tools_record(tools: tuple[ToolDefinition, ...]) -> list[dict[str, object]]:
    if type(tools) is not tuple:
        raise GVSSchemaPresentationError("prompt_tools must be an exact tuple")
    if not tools or len(tools) > _MAX_TOOLS:
        raise GVSSchemaPresentationError("prompt tools must be nonempty and bounded")
    records: list[dict[str, object]] = []
    names: set[str] = set()
    for tool_index, tool in enumerate(tools):
        if type(tool) is not ToolDefinition:
            raise GVSSchemaPresentationError(
                f"prompt_tools[{tool_index}] must be an exact ToolDefinition"
            )
        name = _strict_identifier(tool.name, label=f"prompt_tools[{tool_index}].name")
        if name in names:
            raise GVSSchemaPresentationError("prompt tool names must be unique")
        names.add(name)
        description = _strict_description(
            tool.description,
            label=f"prompt_tools[{tool_index}].description",
        )
        if type(tool.arguments) is not tuple:
            raise GVSSchemaPresentationError("prompt tool arguments must be an exact tuple")
        if len(tool.arguments) > _MAX_ARGUMENTS_PER_TOOL:
            raise GVSSchemaPresentationError("prompt tool arguments exceed the bound")
        argument_records: list[dict[str, object]] = []
        argument_names: set[str] = set()
        for argument_index, argument in enumerate(tool.arguments):
            if type(argument) is not ToolArgument:
                raise GVSSchemaPresentationError("prompt argument must be exact ToolArgument")
            argument_name = _strict_identifier(
                argument.name,
                label=f"prompt_tools[{tool_index}].arguments[{argument_index}].name",
            )
            if argument_name in argument_names:
                raise GVSSchemaPresentationError("prompt argument names must be unique per tool")
            argument_names.add(argument_name)
            if type(argument.required) is not bool:
                raise GVSSchemaPresentationError("prompt argument required flag must be boolean")
            type_name = _strict_identifier(
                argument.type_name,
                label=f"prompt_tools[{tool_index}].arguments[{argument_index}].type_name",
            )
            argument_records.append(
                {
                    "description": _strict_description(
                        argument.description,
                        label=(
                            f"prompt_tools[{tool_index}].arguments[{argument_index}].description"
                        ),
                    ),
                    "name": argument_name,
                    "required": argument.required,
                    "type_name": type_name,
                }
            )
        records.append(
            {
                "arguments": argument_records,
                "description": description,
                "name": name,
            }
        )
    return records


def _prompt_tools_from_records(records: object) -> tuple[ToolDefinition, ...]:
    if type(records) is not list or not records or len(records) > _MAX_TOOLS:
        raise GVSSchemaPresentationError("serialized prompt tools must be a bounded array")
    result: list[ToolDefinition] = []
    for tool_index, value in enumerate(records):
        row = _exact_object(
            value,
            frozenset({"arguments", "description", "name"}),
            label=f"prompt_tools[{tool_index}]",
        )
        arguments = row["arguments"]
        if type(arguments) is not list or len(arguments) > _MAX_ARGUMENTS_PER_TOOL:
            raise GVSSchemaPresentationError("serialized prompt arguments must be bounded arrays")
        parsed_arguments: list[ToolArgument] = []
        for argument_index, argument in enumerate(arguments):
            argument_row = _exact_object(
                argument,
                frozenset({"description", "name", "required", "type_name"}),
                label=f"prompt_tools[{tool_index}].arguments[{argument_index}]",
            )
            if type(argument_row["required"]) is not bool:
                raise GVSSchemaPresentationError(
                    "serialized prompt required flags must be exact booleans"
                )
            parsed_arguments.append(
                ToolArgument(
                    name=_strict_identifier(
                        argument_row["name"],
                        label="serialized prompt argument name",
                    ),
                    type_name=_strict_identifier(
                        argument_row["type_name"],
                        label="serialized prompt argument type",
                    ),
                    description=_strict_description(
                        argument_row["description"],
                        label="serialized prompt argument description",
                    ),
                    required=argument_row["required"],
                )
            )
        result.append(
            ToolDefinition(
                name=_strict_identifier(row["name"], label="serialized prompt tool name"),
                description=_strict_description(
                    row["description"],
                    label="serialized prompt tool description",
                ),
                arguments=tuple(parsed_arguments),
            )
        )
    detached = tuple(result)
    if _prompt_tools_record(detached) != records:
        raise GVSSchemaPresentationError("serialized prompt tools are not canonical")
    return detached


def _stable_prompt_tools(
    tools: tuple[ToolDefinition, ...],
) -> tuple[tuple[ToolDefinition, ...], list[dict[str, object]]]:
    first = _prompt_tools_record(tools)
    detached = _prompt_tools_from_records(first)
    second = _prompt_tools_record(tools)
    if first != second:
        raise GVSSchemaPresentationError("prompt tools changed during the stable snapshot")
    return detached, first


def _derive_presented_schemas(
    *,
    canonical_schemas: tuple[ToolSchema, ...],
    presentation: SchemaPresentation,
    prompt_records: list[dict[str, object]],
) -> tuple[tuple[ToolSchema, ...], list[dict[str, object]]]:
    canonical_by_name = {schema.name: schema for schema in canonical_schemas}
    maps_by_canonical = {value.canonical_name: value for value in presentation.tools}
    if set(maps_by_canonical) != set(canonical_by_name):
        raise GVSSchemaPresentationError(
            "presentation must cover exactly every supplied canonical tool"
        )

    presented_by_name: dict[str, ToolSchema] = {}
    for canonical_name in sorted(canonical_by_name):
        canonical = canonical_by_name[canonical_name]
        tool_map = maps_by_canonical[canonical_name]
        arg_maps = {value.canonical_name: value.presented_name for value in tool_map.arguments}
        if set(arg_maps) != set(canonical.arguments):
            raise GVSSchemaPresentationError(
                f"argument map for {canonical_name!r} is not total and exact"
            )
        nodes = [0]
        presented_arguments = {
            arg_maps[name]: _value_schema_from_record(
                _value_schema_record(
                    canonical.arguments[name],
                    path=f"canonical.{canonical_name}.{name}",
                    _nodes=nodes,
                ),
                path=f"presented.{tool_map.presented_name}.{arg_maps[name]}",
                _nodes=[0],
            )
            for name in sorted(canonical.arguments)
        }
        presented_required = frozenset(arg_maps[name] for name in canonical.required)
        presented_by_name[tool_map.presented_name] = ToolSchema(
            name=tool_map.presented_name,
            arguments=presented_arguments,
            required=presented_required,
            additional_arguments=False,
            side_effecting=canonical.side_effecting,
        )

    if {record["name"] for record in prompt_records} != set(presented_by_name):
        raise GVSSchemaPresentationError(
            "prompt tools must match exactly the presented tool-name roster"
        )
    prompt_by_name = {record["name"]: record for record in prompt_records}
    for name, schema in presented_by_name.items():
        prompt = prompt_by_name[name]
        arguments = prompt["arguments"]
        if type(arguments) is not list:
            raise GVSSchemaPresentationError("internal prompt arguments are malformed")
        prompt_arguments = {value["name"]: value for value in arguments}
        if set(prompt_arguments) != set(schema.arguments):
            raise GVSSchemaPresentationError(
                f"prompt arguments for {name!r} differ from the presented schema"
            )
        for argument_name, value_schema in schema.arguments.items():
            prompt_argument = prompt_arguments[argument_name]
            if prompt_argument["type_name"] != value_schema.kind.value:
                raise GVSSchemaPresentationError(
                    f"prompt type for {name!r}.{argument_name!r} differs from canonical semantics"
                )
            if prompt_argument["required"] is not (argument_name in schema.required):
                raise GVSSchemaPresentationError(
                    f"prompt required flag for {name!r}.{argument_name!r} differs"
                )

    presented_schemas = tuple(presented_by_name[record["name"]] for record in prompt_records)
    detached_presented, presented_records = _schema_snapshot(
        presented_schemas,
        label="presented schemas",
        sort_by_name=False,
    )
    return detached_presented, presented_records


@dataclass(frozen=True, slots=True)
class SchemaPresentationReceipt:
    presentation: SchemaPresentation
    presentation_sha256: str
    canonical_schema_sha256: str
    presented_schema_sha256: str
    prompt_tools_sha256: str
    source_files_sha256: str
    runtime_identity_sha256: str
    canonical_schemas: tuple[ToolSchema, ...]
    presented_schemas: tuple[ToolSchema, ...]
    prompt_tools: tuple[ToolDefinition, ...]
    schema_version: str = GVS_SCHEMA_PRESENTATION_RECEIPT_VERSION
    descriptions_semantically_audited: bool = False
    source_custody_authenticated: bool = False
    authorizes_model_or_label_access: bool = False
    authorizes_cuda_or_jarvis_access: bool = False
    launch_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != GVS_SCHEMA_PRESENTATION_RECEIPT_VERSION:
            raise GVSSchemaPresentationError("schema presentation receipt version changed")
        if type(self.presentation) is not SchemaPresentation:
            raise GVSSchemaPresentationError("receipt presentation has the wrong type")
        for name in (
            "presentation_sha256",
            "canonical_schema_sha256",
            "presented_schema_sha256",
            "prompt_tools_sha256",
            "source_files_sha256",
            "runtime_identity_sha256",
        ):
            value = getattr(self, name)
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise GVSSchemaPresentationError(f"{name} must be lowercase SHA-256")
        if self.source_files_sha256 != _INITIAL_SOURCE_FILES_SHA256:
            raise GVSSchemaPresentationError("receipt source differs from the import-time baseline")
        if self.runtime_identity_sha256 != _INITIAL_RUNTIME_IDENTITY_SHA256:
            raise GVSSchemaPresentationError(
                "receipt runtime differs from the import-time baseline"
            )
        detached_presentation = _stable_presentation(self.presentation)
        if detached_presentation.sha256 != self.presentation_sha256:
            raise GVSSchemaPresentationError("presentation hash differs from retained mapping")
        if tuple(schema.name for schema in self.canonical_schemas) != tuple(
            sorted(schema.name for schema in self.canonical_schemas)
        ):
            raise GVSSchemaPresentationError("retained canonical schemas are not in name order")
        canonical_schemas, canonical_records = _canonical_schema_snapshot(self.canonical_schemas)
        _, prompt_records = _stable_prompt_tools(self.prompt_tools)
        _, presented_records = _schema_snapshot(
            self.presented_schemas,
            label="presented schemas",
            sort_by_name=False,
        )
        if _sha256_json(canonical_records) != self.canonical_schema_sha256:
            raise GVSSchemaPresentationError("canonical schema hash differs from retained schemas")
        if _sha256_json(presented_records) != self.presented_schema_sha256:
            raise GVSSchemaPresentationError("presented schema hash differs from retained schemas")
        if _sha256_json(prompt_records) != self.prompt_tools_sha256:
            raise GVSSchemaPresentationError("prompt tool hash differs from retained prompt tools")
        _, expected_presented_records = _derive_presented_schemas(
            canonical_schemas=canonical_schemas,
            presentation=detached_presentation,
            prompt_records=prompt_records,
        )
        if expected_presented_records != presented_records:
            raise GVSSchemaPresentationError(
                "retained presented schemas are not the exact name-only projection"
            )
        for name in (
            "descriptions_semantically_audited",
            "source_custody_authenticated",
            "authorizes_model_or_label_access",
            "authorizes_cuda_or_jarvis_access",
            "launch_authorized",
        ):
            if getattr(self, name) is not False:
                raise GVSSchemaPresentationError(
                    "schema presentation receipt must remain nonauthorizing"
                )

    def to_record(self) -> dict[str, object]:
        canonical_records = [
            _tool_schema_record(schema, path=f"canonical_schemas[{index}]")
            for index, schema in enumerate(self.canonical_schemas)
        ]
        presented_records = [
            _tool_schema_record(schema, path=f"presented_schemas[{index}]")
            for index, schema in enumerate(self.presented_schemas)
        ]
        return {
            "authorizes_cuda_or_jarvis_access": self.authorizes_cuda_or_jarvis_access,
            "authorizes_model_or_label_access": self.authorizes_model_or_label_access,
            "canonical_schema_sha256": self.canonical_schema_sha256,
            "canonical_schemas": canonical_records,
            "descriptions_semantically_audited": self.descriptions_semantically_audited,
            "launch_authorized": self.launch_authorized,
            "presentation": self.presentation.to_record(),
            "presentation_sha256": self.presentation_sha256,
            "presented_schema_sha256": self.presented_schema_sha256,
            "presented_schemas": presented_records,
            "prompt_tools": _prompt_tools_record(self.prompt_tools),
            "prompt_tools_sha256": self.prompt_tools_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "schema_version": self.schema_version,
            "source_custody_authenticated": self.source_custody_authenticated,
            "source_files_sha256": self.source_files_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_record())

    def to_json_bytes(self) -> bytes:
        return (
            _canonical_json(
                {
                    "artifact_schema_version": GVS_SCHEMA_PRESENTATION_ARTIFACT_VERSION,
                    "receipt": self.to_record(),
                    "receipt_sha256": self.sha256,
                }
            )
            + "\n"
        ).encode("utf-8")


def build_schema_presentation(
    *,
    canonical_schemas: tuple[ToolSchema, ...],
    prompt_tools: tuple[ToolDefinition, ...],
    presentation: SchemaPresentation,
) -> SchemaPresentationReceipt:
    """Validate and detach one total name-only model-facing schema presentation."""

    source_before, runtime_before = _assert_live_runtime()
    detached_presentation = _stable_presentation(presentation)
    detached_canonical, canonical_records = _canonical_schema_snapshot(canonical_schemas)
    detached_prompt_tools, prompt_records = _stable_prompt_tools(prompt_tools)
    presented_schemas, presented_records = _derive_presented_schemas(
        canonical_schemas=detached_canonical,
        presentation=detached_presentation,
        prompt_records=prompt_records,
    )
    _, canonical_records_after = _canonical_schema_snapshot(canonical_schemas)
    if canonical_records_after != canonical_records:
        raise GVSSchemaPresentationError("canonical schemas changed during presentation build")
    if _prompt_tools_record(prompt_tools) != prompt_records:
        raise GVSSchemaPresentationError("prompt tools changed during presentation build")
    if presentation.to_record() != detached_presentation.to_record():
        raise GVSSchemaPresentationError("presentation mapping changed during build")
    source_after, runtime_after = _assert_live_runtime()
    if source_after != source_before or runtime_after != runtime_before:
        raise GVSSchemaPresentationError("schema-presentation runtime changed during build")
    return SchemaPresentationReceipt(
        presentation=detached_presentation,
        presentation_sha256=detached_presentation.sha256,
        canonical_schema_sha256=_sha256_json(canonical_records),
        presented_schema_sha256=_sha256_json(presented_records),
        prompt_tools_sha256=_sha256_json(prompt_records),
        source_files_sha256=source_before,
        runtime_identity_sha256=runtime_before,
        canonical_schemas=detached_canonical,
        presented_schemas=presented_schemas,
        prompt_tools=detached_prompt_tools,
    )


def verify_schema_presentation(
    receipt: SchemaPresentationReceipt,
    *,
    canonical_schemas: tuple[ToolSchema, ...],
    prompt_tools: tuple[ToolDefinition, ...],
    presentation: SchemaPresentation,
) -> SchemaPresentationReceipt:
    """Completely recompute and compare a presentation receipt."""

    if type(receipt) is not SchemaPresentationReceipt:
        raise GVSSchemaPresentationError("receipt must be exact SchemaPresentationReceipt")
    source_before, runtime_before = _assert_live_runtime()
    first_receipt_record = _canonical_json(receipt.to_record())
    second_receipt_record = _canonical_json(receipt.to_record())
    if first_receipt_record != second_receipt_record:
        raise GVSSchemaPresentationError("schema presentation receipt changed during snapshot")
    expected = build_schema_presentation(
        canonical_schemas=canonical_schemas,
        prompt_tools=prompt_tools,
        presentation=presentation,
    )
    if first_receipt_record != _canonical_json(receipt.to_record()):
        raise GVSSchemaPresentationError("schema presentation receipt changed during recomputation")
    if first_receipt_record != _canonical_json(expected.to_record()):
        raise GVSSchemaPresentationError("schema presentation differs from live recomputation")
    source_after, runtime_after = _assert_live_runtime()
    if source_after != source_before or runtime_after != runtime_before:
        raise GVSSchemaPresentationError("schema-presentation runtime changed during verification")
    return expected


def load_schema_presentation_receipt_json(
    raw: bytes,
    *,
    expected_receipt_sha256: str,
    canonical_schemas: tuple[ToolSchema, ...],
    prompt_tools: tuple[ToolDefinition, ...],
    presentation: SchemaPresentation,
) -> SchemaPresentationReceipt:
    """Load one bounded strict artifact and completely rederive its receipt."""

    source_before, runtime_before = _assert_live_runtime()
    expected_receipt_sha256 = _strict_sha256(
        expected_receipt_sha256,
        label="expected_receipt_sha256",
    )
    if type(raw) is not bytes or not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        raise GVSSchemaPresentationError(
            "schema-presentation artifact must be nonempty bounded bytes"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_bounded_json_float,
            parse_int=_bounded_json_integer,
        )
    except GVSSchemaPresentationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise GVSSchemaPresentationError(
            "schema-presentation artifact is not strict bounded UTF-8 JSON"
        ) from error
    artifact = _exact_object(
        payload,
        frozenset({"artifact_schema_version", "receipt", "receipt_sha256"}),
        label="schema-presentation artifact",
    )
    if artifact["artifact_schema_version"] != GVS_SCHEMA_PRESENTATION_ARTIFACT_VERSION:
        raise GVSSchemaPresentationError("unsupported schema-presentation artifact version")
    row = _exact_object(
        artifact["receipt"],
        frozenset(
            {
                "authorizes_cuda_or_jarvis_access",
                "authorizes_model_or_label_access",
                "canonical_schema_sha256",
                "canonical_schemas",
                "descriptions_semantically_audited",
                "launch_authorized",
                "presentation",
                "presentation_sha256",
                "presented_schema_sha256",
                "presented_schemas",
                "prompt_tools",
                "prompt_tools_sha256",
                "runtime_identity_sha256",
                "schema_version",
                "source_custody_authenticated",
                "source_files_sha256",
            }
        ),
        label="schema-presentation receipt",
    )
    detached_presentation = _presentation_from_record(row["presentation"])
    if row["presentation_sha256"] != detached_presentation.sha256:
        raise GVSSchemaPresentationError("presentation hash failed strict reconstruction")
    receipt = SchemaPresentationReceipt(
        presentation=detached_presentation,
        presentation_sha256=row["presentation_sha256"],
        canonical_schema_sha256=row["canonical_schema_sha256"],
        presented_schema_sha256=row["presented_schema_sha256"],
        prompt_tools_sha256=row["prompt_tools_sha256"],
        source_files_sha256=row["source_files_sha256"],
        runtime_identity_sha256=row["runtime_identity_sha256"],
        canonical_schemas=_schemas_from_records(
            row["canonical_schemas"],
            label="canonical_schemas",
        ),
        presented_schemas=_schemas_from_records(
            row["presented_schemas"],
            label="presented_schemas",
        ),
        prompt_tools=_prompt_tools_from_records(row["prompt_tools"]),
        schema_version=row["schema_version"],
        descriptions_semantically_audited=row["descriptions_semantically_audited"],
        source_custody_authenticated=row["source_custody_authenticated"],
        authorizes_model_or_label_access=row["authorizes_model_or_label_access"],
        authorizes_cuda_or_jarvis_access=row["authorizes_cuda_or_jarvis_access"],
        launch_authorized=row["launch_authorized"],
    )
    embedded_sha256 = _strict_sha256(
        artifact["receipt_sha256"],
        label="embedded receipt_sha256",
    )
    if embedded_sha256 != receipt.sha256:
        raise GVSSchemaPresentationError("embedded receipt hash failed recomputation")
    if receipt.sha256 != expected_receipt_sha256:
        raise GVSSchemaPresentationError("schema-presentation receipt differs from commitment")
    source_after, runtime_after = _assert_live_runtime()
    if source_after != source_before or runtime_after != runtime_before:
        raise GVSSchemaPresentationError(
            "schema-presentation runtime changed during artifact loading"
        )
    return verify_schema_presentation(
        receipt,
        canonical_schemas=canonical_schemas,
        prompt_tools=prompt_tools,
        presentation=presentation,
    )


def lower_presented_action_ir(
    action: ActionIR,
    *,
    receipt: SchemaPresentationReceipt,
    canonical_schemas: tuple[ToolSchema, ...],
    prompt_tools: tuple[ToolDefinition, ...],
    presentation: SchemaPresentation,
) -> ActionIR:
    """Lower only tool/argument names; preserve every semantic value and control."""

    verified = verify_schema_presentation(
        receipt,
        canonical_schemas=canonical_schemas,
        prompt_tools=prompt_tools,
        presentation=presentation,
    )
    if type(action) is not ActionIR:
        raise GVSSchemaPresentationError("action must be an exact ActionIR")
    try:
        action_json_before = action.canonical_json()
        detached_action = parse_action_ir(action_json_before, verified.presented_schemas)
        if action.canonical_json() != action_json_before:
            raise GVSSchemaPresentationError("presented action changed during lowering")
    except GVSSchemaPresentationError:
        raise
    except (ActionIRError, TypeError, ValueError, UnicodeError) as error:
        raise GVSSchemaPresentationError(
            "presented action failed exact live-schema revalidation"
        ) from error
    action = detached_action
    canonical_detached, canonical_records = _canonical_schema_snapshot(canonical_schemas)
    if _sha256_json(canonical_records) != verified.canonical_schema_sha256:
        raise GVSSchemaPresentationError("canonical schemas changed after receipt verification")
    tool_by_presented = {value.presented_name: value for value in verified.presentation.tools}

    if action.decision is Decision.ABSTAIN:
        lowered_no_call = ActionIR(decision=Decision.ABSTAIN)
        _assert_live_runtime()
        return lowered_no_call
    if action.decision is Decision.CLARIFY:
        lowered_no_call = ActionIR(decision=Decision.CLARIFY, missing=action.missing)
        if lowered_no_call.canonical_json() != action.canonical_json():
            raise GVSSchemaPresentationError("schema lowering changed CLARIFY semantics")
        _assert_live_runtime()
        return lowered_no_call
    if (
        action.decision not in {Decision.CALL, Decision.CONFIRM}
        or type(action.mode) is not CallMode
    ):
        raise GVSSchemaPresentationError("presented action has an unsupported control shape")

    calls: list[dict[str, object]] = []
    for index, call in enumerate(action.calls):
        tool_map = tool_by_presented.get(call.tool)
        if tool_map is None:
            raise GVSSchemaPresentationError(
                f"presented action call {index} uses a tool outside the frozen mapping"
            )
        args_by_presented = {
            value.presented_name: value.canonical_name for value in tool_map.arguments
        }
        if not set(call.args).issubset(args_by_presented):
            raise GVSSchemaPresentationError(
                f"presented action call {index} argument roster changed after validation"
            )
        canonical_args: dict[str, CanonicalJSONValue] = {
            args_by_presented[name]: value for name, value in call.args.items()
        }
        calls.append({"args": canonical_args, "tool": tool_map.canonical_name})
    payload: Mapping[str, CanonicalJSONValue] = MappingProxyType(
        {
            "calls": tuple(calls),
            "decision": action.decision.value,
            "mode": action.mode.value,
        }
    )
    lowered = validate_action_ir(payload, canonical_detached)
    if lowered.decision is not action.decision or lowered.mode is not action.mode:
        raise GVSSchemaPresentationError("schema lowering changed Action IR control semantics")
    if len(lowered.calls) != len(action.calls):
        raise GVSSchemaPresentationError("schema lowering changed Action IR call count")
    for index, (presented_call, canonical_call) in enumerate(
        zip(action.calls, lowered.calls, strict=True)
    ):
        tool_map = tool_by_presented[presented_call.tool]
        if canonical_call.tool != tool_map.canonical_name:
            raise GVSSchemaPresentationError(
                f"schema lowering changed tool order or identity at call {index}"
            )
        args_by_presented = {
            value.presented_name: value.canonical_name for value in tool_map.arguments
        }
        expected_args = {
            args_by_presented[name]: value for name, value in presented_call.args.items()
        }
        if canonical_call.args != expected_args:
            raise GVSSchemaPresentationError(
                f"schema lowering changed argument values at call {index}"
            )
    _assert_live_runtime()
    return lowered


def action_round_trips_identity(
    action: ActionIR,
    *,
    receipt: SchemaPresentationReceipt,
    canonical_schemas: tuple[ToolSchema, ...],
    prompt_tools: tuple[ToolDefinition, ...],
    presentation: SchemaPresentation,
) -> bool:
    """Convenience check for identity presentations only."""

    if presentation.mode != IDENTITY_PRESENTATION:
        raise GVSSchemaPresentationError("identity round-trip requires identity presentation")
    lowered = lower_presented_action_ir(
        action,
        receipt=receipt,
        canonical_schemas=canonical_schemas,
        prompt_tools=prompt_tools,
        presentation=presentation,
    )
    return action_ir_equal(action, lowered)


__all__ = [
    "GVS_SCHEMA_PRESENTATION_ARTIFACT_VERSION",
    "GVS_SCHEMA_PRESENTATION_RECEIPT_VERSION",
    "GVS_SCHEMA_PRESENTATION_VERSION",
    "IDENTITY_PRESENTATION",
    "RENAMED_PRESENTATION",
    "ArgumentNameMap",
    "GVSSchemaPresentationError",
    "SchemaPresentation",
    "SchemaPresentationReceipt",
    "ToolNameMap",
    "action_round_trips_identity",
    "build_schema_presentation",
    "load_schema_presentation_receipt_json",
    "lower_presented_action_ir",
    "verify_schema_presentation",
]


_INITIAL_SOURCE_FILES_SHA256 = _sha256_json(_source_files_record())
_INITIAL_RUNTIME_IDENTITY_SHA256 = _runtime_identity_sha256()
