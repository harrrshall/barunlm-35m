"""Strict parsing and canonicalization for BarunAction Action IR v1.

This module deliberately uses only the Python standard library.  It does not try to
recover JSON from prose, Markdown fences, truncated generations, or malformed output.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
CanonicalJSONValue: TypeAlias = (
    JSONScalar | tuple["CanonicalJSONValue", ...] | Mapping[str, "CanonicalJSONValue"]
)
MAX_JSON_NESTING = 64


class ActionIRError(ValueError):
    """Base class carrying a stable machine-readable evaluator error code."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(f"{path}: {message}")
        self.code = code
        self.path = path


class ActionIRParseError(ActionIRError):
    """The raw model string is not exactly one strict JSON object."""


class ActionIRValidationError(ActionIRError):
    """The decoded object violates Action IR or a supplied tool schema."""


class Decision(str, Enum):
    CALL = "CALL"
    CONFIRM = "CONFIRM"
    CLARIFY = "CLARIFY"
    ABSTAIN = "ABSTAIN"


class CallMode(str, Enum):
    SINGLE = "SINGLE"
    SERIAL = "SERIAL"
    PARALLEL = "PARALLEL"


class JSONType(str, Enum):
    NULL = "null"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    ARRAY = "array"
    OBJECT = "object"


def _nfc(value: str, *, path: str, error_type: type[ActionIRError]) -> str:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise error_type("invalid_utf8", "string is not valid UTF-8", path) from exc
    return unicodedata.normalize("NFC", value)


def _to_builtin(value: CanonicalJSONValue) -> JSONValue:
    if isinstance(value, Mapping):
        return {key: _to_builtin(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_to_builtin(child) for child in value]
    return value


def _canonical_json(value: CanonicalJSONValue) -> str:
    return json.dumps(
        _to_builtin(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_value(value: CanonicalJSONValue) -> str:
    """Serialize one already-canonical JSON value deterministically."""

    return _canonical_json(value)


def _canonicalize_untyped(
    value: object,
    *,
    path: str,
    error_type: type[ActionIRError],
) -> CanonicalJSONValue:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise error_type("non_finite_number", "number must be finite", path)
        return value
    if isinstance(value, str):
        return _nfc(value, path=path, error_type=error_type)
    if isinstance(value, (list, tuple)):
        return tuple(
            _canonicalize_untyped(
                child,
                path=f"{path}[{index}]",
                error_type=error_type,
            )
            for index, child in enumerate(value)
        )
    if isinstance(value, Mapping):
        normalized: dict[str, CanonicalJSONValue] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):  # Defensive: JSON always supplies strings.
                raise error_type("type_mismatch", "object key must be a string", path)
            key = _nfc(raw_key, path=path, error_type=error_type)
            if key in normalized:
                raise error_type(
                    "duplicate_key",
                    f"object contains duplicate key after NFC normalization: {key!r}",
                    path,
                )
            normalized[key] = _canonicalize_untyped(
                child,
                path=f"{path}.{key}",
                error_type=error_type,
            )
        return MappingProxyType(dict(sorted(normalized.items())))
    raise error_type("type_mismatch", f"unsupported JSON value {type(value).__name__}", path)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ActionIRParseError("duplicate_key", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ActionIRParseError("non_finite_number", "number must be finite")
    return parsed


def _reject_constant(value: str) -> float:
    raise ActionIRParseError("non_finite_number", f"non-finite JSON number {value!r}")


def _check_nesting_depth(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > MAX_JSON_NESTING:
            raise ActionIRParseError(
                "max_nesting_exceeded",
                f"JSON nesting exceeds {MAX_JSON_NESTING}",
            )
        if isinstance(current, Mapping):
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend((child, depth + 1) for child in current)


def decode_json_object(raw: str) -> Mapping[str, CanonicalJSONValue]:
    """Decode exactly one strict JSON object without attempting any repair.

    Leading/trailing JSON whitespace is accepted.  Prose, Markdown fences, a second
    value, duplicate keys, NaN/Infinity, numeric overflow, invalid UTF-8 strings, and
    non-object top-level values are rejected.
    """

    if not isinstance(raw, str):
        raise ActionIRParseError("type_mismatch", "model output must be a string")
    try:
        raw.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ActionIRParseError("invalid_utf8", "model output is not valid UTF-8") from exc

    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except ActionIRParseError:
        raise
    except RecursionError as exc:
        raise ActionIRParseError(
            "max_nesting_exceeded",
            f"JSON nesting exceeds {MAX_JSON_NESTING}",
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ActionIRParseError("invalid_json", "output is not exactly one JSON value") from exc

    if not isinstance(decoded, Mapping):
        raise ActionIRParseError("top_level_not_object", "top-level JSON value must be an object")
    _check_nesting_depth(decoded)
    try:
        canonical = _canonicalize_untyped(decoded, path="$", error_type=ActionIRParseError)
    except RecursionError as exc:  # Defensive if interpreter recursion limits are unusually low.
        raise ActionIRParseError(
            "max_nesting_exceeded",
            f"JSON nesting exceeds {MAX_JSON_NESTING}",
        ) from exc
    assert isinstance(canonical, Mapping)
    return canonical


@dataclass(frozen=True, slots=True)
class ValueSchema:
    """A deterministic, typed subset of JSON Schema used for tool arguments."""

    kind: JSONType
    items: ValueSchema | None = None
    properties: Mapping[str, ValueSchema] = field(default_factory=dict)
    required: frozenset[str] = field(default_factory=frozenset)
    additional_properties: bool = False
    set_semantics: bool = False
    enum: tuple[JSONScalar, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, JSONType):
            try:
                object.__setattr__(self, "kind", JSONType(self.kind))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unknown JSON type: {self.kind!r}") from exc

        properties: dict[str, ValueSchema] = {}
        for name, schema in self.properties.items():
            if not isinstance(name, str) or not name:
                raise ValueError("schema property names must be non-empty strings")
            if unicodedata.normalize("NFC", name) != name:
                raise ValueError(f"schema property name is not NFC: {name!r}")
            if name in properties:
                raise ValueError(f"duplicate schema property: {name!r}")
            if not isinstance(schema, ValueSchema):
                raise TypeError(f"schema for property {name!r} must be ValueSchema")
            properties[name] = schema
        object.__setattr__(self, "properties", MappingProxyType(dict(sorted(properties.items()))))
        object.__setattr__(self, "required", frozenset(self.required))

        if self.kind is JSONType.ARRAY:
            if self.items is None:
                raise ValueError("array schemas require an items schema")
        elif self.items is not None or self.set_semantics:
            raise ValueError("items and set_semantics are valid only for array schemas")

        if self.kind is JSONType.OBJECT:
            unknown_required = self.required.difference(properties)
            if unknown_required:
                raise ValueError(
                    f"required properties are undeclared: {sorted(unknown_required)!r}"
                )
        elif properties or self.required or self.additional_properties:
            raise ValueError(
                "properties, required, and additional_properties are valid only for object schemas"
            )

        if not isinstance(self.additional_properties, bool):
            raise TypeError("additional_properties must be bool")
        if not isinstance(self.set_semantics, bool):
            raise TypeError("set_semantics must be bool")

        if self.enum is not None:
            canonical_enum: list[JSONScalar] = []
            seen: set[str] = set()
            for index, item in enumerate(self.enum):
                canonical = _validate_value(item, self, f"$enum[{index}]", check_enum=False)
                if isinstance(canonical, (tuple, Mapping)):
                    raise TypeError("enum values must be JSON scalars")
                key = _canonical_json(canonical)
                if key in seen:
                    raise ValueError("enum values must be unique after canonicalization")
                seen.add(key)
                canonical_enum.append(canonical)
            object.__setattr__(self, "enum", tuple(canonical_enum))


@dataclass(frozen=True, slots=True)
class ToolSchema:
    name: str
    arguments: Mapping[str, ValueSchema]
    required: frozenset[str] = field(default_factory=frozenset)
    additional_arguments: bool = False
    side_effecting: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tool name must be a non-empty string")
        if unicodedata.normalize("NFC", self.name) != self.name:
            raise ValueError("tool name must be NFC-normalized")

        arguments: dict[str, ValueSchema] = {}
        for name, schema in self.arguments.items():
            if not isinstance(name, str) or not name:
                raise ValueError("argument names must be non-empty strings")
            if unicodedata.normalize("NFC", name) != name:
                raise ValueError(f"argument name is not NFC: {name!r}")
            if name in arguments:
                raise ValueError(f"duplicate argument name: {name!r}")
            if not isinstance(schema, ValueSchema):
                raise TypeError(f"schema for argument {name!r} must be ValueSchema")
            arguments[name] = schema
        required = frozenset(self.required)
        unknown_required = required.difference(arguments)
        if unknown_required:
            raise ValueError(f"required arguments are undeclared: {sorted(unknown_required)!r}")
        object.__setattr__(self, "arguments", MappingProxyType(dict(sorted(arguments.items()))))
        object.__setattr__(self, "required", required)
        if not isinstance(self.additional_arguments, bool):
            raise TypeError("additional_arguments must be bool")
        if not isinstance(self.side_effecting, bool):
            raise TypeError("side_effecting must be bool")


def _kind_matches(value: object, kind: JSONType) -> bool:
    if kind is JSONType.NULL:
        return value is None
    if kind is JSONType.BOOLEAN:
        return isinstance(value, bool)
    if kind is JSONType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind is JSONType.NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind is JSONType.STRING:
        return isinstance(value, str)
    if kind is JSONType.ARRAY:
        return isinstance(value, (list, tuple))
    if kind is JSONType.OBJECT:
        return isinstance(value, Mapping)
    return False


def _validate_value(
    value: object,
    schema: ValueSchema,
    path: str,
    *,
    check_enum: bool = True,
) -> CanonicalJSONValue:
    if not _kind_matches(value, schema.kind):
        raise ActionIRValidationError(
            "type_mismatch",
            f"expected {schema.kind.value}, got {type(value).__name__}",
            path,
        )

    if schema.kind is JSONType.STRING:
        assert isinstance(value, str)
        canonical: CanonicalJSONValue = _nfc(value, path=path, error_type=ActionIRValidationError)
    elif schema.kind is JSONType.NUMBER:
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        try:
            numeric = float(value)
        except OverflowError as exc:
            raise ActionIRValidationError(
                "non_finite_number", "number must be finite", path
            ) from exc
        if not math.isfinite(numeric):
            raise ActionIRValidationError("non_finite_number", "number must be finite", path)
        canonical = numeric
    elif schema.kind is JSONType.ARRAY:
        assert isinstance(value, (list, tuple)) and schema.items is not None
        children = tuple(
            _validate_value(child, schema.items, f"{path}[{index}]")
            for index, child in enumerate(value)
        )
        if schema.set_semantics:
            unique = {_canonical_json(child): child for child in children}
            canonical = tuple(unique[key] for key in sorted(unique))
        else:
            canonical = children
    elif schema.kind is JSONType.OBJECT:
        assert isinstance(value, Mapping)
        missing = schema.required.difference(value)
        if missing:
            raise ActionIRValidationError(
                "missing_argument",
                f"missing required object properties: {sorted(missing)!r}",
                path,
            )
        extras = set(value).difference(schema.properties)
        if extras and not schema.additional_properties:
            raise ActionIRValidationError(
                "extra_argument",
                f"unexpected object properties: {sorted(extras)!r}",
                path,
            )
        output: dict[str, CanonicalJSONValue] = {}
        for key in sorted(value):
            child_path = f"{path}.{key}"
            if key in schema.properties:
                output[key] = _validate_value(value[key], schema.properties[key], child_path)
            else:
                output[key] = _canonicalize_untyped(
                    value[key], path=child_path, error_type=ActionIRValidationError
                )
        canonical = MappingProxyType(output)
    else:
        canonical = _canonicalize_untyped(value, path=path, error_type=ActionIRValidationError)

    if check_enum and schema.enum is not None:
        key = _canonical_json(canonical)
        valid = {_canonical_json(item) for item in schema.enum}
        if key not in valid:
            raise ActionIRValidationError(
                "enum_mismatch", "value is not one of the declared enum values", path
            )
    return canonical


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool: str
    args: Mapping[str, CanonicalJSONValue]

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "args": {key: _to_builtin(value) for key, value in self.args.items()},
            "tool": self.tool,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class ActionIR:
    decision: Decision
    calls: tuple[ToolCall, ...] = ()
    mode: CallMode | None = None
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, JSONValue]:
        if self.decision in (Decision.CALL, Decision.CONFIRM):
            calls = self.calls
            if self.mode is CallMode.PARALLEL:
                calls = tuple(sorted(calls, key=ToolCall.canonical_json))
            return {
                "calls": [call.to_dict() for call in calls],
                "decision": self.decision.value,
                "mode": self.mode.value if self.mode is not None else None,
            }
        if self.decision is Decision.CLARIFY:
            return {"decision": self.decision.value, "missing": list(self.missing)}
        return {"decision": self.decision.value}

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _expect_exact_fields(
    obj: Mapping[str, CanonicalJSONValue], expected: set[str], path: str
) -> None:
    missing = expected.difference(obj)
    if missing:
        raise ActionIRValidationError("missing_field", f"missing fields: {sorted(missing)!r}", path)
    extra = set(obj).difference(expected)
    if extra:
        raise ActionIRValidationError("unknown_field", f"unknown fields: {sorted(extra)!r}", path)


def _schema_registry(
    schemas: Mapping[str, ToolSchema] | Iterable[ToolSchema],
) -> Mapping[str, ToolSchema]:
    if isinstance(schemas, Mapping):
        registry = dict(schemas)
        for key, schema in registry.items():
            if not isinstance(schema, ToolSchema):
                raise TypeError(f"schema {key!r} must be ToolSchema")
            if key != schema.name:
                raise ValueError(f"schema mapping key {key!r} does not match {schema.name!r}")
    else:
        registry = {}
        for schema in schemas:
            if not isinstance(schema, ToolSchema):
                raise TypeError("all schemas must be ToolSchema")
            if schema.name in registry:
                raise ValueError(f"duplicate tool schema {schema.name!r}")
            registry[schema.name] = schema
    return MappingProxyType(dict(sorted(registry.items())))


def validate_action_ir(
    obj: Mapping[str, CanonicalJSONValue],
    schemas: Mapping[str, ToolSchema] | Iterable[ToolSchema],
) -> ActionIR:
    """Validate and canonicalize a decoded object as Action IR v1."""

    registry = _schema_registry(schemas)
    decision_value = obj.get("decision")
    if not isinstance(decision_value, str):
        raise ActionIRValidationError("type_mismatch", "decision must be a string", "$.decision")
    try:
        decision = Decision(decision_value)
    except ValueError as exc:
        raise ActionIRValidationError(
            "invalid_decision", f"unknown decision {decision_value!r}", "$.decision"
        ) from exc

    if decision is Decision.ABSTAIN:
        _expect_exact_fields(obj, {"decision"}, "$")
        return ActionIR(decision=decision)

    if decision is Decision.CLARIFY:
        _expect_exact_fields(obj, {"decision", "missing"}, "$")
        missing_value = obj["missing"]
        if not isinstance(missing_value, tuple):
            raise ActionIRValidationError("type_mismatch", "missing must be an array", "$.missing")
        if not missing_value:
            raise ActionIRValidationError(
                "empty_missing", "missing must contain at least one item", "$.missing"
            )
        missing: list[str] = []
        for index, item in enumerate(missing_value):
            if not isinstance(item, str):
                raise ActionIRValidationError(
                    "type_mismatch", "missing items must be strings", f"$.missing[{index}]"
                )
            normalized = _nfc(item, path=f"$.missing[{index}]", error_type=ActionIRValidationError)
            if not normalized:
                raise ActionIRValidationError(
                    "empty_missing_item", "missing items must be non-empty", f"$.missing[{index}]"
                )
            missing.append(normalized)
        if len(set(missing)) != len(missing):
            raise ActionIRValidationError(
                "duplicate_missing", "missing items must be unique", "$.missing"
            )
        if missing != sorted(missing):
            raise ActionIRValidationError(
                "unsorted_missing", "missing items must be sorted", "$.missing"
            )
        return ActionIR(decision=decision, missing=tuple(missing))

    _expect_exact_fields(obj, {"decision", "calls", "mode"}, "$")
    mode_value = obj["mode"]
    if not isinstance(mode_value, str):
        raise ActionIRValidationError("type_mismatch", "mode must be a string", "$.mode")
    try:
        mode = CallMode(mode_value)
    except ValueError as exc:
        raise ActionIRValidationError(
            "invalid_mode", f"unknown call mode {mode_value!r}", "$.mode"
        ) from exc

    calls_value = obj["calls"]
    if not isinstance(calls_value, tuple):
        raise ActionIRValidationError("type_mismatch", "calls must be an array", "$.calls")
    if not calls_value:
        raise ActionIRValidationError(
            "empty_calls", "CALL and CONFIRM require at least one call", "$.calls"
        )
    if mode is CallMode.SINGLE and len(calls_value) != 1:
        raise ActionIRValidationError(
            "invalid_call_count", "SINGLE mode requires exactly one call", "$.calls"
        )

    calls: list[ToolCall] = []
    for index, raw_call in enumerate(calls_value):
        path = f"$.calls[{index}]"
        if not isinstance(raw_call, Mapping):
            raise ActionIRValidationError("type_mismatch", "call must be an object", path)
        _expect_exact_fields(raw_call, {"tool", "args"}, path)
        tool = raw_call["tool"]
        if not isinstance(tool, str):
            raise ActionIRValidationError("type_mismatch", "tool must be a string", f"{path}.tool")
        tool = _nfc(tool, path=f"{path}.tool", error_type=ActionIRValidationError)
        schema = registry.get(tool)
        if schema is None:
            raise ActionIRValidationError("unknown_tool", f"unknown tool {tool!r}", f"{path}.tool")
        args = raw_call["args"]
        if not isinstance(args, Mapping):
            raise ActionIRValidationError("type_mismatch", "args must be an object", f"{path}.args")
        missing_args = schema.required.difference(args)
        if missing_args:
            raise ActionIRValidationError(
                "missing_argument",
                f"missing required arguments: {sorted(missing_args)!r}",
                f"{path}.args",
            )
        extra_args = set(args).difference(schema.arguments)
        if extra_args and not schema.additional_arguments:
            raise ActionIRValidationError(
                "extra_argument",
                f"unexpected arguments: {sorted(extra_args)!r}",
                f"{path}.args",
            )
        canonical_args: dict[str, CanonicalJSONValue] = {}
        for name in sorted(args):
            arg_path = f"{path}.args.{name}"
            if name in schema.arguments:
                canonical_args[name] = _validate_value(args[name], schema.arguments[name], arg_path)
            else:
                canonical_args[name] = _canonicalize_untyped(
                    args[name], path=arg_path, error_type=ActionIRValidationError
                )
        calls.append(ToolCall(tool=tool, args=MappingProxyType(canonical_args)))

    return ActionIR(decision=decision, calls=tuple(calls), mode=mode)


def parse_action_ir(
    raw: str,
    schemas: Mapping[str, ToolSchema] | Iterable[ToolSchema],
) -> ActionIR:
    """Strictly decode, validate, and canonicalize one Action IR output."""

    return validate_action_ir(decode_json_object(raw), schemas)


def action_ir_equal(left: ActionIR, right: ActionIR) -> bool:
    """Strict AST equality with the protocol's SERIAL/PARALLEL semantics."""

    if left.decision is not right.decision:
        return False
    if left.decision is Decision.ABSTAIN:
        return True
    if left.decision is Decision.CLARIFY:
        return left.missing == right.missing
    if left.mode is not right.mode:
        return False
    left_calls = [call.canonical_json() for call in left.calls]
    right_calls = [call.canonical_json() for call in right.calls]
    if left.mode is CallMode.PARALLEL:
        return Counter(left_calls) == Counter(right_calls)
    return left_calls == right_calls


__all__ = [
    "MAX_JSON_NESTING",
    "ActionIR",
    "ActionIRError",
    "ActionIRParseError",
    "ActionIRValidationError",
    "CallMode",
    "CanonicalJSONValue",
    "Decision",
    "JSONScalar",
    "JSONType",
    "JSONValue",
    "ToolCall",
    "ToolSchema",
    "ValueSchema",
    "action_ir_equal",
    "canonical_json_value",
    "decode_json_object",
    "parse_action_ir",
    "validate_action_ir",
]
