"""Strict caller-supplied tool and request contracts for BarunAction-35M."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

from barunlm.evaluation.action_ir import (
    ActionIRError,
    JSONType,
    ToolSchema,
    ValueSchema,
    canonical_json_value,
    decode_json_object,
)

RESERVED_TOKENS = ("<bos>", "<system>", "<user>", "<assistant>", "<eos>")
SCHEMA_CONTRACT_VERSION = "barunaction-tool-schema-v1"


class ContractError(ValueError):
    """A stable, machine-readable caller-contract error."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(f"{path}: {message}")
        self.code = code
        self.message = message
        self.path = path

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "path": self.path, "stage": "input"}


@dataclass(frozen=True, slots=True)
class ToolDeclaration:
    """Prompt metadata paired with the evaluator's canonical typed schema."""

    description: str
    schema: ToolSchema
    argument_order: tuple[str, ...]
    type_labels: Mapping[str, str]

    def render(self) -> str:
        arguments = ", ".join(
            f"{name}:{self.type_labels[name]}{'!' if name in self.schema.required else ''}"
            for name in self.argument_order
        )
        return f"{self.schema.name}({arguments}): {self.description}"


@dataclass(frozen=True, slots=True)
class PreparedInput:
    request: str
    now: str
    context_json: str
    declarations: tuple[ToolDeclaration, ...]
    registry: Mapping[str, ToolSchema]


def _exact_fields(
    value: Mapping[str, Any],
    *,
    required: AbstractSet[str],
    optional: AbstractSet[str] = frozenset(),
    path: str,
) -> None:
    missing = required.difference(value)
    if missing:
        raise ContractError("missing_field", f"missing fields: {sorted(missing)!r}", path)
    unknown = set(value).difference(required | optional)
    if unknown:
        raise ContractError("unknown_field", f"unknown fields: {sorted(unknown)!r}", path)


def _text(value: Any, *, path: str, single_line: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError("type_mismatch", "must be a non-empty string", path)
    normalized = unicodedata.normalize("NFC", value)
    if single_line and ("\n" in normalized or "\r" in normalized):
        raise ContractError("multiline_text", "must be a single-line string", path)
    lowered = normalized.casefold()
    for token in RESERVED_TOKENS:
        if token in lowered:
            raise ContractError("reserved_token", f"contains reserved token {token!r}", path)
    return normalized


def _bool(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        raise ContractError("type_mismatch", "must be a boolean", path)
    return value


def _unique_names(value: Any, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ContractError("type_mismatch", "must be an array", path)
    names = tuple(
        _text(item, path=f"{path}[{index}]", single_line=True) for index, item in enumerate(value)
    )
    if len(names) != len(set(names)):
        raise ContractError("duplicate_name", "must not contain duplicate names", path)
    return names


def _value_schema(value: Any, *, path: str) -> tuple[ValueSchema, str]:
    if not isinstance(value, Mapping):
        raise ContractError("type_mismatch", "value schema must be an object", path)
    _exact_fields(
        value,
        required={"type", "description"},
        optional={
            "enum",
            "items",
            "properties",
            "required",
            "additional_properties",
            "set_semantics",
        },
        path=path,
    )
    type_name = _text(value["type"], path=f"{path}.type", single_line=True)
    try:
        kind = JSONType(type_name)
    except ValueError as error:
        raise ContractError(
            "invalid_type", f"unknown JSON type {type_name!r}", f"{path}.type"
        ) from error
    _text(value["description"], path=f"{path}.description", single_line=True)

    enum: tuple[Any, ...] | None = None
    if "enum" in value:
        raw_enum = value["enum"]
        if not isinstance(raw_enum, list) or not raw_enum:
            raise ContractError("invalid_enum", "enum must be a non-empty array", f"{path}.enum")
        enum = tuple(raw_enum)

    common = {"type", "description", "enum"}
    if kind is JSONType.ARRAY:
        if enum is not None:
            raise ContractError(
                "invalid_enum", "enum values are supported only for scalar schemas", f"{path}.enum"
            )
        allowed = common | {"items", "set_semantics"}
        if set(value) != allowed and set(value) != allowed - {"enum"}:
            raise ContractError(
                "schema_shape",
                "array schema requires items and set_semantics and forbids object fields",
                path,
            )
        item_schema, item_label = _value_schema(value["items"], path=f"{path}.items")
        try:
            schema = ValueSchema(
                kind=kind,
                items=item_schema,
                set_semantics=_bool(value["set_semantics"], path=f"{path}.set_semantics"),
            )
        except (TypeError, ValueError) as error:
            raise ContractError("invalid_schema", str(error), path) from error
        return schema, f"array<{item_label}>"

    if kind is JSONType.OBJECT:
        if enum is not None:
            raise ContractError(
                "invalid_enum", "enum values are supported only for scalar schemas", f"{path}.enum"
            )
        expected = common | {"properties", "required", "additional_properties"}
        if set(value) != expected and set(value) != expected - {"enum"}:
            raise ContractError(
                "schema_shape",
                "object schema requires properties, required, and additional_properties",
                path,
            )
        raw_properties = value["properties"]
        if not isinstance(raw_properties, Mapping):
            raise ContractError(
                "type_mismatch", "properties must be an object", f"{path}.properties"
            )
        properties: dict[str, ValueSchema] = {}
        labels: list[str] = []
        for raw_name, raw_schema in raw_properties.items():
            name = _text(raw_name, path=f"{path}.properties", single_line=True)
            if name in properties:
                raise ContractError(
                    "duplicate_name", f"duplicate property {name!r}", f"{path}.properties"
                )
            child, label = _value_schema(raw_schema, path=f"{path}.properties.{name}")
            properties[name] = child
            labels.append(f"{name}:{label}")
        required = _unique_names(value["required"], path=f"{path}.required")
        try:
            schema = ValueSchema(
                kind=kind,
                properties=properties,
                required=frozenset(required),
                additional_properties=_bool(
                    value["additional_properties"], path=f"{path}.additional_properties"
                ),
            )
        except (TypeError, ValueError) as error:
            raise ContractError("invalid_schema", str(error), path) from error
        return schema, "object{" + ",".join(labels) + "}"

    if set(value).difference(common):
        raise ContractError("schema_shape", "scalar schema contains collection-only fields", path)
    try:
        return ValueSchema(kind=kind, enum=enum), kind.value
    except (TypeError, ValueError) as error:
        raise ContractError("invalid_schema", str(error), path) from error


def parse_tool_declarations(value: Any) -> tuple[ToolDeclaration, ...]:
    """Parse the versioned caller schema into prompt and Action IR representations."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ContractError("type_mismatch", "tool_schemas must be a non-empty array", "$.tools")
    declarations: list[ToolDeclaration] = []
    names: set[str] = set()
    for index, raw_tool in enumerate(value):
        path = f"$.tools[{index}]"
        if not isinstance(raw_tool, Mapping):
            raise ContractError("type_mismatch", "tool declaration must be an object", path)
        _exact_fields(
            raw_tool,
            required={
                "name",
                "description",
                "arguments",
                "required",
                "additional_arguments",
                "side_effecting",
            },
            path=path,
        )
        name = _text(raw_tool["name"], path=f"{path}.name", single_line=True)
        if name in names:
            raise ContractError("duplicate_tool", f"duplicate tool {name!r}", f"{path}.name")
        names.add(name)
        description = _text(raw_tool["description"], path=f"{path}.description", single_line=True)
        raw_arguments = raw_tool["arguments"]
        if not isinstance(raw_arguments, Mapping):
            raise ContractError("type_mismatch", "arguments must be an object", f"{path}.arguments")
        arguments: dict[str, ValueSchema] = {}
        labels: dict[str, str] = {}
        order: list[str] = []
        for raw_name, raw_schema in raw_arguments.items():
            argument_name = _text(raw_name, path=f"{path}.arguments", single_line=True)
            if argument_name in arguments:
                raise ContractError(
                    "duplicate_name", f"duplicate argument {argument_name!r}", f"{path}.arguments"
                )
            schema, label = _value_schema(raw_schema, path=f"{path}.arguments.{argument_name}")
            arguments[argument_name] = schema
            labels[argument_name] = label
            order.append(argument_name)
        required = _unique_names(raw_tool["required"], path=f"{path}.required")
        try:
            schema = ToolSchema(
                name=name,
                arguments=arguments,
                required=frozenset(required),
                additional_arguments=_bool(
                    raw_tool["additional_arguments"], path=f"{path}.additional_arguments"
                ),
                side_effecting=_bool(raw_tool["side_effecting"], path=f"{path}.side_effecting"),
            )
        except (TypeError, ValueError) as error:
            raise ContractError("invalid_schema", str(error), path) from error
        declarations.append(
            ToolDeclaration(
                description=description,
                schema=schema,
                argument_order=tuple(order),
                type_labels=MappingProxyType(labels),
            )
        )
    return tuple(declarations)


def _validate_json_strings(value: Any, *, path: str = "$.context") -> None:
    if isinstance(value, str):
        _text(value, path=path)
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = _text(raw_key, path=path, single_line=True)
            _validate_json_strings(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_strings(child, path=f"{path}[{index}]")
        return
    raise ContractError("type_mismatch", f"unsupported JSON value {type(value).__name__}", path)


def canonical_context(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise ContractError("type_mismatch", "context must be a JSON object", "$.context")
    _validate_json_strings(value)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        canonical = decode_json_object(encoded)
    except (TypeError, ValueError, ActionIRError) as error:
        code = error.code if isinstance(error, ActionIRError) else "invalid_context"
        raise ContractError(code, "context must be strict finite JSON", "$.context") from error
    return canonical_json_value(canonical)


def canonical_now(value: Any) -> str:
    now = _text(value, path="$.now", single_line=True)
    try:
        parsed = datetime.fromisoformat(now.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError("invalid_now", "NOW must be an ISO-8601 timestamp", "$.now") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError("timezone_required", "NOW must include an explicit UTC offset", "$.now")
    return parsed.isoformat()


def prepare_input(
    *,
    request: Any,
    tool_schemas: Any,
    context: Any,
    now: Any,
) -> PreparedInput:
    normalized_request = _text(request, path="$.request")
    if not normalized_request.strip():
        raise ContractError(
            "empty_request", "request must contain non-whitespace text", "$.request"
        )
    declarations = parse_tool_declarations(tool_schemas)
    registry = MappingProxyType({item.schema.name: item.schema for item in declarations})
    return PreparedInput(
        request=normalized_request,
        now=canonical_now(now),
        context_json=canonical_context(context),
        declarations=declarations,
        registry=registry,
    )


__all__ = [
    "SCHEMA_CONTRACT_VERSION",
    "ContractError",
    "PreparedInput",
    "ToolDeclaration",
    "canonical_context",
    "canonical_now",
    "parse_tool_declarations",
    "prepare_input",
]
