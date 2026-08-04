"""Structural schema-name partitions intended for pre-authoring freeze in GVS-v1.

This module freezes exact, role-disjoint name-only schema presentations for ``T-new``,
``D-support``, ``S-new``, and ``C-new``.  Each family covers the complete canonical
personal-action simulator schema.  The existing schema-presentation contract proves the tool and
argument bijections and lowers representative presented actions back to byte-identical canonical
Action IR.

The resulting claim is intentionally narrow: exact names are disjoint across roles while the
semantic tool and argument mapping remains frozen.  Without external temporal custody and binding
to materialized populations, this is not evidence that any name was actually held out.  It is also
not evidence for a genuinely new tool or argument semantics, model quality, or authorization to
access models, labels, CUDA, or JarvisLabs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

from barunlm.datasets.mobile_actions import ToolArgument, ToolDefinition

from . import action_simulator as _simulator_module
from . import gvs_schema_presentation as _presentation_module
from . import sim_program as _sim_program_module
from .action_ir import JSONType, ValueSchema, action_ir_equal, parse_action_ir
from .action_simulator import (
    SIMULATOR_SCHEMA_SHA256,
    SIMULATOR_TOOL_SCHEMAS,
    action_simulator_runtime_sha256,
    assert_action_simulator_runtime_integrity,
)
from .gvs_schema_presentation import (
    RENAMED_PRESENTATION,
    ArgumentNameMap,
    SchemaPresentation,
    SchemaPresentationReceipt,
    ToolNameMap,
    build_schema_presentation,
    lower_presented_action_ir,
    verify_schema_presentation,
)
from .sim_program import SimProgramError, module_runtime_sha256, runtime_callable_identity

SCHEMA_PARTITION_SPEC_VERSION = "barun-gvs-schema-partition-spec-v1"
SCHEMA_PARTITION_RECEIPT_VERSION = "barun-gvs-schema-partition-receipt-v1"
SCHEMA_PARTITION_ARTIFACT_VERSION = "barun-gvs-schema-partition-artifact-v1"
SCHEMA_PARTITION_RUNTIME_VERSION = "barun-gvs-schema-partition-runtime-v1"

SCHEMA_PARTITION_ROLES = ("T-new", "D-support", "S-new", "C-new")
MAX_FAMILIES_PER_ROLE = 8
MAX_TOTAL_FAMILIES = len(SCHEMA_PARTITION_ROLES) * MAX_FAMILIES_PER_ROLE
MAX_LINEAGES_PER_FAMILY = 64
MAX_TOTAL_LINEAGE_MEMBERSHIPS = 2_048
MAX_ARTIFACT_BYTES = 16 * 1_024 * 1_024
MAX_SOURCE_BYTES = 4 * 1_024 * 1_024
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 1_000_000
MAX_JSON_STRING_BYTES = 65_536
MAX_JSON_INTEGER_ABS = 2**63 - 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
_ROLE_INDEX = {role: index for index, role in enumerate(SCHEMA_PARTITION_ROLES)}

_SPEC_FIELDS = frozenset(
    {
        "assignment_phase",
        "contains_label_text",
        "contains_request_text",
        "families",
        "request_authoring_started",
        "schema_version",
        "source_commitment_sha256",
    }
)
_FAMILY_FIELDS = frozenset(
    {
        "family_id",
        "family_source_sha256",
        "lineage_ids",
        "presentation",
        "role",
    }
)
_PRESENTATION_FIELDS = frozenset({"mode", "presentation_id", "schema_version", "tools"})
_TOOL_MAP_FIELDS = frozenset({"arguments", "canonical_name", "presented_name"})
_ARGUMENT_MAP_FIELDS = frozenset({"canonical_name", "presented_name"})
_ARTIFACT_FIELDS = frozenset({"artifact_schema_version", "partition", "partition_sha256"})
_ROUND_TRIP_FIELDS = frozenset(
    {
        "canonical_action_sha256",
        "canonical_tool",
        "lowered_action_sha256",
        "presented_action_sha256",
        "presented_tool",
        "semantic_equal",
    }
)

_TOOL_DESCRIPTIONS = {
    "calendar_create": "Create a calendar event with a title and bounded start and end times.",
    "calendar_reschedule": "Move an existing calendar event to bounded start and end times.",
    "contact_lookup": "Look up one contact by its stable identifier.",
    "list_add_item": "Add one identified text item to an identified list.",
    "list_check_item": "Set the checked state of one identified list item.",
    "map_route": "Find a route between two known places using a requested travel mode.",
    "media_pause": "Pause the current simulated media session.",
    "media_play": "Play one catalog track by its stable identifier.",
    "message_send": "Propose a message to one contact over a specified channel.",
    "note_create": "Create a note with an identifier, title, and body.",
    "reminder_create": "Create a reminder with an identifier, title, and due time.",
    "reminder_update": "Update an existing reminder's title and due time.",
    "setting_set_bool": "Set one allowlisted boolean device setting.",
}
_ARGUMENT_DESCRIPTIONS = {
    "body": "Message or note body text.",
    "channel": "Delivery channel name.",
    "checked": "Requested checked state.",
    "contact_id": "Stable contact identifier.",
    "destination_id": "Stable destination-place identifier.",
    "due_at": "Offset-aware ISO due timestamp.",
    "enabled": "Requested boolean setting value.",
    "end_at": "Offset-aware ISO end timestamp.",
    "event_id": "Stable calendar-event identifier.",
    "item_id": "Stable list-item identifier.",
    "key": "Allowlisted setting key.",
    "list_id": "Stable list identifier.",
    "message_id": "Stable message identifier.",
    "mode": "Travel mode.",
    "note_id": "Stable note identifier.",
    "origin_id": "Stable origin-place identifier.",
    "reminder_id": "Stable reminder identifier.",
    "start_at": "Offset-aware ISO start timestamp.",
    "text": "List-item text.",
    "title": "Human-readable title.",
    "to_contact_id": "Stable recipient-contact identifier.",
    "track_id": "Stable catalog-track identifier.",
}
_SAMPLE_VALUES = {
    "body": "Sample body",
    "channel": "sms",
    "checked": True,
    "contact_id": "contact.sample",
    "destination_id": "place.office",
    "due_at": "2026-08-05T10:00:00+05:30",
    "enabled": False,
    "end_at": "2026-08-05T11:00:00+05:30",
    "event_id": "event.sample",
    "item_id": "item.sample",
    "key": "wifi",
    "list_id": "list.sample",
    "message_id": "message.sample",
    "mode": "walking",
    "note_id": "note.sample",
    "origin_id": "place.home",
    "reminder_id": "reminder.sample",
    "start_at": "2026-08-05T10:00:00+05:30",
    "text": "Sample item",
    "title": "Sample title",
    "to_contact_id": "contact.sample",
    "track_id": "track.sample",
}

_HASHLIB_SHA256 = hashlib.sha256
_JSON_DUMPS = json.dumps
_JSON_LOADS = json.loads
_JSON_DECODE_ERROR = json.JSONDecodeError
_OS_FSTAT = os.fstat
_PATH_OPEN = Path.open
_MODULE_RUNTIME_SHA256 = module_runtime_sha256
_RUNTIME_CALLABLE_IDENTITY = runtime_callable_identity
_ACTION_SIMULATOR_RUNTIME_SHA256 = action_simulator_runtime_sha256
_ASSERT_ACTION_SIMULATOR_RUNTIME_INTEGRITY = assert_action_simulator_runtime_integrity
_BUILD_SCHEMA_PRESENTATION = build_schema_presentation
_VERIFY_SCHEMA_PRESENTATION = verify_schema_presentation
_LOWER_PRESENTED_ACTION_IR = lower_presented_action_ir
_PARSE_ACTION_IR = parse_action_ir
_ACTION_IR_EQUAL = action_ir_equal

_FAMILY_TOKEN = object()
_SPEC_TOKEN = object()
_FAMILY_RECEIPT_TOKEN = object()
_PARTITION_RECEIPT_TOKEN = object()
_PINNED_SOURCE_FILES_SHA256 = ""
_PINNED_RUNTIME_SHA256 = ""


class GVSSchemaPartitionError(ValueError):
    """The schema-family partition is malformed, overlapping, or runtime-substituted."""


def _require_string(value: object, *, path: str, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise GVSSchemaPartitionError(f"{path} must be an exact bounded string")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise GVSSchemaPartitionError(f"{path} must be valid UTF-8") from error
    if size > MAX_JSON_STRING_BYTES:
        raise GVSSchemaPartitionError(f"{path} exceeds the string byte bound")
    return value


def _require_identifier(value: object, *, path: str) -> str:
    value = _require_string(value, path=path)
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise GVSSchemaPartitionError(f"{path} must be a bounded ASCII identifier")
    return value


def _require_sha256(value: object, *, path: str) -> str:
    value = _require_string(value, path=path)
    if _SHA256_RE.fullmatch(value) is None:
        raise GVSSchemaPartitionError(f"{path} must be a lowercase SHA-256")
    return value


def _canonical_bytes(value: object, *, limit: int = MAX_ARTIFACT_BYTES) -> bytes:
    try:
        payload = _JSON_DUMPS(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as error:
        raise GVSSchemaPartitionError("schema partition is not strict JSON") from error
    if len(payload) > limit:
        raise GVSSchemaPartitionError("schema partition exceeds the artifact byte bound")
    return payload


def _sha256(value: object) -> str:
    return _HASHLIB_SHA256(_canonical_bytes(value)).hexdigest()


def _exact_object(value: object, *, fields: frozenset[str], path: str) -> dict[str, object]:
    if type(value) is not dict:
        raise GVSSchemaPartitionError(f"{path} must be an exact object")
    if any(type(key) is not str for key in value):
        raise GVSSchemaPartitionError(f"{path} keys must be exact strings")
    keys = set(value)
    if keys != fields:
        raise GVSSchemaPartitionError(
            f"{path} fields are not exact; missing={sorted(fields - keys)!r}, "
            f"extra={sorted(keys - fields)!r}"
        )
    return value


def _file_sha256(path: Path) -> str:
    try:
        with _PATH_OPEN(path, "rb") as handle:
            before = _OS_FSTAT(handle.fileno())
            payload = handle.read(MAX_SOURCE_BYTES + 1)
            after = _OS_FSTAT(handle.fileno())
    except OSError as error:
        raise GVSSchemaPartitionError(f"could not snapshot source {path.name!r}") from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or len(payload) != before.st_size:
        raise GVSSchemaPartitionError(f"source {path.name!r} changed while hashing")
    if len(payload) > MAX_SOURCE_BYTES:
        raise GVSSchemaPartitionError(f"source {path.name!r} exceeds the source byte bound")
    return _HASHLIB_SHA256(payload).hexdigest()


def _module_path(module: object, *, label: str) -> Path:
    raw = getattr(module, "__file__", None)
    if type(raw) is not str:
        raise GVSSchemaPartitionError(f"{label} has no exact source path")
    return Path(raw).resolve(strict=True)


def _source_files() -> dict[str, str]:
    return {
        "action_simulator": _file_sha256(_module_path(_simulator_module, label="simulator")),
        "gvs_schema_partition": _file_sha256(Path(__file__).resolve(strict=True)),
        "gvs_schema_presentation": _file_sha256(
            _module_path(_presentation_module, label="schema presentation")
        ),
    }


def _assert_import_aliases() -> None:
    aliases = {
        "ArgumentNameMap": (ArgumentNameMap, _presentation_module.ArgumentNameMap),
        "JSONType": (JSONType, _presentation_module.JSONType),
        "SchemaPresentation": (SchemaPresentation, _presentation_module.SchemaPresentation),
        "SchemaPresentationReceipt": (
            SchemaPresentationReceipt,
            _presentation_module.SchemaPresentationReceipt,
        ),
        "SIMULATOR_TOOL_SCHEMAS": (
            SIMULATOR_TOOL_SCHEMAS,
            _simulator_module.SIMULATOR_TOOL_SCHEMAS,
        ),
        "ToolArgument": (ToolArgument, _presentation_module.ToolArgument),
        "ToolDefinition": (ToolDefinition, _presentation_module.ToolDefinition),
        "ToolNameMap": (ToolNameMap, _presentation_module.ToolNameMap),
        "ValueSchema": (ValueSchema, _presentation_module.ValueSchema),
        "action_ir_equal": (_ACTION_IR_EQUAL, _presentation_module.action_ir_equal),
        "action_simulator_runtime_sha256": (
            _ACTION_SIMULATOR_RUNTIME_SHA256,
            _simulator_module.action_simulator_runtime_sha256,
        ),
        "assert_action_simulator_runtime_integrity": (
            _ASSERT_ACTION_SIMULATOR_RUNTIME_INTEGRITY,
            _simulator_module.assert_action_simulator_runtime_integrity,
        ),
        "build_schema_presentation": (
            _BUILD_SCHEMA_PRESENTATION,
            _presentation_module.build_schema_presentation,
        ),
        "verify_schema_presentation": (
            _VERIFY_SCHEMA_PRESENTATION,
            _presentation_module.verify_schema_presentation,
        ),
        "lower_presented_action_ir": (
            _LOWER_PRESENTED_ACTION_IR,
            _presentation_module.lower_presented_action_ir,
        ),
        "module_runtime_sha256": (
            _MODULE_RUNTIME_SHA256,
            _sim_program_module.module_runtime_sha256,
        ),
        "parse_action_ir": (_PARSE_ACTION_IR, _presentation_module.parse_action_ir),
        "runtime_callable_identity": (
            _RUNTIME_CALLABLE_IDENTITY,
            _sim_program_module.runtime_callable_identity,
        ),
    }
    for name, (imported, live) in aliases.items():
        if imported is not live:
            raise GVSSchemaPartitionError(f"runtime dependency {name} changed identity")
    if SIMULATOR_SCHEMA_SHA256 != _simulator_module.SIMULATOR_SCHEMA_SHA256:
        raise GVSSchemaPartitionError("simulator schema hash alias changed")
    support_aliases = {
        "hashlib.sha256": (_HASHLIB_SHA256, hashlib.sha256),
        "json.JSONDecodeError": (_JSON_DECODE_ERROR, json.JSONDecodeError),
        "json.dumps": (_JSON_DUMPS, json.dumps),
        "json.loads": (_JSON_LOADS, json.loads),
        "os.fstat": (_OS_FSTAT, os.fstat),
        "Path.open": (_PATH_OPEN, Path.open),
    }
    for name, (captured, live) in support_aliases.items():
        if captured is not live:
            raise GVSSchemaPartitionError(f"runtime support dependency {name} changed identity")
    if RENAMED_PRESENTATION != _presentation_module.RENAMED_PRESENTATION:
        raise GVSSchemaPartitionError("renamed-presentation constant changed")
    if type(_IDENTIFIER_RE) is not re.Pattern or type(_SHA256_RE) is not re.Pattern:
        raise GVSSchemaPartitionError("schema-partition regular expression runtime changed")


def schema_partition_runtime_sha256() -> str:
    """Bind this source/runtime and the exact simulator/presentation dependencies."""

    _assert_import_aliases()
    dependencies: dict[str, object] = {}
    for name, value in (
        ("action_ir_equal", _ACTION_IR_EQUAL),
        ("build_schema_presentation", _BUILD_SCHEMA_PRESENTATION),
        ("hashlib_sha256", _HASHLIB_SHA256),
        ("json_dumps", _JSON_DUMPS),
        ("json_loads", _JSON_LOADS),
        ("lower_presented_action_ir", _LOWER_PRESENTED_ACTION_IR),
        ("module_runtime_sha256", _MODULE_RUNTIME_SHA256),
        ("os_fstat", _OS_FSTAT),
        ("parse_action_ir", _PARSE_ACTION_IR),
        ("path_open", _PATH_OPEN),
        ("runtime_callable_identity", _RUNTIME_CALLABLE_IDENTITY),
        ("verify_schema_presentation", _VERIFY_SCHEMA_PRESENTATION),
    ):
        try:
            dependencies[name] = dict(_RUNTIME_CALLABLE_IDENTITY(value))
        except SimProgramError as error:
            raise GVSSchemaPartitionError(f"could not bind dependency {name}") from error
    contract = {
        "bounds": {
            "artifact_bytes": MAX_ARTIFACT_BYTES,
            "families_per_role": MAX_FAMILIES_PER_ROLE,
            "integer_abs": MAX_JSON_INTEGER_ABS,
            "json_depth": MAX_JSON_DEPTH,
            "json_nodes": MAX_JSON_NODES,
            "lineages_per_family": MAX_LINEAGES_PER_FAMILY,
            "source_bytes": MAX_SOURCE_BYTES,
            "string_bytes": MAX_JSON_STRING_BYTES,
            "total_families": MAX_TOTAL_FAMILIES,
            "total_lineage_memberships": MAX_TOTAL_LINEAGE_MEMBERSHIPS,
        },
        "canonical_support": {
            "argument_descriptions": dict(sorted(_ARGUMENT_DESCRIPTIONS.items())),
            "artifact_fields": sorted(_ARTIFACT_FIELDS),
            "argument_map_fields": sorted(_ARGUMENT_MAP_FIELDS),
            "family_fields": sorted(_FAMILY_FIELDS),
            "identifier_pattern": _IDENTIFIER_RE.pattern,
            "identifier_pattern_flags": _IDENTIFIER_RE.flags,
            "presentation_fields": sorted(_PRESENTATION_FIELDS),
            "role_index": dict(sorted(_ROLE_INDEX.items())),
            "round_trip_fields": sorted(_ROUND_TRIP_FIELDS),
            "sample_values": dict(sorted(_SAMPLE_VALUES.items())),
            "sha256_pattern": _SHA256_RE.pattern,
            "sha256_pattern_flags": _SHA256_RE.flags,
            "spec_fields": sorted(_SPEC_FIELDS),
            "tool_descriptions": dict(sorted(_TOOL_DESCRIPTIONS.items())),
            "tool_map_fields": sorted(_TOOL_MAP_FIELDS),
        },
        "dependencies": dependencies,
        "roles": list(SCHEMA_PARTITION_ROLES),
        "schema_versions": {
            "artifact": SCHEMA_PARTITION_ARTIFACT_VERSION,
            "receipt": SCHEMA_PARTITION_RECEIPT_VERSION,
            "runtime": SCHEMA_PARTITION_RUNTIME_VERSION,
            "spec": SCHEMA_PARTITION_SPEC_VERSION,
        },
        "simulator_runtime_sha256": _ACTION_SIMULATOR_RUNTIME_SHA256(),
        "simulator_schema_sha256": SIMULATOR_SCHEMA_SHA256,
        "source_files_sha256": _sha256(_source_files()),
    }
    try:
        return _MODULE_RUNTIME_SHA256(
            globals(),
            module_name=__name__,
            source_path=__file__,
            contract=contract,
        )
    except SimProgramError as error:
        raise GVSSchemaPartitionError("could not bind schema-partition runtime") from error


def _assert_runtime_integrity() -> tuple[dict[str, str], str, str]:
    _assert_import_aliases()
    try:
        _ASSERT_ACTION_SIMULATOR_RUNTIME_INTEGRITY()
    except Exception as error:
        raise GVSSchemaPartitionError("simulator runtime integrity failed") from error
    source_files = _source_files()
    source_sha256 = _sha256(source_files)
    runtime_sha256 = schema_partition_runtime_sha256()
    if source_sha256 != _PINNED_SOURCE_FILES_SHA256:
        raise GVSSchemaPartitionError("schema-partition source differs from import-time identity")
    if runtime_sha256 != _PINNED_RUNTIME_SHA256:
        raise GVSSchemaPartitionError("schema-partition runtime differs from import-time identity")
    return source_files, source_sha256, runtime_sha256


def _presentation_from_record(value: object, *, path: str) -> SchemaPresentation:
    row = _exact_object(value, fields=_PRESENTATION_FIELDS, path=path)
    raw_tools = row["tools"]
    if type(raw_tools) is not list or not raw_tools or len(raw_tools) > 256:
        raise GVSSchemaPartitionError(f"{path}.tools must be a bounded nonempty array")
    tools: list[ToolNameMap] = []
    for tool_index, raw_tool in enumerate(raw_tools):
        tool = _exact_object(
            raw_tool,
            fields=_TOOL_MAP_FIELDS,
            path=f"{path}.tools[{tool_index}]",
        )
        raw_arguments = tool["arguments"]
        if type(raw_arguments) is not list or len(raw_arguments) > 256:
            raise GVSSchemaPartitionError(
                f"{path}.tools[{tool_index}].arguments must be a bounded array"
            )
        arguments: list[ArgumentNameMap] = []
        for argument_index, raw_argument in enumerate(raw_arguments):
            argument = _exact_object(
                raw_argument,
                fields=_ARGUMENT_MAP_FIELDS,
                path=f"{path}.tools[{tool_index}].arguments[{argument_index}]",
            )
            arguments.append(
                ArgumentNameMap(
                    canonical_name=argument["canonical_name"],
                    presented_name=argument["presented_name"],
                )
            )
        tools.append(
            ToolNameMap(
                canonical_name=tool["canonical_name"],
                presented_name=tool["presented_name"],
                arguments=tuple(arguments),
            )
        )
    try:
        return SchemaPresentation(
            presentation_id=row["presentation_id"],
            mode=row["mode"],
            tools=tuple(tools),
            schema_version=row["schema_version"],
        )
    except (TypeError, ValueError) as error:
        raise GVSSchemaPartitionError(f"{path} is not a valid schema presentation") from error


@dataclass(frozen=True, slots=True)
class SchemaFamilySpec:
    """One exact rename-map family assigned to one population role."""

    role: str
    family_id: str
    lineage_ids: tuple[str, ...]
    family_source_sha256: str
    presentation: SchemaPresentation
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _FAMILY_TOKEN:
            raise TypeError("SchemaFamilySpec must be created through make_schema_family_spec")
        if type(self.role) is not str or self.role not in _ROLE_INDEX:
            raise GVSSchemaPartitionError("schema family role is unsupported")
        _require_identifier(self.family_id, path="$.family_id")
        _require_sha256(self.family_source_sha256, path="$.family_source_sha256")
        if (
            type(self.lineage_ids) is not tuple
            or not self.lineage_ids
            or len(self.lineage_ids) > MAX_LINEAGES_PER_FAMILY
        ):
            raise GVSSchemaPartitionError("lineage_ids must be a nonempty bounded exact tuple")
        checked = tuple(
            _require_identifier(value, path="$.lineage_ids[]") for value in self.lineage_ids
        )
        if checked != tuple(sorted(set(checked))):
            raise GVSSchemaPartitionError("lineage_ids must be canonical and unique")
        if type(self.presentation) is not SchemaPresentation:
            raise GVSSchemaPartitionError("family presentation must be exact SchemaPresentation")
        if self.presentation.mode != RENAMED_PRESENTATION:
            raise GVSSchemaPartitionError("held-out schema families must use renamed presentation")

    def to_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "family_source_sha256": self.family_source_sha256,
            "lineage_ids": list(self.lineage_ids),
            "presentation": self.presentation.to_record(),
            "role": self.role,
        }

    def sha256(self) -> str:
        return _sha256(self.to_dict())


def _family_from_object(value: object, *, path: str) -> SchemaFamilySpec:
    row = _exact_object(value, fields=_FAMILY_FIELDS, path=path)
    lineages = row["lineage_ids"]
    if type(lineages) is not list:
        raise GVSSchemaPartitionError(f"{path}.lineage_ids must be an array")
    return SchemaFamilySpec(
        role=row["role"],
        family_id=row["family_id"],
        lineage_ids=tuple(lineages),
        family_source_sha256=row["family_source_sha256"],
        presentation=_presentation_from_record(row["presentation"], path=f"{path}.presentation"),
        _factory_token=_FAMILY_TOKEN,
    )


def make_schema_family_spec(
    *,
    role: str,
    family_id: str,
    lineage_ids: tuple[str, ...],
    family_source_sha256: str,
    presentation: SchemaPresentation,
) -> SchemaFamilySpec:
    """Detach one pre-authoring family assignment from its caller-owned objects."""

    provisional = SchemaFamilySpec(
        role=role,
        family_id=family_id,
        lineage_ids=lineage_ids,
        family_source_sha256=family_source_sha256,
        presentation=presentation,
        _factory_token=_FAMILY_TOKEN,
    )
    first = provisional.to_dict()
    detached = _family_from_object(first, path="$.family")
    if provisional.to_dict() != first or detached.to_dict() != first:
        raise GVSSchemaPartitionError("schema family changed during stable snapshot")
    return detached


def _mapping_sets(family: SchemaFamilySpec) -> tuple[set[str], set[str], bytes]:
    names: set[str] = set()
    edges: set[str] = set()
    for tool in family.presentation.tools:
        names.add(tool.presented_name)
        edges.add(f"tool:{tool.canonical_name}->{tool.presented_name}")
        for argument in tool.arguments:
            names.add(argument.presented_name)
            edges.add(
                f"argument:{tool.canonical_name}.{argument.canonical_name}"
                f"->{argument.presented_name}"
            )
    return (
        names,
        edges,
        _canonical_bytes([tool.to_record() for tool in family.presentation.tools]),
    )


def _map_commitments(values: set[bytes]) -> list[str]:
    return sorted(_HASHLIB_SHA256(value).hexdigest() for value in values)


def _validate_partition(families: tuple[SchemaFamilySpec, ...]) -> None:
    if (
        type(families) is not tuple
        or not families
        or len(families) > MAX_TOTAL_FAMILIES
        or any(type(family) is not SchemaFamilySpec for family in families)
    ):
        raise GVSSchemaPartitionError("families must be a bounded nonempty exact tuple")
    expected_order = tuple(
        sorted(families, key=lambda value: (_ROLE_INDEX[value.role], value.family_id))
    )
    if families != expected_order:
        raise GVSSchemaPartitionError("schema families must be in canonical role/family order")
    family_ids = [family.family_id for family in families]
    presentation_ids = [family.presentation.presentation_id for family in families]
    if len(family_ids) != len(set(family_ids)):
        raise GVSSchemaPartitionError("family IDs must be globally unique")
    if len(presentation_ids) != len(set(presentation_ids)):
        raise GVSSchemaPartitionError("presentation IDs must be globally unique")
    family_sources = [family.family_source_sha256 for family in families]
    if len(family_sources) != len(set(family_sources)):
        raise GVSSchemaPartitionError("family source hashes must be globally unique")
    counts = {role: 0 for role in SCHEMA_PARTITION_ROLES}
    for family in families:
        counts[family.role] += 1
    if any(not 1 <= count <= MAX_FAMILIES_PER_ROLE for count in counts.values()):
        raise GVSSchemaPartitionError("every role needs a bounded nonempty schema-family roster")
    if sum(len(family.lineage_ids) for family in families) > MAX_TOTAL_LINEAGE_MEMBERSHIPS:
        raise GVSSchemaPartitionError("total lineage membership exceeds the partition bound")

    by_role = {
        role: tuple(family for family in families if family.role == role)
        for role in SCHEMA_PARTITION_ROLES
    }
    for role, role_families in by_role.items():
        role_lineages = [lineage for family in role_families for lineage in family.lineage_ids]
        if len(role_lineages) != len(set(role_lineages)):
            raise GVSSchemaPartitionError(f"duplicate lineage membership within role {role!r}")
        role_maps = [_mapping_sets(family)[2] for family in role_families]
        if len(role_maps) != len(set(role_maps)):
            raise GVSSchemaPartitionError(f"duplicate complete rename map within role {role!r}")
    for left_index, left_role in enumerate(SCHEMA_PARTITION_ROLES):
        left_lineages = {value for family in by_role[left_role] for value in family.lineage_ids}
        left_names = set().union(*(_mapping_sets(family)[0] for family in by_role[left_role]))
        left_edges = set().union(*(_mapping_sets(family)[1] for family in by_role[left_role]))
        left_maps = {_mapping_sets(family)[2] for family in by_role[left_role]}
        for right_role in SCHEMA_PARTITION_ROLES[left_index + 1 :]:
            right_lineages = {
                value for family in by_role[right_role] for value in family.lineage_ids
            }
            right_names = set().union(*(_mapping_sets(family)[0] for family in by_role[right_role]))
            right_edges = set().union(*(_mapping_sets(family)[1] for family in by_role[right_role]))
            right_maps = {_mapping_sets(family)[2] for family in by_role[right_role]}
            if overlap := left_lineages.intersection(right_lineages):
                raise GVSSchemaPartitionError(f"cross-role lineage overlap: {sorted(overlap)!r}")
            if overlap := left_maps.intersection(right_maps):
                raise GVSSchemaPartitionError(
                    f"cross-role complete rename-map overlap: {_map_commitments(overlap)!r}"
                )
            if overlap := left_edges.intersection(right_edges):
                raise GVSSchemaPartitionError(
                    f"cross-role rename-map edge overlap: {sorted(overlap)!r}"
                )
            if overlap := left_names.intersection(right_names):
                raise GVSSchemaPartitionError(
                    f"cross-role presented-name overlap: {sorted(overlap)!r}"
                )


@dataclass(frozen=True, slots=True)
class SchemaPartitionSpec:
    """A structural partition declaring intended pre-authoring role assignment."""

    source_commitment_sha256: str
    families: tuple[SchemaFamilySpec, ...]
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _SPEC_TOKEN:
            raise TypeError(
                "SchemaPartitionSpec must be created through make_schema_partition_spec"
            )
        _require_sha256(self.source_commitment_sha256, path="$.source_commitment_sha256")
        _validate_partition(self.families)

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_phase": "pre_request_authoring",
            "contains_label_text": False,
            "contains_request_text": False,
            "families": [family.to_dict() for family in self.families],
            "request_authoring_started": False,
            "schema_version": SCHEMA_PARTITION_SPEC_VERSION,
            "source_commitment_sha256": self.source_commitment_sha256,
        }

    def canonical_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    def sha256(self) -> str:
        return _HASHLIB_SHA256(self.canonical_json().encode("utf-8")).hexdigest()


def _spec_from_object(value: object, *, path: str = "$.spec") -> SchemaPartitionSpec:
    row = _exact_object(value, fields=_SPEC_FIELDS, path=path)
    if row["schema_version"] != SCHEMA_PARTITION_SPEC_VERSION:
        raise GVSSchemaPartitionError("unsupported schema-partition spec version")
    if row["assignment_phase"] != "pre_request_authoring":
        raise GVSSchemaPartitionError("schema assignment was not frozen pre-authoring")
    for field in (
        "contains_label_text",
        "contains_request_text",
        "request_authoring_started",
    ):
        if row[field] is not False:
            raise GVSSchemaPartitionError(f"{field} must remain false")
    raw_families = row["families"]
    if type(raw_families) is not list or len(raw_families) > MAX_TOTAL_FAMILIES:
        raise GVSSchemaPartitionError("serialized families must be a bounded array")
    families = tuple(
        _family_from_object(value, path=f"{path}.families[{index}]")
        for index, value in enumerate(raw_families)
    )
    return SchemaPartitionSpec(
        source_commitment_sha256=row["source_commitment_sha256"],
        families=families,
        _factory_token=_SPEC_TOKEN,
    )


def make_schema_partition_spec(
    *,
    source_commitment_sha256: str,
    families: tuple[SchemaFamilySpec, ...],
) -> SchemaPartitionSpec:
    """Freeze and detach the complete four-role family roster."""

    if type(families) is not tuple or any(
        type(family) is not SchemaFamilySpec for family in families
    ):
        raise GVSSchemaPartitionError("families must be an exact tuple of SchemaFamilySpec values")
    detached_families: list[SchemaFamilySpec] = []
    for index, family in enumerate(families):
        first = family.to_dict()
        detached = _family_from_object(first, path=f"$.families[{index}]")
        if family.to_dict() != first or detached.to_dict() != first:
            raise GVSSchemaPartitionError(
                f"schema family {index} changed during partition snapshot"
            )
        detached_families.append(detached)
    ordered = tuple(
        sorted(
            detached_families,
            key=lambda value: (_ROLE_INDEX[value.role], value.family_id),
        )
    )
    provisional = SchemaPartitionSpec(
        source_commitment_sha256=source_commitment_sha256,
        families=ordered,
        _factory_token=_SPEC_TOKEN,
    )
    first = provisional.to_dict()
    detached = _spec_from_object(first)
    if provisional.to_dict() != first or detached.to_dict() != first:
        raise GVSSchemaPartitionError("schema partition spec changed during stable snapshot")
    return detached


def _prompt_tools(family: SchemaFamilySpec) -> tuple[ToolDefinition, ...]:
    canonical_by_name = {schema.name: schema for schema in SIMULATOR_TOOL_SCHEMAS}
    tools: list[ToolDefinition] = []
    for tool_map in family.presentation.tools:
        schema = canonical_by_name.get(tool_map.canonical_name)
        if schema is None:
            raise GVSSchemaPartitionError("presentation refers to a non-simulator tool")
        arguments: list[ToolArgument] = []
        for argument_map in tool_map.arguments:
            value_schema = schema.arguments.get(argument_map.canonical_name)
            if value_schema is None:
                raise GVSSchemaPartitionError("presentation refers to a non-simulator argument")
            description = _ARGUMENT_DESCRIPTIONS.get(argument_map.canonical_name)
            if description is None:
                raise GVSSchemaPartitionError("canonical argument lacks a pinned description")
            arguments.append(
                ToolArgument(
                    name=argument_map.presented_name,
                    type_name=value_schema.kind.value,
                    description=description,
                    required=argument_map.canonical_name in schema.required,
                )
            )
        description = _TOOL_DESCRIPTIONS.get(tool_map.canonical_name)
        if description is None:
            raise GVSSchemaPartitionError("canonical tool lacks a pinned description")
        tools.append(
            ToolDefinition(
                name=tool_map.presented_name,
                description=description,
                arguments=tuple(arguments),
            )
        )
    return tuple(tools)


def _sample_value(name: str, schema: ValueSchema) -> object:
    if schema.enum:
        return schema.enum[0]
    value = _SAMPLE_VALUES.get(name)
    if value is None:
        if schema.kind is JSONType.STRING:
            return "sample"
        if schema.kind is JSONType.BOOLEAN:
            return True
        if schema.kind is JSONType.INTEGER:
            return 1
        raise GVSSchemaPartitionError(f"no bounded semantic sample for {name!r}")
    return value


def _round_trip_records(
    family: SchemaFamilySpec,
    receipt: SchemaPresentationReceipt,
    prompt_tools: tuple[ToolDefinition, ...],
) -> tuple[dict[str, object], ...]:
    maps = {tool.canonical_name: tool for tool in family.presentation.tools}
    records: list[dict[str, object]] = []
    for schema in sorted(SIMULATOR_TOOL_SCHEMAS, key=lambda value: value.name):
        tool_map = maps[schema.name]
        argument_names = {
            argument.canonical_name: argument.presented_name for argument in tool_map.arguments
        }
        canonical_args = {
            name: _sample_value(name, value_schema)
            for name, value_schema in sorted(schema.arguments.items())
        }
        presented_args = {argument_names[name]: value for name, value in canonical_args.items()}
        decision = "CONFIRM" if schema.side_effecting else "CALL"
        canonical_payload = {
            "calls": [{"args": canonical_args, "tool": schema.name}],
            "decision": decision,
            "mode": "SINGLE",
        }
        presented_payload = {
            "calls": [{"args": presented_args, "tool": tool_map.presented_name}],
            "decision": decision,
            "mode": "SINGLE",
        }
        canonical = _PARSE_ACTION_IR(
            _canonical_bytes(canonical_payload).decode("utf-8"),
            SIMULATOR_TOOL_SCHEMAS,
        )
        presented = _PARSE_ACTION_IR(
            _canonical_bytes(presented_payload).decode("utf-8"),
            receipt.presented_schemas,
        )
        lowered = _LOWER_PRESENTED_ACTION_IR(
            presented,
            receipt=receipt,
            canonical_schemas=SIMULATOR_TOOL_SCHEMAS,
            prompt_tools=prompt_tools,
            presentation=family.presentation,
        )
        if not _ACTION_IR_EQUAL(canonical, lowered):
            raise GVSSchemaPartitionError(
                f"semantic name-only round trip failed for {schema.name!r}"
            )
        records.append(
            {
                "canonical_action_sha256": _HASHLIB_SHA256(
                    canonical.canonical_json().encode("utf-8")
                ).hexdigest(),
                "canonical_tool": schema.name,
                "lowered_action_sha256": _HASHLIB_SHA256(
                    lowered.canonical_json().encode("utf-8")
                ).hexdigest(),
                "presented_action_sha256": _HASHLIB_SHA256(
                    presented.canonical_json().encode("utf-8")
                ).hexdigest(),
                "presented_tool": tool_map.presented_name,
                "semantic_equal": True,
            }
        )
    return tuple(records)


def _freeze_round_trip_records(
    family: SchemaFamilySpec,
    values: tuple[Mapping[str, object], ...],
) -> tuple[Mapping[str, object], ...]:
    expected_tools = tuple(sorted(schema.name for schema in SIMULATOR_TOOL_SCHEMAS))
    if type(values) is not tuple or len(values) != len(expected_tools):
        raise GVSSchemaPartitionError("semantic round-trip record roster is incomplete")
    presented_by_canonical = {
        tool.canonical_name: tool.presented_name for tool in family.presentation.tools
    }
    detached: list[Mapping[str, object]] = []
    observed_tools: list[str] = []
    for index, value in enumerate(values):
        if type(value) not in {dict, MappingProxyType}:
            raise GVSSchemaPartitionError(
                f"semantic round-trip record {index} must be an exact immutable record"
            )
        row = _exact_object(
            dict(value),
            fields=_ROUND_TRIP_FIELDS,
            path=f"$.semantic_round_trip_records[{index}]",
        )
        canonical_tool = _require_identifier(
            row["canonical_tool"],
            path=f"$.semantic_round_trip_records[{index}].canonical_tool",
        )
        presented_tool = _require_identifier(
            row["presented_tool"],
            path=f"$.semantic_round_trip_records[{index}].presented_tool",
        )
        if presented_by_canonical.get(canonical_tool) != presented_tool:
            raise GVSSchemaPartitionError(
                "semantic round-trip record differs from the family rename map"
            )
        for field in (
            "canonical_action_sha256",
            "lowered_action_sha256",
            "presented_action_sha256",
        ):
            _require_sha256(row[field], path=f"$.semantic_round_trip_records[{index}].{field}")
        if row["semantic_equal"] is not True:
            raise GVSSchemaPartitionError("semantic round-trip record is not exact")
        if row["canonical_action_sha256"] != row["lowered_action_sha256"]:
            raise GVSSchemaPartitionError("semantic round-trip hashes differ after lowering")
        observed_tools.append(canonical_tool)
        detached.append(MappingProxyType(dict(row)))
    if tuple(observed_tools) != expected_tools:
        raise GVSSchemaPartitionError(
            "semantic round-trip record roster does not cover every canonical tool exactly"
        )
    return tuple(detached)


def _family_membership_record(
    family: SchemaFamilySpec,
    presentation_receipt: SchemaPresentationReceipt,
) -> dict[str, object]:
    return {
        "family_id": family.family_id,
        "family_source_sha256": family.family_source_sha256,
        "lineage_ids": list(family.lineage_ids),
        "presentation_receipt_sha256": presentation_receipt.sha256,
        "role": family.role,
    }


@dataclass(frozen=True, slots=True)
class SchemaFamilyReceipt:
    family: SchemaFamilySpec
    presentation_receipt: SchemaPresentationReceipt
    family_membership_sha256: str
    semantic_round_trip_records: tuple[Mapping[str, object], ...]
    semantic_round_trip_sha256: str
    tool_bijection_count: int
    argument_bijection_count: int
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _FAMILY_RECEIPT_TOKEN:
            raise TypeError("SchemaFamilyReceipt is factory-only evidence")
        if type(self.family) is not SchemaFamilySpec:
            raise GVSSchemaPartitionError("family receipt contains an invalid family")
        if type(self.presentation_receipt) is not SchemaPresentationReceipt:
            raise GVSSchemaPartitionError("family receipt contains an invalid presentation receipt")
        if (
            self.presentation_receipt.presentation.to_record()
            != self.family.presentation.to_record()
        ):
            raise GVSSchemaPartitionError(
                "family receipt presentation differs from its family mapping"
            )
        _require_sha256(self.family_membership_sha256, path="$.family_membership_sha256")
        _require_sha256(self.semantic_round_trip_sha256, path="$.semantic_round_trip_sha256")
        expected_tools = len(SIMULATOR_TOOL_SCHEMAS)
        expected_arguments = sum(len(schema.arguments) for schema in SIMULATOR_TOOL_SCHEMAS)
        if self.tool_bijection_count != expected_tools:
            raise GVSSchemaPartitionError("tool bijection count differs from simulator roster")
        if self.argument_bijection_count != expected_arguments:
            raise GVSSchemaPartitionError("argument bijection count differs from simulator roster")
        detached_records = _freeze_round_trip_records(
            self.family,
            self.semantic_round_trip_records,
        )
        object.__setattr__(self, "semantic_round_trip_records", detached_records)
        records = [dict(value) for value in detached_records]
        if _sha256(records) != self.semantic_round_trip_sha256:
            raise GVSSchemaPartitionError("semantic round-trip hash differs from retained records")
        expected_membership = _family_membership_record(
            self.family,
            self.presentation_receipt,
        )
        if _sha256(expected_membership) != self.family_membership_sha256:
            raise GVSSchemaPartitionError("family membership hash differs from retained evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "argument_bijection_count": self.argument_bijection_count,
            "family": self.family.to_dict(),
            "family_membership_sha256": self.family_membership_sha256,
            "family_spec_sha256": self.family.sha256(),
            "presentation_receipt": self.presentation_receipt.to_record(),
            "presentation_receipt_sha256": self.presentation_receipt.sha256,
            "semantic_round_trip_count": len(self.semantic_round_trip_records),
            "semantic_round_trip_records": [
                dict(value) for value in self.semantic_round_trip_records
            ],
            "semantic_round_trip_sha256": self.semantic_round_trip_sha256,
            "tool_bijection_count": self.tool_bijection_count,
        }


def _partition_body_dict(
    *,
    spec: SchemaPartitionSpec,
    family_receipts: tuple[SchemaFamilyReceipt, ...],
    membership_sha256: str,
    source_files: Mapping[str, str],
    source_files_sha256: str,
    partition_runtime_sha256: str,
    simulator_runtime_sha256: str,
    presentation_runtime_sha256: str,
) -> dict[str, object]:
    role_memberships = {
        role: [
            {
                "family_id": value.family.family_id,
                "family_membership_sha256": value.family_membership_sha256,
                "presentation_receipt_sha256": value.presentation_receipt.sha256,
            }
            for value in family_receipts
            if value.family.role == role
        ]
        for role in SCHEMA_PARTITION_ROLES
    }
    return {
        "schema_version": SCHEMA_PARTITION_RECEIPT_VERSION,
        "spec": spec.to_dict(),
        "spec_sha256": spec.sha256(),
        "pre_authoring_partition_only": False,
        "pre_authoring_intent_declared": True,
        "partition_contains_request_or_label_text": False,
        "role_assignment_precedes_request_authoring": False,
        "role_assignment_precedes_request_authoring_declared": True,
        "role_assignment_precedes_request_authoring_authenticated": False,
        "external_temporal_custody_authenticated": False,
        "structural_role_disjoint_schema_names_verified": True,
        "structural_role_disjoint_argument_names_verified": True,
        "heldout_schema_names_supported": False,
        "heldout_argument_names_supported": False,
        "heldout_claim_scope": "none_without_external_custody_and_population_binding",
        "novel_semantics_supported": False,
        "semantic_scope": "frozen_name_only_bijections_over_canonical_simulator_semantics",
        "model_performance_measured": False,
        "descriptions_semantically_audited": False,
        "authorizes_model_access": False,
        "authorizes_label_access": False,
        "authorizes_cuda": False,
        "authorizes_jarvislabs": False,
        "authorizes_training": False,
        "authorizes_launch": False,
        "authorizes_execution": False,
        "canonical_simulator_schema_sha256": SIMULATOR_SCHEMA_SHA256,
        "source_files": dict(source_files),
        "source_files_sha256": source_files_sha256,
        "schema_partition_runtime_sha256": partition_runtime_sha256,
        "action_simulator_runtime_sha256": simulator_runtime_sha256,
        "schema_presentation_runtime_sha256": presentation_runtime_sha256,
        "role_memberships": role_memberships,
        "membership_sha256": membership_sha256,
        "family_receipts": [value.to_dict() for value in family_receipts],
    }


@dataclass(frozen=True, slots=True)
class SchemaPartitionReceipt:
    spec: SchemaPartitionSpec
    family_receipts: tuple[SchemaFamilyReceipt, ...]
    membership_sha256: str
    source_files: Mapping[str, str]
    source_files_sha256: str
    schema_partition_runtime_sha256: str
    action_simulator_runtime_sha256: str
    schema_presentation_runtime_sha256: str
    artifact_body_sha256: str
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _PARTITION_RECEIPT_TOKEN:
            raise TypeError("SchemaPartitionReceipt is factory-only evidence")
        if type(self.spec) is not SchemaPartitionSpec:
            raise GVSSchemaPartitionError("partition receipt has an invalid spec")
        if (
            type(self.family_receipts) is not tuple
            or len(self.family_receipts) != len(self.spec.families)
            or any(type(value) is not SchemaFamilyReceipt for value in self.family_receipts)
        ):
            raise GVSSchemaPartitionError("partition family receipt roster is invalid")
        if tuple(value.family for value in self.family_receipts) != self.spec.families:
            raise GVSSchemaPartitionError("partition family receipt membership differs from spec")
        for name in (
            "membership_sha256",
            "source_files_sha256",
            "schema_partition_runtime_sha256",
            "action_simulator_runtime_sha256",
            "schema_presentation_runtime_sha256",
            "artifact_body_sha256",
        ):
            _require_sha256(getattr(self, name), path=f"$.{name}")
        if not isinstance(self.source_files, MappingProxyType):
            raise GVSSchemaPartitionError("source file hashes must be an immutable mapping")
        if set(self.source_files) != {
            "action_simulator",
            "gvs_schema_partition",
            "gvs_schema_presentation",
        }:
            raise GVSSchemaPartitionError("source file hash roster is not exact")
        for name, value in self.source_files.items():
            _require_sha256(value, path=f"$.source_files.{name}")
        if _sha256(dict(self.source_files)) != self.source_files_sha256:
            raise GVSSchemaPartitionError("source file aggregate hash differs")
        if _sha256(self._role_memberships()) != self.membership_sha256:
            raise GVSSchemaPartitionError("partition membership hash differs")
        if _sha256(self.body_dict()) != self.artifact_body_sha256:
            raise GVSSchemaPartitionError("artifact body hash differs from retained evidence")

    @property
    def schema_presentation_receipts_by_role(
        self,
    ) -> Mapping[str, tuple[SchemaPresentationReceipt, ...]]:
        return MappingProxyType(
            {
                role: tuple(
                    value.presentation_receipt
                    for value in self.family_receipts
                    if value.family.role == role
                )
                for role in SCHEMA_PARTITION_ROLES
            }
        )

    def receipts_for_role(self, role: str) -> tuple[SchemaPresentationReceipt, ...]:
        if type(role) is not str or role not in _ROLE_INDEX:
            raise GVSSchemaPartitionError("requested schema-partition role is unsupported")
        return self.schema_presentation_receipts_by_role[role]

    def _role_memberships(self) -> dict[str, list[dict[str, str]]]:
        return {
            role: [
                {
                    "family_id": value.family.family_id,
                    "family_membership_sha256": value.family_membership_sha256,
                    "presentation_receipt_sha256": value.presentation_receipt.sha256,
                }
                for value in self.family_receipts
                if value.family.role == role
            ]
            for role in SCHEMA_PARTITION_ROLES
        }

    def body_dict(self) -> dict[str, object]:
        return _partition_body_dict(
            spec=self.spec,
            family_receipts=self.family_receipts,
            membership_sha256=self.membership_sha256,
            source_files=self.source_files,
            source_files_sha256=self.source_files_sha256,
            partition_runtime_sha256=self.schema_partition_runtime_sha256,
            simulator_runtime_sha256=self.action_simulator_runtime_sha256,
            presentation_runtime_sha256=self.schema_presentation_runtime_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {**self.body_dict(), "artifact_body_sha256": self.artifact_body_sha256}

    def canonical_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    def sha256(self) -> str:
        return _HASHLIB_SHA256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_json_bytes(self) -> bytes:
        return (
            _canonical_bytes(
                {
                    "artifact_schema_version": SCHEMA_PARTITION_ARTIFACT_VERSION,
                    "partition": self.to_dict(),
                    "partition_sha256": self.sha256(),
                }
            )
            + b"\n"
        )


def _build_family_receipt(family: SchemaFamilySpec) -> SchemaFamilyReceipt:
    prompt_tools = _prompt_tools(family)
    try:
        presentation_receipt = _BUILD_SCHEMA_PRESENTATION(
            canonical_schemas=SIMULATOR_TOOL_SCHEMAS,
            prompt_tools=prompt_tools,
            presentation=family.presentation,
        )
        presentation_receipt = _VERIFY_SCHEMA_PRESENTATION(
            presentation_receipt,
            canonical_schemas=SIMULATOR_TOOL_SCHEMAS,
            prompt_tools=prompt_tools,
            presentation=family.presentation,
        )
    except Exception as error:
        raise GVSSchemaPartitionError(
            f"schema family {family.family_id!r} failed presentation verification"
        ) from error
    round_trips = _round_trip_records(family, presentation_receipt, prompt_tools)
    membership = _family_membership_record(family, presentation_receipt)
    return SchemaFamilyReceipt(
        family=family,
        presentation_receipt=presentation_receipt,
        family_membership_sha256=_sha256(membership),
        semantic_round_trip_records=round_trips,
        semantic_round_trip_sha256=_sha256(list(round_trips)),
        tool_bijection_count=len(SIMULATOR_TOOL_SCHEMAS),
        argument_bijection_count=sum(len(schema.arguments) for schema in SIMULATOR_TOOL_SCHEMAS),
        _factory_token=_FAMILY_RECEIPT_TOKEN,
    )


def build_schema_partition(
    spec: SchemaPartitionSpec,
    *,
    expected_spec_sha256: str,
) -> SchemaPartitionReceipt:
    """Build complete nonauthorizing evidence for one committed pre-authoring partition."""

    source_files, source_files_sha256, partition_runtime = _assert_runtime_integrity()
    if type(spec) is not SchemaPartitionSpec:
        raise GVSSchemaPartitionError("spec must be exact SchemaPartitionSpec")
    expected_spec_sha256 = _require_sha256(
        expected_spec_sha256,
        path="$.expected_spec_sha256",
    )
    first_spec = spec.to_dict()
    detached_spec = _spec_from_object(first_spec)
    if spec.to_dict() != first_spec or detached_spec.to_dict() != first_spec:
        raise GVSSchemaPartitionError("schema partition spec changed during build snapshot")
    if detached_spec.sha256() != expected_spec_sha256:
        raise GVSSchemaPartitionError("schema partition spec differs from its commitment")
    family_receipts = tuple(_build_family_receipt(family) for family in detached_spec.families)
    if spec.to_dict() != first_spec:
        raise GVSSchemaPartitionError("schema partition spec changed during family verification")
    presentation_runtimes = {
        value.presentation_receipt.runtime_identity_sha256 for value in family_receipts
    }
    if len(presentation_runtimes) != 1:
        raise GVSSchemaPartitionError("schema families were built under different runtimes")
    role_memberships = {
        role: [
            {
                "family_id": value.family.family_id,
                "family_membership_sha256": value.family_membership_sha256,
                "presentation_receipt_sha256": value.presentation_receipt.sha256,
            }
            for value in family_receipts
            if value.family.role == role
        ]
        for role in SCHEMA_PARTITION_ROLES
    }
    membership_sha256 = _sha256(role_memberships)
    simulator_runtime = _ACTION_SIMULATOR_RUNTIME_SHA256()
    presentation_runtime = next(iter(presentation_runtimes))
    immutable_source_files = MappingProxyType(dict(sorted(source_files.items())))
    body = _partition_body_dict(
        spec=detached_spec,
        family_receipts=family_receipts,
        membership_sha256=membership_sha256,
        source_files=immutable_source_files,
        source_files_sha256=source_files_sha256,
        partition_runtime_sha256=partition_runtime,
        simulator_runtime_sha256=simulator_runtime,
        presentation_runtime_sha256=presentation_runtime,
    )
    receipt = SchemaPartitionReceipt(
        spec=detached_spec,
        family_receipts=family_receipts,
        membership_sha256=membership_sha256,
        source_files=immutable_source_files,
        source_files_sha256=source_files_sha256,
        schema_partition_runtime_sha256=partition_runtime,
        action_simulator_runtime_sha256=simulator_runtime,
        schema_presentation_runtime_sha256=presentation_runtime,
        artifact_body_sha256=_sha256(body),
        _factory_token=_PARTITION_RECEIPT_TOKEN,
    )
    source_after, source_after_sha, runtime_after = _assert_runtime_integrity()
    if (
        source_after != source_files
        or source_after_sha != source_files_sha256
        or runtime_after != partition_runtime
    ):
        raise GVSSchemaPartitionError("schema-partition runtime changed during build")
    return receipt


def verify_schema_partition(
    receipt: SchemaPartitionReceipt,
    *,
    spec: SchemaPartitionSpec,
    expected_spec_sha256: str,
) -> SchemaPartitionReceipt:
    """Live-rederive every family receipt and compare exact canonical evidence."""

    _assert_runtime_integrity()
    if type(receipt) is not SchemaPartitionReceipt:
        raise GVSSchemaPartitionError("receipt must be exact SchemaPartitionReceipt")
    first = receipt.canonical_json()
    if receipt.canonical_json() != first:
        raise GVSSchemaPartitionError("schema partition receipt changed during snapshot")
    rebuilt = build_schema_partition(spec, expected_spec_sha256=expected_spec_sha256)
    if receipt.canonical_json() != first:
        raise GVSSchemaPartitionError("schema partition receipt changed during recomputation")
    if rebuilt.canonical_json() != first:
        raise GVSSchemaPartitionError("schema partition differs from complete live recomputation")
    return rebuilt


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GVSSchemaPartitionError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_integer(value: str) -> int:
    if len(value.removeprefix("-")) > 20:
        raise GVSSchemaPartitionError("JSON integer exceeds the digit bound")
    parsed = int(value)
    if abs(parsed) > MAX_JSON_INTEGER_ABS:
        raise GVSSchemaPartitionError("JSON integer exceeds the magnitude bound")
    return parsed


def _reject_float(value: str) -> NoReturn:
    raise GVSSchemaPartitionError(f"floating-point JSON is forbidden: {value!r}")


def _reject_constant(value: str) -> NoReturn:
    raise GVSSchemaPartitionError(f"non-finite JSON constant is forbidden: {value!r}")


def _validate_loaded_json(
    value: object,
    *,
    path: str = "$",
    depth: int = 0,
    budget: list[int],
) -> None:
    if depth > MAX_JSON_DEPTH:
        raise GVSSchemaPartitionError("JSON nesting exceeds the schema-partition bound")
    budget[0] += 1
    if budget[0] > MAX_JSON_NODES:
        raise GVSSchemaPartitionError("JSON node count exceeds the schema-partition bound")
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is str:
        _require_string(value, path=path, nonempty=False)
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_loaded_json(
                child,
                path=f"{path}[{index}]",
                depth=depth + 1,
                budget=budget,
            )
        return
    if type(value) is dict:
        for key, child in value.items():
            _require_string(key, path=f"{path}.<key>")
            _validate_loaded_json(
                child,
                path=f"{path}.{key}",
                depth=depth + 1,
                budget=budget,
            )
        return
    raise GVSSchemaPartitionError(f"{path} contains a non-JSON value")


def _loads_bounded_object(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > MAX_ARTIFACT_BYTES:
        raise GVSSchemaPartitionError("artifact must be nonempty bounded exact bytes")
    try:
        value = _JSON_LOADS(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_int=_parse_integer,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except GVSSchemaPartitionError:
        raise
    except (_JSON_DECODE_ERROR, UnicodeError, RecursionError, ValueError) as error:
        raise GVSSchemaPartitionError("artifact is not one strict UTF-8 JSON value") from error
    if type(value) is not dict:
        raise GVSSchemaPartitionError("artifact top level must be an exact object")
    _validate_loaded_json(value, budget=[0])
    return value


def load_schema_partition_json(
    raw: bytes,
    *,
    expected_partition_sha256: str,
) -> SchemaPartitionReceipt:
    """Strictly load and completely live-rederive a committed partition artifact."""

    _assert_runtime_integrity()
    expected_partition_sha256 = _require_sha256(
        expected_partition_sha256,
        path="$.expected_partition_sha256",
    )
    artifact = _exact_object(
        _loads_bounded_object(raw),
        fields=_ARTIFACT_FIELDS,
        path="$",
    )
    if artifact["artifact_schema_version"] != SCHEMA_PARTITION_ARTIFACT_VERSION:
        raise GVSSchemaPartitionError("unsupported schema-partition artifact version")
    partition = artifact["partition"]
    if type(partition) is not dict:
        raise GVSSchemaPartitionError("serialized partition must be an exact object")
    embedded_sha256 = _require_sha256(
        artifact["partition_sha256"],
        path="$.partition_sha256",
    )
    if embedded_sha256 != expected_partition_sha256:
        raise GVSSchemaPartitionError("partition artifact differs from expected commitment")
    spec = _spec_from_object(partition.get("spec"), path="$.partition.spec")
    rebuilt = build_schema_partition(spec, expected_spec_sha256=spec.sha256())
    if rebuilt.sha256() != embedded_sha256:
        raise GVSSchemaPartitionError("partition hash failed complete live reconstruction")
    if raw != rebuilt.to_json_bytes():
        raise GVSSchemaPartitionError("partition artifact is not exact canonical live evidence")
    return rebuilt


__all__ = [
    "MAX_ARTIFACT_BYTES",
    "MAX_FAMILIES_PER_ROLE",
    "MAX_LINEAGES_PER_FAMILY",
    "MAX_TOTAL_FAMILIES",
    "MAX_TOTAL_LINEAGE_MEMBERSHIPS",
    "SCHEMA_PARTITION_ARTIFACT_VERSION",
    "SCHEMA_PARTITION_RECEIPT_VERSION",
    "SCHEMA_PARTITION_ROLES",
    "SCHEMA_PARTITION_RUNTIME_VERSION",
    "SCHEMA_PARTITION_SPEC_VERSION",
    "GVSSchemaPartitionError",
    "SchemaFamilyReceipt",
    "SchemaFamilySpec",
    "SchemaPartitionReceipt",
    "SchemaPartitionSpec",
    "build_schema_partition",
    "load_schema_partition_json",
    "make_schema_family_spec",
    "make_schema_partition_spec",
    "schema_partition_runtime_sha256",
    "verify_schema_partition",
]


_PINNED_SOURCE_FILES_SHA256 = _sha256(_source_files())
_PINNED_RUNTIME_SHA256 = schema_partition_runtime_sha256()
