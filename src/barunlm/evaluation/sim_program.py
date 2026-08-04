"""Strict, immutable program and world-state contracts for the GVS semantic oracle.

The module is intentionally standard-library-only.  It snapshots caller-owned JSON once,
rejects implicit defaults and non-integral numbers, and has no network, wall-clock,
randomness, subprocess, model, or device surface.  Its sole local-file read is the exact
TZif payload used to construct a pinned :class:`zoneinfo.ZoneInfo`; hashing those bytes is
what makes timestamp semantics portable and auditable instead of trusting an OS tzdata name.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import io
import json
import os
import platform
import re
import sys
import types
import unicodedata
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, TypeAlias
from zoneinfo import TZPATH, ZoneInfo, ZoneInfoNotFoundError

from .action_ir import CallMode, Decision

SIM_PROGRAM_VERSION = "barun-gvs-sim-program-v1"
WORLD_STATE_VERSION = "barun-gvs-world-state-v1"
MAX_SIM_JSON_DEPTH = 48
MAX_SIM_JSON_BYTES = 1_048_576
MAX_SIM_COLLECTION_ITEMS = 4_096
MAX_SIM_STRING_BYTES = 16_384
MAX_SIM_TOTAL_NODES = 100_000
MAX_SIM_INTEGER_ABS = 2**63 - 1
MIN_SIM_YEAR = 1970
MAX_SIM_YEAR = 2100
MAX_TZIF_BYTES = 16_777_216
TIMEZONE_RUNTIME_VERSION = "barun-gvs-timezone-runtime-v3"

StrictScalar: TypeAlias = None | bool | int | str
StrictJSON: TypeAlias = StrictScalar | tuple["StrictJSON", ...] | Mapping[str, "StrictJSON"]

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
_ROUTE_MODES = frozenset({"walking", "driving", "transit", "cycling"})
_MEDIA_STATUSES = frozenset({"stopped", "playing", "paused"})
_BUILTIN_MAPPING_TYPES = (dict, MappingProxyType)
_WORLD_FACTORY_TOKEN = object()
_PROGRAM_FACTORY_TOKEN = object()


@dataclass(slots=True)
class _SnapshotBudget:
    nodes: int = 0
    string_bytes: int = 0

    def consume_node(self, *, path: str) -> None:
        self.nodes += 1
        if self.nodes > MAX_SIM_TOTAL_NODES:
            raise SimProgramError(
                "cardinality_exceeded",
                f"strict JSON contains more than {MAX_SIM_TOTAL_NODES} nodes",
                path,
            )

    def consume_string(self, value: str, *, path: str) -> None:
        size = len(value.encode("utf-8"))
        if size > MAX_SIM_STRING_BYTES:
            raise SimProgramError(
                "string_too_large",
                f"string exceeds {MAX_SIM_STRING_BYTES} UTF-8 bytes",
                path,
            )
        self.string_bytes += size
        if self.string_bytes > MAX_SIM_JSON_BYTES:
            raise SimProgramError(
                "json_too_large",
                f"strict JSON strings exceed {MAX_SIM_JSON_BYTES} UTF-8 bytes",
                path,
            )


class SimProgramError(ValueError):
    """Stable fail-closed program/state validation error."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(f"{path}: {message}")
        self.code = code
        self.path = path


def _stable_source_sha256(path: str) -> str:
    try:
        with Path(path).resolve(strict=True).open("rb") as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read(MAX_SIM_JSON_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise SimProgramError(
            "runtime_identity",
            "could not snapshot simulator source",
        ) from exc
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
        raise SimProgramError("runtime_identity", "simulator source changed while read")
    if len(payload) > MAX_SIM_JSON_BYTES:
        raise SimProgramError("runtime_identity", "simulator source exceeds bounded size")
    return hashlib.sha256(payload).hexdigest()


def _code_constant_identity(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return {"float_hex": value.hex()}
    if type(value) is complex:
        return {"complex_hex": [value.real.hex(), value.imag.hex()]}
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if type(value) is tuple:
        return {"tuple": [_code_constant_identity(child) for child in value]}
    if type(value) is frozenset:
        children = [_code_constant_identity(child) for child in value]
        return {
            "frozenset": sorted(
                children,
                key=lambda child: json.dumps(child, sort_keys=True, separators=(",", ":")),
            )
        }
    if type(value) is types.CodeType:
        return {"code": _code_identity(value)}
    if value is Ellipsis:
        return {"singleton": "Ellipsis"}
    raise SimProgramError(
        "runtime_identity",
        f"unsupported Python code constant {type(value).__name__}",
    )


def _code_identity(code: types.CodeType) -> Mapping[str, object]:
    # CPython 3.11+ specializes live bytecode in place.  ``marshal.dumps(code)`` includes
    # that transient state even though ``co_code`` retains the stable instruction stream.
    # Hash explicit semantic fields instead, recursively including nested code constants.
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode_hex": code.co_code.hex(),
        "constants": [_code_constant_identity(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _callable_code_sha256(value: object) -> str | None:
    code = getattr(value, "__code__", None)
    if code is None:
        return None
    if type(code) is not types.CodeType:
        raise SimProgramError("runtime_identity", "callable exposes a malformed code object")
    digest = hashlib.sha256(
        json.dumps(
            _code_identity(code),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    defaults = getattr(value, "__defaults__", None)
    kwdefaults = getattr(value, "__kwdefaults__", None)
    digest.update(repr(defaults).encode("utf-8"))
    digest.update(repr(kwdefaults).encode("utf-8"))
    return digest.hexdigest()


def runtime_callable_identity(value: object) -> Mapping[str, StrictJSON]:
    """Return stable callable identity for runtime dependency receipts."""

    if not callable(value):
        raise SimProgramError("runtime_identity", "runtime dependency is not callable")
    value_type = type(value)
    record = snapshot_strict_json(
        {
            "module": getattr(value, "__module__", None),
            "qualname": getattr(value, "__qualname__", getattr(value, "__name__", None)),
            "type_module": value_type.__module__,
            "type_qualname": value_type.__qualname__,
            "code_sha256": _callable_code_sha256(value),
        }
    )
    assert isinstance(record, Mapping)
    return record


def _loaded_module_code_sha256(namespace: Mapping[str, object], module_name: str) -> str:
    digest = hashlib.sha256()
    records: list[tuple[str, object]] = []
    try:
        namespace_items = tuple(namespace.items())
    except RuntimeError as exc:
        raise SimProgramError("runtime_identity", "module namespace mutated during audit") from exc
    for name, value in namespace_items:
        if getattr(value, "__module__", None) != module_name:
            continue
        if isinstance(value, types.FunctionType):
            records.append((name, value))
            continue
        if isinstance(value, type):
            for member_name, raw_member in vars(value).items():
                member = raw_member
                if isinstance(raw_member, (classmethod, staticmethod)):
                    member = raw_member.__func__
                elif isinstance(raw_member, property):
                    member = raw_member.fget
                if member is not None and hasattr(member, "__code__"):
                    records.append((f"{name}.{member_name}", member))
    for name, value in sorted(records, key=lambda item: item[0]):
        code_sha = _callable_code_sha256(value)
        if code_sha is None:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(code_sha.encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


def module_runtime_sha256(
    namespace: Mapping[str, object],
    *,
    module_name: str,
    source_path: str,
    contract: Mapping[str, object],
) -> str:
    """Bind source bytes, live Python code, runtime, and an exact module contract."""

    try:
        contract_bytes = json.dumps(
            contract,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SimProgramError("runtime_identity", "runtime contract is not strict JSON") from exc
    record = {
        "module": module_name,
        "module_source_sha256": _stable_source_sha256(source_path),
        "loaded_code_sha256": _loaded_module_code_sha256(namespace, module_name),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nfc(value: str, *, path: str) -> str:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise SimProgramError("invalid_utf8", "string is not valid UTF-8", path) from exc
    return unicodedata.normalize("NFC", value)


def _snapshot(
    value: object,
    *,
    path: str = "$",
    depth: int = 0,
    budget: _SnapshotBudget | None = None,
) -> StrictJSON:
    if budget is None:
        budget = _SnapshotBudget()
    if depth > MAX_SIM_JSON_DEPTH:
        raise SimProgramError(
            "max_nesting_exceeded",
            f"JSON nesting exceeds {MAX_SIM_JSON_DEPTH}",
            path,
        )
    budget.consume_node(path=path)
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if abs(value) > MAX_SIM_INTEGER_ABS:
            raise SimProgramError(
                "integer_out_of_range",
                f"integer magnitude exceeds {MAX_SIM_INTEGER_ABS}",
                path,
            )
        return value
    if isinstance(value, float):
        raise SimProgramError("non_integral_number", "floating-point values are forbidden", path)
    if isinstance(value, str):
        normalized = _nfc(value, path=path)
        budget.consume_string(normalized, path=path)
        return normalized
    if type(value) in _BUILTIN_MAPPING_TYPES:
        if len(value) > MAX_SIM_COLLECTION_ITEMS:
            raise SimProgramError(
                "cardinality_exceeded",
                f"object contains more than {MAX_SIM_COLLECTION_ITEMS} fields",
                path,
            )
        try:
            items = tuple(value.items())
        except RuntimeError as exc:
            raise SimProgramError(
                "concurrent_mutation", "object mutated during snapshot", path
            ) from exc
        output: dict[str, StrictJSON] = {}
        for raw_key, child in items:
            if type(raw_key) is not str:
                raise SimProgramError("type_mismatch", "object key must be a string", path)
            key = _nfc(raw_key, path=path)
            budget.consume_string(key, path=path)
            if key in output:
                raise SimProgramError(
                    "duplicate_key",
                    f"duplicate object key after NFC normalization: {key!r}",
                    path,
                )
            output[key] = _snapshot(
                child,
                path=f"{path}.{key}",
                depth=depth + 1,
                budget=budget,
            )
        return MappingProxyType(dict(sorted(output.items())))
    if type(value) in (list, tuple):
        if len(value) > MAX_SIM_COLLECTION_ITEMS:
            raise SimProgramError(
                "cardinality_exceeded",
                f"array contains more than {MAX_SIM_COLLECTION_ITEMS} items",
                path,
            )
        return tuple(
            _snapshot(
                child,
                path=f"{path}[{index}]",
                depth=depth + 1,
                budget=budget,
            )
            for index, child in enumerate(value)
        )
    raise SimProgramError(
        "type_mismatch",
        f"unsupported strict JSON value {type(value).__name__}",
        path,
    )


def snapshot_strict_json(value: object) -> StrictJSON:
    """Detach a stable strict JSON snapshot from caller-owned objects.

    Exact built-in containers are mutable and Python provides no reader-side lock that
    their owners must honor.  Two consecutive detached traversals therefore have to
    agree byte-for-byte.  A one-way mutation before both traversals or after both has a
    valid linearization point; a mutation observed during the snapshot fails closed
    instead of binding a hybrid value that never existed.
    """

    first = _snapshot(value)
    second = _snapshot(value)
    first_encoded = json.dumps(
        _to_builtin(first),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    second_encoded = json.dumps(
        _to_builtin(second),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if first_encoded != second_encoded:
        raise SimProgramError(
            "concurrent_mutation",
            "strict JSON changed during the stable snapshot",
        )
    if len(first_encoded) > MAX_SIM_JSON_BYTES:
        raise SimProgramError(
            "json_too_large",
            f"canonical JSON exceeds {MAX_SIM_JSON_BYTES} UTF-8 bytes",
        )
    return first


def _to_builtin(value: StrictJSON) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_builtin(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_to_builtin(child) for child in value]
    return value


def canonical_json(value: StrictJSON) -> str:
    value = snapshot_strict_json(value)
    encoded = json.dumps(
        _to_builtin(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > MAX_SIM_JSON_BYTES:
        raise SimProgramError(
            "json_too_large",
            f"canonical JSON exceeds {MAX_SIM_JSON_BYTES} UTF-8 bytes",
        )
    return encoded


def canonical_sha256(value: StrictJSON) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SimProgramError("duplicate_key", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_float(value: str) -> NoReturn:
    raise SimProgramError("non_integral_number", f"floating-point value {value!r} is forbidden")


def _reject_constant(value: str) -> NoReturn:
    raise SimProgramError("non_finite_number", f"non-finite value {value!r} is forbidden")


def _parse_integer(value: str) -> int:
    """Parse only the bounded signed-64-bit integer domain without huge ``int`` work."""

    digits = value.removeprefix("-")
    if len(digits) > 19:
        raise SimProgramError(
            "integer_out_of_range",
            f"integer magnitude exceeds {MAX_SIM_INTEGER_ABS}",
        )
    parsed = int(value)
    if abs(parsed) > MAX_SIM_INTEGER_ABS:
        raise SimProgramError(
            "integer_out_of_range",
            f"integer magnitude exceeds {MAX_SIM_INTEGER_ABS}",
        )
    return parsed


def loads_strict_json_object(text: str) -> Mapping[str, StrictJSON]:
    if not isinstance(text, str):
        raise SimProgramError("type_mismatch", "JSON input must be a string")
    try:
        encoded_size = len(text.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as exc:
        raise SimProgramError("invalid_utf8", "JSON input is not valid UTF-8") from exc
    if encoded_size > MAX_SIM_JSON_BYTES:
        raise SimProgramError(
            "json_too_large",
            f"JSON input exceeds {MAX_SIM_JSON_BYTES} UTF-8 bytes",
        )
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_int=_parse_integer,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except SimProgramError:
        raise
    except (json.JSONDecodeError, UnicodeError, RecursionError, ValueError) as exc:
        raise SimProgramError("invalid_json", "input is not one strict JSON value") from exc
    snapshot = snapshot_strict_json(decoded)
    if not isinstance(snapshot, Mapping):
        raise SimProgramError("top_level_not_object", "top-level value must be an object")
    return snapshot


def _object(value: object, *, path: str, fields: frozenset[str]) -> Mapping[str, StrictJSON]:
    if not isinstance(value, Mapping):
        raise SimProgramError("type_mismatch", "must be an object", path)
    missing = fields.difference(value)
    if missing:
        raise SimProgramError("missing_field", f"missing fields: {sorted(missing)!r}", path)
    extras = set(value).difference(fields)
    if extras:
        raise SimProgramError("unknown_field", f"unknown fields: {sorted(extras)!r}", path)
    return value


def _text(value: object, *, path: str, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise SimProgramError("type_mismatch", f"must be a {qualifier}string", path)
    return _nfc(value, path=path)


def _identifier(value: object, *, path: str) -> str:
    identifier = _text(value, path=path)
    if _IDENTIFIER.fullmatch(identifier) is None:
        raise SimProgramError("invalid_identifier", "must be a stable ASCII identifier", path)
    return identifier


def canonical_identifier(value: object, *, path: str = "$") -> str:
    return _identifier(value, path=path)


def _boolean(value: object, *, path: str) -> bool:
    if type(value) is not bool:
        raise SimProgramError("type_mismatch", "must be a boolean", path)
    return value


def _nonnegative_integer(value: object, *, path: str) -> int:
    if type(value) is not int or value < 0:
        raise SimProgramError("type_mismatch", "must be a non-negative integer", path)
    return value


def _timezone_key(value: object, *, path: str) -> str:
    name = _text(value, path=path)
    parts = name.split("/")
    if (
        not name
        or len(name.encode("utf-8")) > 255
        or name.startswith(("/", "\\"))
        or "\\" in name
        or any(
            character.isspace() or unicodedata.category(character).startswith("C")
            for character in name
        )
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise SimProgramError("unknown_timezone", f"invalid IANA timezone key {name!r}", path)
    return name


def _read_tzif_bytes(name: str) -> tuple[bytes, str, str | None]:
    """Read the exact TZif payload in the same search order as :mod:`zoneinfo`.

    The bytes, rather than a path or a sparse set of offset probes, are the complete rule
    identity.  ``ZoneInfo.from_file`` below consumes this same detached payload, closing the
    race where a separately hashed database could differ from the database actually used.
    """

    parts = name.split("/")
    for raw_root in TZPATH:
        candidate = Path(raw_root).joinpath(*parts)
        try:
            with candidate.open("rb") as handle:
                payload = handle.read(MAX_TZIF_BYTES + 1)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SimProgramError(
                "timezone_source_unreadable",
                f"could not read local timezone rules for {name!r}",
                "$.timezone",
            ) from exc
        return payload, "system_tzpath", None

    try:
        resource = importlib.resources.files("tzdata.zoneinfo").joinpath(*parts)
        with resource.open("rb") as handle:
            payload = handle.read(MAX_TZIF_BYTES + 1)
        try:
            tzdata_version = importlib.metadata.version("tzdata")
        except importlib.metadata.PackageNotFoundError:
            tzdata_version = None
    except (FileNotFoundError, ModuleNotFoundError, ImportError, OSError) as exc:
        raise SimProgramError(
            "unknown_timezone",
            f"unknown IANA timezone {name!r}",
            "$.timezone",
        ) from exc
    return payload, "tzdata_package", tzdata_version


@lru_cache(maxsize=128)
def _timezone_runtime(name: str) -> tuple[ZoneInfo, Mapping[str, StrictJSON]]:
    assert_sim_program_runtime_integrity()
    name = _timezone_key(name, path="$.timezone")
    payload, source_kind, tzdata_version = _read_tzif_bytes(name)
    if len(payload) > MAX_TZIF_BYTES:
        raise SimProgramError(
            "timezone_source_too_large",
            f"timezone rule payload exceeds {MAX_TZIF_BYTES} bytes",
            "$.timezone",
        )
    if not payload.startswith(b"TZif"):
        raise SimProgramError(
            "invalid_timezone_source",
            "timezone rule payload is not a TZif file",
            "$.timezone",
        )
    try:
        zone = ZoneInfo.from_file(io.BytesIO(payload), key=name)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise SimProgramError(
            "invalid_timezone_source",
            f"could not decode timezone rules for {name!r}",
            "$.timezone",
        ) from exc
    identity = snapshot_strict_json(
        {
            "schema_version": TIMEZONE_RUNTIME_VERSION,
            "timezone": name,
            "python_implementation": platform.python_implementation(),
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            "zoneinfo_implementation": f"{ZoneInfo.__module__}.{ZoneInfo.__qualname__}",
            "tzif_sha256": hashlib.sha256(payload).hexdigest(),
            "tzif_bytes": len(payload),
            "source_kind": source_kind,
            "tzdata_version": tzdata_version,
            "year_min": MIN_SIM_YEAR,
            "year_max": MAX_SIM_YEAR,
            "sim_program_runtime_sha256": sim_program_runtime_sha256(),
        }
    )
    assert isinstance(identity, Mapping)
    return zone, identity


def _timezone(value: object, *, path: str) -> str:
    name = _timezone_key(value, path=path)
    try:
        _timezone_runtime(name)
    except SimProgramError as exc:
        raise SimProgramError(exc.code, str(exc).split(": ", 1)[-1], path) from exc
    return name


def _canonical_timestamp(value: object, *, timezone: str, path: str = "$") -> str:
    """Validate and canonicalize one second-precision local timestamp.

    Canonicalization uses the pinned IANA zone's observed offset, so the equivalent
    ``-00:00`` and ``+00:00`` spellings collapse to one byte representation.
    """

    text = _text(value, path=path)
    if _TIMESTAMP.fullmatch(text) is None:
        raise SimProgramError(
            "invalid_timestamp",
            "must be an offset-aware ISO timestamp with second precision",
            path,
        )
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SimProgramError(
            "invalid_timestamp", "timestamp is not a valid date/time", path
        ) from exc
    if not MIN_SIM_YEAR <= parsed.year <= MAX_SIM_YEAR:
        raise SimProgramError(
            "timestamp_year_out_of_range",
            f"timestamp year must be between {MIN_SIM_YEAR} and {MAX_SIM_YEAR}",
            path,
        )
    name = _timezone_key(timezone, path=path)
    try:
        zone, _ = _timezone_runtime(name)
    except SimProgramError as exc:
        raise SimProgramError(exc.code, str(exc).split(": ", 1)[-1], path) from exc
    zoned = parsed.astimezone(zone)
    if (
        zoned.replace(tzinfo=None) != parsed.replace(tzinfo=None)
        or zoned.utcoffset() != parsed.utcoffset()
    ):
        raise SimProgramError(
            "timezone_offset_mismatch",
            "timestamp local time and offset do not match the pinned IANA timezone",
            path,
        )
    return zoned.isoformat(timespec="seconds")


def canonical_timestamp(value: object, *, timezone: str, path: str = "$") -> str:
    """Public audited timestamp canonicalizer."""

    assert_sim_program_runtime_integrity()
    result = _canonical_timestamp(value, timezone=timezone, path=path)
    assert_sim_program_runtime_integrity()
    return result


def _timestamp(value: object, *, timezone: str, path: str) -> str:
    return _canonical_timestamp(value, timezone=timezone, path=path)


def timezone_runtime_identity(timezone: str) -> Mapping[str, StrictJSON]:
    """Return the complete, immutable identity of the TZif rules actually in use."""

    assert_sim_program_runtime_integrity()
    name = _timezone(timezone, path="$.timezone")
    _, identity = _timezone_runtime(name)
    return identity


def timezone_runtime_sha256(timezone: str) -> str:
    return canonical_sha256(timezone_runtime_identity(timezone))


def _entity_table(
    value: object,
    *,
    path: str,
    fields: frozenset[str],
    validator: Any,
) -> Mapping[str, StrictJSON]:
    if not isinstance(value, Mapping):
        raise SimProgramError("type_mismatch", "must be an object keyed by entity ID", path)
    output: dict[str, StrictJSON] = {}
    for raw_id, raw_entity in value.items():
        entity_id = _identifier(raw_id, path=path)
        entity_path = f"{path}.{entity_id}"
        entity = _object(raw_entity, path=entity_path, fields=fields)
        output[entity_id] = validator(entity, entity_path)
    return MappingProxyType(dict(sorted(output.items())))


@dataclass(frozen=True, slots=True)
class WorldState:
    """Immutable, fully materialized state for the bounded personal-action simulator."""

    reference_time: str
    timezone: str
    reminders: Mapping[str, StrictJSON]
    calendar: Mapping[str, StrictJSON]
    contacts: Mapping[str, StrictJSON]
    notes: Mapping[str, StrictJSON]
    lists: Mapping[str, StrictJSON]
    places: Mapping[str, StrictJSON]
    routes: Mapping[str, StrictJSON]
    outbox: tuple[StrictJSON, ...]
    media: Mapping[str, StrictJSON]
    settings: Mapping[str, StrictJSON]
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _WORLD_FACTORY_TOKEN:
            raise TypeError("WorldState must be constructed through WorldState.from_dict")
        mapping_fields = (
            "reminders",
            "calendar",
            "contacts",
            "notes",
            "lists",
            "places",
            "routes",
            "media",
            "settings",
        )
        if any(
            not isinstance(getattr(self, field_name), MappingProxyType)
            for field_name in mapping_fields
        ) or not isinstance(self.outbox, tuple):
            raise TypeError("WorldState factory produced non-immutable state")

    @classmethod
    def from_dict(cls, payload: object) -> WorldState:
        assert_sim_program_runtime_integrity()
        snapshot = snapshot_strict_json(payload)
        obj = _object(
            snapshot,
            path="$.initial_state",
            fields=frozenset(
                {
                    "schema_version",
                    "reference_time",
                    "timezone",
                    "reminders",
                    "calendar",
                    "contacts",
                    "notes",
                    "lists",
                    "places",
                    "routes",
                    "outbox",
                    "media",
                    "settings",
                }
            ),
        )
        if obj["schema_version"] != WORLD_STATE_VERSION:
            raise SimProgramError(
                "schema_version",
                f"must equal {WORLD_STATE_VERSION!r}",
                "$.initial_state.schema_version",
            )
        timezone = _timezone(obj["timezone"], path="$.initial_state.timezone")
        reference_time = _timestamp(
            obj["reference_time"],
            timezone=timezone,
            path="$.initial_state.reference_time",
        )

        def reminder(entity: Mapping[str, StrictJSON], path: str) -> StrictJSON:
            due = entity["due_at"]
            if due is not None:
                due = _timestamp(due, timezone=timezone, path=f"{path}.due_at")
            return _snapshot(
                {
                    "title": _text(entity["title"], path=f"{path}.title"),
                    "due_at": due,
                    "completed": _boolean(entity["completed"], path=f"{path}.completed"),
                }
            )

        reminders = _entity_table(
            obj["reminders"],
            path="$.initial_state.reminders",
            fields=frozenset({"title", "due_at", "completed"}),
            validator=reminder,
        )

        def event(entity: Mapping[str, StrictJSON], path: str) -> StrictJSON:
            start = _timestamp(entity["start_at"], timezone=timezone, path=f"{path}.start_at")
            end = _timestamp(entity["end_at"], timezone=timezone, path=f"{path}.end_at")
            if datetime.fromisoformat(end) <= datetime.fromisoformat(start):
                raise SimProgramError("invalid_interval", "end_at must be after start_at", path)
            return _snapshot(
                {
                    "title": _text(entity["title"], path=f"{path}.title"),
                    "start_at": start,
                    "end_at": end,
                }
            )

        calendar = _entity_table(
            obj["calendar"],
            path="$.initial_state.calendar",
            fields=frozenset({"title", "start_at", "end_at"}),
            validator=event,
        )

        def contact(entity: Mapping[str, StrictJSON], path: str) -> StrictJSON:
            return _snapshot(
                {
                    "name": _text(entity["name"], path=f"{path}.name"),
                    "channel": _text(entity["channel"], path=f"{path}.channel"),
                }
            )

        contacts = _entity_table(
            obj["contacts"],
            path="$.initial_state.contacts",
            fields=frozenset({"name", "channel"}),
            validator=contact,
        )

        def note(entity: Mapping[str, StrictJSON], path: str) -> StrictJSON:
            return _snapshot(
                {
                    "title": _text(entity["title"], path=f"{path}.title"),
                    "body": _text(entity["body"], path=f"{path}.body", nonempty=False),
                }
            )

        notes = _entity_table(
            obj["notes"],
            path="$.initial_state.notes",
            fields=frozenset({"title", "body"}),
            validator=note,
        )

        def list_entity(entity: Mapping[str, StrictJSON], path: str) -> StrictJSON:
            def item(child: Mapping[str, StrictJSON], child_path: str) -> StrictJSON:
                return _snapshot(
                    {
                        "text": _text(child["text"], path=f"{child_path}.text"),
                        "checked": _boolean(child["checked"], path=f"{child_path}.checked"),
                    }
                )

            items = _entity_table(
                entity["items"],
                path=f"{path}.items",
                fields=frozenset({"text", "checked"}),
                validator=item,
            )
            return _snapshot(
                {"title": _text(entity["title"], path=f"{path}.title"), "items": items}
            )

        lists = _entity_table(
            obj["lists"],
            path="$.initial_state.lists",
            fields=frozenset({"title", "items"}),
            validator=list_entity,
        )

        def place(entity: Mapping[str, StrictJSON], path: str) -> StrictJSON:
            return _snapshot({"name": _text(entity["name"], path=f"{path}.name")})

        places = _entity_table(
            obj["places"],
            path="$.initial_state.places",
            fields=frozenset({"name"}),
            validator=place,
        )

        route_keys: set[tuple[str, str, str]] = set()

        def route(entity: Mapping[str, StrictJSON], path: str) -> StrictJSON:
            origin = _identifier(entity["origin_id"], path=f"{path}.origin_id")
            destination = _identifier(entity["destination_id"], path=f"{path}.destination_id")
            if origin not in places or destination not in places:
                raise SimProgramError("dangling_reference", "route place does not exist", path)
            mode = _text(entity["mode"], path=f"{path}.mode")
            if mode not in _ROUTE_MODES:
                raise SimProgramError("invalid_enum", "unsupported route mode", f"{path}.mode")
            route_key = (origin, destination, mode)
            if route_key in route_keys:
                raise SimProgramError(
                    "duplicate_route",
                    "origin, destination, and mode must identify exactly one route",
                    path,
                )
            route_keys.add(route_key)
            return _snapshot(
                {
                    "origin_id": origin,
                    "destination_id": destination,
                    "mode": mode,
                    "distance_m": _nonnegative_integer(
                        entity["distance_m"], path=f"{path}.distance_m"
                    ),
                    "duration_s": _nonnegative_integer(
                        entity["duration_s"], path=f"{path}.duration_s"
                    ),
                }
            )

        routes = _entity_table(
            obj["routes"],
            path="$.initial_state.routes",
            fields=frozenset({"origin_id", "destination_id", "mode", "distance_m", "duration_s"}),
            validator=route,
        )

        raw_outbox = obj["outbox"]
        if not isinstance(raw_outbox, tuple):
            raise SimProgramError("type_mismatch", "must be an array", "$.initial_state.outbox")
        outbox: list[StrictJSON] = []
        message_ids: set[str] = set()
        for index, raw_message in enumerate(raw_outbox):
            path = f"$.initial_state.outbox[{index}]"
            message = _object(
                raw_message,
                path=path,
                fields=frozenset({"message_id", "to_contact_id", "channel", "body", "sent_at"}),
            )
            message_id = _identifier(message["message_id"], path=f"{path}.message_id")
            if message_id in message_ids:
                raise SimProgramError("duplicate_entity", "duplicate outbox message ID", path)
            message_ids.add(message_id)
            contact_id = _identifier(message["to_contact_id"], path=f"{path}.to_contact_id")
            if contact_id not in contacts:
                raise SimProgramError("dangling_reference", "message contact does not exist", path)
            channel = _text(message["channel"], path=f"{path}.channel")
            if channel != contacts[contact_id]["channel"]:
                raise SimProgramError(
                    "channel_mismatch",
                    "outbox channel is not registered for contact",
                    f"{path}.channel",
                )
            outbox.append(
                _snapshot(
                    {
                        "message_id": message_id,
                        "to_contact_id": contact_id,
                        "channel": channel,
                        "body": _text(message["body"], path=f"{path}.body"),
                        "sent_at": _timestamp(
                            message["sent_at"], timezone=timezone, path=f"{path}.sent_at"
                        ),
                    }
                )
            )

        media_obj = _object(
            obj["media"],
            path="$.initial_state.media",
            fields=frozenset({"status", "track_id", "catalog"}),
        )

        def track(entity: Mapping[str, StrictJSON], path: str) -> StrictJSON:
            return _snapshot({"title": _text(entity["title"], path=f"{path}.title")})

        catalog = _entity_table(
            media_obj["catalog"],
            path="$.initial_state.media.catalog",
            fields=frozenset({"title"}),
            validator=track,
        )
        status = _text(media_obj["status"], path="$.initial_state.media.status")
        if status not in _MEDIA_STATUSES:
            raise SimProgramError(
                "invalid_enum", "unsupported media status", "$.initial_state.media.status"
            )
        track_id_value = media_obj["track_id"]
        track_id = None
        if track_id_value is not None:
            track_id = _identifier(track_id_value, path="$.initial_state.media.track_id")
            if track_id not in catalog:
                raise SimProgramError(
                    "dangling_reference",
                    "media track does not exist",
                    "$.initial_state.media.track_id",
                )
        if (status == "stopped") != (track_id is None):
            raise SimProgramError(
                "invalid_media_state",
                "stopped requires null track_id; playing/paused require a track",
                "$.initial_state.media",
            )
        media = _snapshot({"status": status, "track_id": track_id, "catalog": catalog})

        raw_settings = obj["settings"]
        if not isinstance(raw_settings, Mapping):
            raise SimProgramError("type_mismatch", "must be an object", "$.initial_state.settings")
        settings: dict[str, StrictJSON] = {}
        for raw_key, raw_value in raw_settings.items():
            key = _identifier(raw_key, path="$.initial_state.settings")
            if type(raw_value) not in (bool, int, str):
                raise SimProgramError(
                    "type_mismatch",
                    "setting values must be boolean, integer, or string",
                    f"$.initial_state.settings.{key}",
                )
            settings[key] = _snapshot(raw_value, path=f"$.initial_state.settings.{key}")

        result = cls(
            reference_time=reference_time,
            timezone=timezone,
            reminders=reminders,
            calendar=calendar,
            contacts=contacts,
            notes=notes,
            lists=lists,
            places=places,
            routes=routes,
            outbox=tuple(sorted(outbox, key=lambda message: message["message_id"])),
            media=media,
            settings=MappingProxyType(dict(sorted(settings.items()))),
            _factory_token=_WORLD_FACTORY_TOKEN,
        )
        assert_sim_program_runtime_integrity()
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORLD_STATE_VERSION,
            "reference_time": self.reference_time,
            "timezone": self.timezone,
            "reminders": _to_builtin(self.reminders),
            "calendar": _to_builtin(self.calendar),
            "contacts": _to_builtin(self.contacts),
            "notes": _to_builtin(self.notes),
            "lists": _to_builtin(self.lists),
            "places": _to_builtin(self.places),
            "routes": _to_builtin(self.routes),
            "outbox": _to_builtin(self.outbox),
            "media": _to_builtin(self.media),
            "settings": _to_builtin(self.settings),
        }

    def canonical_json(self) -> str:
        return canonical_json(_snapshot(self.to_dict()))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


SEMANTIC_OPERATION_SPECS: Mapping[str, tuple[frozenset[str], bool]] = MappingProxyType(
    {
        "ADD_LIST_ITEM": (frozenset({"list_ref", "item_ref", "item_text"}), True),
        "CREATE_CALENDAR_EVENT": (
            frozenset({"event_ref", "summary", "begins_at", "ends_at"}),
            True,
        ),
        "CREATE_NOTE": (frozenset({"note_ref", "heading", "content"}), True),
        "CREATE_REMINDER": (
            frozenset({"reminder_ref", "summary", "due_at"}),
            True,
        ),
        "LOOK_UP_CONTACT": (frozenset({"contact_ref"}), False),
        "LOOK_UP_ROUTE": (
            frozenset({"from_place_ref", "to_place_ref", "travel_mode"}),
            False,
        ),
        "PAUSE_MEDIA": (frozenset(), True),
        "PLAY_MEDIA": (frozenset({"track_ref"}), True),
        "PROPOSE_MESSAGE": (
            frozenset({"message_ref", "recipient_ref", "delivery_channel", "content"}),
            True,
        ),
        "RESCHEDULE_CALENDAR_EVENT": (
            frozenset({"event_ref", "begins_at", "ends_at"}),
            True,
        ),
        "SET_BOOLEAN_SETTING": (frozenset({"setting_ref", "value"}), True),
        "SET_LIST_ITEM_CHECKED": (
            frozenset({"list_ref", "item_ref", "is_checked"}),
            True,
        ),
        "UPDATE_REMINDER": (
            frozenset({"reminder_ref", "summary", "due_at"}),
            True,
        ),
    }
)
_SEMANTIC_IDENTIFIER_FIELDS = frozenset(
    {
        "contact_ref",
        "event_ref",
        "from_place_ref",
        "item_ref",
        "list_ref",
        "message_ref",
        "note_ref",
        "recipient_ref",
        "reminder_ref",
        "setting_ref",
        "to_place_ref",
        "track_ref",
    }
)
_SEMANTIC_BOOLEAN_FIELDS = frozenset({"is_checked", "value"})
_SEMANTIC_TIMESTAMP_FIELDS = frozenset({"due_at", "begins_at", "ends_at"})


def semantic_operation_is_side_effecting(kind: str) -> bool:
    try:
        return SEMANTIC_OPERATION_SPECS[kind][1]
    except KeyError as exc:
        raise SimProgramError("unknown_semantic_operation", f"unknown operation {kind!r}") from exc


def _semantic_parameters(
    kind: str,
    value: object,
    *,
    path: str,
    timezone: str | None = None,
) -> Mapping[str, StrictJSON]:
    if kind not in SEMANTIC_OPERATION_SPECS:
        raise SimProgramError(
            "unknown_semantic_operation", f"unknown semantic operation {kind!r}", path
        )
    fields = SEMANTIC_OPERATION_SPECS[kind][0]
    parameters = _object(value, path=path, fields=fields)
    output: dict[str, StrictJSON] = {}
    for name in sorted(fields):
        field_path = f"{path}.{name}"
        raw = parameters[name]
        if name in _SEMANTIC_BOOLEAN_FIELDS:
            output[name] = _boolean(raw, path=field_path)
        elif name in _SEMANTIC_IDENTIFIER_FIELDS:
            output[name] = _identifier(raw, path=field_path)
        elif name in _SEMANTIC_TIMESTAMP_FIELDS:
            if timezone is None:
                output[name] = _text(raw, path=field_path)
            else:
                output[name] = canonical_timestamp(raw, timezone=timezone, path=field_path)
        else:
            allow_empty = kind == "CREATE_NOTE" and name == "content"
            output[name] = _text(raw, path=field_path, nonempty=not allow_empty)
    if kind == "LOOK_UP_ROUTE" and output["travel_mode"] not in _ROUTE_MODES:
        raise SimProgramError("invalid_enum", "unsupported route mode", f"{path}.travel_mode")
    return snapshot_strict_json(output)


@dataclass(frozen=True, slots=True)
class SemanticOperation:
    """A domain operation whose vocabulary is independent of model-facing tool names."""

    kind: str
    parameters: Mapping[str, StrictJSON]

    def __post_init__(self) -> None:
        kind = _text(self.kind, path="$.kind")
        parameters = snapshot_strict_json(self.parameters)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "parameters",
            _semantic_parameters(kind, parameters, path="$.parameters"),
        )

    def with_timezone(self, timezone: str) -> SemanticOperation:
        return SemanticOperation(
            kind=self.kind,
            parameters=_semantic_parameters(
                self.kind,
                self.parameters,
                path="$.parameters",
                timezone=timezone,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "parameters": _to_builtin(self.parameters)}


@dataclass(frozen=True, slots=True)
class ProgramStep:
    step_id: str
    decision: Decision
    mode: CallMode | None
    operations: tuple[SemanticOperation, ...]
    missing: tuple[str, ...]

    def __post_init__(self) -> None:
        step_id = _identifier(self.step_id, path="$.step_id")
        if not isinstance(self.decision, Decision):
            raise TypeError("decision must be Decision")
        if self.mode is not None and not isinstance(self.mode, CallMode):
            raise TypeError("mode must be CallMode or None")
        operations = tuple(self.operations)
        if any(not isinstance(item, SemanticOperation) for item in operations):
            raise TypeError("operations must contain SemanticOperation values")
        if self.mode is CallMode.PARALLEL:
            operations = tuple(
                sorted(
                    operations,
                    key=lambda operation: canonical_json(snapshot_strict_json(operation.to_dict())),
                )
            )
        missing = tuple(_identifier(item, path="$.missing") for item in self.missing)
        if self.decision in (Decision.CALL, Decision.CONFIRM):
            if self.mode is None or not operations or missing:
                raise SimProgramError(
                    "decision_shape",
                    "CALL/CONFIRM require mode and operations and forbid missing",
                )
            if self.mode is CallMode.SINGLE and len(operations) != 1:
                raise SimProgramError("invalid_call_count", "SINGLE requires one operation")
            has_side_effect = any(
                semantic_operation_is_side_effecting(operation.kind) for operation in operations
            )
            if self.decision is Decision.CALL and has_side_effect:
                raise SimProgramError(
                    "unsafe_gold_decision",
                    "side-effecting semantic operations require CONFIRM",
                )
            if self.decision is Decision.CONFIRM and not has_side_effect:
                raise SimProgramError(
                    "overconfirm_gold_decision",
                    "safe-only semantic operations require CALL",
                )
        elif self.decision is Decision.CLARIFY:
            if (
                self.mode is not None
                or operations
                or not missing
                or missing != tuple(sorted(set(missing)))
            ):
                raise SimProgramError(
                    "decision_shape",
                    "CLARIFY requires sorted unique missing and forbids mode/operations",
                )
        elif self.mode is not None or operations or missing:
            raise SimProgramError("decision_shape", "ABSTAIN forbids mode, operations, and missing")
        object.__setattr__(self, "step_id", step_id)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "missing", missing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "decision": self.decision.value,
            "mode": self.mode.value if self.mode is not None else None,
            "operations": [operation.to_dict() for operation in self.operations],
            "missing": list(self.missing),
        }


@dataclass(frozen=True, slots=True)
class SimProgram:
    case_id: str
    initial_state: WorldState
    steps: tuple[ProgramStep, ...]
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _PROGRAM_FACTORY_TOKEN:
            raise TypeError("SimProgram must be constructed through parse_sim_program")
        case_id = _identifier(self.case_id, path="$.case_id")
        if not isinstance(self.initial_state, WorldState):
            raise TypeError("initial_state must be WorldState")
        steps = tuple(self.steps)
        if not steps or any(not isinstance(step, ProgramStep) for step in steps):
            raise TypeError("steps must be a non-empty tuple of ProgramStep values")
        if len({step.step_id for step in steps}) != len(steps):
            raise SimProgramError("duplicate_step", "step IDs must be unique")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "steps", steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SIM_PROGRAM_VERSION,
            "case_id": self.case_id,
            "initial_state": self.initial_state.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }

    def canonical_json(self) -> str:
        return canonical_json(_snapshot(self.to_dict()))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def parse_sim_program(payload: object) -> SimProgram:
    """Snapshot and validate one exact program object without implicit defaults."""

    assert_sim_program_runtime_integrity()
    snapshot = snapshot_strict_json(payload)
    obj = _object(
        snapshot,
        path="$",
        fields=frozenset({"schema_version", "case_id", "initial_state", "steps"}),
    )
    if obj["schema_version"] != SIM_PROGRAM_VERSION:
        raise SimProgramError(
            "schema_version", f"must equal {SIM_PROGRAM_VERSION!r}", "$.schema_version"
        )
    case_id = _identifier(obj["case_id"], path="$.case_id")
    state = WorldState.from_dict(obj["initial_state"])
    raw_steps = obj["steps"]
    if not isinstance(raw_steps, tuple) or not raw_steps:
        raise SimProgramError("type_mismatch", "steps must be a non-empty array", "$.steps")
    steps: list[ProgramStep] = []
    seen_step_ids: set[str] = set()
    for index, raw_step in enumerate(raw_steps):
        path = f"$.steps[{index}]"
        step = _object(
            raw_step,
            path=path,
            fields=frozenset({"step_id", "decision", "mode", "operations", "missing"}),
        )
        step_id = _identifier(step["step_id"], path=f"{path}.step_id")
        if step_id in seen_step_ids:
            raise SimProgramError("duplicate_step", "step IDs must be unique", f"{path}.step_id")
        seen_step_ids.add(step_id)
        try:
            decision = Decision(_text(step["decision"], path=f"{path}.decision"))
        except ValueError as exc:
            raise SimProgramError(
                "invalid_decision", "unknown Action IR decision", f"{path}.decision"
            ) from exc
        raw_mode = step["mode"]
        mode: CallMode | None = None
        if raw_mode is not None:
            try:
                mode = CallMode(_text(raw_mode, path=f"{path}.mode"))
            except ValueError as exc:
                raise SimProgramError(
                    "invalid_mode", "unknown Action IR call mode", f"{path}.mode"
                ) from exc
        raw_operations = step["operations"]
        if not isinstance(raw_operations, tuple):
            raise SimProgramError(
                "type_mismatch", "operations must be an array", f"{path}.operations"
            )
        operations: list[SemanticOperation] = []
        for operation_index, raw_operation in enumerate(raw_operations):
            operation_path = f"{path}.operations[{operation_index}]"
            operation = _object(
                raw_operation,
                path=operation_path,
                fields=frozenset({"kind", "parameters"}),
            )
            kind = _text(operation["kind"], path=f"{operation_path}.kind")
            parameters = operation["parameters"]
            if not isinstance(parameters, Mapping):
                raise SimProgramError(
                    "type_mismatch",
                    "parameters must be an object",
                    f"{operation_path}.parameters",
                )
            operations.append(
                SemanticOperation(kind=kind, parameters=parameters).with_timezone(state.timezone)
            )
        raw_missing = step["missing"]
        if not isinstance(raw_missing, tuple):
            raise SimProgramError("type_mismatch", "missing must be an array", f"{path}.missing")
        missing = tuple(
            _identifier(item, path=f"{path}.missing[{missing_index}]")
            for missing_index, item in enumerate(raw_missing)
        )
        try:
            steps.append(
                ProgramStep(
                    step_id=step_id,
                    decision=decision,
                    mode=mode,
                    operations=tuple(operations),
                    missing=missing,
                )
            )
        except SimProgramError as exc:
            raise SimProgramError(exc.code, str(exc).split(": ", 1)[-1], path) from exc
    result = SimProgram(
        case_id=case_id,
        initial_state=state,
        steps=tuple(steps),
        _factory_token=_PROGRAM_FACTORY_TOKEN,
    )
    assert_sim_program_runtime_integrity()
    return result


def loads_sim_program(text: str) -> SimProgram:
    return parse_sim_program(loads_strict_json_object(text))


def sim_program_runtime_sha256() -> str:
    semantic_specs = {
        kind: {"fields": sorted(fields), "side_effecting": side_effecting}
        for kind, (fields, side_effecting) in SEMANTIC_OPERATION_SPECS.items()
    }
    dependencies = {
        name: _to_builtin(runtime_callable_identity(value))
        for name, value in (
            ("datetime.fromisoformat", datetime.fromisoformat),
            ("hashlib.sha256", hashlib.sha256),
            ("importlib.resources.files", importlib.resources.files),
            ("json.dumps", json.dumps),
            ("json.loads", json.loads),
            ("os.fstat", os.fstat),
            ("pathlib.Path.open", Path.open),
            ("unicodedata.category", unicodedata.category),
            ("unicodedata.normalize", unicodedata.normalize),
            ("zoneinfo.ZoneInfo.from_file", ZoneInfo.from_file),
        )
    }
    contract = {
        "schema_versions": {
            "program": SIM_PROGRAM_VERSION,
            "world": WORLD_STATE_VERSION,
            "timezone_runtime": TIMEZONE_RUNTIME_VERSION,
        },
        "limits": {
            "collection_items": MAX_SIM_COLLECTION_ITEMS,
            "integer_abs": MAX_SIM_INTEGER_ABS,
            "json_bytes": MAX_SIM_JSON_BYTES,
            "json_depth": MAX_SIM_JSON_DEPTH,
            "string_bytes": MAX_SIM_STRING_BYTES,
            "total_nodes": MAX_SIM_TOTAL_NODES,
            "tzif_bytes": MAX_TZIF_BYTES,
            "year_min": MIN_SIM_YEAR,
            "year_max": MAX_SIM_YEAR,
        },
        "identifier_pattern": _IDENTIFIER.pattern,
        "timestamp_pattern": _TIMESTAMP.pattern,
        "route_modes": sorted(_ROUTE_MODES),
        "media_statuses": sorted(_MEDIA_STATUSES),
        "semantic_operation_specs": semantic_specs,
        "tzpath": list(TZPATH),
        "dependencies": dependencies,
    }
    return module_runtime_sha256(
        globals(),
        module_name=__name__,
        source_path=__file__,
        contract=contract,
    )


def assert_sim_program_runtime_integrity() -> None:
    current = sim_program_runtime_sha256()
    if current != _PINNED_SIM_PROGRAM_RUNTIME_SHA256:
        raise SimProgramError(
            "runtime_identity",
            "sim-program runtime differs from its import-time identity",
        )


__all__ = [
    "MAX_SIM_COLLECTION_ITEMS",
    "MAX_SIM_INTEGER_ABS",
    "MAX_SIM_JSON_BYTES",
    "MAX_SIM_JSON_DEPTH",
    "MAX_SIM_STRING_BYTES",
    "MAX_SIM_TOTAL_NODES",
    "MAX_SIM_YEAR",
    "MAX_TZIF_BYTES",
    "MIN_SIM_YEAR",
    "SEMANTIC_OPERATION_SPECS",
    "SIM_PROGRAM_VERSION",
    "TIMEZONE_RUNTIME_VERSION",
    "WORLD_STATE_VERSION",
    "ProgramStep",
    "SemanticOperation",
    "SimProgram",
    "SimProgramError",
    "StrictJSON",
    "WorldState",
    "assert_sim_program_runtime_integrity",
    "canonical_identifier",
    "canonical_json",
    "canonical_sha256",
    "canonical_timestamp",
    "loads_sim_program",
    "loads_strict_json_object",
    "parse_sim_program",
    "semantic_operation_is_side_effecting",
    "sim_program_runtime_sha256",
    "snapshot_strict_json",
    "timezone_runtime_identity",
    "timezone_runtime_sha256",
]


# Freeze the complete source/live-code/dependency contract only after every executable
# definition exists.  Later monkeypatches and source replacement therefore fail closed.
_PINNED_SIM_PROGRAM_RUNTIME_SHA256 = sim_program_runtime_sha256()
