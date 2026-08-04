"""Fail-closed one-shot population claims for proposed BarunAction GVS-v1.

This module defines protocol objects for GVS-specific authorization receipts and a
global population-claim ledger.  A whole D-support, selection, or confirmation
population must be durably claimed before any private label is parsed; a crash after
the durable append therefore retires the complete population and cannot be rescued
with a partial retry.

The primitives here perform no key management, encryption, filesystem, network,
model, dataset, CUDA, or Jarvis operation.  Real secrecy requires external custody,
and real one-shot behavior requires an external atomic compare-and-append service.
An in-memory callback can lie; this module validates the protocol object but cannot
turn Python process memory into durable infrastructure.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from barunlm.evaluation.sim_program import (
    SimProgramError,
    assert_sim_program_runtime_integrity,
    module_runtime_sha256,
    runtime_callable_identity,
    sim_program_runtime_sha256,
)

GVS_AUTHORIZATION_RECEIPT_VERSION = "barun-gvs-authorization-receipt-v2"
GVS_POPULATION_CLAIM_EVENT_VERSION = "barun-gvs-population-claim-event-v3"

GVS_PHASE_ORDER = ("d_support", "selection", "confirmation")
GVS_PHASES = frozenset(GVS_PHASE_ORDER)
GVS_AUTHORIZATION_BINDING_FIELDS = frozenset(
    {
        "population_manifest_sha256",
        "component_manifest_sha256",
        "prompt_collection_sha256",
        "label_commitment_root_sha256",
        "custody_policy_sha256",
        "generator_checkpoint_sha256",
        "ranker_artifact_root_sha256",
        "tokenizer_sha256",
        "eos_token_id",
        "pad_token_id",
        "decoding_contract_sha256",
        "renderer_sha256",
        "schema_presentation_sha256",
        "simulator_sha256",
        "policy_sha256",
        "candidate_trace_root_sha256",
        "arm_checkpoint_root_sha256",
        "prediction_root_sha256",
        "evaluator_sha256",
        "scorer_sha256",
        "bootstrap_sha256",
        "runtime_sha256",
        "power_report_sha256",
        "source_tree_sha256",
        "ledger_service_sha256",
    }
)

_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "run_id",
        "created_at_utc",
        "key_id",
        "bindings",
        "gate_spec_sha256",
        "gate_results",
        "passed",
        "authorizes_model_cuda_training_or_jarvis",
        "authorizes_private_label_access",
        "authorization_runtime_sha256",
        "receipt_sha256",
        "receipt_hmac_sha256",
    }
)
_CLAIM_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "claim_id",
        "phase",
        "run_id",
        "population_manifest_sha256",
        "prompt_collection_sha256",
        "label_commitment_root_sha256",
        "authorization_receipt_sha256",
        "scoring_session_id",
        "accessor_id",
        "purpose",
        "claimed_at_utc",
        "previous_event_sha256",
        "key_id",
        "partial_retry_authorized",
        "authorizes_model_cuda_training_or_jarvis",
        "authorizes_private_label_access",
        "proves_external_atomicity",
        "authorization_runtime_sha256",
        "event_sha256",
        "event_hmac_sha256",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 100_000
_MAX_SERIALIZED_BYTES = 2 * 1024 * 1024
_MAX_GATES = 128
_MAX_TOKEN_ID = 2**31 - 1
_MAX_HMAC_KEY_BYTES = 4096

_RECEIPT_HASH_DOMAIN = b"barun-gvs-authorization-receipt-v2/hash"
_RECEIPT_HMAC_DOMAIN = b"barun-gvs-authorization-receipt-v2/hmac"
_EVENT_HASH_DOMAIN = b"barun-gvs-population-claim-event-v3/hash"
_EVENT_HMAC_DOMAIN = b"barun-gvs-population-claim-event-v3/hmac"

# Security-critical imported callables are used through import-time pins.  Public entrypoints
# separately verify that their module-visible counterparts were not replaced.  This cannot make
# mutable Python process memory a trust anchor, but it makes ordinary import/runtime monkeypatches
# fail closed and binds the exact runtime into every v2 receipt and event.
_HASHLIB_SHA256 = hashlib.sha256
_HMAC_NEW = hmac.new
_HMAC_COMPARE_DIGEST = hmac.compare_digest
_JSON_DUMPS = json.dumps
_JSON_LOADS = json.loads
_JSON_DECODE_ERROR = json.JSONDecodeError
_MATH_ISFINITE = math.isfinite
_UNICODE_CATEGORY = unicodedata.category
_UNICODE_NORMALIZE = unicodedata.normalize
_DATETIME_CLASS = datetime
_DATETIME_FROMISOFORMAT = datetime.fromisoformat
_ASSERT_SIM_PROGRAM_RUNTIME_INTEGRITY = assert_sim_program_runtime_integrity
_MODULE_RUNTIME_SHA256 = module_runtime_sha256
_RUNTIME_CALLABLE_IDENTITY = runtime_callable_identity
_SIM_PROGRAM_RUNTIME_SHA256 = sim_program_runtime_sha256


class GVSAuthorizationError(ValueError):
    """A GVS authorization receipt or population claim is invalid."""


_CONTRACT_OBJECT_PINS = (
    ("GVS_AUTHORIZATION_RECEIPT_VERSION", GVS_AUTHORIZATION_RECEIPT_VERSION),
    ("GVS_POPULATION_CLAIM_EVENT_VERSION", GVS_POPULATION_CLAIM_EVENT_VERSION),
    ("GVS_PHASE_ORDER", GVS_PHASE_ORDER),
    ("GVS_PHASES", GVS_PHASES),
    ("GVS_AUTHORIZATION_BINDING_FIELDS", GVS_AUTHORIZATION_BINDING_FIELDS),
    ("_RECEIPT_FIELDS", _RECEIPT_FIELDS),
    ("_CLAIM_EVENT_FIELDS", _CLAIM_EVENT_FIELDS),
    ("_SHA256_RE", _SHA256_RE),
    ("_IDENTIFIER_RE", _IDENTIFIER_RE),
    ("_UTC_RE", _UTC_RE),
    ("_MAX_JSON_DEPTH", _MAX_JSON_DEPTH),
    ("_MAX_JSON_NODES", _MAX_JSON_NODES),
    ("_MAX_SERIALIZED_BYTES", _MAX_SERIALIZED_BYTES),
    ("_MAX_GATES", _MAX_GATES),
    ("_MAX_TOKEN_ID", _MAX_TOKEN_ID),
    ("_MAX_HMAC_KEY_BYTES", _MAX_HMAC_KEY_BYTES),
    ("_RECEIPT_HASH_DOMAIN", _RECEIPT_HASH_DOMAIN),
    ("_RECEIPT_HMAC_DOMAIN", _RECEIPT_HMAC_DOMAIN),
    ("_EVENT_HASH_DOMAIN", _EVENT_HASH_DOMAIN),
    ("_EVENT_HMAC_DOMAIN", _EVENT_HMAC_DOMAIN),
    ("GVSAuthorizationError", GVSAuthorizationError),
)


def _strict_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise GVSAuthorizationError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise GVSAuthorizationError(f"{label} must be a bounded portable identifier")
    if value != _UNICODE_NORMALIZE("NFC", value):
        raise GVSAuthorizationError(f"{label} must be NFC-normalized")
    return value


def _strict_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise GVSAuthorizationError(f"{label} must be bounded nonempty UTF-8 text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise GVSAuthorizationError(f"{label} must be bounded nonempty UTF-8 text") from error
    if len(encoded) > 4096:
        raise GVSAuthorizationError(f"{label} must be bounded nonempty UTF-8 text")
    if value != _UNICODE_NORMALIZE("NFC", value):
        raise GVSAuthorizationError(f"{label} must be NFC-normalized")
    if value != value.strip() or any(
        character != " " and (character.isspace() or _UNICODE_CATEGORY(character).startswith("C"))
        for character in value
    ):
        raise GVSAuthorizationError(
            f"{label} must not contain leading, trailing, control, or format characters"
        )
    return value


def _strict_timestamp(value: object, *, label: str) -> datetime:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        raise GVSAuthorizationError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = _DATETIME_FROMISOFORMAT(value[:-1] + "+00:00")
    except ValueError as error:
        raise GVSAuthorizationError(f"{label} is not a real UTC timestamp") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise GVSAuthorizationError(f"{label} must resolve to UTC")
    canonical = parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if parsed.microsecond == 0:
        canonical = canonical.replace(".000000Z", "Z")
    else:
        canonical = canonical[:-1].rstrip("0") + "Z"
    if value != canonical:
        raise GVSAuthorizationError(f"{label} must use one canonical UTC representation")
    return parsed


def _strict_hmac_key(value: object, *, label: str) -> bytes:
    if type(value) is not bytes or not 32 <= len(value) <= _MAX_HMAC_KEY_BYTES:
        raise GVSAuthorizationError(
            f"{label} must contain 32 to {_MAX_HMAC_KEY_BYTES} secret bytes"
        )
    return value


def _snapshot_json(
    value: object,
    *,
    label: str,
    depth: int = 0,
    active: set[int] | None = None,
    counter: list[int] | None = None,
) -> object:
    """Detach exact built-in JSON values without tuple/list or bool/int coercion."""

    if depth > _MAX_JSON_DEPTH:
        raise GVSAuthorizationError(f"{label} exceeds the JSON depth bound")
    active_ids = set() if active is None else active
    node_counter = [0] if counter is None else counter
    node_counter[0] += 1
    if node_counter[0] > _MAX_JSON_NODES:
        raise GVSAuthorizationError(f"{label} exceeds the JSON node bound")
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise GVSAuthorizationError(f"{label} contains invalid UTF-8 text") from error
        return value
    if type(value) is float:
        if not _MATH_ISFINITE(value):
            raise GVSAuthorizationError(f"{label} contains a non-finite float")
        return value
    if type(value) not in {list, dict}:
        raise GVSAuthorizationError(f"{label} must contain exact built-in JSON values")
    object_id = id(value)
    if object_id in active_ids:
        raise GVSAuthorizationError(f"{label} contains a JSON cycle")
    active_ids.add(object_id)
    try:
        if type(value) is list:
            try:
                children = tuple(value)
            except RuntimeError as error:
                raise GVSAuthorizationError(f"{label} mutated during its snapshot") from error
            return [
                _snapshot_json(
                    child,
                    label=f"{label}[]",
                    depth=depth + 1,
                    active=active_ids,
                    counter=node_counter,
                )
                for child in children
            ]
        result: dict[str, object] = {}
        try:
            items = tuple(value.items())
        except RuntimeError as error:
            raise GVSAuthorizationError(f"{label} mutated during its snapshot") from error
        for key, child in items:
            if type(key) is not str or key in result:
                raise GVSAuthorizationError(f"{label} contains an invalid or duplicate key")
            try:
                key.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise GVSAuthorizationError(f"{label} contains an invalid UTF-8 key") from error
            result[key] = _snapshot_json(
                child,
                label=f"{label}.{key}",
                depth=depth + 1,
                active=active_ids,
                counter=node_counter,
            )
        return result
    finally:
        active_ids.remove(object_id)


def _exact_object(value: object, *, fields: frozenset[str], label: str) -> dict[str, Any]:
    snapshot = _snapshot_json(value, label=label)
    if type(snapshot) is not dict:
        raise GVSAuthorizationError(f"{label} must be an exact JSON object")
    actual = set(snapshot)
    if actual != fields:
        raise GVSAuthorizationError(
            f"{label} fields changed; missing={sorted(fields - actual)!r}, "
            f"extra={sorted(actual - fields)!r}"
        )
    return snapshot


def _snapshot_mapping(value: object, *, label: str) -> dict[str, Any]:
    snapshot = _snapshot_json(value, label=label)
    if type(snapshot) is not dict:
        raise GVSAuthorizationError(f"{label} must be an exact JSON object")
    return snapshot


def _canonical_bytes(value: object) -> bytes:
    try:
        encoded = _JSON_DUMPS(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise GVSAuthorizationError("authorization evidence is not strict JSON") from error
    if len(encoded) > _MAX_SERIALIZED_BYTES:
        raise GVSAuthorizationError("authorization evidence exceeds the serialized size bound")
    return encoded


def _sha256_json(value: object, *, domain: bytes) -> str:
    return _HASHLIB_SHA256(domain + b"\x00" + _canonical_bytes(value)).hexdigest()


def _hmac_sha256(key: bytes, value: object, *, domain: bytes) -> str:
    return _HMAC_NEW(key, domain + b"\x00" + _canonical_bytes(value), _HASHLIB_SHA256).hexdigest()


def _strict_json_value_load(raw: bytes | str, *, label: str) -> object:
    if type(raw) is bytes:
        if len(raw) > _MAX_SERIALIZED_BYTES:
            raise GVSAuthorizationError(f"{label} exceeds the serialized size bound")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise GVSAuthorizationError(f"{label} is not UTF-8") from error
    elif type(raw) is str:
        try:
            raw_size = len(raw.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as error:
            raise GVSAuthorizationError(f"{label} is not valid UTF-8 text") from error
        if raw_size > _MAX_SERIALIZED_BYTES:
            raise GVSAuthorizationError(f"{label} exceeds the serialized size bound")
        text = raw
    else:
        raise GVSAuthorizationError(f"{label} must be exact bytes or text")

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise GVSAuthorizationError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = _JSON_LOADS(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GVSAuthorizationError(f"{label} contains forbidden constant {token!r}")
            ),
        )
    except GVSAuthorizationError:
        raise
    except (_JSON_DECODE_ERROR, TypeError, ValueError, RecursionError) as error:
        raise GVSAuthorizationError(f"{label} is not strict JSON") from error
    return _snapshot_json(value, label=label)


def _strict_json_load(raw: bytes | str, *, label: str) -> dict[str, Any]:
    snapshot = _strict_json_value_load(raw, label=label)
    if type(snapshot) is not dict:
        raise GVSAuthorizationError(f"{label} must decode to a JSON object")
    return snapshot


def load_gvs_authorization_receipt_json(raw: bytes | str) -> dict[str, Any]:
    """Load bounded strict JSON; call the validator before trusting its contents."""

    _assert_gvs_authorization_runtime_integrity()
    result = _strict_json_load(raw, label="GVS authorization receipt")
    _assert_gvs_authorization_runtime_integrity()
    return result


def load_gvs_population_claim_ledger_json(raw: bytes | str) -> list[dict[str, Any]]:
    """Load a bounded strict JSON list of claim events."""

    _assert_gvs_authorization_runtime_integrity()
    if type(raw) not in {bytes, str}:
        raise GVSAuthorizationError("GVS population claim ledger must be exact bytes or text")
    events = _strict_json_value_load(raw, label="GVS population claim ledger")
    if type(events) is not list or any(type(event) is not dict for event in events):
        raise GVSAuthorizationError("GVS population claim ledger must be a JSON event list")
    _assert_gvs_authorization_runtime_integrity()
    return events


def _validate_bindings(value: object) -> dict[str, Any]:
    bindings = _exact_object(
        value,
        fields=GVS_AUTHORIZATION_BINDING_FIELDS,
        label="authorization bindings",
    )
    for name, field_value in bindings.items():
        if name in {"eos_token_id", "pad_token_id"}:
            if type(field_value) is not int or not 0 <= field_value <= _MAX_TOKEN_ID:
                raise GVSAuthorizationError(
                    f"authorization bindings.{name} must be a bounded nonnegative int"
                )
        else:
            _strict_sha256(field_value, label=f"authorization bindings.{name}")
    if bindings["eos_token_id"] == bindings["pad_token_id"]:
        raise GVSAuthorizationError("EOS and pad token IDs must be distinct")
    return bindings


def _validate_gate_results(value: object) -> dict[str, bool]:
    if type(value) is not dict or not value or len(value) > _MAX_GATES:
        raise GVSAuthorizationError("gate_results must be a bounded nonempty exact object")
    result: dict[str, bool] = {}
    for name, passed in value.items():
        checked_name = _strict_identifier(name, label="gate_results key")
        if type(passed) is not bool:
            raise GVSAuthorizationError(f"gate result {checked_name!r} must be boolean")
        result[checked_name] = passed
    return result


def _strict_phase(value: object, *, label: str) -> str:
    if type(value) is not str or value not in GVS_PHASES:
        raise GVSAuthorizationError(f"{label} is unsupported")
    return value


def _receipt_hash_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    result = _snapshot_mapping(receipt, label="authorization receipt hash payload")
    result["receipt_sha256"] = "0" * 64
    result["receipt_hmac_sha256"] = "0" * 64
    return result


def _receipt_mac_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    result = _snapshot_mapping(receipt, label="authorization receipt HMAC payload")
    result["receipt_hmac_sha256"] = "0" * 64
    return result


def _validate_receipt_shape(value: object) -> dict[str, Any]:
    receipt = _exact_object(value, fields=_RECEIPT_FIELDS, label="GVS authorization receipt")
    if receipt["schema_version"] != GVS_AUTHORIZATION_RECEIPT_VERSION:
        raise GVSAuthorizationError("unsupported GVS authorization receipt version")
    _strict_phase(receipt["phase"], label="GVS authorization phase")
    _strict_identifier(receipt["run_id"], label="authorization receipt run_id")
    _strict_timestamp(receipt["created_at_utc"], label="authorization receipt created_at_utc")
    _strict_identifier(receipt["key_id"], label="authorization receipt key_id")
    receipt["bindings"] = _validate_bindings(receipt["bindings"])
    _strict_sha256(receipt["gate_spec_sha256"], label="authorization gate_spec_sha256")
    receipt["gate_results"] = _validate_gate_results(receipt["gate_results"])
    if type(receipt["passed"]) is not bool or receipt["passed"] != all(
        receipt["gate_results"].values()
    ):
        raise GVSAuthorizationError("receipt passed flag contradicts exact gate results")
    if receipt["authorizes_model_cuda_training_or_jarvis"] is not False:
        raise GVSAuthorizationError("a GVS label receipt cannot authorize model or compute access")
    if receipt["authorizes_private_label_access"] is not False:
        raise GVSAuthorizationError(
            "an in-process GVS receipt cannot authorize private-label access"
        )
    _strict_sha256(
        receipt["authorization_runtime_sha256"],
        label="authorization runtime_sha256",
    )
    _strict_sha256(receipt["receipt_sha256"], label="authorization receipt_sha256")
    _strict_sha256(receipt["receipt_hmac_sha256"], label="authorization receipt_hmac_sha256")
    return receipt


def seal_gvs_authorization_receipt(
    receipt: Mapping[str, Any],
    *,
    signing_key: bytes,
) -> dict[str, Any]:
    """Seal an exact GVS receipt; failed gate receipts may be preserved but cannot claim."""

    _assert_gvs_authorization_runtime_integrity()
    key = _strict_hmac_key(signing_key, label="authorization receipt signing_key")
    result = _exact_object(receipt, fields=_RECEIPT_FIELDS, label="GVS authorization receipt")
    result["authorization_runtime_sha256"] = _gvs_authorization_runtime_sha256()
    result["receipt_sha256"] = "0" * 64
    result["receipt_hmac_sha256"] = "0" * 64
    result = _validate_receipt_shape(result)
    result["receipt_sha256"] = _sha256_json(
        _receipt_hash_payload(result), domain=_RECEIPT_HASH_DOMAIN
    )
    result["receipt_hmac_sha256"] = _hmac_sha256(
        key,
        _receipt_mac_payload(result),
        domain=_RECEIPT_HMAC_DOMAIN,
    )
    result = _validate_receipt_shape(result)
    _assert_gvs_authorization_runtime_integrity()
    return result


def validate_gvs_authorization_receipt(
    receipt: object,
    *,
    signing_key: bytes,
    expected_key_id: str,
    expected_phase: str,
    expected_run_id: str,
    expected_bindings: Mapping[str, Any],
    expected_gate_spec_sha256: str,
    expected_gate_ids: tuple[str, ...],
    require_passed: bool = True,
) -> dict[str, Any]:
    """Verify HMAC, self-hash, phase, exact bindings, and the frozen gate roster."""

    _assert_gvs_authorization_runtime_integrity()
    if type(require_passed) is not bool:
        raise GVSAuthorizationError("require_passed must be an exact bool")
    key = _strict_hmac_key(signing_key, label="authorization receipt signing_key")
    checked = _validate_receipt_shape(receipt)
    if checked["key_id"] != _strict_identifier(expected_key_id, label="expected_key_id"):
        raise GVSAuthorizationError("authorization receipt key_id mismatch")
    phase = _strict_phase(expected_phase, label="expected_phase")
    if checked["phase"] != phase:
        raise GVSAuthorizationError("authorization receipt phase mismatch")
    if checked["run_id"] != _strict_identifier(expected_run_id, label="expected_run_id"):
        raise GVSAuthorizationError("authorization receipt run_id mismatch")
    bindings = _validate_bindings(expected_bindings)
    if checked["bindings"] != bindings:
        raise GVSAuthorizationError("authorization receipt bindings differ from frozen evidence")
    gate_spec = _strict_sha256(expected_gate_spec_sha256, label="expected_gate_spec_sha256")
    if checked["gate_spec_sha256"] != gate_spec:
        raise GVSAuthorizationError("authorization receipt gate specification mismatch")
    if (
        type(expected_gate_ids) is not tuple
        or not expected_gate_ids
        or len(expected_gate_ids) > _MAX_GATES
    ):
        raise GVSAuthorizationError("expected_gate_ids must be a bounded nonempty tuple")
    checked_gate_ids = tuple(
        _strict_identifier(value, label=f"expected_gate_ids[{index}]")
        for index, value in enumerate(expected_gate_ids)
    )
    if checked_gate_ids != tuple(sorted(set(checked_gate_ids))):
        raise GVSAuthorizationError("expected_gate_ids must be sorted and unique")
    if set(checked["gate_results"]) != set(checked_gate_ids):
        raise GVSAuthorizationError("authorization receipt has a different frozen gate roster")
    runtime_sha256 = _gvs_authorization_runtime_sha256()
    if checked["authorization_runtime_sha256"] != runtime_sha256:
        raise GVSAuthorizationError("authorization receipt runtime identity mismatch")
    if checked["receipt_sha256"] != _sha256_json(
        _receipt_hash_payload(checked), domain=_RECEIPT_HASH_DOMAIN
    ):
        raise GVSAuthorizationError("authorization receipt self-hash mismatch")
    expected_mac = _hmac_sha256(
        key,
        _receipt_mac_payload(checked),
        domain=_RECEIPT_HMAC_DOMAIN,
    )
    if not _HMAC_COMPARE_DIGEST(checked["receipt_hmac_sha256"], expected_mac):
        raise GVSAuthorizationError("authorization receipt HMAC mismatch")
    if require_passed and not checked["passed"]:
        raise GVSAuthorizationError("authorization receipt contains a failed gate")
    _assert_gvs_authorization_runtime_integrity()
    return checked


def _event_hash_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    result = _snapshot_mapping(event, label="population claim event hash payload")
    result["event_sha256"] = "0" * 64
    result["event_hmac_sha256"] = "0" * 64
    return result


def _event_mac_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    result = _snapshot_mapping(event, label="population claim event HMAC payload")
    result["event_hmac_sha256"] = "0" * 64
    return result


def _validate_claim_event(value: object, *, expected_sequence: int) -> dict[str, Any]:
    event = _exact_object(value, fields=_CLAIM_EVENT_FIELDS, label="GVS population claim event")
    if event["schema_version"] != GVS_POPULATION_CLAIM_EVENT_VERSION:
        raise GVSAuthorizationError("unsupported GVS population claim event version")
    if type(event["sequence"]) is not int or event["sequence"] != expected_sequence:
        raise GVSAuthorizationError("population claim sequence is not contiguous")
    for name in ("claim_id", "run_id", "scoring_session_id", "accessor_id", "key_id"):
        _strict_identifier(event[name], label=f"population claim {name}")
    _strict_phase(event["phase"], label="population claim phase")
    _strict_text(event["purpose"], label="population claim purpose")
    _strict_timestamp(event["claimed_at_utc"], label="population claim claimed_at_utc")
    for name in (
        "population_manifest_sha256",
        "prompt_collection_sha256",
        "label_commitment_root_sha256",
        "authorization_receipt_sha256",
        "previous_event_sha256",
        "event_sha256",
        "event_hmac_sha256",
    ):
        _strict_sha256(event[name], label=f"population claim {name}")
    if event["partial_retry_authorized"] is not False:
        raise GVSAuthorizationError("population claims can never authorize partial retry")
    if event["authorizes_model_cuda_training_or_jarvis"] is not False:
        raise GVSAuthorizationError("an in-process claim cannot authorize model or compute access")
    if event["authorizes_private_label_access"] is not False:
        raise GVSAuthorizationError("an in-process claim cannot authorize private-label access")
    if event["proves_external_atomicity"] is not False:
        raise GVSAuthorizationError("an in-process claim cannot prove external atomicity")
    _strict_sha256(
        event["authorization_runtime_sha256"],
        label="population claim authorization_runtime_sha256",
    )
    return event


def validate_gvs_population_claim_ledger(
    events: Sequence[Mapping[str, Any]],
    *,
    signing_key: bytes,
    expected_key_id: str,
) -> str:
    """Validate the complete global claim chain and return its current head hash."""

    _assert_gvs_authorization_runtime_integrity()
    key = _strict_hmac_key(signing_key, label="population ledger signing_key")
    key_id = _strict_identifier(expected_key_id, label="expected population ledger key_id")
    if type(events) is not list:
        raise GVSAuthorizationError("population claim ledger must be an exact list")
    snapshot = _snapshot_json(events, label="GVS population claim ledger")
    assert isinstance(snapshot, list)
    previous_hash = "0" * 64
    previous_time: datetime | None = None
    claim_ids: set[str] = set()
    population_ids: set[str] = set()
    prompt_collection_ids: set[str] = set()
    label_commitment_ids: set[str] = set()
    run_phase_positions: dict[str, int] = {}
    runtime_sha256 = _gvs_authorization_runtime_sha256()
    for index, event_value in enumerate(snapshot):
        event = _validate_claim_event(event_value, expected_sequence=index)
        if event["key_id"] != key_id:
            raise GVSAuthorizationError("population claim ledger key_id mismatch")
        if event["claim_id"] in claim_ids:
            raise GVSAuthorizationError("population claim IDs must be globally unique")
        claim_ids.add(event["claim_id"])
        if event["population_manifest_sha256"] in population_ids:
            raise GVSAuthorizationError("a GVS population has already been claimed and retired")
        population_ids.add(event["population_manifest_sha256"])
        if event["prompt_collection_sha256"] in prompt_collection_ids:
            raise GVSAuthorizationError(
                "a GVS prompt collection has already been claimed and retired"
            )
        prompt_collection_ids.add(event["prompt_collection_sha256"])
        if event["label_commitment_root_sha256"] in label_commitment_ids:
            raise GVSAuthorizationError(
                "a GVS label commitment has already been claimed and retired"
            )
        label_commitment_ids.add(event["label_commitment_root_sha256"])
        phase_position = GVS_PHASE_ORDER.index(event["phase"])
        previous_phase_position = run_phase_positions.get(event["run_id"])
        if previous_phase_position is not None and phase_position <= previous_phase_position:
            raise GVSAuthorizationError(
                "GVS population claim phases must strictly advance once per run"
            )
        if event["phase"] == "confirmation" and previous_phase_position != GVS_PHASE_ORDER.index(
            "selection"
        ):
            raise GVSAuthorizationError(
                "a confirmation claim requires an earlier same-run selection claim"
            )
        run_phase_positions[event["run_id"]] = phase_position
        if event["previous_event_sha256"] != previous_hash:
            raise GVSAuthorizationError("population claim ledger hash chain is broken")
        if event["authorization_runtime_sha256"] != runtime_sha256:
            raise GVSAuthorizationError("population claim authorization runtime mismatch")
        claimed_at = _strict_timestamp(
            event["claimed_at_utc"], label=f"population claim ledger[{index}] time"
        )
        if previous_time is not None and claimed_at <= previous_time:
            raise GVSAuthorizationError("population claim timestamps must strictly increase")
        if event["event_sha256"] != _sha256_json(
            _event_hash_payload(event), domain=_EVENT_HASH_DOMAIN
        ):
            raise GVSAuthorizationError("population claim event self-hash mismatch")
        expected_mac = _hmac_sha256(
            key,
            _event_mac_payload(event),
            domain=_EVENT_HMAC_DOMAIN,
        )
        if not _HMAC_COMPARE_DIGEST(event["event_hmac_sha256"], expected_mac):
            raise GVSAuthorizationError("population claim event HMAC mismatch")
        previous_hash = event["event_sha256"]
        previous_time = claimed_at
    _assert_gvs_authorization_runtime_integrity()
    return previous_hash


def claim_gvs_population_once(
    *,
    authorization_receipt: Mapping[str, Any],
    receipt_signing_key: bytes,
    expected_receipt_key_id: str,
    expected_phase: str,
    expected_run_id: str,
    expected_bindings: Mapping[str, Any],
    expected_gate_spec_sha256: str,
    expected_gate_ids: tuple[str, ...],
    prior_durable_ledger: Sequence[Mapping[str, Any]],
    ledger_signing_key: bytes,
    expected_ledger_key_id: str,
    claim_id: str,
    scoring_session_id: str,
    accessor_id: str,
    purpose: str,
    claimed_at_utc: str,
    durable_compare_and_append: Callable[[bytes, str], str],
) -> dict[str, Any]:
    """Construct and submit a complete-population retirement event.

    The callback receives immutable canonical event bytes and the expected prior head.  It must
    atomically append those exact bytes only when its durable head matches, then return the new
    durable head.  This function checks that acknowledgement, but cannot prove the callback wrote
    anything.  Its returned event therefore explicitly does *not* authorize private-label access
    or prove external atomicity.  External custody must independently establish those facts before
    deserializing even one label.  A crash after a real append still retires the population.
    """

    _assert_gvs_authorization_runtime_integrity()
    if not callable(durable_compare_and_append):
        raise GVSAuthorizationError("durable_compare_and_append must be callable")
    receipt = validate_gvs_authorization_receipt(
        authorization_receipt,
        signing_key=receipt_signing_key,
        expected_key_id=expected_receipt_key_id,
        expected_phase=expected_phase,
        expected_run_id=expected_run_id,
        expected_bindings=expected_bindings,
        expected_gate_spec_sha256=expected_gate_spec_sha256,
        expected_gate_ids=expected_gate_ids,
        require_passed=True,
    )
    if type(prior_durable_ledger) is not list:
        raise GVSAuthorizationError("prior durable ledger must be an exact list snapshot")
    ledger_snapshot = _snapshot_json(prior_durable_ledger, label="prior durable claim ledger")
    assert isinstance(ledger_snapshot, list)
    ledger_key = _strict_hmac_key(ledger_signing_key, label="population ledger signing_key")
    ledger_key_id = _strict_identifier(
        expected_ledger_key_id, label="expected population ledger key_id"
    )
    previous_hash = validate_gvs_population_claim_ledger(
        ledger_snapshot,
        signing_key=ledger_key,
        expected_key_id=ledger_key_id,
    )
    population_sha256 = receipt["bindings"]["population_manifest_sha256"]
    prompt_collection_sha256 = receipt["bindings"]["prompt_collection_sha256"]
    label_commitment_root_sha256 = receipt["bindings"]["label_commitment_root_sha256"]
    if any(event["population_manifest_sha256"] == population_sha256 for event in ledger_snapshot):
        raise GVSAuthorizationError("the complete GVS population is already claimed and retired")
    if any(
        event["prompt_collection_sha256"] == prompt_collection_sha256 for event in ledger_snapshot
    ):
        raise GVSAuthorizationError(
            "the complete GVS prompt collection is already claimed and retired"
        )
    if any(
        event["label_commitment_root_sha256"] == label_commitment_root_sha256
        for event in ledger_snapshot
    ):
        raise GVSAuthorizationError(
            "the complete GVS label commitment is already claimed and retired"
        )
    prior_run_phases = [
        event["phase"] for event in ledger_snapshot if event["run_id"] == receipt["run_id"]
    ]
    if prior_run_phases and GVS_PHASE_ORDER.index(receipt["phase"]) <= GVS_PHASE_ORDER.index(
        prior_run_phases[-1]
    ):
        raise GVSAuthorizationError(
            "GVS population claim phases must strictly advance once per run"
        )
    if receipt["phase"] == "confirmation" and (
        not prior_run_phases or prior_run_phases[-1] != "selection"
    ):
        raise GVSAuthorizationError(
            "a confirmation claim requires an earlier same-run selection claim"
        )
    checked_claim_id = _strict_identifier(claim_id, label="claim_id")
    if any(event["claim_id"] == checked_claim_id for event in ledger_snapshot):
        raise GVSAuthorizationError("claim_id already exists in the durable ledger")
    claimed_at = _strict_timestamp(claimed_at_utc, label="claimed_at_utc")
    receipt_time = _strict_timestamp(
        receipt["created_at_utc"], label="authorization receipt created_at_utc"
    )
    if claimed_at <= receipt_time:
        raise GVSAuthorizationError("population claim must strictly postdate its receipt")
    if ledger_snapshot:
        last_time = _strict_timestamp(
            ledger_snapshot[-1]["claimed_at_utc"], label="prior durable ledger last timestamp"
        )
        if claimed_at <= last_time:
            raise GVSAuthorizationError("population claim must strictly postdate the ledger head")
    event: dict[str, Any] = {
        "schema_version": GVS_POPULATION_CLAIM_EVENT_VERSION,
        "sequence": len(ledger_snapshot),
        "claim_id": checked_claim_id,
        "phase": receipt["phase"],
        "run_id": receipt["run_id"],
        "population_manifest_sha256": population_sha256,
        "prompt_collection_sha256": prompt_collection_sha256,
        "label_commitment_root_sha256": label_commitment_root_sha256,
        "authorization_receipt_sha256": receipt["receipt_sha256"],
        "scoring_session_id": _strict_identifier(scoring_session_id, label="scoring_session_id"),
        "accessor_id": _strict_identifier(accessor_id, label="accessor_id"),
        "purpose": _strict_text(purpose, label="purpose"),
        "claimed_at_utc": claimed_at_utc,
        "previous_event_sha256": previous_hash,
        "key_id": ledger_key_id,
        "partial_retry_authorized": False,
        "authorizes_model_cuda_training_or_jarvis": False,
        "authorizes_private_label_access": False,
        "proves_external_atomicity": False,
        "authorization_runtime_sha256": _gvs_authorization_runtime_sha256(),
        "event_sha256": "0" * 64,
        "event_hmac_sha256": "0" * 64,
    }
    event["event_sha256"] = _sha256_json(_event_hash_payload(event), domain=_EVENT_HASH_DOMAIN)
    event["event_hmac_sha256"] = _hmac_sha256(
        ledger_key,
        _event_mac_payload(event),
        domain=_EVENT_HMAC_DOMAIN,
    )
    event = _validate_claim_event(event, expected_sequence=len(ledger_snapshot))
    event_bytes = _canonical_bytes(event)
    # Capture the post-callback verifier boundary before entering untrusted callback code.  A
    # callback that merely rebinds module globals cannot replace these local references or the
    # import-time runtime value used for the postcondition.
    post_callback_runtime_probe = _gvs_authorization_runtime_sha256
    post_callback_pinned_runtime = _PINNED_GVS_AUTHORIZATION_RUNTIME_SHA256
    post_callback_strict_sha256 = _strict_sha256
    post_callback_compare_digest = _HMAC_COMPARE_DIGEST
    post_callback_snapshot = _snapshot_mapping
    try:
        committed_head = durable_compare_and_append(event_bytes, previous_hash)
    except Exception as error:
        raise GVSAuthorizationError("durable population-claim append failed") from error
    if post_callback_runtime_probe() != post_callback_pinned_runtime:
        raise GVSAuthorizationError(
            "GVS authorization runtime changed across the durable callback boundary"
        )
    checked_committed_head = post_callback_strict_sha256(
        committed_head, label="durable callback committed head"
    )
    if not post_callback_compare_digest(checked_committed_head, event["event_sha256"]):
        raise GVSAuthorizationError("population claim callback acknowledged a different head")
    result = post_callback_snapshot(event, label="returned population claim event")
    _assert_gvs_authorization_runtime_integrity()
    return result


def _assert_import_bindings_unchanged() -> None:
    for name, pinned in _CONTRACT_OBJECT_PINS:
        if globals().get(name) is not pinned:
            raise GVSAuthorizationError(
                f"authorization contract {name} differs from its import-time identity"
            )
    current_and_pinned = (
        ("hashlib.sha256", hashlib.sha256, _HASHLIB_SHA256),
        ("hmac.new", hmac.new, _HMAC_NEW),
        ("hmac.compare_digest", hmac.compare_digest, _HMAC_COMPARE_DIGEST),
        ("json.dumps", json.dumps, _JSON_DUMPS),
        ("json.loads", json.loads, _JSON_LOADS),
        ("json.JSONDecodeError", json.JSONDecodeError, _JSON_DECODE_ERROR),
        ("math.isfinite", math.isfinite, _MATH_ISFINITE),
        ("unicodedata.category", unicodedata.category, _UNICODE_CATEGORY),
        ("unicodedata.normalize", unicodedata.normalize, _UNICODE_NORMALIZE),
        ("datetime", datetime, _DATETIME_CLASS),
        (
            "assert_sim_program_runtime_integrity",
            assert_sim_program_runtime_integrity,
            _ASSERT_SIM_PROGRAM_RUNTIME_INTEGRITY,
        ),
        ("module_runtime_sha256", module_runtime_sha256, _MODULE_RUNTIME_SHA256),
        ("runtime_callable_identity", runtime_callable_identity, _RUNTIME_CALLABLE_IDENTITY),
        ("sim_program_runtime_sha256", sim_program_runtime_sha256, _SIM_PROGRAM_RUNTIME_SHA256),
    )
    for name, current, pinned in current_and_pinned:
        if current is not pinned:
            raise GVSAuthorizationError(
                f"authorization runtime import {name} differs from its import-time identity"
            )


def _gvs_authorization_runtime_sha256() -> str:
    """Bind exact source, loaded code, schemas, cryptographic imports, and lower runtime."""

    _assert_import_bindings_unchanged()
    _ASSERT_SIM_PROGRAM_RUNTIME_INTEGRITY()
    dependencies: dict[str, object] = {}
    for name, value in (
        ("datetime.fromisoformat", _DATETIME_FROMISOFORMAT),
        ("hashlib.sha256", _HASHLIB_SHA256),
        ("hmac.compare_digest", _HMAC_COMPARE_DIGEST),
        ("hmac.new", _HMAC_NEW),
        ("json.dumps", _JSON_DUMPS),
        ("json.loads", _JSON_LOADS),
        ("math.isfinite", _MATH_ISFINITE),
        ("sha256_pattern.fullmatch", _SHA256_RE.fullmatch),
        ("identifier_pattern.fullmatch", _IDENTIFIER_RE.fullmatch),
        ("utc_pattern.fullmatch", _UTC_RE.fullmatch),
        ("unicodedata.category", _UNICODE_CATEGORY),
        ("unicodedata.normalize", _UNICODE_NORMALIZE),
    ):
        identity = _RUNTIME_CALLABLE_IDENTITY(value)
        dependencies[name] = _snapshot_mapping(
            dict(identity), label=f"authorization runtime dependency {name}"
        )
    contract = {
        "schema_versions": {
            "authorization_receipt": GVS_AUTHORIZATION_RECEIPT_VERSION,
            "population_claim_event": GVS_POPULATION_CLAIM_EVENT_VERSION,
        },
        "phases": sorted(GVS_PHASES),
        "phase_order": list(GVS_PHASE_ORDER),
        "binding_fields": sorted(GVS_AUTHORIZATION_BINDING_FIELDS),
        "receipt_fields": sorted(_RECEIPT_FIELDS),
        "claim_event_fields": sorted(_CLAIM_EVENT_FIELDS),
        "limits": {
            "json_depth": _MAX_JSON_DEPTH,
            "json_nodes": _MAX_JSON_NODES,
            "serialized_bytes": _MAX_SERIALIZED_BYTES,
            "gates": _MAX_GATES,
            "token_id": _MAX_TOKEN_ID,
            "hmac_key_bytes": _MAX_HMAC_KEY_BYTES,
        },
        "patterns": {
            "sha256": {
                "pattern": _SHA256_RE.pattern,
                "type": f"{type(_SHA256_RE).__module__}.{type(_SHA256_RE).__qualname__}",
            },
            "identifier": {
                "pattern": _IDENTIFIER_RE.pattern,
                "type": f"{type(_IDENTIFIER_RE).__module__}.{type(_IDENTIFIER_RE).__qualname__}",
            },
            "utc": {
                "pattern": _UTC_RE.pattern,
                "type": f"{type(_UTC_RE).__module__}.{type(_UTC_RE).__qualname__}",
            },
        },
        "domains": {
            "receipt_hash": _RECEIPT_HASH_DOMAIN.hex(),
            "receipt_hmac": _RECEIPT_HMAC_DOMAIN.hex(),
            "event_hash": _EVENT_HASH_DOMAIN.hex(),
            "event_hmac": _EVENT_HMAC_DOMAIN.hex(),
        },
        "sim_program_runtime_sha256": _SIM_PROGRAM_RUNTIME_SHA256(),
        "dependencies": dependencies,
    }
    try:
        return _MODULE_RUNTIME_SHA256(
            globals(),
            module_name=__name__,
            source_path=__file__,
            contract=contract,
        )
    except SimProgramError as error:
        raise GVSAuthorizationError("could not bind the GVS authorization runtime") from error


def _assert_gvs_authorization_runtime_integrity() -> None:
    """Fail closed if source, live code, imports, constants, or lower runtime changed."""

    current = _gvs_authorization_runtime_sha256()
    if current != _PINNED_GVS_AUTHORIZATION_RUNTIME_SHA256:
        raise GVSAuthorizationError(
            "GVS authorization runtime differs from its import-time identity"
        )


def gvs_authorization_runtime_sha256() -> str:
    """Return the independently recomputed authorization runtime commitment."""

    return _gvs_authorization_runtime_sha256()


def assert_gvs_authorization_runtime_integrity() -> None:
    """Public runtime-integrity check; protocol entrypoints use the private guard."""

    _assert_gvs_authorization_runtime_integrity()


__all__ = [
    "GVS_AUTHORIZATION_BINDING_FIELDS",
    "GVS_AUTHORIZATION_RECEIPT_VERSION",
    "GVS_PHASES",
    "GVS_PHASE_ORDER",
    "GVS_POPULATION_CLAIM_EVENT_VERSION",
    "GVSAuthorizationError",
    "assert_gvs_authorization_runtime_integrity",
    "claim_gvs_population_once",
    "gvs_authorization_runtime_sha256",
    "load_gvs_authorization_receipt_json",
    "load_gvs_population_claim_ledger_json",
    "seal_gvs_authorization_receipt",
    "validate_gvs_authorization_receipt",
    "validate_gvs_population_claim_ledger",
]


_PINNED_GVS_AUTHORIZATION_RUNTIME_SHA256 = _gvs_authorization_runtime_sha256()
