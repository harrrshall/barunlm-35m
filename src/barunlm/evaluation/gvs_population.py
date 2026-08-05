"""CPU-only population firewall primitives for proposed BarunAction GVS-v1.

The firewall binds metadata commitments for ``T-new``, ``D-support``, ``S-new``,
and ``C-new`` before any model outcome is observed. It computes connected
components across every frozen provenance and duplicate lineage, rejects any
direct or transitive component that crosses population roles, rejects components
whose members disagree on eligibility or a pre-outcome classification, and records
all pre-outcome exclusions.

Every returned artifact says that it is structural-only and cannot authorize model
or label access. The scan artifact is an exact detached commitment, not proof that a
scan ran honestly: later authenticated custody and one-shot receipts remain mandatory.
To fail closed, all four roles must be nonempty, human S/C records must attest to no
model assistance, every role assignment and authorship must postdate the compared
release cutoff, entity IDs must be sorted, every chronology comparison is strict, and
every raw JSON/document/cardinality limit is enforced before an artifact is used.

This module does not inspect prompt or label bytes, set sample-size thresholds, score
outcomes, authorize label access, load a model, or contact a remote service. External
collection and custody systems remain responsible for proving that the committed
metadata and scan artifacts correspond to the protected source bytes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from datetime import timezone as datetime_timezone
from typing import Any

from barunlm.evaluation.sim_program import (
    SimProgramError,
    assert_sim_program_runtime_integrity,
    module_runtime_sha256,
    runtime_callable_identity,
    sim_program_runtime_sha256,
)

GVS_POPULATION_RECORD_SCHEMA_VERSION = "barun-gvs-population-record-v1"
GVS_POPULATION_FIREWALL_SCHEMA_VERSION = "barun-gvs-population-firewall-v1"
GVS_SCAN_EVIDENCE_SCHEMA_VERSION = "barun-gvs-scan-evidence-v1"

MAX_RECORD_JSON_BYTES = 64 * 1024
MAX_POPULATION_METADATA_JSON_BYTES = 256 * 1024 * 1024
MAX_SCAN_EVIDENCE_JSON_BYTES = 64 * 1024 * 1024
MAX_FIREWALL_JSON_BYTES = 64 * 1024 * 1024
MAX_RECORDS_PER_ROLE = 100_000
MAX_TOTAL_RECORDS = 200_000
MAX_ENTITY_POOL_IDS = 64

POPULATION_ROLES = ("T-new", "D-support", "S-new", "C-new")
TASK_CLASSES = frozenset({"efficacy", "safety"})
EXPECTED_OUTCOMES = frozenset({"ACTION", "ABSTAIN", "CLARIFY", "CONFIRM"})
ACTION_FAMILIES = frozenset(
    {
        "reminders",
        "calendars",
        "contacts",
        "notes",
        "lists",
        "maps",
        "messages",
        "media",
        "device_settings",
        "multi_family",
        "none",
    }
)
AUDITED_BOOLEAN_STRATA = (
    "context_grounding",
    "revision",
    "disfluency",
    "distractor_tools",
    "timezone_or_relative_time",
    "multi_action",
    "renamed_schema",
    "unseen_schema",
    "unsafe_or_adversarial",
)
PRE_OUTCOME_EXCLUSION_REASONS = frozenset(
    {
        "collection_quality",
        "duplicate_or_lineage_conflict",
        "invalid_schema",
        "license_or_consent",
        "provenance_incomplete",
        "semantic_review_failure",
        "simulator_roundtrip_failure",
        "token_limit",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_UTC_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_MAX_JSON_DEPTH = 64

# Security-relevant imports are used through stable aliases and their module-visible
# identities are checked at every public boundary.  This is tamper evidence for an
# ordinary long-lived Python worker, not a replacement for an isolated trusted runtime.
_COPY_DEEPCOPY = copy.deepcopy
_HASHLIB_SHA256 = hashlib.sha256
_JSON_DUMPS = json.dumps
_JSON_LOADS = json.loads
_JSON_DECODE_ERROR = json.JSONDecodeError
_MATH_ISFINITE = math.isfinite
_UNICODE_NORMALIZE = unicodedata.normalize
_DATETIME_CLASS = datetime
_DATETIME_TIMEZONE_CLASS = datetime_timezone
_DATETIME_FROMISOFORMAT = datetime.fromisoformat
_DEFAULTDICT_CLASS = defaultdict
_ASSERT_SIM_PROGRAM_RUNTIME_INTEGRITY = assert_sim_program_runtime_integrity
_MODULE_RUNTIME_SHA256 = module_runtime_sha256
_RUNTIME_CALLABLE_IDENTITY = runtime_callable_identity
_SIM_PROGRAM_RUNTIME_SHA256 = sim_program_runtime_sha256

_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "record_id",
        "population_role",
        "task_class",
        "expected_outcome",
        "action_family",
        "strata",
        "provenance",
        "duplicate_evidence",
        "eligibility",
        "integrity",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "author_id",
        "source_id",
        "collection_batch",
        "schema_family",
        "program_template_id",
        "paraphrase_family",
        "entity_pool_ids",
        "temporal_construction_id",
        "source_commitment_sha256",
        "role_assigned_at_utc",
        "authored_at_utc",
        "license",
        "consent",
        "no_model_assistance",
        "authoring_protocol_revision",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "exact_scan_manifest_sha256",
        "near_scan_manifest_sha256",
        "scan_config_sha256",
        "scan_code_sha256",
        "lineage_config_sha256",
        "firewall_code_sha256",
    }
)
_DUPLICATE_EVIDENCE_FIELDS = frozenset(
    {
        "exact_content_sha256",
        "normalized_request_sha256",
        "delexicalized_template_sha256",
        "near_duplicate_lineage_id",
        "scan_completed_at_utc",
        *_BINDING_FIELDS,
    }
)
_ELIGIBILITY_FIELDS = frozenset(
    {
        "eligible",
        "decision",
        "reason_code",
        "decision_stage",
        "evidence_sha256",
        "decided_at_utc",
    }
)
_INTEGRITY_FIELDS = frozenset({"record_sha256"})
_FIREWALL_FIELDS = frozenset(
    {
        "schema_version",
        "structural_only",
        "authorizes_model_or_label_access",
        "requires_authenticated_custody_receipt",
        "compared_release_cutoff_utc",
        "scan_evidence_frozen_at_utc",
        "firewall_frozen_at_utc",
        "scan_bindings",
        "scan_bindings_sha256",
        "scan_evidence_sha256",
        "population_membership_sha256",
        "record_set_sha256",
        "populations",
        "eligible_component_summaries",
        "graph",
        "graph_sha256",
        "strata_assignments",
        "strata_sha256",
        "eligibility_assignments",
        "exclusions",
        "eligibility_sha256",
        "firewall_sha256",
    }
)
_SCAN_ENTRY_FIELDS = frozenset(
    {
        "record_id",
        "population_role",
        "record_sha256",
        "source_commitment_sha256",
        "exact_content_sha256",
        "normalized_request_sha256",
        "delexicalized_template_sha256",
        "near_duplicate_lineage_id",
        "lineage_commitment_sha256",
        "scan_completed_at_utc",
    }
)
_SCAN_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "structural_only",
        "authorizes_model_or_label_access",
        "requires_authenticated_custody_receipt",
        "compared_release_cutoff_utc",
        "scan_evidence_frozen_at_utc",
        "scan_bindings",
        "scan_bindings_sha256",
        "record_membership_sha256",
        "record_set_sha256",
        "entries",
        "entries_sha256",
        "scan_evidence_sha256",
    }
)


class GVSPopulationError(ValueError):
    """Population metadata violates the GVS-v1 structural firewall."""


_CONTRACT_OBJECT_PINS = (
    ("GVS_POPULATION_RECORD_SCHEMA_VERSION", GVS_POPULATION_RECORD_SCHEMA_VERSION),
    ("GVS_POPULATION_FIREWALL_SCHEMA_VERSION", GVS_POPULATION_FIREWALL_SCHEMA_VERSION),
    ("GVS_SCAN_EVIDENCE_SCHEMA_VERSION", GVS_SCAN_EVIDENCE_SCHEMA_VERSION),
    ("MAX_RECORD_JSON_BYTES", MAX_RECORD_JSON_BYTES),
    ("MAX_POPULATION_METADATA_JSON_BYTES", MAX_POPULATION_METADATA_JSON_BYTES),
    ("MAX_SCAN_EVIDENCE_JSON_BYTES", MAX_SCAN_EVIDENCE_JSON_BYTES),
    ("MAX_FIREWALL_JSON_BYTES", MAX_FIREWALL_JSON_BYTES),
    ("MAX_RECORDS_PER_ROLE", MAX_RECORDS_PER_ROLE),
    ("MAX_TOTAL_RECORDS", MAX_TOTAL_RECORDS),
    ("MAX_ENTITY_POOL_IDS", MAX_ENTITY_POOL_IDS),
    ("POPULATION_ROLES", POPULATION_ROLES),
    ("TASK_CLASSES", TASK_CLASSES),
    ("EXPECTED_OUTCOMES", EXPECTED_OUTCOMES),
    ("ACTION_FAMILIES", ACTION_FAMILIES),
    ("AUDITED_BOOLEAN_STRATA", AUDITED_BOOLEAN_STRATA),
    ("PRE_OUTCOME_EXCLUSION_REASONS", PRE_OUTCOME_EXCLUSION_REASONS),
    ("_SHA256_RE", _SHA256_RE),
    ("_IDENTIFIER_RE", _IDENTIFIER_RE),
    ("_UTC_TIMESTAMP_RE", _UTC_TIMESTAMP_RE),
    ("_MAX_JSON_DEPTH", _MAX_JSON_DEPTH),
    ("_RECORD_FIELDS", _RECORD_FIELDS),
    ("_PROVENANCE_FIELDS", _PROVENANCE_FIELDS),
    ("_BINDING_FIELDS", _BINDING_FIELDS),
    ("_DUPLICATE_EVIDENCE_FIELDS", _DUPLICATE_EVIDENCE_FIELDS),
    ("_ELIGIBILITY_FIELDS", _ELIGIBILITY_FIELDS),
    ("_INTEGRITY_FIELDS", _INTEGRITY_FIELDS),
    ("_FIREWALL_FIELDS", _FIREWALL_FIELDS),
    ("_SCAN_ENTRY_FIELDS", _SCAN_ENTRY_FIELDS),
    ("_SCAN_EVIDENCE_FIELDS", _SCAN_EVIDENCE_FIELDS),
    ("GVSPopulationError", GVSPopulationError),
)


def _strict_json(value: object, *, path: str = "$", depth: int = 0) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise GVSPopulationError(f"{path} exceeds maximum JSON depth {_MAX_JSON_DEPTH}")
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        if value_type is str:
            try:
                value.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise GVSPopulationError(f"{path} is not valid UTF-8") from error
        return
    if value_type is float:
        if not _MATH_ISFINITE(value):
            raise GVSPopulationError(f"{path} contains a non-finite number")
        return
    if value_type is list:
        for index, child in enumerate(value):
            _strict_json(child, path=f"{path}[{index}]", depth=depth + 1)
        return
    if value_type is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise GVSPopulationError(f"{path} object keys must be exact strings")
            _strict_json(key, path=f"{path}.<key>", depth=depth + 1)
            _strict_json(child, path=f"{path}.{key}", depth=depth + 1)
        return
    raise GVSPopulationError(f"{path} contains non-JSON type {value_type.__name__}")


def _canonical_bytes(value: object) -> bytes:
    _strict_json(value)
    try:
        return _JSON_DUMPS(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise GVSPopulationError("value is not strict canonical JSON") from error


def _canonical_sha256(value: object) -> str:
    return _HASHLIB_SHA256(_canonical_bytes(value)).hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash exact canonical JSON after rejecting custom containers and non-finite values."""

    _assert_gvs_population_runtime_integrity()
    result = _canonical_sha256(value)
    _assert_gvs_population_runtime_integrity()
    return result


def _snapshot_object(
    value: object,
    *,
    label: str,
    fields: frozenset[str],
    maximum_bytes: int | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise GVSPopulationError(f"{label} must be an exact JSON object")
    try:
        encoded = _canonical_bytes(value)
        if maximum_bytes is not None and len(encoded) > maximum_bytes:
            raise GVSPopulationError(f"{label} exceeds its bounded JSON size")
        snapshot = _JSON_LOADS(encoded)
    except (_JSON_DECODE_ERROR, RuntimeError) as error:
        raise GVSPopulationError(f"{label} could not be snapshotted") from error
    return _exact_object(snapshot, label=label, fields=fields)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GVSPopulationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise GVSPopulationError(f"non-finite JSON constant {value!r}")


def _loads_bounded_json(text: object, *, label: str, maximum_bytes: int) -> object:
    if type(text) is not str:
        raise GVSPopulationError(f"{label} JSON must be a string")
    try:
        encoded = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise GVSPopulationError(f"{label} JSON is not valid UTF-8") from error
    if len(encoded) > maximum_bytes:
        raise GVSPopulationError(f"{label} JSON exceeds its bounded size")
    try:
        return _JSON_LOADS(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except GVSPopulationError:
        raise
    except (_JSON_DECODE_ERROR, RecursionError, UnicodeError, ValueError) as error:
        raise GVSPopulationError(f"{label} is not strict JSON") from error


def _exact_object(value: object, *, label: str, fields: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict:
        raise GVSPopulationError(f"{label} must be an exact JSON object")
    _strict_json(value, path=label)
    actual = set(value)
    missing = sorted(fields.difference(actual))
    extra = sorted(actual.difference(fields))
    if missing or extra:
        raise GVSPopulationError(f"{label} fields changed; missing={missing!r}, extra={extra!r}")
    return value


def _identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None or "@" in value:
        raise GVSPopulationError(f"{label} must be a pseudonymous safe identifier")
    if value != _UNICODE_NORMALIZE("NFC", value):
        raise GVSPopulationError(f"{label} must be NFC-normalized")
    return value


def _text(value: object, *, label: str, maximum_length: int = 512) -> str:
    if type(value) is not str or not value or len(value) > maximum_length:
        raise GVSPopulationError(
            f"{label} must be a nonempty string of at most {maximum_length} characters"
        )
    if value != _UNICODE_NORMALIZE("NFC", value):
        raise GVSPopulationError(f"{label} must be NFC-normalized")
    if value != value.strip() or any(character in value for character in "\r\n\x00"):
        raise GVSPopulationError(f"{label} contains forbidden whitespace or control characters")
    return value


def _utc(value: object, *, label: str) -> datetime:
    text = _text(value, label=label, maximum_length=32)
    if _UTC_TIMESTAMP_RE.fullmatch(text) is None:
        raise GVSPopulationError(f"{label} must be a full RFC 3339 UTC timestamp ending in Z")
    try:
        parsed = _DATETIME_FROMISOFORMAT(text[:-1] + "+00:00")
    except ValueError as error:
        raise GVSPopulationError(f"{label} is not a valid UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != datetime_timezone.utc.utcoffset(parsed):
        raise GVSPopulationError(f"{label} must be UTC")
    return parsed


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise GVSPopulationError(f"{label} must be a lowercase SHA-256")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise GVSPopulationError(f"{label} must be a boolean")
    return value


def _record_hash_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = _COPY_DEEPCOPY(dict(record))
    payload["integrity"]["record_sha256"] = "0" * 64
    return payload


def _validate_record_snapshot(
    record: dict[str, Any], *, compared_release_cutoff_utc: str
) -> dict[str, Any]:
    checked = _exact_object(record, label="population_record", fields=_RECORD_FIELDS)
    cutoff = _utc(compared_release_cutoff_utc, label="compared_release_cutoff_utc")
    if checked["schema_version"] != GVS_POPULATION_RECORD_SCHEMA_VERSION:
        raise GVSPopulationError(
            f"population record schema must be {GVS_POPULATION_RECORD_SCHEMA_VERSION!r}"
        )
    _identifier(checked["record_id"], label="record_id")
    if checked["population_role"] not in POPULATION_ROLES:
        raise GVSPopulationError(f"population_role must be one of {POPULATION_ROLES!r}")
    if checked["task_class"] not in TASK_CLASSES:
        raise GVSPopulationError(f"task_class must be one of {sorted(TASK_CLASSES)!r}")
    if checked["expected_outcome"] not in EXPECTED_OUTCOMES:
        raise GVSPopulationError(f"expected_outcome must be one of {sorted(EXPECTED_OUTCOMES)!r}")
    if checked["action_family"] not in ACTION_FAMILIES:
        raise GVSPopulationError(f"action_family must be one of {sorted(ACTION_FAMILIES)!r}")

    strata = _exact_object(
        checked["strata"],
        label="strata",
        fields=frozenset(AUDITED_BOOLEAN_STRATA),
    )
    for name in AUDITED_BOOLEAN_STRATA:
        _boolean(strata[name], label=f"strata.{name}")

    provenance = _exact_object(checked["provenance"], label="provenance", fields=_PROVENANCE_FIELDS)
    for name in (
        "author_id",
        "source_id",
        "collection_batch",
        "schema_family",
        "program_template_id",
        "paraphrase_family",
        "temporal_construction_id",
        "authoring_protocol_revision",
    ):
        _identifier(provenance[name], label=f"provenance.{name}")
    _sha256(
        provenance["source_commitment_sha256"],
        label="provenance.source_commitment_sha256",
    )
    role_assigned = _utc(
        provenance["role_assigned_at_utc"], label="provenance.role_assigned_at_utc"
    )
    authored = _utc(provenance["authored_at_utc"], label="provenance.authored_at_utc")
    if not cutoff < role_assigned < authored:
        raise GVSPopulationError(
            "chronology must satisfy compared-release cutoff < role assignment < authoring"
        )
    _text(provenance["license"], label="provenance.license")
    if _boolean(provenance["consent"], label="provenance.consent") is not True:
        raise GVSPopulationError("provenance.consent must be true")
    no_model_assistance = _boolean(
        provenance["no_model_assistance"], label="provenance.no_model_assistance"
    )
    if checked["population_role"] in {"S-new", "C-new"} and not no_model_assistance:
        raise GVSPopulationError("human S-new/C-new records require no_model_assistance=true")
    entity_pool_ids = provenance["entity_pool_ids"]
    if type(entity_pool_ids) is not list or not entity_pool_ids:
        raise GVSPopulationError("provenance.entity_pool_ids must be a nonempty JSON array")
    if len(entity_pool_ids) > MAX_ENTITY_POOL_IDS:
        raise GVSPopulationError(
            f"provenance.entity_pool_ids exceeds maximum {MAX_ENTITY_POOL_IDS}"
        )
    checked_entity_pools = [
        _identifier(value, label="provenance.entity_pool_ids[]") for value in entity_pool_ids
    ]
    if len(set(checked_entity_pools)) != len(checked_entity_pools):
        raise GVSPopulationError("provenance.entity_pool_ids must be unique")
    if checked_entity_pools != sorted(checked_entity_pools):
        raise GVSPopulationError("provenance.entity_pool_ids must be canonically sorted")

    duplicate_evidence = _exact_object(
        checked["duplicate_evidence"],
        label="duplicate_evidence",
        fields=_DUPLICATE_EVIDENCE_FIELDS,
    )
    for name in (
        "exact_content_sha256",
        "normalized_request_sha256",
        "delexicalized_template_sha256",
        *_BINDING_FIELDS,
    ):
        _sha256(duplicate_evidence[name], label=f"duplicate_evidence.{name}")
    _identifier(
        duplicate_evidence["near_duplicate_lineage_id"],
        label="duplicate_evidence.near_duplicate_lineage_id",
    )
    scan_completed = _utc(
        duplicate_evidence["scan_completed_at_utc"],
        label="duplicate_evidence.scan_completed_at_utc",
    )

    eligibility = _exact_object(
        checked["eligibility"], label="eligibility", fields=_ELIGIBILITY_FIELDS
    )
    eligible = _boolean(eligibility["eligible"], label="eligibility.eligible")
    expected_decision = "include" if eligible else "exclude"
    if eligibility["decision"] != expected_decision:
        raise GVSPopulationError(f"eligibility.decision must be {expected_decision!r}")
    if eligibility["decision_stage"] != "pre-outcome":
        raise GVSPopulationError("eligibility.decision_stage must be 'pre-outcome'")
    if eligible:
        if eligibility["reason_code"] is not None:
            raise GVSPopulationError("included records must have null eligibility.reason_code")
    elif eligibility["reason_code"] not in PRE_OUTCOME_EXCLUSION_REASONS:
        raise GVSPopulationError(
            "excluded records require a frozen pre-outcome eligibility.reason_code"
        )
    _sha256(eligibility["evidence_sha256"], label="eligibility.evidence_sha256")
    eligibility_decided = _utc(eligibility["decided_at_utc"], label="eligibility.decided_at_utc")
    if not authored < scan_completed < eligibility_decided:
        raise GVSPopulationError(
            "chronology must satisfy authoring < scan completion < eligibility decision"
        )

    integrity = _exact_object(checked["integrity"], label="integrity", fields=_INTEGRITY_FIELDS)
    received_hash = _sha256(integrity["record_sha256"], label="integrity.record_sha256")
    expected_hash = _canonical_sha256(_record_hash_payload(checked))
    if received_hash != expected_hash:
        raise GVSPopulationError("population record hash mismatch")
    return checked


def seal_population_record(
    record: Mapping[str, Any], *, compared_release_cutoff_utc: str
) -> dict[str, Any]:
    """Seal one exact-shape record and return a detached strict-JSON snapshot."""

    _assert_gvs_population_runtime_integrity()
    result = _snapshot_object(
        record,
        label="population_record",
        fields=_RECORD_FIELDS,
        maximum_bytes=MAX_RECORD_JSON_BYTES,
    )
    provenance = _exact_object(result["provenance"], label="provenance", fields=_PROVENANCE_FIELDS)
    entity_pool_ids = provenance["entity_pool_ids"]
    if type(entity_pool_ids) is not list or not entity_pool_ids:
        raise GVSPopulationError("provenance.entity_pool_ids must be a nonempty exact JSON array")
    if len(entity_pool_ids) > MAX_ENTITY_POOL_IDS:
        raise GVSPopulationError(
            f"provenance.entity_pool_ids exceeds maximum {MAX_ENTITY_POOL_IDS}"
        )
    canonical_entity_pool_ids = [
        _identifier(value, label="provenance.entity_pool_ids[]") for value in entity_pool_ids
    ]
    if len(set(canonical_entity_pool_ids)) != len(canonical_entity_pool_ids):
        raise GVSPopulationError("provenance.entity_pool_ids must be unique")
    provenance["entity_pool_ids"] = sorted(canonical_entity_pool_ids)
    integrity = _exact_object(result["integrity"], label="integrity", fields=_INTEGRITY_FIELDS)
    integrity["record_sha256"] = "0" * 64
    integrity["record_sha256"] = _canonical_sha256(_record_hash_payload(result))
    checked = _validate_record_snapshot(
        result, compared_release_cutoff_utc=compared_release_cutoff_utc
    )
    _assert_gvs_population_runtime_integrity()
    return checked


def validate_population_record(
    record: object, *, compared_release_cutoff_utc: str
) -> dict[str, Any]:
    """Validate one sealed record from a single caller-detached snapshot."""

    _assert_gvs_population_runtime_integrity()
    checked = _snapshot_object(
        record,
        label="population_record",
        fields=_RECORD_FIELDS,
        maximum_bytes=MAX_RECORD_JSON_BYTES,
    )
    result = _validate_record_snapshot(
        checked, compared_release_cutoff_utc=compared_release_cutoff_utc
    )
    _assert_gvs_population_runtime_integrity()
    return result


def loads_population_record(text: str, *, compared_release_cutoff_utc: str) -> dict[str, Any]:
    """Load one bounded record, rejecting duplicate keys and non-finite constants."""

    _assert_gvs_population_runtime_integrity()
    payload = _loads_bounded_json(
        text, label="population_record", maximum_bytes=MAX_RECORD_JSON_BYTES
    )
    result = validate_population_record(
        payload, compared_release_cutoff_utc=compared_release_cutoff_utc
    )
    _assert_gvs_population_runtime_integrity()
    return result


def _validate_scan_bindings(value: object) -> dict[str, str]:
    bindings = _snapshot_object(value, label="scan_bindings", fields=_BINDING_FIELDS)
    return {
        name: _sha256(bindings[name], label=f"scan_bindings.{name}")
        for name in sorted(_BINDING_FIELDS)
    }


def _snapshot_role_arrays(value: object, *, label: str) -> dict[str, list[Any]]:
    if type(value) is not dict:
        raise GVSPopulationError(f"{label} must be an exact JSON object")
    actual = set(value)
    expected = set(POPULATION_ROLES)
    if actual != expected:
        raise GVSPopulationError(
            f"{label} fields changed; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    total = 0
    for role in POPULATION_ROLES:
        role_value = value[role]
        if type(role_value) is not list:
            raise GVSPopulationError(f"{label}.{role} must be an exact JSON array")
        if not role_value:
            raise GVSPopulationError(f"{label}.{role} must be nonempty")
        if len(role_value) > MAX_RECORDS_PER_ROLE:
            raise GVSPopulationError(
                f"{label}.{role} exceeds maximum {MAX_RECORDS_PER_ROLE} records"
            )
        total += len(role_value)
    if total > MAX_TOTAL_RECORDS:
        raise GVSPopulationError(f"{label} exceeds maximum {MAX_TOTAL_RECORDS} total records")
    checked = _snapshot_object(
        value,
        label=label,
        fields=frozenset(POPULATION_ROLES),
        maximum_bytes=MAX_POPULATION_METADATA_JSON_BYTES,
    )
    detached_total = 0
    for role in POPULATION_ROLES:
        role_value = checked[role]
        if type(role_value) is not list or not role_value:
            raise GVSPopulationError(f"{label}.{role} changed during snapshot")
        if len(role_value) > MAX_RECORDS_PER_ROLE:
            raise GVSPopulationError(
                f"{label}.{role} exceeds maximum {MAX_RECORDS_PER_ROLE} records"
            )
        detached_total += len(role_value)
    if detached_total > MAX_TOTAL_RECORDS:
        raise GVSPopulationError(f"{label} exceeds maximum {MAX_TOTAL_RECORDS} total records")
    return checked


def _validated_membership(value: object) -> dict[str, list[str]]:
    membership = _snapshot_role_arrays(value, label="expected_membership")
    result: dict[str, list[str]] = {}
    global_ids: set[str] = set()
    for role in POPULATION_ROLES:
        ids = [
            _identifier(item, label=f"expected_membership.{role}[]") for item in membership[role]
        ]
        if len(ids) != len(set(ids)):
            raise GVSPopulationError(f"expected_membership.{role} contains duplicate record IDs")
        overlap = global_ids.intersection(ids)
        if overlap:
            raise GVSPopulationError(
                f"expected membership record IDs cross roles: {sorted(overlap)!r}"
            )
        global_ids.update(ids)
        result[role] = sorted(ids)
    return result


def _collect_validated_records(
    population_snapshot: Mapping[str, list[Any]],
    *,
    membership: Mapping[str, list[str]],
    scan_bindings: Mapping[str, str],
    compared_release_cutoff_utc: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    record_hashes: set[str] = set()
    for role in POPULATION_ROLES:
        checked_role: list[dict[str, Any]] = []
        for index, value in enumerate(population_snapshot[role]):
            if type(value) is not dict:
                raise GVSPopulationError(
                    f"populations.{role}[{index}] must be an exact JSON object"
                )
            if len(_canonical_bytes(value)) > MAX_RECORD_JSON_BYTES:
                raise GVSPopulationError(
                    f"populations.{role}[{index}] exceeds its bounded JSON size"
                )
            record = _validate_record_snapshot(
                value, compared_release_cutoff_utc=compared_release_cutoff_utc
            )
            if record["population_role"] != role:
                raise GVSPopulationError(
                    f"populations.{role}[{index}] is relabeled as {record['population_role']!r}"
                )
            record_id = record["record_id"]
            record_hash = record["integrity"]["record_sha256"]
            if record_id in record_ids:
                raise GVSPopulationError("population record IDs must be globally unique")
            if record_hash in record_hashes:
                raise GVSPopulationError("population record hashes must be globally unique")
            for name in _BINDING_FIELDS:
                if record["duplicate_evidence"][name] != scan_bindings[name]:
                    raise GVSPopulationError(f"record {record_id!r} is bound to a different {name}")
            record_ids.add(record_id)
            record_hashes.add(record_hash)
            checked_role.append(record)
            records.append(record)

        actual_ids = sorted(record["record_id"] for record in checked_role)
        missing = sorted(set(membership[role]).difference(actual_ids))
        extra = sorted(set(actual_ids).difference(membership[role]))
        if missing or extra:
            raise GVSPopulationError(
                f"{role} membership changed; missing={missing!r}, extra={extra!r}"
            )
    records.sort(key=lambda record: record["integrity"]["record_sha256"])
    return records


def _lineages(record: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    provenance = record["provenance"]
    duplicate = record["duplicate_evidence"]
    values: list[tuple[str, str]] = [
        ("author_id", provenance["author_id"]),
        ("source_id", provenance["source_id"]),
        ("collection_batch", provenance["collection_batch"]),
        ("schema_family", provenance["schema_family"]),
        ("program_template_id", provenance["program_template_id"]),
        ("paraphrase_family", provenance["paraphrase_family"]),
        ("temporal_construction_id", provenance["temporal_construction_id"]),
        ("source_commitment_sha256", provenance["source_commitment_sha256"]),
        ("exact_content_sha256", duplicate["exact_content_sha256"]),
        ("normalized_request_sha256", duplicate["normalized_request_sha256"]),
        ("delexicalized_template_sha256", duplicate["delexicalized_template_sha256"]),
        ("near_duplicate_lineage_id", duplicate["near_duplicate_lineage_id"]),
    ]
    values.extend(("entity_pool_id", value) for value in provenance["entity_pool_ids"])
    return tuple(values)


def _scan_evidence_hash_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    payload = _COPY_DEEPCOPY(dict(evidence))
    payload["scan_evidence_sha256"] = "0" * 64
    return payload


def build_scan_evidence(
    populations: Mapping[str, list[Mapping[str, Any]]],
    *,
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
) -> dict[str, Any]:
    """Build a non-authorizing exact commitment to scan inputs and lineage evidence."""

    _assert_gvs_population_runtime_integrity()
    population_snapshot = _snapshot_role_arrays(populations, label="populations")
    membership = _validated_membership(expected_membership)
    scan_bindings = _validate_scan_bindings(expected_scan_bindings)
    cutoff = _utc(compared_release_cutoff_utc, label="compared_release_cutoff_utc")
    scan_frozen = _utc(scan_evidence_frozen_at_utc, label="scan_evidence_frozen_at_utc")
    if scan_frozen <= cutoff:
        raise GVSPopulationError(
            "scan evidence freeze must strictly postdate compared-release cutoff"
        )
    records = _collect_validated_records(
        population_snapshot,
        membership=membership,
        scan_bindings=scan_bindings,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
    )
    entries: list[dict[str, Any]] = []
    for record in records:
        scan_completed = _utc(
            record["duplicate_evidence"]["scan_completed_at_utc"],
            label="duplicate_evidence.scan_completed_at_utc",
        )
        eligibility_decided = _utc(
            record["eligibility"]["decided_at_utc"],
            label="eligibility.decided_at_utc",
        )
        if not scan_completed < scan_frozen < eligibility_decided:
            raise GVSPopulationError(
                "chronology must satisfy scan completion < scan-evidence freeze "
                "< eligibility decision"
            )
        duplicate = record["duplicate_evidence"]
        provenance = record["provenance"]
        lineage_payload = [{"kind": kind, "value": value} for kind, value in _lineages(record)]
        entries.append(
            {
                "record_id": record["record_id"],
                "population_role": record["population_role"],
                "record_sha256": record["integrity"]["record_sha256"],
                "source_commitment_sha256": provenance["source_commitment_sha256"],
                "exact_content_sha256": duplicate["exact_content_sha256"],
                "normalized_request_sha256": duplicate["normalized_request_sha256"],
                "delexicalized_template_sha256": duplicate["delexicalized_template_sha256"],
                "near_duplicate_lineage_id": duplicate["near_duplicate_lineage_id"],
                "lineage_commitment_sha256": _canonical_sha256(lineage_payload),
                "scan_completed_at_utc": duplicate["scan_completed_at_utc"],
            }
        )
    entries.sort(key=lambda entry: entry["record_sha256"])
    normalized_membership = {role: membership[role] for role in POPULATION_ROLES}
    evidence: dict[str, Any] = {
        "schema_version": GVS_SCAN_EVIDENCE_SCHEMA_VERSION,
        "structural_only": True,
        "authorizes_model_or_label_access": False,
        "requires_authenticated_custody_receipt": True,
        "compared_release_cutoff_utc": compared_release_cutoff_utc,
        "scan_evidence_frozen_at_utc": scan_evidence_frozen_at_utc,
        "scan_bindings": scan_bindings,
        "scan_bindings_sha256": _canonical_sha256(scan_bindings),
        "record_membership_sha256": _canonical_sha256(normalized_membership),
        "record_set_sha256": _canonical_sha256(
            sorted(record["integrity"]["record_sha256"] for record in records)
        ),
        "entries": entries,
        "entries_sha256": _canonical_sha256(entries),
        "scan_evidence_sha256": "0" * 64,
    }
    evidence["scan_evidence_sha256"] = _canonical_sha256(_scan_evidence_hash_payload(evidence))
    if len(_canonical_bytes(evidence)) > MAX_SCAN_EVIDENCE_JSON_BYTES:
        raise GVSPopulationError("scan evidence exceeds its bounded JSON size")
    _assert_gvs_population_runtime_integrity()
    return evidence


def validate_scan_evidence(
    evidence: object,
    *,
    populations: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
) -> dict[str, Any]:
    """Cross-check every scan entry and content commitment against exact membership."""

    _assert_gvs_population_runtime_integrity()
    checked = _snapshot_object(
        evidence,
        label="scan_evidence",
        fields=_SCAN_EVIDENCE_FIELDS,
        maximum_bytes=MAX_SCAN_EVIDENCE_JSON_BYTES,
    )
    if checked["schema_version"] != GVS_SCAN_EVIDENCE_SCHEMA_VERSION:
        raise GVSPopulationError(
            f"scan evidence schema must be {GVS_SCAN_EVIDENCE_SCHEMA_VERSION!r}"
        )
    if checked["structural_only"] is not True:
        raise GVSPopulationError("scan evidence must be structural_only=true")
    if checked["authorizes_model_or_label_access"] is not False:
        raise GVSPopulationError("scan evidence cannot authorize model or label access")
    if checked["requires_authenticated_custody_receipt"] is not True:
        raise GVSPopulationError("scan evidence must require an authenticated custody receipt")
    if checked["compared_release_cutoff_utc"] != compared_release_cutoff_utc:
        raise GVSPopulationError("scan evidence is bound to a different release cutoff")
    if checked["scan_evidence_frozen_at_utc"] != scan_evidence_frozen_at_utc:
        raise GVSPopulationError("scan evidence is bound to a different freeze time")
    entries = checked["entries"]
    if type(entries) is not list or not entries:
        raise GVSPopulationError("scan evidence entries must be a nonempty exact JSON array")
    if len(entries) > MAX_TOTAL_RECORDS:
        raise GVSPopulationError("scan evidence exceeds the maximum record cardinality")
    for index, entry in enumerate(entries):
        _exact_object(entry, label=f"scan_evidence.entries[{index}]", fields=_SCAN_ENTRY_FIELDS)
    for name in (
        "scan_bindings_sha256",
        "record_membership_sha256",
        "record_set_sha256",
        "entries_sha256",
        "scan_evidence_sha256",
    ):
        _sha256(checked[name], label=f"scan_evidence.{name}")
    if checked["scan_evidence_sha256"] != _canonical_sha256(_scan_evidence_hash_payload(checked)):
        raise GVSPopulationError("scan evidence hash mismatch")
    expected = build_scan_evidence(
        populations,
        expected_membership=expected_membership,
        expected_scan_bindings=expected_scan_bindings,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
    )
    if _canonical_bytes(checked) != _canonical_bytes(expected):
        raise GVSPopulationError("scan evidence differs from recomputed commitments")
    _assert_gvs_population_runtime_integrity()
    return checked


def loads_scan_evidence(
    text: str,
    *,
    populations: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
) -> dict[str, Any]:
    """Load and validate one bounded exact scan-evidence artifact."""

    _assert_gvs_population_runtime_integrity()
    payload = _loads_bounded_json(
        text, label="scan_evidence", maximum_bytes=MAX_SCAN_EVIDENCE_JSON_BYTES
    )
    result = validate_scan_evidence(
        payload,
        populations=populations,
        expected_membership=expected_membership,
        expected_scan_bindings=expected_scan_bindings,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
    )
    _assert_gvs_population_runtime_integrity()
    return result


def _firewall_hash_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = _COPY_DEEPCOPY(dict(manifest))
    payload["firewall_sha256"] = "0" * 64
    return payload


def build_population_firewall(
    populations: Mapping[str, list[Mapping[str, Any]]],
    *,
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> dict[str, Any]:
    """Build a deterministic four-role component firewall from metadata commitments.

    Caller-owned containers are detached before authorization-relevant comparisons.
    The resulting manifest contains no outcome metric, sample-size threshold, model
    identity, prompt byte, private label, or access authorization.
    """

    _assert_gvs_population_runtime_integrity()
    population_snapshot = _snapshot_role_arrays(populations, label="populations")
    membership = _validated_membership(expected_membership)
    scan_bindings = _validate_scan_bindings(expected_scan_bindings)
    scan_checked = validate_scan_evidence(
        scan_evidence,
        populations=population_snapshot,
        expected_membership=membership,
        expected_scan_bindings=scan_bindings,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
    )
    scan_frozen = _utc(scan_evidence_frozen_at_utc, label="scan_evidence_frozen_at_utc")
    firewall_frozen = _utc(firewall_frozen_at_utc, label="firewall_frozen_at_utc")
    if firewall_frozen <= scan_frozen:
        raise GVSPopulationError("firewall freeze must strictly postdate the scan-evidence freeze")
    records = _collect_validated_records(
        population_snapshot,
        membership=membership,
        scan_bindings=scan_bindings,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
    )
    for record in records:
        eligibility_decided = _utc(
            record["eligibility"]["decided_at_utc"],
            label="eligibility.decided_at_utc",
        )
        if eligibility_decided >= firewall_frozen:
            raise GVSPopulationError(
                "firewall freeze must strictly postdate every eligibility decision"
            )
    index_by_hash = {
        record["integrity"]["record_sha256"]: index for index, record in enumerate(records)
    }
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    lineage_buckets: dict[tuple[str, str], list[str]] = _DEFAULTDICT_CLASS(list)
    for record in records:
        record_hash = record["integrity"]["record_sha256"]
        for lineage in _lineages(record):
            lineage_buckets[lineage].append(record_hash)

    edges: list[dict[str, str]] = []
    for (kind, value), member_hashes in sorted(lineage_buckets.items()):
        ordered_hashes = sorted(set(member_hashes))
        first = ordered_hashes[0]
        for other in ordered_hashes[1:]:
            union(index_by_hash[first], index_by_hash[other])
            edges.append(
                {
                    "left_record_sha256": first,
                    "lineage_kind": kind,
                    "lineage_value_sha256": _canonical_sha256(value),
                    "right_record_sha256": other,
                }
            )

    grouped: dict[int, list[dict[str, Any]]] = _DEFAULTDICT_CLASS(list)
    for index, record in enumerate(records):
        grouped[find(index)].append(record)

    components: list[dict[str, Any]] = []
    component_by_record_hash: dict[str, str] = {}
    for members in grouped.values():
        member_hashes = sorted(member["integrity"]["record_sha256"] for member in members)
        roles = sorted({member["population_role"] for member in members})
        if len(roles) != 1:
            member_ids = sorted(member["record_id"] for member in members)
            raise GVSPopulationError(
                "transitive component crosses population roles; "
                f"roles={roles!r}, records={member_ids!r}"
            )
        eligibility_values = {member["eligibility"]["eligible"] for member in members}
        if len(eligibility_values) != 1:
            raise GVSPopulationError("a lineage component mixes eligibility decisions")
        for field_name in ("task_class", "expected_outcome", "action_family"):
            values = {member[field_name] for member in members}
            if len(values) != 1:
                raise GVSPopulationError(f"a lineage component mixes {field_name} classifications")
        strata_payloads = {_canonical_sha256(member["strata"]) for member in members}
        if len(strata_payloads) != 1:
            raise GVSPopulationError("a lineage component mixes audited boolean strata")
        representative = members[0]
        component_eligible = next(iter(eligibility_values))
        component_id = _canonical_sha256(member_hashes)
        for record_hash in member_hashes:
            component_by_record_hash[record_hash] = component_id
        components.append(
            {
                "component_id": component_id,
                "member_record_ids": sorted(member["record_id"] for member in members),
                "member_record_sha256s": member_hashes,
                "population_role": roles[0],
                "eligible": component_eligible,
                "classification": {
                    "task_class": representative["task_class"],
                    "expected_outcome": representative["expected_outcome"],
                    "action_family": representative["action_family"],
                    "strata": _COPY_DEEPCOPY(representative["strata"]),
                },
                "exclusion_reason_codes": sorted(
                    {
                        member["eligibility"]["reason_code"]
                        for member in members
                        if member["eligibility"]["reason_code"] is not None
                    }
                ),
            }
        )
    components.sort(key=lambda component: component["component_id"])
    edges.sort(
        key=lambda edge: (
            edge["lineage_kind"],
            edge["lineage_value_sha256"],
            edge["left_record_sha256"],
            edge["right_record_sha256"],
        )
    )
    graph = {"components": components, "lineage_edges": edges}

    strata_assignments: list[dict[str, Any]] = []
    eligibility_assignments: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    population_summaries: dict[str, dict[str, Any]] = {}
    eligible_component_summaries: dict[str, dict[str, Any]] = {}
    for record in records:
        record_hash = record["integrity"]["record_sha256"]
        component_id = component_by_record_hash[record_hash]
        strata_assignments.append(
            {
                "action_family": record["action_family"],
                "component_id": component_id,
                "expected_outcome": record["expected_outcome"],
                "population_role": record["population_role"],
                "record_id": record["record_id"],
                "record_sha256": record_hash,
                "strata": _COPY_DEEPCOPY(record["strata"]),
                "task_class": record["task_class"],
            }
        )
        eligibility = record["eligibility"]
        assignment = {
            "component_id": component_id,
            "decision": eligibility["decision"],
            "decision_stage": eligibility["decision_stage"],
            "eligible": eligibility["eligible"],
            "evidence_sha256": eligibility["evidence_sha256"],
            "population_role": record["population_role"],
            "reason_code": eligibility["reason_code"],
            "record_id": record["record_id"],
            "record_sha256": record_hash,
        }
        eligibility_assignments.append(assignment)
        if not eligibility["eligible"]:
            exclusions.append(_COPY_DEEPCOPY(assignment))

    for role in POPULATION_ROLES:
        role_records = [record for record in records if record["population_role"] == role]
        role_components = [
            component for component in components if component["population_role"] == role
        ]
        eligible_component_ids = sorted(
            component["component_id"] for component in role_components if component["eligible"]
        )
        excluded_component_ids = sorted(
            component["component_id"] for component in role_components if not component["eligible"]
        )
        population_summaries[role] = {
            "eligible_record_sha256s": sorted(
                record["integrity"]["record_sha256"]
                for record in role_records
                if record["eligibility"]["eligible"]
            ),
            "excluded_record_sha256s": sorted(
                record["integrity"]["record_sha256"]
                for record in role_records
                if not record["eligibility"]["eligible"]
            ),
            "record_ids": sorted(record["record_id"] for record in role_records),
            "record_sha256s": sorted(
                record["integrity"]["record_sha256"] for record in role_records
            ),
            "component_ids": sorted(component["component_id"] for component in role_components),
        }
        eligible_component_summaries[role] = {
            "eligible_component_count": len(eligible_component_ids),
            "eligible_component_ids": eligible_component_ids,
            "excluded_component_count": len(excluded_component_ids),
            "excluded_component_ids": excluded_component_ids,
        }

    strata_assignments.sort(key=lambda assignment: assignment["record_sha256"])
    eligibility_assignments.sort(key=lambda assignment: assignment["record_sha256"])
    exclusions.sort(key=lambda assignment: assignment["record_sha256"])
    normalized_membership = {role: membership[role] for role in POPULATION_ROLES}
    manifest: dict[str, Any] = {
        "schema_version": GVS_POPULATION_FIREWALL_SCHEMA_VERSION,
        "structural_only": True,
        "authorizes_model_or_label_access": False,
        "requires_authenticated_custody_receipt": True,
        "compared_release_cutoff_utc": compared_release_cutoff_utc,
        "scan_evidence_frozen_at_utc": scan_evidence_frozen_at_utc,
        "firewall_frozen_at_utc": firewall_frozen_at_utc,
        "scan_bindings": scan_bindings,
        "scan_bindings_sha256": _canonical_sha256(scan_bindings),
        "scan_evidence_sha256": scan_checked["scan_evidence_sha256"],
        "population_membership_sha256": _canonical_sha256(normalized_membership),
        "record_set_sha256": _canonical_sha256(
            sorted(record["integrity"]["record_sha256"] for record in records)
        ),
        "populations": population_summaries,
        "eligible_component_summaries": eligible_component_summaries,
        "graph": graph,
        "graph_sha256": _canonical_sha256(graph),
        "strata_assignments": strata_assignments,
        "strata_sha256": _canonical_sha256(strata_assignments),
        "eligibility_assignments": eligibility_assignments,
        "exclusions": exclusions,
        "eligibility_sha256": _canonical_sha256(eligibility_assignments),
        "firewall_sha256": "0" * 64,
    }
    manifest["firewall_sha256"] = _canonical_sha256(_firewall_hash_payload(manifest))
    if len(_canonical_bytes(manifest)) > MAX_FIREWALL_JSON_BYTES:
        raise GVSPopulationError("population firewall exceeds its bounded JSON size")
    _assert_gvs_population_runtime_integrity()
    return manifest


def validate_population_firewall(
    manifest: object,
    *,
    populations: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> dict[str, Any]:
    """Validate a firewall against freshly recomputed detached source commitments."""

    _assert_gvs_population_runtime_integrity()
    checked = _snapshot_object(
        manifest,
        label="population_firewall",
        fields=_FIREWALL_FIELDS,
        maximum_bytes=MAX_FIREWALL_JSON_BYTES,
    )
    if checked["schema_version"] != GVS_POPULATION_FIREWALL_SCHEMA_VERSION:
        raise GVSPopulationError(
            f"firewall schema must be {GVS_POPULATION_FIREWALL_SCHEMA_VERSION!r}"
        )
    if checked["structural_only"] is not True:
        raise GVSPopulationError("population firewall must be structural_only=true")
    if checked["authorizes_model_or_label_access"] is not False:
        raise GVSPopulationError("population firewall cannot authorize model or label access")
    if checked["requires_authenticated_custody_receipt"] is not True:
        raise GVSPopulationError(
            "population firewall must require an authenticated custody receipt"
        )
    expected_times = {
        "compared_release_cutoff_utc": compared_release_cutoff_utc,
        "scan_evidence_frozen_at_utc": scan_evidence_frozen_at_utc,
        "firewall_frozen_at_utc": firewall_frozen_at_utc,
    }
    for name, expected_time in expected_times.items():
        if checked[name] != expected_time:
            raise GVSPopulationError(f"population firewall is bound to a different {name}")
    for name in (
        "scan_bindings_sha256",
        "scan_evidence_sha256",
        "population_membership_sha256",
        "record_set_sha256",
        "graph_sha256",
        "strata_sha256",
        "eligibility_sha256",
        "firewall_sha256",
    ):
        _sha256(checked[name], label=f"population_firewall.{name}")
    if checked["firewall_sha256"] != _canonical_sha256(_firewall_hash_payload(checked)):
        raise GVSPopulationError("population firewall hash mismatch")
    expected = build_population_firewall(
        populations,
        expected_membership=expected_membership,
        expected_scan_bindings=expected_scan_bindings,
        scan_evidence=scan_evidence,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
        firewall_frozen_at_utc=firewall_frozen_at_utc,
    )
    if _canonical_bytes(checked) != _canonical_bytes(expected):
        raise GVSPopulationError("population firewall differs from recomputed commitments")
    _assert_gvs_population_runtime_integrity()
    return checked


def loads_population_firewall(
    text: str,
    *,
    populations: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> dict[str, Any]:
    """Load and validate one bounded exact structural firewall."""

    _assert_gvs_population_runtime_integrity()
    payload = _loads_bounded_json(
        text, label="population_firewall", maximum_bytes=MAX_FIREWALL_JSON_BYTES
    )
    result = validate_population_firewall(
        payload,
        populations=populations,
        expected_membership=expected_membership,
        expected_scan_bindings=expected_scan_bindings,
        scan_evidence=scan_evidence,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
        firewall_frozen_at_utc=firewall_frozen_at_utc,
    )
    _assert_gvs_population_runtime_integrity()
    return result


def _assert_import_bindings_unchanged() -> None:
    for name, pinned in _CONTRACT_OBJECT_PINS:
        if globals().get(name) is not pinned:
            raise GVSPopulationError(
                f"population contract {name} differs from its import-time identity"
            )
    current_and_pinned = (
        ("copy.deepcopy", copy.deepcopy, _COPY_DEEPCOPY),
        ("hashlib.sha256", hashlib.sha256, _HASHLIB_SHA256),
        ("json.dumps", json.dumps, _JSON_DUMPS),
        ("json.loads", json.loads, _JSON_LOADS),
        ("json.JSONDecodeError", json.JSONDecodeError, _JSON_DECODE_ERROR),
        ("math.isfinite", math.isfinite, _MATH_ISFINITE),
        ("unicodedata.normalize", unicodedata.normalize, _UNICODE_NORMALIZE),
        ("datetime", datetime, _DATETIME_CLASS),
        ("datetime_timezone", datetime_timezone, _DATETIME_TIMEZONE_CLASS),
        ("defaultdict", defaultdict, _DEFAULTDICT_CLASS),
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
            raise GVSPopulationError(
                f"population runtime import {name} differs from its import-time identity"
            )


def _gvs_population_runtime_sha256() -> str:
    """Bind population source, live code, schemas, imports, and lower runtime."""

    _assert_import_bindings_unchanged()
    _ASSERT_SIM_PROGRAM_RUNTIME_INTEGRITY()
    dependencies: dict[str, object] = {}
    for name, value in (
        ("copy.deepcopy", _COPY_DEEPCOPY),
        ("datetime.fromisoformat", _DATETIME_FROMISOFORMAT),
        ("hashlib.sha256", _HASHLIB_SHA256),
        ("json.dumps", _JSON_DUMPS),
        ("json.loads", _JSON_LOADS),
        ("math.isfinite", _MATH_ISFINITE),
        ("sha256_pattern.fullmatch", _SHA256_RE.fullmatch),
        ("identifier_pattern.fullmatch", _IDENTIFIER_RE.fullmatch),
        ("utc_pattern.fullmatch", _UTC_TIMESTAMP_RE.fullmatch),
        ("unicodedata.normalize", _UNICODE_NORMALIZE),
    ):
        identity = _RUNTIME_CALLABLE_IDENTITY(value)
        dependencies[name] = dict(identity)
    contract = {
        "schema_versions": {
            "population_record": GVS_POPULATION_RECORD_SCHEMA_VERSION,
            "scan_evidence": GVS_SCAN_EVIDENCE_SCHEMA_VERSION,
            "population_firewall": GVS_POPULATION_FIREWALL_SCHEMA_VERSION,
        },
        "roles": list(POPULATION_ROLES),
        "task_classes": sorted(TASK_CLASSES),
        "expected_outcomes": sorted(EXPECTED_OUTCOMES),
        "action_families": sorted(ACTION_FAMILIES),
        "audited_boolean_strata": list(AUDITED_BOOLEAN_STRATA),
        "pre_outcome_exclusion_reasons": sorted(PRE_OUTCOME_EXCLUSION_REASONS),
        "fields": {
            "record": sorted(_RECORD_FIELDS),
            "provenance": sorted(_PROVENANCE_FIELDS),
            "scan_bindings": sorted(_BINDING_FIELDS),
            "duplicate_evidence": sorted(_DUPLICATE_EVIDENCE_FIELDS),
            "eligibility": sorted(_ELIGIBILITY_FIELDS),
            "integrity": sorted(_INTEGRITY_FIELDS),
            "firewall": sorted(_FIREWALL_FIELDS),
            "scan_entry": sorted(_SCAN_ENTRY_FIELDS),
            "scan_evidence": sorted(_SCAN_EVIDENCE_FIELDS),
        },
        "limits": {
            "record_json_bytes": MAX_RECORD_JSON_BYTES,
            "population_metadata_json_bytes": MAX_POPULATION_METADATA_JSON_BYTES,
            "scan_evidence_json_bytes": MAX_SCAN_EVIDENCE_JSON_BYTES,
            "firewall_json_bytes": MAX_FIREWALL_JSON_BYTES,
            "records_per_role": MAX_RECORDS_PER_ROLE,
            "total_records": MAX_TOTAL_RECORDS,
            "entity_pool_ids": MAX_ENTITY_POOL_IDS,
            "json_depth": _MAX_JSON_DEPTH,
        },
        "patterns": {
            "sha256": _SHA256_RE.pattern,
            "identifier": _IDENTIFIER_RE.pattern,
            "utc": _UTC_TIMESTAMP_RE.pattern,
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
        raise GVSPopulationError("could not bind the GVS population runtime") from error


def _assert_gvs_population_runtime_integrity() -> None:
    """Fail closed if population source, live code, imports, or constants changed."""

    current = _gvs_population_runtime_sha256()
    if current != _PINNED_GVS_POPULATION_RUNTIME_SHA256:
        raise GVSPopulationError("GVS population runtime differs from its import-time identity")


def gvs_population_runtime_sha256() -> str:
    """Return the independently recomputed population runtime commitment."""

    return _gvs_population_runtime_sha256()


def assert_gvs_population_runtime_integrity() -> None:
    """Public runtime-integrity check; firewall entrypoints use the private guard."""

    _assert_gvs_population_runtime_integrity()


__all__ = [
    "ACTION_FAMILIES",
    "AUDITED_BOOLEAN_STRATA",
    "EXPECTED_OUTCOMES",
    "GVS_POPULATION_FIREWALL_SCHEMA_VERSION",
    "GVS_POPULATION_RECORD_SCHEMA_VERSION",
    "GVS_SCAN_EVIDENCE_SCHEMA_VERSION",
    "MAX_ENTITY_POOL_IDS",
    "MAX_FIREWALL_JSON_BYTES",
    "MAX_POPULATION_METADATA_JSON_BYTES",
    "MAX_RECORDS_PER_ROLE",
    "MAX_RECORD_JSON_BYTES",
    "MAX_SCAN_EVIDENCE_JSON_BYTES",
    "MAX_TOTAL_RECORDS",
    "POPULATION_ROLES",
    "PRE_OUTCOME_EXCLUSION_REASONS",
    "TASK_CLASSES",
    "GVSPopulationError",
    "assert_gvs_population_runtime_integrity",
    "build_population_firewall",
    "build_scan_evidence",
    "canonical_sha256",
    "gvs_population_runtime_sha256",
    "loads_population_firewall",
    "loads_population_record",
    "loads_scan_evidence",
    "seal_population_record",
    "validate_population_firewall",
    "validate_population_record",
    "validate_scan_evidence",
]


_PINNED_GVS_POPULATION_RUNTIME_SHA256 = _gvs_population_runtime_sha256()
