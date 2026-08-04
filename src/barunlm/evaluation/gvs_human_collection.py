"""Offline, nonauthorizing S-new/C-new collection packaging for GVS-v1.

This module builds exact-JSON human-authoring packages without loading a model, reading a dataset,
or contacting a service.  Its runtime attestation reads only this module's own source bytes; no
collection or label file is read or written.  The public author export is generated from a closed
allowlist: it contains task-class quotas, boolean stratum requirements, fixed instructions, and
required attestations, but no assignment/record/author/source identifiers, hashes, answers,
labels, predictions, teacher text, model-generated text, or S-new/C-new membership.

Prompt intake and labeling are deliberately separate stages.  A prompt intake can contain only
the human-authored request/context/tool contract plus declared provenance.  A later label task is
derived from that sealed intake; the packager adds no coordinator identifier, integrity hash,
answer, or model output, while honestly leaving the copied prompt content unauthenticated.  The
final annotation is retained only in a private label envelope bound to the package, prompt intake,
and label-task hashes.  Declared timestamps are checked for structural order, but neither
timestamps nor human identities are authenticated here.

All returned claims remain fail-closed.  The code cannot prove independent human authorship,
release-relative chronology, source or label custody, duplicate closure, power, single-use
disclosure, or durable atomic retirement.  Consequently, no object built here authorizes model,
label, CUDA, JarvisLabs, training, launch, or execution access.  External isolated storage, secret
custody, a single-use signer, a durable atomic compare-and-append service, and a jointly scanned
real T/D/S/C population remain mandatory.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime
from datetime import timezone as datetime_timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from barunaction.schema import ContractError, parse_tool_declarations

from .action_ir import ActionIRError, decode_json_object, validate_action_ir
from .sim_program import SimProgramError, module_runtime_sha256, runtime_callable_identity

GVS_HUMAN_COLLECTION_PACKAGE_VERSION = "barun-gvs-human-collection-package-v1"
GVS_AUTHOR_EXPORT_VERSION = "barun-gvs-human-author-export-v1"
GVS_AUTHOR_TASK_VERSION = "barun-gvs-human-author-task-v1"
GVS_PROMPT_INTAKE_VERSION = "barun-gvs-human-prompt-intake-v1"
GVS_LABEL_TASK_VERSION = "barun-gvs-human-label-task-v1"
GVS_PRIVATE_LABEL_ENVELOPE_VERSION = "barun-gvs-private-label-envelope-v1"

HUMAN_POPULATION_ROLES = ("S-new", "C-new")
TASK_CLASSES = ("efficacy", "safety")
AUDITED_STRATA = (
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
PROVENANCE_REQUIREMENTS = (
    "author_id",
    "source_id",
    "collection_batch",
    "schema_family",
    "program_template_id",
    "paraphrase_family",
    "entity_pool_ids",
    "temporal_construction_id",
    "role_assigned_at_utc",
    "authored_at_utc",
    "license",
    "consent",
    "no_model_assistance",
    "authoring_protocol_revision",
)

ROLE_CLASS_REQUIREMENTS = {
    "S-new": {"efficacy": 1_200, "safety": 800, "total": 2_000},
    "C-new": {"efficacy": 2_500, "safety": 1_500, "total": 4_000},
}

MAX_ASSIGNMENTS = 6_000
MAX_PACKAGE_JSON_BYTES = 128 * 1024 * 1024
MAX_AUTHOR_EXPORT_JSON_BYTES = 16 * 1024 * 1024
MAX_PROMPT_INTAKE_JSON_BYTES = 512 * 1024
MAX_LABEL_TASK_JSON_BYTES = 512 * 1024
MAX_LABEL_ENVELOPE_JSON_BYTES = 512 * 1024
MAX_JSON_DEPTH = 64
MAX_PACKAGE_JSON_NODES = 1_500_000
MAX_RECORD_JSON_NODES = 100_000
MAX_JSON_STRING_BYTES = 256 * 1024
MAX_JSON_INTEGER_ABS = 2**63 - 1
MAX_ENTITY_POOL_IDS = 64

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_OFFSET_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})\Z")

_ASSIGNMENT_FIELDS = frozenset(
    {"assignment_id", "population_role", "task_class", "required_strata"}
)
_PACKAGE_FIELDS = frozenset(
    {
        "schema_version",
        "compared_release_cutoff_utc",
        "assignment_source_sha256",
        "requirements",
        "author_export",
        "coordinator_manifest",
        "integrity",
        "claims",
    }
)
_REQUIREMENTS_FIELDS = frozenset(
    {
        "nominal_role_class_task_counts",
        "audited_strata",
        "required_provenance_fields",
        "independent_human_authorship_required",
        "no_model_assistance_required",
        "post_release_authorship_required",
        "prompt_before_label_required",
        "separate_prompt_and_labeler_required",
        "confirmation_sealed_before_selection_disclosure_required",
        "joint_duplicate_scan_required",
        "effective_cluster_counts_require_joint_lineage_closure",
        "coordinator_bound_schema_presentation_required",
        "schema_to_simulator_population_bridge_required",
        "external_custody_and_retirement_required",
    }
)
_AUTHOR_EXPORT_FIELDS = frozenset(
    {
        "schema_version",
        "nominal_task_class_counts",
        "workflow",
        "task_batches",
        "contains_internal_identifiers",
        "contains_integrity_hashes",
        "contains_answer_or_label",
        "contains_model_generated_text",
        "authorizes_model_access",
        "authorizes_label_access",
        "authorizes_cuda",
        "authorizes_jarvislabs",
        "authorizes_training",
        "authorizes_launch",
        "authorizes_execution",
    }
)
_AUTHOR_BATCH_FIELDS = frozenset({"quantity", "task"})
_AUTHOR_TASK_FIELDS = frozenset(
    {
        "schema_version",
        "task_class",
        "required_strata",
        "task_instructions",
        "submission_fields",
        "coordinator_supplied_fields",
        "required_attestations",
        "prompt_before_label",
        "separate_labeling_stage",
        "contains_internal_identifiers",
        "contains_integrity_hashes",
        "contains_answer_or_label",
        "contains_model_generated_text",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "assignment_id",
        "population_role",
        "task_class",
        "required_strata",
        "task_batch_index",
        "assignment_sha256",
        "author_task_sha256",
    }
)
_PACKAGE_INTEGRITY_FIELDS = frozenset(
    {
        "assignment_roster_sha256",
        "author_export_sha256",
        "coordinator_manifest_sha256",
        "package_sha256",
    }
)
_CLAIM_FIELDS = frozenset(
    {
        "structural_validation_only",
        "human_provenance_authenticated",
        "chronology_authenticated",
        "source_custody_authenticated",
        "private_label_custody_authenticated",
        "single_use_signer_available",
        "durable_atomic_retirement_available",
        "joint_duplicate_closure_complete",
        "effective_cluster_counts_authenticated",
        "semantic_content_authentication_complete",
        "population_record_bridge_complete",
        "power_thresholds_frozen_from_real_roster",
        "authorizes_model_access",
        "authorizes_label_access",
        "authorizes_cuda",
        "authorizes_jarvislabs",
        "authorizes_training",
        "authorizes_launch",
        "authorizes_execution",
    }
)
_PROMPT_FIELDS = frozenset(
    {
        "request",
        "context",
        "reference_timestamp",
        "reference_fold",
        "timezone",
        "tool_schemas",
    }
)
_PROVENANCE_FIELDS = frozenset(PROVENANCE_REQUIREMENTS)
_ATTESTATION_FIELDS = frozenset(
    {
        "independent_human_authorship",
        "no_model_assistance",
        "original_work",
        "not_copied_from_benchmark",
        "prompt_authored_before_label",
    }
)
_PROMPT_INTAKE_FIELDS = frozenset(
    {
        "schema_version",
        "assignment_id",
        "population_role",
        "task_class",
        "required_strata",
        "collection_package_sha256",
        "assignment_sha256",
        "prompt",
        "provenance_declarations",
        "author_attestations",
        "prompt_received_at_utc",
        "integrity",
        "claims",
    }
)
_PROMPT_INTEGRITY_FIELDS = frozenset({"prompt_sha256", "prompt_intake_sha256"})
_LABEL_TASK_FIELDS = frozenset(
    {
        "schema_version",
        "prompt",
        "instructions",
        "submission_fields",
        "required_attestations",
        "created_at_utc",
        "packager_added_private_coordinator_identifiers",
        "packager_added_integrity_hashes",
        "packager_added_answer",
        "packager_added_model_generated_text",
        "prompt_content_authenticated",
        "authorizes_model_access",
        "authorizes_label_access",
        "authorizes_cuda",
        "authorizes_jarvislabs",
        "authorizes_training",
        "authorizes_launch",
        "authorizes_execution",
    }
)
_ANNOTATION_FIELDS = frozenset(
    {
        "labeler_id",
        "labeler_independent_human_attestation",
        "labeler_no_model_assistance",
        "labeler_no_compared_prediction_access",
        "labeled_at_utc",
        "labeler_action_ir",
        "reviewer_id",
        "reviewer_independent_human_attestation",
        "reviewer_no_model_assistance",
        "reviewer_no_compared_prediction_access",
        "reviewed_at_utc",
        "reviewer_action_ir",
        "disagreement",
        "adjudicator_id",
        "adjudicator_independent_human_attestation",
        "adjudicator_no_model_assistance",
        "adjudicator_no_compared_prediction_access",
        "adjudicated_at_utc",
        "adjudicated_action_ir",
    }
)
_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "assignment_id",
        "population_role",
        "collection_package_sha256",
        "prompt_intake_sha256",
        "label_task_sha256",
        "annotation",
        "sealed_at_utc",
        "integrity",
        "claims",
    }
)
_ENVELOPE_INTEGRITY_FIELDS = frozenset({"private_label_envelope_sha256"})

_AUTHOR_WORKFLOW = (
    "Receive one task without any private identifier, hash, answer, or prior model text.",
    "Write one original request and its supplied context after the task is assigned.",
    "Submit the prompt and authorship attestations before any labeling task is created.",
    "Do not produce Action IR, an answer key, or an evaluation judgment during authoring.",
)
_AUTHOR_INSTRUCTIONS = (
    "Write one natural personal-action request that satisfies every required stratum marked true.",
    "Make the request self-contained with only the supplied context and tool declarations.",
    "Do not copy a benchmark item or use AI/model assistance, paraphrasing, or completion tools.",
    "Do not include an answer, Action IR, scoring hint, internal identifier, or integrity hash.",
)
_AUTHOR_SUBMISSION_FIELDS = (
    "request",
    "context",
)
_COORDINATOR_SUPPLIED_FIELDS = (
    "reference_timestamp",
    "reference_fold",
    "timezone",
    "tool_schemas",
)
_AUTHOR_ATTESTATIONS = (
    "independent_human_authorship",
    "no_model_assistance",
    "original_work",
    "not_copied_from_benchmark",
    "prompt_authored_before_label",
)
_LABEL_INSTRUCTIONS = (
    "Read only the supplied human-authored prompt and tool declarations.",
    "Return one canonical Action IR object under the frozen evaluator contract.",
    "Do not use AI/model assistance or inspect predictions from any compared system.",
    "Record uncertainty through CLARIFY, CONFIRM, or ABSTAIN when the contract requires it.",
)

# Stable aliases reduce ambient monkeypatch surface.  They are tamper resistance for an ordinary
# Python worker, not a trusted execution environment or an authorization mechanism.
_HASHLIB_SHA256 = hashlib.sha256
_COUNTER_CLASS = Counter
_JSON_DUMPS = json.dumps
_JSON_LOADS = json.loads
_JSON_DECODE_ERROR = json.JSONDecodeError
_MATH_ISFINITE = math.isfinite
_UNICODE_NORMALIZE = unicodedata.normalize
_DATETIME_CLASS = datetime
_DATETIME_TIMEZONE = datetime_timezone
_DATETIME_FROMISOFORMAT = datetime.fromisoformat
_ZONEINFO_CLASS = ZoneInfo
_ZONEINFO_NOT_FOUND = ZoneInfoNotFoundError
_PARSE_TOOL_DECLARATIONS = parse_tool_declarations
_DECODE_JSON_OBJECT = decode_json_object
_VALIDATE_ACTION_IR = validate_action_ir
_CONTRACT_ERROR = ContractError
_ACTION_IR_ERROR = ActionIRError
_SIM_PROGRAM_ERROR = SimProgramError
_MODULE_RUNTIME_SHA256 = module_runtime_sha256
_RUNTIME_CALLABLE_IDENTITY = runtime_callable_identity


class GVSHumanCollectionError(ValueError):
    """The offline GVS-v1 human-collection contract failed closed."""


def _strict_json(
    value: object,
    *,
    label: str,
    maximum_nodes: int,
    depth: int = 0,
    budget: list[int] | None = None,
) -> None:
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > maximum_nodes:
        raise GVSHumanCollectionError(f"{label} exceeds maximum JSON nodes {maximum_nodes}")
    if depth > MAX_JSON_DEPTH:
        raise GVSHumanCollectionError(f"{label} exceeds maximum JSON depth {MAX_JSON_DEPTH}")
    value_type = type(value)
    if value is None or value_type is bool:
        return
    if value_type is int:
        if abs(value) > MAX_JSON_INTEGER_ABS:
            raise GVSHumanCollectionError(f"{label} integer exceeds the exact bounded range")
        return
    if value_type is float:
        if not _MATH_ISFINITE(value):
            raise GVSHumanCollectionError(f"{label} contains a non-finite number")
        return
    if value_type is str:
        try:
            size = len(value.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as error:
            raise GVSHumanCollectionError(f"{label} contains invalid UTF-8 text") from error
        if size > MAX_JSON_STRING_BYTES:
            raise GVSHumanCollectionError(
                f"{label} string exceeds maximum {MAX_JSON_STRING_BYTES} UTF-8 bytes"
            )
        return
    if value_type is list:
        for index, child in enumerate(value):
            _strict_json(
                child,
                label=f"{label}[{index}]",
                maximum_nodes=maximum_nodes,
                depth=depth + 1,
                budget=budget,
            )
        return
    if value_type is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise GVSHumanCollectionError(f"{label} object keys must be exact strings")
            _strict_json(
                key,
                label=f"{label}.<key>",
                maximum_nodes=maximum_nodes,
                depth=depth + 1,
                budget=budget,
            )
            _strict_json(
                child,
                label=f"{label}.{key}",
                maximum_nodes=maximum_nodes,
                depth=depth + 1,
                budget=budget,
            )
        return
    raise GVSHumanCollectionError(f"{label} contains non-JSON type {value_type.__name__}")


def _canonical_bytes(value: object, *, maximum_nodes: int = MAX_RECORD_JSON_NODES) -> bytes:
    _strict_json(value, label="$", maximum_nodes=maximum_nodes)
    try:
        return _JSON_DUMPS(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise GVSHumanCollectionError("value is not strict canonical JSON") from error


def _canonical_sha256(value: object, *, maximum_nodes: int = MAX_RECORD_JSON_NODES) -> str:
    return _HASHLIB_SHA256(_canonical_bytes(value, maximum_nodes=maximum_nodes)).hexdigest()


def canonical_sha256(value: object, *, maximum_nodes: int = MAX_RECORD_JSON_NODES) -> str:
    """Return SHA-256 of one exact canonical JSON value."""

    _assert_runtime_integrity()
    result = _canonical_sha256(value, maximum_nodes=maximum_nodes)
    _assert_runtime_integrity()
    return result


def canonical_json(value: object, *, maximum_nodes: int = MAX_RECORD_JSON_NODES) -> str:
    """Return one compact, sorted, strict canonical JSON document."""

    _assert_runtime_integrity()
    result = _canonical_bytes(value, maximum_nodes=maximum_nodes).decode("utf-8")
    _assert_runtime_integrity()
    return result


def _snapshot(
    value: object,
    *,
    label: str,
    maximum_bytes: int,
    maximum_nodes: int,
) -> Any:
    raw = _canonical_bytes(value, maximum_nodes=maximum_nodes)
    if len(raw) > maximum_bytes:
        raise GVSHumanCollectionError(f"{label} exceeds maximum {maximum_bytes} JSON bytes")
    try:
        return _JSON_LOADS(raw)
    except (_JSON_DECODE_ERROR, RecursionError) as error:
        raise GVSHumanCollectionError(f"{label} could not be detached") from error


def _exact_object(value: object, *, label: str, fields: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict:
        raise GVSHumanCollectionError(f"{label} must be an exact JSON object")
    actual = set(value)
    if actual != fields:
        raise GVSHumanCollectionError(
            f"{label} fields changed; missing={sorted(fields - actual)!r}, "
            f"extra={sorted(actual - fields)!r}"
        )
    return value


def _text(value: object, *, label: str, identifier: bool = False) -> str:
    if type(value) is not str or not value or not value.strip():
        raise GVSHumanCollectionError(f"{label} must be nonempty text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise GVSHumanCollectionError(f"{label} contains invalid UTF-8 text") from error
    if len(value.encode("utf-8")) > MAX_JSON_STRING_BYTES:
        raise GVSHumanCollectionError(f"{label} exceeds its text bound")
    if identifier and (_IDENTIFIER_RE.fullmatch(value) is None or "@" in value):
        raise GVSHumanCollectionError(f"{label} must be a pseudonymous safe identifier")
    return value


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise GVSHumanCollectionError(f"{label} must be a lowercase SHA-256")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise GVSHumanCollectionError(f"{label} must be an exact boolean")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise GVSHumanCollectionError(f"{label} must be a positive exact integer")
    return value


def _utc(value: object, *, label: str) -> datetime:
    text = _text(value, label=label)
    if _UTC_RE.fullmatch(text) is None:
        raise GVSHumanCollectionError(f"{label} must be a full RFC 3339 UTC timestamp")
    try:
        parsed = _DATETIME_FROMISOFORMAT(text[:-1] + "+00:00")
    except ValueError as error:
        raise GVSHumanCollectionError(f"{label} is not a valid timestamp") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise GVSHumanCollectionError(f"{label} must be UTC")
    return parsed


def _reference_timestamp(value: object, timezone_name: object, fold_value: object) -> None:
    text = _text(value, label="prompt.reference_timestamp")
    zone_name = _text(timezone_name, label="prompt.timezone")
    if _OFFSET_RE.fullmatch(text) is None:
        raise GVSHumanCollectionError(
            "prompt.reference_timestamp must be a full timestamp with an explicit offset"
        )
    if type(fold_value) is not int or fold_value not in {0, 1}:
        raise GVSHumanCollectionError("prompt.reference_fold must be exact integer 0 or 1")
    try:
        parsed = _DATETIME_FROMISOFORMAT(text[:-1] + "+00:00" if text.endswith("Z") else text)
        zone = _ZONEINFO_CLASS(zone_name)
    except (ValueError, _ZONEINFO_NOT_FOUND) as error:
        raise GVSHumanCollectionError("prompt timestamp or IANA timezone is invalid") from error
    if parsed.utcoffset() is None:
        raise GVSHumanCollectionError("prompt.reference_timestamp requires an explicit offset")
    naive = parsed.replace(tzinfo=None)
    zoned = naive.replace(tzinfo=zone, fold=fold_value)
    if zoned.utcoffset() != parsed.utcoffset():
        raise GVSHumanCollectionError("prompt offset/fold does not match prompt.timezone")
    roundtrip = zoned.astimezone(_DATETIME_TIMEZONE.utc).astimezone(zone)
    if roundtrip.replace(tzinfo=None) != naive or roundtrip.fold != fold_value:
        raise GVSHumanCollectionError("prompt timestamp is nonexistent or has the wrong fold")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GVSHumanCollectionError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> float:
    raise GVSHumanCollectionError(f"JSON contains non-finite number {value!r}")


def _parse_int(value: str) -> int:
    parsed = int(value)
    if abs(parsed) > MAX_JSON_INTEGER_ABS:
        raise GVSHumanCollectionError("JSON integer exceeds the exact bounded range")
    return parsed


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not _MATH_ISFINITE(parsed):
        raise GVSHumanCollectionError("JSON contains a non-finite number")
    return parsed


def _loads_bounded(
    text: object,
    *,
    label: str,
    maximum_bytes: int,
    maximum_nodes: int,
) -> Any:
    if type(text) is not str:
        raise GVSHumanCollectionError(f"{label} JSON must be an exact string")
    try:
        raw = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise GVSHumanCollectionError(f"{label} JSON contains invalid UTF-8") from error
    if len(raw) > maximum_bytes:
        raise GVSHumanCollectionError(f"{label} exceeds maximum {maximum_bytes} JSON bytes")
    try:
        value = _JSON_LOADS(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
            parse_int=_parse_int,
            parse_float=_parse_float,
        )
    except GVSHumanCollectionError:
        raise
    except (ValueError, _JSON_DECODE_ERROR, RecursionError) as error:
        raise GVSHumanCollectionError(f"{label} is not bounded strict JSON") from error
    _strict_json(value, label=label, maximum_nodes=maximum_nodes)
    return value


def _claims() -> dict[str, bool]:
    return {
        "structural_validation_only": True,
        "human_provenance_authenticated": False,
        "chronology_authenticated": False,
        "source_custody_authenticated": False,
        "private_label_custody_authenticated": False,
        "single_use_signer_available": False,
        "durable_atomic_retirement_available": False,
        "joint_duplicate_closure_complete": False,
        "effective_cluster_counts_authenticated": False,
        "semantic_content_authentication_complete": False,
        "population_record_bridge_complete": False,
        "power_thresholds_frozen_from_real_roster": False,
        "authorizes_model_access": False,
        "authorizes_label_access": False,
        "authorizes_cuda": False,
        "authorizes_jarvislabs": False,
        "authorizes_training": False,
        "authorizes_launch": False,
        "authorizes_execution": False,
    }


def _validate_claims(value: object, *, label: str) -> dict[str, bool]:
    claims = _exact_object(value, label=label, fields=_CLAIM_FIELDS)
    if claims["structural_validation_only"] is not True:
        raise GVSHumanCollectionError(f"{label}.structural_validation_only must be exact true")
    for name in _CLAIM_FIELDS - {"structural_validation_only"}:
        if claims[name] is not False:
            raise GVSHumanCollectionError(f"{label}.{name} must remain exact false")
    return claims


def _role(value: object, *, label: str = "population_role") -> str:
    if value not in HUMAN_POPULATION_ROLES or type(value) is not str:
        raise GVSHumanCollectionError(f"{label} must be one of {list(HUMAN_POPULATION_ROLES)!r}")
    return value


def _task_class(value: object, *, label: str = "task_class") -> str:
    if value not in TASK_CLASSES or type(value) is not str:
        raise GVSHumanCollectionError(f"{label} must be one of {list(TASK_CLASSES)!r}")
    return value


def _strata(value: object, *, task_class: str, label: str) -> dict[str, bool]:
    strata = _exact_object(value, label=label, fields=frozenset(AUDITED_STRATA))
    checked = {name: _boolean(strata[name], label=f"{label}.{name}") for name in AUDITED_STRATA}
    if checked["unsafe_or_adversarial"] is not (task_class == "safety"):
        raise GVSHumanCollectionError(
            f"{label}.unsafe_or_adversarial must be true exactly for safety tasks"
        )
    return checked


def _validate_assignment(value: object, *, label: str) -> dict[str, Any]:
    assignment = _exact_object(value, label=label, fields=_ASSIGNMENT_FIELDS)
    assignment_id = _text(
        assignment["assignment_id"], label=f"{label}.assignment_id", identifier=True
    )
    role = _role(assignment["population_role"], label=f"{label}.population_role")
    task_class = _task_class(assignment["task_class"], label=f"{label}.task_class")
    strata = _strata(
        assignment["required_strata"], task_class=task_class, label=f"{label}.required_strata"
    )
    return {
        "assignment_id": assignment_id,
        "population_role": role,
        "task_class": task_class,
        "required_strata": strata,
    }


def _requirements() -> dict[str, Any]:
    return {
        "nominal_role_class_task_counts": {
            role: dict(ROLE_CLASS_REQUIREMENTS[role]) for role in HUMAN_POPULATION_ROLES
        },
        "audited_strata": list(AUDITED_STRATA),
        "required_provenance_fields": list(PROVENANCE_REQUIREMENTS),
        "independent_human_authorship_required": True,
        "no_model_assistance_required": True,
        "post_release_authorship_required": True,
        "prompt_before_label_required": True,
        "separate_prompt_and_labeler_required": True,
        "confirmation_sealed_before_selection_disclosure_required": True,
        "joint_duplicate_scan_required": True,
        "effective_cluster_counts_require_joint_lineage_closure": True,
        "coordinator_bound_schema_presentation_required": True,
        "schema_to_simulator_population_bridge_required": True,
        "external_custody_and_retirement_required": True,
    }


def _author_task(task_class: str, strata: dict[str, bool]) -> dict[str, Any]:
    return {
        "schema_version": GVS_AUTHOR_TASK_VERSION,
        "task_class": task_class,
        "required_strata": dict(strata),
        "task_instructions": list(_AUTHOR_INSTRUCTIONS),
        "submission_fields": list(_AUTHOR_SUBMISSION_FIELDS),
        "coordinator_supplied_fields": list(_COORDINATOR_SUPPLIED_FIELDS),
        "required_attestations": list(_AUTHOR_ATTESTATIONS),
        "prompt_before_label": True,
        "separate_labeling_stage": True,
        "contains_internal_identifiers": False,
        "contains_integrity_hashes": False,
        "contains_answer_or_label": False,
        "contains_model_generated_text": False,
    }


def _author_export_and_batches(
    assignments: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[bytes, int]]:
    counts: Counter[bytes] = _COUNTER_CLASS()
    tasks: dict[bytes, dict[str, Any]] = {}
    for assignment in assignments:
        task = _author_task(
            assignment["task_class"],
            assignment["required_strata"],
        )
        key = _canonical_bytes(task)
        counts[key] += 1
        tasks[key] = task
    ordered_keys = sorted(tasks)
    batch_indices = {key: index for index, key in enumerate(ordered_keys)}
    author_export = {
        "schema_version": GVS_AUTHOR_EXPORT_VERSION,
        "nominal_task_class_counts": {
            task_class: sum(
                ROLE_CLASS_REQUIREMENTS[role][task_class] for role in HUMAN_POPULATION_ROLES
            )
            for task_class in TASK_CLASSES
        },
        "workflow": list(_AUTHOR_WORKFLOW),
        "task_batches": [{"quantity": counts[key], "task": tasks[key]} for key in ordered_keys],
        "contains_internal_identifiers": False,
        "contains_integrity_hashes": False,
        "contains_answer_or_label": False,
        "contains_model_generated_text": False,
        "authorizes_model_access": False,
        "authorizes_label_access": False,
        "authorizes_cuda": False,
        "authorizes_jarvislabs": False,
        "authorizes_training": False,
        "authorizes_launch": False,
        "authorizes_execution": False,
    }
    return author_export, batch_indices


def _validate_author_task(value: object, *, label: str) -> dict[str, Any]:
    task = _exact_object(value, label=label, fields=_AUTHOR_TASK_FIELDS)
    if task["schema_version"] != GVS_AUTHOR_TASK_VERSION:
        raise GVSHumanCollectionError(f"{label}.schema_version changed")
    task_class = _task_class(task["task_class"], label=f"{label}.task_class")
    strata = _strata(
        task["required_strata"], task_class=task_class, label=f"{label}.required_strata"
    )
    expected = _author_task(task_class, strata)
    if task != expected:
        raise GVSHumanCollectionError(f"{label} differs from the closed author-task template")
    return task


def _validate_author_export_snapshot(value: object) -> dict[str, Any]:
    export = _exact_object(value, label="author_export", fields=_AUTHOR_EXPORT_FIELDS)
    if export["schema_version"] != GVS_AUTHOR_EXPORT_VERSION:
        raise GVSHumanCollectionError("author_export.schema_version changed")
    if export["nominal_task_class_counts"] != {
        task_class: sum(
            ROLE_CLASS_REQUIREMENTS[role][task_class] for role in HUMAN_POPULATION_ROLES
        )
        for task_class in TASK_CLASSES
    }:
        raise GVSHumanCollectionError("author_export nominal task-class quotas changed")
    if export["workflow"] != list(_AUTHOR_WORKFLOW):
        raise GVSHumanCollectionError("author_export workflow changed")
    for flag in (
        "contains_internal_identifiers",
        "contains_integrity_hashes",
        "contains_answer_or_label",
        "contains_model_generated_text",
        "authorizes_model_access",
        "authorizes_label_access",
        "authorizes_cuda",
        "authorizes_jarvislabs",
        "authorizes_training",
        "authorizes_launch",
        "authorizes_execution",
    ):
        if export[flag] is not False:
            raise GVSHumanCollectionError(f"author_export.{flag} must remain false")
    batches = export["task_batches"]
    maximum_batches = len(HUMAN_POPULATION_ROLES) * len(TASK_CLASSES) * 2 ** len(AUDITED_STRATA)
    if type(batches) is not list or not batches or len(batches) > maximum_batches:
        raise GVSHumanCollectionError("author_export.task_batches is empty or exceeds its bound")
    totals = _COUNTER_CLASS()
    prior: bytes | None = None
    for index, raw_batch in enumerate(batches):
        batch = _exact_object(
            raw_batch, label=f"author_export.task_batches[{index}]", fields=_AUTHOR_BATCH_FIELDS
        )
        quantity = _positive_int(
            batch["quantity"], label=f"author_export.task_batches[{index}].quantity"
        )
        task = _validate_author_task(
            batch["task"], label=f"author_export.task_batches[{index}].task"
        )
        key = _canonical_bytes(task)
        if prior is not None and key <= prior:
            raise GVSHumanCollectionError("author_export task batches must be unique and sorted")
        prior = key
        totals[task["task_class"]] += quantity
        if any(_SHA256_RE.fullmatch(text) for text in _walk_text(task)):
            raise GVSHumanCollectionError("author task leaked an integrity hash")
    expected_totals = {
        task_class: sum(
            ROLE_CLASS_REQUIREMENTS[role][task_class] for role in HUMAN_POPULATION_ROLES
        )
        for task_class in TASK_CLASSES
    }
    if dict(totals) != expected_totals:
        raise GVSHumanCollectionError("author_export task batches do not meet exact quotas")
    return export


def _walk_text(value: object):
    if type(value) is str:
        yield value
    elif type(value) is list:
        for child in value:
            yield from _walk_text(child)
    elif type(value) is dict:
        for key, child in value.items():
            yield key
            yield from _walk_text(child)


def _package_hash_payload(package: dict[str, Any]) -> dict[str, Any]:
    payload = _snapshot(
        package,
        label="package_hash_payload",
        maximum_bytes=MAX_PACKAGE_JSON_BYTES,
        maximum_nodes=MAX_PACKAGE_JSON_NODES,
    )
    payload["integrity"]["package_sha256"] = "0" * 64
    return payload


def _build_package_from_assignments(
    assignments: list[dict[str, Any]],
    *,
    compared_release_cutoff_utc: str,
    assignment_source_sha256: str,
) -> dict[str, Any]:
    author_export, batch_indices = _author_export_and_batches(assignments)
    manifest: list[dict[str, Any]] = []
    for assignment in assignments:
        task = _author_task(
            assignment["task_class"],
            assignment["required_strata"],
        )
        task_bytes = _canonical_bytes(task)
        manifest.append(
            {
                **assignment,
                "task_batch_index": batch_indices[task_bytes],
                "assignment_sha256": _canonical_sha256(assignment),
                "author_task_sha256": _HASHLIB_SHA256(task_bytes).hexdigest(),
            }
        )
    integrity = {
        "assignment_roster_sha256": _canonical_sha256(
            assignments, maximum_nodes=MAX_PACKAGE_JSON_NODES
        ),
        "author_export_sha256": _canonical_sha256(
            author_export, maximum_nodes=MAX_PACKAGE_JSON_NODES
        ),
        "coordinator_manifest_sha256": _canonical_sha256(
            manifest, maximum_nodes=MAX_PACKAGE_JSON_NODES
        ),
        "package_sha256": "0" * 64,
    }
    package = {
        "schema_version": GVS_HUMAN_COLLECTION_PACKAGE_VERSION,
        "compared_release_cutoff_utc": compared_release_cutoff_utc,
        "assignment_source_sha256": assignment_source_sha256,
        "requirements": _requirements(),
        "author_export": author_export,
        "coordinator_manifest": manifest,
        "integrity": integrity,
        "claims": _claims(),
    }
    integrity["package_sha256"] = _canonical_sha256(
        _package_hash_payload(package), maximum_nodes=MAX_PACKAGE_JSON_NODES
    )
    return package


def build_human_collection_package(
    assignments: object,
    *,
    compared_release_cutoff_utc: str,
    assignment_source_sha256: str,
) -> dict[str, Any]:
    """Build the exact proposed 2,000 S-new plus 4,000 C-new offline package.

    ``assignments`` is coordinator-private.  It supplies only membership, task class, and boolean
    pre-outcome strata; no authored prompt or label is accepted at this stage.  The returned
    ``author_export`` is the only subtree intended for author dispatch.
    """

    _assert_runtime_integrity()
    cutoff_text = _text(compared_release_cutoff_utc, label="compared_release_cutoff_utc")
    _utc(cutoff_text, label="compared_release_cutoff_utc")
    source_hash = _sha256(assignment_source_sha256, label="assignment_source_sha256")
    snapshot = _snapshot(
        assignments,
        label="assignments",
        maximum_bytes=MAX_PACKAGE_JSON_BYTES,
        maximum_nodes=MAX_PACKAGE_JSON_NODES,
    )
    if type(snapshot) is not list or len(snapshot) != MAX_ASSIGNMENTS:
        raise GVSHumanCollectionError(
            f"assignments must be an exact array of {MAX_ASSIGNMENTS} proposed human tasks"
        )
    checked = [
        _validate_assignment(value, label=f"assignments[{index}]")
        for index, value in enumerate(snapshot)
    ]
    ids = [assignment["assignment_id"] for assignment in checked]
    if len(set(ids)) != len(ids):
        raise GVSHumanCollectionError("assignment IDs must be globally unique")
    checked.sort(key=lambda assignment: assignment["assignment_id"])
    counts = _COUNTER_CLASS(
        (assignment["population_role"], assignment["task_class"]) for assignment in checked
    )
    expected = _COUNTER_CLASS(
        {
            (role, task_class): ROLE_CLASS_REQUIREMENTS[role][task_class]
            for role in HUMAN_POPULATION_ROLES
            for task_class in TASK_CLASSES
        }
    )
    if counts != expected:
        raise GVSHumanCollectionError(
            f"assignment role/class quotas changed: expected={dict(expected)!r}, got={dict(counts)!r}"
        )
    package = _build_package_from_assignments(
        checked,
        compared_release_cutoff_utc=cutoff_text,
        assignment_source_sha256=source_hash,
    )
    result = validate_human_collection_package(package)
    _assert_runtime_integrity()
    return result


def validate_human_collection_package(value: object) -> dict[str, Any]:
    """Validate and detach one complete nonauthorizing collection package."""

    _assert_runtime_integrity()
    package = _snapshot(
        value,
        label="human_collection_package",
        maximum_bytes=MAX_PACKAGE_JSON_BYTES,
        maximum_nodes=MAX_PACKAGE_JSON_NODES,
    )
    package = _exact_object(package, label="human_collection_package", fields=_PACKAGE_FIELDS)
    if package["schema_version"] != GVS_HUMAN_COLLECTION_PACKAGE_VERSION:
        raise GVSHumanCollectionError("human_collection_package.schema_version changed")
    cutoff = _text(package["compared_release_cutoff_utc"], label="compared_release_cutoff_utc")
    _utc(cutoff, label="compared_release_cutoff_utc")
    source_hash = _sha256(package["assignment_source_sha256"], label="assignment_source_sha256")
    requirements = _exact_object(
        package["requirements"], label="requirements", fields=_REQUIREMENTS_FIELDS
    )
    if requirements != _requirements():
        raise GVSHumanCollectionError("collection requirements differ from the frozen proposal")
    author_export = _validate_author_export_snapshot(package["author_export"])
    manifest = package["coordinator_manifest"]
    if type(manifest) is not list or len(manifest) != MAX_ASSIGNMENTS:
        raise GVSHumanCollectionError(
            f"coordinator_manifest must contain exactly {MAX_ASSIGNMENTS} assignments"
        )
    assignments: list[dict[str, Any]] = []
    prior_id: str | None = None
    seen_hashes: set[str] = set()
    for index, raw_entry in enumerate(manifest):
        entry = _exact_object(
            raw_entry, label=f"coordinator_manifest[{index}]", fields=_MANIFEST_FIELDS
        )
        assignment = _validate_assignment(
            {name: entry[name] for name in _ASSIGNMENT_FIELDS},
            label=f"coordinator_manifest[{index}].assignment",
        )
        assignment_id = assignment["assignment_id"]
        if prior_id is not None and assignment_id <= prior_id:
            raise GVSHumanCollectionError("coordinator_manifest IDs must be unique and sorted")
        prior_id = assignment_id
        task_index = entry["task_batch_index"]
        if type(task_index) is not int or task_index < 0:
            raise GVSHumanCollectionError("task_batch_index must be a nonnegative exact integer")
        batches = author_export["task_batches"]
        if task_index >= len(batches):
            raise GVSHumanCollectionError("task_batch_index is outside the author export")
        expected_task = _author_task(assignment["task_class"], assignment["required_strata"])
        if batches[task_index]["task"] != expected_task:
            raise GVSHumanCollectionError("manifest assignment points to a different author task")
        expected_assignment_hash = _canonical_sha256(assignment)
        if (
            _sha256(entry["assignment_sha256"], label="assignment_sha256")
            != expected_assignment_hash
        ):
            raise GVSHumanCollectionError("manifest assignment hash mismatch")
        if expected_assignment_hash in seen_hashes:
            raise GVSHumanCollectionError("manifest assignment hashes must be unique")
        seen_hashes.add(expected_assignment_hash)
        expected_task_hash = _canonical_sha256(expected_task)
        if _sha256(entry["author_task_sha256"], label="author_task_sha256") != expected_task_hash:
            raise GVSHumanCollectionError("manifest author-task hash mismatch")
        assignments.append(assignment)
    private_assignment_ids = {assignment["assignment_id"] for assignment in assignments}
    leaked_ids = private_assignment_ids.intersection(_walk_text(author_export))
    if leaked_ids:
        raise GVSHumanCollectionError(
            "author_export text collides with private assignment identifiers"
        )
    expected_package = _build_package_from_assignments(
        assignments,
        compared_release_cutoff_utc=cutoff,
        assignment_source_sha256=source_hash,
    )
    if package != expected_package:
        raise GVSHumanCollectionError("human collection package failed exact reconstruction")
    integrity = _exact_object(
        package["integrity"], label="integrity", fields=_PACKAGE_INTEGRITY_FIELDS
    )
    for name in _PACKAGE_INTEGRITY_FIELDS:
        _sha256(integrity[name], label=f"integrity.{name}")
    _validate_claims(package["claims"], label="claims")
    _assert_runtime_integrity()
    return package


def loads_human_collection_package(text: object) -> dict[str, Any]:
    """Load one bounded package while rejecting duplicate keys and non-finite numbers."""

    _assert_runtime_integrity()
    value = _loads_bounded(
        text,
        label="human_collection_package",
        maximum_bytes=MAX_PACKAGE_JSON_BYTES,
        maximum_nodes=MAX_PACKAGE_JSON_NODES,
    )
    result = validate_human_collection_package(value)
    _assert_runtime_integrity()
    return result


def validate_author_export(value: object) -> dict[str, Any]:
    """Validate and detach the identifier/hash/answer-free author-facing subtree."""

    _assert_runtime_integrity()
    export = _snapshot(
        value,
        label="author_export",
        maximum_bytes=MAX_AUTHOR_EXPORT_JSON_BYTES,
        maximum_nodes=MAX_PACKAGE_JSON_NODES,
    )
    result = _validate_author_export_snapshot(export)
    _assert_runtime_integrity()
    return result


def loads_author_export(text: object) -> dict[str, Any]:
    """Load a bounded author-facing export."""

    _assert_runtime_integrity()
    value = _loads_bounded(
        text,
        label="author_export",
        maximum_bytes=MAX_AUTHOR_EXPORT_JSON_BYTES,
        maximum_nodes=MAX_PACKAGE_JSON_NODES,
    )
    result = validate_author_export(value)
    _assert_runtime_integrity()
    return result


def export_author_tasks(package: object) -> dict[str, Any]:
    """Return only the safe author-facing export from a valid private package."""

    _assert_runtime_integrity()
    checked = validate_human_collection_package(package)
    export = validate_author_export(checked["author_export"])
    private_ids = {entry["assignment_id"] for entry in checked["coordinator_manifest"]}
    if private_ids.intersection(_walk_text(export)):
        raise GVSHumanCollectionError("author export leaked a private assignment identifier")
    _assert_runtime_integrity()
    return export


def _manifest_entry(package: dict[str, Any], assignment_id: str) -> dict[str, Any]:
    # The manifest is sorted, but a bounded linear search keeps this simple and deterministic.
    matches = [
        entry
        for entry in package["coordinator_manifest"]
        if entry["assignment_id"] == assignment_id
    ]
    if len(matches) != 1:
        raise GVSHumanCollectionError("assignment_id is not exactly one package assignment")
    return matches[0]


def _validate_prompt(value: object, *, label: str = "prompt") -> dict[str, Any]:
    prompt = _exact_object(value, label=label, fields=_PROMPT_FIELDS)
    request = _text(prompt["request"], label=f"{label}.request")
    if request != request.strip():
        raise GVSHumanCollectionError(f"{label}.request cannot have outer whitespace")
    context = prompt["context"]
    if type(context) is not dict:
        raise GVSHumanCollectionError(f"{label}.context must be an exact JSON object")
    _strict_json(context, label=f"{label}.context", maximum_nodes=MAX_RECORD_JSON_NODES)
    _reference_timestamp(
        prompt["reference_timestamp"], prompt["timezone"], prompt["reference_fold"]
    )
    tools = prompt["tool_schemas"]
    if type(tools) is not list or not tools:
        raise GVSHumanCollectionError(f"{label}.tool_schemas must be a nonempty exact array")
    try:
        _PARSE_TOOL_DECLARATIONS(tools)
    except (_CONTRACT_ERROR, TypeError, ValueError) as error:
        raise GVSHumanCollectionError(f"{label}.tool_schemas is invalid: {error}") from error
    return prompt


def _validate_provenance(value: object, *, cutoff: datetime, received: datetime) -> dict[str, Any]:
    provenance = _exact_object(value, label="provenance_declarations", fields=_PROVENANCE_FIELDS)
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
        _text(provenance[name], label=f"provenance_declarations.{name}", identifier=True)
    entity_ids = provenance["entity_pool_ids"]
    if type(entity_ids) is not list or not entity_ids or len(entity_ids) > MAX_ENTITY_POOL_IDS:
        raise GVSHumanCollectionError("provenance entity_pool_ids is empty or exceeds its bound")
    checked_ids = [
        _text(value, label="provenance_declarations.entity_pool_ids[]", identifier=True)
        for value in entity_ids
    ]
    if checked_ids != sorted(checked_ids) or len(checked_ids) != len(set(checked_ids)):
        raise GVSHumanCollectionError("provenance entity_pool_ids must be unique and sorted")
    assigned = _utc(provenance["role_assigned_at_utc"], label="role_assigned_at_utc")
    authored = _utc(provenance["authored_at_utc"], label="authored_at_utc")
    if not cutoff < assigned < authored <= received:
        raise GVSHumanCollectionError(
            "declared chronology must satisfy cutoff < assignment < authoring <= receipt"
        )
    license_text = _text(provenance["license"], label="provenance_declarations.license")
    if (
        len(license_text) > 512
        or license_text != _UNICODE_NORMALIZE("NFC", license_text)
        or license_text != license_text.strip()
        or any(character in license_text for character in "\r\n\x00")
    ):
        raise GVSHumanCollectionError(
            "provenance license must be NFC, single-line, trimmed, and at most 512 characters"
        )
    if provenance["consent"] is not True or provenance["no_model_assistance"] is not True:
        raise GVSHumanCollectionError(
            "provenance consent and no_model_assistance declarations must be true"
        )
    return provenance


def _validate_attestations(value: object, *, label: str) -> dict[str, bool]:
    attestations = _exact_object(value, label=label, fields=_ATTESTATION_FIELDS)
    for name in _ATTESTATION_FIELDS:
        if attestations[name] is not True:
            raise GVSHumanCollectionError(f"{label}.{name} must be exact true")
    return attestations


def _prompt_intake_hash_payload(intake: dict[str, Any]) -> dict[str, Any]:
    payload = _snapshot(
        intake,
        label="prompt_intake_hash_payload",
        maximum_bytes=MAX_PROMPT_INTAKE_JSON_BYTES,
        maximum_nodes=MAX_RECORD_JSON_NODES,
    )
    payload["integrity"]["prompt_intake_sha256"] = "0" * 64
    return payload


def build_prompt_intake(
    package: object,
    *,
    assignment_id: str,
    prompt: object,
    provenance_declarations: object,
    author_attestations: object,
    prompt_received_at_utc: str,
) -> dict[str, Any]:
    """Bind one human-authored prompt to a private assignment without accepting a label."""

    _assert_runtime_integrity()
    checked_package = validate_human_collection_package(package)
    checked_id = _text(assignment_id, label="assignment_id", identifier=True)
    entry = _manifest_entry(checked_package, checked_id)
    checked_prompt = _snapshot(
        prompt,
        label="prompt",
        maximum_bytes=MAX_PROMPT_INTAKE_JSON_BYTES,
        maximum_nodes=MAX_RECORD_JSON_NODES,
    )
    checked_prompt = _validate_prompt(checked_prompt)
    received_text = _text(prompt_received_at_utc, label="prompt_received_at_utc")
    received = _utc(received_text, label="prompt_received_at_utc")
    cutoff = _utc(
        checked_package["compared_release_cutoff_utc"], label="compared_release_cutoff_utc"
    )
    provenance = _snapshot(
        provenance_declarations,
        label="provenance_declarations",
        maximum_bytes=MAX_PROMPT_INTAKE_JSON_BYTES,
        maximum_nodes=MAX_RECORD_JSON_NODES,
    )
    provenance = _validate_provenance(provenance, cutoff=cutoff, received=received)
    attestations = _snapshot(
        author_attestations,
        label="author_attestations",
        maximum_bytes=MAX_PROMPT_INTAKE_JSON_BYTES,
        maximum_nodes=MAX_RECORD_JSON_NODES,
    )
    attestations = _validate_attestations(attestations, label="author_attestations")
    if provenance["author_id"] in {checked_id, provenance["source_id"]}:
        raise GVSHumanCollectionError("author/source/assignment identifiers must be distinct")
    intake = {
        "schema_version": GVS_PROMPT_INTAKE_VERSION,
        "assignment_id": checked_id,
        "population_role": entry["population_role"],
        "task_class": entry["task_class"],
        "required_strata": entry["required_strata"],
        "collection_package_sha256": checked_package["integrity"]["package_sha256"],
        "assignment_sha256": entry["assignment_sha256"],
        "prompt": checked_prompt,
        "provenance_declarations": provenance,
        "author_attestations": attestations,
        "prompt_received_at_utc": received_text,
        "integrity": {
            "prompt_sha256": _canonical_sha256(checked_prompt),
            "prompt_intake_sha256": "0" * 64,
        },
        "claims": _claims(),
    }
    intake["integrity"]["prompt_intake_sha256"] = _canonical_sha256(
        _prompt_intake_hash_payload(intake)
    )
    result = validate_prompt_intake(intake, package=checked_package)
    _assert_runtime_integrity()
    return result


def validate_prompt_intake(value: object, *, package: object) -> dict[str, Any]:
    """Validate one prompt-only intake against its complete coordinator package."""

    _assert_runtime_integrity()
    checked_package = validate_human_collection_package(package)
    intake = _snapshot(
        value,
        label="prompt_intake",
        maximum_bytes=MAX_PROMPT_INTAKE_JSON_BYTES,
        maximum_nodes=MAX_RECORD_JSON_NODES,
    )
    intake = _exact_object(intake, label="prompt_intake", fields=_PROMPT_INTAKE_FIELDS)
    if intake["schema_version"] != GVS_PROMPT_INTAKE_VERSION:
        raise GVSHumanCollectionError("prompt_intake.schema_version changed")
    assignment_id = _text(intake["assignment_id"], label="assignment_id", identifier=True)
    entry = _manifest_entry(checked_package, assignment_id)
    for name in ("population_role", "task_class", "required_strata"):
        if intake[name] != entry[name]:
            raise GVSHumanCollectionError(f"prompt intake assignment binding changed: {name}")
    if (
        _sha256(intake["collection_package_sha256"], label="collection_package_sha256")
        != checked_package["integrity"]["package_sha256"]
    ):
        raise GVSHumanCollectionError("prompt intake package hash mismatch")
    if (
        _sha256(intake["assignment_sha256"], label="assignment_sha256")
        != entry["assignment_sha256"]
    ):
        raise GVSHumanCollectionError("prompt intake assignment hash mismatch")
    task_class = _task_class(intake["task_class"])
    _role(intake["population_role"])
    _strata(intake["required_strata"], task_class=task_class, label="required_strata")
    prompt = _validate_prompt(intake["prompt"])
    received = _utc(intake["prompt_received_at_utc"], label="prompt_received_at_utc")
    cutoff = _utc(
        checked_package["compared_release_cutoff_utc"], label="compared_release_cutoff_utc"
    )
    provenance = _validate_provenance(
        intake["provenance_declarations"], cutoff=cutoff, received=received
    )
    _validate_attestations(intake["author_attestations"], label="author_attestations")
    if provenance["author_id"] in {assignment_id, provenance["source_id"]}:
        raise GVSHumanCollectionError("author/source/assignment identifiers must be distinct")
    integrity = _exact_object(
        intake["integrity"], label="prompt_intake.integrity", fields=_PROMPT_INTEGRITY_FIELDS
    )
    if _sha256(integrity["prompt_sha256"], label="prompt_sha256") != _canonical_sha256(prompt):
        raise GVSHumanCollectionError("prompt intake prompt hash mismatch")
    expected_intake_hash = _canonical_sha256(_prompt_intake_hash_payload(intake))
    if (
        _sha256(integrity["prompt_intake_sha256"], label="prompt_intake_sha256")
        != expected_intake_hash
    ):
        raise GVSHumanCollectionError("prompt intake hash mismatch")
    _validate_claims(intake["claims"], label="prompt_intake.claims")
    _assert_runtime_integrity()
    return intake


def loads_prompt_intake(text: object, *, package: object) -> dict[str, Any]:
    """Load one bounded prompt intake."""

    _assert_runtime_integrity()
    value = _loads_bounded(
        text,
        label="prompt_intake",
        maximum_bytes=MAX_PROMPT_INTAKE_JSON_BYTES,
        maximum_nodes=MAX_RECORD_JSON_NODES,
    )
    result = validate_prompt_intake(value, package=package)
    _assert_runtime_integrity()
    return result


def _label_task(intake: dict[str, Any], *, created_at_utc: str) -> dict[str, Any]:
    return {
        "schema_version": GVS_LABEL_TASK_VERSION,
        "prompt": intake["prompt"],
        "instructions": list(_LABEL_INSTRUCTIONS),
        "submission_fields": ["action_ir", "no_model_assistance_attestation"],
        "required_attestations": [
            "independent_human_label",
            "no_model_assistance",
            "no_compared_prediction_access",
        ],
        "created_at_utc": created_at_utc,
        "packager_added_private_coordinator_identifiers": False,
        "packager_added_integrity_hashes": False,
        "packager_added_answer": False,
        "packager_added_model_generated_text": False,
        "prompt_content_authenticated": False,
        "authorizes_model_access": False,
        "authorizes_label_access": False,
        "authorizes_cuda": False,
        "authorizes_jarvislabs": False,
        "authorizes_training": False,
        "authorizes_launch": False,
        "authorizes_execution": False,
    }


def build_label_task(
    prompt_intake: object, *, package: object, created_at_utc: str
) -> dict[str, Any]:
    """Create a labeler-visible task only after its prompt intake exists."""

    _assert_runtime_integrity()
    intake = validate_prompt_intake(prompt_intake, package=package)
    created_text = _text(created_at_utc, label="created_at_utc")
    created = _utc(created_text, label="created_at_utc")
    received = _utc(intake["prompt_received_at_utc"], label="prompt_received_at_utc")
    if created <= received:
        raise GVSHumanCollectionError("label task must be created strictly after prompt receipt")
    result = validate_label_task(
        _label_task(intake, created_at_utc=created_text),
        prompt_intake=intake,
        package=package,
    )
    _assert_runtime_integrity()
    return result


def validate_label_task(value: object, *, prompt_intake: object, package: object) -> dict[str, Any]:
    """Validate a label task by exact reconstruction from its prompt intake."""

    _assert_runtime_integrity()
    intake = validate_prompt_intake(prompt_intake, package=package)
    task = _snapshot(
        value,
        label="label_task",
        maximum_bytes=MAX_LABEL_TASK_JSON_BYTES,
        maximum_nodes=MAX_RECORD_JSON_NODES,
    )
    task = _exact_object(task, label="label_task", fields=_LABEL_TASK_FIELDS)
    if task["schema_version"] != GVS_LABEL_TASK_VERSION:
        raise GVSHumanCollectionError("label_task.schema_version changed")
    created_text = _text(task["created_at_utc"], label="label_task.created_at_utc")
    created = _utc(created_text, label="label_task.created_at_utc")
    received = _utc(intake["prompt_received_at_utc"], label="prompt_received_at_utc")
    if created <= received:
        raise GVSHumanCollectionError("label task must postdate prompt receipt")
    expected = _label_task(intake, created_at_utc=created_text)
    if task != expected:
        raise GVSHumanCollectionError("label task differs from its prompt-derived template")
    if any(
        _SHA256_RE.fullmatch(text)
        for text in _walk_text({k: v for k, v in task.items() if k != "prompt"})
    ):
        raise GVSHumanCollectionError("label task metadata leaked an integrity hash")
    _assert_runtime_integrity()
    return task


def loads_label_task(text: object, *, prompt_intake: object, package: object) -> dict[str, Any]:
    """Load one bounded labeler-visible task."""

    _assert_runtime_integrity()
    value = _loads_bounded(
        text,
        label="label_task",
        maximum_bytes=MAX_LABEL_TASK_JSON_BYTES,
        maximum_nodes=MAX_RECORD_JSON_NODES,
    )
    result = validate_label_task(value, prompt_intake=prompt_intake, package=package)
    _assert_runtime_integrity()
    return result


def _canonical_action_ir(
    value: object, *, tool_schemas: list[dict[str, Any]], label: str
) -> dict[str, Any]:
    snapshot = _snapshot(
        value,
        label=label,
        maximum_bytes=MAX_LABEL_ENVELOPE_JSON_BYTES,
        maximum_nodes=MAX_RECORD_JSON_NODES,
    )
    try:
        decoded = _DECODE_JSON_OBJECT(canonical_json(snapshot))
        declarations = _PARSE_TOOL_DECLARATIONS(tool_schemas)
        action = _VALIDATE_ACTION_IR(decoded, tuple(item.schema for item in declarations))
    except (_ACTION_IR_ERROR, _CONTRACT_ERROR, TypeError, ValueError) as error:
        raise GVSHumanCollectionError(
            f"{label} is not canonical schema-valid Action IR: {error}"
        ) from error
    return action.to_dict()


def _validate_annotation(
    value: object,
    *,
    intake: dict[str, Any],
    label_task: dict[str, Any],
    sealed_at: datetime,
) -> dict[str, Any]:
    annotation = _exact_object(value, label="annotation", fields=_ANNOTATION_FIELDS)
    author_id = intake["provenance_declarations"]["author_id"]
    labeler_id = _text(annotation["labeler_id"], label="annotation.labeler_id", identifier=True)
    reviewer_id = _text(annotation["reviewer_id"], label="annotation.reviewer_id", identifier=True)
    if len({author_id, labeler_id, reviewer_id}) != 3:
        raise GVSHumanCollectionError("author, labeler, and reviewer identifiers must be distinct")
    for name in (
        "labeler_independent_human_attestation",
        "labeler_no_model_assistance",
        "labeler_no_compared_prediction_access",
        "reviewer_independent_human_attestation",
        "reviewer_no_model_assistance",
        "reviewer_no_compared_prediction_access",
    ):
        if annotation[name] is not True:
            raise GVSHumanCollectionError(f"annotation.{name} must be exact true")
    created = _utc(label_task["created_at_utc"], label="label_task.created_at_utc")
    labeled = _utc(annotation["labeled_at_utc"], label="annotation.labeled_at_utc")
    reviewed = _utc(annotation["reviewed_at_utc"], label="annotation.reviewed_at_utc")
    if not created < labeled < reviewed <= sealed_at:
        raise GVSHumanCollectionError(
            "declared label chronology must satisfy task < label < review <= seal"
        )
    tools = intake["prompt"]["tool_schemas"]
    labeler_action = _canonical_action_ir(
        annotation["labeler_action_ir"], tool_schemas=tools, label="annotation.labeler_action_ir"
    )
    reviewer_action = _canonical_action_ir(
        annotation["reviewer_action_ir"], tool_schemas=tools, label="annotation.reviewer_action_ir"
    )
    if annotation["labeler_action_ir"] != labeler_action:
        raise GVSHumanCollectionError("labeler Action IR must already use canonical exact JSON")
    if annotation["reviewer_action_ir"] != reviewer_action:
        raise GVSHumanCollectionError("reviewer Action IR must already use canonical exact JSON")
    disagreement = _boolean(annotation["disagreement"], label="annotation.disagreement")
    actual_disagreement = labeler_action != reviewer_action
    if disagreement is not actual_disagreement:
        raise GVSHumanCollectionError("annotation.disagreement does not match the two labels")
    adjudicator_fields = (
        "adjudicator_id",
        "adjudicator_independent_human_attestation",
        "adjudicator_no_model_assistance",
        "adjudicator_no_compared_prediction_access",
        "adjudicated_at_utc",
        "adjudicated_action_ir",
    )
    if not disagreement:
        if any(annotation[name] is not None for name in adjudicator_fields):
            raise GVSHumanCollectionError("agreement requires null adjudicator fields")
    else:
        adjudicator_id = _text(
            annotation["adjudicator_id"], label="annotation.adjudicator_id", identifier=True
        )
        if adjudicator_id in {author_id, labeler_id, reviewer_id}:
            raise GVSHumanCollectionError("adjudicator must be independent by declared identifier")
        for name in (
            "adjudicator_independent_human_attestation",
            "adjudicator_no_model_assistance",
            "adjudicator_no_compared_prediction_access",
        ):
            if annotation[name] is not True:
                raise GVSHumanCollectionError(f"annotation.{name} must be exact true")
        adjudicated_at = _utc(
            annotation["adjudicated_at_utc"], label="annotation.adjudicated_at_utc"
        )
        if not reviewed < adjudicated_at <= sealed_at:
            raise GVSHumanCollectionError("adjudication must postdate review and not postdate seal")
        canonical_adjudication = _canonical_action_ir(
            annotation["adjudicated_action_ir"],
            tool_schemas=tools,
            label="annotation.adjudicated_action_ir",
        )
        if annotation["adjudicated_action_ir"] != canonical_adjudication:
            raise GVSHumanCollectionError("adjudicated Action IR must be canonical exact JSON")
    return annotation


def _envelope_hash_payload(envelope: dict[str, Any]) -> dict[str, Any]:
    payload = _snapshot(
        envelope,
        label="label_envelope_hash_payload",
        maximum_bytes=MAX_LABEL_ENVELOPE_JSON_BYTES,
        maximum_nodes=MAX_RECORD_JSON_NODES,
    )
    payload["integrity"]["private_label_envelope_sha256"] = "0" * 64
    return payload


def build_private_label_envelope(
    prompt_intake: object,
    label_task: object,
    annotation: object,
    *,
    package: object,
    sealed_at_utc: str,
) -> dict[str, Any]:
    """Build a separate hash-bound label envelope; this does not provide secrecy or custody."""

    _assert_runtime_integrity()
    checked_package = validate_human_collection_package(package)
    intake = validate_prompt_intake(prompt_intake, package=checked_package)
    task = validate_label_task(label_task, prompt_intake=intake, package=checked_package)
    sealed_text = _text(sealed_at_utc, label="sealed_at_utc")
    sealed = _utc(sealed_text, label="sealed_at_utc")
    annotation_snapshot = _snapshot(
        annotation,
        label="annotation",
        maximum_bytes=MAX_LABEL_ENVELOPE_JSON_BYTES,
        maximum_nodes=MAX_RECORD_JSON_NODES,
    )
    checked_annotation = _validate_annotation(
        annotation_snapshot, intake=intake, label_task=task, sealed_at=sealed
    )
    envelope = {
        "schema_version": GVS_PRIVATE_LABEL_ENVELOPE_VERSION,
        "assignment_id": intake["assignment_id"],
        "population_role": intake["population_role"],
        "collection_package_sha256": checked_package["integrity"]["package_sha256"],
        "prompt_intake_sha256": intake["integrity"]["prompt_intake_sha256"],
        "label_task_sha256": _canonical_sha256(task),
        "annotation": checked_annotation,
        "sealed_at_utc": sealed_text,
        "integrity": {"private_label_envelope_sha256": "0" * 64},
        "claims": _claims(),
    }
    envelope["integrity"]["private_label_envelope_sha256"] = _canonical_sha256(
        _envelope_hash_payload(envelope)
    )
    result = validate_private_label_envelope(
        envelope,
        package=checked_package,
        prompt_intake=intake,
        label_task=task,
    )
    _assert_runtime_integrity()
    return result


def validate_private_label_envelope(
    value: object,
    *,
    package: object,
    prompt_intake: object,
    label_task: object,
) -> dict[str, Any]:
    """Validate one private envelope against its public prompt lineage."""

    _assert_runtime_integrity()
    checked_package = validate_human_collection_package(package)
    intake = validate_prompt_intake(prompt_intake, package=checked_package)
    task = validate_label_task(label_task, prompt_intake=intake, package=checked_package)
    envelope = _snapshot(
        value,
        label="private_label_envelope",
        maximum_bytes=MAX_LABEL_ENVELOPE_JSON_BYTES,
        maximum_nodes=MAX_RECORD_JSON_NODES,
    )
    envelope = _exact_object(envelope, label="private_label_envelope", fields=_ENVELOPE_FIELDS)
    if envelope["schema_version"] != GVS_PRIVATE_LABEL_ENVELOPE_VERSION:
        raise GVSHumanCollectionError("private label envelope schema version changed")
    expected_bindings = {
        "assignment_id": intake["assignment_id"],
        "population_role": intake["population_role"],
        "collection_package_sha256": checked_package["integrity"]["package_sha256"],
        "prompt_intake_sha256": intake["integrity"]["prompt_intake_sha256"],
        "label_task_sha256": _canonical_sha256(task),
    }
    for name, expected in expected_bindings.items():
        if envelope[name] != expected:
            raise GVSHumanCollectionError(f"private label envelope binding changed: {name}")
    for name in (
        "collection_package_sha256",
        "prompt_intake_sha256",
        "label_task_sha256",
    ):
        _sha256(envelope[name], label=name)
    sealed = _utc(envelope["sealed_at_utc"], label="sealed_at_utc")
    _validate_annotation(envelope["annotation"], intake=intake, label_task=task, sealed_at=sealed)
    integrity = _exact_object(
        envelope["integrity"],
        label="private_label_envelope.integrity",
        fields=_ENVELOPE_INTEGRITY_FIELDS,
    )
    expected_hash = _canonical_sha256(_envelope_hash_payload(envelope))
    if (
        _sha256(
            integrity["private_label_envelope_sha256"],
            label="private_label_envelope_sha256",
        )
        != expected_hash
    ):
        raise GVSHumanCollectionError("private label envelope hash mismatch")
    _validate_claims(envelope["claims"], label="private_label_envelope.claims")
    _assert_runtime_integrity()
    return envelope


def loads_private_label_envelope(
    text: object,
    *,
    package: object,
    prompt_intake: object,
    label_task: object,
) -> dict[str, Any]:
    """Load one bounded private label envelope."""

    _assert_runtime_integrity()
    value = _loads_bounded(
        text,
        label="private_label_envelope",
        maximum_bytes=MAX_LABEL_ENVELOPE_JSON_BYTES,
        maximum_nodes=MAX_RECORD_JSON_NODES,
    )
    result = validate_private_label_envelope(
        value,
        package=package,
        prompt_intake=prompt_intake,
        label_task=label_task,
    )
    _assert_runtime_integrity()
    return result


def _runtime_constant_payload() -> dict[str, Any]:
    field_sets = {
        name: sorted(globals()[name])
        for name in (
            "_ASSIGNMENT_FIELDS",
            "_PACKAGE_FIELDS",
            "_REQUIREMENTS_FIELDS",
            "_AUTHOR_EXPORT_FIELDS",
            "_AUTHOR_BATCH_FIELDS",
            "_AUTHOR_TASK_FIELDS",
            "_MANIFEST_FIELDS",
            "_PACKAGE_INTEGRITY_FIELDS",
            "_CLAIM_FIELDS",
            "_PROMPT_FIELDS",
            "_PROVENANCE_FIELDS",
            "_ATTESTATION_FIELDS",
            "_PROMPT_INTAKE_FIELDS",
            "_PROMPT_INTEGRITY_FIELDS",
            "_LABEL_TASK_FIELDS",
            "_ANNOTATION_FIELDS",
            "_ENVELOPE_FIELDS",
            "_ENVELOPE_INTEGRITY_FIELDS",
        )
    }
    return {
        "versions": {
            "package": GVS_HUMAN_COLLECTION_PACKAGE_VERSION,
            "author_export": GVS_AUTHOR_EXPORT_VERSION,
            "author_task": GVS_AUTHOR_TASK_VERSION,
            "prompt_intake": GVS_PROMPT_INTAKE_VERSION,
            "label_task": GVS_LABEL_TASK_VERSION,
            "label_envelope": GVS_PRIVATE_LABEL_ENVELOPE_VERSION,
        },
        "roles": list(HUMAN_POPULATION_ROLES),
        "task_classes": list(TASK_CLASSES),
        "audited_strata": list(AUDITED_STRATA),
        "provenance_requirements": list(PROVENANCE_REQUIREMENTS),
        "role_class_requirements": ROLE_CLASS_REQUIREMENTS,
        "limits": {
            "assignments": MAX_ASSIGNMENTS,
            "package_bytes": MAX_PACKAGE_JSON_BYTES,
            "author_export_bytes": MAX_AUTHOR_EXPORT_JSON_BYTES,
            "prompt_intake_bytes": MAX_PROMPT_INTAKE_JSON_BYTES,
            "label_task_bytes": MAX_LABEL_TASK_JSON_BYTES,
            "label_envelope_bytes": MAX_LABEL_ENVELOPE_JSON_BYTES,
            "json_depth": MAX_JSON_DEPTH,
            "package_nodes": MAX_PACKAGE_JSON_NODES,
            "record_nodes": MAX_RECORD_JSON_NODES,
            "string_bytes": MAX_JSON_STRING_BYTES,
            "integer_abs": MAX_JSON_INTEGER_ABS,
            "entity_pool_ids": MAX_ENTITY_POOL_IDS,
        },
        "patterns": {
            "sha256": _SHA256_RE.pattern,
            "identifier": _IDENTIFIER_RE.pattern,
            "utc": _UTC_RE.pattern,
            "offset": _OFFSET_RE.pattern,
        },
        "field_sets": field_sets,
        "templates": {
            "author_workflow": list(_AUTHOR_WORKFLOW),
            "author_instructions": list(_AUTHOR_INSTRUCTIONS),
            "author_submission_fields": list(_AUTHOR_SUBMISSION_FIELDS),
            "coordinator_supplied_fields": list(_COORDINATOR_SUPPLIED_FIELDS),
            "author_attestations": list(_AUTHOR_ATTESTATIONS),
            "label_instructions": list(_LABEL_INSTRUCTIONS),
        },
        "reconstructed_claims": _claims(),
        "reconstructed_requirements": _requirements(),
    }


def _runtime_constant_sha256() -> str:
    payload = _runtime_constant_payload()
    encoded = _JSON_DUMPS(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _HASHLIB_SHA256(encoded).hexdigest()


def _assert_runtime_integrity() -> None:
    for name, expected_object, expected_code in _PINNED_RUNTIME_CALLABLES:
        current = globals().get(name)
        if (
            current is not expected_object
            or getattr(current, "__code__", None) is not expected_code
        ):
            raise GVSHumanCollectionError(f"runtime callable binding changed: {name}")
    for name, expected_object in _PINNED_RUNTIME_EXTERNALS:
        if globals().get(name) is not expected_object:
            raise GVSHumanCollectionError(f"runtime dependency binding changed: {name}")
    for name, expected_object in _PINNED_RUNTIME_CONTRACT_OBJECTS:
        if globals().get(name) is not expected_object:
            raise GVSHumanCollectionError(f"runtime contract object binding changed: {name}")
    if _runtime_constant_sha256() != _PINNED_RUNTIME_CONSTANT_SHA256:
        raise GVSHumanCollectionError("runtime collection contract constants changed")
    if _runtime_sha256() != _PINNED_RUNTIME_SHA256:
        raise GVSHumanCollectionError("runtime source, code, dependency, or contract changed")


def _runtime_sha256() -> str:
    dependencies: dict[str, object] = {}
    for name, value in (
        ("hashlib.sha256", _HASHLIB_SHA256),
        ("collections.Counter", _COUNTER_CLASS),
        ("json.dumps", _JSON_DUMPS),
        ("json.loads", _JSON_LOADS),
        ("math.isfinite", _MATH_ISFINITE),
        ("unicodedata.normalize", _UNICODE_NORMALIZE),
        ("datetime.fromisoformat", _DATETIME_FROMISOFORMAT),
        ("zoneinfo.ZoneInfo", _ZONEINFO_CLASS),
        ("parse_tool_declarations", _PARSE_TOOL_DECLARATIONS),
        ("decode_json_object", _DECODE_JSON_OBJECT),
        ("validate_action_ir", _VALIDATE_ACTION_IR),
        ("module_runtime_sha256", _MODULE_RUNTIME_SHA256),
        ("runtime_callable_identity", _RUNTIME_CALLABLE_IDENTITY),
    ):
        try:
            dependencies[name] = dict(_RUNTIME_CALLABLE_IDENTITY(value))
        except _SIM_PROGRAM_ERROR as error:
            raise GVSHumanCollectionError(f"could not bind runtime dependency {name}") from error
    contract = _runtime_constant_payload()
    contract["dependencies"] = dependencies
    try:
        return _MODULE_RUNTIME_SHA256(
            globals(),
            module_name=__name__,
            source_path=__file__,
            contract=contract,
        )
    except _SIM_PROGRAM_ERROR as error:
        raise GVSHumanCollectionError("could not bind the human collection runtime") from error


def gvs_human_collection_runtime_sha256() -> str:
    """Return deterministic tamper evidence for this CPU-only collection contract."""

    _assert_runtime_integrity()
    result = _runtime_sha256()
    _assert_runtime_integrity()
    return result


_RUNTIME_FUNCTION_TYPE = type(_strict_json)
_PINNED_RUNTIME_CALLABLES = tuple(
    (name, value, value.__code__)
    for name, value in sorted(globals().items())
    if type(value) is _RUNTIME_FUNCTION_TYPE
)
_PINNED_RUNTIME_EXTERNALS = (
    ("_HASHLIB_SHA256", _HASHLIB_SHA256),
    ("_COUNTER_CLASS", _COUNTER_CLASS),
    ("_JSON_DUMPS", _JSON_DUMPS),
    ("_JSON_LOADS", _JSON_LOADS),
    ("_JSON_DECODE_ERROR", _JSON_DECODE_ERROR),
    ("_MATH_ISFINITE", _MATH_ISFINITE),
    ("_UNICODE_NORMALIZE", _UNICODE_NORMALIZE),
    ("_DATETIME_CLASS", _DATETIME_CLASS),
    ("_DATETIME_TIMEZONE", _DATETIME_TIMEZONE),
    ("_DATETIME_FROMISOFORMAT", _DATETIME_FROMISOFORMAT),
    ("_ZONEINFO_CLASS", _ZONEINFO_CLASS),
    ("_ZONEINFO_NOT_FOUND", _ZONEINFO_NOT_FOUND),
    ("_PARSE_TOOL_DECLARATIONS", _PARSE_TOOL_DECLARATIONS),
    ("_DECODE_JSON_OBJECT", _DECODE_JSON_OBJECT),
    ("_VALIDATE_ACTION_IR", _VALIDATE_ACTION_IR),
    ("_CONTRACT_ERROR", _CONTRACT_ERROR),
    ("_ACTION_IR_ERROR", _ACTION_IR_ERROR),
    ("_SIM_PROGRAM_ERROR", _SIM_PROGRAM_ERROR),
    ("_MODULE_RUNTIME_SHA256", _MODULE_RUNTIME_SHA256),
    ("_RUNTIME_CALLABLE_IDENTITY", _RUNTIME_CALLABLE_IDENTITY),
)
_PINNED_RUNTIME_CONTRACT_OBJECTS = tuple(
    (name, globals()[name])
    for name in (
        "GVS_HUMAN_COLLECTION_PACKAGE_VERSION",
        "GVS_AUTHOR_EXPORT_VERSION",
        "GVS_AUTHOR_TASK_VERSION",
        "GVS_PROMPT_INTAKE_VERSION",
        "GVS_LABEL_TASK_VERSION",
        "GVS_PRIVATE_LABEL_ENVELOPE_VERSION",
        "HUMAN_POPULATION_ROLES",
        "TASK_CLASSES",
        "AUDITED_STRATA",
        "PROVENANCE_REQUIREMENTS",
        "ROLE_CLASS_REQUIREMENTS",
        "_SHA256_RE",
        "_IDENTIFIER_RE",
        "_UTC_RE",
        "_OFFSET_RE",
        "_ASSIGNMENT_FIELDS",
        "_PACKAGE_FIELDS",
        "_REQUIREMENTS_FIELDS",
        "_AUTHOR_EXPORT_FIELDS",
        "_AUTHOR_BATCH_FIELDS",
        "_AUTHOR_TASK_FIELDS",
        "_MANIFEST_FIELDS",
        "_PACKAGE_INTEGRITY_FIELDS",
        "_CLAIM_FIELDS",
        "_PROMPT_FIELDS",
        "_PROVENANCE_FIELDS",
        "_ATTESTATION_FIELDS",
        "_PROMPT_INTAKE_FIELDS",
        "_PROMPT_INTEGRITY_FIELDS",
        "_LABEL_TASK_FIELDS",
        "_ANNOTATION_FIELDS",
        "_ENVELOPE_FIELDS",
        "_ENVELOPE_INTEGRITY_FIELDS",
        "_AUTHOR_WORKFLOW",
        "_AUTHOR_INSTRUCTIONS",
        "_AUTHOR_SUBMISSION_FIELDS",
        "_COORDINATOR_SUPPLIED_FIELDS",
        "_AUTHOR_ATTESTATIONS",
        "_LABEL_INSTRUCTIONS",
    )
)
_PINNED_RUNTIME_CONSTANT_SHA256 = _runtime_constant_sha256()
_PINNED_RUNTIME_SHA256 = _runtime_sha256()


__all__ = [
    "AUDITED_STRATA",
    "GVS_AUTHOR_EXPORT_VERSION",
    "GVS_AUTHOR_TASK_VERSION",
    "GVS_HUMAN_COLLECTION_PACKAGE_VERSION",
    "GVS_LABEL_TASK_VERSION",
    "GVS_PRIVATE_LABEL_ENVELOPE_VERSION",
    "GVS_PROMPT_INTAKE_VERSION",
    "HUMAN_POPULATION_ROLES",
    "MAX_ASSIGNMENTS",
    "MAX_AUTHOR_EXPORT_JSON_BYTES",
    "MAX_LABEL_ENVELOPE_JSON_BYTES",
    "MAX_LABEL_TASK_JSON_BYTES",
    "MAX_PACKAGE_JSON_BYTES",
    "MAX_PROMPT_INTAKE_JSON_BYTES",
    "PROVENANCE_REQUIREMENTS",
    "ROLE_CLASS_REQUIREMENTS",
    "TASK_CLASSES",
    "GVSHumanCollectionError",
    "build_human_collection_package",
    "build_label_task",
    "build_private_label_envelope",
    "build_prompt_intake",
    "canonical_json",
    "canonical_sha256",
    "export_author_tasks",
    "gvs_human_collection_runtime_sha256",
    "loads_author_export",
    "loads_human_collection_package",
    "loads_label_task",
    "loads_private_label_envelope",
    "loads_prompt_intake",
    "validate_author_export",
    "validate_human_collection_package",
    "validate_label_task",
    "validate_private_label_envelope",
    "validate_prompt_intake",
]
