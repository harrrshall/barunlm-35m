"""Fail-closed protocol primitives for human evaluation of BarunAction-35M.

The public prompt-record schema has no dedicated annotation or gold-label field.  Labels are held
in a separately authenticated private envelope that must remain opaque to model and training
processes, and confirmation-label disclosure is authorized only by an HMAC-authenticated selection
receipt and an HMAC-authenticated append-only access ledger.  A required pinned renderer recomputes
``input.model_input`` from the record source fields, which exclude the private annotation envelope.
It cannot prove that arbitrary human-authored request/context/tool content does not semantically or
structurally encode a label; the external collection audit must establish that precondition before
these primitives are used.

This module deliberately performs no filesystem, network, encryption, or key-management I/O.
Real secrecy requires an external storage boundary that never exposes private envelopes to model
or training processes.  Real once-only access requires the caller's ``durable_append`` callback
to perform an atomic compare-and-append against a durable global ledger.  These functions validate
the protocol objects and fail closed; they do not claim that in-memory Python provides those
external guarantees.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timezone as datetime_timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from barunaction.schema import SCHEMA_CONTRACT_VERSION, ContractError, parse_tool_declarations

from .action_ir import ActionIRError, decode_json_object, validate_action_ir

HUMAN_COLLECTION_SCHEMA_VERSION = "barun-human-prompt-record-v2"
LABEL_ENVELOPE_SCHEMA_VERSION = "barun-human-label-envelope-v2"
SELECTION_RECEIPT_SCHEMA_VERSION = "barun-human-selection-receipt-v2"
LABEL_ACCESS_EVENT_SCHEMA_VERSION = "barun-human-label-access-v2"
TEXT_NORMALIZER_VERSION = "barun-human-text-normalizer-v1"

FROZEN_SELECTION_GATES = frozenset(
    {
        "overall_vs_a_ge_3",
        "overall_vs_b_ge_3",
        "argument_value_gain_ge_5",
        "positive_two_of_three_seeds",
        "family_policy_loss_le_2",
        "planir_parse_ge_995",
        "action_ir_schema_100",
        "zero_truncation_missing_generation_catastrophic",
    }
)

_ROLES = frozenset({"selection", "confirmation"})
_TASK_CLASSES = frozenset({"efficacy", "safety"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_UTC_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_OFFSET_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})\Z"
)
_MAX_JSON_DEPTH = 64

_PROMPT_FIELDS = frozenset(
    {
        "schema_version",
        "record_id",
        "cluster_id",
        "role",
        "task_class",
        "provenance",
        "input",
        "eligibility",
        "duplicate_evidence",
        "tokenization",
        "integrity",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "author_id",
        "source_id",
        "collection_batch",
        "role_assigned_at_utc",
        "authored_at_utc",
        "license",
        "consent",
        "no_model_assistance",
        "authoring_protocol_revision",
        "schema_family",
        "paraphrase_family",
        "entity_source_ids",
        "temporal_construction_id",
    }
)
_INPUT_FIELDS = frozenset(
    {
        "request",
        "context",
        "reference_timestamp",
        "reference_fold",
        "timezone",
        "tool_schemas",
        "tool_schema_identity",
        "model_input",
        "model_input_renderer_identity",
    }
)
_TOOL_IDENTITY_FIELDS = frozenset({"identifier", "revision", "contract_version", "sha256"})
_RENDERER_IDENTITY_FIELDS = frozenset({"identifier", "revision", "sha256"})
_ELIGIBILITY_FIELDS = frozenset({"eligible", "decision", "reason", "decider_id", "decided_at_utc"})
_DUPLICATE_EVIDENCE_FIELDS = frozenset(
    {
        "input_sha256",
        "normalized_request_sha256",
        "delexicalized_template_sha256",
        "near_duplicate_cluster_id",
        "normalizer_revision",
        "exact_scan_manifest_sha256",
        "near_scan_manifest_sha256",
        "scan_code_sha256",
        "scan_completed_at_utc",
    }
)
_PROMPT_TOKENIZATION_FIELDS = frozenset(
    {"tokenizer_id", "tokenizer_revision", "tokenizer_sha256", "input_tokens"}
)
_PROMPT_INTEGRITY_FIELDS = frozenset({"input_sha256", "prompt_record_sha256"})

_LABEL_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "record_id",
        "prompt_record_sha256",
        "annotation",
        "tokenization",
        "sealed_at_utc",
        "key_id",
        "envelope_sha256",
        "envelope_hmac_sha256",
    }
)
_ANNOTATION_FIELDS = frozenset(
    {
        "labeler_id",
        "labeler_no_model_assistance",
        "labeler_labeled_at_utc",
        "labeler_action_ir",
        "reviewer_id",
        "reviewer_no_model_assistance",
        "reviewer_labeled_at_utc",
        "reviewer_action_ir",
        "disagreement",
        "adjudicator_id",
        "adjudicator_no_model_assistance",
        "adjudicated_at_utc",
        "adjudicated_action_ir",
    }
)
_LABEL_TOKENIZATION_FIELDS = frozenset(
    {"tokenizer_id", "tokenizer_revision", "tokenizer_sha256", "output_tokens"}
)

_RECEIPT_BINDING_FIELDS = frozenset(
    {
        "selection_collection_sha256",
        "confirmation_prompt_collection_sha256",
        "experiment_config_sha256",
        "candidate_set_sha256",
        "compiler_sha256",
        "evaluator_sha256",
        "source_revision",
        "code_revision",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        *_RECEIPT_BINDING_FIELDS,
        "gate_results",
        "passed",
        "created_at_utc",
        "key_id",
        "receipt_hmac_sha256",
    }
)

_ACCESS_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "event_id",
        "record_id",
        "scoring_session_id",
        "accessor_id",
        "purpose",
        "accessed_at_utc",
        "selection_receipt_sha256",
        "prompt_record_sha256",
        "label_envelope_sha256",
        "previous_event_sha256",
        "key_id",
        "event_sha256",
        "event_hmac_sha256",
    }
)


class HumanCollectionError(ValueError):
    """Raised when evidence violates the frozen human-evaluation protocol."""


@dataclass(frozen=True, slots=True)
class PinnedTokenCounter:
    """An externally verified tokenizer identity plus its deterministic count function.

    The caller must construct this object only after verifying the tokenizer artifact against
    ``tokenizer_sha256``.  The module recomputes every stored count through ``counter``; it cannot
    itself prove that a malicious caller supplied the correct tokenizer implementation.
    """

    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_sha256: str
    counter: Callable[[str], int] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class PinnedModelInputRenderer:
    """A verified deterministic renderer for the exact model-visible input string.

    ``renderer`` receives a deep copy of the prompt input fields excluding ``model_input`` and
    ``model_input_renderer_identity``.  The caller must verify its implementation against
    ``renderer_sha256``.  The renderer proves reproducibility from source fields outside the
    private envelope; a separate collection audit must still establish that request/context/tool
    content is not contaminated with evaluation labels.
    """

    renderer_id: str
    renderer_revision: str
    renderer_sha256: str
    renderer: Callable[[Mapping[str, Any]], str] = field(repr=False, compare=False)


def _strict_json(value: object, *, path: str = "$", depth: int = 0) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise HumanCollectionError(f"{path} exceeds maximum JSON depth {_MAX_JSON_DEPTH}")
    value_type = type(value)
    if value is None or value_type in {bool, int}:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise HumanCollectionError(f"{path} must not contain a non-finite number")
        return
    if value_type is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise HumanCollectionError(f"{path} contains invalid UTF-8 text") from error
        return
    if value_type is list:
        for index, child in enumerate(value):
            _strict_json(child, path=f"{path}[{index}]", depth=depth + 1)
        return
    if value_type is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise HumanCollectionError(f"{path} object keys must be exact strings")
            _strict_json(key, path=f"{path}.<key>", depth=depth + 1)
            _strict_json(child, path=f"{path}.{key}", depth=depth + 1)
        return
    raise HumanCollectionError(f"{path} contains non-JSON type {value_type.__name__}")


def _canonical_bytes(value: object) -> bytes:
    _strict_json(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise HumanCollectionError(f"value is not strict JSON: {error}") from error


def canonical_sha256(value: object) -> str:
    """Hash strict canonical JSON after recursively rejecting non-string object keys."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _snapshot_json_object(value: object, field_name: str, fields: frozenset[str]) -> dict[str, Any]:
    """Take one canonical, caller-detached snapshot of an exact JSON object.

    Protocol validation must never validate a caller-owned container and then reuse that mutable
    container for authorization.  Canonical serialization followed by decoding gives the rest of
    the function an owned tree made only from exact JSON builtins.  Concurrent mutation while the
    snapshot is being encoded either contributes to that one snapshot or fails validation; later
    caller mutation cannot change the checked value.
    """

    if type(value) is not dict:
        raise HumanCollectionError(f"{field_name} must be a strict JSON object")
    try:
        snapshot = json.loads(_canonical_bytes(value))
    except (json.JSONDecodeError, RuntimeError) as error:
        raise HumanCollectionError(f"{field_name} could not be snapshotted") from error
    return _object(snapshot, field_name, fields)


def _snapshot_json_sequence(value: object, field_name: str) -> tuple[Any, ...]:
    """Take one caller-detached snapshot of an exact list or tuple.

    Custom ``Sequence`` implementations are intentionally rejected: membership, iteration, and
    indexing must all observe the same values in an authorization decision.
    """

    if type(value) not in {list, tuple}:
        raise HumanCollectionError(f"{field_name} must be an exact list or tuple")
    source = list(value)
    try:
        snapshot = json.loads(_canonical_bytes(source))
    except (json.JSONDecodeError, RuntimeError) as error:
        raise HumanCollectionError(f"{field_name} could not be snapshotted") from error
    return tuple(snapshot)


def _object(value: object, field_name: str, fields: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict:
        raise HumanCollectionError(f"{field_name} must be a strict JSON object")
    _strict_json(value, path=field_name)
    actual = set(value)
    missing = sorted(fields - actual)
    unknown = sorted(actual - fields)
    if missing or unknown:
        raise HumanCollectionError(
            f"{field_name} fields must be exact; missing={missing}, unknown={unknown}"
        )
    return value


def _text(value: object, field_name: str, *, identifier: bool = False) -> str:
    if type(value) is not str or not value:
        raise HumanCollectionError(f"{field_name} must be a non-empty string")
    _strict_json(value, path=field_name)
    if identifier and (_ID_RE.fullmatch(value) is None or "@" in value):
        raise HumanCollectionError(f"{field_name} must be a pseudonymous safe identifier")
    return value


def _sha256(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise HumanCollectionError(f"{field_name} must be a lowercase SHA-256")
    return value


def _utc(value: object, field_name: str) -> datetime:
    text = _text(value, field_name)
    if _UTC_TIMESTAMP_RE.fullmatch(text) is None:
        raise HumanCollectionError(
            f"{field_name} must be a full RFC 3339 UTC timestamp ending in Z"
        )
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise HumanCollectionError(f"{field_name} is not a valid timestamp") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise HumanCollectionError(f"{field_name} must be UTC")
    return parsed


def _nonnegative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise HumanCollectionError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: object, field_name: str) -> int:
    checked = _nonnegative_int(value, field_name)
    if checked == 0:
        raise HumanCollectionError(f"{field_name} must be positive")
    return checked


def _hmac_key(value: object, field_name: str) -> bytes:
    if type(value) is not bytes or len(value) < 32:
        raise HumanCollectionError(f"{field_name} must be at least 32 secret bytes")
    return value


def _hmac_sha256(key: bytes, value: object) -> str:
    return hmac.new(key, _canonical_bytes(value), hashlib.sha256).hexdigest()


def _validate_token_counter(tokenizer: PinnedTokenCounter) -> None:
    if not isinstance(tokenizer, PinnedTokenCounter):
        raise HumanCollectionError("tokenizer must be a PinnedTokenCounter")
    _text(tokenizer.tokenizer_id, "tokenizer.tokenizer_id")
    _text(tokenizer.tokenizer_revision, "tokenizer.tokenizer_revision")
    _sha256(tokenizer.tokenizer_sha256, "tokenizer.tokenizer_sha256")
    if not callable(tokenizer.counter):
        raise HumanCollectionError("tokenizer.counter must be callable")


def _count_tokens(tokenizer: PinnedTokenCounter, text: str, field_name: str) -> int:
    _validate_token_counter(tokenizer)
    try:
        count = tokenizer.counter(text)
    except Exception as error:
        raise HumanCollectionError(f"{field_name} token counter failed") from error
    return _nonnegative_int(count, field_name)


def _validate_token_identity(
    tokenization: Mapping[str, Any], tokenizer: PinnedTokenCounter
) -> None:
    if tokenization["tokenizer_id"] != tokenizer.tokenizer_id:
        raise HumanCollectionError("tokenization.tokenizer_id does not match pinned counter")
    if tokenization["tokenizer_revision"] != tokenizer.tokenizer_revision:
        raise HumanCollectionError("tokenization.tokenizer_revision does not match pinned counter")
    if tokenization["tokenizer_sha256"] != tokenizer.tokenizer_sha256:
        raise HumanCollectionError("tokenization.tokenizer_sha256 does not match pinned counter")


def _validate_model_input_renderer(renderer: PinnedModelInputRenderer) -> None:
    if not isinstance(renderer, PinnedModelInputRenderer):
        raise HumanCollectionError("model_input_renderer must be a PinnedModelInputRenderer")
    _text(renderer.renderer_id, "model_input_renderer.renderer_id", identifier=True)
    _text(renderer.renderer_revision, "model_input_renderer.renderer_revision", identifier=True)
    _sha256(renderer.renderer_sha256, "model_input_renderer.renderer_sha256")
    if not callable(renderer.renderer):
        raise HumanCollectionError("model_input_renderer.renderer must be callable")


def _recompute_model_input(inputs: Mapping[str, Any], renderer: PinnedModelInputRenderer) -> str:
    _validate_model_input_renderer(renderer)
    identity = _object(
        inputs["model_input_renderer_identity"],
        "input.model_input_renderer_identity",
        _RENDERER_IDENTITY_FIELDS,
    )
    expected_identity = {
        "identifier": renderer.renderer_id,
        "revision": renderer.renderer_revision,
        "sha256": renderer.renderer_sha256,
    }
    if identity != expected_identity:
        raise HumanCollectionError(
            "input.model_input_renderer_identity does not match pinned renderer"
        )
    source = {
        key: copy.deepcopy(value)
        for key, value in inputs.items()
        if key not in {"model_input", "model_input_renderer_identity"}
    }
    _strict_json(source, path="model_input_renderer.source")
    try:
        rendered = renderer.renderer(source)
    except Exception as error:
        raise HumanCollectionError("pinned model-input renderer failed") from error
    return _text(rendered, "pinned model-input renderer output")


def _reference_timestamp(value: object, timezone: ZoneInfo, fold_value: object) -> datetime:
    text = _text(value, "input.reference_timestamp")
    if _OFFSET_TIMESTAMP_RE.fullmatch(text) is None:
        raise HumanCollectionError(
            "input.reference_timestamp must be a full ISO 8601 timestamp with an offset"
        )
    fold = _nonnegative_int(fold_value, "input.reference_fold")
    if fold not in {0, 1}:
        raise HumanCollectionError("input.reference_fold must be 0 or 1")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as error:
        raise HumanCollectionError(
            "input.reference_timestamp is not a valid ISO 8601 timestamp"
        ) from error
    if parsed.utcoffset() is None:
        raise HumanCollectionError("input.reference_timestamp must include an explicit UTC offset")

    naive = parsed.replace(tzinfo=None)
    zoned = naive.replace(tzinfo=timezone, fold=fold)
    if zoned.utcoffset() != parsed.utcoffset():
        raise HumanCollectionError(
            "input.reference_timestamp offset/fold does not match input.timezone"
        )
    roundtrip = zoned.astimezone(datetime_timezone.utc).astimezone(timezone)
    if roundtrip.replace(tzinfo=None) != naive or roundtrip.fold != fold:
        raise HumanCollectionError(
            "input.reference_timestamp is a nonexistent local time or has the wrong fold"
        )
    return parsed


def _normalized_request_hash(request: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFC", request).casefold().split())
    return canonical_sha256(normalized)


def _prompt_hash_payload(record: Mapping[str, Any]) -> dict[str, object]:
    payload = copy.deepcopy(dict(record))
    payload["integrity"]["prompt_record_sha256"] = "0" * 64
    return payload


def _validate_prompt_metadata(
    record: Mapping[str, Any],
    *,
    compared_release_cutoff_utc: str,
    tokenizer: PinnedTokenCounter,
    model_input_renderer: PinnedModelInputRenderer,
) -> None:
    if record["schema_version"] != HUMAN_COLLECTION_SCHEMA_VERSION:
        raise HumanCollectionError(f"schema_version must be {HUMAN_COLLECTION_SCHEMA_VERSION!r}")
    _text(record["record_id"], "record_id", identifier=True)
    _text(record["cluster_id"], "cluster_id", identifier=True)
    if record["role"] not in _ROLES:
        raise HumanCollectionError(f"role must be one of {sorted(_ROLES)}")
    if record["task_class"] not in _TASK_CLASSES:
        raise HumanCollectionError(f"task_class must be one of {sorted(_TASK_CLASSES)}")

    cutoff = _utc(compared_release_cutoff_utc, "compared_release_cutoff_utc")
    provenance = _object(record["provenance"], "provenance", _PROVENANCE_FIELDS)
    for field_name in (
        "author_id",
        "source_id",
        "collection_batch",
        "authoring_protocol_revision",
        "schema_family",
        "paraphrase_family",
        "temporal_construction_id",
    ):
        _text(provenance[field_name], f"provenance.{field_name}", identifier=True)
    assigned = _utc(provenance["role_assigned_at_utc"], "provenance.role_assigned_at_utc")
    authored = _utc(provenance["authored_at_utc"], "provenance.authored_at_utc")
    if assigned >= authored:
        raise HumanCollectionError("role must be assigned strictly before authoring")
    if authored <= cutoff:
        raise HumanCollectionError("authored_at_utc must be strictly after compared model releases")
    _text(provenance["license"], "provenance.license")
    if provenance["consent"] is not True:
        raise HumanCollectionError("provenance.consent must be true")
    if provenance["no_model_assistance"] is not True:
        raise HumanCollectionError("provenance.no_model_assistance must be true")
    entity_sources = provenance["entity_source_ids"]
    if type(entity_sources) is not list or not entity_sources:
        raise HumanCollectionError("provenance.entity_source_ids must be a non-empty array")
    checked_sources = [
        _text(value, "provenance.entity_source_ids[]", identifier=True) for value in entity_sources
    ]
    if len(set(checked_sources)) != len(checked_sources):
        raise HumanCollectionError("provenance.entity_source_ids must not contain duplicates")

    inputs = _object(record["input"], "input", _INPUT_FIELDS)
    request = _text(inputs["request"], "input.request")
    if not request.strip():
        raise HumanCollectionError("input.request must contain non-whitespace text")
    if type(inputs["context"]) is not dict:
        raise HumanCollectionError("input.context must be a strict JSON object")
    _strict_json(inputs["context"], path="input.context")
    model_input = _text(inputs["model_input"], "input.model_input")
    if model_input != _recompute_model_input(inputs, model_input_renderer):
        raise HumanCollectionError("input.model_input does not match the pinned renderer")
    timezone_name = _text(inputs["timezone"], "input.timezone")
    try:
        timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise HumanCollectionError("input.timezone must be a valid IANA timezone") from error
    _reference_timestamp(inputs["reference_timestamp"], timezone, inputs["reference_fold"])

    tool_schemas = inputs["tool_schemas"]
    if type(tool_schemas) is not list or not tool_schemas:
        raise HumanCollectionError("input.tool_schemas must be a non-empty array")
    _strict_json(tool_schemas, path="input.tool_schemas")
    try:
        parse_tool_declarations(tool_schemas)
    except (ContractError, TypeError, ValueError) as error:
        raise HumanCollectionError(f"input.tool_schemas is invalid: {error}") from error
    identity = _object(
        inputs["tool_schema_identity"], "input.tool_schema_identity", _TOOL_IDENTITY_FIELDS
    )
    _text(identity["identifier"], "input.tool_schema_identity.identifier", identifier=True)
    _text(identity["revision"], "input.tool_schema_identity.revision", identifier=True)
    if identity["contract_version"] != SCHEMA_CONTRACT_VERSION:
        raise HumanCollectionError(
            f"input.tool_schema_identity.contract_version must be {SCHEMA_CONTRACT_VERSION!r}"
        )
    if _sha256(identity["sha256"], "input.tool_schema_identity.sha256") != canonical_sha256(
        tool_schemas
    ):
        raise HumanCollectionError("input.tool_schema_identity.sha256 does not match tool_schemas")

    eligibility = _object(record["eligibility"], "eligibility", _ELIGIBILITY_FIELDS)
    if type(eligibility["eligible"]) is not bool:
        raise HumanCollectionError("eligibility.eligible must be boolean")
    expected_decision = "include" if eligibility["eligible"] else "exclude"
    if eligibility["decision"] != expected_decision:
        raise HumanCollectionError(f"eligibility.decision must be {expected_decision!r}")
    if eligibility["eligible"]:
        if eligibility["reason"] is not None:
            raise HumanCollectionError("included records must have null eligibility.reason")
    else:
        _text(eligibility["reason"], "eligibility.reason")
    _text(eligibility["decider_id"], "eligibility.decider_id", identifier=True)
    eligibility_at = _utc(eligibility["decided_at_utc"], "eligibility.decided_at_utc")
    if eligibility_at <= authored:
        raise HumanCollectionError("eligibility must be decided strictly after authoring")

    evidence = _object(
        record["duplicate_evidence"], "duplicate_evidence", _DUPLICATE_EVIDENCE_FIELDS
    )
    expected_input_hash = canonical_sha256(inputs)
    if _sha256(evidence["input_sha256"], "duplicate_evidence.input_sha256") != expected_input_hash:
        raise HumanCollectionError("duplicate_evidence.input_sha256 does not match input")
    expected_request_hash = _normalized_request_hash(request)
    if (
        _sha256(
            evidence["normalized_request_sha256"],
            "duplicate_evidence.normalized_request_sha256",
        )
        != expected_request_hash
    ):
        raise HumanCollectionError("normalized request fingerprint mismatch")
    _sha256(
        evidence["delexicalized_template_sha256"],
        "duplicate_evidence.delexicalized_template_sha256",
    )
    _text(
        evidence["near_duplicate_cluster_id"],
        "duplicate_evidence.near_duplicate_cluster_id",
        identifier=True,
    )
    if evidence["normalizer_revision"] != TEXT_NORMALIZER_VERSION:
        raise HumanCollectionError(
            f"duplicate_evidence.normalizer_revision must be {TEXT_NORMALIZER_VERSION!r}"
        )
    for field_name in (
        "exact_scan_manifest_sha256",
        "near_scan_manifest_sha256",
        "scan_code_sha256",
    ):
        _sha256(evidence[field_name], f"duplicate_evidence.{field_name}")
    scan_completed_at = _utc(
        evidence["scan_completed_at_utc"], "duplicate_evidence.scan_completed_at_utc"
    )
    if scan_completed_at <= authored:
        raise HumanCollectionError("duplicate scan must complete strictly after authoring")
    if scan_completed_at >= eligibility_at:
        raise HumanCollectionError(
            "input-only eligibility must be decided strictly after duplicate scanning completes"
        )

    tokenization = _object(record["tokenization"], "tokenization", _PROMPT_TOKENIZATION_FIELDS)
    _validate_token_identity(tokenization, tokenizer)
    stored_tokens = _nonnegative_int(tokenization["input_tokens"], "tokenization.input_tokens")
    recomputed_tokens = _count_tokens(tokenizer, inputs["model_input"], "input_tokens")
    if stored_tokens != recomputed_tokens:
        raise HumanCollectionError("tokenization.input_tokens does not match pinned counter")


def seal_prompt_record(
    record: Mapping[str, Any],
    *,
    compared_release_cutoff_utc: str,
    tokenizer: PinnedTokenCounter,
    model_input_renderer: PinnedModelInputRenderer,
) -> dict[str, Any]:
    """Seal a structurally label-separated prompt after renderer and token recomputation."""

    result = _snapshot_json_object(record, "prompt_record", _PROMPT_FIELDS)
    inputs = _object(result["input"], "input", _INPUT_FIELDS)
    renderer_identity = _object(
        inputs["model_input_renderer_identity"],
        "input.model_input_renderer_identity",
        _RENDERER_IDENTITY_FIELDS,
    )
    _validate_model_input_renderer(model_input_renderer)
    renderer_identity.update(
        {
            "identifier": model_input_renderer.renderer_id,
            "revision": model_input_renderer.renderer_revision,
            "sha256": model_input_renderer.renderer_sha256,
        }
    )
    inputs["model_input"] = _recompute_model_input(inputs, model_input_renderer)
    evidence = _object(
        result["duplicate_evidence"], "duplicate_evidence", _DUPLICATE_EVIDENCE_FIELDS
    )
    evidence["input_sha256"] = canonical_sha256(inputs)
    evidence["normalized_request_sha256"] = _normalized_request_hash(inputs["request"])
    tokenization = _object(result["tokenization"], "tokenization", _PROMPT_TOKENIZATION_FIELDS)
    _validate_token_counter(tokenizer)
    tokenization.update(
        {
            "tokenizer_id": tokenizer.tokenizer_id,
            "tokenizer_revision": tokenizer.tokenizer_revision,
            "tokenizer_sha256": tokenizer.tokenizer_sha256,
            "input_tokens": _count_tokens(tokenizer, inputs["model_input"], "input_tokens"),
        }
    )
    integrity = _object(result["integrity"], "integrity", _PROMPT_INTEGRITY_FIELDS)
    integrity["input_sha256"] = canonical_sha256(inputs)
    integrity["prompt_record_sha256"] = "0" * 64
    integrity["prompt_record_sha256"] = canonical_sha256(_prompt_hash_payload(result))
    validate_prompt_record(
        result,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        tokenizer=tokenizer,
        model_input_renderer=model_input_renderer,
    )
    return result


def validate_prompt_record(
    record: object,
    *,
    compared_release_cutoff_utc: str,
    tokenizer: PinnedTokenCounter,
    model_input_renderer: PinnedModelInputRenderer,
) -> Mapping[str, Any]:
    """Validate a prompt schema with no dedicated annotation or gold-label field.

    The pinned renderer recomputes ``model_input`` from the exact source fields.  An external
    collection audit must still verify that arbitrary source content is label-free.
    """

    checked = _snapshot_json_object(record, "prompt_record", _PROMPT_FIELDS)
    _validate_prompt_metadata(
        checked,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        tokenizer=tokenizer,
        model_input_renderer=model_input_renderer,
    )
    integrity = _object(checked["integrity"], "integrity", _PROMPT_INTEGRITY_FIELDS)
    input_hash = _sha256(integrity["input_sha256"], "integrity.input_sha256")
    if input_hash != canonical_sha256(checked["input"]):
        raise HumanCollectionError("integrity.input_sha256 mismatch")
    record_hash = _sha256(integrity["prompt_record_sha256"], "integrity.prompt_record_sha256")
    if record_hash != canonical_sha256(_prompt_hash_payload(checked)):
        raise HumanCollectionError("integrity.prompt_record_sha256 mismatch")
    return checked


def prompt_population_sha256(
    records: Sequence[Mapping[str, Any]],
    *,
    compared_release_cutoff_utc: str,
    tokenizer: PinnedTokenCounter,
    model_input_renderer: PinnedModelInputRenderer,
) -> str:
    """Hash sorted public prompt-record hashes after complete validation."""

    snapshot = _snapshot_json_sequence(records, "prompt population")
    hashes: list[str] = []
    for record in snapshot:
        checked = validate_prompt_record(
            record,
            compared_release_cutoff_utc=compared_release_cutoff_utc,
            tokenizer=tokenizer,
            model_input_renderer=model_input_renderer,
        )
        hashes.append(checked["integrity"]["prompt_record_sha256"])
    if len(set(hashes)) != len(hashes):
        raise HumanCollectionError("population contains duplicate prompt-record hashes")
    return canonical_sha256(sorted(hashes))


def audit_population_firewall(
    selection: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
    *,
    compared_release_cutoff_utc: str,
    tokenizer: PinnedTokenCounter,
    model_input_renderer: PinnedModelInputRenderer,
    expected_exact_scan_manifest_sha256: str,
    expected_near_scan_manifest_sha256: str,
    expected_scan_code_sha256: str,
    require_schema_family_separation: bool,
    minimum_selection_eligible_clusters: int,
    minimum_confirmation_efficacy_clusters: int = 2_500,
    minimum_confirmation_safety_clusters: int = 1_500,
) -> dict[str, int | str]:
    """Audit fresh provenance, scan evidence, population separation, and eligible power.

    Exact and near-duplicate scan manifests remain external evidence.  This function requires
    their pinned identities on every record and independently rejects overlapping deterministic
    fingerprints and near-duplicate cluster IDs; it never claims to have run a similarity scan.
    """

    minimum_selection = _positive_int(
        minimum_selection_eligible_clusters, "minimum_selection_eligible_clusters"
    )
    minimum_efficacy = _positive_int(
        minimum_confirmation_efficacy_clusters,
        "minimum_confirmation_efficacy_clusters",
    )
    minimum_safety = _positive_int(
        minimum_confirmation_safety_clusters,
        "minimum_confirmation_safety_clusters",
    )
    exact_manifest = _sha256(
        expected_exact_scan_manifest_sha256, "expected_exact_scan_manifest_sha256"
    )
    near_manifest = _sha256(
        expected_near_scan_manifest_sha256, "expected_near_scan_manifest_sha256"
    )
    scan_code = _sha256(expected_scan_code_sha256, "expected_scan_code_sha256")
    if type(require_schema_family_separation) is not bool:
        raise HumanCollectionError("require_schema_family_separation must be boolean")

    populations = {
        "selection": _snapshot_json_sequence(selection, "selection population"),
        "confirmation": _snapshot_json_sequence(confirmation, "confirmation population"),
    }
    checked: dict[str, list[Mapping[str, Any]]] = {"selection": [], "confirmation": []}
    for expected_role, records in populations.items():
        if not records:
            raise HumanCollectionError(f"{expected_role} population must be non-empty")
        for index, record in enumerate(records):
            validated = validate_prompt_record(
                record,
                compared_release_cutoff_utc=compared_release_cutoff_utc,
                tokenizer=tokenizer,
                model_input_renderer=model_input_renderer,
            )
            if validated["role"] != expected_role:
                raise HumanCollectionError(
                    f"{expected_role}[{index}] has leaked role {validated['role']!r}"
                )
            evidence = validated["duplicate_evidence"]
            if evidence["exact_scan_manifest_sha256"] != exact_manifest:
                raise HumanCollectionError("record is bound to a different exact-scan manifest")
            if evidence["near_scan_manifest_sha256"] != near_manifest:
                raise HumanCollectionError("record is bound to a different near-scan manifest")
            if evidence["scan_code_sha256"] != scan_code:
                raise HumanCollectionError("record is bound to different duplicate-scan code")
            checked[expected_role].append(validated)
        record_ids = [record["record_id"] for record in checked[expected_role]]
        if len(set(record_ids)) != len(record_ids):
            raise HumanCollectionError(f"{expected_role} record IDs must be unique")

    def values(records: Sequence[Mapping[str, Any]], key: str) -> set[str]:
        if key in {"record_id", "cluster_id"}:
            return {str(record[key]) for record in records}
        if key == "entity_source_ids":
            return {
                str(value)
                for record in records
                for value in record["provenance"]["entity_source_ids"]
            }
        if key in _DUPLICATE_EVIDENCE_FIELDS:
            return {str(record["duplicate_evidence"][key]) for record in records}
        return {str(record["provenance"][key]) for record in records}

    dimensions = [
        "record_id",
        "cluster_id",
        "author_id",
        "source_id",
        "collection_batch",
        "paraphrase_family",
        "entity_source_ids",
        "temporal_construction_id",
        "input_sha256",
        "normalized_request_sha256",
        "delexicalized_template_sha256",
        "near_duplicate_cluster_id",
    ]
    if require_schema_family_separation:
        dimensions.append("schema_family")
    for dimension in dimensions:
        overlap = values(checked["selection"], dimension) & values(
            checked["confirmation"], dimension
        )
        if overlap:
            raise HumanCollectionError(
                f"cross-population {dimension} overlap ({len(overlap)}): {sorted(overlap)[:3]}"
            )

    all_input_hashes = [
        str(record["duplicate_evidence"]["input_sha256"])
        for role in ("selection", "confirmation")
        for record in checked[role]
    ]
    if len(set(all_input_hashes)) != len(all_input_hashes):
        raise HumanCollectionError("exact duplicate inputs exist within a population")

    def cluster_summary(records: Sequence[Mapping[str, Any]]) -> dict[int, tuple[str, bool]]:
        """Count connected independence units, not caller-declared cluster names.

        A new ``cluster_id`` cannot manufacture another independent observation when two records
        still share an author/source/batch or any semantic, entity, temporal, schema, or duplicate
        lineage.  The connected-component definition is deliberately conservative because these
        counts gate a one-shot confirmatory evaluation.
        """

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

        first_by_lineage: dict[tuple[str, str], int] = {}
        for index, record in enumerate(records):
            provenance = record["provenance"]
            duplicate = record["duplicate_evidence"]
            lineages: list[tuple[str, str]] = [
                ("cluster_id", str(record["cluster_id"])),
                ("author_id", str(provenance["author_id"])),
                ("source_id", str(provenance["source_id"])),
                ("collection_batch", str(provenance["collection_batch"])),
                ("paraphrase_family", str(provenance["paraphrase_family"])),
                ("temporal_construction_id", str(provenance["temporal_construction_id"])),
                ("input_sha256", str(duplicate["input_sha256"])),
                (
                    "normalized_request_sha256",
                    str(duplicate["normalized_request_sha256"]),
                ),
                (
                    "delexicalized_template_sha256",
                    str(duplicate["delexicalized_template_sha256"]),
                ),
                ("near_duplicate_cluster_id", str(duplicate["near_duplicate_cluster_id"])),
            ]
            lineages.extend(
                ("entity_source_id", str(value)) for value in provenance["entity_source_ids"]
            )
            if require_schema_family_separation:
                lineages.append(("schema_family", str(provenance["schema_family"])))
            for lineage in lineages:
                previous = first_by_lineage.setdefault(lineage, index)
                union(index, previous)

        summary: dict[int, tuple[str, bool]] = {}
        for index, record in enumerate(records):
            state = (record["task_class"], record["eligibility"]["eligible"])
            root = find(index)
            previous = summary.setdefault(root, state)
            if previous != state:
                raise HumanCollectionError(
                    "an effective independent cluster mixes task classes or eligibility decisions"
                )
        return summary

    selection_clusters = cluster_summary(checked["selection"])
    confirmation_clusters = cluster_summary(checked["confirmation"])
    selection_eligible_clusters = sum(eligible for _, eligible in selection_clusters.values())
    selection_excluded_clusters = len(selection_clusters) - selection_eligible_clusters
    confirmation_eligible_efficacy_clusters = sum(
        task_class == "efficacy" and eligible
        for task_class, eligible in confirmation_clusters.values()
    )
    confirmation_excluded_efficacy_clusters = sum(
        task_class == "efficacy" and not eligible
        for task_class, eligible in confirmation_clusters.values()
    )
    confirmation_eligible_safety_clusters = sum(
        task_class == "safety" and eligible
        for task_class, eligible in confirmation_clusters.values()
    )
    confirmation_excluded_safety_clusters = sum(
        task_class == "safety" and not eligible
        for task_class, eligible in confirmation_clusters.values()
    )
    confirmation_collected_efficacy_clusters = (
        confirmation_eligible_efficacy_clusters + confirmation_excluded_efficacy_clusters
    )
    confirmation_collected_safety_clusters = (
        confirmation_eligible_safety_clusters + confirmation_excluded_safety_clusters
    )
    confirmation_eligible_clusters = (
        confirmation_eligible_efficacy_clusters + confirmation_eligible_safety_clusters
    )
    confirmation_excluded_clusters = (
        confirmation_excluded_efficacy_clusters + confirmation_excluded_safety_clusters
    )
    if selection_eligible_clusters < minimum_selection:
        raise HumanCollectionError(
            "selection eligible clusters below minimum: "
            f"{selection_eligible_clusters} < {minimum_selection}"
        )
    if confirmation_eligible_efficacy_clusters < minimum_efficacy:
        raise HumanCollectionError(
            "confirmation eligible efficacy clusters below minimum: "
            f"{confirmation_eligible_efficacy_clusters} < {minimum_efficacy}"
        )
    if confirmation_eligible_safety_clusters < minimum_safety:
        raise HumanCollectionError(
            "confirmation eligible safety clusters below minimum: "
            f"{confirmation_eligible_safety_clusters} < {minimum_safety}"
        )

    selection_eligible_records = sum(
        record["eligibility"]["eligible"] for record in checked["selection"]
    )
    confirmation_eligible_records = sum(
        record["eligibility"]["eligible"] for record in checked["confirmation"]
    )
    return {
        "schema_version": HUMAN_COLLECTION_SCHEMA_VERSION,
        "selection_collected_records": len(checked["selection"]),
        "selection_excluded_records": len(checked["selection"]) - selection_eligible_records,
        "selection_eligible_records": selection_eligible_records,
        "selection_collected_clusters": len(selection_clusters),
        "selection_excluded_clusters": selection_excluded_clusters,
        "selection_eligible_clusters": selection_eligible_clusters,
        "confirmation_collected_records": len(checked["confirmation"]),
        "confirmation_excluded_records": (
            len(checked["confirmation"]) - confirmation_eligible_records
        ),
        "confirmation_eligible_records": confirmation_eligible_records,
        "confirmation_collected_clusters": len(confirmation_clusters),
        "confirmation_excluded_clusters": confirmation_excluded_clusters,
        "confirmation_eligible_clusters": confirmation_eligible_clusters,
        "confirmation_collected_efficacy_clusters": (confirmation_collected_efficacy_clusters),
        "confirmation_excluded_efficacy_clusters": confirmation_excluded_efficacy_clusters,
        "confirmation_eligible_efficacy_clusters": confirmation_eligible_efficacy_clusters,
        "confirmation_collected_safety_clusters": confirmation_collected_safety_clusters,
        "confirmation_excluded_safety_clusters": confirmation_excluded_safety_clusters,
        "confirmation_eligible_safety_clusters": confirmation_eligible_safety_clusters,
    }


def _canonical_action_ir(
    value: object, prompt_record: Mapping[str, Any], field_name: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise HumanCollectionError(f"{field_name} must be an Action IR JSON object")
    _strict_json(value, path=field_name)
    try:
        decoded = decode_json_object(_canonical_bytes(value).decode("utf-8"))
        declarations = parse_tool_declarations(prompt_record["input"]["tool_schemas"])
        registry = {declaration.schema.name: declaration.schema for declaration in declarations}
        action = validate_action_ir(decoded, registry)
    except (ActionIRError, ContractError, TypeError, ValueError) as error:
        raise HumanCollectionError(
            f"{field_name} is not schema-valid Action IR: {error}"
        ) from error
    return action.to_dict()


def _canonicalize_annotation(
    annotation_value: object,
    prompt_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], datetime]:
    annotation = copy.deepcopy(
        _object(annotation_value, "label_envelope.annotation", _ANNOTATION_FIELDS)
    )
    authored = _utc(prompt_record["provenance"]["authored_at_utc"], "provenance.authored_at_utc")
    eligibility_at = _utc(
        prompt_record["eligibility"]["decided_at_utc"], "eligibility.decided_at_utc"
    )
    author_id = prompt_record["provenance"]["author_id"]
    labeler_id = _text(annotation["labeler_id"], "annotation.labeler_id", identifier=True)
    reviewer_id = _text(annotation["reviewer_id"], "annotation.reviewer_id", identifier=True)
    if len({author_id, labeler_id, reviewer_id}) != 3:
        raise HumanCollectionError("author, labeler, and reviewer must be independent")
    if annotation["labeler_no_model_assistance"] is not True:
        raise HumanCollectionError("annotation.labeler_no_model_assistance must be true")
    if annotation["reviewer_no_model_assistance"] is not True:
        raise HumanCollectionError("annotation.reviewer_no_model_assistance must be true")
    labeler_at = _utc(annotation["labeler_labeled_at_utc"], "annotation.labeler_labeled_at_utc")
    reviewer_at = _utc(annotation["reviewer_labeled_at_utc"], "annotation.reviewer_labeled_at_utc")
    if min(labeler_at, reviewer_at) <= authored:
        raise HumanCollectionError("labels must be created strictly after authoring")
    if eligibility_at >= min(labeler_at, reviewer_at):
        raise HumanCollectionError(
            "input-only eligibility must be decided strictly before labeling"
        )

    annotation["labeler_action_ir"] = _canonical_action_ir(
        annotation["labeler_action_ir"], prompt_record, "annotation.labeler_action_ir"
    )
    annotation["reviewer_action_ir"] = _canonical_action_ir(
        annotation["reviewer_action_ir"], prompt_record, "annotation.reviewer_action_ir"
    )
    actual_disagreement = canonical_sha256(annotation["labeler_action_ir"]) != canonical_sha256(
        annotation["reviewer_action_ir"]
    )
    if type(annotation["disagreement"]) is not bool:
        raise HumanCollectionError("annotation.disagreement must be boolean")
    if annotation["disagreement"] != actual_disagreement:
        raise HumanCollectionError("annotation.disagreement contradicts independent labels")

    adjudication_values = (
        annotation["adjudicator_id"],
        annotation["adjudicator_no_model_assistance"],
        annotation["adjudicated_at_utc"],
        annotation["adjudicated_action_ir"],
    )
    if actual_disagreement:
        if any(value is None for value in adjudication_values):
            raise HumanCollectionError("disagreements require complete adjudication metadata")
        adjudicator_id = _text(
            annotation["adjudicator_id"], "annotation.adjudicator_id", identifier=True
        )
        if adjudicator_id in {author_id, labeler_id, reviewer_id}:
            raise HumanCollectionError("adjudicator must be independent")
        if annotation["adjudicator_no_model_assistance"] is not True:
            raise HumanCollectionError("annotation.adjudicator_no_model_assistance must be true")
        adjudicated_at = _utc(annotation["adjudicated_at_utc"], "annotation.adjudicated_at_utc")
        if adjudicated_at <= max(labeler_at, reviewer_at):
            raise HumanCollectionError(
                "adjudication must occur strictly after both independent labels"
            )
        annotation["adjudicated_action_ir"] = _canonical_action_ir(
            annotation["adjudicated_action_ir"],
            prompt_record,
            "annotation.adjudicated_action_ir",
        )
        final_label = annotation["adjudicated_action_ir"]
        final_created_at = adjudicated_at
    else:
        if any(value is not None for value in adjudication_values):
            raise HumanCollectionError("agreements must not contain adjudication metadata")
        final_label = annotation["labeler_action_ir"]
        final_created_at = max(labeler_at, reviewer_at)
    return annotation, final_label, final_created_at


def _label_envelope_hash_payload(envelope: Mapping[str, Any]) -> dict[str, object]:
    payload = copy.deepcopy(dict(envelope))
    payload["envelope_sha256"] = "0" * 64
    payload["envelope_hmac_sha256"] = "0" * 64
    return payload


def _label_envelope_mac_payload(envelope: Mapping[str, Any]) -> dict[str, object]:
    payload = copy.deepcopy(dict(envelope))
    payload["envelope_hmac_sha256"] = "0" * 64
    return payload


def seal_label_envelope(
    envelope: Mapping[str, Any],
    *,
    prompt_record: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    tokenizer: PinnedTokenCounter,
    model_input_renderer: PinnedModelInputRenderer,
    signing_key: bytes,
) -> dict[str, Any]:
    """Canonicalize and authenticate an envelope that is opaque outside trusted storage."""

    key = _hmac_key(signing_key, "label signing_key")
    prompt = validate_prompt_record(
        prompt_record,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        tokenizer=tokenizer,
        model_input_renderer=model_input_renderer,
    )
    if not prompt["eligibility"]["eligible"]:
        raise HumanCollectionError("excluded prompts must not receive label envelopes")
    result = _snapshot_json_object(envelope, "label_envelope", _LABEL_ENVELOPE_FIELDS)
    annotation, final_label, final_created_at = _canonicalize_annotation(
        result["annotation"], prompt
    )
    result["annotation"] = annotation
    result["record_id"] = prompt["record_id"]
    result["prompt_record_sha256"] = prompt["integrity"]["prompt_record_sha256"]
    sealed_at = _utc(result["sealed_at_utc"], "label_envelope.sealed_at_utc")
    if sealed_at <= final_created_at:
        raise HumanCollectionError(
            "label envelope must be sealed strictly after final label creation"
        )
    _text(result["key_id"], "label_envelope.key_id", identifier=True)
    tokenization = _object(
        result["tokenization"], "label_envelope.tokenization", _LABEL_TOKENIZATION_FIELDS
    )
    tokenization.update(
        {
            "tokenizer_id": tokenizer.tokenizer_id,
            "tokenizer_revision": tokenizer.tokenizer_revision,
            "tokenizer_sha256": tokenizer.tokenizer_sha256,
            "output_tokens": _count_tokens(
                tokenizer,
                _canonical_bytes(final_label).decode("utf-8"),
                "output_tokens",
            ),
        }
    )
    result["envelope_sha256"] = "0" * 64
    result["envelope_hmac_sha256"] = "0" * 64
    result["envelope_sha256"] = canonical_sha256(_label_envelope_hash_payload(result))
    result["envelope_hmac_sha256"] = _hmac_sha256(key, _label_envelope_mac_payload(result))
    validate_label_envelope(
        result,
        prompt_record=prompt,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        tokenizer=tokenizer,
        model_input_renderer=model_input_renderer,
        signing_key=key,
        expected_key_id=result["key_id"],
    )
    return result


def validate_label_envelope(
    envelope: object,
    *,
    prompt_record: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    tokenizer: PinnedTokenCounter,
    model_input_renderer: PinnedModelInputRenderer,
    signing_key: bytes,
    expected_key_id: str,
) -> Mapping[str, Any]:
    """Validate a private envelope; callers must keep it inside an external isolation boundary."""

    key = _hmac_key(signing_key, "label signing_key")
    checked = _snapshot_json_object(envelope, "label_envelope", _LABEL_ENVELOPE_FIELDS)
    if checked["schema_version"] != LABEL_ENVELOPE_SCHEMA_VERSION:
        raise HumanCollectionError(
            f"label envelope schema must be {LABEL_ENVELOPE_SCHEMA_VERSION!r}"
        )
    prompt = validate_prompt_record(
        prompt_record,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        tokenizer=tokenizer,
        model_input_renderer=model_input_renderer,
    )
    if not prompt["eligibility"]["eligible"]:
        raise HumanCollectionError("excluded prompts must not have label envelopes")
    if checked["record_id"] != prompt["record_id"]:
        raise HumanCollectionError("label envelope is bound to a different record")
    if checked["prompt_record_sha256"] != prompt["integrity"]["prompt_record_sha256"]:
        raise HumanCollectionError("label envelope is bound to a different prompt hash")
    if checked["key_id"] != _text(expected_key_id, "expected_label_key_id", identifier=True):
        raise HumanCollectionError("label envelope key_id mismatch")
    annotation, final_label, final_created_at = _canonicalize_annotation(
        checked["annotation"], prompt
    )
    if canonical_sha256(annotation) != canonical_sha256(checked["annotation"]):
        raise HumanCollectionError("stored human labels are not canonical Action IR")
    if _utc(checked["sealed_at_utc"], "label_envelope.sealed_at_utc") <= final_created_at:
        raise HumanCollectionError(
            "label envelope must be sealed strictly after final label creation"
        )
    tokenization = _object(
        checked["tokenization"], "label_envelope.tokenization", _LABEL_TOKENIZATION_FIELDS
    )
    _validate_token_identity(tokenization, tokenizer)
    stored_output_tokens = _nonnegative_int(
        tokenization["output_tokens"], "label_envelope.tokenization.output_tokens"
    )
    expected_output_tokens = _count_tokens(
        tokenizer, _canonical_bytes(final_label).decode("utf-8"), "output_tokens"
    )
    if stored_output_tokens != expected_output_tokens:
        raise HumanCollectionError("label output token count does not match pinned counter")
    envelope_hash = _sha256(checked["envelope_sha256"], "label_envelope.envelope_sha256")
    if envelope_hash != canonical_sha256(_label_envelope_hash_payload(checked)):
        raise HumanCollectionError("label envelope hash mismatch")
    envelope_mac = _sha256(checked["envelope_hmac_sha256"], "label_envelope.envelope_hmac_sha256")
    expected_mac = _hmac_sha256(key, _label_envelope_mac_payload(checked))
    if not hmac.compare_digest(envelope_mac, expected_mac):
        raise HumanCollectionError("label envelope HMAC mismatch")
    return checked


def _receipt_mac_payload(receipt: Mapping[str, Any]) -> dict[str, object]:
    payload = copy.deepcopy(dict(receipt))
    payload["receipt_hmac_sha256"] = "0" * 64
    return payload


def _validate_receipt_semantics(receipt: Mapping[str, Any]) -> None:
    if receipt["schema_version"] != SELECTION_RECEIPT_SCHEMA_VERSION:
        raise HumanCollectionError(
            f"selection receipt schema must be {SELECTION_RECEIPT_SCHEMA_VERSION!r}"
        )
    for field_name in _RECEIPT_BINDING_FIELDS - {"source_revision", "code_revision"}:
        _sha256(receipt[field_name], f"selection_receipt.{field_name}")
    _text(receipt["source_revision"], "selection_receipt.source_revision")
    _text(receipt["code_revision"], "selection_receipt.code_revision")
    gates = receipt["gate_results"]
    if type(gates) is not dict or set(gates) != FROZEN_SELECTION_GATES:
        raise HumanCollectionError(
            "selection_receipt.gate_results must contain the exact frozen gate set"
        )
    if any(type(value) is not bool for value in gates.values()):
        raise HumanCollectionError("every selection gate result must be boolean")
    all_passed = all(gates.values())
    if type(receipt["passed"]) is not bool or receipt["passed"] != all_passed:
        raise HumanCollectionError("selection_receipt.passed contradicts gate_results")
    _utc(receipt["created_at_utc"], "selection_receipt.created_at_utc")
    _text(receipt["key_id"], "selection_receipt.key_id", identifier=True)
    _sha256(receipt["receipt_hmac_sha256"], "selection_receipt.receipt_hmac_sha256")


def seal_selection_receipt(
    receipt: Mapping[str, Any],
    *,
    signing_key: bytes,
) -> dict[str, Any]:
    """HMAC-authenticate an exact-shape receipt with an external secret key."""

    key = _hmac_key(signing_key, "receipt signing_key")
    result = _snapshot_json_object(receipt, "selection_receipt", _RECEIPT_FIELDS)
    result["receipt_hmac_sha256"] = "0" * 64
    _validate_receipt_semantics(result)
    result["receipt_hmac_sha256"] = _hmac_sha256(key, _receipt_mac_payload(result))
    return result


def validate_selection_receipt(
    receipt: object,
    *,
    signing_key: bytes,
    expected_key_id: str,
    expected_bindings: Mapping[str, str],
) -> Mapping[str, Any]:
    """Verify receipt HMAC, exact gates, all passing results, and every frozen binding."""

    key = _hmac_key(signing_key, "receipt signing_key")
    checked = _snapshot_json_object(receipt, "selection_receipt", _RECEIPT_FIELDS)
    _validate_receipt_semantics(checked)
    if checked["key_id"] != _text(expected_key_id, "expected_receipt_key_id", identifier=True):
        raise HumanCollectionError("selection receipt key_id mismatch")
    bindings = _snapshot_json_object(
        expected_bindings, "expected_bindings", _RECEIPT_BINDING_FIELDS
    )
    for field_name in _RECEIPT_BINDING_FIELDS:
        expected = bindings[field_name]
        if field_name not in {"source_revision", "code_revision"}:
            _sha256(expected, f"expected_bindings.{field_name}")
        else:
            _text(expected, f"expected_bindings.{field_name}")
        if checked[field_name] != expected:
            raise HumanCollectionError(f"selection receipt is bound to a different {field_name}")
    received_mac = checked["receipt_hmac_sha256"]
    expected_mac = _hmac_sha256(key, _receipt_mac_payload(checked))
    if not hmac.compare_digest(received_mac, expected_mac):
        raise HumanCollectionError("selection receipt HMAC mismatch")
    if not checked["passed"]:
        raise HumanCollectionError("selection receipt contains a failed gate")
    return checked


def _event_hash_payload(event: Mapping[str, Any]) -> dict[str, object]:
    payload = copy.deepcopy(dict(event))
    payload["event_sha256"] = "0" * 64
    payload["event_hmac_sha256"] = "0" * 64
    return payload


def _event_mac_payload(event: Mapping[str, Any]) -> dict[str, object]:
    payload = copy.deepcopy(dict(event))
    payload["event_hmac_sha256"] = "0" * 64
    return payload


def _validate_access_event_shape(event: object, index: int) -> Mapping[str, Any]:
    checked = _snapshot_json_object(event, f"access_ledger[{index}]", _ACCESS_EVENT_FIELDS)
    if checked["schema_version"] != LABEL_ACCESS_EVENT_SCHEMA_VERSION:
        raise HumanCollectionError(
            f"access event schema must be {LABEL_ACCESS_EVENT_SCHEMA_VERSION!r}"
        )
    if _nonnegative_int(checked["sequence"], f"access_ledger[{index}].sequence") != index:
        raise HumanCollectionError("access ledger sequence is not contiguous")
    for field_name in ("event_id", "record_id", "scoring_session_id", "accessor_id"):
        _text(checked[field_name], f"access_ledger[{index}].{field_name}", identifier=True)
    _text(checked["purpose"], f"access_ledger[{index}].purpose")
    _utc(checked["accessed_at_utc"], f"access_ledger[{index}].accessed_at_utc")
    for field_name in (
        "selection_receipt_sha256",
        "prompt_record_sha256",
        "label_envelope_sha256",
        "previous_event_sha256",
        "event_sha256",
        "event_hmac_sha256",
    ):
        _sha256(checked[field_name], f"access_ledger[{index}].{field_name}")
    _text(checked["key_id"], f"access_ledger[{index}].key_id", identifier=True)
    return checked


def validate_access_ledger(
    events: Sequence[Mapping[str, Any]],
    *,
    signing_key: bytes,
    expected_key_id: str,
) -> str:
    """Validate a complete durable global chain and return its head event hash."""

    key = _hmac_key(signing_key, "ledger signing_key")
    key_id = _text(expected_key_id, "expected_ledger_key_id", identifier=True)
    snapshot = _snapshot_json_sequence(events, "access ledger")
    previous_hash = "0" * 64
    prior_time: datetime | None = None
    event_ids: set[str] = set()
    record_ids: set[str] = set()
    for index, event_value in enumerate(snapshot):
        event = _validate_access_event_shape(event_value, index)
        if event["key_id"] != key_id:
            raise HumanCollectionError("access ledger key_id mismatch")
        if event["event_id"] in event_ids:
            raise HumanCollectionError("access ledger event IDs must be unique")
        event_ids.add(event["event_id"])
        if event["record_id"] in record_ids:
            raise HumanCollectionError(
                "a prompt record has already been disclosed, including across scoring sessions"
            )
        record_ids.add(event["record_id"])
        if event["previous_event_sha256"] != previous_hash:
            raise HumanCollectionError("access ledger hash chain is broken")
        accessed_at = _utc(event["accessed_at_utc"], f"access_ledger[{index}].accessed_at_utc")
        if prior_time is not None and accessed_at <= prior_time:
            raise HumanCollectionError("access ledger timestamps must be strictly increasing")
        expected_hash = canonical_sha256(_event_hash_payload(event))
        if event["event_sha256"] != expected_hash:
            raise HumanCollectionError("access event hash mismatch")
        expected_mac = _hmac_sha256(key, _event_mac_payload(event))
        if not hmac.compare_digest(event["event_hmac_sha256"], expected_mac):
            raise HumanCollectionError("access event HMAC mismatch")
        previous_hash = event["event_sha256"]
        prior_time = accessed_at
    return previous_hash


def _population_hash_from_record_hashes(record_hashes: Sequence[str]) -> str:
    snapshot = _snapshot_json_sequence(record_hashes, "confirmation prompt hashes")
    hashes = [_sha256(value, "confirmation_prompt_record_hashes[]") for value in snapshot]
    if not hashes:
        raise HumanCollectionError("confirmation prompt hashes must be non-empty")
    if len(set(hashes)) != len(hashes):
        raise HumanCollectionError("confirmation prompt hashes must be unique")
    return canonical_sha256(sorted(hashes))


def access_final_confirmation_label(
    prompt_record: Mapping[str, Any],
    private_label_envelope: Mapping[str, Any],
    *,
    compared_release_cutoff_utc: str,
    tokenizer: PinnedTokenCounter,
    model_input_renderer: PinnedModelInputRenderer,
    label_signing_key: bytes,
    expected_label_key_id: str,
    selection_receipt: Mapping[str, Any],
    receipt_signing_key: bytes,
    expected_receipt_key_id: str,
    expected_receipt_bindings: Mapping[str, str],
    confirmation_prompt_record_hashes: Sequence[str],
    prior_durable_ledger: Sequence[Mapping[str, Any]],
    ledger_signing_key: bytes,
    expected_ledger_key_id: str,
    scoring_session_id: str,
    accessor_id: str,
    purpose: str,
    event_id: str,
    accessed_at_utc: str,
    durable_append: Callable[[Mapping[str, Any], str], bool],
) -> dict[str, object]:
    """Atomically retire one confirmation record and disclose only its final Action IR.

    This function must execute inside the private label-store service.  The caller must supply the
    complete durable global ledger and an atomic ``durable_append(event, expected_head)`` operation.
    The callback must persist exactly ``event`` only if the current durable head equals
    ``expected_head``.  The final label is returned only after that callback succeeds.  Passing a
    plaintext envelope outside the isolation boundary or a callback that lies about persistence
    violates the protocol and cannot be detected by in-memory code.
    """

    prompt = validate_prompt_record(
        prompt_record,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        tokenizer=tokenizer,
        model_input_renderer=model_input_renderer,
    )
    if prompt["role"] != "confirmation":
        raise HumanCollectionError("once-only access accepts confirmation records only")
    if not prompt["eligibility"]["eligible"]:
        raise HumanCollectionError("excluded confirmation records cannot be scored")
    receipt = validate_selection_receipt(
        selection_receipt,
        signing_key=receipt_signing_key,
        expected_key_id=expected_receipt_key_id,
        expected_bindings=expected_receipt_bindings,
    )

    confirmation_hashes = _snapshot_json_sequence(
        confirmation_prompt_record_hashes, "confirmation prompt hashes"
    )
    population_hash = _population_hash_from_record_hashes(confirmation_hashes)
    if population_hash != receipt["confirmation_prompt_collection_sha256"]:
        raise HumanCollectionError("receipt is not bound to supplied confirmation membership")
    prompt_hash = prompt["integrity"]["prompt_record_sha256"]
    confirmation_hash_set = frozenset(confirmation_hashes)
    if prompt_hash not in confirmation_hash_set:
        raise HumanCollectionError("prompt record is not in the receipt-bound confirmation set")

    ledger_key = _hmac_key(ledger_signing_key, "ledger signing_key")
    ledger_key_id = _text(expected_ledger_key_id, "expected_ledger_key_id", identifier=True)
    ledger_snapshot = _snapshot_json_sequence(prior_durable_ledger, "prior durable ledger")
    previous_hash = validate_access_ledger(
        ledger_snapshot,
        signing_key=ledger_key,
        expected_key_id=ledger_key_id,
    )
    prior_record_ids = frozenset(event["record_id"] for event in ledger_snapshot)
    if prompt["record_id"] in prior_record_ids:
        raise HumanCollectionError(
            "confirmation label was already disclosed, possibly in another scoring session"
        )
    checked_event_id = _text(event_id, "event_id", identifier=True)
    if checked_event_id in frozenset(event["event_id"] for event in ledger_snapshot):
        raise HumanCollectionError("access event_id already exists in the durable ledger")

    accessed_at = _utc(accessed_at_utc, "accessed_at_utc")
    receipt_created_at = _utc(receipt["created_at_utc"], "selection_receipt.created_at_utc")
    if accessed_at <= receipt_created_at:
        raise HumanCollectionError(
            "label access must occur strictly after the passing selection receipt"
        )
    prompt_ready_at = max(
        _utc(prompt["eligibility"]["decided_at_utc"], "eligibility.decided_at_utc"),
        _utc(
            prompt["duplicate_evidence"]["scan_completed_at_utc"],
            "duplicate_evidence.scan_completed_at_utc",
        ),
    )
    if receipt_created_at <= prompt_ready_at:
        raise HumanCollectionError(
            "selection receipt must strictly postdate confirmation prompt scanning and eligibility"
        )
    if ledger_snapshot:
        last_access = _utc(
            ledger_snapshot[-1]["accessed_at_utc"],
            "prior_durable_ledger[-1].accessed_at_utc",
        )
        if accessed_at <= last_access:
            raise HumanCollectionError(
                "new ledger event timestamp must strictly postdate the prior event"
            )

    # Do not even validate/deserialise the private envelope until the receipt, membership,
    # timestamp, and global retirement checks have all passed.
    envelope = validate_label_envelope(
        private_label_envelope,
        prompt_record=prompt,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        tokenizer=tokenizer,
        model_input_renderer=model_input_renderer,
        signing_key=label_signing_key,
        expected_key_id=expected_label_key_id,
    )
    _, final_label, final_created_at = _canonicalize_annotation(envelope["annotation"], prompt)
    if accessed_at <= final_created_at:
        raise HumanCollectionError("label access must occur strictly after final label creation")
    envelope_sealed_at = _utc(envelope["sealed_at_utc"], "label_envelope.sealed_at_utc")
    if receipt_created_at <= envelope_sealed_at:
        raise HumanCollectionError(
            "selection receipt must strictly postdate the frozen confirmation label envelope"
        )

    event: dict[str, Any] = {
        "schema_version": LABEL_ACCESS_EVENT_SCHEMA_VERSION,
        "sequence": len(ledger_snapshot),
        "event_id": checked_event_id,
        "record_id": prompt["record_id"],
        "scoring_session_id": _text(scoring_session_id, "scoring_session_id", identifier=True),
        "accessor_id": _text(accessor_id, "accessor_id", identifier=True),
        "purpose": _text(purpose, "purpose"),
        "accessed_at_utc": accessed_at_utc,
        "selection_receipt_sha256": canonical_sha256(receipt),
        "prompt_record_sha256": prompt_hash,
        "label_envelope_sha256": envelope["envelope_sha256"],
        "previous_event_sha256": previous_hash,
        "key_id": ledger_key_id,
        "event_sha256": "0" * 64,
        "event_hmac_sha256": "0" * 64,
    }
    event["event_sha256"] = canonical_sha256(_event_hash_payload(event))
    event["event_hmac_sha256"] = _hmac_sha256(ledger_key, _event_mac_payload(event))
    _validate_access_event_shape(event, len(ledger_snapshot))

    if not callable(durable_append):
        raise HumanCollectionError("durable_append must be an atomic compare-and-append callback")
    try:
        committed = durable_append(copy.deepcopy(event), previous_hash)
    except Exception as error:
        raise HumanCollectionError("durable access-ledger append failed") from error
    if committed is not True:
        raise HumanCollectionError("durable access-ledger append was not atomically committed")
    return {
        "final_action_ir": copy.deepcopy(final_label),
        "access_event": copy.deepcopy(event),
    }


__all__ = [
    "FROZEN_SELECTION_GATES",
    "HUMAN_COLLECTION_SCHEMA_VERSION",
    "LABEL_ACCESS_EVENT_SCHEMA_VERSION",
    "LABEL_ENVELOPE_SCHEMA_VERSION",
    "SELECTION_RECEIPT_SCHEMA_VERSION",
    "TEXT_NORMALIZER_VERSION",
    "HumanCollectionError",
    "PinnedModelInputRenderer",
    "PinnedTokenCounter",
    "access_final_confirmation_label",
    "audit_population_firewall",
    "canonical_sha256",
    "prompt_population_sha256",
    "seal_label_envelope",
    "seal_prompt_record",
    "seal_selection_receipt",
    "validate_access_ledger",
    "validate_label_envelope",
    "validate_prompt_record",
    "validate_selection_receipt",
]
