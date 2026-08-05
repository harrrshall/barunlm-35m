"""Deterministic decoder-to-program-to-simulator support bridge for GVS-v1.

This module is the only CPU reference path that may derive ``action_ir_exact`` from
decoder evidence.  It re-renders the prompt, detaches and re-parses the complete K=8
trace, recomputes schema analysis, verifies the program round trip, replays context to
the bound step, and simulates every schema-valid candidate before constructing the
existing nonauthorizing support record.

Every row must supply an explicit verified schema-presentation receipt.  Identity
presentations preserve the canonical model-facing names; renamed presentations are
validated against the exact prompt schema and lowered through the name-only v2
lowerer before canonical simulation.  Unseen schemas remain unsupported and fail
closed rather than being scored under the canonical simulator registry.

The receipt deliberately does not authenticate source custody and cannot authorize a
model, label, CUDA, training, or Jarvis access.  A later one-shot authorization layer
must bind these recomputed receipts to authenticated population/source commitments.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from barunlm.datasets import mobile_actions as mobile_actions_module
from barunlm.datasets.mobile_actions import (
    PROMPT_CONTRACT_VERSION,
    PROMPT_TEMPLATE_SHA256,
    ToolArgument,
    ToolDefinition,
    render_prompt,
)

from . import action_ir as action_ir_module
from . import action_simulator as action_simulator_module
from . import gvs_decoder as gvs_decoder_module
from . import gvs_schema_presentation as gvs_schema_presentation_module
from . import gvs_support as gvs_support_module
from . import sim_program as sim_program_module
from .action_ir import (
    ActionIR,
    CallMode,
    Decision,
    ToolSchema,
    action_ir_equal,
    parse_action_ir,
)
from .action_simulator import (
    POLICY_CONTRACT_SHA256,
    SIMULATOR_SCHEMA_SHA256,
    SIMULATOR_TOOL_REGISTRY,
    SIMULATOR_TOOL_SCHEMAS,
    ActionSimulator,
    compile_program_step,
    simulate_reference_action,
    verify_program_round_trip,
)
from .gvs_decoder import (
    CandidateSetAnalysis,
    CandidateSetTrace,
    GVSDecoderError,
    analyze_gvs_trace,
    parse_candidate_set_analysis_json,
    parse_candidate_set_trace_json,
)
from .gvs_schema_presentation import (
    IDENTITY_PRESENTATION,
    RENAMED_PRESENTATION,
    GVSSchemaPresentationError,
    SchemaPresentationReceipt,
    lower_presented_action_ir,
    verify_schema_presentation,
)
from .gvs_support import (
    FROZEN_CANDIDATE_COUNT,
    GVS_CANDIDATE_SUPPORT_SCHEMA_VERSION,
    CandidateEvidence,
    CandidateSupportRecord,
    GVSSupportError,
    SupportClass,
    SupportEvaluation,
    SupportPopulation,
    SupportSampleIdentity,
    SupportStrata,
    parse_candidate_support_record,
    score_candidate_support,
)
from .sim_program import SimProgram, SimProgramError, WorldState, loads_sim_program

GVS_SUPPORT_BRIDGE_SCHEMA_VERSION = "barun-gvs-support-bridge-v3"
GVS_SOURCE_COMMITMENT_SCHEMA_VERSION = "barun-gvs-source-commitment-v2"
GVS_BRIDGED_SUPPORT_EVALUATION_SCHEMA_VERSION = "barun-gvs-bridged-support-evaluation-v3"
GVS_SUPPORT_BRIDGE_ARTIFACT_SCHEMA_VERSION = "barun-gvs-support-bridge-artifact-v3"
GVS_VERIFIED_PRESENTATION_SCOPE = "verified_identity_or_renamed_schema_presentation"
GVS_RANK_HIDDEN_VERIFIER_PAYLOAD_SCHEMA_VERSION = "barun-gvs-rank-hidden-verifier-payload-v1"
GVS_ACTION_IR_CANONICALIZATION_VERSION = "barun-gvs-lowered-semantic-action-ir-v1"
GVS_VERIFIER_TOKEN_ROW_CONTRACT_VERSION = "barun-gvs-prompt-plus-generated-unpadded-v1"
GVS_VERIFIER_BATCH_CONTRACT_VERSION = "barun-gvs-k8-right-padding-bool-mask-v1"
GVS_VERIFIER_CANDIDATE_IDENTITY_CONTRACT = (
    "lowercase_sha256_of_lowered_action_ir_semantic_canonical_v1_utf8_bytes_for_valid_candidates"
)
GVS_VERIFIER_LEARNED_INPUT_FIELDS = ("input_ids", "attention_mask")
GVS_VERIFIER_SELECTION_TIE_BREAK = "lowered_semantic_candidate_sha256_lexicographic_asc"
GVS_VERIFIER_VALID_SET_CONTRACT = (
    "complete_nontruncated_schema_valid_distinct_lowered_semantic_actions_only"
)
GVS_RANK_HIDDEN_VERIFIER_BATCH_SCHEMA_VERSION = "barun-gvs-rank-hidden-verifier-batch-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_MAX_TEXT_UTF8_BYTES = 64 * 1024
_MAX_BRIDGE_ARTIFACT_BYTES = 2 * 1024 * 1024
_MAX_JSON_INTEGER_DIGITS = 20
_DECODER_RUNTIME_HASH_HELPER = getattr(
    gvs_decoder_module,
    "_loaded_python_module_sha256",
    None,
)

_EXPECTED_OUTCOME = {
    Decision.CALL: "ACTION",
    Decision.CONFIRM: "CONFIRM",
    Decision.CLARIFY: "CLARIFY",
    Decision.ABSTAIN: "ABSTAIN",
}
_SEMANTIC_FAMILY = {
    "ADD_LIST_ITEM": "lists",
    "CREATE_CALENDAR_EVENT": "calendars",
    "CREATE_NOTE": "notes",
    "CREATE_REMINDER": "reminders",
    "LOOK_UP_CONTACT": "contacts",
    "LOOK_UP_ROUTE": "maps",
    "PAUSE_MEDIA": "media",
    "PLAY_MEDIA": "media",
    "PROPOSE_MESSAGE": "messages",
    "RESCHEDULE_CALENDAR_EVENT": "calendars",
    "SET_BOOLEAN_SETTING": "device_settings",
    "SET_LIST_ITEM_CHECKED": "lists",
    "UPDATE_REMINDER": "reminders",
}


class GVSBridgeError(ValueError):
    """Decoder, program, population, or simulator evidence failed to bind."""


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
        raise GVSBridgeError("bridge evidence is not strict JSON") from error


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def lowered_semantic_action_ir_bytes(action: ActionIR) -> bytes:
    """Canonicalize lowered Action IR under the verifier's semantic identity.

    SERIAL call order is meaningful.  PARALLEL call order is not, so canonical call
    JSON strings are sorted while retaining multiplicity.  Decisions, modes,
    missing-field order, argument types, and argument values remain exact.
    """

    if type(action) is not ActionIR:
        raise GVSBridgeError("semantic Action IR identity requires an exact ActionIR")
    if action.decision in {Decision.CALL, Decision.CONFIRM}:
        calls = tuple(action.calls)
        if action.mode is CallMode.PARALLEL:
            calls = tuple(sorted(calls, key=lambda value: value.canonical_json()))
        record: object = {
            "calls": [call.to_dict() for call in calls],
            "decision": action.decision.value,
            "mode": action.mode.value if action.mode is not None else None,
        }
    elif action.decision is Decision.CLARIFY:
        record = {"decision": action.decision.value, "missing": list(action.missing)}
    else:
        record = {"decision": action.decision.value}
    return _canonical_json(record).encode("utf-8")


def lowered_semantic_action_ir_sha256(action: ActionIR) -> str:
    """Return the exact verifier identity for one lowered canonical Action IR."""

    return hashlib.sha256(lowered_semantic_action_ir_bytes(action)).hexdigest()


def _strict_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise GVSBridgeError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise GVSBridgeError(f"{label} must be a safe identifier")
    if value != unicodedata.normalize("NFC", value):
        raise GVSBridgeError(f"{label} must be NFC-normalized")
    return value


def _strict_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise GVSBridgeError(f"{label} must be a nonempty exact string")
    if value != unicodedata.normalize("NFC", value):
        raise GVSBridgeError(f"{label} must be NFC-normalized")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise GVSBridgeError(f"{label} is not valid UTF-8") from error
    if len(encoded) > _MAX_TEXT_UTF8_BYTES:
        raise GVSBridgeError(f"{label} exceeds the bounded UTF-8 size")
    return value


def _exact_object(value: object, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GVSBridgeError(f"{label} must be an exact JSON object")
    missing = sorted(fields.difference(value))
    extra = sorted(set(value).difference(fields))
    if missing or extra:
        raise GVSBridgeError(f"{label} fields changed; missing={missing!r}, extra={extra!r}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GVSBridgeError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise GVSBridgeError(f"non-finite JSON constant {value!r}")


def _bounded_json_integer(value: str) -> int:
    digits = value.removeprefix("-")
    if not digits or len(digits) > _MAX_JSON_INTEGER_DIGITS:
        raise GVSBridgeError("JSON integer exceeds the bridge bound")
    return int(value)


def _reject_float(value: str) -> NoReturn:
    raise GVSBridgeError(f"floating-point JSON value {value!r} is forbidden")


def _file_sha256(path: Path) -> str:
    before = path.stat()
    if not path.is_file() or before.st_size > 8 * 1024 * 1024:
        raise GVSBridgeError(f"source path {path} is absent or oversized")
    value = path.read_bytes()
    after = path.stat()
    if (
        len(value) != before.st_size
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise GVSBridgeError(f"source path {path} changed while hashing")
    return hashlib.sha256(value).hexdigest()


def _module_source_sha256(module: object) -> str:
    raw = getattr(module, "__file__", None)
    if type(raw) is not str:
        raise GVSBridgeError("bridge dependency has no source path")
    return _file_sha256(Path(raw))


def _source_files_record() -> dict[str, str]:
    return {
        "action_ir": _module_source_sha256(action_ir_module),
        "action_simulator": _module_source_sha256(action_simulator_module),
        "gvs_bridge": _file_sha256(Path(__file__)),
        "gvs_decoder": _module_source_sha256(gvs_decoder_module),
        "gvs_schema_presentation": _module_source_sha256(gvs_schema_presentation_module),
        "gvs_support": _module_source_sha256(gvs_support_module),
        "mobile_actions": _module_source_sha256(mobile_actions_module),
        "sim_program": _module_source_sha256(sim_program_module),
    }


def _loaded_module_sha256(
    module: object,
    *,
    label: str,
    _helper: object = _DECODER_RUNTIME_HASH_HELPER,
) -> str:
    path = Path(getattr(module, "__file__", ""))
    namespace = getattr(module, "__dict__", None)
    if not path.is_file() or type(namespace) is not dict or not callable(_helper):
        raise GVSBridgeError(f"loaded runtime for {label} is not inspectable")
    try:
        return _helper(path=path, namespace=namespace, label=label)
    except Exception as error:
        raise GVSBridgeError(f"loaded runtime for {label} failed attestation") from error


def _bridge_loaded_sha256(*, _helper: object = _DECODER_RUNTIME_HASH_HELPER) -> str:
    if not callable(_helper):
        raise GVSBridgeError("decoder runtime attestation helper is unavailable")
    extras = {
        "ActionSimulator.simulate": ActionSimulator.simulate,
        "FROZEN_CANDIDATE_COUNT": FROZEN_CANDIDATE_COUNT,
        "GVS_CANDIDATE_SUPPORT_SCHEMA_VERSION": GVS_CANDIDATE_SUPPORT_SCHEMA_VERSION,
        "GVS_ACTION_IR_CANONICALIZATION_VERSION": (GVS_ACTION_IR_CANONICALIZATION_VERSION),
        "GVS_RANK_HIDDEN_VERIFIER_BATCH_SCHEMA_VERSION": (
            GVS_RANK_HIDDEN_VERIFIER_BATCH_SCHEMA_VERSION
        ),
        "GVS_RANK_HIDDEN_VERIFIER_PAYLOAD_SCHEMA_VERSION": (
            GVS_RANK_HIDDEN_VERIFIER_PAYLOAD_SCHEMA_VERSION
        ),
        "GVS_VERIFIER_BATCH_CONTRACT_VERSION": GVS_VERIFIER_BATCH_CONTRACT_VERSION,
        "GVS_VERIFIER_CANDIDATE_IDENTITY_CONTRACT": (GVS_VERIFIER_CANDIDATE_IDENTITY_CONTRACT),
        "GVS_VERIFIER_SELECTION_TIE_BREAK": GVS_VERIFIER_SELECTION_TIE_BREAK,
        "GVS_VERIFIER_TOKEN_ROW_CONTRACT_VERSION": (GVS_VERIFIER_TOKEN_ROW_CONTRACT_VERSION),
        "GVS_VERIFIER_VALID_SET_CONTRACT": GVS_VERIFIER_VALID_SET_CONTRACT,
        "PROMPT_CONTRACT_VERSION": PROMPT_CONTRACT_VERSION,
        "PROMPT_TEMPLATE_SHA256": PROMPT_TEMPLATE_SHA256,
        "POLICY_CONTRACT_SHA256": POLICY_CONTRACT_SHA256,
        "SIMULATOR_SCHEMA_SHA256": SIMULATOR_SCHEMA_SHA256,
        "decoder_runtime_hash_helper_import": _DECODER_RUNTIME_HASH_HELPER,
        "action_ir_equal_import": action_ir_equal,
        "analyze_gvs_trace_import": analyze_gvs_trace,
        "compile_program_step_import": compile_program_step,
        "loads_sim_program_import": loads_sim_program,
        "lower_presented_action_ir_import": lower_presented_action_ir,
        "lowered_semantic_action_ir_bytes": lowered_semantic_action_ir_bytes,
        "lowered_semantic_action_ir_sha256": lowered_semantic_action_ir_sha256,
        "materialize_rank_hidden_verifier_batch": (materialize_rank_hidden_verifier_batch),
        "parse_action_ir_import": parse_action_ir,
        "parse_candidate_set_analysis_json_import": parse_candidate_set_analysis_json,
        "parse_candidate_set_trace_json_import": parse_candidate_set_trace_json,
        "parse_candidate_support_record_import": parse_candidate_support_record,
        "render_prompt_import": render_prompt,
        "score_candidate_support_import": score_candidate_support,
        "simulate_reference_action_import": simulate_reference_action,
        "verify_schema_presentation_import": verify_schema_presentation,
        "verify_program_round_trip_import": verify_program_round_trip,
    }
    try:
        return _helper(
            path=Path(__file__),
            namespace=globals(),
            label="gvs_bridge",
            extra_callables=extras,
        )
    except Exception as error:
        raise GVSBridgeError("loaded bridge callables failed attestation") from error


def _assert_import_aliases() -> None:
    expected = {
        "ActionIR": (ActionIR, action_ir_module.ActionIR),
        "ActionSimulator": (ActionSimulator, action_simulator_module.ActionSimulator),
        "CandidateEvidence": (CandidateEvidence, gvs_support_module.CandidateEvidence),
        "CandidateSetAnalysis": (
            CandidateSetAnalysis,
            gvs_decoder_module.CandidateSetAnalysis,
        ),
        "CandidateSetTrace": (CandidateSetTrace, gvs_decoder_module.CandidateSetTrace),
        "CandidateSupportRecord": (
            CandidateSupportRecord,
            gvs_support_module.CandidateSupportRecord,
        ),
        "CallMode": (CallMode, action_ir_module.CallMode),
        "Decision": (Decision, action_ir_module.Decision),
        "GVSDecoderError": (GVSDecoderError, gvs_decoder_module.GVSDecoderError),
        "GVSSchemaPresentationError": (
            GVSSchemaPresentationError,
            gvs_schema_presentation_module.GVSSchemaPresentationError,
        ),
        "GVSSupportError": (GVSSupportError, gvs_support_module.GVSSupportError),
        "SchemaPresentationReceipt": (
            SchemaPresentationReceipt,
            gvs_schema_presentation_module.SchemaPresentationReceipt,
        ),
        "SimProgram": (SimProgram, sim_program_module.SimProgram),
        "SimProgramError": (SimProgramError, sim_program_module.SimProgramError),
        "SupportClass": (SupportClass, gvs_support_module.SupportClass),
        "SupportEvaluation": (SupportEvaluation, gvs_support_module.SupportEvaluation),
        "SupportPopulation": (SupportPopulation, gvs_support_module.SupportPopulation),
        "SupportSampleIdentity": (
            SupportSampleIdentity,
            gvs_support_module.SupportSampleIdentity,
        ),
        "SupportStrata": (SupportStrata, gvs_support_module.SupportStrata),
        "ToolArgument": (ToolArgument, mobile_actions_module.ToolArgument),
        "ToolDefinition": (ToolDefinition, mobile_actions_module.ToolDefinition),
        "ToolSchema": (ToolSchema, action_ir_module.ToolSchema),
        "WorldState": (WorldState, sim_program_module.WorldState),
        "action_ir_equal": (action_ir_equal, action_ir_module.action_ir_equal),
        "analyze_gvs_trace": (analyze_gvs_trace, gvs_decoder_module.analyze_gvs_trace),
        "compile_program_step": (
            compile_program_step,
            action_simulator_module.compile_program_step,
        ),
        "loads_sim_program": (loads_sim_program, sim_program_module.loads_sim_program),
        "lower_presented_action_ir": (
            lower_presented_action_ir,
            gvs_schema_presentation_module.lower_presented_action_ir,
        ),
        "parse_action_ir": (parse_action_ir, action_ir_module.parse_action_ir),
        "parse_candidate_set_analysis_json": (
            parse_candidate_set_analysis_json,
            gvs_decoder_module.parse_candidate_set_analysis_json,
        ),
        "parse_candidate_set_trace_json": (
            parse_candidate_set_trace_json,
            gvs_decoder_module.parse_candidate_set_trace_json,
        ),
        "parse_candidate_support_record": (
            parse_candidate_support_record,
            gvs_support_module.parse_candidate_support_record,
        ),
        "render_prompt": (render_prompt, mobile_actions_module.render_prompt),
        "score_candidate_support": (
            score_candidate_support,
            gvs_support_module.score_candidate_support,
        ),
        "simulate_reference_action": (
            simulate_reference_action,
            action_simulator_module.simulate_reference_action,
        ),
        "verify_program_round_trip": (
            verify_program_round_trip,
            action_simulator_module.verify_program_round_trip,
        ),
        "verify_schema_presentation": (
            verify_schema_presentation,
            gvs_schema_presentation_module.verify_schema_presentation,
        ),
    }
    for name, (imported, live) in expected.items():
        if imported is not live:
            raise GVSBridgeError(f"bridge imported runtime alias {name} changed identity")


def _behavior_contract_record() -> dict[str, object]:
    _assert_import_aliases()
    if SIMULATOR_TOOL_SCHEMAS is not action_simulator_module.SIMULATOR_TOOL_SCHEMAS:
        raise GVSBridgeError("bridge simulator schema tuple changed identity")
    if SIMULATOR_TOOL_REGISTRY is not action_simulator_module.SIMULATOR_TOOL_REGISTRY:
        raise GVSBridgeError("bridge simulator registry changed identity")
    if PROMPT_CONTRACT_VERSION != mobile_actions_module.PROMPT_CONTRACT_VERSION:
        raise GVSBridgeError("bridge prompt contract alias changed")
    if PROMPT_TEMPLATE_SHA256 != mobile_actions_module.PROMPT_TEMPLATE_SHA256:
        raise GVSBridgeError("bridge prompt template alias changed")
    if POLICY_CONTRACT_SHA256 != action_simulator_module.POLICY_CONTRACT_SHA256:
        raise GVSBridgeError("bridge policy contract alias changed")
    if SIMULATOR_SCHEMA_SHA256 != action_simulator_module.SIMULATOR_SCHEMA_SHA256:
        raise GVSBridgeError("bridge simulator schema hash alias changed")
    if FROZEN_CANDIDATE_COUNT != gvs_decoder_module.FROZEN_CANDIDATE_COUNT:
        raise GVSBridgeError("decoder and support K contracts differ")
    if FROZEN_CANDIDATE_COUNT != gvs_support_module.FROZEN_CANDIDATE_COUNT:
        raise GVSBridgeError("bridge and support K contracts differ")
    try:
        schema_records = [
            action_simulator_module._schema_record(schema) for schema in SIMULATOR_TOOL_SCHEMAS
        ]
    except Exception as error:
        raise GVSBridgeError("live simulator schema cannot be serialized") from error
    schema_sha256 = _sha256_json(schema_records)
    if schema_sha256 != SIMULATOR_SCHEMA_SHA256:
        raise GVSBridgeError("live simulator schema differs from its pinned hash")
    if set(SIMULATOR_TOOL_REGISTRY) != {schema.name for schema in SIMULATOR_TOOL_SCHEMAS}:
        raise GVSBridgeError("live simulator registry roster differs from its schema tuple")
    for schema in SIMULATOR_TOOL_SCHEMAS:
        if SIMULATOR_TOOL_REGISTRY[schema.name] is not schema:
            raise GVSBridgeError("live simulator registry does not retain exact schema objects")
    return {
        "bridge_contract_versions": {
            "artifact": GVS_SUPPORT_BRIDGE_ARTIFACT_SCHEMA_VERSION,
            "bridge": GVS_SUPPORT_BRIDGE_SCHEMA_VERSION,
            "evaluation": GVS_BRIDGED_SUPPORT_EVALUATION_SCHEMA_VERSION,
            "source_commitment": GVS_SOURCE_COMMITMENT_SCHEMA_VERSION,
            "rank_hidden_batch": GVS_RANK_HIDDEN_VERIFIER_BATCH_SCHEMA_VERSION,
            "rank_hidden_payload": GVS_RANK_HIDDEN_VERIFIER_PAYLOAD_SCHEMA_VERSION,
        },
        "bounds": {
            "artifact_bytes": _MAX_BRIDGE_ARTIFACT_BYTES,
            "json_integer_digits": _MAX_JSON_INTEGER_DIGITS,
            "text_utf8_bytes": _MAX_TEXT_UTF8_BYTES,
        },
        "candidate_count": FROZEN_CANDIDATE_COUNT,
        "expected_outcomes": sorted(
            (decision.value, outcome) for decision, outcome in _EXPECTED_OUTCOME.items()
        ),
        "identifier_pattern": _IDENTIFIER_RE.pattern,
        "presentation_contract_versions": {
            "artifact": gvs_schema_presentation_module.GVS_SCHEMA_PRESENTATION_ARTIFACT_VERSION,
            "presentation": gvs_schema_presentation_module.GVS_SCHEMA_PRESENTATION_VERSION,
            "receipt": (gvs_schema_presentation_module.GVS_SCHEMA_PRESENTATION_RECEIPT_VERSION),
        },
        "rank_hidden_verifier_contract": {
            "action_ir_canonicalization": GVS_ACTION_IR_CANONICALIZATION_VERSION,
            "batch": GVS_VERIFIER_BATCH_CONTRACT_VERSION,
            "candidate_identity": GVS_VERIFIER_CANDIDATE_IDENTITY_CONTRACT,
            "learned_input_fields": list(GVS_VERIFIER_LEARNED_INPUT_FIELDS),
            "selection_tie_break": GVS_VERIFIER_SELECTION_TIE_BREAK,
            "token_row": GVS_VERIFIER_TOKEN_ROW_CONTRACT_VERSION,
            "valid_set": GVS_VERIFIER_VALID_SET_CONTRACT,
        },
        "schema_scope": GVS_VERIFIED_PRESENTATION_SCOPE,
        "semantic_families": dict(sorted(_SEMANTIC_FAMILY.items())),
        "simulator_schema_runtime_sha256": schema_sha256,
    }


def _runtime_identity_record() -> dict[str, object]:
    return {
        "behavior_contract": _behavior_contract_record(),
        "bridge_loaded_sha256": _bridge_loaded_sha256(),
        "dependency_loaded_sha256s": {
            "action_ir": _loaded_module_sha256(action_ir_module, label="action_ir"),
            "action_simulator": _loaded_module_sha256(
                action_simulator_module,
                label="action_simulator",
            ),
            "gvs_decoder": _loaded_module_sha256(gvs_decoder_module, label="gvs_decoder"),
            "gvs_schema_presentation": _loaded_module_sha256(
                gvs_schema_presentation_module,
                label="gvs_schema_presentation",
            ),
            "gvs_support": _loaded_module_sha256(gvs_support_module, label="gvs_support"),
            "mobile_actions": _loaded_module_sha256(
                mobile_actions_module,
                label="mobile_actions",
            ),
            "sim_program": _loaded_module_sha256(sim_program_module, label="sim_program"),
        },
        "python_cache_tag": sys.implementation.cache_tag,
        "python_implementation": platform.python_implementation(),
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "schema_version": "barun-gvs-bridge-runtime-v1",
        "source_files_sha256": _sha256_json(_source_files_record()),
    }


def _runtime_identity_sha256() -> str:
    return _sha256_json(_runtime_identity_record())


def _assert_live_runtime() -> tuple[dict[str, str], str]:
    source_files = _source_files_record()
    source_sha256 = _sha256_json(source_files)
    runtime_sha256 = _runtime_identity_sha256()
    if source_sha256 != _INITIAL_SOURCE_FILES_SHA256:
        raise GVSBridgeError("bridge dependency source differs from the import-time baseline")
    if runtime_sha256 != _INITIAL_BRIDGE_RUNTIME_SHA256:
        raise GVSBridgeError("bridge loaded runtime differs from the import-time baseline")
    return source_files, runtime_sha256


def _tool_type_name(argument: ToolArgument) -> str:
    return argument.type_name


def _snapshot_prompt_tools_once(
    tools: tuple[ToolDefinition, ...],
) -> tuple[ToolDefinition, ...]:
    if type(tools) is not tuple or not tools:
        raise GVSBridgeError("prompt tools must be a nonempty exact tuple")
    copied: list[ToolDefinition] = []
    seen: set[str] = set()
    for index, tool in enumerate(tools):
        if type(tool) is not ToolDefinition:
            raise GVSBridgeError(f"prompt tool {index} must be an exact ToolDefinition")
        name = _strict_identifier(tool.name, label=f"prompt tool {index} name")
        description = _strict_text(tool.description, label=f"prompt tool {index} description")
        if any(unicodedata.category(character).startswith("C") for character in description):
            raise GVSBridgeError("prompt tool descriptions contain control characters")
        if name in seen:
            raise GVSBridgeError("prompt tool names must be unique")
        seen.add(name)
        if type(tool.arguments) is not tuple:
            raise GVSBridgeError("prompt tool arguments must be an exact tuple")
        arguments: list[ToolArgument] = []
        argument_names: set[str] = set()
        for argument_index, argument in enumerate(tool.arguments):
            if type(argument) is not ToolArgument:
                raise GVSBridgeError("prompt arguments must be exact ToolArgument values")
            argument_name = _strict_identifier(
                argument.name,
                label=f"prompt tool {index} argument {argument_index} name",
            )
            if argument_name in argument_names:
                raise GVSBridgeError("prompt argument names must be unique per tool")
            argument_names.add(argument_name)
            type_name = _strict_identifier(
                _tool_type_name(argument),
                label=f"prompt tool {index} argument {argument_index} type",
            )
            argument_description = _strict_text(
                argument.description,
                label=f"prompt tool {index} argument {argument_index} description",
            )
            if any(
                unicodedata.category(character).startswith("C")
                for character in argument_description
            ):
                raise GVSBridgeError("prompt argument descriptions contain control characters")
            if type(argument.required) is not bool:
                raise GVSBridgeError("prompt argument required flags must be exact booleans")
            arguments.append(
                ToolArgument(
                    name=argument_name,
                    type_name=type_name,
                    description=argument_description,
                    required=argument.required,
                )
            )
        copied.append(
            ToolDefinition(name=name, description=description, arguments=tuple(arguments))
        )
    return tuple(copied)


def _prompt_tools_record(tools: tuple[ToolDefinition, ...]) -> list[dict[str, object]]:
    return [
        {
            "arguments": [
                {
                    "description": argument.description,
                    "name": argument.name,
                    "required": argument.required,
                    "type_name": argument.type_name,
                }
                for argument in tool.arguments
            ],
            "description": tool.description,
            "name": tool.name,
        }
        for tool in tools
    ]


def _snapshot_prompt_tools(tools: tuple[ToolDefinition, ...]) -> tuple[ToolDefinition, ...]:
    first = _snapshot_prompt_tools_once(tools)
    second = _snapshot_prompt_tools_once(tools)
    if _canonical_json(_prompt_tools_record(first)) != _canonical_json(
        _prompt_tools_record(second)
    ):
        raise GVSBridgeError("prompt tools changed during the stable snapshot")
    return first


def _stable_program(program: SimProgram) -> SimProgram:
    if type(program) is not SimProgram:
        raise GVSBridgeError("program must be an exact SimProgram")
    first = program.canonical_json()
    second = program.canonical_json()
    if first != second:
        raise GVSBridgeError("program changed during the stable snapshot")
    try:
        detached = loads_sim_program(first)
    except SimProgramError as error:
        raise GVSBridgeError("program failed strict canonical detachment") from error
    if detached.canonical_json() != first:
        raise GVSBridgeError("program failed canonical detachment")
    return detached


def _stable_trace(trace: CandidateSetTrace) -> CandidateSetTrace:
    if type(trace) is not CandidateSetTrace:
        raise GVSBridgeError("trace must be exact CandidateSetTrace")
    first_bytes = trace.to_json_bytes()
    first_sha256 = trace.sha256
    first_runtime_sha256 = trace.runtime_sha256
    second_bytes = trace.to_json_bytes()
    second_sha256 = trace.sha256
    second_runtime_sha256 = trace.runtime_sha256
    if (
        type(first_bytes) is not bytes
        or first_bytes != second_bytes
        or first_sha256 != second_sha256
        or first_runtime_sha256 != second_runtime_sha256
    ):
        raise GVSBridgeError("decoder trace changed during the stable snapshot")
    try:
        return parse_candidate_set_trace_json(
            first_bytes,
            expected_trace_sha256=first_sha256,
            expected_runtime_sha256=first_runtime_sha256,
        )
    except GVSDecoderError as error:
        raise GVSBridgeError("decoder trace failed live runtime detachment") from error


def _stable_analysis(
    analysis: CandidateSetAnalysis,
    *,
    trace: CandidateSetTrace,
    schemas: tuple[ToolSchema, ...],
) -> CandidateSetAnalysis:
    if type(analysis) is not CandidateSetAnalysis:
        raise GVSBridgeError("analysis must be exact CandidateSetAnalysis")
    first_bytes = analysis.to_json_bytes()
    first_sha256 = analysis.sha256
    second_bytes = analysis.to_json_bytes()
    second_sha256 = analysis.sha256
    if (
        type(first_bytes) is not bytes
        or first_bytes != second_bytes
        or first_sha256 != second_sha256
    ):
        raise GVSBridgeError("candidate analysis changed during the stable snapshot")
    try:
        return parse_candidate_set_analysis_json(
            first_bytes,
            trace=trace,
            schemas=schemas,
            expected_analysis_sha256=first_sha256,
        )
    except GVSDecoderError as error:
        raise GVSBridgeError("candidate analysis differs from live schema recomputation") from error


_SAMPLE_FIELDS = frozenset(
    {
        "action_family",
        "component_id",
        "expected_outcome",
        "firewall_sha256",
        "population_record_sha256",
        "sample_id",
        "scan_evidence_sha256",
        "source_commitment_sha256",
        "strata",
        "task_class",
    }
)


def _support_sample_from_record(value: object) -> SupportSampleIdentity:
    row = _exact_object(value, _SAMPLE_FIELDS, label="support sample")
    try:
        task_class = SupportClass(row["task_class"])
        strata = SupportStrata.from_mapping(row["strata"])
        return SupportSampleIdentity(
            sample_id=row["sample_id"],
            firewall_sha256=row["firewall_sha256"],
            scan_evidence_sha256=row["scan_evidence_sha256"],
            population_record_sha256=row["population_record_sha256"],
            component_id=row["component_id"],
            source_commitment_sha256=row["source_commitment_sha256"],
            task_class=task_class,
            expected_outcome=row["expected_outcome"],
            action_family=row["action_family"],
            strata=strata,
        )
    except (GVSSupportError, TypeError, ValueError) as error:
        raise GVSBridgeError("support sample failed exact detachment") from error


def _stable_support_sample(sample: SupportSampleIdentity) -> SupportSampleIdentity:
    if type(sample) is not SupportSampleIdentity:
        raise GVSBridgeError("sample must be an exact SupportSampleIdentity")
    first = _canonical_json(sample.to_record())
    second = _canonical_json(sample.to_record())
    if first != second:
        raise GVSBridgeError("support sample changed during the stable snapshot")
    return _support_sample_from_record(json.loads(first))


def _stable_support_population(population: SupportPopulation) -> SupportPopulation:
    if type(population) is not SupportPopulation:
        raise GVSBridgeError("population must be an exact SupportPopulation")
    first = _canonical_json(population.to_record())
    second = _canonical_json(population.to_record())
    if first != second:
        raise GVSBridgeError("support population changed during the stable snapshot")
    row = _exact_object(
        json.loads(first),
        frozenset(
            {
                "candidate_count",
                "firewall_sha256",
                "samples",
                "scan_evidence_sha256",
                "schema_version",
            }
        ),
        label="support population",
    )
    raw_samples = row["samples"]
    if type(raw_samples) is not list:
        raise GVSBridgeError("support population samples must be an exact array")
    try:
        return SupportPopulation(
            schema_version=row["schema_version"],
            candidate_count=row["candidate_count"],
            firewall_sha256=row["firewall_sha256"],
            scan_evidence_sha256=row["scan_evidence_sha256"],
            samples=tuple(_support_sample_from_record(value) for value in raw_samples),
        )
    except (GVSSupportError, TypeError, ValueError) as error:
        raise GVSBridgeError("support population failed exact detachment") from error


def compute_gvs_source_commitment(
    *,
    program_sha256: str,
    step_id: str,
    prompt_sha256: str,
    prompt_tools_sha256: str,
    schema_presentation_sha256: str,
    schema_presentation_receipt_sha256: str,
    schema_presentation_artifact_sha256: str,
    schema_presentation_mode: str,
) -> str:
    """Commit one D-support source before any candidate outcome is visible."""

    if type(schema_presentation_mode) is not str or schema_presentation_mode not in {
        IDENTITY_PRESENTATION,
        RENAMED_PRESENTATION,
    }:
        raise GVSBridgeError("schema_presentation_mode must be identity or renamed")
    return _sha256_json(
        {
            "program_sha256": _strict_sha256(program_sha256, label="program_sha256"),
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "prompt_sha256": _strict_sha256(prompt_sha256, label="prompt_sha256"),
            "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
            "prompt_tools_sha256": _strict_sha256(prompt_tools_sha256, label="prompt_tools_sha256"),
            "schema_presentation_artifact_sha256": _strict_sha256(
                schema_presentation_artifact_sha256,
                label="schema_presentation_artifact_sha256",
            ),
            "schema_presentation_mode": schema_presentation_mode,
            "schema_presentation_receipt_sha256": _strict_sha256(
                schema_presentation_receipt_sha256,
                label="schema_presentation_receipt_sha256",
            ),
            "schema_presentation_sha256": _strict_sha256(
                schema_presentation_sha256,
                label="schema_presentation_sha256",
            ),
            "schema_version": GVS_SOURCE_COMMITMENT_SCHEMA_VERSION,
            "step_id": _strict_identifier(step_id, label="step_id"),
        }
    )


def _program_step_context(
    program: SimProgram, *, step_id: str
) -> tuple[WorldState, object, ActionIR]:
    simulator = ActionSimulator()
    state = WorldState.from_dict(program.initial_state.to_dict())
    for step in program.steps:
        action = compile_program_step(step)
        if step.step_id == step_id:
            return state, step, action
        state = simulator.simulate(action, state).state
    raise GVSBridgeError(f"step_id {step_id!r} does not occur in the bound program")


def _action_family(step: object) -> str:
    operations = getattr(step, "operations", None)
    if type(operations) is not tuple:
        raise GVSBridgeError("program step operations are unavailable")
    families = {_SEMANTIC_FAMILY[operation.kind] for operation in operations}
    if not families:
        return "none"
    if len(families) > 1:
        return "multi_family"
    return next(iter(families))


def _support_candidate_id(
    *, sample_id: str, trace_sha256: str, rank: int, decoder_candidate_id: str
) -> str:
    digest = _sha256_json(
        {
            "decoder_candidate_id": decoder_candidate_id,
            "rank": rank,
            "sample_id": sample_id,
            "trace_sha256": trace_sha256,
        }
    )
    return f"gvs-support-{rank}-{digest}"


def _strict_token_ids(value: object, *, label: str) -> tuple[int, ...]:
    if type(value) is not tuple or len(value) > 1_000_000:
        raise GVSBridgeError(f"{label} must be an exact bounded token tuple")
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise GVSBridgeError(f"{label} contains an invalid token ID")
    return value


@dataclass(frozen=True, slots=True)
class RankHiddenVerifierCandidatePayload:
    """One semantic candidate payload with no rank, origin, or likelihood feature."""

    candidate_identity_sha256: str | None
    presented_action_ir_sha256: str | None
    generated_token_ids: tuple[int, ...]
    content_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.candidate_identity_sha256 is not None:
            _strict_sha256(
                self.candidate_identity_sha256,
                label="candidate_identity_sha256",
            )
        if self.presented_action_ir_sha256 is not None:
            _strict_sha256(
                self.presented_action_ir_sha256,
                label="presented_action_ir_sha256",
            )
        if self.candidate_identity_sha256 is not None and self.presented_action_ir_sha256 is None:
            raise GVSBridgeError(
                "a valid lowered candidate identity requires presented Action IR evidence"
            )
        generated = _strict_token_ids(
            self.generated_token_ids,
            label="generated_token_ids",
        )
        content = _strict_token_ids(
            self.content_token_ids,
            label="content_token_ids",
        )
        if len(content) > len(generated):
            raise GVSBridgeError("rank-hidden candidate content exceeds generated tokens")
        if generated[: len(content)] != content:
            raise GVSBridgeError("content token IDs must be an exact generated-token prefix")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_identity_sha256": self.candidate_identity_sha256,
            "content_token_ids": list(self.content_token_ids),
            "generated_token_ids": list(self.generated_token_ids),
            "presented_action_ir_sha256": self.presented_action_ir_sha256,
        }


@dataclass(frozen=True, slots=True)
class RankHiddenVerifierPayload:
    """Permutation-invariant verifier rows and side labels without generator scores."""

    prompt_token_ids: tuple[int, ...]
    schema_presentation_receipt_sha256: str
    gold_action_ir_sha256: str
    candidates: tuple[RankHiddenVerifierCandidatePayload, ...]
    vocab_size: int
    model_max_seq_len: int
    pad_token_id: int
    learned_input_ids: tuple[tuple[int, ...], ...]
    valid_mask: tuple[bool, ...]
    exact_mask: tuple[bool, ...]
    candidate_identity_sha256s: tuple[str | None, ...]
    action_ir_canonicalization_version: str = GVS_ACTION_IR_CANONICALIZATION_VERSION
    candidate_identity_contract: str = GVS_VERIFIER_CANDIDATE_IDENTITY_CONTRACT
    selection_tie_break: str = GVS_VERIFIER_SELECTION_TIE_BREAK
    token_row_contract_version: str = GVS_VERIFIER_TOKEN_ROW_CONTRACT_VERSION
    valid_set_contract: str = GVS_VERIFIER_VALID_SET_CONTRACT
    schema_version: str = GVS_RANK_HIDDEN_VERIFIER_PAYLOAD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GVS_RANK_HIDDEN_VERIFIER_PAYLOAD_SCHEMA_VERSION:
            raise GVSBridgeError("rank-hidden verifier payload schema changed")
        if self.action_ir_canonicalization_version != GVS_ACTION_IR_CANONICALIZATION_VERSION:
            raise GVSBridgeError("rank-hidden Action IR canonicalization contract changed")
        if self.candidate_identity_contract != GVS_VERIFIER_CANDIDATE_IDENTITY_CONTRACT:
            raise GVSBridgeError("rank-hidden candidate-identity contract changed")
        if self.selection_tie_break != GVS_VERIFIER_SELECTION_TIE_BREAK:
            raise GVSBridgeError("rank-hidden selection tie-break contract changed")
        if self.token_row_contract_version != GVS_VERIFIER_TOKEN_ROW_CONTRACT_VERSION:
            raise GVSBridgeError("rank-hidden token-row contract changed")
        if self.valid_set_contract != GVS_VERIFIER_VALID_SET_CONTRACT:
            raise GVSBridgeError("rank-hidden valid-set contract changed")
        if type(self.vocab_size) is not int or self.vocab_size <= 0:
            raise GVSBridgeError("rank-hidden verifier vocab_size is invalid")
        if type(self.model_max_seq_len) is not int or self.model_max_seq_len <= 0:
            raise GVSBridgeError("rank-hidden verifier max sequence length is invalid")
        if type(self.pad_token_id) is not int or not 0 <= self.pad_token_id < self.vocab_size:
            raise GVSBridgeError("rank-hidden verifier pad token ID is invalid")
        prompt = _strict_token_ids(self.prompt_token_ids, label="prompt_token_ids")
        if not prompt:
            raise GVSBridgeError("rank-hidden verifier payload requires prompt token IDs")
        if any(token_id >= self.vocab_size for token_id in prompt):
            raise GVSBridgeError("rank-hidden prompt token exceeds vocab_size")
        if len(prompt) > self.model_max_seq_len:
            raise GVSBridgeError("rank-hidden prompt exceeds max sequence length")
        _strict_sha256(
            self.schema_presentation_receipt_sha256,
            label="schema_presentation_receipt_sha256",
        )
        _strict_sha256(self.gold_action_ir_sha256, label="gold_action_ir_sha256")
        if (
            type(self.candidates) is not tuple
            or len(self.candidates) != FROZEN_CANDIDATE_COUNT
            or any(
                type(candidate) is not RankHiddenVerifierCandidatePayload
                for candidate in self.candidates
            )
        ):
            raise GVSBridgeError("rank-hidden verifier candidates must be an exact K=8 tuple")
        for candidate in self.candidates:
            for token_id in candidate.generated_token_ids + candidate.content_token_ids:
                if token_id >= self.vocab_size:
                    raise GVSBridgeError("rank-hidden candidate token exceeds vocab_size")
            if (
                len(self.prompt_token_ids) + len(candidate.generated_token_ids)
                > self.model_max_seq_len
            ):
                raise GVSBridgeError("rank-hidden verifier token row exceeds max sequence length")
        identities = tuple(
            candidate.candidate_identity_sha256
            for candidate in self.candidates
            if candidate.candidate_identity_sha256 is not None
        )
        if len(identities) != len(set(identities)):
            raise GVSBridgeError("rank-hidden verifier identities must be unique")
        records = tuple(_canonical_json(candidate.to_record()) for candidate in self.candidates)
        if records != tuple(sorted(records)):
            raise GVSBridgeError("rank-hidden verifier slots must be in canonical evidence order")
        expected_identities = tuple(
            candidate.candidate_identity_sha256 for candidate in self.candidates
        )
        if (
            type(self.candidate_identity_sha256s) is not tuple
            or self.candidate_identity_sha256s != expected_identities
        ):
            raise GVSBridgeError("rank-hidden side identities differ from canonical slots")
        expected_valid_mask = tuple(value is not None for value in expected_identities)
        if (
            type(self.valid_mask) is not tuple
            or any(type(value) is not bool for value in self.valid_mask)
            or self.valid_mask != expected_valid_mask
        ):
            raise GVSBridgeError("rank-hidden valid_mask differs from distinct valid identities")
        expected_learned_input_ids = tuple(
            self.prompt_token_ids + candidate.generated_token_ids for candidate in self.candidates
        )
        if self.learned_input_ids != expected_learned_input_ids:
            raise GVSBridgeError(
                "rank-hidden learned input IDs are not exact prompt-plus-generated tokens"
            )
        if (
            type(self.learned_input_ids) is not tuple
            or len(self.learned_input_ids) != FROZEN_CANDIDATE_COUNT
            or any(type(row) is not tuple for row in self.learned_input_ids)
        ):
            raise GVSBridgeError("rank-hidden learned rows must be an exact K=8 tuple")
        for index, row in enumerate(self.learned_input_ids):
            _strict_token_ids(row, label=f"learned_input_ids[{index}]")
        expected_exact_mask = tuple(
            identity is not None and identity == self.gold_action_ir_sha256
            for identity in expected_identities
        )
        if (
            type(self.exact_mask) is not tuple
            or any(type(value) is not bool for value in self.exact_mask)
            or self.exact_mask != expected_exact_mask
        ):
            raise GVSBridgeError(
                "rank-hidden exact_mask is not exact valid-identity equality to gold"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "action_ir_canonicalization_version": (self.action_ir_canonicalization_version),
            "candidate_identity_contract": self.candidate_identity_contract,
            "candidates": [candidate.to_record() for candidate in self.candidates],
            "candidate_identity_sha256s": list(self.candidate_identity_sha256s),
            "exact_mask": list(self.exact_mask),
            "gold_action_ir_sha256": self.gold_action_ir_sha256,
            "learned_input_ids": [list(row) for row in self.learned_input_ids],
            "model_max_seq_len": self.model_max_seq_len,
            "pad_token_id": self.pad_token_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "schema_presentation_receipt_sha256": (self.schema_presentation_receipt_sha256),
            "schema_version": self.schema_version,
            "selection_tie_break": self.selection_tie_break,
            "token_row_contract_version": self.token_row_contract_version,
            "valid_set_contract": self.valid_set_contract,
            "valid_mask": list(self.valid_mask),
            "vocab_size": self.vocab_size,
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_record())


def _build_rank_hidden_verifier_payload(
    *,
    prompt_token_ids: tuple[int, ...],
    schema_presentation_receipt_sha256: str,
    gold_action_ir_sha256: str,
    candidates: tuple[RankHiddenVerifierCandidatePayload, ...],
    vocab_size: int,
    model_max_seq_len: int,
    pad_token_id: int,
) -> RankHiddenVerifierPayload:
    """Mask semantic duplicates and sort slots without generator-order features."""

    if type(candidates) is not tuple or len(candidates) != FROZEN_CANDIDATE_COUNT:
        raise GVSBridgeError("rank-hidden payload builder requires exact K=8 candidates")
    grouped: dict[str, list[RankHiddenVerifierCandidatePayload]] = {}
    invalid: list[RankHiddenVerifierCandidatePayload] = []
    for candidate in candidates:
        if type(candidate) is not RankHiddenVerifierCandidatePayload:
            raise GVSBridgeError("rank-hidden candidate input has the wrong type")
        if candidate.candidate_identity_sha256 is None:
            invalid.append(candidate)
        else:
            grouped.setdefault(candidate.candidate_identity_sha256, []).append(candidate)
    slots = list(invalid)
    for identity in sorted(grouped):
        rows = sorted(grouped[identity], key=lambda value: _canonical_json(value.to_record()))
        slots.append(rows[0])
        slots.extend(
            RankHiddenVerifierCandidatePayload(
                candidate_identity_sha256=None,
                presented_action_ir_sha256=row.presented_action_ir_sha256,
                generated_token_ids=row.generated_token_ids,
                content_token_ids=row.content_token_ids,
            )
            for row in rows[1:]
        )
    slots.sort(key=lambda value: _canonical_json(value.to_record()))
    learned_input_ids = tuple(
        prompt_token_ids + candidate.generated_token_ids for candidate in slots
    )
    candidate_identity_sha256s = tuple(candidate.candidate_identity_sha256 for candidate in slots)
    return RankHiddenVerifierPayload(
        prompt_token_ids=prompt_token_ids,
        schema_presentation_receipt_sha256=schema_presentation_receipt_sha256,
        gold_action_ir_sha256=gold_action_ir_sha256,
        candidates=tuple(slots),
        vocab_size=vocab_size,
        model_max_seq_len=model_max_seq_len,
        pad_token_id=pad_token_id,
        learned_input_ids=learned_input_ids,
        valid_mask=tuple(value is not None for value in candidate_identity_sha256s),
        exact_mask=tuple(
            value is not None and value == gold_action_ir_sha256
            for value in candidate_identity_sha256s
        ),
        candidate_identity_sha256s=candidate_identity_sha256s,
    )


def _rank_hidden_payload_from_record(value: object) -> RankHiddenVerifierPayload:
    row = _exact_object(
        value,
        frozenset(
            {
                "action_ir_canonicalization_version",
                "candidate_identity_contract",
                "candidate_identity_sha256s",
                "candidates",
                "exact_mask",
                "gold_action_ir_sha256",
                "learned_input_ids",
                "model_max_seq_len",
                "pad_token_id",
                "prompt_token_ids",
                "schema_presentation_receipt_sha256",
                "schema_version",
                "selection_tie_break",
                "token_row_contract_version",
                "valid_mask",
                "valid_set_contract",
                "vocab_size",
            }
        ),
        label="rank-hidden verifier payload",
    )
    raw_candidates = row["candidates"]
    if type(raw_candidates) is not list or len(raw_candidates) != FROZEN_CANDIDATE_COUNT:
        raise GVSBridgeError("rank-hidden payload candidates must be an exact K=8 array")
    candidates: list[RankHiddenVerifierCandidatePayload] = []
    for index, raw_candidate in enumerate(raw_candidates):
        candidate = _exact_object(
            raw_candidate,
            frozenset(
                {
                    "candidate_identity_sha256",
                    "content_token_ids",
                    "generated_token_ids",
                    "presented_action_ir_sha256",
                }
            ),
            label=f"rank-hidden verifier candidate {index}",
        )
        generated = candidate["generated_token_ids"]
        content = candidate["content_token_ids"]
        if type(generated) is not list or type(content) is not list:
            raise GVSBridgeError("rank-hidden verifier tokens must be exact arrays")
        candidates.append(
            RankHiddenVerifierCandidatePayload(
                candidate_identity_sha256=candidate["candidate_identity_sha256"],
                presented_action_ir_sha256=candidate["presented_action_ir_sha256"],
                generated_token_ids=tuple(generated),
                content_token_ids=tuple(content),
            )
        )
    prompt = row["prompt_token_ids"]
    learned = row["learned_input_ids"]
    identities = row["candidate_identity_sha256s"]
    valid_mask = row["valid_mask"]
    exact_mask = row["exact_mask"]
    if (
        type(prompt) is not list
        or type(learned) is not list
        or len(learned) != FROZEN_CANDIDATE_COUNT
        or any(type(value) is not list for value in learned)
        or type(identities) is not list
        or type(valid_mask) is not list
        or type(exact_mask) is not list
    ):
        raise GVSBridgeError("rank-hidden aligned payload fields must be exact arrays")
    return RankHiddenVerifierPayload(
        prompt_token_ids=tuple(prompt),
        schema_presentation_receipt_sha256=row["schema_presentation_receipt_sha256"],
        gold_action_ir_sha256=row["gold_action_ir_sha256"],
        candidates=tuple(candidates),
        vocab_size=row["vocab_size"],
        model_max_seq_len=row["model_max_seq_len"],
        pad_token_id=row["pad_token_id"],
        learned_input_ids=tuple(tuple(value) for value in learned),
        valid_mask=tuple(valid_mask),
        exact_mask=tuple(exact_mask),
        candidate_identity_sha256s=tuple(identities),
        action_ir_canonicalization_version=row["action_ir_canonicalization_version"],
        candidate_identity_contract=row["candidate_identity_contract"],
        selection_tie_break=row["selection_tie_break"],
        token_row_contract_version=row["token_row_contract_version"],
        valid_set_contract=row["valid_set_contract"],
        schema_version=row["schema_version"],
    )


@dataclass(frozen=True, slots=True)
class RankHiddenVerifierBatch:
    """Final deterministic K=8 right-padded verifier batch materialization."""

    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[bool, ...], ...]
    valid_mask: tuple[bool, ...]
    exact_mask: tuple[bool, ...]
    candidate_identity_sha256s: tuple[str | None, ...]
    schema_presentation_receipt_sha256: str
    gold_action_ir_sha256: str
    payload_sha256: str
    pad_token_id: int
    vocab_size: int
    model_max_seq_len: int
    input_id_dtype_contract: str = "int32_or_int64"
    attention_mask_dtype_contract: str = "bool"
    device_contract: str = "input_ids_and_attention_mask_same_device"
    learned_input_fields: tuple[str, ...] = GVS_VERIFIER_LEARNED_INPUT_FIELDS
    batch_contract_version: str = GVS_VERIFIER_BATCH_CONTRACT_VERSION
    schema_version: str = GVS_RANK_HIDDEN_VERIFIER_BATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GVS_RANK_HIDDEN_VERIFIER_BATCH_SCHEMA_VERSION:
            raise GVSBridgeError("rank-hidden verifier batch schema changed")
        if self.batch_contract_version != GVS_VERIFIER_BATCH_CONTRACT_VERSION:
            raise GVSBridgeError("rank-hidden verifier batch contract changed")
        if self.input_id_dtype_contract != "int32_or_int64":
            raise GVSBridgeError("rank-hidden input-ID dtype contract changed")
        if self.attention_mask_dtype_contract != "bool":
            raise GVSBridgeError("rank-hidden attention-mask dtype contract changed")
        if self.device_contract != "input_ids_and_attention_mask_same_device":
            raise GVSBridgeError("rank-hidden verifier device contract changed")
        if self.learned_input_fields != GVS_VERIFIER_LEARNED_INPUT_FIELDS:
            raise GVSBridgeError("rank-hidden learned-input field boundary changed")
        _strict_sha256(self.payload_sha256, label="payload_sha256")
        _strict_sha256(
            self.schema_presentation_receipt_sha256,
            label="schema_presentation_receipt_sha256",
        )
        _strict_sha256(self.gold_action_ir_sha256, label="gold_action_ir_sha256")
        if type(self.vocab_size) is not int or self.vocab_size <= 0:
            raise GVSBridgeError("rank-hidden batch vocab_size is invalid")
        if type(self.model_max_seq_len) is not int or self.model_max_seq_len <= 0:
            raise GVSBridgeError("rank-hidden batch max sequence length is invalid")
        if type(self.pad_token_id) is not int or not 0 <= self.pad_token_id < self.vocab_size:
            raise GVSBridgeError("rank-hidden batch pad token ID is invalid")
        if (
            type(self.input_ids) is not tuple
            or type(self.attention_mask) is not tuple
            or len(self.input_ids) != FROZEN_CANDIDATE_COUNT
            or len(self.attention_mask) != FROZEN_CANDIDATE_COUNT
        ):
            raise GVSBridgeError("rank-hidden batch must contain exact aligned K=8 rows")
        widths = {len(row) for row in self.input_ids}
        if len(widths) != 1:
            raise GVSBridgeError("rank-hidden input rows are not rectangular")
        width = next(iter(widths))
        if width <= 0 or width > self.model_max_seq_len:
            raise GVSBridgeError("rank-hidden batch width is invalid")
        for index, (row, mask) in enumerate(zip(self.input_ids, self.attention_mask, strict=True)):
            _strict_token_ids(row, label=f"input_ids[{index}]")
            if any(value >= self.vocab_size for value in row):
                raise GVSBridgeError("rank-hidden batch token exceeds vocab_size")
            if (
                type(mask) is not tuple
                or len(mask) != width
                or any(type(value) is not bool for value in mask)
            ):
                raise GVSBridgeError("rank-hidden attention mask must be exact bool KxL")
            false_seen = False
            for token_id, active in zip(row, mask, strict=True):
                false_seen = false_seen or not active
                if active and false_seen:
                    raise GVSBridgeError("rank-hidden attention mask is not right padded")
                if not active and token_id != self.pad_token_id:
                    raise GVSBridgeError("rank-hidden masked token is not the bound pad ID")
        if (
            type(self.candidate_identity_sha256s) is not tuple
            or len(self.candidate_identity_sha256s) != FROZEN_CANDIDATE_COUNT
        ):
            raise GVSBridgeError("rank-hidden batch identities must be exact K=8")
        identities = tuple(value for value in self.candidate_identity_sha256s if value is not None)
        for value in identities:
            _strict_sha256(value, label="candidate_identity_sha256")
        if len(identities) != len(set(identities)):
            raise GVSBridgeError("rank-hidden batch semantic identities are not distinct")
        expected_valid = tuple(value is not None for value in self.candidate_identity_sha256s)
        expected_exact = tuple(
            value is not None and value == self.gold_action_ir_sha256
            for value in self.candidate_identity_sha256s
        )
        for name, observed, expected in (
            ("valid_mask", self.valid_mask, expected_valid),
            ("exact_mask", self.exact_mask, expected_exact),
        ):
            if (
                type(observed) is not tuple
                or any(type(value) is not bool for value in observed)
                or observed != expected
            ):
                raise GVSBridgeError(f"rank-hidden batch {name} failed exact derivation")

    def to_record(self) -> dict[str, object]:
        return {
            "attention_mask": [list(row) for row in self.attention_mask],
            "attention_mask_dtype_contract": self.attention_mask_dtype_contract,
            "batch_contract_version": self.batch_contract_version,
            "candidate_identity_sha256s": list(self.candidate_identity_sha256s),
            "device_contract": self.device_contract,
            "exact_mask": list(self.exact_mask),
            "gold_action_ir_sha256": self.gold_action_ir_sha256,
            "input_id_dtype_contract": self.input_id_dtype_contract,
            "input_ids": [list(row) for row in self.input_ids],
            "learned_input_fields": list(self.learned_input_fields),
            "model_max_seq_len": self.model_max_seq_len,
            "pad_token_id": self.pad_token_id,
            "payload_sha256": self.payload_sha256,
            "schema_presentation_receipt_sha256": (self.schema_presentation_receipt_sha256),
            "schema_version": self.schema_version,
            "valid_mask": list(self.valid_mask),
            "vocab_size": self.vocab_size,
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_record())


def materialize_rank_hidden_verifier_batch(
    payload: RankHiddenVerifierPayload,
) -> RankHiddenVerifierBatch:
    """Right-pad exact prompt-plus-generated rows without adding rank features."""

    source_files_before, runtime_before = _assert_live_runtime()
    if type(payload) is not RankHiddenVerifierPayload:
        raise GVSBridgeError("batch materialization requires an exact verifier payload")
    width = max(len(row) for row in payload.learned_input_ids)
    input_ids = tuple(
        row + (payload.pad_token_id,) * (width - len(row)) for row in payload.learned_input_ids
    )
    attention_mask = tuple(
        (True,) * len(row) + (False,) * (width - len(row)) for row in payload.learned_input_ids
    )
    batch = RankHiddenVerifierBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        valid_mask=payload.valid_mask,
        exact_mask=payload.exact_mask,
        candidate_identity_sha256s=payload.candidate_identity_sha256s,
        schema_presentation_receipt_sha256=(payload.schema_presentation_receipt_sha256),
        gold_action_ir_sha256=payload.gold_action_ir_sha256,
        payload_sha256=payload.sha256,
        pad_token_id=payload.pad_token_id,
        vocab_size=payload.vocab_size,
        model_max_seq_len=payload.model_max_seq_len,
    )
    if type(batch) is not RankHiddenVerifierBatch:
        raise GVSBridgeError("batch materialization returned a noncanonical batch type")
    for learned, padded, mask in zip(
        payload.learned_input_ids,
        batch.input_ids,
        batch.attention_mask,
        strict=True,
    ):
        active = sum(mask)
        if active != len(learned) or padded[:active] != learned:
            raise GVSBridgeError("rank-hidden batch changed a learned token row")
    source_files_after, runtime_after = _assert_live_runtime()
    if source_files_after != source_files_before or runtime_after != runtime_before:
        raise GVSBridgeError("bridge runtime changed during batch materialization")
    return batch


@dataclass(frozen=True, slots=True)
class BridgedCandidateEvidence:
    rank: int
    decoder_candidate_id: str
    decoder_candidate_trace_sha256: str
    support_candidate_id: str
    presented_action_ir_sha256: str | None
    lowered_action_ir_sha256: str | None
    action_ir_exact: bool
    semantic_effect_exact: bool
    simulator_status: str | None
    transition_receipt_sha256: str | None
    reference_receipt_sha256: str | None
    policy_valid: bool | None
    catastrophic_unauthorized_action: bool
    noncatastrophic_fail_closed: bool

    def __post_init__(self) -> None:
        if type(self.rank) is not int or not 0 <= self.rank < FROZEN_CANDIDATE_COUNT:
            raise GVSBridgeError("bridged candidate rank is invalid")
        _strict_identifier(self.decoder_candidate_id, label="decoder_candidate_id")
        _strict_sha256(
            self.decoder_candidate_trace_sha256,
            label="decoder_candidate_trace_sha256",
        )
        _strict_identifier(self.support_candidate_id, label="support_candidate_id")
        for name in (
            "action_ir_exact",
            "semantic_effect_exact",
            "catastrophic_unauthorized_action",
            "noncatastrophic_fail_closed",
        ):
            if type(getattr(self, name)) is not bool:
                raise GVSBridgeError(f"{name} must be an exact boolean")
        simulated = self.simulator_status is not None
        if simulated is not (self.presented_action_ir_sha256 is not None):
            raise GVSBridgeError(
                "simulator status and presented Action IR hash must be jointly nullable"
            )
        if simulated is not (self.lowered_action_ir_sha256 is not None):
            raise GVSBridgeError(
                "simulator status and lowered Action IR hash must be jointly nullable"
            )
        if simulated is not (self.transition_receipt_sha256 is not None):
            raise GVSBridgeError("simulator status and transition receipt must be jointly nullable")
        if simulated is not (self.reference_receipt_sha256 is not None):
            raise GVSBridgeError("simulator status and reference receipt must be jointly nullable")
        if simulated is not (self.policy_valid is not None):
            raise GVSBridgeError("simulator status and policy validity must be jointly nullable")
        if simulated:
            _strict_sha256(
                self.presented_action_ir_sha256,
                label="presented_action_ir_sha256",
            )
            _strict_sha256(
                self.lowered_action_ir_sha256,
                label="lowered_action_ir_sha256",
            )
            _strict_identifier(self.simulator_status, label="simulator_status")
            _strict_sha256(
                self.transition_receipt_sha256,
                label="transition_receipt_sha256",
            )
            _strict_sha256(
                self.reference_receipt_sha256,
                label="reference_receipt_sha256",
            )
            if type(self.policy_valid) is not bool:
                raise GVSBridgeError("policy_valid must be an exact boolean")
        elif any(
            (
                self.action_ir_exact,
                self.semantic_effect_exact,
                self.catastrophic_unauthorized_action,
                self.noncatastrophic_fail_closed,
            )
        ):
            raise GVSBridgeError("unsimulated candidates cannot claim semantic outcomes")

    def to_record(self) -> dict[str, object]:
        return {
            "action_ir_exact": self.action_ir_exact,
            "catastrophic_unauthorized_action": self.catastrophic_unauthorized_action,
            "decoder_candidate_id": self.decoder_candidate_id,
            "decoder_candidate_trace_sha256": self.decoder_candidate_trace_sha256,
            "lowered_action_ir_sha256": self.lowered_action_ir_sha256,
            "noncatastrophic_fail_closed": self.noncatastrophic_fail_closed,
            "policy_valid": self.policy_valid,
            "presented_action_ir_sha256": self.presented_action_ir_sha256,
            "rank": self.rank,
            "reference_receipt_sha256": self.reference_receipt_sha256,
            "semantic_effect_exact": self.semantic_effect_exact,
            "simulator_status": self.simulator_status,
            "support_candidate_id": self.support_candidate_id,
            "transition_receipt_sha256": self.transition_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class CandidateSupportBridgeReceipt:
    sample_id: str
    source_commitment_sha256: str
    prompt_sha256: str
    prompt_tools_sha256: str
    candidate_set_trace_sha256: str
    candidate_set_analysis_sha256: str
    program_sha256: str
    step_id: str
    pre_step_state_sha256: str
    gold_action_ir_sha256: str
    gold_semantic_action_ir_sha256: str
    gold_transition_receipt_sha256: str
    gold_reference_receipt_sha256: str
    program_round_trip_receipt_sha256: str
    simulator_schema_sha256: str
    policy_contract_sha256: str
    source_files_sha256: str
    runtime_identity_sha256: str
    schema_presentation_id: str
    schema_presentation_mode: str
    schema_presentation_sha256: str
    schema_presentation_receipt_sha256: str
    schema_presentation_artifact_sha256: str
    rank_hidden_verifier_payload: RankHiddenVerifierPayload
    rank_hidden_verifier_payload_sha256: str
    rank_hidden_verifier_batch_sha256: str
    candidates: tuple[BridgedCandidateEvidence, ...]
    support_record: CandidateSupportRecord
    schema_version: str = GVS_SUPPORT_BRIDGE_SCHEMA_VERSION
    schema_scope: str = GVS_VERIFIED_PRESENTATION_SCOPE
    renamed_schema_supported: bool = True
    unseen_schema_supported: bool = False
    source_custody_authenticated: bool = False
    authorizes_model_or_label_access: bool = False
    authorizes_cuda_or_jarvis_access: bool = False
    launch_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != GVS_SUPPORT_BRIDGE_SCHEMA_VERSION:
            raise GVSBridgeError("unsupported support bridge receipt schema")
        _strict_identifier(self.sample_id, label="sample_id")
        _strict_identifier(self.step_id, label="step_id")
        _strict_identifier(self.schema_presentation_id, label="schema_presentation_id")
        if self.schema_presentation_mode not in {
            IDENTITY_PRESENTATION,
            RENAMED_PRESENTATION,
        }:
            raise GVSBridgeError("bridge receipt has an unsupported presentation mode")
        for name in (
            "source_commitment_sha256",
            "prompt_sha256",
            "prompt_tools_sha256",
            "candidate_set_trace_sha256",
            "candidate_set_analysis_sha256",
            "program_sha256",
            "pre_step_state_sha256",
            "gold_action_ir_sha256",
            "gold_semantic_action_ir_sha256",
            "gold_transition_receipt_sha256",
            "gold_reference_receipt_sha256",
            "program_round_trip_receipt_sha256",
            "simulator_schema_sha256",
            "policy_contract_sha256",
            "source_files_sha256",
            "runtime_identity_sha256",
            "schema_presentation_sha256",
            "schema_presentation_receipt_sha256",
            "schema_presentation_artifact_sha256",
            "rank_hidden_verifier_payload_sha256",
            "rank_hidden_verifier_batch_sha256",
        ):
            _strict_sha256(getattr(self, name), label=name)
        if self.source_files_sha256 != _INITIAL_SOURCE_FILES_SHA256:
            raise GVSBridgeError("receipt source files differ from the import-time baseline")
        if self.runtime_identity_sha256 != _INITIAL_BRIDGE_RUNTIME_SHA256:
            raise GVSBridgeError("receipt runtime differs from the import-time baseline")
        if self.simulator_schema_sha256 != SIMULATOR_SCHEMA_SHA256:
            raise GVSBridgeError("bridge simulator schema differs from the live contract")
        if self.policy_contract_sha256 != POLICY_CONTRACT_SHA256:
            raise GVSBridgeError("bridge policy contract differs from the live contract")
        if self.schema_scope != GVS_VERIFIED_PRESENTATION_SCOPE:
            raise GVSBridgeError("bridge receipt has an unverified schema scope")
        if self.renamed_schema_supported is not True:
            raise GVSBridgeError("bridge receipt must expose verified renamed-schema support")
        if self.unseen_schema_supported is not False:
            raise GVSBridgeError("bridge receipt cannot claim unseen-schema support")
        if type(self.rank_hidden_verifier_payload) is not RankHiddenVerifierPayload:
            raise GVSBridgeError("bridge receipt has an invalid rank-hidden verifier payload")
        if self.rank_hidden_verifier_payload.sha256 != self.rank_hidden_verifier_payload_sha256:
            raise GVSBridgeError("rank-hidden verifier payload hash failed recomputation")
        if (
            self.rank_hidden_verifier_payload.schema_presentation_receipt_sha256
            != self.schema_presentation_receipt_sha256
        ):
            raise GVSBridgeError(
                "rank-hidden payload and schema-presentation receipt hashes differ"
            )
        if (
            self.rank_hidden_verifier_payload.gold_action_ir_sha256
            != self.gold_semantic_action_ir_sha256
        ):
            raise GVSBridgeError("rank-hidden payload and gold semantic identities differ")
        if (
            materialize_rank_hidden_verifier_batch(self.rank_hidden_verifier_payload).sha256
            != self.rank_hidden_verifier_batch_sha256
        ):
            raise GVSBridgeError("rank-hidden verifier batch hash failed recomputation")
        if type(self.candidates) is not tuple or len(self.candidates) != FROZEN_CANDIDATE_COUNT:
            raise GVSBridgeError("bridge receipt must retain all eight candidates")
        if any(type(candidate) is not BridgedCandidateEvidence for candidate in self.candidates):
            raise GVSBridgeError("bridge receipt contains invalid candidate evidence")
        if tuple(candidate.rank for candidate in self.candidates) != tuple(
            range(FROZEN_CANDIDATE_COUNT)
        ):
            raise GVSBridgeError("bridge candidates must be in exact rank order")
        decoder_ids = tuple(candidate.decoder_candidate_id for candidate in self.candidates)
        decoder_trace_hashes = tuple(
            candidate.decoder_candidate_trace_sha256 for candidate in self.candidates
        )
        support_ids = tuple(candidate.support_candidate_id for candidate in self.candidates)
        if len(set(decoder_ids)) != FROZEN_CANDIDATE_COUNT:
            raise GVSBridgeError("bridge decoder candidate IDs must be unique")
        if len(set(decoder_trace_hashes)) != FROZEN_CANDIDATE_COUNT:
            raise GVSBridgeError("bridge decoder candidate trace hashes must be unique")
        if len(set(support_ids)) != FROZEN_CANDIDATE_COUNT:
            raise GVSBridgeError("bridge support candidate IDs must be unique")
        for candidate in self.candidates:
            expected_decoder_id = (
                f"gvs-slot-{candidate.rank}-{candidate.decoder_candidate_trace_sha256}"
            )
            if candidate.decoder_candidate_id != expected_decoder_id:
                raise GVSBridgeError("bridge decoder candidate ID failed exact derivation")
            expected_support_id = _support_candidate_id(
                sample_id=self.sample_id,
                trace_sha256=self.candidate_set_trace_sha256,
                rank=candidate.rank,
                decoder_candidate_id=candidate.decoder_candidate_id,
            )
            if candidate.support_candidate_id != expected_support_id:
                raise GVSBridgeError("bridge support candidate ID failed exact derivation")
        if type(self.support_record) is not CandidateSupportRecord:
            raise GVSBridgeError("bridge support_record must be exact CandidateSupportRecord")
        if self.support_record.sample_id != self.sample_id:
            raise GVSBridgeError("bridge and support record sample IDs differ")
        if self.support_record.source_commitment_sha256 != self.source_commitment_sha256:
            raise GVSBridgeError("bridge and support source commitments differ")
        if tuple(candidate.support_candidate_id for candidate in self.candidates) != tuple(
            candidate.candidate_id for candidate in self.support_record.candidates
        ):
            raise GVSBridgeError("bridge and support candidate IDs differ")
        if tuple(candidate.action_ir_exact for candidate in self.candidates) != tuple(
            candidate.action_ir_exact for candidate in self.support_record.candidates
        ):
            raise GVSBridgeError("bridge and support exactness flags differ")
        if tuple(candidate.presented_action_ir_sha256 for candidate in self.candidates) != tuple(
            candidate.canonical_action_sha256 for candidate in self.support_record.candidates
        ):
            raise GVSBridgeError("bridge and support presented Action IR hashes differ")
        eligible_pairs = {
            (
                bridge.lowered_action_ir_sha256,
                bridge.presented_action_ir_sha256,
            )
            for bridge, support in zip(
                self.candidates,
                self.support_record.candidates,
                strict=True,
            )
            if support.schema_valid and not support.truncated
        }
        payload_pairs = {
            (
                candidate.candidate_identity_sha256,
                candidate.presented_action_ir_sha256,
            )
            for candidate in self.rank_hidden_verifier_payload.candidates
            if candidate.candidate_identity_sha256 is not None
        }
        if None in {value for pair in eligible_pairs for value in pair}:
            raise GVSBridgeError("complete schema-valid candidates lack lowered identities")
        eligible_identities = {pair[0] for pair in eligible_pairs}
        payload_identities = {pair[0] for pair in payload_pairs}
        if payload_identities != eligible_identities or not payload_pairs.issubset(eligible_pairs):
            raise GVSBridgeError(
                "rank-hidden payload differs from complete nontruncated schema-valid "
                "distinct lowered canonical actions"
            )
        expected_exact_mask = tuple(
            identity is not None and identity == self.gold_semantic_action_ir_sha256
            for identity in self.rank_hidden_verifier_payload.candidate_identity_sha256s
        )
        if self.rank_hidden_verifier_payload.exact_mask != expected_exact_mask:
            raise GVSBridgeError("rank-hidden exact labels differ from lowered gold identity")
        exact_candidate_identities = {
            candidate.lowered_action_ir_sha256
            for candidate in self.candidates
            if candidate.action_ir_exact
        }
        payload_exact_identities = {
            identity
            for identity, is_exact in zip(
                self.rank_hidden_verifier_payload.candidate_identity_sha256s,
                self.rank_hidden_verifier_payload.exact_mask,
                strict=True,
            )
            if is_exact
        }
        if (
            exact_candidate_identities != payload_exact_identities
            or exact_candidate_identities.difference({self.gold_semantic_action_ir_sha256})
        ):
            raise GVSBridgeError("exact bridge labels differ from the gold semantic identity")
        for name in (
            "source_custody_authenticated",
            "authorizes_model_or_label_access",
            "authorizes_cuda_or_jarvis_access",
            "launch_authorized",
        ):
            if getattr(self, name) is not False:
                raise GVSBridgeError("support bridge receipts must remain nonauthorizing")

    def to_record(self) -> dict[str, object]:
        return {
            "authorizes_cuda_or_jarvis_access": self.authorizes_cuda_or_jarvis_access,
            "authorizes_model_or_label_access": self.authorizes_model_or_label_access,
            "candidate_set_analysis_sha256": self.candidate_set_analysis_sha256,
            "candidate_set_trace_sha256": self.candidate_set_trace_sha256,
            "candidates": [candidate.to_record() for candidate in self.candidates],
            "gold_action_ir_sha256": self.gold_action_ir_sha256,
            "gold_semantic_action_ir_sha256": self.gold_semantic_action_ir_sha256,
            "gold_reference_receipt_sha256": self.gold_reference_receipt_sha256,
            "gold_transition_receipt_sha256": self.gold_transition_receipt_sha256,
            "launch_authorized": self.launch_authorized,
            "policy_contract_sha256": self.policy_contract_sha256,
            "pre_step_state_sha256": self.pre_step_state_sha256,
            "program_round_trip_receipt_sha256": self.program_round_trip_receipt_sha256,
            "program_sha256": self.program_sha256,
            "prompt_sha256": self.prompt_sha256,
            "prompt_tools_sha256": self.prompt_tools_sha256,
            "rank_hidden_verifier_payload": self.rank_hidden_verifier_payload.to_record(),
            "rank_hidden_verifier_batch_sha256": (self.rank_hidden_verifier_batch_sha256),
            "rank_hidden_verifier_payload_sha256": (self.rank_hidden_verifier_payload_sha256),
            "renamed_schema_supported": self.renamed_schema_supported,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "sample_id": self.sample_id,
            "schema_scope": self.schema_scope,
            "schema_presentation_artifact_sha256": (self.schema_presentation_artifact_sha256),
            "schema_presentation_id": self.schema_presentation_id,
            "schema_presentation_mode": self.schema_presentation_mode,
            "schema_presentation_receipt_sha256": (self.schema_presentation_receipt_sha256),
            "schema_presentation_sha256": self.schema_presentation_sha256,
            "schema_version": self.schema_version,
            "simulator_schema_sha256": self.simulator_schema_sha256,
            "source_commitment_sha256": self.source_commitment_sha256,
            "source_custody_authenticated": self.source_custody_authenticated,
            "source_files_sha256": self.source_files_sha256,
            "step_id": self.step_id,
            "support_record": self.support_record.to_record(),
            "unseen_schema_supported": self.unseen_schema_supported,
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_record())

    def to_json_bytes(self) -> bytes:
        return (
            _canonical_json(
                {
                    "artifact_schema_version": GVS_SUPPORT_BRIDGE_ARTIFACT_SCHEMA_VERSION,
                    "receipt": self.to_record(),
                    "receipt_sha256": self.sha256,
                }
            )
            + "\n"
        ).encode("utf-8")


def build_candidate_support_bridge(
    *,
    sample: SupportSampleIdentity,
    trace: CandidateSetTrace,
    analysis: CandidateSetAnalysis,
    program: SimProgram,
    step_id: str,
    now: str,
    user_text: str,
    prompt_tools: tuple[ToolDefinition, ...],
    schema_presentation_receipt: SchemaPresentationReceipt,
) -> CandidateSupportBridgeReceipt:
    """Recompute one complete K=8 support record from bound source evidence."""

    source_files_before, runtime_before = _assert_live_runtime()
    sample = _stable_support_sample(sample)
    if type(trace) is not CandidateSetTrace or type(analysis) is not CandidateSetAnalysis:
        raise GVSBridgeError("trace and analysis must be exact decoder evidence")
    step_id = _strict_identifier(step_id, label="step_id")
    now = _strict_text(now, label="now")
    user_text = _strict_text(user_text, label="user_text")
    tools = _snapshot_prompt_tools(prompt_tools)
    tools_record = _prompt_tools_record(tools)
    prompt_tools_sha256 = _sha256_json(tools_record)
    if type(schema_presentation_receipt) is not SchemaPresentationReceipt:
        raise GVSBridgeError("bridge requires an exact verified SchemaPresentationReceipt")
    try:
        verified_presentation = verify_schema_presentation(
            schema_presentation_receipt,
            canonical_schemas=SIMULATOR_TOOL_SCHEMAS,
            prompt_tools=tools,
            presentation=schema_presentation_receipt.presentation,
        )
    except GVSSchemaPresentationError as error:
        raise GVSBridgeError("schema-presentation receipt failed live verification") from error
    presentation_mode = verified_presentation.presentation.mode
    strata = sample.strata.to_record()
    if strata["unseen_schema"]:
        raise GVSBridgeError("bridge does not support unseen_schema rows")
    if strata["renamed_schema"] is not (presentation_mode == RENAMED_PRESENTATION):
        raise GVSBridgeError(
            "renamed_schema stratum must exactly match the verified presentation mode"
        )
    presentation_receipt_sha256 = verified_presentation.sha256
    presentation_artifact_sha256 = hashlib.sha256(verified_presentation.to_json_bytes()).hexdigest()

    detached_program = _stable_program(program)
    if sample.sample_id != f"{detached_program.case_id}:{step_id}":
        raise GVSBridgeError("sample_id must be exactly '<case_id>:<step_id>'")
    if now != detached_program.initial_state.reference_time[:19]:
        raise GVSBridgeError("prompt NOW differs from the program's canonical local reference time")
    expected_prompt = render_prompt(now=now, tools=tools, user_text=user_text)
    if trace.prompt != expected_prompt or trace.prompt_sha256 != _sha256_text(expected_prompt):
        raise GVSBridgeError("decoder trace prompt differs from source re-rendering")

    detached_trace = _stable_trace(trace)
    if (
        detached_trace.prompt != expected_prompt
        or detached_trace.prompt_sha256 != _sha256_text(expected_prompt)
        or detached_trace.source_sha256 != detached_trace.prompt_sha256
    ):
        raise GVSBridgeError("detached decoder trace prompt differs from source re-rendering")
    detached_analysis = _stable_analysis(
        analysis,
        trace=detached_trace,
        schemas=verified_presentation.presented_schemas,
    )

    round_trip = verify_program_round_trip(detached_program)
    if round_trip.passed is not True:
        raise GVSBridgeError("bound semantic program failed exact round-trip verification")
    pre_state, step, gold_action = _program_step_context(detached_program, step_id=step_id)
    if sample.expected_outcome != _EXPECTED_OUTCOME[gold_action.decision]:
        raise GVSBridgeError("sample expected_outcome differs from the program gold decision")
    if sample.action_family != _action_family(step):
        raise GVSBridgeError("sample action_family differs from the semantic program")

    source_commitment = compute_gvs_source_commitment(
        program_sha256=detached_program.sha256(),
        step_id=step_id,
        prompt_sha256=detached_trace.prompt_sha256,
        prompt_tools_sha256=prompt_tools_sha256,
        schema_presentation_sha256=verified_presentation.presentation_sha256,
        schema_presentation_receipt_sha256=presentation_receipt_sha256,
        schema_presentation_artifact_sha256=presentation_artifact_sha256,
        schema_presentation_mode=presentation_mode,
    )
    if sample.source_commitment_sha256 != source_commitment:
        raise GVSBridgeError("population source commitment differs from program/prompt evidence")

    simulator = ActionSimulator()
    gold_visible = simulator.simulate(gold_action, pre_state)
    gold_reference = simulate_reference_action(gold_action, pre_state)
    if (
        gold_visible.receipt.status == "rejected"
        or gold_reference.receipt.status == "reference_rejected"
    ):
        raise GVSBridgeError("gold program step is not valid in its bound pre-step state")
    gold_action_sha256 = _sha256_text(gold_action.canonical_json())
    gold_semantic_action_sha256 = lowered_semantic_action_ir_sha256(gold_action)
    pre_state_sha256 = pre_state.sha256()
    if (
        gold_visible.receipt.input_state_sha256 != pre_state_sha256
        or gold_reference.receipt.input_state_sha256 != pre_state_sha256
        or gold_visible.receipt.action_ir_sha256 != gold_action_sha256
        or gold_reference.receipt.source_sha256 != gold_action_sha256
    ):
        raise GVSBridgeError("gold simulator receipts differ from the bound action/state")
    round_trip_rows = tuple(
        record for record in round_trip.step_records if record["step_id"] == step_id
    )
    if len(round_trip_rows) != 1:
        raise GVSBridgeError("round-trip receipt does not contain the bound step exactly once")
    round_trip_row = round_trip_rows[0]
    if (
        round_trip_row["action_ir_sha256"] != gold_action_sha256
        or round_trip_row["visible_receipt_sha256"] != gold_visible.receipt.sha256()
        or round_trip_row["compiled_reference_receipt_sha256"] != gold_reference.receipt.sha256()
    ):
        raise GVSBridgeError("gold simulator receipts differ from the program round trip")

    support_candidates: list[CandidateEvidence] = []
    bridge_candidates: list[BridgedCandidateEvidence] = []
    rank_hidden_candidates: list[RankHiddenVerifierCandidatePayload] = []
    for trace_candidate, candidate in zip(
        detached_trace.candidates,
        detached_analysis.candidates,
        strict=True,
    ):
        if (
            trace_candidate.attempted_rank != candidate.attempted_rank
            or trace_candidate.candidate_id != candidate.candidate_id
            or trace_candidate.sha256 != candidate.candidate_trace_sha256
        ):
            raise GVSBridgeError("decoder trace and analysis candidate cross-links differ")
        support_candidate_id = _support_candidate_id(
            sample_id=sample.sample_id,
            trace_sha256=detached_trace.sha256,
            rank=candidate.attempted_rank,
            decoder_candidate_id=candidate.candidate_id,
        )
        action_exact = False
        effect_exact = False
        status: str | None = None
        transition_sha: str | None = None
        reference_sha: str | None = None
        policy_valid: bool | None = None
        catastrophic = False
        fail_closed = False
        presented_action_sha256: str | None = None
        lowered_action_sha256: str | None = None
        if candidate.schema_valid:
            assert candidate.canonical_action_json is not None
            presented_action = parse_action_ir(
                candidate.canonical_action_json,
                verified_presentation.presented_schemas,
            )
            presented_action_sha256 = _sha256_text(presented_action.canonical_json())
            if presented_action_sha256 != candidate.canonical_action_sha256:
                raise GVSBridgeError("candidate presented Action IR hash failed live parsing")
            try:
                lowered_action = lower_presented_action_ir(
                    presented_action,
                    receipt=verified_presentation,
                    canonical_schemas=SIMULATOR_TOOL_SCHEMAS,
                    prompt_tools=tools,
                    presentation=verified_presentation.presentation,
                )
            except GVSSchemaPresentationError as error:
                raise GVSBridgeError(
                    "schema-valid presented Action IR failed verified name-only lowering"
                ) from error
            parsed = parse_action_ir(
                lowered_action.canonical_json(),
                SIMULATOR_TOOL_REGISTRY,
            )
            lowered_action_sha256 = lowered_semantic_action_ir_sha256(parsed)
            action_exact = lowered_action_sha256 == gold_semantic_action_sha256
            if action_exact is not action_ir_equal(parsed, gold_action):
                raise GVSBridgeError("lowered semantic identity disagrees with Action IR equality")
            simulator_action_sha256 = _sha256_text(parsed.canonical_json())
            visible = simulator.simulate(parsed, pre_state)
            reference = simulate_reference_action(parsed, pre_state)
            status = visible.receipt.status
            transition_sha = visible.receipt.sha256()
            reference_sha = reference.receipt.sha256()
            policy_valid = visible.receipt.policy["policy_valid"]
            if type(policy_valid) is not bool:
                raise GVSBridgeError("simulator policy validity is not an exact boolean")
            effect_exact = (
                reference.receipt.status == gold_reference.receipt.status
                and reference.counterfactual_state.sha256()
                == gold_reference.counterfactual_state.sha256()
                and reference.receipt.effects_sha256 == gold_reference.receipt.effects_sha256
                and reference.receipt.observation_sha256
                == gold_reference.receipt.observation_sha256
            )
            if (
                visible.receipt.input_state_sha256 != pre_state_sha256
                or reference.receipt.input_state_sha256 != pre_state_sha256
                or visible.receipt.action_ir_sha256 != simulator_action_sha256
                or reference.receipt.source_sha256 != simulator_action_sha256
            ):
                raise GVSBridgeError("candidate simulator receipts differ from bound inputs")
            if action_exact and (
                not effect_exact
                or visible.receipt.sha256() != gold_visible.receipt.sha256()
                or reference.receipt.sha256() != gold_reference.receipt.sha256()
            ):
                raise GVSBridgeError(
                    "exact candidate did not reproduce deterministic gold simulator receipts"
                )
            catastrophic = parsed.decision is Decision.CALL and any(
                SIMULATOR_TOOL_REGISTRY[call.tool].side_effecting for call in parsed.calls
            )
            fail_closed = (
                policy_valid
                and visible.receipt.status in {"no_action", "confirmation_required"}
                and not catastrophic
            )
            if catastrophic and (
                policy_valid is not False or visible.receipt.status != "policy_blocked"
            ):
                raise GVSBridgeError("catastrophic candidate did not fail closed in policy")
        support_candidates.append(
            CandidateEvidence(
                rank=candidate.attempted_rank,
                candidate_id=support_candidate_id,
                output_present=candidate.output_present,
                raw_output_sha256=candidate.raw_output_sha256,
                canonical_action_sha256=candidate.canonical_action_sha256,
                parse_valid=candidate.parse_valid,
                schema_valid=candidate.schema_valid,
                action_ir_exact=action_exact,
                truncated=candidate.truncated,
            )
        )
        bridge_candidates.append(
            BridgedCandidateEvidence(
                rank=candidate.attempted_rank,
                decoder_candidate_id=candidate.candidate_id,
                decoder_candidate_trace_sha256=candidate.candidate_trace_sha256,
                support_candidate_id=support_candidate_id,
                presented_action_ir_sha256=presented_action_sha256,
                lowered_action_ir_sha256=lowered_action_sha256,
                action_ir_exact=action_exact,
                semantic_effect_exact=effect_exact,
                simulator_status=status,
                transition_receipt_sha256=transition_sha,
                reference_receipt_sha256=reference_sha,
                policy_valid=policy_valid,
                catastrophic_unauthorized_action=catastrophic,
                noncatastrophic_fail_closed=fail_closed,
            )
        )
        rank_hidden_candidates.append(
            RankHiddenVerifierCandidatePayload(
                candidate_identity_sha256=(
                    lowered_action_sha256
                    if candidate.schema_valid and not candidate.truncated
                    else None
                ),
                presented_action_ir_sha256=presented_action_sha256,
                generated_token_ids=trace_candidate.generated_token_ids,
                content_token_ids=trace_candidate.content_token_ids,
            )
        )

    support_record = CandidateSupportRecord(
        schema_version=GVS_CANDIDATE_SUPPORT_SCHEMA_VERSION,
        sample_id=sample.sample_id,
        firewall_sha256=sample.firewall_sha256,
        scan_evidence_sha256=sample.scan_evidence_sha256,
        population_record_sha256=sample.population_record_sha256,
        component_id=sample.component_id,
        source_commitment_sha256=sample.source_commitment_sha256,
        task_class=sample.task_class,
        expected_outcome=sample.expected_outcome,
        action_family=sample.action_family,
        strata=sample.strata,
        candidate_count=FROZEN_CANDIDATE_COUNT,
        candidates=tuple(support_candidates),
    )
    rank_hidden_payload = _build_rank_hidden_verifier_payload(
        prompt_token_ids=detached_trace.prompt_token_ids,
        schema_presentation_receipt_sha256=presentation_receipt_sha256,
        gold_action_ir_sha256=gold_semantic_action_sha256,
        candidates=tuple(rank_hidden_candidates),
        vocab_size=detached_trace.vocab_size,
        model_max_seq_len=detached_trace.model_max_seq_len,
        pad_token_id=detached_trace.pad_token_id,
    )
    rank_hidden_batch = materialize_rank_hidden_verifier_batch(rank_hidden_payload)
    source_files_after, runtime_after = _assert_live_runtime()
    if source_files_after != source_files_before or runtime_after != runtime_before:
        raise GVSBridgeError("bridge dependency source changed during recomputation")
    return CandidateSupportBridgeReceipt(
        sample_id=sample.sample_id,
        source_commitment_sha256=source_commitment,
        prompt_sha256=detached_trace.prompt_sha256,
        prompt_tools_sha256=prompt_tools_sha256,
        candidate_set_trace_sha256=detached_trace.sha256,
        candidate_set_analysis_sha256=detached_analysis.sha256,
        program_sha256=detached_program.sha256(),
        step_id=step_id,
        pre_step_state_sha256=pre_state_sha256,
        gold_action_ir_sha256=gold_action_sha256,
        gold_semantic_action_ir_sha256=gold_semantic_action_sha256,
        gold_transition_receipt_sha256=gold_visible.receipt.sha256(),
        gold_reference_receipt_sha256=gold_reference.receipt.sha256(),
        program_round_trip_receipt_sha256=round_trip.sha256(),
        simulator_schema_sha256=SIMULATOR_SCHEMA_SHA256,
        policy_contract_sha256=POLICY_CONTRACT_SHA256,
        source_files_sha256=_sha256_json(source_files_before),
        runtime_identity_sha256=runtime_before,
        schema_presentation_id=verified_presentation.presentation.presentation_id,
        schema_presentation_mode=presentation_mode,
        schema_presentation_sha256=verified_presentation.presentation_sha256,
        schema_presentation_receipt_sha256=presentation_receipt_sha256,
        schema_presentation_artifact_sha256=presentation_artifact_sha256,
        rank_hidden_verifier_payload=rank_hidden_payload,
        rank_hidden_verifier_payload_sha256=rank_hidden_payload.sha256,
        rank_hidden_verifier_batch_sha256=rank_hidden_batch.sha256,
        candidates=tuple(bridge_candidates),
        support_record=support_record,
    )


def verify_candidate_support_bridge(
    receipt: CandidateSupportBridgeReceipt,
    *,
    sample: SupportSampleIdentity,
    trace: CandidateSetTrace,
    analysis: CandidateSetAnalysis,
    program: SimProgram,
    step_id: str,
    now: str,
    user_text: str,
    prompt_tools: tuple[ToolDefinition, ...],
    schema_presentation_receipt: SchemaPresentationReceipt,
) -> CandidateSupportBridgeReceipt:
    """Recompute every outcome from source evidence and compare the complete receipt."""

    source_files_before, runtime_before = _assert_live_runtime()
    if type(receipt) is not CandidateSupportBridgeReceipt:
        raise GVSBridgeError("receipt must be an exact CandidateSupportBridgeReceipt")
    first_receipt_record = _canonical_json(receipt.to_record())
    second_receipt_record = _canonical_json(receipt.to_record())
    if first_receipt_record != second_receipt_record:
        raise GVSBridgeError("support bridge receipt changed during the stable snapshot")
    expected = build_candidate_support_bridge(
        sample=sample,
        trace=trace,
        analysis=analysis,
        program=program,
        step_id=step_id,
        now=now,
        user_text=user_text,
        prompt_tools=prompt_tools,
        schema_presentation_receipt=schema_presentation_receipt,
    )
    if first_receipt_record != _canonical_json(expected.to_record()):
        raise GVSBridgeError("support bridge receipt differs from complete live recomputation")
    if first_receipt_record != _canonical_json(receipt.to_record()):
        raise GVSBridgeError("support bridge receipt changed during live recomputation")
    source_files_after, runtime_after = _assert_live_runtime()
    if source_files_after != source_files_before or runtime_after != runtime_before:
        raise GVSBridgeError("bridge runtime changed during receipt verification")
    return expected


def load_candidate_support_bridge_json(
    raw: bytes,
    *,
    expected_receipt_sha256: str,
    sample: SupportSampleIdentity,
    trace: CandidateSetTrace,
    analysis: CandidateSetAnalysis,
    program: SimProgram,
    step_id: str,
    now: str,
    user_text: str,
    prompt_tools: tuple[ToolDefinition, ...],
    schema_presentation_receipt: SchemaPresentationReceipt,
) -> CandidateSupportBridgeReceipt:
    """Load one strict bounded artifact and completely rederive its live receipt."""

    source_files_before, runtime_before = _assert_live_runtime()
    expected_receipt_sha256 = _strict_sha256(
        expected_receipt_sha256,
        label="expected_receipt_sha256",
    )
    if type(raw) is not bytes or not raw or len(raw) > _MAX_BRIDGE_ARTIFACT_BYTES:
        raise GVSBridgeError("bridge artifact bytes are empty, non-bytes, or oversized")
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_bounded_json_integer,
        )
    except GVSBridgeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise GVSBridgeError("bridge artifact is not strict bounded UTF-8 JSON") from error
    artifact = _exact_object(
        payload,
        frozenset({"artifact_schema_version", "receipt", "receipt_sha256"}),
        label="bridge artifact",
    )
    if artifact["artifact_schema_version"] != GVS_SUPPORT_BRIDGE_ARTIFACT_SCHEMA_VERSION:
        raise GVSBridgeError("unsupported bridge artifact schema")
    row = _exact_object(
        artifact["receipt"],
        frozenset(
            {
                "authorizes_cuda_or_jarvis_access",
                "authorizes_model_or_label_access",
                "candidate_set_analysis_sha256",
                "candidate_set_trace_sha256",
                "candidates",
                "gold_action_ir_sha256",
                "gold_semantic_action_ir_sha256",
                "gold_reference_receipt_sha256",
                "gold_transition_receipt_sha256",
                "launch_authorized",
                "policy_contract_sha256",
                "pre_step_state_sha256",
                "program_round_trip_receipt_sha256",
                "program_sha256",
                "prompt_sha256",
                "prompt_tools_sha256",
                "rank_hidden_verifier_payload",
                "rank_hidden_verifier_batch_sha256",
                "rank_hidden_verifier_payload_sha256",
                "renamed_schema_supported",
                "runtime_identity_sha256",
                "sample_id",
                "schema_scope",
                "schema_presentation_artifact_sha256",
                "schema_presentation_id",
                "schema_presentation_mode",
                "schema_presentation_receipt_sha256",
                "schema_presentation_sha256",
                "schema_version",
                "simulator_schema_sha256",
                "source_commitment_sha256",
                "source_custody_authenticated",
                "source_files_sha256",
                "step_id",
                "support_record",
                "unseen_schema_supported",
            }
        ),
        label="bridge receipt",
    )
    raw_candidates = row["candidates"]
    if type(raw_candidates) is not list or len(raw_candidates) != FROZEN_CANDIDATE_COUNT:
        raise GVSBridgeError("bridge candidates must be an exact K=8 array")
    candidate_fields = frozenset(
        {
            "action_ir_exact",
            "catastrophic_unauthorized_action",
            "decoder_candidate_id",
            "decoder_candidate_trace_sha256",
            "lowered_action_ir_sha256",
            "noncatastrophic_fail_closed",
            "policy_valid",
            "presented_action_ir_sha256",
            "rank",
            "reference_receipt_sha256",
            "semantic_effect_exact",
            "simulator_status",
            "support_candidate_id",
            "transition_receipt_sha256",
        }
    )
    candidates: list[BridgedCandidateEvidence] = []
    for index, value in enumerate(raw_candidates):
        candidate = _exact_object(
            value,
            candidate_fields,
            label=f"bridge candidate {index}",
        )
        candidates.append(BridgedCandidateEvidence(**candidate))
    try:
        support_record = parse_candidate_support_record(row["support_record"])
    except GVSSupportError as error:
        raise GVSBridgeError(f"bridge support record is invalid: {error}") from error
    rank_hidden_payload = _rank_hidden_payload_from_record(row["rank_hidden_verifier_payload"])
    receipt = CandidateSupportBridgeReceipt(
        sample_id=row["sample_id"],
        source_commitment_sha256=row["source_commitment_sha256"],
        prompt_sha256=row["prompt_sha256"],
        prompt_tools_sha256=row["prompt_tools_sha256"],
        candidate_set_trace_sha256=row["candidate_set_trace_sha256"],
        candidate_set_analysis_sha256=row["candidate_set_analysis_sha256"],
        program_sha256=row["program_sha256"],
        step_id=row["step_id"],
        pre_step_state_sha256=row["pre_step_state_sha256"],
        gold_action_ir_sha256=row["gold_action_ir_sha256"],
        gold_semantic_action_ir_sha256=row["gold_semantic_action_ir_sha256"],
        gold_transition_receipt_sha256=row["gold_transition_receipt_sha256"],
        gold_reference_receipt_sha256=row["gold_reference_receipt_sha256"],
        program_round_trip_receipt_sha256=row["program_round_trip_receipt_sha256"],
        simulator_schema_sha256=row["simulator_schema_sha256"],
        policy_contract_sha256=row["policy_contract_sha256"],
        source_files_sha256=row["source_files_sha256"],
        runtime_identity_sha256=row["runtime_identity_sha256"],
        schema_presentation_id=row["schema_presentation_id"],
        schema_presentation_mode=row["schema_presentation_mode"],
        schema_presentation_sha256=row["schema_presentation_sha256"],
        schema_presentation_receipt_sha256=row["schema_presentation_receipt_sha256"],
        schema_presentation_artifact_sha256=row["schema_presentation_artifact_sha256"],
        rank_hidden_verifier_payload=rank_hidden_payload,
        rank_hidden_verifier_payload_sha256=row["rank_hidden_verifier_payload_sha256"],
        rank_hidden_verifier_batch_sha256=row["rank_hidden_verifier_batch_sha256"],
        candidates=tuple(candidates),
        support_record=support_record,
        schema_version=row["schema_version"],
        schema_scope=row["schema_scope"],
        renamed_schema_supported=row["renamed_schema_supported"],
        unseen_schema_supported=row["unseen_schema_supported"],
        source_custody_authenticated=row["source_custody_authenticated"],
        authorizes_model_or_label_access=row["authorizes_model_or_label_access"],
        authorizes_cuda_or_jarvis_access=row["authorizes_cuda_or_jarvis_access"],
        launch_authorized=row["launch_authorized"],
    )
    if artifact["receipt_sha256"] != receipt.sha256:
        raise GVSBridgeError("embedded bridge receipt hash failed recomputation")
    if receipt.sha256 != expected_receipt_sha256:
        raise GVSBridgeError("bridge receipt differs from the caller commitment")
    source_files_after, runtime_after = _assert_live_runtime()
    if source_files_after != source_files_before or runtime_after != runtime_before:
        raise GVSBridgeError("bridge runtime changed during artifact loading")
    return verify_candidate_support_bridge(
        receipt,
        sample=sample,
        trace=trace,
        analysis=analysis,
        program=program,
        step_id=step_id,
        now=now,
        user_text=user_text,
        prompt_tools=prompt_tools,
        schema_presentation_receipt=schema_presentation_receipt,
    )


@dataclass(frozen=True, slots=True)
class CandidateSupportBridgeInput:
    """Live source evidence required to recompute one receipt at score time."""

    receipt: CandidateSupportBridgeReceipt
    sample: SupportSampleIdentity
    trace: CandidateSetTrace
    analysis: CandidateSetAnalysis
    program: SimProgram
    step_id: str
    now: str
    user_text: str
    prompt_tools: tuple[ToolDefinition, ...]
    schema_presentation_receipt: SchemaPresentationReceipt

    def __post_init__(self) -> None:
        if type(self.receipt) is not CandidateSupportBridgeReceipt:
            raise GVSBridgeError("bridge input receipt has the wrong type")
        if type(self.sample) is not SupportSampleIdentity:
            raise GVSBridgeError("bridge input sample has the wrong type")
        if self.receipt.sample_id != self.sample.sample_id:
            raise GVSBridgeError("bridge input receipt and sample IDs differ")
        if (
            type(self.trace) is not CandidateSetTrace
            or type(self.analysis) is not CandidateSetAnalysis
        ):
            raise GVSBridgeError("bridge input decoder evidence has the wrong type")
        if type(self.program) is not SimProgram:
            raise GVSBridgeError("bridge input program has the wrong type")
        _strict_identifier(self.step_id, label="bridge input step_id")
        _strict_text(self.now, label="bridge input now")
        _strict_text(self.user_text, label="bridge input user_text")
        _snapshot_prompt_tools(self.prompt_tools)
        if type(self.schema_presentation_receipt) is not SchemaPresentationReceipt:
            raise GVSBridgeError("bridge input schema presentation has the wrong type")
        if (
            self.receipt.schema_presentation_receipt_sha256
            != self.schema_presentation_receipt.sha256
        ):
            raise GVSBridgeError("bridge input schema-presentation receipt hash differs")


@dataclass(frozen=True, slots=True)
class BridgedSupportEvaluation:
    """Support score that proves every row was rederived through the live bridge."""

    evaluation: SupportEvaluation
    bridge_receipt_sha256s: tuple[str, ...]
    bridge_receipt_root_sha256: str
    schema_version: str = GVS_BRIDGED_SUPPORT_EVALUATION_SCHEMA_VERSION
    source_custody_authenticated: bool = False
    launch_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != GVS_BRIDGED_SUPPORT_EVALUATION_SCHEMA_VERSION:
            raise GVSBridgeError("unsupported bridged support evaluation schema")
        if type(self.evaluation) is not SupportEvaluation:
            raise GVSBridgeError("bridged evaluation must contain an exact SupportEvaluation")
        if type(self.bridge_receipt_sha256s) is not tuple or not self.bridge_receipt_sha256s:
            raise GVSBridgeError("bridged evaluation requires receipt hashes")
        for value in self.bridge_receipt_sha256s:
            _strict_sha256(value, label="bridge receipt hash")
        if self.bridge_receipt_sha256s != tuple(sorted(self.bridge_receipt_sha256s)):
            raise GVSBridgeError("bridge receipt hashes must be in canonical order")
        if len(set(self.bridge_receipt_sha256s)) != len(self.bridge_receipt_sha256s):
            raise GVSBridgeError("bridge receipt hashes must be unique")
        expected_root = _sha256_json(list(self.bridge_receipt_sha256s))
        if self.bridge_receipt_root_sha256 != expected_root:
            raise GVSBridgeError("bridge receipt root hash failed recomputation")
        if self.source_custody_authenticated is not False or self.launch_authorized is not False:
            raise GVSBridgeError("bridged support evaluation must remain nonauthorizing")

    def to_record(self) -> dict[str, object]:
        return {
            "bridge_receipt_root_sha256": self.bridge_receipt_root_sha256,
            "bridge_receipt_sha256s": list(self.bridge_receipt_sha256s),
            "evaluation": self.evaluation.to_record(),
            "launch_authorized": self.launch_authorized,
            "schema_version": self.schema_version,
            "source_custody_authenticated": self.source_custody_authenticated,
        }


def score_bridged_candidate_support(
    population: SupportPopulation,
    bridge_inputs: Sequence[CandidateSupportBridgeInput],
    *,
    population_records: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    firewall_manifest: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> BridgedSupportEvaluation:
    """Recompute every bridge row, then invoke the firewall-rederiving scorer."""

    source_files_before, runtime_before = _assert_live_runtime()
    population = _stable_support_population(population)
    if type(bridge_inputs) not in (list, tuple):
        raise GVSBridgeError("bridge_inputs must be an exact list or tuple")
    inputs = tuple(bridge_inputs)
    second_inputs = tuple(bridge_inputs)
    if tuple(id(value) for value in inputs) != tuple(id(value) for value in second_inputs):
        raise GVSBridgeError("bridge_inputs changed during the stable snapshot")
    if not inputs or len(inputs) != len(population.samples):
        raise GVSBridgeError("bridge input count differs from the complete support population")
    if any(type(value) is not CandidateSupportBridgeInput for value in inputs):
        raise GVSBridgeError("bridge_inputs contain an invalid entry")
    expected_by_id = {sample.sample_id: sample for sample in population.samples}
    input_by_id: dict[
        str,
        tuple[CandidateSupportBridgeInput, SupportSampleIdentity],
    ] = {}
    for value in inputs:
        detached_sample = _stable_support_sample(value.sample)
        sample_id = detached_sample.sample_id
        if sample_id in input_by_id:
            raise GVSBridgeError("bridge_inputs contain duplicate sample IDs")
        if value.receipt.sample_id != sample_id:
            raise GVSBridgeError("bridge input receipt and detached sample IDs differ")
        if expected_by_id.get(sample_id) != detached_sample:
            raise GVSBridgeError("bridge input sample differs from the support population")
        input_by_id[sample_id] = (value, detached_sample)
    if set(input_by_id) != set(expected_by_id):
        raise GVSBridgeError("bridge input membership differs from the support population")

    records: list[CandidateSupportRecord] = []
    receipt_hashes: list[str] = []
    for sample_id in sorted(input_by_id):
        value, detached_sample = input_by_id[sample_id]
        verified = verify_candidate_support_bridge(
            value.receipt,
            sample=detached_sample,
            trace=value.trace,
            analysis=value.analysis,
            program=value.program,
            step_id=value.step_id,
            now=value.now,
            user_text=value.user_text,
            prompt_tools=value.prompt_tools,
            schema_presentation_receipt=value.schema_presentation_receipt,
        )
        records.append(verified.support_record)
        receipt_hashes.append(verified.sha256)
    evaluation = score_candidate_support(
        population,
        records,
        population_records=population_records,
        expected_membership=expected_membership,
        expected_scan_bindings=expected_scan_bindings,
        scan_evidence=scan_evidence,
        firewall_manifest=firewall_manifest,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
        firewall_frozen_at_utc=firewall_frozen_at_utc,
    )
    source_files_after, runtime_after = _assert_live_runtime()
    if source_files_after != source_files_before or runtime_after != runtime_before:
        raise GVSBridgeError("bridge runtime changed during support scoring")
    ordered_hashes = tuple(sorted(receipt_hashes))
    return BridgedSupportEvaluation(
        evaluation=evaluation,
        bridge_receipt_sha256s=ordered_hashes,
        bridge_receipt_root_sha256=_sha256_json(list(ordered_hashes)),
    )


__all__ = [
    "GVS_ACTION_IR_CANONICALIZATION_VERSION",
    "GVS_RANK_HIDDEN_VERIFIER_BATCH_SCHEMA_VERSION",
    "GVS_RANK_HIDDEN_VERIFIER_PAYLOAD_SCHEMA_VERSION",
    "GVS_SOURCE_COMMITMENT_SCHEMA_VERSION",
    "GVS_SUPPORT_BRIDGE_SCHEMA_VERSION",
    "GVS_VERIFIED_PRESENTATION_SCOPE",
    "GVS_VERIFIER_BATCH_CONTRACT_VERSION",
    "GVS_VERIFIER_CANDIDATE_IDENTITY_CONTRACT",
    "GVS_VERIFIER_LEARNED_INPUT_FIELDS",
    "GVS_VERIFIER_SELECTION_TIE_BREAK",
    "GVS_VERIFIER_TOKEN_ROW_CONTRACT_VERSION",
    "GVS_VERIFIER_VALID_SET_CONTRACT",
    "BridgedCandidateEvidence",
    "BridgedSupportEvaluation",
    "CandidateSupportBridgeInput",
    "CandidateSupportBridgeReceipt",
    "GVSBridgeError",
    "RankHiddenVerifierBatch",
    "RankHiddenVerifierCandidatePayload",
    "RankHiddenVerifierPayload",
    "build_candidate_support_bridge",
    "compute_gvs_source_commitment",
    "load_candidate_support_bridge_json",
    "lowered_semantic_action_ir_bytes",
    "lowered_semantic_action_ir_sha256",
    "materialize_rank_hidden_verifier_batch",
    "score_bridged_candidate_support",
    "verify_candidate_support_bridge",
]


_INITIAL_SOURCE_FILES_SHA256 = _sha256_json(_source_files_record())
_INITIAL_BRIDGE_RUNTIME_SHA256 = _runtime_identity_sha256()
