"""CPU-only raw-draft plumbing for the nonauthorizing Generate-Correct proposal.

This module is deliberately smaller than a generation or training runtime.  It has no
model, tokenizer, Torch, CUDA, network, dataset, JarvisLabs, or external-execution
surface.  It does three label-free things:

* preserve one generated draft exactly as a JSON *string* named ``draft_raw``;
* derive strict Action IR, policy, and sandbox-simulator diagnostics from that exact
  string and the state already present in a canonical forge prompt; and
* derive (rather than accept) the validity evidence used by the frozen pass-2, pass-1,
  then ``ABSTAIN`` fallback.

Nothing in this file authorizes population materialization, model access, training, or
launch.  A later experiment would still need a distinct frozen contract and audit.

The integrity wrapper below detects ordinary same-process rebinding of this module's
checked globals, dependency callables, constants, and source bytes.  It is tamper
evidence for a controlled worker, not a security boundary against hostile same-UID
code, closure introspection, replacement of the public API object itself, or operating-
system compromise.

On CPython 3.10/3.11, the contract's standard-library ``pathlib`` import may load
``urllib.parse``.  That parser-only module is allowed; ``urllib.request``, HTTP clients,
socket modules, subprocesses, and network audit events are not.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from . import action_correction_contract as _contract_module
from . import action_ir as _action_ir_module
from . import action_simulator as _action_simulator_module
from . import sim_program as _sim_program_module
from .action_correction_contract import (
    CANONICAL_ABSTAIN,
    DIAGNOSTIC_FIELDS,
    DraftDiagnostics,
    PassOutput,
    SelectedPassOutput,
    select_two_pass_output,
    validate_draft_diagnostics,
)
from .action_ir import (
    ActionIRParseError,
    ActionIRValidationError,
    decode_json_object,
    validate_action_ir,
)
from .action_simulator import (
    ActionSimulator,
    assert_action_simulator_runtime_integrity,
    simulator_tool_registry_snapshot,
)
from .sim_program import (
    SimProgramError,
    WorldState,
    assert_sim_program_runtime_integrity,
    module_runtime_sha256,
    runtime_callable_identity,
)

RUNTIME_VERSION = "barun-action-correction-raw-runtime-v1"
DRAFT_TRANSPORT_VERSION = "barun-action-correction-draft-raw-transport-v1"
MODEL_INPUT_MODE = "inspect_one_raw_draft_and_return_complete_action_ir"
ACTION_IR_CONTRACT = "ACTION_IR_V1"

# This is a transport/defence limit, not a token budget or an experiment decoding cap.
# Token budgets remain unresolved and therefore cannot authorize model access.
MAX_DRAFT_UTF8_BYTES = 65_536
MAX_ORIGINAL_PROMPT_UTF8_BYTES = 1_048_576
_MAX_CORRECTION_PROMPT_UTF8_BYTES = 2_097_152

# These are the exact special strings in the pinned BarunLM/BarunAction tokenizer.  The
# runtime does not load a tokenizer.  Case-insensitive rejection is intentionally more
# conservative than tokenizer matching so a later renderer cannot make case handling an
# injection ambiguity.
RESERVED_TOKENIZER_CONTROL_TOKENS = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<system>",
    "<user>",
    "<assistant>",
)

RUNTIME_FLAGS = MappingProxyType(
    {
        "population_materialization_authorized": False,
        "model_access_authorized": False,
        "tokenizer_access_authorized": False,
        "training_authorized": False,
        "cuda_authorized": False,
        "jarvislabs_resource_creation_authorized": False,
        "network_access_authorized": False,
        "dataset_access_authorized": False,
        "human_label_access_authorized": False,
        "external_execution_permitted": False,
        "launch_authorized": False,
    }
)

_T_FIELDS = frozenset({"action_ir_contract", "instruction", "request", "context", "tools"})
_D_FIELDS = frozenset({"contract", "task", "user_request", "supplied_context", "tool_schemas"})
_CORRECTION_PROMPT_FIELDS = frozenset(
    {"schema_version", "mode", "original", "draft_raw", "diagnostics"}
)
_T_INSTRUCTION = "Compile the request into one canonical Action IR proposal."
_D_TASK = "Return only strict canonical JSON for the requested personal-action proposal."
_ACCEPTED_SIMULATOR_STATUSES = frozenset({"no_action", "simulated", "confirmation_required"})
_REJECTED_SIMULATOR_STATUSES = frozenset({"rejected", "policy_blocked"})


class ActionCorrectionRuntimeError(ValueError):
    """Stable fail-closed error for raw transport or original-prompt validation."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(f"{path}: {message}")
        self.code = code
        self.path = path


class PassTermination(str, Enum):
    """Evidence supplied by a generation boundary, without validity booleans."""

    COMPLETE = "complete"
    MISSING = "missing"
    FAILURE = "failure"
    TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class _TextSnapshot:
    text: str
    utf8: bytes
    sha256: str

    @property
    def byte_count(self) -> int:
        return len(self.utf8)


@dataclass(frozen=True, slots=True)
class _OriginalPrompt:
    canonical_json: str
    payload: dict[str, Any]
    shape: str
    state: WorldState


@dataclass(frozen=True, slots=True)
class _DerivedDraft:
    diagnostics: DraftDiagnostics
    pass_output: PassOutput


@dataclass(frozen=True, slots=True)
class _CorrectionEvidence:
    original: _OriginalPrompt
    draft: _TextSnapshot
    derived: _DerivedDraft


@dataclass(frozen=True, slots=True)
class CorrectionPrompt:
    """One immutable, label-free correction prompt and its hidden audit evidence."""

    prompt_json: str
    original_shape: str
    draft_raw: str
    draft_utf8_bytes: int
    draft_sha256: str
    state_sha256: str
    diagnostics: DraftDiagnostics
    pass_output: PassOutput

    def __post_init__(self) -> None:
        # This is semantic validation, not a private-token provenance assertion.  A direct
        # caller may construct the object only by supplying evidence that rederives exactly
        # from the serialized prompt.
        _validate_correction_prompt_instance(self)

    def audit_record(self) -> dict[str, object]:
        """Rederive and return nonauthorizing evidence without raw prompt or draft."""

        evidence = _validate_correction_prompt_instance(self)

        return {
            "schema_version": RUNTIME_VERSION,
            "draft_transport_version": DRAFT_TRANSPORT_VERSION,
            "original_shape": evidence.original.shape,
            "draft_utf8_bytes": evidence.draft.byte_count,
            "draft_sha256": evidence.draft.sha256,
            "state_sha256": evidence.original.state.sha256(),
            "diagnostics": _diagnostics_dict(evidence.derived.diagnostics),
            "flags": dict(RUNTIME_FLAGS),
        }


@dataclass(frozen=True, slots=True)
class PassAssessment:
    """Guarded-call transport, not an independently accepted audit or receipt surface."""

    termination: PassTermination
    raw: str | None
    raw_sha256: str | None
    diagnostics: DraftDiagnostics | None
    pass_output: PassOutput

    def __post_init__(self) -> None:
        if type(self.termination) is not PassTermination:
            raise TypeError("termination must be an exact PassTermination")
        if type(self.pass_output) is not PassOutput:
            raise TypeError("pass_output must be an exact PassOutput")
        if self.termination in {PassTermination.MISSING, PassTermination.FAILURE}:
            if self.raw is not None or self.raw_sha256 is not None or self.diagnostics is not None:
                raise ValueError("missing and failed pass evidence must be empty")
            expected = PassOutput(
                raw=None,
                parse_valid=False,
                schema_valid=False,
                policy_conformant_proposal=False,
            )
            if self.pass_output != expected:
                raise ValueError("missing or failed pass output is inconsistent")
            return

        snapshot = _snapshot_draft(self.raw)
        if self.raw != snapshot.text or self.raw_sha256 != snapshot.sha256:
            raise ValueError("pass raw-text evidence is inconsistent")
        if type(self.diagnostics) is not DraftDiagnostics:
            raise TypeError("complete and truncated passes require exact diagnostics")
        _diagnostics_dict(self.diagnostics)
        if self.termination is PassTermination.TRUNCATED:
            expected = PassOutput(
                raw=snapshot.text or None,
                parse_valid=False,
                schema_valid=False,
                policy_conformant_proposal=False,
            )
        else:
            expected = PassOutput(
                raw=snapshot.text or None,
                parse_valid=self.diagnostics.parse_valid,
                schema_valid=self.diagnostics.schema_valid,
                policy_conformant_proposal=(
                    self.diagnostics.schema_valid
                    and self.diagnostics.policy_status in {"conformant", "requires_confirmation"}
                ),
            )
        if self.pass_output != expected:
            raise ValueError("pass validity evidence is inconsistent")


@dataclass(frozen=True, slots=True)
class TwoPassSelection:
    """Guarded-call transport, not an independently accepted audit or receipt surface."""

    selected: SelectedPassOutput
    pass1: PassAssessment
    pass2: PassAssessment

    def __post_init__(self) -> None:
        if type(self.selected) is not SelectedPassOutput:
            raise TypeError("selected must be an exact SelectedPassOutput")
        if type(self.pass1) is not PassAssessment or type(self.pass2) is not PassAssessment:
            raise TypeError("pass evidence must use exact PassAssessment values")
        expected = select_two_pass_output(
            pass1=self.pass1.pass_output,
            pass2=self.pass2.pass_output,
        )
        if self.selected != expected:
            raise ValueError("two-pass selection differs from the deterministic fallback")

    @property
    def raw(self) -> str:
        return self.selected.raw

    @property
    def source(self) -> str:
        return self.selected.source


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise _DuplicateKeyError(key)
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite constant {value!r}")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ActionCorrectionRuntimeError(
            "invalid_json", "value cannot be represented as strict canonical JSON"
        ) from exc


def _snapshot_text(
    value: object,
    *,
    path: str,
    byte_limit: int,
    reject_reserved_tokens: bool,
) -> _TextSnapshot:
    if type(value) is not str:
        raise ActionCorrectionRuntimeError("type_mismatch", "must be an exact string", path)
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ActionCorrectionRuntimeError(
            "invalid_utf8", "contains a value that cannot be encoded as strict UTF-8", path
        ) from exc
    if len(encoded) > byte_limit:
        raise ActionCorrectionRuntimeError(
            "size_limit",
            f"UTF-8 payload exceeds the fixed {byte_limit}-byte transport limit",
            path,
        )

    # Decode the detached bytes rather than coercing or normalizing the caller object.  The
    # equality check documents the byte-exact snapshot invariant explicitly.
    snapshot = encoded.decode("utf-8", errors="strict")
    if snapshot != value:  # pragma: no cover - strict UTF-8 round trips by definition.
        raise ActionCorrectionRuntimeError("snapshot_mismatch", "UTF-8 snapshot changed", path)
    if reject_reserved_tokens:
        folded = snapshot.casefold()
        for token in RESERVED_TOKENIZER_CONTROL_TOKENS:
            if token in folded:
                raise ActionCorrectionRuntimeError(
                    "reserved_token", "contains a reserved tokenizer control token", path
                )
    return _TextSnapshot(
        text=snapshot,
        utf8=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _validate_decoded_utf8(value: object, *, path: str = "$") -> None:
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ActionCorrectionRuntimeError(
                "invalid_utf8", "decoded JSON string is not strict UTF-8", path
            ) from exc
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_decoded_utf8(child, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            _validate_decoded_utf8(key, path=path)
            _validate_decoded_utf8(child, path=f"{path}.{key}")


def _load_canonical_object(snapshot: _TextSnapshot) -> dict[str, Any]:
    try:
        decoded = json.loads(
            snapshot.text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except _DuplicateKeyError as exc:
        raise ActionCorrectionRuntimeError(
            "duplicate_key", "original prompt contains a duplicate JSON key", "$.original"
        ) from exc
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ActionCorrectionRuntimeError(
            "invalid_json", "original prompt is not one strict JSON object", "$.original"
        ) from exc
    if type(decoded) is not dict:
        raise ActionCorrectionRuntimeError(
            "top_level_not_object", "original prompt must be a JSON object", "$.original"
        )
    _validate_decoded_utf8(decoded, path="$.original")
    if _canonical_json(decoded) != snapshot.text:
        raise ActionCorrectionRuntimeError(
            "noncanonical_original",
            "original prompt must use the existing exact canonical JSON encoding",
            "$.original",
        )
    return decoded


def _expected_schema_presentations() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    registry = simulator_tool_registry_snapshot()
    records: list[dict[str, object]] = []
    for schema in registry.values():
        records.append(
            {
                "name": schema.name,
                "arguments": {
                    name: {
                        "kind": value.kind.value,
                        "enum": list(value.enum) if value.enum is not None else None,
                    }
                    for name, value in schema.arguments.items()
                },
                "required": sorted(schema.required),
                "side_effecting": schema.side_effecting,
            }
        )
    train = [
        {
            "tool": record["name"],
            "args": record["arguments"],
            "required": record["required"],
            "side_effecting": record["side_effecting"],
        }
        for record in records
    ]
    screen = [
        {
            "name": record["name"],
            "parameters": record["arguments"],
            "required_parameters": record["required"],
            "effect": "proposal" if record["side_effecting"] else "read_only",
        }
        for record in records
    ]
    return train, screen


def _require_exact_object(value: object, *, fields: frozenset[str], path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ActionCorrectionRuntimeError("prompt_shape", "must be an exact object", path)
    if set(value) != fields:
        raise ActionCorrectionRuntimeError(
            "prompt_shape", "fields differ from the existing canonical prompt shape", path
        )
    return value


def _extract_original_prompt(original_prompt_json: object) -> _OriginalPrompt:
    snapshot = _snapshot_text(
        original_prompt_json,
        path="$.original",
        byte_limit=MAX_ORIGINAL_PROMPT_UTF8_BYTES,
        reject_reserved_tokens=True,
    )
    payload = _load_canonical_object(snapshot)
    train_schemas, screen_schemas = _expected_schema_presentations()

    if set(payload) == _T_FIELDS:
        if payload["action_ir_contract"] != ACTION_IR_CONTRACT:
            raise ActionCorrectionRuntimeError(
                "prompt_shape", "training prompt Action IR contract changed", "$.original"
            )
        if payload["instruction"] != _T_INSTRUCTION:
            raise ActionCorrectionRuntimeError(
                "prompt_shape", "training prompt instruction changed", "$.original"
            )
        if type(payload["request"]) is not str or not payload["request"]:
            raise ActionCorrectionRuntimeError(
                "prompt_shape", "training request must be a non-empty string", "$.original"
            )
        context = _require_exact_object(
            payload["context"], fields=frozenset({"state"}), path="$.original.context"
        )
        if payload["tools"] != train_schemas:
            raise ActionCorrectionRuntimeError(
                "prompt_shape", "training tool presentation changed", "$.original.tools"
            )
        state_payload = context["state"]
        shape = "T-synth"
    elif set(payload) == _D_FIELDS:
        if payload["contract"] != ACTION_IR_CONTRACT:
            raise ActionCorrectionRuntimeError(
                "prompt_shape", "screen prompt Action IR contract changed", "$.original"
            )
        if payload["task"] != _D_TASK:
            raise ActionCorrectionRuntimeError(
                "prompt_shape", "screen prompt task changed", "$.original"
            )
        if type(payload["user_request"]) is not str or not payload["user_request"]:
            raise ActionCorrectionRuntimeError(
                "prompt_shape", "screen request must be a non-empty string", "$.original"
            )
        context = _require_exact_object(
            payload["supplied_context"],
            fields=frozenset({"available_state"}),
            path="$.original.supplied_context",
        )
        if payload["tool_schemas"] != screen_schemas:
            raise ActionCorrectionRuntimeError(
                "prompt_shape", "screen tool presentation changed", "$.original.tool_schemas"
            )
        state_payload = context["available_state"]
        shape = "D-internal"
    else:
        raise ActionCorrectionRuntimeError(
            "prompt_shape",
            "original prompt is neither an existing canonical T nor D shape",
            "$.original",
        )

    try:
        state = WorldState.from_dict(state_payload)
    except (SimProgramError, TypeError, ValueError) as exc:
        raise ActionCorrectionRuntimeError(
            "invalid_state", "original prompt contains an invalid simulator state", "$.original"
        ) from exc
    if state.canonical_json() != _canonical_json(state_payload):
        raise ActionCorrectionRuntimeError(
            "noncanonical_state",
            "original prompt state changes under the canonical simulator contract",
            "$.original",
        )
    return _OriginalPrompt(
        canonical_json=snapshot.text,
        payload=payload,
        shape=shape,
        state=state,
    )


def _diagnostics_dict(diagnostics: DraftDiagnostics) -> dict[str, object]:
    result = {
        "parse_valid": diagnostics.parse_valid,
        "schema_valid": diagnostics.schema_valid,
        "policy_status": diagnostics.policy_status,
        "simulator_status": diagnostics.simulator_status,
        "error_codes": list(diagnostics.error_codes),
    }
    if tuple(result) != DIAGNOSTIC_FIELDS:
        raise ActionCorrectionRuntimeError(
            "diagnostic_shape", "model-visible diagnostic field order or roster changed"
        )
    # Revalidate through the frozen independent contract before anything becomes visible.
    validate_draft_diagnostics(result)
    return result


def _draft_diagnostics(
    *,
    parse_valid: bool,
    schema_valid: bool,
    policy_status: str,
    simulator_status: str,
    error_codes: tuple[str, ...],
) -> DraftDiagnostics:
    return validate_draft_diagnostics(
        {
            "parse_valid": parse_valid,
            "schema_valid": schema_valid,
            "policy_status": policy_status,
            "simulator_status": simulator_status,
            "error_codes": list(error_codes),
        }
    )


def _derive_draft(snapshot: _TextSnapshot, state: WorldState) -> _DerivedDraft:
    """Parse only for evidence; ``snapshot.text`` remains the transported value."""

    try:
        decoded = decode_json_object(snapshot.text)
    except ActionIRParseError as exc:
        diagnostics = _draft_diagnostics(
            parse_valid=False,
            schema_valid=False,
            policy_status="unavailable",
            simulator_status="unavailable",
            error_codes=(exc.code,),
        )
        return _DerivedDraft(
            diagnostics=diagnostics,
            pass_output=PassOutput(
                raw=snapshot.text or None,
                parse_valid=False,
                schema_valid=False,
                policy_conformant_proposal=False,
            ),
        )

    try:
        action = validate_action_ir(decoded, simulator_tool_registry_snapshot())
    except ActionIRValidationError as exc:
        diagnostics = _draft_diagnostics(
            parse_valid=True,
            schema_valid=False,
            policy_status="unavailable",
            simulator_status="unavailable",
            error_codes=(exc.code,),
        )
        return _DerivedDraft(
            diagnostics=diagnostics,
            pass_output=PassOutput(
                raw=snapshot.text or None,
                parse_valid=True,
                schema_valid=False,
                policy_conformant_proposal=False,
            ),
        )

    result = ActionSimulator().simulate(action, state)
    policy_valid = result.receipt.policy["policy_valid"]
    if type(policy_valid) is not bool:
        raise ActionCorrectionRuntimeError(
            "runtime_evidence", "simulator policy validity is not an exact boolean"
        )
    status = result.receipt.status
    if not policy_valid:
        policy_status = "blocked"
    elif status == "confirmation_required":
        policy_status = "requires_confirmation"
    else:
        policy_status = "conformant"
    if status in _ACCEPTED_SIMULATOR_STATUSES:
        simulator_status = "accepted"
    elif status in _REJECTED_SIMULATOR_STATUSES:
        simulator_status = "rejected"
    else:
        raise ActionCorrectionRuntimeError(
            "runtime_evidence", "simulator returned an unknown status"
        )
    error_codes = () if result.receipt.error_code is None else (result.receipt.error_code,)
    diagnostics = _draft_diagnostics(
        parse_valid=True,
        schema_valid=True,
        policy_status=policy_status,
        simulator_status=simulator_status,
        error_codes=tuple(sorted(set(error_codes))),
    )
    return _DerivedDraft(
        diagnostics=diagnostics,
        pass_output=PassOutput(
            raw=snapshot.text,
            parse_valid=True,
            schema_valid=True,
            policy_conformant_proposal=policy_valid,
        ),
    )


def _snapshot_draft(raw: object) -> _TextSnapshot:
    return _snapshot_text(
        raw,
        path="$.draft_raw",
        byte_limit=MAX_DRAFT_UTF8_BYTES,
        reject_reserved_tokens=True,
    )


def _rederive_correction_prompt(prompt_json: object) -> _CorrectionEvidence:
    snapshot = _snapshot_text(
        prompt_json,
        path="$.prompt_json",
        byte_limit=_MAX_CORRECTION_PROMPT_UTF8_BYTES,
        reject_reserved_tokens=True,
    )
    try:
        payload = json.loads(
            snapshot.text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except _DuplicateKeyError as exc:
        raise ActionCorrectionRuntimeError(
            "duplicate_key",
            "correction prompt contains a duplicate JSON key",
            "$.prompt_json",
        ) from exc
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ActionCorrectionRuntimeError(
            "invalid_json",
            "correction prompt is not one strict JSON object",
            "$.prompt_json",
        ) from exc
    if type(payload) is not dict:
        raise ActionCorrectionRuntimeError(
            "top_level_not_object",
            "correction prompt must be a JSON object",
            "$.prompt_json",
        )
    try:
        _validate_decoded_utf8(payload, path="$.prompt_json")
        canonical = _canonical_json(payload)
    except RecursionError as exc:
        raise ActionCorrectionRuntimeError(
            "invalid_json",
            "correction prompt exceeds the bounded JSON nesting",
            "$.prompt_json",
        ) from exc
    if canonical != snapshot.text:
        raise ActionCorrectionRuntimeError(
            "noncanonical_prompt",
            "correction prompt must use exact canonical JSON encoding",
            "$.prompt_json",
        )
    if set(payload) != _CORRECTION_PROMPT_FIELDS:
        raise ActionCorrectionRuntimeError(
            "prompt_shape",
            "correction prompt fields differ from the exact transport shape",
            "$.prompt_json",
        )
    if payload["schema_version"] != DRAFT_TRANSPORT_VERSION or payload["mode"] != MODEL_INPUT_MODE:
        raise ActionCorrectionRuntimeError(
            "prompt_shape",
            "correction prompt contract identity changed",
            "$.prompt_json",
        )

    original_json = _canonical_json(payload["original"])
    original = _extract_original_prompt(original_json)
    draft = _snapshot_draft(payload["draft_raw"])
    derived = _derive_draft(draft, original.state)
    try:
        visible_diagnostics = validate_draft_diagnostics(payload["diagnostics"])
    except (TypeError, ValueError) as exc:
        raise ActionCorrectionRuntimeError(
            "diagnostic_mismatch",
            "correction prompt diagnostics violate the exact five-field contract",
            "$.prompt_json.diagnostics",
        ) from exc
    if visible_diagnostics != derived.diagnostics:
        raise ActionCorrectionRuntimeError(
            "diagnostic_mismatch",
            "correction prompt diagnostics differ from live draft/state derivation",
            "$.prompt_json.diagnostics",
        )
    return _CorrectionEvidence(original=original, draft=draft, derived=derived)


def _instance_field(instance: object, name: str) -> object:
    try:
        return object.__getattribute__(instance, name)
    except AttributeError as exc:
        raise ActionCorrectionRuntimeError(
            "evidence_mismatch",
            f"correction evidence is missing field {name!r}",
            f"$.{name}",
        ) from exc


def _validate_correction_prompt_instance(instance: object) -> _CorrectionEvidence:
    prompt_json = _instance_field(instance, "prompt_json")
    evidence = _rederive_correction_prompt(prompt_json)
    expected_fields: tuple[tuple[str, object, type[object]], ...] = (
        ("prompt_json", prompt_json, str),
        ("original_shape", evidence.original.shape, str),
        ("draft_raw", evidence.draft.text, str),
        ("draft_utf8_bytes", evidence.draft.byte_count, int),
        ("draft_sha256", evidence.draft.sha256, str),
        ("state_sha256", evidence.original.state.sha256(), str),
        ("diagnostics", evidence.derived.diagnostics, DraftDiagnostics),
        ("pass_output", evidence.derived.pass_output, PassOutput),
    )
    for name, expected, expected_type in expected_fields:
        observed = _instance_field(instance, name)
        if type(observed) is not expected_type or observed != expected:
            raise ActionCorrectionRuntimeError(
                "evidence_mismatch",
                f"correction evidence field {name!r} differs from live derivation",
                f"$.{name}",
            )
    return evidence


def build_correction_prompt(*, original_prompt_json: str, draft_raw: str) -> CorrectionPrompt:
    """Build one exact-string correction input from a canonical T/D source prompt.

    ``draft_raw`` is snapshotted before it is parsed.  The model-visible payload stores
    that snapshot directly under ``draft_raw``; the parser is used only to populate the
    separate five-field diagnostics object.  Empty and malformed drafts are preserved and
    diagnosed rather than repaired.
    """

    original = _extract_original_prompt(original_prompt_json)
    snapshot = _snapshot_draft(draft_raw)

    # Capture the raw string in the transport payload before deriving any semantic view.
    # It is never replaced with json.loads(snapshot.text) or an Action IR serialization.
    model_payload: dict[str, object] = {
        "schema_version": DRAFT_TRANSPORT_VERSION,
        "mode": MODEL_INPUT_MODE,
        "original": original.payload,
        "draft_raw": snapshot.text,
    }
    derived = _derive_draft(snapshot, original.state)
    model_payload["diagnostics"] = _diagnostics_dict(derived.diagnostics)
    prompt_json = _canonical_json(model_payload)

    # Prove that JSON string escaping is reversible and did not normalize draft bytes.
    decoded_prompt = json.loads(prompt_json)
    if decoded_prompt.get("draft_raw") != snapshot.text:
        raise ActionCorrectionRuntimeError(
            "transport_round_trip", "draft_raw changed during JSON-string transport"
        )
    visible_diagnostics = decoded_prompt.get("diagnostics")
    if type(visible_diagnostics) is not dict or set(visible_diagnostics) != set(DIAGNOSTIC_FIELDS):
        raise ActionCorrectionRuntimeError(
            "diagnostic_shape", "model-visible diagnostics are not exactly the five allowed fields"
        )
    return CorrectionPrompt(
        prompt_json=prompt_json,
        original_shape=original.shape,
        draft_raw=snapshot.text,
        draft_utf8_bytes=snapshot.byte_count,
        draft_sha256=snapshot.sha256,
        state_sha256=original.state.sha256(),
        diagnostics=derived.diagnostics,
        pass_output=derived.pass_output,
    )


def _assess_pass(
    *,
    original: _OriginalPrompt,
    raw: object,
    termination: PassTermination,
) -> PassAssessment:
    if type(termination) is not PassTermination:
        raise TypeError("termination must be an exact PassTermination")
    if termination in {PassTermination.MISSING, PassTermination.FAILURE}:
        if raw is not None:
            raise ActionCorrectionRuntimeError(
                "termination_shape",
                "missing and failed generations must not synthesize or retain output text",
                "$.raw",
            )
        return PassAssessment(
            termination=termination,
            raw=None,
            raw_sha256=None,
            diagnostics=None,
            pass_output=PassOutput(
                raw=None,
                parse_valid=False,
                schema_valid=False,
                policy_conformant_proposal=False,
            ),
        )

    snapshot = _snapshot_draft(raw)
    derived = _derive_draft(snapshot, original.state)
    if termination is PassTermination.TRUNCATED:
        # A prefix can accidentally be valid JSON.  Termination evidence dominates: a
        # truncated generation is never eligible, while its raw-only diagnostics remain
        # available for inspection.
        output = PassOutput(
            raw=snapshot.text or None,
            parse_valid=False,
            schema_valid=False,
            policy_conformant_proposal=False,
        )
    else:
        output = derived.pass_output
    return PassAssessment(
        termination=termination,
        raw=snapshot.text,
        raw_sha256=snapshot.sha256,
        diagnostics=derived.diagnostics,
        pass_output=output,
    )


def assess_pass_output(
    *,
    original_prompt_json: str,
    raw: str | None,
    termination: PassTermination,
) -> PassAssessment:
    """Recompute one pass from raw text and termination, accepting no validity flags."""

    original = _extract_original_prompt(original_prompt_json)
    return _assess_pass(original=original, raw=raw, termination=termination)


def select_two_pass_outputs(
    *,
    original_prompt_json: str,
    pass1_raw: str | None,
    pass1_termination: PassTermination,
    pass2_raw: str | None,
    pass2_termination: PassTermination,
) -> TwoPassSelection:
    """Derive both pass records and apply pass 2, pass 1, then canonical abstention."""

    original = _extract_original_prompt(original_prompt_json)
    pass1 = _assess_pass(
        original=original,
        raw=pass1_raw,
        termination=pass1_termination,
    )
    pass2 = _assess_pass(
        original=original,
        raw=pass2_raw,
        termination=pass2_termination,
    )
    selected = select_two_pass_output(
        pass1=pass1.pass_output,
        pass2=pass2.pass_output,
    )
    if selected.source == "canonical_abstain" and selected.raw != CANONICAL_ABSTAIN:
        raise ActionCorrectionRuntimeError(
            "fallback_identity", "fallback abstention differs from the canonical contract"
        )
    return TwoPassSelection(selected=selected, pass1=pass1, pass2=pass2)


def runtime_boundary() -> dict[str, object]:
    """Return the explicit CPU-only, nonauthorizing boundary."""

    return {
        "schema_version": RUNTIME_VERSION,
        "draft_transport_version": DRAFT_TRANSPORT_VERSION,
        "model_input_mode": MODEL_INPUT_MODE,
        "max_draft_utf8_bytes": MAX_DRAFT_UTF8_BYTES,
        "max_original_prompt_utf8_bytes": MAX_ORIGINAL_PROMPT_UTF8_BYTES,
        "reserved_tokenizer_control_tokens": list(RESERVED_TOKENIZER_CONTROL_TOKENS),
        "model_visible_diagnostic_fields": list(DIAGNOSTIC_FIELDS),
        "fallback_order": ["pass2", "pass1", "canonical_abstain"],
        "accepted_audit_surfaces": ["CorrectionPrompt.audit_record"],
        "transport_only_not_receipts": ["PassAssessment", "TwoPassSelection"],
        "integrity_boundary": (
            "ordinary checked-global, dependency, constant, and source tamper evidence; "
            "not a hostile same-UID or public-API-replacement security boundary"
        ),
        "flags": dict(RUNTIME_FLAGS),
    }


__all__ = [
    "ACTION_IR_CONTRACT",
    "DRAFT_TRANSPORT_VERSION",
    "MAX_DRAFT_UTF8_BYTES",
    "MAX_ORIGINAL_PROMPT_UTF8_BYTES",
    "MODEL_INPUT_MODE",
    "RESERVED_TOKENIZER_CONTROL_TOKENS",
    "RUNTIME_FLAGS",
    "RUNTIME_VERSION",
    "ActionCorrectionRuntimeError",
    "CorrectionPrompt",
    "PassAssessment",
    "PassTermination",
    "TwoPassSelection",
    "assess_pass_output",
    "build_correction_prompt",
    "runtime_boundary",
    "select_two_pass_outputs",
]


def _install_runtime_integrity() -> None:
    """Install closure-captured entry/exit guards on evidence-returning APIs.

    The checker deliberately lives in each wrapper's closure.  Rebinding the module's
    ``_assert_runtime_integrity`` name therefore cannot disable the check; that rebinding
    is itself one of the checked identities.
    """

    namespace = globals()
    module_name = __name__
    source_path = __file__
    error_type = ActionCorrectionRuntimeError
    function_type = type(_canonical_json)

    pinned_module_runtime_sha256 = module_runtime_sha256
    pinned_runtime_callable_identity = runtime_callable_identity
    pinned_sim_program_guard = assert_sim_program_runtime_integrity
    pinned_action_simulator_guard = assert_action_simulator_runtime_integrity
    pinned_inspect_signature = inspect.signature

    state: dict[str, object] = {}

    def fail(message: str, cause: Exception | None = None) -> None:
        error = error_type("runtime_identity", message)
        if cause is None:
            raise error
        raise error from cause

    def check() -> None:
        expected_functions = state.get("functions")
        expected_globals = state.get("globals")
        expected_module_attributes = state.get("module_attributes")
        expected_runtime_sha256 = state.get("runtime_sha256")
        expected_contract_sha256 = state.get("contract_sha256")
        runtime_contract = state.get("runtime_contract")
        contract_contract = state.get("contract_contract")
        if not (
            type(expected_functions) is tuple
            and type(expected_globals) is tuple
            and type(expected_module_attributes) is tuple
            and type(expected_runtime_sha256) is str
            and type(expected_contract_sha256) is str
            and type(runtime_contract) is dict
            and type(contract_contract) is dict
        ):
            fail("runtime integrity guard was not initialized")

        for name, expected_object, expected_code in expected_functions:
            current = namespace.get(name)
            if (
                current is not expected_object
                or getattr(current, "__code__", None) is not expected_code
            ):
                fail(f"runtime callable binding changed: {name}")
        for name, expected_object, expected_code in expected_globals:
            current = namespace.get(name)
            if current is not expected_object:
                fail(f"runtime global binding changed: {name}")
            if (
                expected_code is not None
                and getattr(current, "__code__", None) is not expected_code
            ):
                fail(f"runtime dependency code changed: {name}")
        for owner, attribute, expected_object, expected_code, label in expected_module_attributes:
            current = getattr(owner, attribute, None)
            if current is not expected_object:
                fail(f"runtime dependency binding changed: {label}")
            if (
                expected_code is not None
                and getattr(current, "__code__", None) is not expected_code
            ):
                fail(f"runtime dependency code changed: {label}")

        try:
            pinned_sim_program_guard()
            pinned_action_simulator_guard()
        except Exception as exc:  # noqa: BLE001 - dependency tamper must fail stably.
            fail("dependent parser or simulator runtime changed", exc)

        try:
            current_contract_sha256 = pinned_module_runtime_sha256(
                vars(_contract_module),
                module_name=_contract_module.__name__,
                source_path=_contract_module.__file__,
                contract=contract_contract,
            )
        except Exception as exc:  # noqa: BLE001 - dependency tamper must fail stably.
            fail("could not verify the correction-contract runtime", exc)
        if current_contract_sha256 != expected_contract_sha256:
            fail("correction-contract source, code, or constants changed")

        try:
            current_runtime_sha256 = pinned_module_runtime_sha256(
                namespace,
                module_name=module_name,
                source_path=source_path,
                contract=runtime_contract,
            )
        except Exception as exc:  # noqa: BLE001 - dependency tamper must fail stably.
            fail("could not verify the action-correction runtime", exc)
        if current_runtime_sha256 != expected_runtime_sha256:
            fail("action-correction source, code, dependency, or constants changed")

    def guarded(function: object) -> object:
        if type(function) is not function_type:
            fail("only exact Python functions can receive the runtime guard")
        signature = pinned_inspect_signature(function)

        def wrapper(*args: object, **kwargs: object) -> object:
            check()
            try:
                result = function(*args, **kwargs)  # type: ignore[operator]
            finally:
                check()
            return result

        wrapper.__name__ = function.__name__  # type: ignore[attr-defined]
        wrapper.__qualname__ = function.__qualname__  # type: ignore[attr-defined]
        wrapper.__doc__ = function.__doc__  # type: ignore[attr-defined]
        wrapper.__annotations__ = dict(function.__annotations__)  # type: ignore[attr-defined]
        wrapper.__signature__ = signature  # type: ignore[attr-defined]
        return wrapper

    for public_name in (
        "assess_pass_output",
        "build_correction_prompt",
        "runtime_boundary",
        "select_two_pass_outputs",
    ):
        namespace[public_name] = guarded(namespace[public_name])
    for evidence_type in (CorrectionPrompt, PassAssessment, TwoPassSelection):
        evidence_type.__init__ = guarded(evidence_type.__init__)  # type: ignore[method-assign]
    CorrectionPrompt.audit_record = guarded(CorrectionPrompt.audit_record)  # type: ignore[method-assign]

    # Publish the diagnostic name only after wrappers have captured ``check``.  Rebinding
    # this global later cannot change the callable held by any wrapper.
    namespace["_assert_runtime_integrity"] = check

    checked_global_names = (
        "__file__",
        "_contract_module",
        "_action_ir_module",
        "_action_simulator_module",
        "_sim_program_module",
        "hashlib",
        "inspect",
        "json",
        "MappingProxyType",
        "CANONICAL_ABSTAIN",
        "DIAGNOSTIC_FIELDS",
        "DraftDiagnostics",
        "PassOutput",
        "SelectedPassOutput",
        "select_two_pass_output",
        "validate_draft_diagnostics",
        "ActionIRParseError",
        "ActionIRValidationError",
        "decode_json_object",
        "validate_action_ir",
        "ActionSimulator",
        "assert_action_simulator_runtime_integrity",
        "simulator_tool_registry_snapshot",
        "SimProgramError",
        "WorldState",
        "assert_sim_program_runtime_integrity",
        "module_runtime_sha256",
        "runtime_callable_identity",
        "RUNTIME_VERSION",
        "DRAFT_TRANSPORT_VERSION",
        "MODEL_INPUT_MODE",
        "ACTION_IR_CONTRACT",
        "MAX_DRAFT_UTF8_BYTES",
        "MAX_ORIGINAL_PROMPT_UTF8_BYTES",
        "_MAX_CORRECTION_PROMPT_UTF8_BYTES",
        "RESERVED_TOKENIZER_CONTROL_TOKENS",
        "RUNTIME_FLAGS",
        "_T_FIELDS",
        "_D_FIELDS",
        "_CORRECTION_PROMPT_FIELDS",
        "_T_INSTRUCTION",
        "_D_TASK",
        "_ACCEPTED_SIMULATOR_STATUSES",
        "_REJECTED_SIMULATOR_STATUSES",
        "ActionCorrectionRuntimeError",
        "PassTermination",
        "CorrectionPrompt",
        "PassAssessment",
        "TwoPassSelection",
    )
    state["globals"] = tuple(
        (
            name,
            namespace[name],
            getattr(namespace[name], "__code__", None),
        )
        for name in checked_global_names
    )
    state["module_attributes"] = tuple(
        (
            owner,
            attribute,
            getattr(owner, attribute),
            getattr(getattr(owner, attribute), "__code__", None),
            label,
        )
        for owner, attribute, label in (
            (hashlib, "sha256", "hashlib.sha256"),
            (inspect, "signature", "inspect.signature"),
            (json, "dumps", "json.dumps"),
            (json, "loads", "json.loads"),
            (_contract_module, "CANONICAL_ABSTAIN", "contract.CANONICAL_ABSTAIN"),
            (_contract_module, "DIAGNOSTIC_FIELDS", "contract.DIAGNOSTIC_FIELDS"),
            (_contract_module, "DraftDiagnostics", "contract.DraftDiagnostics"),
            (_contract_module, "PassOutput", "contract.PassOutput"),
            (_contract_module, "SelectedPassOutput", "contract.SelectedPassOutput"),
            (
                _contract_module,
                "select_two_pass_output",
                "contract.select_two_pass_output",
            ),
            (
                _contract_module,
                "validate_draft_diagnostics",
                "contract.validate_draft_diagnostics",
            ),
            (_action_ir_module, "decode_json_object", "action_ir.decode_json_object"),
            (_action_ir_module, "validate_action_ir", "action_ir.validate_action_ir"),
            (
                _action_simulator_module,
                "ActionSimulator",
                "action_simulator.ActionSimulator",
            ),
            (
                _action_simulator_module,
                "assert_action_simulator_runtime_integrity",
                "action_simulator.assert_action_simulator_runtime_integrity",
            ),
            (
                _action_simulator_module,
                "simulator_tool_registry_snapshot",
                "action_simulator.simulator_tool_registry_snapshot",
            ),
            (_sim_program_module, "SimProgramError", "sim_program.SimProgramError"),
            (_sim_program_module, "WorldState", "sim_program.WorldState"),
            (
                _sim_program_module,
                "assert_sim_program_runtime_integrity",
                "sim_program.assert_sim_program_runtime_integrity",
            ),
            (
                _sim_program_module,
                "module_runtime_sha256",
                "sim_program.module_runtime_sha256",
            ),
            (
                _sim_program_module,
                "runtime_callable_identity",
                "sim_program.runtime_callable_identity",
            ),
        )
    )
    state["functions"] = tuple(
        (name, value, value.__code__)
        for name, value in sorted(namespace.items())
        if type(value) is function_type and getattr(value, "__module__", None) == module_name
    )

    dependency_identities = {
        label: dict(pinned_runtime_callable_identity(value))
        for label, value in (
            ("hashlib.sha256", hashlib.sha256),
            ("json.dumps", json.dumps),
            ("json.loads", json.loads),
            ("decode_json_object", decode_json_object),
            ("validate_action_ir", validate_action_ir),
            ("select_two_pass_output", select_two_pass_output),
            ("validate_draft_diagnostics", validate_draft_diagnostics),
            ("simulator_tool_registry_snapshot", simulator_tool_registry_snapshot),
        )
    }
    state["contract_contract"] = {
        "scope": "action-correction-runtime-minimal-contract-dependency-v1",
        "canonical_abstain": CANONICAL_ABSTAIN,
        "diagnostic_fields": list(DIAGNOSTIC_FIELDS),
    }
    state["runtime_contract"] = {
        "schema_version": RUNTIME_VERSION,
        "draft_transport_version": DRAFT_TRANSPORT_VERSION,
        "checked_global_names": list(checked_global_names),
        "authorization_flags": dict(RUNTIME_FLAGS),
        "dependencies": dependency_identities,
        "integrity_boundary": (
            "ordinary checked-global, dependency, constant, and source tamper evidence; "
            "not hostile same-UID isolation"
        ),
    }
    state["contract_sha256"] = pinned_module_runtime_sha256(
        vars(_contract_module),
        module_name=_contract_module.__name__,
        source_path=_contract_module.__file__,
        contract=state["contract_contract"],  # type: ignore[arg-type]
    )
    state["runtime_sha256"] = pinned_module_runtime_sha256(
        namespace,
        module_name=module_name,
        source_path=source_path,
        contract=state["runtime_contract"],  # type: ignore[arg-type]
    )


_install_runtime_integrity()
