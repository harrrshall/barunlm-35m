"""Semantically typed, schema-valid, single-fault certificates for GVS negatives."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from enum import Enum
from typing import Any

from .action_ir import ActionIR, ActionIRError, Decision, ToolSchema, parse_action_ir
from .action_simulator import (
    ACTION_SIMULATOR_VERSION,
    POLICY_CONTRACT_SHA256,
    REFERENCE_EFFECT_VERSION,
    SIMULATOR_SCHEMA_SHA256,
    ActionSimulator,
    action_simulator_runtime_sha256,
    assert_action_simulator_runtime_integrity,
    simulate_reference_action,
    simulator_tool_registry_snapshot,
)
from .sim_program import (
    SIM_PROGRAM_VERSION,
    TIMEZONE_RUNTIME_VERSION,
    WORLD_STATE_VERSION,
    SimProgramError,
    StrictJSON,
    WorldState,
    canonical_json,
    canonical_sha256,
    loads_strict_json_object,
    module_runtime_sha256,
    runtime_callable_identity,
    snapshot_strict_json,
    timezone_runtime_sha256,
)

GVS_FAULT_CONTRACT_VERSION = "barun-gvs-single-fault-v5"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FAULT_CERTIFICATION_FACTORY_TOKEN = object()
_SEMANTIC_OUTPUT_NAMES = frozenset(
    {
        "counterfactual_state",
        "effects",
        "policy",
        "policy_status",
        "reference_observation",
        "reference_status",
    }
)


class GVSFaultError(ValueError):
    """A proposed hard negative is not one certified semantic fault."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(f"{path}: {message}")
        self.code = code
        self.path = path


class FaultKind(str, Enum):
    ENTITY = "entity_substitution"
    ARGUMENT_VALUE = "argument_value_substitution"
    TEMPORAL = "temporal_substitution"
    RECIPIENT = "recipient_substitution"
    BOOLEAN = "boolean_flip"
    ITEM = "item_substitution"
    OPERATION = "operation_substitution"
    DECISION = "decision_substitution"


_ARGUMENTS_BY_KIND = {
    FaultKind.ENTITY: frozenset(
        {
            "reminder_id",
            "event_id",
            "contact_id",
            "note_id",
            "message_id",
            "origin_id",
            "destination_id",
            "track_id",
            "key",
        }
    ),
    FaultKind.ARGUMENT_VALUE: frozenset({"title", "body", "text", "mode"}),
    FaultKind.TEMPORAL: frozenset({"due_at", "start_at", "end_at"}),
    FaultKind.RECIPIENT: frozenset({"to_contact_id", "channel"}),
    FaultKind.BOOLEAN: frozenset({"checked", "enabled"}),
    FaultKind.ITEM: frozenset({"list_id", "item_id"}),
}


@dataclass(frozen=True, slots=True)
class FaultDescriptor:
    kind: FaultKind
    call_index: int | None
    argument: str
    replacement: StrictJSON

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FaultKind):
            raise TypeError("kind must be FaultKind")
        if self.call_index is not None and (
            type(self.call_index) is not int or self.call_index < 0
        ):
            raise ValueError("call_index must be a non-negative integer or None")
        if not isinstance(self.argument, str) or not self.argument:
            raise ValueError("argument must be a non-empty string")
        object.__setattr__(self, "replacement", snapshot_strict_json(self.replacement))

    @property
    def semantic_path(self) -> str:
        if self.kind is FaultKind.DECISION:
            return "$.decision"
        if self.call_index is None:
            raise GVSFaultError("descriptor_shape", "call_index is required for call faults")
        if self.kind is FaultKind.OPERATION:
            return f"$.calls[{self.call_index}]"
        return f"$.calls[{self.call_index}].args.{self.argument}"

    def to_dict(self) -> dict[str, Any]:
        replacement = json.loads(canonical_json(self.replacement))
        return {
            "kind": self.kind.value,
            "call_index": self.call_index,
            "argument": self.argument,
            "replacement": replacement,
            "semantic_path": self.semantic_path,
        }


@dataclass(frozen=True, slots=True)
class FaultCertification:
    descriptor: FaultDescriptor
    gold_action: ActionIR
    faulty_action: ActionIR
    input_state_sha256: str
    gold_action_sha256: str
    faulty_action_sha256: str
    gold_visible_receipt_sha256: str
    faulty_visible_receipt_sha256: str
    gold_reference_receipt_sha256: str
    faulty_reference_receipt_sha256: str
    gold_counterfactual_state_sha256: str
    faulty_counterfactual_state_sha256: str
    timezone_runtime_sha256: str
    simulator_runtime_sha256: str
    fault_runtime_sha256: str
    differing_semantic_outputs: tuple[str, ...]
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _FAULT_CERTIFICATION_FACTORY_TOKEN:
            raise TypeError("FaultCertification must be constructed by the certifier or loader")
        if not isinstance(self.descriptor, FaultDescriptor):
            raise TypeError("descriptor must be FaultDescriptor")
        if not isinstance(self.gold_action, ActionIR) or not isinstance(
            self.faulty_action, ActionIR
        ):
            raise TypeError("certificate actions must be ActionIR")
        gold = _validated(self.gold_action)
        faulty = _validated(self.faulty_action)
        object.__setattr__(self, "gold_action", gold)
        object.__setattr__(self, "faulty_action", faulty)
        for name in (
            "input_state_sha256",
            "gold_action_sha256",
            "faulty_action_sha256",
            "gold_visible_receipt_sha256",
            "faulty_visible_receipt_sha256",
            "gold_reference_receipt_sha256",
            "faulty_reference_receipt_sha256",
            "gold_counterfactual_state_sha256",
            "faulty_counterfactual_state_sha256",
            "timezone_runtime_sha256",
            "simulator_runtime_sha256",
            "fault_runtime_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.simulator_runtime_sha256 != action_simulator_runtime_sha256():
            raise ValueError("simulator runtime hash mismatch")
        if self.fault_runtime_sha256 != gvs_fault_runtime_sha256():
            raise ValueError("fault runtime hash mismatch")
        if _action_sha256(gold) != self.gold_action_sha256:
            raise ValueError("gold action hash mismatch")
        if _action_sha256(faulty) != self.faulty_action_sha256:
            raise ValueError("faulty action hash mismatch")
        regenerated = make_single_fault(gold, self.descriptor)
        if regenerated.canonical_json() != faulty.canonical_json():
            raise ValueError("descriptor does not reproduce the faulty action")
        differences = tuple(self.differing_semantic_outputs)
        if (
            not differences
            or any(
                type(item) is not str or item not in _SEMANTIC_OUTPUT_NAMES for item in differences
            )
            or differences != tuple(sorted(set(differences)))
        ):
            raise ValueError("differing_semantic_outputs must be sorted and unique")
        object.__setattr__(self, "differing_semantic_outputs", differences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GVS_FAULT_CONTRACT_VERSION,
            "simulator_version": ACTION_SIMULATOR_VERSION,
            "schema_sha256": SIMULATOR_SCHEMA_SHA256,
            "policy_contract_sha256": POLICY_CONTRACT_SHA256,
            "program_contract_version": SIM_PROGRAM_VERSION,
            "world_contract_version": WORLD_STATE_VERSION,
            "reference_effect_version": REFERENCE_EFFECT_VERSION,
            "timezone_runtime_version": TIMEZONE_RUNTIME_VERSION,
            "descriptor": self.descriptor.to_dict(),
            "gold_action": self.gold_action.to_dict(),
            "faulty_action": self.faulty_action.to_dict(),
            "input_state_sha256": self.input_state_sha256,
            "gold_action_sha256": self.gold_action_sha256,
            "faulty_action_sha256": self.faulty_action_sha256,
            "gold_visible_receipt_sha256": self.gold_visible_receipt_sha256,
            "faulty_visible_receipt_sha256": self.faulty_visible_receipt_sha256,
            "gold_reference_receipt_sha256": self.gold_reference_receipt_sha256,
            "faulty_reference_receipt_sha256": self.faulty_reference_receipt_sha256,
            "gold_counterfactual_state_sha256": self.gold_counterfactual_state_sha256,
            "faulty_counterfactual_state_sha256": self.faulty_counterfactual_state_sha256,
            "timezone_runtime_sha256": self.timezone_runtime_sha256,
            "simulator_runtime_sha256": self.simulator_runtime_sha256,
            "fault_runtime_sha256": self.fault_runtime_sha256,
            "differing_semantic_outputs": list(self.differing_semantic_outputs),
            "schema_valid": True,
            "semantic_type_valid": True,
            "semantic_change_count": 1,
            "reference_only": True,
            "authorizes_execution": False,
            "external_side_effects": False,
        }

    def canonical_json(self) -> str:
        return canonical_json(snapshot_strict_json(self.to_dict()))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _action_sha256(action: ActionIR) -> str:
    return hashlib.sha256(action.canonical_json().encode("utf-8")).hexdigest()


def _same_schema(left: ToolSchema, right: ToolSchema) -> bool:
    return (
        left.arguments == right.arguments
        and left.required == right.required
        and left.additional_arguments == right.additional_arguments
        and left.side_effecting == right.side_effecting
    )


def _strict_same_type(left: object, right: object) -> bool:
    return type(left) is type(right)


def _stable_state(state: WorldState) -> WorldState:
    first = state.canonical_json()
    second = state.canonical_json()
    if first != second:
        raise GVSFaultError("concurrent_mutation", "world state changed during its stable snapshot")
    return WorldState.from_dict(json.loads(first))


def _validated(
    action: ActionIR,
    schemas: Mapping[str, ToolSchema] | None = None,
) -> ActionIR:
    if not isinstance(action, ActionIR):
        raise TypeError("action must be ActionIR")
    if schemas is None:
        schemas = simulator_tool_registry_snapshot()
    first = action.canonical_json()
    second = action.canonical_json()
    if first != second:
        raise GVSFaultError("concurrent_mutation", "Action IR changed during its stable snapshot")
    return parse_action_ir(first, schemas)


def _decision_replacement(
    gold: ActionIR,
    replacement: StrictJSON,
    schemas: Mapping[str, ToolSchema],
) -> ActionIR:
    replacement_is_object = isinstance(replacement, Mapping)
    if replacement_is_object:
        raw = canonical_json(replacement)
        candidate = parse_action_ir(raw, schemas)
    elif isinstance(replacement, str):
        try:
            decision = Decision(replacement)
        except ValueError as exc:
            raise GVSFaultError("invalid_decision", "unknown replacement decision") from exc
        if decision is Decision.ABSTAIN:
            payload: dict[str, Any] = {"decision": decision.value}
        elif decision is Decision.CLARIFY:
            payload = {"decision": decision.value, "missing": ["required_information"]}
        elif gold.decision in (Decision.CALL, Decision.CONFIRM):
            payload = gold.to_dict()
            payload["decision"] = decision.value
        else:
            raise GVSFaultError(
                "decision_payload_required",
                "CALL/CONFIRM replacement from a control requires a complete Action IR object",
            )
        candidate = parse_action_ir(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            schemas,
        )
    else:
        raise GVSFaultError(
            "type_mismatch", "decision replacement must be a string or Action IR object"
        )
    if candidate.decision is gold.decision:
        raise GVSFaultError("unchanged", "replacement decision must differ from gold")
    callable_decisions = (Decision.CALL, Decision.CONFIRM)
    gold_callable = gold.decision in callable_decisions
    candidate_callable = candidate.decision in callable_decisions
    if replacement_is_object and not candidate_callable:
        raise GVSFaultError(
            "decision_payload_scope",
            "object replacements are only allowed when the replacement is callable",
        )
    if gold_callable and candidate_callable:
        gold_payload = gold.to_dict()
        candidate_payload = candidate.to_dict()
        if (
            gold_payload["mode"] != candidate_payload["mode"]
            or gold_payload["calls"] != candidate_payload["calls"]
        ):
            raise GVSFaultError(
                "multiple_semantic_changes",
                "callable decision substitutions must preserve mode and calls byte-for-byte",
            )
    return candidate


def make_single_fault(gold: ActionIR, descriptor: FaultDescriptor) -> ActionIR:
    """Apply one declared high-level semantic substitution and revalidate Action IR."""

    assert_gvs_fault_runtime_integrity()
    schemas = simulator_tool_registry_snapshot()
    gold = _validated(gold, schemas)
    if not isinstance(descriptor, FaultDescriptor):
        raise TypeError("descriptor must be FaultDescriptor")
    descriptor_json = canonical_json(snapshot_strict_json(descriptor.to_dict()))
    if descriptor_json != canonical_json(snapshot_strict_json(descriptor.to_dict())):
        raise GVSFaultError(
            "concurrent_mutation", "fault descriptor changed during its stable snapshot"
        )
    descriptor = _descriptor_from_value(loads_strict_json_object(descriptor_json))
    if descriptor.kind is FaultKind.DECISION:
        if descriptor.call_index is not None or descriptor.argument != "decision":
            raise GVSFaultError("descriptor_shape", "decision fault must target $.decision")
        return _decision_replacement(gold, descriptor.replacement, schemas)

    if gold.decision not in (Decision.CALL, Decision.CONFIRM):
        raise GVSFaultError("not_callable", "argument/tool faults require CALL or CONFIRM")
    payload = gold.to_dict()
    call_index = descriptor.call_index
    if call_index is None or call_index >= len(payload["calls"]):
        raise GVSFaultError("call_index", "call_index is outside the canonical call sequence")
    call = payload["calls"][call_index]

    if descriptor.kind is FaultKind.OPERATION:
        if descriptor.argument != "tool":
            raise GVSFaultError("descriptor_shape", "operation fault must target one call")
        old_tool = call["tool"]
        replacement = descriptor.replacement
        if isinstance(replacement, Mapping):
            if set(replacement) != {"tool", "args"}:
                raise GVSFaultError(
                    "descriptor_shape",
                    "high-level operation replacement requires exact tool and args fields",
                )
            replacement_call = json.loads(canonical_json(replacement))
            new_tool = replacement_call["tool"]
        elif isinstance(replacement, str):
            new_tool = replacement
            replacement_call = {"tool": new_tool, "args": call["args"]}
        else:
            raise GVSFaultError(
                "descriptor_shape",
                "operation replacement must be a tool string or complete call object",
            )
        if not isinstance(new_tool, str):
            raise GVSFaultError("descriptor_shape", "replacement tool must be a string")
        if old_tool == new_tool:
            raise GVSFaultError("unchanged", "replacement must differ from gold")
        if new_tool not in schemas:
            raise GVSFaultError("unknown_tool", "replacement tool is not in the simulator registry")
        if schemas[old_tool].side_effecting != schemas[new_tool].side_effecting:
            raise GVSFaultError(
                "not_policy_preserving",
                "operation substitutions must preserve the side-effect policy class",
            )
        if isinstance(replacement, str) and not _same_schema(schemas[old_tool], schemas[new_tool]):
            raise GVSFaultError(
                "not_type_preserving",
                "tool-only substitutions require identical semantic argument contracts",
            )
        payload["calls"][call_index] = replacement_call
    else:
        allowed = _ARGUMENTS_BY_KIND.get(descriptor.kind)
        if allowed is None or descriptor.argument not in allowed:
            raise GVSFaultError(
                "fault_taxonomy",
                f"argument {descriptor.argument!r} is invalid for {descriptor.kind.value}",
            )
        if descriptor.argument not in call["args"]:
            raise GVSFaultError("missing_argument", "gold call lacks the targeted argument")
        old_value = call["args"][descriptor.argument]
        new_value = descriptor.replacement
        if not _strict_same_type(old_value, new_value):
            raise GVSFaultError("not_type_preserving", "replacement changes the JSON scalar type")
        if old_value == new_value:
            raise GVSFaultError("unchanged", "replacement must differ from gold")
        call["args"][descriptor.argument] = new_value

    try:
        return parse_action_ir(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            schemas,
        )
    except ActionIRError as exc:
        raise GVSFaultError(
            "semantic_type_invalid",
            "replacement violates the model-facing typed schema",
            descriptor.semantic_path,
        ) from exc


def _semantic_diff(left: ActionIR, right: ActionIR) -> tuple[str, ...]:
    left_payload = left.to_dict()
    right_payload = right.to_dict()
    differences: list[str] = []
    if left_payload.get("decision") != right_payload.get("decision"):
        differences.append("$.decision")
    if left_payload.get("mode") != right_payload.get("mode"):
        differences.append("$.mode")
    left_calls = left_payload.get("calls", [])
    right_calls = right_payload.get("calls", [])
    if len(left_calls) != len(right_calls):
        differences.append("$.calls.length")
        return tuple(differences)
    for index, (left_call, right_call) in enumerate(zip(left_calls, right_calls, strict=True)):
        if left_call["tool"] != right_call["tool"]:
            differences.append(f"$.calls[{index}].tool")
        all_args = sorted(set(left_call["args"]) | set(right_call["args"]))
        for argument in all_args:
            if left_call["args"].get(argument) != right_call["args"].get(argument):
                differences.append(f"$.calls[{index}].args.{argument}")
    if left_payload.get("missing") != right_payload.get("missing"):
        differences.append("$.missing")
    return tuple(differences)


def certify_single_fault(
    state: WorldState,
    gold: ActionIR,
    descriptor: FaultDescriptor,
) -> FaultCertification:
    """Certify one semantically typed, non-equivalent negative without real execution."""

    assert_gvs_fault_runtime_integrity()
    if not isinstance(state, WorldState):
        raise TypeError("state must be WorldState")
    state = _stable_state(state)
    gold = _validated(gold)
    simulator = ActionSimulator()
    gold_visible = simulator.simulate(gold, state)
    if gold_visible.receipt.status in {"rejected", "policy_blocked"}:
        raise GVSFaultError(
            "gold_invalid",
            "gold action must be schema-, policy-, and semantically valid",
        )
    gold_reference = simulate_reference_action(gold, state)
    if gold_reference.receipt.status == "reference_rejected":
        raise GVSFaultError("gold_rejected", "gold action must produce a valid reference effect")

    faulty = make_single_fault(gold, descriptor)
    differences = _semantic_diff(gold, faulty)
    if descriptor.kind is FaultKind.DECISION:
        if "$.decision" not in differences or faulty.decision is gold.decision:
            raise GVSFaultError(
                "decision_not_changed", "decision fault must change one high-level decision"
            )
        if (
            gold.decision in (Decision.CALL, Decision.CONFIRM)
            and faulty.decision
            in (
                Decision.CALL,
                Decision.CONFIRM,
            )
            and differences != ("$.decision",)
        ):
            raise GVSFaultError(
                "multiple_semantic_changes",
                f"callable decision fault changed more than $.decision: {differences!r}",
            )
    elif descriptor.kind is FaultKind.OPERATION:
        call_prefix = f"$.calls[{descriptor.call_index}]"
        if f"{call_prefix}.tool" not in differences or any(
            not difference.startswith(f"{call_prefix}.") for difference in differences
        ):
            raise GVSFaultError(
                "multiple_semantic_changes",
                f"operation fault changed content outside {call_prefix!r}: {differences!r}",
            )
    elif differences != (descriptor.semantic_path,):
        raise GVSFaultError(
            "multiple_semantic_changes",
            f"expected only {descriptor.semantic_path!r}, observed {differences!r}",
        )

    faulty_reference = simulate_reference_action(faulty, state)
    if (
        descriptor.kind is not FaultKind.DECISION
        and faulty_reference.receipt.status != "reference_applied"
    ):
        raise GVSFaultError(
            "semantic_type_invalid",
            "hard argument/tool negative must remain a valid counterfactual transition",
            descriptor.semantic_path,
        )
    if (
        descriptor.kind is FaultKind.DECISION
        and faulty_reference.receipt.status == "reference_rejected"
    ):
        raise GVSFaultError(
            "semantic_type_invalid",
            "decision replacement contains an invalid callable transition",
            descriptor.semantic_path,
        )
    faulty_visible = simulator.simulate(faulty, state)

    output_differences: list[str] = []
    if (
        gold_reference.counterfactual_state.sha256()
        != faulty_reference.counterfactual_state.sha256()
    ):
        output_differences.append("counterfactual_state")
    if gold_reference.receipt.effects_sha256 != faulty_reference.receipt.effects_sha256:
        output_differences.append("effects")
    if gold_reference.receipt.observation_sha256 != faulty_reference.receipt.observation_sha256:
        output_differences.append("reference_observation")
    if canonical_sha256(gold_visible.receipt.policy) != canonical_sha256(
        faulty_visible.receipt.policy
    ):
        output_differences.append("policy")
    if gold_visible.receipt.status != faulty_visible.receipt.status:
        output_differences.append("policy_status")
    if gold_reference.receipt.status != faulty_reference.receipt.status:
        output_differences.append("reference_status")
    output_differences = sorted(set(output_differences))
    if not output_differences:
        raise GVSFaultError(
            "semantic_equivalence",
            "fault is semantically equivalent and cannot be used as a negative",
        )
    receipts = (
        gold_visible.receipt,
        faulty_visible.receipt,
        gold_reference.receipt,
        faulty_reference.receipt,
    )
    if any(receipt.external_side_effects or receipt.authorizes_execution for receipt in receipts):
        raise AssertionError("semantic oracle receipts must never authorize or report side effects")
    return FaultCertification(
        descriptor=descriptor,
        gold_action=gold,
        faulty_action=faulty,
        input_state_sha256=state.sha256(),
        gold_action_sha256=_action_sha256(gold),
        faulty_action_sha256=_action_sha256(faulty),
        gold_visible_receipt_sha256=gold_visible.receipt.sha256(),
        faulty_visible_receipt_sha256=faulty_visible.receipt.sha256(),
        gold_reference_receipt_sha256=gold_reference.receipt.sha256(),
        faulty_reference_receipt_sha256=faulty_reference.receipt.sha256(),
        gold_counterfactual_state_sha256=gold_reference.counterfactual_state.sha256(),
        faulty_counterfactual_state_sha256=faulty_reference.counterfactual_state.sha256(),
        timezone_runtime_sha256=timezone_runtime_sha256(state.timezone),
        simulator_runtime_sha256=action_simulator_runtime_sha256(),
        fault_runtime_sha256=gvs_fault_runtime_sha256(),
        differing_semantic_outputs=tuple(output_differences),
        _factory_token=_FAULT_CERTIFICATION_FACTORY_TOKEN,
    )


def _descriptor_from_value(value: object) -> FaultDescriptor:
    if not isinstance(value, Mapping):
        raise GVSFaultError("certificate_shape", "descriptor must be an object")
    expected = {"kind", "call_index", "argument", "replacement", "semantic_path"}
    if set(value) != expected:
        raise GVSFaultError("certificate_shape", "descriptor fields are not exact")
    try:
        kind = FaultKind(value["kind"])
    except (TypeError, ValueError) as exc:
        raise GVSFaultError("certificate_shape", "unknown fault kind") from exc
    descriptor = FaultDescriptor(
        kind=kind,
        call_index=value["call_index"],
        argument=value["argument"],
        replacement=value["replacement"],
    )
    if value["semantic_path"] != descriptor.semantic_path:
        raise GVSFaultError("certificate_shape", "descriptor semantic path mismatch")
    return descriptor


def loads_fault_certification(text: str, *, state: WorldState) -> FaultCertification:
    """Load and replay a certificate against its mandatory immutable world state."""

    assert_gvs_fault_runtime_integrity()
    if not isinstance(state, WorldState):
        raise GVSFaultError("certificate_context", "certificate replay requires a WorldState")
    value = loads_strict_json_object(text)
    expected = {
        "schema_version",
        "simulator_version",
        "schema_sha256",
        "policy_contract_sha256",
        "program_contract_version",
        "world_contract_version",
        "reference_effect_version",
        "timezone_runtime_version",
        "descriptor",
        "gold_action",
        "faulty_action",
        "input_state_sha256",
        "gold_action_sha256",
        "faulty_action_sha256",
        "gold_visible_receipt_sha256",
        "faulty_visible_receipt_sha256",
        "gold_reference_receipt_sha256",
        "faulty_reference_receipt_sha256",
        "gold_counterfactual_state_sha256",
        "faulty_counterfactual_state_sha256",
        "timezone_runtime_sha256",
        "simulator_runtime_sha256",
        "fault_runtime_sha256",
        "differing_semantic_outputs",
        "schema_valid",
        "semantic_type_valid",
        "semantic_change_count",
        "reference_only",
        "authorizes_execution",
        "external_side_effects",
    }
    if set(value) != expected:
        raise GVSFaultError("certificate_shape", "certificate fields are not exact")
    identities = {
        "schema_version": GVS_FAULT_CONTRACT_VERSION,
        "simulator_version": ACTION_SIMULATOR_VERSION,
        "schema_sha256": SIMULATOR_SCHEMA_SHA256,
        "policy_contract_sha256": POLICY_CONTRACT_SHA256,
        "program_contract_version": SIM_PROGRAM_VERSION,
        "world_contract_version": WORLD_STATE_VERSION,
        "reference_effect_version": REFERENCE_EFFECT_VERSION,
        "timezone_runtime_version": TIMEZONE_RUNTIME_VERSION,
        "schema_valid": True,
        "semantic_type_valid": True,
        "semantic_change_count": 1,
        "reference_only": True,
        "authorizes_execution": False,
        "external_side_effects": False,
        "simulator_runtime_sha256": action_simulator_runtime_sha256(),
        "fault_runtime_sha256": gvs_fault_runtime_sha256(),
    }
    boolean_identity_fields = {
        "schema_valid",
        "semantic_type_valid",
        "reference_only",
        "authorizes_execution",
        "external_side_effects",
    }
    if any(type(value[key]) is not bool for key in boolean_identity_fields) or (
        type(value["semantic_change_count"]) is not int
    ):
        raise GVSFaultError(
            "certificate_identity", "certificate identity fields have invalid JSON types"
        )
    if any(
        value[key] != expected_value or type(value[key]) is not type(expected_value)
        for key, expected_value in identities.items()
    ):
        raise GVSFaultError("certificate_identity", "certificate identity mismatch")
    descriptor = _descriptor_from_value(value["descriptor"])
    schemas = simulator_tool_registry_snapshot()
    gold = parse_action_ir(canonical_json(value["gold_action"]), schemas)
    faulty = parse_action_ir(canonical_json(value["faulty_action"]), schemas)
    if _action_sha256(gold) != value["gold_action_sha256"]:
        raise GVSFaultError("certificate_hash", "gold action hash mismatch")
    if _action_sha256(faulty) != value["faulty_action_sha256"]:
        raise GVSFaultError("certificate_hash", "faulty action hash mismatch")
    raw_differences = value["differing_semantic_outputs"]
    if not isinstance(raw_differences, tuple):
        raise GVSFaultError("certificate_shape", "differences must be an array")
    certification = FaultCertification(
        descriptor=descriptor,
        gold_action=gold,
        faulty_action=faulty,
        input_state_sha256=value["input_state_sha256"],
        gold_action_sha256=value["gold_action_sha256"],
        faulty_action_sha256=value["faulty_action_sha256"],
        gold_visible_receipt_sha256=value["gold_visible_receipt_sha256"],
        faulty_visible_receipt_sha256=value["faulty_visible_receipt_sha256"],
        gold_reference_receipt_sha256=value["gold_reference_receipt_sha256"],
        faulty_reference_receipt_sha256=value["faulty_reference_receipt_sha256"],
        gold_counterfactual_state_sha256=value["gold_counterfactual_state_sha256"],
        faulty_counterfactual_state_sha256=value["faulty_counterfactual_state_sha256"],
        timezone_runtime_sha256=value["timezone_runtime_sha256"],
        simulator_runtime_sha256=value["simulator_runtime_sha256"],
        fault_runtime_sha256=value["fault_runtime_sha256"],
        differing_semantic_outputs=tuple(raw_differences),
        _factory_token=_FAULT_CERTIFICATION_FACTORY_TOKEN,
    )
    rebuilt = certify_single_fault(state, gold, descriptor)
    if rebuilt.canonical_json() != certification.canonical_json():
        raise GVSFaultError(
            "certificate_replay", "certificate does not reproduce against supplied state"
        )
    return certification


def gvs_fault_runtime_sha256() -> str:
    """Bind the exact live fault taxonomy, implementation, and simulator runtime."""

    dependencies = {
        name: json.loads(canonical_json(runtime_callable_identity(value)))
        for name, value in (
            ("ActionIR.canonical_json", ActionIR.canonical_json),
            ("action_simulator_runtime_sha256", action_simulator_runtime_sha256),
            ("hashlib.sha256", hashlib.sha256),
            ("json.dumps", json.dumps),
            ("json.loads", json.loads),
            ("parse_action_ir", parse_action_ir),
            ("simulate_reference_action", simulate_reference_action),
        )
    }
    contract = {
        "schema_version": GVS_FAULT_CONTRACT_VERSION,
        "fault_kinds": [kind.value for kind in FaultKind],
        "arguments_by_kind": {
            kind.value: sorted(arguments) for kind, arguments in _ARGUMENTS_BY_KIND.items()
        },
        "semantic_output_names": sorted(_SEMANTIC_OUTPUT_NAMES),
        "simulator_runtime_sha256": action_simulator_runtime_sha256(),
        "dependencies": dependencies,
    }
    try:
        return module_runtime_sha256(
            globals(),
            module_name=__name__,
            source_path=__file__,
            contract=contract,
        )
    except SimProgramError as exc:
        raise GVSFaultError("runtime_identity", "could not bind the GVS fault runtime") from exc


def assert_gvs_fault_runtime_integrity() -> None:
    assert_action_simulator_runtime_integrity()
    current = gvs_fault_runtime_sha256()
    if current != _PINNED_GVS_FAULT_RUNTIME_SHA256:
        raise GVSFaultError(
            "runtime_identity",
            "GVS fault runtime differs from its import-time identity",
        )


__all__ = [
    "GVS_FAULT_CONTRACT_VERSION",
    "FaultCertification",
    "FaultDescriptor",
    "FaultKind",
    "GVSFaultError",
    "assert_gvs_fault_runtime_integrity",
    "certify_single_fault",
    "gvs_fault_runtime_sha256",
    "loads_fault_certification",
    "make_single_fault",
]


_PINNED_GVS_FAULT_RUNTIME_SHA256 = gvs_fault_runtime_sha256()
