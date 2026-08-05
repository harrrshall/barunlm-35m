"""Pure, policy-safe execution and independent reference effects for GVS.

Model-visible simulation never executes external actions.  Side-effecting ``CALL`` values
are policy-blocked, while ``CONFIRM`` is proposal-only.  Counterfactual effects are available
only through explicitly named reference-only APIs and receipts that cannot authorize execution.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

from . import action_ir as _action_ir_module
from .action_ir import (
    ActionIR,
    ActionIRError,
    CallMode,
    Decision,
    JSONType,
    ToolCall,
    ToolSchema,
    ValueSchema,
    parse_action_ir,
)
from .sim_program import (
    SIM_PROGRAM_VERSION,
    TIMEZONE_RUNTIME_VERSION,
    WORLD_STATE_VERSION,
    ProgramStep,
    SemanticOperation,
    SimProgram,
    SimProgramError,
    StrictJSON,
    WorldState,
    assert_sim_program_runtime_integrity,
    canonical_identifier,
    canonical_json,
    canonical_sha256,
    canonical_timestamp,
    loads_strict_json_object,
    module_runtime_sha256,
    parse_sim_program,
    runtime_callable_identity,
    sim_program_runtime_sha256,
    snapshot_strict_json,
    timezone_runtime_sha256,
)

ACTION_SIMULATOR_VERSION = "barun-gvs-action-simulator-v5"
TRANSITION_RECEIPT_VERSION = "barun-gvs-transition-receipt-v5"
REFERENCE_EFFECT_VERSION = "barun-gvs-reference-effect-v4"
REFERENCE_RECEIPT_VERSION = "barun-gvs-reference-receipt-v4"
PROGRAM_ROUND_TRIP_VERSION = "barun-gvs-program-round-trip-v4"
POLICY_CONTRACT_VERSION = "barun-gvs-policy-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STRING = ValueSchema(JSONType.STRING)
_BOOLEAN = ValueSchema(JSONType.BOOLEAN)
_ROUTE_MODE = ValueSchema(
    JSONType.STRING,
    enum=("cycling", "driving", "transit", "walking"),
)
_TRANSITION_RECEIPT_FACTORY_TOKEN = object()
_REFERENCE_RECEIPT_FACTORY_TOKEN = object()
_ROUND_TRIP_RECEIPT_FACTORY_TOKEN = object()


def _tool(
    name: str,
    arguments: tuple[tuple[str, ValueSchema], ...],
    *,
    side_effecting: bool,
) -> ToolSchema:
    names = frozenset(item[0] for item in arguments)
    return ToolSchema(
        name=name,
        arguments=dict(arguments),
        required=names,
        additional_arguments=False,
        side_effecting=side_effecting,
    )


SIMULATOR_TOOL_SCHEMAS = (
    _tool(
        "reminder_create",
        (("reminder_id", _STRING), ("title", _STRING), ("due_at", _STRING)),
        side_effecting=True,
    ),
    _tool(
        "reminder_update",
        (("reminder_id", _STRING), ("title", _STRING), ("due_at", _STRING)),
        side_effecting=True,
    ),
    _tool(
        "calendar_create",
        (
            ("event_id", _STRING),
            ("title", _STRING),
            ("start_at", _STRING),
            ("end_at", _STRING),
        ),
        side_effecting=True,
    ),
    _tool(
        "calendar_reschedule",
        (("event_id", _STRING), ("start_at", _STRING), ("end_at", _STRING)),
        side_effecting=True,
    ),
    _tool("contact_lookup", (("contact_id", _STRING),), side_effecting=False),
    _tool(
        "note_create",
        (("note_id", _STRING), ("title", _STRING), ("body", _STRING)),
        side_effecting=True,
    ),
    _tool(
        "list_add_item",
        (("list_id", _STRING), ("item_id", _STRING), ("text", _STRING)),
        side_effecting=True,
    ),
    _tool(
        "list_check_item",
        (("list_id", _STRING), ("item_id", _STRING), ("checked", _BOOLEAN)),
        side_effecting=True,
    ),
    _tool(
        "map_route",
        (("origin_id", _STRING), ("destination_id", _STRING), ("mode", _ROUTE_MODE)),
        side_effecting=False,
    ),
    _tool(
        "message_send",
        (
            ("message_id", _STRING),
            ("to_contact_id", _STRING),
            ("channel", _STRING),
            ("body", _STRING),
        ),
        side_effecting=True,
    ),
    _tool("media_play", (("track_id", _STRING),), side_effecting=True),
    _tool("media_pause", (), side_effecting=True),
    _tool(
        "setting_set_bool",
        (("key", _STRING), ("enabled", _BOOLEAN)),
        side_effecting=True,
    ),
)
SIMULATOR_TOOL_REGISTRY = MappingProxyType(
    {schema.name: schema for schema in SIMULATOR_TOOL_SCHEMAS}
)


def _schema_record(schema: ToolSchema) -> dict[str, Any]:
    return {
        "name": schema.name,
        "arguments": {
            name: {
                "kind": value.kind.value,
                "enum": list(value.enum) if value.enum is not None else None,
            }
            for name, value in schema.arguments.items()
        },
        "required": sorted(schema.required),
        "additional_arguments": schema.additional_arguments,
        "side_effecting": schema.side_effecting,
    }


SIMULATOR_SCHEMA_BYTES = canonical_json(
    snapshot_strict_json([_schema_record(schema) for schema in SIMULATOR_TOOL_SCHEMAS])
)
SIMULATOR_SCHEMA_SHA256 = hashlib.sha256(SIMULATOR_SCHEMA_BYTES.encode("utf-8")).hexdigest()


def simulator_tool_registry_snapshot(
    schema_bytes: str = SIMULATOR_SCHEMA_BYTES,
    schema_sha256: str = SIMULATOR_SCHEMA_SHA256,
) -> Mapping[str, ToolSchema]:
    """Rebuild a private registry from the import-time canonical schema snapshot."""

    if hashlib.sha256(schema_bytes.encode("utf-8")).hexdigest() != schema_sha256:
        raise ActionSimulationError(
            "runtime_identity", "captured simulator schema bytes do not match their hash"
        )
    try:
        records = json.loads(schema_bytes)
        schemas: list[ToolSchema] = []
        for record in records:
            arguments = {
                name: ValueSchema(
                    JSONType(specification["kind"]),
                    enum=(
                        tuple(specification["enum"]) if specification["enum"] is not None else None
                    ),
                )
                for name, specification in record["arguments"].items()
            }
            schemas.append(
                ToolSchema(
                    name=record["name"],
                    arguments=arguments,
                    required=frozenset(record["required"]),
                    additional_arguments=record["additional_arguments"],
                    side_effecting=record["side_effecting"],
                )
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ActionSimulationError(
            "runtime_identity", "captured simulator schema bytes are malformed"
        ) from exc
    registry = MappingProxyType({schema.name: schema for schema in schemas})
    if canonical_json(snapshot_strict_json([_schema_record(schema) for schema in schemas])) != (
        schema_bytes
    ):
        raise ActionSimulationError(
            "runtime_identity", "rebuilt simulator schema is not byte-identical"
        )
    return registry


_POLICY_RECORD = snapshot_strict_json(
    {
        "schema_version": POLICY_CONTRACT_VERSION,
        "safe_calls_require": "CALL",
        "side_effecting_calls_require": "CONFIRM",
        "confirm_is_proposal_only": True,
        "side_effecting_call_is_blocked": True,
        "external_execution_permitted": False,
        "reference_counterfactual_authorizes_execution": False,
    }
)
POLICY_CONTRACT_SHA256 = canonical_sha256(_POLICY_RECORD)
_POLICY_FIELDS = frozenset(
    {
        "policy_contract_sha256",
        "policy_valid",
        "expected_decision",
        "authorization_required",
        "confirmation_required",
        "execution_permitted",
        "simulation_transition_permitted",
        "reference_counterfactual_permitted",
        "proposed_call_count",
        "side_effecting_tools",
        "reason_codes",
    }
)


class ActionSimulationError(ValueError):
    """Stable semantic transition error; a batch containing one is rolled back."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(f"{path}: {message}")
        self.code = code
        self.path = path


def _mutable(value: StrictJSON) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_mutable(child) for child in value]
    return value


def _require_sha(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ActionSimulationError("invalid_sha256", "must be a lowercase SHA-256", path)
    return value


def _require_exact_bool(value: object, *, path: str) -> bool:
    if type(value) is not bool:
        raise ActionSimulationError("type_mismatch", "must be a JSON boolean", path)
    return value


def _require_error_field(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ActionSimulationError("type_mismatch", "must be null or a non-empty string", path)
    return value


def _validate_policy_record(policy: Mapping[str, StrictJSON], decision: Decision) -> None:
    """Validate the complete policy relation instead of trusting self-hashed booleans."""

    if set(policy) != _POLICY_FIELDS:
        raise ActionSimulationError("policy_shape", "policy fields are not exact", "$.policy")
    if policy["policy_contract_sha256"] != POLICY_CONTRACT_SHA256:
        raise ActionSimulationError("policy_identity", "policy contract hash mismatch", "$.policy")
    boolean_fields = (
        "policy_valid",
        "authorization_required",
        "confirmation_required",
        "execution_permitted",
        "simulation_transition_permitted",
        "reference_counterfactual_permitted",
    )
    for name in boolean_fields:
        _require_exact_bool(policy[name], path=f"$.policy.{name}")
    proposed = policy["proposed_call_count"]
    if type(proposed) is not int or proposed < 0:
        raise ActionSimulationError(
            "policy_shape",
            "proposed_call_count must be a non-negative integer",
            "$.policy.proposed_call_count",
        )
    side_effecting = policy["side_effecting_tools"]
    reasons = policy["reason_codes"]
    if not isinstance(side_effecting, tuple) or any(
        type(tool) is not str for tool in side_effecting
    ):
        raise ActionSimulationError(
            "policy_shape", "side_effecting_tools must be an array of strings", "$.policy"
        )
    if side_effecting != tuple(sorted(set(side_effecting))):
        raise ActionSimulationError(
            "policy_shape", "side_effecting_tools must be sorted and unique", "$.policy"
        )
    schemas = simulator_tool_registry_snapshot()
    if any(tool not in schemas or not schemas[tool].side_effecting for tool in side_effecting):
        raise ActionSimulationError(
            "policy_shape", "side_effecting_tools contains an invalid tool", "$.policy"
        )
    if not isinstance(reasons, tuple) or any(type(reason) is not str for reason in reasons):
        raise ActionSimulationError(
            "policy_shape", "reason_codes must be an array of strings", "$.policy"
        )

    schema_rejected = reasons == ("schema_rejected",)
    if schema_rejected:
        expected = {
            "policy_valid": False,
            "expected_decision": None,
            "authorization_required": False,
            "confirmation_required": False,
            "execution_permitted": False,
            "simulation_transition_permitted": False,
            "reference_counterfactual_permitted": False,
            "proposed_call_count": 0,
            "side_effecting_tools": (),
        }
    else:
        has_calls = proposed > 0
        if has_calls != (decision in (Decision.CALL, Decision.CONFIRM)):
            raise ActionSimulationError(
                "policy_relation",
                "decision and proposed_call_count disagree",
                "$.policy.proposed_call_count",
            )
        expected_decision = (
            (Decision.CONFIRM.value if side_effecting else Decision.CALL.value)
            if has_calls
            else None
        )
        policy_valid = not has_calls or decision.value == expected_decision
        expected_reasons: tuple[str, ...]
        if not has_calls:
            expected_reasons = ()
        elif policy_valid:
            expected_reasons = ("model_output_is_proposal_only",)
        else:
            expected_reasons = (
                "model_output_is_proposal_only",
                (
                    "side_effect_requires_confirmation"
                    if side_effecting
                    else "safe_action_must_not_confirm"
                ),
            )
        if reasons != expected_reasons:
            raise ActionSimulationError(
                "policy_relation", "reason_codes disagree with policy", "$.policy.reason_codes"
            )
        expected = {
            "policy_valid": policy_valid,
            "expected_decision": expected_decision,
            "authorization_required": has_calls,
            "confirmation_required": bool(side_effecting),
            "execution_permitted": False,
            "simulation_transition_permitted": (
                has_calls and policy_valid and decision is Decision.CALL
            ),
            "reference_counterfactual_permitted": has_calls,
            "proposed_call_count": proposed,
            "side_effecting_tools": side_effecting,
        }
    for name, expected_value in expected.items():
        if policy[name] != expected_value or type(policy[name]) is not type(expected_value):
            raise ActionSimulationError(
                "policy_relation", f"{name} disagrees with policy semantics", f"$.policy.{name}"
            )


@dataclass(frozen=True, slots=True)
class Effect:
    kind: str
    path: tuple[str, ...]
    before: StrictJSON
    after: StrictJSON

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise TypeError("effect kind must be a non-empty string")
        path = tuple(self.path)
        if not path or any(not isinstance(part, str) or not part for part in path):
            raise TypeError("effect path must contain non-empty string components")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "before", snapshot_strict_json(self.before))
        object.__setattr__(self, "after", snapshot_strict_json(self.after))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": list(self.path),
            "before": _mutable(self.before),
            "after": _mutable(self.after),
        }


def _effect_from_value(value: object, *, path: str) -> Effect:
    if not isinstance(value, Mapping):
        raise ActionSimulationError("receipt_shape", "effect must be an object", path)
    expected = {"kind", "path", "before", "after"}
    if set(value) != expected:
        raise ActionSimulationError("receipt_shape", "effect fields are not exact", path)
    raw_path = value["path"]
    if not isinstance(raw_path, tuple):
        raise ActionSimulationError("receipt_shape", "effect path must be an array", f"{path}.path")
    return Effect(
        kind=value["kind"],
        path=tuple(raw_path),
        before=value["before"],
        after=value["after"],
    )


_TRANSITION_STATUSES = frozenset(
    {"simulated", "no_action", "confirmation_required", "policy_blocked", "rejected"}
)


@dataclass(frozen=True, slots=True)
class TransitionReceipt:
    status: str
    decision: Decision
    input_state_sha256: str
    action_ir_sha256: str
    output_state_sha256: str
    effects: tuple[Effect, ...]
    effects_sha256: str
    observation: StrictJSON
    observation_sha256: str
    policy: StrictJSON
    policy_sha256: str
    timezone_runtime_sha256: str
    simulator_runtime_sha256: str
    error_code: str | None = None
    error_path: str | None = None
    external_side_effects: bool = False
    authorizes_execution: bool = False
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _TRANSITION_RECEIPT_FACTORY_TOKEN:
            raise TypeError("TransitionReceipt must be constructed by the simulator or loader")
        if self.status not in _TRANSITION_STATUSES:
            raise ValueError("unknown transition status")
        if not isinstance(self.decision, Decision):
            raise TypeError("decision must be Decision")
        for name in (
            "input_state_sha256",
            "action_ir_sha256",
            "output_state_sha256",
            "effects_sha256",
            "observation_sha256",
            "policy_sha256",
            "timezone_runtime_sha256",
            "simulator_runtime_sha256",
        ):
            _require_sha(getattr(self, name), path=f"$.{name}")
        if self.simulator_runtime_sha256 != action_simulator_runtime_sha256():
            raise ValueError("simulator runtime hash mismatch")
        effects = tuple(self.effects)
        if any(not isinstance(effect, Effect) for effect in effects):
            raise TypeError("effects must contain Effect values")
        observation = snapshot_strict_json(self.observation)
        policy = snapshot_strict_json(self.policy)
        if not isinstance(observation, Mapping) or not isinstance(policy, Mapping):
            raise TypeError("observation and policy must be strict JSON objects")
        _validate_policy_record(policy, self.decision)
        if (
            canonical_sha256(snapshot_strict_json([effect.to_dict() for effect in effects]))
            != self.effects_sha256
        ):
            raise ValueError("effects hash mismatch")
        if canonical_sha256(observation) != self.observation_sha256:
            raise ValueError("observation hash mismatch")
        if canonical_sha256(policy) != self.policy_sha256:
            raise ValueError("policy hash mismatch")
        _require_exact_bool(self.external_side_effects, path="$.external_side_effects")
        _require_exact_bool(self.authorizes_execution, path="$.authorizes_execution")
        if self.external_side_effects is not False or self.authorizes_execution is not False:
            raise ValueError("simulator receipts must remain non-authorizing and side-effect-free")
        error_code = _require_error_field(self.error_code, path="$.error_code")
        error_path = _require_error_field(self.error_path, path="$.error_path")
        blocked_status = self.status in {
            "no_action",
            "confirmation_required",
            "policy_blocked",
            "rejected",
        }
        if blocked_status and (effects or self.input_state_sha256 != self.output_state_sha256):
            raise ValueError("non-transition receipt mutated state or retained effects")
        failed_status = self.status in {"policy_blocked", "rejected"}
        if failed_status != (error_code is not None and error_path is not None):
            raise ValueError("transition error fields do not match status")
        if (error_code is None) != (error_path is None):
            raise ValueError("transition error code and path must be present together")

        schema_rejected = policy["reason_codes"] == ("schema_rejected",)
        if schema_rejected:
            allowed_statuses = {"rejected"}
        elif self.decision in (Decision.ABSTAIN, Decision.CLARIFY):
            allowed_statuses = {"no_action"}
        elif policy["policy_valid"] is False:
            allowed_statuses = {"policy_blocked"}
        elif self.decision is Decision.CONFIRM:
            allowed_statuses = {"confirmation_required", "rejected"}
        else:
            allowed_statuses = {"simulated", "rejected"}
        if self.status not in allowed_statuses:
            raise ValueError("transition status disagrees with decision and policy")
        expected_observation_kind = {
            "simulated": "call_result",
            "confirmation_required": "confirm",
            "policy_blocked": "policy_blocked",
            "rejected": "rejected",
        }.get(self.status)
        if self.status == "no_action":
            expected_observation_kind = (
                "abstain" if self.decision is Decision.ABSTAIN else "clarify"
            )
        if observation.get("kind") != expected_observation_kind:
            raise ValueError("observation kind disagrees with transition status")
        if self.status == "simulated" and not effects:
            raise ValueError("simulated calls must retain their semantic effects")
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "error_code", error_code)
        object.__setattr__(self, "error_path", error_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRANSITION_RECEIPT_VERSION,
            "simulator_version": ACTION_SIMULATOR_VERSION,
            "schema_sha256": SIMULATOR_SCHEMA_SHA256,
            "policy_contract_sha256": POLICY_CONTRACT_SHA256,
            "program_contract_version": SIM_PROGRAM_VERSION,
            "world_contract_version": WORLD_STATE_VERSION,
            "reference_effect_version": REFERENCE_EFFECT_VERSION,
            "timezone_runtime_version": TIMEZONE_RUNTIME_VERSION,
            "status": self.status,
            "decision": self.decision.value,
            "input_state_sha256": self.input_state_sha256,
            "action_ir_sha256": self.action_ir_sha256,
            "output_state_sha256": self.output_state_sha256,
            "effects": [effect.to_dict() for effect in self.effects],
            "effects_sha256": self.effects_sha256,
            "observation": _mutable(self.observation),
            "observation_sha256": self.observation_sha256,
            "policy": _mutable(self.policy),
            "policy_sha256": self.policy_sha256,
            "timezone_runtime_sha256": self.timezone_runtime_sha256,
            "simulator_runtime_sha256": self.simulator_runtime_sha256,
            "error_code": self.error_code,
            "error_path": self.error_path,
            "external_side_effects": self.external_side_effects,
            "authorizes_execution": self.authorizes_execution,
        }

    def canonical_json(self) -> str:
        return canonical_json(snapshot_strict_json(self.to_dict()))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SimulationResult:
    state: WorldState
    receipt: TransitionReceipt


@dataclass(frozen=True, slots=True)
class ReferenceEffectReceipt:
    status: str
    source_kind: str
    source_sha256: str
    program_sha256: str | None
    step_id: str | None
    input_state_sha256: str
    counterfactual_state_sha256: str
    effects: tuple[Effect, ...]
    effects_sha256: str
    observation: StrictJSON
    observation_sha256: str
    timezone_runtime_sha256: str
    simulator_runtime_sha256: str
    error_code: str | None = None
    error_path: str | None = None
    reference_only: bool = True
    authorizes_execution: bool = False
    external_side_effects: bool = False
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _REFERENCE_RECEIPT_FACTORY_TOKEN:
            raise TypeError(
                "ReferenceEffectReceipt must be constructed by the reference simulator or loader"
            )
        if self.status not in {"reference_applied", "reference_control", "reference_rejected"}:
            raise ValueError("unknown reference status")
        if self.source_kind not in {"compiled_action", "semantic_program"}:
            raise ValueError("unknown reference source kind")
        for name in (
            "source_sha256",
            "input_state_sha256",
            "counterfactual_state_sha256",
            "effects_sha256",
            "observation_sha256",
            "timezone_runtime_sha256",
            "simulator_runtime_sha256",
        ):
            _require_sha(getattr(self, name), path=f"$.{name}")
        if self.simulator_runtime_sha256 != action_simulator_runtime_sha256():
            raise ValueError("simulator runtime hash mismatch")
        if self.program_sha256 is not None:
            _require_sha(self.program_sha256, path="$.program_sha256")
        effects = tuple(self.effects)
        observation = snapshot_strict_json(self.observation)
        if any(not isinstance(effect, Effect) for effect in effects):
            raise TypeError("reference effects must contain Effect values")
        if not isinstance(observation, Mapping):
            raise TypeError("reference observation must be a strict JSON object")
        if (
            canonical_sha256(snapshot_strict_json([effect.to_dict() for effect in effects]))
            != self.effects_sha256
        ):
            raise ValueError("reference effects hash mismatch")
        if canonical_sha256(observation) != self.observation_sha256:
            raise ValueError("reference observation hash mismatch")
        _require_exact_bool(self.reference_only, path="$.reference_only")
        _require_exact_bool(self.authorizes_execution, path="$.authorizes_execution")
        _require_exact_bool(self.external_side_effects, path="$.external_side_effects")
        if (
            self.reference_only is not True
            or self.authorizes_execution is not False
            or self.external_side_effects is not False
        ):
            raise ValueError("reference receipts must remain reference-only and non-authorizing")
        if self.source_kind == "semantic_program":
            if self.program_sha256 is None or type(self.step_id) is not str or not self.step_id:
                raise ValueError("semantic-program reference requires program hash and step ID")
        elif self.program_sha256 is not None or self.step_id is not None:
            raise ValueError("compiled-action reference forbids program hash and step ID")
        error_code = _require_error_field(self.error_code, path="$.error_code")
        error_path = _require_error_field(self.error_path, path="$.error_path")
        failed = self.status == "reference_rejected"
        if failed != (error_code is not None and error_path is not None):
            raise ValueError("reference error fields do not match status")
        if (error_code is None) != (error_path is None):
            raise ValueError("reference error code and path must be present together")
        if self.status in {"reference_control", "reference_rejected"} and (
            effects or self.input_state_sha256 != self.counterfactual_state_sha256
        ):
            raise ValueError("non-applied reference receipt mutated state or retained effects")
        expected_observation_kind = {
            "reference_applied": "reference_effects",
            "reference_control": "reference_control",
            "reference_rejected": "reference_rejected",
        }[self.status]
        if observation.get("kind") != expected_observation_kind:
            raise ValueError("reference observation kind disagrees with status")
        if self.status == "reference_applied" and not effects:
            raise ValueError("applied references must retain their semantic effects")
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "error_code", error_code)
        object.__setattr__(self, "error_path", error_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REFERENCE_RECEIPT_VERSION,
            "reference_effect_version": REFERENCE_EFFECT_VERSION,
            "simulator_version": ACTION_SIMULATOR_VERSION,
            "schema_sha256": SIMULATOR_SCHEMA_SHA256,
            "policy_contract_sha256": POLICY_CONTRACT_SHA256,
            "program_contract_version": SIM_PROGRAM_VERSION,
            "world_contract_version": WORLD_STATE_VERSION,
            "timezone_runtime_version": TIMEZONE_RUNTIME_VERSION,
            "status": self.status,
            "source_kind": self.source_kind,
            "source_sha256": self.source_sha256,
            "program_sha256": self.program_sha256,
            "step_id": self.step_id,
            "input_state_sha256": self.input_state_sha256,
            "counterfactual_state_sha256": self.counterfactual_state_sha256,
            "effects": [effect.to_dict() for effect in self.effects],
            "effects_sha256": self.effects_sha256,
            "observation": _mutable(self.observation),
            "observation_sha256": self.observation_sha256,
            "timezone_runtime_sha256": self.timezone_runtime_sha256,
            "simulator_runtime_sha256": self.simulator_runtime_sha256,
            "error_code": self.error_code,
            "error_path": self.error_path,
            "reference_only": self.reference_only,
            "authorizes_execution": self.authorizes_execution,
            "external_side_effects": self.external_side_effects,
        }

    def canonical_json(self) -> str:
        return canonical_json(snapshot_strict_json(self.to_dict()))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ReferenceSimulationResult:
    counterfactual_state: WorldState
    receipt: ReferenceEffectReceipt


@dataclass(frozen=True, slots=True)
class ProgramSimulation:
    final_state: WorldState
    actions: tuple[ActionIR, ...]
    results: tuple[SimulationResult, ...]


@dataclass(frozen=True, slots=True)
class ProgramRoundTripReceipt:
    program_sha256: str
    final_state_sha256: str
    step_records: tuple[StrictJSON, ...]
    timezone_runtime_sha256: str
    simulator_runtime_sha256: str
    passed: bool
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _ROUND_TRIP_RECEIPT_FACTORY_TOKEN:
            raise TypeError("ProgramRoundTripReceipt must be constructed by the verifier or loader")
        _require_exact_bool(self.passed, path="$.passed")
        _require_sha(self.program_sha256, path="$.program_sha256")
        _require_sha(self.final_state_sha256, path="$.final_state_sha256")
        _require_sha(self.timezone_runtime_sha256, path="$.timezone_runtime_sha256")
        _require_sha(self.simulator_runtime_sha256, path="$.simulator_runtime_sha256")
        if self.simulator_runtime_sha256 != action_simulator_runtime_sha256():
            raise ValueError("simulator runtime hash mismatch")
        records = tuple(snapshot_strict_json(record) for record in self.step_records)
        expected_fields = {
            "step_id",
            "action_ir_sha256",
            "visible_receipt_sha256",
            "compiled_reference_receipt_sha256",
            "semantic_reference_receipt_sha256",
            "reference_match",
            "visible_policy_safe",
            "passed",
        }
        step_ids: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping) or set(record) != expected_fields:
                raise ValueError(f"step record {index} fields are not exact")
            step_id = record["step_id"]
            if type(step_id) is not str or not step_id or step_id in step_ids:
                raise ValueError("step record IDs must be non-empty and unique")
            step_ids.add(step_id)
            for name in (
                "action_ir_sha256",
                "visible_receipt_sha256",
                "compiled_reference_receipt_sha256",
                "semantic_reference_receipt_sha256",
            ):
                _require_sha(record[name], path=f"$.step_records[{index}].{name}")
            for name in ("reference_match", "visible_policy_safe", "passed"):
                if type(record[name]) is not bool:
                    raise ValueError(f"step record {index} {name} must be boolean")
            if record["passed"] is not (
                record["reference_match"] and record["visible_policy_safe"]
            ):
                raise ValueError("step record pass flag is inconsistent")
        if not records or self.passed is not all(record["passed"] for record in records):
            raise ValueError("aggregate pass flag is inconsistent")
        object.__setattr__(self, "step_records", records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROGRAM_ROUND_TRIP_VERSION,
            "simulator_version": ACTION_SIMULATOR_VERSION,
            "reference_effect_version": REFERENCE_EFFECT_VERSION,
            "schema_sha256": SIMULATOR_SCHEMA_SHA256,
            "policy_contract_sha256": POLICY_CONTRACT_SHA256,
            "program_contract_version": SIM_PROGRAM_VERSION,
            "world_contract_version": WORLD_STATE_VERSION,
            "program_sha256": self.program_sha256,
            "final_state_sha256": self.final_state_sha256,
            "step_records": [_mutable(record) for record in self.step_records],
            "timezone_runtime_sha256": self.timezone_runtime_sha256,
            "simulator_runtime_sha256": self.simulator_runtime_sha256,
            "passed": self.passed,
            "authorizes_execution": False,
            "external_side_effects": False,
        }

    def canonical_json(self) -> str:
        return canonical_json(snapshot_strict_json(self.to_dict()))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _policy(action: ActionIR, schemas: Mapping[str, ToolSchema]) -> StrictJSON:
    calls = action.calls if action.decision in (Decision.CALL, Decision.CONFIRM) else ()
    side_effecting = tuple(
        sorted({call.tool for call in calls if schemas[call.tool].side_effecting})
    )
    reasons: list[str] = ["model_output_is_proposal_only"] if calls else []
    expected_decision: str | None = None
    policy_valid = True
    if calls:
        expected_decision = Decision.CONFIRM.value if side_effecting else Decision.CALL.value
        if action.decision.value != expected_decision:
            policy_valid = False
            reasons.append(
                "side_effect_requires_confirmation"
                if side_effecting
                else "safe_action_must_not_confirm"
            )
    return snapshot_strict_json(
        {
            "policy_contract_sha256": POLICY_CONTRACT_SHA256,
            "policy_valid": policy_valid,
            "expected_decision": expected_decision,
            "authorization_required": bool(calls),
            "confirmation_required": bool(side_effecting),
            "execution_permitted": False,
            "simulation_transition_permitted": bool(calls)
            and policy_valid
            and action.decision is Decision.CALL,
            "reference_counterfactual_permitted": bool(calls),
            "proposed_call_count": len(calls),
            "side_effecting_tools": list(side_effecting),
            "reason_codes": reasons,
        }
    )


_SEMANTIC_TO_TOOL = MappingProxyType(
    {
        "ADD_LIST_ITEM": "list_add_item",
        "CREATE_CALENDAR_EVENT": "calendar_create",
        "CREATE_NOTE": "note_create",
        "CREATE_REMINDER": "reminder_create",
        "LOOK_UP_CONTACT": "contact_lookup",
        "LOOK_UP_ROUTE": "map_route",
        "PAUSE_MEDIA": "media_pause",
        "PLAY_MEDIA": "media_play",
        "PROPOSE_MESSAGE": "message_send",
        "RESCHEDULE_CALENDAR_EVENT": "calendar_reschedule",
        "SET_BOOLEAN_SETTING": "setting_set_bool",
        "SET_LIST_ITEM_CHECKED": "list_check_item",
        "UPDATE_REMINDER": "reminder_update",
    }
)


def _compile_operation(operation: SemanticOperation) -> dict[str, Any]:
    parameters = operation.parameters
    kind = operation.kind
    if kind in {"CREATE_REMINDER", "UPDATE_REMINDER"}:
        args = {
            "reminder_id": parameters["reminder_ref"],
            "title": parameters["summary"],
            "due_at": parameters["due_at"],
        }
    elif kind == "CREATE_CALENDAR_EVENT":
        args = {
            "event_id": parameters["event_ref"],
            "title": parameters["summary"],
            "start_at": parameters["begins_at"],
            "end_at": parameters["ends_at"],
        }
    elif kind == "RESCHEDULE_CALENDAR_EVENT":
        args = {
            "event_id": parameters["event_ref"],
            "start_at": parameters["begins_at"],
            "end_at": parameters["ends_at"],
        }
    elif kind == "LOOK_UP_CONTACT":
        args = {"contact_id": parameters["contact_ref"]}
    elif kind == "CREATE_NOTE":
        args = {
            "note_id": parameters["note_ref"],
            "title": parameters["heading"],
            "body": parameters["content"],
        }
    elif kind == "ADD_LIST_ITEM":
        args = {
            "list_id": parameters["list_ref"],
            "item_id": parameters["item_ref"],
            "text": parameters["item_text"],
        }
    elif kind == "SET_LIST_ITEM_CHECKED":
        args = {
            "list_id": parameters["list_ref"],
            "item_id": parameters["item_ref"],
            "checked": parameters["is_checked"],
        }
    elif kind == "LOOK_UP_ROUTE":
        args = {
            "origin_id": parameters["from_place_ref"],
            "destination_id": parameters["to_place_ref"],
            "mode": parameters["travel_mode"],
        }
    elif kind == "PROPOSE_MESSAGE":
        args = {
            "message_id": parameters["message_ref"],
            "to_contact_id": parameters["recipient_ref"],
            "channel": parameters["delivery_channel"],
            "body": parameters["content"],
        }
    elif kind == "PLAY_MEDIA":
        args = {"track_id": parameters["track_ref"]}
    elif kind == "PAUSE_MEDIA":
        args = {}
    elif kind == "SET_BOOLEAN_SETTING":
        args = {"key": parameters["setting_ref"], "enabled": parameters["value"]}
    else:  # pragma: no cover - SemanticOperation validation closes this branch.
        raise ActionSimulationError("unsupported_semantic_operation", f"unknown kind {kind!r}")
    return {"tool": _SEMANTIC_TO_TOOL[kind], "args": args}


def compile_program_step(step: ProgramStep) -> ActionIR:
    """Compile the independent domain vocabulary through canonical Action IR validation."""

    assert_action_simulator_runtime_integrity()
    if not isinstance(step, ProgramStep):
        raise TypeError("step must be ProgramStep")
    if step.decision in (Decision.CALL, Decision.CONFIRM):
        payload: dict[str, Any] = {
            "decision": step.decision.value,
            "mode": step.mode.value if step.mode is not None else None,
            "calls": [_compile_operation(operation) for operation in step.operations],
        }
    elif step.decision is Decision.CLARIFY:
        payload = {"decision": step.decision.value, "missing": list(step.missing)}
    else:
        payload = {"decision": step.decision.value}
    return parse_action_ir(
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ),
        simulator_tool_registry_snapshot(),
    )


def _require_id(value: StrictJSON, path: str) -> str:
    try:
        return canonical_identifier(value, path=path)
    except SimProgramError as exc:
        raise ActionSimulationError(exc.code, str(exc).split(": ", 1)[-1], path) from exc


def _require_text(value: StrictJSON, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ActionSimulationError("invalid_text", "text has invalid shape", path)
    return value


def _require_bool(value: StrictJSON, path: str) -> bool:
    if type(value) is not bool:
        raise ActionSimulationError("invalid_boolean", "value must be boolean", path)
    return value


def _require_timestamp(state: Mapping[str, Any], value: StrictJSON, path: str) -> str:
    try:
        return canonical_timestamp(value, timezone=state["timezone"], path=path)
    except SimProgramError as exc:
        raise ActionSimulationError(exc.code, str(exc).split(": ", 1)[-1], path) from exc


def _effect(kind: str, path: tuple[str, ...], before: object, after: object) -> Effect:
    return Effect(kind=kind, path=path, before=before, after=after)


def _entity_observation(tool: str, entity_id: str, value: object) -> StrictJSON:
    return snapshot_strict_json(
        {"kind": "result", "tool": tool, "entity_id": entity_id, "value": value}
    )


def _apply_call(
    state: dict[str, Any], call: ToolCall, call_index: int
) -> tuple[Effect, StrictJSON]:
    args = call.args
    base_path = f"$.calls[{call_index}]"
    tool = call.tool
    if tool in {"reminder_create", "reminder_update"}:
        reminder_id = _require_id(args["reminder_id"], f"{base_path}.args.reminder_id")
        reminders = state["reminders"]
        exists = reminder_id in reminders
        if tool == "reminder_create" and exists:
            raise ActionSimulationError("entity_exists", "reminder already exists", base_path)
        if tool == "reminder_update" and not exists:
            raise ActionSimulationError("entity_missing", "reminder does not exist", base_path)
        before = reminders.get(reminder_id)
        after = {
            "title": _require_text(args["title"], f"{base_path}.args.title"),
            "due_at": _require_timestamp(state, args["due_at"], f"{base_path}.args.due_at"),
            "completed": before["completed"] if before is not None else False,
        }
        reminders[reminder_id] = after
        path = ("reminders", reminder_id)
        return _effect("upsert", path, before, after), _entity_observation(tool, reminder_id, after)

    if tool in {"calendar_create", "calendar_reschedule"}:
        event_id = _require_id(args["event_id"], f"{base_path}.args.event_id")
        calendar = state["calendar"]
        exists = event_id in calendar
        if tool == "calendar_create" and exists:
            raise ActionSimulationError("entity_exists", "calendar event already exists", base_path)
        if tool == "calendar_reschedule" and not exists:
            raise ActionSimulationError(
                "entity_missing", "calendar event does not exist", base_path
            )
        before = calendar.get(event_id)
        start_at = _require_timestamp(state, args["start_at"], f"{base_path}.args.start_at")
        end_at = _require_timestamp(state, args["end_at"], f"{base_path}.args.end_at")
        if datetime.fromisoformat(end_at) <= datetime.fromisoformat(start_at):
            raise ActionSimulationError(
                "invalid_interval", "end_at must be after start_at", base_path
            )
        after = {
            "title": (
                _require_text(args["title"], f"{base_path}.args.title")
                if tool == "calendar_create"
                else before["title"]
            ),
            "start_at": start_at,
            "end_at": end_at,
        }
        calendar[event_id] = after
        path = ("calendar", event_id)
        return _effect("upsert", path, before, after), _entity_observation(tool, event_id, after)

    if tool == "contact_lookup":
        contact_id = _require_id(args["contact_id"], f"{base_path}.args.contact_id")
        contact = state["contacts"].get(contact_id)
        if contact is None:
            raise ActionSimulationError("entity_missing", "contact does not exist", base_path)
        path = ("contacts", contact_id)
        return _effect("observe", path, contact, contact), _entity_observation(
            tool, contact_id, contact
        )

    if tool == "note_create":
        note_id = _require_id(args["note_id"], f"{base_path}.args.note_id")
        notes = state["notes"]
        if note_id in notes:
            raise ActionSimulationError("entity_exists", "note already exists", base_path)
        note = {
            "title": _require_text(args["title"], f"{base_path}.args.title"),
            "body": _require_text(args["body"], f"{base_path}.args.body", nonempty=False),
        }
        notes[note_id] = note
        path = ("notes", note_id)
        return _effect("create", path, None, note), _entity_observation(tool, note_id, note)

    if tool in {"list_add_item", "list_check_item"}:
        list_id = _require_id(args["list_id"], f"{base_path}.args.list_id")
        item_id = _require_id(args["item_id"], f"{base_path}.args.item_id")
        list_value = state["lists"].get(list_id)
        if list_value is None:
            raise ActionSimulationError("entity_missing", "list does not exist", base_path)
        items = list_value["items"]
        before = items.get(item_id)
        if tool == "list_add_item":
            if before is not None:
                raise ActionSimulationError("entity_exists", "list item already exists", base_path)
            after = {
                "text": _require_text(args["text"], f"{base_path}.args.text"),
                "checked": False,
            }
        else:
            if before is None:
                raise ActionSimulationError("entity_missing", "list item does not exist", base_path)
            after = {
                "text": before["text"],
                "checked": _require_bool(args["checked"], f"{base_path}.args.checked"),
            }
        items[item_id] = after
        path = ("lists", list_id, "items", item_id)
        return _effect("upsert", path, before, after), _entity_observation(tool, item_id, after)

    if tool == "map_route":
        origin_id = _require_id(args["origin_id"], f"{base_path}.args.origin_id")
        destination_id = _require_id(args["destination_id"], f"{base_path}.args.destination_id")
        mode = _require_text(args["mode"], f"{base_path}.args.mode")
        matches = [
            (route_id, route)
            for route_id, route in state["routes"].items()
            if route["origin_id"] == origin_id
            and route["destination_id"] == destination_id
            and route["mode"] == mode
        ]
        if not matches:
            raise ActionSimulationError("route_missing", "route does not exist", base_path)
        if len(matches) != 1:
            raise ActionSimulationError("route_ambiguous", "route is not unique", base_path)
        route_id, route = matches[0]
        path = ("routes", route_id)
        return _effect("observe", path, route, route), _entity_observation(tool, route_id, route)

    if tool == "message_send":
        message_id = _require_id(args["message_id"], f"{base_path}.args.message_id")
        if any(message["message_id"] == message_id for message in state["outbox"]):
            raise ActionSimulationError("entity_exists", "message ID already exists", base_path)
        contact_id = _require_id(args["to_contact_id"], f"{base_path}.args.to_contact_id")
        contact = state["contacts"].get(contact_id)
        if contact is None:
            raise ActionSimulationError(
                "entity_missing", "message contact does not exist", base_path
            )
        channel = _require_text(args["channel"], f"{base_path}.args.channel")
        if channel != contact["channel"]:
            raise ActionSimulationError(
                "channel_mismatch", "channel is not registered for contact", base_path
            )
        message = {
            "message_id": message_id,
            "to_contact_id": contact_id,
            "channel": channel,
            "body": _require_text(args["body"], f"{base_path}.args.body"),
            "sent_at": state["reference_time"],
        }
        state["outbox"].append(message)
        path = ("outbox", message_id)
        return _effect("append", path, None, message), _entity_observation(
            tool, message_id, message
        )

    if tool == "media_play":
        track_id = _require_id(args["track_id"], f"{base_path}.args.track_id")
        media = state["media"]
        if track_id not in media["catalog"]:
            raise ActionSimulationError("entity_missing", "media track does not exist", base_path)
        before = {"status": media["status"], "track_id": media["track_id"]}
        media["status"] = "playing"
        media["track_id"] = track_id
        after = {"status": "playing", "track_id": track_id}
        return _effect("set", ("media", "session"), before, after), _entity_observation(
            tool, track_id, after
        )

    if tool == "media_pause":
        media = state["media"]
        if media["status"] != "playing" or media["track_id"] is None:
            raise ActionSimulationError("invalid_media_transition", "nothing is playing", base_path)
        before = {"status": media["status"], "track_id": media["track_id"]}
        media["status"] = "paused"
        after = {"status": "paused", "track_id": media["track_id"]}
        return _effect("set", ("media", "session"), before, after), _entity_observation(
            tool, media["track_id"], after
        )

    if tool == "setting_set_bool":
        key = _require_id(args["key"], f"{base_path}.args.key")
        settings = state["settings"]
        if key not in settings:
            raise ActionSimulationError("entity_missing", "setting does not exist", base_path)
        if type(settings[key]) is not bool:
            raise ActionSimulationError(
                "setting_type_mismatch", "setting is not boolean", base_path
            )
        before = settings[key]
        after = _require_bool(args["enabled"], f"{base_path}.args.enabled")
        settings[key] = after
        return _effect("set", ("settings", key), before, after), _entity_observation(
            tool, key, after
        )

    raise ActionSimulationError("unsupported_tool", f"no semantic adapter for {tool!r}", base_path)


def _parallel_write_target(call: ToolCall) -> tuple[str, ...] | None:
    args = call.args
    if call.tool.startswith("reminder_"):
        return ("reminders", str(args["reminder_id"]))
    if call.tool.startswith("calendar_"):
        return ("calendar", str(args["event_id"]))
    if call.tool == "note_create":
        return ("notes", str(args["note_id"]))
    if call.tool.startswith("list_"):
        return ("lists", str(args["list_id"]), "items", str(args["item_id"]))
    if call.tool == "message_send":
        return ("outbox", str(args["message_id"]))
    if call.tool.startswith("media_"):
        return ("media", "session")
    if call.tool == "setting_set_bool":
        return ("settings", str(args["key"]))
    return None


def _semantic_parallel_write_target(operation: SemanticOperation) -> tuple[str, ...] | None:
    parameters = operation.parameters
    if operation.kind in {"CREATE_REMINDER", "UPDATE_REMINDER"}:
        return ("reminders", str(parameters["reminder_ref"]))
    if operation.kind in {"CREATE_CALENDAR_EVENT", "RESCHEDULE_CALENDAR_EVENT"}:
        return ("calendar", str(parameters["event_ref"]))
    if operation.kind == "CREATE_NOTE":
        return ("notes", str(parameters["note_ref"]))
    if operation.kind in {"ADD_LIST_ITEM", "SET_LIST_ITEM_CHECKED"}:
        return ("lists", str(parameters["list_ref"]), "items", str(parameters["item_ref"]))
    if operation.kind == "PROPOSE_MESSAGE":
        return ("outbox", str(parameters["message_ref"]))
    if operation.kind in {"PLAY_MEDIA", "PAUSE_MEDIA"}:
        return ("media", "session")
    if operation.kind == "SET_BOOLEAN_SETTING":
        return ("settings", str(parameters["setting_ref"]))
    return None


def _check_parallel_targets(targets: Sequence[tuple[str, ...] | None], *, path: str) -> None:
    seen: set[tuple[str, ...]] = set()
    for target in targets:
        if target is not None and target in seen:
            raise ActionSimulationError(
                "parallel_write_conflict",
                "parallel calls may not write the same semantic target",
                path,
            )
        if target is not None:
            seen.add(target)


def _working_state(state: WorldState) -> dict[str, Any]:
    working = state.to_dict()
    working["outbox"] = list(working["outbox"])
    return working


def _stable_action_json(action: ActionIR) -> str:
    first = action.canonical_json()
    second = action.canonical_json()
    if first != second:
        raise ActionSimulationError(
            "concurrent_mutation", "Action IR changed during its stable snapshot"
        )
    return first


def _stable_world_state(state: WorldState) -> WorldState:
    first = state.canonical_json()
    second = state.canonical_json()
    if first != second:
        raise ActionSimulationError(
            "concurrent_mutation", "world state changed during its stable snapshot"
        )
    return WorldState.from_dict(json.loads(first))


def _stable_sim_program(program: SimProgram) -> SimProgram:
    first = program.canonical_json()
    second = program.canonical_json()
    if first != second:
        raise ActionSimulationError(
            "concurrent_mutation", "simulation program changed during its stable snapshot"
        )
    return parse_sim_program(json.loads(first))


def _effects_value(effects: tuple[Effect, ...]) -> StrictJSON:
    return snapshot_strict_json([effect.to_dict() for effect in effects])


def _ordered_effects(effects: Sequence[Effect], mode: CallMode | None) -> tuple[Effect, ...]:
    values = tuple(effects)
    if mode is CallMode.PARALLEL:
        return tuple(
            sorted(
                values,
                key=lambda effect: canonical_json(snapshot_strict_json(effect.to_dict())),
            )
        )
    return values


def _receipt(
    *,
    status: str,
    decision: Decision,
    action_ir_sha256: str,
    input_state: WorldState,
    output_state: WorldState,
    effects: tuple[Effect, ...],
    observation: StrictJSON,
    policy: StrictJSON,
    error_code: str | None = None,
    error_path: str | None = None,
) -> TransitionReceipt:
    return TransitionReceipt(
        status=status,
        decision=decision,
        input_state_sha256=input_state.sha256(),
        action_ir_sha256=action_ir_sha256,
        output_state_sha256=output_state.sha256(),
        effects=effects,
        effects_sha256=canonical_sha256(_effects_value(effects)),
        observation=observation,
        observation_sha256=canonical_sha256(observation),
        policy=policy,
        policy_sha256=canonical_sha256(policy),
        timezone_runtime_sha256=timezone_runtime_sha256(input_state.timezone),
        simulator_runtime_sha256=action_simulator_runtime_sha256(),
        error_code=error_code,
        error_path=error_path,
        _factory_token=_TRANSITION_RECEIPT_FACTORY_TOKEN,
    )


class ActionSimulator:
    """Policy-safe model-visible simulator with no external execution surface."""

    schemas = SIMULATOR_TOOL_REGISTRY

    def simulate(self, action: ActionIR, state: WorldState) -> SimulationResult:
        assert_action_simulator_runtime_integrity()
        if not isinstance(action, ActionIR):
            raise TypeError("action must be ActionIR")
        if not isinstance(state, WorldState):
            raise TypeError("state must be WorldState")
        input_state = _stable_world_state(state)
        source_action_json = _stable_action_json(action)
        source_action_sha256 = hashlib.sha256(source_action_json.encode("utf-8")).hexdigest()
        source_decision = Decision(json.loads(source_action_json)["decision"])
        schemas = simulator_tool_registry_snapshot()
        try:
            action = parse_action_ir(source_action_json, schemas)
        except ActionIRError as exc:
            observation = snapshot_strict_json(
                {"kind": "rejected", "code": exc.code, "path": exc.path}
            )
            empty_policy = snapshot_strict_json(
                {
                    "policy_contract_sha256": POLICY_CONTRACT_SHA256,
                    "policy_valid": False,
                    "expected_decision": None,
                    "authorization_required": False,
                    "confirmation_required": False,
                    "execution_permitted": False,
                    "simulation_transition_permitted": False,
                    "reference_counterfactual_permitted": False,
                    "proposed_call_count": 0,
                    "side_effecting_tools": [],
                    "reason_codes": ["schema_rejected"],
                }
            )
            receipt = _receipt(
                status="rejected",
                decision=source_decision,
                action_ir_sha256=source_action_sha256,
                input_state=input_state,
                output_state=input_state,
                effects=(),
                observation=observation,
                policy=empty_policy,
                error_code=exc.code,
                error_path=exc.path,
            )
            return SimulationResult(state=input_state, receipt=receipt)

        policy = _policy(action, schemas)
        if action.decision is Decision.ABSTAIN:
            observation = snapshot_strict_json({"kind": "abstain"})
            return SimulationResult(
                input_state,
                _receipt(
                    status="no_action",
                    decision=source_decision,
                    action_ir_sha256=source_action_sha256,
                    input_state=input_state,
                    output_state=input_state,
                    effects=(),
                    observation=observation,
                    policy=policy,
                ),
            )
        if action.decision is Decision.CLARIFY:
            observation = snapshot_strict_json({"kind": "clarify", "missing": list(action.missing)})
            return SimulationResult(
                input_state,
                _receipt(
                    status="no_action",
                    decision=source_decision,
                    action_ir_sha256=source_action_sha256,
                    input_state=input_state,
                    output_state=input_state,
                    effects=(),
                    observation=observation,
                    policy=policy,
                ),
            )
        if not policy["policy_valid"]:
            observation = snapshot_strict_json(
                {
                    "kind": "policy_blocked",
                    "expected_decision": policy["expected_decision"],
                    "reason_codes": policy["reason_codes"],
                }
            )
            return SimulationResult(
                input_state,
                _receipt(
                    status="policy_blocked",
                    decision=source_decision,
                    action_ir_sha256=source_action_sha256,
                    input_state=input_state,
                    output_state=input_state,
                    effects=(),
                    observation=observation,
                    policy=policy,
                    error_code="policy_decision_mismatch",
                    error_path="$.decision",
                ),
            )
        if action.decision is Decision.CONFIRM:
            # Validate the complete proposal against a detached working state before asking
            # the user to confirm.  The counterfactual state and effects are discarded here;
            # only the explicitly named reference API may expose them.
            working = _working_state(input_state)
            ordered_calls = action.calls
            if action.mode is CallMode.PARALLEL:
                ordered_calls = tuple(sorted(ordered_calls, key=ToolCall.canonical_json))
            try:
                if action.mode is CallMode.PARALLEL:
                    _check_parallel_targets(
                        tuple(_parallel_write_target(call) for call in ordered_calls),
                        path="$.calls",
                    )
                for call_index, call in enumerate(ordered_calls):
                    _apply_call(working, call, call_index)
                WorldState.from_dict(working)
            except (ActionSimulationError, SimProgramError) as exc:
                observation = snapshot_strict_json(
                    {"kind": "rejected", "code": exc.code, "path": exc.path}
                )
                return SimulationResult(
                    input_state,
                    _receipt(
                        status="rejected",
                        decision=source_decision,
                        action_ir_sha256=source_action_sha256,
                        input_state=input_state,
                        output_state=input_state,
                        effects=(),
                        observation=observation,
                        policy=policy,
                        error_code=exc.code,
                        error_path=exc.path,
                    ),
                )
            observation = snapshot_strict_json(
                {"kind": "confirm", "proposed_calls": [call.to_dict() for call in action.calls]}
            )
            return SimulationResult(
                input_state,
                _receipt(
                    status="confirmation_required",
                    decision=source_decision,
                    action_ir_sha256=source_action_sha256,
                    input_state=input_state,
                    output_state=input_state,
                    effects=(),
                    observation=observation,
                    policy=policy,
                ),
            )

        # A policy-valid CALL contains safe tools only.  These handlers can observe but cannot
        # perform an external side effect.  The same atomic machinery still applies.
        working = _working_state(input_state)
        ordered_calls = action.calls
        if action.mode is CallMode.PARALLEL:
            ordered_calls = tuple(sorted(ordered_calls, key=ToolCall.canonical_json))
        effects: list[Effect] = []
        observations: list[StrictJSON] = []
        try:
            if action.mode is CallMode.PARALLEL:
                _check_parallel_targets(
                    tuple(_parallel_write_target(call) for call in ordered_calls), path="$.calls"
                )
            for call_index, call in enumerate(ordered_calls):
                effect, observation = _apply_call(working, call, call_index)
                effects.append(effect)
                observations.append(observation)
            output_state = WorldState.from_dict(working)
        except (ActionSimulationError, SimProgramError) as exc:
            observation = snapshot_strict_json(
                {"kind": "rejected", "code": exc.code, "path": exc.path}
            )
            return SimulationResult(
                input_state,
                _receipt(
                    status="rejected",
                    decision=source_decision,
                    action_ir_sha256=source_action_sha256,
                    input_state=input_state,
                    output_state=input_state,
                    effects=(),
                    observation=observation,
                    policy=policy,
                    error_code=exc.code,
                    error_path=exc.path,
                ),
            )
        observation = snapshot_strict_json(
            {"kind": "call_result", "mode": action.mode.value, "results": observations}
        )
        return SimulationResult(
            output_state,
            _receipt(
                status="simulated",
                decision=source_decision,
                action_ir_sha256=source_action_sha256,
                input_state=input_state,
                output_state=output_state,
                effects=_ordered_effects(effects, action.mode),
                observation=observation,
                policy=policy,
            ),
        )


def _reference_observation(effects: tuple[Effect, ...]) -> StrictJSON:
    return snapshot_strict_json(
        {"kind": "reference_effects", "effects": [effect.to_dict() for effect in effects]}
    )


def _reference_receipt(
    *,
    status: str,
    source_kind: str,
    source_sha256: str,
    program_sha256: str | None,
    step_id: str | None,
    input_state: WorldState,
    counterfactual_state: WorldState,
    effects: tuple[Effect, ...],
    observation: StrictJSON,
    error_code: str | None = None,
    error_path: str | None = None,
) -> ReferenceEffectReceipt:
    return ReferenceEffectReceipt(
        status=status,
        source_kind=source_kind,
        source_sha256=source_sha256,
        program_sha256=program_sha256,
        step_id=step_id,
        input_state_sha256=input_state.sha256(),
        counterfactual_state_sha256=counterfactual_state.sha256(),
        effects=effects,
        effects_sha256=canonical_sha256(_effects_value(effects)),
        observation=observation,
        observation_sha256=canonical_sha256(observation),
        timezone_runtime_sha256=timezone_runtime_sha256(input_state.timezone),
        simulator_runtime_sha256=action_simulator_runtime_sha256(),
        error_code=error_code,
        error_path=error_path,
        _factory_token=_REFERENCE_RECEIPT_FACTORY_TOKEN,
    )


def simulate_reference_action(action: ActionIR, state: WorldState) -> ReferenceSimulationResult:
    """Counterfactually apply Action IR in memory under an unmistakably reference-only receipt."""

    assert_action_simulator_runtime_integrity()
    if not isinstance(action, ActionIR) or not isinstance(state, WorldState):
        raise TypeError("reference action simulation requires ActionIR and WorldState")
    source_json = _stable_action_json(action)
    action = parse_action_ir(source_json, simulator_tool_registry_snapshot())
    input_state = _stable_world_state(state)
    source_sha = hashlib.sha256(source_json.encode("utf-8")).hexdigest()
    if action.decision not in (Decision.CALL, Decision.CONFIRM):
        observation = snapshot_strict_json(
            {
                "kind": "reference_control",
                "decision": action.decision.value,
                "missing": list(action.missing),
            }
        )
        receipt = _reference_receipt(
            status="reference_control",
            source_kind="compiled_action",
            source_sha256=source_sha,
            program_sha256=None,
            step_id=None,
            input_state=input_state,
            counterfactual_state=input_state,
            effects=(),
            observation=observation,
        )
        return ReferenceSimulationResult(input_state, receipt)
    working = _working_state(input_state)
    ordered_calls = action.calls
    if action.mode is CallMode.PARALLEL:
        ordered_calls = tuple(sorted(ordered_calls, key=ToolCall.canonical_json))
    effects: list[Effect] = []
    try:
        if action.mode is CallMode.PARALLEL:
            _check_parallel_targets(
                tuple(_parallel_write_target(call) for call in ordered_calls), path="$.calls"
            )
        for call_index, call in enumerate(ordered_calls):
            effect, _ = _apply_call(working, call, call_index)
            effects.append(effect)
        counterfactual_state = WorldState.from_dict(working)
    except (ActionSimulationError, SimProgramError) as exc:
        observation = snapshot_strict_json(
            {"kind": "reference_rejected", "code": exc.code, "path": exc.path}
        )
        receipt = _reference_receipt(
            status="reference_rejected",
            source_kind="compiled_action",
            source_sha256=source_sha,
            program_sha256=None,
            step_id=None,
            input_state=input_state,
            counterfactual_state=input_state,
            effects=(),
            observation=observation,
            error_code=exc.code,
            error_path=exc.path,
        )
        return ReferenceSimulationResult(input_state, receipt)
    effect_tuple = _ordered_effects(effects, action.mode)
    observation = _reference_observation(effect_tuple)
    receipt = _reference_receipt(
        status="reference_applied",
        source_kind="compiled_action",
        source_sha256=source_sha,
        program_sha256=None,
        step_id=None,
        input_state=input_state,
        counterfactual_state=counterfactual_state,
        effects=effect_tuple,
        observation=observation,
    )
    return ReferenceSimulationResult(counterfactual_state, receipt)


def _apply_semantic_operation(
    state: dict[str, Any], operation: SemanticOperation, index: int
) -> Effect:
    """Independent domain reference implementation; it never compiles or calls `_apply_call`."""

    p = operation.parameters
    base = f"$.semantic_operations[{index}]"
    kind = operation.kind
    if kind in {"CREATE_REMINDER", "UPDATE_REMINDER"}:
        entity_id = _require_id(p["reminder_ref"], f"{base}.parameters.reminder_ref")
        table = state["reminders"]
        exists = entity_id in table
        if kind == "CREATE_REMINDER" and exists:
            raise ActionSimulationError("entity_exists", "reminder already exists", base)
        if kind == "UPDATE_REMINDER" and not exists:
            raise ActionSimulationError("entity_missing", "reminder does not exist", base)
        before = table.get(entity_id)
        after = {
            "title": _require_text(p["summary"], f"{base}.parameters.summary"),
            "due_at": _require_timestamp(state, p["due_at"], f"{base}.parameters.due_at"),
            "completed": before["completed"] if before is not None else False,
        }
        table[entity_id] = after
        return _effect("upsert", ("reminders", entity_id), before, after)
    if kind in {"CREATE_CALENDAR_EVENT", "RESCHEDULE_CALENDAR_EVENT"}:
        entity_id = _require_id(p["event_ref"], f"{base}.parameters.event_ref")
        table = state["calendar"]
        exists = entity_id in table
        if kind == "CREATE_CALENDAR_EVENT" and exists:
            raise ActionSimulationError("entity_exists", "calendar event already exists", base)
        if kind == "RESCHEDULE_CALENDAR_EVENT" and not exists:
            raise ActionSimulationError("entity_missing", "calendar event does not exist", base)
        before = table.get(entity_id)
        start_at = _require_timestamp(state, p["begins_at"], f"{base}.parameters.begins_at")
        end_at = _require_timestamp(state, p["ends_at"], f"{base}.parameters.ends_at")
        if datetime.fromisoformat(end_at) <= datetime.fromisoformat(start_at):
            raise ActionSimulationError("invalid_interval", "ends_at must be after begins_at", base)
        after = {
            "title": (
                _require_text(p["summary"], f"{base}.parameters.summary")
                if kind == "CREATE_CALENDAR_EVENT"
                else before["title"]
            ),
            "start_at": start_at,
            "end_at": end_at,
        }
        table[entity_id] = after
        return _effect("upsert", ("calendar", entity_id), before, after)
    if kind == "LOOK_UP_CONTACT":
        entity_id = _require_id(p["contact_ref"], f"{base}.parameters.contact_ref")
        contact = state["contacts"].get(entity_id)
        if contact is None:
            raise ActionSimulationError("entity_missing", "contact does not exist", base)
        return _effect("observe", ("contacts", entity_id), contact, contact)
    if kind == "CREATE_NOTE":
        entity_id = _require_id(p["note_ref"], f"{base}.parameters.note_ref")
        table = state["notes"]
        if entity_id in table:
            raise ActionSimulationError("entity_exists", "note already exists", base)
        value = {
            "title": _require_text(p["heading"], f"{base}.parameters.heading"),
            "body": _require_text(p["content"], f"{base}.parameters.content", nonempty=False),
        }
        table[entity_id] = value
        return _effect("create", ("notes", entity_id), None, value)
    if kind in {"ADD_LIST_ITEM", "SET_LIST_ITEM_CHECKED"}:
        list_id = _require_id(p["list_ref"], f"{base}.parameters.list_ref")
        item_id = _require_id(p["item_ref"], f"{base}.parameters.item_ref")
        list_value = state["lists"].get(list_id)
        if list_value is None:
            raise ActionSimulationError("entity_missing", "list does not exist", base)
        items = list_value["items"]
        before = items.get(item_id)
        if kind == "ADD_LIST_ITEM":
            if before is not None:
                raise ActionSimulationError("entity_exists", "list item already exists", base)
            after = {
                "text": _require_text(p["item_text"], f"{base}.parameters.item_text"),
                "checked": False,
            }
        else:
            if before is None:
                raise ActionSimulationError("entity_missing", "list item does not exist", base)
            after = {
                "text": before["text"],
                "checked": _require_bool(p["is_checked"], f"{base}.parameters.is_checked"),
            }
        items[item_id] = after
        return _effect("upsert", ("lists", list_id, "items", item_id), before, after)
    if kind == "LOOK_UP_ROUTE":
        origin = _require_id(p["from_place_ref"], f"{base}.parameters.from_place_ref")
        destination = _require_id(p["to_place_ref"], f"{base}.parameters.to_place_ref")
        mode = _require_text(p["travel_mode"], f"{base}.parameters.travel_mode")
        matches = [
            (route_id, route)
            for route_id, route in state["routes"].items()
            if route["origin_id"] == origin
            and route["destination_id"] == destination
            and route["mode"] == mode
        ]
        if len(matches) != 1:
            raise ActionSimulationError(
                "route_missing" if not matches else "route_ambiguous",
                "route does not resolve uniquely",
                base,
            )
        route_id, route = matches[0]
        return _effect("observe", ("routes", route_id), route, route)
    if kind == "PROPOSE_MESSAGE":
        message_id = _require_id(p["message_ref"], f"{base}.parameters.message_ref")
        if any(message["message_id"] == message_id for message in state["outbox"]):
            raise ActionSimulationError("entity_exists", "message ID already exists", base)
        recipient = _require_id(p["recipient_ref"], f"{base}.parameters.recipient_ref")
        contact = state["contacts"].get(recipient)
        if contact is None:
            raise ActionSimulationError("entity_missing", "message contact does not exist", base)
        channel = _require_text(p["delivery_channel"], f"{base}.parameters.delivery_channel")
        if channel != contact["channel"]:
            raise ActionSimulationError(
                "channel_mismatch", "channel is not registered for contact", base
            )
        message = {
            "message_id": message_id,
            "to_contact_id": recipient,
            "channel": channel,
            "body": _require_text(p["content"], f"{base}.parameters.content"),
            "sent_at": state["reference_time"],
        }
        state["outbox"].append(message)
        return _effect("append", ("outbox", message_id), None, message)
    if kind == "PLAY_MEDIA":
        track = _require_id(p["track_ref"], f"{base}.parameters.track_ref")
        media = state["media"]
        if track not in media["catalog"]:
            raise ActionSimulationError("entity_missing", "media track does not exist", base)
        before = {"status": media["status"], "track_id": media["track_id"]}
        after = {"status": "playing", "track_id": track}
        media.update(after)
        return _effect("set", ("media", "session"), before, after)
    if kind == "PAUSE_MEDIA":
        media = state["media"]
        if media["status"] != "playing" or media["track_id"] is None:
            raise ActionSimulationError("invalid_media_transition", "nothing is playing", base)
        before = {"status": media["status"], "track_id": media["track_id"]}
        after = {"status": "paused", "track_id": media["track_id"]}
        media.update(after)
        return _effect("set", ("media", "session"), before, after)
    if kind == "SET_BOOLEAN_SETTING":
        key = _require_id(p["setting_ref"], f"{base}.parameters.setting_ref")
        settings = state["settings"]
        if key not in settings:
            raise ActionSimulationError("entity_missing", "setting does not exist", base)
        if type(settings[key]) is not bool:
            raise ActionSimulationError("setting_type_mismatch", "setting is not boolean", base)
        before = settings[key]
        after = _require_bool(p["value"], f"{base}.parameters.value")
        settings[key] = after
        return _effect("set", ("settings", key), before, after)
    raise ActionSimulationError("unsupported_semantic_operation", f"unknown kind {kind!r}", base)


def simulate_reference_program_step(
    step: ProgramStep,
    state: WorldState,
    *,
    program_sha256: str,
) -> ReferenceSimulationResult:
    """Independently derive domain effects without model tool names or compiler calls."""

    assert_action_simulator_runtime_integrity()
    if not isinstance(step, ProgramStep) or not isinstance(state, WorldState):
        raise TypeError("reference program simulation requires ProgramStep and WorldState")
    _require_sha(program_sha256, path="$.program_sha256")
    input_state = _stable_world_state(state)
    source_value = snapshot_strict_json(step.to_dict())
    if canonical_json(source_value) != canonical_json(snapshot_strict_json(step.to_dict())):
        raise ActionSimulationError(
            "concurrent_mutation", "program step changed during its stable snapshot"
        )
    source_sha = canonical_sha256(source_value)
    step = parse_sim_program(
        {
            "schema_version": SIM_PROGRAM_VERSION,
            "case_id": "case.reference-step-snapshot",
            "initial_state": input_state.to_dict(),
            "steps": [json.loads(canonical_json(source_value))],
        }
    ).steps[0]
    if step.decision not in (Decision.CALL, Decision.CONFIRM):
        observation = snapshot_strict_json(
            {
                "kind": "reference_control",
                "decision": step.decision.value,
                "missing": list(step.missing),
            }
        )
        receipt = _reference_receipt(
            status="reference_control",
            source_kind="semantic_program",
            source_sha256=source_sha,
            program_sha256=program_sha256,
            step_id=step.step_id,
            input_state=input_state,
            counterfactual_state=input_state,
            effects=(),
            observation=observation,
        )
        return ReferenceSimulationResult(input_state, receipt)
    working = _working_state(input_state)
    operations = step.operations
    effects: list[Effect] = []
    try:
        if step.mode is CallMode.PARALLEL:
            _check_parallel_targets(
                tuple(_semantic_parallel_write_target(operation) for operation in operations),
                path="$.semantic_operations",
            )
        for index, operation in enumerate(operations):
            effects.append(_apply_semantic_operation(working, operation, index))
        counterfactual_state = WorldState.from_dict(working)
    except (ActionSimulationError, SimProgramError) as exc:
        observation = snapshot_strict_json(
            {"kind": "reference_rejected", "code": exc.code, "path": exc.path}
        )
        receipt = _reference_receipt(
            status="reference_rejected",
            source_kind="semantic_program",
            source_sha256=source_sha,
            program_sha256=program_sha256,
            step_id=step.step_id,
            input_state=input_state,
            counterfactual_state=input_state,
            effects=(),
            observation=observation,
            error_code=exc.code,
            error_path=exc.path,
        )
        return ReferenceSimulationResult(input_state, receipt)
    effect_tuple = _ordered_effects(effects, step.mode)
    observation = _reference_observation(effect_tuple)
    receipt = _reference_receipt(
        status="reference_applied",
        source_kind="semantic_program",
        source_sha256=source_sha,
        program_sha256=program_sha256,
        step_id=step.step_id,
        input_state=input_state,
        counterfactual_state=counterfactual_state,
        effects=effect_tuple,
        observation=observation,
    )
    return ReferenceSimulationResult(counterfactual_state, receipt)


def verify_program_round_trip(program: SimProgram) -> ProgramRoundTripReceipt:
    """Compare independent domain effects with compiled Action IR effects for every step."""

    assert_action_simulator_runtime_integrity()
    if not isinstance(program, SimProgram):
        raise TypeError("program must be SimProgram")
    program = _stable_sim_program(program)
    program_sha = program.sha256()
    simulator = ActionSimulator()
    state = _stable_world_state(program.initial_state)
    records: list[StrictJSON] = []
    passed = True
    for step in program.steps:
        action = compile_program_step(step)
        visible = simulator.simulate(action, state)
        compiled_reference = simulate_reference_action(action, state)
        semantic_reference = simulate_reference_program_step(
            step, state, program_sha256=program_sha
        )
        expected_reference_status = (
            "reference_applied"
            if step.decision in (Decision.CALL, Decision.CONFIRM)
            else "reference_control"
        )
        reference_match = (
            compiled_reference.receipt.status == expected_reference_status
            and semantic_reference.receipt.status == expected_reference_status
            and compiled_reference.counterfactual_state.sha256()
            == semantic_reference.counterfactual_state.sha256()
            and compiled_reference.receipt.effects_sha256
            == semantic_reference.receipt.effects_sha256
            and compiled_reference.receipt.observation_sha256
            == semantic_reference.receipt.observation_sha256
        )
        expected_status = {
            Decision.CALL: "simulated",
            Decision.CONFIRM: "confirmation_required",
            Decision.CLARIFY: "no_action",
            Decision.ABSTAIN: "no_action",
        }[step.decision]
        visible_policy_safe = (
            visible.receipt.status == expected_status
            and visible.receipt.policy["policy_valid"] is True
            and visible.receipt.external_side_effects is False
            and visible.receipt.authorizes_execution is False
        )
        if step.decision is Decision.CONFIRM:
            visible_policy_safe = visible_policy_safe and visible.state.sha256() == state.sha256()
        step_passed = reference_match and visible_policy_safe
        passed = passed and step_passed
        records.append(
            snapshot_strict_json(
                {
                    "step_id": step.step_id,
                    "action_ir_sha256": hashlib.sha256(
                        action.canonical_json().encode("utf-8")
                    ).hexdigest(),
                    "visible_receipt_sha256": visible.receipt.sha256(),
                    "compiled_reference_receipt_sha256": compiled_reference.receipt.sha256(),
                    "semantic_reference_receipt_sha256": semantic_reference.receipt.sha256(),
                    "reference_match": reference_match,
                    "visible_policy_safe": visible_policy_safe,
                    "passed": step_passed,
                }
            )
        )
        state = visible.state
    return ProgramRoundTripReceipt(
        program_sha256=program_sha,
        final_state_sha256=state.sha256(),
        step_records=tuple(records),
        timezone_runtime_sha256=timezone_runtime_sha256(state.timezone),
        simulator_runtime_sha256=action_simulator_runtime_sha256(),
        passed=passed,
        _factory_token=_ROUND_TRIP_RECEIPT_FACTORY_TOKEN,
    )


def simulate_program(program: SimProgram) -> ProgramSimulation:
    assert_action_simulator_runtime_integrity()
    if not isinstance(program, SimProgram):
        raise TypeError("program must be SimProgram")
    program = _stable_sim_program(program)
    simulator = ActionSimulator()
    state = _stable_world_state(program.initial_state)
    actions: list[ActionIR] = []
    results: list[SimulationResult] = []
    for step in program.steps:
        action = compile_program_step(step)
        result = simulator.simulate(action, state)
        actions.append(action)
        results.append(result)
        state = result.state
    return ProgramSimulation(final_state=state, actions=tuple(actions), results=tuple(results))


def _exact_receipt_fields(value: Mapping[str, StrictJSON], expected: set[str]) -> None:
    if set(value) != expected:
        raise ActionSimulationError("receipt_shape", "receipt fields are not exact")


def loads_transition_receipt(
    text: str,
    *,
    action: ActionIR,
    state: WorldState,
) -> TransitionReceipt:
    """Load and replay one transition receipt against its required action and state."""

    assert_action_simulator_runtime_integrity()
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
        "status",
        "decision",
        "input_state_sha256",
        "action_ir_sha256",
        "output_state_sha256",
        "effects",
        "effects_sha256",
        "observation",
        "observation_sha256",
        "policy",
        "policy_sha256",
        "timezone_runtime_sha256",
        "simulator_runtime_sha256",
        "error_code",
        "error_path",
        "external_side_effects",
        "authorizes_execution",
    }
    _exact_receipt_fields(value, expected)
    identities = {
        "schema_version": TRANSITION_RECEIPT_VERSION,
        "simulator_version": ACTION_SIMULATOR_VERSION,
        "schema_sha256": SIMULATOR_SCHEMA_SHA256,
        "policy_contract_sha256": POLICY_CONTRACT_SHA256,
        "program_contract_version": SIM_PROGRAM_VERSION,
        "world_contract_version": WORLD_STATE_VERSION,
        "reference_effect_version": REFERENCE_EFFECT_VERSION,
        "timezone_runtime_version": TIMEZONE_RUNTIME_VERSION,
        "simulator_runtime_sha256": action_simulator_runtime_sha256(),
    }
    if any(value[key] != expected_value for key, expected_value in identities.items()):
        raise ActionSimulationError("receipt_identity", "receipt identity mismatch")
    raw_effects = value["effects"]
    if not isinstance(raw_effects, tuple):
        raise ActionSimulationError("receipt_shape", "effects must be an array")
    try:
        decision = Decision(value["decision"])
    except (TypeError, ValueError) as exc:
        raise ActionSimulationError("receipt_shape", "invalid decision") from exc
    receipt = TransitionReceipt(
        status=value["status"],
        decision=decision,
        input_state_sha256=value["input_state_sha256"],
        action_ir_sha256=value["action_ir_sha256"],
        output_state_sha256=value["output_state_sha256"],
        effects=tuple(
            _effect_from_value(effect, path=f"$.effects[{index}]")
            for index, effect in enumerate(raw_effects)
        ),
        effects_sha256=value["effects_sha256"],
        observation=value["observation"],
        observation_sha256=value["observation_sha256"],
        policy=value["policy"],
        policy_sha256=value["policy_sha256"],
        timezone_runtime_sha256=value["timezone_runtime_sha256"],
        simulator_runtime_sha256=value["simulator_runtime_sha256"],
        error_code=value["error_code"],
        error_path=value["error_path"],
        external_side_effects=value["external_side_effects"],
        authorizes_execution=value["authorizes_execution"],
        _factory_token=_TRANSITION_RECEIPT_FACTORY_TOKEN,
    )
    if not isinstance(action, ActionIR) or not isinstance(state, WorldState):
        raise ActionSimulationError(
            "receipt_context", "transition replay requires ActionIR and WorldState"
        )
    replayed = ActionSimulator().simulate(action, state).receipt
    if replayed.canonical_json() != receipt.canonical_json():
        raise ActionSimulationError(
            "receipt_replay", "transition receipt does not reproduce from supplied context"
        )
    return receipt


def loads_reference_receipt(
    text: str,
    *,
    source: ActionIR | ProgramStep,
    state: WorldState,
    program_sha256: str | None = None,
) -> ReferenceEffectReceipt:
    """Load and replay one reference receipt against its required source and state."""

    assert_action_simulator_runtime_integrity()
    value = loads_strict_json_object(text)
    expected = {
        "schema_version",
        "reference_effect_version",
        "simulator_version",
        "schema_sha256",
        "policy_contract_sha256",
        "program_contract_version",
        "world_contract_version",
        "timezone_runtime_version",
        "status",
        "source_kind",
        "source_sha256",
        "program_sha256",
        "step_id",
        "input_state_sha256",
        "counterfactual_state_sha256",
        "effects",
        "effects_sha256",
        "observation",
        "observation_sha256",
        "timezone_runtime_sha256",
        "simulator_runtime_sha256",
        "error_code",
        "error_path",
        "reference_only",
        "authorizes_execution",
        "external_side_effects",
    }
    _exact_receipt_fields(value, expected)
    identities = {
        "schema_version": REFERENCE_RECEIPT_VERSION,
        "reference_effect_version": REFERENCE_EFFECT_VERSION,
        "simulator_version": ACTION_SIMULATOR_VERSION,
        "schema_sha256": SIMULATOR_SCHEMA_SHA256,
        "policy_contract_sha256": POLICY_CONTRACT_SHA256,
        "program_contract_version": SIM_PROGRAM_VERSION,
        "world_contract_version": WORLD_STATE_VERSION,
        "timezone_runtime_version": TIMEZONE_RUNTIME_VERSION,
        "simulator_runtime_sha256": action_simulator_runtime_sha256(),
    }
    if any(value[key] != expected_value for key, expected_value in identities.items()):
        raise ActionSimulationError("receipt_identity", "reference receipt identity mismatch")
    raw_effects = value["effects"]
    if not isinstance(raw_effects, tuple):
        raise ActionSimulationError("receipt_shape", "effects must be an array")
    receipt = ReferenceEffectReceipt(
        status=value["status"],
        source_kind=value["source_kind"],
        source_sha256=value["source_sha256"],
        program_sha256=value["program_sha256"],
        step_id=value["step_id"],
        input_state_sha256=value["input_state_sha256"],
        counterfactual_state_sha256=value["counterfactual_state_sha256"],
        effects=tuple(
            _effect_from_value(effect, path=f"$.effects[{index}]")
            for index, effect in enumerate(raw_effects)
        ),
        effects_sha256=value["effects_sha256"],
        observation=value["observation"],
        observation_sha256=value["observation_sha256"],
        timezone_runtime_sha256=value["timezone_runtime_sha256"],
        simulator_runtime_sha256=value["simulator_runtime_sha256"],
        error_code=value["error_code"],
        error_path=value["error_path"],
        reference_only=value["reference_only"],
        authorizes_execution=value["authorizes_execution"],
        external_side_effects=value["external_side_effects"],
        _factory_token=_REFERENCE_RECEIPT_FACTORY_TOKEN,
    )
    if not isinstance(state, WorldState):
        raise ActionSimulationError("receipt_context", "reference replay requires WorldState")
    if receipt.source_kind == "compiled_action":
        if not isinstance(source, ActionIR) or program_sha256 is not None:
            raise ActionSimulationError(
                "receipt_context",
                "compiled-action receipt requires ActionIR and forbids program_sha256",
            )
        replayed = simulate_reference_action(source, state).receipt
    else:
        if not isinstance(source, ProgramStep) or program_sha256 is None:
            raise ActionSimulationError(
                "receipt_context",
                "semantic-program receipt requires ProgramStep and program_sha256",
            )
        replayed = simulate_reference_program_step(
            source,
            state,
            program_sha256=program_sha256,
        ).receipt
    if replayed.canonical_json() != receipt.canonical_json():
        raise ActionSimulationError(
            "receipt_replay", "reference receipt does not reproduce from supplied context"
        )
    return receipt


def loads_program_round_trip_receipt(
    text: str,
    *,
    program: SimProgram,
) -> ProgramRoundTripReceipt:
    """Load and fully recompute one round-trip receipt from its required program."""

    assert_action_simulator_runtime_integrity()
    value = loads_strict_json_object(text)
    expected = {
        "schema_version",
        "simulator_version",
        "reference_effect_version",
        "schema_sha256",
        "policy_contract_sha256",
        "program_contract_version",
        "world_contract_version",
        "program_sha256",
        "final_state_sha256",
        "step_records",
        "timezone_runtime_sha256",
        "simulator_runtime_sha256",
        "passed",
        "authorizes_execution",
        "external_side_effects",
    }
    _exact_receipt_fields(value, expected)
    identities = {
        "schema_version": PROGRAM_ROUND_TRIP_VERSION,
        "simulator_version": ACTION_SIMULATOR_VERSION,
        "reference_effect_version": REFERENCE_EFFECT_VERSION,
        "schema_sha256": SIMULATOR_SCHEMA_SHA256,
        "policy_contract_sha256": POLICY_CONTRACT_SHA256,
        "program_contract_version": SIM_PROGRAM_VERSION,
        "world_contract_version": WORLD_STATE_VERSION,
        "authorizes_execution": False,
        "external_side_effects": False,
        "simulator_runtime_sha256": action_simulator_runtime_sha256(),
    }
    if any(value[key] != expected_value for key, expected_value in identities.items()):
        raise ActionSimulationError("receipt_identity", "round-trip receipt identity mismatch")
    records = value["step_records"]
    if (
        not isinstance(records, tuple)
        or type(value["passed"]) is not bool
        or type(value["authorizes_execution"]) is not bool
        or type(value["external_side_effects"]) is not bool
    ):
        raise ActionSimulationError("receipt_shape", "invalid round-trip receipt fields")
    receipt = ProgramRoundTripReceipt(
        program_sha256=value["program_sha256"],
        final_state_sha256=value["final_state_sha256"],
        step_records=tuple(records),
        timezone_runtime_sha256=value["timezone_runtime_sha256"],
        simulator_runtime_sha256=value["simulator_runtime_sha256"],
        passed=value["passed"],
        _factory_token=_ROUND_TRIP_RECEIPT_FACTORY_TOKEN,
    )
    if not isinstance(program, SimProgram):
        raise ActionSimulationError("receipt_context", "round-trip replay requires SimProgram")
    replayed = verify_program_round_trip(program)
    if replayed.canonical_json() != receipt.canonical_json():
        raise ActionSimulationError(
            "receipt_replay", "round-trip receipt does not reproduce from supplied program"
        )
    return receipt


def _live_simulator_schema_bytes() -> str:
    try:
        return canonical_json(
            snapshot_strict_json([_schema_record(schema) for schema in SIMULATOR_TOOL_SCHEMAS])
        )
    except (AttributeError, KeyError, SimProgramError, TypeError, ValueError) as exc:
        raise ActionSimulationError(
            "runtime_identity", "simulator schema registry is malformed"
        ) from exc


def _assert_simulator_schema_integrity() -> None:
    if (
        SIMULATOR_TOOL_SCHEMAS is not _PINNED_SIMULATOR_TOOL_SCHEMAS
        or SIMULATOR_TOOL_REGISTRY is not _PINNED_SIMULATOR_TOOL_REGISTRY
        or _POLICY_RECORD is not _PINNED_POLICY_RECORD
        or ActionSimulator.schemas is not _PINNED_SIMULATOR_TOOL_REGISTRY
    ):
        raise ActionSimulationError(
            "runtime_identity", "simulator schema or policy registry was replaced"
        )
    if tuple(SIMULATOR_TOOL_REGISTRY) != tuple(
        schema.name for schema in SIMULATOR_TOOL_SCHEMAS
    ) or any(
        SIMULATOR_TOOL_REGISTRY.get(schema.name) is not schema for schema in SIMULATOR_TOOL_SCHEMAS
    ):
        raise ActionSimulationError(
            "runtime_identity", "simulator schema registry membership changed"
        )
    live_schema_bytes = _live_simulator_schema_bytes()
    if (
        live_schema_bytes != SIMULATOR_SCHEMA_BYTES
        or hashlib.sha256(live_schema_bytes.encode("utf-8")).hexdigest() != SIMULATOR_SCHEMA_SHA256
        or canonical_sha256(_POLICY_RECORD) != POLICY_CONTRACT_SHA256
    ):
        raise ActionSimulationError(
            "runtime_identity", "simulator schema or policy contract changed"
        )


def _action_ir_runtime_sha256() -> str:
    source_path = getattr(_action_ir_module, "__file__", None)
    if not isinstance(source_path, str):
        raise ActionSimulationError("runtime_identity", "Action IR module has no source identity")
    contract = {
        "max_json_nesting": _action_ir_module.MAX_JSON_NESTING,
        "decisions": [item.value for item in Decision],
        "call_modes": [item.value for item in CallMode],
        "json_types": [item.value for item in JSONType],
    }
    try:
        return module_runtime_sha256(
            vars(_action_ir_module),
            module_name=_action_ir_module.__name__,
            source_path=source_path,
            contract=contract,
        )
    except SimProgramError as exc:
        raise ActionSimulationError(
            "runtime_identity", "could not bind the Action IR runtime"
        ) from exc


def action_simulator_runtime_sha256() -> str:
    """Bind source, live code, schemas, policy, dependencies, and lower contracts."""

    dependencies = {
        name: _mutable(runtime_callable_identity(value))
        for name, value in (
            ("ActionIR.canonical_json", ActionIR.canonical_json),
            ("ToolCall.canonical_json", ToolCall.canonical_json),
            ("WorldState.from_dict", WorldState.from_dict),
            ("canonical_timestamp", canonical_timestamp),
            ("datetime.fromisoformat", datetime.fromisoformat),
            ("hashlib.sha256", hashlib.sha256),
            ("json.dumps", json.dumps),
            ("json.loads", json.loads),
            ("parse_action_ir", parse_action_ir),
        )
    }
    contract = {
        "schema_versions": {
            "simulator": ACTION_SIMULATOR_VERSION,
            "transition_receipt": TRANSITION_RECEIPT_VERSION,
            "reference_effect": REFERENCE_EFFECT_VERSION,
            "reference_receipt": REFERENCE_RECEIPT_VERSION,
            "program_round_trip": PROGRAM_ROUND_TRIP_VERSION,
            "policy": POLICY_CONTRACT_VERSION,
        },
        "simulator_schema": json.loads(_live_simulator_schema_bytes()),
        "simulator_schema_sha256": SIMULATOR_SCHEMA_SHA256,
        "policy_record": _mutable(_POLICY_RECORD),
        "policy_contract_sha256": POLICY_CONTRACT_SHA256,
        "policy_fields": sorted(_POLICY_FIELDS),
        "semantic_to_tool": dict(_SEMANTIC_TO_TOOL),
        "transition_statuses": sorted(_TRANSITION_STATUSES),
        "action_ir_runtime_sha256": _action_ir_runtime_sha256(),
        "sim_program_runtime_sha256": sim_program_runtime_sha256(),
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
        raise ActionSimulationError(
            "runtime_identity", "could not bind the action-simulator runtime"
        ) from exc


def assert_action_simulator_runtime_integrity() -> None:
    assert_sim_program_runtime_integrity()
    _assert_simulator_schema_integrity()
    current = action_simulator_runtime_sha256()
    if current != _PINNED_ACTION_SIMULATOR_RUNTIME_SHA256:
        raise ActionSimulationError(
            "runtime_identity",
            "action-simulator runtime differs from its import-time identity",
        )


__all__ = [
    "ACTION_SIMULATOR_VERSION",
    "POLICY_CONTRACT_SHA256",
    "POLICY_CONTRACT_VERSION",
    "PROGRAM_ROUND_TRIP_VERSION",
    "REFERENCE_EFFECT_VERSION",
    "REFERENCE_RECEIPT_VERSION",
    "SIMULATOR_SCHEMA_BYTES",
    "SIMULATOR_SCHEMA_SHA256",
    "SIMULATOR_TOOL_REGISTRY",
    "SIMULATOR_TOOL_SCHEMAS",
    "TRANSITION_RECEIPT_VERSION",
    "ActionSimulationError",
    "ActionSimulator",
    "Effect",
    "ProgramRoundTripReceipt",
    "ProgramSimulation",
    "ReferenceEffectReceipt",
    "ReferenceSimulationResult",
    "SimulationResult",
    "TransitionReceipt",
    "action_simulator_runtime_sha256",
    "assert_action_simulator_runtime_integrity",
    "compile_program_step",
    "loads_program_round_trip_receipt",
    "loads_reference_receipt",
    "loads_transition_receipt",
    "simulate_program",
    "simulate_reference_action",
    "simulate_reference_program_step",
    "simulator_tool_registry_snapshot",
    "verify_program_round_trip",
]


_PINNED_SIMULATOR_TOOL_SCHEMAS = SIMULATOR_TOOL_SCHEMAS
_PINNED_SIMULATOR_TOOL_REGISTRY = SIMULATOR_TOOL_REGISTRY
_PINNED_POLICY_RECORD = _POLICY_RECORD
_PINNED_ACTION_SIMULATOR_RUNTIME_SHA256 = action_simulator_runtime_sha256()
