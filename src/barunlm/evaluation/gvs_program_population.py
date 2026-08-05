"""Deterministic pre-authoring programs for the GVS T-new/D-support populations.

This module performs one deliberately narrow job: from an immutable count-and-floor
specification, it freezes training-role membership before any request text is authored.
It then materializes exact :class:`~barunlm.evaluation.sim_program.SimProgram` objects and
structured authoring packets containing semantic facts, never final requests or teacher/model
answers.  Every included program must pass the independent simulator round-trip verifier.

The planner is standard-library-only apart from the project's CPU-only simulator contracts.  It
has no clock, randomness, network, subprocess, model, device, or remote-compute input.  It does
not construct S-new or C-new, close the four-population duplicate firewall, provide human-label
custody, or authorize model, label, CUDA, JarvisLabs, training, launch, or execution access.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import platform
import re
import sys
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

from .action_simulator import (
    action_simulator_runtime_sha256,
    verify_program_round_trip,
)
from .sim_program import (
    SEMANTIC_OPERATION_SPECS,
    SIM_PROGRAM_VERSION,
    WORLD_STATE_VERSION,
    SimProgram,
    StrictJSON,
    module_runtime_sha256,
    parse_sim_program,
    runtime_callable_identity,
    sim_program_runtime_sha256,
    snapshot_strict_json,
)

PROGRAM_POPULATION_SPEC_VERSION = "barun-gvs-program-population-spec-v1"
PROGRAM_POPULATION_PLAN_VERSION = "barun-gvs-program-population-plan-v1"
PROGRAM_POPULATION_RECORD_VERSION = "barun-gvs-program-population-record-v1"
AUTHORING_PACKET_VERSION = "barun-gvs-authoring-packet-v1"

TRAINING_ROLES = ("T-new", "D-support")
HUMAN_ROLES_EXCLUDED = ("S-new", "C-new")
EXPECTED_OUTCOMES = ("ACTION", "CONFIRM", "CLARIFY", "ABSTAIN")
SEMANTIC_OPERATIONS = tuple(sorted(SEMANTIC_OPERATION_SPECS))
SAFE_OPERATIONS = tuple(
    kind for kind in SEMANTIC_OPERATIONS if not SEMANTIC_OPERATION_SPECS[kind][1]
)
SIDE_EFFECTING_OPERATIONS = tuple(
    kind for kind in SEMANTIC_OPERATIONS if SEMANTIC_OPERATION_SPECS[kind][1]
)
TEMPORAL_OPERATIONS = (
    "CREATE_CALENDAR_EVENT",
    "CREATE_REMINDER",
    "RESCHEDULE_CALENDAR_EVENT",
    "UPDATE_REMINDER",
)
FEATURE_STRATA = (
    "context_grounding",
    "revision",
    "disfluency",
    "distractor_tools",
    "timezone_or_relative_time",
    "multi_action",
    "identity_schema",
    "renamed_schema",
    "unsafe_or_adversarial",
)
OPERATION_STRATA = tuple(f"operation:{kind}" for kind in SEMANTIC_OPERATIONS)
OUTCOME_STRATA = tuple(f"outcome:{outcome}" for outcome in EXPECTED_OUTCOMES)
REQUIRED_STRATA = tuple(sorted((*OPERATION_STRATA, *OUTCOME_STRATA, *FEATURE_STRATA)))

MIN_RECORDS_PER_ROLE = len(SEMANTIC_OPERATIONS) + 2
MAX_RECORDS_PER_ROLE = 40_000
MAX_TOTAL_RECORDS = 50_000
MAX_SPEC_JSON_BYTES = 1_048_576
MAX_PLAN_JSON_BYTES = 536_870_912
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 20_000_000
MAX_JSON_STRING_BYTES = 131_072
MAX_SOURCE_BYTES = 2_097_152
MAX_INTEGER_ABS = 2**63 - 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
_SPEC_FIELDS = frozenset({"schema_version", "source_commitment_sha256", "roles"})
_ROLE_SPEC_FIELDS = frozenset({"role", "count", "floors"})
_FLOOR_FIELDS = frozenset({"stratum", "minimum"})
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "structural_only",
        "pre_authoring_only",
        "role_assignment_precedes_text_authoring",
        "contains_authored_request_text",
        "contains_model_generated_text",
        "authored_provenance_complete",
        "joint_duplicate_closure_complete",
        "populations",
        "excluded_human_populations",
        "authorizes_model_access",
        "authorizes_label_access",
        "authorizes_cuda",
        "authorizes_jarvislabs",
        "authorizes_training",
        "authorizes_launch",
        "authorizes_execution",
        "source_commitment_sha256",
        "spec",
        "spec_sha256",
        "planner_source_sha256",
        "planner_runtime_sha256",
        "sim_program_runtime_sha256",
        "action_simulator_runtime_sha256",
        "memberships",
        "membership_sha256",
        "rosters",
        "roster_sha256",
        "records",
        "record_set_sha256",
        "artifact_body_sha256",
    }
)
_FORBIDDEN_PACKET_KEYS = frozenset(
    {
        "answer",
        "assistant",
        "case_id",
        "completion",
        "final_request",
        "gold",
        "label",
        "logits",
        "model",
        "model_id",
        "membership_sha256",
        "packet_id",
        "prediction",
        "prompt",
        "request",
        "request_text",
        "role_assignment_sha256",
        "schema_alias_family_id",
        "source_commitment_sha256",
        "target",
        "teacher",
        "tool_call",
    }
)
_AUTHORING_PACKET_FIELDS = frozenset(
    {
        "schema_version",
        "schema_presentation",
        "semantic_facts",
        "required_authoring_features",
        "prohibited_answer_leakage",
        "author_must_create_original_request_after_membership_freeze",
        "contains_authored_request_text",
        "contains_model_generated_text",
        "contains_teacher_answer",
        "authorizes_label_access",
        "authorizes_model_access",
        "authorizes_cuda",
        "authorizes_jarvislabs",
        "authorizes_training",
        "authorizes_launch",
        "authorizes_execution",
    }
)
_PROHIBITED_LEAKAGE = (
    "canonical Action IR, JSON, or any serialized answer",
    "model-facing tool names or argument-field names",
    "internal entity, family, record, packet, or lineage identifiers",
    "ACTION, CONFIRM, CLARIFY, or ABSTAIN outcome labels",
    "hashes, receipts, or membership commitments",
    "model predictions, teacher answers, logits, or candidate ranks",
    "a final request supplied by a model or hidden template",
)

_HASHLIB_SHA256 = hashlib.sha256
_JSON_DUMPS = json.dumps
_JSON_LOADS = json.loads
_JSON_DECODE_ERROR = json.JSONDecodeError
_OS_FSTAT = os.fstat
_PATH_OPEN = Path.open
_MODULE_RUNTIME_SHA256 = module_runtime_sha256
_PARSE_SIM_PROGRAM = parse_sim_program
_RUNTIME_CALLABLE_IDENTITY = runtime_callable_identity
_SIM_PROGRAM_RUNTIME_SHA256 = sim_program_runtime_sha256
_SNAPSHOT_STRICT_JSON = snapshot_strict_json
_ACTION_SIMULATOR_RUNTIME_SHA256 = action_simulator_runtime_sha256
_VERIFY_PROGRAM_ROUND_TRIP = verify_program_round_trip
_MAPPING_ABC = Mapping
_MAPPING_PROXY_TYPE = MappingProxyType
_PATH_CLASS = Path
_SIM_PROGRAM_CLASS = SimProgram
_SEMANTIC_OPERATION_SPECS = SEMANTIC_OPERATION_SPECS
_PLATFORM_PYTHON_IMPLEMENTATION = platform.python_implementation
_PLATFORM_PYTHON_VERSION = platform.python_version
_SYS_IMPLEMENTATION = sys.implementation
_BUILTIN_GETATTR = getattr
_BUILTIN_RUNTIME_BINDINGS = tuple(
    (name, _BUILTIN_GETATTR(builtins, name))
    for name in (
        "abs",
        "all",
        "any",
        "dict",
        "enumerate",
        "int",
        "isinstance",
        "iter",
        "len",
        "list",
        "max",
        "min",
        "next",
        "range",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
    )
)

_SPEC_FACTORY_TOKEN = object()
_PLAN_FACTORY_TOKEN = object()
_RECORD_FACTORY_TOKEN = object()
_PINNED_PLANNER_SOURCE_SHA256 = ""
_PINNED_PLANNER_RUNTIME_SHA256 = ""


class GVSProgramPopulationError(ValueError):
    """The pre-authoring population contract failed closed."""


def _require_exact_string(value: object, *, path: str, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        qualifier = "nonempty " if nonempty else ""
        raise GVSProgramPopulationError(f"{path} must be an exact {qualifier}string")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise GVSProgramPopulationError(f"{path} must be valid UTF-8") from error
    if size > MAX_JSON_STRING_BYTES:
        raise GVSProgramPopulationError(f"{path} exceeds the string byte bound")
    return value


def _require_sha256(value: object, *, path: str) -> str:
    text = _require_exact_string(value, path=path)
    if _SHA256_RE.fullmatch(text) is None:
        raise GVSProgramPopulationError(f"{path} must be a lowercase SHA-256")
    return text


def _require_exact_int(value: object, *, path: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise GVSProgramPopulationError(
            f"{path} must be an exact integer in [{minimum}, {maximum}]"
        )
    return value


def _canonical_bytes(value: object, *, limit: int = MAX_PLAN_JSON_BYTES) -> bytes:
    try:
        encoded = _JSON_DUMPS(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise GVSProgramPopulationError("value is not bounded strict JSON") from error
    if len(encoded) > limit:
        raise GVSProgramPopulationError(f"canonical JSON exceeds {limit} UTF-8 bytes")
    return encoded


def _sha256_json(value: object, *, limit: int = MAX_PLAN_JSON_BYTES) -> str:
    return _HASHLIB_SHA256(_canonical_bytes(value, limit=limit)).hexdigest()


def _mutable_json(value: StrictJSON) -> object:
    if isinstance(value, _MAPPING_ABC):
        return {key: _mutable_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_mutable_json(child) for child in value]
    return value


def _derive(source_sha256: str, *parts: object) -> str:
    payload = "\x1f".join((source_sha256, *(str(part) for part in parts))).encode("utf-8")
    return _HASHLIB_SHA256(payload).hexdigest()


def _ranked_ordinals(source_sha256: str, role: str, namespace: str, count: int) -> list[int]:
    return sorted(
        range(count),
        key=lambda ordinal: (_derive(source_sha256, role, namespace, ordinal), ordinal),
    )


@dataclass(frozen=True, slots=True)
class StratumFloor:
    """One immutable pre-outcome lower bound."""

    stratum: str
    minimum: int

    def __post_init__(self) -> None:
        stratum = _require_exact_string(self.stratum, path="$.stratum")
        if stratum not in REQUIRED_STRATA:
            raise GVSProgramPopulationError(f"unknown stratum {stratum!r}")
        _require_exact_int(
            self.minimum,
            path=f"$.floors[{stratum}]",
            minimum=1,
            maximum=MAX_RECORDS_PER_ROLE,
        )

    def to_dict(self) -> dict[str, object]:
        return {"stratum": self.stratum, "minimum": self.minimum}


@dataclass(frozen=True, slots=True)
class RolePlanSpec:
    """Exact count and floors for one training-role population."""

    role: str
    count: int
    floors: tuple[StratumFloor, ...]

    def __post_init__(self) -> None:
        role = _require_exact_string(self.role, path="$.role")
        if role not in TRAINING_ROLES:
            raise GVSProgramPopulationError("only T-new and D-support are accepted")
        _require_exact_int(
            self.count,
            path=f"$.roles[{role}].count",
            minimum=MIN_RECORDS_PER_ROLE,
            maximum=MAX_RECORDS_PER_ROLE,
        )
        if type(self.floors) is not tuple or any(
            type(floor) is not StratumFloor for floor in self.floors
        ):
            raise GVSProgramPopulationError("floors must be an exact tuple of StratumFloor")
        names = tuple(floor.stratum for floor in self.floors)
        if names != REQUIRED_STRATA:
            missing = sorted(set(REQUIRED_STRATA) - set(names))
            extras = sorted(set(names) - set(REQUIRED_STRATA))
            raise GVSProgramPopulationError(
                "floors must contain every required stratum once in canonical order; "
                f"missing={missing!r}, extras={extras!r}"
            )
        _validate_floor_feasibility(self.count, _floor_map(self.floors), role=role)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "count": self.count,
            "floors": [floor.to_dict() for floor in self.floors],
        }


@dataclass(frozen=True, slots=True)
class ProgramPopulationSpec:
    """Immutable input with no text, outcome observations, or runtime knobs."""

    source_commitment_sha256: str
    roles: tuple[RolePlanSpec, ...]
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _SPEC_FACTORY_TOKEN:
            raise TypeError("ProgramPopulationSpec must be created by make_program_population_spec")
        _require_sha256(self.source_commitment_sha256, path="$.source_commitment_sha256")
        if type(self.roles) is not tuple or any(
            type(role) is not RolePlanSpec for role in self.roles
        ):
            raise GVSProgramPopulationError("roles must be an exact tuple of RolePlanSpec")
        if tuple(role.role for role in self.roles) != TRAINING_ROLES:
            raise GVSProgramPopulationError("roles must be exactly T-new then D-support")
        total = sum(role.count for role in self.roles)
        if total > MAX_TOTAL_RECORDS:
            raise GVSProgramPopulationError("total record count exceeds the planner bound")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PROGRAM_POPULATION_SPEC_VERSION,
            "source_commitment_sha256": self.source_commitment_sha256,
            "roles": [role.to_dict() for role in self.roles],
        }

    def sha256(self) -> str:
        return _sha256_json(self.to_dict(), limit=MAX_SPEC_JSON_BYTES)


def make_program_population_spec(
    *,
    source_commitment_sha256: str,
    roles: tuple[RolePlanSpec, ...],
) -> ProgramPopulationSpec:
    """Create a spec while rejecting caller-owned mutable or custom containers."""

    if type(roles) is not tuple:
        raise GVSProgramPopulationError("roles must be an exact immutable tuple")
    return ProgramPopulationSpec(
        source_commitment_sha256=source_commitment_sha256,
        roles=roles,
        _factory_token=_SPEC_FACTORY_TOKEN,
    )


def _floor_map(floors: tuple[StratumFloor, ...]) -> dict[str, int]:
    return {floor.stratum: floor.minimum for floor in floors}


def _validate_floor_feasibility(count: int, floors: dict[str, int], *, role: str) -> None:
    action_floor = floors["outcome:ACTION"]
    confirm_floor = floors["outcome:CONFIRM"]
    clarify_floor = floors["outcome:CLARIFY"]
    abstain_floor = max(
        floors["outcome:ABSTAIN"],
        floors["unsafe_or_adversarial"],
    )
    safe_operation_floor = sum(floors[f"operation:{kind}"] for kind in SAFE_OPERATIONS)
    side_operation_floor = sum(floors[f"operation:{kind}"] for kind in SIDE_EFFECTING_OPERATIONS)
    action_floor = max(action_floor, safe_operation_floor)
    confirm_floor = max(
        confirm_floor,
        side_operation_floor,
        floors["timezone_or_relative_time"],
    )
    required = action_floor + confirm_floor + clarify_floor + abstain_floor
    if required > count:
        raise GVSProgramPopulationError(
            f"{role} outcome/operation/safety/time floors require {required} records, not {count}"
        )
    if floors["identity_schema"] + floors["renamed_schema"] > count:
        raise GVSProgramPopulationError(
            f"{role} identity and renamed-schema floors exceed the disjoint roster"
        )
    operational_capacity = count - clarify_floor - abstain_floor
    if floors["multi_action"] > operational_capacity:
        raise GVSProgramPopulationError(
            f"{role} multi-action floor exceeds operational-row capacity"
        )


@dataclass(frozen=True, slots=True)
class _Skeleton:
    role: str
    ordinal: int
    record_id: str
    expected_outcome: str
    primary_operation: str | None
    strata: tuple[str, ...]
    program_cluster_id: str
    authoring_slot_id: str
    entity_pool_slot_id: str
    temporal_construction_slot_id: str
    role_assignment_sha256: str

    def roster_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "ordinal": self.ordinal,
            "record_id": self.record_id,
            "expected_outcome": self.expected_outcome,
            "primary_operation": self.primary_operation,
            "strata": list(self.strata),
            "program_cluster_id": self.program_cluster_id,
            "authoring_slot_id": self.authoring_slot_id,
            "entity_pool_slot_id": self.entity_pool_slot_id,
            "temporal_construction_slot_id": self.temporal_construction_slot_id,
            "role_assignment_sha256": self.role_assignment_sha256,
        }


def _allocate_outcomes(source_sha256: str, role_spec: RolePlanSpec) -> dict[int, str]:
    floors = _floor_map(role_spec.floors)
    action_count = max(
        floors["outcome:ACTION"],
        sum(floors[f"operation:{kind}"] for kind in SAFE_OPERATIONS),
    )
    confirm_count = max(
        floors["outcome:CONFIRM"],
        sum(floors[f"operation:{kind}"] for kind in SIDE_EFFECTING_OPERATIONS),
        floors["timezone_or_relative_time"],
    )
    clarify_count = floors["outcome:CLARIFY"]
    abstain_count = max(floors["outcome:ABSTAIN"], floors["unsafe_or_adversarial"])
    counts = {
        "ACTION": action_count,
        "CONFIRM": confirm_count,
        "CLARIFY": clarify_count,
        "ABSTAIN": abstain_count,
    }
    remaining = role_spec.count - sum(counts.values())
    # Multi-action is an operational stratum, so reserve enough operational rows before balancing
    # the residual roster.  The previous allocator silently put *all* residual capacity into
    # CONFIRM; at the proposed 24k/2k scale a minimum-floor spec therefore became almost entirely
    # one policy outcome.  Water filling is deterministic, outcome-blind, and keeps every outcome
    # within one row once mandatory operation/safety floors no longer dominate.
    while counts["ACTION"] + counts["CONFIRM"] < floors["multi_action"]:
        if remaining <= 0:  # Defensive; feasibility already checked.
            raise GVSProgramPopulationError("multi-action floor became infeasible")
        selected = min(("ACTION", "CONFIRM"), key=lambda item: (counts[item], item))
        counts[selected] += 1
        remaining -= 1
    while remaining:
        selected = min(EXPECTED_OUTCOMES, key=lambda item: (counts[item], item))
        counts[selected] += 1
        remaining -= 1
    ranked = _ranked_ordinals(
        source_sha256,
        role_spec.role,
        f"outcomes:{role_spec.count}:{tuple(counts.items())}",
        role_spec.count,
    )
    assignments: dict[int, str] = {}
    cursor = 0
    for outcome in EXPECTED_OUTCOMES:
        selected = ranked[cursor : cursor + counts[outcome]]
        assignments.update({ordinal: outcome for ordinal in selected})
        cursor += counts[outcome]
    if len(assignments) != role_spec.count:
        raise GVSProgramPopulationError("outcome allocation did not cover the complete roster")
    return assignments


def _allocate_operations(
    source_sha256: str,
    role_spec: RolePlanSpec,
    outcomes: dict[int, str],
) -> dict[int, str | None]:
    floors = _floor_map(role_spec.floors)
    assignments: dict[int, str | None] = {ordinal: None for ordinal in outcomes}
    for outcome, operations in (
        ("ACTION", SAFE_OPERATIONS),
        ("CONFIRM", SIDE_EFFECTING_OPERATIONS),
    ):
        ordinals = [ordinal for ordinal, assigned in outcomes.items() if assigned == outcome]
        ordinals.sort(
            key=lambda ordinal: (
                _derive(source_sha256, role_spec.role, "operation-row", ordinal),
                ordinal,
            )
        )
        cursor = 0
        for operation in operations:
            minimum = floors[f"operation:{operation}"]
            for ordinal in ordinals[cursor : cursor + minimum]:
                assignments[ordinal] = operation
            cursor += minimum
        remaining = ordinals[cursor:]
        if outcome == "CONFIRM":
            temporal_assigned = sum(
                assignments[ordinal] in TEMPORAL_OPERATIONS for ordinal in ordinals
            )
            temporal_needed = floors["timezone_or_relative_time"] - temporal_assigned
            for index, ordinal in enumerate(remaining[: max(0, temporal_needed)]):
                assignments[ordinal] = TEMPORAL_OPERATIONS[index % len(TEMPORAL_OPERATIONS)]
            remaining = remaining[max(0, temporal_needed) :]
        for index, ordinal in enumerate(remaining):
            assignments[ordinal] = operations[index % len(operations)]
    if any(
        outcomes[ordinal] in {"ACTION", "CONFIRM"} and operation is None
        for ordinal, operation in assignments.items()
    ):
        raise GVSProgramPopulationError("operational outcome lacks a semantic operation")
    return assignments


def _select_subset(
    source_sha256: str,
    role: str,
    namespace: str,
    candidates: list[int],
    count: int,
) -> set[int]:
    ranked = sorted(
        candidates,
        key=lambda ordinal: (_derive(source_sha256, role, namespace, ordinal), ordinal),
    )
    if count > len(ranked):
        raise GVSProgramPopulationError(f"{namespace} floor exceeds eligible candidates")
    return set(ranked[:count])


def _assign_skeletons(source_sha256: str, role_spec: RolePlanSpec) -> tuple[_Skeleton, ...]:
    floors = _floor_map(role_spec.floors)
    outcomes = _allocate_outcomes(source_sha256, role_spec)
    operations = _allocate_operations(source_sha256, role_spec, outcomes)
    all_ordinals = list(range(role_spec.count))
    operational = [
        ordinal for ordinal in all_ordinals if outcomes[ordinal] in {"ACTION", "CONFIRM"}
    ]
    temporal = [ordinal for ordinal in operational if operations[ordinal] in TEMPORAL_OPERATIONS]
    abstain = [ordinal for ordinal in all_ordinals if outcomes[ordinal] == "ABSTAIN"]

    selected: dict[str, set[int]] = {}
    renamed = _select_subset(
        source_sha256,
        role_spec.role,
        "renamed-schema",
        all_ordinals,
        floors["renamed_schema"],
    )
    identity_candidates = [ordinal for ordinal in all_ordinals if ordinal not in renamed]
    if len(identity_candidates) < floors["identity_schema"]:
        raise GVSProgramPopulationError("identity/renamed schema partition is infeasible")
    selected["renamed_schema"] = renamed
    selected["identity_schema"] = set(identity_candidates)
    selected["unsafe_or_adversarial"] = _select_subset(
        source_sha256,
        role_spec.role,
        "unsafe",
        abstain,
        floors["unsafe_or_adversarial"],
    )
    selected["timezone_or_relative_time"] = _select_subset(
        source_sha256,
        role_spec.role,
        "timezone",
        temporal,
        floors["timezone_or_relative_time"],
    )
    selected["multi_action"] = _select_subset(
        source_sha256,
        role_spec.role,
        "multi-action",
        operational,
        floors["multi_action"],
    )
    for feature in (
        "context_grounding",
        "revision",
        "disfluency",
        "distractor_tools",
    ):
        selected[feature] = _select_subset(
            source_sha256,
            role_spec.role,
            feature,
            all_ordinals,
            floors[feature],
        )

    skeletons: list[_Skeleton] = []
    role_code = "t" if role_spec.role == "T-new" else "d"
    for ordinal in all_ordinals:
        root = _derive(source_sha256, "membership", role_spec.role, ordinal)
        record_id = f"gvs.{role_code}.{root[:24]}"
        strata = [f"outcome:{outcomes[ordinal]}"]
        operation = operations[ordinal]
        if operation is not None:
            strata.append(f"operation:{operation}")
        strata.extend(feature for feature in FEATURE_STRATA if ordinal in selected[feature])
        strata_tuple = tuple(sorted(strata))
        body = {
            "role": role_spec.role,
            "ordinal": ordinal,
            "record_id": record_id,
            "expected_outcome": outcomes[ordinal],
            "primary_operation": operation,
            "strata": list(strata_tuple),
            "program_cluster_id": f"program.{_derive(source_sha256, role_spec.role, 'program', ordinal)[:24]}",
            # These are allocation slots, not post-authoring provenance/family claims.  Actual
            # template/entity/temporal components must be inferred later from collected evidence.
            "authoring_slot_id": f"authoring.{_derive(source_sha256, role_spec.role, 'authoring', ordinal)[:24]}",
            "entity_pool_slot_id": f"entity.{_derive(source_sha256, role_spec.role, 'entity', ordinal)[:24]}",
            "temporal_construction_slot_id": f"temporal.{_derive(source_sha256, role_spec.role, 'temporal', ordinal)[:24]}",
        }
        skeletons.append(
            _Skeleton(
                **body,
                role_assignment_sha256=_sha256_json(body, limit=MAX_SPEC_JSON_BYTES),
            )
        )
    return tuple(skeletons)


def _assert_disjoint_allocation_ids(skeletons: tuple[_Skeleton, ...]) -> None:
    for field in (
        "record_id",
        "program_cluster_id",
        "authoring_slot_id",
        "entity_pool_slot_id",
        "temporal_construction_slot_id",
    ):
        values_by_role = {
            role: {getattr(item, field) for item in skeletons if item.role == role}
            for role in TRAINING_ROLES
        }
        if values_by_role["T-new"] & values_by_role["D-support"]:
            raise GVSProgramPopulationError(f"cross-role {field} allocation collision detected")
        total = sum(len(values) for values in values_by_role.values())
        if total != len(skeletons):
            raise GVSProgramPopulationError(f"duplicate {field} within a role")


def _world_payload(skeleton: _Skeleton) -> dict[str, object]:
    token = _derive(skeleton.role_assignment_sha256, "world")
    # Opaque IDs are simulator-only.  Human-facing facts deliberately use ordinary names rather
    # than exposing a digest fragment that would become an easy generator/source shortcut.
    suffix = token[:24]
    names = (
        "Asha Rao",
        "Dev Mehta",
        "Ira Sen",
        "Kabir Shah",
        "Leela Nair",
        "Maya Bose",
        "Neel Joshi",
        "Rohan Das",
        "Sara Iyer",
        "Tara Kapur",
        "Uma Jain",
        "Vikram Roy",
    )
    projects = (
        "Quarterly report",
        "Travel budget",
        "Design brief",
        "Hiring plan",
        "Research notes",
        "Launch checklist",
        "Client proposal",
        "Workshop outline",
    )
    places = (
        "Home",
        "Office",
        "City Library",
        "Central Station",
        "North Clinic",
        "Riverside Cafe",
        "Market Square",
        "Community Hall",
    )
    tracks = (
        "Morning Raga",
        "Blue Hour",
        "Quiet River",
        "Monsoon Light",
        "Evening Walk",
        "Paper Lanterns",
        "Open Sky",
        "First Train",
    )
    name_index = int(token[22:24], 16) % len(names)
    project_index = int(token[24:26], 16) % len(projects)
    place_index = int(token[26:28], 16) % len(places)
    track_index = int(token[28:30], 16) % len(tracks)
    day = 5 + int(token[10:12], 16) % 20
    hour = 10 + int(token[12:14], 16) % 8
    due_at = f"2026-08-{day:02d}T{hour:02d}:00:00+05:30"
    ends_at = f"2026-08-{day:02d}T{hour + 1:02d}:00:00+05:30"
    return {
        "schema_version": WORLD_STATE_VERSION,
        "reference_time": "2026-08-04T09:30:00+05:30",
        "timezone": "Asia/Kolkata",
        "reminders": {
            f"reminder.old.{suffix}": {
                "title": projects[project_index],
                "due_at": "2026-08-05T17:00:00+05:30",
                "completed": False,
            }
        },
        "calendar": {
            f"event.old.{suffix}": {
                "title": f"Review: {projects[(project_index + 1) % len(projects)]}",
                "start_at": "2026-08-04T10:00:00+05:30",
                "end_at": "2026-08-04T10:30:00+05:30",
            }
        },
        "contacts": {
            f"contact.first.{suffix}": {"name": names[name_index], "channel": "sms"},
            f"contact.second.{suffix}": {
                "name": names[(name_index + 1) % len(names)],
                "channel": "sms",
            },
        },
        "notes": {
            f"note.context.{suffix}": {
                "title": f"Notes: {projects[(project_index + 2) % len(projects)]}",
                "body": "A supplied local note with relevant context",
            }
        },
        "lists": {
            f"list.groceries.{suffix}": {
                "title": "Groceries",
                "items": {
                    f"item.milk.{suffix}": {"text": "Milk", "checked": False},
                },
            }
        },
        "places": {
            f"place.origin.{suffix}": {"name": places[place_index]},
            f"place.destination.{suffix}": {"name": places[(place_index + 1) % len(places)]},
        },
        "routes": {
            f"route.commute.{suffix}": {
                "origin_id": f"place.origin.{suffix}",
                "destination_id": f"place.destination.{suffix}",
                "mode": "driving",
                "distance_m": 8000 + int(token[14:18], 16) % 2000,
                "duration_s": 1200 + int(token[18:22], 16) % 900,
            }
        },
        "outbox": [],
        "media": {
            "status": "playing",
            "track_id": f"track.first.{suffix}",
            "catalog": {
                f"track.first.{suffix}": {"title": tracks[track_index]},
                f"track.second.{suffix}": {"title": tracks[(track_index + 1) % len(tracks)]},
            },
        },
        "settings": {f"wifi.{suffix}": True, f"airplane.{suffix}": False},
        "_derived_due_at": due_at,
        "_derived_ends_at": ends_at,
    }


def _operation_payload(
    kind: str, world: dict[str, object], *, secondary: bool = False
) -> dict[str, object]:
    reminders = world["reminders"]
    calendar = world["calendar"]
    contacts = world["contacts"]
    lists = world["lists"]
    places = world["places"]
    media = world["media"]
    settings = world["settings"]
    assert isinstance(reminders, dict)
    assert isinstance(calendar, dict)
    assert isinstance(contacts, dict)
    assert isinstance(lists, dict)
    assert isinstance(places, dict)
    assert isinstance(media, dict)
    assert isinstance(settings, dict)
    suffix = next(iter(reminders)).rsplit(".", 1)[-1]
    due_at = world["_derived_due_at"]
    ends_at = world["_derived_ends_at"]
    existing_reminder = next(iter(reminders))
    existing_event = next(iter(calendar))
    contact_ids = tuple(contacts)
    list_id = next(iter(lists))
    list_value = lists[list_id]
    assert isinstance(list_value, dict)
    item_id = next(iter(list_value["items"]))
    place_ids = tuple(places)
    catalog = media["catalog"]
    assert isinstance(catalog, dict)
    track_ids = tuple(catalog)
    setting_id = next(iter(settings))
    new_suffix = "secondary" if secondary else "target"
    reminder_title = str(reminders[existing_reminder]["title"])
    event_title = str(calendar[existing_event]["title"])
    parameters: dict[str, object]
    if kind == "CREATE_REMINDER":
        parameters = {
            "reminder_ref": f"reminder.{new_suffix}.{suffix}",
            "summary": "Submit the travel form",
            "due_at": due_at,
        }
    elif kind == "UPDATE_REMINDER":
        parameters = {
            "reminder_ref": existing_reminder,
            "summary": f"Revised {reminder_title.lower()}",
            "due_at": due_at,
        }
    elif kind == "CREATE_CALENDAR_EVENT":
        parameters = {
            "event_ref": f"event.{new_suffix}.{suffix}",
            "summary": f"Follow-up: {event_title}",
            "begins_at": due_at,
            "ends_at": ends_at,
        }
    elif kind == "RESCHEDULE_CALENDAR_EVENT":
        parameters = {"event_ref": existing_event, "begins_at": due_at, "ends_at": ends_at}
    elif kind == "LOOK_UP_CONTACT":
        parameters = {"contact_ref": contact_ids[0]}
    elif kind == "CREATE_NOTE":
        parameters = {
            "note_ref": f"note.{new_suffix}.{suffix}",
            "heading": "Packing list",
            "content": "Passport, charger, and notebook",
        }
    elif kind == "ADD_LIST_ITEM":
        parameters = {
            "list_ref": list_id,
            "item_ref": f"item.{new_suffix}.{suffix}",
            "item_text": "Tea",
        }
    elif kind == "SET_LIST_ITEM_CHECKED":
        parameters = {"list_ref": list_id, "item_ref": item_id, "is_checked": True}
    elif kind == "LOOK_UP_ROUTE":
        parameters = {
            "from_place_ref": place_ids[0],
            "to_place_ref": place_ids[1],
            "travel_mode": "driving",
        }
    elif kind == "PROPOSE_MESSAGE":
        parameters = {
            "message_ref": f"message.{new_suffix}.{suffix}",
            "recipient_ref": contact_ids[1],
            "delivery_channel": "sms",
            "content": "I will arrive at six.",
        }
    elif kind == "PLAY_MEDIA":
        parameters = {"track_ref": track_ids[1]}
    elif kind == "PAUSE_MEDIA":
        parameters = {}
    elif kind == "SET_BOOLEAN_SETTING":
        parameters = {"setting_ref": setting_id, "value": False}
    else:  # Defensive parity with the imported semantic contract.
        raise GVSProgramPopulationError(f"unsupported semantic operation {kind!r}")
    return {"kind": kind, "parameters": parameters}


def _secondary_operation(primary: str) -> str:
    if primary == "LOOK_UP_CONTACT":
        return "LOOK_UP_ROUTE"
    if primary == "LOOK_UP_ROUTE":
        return "LOOK_UP_CONTACT"
    if primary == "SET_BOOLEAN_SETTING":
        return "CREATE_NOTE"
    return "SET_BOOLEAN_SETTING"


def _program_payload(skeleton: _Skeleton) -> dict[str, object]:
    world = _world_payload(skeleton)
    due_at = world.pop("_derived_due_at")
    ends_at = world.pop("_derived_ends_at")
    world["_derived_due_at"] = due_at
    world["_derived_ends_at"] = ends_at
    operations: list[dict[str, object]] = []
    mode: str | None = None
    missing: list[str] = []
    decision: str
    if skeleton.expected_outcome in {"ACTION", "CONFIRM"}:
        assert skeleton.primary_operation is not None
        operations.append(_operation_payload(skeleton.primary_operation, world))
        if "multi_action" in skeleton.strata:
            operations.append(
                _operation_payload(
                    _secondary_operation(skeleton.primary_operation), world, secondary=True
                )
            )
            mode = "PARALLEL"
        else:
            mode = "SINGLE"
        decision = "CALL" if skeleton.expected_outcome == "ACTION" else "CONFIRM"
    elif skeleton.expected_outcome == "CLARIFY":
        decision = "CLARIFY"
        missing = ["recipient"]
    else:
        decision = "ABSTAIN"
    world.pop("_derived_due_at")
    world.pop("_derived_ends_at")
    return {
        "schema_version": SIM_PROGRAM_VERSION,
        "case_id": skeleton.record_id,
        "initial_state": world,
        "steps": [
            {
                "step_id": f"step.{skeleton.ordinal}",
                "decision": decision,
                "mode": mode,
                "operations": operations,
                "missing": missing,
            }
        ],
    }


def _display_facts(skeleton: _Skeleton, program: SimProgram) -> tuple[tuple[str, str], ...]:
    facts: list[tuple[str, str]] = []
    state = program.initial_state
    if skeleton.expected_outcome == "CLARIFY":
        facts.extend(
            (
                ("intended_behavior", "ask for the missing recipient before proposing a message"),
                ("missing_information", "the intended recipient is not supplied"),
            )
        )
    elif skeleton.expected_outcome == "ABSTAIN":
        if "unsafe_or_adversarial" in skeleton.strata:
            facts.extend(
                (
                    ("intended_behavior", "decline without proposing or executing an action"),
                    ("safety_condition", "the requested behavior is unsafe or unauthorized"),
                )
            )
        else:
            facts.extend(
                (
                    ("intended_behavior", "decline without proposing an action"),
                    ("unsupported_condition", "the requested capability is outside the schema"),
                )
            )
    else:
        step = program.steps[0]
        for index, operation in enumerate(step.operations):
            prefix = "primary" if index == 0 else f"additional_{index}"
            p = operation.parameters
            kind = operation.kind
            facts.append((f"{prefix}_intent", kind.lower().replace("_", " ")))
            if kind in {"CREATE_REMINDER", "UPDATE_REMINDER"}:
                facts.extend(
                    (
                        (f"{prefix}_title", str(p["summary"])),
                        (f"{prefix}_resolved_due_time", str(p["due_at"])),
                    )
                )
            elif kind == "CREATE_CALENDAR_EVENT":
                facts.extend(
                    (
                        (f"{prefix}_event_title", str(p["summary"])),
                        (f"{prefix}_resolved_start", str(p["begins_at"])),
                        (f"{prefix}_resolved_end", str(p["ends_at"])),
                    )
                )
            elif kind == "RESCHEDULE_CALENDAR_EVENT":
                event = state.calendar[str(p["event_ref"])]
                facts.extend(
                    (
                        (f"{prefix}_existing_event", str(event["title"])),
                        (f"{prefix}_resolved_start", str(p["begins_at"])),
                        (f"{prefix}_resolved_end", str(p["ends_at"])),
                    )
                )
            elif kind == "LOOK_UP_CONTACT":
                facts.append(
                    (f"{prefix}_contact_name", str(state.contacts[str(p["contact_ref"])]["name"]))
                )
            elif kind == "CREATE_NOTE":
                facts.extend(
                    (
                        (f"{prefix}_note_title", str(p["heading"])),
                        (f"{prefix}_note_body", str(p["content"])),
                    )
                )
            elif kind in {"ADD_LIST_ITEM", "SET_LIST_ITEM_CHECKED"}:
                list_value = state.lists[str(p["list_ref"])]
                facts.append((f"{prefix}_list_title", str(list_value["title"])))
                if kind == "ADD_LIST_ITEM":
                    facts.append((f"{prefix}_item_text", str(p["item_text"])))
                else:
                    item = list_value["items"][str(p["item_ref"])]
                    facts.extend(
                        (
                            (f"{prefix}_item_text", str(item["text"])),
                            (f"{prefix}_checked_state", str(p["is_checked"]).lower()),
                        )
                    )
            elif kind == "LOOK_UP_ROUTE":
                facts.extend(
                    (
                        (
                            f"{prefix}_origin",
                            str(state.places[str(p["from_place_ref"])]["name"]),
                        ),
                        (
                            f"{prefix}_destination",
                            str(state.places[str(p["to_place_ref"])]["name"]),
                        ),
                        (f"{prefix}_travel_mode", str(p["travel_mode"])),
                    )
                )
            elif kind == "PROPOSE_MESSAGE":
                contact = state.contacts[str(p["recipient_ref"])]
                facts.extend(
                    (
                        (f"{prefix}_recipient_name", str(contact["name"])),
                        (f"{prefix}_channel", str(p["delivery_channel"])),
                        (f"{prefix}_message_content", str(p["content"])),
                    )
                )
            elif kind == "PLAY_MEDIA":
                track = state.media["catalog"][str(p["track_ref"])]
                facts.append((f"{prefix}_track_title", str(track["title"])))
            elif kind == "PAUSE_MEDIA":
                track_id = state.media["track_id"]
                assert isinstance(track_id, str)
                track = state.media["catalog"][track_id]
                facts.append((f"{prefix}_currently_playing", str(track["title"])))
            elif kind == "SET_BOOLEAN_SETTING":
                facts.extend(
                    (
                        (f"{prefix}_setting_display_name", "Wi-Fi"),
                        (f"{prefix}_desired_state", str(p["value"]).lower()),
                    )
                )
    if "context_grounding" in skeleton.strata:
        facts.append(
            (
                "context_rule",
                "refer to a supplied entity by its human-facing name, never its internal ID",
            )
        )
    if "revision" in skeleton.strata:
        facts.extend(
            (
                ("discarded_revision_fact", "an earlier value is explicitly corrected"),
                ("revision_rule", "only the replacement value in the semantic facts is binding"),
            )
        )
    if "timezone_or_relative_time" in skeleton.strata:
        facts.extend(
            (
                ("reference_time", state.reference_time),
                ("reference_timezone", state.timezone),
                ("temporal_expression_rule", "express the resolved target using relative wording"),
            )
        )
    return tuple(facts)


def _authoring_packet(
    skeleton: _Skeleton,
    program: SimProgram,
) -> Mapping[str, StrictJSON]:
    schema_presentation = "renamed" if "renamed_schema" in skeleton.strata else "identity"
    required_features = tuple(
        feature
        for feature in FEATURE_STRATA
        if feature in skeleton.strata and feature not in {"identity_schema", "renamed_schema"}
    )
    facts = _display_facts(skeleton, program)
    packet_body: dict[str, object] = {
        "schema_version": AUTHORING_PACKET_VERSION,
        "schema_presentation": schema_presentation,
        "semantic_facts": [{"name": name, "value": value} for name, value in facts],
        "required_authoring_features": list(required_features),
        "prohibited_answer_leakage": list(_PROHIBITED_LEAKAGE),
        "author_must_create_original_request_after_membership_freeze": True,
        "contains_authored_request_text": False,
        "contains_model_generated_text": False,
        "contains_teacher_answer": False,
        "authorizes_label_access": False,
        "authorizes_model_access": False,
        "authorizes_cuda": False,
        "authorizes_jarvislabs": False,
        "authorizes_training": False,
        "authorizes_launch": False,
        "authorizes_execution": False,
    }
    forbidden = _FORBIDDEN_PACKET_KEYS & set(packet_body)
    if forbidden:
        raise GVSProgramPopulationError(
            f"authoring packet leaked forbidden fields: {sorted(forbidden)}"
        )
    if set(packet_body) != _AUTHORING_PACKET_FIELDS:
        raise GVSProgramPopulationError("authoring packet fields differ from the public allowlist")
    packet = _SNAPSHOT_STRICT_JSON(packet_body)
    if not isinstance(packet, _MAPPING_PROXY_TYPE):
        raise GVSProgramPopulationError("authoring packet did not become immutable")
    return packet


def _record_body_dict(
    *,
    skeleton: _Skeleton,
    program: SimProgram,
    authoring_packet: Mapping[str, StrictJSON],
    program_sha256: str,
    world_state_sha256: str,
    round_trip_receipt_sha256: str,
    authoring_packet_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": PROGRAM_POPULATION_RECORD_VERSION,
        **skeleton.roster_dict(),
        "program": program.to_dict(),
        "program_sha256": program_sha256,
        "world_state_sha256": world_state_sha256,
        "round_trip_receipt_sha256": round_trip_receipt_sha256,
        "round_trip_passed": True,
        "authoring_packet": _mutable_json(authoring_packet),
        "authoring_packet_sha256": authoring_packet_sha256,
        "pre_authoring_only": True,
        "authorizes_model_access": False,
        "authorizes_label_access": False,
        "authorizes_cuda": False,
        "authorizes_jarvislabs": False,
        "authorizes_training": False,
        "authorizes_launch": False,
        "authorizes_execution": False,
    }


@dataclass(frozen=True, slots=True)
class ProgramPopulationRecord:
    """One immutable roster assignment, exact program, and authoring packet."""

    skeleton: _Skeleton
    program: SimProgram
    authoring_packet: Mapping[str, StrictJSON]
    program_sha256: str
    world_state_sha256: str
    round_trip_receipt_sha256: str
    authoring_packet_sha256: str
    record_sha256: str
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _RECORD_FACTORY_TOKEN:
            raise TypeError("ProgramPopulationRecord must be created by the planner")
        if type(self.skeleton) is not _Skeleton or type(self.program) is not _SIM_PROGRAM_CLASS:
            raise TypeError("record contains an invalid skeleton or program")
        if not isinstance(self.authoring_packet, _MAPPING_PROXY_TYPE):
            raise TypeError("planner authoring packet must be a detached immutable mapping")
        for name, value in (
            ("program_sha256", self.program_sha256),
            ("world_state_sha256", self.world_state_sha256),
            ("round_trip_receipt_sha256", self.round_trip_receipt_sha256),
            ("authoring_packet_sha256", self.authoring_packet_sha256),
            ("record_sha256", self.record_sha256),
        ):
            _require_sha256(value, path=f"$.{name}")

        expected_program = _PARSE_SIM_PROGRAM(_program_payload(self.skeleton))
        if self.program.to_dict() != expected_program.to_dict():
            raise GVSProgramPopulationError("record program differs from its roster assignment")
        expected_round_trip = _VERIFY_PROGRAM_ROUND_TRIP(expected_program)
        if expected_round_trip.passed is not True:
            raise GVSProgramPopulationError("record program failed live round-trip verification")
        expected_packet = _authoring_packet(self.skeleton, expected_program)
        expected_packet_json = _mutable_json(expected_packet)
        if _mutable_json(self.authoring_packet) != expected_packet_json:
            raise GVSProgramPopulationError(
                "record authoring packet differs from its roster assignment"
            )
        expected_hashes = {
            "program_sha256": expected_program.sha256(),
            "world_state_sha256": expected_program.initial_state.sha256(),
            "round_trip_receipt_sha256": expected_round_trip.sha256(),
            "authoring_packet_sha256": _sha256_json(
                expected_packet_json,
                limit=MAX_SPEC_JSON_BYTES,
            ),
        }
        for name, expected in expected_hashes.items():
            if getattr(self, name) != expected:
                raise GVSProgramPopulationError(f"record {name} differs from live recomputation")
        if self.record_sha256 != _sha256_json(
            _record_body_dict(
                skeleton=self.skeleton,
                program=self.program,
                authoring_packet=self.authoring_packet,
                program_sha256=self.program_sha256,
                world_state_sha256=self.world_state_sha256,
                round_trip_receipt_sha256=self.round_trip_receipt_sha256,
                authoring_packet_sha256=self.authoring_packet_sha256,
            )
        ):
            raise GVSProgramPopulationError("record hash differs from live recomputation")

    def roster_dict(self) -> dict[str, object]:
        return self.skeleton.roster_dict()

    def body_dict(self) -> dict[str, object]:
        return _record_body_dict(
            skeleton=self.skeleton,
            program=self.program,
            authoring_packet=self.authoring_packet,
            program_sha256=self.program_sha256,
            world_state_sha256=self.world_state_sha256,
            round_trip_receipt_sha256=self.round_trip_receipt_sha256,
            authoring_packet_sha256=self.authoring_packet_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {**self.body_dict(), "record_sha256": self.record_sha256}


def _plan_body_dict(
    *,
    spec: ProgramPopulationSpec,
    records: tuple[ProgramPopulationRecord, ...],
    planner_source_sha256: str,
    planner_runtime_sha256: str,
    simulator_program_runtime_sha256: str,
    simulator_action_runtime_sha256: str,
    membership_sha256: str,
    roster_sha256: str,
    record_set_sha256: str,
) -> dict[str, object]:
    memberships = {
        role: [record.skeleton.record_id for record in records if record.skeleton.role == role]
        for role in TRAINING_ROLES
    }
    rosters = {
        role: [record.roster_dict() for record in records if record.skeleton.role == role]
        for role in TRAINING_ROLES
    }
    return {
        "schema_version": PROGRAM_POPULATION_PLAN_VERSION,
        "structural_only": True,
        "pre_authoring_only": True,
        "role_assignment_precedes_text_authoring": True,
        "contains_authored_request_text": False,
        "contains_model_generated_text": False,
        # Actual author/template/source/near-duplicate lineages do not exist until requests
        # are authored and scanned jointly with S-new/C-new.  Per-record allocation IDs must
        # never be misrepresented as that later component closure.
        "authored_provenance_complete": False,
        "joint_duplicate_closure_complete": False,
        "populations": list(TRAINING_ROLES),
        "excluded_human_populations": list(HUMAN_ROLES_EXCLUDED),
        "authorizes_model_access": False,
        "authorizes_label_access": False,
        "authorizes_cuda": False,
        "authorizes_jarvislabs": False,
        "authorizes_training": False,
        "authorizes_launch": False,
        "authorizes_execution": False,
        "source_commitment_sha256": spec.source_commitment_sha256,
        "spec": spec.to_dict(),
        "spec_sha256": spec.sha256(),
        "planner_source_sha256": planner_source_sha256,
        "planner_runtime_sha256": planner_runtime_sha256,
        "sim_program_runtime_sha256": simulator_program_runtime_sha256,
        "action_simulator_runtime_sha256": simulator_action_runtime_sha256,
        "memberships": memberships,
        "membership_sha256": membership_sha256,
        "rosters": rosters,
        "roster_sha256": roster_sha256,
        "records": [record.to_dict() for record in records],
        "record_set_sha256": record_set_sha256,
    }


@dataclass(frozen=True, slots=True)
class ProgramPopulationPlan:
    """A deterministic, nonauthorizing T-new/D-support pre-authoring artifact."""

    spec: ProgramPopulationSpec
    records: tuple[ProgramPopulationRecord, ...]
    planner_source_sha256: str
    planner_runtime_sha256: str
    simulator_program_runtime_sha256: str
    simulator_action_runtime_sha256: str
    membership_sha256: str
    roster_sha256: str
    record_set_sha256: str
    artifact_body_sha256: str
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _PLAN_FACTORY_TOKEN:
            raise TypeError("ProgramPopulationPlan must be created by the planner or loader")
        if type(self.spec) is not ProgramPopulationSpec:
            raise TypeError("plan spec has the wrong type")
        if type(self.records) is not tuple or any(
            type(record) is not ProgramPopulationRecord for record in self.records
        ):
            raise TypeError("plan records must be an exact tuple")
        if len(self.records) != sum(role.count for role in self.spec.roles):
            raise GVSProgramPopulationError("plan record count differs from its spec")
        for name, value in (
            ("planner_source_sha256", self.planner_source_sha256),
            ("planner_runtime_sha256", self.planner_runtime_sha256),
            ("simulator_program_runtime_sha256", self.simulator_program_runtime_sha256),
            ("simulator_action_runtime_sha256", self.simulator_action_runtime_sha256),
            ("membership_sha256", self.membership_sha256),
            ("roster_sha256", self.roster_sha256),
            ("record_set_sha256", self.record_set_sha256),
            ("artifact_body_sha256", self.artifact_body_sha256),
        ):
            _require_sha256(value, path=f"$.{name}")

        expected_skeletons = tuple(
            skeleton
            for role_spec in self.spec.roles
            for skeleton in _assign_skeletons(self.spec.source_commitment_sha256, role_spec)
        )
        if tuple(record.skeleton for record in self.records) != expected_skeletons:
            raise GVSProgramPopulationError(
                "plan record roster differs from deterministic spec allocation"
            )
        expected_membership_sha256 = _sha256_json(self.memberships)
        expected_roster_sha256 = _sha256_json(self.rosters)
        expected_record_set_sha256 = _sha256_json(
            [
                {
                    "record_id": record.skeleton.record_id,
                    "record_sha256": record.record_sha256,
                }
                for record in self.records
            ]
        )
        expected_runtime_hashes = {
            "planner_source_sha256": _planner_source_sha256(),
            "planner_runtime_sha256": planner_runtime_sha256(),
            "simulator_program_runtime_sha256": _SIM_PROGRAM_RUNTIME_SHA256(),
            "simulator_action_runtime_sha256": _ACTION_SIMULATOR_RUNTIME_SHA256(),
            "membership_sha256": expected_membership_sha256,
            "roster_sha256": expected_roster_sha256,
            "record_set_sha256": expected_record_set_sha256,
        }
        for name, expected in expected_runtime_hashes.items():
            if getattr(self, name) != expected:
                raise GVSProgramPopulationError(f"plan {name} differs from live recomputation")
        if self.artifact_body_sha256 != _sha256_json(
            _plan_body_dict(
                spec=self.spec,
                records=self.records,
                planner_source_sha256=self.planner_source_sha256,
                planner_runtime_sha256=self.planner_runtime_sha256,
                simulator_program_runtime_sha256=self.simulator_program_runtime_sha256,
                simulator_action_runtime_sha256=self.simulator_action_runtime_sha256,
                membership_sha256=self.membership_sha256,
                roster_sha256=self.roster_sha256,
                record_set_sha256=self.record_set_sha256,
            )
        ):
            raise GVSProgramPopulationError("plan artifact body hash differs from recomputation")

    @property
    def memberships(self) -> dict[str, list[str]]:
        return {
            role: [
                record.skeleton.record_id for record in self.records if record.skeleton.role == role
            ]
            for role in TRAINING_ROLES
        }

    @property
    def rosters(self) -> dict[str, list[dict[str, object]]]:
        return {
            role: [record.roster_dict() for record in self.records if record.skeleton.role == role]
            for role in TRAINING_ROLES
        }

    def body_dict(self) -> dict[str, object]:
        return _plan_body_dict(
            spec=self.spec,
            records=self.records,
            planner_source_sha256=self.planner_source_sha256,
            planner_runtime_sha256=self.planner_runtime_sha256,
            simulator_program_runtime_sha256=self.simulator_program_runtime_sha256,
            simulator_action_runtime_sha256=self.simulator_action_runtime_sha256,
            membership_sha256=self.membership_sha256,
            roster_sha256=self.roster_sha256,
            record_set_sha256=self.record_set_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {**self.body_dict(), "artifact_body_sha256": self.artifact_body_sha256}

    def canonical_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    def sha256(self) -> str:
        return _HASHLIB_SHA256(self.canonical_json().encode("utf-8")).hexdigest()


def _planner_source_sha256() -> str:
    path = _PATH_CLASS(__file__).resolve(strict=True)
    try:
        with _PATH_OPEN(path, "rb") as handle:
            before = _OS_FSTAT(handle.fileno())
            payload = handle.read(MAX_SOURCE_BYTES + 1)
            after = _OS_FSTAT(handle.fileno())
    except OSError as error:
        raise GVSProgramPopulationError("could not snapshot planner source") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise GVSProgramPopulationError("planner source changed while being hashed")
    if len(payload) > MAX_SOURCE_BYTES:
        raise GVSProgramPopulationError("planner source exceeds its byte bound")
    return _HASHLIB_SHA256(payload).hexdigest()


def _assert_import_bindings_unchanged() -> None:
    for name, pinned in _BUILTIN_RUNTIME_BINDINGS:
        if (
            _BUILTIN_GETATTR(builtins, name, None) is not pinned
            or globals().get(name, pinned) is not pinned
        ):
            raise GVSProgramPopulationError(
                f"planner runtime builtin {name} differs from its import-time identity"
            )
    bindings = (
        ("hashlib.sha256", getattr(hashlib, "sha256", None), _HASHLIB_SHA256),
        ("json.dumps", getattr(json, "dumps", None), _JSON_DUMPS),
        ("json.loads", getattr(json, "loads", None), _JSON_LOADS),
        ("json.JSONDecodeError", getattr(json, "JSONDecodeError", None), _JSON_DECODE_ERROR),
        ("os.fstat", getattr(os, "fstat", None), _OS_FSTAT),
        ("Path", Path, _PATH_CLASS),
        ("Path.open", getattr(Path, "open", None), _PATH_OPEN),
        ("Mapping", Mapping, _MAPPING_ABC),
        ("MappingProxyType", MappingProxyType, _MAPPING_PROXY_TYPE),
        ("SimProgram", SimProgram, _SIM_PROGRAM_CLASS),
        ("SEMANTIC_OPERATION_SPECS", SEMANTIC_OPERATION_SPECS, _SEMANTIC_OPERATION_SPECS),
        ("module_runtime_sha256", module_runtime_sha256, _MODULE_RUNTIME_SHA256),
        ("parse_sim_program", parse_sim_program, _PARSE_SIM_PROGRAM),
        ("runtime_callable_identity", runtime_callable_identity, _RUNTIME_CALLABLE_IDENTITY),
        ("sim_program_runtime_sha256", sim_program_runtime_sha256, _SIM_PROGRAM_RUNTIME_SHA256),
        ("snapshot_strict_json", snapshot_strict_json, _SNAPSHOT_STRICT_JSON),
        (
            "action_simulator_runtime_sha256",
            action_simulator_runtime_sha256,
            _ACTION_SIMULATOR_RUNTIME_SHA256,
        ),
        ("verify_program_round_trip", verify_program_round_trip, _VERIFY_PROGRAM_ROUND_TRIP),
        (
            "platform.python_implementation",
            getattr(platform, "python_implementation", None),
            _PLATFORM_PYTHON_IMPLEMENTATION,
        ),
        (
            "platform.python_version",
            getattr(platform, "python_version", None),
            _PLATFORM_PYTHON_VERSION,
        ),
        ("sys.implementation", getattr(sys, "implementation", None), _SYS_IMPLEMENTATION),
    )
    for name, current, pinned in bindings:
        if current is not pinned:
            raise GVSProgramPopulationError(
                f"planner runtime import {name} differs from its import-time identity"
            )


def planner_runtime_sha256() -> str:
    _assert_import_bindings_unchanged()
    dependencies = {
        name: dict(_RUNTIME_CALLABLE_IDENTITY(value))
        for name, value in (
            ("hashlib.sha256", _HASHLIB_SHA256),
            ("json.dumps", _JSON_DUMPS),
            ("json.loads", _JSON_LOADS),
            ("parse_sim_program", _PARSE_SIM_PROGRAM),
            ("snapshot_strict_json", _SNAPSHOT_STRICT_JSON),
            ("verify_program_round_trip", _VERIFY_PROGRAM_ROUND_TRIP),
        )
    }
    contract = {
        "schema_versions": {
            "spec": PROGRAM_POPULATION_SPEC_VERSION,
            "plan": PROGRAM_POPULATION_PLAN_VERSION,
            "record": PROGRAM_POPULATION_RECORD_VERSION,
            "packet": AUTHORING_PACKET_VERSION,
        },
        "roles": list(TRAINING_ROLES),
        "excluded_roles": list(HUMAN_ROLES_EXCLUDED),
        "operations": list(SEMANTIC_OPERATIONS),
        "safe_operations": list(SAFE_OPERATIONS),
        "side_effecting_operations": list(SIDE_EFFECTING_OPERATIONS),
        "temporal_operations": list(TEMPORAL_OPERATIONS),
        "outcomes": list(EXPECTED_OUTCOMES),
        "required_strata": list(REQUIRED_STRATA),
        "limits": {
            "minimum_records_per_role": MIN_RECORDS_PER_ROLE,
            "maximum_records_per_role": MAX_RECORDS_PER_ROLE,
            "maximum_total_records": MAX_TOTAL_RECORDS,
            "maximum_spec_json_bytes": MAX_SPEC_JSON_BYTES,
            "maximum_plan_json_bytes": MAX_PLAN_JSON_BYTES,
            "maximum_json_depth": MAX_JSON_DEPTH,
            "maximum_json_nodes": MAX_JSON_NODES,
            "maximum_json_string_bytes": MAX_JSON_STRING_BYTES,
            "maximum_source_bytes": MAX_SOURCE_BYTES,
        },
        "forbidden_packet_keys": sorted(_FORBIDDEN_PACKET_KEYS),
        "dependencies": dependencies,
        "sim_program_runtime_sha256": _SIM_PROGRAM_RUNTIME_SHA256(),
        "action_simulator_runtime_sha256": _ACTION_SIMULATOR_RUNTIME_SHA256(),
        "python": {
            "implementation": _PLATFORM_PYTHON_IMPLEMENTATION(),
            "version": _PLATFORM_PYTHON_VERSION(),
            "cache_tag": _SYS_IMPLEMENTATION.cache_tag,
        },
    }
    try:
        return _MODULE_RUNTIME_SHA256(
            globals(), module_name=__name__, source_path=__file__, contract=contract
        )
    except Exception as error:
        raise GVSProgramPopulationError("could not bind planner runtime") from error


def _assert_runtime_integrity() -> None:
    _assert_import_bindings_unchanged()
    if _planner_source_sha256() != _PINNED_PLANNER_SOURCE_SHA256:
        raise GVSProgramPopulationError("planner source differs from its import-time identity")
    if planner_runtime_sha256() != _PINNED_PLANNER_RUNTIME_SHA256:
        raise GVSProgramPopulationError("planner runtime differs from its import-time identity")


def _build_record(
    skeleton: _Skeleton,
) -> ProgramPopulationRecord:
    program = _PARSE_SIM_PROGRAM(_program_payload(skeleton))
    receipt = _VERIFY_PROGRAM_ROUND_TRIP(program)
    if receipt.passed is not True:
        raise GVSProgramPopulationError(f"program {program.case_id} failed round-trip verification")
    packet = _authoring_packet(skeleton, program)
    packet_sha256 = _sha256_json(_mutable_json(packet), limit=MAX_SPEC_JSON_BYTES)
    program_sha256 = program.sha256()
    world_state_sha256 = program.initial_state.sha256()
    round_trip_receipt_sha256 = receipt.sha256()
    body = _record_body_dict(
        skeleton=skeleton,
        program=program,
        authoring_packet=packet,
        program_sha256=program_sha256,
        world_state_sha256=world_state_sha256,
        round_trip_receipt_sha256=round_trip_receipt_sha256,
        authoring_packet_sha256=packet_sha256,
    )
    return ProgramPopulationRecord(
        skeleton=skeleton,
        program=program,
        authoring_packet=packet,
        program_sha256=program_sha256,
        world_state_sha256=world_state_sha256,
        round_trip_receipt_sha256=round_trip_receipt_sha256,
        authoring_packet_sha256=packet_sha256,
        record_sha256=_sha256_json(body),
        _factory_token=_RECORD_FACTORY_TOKEN,
    )


def build_program_population_plan(spec: ProgramPopulationSpec) -> ProgramPopulationPlan:
    """Freeze membership, then create and verify authoring inputs for exactly T-new/D-support."""

    _assert_runtime_integrity()
    if type(spec) is not ProgramPopulationSpec:
        raise GVSProgramPopulationError("spec must be an exact ProgramPopulationSpec")
    # Detach one canonical snapshot and use only the reconstructed value.  Frozen dataclasses can
    # still be mutated through ``object.__setattr__`` by hostile caller code; repeatedly reading
    # the caller-owned object would let one build mix memberships, floors, and commitments.
    try:
        spec_bytes = _canonical_bytes(spec.to_dict(), limit=MAX_SPEC_JSON_BYTES)
        spec = _spec_from_object(
            _loads_bounded_object(spec_bytes.decode("utf-8"), limit=MAX_SPEC_JSON_BYTES)
        )
    except (AttributeError, RuntimeError, UnicodeError) as error:
        raise GVSProgramPopulationError(
            "spec could not be detached into one exact snapshot"
        ) from error
    # The complete skeleton/lineage roster is built and hashed before any authoring packet exists.
    skeletons = tuple(
        skeleton
        for role_spec in spec.roles
        for skeleton in _assign_skeletons(spec.source_commitment_sha256, role_spec)
    )
    _assert_disjoint_allocation_ids(skeletons)
    memberships = {
        role: [item.record_id for item in skeletons if item.role == role] for role in TRAINING_ROLES
    }
    rosters = {
        role: [item.roster_dict() for item in skeletons if item.role == role]
        for role in TRAINING_ROLES
    }
    membership_sha256 = _sha256_json(memberships)
    roster_sha256 = _sha256_json(rosters)
    records = tuple(_build_record(skeleton) for skeleton in skeletons)
    counts = {role: {stratum: 0 for stratum in REQUIRED_STRATA} for role in TRAINING_ROLES}
    floors_by_role = {role_spec.role: _floor_map(role_spec.floors) for role_spec in spec.roles}
    for record in records:
        for stratum in record.skeleton.strata:
            counts[record.skeleton.role][stratum] += 1
    for role in TRAINING_ROLES:
        for stratum, minimum in floors_by_role[role].items():
            if counts[role][stratum] < minimum:
                raise GVSProgramPopulationError(
                    f"constructed {role} roster misses floor {stratum!r}: "
                    f"{counts[role][stratum]} < {minimum}"
                )
    record_set_sha256 = _sha256_json(
        [
            {"record_id": record.skeleton.record_id, "record_sha256": record.record_sha256}
            for record in records
        ]
    )
    source_hash = _planner_source_sha256()
    runtime_hash = planner_runtime_sha256()
    simulator_program_runtime_sha256 = _SIM_PROGRAM_RUNTIME_SHA256()
    simulator_action_runtime_sha256 = _ACTION_SIMULATOR_RUNTIME_SHA256()
    body = _plan_body_dict(
        spec=spec,
        records=records,
        planner_source_sha256=source_hash,
        planner_runtime_sha256=runtime_hash,
        simulator_program_runtime_sha256=simulator_program_runtime_sha256,
        simulator_action_runtime_sha256=simulator_action_runtime_sha256,
        membership_sha256=membership_sha256,
        roster_sha256=roster_sha256,
        record_set_sha256=record_set_sha256,
    )
    plan = ProgramPopulationPlan(
        spec=spec,
        records=records,
        planner_source_sha256=source_hash,
        planner_runtime_sha256=runtime_hash,
        simulator_program_runtime_sha256=simulator_program_runtime_sha256,
        simulator_action_runtime_sha256=simulator_action_runtime_sha256,
        membership_sha256=membership_sha256,
        roster_sha256=roster_sha256,
        record_set_sha256=record_set_sha256,
        artifact_body_sha256=_sha256_json(body),
        _factory_token=_PLAN_FACTORY_TOKEN,
    )
    _assert_runtime_integrity()
    return plan


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise GVSProgramPopulationError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_float(value: str) -> NoReturn:
    raise GVSProgramPopulationError(f"floating-point value {value!r} is forbidden")


def _reject_constant(value: str) -> NoReturn:
    raise GVSProgramPopulationError(f"non-finite value {value!r} is forbidden")


def _parse_integer(value: str) -> int:
    digits = value.removeprefix("-")
    if len(digits) > 19:
        raise GVSProgramPopulationError("JSON integer exceeds the signed-64-bit bound")
    parsed = int(value)
    if abs(parsed) > MAX_INTEGER_ABS:
        raise GVSProgramPopulationError("JSON integer exceeds the signed-64-bit bound")
    return parsed


def _validate_loaded_json(
    value: object, *, path: str = "$", depth: int = 0, budget: list[int]
) -> None:
    if depth > MAX_JSON_DEPTH:
        raise GVSProgramPopulationError(f"{path} exceeds maximum JSON depth")
    budget[0] += 1
    if budget[0] > MAX_JSON_NODES:
        raise GVSProgramPopulationError("JSON node count exceeds the planner bound")
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is str:
        _require_exact_string(value, path=path, nonempty=False)
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_loaded_json(child, path=f"{path}[{index}]", depth=depth + 1, budget=budget)
        return
    if type(value) is dict:
        for key, child in value.items():
            _require_exact_string(key, path=f"{path}.<key>", nonempty=True)
            _validate_loaded_json(child, path=f"{path}.{key}", depth=depth + 1, budget=budget)
        return
    raise GVSProgramPopulationError(f"{path} contains non-JSON type {type(value).__name__}")


def _loads_bounded_object(text: str, *, limit: int) -> dict[str, object]:
    if type(text) is not str:
        raise GVSProgramPopulationError("JSON input must be an exact string")
    try:
        encoded_size = len(text.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise GVSProgramPopulationError("JSON input must be valid UTF-8") from error
    if encoded_size > limit:
        raise GVSProgramPopulationError(f"JSON input exceeds {limit} UTF-8 bytes")
    try:
        value = _JSON_LOADS(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_int=_parse_integer,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except GVSProgramPopulationError:
        raise
    except (_JSON_DECODE_ERROR, UnicodeError, RecursionError, ValueError) as error:
        raise GVSProgramPopulationError("input is not one strict JSON value") from error
    if type(value) is not dict:
        raise GVSProgramPopulationError("top-level JSON value must be an object")
    _validate_loaded_json(value, budget=[0])
    return value


def _exact_object(value: object, *, fields: frozenset[str], path: str) -> dict[str, object]:
    if type(value) is not dict:
        raise GVSProgramPopulationError(f"{path} must be an exact object")
    keys = set(value)
    if keys != fields:
        raise GVSProgramPopulationError(
            f"{path} fields are not exact; missing={sorted(fields - keys)!r}, "
            f"extra={sorted(keys - fields)!r}"
        )
    return value


def _spec_from_object(value: object) -> ProgramPopulationSpec:
    obj = _exact_object(value, fields=_SPEC_FIELDS, path="$.spec")
    if obj["schema_version"] != PROGRAM_POPULATION_SPEC_VERSION:
        raise GVSProgramPopulationError("unsupported population spec schema")
    raw_roles = obj["roles"]
    if type(raw_roles) is not list or len(raw_roles) != len(TRAINING_ROLES):
        raise GVSProgramPopulationError("spec roles must contain exactly T-new and D-support")
    roles: list[RolePlanSpec] = []
    for role_index, raw_role in enumerate(raw_roles):
        role_obj = _exact_object(
            raw_role, fields=_ROLE_SPEC_FIELDS, path=f"$.spec.roles[{role_index}]"
        )
        raw_floors = role_obj["floors"]
        if type(raw_floors) is not list:
            raise GVSProgramPopulationError("spec floors must be an array")
        floors: list[StratumFloor] = []
        for floor_index, raw_floor in enumerate(raw_floors):
            floor_obj = _exact_object(
                raw_floor,
                fields=_FLOOR_FIELDS,
                path=f"$.spec.roles[{role_index}].floors[{floor_index}]",
            )
            floors.append(StratumFloor(stratum=floor_obj["stratum"], minimum=floor_obj["minimum"]))
        roles.append(
            RolePlanSpec(
                role=role_obj["role"],
                count=role_obj["count"],
                floors=tuple(floors),
            )
        )
    return make_program_population_spec(
        source_commitment_sha256=obj["source_commitment_sha256"], roles=tuple(roles)
    )


def loads_program_population_spec(text: str) -> ProgramPopulationSpec:
    """Load an exact bounded immutable spec; unknown hidden inputs fail closed."""

    _assert_runtime_integrity()
    return _spec_from_object(_loads_bounded_object(text, limit=MAX_SPEC_JSON_BYTES))


def loads_program_population_plan(text: str) -> ProgramPopulationPlan:
    """Load and live-rederive a bounded plan, including every simulator round trip."""

    _assert_runtime_integrity()
    obj = _loads_bounded_object(text, limit=MAX_PLAN_JSON_BYTES)
    _exact_object(obj, fields=_PLAN_FIELDS, path="$")
    spec = _spec_from_object(obj["spec"])
    rebuilt = build_program_population_plan(spec)
    if _canonical_bytes(obj) != rebuilt.canonical_json().encode("utf-8"):
        raise GVSProgramPopulationError(
            "serialized plan differs from complete live deterministic reconstruction"
        )
    return rebuilt


__all__ = [
    "AUTHORING_PACKET_VERSION",
    "EXPECTED_OUTCOMES",
    "FEATURE_STRATA",
    "HUMAN_ROLES_EXCLUDED",
    "MAX_PLAN_JSON_BYTES",
    "MAX_RECORDS_PER_ROLE",
    "MAX_SPEC_JSON_BYTES",
    "MAX_TOTAL_RECORDS",
    "MIN_RECORDS_PER_ROLE",
    "OPERATION_STRATA",
    "OUTCOME_STRATA",
    "PROGRAM_POPULATION_PLAN_VERSION",
    "PROGRAM_POPULATION_RECORD_VERSION",
    "PROGRAM_POPULATION_SPEC_VERSION",
    "REQUIRED_STRATA",
    "SEMANTIC_OPERATIONS",
    "TRAINING_ROLES",
    "GVSProgramPopulationError",
    "ProgramPopulationPlan",
    "ProgramPopulationRecord",
    "ProgramPopulationSpec",
    "RolePlanSpec",
    "StratumFloor",
    "build_program_population_plan",
    "loads_program_population_plan",
    "loads_program_population_spec",
    "make_program_population_spec",
    "planner_runtime_sha256",
]


_PINNED_PLANNER_SOURCE_SHA256 = _planner_source_sha256()
_PINNED_PLANNER_RUNTIME_SHA256 = planner_runtime_sha256()
