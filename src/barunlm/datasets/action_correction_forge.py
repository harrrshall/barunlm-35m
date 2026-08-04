"""Deterministic, CPU-only source forge for Generate-Correct SFT fixtures.

This is a deliberately bounded prototype, not a population materializer.  It builds
small program-first fixtures from the sandbox simulator's public semantic contract and
never reads an existing prompt/label dataset.  The returned lineage is descriptive:
family identifiers are shared by many rows and never claim row-level independence.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Literal

from barunlm.evaluation.action_ir import (
    ActionIR,
    ActionIRError,
    ActionIRParseError,
    ActionIRValidationError,
    decode_json_object,
    parse_action_ir,
    validate_action_ir,
)
from barunlm.evaluation.action_simulator import (
    SIMULATOR_SCHEMA_BYTES,
    ActionSimulator,
    compile_program_step,
    simulate_reference_action,
    simulator_tool_registry_snapshot,
    verify_program_round_trip,
)
from barunlm.evaluation.sim_program import (
    SEMANTIC_OPERATION_SPECS,
    SIM_PROGRAM_VERSION,
    WORLD_STATE_VERSION,
    SimProgram,
    WorldState,
    parse_sim_program,
)

FORGE_VERSION = "barun-action-correction-forge-prototype-v1"
RUN_ID = "20260805-0230-action-correction-forge-screen-s17"
ACTION_IR_CONTRACT = "ACTION_IR_V1"
MASK_TOKEN = "<MASK>"
C_SPLIT_RULE = "sha256(barun-action-correction-c-split-v1\\0source_id)-low-bit"
FINGERPRINT_VERSION = "barun-action-correction-fingerprints-v1"
PROTOTYPE_MAX_FIXTURE_ROWS_PER_STRATUM = 2

Role = Literal["T-synth", "D-internal"]
Arm = Literal["A", "B", "C"]

SEMANTIC_OPERATION_KINDS = (
    "ADD_LIST_ITEM",
    "CREATE_CALENDAR_EVENT",
    "CREATE_NOTE",
    "CREATE_REMINDER",
    "LOOK_UP_CONTACT",
    "LOOK_UP_ROUTE",
    "PAUSE_MEDIA",
    "PLAY_MEDIA",
    "PROPOSE_MESSAGE",
    "RESCHEDULE_CALENDAR_EVENT",
    "SET_BOOLEAN_SETTING",
    "SET_LIST_ITEM_CHECKED",
    "UPDATE_REMINDER",
)
CONTROL_DECISION_STRATA = ("ABSTAIN", "CLARIFY", "CONFIRM")
DESCRIPTIVE_STRATA = SEMANTIC_OPERATION_KINDS + CONTROL_DECISION_STRATA

PROTOTYPE_FLAGS = MappingProxyType(
    {
        "population_materialized": False,
        "model_access_authorized": False,
        "model_cuda_authorized": False,
        "cuda_authorized": False,
        "jarvislabs_resource_creation_authorized": False,
        "human_label_access_authorized": False,
        "launch_authorized": False,
    }
)


class ActionCorrectionForgeError(ValueError):
    """The bounded source or view contract failed closed."""


@dataclass(frozen=True, slots=True)
class RoleFamilies:
    """Finite role-scoped families chosen before any row is rendered."""

    role: Role
    renderer_family: str
    entity_family: str
    temporal_family: str
    schema_family: str
    context_family: str
    source_family: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "renderer_family": self.renderer_family,
            "entity_family": self.entity_family,
            "temporal_family": self.temporal_family,
            "schema_family": self.schema_family,
            "context_family": self.context_family,
            "source_family": self.source_family,
        }


ROLE_FAMILIES = MappingProxyType(
    {
        "T-synth": RoleFamilies(
            role="T-synth",
            renderer_family="acf.renderer.train.imperative.v1",
            entity_family="acf.entities.train.v1",
            temporal_family="acf.temporal.train.kolkata.v1",
            schema_family="acf.schema.train.tool-args.v1",
            context_family="acf.context.train.state.v1",
            source_family="acf.source.train.batch-a.v1",
        ),
        "D-internal": RoleFamilies(
            role="D-internal",
            renderer_family="acf.renderer.screen.declarative.v1",
            entity_family="acf.entities.screen.v1",
            temporal_family="acf.temporal.screen.new-york.v1",
            schema_family="acf.schema.screen.function-parameters.v1",
            context_family="acf.context.screen.available-context.v1",
            source_family="acf.source.screen.batch-a.v1",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class DeclaredLineage:
    role: Role
    renderer_family: str
    entity_family: str
    temporal_family: str
    schema_family: str
    context_family: str
    source_family: str
    program_skeleton_family: str
    family_assignments_fixed_before_render: bool = True
    row_is_independent_component: bool = False
    effective_component_claimed: bool = False
    power_claimed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "renderer_family": self.renderer_family,
            "entity_family": self.entity_family,
            "temporal_family": self.temporal_family,
            "schema_family": self.schema_family,
            "context_family": self.context_family,
            "source_family": self.source_family,
            "program_skeleton_family": self.program_skeleton_family,
            "family_assignments_fixed_before_render": (self.family_assignments_fixed_before_render),
            "row_is_independent_component": self.row_is_independent_component,
            "effective_component_claimed": self.effective_component_claimed,
            "power_claimed": self.power_claimed,
        }


@dataclass(frozen=True, slots=True)
class Fingerprints:
    exact_sha256: str
    normalized_sha256: str
    delexicalized_sha256: str
    gold_action_ir_sha256: str
    program_sha256: str
    world_state_sha256: str
    version: str = FINGERPRINT_VERSION

    def to_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "exact_sha256": self.exact_sha256,
            "normalized_sha256": self.normalized_sha256,
            "delexicalized_sha256": self.delexicalized_sha256,
            "gold_action_ir_sha256": self.gold_action_ir_sha256,
            "program_sha256": self.program_sha256,
            "world_state_sha256": self.world_state_sha256,
        }


@dataclass(frozen=True, slots=True)
class DraftDiagnostics:
    """Draft-only evidence; the proposal predicate is deliberately not model-visible."""

    parse_valid: bool
    schema_valid: bool
    policy_conformant_proposal: bool
    policy_status: str
    simulator_status: str
    error_codes: tuple[str, ...]

    def model_visible_dict(self) -> dict[str, object]:
        return {
            "parse_valid": self.parse_valid,
            "schema_valid": self.schema_valid,
            "policy_status": self.policy_status,
            "simulator_status": self.simulator_status,
            "error_codes": list(self.error_codes),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.model_visible_dict())


@dataclass(frozen=True, slots=True)
class FaultCertificate:
    """Audit-only proof for one constructed draft; never included in model input."""

    source_id: str
    mutation_path: str
    mutation_count: int
    gold_action_ir_sha256: str
    draft_sha256: str
    draft_parse_valid: bool
    draft_schema_valid: bool
    policy_conformant_proposal: bool
    semantic_outcome_distinct: bool
    model_visible: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "mutation_path": self.mutation_path,
            "mutation_count": self.mutation_count,
            "gold_action_ir_sha256": self.gold_action_ir_sha256,
            "draft_sha256": self.draft_sha256,
            "draft_parse_valid": self.draft_parse_valid,
            "draft_schema_valid": self.draft_schema_valid,
            "policy_conformant_proposal": self.policy_conformant_proposal,
            "semantic_outcome_distinct": self.semantic_outcome_distinct,
            "model_visible": self.model_visible,
        }


@dataclass(frozen=True, slots=True)
class ForgeSource:
    source_id: str
    role: Role
    stratum: str
    ordinal: int
    lineage: DeclaredLineage
    request: str
    direct_prompt_json: str
    program: SimProgram
    gold_action_ir: ActionIR
    fingerprints: Fingerprints
    reference_status: str
    expected_state_sha256: str
    round_trip_sha256: str

    @property
    def gold_target_json(self) -> str:
        return self.gold_action_ir.canonical_json()

    def audit_record(self) -> dict[str, object]:
        return {
            "schema_version": FORGE_VERSION,
            "source_id": self.source_id,
            "role": self.role,
            "stratum": self.stratum,
            "ordinal": self.ordinal,
            "lineage": self.lineage.to_dict(),
            "request": self.request,
            "direct_prompt_json": self.direct_prompt_json,
            "program_json": self.program.canonical_json(),
            "gold_action_ir_json": self.gold_target_json,
            "fingerprints": self.fingerprints.to_dict(),
            "reference_status": self.reference_status,
            "expected_state_sha256": self.expected_state_sha256,
            "round_trip_sha256": self.round_trip_sha256,
            "prototype_flags": dict(PROTOTYPE_FLAGS),
        }


@dataclass(frozen=True, slots=True)
class TrainingView:
    source_id: str
    role: Role
    arm: Arm
    view_kind: str
    input_json: str
    target_json: str
    masked_path: str | None = None
    c_assignment: str | None = None
    fault_certificate: FaultCertificate | None = None
    diagnostics: DraftDiagnostics | None = None

    def model_record(self) -> dict[str, str]:
        return {"source_id": self.source_id, "input": self.input_json, "target": self.target_json}

    def audit_record(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "role": self.role,
            "arm": self.arm,
            "view_kind": self.view_kind,
            "input_json": self.input_json,
            "target_json": self.target_json,
            "masked_path": self.masked_path,
            "c_assignment": self.c_assignment,
            "fault_certificate": (
                self.fault_certificate.to_dict() if self.fault_certificate is not None else None
            ),
            "diagnostics": (
                self.diagnostics.model_visible_dict() if self.diagnostics is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class _ValuePool:
    prefix: str
    reference_time: str
    timezone: str
    due_at: str
    event_start: str
    event_end: str
    reschedule_start: str
    reschedule_alt_start: str
    reschedule_end: str
    reminder_existing: str
    reminder_new: str
    event_existing: str
    event_new: str
    contact_primary: str
    contact_secondary: str
    note_new: str
    list_id: str
    item_existing: str
    item_new: str
    place_origin: str
    place_destination: str
    message_new: str
    track_primary: str
    track_secondary: str
    setting_bool: str

    def replacements(self) -> tuple[tuple[str, str], ...]:
        values: list[tuple[str, str]] = [
            (self.reference_time, "<REFERENCE_TIME>"),
            (self.due_at, "<TIMESTAMP>"),
            (self.event_start, "<TIMESTAMP>"),
            (self.event_end, "<TIMESTAMP>"),
            (self.reschedule_start, "<TIMESTAMP>"),
            (self.reschedule_alt_start, "<TIMESTAMP>"),
            (self.reschedule_end, "<TIMESTAMP>"),
            (self.reminder_existing, "<REMINDER_ID>"),
            (self.reminder_new, "<REMINDER_ID>"),
            (self.event_existing, "<EVENT_ID>"),
            (self.event_new, "<EVENT_ID>"),
            (self.contact_primary, "<CONTACT_ID>"),
            (self.contact_secondary, "<CONTACT_ID>"),
            (self.note_new, "<NOTE_ID>"),
            (self.list_id, "<LIST_ID>"),
            (self.item_existing, "<ITEM_ID>"),
            (self.item_new, "<ITEM_ID>"),
            (self.place_origin, "<PLACE_ID>"),
            (self.place_destination, "<PLACE_ID>"),
            (self.message_new, "<MESSAGE_ID>"),
            (self.track_primary, "<TRACK_ID>"),
            (self.track_secondary, "<TRACK_ID>"),
            (self.setting_bool, "<SETTING_ID>"),
            ("Asha Train", "<PERSON_NAME>"),
            ("Mira Screen", "<PERSON_NAME>"),
            ("Dev Train", "<PERSON_NAME>"),
            ("Noah Screen", "<PERSON_NAME>"),
        ]
        return tuple(sorted(set(values), key=lambda item: (-len(item[0]), item)))


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ActionCorrectionForgeError("value is not strict canonical JSON") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_frozen_simulator_vocabulary() -> None:
    observed = tuple(sorted(SEMANTIC_OPERATION_SPECS))
    if observed != SEMANTIC_OPERATION_KINDS:
        raise ActionCorrectionForgeError(
            "simulator semantic operation vocabulary changed from the frozen 13-kind contract"
        )
    if len(SIMULATOR_SCHEMA_BYTES.encode("utf-8")) == 0:
        raise ActionCorrectionForgeError("simulator schema snapshot is empty")


def _families(role: str) -> RoleFamilies:
    try:
        return ROLE_FAMILIES[role]
    except KeyError as exc:
        raise ActionCorrectionForgeError(f"unknown forge role {role!r}") from exc


def _value_pool(role: Role, ordinal: int) -> _ValuePool:
    if type(ordinal) is not int or not 0 <= ordinal < PROTOTYPE_MAX_FIXTURE_ROWS_PER_STRATUM:
        raise ActionCorrectionForgeError("prototype ordinal exceeds the bounded fixture-only quota")
    if role == "T-synth":
        prefix = f"train.{ordinal:02d}"
        return _ValuePool(
            prefix=prefix,
            reference_time="2026-09-14T09:00:00+05:30",
            timezone="Asia/Kolkata",
            due_at="2026-09-15T17:00:00+05:30",
            event_start="2026-09-14T13:00:00+05:30",
            event_end="2026-09-14T14:00:00+05:30",
            reschedule_start="2026-09-14T11:00:00+05:30",
            reschedule_alt_start="2026-09-14T11:15:00+05:30",
            reschedule_end="2026-09-14T11:45:00+05:30",
            reminder_existing=f"reminder.{prefix}.existing",
            reminder_new=f"reminder.{prefix}.new",
            event_existing=f"event.{prefix}.existing",
            event_new=f"event.{prefix}.new",
            contact_primary=f"contact.{prefix}.primary",
            contact_secondary=f"contact.{prefix}.secondary",
            note_new=f"note.{prefix}.new",
            list_id=f"list.{prefix}.main",
            item_existing=f"item.{prefix}.existing",
            item_new=f"item.{prefix}.new",
            place_origin=f"place.{prefix}.origin",
            place_destination=f"place.{prefix}.destination",
            message_new=f"message.{prefix}.new",
            track_primary=f"track.{prefix}.primary",
            track_secondary=f"track.{prefix}.secondary",
            setting_bool=f"setting.{prefix}.enabled",
        )
    prefix = f"screen.{ordinal:02d}"
    return _ValuePool(
        prefix=prefix,
        reference_time="2026-11-08T08:00:00-05:00",
        timezone="America/New_York",
        due_at="2026-11-09T17:00:00-05:00",
        event_start="2026-11-08T13:00:00-05:00",
        event_end="2026-11-08T14:00:00-05:00",
        reschedule_start="2026-11-08T11:00:00-05:00",
        reschedule_alt_start="2026-11-08T11:15:00-05:00",
        reschedule_end="2026-11-08T11:45:00-05:00",
        reminder_existing=f"reminder.{prefix}.existing",
        reminder_new=f"reminder.{prefix}.new",
        event_existing=f"event.{prefix}.existing",
        event_new=f"event.{prefix}.new",
        contact_primary=f"contact.{prefix}.primary",
        contact_secondary=f"contact.{prefix}.secondary",
        note_new=f"note.{prefix}.new",
        list_id=f"list.{prefix}.main",
        item_existing=f"item.{prefix}.existing",
        item_new=f"item.{prefix}.new",
        place_origin=f"place.{prefix}.origin",
        place_destination=f"place.{prefix}.destination",
        message_new=f"message.{prefix}.new",
        track_primary=f"track.{prefix}.primary",
        track_secondary=f"track.{prefix}.secondary",
        setting_bool=f"setting.{prefix}.enabled",
    )


def _world_payload(role: Role, pool: _ValuePool) -> dict[str, object]:
    primary_name = "Asha Train" if role == "T-synth" else "Mira Screen"
    secondary_name = "Dev Train" if role == "T-synth" else "Noah Screen"
    return {
        "schema_version": WORLD_STATE_VERSION,
        "reference_time": pool.reference_time,
        "timezone": pool.timezone,
        "reminders": {
            pool.reminder_existing: {
                "title": "Submit the report",
                "due_at": pool.due_at,
                "completed": False,
            }
        },
        "calendar": {
            pool.event_existing: {
                "title": "Stand-up",
                "start_at": pool.event_start,
                "end_at": pool.event_end,
            }
        },
        "contacts": {
            pool.contact_primary: {"name": primary_name, "channel": "sms"},
            pool.contact_secondary: {"name": secondary_name, "channel": "sms"},
        },
        "notes": {},
        "lists": {
            pool.list_id: {
                "title": "Groceries",
                "items": {
                    pool.item_existing: {"text": "Milk", "checked": False},
                },
            }
        },
        "places": {
            pool.place_origin: {"name": "Home"},
            pool.place_destination: {"name": "Office"},
        },
        "routes": {
            f"route.{pool.prefix}.driving": {
                "origin_id": pool.place_origin,
                "destination_id": pool.place_destination,
                "mode": "driving",
                "distance_m": 8200,
                "duration_s": 1500,
            },
            f"route.{pool.prefix}.walking": {
                "origin_id": pool.place_origin,
                "destination_id": pool.place_destination,
                "mode": "walking",
                "distance_m": 7900,
                "duration_s": 6300,
            },
        },
        "outbox": [],
        "media": {
            "status": "playing",
            "track_id": pool.track_primary,
            "catalog": {
                pool.track_primary: {"title": "Morning Raga"},
                pool.track_secondary: {"title": "Blue Hour"},
            },
        },
        "settings": {pool.setting_bool: False},
    }


def _operation_parameters(kind: str, pool: _ValuePool) -> dict[str, object]:
    values: dict[str, dict[str, object]] = {
        "ADD_LIST_ITEM": {
            "list_ref": pool.list_id,
            "item_ref": pool.item_new,
            "item_text": "Rice",
        },
        "CREATE_CALENDAR_EVENT": {
            "event_ref": pool.event_new,
            "summary": "Lunch",
            "begins_at": pool.event_start,
            "ends_at": pool.event_end,
        },
        "CREATE_NOTE": {
            "note_ref": pool.note_new,
            "heading": "Trip",
            "content": "Pack light",
        },
        "CREATE_REMINDER": {
            "reminder_ref": pool.reminder_new,
            "summary": "Call the clinic",
            "due_at": pool.due_at,
        },
        "LOOK_UP_CONTACT": {"contact_ref": pool.contact_primary},
        "LOOK_UP_ROUTE": {
            "from_place_ref": pool.place_origin,
            "to_place_ref": pool.place_destination,
            "travel_mode": "driving",
        },
        "PAUSE_MEDIA": {},
        "PLAY_MEDIA": {"track_ref": pool.track_secondary},
        "PROPOSE_MESSAGE": {
            "message_ref": pool.message_new,
            "recipient_ref": pool.contact_primary,
            "delivery_channel": "sms",
            "content": "On my way",
        },
        "RESCHEDULE_CALENDAR_EVENT": {
            "event_ref": pool.event_existing,
            "begins_at": pool.reschedule_start,
            "ends_at": pool.reschedule_end,
        },
        "SET_BOOLEAN_SETTING": {"setting_ref": pool.setting_bool, "value": True},
        "SET_LIST_ITEM_CHECKED": {
            "list_ref": pool.list_id,
            "item_ref": pool.item_existing,
            "is_checked": True,
        },
        "UPDATE_REMINDER": {
            "reminder_ref": pool.reminder_existing,
            "summary": "Submit the final report",
            "due_at": pool.due_at,
        },
    }
    try:
        return values[kind]
    except KeyError as exc:
        raise ActionCorrectionForgeError(f"unknown semantic operation stratum {kind!r}") from exc


def _request_for(stratum: str, role: Role, pool: _ValuePool) -> str:
    primary_name = "Asha Train" if role == "T-synth" else "Mira Screen"
    if stratum == "ABSTAIN":
        request = "Transfer money from my bank account; no banking tool is supplied."
    elif stratum == "CLARIFY":
        request = "Send the weekly update, but I did not say who should receive it."
    elif stratum == "CONFIRM":
        request = (
            f"Prepare SMS {pool.message_new} to {primary_name} saying 'Please call me'; "
            "ask for confirmation before the proposed send."
        )
    elif stratum == "ADD_LIST_ITEM":
        request = f"Add Rice as {pool.item_new} to list {pool.list_id}."
    elif stratum == "CREATE_CALENDAR_EVENT":
        request = (
            f"Create event {pool.event_new} titled Lunch from {pool.event_start} "
            f"to {pool.event_end}."
        )
    elif stratum == "CREATE_NOTE":
        request = f"Create note {pool.note_new} titled Trip with body 'Pack light'."
    elif stratum == "CREATE_REMINDER":
        request = f"Create reminder {pool.reminder_new} titled 'Call the clinic' due {pool.due_at}."
    elif stratum == "LOOK_UP_CONTACT":
        request = f"Look up the contact record for {primary_name}."
    elif stratum == "LOOK_UP_ROUTE":
        request = f"Find the driving route from {pool.place_origin} to {pool.place_destination}."
    elif stratum == "PAUSE_MEDIA":
        request = "Pause the track that is currently playing."
    elif stratum == "PLAY_MEDIA":
        request = f"Play catalog track {pool.track_secondary}."
    elif stratum == "PROPOSE_MESSAGE":
        request = f"Prepare SMS {pool.message_new} to {primary_name} saying 'On my way'."
    elif stratum == "RESCHEDULE_CALENDAR_EVENT":
        request = (
            f"Move event {pool.event_existing} to {pool.reschedule_start} through "
            f"{pool.reschedule_end}."
        )
    elif stratum == "SET_BOOLEAN_SETTING":
        request = f"Turn on setting {pool.setting_bool}."
    elif stratum == "SET_LIST_ITEM_CHECKED":
        request = f"Mark item {pool.item_existing} checked in list {pool.list_id}."
    elif stratum == "UPDATE_REMINDER":
        request = (
            f"Update reminder {pool.reminder_existing} to 'Submit the final report' due "
            f"{pool.due_at}."
        )
    else:
        raise ActionCorrectionForgeError(f"unknown descriptive stratum {stratum!r}")
    if role == "T-synth":
        return request
    return f"Given the supplied state, {request[0].lower()}{request[1:]}"


def _program_payload(
    *, source_id: str, stratum: str, role: Role, pool: _ValuePool
) -> dict[str, object]:
    if stratum == "ABSTAIN":
        step = {
            "step_id": "step.one",
            "decision": "ABSTAIN",
            "mode": None,
            "operations": [],
            "missing": [],
        }
    elif stratum == "CLARIFY":
        step = {
            "step_id": "step.one",
            "decision": "CLARIFY",
            "mode": None,
            "operations": [],
            "missing": ["recipient"],
        }
    else:
        operation_kind = "PROPOSE_MESSAGE" if stratum == "CONFIRM" else stratum
        side_effecting = SEMANTIC_OPERATION_SPECS[operation_kind][1]
        step = {
            "step_id": "step.one",
            "decision": "CONFIRM" if side_effecting else "CALL",
            "mode": "SINGLE",
            "operations": [
                {"kind": operation_kind, "parameters": _operation_parameters(operation_kind, pool)}
            ],
            "missing": [],
        }
    return {
        "schema_version": SIM_PROGRAM_VERSION,
        "case_id": f"case.{source_id}",
        "initial_state": _world_payload(role, pool),
        "steps": [step],
    }


def _schema_presentation(role: Role) -> list[dict[str, object]]:
    records = json.loads(SIMULATOR_SCHEMA_BYTES)
    if not isinstance(records, list) or len(records) != 13:
        raise ActionCorrectionForgeError("simulator schema snapshot is not the frozen 13-tool list")
    rendered: list[dict[str, object]] = []
    for record in records:
        if role == "T-synth":
            rendered.append(
                {
                    "tool": record["name"],
                    "args": record["arguments"],
                    "required": record["required"],
                    "side_effecting": record["side_effecting"],
                }
            )
        else:
            rendered.append(
                {
                    "name": record["name"],
                    "parameters": record["arguments"],
                    "required_parameters": record["required"],
                    "effect": "proposal" if record["side_effecting"] else "read_only",
                }
            )
    return rendered


def _direct_prompt(*, role: Role, request: str, world: WorldState) -> dict[str, object]:
    if role == "T-synth":
        return {
            "action_ir_contract": ACTION_IR_CONTRACT,
            "instruction": "Compile the request into one canonical Action IR proposal.",
            "request": request,
            "context": {"state": world.to_dict()},
            "tools": _schema_presentation(role),
        }
    return {
        "contract": ACTION_IR_CONTRACT,
        "task": "Return only strict canonical JSON for the requested personal-action proposal.",
        "user_request": request,
        "supplied_context": {"available_state": world.to_dict()},
        "tool_schemas": _schema_presentation(role),
    }


def _normalize_value(value: object) -> object:
    if isinstance(value, dict):
        return {key: _normalize_value(child) for key, child in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_value(child) for child in value]
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFC", value).casefold()
        return " ".join(normalized.split())
    return value


def _delexicalize_value(value: object, replacements: tuple[tuple[str, str], ...]) -> object:
    if isinstance(value, dict):
        return {
            _replace_dynamic_text(key, replacements): _delexicalize_value(child, replacements)
            for key, child in sorted(value.items())
        }
    if isinstance(value, list):
        return [_delexicalize_value(child, replacements) for child in value]
    if isinstance(value, str):
        return _replace_dynamic_text(value, replacements)
    return value


def _replace_dynamic_text(value: str, replacements: tuple[tuple[str, str], ...]) -> str:
    output = value
    for concrete, placeholder in replacements:
        output = output.replace(concrete, placeholder)
    return output


def _source_id(role: Role, stratum: str, ordinal: int, lineage: DeclaredLineage) -> str:
    commitment = _canonical_json(
        {
            "version": FORGE_VERSION,
            "role": role,
            "stratum": stratum,
            "ordinal": ordinal,
            "lineage": lineage.to_dict(),
        }
    )
    slug = stratum.lower().replace("_", "-")
    role_slug = "t" if role == "T-synth" else "d"
    return f"acf.{role_slug}.{slug}.{ordinal:02d}.{_sha256_text(commitment)[:12]}"


@lru_cache(maxsize=64)
def build_forge_source(role: Role, stratum: str, ordinal: int = 0) -> ForgeSource:
    """Build one bounded source without reading any existing dataset or model output."""

    _validate_frozen_simulator_vocabulary()
    families = _families(role)
    if stratum not in DESCRIPTIVE_STRATA:
        raise ActionCorrectionForgeError(f"unknown descriptive stratum {stratum!r}")
    pool = _value_pool(role, ordinal)
    lineage = DeclaredLineage(
        role=role,
        renderer_family=families.renderer_family,
        entity_family=families.entity_family,
        temporal_family=families.temporal_family,
        schema_family=families.schema_family,
        context_family=families.context_family,
        source_family=families.source_family,
        program_skeleton_family=(
            f"acf.program.{'train' if role == 'T-synth' else 'screen'}."
            f"{stratum.lower().replace('_', '-')}.v1"
        ),
    )
    source_id = _source_id(role, stratum, ordinal, lineage)
    program = parse_sim_program(
        _program_payload(
            source_id=source_id,
            stratum=stratum,
            role=role,
            pool=pool,
        )
    )
    gold_action = compile_program_step(program.steps[0])
    round_trip = verify_program_round_trip(program)
    if not round_trip.passed:
        raise ActionCorrectionForgeError("program/compiler/reference round trip did not pass")
    reference = simulate_reference_action(gold_action, program.initial_state)
    if reference.receipt.status not in {"reference_applied", "reference_control"}:
        raise ActionCorrectionForgeError("gold program did not have a valid reference outcome")
    request = _request_for(stratum, role, pool)
    prompt = _direct_prompt(
        role=role,
        request=request,
        world=program.initial_state,
    )
    prompt_json = _canonical_json(prompt)
    normalized_json = _canonical_json(_normalize_value(prompt))
    delexicalized_json = _canonical_json(_delexicalize_value(prompt, pool.replacements()))
    gold_json = gold_action.canonical_json()
    fingerprints = Fingerprints(
        exact_sha256=_sha256_text(prompt_json),
        normalized_sha256=_sha256_text(normalized_json),
        delexicalized_sha256=_sha256_text(delexicalized_json),
        gold_action_ir_sha256=_sha256_text(gold_json),
        program_sha256=program.sha256(),
        world_state_sha256=program.initial_state.sha256(),
    )
    return ForgeSource(
        source_id=source_id,
        role=role,
        stratum=stratum,
        ordinal=ordinal,
        lineage=lineage,
        request=request,
        direct_prompt_json=prompt_json,
        program=program,
        gold_action_ir=gold_action,
        fingerprints=fingerprints,
        reference_status=reference.receipt.status,
        expected_state_sha256=reference.counterfactual_state.sha256(),
        round_trip_sha256=round_trip.sha256(),
    )


def build_fixture_population(role: Role, *, rows_per_stratum: int = 1) -> tuple[ForgeSource, ...]:
    """Build at most 32 role-local test fixtures; this cannot materialize T or D."""

    _families(role)
    if type(rows_per_stratum) is not int or not (
        1 <= rows_per_stratum <= PROTOTYPE_MAX_FIXTURE_ROWS_PER_STRATUM
    ):
        raise ActionCorrectionForgeError("rows_per_stratum exceeds the fixture-only ceiling")
    return tuple(
        build_forge_source(role, stratum, ordinal)
        for stratum in DESCRIPTIVE_STRATA
        for ordinal in range(rows_per_stratum)
    )


def derive_draft_diagnostics(raw_draft: str, state: WorldState) -> DraftDiagnostics:
    """Derive the complete diagnostic from draft bytes and supplied state only."""

    if type(raw_draft) is not str:
        raise TypeError("raw_draft must be an exact string")
    if not isinstance(state, WorldState):
        raise TypeError("state must be WorldState")
    try:
        decoded = decode_json_object(raw_draft)
    except ActionIRParseError as exc:
        return DraftDiagnostics(
            parse_valid=False,
            schema_valid=False,
            policy_conformant_proposal=False,
            policy_status="unavailable",
            simulator_status="unavailable",
            error_codes=(exc.code,),
        )
    try:
        action = validate_action_ir(decoded, simulator_tool_registry_snapshot())
    except ActionIRValidationError as exc:
        return DraftDiagnostics(
            parse_valid=True,
            schema_valid=False,
            policy_conformant_proposal=False,
            policy_status="unavailable",
            simulator_status="unavailable",
            error_codes=(exc.code,),
        )
    result = ActionSimulator().simulate(action, state)
    policy_valid = result.receipt.policy["policy_valid"] is True
    if not policy_valid:
        policy_status = "blocked"
    elif result.receipt.status == "confirmation_required":
        policy_status = "requires_confirmation"
    else:
        policy_status = "conformant"
    simulator_status = (
        "accepted"
        if result.receipt.status in {"no_action", "simulated", "confirmation_required"}
        else "rejected"
    )
    errors = () if result.receipt.error_code is None else (result.receipt.error_code,)
    return DraftDiagnostics(
        parse_valid=True,
        schema_valid=True,
        policy_conformant_proposal=policy_valid,
        policy_status=policy_status,
        simulator_status=simulator_status,
        error_codes=tuple(sorted(set(errors))),
    )


def _masked_action(source: ForgeSource) -> tuple[dict[str, object], str]:
    payload = json.loads(source.gold_target_json)
    if source.stratum == "ABSTAIN":
        payload["decision"] = MASK_TOKEN
        return payload, "$.decision"
    if source.stratum == "CLARIFY":
        payload["missing"][0] = MASK_TOKEN
        return payload, "$.missing[0]"
    args = payload["calls"][0]["args"]
    if args:
        field = {
            "calendar_create": "title",
            "calendar_reschedule": "start_at",
            "contact_lookup": "contact_id",
            "list_add_item": "text",
            "list_check_item": "checked",
            "map_route": "mode",
            "media_play": "track_id",
            "message_send": "body",
            "note_create": "body",
            "reminder_create": "title",
            "reminder_update": "title",
            "setting_set_bool": "enabled",
        }[payload["calls"][0]["tool"]]
        args[field] = MASK_TOKEN
        return payload, f"$.calls[0].args.{field}"
    payload["decision"] = MASK_TOKEN
    return payload, "$.decision"


def _leaf_differences(left: object, right: object, path: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        if set(left) != set(right):
            return [path]
        differences: list[str] = []
        for key in sorted(left):
            differences.extend(_leaf_differences(left[key], right[key], f"{path}.{key}"))
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [path]
        differences = []
        for index, (left_child, right_child) in enumerate(zip(left, right, strict=True)):
            differences.extend(_leaf_differences(left_child, right_child, f"{path}[{index}]"))
        return differences
    return [] if left == right else [path]


def _fault_payload(source: ForgeSource) -> tuple[dict[str, object], str]:
    payload = json.loads(source.gold_target_json)
    pool = _value_pool(source.role, source.ordinal)
    if source.stratum == "ABSTAIN":
        payload["decision"] = "CALL"
        return payload, "$.decision"
    if source.stratum == "CLARIFY":
        payload["missing"][0] = "time"
        return payload, "$.missing[0]"
    call = payload["calls"][0]
    tool = call["tool"]
    args = call["args"]
    if tool == "media_pause":
        payload["decision"] = "CALL"
        return payload, "$.decision"
    field_and_value: dict[str, tuple[str, object]] = {
        "calendar_create": ("title", "Dinner"),
        "calendar_reschedule": ("start_at", pool.reschedule_alt_start),
        "contact_lookup": ("contact_id", pool.contact_secondary),
        "list_add_item": ("text", "Pasta"),
        "list_check_item": ("checked", False),
        "map_route": ("mode", "walking"),
        "media_play": ("track_id", pool.track_primary),
        "message_send": ("body", "Running late"),
        "note_create": ("body", "Pack heavy"),
        "reminder_create": ("title", "Call the dentist"),
        "reminder_update": ("title", "Submit the draft report"),
        "setting_set_bool": ("enabled", False),
    }
    try:
        field, value = field_and_value[tool]
    except KeyError as exc:
        raise ActionCorrectionForgeError(f"tool {tool!r} lacks a pinned single-fault rule") from exc
    args[field] = value
    return payload, f"$.calls[0].args.{field}"


def _valid_action_or_none(raw: str) -> ActionIR | None:
    try:
        return parse_action_ir(raw, simulator_tool_registry_snapshot())
    except ActionIRError:
        return None


def _semantic_outcome_signature(action: ActionIR, state: WorldState) -> tuple[object, ...]:
    visible = ActionSimulator().simulate(action, state)
    reference = simulate_reference_action(action, state)
    return (
        action.decision.value,
        visible.receipt.status,
        visible.receipt.observation_sha256,
        visible.receipt.policy_sha256,
        reference.receipt.status,
        reference.receipt.observation_sha256,
        reference.receipt.effects_sha256,
        reference.counterfactual_state.sha256(),
    )


def certified_single_fault(source: ForgeSource) -> tuple[str, FaultCertificate]:
    """Create and certify one audit-visible, model-hidden semantic field mutation."""

    payload, expected_path = _fault_payload(source)
    gold_payload = json.loads(source.gold_target_json)
    differences = _leaf_differences(gold_payload, payload)
    if differences != [expected_path]:
        raise ActionCorrectionForgeError(
            f"single-fault constructor changed {differences!r}, expected only {expected_path!r}"
        )
    draft_json = _canonical_json(payload)
    diagnostics = derive_draft_diagnostics(draft_json, source.program.initial_state)
    draft_action = _valid_action_or_none(draft_json)
    if draft_action is None:
        outcome_distinct = True
    else:
        outcome_distinct = _semantic_outcome_signature(
            draft_action, source.program.initial_state
        ) != _semantic_outcome_signature(source.gold_action_ir, source.program.initial_state)
    if not outcome_distinct:
        raise ActionCorrectionForgeError("constructed fault did not change the semantic outcome")
    certificate = FaultCertificate(
        source_id=source.source_id,
        mutation_path=expected_path,
        mutation_count=1,
        gold_action_ir_sha256=_sha256_text(source.gold_target_json),
        draft_sha256=_sha256_text(draft_json),
        draft_parse_valid=diagnostics.parse_valid,
        draft_schema_valid=diagnostics.schema_valid,
        policy_conformant_proposal=diagnostics.policy_conformant_proposal,
        semantic_outcome_distinct=True,
    )
    return draft_json, certificate


def c_assignment(source_id: str) -> str:
    """Apply the frozen hash rule without promising aggregate 50/50 fixture counts."""

    if type(source_id) is not str or not source_id:
        raise ActionCorrectionForgeError("source_id must be a non-empty string")
    digest = hashlib.sha256(
        b"barun-action-correction-c-split-v1\0" + source_id.encode("utf-8")
    ).digest()
    return "single_fault" if digest[0] & 1 else "exact"


def build_view_a(source: ForgeSource) -> TrainingView:
    return TrainingView(
        source_id=source.source_id,
        role=source.role,
        arm="A",
        view_kind="direct_action_ir_repeat",
        input_json=source.direct_prompt_json,
        target_json=source.gold_target_json,
    )


def build_view_b(source: ForgeSource) -> TrainingView:
    masked, path = _masked_action(source)
    prompt = {
        "mode": "reconstruct_one_masked_semantic_field",
        "original": json.loads(source.direct_prompt_json),
        "masked_action_ir": masked,
    }
    return TrainingView(
        source_id=source.source_id,
        role=source.role,
        arm="B",
        view_kind="single_semantic_field_masked_full_action_reconstruction",
        input_json=_canonical_json(prompt),
        target_json=source.gold_target_json,
        masked_path=path,
    )


def build_view_c(source: ForgeSource) -> TrainingView:
    assignment = c_assignment(source.source_id)
    certificate: FaultCertificate | None = None
    if assignment == "single_fault":
        draft_json, certificate = certified_single_fault(source)
    else:
        draft_json = source.gold_target_json
    diagnostics = derive_draft_diagnostics(draft_json, source.program.initial_state)
    prompt = {
        "mode": "inspect_one_draft_and_return_complete_action_ir",
        "original": json.loads(source.direct_prompt_json),
        "draft": json.loads(draft_json),
        "diagnostics": diagnostics.model_visible_dict(),
    }
    return TrainingView(
        source_id=source.source_id,
        role=source.role,
        arm="C",
        view_kind="exact_or_certified_single_fault_correction",
        input_json=_canonical_json(prompt),
        target_json=source.gold_target_json,
        c_assignment=assignment,
        fault_certificate=certificate,
        diagnostics=diagnostics,
    )


def build_matched_views(source: ForgeSource) -> tuple[TrainingView, TrainingView, TrainingView]:
    """Return A/B/C for exactly one source and one complete shared gold target."""

    views = (build_view_a(source), build_view_b(source), build_view_c(source))
    if (
        len({view.source_id for view in views}) != 1
        or len({view.target_json for view in views}) != 1
    ):
        raise ActionCorrectionForgeError("matched views lost their source or target identity")
    return views


def prototype_record() -> dict[str, object]:
    """Return the nonauthorizing, nonmaterialized prototype boundary."""

    _validate_frozen_simulator_vocabulary()
    return {
        "schema_version": FORGE_VERSION,
        "run_id": RUN_ID,
        "descriptive_strata": list(DESCRIPTIVE_STRATA),
        "semantic_operation_kind_count": len(SEMANTIC_OPERATION_KINDS),
        "control_decision_strata": list(CONTROL_DECISION_STRATA),
        "role_families": {key: value.to_dict() for key, value in ROLE_FAMILIES.items()},
        "c_split_rule": C_SPLIT_RULE,
        "fixture_rows_per_stratum_ceiling": PROTOTYPE_MAX_FIXTURE_ROWS_PER_STRATUM,
        "flags": dict(PROTOTYPE_FLAGS),
        "claims": {
            "row_level_independence": False,
            "effective_component_count": False,
            "population_quota_satisfied": False,
            "model_quality_measured": False,
        },
    }


__all__ = [
    "ACTION_IR_CONTRACT",
    "CONTROL_DECISION_STRATA",
    "C_SPLIT_RULE",
    "DESCRIPTIVE_STRATA",
    "FINGERPRINT_VERSION",
    "FORGE_VERSION",
    "MASK_TOKEN",
    "PROTOTYPE_FLAGS",
    "PROTOTYPE_MAX_FIXTURE_ROWS_PER_STRATUM",
    "ROLE_FAMILIES",
    "RUN_ID",
    "SEMANTIC_OPERATION_KINDS",
    "ActionCorrectionForgeError",
    "DeclaredLineage",
    "DraftDiagnostics",
    "FaultCertificate",
    "Fingerprints",
    "ForgeSource",
    "RoleFamilies",
    "TrainingView",
    "build_fixture_population",
    "build_forge_source",
    "build_matched_views",
    "build_view_a",
    "build_view_b",
    "build_view_c",
    "c_assignment",
    "certified_single_fault",
    "derive_draft_diagnostics",
    "prototype_record",
]
