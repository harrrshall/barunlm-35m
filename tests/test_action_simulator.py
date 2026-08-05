from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

import barunlm.evaluation.action_simulator as simulator_module
from barunlm.evaluation.action_ir import ActionIR, parse_action_ir
from barunlm.evaluation.action_simulator import (
    POLICY_CONTRACT_SHA256,
    SIMULATOR_SCHEMA_SHA256,
    SIMULATOR_TOOL_REGISTRY,
    ActionSimulationError,
    ActionSimulator,
    ProgramRoundTripReceipt,
    ReferenceEffectReceipt,
    TransitionReceipt,
    action_simulator_runtime_sha256,
    compile_program_step,
    loads_program_round_trip_receipt,
    loads_reference_receipt,
    loads_transition_receipt,
    simulate_program,
    simulate_reference_action,
    simulate_reference_program_step,
    simulator_tool_registry_snapshot,
    verify_program_round_trip,
)
from barunlm.evaluation.sim_program import (
    SIM_PROGRAM_VERSION,
    WORLD_STATE_VERSION,
    WorldState,
    parse_sim_program,
)


def world_payload() -> dict[str, object]:
    return {
        "schema_version": WORLD_STATE_VERSION,
        "reference_time": "2026-08-04T09:30:00+05:30",
        "timezone": "Asia/Kolkata",
        "reminders": {
            "reminder.old": {
                "title": "Submit report",
                "due_at": "2026-08-05T17:00:00+05:30",
                "completed": False,
            }
        },
        "calendar": {
            "event.standup": {
                "title": "Stand-up",
                "start_at": "2026-08-04T10:00:00+05:30",
                "end_at": "2026-08-04T10:30:00+05:30",
            }
        },
        "contacts": {
            "contact.asha": {"name": "Asha", "channel": "sms"},
            "contact.dev": {"name": "Dev", "channel": "sms"},
        },
        "notes": {},
        "lists": {
            "list.groceries": {
                "title": "Groceries",
                "items": {
                    "item.milk": {"text": "Milk", "checked": False},
                    "item.tea": {"text": "Tea", "checked": False},
                },
            }
        },
        "places": {"place.home": {"name": "Home"}, "place.office": {"name": "Office"}},
        "routes": {
            "route.commute": {
                "origin_id": "place.home",
                "destination_id": "place.office",
                "mode": "driving",
                "distance_m": 8200,
                "duration_s": 1500,
            }
        },
        "outbox": [],
        "media": {
            "status": "stopped",
            "track_id": None,
            "catalog": {
                "track.raga": {"title": "Morning Raga"},
                "track.jazz": {"title": "Blue Hour"},
            },
        },
        "settings": {"wifi": True, "airplane_mode": False, "brightness": 50},
    }


def action(
    decision: str,
    calls: list[dict[str, object]] | None = None,
    *,
    mode: str = "SINGLE",
    missing: list[str] | None = None,
):
    payload: dict[str, object] = {"decision": decision}
    if decision in {"CALL", "CONFIRM"}:
        payload.update(calls=calls, mode=mode)
    elif decision == "CLARIFY":
        payload["missing"] = missing or ["recipient"]
    return parse_action_ir(json.dumps(payload), SIMULATOR_TOOL_REGISTRY)


CASES = [
    (
        "CREATE_REMINDER",
        {"reminder_ref": "reminder.new", "summary": "Call", "due_at": "2026-08-06T12:00:00+05:30"},
        "reminder_create",
        {"reminder_id": "reminder.new", "title": "Call", "due_at": "2026-08-06T12:00:00+05:30"},
        ("reminders", "reminder.new"),
        False,
    ),
    (
        "UPDATE_REMINDER",
        {
            "reminder_ref": "reminder.old",
            "summary": "Final report",
            "due_at": "2026-08-06T17:00:00+05:30",
        },
        "reminder_update",
        {
            "reminder_id": "reminder.old",
            "title": "Final report",
            "due_at": "2026-08-06T17:00:00+05:30",
        },
        ("reminders", "reminder.old"),
        False,
    ),
    (
        "CREATE_CALENDAR_EVENT",
        {
            "event_ref": "event.lunch",
            "summary": "Lunch",
            "begins_at": "2026-08-04T13:00:00+05:30",
            "ends_at": "2026-08-04T14:00:00+05:30",
        },
        "calendar_create",
        {
            "event_id": "event.lunch",
            "title": "Lunch",
            "start_at": "2026-08-04T13:00:00+05:30",
            "end_at": "2026-08-04T14:00:00+05:30",
        },
        ("calendar", "event.lunch"),
        False,
    ),
    (
        "RESCHEDULE_CALENDAR_EVENT",
        {
            "event_ref": "event.standup",
            "begins_at": "2026-08-04T11:00:00+05:30",
            "ends_at": "2026-08-04T11:30:00+05:30",
        },
        "calendar_reschedule",
        {
            "event_id": "event.standup",
            "start_at": "2026-08-04T11:00:00+05:30",
            "end_at": "2026-08-04T11:30:00+05:30",
        },
        ("calendar", "event.standup"),
        False,
    ),
    (
        "LOOK_UP_CONTACT",
        {"contact_ref": "contact.asha"},
        "contact_lookup",
        {"contact_id": "contact.asha"},
        ("contacts", "contact.asha"),
        False,
    ),
    (
        "CREATE_NOTE",
        {"note_ref": "note.trip", "heading": "Trip", "content": "Pack light"},
        "note_create",
        {"note_id": "note.trip", "title": "Trip", "body": "Pack light"},
        ("notes", "note.trip"),
        False,
    ),
    (
        "ADD_LIST_ITEM",
        {"list_ref": "list.groceries", "item_ref": "item.rice", "item_text": "Rice"},
        "list_add_item",
        {"list_id": "list.groceries", "item_id": "item.rice", "text": "Rice"},
        ("lists", "list.groceries", "items", "item.rice"),
        False,
    ),
    (
        "SET_LIST_ITEM_CHECKED",
        {"list_ref": "list.groceries", "item_ref": "item.milk", "is_checked": True},
        "list_check_item",
        {"list_id": "list.groceries", "item_id": "item.milk", "checked": True},
        ("lists", "list.groceries", "items", "item.milk"),
        False,
    ),
    (
        "LOOK_UP_ROUTE",
        {"from_place_ref": "place.home", "to_place_ref": "place.office", "travel_mode": "driving"},
        "map_route",
        {"origin_id": "place.home", "destination_id": "place.office", "mode": "driving"},
        ("routes", "route.commute"),
        False,
    ),
    (
        "PROPOSE_MESSAGE",
        {
            "message_ref": "message.one",
            "recipient_ref": "contact.asha",
            "delivery_channel": "sms",
            "content": "On my way",
        },
        "message_send",
        {
            "message_id": "message.one",
            "to_contact_id": "contact.asha",
            "channel": "sms",
            "body": "On my way",
        },
        ("outbox", "message.one"),
        False,
    ),
    (
        "PLAY_MEDIA",
        {"track_ref": "track.raga"},
        "media_play",
        {"track_id": "track.raga"},
        ("media", "session"),
        False,
    ),
    (
        "PAUSE_MEDIA",
        {},
        "media_pause",
        {},
        ("media", "session"),
        True,
    ),
    (
        "SET_BOOLEAN_SETTING",
        {"setting_ref": "airplane_mode", "value": True},
        "setting_set_bool",
        {"key": "airplane_mode", "enabled": True},
        ("settings", "airplane_mode"),
        False,
    ),
]


def test_simulator_semantic_golden_vector_is_byte_stable() -> None:
    program = parse_sim_program(
        {
            "schema_version": SIM_PROGRAM_VERSION,
            "case_id": "case.golden",
            "initial_state": world_payload(),
            "steps": [
                {
                    "step_id": "step.one",
                    "decision": "CONFIRM",
                    "mode": "SINGLE",
                    "operations": [
                        {
                            "kind": "CREATE_NOTE",
                            "parameters": {
                                "note_ref": "note.golden",
                                "heading": "Golden",
                                "content": "Stable body",
                            },
                        }
                    ],
                    "missing": [],
                }
            ],
        }
    )
    compiled = compile_program_step(program.steps[0])
    reference = simulate_reference_action(compiled, program.initial_state)

    assert SIMULATOR_SCHEMA_SHA256 == (
        "3d59af1095f0f893fb8b36f6f45e5fd707b338726e6b04c8e8776f5d2daf8c39"
    )
    assert POLICY_CONTRACT_SHA256 == (
        "8431341158b2705280bcaec9c81f6a18ab5e8b70e3cd0b4bba3212b839a40b23"
    )
    assert program.initial_state.sha256() == (
        "818803164126fbbc1bf46be8c3676e4660cb3e9dae3da70096534ccafb33f34d"
    )
    assert program.sha256() == "5e7290c782ecffd633e2a0371615affd09b772b8647aae6436f37aad886e44b9"
    assert hashlib.sha256(compiled.canonical_json().encode("utf-8")).hexdigest() == (
        "1f4e1da275cf651480e3fa45f79c6a22b8308c90f3d41d54949423b1741c6a78"
    )
    assert reference.receipt.effects_sha256 == (
        "21766cbb1a777dfe7378779e500d2ecbaeee8ead65cdfc2db3f4142508f03bbb"
    )
    assert reference.counterfactual_state.sha256() == (
        "6cee6c63868e15ac9dda639eac44ba0c4264842ca7fe73916ebb95dd7d617525"
    )
    assert reference.receipt.observation_sha256 == (
        "bce153264844c7e57605283aa8c2ee3d92ad87d9a1f1b8e6b3a8425cba4feab2"
    )


@pytest.mark.parametrize(
    ("kind", "parameters", "tool", "args", "effect_path", "media_playing"), CASES
)
def test_all_thirteen_domain_operations_compile_and_match_independent_reference_effects(
    kind: str,
    parameters: dict[str, object],
    tool: str,
    args: dict[str, object],
    effect_path: tuple[str, ...],
    media_playing: bool,
) -> None:
    world = world_payload()
    if media_playing:
        world["media"].update(status="playing", track_id="track.raga")
    side_effecting = SIMULATOR_TOOL_REGISTRY[tool].side_effecting
    decision = "CONFIRM" if side_effecting else "CALL"
    program = parse_sim_program(
        {
            "schema_version": SIM_PROGRAM_VERSION,
            "case_id": f"case.{kind.lower()}",
            "initial_state": world,
            "steps": [
                {
                    "step_id": "step.one",
                    "decision": decision,
                    "mode": "SINGLE",
                    "operations": [{"kind": kind, "parameters": parameters}],
                    "missing": [],
                }
            ],
        }
    )
    compiled = compile_program_step(program.steps[0])
    reference = simulate_reference_action(compiled, program.initial_state)
    semantic_reference = simulate_reference_program_step(
        program.steps[0], program.initial_state, program_sha256=program.sha256()
    )
    visible = ActionSimulator().simulate(compiled, program.initial_state)
    round_trip = verify_program_round_trip(program)

    assert compiled.calls[0].tool == tool
    assert compiled.calls[0].to_dict()["args"] == args
    assert reference.receipt.status == "reference_applied"
    assert reference.receipt.effects[0].path == effect_path
    assert reference.receipt.reference_only is True
    assert reference.receipt.authorizes_execution is False
    assert visible.receipt.status == ("confirmation_required" if side_effecting else "simulated")
    assert visible.state.sha256() == program.initial_state.sha256()
    assert round_trip.passed is True
    assert (
        loads_reference_receipt(
            reference.receipt.canonical_json(),
            source=compiled,
            state=program.initial_state,
        ).sha256()
        == reference.receipt.sha256()
    )
    assert (
        loads_reference_receipt(
            semantic_reference.receipt.canonical_json(),
            source=program.steps[0],
            state=program.initial_state,
            program_sha256=program.sha256(),
        ).sha256()
        == semantic_reference.receipt.sha256()
    )
    assert (
        loads_transition_receipt(
            visible.receipt.canonical_json(),
            action=compiled,
            state=program.initial_state,
        ).sha256()
        == visible.receipt.sha256()
    )
    assert (
        loads_program_round_trip_receipt(round_trip.canonical_json(), program=program).sha256()
        == round_trip.sha256()
    )


def test_side_effecting_call_is_policy_blocked_and_never_mutates() -> None:
    state = WorldState.from_dict(world_payload())
    proposed = action(
        "CALL",
        [
            {
                "tool": "message_send",
                "args": {
                    "message_id": "message.one",
                    "to_contact_id": "contact.asha",
                    "channel": "sms",
                    "body": "Hello",
                },
            }
        ],
    )
    result = ActionSimulator().simulate(proposed, state)

    assert result.receipt.status == "policy_blocked"
    assert result.receipt.policy["policy_valid"] is False
    assert result.receipt.policy["expected_decision"] == "CONFIRM"
    assert result.state.sha256() == state.sha256()
    assert result.receipt.effects == ()
    assert result.receipt.external_side_effects is False
    assert result.receipt.authorizes_execution is False


def test_confirm_is_proposal_only_while_reference_api_is_explicitly_counterfactual() -> None:
    state = WorldState.from_dict(world_payload())
    proposed = action(
        "CONFIRM",
        [{"tool": "note_create", "args": {"note_id": "note.one", "title": "One", "body": "Body"}}],
    )
    visible = ActionSimulator().simulate(proposed, state)
    reference = simulate_reference_action(proposed, state)

    assert visible.receipt.status == "confirmation_required"
    assert visible.state.sha256() == state.sha256()
    assert "note.one" not in visible.state.notes
    assert reference.receipt.status == "reference_applied"
    assert reference.counterfactual_state.notes["note.one"]["body"] == "Body"
    assert reference.receipt.reference_only is True
    assert reference.receipt.authorizes_execution is False


def test_reference_serial_batch_rolls_back_when_later_call_fails() -> None:
    state = WorldState.from_dict(world_payload())
    proposed = action(
        "CONFIRM",
        [
            {
                "tool": "note_create",
                "args": {"note_id": "note.new", "title": "One", "body": "First"},
            },
            {
                "tool": "note_create",
                "args": {"note_id": "note.new", "title": "Two", "body": "Duplicate"},
            },
        ],
        mode="SERIAL",
    )
    result = simulate_reference_action(proposed, state)

    assert result.receipt.status == "reference_rejected"
    assert result.receipt.error_code == "entity_exists"
    assert result.receipt.effects == ()
    assert result.counterfactual_state.sha256() == state.sha256()


def test_parallel_conflicts_use_structured_targets_without_delimiter_collision() -> None:
    world = world_payload()
    world["lists"] = {
        "a": {"title": "A", "items": {}},
        "a:b": {"title": "AB", "items": {}},
    }
    state = WorldState.from_dict(world)
    distinct = action(
        "CONFIRM",
        [
            {"tool": "list_add_item", "args": {"list_id": "a:b", "item_id": "c", "text": "X"}},
            {"tool": "list_add_item", "args": {"list_id": "a", "item_id": "b:c", "text": "Y"}},
        ],
        mode="PARALLEL",
    )
    distinct_result = simulate_reference_action(distinct, state)
    assert distinct_result.receipt.status == "reference_applied"
    assert "c" in distinct_result.counterfactual_state.lists["a:b"]["items"]
    assert "b:c" in distinct_result.counterfactual_state.lists["a"]["items"]

    conflicting = action(
        "CONFIRM",
        [
            {"tool": "setting_set_bool", "args": {"key": "wifi", "enabled": False}},
            {"tool": "setting_set_bool", "args": {"key": "wifi", "enabled": True}},
        ],
        mode="PARALLEL",
    )
    conflict_result = simulate_reference_action(conflicting, state)
    assert conflict_result.receipt.status == "reference_rejected"
    assert conflict_result.receipt.error_code == "parallel_write_conflict"
    assert conflict_result.counterfactual_state.sha256() == state.sha256()


def test_round_trip_verifier_fails_closed_on_live_compiler_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = parse_sim_program(
        {
            "schema_version": SIM_PROGRAM_VERSION,
            "case_id": "case.compiler-mismatch",
            "initial_state": world_payload(),
            "steps": [
                {
                    "step_id": "step.lookup",
                    "decision": "CALL",
                    "mode": "SINGLE",
                    "operations": [
                        {"kind": "LOOK_UP_CONTACT", "parameters": {"contact_ref": "contact.asha"}}
                    ],
                    "missing": [],
                }
            ],
        }
    )

    def wrong_compile(_: object) -> dict[str, object]:
        return {"tool": "contact_lookup", "args": {"contact_id": "contact.dev"}}

    monkeypatch.setattr(simulator_module, "_compile_operation", wrong_compile)
    with pytest.raises(ActionSimulationError, match="runtime differs"):
        verify_program_round_trip(program)


def test_round_trip_rejects_identically_failed_confirm_references() -> None:
    program = parse_sim_program(
        {
            "schema_version": SIM_PROGRAM_VERSION,
            "case_id": "case.invalid-interval",
            "initial_state": world_payload(),
            "steps": [
                {
                    "step_id": "step.invalid-interval",
                    "decision": "CONFIRM",
                    "mode": "SINGLE",
                    "operations": [
                        {
                            "kind": "CREATE_CALENDAR_EVENT",
                            "parameters": {
                                "event_ref": "event.invalid",
                                "summary": "Impossible interval",
                                "begins_at": "2026-08-04T14:00:00+05:30",
                                "ends_at": "2026-08-04T13:00:00+05:30",
                            },
                        }
                    ],
                    "missing": [],
                }
            ],
        }
    )
    receipt = verify_program_round_trip(program)
    compiled = compile_program_step(program.steps[0])
    visible = ActionSimulator().simulate(compiled, program.initial_state)

    assert receipt.passed is False
    assert receipt.step_records[0]["reference_match"] is False
    assert visible.receipt.status == "rejected"
    assert visible.receipt.error_code == "invalid_interval"
    assert visible.receipt.effects == ()
    assert visible.state.sha256() == program.initial_state.sha256()
    assert (
        loads_transition_receipt(
            visible.receipt.canonical_json(),
            action=compiled,
            state=program.initial_state,
        ).sha256()
        == visible.receipt.sha256()
    )


def test_serial_calendar_repair_cannot_hide_an_invalid_intermediate_interval() -> None:
    program = parse_sim_program(
        {
            "schema_version": SIM_PROGRAM_VERSION,
            "case_id": "case.invalid-then-repaired",
            "initial_state": world_payload(),
            "steps": [
                {
                    "step_id": "step.invalid-then-repaired",
                    "decision": "CONFIRM",
                    "mode": "SERIAL",
                    "operations": [
                        {
                            "kind": "CREATE_CALENDAR_EVENT",
                            "parameters": {
                                "event_ref": "event.transient",
                                "summary": "Invalid first interval",
                                "begins_at": "2026-08-04T14:00:00+05:30",
                                "ends_at": "2026-08-04T13:00:00+05:30",
                            },
                        },
                        {
                            "kind": "RESCHEDULE_CALENDAR_EVENT",
                            "parameters": {
                                "event_ref": "event.transient",
                                "begins_at": "2026-08-04T15:00:00+05:30",
                                "ends_at": "2026-08-04T16:00:00+05:30",
                            },
                        },
                    ],
                    "missing": [],
                }
            ],
        }
    )
    action_ir = compile_program_step(program.steps[0])
    visible = ActionSimulator().simulate(action_ir, program.initial_state)
    compiled_reference = simulate_reference_action(action_ir, program.initial_state)
    semantic_reference = simulate_reference_program_step(
        program.steps[0],
        program.initial_state,
        program_sha256=program.sha256(),
    )
    round_trip = verify_program_round_trip(program)

    assert visible.receipt.status == "rejected"
    assert visible.receipt.error_code == "invalid_interval"
    assert visible.receipt.effects == ()
    assert visible.state.sha256() == program.initial_state.sha256()
    assert compiled_reference.receipt.status == "reference_rejected"
    assert semantic_reference.receipt.status == "reference_rejected"
    assert compiled_reference.receipt.effects == ()
    assert semantic_reference.receipt.effects == ()
    assert round_trip.passed is False
    assert round_trip.step_records[0]["reference_match"] is False
    assert (
        loads_program_round_trip_receipt(round_trip.canonical_json(), program=program).sha256()
        == round_trip.sha256()
    )


@pytest.mark.parametrize("decision", ["ABSTAIN", "CLARIFY"])
def test_control_decisions_never_mutate_and_are_non_authorizing(decision: str) -> None:
    state = WorldState.from_dict(world_payload())
    result = ActionSimulator().simulate(action(decision), state)
    assert result.receipt.status == "no_action"
    assert result.state.sha256() == state.sha256()
    assert result.receipt.effects == ()
    assert result.receipt.authorizes_execution is False


def test_receipt_loaders_fail_closed_on_hash_identity_extra_and_duplicate_tampering() -> None:
    state = WorldState.from_dict(world_payload())
    result_action = action(
        "CALL", [{"tool": "contact_lookup", "args": {"contact_id": "contact.asha"}}]
    )
    result = ActionSimulator().simulate(result_action, state)
    record = result.receipt.to_dict()
    assert record["schema_sha256"] == SIMULATOR_SCHEMA_SHA256
    assert record["policy_contract_sha256"] == POLICY_CONTRACT_SHA256

    bad_hash = copy.deepcopy(record)
    bad_hash["effects_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="effects hash"):
        loads_transition_receipt(json.dumps(bad_hash), action=result_action, state=state)

    bad_identity = copy.deepcopy(record)
    bad_identity["schema_sha256"] = "0" * 64
    with pytest.raises(ActionSimulationError, match="identity"):
        loads_transition_receipt(json.dumps(bad_identity), action=result_action, state=state)

    extra = copy.deepcopy(record)
    extra["unexpected"] = True
    with pytest.raises(ActionSimulationError, match="fields"):
        loads_transition_receipt(json.dumps(extra), action=result_action, state=state)

    with pytest.raises(Exception, match="duplicate"):
        loads_transition_receipt(
            '{"schema_version":"x","schema_version":"y"}',
            action=result_action,
            state=state,
        )


def test_receipt_loaders_reject_cross_field_nonboolean_and_context_forgery() -> None:
    state = WorldState.from_dict(world_payload())
    result_action = action(
        "CALL", [{"tool": "contact_lookup", "args": {"contact_id": "contact.asha"}}]
    )
    receipt = ActionSimulator().simulate(result_action, state).receipt

    cross_field = receipt.to_dict()
    cross_field["decision"] = "CONFIRM"
    with pytest.raises(ValueError, match="disagree"):
        loads_transition_receipt(json.dumps(cross_field), action=result_action, state=state)

    nonboolean = receipt.to_dict()
    nonboolean["external_side_effects"] = 0
    nonboolean["authorizes_execution"] = 0
    with pytest.raises(ActionSimulationError, match="JSON boolean"):
        loads_transition_receipt(json.dumps(nonboolean), action=result_action, state=state)

    forged_context = receipt.to_dict()
    forged_context["input_state_sha256"] = "0" * 64
    forged_context["output_state_sha256"] = "0" * 64
    with pytest.raises(ActionSimulationError, match="does not reproduce"):
        loads_transition_receipt(json.dumps(forged_context), action=result_action, state=state)

    reference = simulate_reference_action(result_action, state).receipt
    reference_nonboolean = reference.to_dict()
    reference_nonboolean["reference_only"] = 1
    reference_nonboolean["authorizes_execution"] = 0
    reference_nonboolean["external_side_effects"] = 0
    with pytest.raises(ActionSimulationError, match="JSON boolean"):
        loads_reference_receipt(json.dumps(reference_nonboolean), source=result_action, state=state)

    forged_reference = reference.to_dict()
    forged_reference["source_sha256"] = "0" * 64
    with pytest.raises(ActionSimulationError, match="does not reproduce"):
        loads_reference_receipt(json.dumps(forged_reference), source=result_action, state=state)


def test_receipts_bind_live_runtime_and_direct_crosslink_construction_is_sealed() -> None:
    state = WorldState.from_dict(world_payload())
    result_action = action(
        "CALL", [{"tool": "contact_lookup", "args": {"contact_id": "contact.asha"}}]
    )
    transition = ActionSimulator().simulate(result_action, state).receipt
    reference = simulate_reference_action(result_action, state).receipt
    program = parse_sim_program(
        {
            "schema_version": SIM_PROGRAM_VERSION,
            "case_id": "case.runtime-receipts",
            "initial_state": world_payload(),
            "steps": [
                {
                    "step_id": "step.lookup",
                    "decision": "CALL",
                    "mode": "SINGLE",
                    "operations": [
                        {
                            "kind": "LOOK_UP_CONTACT",
                            "parameters": {"contact_ref": "contact.asha"},
                        }
                    ],
                    "missing": [],
                }
            ],
        }
    )
    round_trip = verify_program_round_trip(program)
    runtime_sha = action_simulator_runtime_sha256()

    assert isinstance(transition, TransitionReceipt)
    assert isinstance(reference, ReferenceEffectReceipt)
    assert isinstance(round_trip, ProgramRoundTripReceipt)
    assert transition.simulator_runtime_sha256 == runtime_sha
    assert reference.simulator_runtime_sha256 == runtime_sha
    assert round_trip.simulator_runtime_sha256 == runtime_sha

    with pytest.raises(TypeError, match="constructed by"):
        replace(transition, input_state_sha256="0" * 64)
    with pytest.raises(TypeError, match="constructed by"):
        replace(reference, source_sha256="0" * 64)
    with pytest.raises(TypeError, match="constructed by"):
        replace(round_trip, final_state_sha256="0" * 64)

    forged_runtime = transition.to_dict()
    forged_runtime["simulator_runtime_sha256"] = "0" * 64
    with pytest.raises(ActionSimulationError, match="identity"):
        loads_transition_receipt(json.dumps(forged_runtime), action=result_action, state=state)


def test_schema_and_live_code_mutation_fail_closed_before_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorldState.from_dict(world_payload())
    safe_action = action(
        "CALL", [{"tool": "contact_lookup", "args": {"contact_id": "contact.asha"}}]
    )
    side_effecting = SIMULATOR_TOOL_REGISTRY["message_send"]
    baseline_runtime = action_simulator_runtime_sha256()

    object.__setattr__(side_effecting, "side_effecting", False)
    try:
        assert simulator_tool_registry_snapshot()["message_send"].side_effecting is True
        with pytest.raises(ActionSimulationError, match="schema or policy contract changed"):
            ActionSimulator().simulate(safe_action, state)
    finally:
        object.__setattr__(side_effecting, "side_effecting", True)
    assert action_simulator_runtime_sha256() == baseline_runtime

    with monkeypatch.context() as patcher:
        patcher.setattr(ActionSimulator, "schemas", {})
        with pytest.raises(ActionSimulationError, match="registry was replaced"):
            ActionSimulator().simulate(safe_action, state)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            simulator_module,
            "_entity_observation",
            lambda _tool, _entity_id, _value: {"kind": "forged"},
        )
        assert action_simulator_runtime_sha256() != baseline_runtime
        with pytest.raises(ActionSimulationError, match="runtime differs"):
            ActionSimulator().simulate(safe_action, state)

    assert action_simulator_runtime_sha256() == baseline_runtime


def test_action_snapshot_rejects_mid_read_nested_payload_change() -> None:
    template = action("ABSTAIN")

    class FlakyActionIR(ActionIR):
        __slots__ = ("reads",)

        def __init__(self) -> None:
            super().__init__(
                decision=template.decision,
                calls=template.calls,
                mode=template.mode,
                missing=template.missing,
            )
            object.__setattr__(self, "reads", 0)

        def canonical_json(self) -> str:
            object.__setattr__(self, "reads", self.reads + 1)
            if self.reads == 1:
                return '{"decision":"ABSTAIN"}'
            return '{"decision":"CLARIFY","missing":["recipient"]}'

    with pytest.raises(ActionSimulationError, match="changed during its stable snapshot"):
        ActionSimulator().simulate(FlakyActionIR(), WorldState.from_dict(world_payload()))


def test_simulate_program_uses_only_policy_safe_visible_state() -> None:
    program = parse_sim_program(
        {
            "schema_version": SIM_PROGRAM_VERSION,
            "case_id": "case.visible",
            "initial_state": world_payload(),
            "steps": [
                {
                    "step_id": "step.note",
                    "decision": "CONFIRM",
                    "mode": "SINGLE",
                    "operations": [
                        {
                            "kind": "CREATE_NOTE",
                            "parameters": {
                                "note_ref": "note.one",
                                "heading": "One",
                                "content": "Body",
                            },
                        }
                    ],
                    "missing": [],
                }
            ],
        }
    )
    result = simulate_program(program)
    assert result.results[0].receipt.status == "confirmation_required"
    assert result.final_state.sha256() == program.initial_state.sha256()
    assert "note.one" not in result.final_state.notes


def test_semantic_oracle_has_no_external_execution_surface() -> None:
    assert not hasattr(ActionSimulator, "execute_external")
    assert not hasattr(ActionSimulator, "connect")
    assert len(SIMULATOR_TOOL_REGISTRY) == 13
