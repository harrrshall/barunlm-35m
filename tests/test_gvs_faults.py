from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

import barunlm.evaluation.gvs_faults as faults_module
from barunlm.evaluation.action_ir import parse_action_ir
from barunlm.evaluation.action_simulator import SIMULATOR_TOOL_REGISTRY, ActionSimulator
from barunlm.evaluation.gvs_faults import (
    FaultCertification,
    FaultDescriptor,
    FaultKind,
    GVSFaultError,
    certify_single_fault,
    gvs_fault_runtime_sha256,
    loads_fault_certification,
    make_single_fault,
)
from barunlm.evaluation.sim_program import WORLD_STATE_VERSION, WorldState


def state_payload() -> dict[str, object]:
    return {
        "schema_version": WORLD_STATE_VERSION,
        "reference_time": "2026-08-04T09:30:00+05:30",
        "timezone": "Asia/Kolkata",
        "reminders": {
            "reminder.old": {
                "title": "Report",
                "due_at": "2026-08-05T17:00:00+05:30",
                "completed": False,
            }
        },
        "calendar": {
            "event.old": {
                "title": "Meet",
                "start_at": "2026-08-04T10:00:00+05:30",
                "end_at": "2026-08-04T10:30:00+05:30",
            }
        },
        "contacts": {
            "contact.asha": {"name": "Asha", "channel": "sms"},
            "contact.dev": {"name": "Dev", "channel": "sms"},
            "contact.email": {"name": "Email", "channel": "email"},
        },
        "notes": {},
        "lists": {
            "list.one": {
                "title": "One",
                "items": {
                    "item.a": {"text": "A", "checked": False},
                    "item.b": {"text": "B", "checked": False},
                },
            }
        },
        "places": {
            "place.home": {"name": "Home"},
            "place.office": {"name": "Office"},
        },
        "routes": {
            "route.drive": {
                "origin_id": "place.home",
                "destination_id": "place.office",
                "mode": "driving",
                "distance_m": 100,
                "duration_s": 30,
            },
            "route.walk": {
                "origin_id": "place.home",
                "destination_id": "place.office",
                "mode": "walking",
                "distance_m": 120,
                "duration_s": 90,
            },
        },
        "outbox": [],
        "media": {
            "status": "stopped",
            "track_id": None,
            "catalog": {"track.a": {"title": "A"}, "track.b": {"title": "B"}},
        },
        "settings": {"wifi": True, "airplane_mode": False, "brightness": 50},
    }


def state() -> WorldState:
    return WorldState.from_dict(state_payload())


def call(tool: str, args: dict[str, object], *, decision: str | None = None):
    if decision is None:
        decision = "CONFIRM" if SIMULATOR_TOOL_REGISTRY[tool].side_effecting else "CALL"
    return parse_action_ir(
        json.dumps(
            {"decision": decision, "mode": "SINGLE", "calls": [{"tool": tool, "args": args}]}
        ),
        SIMULATOR_TOOL_REGISTRY,
    )


def control(decision: str):
    payload: dict[str, object] = {"decision": decision}
    if decision == "CLARIFY":
        payload["missing"] = ["recipient"]
    return parse_action_ir(json.dumps(payload), SIMULATOR_TOOL_REGISTRY)


@pytest.mark.parametrize(
    ("gold", "descriptor", "expected_output"),
    [
        (
            call("contact_lookup", {"contact_id": "contact.asha"}),
            FaultDescriptor(FaultKind.ENTITY, 0, "contact_id", "contact.dev"),
            "effects",
        ),
        (
            call("note_create", {"note_id": "note.one", "title": "One", "body": "Gold"}),
            FaultDescriptor(FaultKind.ARGUMENT_VALUE, 0, "body", "Near miss"),
            "counterfactual_state",
        ),
        (
            call(
                "reminder_update",
                {
                    "reminder_id": "reminder.old",
                    "title": "Report",
                    "due_at": "2026-08-05T17:00:00+05:30",
                },
            ),
            FaultDescriptor(FaultKind.TEMPORAL, 0, "due_at", "2026-08-06T17:00:00+05:30"),
            "counterfactual_state",
        ),
        (
            call(
                "message_send",
                {
                    "message_id": "message.one",
                    "to_contact_id": "contact.asha",
                    "channel": "sms",
                    "body": "Hi",
                },
            ),
            FaultDescriptor(FaultKind.RECIPIENT, 0, "to_contact_id", "contact.dev"),
            "counterfactual_state",
        ),
        (
            call("setting_set_bool", {"key": "airplane_mode", "enabled": True}),
            FaultDescriptor(FaultKind.BOOLEAN, 0, "enabled", False),
            "counterfactual_state",
        ),
        (
            call(
                "list_check_item",
                {"list_id": "list.one", "item_id": "item.a", "checked": True},
            ),
            FaultDescriptor(FaultKind.ITEM, 0, "item_id", "item.b"),
            "counterfactual_state",
        ),
        (
            call(
                "map_route",
                {"origin_id": "place.home", "destination_id": "place.office", "mode": "driving"},
            ),
            FaultDescriptor(FaultKind.ARGUMENT_VALUE, 0, "mode", "walking"),
            "effects",
        ),
        (
            call("contact_lookup", {"contact_id": "contact.asha"}),
            FaultDescriptor(
                FaultKind.OPERATION,
                0,
                "tool",
                {
                    "tool": "map_route",
                    "args": {
                        "origin_id": "place.home",
                        "destination_id": "place.office",
                        "mode": "driving",
                    },
                },
            ),
            "effects",
        ),
    ],
)
def test_semantic_fault_families_require_valid_non_equivalent_reference_effects(
    gold: object, descriptor: FaultDescriptor, expected_output: str
) -> None:
    certification = certify_single_fault(state(), gold, descriptor)
    record = certification.to_dict()

    assert record["schema_valid"] is True
    assert record["semantic_type_valid"] is True
    assert record["semantic_change_count"] == 1
    assert record["reference_only"] is True
    assert record["authorizes_execution"] is False
    assert record["external_side_effects"] is False
    assert expected_output in certification.differing_semantic_outputs
    assert certification.gold_action_sha256 != certification.faulty_action_sha256
    assert (
        loads_fault_certification(certification.canonical_json(), state=state()).sha256()
        == certification.sha256()
    )
    assert (
        loads_fault_certification(certification.canonical_json(), state=state()).canonical_json()
        == certification.canonical_json()
    )


@pytest.mark.parametrize(
    ("gold", "replacement"),
    [
        (call("contact_lookup", {"contact_id": "contact.asha"}), "CONFIRM"),
        (call("contact_lookup", {"contact_id": "contact.asha"}), "ABSTAIN"),
        (call("contact_lookup", {"contact_id": "contact.asha"}), "CLARIFY"),
        (
            call(
                "message_send",
                {
                    "message_id": "message.one",
                    "to_contact_id": "contact.asha",
                    "channel": "sms",
                    "body": "Hi",
                },
            ),
            "CALL",
        ),
        (
            call(
                "message_send",
                {
                    "message_id": "message.one",
                    "to_contact_id": "contact.asha",
                    "channel": "sms",
                    "body": "Hi",
                },
            ),
            "ABSTAIN",
        ),
        (
            call(
                "message_send",
                {
                    "message_id": "message.one",
                    "to_contact_id": "contact.asha",
                    "channel": "sms",
                    "body": "Hi",
                },
            ),
            "CLARIFY",
        ),
        (
            control("ABSTAIN"),
            {
                "decision": "CALL",
                "mode": "SINGLE",
                "calls": [{"tool": "contact_lookup", "args": {"contact_id": "contact.asha"}}],
            },
        ),
        (
            control("ABSTAIN"),
            {
                "decision": "CONFIRM",
                "mode": "SINGLE",
                "calls": [
                    {
                        "tool": "message_send",
                        "args": {
                            "message_id": "message.one",
                            "to_contact_id": "contact.asha",
                            "channel": "sms",
                            "body": "Hi",
                        },
                    }
                ],
            },
        ),
        (control("ABSTAIN"), "CLARIFY"),
        (
            control("CLARIFY"),
            {
                "decision": "CONFIRM",
                "mode": "SINGLE",
                "calls": [
                    {
                        "tool": "message_send",
                        "args": {
                            "message_id": "message.one",
                            "to_contact_id": "contact.asha",
                            "channel": "sms",
                            "body": "Hi",
                        },
                    }
                ],
            },
        ),
        (
            control("CLARIFY"),
            {
                "decision": "CALL",
                "mode": "SINGLE",
                "calls": [{"tool": "contact_lookup", "args": {"contact_id": "contact.asha"}}],
            },
        ),
        (control("CLARIFY"), "ABSTAIN"),
    ],
)
def test_all_control_decision_directions_are_one_high_level_semantic_fault(
    gold: object, replacement: object
) -> None:
    descriptor = FaultDescriptor(FaultKind.DECISION, None, "decision", replacement)
    certification = certify_single_fault(state(), gold, descriptor)
    assert certification.faulty_action.decision is not certification.gold_action.decision
    assert certification.to_dict()["semantic_change_count"] == 1
    assert certification.differing_semantic_outputs


def test_callable_decision_fault_cannot_smuggle_different_calls_or_arguments() -> None:
    gold = call("contact_lookup", {"contact_id": "contact.asha"})
    smuggled = FaultDescriptor(
        FaultKind.DECISION,
        None,
        "decision",
        {
            "decision": "CONFIRM",
            "mode": "SINGLE",
            "calls": [
                {
                    "tool": "message_send",
                    "args": {
                        "message_id": "message.smuggled",
                        "to_contact_id": "contact.email",
                        "channel": "email",
                        "body": "Unrelated payload",
                    },
                }
            ],
        },
    )
    with pytest.raises(GVSFaultError) as captured:
        make_single_fault(gold, smuggled)
    assert captured.value.code == "multiple_semantic_changes"

    same_calls = gold.to_dict()
    same_calls["decision"] = "CONFIRM"
    exact = FaultDescriptor(FaultKind.DECISION, None, "decision", same_calls)
    certification = certify_single_fault(state(), gold, exact)
    assert certification.faulty_action.calls == certification.gold_action.calls
    assert certification.faulty_action.mode is certification.gold_action.mode


def test_decision_object_replacement_is_only_for_an_explicit_callable_payload() -> None:
    descriptor = FaultDescriptor(
        FaultKind.DECISION,
        None,
        "decision",
        {"decision": "CLARIFY", "missing": ["custom_missing_field"]},
    )
    with pytest.raises(GVSFaultError) as captured:
        make_single_fault(control("ABSTAIN"), descriptor)
    assert captured.value.code == "decision_payload_scope"


def test_side_effecting_call_cannot_be_used_as_gold_and_confirm_is_not_mislabeled() -> None:
    unsafe_gold = call(
        "message_send",
        {
            "message_id": "message.one",
            "to_contact_id": "contact.asha",
            "channel": "sms",
            "body": "Hi",
        },
        decision="CALL",
    )
    descriptor = FaultDescriptor(FaultKind.DECISION, None, "decision", "CONFIRM")
    with pytest.raises(GVSFaultError, match="policy"):
        certify_single_fault(state(), unsafe_gold, descriptor)
    visible = ActionSimulator().simulate(unsafe_gold, state())
    assert visible.receipt.status == "policy_blocked"
    assert visible.state.sha256() == state().sha256()


@pytest.mark.parametrize(
    ("gold", "descriptor", "error_code"),
    [
        (
            call(
                "reminder_update",
                {
                    "reminder_id": "reminder.old",
                    "title": "Report",
                    "due_at": "2026-08-05T17:00:00+05:30",
                },
            ),
            FaultDescriptor(FaultKind.TEMPORAL, 0, "due_at", "not-a-timestamp"),
            "semantic_type_invalid",
        ),
        (
            call("contact_lookup", {"contact_id": "contact.asha"}),
            FaultDescriptor(FaultKind.ENTITY, 0, "contact_id", "contact.missing"),
            "semantic_type_invalid",
        ),
        (
            call(
                "map_route",
                {"origin_id": "place.home", "destination_id": "place.office", "mode": "driving"},
            ),
            FaultDescriptor(FaultKind.ARGUMENT_VALUE, 0, "mode", "teleport"),
            "semantic_type_invalid",
        ),
        (
            call("setting_set_bool", {"key": "wifi", "enabled": False}),
            FaultDescriptor(FaultKind.ENTITY, 0, "key", "brightness"),
            "semantic_type_invalid",
        ),
        (
            call(
                "message_send",
                {
                    "message_id": "message.one",
                    "to_contact_id": "contact.asha",
                    "channel": "sms",
                    "body": "Hi",
                },
            ),
            FaultDescriptor(FaultKind.RECIPIENT, 0, "to_contact_id", "contact.email"),
            "semantic_type_invalid",
        ),
        (
            call("note_create", {"note_id": "note.one", "title": "One", "body": "Body"}),
            FaultDescriptor(FaultKind.ENTITY, 0, "note_id", "bad?identifier"),
            "semantic_type_invalid",
        ),
        (
            call(
                "reminder_create",
                {
                    "reminder_id": "reminder.new",
                    "title": "New",
                    "due_at": "2026-08-06T10:00:00+05:30",
                },
            ),
            FaultDescriptor(FaultKind.OPERATION, 0, "tool", "reminder_update"),
            "semantic_type_invalid",
        ),
    ],
)
def test_malformed_wrong_kind_nonexistent_channel_and_easy_rejection_faults_are_rejected(
    gold: object, descriptor: FaultDescriptor, error_code: str
) -> None:
    with pytest.raises(GVSFaultError) as captured:
        certify_single_fault(state(), gold, descriptor)
    assert captured.value.code == error_code


def test_semantically_equivalent_timestamp_spelling_is_not_a_negative() -> None:
    utc_state = WorldState.from_dict(
        {
            "schema_version": WORLD_STATE_VERSION,
            "reference_time": "2026-08-04T04:00:00+00:00",
            "timezone": "UTC",
            "reminders": {
                "reminder.old": {
                    "title": "Report",
                    "due_at": "2026-08-05T12:00:00+00:00",
                    "completed": False,
                }
            },
            "calendar": {},
            "contacts": {},
            "notes": {},
            "lists": {},
            "places": {},
            "routes": {},
            "outbox": [],
            "media": {"status": "stopped", "track_id": None, "catalog": {}},
            "settings": {},
        }
    )
    gold = call(
        "reminder_update",
        {
            "reminder_id": "reminder.old",
            "title": "Report",
            "due_at": "2026-08-05T12:00:00+00:00",
        },
    )
    descriptor = FaultDescriptor(FaultKind.TEMPORAL, 0, "due_at", "2026-08-05T12:00:00-00:00")
    with pytest.raises(GVSFaultError) as captured:
        certify_single_fault(utc_state, gold, descriptor)
    assert captured.value.code == "semantic_equivalence"


def test_temporal_fault_cannot_be_repaired_by_a_later_serial_calendar_call() -> None:
    gold = parse_action_ir(
        json.dumps(
            {
                "decision": "CONFIRM",
                "mode": "SERIAL",
                "calls": [
                    {
                        "tool": "calendar_create",
                        "args": {
                            "event_id": "event.transient",
                            "title": "Transient",
                            "start_at": "2026-08-04T12:00:00+05:30",
                            "end_at": "2026-08-04T13:00:00+05:30",
                        },
                    },
                    {
                        "tool": "calendar_reschedule",
                        "args": {
                            "event_id": "event.transient",
                            "start_at": "2026-08-04T15:00:00+05:30",
                            "end_at": "2026-08-04T16:00:00+05:30",
                        },
                    },
                ],
            }
        ),
        SIMULATOR_TOOL_REGISTRY,
    )
    descriptor = FaultDescriptor(
        FaultKind.TEMPORAL,
        0,
        "end_at",
        "2026-08-04T11:00:00+05:30",
    )
    with pytest.raises(GVSFaultError) as captured:
        certify_single_fault(state(), gold, descriptor)
    assert captured.value.code == "semantic_type_invalid"


def test_operation_substitution_is_one_high_level_fault_and_preserves_policy_class() -> None:
    gold = call("contact_lookup", {"contact_id": "contact.asha"})
    descriptor = FaultDescriptor(
        FaultKind.OPERATION,
        0,
        "tool",
        {
            "tool": "map_route",
            "args": {
                "origin_id": "place.home",
                "destination_id": "place.office",
                "mode": "driving",
            },
        },
    )
    certification = certify_single_fault(state(), gold, descriptor)

    assert descriptor.semantic_path == "$.calls[0]"
    assert certification.faulty_action.calls[0].tool == "map_route"
    assert certification.to_dict()["semantic_change_count"] == 1
    assert "effects" in certification.differing_semantic_outputs

    unsafe_replacement = FaultDescriptor(
        FaultKind.OPERATION,
        0,
        "tool",
        {
            "tool": "note_create",
            "args": {"note_id": "note.one", "title": "One", "body": "Body"},
        },
    )
    with pytest.raises(GVSFaultError) as captured:
        make_single_fault(gold, unsafe_replacement)
    assert captured.value.code == "not_policy_preserving"


def test_different_valid_setting_target_is_semantic_even_if_both_writes_are_noops() -> None:
    payload = state_payload()
    payload["settings"]["wifi"] = False
    current = WorldState.from_dict(payload)
    gold = call("setting_set_bool", {"key": "airplane_mode", "enabled": False})
    descriptor = FaultDescriptor(FaultKind.ENTITY, 0, "key", "wifi")
    certification = certify_single_fault(current, gold, descriptor)
    assert "effects" in certification.differing_semantic_outputs
    assert "counterfactual_state" not in certification.differing_semantic_outputs


def test_fault_construction_and_certificate_are_deterministic_and_immutable() -> None:
    replacement = "contact.dev"
    descriptor = FaultDescriptor(FaultKind.ENTITY, 0, "contact_id", replacement)
    gold = call("contact_lookup", {"contact_id": "contact.asha"})
    first = certify_single_fault(state(), gold, descriptor)
    second = certify_single_fault(state(), gold, descriptor)

    assert first.canonical_json() == second.canonical_json()
    assert first.sha256() == second.sha256()
    assert (
        first.faulty_action.canonical_json() == make_single_fault(gold, descriptor).canonical_json()
    )


def test_certificate_loader_rejects_hash_identity_extra_and_duplicate_tampering() -> None:
    certification = certify_single_fault(
        state(),
        call("contact_lookup", {"contact_id": "contact.asha"}),
        FaultDescriptor(FaultKind.ENTITY, 0, "contact_id", "contact.dev"),
    )
    record = certification.to_dict()

    bad_hash = copy.deepcopy(record)
    bad_hash["gold_action_sha256"] = "0" * 64
    with pytest.raises(GVSFaultError, match="hash"):
        loads_fault_certification(json.dumps(bad_hash), state=state())

    bad_identity = copy.deepcopy(record)
    bad_identity["schema_sha256"] = "0" * 64
    with pytest.raises(GVSFaultError, match="identity"):
        loads_fault_certification(json.dumps(bad_identity), state=state())

    extra = copy.deepcopy(record)
    extra["unexpected"] = True
    with pytest.raises(GVSFaultError, match="fields"):
        loads_fault_certification(json.dumps(extra), state=state())

    with pytest.raises(Exception, match="duplicate"):
        loads_fault_certification('{"schema_version":"x","schema_version":"y"}', state=state())


def test_certificate_loader_requires_state_replay_and_exact_security_types() -> None:
    certification = certify_single_fault(
        state(),
        call("contact_lookup", {"contact_id": "contact.asha"}),
        FaultDescriptor(FaultKind.ENTITY, 0, "contact_id", "contact.dev"),
    )
    with pytest.raises(TypeError):
        loads_fault_certification(certification.canonical_json())  # type: ignore[call-arg]

    forged = certification.to_dict()
    for name in (
        "input_state_sha256",
        "gold_visible_receipt_sha256",
        "faulty_visible_receipt_sha256",
        "gold_reference_receipt_sha256",
        "faulty_reference_receipt_sha256",
        "gold_counterfactual_state_sha256",
        "faulty_counterfactual_state_sha256",
        "timezone_runtime_sha256",
    ):
        forged[name] = "0" * 64
    forged["differing_semantic_outputs"] = ["policy"]
    with pytest.raises(GVSFaultError, match="does not reproduce"):
        loads_fault_certification(json.dumps(forged), state=state())

    nonboolean = certification.to_dict()
    nonboolean["schema_valid"] = 1
    nonboolean["semantic_type_valid"] = 1
    nonboolean["semantic_change_count"] = True
    nonboolean["reference_only"] = 1
    nonboolean["authorizes_execution"] = 0
    nonboolean["external_side_effects"] = 0
    with pytest.raises(GVSFaultError, match="invalid JSON types"):
        loads_fault_certification(json.dumps(nonboolean), state=state())


def test_certificate_runtime_crosslinks_are_bound_and_constructor_is_sealed() -> None:
    certification = certify_single_fault(
        state(),
        call("contact_lookup", {"contact_id": "contact.asha"}),
        FaultDescriptor(FaultKind.ENTITY, 0, "contact_id", "contact.dev"),
    )
    assert isinstance(certification, FaultCertification)
    assert certification.fault_runtime_sha256 == gvs_fault_runtime_sha256()
    assert certification.simulator_runtime_sha256 == faults_module.action_simulator_runtime_sha256()
    with pytest.raises(TypeError, match="constructed by"):
        replace(certification, input_state_sha256="0" * 64)

    forged = certification.to_dict()
    forged["fault_runtime_sha256"] = "0" * 64
    with pytest.raises(GVSFaultError, match="identity"):
        loads_fault_certification(json.dumps(forged), state=state())


def test_fault_taxonomy_and_live_code_mutation_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gold = call("contact_lookup", {"contact_id": "contact.asha"})
    descriptor = FaultDescriptor(FaultKind.ENTITY, 0, "contact_id", "contact.dev")
    baseline_runtime = gvs_fault_runtime_sha256()
    original_arguments = faults_module._ARGUMENTS_BY_KIND[FaultKind.ENTITY]
    faults_module._ARGUMENTS_BY_KIND[FaultKind.ENTITY] = frozenset({"contact_id"})
    try:
        assert gvs_fault_runtime_sha256() != baseline_runtime
        with pytest.raises(GVSFaultError, match="runtime differs"):
            make_single_fault(gold, descriptor)
    finally:
        faults_module._ARGUMENTS_BY_KIND[FaultKind.ENTITY] = original_arguments

    with monkeypatch.context() as patcher:
        patcher.setattr(faults_module, "_semantic_diff", lambda _left, _right: ())
        assert gvs_fault_runtime_sha256() != baseline_runtime
        with pytest.raises(GVSFaultError, match="runtime differs"):
            certify_single_fault(state(), gold, descriptor)
    assert gvs_fault_runtime_sha256() == baseline_runtime


@pytest.mark.parametrize(
    "descriptor",
    [
        FaultDescriptor(FaultKind.BOOLEAN, 0, "enabled", "false"),
        FaultDescriptor(FaultKind.BOOLEAN, 0, "enabled", True),
        FaultDescriptor(FaultKind.TEMPORAL, 0, "enabled", False),
        FaultDescriptor(FaultKind.OPERATION, 0, "tool", "message_send"),
    ],
)
def test_faults_fail_closed_when_not_single_typed_or_supported(
    descriptor: FaultDescriptor,
) -> None:
    gold = call("setting_set_bool", {"key": "airplane_mode", "enabled": True})
    with pytest.raises(GVSFaultError):
        make_single_fault(gold, descriptor)
