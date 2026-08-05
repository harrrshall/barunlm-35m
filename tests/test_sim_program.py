from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections.abc import Iterator, Mapping
from types import MappingProxyType

import pytest

import barunlm.evaluation.sim_program as sim_program_module
from barunlm.evaluation.action_ir import CallMode, Decision
from barunlm.evaluation.sim_program import (
    MAX_SIM_COLLECTION_ITEMS,
    MAX_SIM_INTEGER_ABS,
    MAX_SIM_JSON_BYTES,
    MAX_SIM_STRING_BYTES,
    SEMANTIC_OPERATION_SPECS,
    SIM_PROGRAM_VERSION,
    TIMEZONE_RUNTIME_VERSION,
    WORLD_STATE_VERSION,
    ProgramStep,
    SemanticOperation,
    SimProgramError,
    WorldState,
    loads_sim_program,
    loads_strict_json_object,
    parse_sim_program,
    snapshot_strict_json,
    timezone_runtime_identity,
    timezone_runtime_sha256,
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
        "places": {
            "place.home": {"name": "Home"},
            "place.office": {"name": "Office"},
        },
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


def program_payload() -> dict[str, object]:
    return {
        "schema_version": SIM_PROGRAM_VERSION,
        "case_id": "case.roundtrip",
        "initial_state": world_payload(),
        "steps": [
            {
                "step_id": "step.create",
                "decision": "CONFIRM",
                "mode": "SINGLE",
                "operations": [
                    {
                        "kind": "CREATE_NOTE",
                        "parameters": {
                            "note_ref": "note.trip",
                            "heading": "Trip",
                            "content": "Pack light",
                        },
                    }
                ],
                "missing": [],
            },
            {
                "step_id": "step.clarify",
                "decision": "CLARIFY",
                "mode": None,
                "operations": [],
                "missing": ["recipient"],
            },
        ],
    }


def test_program_snapshot_is_deeply_detached_immutable_and_round_trips() -> None:
    payload = program_payload()
    program = parse_sim_program(payload)
    original_hash = program.sha256()

    payload["initial_state"]["contacts"]["contact.asha"]["name"] = "Changed"
    payload["steps"][0]["operations"][0]["parameters"]["content"] = "Changed"

    assert program.initial_state.contacts["contact.asha"]["name"] == "Asha"
    assert program.steps[0].operations[0].parameters["content"] == "Pack light"
    assert program.sha256() == original_hash
    assert loads_sim_program(program.canonical_json()).canonical_json() == program.canonical_json()
    with pytest.raises(TypeError):
        program.initial_state.contacts["contact.new"] = {}  # type: ignore[index]


def test_world_state_direct_constructor_cannot_bypass_deep_immutability() -> None:
    state = WorldState.from_dict(world_payload())
    nested = {"reminder.old": {"title": "Mutable"}}
    with pytest.raises(TypeError, match="from_dict"):
        WorldState(
            reference_time=state.reference_time,
            timezone=state.timezone,
            reminders=MappingProxyType(nested),
            calendar=state.calendar,
            contacts=state.contacts,
            notes=state.notes,
            lists=state.lists,
            places=state.places,
            routes=state.routes,
            outbox=state.outbox,
            media=state.media,
            settings=state.settings,
        )


def test_semantic_operation_has_independent_vocabulary_and_snapshots_parameters() -> None:
    raw = {"note_ref": "note.one", "heading": "One", "content": "Body"}
    operation = SemanticOperation("CREATE_NOTE", raw)
    raw["content"] = "mutated"

    assert operation.parameters["content"] == "Body"
    assert operation.to_dict() == {
        "kind": "CREATE_NOTE",
        "parameters": {"content": "Body", "heading": "One", "note_ref": "note.one"},
    }
    assert set(SEMANTIC_OPERATION_SPECS) == {
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
    }


def test_gold_program_policy_requires_confirm_for_side_effects_and_call_for_safe_tools() -> None:
    unsafe_call = program_payload()
    unsafe_call["steps"][0]["decision"] = "CALL"
    with pytest.raises(SimProgramError, match="require CONFIRM"):
        parse_sim_program(unsafe_call)

    overconfirm = program_payload()
    overconfirm["steps"][0]["operations"] = [
        {"kind": "LOOK_UP_CONTACT", "parameters": {"contact_ref": "contact.asha"}}
    ]
    with pytest.raises(SimProgramError, match="require CALL"):
        parse_sim_program(overconfirm)

    safe = program_payload()
    safe["steps"][0]["decision"] = "CALL"
    safe["steps"][0]["operations"] = [
        {"kind": "LOOK_UP_CONTACT", "parameters": {"contact_ref": "contact.asha"}}
    ]
    assert parse_sim_program(safe).steps[0].decision is Decision.CALL


def test_direct_program_step_enforces_the_same_gold_policy() -> None:
    operation = SemanticOperation("SET_BOOLEAN_SETTING", {"setting_ref": "wifi", "value": False})
    with pytest.raises(SimProgramError, match="require CONFIRM"):
        ProgramStep(
            step_id="step.unsafe",
            decision=Decision.CALL,
            mode=CallMode.SINGLE,
            operations=(operation,),
            missing=(),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(extra=True), "unknown fields"),
        (lambda payload: payload["initial_state"].update(timezone="Not/AZone"), "unknown IANA"),
        (
            lambda payload: payload["initial_state"].update(
                reference_time="2026-08-04T09:30:00+00:00"
            ),
            "do not match",
        ),
        (
            lambda payload: payload["initial_state"].update(
                reference_time="2201-01-01T00:00:00+05:30"
            ),
            "between",
        ),
        (
            lambda payload: payload["initial_state"]["routes"]["route.commute"].update(
                distance_m=1.5
            ),
            "floating-point",
        ),
        (
            lambda payload: payload["initial_state"]["calendar"]["event.standup"].update(
                end_at="2026-08-04T09:00:00+05:30"
            ),
            "after start_at",
        ),
    ],
)
def test_program_fails_closed_on_noncanonical_worlds(mutation: object, match: str) -> None:
    payload = program_payload()
    mutation(payload)
    with pytest.raises(SimProgramError, match=match):
        parse_sim_program(payload)


def test_strict_json_rejects_duplicate_keys_floats_nonobjects_and_oversize() -> None:
    with pytest.raises(SimProgramError, match="duplicate JSON key"):
        loads_strict_json_object('{"a":1,"a":2}')
    with pytest.raises(SimProgramError, match="floating-point"):
        loads_strict_json_object('{"a":1.25}')
    for nonfinite in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(SimProgramError, match="non-finite"):
            loads_strict_json_object('{"a":' + nonfinite + "}")
    with pytest.raises(SimProgramError, match="duplicate object key after NFC"):
        loads_strict_json_object('{"\u00e9":1,"e\u0301":2}')
    with pytest.raises(SimProgramError, match="valid UTF-8"):
        loads_strict_json_object('{"a":"\ud800"}')
    with pytest.raises(SimProgramError, match="top-level"):
        loads_strict_json_object("[]")
    with pytest.raises(SimProgramError, match="exceeds"):
        loads_strict_json_object('{"a":"' + ("x" * MAX_SIM_JSON_BYTES) + '"}')
    with pytest.raises(SimProgramError, match="magnitude"):
        snapshot_strict_json({"value": MAX_SIM_INTEGER_ABS + 1})
    with pytest.raises(SimProgramError) as oversized_integer:
        loads_strict_json_object('{"value":' + ("9" * 4_301) + "}")
    assert oversized_integer.value.code == "integer_out_of_range"
    with pytest.raises(SimProgramError, match="string exceeds"):
        snapshot_strict_json({"value": "x" * (MAX_SIM_STRING_BYTES + 1)})
    with pytest.raises(SimProgramError, match="more than"):
        snapshot_strict_json({str(index): index for index in range(MAX_SIM_COLLECTION_ITEMS + 1)})
    deeply_nested: object = "leaf"
    for _ in range(sim_program_module.MAX_SIM_JSON_DEPTH + 1):
        deeply_nested = [deeply_nested]
    with pytest.raises(SimProgramError, match="nesting exceeds"):
        snapshot_strict_json(deeply_nested)


def test_snapshot_rejects_nested_mutation_instead_of_binding_a_hybrid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    resume = threading.Event()
    original_nfc = sim_program_module._nfc
    blocked = False

    def synchronized_nfc(value: str, *, path: str) -> str:
        nonlocal blocked
        normalized = original_nfc(value, path=path)
        if value == "snapshot-barrier" and not blocked:
            blocked = True
            entered.set()
            assert resume.wait(timeout=5)
        return normalized

    monkeypatch.setattr(sim_program_module, "_nfc", synchronized_nfc)
    payload = {"a": [0], "barrier": "snapshot-barrier", "b": [0]}
    outcome: dict[str, object] = {}

    def collect() -> None:
        try:
            outcome["snapshot"] = snapshot_strict_json(payload)
        except Exception as exc:  # noqa: BLE001 - the thread must return its exact failure.
            outcome["error"] = exc

    worker = threading.Thread(target=collect)
    worker.start()
    assert entered.wait(timeout=5)
    payload["a"][0] = 1
    payload["b"][0] = 1
    resume.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert "snapshot" not in outcome
    assert isinstance(outcome.get("error"), SimProgramError)
    assert outcome["error"].code == "concurrent_mutation"


class _CustomMapping(Mapping[str, object]):
    def __init__(self) -> None:
        self._value = {"a": 1}

    def __getitem__(self, key: str) -> object:
        return self._value[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)


def test_custom_containers_are_rejected_instead_of_trusted() -> None:
    with pytest.raises(SimProgramError, match="unsupported strict JSON"):
        snapshot_strict_json(_CustomMapping())
    with pytest.raises(SimProgramError, match="unsupported strict JSON"):
        snapshot_strict_json({"value": range(3)})


def test_timestamps_canonicalize_negative_zero_and_runtime_identity_is_bound() -> None:
    plus_payload = world_payload()
    plus_payload.update(timezone="UTC", reference_time="2026-08-04T04:00:00+00:00")
    plus_payload["reminders"] = {}
    plus_payload["calendar"] = {}
    minus_payload = copy.deepcopy(plus_payload)
    minus_payload["reference_time"] = "2026-08-04T04:00:00-00:00"

    plus = WorldState.from_dict(plus_payload)
    minus = WorldState.from_dict(minus_payload)
    identity = timezone_runtime_identity("UTC")

    assert plus.reference_time == minus.reference_time == "2026-08-04T04:00:00+00:00"
    assert plus.sha256() == minus.sha256()
    assert identity["schema_version"] == TIMEZONE_RUNTIME_VERSION
    assert identity["timezone"] == "UTC"
    assert len(identity["tzif_sha256"]) == 64
    assert identity["tzif_bytes"] > 0
    assert identity["source_kind"] in {"system_tzpath", "tzdata_package"}
    assert timezone_runtime_sha256("UTC") == timezone_runtime_sha256("UTC")


def test_timezone_identity_hashes_complete_tzif_bytes_used_for_semantics() -> None:
    utc_bytes, _, _ = sim_program_module._read_tzif_bytes("UTC")
    kolkata_bytes, _, _ = sim_program_module._read_tzif_bytes("Asia/Kolkata")
    utc_identity = timezone_runtime_identity("UTC")
    kolkata_identity = timezone_runtime_identity("Asia/Kolkata")

    assert utc_identity["tzif_bytes"] == len(utc_bytes)
    assert kolkata_identity["tzif_bytes"] == len(kolkata_bytes)
    assert utc_identity["tzif_sha256"] == hashlib.sha256(utc_bytes).hexdigest()
    assert kolkata_identity["tzif_sha256"] == hashlib.sha256(kolkata_bytes).hexdigest()
    assert utc_identity["tzif_sha256"] != kolkata_identity["tzif_sha256"]
    assert sim_program_module.canonical_sha256(utc_identity) != sim_program_module.canonical_sha256(
        kolkata_identity
    )


def test_timezone_key_errors_and_live_rule_reader_replacement_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SimProgramError, match="invalid IANA timezone key"):
        timezone_runtime_identity("A\x00B")

    baseline_runtime = sim_program_module.sim_program_runtime_sha256()
    with monkeypatch.context() as patcher:
        patcher.setattr(
            sim_program_module,
            "_read_tzif_bytes",
            lambda _name: (b"TZif-forged", "synthetic", None),
        )
        assert sim_program_module.sim_program_runtime_sha256() != baseline_runtime
        with pytest.raises(SimProgramError, match="runtime differs"):
            timezone_runtime_identity("UTC")
    assert sim_program_module.sim_program_runtime_sha256() == baseline_runtime


def test_world_state_rejects_dangling_references_bad_channels_and_bad_media() -> None:
    dangling = world_payload()
    dangling["routes"]["route.commute"]["origin_id"] = "place.unknown"
    with pytest.raises(SimProgramError, match="route place"):
        WorldState.from_dict(dangling)

    bad_channel = copy.deepcopy(world_payload())
    bad_channel["outbox"] = [
        {
            "message_id": "message.one",
            "to_contact_id": "contact.asha",
            "channel": "email",
            "body": "Hi",
            "sent_at": "2026-08-04T09:30:00+05:30",
        }
    ]
    with pytest.raises(SimProgramError, match="not registered"):
        WorldState.from_dict(bad_channel)

    bad_media = copy.deepcopy(world_payload())
    bad_media["media"].update(status="playing", track_id=None)
    with pytest.raises(SimProgramError, match="require a track"):
        WorldState.from_dict(bad_media)


def test_canonical_program_json_is_stable_under_input_key_order() -> None:
    first = parse_sim_program(program_payload())
    reordered = json.loads(json.dumps(program_payload()))
    reordered["initial_state"] = dict(reversed(list(reordered["initial_state"].items())))
    second = parse_sim_program(reordered)

    assert first.canonical_json() == second.canonical_json()
    assert first.sha256() == second.sha256()
