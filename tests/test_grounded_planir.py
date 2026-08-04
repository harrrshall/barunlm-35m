from __future__ import annotations

import calendar
import copy
import importlib.util
import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import barunlm.evaluation.grounded_planir as grounded_planir_module
from barunlm.evaluation.action_ir import JSONType, ToolSchema, ValueSchema
from barunlm.evaluation.grounded_planir import (
    MOBILE_TOOL_SCHEMAS,
    SCHEMA_VERSION,
    CompileFailure,
    CompileResult,
    GroundedPlanIRError,
    compile_mobile_plan,
    compile_mobile_plan_or_raise,
    resolve_span_ref,
)


def _ref(quote: str, occurrence: int = 0) -> dict[str, object]:
    return {"occurrence": occurrence, "quote": quote}


def _calendar_plan(
    date_expr: Mapping[str, object],
    clock: str,
    *,
    title: str = "Team Sync",
    decision: str = "CALL",
) -> str:
    return json.dumps(
        {
            "calls": [
                {
                    "args": {
                        "datetime": {"date": date_expr, "time": {"ref": _ref(clock)}},
                        "title": title,
                    },
                    "tool": "create_calendar_event",
                }
            ],
            "decision": decision,
            "mode": "SINGLE",
            "schema_version": SCHEMA_VERSION,
        },
        separators=(",", ":"),
    )


def _date(op: str, quote: str, **extra: object) -> dict[str, object]:
    return {"op": op, "ref": _ref(quote), **extra}


def _compile_calendar(
    request: str,
    date_expr: Mapping[str, object],
    clock: str,
    *,
    now: str = "2026-08-03T16:30:00",
) -> CompileResult:
    return compile_mobile_plan(_calendar_plan(date_expr, clock), request, now)


def _failure(result: CompileResult) -> CompileFailure:
    assert not result.ok
    assert result.action_ir is None
    assert result.error is not None
    return result.error


def _replace_path(root: dict[str, Any], path: tuple[str | int, ...], value: Any) -> dict[str, Any]:
    mutated = copy.deepcopy(root)
    cursor: Any = mutated
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = copy.deepcopy(value)
    return mutated


def _schema_with(name: str, replacement: ToolSchema) -> tuple[ToolSchema, ...]:
    return tuple(replacement if schema.name == name else schema for schema in MOBILE_TOOL_SCHEMAS)


def test_span_ref_is_nfc_exact_case_sensitive_zero_based_and_nonoverlapping() -> None:
    request = "Cafe\u0301 and Café; aaa"
    assert resolve_span_ref(request, _ref("Café", 0)) == "Café"
    assert resolve_span_ref(request, _ref("Café", 1)) == "Café"
    assert resolve_span_ref(request, _ref("aa", 0)) == "aa"

    with pytest.raises(GroundedPlanIRError) as case_mismatch:
        resolve_span_ref(request, _ref("café"))
    assert case_mismatch.value.code == "reference_not_found"

    with pytest.raises(GroundedPlanIRError) as overlap:
        resolve_span_ref(request, _ref("aa", 1))
    assert overlap.value.code == "reference_occurrence_out_of_range"


def test_english_date_names_are_independent_of_localized_calendar_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load a fresh module while stdlib calendar exposes deliberately poisoned names."""

    monkeypatch.setattr(calendar, "month_name", ("",) + tuple(f"M{i}" for i in range(1, 13)))
    monkeypatch.setattr(calendar, "day_name", tuple(f"D{i}" for i in range(7)))
    module_name = "barunlm.evaluation._grounded_planir_locale_probe"
    module_path = Path(grounded_planir_module.__file__)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = probe
    try:
        spec.loader.exec_module(probe)
        august = probe.compile_mobile_plan(
            _calendar_plan(_date("ABSOLUTE_DATE", "August 4th, 2026"), "noon"),
            "Meet August 4th, 2026 at noon",
            "2026-08-03T00:00:00",
        )
        monday = probe.compile_mobile_plan(
            _calendar_plan(_date("WEEKDAY", "Monday", ordinal=1), "noon"),
            "Meet Monday at noon",
            "2026-08-03T00:00:00",
        )
    finally:
        sys.modules.pop(module_name, None)

    assert august.ok and '"datetime":"2026-08-04T12:00:00"' in (august.action_ir or "")
    assert monday.ok and '"datetime":"2026-08-10T12:00:00"' in (monday.action_ir or "")


@pytest.mark.parametrize(
    ("user_request", "date_quote", "expected"),
    [
        ("Meet 2026-08-04 at noon", "2026-08-04", "2026-08-04T12:00:00"),
        ("Meet August 4th, 2026 at noon", "August 4th, 2026", "2026-08-04T12:00:00"),
        ("Meet 4th of August, 2026 at noon", "4th of August, 2026", "2026-08-04T12:00:00"),
        ("Meet 4 August, 2026 at noon", "4 August, 2026", "2026-08-04T12:00:00"),
    ],
)
def test_absolute_date_grammars(user_request: str, date_quote: str, expected: str) -> None:
    result = _compile_calendar(user_request, _date("ABSOLUTE_DATE", date_quote), "noon")
    assert result.ok
    assert result.action_ir is not None
    assert f'"datetime":"{expected}"' in result.action_ir


def test_month_day_next_is_on_or_after_now_and_skips_to_next_leap_day() -> None:
    same_day = _compile_calendar(
        "Meet August 3 at 5 PM",
        _date("MONTH_DAY_NEXT", "August 3"),
        "5 PM",
    )
    next_year = _compile_calendar(
        "Meet August 2 at 5 PM",
        _date("MONTH_DAY_NEXT", "August 2"),
        "5 PM",
    )
    leap = _compile_calendar(
        "Meet February 29 at noon",
        _date("MONTH_DAY_NEXT", "February 29"),
        "noon",
        now="2025-03-01T00:00:00",
    )

    assert same_day.ok and '"datetime":"2026-08-03T17:00:00"' in (same_day.action_ir or "")
    assert next_year.ok and '"datetime":"2027-08-02T17:00:00"' in (next_year.action_ir or "")
    assert leap.ok and '"datetime":"2028-02-29T12:00:00"' in (leap.action_ir or "")


@pytest.mark.parametrize(
    ("phrase", "expected_date"),
    [
        ("today", "2026-08-03"),
        ("tonight", "2026-08-03"),
        ("this morning", "2026-08-03"),
        ("this afternoon", "2026-08-03"),
        ("this evening", "2026-08-03"),
        ("tomorrow", "2026-08-04"),
        ("day after tomorrow", "2026-08-05"),
        ("in 366 days", "2027-08-04"),
        ("2 days from now", "2026-08-05"),
    ],
)
def test_relative_day_grammar_and_inclusive_bound(phrase: str, expected_date: str) -> None:
    result = _compile_calendar(f"Meet {phrase} at 5 PM", _date("RELATIVE_DAY", phrase), "5 PM")
    assert result.ok
    assert f'"datetime":"{expected_date}T17:00:00"' in (result.action_ir or "")


def test_relative_day_outside_bound_and_impossible_dates_fail_closed() -> None:
    bounded = _compile_calendar(
        "Meet in 367 days at noon", _date("RELATIVE_DAY", "in 367 days"), "noon"
    )
    impossible = _compile_calendar(
        "Meet February 30, 2027 at noon",
        _date("ABSOLUTE_DATE", "February 30, 2027"),
        "noon",
    )
    bad_ordinal = _compile_calendar(
        "Meet August 2th, 2026 at noon",
        _date("ABSOLUTE_DATE", "August 2th, 2026"),
        "noon",
    )
    enormous_offset = "9" * 5_000
    enormous = _compile_calendar(
        f"Meet in {enormous_offset} days at noon",
        _date("RELATIVE_DAY", f"in {enormous_offset} days"),
        "noon",
    )

    assert _failure(bounded).code == "invalid_calendar_date"
    assert _failure(impossible).code == "invalid_calendar_date"
    assert _failure(bad_ordinal).code == "invalid_calendar_date"
    assert _failure(enormous).code == "invalid_calendar_date"


def test_weekday_ordinals_are_strictly_future_including_same_weekday() -> None:
    first = _compile_calendar("Meet Monday at noon", _date("WEEKDAY", "Monday", ordinal=1), "noon")
    second = _compile_calendar("Meet Monday at noon", _date("WEEKDAY", "Monday", ordinal=2), "noon")

    assert first.ok and '"datetime":"2026-08-10T12:00:00"' in (first.action_ir or "")
    assert second.ok and '"datetime":"2026-08-17T12:00:00"' in (second.action_ir or "")


@pytest.mark.parametrize(
    ("clock", "expected"),
    [
        ("9 AM", "09:00:00"),
        ("9:07:05 p.m.", "21:07:05"),
        ("23:59", "23:59:00"),
        ("00:00:09", "00:00:09"),
        ("nine in the morning", "09:00:00"),
        ("ten thirty in the morning", "10:30:00"),
        ("4 in the afternoon", "16:00:00"),
        ("7 in the evening", "19:00:00"),
        ("8 in the night", "20:00:00"),
        ("noon", "12:00:00"),
        ("midnight", "00:00:00"),
    ],
)
def test_clock_grammar(clock: str, expected: str) -> None:
    result = _compile_calendar(
        f"Meet tomorrow at {clock}", _date("RELATIVE_DAY", "tomorrow"), clock
    )
    assert result.ok
    assert f'"datetime":"2026-08-04T{expected}"' in (result.action_ir or "")


def test_ambiguous_invalid_mismatched_and_past_temporal_values_are_distinct_failures() -> None:
    ambiguous = _compile_calendar(
        "Meet tomorrow at 4 PM or 5 PM",
        _date("RELATIVE_DAY", "tomorrow"),
        "4 PM or 5 PM",
    )
    invalid_clock = _compile_calendar(
        "Meet tomorrow at 25:00", _date("RELATIVE_DAY", "tomorrow"), "25:00"
    )
    mismatched = _compile_calendar(
        "Meet tomorrow at noon", _date("ABSOLUTE_DATE", "tomorrow"), "noon"
    )
    past = _compile_calendar("Meet today at 4 PM", _date("RELATIVE_DAY", "today"), "4 PM")

    assert _failure(ambiguous).code == "ambiguous_temporal_reference"
    assert _failure(invalid_clock).code == "invalid_clock_time"
    assert _failure(mismatched).code == "temporal_reference_mismatch"
    assert _failure(past).code == "past_datetime"


def test_all_seven_mobile_tools_compile_and_noncalendar_strings_are_literal() -> None:
    plan = {
        "calls": [
            {
                "args": {
                    "datetime": {
                        "date": _date("RELATIVE_DAY", "tomorrow"),
                        "time": {"ref": _ref("9 AM")},
                    },
                    "title": "Literal title not required to occur in request",
                },
                "tool": "create_calendar_event",
            },
            {
                "args": {
                    "email": "asha@example.com",
                    "first_name": "Asha",
                    "last_name": "Rao",
                    "phone_number": "+91 555 0100",
                },
                "tool": "create_contact",
            },
            {"args": {}, "tool": "open_wifi_settings"},
            {
                "args": {"body": "Body", "subject": "Subject", "to": "li@example.com"},
                "tool": "send_email",
            },
            {"args": {"query": "coffee"}, "tool": "show_map"},
            {"args": {}, "tool": "turn_off_flashlight"},
            {"args": {}, "tool": "turn_on_flashlight"},
        ],
        "decision": "CALL",
        "mode": "SERIAL",
        "schema_version": SCHEMA_VERSION,
    }
    result = compile_mobile_plan(
        json.dumps(plan), "Do these tomorrow at 9 AM", "2026-08-03T16:30:00"
    )

    assert result.ok
    assert result.action_ir is not None
    assert "grounded-plan-ir-v1" not in result.action_ir
    assert "Literal title not required to occur in request" in result.action_ir
    assert [schema.name for schema in MOBILE_TOOL_SCHEMAS] == [
        "create_calendar_event",
        "create_contact",
        "open_wifi_settings",
        "send_email",
        "show_map",
        "turn_off_flashlight",
        "turn_on_flashlight",
    ]


def test_frozen_mobile_schema_identity_accepts_only_order_variation() -> None:
    abstain = json.dumps({"decision": "ABSTAIN", "schema_version": SCHEMA_VERSION})
    reversed_schemas = tuple(reversed(MOBILE_TOOL_SCHEMAS))
    result = compile_mobile_plan(
        abstain,
        "request",
        "2026-08-03T00:00:00",
        schemas=reversed_schemas,
    )
    mapping_result = compile_mobile_plan(
        abstain,
        "request",
        "2026-08-03T00:00:00",
        schemas={schema.name: schema for schema in reversed_schemas},
    )

    assert result.ok
    assert result.action_ir == '{"decision":"ABSTAIN"}'
    assert mapping_result.ok
    assert mapping_result.action_ir == result.action_ir
    assert tuple((schema.name, schema.side_effecting) for schema in MOBILE_TOOL_SCHEMAS) == (
        ("create_calendar_event", True),
        ("create_contact", True),
        ("open_wifi_settings", True),
        ("send_email", True),
        ("show_map", True),
        ("turn_off_flashlight", True),
        ("turn_on_flashlight", True),
    )


def test_frozen_mobile_schema_identity_rejects_every_semantic_drift() -> None:
    by_name = {schema.name: schema for schema in MOBILE_TOOL_SCHEMAS}
    show_map = by_name["show_map"]
    send_email = by_name["send_email"]
    schema_drifts: list[tuple[ToolSchema, ...]] = [
        MOBILE_TOOL_SCHEMAS[:-1],
        (*MOBILE_TOOL_SCHEMAS[:-1], MOBILE_TOOL_SCHEMAS[0]),
        MOBILE_TOOL_SCHEMAS + (ToolSchema("unexpected_tool", {}),),
        _schema_with(
            "show_map",
            replace(show_map, arguments={"query": ValueSchema(JSONType.INTEGER)}),
        ),
        _schema_with(
            "show_map",
            replace(
                show_map,
                arguments={
                    "query": ValueSchema(JSONType.STRING),
                    "radius": ValueSchema(JSONType.INTEGER),
                },
            ),
        ),
        _schema_with("send_email", replace(send_email, required=frozenset({"subject"}))),
        _schema_with("show_map", replace(show_map, additional_arguments=True)),
        _schema_with("show_map", replace(show_map, side_effecting=False)),
    ]
    abstain = json.dumps({"decision": "ABSTAIN", "schema_version": SCHEMA_VERSION})

    for schemas in schema_drifts:
        failure = _failure(
            compile_mobile_plan(
                abstain,
                "request",
                "2026-08-03T00:00:00",
                schemas=schemas,
            )
        )
        assert failure.code == "final_action_ir_validation_failure"
        assert failure.path.startswith("$.schemas")


def test_frozen_mobile_schema_identity_rejects_mapping_key_and_non_schema_values() -> None:
    abstain = json.dumps({"decision": "ABSTAIN", "schema_version": SCHEMA_VERSION})
    keyed = {schema.name: schema for schema in MOBILE_TOOL_SCHEMAS}
    keyed["wrong-key"] = keyed.pop("show_map")

    key_failure = _failure(
        compile_mobile_plan(
            abstain,
            "request",
            "2026-08-03T00:00:00",
            schemas=keyed,
        )
    )
    value_failure = _failure(
        compile_mobile_plan(
            abstain,
            "request",
            "2026-08-03T00:00:00",
            schemas=(*MOBILE_TOOL_SCHEMAS[:-1], object()),  # type: ignore[arg-type]
        )
    )

    assert key_failure.code == "final_action_ir_validation_failure"
    assert value_failure.code == "final_action_ir_validation_failure"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"raw": "not json"}, "invalid_plan_json"),
        ({"plan": {"decision": "ABSTAIN"}}, "missing_field"),
        (
            {"plan": {"decision": "ABSTAIN", "schema_version": "wrong"}},
            "schema_version_mismatch",
        ),
        (
            {
                "plan": {
                    "decision": "ABSTAIN",
                    "extra": True,
                    "schema_version": SCHEMA_VERSION,
                }
            },
            "unknown_field",
        ),
    ],
)
def test_strict_json_and_top_level_shape_failures(mutation: Mapping[str, Any], code: str) -> None:
    raw = mutation.get("raw") or json.dumps(mutation["plan"])
    assert _failure(compile_mobile_plan(raw, "request", "2026-08-03T00:00:00")).code == code


@pytest.mark.parametrize(
    "raw",
    [
        ('{"schema_version":"grounded-plan-ir-v1","decision":"ABSTAIN","decision":"CALL"}'),
        '```json\n{"schema_version":"grounded-plan-ir-v1","decision":"ABSTAIN"}\n```',
        'prose {"schema_version":"grounded-plan-ir-v1","decision":"ABSTAIN"}',
        ('{"schema_version":"grounded-plan-ir-v1","decision":"ABSTAIN"}{"decision":"ABSTAIN"}'),
        '{"schema_version":"grounded-plan-ir-v1","decision":"ABSTAIN","x":NaN}',
        '{"schema_version":"grounded-plan-ir-v1","decision":"ABSTAIN","x":Infinity}',
        "[]",
        "null",
        "true",
        "7",
        '{"schema_version":"grounded-plan-ir-v1","decision":"ABSTAIN","x":'
        + "[" * 900
        + "0"
        + "]" * 900
        + "}",
    ],
    ids=[
        "duplicate-key",
        "fence",
        "prose",
        "trailing-value",
        "nan",
        "infinity",
        "array-top",
        "null-top",
        "bool-top",
        "number-top",
        "excessive-nesting",
    ],
)
def test_adversarial_json_is_one_structured_failure(raw: str) -> None:
    result = compile_mobile_plan(raw, "request", "2026-08-03T00:00:00")
    assert _failure(result).code == "invalid_plan_json"


_RECURSIVE_WRONG_JSON_VALUES = (
    {"nested": {"value": [None, False, 3]}},
    [{"nested": [None, False, 3]}],
    None,
    False,
    3.5,
)
_CALL_SHAPE_PATHS: tuple[tuple[str | int, ...], ...] = (
    ("schema_version",),
    ("decision",),
    ("mode",),
    ("calls",),
    ("calls", 0),
    ("calls", 0, "tool"),
    ("calls", 0, "args"),
    ("calls", 0, "args", "title"),
    ("calls", 0, "args", "datetime"),
    ("calls", 0, "args", "datetime", "date"),
    ("calls", 0, "args", "datetime", "date", "op"),
    ("calls", 0, "args", "datetime", "date", "ref"),
    ("calls", 0, "args", "datetime", "date", "ref", "quote"),
    ("calls", 0, "args", "datetime", "date", "ref", "occurrence"),
    ("calls", 0, "args", "datetime", "time"),
    ("calls", 0, "args", "datetime", "time", "ref"),
    ("calls", 0, "args", "datetime", "time", "ref", "quote"),
    ("calls", 0, "args", "datetime", "time", "ref", "occurrence"),
)


@pytest.mark.parametrize("path", _CALL_SHAPE_PATHS, ids=lambda path: ".".join(map(str, path)))
@pytest.mark.parametrize(
    "bad_value",
    _RECURSIVE_WRONG_JSON_VALUES,
    ids=("object", "array", "null", "boolean", "number"),
)
def test_every_call_discriminator_and_shape_field_is_total_for_all_json_kinds(
    path: tuple[str | int, ...], bad_value: Any
) -> None:
    base = json.loads(_calendar_plan(_date("RELATIVE_DAY", "tomorrow"), "noon"))
    mutated = _replace_path(base, path, bad_value)
    result = compile_mobile_plan(
        json.dumps(mutated), "Meet tomorrow at noon", "2026-08-03T00:00:00"
    )

    assert not result.ok
    assert result.action_ir is None
    assert result.error is not None


@pytest.mark.parametrize(
    "bad_value",
    _RECURSIVE_WRONG_JSON_VALUES,
    ids=("object", "array", "null", "boolean", "number"),
)
def test_clarify_and_weekday_specific_shape_fields_are_total_for_all_json_kinds(
    bad_value: Any,
) -> None:
    clarify_missing = {
        "decision": "CLARIFY",
        "missing": copy.deepcopy(bad_value),
        "schema_version": SCHEMA_VERSION,
    }
    clarify_item = {
        "decision": "CLARIFY",
        "missing": [copy.deepcopy(bad_value)],
        "schema_version": SCHEMA_VERSION,
    }
    weekday = json.loads(_calendar_plan(_date("WEEKDAY", "Monday", ordinal=1), "noon"))
    weekday["calls"][0]["args"]["datetime"]["date"]["ordinal"] = copy.deepcopy(bad_value)

    results = (
        compile_mobile_plan(json.dumps(clarify_missing), "request", "2026-08-03T00:00:00"),
        compile_mobile_plan(json.dumps(clarify_item), "request", "2026-08-03T00:00:00"),
        compile_mobile_plan(json.dumps(weekday), "Meet Monday at noon", "2026-08-03T00:00:00"),
    )
    assert all(not result.ok and result.error is not None for result in results)


def test_tool_argument_mode_reference_context_and_final_schema_failures() -> None:
    base = json.loads(_calendar_plan(_date("RELATIVE_DAY", "tomorrow"), "noon"))

    unknown = json.loads(json.dumps(base))
    unknown["calls"][0]["tool"] = "delete_everything"
    missing = json.loads(json.dumps(base))
    del missing["calls"][0]["args"]["title"]
    extra = json.loads(json.dumps(base))
    extra["calls"][0]["args"]["location"] = "office"
    count = json.loads(json.dumps(base))
    count["mode"] = "SERIAL"
    absent_ref = json.loads(json.dumps(base))
    absent_ref["calls"][0]["args"]["datetime"]["date"]["ref"]["quote"] = "next week"

    request = "Meet tomorrow at noon"
    now = "2026-08-03T00:00:00"
    assert _failure(compile_mobile_plan(json.dumps(unknown), request, now)).code == "unknown_tool"
    assert (
        _failure(compile_mobile_plan(json.dumps(missing), request, now)).code == "missing_argument"
    )
    assert _failure(compile_mobile_plan(json.dumps(extra), request, now)).code == "extra_argument"
    assert (
        _failure(compile_mobile_plan(json.dumps(count), request, now)).code == "invalid_call_count"
    )
    assert (
        _failure(compile_mobile_plan(json.dumps(absent_ref), request, now)).code
        == "reference_not_found"
    )
    assert (
        _failure(compile_mobile_plan(json.dumps(base), request, now, context={"x": 1})).code
        == "unsupported_context"
    )
    assert (
        _failure(compile_mobile_plan(json.dumps(base), request, now, schemas=())).code
        == "final_action_ir_validation_failure"
    )


def test_no_call_shapes_use_action_ir_validation_and_public_api_never_raises() -> None:
    abstain = json.dumps({"decision": "ABSTAIN", "schema_version": SCHEMA_VERSION})
    clarify = json.dumps(
        {"decision": "CLARIFY", "missing": ["datetime"], "schema_version": SCHEMA_VERSION}
    )
    bad_clarify = json.dumps(
        {
            "decision": "CLARIFY",
            "missing": ["who", "datetime"],
            "schema_version": SCHEMA_VERSION,
        }
    )

    assert compile_mobile_plan(abstain, "request", "2026-08-03T00:00:00").action_ir == (
        '{"decision":"ABSTAIN"}'
    )
    assert compile_mobile_plan(clarify, "request", "2026-08-03T00:00:00").action_ir == (
        '{"decision":"CLARIFY","missing":["datetime"]}'
    )
    assert (
        _failure(compile_mobile_plan(bad_clarify, "request", "2026-08-03T00:00:00")).code
        == "final_action_ir_validation_failure"
    )

    with pytest.raises(GroundedPlanIRError):
        compile_mobile_plan_or_raise("invalid", "request", "2026-08-03T00:00:00")


def test_compile_result_enforces_exclusive_success_or_failure_state() -> None:
    failure = CompileFailure("code", "message", "$")
    with pytest.raises(ValueError):
        CompileResult(ok=True, action_ir=None)
    with pytest.raises(ValueError):
        CompileResult(ok=True, action_ir="{}", error=failure)
    with pytest.raises(ValueError):
        CompileResult(ok=False, error=None)
    with pytest.raises(ValueError):
        CompileResult(ok=False, action_ir="{}", error=failure)
