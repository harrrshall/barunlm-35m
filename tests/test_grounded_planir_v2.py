from __future__ import annotations

import inspect
import json
from collections.abc import Iterator, Mapping
from dataclasses import replace

import pytest

import barunlm.evaluation.grounded_planir_v2 as grounded_planir_v2_module
from barunlm.evaluation.action_ir import JSONType, ValueSchema
from barunlm.evaluation.grounded_planir import (
    MOBILE_TOOL_SCHEMAS,
    SCHEMA_VERSION,
    CompileFailure,
)
from barunlm.evaluation.grounded_planir import (
    compile_mobile_plan as compile_mobile_plan_v1,
)
from barunlm.evaluation.grounded_planir_v2 import (
    PROMPT_CONTRACT,
    GroundedPlanIRV2Error,
    PromptEvidence,
    build_reference_table,
    compile_mobile_plan,
    enumerate_grounding_candidates,
    make_prompt_evidence,
    render_construction_prompt,
    verify_reference_table,
)


def _plan(placeholder: object, title: str = "Team Sync") -> str:
    return json.dumps(
        {
            "calls": [
                {
                    "args": {"datetime": placeholder, "title": title},
                    "tool": "create_calendar_event",
                }
            ],
            "decision": "CALL",
            "mode": "SINGLE",
        },
        separators=(",", ":"),
    )


def _prompt_binding(request: str, now: str, table, contract: str = PROMPT_CONTRACT):
    prompt = render_construction_prompt(
        prompt_contract=contract,
        system_body=f"NOW {now}\nTOOLS\n",
        request=request,
        rendered_table=table.render(),
    )
    return prompt, make_prompt_evidence(prompt, contract, table, request, now)


def _compile(raw: object, request: object, now: object):
    if isinstance(request, str) and isinstance(now, str):
        try:
            table = build_reference_table(request, now)
        except GroundedPlanIRV2Error:
            rendered = None
            digest = "0" * 64
        else:
            rendered = table.render()
            digest = table.sha256()
    else:
        rendered = None
        digest = "0" * 64
    if isinstance(request, str) and isinstance(now, str):
        try:
            rendered_prompt, evidence = _prompt_binding(request, now, table)
        except (GroundedPlanIRV2Error, UnboundLocalError):
            rendered_prompt = "invalid"
            evidence = PromptEvidence(
                renderer_version="invalid",
                prompt_contract=PROMPT_CONTRACT,
                prompt_sha256="0" * 64,
                request_sha256="0" * 64,
                now=now,
                table_sha256=digest,
            )
    else:
        rendered_prompt = "invalid"
        evidence = PromptEvidence(
            renderer_version="invalid",
            prompt_contract=PROMPT_CONTRACT,
            prompt_sha256="0" * 64,
            request_sha256="0" * 64,
            now=str(now),
            table_sha256=digest,
        )
    return compile_mobile_plan(  # type: ignore[arg-type]
        raw,
        request,
        now,
        prompt_contract=PROMPT_CONTRACT,
        rendered_prompt=rendered_prompt,
        prompt_evidence=evidence,
        rendered_table=rendered,
        table_sha256=digest,
    )


def _failure(result) -> CompileFailure:
    assert not result.ok and result.action_ir is None and result.error is not None
    return result.error


def test_enumerator_is_input_only_and_prompt_contract_is_compact() -> None:
    assert tuple(inspect.signature(enumerate_grounding_candidates).parameters) == (
        "request",
        "now",
    )
    assert PROMPT_CONTRACT == "PLAN_IR_V2"


def test_reference_table_render_hash_overlap_occurrence_and_distractors() -> None:
    request = "Meet August 20th, 2026 or August 20th next Tuesday at 2 PM or 14:00."
    table = build_reference_table(request, "2026-08-17T09:00:00")

    assert table.render() == (
        'REFS_V2 {"D":[["August 20th",0],["August 20th, 2026",0],'
        '["August 20th",1],["Tuesday",0]],"T":[["2 PM",0],["14:00",0]]}'
    )
    assert table.sha256() == "fdfad7c46b4b6373ff9e8d80a3866b457945f2c6d5cd6e4fa01b3f335e4af7b8"
    assert {(item.opcode, item.ref.quote) for item in table.candidates.dates} >= {
        ("A", "August 20th, 2026"),
        ("M", "August 20th"),
        ("W1", "Tuesday"),
        ("W2", "Tuesday"),
    }
    assert {item.ref.quote for item in table.candidates.clocks} == {"2 PM", "14:00"}


def test_empty_table_is_omitted_but_still_has_a_bound_digest() -> None:
    table = build_reference_table("Turn off flashlight", "2026-08-17T09:00:00")
    assert table.render() is None
    assert table.canonical_json() == '{"D":[],"T":[]}'
    assert table.sha256() == "fa5b923931a3dce104980358bcb9de4f69fa1c567a124cb43798d1978eae204f"
    result = _compile('{"decision":"ABSTAIN"}', "Turn off flashlight", "2026-08-17T09:00:00")
    assert result.ok and result.action_ir == '{"decision":"ABSTAIN"}'


def test_exact_render_and_hash_are_both_verified() -> None:
    request = "Meet tomorrow at noon"
    now = "2026-08-17T09:00:00"
    table = build_reference_table(request, now)
    with pytest.raises(GroundedPlanIRV2Error, match="rendered table differs"):
        verify_reference_table(request, now, table.render() + " ", table.sha256())  # type: ignore[operator]
    with pytest.raises(GroundedPlanIRV2Error, match="digest differs"):
        verify_reference_table(request, now, table.render(), "0" * 64)
    rendered_prompt, evidence = _prompt_binding(request, now, table)

    render_failure = compile_mobile_plan(
        _plan("@R:D00:T00"),
        request,
        now,
        prompt_contract=PROMPT_CONTRACT,
        rendered_prompt=rendered_prompt,
        prompt_evidence=evidence,
        rendered_table=None,
        table_sha256=table.sha256(),
    )
    hash_failure = compile_mobile_plan(
        _plan("@R:D00:T00"),
        request,
        now,
        prompt_contract=PROMPT_CONTRACT,
        rendered_prompt=rendered_prompt,
        prompt_evidence=evidence,
        rendered_table=table.render(),
        table_sha256="0" * 64,
    )
    assert _failure(render_failure).code == "reference_table_mismatch"
    assert _failure(hash_failure).code == "reference_table_hash_mismatch"


def test_prompt_evidence_binds_exact_bytes_request_now_and_table() -> None:
    request = "Meet tomorrow at noon"
    now = "2026-08-17T09:00:00"
    table = build_reference_table(request, now)
    prompt, evidence = _prompt_binding(request, now, table)
    tampered_hash = replace(evidence, prompt_sha256="0" * 64)
    hash_failure = compile_mobile_plan(
        _plan("@R:D00:T00"),
        request,
        now,
        prompt_contract=PROMPT_CONTRACT,
        rendered_prompt=prompt,
        prompt_evidence=tampered_hash,
        rendered_table=table.render(),
        table_sha256=table.sha256(),
    )
    mixed_now = compile_mobile_plan(
        _plan("@R:D00:T00"),
        request,
        "2026-08-18T09:00:00",
        prompt_contract=PROMPT_CONTRACT,
        rendered_prompt=prompt,
        prompt_evidence=evidence,
        rendered_table=table.render(),
        table_sha256=table.sha256(),
    )
    assert _failure(hash_failure).code == "prompt_hash_mismatch"
    assert _failure(mixed_now).code == "prompt_input_mismatch"
    with pytest.raises(GroundedPlanIRV2Error, match="exact system prefix"):
        make_prompt_evidence("EVIL", PROMPT_CONTRACT, table, request, now)


def test_compiler_requires_the_exact_explicit_prompt_contract() -> None:
    request = "Meet tomorrow at noon"
    now = "2026-08-17T09:00:00"
    table = build_reference_table(request, now)
    rendered_prompt, evidence = _prompt_binding(request, now, table)
    result = compile_mobile_plan(
        _plan("@R:D00:T00"),
        request,
        now,
        prompt_contract="ACTION_IR_V1",
        rendered_prompt=rendered_prompt,
        prompt_evidence=evidence,
        rendered_table=table.render(),
        table_sha256=table.sha256(),
    )
    assert _failure(result).code == "prompt_contract_mismatch"


@pytest.mark.parametrize(
    ("user_request", "now", "placeholder", "expected"),
    [
        (
            "Meet August 20th, 2026 at noon",
            "2026-08-17T09:00:00",
            "@A:D01:T00",
            "2026-08-20T12:00:00",
        ),
        (
            "Meet February 29 at midnight",
            "2025-03-01T00:00:00",
            "@M:D00:T00",
            "2028-02-29T00:00:00",
        ),
        (
            "Meet in 2 days at 8 in the night",
            "2026-08-17T09:00:00",
            "@R:D00:T00",
            "2026-08-19T20:00:00",
        ),
        (
            "Meet 2 days from now at 10:30 AM",
            "2026-08-17T09:00:00",
            "@R:D00:T00",
            "2026-08-19T10:30:00",
        ),
        (
            "Meet next Tuesday at noon",
            "2026-08-17T09:00:00",
            "@W1:D00:T00",
            "2026-08-18T12:00:00",
        ),
        (
            "Meet Tuesday at midnight",
            "2026-08-17T09:00:00",
            "@W2:D00:T00",
            "2026-08-25T00:00:00",
        ),
    ],
)
def test_all_placeholder_opcodes_use_one_shared_grammar(
    user_request: str, now: str, placeholder: str, expected: str
) -> None:
    result = _compile(_plan(placeholder), user_request, now)
    assert result.ok
    assert f'"datetime":"{expected}"' in (result.action_ir or "")


@pytest.mark.parametrize(
    ("placeholder", "code"),
    [
        ("2026-08-18T12:00:00", "invalid_placeholder"),
        ("@R:D0:T00", "noncanonical_reference_id"),
        ("@R:D000:T00", "noncanonical_reference_id"),
        ("@R:D" + "0" * 10_000 + ":T00", "noncanonical_reference_id"),
        ("@R:D٠٠:T00", "noncanonical_reference_id"),
        ("@R:D99:T00", "reference_id_out_of_range"),
        ("@R:D00:T99", "reference_id_out_of_range"),
        ("@R:D00:T00 trailing", "invalid_placeholder"),
        ("@W3:D00:T00", "invalid_placeholder"),
        (None, "invalid_placeholder"),
    ],
)
def test_malformed_noncanonical_and_out_of_range_placeholders_fail(
    placeholder: object, code: str
) -> None:
    result = _compile(_plan(placeholder), "Meet tomorrow at noon", "2026-08-17T09:00:00")
    assert _failure(result).code == code


def test_opcode_reference_compatibility_is_strict_and_does_not_choose_a_distractor() -> None:
    request = "Meet tomorrow or Tuesday at noon or 4 PM"
    now = "2026-08-17T09:00:00"
    table = build_reference_table(request, now)
    assert table.render() == (
        'REFS_V2 {"D":[["tomorrow",0],["Tuesday",0]],"T":[["noon",0],["4 PM",0]]}'
    )
    good = _compile(_plan("@W1:D01:T01"), request, now)
    wrong = _compile(_plan("@A:D00:T00"), request, now)
    assert good.ok and '"datetime":"2026-08-18T16:00:00"' in (good.action_ir or "")
    assert _failure(wrong).code == "incompatible_date_reference"


@pytest.mark.parametrize(
    "ambiguous_clock", ["1 in the night", "six in the night", "12 in the night"]
)
def test_ambiguous_night_hours_and_past_tonight_midnight_fail_closed(
    ambiguous_clock: str,
) -> None:
    request = f"Meet tomorrow at {ambiguous_clock}"
    with pytest.raises(GroundedPlanIRV2Error) as ambiguous:
        build_reference_table(request, "2026-08-17T09:00:00")
    assert ambiguous.value.code == "ambiguous_clock_time"
    quote_v1 = compile_mobile_plan_v1(
        json.dumps(
            {
                "calls": [
                    {
                        "args": {
                            "datetime": {
                                "date": {
                                    "op": "RELATIVE_DAY",
                                    "ref": {"occurrence": 0, "quote": "tomorrow"},
                                },
                                "time": {"ref": {"occurrence": 0, "quote": ambiguous_clock}},
                            },
                            "title": "Meeting",
                        },
                        "tool": "create_calendar_event",
                    }
                ],
                "decision": "CALL",
                "mode": "SINGLE",
                "schema_version": SCHEMA_VERSION,
            }
        ),
        request,
        "2026-08-17T09:00:00",
    )
    assert _failure(quote_v1).code == "ambiguous_temporal_reference"
    past = _compile(
        _plan("@R:D00:T00"),
        "Meet tonight at midnight",
        "2026-08-17T09:00:00",
    )
    assert _failure(past).code == "past_datetime"


@pytest.mark.parametrize(
    "user_request",
    ["Meet last Tuesday at noon", "Meet previous Tuesday at noon", "Meet Tuesday ago at noon"],
)
def test_explicitly_past_weekday_cannot_compile_as_future(user_request: str) -> None:
    now = "2026-08-17T09:00:00"
    table = build_reference_table(user_request, now)
    assert table.payload()["D"] == [["Tuesday", 0]]
    assert table.candidates.date_distractors == (table.date_refs[0],)
    plan_v2 = _compile(_plan("@W1:D00:T00"), user_request, now)
    assert _failure(plan_v2).code == "incompatible_date_reference"

    quote_v1 = compile_mobile_plan_v1(
        json.dumps(
            {
                "calls": [
                    {
                        "args": {
                            "datetime": {
                                "date": {
                                    "op": "WEEKDAY",
                                    "ordinal": 1,
                                    "ref": {"occurrence": 0, "quote": "Tuesday"},
                                },
                                "time": {"ref": {"occurrence": 0, "quote": "noon"}},
                            },
                            "title": "Meeting",
                        },
                        "tool": "create_calendar_event",
                    }
                ],
                "decision": "CALL",
                "mode": "SINGLE",
                "schema_version": SCHEMA_VERSION,
            }
        ),
        user_request,
        now,
    )
    assert _failure(quote_v1).code == "past_datetime"


def test_occurrence_positions_are_cached_once_per_quote(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {}
    original = grounded_planir_v2_module._exact_nonoverlapping_positions

    def counted(request: str, quote: str) -> tuple[int, ...]:
        calls[quote] = calls.get(quote, 0) + 1
        return original(request, quote)

    monkeypatch.setattr(grounded_planir_v2_module, "_exact_nonoverlapping_positions", counted)
    table = build_reference_table(" ".join(["today"] * 100) + " at noon", "2026-08-17T09:00:00")
    assert len(table.date_refs) == 100
    assert calls["today"] == 1


def test_reference_limit_aborts_before_clock_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_clock_enumeration(*_args: object) -> None:
        raise AssertionError("clock enumeration should not run after D overflow")

    monkeypatch.setattr(grounded_planir_v2_module, "_enumerate_clocks", forbidden_clock_enumeration)
    with pytest.raises(GroundedPlanIRV2Error) as failure:
        build_reference_table(
            " ".join(["today"] * 101) + " at 12 in the night",
            "2026-08-17T09:00:00",
        )
    assert failure.value.code == "too_many_references"


def test_d99_is_allowed_but_101_reference_instances_fail_before_compilation() -> None:
    now = "2026-08-17T09:00:00"
    request = " ".join(["today"] * 100) + " at noon"
    table = build_reference_table(request, now)
    assert len(table.date_refs) == 100
    boundary = _compile(_plan("@R:D99:T00"), request, now)
    assert boundary.ok and '"datetime":"2026-08-17T12:00:00"' in (boundary.action_ir or "")

    with pytest.raises(GroundedPlanIRV2Error) as too_many:
        build_reference_table(" ".join(["today"] * 101), now)
    assert too_many.value.code == "too_many_references"
    assert _failure(
        _compile('{"decision":"ABSTAIN"}', "Cafe\u0301", "2026-08-17T09:00:00")
    ).code == ("noncanonical_request")
    assert _failure(_compile('{"decision":"ABSTAIN"}', "request", "2026-8-17T09:00:00")).code == (
        "invalid_now"
    )


@pytest.mark.parametrize(
    "raw",
    [
        None,
        7,
        [],
        {},
        "",
        "not JSON",
        "[]",
        "null",
        '{"decision":"ABSTAIN","decision":"CALL"}',
        '```json\n{"decision":"ABSTAIN"}\n```',
        '{"decision":"ABSTAIN"}{"decision":"ABSTAIN"}',
        '{"decision":"ABSTAIN","x":NaN}',
        "[" * 1000 + "]" * 1000,
    ],
)
def test_public_api_is_total_for_arbitrary_strict_json(raw: object) -> None:
    result = _compile(raw, "request", "2026-08-17T09:00:00")
    assert not result.ok and result.error is not None


def test_final_action_ir_validation_uses_the_frozen_seven_schemas() -> None:
    request = "request"
    now = "2026-08-17T09:00:00"
    table = build_reference_table(request, now)
    rendered_prompt, evidence = _prompt_binding(request, now, table)
    assert [schema.name for schema in MOBILE_TOOL_SCHEMAS] == [
        "create_calendar_event",
        "create_contact",
        "open_wifi_settings",
        "send_email",
        "show_map",
        "turn_off_flashlight",
        "turn_on_flashlight",
    ]
    show_map = next(schema for schema in MOBILE_TOOL_SCHEMAS if schema.name == "show_map")
    drift = tuple(
        replace(schema, arguments={"query": ValueSchema(JSONType.INTEGER)})
        if schema is show_map
        else schema
        for schema in MOBILE_TOOL_SCHEMAS
    )
    result = compile_mobile_plan(
        '{"decision":"ABSTAIN"}',
        request,
        now,
        prompt_contract=PROMPT_CONTRACT,
        rendered_prompt=rendered_prompt,
        prompt_evidence=evidence,
        rendered_table=table.render(),
        table_sha256=table.sha256(),
        schemas=drift,
    )
    assert _failure(result).code == "final_action_ir_validation_failure"


class _ExplodingSchemaMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        raise RuntimeError("hostile mapping")


class _ExplodingContext(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        raise RuntimeError("hostile context")


def _exploding_schema_iterable():
    yield MOBILE_TOOL_SCHEMAS[0]
    raise RuntimeError("hostile iterable")


@pytest.mark.parametrize("schemas", [_ExplodingSchemaMapping(), _exploding_schema_iterable()])
def test_hostile_schema_materialization_is_structured_for_both_compilers(schemas) -> None:
    request = "request"
    now = "2026-08-17T09:00:00"
    table = build_reference_table(request, now)
    prompt, evidence = _prompt_binding(request, now, table)
    v2 = compile_mobile_plan(
        '{"decision":"ABSTAIN"}',
        request,
        now,
        prompt_contract=PROMPT_CONTRACT,
        rendered_prompt=prompt,
        prompt_evidence=evidence,
        rendered_table=table.render(),
        table_sha256=table.sha256(),
        schemas=schemas,
    )
    # Recreate the one-shot generator for the second compiler.
    v1_schemas = (
        _exploding_schema_iterable()
        if not isinstance(schemas, Mapping)
        else _ExplodingSchemaMapping()
    )
    v1 = compile_mobile_plan_v1(
        json.dumps({"decision": "ABSTAIN", "schema_version": SCHEMA_VERSION}),
        request,
        now,
        schemas=v1_schemas,
    )
    assert _failure(v2).code == "final_action_ir_validation_failure"
    assert _failure(v1).code == "final_action_ir_validation_failure"


def test_hostile_context_inspection_is_structured_for_both_compilers() -> None:
    request = "request"
    now = "2026-08-17T09:00:00"
    table = build_reference_table(request, now)
    prompt, evidence = _prompt_binding(request, now, table)
    v2 = compile_mobile_plan(
        '{"decision":"ABSTAIN"}',
        request,
        now,
        prompt_contract=PROMPT_CONTRACT,
        rendered_prompt=prompt,
        prompt_evidence=evidence,
        rendered_table=table.render(),
        table_sha256=table.sha256(),
        context=_ExplodingContext(),
    )
    v1 = compile_mobile_plan_v1(
        json.dumps({"decision": "ABSTAIN", "schema_version": SCHEMA_VERSION}),
        request,
        now,
        context=_ExplodingContext(),
    )
    assert _failure(v2).code == "unsupported_context"
    assert _failure(v1).code == "unsupported_context"


def test_noncalendar_action_ir_is_unchanged_and_context_is_empty_only() -> None:
    request = "Show a map of Paris tomorrow at noon"
    now = "2026-08-17T09:00:00"
    table = build_reference_table(request, now)
    rendered_prompt, evidence = _prompt_binding(request, now, table)
    raw = (
        '{"calls":[{"args":{"query":"Paris"},"tool":"show_map"}],"decision":"CALL","mode":"SINGLE"}'
    )
    result = compile_mobile_plan(
        raw,
        request,
        now,
        prompt_contract=PROMPT_CONTRACT,
        rendered_prompt=rendered_prompt,
        prompt_evidence=evidence,
        rendered_table=table.render(),
        table_sha256=table.sha256(),
    )
    blocked = compile_mobile_plan(
        raw,
        request,
        now,
        prompt_contract=PROMPT_CONTRACT,
        rendered_prompt=rendered_prompt,
        prompt_evidence=evidence,
        rendered_table=table.render(),
        table_sha256=table.sha256(),
        context={"hidden": "state"},
    )
    assert result.ok and result.action_ir == raw
    assert _failure(blocked).code == "unsupported_context"
