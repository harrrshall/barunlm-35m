from __future__ import annotations

import json
from dataclasses import replace

import pytest

import barunlm.evaluation.gvs_schema_presentation as presentation_module
from barunlm.datasets.mobile_actions import ToolArgument, ToolDefinition
from barunlm.evaluation.action_ir import (
    ActionIRValidationError,
    JSONType,
    ToolSchema,
    ValueSchema,
    action_ir_equal,
    parse_action_ir,
)
from barunlm.evaluation.gvs_schema_presentation import (
    IDENTITY_PRESENTATION,
    RENAMED_PRESENTATION,
    ArgumentNameMap,
    GVSSchemaPresentationError,
    SchemaPresentation,
    ToolNameMap,
    action_round_trips_identity,
    build_schema_presentation,
    load_schema_presentation_receipt_json,
    lower_presented_action_ir,
    verify_schema_presentation,
)


def _canonical_schemas() -> tuple[ToolSchema, ...]:
    string = ValueSchema(JSONType.STRING)
    boolean = ValueSchema(JSONType.BOOLEAN)
    route_mode = ValueSchema(JSONType.STRING, enum=("drive", "walk"))
    return (
        ToolSchema(
            name="calendar_create",
            arguments={
                "notify": boolean,
                "start_at": string,
                "title": string,
            },
            required=frozenset({"start_at", "title"}),
            additional_arguments=False,
            side_effecting=True,
        ),
        ToolSchema(
            name="map_route",
            arguments={
                "destination_id": string,
                "mode": route_mode,
                "origin_id": string,
            },
            required=frozenset({"destination_id", "mode", "origin_id"}),
            additional_arguments=False,
            side_effecting=False,
        ),
    )


def _renamed_presentation() -> SchemaPresentation:
    return SchemaPresentation(
        presentation_id="renamed-calendar-map-a",
        mode=RENAMED_PRESENTATION,
        tools=(
            ToolNameMap(
                canonical_name="calendar_create",
                presented_name="add_agenda",
                arguments=(
                    ArgumentNameMap("notify", "alert"),
                    ArgumentNameMap("start_at", "begins"),
                    ArgumentNameMap("title", "subject"),
                ),
            ),
            ToolNameMap(
                canonical_name="map_route",
                presented_name="find_path",
                arguments=(
                    ArgumentNameMap("destination_id", "to_place"),
                    ArgumentNameMap("mode", "travel_mode"),
                    ArgumentNameMap("origin_id", "from_place"),
                ),
            ),
        ),
    )


def _renamed_prompt_tools() -> tuple[ToolDefinition, ...]:
    # Prompt order is deliberately distinct from canonical schema order and is bound.
    return (
        ToolDefinition(
            name="find_path",
            description="Find a route between two known places.",
            arguments=(
                ToolArgument("from_place", "string", "Starting place identifier.", True),
                ToolArgument("to_place", "string", "Destination place identifier.", True),
                ToolArgument("travel_mode", "string", "Travel mode.", True),
            ),
        ),
        ToolDefinition(
            name="add_agenda",
            description="Propose a new agenda entry.",
            arguments=(
                ToolArgument("subject", "string", "Agenda title.", True),
                ToolArgument("begins", "string", "ISO start datetime.", True),
                ToolArgument("alert", "boolean", "Whether an alert is requested.", False),
            ),
        ),
    )


def _identity_presentation() -> SchemaPresentation:
    return SchemaPresentation(
        presentation_id="identity-calendar-map-a",
        mode=IDENTITY_PRESENTATION,
        tools=(
            ToolNameMap(
                canonical_name="calendar_create",
                presented_name="calendar_create",
                arguments=(
                    ArgumentNameMap("notify", "notify"),
                    ArgumentNameMap("start_at", "start_at"),
                    ArgumentNameMap("title", "title"),
                ),
            ),
            ToolNameMap(
                canonical_name="map_route",
                presented_name="map_route",
                arguments=(
                    ArgumentNameMap("destination_id", "destination_id"),
                    ArgumentNameMap("mode", "mode"),
                    ArgumentNameMap("origin_id", "origin_id"),
                ),
            ),
        ),
    )


def _identity_prompt_tools() -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="calendar_create",
            description="Propose a calendar event.",
            arguments=(
                ToolArgument("notify", "boolean", "Whether to notify.", False),
                ToolArgument("start_at", "string", "ISO start datetime.", True),
                ToolArgument("title", "string", "Event title.", True),
            ),
        ),
        ToolDefinition(
            name="map_route",
            description="Find a route.",
            arguments=(
                ToolArgument("destination_id", "string", "Destination identifier.", True),
                ToolArgument("mode", "string", "Travel mode.", True),
                ToolArgument("origin_id", "string", "Origin identifier.", True),
            ),
        ),
    )


def test_renamed_tool_and_argument_names_lower_without_changing_values() -> None:
    schemas = _canonical_schemas()
    presentation = _renamed_presentation()
    prompt_tools = _renamed_prompt_tools()
    receipt = build_schema_presentation(
        canonical_schemas=schemas,
        prompt_tools=prompt_tools,
        presentation=presentation,
    )
    presented = parse_action_ir(
        """{"calls":[{"args":{"from_place":"home","to_place":"lab","travel_mode":"walk"},"tool":"find_path"},{"args":{"begins":"2026-09-01T09:00:00","subject":"Planning"},"tool":"add_agenda"}],"decision":"CONFIRM","mode":"SERIAL"}""",
        receipt.presented_schemas,
    )
    lowered = lower_presented_action_ir(
        presented,
        receipt=receipt,
        canonical_schemas=schemas,
        prompt_tools=prompt_tools,
        presentation=presentation,
    )
    expected = parse_action_ir(
        """{"calls":[{"args":{"destination_id":"lab","mode":"walk","origin_id":"home"},"tool":"map_route"},{"args":{"start_at":"2026-09-01T09:00:00","title":"Planning"},"tool":"calendar_create"}],"decision":"CONFIRM","mode":"SERIAL"}""",
        schemas,
    )
    assert action_ir_equal(lowered, expected)
    assert lowered.canonical_json() == expected.canonical_json()
    assert receipt.launch_authorized is False
    assert receipt.descriptions_semantically_audited is False
    assert receipt.authorizes_cuda_or_jarvis_access is False


def test_identity_presentation_round_trips_exact_action_ir() -> None:
    schemas = _canonical_schemas()
    presentation = _identity_presentation()
    prompt_tools = _identity_prompt_tools()
    receipt = build_schema_presentation(
        canonical_schemas=schemas,
        prompt_tools=prompt_tools,
        presentation=presentation,
    )
    action = parse_action_ir(
        """{"calls":[{"args":{"destination_id":"lab","mode":"drive","origin_id":"home"},"tool":"map_route"}],"decision":"CALL","mode":"SINGLE"}""",
        receipt.presented_schemas,
    )
    assert action_round_trips_identity(
        action,
        receipt=receipt,
        canonical_schemas=schemas,
        prompt_tools=prompt_tools,
        presentation=presentation,
    )


@pytest.mark.parametrize(
    "raw",
    [
        '{"decision":"ABSTAIN"}',
        '{"decision":"CLARIFY","missing":["datetime","recipient"]}',
    ],
)
def test_policy_controls_preserve_exact_semantic_labels(raw: str) -> None:
    schemas = _canonical_schemas()
    presentation = _renamed_presentation()
    prompt_tools = _renamed_prompt_tools()
    receipt = build_schema_presentation(
        canonical_schemas=schemas,
        prompt_tools=prompt_tools,
        presentation=presentation,
    )
    action = parse_action_ir(raw, receipt.presented_schemas)
    lowered = lower_presented_action_ir(
        action,
        receipt=receipt,
        canonical_schemas=schemas,
        prompt_tools=prompt_tools,
        presentation=presentation,
    )
    assert lowered.canonical_json() == action.canonical_json()


def test_canonical_names_are_not_accepted_under_renamed_presentation() -> None:
    receipt = build_schema_presentation(
        canonical_schemas=_canonical_schemas(),
        prompt_tools=_renamed_prompt_tools(),
        presentation=_renamed_presentation(),
    )
    with pytest.raises(ActionIRValidationError, match="unknown tool"):
        parse_action_ir(
            """{"calls":[{"args":{"destination_id":"lab","mode":"walk","origin_id":"home"},"tool":"map_route"}],"decision":"CALL","mode":"SINGLE"}""",
            receipt.presented_schemas,
        )


def test_presentation_requires_total_tool_and_argument_bijections() -> None:
    missing_tool = replace(
        _renamed_presentation(),
        tools=(_renamed_presentation().tools[0],),
    )
    with pytest.raises(GVSSchemaPresentationError, match="cover exactly"):
        build_schema_presentation(
            canonical_schemas=_canonical_schemas(),
            prompt_tools=_renamed_prompt_tools(),
            presentation=missing_tool,
        )

    missing_argument = replace(
        _renamed_presentation(),
        tools=(
            replace(
                _renamed_presentation().tools[0],
                arguments=_renamed_presentation().tools[0].arguments[:-1],
            ),
            _renamed_presentation().tools[1],
        ),
    )
    with pytest.raises(GVSSchemaPresentationError, match="not total and exact"):
        build_schema_presentation(
            canonical_schemas=_canonical_schemas(),
            prompt_tools=_renamed_prompt_tools(),
            presentation=missing_argument,
        )


def test_duplicate_presented_names_and_false_modes_fail_closed() -> None:
    base = _renamed_presentation()
    with pytest.raises(GVSSchemaPresentationError, match="globally unique"):
        SchemaPresentation(
            presentation_id="duplicate-tools",
            mode=RENAMED_PRESENTATION,
            tools=(base.tools[0], replace(base.tools[1], presented_name="add_agenda")),
        )
    with pytest.raises(GVSSchemaPresentationError, match="must rename"):
        replace(_identity_presentation(), mode=RENAMED_PRESENTATION)
    with pytest.raises(GVSSchemaPresentationError, match="cannot rename"):
        replace(_renamed_presentation(), mode=IDENTITY_PRESENTATION)


@pytest.mark.parametrize("field", ["type", "required", "description"])
def test_prompt_schema_mismatch_or_unsafe_text_fails_closed(field: str) -> None:
    tools = list(_renamed_prompt_tools())
    calendar = tools[1]
    arguments = list(calendar.arguments)
    if field == "type":
        arguments[0] = replace(arguments[0], type_name="boolean")
    elif field == "required":
        arguments[0] = replace(arguments[0], required=False)
    else:
        arguments[0] = replace(arguments[0], description="bad\nshortcut")
    tools[1] = replace(calendar, arguments=tuple(arguments))
    message = "prompt type|prompt required|newline"
    with pytest.raises(GVSSchemaPresentationError, match=message):
        build_schema_presentation(
            canonical_schemas=_canonical_schemas(),
            prompt_tools=tuple(tools),
            presentation=_renamed_presentation(),
        )


def test_receipt_recomputation_detects_schema_prompt_and_mapping_mutation() -> None:
    schemas = _canonical_schemas()
    presentation = _renamed_presentation()
    tools = _renamed_prompt_tools()
    receipt = build_schema_presentation(
        canonical_schemas=schemas,
        prompt_tools=tools,
        presentation=presentation,
    )
    verify_schema_presentation(
        receipt,
        canonical_schemas=schemas,
        prompt_tools=tools,
        presentation=presentation,
    )

    object.__setattr__(schemas[0], "side_effecting", False)
    with pytest.raises(GVSSchemaPresentationError, match="live recomputation"):
        verify_schema_presentation(
            receipt,
            canonical_schemas=schemas,
            prompt_tools=tools,
            presentation=presentation,
        )

    fresh_schemas = _canonical_schemas()
    fresh_receipt = build_schema_presentation(
        canonical_schemas=fresh_schemas,
        prompt_tools=tools,
        presentation=presentation,
    )
    object.__setattr__(fresh_receipt.presented_schemas[0], "side_effecting", True)
    with pytest.raises(GVSSchemaPresentationError, match="live recomputation"):
        verify_schema_presentation(
            fresh_receipt,
            canonical_schemas=fresh_schemas,
            prompt_tools=tools,
            presentation=presentation,
        )


def test_receipt_constructor_and_lowering_reject_forged_live_objects() -> None:
    schemas = _canonical_schemas()
    presentation = _renamed_presentation()
    tools = _renamed_prompt_tools()
    receipt = build_schema_presentation(
        canonical_schemas=schemas,
        prompt_tools=tools,
        presentation=presentation,
    )
    with pytest.raises(GVSSchemaPresentationError, match="hash differs"):
        replace(receipt, presented_schema_sha256="0" * 64)

    action = parse_action_ir(
        """{"calls":[{"args":{"from_place":"home","to_place":"lab","travel_mode":"walk"},"tool":"find_path"}],"decision":"CALL","mode":"SINGLE"}""",
        receipt.presented_schemas,
    )
    object.__setattr__(action.calls[0], "tool", "map_route")
    with pytest.raises(GVSSchemaPresentationError, match="live-schema revalidation"):
        lower_presented_action_ir(
            action,
            receipt=receipt,
            canonical_schemas=schemas,
            prompt_tools=tools,
            presentation=presentation,
        )


def test_prompt_and_canonical_schema_inputs_require_exact_immutable_tuples() -> None:
    with pytest.raises(
        GVSSchemaPresentationError, match="canonical schemas must be an exact tuple"
    ):
        build_schema_presentation(
            canonical_schemas=list(_canonical_schemas()),  # type: ignore[arg-type]
            prompt_tools=_renamed_prompt_tools(),
            presentation=_renamed_presentation(),
        )
    with pytest.raises(GVSSchemaPresentationError, match="prompt_tools must be an exact tuple"):
        build_schema_presentation(
            canonical_schemas=_canonical_schemas(),
            prompt_tools=list(_renamed_prompt_tools()),  # type: ignore[arg-type]
            presentation=_renamed_presentation(),
        )


def test_unmappable_additional_arguments_are_rejected() -> None:
    schemas = list(_canonical_schemas())
    schemas[0] = ToolSchema(
        name=schemas[0].name,
        arguments=schemas[0].arguments,
        required=schemas[0].required,
        additional_arguments=True,
        side_effecting=schemas[0].side_effecting,
    )
    with pytest.raises(GVSSchemaPresentationError, match="additional arguments"):
        build_schema_presentation(
            canonical_schemas=tuple(schemas),
            prompt_tools=_renamed_prompt_tools(),
            presentation=_renamed_presentation(),
        )


def test_nested_schema_enums_optional_arguments_and_values_are_name_only() -> None:
    string = ValueSchema(JSONType.STRING)
    nested = ValueSchema(
        JSONType.OBJECT,
        properties={
            "labels": ValueSchema(
                JSONType.ARRAY,
                items=ValueSchema(JSONType.STRING, enum=("home", "work")),
                set_semantics=False,
            ),
            "priority": ValueSchema(JSONType.INTEGER, enum=(1, 2, 3)),
        },
        required=frozenset({"priority"}),
        additional_properties=False,
    )
    schemas = (
        ToolSchema(
            name="note_create",
            arguments={"metadata": nested, "title": string},
            required=frozenset({"title"}),
            additional_arguments=False,
            side_effecting=True,
        ),
    )
    presentation = SchemaPresentation(
        presentation_id="nested-renamed-a",
        mode=RENAMED_PRESENTATION,
        tools=(
            ToolNameMap(
                canonical_name="note_create",
                presented_name="write_memo",
                arguments=(
                    ArgumentNameMap("metadata", "details"),
                    ArgumentNameMap("title", "heading"),
                ),
            ),
        ),
    )
    tools = (
        ToolDefinition(
            name="write_memo",
            description="Propose a structured memo.",
            arguments=(
                ToolArgument("heading", "string", "Memo heading.", True),
                ToolArgument("details", "object", "Optional structured details.", False),
            ),
        ),
    )
    receipt = build_schema_presentation(
        canonical_schemas=schemas,
        prompt_tools=tools,
        presentation=presentation,
    )
    action = parse_action_ir(
        """{"calls":[{"args":{"details":{"labels":["work","home"],"priority":2},"heading":"Plan"},"tool":"write_memo"},{"args":{"heading":"Second"},"tool":"write_memo"}],"decision":"CONFIRM","mode":"SERIAL"}""",
        receipt.presented_schemas,
    )
    lowered = lower_presented_action_ir(
        action,
        receipt=receipt,
        canonical_schemas=schemas,
        prompt_tools=tools,
        presentation=presentation,
    )
    expected = parse_action_ir(
        """{"calls":[{"args":{"metadata":{"labels":["work","home"],"priority":2},"title":"Plan"},"tool":"note_create"},{"args":{"title":"Second"},"tool":"note_create"}],"decision":"CONFIRM","mode":"SERIAL"}""",
        schemas,
    )
    assert lowered.canonical_json() == expected.canonical_json()


def test_receipt_direct_constructor_seals_all_retained_crosslinks() -> None:
    receipt = build_schema_presentation(
        canonical_schemas=_canonical_schemas(),
        prompt_tools=_renamed_prompt_tools(),
        presentation=_renamed_presentation(),
    )
    with pytest.raises(GVSSchemaPresentationError, match="canonical schema hash differs"):
        replace(receipt, canonical_schema_sha256="0" * 64)
    with pytest.raises(GVSSchemaPresentationError, match="presentation hash differs"):
        replace(
            receipt,
            presentation=replace(receipt.presentation, presentation_id="forged-presentation"),
        )
    with pytest.raises(GVSSchemaPresentationError, match="prompt tool hash differs"):
        replace(receipt, prompt_tools_sha256="0" * 64)
    with pytest.raises(GVSSchemaPresentationError, match="names must be unique"):
        replace(
            receipt,
            presented_schemas=(receipt.presented_schemas[0], receipt.presented_schemas[0]),
        )
    with pytest.raises(GVSSchemaPresentationError, match="must remain nonauthorizing"):
        replace(receipt, launch_authorized=True)


def test_strict_receipt_artifact_round_trip_and_malformed_inputs() -> None:
    schemas = _canonical_schemas()
    tools = _renamed_prompt_tools()
    presentation = _renamed_presentation()
    receipt = build_schema_presentation(
        canonical_schemas=schemas,
        prompt_tools=tools,
        presentation=presentation,
    )
    loaded = load_schema_presentation_receipt_json(
        receipt.to_json_bytes(),
        expected_receipt_sha256=receipt.sha256,
        canonical_schemas=schemas,
        prompt_tools=tools,
        presentation=presentation,
    )
    assert loaded is not receipt
    assert loaded.sha256 == receipt.sha256

    kwargs = {
        "expected_receipt_sha256": receipt.sha256,
        "canonical_schemas": schemas,
        "prompt_tools": tools,
        "presentation": presentation,
    }
    with pytest.raises(GVSSchemaPresentationError, match="duplicate JSON key"):
        load_schema_presentation_receipt_json(b'{"x":1,"x":2}', **kwargs)
    with pytest.raises(GVSSchemaPresentationError, match="integer exceeds"):
        load_schema_presentation_receipt_json(b'{"x":' + b"9" * 21 + b"}", **kwargs)
    with pytest.raises(GVSSchemaPresentationError, match="float must be finite"):
        load_schema_presentation_receipt_json(b'{"x":1e999}', **kwargs)
    with pytest.raises(GVSSchemaPresentationError, match="strict bounded"):
        load_schema_presentation_receipt_json(
            b"[" * 1_500 + b"0" + b"]" * 1_500,
            **kwargs,
        )
    with pytest.raises(GVSSchemaPresentationError, match="nonempty bounded bytes"):
        load_schema_presentation_receipt_json(b"x" * (4 * 1_024 * 1_024 + 1), **kwargs)


def test_schema_presentation_loader_rejects_live_verifier_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schemas = _canonical_schemas()
    tools = _renamed_prompt_tools()
    presentation = _renamed_presentation()
    receipt = build_schema_presentation(
        canonical_schemas=schemas,
        prompt_tools=tools,
        presentation=presentation,
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            presentation_module,
            "verify_schema_presentation",
            lambda *_args, **_kwargs: receipt,
        )
        with pytest.raises(GVSSchemaPresentationError, match="runtime"):
            load_schema_presentation_receipt_json(
                receipt.to_json_bytes(),
                expected_receipt_sha256=receipt.sha256,
                canonical_schemas=schemas,
                prompt_tools=tools,
                presentation=presentation,
            )


def test_artifact_nonauthorization_and_prompt_description_are_hash_bound() -> None:
    schemas = _canonical_schemas()
    tools = _renamed_prompt_tools()
    presentation = _renamed_presentation()
    receipt = build_schema_presentation(
        canonical_schemas=schemas,
        prompt_tools=tools,
        presentation=presentation,
    )
    artifact = json.loads(receipt.to_json_bytes())
    artifact["receipt"]["launch_authorized"] = True
    tampered = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(GVSSchemaPresentationError, match="must remain nonauthorizing"):
        load_schema_presentation_receipt_json(
            tampered,
            expected_receipt_sha256=receipt.sha256,
            canonical_schemas=schemas,
            prompt_tools=tools,
            presentation=presentation,
        )

    changed_tools = list(tools)
    changed_tools[0] = replace(changed_tools[0], description="A different safe description.")
    with pytest.raises(GVSSchemaPresentationError, match="live recomputation"):
        verify_schema_presentation(
            receipt,
            canonical_schemas=schemas,
            prompt_tools=tuple(changed_tools),
            presentation=presentation,
        )


def test_prompt_snapshot_detects_nested_caller_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = _renamed_prompt_tools()
    original = presentation_module._prompt_tools_record
    calls = 0

    def mutate_after_first(value: tuple[ToolDefinition, ...]) -> list[dict[str, object]]:
        nonlocal calls
        record = original(value)
        calls += 1
        if calls == 1:
            object.__setattr__(value[0].arguments[0], "description", "Mutated description.")
        return record

    monkeypatch.setattr(presentation_module, "_prompt_tools_record", mutate_after_first)
    with pytest.raises(GVSSchemaPresentationError, match="changed during the stable snapshot"):
        presentation_module._stable_prompt_tools(tools)


def test_loaded_runtime_and_behavior_global_mutation_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = {
        "canonical_schemas": _canonical_schemas(),
        "prompt_tools": _renamed_prompt_tools(),
        "presentation": _renamed_presentation(),
    }
    with monkeypatch.context() as scoped:
        scoped.setattr(presentation_module, "parse_action_ir", lambda *_args: None)
        with pytest.raises(GVSSchemaPresentationError, match="alias parse_action_ir"):
            build_schema_presentation(**kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(presentation_module, "_MAX_TOOLS", 1)
        with pytest.raises(GVSSchemaPresentationError, match="loaded runtime"):
            build_schema_presentation(**kwargs)


def test_deep_and_huge_schema_inputs_fail_before_receipt_creation() -> None:
    nested = ValueSchema(JSONType.STRING)
    for _ in range(34):
        nested = ValueSchema(JSONType.ARRAY, items=nested)
    schemas = list(_canonical_schemas())
    schemas[0] = ToolSchema(
        name="calendar_create",
        arguments={"nested": nested},
        required=frozenset({"nested"}),
        additional_arguments=False,
        side_effecting=True,
    )
    with pytest.raises(GVSSchemaPresentationError, match="nesting exceeds"):
        build_schema_presentation(
            canonical_schemas=tuple(schemas),
            prompt_tools=_renamed_prompt_tools(),
            presentation=_renamed_presentation(),
        )

    huge_enum = ValueSchema(
        JSONType.STRING,
        enum=tuple(f"value-{index}" for index in range(1_025)),
    )
    schemas = list(_canonical_schemas())
    schemas[0] = ToolSchema(
        name="calendar_create",
        arguments={
            "notify": huge_enum,
            "start_at": ValueSchema(JSONType.STRING),
            "title": ValueSchema(JSONType.STRING),
        },
        required=frozenset({"start_at", "title"}),
        additional_arguments=False,
        side_effecting=True,
    )
    with pytest.raises(GVSSchemaPresentationError, match="enum exceeds"):
        build_schema_presentation(
            canonical_schemas=tuple(schemas),
            prompt_tools=_renamed_prompt_tools(),
            presentation=_renamed_presentation(),
        )
