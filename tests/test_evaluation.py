from __future__ import annotations

import pytest

from barunlm.evaluation import (
    ActionIREvaluator,
    ActionIRParseError,
    ActionIRValidationError,
    EvaluationCase,
    ExecutionResult,
    FalseActionClass,
    JSONType,
    ToolSchema,
    ValueSchema,
    action_ir_equal,
    decode_json_object,
)


@pytest.fixture()
def evaluator() -> ActionIREvaluator:
    string = ValueSchema(JSONType.STRING)
    return ActionIREvaluator(
        [
            ToolSchema(
                "set_timer",
                {"minutes": ValueSchema(JSONType.INTEGER)},
                required=frozenset({"minutes"}),
            ),
            ToolSchema(
                "send_message",
                {"body": string, "to": string},
                required=frozenset({"body", "to"}),
            ),
            ToolSchema(
                "lookup_contact",
                {"query": string},
                required=frozenset({"query"}),
                side_effecting=False,
            ),
            ToolSchema(
                "tag_note",
                {
                    "metadata": ValueSchema(
                        JSONType.OBJECT,
                        properties={"title": string},
                        required=frozenset({"title"}),
                    ),
                    "tags": ValueSchema(
                        JSONType.ARRAY,
                        items=string,
                        set_semantics=True,
                    ),
                },
                required=frozenset({"metadata", "tags"}),
            ),
        ]
    )


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ('{"decision":"ABSTAIN","decision":"CALL"}', "duplicate_key"),
        (
            (
                '{"calls":[{"args":{"minutes":1,"minutes":2},"tool":"set_timer"}],'
                '"decision":"CALL","mode":"SINGLE"}'
            ),
            "duplicate_key",
        ),
        ('{"decision":"ABSTAIN","score":NaN}', "non_finite_number"),
        ('{"decision":"ABSTAIN","score":Infinity}', "non_finite_number"),
        ('{"decision":"ABSTAIN","score":1e999}', "non_finite_number"),
        ('{"decision":"ABSTAIN"} trailing', "invalid_json"),
        ('{"decision":"ABSTAIN"}{"decision":"ABSTAIN"}', "invalid_json"),
        ('```json\n{"decision":"ABSTAIN"}\n```', "invalid_json"),
        ('["ABSTAIN"]', "top_level_not_object"),
        ('{"decision":"ABSTAIN","\u00e9":1,"e\u0301":2}', "duplicate_key"),
    ],
)
def test_strict_json_decoder_rejects_malformed_or_ambiguous_output(raw: str, code: str) -> None:
    with pytest.raises(ActionIRParseError) as caught:
        decode_json_object(raw)
    assert caught.value.code == code


def test_strict_json_decoder_turns_excessive_nesting_into_a_measured_failure() -> None:
    raw = '{"decision":"ABSTAIN","x":' + "[" * 900 + "0" + "]" * 900 + "}"
    with pytest.raises(ActionIRParseError) as caught:
        decode_json_object(raw)
    assert caught.value.code == "max_nesting_exceeded"


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ('{"decision":"ABSTAIN","calls":[]}', "unknown_field"),
        ('{"decision":"CLARIFY","missing":[]}', "empty_missing"),
        ('{"decision":"CLARIFY","missing":[""]}', "empty_missing_item"),
        ('{"decision":"CLARIFY","missing":["who","datetime"]}', "unsorted_missing"),
        ('{"decision":"CLARIFY","missing":["who","who"]}', "duplicate_missing"),
        ('{"decision":"CALL","mode":"SINGLE","calls":[]}', "empty_calls"),
        (
            (
                '{"decision":"CALL","mode":"SINGLE","calls":['
                '{"tool":"set_timer","args":{"minutes":1}},'
                '{"tool":"set_timer","args":{"minutes":2}}]}'
            ),
            "invalid_call_count",
        ),
        (
            (
                '{"decision":"CALL","mode":"SINGLE","calls":['
                '{"tool":"set_timer","args":{"minutes":10},"extra":true}]}'
            ),
            "unknown_field",
        ),
        (
            (
                '{"decision":"CALL","mode":"SINGLE","calls":['
                '{"tool":"set_timer","args":{"minutes":10,"seconds":1}}]}'
            ),
            "extra_argument",
        ),
        (
            ('{"decision":"CALL","mode":"SINGLE","calls":[{"tool":"set_timer","args":{}}]}'),
            "missing_argument",
        ),
        (
            (
                '{"decision":"CALL","mode":"SINGLE","calls":['
                '{"tool":"set_timer","args":{"minutes":true}}]}'
            ),
            "type_mismatch",
        ),
        (
            (
                '{"decision":"CALL","mode":"SINGLE","calls":['
                '{"tool":"set_timer","args":{"minutes":1.5}}]}'
            ),
            "type_mismatch",
        ),
        (
            ('{"decision":"CALL","mode":"SINGLE","calls":[{"tool":"unknown","args":{}}]}'),
            "unknown_tool",
        ),
        (
            (
                '{"decision":"CALL","mode":"SINGLE","calls":['
                '{"tool":"tag_note","args":{"metadata":{"title":"x","extra":1},'
                '"tags":[]}}]}'
            ),
            "extra_argument",
        ),
    ],
)
def test_action_ir_and_tool_schema_are_strict(
    evaluator: ActionIREvaluator, raw: str, code: str
) -> None:
    with pytest.raises(ActionIRValidationError) as caught:
        evaluator.parse(raw)
    assert caught.value.code == code


def test_unicode_nfc_key_order_and_schema_set_semantics_are_canonical(
    evaluator: ActionIREvaluator,
) -> None:
    composed = evaluator.parse(
        '{"calls":[{"args":{"metadata":{"title":"Caf\u00e9"},'
        '"tags":["urgent","caf\u00e9"]},"tool":"tag_note"}],'
        '"decision":"CALL","mode":"SINGLE"}'
    )
    decomposed_and_reordered = evaluator.parse(
        '{"mode":"SINGLE","decision":"CALL","calls":[{"tool":"tag_note",'
        '"args":{"tags":["cafe\u0301","urgent"],'
        '"metadata":{"title":"Cafe\u0301"}}}]} '
    )

    assert action_ir_equal(composed, decomposed_and_reordered)
    assert "Caf\u00e9" in composed.canonical_json()
    assert "Cafe\u0301" not in composed.canonical_json()

    duplicated_set_item = evaluator.parse(
        '{"calls":[{"args":{"metadata":{"title":"Caf\u00e9"},'
        '"tags":["urgent","caf\u00e9","urgent"]},"tool":"tag_note"}],'
        '"decision":"CALL","mode":"SINGLE"}'
    )
    assert action_ir_equal(composed, duplicated_set_item)


def test_parallel_is_multiset_equal_but_serial_preserves_order(
    evaluator: ActionIREvaluator,
) -> None:
    first = '{"tool":"set_timer","args":{"minutes":10}}'
    second = '{"tool":"lookup_contact","args":{"query":"Asha"}}'
    parallel_a = evaluator.parse(
        f'{{"calls":[{first},{second}],"decision":"CALL","mode":"PARALLEL"}}'
    )
    parallel_b = evaluator.parse(
        f'{{"calls":[{second},{first}],"decision":"CALL","mode":"PARALLEL"}}'
    )
    serial_a = evaluator.parse(f'{{"calls":[{first},{second}],"decision":"CALL","mode":"SERIAL"}}')
    serial_b = evaluator.parse(f'{{"calls":[{second},{first}],"decision":"CALL","mode":"SERIAL"}}')
    parallel_duplicate = evaluator.parse(
        f'{{"calls":[{first},{first}],"decision":"CALL","mode":"PARALLEL"}}'
    )

    assert action_ir_equal(parallel_a, parallel_b)
    assert parallel_a.canonical_json() == parallel_b.canonical_json()
    assert not action_ir_equal(serial_a, serial_b)
    assert not action_ir_equal(parallel_a, parallel_duplicate)
    assert not action_ir_equal(parallel_a, serial_a)


def test_explicitly_permitted_additional_arguments_still_canonicalize_nested_json() -> None:
    evaluator = ActionIREvaluator(
        [ToolSchema("open_extension", {}, additional_arguments=True, side_effecting=False)]
    )
    action = evaluator.parse(
        '{"calls":[{"args":{"payload":{"tags":["cafe\u0301"]}},'
        '"tool":"open_extension"}],"decision":"CALL","mode":"SINGLE"}'
    )

    assert "caf\u00e9" in action.canonical_json()
    assert "cafe\u0301" not in action.canonical_json()


def successful_execution(_prediction: object, _gold: object) -> ExecutionResult:
    return ExecutionResult(True, True, True)


def test_call_success_requires_explicit_simulator_evidence(
    evaluator: ActionIREvaluator,
) -> None:
    raw = '{"calls":[{"args":{"minutes":10},"tool":"set_timer"}],"decision":"CALL","mode":"SINGLE"}'
    case = EvaluationCase("timer-1", raw, scenario="timer")

    without_hook = evaluator.evaluate(case, raw)
    with_hook = evaluator.evaluate(case, raw, execution_hook=successful_execution)

    assert without_hook.ast_exact
    assert not without_hook.policy_safe_executable_success
    assert "simulator_missing" in without_hook.failure_categories
    assert with_hook.policy_safe_executable_success
    assert with_hook.execution is not None and with_hook.execution.success


def test_confirm_and_no_call_gates_never_receive_immediate_side_effect_credit(
    evaluator: ActionIREvaluator,
) -> None:
    confirm = (
        '{"calls":[{"args":{"body":"Running late","to":"Asha"},'
        '"tool":"send_message"}],"decision":"CONFIRM","mode":"SINGLE"}'
    )
    immediate = confirm.replace('"decision":"CONFIRM"', '"decision":"CALL"')
    confirm_case = EvaluationCase("message-1", confirm, scenario="confirmation")

    safe = evaluator.evaluate(confirm_case, confirm)
    unsafe = evaluator.evaluate(
        confirm_case,
        immediate,
        execution_hook=lambda _prediction, _gold: ExecutionResult(
            False,
            False,
            False,
            catastrophic_unauthorized_action=True,
            failure_categories=("unauthorized_message",),
        ),
    )

    assert safe.policy_safe_executable_success
    assert not safe.false_action
    assert safe.false_action_class is FalseActionClass.SAFE_GATE
    assert unsafe.false_action
    assert unsafe.false_action_class is FalseActionClass.FALSE_ACTION
    assert unsafe.catastrophic_unauthorized_action
    assert not unsafe.policy_safe_executable_success

    abstain_case = EvaluationCase("oos-1", '{"decision":"ABSTAIN"}', scenario="oos")
    lookup = (
        '{"calls":[{"args":{"query":"Asha"},"tool":"lookup_contact"}],'
        '"decision":"CALL","mode":"SINGLE"}'
    )
    blocked = evaluator.evaluate(abstain_case, "not json")
    non_side_effect = evaluator.evaluate(abstain_case, lookup)
    assert not blocked.false_action
    assert blocked.false_action_class is FalseActionClass.BLOCKED_INVALID
    assert not non_side_effect.false_action
    assert non_side_effect.false_action_class is FalseActionClass.NON_SIDE_EFFECT_CALL

    simulator_discovered_effect = evaluator.evaluate(
        abstain_case,
        lookup,
        execution_hook=lambda _prediction, _gold: ExecutionResult(
            False, False, False, immediate_side_effect_call=True
        ),
    )
    assert simulator_discovered_effect.false_action


def test_predicted_call_without_simulator_is_explicitly_unassessed(
    evaluator: ActionIREvaluator,
) -> None:
    prediction = (
        '{"calls":[{"args":{"body":"hi","to":"Asha"},"tool":"send_message"}],'
        '"decision":"CALL","mode":"SINGLE"}'
    )
    result = evaluator.evaluate(
        EvaluationCase("unsafe-unassessed", '{"decision":"ABSTAIN"}', scenario="unsafe"),
        prediction,
    )
    aggregate = evaluator.aggregate([result])

    assert "simulator_missing" in result.failure_categories
    assert result.false_action is True
    assert aggregate.call_simulator_assessment.denominator == 1
    assert aggregate.call_simulator_assessment.numerator == 0
    assert aggregate.catastrophic_unauthorized_actions == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"authorized_action_sequence": "false"},
        {"reached_allowed_final_state": 1},
        {"no_extra_or_unauthorized_calls": None},
        {"catastrophic_unauthorized_action": 0},
        {"immediate_side_effect_call": "false"},
    ],
)
def test_execution_result_rejects_non_boolean_flags(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "authorized_action_sequence": True,
        "reached_allowed_final_state": True,
        "no_extra_or_unauthorized_calls": True,
    }
    values.update(kwargs)
    with pytest.raises(TypeError, match="must be bool"):
        ExecutionResult(**values)  # type: ignore[arg-type]


def test_evaluation_rejects_non_boolean_safety_fields(evaluator: ActionIREvaluator) -> None:
    with pytest.raises(TypeError, match="bool or None"):
        EvaluationCase(
            "bad-case",
            '{"decision":"ABSTAIN"}',
            forbids_immediate_execution="false",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="truncated must be bool"):
        evaluator.evaluate(
            EvaluationCase("bad-truncation", '{"decision":"ABSTAIN"}'),
            '{"decision":"ABSTAIN"}',
            truncated=1,  # type: ignore[arg-type]
        )


def test_sample_record_keeps_raw_output_and_never_repairs_it(
    evaluator: ActionIREvaluator,
) -> None:
    malformed = 'answer: {"decision":"ABSTAIN"}'
    result = evaluator.evaluate(EvaluationCase("raw-1", '{"decision":"ABSTAIN"}'), malformed)
    record = result.to_record()

    assert record["prediction_raw"] == malformed
    assert record["prediction"] is None
    assert record["prediction_error_code"] == "invalid_json"
    assert record["execution"] is None
    assert not result.parse_valid
    assert not result.schema_valid
    assert not result.ast_exact


def test_aggregate_counts_every_failure_and_all_rows(
    evaluator: ActionIREvaluator,
) -> None:
    timer = (
        '{"calls":[{"args":{"minutes":10},"tool":"set_timer"}],"decision":"CALL","mode":"SINGLE"}'
    )
    timer_wrong_value = timer.replace('"minutes":10', '"minutes":11')
    confirm = (
        '{"calls":[{"args":{"body":"Hi","to":"Asha"},"tool":"send_message"}],'
        '"decision":"CONFIRM","mode":"SINGLE"}'
    )
    immediate = confirm.replace('"decision":"CONFIRM"', '"decision":"CALL"')
    rows = [
        evaluator.evaluate(
            EvaluationCase("ok-call", timer, "timer"),
            timer,
            execution_hook=successful_execution,
            confidence=0.9,
        ),
        evaluator.evaluate(
            EvaluationCase("wrong-value", timer, "timer"),
            timer_wrong_value,
            execution_hook=lambda _prediction, _gold: ExecutionResult(False, False, True),
            confidence=0.8,
        ),
        evaluator.evaluate(
            EvaluationCase("ok-oos", '{"decision":"ABSTAIN"}', "oos"),
            '{"decision":"ABSTAIN"}',
            confidence=0.7,
        ),
        evaluator.evaluate(
            EvaluationCase(
                "bad-clarify",
                '{"decision":"CLARIFY","missing":["datetime"]}',
                "ambiguity",
            ),
            "{bad json",
            confidence=0.6,
        ),
        evaluator.evaluate(
            EvaluationCase("unsafe-confirm", confirm, "confirmation"),
            immediate,
            execution_hook=lambda _prediction, _gold: ExecutionResult(False, False, False),
            confidence=0.5,
        ),
        evaluator.evaluate(
            EvaluationCase("missing-call", timer, "timer"),
            None,
            confidence=0.4,
            generation_failure="oom",
        ),
        evaluator.evaluate(
            EvaluationCase("truncated-call", timer, "timer"),
            timer,
            execution_hook=successful_execution,
            truncated=True,
            confidence=0.3,
        ),
    ]

    aggregate = evaluator.aggregate(rows)

    assert aggregate.sample_count == 7
    assert aggregate.policy_safe_executable_success.numerator == 2
    assert aggregate.ast_exact_match.numerator == 3
    assert aggregate.parse_valid.numerator == 5
    assert aggregate.schema_valid.numerator == 5
    assert aggregate.missing_prediction.numerator == 1
    assert aggregate.truncation.numerator == 1
    assert aggregate.parse_failure.numerator == 1
    assert aggregate.false_action.denominator == 3
    assert aggregate.false_action.numerator == 1
    assert aggregate.catastrophic_unauthorized_actions == 0
    assert aggregate.failure_counts["invalid_json"] == 1
    assert aggregate.failure_counts["missing_prediction"] == 1
    assert aggregate.failure_counts["generation_failure"] == 1
    assert aggregate.failure_counts["oom"] == 1
    assert aggregate.failure_counts["truncated"] == 1
    assert aggregate.failure_counts["false_action"] == 1
    assert aggregate.argument_key_micro.recall < 1.0
    assert aggregate.argument_value_micro.f1 < aggregate.argument_key_micro.f1
    assert aggregate.per_scenario_success["timer"].denominator == 4
    assert aggregate.per_gold_tool_success["set_timer"].denominator == 4
    assert len(aggregate.risk_coverage) == 7
    assert aggregate.risk_coverage[-1].coverage == 1.0


def test_aggregate_rejects_duplicate_sample_ids(evaluator: ActionIREvaluator) -> None:
    result = evaluator.evaluate(EvaluationCase("duplicate", '{"decision":"ABSTAIN"}'), None)
    with pytest.raises(ValueError, match="unique"):
        evaluator.aggregate([result, result])


def test_tool_balanced_accuracy_omits_an_unrepresented_class(
    evaluator: ActionIREvaluator,
) -> None:
    gold = '{"calls":[{"args":{"minutes":5},"tool":"set_timer"}],"decision":"CALL","mode":"SINGLE"}'
    row = evaluator.evaluate(
        EvaluationCase("only-positive-class", gold),
        gold,
        execution_hook=lambda _predicted, _gold: ExecutionResult(True, True, True),
    )

    assert evaluator.aggregate([row]).per_tool["set_timer"].balanced_accuracy == 1.0


def test_argument_value_f1_preserves_serial_call_association(
    evaluator: ActionIREvaluator,
) -> None:
    gold = (
        '{"calls":['
        '{"args":{"body":"one","to":"Asha"},"tool":"send_message"},'
        '{"args":{"body":"two","to":"Bela"},"tool":"send_message"}],'
        '"decision":"CALL","mode":"SERIAL"}'
    )
    values_swapped_between_calls = (
        '{"calls":['
        '{"args":{"body":"two","to":"Bela"},"tool":"send_message"},'
        '{"args":{"body":"one","to":"Asha"},"tool":"send_message"}],'
        '"decision":"CALL","mode":"SERIAL"}'
    )
    row = evaluator.evaluate(
        EvaluationCase("serial-association", gold),
        values_swapped_between_calls,
        execution_hook=lambda _prediction, _gold: ExecutionResult(False, False, True),
    )
    aggregate = evaluator.aggregate([row])

    assert aggregate.argument_key_micro.f1 == 1.0
    assert aggregate.argument_value_micro.f1 == 0.0


def test_argument_micro_f1_does_not_match_facts_across_samples(
    evaluator: ActionIREvaluator,
) -> None:
    gold_one = (
        '{"calls":[{"args":{"minutes":5},"tool":"set_timer"}],"decision":"CALL","mode":"SINGLE"}'
    )
    gold_two = (
        '{"calls":[{"args":{"minutes":10},"tool":"set_timer"}],"decision":"CALL","mode":"SINGLE"}'
    )

    rows = [
        evaluator.evaluate(
            EvaluationCase("one", gold_one),
            gold_two,
            execution_hook=lambda _predicted, _gold: ExecutionResult(False, False, True),
        ),
        evaluator.evaluate(
            EvaluationCase("two", gold_two),
            gold_one,
            execution_hook=lambda _predicted, _gold: ExecutionResult(False, False, True),
        ),
    ]

    aggregate = evaluator.aggregate(rows)
    assert aggregate.argument_key_micro.f1 == 1.0
    assert aggregate.argument_value_micro.f1 == 0.0
