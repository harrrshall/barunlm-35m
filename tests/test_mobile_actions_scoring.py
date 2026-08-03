from __future__ import annotations

import json

import pytest

from barunlm.evaluation.mobile_actions import (
    MobileActionsScoreError,
    aggregate_record,
    schemas_from_prompt,
    score_rows,
)

PROMPT = (
    "<bos><system>\nACTION_IR_V1\nNOW 2026-08-03T16:30:00\nTOOLS\n"
    "set_timer(minutes:integer!): Start a timer.\n"
    "send_email(to:string!, subject:string!, body:string): Send an email.\n"
    "<user>\nSet a five minute timer.\n<assistant>\n"
)
GOLD_FIVE = (
    '{"calls":[{"args":{"minutes":5},"tool":"set_timer"}],"decision":"CALL","mode":"SINGLE"}'
)
GOLD_TEN = (
    '{"calls":[{"args":{"minutes":10},"tool":"set_timer"}],"decision":"CALL","mode":"SINGLE"}'
)


def _manifest(sample_id: str, target: str = GOLD_FIVE) -> dict[str, object]:
    return {
        "id": sample_id,
        "prompt": PROMPT,
        "target": target,
        "metadata": {"call_names": ["set_timer"]},
    }


def _prediction(sample_id: str, raw: str) -> dict[str, object]:
    return {"id": sample_id, "prediction_raw": raw, "truncated": False}


def test_schema_parser_preserves_types_and_required_fields() -> None:
    schemas = {schema.name: schema for schema in schemas_from_prompt(PROMPT)}
    assert schemas["set_timer"].required == frozenset({"minutes"})
    assert schemas["send_email"].required == frozenset({"to", "subject"})
    assert set(schemas["send_email"].arguments) == {"to", "subject", "body"}


def test_mobile_score_requires_exact_executable_ast() -> None:
    aggregate, samples = score_rows(
        [_manifest("correct"), _manifest("wrong")],
        [_prediction("correct", GOLD_FIVE), _prediction("wrong", GOLD_TEN)],
    )
    assert aggregate.ast_exact_match.numerator == 1
    assert aggregate.policy_safe_executable_success.numerator == 1
    assert aggregate.schema_valid.numerator == 2
    assert samples[1].failure_categories == (
        "ast_mismatch",
        "simulator_failure",
        "mobile_actions_ast_mismatch",
    )


def test_mobile_score_counts_malformed_and_truncated_outputs_as_failures() -> None:
    predictions = [
        {"id": "bad", "prediction_raw": "```json\n{}\n```", "truncated": False},
        {"id": "cut", "prediction_raw": GOLD_FIVE, "truncated": True},
    ]
    aggregate, _ = score_rows(
        [_manifest("bad"), _manifest("cut")],
        predictions,
    )
    assert aggregate.parse_failure.numerator == 1
    assert aggregate.truncation.numerator == 1
    assert aggregate.policy_safe_executable_success.numerator == 0


def test_mobile_score_refuses_missing_or_extra_ids_and_schema_changes() -> None:
    with pytest.raises(MobileActionsScoreError, match="missing ID"):
        score_rows([_manifest("one")], [])
    with pytest.raises(MobileActionsScoreError, match="unknown IDs"):
        score_rows(
            [_manifest("one")],
            [_prediction("one", GOLD_FIVE), _prediction("two", GOLD_FIVE)],
        )

    changed_prompt = PROMPT.replace("minutes:integer!", "minutes:string!")
    changed = _manifest("two")
    changed["prompt"] = changed_prompt
    with pytest.raises(MobileActionsScoreError, match="changes within"):
        score_rows(
            [_manifest("one"), changed],
            [_prediction("one", GOLD_FIVE), _prediction("two", GOLD_FIVE)],
        )


def test_aggregate_record_is_json_serializable() -> None:
    aggregate, _ = score_rows([_manifest("one")], [_prediction("one", GOLD_FIVE)])
    # The public writer relies on this property for immutable artifact creation.
    json.dumps(aggregate_record(aggregate), allow_nan=False)
