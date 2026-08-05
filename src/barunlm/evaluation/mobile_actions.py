"""Deterministic Mobile Actions scoring against Action IR v1.

The public dataset is a useful SFT diagnostic, not the sealed product benchmark.  This
module derives the typed schema from the versioned prompt artifact, performs no output
repair, and writes sample-level evidence before producing aggregate metrics.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .action_ir import (
    ActionIR,
    JSONType,
    ToolSchema,
    ValueSchema,
    action_ir_equal,
)
from .evaluator import (
    ActionIREvaluator,
    AggregateEvaluation,
    EvaluationCase,
    ExecutionResult,
    SampleEvaluation,
)

MOBILE_ACTIONS_SCORER_VERSION = "barun-mobile-actions-score-v1"
_TOOL_LINE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<arguments>.*)\): (?P<description>[^\n]*)$"
)
_ARGUMENT = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*):(?P<kind>string|integer|number|boolean)"
    r"(?P<required>!)?$"
)
_TYPE_MAP = {
    "string": JSONType.STRING,
    "integer": JSONType.INTEGER,
    "number": JSONType.NUMBER,
    "boolean": JSONType.BOOLEAN,
}


class MobileActionsScoreError(ValueError):
    """A scored manifest or prediction artifact violates its frozen contract."""


def schemas_from_prompt(prompt: str) -> tuple[ToolSchema, ...]:
    """Parse the exact ``barun-action-prompt-v1`` tool block.

    Descriptions are intentionally ignored by the evaluator, but the complete line must
    follow the frozen renderer grammar.  This parser is strict so a renderer change cannot
    silently change scoring behavior.
    """

    try:
        tool_block = prompt.split("\nTOOLS\n", 1)[1].split("\n<user>\n", 1)[0]
    except (IndexError, ValueError) as error:
        raise MobileActionsScoreError("prompt lacks the versioned TOOLS block") from error
    if not tool_block:
        raise MobileActionsScoreError("prompt TOOLS block is empty")

    schemas: list[ToolSchema] = []
    seen: set[str] = set()
    for line in tool_block.splitlines():
        match = _TOOL_LINE.fullmatch(line)
        if match is None:
            raise MobileActionsScoreError(f"invalid rendered tool line: {line!r}")
        name = match.group("name")
        if name in seen:
            raise MobileActionsScoreError(f"duplicate rendered tool {name!r}")
        seen.add(name)
        arguments: dict[str, ValueSchema] = {}
        required: set[str] = set()
        raw_arguments = match.group("arguments")
        if raw_arguments:
            for raw_argument in raw_arguments.split(", "):
                argument = _ARGUMENT.fullmatch(raw_argument)
                if argument is None:
                    raise MobileActionsScoreError(
                        f"invalid rendered argument in tool {name!r}: {raw_argument!r}"
                    )
                argument_name = argument.group("name")
                if argument_name in arguments:
                    raise MobileActionsScoreError(
                        f"duplicate rendered argument {argument_name!r} in {name!r}"
                    )
                arguments[argument_name] = ValueSchema(_TYPE_MAP[argument.group("kind")])
                if argument.group("required"):
                    required.add(argument_name)
        schemas.append(
            ToolSchema(
                name=name,
                arguments=arguments,
                required=frozenset(required),
                side_effecting=True,
            )
        )
    return tuple(schemas)


def _execution_hook(predicted: ActionIR, gold: ActionIR) -> ExecutionResult:
    exact = action_ir_equal(predicted, gold)
    return ExecutionResult(
        authorized_action_sequence=exact,
        reached_allowed_final_state=exact,
        no_extra_or_unauthorized_calls=exact,
        failure_categories=() if exact else ("mobile_actions_ast_mismatch",),
    )


def score_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[AggregateEvaluation, tuple[SampleEvaluation, ...]]:
    """Score aligned manifest/prediction rows with strict ID and schema checks."""

    predictions: dict[str, Mapping[str, Any]] = {}
    for row in prediction_rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise MobileActionsScoreError("every prediction row requires a non-empty id")
        if sample_id in predictions:
            raise MobileActionsScoreError(f"duplicate prediction ID {sample_id!r}")
        predictions[sample_id] = row

    results: list[SampleEvaluation] = []
    registry: dict[str, ToolSchema] = {}
    seen_manifest_ids: set[str] = set()
    for row in manifest_rows:
        sample_id = row.get("id")
        prompt = row.get("prompt")
        target = row.get("target")
        metadata = row.get("metadata", {})
        if not isinstance(sample_id, str) or not sample_id:
            raise MobileActionsScoreError("every manifest row requires a non-empty id")
        if sample_id in seen_manifest_ids:
            raise MobileActionsScoreError(f"duplicate manifest ID {sample_id!r}")
        seen_manifest_ids.add(sample_id)
        if not isinstance(prompt, str) or not isinstance(target, str):
            raise MobileActionsScoreError(f"manifest row {sample_id!r} lacks prompt or target")
        if not isinstance(metadata, Mapping):
            raise MobileActionsScoreError(f"manifest row {sample_id!r} metadata is not an object")
        prediction = predictions.get(sample_id)
        if prediction is None:
            raise MobileActionsScoreError(f"prediction artifact is missing ID {sample_id!r}")
        allowed_prediction_fields = {
            "id",
            "prediction_raw",
            "truncated",
            "generation_failure",
            "prompt_tokens",
            "generated_tokens",
        }
        extras = set(prediction).difference(allowed_prediction_fields)
        if extras:
            raise MobileActionsScoreError(
                f"prediction {sample_id!r} has unsupported fields {sorted(extras)!r}"
            )
        prediction_raw = prediction.get("prediction_raw")
        if prediction_raw is not None and not isinstance(prediction_raw, str):
            raise MobileActionsScoreError(
                f"prediction_raw for {sample_id!r} must be a string or null"
            )
        truncated = prediction.get("truncated", False)
        if not isinstance(truncated, bool):
            raise MobileActionsScoreError(f"truncated for {sample_id!r} must be boolean")
        generation_failure = prediction.get("generation_failure")
        if generation_failure is not None and not isinstance(generation_failure, str):
            raise MobileActionsScoreError(
                f"generation_failure for {sample_id!r} must be a string or null"
            )

        schemas = schemas_from_prompt(prompt)
        for schema in schemas:
            previous = registry.get(schema.name)
            if previous is not None and previous != schema:
                raise MobileActionsScoreError(
                    f"tool schema {schema.name!r} changes within the scored manifest"
                )
            registry[schema.name] = schema
        evaluator = ActionIREvaluator(schemas)
        call_names = metadata.get("call_names", [])
        scenario = call_names[0] if isinstance(call_names, list) and call_names else "unspecified"
        results.append(
            evaluator.evaluate(
                EvaluationCase(sample_id=sample_id, gold=target, scenario=scenario),
                prediction_raw,
                execution_hook=_execution_hook,
                truncated=truncated,
                generation_failure=generation_failure,
            )
        )

    unexpected = sorted(set(predictions).difference(seen_manifest_ids))
    if unexpected:
        raise MobileActionsScoreError(f"prediction artifact has unknown IDs {unexpected[:5]!r}")
    aggregate = ActionIREvaluator(registry).aggregate(results)
    return aggregate, tuple(results)


def read_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise MobileActionsScoreError(f"{path}: blank line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise MobileActionsScoreError(
                    f"{path}: invalid JSON on line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise MobileActionsScoreError(f"{path}: line {line_number} is not an object")
            rows.append(row)
    return rows


def aggregate_record(aggregate: AggregateEvaluation) -> dict[str, Any]:
    """Convert the frozen aggregate dataclass tree to strict JSON primitives."""

    return {
        "schema_version": MOBILE_ACTIONS_SCORER_VERSION,
        **asdict(aggregate),
    }


def write_scores(
    manifest_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write sample-level records and aggregate metrics without overwriting artifacts."""

    aggregate, samples = score_rows(read_jsonl(manifest_path), read_jsonl(predictions_path))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    sample_path = output / "sample_scores.jsonl"
    aggregate_path = output / "aggregate.json"
    with sample_path.open("x", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(
                json.dumps(
                    sample.to_record(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    aggregate_path.write_text(
        json.dumps(
            aggregate_record(aggregate),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"samples": sample_path, "aggregate": aggregate_path}


__all__ = [
    "MOBILE_ACTIONS_SCORER_VERSION",
    "MobileActionsScoreError",
    "aggregate_record",
    "read_jsonl",
    "schemas_from_prompt",
    "score_rows",
    "write_scores",
]
