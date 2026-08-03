"""Strict development scoring for PRESTO-derived Action IR v1 manifests.

PRESTO represents semantic arguments as an ordered, recursively nested ``slots``
array.  This scorer validates that representation without repairing model output,
aligns predictions by immutable example ID, and reports the upstream phenomenon
labels separately.  It is only for the derived Action IR task; it is not PRESTO's
native semantic-parse string metric.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .action_ir import (
    ActionIR,
    ActionIRError,
    ActionIRParseError,
    ActionIRValidationError,
    CallMode,
    Decision,
    ToolSchema,
    action_ir_equal,
    decode_json_object,
    validate_action_ir,
)

PRESTO_SCORER_VERSION_V1 = "barun-presto-action-ir-score-v1"
PRESTO_SCORER_VERSION = "barun-presto-action-ir-score-v2"
PRESTO_PHENOMENON_TAXONOMY_VERSION = "barun-presto-phenomenon-taxonomy-v2"
PRESTO_USER_REVISION_ALIAS_SET_VERSION = "barun-presto-user-revision-raw-aliases-v2"
PRESTO_USER_REVISION_RAW_LABELS_V2 = frozenset(
    {
        "cancel-action",
        "correct-action",
        "correct-argument",
        "within-turn-correction",
    }
)
_PRESTO_USER_REVISION_NORMALIZED_LABELS_V2 = frozenset(
    re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    for label in PRESTO_USER_REVISION_RAW_LABELS_V2
)
MAX_SLOT_DEPTH = 32
_SYMBOLS = frozenset({"audio", "infer_from_context", "visual"})
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*")

# Keep these frozen independently of the adapter's implementation so an adapter
# mapping change cannot silently change evaluator semantics.
ABSTAIN_ROOTS = frozenset({"Cancel", "Other"})
CALL_ROOTS = frozenset(
    {
        "Check_order_status",
        "Find_parking",
        "GetGenericBusinessType",
        "Get_bill",
        "Get_health_stats",
        "Get_list",
        "Get_message_content",
        "Get_note",
        "Get_product",
        "Get_security_price",
        "Open_app",
        "Play_game",
    }
)
CONFIRM_ROOTS = frozenset(
    {
        "Add_contact",
        "Add_item_to_list",
        "BuyEventTickets",
        "Cancel_ride",
        "Create_list",
        "Create_note",
        "Initiate_call",
        "Log_exercise",
        "Log_nutrition",
        "Order_menu_item",
        "Order_ride",
        "Pause_exercise",
        "Pay_bill",
        "Post_message",
        "Record_video",
        "Resume_exercise",
        "Send_digital_object",
        "Start_exercise",
        "Stop_exercise",
        "Take_photo",
    }
)


def _snake_case(value: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value.replace("-", "_"))
    second = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first)
    return re.sub(r"_+", "_", second).strip("_").casefold()


CALL_TOOLS = frozenset(_snake_case(root) for root in CALL_ROOTS)
CONFIRM_TOOLS = frozenset(_snake_case(root) for root in CONFIRM_ROOTS)
TOOL_TO_ROOT = {_snake_case(root): root for root in sorted(CALL_ROOTS | CONFIRM_ROOTS)}
_TOOL_REGISTRY = {
    tool: ToolSchema(
        name=tool,
        arguments={},
        additional_arguments=True,
        side_effecting=tool in CONFIRM_TOOLS,
    )
    for tool in sorted(TOOL_TO_ROOT)
}


class PrestoScoreError(ValueError):
    """A PRESTO manifest, prediction, or recursive slot violates the contract."""

    def __init__(self, message: str, *, code: str = "presto_schema", path: str = "$") -> None:
        super().__init__(f"{path}: {message}")
        self.code = code
        self.path = path


@dataclass(frozen=True, slots=True)
class _ScoredRow:
    record: Mapping[str, Any]
    exact: bool
    parse_valid: bool
    schema_valid: bool
    decision_correct: bool
    gold_decision: str
    predicted_decision: str | None
    phenomenon: str
    phenomenon_group: str
    root_intent: str
    contextual: bool
    false_call_on_gate: bool
    truncated: bool
    generation_failed: bool


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise PrestoScoreError(
            f"expected fields {sorted(expected)}, got {sorted(value)}",
            code="presto_fields",
            path=path,
        )


def _validate_slot_value(value: Any, *, path: str, depth: int) -> None:
    if depth > MAX_SLOT_DEPTH:
        raise PrestoScoreError(
            f"slot nesting exceeds {MAX_SLOT_DEPTH}",
            code="presto_slot_depth",
            path=path,
        )
    if not isinstance(value, Mapping):
        raise PrestoScoreError(
            "slot value must be an object",
            code="presto_slot_value_type",
            path=path,
        )
    keys = set(value)
    if keys == {"text"}:
        if not isinstance(value["text"], str):
            raise PrestoScoreError(
                "text value must be a string", code="presto_text_type", path=f"{path}.text"
            )
        return
    if keys == {"symbol"}:
        symbol = value["symbol"]
        if not isinstance(symbol, str) or symbol not in _SYMBOLS:
            raise PrestoScoreError(
                f"symbol must be one of {sorted(_SYMBOLS)}",
                code="presto_symbol",
                path=f"{path}.symbol",
            )
        return
    if keys == {"node", "slots"}:
        node = value["node"]
        if not isinstance(node, str) or _IDENTIFIER.fullmatch(node) is None:
            raise PrestoScoreError(
                "node must be a lowercase snake-case identifier",
                code="presto_node",
                path=f"{path}.node",
            )
        _validate_slots(value["slots"], path=f"{path}.slots", depth=depth + 1)
        return
    raise PrestoScoreError(
        "slot value must be exactly a text, symbol, or node/slots variant",
        code="presto_slot_variant",
        path=path,
    )


def _validate_slots(value: Any, *, path: str, depth: int = 0) -> None:
    if not isinstance(value, tuple | list):
        raise PrestoScoreError("slots must be an array", code="presto_slots_type", path=path)
    for index, slot in enumerate(value):
        slot_path = f"{path}[{index}]"
        if not isinstance(slot, Mapping):
            raise PrestoScoreError(
                "slot must be an object", code="presto_slot_type", path=slot_path
            )
        _exact_keys(slot, {"name", "value"}, slot_path)
        name = slot["name"]
        if not isinstance(name, str) or _IDENTIFIER.fullmatch(name) is None:
            raise PrestoScoreError(
                "slot name must be a lowercase snake-case identifier",
                code="presto_slot_name",
                path=f"{slot_path}.name",
            )
        _validate_slot_value(slot["value"], path=f"{slot_path}.value", depth=depth)


def validate_presto_action(action: ActionIR) -> ActionIR:
    """Apply the frozen PRESTO policy and recursive-slot schema to an Action IR."""

    if action.decision is Decision.ABSTAIN:
        return action
    if action.decision is Decision.CLARIFY:
        raise PrestoScoreError(
            "PRESTO-derived targets do not use CLARIFY",
            code="presto_decision",
            path="$.decision",
        )
    if action.mode is not CallMode.SINGLE or len(action.calls) != 1:
        raise PrestoScoreError(
            "PRESTO actions require exactly one SINGLE call",
            code="presto_call_shape",
            path="$.calls",
        )
    call = action.calls[0]
    expected = Decision.CALL if call.tool in CALL_TOOLS else Decision.CONFIRM
    if action.decision is not expected:
        raise PrestoScoreError(
            f"tool {call.tool!r} requires decision {expected.value}",
            code="presto_policy_decision",
            path="$.decision",
        )
    _exact_keys(call.args, {"slots"}, "$.calls[0].args")
    _validate_slots(call.args["slots"], path="$.calls[0].args.slots")
    return action


def parse_presto_action(raw: str) -> ActionIR:
    """Parse strict JSON, generic Action IR, and PRESTO's recursive payload."""

    action = validate_action_ir(decode_json_object(raw), _TOOL_REGISTRY)
    return validate_presto_action(action)


def _contextual_prompt(prompt: str) -> bool:
    try:
        context_text = prompt.split("\nCONTEXT ", 1)[1].split("\nDIALOGUE ", 1)[0]
        dialogue_text = prompt.split("\nDIALOGUE ", 1)[1].split("\n<user>\n", 1)[0]
        context = json.loads(context_text)
        dialogue = json.loads(dialogue_text)
    except (IndexError, json.JSONDecodeError) as error:
        raise PrestoScoreError("manifest prompt lacks strict PRESTO context fields") from error
    if not isinstance(context, dict) or set(context) != {"contacts", "lists", "notes"}:
        raise PrestoScoreError("manifest PRESTO context has unexpected shape")
    if not isinstance(dialogue, list) or any(
        not isinstance(turn, dict) or set(turn) != {"assistant", "user"} for turn in dialogue
    ):
        raise PrestoScoreError("manifest PRESTO dialogue has unexpected shape")
    if any(not isinstance(context[name], list) for name in ("contacts", "lists", "notes")):
        raise PrestoScoreError("manifest PRESTO context collections must be arrays")
    return bool(dialogue or context["contacts"] or context["lists"] or context["notes"])


def phenomenon_group(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if normalized in {"", "none", "no_phenomenon", "no_phenomena"}:
        return "no_phenomenon"
    # PRESTO's primary paper defines all four raw tags as members of the
    # broader user-revision phenomenon. The post-hoc audit remains stricter:
    # it accepts only the exact raw spellings present in the pinned evidence.
    if normalized in _PRESTO_USER_REVISION_NORMALIZED_LABELS_V2:
        return "revision"
    if "revision" in normalized:
        return "revision"
    if "disfluen" in normalized:
        return "disfluency"
    if "code" in normalized and "switch" in normalized:
        return "code_switching"
    return "other"


def _read_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise PrestoScoreError(f"blank JSONL row at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise PrestoScoreError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise PrestoScoreError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def _rate(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else 0.0,
    }


def _binary_f1(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def _bucket(rows: Sequence[_ScoredRow]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "ast_exact_match": _rate(sum(row.exact for row in rows), len(rows)),
        "schema_valid": _rate(sum(row.schema_valid for row in rows), len(rows)),
        "decision_accuracy": _rate(sum(row.decision_correct for row in rows), len(rows)),
        "truncation": _rate(sum(row.truncated for row in rows), len(rows)),
        "generation_failure": _rate(sum(row.generation_failed for row in rows), len(rows)),
    }


def _aggregate(rows: Sequence[_ScoredRow]) -> dict[str, Any]:
    by_phenomenon: dict[str, list[_ScoredRow]] = defaultdict(list)
    by_group: dict[str, list[_ScoredRow]] = defaultdict(list)
    by_root: dict[str, list[_ScoredRow]] = defaultdict(list)
    by_decision: dict[str, list[_ScoredRow]] = defaultdict(list)
    for row in rows:
        by_phenomenon[row.phenomenon].append(row)
        by_group[row.phenomenon_group].append(row)
        by_root[row.root_intent].append(row)
        by_decision[row.gold_decision].append(row)
    contextual = [row for row in rows if row.contextual]
    gold_abstain = [row for row in rows if row.gold_decision == Decision.ABSTAIN.value]
    abstain_tp = sum(
        row.gold_decision == Decision.ABSTAIN.value
        and row.predicted_decision == Decision.ABSTAIN.value
        for row in rows
    )
    abstain_fp = sum(
        row.gold_decision != Decision.ABSTAIN.value
        and row.predicted_decision == Decision.ABSTAIN.value
        for row in rows
    )
    abstain_fn = sum(
        row.gold_decision == Decision.ABSTAIN.value
        and row.predicted_decision != Decision.ABSTAIN.value
        for row in rows
    )
    gate_rows = [
        row for row in rows if row.gold_decision in {Decision.ABSTAIN.value, Decision.CONFIRM.value}
    ]
    return {
        "schema_version": PRESTO_SCORER_VERSION,
        "metric_scope": "derived_action_ir_not_native_presto_semantic_parse",
        "sample_count": len(rows),
        "ast_exact_match": _rate(sum(row.exact for row in rows), len(rows)),
        "parse_valid": _rate(sum(row.parse_valid for row in rows), len(rows)),
        "schema_valid": _rate(sum(row.schema_valid for row in rows), len(rows)),
        "decision_accuracy": _rate(sum(row.decision_correct for row in rows), len(rows)),
        "confirmation_exact": _bucket(by_decision.get(Decision.CONFIRM.value, []))[
            "ast_exact_match"
        ],
        "abstention": _binary_f1(abstain_tp, abstain_fp, abstain_fn),
        "abstention_gold_count": len(gold_abstain),
        "false_call_on_gate": _rate(
            sum(row.false_call_on_gate for row in gate_rows), len(gate_rows)
        ),
        "per_phenomenon": {name: _bucket(bucket) for name, bucket in sorted(by_phenomenon.items())},
        "headline_phenomenon_buckets": {
            name: _bucket(bucket) for name, bucket in sorted(by_group.items())
        },
        "contextual": _bucket(contextual),
        "per_root_intent": {name: _bucket(bucket) for name, bucket in sorted(by_root.items())},
        "per_gold_decision": {
            name: _bucket(bucket) for name, bucket in sorted(by_decision.items())
        },
        "failure_counts": dict(
            sorted(
                Counter(
                    str(row.record["prediction_error_code"])
                    for row in rows
                    if row.record["prediction_error_code"] is not None
                ).items()
            )
        ),
    }


def score_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    """Return aggregate and sample evidence for aligned PRESTO development rows."""

    predictions: dict[str, Mapping[str, Any]] = {}
    allowed_prediction_fields = {
        "generated_tokens",
        "generation_failure",
        "id",
        "prediction_raw",
        "prompt_tokens",
        "truncated",
    }
    for prediction in prediction_rows:
        sample_id = prediction.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise PrestoScoreError("every prediction requires a non-empty id")
        if sample_id in predictions:
            raise PrestoScoreError(f"duplicate prediction ID {sample_id!r}")
        extras = set(prediction).difference(allowed_prediction_fields)
        if extras:
            raise PrestoScoreError(
                f"prediction {sample_id!r} has unsupported fields {sorted(extras)}"
            )
        predictions[sample_id] = prediction

    scored: list[_ScoredRow] = []
    seen: set[str] = set()
    for manifest in manifest_rows:
        sample_id = manifest.get("id")
        prompt = manifest.get("prompt")
        target = manifest.get("target")
        metadata = manifest.get("metadata")
        if not isinstance(sample_id, str) or not sample_id:
            raise PrestoScoreError("every manifest row requires a non-empty id")
        if sample_id in seen:
            raise PrestoScoreError(f"duplicate manifest ID {sample_id!r}")
        seen.add(sample_id)
        if not isinstance(prompt, str) or not isinstance(target, str):
            raise PrestoScoreError(f"manifest row {sample_id!r} lacks prompt or target")
        if not isinstance(metadata, Mapping):
            raise PrestoScoreError(f"manifest row {sample_id!r} metadata must be an object")
        if metadata.get("source_split") != "dev" or metadata.get("derived_split") != "dev":
            raise PrestoScoreError(f"manifest row {sample_id!r} is not official development")
        phenomenon = metadata.get("linguistic_phenomenon")
        root_intent = metadata.get("root_intent")
        policy_decision = metadata.get("policy_decision")
        if not isinstance(phenomenon, str) or not isinstance(root_intent, str):
            raise PrestoScoreError(f"manifest row {sample_id!r} lacks PRESTO bucket metadata")
        if policy_decision not in {"ABSTAIN", "CALL", "CONFIRM"}:
            raise PrestoScoreError(f"manifest row {sample_id!r} has invalid policy decision")

        try:
            gold = parse_presto_action(target)
        except (ActionIRError, PrestoScoreError) as error:
            raise PrestoScoreError(f"invalid gold for {sample_id!r}: {error}") from error
        if gold.decision.value != policy_decision:
            raise PrestoScoreError(
                f"manifest row {sample_id!r} policy metadata disagrees with gold"
            )
        if gold.decision is Decision.ABSTAIN and root_intent not in ABSTAIN_ROOTS:
            raise PrestoScoreError(
                f"manifest row {sample_id!r} has a non-abstention root for ABSTAIN gold"
            )
        expected_root = (
            root_intent
            if gold.decision is Decision.ABSTAIN
            else TOOL_TO_ROOT.get(gold.calls[0].tool)
        )
        if expected_root != root_intent:
            raise PrestoScoreError(f"manifest row {sample_id!r} root metadata disagrees with gold")

        prediction = predictions.get(sample_id)
        if prediction is None:
            raise PrestoScoreError(f"prediction artifact is missing ID {sample_id!r}")
        prediction_raw = prediction.get("prediction_raw")
        truncated = prediction.get("truncated", False)
        generation_failure = prediction.get("generation_failure")
        if prediction_raw is not None and not isinstance(prediction_raw, str):
            raise PrestoScoreError(f"prediction_raw for {sample_id!r} must be string or null")
        if type(truncated) is not bool:
            raise PrestoScoreError(f"truncated for {sample_id!r} must be boolean")
        if generation_failure is not None and (
            not isinstance(generation_failure, str) or not generation_failure
        ):
            raise PrestoScoreError(
                f"generation_failure for {sample_id!r} must be a non-empty string or null"
            )

        parsed: ActionIR | None = None
        parse_valid = False
        schema_valid = False
        attempted_call = False
        error_code: str | None = None
        if prediction_raw is None:
            error_code = generation_failure or "missing_prediction"
        else:
            try:
                decoded = decode_json_object(prediction_raw)
                parse_valid = True
                # Count a parse-valid attempted CALL conservatively even when its tool or
                # payload later fails schema validation. This metric measures unsafe intent
                # on an ABSTAIN/CONFIRM row, not only calls an outer policy layer could run.
                attempted_call = decoded.get("decision") == Decision.CALL.value
                parsed = validate_action_ir(decoded, _TOOL_REGISTRY)
                validate_presto_action(parsed)
                schema_valid = True
            except ActionIRParseError as error:
                error_code = error.code
            except ActionIRValidationError as error:
                error_code = error.code
            except PrestoScoreError as error:
                error_code = error.code
        exact = bool(
            schema_valid
            and parsed is not None
            and not truncated
            and generation_failure is None
            and action_ir_equal(gold, parsed)
        )
        predicted_decision = parsed.decision.value if schema_valid and parsed is not None else None
        decision_correct = predicted_decision == gold.decision.value
        false_call = bool(gold.decision in {Decision.ABSTAIN, Decision.CONFIRM} and attempted_call)
        record = {
            "schema_version": PRESTO_SCORER_VERSION,
            "sample_id": sample_id,
            "phenomenon": phenomenon,
            "phenomenon_group": phenomenon_group(phenomenon),
            "root_intent": root_intent,
            "contextual": _contextual_prompt(prompt),
            "gold": gold.to_dict(),
            "gold_decision": gold.decision.value,
            "predicted_decision": predicted_decision,
            "prediction_raw": prediction_raw,
            "prediction": parsed.to_dict() if schema_valid and parsed is not None else None,
            "prediction_error_code": error_code,
            "generation_failure": generation_failure,
            "truncated": truncated,
            "parse_valid": parse_valid,
            "schema_valid": schema_valid,
            "ast_exact": exact,
            "decision_correct": decision_correct,
            "false_call_on_gate": false_call,
        }
        scored.append(
            _ScoredRow(
                record=record,
                exact=exact,
                parse_valid=parse_valid,
                schema_valid=schema_valid,
                decision_correct=decision_correct,
                gold_decision=gold.decision.value,
                predicted_decision=predicted_decision,
                phenomenon=phenomenon,
                phenomenon_group=phenomenon_group(phenomenon),
                root_intent=root_intent,
                contextual=bool(record["contextual"]),
                false_call_on_gate=false_call,
                truncated=truncated,
                generation_failed=generation_failure is not None,
            )
        )

    unexpected = sorted(set(predictions).difference(seen))
    if unexpected:
        raise PrestoScoreError(f"prediction artifact has unknown IDs {unexpected[:5]}")
    return _aggregate(scored), tuple(row.record for row in scored)


def write_scores(
    manifest_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write immutable sample scores and aggregate PRESTO development metrics."""

    aggregate, samples = score_rows(_read_jsonl(manifest_path), _read_jsonl(predictions_path))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    samples_path = destination / "sample_scores.jsonl"
    aggregate_path = destination / "aggregate.json"
    with samples_path.open("x", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    aggregate_path.write_text(
        json.dumps(
            aggregate,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"aggregate": aggregate_path, "samples": samples_path}


__all__ = [
    "PRESTO_PHENOMENON_TAXONOMY_VERSION",
    "PRESTO_SCORER_VERSION",
    "PRESTO_SCORER_VERSION_V1",
    "PRESTO_USER_REVISION_ALIAS_SET_VERSION",
    "PRESTO_USER_REVISION_RAW_LABELS_V2",
    "PrestoScoreError",
    "parse_presto_action",
    "phenomenon_group",
    "score_rows",
    "validate_presto_action",
    "write_scores",
]
