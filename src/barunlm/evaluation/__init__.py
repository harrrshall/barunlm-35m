"""Versioned BarunAction evaluation API with side-effect-free discovery.

Concrete modules stay lazy so importing a CPU-only contract does not also import
the Torch-backed generation runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_ACTION_IR_EXPORTS = (
    "MAX_JSON_NESTING",
    "ActionIR",
    "ActionIRError",
    "ActionIRParseError",
    "ActionIRValidationError",
    "CallMode",
    "Decision",
    "JSONType",
    "ToolCall",
    "ToolSchema",
    "ValueSchema",
    "action_ir_equal",
    "canonical_json_value",
    "decode_json_object",
    "parse_action_ir",
    "validate_action_ir",
)
_EVALUATOR_EXPORTS = (
    "EVALUATOR_VERSION",
    "PRF",
    "ActionIREvaluator",
    "AggregateEvaluation",
    "EvaluationCase",
    "ExecutionHook",
    "ExecutionResult",
    "FalseActionClass",
    "Rate",
    "RiskCoveragePoint",
    "SampleEvaluation",
    "ToolMetrics",
)
_GENERATION_EXPORTS = (
    "GENERATION_VERSION",
    "INT8_GENERATION_VERSION",
    "GenerationError",
    "GenerationSummary",
    "generate_manifest",
    "load_verified_model",
    "verify_checkpoint",
)
_MOBILE_EXPORTS = (
    "MOBILE_ACTIONS_SCORER_VERSION",
    "MobileActionsScoreError",
    "schemas_from_prompt",
    "score_rows",
    "write_scores",
)
_PRESTO_EXPORTS = (
    "PRESTO_SCORER_VERSION",
    "PrestoScoreError",
    "parse_presto_action",
    "phenomenon_group",
    "validate_presto_action",
)

_LAZY_EXPORTS = {
    **{name: (".action_ir", name) for name in _ACTION_IR_EXPORTS},
    **{name: (".evaluator", name) for name in _EVALUATOR_EXPORTS},
    **{name: (".generation", name) for name in _GENERATION_EXPORTS},
    **{name: (".mobile_actions", name) for name in _MOBILE_EXPORTS},
    **{name: (".presto", name) for name in _PRESTO_EXPORTS},
    "score_presto_rows": (".presto", "score_rows"),
    "write_presto_scores": (".presto", "write_scores"),
}

__all__ = [
    "EVALUATOR_VERSION",
    "GENERATION_VERSION",
    "INT8_GENERATION_VERSION",
    "MAX_JSON_NESTING",
    "MOBILE_ACTIONS_SCORER_VERSION",
    "PRESTO_SCORER_VERSION",
    "PRF",
    "ActionIR",
    "ActionIRError",
    "ActionIREvaluator",
    "ActionIRParseError",
    "ActionIRValidationError",
    "AggregateEvaluation",
    "CallMode",
    "Decision",
    "EvaluationCase",
    "ExecutionHook",
    "ExecutionResult",
    "FalseActionClass",
    "GenerationError",
    "GenerationSummary",
    "JSONType",
    "MobileActionsScoreError",
    "PrestoScoreError",
    "Rate",
    "RiskCoveragePoint",
    "SampleEvaluation",
    "ToolCall",
    "ToolMetrics",
    "ToolSchema",
    "ValueSchema",
    "action_ir_equal",
    "canonical_json_value",
    "decode_json_object",
    "generate_manifest",
    "load_verified_model",
    "parse_action_ir",
    "parse_presto_action",
    "phenomenon_group",
    "schemas_from_prompt",
    "score_presto_rows",
    "score_rows",
    "validate_action_ir",
    "validate_presto_action",
    "verify_checkpoint",
    "write_presto_scores",
    "write_scores",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
