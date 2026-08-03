"""Exact Action IR fidelity smoke tests for float and dynamic-int8 checkpoints."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from barunlm.evaluation.generation import REQUIRED_CHECKPOINT_FILES
from barunlm.quantization import verify_int8_checkpoint

from .inference import BarunActionCompiler, InferenceOutcome, validate_action_output
from .schema import PreparedInput, prepare_input

INT8_SMOKE_CASES_VERSION = "barunaction-int8-smoke-cases-v1"
INT8_SMOKE_REPORT_VERSION = "barunaction-int8-smoke-report-v1"


class QuantizationSmokeError(ValueError):
    """The requested fidelity comparison is malformed or compares different sources."""


@dataclass(frozen=True, slots=True)
class ActionIRSmokeCase:
    case_id: str
    prepared: PreparedInput
    tool_schemas: Any
    context: Any
    expected_action: Mapping[str, Any]


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    missing = expected.difference(value)
    unknown = set(value).difference(expected)
    if missing or unknown:
        raise QuantizationSmokeError(
            f"{path} fields differ: missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
        )


def parse_int8_smoke_cases(value: Any) -> tuple[ActionIRSmokeCase, ...]:
    if not isinstance(value, Mapping):
        raise QuantizationSmokeError("smoke case artifact must be an object")
    _exact_fields(value, {"cases", "schema_version"}, path="$")
    if value["schema_version"] != INT8_SMOKE_CASES_VERSION:
        raise QuantizationSmokeError("unsupported int8 smoke case version")
    raw_cases = value["cases"]
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)) or not raw_cases:
        raise QuantizationSmokeError("$.cases must be a non-empty array")
    cases: list[ActionIRSmokeCase] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_cases):
        path = f"$.cases[{index}]"
        if not isinstance(raw, Mapping):
            raise QuantizationSmokeError(f"{path} must be an object")
        _exact_fields(
            raw,
            {"context", "expected_action", "id", "now", "request", "tool_schemas"},
            path=path,
        )
        case_id = raw["id"]
        if not isinstance(case_id, str) or not case_id:
            raise QuantizationSmokeError(f"{path}.id must be a non-empty string")
        if case_id in seen_ids:
            raise QuantizationSmokeError(f"duplicate int8 smoke case id {case_id!r}")
        seen_ids.add(case_id)
        prepared = prepare_input(
            request=raw["request"],
            tool_schemas=raw["tool_schemas"],
            context=raw["context"],
            now=raw["now"],
        )
        expected = raw["expected_action"]
        if not isinstance(expected, Mapping):
            raise QuantizationSmokeError(f"{path}.expected_action must be an object")
        expected_raw = _canonical_json(expected)
        validated = validate_action_output(expected_raw, declarations=prepared.declarations)
        if validated.action is None:
            assert validated.error is not None
            raise QuantizationSmokeError(
                f"{path}.expected_action is invalid: "
                f"{validated.error.code} at {validated.error.path}"
            )
        cases.append(
            ActionIRSmokeCase(
                case_id=case_id,
                prepared=prepared,
                tool_schemas=raw["tool_schemas"],
                context=raw["context"],
                expected_action=validated.action.to_dict(),
            )
        )
    return tuple(cases)


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise QuantizationSmokeError("expected Action IR must be finite JSON") from error


def _outcome_record(outcome: InferenceOutcome) -> dict[str, Any]:
    return {
        "action": outcome.action.to_dict() if outcome.action is not None else None,
        "error": outcome.error.to_dict() if outcome.error is not None else None,
        "generated_tokens": outcome.generated_tokens,
        "ok": outcome.ok,
        "prompt_sha256": outcome.prompt_sha256,
        "prompt_tokens": outcome.prompt_tokens,
        "raw_output_sha256": (
            hashlib.sha256(outcome.raw_output.encode("utf-8")).hexdigest()
            if outcome.raw_output is not None
            else None
        ),
    }


def compare_int8_action_ir(
    *,
    float_checkpoint: str | Path,
    expected_float_sha256: Mapping[str, str],
    int8_checkpoint: str | Path,
    expected_int8_manifest_sha256: str,
    cases: tuple[ActionIRSmokeCase, ...],
    max_new_tokens: int = 192,
) -> dict[str, Any]:
    """Compare exact expected Action IR on the same source weights and prompts."""

    if not cases:
        raise QuantizationSmokeError("at least one Action IR smoke case is required")
    info = verify_int8_checkpoint(
        int8_checkpoint,
        expected_manifest_sha256=expected_int8_manifest_sha256,
    )
    required_float_hashes = {
        name: expected_float_sha256[name]
        for name in REQUIRED_CHECKPOINT_FILES
        if name in expected_float_sha256
    }
    if set(required_float_hashes) != set(REQUIRED_CHECKPOINT_FILES):
        raise QuantizationSmokeError("float checkpoint hash map omits required model files")
    if dict(info.source_checkpoint_sha256) != required_float_hashes:
        raise QuantizationSmokeError(
            "int8 source hashes differ from the float checkpoint selected for comparison"
        )

    started = time.perf_counter()
    float_compiler = BarunActionCompiler(
        float_checkpoint,
        expected_sha256=expected_float_sha256,
        checkpoint_format="float",
        device="cpu",
    )
    float_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    int8_compiler = BarunActionCompiler(
        int8_checkpoint,
        checkpoint_format="int8",
        expected_int8_manifest_sha256=expected_int8_manifest_sha256,
        device="cpu",
    )
    int8_load_seconds = time.perf_counter() - started

    records: list[dict[str, Any]] = []
    all_exact = True
    float_inference_seconds = 0.0
    int8_inference_seconds = 0.0
    for case in cases:
        started = time.perf_counter()
        float_outcome = float_compiler.infer(
            request=case.prepared.request,
            tool_schemas=case.tool_schemas,
            context=case.context,
            now=case.prepared.now,
            max_new_tokens=max_new_tokens,
        )
        float_seconds = time.perf_counter() - started
        float_inference_seconds += float_seconds
        started = time.perf_counter()
        int8_outcome = int8_compiler.infer(
            request=case.prepared.request,
            tool_schemas=case.tool_schemas,
            context=case.context,
            now=case.prepared.now,
            max_new_tokens=max_new_tokens,
        )
        int8_seconds = time.perf_counter() - started
        int8_inference_seconds += int8_seconds

        float_action = float_outcome.action.to_dict() if float_outcome.action is not None else None
        int8_action = int8_outcome.action.to_dict() if int8_outcome.action is not None else None
        float_exact = float_action == case.expected_action
        int8_exact = int8_action == case.expected_action
        cross_exact = float_action is not None and float_action == int8_action
        raw_output_equal = (
            float_outcome.raw_output is not None
            and float_outcome.raw_output == int8_outcome.raw_output
        )
        case_exact = float_exact and int8_exact and cross_exact
        all_exact &= case_exact
        records.append(
            {
                "case_id": case.case_id,
                "cross_action_ir_exact": cross_exact,
                "expected_action": dict(case.expected_action),
                "float": _outcome_record(float_outcome),
                "float_expected_exact": float_exact,
                "float_inference_seconds": float_seconds,
                "int8": _outcome_record(int8_outcome),
                "int8_expected_exact": int8_exact,
                "int8_inference_seconds": int8_seconds,
                "raw_output_equal": raw_output_equal,
            }
        )

    return {
        "all_action_ir_exact": all_exact,
        "cases": records,
        "checkpoint_size": {
            "int8_package_bytes": info.package_bytes,
            "int8_payload_bytes": info.quantized_payload_bytes,
            "payload_reduction_fraction": info.reduction_fraction,
            "source_payload_bytes": info.source_payload_bytes,
        },
        "int8_manifest_sha256": info.manifest_sha256,
        "runtime": {
            "float_inference_seconds": float_inference_seconds,
            "float_load_seconds": float_load_seconds,
            "int8_inference_seconds": int8_inference_seconds,
            "int8_load_seconds": int8_load_seconds,
            "platform_machine": platform.machine(),
            "platform_system": platform.system(),
            "qengine": info.qengine,
            "timing_interpretation": "provisional_current_host_only_not_target_device_evidence",
            "torch_num_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
        },
        "schema_version": INT8_SMOKE_REPORT_VERSION,
        "source_checkpoint_sha256": dict(sorted(required_float_hashes.items())),
    }


__all__ = [
    "INT8_SMOKE_CASES_VERSION",
    "INT8_SMOKE_REPORT_VERSION",
    "ActionIRSmokeCase",
    "QuantizationSmokeError",
    "compare_int8_action_ir",
    "parse_int8_smoke_cases",
]
