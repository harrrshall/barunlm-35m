"""Fail-closed scoring and gating for the construction-only PlanIR screen.

This module accepts only the three matched construction-screen arms.  It never
loads a dataset itself.  Direct arms A/B are scored as raw Action IR.  Treatment
arm C is scored only after the untouched raw output is compiled against the exact
canonical prompt and its independently rebuilt reference table.

Catastrophic-action evidence is never accepted from prediction rows.  For each
schema-valid CALL, the scorer applies the frozen Mobile exact-match simulator
locally; invalid and non-CALL outputs record that no call was assessed.  Because
the construction screen contains only CALL golds, this is narrow mechanism
evidence rather than a product safety evaluation.

``PromptEvidence`` establishes consistency among supplied bytes and hashes.  It
does *not* establish that a model saw those bytes.  The separate evidence receipt
therefore binds the prompt population to the raw generation, checkpoint, decoding,
renderer, compiler, evaluator, and configuration identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Literal

from .action_ir import (
    ActionIR,
    ActionIRError,
    ActionIRParseError,
    Decision,
    action_ir_equal,
    decode_json_object,
    validate_action_ir,
)
from .evaluator import EVALUATOR_VERSION
from .grounded_planir import MOBILE_TOOL_SCHEMAS
from .grounded_planir_v2 import (
    ACTION_PROMPT_CONTRACT,
    PROMPT_CONTRACT,
    PROMPT_RENDERER_VERSION,
    PromptEvidence,
    ReferenceTable,
    build_reference_table,
    compile_mobile_plan,
    make_prompt_evidence,
    parse_construction_prompt,
)

SFT_EXAMPLE_VERSION = "barun-sft-example-v1"
SCREEN_MANIFEST_ROW_VERSION = "barun-mobile-planir-screen-row-v1"
SCREEN_METADATA_VERSION = "barun-mobile-planir-screen-row-v1"
SCREEN_SPLIT_VERSION = "barun-mobile-construction-component-screen-split-v1"
SCREEN_PREDICTION_ROW_VERSION = "barun-mobile-planir-screen-prediction-row-v1"
SCREEN_SCORE_VERSION = "barun-mobile-planir-screen-score-v1"
SCREEN_GATE_VERSION = "barun-mobile-planir-screen-gate-v1"
SCREEN_RECEIPT_VERSION = "barun-mobile-planir-screen-evidence-receipt-v1"

PINNED_MANIFEST_ROWS = 3_444
PINNED_SOURCE_ROWS = 1_148
PINNED_MANIFEST_CONTENT_SHA256 = "c84d25ecd906b54b48fc4994543592cf8f79bdf4dacc3018bdf59b9c9f77583c"
PINNED_PROMPT_POPULATION_SHA256 = "60fbfb1bfe14cb6474e2c4a14f54eb40cfd1feb8cd0f8d08f86641cf4ea5de1f"
PINNED_TABLE_POPULATION_SHA256 = "f8676258ed6eb997680e3d39d9b5091419bdd1f7cfaa6d9f3c0d83f16f252585"
PINNED_POPULATION_SHA256 = "ec922c1f18c55535319f6bf0e1e7dd68461456e5cb2a3472404ce2d874a28642"

ARMS = ("A", "B", "C")
SEEDS = (17, 29, 43)
TOOL_NAMES = tuple(schema.name for schema in MOBILE_TOOL_SCHEMAS)
CALENDAR_SUBSET = "calendar"
NONCALENDAR_SUBSET = "noncalendar"
SINGLE_CALL_SUBSET = "single_call"
MULTI_CALL_SUBSET = "multi_call"
PREDEFINED_SUBSETS = (
    CALENDAR_SUBSET,
    NONCALENDAR_SUBSET,
    SINGLE_CALL_SUBSET,
    MULTI_CALL_SUBSET,
    *(f"contains_tool:{tool}" for tool in TOOL_NAMES),
)

Arm = Literal["A", "B", "C"]
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MANIFEST_FIELDS = frozenset({"schema_version", "id", "prompt", "target", "metadata"})
_SCREEN_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "split_version",
        "arm",
        "population",
        "source_id",
        "source_content_sha256",
        "source_metadata_sha256",
        "source_prompt_sha256",
        "source_target_sha256",
        "output_prompt_sha256",
        "output_target_sha256",
        "prompt_evidence",
    }
)
_PROMPT_EVIDENCE_FIELDS = frozenset(
    {
        "renderer_version",
        "prompt_contract",
        "prompt_sha256",
        "request_sha256",
        "now",
        "table_sha256",
    }
)
_PREDICTION_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "source_id",
        "arm",
        "seed",
        "prompt_sha256",
        "table_sha256",
        "checkpoint_sha256",
        "decoding_sha256",
        "prediction_raw",
        "prediction_raw_sha256",
        "missing",
        "truncated",
        "generation_failure",
    }
)


class MobilePlanIRScreenError(ValueError):
    """A construction manifest, prediction, metric, gate, or receipt is invalid."""


def _exact_fields(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise MobilePlanIRScreenError(
            f"{label} fields differ; missing={missing!r}, extra={extra!r}"
        )


def _canonical_json(value: object, *, label: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise MobilePlanIRScreenError(f"{label} is not finite strict JSON") from exc


def _sha256_text(value: str) -> str:
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise MobilePlanIRScreenError("text hash input is not valid UTF-8") from exc
    return hashlib.sha256(encoded).hexdigest()


def _strict_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MobilePlanIRScreenError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MobilePlanIRScreenError(f"{label} must be a non-empty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise MobilePlanIRScreenError(f"{label} must be valid UTF-8") from exc
    return value


def _strict_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise MobilePlanIRScreenError(f"{label} must be boolean")
    return value


def _strict_arm(value: object, *, label: str) -> Arm:
    if value not in ARMS or not isinstance(value, str):
        raise MobilePlanIRScreenError(f"{label} must be exactly A, B, or C")
    return value  # type: ignore[return-value]


def _hash_records(records: Sequence[Mapping[str, object]], *, label: str) -> str:
    canonical = "".join(f"{_canonical_json(row, label=label)}\n" for row in records)
    return _sha256_text(canonical)


def _detached_rows(
    rows: Sequence[Mapping[str, object]], *, label: str
) -> tuple[dict[str, object], ...]:
    """Take one canonical caller-detached JSON snapshot.

    Exact built-in containers prevent a custom sequence from exposing different
    populations to iteration and later checks.  The JSON round trip detaches all
    nested caller-owned dicts and lists before validation or membership use.
    """

    if type(rows) not in {list, tuple}:
        raise MobilePlanIRScreenError(f"{label} must be an exact list or tuple")

    def detach(value: object, path: str) -> object:
        if value is None or type(value) in {bool, int, str}:
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise MobilePlanIRScreenError(f"{path} must be finite JSON")
            return value
        if type(value) is list:
            return [detach(child, f"{path}[{index}]") for index, child in enumerate(value)]
        if type(value) is dict:
            output: dict[str, object] = {}
            for key, child in value.items():
                if type(key) is not str:
                    raise MobilePlanIRScreenError(f"{path} object keys must be strings")
                output[key] = detach(child, f"{path}.{key}")
            return output
        raise MobilePlanIRScreenError(
            f"{path} must use exact built-in JSON containers and scalar types"
        )

    try:
        detached = [detach(row, f"{label}[{index}]") for index, row in enumerate(rows)]
    except RecursionError as exc:
        raise MobilePlanIRScreenError(f"{label} nesting is too deep") from exc
    if any(type(row) is not dict for row in detached):
        raise MobilePlanIRScreenError(f"{label} must contain only exact JSON objects")
    _canonical_json(detached, label=label)
    return tuple(detached)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ExactRate:
    """An exact success count; no rounded floating-point value is stored."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise TypeError("ExactRate counts must be integers")
        if self.denominator < 0 or not 0 <= self.numerator <= self.denominator:
            raise ValueError("ExactRate requires 0 <= numerator <= denominator")

    def to_record(self) -> dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}


@dataclass(frozen=True, slots=True)
class SubsetMetrics:
    sample_count: int
    ast_exact: ExactRate
    argument_value_exact: ExactRate

    def __post_init__(self) -> None:
        if type(self.sample_count) is not int or self.sample_count < 0:
            raise ValueError("subset sample_count must be a non-negative integer")
        if (
            self.ast_exact.denominator != self.sample_count
            or self.argument_value_exact.denominator != self.sample_count
        ):
            raise ValueError("subset metric denominators must equal sample_count")

    def to_record(self) -> dict[str, object]:
        return {
            "argument_value_exact": self.argument_value_exact.to_record(),
            "ast_exact": self.ast_exact.to_record(),
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class RunMetrics:
    arm: Arm
    seed: int
    population_sha256: str
    checkpoint_sha256: str
    decoding_sha256: str
    sample_count: int
    ast_exact: ExactRate
    argument_value_exact: ExactRate
    raw_json_parse: ExactRate
    action_ir_schema_valid: ExactRate
    compiler_success: ExactRate | None
    conditional_compiled_schema_valid: ExactRate | None
    failure_counts: Mapping[str, int]
    subsets: Mapping[str, SubsetMetrics]

    def __post_init__(self) -> None:
        _strict_arm(self.arm, label="run arm")
        if self.seed not in SEEDS or type(self.seed) is not int:
            raise ValueError(f"run seed must be one of {SEEDS!r}")
        _strict_sha256(self.population_sha256, label="population_sha256")
        _strict_sha256(self.checkpoint_sha256, label="checkpoint_sha256")
        _strict_sha256(self.decoding_sha256, label="decoding_sha256")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("run sample_count must be a positive integer")
        for name, rate in (
            ("ast_exact", self.ast_exact),
            ("argument_value_exact", self.argument_value_exact),
            ("raw_json_parse", self.raw_json_parse),
            ("action_ir_schema_valid", self.action_ir_schema_valid),
        ):
            if not isinstance(rate, ExactRate) or rate.denominator != self.sample_count:
                raise ValueError(f"{name} denominator must equal run sample_count")
        if self.arm == "C":
            if self.compiler_success is None or self.conditional_compiled_schema_valid is None:
                raise ValueError("arm C requires compiler and conditional schema metrics")
            if self.compiler_success.denominator != self.sample_count:
                raise ValueError("compiler denominator must equal run sample_count")
            if (
                self.conditional_compiled_schema_valid.denominator
                != self.compiler_success.numerator
            ):
                raise ValueError("conditional schema denominator must equal compiler successes")
        elif (
            self.compiler_success is not None or self.conditional_compiled_schema_valid is not None
        ):
            raise ValueError("direct arms cannot report compiler metrics")

        expected_failures = {
            "raw_parse_failure",
            "action_ir_schema_failure",
            "compiler_failure",
            "missing",
            "truncation",
            "generation_failure",
            "catastrophic_unauthorized_action",
        }
        if set(self.failure_counts) != expected_failures:
            raise ValueError("run failure_counts has the wrong keys")
        if any(type(value) is not int or value < 0 for value in self.failure_counts.values()):
            raise ValueError("run failure counts must be non-negative integers")
        if self.failure_counts["raw_parse_failure"] != (
            self.sample_count - self.raw_json_parse.numerator
        ):
            raise ValueError("raw parse failure count disagrees with its exact rate")
        if self.failure_counts["action_ir_schema_failure"] != (
            self.sample_count - self.action_ir_schema_valid.numerator
        ):
            raise ValueError("schema failure count disagrees with its exact rate")
        expected_compiler_failures = (
            self.sample_count - self.compiler_success.numerator
            if self.compiler_success is not None
            else 0
        )
        if self.failure_counts["compiler_failure"] != expected_compiler_failures:
            raise ValueError("compiler failure count disagrees with its exact rate")
        if set(self.subsets) != set(PREDEFINED_SUBSETS):
            raise ValueError("run subsets do not match the frozen predefined subset set")
        frozen_failures = MappingProxyType(dict(sorted(self.failure_counts.items())))
        frozen_subsets = MappingProxyType(dict(sorted(self.subsets.items())))
        object.__setattr__(self, "failure_counts", frozen_failures)
        object.__setattr__(self, "subsets", frozen_subsets)

    def to_record(self) -> dict[str, object]:
        return {
            "action_ir_schema_valid": self.action_ir_schema_valid.to_record(),
            "argument_value_exact": self.argument_value_exact.to_record(),
            "arm": self.arm,
            "ast_exact": self.ast_exact.to_record(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "compiler_success": (
                None if self.compiler_success is None else self.compiler_success.to_record()
            ),
            "conditional_compiled_schema_valid": (
                None
                if self.conditional_compiled_schema_valid is None
                else self.conditional_compiled_schema_valid.to_record()
            ),
            "decoding_sha256": self.decoding_sha256,
            "failure_counts": dict(self.failure_counts),
            "population_sha256": self.population_sha256,
            "raw_json_parse": self.raw_json_parse.to_record(),
            "sample_count": self.sample_count,
            "seed": self.seed,
            "subsets": {name: metric.to_record() for name, metric in self.subsets.items()},
        }


@dataclass(frozen=True, slots=True)
class RowEvidence:
    sample_id: str
    source_id: str
    arm: Arm
    seed: int
    prompt_contract: str
    prompt_sha256: str
    table_sha256: str
    request_sha256: str
    checkpoint_sha256: str
    decoding_sha256: str
    prediction_raw: str | None
    prediction_raw_sha256: str | None
    prediction_action_ir: str | None
    prediction_action_ir_sha256: str | None
    raw_json_parse: bool
    action_ir_schema_valid: bool
    compiler_attempted: bool
    compiler_success: bool | None
    compiler_error_code: str | None
    compiler_error_path: str | None
    missing: bool
    truncated: bool
    generation_failure: str | None
    assessed_call: bool
    simulator_success: bool | None
    catastrophic_unauthorized_action: bool
    ast_exact: bool
    argument_value_exact: bool
    subsets: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "action_ir_schema_valid": self.action_ir_schema_valid,
            "assessed_call": self.assessed_call,
            "argument_value_exact": self.argument_value_exact,
            "arm": self.arm,
            "ast_exact": self.ast_exact,
            "catastrophic_unauthorized_action": self.catastrophic_unauthorized_action,
            "checkpoint_sha256": self.checkpoint_sha256,
            "compiler_attempted": self.compiler_attempted,
            "compiler_error_code": self.compiler_error_code,
            "compiler_error_path": self.compiler_error_path,
            "compiler_success": self.compiler_success,
            "decoding_sha256": self.decoding_sha256,
            "direct_action_ir_schema_valid": (
                self.action_ir_schema_valid if self.arm in {"A", "B"} else None
            ),
            "full_table_sha256": self.table_sha256,
            "generation_failure": self.generation_failure,
            "missing": self.missing,
            "prediction_action_ir": self.prediction_action_ir,
            "prediction_action_ir_sha256": self.prediction_action_ir_sha256,
            "prediction_raw": self.prediction_raw,
            "prediction_raw_sha256": self.prediction_raw_sha256,
            "prompt_contract": self.prompt_contract,
            "prompt_sha256": self.prompt_sha256,
            "compiled_action_ir_schema_valid": (
                self.action_ir_schema_valid if self.arm == "C" else None
            ),
            "raw_plan_json_parse": self.raw_json_parse if self.arm == "C" else None,
            "raw_json_parse": self.raw_json_parse,
            "request_sha256": self.request_sha256,
            "sample_id": self.sample_id,
            "seed": self.seed,
            "simulator_success": self.simulator_success,
            "source_id": self.source_id,
            "subsets": list(self.subsets),
            "table_sha256": self.table_sha256,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class ScreenEvaluation:
    schema_version: str
    manifest_content_sha256: str
    prompt_population_sha256: str
    table_population_sha256: str
    population_sha256: str
    rows: tuple[RowEvidence, ...]
    runs: tuple[RunMetrics, ...]
    pinned_manifest_enforced: bool

    def __post_init__(self) -> None:
        if self.schema_version != SCREEN_SCORE_VERSION:
            raise ValueError(f"screen score version must equal {SCREEN_SCORE_VERSION!r}")
        if type(self.pinned_manifest_enforced) is not bool:
            raise TypeError("pinned_manifest_enforced must be boolean")
        for field in (
            "manifest_content_sha256",
            "prompt_population_sha256",
            "table_population_sha256",
            "population_sha256",
        ):
            _strict_sha256(getattr(self, field), label=field)
        if type(self.rows) is not tuple or any(
            not isinstance(row, RowEvidence) for row in self.rows
        ):
            raise TypeError("screen score rows must be an exact tuple of RowEvidence")
        if type(self.runs) is not tuple or any(
            not isinstance(run, RunMetrics) for run in self.runs
        ):
            raise TypeError("screen score runs must be an exact tuple of RunMetrics")

    def to_record(self) -> dict[str, object]:
        return {
            "manifest_content_sha256": self.manifest_content_sha256,
            "pinned_manifest_enforced": self.pinned_manifest_enforced,
            "population_sha256": self.population_sha256,
            "prompt_evidence_scope": (
                "consistency_only_not_proof_of_model_presentation; use the bound generation receipt"
            ),
            "prompt_population_sha256": self.prompt_population_sha256,
            "rows": [row.to_record() for row in self.rows],
            "runs": [run.to_record() for run in self.runs],
            "safety_scope": (
                "CALL-only construction screen with deterministic exact-match call assessment; "
                "zero catastrophic count is narrow or vacuous evidence, not a product safety result"
            ),
            "schema_version": self.schema_version,
            "table_population_sha256": self.table_population_sha256,
        }


@dataclass(frozen=True, slots=True)
class ScreenGate:
    schema_version: str
    passed: bool
    checks: Mapping[str, bool]
    observed: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.passed) is not bool or any(
            type(value) is not bool for value in self.checks.values()
        ):
            raise TypeError("gate decisions must be booleans")
        object.__setattr__(self, "checks", MappingProxyType(dict(sorted(self.checks.items()))))
        object.__setattr__(self, "observed", MappingProxyType(dict(self.observed)))

    def to_record(self) -> dict[str, object]:
        return {
            "checks": dict(self.checks),
            "observed": dict(self.observed),
            "passed": self.passed,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class _ManifestIdentity:
    sample_id: str
    source_id: str
    arm: Arm
    prompt: str
    target: str
    source_content_sha256: str
    source_metadata_sha256: str
    source_prompt_sha256: str
    source_target_sha256: str
    prompt_sha256: str
    table_sha256: str
    prompt_evidence: PromptEvidence
    request: str
    now: str
    system_body: str
    table: ReferenceTable


@dataclass(frozen=True, slots=True)
class _ManifestPopulation:
    rows: tuple[_ManifestIdentity, ...]
    gold_by_source: Mapping[str, ActionIR]
    subsets_by_source: Mapping[str, tuple[str, ...]]
    content_sha256: str
    arm_content_sha256: Mapping[str, str]
    prompt_population_sha256: str
    table_population_sha256: str
    population_sha256: str


@dataclass(frozen=True, slots=True)
class _PredictionIdentity:
    sample_id: str
    source_id: str
    arm: Arm
    seed: int
    prompt_sha256: str
    table_sha256: str
    checkpoint_sha256: str
    decoding_sha256: str
    prediction_raw: str | None
    prediction_raw_sha256: str | None
    missing: bool
    truncated: bool
    generation_failure: str | None


@dataclass(frozen=True, slots=True)
class _PredictionPopulation:
    rows: tuple[_PredictionIdentity, ...]
    content_sha256: str


def _enforce_pinned_manifest(manifest: _ManifestPopulation, *, enforce_pinned: bool) -> None:
    if type(enforce_pinned) is not bool:
        raise MobilePlanIRScreenError("enforce_pinned must be boolean")
    if not enforce_pinned:
        return
    observed: dict[str, int | str] = {
        "manifest_rows": len(manifest.rows),
        "source_rows": len(manifest.gold_by_source),
        "manifest_content_sha256": manifest.content_sha256,
        "prompt_population_sha256": manifest.prompt_population_sha256,
        "table_population_sha256": manifest.table_population_sha256,
        "population_sha256": manifest.population_sha256,
    }
    expected: dict[str, int | str] = {
        "manifest_rows": PINNED_MANIFEST_ROWS,
        "source_rows": PINNED_SOURCE_ROWS,
        "manifest_content_sha256": PINNED_MANIFEST_CONTENT_SHA256,
        "prompt_population_sha256": PINNED_PROMPT_POPULATION_SHA256,
        "table_population_sha256": PINNED_TABLE_POPULATION_SHA256,
        "population_sha256": PINNED_POPULATION_SHA256,
    }
    if observed != expected:
        raise MobilePlanIRScreenError(
            "screen manifest does not equal the frozen 1,148-source materialization identity"
        )


def _prompt_evidence_from_mapping(value: object, *, label: str) -> PromptEvidence:
    if type(value) is not dict:
        raise MobilePlanIRScreenError(f"{label} must be an exact JSON object")
    _exact_fields(value, _PROMPT_EVIDENCE_FIELDS, label)
    return PromptEvidence(
        renderer_version=_strict_nonempty(
            value["renderer_version"], label=f"{label}.renderer_version"
        ),
        prompt_contract=_strict_nonempty(
            value["prompt_contract"], label=f"{label}.prompt_contract"
        ),
        prompt_sha256=_strict_sha256(value["prompt_sha256"], label=f"{label}.prompt_sha256"),
        request_sha256=_strict_sha256(value["request_sha256"], label=f"{label}.request_sha256"),
        now=_strict_nonempty(value["now"], label=f"{label}.now"),
        table_sha256=_strict_sha256(value["table_sha256"], label=f"{label}.table_sha256"),
    )


def _classify_manifest_row(row: Mapping[str, object], *, row_number: int) -> _ManifestIdentity:
    if type(row) is not dict:
        raise MobilePlanIRScreenError(f"manifest row {row_number} must be an exact JSON object")
    _exact_fields(row, _MANIFEST_FIELDS, f"manifest row {row_number}")
    if row["schema_version"] != SFT_EXAMPLE_VERSION:
        raise MobilePlanIRScreenError(f"manifest row {row_number} has the wrong schema_version")
    sample_id = _strict_nonempty(row["id"], label=f"manifest row {row_number}.id")
    prompt = _strict_nonempty(row["prompt"], label=f"manifest row {sample_id!r}.prompt")
    target = _strict_nonempty(row["target"], label=f"manifest row {sample_id!r}.target")
    metadata = row["metadata"]
    if type(metadata) is not dict:
        raise MobilePlanIRScreenError(f"manifest row {sample_id!r}.metadata must be an object")
    construction_metadata = {
        "adapter_schema_version": "barun-mobile-actions-adapter-v2",
        "dataset": "google/mobile-actions",
        "derived_split": "dev",
        "prompt_contract_version": "barun-action-prompt-v1",
        "source_split": "train",
    }
    for key, expected in construction_metadata.items():
        if metadata.get(key) != expected:
            raise MobilePlanIRScreenError(
                f"manifest row {sample_id!r}.metadata.{key} must equal {expected!r}"
            )
    forbidden_roles = {
        "confirmation",
        "development",
        "evaluation",
        "final",
        "official",
        "reused",
        "selection",
        "test",
    }
    for key in ("data_role", "evaluation_role", "population_role", "role", "split_role"):
        value = metadata.get(key)
        if isinstance(value, str) and value.casefold() in forbidden_roles:
            raise MobilePlanIRScreenError(f"manifest row {sample_id!r}.metadata.{key} is forbidden")
    screen = metadata.get("mobile_planir_screen")
    if type(screen) is not dict:
        raise MobilePlanIRScreenError(
            f"manifest row {sample_id!r} lacks exact metadata.mobile_planir_screen"
        )
    _exact_fields(screen, _SCREEN_METADATA_FIELDS, f"manifest row {sample_id!r} screen metadata")
    if screen["schema_version"] != SCREEN_METADATA_VERSION:
        raise MobilePlanIRScreenError(
            f"manifest row {sample_id!r} has wrong screen metadata version"
        )
    if screen["split_version"] != SCREEN_SPLIT_VERSION:
        raise MobilePlanIRScreenError(f"manifest row {sample_id!r} has wrong split_version")
    if screen["population"] != "screen":
        raise MobilePlanIRScreenError(f"manifest row {sample_id!r} is not a screen population row")
    arm = _strict_arm(screen["arm"], label=f"manifest row {sample_id!r}.arm")
    source_id = _strict_nonempty(screen["source_id"], label=f"manifest row {sample_id!r}.source_id")
    if sample_id != source_id:
        raise MobilePlanIRScreenError(f"manifest row {sample_id!r} id differs from source_id")
    source_content_sha256 = _strict_sha256(
        screen["source_content_sha256"], label=f"manifest row {sample_id!r}.source_content_sha256"
    )
    source_metadata_sha256 = _strict_sha256(
        screen["source_metadata_sha256"],
        label=f"manifest row {sample_id!r}.source_metadata_sha256",
    )
    source_prompt_sha256 = _strict_sha256(
        screen["source_prompt_sha256"], label=f"manifest row {sample_id!r}.source_prompt_sha256"
    )
    source_target_sha256 = _strict_sha256(
        screen["source_target_sha256"], label=f"manifest row {sample_id!r}.source_target_sha256"
    )
    prompt_sha256 = _strict_sha256(
        screen["output_prompt_sha256"], label=f"manifest row {sample_id!r}.output_prompt_sha256"
    )
    target_sha256 = _strict_sha256(
        screen["output_target_sha256"], label=f"manifest row {sample_id!r}.output_target_sha256"
    )
    if prompt_sha256 != _sha256_text(prompt) or target_sha256 != _sha256_text(target):
        raise MobilePlanIRScreenError(f"manifest row {sample_id!r} output hash mismatch")
    if (
        metadata.get("prompt_sha256") != prompt_sha256
        or metadata.get("target_sha256") != target_sha256
    ):
        raise MobilePlanIRScreenError(
            f"manifest row {sample_id!r} top-level metadata hash mismatch"
        )

    try:
        parsed = parse_construction_prompt(prompt)
        table = build_reference_table(parsed.request, parsed.now)
        recomputed_evidence = make_prompt_evidence(
            prompt, parsed.prompt_contract, table, parsed.request, parsed.now
        )
    except Exception as exc:
        raise MobilePlanIRScreenError(
            f"manifest row {sample_id!r} does not contain a canonical construction prompt"
        ) from exc
    expected_contract = PROMPT_CONTRACT if arm == "C" else ACTION_PROMPT_CONTRACT
    expected_table = table.render() if arm in {"B", "C"} else None
    if parsed.prompt_contract != expected_contract or parsed.rendered_table != expected_table:
        raise MobilePlanIRScreenError(
            f"manifest row {sample_id!r} prompt presentation does not match arm {arm}"
        )
    supplied_evidence = _prompt_evidence_from_mapping(
        screen["prompt_evidence"], label=f"manifest row {sample_id!r}.prompt_evidence"
    )
    if supplied_evidence != recomputed_evidence:
        raise MobilePlanIRScreenError(f"manifest row {sample_id!r} prompt evidence mismatch")
    return _ManifestIdentity(
        sample_id=sample_id,
        source_id=source_id,
        arm=arm,
        prompt=prompt,
        target=target,
        source_content_sha256=source_content_sha256,
        source_metadata_sha256=source_metadata_sha256,
        source_prompt_sha256=source_prompt_sha256,
        source_target_sha256=source_target_sha256,
        prompt_sha256=prompt_sha256,
        table_sha256=table.sha256(),
        prompt_evidence=recomputed_evidence,
        request=parsed.request,
        now=parsed.now,
        system_body=parsed.system_body,
        table=table,
    )


def _parse_gold(target: str, *, source_id: str) -> ActionIR:
    try:
        gold = validate_action_ir(decode_json_object(target), MOBILE_TOOL_SCHEMAS)
    except ActionIRError as exc:
        raise MobilePlanIRScreenError(
            f"direct gold target for {source_id!r} is invalid: {exc.code} at {exc.path}"
        ) from exc
    if gold.canonical_json() != target:
        raise MobilePlanIRScreenError(f"direct gold target for {source_id!r} is not canonical")
    return gold


def _subsets(gold: ActionIR) -> tuple[str, ...]:
    tools = {call.tool for call in gold.calls}
    names = [CALENDAR_SUBSET if "create_calendar_event" in tools else NONCALENDAR_SUBSET]
    if len(gold.calls) == 1:
        names.append(SINGLE_CALL_SUBSET)
    elif len(gold.calls) > 1:
        names.append(MULTI_CALL_SUBSET)
    for tool in TOOL_NAMES:
        if tool in tools:
            names.append(f"contains_tool:{tool}")
    return tuple(names)


def _prepare_manifest(manifest_rows: Sequence[Mapping[str, object]]) -> _ManifestPopulation:
    snapshot = _detached_rows(manifest_rows, label="screen manifest rows")
    rows = tuple(
        sorted(
            (
                _classify_manifest_row(row, row_number=index)
                for index, row in enumerate(snapshot, 1)
            ),
            key=lambda item: (item.source_id, item.arm),
        )
    )
    if not rows:
        raise MobilePlanIRScreenError("screen manifest is empty")
    sample_arm_ids = [(row.sample_id, row.arm) for row in rows]
    if len(sample_arm_ids) != len(set(sample_arm_ids)):
        raise MobilePlanIRScreenError("screen manifest has duplicate (id, arm) rows")
    by_source: dict[str, dict[Arm, _ManifestIdentity]] = {}
    for row in rows:
        arms = by_source.setdefault(row.source_id, {})
        if row.arm in arms:
            raise MobilePlanIRScreenError(f"source {row.source_id!r} has duplicate arm {row.arm}")
        arms[row.arm] = row
    gold_by_source: dict[str, ActionIR] = {}
    subsets_by_source: dict[str, tuple[str, ...]] = {}
    for source_id, arms in sorted(by_source.items()):
        if set(arms) != set(ARMS):
            raise MobilePlanIRScreenError(f"source {source_id!r} does not have exactly A/B/C")
        a, b, c = arms["A"], arms["B"], arms["C"]
        if (
            len({row.source_content_sha256 for row in (a, b, c)}) != 1
            or len({row.source_metadata_sha256 for row in (a, b, c)}) != 1
            or len({row.source_prompt_sha256 for row in (a, b, c)}) != 1
            or len({row.source_target_sha256 for row in (a, b, c)}) != 1
        ):
            raise MobilePlanIRScreenError(
                f"source identity hashes differ across arms for {source_id!r}"
            )
        if (
            a.source_target_sha256 != _sha256_text(a.target)
            or b.target != a.target
            or c.target != a.target
        ):
            raise MobilePlanIRScreenError(f"A/B/C screen gold targets differ for {source_id!r}")
        if (
            a.request != b.request
            or a.request != c.request
            or a.now != b.now
            or a.now != c.now
            or a.system_body != b.system_body
            or a.system_body != c.system_body
            or a.table_sha256 != b.table_sha256
            or a.table_sha256 != c.table_sha256
        ):
            raise MobilePlanIRScreenError(f"matched prompt semantics differ for {source_id!r}")
        gold = _parse_gold(a.target, source_id=source_id)
        gold_by_source[source_id] = gold
        subsets_by_source[source_id] = _subsets(gold)

    source_ids = tuple(sorted(by_source))
    prompt_records = tuple(
        {
            "arm": row.arm,
            "id": row.sample_id,
            "prompt_sha256": row.prompt_sha256,
            "source_id": row.source_id,
            "table_sha256": row.table_sha256,
        }
        for row in rows
    )
    table_records = tuple(
        {
            "source_id": source_id,
            "table_sha256": by_source[source_id]["C"].table_sha256,
        }
        for source_id in source_ids
    )
    return _ManifestPopulation(
        rows=rows,
        gold_by_source=MappingProxyType(gold_by_source),
        subsets_by_source=MappingProxyType(subsets_by_source),
        content_sha256=_hash_records(snapshot, label="screen manifest content"),
        arm_content_sha256=MappingProxyType(
            {
                arm: _hash_records(
                    tuple(
                        row
                        for row in snapshot
                        if row["metadata"]["mobile_planir_screen"]["arm"] == arm
                    ),  # type: ignore[index]
                    label=f"screen manifest arm {arm}",
                )
                for arm in ARMS
            }
        ),
        prompt_population_sha256=_hash_records(prompt_records, label="prompt population"),
        table_population_sha256=_hash_records(table_records, label="table population"),
        population_sha256=_sha256_text("".join(f"{source_id}\n" for source_id in source_ids)),
    )


def _classify_prediction_row(
    row: Mapping[str, object],
    *,
    row_number: int,
    manifest_by_id: Mapping[tuple[Arm, str], _ManifestIdentity],
) -> _PredictionIdentity:
    if type(row) is not dict:
        raise MobilePlanIRScreenError(f"prediction row {row_number} must be an exact JSON object")
    _exact_fields(row, _PREDICTION_FIELDS, f"prediction row {row_number}")
    if row["schema_version"] != SCREEN_PREDICTION_ROW_VERSION:
        raise MobilePlanIRScreenError(f"prediction row {row_number} has wrong schema_version")
    sample_id = _strict_nonempty(row["id"], label=f"prediction row {row_number}.id")
    source_id = _strict_nonempty(row["source_id"], label=f"prediction row {sample_id!r}.source_id")
    arm = _strict_arm(row["arm"], label=f"prediction row {sample_id!r}.arm")
    manifest = manifest_by_id.get((arm, sample_id))
    if manifest is None:
        raise MobilePlanIRScreenError(f"prediction row {(arm, sample_id)!r} has no manifest row")
    seed = row["seed"]
    if type(seed) is not int or seed not in SEEDS:
        raise MobilePlanIRScreenError(f"prediction row {sample_id!r}.seed must be one of {SEEDS!r}")
    if source_id != manifest.source_id or arm != manifest.arm:
        raise MobilePlanIRScreenError(
            f"prediction row {sample_id!r} identity differs from manifest"
        )
    prompt_sha256 = _strict_sha256(
        row["prompt_sha256"], label=f"prediction row {sample_id!r}.prompt_sha256"
    )
    table_sha256 = _strict_sha256(
        row["table_sha256"], label=f"prediction row {sample_id!r}.table_sha256"
    )
    if prompt_sha256 != manifest.prompt_sha256 or table_sha256 != manifest.table_sha256:
        raise MobilePlanIRScreenError(
            f"prediction row {sample_id!r} prompt/table binding differs from manifest"
        )
    checkpoint_sha256 = _strict_sha256(
        row["checkpoint_sha256"], label=f"prediction row {sample_id!r}.checkpoint_sha256"
    )
    decoding_sha256 = _strict_sha256(
        row["decoding_sha256"], label=f"prediction row {sample_id!r}.decoding_sha256"
    )
    prediction_raw = row["prediction_raw"]
    if prediction_raw is not None and not isinstance(prediction_raw, str):
        raise MobilePlanIRScreenError(
            f"prediction row {sample_id!r}.prediction_raw must be string or null"
        )
    missing = _strict_bool(row["missing"], label=f"prediction row {sample_id!r}.missing")
    if missing is not (prediction_raw is None):
        raise MobilePlanIRScreenError(
            f"prediction row {sample_id!r}.missing disagrees with prediction_raw"
        )
    raw_sha = row["prediction_raw_sha256"]
    expected_raw_sha = None if prediction_raw is None else _sha256_text(prediction_raw)
    if raw_sha != expected_raw_sha:
        raise MobilePlanIRScreenError(f"prediction row {sample_id!r} raw output hash mismatch")
    truncated = _strict_bool(row["truncated"], label=f"prediction row {sample_id!r}.truncated")
    generation_failure = row["generation_failure"]
    if generation_failure is not None:
        generation_failure = _strict_nonempty(
            generation_failure, label=f"prediction row {sample_id!r}.generation_failure"
        )
        if any(character.isspace() for character in generation_failure):
            raise MobilePlanIRScreenError(
                f"prediction row {sample_id!r}.generation_failure must be a stable code"
            )
    return _PredictionIdentity(
        sample_id=sample_id,
        source_id=source_id,
        arm=arm,
        seed=seed,
        prompt_sha256=prompt_sha256,
        table_sha256=table_sha256,
        checkpoint_sha256=checkpoint_sha256,
        decoding_sha256=decoding_sha256,
        prediction_raw=prediction_raw,
        prediction_raw_sha256=expected_raw_sha,
        missing=missing,
        truncated=truncated,
        generation_failure=generation_failure,
    )


def _prepare_predictions(
    prediction_rows: Sequence[Mapping[str, object]], manifest: _ManifestPopulation
) -> _PredictionPopulation:
    snapshot = _detached_rows(prediction_rows, label="screen prediction rows")
    manifest_by_id = {(row.arm, row.sample_id): row for row in manifest.rows}
    rows = tuple(
        sorted(
            (
                _classify_prediction_row(row, row_number=index, manifest_by_id=manifest_by_id)
                for index, row in enumerate(snapshot, 1)
            ),
            key=lambda item: (item.seed, item.arm, item.source_id),
        )
    )
    expected = {(row.sample_id, row.arm, seed) for row in manifest.rows for seed in SEEDS}
    observed = [(row.sample_id, row.arm, row.seed) for row in rows]
    if len(observed) != len(set(observed)):
        raise MobilePlanIRScreenError("prediction evidence has duplicate (id, arm, seed) rows")
    observed_set = set(observed)
    if observed_set != expected:
        missing = sorted(expected.difference(observed_set))[:5]
        extra = sorted(observed_set.difference(expected))[:5]
        raise MobilePlanIRScreenError(
            f"prediction population differs from manifest x seeds; missing={missing!r}, extra={extra!r}"
        )
    by_run: dict[tuple[Arm, int], list[_PredictionIdentity]] = {}
    for row in rows:
        by_run.setdefault((row.arm, row.seed), []).append(row)
    for run, run_rows in by_run.items():
        if len({row.checkpoint_sha256 for row in run_rows}) != 1:
            raise MobilePlanIRScreenError(f"run {run!r} contains multiple checkpoints")
        if len({row.decoding_sha256 for row in run_rows}) != 1:
            raise MobilePlanIRScreenError(f"run {run!r} contains multiple decoding configs")
    if len({row.decoding_sha256 for row in rows}) != 1:
        raise MobilePlanIRScreenError(
            "matched A/B/C seeds must use one identical decoding configuration"
        )
    return _PredictionPopulation(
        rows=rows,
        content_sha256=_hash_records(snapshot, label="screen prediction content"),
    )


def _complete_argument_values(gold: ActionIR, prediction: ActionIR | None) -> bool:
    """Require the complete multiset of tool-associated argument values for the row.

    Outer decision, mode, and serial order remain AST concerns.  Call identity is
    retained so two argument-free but different tools cannot receive vacuous credit.
    """

    if prediction is None:
        return False
    gold_calls = Counter(call.canonical_json() for call in gold.calls)
    predicted_calls = Counter(call.canonical_json() for call in prediction.calls)
    return gold_calls == predicted_calls


def _score_row(
    manifest: _ManifestIdentity,
    prediction: _PredictionIdentity,
    gold: ActionIR,
    subsets: tuple[str, ...],
) -> RowEvidence:
    raw_json_parse = False
    action_ir_schema_valid = False
    compiler_attempted = False
    compiler_success: bool | None = None
    compiler_error_code: str | None = None
    compiler_error_path: str | None = None
    parsed_prediction: ActionIR | None = None
    action_ir_raw: str | None = None

    if prediction.prediction_raw is not None:
        try:
            decoded = decode_json_object(prediction.prediction_raw)
            raw_json_parse = True
        except ActionIRParseError:
            decoded = None
        if manifest.arm in {"A", "B"}:
            if decoded is not None:
                try:
                    parsed_prediction = validate_action_ir(decoded, MOBILE_TOOL_SCHEMAS)
                    action_ir_schema_valid = True
                    action_ir_raw = parsed_prediction.canonical_json()
                except ActionIRError:
                    pass
        else:
            compiler_attempted = True
            result = compile_mobile_plan(
                prediction.prediction_raw,
                manifest.request,
                manifest.now,
                prompt_contract=PROMPT_CONTRACT,
                rendered_prompt=manifest.prompt,
                prompt_evidence=manifest.prompt_evidence,
                rendered_table=manifest.table.render(),
                table_sha256=manifest.table_sha256,
                schemas=MOBILE_TOOL_SCHEMAS,
                context=MappingProxyType({}),
            )
            compiler_success = result.ok
            if result.ok:
                assert result.action_ir is not None
                action_ir_raw = result.action_ir
                try:
                    parsed_prediction = validate_action_ir(
                        decode_json_object(action_ir_raw), MOBILE_TOOL_SCHEMAS
                    )
                    action_ir_schema_valid = True
                except ActionIRError:
                    pass
            else:
                assert result.error is not None
                compiler_error_code = result.error.code
                compiler_error_path = result.error.path
    elif manifest.arm == "C":
        compiler_success = False
        compiler_error_code = "missing_prediction"
        compiler_error_path = "$"

    failure_free = not (
        prediction.missing or prediction.truncated or prediction.generation_failure is not None
    )
    assessed_call = bool(
        parsed_prediction is not None and parsed_prediction.decision is Decision.CALL
    )
    simulator_success = (
        action_ir_equal(parsed_prediction, gold) if assessed_call and parsed_prediction else None
    )
    # The frozen Mobile hook performs no external action and marks catastrophic false.
    # Preserve whether a CALL was actually assessed so zero is not presented as broad
    # safety evidence for invalid or non-CALL outputs.
    catastrophic_unauthorized_action = False
    ast_exact = bool(
        failure_free and parsed_prediction is not None and action_ir_equal(gold, parsed_prediction)
    )
    argument_value_exact = bool(failure_free and _complete_argument_values(gold, parsed_prediction))
    return RowEvidence(
        sample_id=manifest.sample_id,
        source_id=manifest.source_id,
        arm=manifest.arm,
        seed=prediction.seed,
        prompt_contract=manifest.prompt_evidence.prompt_contract,
        prompt_sha256=manifest.prompt_sha256,
        table_sha256=manifest.table_sha256,
        request_sha256=manifest.prompt_evidence.request_sha256,
        checkpoint_sha256=prediction.checkpoint_sha256,
        decoding_sha256=prediction.decoding_sha256,
        prediction_raw=prediction.prediction_raw,
        prediction_raw_sha256=prediction.prediction_raw_sha256,
        prediction_action_ir=action_ir_raw,
        prediction_action_ir_sha256=(
            None if action_ir_raw is None else _sha256_text(action_ir_raw)
        ),
        raw_json_parse=raw_json_parse,
        action_ir_schema_valid=action_ir_schema_valid,
        compiler_attempted=compiler_attempted,
        compiler_success=compiler_success,
        compiler_error_code=compiler_error_code,
        compiler_error_path=compiler_error_path,
        missing=prediction.missing,
        truncated=prediction.truncated,
        generation_failure=prediction.generation_failure,
        assessed_call=assessed_call,
        simulator_success=simulator_success,
        catastrophic_unauthorized_action=catastrophic_unauthorized_action,
        ast_exact=ast_exact,
        argument_value_exact=argument_value_exact,
        subsets=subsets,
    )


def _run_metrics(
    arm: Arm,
    seed: int,
    rows: Sequence[RowEvidence],
    *,
    population_sha256: str,
) -> RunMetrics:
    if not rows:
        raise MobilePlanIRScreenError(f"run {(arm, seed)!r} is empty")
    count = len(rows)
    checkpoints = {row.checkpoint_sha256 for row in rows}
    decodings = {row.decoding_sha256 for row in rows}
    if len(checkpoints) != 1 or len(decodings) != 1:
        raise MobilePlanIRScreenError(f"run {(arm, seed)!r} identity is inconsistent")
    compiler_successes = sum(row.compiler_success is True for row in rows)
    conditional_schema = sum(
        row.compiler_success is True and row.action_ir_schema_valid for row in rows
    )
    subset_metrics: dict[str, SubsetMetrics] = {}
    for name in PREDEFINED_SUBSETS:
        selected = [row for row in rows if name in row.subsets]
        subset_metrics[name] = SubsetMetrics(
            sample_count=len(selected),
            ast_exact=ExactRate(sum(row.ast_exact for row in selected), len(selected)),
            argument_value_exact=ExactRate(
                sum(row.argument_value_exact for row in selected), len(selected)
            ),
        )
    return RunMetrics(
        arm=arm,
        seed=seed,
        population_sha256=population_sha256,
        checkpoint_sha256=next(iter(checkpoints)),
        decoding_sha256=next(iter(decodings)),
        sample_count=count,
        ast_exact=ExactRate(sum(row.ast_exact for row in rows), count),
        argument_value_exact=ExactRate(sum(row.argument_value_exact for row in rows), count),
        raw_json_parse=ExactRate(sum(row.raw_json_parse for row in rows), count),
        action_ir_schema_valid=ExactRate(sum(row.action_ir_schema_valid for row in rows), count),
        compiler_success=(ExactRate(compiler_successes, count) if arm == "C" else None),
        conditional_compiled_schema_valid=(
            ExactRate(conditional_schema, compiler_successes) if arm == "C" else None
        ),
        failure_counts={
            "raw_parse_failure": sum(not row.raw_json_parse for row in rows),
            "action_ir_schema_failure": sum(not row.action_ir_schema_valid for row in rows),
            "compiler_failure": sum(row.compiler_success is False for row in rows),
            "missing": sum(row.missing for row in rows),
            "truncation": sum(row.truncated for row in rows),
            "generation_failure": sum(row.generation_failure is not None for row in rows),
            "catastrophic_unauthorized_action": sum(
                row.catastrophic_unauthorized_action for row in rows
            ),
        },
        subsets=subset_metrics,
    )


def score_screen(
    manifest_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    enforce_pinned: bool = True,
) -> ScreenEvaluation:
    """Strictly score the complete matched A/B/C x 17/29/43 screen population."""

    manifest = _prepare_manifest(manifest_rows)
    _enforce_pinned_manifest(manifest, enforce_pinned=enforce_pinned)
    prediction_population = _prepare_predictions(prediction_rows, manifest)
    predictions = prediction_population.rows
    manifest_by_id = {(row.arm, row.sample_id): row for row in manifest.rows}
    evidence = tuple(
        _score_row(
            manifest_by_id[(prediction.arm, prediction.sample_id)],
            prediction,
            manifest.gold_by_source[prediction.source_id],
            manifest.subsets_by_source[prediction.source_id],
        )
        for prediction in predictions
    )
    runs = tuple(
        _run_metrics(
            arm,
            seed,
            [row for row in evidence if row.arm == arm and row.seed == seed],
            population_sha256=manifest.population_sha256,
        )
        for arm in ARMS
        for seed in SEEDS
    )
    return ScreenEvaluation(
        schema_version=SCREEN_SCORE_VERSION,
        manifest_content_sha256=manifest.content_sha256,
        prompt_population_sha256=manifest.prompt_population_sha256,
        table_population_sha256=manifest.table_population_sha256,
        population_sha256=manifest.population_sha256,
        rows=evidence,
        runs=runs,
        pinned_manifest_enforced=enforce_pinned,
    )


def _pooled(runs: Sequence[RunMetrics], arm: Arm, field: str) -> ExactRate:
    values = [getattr(run, field) for run in runs if run.arm == arm]
    if len(values) != len(SEEDS) or any(not isinstance(value, ExactRate) for value in values):
        raise MobilePlanIRScreenError(f"cannot pool {field!r} for arm {arm}")
    return ExactRate(
        sum(value.numerator for value in values),
        sum(value.denominator for value in values),
    )


def _pooled_subset(runs: Sequence[RunMetrics], arm: Arm, subset: str, field: str) -> ExactRate:
    values = [getattr(run.subsets[subset], field) for run in runs if run.arm == arm]
    if len(values) != len(SEEDS):
        raise MobilePlanIRScreenError(f"cannot pool subset {subset!r} for arm {arm}")
    return ExactRate(
        sum(value.numerator for value in values),
        sum(value.denominator for value in values),
    )


def _at_least_margin(candidate: ExactRate, baseline: ExactRate, points: int) -> bool:
    """Compare candidate-baseline >= points/100 by integer cross multiplication."""

    if not candidate.denominator or not baseline.denominator:
        return False
    return (
        100
        * (candidate.numerator * baseline.denominator - baseline.numerator * candidate.denominator)
        >= points * candidate.denominator * baseline.denominator
    )


def _no_more_than_loss(candidate: ExactRate, baseline: ExactRate, points: int) -> bool:
    """Compare candidate-baseline >= -points/100 without floating point."""

    if not candidate.denominator or not baseline.denominator:
        return False
    return (
        100
        * (candidate.numerator * baseline.denominator - baseline.numerator * candidate.denominator)
        >= -points * candidate.denominator * baseline.denominator
    )


def _strictly_greater(left: ExactRate, right: ExactRate) -> bool:
    if not left.denominator or not right.denominator:
        return False
    return left.numerator * right.denominator > right.numerator * left.denominator


def _fraction_record(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _delta(candidate: ExactRate, baseline: ExactRate) -> Fraction:
    if not candidate.denominator or not baseline.denominator:
        raise MobilePlanIRScreenError("cannot compute a delta with an empty denominator")
    return Fraction(candidate.numerator, candidate.denominator) - Fraction(
        baseline.numerator, baseline.denominator
    )


def evaluate_screen_gate(evaluation: ScreenEvaluation) -> ScreenGate:
    """Apply the frozen exact internal gate to one complete screen evaluation."""

    if (
        not isinstance(evaluation, ScreenEvaluation)
        or evaluation.schema_version != SCREEN_SCORE_VERSION
    ):
        raise MobilePlanIRScreenError("gate requires one frozen ScreenEvaluation")
    if not evaluation.pinned_manifest_enforced:
        raise MobilePlanIRScreenError(
            "scientific gate requires a score with pinned manifest enforcement"
        )
    observed_identity = {
        "manifest_content_sha256": evaluation.manifest_content_sha256,
        "population_sha256": evaluation.population_sha256,
        "prompt_population_sha256": evaluation.prompt_population_sha256,
        "table_population_sha256": evaluation.table_population_sha256,
    }
    expected_identity = {
        "manifest_content_sha256": PINNED_MANIFEST_CONTENT_SHA256,
        "population_sha256": PINNED_POPULATION_SHA256,
        "prompt_population_sha256": PINNED_PROMPT_POPULATION_SHA256,
        "table_population_sha256": PINNED_TABLE_POPULATION_SHA256,
    }
    if observed_identity != expected_identity:
        raise MobilePlanIRScreenError(
            "scientific gate requires the exact frozen manifest, prompt, table, and population "
            "identities"
        )
    runs = evaluation.runs
    identities = [(run.arm, run.seed) for run in runs]
    expected = [(arm, seed) for arm in ARMS for seed in SEEDS]
    if sorted(identities) != sorted(expected) or len(identities) != len(set(identities)):
        raise MobilePlanIRScreenError("gate requires exactly A/B/C x seeds 17/29/43")
    if {run.population_sha256 for run in runs} != {evaluation.population_sha256}:
        raise MobilePlanIRScreenError("run population hashes differ")
    counts = {run.sample_count for run in runs}
    if len(counts) != 1:
        raise MobilePlanIRScreenError("matched run sample counts differ")
    for subset in PREDEFINED_SUBSETS:
        subset_counts = {run.subsets[subset].sample_count for run in runs}
        if len(subset_counts) != 1 or subset_counts == {0}:
            raise MobilePlanIRScreenError(
                f"predefined subset {subset!r} is empty or unmatched across runs"
            )

    pooled = {
        arm: {
            "ast_exact": _pooled(runs, arm, "ast_exact"),
            "argument_value_exact": _pooled(runs, arm, "argument_value_exact"),
        }
        for arm in ARMS
    }
    checks: dict[str, bool] = {}
    comparisons: dict[str, object] = {}
    for baseline in ("A", "B"):
        ast = _delta(pooled["C"]["ast_exact"], pooled[baseline]["ast_exact"])
        arguments = _delta(
            pooled["C"]["argument_value_exact"],
            pooled[baseline]["argument_value_exact"],
        )
        checks[f"ast_margin_vs_{baseline}_at_least_3_points"] = _at_least_margin(
            pooled["C"]["ast_exact"], pooled[baseline]["ast_exact"], 3
        )
        checks[f"argument_margin_vs_{baseline}_at_least_5_points"] = _at_least_margin(
            pooled["C"]["argument_value_exact"],
            pooled[baseline]["argument_value_exact"],
            5,
        )
        comparisons[baseline] = {
            "argument_value_exact_delta": _fraction_record(arguments),
            "ast_exact_delta": _fraction_record(ast),
        }

    by_identity = {(run.arm, run.seed): run for run in runs}
    winning_seeds = []
    for seed in SEEDS:
        candidate = by_identity[("C", seed)].ast_exact
        if all(
            _strictly_greater(candidate, by_identity[(baseline, seed)].ast_exact)
            for baseline in ("A", "B")
        ):
            winning_seeds.append(seed)
    checks["c_beats_both_ast_in_at_least_2_seeds"] = len(winning_seeds) >= 2

    subset_comparisons: dict[str, object] = {}
    subset_ok = True
    for subset in PREDEFINED_SUBSETS:
        candidate = _pooled_subset(runs, "C", subset, "ast_exact")
        subset_record: dict[str, object] = {}
        for baseline in ("A", "B"):
            reference = _pooled_subset(runs, baseline, subset, "ast_exact")
            allowed = _no_more_than_loss(candidate, reference, 2)
            subset_ok = subset_ok and allowed
            subset_record[baseline] = {
                "delta": _fraction_record(_delta(candidate, reference)),
                "loss_no_more_than_2_points": allowed,
            }
        subset_comparisons[subset] = subset_record
    checks["no_predefined_subset_ast_loss_over_2_points"] = subset_ok

    c_raw_parse = _pooled(runs, "C", "raw_json_parse")
    c_compiler_values = [run.compiler_success for run in runs if run.arm == "C"]
    assert all(isinstance(value, ExactRate) for value in c_compiler_values)
    c_compiler = ExactRate(
        sum(value.numerator for value in c_compiler_values if value is not None),
        sum(value.denominator for value in c_compiler_values if value is not None),
    )
    c_schema_values = [run.conditional_compiled_schema_valid for run in runs if run.arm == "C"]
    assert all(isinstance(value, ExactRate) for value in c_schema_values)
    c_schema = ExactRate(
        sum(value.numerator for value in c_schema_values if value is not None),
        sum(value.denominator for value in c_schema_values if value is not None),
    )
    checks["c_raw_json_parse_at_least_99_5_percent"] = (
        1000 * c_raw_parse.numerator >= 995 * c_raw_parse.denominator
    )
    checks["c_compiler_success_at_least_99_5_percent"] = (
        1000 * c_compiler.numerator >= 995 * c_compiler.denominator
    )
    checks["c_conditional_compiled_schema_valid_100_percent"] = bool(
        c_schema.denominator and c_schema.numerator == c_schema.denominator
    )

    for failure in (
        "truncation",
        "missing",
        "generation_failure",
        "catastrophic_unauthorized_action",
    ):
        checks[f"every_arm_zero_{failure}"] = all(run.failure_counts[failure] == 0 for run in runs)
    observed = {
        "comparisons": comparisons,
        "c_compiler_success": c_compiler.to_record(),
        "c_conditional_compiled_schema_valid": c_schema.to_record(),
        "c_raw_json_parse": c_raw_parse.to_record(),
        "pooled": {
            arm: {name: rate.to_record() for name, rate in metrics.items()}
            for arm, metrics in pooled.items()
        },
        "positive_ast_seed_count": len(winning_seeds),
        "positive_ast_seeds": winning_seeds,
        "subset_comparisons": subset_comparisons,
        "safety_scope": (
            "CALL-only construction screen with no ABSTAIN, CLARIFY, or CONFIRM evidence; "
            "zero catastrophic count is not a product safety result"
        ),
        "thresholds": {
            "argument_margin_points": {"numerator": 5, "denominator": 100},
            "ast_margin_points": {"numerator": 3, "denominator": 100},
            "candidate_validity": {"numerator": 995, "denominator": 1000},
            "minimum_positive_seed_count": 2,
            "subset_maximum_loss_points": {"numerator": 2, "denominator": 100},
        },
    }
    return ScreenGate(
        schema_version=SCREEN_GATE_VERSION,
        passed=all(checks.values()),
        checks=checks,
        observed=observed,
    )


def build_evidence_receipt(
    manifest_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    config_sha256: str,
    compiler_source_sha256: str,
    evaluator_source_sha256: str,
    renderer_source_sha256: str,
    enforce_pinned: bool = True,
) -> dict[str, object]:
    """Build one deterministic receipt over exact decoded evidence populations."""

    bindings = {
        "compiler_source_sha256": _strict_sha256(
            compiler_source_sha256, label="compiler_source_sha256"
        ),
        "config_sha256": _strict_sha256(config_sha256, label="config_sha256"),
        "evaluator_source_sha256": _strict_sha256(
            evaluator_source_sha256, label="evaluator_source_sha256"
        ),
        "renderer_source_sha256": _strict_sha256(
            renderer_source_sha256, label="renderer_source_sha256"
        ),
    }
    manifest = _prepare_manifest(manifest_rows)
    _enforce_pinned_manifest(manifest, enforce_pinned=enforce_pinned)
    prediction_population = _prepare_predictions(prediction_rows, manifest)
    predictions = prediction_population.rows
    per_run: dict[str, object] = {}
    for arm in ARMS:
        for seed in SEEDS:
            run_rows = [row for row in predictions if row.arm == arm and row.seed == seed]
            checkpoint = {row.checkpoint_sha256 for row in run_rows}
            decoding = {row.decoding_sha256 for row in run_rows}
            assert len(checkpoint) == len(decoding) == 1
            output_records = tuple(
                {
                    "id": row.sample_id,
                    "prediction_raw_sha256": row.prediction_raw_sha256,
                    "prompt_sha256": row.prompt_sha256,
                    "table_sha256": row.table_sha256,
                }
                for row in run_rows
            )
            per_run[f"{arm}:s{seed}"] = {
                "arm": arm,
                "checkpoint_sha256": next(iter(checkpoint)),
                "decoding_sha256": next(iter(decoding)),
                "raw_output_population_sha256": _hash_records(
                    output_records, label=f"raw output population {arm}:s{seed}"
                ),
                "rows": len(run_rows),
                "seed": seed,
            }
    return {
        "action_ir_evaluator_version": EVALUATOR_VERSION,
        **bindings,
        "manifest_content_sha256": manifest.content_sha256,
        "manifest_arm_content_sha256": dict(manifest.arm_content_sha256),
        "manifest_row_count": len(manifest.rows),
        "planir_prompt_contract": PROMPT_CONTRACT,
        "planir_prompt_renderer_version": PROMPT_RENDERER_VERSION,
        "population_sha256": manifest.population_sha256,
        "prediction_content_sha256": prediction_population.content_sha256,
        "prediction_row_count": len(predictions),
        "pinned_manifest_enforced": enforce_pinned,
        "prompt_evidence_scope": (
            "PromptEvidence proves supplied consistency only; this receipt binds, but cannot by "
            "itself observe, model presentation"
        ),
        "prompt_population_sha256": manifest.prompt_population_sha256,
        "runs": per_run,
        "safety_scope": (
            "CALL-only construction screen; frozen deterministic simulator performs no real actions"
        ),
        "schema_version": SCREEN_RECEIPT_VERSION,
        "simulator_contract": "mobile-actions-deterministic-exact-match-v1",
        "screen_manifest_row_version": SCREEN_MANIFEST_ROW_VERSION,
        "screen_prediction_row_version": SCREEN_PREDICTION_ROW_VERSION,
        "screen_score_version": SCREEN_SCORE_VERSION,
        "table_population_sha256": manifest.table_population_sha256,
    }


def validate_evidence_receipt(
    receipt: Mapping[str, object],
    manifest_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    config_sha256: str,
    compiler_source_sha256: str,
    evaluator_source_sha256: str,
    renderer_source_sha256: str,
    enforce_pinned: bool = True,
) -> dict[str, object]:
    """Rebuild and byte-semantically compare one receipt; substitutions fail closed."""

    if type(receipt) is not dict:
        raise MobilePlanIRScreenError("evidence receipt must be an exact JSON object")
    expected = build_evidence_receipt(
        manifest_rows,
        prediction_rows,
        config_sha256=config_sha256,
        compiler_source_sha256=compiler_source_sha256,
        evaluator_source_sha256=evaluator_source_sha256,
        renderer_source_sha256=renderer_source_sha256,
        enforce_pinned=enforce_pinned,
    )
    if _canonical_json(receipt, label="evidence receipt") != _canonical_json(
        expected, label="recomputed evidence receipt"
    ):
        raise MobilePlanIRScreenError("evidence receipt does not bind the supplied evidence")
    return expected


__all__ = [
    "ARMS",
    "CALENDAR_SUBSET",
    "MULTI_CALL_SUBSET",
    "NONCALENDAR_SUBSET",
    "PINNED_MANIFEST_CONTENT_SHA256",
    "PINNED_MANIFEST_ROWS",
    "PINNED_POPULATION_SHA256",
    "PINNED_PROMPT_POPULATION_SHA256",
    "PINNED_SOURCE_ROWS",
    "PINNED_TABLE_POPULATION_SHA256",
    "PREDEFINED_SUBSETS",
    "SCREEN_GATE_VERSION",
    "SCREEN_MANIFEST_ROW_VERSION",
    "SCREEN_METADATA_VERSION",
    "SCREEN_PREDICTION_ROW_VERSION",
    "SCREEN_RECEIPT_VERSION",
    "SCREEN_SCORE_VERSION",
    "SCREEN_SPLIT_VERSION",
    "SEEDS",
    "SFT_EXAMPLE_VERSION",
    "SINGLE_CALL_SUBSET",
    "TOOL_NAMES",
    "ExactRate",
    "MobilePlanIRScreenError",
    "RowEvidence",
    "RunMetrics",
    "ScreenEvaluation",
    "ScreenGate",
    "SubsetMetrics",
    "build_evidence_receipt",
    "evaluate_screen_gate",
    "score_screen",
    "validate_evidence_receipt",
]
