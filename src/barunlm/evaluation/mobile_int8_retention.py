"""Frozen candidate-v2 ARM64 int8 Mobile Actions retention diagnostic.

The protocol compares one explicitly loaded int8 artifact with the immutable candidate-v2
float sample evidence.  It accepts only the already-derived development population and never
accepts a path to Mobile Actions official-evaluation data.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

from barunlm.datasets.mobile_actions import (
    MOBILE_ACTIONS_DATASET_ID,
    MOBILE_ACTIONS_FINAL_EVAL_ROWS,
    MOBILE_ACTIONS_REVISION,
)
from barunlm.quantization import (
    INT8_ARTIFACT_FILES,
    INT8_FORMAT_VERSION,
    Int8CheckpointInfo,
    QuantizationError,
    verify_int8_checkpoint,
)
from barunlm.training.data import sha256_file

from .evaluator import EVALUATOR_VERSION
from .generation import (
    GENERATION_VERSION,
    INT8_GENERATION_VERSION,
    REQUIRED_CHECKPOINT_FILES,
)
from .mobile_actions import MOBILE_ACTIONS_SCORER_VERSION

MOBILE_INT8_RETENTION_PROTOCOL_VERSION = "barunaction-arm64-int8-retention-preregistration-v1"
MOBILE_INT8_RETENTION_RESULT_VERSION = "barunaction-arm64-int8-retention-result-v1"
MOBILE_INT8_RETENTION_PAIRED_VERSION = "barunaction-arm64-int8-retention-paired-v1"
MOBILE_INT8_RETENTION_PROTOCOL_SHA256 = (
    "7229572bea461a7899b09c0310a77b9b4d8af74211d1b9ac8b2b5f77dc6b67cb"
)


class MobileInt8RetentionError(RuntimeError):
    """An input or receipt violates the frozen int8 retention contract."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MobileInt8RetentionError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise MobileInt8RetentionError(f"non-finite JSON constant {value!r}")


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except MobileInt8RetentionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MobileInt8RetentionError(f"cannot read strict JSON from {source}") from error
    if not isinstance(payload, dict):
        raise MobileInt8RetentionError(f"JSON artifact must contain an object: {source}")
    return payload


def _strict_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise MobileInt8RetentionError(f"{source}: blank JSONL line {line_number}")
                try:
                    row = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_pairs,
                        parse_constant=_reject_constant,
                    )
                except json.JSONDecodeError as error:
                    raise MobileInt8RetentionError(
                        f"{source}: invalid JSON on line {line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise MobileInt8RetentionError(f"{source}: line {line_number} is not an object")
                rows.append(row)
    except MobileInt8RetentionError:
        raise
    except (OSError, UnicodeError) as error:
        raise MobileInt8RetentionError(f"cannot read JSONL artifact {source}") from error
    return rows


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MobileInt8RetentionError(f"{name} must be a JSON object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash_map(value: object, *, names: Sequence[str], path: str) -> dict[str, str]:
    source = _mapping(value, name=path)
    if set(source) != set(names):
        raise MobileInt8RetentionError(f"{path} must contain exactly {sorted(names)!r}")
    result = {str(name): str(digest) for name, digest in source.items()}
    if any(not _is_sha256(digest) for digest in result.values()):
        raise MobileInt8RetentionError(f"{path} contains an invalid SHA-256")
    return result


def _require_hash(path: str | Path, expected: object, *, name: str) -> str:
    if not _is_sha256(expected):
        raise MobileInt8RetentionError(f"{name} expected SHA-256 is invalid")
    actual = sha256_file(path)
    if actual != expected:
        raise MobileInt8RetentionError(
            f"{name} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _resolve(root: Path, contract: Mapping[str, Any], *, name: str) -> Path:
    relative = contract.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise MobileInt8RetentionError(f"{name} lacks a relative_path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise MobileInt8RetentionError(f"{name} escapes the repository root") from error
    return candidate


def load_retention_protocol(path: str | Path) -> dict[str, Any]:
    """Load only the checked-in, hash-pinned candidate-v2 retention protocol."""

    _require_hash(
        path,
        MOBILE_INT8_RETENTION_PROTOCOL_SHA256,
        name="ARM64 int8 retention protocol",
    )
    protocol = _strict_json(path)
    if protocol.get("schema_version") != MOBILE_INT8_RETENTION_PROTOCOL_VERSION:
        raise MobileInt8RetentionError("unsupported ARM64 int8 retention protocol version")
    if protocol.get("protocol_id") != "candidate-v2-arm64-int8-retention-v1":
        raise MobileInt8RetentionError("unexpected ARM64 int8 retention protocol ID")
    if protocol.get("status") != "frozen_unrun":
        raise MobileInt8RetentionError("the immutable protocol status changed")

    candidate = _mapping(protocol.get("candidate"), name="candidate")
    if candidate.get("candidate_id") != "candidate-v2":
        raise MobileInt8RetentionError("float source must remain candidate-v2")
    if not _is_sha256(candidate.get("float_checkpoint_manifest_sha256")):
        raise MobileInt8RetentionError("candidate float manifest SHA-256 is invalid")
    float_hashes = _hash_map(
        candidate.get("float_checkpoint_file_sha256"),
        names=REQUIRED_CHECKPOINT_FILES,
        path="candidate.float_checkpoint_file_sha256",
    )

    int8 = _mapping(protocol.get("int8_checkpoint"), name="int8_checkpoint")
    if int8.get("format_version") != INT8_FORMAT_VERSION:
        raise MobileInt8RetentionError("int8 format version changed")
    if not _is_sha256(int8.get("manifest_sha256")):
        raise MobileInt8RetentionError("int8 manifest SHA-256 is invalid")
    artifact_hashes = _hash_map(
        int8.get("artifact_sha256"),
        names=INT8_ARTIFACT_FILES,
        path="int8_checkpoint.artifact_sha256",
    )
    source_hashes = _hash_map(
        int8.get("source_checkpoint_file_sha256"),
        names=REQUIRED_CHECKPOINT_FILES,
        path="int8_checkpoint.source_checkpoint_file_sha256",
    )
    if source_hashes != float_hashes:
        raise MobileInt8RetentionError("int8 source hashes differ from candidate-v2")
    for name in ("barun_config.json", "tokenizer.json"):
        if artifact_hashes[name] != float_hashes[name]:
            raise MobileInt8RetentionError(f"int8 {name} differs from candidate-v2")
    runtime = _mapping(int8.get("runtime"), name="int8_checkpoint.runtime")
    if runtime != {
        "platform_machine": "arm64",
        "platform_system": "Darwin",
        "torch_version": "2.13.0",
        "qengine": "qnnpack",
    }:
        raise MobileInt8RetentionError("int8 runtime contract changed")

    dataset = _mapping(protocol.get("dataset"), name="dataset")
    if dataset.get("dataset_id") != MOBILE_ACTIONS_DATASET_ID:
        raise MobileInt8RetentionError("Mobile dataset ID changed")
    if dataset.get("revision") != MOBILE_ACTIONS_REVISION:
        raise MobileInt8RetentionError("Mobile dataset revision changed")
    manifest = _mapping(dataset.get("manifest"), name="dataset.manifest")
    audit = _mapping(dataset.get("audit"), name="dataset.audit")
    firewall = _mapping(
        dataset.get("official_evaluation_firewall"),
        name="dataset.official_evaluation_firewall",
    )
    if manifest.get("rows") != 756 or not _is_sha256(manifest.get("sha256")):
        raise MobileInt8RetentionError("Mobile development manifest contract changed")
    if not _is_sha256(audit.get("sha256")):
        raise MobileInt8RetentionError("Mobile audit SHA-256 is invalid")
    if firewall != {
        "rows": MOBILE_ACTIONS_FINAL_EVAL_ROWS,
        "opaque_unparsed": True,
        "accepted_input_paths": [],
    }:
        raise MobileInt8RetentionError("official Mobile evaluation firewall changed")

    reference = _mapping(protocol.get("source_float_reference"), name="source_float_reference")
    if reference.get("checkpoint_format") != "float":
        raise MobileInt8RetentionError("source reference checkpoint format changed")
    reference_aggregate = _mapping(reference.get("aggregate"), name="reference.aggregate")
    reference_samples = _mapping(reference.get("sample_scores"), name="reference.sample_scores")
    reference_metric = _mapping(
        reference_aggregate.get("ast_exact_match"),
        name="reference.aggregate.ast_exact_match",
    )
    if reference_metric != {"numerator": 602, "denominator": 756}:
        raise MobileInt8RetentionError("candidate-v2 float reference changed")
    if reference_samples.get("rows") != 756:
        raise MobileInt8RetentionError("candidate-v2 reference sample count changed")
    for contract, name in (
        (reference_aggregate, "reference aggregate"),
        (reference_samples, "reference samples"),
    ):
        if not _is_sha256(contract.get("sha256")):
            raise MobileInt8RetentionError(f"{name} SHA-256 is invalid")

    evaluation = _mapping(protocol.get("evaluation"), name="evaluation")
    required_evaluation = {
        "generation_version_float": GENERATION_VERSION,
        "generation_version_int8": INT8_GENERATION_VERSION,
        "scorer_version": MOBILE_ACTIONS_SCORER_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "checkpoint_format": "int8",
        "device": "cpu",
        "decoding": "unconstrained_deterministic_greedy",
        "batch_size": 16,
        "max_new_tokens": 192,
        "no_checkpoint_format_fallback": True,
        "missing_parse_truncation_and_generation_failures_count_as_wrong": True,
        "sample_level_predictions_and_scores_required": True,
    }
    if evaluation != required_evaluation:
        raise MobileInt8RetentionError("int8 evaluation contract changed")

    gate = _mapping(protocol.get("retention_gate"), name="retention_gate")
    if gate != {
        "metric": "ast_exact_match",
        "minimum_int8_numerator": 587,
        "required_denominator": 756,
        "maximum_float_to_int8_correct_rows_lost": 15,
        "require_complete_aligned_samples": True,
        "require_same_sample_ids_scenarios_and_gold": True,
    }:
        raise MobileInt8RetentionError("int8 retention gate changed")
    return protocol


def _validate_manifest_and_audit(
    *, root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, list[dict[str, Any]], Path]:
    dataset = _mapping(protocol.get("dataset"), name="dataset")
    manifest_contract = _mapping(dataset.get("manifest"), name="dataset.manifest")
    audit_contract = _mapping(dataset.get("audit"), name="dataset.audit")
    manifest_path = _resolve(root, manifest_contract, name="Mobile development manifest")
    audit_path = _resolve(root, audit_contract, name="Mobile audit")
    _require_hash(
        manifest_path,
        manifest_contract.get("sha256"),
        name="Mobile development manifest",
    )
    _require_hash(audit_path, audit_contract.get("sha256"), name="Mobile audit")
    manifest_rows = _strict_jsonl(manifest_path)
    if len(manifest_rows) != manifest_contract.get("rows"):
        raise MobileInt8RetentionError("Mobile development row count changed")
    ids = [row.get("id") for row in manifest_rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids):
        raise MobileInt8RetentionError("Mobile development manifest has an invalid ID")
    if len(set(ids)) != len(ids):
        raise MobileInt8RetentionError("Mobile development manifest has duplicate IDs")

    audit = _strict_json(audit_path)
    official = _mapping(audit.get("official_eval"), name="audit.official_eval")
    if official.get("rows") != MOBILE_ACTIONS_FINAL_EVAL_ROWS:
        raise MobileInt8RetentionError("Mobile audit official-evaluation row count changed")
    if official.get("opaque_unparsed") is not True:
        raise MobileInt8RetentionError("Mobile official evaluation is no longer opaque")
    for field in (
        "labels_parsed",
        "lengths_computed",
        "materialized",
        "overlaps_computed",
        "prompts_parsed",
        "summaries_computed",
        "targets_parsed",
        "tool_schemas_parsed",
    ):
        if official.get(field) is not False:
            raise MobileInt8RetentionError(
                f"Mobile official-evaluation firewall changed at {field}"
            )
    counts = _mapping(audit.get("counts"), name="audit.counts")
    derived = _mapping(counts.get("derived"), name="audit.counts.derived")
    if derived.get("dev") != len(manifest_rows):
        raise MobileInt8RetentionError("Mobile audit and manifest row counts differ")
    return manifest_path, manifest_rows, audit_path


def _validate_score_evidence(
    *,
    aggregate_path: Path,
    samples_path: Path,
    expected_rows: int,
    manifest_ids: Sequence[str],
    expected_aggregate_sha256: object | None = None,
    expected_samples_sha256: object | None = None,
    expected_numerator: int | None = None,
    name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    if expected_aggregate_sha256 is not None:
        _require_hash(aggregate_path, expected_aggregate_sha256, name=f"{name} aggregate")
    if expected_samples_sha256 is not None:
        _require_hash(samples_path, expected_samples_sha256, name=f"{name} samples")
    aggregate = _strict_json(aggregate_path)
    samples = _strict_jsonl(samples_path)
    if aggregate.get("schema_version") != MOBILE_ACTIONS_SCORER_VERSION:
        raise MobileInt8RetentionError(f"{name} scorer version changed")
    if aggregate.get("evaluator_version") != EVALUATOR_VERSION:
        raise MobileInt8RetentionError(f"{name} evaluator version changed")
    if aggregate.get("sample_count") != expected_rows:
        raise MobileInt8RetentionError(f"{name} aggregate sample count changed")
    metric = _mapping(aggregate.get("ast_exact_match"), name=f"{name}.ast_exact_match")
    numerator = metric.get("numerator")
    denominator = metric.get("denominator")
    if type(numerator) is not int or denominator != expected_rows:
        raise MobileInt8RetentionError(f"{name} AST exact metric has the wrong population")
    if expected_numerator is not None and numerator != expected_numerator:
        raise MobileInt8RetentionError(f"{name} AST exact numerator changed")
    if len(samples) != expected_rows:
        raise MobileInt8RetentionError(f"{name} sample evidence is incomplete")
    sample_ids = [row.get("sample_id") for row in samples]
    if sample_ids != list(manifest_ids):
        raise MobileInt8RetentionError(f"{name} samples are not aligned to the manifest")
    if any(type(row.get("ast_exact")) is not bool for row in samples):
        raise MobileInt8RetentionError(f"{name} samples lack boolean ast_exact")
    if sum(bool(row["ast_exact"]) for row in samples) != numerator:
        raise MobileInt8RetentionError(f"{name} samples disagree with the aggregate")
    return aggregate, samples, numerator


def _validate_generation_evidence(
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    manifest_path: Path,
    protocol: Mapping[str, Any],
    info: Int8CheckpointInfo,
    manifest_ids: Sequence[str],
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "predictions.jsonl.manifest.json"
    summary = _strict_json(summary_path)
    expected_fields = {
        "batch_size",
        "checkpoint_dir",
        "checkpoint_format",
        "checkpoint_sha256",
        "device",
        "dtype",
        "elapsed_seconds",
        "examples",
        "failed",
        "generated",
        "manifest",
        "manifest_sha256",
        "max_new_tokens",
        "predictions",
        "predictions_sha256",
        "qengine",
        "quantization_manifest_sha256",
        "runtime",
        "schema_version",
        "source_checkpoint_sha256",
        "truncated",
    }
    if set(summary) != expected_fields:
        raise MobileInt8RetentionError("int8 generation receipt fields changed")
    evaluation = _mapping(protocol.get("evaluation"), name="evaluation")
    int8 = _mapping(protocol.get("int8_checkpoint"), name="int8_checkpoint")
    dataset = _mapping(protocol.get("dataset"), name="dataset")
    manifest_contract = _mapping(dataset.get("manifest"), name="dataset.manifest")
    if summary.get("schema_version") != INT8_GENERATION_VERSION:
        raise MobileInt8RetentionError("int8 generation receipt version changed")
    if summary.get("checkpoint_format") != "int8":
        raise MobileInt8RetentionError("int8 generation did not explicitly select int8")
    if summary.get("device") != "cpu":
        raise MobileInt8RetentionError("int8 generation was not CPU-only")
    if summary.get("dtype") != "torch.qint8_dynamic_with_float_remainder":
        raise MobileInt8RetentionError("int8 generation dtype label changed")
    if summary.get("qengine") != int8["runtime"]["qengine"]:
        raise MobileInt8RetentionError("int8 generation qengine changed")
    if summary.get("runtime") != {
        name: int8["runtime"][name]
        for name in ("platform_machine", "platform_system", "torch_version")
    }:
        raise MobileInt8RetentionError("int8 generation runtime changed")
    if summary.get("quantization_manifest_sha256") != int8.get("manifest_sha256"):
        raise MobileInt8RetentionError("int8 generation manifest identity changed")
    if summary.get("checkpoint_sha256") != dict(info.artifact_sha256):
        raise MobileInt8RetentionError("int8 generation artifact hashes changed")
    if summary.get("source_checkpoint_sha256") != dict(info.source_checkpoint_sha256):
        raise MobileInt8RetentionError("int8 generation source hashes changed")
    if Path(str(summary.get("checkpoint_dir"))).resolve() != checkpoint_dir:
        raise MobileInt8RetentionError("int8 generation checkpoint path changed")
    if Path(str(summary.get("manifest"))).resolve() != manifest_path:
        raise MobileInt8RetentionError("int8 generation manifest path changed")
    if Path(str(summary.get("predictions"))).resolve() != predictions_path.resolve():
        raise MobileInt8RetentionError("int8 generation predictions path changed")
    if summary.get("manifest_sha256") != manifest_contract.get("sha256"):
        raise MobileInt8RetentionError("int8 generation population changed")
    if summary.get("batch_size") != evaluation.get("batch_size"):
        raise MobileInt8RetentionError("int8 generation batch size changed")
    if summary.get("max_new_tokens") != evaluation.get("max_new_tokens"):
        raise MobileInt8RetentionError("int8 generation length cap changed")
    row_count = len(manifest_ids)
    for name in ("examples", "generated", "failed", "truncated"):
        if type(summary.get(name)) is not int or not 0 <= int(summary[name]) <= row_count:
            raise MobileInt8RetentionError(f"invalid int8 generation count {name}")
    if summary["examples"] != row_count or summary["generated"] + summary["failed"] != row_count:
        raise MobileInt8RetentionError("int8 generation evidence is incomplete")
    elapsed = summary.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise MobileInt8RetentionError("invalid int8 generation elapsed time")
    _require_hash(
        predictions_path,
        summary.get("predictions_sha256"),
        name="int8 predictions",
    )
    predictions = _strict_jsonl(predictions_path)
    if len(predictions) != row_count:
        raise MobileInt8RetentionError("int8 prediction evidence is incomplete")
    if [row.get("id") for row in predictions] != list(manifest_ids):
        raise MobileInt8RetentionError("int8 predictions are not aligned to the manifest")
    failed = sum(row.get("generation_failure") is not None for row in predictions)
    truncated = sum(row.get("truncated") is True for row in predictions)
    if failed != summary["failed"] or row_count - failed != summary["generated"]:
        raise MobileInt8RetentionError("int8 prediction failures disagree with generation receipt")
    if truncated != summary["truncated"]:
        raise MobileInt8RetentionError("int8 truncations disagree with generation receipt")
    return summary, predictions_path, predictions


def compare_retention_samples(
    *,
    int8_aggregate: Mapping[str, Any],
    int8_samples: Sequence[Mapping[str, Any]],
    float_samples: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the frozen integer gates and classify every paired transition."""

    gate_contract = _mapping(protocol.get("retention_gate"), name="retention_gate")
    reference = _mapping(protocol.get("source_float_reference"), name="source_float_reference")
    reference_aggregate = _mapping(reference.get("aggregate"), name="reference.aggregate")
    reference_metric = _mapping(
        reference_aggregate.get("ast_exact_match"), name="reference.ast_exact_match"
    )
    metric = _mapping(int8_aggregate.get("ast_exact_match"), name="int8.ast_exact_match")
    denominator = gate_contract.get("required_denominator")
    numerator = metric.get("numerator")
    if (
        type(numerator) is not int
        or metric.get("denominator") != denominator
        or len(int8_samples) != denominator
        or len(float_samples) != denominator
    ):
        raise MobileInt8RetentionError("paired retention evidence has the wrong population")

    transitions = {
        "retained_correct": 0,
        "fixed": 0,
        "regressed": 0,
        "retained_incorrect": 0,
    }
    paired: list[dict[str, Any]] = []
    for index, (int8_row, float_row) in enumerate(
        zip(int8_samples, float_samples, strict=True), start=1
    ):
        sample_id = int8_row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id != float_row.get("sample_id"):
            raise MobileInt8RetentionError(f"paired sample ID mismatch at row {index}")
        int8_exact = int8_row.get("ast_exact")
        float_exact = float_row.get("ast_exact")
        if type(int8_exact) is not bool or type(float_exact) is not bool:
            raise MobileInt8RetentionError(f"non-boolean paired outcome at {sample_id}")
        if int8_row.get("scenario") != float_row.get("scenario"):
            raise MobileInt8RetentionError(f"paired scenario mismatch at {sample_id}")
        if int8_row.get("gold") != float_row.get("gold"):
            raise MobileInt8RetentionError(f"paired gold AST mismatch at {sample_id}")
        if int8_exact and float_exact:
            transition = "retained_correct"
        elif int8_exact:
            transition = "fixed"
        elif float_exact:
            transition = "regressed"
        else:
            transition = "retained_incorrect"
        transitions[transition] += 1
        paired.append(
            {
                "schema_version": MOBILE_INT8_RETENTION_PAIRED_VERSION,
                "sample_id": sample_id,
                "scenario": int8_row.get("scenario"),
                "float_ast_exact": float_exact,
                "int8_ast_exact": int8_exact,
                "transition": transition,
            }
        )

    observed_int8 = sum(bool(row.get("ast_exact")) for row in int8_samples)
    if observed_int8 != numerator:
        raise MobileInt8RetentionError("int8 samples disagree with the aggregate")
    float_numerator = reference_metric.get("numerator")
    if type(float_numerator) is not int:
        raise MobileInt8RetentionError("float reference numerator is invalid")
    observed_float = sum(bool(row.get("ast_exact")) for row in float_samples)
    if observed_float != float_numerator:
        raise MobileInt8RetentionError("float samples disagree with the frozen reference")
    correct_rows_lost = float_numerator - numerator
    if transitions["regressed"] - transitions["fixed"] != correct_rows_lost:
        raise MobileInt8RetentionError("paired transitions disagree with aggregate loss")
    minimum = gate_contract.get("minimum_int8_numerator")
    maximum_loss = gate_contract.get("maximum_float_to_int8_correct_rows_lost")
    if type(minimum) is not int or type(maximum_loss) is not int:
        raise MobileInt8RetentionError("retention thresholds are invalid")
    accuracy_gate = numerator >= minimum
    loss_gate = correct_rows_lost <= maximum_loss
    passed = accuracy_gate and loss_gate
    return (
        {
            "metric": "ast_exact_match",
            "passed": passed,
            "status": "pass" if passed else "fail",
            "source_float": {
                "numerator": float_numerator,
                "denominator": denominator,
                "value": float_numerator / denominator,
            },
            "int8": {
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator,
            },
            "float_to_int8": {
                "correct_rows_lost": correct_rows_lost,
                "absolute_accuracy_change": (numerator - float_numerator) / denominator,
            },
            "thresholds": {
                "minimum_int8_numerator": minimum,
                "maximum_float_to_int8_correct_rows_lost": maximum_loss,
            },
            "component_gates": {
                "minimum_int8_accuracy": accuracy_gate,
                "maximum_float_to_int8_loss": loss_gate,
                "complete_aligned_samples": True,
                "explicit_int8_no_fallback": True,
            },
            "paired_transition_counts": transitions,
            "decision": (
                "retain_candidate_v2_int8_artifact"
                if passed
                else "reject_candidate_v2_int8_artifact"
            ),
        },
        paired,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise MobileInt8RetentionError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise MobileInt8RetentionError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _artifact_manifest(root: Path) -> dict[str, Any]:
    return {
        "schema_version": "barunaction-arm64-int8-retention-artifacts-v1",
        "file_sha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "artifact-sha256.json"
        },
    }


def create_retention_receipt(
    *,
    repository_root: str | Path,
    protocol_path: str | Path,
    int8_evaluation_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify int8 evaluation evidence and write an immutable paired retention receipt."""

    root = Path(repository_root).resolve()
    frozen_protocol_path = Path(protocol_path).resolve()
    protocol = load_retention_protocol(frozen_protocol_path)
    manifest_path, manifest_rows, audit_path = _validate_manifest_and_audit(
        root=root, protocol=protocol
    )
    manifest_ids = [str(row["id"]) for row in manifest_rows]

    candidate = _mapping(protocol.get("candidate"), name="candidate")
    int8_contract = _mapping(protocol.get("int8_checkpoint"), name="int8_checkpoint")
    checkpoint_dir = _resolve(root, int8_contract, name="int8 checkpoint")
    try:
        int8_info = verify_int8_checkpoint(
            checkpoint_dir,
            expected_manifest_sha256=str(int8_contract["manifest_sha256"]),
        )
    except QuantizationError as error:
        raise MobileInt8RetentionError(f"int8 checkpoint verification failed: {error}") from error
    if dict(int8_info.artifact_sha256) != dict(int8_contract["artifact_sha256"]):
        raise MobileInt8RetentionError("verified int8 artifact hashes differ from the protocol")
    if dict(int8_info.source_checkpoint_sha256) != dict(candidate["float_checkpoint_file_sha256"]):
        raise MobileInt8RetentionError("verified int8 source differs from candidate-v2")
    runtime_contract = _mapping(int8_contract.get("runtime"), name="int8_checkpoint.runtime")
    if int8_info.qengine != runtime_contract.get("qengine"):
        raise MobileInt8RetentionError("verified int8 qengine differs from the protocol")
    if int8_info.torch_version != runtime_contract.get("torch_version"):
        raise MobileInt8RetentionError("verified int8 torch version differs from the protocol")

    reference = _mapping(protocol.get("source_float_reference"), name="source_float_reference")
    float_aggregate_contract = _mapping(reference.get("aggregate"), name="reference.aggregate")
    float_samples_contract = _mapping(
        reference.get("sample_scores"), name="reference.sample_scores"
    )
    float_aggregate_path = _resolve(root, float_aggregate_contract, name="float aggregate")
    float_samples_path = _resolve(root, float_samples_contract, name="float samples")
    _, float_samples, _ = _validate_score_evidence(
        aggregate_path=float_aggregate_path,
        samples_path=float_samples_path,
        expected_rows=len(manifest_rows),
        manifest_ids=manifest_ids,
        expected_aggregate_sha256=float_aggregate_contract.get("sha256"),
        expected_samples_sha256=float_samples_contract.get("sha256"),
        expected_numerator=602,
        name="candidate-v2 float",
    )

    evaluation_dir = Path(int8_evaluation_dir).resolve()
    generation, predictions_path, predictions = _validate_generation_evidence(
        output_dir=evaluation_dir,
        checkpoint_dir=checkpoint_dir,
        manifest_path=manifest_path,
        protocol=protocol,
        info=int8_info,
        manifest_ids=manifest_ids,
    )
    int8_aggregate_path = evaluation_dir / "scores" / "aggregate.json"
    int8_samples_path = evaluation_dir / "scores" / "sample_scores.jsonl"
    int8_aggregate, int8_samples, _ = _validate_score_evidence(
        aggregate_path=int8_aggregate_path,
        samples_path=int8_samples_path,
        expected_rows=len(manifest_rows),
        manifest_ids=manifest_ids,
        name="candidate-v2 int8",
    )
    for prediction, score in zip(predictions, int8_samples, strict=True):
        sample_id = score.get("sample_id")
        if prediction.get("id") != sample_id:
            raise MobileInt8RetentionError(f"prediction/score ID mismatch at {sample_id}")
        if prediction.get("generation_failure") != score.get("generation_failure"):
            raise MobileInt8RetentionError(f"generation failure mismatch at {sample_id}")
        if prediction.get("truncated") != score.get("truncated"):
            raise MobileInt8RetentionError(f"truncation mismatch at {sample_id}")
        if (
            prediction.get("generation_failure") is not None or prediction.get("truncated")
        ) and score.get("ast_exact") is not False:
            raise MobileInt8RetentionError(
                f"failed or truncated generation counted as correct at {sample_id}"
            )

    gate, paired = compare_retention_samples(
        int8_aggregate=int8_aggregate,
        int8_samples=int8_samples,
        float_samples=float_samples,
        protocol=protocol,
    )

    output = Path(output_dir).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise MobileInt8RetentionError(
            f"refusing to overwrite retention receipt {output}"
        ) from error
    shutil.copy2(frozen_protocol_path, output / "protocol.json")
    paired_path = output / "paired-samples.jsonl"
    _write_jsonl(paired_path, paired)
    result = {
        "schema_version": MOBILE_INT8_RETENTION_RESULT_VERSION,
        "protocol": {
            "protocol_id": protocol.get("protocol_id"),
            "path": str(output / "protocol.json"),
            "sha256": MOBILE_INT8_RETENTION_PROTOCOL_SHA256,
        },
        "candidate": {
            "candidate_id": candidate.get("candidate_id"),
            "float_checkpoint_file_sha256": candidate.get("float_checkpoint_file_sha256"),
        },
        "int8_checkpoint": {
            "directory": str(checkpoint_dir),
            "format_version": INT8_FORMAT_VERSION,
            "manifest_sha256": int8_info.manifest_sha256,
            "artifact_sha256": dict(int8_info.artifact_sha256),
            "source_checkpoint_file_sha256": dict(int8_info.source_checkpoint_sha256),
            "qengine": int8_info.qengine,
        },
        "population": {
            "dataset_id": MOBILE_ACTIONS_DATASET_ID,
            "revision": MOBILE_ACTIONS_REVISION,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "audit": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
            "rows": len(manifest_rows),
            "official_evaluation_rows_opaque_unparsed": MOBILE_ACTIONS_FINAL_EVAL_ROWS,
            "official_evaluation_artifacts_accessed": [],
        },
        "generation": generation,
        "evidence": {
            "source_float_aggregate": str(float_aggregate_path),
            "source_float_aggregate_sha256": sha256_file(float_aggregate_path),
            "source_float_samples": str(float_samples_path),
            "source_float_samples_sha256": sha256_file(float_samples_path),
            "int8_predictions": str(predictions_path),
            "int8_predictions_sha256": sha256_file(predictions_path),
            "int8_generation_manifest": str(evaluation_dir / "predictions.jsonl.manifest.json"),
            "int8_generation_manifest_sha256": sha256_file(
                evaluation_dir / "predictions.jsonl.manifest.json"
            ),
            "int8_aggregate": str(int8_aggregate_path),
            "int8_aggregate_sha256": sha256_file(int8_aggregate_path),
            "int8_samples": str(int8_samples_path),
            "int8_samples_sha256": sha256_file(int8_samples_path),
            "paired_samples": str(paired_path),
            "paired_samples_sha256": sha256_file(paired_path),
        },
        "gate": gate,
        "interpretation": {
            "scope": "post_selection_arm64_int8_retention_diagnostic",
            "new_mobile_candidate_selection_trial": False,
            "official_mobile_evaluation_result": False,
            "sealed_safety_evidence": False,
            "larger_model_superiority_claim": False,
        },
    }
    _write_json(output / "result.json", result)
    _write_json(output / "artifact-sha256.json", _artifact_manifest(output))
    return result


__all__ = [
    "MOBILE_INT8_RETENTION_PAIRED_VERSION",
    "MOBILE_INT8_RETENTION_PROTOCOL_SHA256",
    "MOBILE_INT8_RETENTION_PROTOCOL_VERSION",
    "MOBILE_INT8_RETENTION_RESULT_VERSION",
    "MobileInt8RetentionError",
    "compare_retention_samples",
    "create_retention_receipt",
    "load_retention_protocol",
]
