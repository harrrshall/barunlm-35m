"""Frozen checkpoint interpolation and joint development gating for BarunAction-35M."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, NoReturn

import torch
from safetensors import safe_open
from safetensors.torch import load_model, save_file

from barunlm.config import BarunConfig
from barunlm.model import BarunLM
from barunlm.training.data import sha256_file

from .generation import REQUIRED_CHECKPOINT_FILES, verify_checkpoint
from .presto import (
    PRESTO_PHENOMENON_TAXONOMY_VERSION,
    PRESTO_SCORER_VERSION,
    PRESTO_USER_REVISION_ALIAS_SET_VERSION,
    PRESTO_USER_REVISION_RAW_LABELS_V2,
)

INTERPOLATION_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "interpolation_rescue_v1.json"
)
INTERPOLATION_PROTOCOL_SHA256 = (
    "1e2f005f61bf8f8cb62c01928c23266023e2e20760321832f80bc4dbe3c9ee3c"
)
INTERPOLATION_CHECKPOINT_SCHEMA_VERSION = "barun-interpolation-checkpoint-v1"
INTERPOLATION_RESULT_SCHEMA_VERSION = "barun-interpolation-rescue-result-v1"
_EXPECTED_ARMS = (
    ("alpha-025", 1, 4),
    ("alpha-050", 1, 2),
    ("alpha-075", 3, 4),
)


class InterpolationRescueError(RuntimeError):
    """The frozen interpolation protocol or evidence is inconsistent."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise InterpolationRescueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> NoReturn:
    raise InterpolationRescueError(f"non-finite JSON constant {value!r}")


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except InterpolationRescueError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InterpolationRescueError(f"invalid JSON artifact {source}") from error
    if not isinstance(value, dict):
        raise InterpolationRescueError(f"JSON artifact must be an object: {source}")
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InterpolationRescueError(f"{name} must be an object")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InterpolationRescueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InterpolationRescueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InterpolationRescueError(f"{name} must be finite")
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hashes(value: object, name: str) -> dict[str, str]:
    payload = _mapping(value, name)
    normalized = {str(key): str(digest) for key, digest in payload.items()}
    if set(normalized) != set(REQUIRED_CHECKPOINT_FILES):
        raise InterpolationRescueError(
            f"{name} must contain exactly {sorted(REQUIRED_CHECKPOINT_FILES)}"
        )
    if any(not _is_sha256(digest) for digest in normalized.values()):
        raise InterpolationRescueError(f"{name} contains an invalid SHA-256")
    return normalized


def _metric_value(payload: Mapping[str, Any], path: tuple[str, ...]) -> float:
    current: object = payload
    for key in path:
        current = _mapping(current, ".".join(path)).get(key)
    return _number(current, ".".join(path))


def _metric_integer(payload: Mapping[str, Any], path: tuple[str, ...]) -> int:
    current: object = payload
    for key in path:
        current = _mapping(current, ".".join(path)).get(key)
    return _integer(current, ".".join(path))


def load_interpolation_protocol(
    path: str | Path = INTERPOLATION_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Load only the immutable, checked-in three-arm interpolation protocol."""

    source = Path(path)
    actual = sha256_file(source)
    if actual != INTERPOLATION_PROTOCOL_SHA256:
        raise InterpolationRescueError(
            "interpolation protocol SHA-256 mismatch: "
            f"expected {INTERPOLATION_PROTOCOL_SHA256}, got {actual}"
        )
    protocol = _strict_json(source)
    if protocol.get("schema_version") != "barun-interpolation-rescue-protocol-v1":
        raise InterpolationRescueError("unsupported interpolation protocol version")
    if protocol.get("status") != "frozen_before_any_interpolation_arm_is_scored":
        raise InterpolationRescueError("interpolation protocol is not frozen before scoring")

    trajectory = _mapping(protocol.get("trajectory"), "trajectory")
    candidate = _mapping(trajectory.get("candidate_v2"), "trajectory.candidate_v2")
    presto = _mapping(trajectory.get("presto"), "trajectory.presto")
    candidate_hashes = _hashes(candidate.get("file_sha256"), "candidate-v2 hashes")
    presto_hashes = _hashes(presto.get("file_sha256"), "PRESTO hashes")
    recorded_input = _hashes(
        presto.get("recorded_input_checkpoint_sha256"),
        "PRESTO recorded input hashes",
    )
    if recorded_input != candidate_hashes:
        raise InterpolationRescueError("PRESTO endpoint is not recorded as candidate-v2 continuation")
    for name in ("barun_config.json", "tokenizer.json"):
        if candidate_hashes[name] != presto_hashes[name]:
            raise InterpolationRescueError(f"trajectory endpoints differ at {name}")
    for endpoint, name in ((candidate, "candidate-v2"), (presto, "PRESTO")):
        if not _is_sha256(endpoint.get("checkpoint_manifest_sha256")):
            raise InterpolationRescueError(f"{name} checkpoint manifest hash is invalid")

    interpolation = _mapping(protocol.get("interpolation"), "interpolation")
    arms = interpolation.get("arms")
    if not isinstance(arms, list):
        raise InterpolationRescueError("interpolation arms must be an array")
    observed_arms = []
    for index, arm in enumerate(arms):
        arm_mapping = _mapping(arm, f"interpolation.arms[{index}]")
        observed_arms.append(
            (
                arm_mapping.get("arm_id"),
                _integer(arm_mapping.get("numerator"), "alpha numerator", minimum=1),
                _integer(arm_mapping.get("denominator"), "alpha denominator", minimum=2),
            )
        )
    if tuple(observed_arms) != _EXPECTED_ARMS:
        raise InterpolationRescueError("frozen alpha arms changed")
    trial_budget = _mapping(interpolation.get("trial_budget"), "trial budget")
    expected_budget = {
        "checkpoint_arms": 3,
        "mobile_generations_per_arm": 1,
        "presto_generations_per_arm": 1,
        "adaptive_followup_alphas": 0,
        "endpoint_regenerations": 0,
        "early_stopping_before_both_evaluations": False,
        "failed_or_truncated_generations_count_as_wrong": True,
    }
    if dict(trial_budget) != expected_budget:
        raise InterpolationRescueError("interpolation trial budget changed")
    if interpolation.get("accumulator_device") != "cpu":
        raise InterpolationRescueError("interpolation must use a CPU accumulator")
    if interpolation.get("accumulator_dtype") != "float64":
        raise InterpolationRescueError("interpolation must use float64 accumulation")
    if interpolation.get("source_and_output_tensor_dtype") != "float32":
        raise InterpolationRescueError("endpoint and output tensors must remain float32")

    mobile = _mapping(protocol.get("mobile_evaluation"), "mobile evaluation")
    if mobile.get("minimum_ast_exact_numerator") != 587 or mobile.get(
        "required_ast_exact_denominator"
    ) != 756:
        raise InterpolationRescueError("Mobile joint gate changed")
    if mobile.get("preregistration_sha256") != (
        "96afa0ac407c53fd375ed0910a08ec66463718e3b826cf02b73f8bbc87a3c47e"
    ):
        raise InterpolationRescueError("Mobile regression protocol identity changed")

    presto_evaluation = _mapping(protocol.get("presto_evaluation"), "PRESTO evaluation")
    if presto_evaluation.get("scorer_version") != PRESTO_SCORER_VERSION:
        raise InterpolationRescueError("PRESTO scorer differs from the corrected scorer")
    if presto_evaluation.get("taxonomy_version") != PRESTO_PHENOMENON_TAXONOMY_VERSION:
        raise InterpolationRescueError("PRESTO corrected taxonomy version changed")
    if (
        presto_evaluation.get("revision_alias_set_version")
        != PRESTO_USER_REVISION_ALIAS_SET_VERSION
    ):
        raise InterpolationRescueError("PRESTO corrected revision alias set changed")
    expected_labels = set(PRESTO_USER_REVISION_RAW_LABELS_V2)
    configured_labels = set(
        _mapping(
            presto_evaluation.get("revision_raw_label_rows"),
            "PRESTO revision label rows",
        )
    )
    if configured_labels != expected_labels:
        raise InterpolationRescueError("PRESTO corrected revision family is incomplete")
    return protocol


def interpolation_arms(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return the frozen alpha arms after validating their reduced fractions."""

    interpolation = _mapping(protocol.get("interpolation"), "interpolation")
    arms = interpolation.get("arms")
    if not isinstance(arms, list):
        raise InterpolationRescueError("interpolation arms must be an array")
    result: list[dict[str, Any]] = []
    for index, arm in enumerate(arms):
        payload = _mapping(arm, f"arm {index}")
        arm_id = payload.get("arm_id")
        if not isinstance(arm_id, str) or not arm_id:
            raise InterpolationRescueError(f"arm {index} lacks an ID")
        numerator = _integer(payload.get("numerator"), f"{arm_id} numerator", minimum=1)
        denominator = _integer(payload.get("denominator"), f"{arm_id} denominator", minimum=2)
        alpha = Fraction(numerator, denominator)
        if alpha.numerator != numerator or alpha.denominator != denominator or not 0 < alpha < 1:
            raise InterpolationRescueError(f"{arm_id} alpha is not a reduced interior fraction")
        result.append(
            {
                "arm_id": arm_id,
                "numerator": numerator,
                "denominator": denominator,
                "value": float(alpha),
            }
        )
    return tuple(result)


def endpoint_hashes(protocol: Mapping[str, Any], endpoint: str) -> dict[str, str]:
    trajectory = _mapping(protocol.get("trajectory"), "trajectory")
    payload = _mapping(trajectory.get(endpoint), f"trajectory.{endpoint}")
    return _hashes(payload.get("file_sha256"), f"{endpoint} hashes")


def _endpoint_metadata(path: Path) -> tuple[list[str], dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return list(handle.keys()), dict(handle.metadata() or {})


def interpolate_safetensors(
    candidate_path: str | Path,
    presto_path: str | Path,
    output_path: str | Path,
    *,
    numerator: int,
    denominator: int,
    expected_metadata: Mapping[str, str],
    expected_tensor_count: int,
    expected_parameter_count: int,
) -> dict[str, Any]:
    """Interpolate exact stored tensors with float64 CPU accumulation.

    Source float32 values are exactly representable in float64. Integer-weighted dyadic
    arithmetic is performed in float64 and rounded once when cast back to float32.
    """

    alpha = Fraction(numerator, denominator)
    if alpha.numerator != numerator or alpha.denominator != denominator or not 0 < alpha < 1:
        raise InterpolationRescueError("alpha must be a reduced fraction strictly between 0 and 1")
    candidate = Path(candidate_path)
    presto = Path(presto_path)
    output = Path(output_path)
    if output.exists():
        raise InterpolationRescueError(f"refusing to overwrite interpolation tensor file {output}")
    if not candidate.is_file() or not presto.is_file():
        raise InterpolationRescueError("both interpolation tensor files must exist")

    candidate_keys, candidate_metadata = _endpoint_metadata(candidate)
    presto_keys, presto_metadata = _endpoint_metadata(presto)
    expected_metadata_dict = dict(expected_metadata)
    if candidate_metadata != expected_metadata_dict or presto_metadata != expected_metadata_dict:
        raise InterpolationRescueError("endpoint tied-weight metadata differs from the protocol")
    if candidate_keys != presto_keys:
        raise InterpolationRescueError("endpoint safetensors key sets or ordering differ")
    if len(candidate_keys) != expected_tensor_count:
        raise InterpolationRescueError("stored tensor count differs from the protocol")

    tensors: dict[str, torch.Tensor] = {}
    total_parameters = 0
    with (
        safe_open(candidate, framework="pt", device="cpu") as candidate_handle,
        safe_open(presto, framework="pt", device="cpu") as presto_handle,
    ):
        for key in candidate_keys:
            left = candidate_handle.get_tensor(key)
            right = presto_handle.get_tensor(key)
            if left.shape != right.shape:
                raise InterpolationRescueError(f"endpoint tensor shape mismatch at {key}")
            if left.dtype != right.dtype:
                raise InterpolationRescueError(f"endpoint tensor dtype mismatch at {key}")
            total_parameters += left.numel()
            if torch.is_floating_point(left):
                if left.dtype is not torch.float32:
                    raise InterpolationRescueError(f"endpoint floating tensor is not float32 at {key}")
                if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                    raise InterpolationRescueError(f"endpoint tensor contains non-finite values at {key}")
                accumulator = left.to(dtype=torch.float64)
                accumulator.mul_(denominator - numerator)
                accumulator.add_(right, alpha=numerator)
                accumulator.div_(denominator)
                interpolated = accumulator.to(dtype=left.dtype)
                if not torch.isfinite(interpolated).all():
                    raise InterpolationRescueError(
                        f"interpolated tensor contains non-finite values at {key}"
                    )
                tensors[key] = interpolated.contiguous()
            else:
                if not torch.equal(left, right):
                    raise InterpolationRescueError(
                        f"non-floating endpoint tensor differs and cannot be interpolated at {key}"
                    )
                tensors[key] = left.clone().contiguous()
    if total_parameters != expected_parameter_count:
        raise InterpolationRescueError("stored parameter count differs from the protocol")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output, metadata=expected_metadata_dict)
    output_keys, output_metadata = _endpoint_metadata(output)
    if output_keys != candidate_keys or output_metadata != expected_metadata_dict:
        raise InterpolationRescueError("saved interpolation tensor contract changed")
    return {
        "alpha": {
            "numerator": numerator,
            "denominator": denominator,
            "value": float(alpha),
        },
        "accumulator_device": "cpu",
        "accumulator_dtype": "torch.float64",
        "output_dtype": "torch.float32",
        "stored_tensor_count": len(output_keys),
        "stored_parameter_count": total_parameters,
        "safetensors_metadata": output_metadata,
        "model_sha256": sha256_file(output),
    }


def _verify_endpoint_manifest(
    checkpoint_dir: Path,
    *,
    expected_manifest_sha256: str,
    expected_hashes: Mapping[str, str],
    expected_input_hashes: Mapping[str, str] | None = None,
) -> None:
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise InterpolationRescueError(f"checkpoint manifest SHA-256 mismatch: {manifest_path}")
    manifest = _strict_json(manifest_path)
    if _hashes(manifest.get("file_sha256"), "manifest file hashes") != dict(expected_hashes):
        raise InterpolationRescueError("checkpoint manifest file hashes changed")
    if expected_input_hashes is not None:
        observed = _hashes(
            manifest.get("input_checkpoint_sha256"),
            "PRESTO manifest input hashes",
        )
        if observed != dict(expected_input_hashes):
            raise InterpolationRescueError("PRESTO manifest does not continue candidate-v2")


def interpolate_checkpoint(
    candidate_dir: str | Path,
    presto_dir: str | Path,
    output_dir: str | Path,
    *,
    arm: Mapping[str, Any],
    protocol: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Create and fully verify one immutable release-format interpolation checkpoint."""

    candidate = Path(candidate_dir).resolve()
    presto = Path(presto_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise InterpolationRescueError(f"refusing to overwrite interpolation checkpoint {output}")
    candidate_hashes = endpoint_hashes(protocol, "candidate_v2")
    presto_hashes = endpoint_hashes(protocol, "presto")
    verify_checkpoint(candidate, expected_sha256=candidate_hashes)
    verify_checkpoint(presto, expected_sha256=presto_hashes)
    trajectory = _mapping(protocol.get("trajectory"), "trajectory")
    candidate_contract = _mapping(trajectory.get("candidate_v2"), "candidate-v2 endpoint")
    presto_contract = _mapping(trajectory.get("presto"), "PRESTO endpoint")
    _verify_endpoint_manifest(
        candidate,
        expected_manifest_sha256=str(candidate_contract["checkpoint_manifest_sha256"]),
        expected_hashes=candidate_hashes,
    )
    _verify_endpoint_manifest(
        presto,
        expected_manifest_sha256=str(presto_contract["checkpoint_manifest_sha256"]),
        expected_hashes=presto_hashes,
        expected_input_hashes=candidate_hashes,
    )
    for name in ("barun_config.json", "tokenizer.json"):
        if (candidate / name).read_bytes() != (presto / name).read_bytes():
            raise InterpolationRescueError(f"endpoint {name} bytes differ")
    config = BarunConfig.from_json(candidate / "barun_config.json")
    if config.tie_embeddings is not True:
        raise InterpolationRescueError("BarunLM endpoint does not use tied embeddings")

    interpolation = _mapping(protocol.get("interpolation"), "interpolation")
    metadata = _mapping(
        interpolation.get("expected_safetensors_metadata"),
        "expected safetensors metadata",
    )
    arm_id = arm.get("arm_id")
    if not isinstance(arm_id, str) or not arm_id:
        raise InterpolationRescueError("interpolation arm lacks an ID")
    numerator = _integer(arm.get("numerator"), "alpha numerator", minimum=1)
    denominator = _integer(arm.get("denominator"), "alpha denominator", minimum=2)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{arm_id}-", dir=output.parent))
    try:
        shutil.copy2(candidate / "barun_config.json", temporary / "barun_config.json")
        shutil.copy2(candidate / "tokenizer.json", temporary / "tokenizer.json")
        tensor_evidence = interpolate_safetensors(
            candidate / "model.safetensors",
            presto / "model.safetensors",
            temporary / "model.safetensors",
            numerator=numerator,
            denominator=denominator,
            expected_metadata={str(key): str(value) for key, value in metadata.items()},
            expected_tensor_count=_integer(
                interpolation.get("expected_stored_tensor_count"),
                "expected tensor count",
                minimum=1,
            ),
            expected_parameter_count=_integer(
                interpolation.get("expected_stored_parameter_count"),
                "expected parameter count",
                minimum=1,
            ),
        )
        output_hashes = {
            name: sha256_file(temporary / name) for name in REQUIRED_CHECKPOINT_FILES
        }
        manifest = {
            "schema_version": "barun-release-checkpoint-v1",
            "interpolation_schema_version": INTERPOLATION_CHECKPOINT_SCHEMA_VERSION,
            "run_id": run_id,
            "arm_id": arm_id,
            "alpha": tensor_evidence["alpha"],
            "equation": interpolation.get("equation"),
            "rounding": interpolation.get("rounding"),
            "source_checkpoint_sha256": {
                "candidate_v2": candidate_hashes,
                "presto": presto_hashes,
            },
            "source_checkpoint_manifest_sha256": {
                "candidate_v2": candidate_contract["checkpoint_manifest_sha256"],
                "presto": presto_contract["checkpoint_manifest_sha256"],
            },
            "tensor_evidence": tensor_evidence,
            "file_sha256": output_hashes,
        }
        manifest_path = temporary / "checkpoint_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        verify_checkpoint(temporary, expected_sha256=output_hashes)
        loaded_config = BarunConfig.from_json(temporary / "barun_config.json")
        model = BarunLM(loaded_config)
        missing, unexpected = load_model(
            model,
            temporary / "model.safetensors",
            strict=False,
        )
        if missing or unexpected:
            raise InterpolationRescueError(
                f"interpolation checkpoint tensor mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
        if model.lm_head.weight.data_ptr() != model.embedding.weight.data_ptr():
            raise InterpolationRescueError("loaded interpolation checkpoint lost its tied weights")
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "arm_id": arm_id,
        "alpha": tensor_evidence["alpha"],
        "directory": str(output),
        "file_sha256": output_hashes,
        "checkpoint_manifest_sha256": sha256_file(output / "checkpoint_manifest.json"),
        "tensor_evidence": tensor_evidence,
        "tied_weight_storage_verified": True,
    }


def evaluate_presto_gate(
    aggregate: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the corrected, full-family PRESTO development gate without repair."""

    contract = _mapping(protocol.get("presto_evaluation"), "PRESTO evaluation")
    if aggregate.get("schema_version") != contract.get("scorer_version"):
        raise InterpolationRescueError("PRESTO aggregate uses the wrong scorer")
    if aggregate.get("metric_scope") != contract.get("metric_scope"):
        raise InterpolationRescueError("PRESTO aggregate metric scope changed")
    if aggregate.get("sample_count") != contract.get("rows"):
        raise InterpolationRescueError("PRESTO aggregate sample count changed")
    thresholds = _mapping(contract.get("gate"), "PRESTO gate thresholds")
    exact = _metric_value(aggregate, ("ast_exact_match", "value"))
    schema = _metric_value(aggregate, ("schema_valid", "value"))
    abstention = _metric_value(aggregate, ("abstention", "f1"))
    false_call = _metric_value(aggregate, ("false_call_on_gate", "value"))
    buckets = _mapping(
        aggregate.get("headline_phenomenon_buckets"),
        "PRESTO headline phenomenon buckets",
    )
    expected_bucket_rows = _mapping(
        contract.get("headline_bucket_rows"),
        "PRESTO headline bucket rows",
    )
    bucket_values: dict[str, float] = {}
    bucket_counts_match = True
    for name in ("no_phenomenon", "revision", "disfluency"):
        bucket = _mapping(buckets.get(name), f"PRESTO {name} bucket")
        count = _integer(bucket.get("count"), f"PRESTO {name} count", minimum=1)
        bucket_counts_match &= count == expected_bucket_rows.get(name)
        bucket_values[name] = _metric_value(bucket, ("ast_exact_match", "value"))

    per_phenomenon = _mapping(
        aggregate.get("per_phenomenon"),
        "PRESTO raw phenomenon buckets",
    )
    expected_revision_rows = _mapping(
        contract.get("revision_raw_label_rows"),
        "PRESTO revision raw-label rows",
    )
    raw_family_counts_match = set(expected_revision_rows) == set(
        PRESTO_USER_REVISION_RAW_LABELS_V2
    )
    observed_revision_rows: dict[str, int] = {}
    for label in PRESTO_USER_REVISION_RAW_LABELS_V2:
        raw_bucket = _mapping(per_phenomenon.get(label), f"PRESTO raw bucket {label}")
        observed_count = _integer(
            raw_bucket.get("count"), f"PRESTO raw count {label}", minimum=1
        )
        observed_revision_rows[label] = observed_count
        raw_family_counts_match &= observed_count == expected_revision_rows.get(label)

    gaps = {
        name: bucket_values["no_phenomenon"] - bucket_values[name]
        for name in ("revision", "disfluency")
    }
    maximum_gap = _number(
        thresholds.get("maximum_no_phenomenon_to_revision_or_disfluency_gap"),
        "PRESTO maximum hard-bucket gap",
    )
    represented = all(
        _integer(_mapping(buckets.get(name), name).get("count"), f"{name} count", minimum=1)
        > 0
        for name in ("no_phenomenon", "revision", "disfluency")
    )
    checks = {
        "derived_ast_exact_at_least_threshold": exact
        >= _number(thresholds.get("derived_ast_exact_at_least"), "PRESTO exact threshold"),
        "schema_valid_at_least_threshold": schema
        >= _number(thresholds.get("schema_valid_at_least"), "PRESTO schema threshold"),
        "abstention_f1_at_least_threshold": abstention
        >= _number(thresholds.get("abstention_f1_at_least"), "PRESTO abstention threshold"),
        "false_call_on_gate_at_most_threshold": false_call
        <= _number(thresholds.get("false_call_on_gate_at_most"), "PRESTO false-call threshold"),
        "hard_buckets_represented": represented
        and thresholds.get("require_represented_gap_buckets") is True,
        "revision_gap_at_most_threshold": represented and gaps["revision"] <= maximum_gap,
        "disfluency_gap_at_most_threshold": represented and gaps["disfluency"] <= maximum_gap,
        "corrected_revision_family_complete": raw_family_counts_match,
        "frozen_bucket_membership_counts_match": bucket_counts_match,
    }
    return {
        "thresholds": dict(thresholds),
        "observed": {
            "derived_ast_exact": exact,
            "schema_valid": schema,
            "abstention_f1": abstention,
            "false_call_on_gate": false_call,
            "headline_bucket_ast_exact": bucket_values,
            "no_phenomenon_minus_hard_bucket_gap": gaps,
            "revision_raw_label_rows": dict(sorted(observed_revision_rows.items())),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate_joint_gate(
    mobile_result: Mapping[str, Any],
    presto_aggregate: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the Mobile integer floor and every corrected PRESTO gate check."""

    mobile_contract = _mapping(protocol.get("mobile_evaluation"), "Mobile evaluation")
    mobile_gate = _mapping(mobile_result.get("gate"), "Mobile gate")
    candidate = _mapping(mobile_gate.get("candidate"), "Mobile candidate metric")
    numerator = _integer(candidate.get("numerator"), "Mobile exact numerator")
    denominator = _integer(candidate.get("denominator"), "Mobile exact denominator", minimum=1)
    required_numerator = _integer(
        mobile_contract.get("minimum_ast_exact_numerator"),
        "Mobile minimum numerator",
        minimum=1,
    )
    required_denominator = _integer(
        mobile_contract.get("required_ast_exact_denominator"),
        "Mobile required denominator",
        minimum=1,
    )
    if denominator != required_denominator:
        raise InterpolationRescueError("Mobile result denominator changed")
    mobile_passed = numerator >= required_numerator
    if mobile_gate.get("passed") is not mobile_passed:
        raise InterpolationRescueError("Mobile nested gate disagrees with the frozen integer gate")
    presto_gate = evaluate_presto_gate(presto_aggregate, protocol)
    return {
        "mobile": {
            "observed_numerator": numerator,
            "required_numerator": required_numerator,
            "denominator": denominator,
            "passed": mobile_passed,
        },
        "presto": presto_gate,
        "passed": mobile_passed and presto_gate["passed"],
    }


def select_interpolation_arm(
    arm_results: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen lexicographic rule after all three arms have full evidence."""

    expected = interpolation_arms(protocol)
    if len(arm_results) != len(expected):
        raise InterpolationRescueError("selection requires every frozen interpolation arm")
    by_id: dict[str, Mapping[str, Any]] = {}
    for result in arm_results:
        arm_id = result.get("arm_id")
        if not isinstance(arm_id, str) or arm_id in by_id:
            raise InterpolationRescueError("arm results contain a missing or duplicate arm ID")
        by_id[arm_id] = result
    if set(by_id) != {str(arm["arm_id"]) for arm in expected}:
        raise InterpolationRescueError("arm result IDs differ from the frozen protocol")

    passers: list[tuple[int, int, Fraction, str]] = []
    for arm in expected:
        arm_id = str(arm["arm_id"])
        result = by_id[arm_id]
        result_alpha = _mapping(result.get("alpha"), f"{arm_id} alpha")
        if (
            result_alpha.get("numerator") != arm["numerator"]
            or result_alpha.get("denominator") != arm["denominator"]
        ):
            raise InterpolationRescueError(f"{arm_id} result alpha differs from the protocol")
        joint = _mapping(result.get("joint_gate"), f"{arm_id} joint gate")
        if type(joint.get("passed")) is not bool:
            raise InterpolationRescueError(f"{arm_id} joint gate lacks a boolean decision")
        presto_aggregate = _mapping(result.get("presto_aggregate"), f"{arm_id} PRESTO metrics")
        mobile_result = _mapping(result.get("mobile_result"), f"{arm_id} Mobile result")
        recomputed_joint = evaluate_joint_gate(mobile_result, presto_aggregate, protocol)
        if joint["passed"] is not recomputed_joint["passed"]:
            raise InterpolationRescueError(
                f"{arm_id} recorded joint gate disagrees with independently recomputed metrics"
            )
        if not recomputed_joint["passed"]:
            continue
        presto_exact = _metric_integer(
            presto_aggregate,
            ("ast_exact_match", "numerator"),
        )
        mobile_exact = _metric_integer(
            _mapping(mobile_result.get("gate"), f"{arm_id} Mobile gate"),
            ("candidate", "numerator"),
        )
        alpha = Fraction(int(arm["numerator"]), int(arm["denominator"]))
        passers.append((presto_exact, mobile_exact, alpha, arm_id))
    ordered = sorted(passers, key=lambda item: (-item[0], -item[1], item[2], item[3]))
    if not ordered:
        return {
            "status": "reject_interpolation_no_joint_passer",
            "selected_arm_id": None,
            "ordered_joint_passers": [],
            "fallback": "candidate-v2",
            "additional_alpha_authorized": False,
        }
    winner = ordered[0]
    return {
        "status": "select_interpolation_joint_passer",
        "selected_arm_id": winner[3],
        "ordered_joint_passers": [
            {
                "arm_id": arm_id,
                "presto_ast_exact_numerator": presto_exact,
                "mobile_ast_exact_numerator": mobile_exact,
                "alpha": {
                    "numerator": alpha.numerator,
                    "denominator": alpha.denominator,
                    "value": float(alpha),
                },
            }
            for presto_exact, mobile_exact, alpha, arm_id in ordered
        ],
        "fallback": None,
        "additional_alpha_authorized": False,
    }


def file_manifest(root: str | Path) -> dict[str, str]:
    """Hash an immutable interpolation bundle, excluding its self-referential receipt."""

    directory = Path(root)
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "artifact-sha256.json"
    }


def protocol_sha256() -> str:
    """Return the frozen protocol hash through an independently computed digest."""

    return hashlib.sha256(INTERPOLATION_PROTOCOL_PATH.read_bytes()).hexdigest()


__all__ = [
    "INTERPOLATION_CHECKPOINT_SCHEMA_VERSION",
    "INTERPOLATION_PROTOCOL_PATH",
    "INTERPOLATION_PROTOCOL_SHA256",
    "INTERPOLATION_RESULT_SCHEMA_VERSION",
    "InterpolationRescueError",
    "endpoint_hashes",
    "evaluate_joint_gate",
    "evaluate_presto_gate",
    "file_manifest",
    "interpolate_checkpoint",
    "interpolate_safetensors",
    "interpolation_arms",
    "load_interpolation_protocol",
    "protocol_sha256",
    "select_interpolation_arm",
]
