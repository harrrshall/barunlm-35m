"""Frozen Mobile Actions development regression gate for compatible checkpoints.

This module can read only the already-derived internal-train development manifest.  It
does not accept a dataset source or an official-evaluation path, and it verifies every
input artifact before loading model tensors.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, NoReturn

from barunlm.datasets.mobile_actions import (
    MOBILE_ACTIONS_DATASET_ID,
    MOBILE_ACTIONS_FINAL_EVAL_ROWS,
    MOBILE_ACTIONS_REVISION,
)
from barunlm.training.data import EXAMPLE_SCHEMA_VERSION, sha256_file

from .evaluator import EVALUATOR_VERSION
from .generation import (
    GENERATION_VERSION,
    REQUIRED_CHECKPOINT_FILES,
    GenerationSummary,
    generate_manifest,
    verify_checkpoint,
)
from .mobile_actions import (
    MOBILE_ACTIONS_SCORER_VERSION,
    read_jsonl,
    write_scores,
)

MOBILE_REGRESSION_PREREGISTRATION_VERSION = "barun-mobile-regression-preregistration-v1"
MOBILE_REGRESSION_RESULT_VERSION = "barun-mobile-regression-result-v1"
MOBILE_REGRESSION_PAIRED_VERSION = "barun-mobile-regression-paired-sample-v1"
MOBILE_REGRESSION_PREREGISTRATION_SHA256 = (
    "96afa0ac407c53fd375ed0910a08ec66463718e3b826cf02b73f8bbc87a3c47e"
)


class MobileRegressionError(RuntimeError):
    """A frozen regression input or result violates the preregistered contract."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise MobileRegressionError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> NoReturn:
    raise MobileRegressionError(f"non-finite JSON constant {value!r}")


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except MobileRegressionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MobileRegressionError(f"cannot read strict JSON from {source}") from error
    if not isinstance(payload, dict):
        raise MobileRegressionError(f"JSON artifact must contain an object: {source}")
    return payload


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MobileRegressionError(f"{name} must be a JSON object")
    return value


def _require_exact_hash(path: str | Path, expected: object, *, name: str) -> str:
    if not _is_sha256(expected):
        raise MobileRegressionError(f"{name} preregistration SHA-256 is invalid")
    actual = sha256_file(path)
    if actual != expected:
        raise MobileRegressionError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def load_preregistration(path: str | Path) -> dict[str, Any]:
    """Load only the checked-in immutable v1 regression preregistration."""

    actual = _require_exact_hash(
        path,
        MOBILE_REGRESSION_PREREGISTRATION_SHA256,
        name="Mobile regression preregistration",
    )
    payload = _strict_json(path)
    if payload.get("schema_version") != MOBILE_REGRESSION_PREREGISTRATION_VERSION:
        raise MobileRegressionError("unsupported Mobile regression preregistration version")
    if payload.get("protocol_id") != "mobile-derived-dev-regression-v1":
        raise MobileRegressionError("unexpected Mobile regression protocol ID")

    dataset = _require_mapping(payload.get("dataset"), name="dataset")
    if dataset.get("dataset_id") != MOBILE_ACTIONS_DATASET_ID:
        raise MobileRegressionError("Mobile regression dataset ID changed")
    if dataset.get("revision") != MOBILE_ACTIONS_REVISION:
        raise MobileRegressionError("Mobile regression dataset revision changed")
    manifest = _require_mapping(dataset.get("manifest"), name="dataset.manifest")
    audit = _require_mapping(dataset.get("audit"), name="dataset.audit")
    firewall = _require_mapping(
        dataset.get("official_evaluation_firewall"),
        name="dataset.official_evaluation_firewall",
    )
    if manifest.get("rows") != 756:
        raise MobileRegressionError("frozen Mobile development row count changed")
    if not _is_sha256(manifest.get("sha256")) or not _is_sha256(audit.get("sha256")):
        raise MobileRegressionError("frozen Mobile data hashes are invalid")
    if firewall.get("rows") != MOBILE_ACTIONS_FINAL_EVAL_ROWS:
        raise MobileRegressionError("official Mobile evaluation row count changed")
    if firewall.get("accepted_input_paths") != []:
        raise MobileRegressionError("official evaluation paths must not be accepted")
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
        if firewall.get(field) is not False:
            raise MobileRegressionError(f"official evaluation firewall changed at {field}")
    if firewall.get("opaque_unparsed") is not True:
        raise MobileRegressionError("official evaluation must remain opaque and unparsed")

    reference = _require_mapping(payload.get("reference"), name="reference")
    reference_aggregate = _require_mapping(reference.get("aggregate"), name="reference.aggregate")
    reference_metric = _require_mapping(
        reference_aggregate.get("ast_exact_match"),
        name="reference.aggregate.ast_exact_match",
    )
    reference_samples = _require_mapping(
        reference.get("sample_scores"), name="reference.sample_scores"
    )
    if reference.get("candidate_id") != "candidate-v2":
        raise MobileRegressionError("regression reference must remain candidate-v2")
    if reference_metric != {"numerator": 602, "denominator": 756}:
        raise MobileRegressionError("candidate-v2 reference score changed")
    if reference_samples.get("rows") != 756:
        raise MobileRegressionError("candidate-v2 sample evidence row count changed")
    if not _is_sha256(reference_aggregate.get("sha256")) or not _is_sha256(
        reference_samples.get("sha256")
    ):
        raise MobileRegressionError("candidate-v2 evidence hashes are invalid")

    evaluation = _require_mapping(payload.get("evaluation"), name="evaluation")
    if evaluation.get("generation_version") != GENERATION_VERSION:
        raise MobileRegressionError("generation version differs from the frozen protocol")
    if evaluation.get("scorer_version") != MOBILE_ACTIONS_SCORER_VERSION:
        raise MobileRegressionError("Mobile scorer version differs from the frozen protocol")
    if evaluation.get("evaluator_version") != EVALUATOR_VERSION:
        raise MobileRegressionError("Action IR evaluator version differs from the frozen protocol")
    if evaluation.get("decoding") != "unconstrained_deterministic_greedy":
        raise MobileRegressionError("regression decoding policy changed")
    if evaluation.get("batch_size") != 128 or evaluation.get("max_new_tokens") != 192:
        raise MobileRegressionError("regression generation settings changed")
    if set(evaluation.get("checkpoint_files", ())) != set(REQUIRED_CHECKPOINT_FILES):
        raise MobileRegressionError("regression checkpoint file contract changed")

    gate = _require_mapping(payload.get("regression_gate"), name="regression_gate")
    maximum_drop = _require_mapping(
        gate.get("maximum_absolute_drop"), name="regression_gate.maximum_absolute_drop"
    )
    try:
        drop = Fraction(maximum_drop["numerator"], maximum_drop["denominator"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise MobileRegressionError("invalid preregistered maximum regression") from error
    if drop != Fraction(2, 100):
        raise MobileRegressionError("maximum absolute regression changed")
    required_denominator = gate.get("required_denominator")
    if required_denominator != reference_metric["denominator"]:
        raise MobileRegressionError("gate and reference denominators differ")
    exact_threshold = math.ceil(
        Fraction(reference_metric["numerator"], 1) - drop * required_denominator
    )
    if gate.get("minimum_candidate_numerator") != exact_threshold or exact_threshold != 587:
        raise MobileRegressionError("preregistered integer regression threshold is inconsistent")
    if gate.get("maximum_net_correct_rows_lost") != 15:
        raise MobileRegressionError("preregistered net-loss limit changed")
    if actual != MOBILE_REGRESSION_PREREGISTRATION_SHA256:  # pragma: no cover - explicit proof
        raise MobileRegressionError("preregistration identity changed")
    return payload


def load_checkpoint_hashes(
    path: str | Path,
    *,
    expected_files: Sequence[str] = REQUIRED_CHECKPOINT_FILES,
) -> dict[str, str]:
    """Load an explicit hash map or a checkpoint manifest containing ``file_sha256``."""

    payload: object = _strict_json(path)
    if isinstance(payload, Mapping) and "file_sha256" in payload:
        payload = payload["file_sha256"]
    hashes = _require_mapping(payload, name="checkpoint hashes")
    normalized = {str(name): str(digest) for name, digest in hashes.items()}
    if set(normalized) != set(expected_files):
        raise MobileRegressionError(
            f"checkpoint hash map must contain exactly {sorted(expected_files)!r}"
        )
    for name, digest in normalized.items():
        if not _is_sha256(digest):
            raise MobileRegressionError(f"invalid checkpoint SHA-256 for {name!r}")
    return normalized


def _validate_manifest(
    path: str | Path,
    preregistration: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    dataset = _require_mapping(preregistration.get("dataset"), name="dataset")
    contract = _require_mapping(dataset.get("manifest"), name="dataset.manifest")
    _require_exact_hash(path, contract.get("sha256"), name="Mobile development manifest")
    rows = read_jsonl(path)
    if len(rows) != contract.get("rows"):
        raise MobileRegressionError("Mobile development manifest row count changed")
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        sample_id = row.get("id")
        metadata = row.get("metadata")
        if row.get("schema_version") != EXAMPLE_SCHEMA_VERSION:
            raise MobileRegressionError(f"Mobile manifest schema changed at line {line_number}")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise MobileRegressionError(
                f"invalid or duplicate Mobile manifest ID at line {line_number}"
            )
        if not isinstance(metadata, Mapping):
            raise MobileRegressionError(f"Mobile manifest metadata missing at {sample_id}")
        if metadata.get("dataset") != dataset.get("dataset_id"):
            raise MobileRegressionError(f"dataset identity changed at {sample_id}")
        if metadata.get("source_revision") != dataset.get("revision"):
            raise MobileRegressionError(f"source revision changed at {sample_id}")
        if metadata.get("source_split") != contract.get("required_source_split"):
            raise MobileRegressionError(f"non-training source row entered dev at {sample_id}")
        if metadata.get("derived_split") != contract.get("required_derived_split"):
            raise MobileRegressionError(f"non-development row entered dev at {sample_id}")
        seen.add(sample_id)
    return rows


def _validate_audit(path: str | Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _require_mapping(preregistration.get("dataset"), name="dataset")
    contract = _require_mapping(dataset.get("audit"), name="dataset.audit")
    expected_firewall = _require_mapping(
        dataset.get("official_evaluation_firewall"),
        name="dataset.official_evaluation_firewall",
    )
    _require_exact_hash(path, contract.get("sha256"), name="Mobile adapter audit")
    audit = _strict_json(path)
    observed = _require_mapping(audit.get("official_eval"), name="audit.official_eval")
    for name, expected in expected_firewall.items():
        if name == "accepted_input_paths":
            continue
        if observed.get(name) != expected:
            raise MobileRegressionError(f"Mobile official-evaluation firewall differs at {name!r}")
    counts = _require_mapping(audit.get("counts"), name="audit.counts")
    derived = _require_mapping(counts.get("derived"), name="audit.counts.derived")
    source = _require_mapping(counts.get("source"), name="audit.counts.source")
    if derived.get("dev") != 756 or source.get("official_eval_opaque_unparsed") != 961:
        raise MobileRegressionError("Mobile audit population counts changed")
    artifacts = _require_mapping(audit.get("artifacts"), name="audit.artifacts")
    manifests = _require_mapping(artifacts.get("manifests"), name="audit.artifacts.manifests")
    dev = _require_mapping(manifests.get("dev"), name="audit.artifacts.manifests.dev")
    manifest = _require_mapping(dataset.get("manifest"), name="dataset.manifest")
    if dev.get("sha256") != manifest.get("sha256"):
        raise MobileRegressionError("Mobile audit does not identify the frozen dev manifest")
    return audit


def _validate_reference(
    *,
    aggregate_path: str | Path,
    samples_path: str | Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    reference = _require_mapping(preregistration.get("reference"), name="reference")
    aggregate_contract = _require_mapping(reference.get("aggregate"), name="reference.aggregate")
    samples_contract = _require_mapping(
        reference.get("sample_scores"), name="reference.sample_scores"
    )
    _require_exact_hash(
        aggregate_path,
        aggregate_contract.get("sha256"),
        name="candidate-v2 aggregate evidence",
    )
    _require_exact_hash(
        samples_path,
        samples_contract.get("sha256"),
        name="candidate-v2 sample evidence",
    )
    aggregate = _strict_json(aggregate_path)
    metric = _require_mapping(aggregate.get("ast_exact_match"), name="ast_exact_match")
    expected_metric = _require_mapping(
        aggregate_contract.get("ast_exact_match"),
        name="reference.aggregate.ast_exact_match",
    )
    if metric.get("numerator") != expected_metric.get("numerator") or metric.get(
        "denominator"
    ) != expected_metric.get("denominator"):
        raise MobileRegressionError("candidate-v2 aggregate metric changed")
    if aggregate.get("schema_version") != MOBILE_ACTIONS_SCORER_VERSION:
        raise MobileRegressionError("candidate-v2 scorer version changed")
    if aggregate.get("evaluator_version") != EVALUATOR_VERSION:
        raise MobileRegressionError("candidate-v2 evaluator version changed")

    rows = read_jsonl(samples_path)
    if len(rows) != samples_contract.get("rows"):
        raise MobileRegressionError("candidate-v2 sample evidence row count changed")
    manifest_ids = [row.get("id") for row in manifest_rows]
    sample_ids = [row.get("sample_id") for row in rows]
    if sample_ids != manifest_ids:
        raise MobileRegressionError(
            "candidate-v2 sample evidence is not aligned to the frozen manifest"
        )
    if any(type(row.get("ast_exact")) is not bool for row in rows):
        raise MobileRegressionError("candidate-v2 sample evidence lacks boolean ast_exact")
    if sum(bool(row["ast_exact"]) for row in rows) != expected_metric.get("numerator"):
        raise MobileRegressionError("candidate-v2 sample evidence disagrees with its aggregate")
    return aggregate, rows


def validate_runtime_environment(
    observed: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> None:
    """Require the candidate-v2 H200/bfloat16 reference decoding environment."""

    reference = _require_mapping(preregistration.get("reference"), name="reference")
    expected = _require_mapping(reference.get("environment"), name="reference.environment")
    for name in (
        "device",
        "dtype",
        "cuda_device_name",
        "cuda_capability",
        "torch",
        "cuda_runtime",
        "cublas_workspace_config",
        "sdpa_backends",
    ):
        if observed.get(name) != expected.get(name):
            raise MobileRegressionError(
                f"regression runtime differs from candidate-v2 at {name}: "
                f"expected {expected.get(name)!r}, got {observed.get(name)!r}"
            )


def compare_paired_samples(
    *,
    candidate_aggregate: Mapping[str, Any],
    candidate_samples: Sequence[Mapping[str, Any]],
    reference_samples: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the frozen integer gate and return paired sample transitions."""

    if len(candidate_samples) != len(reference_samples):
        raise MobileRegressionError("candidate and reference sample counts differ")
    candidate_metric = _require_mapping(
        candidate_aggregate.get("ast_exact_match"), name="candidate.ast_exact_match"
    )
    gate_contract = _require_mapping(preregistration.get("regression_gate"), name="regression_gate")
    reference = _require_mapping(preregistration.get("reference"), name="reference")
    reference_aggregate = _require_mapping(reference.get("aggregate"), name="reference.aggregate")
    reference_metric = _require_mapping(
        reference_aggregate.get("ast_exact_match"),
        name="reference.aggregate.ast_exact_match",
    )
    denominator = gate_contract.get("required_denominator")
    numerator = candidate_metric.get("numerator")
    if (
        type(numerator) is not int
        or candidate_metric.get("denominator") != denominator
        or len(candidate_samples) != denominator
    ):
        raise MobileRegressionError("candidate aggregate has the wrong regression denominator")
    if candidate_aggregate.get("schema_version") != MOBILE_ACTIONS_SCORER_VERSION:
        raise MobileRegressionError("candidate scorer version differs from the preregistration")
    if candidate_aggregate.get("evaluator_version") != EVALUATOR_VERSION:
        raise MobileRegressionError("candidate evaluator version differs from the preregistration")

    paired: list[dict[str, Any]] = []
    transition_counts = {
        "retained_correct": 0,
        "fixed": 0,
        "regressed": 0,
        "retained_incorrect": 0,
    }
    for index, (candidate, incumbent) in enumerate(
        zip(candidate_samples, reference_samples, strict=True), start=1
    ):
        sample_id = candidate.get("sample_id")
        if not isinstance(sample_id, str) or sample_id != incumbent.get("sample_id"):
            raise MobileRegressionError(f"paired sample ID mismatch at row {index}")
        candidate_exact = candidate.get("ast_exact")
        reference_exact = incumbent.get("ast_exact")
        if type(candidate_exact) is not bool or type(reference_exact) is not bool:
            raise MobileRegressionError(f"non-boolean paired outcome at {sample_id}")
        if candidate.get("scenario") != incumbent.get("scenario"):
            raise MobileRegressionError(f"paired scenario mismatch at {sample_id}")
        if candidate.get("gold") != incumbent.get("gold"):
            raise MobileRegressionError(f"paired gold AST mismatch at {sample_id}")
        if candidate_exact and reference_exact:
            transition = "retained_correct"
        elif candidate_exact:
            transition = "fixed"
        elif reference_exact:
            transition = "regressed"
        else:
            transition = "retained_incorrect"
        transition_counts[transition] += 1
        paired.append(
            {
                "schema_version": MOBILE_REGRESSION_PAIRED_VERSION,
                "sample_id": sample_id,
                "scenario": candidate.get("scenario"),
                "reference_ast_exact": reference_exact,
                "candidate_ast_exact": candidate_exact,
                "transition": transition,
            }
        )

    observed_numerator = sum(bool(row.get("ast_exact")) for row in candidate_samples)
    if observed_numerator != numerator:
        raise MobileRegressionError("candidate sample evidence disagrees with its aggregate")
    reference_numerator = reference_metric.get("numerator")
    if type(reference_numerator) is not int:
        raise MobileRegressionError("reference numerator is invalid")
    net_change = numerator - reference_numerator
    if transition_counts["fixed"] - transition_counts["regressed"] != net_change:
        raise MobileRegressionError("paired transition counts disagree with aggregate delta")
    minimum = gate_contract.get("minimum_candidate_numerator")
    if type(minimum) is not int:
        raise MobileRegressionError("preregistered minimum numerator is invalid")
    passed = numerator >= minimum
    return (
        {
            "metric": "ast_exact_match",
            "passed": passed,
            "status": "pass" if passed else "fail",
            "reference": {
                "candidate_id": reference.get("candidate_id"),
                "numerator": reference_numerator,
                "denominator": denominator,
                "value": reference_numerator / denominator,
            },
            "candidate": {
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator,
            },
            "delta": {
                "correct_rows": net_change,
                "absolute_accuracy": net_change / denominator,
            },
            "threshold": {
                "minimum_candidate_numerator": minimum,
                "denominator": denominator,
                "minimum_value": minimum / denominator,
                "maximum_absolute_drop": 0.02,
                "maximum_net_correct_rows_lost": gate_contract.get("maximum_net_correct_rows_lost"),
            },
            "paired_transition_counts": transition_counts,
            "decision": (
                "mobile_regression_gate_passed"
                if passed
                else "mobile_regression_gate_failed_reject_broader_checkpoint"
            ),
        },
        paired,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise MobileRegressionError(f"refusing to overwrite artifact: {path}")
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
        raise MobileRegressionError(f"refusing to overwrite artifact: {path}")
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


def _resolve_artifact(
    *,
    explicit: str | Path | None,
    repository_root: Path,
    contract: Mapping[str, Any],
    name: str,
) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    relative = contract.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise MobileRegressionError(f"{name} lacks a preregistered relative path")
    return (repository_root / relative).resolve()


def _artifact_manifest(root: Path) -> dict[str, Any]:
    hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact-sha256.json"
    }
    return {
        "schema_version": "barun-mobile-regression-artifacts-v1",
        "file_sha256": hashes,
    }


def run_mobile_regression(
    *,
    checkpoint_dir: str | Path,
    checkpoint_hashes_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
    preregistration_path: str | Path,
    runtime_environment: Mapping[str, Any],
    manifest_path: str | Path | None = None,
    audit_path: str | Path | None = None,
    reference_aggregate_path: str | Path | None = None,
    reference_samples_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate, strictly score, and gate one exact compatible checkpoint.

    The development/audit/reference artifacts and checkpoint hashes are all verified before
    model tensors are loaded.  Every output directory is immutable.
    """

    root = Path(repository_root).resolve()
    prereg_path = Path(preregistration_path).resolve()
    preregistration = load_preregistration(prereg_path)
    validate_runtime_environment(runtime_environment, preregistration)

    dataset = _require_mapping(preregistration.get("dataset"), name="dataset")
    manifest_contract = _require_mapping(dataset.get("manifest"), name="dataset.manifest")
    audit_contract = _require_mapping(dataset.get("audit"), name="dataset.audit")
    reference = _require_mapping(preregistration.get("reference"), name="reference")
    reference_aggregate_contract = _require_mapping(
        reference.get("aggregate"), name="reference.aggregate"
    )
    reference_samples_contract = _require_mapping(
        reference.get("sample_scores"), name="reference.sample_scores"
    )
    manifest = _resolve_artifact(
        explicit=manifest_path,
        repository_root=root,
        contract=manifest_contract,
        name="Mobile development manifest",
    )
    audit = _resolve_artifact(
        explicit=audit_path,
        repository_root=root,
        contract=audit_contract,
        name="Mobile adapter audit",
    )
    reference_aggregate = _resolve_artifact(
        explicit=reference_aggregate_path,
        repository_root=root,
        contract=reference_aggregate_contract,
        name="candidate-v2 aggregate evidence",
    )
    reference_samples = _resolve_artifact(
        explicit=reference_samples_path,
        repository_root=root,
        contract=reference_samples_contract,
        name="candidate-v2 sample evidence",
    )

    # Data and incumbent evidence are frozen and checked before checkpoint model bytes are read.
    manifest_rows = _validate_manifest(manifest, preregistration)
    _validate_audit(audit, preregistration)
    _, incumbent_samples = _validate_reference(
        aggregate_path=reference_aggregate,
        samples_path=reference_samples,
        manifest_rows=manifest_rows,
        preregistration=preregistration,
    )

    evaluation = _require_mapping(preregistration.get("evaluation"), name="evaluation")
    checkpoint_hashes = load_checkpoint_hashes(
        checkpoint_hashes_path,
        expected_files=tuple(str(name) for name in evaluation["checkpoint_files"]),
    )
    incumbent_hashes = _require_mapping(
        reference.get("checkpoint_file_sha256"),
        name="reference.checkpoint_file_sha256",
    )
    for compatibility_file in ("barun_config.json", "tokenizer.json"):
        if checkpoint_hashes[compatibility_file] != incumbent_hashes.get(compatibility_file):
            raise MobileRegressionError(
                f"checkpoint {compatibility_file} differs from the frozen prompt/model contract"
            )
    verified_hashes = verify_checkpoint(
        checkpoint_dir,
        expected_sha256=checkpoint_hashes,
    )

    output = Path(output_dir).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise MobileRegressionError(f"refusing to overwrite regression output {output}") from error
    shutil.copy2(prereg_path, output / "preregistration.json")
    _write_json(output / "runtime-environment.json", dict(runtime_environment))

    predictions_path = output / "predictions.jsonl"
    generation: GenerationSummary = generate_manifest(
        checkpoint_dir=checkpoint_dir,
        manifest_path=manifest,
        manifest_sha256=str(manifest_contract["sha256"]),
        predictions_path=predictions_path,
        device_name="cuda",
        batch_size=int(evaluation["batch_size"]),
        max_new_tokens=int(evaluation["max_new_tokens"]),
        expected_checkpoint_sha256=checkpoint_hashes,
    )
    score_paths = write_scores(manifest, predictions_path, output / "scores")
    candidate_aggregate = _strict_json(score_paths["aggregate"])
    candidate_samples = read_jsonl(score_paths["samples"])
    gate, paired = compare_paired_samples(
        candidate_aggregate=candidate_aggregate,
        candidate_samples=candidate_samples,
        reference_samples=incumbent_samples,
        preregistration=preregistration,
    )
    paired_path = output / "paired-samples.jsonl"
    _write_jsonl(paired_path, paired)

    result = {
        "schema_version": MOBILE_REGRESSION_RESULT_VERSION,
        "preregistration": {
            "path": str(output / "preregistration.json"),
            "sha256": MOBILE_REGRESSION_PREREGISTRATION_SHA256,
        },
        "checkpoint": {
            "directory": str(Path(checkpoint_dir).resolve()),
            "file_sha256": verified_hashes,
        },
        "population": {
            "dataset_id": dataset.get("dataset_id"),
            "revision": dataset.get("revision"),
            "manifest": str(manifest),
            "manifest_sha256": manifest_contract.get("sha256"),
            "rows": len(manifest_rows),
            "audit": str(audit),
            "audit_sha256": audit_contract.get("sha256"),
            "official_evaluation_rows_opaque_unparsed": MOBILE_ACTIONS_FINAL_EVAL_ROWS,
            "official_evaluation_artifacts_accessed": [],
        },
        "generation": generation.to_dict(),
        "scores": {
            "aggregate": str(score_paths["aggregate"]),
            "aggregate_sha256": sha256_file(score_paths["aggregate"]),
            "samples": str(score_paths["samples"]),
            "samples_sha256": sha256_file(score_paths["samples"]),
            "paired_samples": str(paired_path),
            "paired_samples_sha256": sha256_file(paired_path),
        },
        "gate": gate,
        "interpretation": {
            "scope": "development_regression_gate_only",
            "new_mobile_selection_trial": False,
            "official_mobile_evaluation_result": False,
            "larger_model_superiority_claim": False,
        },
    }
    _write_json(output / "result.json", result)
    _write_json(output / "artifact-sha256.json", _artifact_manifest(output))
    return result


__all__ = [
    "MOBILE_REGRESSION_PAIRED_VERSION",
    "MOBILE_REGRESSION_PREREGISTRATION_SHA256",
    "MOBILE_REGRESSION_PREREGISTRATION_VERSION",
    "MOBILE_REGRESSION_RESULT_VERSION",
    "MobileRegressionError",
    "compare_paired_samples",
    "load_checkpoint_hashes",
    "load_preregistration",
    "run_mobile_regression",
    "validate_runtime_environment",
]
