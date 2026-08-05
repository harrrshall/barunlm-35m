"""Offline, fail-closed selection across the four frozen rescue trials.

The selector deliberately has no model, CUDA, network, JarvisLabs, Hugging Face, or
Weights & Biases dependency.  It accepts only downloaded immutable evidence bundles
and the hash-pinned shared selection protocol.  Every metric used for eligibility is
recomputed from sample-level JSONL evidence before exact-rational ranking.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, NoReturn

PARALLEL_RESCUE_SELECTION_SCHEMA_VERSION = "barun-parallel-rescue-selection-v1"
PARALLEL_RESCUE_SELECTION_RECEIPT_VERSION = "barun-parallel-rescue-selection-receipt-v1"
PARALLEL_RESCUE_SELECTION_SHA256 = (
    "68bb6132a486dfbd6c9ad478db9c87ae9f2350bfc1d302193169fd7f5c290130"
)
PARALLEL_RESCUE_SELECTION_RELATIVE_PATH = "configs/parallel_rescue_selection_v1.json"
CONTINUAL_PROTOCOL_SHA256 = "b45f6efb28af44e0a2772b9239aff3f321a38765b964f1925e5bfc47c24fbf0d"
INTERPOLATION_PROTOCOL_SHA256 = "1e2f005f61bf8f8cb62c01928c23266023e2e20760321832f80bc4dbe3c9ee3c"
MOBILE_REGRESSION_PROTOCOL_SHA256 = (
    "96afa0ac407c53fd375ed0910a08ec66463718e3b826cf02b73f8bbc87a3c47e"
)
PRESTO_ENDPOINT_MANIFEST_SHA256 = "90e50f316456947bbee10b710b6f828f1ca8955ed0e43998b15adb0e09f46e23"

CONTINUAL_RESULT_SCHEMA = "barun-continual-recovery-result-v1"
INTERPOLATION_RESULT_SCHEMA = "barun-interpolation-rescue-result-v1"
MOBILE_RESULT_SCHEMA = "barun-mobile-regression-result-v1"
MOBILE_SCORE_SCHEMA = "barun-mobile-actions-score-v1"
PRESTO_SCORE_SCHEMA = "barun-presto-action-ir-score-v2"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RUN_ID_PATTERN = re.compile(r"\d{8}-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*-s\d+")
_TRIAL_ORDER = ("I25", "I50", "I75", "C")
_ARM_TO_TRIAL = {
    "alpha-025": ("I25", Fraction(1, 4)),
    "alpha-050": ("I50", Fraction(1, 2)),
    "alpha-075": ("I75", Fraction(3, 4)),
}
_REVISION_TAGS = frozenset(
    {"cancel-action", "correct-action", "correct-argument", "within-turn-correction"}
)
_PRESTO_FIREWALL = {
    "archive_payload_hashing_only": True,
    "bytes_read_from_sensitive_members": 0,
    "combined_dataset_member_opened": False,
    "labels_parsed": False,
    "materialized": False,
    "member_opened": False,
    "opaque_unparsed": True,
    "prompts_parsed": False,
    "sensitive_members_opened": [],
    "targets_parsed": False,
    "test_partition_members_opened": False,
    "official_test_rows_opaque_unparsed": 194_118,
}


class ParallelRescueSelectionError(RuntimeError):
    """Evidence is missing, mutable, inconsistent, or outside the frozen protocol."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ParallelRescueSelectionError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> NoReturn:
    raise ParallelRescueSelectionError(f"non-finite JSON constant {value!r}")


def _loads_json(text: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_pairs,
            parse_float=Decimal,
            parse_constant=_reject_constant,
        )
    except ParallelRescueSelectionError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ParallelRescueSelectionError(f"invalid strict JSON in {label}") from error
    if not isinstance(payload, dict):
        raise ParallelRescueSelectionError(f"{label} must contain one JSON object")
    return payload


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _loads_json(path.read_text(encoding="utf-8"), label)
    except ParallelRescueSelectionError:
        raise
    except (OSError, UnicodeError) as error:
        raise ParallelRescueSelectionError(f"cannot read {label}: {path}") from error


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ParallelRescueSelectionError(f"missing or invalid required object: {label}")
    return value


def _field(payload: Mapping[str, Any], path: Sequence[str], label: str | None = None) -> Any:
    current: object = payload
    traversed: list[str] = []
    for key in path:
        traversed.append(key)
        if not isinstance(current, Mapping) or key not in current:
            name = label or ".".join(traversed)
            raise ParallelRescueSelectionError(f"missing required field: {name}")
        current = current[key]
    return current


def _exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    payload = _mapping(value, label)
    if set(payload) != expected:
        missing = sorted(expected.difference(payload))
        extra = sorted(set(payload).difference(expected))
        raise ParallelRescueSelectionError(
            f"{label} fields changed; missing={missing!r}, extra={extra!r}"
        )
    return payload


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParallelRescueSelectionError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ParallelRescueSelectionError(f"{label} must be at least {minimum}")
    return value


def _boolean(value: object, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise ParallelRescueSelectionError(f"{label} must be {expected}")
    return expected


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ParallelRescueSelectionError(f"{label} must be a nonempty string")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ParallelRescueSelectionError(f"{label} must be a lowercase SHA-256")
    return value


def _run_id(value: object, label: str) -> str:
    run_id = _string(value, label)
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ParallelRescueSelectionError(f"{label} is not an immutable run ID")
    return run_id


def _fraction(value: object, label: str) -> Fraction:
    if isinstance(value, bool):
        raise ParallelRescueSelectionError(f"{label} must be numeric")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Decimal) and value.is_finite():
        return Fraction(value)
    raise ParallelRescueSelectionError(f"{label} must be a finite JSON number")


def _fraction_record(value: Fraction) -> dict[str, int | str]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact": f"{value.numerator}/{value.denominator}",
    }


def load_selection_protocol(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load only the exact frozen cross-method selector supplied out of band."""

    expected = _sha256(expected_sha256, "outer-attempt SHA-256")
    if expected != PARALLEL_RESCUE_SELECTION_SHA256:
        raise ParallelRescueSelectionError(
            "outer-attempt hash differs from the compiled frozen shared selector"
        )
    source = Path(path)
    actual = sha256_file(source)
    if actual != expected:
        raise ParallelRescueSelectionError(
            f"outer-attempt SHA-256 mismatch: expected {expected}, got {actual}"
        )
    protocol = _load_json(source, "outer shared-selector attempt")
    if protocol.get("schema_version") != PARALLEL_RESCUE_SELECTION_SCHEMA_VERSION:
        raise ParallelRescueSelectionError("unsupported shared selector schema")
    if protocol.get("status") != "frozen_before_any_rescue_score":
        raise ParallelRescueSelectionError("shared selector was not frozen before rescue scores")
    methods = _mapping(protocol.get("method_protocols"), "method_protocols")
    continual = _mapping(methods.get("continual_recovery"), "method_protocols.continual_recovery")
    interpolation = _mapping(
        methods.get("checkpoint_interpolation"),
        "method_protocols.checkpoint_interpolation",
    )
    if continual.get("sha256") != CONTINUAL_PROTOCOL_SHA256 or continual.get("trial_ids") != ["C"]:
        raise ParallelRescueSelectionError("continual method identity changed")
    if interpolation.get("sha256") != INTERPOLATION_PROTOCOL_SHA256 or interpolation.get(
        "trial_ids"
    ) != ["I25", "I50", "I75"]:
        raise ParallelRescueSelectionError("interpolation method identity changed")
    budget = _mapping(protocol.get("trial_budget"), "trial_budget")
    for field, expected_value in {
        "total_joint_development_trials": 4,
        "continual_final_checkpoints": 1,
        "interpolation_alphas": 3,
        "adaptive_followup_alphas": 0,
        "recipe_retries_after_observing_scores": 0,
        "outer_selection_adds_evaluation_trials": 0,
    }.items():
        if budget.get(field) != expected_value:
            raise ParallelRescueSelectionError(f"frozen trial budget changed at {field}")
    ranking = _field(protocol, ("selection", "ranking"), "selection.ranking")
    if ranking != [
        "maximum min(m, p)",
        "maximum m + p",
        "fixed order I25, I50, I75, C",
    ]:
        raise ParallelRescueSelectionError("frozen cross-method ranking changed")
    if _field(protocol, ("selection", "interpolation_internal_winner")) != (
        "diagnostic only; the outer four-trial rule controls final promotion"
    ):
        raise ParallelRescueSelectionError("outer selector no longer controls promotion")
    return protocol


def _regular_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            path = current_path / name
            if path.is_symlink():
                raise ParallelRescueSelectionError(
                    f"bundle contains a symlink directory: {path.relative_to(root).as_posix()}"
                )
        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if any(character in relative for character in ("\n", "\r", "\0")):
                raise ParallelRescueSelectionError("bundle contains a path unsafe for hashing")
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ParallelRescueSelectionError(
                    f"bundle contains a non-regular file: {relative}"
                )
            files[relative] = path
    return files


def _hash_map(value: object, label: str) -> dict[str, str]:
    payload = _mapping(value, label)
    output: dict[str, str] = {}
    for name, digest in payload.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ParallelRescueSelectionError(f"{label} contains an unsafe relative path")
        if Path(name).as_posix() != name:
            raise ParallelRescueSelectionError(f"{label} path is not canonical POSIX: {name!r}")
        output[name] = _sha256(digest, f"{label}[{name!r}]")
    return output


def _verify_tree_manifest(
    root: str | Path,
    *,
    expected_schema: str,
    expected_manifest_paths: set[str],
) -> dict[str, Any]:
    source = Path(root)
    if source.is_symlink():
        raise ParallelRescueSelectionError("essential bundle root cannot be a symlink")
    try:
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise ParallelRescueSelectionError(
            f"essential bundle root does not exist: {source}"
        ) from error
    if not resolved.is_dir():
        raise ParallelRescueSelectionError(f"essential bundle root is not a directory: {resolved}")
    files = _regular_files(resolved)
    manifest_paths = {name for name in files if Path(name).name == "artifact-sha256.json"}
    if manifest_paths != expected_manifest_paths:
        raise ParallelRescueSelectionError(
            "artifact manifest locations changed: "
            f"expected {sorted(expected_manifest_paths)!r}, got {sorted(manifest_paths)!r}"
        )
    root_manifest_path = files["artifact-sha256.json"]
    manifest = _exact_keys(
        _load_json(root_manifest_path, "essential artifact manifest"),
        {"schema_version", "file_sha256"},
        "essential artifact manifest",
    )
    if manifest["schema_version"] != expected_schema:
        raise ParallelRescueSelectionError("essential artifact manifest schema changed")
    recorded = _hash_map(manifest["file_sha256"], "essential artifact hashes")
    observed = {
        name: sha256_file(path)
        for name, path in sorted(files.items())
        if Path(name).name != "artifact-sha256.json"
    }
    if recorded != observed:
        missing = sorted(set(observed).difference(recorded))
        extra = sorted(set(recorded).difference(observed))
        changed = sorted(
            name
            for name in set(observed).intersection(recorded)
            if observed[name] != recorded[name]
        )
        raise ParallelRescueSelectionError(
            "essential artifact manifest mismatch; "
            f"unrecorded={missing[:5]!r}, missing_files={extra[:5]!r}, changed={changed[:5]!r}"
        )
    return {
        "root": resolved,
        "files": files,
        "manifest_sha256": sha256_file(root_manifest_path),
        "manifest_file_count": len(recorded),
    }


def _verify_nested_mobile_manifest(root: Path, relative_root: str) -> None:
    directory = root / relative_root
    manifest_path = directory / "artifact-sha256.json"
    manifest = _exact_keys(
        _load_json(manifest_path, f"{relative_root} artifact manifest"),
        {"schema_version", "file_sha256"},
        f"{relative_root} artifact manifest",
    )
    if manifest["schema_version"] != "barun-mobile-regression-artifacts-v1":
        raise ParallelRescueSelectionError(f"{relative_root} artifact schema changed")
    recorded = _hash_map(manifest["file_sha256"], f"{relative_root} hashes")
    files = _regular_files(directory)
    observed = {
        name: sha256_file(path)
        for name, path in sorted(files.items())
        if Path(name).name != "artifact-sha256.json"
    }
    if recorded != observed:
        raise ParallelRescueSelectionError(f"{relative_root} artifact manifest mismatch")


def _require_file_hash(path: Path, expected: object, label: str) -> str:
    digest = _sha256(expected, f"{label} recorded SHA-256")
    actual = sha256_file(path)
    if actual != digest:
        raise ParallelRescueSelectionError(
            f"{label} SHA-256 mismatch: expected {digest}, got {actual}"
        )
    return actual


def _jsonl_rows(path: Path, expected_rows: int, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ParallelRescueSelectionError(
                        f"blank JSONL row in {label} at line {line_number}"
                    )
                row = _loads_json(line, f"{label}:{line_number}")
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                    raise ParallelRescueSelectionError(
                        f"missing or duplicate sample_id in {label} at line {line_number}"
                    )
                seen.add(sample_id)
                rows.append(row)
    except ParallelRescueSelectionError:
        raise
    except (OSError, UnicodeError) as error:
        raise ParallelRescueSelectionError(f"cannot read sample evidence: {path}") from error
    if len(rows) != expected_rows:
        raise ParallelRescueSelectionError(
            f"{label} row count changed: expected {expected_rows}, got {len(rows)}"
        )
    return rows


def _verify_rate(
    payload: object,
    *,
    numerator: int,
    denominator: int,
    label: str,
) -> None:
    metric = _mapping(payload, label)
    if metric.get("numerator") != numerator or metric.get("denominator") != denominator:
        raise ParallelRescueSelectionError(
            f"{label} disagrees with sample evidence: expected {numerator}/{denominator}"
        )
    value = _fraction(metric.get("value"), f"{label}.value")
    expected = Fraction(numerator, denominator) if denominator else Fraction(0)
    if abs(value - expected) > Fraction(1, 10**15):
        raise ParallelRescueSelectionError(f"{label}.value disagrees with its integer counts")


def _mobile_sample_metrics(path: Path) -> dict[str, int]:
    rows = _jsonl_rows(path, 756, "Mobile sample scores")
    exact = catastrophic = false_action = false_action_denominator = 0
    for index, row in enumerate(rows, start=1):
        if row.get("evaluator_version") != "action-ir-v1.0.0":
            raise ParallelRescueSelectionError(f"Mobile sample evaluator changed at row {index}")
        for field in ("ast_exact", "catastrophic_unauthorized_action", "false_action"):
            if type(row.get(field)) is not bool:
                raise ParallelRescueSelectionError(f"Mobile sample {index} lacks boolean {field}")
        if type(row.get("false_action_eligible")) is not bool:
            raise ParallelRescueSelectionError(
                f"Mobile sample {index} lacks boolean false_action_eligible"
            )
        exact += int(row["ast_exact"])
        catastrophic += int(row["catastrophic_unauthorized_action"])
        false_action += int(row["false_action"])
        false_action_denominator += int(row["false_action_eligible"])
    return {
        "sample_count": len(rows),
        "ast_exact_numerator": exact,
        "catastrophic_unauthorized_actions": catastrophic,
        "false_action_numerator": false_action,
        "false_action_denominator": false_action_denominator,
    }


def _presto_sample_metrics(path: Path) -> dict[str, Any]:
    rows = _jsonl_rows(path, 14_288, "PRESTO sample scores")
    exact = schema = false_call = gate_rows = 0
    abstain_tp = abstain_fp = abstain_fn = 0
    bucket_counts: Counter[str] = Counter()
    bucket_exact: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        if row.get("schema_version") != PRESTO_SCORE_SCHEMA:
            raise ParallelRescueSelectionError(f"PRESTO sample scorer changed at row {index}")
        for field in ("ast_exact", "schema_valid", "false_call_on_gate"):
            if type(row.get(field)) is not bool:
                raise ParallelRescueSelectionError(f"PRESTO sample {index} lacks boolean {field}")
        gold = row.get("gold_decision")
        predicted = row.get("predicted_decision")
        if not isinstance(gold, str) or (predicted is not None and not isinstance(predicted, str)):
            raise ParallelRescueSelectionError(f"PRESTO sample {index} has invalid decisions")
        phenomenon = row.get("phenomenon")
        group = row.get("phenomenon_group")
        if not isinstance(phenomenon, str) or not isinstance(group, str):
            raise ParallelRescueSelectionError(f"PRESTO sample {index} lacks phenomenon evidence")
        exact += int(row["ast_exact"])
        schema += int(row["schema_valid"])
        false_call += int(row["false_call_on_gate"])
        gate_rows += int(gold in {"ABSTAIN", "CONFIRM"})
        abstain_tp += int(gold == "ABSTAIN" and predicted == "ABSTAIN")
        abstain_fp += int(gold != "ABSTAIN" and predicted == "ABSTAIN")
        abstain_fn += int(gold == "ABSTAIN" and predicted != "ABSTAIN")
        bucket_counts[group] += 1
        bucket_exact[group] += int(row["ast_exact"])
        raw_counts[phenomenon] += 1
    f1_denominator = 2 * abstain_tp + abstain_fp + abstain_fn
    f1 = Fraction(2 * abstain_tp, f1_denominator) if f1_denominator else Fraction(0)
    return {
        "sample_count": len(rows),
        "ast_exact_numerator": exact,
        "schema_valid_numerator": schema,
        "false_call_numerator": false_call,
        "false_call_denominator": gate_rows,
        "abstention": {
            "true_positive": abstain_tp,
            "false_positive": abstain_fp,
            "false_negative": abstain_fn,
            "f1": f1,
        },
        "bucket_counts": dict(bucket_counts),
        "bucket_exact": dict(bucket_exact),
        "raw_counts": dict(raw_counts),
    }


def _verify_mobile_evidence(
    *,
    bundle_root: Path,
    relative_root: str,
    result: Mapping[str, Any],
    checkpoint_hashes: Mapping[str, str],
) -> dict[str, int]:
    if result.get("schema_version") != MOBILE_RESULT_SCHEMA:
        raise ParallelRescueSelectionError(f"{relative_root} Mobile result schema changed")
    _verify_nested_mobile_manifest(bundle_root, relative_root)
    mobile_root = bundle_root / relative_root
    aggregate_path = mobile_root / "scores" / "aggregate.json"
    samples_path = mobile_root / "scores" / "sample_scores.jsonl"
    result_path = mobile_root / "result.json"
    if _load_json(result_path, f"{relative_root} result") != result:
        raise ParallelRescueSelectionError(
            f"embedded and file-backed Mobile results differ at {relative_root}"
        )
    scores = _mapping(result.get("scores"), f"{relative_root}.scores")
    _require_file_hash(
        aggregate_path,
        _field(scores, ("aggregate_sha256",)),
        f"{relative_root} Mobile aggregate",
    )
    _require_file_hash(
        samples_path,
        _field(scores, ("samples_sha256",)),
        f"{relative_root} Mobile samples",
    )
    preregistration = _mapping(result.get("preregistration"), f"{relative_root}.preregistration")
    _require_file_hash(
        mobile_root / "preregistration.json",
        preregistration.get("sha256"),
        f"{relative_root} Mobile preregistration",
    )
    if preregistration.get("sha256") != MOBILE_REGRESSION_PROTOCOL_SHA256:
        raise ParallelRescueSelectionError("Mobile regression protocol hash changed")
    if _mapping(result.get("checkpoint"), f"{relative_root}.checkpoint").get("file_sha256") != dict(
        checkpoint_hashes
    ):
        raise ParallelRescueSelectionError(f"{relative_root} evaluated a different checkpoint")
    population = _mapping(result.get("population"), f"{relative_root}.population")
    if population.get("rows") != 756:
        raise ParallelRescueSelectionError(f"{relative_root} Mobile population changed")
    if population.get("official_evaluation_rows_opaque_unparsed") != 961:
        raise ParallelRescueSelectionError(f"{relative_root} Mobile official row count changed")
    if population.get("official_evaluation_artifacts_accessed") != []:
        raise ParallelRescueSelectionError(f"{relative_root} accessed Mobile official evidence")
    interpretation = _mapping(result.get("interpretation"), f"{relative_root}.interpretation")
    _boolean(
        interpretation.get("official_mobile_evaluation_result"),
        False,
        f"{relative_root} official-Mobile interpretation",
    )
    _verify_generation_population(
        result.get("generation"), 756, f"{relative_root} Mobile generation"
    )
    metrics = _mobile_sample_metrics(samples_path)
    aggregate = _load_json(aggregate_path, f"{relative_root} Mobile aggregate")
    if (
        aggregate.get("schema_version") != MOBILE_SCORE_SCHEMA
        or aggregate.get("sample_count") != 756
    ):
        raise ParallelRescueSelectionError(f"{relative_root} Mobile aggregate population changed")
    _verify_rate(
        aggregate.get("ast_exact_match"),
        numerator=metrics["ast_exact_numerator"],
        denominator=756,
        label=f"{relative_root} Mobile AST exact",
    )
    _verify_rate(
        aggregate.get("false_action"),
        numerator=metrics["false_action_numerator"],
        denominator=metrics["false_action_denominator"],
        label=f"{relative_root} Mobile false action",
    )
    if (
        aggregate.get("catastrophic_unauthorized_actions")
        != metrics["catastrophic_unauthorized_actions"]
    ):
        raise ParallelRescueSelectionError(
            f"{relative_root} Mobile catastrophic count disagrees with samples"
        )
    candidate = _mapping(
        _field(result, ("gate", "candidate"), f"{relative_root}.gate.candidate"),
        f"{relative_root}.gate.candidate",
    )
    if (
        candidate.get("numerator") != metrics["ast_exact_numerator"]
        or candidate.get("denominator") != 756
    ):
        raise ParallelRescueSelectionError(f"{relative_root} Mobile gate disagrees with samples")
    expected_pass = metrics["ast_exact_numerator"] >= 587
    if _field(result, ("gate", "passed")) is not expected_pass:
        raise ParallelRescueSelectionError(f"{relative_root} Mobile gate boolean is inconsistent")
    return metrics


def _verify_presto_evidence(
    *,
    aggregate_path: Path,
    samples_path: Path,
    aggregate: Mapping[str, Any],
    expected_samples_sha256: object,
) -> dict[str, Any]:
    if _load_json(aggregate_path, "PRESTO aggregate file") != aggregate:
        raise ParallelRescueSelectionError("embedded and file-backed PRESTO aggregates differ")
    _require_file_hash(samples_path, expected_samples_sha256, "PRESTO sample evidence")
    metrics = _presto_sample_metrics(samples_path)
    if aggregate.get("schema_version") != PRESTO_SCORE_SCHEMA:
        raise ParallelRescueSelectionError("PRESTO scorer schema changed")
    if aggregate.get("metric_scope") != "derived_action_ir_not_native_presto_semantic_parse":
        raise ParallelRescueSelectionError("PRESTO metric scope changed")
    if aggregate.get("sample_count") != 14_288:
        raise ParallelRescueSelectionError("PRESTO aggregate sample count changed")
    _verify_rate(
        aggregate.get("ast_exact_match"),
        numerator=metrics["ast_exact_numerator"],
        denominator=14_288,
        label="PRESTO AST exact",
    )
    _verify_rate(
        aggregate.get("schema_valid"),
        numerator=metrics["schema_valid_numerator"],
        denominator=14_288,
        label="PRESTO schema valid",
    )
    _verify_rate(
        aggregate.get("false_call_on_gate"),
        numerator=metrics["false_call_numerator"],
        denominator=metrics["false_call_denominator"],
        label="PRESTO false call",
    )
    abstention = _mapping(aggregate.get("abstention"), "PRESTO abstention")
    for field in ("true_positive", "false_positive", "false_negative"):
        if abstention.get(field) != metrics["abstention"][field]:
            raise ParallelRescueSelectionError(
                f"PRESTO abstention {field} disagrees with sample evidence"
            )
    if abs(
        _fraction(abstention.get("f1"), "PRESTO abstention.f1") - metrics["abstention"]["f1"]
    ) > Fraction(1, 10**15):
        raise ParallelRescueSelectionError("PRESTO abstention F1 disagrees with integer counts")
    buckets = _mapping(aggregate.get("headline_phenomenon_buckets"), "PRESTO buckets")
    for name in ("no_phenomenon", "revision", "disfluency"):
        bucket = _mapping(buckets.get(name), f"PRESTO bucket {name}")
        count = metrics["bucket_counts"].get(name, 0)
        if bucket.get("count") != count:
            raise ParallelRescueSelectionError(f"PRESTO {name} bucket count disagrees with samples")
        _verify_rate(
            bucket.get("ast_exact_match"),
            numerator=metrics["bucket_exact"].get(name, 0),
            denominator=count,
            label=f"PRESTO {name} AST exact",
        )
    raw = _mapping(aggregate.get("per_phenomenon"), "PRESTO raw phenomena")
    for name in _REVISION_TAGS:
        bucket = _mapping(raw.get(name), f"PRESTO raw tag {name}")
        if bucket.get("count") != metrics["raw_counts"].get(name, 0):
            raise ParallelRescueSelectionError(f"PRESTO raw tag {name} disagrees with samples")
    return metrics


def _verify_presto_firewall(value: object, label: str) -> None:
    firewall = _mapping(value, label)
    for field, expected in _PRESTO_FIREWALL.items():
        if field not in firewall:
            raise ParallelRescueSelectionError(f"missing required field: {label}.{field}")
        if firewall[field] != expected:
            raise ParallelRescueSelectionError(f"{label}.{field} changed")


def _verify_generation_population(value: object, expected: int, label: str) -> None:
    generation = _mapping(value, label)
    examples = _integer(generation.get("examples"), f"{label}.examples", minimum=1)
    generated = _integer(generation.get("generated"), f"{label}.generated", minimum=0)
    failed = _integer(generation.get("failed"), f"{label}.failed", minimum=0)
    if examples != expected or generated + failed != expected:
        raise ParallelRescueSelectionError(
            f"{label} population accounting changed: "
            f"examples={examples}, generated={generated}, failed={failed}"
        )


def _checkpoint_hashes(checkpoint_root: Path, contract: object, label: str) -> dict[str, str]:
    hashes = _hash_map(contract, f"{label} checkpoint hashes")
    expected_names = {"barun_config.json", "model.safetensors", "tokenizer.json"}
    if set(hashes) != expected_names:
        raise ParallelRescueSelectionError(f"{label} checkpoint file contract changed")
    for name, digest in hashes.items():
        _require_file_hash(checkpoint_root / name, digest, f"{label} checkpoint {name}")
    manifest_path = checkpoint_root / "checkpoint_manifest.json"
    manifest = _load_json(manifest_path, f"{label} checkpoint manifest")
    if _mapping(manifest.get("file_sha256"), f"{label} checkpoint manifest hashes") != hashes:
        raise ParallelRescueSelectionError(f"{label} checkpoint manifest hash map changed")
    return hashes


def _eligibility(
    mobile: Mapping[str, int],
    presto: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    contract = _mapping(protocol.get("eligibility"), "eligibility")
    mobile_contract = _mapping(contract.get("mobile"), "eligibility.mobile")
    presto_contract = _mapping(contract.get("presto"), "eligibility.presto")
    bucket_counts = presto["bucket_counts"]
    bucket_exact = presto["bucket_exact"]
    simple = Fraction(bucket_exact["no_phenomenon"], bucket_counts["no_phenomenon"])
    revision = Fraction(bucket_exact["revision"], bucket_counts["revision"])
    disfluency = Fraction(bucket_exact["disfluency"], bucket_counts["disfluency"])
    max_gap = _fraction(
        presto_contract["maximum_no_phenomenon_to_revision_or_disfluency_gap"],
        "eligibility PRESTO maximum gap",
    )
    checks = {
        "mobile_ast_exact": mobile["ast_exact_numerator"]
        >= _integer(mobile_contract["ast_exact_numerator_at_least"], "Mobile threshold"),
        "mobile_catastrophic_unauthorized_actions": mobile["catastrophic_unauthorized_actions"]
        == _integer(
            mobile_contract["catastrophic_unauthorized_actions_equal"],
            "Mobile catastrophic threshold",
        ),
        "mobile_false_action": mobile["false_action_numerator"]
        == _integer(
            mobile_contract["false_action_numerator_equal"], "Mobile false-action threshold"
        ),
        "presto_ast_exact": presto["ast_exact_numerator"]
        >= _integer(presto_contract["ast_exact_numerator_at_least"], "PRESTO exact threshold"),
        "presto_schema_valid": presto["schema_valid_numerator"]
        >= _integer(presto_contract["schema_valid_numerator_at_least"], "PRESTO schema threshold"),
        "presto_abstention_f1": presto["abstention"]["f1"]
        >= _fraction(presto_contract["abstention_f1_at_least"], "PRESTO abstention threshold"),
        "presto_false_call": presto["false_call_numerator"]
        <= _integer(presto_contract["false_call_numerator_at_most"], "PRESTO false-call threshold"),
        "presto_revision_gap": simple - revision <= max_gap,
        "presto_disfluency_gap": simple - disfluency <= max_gap,
        "presto_revision_raw_tags": all(
            presto["raw_counts"].get(tag, 0) > 0 for tag in _REVISION_TAGS
        ),
    }
    expected_counts = {
        "no_phenomenon": _integer(presto_contract["no_phenomenon_rows"], "no-phenomenon rows"),
        "revision": _integer(presto_contract["revision_rows"], "revision rows"),
        "disfluency": _integer(presto_contract["disfluency_rows"], "disfluency rows"),
    }
    if {name: bucket_counts.get(name, 0) for name in expected_counts} != expected_counts:
        raise ParallelRescueSelectionError("PRESTO frozen headline bucket counts changed")
    if mobile["sample_count"] != mobile_contract["sample_count"]:
        raise ParallelRescueSelectionError("Mobile frozen sample count changed")
    if presto["sample_count"] != presto_contract["sample_count"]:
        raise ParallelRescueSelectionError("PRESTO frozen sample count changed")
    if presto["false_call_denominator"] != presto_contract["false_call_denominator"]:
        raise ParallelRescueSelectionError("PRESTO false-call denominator changed")
    details = {
        "presto_abstention_f1": _fraction_record(presto["abstention"]["f1"]),
        "presto_revision_gap": _fraction_record(simple - revision),
        "presto_disfluency_gap": _fraction_record(simple - disfluency),
    }
    return checks, details


def _candidate_record(
    *,
    trial_id: str,
    method: str,
    run_id: str,
    arm_id: str | None,
    checkpoint_relative_path: str,
    checkpoint_hashes: Mapping[str, str],
    checkpoint_manifest_sha256: str,
    mobile: Mapping[str, int],
    presto: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    checks, exact_details = _eligibility(mobile, presto, protocol)
    endpoint = _mapping(protocol.get("frozen_endpoints"), "frozen_endpoints")
    mobile_endpoint = _mapping(endpoint.get("mobile"), "frozen_endpoints.mobile")
    presto_endpoint = _mapping(endpoint.get("presto"), "frozen_endpoints.presto")
    mobile_reference = _integer(mobile_endpoint["ast_exact_numerator"], "Mobile endpoint")
    presto_reference = _integer(presto_endpoint["ast_exact_numerator"], "PRESTO endpoint")
    mobile_retention = Fraction(
        min(mobile["ast_exact_numerator"], mobile_reference), mobile_reference
    )
    presto_retention = Fraction(
        min(presto["ast_exact_numerator"], presto_reference), presto_reference
    )
    worst = min(mobile_retention, presto_retention)
    total = mobile_retention + presto_retention
    return {
        "trial_id": trial_id,
        "method": method,
        "run_id": run_id,
        "arm_id": arm_id,
        "checkpoint": {
            "relative_path": checkpoint_relative_path,
            "file_sha256": dict(sorted(checkpoint_hashes.items())),
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        },
        "metrics": {
            "mobile": {
                "sample_count": mobile["sample_count"],
                "ast_exact": {
                    "numerator": mobile["ast_exact_numerator"],
                    "denominator": 756,
                },
                "catastrophic_unauthorized_actions": mobile["catastrophic_unauthorized_actions"],
                "false_action": {
                    "numerator": mobile["false_action_numerator"],
                    "denominator": mobile["false_action_denominator"],
                },
            },
            "presto": {
                "sample_count": presto["sample_count"],
                "ast_exact": {
                    "numerator": presto["ast_exact_numerator"],
                    "denominator": 14_288,
                },
                "schema_valid": {
                    "numerator": presto["schema_valid_numerator"],
                    "denominator": 14_288,
                },
                "abstention_f1": exact_details["presto_abstention_f1"],
                "false_call_on_gate": {
                    "numerator": presto["false_call_numerator"],
                    "denominator": presto["false_call_denominator"],
                },
                "headline_bucket_counts": {
                    name: presto["bucket_counts"][name]
                    for name in ("no_phenomenon", "revision", "disfluency")
                },
                "no_phenomenon_minus_revision_gap": exact_details["presto_revision_gap"],
                "no_phenomenon_minus_disfluency_gap": exact_details["presto_disfluency_gap"],
                "revision_raw_tag_counts": {
                    tag: presto["raw_counts"].get(tag, 0) for tag in sorted(_REVISION_TAGS)
                },
            },
        },
        "hard_gate_checks": checks,
        "eligible": all(checks.values()),
        "retention": {
            "mobile": _fraction_record(mobile_retention),
            "presto": _fraction_record(presto_retention),
            "minimum": _fraction_record(worst),
            "sum": _fraction_record(total),
        },
        "_ranking": (worst, total),
    }


def _verify_continual_provenance(
    root: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = _mapping(result.get("provenance"), "continual result.provenance")
    file_provenance = _load_json(root / "provenance-validation.json", "continual provenance")
    if provenance != file_provenance:
        raise ParallelRescueSelectionError("continual embedded provenance differs from its file")
    if provenance.get("schema_version") != "barun-continual-recovery-provenance-validation-v1":
        raise ParallelRescueSelectionError("continual provenance schema changed")
    _boolean(
        provenance.get("validated_before_data_prep_training_or_scoring"),
        True,
        "continual provenance pre-scoring validation",
    )
    run_id = _run_id(result.get("run_id"), "continual run_id")
    if provenance.get("run_id") != run_id:
        raise ParallelRescueSelectionError("continual result/provenance run IDs differ")
    attempt_path = root / "attempt-preregistration.json"
    snapshot_path = root / "source-snapshot.json"
    attempt = _load_json(attempt_path, "continual attempt preregistration")
    snapshot = _load_json(snapshot_path, "continual source snapshot")
    attempt_contract = _mapping(provenance.get("attempt"), "continual provenance.attempt")
    snapshot_contract = _mapping(
        provenance.get("source_snapshot"), "continual provenance.source_snapshot"
    )
    _require_file_hash(attempt_path, attempt_contract.get("sha256"), "continual attempt")
    _require_file_hash(snapshot_path, snapshot_contract.get("sha256"), "continual source snapshot")
    if attempt.get("run_id") != run_id or snapshot.get("run_id") != run_id:
        raise ParallelRescueSelectionError("continual attempt/snapshot run ID mismatch")
    if attempt.get("schema_version") != "barun-continual-recovery-attempt-v1":
        raise ParallelRescueSelectionError("continual attempt schema changed")
    if attempt.get("status") != "frozen_before_remote_data_prep_training_or_scoring":
        raise ParallelRescueSelectionError("continual attempt was not frozen before scoring")
    selector = _mapping(attempt.get("shared_selector"), "continual attempt.shared_selector")
    expected_selector = {
        "path": PARALLEL_RESCUE_SELECTION_RELATIVE_PATH,
        "sha256": PARALLEL_RESCUE_SELECTION_SHA256,
    }
    if dict(selector) != expected_selector:
        raise ParallelRescueSelectionError("continual attempt lacks exact shared-selector binding")
    snapshot_frozen = _mapping(snapshot.get("frozen_files"), "continual snapshot.frozen_files")
    if snapshot_frozen.get("configs/continual_recovery_v1.json") != CONTINUAL_PROTOCOL_SHA256:
        raise ParallelRescueSelectionError("continual snapshot protocol binding changed")
    if (
        snapshot_frozen.get(PARALLEL_RESCUE_SELECTION_RELATIVE_PATH)
        != PARALLEL_RESCUE_SELECTION_SHA256
    ):
        raise ParallelRescueSelectionError("continual snapshot lacks shared-selector binding")
    snapshot_runner_sha = _sha256(
        snapshot_frozen.get("scripts/run_continual_recovery.py"),
        "continual snapshot runner hash",
    )
    provenance_selector = _mapping(
        provenance.get("shared_selector"), "continual provenance.shared_selector"
    )
    if (
        provenance_selector.get("path") != PARALLEL_RESCUE_SELECTION_RELATIVE_PATH
        or provenance_selector.get("sha256") != PARALLEL_RESCUE_SELECTION_SHA256
    ):
        raise ParallelRescueSelectionError("continual provenance lacks shared-selector binding")
    config = _mapping(provenance.get("recovery_config"), "continual provenance.recovery_config")
    if config.get("sha256") != CONTINUAL_PROTOCOL_SHA256:
        raise ParallelRescueSelectionError("continual provenance protocol hash changed")
    attempt_config = _mapping(attempt.get("recovery_config"), "continual attempt.recovery_config")
    if attempt_config.get("sha256") != CONTINUAL_PROTOCOL_SHA256:
        raise ParallelRescueSelectionError("continual attempt protocol hash changed")
    attempt_runner = _mapping(attempt.get("runner"), "continual attempt.runner")
    provenance_runner = _mapping(provenance.get("runner"), "continual provenance.runner")
    if (
        attempt_runner.get("sha256") != snapshot_runner_sha
        or provenance_runner.get("sha256") != snapshot_runner_sha
    ):
        raise ParallelRescueSelectionError("continual runner provenance anchors disagree")
    if result.get("preregistration_sha256") != CONTINUAL_PROTOCOL_SHA256:
        raise ParallelRescueSelectionError("continual result protocol hash changed")
    _require_file_hash(
        root / "preregistration.json", CONTINUAL_PROTOCOL_SHA256, "continual protocol"
    )
    if _mapping(attempt.get("source_snapshot"), "continual attempt.source_snapshot").get(
        "sha256"
    ) != sha256_file(snapshot_path):
        raise ParallelRescueSelectionError("continual attempt does not bind its snapshot")
    tree = _mapping(provenance.get("content_tree"), "continual provenance.content_tree")
    snapshot_tree = _mapping(snapshot.get("content_tree"), "continual snapshot.content_tree")
    tree_hash = _sha256(tree.get("sha256"), "continual staged content-tree hash")
    if snapshot_tree.get("sha256") != tree_hash:
        raise ParallelRescueSelectionError("continual snapshot/provenance tree hashes differ")
    checkpoint_chain = _mapping(
        provenance.get("checkpoint_chain"), "continual provenance.checkpoint_chain"
    )
    if checkpoint_chain.get("checkpoint_manifest_sha256") != PRESTO_ENDPOINT_MANIFEST_SHA256:
        raise ParallelRescueSelectionError("continual PRESTO endpoint manifest chain changed")
    machine_id = _integer(result.get("jarvis_machine_id"), "continual Jarvis machine ID", minimum=1)
    inventory = _mapping(
        provenance.get("prelaunch_inventory"), "continual provenance.prelaunch_inventory"
    )
    if inventory.get("project_machine_id") != machine_id:
        raise ParallelRescueSelectionError("continual machine ID differs from its inventory")
    protected = inventory.get("protected_machine_ids")
    if not isinstance(protected, list) or 463058 not in protected or machine_id in protected:
        raise ParallelRescueSelectionError("continual protected-machine inventory is invalid")
    if inventory.get("fresh_project_instance") is not True:
        raise ParallelRescueSelectionError("continual evidence does not identify a fresh instance")
    return {
        "run_id": run_id,
        "jarvis_machine_id": machine_id,
        "attempt_sha256": sha256_file(attempt_path),
        "source_snapshot_sha256": sha256_file(snapshot_path),
        "content_tree_sha256": tree_hash,
    }


def _verify_interpolation_provenance(
    root: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = _mapping(result.get("provenance"), "interpolation result.provenance")
    file_provenance = _load_json(
        root / "interpolation-provenance-verification.json", "interpolation provenance"
    )
    if provenance != file_provenance:
        raise ParallelRescueSelectionError(
            "interpolation embedded provenance differs from its file"
        )
    if provenance.get("schema_version") != "barun-interpolation-provenance-verification-v1":
        raise ParallelRescueSelectionError("interpolation provenance schema changed")
    _boolean(
        provenance.get("verified_before_model_or_data_scoring"),
        True,
        "interpolation provenance pre-scoring validation",
    )
    run_id = _run_id(result.get("run_id"), "interpolation run ID")
    if provenance.get("run_id") != run_id:
        raise ParallelRescueSelectionError("interpolation result/provenance run IDs differ")
    attempt_path = root / "interpolation-attempt-preregistration.json"
    snapshot_path = root / "interpolation-source-snapshot.json"
    attempt = _load_json(attempt_path, "interpolation attempt preregistration")
    snapshot = _load_json(snapshot_path, "interpolation source snapshot")
    attempt_sha = _sha256(
        provenance.get("attempt_preregistration_sha256"),
        "interpolation provenance attempt hash",
    )
    snapshot_sha = _sha256(
        provenance.get("source_snapshot_sha256"),
        "interpolation provenance snapshot hash",
    )
    _require_file_hash(attempt_path, attempt_sha, "interpolation attempt")
    _require_file_hash(snapshot_path, snapshot_sha, "interpolation source snapshot")
    if attempt.get("run_id") != run_id or snapshot.get("run_id") != run_id:
        raise ParallelRescueSelectionError("interpolation attempt/snapshot run ID mismatch")
    if attempt.get("schema_version") != "barun-interpolation-attempt-preregistration-v1":
        raise ParallelRescueSelectionError("interpolation attempt schema changed")
    if attempt.get("status") != "frozen_before_remote_model_or_data_scoring":
        raise ParallelRescueSelectionError("interpolation attempt was not frozen before scoring")
    expected_frozen = {
        PARALLEL_RESCUE_SELECTION_RELATIVE_PATH: PARALLEL_RESCUE_SELECTION_SHA256,
    }
    frozen_records: list[Mapping[str, Any]] = []
    for payload, label in (
        (attempt, "interpolation attempt.frozen_files"),
        (snapshot, "interpolation snapshot.frozen_files"),
    ):
        frozen = _mapping(payload.get("frozen_files"), label)
        frozen_records.append(frozen)
        for name, digest in expected_frozen.items():
            if frozen.get(name) != digest:
                raise ParallelRescueSelectionError(f"{label} lacks shared-selector binding")
        if frozen.get("configs/interpolation_rescue_v1.json") != INTERPOLATION_PROTOCOL_SHA256:
            raise ParallelRescueSelectionError(f"{label} protocol binding changed")
    if dict(frozen_records[0]) != dict(frozen_records[1]):
        raise ParallelRescueSelectionError("interpolation attempt/snapshot frozen files differ")
    frozen_runner_sha = _sha256(
        frozen_records[0].get("scripts/run_interpolation_rescue.py"),
        "interpolation frozen runner hash",
    )
    if provenance.get("runner_sha256") != frozen_runner_sha:
        raise ParallelRescueSelectionError("interpolation runner provenance anchors disagree")
    if provenance.get("shared_selector_sha256") != PARALLEL_RESCUE_SELECTION_SHA256:
        raise ParallelRescueSelectionError("interpolation provenance lacks shared-selector binding")
    if provenance.get("protocol_sha256") != INTERPOLATION_PROTOCOL_SHA256:
        raise ParallelRescueSelectionError("interpolation provenance protocol hash changed")
    if result.get("protocol_sha256") != INTERPOLATION_PROTOCOL_SHA256:
        raise ParallelRescueSelectionError("interpolation result protocol hash changed")
    _require_file_hash(
        root / "interpolation-protocol.json",
        INTERPOLATION_PROTOCOL_SHA256,
        "interpolation protocol",
    )
    attempt_snapshot = _mapping(
        attempt.get("source_snapshot"), "interpolation attempt.source_snapshot"
    )
    if attempt_snapshot.get("sha256") != snapshot_sha:
        raise ParallelRescueSelectionError("interpolation attempt does not bind its snapshot")
    tree = _mapping(provenance.get("content_tree"), "interpolation provenance.content_tree")
    snapshot_tree = _mapping(snapshot.get("content_tree"), "interpolation snapshot.content_tree")
    tree_hash = _sha256(tree.get("sha256"), "interpolation staged content-tree hash")
    if snapshot_tree.get("sha256") != tree_hash:
        raise ParallelRescueSelectionError("interpolation snapshot/provenance tree hashes differ")
    machine_id = _integer(
        result.get("jarvis_machine_id"), "interpolation Jarvis machine ID", minimum=1
    )
    if provenance.get("jarvis_machine_id") != machine_id:
        raise ParallelRescueSelectionError("interpolation result/provenance machine IDs differ")
    inventory = _mapping(
        attempt.get("prelaunch_inventory"), "interpolation attempt.prelaunch_inventory"
    )
    protected = inventory.get("protected_machine_ids")
    if (
        inventory.get("project_machine_id") != machine_id
        or inventory.get("fresh_project_instance") is not True
        or not isinstance(protected, list)
        or 463058 not in protected
        or machine_id in protected
    ):
        raise ParallelRescueSelectionError("interpolation protected-machine inventory is invalid")
    return {
        "run_id": run_id,
        "jarvis_machine_id": machine_id,
        "attempt_sha256": attempt_sha,
        "source_snapshot_sha256": snapshot_sha,
        "content_tree_sha256": tree_hash,
    }


def audit_continual_bundle(
    root: str | Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_manifests = {
        "artifact-sha256.json",
        "terminal-evaluation/mobile/artifact-sha256.json",
    }
    tree = _verify_tree_manifest(
        root,
        expected_schema="barun-continual-recovery-artifacts-v1",
        expected_manifest_paths=expected_manifests,
    )
    bundle_root = tree["root"]
    result = _load_json(bundle_root / "result.json", "continual result")
    if result.get("schema_version") != CONTINUAL_RESULT_SCHEMA:
        raise ParallelRescueSelectionError("continual result schema changed")
    provenance = _verify_continual_provenance(bundle_root, result)
    terminal = _mapping(result.get("terminal_evaluation"), "continual terminal_evaluation")
    if (
        terminal.get("selection_policy") != "final_checkpoint_only_no_dev_tuning"
        or terminal.get("rounds") != 1
        or terminal.get("mobile_evaluations") != 1
        or terminal.get("presto_evaluations") != 1
    ):
        raise ParallelRescueSelectionError("continual terminal trial accounting changed")
    data = _mapping(result.get("data"), "continual data")
    mobile_data = _mapping(data.get("mobile"), "continual data.mobile")
    if (
        mobile_data.get("development_rows") != 756
        or mobile_data.get("official_evaluation_rows_opaque_unparsed") != 961
        or mobile_data.get("official_evaluation_rows_read") != 0
    ):
        raise ParallelRescueSelectionError("continual Mobile firewall or population changed")
    presto_data = _mapping(data.get("presto"), "continual data.presto")
    if (
        presto_data.get("development_rows") != 14_288
        or presto_data.get("official_test_rows_read") != 0
    ):
        raise ParallelRescueSelectionError("continual PRESTO population or firewall changed")
    _verify_presto_firewall(presto_data.get("official_test_firewall"), "continual PRESTO firewall")
    if _field(result, ("training", "optimizer_steps"), "continual optimizer steps") != 168:
        raise ParallelRescueSelectionError("continual optimizer-step count changed")
    checkpoint_root = bundle_root / "checkpoint"
    output_checkpoint = _mapping(result.get("output_checkpoint"), "continual output_checkpoint")
    checkpoint_hashes = _checkpoint_hashes(
        checkpoint_root,
        output_checkpoint.get("file_sha256"),
        "continual C",
    )
    checkpoint_manifest = _load_json(
        checkpoint_root / "checkpoint_manifest.json", "continual output checkpoint manifest"
    )
    endpoint = _mapping(protocol.get("frozen_endpoints"), "frozen_endpoints")
    presto_endpoint = _mapping(endpoint.get("presto"), "frozen_endpoints.presto")
    input_hashes = _mapping(
        checkpoint_manifest.get("input_checkpoint_sha256"),
        "continual output checkpoint input hashes",
    )
    if input_hashes.get("model.safetensors") != presto_endpoint.get("model_sha256"):
        raise ParallelRescueSelectionError("continual output checkpoint parent model changed")
    result_input = _mapping(result.get("input_checkpoint"), "continual input_checkpoint")
    if _mapping(result_input.get("file_sha256"), "continual input checkpoint hashes").get(
        "model.safetensors"
    ) != presto_endpoint.get("model_sha256"):
        raise ParallelRescueSelectionError("continual result input endpoint changed")
    mobile_result = _mapping(terminal.get("mobile"), "continual terminal Mobile result")
    mobile_metrics = _verify_mobile_evidence(
        bundle_root=bundle_root,
        relative_root="terminal-evaluation/mobile",
        result=mobile_result,
        checkpoint_hashes=checkpoint_hashes,
    )
    presto_result = _mapping(terminal.get("presto"), "continual terminal PRESTO result")
    aggregate = _mapping(presto_result.get("aggregate"), "continual PRESTO aggregate")
    aggregate_path = bundle_root / "terminal-evaluation" / "presto" / "scores" / "aggregate.json"
    samples_path = bundle_root / "terminal-evaluation" / "presto" / "scores" / "sample_scores.jsonl"
    _require_file_hash(
        aggregate_path,
        presto_result.get("aggregate_sha256"),
        "continual PRESTO aggregate",
    )
    presto_metrics = _verify_presto_evidence(
        aggregate_path=aggregate_path,
        samples_path=samples_path,
        aggregate=aggregate,
        expected_samples_sha256=presto_result.get("sample_scores_sha256"),
    )
    _verify_generation_population(
        presto_result.get("generation"), 14_288, "continual PRESTO generation"
    )
    candidate = _candidate_record(
        trial_id="C",
        method="continual_recovery",
        run_id=provenance["run_id"],
        arm_id=None,
        checkpoint_relative_path="checkpoint",
        checkpoint_hashes=checkpoint_hashes,
        checkpoint_manifest_sha256=sha256_file(checkpoint_root / "checkpoint_manifest.json"),
        mobile=mobile_metrics,
        presto=presto_metrics,
        protocol=protocol,
    )
    capability_pass = candidate["hard_gate_checks"]["mobile_ast_exact"] and all(
        candidate["hard_gate_checks"][name]
        for name in candidate["hard_gate_checks"]
        if name.startswith("presto_")
    )
    gate = _mapping(result.get("gate"), "continual result.gate")
    if gate.get("mobile_passed") is not candidate["hard_gate_checks"]["mobile_ast_exact"]:
        raise ParallelRescueSelectionError("continual stored Mobile gate is inconsistent")
    if gate.get("presto_passed") is not all(
        candidate["hard_gate_checks"][name]
        for name in candidate["hard_gate_checks"]
        if name.startswith("presto_")
    ):
        raise ParallelRescueSelectionError("continual stored PRESTO gate is inconsistent")
    if gate.get("passed") is not capability_pass:
        raise ParallelRescueSelectionError("continual stored joint gate is inconsistent")
    audit = {
        **provenance,
        "artifact_manifest_sha256": tree["manifest_sha256"],
        "artifact_file_count": tree["manifest_file_count"],
        "official_test_evaluations": 0,
    }
    return candidate, audit


def audit_interpolation_bundle(
    root: str | Path,
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_manifests = {"artifact-sha256.json"}
    for arm_id in _ARM_TO_TRIAL:
        expected_manifests.add(
            f"interpolation-arms/{arm_id}/interpolation-mobile-evaluation/artifact-sha256.json"
        )
    tree = _verify_tree_manifest(
        root,
        expected_schema="barun-interpolation-artifacts-v1",
        expected_manifest_paths=expected_manifests,
    )
    bundle_root = tree["root"]
    result = _load_json(bundle_root / "interpolation-result.json", "interpolation result")
    if result.get("schema_version") != INTERPOLATION_RESULT_SCHEMA:
        raise ParallelRescueSelectionError("interpolation result schema changed")
    provenance = _verify_interpolation_provenance(bundle_root, result)
    endpoint = _mapping(protocol.get("frozen_endpoints"), "frozen_endpoints")
    mobile_endpoint = _mapping(endpoint.get("mobile"), "frozen_endpoints.mobile")
    presto_endpoint = _mapping(endpoint.get("presto"), "frozen_endpoints.presto")
    trajectory = _mapping(
        result.get("trajectory_endpoint_sha256"), "interpolation trajectory endpoints"
    )
    if _mapping(trajectory.get("candidate_v2"), "candidate-v2 trajectory hashes").get(
        "model.safetensors"
    ) != mobile_endpoint.get("model_sha256"):
        raise ParallelRescueSelectionError("interpolation Mobile endpoint changed")
    if _mapping(trajectory.get("presto"), "PRESTO trajectory hashes").get(
        "model.safetensors"
    ) != presto_endpoint.get("model_sha256"):
        raise ParallelRescueSelectionError("interpolation PRESTO endpoint changed")
    accounting = _mapping(result.get("trial_accounting"), "interpolation trial_accounting")
    expected_accounting = {
        "frozen_arms": 3,
        "mobile_generations": 3,
        "presto_generations": 3,
        "adaptive_followup_alphas": 0,
        "every_arm_received_both_evaluations": True,
        "new_joint_development_selection_trials": 3,
    }
    for name, expected in expected_accounting.items():
        if accounting.get(name) != expected:
            raise ParallelRescueSelectionError(f"interpolation trial accounting changed at {name}")
    if result.get("official_test_evaluations") != 0:
        raise ParallelRescueSelectionError("interpolation recorded official-test evaluations")
    data = _mapping(result.get("data"), "interpolation data")
    if data.get("mobile_official_test_artifacts_accessed") != []:
        raise ParallelRescueSelectionError("interpolation accessed Mobile official artifacts")
    _verify_presto_firewall(
        data.get("presto_official_test_firewall"), "interpolation PRESTO firewall"
    )
    bundle_manifest = _load_json(
        bundle_root / "interpolation-bundle-manifest.json", "interpolation bundle manifest"
    )
    for field, expected in {
        "sample_level_mobile_and_presto_evidence_included": True,
        "all_three_interpolation_checkpoints_included": True,
        "launch_provenance_included": True,
        "optimizer_state_included": False,
        "official_test_artifacts_included": False,
    }.items():
        if bundle_manifest.get(field) is not expected:
            raise ParallelRescueSelectionError(f"interpolation bundle manifest changed at {field}")
    arms = result.get("arms")
    if not isinstance(arms, list) or len(arms) != 3:
        raise ParallelRescueSelectionError("interpolation result must expose exactly three arms")
    by_id: dict[str, Mapping[str, Any]] = {}
    for arm in arms:
        payload = _mapping(arm, "interpolation arm result")
        arm_id = _string(payload.get("arm_id"), "interpolation arm_id")
        if arm_id in by_id:
            raise ParallelRescueSelectionError("duplicate interpolation arm ID")
        by_id[arm_id] = payload
    if set(by_id) != set(_ARM_TO_TRIAL):
        raise ParallelRescueSelectionError("interpolation arm IDs differ from the frozen three")
    candidates: list[dict[str, Any]] = []
    for arm_id, (trial_id, alpha) in _ARM_TO_TRIAL.items():
        arm = by_id[arm_id]
        arm_root_relative = f"interpolation-arms/{arm_id}"
        arm_root = bundle_root / arm_root_relative
        file_result = _load_json(arm_root / "interpolation-arm-result.json", f"{arm_id} result")
        if file_result != arm:
            raise ParallelRescueSelectionError(f"embedded and file-backed {arm_id} results differ")
        _boolean(
            arm.get("both_frozen_evaluations_completed_once"),
            True,
            f"{arm_id} paired evaluations",
        )
        alpha_payload = _mapping(arm.get("alpha"), f"{arm_id}.alpha")
        observed_alpha = Fraction(
            _integer(alpha_payload.get("numerator"), f"{arm_id} alpha numerator"),
            _integer(alpha_payload.get("denominator"), f"{arm_id} alpha denominator", minimum=1),
        )
        if observed_alpha != alpha:
            raise ParallelRescueSelectionError(f"{arm_id} alpha changed")
        checkpoint_root = arm_root / "interpolation-checkpoint"
        checkpoint = _mapping(arm.get("checkpoint"), f"{arm_id}.checkpoint")
        checkpoint_hashes = _checkpoint_hashes(
            checkpoint_root,
            checkpoint.get("file_sha256"),
            trial_id,
        )
        checkpoint_manifest_hash = _require_file_hash(
            checkpoint_root / "checkpoint_manifest.json",
            checkpoint.get("checkpoint_manifest_sha256"),
            f"{trial_id} checkpoint manifest",
        )
        mobile_relative = f"{arm_root_relative}/interpolation-mobile-evaluation"
        mobile_result = _mapping(arm.get("mobile_result"), f"{arm_id}.mobile_result")
        mobile_metrics = _verify_mobile_evidence(
            bundle_root=bundle_root,
            relative_root=mobile_relative,
            result=mobile_result,
            checkpoint_hashes=checkpoint_hashes,
        )
        aggregate = _mapping(arm.get("presto_aggregate"), f"{arm_id}.presto_aggregate")
        presto_root = arm_root / "interpolation-presto-evaluation"
        aggregate_path = presto_root / "interpolation-scores" / "aggregate.json"
        samples_path = presto_root / "interpolation-scores" / "sample_scores.jsonl"
        sample_contract = _mapping(
            arm.get("presto_sample_scores"), f"{arm_id}.presto_sample_scores"
        )
        presto_metrics = _verify_presto_evidence(
            aggregate_path=aggregate_path,
            samples_path=samples_path,
            aggregate=aggregate,
            expected_samples_sha256=sample_contract.get("sha256"),
        )
        _verify_generation_population(
            arm.get("presto_generation"), 14_288, f"{arm_id} PRESTO generation"
        )
        candidate = _candidate_record(
            trial_id=trial_id,
            method="checkpoint_interpolation",
            run_id=provenance["run_id"],
            arm_id=arm_id,
            checkpoint_relative_path=f"{arm_root_relative}/interpolation-checkpoint",
            checkpoint_hashes=checkpoint_hashes,
            checkpoint_manifest_sha256=checkpoint_manifest_hash,
            mobile=mobile_metrics,
            presto=presto_metrics,
            protocol=protocol,
        )
        method_capability = candidate["hard_gate_checks"]["mobile_ast_exact"] and all(
            candidate["hard_gate_checks"][name]
            for name in candidate["hard_gate_checks"]
            if name.startswith("presto_")
        )
        joint = _mapping(arm.get("joint_gate"), f"{arm_id}.joint_gate")
        if joint.get("passed") is not method_capability:
            raise ParallelRescueSelectionError(f"{arm_id} stored joint gate is inconsistent")
        candidates.append(candidate)
    audit = {
        **provenance,
        "artifact_manifest_sha256": tree["manifest_sha256"],
        "artifact_file_count": tree["manifest_file_count"],
        "official_test_evaluations": 0,
    }
    return candidates, audit


def rank_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen exact-rational maximin, sum, and fixed-order rule."""

    by_id: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        trial_id = candidate.get("trial_id")
        if not isinstance(trial_id, str) or trial_id in by_id:
            raise ParallelRescueSelectionError("candidate trial IDs are missing or duplicated")
        by_id[trial_id] = candidate
    if set(by_id) != set(_TRIAL_ORDER):
        raise ParallelRescueSelectionError("selection requires exactly C, I25, I50, and I75")
    eligible = [
        by_id[trial_id] for trial_id in _TRIAL_ORDER if by_id[trial_id].get("eligible") is True
    ]
    ordered = sorted(
        eligible,
        key=lambda candidate: (
            -candidate["_ranking"][0],
            -candidate["_ranking"][1],
            _TRIAL_ORDER.index(str(candidate["trial_id"])),
        ),
    )
    if not ordered:
        return {
            "status": "no_eligible_trial",
            "selected_trial_id": None,
            "ordered_eligible_trial_ids": [],
            "promotion": "none",
            "additional_alpha_or_recipe_retry": False,
        }
    return {
        "status": "selected_eligible_trial",
        "selected_trial_id": ordered[0]["trial_id"],
        "ordered_eligible_trial_ids": [candidate["trial_id"] for candidate in ordered],
        "promotion": "one_checkpoint",
        "additional_alpha_or_recipe_retry": False,
    }


def select_parallel_rescue(
    *,
    continual_bundle_root: str | Path,
    interpolation_bundle_root: str | Path,
    outer_attempt_path: str | Path,
    outer_attempt_sha256: str,
) -> dict[str, Any]:
    """Verify both downloaded bundles and select at most one checkpoint offline."""

    protocol = load_selection_protocol(
        outer_attempt_path,
        expected_sha256=outer_attempt_sha256,
    )
    continual, continual_audit = audit_continual_bundle(continual_bundle_root, protocol)
    interpolation, interpolation_audit = audit_interpolation_bundle(
        interpolation_bundle_root, protocol
    )
    candidates = [continual, *interpolation]
    if continual_audit["run_id"] == interpolation_audit["run_id"]:
        raise ParallelRescueSelectionError("continual and interpolation run IDs must be distinct")
    selection = rank_candidates(candidates)
    public_candidates: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: _TRIAL_ORDER.index(item["trial_id"])):
        public_candidates.append(
            {key: value for key, value in candidate.items() if key != "_ranking"}
        )
    return {
        "schema_version": PARALLEL_RESCUE_SELECTION_RECEIPT_VERSION,
        "selection_protocol": {
            "path": PARALLEL_RESCUE_SELECTION_RELATIVE_PATH,
            "sha256": PARALLEL_RESCUE_SELECTION_SHA256,
            "status": protocol["status"],
        },
        "bundle_audits": {
            "continual_recovery": continual_audit,
            "checkpoint_interpolation": interpolation_audit,
        },
        "trial_accounting": {
            "joint_development_trials": 4,
            "trial_ids": ["C", "I25", "I50", "I75"],
            "adaptive_followups": 0,
            "outer_evaluations": 0,
        },
        "candidates": public_candidates,
        "selection": selection,
        "official_test_evaluations": 0,
        "claim_limits": list(protocol["claim_limits"]),
    }


def write_exclusive_receipt(
    path: str | Path,
    receipt: Mapping[str, Any],
    *,
    forbidden_roots: Sequence[str | Path],
) -> Path:
    """Write one optional receipt and refuse paths inside either immutable bundle."""

    destination = Path(path).resolve()
    for root in forbidden_roots:
        resolved_root = Path(root).resolve(strict=True)
        try:
            destination.relative_to(resolved_root)
        except ValueError:
            continue
        raise ParallelRescueSelectionError(
            f"receipt must be outside immutable evidence bundle {resolved_root}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                receipt,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise ParallelRescueSelectionError(
            f"refusing to overwrite existing selection receipt: {destination}"
        ) from error
    return destination


__all__ = [
    "PARALLEL_RESCUE_SELECTION_RECEIPT_VERSION",
    "PARALLEL_RESCUE_SELECTION_SCHEMA_VERSION",
    "PARALLEL_RESCUE_SELECTION_SHA256",
    "ParallelRescueSelectionError",
    "audit_continual_bundle",
    "audit_interpolation_bundle",
    "load_selection_protocol",
    "rank_candidates",
    "select_parallel_rescue",
    "sha256_file",
    "write_exclusive_receipt",
]
