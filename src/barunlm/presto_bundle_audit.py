"""Independent, fail-closed audit of a downloaded PRESTO essential bundle.

The remote runner's aggregate metrics and result decision are treated as claims. This
module anchors them to a local preregistration and completed JarvisLabs safe-run record,
verifies every downloaded byte, and recomputes decision metrics from sample evidence.
It performs no network access and never loads model weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from barunlm.evaluation.action_ir import (
    ActionIRError,
    action_ir_equal,
    decode_json_object,
)
from barunlm.evaluation.presto import (
    PRESTO_SCORER_VERSION_V1,
    PrestoScoreError,
    parse_presto_action,
)

SHA256_RE = re.compile(r"[0-9a-f]{64}")
RUN_ID_RE = re.compile(r"\d{8}-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*-s\d+")
MANAGED_RUN_ID_RE = re.compile(r"r_[0-9a-f]+")
CHECKPOINT_FILES = frozenset({"barun_config.json", "model.safetensors", "tokenizer.json"})
METRIC_SCOPE = "derived_action_ir_not_native_presto_semantic_parse"

_REQUIRED_BUNDLE_FILES = frozenset(
    {
        "artifact-sha256.json",
        "base-eval/predictions.jsonl",
        "base-eval/predictions.jsonl.manifest.json",
        "base-eval/scores/aggregate.json",
        "base-eval/scores/sample_scores.jsonl",
        "bundle-manifest.json",
        "checkpoint/barun_config.json",
        "checkpoint/checkpoint_manifest.json",
        "checkpoint/model.safetensors",
        "checkpoint/tokenizer.json",
        "data/audit.json",
        "data/train-focus-audit.json",
        "environment.json",
        "post-eval/predictions.jsonl",
        "post-eval/predictions.jsonl.manifest.json",
        "post-eval/scores/aggregate.json",
        "post-eval/scores/sample_scores.jsonl",
        "preregistration.json",
        "progress.jsonl",
        "recipe.json",
        "repository-tests.log",
        "result.json",
        "training-config.json",
        "training/metrics.jsonl",
        "training/run_manifest.json",
        "training/summary.json",
    }
)
_OPTIONAL_BUNDLE_FILES = frozenset({"training/best_checkpoint.json"})
_SAMPLE_FIELDS = frozenset(
    {
        "ast_exact",
        "contextual",
        "decision_correct",
        "false_call_on_gate",
        "generation_failure",
        "gold",
        "gold_decision",
        "phenomenon",
        "phenomenon_group",
        "parse_valid",
        "predicted_decision",
        "prediction",
        "prediction_error_code",
        "prediction_raw",
        "root_intent",
        "sample_id",
        "schema_valid",
        "schema_version",
        "truncated",
    }
)
_PREDICTION_FIELDS = frozenset(
    {
        "generated_tokens",
        "generation_failure",
        "id",
        "prediction_raw",
        "prompt_tokens",
        "truncated",
    }
)
_FIREWALL_ZERO_ACCESS = {
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
}


class PrestoBundleAuditError(RuntimeError):
    """A downloaded PRESTO bundle cannot support its recorded result."""


@dataclass(frozen=True, slots=True)
class PrestoBundleAuditResult:
    """Machine-readable receipt returned only after every audit check passes."""

    schema_version: str
    passed: bool
    bundle_root: str
    artifact_count: int
    artifact_manifest_sha256: str
    run_id: str
    jarvis_machine_id: int
    jarvis_managed_run_id: str
    jarvis_final_status: str
    recipe_sha256: str
    input_checkpoint_sha256: Mapping[str, str]
    output_checkpoint_sha256: Mapping[str, str]
    development_sample_count: int
    recomputed_base: Mapping[str, Any]
    recomputed_post_sft: Mapping[str, Any]
    recomputed_gate: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fail(message: str) -> None:
    raise PrestoBundleAuditError(message)


def _reject_constant(value: str) -> None:
    _fail(f"JSON contains a non-finite numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _decode_json(raw: str, *, source: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise PrestoBundleAuditError(f"invalid JSON in {source}: {error}") from error


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise PrestoBundleAuditError(f"cannot read JSON artifact {path}: {error}") from error
    value = _decode_json(raw, source=str(path))
    if not isinstance(value, dict):
        _fail(f"JSON artifact must contain one object: {path}")
    return value


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    _fail(f"blank JSONL row at {path}:{line_number}")
                value = _decode_json(line, source=f"{path}:{line_number}")
                if not isinstance(value, dict):
                    _fail(f"JSONL row must be an object at {path}:{line_number}")
                rows.append(value)
    except (OSError, UnicodeError) as error:
        raise PrestoBundleAuditError(f"cannot read JSONL artifact {path}: {error}") from error
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be an array")
    return value


def _string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        _fail(f"{label} must be a string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label} must be finite")
    return number


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be boolean")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = _string(value, label)
    if SHA256_RE.fullmatch(digest) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _sha_map(value: Any, label: str, *, exact_names: set[str] | None = None) -> dict[str, str]:
    mapping = _mapping(value, label)
    if exact_names is not None and set(mapping) != exact_names:
        _fail(f"{label} must contain exactly {sorted(exact_names)}, got {sorted(mapping)}")
    return {str(name): _sha256(digest, f"{label}.{name}") for name, digest in mapping.items()}


def _expect_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _expect_close(actual: Any, expected: float, label: str) -> None:
    number = _number(actual, label)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12):
        _fail(f"{label} mismatch: expected {expected!r}, got {number!r}")


def _file_sha256(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.resolve(strict=True)
    if resolved not in cache:
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        cache[resolved] = digest.hexdigest()
    return cache[resolved]


def _safe_relative_path(value: Any, label: str) -> str:
    raw = _string(value, label)
    path = PurePosixPath(raw)
    if raw != path.as_posix() or path.is_absolute() or not path.parts:
        _fail(f"{label} is not a normalized relative POSIX path: {raw!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"{label} contains an unsafe path component: {raw!r}")
    return raw


def _verify_artifact_manifest(root: Path, cache: dict[Path, str]) -> tuple[dict[str, str], str]:
    if not root.is_dir():
        _fail(f"bundle root is not a directory: {root}")
    actual_files: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in tuple(directory_names) + tuple(file_names):
            candidate = directory_path / name
            if candidate.is_symlink():
                _fail(f"bundle contains a symbolic link: {candidate.relative_to(root)}")
        for name in file_names:
            candidate = directory_path / name
            if not candidate.is_file():
                _fail(f"bundle contains a non-regular file: {candidate.relative_to(root)}")
            actual_files.add(candidate.relative_to(root).as_posix())

    allowed = _REQUIRED_BUNDLE_FILES | _OPTIONAL_BUNDLE_FILES
    missing = sorted(_REQUIRED_BUNDLE_FILES.difference(actual_files))
    unknown = sorted(actual_files.difference(allowed))
    if missing or unknown:
        _fail(f"bundle file set mismatch: missing={missing}, unknown={unknown}")

    manifest_path = root / "artifact-sha256.json"
    raw_manifest = _load_json(manifest_path)
    manifest: dict[str, str] = {}
    for raw_name, raw_digest in raw_manifest.items():
        name = _safe_relative_path(raw_name, "artifact manifest key")
        if name == "artifact-sha256.json":
            _fail("artifact manifest must not recursively list itself")
        manifest[name] = _sha256(raw_digest, f"artifact manifest digest for {name}")
    expected_names = actual_files - {"artifact-sha256.json"}
    if set(manifest) != expected_names:
        _fail(
            "artifact manifest coverage mismatch: "
            f"missing={sorted(expected_names - set(manifest))}, "
            f"unknown={sorted(set(manifest) - expected_names)}"
        )
    for name, expected in sorted(manifest.items()):
        actual = _file_sha256(root / name, cache)
        if actual != expected:
            _fail(f"artifact SHA-256 mismatch for {name}: expected {expected}, got {actual}")
    return manifest, _file_sha256(manifest_path, cache)


def _flag_value(arguments: Sequence[Any], flag: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1:
        _fail(f"Jarvis requested arguments must contain {flag!r} exactly once")
    position = positions[0]
    if position + 1 >= len(arguments):
        _fail(f"Jarvis requested argument {flag!r} has no value")
    return _string(arguments[position + 1], f"value after {flag}")


def _path_has_suffix(path: str, suffix: str) -> bool:
    path_parts = PurePosixPath(path).parts
    suffix_parts = PurePosixPath(suffix).parts
    return len(path_parts) >= len(suffix_parts) and path_parts[-len(suffix_parts) :] == suffix_parts


def _original_phenomenon_group(value: str) -> str:
    """Reproduce the immutable run's v1 taxonomy, including its known omission."""

    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if normalized in {"", "none", "no_phenomenon", "no_phenomena"}:
        return "no_phenomenon"
    if "revision" in normalized:
        return "revision"
    if "disfluen" in normalized:
        return "disfluency"
    if "code" in normalized and "switch" in normalized:
        return "code_switching"
    return "other"


def _audit_external_anchors(
    prereg: Mapping[str, Any], jarvis: Mapping[str, Any], bundle_root: Path
) -> tuple[str, int, str, str, dict[str, str]]:
    _expect_equal(
        prereg.get("schema_version"),
        "barun-experiment-preregistration-v1",
        "local preregistration schema",
    )
    _expect_equal(
        prereg.get("status"),
        "frozen_before_remote_launch_or_model_scoring",
        "local preregistration status",
    )
    run_id = _string(prereg.get("run_id"), "local preregistration run_id")
    if RUN_ID_RE.fullmatch(run_id) is None:
        _fail("local preregistration run_id has an invalid immutable-run format")
    frozen = _mapping(prereg.get("hypothesis_and_decision"), "hypothesis_and_decision")
    recipe_sha = _sha256(frozen.get("sha256"), "frozen recipe SHA-256")
    _expect_equal(frozen.get("path"), "configs/presto_stage_v1.json", "frozen recipe path")
    input_record = _mapping(prereg.get("input_checkpoint"), "input_checkpoint")
    input_hashes = {
        "model.safetensors": _sha256(input_record.get("model_sha256"), "input model SHA-256"),
        "barun_config.json": _sha256(input_record.get("config_sha256"), "input config SHA-256"),
        "tokenizer.json": _sha256(input_record.get("tokenizer_sha256"), "input tokenizer SHA-256"),
    }
    input_path = _string(input_record.get("path"), "input checkpoint path")

    _expect_equal(jarvis.get("schema_version"), 1, "Jarvis record schema")
    _expect_equal(jarvis.get("controller"), "infra/jarvis/safe_run.py", "Jarvis controller")
    _expect_equal(jarvis.get("created_by_safe_run"), True, "safe-run ownership evidence")
    machine_id = _integer(jarvis.get("machine_id"), "Jarvis machine_id", minimum=1)
    managed_run_id = _string(jarvis.get("run_id"), "Jarvis managed run_id")
    if MANAGED_RUN_ID_RE.fullmatch(managed_run_id) is None:
        _fail("Jarvis managed run_id has an invalid format")

    create = _mapping(jarvis.get("create_summary"), "Jarvis create_summary")
    final = _mapping(jarvis.get("final_instance"), "Jarvis final_instance")
    latest = _mapping(jarvis.get("latest_run_status"), "Jarvis latest_run_status")
    launch = _mapping(jarvis.get("launch_summary"), "Jarvis launch_summary")
    for label, record in (
        ("create_summary", create),
        ("final_instance", final),
        ("latest_run_status", latest),
        ("launch_summary", launch),
    ):
        _expect_equal(record.get("machine_id"), machine_id, f"Jarvis {label} machine_id")
    _expect_equal(latest.get("run_id"), managed_run_id, "Jarvis latest managed run_id")
    _expect_equal(launch.get("run_id"), managed_run_id, "Jarvis launch managed run_id")
    _expect_equal(latest.get("state"), "succeeded", "Jarvis managed-run state")
    _expect_equal(latest.get("exit_code"), 0, "Jarvis managed-run exit code")
    _expect_equal(final.get("status"), "Paused", "Jarvis final instance status")
    instance_name = _string(jarvis.get("instance_name"), "Jarvis instance name")
    if not instance_name.startswith("barun-"):
        _fail("Jarvis project instance name must start with 'barun-'")
    _expect_equal(create.get("name"), instance_name, "Jarvis created instance name")
    _expect_equal(final.get("name"), instance_name, "Jarvis final instance name")

    compute = _mapping(prereg.get("compute"), "local preregistration compute")
    _expect_equal(compute.get("provider"), "JarvisLabs", "compute provider")
    _expect_equal(compute.get("instance_name"), instance_name, "preregistered instance name")
    protected = _mapping(compute.get("protected_running_instance"), "protected instance")
    if machine_id == _integer(protected.get("machine_id"), "protected machine_id", minimum=1):
        _fail("project run used the explicitly protected Jarvis instance")

    preexisting = _sequence(jarvis.get("preexisting_resources"), "preexisting_resources")
    preexisting_ids = {
        _integer(
            _mapping(item, "preexisting resource").get("machine_id"),
            "preexisting machine_id",
            minimum=1,
        )
        for item in preexisting
    }
    if machine_id in preexisting_ids:
        _fail("project machine_id appears in the pre-existing resource denylist")

    events = _sequence(jarvis.get("events"), "Jarvis events")
    pause_events = [
        _mapping(event, "Jarvis event")
        for event in events
        if _mapping(event, "Jarvis event").get("event") == "pause_verified"
    ]
    if len(pause_events) != 1 or pause_events[0].get("status") != "Paused":
        _fail("Jarvis record lacks one verified final pause event")

    download = _mapping(jarvis.get("artifact_download"), "Jarvis artifact_download")
    _expect_equal(download.get("direction"), "download", "artifact transfer direction")
    _expect_equal(download.get("exit_code"), 0, "artifact download exit code")
    _expect_equal(download.get("machine_id"), machine_id, "artifact download machine_id")
    _expect_equal(download.get("recursive"), True, "artifact recursive download")
    destination = Path(_string(download.get("dest"), "artifact download destination")).resolve(
        strict=True
    )
    if destination != bundle_root:
        _fail(f"audited bundle differs from safe-run download destination: {bundle_root}")

    requested = _mapping(jarvis.get("requested"), "Jarvis requested launch")
    _expect_equal(requested.get("append_jarvis_machine_id"), True, "Jarvis ID append policy")
    _expect_equal(requested.get("script"), "scripts/run_presto_stage.py", "remote script")
    arguments = _sequence(requested.get("remote_args"), "Jarvis remote_args")
    _expect_equal(_flag_value(arguments, "--run-id"), run_id, "requested experiment run_id")
    for flag, name in (
        ("--input-model-sha256", "model.safetensors"),
        ("--input-config-sha256", "barun_config.json"),
        ("--input-tokenizer-sha256", "tokenizer.json"),
    ):
        _expect_equal(_flag_value(arguments, flag), input_hashes[name], f"requested {name} hash")
    requested_input_path = _flag_value(arguments, "--input-checkpoint-dir")
    if not _path_has_suffix(requested_input_path, input_path):
        _fail("requested remote input checkpoint differs from frozen checkpoint path")
    return run_id, machine_id, managed_run_id, recipe_sha, input_hashes


def _audit_recipe_and_identities(
    *,
    root: Path,
    cache: dict[Path, str],
    run_id: str,
    machine_id: int,
    recipe_sha: str,
    input_hashes: Mapping[str, str],
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, str]]:
    recipe_path = root / "recipe.json"
    _expect_equal(_file_sha256(recipe_path, cache), recipe_sha, "downloaded recipe SHA-256")
    recipe = _load_json(recipe_path)
    _expect_equal(
        recipe.get("schema_version"),
        "barun-presto-stage-preregistration-v1",
        "recipe schema",
    )
    _expect_equal(recipe.get("recipe_id"), "presto-context-safety-sft-v1", "recipe ID")
    proposed = _mapping(recipe.get("proposed_input_checkpoint"), "recipe input checkpoint")
    proposed_hashes = {
        "model.safetensors": proposed.get("model_sha256"),
        "barun_config.json": proposed.get("config_sha256"),
        "tokenizer.json": proposed.get("tokenizer_sha256"),
    }
    _expect_equal(proposed_hashes, dict(input_hashes), "recipe input checkpoint hashes")
    tokenizer = _mapping(recipe.get("tokenizer_identity"), "recipe tokenizer identity")
    _expect_equal(tokenizer.get("sha256"), input_hashes["tokenizer.json"], "recipe tokenizer hash")
    evaluation = _mapping(recipe.get("evaluation"), "recipe evaluation")
    _expect_equal(evaluation.get("native_presto_metric"), False, "native metric policy")
    _expect_equal(
        evaluation.get("decoding"),
        "unconstrained deterministic greedy",
        "decoding policy",
    )

    effective = _load_json(root / "preregistration.json")
    _expect_equal(
        effective.get("schema_version"),
        "barun-presto-effective-preregistration-v1",
        "effective preregistration schema",
    )
    _expect_equal(effective.get("run_id"), run_id, "effective preregistration run_id")
    _expect_equal(
        effective.get("jarvis_machine_id"),
        machine_id,
        "effective preregistration machine_id",
    )
    _expect_equal(effective.get("recipe_sha256"), recipe_sha, "effective recipe hash")
    _expect_equal(effective.get("recipe"), recipe, "embedded effective recipe")
    _expect_equal(
        effective.get("created_before_dataset_preparation"),
        True,
        "effective preregistration creation order",
    )
    _expect_equal(
        effective.get("official_test_member_accessed"),
        False,
        "effective official-test declaration",
    )
    effective_input = _mapping(effective.get("input_checkpoint"), "effective input checkpoint")
    _expect_equal(
        effective_input.get("file_sha256"),
        dict(input_hashes),
        "effective input hashes",
    )

    bundle = _load_json(root / "bundle-manifest.json")
    _expect_equal(
        bundle.get("schema_version"),
        "barun-presto-essential-bundle-v1",
        "bundle schema",
    )
    _expect_equal(bundle.get("run_id"), run_id, "bundle run_id")
    _expect_equal(bundle.get("optimizer_state_included"), False, "optimizer exclusion claim")
    _expect_equal(
        bundle.get("sample_level_base_and_post_evidence_included"),
        True,
        "sample evidence inclusion claim",
    )

    result = _load_json(root / "result.json")
    _expect_equal(result.get("schema_version"), "barun-presto-stage-result-v1", "result schema")
    _expect_equal(result.get("run_id"), run_id, "result run_id")
    _expect_equal(result.get("jarvis_machine_id"), machine_id, "result machine_id")
    _expect_equal(result.get("recipe_sha256"), recipe_sha, "result recipe hash")
    _expect_equal(result.get("metric_scope"), METRIC_SCOPE, "result metric scope")
    result_input = _mapping(result.get("input_checkpoint"), "result input checkpoint")
    _expect_equal(result_input.get("file_sha256"), dict(input_hashes), "result input hashes")

    output_hashes = _sha_map(
        result.get("output_checkpoint_sha256"),
        "result output checkpoint hashes",
        exact_names=set(CHECKPOINT_FILES),
    )
    checkpoint_manifest = _load_json(root / "checkpoint" / "checkpoint_manifest.json")
    _expect_equal(
        checkpoint_manifest.get("schema_version"),
        "barun-release-checkpoint-v1",
        "checkpoint schema",
    )
    _expect_equal(checkpoint_manifest.get("run_id"), run_id, "checkpoint run_id")
    _expect_equal(
        checkpoint_manifest.get("input_checkpoint_sha256"),
        dict(input_hashes),
        "checkpoint input lineage",
    )
    _expect_equal(
        checkpoint_manifest.get("file_sha256"),
        output_hashes,
        "checkpoint output hashes",
    )
    for name, expected in output_hashes.items():
        _expect_equal(
            _file_sha256(root / "checkpoint" / name, cache),
            expected,
            f"exported checkpoint hash for {name}",
        )
    _expect_equal(
        output_hashes["barun_config.json"],
        input_hashes["barun_config.json"],
        "post-SFT model config identity",
    )
    _expect_equal(
        output_hashes["tokenizer.json"],
        input_hashes["tokenizer.json"],
        "post-SFT tokenizer identity",
    )

    environment = _load_json(root / "environment.json")
    _expect_equal(environment.get("jarvis_machine_id"), machine_id, "environment machine_id")
    _expect_equal(environment.get("cuda_available"), True, "CUDA execution evidence")
    determinism = _mapping(
        environment.get("determinism_preflight"), "environment determinism preflight"
    )
    _expect_equal(
        determinism.get("configured_before_torch_import"),
        True,
        "cuBLAS pre-import determinism",
    )
    _expect_equal(
        determinism.get("cuda_initialized_before_preflight"),
        False,
        "preflight CUDA initialization",
    )
    return recipe, result, output_hashes


def _audit_firewall_and_data(
    *,
    root: Path,
    cache: dict[Path, str],
    recipe: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[int, int]:
    dataset = _mapping(recipe.get("dataset"), "recipe dataset")
    audit = _load_json(root / "data" / "audit.json")
    _expect_equal(audit.get("schema_version"), "barun-presto-en-audit-v1", "data audit schema")
    source = _mapping(audit.get("source"), "data audit source")
    _expect_equal(source.get("repository"), dataset.get("repository"), "data repository")
    _expect_equal(source.get("revision"), dataset.get("revision"), "data revision")
    archive = _mapping(source.get("archive"), "data source archive")
    _expect_equal(archive.get("actual_sha256"), dataset.get("archive_sha256"), "archive hash")
    _expect_equal(archive.get("expected_sha256"), dataset.get("archive_sha256"), "archive pin")

    official = _mapping(audit.get("official_test"), "official-test audit")
    for key, expected in _FIREWALL_ZERO_ACCESS.items():
        _expect_equal(official.get(key), expected, f"official-test audit {key}")
    official_rows = _integer(official.get("rows"), "opaque official-test row count", minimum=1)
    members = _mapping(source.get("members"), "member-level access evidence")
    sensitive = {
        name: _mapping(evidence, f"source member {name}")
        for name, evidence in members.items()
        if name in {"presto_dataset.jsonl", "presto_test.jsonl"}
        or name.startswith("test_partitions/")
    }
    if {"presto_dataset.jsonl", "presto_test.jsonl"}.difference(sensitive):
        _fail("member firewall evidence omits combined or official-test member")
    sensitive_member_count = _integer(
        official.get("sensitive_member_count"), "sensitive member count", minimum=2
    )
    if sensitive_member_count < len(sensitive):
        _fail("official-test audit undercounts member-level sensitive members")
    for name, evidence in sensitive.items():
        _expect_equal(evidence.get("runtime_open_count"), 0, f"{name} runtime open count")
        _expect_equal(evidence.get("runtime_bytes_read"), 0, f"{name} runtime bytes read")
        _expect_equal(
            evidence.get("verification"),
            "whole_archive_sha256_and_central_directory_only",
            f"{name} verification scope",
        )

    result_data = _mapping(result.get("data"), "result data")
    _expect_equal(result_data.get("revision"), dataset.get("revision"), "result data revision")
    _expect_equal(
        result_data.get("official_test_rows_opaque_unparsed"),
        official_rows,
        "result opaque official-test rows",
    )
    result_firewall = _mapping(
        result_data.get("official_test_firewall"), "result official-test firewall"
    )
    for key, expected in _FIREWALL_ZERO_ACCESS.items():
        _expect_equal(result_firewall.get(key), expected, f"result firewall {key}")
    _expect_equal(
        result_firewall.get("official_test_rows_opaque_unparsed"),
        official_rows,
        "result firewall opaque row count",
    )

    prepared = _sha_map(
        result_data.get("prepared_hashes"),
        "prepared hashes",
        exact_names={"train", "dev", "audit"},
    )
    expected_prepared = {
        "train": dataset.get("train_manifest_sha256"),
        "dev": dataset.get("dev_manifest_sha256"),
        "audit": dataset.get("audit_sha256"),
    }
    _expect_equal(prepared, expected_prepared, "prepared dataset hashes")
    _expect_equal(
        _file_sha256(root / "data" / "audit.json", cache),
        prepared["audit"],
        "pinned data audit hash",
    )
    artifacts = _mapping(audit.get("artifacts"), "audit artifacts")
    manifests = _mapping(artifacts.get("manifests"), "audit manifest identities")
    for split in ("train", "dev"):
        evidence = _mapping(manifests.get(split), f"audit {split} manifest")
        _expect_equal(evidence.get("sha256"), prepared[split], f"audit {split} hash")

    counts = _mapping(audit.get("counts"), "data audit counts")
    english = _mapping(counts.get("english_selected"), "English selected counts")
    train_rows = _integer(result_data.get("train_rows"), "result train rows", minimum=1)
    dev_rows = _integer(result_data.get("dev_rows"), "result dev rows", minimum=1)
    _expect_equal(english.get("train"), train_rows, "audited English train rows")
    _expect_equal(english.get("dev"), dev_rows, "audited English dev rows")
    _expect_equal(counts.get("records_dropped"), 0, "adapter dropped rows")
    _expect_equal(counts.get("records_truncated"), 0, "adapter truncated rows")

    focus = _load_json(root / "data" / "train-focus-audit.json")
    _expect_equal(
        focus.get("schema_version"),
        "barun-presto-focus-replay-audit-v1",
        "focus audit schema",
    )
    _expect_equal(focus.get("source_sha256"), prepared["train"], "focus source hash")
    _expect_equal(focus.get("source_rows"), train_rows, "focus source row count")
    focus_rows = _integer(result_data.get("focused_train_rows"), "focused train rows", minimum=1)
    _expect_equal(focus.get("output_rows"), focus_rows, "focus output row count")
    _expect_equal(
        focus.get("output_sha256"),
        result_data.get("focus_view_sha256"),
        "focus output hash",
    )
    train_view = _mapping(recipe.get("train_view"), "recipe train view")
    _expect_equal(
        focus.get("selector_version"), train_view.get("selector_version"), "focus selector"
    )
    _expect_equal(
        focus.get("maximum_replays_per_category"),
        train_view.get("maximum_replays_per_category"),
        "focus replay limit",
    )
    _expect_equal(
        focus.get("development_rows_read_for_exact_duplicate_exclusion"),
        train_view.get("development_rows_read_for_exact_duplicate_exclusion"),
        "focus development exclusion reads",
    )
    _expect_equal(
        focus.get("development_rows_read_for_focus_selection"),
        0,
        "focus selection development reads",
    )
    _expect_equal(focus.get("official_test_rows_read"), 0, "focus official-test reads")
    rows_after = _integer(
        focus.get("rows_after_exact_dev_exclusion"), "post-exclusion rows", minimum=1
    )
    excluded = _integer(focus.get("exact_prompt_target_train_rows_excluded"), "excluded train rows")
    _expect_equal(rows_after + excluded, train_rows, "focus exclusion accounting")
    replay_counts = _mapping(focus.get("replay_counts"), "focus replay counts")
    replay_total = sum(
        _integer(value, f"focus replay count {name}") for name, value in replay_counts.items()
    )
    _expect_equal(rows_after + replay_total, focus_rows, "focus output accounting")
    return train_rows, dev_rows


def _audit_sample(
    sample: Mapping[str, Any], prediction: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if set(sample) != _SAMPLE_FIELDS:
        _fail(
            f"{label} sample fields changed: expected {sorted(_SAMPLE_FIELDS)}, "
            f"got {sorted(sample)}"
        )
    if set(prediction) != _PREDICTION_FIELDS:
        _fail(
            f"{label} prediction fields changed: expected {sorted(_PREDICTION_FIELDS)}, "
            f"got {sorted(prediction)}"
        )
    sample_id = _string(sample.get("sample_id"), f"{label} sample_id")
    _expect_equal(prediction.get("id"), sample_id, f"{label} aligned prediction ID")
    _expect_equal(
        sample.get("schema_version"),
        PRESTO_SCORER_VERSION_V1,
        f"{label} scorer version",
    )
    for field in ("prediction_raw", "generation_failure", "truncated"):
        _expect_equal(sample.get(field), prediction.get(field), f"{label} aligned {field}")
    _integer(prediction.get("prompt_tokens"), f"{label} prompt tokens")
    _integer(prediction.get("generated_tokens"), f"{label} generated tokens")
    truncated = _boolean(sample.get("truncated"), f"{label} truncated")
    generation_failure = sample.get("generation_failure")
    if generation_failure is not None:
        _string(generation_failure, f"{label} generation failure")
    prediction_raw = sample.get("prediction_raw")
    if prediction_raw is not None and not isinstance(prediction_raw, str):
        _fail(f"{label} prediction_raw must be a string or null")
    if prediction_raw is not None and generation_failure is not None:
        _fail(f"{label} cannot have prediction text and a generation failure")

    gold_payload = _mapping(sample.get("gold"), f"{label} gold")
    try:
        gold = parse_presto_action(
            json.dumps(
                gold_payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    except (ActionIRError, PrestoScoreError) as error:
        raise PrestoBundleAuditError(f"{label} contains invalid gold: {error}") from error
    gold_decision = gold.decision.value
    _expect_equal(sample.get("gold_decision"), gold_decision, f"{label} gold decision")

    parse_valid = False
    schema_valid = False
    attempted_call = False
    parsed = None
    error_code: str | None = None
    if prediction_raw is None:
        error_code = generation_failure or "missing_prediction"
    else:
        try:
            decoded = decode_json_object(prediction_raw)
            parse_valid = True
            # Deliberately count a parse-valid top-level CALL before schema validation.
            attempted_call = decoded.get("decision") == "CALL"
            parsed = parse_presto_action(prediction_raw)
            schema_valid = True
        except (ActionIRError, PrestoScoreError) as error:
            error_code = error.code

    predicted_decision = parsed.decision.value if parsed is not None else None
    exact = bool(
        schema_valid
        and parsed is not None
        and not truncated
        and generation_failure is None
        and action_ir_equal(gold, parsed)
    )
    decision_correct = predicted_decision == gold_decision
    false_call = bool(gold_decision in {"ABSTAIN", "CONFIRM"} and attempted_call)
    checks = {
        "parse_valid": parse_valid,
        "schema_valid": schema_valid,
        "prediction_error_code": error_code,
        "predicted_decision": predicted_decision,
        "prediction": parsed.to_dict() if parsed is not None else None,
        "ast_exact": exact,
        "decision_correct": decision_correct,
        "false_call_on_gate": false_call,
    }
    for field, expected in checks.items():
        _expect_equal(sample.get(field), expected, f"{label} recomputed {field}")
    phenomenon = _string(sample.get("phenomenon"), f"{label} phenomenon", nonempty=False)
    _expect_equal(
        sample.get("phenomenon_group"),
        _original_phenomenon_group(phenomenon),
        f"{label} phenomenon group",
    )
    _string(sample.get("root_intent"), f"{label} root intent")
    _boolean(sample.get("contextual"), f"{label} contextual")
    return {
        "sample_id": sample_id,
        "gold_decision": gold_decision,
        "predicted_decision": predicted_decision,
        "parse_valid": parse_valid,
        "schema_valid": schema_valid,
        "ast_exact": exact,
        "decision_correct": decision_correct,
        "false_call_on_gate": false_call,
        "phenomenon_group": sample["phenomenon_group"],
        "prediction_error_code": error_code,
        "truncated": truncated,
        "generation_failed": generation_failure is not None,
        "ground_truth_identity": {
            key: sample[key]
            for key in (
                "sample_id",
                "phenomenon",
                "phenomenon_group",
                "root_intent",
                "contextual",
                "gold",
                "gold_decision",
            )
        },
    }


def _rate(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else 0.0,
    }


def _abstention(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    true_positive = sum(
        row["gold_decision"] == "ABSTAIN" and row["predicted_decision"] == "ABSTAIN" for row in rows
    )
    false_positive = sum(
        row["gold_decision"] != "ABSTAIN" and row["predicted_decision"] == "ABSTAIN" for row in rows
    )
    false_negative = sum(
        row["gold_decision"] == "ABSTAIN" and row["predicted_decision"] != "ABSTAIN" for row in rows
    )
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _bucket(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "count": count,
        "ast_exact_match": _rate(sum(bool(row["ast_exact"]) for row in rows), count),
        "schema_valid": _rate(sum(bool(row["schema_valid"]) for row in rows), count),
        "decision_accuracy": _rate(sum(bool(row["decision_correct"]) for row in rows), count),
        "truncation": _rate(sum(bool(row["truncated"]) for row in rows), count),
        "generation_failure": _rate(sum(bool(row["generation_failed"]) for row in rows), count),
    }


def _recomputed_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    gate_rows = [row for row in rows if row["gold_decision"] in {"ABSTAIN", "CONFIRM"}]
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["phenomenon_group"])].append(row)
    return {
        "sample_count": count,
        "ast_exact_match": _rate(sum(bool(row["ast_exact"]) for row in rows), count),
        "parse_valid": _rate(sum(bool(row["parse_valid"]) for row in rows), count),
        "schema_valid": _rate(sum(bool(row["schema_valid"]) for row in rows), count),
        "decision_accuracy": _rate(sum(bool(row["decision_correct"]) for row in rows), count),
        "abstention": _abstention(rows),
        "abstention_gold_count": sum(row["gold_decision"] == "ABSTAIN" for row in rows),
        "false_call_on_gate": _rate(
            sum(bool(row["false_call_on_gate"]) for row in gate_rows), len(gate_rows)
        ),
        "headline_phenomenon_buckets": {
            name: _bucket(bucket) for name, bucket in sorted(by_group.items())
        },
        "failure_counts": dict(
            sorted(
                Counter(
                    str(row["prediction_error_code"])
                    for row in rows
                    if row["prediction_error_code"] is not None
                ).items()
            )
        ),
    }


def _compare_rate(recorded: Any, recomputed: Mapping[str, Any], label: str) -> None:
    value = _mapping(recorded, label)
    _expect_equal(value.get("numerator"), recomputed["numerator"], f"{label} numerator")
    _expect_equal(value.get("denominator"), recomputed["denominator"], f"{label} denominator")
    _expect_close(value.get("value"), float(recomputed["value"]), f"{label} value")


def _compare_metrics(
    aggregate: Mapping[str, Any], recomputed: Mapping[str, Any], *, label: str
) -> None:
    _expect_equal(
        aggregate.get("schema_version"),
        PRESTO_SCORER_VERSION_V1,
        f"{label} scorer",
    )
    _expect_equal(aggregate.get("metric_scope"), METRIC_SCOPE, f"{label} metric scope")
    _expect_equal(aggregate.get("sample_count"), recomputed["sample_count"], f"{label} count")
    for name in ("ast_exact_match", "parse_valid", "schema_valid", "decision_accuracy"):
        _compare_rate(aggregate.get(name), recomputed[name], f"{label} {name}")
    recorded_abstention = _mapping(aggregate.get("abstention"), f"{label} abstention")
    recomputed_abstention = _mapping(recomputed["abstention"], "recomputed abstention")
    for name, expected in recomputed_abstention.items():
        if isinstance(expected, float):
            _expect_close(recorded_abstention.get(name), expected, f"{label} abstention {name}")
        else:
            _expect_equal(recorded_abstention.get(name), expected, f"{label} abstention {name}")
    _expect_equal(
        aggregate.get("abstention_gold_count"),
        recomputed["abstention_gold_count"],
        f"{label} abstention gold count",
    )
    _compare_rate(
        aggregate.get("false_call_on_gate"),
        _mapping(recomputed["false_call_on_gate"], "recomputed false call"),
        f"{label} false_call_on_gate",
    )
    recorded_buckets = _mapping(
        aggregate.get("headline_phenomenon_buckets"), f"{label} phenomenon buckets"
    )
    recomputed_buckets = _mapping(
        recomputed["headline_phenomenon_buckets"], "recomputed phenomenon buckets"
    )
    _expect_equal(set(recorded_buckets), set(recomputed_buckets), f"{label} bucket names")
    for name, expected_raw in recomputed_buckets.items():
        expected = _mapping(expected_raw, f"recomputed bucket {name}")
        recorded = _mapping(recorded_buckets[name], f"{label} bucket {name}")
        _expect_equal(recorded.get("count"), expected["count"], f"{label} {name} count")
        for metric in (
            "ast_exact_match",
            "schema_valid",
            "decision_accuracy",
            "truncation",
            "generation_failure",
        ):
            _compare_rate(
                recorded.get(metric),
                _mapping(expected[metric], f"recomputed {name} {metric}"),
                f"{label} {name} {metric}",
            )
    _expect_equal(
        aggregate.get("failure_counts"),
        recomputed["failure_counts"],
        f"{label} failures",
    )


def _audit_generation_manifest(
    *,
    root: Path,
    cache: dict[Path, str],
    stage: str,
    prediction_rows: Sequence[Mapping[str, Any]],
    result_generation: Any,
    dev_sha256: str,
    expected_checkpoint_hashes: Mapping[str, str],
    generation_limit: int,
) -> None:
    stage_root = root / f"{stage}-eval"
    manifest = _load_json(stage_root / "predictions.jsonl.manifest.json")
    _expect_equal(manifest, result_generation, f"{stage} generation result")
    _expect_equal(
        manifest.get("schema_version"),
        "barun-greedy-generation-v1",
        f"{stage} generation schema",
    )
    _expect_equal(manifest.get("manifest_sha256"), dev_sha256, f"{stage} development hash")
    _expect_equal(manifest.get("max_new_tokens"), generation_limit, f"{stage} generation limit")
    _expect_equal(manifest.get("device"), "cuda", f"{stage} generation device")
    _expect_equal(manifest.get("dtype"), "torch.bfloat16", f"{stage} generation dtype")
    _expect_equal(manifest.get("examples"), len(prediction_rows), f"{stage} generation examples")
    failed = sum(row.get("generation_failure") is not None for row in prediction_rows)
    truncated = sum(row.get("truncated") is True for row in prediction_rows)
    _expect_equal(manifest.get("failed"), failed, f"{stage} failed generations")
    _expect_equal(
        manifest.get("generated"),
        len(prediction_rows) - failed,
        f"{stage} generated count",
    )
    _expect_equal(manifest.get("truncated"), truncated, f"{stage} truncated count")
    _expect_equal(
        manifest.get("predictions_sha256"),
        _file_sha256(stage_root / "predictions.jsonl", cache),
        f"{stage} prediction hash",
    )
    checkpoint_hashes = _sha_map(manifest.get("checkpoint_sha256"), f"{stage} checkpoint hashes")
    for name, expected in expected_checkpoint_hashes.items():
        _expect_equal(checkpoint_hashes.get(name), expected, f"{stage} checkpoint {name}")


def _audit_evaluations(
    *,
    root: Path,
    cache: dict[Path, str],
    result: Mapping[str, Any],
    recipe: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    output_hashes: Mapping[str, str],
    dev_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluated: dict[
        str,
        tuple[list[Mapping[str, Any]], dict[str, Any], list[Mapping[str, Any]]],
    ] = {}
    for stage in ("base", "post"):
        stage_root = root / f"{stage}-eval"
        predictions = _load_jsonl(stage_root / "predictions.jsonl")
        samples = _load_jsonl(stage_root / "scores" / "sample_scores.jsonl")
        if len(predictions) != dev_rows or len(samples) != dev_rows:
            _fail(
                f"{stage} sample count mismatch: predictions={len(predictions)}, "
                f"scores={len(samples)}, expected={dev_rows}"
            )
        seen: set[str] = set()
        rows: list[Mapping[str, Any]] = []
        for index, (sample, prediction) in enumerate(zip(samples, predictions, strict=True)):
            audited = _audit_sample(sample, prediction, label=f"{stage}[{index}]")
            sample_id = str(audited["sample_id"])
            if sample_id in seen:
                _fail(f"{stage} contains duplicate sample ID {sample_id!r}")
            seen.add(sample_id)
            rows.append(audited)
        recomputed = _recomputed_metrics(rows)
        aggregate = _load_json(stage_root / "scores" / "aggregate.json")
        _compare_metrics(aggregate, recomputed, label=stage)
        result_key = "base" if stage == "base" else "post_sft"
        _expect_equal(result.get(result_key), aggregate, f"result {result_key} aggregate")
        evaluated[stage] = (rows, recomputed, predictions)

    base_rows, base_metrics, base_predictions = evaluated["base"]
    post_rows, post_metrics, post_predictions = evaluated["post"]
    _expect_equal(
        [row["ground_truth_identity"] for row in post_rows],
        [row["ground_truth_identity"] for row in base_rows],
        "base/post sample alignment and ground truth",
    )

    dataset = _mapping(recipe.get("dataset"), "recipe dataset")
    dev_sha = _sha256(dataset.get("dev_manifest_sha256"), "recipe dev hash")
    generation = _mapping(result.get("generation"), "result generation")
    generation_limit = _integer(
        result.get("generation_limit"), "result generation limit", minimum=1
    )
    _audit_generation_manifest(
        root=root,
        cache=cache,
        stage="base",
        prediction_rows=base_predictions,
        result_generation=generation.get("base"),
        dev_sha256=dev_sha,
        expected_checkpoint_hashes=input_hashes,
        generation_limit=generation_limit,
    )
    _audit_generation_manifest(
        root=root,
        cache=cache,
        stage="post",
        prediction_rows=post_predictions,
        result_generation=generation.get("post_sft"),
        dev_sha256=dev_sha,
        expected_checkpoint_hashes=output_hashes,
        generation_limit=generation_limit,
    )
    delta = _mapping(result.get("delta"), "result metric delta")
    expected_delta = {
        "ast_exact_match": post_metrics["ast_exact_match"]["value"]
        - base_metrics["ast_exact_match"]["value"],
        "schema_valid": post_metrics["schema_valid"]["value"]
        - base_metrics["schema_valid"]["value"],
        "abstention_f1": post_metrics["abstention"]["f1"] - base_metrics["abstention"]["f1"],
    }
    for name, expected in expected_delta.items():
        _expect_close(delta.get(name), float(expected), f"result delta {name}")
    return base_metrics, post_metrics


def _recompute_gate(post: Mapping[str, Any], recipe: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = _mapping(recipe.get("gate"), "recipe gate")
    exact = float(_mapping(post["ast_exact_match"], "post exact")["value"])
    schema = float(_mapping(post["schema_valid"], "post schema")["value"])
    abstention = float(_mapping(post["abstention"], "post abstention")["f1"])
    false_call = float(_mapping(post["false_call_on_gate"], "post false call")["value"])
    buckets = _mapping(post["headline_phenomenon_buckets"], "post buckets")
    simple = buckets.get("no_phenomenon")
    represented = (
        isinstance(simple, Mapping) and _integer(simple.get("count"), "simple bucket count") > 0
    )
    simple_value = (
        float(
            _mapping(_mapping(simple, "simple bucket")["ast_exact_match"], "simple exact")["value"]
        )
        if represented
        else None
    )
    gaps: dict[str, float | None] = {}
    for name in ("revision", "disfluency"):
        bucket = buckets.get(name)
        if (
            not isinstance(bucket, Mapping)
            or _integer(bucket.get("count"), f"{name} count") < 1
            or simple_value is None
        ):
            gaps[name] = None
            represented = False
        else:
            gaps[name] = simple_value - float(
                _mapping(bucket["ast_exact_match"], f"{name} exact")["value"]
            )
    maximum_gap = _number(
        thresholds.get("maximum_no_phenomenon_to_revision_or_disfluency_gap"),
        "maximum hard-bucket gap",
    )
    checks = {
        "derived_ast_exact_at_least_threshold": exact
        >= _number(thresholds.get("derived_ast_exact_at_least"), "exact threshold"),
        "schema_valid_at_least_threshold": schema
        >= _number(thresholds.get("schema_valid_at_least"), "schema threshold"),
        "abstention_f1_at_least_threshold": abstention
        >= _number(thresholds.get("abstention_f1_at_least"), "abstention threshold"),
        "false_call_on_gate_at_most_threshold": false_call
        <= _number(thresholds.get("false_call_on_gate_at_most"), "false-call threshold"),
        "gap_buckets_represented": represented,
        "revision_disfluency_gap_at_most_threshold": represented
        and all(gap is not None and gap <= maximum_gap for gap in gaps.values()),
    }
    return {
        "thresholds": dict(thresholds),
        "observed": {
            "derived_ast_exact": exact,
            "schema_valid": schema,
            "abstention_f1": abstention,
            "false_call_on_gate": false_call,
            "no_phenomenon_minus_hard_bucket_gap": gaps,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _compare_gate(recorded: Any, recomputed: Mapping[str, Any]) -> None:
    gate = _mapping(recorded, "recorded result gate")
    _expect_equal(gate.get("thresholds"), recomputed["thresholds"], "gate thresholds")
    _expect_equal(gate.get("checks"), recomputed["checks"], "gate checks")
    _expect_equal(gate.get("passed"), recomputed["passed"], "gate decision")
    observed = _mapping(gate.get("observed"), "recorded gate observations")
    expected_observed = _mapping(recomputed["observed"], "recomputed gate observations")
    for name in (
        "derived_ast_exact",
        "schema_valid",
        "abstention_f1",
        "false_call_on_gate",
    ):
        _expect_close(observed.get(name), float(expected_observed[name]), f"gate observed {name}")
    recorded_gaps = _mapping(
        observed.get("no_phenomenon_minus_hard_bucket_gap"),
        "recorded hard-bucket gaps",
    )
    expected_gaps = _mapping(
        expected_observed["no_phenomenon_minus_hard_bucket_gap"],
        "recomputed hard-bucket gaps",
    )
    _expect_equal(set(recorded_gaps), set(expected_gaps), "gate hard-bucket gap names")
    for name, expected in expected_gaps.items():
        if expected is None:
            _expect_equal(recorded_gaps.get(name), None, f"gate {name} gap")
        else:
            _expect_close(recorded_gaps.get(name), float(expected), f"gate {name} gap")


def _audit_lineage(
    *,
    root: Path,
    cache: dict[Path, str],
    result: Mapping[str, Any],
    recipe: Mapping[str, Any],
    run_id: str,
    machine_id: int,
    input_hashes: Mapping[str, str],
    output_hashes: Mapping[str, str],
    train_rows: int,
    dev_rows: int,
) -> None:
    training_config_path = root / "training-config.json"
    config = _load_json(training_config_path)
    _expect_equal(config.get("schema_version"), "barun-sft-config-v1", "training config schema")
    _expect_equal(config.get("run_id"), run_id, "training config run_id")
    _expect_equal(config.get("optimization"), recipe.get("optimization"), "training recipe")
    base_checkpoint = _mapping(config.get("base_checkpoint"), "training base checkpoint")
    _expect_equal(base_checkpoint.get("source"), "local", "training checkpoint source")
    _expect_equal(
        base_checkpoint.get("expected_sha256"),
        dict(input_hashes),
        "training input hashes",
    )
    config_data = _mapping(config.get("data"), "training config data")
    result_data = _mapping(result.get("data"), "result data")
    _expect_equal(
        config_data.get("train_sha256"),
        result_data.get("focus_view_sha256"),
        "training focus hash",
    )
    dataset = _mapping(recipe.get("dataset"), "recipe dataset")
    _expect_equal(
        config_data.get("dev_sha256"),
        dataset.get("dev_manifest_sha256"),
        "training dev hash",
    )
    execution = _mapping(config.get("execution"), "training execution")
    _expect_equal(execution.get("device"), "cuda", "training device")
    _expect_equal(execution.get("precision"), "bf16", "training precision")
    _expect_equal(execution.get("deterministic"), True, "training determinism")
    _expect_equal(execution.get("jarvis_resource_id"), str(machine_id), "training machine_id")

    manifest = _load_json(root / "training" / "run_manifest.json")
    _expect_equal(manifest.get("schema_version"), "barun-sft-run-v1", "trainer manifest schema")
    _expect_equal(manifest.get("run_id"), run_id, "trainer run_id")
    _expect_equal(
        manifest.get("config_source_sha256"),
        _file_sha256(training_config_path, cache),
        "trainer config source hash",
    )
    manifest_execution = _mapping(manifest.get("execution"), "trainer execution")
    _expect_equal(
        manifest_execution.get("jarvis_resource_id"),
        str(machine_id),
        "trainer machine_id",
    )
    _expect_equal(manifest_execution.get("deterministic"), True, "trainer determinism")
    trainer_base = _mapping(manifest.get("base_checkpoint"), "trainer base checkpoint")
    _expect_equal(trainer_base.get("file_sha256"), dict(input_hashes), "trainer input lineage")
    parameter_counts = _mapping(trainer_base.get("parameter_counts"), "trainer parameter counts")
    _expect_equal(parameter_counts.get("total"), 35072768, "BarunLM-35M parameter count")
    trainer_data = _mapping(manifest.get("data"), "trainer data")
    _expect_equal(
        trainer_data.get("train_manifest_sha256"),
        result_data.get("focus_view_sha256"),
        "trainer focus hash",
    )
    _expect_equal(
        trainer_data.get("dev_manifest_sha256"),
        dataset.get("dev_manifest_sha256"),
        "trainer dev hash",
    )
    train_summary = _mapping(trainer_data.get("train"), "trainer train summary")
    dev_summary = _mapping(trainer_data.get("dev"), "trainer dev summary")
    _expect_equal(
        train_summary.get("accepted_examples"),
        result_data.get("focused_train_rows"),
        "trainer train count",
    )
    _expect_equal(dev_summary.get("accepted_examples"), dev_rows, "trainer dev count")
    _expect_equal(train_summary.get("rejected_examples"), 0, "trainer rejected train rows")
    _expect_equal(dev_summary.get("rejected_examples"), 0, "trainer rejected dev rows")
    _expect_equal(result_data.get("train_rows"), train_rows, "result train count stability")

    summary = _load_json(root / "training" / "summary.json")
    result_training = _mapping(result.get("training"), "result training summary")
    _expect_equal(summary, result_training, "trainer/result summary")
    _expect_equal(summary.get("run_id"), run_id, "training summary run_id")
    _integer(summary.get("global_steps"), "training global steps", minimum=1)
    final_checkpoint = _string(summary.get("final_checkpoint"), "final checkpoint path")
    best_checkpoint = summary.get("best_checkpoint")
    if best_checkpoint is not None:
        best_checkpoint = _string(best_checkpoint, "best checkpoint path")
    selected_checkpoint = best_checkpoint or final_checkpoint
    checkpoint_manifest = _load_json(root / "checkpoint" / "checkpoint_manifest.json")
    _expect_equal(
        checkpoint_manifest.get("source_checkpoint"),
        selected_checkpoint,
        "selected checkpoint lineage",
    )
    result_generation = _mapping(result.get("generation"), "result generation")
    post_generation = _mapping(result_generation.get("post_sft"), "post generation")
    _expect_equal(
        post_generation.get("checkpoint_dir"),
        selected_checkpoint,
        "post checkpoint path",
    )
    result_input = _mapping(result.get("input_checkpoint"), "result input checkpoint")
    base_generation = _mapping(result_generation.get("base"), "base generation")
    _expect_equal(
        base_generation.get("checkpoint_dir"),
        result_input.get("directory"),
        "base checkpoint path",
    )
    _expect_equal(
        base_checkpoint.get("local_dir"),
        result_input.get("directory"),
        "input path lineage",
    )

    best_path = root / "training" / "best_checkpoint.json"
    if best_checkpoint is None and best_path.exists():
        _fail("best_checkpoint.json exists but trainer selected no best checkpoint")
    if best_checkpoint is not None:
        if not best_path.is_file():
            _fail("trainer selected a best checkpoint but its evidence file is absent")
        best = _load_json(best_path)
        _expect_equal(best.get("checkpoint"), best_checkpoint, "best checkpoint evidence")
        _integer(best.get("global_step"), "best checkpoint global step", minimum=1)

    post_checkpoint_hashes = _sha_map(
        post_generation.get("checkpoint_sha256"),
        "post generation checkpoint hashes",
    )
    for name, digest in output_hashes.items():
        _expect_equal(post_checkpoint_hashes.get(name), digest, f"selected output {name}")

    metrics = _load_jsonl(root / "training" / "metrics.jsonl")
    complete = [row for row in metrics if row.get("event") == "complete"]
    if len(complete) != 1:
        _fail("trainer metrics must contain exactly one complete event")
    for key, value in summary.items():
        _expect_equal(complete[0].get(key), value, f"trainer complete event {key}")


def _audit_progress(root: Path, run_id: str) -> None:
    progress = _load_jsonl(root / "progress.jsonl")
    required = (
        "run_root_claimed",
        "determinism_preflight_passed",
        "effective_preregistration_frozen",
        "repository_tests_passed",
        "input_checkpoint_verified",
        "presto_data_prepared_and_firewall_verified",
        "train_only_focus_view_materialized",
        "input_checkpoint_dev_scored",
        "full_parameter_response_only_sft_completed",
        "post_sft_dev_scored",
        "essential_bundle_completed",
        "run_completed",
    )
    phases = [row.get("phase") for row in progress]
    positions: list[int] = []
    for phase in required:
        matches = [index for index, value in enumerate(phases) if value == phase]
        if len(matches) != 1:
            _fail(f"progress evidence must contain phase {phase!r} exactly once")
        positions.append(matches[0])
    if positions != sorted(positions):
        _fail("PRESTO progress phases are out of order")
    root_claim = _mapping(progress[positions[0]], "run-root progress event")
    if not _path_has_suffix(_string(root_claim.get("run_root"), "claimed run root"), run_id):
        _fail("claimed remote run root does not end in the frozen run_id")


def _audit_optimizer_exclusion(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if re.search(
            r"(^|[._-])(optimizer|optim)([._-]|$)",
            path.name.casefold(),
        ):
            _fail(f"optimizer state file entered the essential bundle: {path.relative_to(root)}")


def validate_presto_bundle(
    bundle_root: str | Path,
    *,
    preregistration_path: str | Path,
    jarvis_record_path: str | Path,
) -> PrestoBundleAuditResult:
    """Validate a completed PRESTO bundle against two local trust anchors.

    A successful audit means the evidence is valid and the gate was faithfully
    recomputed. It does not mean the research gate itself passed.
    """

    root = Path(bundle_root).resolve(strict=True)
    prereg_path = Path(preregistration_path).resolve(strict=True)
    jarvis_path = Path(jarvis_record_path).resolve(strict=True)
    if root in prereg_path.parents or root in jarvis_path.parents:
        _fail("trust-anchor files must be outside the downloaded essential bundle")
    prereg = _load_json(prereg_path)
    jarvis = _load_json(jarvis_path)
    run_id, machine_id, managed_run_id, recipe_sha, input_hashes = _audit_external_anchors(
        prereg, jarvis, root
    )

    cache: dict[Path, str] = {}
    artifact_manifest, artifact_manifest_sha = _verify_artifact_manifest(root, cache)
    _audit_optimizer_exclusion(root)
    recipe, result, output_hashes = _audit_recipe_and_identities(
        root=root,
        cache=cache,
        run_id=run_id,
        machine_id=machine_id,
        recipe_sha=recipe_sha,
        input_hashes=input_hashes,
    )
    train_rows, dev_rows = _audit_firewall_and_data(
        root=root, cache=cache, recipe=recipe, result=result
    )
    base, post = _audit_evaluations(
        root=root,
        cache=cache,
        result=result,
        recipe=recipe,
        input_hashes=input_hashes,
        output_hashes=output_hashes,
        dev_rows=dev_rows,
    )
    gate = _recompute_gate(post, recipe)
    _compare_gate(result.get("gate"), gate)
    _audit_lineage(
        root=root,
        cache=cache,
        result=result,
        recipe=recipe,
        run_id=run_id,
        machine_id=machine_id,
        input_hashes=input_hashes,
        output_hashes=output_hashes,
        train_rows=train_rows,
        dev_rows=dev_rows,
    )
    _audit_progress(root, run_id)
    return PrestoBundleAuditResult(
        schema_version="barun-presto-independent-bundle-audit-v1",
        passed=True,
        bundle_root=str(root),
        artifact_count=len(artifact_manifest),
        artifact_manifest_sha256=artifact_manifest_sha,
        run_id=run_id,
        jarvis_machine_id=machine_id,
        jarvis_managed_run_id=managed_run_id,
        jarvis_final_status="Paused",
        recipe_sha256=recipe_sha,
        input_checkpoint_sha256=dict(input_hashes),
        output_checkpoint_sha256=output_hashes,
        development_sample_count=dev_rows,
        recomputed_base=base,
        recomputed_post_sft=post,
        recomputed_gate=gate,
    )


__all__ = [
    "PrestoBundleAuditError",
    "PrestoBundleAuditResult",
    "validate_presto_bundle",
]
