"""Validate and optionally upload the candidate-v2 release bundle to Weights & Biases.

Dry-run manifest creation is the default and never imports ``wandb``.  Online upload requires the
explicit ``--upload`` flag, a pinned W&B installation, and an external ``--wandb-dir``.  The local
manifest remains the authoritative record of every byte offered to W&B.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

SPEC_SCHEMA_VERSION = "barunaction-wandb-release-spec-v1"
MANIFEST_SCHEMA_VERSION = "barunaction-wandb-release-manifest-v1"
RECEIPT_SCHEMA_VERSION = "barunaction-wandb-release-receipt-v1"
WANDB_VERSION = "0.28.1"
WANDB_PROJECT = "barunaction-35m"
CANDIDATE_ID = "candidate-v2"
INT8_RETENTION_PROTOCOL_SHA256 = "7229572bea461a7899b09c0310a77b9b4d8af74211d1b9ac8b2b5f77dc6b67cb"
INT8_RETENTION_FILES = {
    "artifact-sha256.json",
    "paired-samples.jsonl",
    "protocol.json",
    "result.json",
}
INT8_RETENTION_ARTIFACT_SCHEMA = "barunaction-arm64-int8-retention-artifacts-v1"
INT8_RETENTION_RESULT_SCHEMA = "barunaction-arm64-int8-retention-result-v1"
INT8_RETENTION_PAIRED_SCHEMA = "barunaction-arm64-int8-retention-paired-v1"

ARTIFACT_CONTRACTS = {
    "float": {
        "name": "barunaction-35m-candidate-v2-float",
        "type": "model",
        "required_files": {
            "LICENSE",
            "MODEL_CARD.md",
            "NOTICE",
            "barun_config.json",
            "checkpoint_manifest.json",
            "model.safetensors",
            "tokenizer.json",
        },
    },
    "int8_darwin_arm64_qnnpack": {
        "name": "barunaction-35m-candidate-v2-int8-darwin-arm64-qnnpack",
        "type": "model",
        "required_files": {
            "LICENSE",
            "MODEL_CARD.md",
            "NOTICE",
            "barun_config.json",
            "model.int8.pt",
            "quantization_manifest.json",
            "tokenizer.json",
        },
    },
    "evidence": {
        "name": "barunaction-35m-candidate-v2-evidence",
        "type": "evaluation",
        "required_files": None,
    },
}

SOURCE_RUN_NAMES = frozenset(
    {
        "continual_recovery",
        "interpolation_rescue",
        "matched_baseline",
        "matched_qwen_baseline",
        "mobile_candidate_v2",
        "mobile_int8_retention",
        "parallel_rescue_selection",
        "presto_mobile_regression",
        "presto_stage",
    }
)
ALLOWED_TAGS = frozenset(
    {
        "barunaction-35m",
        "candidate-v2",
        "darwin-arm64",
        "post-training",
        "qnnpack",
        "release-evidence",
    }
)

SUMMARY_METRIC_NAMES = frozenset(
    {
        "mobile_candidate_v2/ast_exact_match",
        "mobile_candidate_v2/schema_valid",
        "mobile_int8_retention/ast_exact_match",
        "mobile_int8_retention/correct_rows_lost",
        "mobile_int8_retention/gate_passed",
        "mobile_int8_retention/source_float_ast_exact_match",
        "presto_stage/ast_exact_match",
        "presto_stage/schema_valid",
        "presto_stage/abstention_f1",
        "presto_stage/false_call_on_gate",
        "presto_stage/gate_passed",
        "matched_baseline/ast_exact_match",
        "matched_baseline/schema_valid",
        "matched_baseline/parameters",
        "matched_baseline/gate_passed",
        "matched_qwen_baseline/barunaction_ast_exact_match",
        "matched_qwen_baseline/parameters",
        "matched_qwen_baseline/qwen_ast_exact_match",
        "matched_qwen_baseline/qwen_outperformed_barunaction",
        "matched_qwen_baseline/qwen_parse_valid",
        "matched_qwen_baseline/qwen_schema_valid",
        "presto_mobile_regression/ast_exact_match",
        "presto_mobile_regression/schema_valid",
        "presto_mobile_regression/gate_passed",
        "continual_recovery/mobile_ast_exact_match",
        "continual_recovery/presto_ast_exact_match",
        "continual_recovery/presto_schema_valid",
        "continual_recovery/presto_abstention_f1",
        "continual_recovery/presto_false_call_on_gate",
        "continual_recovery/gate_passed",
        "interpolation_rescue/alpha_025/mobile_ast_exact_match",
        "interpolation_rescue/alpha_025/presto_ast_exact_match",
        "interpolation_rescue/alpha_025/eligible",
        "interpolation_rescue/alpha_050/mobile_ast_exact_match",
        "interpolation_rescue/alpha_050/presto_ast_exact_match",
        "interpolation_rescue/alpha_050/eligible",
        "interpolation_rescue/alpha_075/mobile_ast_exact_match",
        "interpolation_rescue/alpha_075/presto_ast_exact_match",
        "interpolation_rescue/alpha_075/eligible",
        "interpolation_rescue/selection_exists",
        "interpolation_rescue/gate_passed",
        "parallel_rescue_selection/eligible_trials",
        "parallel_rescue_selection/gate_passed",
    }
)

FORBIDDEN_DIRECTORY_NAMES = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".ssh",
        ".venv",
        "__pycache__",
        "credentials",
        "secrets",
        "venv",
        "wandb",
    }
)
FORBIDDEN_EXACT_FILE_NAMES = frozenset(
    {
        ".netrc",
        "credentials",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
        "netrc",
        "secrets",
        "secrets.json",
        "wandb_api_key",
    }
)
FORBIDDEN_FILE_SUFFIXES = (".key", ".p12", ".pem", ".pfx")
EVIDENCE_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".log", ".md", ".txt"})
EVIDENCE_EXACT_FILE_NAMES = frozenset({"LICENSE", "NOTICE"})
SAFE_METADATA_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/-]{0,159}")
REMOTE_METADATA_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/=-]{0,511}")
ARTIFACT_VERSION_PATTERN = re.compile(r"v[0-9]+")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_NAME_PATTERN = re.compile(r"barunaction-[a-z0-9][a-z0-9-]{2,95}")
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|apikey|auth[_-]?token|access[_-]?token|password|secret|credential)",
    re.IGNORECASE,
)
TOKEN_MARKERS = (
    re.compile(rb"wandb_v1_[A-Za-z0-9_-]{16,}", re.IGNORECASE),
    re.compile(rb"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{16,}", re.IGNORECASE),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"WANDB_API_KEY\s*[:=]", re.IGNORECASE),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)
TOKEN_SCAN_OVERLAP = 256


class ReleaseLoggingError(RuntimeError):
    """The release bundle or requested logging operation failed closed."""


def _fail(message: str) -> NoReturn:
    raise ReleaseLoggingError(message)


def _reject_constant(value: str) -> NoReturn:
    _fail(f"release JSON contains non-finite constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            _fail(f"release JSON contains duplicate key {key!r}")
        value[key] = child
    return value


def _contains_token_marker(payload: bytes) -> bool:
    return any(pattern.search(payload) is not None for pattern in TOKEN_MARKERS)


def _load_spec(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        _fail("release spec must be a non-symlink regular file")
    raw = path.read_bytes()
    if _contains_token_marker(raw):
        _fail("release spec contains a credential token marker")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ReleaseLoggingError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseLoggingError("release spec is not strict UTF-8 JSON") from error
    if not isinstance(payload, dict):
        _fail("release spec must be a JSON object")
    _reject_sensitive_keys(payload, path="spec")
    return payload, hashlib.sha256(raw).hexdigest()


def _reject_sensitive_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail(f"{path} contains a non-string key")
            if SENSITIVE_KEY_PATTERN.search(key) is not None:
                _fail(f"{path} contains a credential-like field name")
            _reject_sensitive_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_keys(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and _contains_token_marker(value.encode("utf-8")):
        _fail(f"{path} contains a credential token marker")


def _exact_fields(value: Any, expected: set[str], *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be an object")
    actual = set(value)
    if actual != expected:
        _fail(
            f"{path} fields differ: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    return value


def _sha256(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{path} must be a lowercase SHA-256")
    return value


def _safe_metadata(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or SAFE_METADATA_PATTERN.fullmatch(value) is None:
        _fail(f"{path} is not a safe metadata identifier")
    if _contains_token_marker(value.encode("utf-8")):
        _fail(f"{path} contains a credential token marker")
    return value


def _remote_metadata(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or REMOTE_METADATA_PATTERN.fullmatch(value) is None:
        _fail(f"{path} is not a safe immutable W&B identifier")
    if _contains_token_marker(value.encode("utf-8")):
        _fail(f"{path} contains a credential token marker")
    return value


def _relative_path(value: Any, *, path: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"{path} must be a normalized relative POSIX path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or value != relative.as_posix()
        or value == "."
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        _fail(f"{path} must be a normalized relative POSIX path")
    _reject_sensitive_relative_path(relative, path=path)
    return relative


def _reject_sensitive_relative_path(relative: PurePosixPath, *, path: str) -> None:
    for part in relative.parts:
        folded = part.casefold()
        if (
            folded in FORBIDDEN_DIRECTORY_NAMES
            or folded == ".env"
            or folded.startswith((".env.", "credentials.", "secrets."))
            or folded in FORBIDDEN_EXACT_FILE_NAMES
        ):
            _fail(f"{path} enters a forbidden credential, cache, or W&B directory")
    name = relative.name.casefold()
    if (
        name == ".env"
        or name.startswith((".env.", "credentials.", "secrets."))
        or name in FORBIDDEN_EXACT_FILE_NAMES
        or name.endswith(FORBIDDEN_FILE_SUFFIXES)
    ):
        _fail(f"{path} names a credential-like file")


def _assert_within_repository(
    repository_root: Path, relative: PurePosixPath, *, path: str, expect_directory: bool
) -> Path:
    candidate = repository_root.joinpath(*relative.parts)
    current = repository_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            _fail(f"{path} contains a symlink component")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ReleaseLoggingError(f"{path} does not exist") from error
    if resolved != repository_root and repository_root not in resolved.parents:
        _fail(f"{path} escapes the repository root")
    if expect_directory and not resolved.is_dir():
        _fail(f"{path} must be a directory")
    if not expect_directory and not resolved.is_file():
        _fail(f"{path} must be a regular file")
    return resolved


def _hash_and_scan(path: Path, *, label: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    previous = b""
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
                window = previous + chunk
                if _contains_token_marker(window):
                    _fail(f"{label} contains a credential token marker")
                previous = window[-TOKEN_SCAN_OVERLAP:]
    except ReleaseLoggingError:
        raise
    except OSError as error:
        raise ReleaseLoggingError(f"cannot read {label}") from error
    return digest.hexdigest(), size


def _declared_hashes(value: Any, *, artifact_kind: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        _fail(f"artifacts.{artifact_kind}.file_sha256 must be a nonempty object")
    if len(value) > 10_000:
        _fail(f"artifacts.{artifact_kind}.file_sha256 is unexpectedly large")
    result: dict[str, str] = {}
    for raw_path, raw_digest in value.items():
        relative = _relative_path(raw_path, path=f"artifacts.{artifact_kind}.file_sha256 path")
        if (
            artifact_kind == "evidence"
            and relative.suffix.casefold() not in EVIDENCE_SUFFIXES
            and relative.name not in EVIDENCE_EXACT_FILE_NAMES
        ):
            _fail("evidence artifact contains a non-evidence file extension")
        result[relative.as_posix()] = _sha256(
            raw_digest, path=f"artifacts.{artifact_kind}.file_sha256.{relative.as_posix()}"
        )
    return dict(sorted(result.items()))


def _validate_complete_root(
    *, root: Path, declared: Mapping[str, str], artifact_kind: str
) -> dict[str, dict[str, Any]]:
    allowed_directories = {
        parent.as_posix()
        for relative in declared
        for parent in PurePosixPath(relative).parents
        if parent.as_posix() != "."
    }
    observed_files: set[str] = set()
    for candidate in sorted(root.rglob("*")):
        relative = PurePosixPath(candidate.relative_to(root).as_posix())
        _reject_sensitive_relative_path(relative, path=f"artifacts.{artifact_kind} source entry")
        if candidate.is_symlink():
            _fail(f"artifacts.{artifact_kind} source contains a symlink")
        if candidate.is_dir():
            if relative.name.casefold() in FORBIDDEN_DIRECTORY_NAMES:
                _fail(f"artifacts.{artifact_kind} source contains a forbidden directory")
            if relative.as_posix() not in allowed_directories:
                _fail(f"artifacts.{artifact_kind} source contains an unlisted directory")
            continue
        if not candidate.is_file():
            _fail(f"artifacts.{artifact_kind} source contains a non-regular entry")
        observed_files.add(relative.as_posix())
    expected_files = set(declared)
    if observed_files != expected_files:
        _fail(
            f"artifacts.{artifact_kind} source file set differs from its explicit allowlist: "
            f"missing={sorted(expected_files - observed_files)!r}, "
            f"unlisted={sorted(observed_files - expected_files)!r}"
        )
    inspected: dict[str, dict[str, Any]] = {}
    for relative, expected_hash in declared.items():
        candidate = root.joinpath(*PurePosixPath(relative).parts)
        digest, size = _hash_and_scan(
            candidate, label=f"artifacts.{artifact_kind} file {relative!r}"
        )
        if digest != expected_hash:
            _fail(f"artifacts.{artifact_kind} file {relative!r} SHA-256 differs")
        inspected[relative] = {"bytes": size, "sha256": digest}
    return inspected


def _load_strict_json_artifact(path: Path, *, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a non-symlink regular file")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ReleaseLoggingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseLoggingError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        _fail(f"{label} must contain a JSON object")
    return payload


def _load_strict_jsonl_artifact(path: Path, *, label: str) -> list[Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a non-symlink regular file")
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    _fail(f"{label} contains a blank line")
                try:
                    row = json.loads(
                        line,
                        object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant,
                    )
                except json.JSONDecodeError as error:
                    raise ReleaseLoggingError(
                        f"{label} contains invalid JSON at line {line_number}"
                    ) from error
                if not isinstance(row, Mapping):
                    _fail(f"{label} contains a non-object row")
                rows.append(row)
    except ReleaseLoggingError:
        raise
    except (OSError, UnicodeError) as error:
        raise ReleaseLoggingError(f"cannot read {label}") from error
    return rows


def _embedded_hash_map(value: Any, *, expected: set[str], label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != expected:
        _fail(f"{label} file set differs from the release contract")
    return {name: _sha256(value[name], path=f"{label}.{name}") for name in sorted(expected)}


def _manifest_file_hashes(artifact: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: _sha256(record["sha256"], path=f"artifact manifest file {name}")
        for name, record in artifact["files"].items()
    }


def _validate_candidate_and_quantization_manifests(
    *, artifacts: Mapping[str, Mapping[str, Any]], repository_root: Path
) -> None:
    checkpoint_files = {"barun_config.json", "model.safetensors", "tokenizer.json"}
    int8_files = {"barun_config.json", "model.int8.pt", "tokenizer.json"}
    float_artifact = artifacts["float"]
    int8_artifact = artifacts["int8_darwin_arm64_qnnpack"]
    float_hashes = _manifest_file_hashes(float_artifact)
    int8_hashes = _manifest_file_hashes(int8_artifact)

    candidate_path = repository_root / "configs/barunaction/candidate-v2.json"
    candidate = _exact_fields(
        _load_strict_json_artifact(candidate_path, label="canonical candidate-v2 manifest"),
        {
            "arm_id",
            "candidate_id",
            "checkpoint_manifest_sha256",
            "checkpoint_relative_path",
            "file_sha256",
            "run_id",
            "schema_version",
            "selection_run_id",
            "step",
        },
        path="canonical candidate-v2 manifest",
    )
    if (
        candidate["schema_version"] != "barunaction-candidate-provenance-v1"
        or candidate["candidate_id"] != CANDIDATE_ID
    ):
        _fail("canonical candidate-v2 manifest identity changed")
    candidate_hashes = _embedded_hash_map(
        candidate["file_sha256"], expected=checkpoint_files, label="candidate-v2 file hashes"
    )
    if candidate_hashes != {name: float_hashes[name] for name in sorted(checkpoint_files)}:
        _fail("float artifact does not match the canonical candidate-v2 file hashes")
    if (
        _sha256(
            candidate["checkpoint_manifest_sha256"],
            path="candidate-v2 checkpoint manifest hash",
        )
        != float_hashes["checkpoint_manifest.json"]
    ):
        _fail("float artifact checkpoint manifest differs from candidate-v2 provenance")

    float_root = repository_root / str(float_artifact["source_root"])
    checkpoint = _exact_fields(
        _load_strict_json_artifact(
            float_root / "checkpoint_manifest.json", label="float checkpoint manifest"
        ),
        {"arm_id", "file_sha256", "schema_version", "source_checkpoint"},
        path="float checkpoint manifest",
    )
    if checkpoint["schema_version"] != "barun-release-checkpoint-v1":
        _fail("float checkpoint manifest schema changed")
    if (
        _embedded_hash_map(
            checkpoint["file_sha256"],
            expected=checkpoint_files,
            label="float checkpoint manifest hashes",
        )
        != candidate_hashes
    ):
        _fail("float checkpoint manifest does not bind its packaged files")

    int8_root = repository_root / str(int8_artifact["source_root"])
    quantization = _exact_fields(
        _load_strict_json_artifact(
            int8_root / "quantization_manifest.json", label="int8 quantization manifest"
        ),
        {
            "algorithm",
            "architecture",
            "artifact_sha256",
            "float_linear_modules",
            "parameter_counts",
            "quantized_modules",
            "runtime",
            "schema_version",
            "size_bytes",
            "source_checkpoint_sha256",
        },
        path="int8 quantization manifest",
    )
    if quantization["schema_version"] != "barun-cpu-dynamic-int8-v1":
        _fail("int8 quantization manifest schema changed")
    if quantization["architecture"] != "barunlm.model.BarunLM":
        _fail("int8 quantization architecture changed")
    algorithm = quantization["algorithm"]
    if not isinstance(algorithm, Mapping) or algorithm.get("qengine") != "qnnpack":
        _fail("int8 artifact is not pinned to QNNPACK")
    runtime = quantization["runtime"]
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("platform_system") != "Darwin"
        or runtime.get("platform_machine") != "arm64"
    ):
        _fail("int8 artifact is not the Darwin ARM64 build")
    parameter_counts = quantization["parameter_counts"]
    if (
        not isinstance(parameter_counts, Mapping)
        or parameter_counts.get("total_unique_parameters") != 35_072_768
    ):
        _fail("int8 artifact parameter identity changed")
    if _embedded_hash_map(
        quantization["artifact_sha256"],
        expected=int8_files,
        label="int8 artifact hashes",
    ) != {name: int8_hashes[name] for name in sorted(int8_files)}:
        _fail("int8 quantization manifest does not bind its packaged files")
    if (
        _embedded_hash_map(
            quantization["source_checkpoint_sha256"],
            expected=checkpoint_files,
            label="int8 source checkpoint hashes",
        )
        != candidate_hashes
    ):
        _fail("int8 artifact was not derived from canonical candidate-v2")


def _validate_int8_retention(
    *,
    value: Any,
    artifacts: Mapping[str, Mapping[str, Any]],
    repository_root: Path,
) -> dict[str, Any]:
    contract = _exact_fields(value, {"file_sha256", "source_root"}, path="int8_retention")
    source_relative = _relative_path(contract["source_root"], path="int8_retention.source_root")
    source_root = _assert_within_repository(
        repository_root,
        source_relative,
        path="int8_retention.source_root",
        expect_directory=True,
    )
    for artifact in artifacts.values():
        artifact_root = repository_root / str(artifact["source_root"])
        if (
            source_root == artifact_root
            or source_root in artifact_root.parents
            or artifact_root in source_root.parents
        ):
            _fail("int8 retention receipt must have a dedicated non-overlapping source root")
    declared = _declared_hashes(contract["file_sha256"], artifact_kind="int8_retention")
    if set(declared) != INT8_RETENTION_FILES:
        _fail("int8 retention receipt does not declare its exact immutable file set")
    inspected = _validate_complete_root(
        root=source_root,
        declared=declared,
        artifact_kind="int8_retention",
    )

    artifact_manifest = _exact_fields(
        _load_strict_json_artifact(
            source_root / "artifact-sha256.json",
            label="int8 retention artifact manifest",
        ),
        {"file_sha256", "schema_version"},
        path="int8 retention artifact manifest",
    )
    if artifact_manifest["schema_version"] != INT8_RETENTION_ARTIFACT_SCHEMA:
        _fail("int8 retention artifact manifest schema changed")
    receipt_payload_files = INT8_RETENTION_FILES - {"artifact-sha256.json"}
    if _embedded_hash_map(
        artifact_manifest["file_sha256"],
        expected=receipt_payload_files,
        label="int8 retention artifact manifest hashes",
    ) != {name: declared[name] for name in sorted(receipt_payload_files)}:
        _fail("int8 retention artifact manifest does not bind its receipt files")
    if declared["protocol.json"] != INT8_RETENTION_PROTOCOL_SHA256:
        _fail("int8 retention receipt does not contain the frozen protocol")

    result = _exact_fields(
        _load_strict_json_artifact(source_root / "result.json", label="int8 retention result"),
        {
            "candidate",
            "evidence",
            "gate",
            "generation",
            "int8_checkpoint",
            "interpretation",
            "population",
            "protocol",
            "schema_version",
        },
        path="int8 retention result",
    )
    if result["schema_version"] != INT8_RETENTION_RESULT_SCHEMA:
        _fail("int8 retention result schema changed")
    protocol = _exact_fields(
        result["protocol"], {"path", "protocol_id", "sha256"}, path="int8 retention protocol"
    )
    if (
        protocol["protocol_id"] != "candidate-v2-arm64-int8-retention-v1"
        or protocol["sha256"] != INT8_RETENTION_PROTOCOL_SHA256
    ):
        _fail("int8 retention result does not bind the frozen protocol")

    float_files = {"barun_config.json", "model.safetensors", "tokenizer.json"}
    int8_files = {"barun_config.json", "model.int8.pt", "tokenizer.json"}
    float_hashes = _manifest_file_hashes(artifacts["float"])
    int8_hashes = _manifest_file_hashes(artifacts["int8_darwin_arm64_qnnpack"])
    candidate = _exact_fields(
        result["candidate"],
        {"candidate_id", "float_checkpoint_file_sha256"},
        path="int8 retention candidate",
    )
    if candidate["candidate_id"] != CANDIDATE_ID:
        _fail("int8 retention result candidate changed")
    if _embedded_hash_map(
        candidate["float_checkpoint_file_sha256"],
        expected=float_files,
        label="int8 retention float hashes",
    ) != {name: float_hashes[name] for name in sorted(float_files)}:
        _fail("int8 retention result is not bound to canonical candidate-v2")

    int8_checkpoint = _exact_fields(
        result["int8_checkpoint"],
        {
            "artifact_sha256",
            "directory",
            "format_version",
            "manifest_sha256",
            "qengine",
            "source_checkpoint_file_sha256",
        },
        path="int8 retention checkpoint",
    )
    if (
        int8_checkpoint["format_version"] != "barun-cpu-dynamic-int8-v1"
        or int8_checkpoint["qengine"] != "qnnpack"
        or int8_checkpoint["manifest_sha256"] != int8_hashes["quantization_manifest.json"]
    ):
        _fail("int8 retention result checkpoint identity changed")
    if _embedded_hash_map(
        int8_checkpoint["artifact_sha256"],
        expected=int8_files,
        label="int8 retention artifact hashes",
    ) != {name: int8_hashes[name] for name in sorted(int8_files)}:
        _fail("int8 retention result does not bind the packaged int8 artifact")
    if _embedded_hash_map(
        int8_checkpoint["source_checkpoint_file_sha256"],
        expected=float_files,
        label="int8 retention source hashes",
    ) != {name: float_hashes[name] for name in sorted(float_files)}:
        _fail("int8 retention result source differs from canonical candidate-v2")

    population = result["population"]
    if (
        not isinstance(population, Mapping)
        or population.get("rows") != 756
        or population.get("official_evaluation_rows_opaque_unparsed") != 961
        or population.get("official_evaluation_artifacts_accessed") != []
    ):
        _fail("int8 retention population or evaluation firewall changed")

    paired_rows = _load_strict_jsonl_artifact(
        source_root / "paired-samples.jsonl", label="int8 retention paired samples"
    )
    if len(paired_rows) != 756:
        _fail("int8 retention receipt must contain exactly 756 paired samples")
    sample_ids: set[str] = set()
    transition_counts = {
        "fixed": 0,
        "regressed": 0,
        "retained_correct": 0,
        "retained_incorrect": 0,
    }
    int8_correct = 0
    float_correct = 0
    for index, row in enumerate(paired_rows, start=1):
        paired = _exact_fields(
            row,
            {
                "float_ast_exact",
                "int8_ast_exact",
                "sample_id",
                "scenario",
                "schema_version",
                "transition",
            },
            path=f"int8 retention paired sample {index}",
        )
        if paired["schema_version"] != INT8_RETENTION_PAIRED_SCHEMA:
            _fail("int8 retention paired sample schema changed")
        sample_id = _safe_metadata(
            paired["sample_id"], path=f"int8 retention paired sample {index} ID"
        )
        if sample_id in sample_ids:
            _fail("int8 retention paired samples contain a duplicate ID")
        sample_ids.add(sample_id)
        if not isinstance(paired["scenario"], str) or not paired["scenario"]:
            _fail("int8 retention paired sample has an invalid scenario")
        if (
            type(paired["float_ast_exact"]) is not bool
            or type(paired["int8_ast_exact"]) is not bool
        ):
            _fail("int8 retention paired outcomes must be boolean")
        float_exact = paired["float_ast_exact"]
        int8_exact = paired["int8_ast_exact"]
        expected_transition = (
            "retained_correct"
            if float_exact and int8_exact
            else "regressed"
            if float_exact
            else "fixed"
            if int8_exact
            else "retained_incorrect"
        )
        if paired["transition"] != expected_transition:
            _fail("int8 retention paired transition disagrees with its outcomes")
        transition_counts[expected_transition] += 1
        int8_correct += int8_exact
        float_correct += float_exact

    gate = _exact_fields(
        result["gate"],
        {
            "component_gates",
            "decision",
            "float_to_int8",
            "int8",
            "metric",
            "paired_transition_counts",
            "passed",
            "source_float",
            "status",
            "thresholds",
        },
        path="int8 retention gate",
    )
    correct_rows_lost = float_correct - int8_correct
    expected_source = {"denominator": 756, "numerator": 602, "value": 602 / 756}
    expected_int8 = {
        "denominator": 756,
        "numerator": int8_correct,
        "value": int8_correct / 756,
    }
    if float_correct != 602 or gate["source_float"] != expected_source:
        _fail("int8 retention float reference changed")
    if gate["int8"] != expected_int8:
        _fail("int8 retention aggregate disagrees with paired samples")
    if gate["paired_transition_counts"] != transition_counts:
        _fail("int8 retention transition counts disagree with paired samples")
    if gate["thresholds"] != {
        "maximum_float_to_int8_correct_rows_lost": 15,
        "minimum_int8_numerator": 587,
    }:
        _fail("int8 retention thresholds changed")
    if gate["float_to_int8"] != {
        "absolute_accuracy_change": (int8_correct - 602) / 756,
        "correct_rows_lost": correct_rows_lost,
    }:
        _fail("int8 retention loss summary disagrees with paired samples")
    if gate["component_gates"] != {
        "complete_aligned_samples": True,
        "explicit_int8_no_fallback": True,
        "maximum_float_to_int8_loss": True,
        "minimum_int8_accuracy": True,
    }:
        _fail("int8 retention component gates did not all pass")
    if (
        gate["metric"] != "ast_exact_match"
        or gate["passed"] is not True
        or gate["status"] != "pass"
        or gate["decision"] != "retain_candidate_v2_int8_artifact"
        or int8_correct < 587
        or correct_rows_lost > 15
    ):
        _fail("int8 artifact cannot be released because its retention gate did not pass")

    return {
        "disposition": "retained-release",
        "file_count": len(inspected),
        "files": inspected,
        "gate": {
            "ast_exact_match": int8_correct / 756,
            "correct_rows_lost": correct_rows_lost,
            "passed": True,
            "source_float_ast_exact_match": 602 / 756,
        },
        "source_root": source_relative.as_posix(),
        "total_bytes": sum(item["bytes"] for item in inspected.values()),
    }


def _validate_run(value: Any) -> dict[str, Any]:
    run = _exact_fields(
        value,
        {"candidate_id", "group", "job_type", "name", "release_id", "source_runs", "tags"},
        path="run",
    )
    if run["candidate_id"] != CANDIDATE_ID:
        _fail("run.candidate_id must remain candidate-v2")
    if run["group"] != "candidate-v2-release":
        _fail("run.group must be candidate-v2-release")
    if run["job_type"] != "release-evidence":
        _fail("run.job_type must be release-evidence")
    name = _safe_metadata(run["name"], path="run.name")
    if RUN_NAME_PATTERN.fullmatch(name) is None:
        _fail("run.name must be a barunaction-* identifier")
    release_id = _safe_metadata(run["release_id"], path="run.release_id")
    tags = run["tags"]
    if not isinstance(tags, list) or not tags or any(not isinstance(tag, str) for tag in tags):
        _fail("run.tags must be a nonempty string array")
    normalized_tags = tuple(tags)
    if normalized_tags != tuple(sorted(set(normalized_tags))):
        _fail("run.tags must be sorted and unique")
    if not set(normalized_tags).issubset(ALLOWED_TAGS):
        _fail("run.tags contains a value outside the release allowlist")
    source_runs = run["source_runs"]
    if not isinstance(source_runs, Mapping) or not source_runs:
        _fail("run.source_runs must be a nonempty object")
    if set(source_runs) != SOURCE_RUN_NAMES:
        _fail(
            "run.source_runs must contain every completed release-evidence class: "
            f"missing={sorted(SOURCE_RUN_NAMES - set(source_runs))!r}, "
            f"unknown={sorted(set(source_runs) - SOURCE_RUN_NAMES)!r}"
        )
    normalized_runs = {
        key: _safe_metadata(value, path=f"run.source_runs.{key}")
        for key, value in sorted(source_runs.items())
    }
    return {
        "candidate_id": CANDIDATE_ID,
        "group": "candidate-v2-release",
        "job_type": "release-evidence",
        "name": name,
        "release_id": release_id,
        "source_runs": normalized_runs,
        "tags": list(normalized_tags),
    }


def _validate_summary(value: Any, *, source_runs: Mapping[str, str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        _fail("summary_metrics must be a nonempty object")
    unknown = set(value) - SUMMARY_METRIC_NAMES
    if unknown:
        _fail(f"summary_metrics contains unknown names: {sorted(unknown)!r}")
    missing = SUMMARY_METRIC_NAMES - set(value)
    if missing:
        _fail(f"summary_metrics omits required completed evidence: {sorted(missing)!r}")
    normalized: dict[str, Any] = {}
    represented_sources: set[str] = set()
    for name, raw_value in sorted(value.items()):
        source = name.split("/", 1)[0]
        if source not in source_runs:
            _fail(f"summary metric {name!r} lacks an allowlisted source run")
        represented_sources.add(source)
        if name.endswith(
            (
                "/gate_passed",
                "/eligible",
                "/selection_exists",
                "/qwen_outperformed_barunaction",
            )
        ):
            if type(raw_value) is not bool:
                _fail(f"summary metric {name!r} must be boolean")
            normalized[name] = raw_value
            continue
        if name.endswith("/correct_rows_lost"):
            if type(raw_value) is not int or not -154 <= raw_value <= 15:
                _fail(f"summary metric {name!r} must be an integer within [-154, 15]")
            normalized[name] = raw_value
            continue
        if name.endswith(("/parameters", "/eligible_trials")):
            if type(raw_value) is not int or raw_value < 0:
                _fail(f"summary metric {name!r} must be a non-negative integer")
            normalized[name] = raw_value
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            _fail(f"summary metric {name!r} must be numeric")
        number = float(raw_value)
        if not math.isfinite(number):
            _fail(f"summary metric {name!r} must be finite")
        if not 0.0 <= number <= 1.0:
            _fail(f"summary metric {name!r} must be within [0, 1]")
        normalized[name] = number
    missing_sources = set(source_runs) - represented_sources
    if missing_sources:
        _fail(f"source runs lack summary metrics: {sorted(missing_sources)!r}")
    return normalized


def _validate_int8_retention_summary(
    summary: Mapping[str, Any], *, retention: Mapping[str, Any]
) -> None:
    gate = retention["gate"]
    expected = {
        "mobile_int8_retention/ast_exact_match": gate["ast_exact_match"],
        "mobile_int8_retention/correct_rows_lost": gate["correct_rows_lost"],
        "mobile_int8_retention/gate_passed": True,
        "mobile_int8_retention/source_float_ast_exact_match": gate["source_float_ast_exact_match"],
    }
    observed = {name: summary.get(name) for name in expected}
    if observed != expected:
        _fail("summary_metrics does not exactly reproduce the passed int8 retention receipt")


def _validate_artifacts(value: Any, *, repository_root: Path) -> dict[str, dict[str, Any]]:
    artifacts = _exact_fields(value, set(ARTIFACT_CONTRACTS), path="artifacts")
    normalized: dict[str, dict[str, Any]] = {}
    source_roots: set[Path] = set()
    for kind, contract in ARTIFACT_CONTRACTS.items():
        artifact = _exact_fields(
            artifacts[kind],
            {"file_sha256", "name", "source_root", "type"},
            path=f"artifacts.{kind}",
        )
        if artifact["name"] != contract["name"] or artifact["type"] != contract["type"]:
            _fail(f"artifacts.{kind} name or type differs from the fixed release contract")
        source_relative = _relative_path(
            artifact["source_root"], path=f"artifacts.{kind}.source_root"
        )
        source_root = _assert_within_repository(
            repository_root,
            source_relative,
            path=f"artifacts.{kind}.source_root",
            expect_directory=True,
        )
        if any(
            source_root == existing
            or source_root in existing.parents
            or existing in source_root.parents
            for existing in source_roots
        ):
            _fail("release artifacts must use distinct non-overlapping dedicated source roots")
        source_roots.add(source_root)
        declared = _declared_hashes(artifact["file_sha256"], artifact_kind=kind)
        required_files = contract["required_files"]
        if required_files is not None and set(declared) != required_files:
            _fail(f"artifacts.{kind} does not declare its exact required file set")
        inspected = _validate_complete_root(root=source_root, declared=declared, artifact_kind=kind)
        normalized[kind] = {
            "file_count": len(inspected),
            "files": inspected,
            "name": contract["name"],
            "source_root": source_relative.as_posix(),
            "total_bytes": sum(item["bytes"] for item in inspected.values()),
            "type": contract["type"],
        }
    _validate_candidate_and_quantization_manifests(
        artifacts=normalized, repository_root=repository_root
    )
    return normalized


def _external_wandb_directory(repository_root: Path, value: Path) -> Path:
    absolute = Path(os.path.abspath(value))
    current = absolute
    while current != current.parent:
        if current.exists() and current.is_symlink():
            _fail("WANDB_DIR cannot contain a symlink component")
        current = current.parent
    resolved = absolute.resolve(strict=False)
    if resolved == repository_root or repository_root in resolved.parents:
        _fail("WANDB_DIR must be outside the repository")
    return resolved


def build_release_manifest(
    *, spec_path: Path, repository_root: Path, wandb_dir: Path
) -> dict[str, Any]:
    if repository_root.is_symlink() or not repository_root.is_dir():
        _fail("repository root must be a non-symlink directory")
    repository_root = repository_root.resolve(strict=True)
    spec_absolute = Path(os.path.abspath(spec_path))
    if spec_absolute == repository_root or repository_root not in spec_absolute.parents:
        _fail("release spec must be inside the repository")
    spec_relative = PurePosixPath(spec_absolute.relative_to(repository_root).as_posix())
    _reject_sensitive_relative_path(spec_relative, path="release spec")
    current = repository_root
    for part in spec_relative.parts:
        current = current / part
        if current.is_symlink():
            _fail("release spec contains a symlink component")
    spec_path = spec_absolute.resolve(strict=True)
    spec, spec_sha256 = _load_spec(spec_path)
    spec = dict(
        _exact_fields(
            spec,
            {
                "artifacts",
                "int8_retention",
                "project",
                "run",
                "schema_version",
                "summary_metrics",
            },
            path="spec",
        )
    )
    if spec["schema_version"] != SPEC_SCHEMA_VERSION:
        _fail("unsupported release spec schema")
    if spec["project"] != WANDB_PROJECT:
        _fail(f"project must be {WANDB_PROJECT!r}")
    run = _validate_run(spec["run"])
    summary = _validate_summary(spec["summary_metrics"], source_runs=run["source_runs"])
    artifacts = _validate_artifacts(spec["artifacts"], repository_root=repository_root)
    int8_retention = _validate_int8_retention(
        value=spec["int8_retention"],
        artifacts=artifacts,
        repository_root=repository_root,
    )
    _validate_int8_retention_summary(summary, retention=int8_retention)
    _external_wandb_directory(repository_root, wandb_dir)
    return {
        "artifacts": artifacts,
        "int8_retention": int8_retention,
        "local_manifest_authoritative": True,
        "project": WANDB_PROJECT,
        "run": run,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "security": {
            "credential_markers_found": 0,
            "symlinks_found": 0,
            "unlisted_source_entries_found": 0,
        },
        "spec": {
            "path": spec_path.relative_to(repository_root).as_posix(),
            "sha256": spec_sha256,
        },
        "summary_metrics": summary,
        "wandb_plan": {
            "automatic_code_capture": False,
            "automatic_git_capture": False,
            "evidence_includes_local_manifest_as": "release-manifest.json",
            "external_wandb_dir_required": True,
            "package_version": WANDB_VERSION,
            "upload_requires_explicit_opt_in": True,
        },
    }


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite local record: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise ReleaseLoggingError(f"refusing to overwrite local record: {path.name}") from error
    return hashlib.sha256(serialized).hexdigest()


def _manifest_output_path(repository_root: Path, path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    if absolute == repository_root or repository_root not in absolute.parents:
        _fail(f"{label} must be inside the repository")
    relative = PurePosixPath(absolute.relative_to(repository_root).as_posix())
    _reject_sensitive_relative_path(relative, path=label)
    current = repository_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            _fail(f"{label} cannot contain a symlink component")
    resolved = absolute.resolve(strict=False)
    if resolved == repository_root or repository_root not in resolved.parents:
        _fail(f"{label} must resolve inside the repository")
    return resolved


def _recheck_upload_file(path: Path, expected_sha256: str, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} changed after dry-run validation")
    observed, _ = _hash_and_scan(path, label=label)
    if observed != expected_sha256:
        _fail(f"{label} changed after dry-run validation")


def _committed_artifact_identity(
    *, logged_artifact: Any, entity: str, project: str, expected_name: str
) -> dict[str, str]:
    wait = getattr(logged_artifact, "wait", None)
    if not callable(wait):
        _fail("W&B did not expose artifact.wait() for immutable version resolution")
    waited = wait()
    committed = logged_artifact if waited is None else waited
    version = _remote_metadata(
        getattr(committed, "version", ""), path=f"W&B artifact {expected_name} version"
    )
    if ARTIFACT_VERSION_PATTERN.fullmatch(version) is None:
        _fail("W&B artifact version is not an immutable vN identifier")
    digest = _remote_metadata(
        getattr(committed, "digest", ""), path=f"W&B artifact {expected_name} digest"
    )
    qualified_name = f"{entity}/{project}/{expected_name}:{version}"
    returned_qualified = getattr(committed, "qualified_name", None)
    if returned_qualified is not None:
        returned_qualified = _remote_metadata(
            returned_qualified, path=f"W&B artifact {expected_name} qualified name"
        )
        if returned_qualified != qualified_name:
            _fail("W&B returned an unexpected artifact qualified name")
    return {
        "digest": digest,
        "name": expected_name,
        "qualified_name": qualified_name,
        "version": version,
    }


def _upload(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    repository_root: Path,
    wandb_dir: Path,
) -> dict[str, Any]:
    try:
        wandb = importlib.import_module("wandb")
    except ImportError as error:
        raise ReleaseLoggingError("W&B upload requires requirements/wandb-release.txt") from error
    if getattr(wandb, "__version__", None) != WANDB_VERSION:
        _fail(f"W&B upload requires exactly wandb=={WANDB_VERSION}")

    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_DISABLE_CODE"] = "true"
    run_contract = manifest["run"]
    try:
        settings = wandb.Settings(disable_git=True)
        run = wandb.init(
            project=WANDB_PROJECT,
            name=run_contract["name"],
            group=run_contract["group"],
            job_type=run_contract["job_type"],
            tags=run_contract["tags"],
            config={
                "candidate_id": CANDIDATE_ID,
                "local_manifest_sha256": manifest_sha256,
                "release_id": run_contract["release_id"],
                "source_runs": run_contract["source_runs"],
            },
            dir=str(wandb_dir),
            mode="online",
            save_code=False,
            settings=settings,
        )
    except Exception as error:
        raise ReleaseLoggingError(
            "W&B initialization failed; local manifest remains authoritative"
        ) from error
    if run is None:
        _fail("wandb.init returned no run")
    committed_artifacts: dict[str, dict[str, str]] = {}
    try:
        entity = _remote_metadata(getattr(run, "entity", ""), path="W&B returned entity")
        project = _remote_metadata(getattr(run, "project", ""), path="W&B returned project")
        run_id = _safe_metadata(getattr(run, "id", ""), path="W&B returned run ID")
        if project != WANDB_PROJECT:
            _fail("W&B initialized a different project than the fixed release project")
        for kind, artifact_contract in manifest["artifacts"].items():
            metadata = {
                "candidate_id": CANDIDATE_ID,
                "local_manifest_sha256": manifest_sha256,
                "source_file_count": artifact_contract["file_count"],
                "source_total_bytes": artifact_contract["total_bytes"],
            }
            if kind == "int8_darwin_arm64_qnnpack":
                metadata.update(
                    {
                        "release_disposition": "retained-release",
                        "retention_gate_passed": True,
                        "retention_receipt_manifest_sha256": manifest["int8_retention"]["files"][
                            "artifact-sha256.json"
                        ]["sha256"],
                    }
                )
            artifact = wandb.Artifact(
                name=artifact_contract["name"],
                type=artifact_contract["type"],
                metadata=metadata,
            )
            source_root = repository_root / artifact_contract["source_root"]
            for relative, file_record in artifact_contract["files"].items():
                source = source_root.joinpath(*PurePosixPath(relative).parts)
                _recheck_upload_file(
                    source,
                    file_record["sha256"],
                    label=f"artifacts.{kind} file {relative!r}",
                )
                artifact.add_file(str(source), name=relative)
                _recheck_upload_file(
                    source,
                    file_record["sha256"],
                    label=f"artifacts.{kind} file {relative!r}",
                )
            if kind == "evidence":
                retention_root = repository_root / manifest["int8_retention"]["source_root"]
                for relative, file_record in manifest["int8_retention"]["files"].items():
                    source = retention_root.joinpath(*PurePosixPath(relative).parts)
                    _recheck_upload_file(
                        source,
                        file_record["sha256"],
                        label=f"int8 retention receipt file {relative!r}",
                    )
                    artifact.add_file(str(source), name=f"int8-retention/{relative}")
                    _recheck_upload_file(
                        source,
                        file_record["sha256"],
                        label=f"int8 retention receipt file {relative!r}",
                    )
                _recheck_upload_file(
                    manifest_path,
                    manifest_sha256,
                    label="local release manifest",
                )
                artifact.add_file(str(manifest_path), name="release-manifest.json")
                _recheck_upload_file(
                    manifest_path,
                    manifest_sha256,
                    label="local release manifest",
                )
            logged_artifact = run.log_artifact(artifact)
            if logged_artifact is None:
                _fail("W&B log_artifact returned no committed-artifact handle")
            committed_artifacts[kind] = _committed_artifact_identity(
                logged_artifact=logged_artifact,
                entity=entity,
                project=project,
                expected_name=artifact_contract["name"],
            )
        run.summary.update(manifest["summary_metrics"])
        run.finish(exit_code=0)
    except Exception as error:
        run.finish(exit_code=1)
        raise ReleaseLoggingError(
            "W&B upload failed; local manifest remains authoritative"
        ) from error
    return {
        "artifacts": committed_artifacts,
        "entity": entity,
        "local_manifest": {
            "path": manifest_path.relative_to(repository_root).as_posix(),
            "sha256": manifest_sha256,
        },
        "project": WANDB_PROJECT,
        "run_id": run_id,
        "schema_version": RECEIPT_SCHEMA_VERSION,
    }


def execute(
    *,
    spec_path: Path,
    manifest_path: Path,
    repository_root: Path,
    wandb_dir: Path,
    upload: bool,
    receipt_path: Path | None,
) -> dict[str, Any]:
    if repository_root.is_symlink() or not repository_root.is_dir():
        _fail("repository root must be a non-symlink directory")
    repository_root = repository_root.resolve(strict=True)
    wandb_dir = _external_wandb_directory(repository_root, wandb_dir)
    manifest_path = _manifest_output_path(
        repository_root, manifest_path, label="local manifest output"
    )
    if upload and receipt_path is None:
        _fail("--upload requires --receipt")
    if not upload and receipt_path is not None:
        _fail("--receipt is valid only with --upload")
    resolved_receipt = (
        _manifest_output_path(repository_root, receipt_path, label="upload receipt")
        if receipt_path is not None
        else None
    )
    if resolved_receipt == manifest_path:
        _fail("manifest and receipt outputs must be distinct")
    for output in (manifest_path, resolved_receipt):
        if output is not None and (output.exists() or output.is_symlink()):
            _fail(f"refusing to overwrite local record: {output.name}")
    manifest = build_release_manifest(
        spec_path=spec_path,
        repository_root=repository_root,
        wandb_dir=wandb_dir,
    )
    source_roots = {
        (repository_root / artifact["source_root"]).resolve(strict=True)
        for artifact in manifest["artifacts"].values()
    }
    source_roots.add(
        (repository_root / manifest["int8_retention"]["source_root"]).resolve(strict=True)
    )
    for output, label in ((manifest_path, "manifest"), (resolved_receipt, "receipt")):
        if output is None:
            continue
        if any(output == root or root in output.parents for root in source_roots):
            _fail(f"{label} output cannot be inside an artifact source root")
    manifest_sha256 = _write_new_json(manifest_path, manifest)
    result: dict[str, Any] = {
        "artifact_names": [manifest["artifacts"][kind]["name"] for kind in ARTIFACT_CONTRACTS],
        "manifest_path": manifest_path.relative_to(repository_root).as_posix(),
        "manifest_sha256": manifest_sha256,
        "mode": "dry-run",
    }
    if upload:
        assert resolved_receipt is not None
        receipt = _upload(
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            repository_root=repository_root,
            wandb_dir=wandb_dir,
        )
        receipt_sha256 = _write_new_json(resolved_receipt, receipt)
        result.update(
            {
                "mode": "upload",
                "receipt_path": resolved_receipt.relative_to(repository_root).as_posix(),
                "receipt_sha256": receipt_sha256,
                "wandb_run_id": receipt["run_id"],
            }
        )
    return result


def _reject_sensitive_cli(arguments: Sequence[str]) -> None:
    for argument in arguments:
        lowered = argument.casefold()
        if SENSITIVE_KEY_PATTERN.search(lowered) is not None or _contains_token_marker(
            argument.encode("utf-8", errors="ignore")
        ):
            _fail("refusing a credential-like CLI argument")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wandb-dir", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        _reject_sensitive_cli(arguments)
        args = build_parser().parse_args(arguments)
        result = execute(
            spec_path=args.spec,
            manifest_path=args.manifest,
            repository_root=args.repository_root,
            wandb_dir=args.wandb_dir,
            upload=args.upload,
            receipt_path=args.receipt,
        )
    except ReleaseLoggingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
