"""Run the frozen three-arm BarunAction-35M checkpoint interpolation rescue."""

from __future__ import annotations

# Match the frozen Mobile reference before importing torch or project modules.
import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
_CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT = (
    os.environ.get("CUBLAS_WORKSPACE_CONFIG") == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
)

import torch
from tokenizers import Tokenizer

from barunlm.datasets.presto import (
    PRESTO_ARCHIVE_SHA256,
    PRESTO_REVISION,
    TokenizerIdentity,
    download_pinned_archive,
    prepare_presto,
)
from barunlm.evaluation.generation import generate_manifest, verify_checkpoint
from barunlm.evaluation.interpolation_rescue import (
    INTERPOLATION_PROTOCOL_PATH,
    INTERPOLATION_PROTOCOL_SHA256,
    INTERPOLATION_RESULT_SCHEMA_VERSION,
    InterpolationRescueError,
    endpoint_hashes,
    evaluate_joint_gate,
    file_manifest,
    interpolate_checkpoint,
    interpolation_arms,
    load_interpolation_protocol,
    select_interpolation_arm,
)
from barunlm.evaluation.mobile_regression import run_mobile_regression
from barunlm.evaluation.presto import write_scores as write_presto_scores
from barunlm.training.data import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_ID_PATTERN = re.compile(r"\d{8}-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*-s\d+")
ATTEMPT_PREREGISTRATION_SCHEMA_VERSION = (
    "barun-interpolation-attempt-preregistration-v1"
)
SOURCE_SNAPSHOT_SCHEMA_VERSION = "barun-interpolation-source-snapshot-v1"
CONTENT_TREE_SCHEMA_VERSION = "barun-interpolation-content-tree-v1"
CONTENT_TREE_HASH_METHOD = (
    "SHA-256 of UTF-8 lines '<file SHA-256><two spaces><relative POSIX path><newline>' "
    "in relative-path sort order"
)
ATTEMPT_PREREGISTRATION_RELATIVE_PATH = "interpolation-attempt-preregistration.json"
SOURCE_SNAPSHOT_RELATIVE_PATH = "interpolation-source-snapshot.json"
FROZEN_RUNNER_RELATIVE_PATH = "scripts/run_interpolation_rescue.py"
FROZEN_PROTOCOL_RELATIVE_PATH = "configs/interpolation_rescue_v1.json"
FROZEN_SHARED_SELECTOR_RELATIVE_PATH = "configs/parallel_rescue_selection_v1.json"
FROZEN_SHARED_SELECTOR_SHA256 = (
    "68bb6132a486dfbd6c9ad478db9c87ae9f2350bfc1d302193169fd7f5c290130"
)
TREE_EXCLUDED_RELATIVE_PATHS = (
    ATTEMPT_PREREGISTRATION_RELATIVE_PATH,
    SOURCE_SNAPSHOT_RELATIVE_PATH,
)
TREE_EXCLUDED_DIRECTORY_NAMES = (
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "jl-runs",
    "node_modules",
    "venv",
    "wandb",
)
TREE_EXCLUDED_DIRECTORY_SUFFIXES = (".egg-info",)
TREE_EXCLUDED_FILE_NAMES = (".DS_Store",)
TREE_EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo")
_CREDENTIAL_FILE_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        ".npmrc",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ed25519",
        "id_rsa",
        "wandb_api_key",
    }
)
_CREDENTIAL_DIRECTORY_NAMES = frozenset({".aws", ".ssh", "credentials", "secrets"})
_CREDENTIAL_FILE_SUFFIXES = (".key", ".p12", ".pem", ".pfx")
_CREDENTIAL_CONTENT_MARKERS = (
    b"wandb_" + b"v1_",
    b"github_" + b"pat_",
    b"gh" + b"p_",
    b"gl" + b"pat-",
    b"sk-" + b"proj-",
    b"AK" + b"IA",
    b"JL_" + b"API_KEY=",
    b"WANDB_" + b"API_KEY=",
)


class ExistingInterpolationRunError(FileExistsError):
    """An immutable interpolation run root already exists."""


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise InterpolationRescueError(f"refusing to overwrite artifact {path}")
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


def _append_progress(path: Path, phase: str, **details: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        **details,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise InterpolationRescueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise InterpolationRescueError(f"non-finite JSON constant {value!r}")


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except InterpolationRescueError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InterpolationRescueError(f"invalid generated JSON artifact {path}") from error
    if not isinstance(payload, dict):
        raise InterpolationRescueError(f"generated JSON artifact is not an object: {path}")
    return payload


def _exact_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise InterpolationRescueError(
            f"{name} fields differ: expected {sorted(expected)}, got {sorted(payload)}"
        )


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InterpolationRescueError(f"{name} must be an object")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise InterpolationRescueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InterpolationRescueError(f"{name} must be a positive integer")
    return value


def _content_tree_exclusions() -> dict[str, list[str]]:
    return {
        "relative_paths": list(TREE_EXCLUDED_RELATIVE_PATHS),
        "directory_names": list(TREE_EXCLUDED_DIRECTORY_NAMES),
        "directory_suffixes": list(TREE_EXCLUDED_DIRECTORY_SUFFIXES),
        "file_names": list(TREE_EXCLUDED_FILE_NAMES),
        "file_suffixes": list(TREE_EXCLUDED_FILE_SUFFIXES),
    }


def _excluded_directory(name: str) -> bool:
    return name in TREE_EXCLUDED_DIRECTORY_NAMES or name.endswith(
        TREE_EXCLUDED_DIRECTORY_SUFFIXES
    )


def _excluded_file(relative_path: str, name: str) -> bool:
    return (
        relative_path in TREE_EXCLUDED_RELATIVE_PATHS
        or name in TREE_EXCLUDED_FILE_NAMES
        or name.endswith(TREE_EXCLUDED_FILE_SUFFIXES)
    )


def _reject_credential_path(relative: Path) -> None:
    lowered_parts = tuple(part.casefold() for part in relative.parts)
    name = lowered_parts[-1]
    if any(part in _CREDENTIAL_DIRECTORY_NAMES for part in lowered_parts[:-1]):
        raise InterpolationRescueError(
            f"credential directory entered staged source tree: {relative.as_posix()}"
        )
    if (
        name in _CREDENTIAL_FILE_NAMES
        or name.startswith(".env.")
        or name.endswith(_CREDENTIAL_FILE_SUFFIXES)
    ):
        raise InterpolationRescueError(
            f"credential-like file entered staged source tree: {relative.as_posix()}"
        )


def _hash_source_file(path: Path, relative: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    tail = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            searchable = tail + block
            if any(marker in searchable for marker in _CREDENTIAL_CONTENT_MARKERS):
                raise InterpolationRescueError(
                    f"credential marker entered staged source tree: {relative.as_posix()}"
                )
            tail = searchable[-64:]
    return digest.hexdigest(), size


def compute_interpolation_content_tree(repository_root: str | Path) -> dict[str, Any]:
    """Recompute the frozen staged tree while pruning only canonical exclusions."""

    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise InterpolationRescueError("staged repository root is not a directory")
    entries: list[tuple[str, str, int]] = []
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = directory_path / name
            if _excluded_directory(name):
                continue
            relative = candidate.relative_to(root)
            if candidate.is_symlink():
                raise InterpolationRescueError(
                    f"symlink entered staged source tree: {relative.as_posix()}"
                )
            _reject_credential_path(relative / "placeholder")
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            candidate = directory_path / name
            relative = candidate.relative_to(root)
            relative_posix = relative.as_posix()
            if "\n" in relative_posix or "\r" in relative_posix:
                raise InterpolationRescueError("staged source path contains a line break")
            if _excluded_file(relative_posix, name):
                continue
            if candidate.is_symlink() or not candidate.is_file():
                raise InterpolationRescueError(
                    f"non-regular file entered staged source tree: {relative_posix}"
                )
            _reject_credential_path(relative)
            digest, size = _hash_source_file(candidate, relative)
            entries.append((relative_posix, digest, size))
    entries.sort(key=lambda item: item[0])
    tree_digest = hashlib.sha256()
    for relative, digest, _ in entries:
        tree_digest.update(f"{digest}  {relative}\n".encode())
    return {
        "schema_version": CONTENT_TREE_SCHEMA_VERSION,
        "sha256": tree_digest.hexdigest(),
        "file_count": len(entries),
        "bytes": sum(size for _, _, size in entries),
        "hash_method": CONTENT_TREE_HASH_METHOD,
        "exclusions": _content_tree_exclusions(),
    }


def _evidence_relative_path(root: Path, path: Path, expected: str, name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise InterpolationRescueError(
            f"{name} must be an existing regular file inside the staged repository"
        ) from error
    if not resolved.is_file() or resolved.is_symlink():
        raise InterpolationRescueError(f"{name} must be a non-symlink regular file")
    if relative.as_posix() != expected:
        raise InterpolationRescueError(
            f"{name} must be staged exactly at {expected!r}, got {relative.as_posix()!r}"
        )
    return resolved


def _validate_frozen_files(
    payload: object,
    *,
    name: str,
    runner_sha256: str,
) -> None:
    frozen = _require_mapping(payload, name)
    expected = {
        FROZEN_PROTOCOL_RELATIVE_PATH: INTERPOLATION_PROTOCOL_SHA256,
        FROZEN_RUNNER_RELATIVE_PATH: runner_sha256,
        FROZEN_SHARED_SELECTOR_RELATIVE_PATH: FROZEN_SHARED_SELECTOR_SHA256,
    }
    if dict(frozen) != expected:
        raise InterpolationRescueError(
            f"{name} differs from the exact runner/protocol/shared-selector hashes"
        )


def verify_interpolation_launch_provenance(
    *,
    repository_root: str | Path,
    attempt_preregistration_path: str | Path,
    source_snapshot_path: str | Path,
    run_id: str,
    jarvis_machine_id: int,
) -> dict[str, Any]:
    """Verify cross-bound preregistration and staged-tree provenance before scoring."""

    root = Path(repository_root).resolve(strict=True)
    attempt_path = _evidence_relative_path(
        root,
        Path(attempt_preregistration_path),
        ATTEMPT_PREREGISTRATION_RELATIVE_PATH,
        "attempt preregistration",
    )
    snapshot_path = _evidence_relative_path(
        root,
        Path(source_snapshot_path),
        SOURCE_SNAPSHOT_RELATIVE_PATH,
        "source snapshot",
    )
    attempt = _strict_json(attempt_path)
    snapshot = _strict_json(snapshot_path)
    _exact_keys(
        attempt,
        {
            "schema_version",
            "run_id",
            "status",
            "created_at",
            "created_before_remote_model_or_data_scoring",
            "protocol",
            "source_snapshot",
            "frozen_files",
            "prelaunch_inventory",
        },
        "attempt preregistration",
    )
    if attempt.get("schema_version") != ATTEMPT_PREREGISTRATION_SCHEMA_VERSION:
        raise InterpolationRescueError("unsupported interpolation attempt preregistration schema")
    if attempt.get("run_id") != run_id:
        raise InterpolationRescueError("attempt preregistration run_id mismatch")
    if attempt.get("status") != "frozen_before_remote_model_or_data_scoring":
        raise InterpolationRescueError("attempt preregistration is not frozen before scoring")
    if attempt.get("created_before_remote_model_or_data_scoring") is not True:
        raise InterpolationRescueError("attempt preregistration timing assertion is missing")
    if not isinstance(attempt.get("created_at"), str) or not attempt["created_at"]:
        raise InterpolationRescueError("attempt preregistration created_at is invalid")
    attempt_protocol = _require_mapping(attempt.get("protocol"), "attempt protocol")
    _exact_keys(attempt_protocol, {"path", "sha256"}, "attempt protocol")
    if attempt_protocol.get("path") != FROZEN_PROTOCOL_RELATIVE_PATH:
        raise InterpolationRescueError("attempt protocol path changed")
    if attempt_protocol.get("sha256") != INTERPOLATION_PROTOCOL_SHA256:
        raise InterpolationRescueError("attempt protocol SHA-256 changed")

    attempt_snapshot = _require_mapping(
        attempt.get("source_snapshot"), "attempt source snapshot"
    )
    _exact_keys(attempt_snapshot, {"path", "sha256"}, "attempt source snapshot")
    if attempt_snapshot.get("path") != SOURCE_SNAPSHOT_RELATIVE_PATH:
        raise InterpolationRescueError("attempt source-snapshot path changed")
    snapshot_sha256 = sha256_file(snapshot_path)
    if _require_sha256(
        attempt_snapshot.get("sha256"), "attempt source-snapshot SHA-256"
    ) != snapshot_sha256:
        raise InterpolationRescueError("attempt-to-snapshot SHA-256 mismatch")

    runner_path = root / FROZEN_RUNNER_RELATIVE_PATH
    protocol_path = root / FROZEN_PROTOCOL_RELATIVE_PATH
    shared_selector_path = root / FROZEN_SHARED_SELECTOR_RELATIVE_PATH
    if (
        not runner_path.is_file()
        or not protocol_path.is_file()
        or not shared_selector_path.is_file()
    ):
        raise InterpolationRescueError("staged runner, protocol, or shared selector is missing")
    running_runner_sha256 = sha256_file(Path(__file__).resolve())
    if sha256_file(runner_path) != running_runner_sha256:
        raise InterpolationRescueError("staged runner differs from the executing runner")
    if sha256_file(protocol_path) != INTERPOLATION_PROTOCOL_SHA256:
        raise InterpolationRescueError("staged scientific protocol SHA-256 mismatch")
    if sha256_file(shared_selector_path) != FROZEN_SHARED_SELECTOR_SHA256:
        raise InterpolationRescueError("staged shared rescue selector SHA-256 mismatch")
    _validate_frozen_files(
        attempt.get("frozen_files"),
        name="attempt frozen_files",
        runner_sha256=running_runner_sha256,
    )

    inventory = _require_mapping(attempt.get("prelaunch_inventory"), "prelaunch inventory")
    _exact_keys(
        inventory,
        {
            "captured_before_project_instance_creation",
            "protected_machine_ids",
            "project_machine_id",
            "fresh_project_instance",
        },
        "prelaunch inventory",
    )
    if inventory.get("captured_before_project_instance_creation") is not True:
        raise InterpolationRescueError("prelaunch inventory was not captured before creation")
    if inventory.get("fresh_project_instance") is not True:
        raise InterpolationRescueError("attempt does not identify a fresh project instance")
    if inventory.get("project_machine_id") != jarvis_machine_id:
        raise InterpolationRescueError("attempt project machine ID differs from the CLI")
    protected = inventory.get("protected_machine_ids")
    if not isinstance(protected, list) or not protected:
        raise InterpolationRescueError("prelaunch protected-machine denylist is missing")
    protected_ids = [
        _require_positive_integer(value, "protected machine ID") for value in protected
    ]
    if len(protected_ids) != len(set(protected_ids)):
        raise InterpolationRescueError("prelaunch protected-machine denylist has duplicates")
    if 463058 not in protected_ids:
        raise InterpolationRescueError("known protected machine 463058 is absent from the denylist")
    if jarvis_machine_id in protected_ids:
        raise InterpolationRescueError("project machine ID was pre-existing and is protected")

    _exact_keys(
        snapshot,
        {
            "schema_version",
            "run_id",
            "created_at",
            "created_before_remote_launch_or_model_scoring",
            "git_commit",
            "git_dirty",
            "content_tree",
            "frozen_files",
            "staging_policy",
        },
        "source snapshot",
    )
    if snapshot.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA_VERSION:
        raise InterpolationRescueError("unsupported interpolation source-snapshot schema")
    if snapshot.get("run_id") != run_id:
        raise InterpolationRescueError("source snapshot run_id mismatch")
    if snapshot.get("created_before_remote_launch_or_model_scoring") is not True:
        raise InterpolationRescueError("source snapshot timing assertion is missing")
    if not isinstance(snapshot.get("created_at"), str) or not snapshot["created_at"]:
        raise InterpolationRescueError("source snapshot created_at is invalid")
    if not isinstance(snapshot.get("git_commit"), str) or re.fullmatch(
        r"[0-9a-f]{40}", snapshot["git_commit"]
    ) is None:
        raise InterpolationRescueError("source snapshot git commit is invalid")
    if type(snapshot.get("git_dirty")) is not bool:
        raise InterpolationRescueError("source snapshot git_dirty must be boolean")
    _validate_frozen_files(
        snapshot.get("frozen_files"),
        name="source snapshot frozen_files",
        runner_sha256=running_runner_sha256,
    )
    staging_policy = _require_mapping(snapshot.get("staging_policy"), "staging policy")
    expected_staging_policy = {
        "credentials_included": False,
        "caches_included": False,
        "official_test_artifacts_included": False,
        "optimizer_state_included": False,
    }
    if dict(staging_policy) != expected_staging_policy:
        raise InterpolationRescueError("source snapshot staging policy changed")

    recorded_tree = _require_mapping(snapshot.get("content_tree"), "content tree")
    _exact_keys(
        recorded_tree,
        {
            "schema_version",
            "sha256",
            "file_count",
            "bytes",
            "hash_method",
            "exclusions",
        },
        "content tree",
    )
    if recorded_tree.get("schema_version") != CONTENT_TREE_SCHEMA_VERSION:
        raise InterpolationRescueError("source snapshot content-tree schema changed")
    if recorded_tree.get("hash_method") != CONTENT_TREE_HASH_METHOD:
        raise InterpolationRescueError("source snapshot content-tree method changed")
    if recorded_tree.get("exclusions") != _content_tree_exclusions():
        raise InterpolationRescueError("source snapshot content-tree exclusions changed")
    _require_sha256(recorded_tree.get("sha256"), "content-tree SHA-256")
    _require_positive_integer(recorded_tree.get("file_count"), "content-tree file count")
    _require_positive_integer(recorded_tree.get("bytes"), "content-tree bytes")
    recomputed_tree = compute_interpolation_content_tree(root)
    if dict(recorded_tree) != recomputed_tree:
        raise InterpolationRescueError(
            "staged content tree differs from the preregistered source snapshot"
        )
    return {
        "schema_version": "barun-interpolation-provenance-verification-v1",
        "verified_before_model_or_data_scoring": True,
        "run_id": run_id,
        "jarvis_machine_id": jarvis_machine_id,
        "attempt_preregistration_sha256": sha256_file(attempt_path),
        "source_snapshot_sha256": snapshot_sha256,
        "protocol_sha256": INTERPOLATION_PROTOCOL_SHA256,
        "shared_selector_sha256": FROZEN_SHARED_SELECTOR_SHA256,
        "runner_sha256": running_runner_sha256,
        "content_tree": recomputed_tree,
        "protected_machine_ids": protected_ids,
        "credential_markers_or_paths_found": 0,
        "caches_included_in_tree_accounting": False,
    }


def _force_math_sdpa() -> dict[str, bool | None]:
    backend = torch.backends.cuda
    required = (
        ("flash", "enable_flash_sdp", "flash_sdp_enabled", False),
        (
            "memory_efficient",
            "enable_mem_efficient_sdp",
            "mem_efficient_sdp_enabled",
            False,
        ),
        ("math", "enable_math_sdp", "math_sdp_enabled", True),
    )
    evidence: dict[str, bool | None] = {}
    for label, setter_name, getter_name, expected in required:
        setter = getattr(backend, setter_name, None)
        getter = getattr(backend, getter_name, None)
        if not callable(setter) or not callable(getter):
            raise TypeError(f"PyTorch cannot configure and verify {label} SDPA")
        setter(expected)
        actual = getter()
        if type(actual) is not bool or actual is not expected:
            raise RuntimeError(f"failed to set {label} SDPA to {expected}: got {actual!r}")
        evidence[label] = actual
    cudnn_setter = getattr(backend, "enable_cudnn_sdp", None)
    cudnn_getter = getattr(backend, "cudnn_sdp_enabled", None)
    if cudnn_setter is None and cudnn_getter is None:
        evidence["cudnn"] = None
    elif callable(cudnn_setter) and callable(cudnn_getter):
        cudnn_setter(False)
        actual = cudnn_getter()
        if type(actual) is not bool or actual is not False:
            raise RuntimeError(f"failed to disable cuDNN SDPA: got {actual!r}")
        evidence["cudnn"] = actual
    else:
        raise RuntimeError("PyTorch exposes a partial, unverifiable cuDNN SDPA interface")
    return evidence


def _runtime_environment() -> dict[str, Any]:
    actual_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual_cublas != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG changed after import")
    if not _CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT:
        raise RuntimeError("cuBLAS determinism was not configured before importing torch")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before interpolation rescue preflight")
    sdpa = _force_math_sdpa()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the interpolation rescue requires CUDA with bfloat16")
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "device": "cuda",
        "dtype": "torch.bfloat16",
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cublas_workspace_config": actual_cublas,
        "sdpa_backends": sdpa,
        "configured_before_torch_import": True,
        "cuda_initialized_before_preflight": False,
    }


def _run_focused_tests(
    repository_root: Path,
    essential: Path,
    protocol: Mapping[str, Any],
) -> None:
    tests = protocol.get("remote_preflight_tests")
    if not isinstance(tests, list) or not tests or any(not isinstance(item, str) for item in tests):
        raise InterpolationRescueError("remote preflight test list is invalid")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path = essential / "interpolation-preflight-tests.log"
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"focused repository tests failed with exit code {result.returncode}")


def _presto_firewall(prepared: Any, audit: Mapping[str, Any]) -> dict[str, Any]:
    official = audit.get("official_test")
    expected = {
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
    if not isinstance(official, Mapping) or any(
        official.get(key) != value for key, value in expected.items()
    ):
        raise InterpolationRescueError("official PRESTO test members did not remain opaque")
    if set(prepared.hashes) != {"train", "dev", "audit"}:
        raise InterpolationRescueError("PRESTO adapter exported an unexpected split")
    source = audit.get("source")
    members = source.get("members") if isinstance(source, Mapping) else None
    if not isinstance(members, Mapping):
        raise InterpolationRescueError("PRESTO audit lacks member-access evidence")
    for name in ("presto_dataset.jsonl", "presto_test.jsonl"):
        member = members.get(name)
        if (
            not isinstance(member, Mapping)
            or member.get("runtime_open_count") != 0
            or member.get("runtime_bytes_read") != 0
        ):
            raise InterpolationRescueError(f"sealed PRESTO archive member was read: {name}")
    return {
        **expected,
        "official_test_rows_opaque_unparsed": prepared.official_test_rows,
    }


def _prepare_presto_development(
    *,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    candidate_dir: Path,
    run_root: Path,
    essential: Path,
) -> tuple[Path, str, dict[str, Any]]:
    contract = protocol["presto_evaluation"]
    if not isinstance(contract, Mapping):
        raise InterpolationRescueError("PRESTO protocol section is invalid")
    archive = (
        args.presto_archive.resolve()
        if args.presto_archive is not None
        else download_pinned_archive(run_root / "cache")
    )
    if sha256_file(archive) != PRESTO_ARCHIVE_SHA256:
        raise InterpolationRescueError("PRESTO archive differs from the pinned source")
    tokenizer = Tokenizer.from_file(str(candidate_dir / "tokenizer.json"))
    prepared = prepare_presto(
        archive,
        run_root / "prepared",
        tokenizer=tokenizer,
        tokenizer_identity=TokenizerIdentity(
            identifier="harrrshall/BarunLM-35M",
            revision="ef3e483a9fd7d906ecf2a7929babeffaf82d1d16",
            sha256=str(endpoint_hashes(protocol, "candidate_v2")["tokenizer.json"]),
        ),
    )
    expected_hashes = {
        "train": "49a553ea37a981f2a8009ce8bc6575fd56182d8f959352c9815fcc463b832e06",
        "dev": contract.get("derived_dev_manifest_sha256"),
        "audit": contract.get("derived_audit_sha256"),
    }
    if dict(prepared.hashes) != expected_hashes:
        raise InterpolationRescueError("derived PRESTO artifact hashes changed")
    if prepared.dev_rows != contract.get("rows") or PRESTO_REVISION != contract.get("revision"):
        raise InterpolationRescueError("PRESTO development population changed")
    audit = _strict_json(prepared.audit_path)
    firewall = _presto_firewall(prepared, audit)
    target = audit.get("tokenization", {}).get("per_split", {}).get("dev", {}).get("target", {})
    if not isinstance(target, Mapping) or target.get("max") != 263:
        raise InterpolationRescueError("PRESTO audited target maximum changed")

    data_dir = essential / "interpolation-data"
    data_dir.mkdir(exist_ok=False)
    dev_path = data_dir / "interpolation-presto-dev.jsonl"
    audit_path = data_dir / "interpolation-presto-audit.json"
    shutil.copy2(prepared.dev_manifest, dev_path)
    shutil.copy2(prepared.audit_path, audit_path)
    if sha256_file(dev_path) != expected_hashes["dev"]:
        raise InterpolationRescueError("PRESTO development manifest changed during staging")
    if sha256_file(audit_path) != expected_hashes["audit"]:
        raise InterpolationRescueError("PRESTO audit changed during staging")
    return dev_path, str(expected_hashes["dev"]), firewall


def _validate_remote_contract(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> None:
    remote = protocol.get("remote_execution")
    if not isinstance(remote, Mapping):
        raise InterpolationRescueError("remote execution contract is missing")
    protected = remote.get("known_protected_machine_ids")
    if not isinstance(protected, list) or args.jarvis_machine_id in protected:
        raise InterpolationRescueError("refusing a known protected JarvisLabs machine ID")
    if environment.get("cuda_device_name") != remote.get("gpu"):
        raise InterpolationRescueError("runtime GPU differs from the frozen H200 environment")
    expected_mobile = protocol.get("mobile_evaluation")
    expected_environment = (
        expected_mobile.get("environment") if isinstance(expected_mobile, Mapping) else None
    )
    if not isinstance(expected_environment, Mapping):
        raise InterpolationRescueError("Mobile reference environment is missing")
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise InterpolationRescueError(
                f"runtime differs from the frozen Mobile environment at {key}: "
                f"expected {expected!r}, got {environment.get(key)!r}"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    repository_root = args.repository_root.resolve()
    artifact_root = args.artifact_root.resolve()
    try:
        artifact_root.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise InterpolationRescueError(
            "artifact root must remain outside the hash-bound staged repository"
        )
    run_root = artifact_root / args.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        run_root.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ExistingInterpolationRunError(
            f"refusing to overwrite interpolation run root {run_root}"
        ) from error
    essential = run_root / "export" / "essential"
    essential.mkdir(parents=True)
    progress = essential / "interpolation-progress.jsonl"
    _append_progress(progress, "interpolation_run_root_claimed", run_root=str(run_root))

    provenance = verify_interpolation_launch_provenance(
        repository_root=repository_root,
        attempt_preregistration_path=args.attempt_preregistration,
        source_snapshot_path=args.source_snapshot,
        run_id=args.run_id,
        jarvis_machine_id=args.jarvis_machine_id,
    )
    shutil.copy2(
        args.attempt_preregistration.resolve(),
        essential / "interpolation-attempt-preregistration.json",
    )
    shutil.copy2(
        args.source_snapshot.resolve(),
        essential / "interpolation-source-snapshot.json",
    )
    _write_json(
        essential / "interpolation-provenance-verification.json",
        provenance,
    )
    _append_progress(
        progress,
        "interpolation_launch_provenance_verified",
        content_tree_sha256=provenance["content_tree"]["sha256"],
    )

    protocol = load_interpolation_protocol()
    shutil.copy2(INTERPOLATION_PROTOCOL_PATH, essential / "interpolation-protocol.json")
    environment = _runtime_environment()
    environment["jarvis_machine_id"] = args.jarvis_machine_id
    _validate_remote_contract(args, protocol, environment)
    _write_json(essential / "interpolation-environment.json", environment)
    _append_progress(progress, "interpolation_environment_verified")

    _run_focused_tests(repository_root, essential, protocol)
    _append_progress(progress, "interpolation_focused_tests_passed")

    candidate_dir = args.candidate_checkpoint_dir.resolve()
    presto_dir = args.presto_checkpoint_dir.resolve()
    candidate_hashes = endpoint_hashes(protocol, "candidate_v2")
    presto_hashes = endpoint_hashes(protocol, "presto")
    verify_checkpoint(candidate_dir, expected_sha256=candidate_hashes)
    verify_checkpoint(presto_dir, expected_sha256=presto_hashes)
    _append_progress(progress, "interpolation_endpoints_verified")

    dev_manifest, dev_sha256, presto_firewall = _prepare_presto_development(
        args=args,
        protocol=protocol,
        candidate_dir=candidate_dir,
        run_root=run_root,
        essential=essential,
    )
    _append_progress(progress, "interpolation_presto_development_and_firewall_verified")

    arms_root = essential / "interpolation-arms"
    arms_root.mkdir()
    arm_results: list[dict[str, Any]] = []
    for arm in interpolation_arms(protocol):
        arm_id = str(arm["arm_id"])
        arm_root = arms_root / arm_id
        arm_root.mkdir()
        checkpoint = interpolate_checkpoint(
            candidate_dir,
            presto_dir,
            arm_root / "interpolation-checkpoint",
            arm=arm,
            protocol=protocol,
            run_id=args.run_id,
        )
        _append_progress(
            progress,
            "interpolation_checkpoint_created",
            arm_id=arm_id,
            model_sha256=checkpoint["file_sha256"]["model.safetensors"],
        )

        mobile_result = run_mobile_regression(
            checkpoint_dir=checkpoint["directory"],
            checkpoint_hashes_path=(
                Path(checkpoint["directory"]) / "checkpoint_manifest.json"
            ),
            output_dir=arm_root / "interpolation-mobile-evaluation",
            repository_root=repository_root,
            preregistration_path=(repository_root / "configs" / "mobile_regression_v1.json"),
            runtime_environment=environment,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _append_progress(
            progress,
            "interpolation_mobile_evaluation_completed",
            arm_id=arm_id,
            passed=mobile_result["gate"]["passed"],
        )

        presto_root = arm_root / "interpolation-presto-evaluation"
        predictions = presto_root / "interpolation-predictions.jsonl"
        presto_contract = protocol["presto_evaluation"]
        if not isinstance(presto_contract, Mapping):
            raise InterpolationRescueError("PRESTO evaluation contract is invalid")
        generation = generate_manifest(
            checkpoint_dir=checkpoint["directory"],
            manifest_path=dev_manifest,
            manifest_sha256=dev_sha256,
            predictions_path=predictions,
            device_name="cuda",
            batch_size=int(presto_contract["batch_size"]),
            max_new_tokens=int(presto_contract["max_new_tokens"]),
            expected_checkpoint_sha256=checkpoint["file_sha256"],
        )
        score_paths = write_presto_scores(
            dev_manifest,
            predictions,
            presto_root / "interpolation-scores",
        )
        presto_aggregate = _strict_json(score_paths["aggregate"])
        joint_gate = evaluate_joint_gate(mobile_result, presto_aggregate, protocol)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        arm_result = {
            "schema_version": "barun-interpolation-arm-result-v1",
            "arm_id": arm_id,
            "alpha": checkpoint["alpha"],
            "checkpoint": checkpoint,
            "mobile_result": mobile_result,
            "presto_aggregate": presto_aggregate,
            "presto_generation": generation.to_dict(),
            "presto_sample_scores": {
                "path": str(score_paths["samples"]),
                "sha256": sha256_file(score_paths["samples"]),
            },
            "joint_gate": joint_gate,
            "both_frozen_evaluations_completed_once": True,
        }
        _write_json(arm_root / "interpolation-arm-result.json", arm_result)
        arm_results.append(arm_result)
        _append_progress(
            progress,
            "interpolation_arm_fully_evaluated",
            arm_id=arm_id,
            joint_gate_passed=joint_gate["passed"],
        )

    selection = select_interpolation_arm(arm_results, protocol)
    elapsed = time.monotonic() - started
    result = {
        "schema_version": INTERPOLATION_RESULT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "jarvis_machine_id": args.jarvis_machine_id,
        "protocol_sha256": INTERPOLATION_PROTOCOL_SHA256,
        "provenance": provenance,
        "trajectory_endpoint_sha256": {
            "candidate_v2": candidate_hashes,
            "presto": presto_hashes,
        },
        "trial_accounting": {
            "frozen_arms": len(arm_results),
            "mobile_generations": len(arm_results),
            "presto_generations": len(arm_results),
            "adaptive_followup_alphas": 0,
            "every_arm_received_both_evaluations": all(
                arm["both_frozen_evaluations_completed_once"] for arm in arm_results
            ),
            "new_joint_development_selection_trials": len(arm_results),
        },
        "data": {
            "presto_revision": PRESTO_REVISION,
            "presto_archive_sha256": PRESTO_ARCHIVE_SHA256,
            "presto_dev_manifest": str(dev_manifest),
            "presto_dev_manifest_sha256": dev_sha256,
            "presto_official_test_firewall": presto_firewall,
            "mobile_official_test_artifacts_accessed": [],
        },
        "arms": arm_results,
        "selection": selection,
        "environment": environment,
        "estimated_cost": args.hourly_cost * elapsed / 3600,
        "elapsed_seconds": elapsed,
        "official_test_evaluations": 0,
        "limitations": list(protocol["limitations"]),
    }
    _write_json(essential / "interpolation-result.json", result)
    _write_json(
        essential / "interpolation-bundle-manifest.json",
        {
            "schema_version": "barun-interpolation-essential-bundle-v1",
            "run_id": args.run_id,
            "sample_level_mobile_and_presto_evidence_included": True,
            "all_three_interpolation_checkpoints_included": True,
            "launch_provenance_included": True,
            "optimizer_state_included": False,
            "official_test_artifacts_included": False,
        },
    )
    if any(path.name == "optimizer.pt" for path in essential.rglob("*")):
        raise InterpolationRescueError("optimizer state entered interpolation evidence")
    _append_progress(
        progress,
        "interpolation_run_completed",
        selected_arm_id=selection["selected_arm_id"],
    )
    _write_json(
        essential / "artifact-sha256.json",
        {
            "schema_version": "barun-interpolation-artifacts-v1",
            "file_sha256": file_manifest(essential),
        },
    )
    return result


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _run_id(value: str) -> str:
    if RUN_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must match YYYYMMDD-HHMM-lowercase-name-sN")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=_run_id, required=True)
    parser.add_argument("--jarvis-machine-id", type=_positive_int, required=True)
    parser.add_argument("--attempt-preregistration", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--presto-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--presto-archive", type=Path)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=Path("/home/barun-artifacts"))
    parser.add_argument("--hourly-cost", type=_nonnegative_float, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = args.artifact_root.resolve() / args.run_id
    essential = run_root / "export" / "essential"
    try:
        result = run(args)
    except ExistingInterpolationRunError:
        raise
    except BaseException as error:
        essential.mkdir(parents=True, exist_ok=True)
        _append_progress(
            essential / "interpolation-progress.jsonl",
            "interpolation_run_failed",
            error_type=type(error).__name__,
            message=str(error),
        )
        failure = {
            "at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        failure_path = essential / "interpolation-failure.json"
        if not failure_path.exists():
            _write_json(failure_path, failure)
        artifact_path = essential / "artifact-sha256.json"
        if not artifact_path.exists():
            _write_json(
                artifact_path,
                {
                    "schema_version": "barun-interpolation-artifacts-v1",
                    "file_sha256": file_manifest(essential),
                },
            )
        raise
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
