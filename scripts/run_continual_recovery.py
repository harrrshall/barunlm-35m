"""Run the single preregistered BarunAction Mobile/PRESTO continual-recovery stage."""

from __future__ import annotations

# Configure deterministic cuBLAS before importing torch or project modules.
import os

DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
_CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT = (
    os.environ.get("CUBLAS_WORKSPACE_CONFIG") == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
)

import argparse
import gc
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

import torch
from tokenizers import Tokenizer

from barunlm.datasets.presto import (
    PRESTO_REVISION,
    TokenizerIdentity,
    download_pinned_archive,
    prepare_presto,
)
from barunlm.evaluation.generation import generate_manifest, verify_checkpoint
from barunlm.evaluation.mobile_regression import run_mobile_regression
from barunlm.evaluation.presto import write_scores
from barunlm.recovery.continual_recovery import (
    RECOVERY_CONFIG_SHA256,
    RecoveryError,
    evaluate_recovery_presto_gate,
    load_recovery_config,
    materialize_recovery_view,
    train_continual_recovery,
)
from barunlm.training.data import load_manifest, sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
RECOVERY_CONFIG = REPOSITORY_ROOT / "configs" / "continual_recovery_v1.json"
MOBILE_REGRESSION_CONFIG = REPOSITORY_ROOT / "configs" / "mobile_regression_v1.json"
SHARED_SELECTOR_PATH = REPOSITORY_ROOT / "configs" / "parallel_rescue_selection_v1.json"
SHARED_SELECTOR_RELATIVE_PATH = "configs/parallel_rescue_selection_v1.json"
SHARED_SELECTOR_SHA256 = (
    "68bb6132a486dfbd6c9ad478db9c87ae9f2350bfc1d302193169fd7f5c290130"
)
SHARED_SELECTOR_SCHEMA_VERSION = "barun-parallel-rescue-selection-v1"
TOKENIZER_IDENTIFIER = "harrrshall/BarunLM-35M"
TOKENIZER_REVISION = "ef3e483a9fd7d906ecf2a7929babeffaf82d1d16"
ATTEMPT_SCHEMA_VERSION = "barun-continual-recovery-attempt-v1"
SNAPSHOT_SCHEMA_VERSION = "barun-continual-recovery-source-snapshot-v1"
CONTENT_TREE_SCHEMA_VERSION = "barun-staged-content-tree-v1"
CONTENT_TREE_HASH_METHOD = (
    "SHA-256 of UTF-8 lines '<file SHA-256><two spaces><relative POSIX "
    "path><newline>' in relative-path sort order"
)
PRESTO_CHECKPOINT_MANIFEST_SHA256 = (
    "90e50f316456947bbee10b710b6f828f1ca8955ed0e43998b15adb0e09f46e23"
)
TREE_EXCLUDED_DIRECTORY_NAMES = tuple(
    sorted(
        {
            ".cache",
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "venv",
            "wandb",
        }
    )
)
TREE_EXCLUDED_DIRECTORY_SUFFIXES = (".egg-info",)
TREE_EXCLUDED_FILE_NAMES = (".DS_Store",)
TREE_EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo")
SENSITIVE_EXACT_FILE_NAMES = frozenset(
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
SENSITIVE_FILE_SUFFIXES = (".key", ".p12", ".pem", ".pfx")
RUN_ID_PATTERN = re.compile(r"\d{8}-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*-s\d+")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ExistingRecoveryRunError(FileExistsError):
    """An immutable artifact root already owns the requested run ID."""


def _fail(message: str) -> NoReturn:
    raise RecoveryError(message)


def _reject_provenance_constant(value: str) -> NoReturn:
    _fail(f"provenance JSON contains non-finite constant {value!r}")


def _unique_provenance_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"provenance JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_provenance_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_provenance_object,
            parse_constant=_reject_provenance_constant,
        )
    except RecoveryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryError(f"cannot read strict {label} JSON from {path}: {error}") from error
    if not isinstance(payload, dict):
        _fail(f"{label} must contain a JSON object")
    return payload


def _exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} fields changed: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _required_sha256(value: object, label: str) -> str:
    digest = _required_string(value, label)
    if SHA256_PATTERN.fullmatch(digest) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _required_bool(value: object, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        _fail(f"{label} must be {expected}")
    return expected


def _required_int(value: object, expected: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        _fail(f"{label} must equal {expected}")
    return value


def _required_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(f"{label} must be a positive integer")
    return value


def _strict_hash_map(value: object, label: str) -> dict[str, str]:
    hashes = _exact_keys(
        value,
        {"barun_config.json", "model.safetensors", "tokenizer.json"},
        label,
    )
    return {name: _required_sha256(hashes[name], f"{label}.{name}") for name in sorted(hashes)}


def _validate_checkpoint_contract(value: object, label: str) -> dict[str, Any]:
    contract = _exact_keys(
        value,
        {
            "checkpoint_manifest_sha256",
            "file_sha256",
            "parent_file_sha256",
            "parent_run_id",
            "path",
            "source_run_id",
        },
        label,
    )
    return {
        "source_run_id": _required_string(contract["source_run_id"], f"{label}.source_run_id"),
        "parent_run_id": _required_string(contract["parent_run_id"], f"{label}.parent_run_id"),
        "path": _required_string(contract["path"], f"{label}.path"),
        "checkpoint_manifest_sha256": _required_sha256(
            contract["checkpoint_manifest_sha256"], f"{label}.checkpoint_manifest_sha256"
        ),
        "file_sha256": _strict_hash_map(contract["file_sha256"], f"{label}.file_sha256"),
        "parent_file_sha256": _strict_hash_map(
            contract["parent_file_sha256"], f"{label}.parent_file_sha256"
        ),
    }


def _validate_attempt_schema(payload: Mapping[str, Any]) -> dict[str, Any]:
    attempt = _exact_keys(
        payload,
        {
            "claim_limits",
            "created_before_remote_launch_or_data_prep",
            "decision",
            "evaluation",
            "hypothesis",
            "input_checkpoint",
            "recovery_config",
            "registered_at",
            "run_id",
            "runner",
            "prelaunch_inventory",
            "schema_version",
            "shared_selector",
            "source_snapshot",
            "status",
        },
        "attempt preregistration",
    )
    if attempt["schema_version"] != ATTEMPT_SCHEMA_VERSION:
        _fail("unsupported continual-recovery attempt schema")
    if attempt["status"] != "frozen_before_remote_data_prep_training_or_scoring":
        _fail("continual-recovery attempt was not frozen before execution")
    _required_bool(
        attempt["created_before_remote_launch_or_data_prep"],
        True,
        "attempt.created_before_remote_launch_or_data_prep",
    )
    try:
        registered = datetime.fromisoformat(_required_string(attempt["registered_at"], "registered_at"))
    except ValueError as error:
        raise RecoveryError("attempt.registered_at must be ISO-8601") from error
    if registered.tzinfo is None:
        _fail("attempt.registered_at must include a timezone")
    _required_string(attempt["hypothesis"], "attempt.hypothesis")
    _required_string(attempt["decision"], "attempt.decision")
    limits = attempt["claim_limits"]
    if not isinstance(limits, list) or not limits or any(
        not isinstance(item, str) or not item for item in limits
    ):
        _fail("attempt.claim_limits must be a nonempty string array")
    recovery_config = _exact_keys(
        attempt["recovery_config"], {"path", "sha256"}, "attempt.recovery_config"
    )
    shared_selector = _exact_keys(
        attempt["shared_selector"], {"path", "sha256"}, "attempt.shared_selector"
    )
    source_snapshot = _exact_keys(
        attempt["source_snapshot"], {"path", "sha256"}, "attempt.source_snapshot"
    )
    runner = _exact_keys(attempt["runner"], {"path", "sha256"}, "attempt.runner")
    inventory = _exact_keys(
        attempt["prelaunch_inventory"],
        {
            "captured_before_project_instance_creation",
            "fresh_project_instance",
            "project_machine_id",
            "protected_machine_ids",
        },
        "attempt.prelaunch_inventory",
    )
    _required_bool(
        inventory["captured_before_project_instance_creation"],
        True,
        "attempt.prelaunch_inventory.captured_before_project_instance_creation",
    )
    _required_bool(
        inventory["fresh_project_instance"],
        True,
        "attempt.prelaunch_inventory.fresh_project_instance",
    )
    project_machine_id = _required_positive_int(
        inventory["project_machine_id"],
        "attempt.prelaunch_inventory.project_machine_id",
    )
    protected = inventory["protected_machine_ids"]
    if not isinstance(protected, list) or not protected:
        _fail("attempt prelaunch protected-machine denylist is missing")
    protected_machine_ids = [
        _required_positive_int(
            value,
            "attempt.prelaunch_inventory protected machine ID",
        )
        for value in protected
    ]
    if len(protected_machine_ids) != len(set(protected_machine_ids)):
        _fail("attempt prelaunch protected-machine denylist has duplicates")
    if 463058 not in protected_machine_ids:
        _fail("known protected machine 463058 is absent from the attempt denylist")
    if project_machine_id in protected_machine_ids:
        _fail("attempt project machine ID was pre-existing and is protected")
    evaluation = _exact_keys(
        attempt["evaluation"],
        {
            "mobile_development_rows",
            "official_mobile_evaluation_rows_read",
            "official_presto_test_rows_read",
            "presto_development_rows",
            "terminal_rounds",
        },
        "attempt.evaluation",
    )
    _required_int(evaluation["mobile_development_rows"], 756, "Mobile development rows")
    _required_int(evaluation["presto_development_rows"], 14_288, "PRESTO development rows")
    _required_int(evaluation["terminal_rounds"], 1, "terminal rounds")
    _required_int(
        evaluation["official_mobile_evaluation_rows_read"], 0, "official Mobile rows read"
    )
    _required_int(
        evaluation["official_presto_test_rows_read"], 0, "official PRESTO rows read"
    )
    return {
        "run_id": _required_string(attempt["run_id"], "attempt.run_id"),
        "recovery_config": {
            "path": _required_string(recovery_config["path"], "attempt recovery config path"),
            "sha256": _required_sha256(
                recovery_config["sha256"], "attempt recovery config hash"
            ),
        },
        "shared_selector": {
            "path": _required_string(
                shared_selector["path"], "attempt shared selector path"
            ),
            "sha256": _required_sha256(
                shared_selector["sha256"], "attempt shared selector hash"
            ),
        },
        "source_snapshot": {
            "path": _required_string(source_snapshot["path"], "attempt snapshot path"),
            "sha256": _required_sha256(source_snapshot["sha256"], "attempt snapshot hash"),
        },
        "runner": {
            "path": _required_string(runner["path"], "attempt runner path"),
            "sha256": _required_sha256(runner["sha256"], "attempt runner hash"),
        },
        "input_checkpoint": _validate_checkpoint_contract(
            attempt["input_checkpoint"], "attempt.input_checkpoint"
        ),
        "prelaunch_inventory": {
            "captured_before_project_instance_creation": True,
            "protected_machine_ids": protected_machine_ids,
            "project_machine_id": project_machine_id,
            "fresh_project_instance": True,
        },
    }


def _validate_snapshot_schema(payload: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = _exact_keys(
        payload,
        {
            "content_tree",
            "created_before_remote_launch_or_data_prep",
            "frozen_files",
            "input_checkpoint",
            "run_id",
            "schema_version",
            "stage_root",
            "staging_policy",
        },
        "source snapshot",
    )
    if snapshot["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        _fail("unsupported continual-recovery source-snapshot schema")
    _required_bool(
        snapshot["created_before_remote_launch_or_data_prep"],
        True,
        "snapshot.created_before_remote_launch_or_data_prep",
    )
    if snapshot["stage_root"] != ".":
        _fail("snapshot.stage_root must be '.'")
    tree = _exact_keys(
        snapshot["content_tree"],
        {
            "content_bytes",
            "excluded_directory_names",
            "excluded_directory_suffixes",
            "excluded_exact_paths",
            "excluded_file_names",
            "excluded_file_suffixes",
            "file_count",
            "hash_method",
            "schema_version",
            "sha256",
        },
        "snapshot.content_tree",
    )
    if tree["schema_version"] != CONTENT_TREE_SCHEMA_VERSION:
        _fail("unsupported staged content-tree schema")
    if tree["hash_method"] != CONTENT_TREE_HASH_METHOD:
        _fail("staged content-tree hash method changed")
    policies = (
        ("excluded_directory_names", TREE_EXCLUDED_DIRECTORY_NAMES),
        ("excluded_directory_suffixes", TREE_EXCLUDED_DIRECTORY_SUFFIXES),
        ("excluded_file_names", TREE_EXCLUDED_FILE_NAMES),
        ("excluded_file_suffixes", TREE_EXCLUDED_FILE_SUFFIXES),
    )
    for field, expected in policies:
        if tree[field] != list(expected):
            _fail(f"snapshot.content_tree.{field} changed")
    exact_exclusions = tree["excluded_exact_paths"]
    if (
        not isinstance(exact_exclusions, list)
        or len(exact_exclusions) != 2
        or exact_exclusions != sorted(exact_exclusions)
        or any(not isinstance(path, str) or not path for path in exact_exclusions)
    ):
        _fail("snapshot exact exclusions must be two sorted nonempty paths")
    file_count = tree["file_count"]
    content_bytes = tree["content_bytes"]
    if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count < 1:
        _fail("snapshot content-tree file_count must be positive")
    if isinstance(content_bytes, bool) or not isinstance(content_bytes, int) or content_bytes < 1:
        _fail("snapshot content-tree content_bytes must be positive")
    frozen = _exact_keys(
        snapshot["frozen_files"],
        {
            "configs/continual_recovery_v1.json",
            SHARED_SELECTOR_RELATIVE_PATH,
            "scripts/run_continual_recovery.py",
        },
        "snapshot.frozen_files",
    )
    policy = _exact_keys(
        snapshot["staging_policy"],
        {
            "cache_directories_present_before_launch",
            "credentials_present",
            "official_mobile_evaluation_present",
            "official_presto_test_present",
            "symlinks_present_before_launch",
        },
        "snapshot.staging_policy",
    )
    for field in policy:
        _required_bool(policy[field], False, f"snapshot.staging_policy.{field}")
    return {
        "run_id": _required_string(snapshot["run_id"], "snapshot.run_id"),
        "content_tree": {
            "sha256": _required_sha256(tree["sha256"], "snapshot content-tree hash"),
            "file_count": file_count,
            "content_bytes": content_bytes,
            "excluded_exact_paths": list(exact_exclusions),
        },
        "frozen_files": {
            str(path): _required_sha256(digest, f"snapshot frozen file {path}")
            for path, digest in frozen.items()
        },
        "input_checkpoint": _validate_checkpoint_contract(
            snapshot["input_checkpoint"], "snapshot.input_checkpoint"
        ),
    }


def _declared_stage_path(stage_root: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        _fail(f"{label} must be a normalized relative POSIX path")
    candidate = stage_root / relative
    try:
        candidate.relative_to(stage_root)
    except ValueError as error:  # pragma: no cover - guarded by normalized relative path
        raise RecoveryError(f"{label} escapes the staged source root") from error
    return candidate


def _relative_stage_path(path: Path, stage_root: Path, label: str) -> str:
    if path.is_symlink():
        _fail(f"{label} cannot be a symlink")
    try:
        return path.resolve(strict=True).relative_to(stage_root).as_posix()
    except (OSError, ValueError) as error:
        raise RecoveryError(f"{label} must be an existing file inside {stage_root}") from error


def _excluded_directory(name: str) -> bool:
    return name in TREE_EXCLUDED_DIRECTORY_NAMES or name.endswith(
        TREE_EXCLUDED_DIRECTORY_SUFFIXES
    )


def _excluded_file(name: str) -> bool:
    return name in TREE_EXCLUDED_FILE_NAMES or name.endswith(TREE_EXCLUDED_FILE_SUFFIXES)


def _sensitive_file(name: str) -> bool:
    folded = name.casefold()
    return (
        folded == ".env"
        or folded.startswith((".env.", "credentials.", "secrets."))
        or folded in SENSITIVE_EXACT_FILE_NAMES
        or folded.endswith(SENSITIVE_FILE_SUFFIXES)
    )


def _recompute_staged_content_tree(
    stage_root: Path, *, excluded_exact_paths: Sequence[str]
) -> dict[str, Any]:
    """Hash staged regular files while excluding only documented runtime/provenance paths."""

    exact = set(excluded_exact_paths)
    records: list[tuple[str, str, int]] = []
    excluded_runtime_paths: list[str] = []
    for current, directory_names, file_names in os.walk(stage_root, topdown=True):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current_path / name
            relative = candidate.relative_to(stage_root).as_posix()
            if candidate.is_symlink():
                _fail(f"staged source contains a symlink: {relative}")
            if _excluded_directory(name):
                excluded_runtime_paths.append(relative + "/")
            else:
                kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            candidate = current_path / name
            relative = candidate.relative_to(stage_root).as_posix()
            if any(character in relative for character in ("\n", "\r", "\0")):
                _fail("staged source contains a path unsafe for tree hashing")
            if candidate.is_symlink():
                _fail(f"staged source contains a symlink: {relative}")
            if relative in exact or _excluded_file(name):
                excluded_runtime_paths.append(relative)
                continue
            if _sensitive_file(name):
                _fail(f"credential-like file is forbidden in staged source: {relative}")
            if not candidate.is_file():
                _fail(f"staged source contains a non-regular entry: {relative}")
            records.append((relative, sha256_file(candidate), candidate.stat().st_size))
    records.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    for relative, file_hash, _ in records:
        digest.update(f"{file_hash}  {relative}\n".encode())
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(records),
        "content_bytes": sum(size for _, _, size in records),
        "excluded_runtime_paths": sorted(excluded_runtime_paths),
    }


def _validate_checkpoint_parent_chain(
    *,
    checkpoint_dir: Path,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    lineage = config.get("lineage")
    if not isinstance(lineage, Mapping):
        _fail("recovery config lacks checkpoint lineage")
    parent = lineage.get("input_checkpoint_parent")
    current_hashes = lineage.get("input_checkpoint_file_sha256")
    if not isinstance(parent, Mapping) or not isinstance(current_hashes, Mapping):
        _fail("recovery config checkpoint lineage is incomplete")
    expected_current = {str(name): str(value) for name, value in current_hashes.items()}
    expected_parent = {
        "barun_config.json": expected_current["barun_config.json"],
        "model.safetensors": str(parent.get("model_sha256")),
        "tokenizer.json": expected_current["tokenizer.json"],
    }
    expected_source_run = str(lineage.get("input_run_id"))
    expected_parent_run = str(parent.get("run_id"))
    if contract.get("source_run_id") != expected_source_run:
        _fail("provenance PRESTO source run differs from recovery config")
    if contract.get("parent_run_id") != expected_parent_run:
        _fail("provenance PRESTO parent run differs from recovery config")
    if contract.get("checkpoint_manifest_sha256") != PRESTO_CHECKPOINT_MANIFEST_SHA256:
        _fail("provenance PRESTO checkpoint-manifest hash changed")
    if contract.get("file_sha256") != expected_current:
        _fail("provenance PRESTO checkpoint files differ from recovery config")
    if contract.get("parent_file_sha256") != expected_parent:
        _fail("provenance PRESTO parent checkpoint files differ from recovery config")

    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    _strict_hash(
        manifest_path,
        PRESTO_CHECKPOINT_MANIFEST_SHA256,
        "PRESTO checkpoint manifest",
    )
    manifest = _load_provenance_json(manifest_path, "PRESTO checkpoint manifest")
    manifest = _exact_keys(
        manifest,
        {
            "file_sha256",
            "input_checkpoint_sha256",
            "run_id",
            "schema_version",
            "source_checkpoint",
        },
        "PRESTO checkpoint manifest",
    )
    if manifest["schema_version"] != "barun-release-checkpoint-v1":
        _fail("PRESTO checkpoint manifest schema changed")
    if manifest["run_id"] != expected_source_run:
        _fail("PRESTO checkpoint manifest source run changed")
    if _strict_hash_map(manifest["file_sha256"], "PRESTO output hashes") != expected_current:
        _fail("PRESTO checkpoint manifest output hashes changed")
    if (
        _strict_hash_map(manifest["input_checkpoint_sha256"], "PRESTO parent hashes")
        != expected_parent
    ):
        _fail("PRESTO checkpoint manifest parent hashes changed")
    source_checkpoint = _required_string(
        manifest["source_checkpoint"], "PRESTO checkpoint manifest source_checkpoint"
    )
    if not source_checkpoint.endswith("/checkpoints/step-00000947"):
        _fail("PRESTO checkpoint selection step changed")
    for name, expected in expected_current.items():
        _strict_hash(checkpoint_dir / name, expected, f"PRESTO checkpoint {name}")
    return {
        "source_run_id": expected_source_run,
        "parent_run_id": expected_parent_run,
        "checkpoint_manifest_sha256": PRESTO_CHECKPOINT_MANIFEST_SHA256,
        "file_sha256": expected_current,
        "parent_file_sha256": expected_parent,
        "selection_step": 947,
    }


def validate_recovery_provenance(
    *,
    run_id: str,
    attempt_path: Path,
    snapshot_path: Path,
    input_checkpoint: Path,
    config: Mapping[str, Any],
    jarvis_machine_id: int,
) -> dict[str, Any]:
    """Validate the prelaunch evidence and exact clean tree before any experiment work."""

    stage_root = snapshot_path.parent.resolve(strict=True)
    if stage_root != REPOSITORY_ROOT.resolve(strict=True):
        _fail("source snapshot must be located at the staged repository root")
    attempt_relative = _relative_stage_path(attempt_path, stage_root, "attempt preregistration")
    snapshot_relative = _relative_stage_path(snapshot_path, stage_root, "source snapshot")
    if attempt_relative == snapshot_relative:
        _fail("attempt preregistration and source snapshot must be distinct files")

    attempt_payload = _load_provenance_json(attempt_path, "attempt preregistration")
    snapshot_payload = _load_provenance_json(snapshot_path, "source snapshot")
    attempt = _validate_attempt_schema(attempt_payload)
    snapshot = _validate_snapshot_schema(snapshot_payload)
    if attempt["run_id"] != run_id or snapshot["run_id"] != run_id:
        _fail("CLI, attempt, and source-snapshot run IDs must match exactly")
    snapshot_sha256 = sha256_file(snapshot_path)
    if attempt["source_snapshot"]["sha256"] != snapshot_sha256:
        _fail("attempt preregistration does not bind the exact source snapshot")
    if attempt["source_snapshot"]["path"] != snapshot_relative:
        _fail("attempt preregistration source-snapshot path changed")

    config_relative = RECOVERY_CONFIG.resolve(strict=True).relative_to(stage_root).as_posix()
    selector_relative = _relative_stage_path(
        SHARED_SELECTOR_PATH, stage_root, "shared rescue selector"
    )
    runner_path = RUNNER_PATH.resolve(strict=True)
    runner_relative = runner_path.relative_to(stage_root).as_posix()
    config_sha256 = sha256_file(RECOVERY_CONFIG)
    selector_sha256 = sha256_file(SHARED_SELECTOR_PATH)
    runner_sha256 = sha256_file(runner_path)
    if config_sha256 != RECOVERY_CONFIG_SHA256:
        _fail("actual recovery config differs from its compiled immutable hash")
    expected_frozen = {
        "configs/continual_recovery_v1.json": RECOVERY_CONFIG_SHA256,
        SHARED_SELECTOR_RELATIVE_PATH: SHARED_SELECTOR_SHA256,
        "scripts/run_continual_recovery.py": runner_sha256,
    }
    if config_relative != "configs/continual_recovery_v1.json":
        _fail("recovery config is not at its canonical staged path")
    if selector_relative != SHARED_SELECTOR_RELATIVE_PATH:
        _fail("shared rescue selector is not at its canonical staged path")
    if selector_sha256 != SHARED_SELECTOR_SHA256:
        _fail("staged shared rescue selector SHA-256 mismatch")
    selector_payload = _load_provenance_json(
        SHARED_SELECTOR_PATH, "shared rescue selector"
    )
    if selector_payload.get("schema_version") != SHARED_SELECTOR_SCHEMA_VERSION:
        _fail("staged shared rescue selector schema changed")
    if runner_relative != "scripts/run_continual_recovery.py":
        _fail("recovery runner is not at its canonical staged path")
    if attempt["recovery_config"] != {
        "path": config_relative,
        "sha256": config_sha256,
    }:
        _fail("attempt preregistration recovery-config identity changed")
    if attempt["shared_selector"] != {
        "path": selector_relative,
        "sha256": selector_sha256,
    }:
        _fail("attempt preregistration shared-selector identity changed")
    if attempt["runner"] != {"path": runner_relative, "sha256": runner_sha256}:
        _fail("attempt preregistration runner identity changed")
    if snapshot["frozen_files"] != expected_frozen:
        _fail(
            "source snapshot does not freeze the exact recovery runner, config, and shared selector"
        )

    inventory = attempt["prelaunch_inventory"]
    protected_machine_ids = inventory["protected_machine_ids"]
    cli_machine_id = _required_positive_int(
        jarvis_machine_id, "CLI Jarvis machine ID"
    )
    if cli_machine_id == 463058 or cli_machine_id in protected_machine_ids:
        _fail("CLI Jarvis machine ID is pre-existing and protected")
    if inventory["project_machine_id"] != cli_machine_id:
        _fail("attempt project machine ID differs from the CLI")

    checkpoint_relative = _relative_stage_path(
        input_checkpoint, stage_root, "PRESTO checkpoint directory"
    )
    # A directory resolves successfully above; reject a regular file explicitly.
    if not input_checkpoint.resolve(strict=True).is_dir():
        _fail("PRESTO checkpoint input must be a directory")
    if attempt["input_checkpoint"] != snapshot["input_checkpoint"]:
        _fail("attempt and source snapshot checkpoint chains differ")
    if attempt["input_checkpoint"]["path"] != checkpoint_relative:
        _fail("provenance PRESTO checkpoint path differs from the CLI input")
    checkpoint_chain = _validate_checkpoint_parent_chain(
        checkpoint_dir=input_checkpoint.resolve(strict=True),
        config=config,
        contract=attempt["input_checkpoint"],
    )

    expected_exact_exclusions = sorted([attempt_relative, snapshot_relative])
    tree_contract = snapshot["content_tree"]
    if tree_contract["excluded_exact_paths"] != expected_exact_exclusions:
        _fail("source snapshot exact tree exclusions do not name both provenance records")
    observed_tree = _recompute_staged_content_tree(
        stage_root,
        excluded_exact_paths=expected_exact_exclusions,
    )
    for field in ("sha256", "file_count", "content_bytes"):
        if observed_tree[field] != tree_contract[field]:
            _fail(
                f"staged content-tree {field} mismatch: expected "
                f"{tree_contract[field]!r}, got {observed_tree[field]!r}"
            )
    return {
        "schema_version": "barun-continual-recovery-provenance-validation-v1",
        "run_id": run_id,
        "stage_root": str(stage_root),
        "attempt": {
            "path": attempt_relative,
            "sha256": sha256_file(attempt_path),
        },
        "source_snapshot": {
            "path": snapshot_relative,
            "sha256": snapshot_sha256,
        },
        "recovery_config": {
            "path": config_relative,
            "sha256": config_sha256,
        },
        "shared_selector": {
            "path": selector_relative,
            "sha256": selector_sha256,
            "schema_version": SHARED_SELECTOR_SCHEMA_VERSION,
        },
        "runner": {
            "path": runner_relative,
            "sha256": runner_sha256,
        },
        "content_tree": observed_tree,
        "checkpoint_chain": checkpoint_chain,
        "prelaunch_inventory": inventory,
        "validated_before_data_prep_training_or_scoring": True,
    }


def _copy_provenance_evidence(
    *,
    essential: Path,
    attempt_path: Path,
    snapshot_path: Path,
    provenance: Mapping[str, Any],
) -> None:
    attempt_destination = essential / "attempt-preregistration.json"
    snapshot_destination = essential / "source-snapshot.json"
    shutil.copy2(attempt_path, attempt_destination)
    shutil.copy2(snapshot_path, snapshot_destination)
    _write_json(essential / "provenance-validation.json", provenance)
    attempt = provenance.get("attempt")
    snapshot = provenance.get("source_snapshot")
    if not isinstance(attempt, Mapping) or not isinstance(snapshot, Mapping):
        _fail("validated provenance lacks evidence hashes")
    if sha256_file(attempt_destination) != attempt.get("sha256"):
        _fail("attempt preregistration changed while entering the essential bundle")
    if sha256_file(snapshot_destination) != snapshot.get("sha256"):
        _fail("source snapshot changed while entering the essential bundle")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        _fail(f"refusing to overwrite recovery artifact: {path}")
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


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _progress(essential: Path, phase: str, **details: Any) -> None:
    _append_jsonl(
        essential / "progress.jsonl",
        {"at": datetime.now(timezone.utc).isoformat(), "phase": phase, **details},
    )


def _tree_manifest(root: Path) -> dict[str, Any]:
    return {
        "schema_version": "barun-continual-recovery-artifacts-v1",
        "file_sha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "artifact-sha256.json"
        },
    }


def _force_math_sdpa() -> dict[str, bool | None]:
    backend = torch.backends.cuda
    required = (
        ("flash", "enable_flash_sdp", "flash_sdp_enabled", False),
        ("memory_efficient", "enable_mem_efficient_sdp", "mem_efficient_sdp_enabled", False),
        ("math", "enable_math_sdp", "math_sdp_enabled", True),
    )
    evidence: dict[str, bool | None] = {}
    for label, setter_name, getter_name, expected in required:
        setter = getattr(backend, setter_name, None)
        getter = getattr(backend, getter_name, None)
        if not callable(setter) or not callable(getter):
            _fail(f"PyTorch cannot configure and verify {label} SDPA")
        setter(expected)
        actual = getter()
        if type(actual) is not bool or actual is not expected:
            _fail(f"failed to set {label} SDPA to {expected}: got {actual!r}")
        evidence[label] = actual
    cudnn_setter = getattr(backend, "enable_cudnn_sdp", None)
    cudnn_getter = getattr(backend, "cudnn_sdp_enabled", None)
    if callable(cudnn_setter) and callable(cudnn_getter):
        cudnn_setter(False)
        actual = cudnn_getter()
        if type(actual) is not bool or actual is not False:
            _fail(f"failed to disable cuDNN SDPA: got {actual!r}")
        evidence["cudnn"] = actual
    elif cudnn_setter is None and cudnn_getter is None:
        evidence["cudnn"] = None
    else:
        _fail("PyTorch exposes a partial, unverifiable cuDNN SDPA interface")
    return evidence


def _runtime_environment() -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        _fail("CUBLAS_WORKSPACE_CONFIG changed after import")
    if not _CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT:
        _fail("cuBLAS determinism was not configured before importing torch")
    if torch.cuda.is_initialized():
        _fail("CUDA initialized before continual-recovery preflight")
    sdpa = _force_math_sdpa()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        _fail("the continual-recovery stage requires CUDA with bfloat16 support")
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
        "cublas_workspace_config": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
        "sdpa_backends": sdpa,
        "configured_before_torch_import": True,
        "cuda_initialized_before_preflight": False,
    }


def _run_repository_tests(essential: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    log = essential / "repository-tests.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        _fail(f"repository tests failed with exit code {result.returncode}")


def _git_output(arguments: list[str]) -> bytes | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_provenance() -> dict[str, Any]:
    commit = _git_output(["rev-parse", "HEAD"])
    status = _git_output(["status", "--porcelain=v1", "-z"])
    diff = _git_output(["diff", "--binary", "HEAD"])
    untracked = _git_output(["ls-files", "--others", "--exclude-standard", "-z"])
    if commit is None or status is None or diff is None or untracked is None:
        return {"available": False}
    digest = hashlib.sha256(status + diff)
    untracked_hashes: dict[str, str] = {}
    for raw_name in filter(None, untracked.split(b"\0")):
        name = raw_name.decode("utf-8", errors="surrogateescape")
        path = REPOSITORY_ROOT / name
        if path.is_file():
            file_hash = sha256_file(path)
            untracked_hashes[name] = file_hash
            digest.update(raw_name)
            digest.update(file_hash.encode("ascii"))
    return {
        "available": True,
        "commit": commit.decode("ascii").strip(),
        "dirty": bool(status),
        "dirty_patch_sha256": digest.hexdigest(),
        "untracked_file_sha256": untracked_hashes,
    }


def _resolve(explicit: Path | None, relative: str) -> Path:
    return explicit.resolve() if explicit is not None else (REPOSITORY_ROOT / relative).resolve()


def _validate_presto_firewall(audit: Mapping[str, Any]) -> dict[str, Any]:
    evidence = audit.get("official_test")
    required = {
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
    if not isinstance(evidence, Mapping) or any(
        evidence.get(name) != expected for name, expected in required.items()
    ):
        _fail("official PRESTO test and combined members did not remain opaque")
    source = audit.get("source")
    members = source.get("members") if isinstance(source, Mapping) else None
    if not isinstance(members, Mapping):
        _fail("PRESTO audit lacks member-level access evidence")
    for name in ("presto_dataset.jsonl", "presto_test.jsonl"):
        member = members.get(name)
        if (
            not isinstance(member, Mapping)
            or member.get("runtime_open_count") != 0
            or member.get("runtime_bytes_read") != 0
        ):
            _fail(f"sealed PRESTO member was read: {name}")
    return {**required, "official_test_rows_opaque_unparsed": evidence.get("rows")}


def _checkpoint_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    lineage = config["lineage"]
    if not isinstance(lineage, Mapping):
        _fail("recovery config lacks lineage")
    hashes = lineage["input_checkpoint_file_sha256"]
    if not isinstance(hashes, Mapping):
        _fail("recovery config lacks input checkpoint hashes")
    return {str(name): str(digest) for name, digest in hashes.items()}


def _strict_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        _fail(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _metric(metrics: Mapping[str, Any], *path: str) -> float:
    value: object = metrics
    for name in path:
        if not isinstance(value, Mapping) or name not in value:
            _fail(f"metric lacks {'.'.join(path)}")
        value = value[name]
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"metric {'.'.join(path)} is nonnumeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"metric {'.'.join(path)} is non-finite")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    config = load_recovery_config(RECOVERY_CONFIG)
    snapshot_path = args.source_snapshot.resolve(strict=True)
    attempt_path = args.attempt_preregistration.resolve(strict=True)
    stage_root = snapshot_path.parent.resolve(strict=True)
    artifact_root = args.artifact_root.resolve()
    try:
        artifact_root.relative_to(stage_root)
    except ValueError:
        pass
    else:
        _fail("artifact root must remain outside the immutable staged source tree")
    provenance = validate_recovery_provenance(
        run_id=args.run_id,
        attempt_path=attempt_path,
        snapshot_path=snapshot_path,
        input_checkpoint=args.input_checkpoint,
        config=config,
        jarvis_machine_id=args.jarvis_machine_id,
    )

    run_root = artifact_root / args.run_id
    try:
        run_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ExistingRecoveryRunError(
            f"refusing to overwrite continual-recovery run root {run_root}"
        ) from error
    export = run_root / "export"
    essential = export / "essential"
    essential.mkdir(parents=True)
    _progress(essential, "run_root_claimed", run_root=str(run_root))

    shutil.copy2(RECOVERY_CONFIG, essential / "preregistration.json")
    _copy_provenance_evidence(
        essential=essential,
        attempt_path=attempt_path,
        snapshot_path=snapshot_path,
        provenance=provenance,
    )
    _progress(
        essential,
        "attempt_snapshot_and_tree_provenance_verified",
        recovery_config_sha256=RECOVERY_CONFIG_SHA256,
        content_tree_sha256=provenance["content_tree"]["sha256"],
    )
    environment = _runtime_environment()
    environment["jarvis_machine_id"] = args.jarvis_machine_id
    environment["git"] = _git_provenance()
    environment["command"] = sys.argv
    environment["hourly_cost_inr"] = args.hourly_cost_inr
    _write_json(essential / "environment.json", environment)
    _progress(essential, "determinism_and_environment_preflight_passed")

    _run_repository_tests(essential)
    _progress(essential, "repository_tests_passed")

    input_checkpoint = args.input_checkpoint.resolve()
    input_hashes = _checkpoint_hashes(config)
    verify_checkpoint(input_checkpoint, expected_sha256=input_hashes)
    tokenizer = Tokenizer.from_file(str(input_checkpoint / "tokenizer.json"))
    _progress(essential, "exact_presto_checkpoint_verified", input_run_id=config["lineage"]["input_run_id"])

    data = config["data"]
    if not isinstance(data, Mapping):
        _fail("recovery config data is invalid")
    mobile = data["mobile"]
    presto = data["presto"]
    if not isinstance(mobile, Mapping) or not isinstance(presto, Mapping):
        _fail("recovery dataset contracts are invalid")
    mobile_train_contract = mobile["train"]
    mobile_dev_contract = mobile["development"]
    mobile_audit_contract = mobile["audit"]
    if not all(
        isinstance(contract, Mapping)
        for contract in (mobile_train_contract, mobile_dev_contract, mobile_audit_contract)
    ):
        _fail("Mobile dataset contracts are invalid")
    mobile_train = _resolve(args.mobile_train, str(mobile_train_contract["relative_path"]))
    mobile_dev = _resolve(args.mobile_dev, str(mobile_dev_contract["relative_path"]))
    mobile_audit = _resolve(args.mobile_audit, str(mobile_audit_contract["relative_path"]))
    _strict_hash(mobile_train, str(mobile_train_contract["sha256"]), "Mobile train")
    _strict_hash(mobile_dev, str(mobile_dev_contract["sha256"]), "Mobile development")
    _strict_hash(mobile_audit, str(mobile_audit_contract["sha256"]), "Mobile audit")
    _progress(essential, "frozen_mobile_artifacts_verified", official_evaluation_rows_read=0)

    cache = args.presto_cache.resolve() if args.presto_cache else run_root / "cache"
    archive = download_pinned_archive(cache)
    prepared = prepare_presto(
        archive,
        run_root / "prepared-presto",
        tokenizer=tokenizer,
        tokenizer_identity=TokenizerIdentity(
            identifier=TOKENIZER_IDENTIFIER,
            revision=TOKENIZER_REVISION,
            sha256=input_hashes["tokenizer.json"],
        ),
    )
    expected_presto = {
        "train": str(presto["train"]["sha256"]),
        "dev": str(presto["development"]["sha256"]),
        "audit": str(presto["audit_sha256"]),
    }
    if dict(prepared.hashes) != expected_presto:
        _fail(f"prepared PRESTO hashes changed: {dict(prepared.hashes)}")
    if prepared.train_rows != 47_806 or prepared.dev_rows != 14_288:
        _fail("prepared PRESTO English population changed")
    presto_audit = json.loads(prepared.audit_path.read_text(encoding="utf-8"))
    if not isinstance(presto_audit, dict):
        _fail("PRESTO audit is not an object")
    presto_firewall = _validate_presto_firewall(presto_audit)
    data_evidence = essential / "data"
    data_evidence.mkdir()
    shutil.copy2(prepared.audit_path, data_evidence / "presto-audit.json")
    shutil.copy2(mobile_audit, data_evidence / "mobile-audit.json")
    _progress(
        essential,
        "pinned_presto_prepared",
        revision=PRESTO_REVISION,
        official_test_rows_read=0,
    )

    view = config["recovery_view"]
    optimization = config["optimization"]
    if not isinstance(view, Mapping) or not isinstance(optimization, Mapping):
        _fail("recovery view or optimization contract is invalid")
    work_data = export / "recovery-data"
    work_data.mkdir()
    combined_manifest = work_data / "continual-recovery-train.jsonl"
    view_audit_path = data_evidence / "continual-recovery-view-audit.json"
    view_audit = materialize_recovery_view(
        mobile_train_path=mobile_train,
        mobile_train_sha256=str(mobile_train_contract["sha256"]),
        mobile_train_rows=int(mobile_train_contract["rows"]),
        mobile_membership_sha256=str(mobile_train_contract["membership_sha256"]),
        presto_train_path=prepared.train_manifest,
        presto_train_sha256=expected_presto["train"],
        presto_train_rows=int(presto["train"]["rows"]),
        presto_dev_path=prepared.dev_manifest,
        presto_dev_sha256=expected_presto["dev"],
        presto_dev_rows=int(presto["development"]["rows"]),
        replay_rows=int(view["presto_replay_rows"]),
        seed=int(optimization["seed"]),
        output_manifest=combined_manifest,
        output_audit=view_audit_path,
    )
    combined = view_audit["combined"]
    if not isinstance(combined, Mapping):
        _fail("recovery view audit lacks combined evidence")
    combined_rows = load_manifest(
        combined_manifest,
        expected_sha256=str(combined["sha256"]),
        expected_derived_split="train",
    )
    mobile_ids = [row.example_id for row in combined_rows if not row.example_id.startswith("recovery-presto-")]
    replay_ids = [row.example_id for row in combined_rows if row.example_id.startswith("recovery-presto-")]
    if len(mobile_ids) != 7_937 or len(replay_ids) != 2_646:
        _fail("combined recovery source populations changed")
    _progress(
        essential,
        "train_only_rehearsal_view_frozen",
        combined_rows=len(combined_rows),
        combined_sha256=combined["sha256"],
    )

    training_work = export / "recovery-training"
    training = train_continual_recovery(
        run_id=args.run_id,
        checkpoint_dir=input_checkpoint,
        checkpoint_sha256=input_hashes,
        combined_manifest=combined_manifest,
        combined_manifest_sha256=str(combined["sha256"]),
        mobile_ids=mobile_ids,
        replay_ids=replay_ids,
        optimization=optimization,
        output_dir=training_work,
        device_name="cuda",
    )
    checkpoint = essential / "checkpoint"
    shutil.copytree(training_work / "checkpoint", checkpoint)
    compact_training = essential / "training"
    compact_training.mkdir()
    shutil.copy2(training_work / "metrics.jsonl", compact_training / "metrics.jsonl")
    shutil.copy2(training_work / "summary.json", compact_training / "summary.json")
    checkpoint_manifest = checkpoint / "checkpoint_manifest.json"
    _progress(
        essential,
        "single_recovery_recipe_trained",
        optimizer_steps=training["optimizer_steps"],
        development_evaluations=0,
    )

    gc.collect()
    torch.cuda.empty_cache()
    terminal = config["terminal_evaluation"]
    if not isinstance(terminal, Mapping):
        _fail("terminal evaluation contract is invalid")
    mobile_eval = terminal["mobile"]
    presto_eval = terminal["presto"]
    if not isinstance(mobile_eval, Mapping) or not isinstance(presto_eval, Mapping):
        _fail("terminal dataset evaluation contracts are invalid")
    final_root = essential / "terminal-evaluation"
    final_root.mkdir()
    mobile_result = run_mobile_regression(
        checkpoint_dir=checkpoint,
        checkpoint_hashes_path=checkpoint_manifest,
        output_dir=final_root / "mobile",
        repository_root=REPOSITORY_ROOT,
        preregistration_path=MOBILE_REGRESSION_CONFIG,
        runtime_environment=environment,
        manifest_path=mobile_dev,
        audit_path=mobile_audit,
        reference_aggregate_path=args.mobile_reference_aggregate,
        reference_samples_path=args.mobile_reference_samples,
    )
    _progress(
        essential,
        "terminal_mobile_evaluated_once",
        numerator=mobile_result["gate"]["candidate"]["numerator"],
        denominator=mobile_result["gate"]["candidate"]["denominator"],
    )

    presto_output = final_root / "presto"
    presto_output.mkdir()
    presto_predictions = presto_output / "predictions.jsonl"
    presto_generation = generate_manifest(
        checkpoint_dir=checkpoint,
        manifest_path=prepared.dev_manifest,
        manifest_sha256=expected_presto["dev"],
        predictions_path=presto_predictions,
        device_name="cuda",
        batch_size=int(presto_eval["batch_size"]),
        max_new_tokens=int(presto_eval["max_new_tokens"]),
    )
    presto_score_paths = write_scores(
        prepared.dev_manifest,
        presto_predictions,
        presto_output / "scores",
    )
    presto_metrics = json.loads(presto_score_paths["aggregate"].read_text(encoding="utf-8"))
    if not isinstance(presto_metrics, dict):
        _fail("PRESTO aggregate is not an object")
    thresholds = presto_eval["gates"]
    if not isinstance(thresholds, Mapping):
        _fail("PRESTO gate thresholds are invalid")
    presto_gate = evaluate_recovery_presto_gate(presto_metrics, thresholds)
    _progress(
        essential,
        "terminal_corrected_presto_evaluated_once",
        ast_exact=_metric(presto_metrics, "ast_exact_match", "value"),
        gate_passed=presto_gate["passed"],
    )

    mobile_passed = mobile_result["gate"].get("passed") is True
    presto_passed = presto_gate["passed"] is True
    elapsed = time.monotonic() - started
    result = {
        "schema_version": "barun-continual-recovery-result-v1",
        "run_id": args.run_id,
        "jarvis_machine_id": args.jarvis_machine_id,
        "preregistration_sha256": RECOVERY_CONFIG_SHA256,
        "provenance": provenance,
        "recipe_id": config["recipe_id"],
        "input_checkpoint": {
            "run_id": config["lineage"]["input_run_id"],
            "directory": str(input_checkpoint),
            "file_sha256": input_hashes,
        },
        "output_checkpoint": {
            "directory": str(checkpoint.resolve()),
            "file_sha256": training["output_checkpoint"]["file_sha256"],
        },
        "training": training,
        "data": {
            "mobile": {
                "train_rows": 7_937,
                "development_rows": 756,
                "official_evaluation_rows_opaque_unparsed": 961,
                "official_evaluation_rows_read": 0,
            },
            "presto": {
                "revision": PRESTO_REVISION,
                "train_rows": prepared.train_rows,
                "replayed_train_rows": 2_646,
                "development_rows": prepared.dev_rows,
                "prepared_sha256": dict(prepared.hashes),
                "official_test_firewall": presto_firewall,
                "official_test_rows_read": 0,
            },
            "recovery_view": view_audit,
        },
        "terminal_evaluation": {
            "selection_policy": "final_checkpoint_only_no_dev_tuning",
            "rounds": 1,
            "mobile_evaluations": 1,
            "presto_evaluations": 1,
            "mobile": mobile_result,
            "presto": {
                "generation": presto_generation.to_dict(),
                "aggregate": presto_metrics,
                "aggregate_sha256": sha256_file(presto_score_paths["aggregate"]),
                "sample_scores_sha256": sha256_file(presto_score_paths["samples"]),
                "gate": presto_gate,
            },
        },
        "gate": {
            "mobile_passed": mobile_passed,
            "presto_passed": presto_passed,
            "passed": mobile_passed and presto_passed,
            "decision": "keep_recovery_checkpoint" if mobile_passed and presto_passed else "reject_recovery_checkpoint",
        },
        "claim_limits": list(config["claim_limits"]),
        "elapsed_seconds": elapsed,
        "estimated_cost_inr": elapsed / 3_600 * args.hourly_cost_inr,
    }
    _write_json(essential / "result.json", result)
    _progress(essential, "run_completed", gate_passed=result["gate"]["passed"])
    _write_json(essential / "artifact-sha256.json", _tree_manifest(essential))
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
    if RUN_ID_PATTERN.fullmatch(value) is None or "recovery" not in value:
        raise argparse.ArgumentTypeError(
            "must match YYYYMMDD-HHMM-...-recovery-...-sN and include recovery"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    config = load_recovery_config(RECOVERY_CONFIG)
    lineage = config["lineage"]
    mobile = config["data"]["mobile"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=_run_id, required=True)
    parser.add_argument("--jarvis-machine-id", type=_positive_int, required=True)
    parser.add_argument(
        "--attempt-preregistration",
        type=Path,
        required=True,
        help="Frozen attempt v1 JSON inside the staged source root.",
    )
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        required=True,
        help="Hash-bound source-snapshot v1 JSON at the staged source root.",
    )
    parser.add_argument(
        "--input-checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / str(lineage["input_checkpoint_relative_path"]),
    )
    parser.add_argument(
        "--mobile-train",
        type=Path,
        default=REPOSITORY_ROOT / str(mobile["train"]["relative_path"]),
    )
    parser.add_argument(
        "--mobile-dev",
        type=Path,
        default=REPOSITORY_ROOT / str(mobile["development"]["relative_path"]),
    )
    parser.add_argument(
        "--mobile-audit",
        type=Path,
        default=REPOSITORY_ROOT / str(mobile["audit"]["relative_path"]),
    )
    parser.add_argument("--mobile-reference-aggregate", type=Path)
    parser.add_argument("--mobile-reference-samples", type=Path)
    parser.add_argument("--presto-cache", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=Path("/root/barun-artifacts"))
    parser.add_argument("--hourly-cost-inr", type=_nonnegative_float, default=378.27)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    essential = args.artifact_root.resolve() / args.run_id / "export" / "essential"
    try:
        result = run(args)
    except ExistingRecoveryRunError:
        raise
    except BaseException as error:
        essential.mkdir(parents=True, exist_ok=True)
        _progress(
            essential,
            "run_failed",
            error_type=type(error).__name__,
            message=str(error),
        )
        failure = {
            "at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        if not (essential / "failure.json").exists():
            _write_json(essential / "failure.json", failure)
        artifact_manifest = essential / "artifact-sha256.json"
        if not artifact_manifest.exists():
            _write_json(artifact_manifest, _tree_manifest(essential))
        raise
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
