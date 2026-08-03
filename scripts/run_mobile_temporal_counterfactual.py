"""Preflight or execute the frozen Month-Boundary Counterfactual SFT v2 experiment.

This entrypoint is deliberately offline and standard-library-only.  It validates
the materialized training and shadow manifests, writes deterministic execution
receipts, and stops before importing torch, loading a checkpoint, or touching CUDA.
Only the explicit ``--execute --device cuda --jarvis-resource-id ID`` path imports
the fixed backend, after setting deterministic cuBLAS configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = "barun-mobile-temporal-counterfactual-v2"
VIEW_AUDIT_SCHEMA_VERSION = "barun-mobile-temporal-view-audit-v1"
SHADOW_AUDIT_SCHEMA_VERSION = "barun-mobile-temporal-shadow-audit-v1"
FULL_VIEW_AUDIT_SCHEMA_VERSION = "barun-mobile-temporal-full-view-audit-v1"
TRANSFORM_RECEIPT_SCHEMA_VERSION = "barun-mobile-temporal-transform-receipt-v1"
SFT_SCHEMA_VERSION = "barun-sft-example-v1"
SHADOW_SPLIT_VERSION = "barun-mobile-temporal-shadow-split-v1"
EXECUTION_BACKEND_VERSION = "barun-mobile-temporal-execution-v1"
ATTEMPT_PREREGISTRATION_SCHEMA_VERSION = "barun-mobile-temporal-attempt-preregistration-v2"
SOURCE_SNAPSHOT_SCHEMA_VERSION = "barun-mobile-temporal-source-snapshot-v1"
SOURCE_TREE_ALGORITHM = "sha256_of_sorted_sha256_two_spaces_posix_path_newline"
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
REMOTE_VALIDATION_TIMEOUT_SECONDS = 600
EXPECTED_JARVIS_TEMPLATE = "axolotl"
EXPECTED_PYTHON_IMPLEMENTATION = "CPython"
EXPECTED_PYTHON_VERSION = "3.11.10"
PINNED_REUSED_756_SHA256 = "988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55"
PINNED_REUSED_756_ROWS = 756
FROZEN_TERMINAL_MAX_NEW_TOKENS = 192
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_PREREGISTRATION_RELATIVE_PATH = "attempt-preregistration.json"
SOURCE_SNAPSHOT_RELATIVE_PATH = "source-snapshot.json"
RETRY_EVIDENCE_ROOT = "retry-evidence"
SOURCE_TREE_EXCLUDED_DIRECTORY_NAMES = tuple(
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
SOURCE_TREE_EXCLUDED_DIRECTORY_SUFFIXES = (".egg-info",)
SOURCE_TREE_EXCLUDED_FILE_NAMES = (".DS_Store",)
SOURCE_TREE_EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo")
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
ATTEMPT_STATUS = (
    "scientific_fields_frozen_before_instance_creation_inventory_bound_after_fresh_creation_"
    "before_upload_and_model_or_cuda_loading"
)
KNOWN_PROTECTED_JARVIS_IDS = frozenset({463058, 463689, 463697, 463719, 463786, 463788})
FROZEN_TOKEN_LENGTH_AUDIT = {
    "schema_version": "barun-mobile-temporal-token-length-audit-v1",
    "tokenizer_sha256": "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6",
    "max_seq_len": 2_048,
    "eos_tokens": 1,
    "role": "pre_score_evidence_used_to_choose_a_static_nontruncating_cap_never_derived_at_runtime",
    "populations": {
        "selection": {
            "rows": 1_024,
            "max_prompt_tokens": 445,
            "max_target_tokens": 194,
            "max_target_plus_eos_tokens": 195,
            "max_full_sequence_tokens": 594,
            "rows_over_max_seq_len": 0,
        },
        "confirmation": {
            "rows": 1_168,
            "max_prompt_tokens": 471,
            "max_target_tokens": 192,
            "max_target_plus_eos_tokens": 193,
            "max_full_sequence_tokens": 637,
            "rows_over_max_seq_len": 0,
        },
        "construction": {
            "rows": 5_745,
            "max_prompt_tokens": 487,
            "max_target_tokens": 196,
            "max_target_plus_eos_tokens": 197,
            "max_full_sequence_tokens": 660,
            "rows_over_max_seq_len": 0,
        },
        "repeat": {
            "rows": 6_838,
            "max_prompt_tokens": 487,
            "max_target_tokens": 196,
            "max_target_plus_eos_tokens": 197,
            "max_full_sequence_tokens": 660,
            "rows_over_max_seq_len": 0,
        },
        "mbcf": {
            "rows": 6_838,
            "max_prompt_tokens": 487,
            "max_target_tokens": 196,
            "max_target_plus_eos_tokens": 197,
            "max_full_sequence_tokens": 660,
            "rows_over_max_seq_len": 0,
        },
        "conditional_full_refit": {
            "rows": 9_483,
            "max_prompt_tokens": 487,
            "max_target_tokens": 196,
            "max_target_plus_eos_tokens": 197,
            "max_full_sequence_tokens": 660,
            "rows_over_max_seq_len": 0,
        },
    },
}

PINNED_TRAIN_ROWS = 7_937
PINNED_TRAIN_SHA256 = "131473ccb5bfb51cac0439b42159e72ec4c598025e50364a52b122b056c2e84e"
PINNED_TRAIN_MEMBERSHIP_SHA256 = "4cdfc3649c21a9f0d3dc4d8d9cffcae3cb5c6abc4414a9fb09636e9220afe96f"
EXPECTED_SHADOW_ROWS = {
    "construction_train": 5_745,
    "selection": 1_024,
    "confirmation": 1_168,
}
EXPECTED_SHADOW_MEMBERSHIP_SHA256 = {
    "construction_train": ("b70f6dfe79cfe49792b6e6f5d414fd509c889d18101f75cd8df6d4e4f146c588"),
    "selection": "b4191540878ec28c6819e08e0b390fbfd57eab836d3361727ad89862a46cf927",
    "confirmation": "519c22724873ee579d0f59be886cfc297088113bd1a297521b9ca280a79c35b9",
}

EXPECTED_ARMS = (
    {"arm_id": "A", "name": "standard", "view": "standard"},
    {"arm_id": "B", "name": "repeat", "view": "repeat"},
    {"arm_id": "C", "name": "mbcf", "view": "counterfactual"},
)
EXPECTED_SEEDS = (17, 29, 43)
EXPECTED_OPTIMIZATION = {
    "adam_epsilon": 1e-8,
    "batch_size": 63,
    "beta1": 0.9,
    "beta2": 0.95,
    "decoding": "unconstrained_greedy",
    "epochs": 1,
    "gradient_accumulation_steps": 1,
    "gradient_clip_norm": 1.0,
    "learning_rate": 1e-4,
    "loss": "response_only_cross_entropy",
    "max_seq_len": 2_048,
    "min_learning_rate_ratio": 0.1,
    "precision": "bf16",
    "warmup_steps": 12,
    "weight_decay": 0.1,
}
EXPECTED_THRESHOLDS = {
    "minimum_cross_month_calendar_fraction_mbcf": 0.45,
    "minimum_safe_variants": 1_000,
}
EXPECTED_TOKENIZER = {
    "identifier": "harrrshall/BarunLM-35M",
    "max_seq_len": 2_048,
    "revision": "ef3e483a9fd7d906ecf2a7929babeffaf82d1d16",
    "sha256": "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6",
}
EXPECTED_OFFICIAL_EVALUATION = {
    "policy": "forbidden",
    "rows": 961,
    "rows_materialized": 0,
    "rows_read": 0,
    "rows_scored": 0,
}
EXPECTED_ARTIFACT_FILENAMES = {
    "construction": "construction-train.jsonl",
    "selection": "selection.jsonl",
    "confirmation": "confirmation.jsonl",
    "repeat": "repeat-train.jsonl",
    "counterfactual": "counterfactual-train.jsonl",
    "receipts": "transform-receipts.jsonl",
}
EXPECTED_FULL_REFIT_FILENAMES = {
    "audit": "full-view-audit.json",
    "receipts": "full-transform-receipts.jsonl",
    "train": "full-counterfactual-train.jsonl",
}
EXPECTED_FULL_MINIMUM_VARIANTS = 1_400
_EXPECTED_FULL_REFIT_KEYS = {
    "audit_sha256",
    "optimizer_steps",
    "receipts_sha256",
    "rows",
    "safe_source_membership_sha256",
    "seed",
    "source_rows",
    "train_sha256",
    "variant_rows",
}

_EXPECTED_MATERIALIZATION_KEYS = {
    "artifact_rows",
    "artifact_sha256",
    "shadow_audit_sha256",
    "view_audit_sha256",
}
_RUN_ID_RE = re.compile(r"^\d{8}-\d{4}-[a-z0-9][a-z0-9-]*-s\d+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NOW_RE = re.compile(r"(?:^|\n)NOW ([^\n]+)(?:\n|$)")
SCIENTIFIC_CONTRACT_FIELDS = (
    "schema_version",
    "run_id",
    "hypothesis",
    "decision",
    "diagnostic_basis",
    "source_data",
    "shadow_split",
    "transform",
    "arms",
    "seeds",
    "base_checkpoint",
    "optimization",
    "decoding_contract",
    "token_length_audit",
    "training_contract",
    "thresholds",
    "metric_contract",
    "checkpoint_policy",
    "screening_fit_budget",
    "conditional_full_refit_budget",
    "materialization",
    "materialization_summary",
    "selection_gate",
    "confirmation_gate",
    "conditional_full_refit",
    "terminal_compatibility_veto",
    "retry_policy",
    "remote_validation_contract",
    "official_evaluation",
    "official_evaluation_after_promotion",
    "compute",
    "claim_limits",
)
EXPECTED_SCIENTIFIC_CONTRACT_SHA256 = (
    "d077effff172a4870ec1d8af2ee4e6b3b3fcf80a8b860eaf65eb5d0781942186"
)


class TemporalPreflightError(RuntimeError):
    """The frozen temporal experiment contract was not proved."""


class ExistingPlanError(FileExistsError):
    """A deterministic plan path already contains different bytes."""


@dataclass(frozen=True, slots=True)
class ManifestObservation:
    path: Path
    sha256: str
    rows: tuple[dict[str, Any], ...]
    example_ids: tuple[str, ...]
    membership_sha256: str
    order_sha256: str
    source_ids: tuple[str, ...]
    source_membership_sha256: str
    source_order_sha256: str

    def receipt(self, *, relative_path: str) -> dict[str, Any]:
        return {
            "membership_sha256": self.membership_sha256,
            "order_sha256": self.order_sha256,
            "path": relative_path,
            "rows": len(self.rows),
            "sha256": self.sha256,
            "source_membership_sha256": self.source_membership_sha256,
            "source_order_sha256": self.source_order_sha256,
        }


@dataclass(frozen=True, slots=True)
class PriorAttemptValidation:
    machine_id: int
    scientific_content_tree: dict[str, Any]
    evidence_paths: tuple[str, ...]


def _reject_constant(value: str) -> None:
    raise TemporalPreflightError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise TemporalPreflightError(f"duplicate JSON key {key!r} is forbidden")
        payload[key] = value
    return payload


def _loads_json(raw: str, *, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise TemporalPreflightError(f"invalid JSON in {label}: {error}") from error


def _load_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise TemporalPreflightError(f"{label} is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TemporalPreflightError(f"{label} is not UTF-8: {path}") from error
    payload = _loads_json(text, label=label)
    if not isinstance(payload, dict):
        raise TemporalPreflightError(f"{label} must contain one JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_sha256(values: Iterable[str], *, sort_values: bool = False) -> str:
    materialized = list(values)
    if sort_values:
        materialized.sort()
    return hashlib.sha256(("\n".join(materialized) + "\n").encode("utf-8")).hexdigest()


def _as_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TemporalPreflightError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _as_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TemporalPreflightError(f"{label} must be a non-negative integer")
    return value


def _scientific_contract_sha256(payload: Mapping[str, Any]) -> str:
    contract = {name: payload.get(name) for name in SCIENTIFIC_CONTRACT_FIELDS}
    return hashlib.sha256(_canonical_bytes(contract)).hexdigest()


def _exact_object(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise TemporalPreflightError(f"{label} must contain exactly {sorted(keys)!r}")
    return value


def _normalized_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TemporalPreflightError(f"{label} must be a non-empty relative POSIX path")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != value
        or value in {".", ""}
    ):
        raise TemporalPreflightError(f"{label} must be a normalized relative POSIX path")
    return value


def _prior_evidence_path(value: Any, *, attempt_id: str, filename: str, label: str) -> str:
    relative = _normalized_relative_path(value, label=label)
    expected = f"{RETRY_EVIDENCE_ROOT}/{attempt_id}/{filename}"
    if relative != expected:
        raise TemporalPreflightError(f"{label} must be the attempt-scoped path {expected!r}")
    return relative


def _repo_file(root: Path, relative: str, *, label: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise TemporalPreflightError(f"{label} cannot be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise TemporalPreflightError(f"{label} is missing: {candidate}") from error
    if resolved != root and root not in resolved.parents:
        raise TemporalPreflightError(f"{label} escapes the staged source root")
    if not resolved.is_file():
        raise TemporalPreflightError(f"{label} is not a regular file")
    return resolved


def _excluded_source_directory(name: str) -> bool:
    return name in SOURCE_TREE_EXCLUDED_DIRECTORY_NAMES or name.endswith(
        SOURCE_TREE_EXCLUDED_DIRECTORY_SUFFIXES
    )


def _excluded_source_file(name: str) -> bool:
    return name in SOURCE_TREE_EXCLUDED_FILE_NAMES or name.endswith(
        SOURCE_TREE_EXCLUDED_FILE_SUFFIXES
    )


def _sensitive_source_file(name: str) -> bool:
    folded = name.casefold()
    return (
        folded == ".env"
        or folded.startswith((".env.", "credentials.", "secrets."))
        or folded in SENSITIVE_EXACT_FILE_NAMES
        or folded.endswith(SENSITIVE_FILE_SUFFIXES)
    )


def _validate_active_runtime_root_venv(*, root: Path, candidate: Path) -> None:
    expected = root / ".venv"
    if candidate != expected or candidate.parent != root or candidate.name != ".venv":
        raise TemporalPreflightError("runtime virtual environment is not repository-root .venv")
    if candidate.is_symlink():
        raise TemporalPreflightError("repository-root runtime .venv cannot be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        prefix = Path(sys.prefix).resolve(strict=True)
        exec_prefix = Path(sys.exec_prefix).resolve(strict=True)
        base_prefix = Path(sys.base_prefix).resolve(strict=True)
    except OSError as error:
        raise TemporalPreflightError(
            "runtime virtual-environment identity is not resolvable"
        ) from error
    if not candidate.is_dir() or resolved != expected:
        raise TemporalPreflightError(
            "runtime virtual environment is not exact repository-root .venv"
        )
    if prefix != expected or exec_prefix != expected or base_prefix == expected:
        raise TemporalPreflightError("repository-root .venv is not the active Python environment")

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if not virtual_env:
        raise TemporalPreflightError("active repository-root .venv lacks VIRTUAL_ENV binding")
    virtual_env_path = Path(virtual_env)
    try:
        resolved_virtual_env = virtual_env_path.resolve(strict=True)
    except OSError as error:
        raise TemporalPreflightError("VIRTUAL_ENV binding is not resolvable") from error
    if not virtual_env_path.is_absolute() or resolved_virtual_env != expected:
        raise TemporalPreflightError("VIRTUAL_ENV does not bind exact repository-root .venv")

    executable = Path(sys.executable)
    if (
        not executable.is_absolute()
        or executable.parent != expected / "bin"
        or not executable.is_file()
    ):
        raise TemporalPreflightError("active Python executable is not repository-root .venv/bin")


def _recompute_source_tree(
    root: Path,
    *,
    excluded_exact_paths: Sequence[str],
    allow_active_runtime_root_venv: bool = False,
) -> dict[str, Any]:
    exact = set(excluded_exact_paths)
    records: list[tuple[str, str, int]] = []
    for current, directory_names, file_names in os.walk(root, topdown=True):
        current_path = Path(current)
        kept: list[str] = []
        for name in sorted(directory_names):
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                raise TemporalPreflightError(f"staged source contains a symlink: {relative}")
            if _excluded_source_directory(name):
                if allow_active_runtime_root_venv and relative == ".venv":
                    _validate_active_runtime_root_venv(root=root, candidate=candidate)
                    continue
                raise TemporalPreflightError(
                    f"clean staged source contains a forbidden excluded directory: {relative}"
                )
            kept.append(name)
        directory_names[:] = kept
        for name in sorted(file_names):
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            if any(character in relative for character in ("\n", "\r", "\0")):
                raise TemporalPreflightError("staged source contains a path unsafe for hashing")
            if candidate.is_symlink():
                raise TemporalPreflightError(f"staged source contains a symlink: {relative}")
            if relative in exact:
                continue
            if _excluded_source_file(name):
                raise TemporalPreflightError(
                    f"clean staged source contains a forbidden excluded file: {relative}"
                )
            if _sensitive_source_file(name):
                raise TemporalPreflightError(
                    f"credential-like file is forbidden in staged source: {relative}"
                )
            if not candidate.is_file():
                raise TemporalPreflightError(
                    f"staged source contains a non-regular entry: {relative}"
                )
            records.append((relative, _sha256_file(candidate), candidate.stat().st_size))
    records.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    for relative, file_sha256, _ in records:
        digest.update(f"{file_sha256}  {relative}\n".encode())
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(records),
        "content_bytes": sum(size for _, _, size in records),
        "files": [
            {"path": relative, "sha256": file_sha256, "bytes": size}
            for relative, file_sha256, size in records
        ],
    }


def _validate_declared_source_tree(
    value: Any,
    *,
    label: str,
    required_excluded_exact_paths: Sequence[str],
    allowed_excluded_exact_paths: Sequence[str],
) -> dict[str, Any]:
    tree = _exact_object(
        value,
        {
            "algorithm",
            "sha256",
            "file_count",
            "content_bytes",
            "excluded_exact_paths",
            "excluded_directory_names",
            "excluded_directory_suffixes",
            "excluded_file_names",
            "excluded_file_suffixes",
            "files",
        },
        label=label,
    )
    expected_tree_metadata = {
        "algorithm": SOURCE_TREE_ALGORITHM,
        "excluded_directory_names": list(SOURCE_TREE_EXCLUDED_DIRECTORY_NAMES),
        "excluded_directory_suffixes": list(SOURCE_TREE_EXCLUDED_DIRECTORY_SUFFIXES),
        "excluded_file_names": list(SOURCE_TREE_EXCLUDED_FILE_NAMES),
        "excluded_file_suffixes": list(SOURCE_TREE_EXCLUDED_FILE_SUFFIXES),
    }
    if any(tree.get(name) != expected for name, expected in expected_tree_metadata.items()):
        raise TemporalPreflightError(f"{label} policy changed")
    exclusions = tree["excluded_exact_paths"]
    if not isinstance(exclusions, list) or exclusions != sorted(set(exclusions)):
        raise TemporalPreflightError(f"{label} exact exclusions must be sorted and unique")
    normalized_exclusions = {
        _normalized_relative_path(path, label=f"{label} exact exclusion") for path in exclusions
    }
    required = set(required_excluded_exact_paths)
    allowed = set(allowed_excluded_exact_paths)
    if not required <= normalized_exclusions or not normalized_exclusions <= allowed:
        raise TemporalPreflightError(f"{label} contains an unauthorized exact exclusion")

    files = tree["files"]
    if not isinstance(files, list) or not files:
        raise TemporalPreflightError(f"{label}.files must be a non-empty list")
    records: list[tuple[str, str, int]] = []
    for index, raw_record in enumerate(files):
        record = _exact_object(
            raw_record, {"path", "sha256", "bytes"}, label=f"{label}.files[{index}]"
        )
        relative = _normalized_relative_path(record["path"], label=f"{label}.files[{index}].path")
        if relative in normalized_exclusions:
            raise TemporalPreflightError(f"{label} contains an exactly excluded file")
        digest = _as_sha256(record["sha256"], label=f"{label}.files[{index}].sha256")
        size = record["bytes"]
        if type(size) is not int or size < 0:
            raise TemporalPreflightError(f"{label}.files[{index}].bytes is invalid")
        records.append((relative, digest, size))
    if records != sorted(records, key=lambda item: item[0]) or len({r[0] for r in records}) != len(
        records
    ):
        raise TemporalPreflightError(f"{label}.files must be path-sorted and unique")
    digest = hashlib.sha256()
    for relative, file_sha256, _ in records:
        digest.update(f"{file_sha256}  {relative}\n".encode())
    projection = {
        "sha256": digest.hexdigest(),
        "file_count": len(records),
        "content_bytes": sum(size for _, _, size in records),
        "files": [
            {"path": relative, "sha256": file_sha256, "bytes": size}
            for relative, file_sha256, size in records
        ],
    }
    if any(tree.get(name) != observed for name, observed in projection.items()):
        raise TemporalPreflightError(f"{label} internal digest/count accounting is invalid")
    return projection


def _validate_source_snapshot(
    *,
    root: Path,
    snapshot_path: Path,
    expected_sha256: str,
    config: Mapping[str, Any],
    terminal_relative_path: str,
    prior_evidence_paths: Sequence[str] = (),
    prior_scientific_trees: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    snapshot, observed_sha256 = _load_json_object(snapshot_path, label="source snapshot")
    if observed_sha256 != expected_sha256:
        raise TemporalPreflightError("source snapshot SHA-256 differs from attempt binding")
    snapshot = _exact_object(
        snapshot,
        {"schema_version", "run_id", "git", "content_tree", "frozen_files", "staging_policy"},
        label="source snapshot",
    )
    if snapshot["schema_version"] != SOURCE_SNAPSHOT_SCHEMA_VERSION:
        raise TemporalPreflightError("source snapshot schema changed")
    if snapshot["run_id"] != config.get("run_id"):
        raise TemporalPreflightError("source snapshot run_id differs from frozen config")
    git = _exact_object(
        snapshot["git"], {"repository", "commit", "branch"}, label="source snapshot git"
    )
    if any(not isinstance(git[name], str) or not git[name] for name in git):
        raise TemporalPreflightError("source snapshot git fields must be non-empty strings")
    if re.fullmatch(r"[0-9a-f]{40}", git["commit"]) is None:
        raise TemporalPreflightError("source snapshot git commit must be a full lowercase hash")

    implementation = config.get("implementation_contract")
    if not isinstance(implementation, Mapping) or not isinstance(
        implementation.get("files"), Mapping
    ):
        raise TemporalPreflightError("config implementation_contract.files is missing")
    expected_files = {str(path): str(digest) for path, digest in implementation["files"].items()}
    if snapshot["frozen_files"] != expected_files:
        raise TemporalPreflightError(
            "source snapshot frozen_files differ from config implementation_contract.files"
        )
    for relative, expected_file_sha256 in expected_files.items():
        normalized = _normalized_relative_path(relative, label="frozen implementation path")
        _as_sha256(expected_file_sha256, label=f"frozen implementation {relative} SHA-256")
        if _sha256_file(
            _repo_file(root, normalized, label=f"frozen implementation {relative}")
        ) != (expected_file_sha256):
            raise TemporalPreflightError(f"frozen implementation hash changed: {relative}")

    exact_exclusions = sorted(
        {
            ATTEMPT_PREREGISTRATION_RELATIVE_PATH,
            SOURCE_SNAPSHOT_RELATIVE_PATH,
            terminal_relative_path,
            *prior_evidence_paths,
        }
    )
    tree = snapshot["content_tree"]
    declared_projection = _validate_declared_source_tree(
        tree,
        label="source snapshot content_tree",
        required_excluded_exact_paths=exact_exclusions,
        allowed_excluded_exact_paths=exact_exclusions,
    )
    observed_tree = _recompute_source_tree(
        root,
        excluded_exact_paths=exact_exclusions,
        allow_active_runtime_root_venv=True,
    )
    if any(tree[name] != observed_tree[name] for name in observed_tree):
        raise TemporalPreflightError("staged source content tree differs from source snapshot")
    if declared_projection != observed_tree:
        raise TemporalPreflightError("source snapshot declared and observed trees differ")
    for index, prior_tree in enumerate(prior_scientific_trees):
        if dict(prior_tree) != observed_tree:
            raise TemporalPreflightError(
                f"retry source differs from prior attempt {index + 1}; byte-identical retry required"
            )

    policy = _exact_object(
        snapshot["staging_policy"],
        {
            "credentials_present",
            "official_mobile_evaluation_present",
            "symlinks_present_before_launch",
        },
        label="source snapshot staging_policy",
    )
    if any(policy[name] is not False for name in policy):
        raise TemporalPreflightError("source snapshot staging policy must assert all false")
    return {
        "path": SOURCE_SNAPSHOT_RELATIVE_PATH,
        "sha256": observed_sha256,
        "content_tree": observed_tree,
        "git": git,
    }


def _validate_prior_attempt_receipt(
    *,
    root: Path,
    prior: Mapping[str, Any],
    run_id: str,
    config_sha256: str,
    current_machine_id: int,
    terminal_relative_path: str,
    expected_frozen_files: Mapping[str, Any],
    allowed_prior_evidence_paths: Sequence[str],
) -> PriorAttemptValidation:
    prior_attempt_id = str(prior["attempt_id"])
    outcome_relative = _prior_evidence_path(
        prior["outcome_receipt_path"],
        attempt_id=prior_attempt_id,
        filename="outcome.json",
        label="prior outcome receipt path",
    )
    outcome_path = _repo_file(root, outcome_relative, label="prior outcome receipt")
    outcome, outcome_sha256 = _load_json_object(outcome_path, label="prior outcome receipt")
    if outcome_sha256 != _as_sha256(
        prior["outcome_receipt_sha256"], label="prior outcome receipt SHA-256"
    ):
        raise TemporalPreflightError("prior outcome receipt SHA-256 changed")
    outcome = _exact_object(
        outcome,
        {
            "schema_version",
            "attempt_id",
            "attempt_ordinal",
            "run_id",
            "machine_id",
            "attempt_preregistration",
            "scientific_config_sha256",
            "source_snapshot_sha256",
            "source_snapshot",
            "execution_failure",
            "essential_manifest",
            "lifecycle",
            "usable_held_out_signals",
        },
        label="prior outcome receipt",
    )
    if (
        outcome["schema_version"] != "barun-mobile-temporal-prior-attempt-outcome-v1"
        or outcome["attempt_id"] != prior["attempt_id"]
        or outcome["attempt_ordinal"] != prior["attempt_ordinal"]
        or outcome["run_id"] != run_id
        or outcome["scientific_config_sha256"] != config_sha256
        or outcome["usable_held_out_signals"] != 0
        or prior["usable_held_out_signals"] != 0
    ):
        raise TemporalPreflightError(
            "prior outcome does not prove a byte-identical zero-signal attempt"
        )
    prior_attempt_reference = _exact_object(
        outcome["attempt_preregistration"],
        {"path", "sha256"},
        label="prior attempt preregistration reference",
    )
    prior_attempt_relative = _prior_evidence_path(
        prior_attempt_reference["path"],
        attempt_id=prior_attempt_id,
        filename="attempt-preregistration.json",
        label="prior attempt preregistration path",
    )
    prior_attempt_path = _repo_file(
        root, prior_attempt_relative, label="prior attempt preregistration"
    )
    prior_attempt, prior_attempt_sha256 = _load_json_object(
        prior_attempt_path, label="prior attempt preregistration"
    )
    if prior_attempt_sha256 != _as_sha256(
        prior_attempt_reference["sha256"], label="prior attempt preregistration SHA-256"
    ):
        raise TemporalPreflightError("prior attempt preregistration SHA-256 changed")
    prior_machine_id = outcome["machine_id"]
    if (
        type(prior_machine_id) is not int
        or prior_machine_id <= 0
        or prior_machine_id == current_machine_id
    ):
        raise TemporalPreflightError("prior attempt machine ID is invalid or reused")
    prior_scientific = prior_attempt.get("scientific_config")
    prior_source = prior_attempt.get("source_snapshot")
    prior_inventory = prior_attempt.get("prelaunch_inventory")
    if (
        prior_attempt.get("schema_version") != ATTEMPT_PREREGISTRATION_SCHEMA_VERSION
        or prior_attempt.get("status") != ATTEMPT_STATUS
        or prior_attempt.get("run_id") != run_id
        or prior_attempt.get("attempt_id") != prior["attempt_id"]
        or prior_attempt.get("attempt_ordinal") != prior["attempt_ordinal"]
        or not isinstance(prior_scientific, Mapping)
        or prior_scientific.get("sha256") != config_sha256
        or not isinstance(prior_source, Mapping)
        or prior_source.get("sha256") != outcome["source_snapshot_sha256"]
        or not isinstance(prior_inventory, Mapping)
        or prior_inventory.get("project_machine_id") != prior_machine_id
    ):
        raise TemporalPreflightError("prior attempt preregistration identity is not proven")

    references: dict[str, tuple[Path, dict[str, Any]]] = {}
    reference_relatives: dict[str, str] = {}
    for name in ("execution_failure", "lifecycle"):
        reference = _exact_object(
            outcome[name], {"path", "sha256"}, label=f"prior {name} reference"
        )
        filename = "essential/failure.json" if name == "execution_failure" else "lifecycle.json"
        relative = _prior_evidence_path(
            reference["path"],
            attempt_id=prior_attempt_id,
            filename=filename,
            label=f"prior {name} path",
        )
        path = _repo_file(root, relative, label=f"prior {name}")
        payload, observed_sha256 = _load_json_object(path, label=f"prior {name}")
        if observed_sha256 != _as_sha256(reference["sha256"], label=f"prior {name} SHA-256"):
            raise TemporalPreflightError(f"prior {name} SHA-256 changed")
        references[name] = (path, payload)
        reference_relatives[name] = relative

    manifest_reference = _exact_object(
        outcome["essential_manifest"], {"path", "sha256"}, label="prior essential manifest"
    )
    manifest_relative = _prior_evidence_path(
        manifest_reference["path"],
        attempt_id=prior_attempt_id,
        filename="essential/artifact-manifest.json",
        label="prior essential manifest path",
    )
    manifest_path = _repo_file(root, manifest_relative, label="prior essential manifest")
    essential_manifest, manifest_sha256 = _load_json_object(
        manifest_path, label="prior essential manifest"
    )
    if manifest_sha256 != _as_sha256(
        manifest_reference["sha256"], label="prior essential manifest SHA-256"
    ):
        raise TemporalPreflightError("prior essential manifest SHA-256 changed")
    manifest_keys = set(essential_manifest)
    required_manifest_keys = {"schema_version", "file_count", "total_bytes", "files"}
    allowed_manifest_keys = required_manifest_keys | {
        "created_at",
        "optimizer_files",
        "screening_model_weight_files",
        "promoted_model_weight_files",
        "status",
    }
    if (
        essential_manifest.get("schema_version") != "barun-mobile-temporal-essential-v1"
        or not required_manifest_keys <= manifest_keys
        or not manifest_keys <= allowed_manifest_keys
        or not isinstance(essential_manifest.get("files"), Mapping)
    ):
        raise TemporalPreflightError("prior essential manifest schema is invalid")
    manifest_files = essential_manifest["files"]
    if (
        type(essential_manifest.get("file_count")) is not int
        or essential_manifest["file_count"] != len(manifest_files)
        or essential_manifest["file_count"] <= 0
        or essential_manifest["file_count"] > 10_000
    ):
        raise TemporalPreflightError("prior essential manifest file count is invalid")
    essential_root_relative = f"{RETRY_EVIDENCE_ROOT}/{prior_attempt_id}/essential"
    archived_essential_paths: set[str] = {manifest_relative}
    observed_total_bytes = 0
    heldout_names = {
        "predictions.jsonl",
        "sample_scores.jsonl",
        "heldout_access_started.json",
        "selection-heldout-access-started.json",
        "terminal-heldout-access-started.json",
    }
    known_metric_events = {
        "checkpoint",
        "complete",
        "completion_only_contract",
        "dev",
        "resume",
        "train",
    }
    for relative_value, entry_value in sorted(manifest_files.items()):
        relative = _normalized_relative_path(
            relative_value, label="prior essential manifest file path"
        )
        entry = _exact_object(
            entry_value, {"bytes", "sha256"}, label=f"prior essential file {relative}"
        )
        expected_bytes = entry["bytes"]
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise TemporalPreflightError("prior essential manifest file size is invalid")
        expected_digest = _as_sha256(
            entry["sha256"], label=f"prior essential file {relative} SHA-256"
        )
        archived_relative = f"{essential_root_relative}/{relative}"
        archived_path = _repo_file(root, archived_relative, label="prior essential artifact")
        if (
            archived_path.stat().st_size != expected_bytes
            or _sha256_file(archived_path) != expected_digest
        ):
            raise TemporalPreflightError(f"prior essential artifact changed: {relative}")
        archived_essential_paths.add(archived_relative)
        observed_total_bytes += expected_bytes
        relative_path = Path(relative)
        if relative_path.name in heldout_names or any(
            part in {"selection", "confirmation", "terminal-compatibility-veto"}
            for part in relative_path.parts
        ):
            raise TemporalPreflightError("prior essential manifest proves a held-out signal")
        if relative_path.name == "metrics.jsonl":
            try:
                metric_rows = [
                    _loads_json(line, label=f"prior metrics {relative}")
                    for line in archived_path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
            except (OSError, UnicodeDecodeError) as error:
                raise TemporalPreflightError("prior metrics evidence is unreadable") from error
            if any(
                not isinstance(row, Mapping)
                or row.get("event") not in known_metric_events
                or row.get("event") == "dev"
                for row in metric_rows
            ):
                raise TemporalPreflightError("prior metrics contain or may contain held-out signal")
    if (
        type(essential_manifest.get("total_bytes")) is not int
        or essential_manifest["total_bytes"] != observed_total_bytes
        or observed_total_bytes > 2 * 1024 * 1024 * 1024
    ):
        raise TemporalPreflightError("prior essential manifest byte accounting is invalid")
    failure_entry = manifest_files.get("failure.json")
    if (
        not isinstance(failure_entry, Mapping)
        or failure_entry.get("sha256") != outcome["execution_failure"]["sha256"]
        or references["execution_failure"][0]
        != _repo_file(root, f"{essential_root_relative}/failure.json", label="manifest failure")
    ):
        raise TemporalPreflightError("prior standalone failure is not bound to essential manifest")

    prior_source_reference = _exact_object(
        outcome["source_snapshot"],
        {"path", "sha256"},
        label="prior preserved source snapshot reference",
    )
    prior_source_relative = _prior_evidence_path(
        prior_source_reference["path"],
        attempt_id=prior_attempt_id,
        filename="source-snapshot.json",
        label="prior preserved source snapshot path",
    )
    prior_source_path = _repo_file(root, prior_source_relative, label="prior source snapshot")
    prior_snapshot, prior_snapshot_sha256 = _load_json_object(
        prior_source_path, label="prior source snapshot"
    )
    expected_prior_source_sha256 = _as_sha256(
        prior_source_reference["sha256"], label="prior preserved source snapshot SHA-256"
    )
    if (
        prior_snapshot_sha256 != expected_prior_source_sha256
        or outcome["source_snapshot_sha256"] != expected_prior_source_sha256
        or prior_source.get("path") != SOURCE_SNAPSHOT_RELATIVE_PATH
        or prior_source.get("sha256") != expected_prior_source_sha256
    ):
        raise TemporalPreflightError("prior source snapshot hash binding changed")
    prior_snapshot = _exact_object(
        prior_snapshot,
        {"schema_version", "run_id", "git", "content_tree", "frozen_files", "staging_policy"},
        label="prior source snapshot",
    )
    if (
        prior_snapshot["schema_version"] != SOURCE_SNAPSHOT_SCHEMA_VERSION
        or prior_snapshot["run_id"] != run_id
        or prior_snapshot["frozen_files"] != dict(expected_frozen_files)
    ):
        raise TemporalPreflightError("prior source snapshot identity or frozen files changed")
    prior_git = _exact_object(
        prior_snapshot["git"], {"repository", "commit", "branch"}, label="prior snapshot git"
    )
    if (
        any(not isinstance(prior_git[name], str) or not prior_git[name] for name in prior_git)
        or re.fullmatch(r"[0-9a-f]{40}", prior_git["commit"]) is None
    ):
        raise TemporalPreflightError("prior source snapshot git identity is invalid")
    prior_policy = _exact_object(
        prior_snapshot["staging_policy"],
        {
            "credentials_present",
            "official_mobile_evaluation_present",
            "symlinks_present_before_launch",
        },
        label="prior source snapshot staging policy",
    )
    if any(prior_policy[name] is not False for name in prior_policy):
        raise TemporalPreflightError("prior source snapshot staging policy is not clean")
    base_exclusions = {
        ATTEMPT_PREREGISTRATION_RELATIVE_PATH,
        SOURCE_SNAPSHOT_RELATIVE_PATH,
        terminal_relative_path,
    }
    evidence_paths = {
        outcome_relative,
        prior_attempt_relative,
        prior_source_relative,
        manifest_relative,
        *reference_relatives.values(),
        *archived_essential_paths,
    }
    scientific_content_tree = _validate_declared_source_tree(
        prior_snapshot["content_tree"],
        label="prior source snapshot content_tree",
        required_excluded_exact_paths=sorted(base_exclusions),
        allowed_excluded_exact_paths=sorted(
            base_exclusions | set(allowed_prior_evidence_paths) | evidence_paths
        ),
    )

    failure = references["execution_failure"][1]
    observed_failure_machine = failure.get("jarvis_resource_id")
    if isinstance(observed_failure_machine, str) and observed_failure_machine.isdigit():
        observed_failure_machine = int(observed_failure_machine)
    if (
        failure.get("schema_version") != "barun-mobile-temporal-execution-failure-v1"
        or observed_failure_machine != prior_machine_id
        or failure.get("automatic_retry_attempted") is not False
        or failure.get("usable_held_out_artifacts") != []
        or failure.get("retry_after_this_attempt") != "external_controller_must_verify_eligibility"
        or failure.get("official_961_rows_read") != 0
        or failure.get("reused_756_rows_read") != 0
        or failure.get("reused_756_rows_scored") != 0
        or failure.get("artifacts_preserved") is not True
    ):
        raise TemporalPreflightError("prior failure does not prove zero held-out signal")

    lifecycle = references["lifecycle"][1]
    final_instance = lifecycle.get("final_instance")
    final_run = lifecycle.get("final_run_status")
    collection = lifecycle.get("evidence_collection")
    inventory_binding = lifecycle.get("attempt_inventory_binding")
    if (
        lifecycle.get("machine_id") != prior_machine_id
        or not isinstance(final_instance, Mapping)
        or final_instance.get("machine_id") != prior_machine_id
        or str(final_instance.get("status", "")).casefold() != "paused"
        or not isinstance(final_run, Mapping)
        or final_run.get("machine_id") != prior_machine_id
        or str(final_run.get("state", "")).casefold() not in {"failed", "stopped"}
        or not isinstance(collection, Mapping)
        or collection.get("artifact_status") != "succeeded"
        or collection.get("log_status") != "succeeded"
        or collection.get("artifact_manifest_path") != manifest_relative
        or collection.get("artifact_manifest_sha256") != manifest_sha256
        or not isinstance(inventory_binding, Mapping)
        or inventory_binding.get("machine_id") != prior_machine_id
        or inventory_binding.get("after_sha256") != prior_attempt_sha256
    ):
        raise TemporalPreflightError(
            "prior lifecycle lacks exact failed-run evidence and pause proof"
        )
    return PriorAttemptValidation(
        machine_id=prior_machine_id,
        scientific_content_tree=scientific_content_tree,
        evidence_paths=tuple(sorted(evidence_paths)),
    )


def _validate_attempt_preregistration(
    *,
    attempt_path: Path,
    config_path: Path,
    config: Mapping[str, Any],
    materialization_receipt: Path,
    terminal_binding: Mapping[str, Any],
    jarvis_resource_id: str,
    expected_attempt_sha256: str,
) -> dict[str, Any]:
    root = REPOSITORY_ROOT.resolve(strict=True)
    expected_attempt_path = root / ATTEMPT_PREREGISTRATION_RELATIVE_PATH
    if attempt_path.is_symlink() or attempt_path.resolve(strict=True) != expected_attempt_path:
        raise TemporalPreflightError(
            "attempt preregistration must be target-root attempt-preregistration.json"
        )
    attempt, attempt_sha256 = _load_json_object(attempt_path, label="attempt preregistration")
    if attempt_sha256 != _as_sha256(
        expected_attempt_sha256, label="CLI attempt preregistration SHA-256"
    ):
        raise TemporalPreflightError(
            "attempt preregistration SHA-256 differs from independent CLI binding"
        )
    attempt = _exact_object(
        attempt,
        {
            "schema_version",
            "status",
            "run_id",
            "attempt_id",
            "attempt_ordinal",
            "prior_attempts",
            "durable_protected_machine_ids",
            "scientific_config",
            "source_snapshot",
            "materialization",
            "terminal_population",
            "prelaunch_inventory",
            "compute",
            "retry_lock",
            "official_evaluation",
        },
        label="attempt preregistration",
    )
    if attempt["schema_version"] != ATTEMPT_PREREGISTRATION_SCHEMA_VERSION:
        raise TemporalPreflightError("attempt preregistration schema changed")
    if attempt["status"] != ATTEMPT_STATUS:
        raise TemporalPreflightError("attempt preregistration status changed")
    if attempt["run_id"] != config.get("run_id"):
        raise TemporalPreflightError("attempt preregistration run_id differs from frozen config")
    attempt_id = attempt["attempt_id"]
    if (
        not isinstance(attempt_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", attempt_id) is None
    ):
        raise TemporalPreflightError("attempt_id must be a stable non-empty identifier")
    ordinal = attempt["attempt_ordinal"]
    prior_attempts = attempt["prior_attempts"]
    if type(ordinal) is not int or ordinal < 1 or not isinstance(prior_attempts, list):
        raise TemporalPreflightError("attempt ordinal/prior-attempt history is invalid")
    retry_policy = config.get("retry_policy")
    maximum_retries = (
        retry_policy.get("maximum_byte_identical_infrastructure_retries")
        if isinstance(retry_policy, Mapping)
        else None
    )
    if type(maximum_retries) is not int or maximum_retries != 2 or ordinal > 1 + maximum_retries:
        raise TemporalPreflightError("attempt ordinal exceeds the frozen retry budget")
    expected_attempt_id = f"{config['run_id']}-attempt-{ordinal}"
    if attempt_id != expected_attempt_id:
        raise TemporalPreflightError("attempt_id does not match run_id and attempt ordinal")
    if len(prior_attempts) != ordinal - 1:
        raise TemporalPreflightError("attempt ordinal does not match prior-attempt history")
    prior_ids: set[str] = set()
    for index, prior_value in enumerate(prior_attempts):
        prior = _exact_object(
            prior_value,
            {
                "attempt_id",
                "attempt_ordinal",
                "outcome_receipt_path",
                "outcome_receipt_sha256",
                "usable_held_out_signals",
            },
            label=f"prior attempt {index}",
        )
        prior_id = prior["attempt_id"]
        expected_prior_id = f"{config['run_id']}-attempt-{index + 1}"
        if (
            prior_id != expected_prior_id
            or prior["attempt_ordinal"] != index + 1
            or prior_id == attempt_id
            or prior_id in prior_ids
        ):
            raise TemporalPreflightError("prior attempt IDs are invalid or reused")
        prior_ids.add(prior_id)
        _as_sha256(prior["outcome_receipt_sha256"], label=f"prior attempt {index} outcome receipt")
        if prior["usable_held_out_signals"] != 0:
            raise TemporalPreflightError("retry is forbidden after any usable held-out signal")
    durable_protected = attempt["durable_protected_machine_ids"]
    if durable_protected != sorted(KNOWN_PROTECTED_JARVIS_IDS):
        raise TemporalPreflightError("attempt durable protected-machine IDs changed")

    scientific = _exact_object(
        attempt["scientific_config"], {"path", "sha256"}, label="attempt scientific_config"
    )
    config_relative = _normalized_relative_path(
        scientific["path"], label="attempt scientific config path"
    )
    if _repo_file(root, config_relative, label="attempt scientific config") != config_path:
        raise TemporalPreflightError("attempt scientific config path differs from CLI config")
    config_sha256 = _sha256_file(config_path)
    if scientific["sha256"] != config_sha256:
        raise TemporalPreflightError("attempt scientific config SHA-256 changed")

    terminal = _exact_object(
        attempt["terminal_population"],
        {"path", "sha256", "rows", "access"},
        label="attempt terminal_population",
    )
    terminal_relative = _normalized_relative_path(
        terminal["path"], label="attempt terminal population path"
    )
    terminal_path = _repo_file(root, terminal_relative, label="attempt terminal population")
    if str(terminal_path) != terminal_binding.get("manifest"):
        raise TemporalPreflightError("attempt terminal path differs from CLI terminal binding")
    if terminal != {
        "path": terminal_relative,
        "sha256": PINNED_REUSED_756_SHA256,
        "rows": PINNED_REUSED_756_ROWS,
        "access": "only_after_confirmation_pass_and_full_refit",
    }:
        raise TemporalPreflightError("attempt terminal population contract changed")

    source = _exact_object(
        attempt["source_snapshot"], {"path", "sha256"}, label="attempt source_snapshot"
    )
    if source["path"] != SOURCE_SNAPSHOT_RELATIVE_PATH:
        raise TemporalPreflightError("attempt source snapshot path changed")
    source_sha256 = _as_sha256(source["sha256"], label="attempt source snapshot SHA-256")
    snapshot_path = _repo_file(root, source["path"], label="source snapshot")

    materialization = _exact_object(
        attempt["materialization"],
        {
            "screening_view_audit_path",
            "screening_view_audit_sha256",
            "full_view_audit_path",
            "full_view_audit_sha256",
        },
        label="attempt materialization",
    )
    screening_relative = _normalized_relative_path(
        materialization["screening_view_audit_path"], label="screening view-audit path"
    )
    screening_path = _repo_file(root, screening_relative, label="screening view audit")
    if screening_path != materialization_receipt:
        raise TemporalPreflightError("attempt screening view audit differs from CLI receipt")
    if (
        materialization["screening_view_audit_sha256"]
        != config["materialization"]["view_audit_sha256"]
        or _sha256_file(screening_path) != materialization["screening_view_audit_sha256"]
    ):
        raise TemporalPreflightError("attempt screening view-audit SHA-256 changed")
    full_relative = _normalized_relative_path(
        materialization["full_view_audit_path"], label="full view-audit path"
    )
    full_path = _repo_file(root, full_relative, label="full view audit")
    if (
        materialization["full_view_audit_sha256"]
        != config["conditional_full_refit"]["audit_sha256"]
        or _sha256_file(full_path) != materialization["full_view_audit_sha256"]
    ):
        raise TemporalPreflightError("attempt full view-audit SHA-256 changed")

    inventory = _exact_object(
        attempt["prelaunch_inventory"],
        {
            "captured_before_project_instance_creation",
            "protected_machine_ids",
            "project_machine_id",
            "fresh_project_instance",
        },
        label="attempt prelaunch_inventory",
    )
    if (
        inventory["captured_before_project_instance_creation"] is not True
        or inventory["fresh_project_instance"] is not True
    ):
        raise TemporalPreflightError("attempt inventory timing/fresh-instance assertions changed")
    protected = inventory["protected_machine_ids"]
    if (
        not isinstance(protected, list)
        or not protected
        or any(type(value) is not int or value <= 0 for value in protected)
        or len(protected) != len(set(protected))
        or protected != sorted(protected)
        or not set(durable_protected) <= set(protected)
    ):
        raise TemporalPreflightError("attempt protected-machine denylist is invalid")
    machine_id = int(jarvis_resource_id)
    if inventory["project_machine_id"] != machine_id or machine_id in protected:
        raise TemporalPreflightError("attempt project machine ID differs from the fresh CLI ID")
    prior_machine_ids: set[int] = set()
    prior_evidence_paths: set[str] = set()
    prior_scientific_trees: list[Mapping[str, Any]] = []
    implementation = config.get("implementation_contract")
    expected_frozen_files = (
        implementation.get("files") if isinstance(implementation, Mapping) else None
    )
    if not isinstance(expected_frozen_files, Mapping):
        raise TemporalPreflightError("config implementation_contract.files is missing")
    for prior in prior_attempts:
        prior_validation = _validate_prior_attempt_receipt(
            root=root,
            prior=prior,
            run_id=str(config["run_id"]),
            config_sha256=config_sha256,
            current_machine_id=machine_id,
            terminal_relative_path=terminal_relative,
            expected_frozen_files=expected_frozen_files,
            allowed_prior_evidence_paths=sorted(prior_evidence_paths),
        )
        prior_machine_id = prior_validation.machine_id
        if prior_machine_id in prior_machine_ids:
            raise TemporalPreflightError("fresh retry policy forbids reused prior machine IDs")
        prior_machine_ids.add(prior_machine_id)
        prior_evidence_paths.update(prior_validation.evidence_paths)
        prior_scientific_trees.append(prior_validation.scientific_content_tree)

    compute = _exact_object(
        attempt["compute"],
        {
            "provider",
            "template",
            "python_implementation",
            "python_version",
            "gpu",
            "num_gpus",
            "region",
            "is_spot",
            "max_gpu_job_minutes",
        },
        label="attempt compute",
    )
    if compute != {
        "provider": "JarvisLabs",
        "template": EXPECTED_JARVIS_TEMPLATE,
        "python_implementation": EXPECTED_PYTHON_IMPLEMENTATION,
        "python_version": EXPECTED_PYTHON_VERSION,
        "gpu": "H200",
        "num_gpus": 1,
        "region": "IN2",
        "is_spot": False,
        "max_gpu_job_minutes": 30,
    }:
        raise TemporalPreflightError("attempt compute contract changed")
    expected_retry_lock = {
        name: retry_policy.get(name) if isinstance(retry_policy, Mapping) else None
        for name in ("retry_after_any_held_out_signal", "retry_execution_policy")
    }
    if attempt["retry_lock"] != expected_retry_lock or any(
        not isinstance(value, str) or not value for value in expected_retry_lock.values()
    ):
        raise TemporalPreflightError("attempt retry lock differs from frozen config")
    if attempt["official_evaluation"] != {
        "policy": "forbidden",
        "rows": 961,
        "rows_present": 0,
        "rows_read": 0,
    }:
        raise TemporalPreflightError("attempt official-evaluation firewall changed")

    snapshot = _validate_source_snapshot(
        root=root,
        snapshot_path=snapshot_path,
        expected_sha256=source_sha256,
        config=config,
        terminal_relative_path=terminal_relative,
        prior_evidence_paths=sorted(prior_evidence_paths),
        prior_scientific_trees=prior_scientific_trees,
    )
    return {
        "status": "validated_before_torch_import_or_cuda_access",
        "attempt_preregistration": {
            "path": ATTEMPT_PREREGISTRATION_RELATIVE_PATH,
            "sha256": attempt_sha256,
            "attempt_id": attempt_id,
            "attempt_ordinal": ordinal,
            "project_machine_id": machine_id,
            "protected_machine_ids": protected,
        },
        "source_snapshot": snapshot,
    }


def _find_nested_source_ids(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "source_id":
                if not isinstance(nested, str) or not nested:
                    raise TemporalPreflightError("metadata source_id must be a non-empty string")
                found.append(nested)
            else:
                found.extend(_find_nested_source_ids(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_find_nested_source_ids(nested))
    return found


def _source_id(row: Mapping[str, Any]) -> str:
    example_id = row["id"]
    metadata = row["metadata"]
    found = _find_nested_source_ids(metadata)
    unique = set(found)
    if not unique:
        return str(example_id)
    if len(unique) != 1:
        raise TemporalPreflightError(
            f"row {example_id!r} has ambiguous nested metadata source_id values"
        )
    return next(iter(unique))


def _load_sft_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_derived_split: str,
) -> ManifestObservation:
    if path.suffix != ".jsonl" or not path.is_file():
        raise TemporalPreflightError(f"frozen SFT artifact is not a JSONL file: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise TemporalPreflightError(
            f"artifact SHA-256 changed for {path.name}: {actual_sha256} != {expected_sha256}"
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise TemporalPreflightError(f"blank JSONL line at {path}:{line_number}")
            row = _loads_json(line, label=f"{path}:{line_number}")
            if not isinstance(row, dict):
                raise TemporalPreflightError(f"non-object JSONL row at {path}:{line_number}")
            if row.get("schema_version") != SFT_SCHEMA_VERSION:
                raise TemporalPreflightError(f"SFT schema drift at {path}:{line_number}")
            example_id = row.get("id")
            prompt = row.get("prompt")
            target = row.get("target")
            metadata = row.get("metadata")
            if not isinstance(example_id, str) or not example_id:
                raise TemporalPreflightError(f"invalid example ID at {path}:{line_number}")
            if not isinstance(prompt, str) or not isinstance(target, str):
                raise TemporalPreflightError(f"invalid prompt/target at {path}:{line_number}")
            if not isinstance(metadata, dict):
                raise TemporalPreflightError(f"invalid metadata at {path}:{line_number}")
            if metadata.get("source_split") != "train":
                raise TemporalPreflightError(
                    f"non-training source entered {path.name} at line {line_number}"
                )
            if metadata.get("derived_split") != expected_derived_split:
                raise TemporalPreflightError(
                    f"derived split drift in {path.name} at line {line_number}"
                )
            rows.append(row)
    if len(rows) != expected_rows:
        raise TemporalPreflightError(
            f"row count changed for {path.name}: {len(rows)} != {expected_rows}"
        )
    example_ids = tuple(str(row["id"]) for row in rows)
    if len(example_ids) != len(set(example_ids)):
        raise TemporalPreflightError(f"duplicate example IDs in {path.name}")
    source_ids = tuple(_source_id(row) for row in rows)
    return ManifestObservation(
        path=path,
        sha256=actual_sha256,
        rows=tuple(rows),
        example_ids=example_ids,
        membership_sha256=_sequence_sha256(example_ids, sort_values=True),
        order_sha256=_sequence_sha256(example_ids),
        source_ids=source_ids,
        source_membership_sha256=_sequence_sha256(source_ids, sort_values=True),
        source_order_sha256=_sequence_sha256(source_ids),
    )


def _load_transform_receipts(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    relative_path: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    if path.suffix != ".jsonl" or not path.is_file():
        raise TemporalPreflightError(f"transform receipt artifact is not JSONL: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise TemporalPreflightError(f"transform receipt SHA-256 changed: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise TemporalPreflightError(f"blank JSONL line at {path}:{line_number}")
            row = _loads_json(line, label=f"{path}:{line_number}")
            if not isinstance(row, dict):
                raise TemporalPreflightError(f"non-object receipt at {path}:{line_number}")
            if row.get("schema_version") != TRANSFORM_RECEIPT_SCHEMA_VERSION:
                raise TemporalPreflightError(
                    f"transform receipt schema drift at line {line_number}"
                )
            source_id = row.get("source_id")
            variant_id = row.get("variant_id")
            invariants = row.get("invariants")
            if not isinstance(source_id, str) or not isinstance(variant_id, str):
                raise TemporalPreflightError(f"invalid transform identity at line {line_number}")
            if not isinstance(invariants, dict) or any(
                invariants.get(key) is not True
                for key in (
                    "cross_month",
                    "non_calendar_calls_unchanged",
                    "source_user_text_unchanged",
                )
            ):
                raise TemporalPreflightError(f"unproved transform invariant at line {line_number}")
            rows.append(row)
    if len(rows) != expected_rows:
        raise TemporalPreflightError(
            f"transform receipt count changed: {len(rows)} != {expected_rows}"
        )
    source_ids = tuple(str(row["source_id"]) for row in rows)
    variant_ids = tuple(str(row["variant_id"]) for row in rows)
    if len(source_ids) != len(set(source_ids)) or len(variant_ids) != len(set(variant_ids)):
        raise TemporalPreflightError("transform receipts contain duplicate source or variant IDs")
    summary = {
        "order_sha256": _sequence_sha256(variant_ids),
        "path": relative_path,
        "rows": len(rows),
        "sha256": actual_sha256,
        "source_membership_sha256": _sequence_sha256(source_ids, sort_values=True),
        "source_order_sha256": _sequence_sha256(source_ids),
    }
    return tuple(rows), summary


def _strict_datetime(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise TemporalPreflightError(f"{label} must be a non-empty ISO datetime")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise TemporalPreflightError(f"invalid ISO datetime for {label}: {value!r}") from error


def _calendar_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    same_month = 0
    cross_month = 0
    for row in rows:
        target = _loads_json(str(row["target"]), label=f"target for {row['id']}")
        if not isinstance(target, dict) or not isinstance(target.get("calls"), list):
            raise TemporalPreflightError(f"invalid Action IR calls for {row['id']!r}")
        calendar_calls = [
            call
            for call in target["calls"]
            if isinstance(call, dict) and call.get("tool") == "create_calendar_event"
        ]
        if len(calendar_calls) > 1:
            raise TemporalPreflightError(f"multiple calendar calls in {row['id']!r}")
        if not calendar_calls:
            continue
        match = _NOW_RE.search(str(row["prompt"]))
        if match is None:
            raise TemporalPreflightError(f"missing NOW contract in {row['id']!r}")
        now = _strict_datetime(match.group(1), label=f"NOW for {row['id']}")
        args = calendar_calls[0].get("args")
        if not isinstance(args, dict):
            raise TemporalPreflightError(f"invalid calendar args in {row['id']!r}")
        target_datetime = _strict_datetime(
            args.get("datetime"), label=f"calendar datetime for {row['id']}"
        )
        if (now.year, now.month) == (target_datetime.year, target_datetime.month):
            same_month += 1
        else:
            cross_month += 1
    return {
        "cross_month": cross_month,
        "rows": same_month + cross_month,
        "same_month": same_month,
    }


def _content_identity(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"prompt": row["prompt"], "target": row["target"]},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _metadata_string(row: Mapping[str, Any], name: str) -> str:
    value = row["metadata"].get(name)
    if not isinstance(value, str) or not value:
        raise TemporalPreflightError(f"row {row['id']!r} lacks metadata.{name}")
    return value


def _verify_shadow_disjointness(manifests: Mapping[str, ManifestObservation]) -> None:
    indexes: dict[str, dict[str, set[str]]] = {
        "example_id": {},
        "cluster_id": {},
        "family_id": {},
        "exact_prompt_target": {},
    }
    for role in ("construction", "selection", "confirmation"):
        rows = manifests[role].rows
        indexes["example_id"][role] = {str(row["id"]) for row in rows}
        indexes["cluster_id"][role] = {_metadata_string(row, "cluster_id") for row in rows}
        indexes["family_id"][role] = {_metadata_string(row, "family_id") for row in rows}
        indexes["exact_prompt_target"][role] = {_content_identity(row) for row in rows}
    roles = ("construction", "selection", "confirmation")
    for label, by_role in indexes.items():
        for left_index, left in enumerate(roles):
            for right in roles[left_index + 1 :]:
                if by_role[left] & by_role[right]:
                    raise TemporalPreflightError(
                        f"shadow {label} overlap between {left} and {right}"
                    )


def _verify_training_view_parity(
    construction: ManifestObservation,
    repeat: ManifestObservation,
    counterfactual: ManifestObservation,
    transform_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    construction_by_id = {str(row["id"]): row for row in construction.rows}
    construction_ids = set(construction_by_id)
    for label, view in (("repeat", repeat), ("counterfactual", counterfactual)):
        view_by_id = {str(row["id"]): row for row in view.rows}
        if not construction_ids <= set(view_by_id):
            raise TemporalPreflightError(f"{label} view dropped construction rows")
        if any(view_by_id[example_id] != row for example_id, row in construction_by_id.items()):
            raise TemporalPreflightError(f"{label} view changed a construction row")

    if repeat.source_ids != counterfactual.source_ids:
        raise TemporalPreflightError("B repeat and C mbcf source presentation order differs")
    repeat_counts = Counter(repeat.source_ids)
    counterfactual_counts = Counter(counterfactual.source_ids)
    construction_counts = Counter(construction.example_ids)
    if repeat_counts != counterfactual_counts:
        raise TemporalPreflightError("B repeat and C mbcf source multiplicities differ")
    if set(repeat_counts) != set(construction_counts):
        raise TemporalPreflightError("B/C source IDs are not exactly the A construction sources")
    deltas = {
        source_id: repeat_counts[source_id] - count
        for source_id, count in construction_counts.items()
    }
    if any(delta not in {0, 1} for delta in deltas.values()):
        raise TemporalPreflightError("B/C must add at most one paired row per A source")
    safe_source_ids = tuple(source_id for source_id, delta in deltas.items() if delta == 1)
    safe_source_membership = _sequence_sha256(safe_source_ids, sort_values=True)

    repeat_added = [row for row in repeat.rows if str(row["id"]) not in construction_ids]
    counterfactual_added = [
        row for row in counterfactual.rows if str(row["id"]) not in construction_ids
    ]
    if len(repeat_added) != len(safe_source_ids) or len(counterfactual_added) != len(
        safe_source_ids
    ):
        raise TemporalPreflightError("B/C added-row counts do not match safe source pairs")
    for row in repeat_added:
        source_id = _source_id(row)
        view = row["metadata"].get("mobile_temporal_view")
        if not isinstance(view, dict) or view.get("kind") != "repeat_control":
            raise TemporalPreflightError("B contains a non-repeat treatment row")
        source = construction_by_id[source_id]
        if row["prompt"] != source["prompt"] or row["target"] != source["target"]:
            raise TemporalPreflightError("B repeat changed source prompt or target")
    for row in counterfactual_added:
        view = row["metadata"].get("mobile_temporal_view")
        if not isinstance(view, dict) or view.get("kind") != "month_boundary_counterfactual":
            raise TemporalPreflightError("C contains a non-counterfactual treatment row")
        source = construction_by_id[_source_id(row)]
        if row["prompt"] == source["prompt"]:
            raise TemporalPreflightError("C counterfactual did not change the NOW-bearing prompt")

    receipt_source_ids = tuple(str(row["source_id"]) for row in transform_receipts)
    receipt_variant_ids = tuple(str(row["variant_id"]) for row in transform_receipts)
    if receipt_source_ids != tuple(_source_id(row) for row in counterfactual_added):
        raise TemporalPreflightError("transform receipt order differs from C treatment order")
    if receipt_variant_ids != tuple(str(row["id"]) for row in counterfactual_added):
        raise TemporalPreflightError("transform receipt variant IDs differ from C")
    if set(receipt_source_ids) != set(safe_source_ids):
        raise TemporalPreflightError("transform receipts do not cover exactly the paired sources")

    return {
        "safe_source_membership_sha256": safe_source_membership,
        "safe_variants": len(safe_source_ids),
        "source_membership_sha256": repeat.source_membership_sha256,
        "source_order_sha256": repeat.source_order_sha256,
    }


def _verify_conditional_full_refit(
    *,
    screening_dir: Path,
    frozen: Mapping[str, Any],
    optimization: Mapping[str, Any],
) -> dict[str, Any]:
    full_dir = screening_dir.parent / "materialized-full-refit-v1"
    audit_path = full_dir / EXPECTED_FULL_REFIT_FILENAMES["audit"]
    train_path = full_dir / EXPECTED_FULL_REFIT_FILENAMES["train"]
    receipts_path = full_dir / EXPECTED_FULL_REFIT_FILENAMES["receipts"]
    audit, audit_sha256 = _load_json_object(audit_path, label="conditional full-refit audit")
    _expect_audit_equal(
        audit_sha256, frozen["audit_sha256"], label="conditional full-refit audit SHA-256"
    )
    _expect_audit_equal(
        audit.get("schema_version"),
        FULL_VIEW_AUDIT_SCHEMA_VERSION,
        label="conditional full-refit audit schema",
    )
    source = audit.get("source")
    if not isinstance(source, dict):
        raise TemporalPreflightError("conditional full-refit source identity is missing")
    for key, expected in (
        ("sha256", PINNED_TRAIN_SHA256),
        ("rows", PINNED_TRAIN_ROWS),
        ("membership_sha256", PINNED_TRAIN_MEMBERSHIP_SHA256),
    ):
        _expect_audit_equal(source.get(key), expected, label=f"full-refit source {key}")
    _expect_audit_equal(frozen["source_rows"], PINNED_TRAIN_ROWS, label="full-refit source rows")
    _expect_audit_equal(audit.get("tokenizer"), EXPECTED_TOKENIZER, label="full-refit tokenizer")
    _expect_audit_equal(audit.get("development_rows_read"), 0, label="full-refit development reads")
    _expect_audit_equal(
        audit.get("official_evaluation_rows_read"), 0, label="full-refit official reads"
    )

    output = audit.get("output")
    if not isinstance(output, dict):
        raise TemporalPreflightError("conditional full-refit output identity is missing")
    expected_output = {
        "filename": EXPECTED_FULL_REFIT_FILENAMES["train"],
        "optimizer_steps_at_batch_63": frozen["optimizer_steps"],
        "rows": frozen["rows"],
        "sha256": frozen["train_sha256"],
    }
    _expect_audit_equal(output, expected_output, label="conditional full-refit output")
    receipt_audit = audit.get("receipts")
    _expect_audit_equal(
        receipt_audit,
        {
            "filename": EXPECTED_FULL_REFIT_FILENAMES["receipts"],
            "rows": frozen["variant_rows"],
            "sha256": frozen["receipts_sha256"],
        },
        label="conditional full-refit receipts",
    )

    full = _load_sft_manifest(
        train_path,
        expected_sha256=frozen["train_sha256"],
        expected_rows=frozen["rows"],
        expected_derived_split="train",
    )
    receipts, receipt_observation = _load_transform_receipts(
        receipts_path,
        expected_sha256=frozen["receipts_sha256"],
        expected_rows=frozen["variant_rows"],
        relative_path=(
            f"../materialized-full-refit-v1/{EXPECTED_FULL_REFIT_FILENAMES['receipts']}"
        ),
    )
    original_rows = [row for row in full.rows if "mobile_temporal_view" not in row["metadata"]]
    added_rows = [row for row in full.rows if "mobile_temporal_view" in row["metadata"]]
    if len(original_rows) != frozen["source_rows"] or len(added_rows) != frozen["variant_rows"]:
        raise TemporalPreflightError("conditional full-refit source/variant row counts changed")
    original_ids = tuple(str(row["id"]) for row in original_rows)
    if _sequence_sha256(original_ids, sort_values=True) != PINNED_TRAIN_MEMBERSHIP_SHA256:
        raise TemporalPreflightError("conditional full-refit original membership changed")
    original_by_id = {str(row["id"]): row for row in original_rows}
    source_counts = Counter(full.source_ids)
    if set(source_counts) != set(original_by_id) or any(
        count not in {1, 2} for count in source_counts.values()
    ):
        raise TemporalPreflightError("conditional full-refit source multiplicities changed")
    safe_source_ids = tuple(source_id for source_id, count in source_counts.items() if count == 2)
    if len(safe_source_ids) != frozen["variant_rows"]:
        raise TemporalPreflightError("conditional full-refit paired source count changed")
    safe_membership = _sequence_sha256(safe_source_ids, sort_values=True)
    _expect_audit_equal(
        safe_membership,
        frozen["safe_source_membership_sha256"],
        label="conditional full-refit safe membership",
    )
    for row in added_rows:
        view = row["metadata"].get("mobile_temporal_view")
        if not isinstance(view, dict) or view.get("kind") != "month_boundary_counterfactual":
            raise TemporalPreflightError("conditional full refit contains a non-mbcf added row")
        source_row = original_by_id[_source_id(row)]
        if row["prompt"] == source_row["prompt"]:
            raise TemporalPreflightError("conditional full-refit counterfactual did not change NOW")
    if tuple(str(row["source_id"]) for row in receipts) != tuple(
        _source_id(row) for row in added_rows
    ):
        raise TemporalPreflightError("conditional full-refit receipt source order changed")
    if tuple(str(row["variant_id"]) for row in receipts) != tuple(
        str(row["id"]) for row in added_rows
    ):
        raise TemporalPreflightError("conditional full-refit receipt variant order changed")

    variants = audit.get("variants")
    if not isinstance(variants, dict):
        raise TemporalPreflightError("conditional full-refit variant audit is missing")
    _expect_audit_equal(variants.get("rows"), len(added_rows), label="full-refit variants")
    _expect_audit_equal(
        variants.get("source_membership_sha256"),
        safe_membership,
        label="full-refit variant membership",
    )
    _expect_audit_equal(
        variants.get("minimum_required"),
        EXPECTED_FULL_MINIMUM_VARIANTS,
        label="full-refit minimum variants",
    )
    by_class = variants.get("by_class")
    if (
        not isinstance(by_class, dict)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in by_class.values()
        )
        or sum(by_class.values()) != len(added_rows)
    ):
        raise TemporalPreflightError("conditional full-refit variant classes are invalid")
    if len(added_rows) < EXPECTED_FULL_MINIMUM_VARIANTS:
        raise TemporalPreflightError(
            "conditional full-refit safe variants are below the frozen minimum"
        )

    original_calendar = _calendar_counts(original_rows)
    full_calendar = _calendar_counts(full.rows)
    expected_full_calendar = {
        "rows": original_calendar["rows"] + len(added_rows),
        "same_month": original_calendar["same_month"],
        "cross_month": original_calendar["cross_month"] + len(added_rows),
    }
    if full_calendar != expected_full_calendar:
        raise TemporalPreflightError("conditional full-refit additions are not all cross-month")
    cross_fraction = full_calendar["cross_month"] / full_calendar["rows"]
    minimum_fraction = float(EXPECTED_THRESHOLDS["minimum_cross_month_calendar_fraction_mbcf"])
    if cross_fraction < minimum_fraction:
        raise TemporalPreflightError("conditional full-refit cross-month fraction is below minimum")
    _expect_audit_equal(
        audit.get("calendar_presentations"),
        {
            "added_cross_month": len(added_rows),
            "minimum_fraction": minimum_fraction,
            "original_cross_month": original_calendar["cross_month"],
            "original_same_month": original_calendar["same_month"],
            "post_cross_month": full_calendar["cross_month"],
            "post_cross_month_fraction": cross_fraction,
            "post_total": full_calendar["rows"],
        },
        label="conditional full-refit calendar presentations",
    )
    optimizer_steps = _optimizer_steps(len(full.rows), optimization)
    _expect_audit_equal(
        optimizer_steps, frozen["optimizer_steps"], label="conditional full-refit optimizer steps"
    )
    return {
        "audit_sha256": audit_sha256,
        "calendar_presentations": {
            **full_calendar,
            "cross_month_fraction": cross_fraction,
        },
        "checkpoint_policy": "final_only",
        "receipts": receipt_observation,
        "safe_source_membership_sha256": safe_membership,
        "source_rows": len(original_rows),
        "status": "conditional_disabled_until_confirmation_gate_passes",
        "train": full.receipt(
            relative_path=(
                f"../materialized-full-refit-v1/{EXPECTED_FULL_REFIT_FILENAMES['train']}"
            )
        ),
        "variant_rows": len(added_rows),
    }


def _optimizer_steps(rows: int, optimization: Mapping[str, Any]) -> int:
    batches = math.ceil(rows / int(optimization["batch_size"]))
    per_epoch = math.ceil(batches / int(optimization["gradient_accumulation_steps"]))
    return per_epoch * int(optimization["epochs"])


def _validate_config(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise TemporalPreflightError(f"config schema must be {CONFIG_SCHEMA_VERSION!r}")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise TemporalPreflightError("run_id must match YYYYMMDD-HHMM-lowercase-name-sN")
    scientific_sha256 = _scientific_contract_sha256(payload)
    if scientific_sha256 != EXPECTED_SCIENTIFIC_CONTRACT_SHA256:
        raise TemporalPreflightError(
            "frozen scientific contract changed before model/CUDA access: "
            f"observed {scientific_sha256}, expected {EXPECTED_SCIENTIFIC_CONTRACT_SHA256}"
        )
    if payload.get("arms") != list(EXPECTED_ARMS):
        raise TemporalPreflightError("config must contain exactly A standard, B repeat, C mbcf")
    if payload.get("seeds") != list(EXPECTED_SEEDS):
        raise TemporalPreflightError("config seeds must be exactly [17, 29, 43]")
    if payload.get("checkpoint_policy") != "final_only":
        raise TemporalPreflightError("checkpoint policy must remain final_only")
    if payload.get("screening_fit_budget") != 9:
        raise TemporalPreflightError("screening fit budget must remain exactly nine")
    if payload.get("conditional_full_refit_budget") != 1:
        raise TemporalPreflightError("conditional full-refit budget must remain exactly one")
    if payload.get("optimization") != EXPECTED_OPTIMIZATION:
        raise TemporalPreflightError("frozen optimization or decoding recipe changed")
    if payload.get("thresholds") != EXPECTED_THRESHOLDS:
        raise TemporalPreflightError("frozen materialization thresholds changed")
    if payload.get("official_evaluation") != EXPECTED_OFFICIAL_EVALUATION:
        raise TemporalPreflightError("official 961-row evaluation firewall changed")
    materialization = payload.get("materialization")
    if (
        not isinstance(materialization, dict)
        or set(materialization) != _EXPECTED_MATERIALIZATION_KEYS
    ):
        raise TemporalPreflightError("config materialization identity is incomplete or ambiguous")
    _as_sha256(materialization.get("view_audit_sha256"), label="view audit SHA-256")
    _as_sha256(materialization.get("shadow_audit_sha256"), label="shadow audit SHA-256")
    artifact_hashes = materialization.get("artifact_sha256")
    artifact_rows = materialization.get("artifact_rows")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(
        EXPECTED_ARTIFACT_FILENAMES
    ):
        raise TemporalPreflightError("config must freeze all six JSONL artifact hashes")
    if not isinstance(artifact_rows, dict) or set(artifact_rows) != set(
        EXPECTED_ARTIFACT_FILENAMES
    ):
        raise TemporalPreflightError("config must freeze all six JSONL artifact row counts")
    for name in EXPECTED_ARTIFACT_FILENAMES:
        _as_sha256(artifact_hashes[name], label=f"artifact {name} SHA-256")
        _as_nonnegative_int(artifact_rows[name], label=f"artifact {name} rows")
    full_refit = payload.get("conditional_full_refit")
    if not isinstance(full_refit, dict) or set(full_refit) != _EXPECTED_FULL_REFIT_KEYS:
        raise TemporalPreflightError("conditional full-refit identity is incomplete or ambiguous")
    for name in (
        "audit_sha256",
        "receipts_sha256",
        "safe_source_membership_sha256",
        "train_sha256",
    ):
        _as_sha256(full_refit.get(name), label=f"conditional full refit {name}")
    for name in ("optimizer_steps", "rows", "seed", "source_rows", "variant_rows"):
        _as_nonnegative_int(full_refit.get(name), label=f"conditional full refit {name}")
    if full_refit["seed"] != 17:
        raise TemporalPreflightError("conditional full refit seed must remain 17")


def _expect_audit_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise TemporalPreflightError(f"{label} changed: observed {actual!r}, expected {expected!r}")


def _view_entry(view_audit: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    views = view_audit.get("views")
    if not isinstance(views, dict) or set(views) != {"standard", "repeat", "counterfactual"}:
        raise TemporalPreflightError("view audit must contain exactly the three frozen views")
    entry = views.get(name)
    if not isinstance(entry, dict):
        raise TemporalPreflightError(f"view audit lacks {name!r}")
    return entry


def _preflight(
    *, config_path: Path, materialization_receipt: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config, config_sha256 = _load_json_object(config_path, label="experiment config")
    _validate_config(config)
    if materialization_receipt.name != "view-audit.json":
        raise TemporalPreflightError("materialization receipt must be the frozen view-audit.json")
    view_audit, view_audit_sha256 = _load_json_object(materialization_receipt, label="view audit")
    shadow_audit_path = materialization_receipt.parent / "shadow-audit.json"
    shadow_audit, shadow_audit_sha256 = _load_json_object(shadow_audit_path, label="shadow audit")
    frozen = config["materialization"]
    _expect_audit_equal(
        view_audit_sha256,
        frozen["view_audit_sha256"],
        label="view audit SHA-256",
    )
    _expect_audit_equal(
        shadow_audit_sha256,
        frozen["shadow_audit_sha256"],
        label="shadow audit SHA-256",
    )
    _expect_audit_equal(
        view_audit.get("schema_version"), VIEW_AUDIT_SCHEMA_VERSION, label="view audit schema"
    )
    _expect_audit_equal(
        shadow_audit.get("schema_version"),
        SHADOW_AUDIT_SCHEMA_VERSION,
        label="shadow audit schema",
    )

    artifact_hashes = frozen["artifact_sha256"]
    artifact_rows = frozen["artifact_rows"]
    for name, filename in EXPECTED_ARTIFACT_FILENAMES.items():
        path = materialization_receipt.parent / filename
        if not path.is_file():
            raise TemporalPreflightError(f"missing frozen artifact {name}: {path}")

    standard_entry = _view_entry(view_audit, "standard")
    repeat_entry = _view_entry(view_audit, "repeat")
    counterfactual_entry = _view_entry(view_audit, "counterfactual")
    for name, entry, artifact_name in (
        ("standard", standard_entry, "construction"),
        ("repeat", repeat_entry, "repeat"),
        ("counterfactual", counterfactual_entry, "counterfactual"),
    ):
        _expect_audit_equal(
            entry.get("filename"),
            EXPECTED_ARTIFACT_FILENAMES[artifact_name],
            label=f"{name} filename",
        )
        _expect_audit_equal(
            entry.get("sha256"), artifact_hashes[artifact_name], label=f"{name} hash"
        )
        _expect_audit_equal(entry.get("rows"), artifact_rows[artifact_name], label=f"{name} rows")

    roles = shadow_audit.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(EXPECTED_SHADOW_ROWS):
        raise TemporalPreflightError("shadow audit roles changed")
    role_artifacts = {
        "construction_train": "construction",
        "selection": "selection",
        "confirmation": "confirmation",
    }
    for role, artifact_name in role_artifacts.items():
        entry = roles[role]
        if not isinstance(entry, dict):
            raise TemporalPreflightError(f"invalid shadow role {role}")
        _expect_audit_equal(entry.get("rows"), EXPECTED_SHADOW_ROWS[role], label=f"{role} rows")
        _expect_audit_equal(
            entry.get("membership_sha256"),
            EXPECTED_SHADOW_MEMBERSHIP_SHA256[role],
            label=f"{role} membership",
        )
        _expect_audit_equal(
            entry.get("manifest_sha256"),
            artifact_hashes[artifact_name],
            label=f"{role} manifest hash",
        )
        _expect_audit_equal(
            artifact_rows[artifact_name], EXPECTED_SHADOW_ROWS[role], label=f"{role} frozen rows"
        )

    source = shadow_audit.get("source")
    if not isinstance(source, dict):
        raise TemporalPreflightError("shadow audit source identity is missing")
    for key, expected in (
        ("sha256", PINNED_TRAIN_SHA256),
        ("rows", PINNED_TRAIN_ROWS),
        ("membership_sha256", PINNED_TRAIN_MEMBERSHIP_SHA256),
    ):
        _expect_audit_equal(source.get(key), expected, label=f"source {key}")
    split_policy = shadow_audit.get("split_policy")
    if not isinstance(split_policy, dict):
        raise TemporalPreflightError("shadow split policy is missing")
    _expect_audit_equal(
        split_policy.get("version"), SHADOW_SPLIT_VERSION, label="shadow split version"
    )
    _expect_audit_equal(shadow_audit.get("development_rows_read"), 0, label="development reads")
    _expect_audit_equal(
        shadow_audit.get("official_evaluation_rows_read"), 0, label="official evaluation reads"
    )
    _expect_audit_equal(
        shadow_audit.get("overlap"),
        {"cluster_id": 0, "exact_prompt_target": 0, "example_id": 0, "family_id": 0},
        label="shadow overlap audit",
    )

    manifest_specs = {
        "construction": ("train", artifact_rows["construction"]),
        "selection": ("dev", artifact_rows["selection"]),
        "confirmation": ("confirmation", artifact_rows["confirmation"]),
        "repeat": ("train", artifact_rows["repeat"]),
        "counterfactual": ("train", artifact_rows["counterfactual"]),
    }
    manifests = {
        name: _load_sft_manifest(
            materialization_receipt.parent / EXPECTED_ARTIFACT_FILENAMES[name],
            expected_sha256=artifact_hashes[name],
            expected_rows=rows,
            expected_derived_split=derived_split,
        )
        for name, (derived_split, rows) in manifest_specs.items()
    }
    _verify_shadow_disjointness(manifests)
    for role, artifact_name in role_artifacts.items():
        _expect_audit_equal(
            manifests[artifact_name].membership_sha256,
            EXPECTED_SHADOW_MEMBERSHIP_SHA256[role],
            label=f"observed {role} membership",
        )

    transform_receipts, receipt_observation = _load_transform_receipts(
        materialization_receipt.parent / EXPECTED_ARTIFACT_FILENAMES["receipts"],
        expected_sha256=artifact_hashes["receipts"],
        expected_rows=artifact_rows["receipts"],
        relative_path=EXPECTED_ARTIFACT_FILENAMES["receipts"],
    )
    parity = _verify_training_view_parity(
        manifests["construction"],
        manifests["repeat"],
        manifests["counterfactual"],
        transform_receipts,
    )
    minimum_variants = int(EXPECTED_THRESHOLDS["minimum_safe_variants"])
    if parity["safe_variants"] < minimum_variants:
        raise TemporalPreflightError(
            f"safe variants {parity['safe_variants']} are below {minimum_variants}"
        )
    if artifact_rows["receipts"] != parity["safe_variants"]:
        raise TemporalPreflightError("receipt rows do not equal accepted transform pairs")

    eligibility = view_audit.get("eligibility")
    if not isinstance(eligibility, dict):
        raise TemporalPreflightError("view audit eligibility is missing")
    for key, expected in (
        ("safe_variant_rows", parity["safe_variants"]),
        ("safe_source_membership_sha256", parity["safe_source_membership_sha256"]),
        ("minimum_required", minimum_variants),
    ):
        _expect_audit_equal(eligibility.get(key), expected, label=f"eligibility {key}")

    matched_control = view_audit.get("matched_control")
    if not isinstance(matched_control, dict):
        raise TemporalPreflightError("matched-control audit is missing")
    for key in ("same_source_ids", "same_added_rows", "same_optimizer_steps"):
        _expect_audit_equal(matched_control.get(key), True, label=f"matched control {key}")
    _expect_audit_equal(
        matched_control.get("source_membership_sha256"),
        parity["safe_source_membership_sha256"],
        label="matched-control safe source membership",
    )

    construction_calendar = _calendar_counts(manifests["construction"].rows)
    repeat_calendar = _calendar_counts(manifests["repeat"].rows)
    counterfactual_calendar = _calendar_counts(manifests["counterfactual"].rows)
    safe_variants = int(parity["safe_variants"])
    if repeat_calendar != {
        "rows": construction_calendar["rows"] + safe_variants,
        "same_month": construction_calendar["same_month"] + safe_variants,
        "cross_month": construction_calendar["cross_month"],
    }:
        raise TemporalPreflightError("B repeat calendar presentations changed temporal labels")
    if counterfactual_calendar != {
        "rows": construction_calendar["rows"] + safe_variants,
        "same_month": construction_calendar["same_month"],
        "cross_month": construction_calendar["cross_month"] + safe_variants,
    }:
        raise TemporalPreflightError("C mbcf added rows are not all cross-month presentations")
    cross_fraction = counterfactual_calendar["cross_month"] / counterfactual_calendar["rows"]
    minimum_fraction = float(EXPECTED_THRESHOLDS["minimum_cross_month_calendar_fraction_mbcf"])
    if cross_fraction < minimum_fraction:
        raise TemporalPreflightError(
            f"C cross-month calendar fraction {cross_fraction:.12f} is below {minimum_fraction}"
        )
    calendar_audit = view_audit.get("calendar_presentations")
    if not isinstance(calendar_audit, dict):
        raise TemporalPreflightError("calendar-presentation audit is missing")
    expected_calendar_audit = {
        "added_cross_month": safe_variants,
        "minimum_fraction": minimum_fraction,
        "original_cross_month": construction_calendar["cross_month"],
        "original_same_month": construction_calendar["same_month"],
        "post_cross_month": counterfactual_calendar["cross_month"],
        "post_cross_month_fraction": cross_fraction,
        "post_total": counterfactual_calendar["rows"],
    }
    _expect_audit_equal(calendar_audit, expected_calendar_audit, label="calendar presentations")

    receipt_audit = view_audit.get("receipts")
    if not isinstance(receipt_audit, dict):
        raise TemporalPreflightError("transform-receipt audit is missing")
    _expect_audit_equal(
        receipt_audit,
        {
            "filename": EXPECTED_ARTIFACT_FILENAMES["receipts"],
            "rows": artifact_rows["receipts"],
            "sha256": artifact_hashes["receipts"],
        },
        label="transform-receipt audit",
    )
    leakage = view_audit.get("leakage")
    _expect_audit_equal(
        leakage,
        {
            "confirmation_source_rows_used_for_variants": 0,
            "development_rows_read": 0,
            "official_evaluation_rows_read": 0,
            "selection_source_rows_used_for_variants": 0,
        },
        label="view leakage firewall",
    )
    tokenizer = view_audit.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise TemporalPreflightError("tokenizer identity is absent from view audit")
    _expect_audit_equal(tokenizer, EXPECTED_TOKENIZER, label="tokenizer identity")

    steps = {
        name: _optimizer_steps(len(manifests[artifact].rows), EXPECTED_OPTIMIZATION)
        for name, artifact in (
            ("standard", "construction"),
            ("repeat", "repeat"),
            ("counterfactual", "counterfactual"),
        )
    }
    if steps["repeat"] != steps["counterfactual"]:
        raise TemporalPreflightError("B/C optimizer-step budgets differ")
    for name, entry in (
        ("standard", standard_entry),
        ("repeat", repeat_entry),
        ("counterfactual", counterfactual_entry),
    ):
        _expect_audit_equal(
            entry.get("optimizer_steps_at_batch_63"), steps[name], label=f"{name} optimizer steps"
        )

    full_refit = _verify_conditional_full_refit(
        screening_dir=materialization_receipt.parent,
        frozen=config["conditional_full_refit"],
        optimization=EXPECTED_OPTIMIZATION,
    )

    observed_artifacts = {
        name: manifests[name].receipt(relative_path=EXPECTED_ARTIFACT_FILENAMES[name])
        for name in ("construction", "selection", "confirmation", "repeat", "counterfactual")
    }
    observed_artifacts["receipts"] = receipt_observation
    preflight_receipt = {
        "schema_version": "barun-mobile-temporal-preflight-receipt-v1",
        "run_id": config["run_id"],
        "config_sha256": config_sha256,
        "scientific_contract_sha256": _scientific_contract_sha256(config),
        "view_audit_sha256": view_audit_sha256,
        "shadow_audit_sha256": shadow_audit_sha256,
        "artifacts": observed_artifacts,
        "arms": list(EXPECTED_ARMS),
        "seeds": list(EXPECTED_SEEDS),
        "screening_fit_budget": 9,
        "conditional_full_refit_budget": 1,
        "checkpoint_policy": "final_only",
        "optimizer_steps": steps,
        "paired_control": parity,
        "calendar_presentations": {
            "standard": construction_calendar,
            "repeat": repeat_calendar,
            "mbcf": {
                **counterfactual_calendar,
                "cross_month_fraction": cross_fraction,
            },
        },
        "thresholds": dict(EXPECTED_THRESHOLDS),
        "official_evaluation": dict(EXPECTED_OFFICIAL_EVALUATION),
        "model_or_cuda_loaded": False,
        "conditional_full_refit": full_refit,
    }
    fit_specs = []
    timestamp = "-".join(str(config["run_id"]).split("-")[:2])
    arm_artifacts = {"A": "construction", "B": "repeat", "C": "counterfactual"}
    for seed in EXPECTED_SEEDS:
        for arm in EXPECTED_ARMS:
            artifact_name = arm_artifacts[arm["arm_id"]]
            fit_specs.append(
                {
                    "arm_id": arm["arm_id"],
                    "arm_name": arm["name"],
                    "checkpoint_policy": "final_only",
                    "expected_optimizer_steps": steps[arm["view"]],
                    "run_id": f"{timestamp}-mob-temporal-{arm['name']}-s{seed}",
                    "seed": seed,
                    "train_manifest": EXPECTED_ARTIFACT_FILENAMES[artifact_name],
                    "train_rows": len(manifests[artifact_name].rows),
                    "train_sha256": manifests[artifact_name].sha256,
                }
            )
    receipt_sha256 = hashlib.sha256(_canonical_bytes(preflight_receipt)).hexdigest()
    execution_plan = {
        "schema_version": "barun-mobile-temporal-execution-plan-v1",
        "run_id": config["run_id"],
        "config_sha256": config_sha256,
        "preflight_receipt_sha256": receipt_sha256,
        "execution_backend": EXECUTION_BACKEND_VERSION,
        "fit_count": len(fit_specs),
        "fits": fit_specs,
        "conditional_full_refit": {
            "authorization": "execute_flag_and_passing_confirmation_gate_required",
            "checkpoint_policy": "final_only",
            "expected_optimizer_steps": config["conditional_full_refit"]["optimizer_steps"],
            "run_id": f"{timestamp}-mob-temporal-full-mbcf-s17",
            "seed": 17,
            "status": "conditional_disabled_until_confirmation_gate_passes",
            "train_manifest": (
                f"../materialized-full-refit-v1/{EXPECTED_FULL_REFIT_FILENAMES['train']}"
            ),
            "train_rows": config["conditional_full_refit"]["rows"],
            "train_sha256": config["conditional_full_refit"]["train_sha256"],
        },
        "maximum_fit_count_if_confirmation_passes": 10,
        "selection_manifest": EXPECTED_ARTIFACT_FILENAMES["selection"],
        "confirmation_manifest": EXPECTED_ARTIFACT_FILENAMES["confirmation"],
        "official_evaluation_access": "forbidden",
    }
    return preflight_receipt, execution_plan


def _write_stable_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = _canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ExistingPlanError(f"refusing to replace non-identical plan artifact {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_stable_file(*, source: Path, destination: Path, expected_sha256: str) -> None:
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise TemporalPreflightError(f"provenance file changed before copy: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != content:
            raise ExistingPlanError(f"refusing to replace non-identical provenance {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_stable_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ExistingPlanError(f"refusing to replace non-identical artifact {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _staged_source_directory() -> Path:
    root = REPOSITORY_ROOT.resolve(strict=True)
    candidate = root / "src"
    if candidate.is_symlink():
        raise TemporalPreflightError("staged src directory cannot be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise TemporalPreflightError("immutable staged src directory is missing") from error
    if resolved != candidate or not resolved.is_dir():
        raise TemporalPreflightError("staged src directory is not an exact regular directory")
    return resolved


def _activate_staged_source_import() -> Path:
    """Make the validated staged package the only importable BarunLM source."""

    staged_src = _staged_source_directory()
    imported = sorted(
        name for name in sys.modules if name == "barunlm" or name.startswith("barunlm.")
    )
    if imported:
        raise TemporalPreflightError(
            "barunlm was imported before staged-source activation: " + ", ".join(imported[:5])
        )
    conflicts: list[str] = []
    retained: list[str] = []
    for raw_entry in sys.path:
        entry = raw_entry if isinstance(raw_entry, str) else str(raw_entry)
        candidate = Path(entry or os.getcwd()).expanduser()
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            retained.append(entry)
            continue
        if resolved == staged_src:
            continue
        if (resolved / "barunlm").exists() or (resolved / "barunlm.py").exists():
            conflicts.append(str(resolved))
        retained.append(entry)
    if conflicts:
        raise TemporalPreflightError(
            "a conflicting non-staged barunlm import path is present: "
            + ", ".join(sorted(set(conflicts)))
        )
    sys.path[:] = [str(staged_src), *retained]
    importlib.invalidate_caches()
    return staged_src


def _validate_runtime_python_identity() -> dict[str, Any]:
    expected = {
        "python_implementation": EXPECTED_PYTHON_IMPLEMENTATION,
        "python_version": EXPECTED_PYTHON_VERSION,
        "sys_implementation_name": "cpython",
    }
    observed = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "sys_implementation_name": getattr(sys.implementation, "name", None),
    }
    if observed != expected:
        raise TemporalPreflightError(
            "runtime Python identity changed before remote tests: "
            f"observed {observed!r}, expected {expected!r}"
        )
    return {"status": "matched", "expected": expected, "observed": observed}


def _run_pre_cuda_validation(
    *, config: Mapping[str, Any], output_dir: Path, terminal_manifest: Path
) -> dict[str, Any]:
    contract = config.get("remote_validation_contract")
    expected = {
        "command": [
            "python",
            "-m",
            "pytest",
            "-q",
            "tests/test_mobile_temporal_counterfactual.py",
            "tests/test_materialize_mobile_temporal_counterfactual_cli.py",
            "tests/test_mobile_temporal_evaluation.py",
            "tests/test_mobile_temporal_counterfactual_runner.py",
            "tests/test_mobile_temporal_experiment_backend.py",
            "tests/test_training.py",
            "tests/test_jarvis_safe_run.py",
        ],
        "environment_overrides": {"CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled"},
        "provider_runtime": {
            "template": EXPECTED_JARVIS_TEMPLATE,
            "python_implementation": EXPECTED_PYTHON_IMPLEMENTATION,
            "python_version": EXPECTED_PYTHON_VERSION,
            "repository_root_virtual_environment": ".venv",
            "virtual_environment_must_be_active": True,
            "virtual_environment_in_scientific_tree": False,
            "preupload_runtime_attestation_required": True,
        },
        "required_exit_code": 0,
        "parent_process_torch_imported_before_validation": False,
        "real_terminal_manifest_path_or_contents_passed_in_test_argv_or_environment": False,
        "test_file_allowlist_has_no_real_terminal_reference": True,
    }
    if contract != expected:
        raise TemporalPreflightError("frozen remote-validation contract changed")
    if _torch_is_imported():
        raise TemporalPreflightError("torch was imported before CPU-only remote validation")
    runtime_identity = _validate_runtime_python_identity()
    staged_src = _staged_source_directory()
    argv = [sys.executable, *expected["command"][1:]]
    terminal_text = str(terminal_manifest)
    if any(terminal_text in value for value in argv):
        raise TemporalPreflightError("real terminal path entered remote-validation argv")
    environment = os.environ.copy()
    environment.update(expected["environment_overrides"])
    controller_safety_overrides = {
        "PYTHONPATH": str(staged_src),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_ADDOPTS": "-p no:cacheprovider",
    }
    environment.update(controller_safety_overrides)
    timed_out = False
    try:
        completed = subprocess.run(
            argv,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=REMOTE_VALIDATION_TIMEOUT_SECONDS,
        )
        log = completed.stdout
        exit_code: int | None = completed.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        captured = error.stdout or b""
        log = captured.encode() if isinstance(captured, str) else captured
        log += b"\nremote CPU-only validation timed out\n"
        exit_code = None
    log_path = output_dir / "pre-cuda-validation.log"
    _write_stable_bytes(log_path, log)
    receipt = {
        "schema_version": "barun-mobile-temporal-pre-cuda-validation-v1",
        "status": "timed_out" if timed_out else ("passed" if exit_code == 0 else "failed"),
        "declared_argv": expected["command"],
        "executed_argv": ["python", *expected["command"][1:]],
        "environment_overrides": expected["environment_overrides"],
        "controller_safety_overrides": controller_safety_overrides,
        "exit_code": exit_code,
        "required_exit_code": 0,
        "timeout_seconds": REMOTE_VALIDATION_TIMEOUT_SECONDS,
        "parent_process_torch_imported_before_validation": False,
        "runtime_identity": runtime_identity,
        "real_terminal_manifest_referenced": False,
        "log_path": "pre-cuda-validation.log",
        "log_sha256": hashlib.sha256(log).hexdigest(),
        "log_bytes": len(log),
    }
    _write_stable_json(output_dir / "pre-cuda-validation.json", receipt)
    if timed_out:
        raise TemporalPreflightError(
            f"CPU-only pre-CUDA validation exceeded {REMOTE_VALIDATION_TIMEOUT_SECONDS} seconds"
        )
    if exit_code != 0:
        raise TemporalPreflightError(
            f"CPU-only pre-CUDA validation failed with exit code {exit_code}"
        )
    if _torch_is_imported():
        raise TemporalPreflightError("child validation contaminated the parent torch import state")
    return receipt


def _run_plan_impl(
    *,
    config_path: Path,
    materialization_receipt: Path,
    output_dir: Path,
    execute: bool = False,
    device: str | None = None,
    jarvis_resource_id: str | None = None,
    terminal_manifest: Path | None = None,
    terminal_manifest_sha256: str | None = None,
    attempt_preregistration: Path | None = None,
    attempt_preregistration_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate frozen bytes, then optionally enter the fixed CUDA backend."""

    config_path = config_path.resolve()
    materialization_receipt = materialization_receipt.resolve()
    output_dir = output_dir.resolve()
    if execute:
        staged_root = REPOSITORY_ROOT.resolve(strict=True)
        if output_dir == staged_root or staged_root in output_dir.parents:
            raise TemporalPreflightError(
                "--execute output-dir must be outside the immutable staged repository root"
            )
    preflight_receipt, execution_plan = _preflight(
        config_path=config_path,
        materialization_receipt=materialization_receipt,
    )
    terminal_binding = _terminal_binding(
        terminal_manifest=terminal_manifest,
        terminal_manifest_sha256=terminal_manifest_sha256,
    )
    provenance_binding: dict[str, Any] = {"status": "unbound_plan_only"}
    pre_cuda_validation: dict[str, Any] = {"status": "not_run_plan_only"}
    if execute:
        if device != "cuda":
            raise TemporalPreflightError("--execute requires explicit --device cuda")
        if (
            not isinstance(jarvis_resource_id, str)
            or not jarvis_resource_id.isascii()
            or not jarvis_resource_id.isdigit()
            or int(jarvis_resource_id) <= 0
        ):
            raise TemporalPreflightError(
                "--execute requires an explicit positive --jarvis-resource-id"
            )
        if terminal_binding["status"] != "bound_for_conditional_terminal_read":
            raise TemporalPreflightError(
                "--execute requires --terminal-manifest and its pinned SHA-256"
            )
        if attempt_preregistration is None or attempt_preregistration_sha256 is None:
            raise TemporalPreflightError(
                "--execute requires --attempt-preregistration and its independently bound SHA-256"
            )
        frozen_config, _ = _load_json_object(config_path, label="frozen experiment config")
        provenance_binding = _validate_attempt_preregistration(
            attempt_path=attempt_preregistration,
            config_path=config_path,
            config=frozen_config,
            materialization_receipt=materialization_receipt,
            terminal_binding=terminal_binding,
            jarvis_resource_id=jarvis_resource_id,
            expected_attempt_sha256=attempt_preregistration_sha256,
        )
        attempt_binding = provenance_binding["attempt_preregistration"]
        source_binding = provenance_binding["source_snapshot"]
        _copy_stable_file(
            source=attempt_preregistration.resolve(strict=True),
            destination=output_dir / ATTEMPT_PREREGISTRATION_RELATIVE_PATH,
            expected_sha256=attempt_binding["sha256"],
        )
        _copy_stable_file(
            source=REPOSITORY_ROOT / SOURCE_SNAPSHOT_RELATIVE_PATH,
            destination=output_dir / SOURCE_SNAPSHOT_RELATIVE_PATH,
            expected_sha256=source_binding["sha256"],
        )
        pre_cuda_validation = _run_pre_cuda_validation(
            config=frozen_config,
            output_dir=output_dir,
            terminal_manifest=terminal_manifest.resolve(strict=True),
        )
        post_test_provenance = _validate_attempt_preregistration(
            attempt_path=attempt_preregistration,
            config_path=config_path,
            config=frozen_config,
            materialization_receipt=materialization_receipt,
            terminal_binding=terminal_binding,
            jarvis_resource_id=jarvis_resource_id,
            expected_attempt_sha256=attempt_preregistration_sha256,
        )
        if post_test_provenance != provenance_binding:
            raise TemporalPreflightError("launch provenance changed during CPU-only validation")
    elif attempt_preregistration is not None or attempt_preregistration_sha256 is not None:
        raise TemporalPreflightError(
            "--attempt-preregistration bindings are valid only with --execute"
        )
    preflight_receipt = {
        **preflight_receipt,
        "launch_provenance": provenance_binding,
        "pre_cuda_validation": pre_cuda_validation,
        "token_length_audit": FROZEN_TOKEN_LENGTH_AUDIT,
        "terminal_compatibility_veto": {
            **terminal_binding,
            "rows_read": 0,
            "rows_scored": 0,
        },
    }
    execution_plan = {
        **execution_plan,
        "launch_provenance": provenance_binding,
        "pre_cuda_validation": pre_cuda_validation,
        "preflight_receipt_sha256": hashlib.sha256(_canonical_bytes(preflight_receipt)).hexdigest(),
        "terminal_compatibility_veto": terminal_binding,
        "token_length_audit": FROZEN_TOKEN_LENGTH_AUDIT,
    }
    _write_stable_json(output_dir / "preflight-receipt.json", preflight_receipt)
    _write_stable_json(output_dir / "execution-plan.json", execution_plan)
    if execute:
        attempt_binding = provenance_binding["attempt_preregistration"]
        source_binding = provenance_binding["source_snapshot"]
        _copy_stable_file(
            source=attempt_preregistration.resolve(strict=True),
            destination=output_dir / ATTEMPT_PREREGISTRATION_RELATIVE_PATH,
            expected_sha256=attempt_binding["sha256"],
        )
        _copy_stable_file(
            source=REPOSITORY_ROOT / SOURCE_SNAPSHOT_RELATIVE_PATH,
            destination=output_dir / SOURCE_SNAPSHOT_RELATIVE_PATH,
            expected_sha256=source_binding["sha256"],
        )
        if _torch_is_imported():
            raise TemporalPreflightError(
                "torch was imported before deterministic execution setup; start a clean process"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        configured_before_import = (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        )
        if not configured_before_import:  # pragma: no cover - os.environ invariant
            raise TemporalPreflightError("could not set deterministic cuBLAS configuration")
        os.environ["WANDB_MODE"] = "disabled"
        return _execute_frozen_backend(
            config_path=config_path,
            materialization_receipt=materialization_receipt,
            output_dir=output_dir,
            preflight_receipt=preflight_receipt,
            execution_plan=execution_plan,
            jarvis_resource_id=jarvis_resource_id,
            cublas_configured_before_torch_import=configured_before_import,
        )
    return execution_plan


def _fallback_evidence_files(
    *, execution_dir: Path, usable_held_out_artifacts: Sequence[str]
) -> list[Path]:
    """Select bounded evidence without importing the torch-backed bundle builder."""

    allowed_tree_roots = {
        "gates",
        "fits",
        "training-configs",
        "conditional-full-refit",
        "terminal-compatibility-veto",
    }
    allowed_training_names = {
        "data_rejections.jsonl",
        "failure.json",
        "heldout_access_started.json",
        "metrics.jsonl",
        "resolved_config.json",
        "run_manifest.json",
        "summary.json",
    }
    selected: dict[str, Path] = {}

    def add(source: Path) -> None:
        relative = source.relative_to(execution_dir).as_posix()
        if source.is_symlink():
            raise TemporalPreflightError(f"fallback evidence cannot be a symlink: {relative}")
        if not source.is_file():
            return
        size = source.stat().st_size
        if size > 256 * 1024 * 1024:
            raise TemporalPreflightError(f"fallback evidence exceeds 256 MiB: {relative}")
        allowed_promoted_model = "conditional-full-refit/inference-checkpoint/model.safetensors"
        if source.name == "optimizer.pt" or (
            source.suffix in {".bin", ".pt", ".pth", ".safetensors"}
            and relative != allowed_promoted_model
        ):
            return
        selected[relative] = source

    for name in (
        "environment-receipt.json",
        "result.json",
        "failure.json",
        "essential-bundle-failure.json",
    ):
        add(execution_dir / name)
    for root_name in sorted(allowed_tree_roots):
        tree_root = execution_dir / root_name
        if tree_root.is_symlink():
            raise TemporalPreflightError(f"fallback evidence root cannot be a symlink: {root_name}")
        if tree_root.is_dir():
            for path in sorted(tree_root.rglob("*")):
                if path.is_symlink():
                    raise TemporalPreflightError(
                        "fallback evidence tree contains a symlink: "
                        + path.relative_to(execution_dir).as_posix()
                    )
                add(path)
    training_root = execution_dir / "training"
    if training_root.is_symlink():
        raise TemporalPreflightError("fallback training evidence root cannot be a symlink")
    if training_root.is_dir():
        for run_dir in sorted(training_root.iterdir()):
            if run_dir.is_symlink():
                raise TemporalPreflightError("fallback training run cannot be a symlink")
            if not run_dir.is_dir():
                continue
            for name in sorted(allowed_training_names):
                add(run_dir / name)
    for path in sorted(execution_dir.rglob("*")):
        relative_parts = path.relative_to(execution_dir).parts
        if (
            path.is_file()
            and relative_parts
            and not relative_parts[0].startswith("essential")
            and not relative_parts[0].startswith(".essential")
            and (
                path.name.endswith("heldout-access-started.json")
                or path.name == "heldout_access_started.json"
            )
        ):
            add(path)

    normalized_usable: set[str] = set()
    for index, value in enumerate(usable_held_out_artifacts):
        relative = _normalized_relative_path(
            value, label=f"existing failure usable_held_out_artifacts[{index}]"
        )
        source = execution_dir / relative
        if source.is_symlink() or not source.is_file():
            raise TemporalPreflightError(
                f"existing failure names missing held-out evidence: {relative}"
            )
        normalized_usable.add(relative)
    missing = normalized_usable - set(selected)
    if missing:
        raise TemporalPreflightError(
            "fallback whitelist would omit usable held-out evidence: " + ", ".join(sorted(missing))
        )
    total_bytes = sum(path.stat().st_size for path in selected.values())
    if total_bytes > 2 * 1024 * 1024 * 1024:
        raise TemporalPreflightError("fallback compact evidence exceeds 2 GiB")
    return [selected[name] for name in sorted(selected)]


def _write_prebackend_failure_bundle(
    *, output_dir: Path, error: Exception, jarvis_resource_id: str | None
) -> None:
    execution_dir = output_dir / "execution"
    execution_dir.mkdir(parents=True, exist_ok=True)
    final_essential = execution_dir / "essential"
    failure_path = execution_dir / "failure.json"
    attempt_copy = output_dir / ATTEMPT_PREREGISTRATION_RELATIVE_PATH
    snapshot_copy = output_dir / SOURCE_SNAPSHOT_RELATIVE_PATH
    provenance_bound = attempt_copy.is_file() and snapshot_copy.is_file()
    valid_resource_id = (
        jarvis_resource_id
        if isinstance(jarvis_resource_id, str)
        and jarvis_resource_id.isascii()
        and jarvis_resource_id.isdigit()
        and int(jarvis_resource_id) > 0
        else None
    )
    if failure_path.is_file():
        failure, _ = _load_json_object(failure_path, label="existing backend failure receipt")
        observed_failure_machine = failure.get("jarvis_resource_id")
        if isinstance(observed_failure_machine, str) and observed_failure_machine.isdigit():
            observed_failure_machine = int(observed_failure_machine)
        usable_artifacts = failure.get("usable_held_out_artifacts")
        if (
            failure.get("schema_version") != "barun-mobile-temporal-execution-failure-v1"
            or failure.get("artifacts_preserved") is not True
            or observed_failure_machine
            != (int(valid_resource_id) if valid_resource_id is not None else None)
            or failure.get("official_961_rows_read") != 0
            or failure.get("reused_756_rows_read") not in {0, PINNED_REUSED_756_ROWS}
            or failure.get("reused_756_rows_scored") not in {0, PINNED_REUSED_756_ROWS}
            or int(failure.get("reused_756_rows_scored", -1))
            > int(failure.get("reused_756_rows_read", -1))
            or not isinstance(usable_artifacts, list)
            or any(not isinstance(value, str) for value in usable_artifacts)
        ):
            raise TemporalPreflightError("existing backend failure receipt is invalid")
        bundle_status = "backend_failure_fallback"
    else:
        failure = {
            "schema_version": "barun-mobile-temporal-execution-failure-v1",
            "failed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "phase": "stdlib_prebackend",
            "jarvis_resource_id": valid_resource_id,
            "error_type": type(error).__name__,
            "message": str(error),
            "artifacts_preserved": True,
            "automatic_retry_attempted": False,
            "usable_held_out_artifacts": [],
            "retry_after_this_attempt": (
                "external_controller_must_verify_eligibility"
                if provenance_bound and valid_resource_id is not None and not _torch_is_imported()
                else "forbidden_unproven_prebackend_failure"
            ),
            "parent_torch_imported": _torch_is_imported(),
            "model_or_cuda_loaded": False if not _torch_is_imported() else "unproven",
            "official_961_rows_read": 0,
            "reused_756_rows_read": 0,
            "reused_756_rows_scored": 0,
        }
        _write_stable_json(failure_path, failure)
        bundle_status = "prebackend_failure"
    usable_artifacts = failure.get("usable_held_out_artifacts")
    if not isinstance(usable_artifacts, list):
        raise TemporalPreflightError("failure usable-held-out artifact list is invalid")
    durable_execution_files = _fallback_evidence_files(
        execution_dir=execution_dir, usable_held_out_artifacts=usable_artifacts
    )
    completed_manifest = final_essential / "artifact-manifest.json"
    completed_failure = final_essential / "failure.json"
    if completed_manifest.is_file() and completed_failure.is_file():
        if completed_failure.read_bytes() != failure_path.read_bytes():
            raise TemporalPreflightError("completed essential failure receipt differs from backend")
        if any(not (final_essential / relative).is_file() for relative in usable_artifacts):
            raise TemporalPreflightError("completed essential bundle omits held-out evidence")
        return
    build = execution_dir / f".essential-prebackend-{os.getpid()}"
    if build.exists():
        shutil.rmtree(build)
    try:
        controller = build / "controller"
        controller.mkdir(parents=True)
        for name in (
            "preflight-receipt.json",
            "execution-plan.json",
            ATTEMPT_PREREGISTRATION_RELATIVE_PATH,
            SOURCE_SNAPSHOT_RELATIVE_PATH,
            "pre-cuda-validation.json",
            "pre-cuda-validation.log",
        ):
            source = output_dir / name
            if source.is_file():
                _write_stable_bytes(controller / name, source.read_bytes())
        _write_stable_bytes(build / "failure.json", failure_path.read_bytes())
        for source in durable_execution_files:
            relative = source.relative_to(execution_dir)
            _write_stable_bytes(build / relative, source.read_bytes())
        files = [path for path in sorted(build.rglob("*")) if path.is_file()]
        entries = {
            path.relative_to(build).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in files
        }
        _write_stable_json(
            build / "artifact-manifest.json",
            {
                "schema_version": "barun-mobile-temporal-essential-v1",
                "status": bundle_status,
                "file_count": len(entries),
                "total_bytes": sum(item["bytes"] for item in entries.values()),
                "files": entries,
            },
        )
        if final_essential.exists():
            quarantine = execution_dir / "essential-prior-incomplete"
            if quarantine.exists():
                raise ExistingPlanError("prior incomplete essential quarantine already exists")
            final_essential.replace(quarantine)
        build.replace(final_essential)
    finally:
        if build.exists():
            shutil.rmtree(build)


def run_plan(
    *,
    config_path: Path,
    materialization_receipt: Path,
    output_dir: Path,
    execute: bool = False,
    device: str | None = None,
    jarvis_resource_id: str | None = None,
    terminal_manifest: Path | None = None,
    terminal_manifest_sha256: str | None = None,
    attempt_preregistration: Path | None = None,
    attempt_preregistration_sha256: str | None = None,
) -> dict[str, Any]:
    resolved_output = output_dir.resolve()
    if execute:
        staged_root = REPOSITORY_ROOT.resolve(strict=True)
        if resolved_output == staged_root or staged_root in resolved_output.parents:
            raise TemporalPreflightError(
                "--execute output-dir must be outside the immutable staged repository root"
            )
        if resolved_output.exists():
            raise TemporalPreflightError("--execute requires a fresh non-existing output directory")
        resolved_output.mkdir(parents=True)
    try:
        return _run_plan_impl(
            config_path=config_path,
            materialization_receipt=materialization_receipt,
            output_dir=resolved_output,
            execute=execute,
            device=device,
            jarvis_resource_id=jarvis_resource_id,
            terminal_manifest=terminal_manifest,
            terminal_manifest_sha256=terminal_manifest_sha256,
            attempt_preregistration=attempt_preregistration,
            attempt_preregistration_sha256=attempt_preregistration_sha256,
        )
    except Exception as error:
        if execute:
            try:
                _write_prebackend_failure_bundle(
                    output_dir=resolved_output,
                    error=error,
                    jarvis_resource_id=jarvis_resource_id,
                )
            except Exception as bundle_error:  # noqa: BLE001 - preserve the original backend error
                if hasattr(error, "add_note"):
                    error.add_note(
                        "secondary failure while preserving compact evidence: "
                        f"{type(bundle_error).__name__}: {bundle_error}"
                    )
        raise


def _torch_is_imported() -> bool:
    return any(name == "torch" or name.startswith("torch.") for name in sys.modules)


def _terminal_binding(
    *, terminal_manifest: Path | None, terminal_manifest_sha256: str | None
) -> dict[str, Any]:
    """Bind the terminal population without reading any of its bytes during preflight."""

    if terminal_manifest is None and terminal_manifest_sha256 is None:
        return {
            "status": "unbound_plan_only",
            "manifest": None,
            "manifest_sha256": PINNED_REUSED_756_SHA256,
            "rows": PINNED_REUSED_756_ROWS,
            "access": "only_after_confirmation_pass_and_full_refit",
            "max_new_tokens": FROZEN_TERMINAL_MAX_NEW_TOKENS,
        }
    if terminal_manifest is None or terminal_manifest_sha256 is None:
        raise TemporalPreflightError("terminal manifest path and SHA-256 must be supplied together")
    if terminal_manifest_sha256 != PINNED_REUSED_756_SHA256:
        raise TemporalPreflightError("terminal manifest SHA-256 is not the pinned reused-756 hash")
    return {
        "status": "bound_for_conditional_terminal_read",
        "manifest": str(terminal_manifest.expanduser().resolve()),
        "manifest_sha256": PINNED_REUSED_756_SHA256,
        "rows": PINNED_REUSED_756_ROWS,
        "access": "only_after_confirmation_pass_and_full_refit",
        "max_new_tokens": FROZEN_TERMINAL_MAX_NEW_TOKENS,
    }


def _execute_frozen_backend(**kwargs: Any) -> dict[str, Any]:
    """Import torch-backed code only after stdlib preflight and cuBLAS setup."""

    staged_src = _activate_staged_source_import()
    from barunlm.training.mobile_temporal_experiment import execute_temporal_experiment

    backend_module = sys.modules.get("barunlm.training.mobile_temporal_experiment")
    backend_file = getattr(backend_module, "__file__", None)
    if not isinstance(backend_file, str):
        raise TemporalPreflightError("staged backend module has no inspectable source path")
    resolved_backend = Path(backend_file).resolve(strict=True)
    if staged_src not in resolved_backend.parents:
        raise TemporalPreflightError("backend imported from outside the immutable staged src tree")
    return execute_temporal_experiment(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the fixed CUDA backend after the byte-level preflight passes.",
    )
    parser.add_argument("--device", choices=("cuda",))
    parser.add_argument(
        "--jarvis-machine-id",
        "--jarvis-resource-id",
        dest="jarvis_resource_id",
        help="Exact project-created JarvisLabs machine ID (resource-id is a compatibility alias).",
    )
    parser.add_argument("--terminal-manifest", type=Path)
    parser.add_argument("--terminal-manifest-sha256")
    parser.add_argument("--attempt-preregistration", type=Path)
    parser.add_argument("--attempt-preregistration-sha256")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.execute and (args.device != "cuda" or args.jarvis_resource_id is None):
        parser.error("--execute requires --device cuda and --jarvis-resource-id ID")
    if not args.execute and (args.device is not None or args.jarvis_resource_id is not None):
        parser.error("--device and --jarvis-resource-id are valid only with --execute")
    if (args.terminal_manifest is None) != (args.terminal_manifest_sha256 is None):
        parser.error("--terminal-manifest and --terminal-manifest-sha256 are required together")
    if args.execute and (
        args.attempt_preregistration is None or args.attempt_preregistration_sha256 is None
    ):
        parser.error(
            "--execute requires --attempt-preregistration and --attempt-preregistration-sha256"
        )
    if not args.execute and (
        args.attempt_preregistration is not None or args.attempt_preregistration_sha256 is not None
    ):
        parser.error("--attempt-preregistration bindings are valid only with --execute")
    result = run_plan(
        config_path=args.config,
        materialization_receipt=args.materialization_receipt,
        output_dir=args.output_dir,
        execute=args.execute,
        device=args.device,
        jarvis_resource_id=args.jarvis_resource_id,
        terminal_manifest=args.terminal_manifest,
        terminal_manifest_sha256=args.terminal_manifest_sha256,
        attempt_preregistration=args.attempt_preregistration,
        attempt_preregistration_sha256=args.attempt_preregistration_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
