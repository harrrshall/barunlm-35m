"""Build and validate the construction-only PlanIR launch provenance.

This utility has no JarvisLabs or model API.  It binds a deliberately small upload
stage to a clean Git commit, the frozen scientific config, exact file bytes, and an
unbound ``safe_run`` attempt template.  The local controller later substitutes only
the fresh machine ID and the dynamic pre-create protected-ID inventory.

Scientific outputs, logs, predictions, checkpoints, and runtime receipts must be
written outside the uploaded stage.  The only root files excluded from the scientific
tree are ``source-snapshot.json`` and ``attempt-preregistration.json``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

BUILD_SCHEMA_VERSION = "barun-mobile-planir-launch-provenance-build-v1"
VALIDATION_SCHEMA_VERSION = "barun-mobile-planir-launch-provenance-validation-v1"
SOURCE_SNAPSHOT_SCHEMA_VERSION = "barun-mobile-planir-source-snapshot-v1"
ATTEMPT_SCHEMA_VERSION = "barun-mobile-planir-attempt-preregistration-v1"
CONFIG_SCHEMA_VERSION = "barun-mobile-planir-construction-screen-config-v1"
RUN_ID = "20260804-0545-mobile-planir-construction-screen-s17"
EXPECTED_CONFIG_SHA256 = "cca598e6f5a76a5e848a8b9a869cad779bbcbb17999217d2488111b22e2aa34c"

MACHINE_ID_PLACEHOLDER = "__SAFE_RUN_MACHINE_ID__"
PREEXISTING_IDS_PLACEHOLDER = "__SAFE_RUN_PREEXISTING_IDS__"

CONFIG_RELATIVE_PATH = "configs/mobile_planir_construction_screen_v1.json"
REQUIREMENTS_RELATIVE_PATH = "requirements/mobile-planir-screen.txt"
ENTRYPOINT_RELATIVE_PATH = "scripts/run_mobile_planir_screen.py"
BUILDER_RELATIVE_PATH = "scripts/build_mobile_planir_launch_provenance.py"
SAFE_RUN_RELATIVE_PATH = "infra/jarvis/safe_run.py"
SOURCE_SNAPSHOT_RELATIVE_PATH = "source-snapshot.json"
ATTEMPT_RELATIVE_PATH = "attempt-preregistration.json"

MATERIALIZATION_RELATIVE_PATH = (
    "experiments/runs/20260804-0545-mobile-planir-construction-screen-s17/materialized-v1"
)
MATERIALIZED_FILENAMES = {
    "assignment": "assignment.jsonl",
    "audit": "audit.json",
    "exclusion": "exclusion.jsonl",
    "screen_a": "screen-a.jsonl",
    "screen_b": "screen-b.jsonl",
    "screen_c": "screen-c.jsonl",
    "train_a": "train-a.jsonl",
    "train_b": "train-b.jsonl",
    "train_c": "train-c.jsonl",
}
MATERIALIZED_RELATIVE_PATHS = {
    name: f"{MATERIALIZATION_RELATIVE_PATH}/{filename}"
    for name, filename in MATERIALIZED_FILENAMES.items()
}

REMOTE_TEST_PATHS = (
    "tests/test_generation.py",
    "tests/test_grounded_planir.py",
    "tests/test_grounded_planir_v2.py",
    "tests/test_mobile_planir.py",
    "tests/test_mobile_planir_experiment.py",
    "tests/test_mobile_planir_launch_provenance.py",
    "tests/test_mobile_planir_screen.py",
    "tests/test_mobile_planir_screen_evaluation.py",
    "tests/test_mobile_planir_v2.py",
    "tests/test_training.py",
)

TRANSITIVE_SOURCE_PATHS = (
    "src/barunlm/__init__.py",
    "src/barunlm/config.py",
    "src/barunlm/model.py",
    "src/barunlm/quantization.py",
    "src/barunlm/datasets/__init__.py",
    "src/barunlm/datasets/mobile_planir.py",
    "src/barunlm/datasets/mobile_planir_screen.py",
    "src/barunlm/datasets/mobile_planir_v2.py",
    "src/barunlm/evaluation/__init__.py",
    "src/barunlm/evaluation/action_ir.py",
    "src/barunlm/evaluation/evaluator.py",
    "src/barunlm/evaluation/generation.py",
    "src/barunlm/evaluation/grounded_planir.py",
    "src/barunlm/evaluation/grounded_planir_v2.py",
    "src/barunlm/evaluation/mobile_action_schemas.py",
    "src/barunlm/evaluation/mobile_actions.py",
    "src/barunlm/evaluation/mobile_planir_screen.py",
    "src/barunlm/evaluation/presto.py",
    "src/barunlm/training/__init__.py",
    "src/barunlm/training/checkpoint.py",
    "src/barunlm/training/config.py",
    "src/barunlm/training/data.py",
    "src/barunlm/training/mobile_planir_experiment.py",
    "src/barunlm/training/trainer.py",
)

# This is the production upload contract.  Tests may inject a smaller allowlist through
# private function parameters; the CLI intentionally has no option to broaden it.
DEFAULT_ALLOWLIST = tuple(
    sorted(
        {
            "pyproject.toml",
            CONFIG_RELATIVE_PATH,
            REQUIREMENTS_RELATIVE_PATH,
            ENTRYPOINT_RELATIVE_PATH,
            BUILDER_RELATIVE_PATH,
            *TRANSITIVE_SOURCE_PATHS,
            *REMOTE_TEST_PATHS,
            *MATERIALIZED_RELATIVE_PATHS.values(),
        }
    )
)
DEFAULT_COMMIT_BOUND_PATHS = tuple(
    path for path in DEFAULT_ALLOWLIST if path not in MATERIALIZED_RELATIVE_PATHS.values()
)

EXCLUDED_EXACT_PATHS = (ATTEMPT_RELATIVE_PATH, SOURCE_SNAPSHOT_RELATIVE_PATH)
EXCLUDED_DIRECTORY_NAMES = (".git", "__pycache__", ".pytest_cache", ".ruff_cache")
EXCLUDED_ROOT_RUNTIME_DIRECTORIES = (".venv",)
EXCLUDED_FILE_NAMES = (".DS_Store",)
EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo")
FORBIDDEN_DIRECTORY_NAMES = (
    ".env",
    "env",
    "venv",
    "wandb",
)

COMPUTE_CONTRACT = {
    "provider": "JarvisLabs",
    "template": "axolotl",
    "python_implementation": "CPython",
    "python_version": "3.11.10",
    "gpu": "H200",
    "num_gpus": 1,
    "region": "IN2",
    "is_spot": False,
    "max_gpu_job_minutes": 45,
    "storage_gb": 40,
}

EXPECTED_REQUIREMENT_LINES = (
    "huggingface-hub==1.26.0",
    "numpy==2.4.6",
    "pytest==9.1.1",
    "ruff==0.16.1",
    "safetensors==0.8.0",
    "tokenizers==0.23.1",
    "torch==2.13.0",
)
EXPECTED_REQUIREMENTS_SHA256 = "6db8f37c0c21aea4a4ad93d7193b82217e73d3090db6b8947a9a9321aefb21c9"
FORBIDDEN_POPULATION_KEYS = frozenset(
    {
        "human_confirmation_rows_read",
        "human_selection_rows_read",
        "official_mobile_961_rows_read",
        "old_confirmation_rows_read",
        "old_selection_rows_read",
        "reused_mobile_756_rows_read",
    }
)
EXPECTED_SOURCE_PROVENANCE = {
    "adapter_schema_version": "barun-mobile-actions-adapter-v2",
    "construction_manifest_membership_sha256": (
        "b70f6dfe79cfe49792b6e6f5d414fd509c889d18101f75cd8df6d4e4f146c588"
    ),
    "construction_manifest_rows": 5_745,
    "construction_manifest_sha256": (
        "800a3ba0a7f0215e5cf95c77c32e47abed48b2f07f375954b0e0b08296a33d10"
    ),
    "dataset": "google/mobile-actions",
    "dataset_revision": "e920309bc2acbc2e99a5e3201cf37df2b9fd9151",
    "license": "CC BY 4.0",
    "original_train_membership_sha256": (
        "4cdfc3649c21a9f0d3dc4d8d9cffcae3cb5c6abc4414a9fb09636e9220afe96f"
    ),
    "original_train_rows": 7_937,
    "original_train_sha256": "131473ccb5bfb51cac0439b42159e72ec4c598025e50364a52b122b056c2e84e",
    "role": "former_v3_construction_rows_for_internal_mechanism_evidence_only",
    "source_is_fresh_evaluation_evidence": False,
}
EXPECTED_PHASE_ORDER = [
    "complete_all_nine_completion_only_fits",
    "freeze_all_nine_final_step_73_checkpoints",
    "begin_construction_screen_access",
    "generate_all_nine_raw_prediction_populations",
    "freeze_all_nine_raw_prediction_populations",
    "score_once_and_evaluate_the_frozen_gate",
]
EXPECTED_EXECUTION = {
    "automatic_retry": False,
    "backend": "barun-mobile-planir-construction-screen-execution-v1",
    "base_reset_per_fit": True,
    "checkpoint_policy": "unconditional_final_only",
    "cuda_runtime": "13.0",
    "device": "cuda",
    "max_runtime_minutes": 45,
    "phase_order": EXPECTED_PHASE_ORDER,
    "precision": "bf16",
    "provider_template": "axolotl",
    "python_implementation": "CPython",
    "python_version": "3.11.10",
    "region": "IN2",
    "required_gpu": "H200",
    "requirements_sha256": EXPECTED_REQUIREMENTS_SHA256,
    "screen_scoring_after_all_raw_predictions_frozen": True,
    "spot": False,
    "storage_gb": 40,
    "torch_version": "2.13.0",
    "visible_gpus": 1,
    "wandb": "disabled",
}
EXPECTED_RETRY_POLICY = {
    "any_new_attempt": "forbidden",
    "rescue_or_threshold_change_after_result": "forbidden",
    "same_attempt_automatic_retry": "forbidden",
}
FORBIDDEN_POPULATION_PATH_PATTERNS = (
    re.compile(r"(?:^|[-_/])official[-_](?:mobile|961)(?:[-_. /]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_/])reused[-_](?:mobile|756)(?:[-_. /]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_/])human[-_](?:selection|confirmation)(?:[-_. /]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_/])old[-_](?:selection|confirmation)(?:[-_. /]|$)", re.IGNORECASE),
    re.compile(r"(?:^|/)mobile-blind(?:[-_/]|$)", re.IGNORECASE),
    re.compile(r"(?:^|/)(?:selection|confirmation)[-_]labels?\.jsonl$", re.IGNORECASE),
)
SECRET_FILE_NAMES = frozenset(
    {
        ".netrc",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
        "known_hosts",
    }
)
SECRET_FILE_SUFFIXES = frozenset({".key", ".p12", ".pfx", ".pem"})
SECRET_PATTERNS = (
    re.compile(rb"wandb_v1_[A-Za-z0-9_-]{16,}", re.IGNORECASE),
    re.compile(rb"(?:ghp|github_pat)_[A-Za-z0-9_]{16,}", re.IGNORECASE),
    re.compile(rb"hf_[A-Za-z0-9]{24,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{24,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(
        rb"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"]"
        rb"[A-Za-z0-9_./+=-]{20,}"
    ),
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
RUN_ID_RE = re.compile(r"[0-9]{8}-[0-9]{4}-[a-z0-9][a-z0-9-]*-s[0-9]+")
CANONICAL_REPOSITORY = "https://github.com/harrrshall/barunlm-35m.git"
TREE_ALGORITHM = "sha256_of_canonical_sorted_path_sha256_bytes_jsonl_v1"


class LaunchProvenanceError(RuntimeError):
    """The launch stage or its provenance is not safe and unambiguous."""


def _fail(message: str) -> NoReturn:
    raise LaunchProvenanceError(message)


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> NoReturn:
        _fail(f"{label} contains a non-finite JSON value: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except LaunchProvenanceError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchProvenanceError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        _fail(f"{label} must contain one JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise LaunchProvenanceError(f"cannot hash staged file: {path}") from error
    return digest.hexdigest()


def _normalized_relative(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail(f"{label} must be a normalized relative POSIX path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or value in {".", ""}:
        _fail(f"{label} must be a normalized relative POSIX path")
    return value


def _normalized_allowlist(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(_normalized_relative(value, label="allowlist path") for value in values)
    if not normalized or len(normalized) != len(set(normalized)):
        _fail("stage allowlist must be nonempty and contain no duplicates")
    if tuple(sorted(normalized)) != normalized:
        _fail("stage allowlist must be lexicographically sorted")
    if set(normalized) & set(EXCLUDED_EXACT_PATHS):
        _fail("stage allowlist cannot include provenance output files")
    for path in normalized:
        _reject_forbidden_population_path(path)
        _reject_secret_path(path)
    return normalized


def _stage_root(path: Path) -> Path:
    if path.is_symlink():
        _fail("stage root cannot be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise LaunchProvenanceError(f"stage root does not exist: {path}") from error
    if not resolved.is_dir():
        _fail("stage root must be a directory")
    return resolved


def _is_cache_directory(relative: Path) -> bool:
    return any(part in EXCLUDED_DIRECTORY_NAMES[1:] for part in relative.parts)


def _is_ignored_file(relative: Path) -> bool:
    return relative.name in EXCLUDED_FILE_NAMES or relative.suffix in EXCLUDED_FILE_SUFFIXES


def _reject_secret_path(relative: str) -> None:
    path = Path(relative)
    lowered_parts = tuple(part.casefold() for part in path.parts)
    name = path.name.casefold()
    if (
        name == ".env"
        or name.startswith(".env.")
        or name in SECRET_FILE_NAMES
        or path.suffix.casefold() in SECRET_FILE_SUFFIXES
        or "credentials" in lowered_parts
        or "secrets" in lowered_parts
    ):
        _fail(f"credential-like path is forbidden in the launch stage: {relative}")


def _reject_forbidden_population_path(relative: str) -> None:
    if any(pattern.search(relative) is not None for pattern in FORBIDDEN_POPULATION_PATH_PATTERNS):
        _fail(f"forbidden held-out population artifact in launch stage: {relative}")


def _allowed_parent_directories(allowlist: Sequence[str]) -> set[str]:
    parents: set[str] = set()
    for value in allowlist:
        path = Path(value).parent
        while path != Path("."):
            parents.add(path.as_posix())
            path = path.parent
    return parents


def _scientific_stage_files(
    root: Path,
    *,
    allowlist: Sequence[str],
    provenance_outputs_expected: bool,
    allow_runtime: bool,
) -> dict[str, Path]:
    """Return exact scientific files after rejecting every unrecognized stage entry."""

    allowed = set(allowlist)
    allowed_parents = _allowed_parent_directories(allowlist)
    if provenance_outputs_expected:
        allowed_parents.update(Path(value).parent.as_posix() for value in EXCLUDED_EXACT_PATHS)
    scientific: dict[str, Path] = {}
    found_outputs: set[str] = set()

    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current_path / name
            relative = candidate.relative_to(root)
            relative_text = relative.as_posix()
            if candidate.is_symlink():
                _fail(f"symlink is forbidden in launch stage: {relative_text}")
            if name == ".git":
                _fail("the minimal upload stage cannot contain .git")
            if name.casefold() in FORBIDDEN_DIRECTORY_NAMES:
                if relative_text == ".venv" and allow_runtime:
                    continue
                _fail(f"forbidden runtime/credential directory in launch stage: {relative_text}")
            if relative_text == ".venv":
                if not allow_runtime:
                    _fail("provider root .venv must not exist when the snapshot is built")
                continue
            if name == ".venv":
                _fail(f"only the provider root .venv may be ignored: {relative_text}")
            if _is_cache_directory(relative):
                if not allow_runtime:
                    _fail(f"runtime cache directory is forbidden before snapshot: {relative_text}")
                continue
            if relative_text not in allowed_parents:
                _fail(f"unexpected directory outside the stage allowlist: {relative_text}")
            kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in sorted(file_names):
            candidate = current_path / name
            relative = candidate.relative_to(root)
            relative_text = relative.as_posix()
            if candidate.is_symlink():
                _fail(f"symlink is forbidden in launch stage: {relative_text}")
            mode = candidate.stat(follow_symlinks=False).st_mode
            if not stat.S_ISREG(mode):
                _fail(f"non-regular entry is forbidden in launch stage: {relative_text}")
            if _is_cache_directory(relative) or _is_ignored_file(relative):
                if not allow_runtime:
                    _fail(f"runtime cache file is forbidden before snapshot: {relative_text}")
                continue
            _reject_secret_path(relative_text)
            _reject_forbidden_population_path(relative_text)
            if relative_text in EXCLUDED_EXACT_PATHS:
                if not provenance_outputs_expected:
                    _fail(f"provenance output already exists: {relative_text}")
                found_outputs.add(relative_text)
                continue
            if relative_text not in allowed:
                _fail(f"unexpected file outside the stage allowlist: {relative_text}")
            scientific[relative_text] = candidate

    missing = sorted(allowed - set(scientific))
    if missing:
        _fail("stage is missing allowlisted files: " + ", ".join(missing[:10]))
    if provenance_outputs_expected and found_outputs != set(EXCLUDED_EXACT_PATHS):
        missing_outputs = sorted(set(EXCLUDED_EXACT_PATHS) - found_outputs)
        _fail("stage is missing provenance outputs: " + ", ".join(missing_outputs))
    return scientific


def _scan_files_for_secrets(files: Mapping[str, Path]) -> None:
    for relative, path in sorted(files.items()):
        carry = b""
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    sample = carry + block
                    if any(pattern.search(sample) is not None for pattern in SECRET_PATTERNS):
                        _fail(f"credential-like content found in staged file: {relative}")
                    carry = sample[-512:]
        except LaunchProvenanceError:
            raise
        except OSError as error:
            raise LaunchProvenanceError(f"cannot scan staged file: {relative}") from error


def _python_module_path(module: str, allowlist: set[str]) -> str | None:
    base = "src/" + module.replace(".", "/")
    candidates = (f"{base}.py", f"{base}/__init__.py")
    return next((candidate for candidate in candidates if candidate in allowlist), None)


def _source_package(relative: str) -> tuple[str, ...] | None:
    path = Path(relative)
    if len(path.parts) < 3 or path.parts[0] != "src" or path.parts[1] != "barunlm":
        return None
    module_parts = list(path.with_suffix("").parts[1:])
    if module_parts[-1] == "__init__":
        return tuple(module_parts[:-1])
    return tuple(module_parts[:-1])


def _resolved_import_module(*, relative: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    package = _source_package(relative)
    if package is None or node.level > len(package):
        _fail(f"cannot resolve relative import in staged source: {relative}")
    prefix = package[: len(package) - node.level + 1]
    suffix = tuple(node.module.split(".")) if node.module else ()
    return ".".join((*prefix, *suffix))


def _validate_python_import_closure(*, files: Mapping[str, Path], allowlist: Sequence[str]) -> None:
    """Require every statically imported in-project Python module in the exact stage."""

    allowed = set(allowlist)
    for relative, path in sorted(files.items()):
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            raise LaunchProvenanceError(f"cannot parse staged Python source: {relative}") from error
        required_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                required_modules.update(
                    alias.name
                    for alias in node.names
                    if alias.name == "barunlm" or alias.name.startswith("barunlm.")
                )
            elif isinstance(node, ast.ImportFrom):
                module = _resolved_import_module(relative=relative, node=node)
                if module == "barunlm" or (module is not None and module.startswith("barunlm.")):
                    required_modules.add(module)
        for module in sorted(required_modules):
            if _python_module_path(module, allowed) is None:
                _fail(
                    f"stage allowlist omits statically imported module {module!r} from {relative}"
                )


def _file_records(files: Mapping[str, Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": relative,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for relative, path in sorted(files.items())
    ]


def _tree_from_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    normalized_records: list[dict[str, Any]] = []
    prior_path: str | None = None
    for record in records:
        path = record.get("path")
        sha256 = record.get("sha256")
        size = record.get("bytes")
        if (
            not isinstance(path, str)
            or not isinstance(sha256, str)
            or SHA256_RE.fullmatch(sha256) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or (prior_path is not None and path <= prior_path)
        ):
            _fail("scientific tree contains an invalid or unsorted file record")
        normalized = {"path": path, "sha256": sha256, "bytes": size}
        digest.update(
            json.dumps(
                normalized,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        normalized_records.append(normalized)
        total_bytes += size
        prior_path = path
    return {
        "algorithm": TREE_ALGORITHM,
        "sha256": digest.hexdigest(),
        "file_count": len(normalized_records),
        "content_bytes": total_bytes,
        "files": normalized_records,
    }


def _run_git(source_root: Path, arguments: Sequence[str], *, allow_failure: bool = False) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode and not allow_failure:
        diagnostic = completed.stderr.decode("utf-8", "replace").strip()
        _fail(f"Git provenance check failed: {diagnostic or arguments[0]}")
    return completed.stdout if completed.returncode == 0 else b""


def _git_identity(source_repository_root: Path) -> tuple[Path, dict[str, Any]]:
    if source_repository_root.is_symlink():
        _fail("source repository root cannot be a symlink")
    try:
        source_root = source_repository_root.resolve(strict=True)
    except OSError as error:
        raise LaunchProvenanceError("source repository root does not exist") from error
    if not source_root.is_dir():
        _fail("source repository root must be a directory")
    commit = _run_git(source_root, ["rev-parse", "HEAD"]).decode().strip()
    if COMMIT_RE.fullmatch(commit) is None:
        _fail("Git HEAD is not one exact 40-character commit")
    status = _run_git(
        source_root,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )
    if status:
        _fail("source repository must be a completely clean checkout")
    origin = _run_git(source_root, ["remote", "get-url", "origin"]).decode().strip()
    if origin.rstrip("/") not in {
        CANONICAL_REPOSITORY,
        CANONICAL_REPOSITORY.removesuffix(".git"),
    }:
        _fail("Git origin is not the canonical BarunLM repository")
    branch_value = _run_git(source_root, ["rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    branch = None if branch_value == "HEAD" else branch_value
    return source_root, {
        "repository": CANONICAL_REPOSITORY,
        "commit": commit,
        "branch": branch,
        "clean_checkout": True,
    }


def _verify_commit_file(
    *, source_root: Path, commit: str, relative: str, staged_path: Path | None = None
) -> dict[str, Any]:
    relative = _normalized_relative(relative, label="commit-bound path")
    tree_line = _run_git(source_root, ["ls-tree", commit, "--", relative]).decode().strip()
    if not tree_line:
        _fail(f"commit-bound path is absent from the claimed Git commit: {relative}")
    try:
        mode, object_type, _rest = tree_line.split(maxsplit=2)
    except ValueError as error:
        raise LaunchProvenanceError(f"cannot parse Git tree entry: {relative}") from error
    if mode not in {"100644", "100755"} or object_type != "blob":
        _fail(f"commit-bound path is not a regular Git blob: {relative}")
    content = _run_git(source_root, ["show", f"{commit}:{relative}"])
    receipt = {
        "path": relative,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "git_mode": mode,
    }
    if staged_path is not None and (
        staged_path.stat().st_size != len(content) or _sha256_file(staged_path) != receipt["sha256"]
    ):
        _fail(f"staged source differs from the claimed Git commit: {relative}")
    return receipt


def _safe_run_identity(source_root: Path, commit: str) -> dict[str, Any]:
    path = source_root / SAFE_RUN_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        _fail("the local safe_run controller is missing or is a symlink")
    receipt = _verify_commit_file(
        source_root=source_root,
        commit=commit,
        relative=SAFE_RUN_RELATIVE_PATH,
        staged_path=path,
    )
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise LaunchProvenanceError("cannot inspect local safe_run controller constants") from error
    literals: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "MACHINE_ID_PLACEHOLDER",
                    "PREEXISTING_IDS_PLACEHOLDER",
                }:
                    try:
                        literals[target.id] = ast.literal_eval(node.value)
                    except (TypeError, ValueError):
                        _fail("safe_run inventory placeholders must be string literals")
    if literals != {
        "MACHINE_ID_PLACEHOLDER": MACHINE_ID_PLACEHOLDER,
        "PREEXISTING_IDS_PLACEHOLDER": PREEXISTING_IDS_PLACEHOLDER,
    }:
        _fail("launch builder and safe_run inventory placeholders differ")
    return {
        **receipt,
        "role": "local_exact_id_lifecycle_controller",
        "uploaded_to_stage": False,
    }


def _validate_requirements(path: Path) -> None:
    if _sha256_file(path) != EXPECTED_REQUIREMENTS_SHA256:
        _fail("frozen requirements bytes differ from the exact SHA-256 binding")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise LaunchProvenanceError("frozen requirements are not readable UTF-8") from error
    active = tuple(
        line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")
    )
    if active != EXPECTED_REQUIREMENT_LINES:
        _fail("frozen requirements must contain only the exact dependency pins")
    forbidden_prefixes = (
        "-e",
        "--editable",
        "--find-links",
        "--index-url",
        "--extra-index-url",
        "file:",
        "git+",
        "http:",
        "https:",
        ".",
        "/",
    )
    if any(line.casefold().startswith(forbidden_prefixes) for line in active):
        _fail("requirements cannot install local, editable, index, or URL targets")


def _require_exact_mapping(
    payload: Mapping[str, Any], key: str, expected: Mapping[str, Any], *, label: str
) -> None:
    if payload.get(key) != dict(expected):
        _fail(f"{label}.{key} differs from the frozen contract")


def _validate_config(config: Mapping[str, Any]) -> tuple[str, dict[str, str], list[int]]:
    expected_top_level = {
        "arms",
        "base_checkpoint",
        "claims",
        "data",
        "decision",
        "decoding",
        "execution",
        "gate",
        "hypothesis",
        "optimization",
        "resource_policy",
        "retry_policy",
        "run_id",
        "schema_version",
        "seeds",
        "token_compute_audit",
        "use",
    }
    if set(config) != expected_top_level:
        _fail("scientific config top-level fields changed")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        _fail("scientific config schema version changed")
    run_id = config.get("run_id")
    if run_id != RUN_ID or not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        _fail("scientific config run_id changed")
    if config.get("use") != "construction-internal-mechanism-screen-only":
        _fail("scientific config is not construction-only")
    if config.get("seeds") != [17, 29, 43]:
        _fail("scientific config seed set changed")

    data = config.get("data")
    if not isinstance(data, Mapping):
        _fail("scientific config lacks a data contract")
    if set(data) != {
        "artifact_sha256",
        "forbidden_populations",
        "materialization_relative_path",
        "screen_label_access",
        "screen_rows_per_arm",
        "source_provenance",
        "split_version",
        "train_rows_per_arm",
        "training_labels",
    }:
        _fail("scientific config data fields changed")
    if data.get("materialization_relative_path") != MATERIALIZATION_RELATIVE_PATH:
        _fail("scientific config materialization path changed")
    hashes = data.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(MATERIALIZED_FILENAMES):
        _fail("scientific config materialized artifact bindings are incomplete")
    normalized_hashes: dict[str, str] = {}
    for name in sorted(MATERIALIZED_FILENAMES):
        digest = hashes.get(name)
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            _fail(f"scientific config has an invalid artifact hash: {name}")
        normalized_hashes[name] = digest
    forbidden = data.get("forbidden_populations")
    if (
        not isinstance(forbidden, Mapping)
        or set(forbidden) != FORBIDDEN_POPULATION_KEYS
        or any(type(value) is not int or value != 0 for value in forbidden.values())
    ):
        _fail("scientific config must bind zero access to every forbidden population")
    if data.get("screen_label_access") != "only_after_all_nine_final_checkpoints_are_frozen":
        _fail("screen-label access boundary changed")
    if data.get("training_labels") != "arm_specific_construction_train_manifests_only":
        _fail("training-label population boundary changed")
    if (
        data.get("train_rows_per_arm") != 4_596
        or data.get("screen_rows_per_arm") != 1_148
        or data.get("split_version") != "barun-mobile-construction-component-screen-split-v1"
        or data.get("source_provenance") != EXPECTED_SOURCE_PROVENANCE
    ):
        _fail("construction data counts, split, or source provenance changed")

    execution = config.get("execution")
    if not isinstance(execution, Mapping) or execution != EXPECTED_EXECUTION:
        _fail("scientific config execution contract changed")

    retry = config.get("retry_policy")
    if not isinstance(retry, Mapping) or retry != EXPECTED_RETRY_POLICY:
        _fail("scientific config retry policy changed")
    resource = config.get("resource_policy")
    if not isinstance(resource, Mapping):
        _fail("scientific config lacks a resource policy")
    durable = resource.get("protected_resource_ids")
    if (
        not isinstance(durable, list)
        or not durable
        or any(type(value) is not int or value < 1 for value in durable)
        or durable != sorted(set(durable))
    ):
        _fail("scientific config protected resource IDs are invalid")
    if resource != {
        "fresh_instance_required": True,
        "protected_resource_ids": durable,
        "provider": "JarvisLabs",
    }:
        _fail("scientific config resource policy changed")

    claims = config.get("claims")
    if not isinstance(claims, Mapping):
        _fail("scientific config lacks claim limits")
    expected_claims = {
        "fresh_evidence": False,
        "human_collection_unlocked": False,
        "larger_model_comparison": "forbidden",
        "maximum_positive_claim": "licenses_generalized_planir_development_only",
        "model_promotion": "forbidden",
        "release_result": False,
    }
    if claims != expected_claims:
        _fail("scientific config claim limits changed")
    return run_id, normalized_hashes, list(durable)


def _validate_materialized_artifacts(
    *, root: Path, config_hashes: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for name, relative in sorted(MATERIALIZED_RELATIVE_PATHS.items()):
        path = root / relative
        digest = _sha256_file(path)
        if digest != config_hashes[name]:
            _fail(f"materialized artifact differs from the scientific config: {name}")
        receipts[name] = {
            "path": relative,
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    audit = _strict_json_object(root / MATERIALIZED_RELATIVE_PATHS["audit"], label="screen audit")
    if (
        type(audit.get("forbidden_population_rows_read")) is not int
        or audit.get("forbidden_population_rows_read") != 0
    ):
        _fail("materialization audit does not prove zero forbidden-population access")
    outputs = audit.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(MATERIALIZED_FILENAMES) - {"audit"}:
        _fail("materialization audit output inventory is incomplete")
    for name, output in outputs.items():
        if not isinstance(output, Mapping):
            _fail(f"materialization audit output is invalid: {name}")
        if (
            output.get("filename") != MATERIALIZED_FILENAMES[name]
            or output.get("sha256") != config_hashes[name]
        ):
            _fail(f"materialization audit output binding changed: {name}")
    return receipts


def _create_new_json(path: Path, payload: Mapping[str, Any]) -> str:
    content = _canonical_bytes(payload)
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite provenance file: {path.name}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise LaunchProvenanceError(
                f"refusing to overwrite provenance file: {path.name}"
            ) from error
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(content).hexdigest()


def _snapshot_payload(
    *,
    run_id: str,
    git: Mapping[str, Any],
    config_sha256: str,
    tree: Mapping[str, Any],
    allowlist: Sequence[str],
    commit_bound_paths: Sequence[str],
    materialized: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "run_id": run_id,
        "git": dict(git),
        "scientific_config": {
            "path": CONFIG_RELATIVE_PATH,
            "sha256": config_sha256,
        },
        "scientific_tree": {
            **tree,
            "allowlist": list(allowlist),
            "commit_bound_paths": list(commit_bound_paths),
            "excluded_exact_paths": list(EXCLUDED_EXACT_PATHS),
            "excluded_directory_names": list(EXCLUDED_DIRECTORY_NAMES),
            "excluded_root_runtime_directories": list(EXCLUDED_ROOT_RUNTIME_DIRECTORIES),
            "excluded_file_names": list(EXCLUDED_FILE_NAMES),
            "excluded_file_suffixes": list(EXCLUDED_FILE_SUFFIXES),
        },
        "materialized_artifacts": dict(materialized),
        "staging_policy": {
            "minimal_exact_allowlist": True,
            "credentials_present": False,
            "symlinks_present": False,
            "git_metadata_uploaded": False,
            "wandb_uploaded": False,
            "forbidden_heldout_population_artifacts_present": False,
            "scientific_outputs_inside_stage": False,
            "runtime_outputs_root": f"/home/barun-artifacts/{run_id}",
        },
    }


def _attempt_payload(
    *,
    run_id: str,
    config_sha256: str,
    snapshot_sha256: str,
    safe_run: Mapping[str, Any],
    durable_protected_ids: Sequence[int],
) -> dict[str, Any]:
    attempt_id = f"{run_id}-attempt-1"
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "status": (
            "scientific_fields_frozen_before_instance_creation_inventory_bound_by_safe_run_"
            "after_fresh_creation_before_upload_model_cuda_or_screen_access"
        ),
        "run_id": run_id,
        "attempt_id": attempt_id,
        "attempt_ordinal": 1,
        "scientific_config": {"path": CONFIG_RELATIVE_PATH, "sha256": config_sha256},
        "source_snapshot": {
            "path": SOURCE_SNAPSHOT_RELATIVE_PATH,
            "sha256": snapshot_sha256,
        },
        "local_controller": dict(safe_run),
        "compute": dict(COMPUTE_CONTRACT),
        "durable_protected_machine_ids": list(durable_protected_ids),
        "prelaunch_inventory": {
            "captured_before_project_instance_creation": True,
            "protected_machine_ids": PREEXISTING_IDS_PLACEHOLDER,
            "project_machine_id": MACHINE_ID_PLACEHOLDER,
            "fresh_project_instance": True,
        },
        "inventory_semantics": {
            "binding_authority": "hash_bound_local_safe_run_controller",
            "protected_ids": "dynamic_precreate_live_inventory_union_durable_external_denylist",
            "config_protected_ids_are_a_required_floor_not_a_live_inventory_substitute": True,
            "binding_must_finish_before_upload_model_cuda_or_screen_access": True,
        },
        "retry_lock": {
            "same_attempt_retry": "forbidden",
            "automatic_retry": "forbidden",
            "resume_or_reuse_instance": "forbidden",
            "every_new_attempt": "forbidden",
            "retry_authorized": False,
        },
        "population_firewall": {
            "construction_train_manifests": "training_only",
            "construction_screen_manifests": "access_only_after_all_nine_checkpoints_frozen",
            "human_selection_rows_read": 0,
            "human_confirmation_rows_read": 0,
            "old_selection_rows_read": 0,
            "old_confirmation_rows_read": 0,
            "reused_mobile_756_rows_read": 0,
            "official_mobile_961_rows_read": 0,
        },
    }


def build_launch_provenance(
    *,
    stage_root: Path,
    source_repository_root: Path,
    attempt_ordinal: int = 1,
    prior_zero_signal_receipts: Sequence[Path] = (),
    _allowlist: Sequence[str] | None = None,
    _commit_bound_paths: Sequence[str] | None = None,
    _expected_config_sha256: str = EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    """Create one no-overwrite snapshot and unbound fresh-instance attempt template."""

    if type(attempt_ordinal) is not int or attempt_ordinal != 1:
        _fail("only preregistered attempt 1 is permitted; every new attempt is forbidden")
    if prior_zero_signal_receipts:
        _fail("attempt 1 cannot cite prior receipts; every new attempt is forbidden")
    root = _stage_root(stage_root)
    allowlist = _normalized_allowlist(
        tuple(_allowlist) if _allowlist is not None else DEFAULT_ALLOWLIST
    )
    commit_bound = _normalized_allowlist(
        tuple(_commit_bound_paths)
        if _commit_bound_paths is not None
        else DEFAULT_COMMIT_BOUND_PATHS
    )
    if not set(commit_bound) <= set(allowlist):
        _fail("commit-bound paths must be a subset of the stage allowlist")
    if CONFIG_RELATIVE_PATH not in allowlist or REQUIREMENTS_RELATIVE_PATH not in allowlist:
        _fail("stage allowlist omits the config or frozen requirements")
    if set(MATERIALIZED_RELATIVE_PATHS.values()) - set(allowlist):
        _fail("stage allowlist omits materialized construction artifacts")
    files = _scientific_stage_files(
        root,
        allowlist=allowlist,
        provenance_outputs_expected=False,
        allow_runtime=False,
    )
    _scan_files_for_secrets(files)
    _validate_python_import_closure(files=files, allowlist=allowlist)

    config_path = files[CONFIG_RELATIVE_PATH]
    config_sha256 = _sha256_file(config_path)
    if (
        SHA256_RE.fullmatch(_expected_config_sha256) is None
        or config_sha256 != _expected_config_sha256
    ):
        _fail("scientific config bytes differ from the exact preregistered SHA-256")
    config = _strict_json_object(config_path, label="scientific config")
    run_id, config_hashes, durable_protected_ids = _validate_config(config)
    _validate_requirements(files[REQUIREMENTS_RELATIVE_PATH])
    materialized = _validate_materialized_artifacts(root=root, config_hashes=config_hashes)

    source_root, git = _git_identity(source_repository_root)
    for relative in commit_bound:
        _verify_commit_file(
            source_root=source_root,
            commit=git["commit"],
            relative=relative,
            staged_path=files[relative],
        )
    safe_run = _safe_run_identity(source_root, git["commit"])
    if SAFE_RUN_RELATIVE_PATH in allowlist:
        _fail("the local safe_run controller must not be uploaded")

    tree = _tree_from_records(_file_records(files))
    snapshot = _snapshot_payload(
        run_id=run_id,
        git=git,
        config_sha256=config_sha256,
        tree=tree,
        allowlist=allowlist,
        commit_bound_paths=commit_bound,
        materialized=materialized,
    )
    snapshot_path = root / SOURCE_SNAPSHOT_RELATIVE_PATH
    attempt_path = root / ATTEMPT_RELATIVE_PATH
    if (
        snapshot_path.exists()
        or snapshot_path.is_symlink()
        or attempt_path.exists()
        or attempt_path.is_symlink()
    ):
        _fail("refusing to overwrite an existing snapshot or attempt preregistration")
    snapshot_sha256 = _create_new_json(snapshot_path, snapshot)
    attempt = _attempt_payload(
        run_id=run_id,
        config_sha256=config_sha256,
        snapshot_sha256=snapshot_sha256,
        safe_run=safe_run,
        durable_protected_ids=durable_protected_ids,
    )
    try:
        attempt_sha256 = _create_new_json(attempt_path, attempt)
    except Exception:
        if snapshot_path.is_file() and _sha256_file(snapshot_path) == snapshot_sha256:
            snapshot_path.unlink()
        raise

    validation = validate_launch_provenance(
        stage_root=root,
        expected_source_snapshot_sha256=snapshot_sha256,
        expected_attempt_sha256=attempt_sha256,
        require_bound_inventory=False,
        allow_runtime=False,
        _allowlist=allowlist,
        _commit_bound_paths=commit_bound,
        _expected_config_sha256=_expected_config_sha256,
    )
    return {
        "schema_version": BUILD_SCHEMA_VERSION,
        "run_id": run_id,
        "attempt_id": attempt["attempt_id"],
        "attempt_ordinal": 1,
        "source_snapshot": {
            "path": SOURCE_SNAPSHOT_RELATIVE_PATH,
            "sha256": snapshot_sha256,
        },
        "attempt_preregistration": {
            "path": ATTEMPT_RELATIVE_PATH,
            "unbound_sha256": attempt_sha256,
        },
        "scientific_tree": {
            key: validation["scientific_tree"][key]
            for key in ("sha256", "file_count", "content_bytes")
        },
        "inventory_binding": "pending_safe_run_after_fresh_instance_creation_before_upload",
        "local_controller": safe_run,
    }


def _validate_snapshot(
    *,
    root: Path,
    snapshot: Mapping[str, Any],
    snapshot_sha256: str,
    expected_snapshot_sha256: str,
    tree: Mapping[str, Any],
    allowlist: Sequence[str],
    commit_bound: Sequence[str],
    config_sha256: str,
    materialized: Mapping[str, Mapping[str, Any]],
) -> None:
    if snapshot_sha256 != expected_snapshot_sha256:
        _fail("source snapshot SHA-256 differs from the externally bound value")
    if snapshot.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA_VERSION:
        _fail("source snapshot schema changed")
    if snapshot.get("run_id") != RUN_ID:
        _fail("source snapshot run_id changed")
    git = snapshot.get("git")
    if (
        not isinstance(git, Mapping)
        or git.get("repository") != CANONICAL_REPOSITORY
        or not isinstance(git.get("commit"), str)
        or COMMIT_RE.fullmatch(git["commit"]) is None
        or git.get("clean_checkout") is not True
        or (git.get("branch") is not None and not isinstance(git.get("branch"), str))
    ):
        _fail("source snapshot Git identity is invalid")
    if snapshot.get("scientific_config") != {
        "path": CONFIG_RELATIVE_PATH,
        "sha256": config_sha256,
    }:
        _fail("source snapshot scientific config binding changed")
    expected_tree = {
        **tree,
        "allowlist": list(allowlist),
        "commit_bound_paths": list(commit_bound),
        "excluded_exact_paths": list(EXCLUDED_EXACT_PATHS),
        "excluded_directory_names": list(EXCLUDED_DIRECTORY_NAMES),
        "excluded_root_runtime_directories": list(EXCLUDED_ROOT_RUNTIME_DIRECTORIES),
        "excluded_file_names": list(EXCLUDED_FILE_NAMES),
        "excluded_file_suffixes": list(EXCLUDED_FILE_SUFFIXES),
    }
    if snapshot.get("scientific_tree") != expected_tree:
        _fail("source snapshot scientific tree differs from staged bytes or frozen policy")
    if snapshot.get("materialized_artifacts") != dict(materialized):
        _fail("source snapshot materialized artifact bindings changed")
    expected_policy = _snapshot_payload(
        run_id=RUN_ID,
        git=git,
        config_sha256=config_sha256,
        tree=tree,
        allowlist=allowlist,
        commit_bound_paths=commit_bound,
        materialized=materialized,
    )["staging_policy"]
    if snapshot.get("staging_policy") != expected_policy:
        _fail("source snapshot staging policy changed")
    snapshot_path = root / SOURCE_SNAPSHOT_RELATIVE_PATH
    if snapshot_path.read_bytes() != _canonical_bytes(snapshot):
        _fail("source snapshot is not exact canonical JSON")


def _validate_attempt(
    *,
    root: Path,
    attempt: Mapping[str, Any],
    attempt_sha256: str,
    expected_attempt_sha256: str,
    snapshot_sha256: str,
    config_sha256: str,
    durable_protected_ids: Sequence[int],
    require_bound_inventory: bool,
) -> None:
    if attempt_sha256 != expected_attempt_sha256:
        _fail("attempt preregistration SHA-256 differs from the externally bound value")
    if attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION or attempt.get("run_id") != RUN_ID:
        _fail("attempt preregistration identity changed")
    ordinal = attempt.get("attempt_ordinal")
    if type(ordinal) is not int or ordinal != 1:
        _fail("attempt preregistration must be exact attempt 1")
    if attempt.get("attempt_id") != f"{RUN_ID}-attempt-1":
        _fail("attempt preregistration must use the frozen attempt-1 ID")
    if attempt.get("scientific_config") != {
        "path": CONFIG_RELATIVE_PATH,
        "sha256": config_sha256,
    } or attempt.get("source_snapshot") != {
        "path": SOURCE_SNAPSHOT_RELATIVE_PATH,
        "sha256": snapshot_sha256,
    }:
        _fail("attempt preregistration source/config binding changed")
    if attempt.get("compute") != COMPUTE_CONTRACT:
        _fail("attempt preregistration compute contract changed")
    if attempt.get("durable_protected_machine_ids") != list(durable_protected_ids):
        _fail("attempt preregistration durable protected IDs changed")

    local_controller = attempt.get("local_controller")
    if (
        not isinstance(local_controller, Mapping)
        or local_controller.get("path") != SAFE_RUN_RELATIVE_PATH
        or not isinstance(local_controller.get("sha256"), str)
        or SHA256_RE.fullmatch(local_controller["sha256"]) is None
        or type(local_controller.get("bytes")) is not int
        or local_controller.get("bytes", -1) < 1
        or local_controller.get("role") != "local_exact_id_lifecycle_controller"
        or local_controller.get("uploaded_to_stage") is not False
    ):
        _fail("attempt preregistration local controller identity is invalid")
    inventory = attempt.get("prelaunch_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != {
        "captured_before_project_instance_creation",
        "protected_machine_ids",
        "project_machine_id",
        "fresh_project_instance",
    }:
        _fail("attempt preregistration inventory block is invalid")
    if (
        inventory["captured_before_project_instance_creation"] is not True
        or inventory["fresh_project_instance"] is not True
    ):
        _fail("attempt preregistration does not require pre-create inventory and a fresh instance")
    protected = inventory["protected_machine_ids"]
    machine = inventory["project_machine_id"]
    if require_bound_inventory:
        if (
            type(machine) is not int
            or machine < 1
            or not isinstance(protected, list)
            or any(type(value) is not int or value < 1 for value in protected)
            or protected != sorted(set(protected))
            or machine in protected
            or not set(durable_protected_ids) <= set(protected)
        ):
            _fail("bound safe_run inventory is invalid or omits protected resources")
    elif protected != PREEXISTING_IDS_PLACEHOLDER or machine != MACHINE_ID_PLACEHOLDER:
        _fail("attempt preregistration must remain unbound before fresh instance creation")

    expected_semantics = _attempt_payload(
        run_id=RUN_ID,
        config_sha256=config_sha256,
        snapshot_sha256=snapshot_sha256,
        safe_run=local_controller,
        durable_protected_ids=durable_protected_ids,
    )
    if set(attempt) != set(expected_semantics):
        _fail("attempt preregistration fields changed")
    for field in (
        "status",
        "inventory_semantics",
        "retry_lock",
        "population_firewall",
    ):
        if attempt.get(field) != expected_semantics[field]:
            _fail(f"attempt preregistration {field} changed")
    attempt_path = root / ATTEMPT_RELATIVE_PATH
    if attempt_path.read_bytes() != _canonical_bytes(attempt):
        _fail("attempt preregistration is not exact canonical JSON")


def validate_launch_provenance(
    *,
    stage_root: Path,
    expected_source_snapshot_sha256: str,
    expected_attempt_sha256: str,
    require_bound_inventory: bool = True,
    allow_runtime: bool = True,
    _allowlist: Sequence[str] | None = None,
    _commit_bound_paths: Sequence[str] | None = None,
    _expected_config_sha256: str = EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    """Validate an unmodified scientific tree before model, CUDA, or screen access."""

    if (
        SHA256_RE.fullmatch(expected_source_snapshot_sha256) is None
        or SHA256_RE.fullmatch(expected_attempt_sha256) is None
    ):
        _fail("validation requires exact lowercase SHA-256 bindings")
    root = _stage_root(stage_root)
    allowlist = _normalized_allowlist(
        tuple(_allowlist) if _allowlist is not None else DEFAULT_ALLOWLIST
    )
    commit_bound = _normalized_allowlist(
        tuple(_commit_bound_paths)
        if _commit_bound_paths is not None
        else DEFAULT_COMMIT_BOUND_PATHS
    )
    if not set(commit_bound) <= set(allowlist):
        _fail("commit-bound paths must be a subset of the stage allowlist")
    files = _scientific_stage_files(
        root,
        allowlist=allowlist,
        provenance_outputs_expected=True,
        allow_runtime=allow_runtime,
    )
    _scan_files_for_secrets(
        {
            **files,
            SOURCE_SNAPSHOT_RELATIVE_PATH: root / SOURCE_SNAPSHOT_RELATIVE_PATH,
            ATTEMPT_RELATIVE_PATH: root / ATTEMPT_RELATIVE_PATH,
        }
    )
    _validate_python_import_closure(files=files, allowlist=allowlist)
    config_sha256 = _sha256_file(files[CONFIG_RELATIVE_PATH])
    if (
        SHA256_RE.fullmatch(_expected_config_sha256) is None
        or config_sha256 != _expected_config_sha256
    ):
        _fail("scientific config bytes differ from the exact preregistered SHA-256")
    config = _strict_json_object(files[CONFIG_RELATIVE_PATH], label="scientific config")
    run_id, config_hashes, durable_protected_ids = _validate_config(config)
    _validate_requirements(files[REQUIREMENTS_RELATIVE_PATH])
    materialized = _validate_materialized_artifacts(root=root, config_hashes=config_hashes)
    tree = _tree_from_records(_file_records(files))

    snapshot_path = root / SOURCE_SNAPSHOT_RELATIVE_PATH
    attempt_path = root / ATTEMPT_RELATIVE_PATH
    snapshot = _strict_json_object(snapshot_path, label="source snapshot")
    attempt = _strict_json_object(attempt_path, label="attempt preregistration")
    snapshot_sha256 = _sha256_file(snapshot_path)
    attempt_sha256 = _sha256_file(attempt_path)
    _validate_snapshot(
        root=root,
        snapshot=snapshot,
        snapshot_sha256=snapshot_sha256,
        expected_snapshot_sha256=expected_source_snapshot_sha256,
        tree=tree,
        allowlist=allowlist,
        commit_bound=commit_bound,
        config_sha256=config_sha256,
        materialized=materialized,
    )
    _validate_attempt(
        root=root,
        attempt=attempt,
        attempt_sha256=attempt_sha256,
        expected_attempt_sha256=expected_attempt_sha256,
        snapshot_sha256=snapshot_sha256,
        config_sha256=config_sha256,
        durable_protected_ids=durable_protected_ids,
        require_bound_inventory=require_bound_inventory,
    )
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "run_id": run_id,
        "attempt_id": attempt["attempt_id"],
        "source_snapshot_sha256": snapshot_sha256,
        "attempt_preregistration_sha256": attempt_sha256,
        "inventory_bound": require_bound_inventory,
        "runtime_exclusions_permitted": allow_runtime,
        "scientific_tree": {key: tree[key] for key in ("sha256", "file_count", "content_bytes")},
        "forbidden_population_rows_read": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="create a source snapshot and unbound attempt")
    build.add_argument("--stage-root", type=Path, required=True)
    build.add_argument("--source-repository-root", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="validate a safe_run-bound stage")
    validate.add_argument("--stage-root", type=Path, required=True)
    validate.add_argument("--source-snapshot-sha256", required=True)
    validate.add_argument("--attempt-preregistration-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        result = build_launch_provenance(
            stage_root=args.stage_root,
            source_repository_root=args.source_repository_root,
        )
    else:
        result = validate_launch_provenance(
            stage_root=args.stage_root,
            expected_source_snapshot_sha256=args.source_snapshot_sha256,
            expected_attempt_sha256=args.attempt_preregistration_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LaunchProvenanceError as error:
        print(f"launch provenance rejected: {error}", file=sys.stderr)
        raise SystemExit(2) from None
