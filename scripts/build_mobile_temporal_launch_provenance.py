"""Build the immutable source snapshot and fresh-attempt template for MBCF v2.

The staged directory must already contain the committed source, the two materialized
view audits and their JSONL payloads, the dependency-only requirements file, and the
reused 756-row terminal manifest.  This utility writes only the two files that the
remote runner deliberately excludes from its scientific content-tree hash:
``source-snapshot.json`` and ``attempt-preregistration.json``.

The JarvisLabs controller replaces the two inventory placeholders after creating one
fresh instance and before uploading this directory.  No credential or Jarvis API call
is made here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

MACHINE_ID_PLACEHOLDER = "__SAFE_RUN_MACHINE_ID__"
PREEXISTING_IDS_PLACEHOLDER = "__SAFE_RUN_PREEXISTING_IDS__"
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_RUN_ID_RE = re.compile(r"[0-9]{8}-[0-9]{4}-[a-z0-9][a-z0-9-]*-s[0-9]+")
_CANONICAL_REPOSITORY = "https://github.com/harrrshall/barunlm-35m.git"
_REQUIREMENTS_RELATIVE_PATH = "requirements/mobile-temporal-counterfactual.txt"
_REQUIRED_REQUIREMENT_LINES = (
    "huggingface-hub==1.26.0",
    "numpy==2.4.6",
    "pytest==9.1.1",
    "ruff==0.16.1",
    "safetensors==0.8.0",
    "tokenizers==0.23.1",
    "torch==2.13.0",
)
_REQUIRED_IMPLEMENTATION_PATHS = frozenset(
    {
        "infra/jarvis/safe_run.py",
        _REQUIREMENTS_RELATIVE_PATH,
        "scripts/build_mobile_temporal_launch_provenance.py",
        "scripts/materialize_mobile_temporal_counterfactual.py",
        "scripts/run_mobile_temporal_counterfactual.py",
        "src/barunlm/config.py",
        "src/barunlm/datasets/mobile_temporal_counterfactual.py",
        "src/barunlm/evaluation/action_ir.py",
        "src/barunlm/evaluation/evaluator.py",
        "src/barunlm/evaluation/generation.py",
        "src/barunlm/evaluation/mobile_actions.py",
        "src/barunlm/evaluation/mobile_temporal_counterfactual.py",
        "src/barunlm/model.py",
        "src/barunlm/training/checkpoint.py",
        "src/barunlm/training/config.py",
        "src/barunlm/training/data.py",
        "src/barunlm/training/mobile_temporal_experiment.py",
        "src/barunlm/training/trainer.py",
    }
)
_SECRET_PATTERNS = (
    re.compile(rb"wandb_v1_[A-Za-z0-9_-]{16,}", re.IGNORECASE),
    re.compile(rb"(?:ghp|github_pat)_[A-Za-z0-9_]{16,}", re.IGNORECASE),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(
        rb"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"]"
        rb"[A-Za-z0-9_./+=-]{20,}"
    ),
)


class LaunchProvenanceError(RuntimeError):
    """Raised when the staged launch cannot be bound unambiguously."""


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchProvenanceError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise LaunchProvenanceError(f"{label} must contain one JSON object")
    return payload


def _relative_file(root: Path, value: str, *, label: str) -> tuple[str, Path]:
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise LaunchProvenanceError(f"{label} must be a normalized relative POSIX path")
    candidate = root / relative
    if candidate.is_symlink():
        raise LaunchProvenanceError(f"{label} cannot be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise LaunchProvenanceError(f"{label} does not exist: {value}") from error
    if not resolved.is_file() or (resolved != root and root not in resolved.parents):
        raise LaunchProvenanceError(f"{label} must be a regular file inside the stage")
    return relative.as_posix(), resolved


def _load_staged_module(root: Path, relative_path: str, module_name: str) -> ModuleType:
    module_path = root / relative_path
    if not module_path.is_file() or module_path.is_symlink():
        raise LaunchProvenanceError(f"staged module is missing or is a symlink: {relative_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise LaunchProvenanceError(f"could not load staged module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    prior_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise LaunchProvenanceError(f"could not import staged module: {relative_path}") from error
    finally:
        sys.dont_write_bytecode = prior_dont_write_bytecode
    return module


def _load_staged_runner(root: Path) -> ModuleType:
    module = _load_staged_module(
        root,
        "scripts/run_mobile_temporal_counterfactual.py",
        "_barun_staged_temporal_runner",
    )
    if Path(module.REPOSITORY_ROOT).resolve() != root:
        raise LaunchProvenanceError("staged temporal runner resolved a different repository root")
    return module


def _validate_requirements(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise LaunchProvenanceError("frozen requirements are not readable UTF-8") from error
    active = tuple(
        line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")
    )
    if active != _REQUIRED_REQUIREMENT_LINES:
        raise LaunchProvenanceError(
            "frozen requirements must contain only exact pinned dependencies"
        )
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
        raise LaunchProvenanceError("requirements cannot install local, editable, or URL targets")


def _scan_stage_for_secrets(root: Path) -> None:
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.is_symlink():
            raise LaunchProvenanceError("staged source contains a symlink during secret scan")
        data = path.read_bytes()
        if any(pattern.search(data) is not None for pattern in _SECRET_PATTERNS):
            relative = path.relative_to(root).as_posix()
            raise LaunchProvenanceError(f"credential-like content found in staged file: {relative}")


def _run_git(source_root: Path, arguments: Sequence[str], *, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise LaunchProvenanceError(
            "git source verification failed: " + completed.stderr.decode("utf-8", "replace").strip()
        )
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8").strip()


def _verify_stage_from_git_commit(
    *,
    source_repository_root: Path,
    stage_root: Path,
    runner: ModuleType,
    screening_audit_relative_path: str,
    full_audit_relative_path: str,
    terminal_relative_path: str,
    requirements_path: Path,
) -> dict[str, str]:
    source_root = source_repository_root.resolve(strict=True)
    if not source_root.is_dir():
        raise LaunchProvenanceError("source repository root must be a directory")
    head = str(_run_git(source_root, ["rev-parse", "HEAD"]))
    branch = str(_run_git(source_root, ["branch", "--show-current"]))
    repository = str(_run_git(source_root, ["remote", "get-url", "origin"]))
    if _COMMIT_RE.fullmatch(head) is None or not branch or not repository:
        raise LaunchProvenanceError("git source identity is incomplete")
    if repository.rstrip("/") not in {
        _CANONICAL_REPOSITORY,
        _CANONICAL_REPOSITORY.removesuffix(".git"),
    }:
        raise LaunchProvenanceError("git origin is not the canonical public BarunLM repository")
    repository = _CANONICAL_REPOSITORY
    for arguments in (["diff", "--quiet", "HEAD", "--"], ["diff", "--cached", "--quiet"]):
        completed = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise LaunchProvenanceError(
                "tracked source repository is not clean at the claimed HEAD"
            )

    archive = _run_git(source_root, ["archive", "--format=tar", head], binary=True)
    assert isinstance(archive, bytes)
    committed: dict[str, tuple[str, int]] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        for member in handle.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise LaunchProvenanceError(
                    f"git commit contains a non-regular entry: {member.name}"
                )
            extracted = handle.extractfile(member)
            if extracted is None:
                raise LaunchProvenanceError(f"could not inspect committed file: {member.name}")
            content = extracted.read()
            committed[member.name] = (hashlib.sha256(content).hexdigest(), len(content))

    stage_files: dict[str, Path] = {}
    for path in sorted(candidate for candidate in stage_root.rglob("*") if candidate.is_file()):
        if path.is_symlink():
            raise LaunchProvenanceError("staged source contains a symlink")
        stage_files[path.relative_to(stage_root).as_posix()] = path
    for relative, (expected_sha256, expected_bytes) in committed.items():
        path = stage_files.get(relative)
        if (
            path is None
            or path.stat().st_size != expected_bytes
            or _sha256_file(path) != expected_sha256
        ):
            raise LaunchProvenanceError(f"stage differs from claimed git commit: {relative}")

    screening_parent = Path(screening_audit_relative_path).parent
    full_parent = Path(full_audit_relative_path).parent
    allowed_additions = {
        terminal_relative_path,
        requirements_path.name,
        runner.ATTEMPT_PREREGISTRATION_RELATIVE_PATH,
        runner.SOURCE_SNAPSHOT_RELATIVE_PATH,
    }
    allowed_additions.update(
        (screening_parent / name).as_posix() for name in runner.EXPECTED_ARTIFACT_FILENAMES.values()
    )
    allowed_additions.update(
        (full_parent / name).as_posix() for name in runner.EXPECTED_FULL_REFIT_FILENAMES.values()
    )
    unexpected = set(stage_files) - set(committed) - allowed_additions
    if unexpected:
        raise LaunchProvenanceError(
            "stage contains files outside the committed tree and frozen data allowlist: "
            + ", ".join(sorted(unexpected)[:10])
        )
    if not (set(stage_files) - set(committed)) <= allowed_additions:
        raise LaunchProvenanceError("stage additions do not match the frozen allowlist")
    return {"repository": repository, "commit": head, "branch": branch}


def _write_new(path: Path, payload: Mapping[str, Any]) -> str:
    content = _canonical_bytes(payload)
    if path.exists():
        if path.is_file() and path.read_bytes() == content:
            return hashlib.sha256(content).hexdigest()
        raise LaunchProvenanceError(f"refusing to replace non-identical provenance file: {path}")
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
    return hashlib.sha256(content).hexdigest()


def build_launch_provenance(
    *,
    stage_root: Path,
    config_relative_path: str,
    screening_audit_relative_path: str,
    full_audit_relative_path: str,
    terminal_relative_path: str,
    source_repository_root: Path,
    runner: ModuleType | None = None,
    safe_run: ModuleType | None = None,
) -> dict[str, Any]:
    root = stage_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise LaunchProvenanceError("stage root must be a real directory")
    runner = runner or _load_staged_runner(root)
    safe_run = safe_run or _load_staged_module(
        root, "infra/jarvis/safe_run.py", "_barun_staged_safe_run"
    )
    if (
        safe_run.MACHINE_ID_PLACEHOLDER != MACHINE_ID_PLACEHOLDER
        or safe_run.PREEXISTING_IDS_PLACEHOLDER != PREEXISTING_IDS_PLACEHOLDER
    ):
        raise LaunchProvenanceError("launch builder and safe_run inventory placeholders differ")

    config_relative, config_path = _relative_file(
        root, config_relative_path, label="scientific config"
    )
    screening_relative, screening_path = _relative_file(
        root, screening_audit_relative_path, label="screening view audit"
    )
    full_relative, full_path = _relative_file(
        root, full_audit_relative_path, label="full view audit"
    )
    terminal_relative, terminal_path = _relative_file(
        root, terminal_relative_path, label="terminal population"
    )
    config = _load_json(config_path, label="scientific config")
    if config.get("schema_version") != runner.CONFIG_SCHEMA_VERSION:
        raise LaunchProvenanceError("scientific config schema differs from the staged runner")
    if config.get("status") != "frozen_before_model_or_cuda_access":
        raise LaunchProvenanceError(
            "scientific config must be frozen before launch provenance is built"
        )
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise LaunchProvenanceError("scientific config has an invalid run_id")
    runner._validate_config(config)
    runner._preflight(config_path=config_path, materialization_receipt=screening_path)

    if _sha256_file(terminal_path) != runner.PINNED_REUSED_756_SHA256:
        raise LaunchProvenanceError("terminal population bytes differ from the pinned reused 756")
    materialization = config.get("materialization")
    full_refit = config.get("conditional_full_refit")
    if not isinstance(materialization, Mapping) or not isinstance(full_refit, Mapping):
        raise LaunchProvenanceError("scientific config lacks materialization contracts")
    screening_sha256 = _sha256_file(screening_path)
    full_sha256 = _sha256_file(full_path)
    if screening_sha256 != materialization.get("view_audit_sha256"):
        raise LaunchProvenanceError("screening view-audit bytes differ from the config")
    if full_sha256 != full_refit.get("audit_sha256"):
        raise LaunchProvenanceError("full view-audit bytes differ from the config")

    implementation = config.get("implementation_contract")
    frozen_files = implementation.get("files") if isinstance(implementation, Mapping) else None
    if not isinstance(frozen_files, dict) or not frozen_files:
        raise LaunchProvenanceError("scientific config lacks implementation_contract.files")
    missing_critical = _REQUIRED_IMPLEMENTATION_PATHS - set(frozen_files)
    if missing_critical:
        raise LaunchProvenanceError(
            "implementation contract omits critical launch files: "
            + ", ".join(sorted(missing_critical))
        )
    for relative, expected_sha256 in frozen_files.items():
        if not isinstance(relative, str) or not isinstance(expected_sha256, str):
            raise LaunchProvenanceError("implementation file bindings must be string pairs")
        _, implementation_path = _relative_file(root, relative, label="implementation file")
        if _sha256_file(implementation_path) != expected_sha256:
            raise LaunchProvenanceError(f"implementation hash differs: {relative}")

    _, requirements_path = _relative_file(
        root, _REQUIREMENTS_RELATIVE_PATH, label="frozen requirements"
    )
    _validate_requirements(requirements_path)
    _, root_requirements_copy = _relative_file(
        root, requirements_path.name, label="provider root requirements copy"
    )
    if root_requirements_copy.read_bytes() != requirements_path.read_bytes():
        raise LaunchProvenanceError("provider root requirements copy differs from frozen bytes")
    git_identity = _verify_stage_from_git_commit(
        source_repository_root=source_repository_root,
        stage_root=root,
        runner=runner,
        screening_audit_relative_path=screening_relative,
        full_audit_relative_path=full_relative,
        terminal_relative_path=terminal_relative,
        requirements_path=requirements_path,
    )
    _scan_stage_for_secrets(root)

    attempt_relative = runner.ATTEMPT_PREREGISTRATION_RELATIVE_PATH
    snapshot_relative = runner.SOURCE_SNAPSHOT_RELATIVE_PATH
    exact_exclusions = sorted({attempt_relative, snapshot_relative, terminal_relative})
    tree = runner._recompute_source_tree(root, excluded_exact_paths=exact_exclusions)
    snapshot = {
        "schema_version": runner.SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "run_id": run_id,
        "git": git_identity,
        "content_tree": {
            "algorithm": runner.SOURCE_TREE_ALGORITHM,
            **tree,
            "excluded_exact_paths": exact_exclusions,
            "excluded_directory_names": list(runner.SOURCE_TREE_EXCLUDED_DIRECTORY_NAMES),
            "excluded_directory_suffixes": list(runner.SOURCE_TREE_EXCLUDED_DIRECTORY_SUFFIXES),
            "excluded_file_names": list(runner.SOURCE_TREE_EXCLUDED_FILE_NAMES),
            "excluded_file_suffixes": list(runner.SOURCE_TREE_EXCLUDED_FILE_SUFFIXES),
        },
        "frozen_files": frozen_files,
        "staging_policy": {
            "credentials_present": False,
            "official_mobile_evaluation_present": False,
            "symlinks_present_before_launch": False,
        },
    }
    snapshot_path = root / snapshot_relative
    snapshot_sha256 = _write_new(snapshot_path, snapshot)

    retry_policy = config.get("retry_policy")
    if not isinstance(retry_policy, Mapping):
        raise LaunchProvenanceError("scientific config lacks retry_policy")
    compute = config.get("compute")
    if not isinstance(compute, Mapping):
        raise LaunchProvenanceError("scientific config lacks compute")
    attempt_compute = {
        name: compute.get(name)
        for name in (
            "provider",
            "template",
            "python_implementation",
            "python_version",
            "gpu",
            "num_gpus",
            "region",
            "is_spot",
        )
    }
    attempt_compute["max_gpu_job_minutes"] = compute.get("maximum_gpu_job_minutes")
    if attempt_compute != {
        "provider": "JarvisLabs",
        "template": runner.EXPECTED_JARVIS_TEMPLATE,
        "python_implementation": runner.EXPECTED_PYTHON_IMPLEMENTATION,
        "python_version": runner.EXPECTED_PYTHON_VERSION,
        "gpu": "H200",
        "num_gpus": 1,
        "region": "IN2",
        "is_spot": False,
        "max_gpu_job_minutes": 30,
    }:
        raise LaunchProvenanceError("scientific config compute binding changed")
    attempt = {
        "schema_version": runner.ATTEMPT_PREREGISTRATION_SCHEMA_VERSION,
        "status": runner.ATTEMPT_STATUS,
        "run_id": run_id,
        "attempt_id": f"{run_id}-attempt-1",
        "attempt_ordinal": 1,
        "prior_attempts": [],
        "durable_protected_machine_ids": sorted(runner.KNOWN_PROTECTED_JARVIS_IDS),
        "scientific_config": {
            "path": config_relative,
            "sha256": _sha256_file(config_path),
        },
        "source_snapshot": {"path": snapshot_relative, "sha256": snapshot_sha256},
        "materialization": {
            "screening_view_audit_path": screening_relative,
            "screening_view_audit_sha256": screening_sha256,
            "full_view_audit_path": full_relative,
            "full_view_audit_sha256": full_sha256,
        },
        "terminal_population": {
            "path": terminal_relative,
            "sha256": runner.PINNED_REUSED_756_SHA256,
            "rows": runner.PINNED_REUSED_756_ROWS,
            "access": "only_after_confirmation_pass_and_full_refit",
        },
        "prelaunch_inventory": {
            "captured_before_project_instance_creation": True,
            "protected_machine_ids": PREEXISTING_IDS_PLACEHOLDER,
            "project_machine_id": MACHINE_ID_PLACEHOLDER,
            "fresh_project_instance": True,
        },
        "compute": attempt_compute,
        "retry_lock": {
            name: retry_policy[name]
            for name in ("retry_after_any_held_out_signal", "retry_execution_policy")
        },
        "official_evaluation": {
            "policy": "forbidden",
            "rows": 961,
            "rows_present": 0,
            "rows_read": 0,
        },
    }
    attempt_path = root / attempt_relative
    attempt_sha256 = _write_new(attempt_path, attempt)
    if runner._recompute_source_tree(root, excluded_exact_paths=exact_exclusions) != tree:
        raise LaunchProvenanceError("writing excluded provenance changed the scientific tree")
    return {
        "schema_version": "barun-mobile-temporal-launch-provenance-build-v2",
        "run_id": run_id,
        "attempt_id": attempt["attempt_id"],
        "attempt_template_sha256": attempt_sha256,
        "source_snapshot_sha256": snapshot_sha256,
        "scientific_content_tree": {
            "sha256": tree["sha256"],
            "file_count": tree["file_count"],
            "content_bytes": tree["content_bytes"],
        },
        "inventory_binding": "pending_safe_run_after_fresh_instance_creation",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--config", required=True, help="stage-relative scientific config")
    parser.add_argument(
        "--screening-view-audit", required=True, help="stage-relative screening view audit"
    )
    parser.add_argument("--full-view-audit", required=True, help="stage-relative full view audit")
    parser.add_argument(
        "--terminal-manifest", required=True, help="stage-relative reused 756 JSONL"
    )
    parser.add_argument(
        "--source-repository-root",
        type=Path,
        required=True,
        help="clean local Git repository whose exact HEAD was archived into the stage",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = build_launch_provenance(
        stage_root=args.stage_root,
        config_relative_path=args.config,
        screening_audit_relative_path=args.screening_view_audit,
        full_audit_relative_path=args.full_view_audit,
        terminal_relative_path=args.terminal_manifest,
        source_repository_root=args.source_repository_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
