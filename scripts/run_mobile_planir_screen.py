"""Run the frozen construction-internal Mobile PlanIR screen on JarvisLabs.

The controller stays stdlib-only until it has validated the hash-bound launch
stage, exact fresh-instance inventory, frozen config/data, CPython runtime, and
the CPU-only test gate.  Only then does it import the Torch-backed execution
backend.  All durable outputs live outside the uploaded source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

# These settings must exist before the first possible Torch import.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

RUNNER_VERSION = "barun-mobile-planir-remote-controller-v1"
RUN_ID = "20260804-0545-mobile-planir-construction-screen-s17"
CONFIG_RELATIVE_PATH = "configs/mobile_planir_construction_screen_v1.json"
MATERIALIZATION_RELATIVE_PATH = (
    "experiments/runs/20260804-0545-mobile-planir-construction-screen-s17/materialized-v1"
)
MATERIALIZED_FILENAMES = (
    "assignment.jsonl",
    "audit.json",
    "exclusion.jsonl",
    "screen-a.jsonl",
    "screen-b.jsonl",
    "screen-c.jsonl",
    "train-a.jsonl",
    "train-b.jsonl",
    "train-c.jsonl",
)
SNAPSHOT_RELATIVE_PATH = "source-snapshot.json"
ATTEMPT_RELATIVE_PATH = "attempt-preregistration.json"
EXPECTED_OUTPUT_DIR = Path(f"/home/barun-artifacts/{RUN_ID}")
EXPECTED_PYTHON_IMPLEMENTATION = "CPython"
EXPECTED_PYTHON_VERSION = "3.11.10"
EXPECTED_TORCH_VERSION = "2.13.0"
EXPECTED_CUDA_RUNTIME = "13.0"
EXPECTED_CONFIG_SHA256 = "cca598e6f5a76a5e848a8b9a869cad779bbcbb17999217d2488111b22e2aa34c"
EXPECTED_REQUIREMENTS_SHA256 = "6db8f37c0c21aea4a4ad93d7193b82217e73d3090db6b8947a9a9321aefb21c9"
EXPECTED_MATERIALIZED_SHA256 = {
    "assignment.jsonl": "13eaad3ba5cf9da6dfad3dc5a73300599bf79ad29697c7d576807fed00ea6c3f",
    "audit.json": "646523ed299bb4b742c25c594aa08f4ecf79e3f2445db84fff9a6b6eead7d946",
    "exclusion.jsonl": "7593b556b9c096308ca1a9e9d9b89627ebf19710da930bd4dd5794c58879616a",
    "screen-a.jsonl": "08fd5e09bfda66b4b0cb31e7ef19bbb3a71a52e13f4fbe7c3b1dd3ea1cf402e7",
    "screen-b.jsonl": "81ab11b96a8330d19ebd41c6fcac5680f86a7652e721b495c1d24fc9f7a6c50d",
    "screen-c.jsonl": "c5c16ba5a785118dfdf8ca4665369ff375592c46ee8d066adbad9dcb0eff04ef",
    "train-a.jsonl": "702b6c0abd4c12d128b4e089cd8b2f7c57058fa7e50c5641978d225fb7529317",
    "train-b.jsonl": "30ccad10845196116c5205445b4b3e3aa2eb108434469e5fed3e27ae9d6fe3d2",
    "train-c.jsonl": "9aa0987b2b43af55e8dcbb1d6f921349282878ac58fe92d3ddadc953f97e51f5",
}
REMOTE_VALIDATION_TIMEOUT_SECONDS = 300
MAX_ESSENTIAL_FILE_BYTES = 256 * 1024 * 1024
MAX_ESSENTIAL_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")

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


class MobilePlanIRControllerError(RuntimeError):
    """The immutable remote execution boundary was violated."""


def _fail(message: str) -> NoReturn:
    raise MobilePlanIRControllerError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except MobilePlanIRControllerError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MobilePlanIRControllerError(f"cannot read {label}: {path}") from error
    if type(payload) is not dict:
        _fail(f"{label} must contain one exact JSON object")
    return payload


def _write_new_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite immutable artifact: {path}")
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


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    _write_new_bytes(path, _canonical_bytes(value))


def _copy_new(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        _fail(f"essential evidence source is not one regular file: {source}")
    size = source.stat().st_size
    if size > MAX_ESSENTIAL_FILE_BYTES:
        _fail(f"essential evidence file exceeds 256 MiB: {source}")
    _write_new_bytes(destination, source.read_bytes())


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if Path(__file__).is_symlink() or not root.is_dir():
        _fail("remote controller must run from one real staged repository root")
    return root


def _exact_stage_path(root: Path, supplied: Path, relative: str, *, label: str) -> Path:
    expected = (root / relative).resolve(strict=True)
    try:
        observed = supplied.resolve(strict=True)
    except OSError as error:
        raise MobilePlanIRControllerError(f"{label} does not exist") from error
    if observed != expected or expected.is_symlink():
        _fail(f"{label} must be the exact staged path {relative}")
    return expected


def _resource_id(value: str) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        _fail("JarvisLabs machine ID must be explicit positive numeric text")
    parsed = int(value)
    if parsed < 1:
        _fail("JarvisLabs machine ID must be positive")
    return parsed


def _runtime_identity() -> dict[str, object]:
    observed = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "sys_implementation_name": getattr(sys.implementation, "name", None),
        "virtual_environment_active": sys.prefix != sys.base_prefix,
    }
    expected = {
        "python_implementation": EXPECTED_PYTHON_IMPLEMENTATION,
        "python_version": EXPECTED_PYTHON_VERSION,
        "sys_implementation_name": "cpython",
        "virtual_environment_active": True,
    }
    if observed != expected:
        _fail(f"remote Python/runtime identity changed: observed {observed!r}")
    return {"expected": expected, "observed": observed, "status": "matched"}


def _load_bound_inventory(
    attempt_path: Path, *, machine_id: int
) -> tuple[list[int], dict[str, Any]]:
    attempt = _strict_json_object(attempt_path, label="bound attempt preregistration")
    inventory = attempt.get("prelaunch_inventory")
    durable = attempt.get("durable_protected_machine_ids")
    if type(inventory) is not dict or type(durable) is not list:
        _fail("bound attempt omits exact inventory evidence")
    protected = inventory.get("protected_machine_ids")
    if (
        inventory.get("captured_before_project_instance_creation") is not True
        or inventory.get("fresh_project_instance") is not True
        or inventory.get("project_machine_id") != machine_id
        or type(protected) is not list
        or any(type(value) is not int or value < 1 for value in protected)
        or protected != sorted(set(protected))
        or machine_id in protected
        or any(type(value) is not int or value < 1 for value in durable)
        or durable != sorted(set(durable))
        or not set(durable).issubset(protected)
    ):
        _fail("safe_run-bound inventory is invalid or does not prove a fresh exact ID")
    return list(protected), attempt


def _run_cpu_validation(root: Path, output_root: Path) -> dict[str, object]:
    if "torch" in sys.modules:
        _fail("Torch was imported before CPU-only validation")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_ADDOPTS": "-p no:cacheprovider",
            "PYTHONPATH": str(root / "src"),
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "disabled",
        }
    )
    commands = (
        [sys.executable, "-m", "pytest", "-q", *REMOTE_TEST_PATHS],
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *REMOTE_TEST_PATHS,
            "src",
            "scripts/run_mobile_planir_screen.py",
            "scripts/build_mobile_planir_launch_provenance.py",
        ],
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            *REMOTE_TEST_PATHS,
            "src",
            "scripts/run_mobile_planir_screen.py",
            "scripts/build_mobile_planir_launch_provenance.py",
        ],
    )
    records: list[dict[str, object]] = []
    combined_log = bytearray()
    for command in commands:
        combined_log.extend(("$ " + " ".join(command[1:]) + "\n").encode())
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=REMOTE_VALIDATION_TIMEOUT_SECONDS,
            )
            output = completed.stdout
            exit_code: int | None = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as error:
            captured = error.stdout or b""
            output = captured.encode() if isinstance(captured, str) else captured
            output += b"\nCPU-only validation timed out\n"
            exit_code = None
            timed_out = True
        combined_log.extend(output)
        combined_log.extend(b"\n")
        records.append(
            {
                "argv": ["python", *command[1:]],
                "exit_code": exit_code,
                "timed_out": timed_out,
                "output_sha256": hashlib.sha256(output).hexdigest(),
                "output_bytes": len(output),
            }
        )
        if timed_out or exit_code != 0:
            break
    log_path = output_root / "pre-cuda-validation.log"
    _write_new_bytes(log_path, bytes(combined_log))
    passed = len(records) == len(commands) and all(item["exit_code"] == 0 for item in records)
    receipt: dict[str, object] = {
        "schema_version": "barun-mobile-planir-pre-cuda-validation-v1",
        "status": "passed" if passed else "failed",
        "started_with_parent_torch_unimported": True,
        "finished_with_parent_torch_unimported": "torch" not in sys.modules,
        "screen_manifest_paths_in_argv_or_environment": False,
        "cuda_visible_devices": "",
        "wandb": "disabled",
        "timeout_seconds_per_command": REMOTE_VALIDATION_TIMEOUT_SECONDS,
        "commands": records,
        "log": "pre-cuda-validation.log",
        "log_sha256": _sha256_file(log_path),
        "log_bytes": log_path.stat().st_size,
    }
    _write_new_json(output_root / "pre-cuda-validation.json", receipt)
    if not passed:
        _fail("CPU-only test/lint/format gate failed")
    if "torch" in sys.modules:
        _fail("CPU child validation contaminated the parent Torch import state")
    return receipt


def _preflight(
    *,
    root: Path,
    config_path: Path,
    materialization_dir: Path,
    snapshot_sha256: str,
    attempt_path: Path,
    attempt_sha256: str,
    machine_id: int,
    output_root: Path,
) -> dict[str, object]:
    if SHA256_RE.fullmatch(snapshot_sha256) is None or SHA256_RE.fullmatch(attempt_sha256) is None:
        _fail("source/attempt bindings must be exact lowercase SHA-256 values")
    if "torch" in sys.modules:
        _fail("Torch was imported before stdlib launch preflight")
    runtime = _runtime_identity()
    scripts_path = root / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    from build_mobile_planir_launch_provenance import (
        validate_launch_provenance,
    )

    validation = validate_launch_provenance(
        stage_root=root,
        expected_source_snapshot_sha256=snapshot_sha256,
        expected_attempt_sha256=attempt_sha256,
        require_bound_inventory=True,
        allow_runtime=True,
    )
    protected, attempt = _load_bound_inventory(attempt_path, machine_id=machine_id)
    if validation.get("attempt_id") != attempt.get("attempt_id"):
        _fail("validated attempt identity differs from the bound attempt")
    config = _strict_json_object(config_path, label="frozen experiment config")
    if config.get("run_id") != RUN_ID:
        _fail("frozen config run ID changed")
    if materialization_dir != (root / MATERIALIZATION_RELATIVE_PATH).resolve(strict=True):
        _fail("materialization directory differs from the frozen staged path")
    cpu_validation = _run_cpu_validation(root, output_root)
    post_validation = validate_launch_provenance(
        stage_root=root,
        expected_source_snapshot_sha256=snapshot_sha256,
        expected_attempt_sha256=attempt_sha256,
        require_bound_inventory=True,
        allow_runtime=True,
    )
    if post_validation != validation:
        _fail("launch provenance changed during CPU-only validation")
    return {
        "schema_version": "barun-mobile-planir-controller-preflight-v1",
        "runner_version": RUNNER_VERSION,
        "captured_at": _utc_now(),
        "run_id": RUN_ID,
        "attempt_id": validation["attempt_id"],
        "jarvis_resource_id": machine_id,
        "fresh_instance": True,
        "protected_resource_ids": protected,
        "config_sha256": _sha256_file(config_path),
        "source_snapshot_sha256": snapshot_sha256,
        "attempt_preregistration_sha256": attempt_sha256,
        "scientific_tree": validation["scientific_tree"],
        "runtime": runtime,
        "cpu_validation": cpu_validation,
        "model_or_cuda_loaded": False,
        "screen_json_decoded": False,
        "construction_screen_json_rows_decoded": 0,
        "official_mobile_961_rows_read": 0,
        "reused_mobile_756_rows_read": 0,
        "human_selection_rows_read": 0,
        "human_confirmation_rows_read": 0,
        "wandb": "disabled",
        "cublas_workspace_configured_before_torch_import": True,
    }


def _verify_torch_runtime(torch_module: Any) -> dict[str, object]:
    torch_version = str(torch_module.__version__)
    cuda_runtime = str(torch_module.version.cuda)
    if torch_version.split("+", 1)[0] != EXPECTED_TORCH_VERSION:
        _fail(f"expected Torch {EXPECTED_TORCH_VERSION}, observed {torch_version}")
    if cuda_runtime != EXPECTED_CUDA_RUNTIME:
        _fail(f"expected CUDA runtime {EXPECTED_CUDA_RUNTIME}, observed {cuda_runtime}")
    return {
        "torch_version": torch_version,
        "torch_base_version": EXPECTED_TORCH_VERSION,
        "cuda_runtime": cuda_runtime,
        "expected_cuda_build": "cu130",
    }


def _selected_execution_files(execution_dir: Path) -> list[Path]:
    selected: dict[str, Path] = {}
    exact_names = {
        "all-checkpoints-frozen.json",
        "all-raw-predictions-frozen.json",
        "environment-receipt.json",
        "failure.json",
        "result.json",
        "screen-access-started.json",
    }
    allowed_suffixes = {".json", ".jsonl"}
    for path in sorted(execution_dir.rglob("*")):
        if path.is_symlink():
            _fail(f"execution evidence contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(execution_dir)
        if any(part.startswith(("essential", ".essential")) for part in relative.parts):
            continue
        if path.name in exact_names:
            selected[relative.as_posix()] = path
            continue
        if path.suffix not in allowed_suffixes:
            continue
        if relative.parts[0] in {"evaluation", "fits", "training-configs"}:
            selected[relative.as_posix()] = path
            continue
        if relative.parts[0] == "training" and path.name in {
            "checkpoint_manifest.json",
            "data_rejections.jsonl",
            "failure.json",
            "metrics.jsonl",
            "resolved_config.json",
            "run_manifest.json",
            "summary.json",
        }:
            selected[relative.as_posix()] = path
    return [selected[key] for key in sorted(selected)]


def _build_essential_bundle(
    *,
    root: Path,
    output_root: Path,
    status: str,
    error: BaseException | None,
) -> Path:
    export_root = output_root / "export"
    essential = export_root / "essential"
    if essential.exists() or essential.is_symlink():
        _fail("refusing to overwrite an existing essential evidence bundle")
    build = export_root / f".essential-build-{os.getpid()}"
    if build.exists() or build.is_symlink():
        _fail("temporary essential evidence path already exists")
    build.mkdir(parents=True)
    try:
        controller = build / "controller"
        preflight_path = output_root / "preflight-receipt.json"
        preflight = (
            _strict_json_object(preflight_path, label="controller preflight receipt")
            if preflight_path.is_file()
            else None
        )
        if error is None and preflight is None:
            _fail("successful export requires the frozen controller preflight receipt")
        snapshot_expected = (
            preflight.get("source_snapshot_sha256") if preflight is not None else None
        )
        attempt_expected = (
            preflight.get("attempt_preregistration_sha256") if preflight is not None else None
        )
        if snapshot_expected is not None and (
            not isinstance(snapshot_expected, str) or SHA256_RE.fullmatch(snapshot_expected) is None
        ):
            _fail("preflight source-snapshot binding is invalid")
        if attempt_expected is not None and (
            not isinstance(attempt_expected, str) or SHA256_RE.fullmatch(attempt_expected) is None
        ):
            _fail("preflight attempt binding is invalid")
        controller_files = (
            (
                root / SNAPSHOT_RELATIVE_PATH,
                SNAPSHOT_RELATIVE_PATH,
                True,
                snapshot_expected,
            ),
            (
                root / ATTEMPT_RELATIVE_PATH,
                ATTEMPT_RELATIVE_PATH,
                True,
                attempt_expected,
            ),
            (
                root / CONFIG_RELATIVE_PATH,
                CONFIG_RELATIVE_PATH,
                True,
                EXPECTED_CONFIG_SHA256,
            ),
            (
                root / "requirements/mobile-planir-screen.txt",
                "requirements/mobile-planir-screen.txt",
                True,
                EXPECTED_REQUIREMENTS_SHA256,
            ),
            (
                output_root / "pre-cuda-validation.log",
                "pre-cuda-validation.log",
                error is None,
                None,
            ),
            (
                output_root / "pre-cuda-validation.json",
                "pre-cuda-validation.json",
                error is None,
                None,
            ),
            (
                output_root / "preflight-receipt.json",
                "preflight-receipt.json",
                error is None,
                None,
            ),
            (
                output_root / "torch-runtime-receipt.json",
                "torch-runtime-receipt.json",
                error is None,
                None,
            ),
        )
        for source, relative, required, expected_sha256 in controller_files:
            if not source.is_file():
                if required:
                    _fail(f"mandatory controller evidence is missing: {relative}")
                continue
            if expected_sha256 is not None and _sha256_file(source) != expected_sha256:
                _fail(f"mandatory controller evidence changed before export: {relative}")
            _copy_new(source, controller / relative)
        materialization_root = root / MATERIALIZATION_RELATIVE_PATH
        observed_materialized_names = tuple(
            sorted(source.name for source in materialization_root.iterdir())
        )
        if observed_materialized_names != MATERIALIZED_FILENAMES:
            _fail("materialized evidence inventory changed before essential export")
        for name in MATERIALIZED_FILENAMES:
            source = materialization_root / name
            if source.is_symlink() or not source.is_file():
                _fail("materialized evidence contains a non-regular entry")
            if _sha256_file(source) != EXPECTED_MATERIALIZED_SHA256[name]:
                _fail(f"materialized evidence changed before export: {name}")
            _copy_new(source, build / "data" / "materialized-v1" / source.name)
        execution_dir = output_root / "execution"
        if execution_dir.is_dir():
            for source in _selected_execution_files(execution_dir):
                _copy_new(source, build / "execution" / source.relative_to(execution_dir))
        if error is not None:
            _write_new_json(
                build / "controller-failure.json",
                {
                    "schema_version": "barun-mobile-planir-controller-failure-v1",
                    "failed_at": _utc_now(),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "artifacts_preserved": True,
                    "automatic_retry_attempted": False,
                    "model_output_rows": "derive_from_preserved_execution_artifacts",
                    "screen_rows_read": "derive_from_preserved_execution_artifacts",
                    "retry_authorized": False,
                },
            )
        files = [path for path in sorted(build.rglob("*")) if path.is_file()]
        weight_like = [
            path.relative_to(build).as_posix()
            for path in files
            if path.name == "optimizer.pt"
            or path.suffix.casefold() in {".bin", ".pt", ".pth", ".safetensors"}
        ]
        if weight_like:
            _fail("essential evidence unexpectedly contains model/optimizer weights")
        records: dict[str, dict[str, object]] = {}
        total = 0
        for path in files:
            size = path.stat().st_size
            if size > MAX_ESSENTIAL_FILE_BYTES:
                _fail(f"essential file exceeds its fixed size bound: {path}")
            total += size
            records[path.relative_to(build).as_posix()] = {
                "bytes": size,
                "sha256": _sha256_file(path),
            }
        if total > MAX_ESSENTIAL_TOTAL_BYTES:
            _fail("essential evidence bundle exceeds 2 GiB")
        checkpoint_manifest_count = sum(
            path.name == "checkpoint_manifest.json"
            for path in files
            if "training" in path.relative_to(build).parts
        )
        remote_weight_count = (
            sum(path.name == "model.safetensors" for path in execution_dir.rglob("*"))
            if execution_dir.is_dir()
            else 0
        )
        _write_new_json(
            build / "artifact-manifest.json",
            {
                "schema_version": "barun-mobile-planir-essential-v1",
                "run_id": RUN_ID,
                "status": status,
                "file_count_excluding_manifest": len(records),
                "total_bytes_excluding_manifest": total,
                "files": records,
                "screening_weights_in_bundle": False,
                "weight_like_files_in_bundle": [],
                "screening_weight_files_present_in_remote_execution_tree": remote_weight_count,
                "checkpoint_manifest_files_in_bundle": checkpoint_manifest_count,
                "materialized_construction_artifacts_in_bundle": len(MATERIALIZED_FILENAMES),
                "official_mobile_961_rows_read": 0,
                "reused_mobile_756_rows_read": 0,
                "human_selection_rows_read": 0,
                "human_confirmation_rows_read": 0,
            },
        )
        export_root.mkdir(parents=True, exist_ok=True)
        build.replace(essential)
    finally:
        if build.exists():
            shutil.rmtree(build)
    return essential


def run_remote(args: argparse.Namespace) -> Mapping[str, object]:
    root = _repository_root()
    config_path = _exact_stage_path(root, args.config, CONFIG_RELATIVE_PATH, label="config")
    materialization_dir = _exact_stage_path(
        root,
        args.materialization_dir,
        MATERIALIZATION_RELATIVE_PATH,
        label="materialization directory",
    )
    attempt_path = _exact_stage_path(
        root,
        args.attempt_preregistration,
        ATTEMPT_RELATIVE_PATH,
        label="attempt preregistration",
    )
    machine_id = _resource_id(args.jarvis_machine_id)
    output_root = args.output_dir.resolve()
    if output_root != EXPECTED_OUTPUT_DIR or output_root == root or root in output_root.parents:
        _fail(f"output directory must be exact external path {EXPECTED_OUTPUT_DIR}")
    output_root.mkdir(parents=True, exist_ok=False)
    preflight = _preflight(
        root=root,
        config_path=config_path,
        materialization_dir=materialization_dir,
        snapshot_sha256=args.source_snapshot_sha256,
        attempt_path=attempt_path,
        attempt_sha256=args.attempt_preregistration_sha256,
        machine_id=machine_id,
        output_root=output_root,
    )
    _write_new_json(output_root / "preflight-receipt.json", preflight)
    if "torch" in sys.modules:
        _fail("Torch was imported before the frozen preflight receipt")
    source_path = root / "src"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    import torch

    from barunlm.training.mobile_planir_experiment import (
        execute_mobile_planir_experiment,
    )

    runtime = _verify_torch_runtime(torch)
    _write_new_json(output_root / "torch-runtime-receipt.json", runtime)
    result = execute_mobile_planir_experiment(
        config_path=config_path,
        materialization_dir=materialization_dir,
        output_dir=output_root,
        preflight_receipt=preflight,
        jarvis_resource_id=str(machine_id),
        cublas_configured_before_torch_import=True,
    )
    _build_essential_bundle(
        root=root,
        output_root=output_root,
        status=str(result.get("status", "completed")),
        error=None,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--materialization-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--attempt-preregistration", type=Path, required=True)
    parser.add_argument("--attempt-preregistration-sha256", required=True)
    parser.add_argument("--jarvis-machine-id", required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_dir.resolve()
    try:
        result = run_remote(args)
    except BaseException as error:
        if output_root == EXPECTED_OUTPUT_DIR and output_root.is_dir():
            try:
                _build_essential_bundle(
                    root=_repository_root(),
                    output_root=output_root,
                    status="failed_closed",
                    error=error,
                )
            except BaseException as bundle_error:  # noqa: BLE001
                print(
                    f"essential evidence preservation also failed: {bundle_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MobilePlanIRControllerError as error:
        print(f"mobile PlanIR execution rejected: {error}", file=sys.stderr)
        raise SystemExit(2) from None
