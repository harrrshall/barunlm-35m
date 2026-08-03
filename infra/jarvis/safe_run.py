"""Run one fresh JarvisLabs job with exact-ID ownership and guaranteed pause attempts.

This controller intentionally cannot run on or mutate arbitrary existing instances. It creates a
fresh ``barun-*`` instance through ``jl create``, records the returned ID before attaching a managed
run, collects logs/artifacts, and pauses only that exact ID. A persisted record supports recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

HERE = Path(__file__).resolve().parent
DEFAULT_PROTECTED_PATH = HERE / "protected-resources.json"
PROTECTED_RESOURCES_ENV = "BARUN_JARVIS_PROTECTED_RESOURCES"
NAME_PATTERN = re.compile(r"barun-[a-z0-9][a-z0-9-]{2,62}$")
TERMINAL_RUN_STATES = {"succeeded", "failed", "stopped"}
RUN_STATUS_FIELDS = (
    "run_id",
    "machine_id",
    "state",
    "exit_code",
    "instance_status",
    "started_at",
    "finished_at",
)
WATCHDOG_POLL_SECONDS = 5.0
WATCHDOG_PAUSE_GRACE_SECONDS = 300.0
SENSITIVE_ARGUMENT_PATTERNS = (
    "api-key",
    "api_key",
    "apikey",
    "password=",
    "secret=",
    "token=",
    "--password",
    "--secret",
    "--token",
)
SENSITIVE_FILE_NAMES = {".netrc", "credentials.json", "id_ed25519", "id_rsa"}
SENSITIVE_FILE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
IGNORED_SCAN_DIRECTORIES = {".git", ".venv", "__pycache__"}
MACHINE_ID_PLACEHOLDER = "__SAFE_RUN_MACHINE_ID__"
PREEXISTING_IDS_PLACEHOLDER = "__SAFE_RUN_PREEXISTING_IDS__"
PYTHON_RUNTIME_IDENTITY_CODE = (
    "import json,platform;"
    "print(json.dumps({'implementation':platform.python_implementation(),"
    "'version':platform.python_version()},sort_keys=True,separators=(',',':')))"
)


class SafetyError(RuntimeError):
    """Raised when ownership or lifecycle invariants cannot be proved."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fail(message: str) -> NoReturn:
    raise SafetyError(message)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def requirements_receipt(path: Path | None) -> dict[str, str] | None:
    """Bind both the path and bytes of an optional environment recipe."""

    if path is None:
        return None
    if not path.is_file():
        fail(f"requirements file does not exist: {path}")
    return {"path": str(path), "sha256": sha256_file(path.resolve())}


def attempt_inventory_file(*, target: Path, relative_path: Path) -> Path:
    """Resolve one regular target-relative attempt file without allowing path escape."""

    if relative_path.is_absolute() or ".." in relative_path.parts:
        fail("attempt-inventory binding path must be a safe target-relative path")
    root = target.resolve()
    candidate = root.joinpath(relative_path)
    if candidate.is_symlink():
        fail("attempt-inventory binding path must not be a symlink")
    try:
        path = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise SafetyError("attempt-inventory binding file does not exist") from error
    if root != path and root not in path.parents:
        fail("attempt-inventory binding path escapes the run target")
    if not path.is_file():
        fail("attempt-inventory binding path must be a regular file")
    return path


def requested_attempt_compute(args: argparse.Namespace) -> dict[str, Any]:
    """Return the exact operational contract that must be frozen before instance creation."""

    return {
        "provider": "JarvisLabs",
        "template": str(args.template),
        "python_implementation": str(args.python_implementation),
        "python_version": str(args.python_version),
        "gpu": str(args.gpu),
        "num_gpus": int(args.num_gpus),
        "region": args.region,
        "is_spot": bool(args.spot),
        "max_gpu_job_minutes": int(args.max_runtime_minutes),
    }


def validate_unbound_attempt_contract(
    *, target: Path, relative_path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Fail before creation unless the attempt is unbound and freezes the requested runtime."""

    path = attempt_inventory_file(target=target, relative_path=relative_path)
    payload = load_json(path)
    if not isinstance(payload, dict):
        fail("attempt-inventory binding file must contain a JSON object")
    expected_inventory = {
        "captured_before_project_instance_creation": True,
        "protected_machine_ids": PREEXISTING_IDS_PLACEHOLDER,
        "project_machine_id": MACHINE_ID_PLACEHOLDER,
        "fresh_project_instance": True,
    }
    if payload.get("prelaunch_inventory") != expected_inventory:
        fail("attempt preregistration must be unbound before fresh instance creation")
    expected_compute = requested_attempt_compute(args)
    if payload.get("compute") != expected_compute:
        fail("attempt preregistration compute contract differs from the requested fresh run")
    return {
        "path": relative_path.as_posix(),
        "sha256": sha256_file(path),
        "compute": expected_compute,
        "unbound_before_instance_creation": True,
        "verified_at": utc_now(),
    }


def bind_attempt_inventory(
    *,
    target: Path,
    relative_path: Path,
    record_path: Path,
    record: dict[str, Any],
    machine_id: int,
) -> None:
    """Bind a fresh machine and its pre-create denylist before the target is uploaded.

    Scientific fields are already frozen in the attempt template. Only the two values that cannot
    exist before ``jl create`` are substituted here. The attempt file is deliberately excluded
    from each experiment's staged content-tree hash and is independently hash-bound in the final
    artifact manifest.
    """

    path = attempt_inventory_file(target=target, relative_path=relative_path)

    payload = load_json(path)
    if not isinstance(payload, dict):
        fail("attempt-inventory binding file must contain a JSON object")
    inventory_block = payload.get("prelaunch_inventory")
    if not isinstance(inventory_block, dict) or set(inventory_block) != {
        "captured_before_project_instance_creation",
        "protected_machine_ids",
        "project_machine_id",
        "fresh_project_instance",
    }:
        fail("attempt preregistration has an invalid prelaunch_inventory block")
    if inventory_block["captured_before_project_instance_creation"] is not True:
        fail("attempt preregistration must assert a pre-create inventory")
    if inventory_block["fresh_project_instance"] is not True:
        fail("attempt preregistration must assert a fresh project instance")

    live_preexisting_ids = {int(item["machine_id"]) for item in record["preexisting_resources"]}
    expected_protected_ids = sorted(live_preexisting_ids | permanent_protected_ids())
    protected_value = inventory_block["protected_machine_ids"]
    if protected_value == PREEXISTING_IDS_PLACEHOLDER:
        inventory_block["protected_machine_ids"] = expected_protected_ids
    elif protected_value != expected_protected_ids:
        fail("attempt preregistration denylist differs from safe_run pre-create inventory")

    machine_value = inventory_block["project_machine_id"]
    prior_owned_ids = {
        int(value)
        for value in [*record.get("machine_id_history", []), record.get("machine_id")]
        if value is not None
    }
    if machine_value != MACHINE_ID_PLACEHOLDER and machine_value not in prior_owned_ids:
        fail("attempt preregistration project ID is neither a placeholder nor an owned ID")
    if machine_id in expected_protected_ids or machine_id in permanent_protected_ids():
        fail("refusing to bind a protected/pre-existing machine ID")

    before_sha256 = sha256_file(path)
    inventory_block["project_machine_id"] = machine_id
    atomic_write_json(path, payload)
    after_sha256 = sha256_file(path)
    record["attempt_inventory_binding"] = {
        "path": relative_path.as_posix(),
        "machine_id": machine_id,
        "protected_machine_ids": expected_protected_ids,
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "bound_before_attached_run": True,
    }
    persist_event(
        record_path,
        record,
        "attempt_inventory_bound",
        machine_id=machine_id,
        path=relative_path.as_posix(),
        after_sha256=after_sha256,
    )


def run_json(command: Sequence[str], *, timeout: float | None = None) -> Any:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or "no diagnostic output"
        raise RuntimeError(f"command failed ({completed.returncode}): {command[0]}: {detail}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{command[0]} did not return valid JSON") from error


def filtered_instance(item: dict[str, Any]) -> dict[str, Any]:
    template = item.get("template")
    if template is None:
        template = item.get("framework")
    return {
        "machine_id": int(item["machine_id"]),
        "name": item.get("name"),
        "status": item.get("status"),
        "gpu_type": item.get("gpu_type"),
        "num_gpus": item.get("num_gpus"),
        "region": item.get("region"),
        "is_spot": item.get("is_spot"),
        "template": template,
    }


def inventory() -> list[dict[str, Any]]:
    payload = run_json(["jl", "list", "--json"], timeout=60)
    if not isinstance(payload, list):
        fail("jl list returned a non-list payload")
    return [filtered_instance(item) for item in payload]


def protected_resources_path() -> Path:
    """Resolve the private denylist without requiring it in the public source tree."""

    configured = os.environ.get(PROTECTED_RESOURCES_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_PROTECTED_PATH


def permanent_protected_ids() -> set[int]:
    path = protected_resources_path()
    if not path.is_file():
        fail(
            "private JarvisLabs denylist is unavailable; create the local default file or set "
            f"{PROTECTED_RESOURCES_ENV} to an external JSON path"
        )
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("resources"), list):
        fail("private JarvisLabs denylist must contain a resources list")
    protected: set[int] = set()
    for index, item in enumerate(payload["resources"]):
        if not isinstance(item, dict) or isinstance(item.get("machine_id"), bool):
            fail(f"private JarvisLabs denylist resource {index} has an invalid machine_id")
        try:
            machine_id = int(item["machine_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise SafetyError(
                f"private JarvisLabs denylist resource {index} has an invalid machine_id"
            ) from error
        if machine_id < 1:
            fail(f"private JarvisLabs denylist resource {index} has an invalid machine_id")
        protected.add(machine_id)
    if not protected:
        fail("private JarvisLabs denylist must contain at least one protected machine ID")
    return protected


def validate_name(name: str) -> None:
    if not NAME_PATTERN.fullmatch(name):
        fail("instance name must match barun-[a-z0-9][a-z0-9-]{2,62}")


def validate_no_sensitive_arguments(arguments: Sequence[str]) -> None:
    for argument in arguments:
        lowered = argument.lower()
        if any(pattern in lowered for pattern in SENSITIVE_ARGUMENT_PATTERNS):
            fail("refusing a command argument that appears to contain a credential")


def validate_no_sensitive_files(target: Path) -> None:
    """Fail closed because jl's directory sync does not honor the repository .gitignore."""
    root = target.resolve()
    if root.is_file():
        candidates = [root]
    else:
        candidates = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and not any(part in IGNORED_SCAN_DIRECTORIES for part in path.relative_to(root).parts)
        ]
    suspicious = []
    for path in candidates:
        name = path.name.lower()
        if (
            name == ".env"
            or name.startswith(".env.")
            or name in SENSITIVE_FILE_NAMES
            or path.suffix.lower() in SENSITIVE_FILE_SUFFIXES
        ):
            suspicious.append(str(path.relative_to(root)) if root.is_dir() else path.name)
    if suspicious:
        fail(f"refusing to sync potentially sensitive files: {sorted(suspicious)}")


def instance_by_id(machine_id: int) -> dict[str, Any]:
    payload = run_json(["jl", "get", str(machine_id), "--json"], timeout=60)
    if not isinstance(payload, dict):
        fail("jl get returned a non-object payload")
    return filtered_instance(payload)


def assert_owned(record: dict[str, Any], live: dict[str, Any]) -> None:
    machine_id = int(record["machine_id"])
    if machine_id in permanent_protected_ids():
        fail(f"machine {machine_id} is permanently protected")
    preexisting = {int(item["machine_id"]) for item in record["preexisting_resources"]}
    if machine_id in preexisting:
        fail(f"machine {machine_id} existed before this run")
    if not record.get("created_by_safe_run"):
        fail("run record does not prove safe_run ownership")
    if live["machine_id"] != machine_id:
        fail("live instance ID does not match run record")
    expected_name = record["instance_name"]
    if live.get("name") != expected_name or not NAME_PATTERN.fullmatch(expected_name):
        fail("live instance name does not match the recorded barun-* name")


def requested_hardware_attestation(
    args: argparse.Namespace, live: dict[str, Any]
) -> dict[str, Any]:
    """Fail before upload when the live fresh instance differs from the requested hardware.

    This is deliberately separate from :func:`assert_owned`: a hardware mismatch must prevent
    work from being uploaded, while exact-ID ownership must remain sufficient to pause the
    mistakenly provisioned project instance during cleanup.
    """

    expected = {
        "gpu_type": str(args.gpu),
        "num_gpus": int(args.num_gpus),
        "region": args.region,
        "is_spot": bool(args.spot),
        "template": str(args.template),
    }
    observed = {
        "gpu_type": live.get("gpu_type"),
        "num_gpus": live.get("num_gpus"),
        "region": live.get("region"),
        "is_spot": live.get("is_spot"),
        "template": live.get("template"),
    }
    mismatches: list[str] = []
    if not isinstance(observed["gpu_type"], str) or (
        observed["gpu_type"].casefold() != expected["gpu_type"].casefold()
    ):
        mismatches.append("gpu_type")
    if isinstance(observed["num_gpus"], bool):
        mismatches.append("num_gpus")
    else:
        try:
            observed_count = int(observed["num_gpus"])
        except (TypeError, ValueError):
            mismatches.append("num_gpus")
        else:
            observed["num_gpus"] = observed_count
            if observed_count != expected["num_gpus"]:
                mismatches.append("num_gpus")
    if expected["region"] is not None and observed["region"] != expected["region"]:
        mismatches.append("region")
    if type(observed["is_spot"]) is not bool or observed["is_spot"] is not expected["is_spot"]:
        mismatches.append("is_spot")
    if observed["template"] != expected["template"]:
        mismatches.append("template")
    if mismatches:
        fail("live instance differs from requested hardware: " + ", ".join(sorted(set(mismatches))))
    return {
        "verified_at": utc_now(),
        "verified_before_upload": True,
        "requested": expected,
        "observed": observed,
    }


def attest_live_hardware(
    record_path: Path,
    record: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    machine_id = int(record["machine_id"])
    live = instance_by_id(machine_id)
    assert_owned(record, live)
    receipt = requested_hardware_attestation(args, live)
    record["hardware_attestation"] = receipt
    persist_event(
        record_path,
        record,
        "hardware_attested_before_upload",
        machine_id=machine_id,
        gpu_type=live.get("gpu_type"),
        num_gpus=live.get("num_gpus"),
        region=live.get("region"),
        is_spot=live.get("is_spot"),
        template=live.get("template"),
    )
    return receipt


def python_runtime_identity(machine_id: int) -> dict[str, str]:
    """Read only the Python implementation/version from one exact instance ID."""

    payload = run_json(
        [
            "jl",
            "exec",
            str(machine_id),
            "--json",
            "--",
            "python3",
            "-c",
            PYTHON_RUNTIME_IDENTITY_CODE,
        ],
        timeout=60,
    )
    if not isinstance(payload, dict) or payload.get("machine_id") != machine_id:
        fail("python runtime probe did not return the exact owned machine ID")
    if type(payload.get("exit_code")) is not int or payload["exit_code"] != 0:
        fail("python runtime probe did not exit successfully")
    stdout = payload.get("stdout")
    if not isinstance(stdout, str):
        fail("python runtime probe did not return JSON stdout")
    try:
        identity = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise SafetyError("python runtime probe stdout is not valid JSON") from error
    if (
        not isinstance(identity, dict)
        or set(identity) != {"implementation", "version"}
        or any(not isinstance(identity[name], str) for name in identity)
    ):
        fail("python runtime probe returned an invalid identity object")
    return identity


def attest_live_python_runtime(
    record_path: Path,
    record: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Attest exact Python identity after hardware checks and before binding/upload."""

    machine_id = int(record["machine_id"])
    preexisting = {int(item["machine_id"]) for item in record["preexisting_resources"]}
    if (
        not record.get("created_by_safe_run")
        or machine_id in preexisting
        or machine_id in permanent_protected_ids()
    ):
        fail("python runtime attestation requires one fresh exact owned machine ID")
    observed = python_runtime_identity(machine_id)
    expected = {
        "implementation": str(args.python_implementation),
        "version": str(args.python_version),
    }
    if observed != expected:
        fail(
            "live python runtime differs from requested identity: "
            f"observed {observed!r}, expected {expected!r}"
        )
    receipt = {
        "verified_at": utc_now(),
        "verified_before_inventory_binding": True,
        "verified_before_upload": True,
        "machine_id": machine_id,
        "requested": expected,
        "observed": observed,
    }
    record["python_runtime_attestation"] = receipt
    persist_event(
        record_path,
        record,
        "python_runtime_attested_before_inventory_binding_and_upload",
        machine_id=machine_id,
        implementation=observed["implementation"],
        version=observed["version"],
    )
    return receipt


def discover_failed_creation(*, name: str, preexisting_ids: set[int]) -> dict[str, Any] | None:
    """Find the one exact-name resource created before jl emitted its JSON summary."""
    candidates = [
        item
        for item in inventory()
        if item["machine_id"] not in preexisting_ids and item.get("name") == name
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        fail(f"ambiguous failed creation: {len(candidates)} new resources named {name!r}")
    candidate = candidates[0]
    if int(candidate["machine_id"]) in permanent_protected_ids():
        fail("failed creation discovery returned a permanently protected machine ID")
    return candidate


def recover_migrated_owned_id(record_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Recover one exact-name replacement ID created by resuming an owned container."""

    old_id = int(record["machine_id"])
    preexisting = {int(item["machine_id"]) for item in record["preexisting_resources"]}
    live_inventory = inventory()
    if any(int(item["machine_id"]) == old_id for item in live_inventory):
        fail("old owned ID is still live; refusing replacement-ID recovery")
    candidates = [
        item
        for item in live_inventory
        if item.get("name") == record["instance_name"]
        and int(item["machine_id"]) not in preexisting
        and int(item["machine_id"]) not in permanent_protected_ids()
    ]
    if len(candidates) != 1:
        fail(f"replacement-ID recovery found {len(candidates)} exact-name candidates")
    replacement = candidates[0]
    new_id = int(replacement["machine_id"])
    record.setdefault("machine_id_history", []).append(old_id)
    record["machine_id"] = new_id
    persist_event(
        record_path,
        record,
        "owned_resume_id_migrated",
        previous_machine_id=old_id,
        machine_id=new_id,
    )
    assert_owned(record, replacement)
    return replacement


def pause_owned(record: dict[str, Any], *, attempts: int = 4) -> dict[str, Any]:
    machine_id = int(record["machine_id"])
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            live = instance_by_id(machine_id)
            assert_owned(record, live)
            if str(live.get("status", "")).lower() == "paused":
                return live
            run_json(["jl", "pause", str(machine_id), "--yes", "--json"], timeout=180)
            for _ in range(12):
                live = instance_by_id(machine_id)
                if str(live.get("status", "")).lower() == "paused":
                    return live
                time.sleep(5)
            last_error = RuntimeError("instance did not report Paused after pause request")
        except (RuntimeError, subprocess.SubprocessError, OSError, KeyError, ValueError) as error:
            # Cleanup retries must retain the original failure for the urgent final diagnostic.
            last_error = error
        if attempt < attempts:
            time.sleep(min(5 * attempt, 15))
    raise RuntimeError(f"URGENT: failed to pause owned machine {machine_id}: {last_error}")


def persist_event(record_path: Path, record: dict[str, Any], event: str, **details: Any) -> None:
    record.setdefault("events", []).append({"at": utc_now(), "event": event, **details})
    atomic_write_json(record_path, record)


def persist_verified_pause(
    record_path: Path,
    record: dict[str, Any],
    proof: dict[str, Any],
    *,
    event: str,
) -> None:
    """Persist the latest pause proof and retire any stale cleanup failure."""

    record["final_instance"] = proof
    record["controller_finished_at"] = utc_now()
    record.pop("cleanup_error", None)
    persist_event(record_path, record, event, status=proof.get("status"))


def resume_owned(record_path: Path, record: dict[str, Any]) -> int:
    """Resume an owned instance and adopt only a proven replacement machine ID."""

    previous_id = int(record["machine_id"])
    try:
        summary = run_json(["jl", "resume", str(previous_id), "--yes", "--json"], timeout=300)
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        persist_event(
            record_path,
            record,
            "owned_resume_cli_failed",
            machine_id=previous_id,
            error_type=type(error).__name__,
        )
        replacement = recover_migrated_owned_id(record_path, record)
        recovered_id = int(replacement["machine_id"])
        persist_event(
            record_path,
            record,
            "owned_resume_recovered_after_cli_failure",
            previous_machine_id=previous_id,
            machine_id=recovered_id,
        )
        return recovered_id

    if not isinstance(summary, dict):
        fail("jl resume returned a non-object payload")
    if "machine_id" not in summary:
        fail("jl resume JSON omitted the required machine_id")
    raw_resumed_id = summary["machine_id"]
    if isinstance(raw_resumed_id, bool):
        fail("jl resume returned an invalid machine_id")
    try:
        resumed_id = int(raw_resumed_id)
    except (TypeError, ValueError) as error:
        raise SafetyError("jl resume returned an invalid machine_id") from error
    if resumed_id < 1:
        fail("jl resume returned an invalid machine_id")

    if resumed_id != previous_id:
        preexisting = {int(item["machine_id"]) for item in record["preexisting_resources"]}
        if resumed_id in preexisting or resumed_id in permanent_protected_ids():
            fail("jl resume returned a protected/pre-existing replacement ID")
        record.setdefault("machine_id_history", []).append(previous_id)
        record["machine_id"] = resumed_id
        persist_event(
            record_path,
            record,
            "owned_resume_id_migrated",
            previous_machine_id=previous_id,
            machine_id=resumed_id,
        )
    return resumed_id


def build_create_command(args: argparse.Namespace) -> list[str]:
    command = [
        "jl",
        "create",
        "--gpu",
        args.gpu,
        "--template",
        args.template,
        "--storage",
        str(args.storage),
        "--name",
        args.name,
        "--num-gpus",
        str(args.num_gpus),
    ]
    if args.region:
        command.extend(["--region", args.region])
    if args.spot:
        command.append("--spot")
    command.extend(["--yes", "--json"])
    return command


def required_create_machine_id(summary: Any) -> int:
    """Extract the explicit positive integer machine ID required from jl create."""

    if not isinstance(summary, dict):
        fail("jl create returned a non-object summary")
    if "machine_id" not in summary:
        fail("jl create JSON omitted the required machine_id")
    raw_machine_id = summary["machine_id"]
    if isinstance(raw_machine_id, bool):
        fail("jl create returned an invalid machine_id")
    if isinstance(raw_machine_id, int):
        machine_id = raw_machine_id
    elif (
        isinstance(raw_machine_id, str) and raw_machine_id.isascii() and raw_machine_id.isdecimal()
    ):
        machine_id = int(raw_machine_id)
    else:
        fail("jl create returned an invalid machine_id")
    if machine_id < 1:
        fail("jl create returned an invalid machine_id")
    return machine_id


def managed_setup_command(args: argparse.Namespace) -> str | None:
    """Avoid mutating an explicitly frozen requirements target before source validation."""

    if args.requirements is not None:
        return None
    return "uv pip install -e '.[dev]'"


def build_attached_run_command(
    args: argparse.Namespace,
    machine_id: int,
    record: dict[str, Any] | None = None,
) -> list[str]:
    command = [
        "jl",
        "run",
        str(args.target),
        "--script",
        args.script,
        "--on",
        str(machine_id),
        "--no-follow",
        "--yes",
        "--json",
    ]
    setup = managed_setup_command(args)
    if setup is not None:
        command.extend(["--setup", setup])
    if args.requirements:
        command.extend(["--requirements", str(args.requirements)])
    remote_args = list(args.remote_args)
    if getattr(args, "append_jarvis_machine_id", False):
        remote_args.extend(["--jarvis-machine-id", str(machine_id)])
    if getattr(args, "append_bound_attempt_sha256", False):
        if not getattr(args, "bind_attempt_inventory", None):
            fail("--append-bound-attempt-sha256 requires --bind-attempt-inventory")
        if record is None:
            fail("bound attempt SHA-256 is unavailable without the run record")
        binding = record.get("attempt_inventory_binding")
        digest = binding.get("after_sha256") if isinstance(binding, dict) else None
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            fail("bound attempt SHA-256 is unavailable or invalid")
        remote_args.extend(["--attempt-preregistration-sha256", digest])
    if remote_args:
        command.append("--")
        command.extend(remote_args)
    return command


def wait_for_ssh(record: dict[str, Any], *, timeout_seconds: int = 300) -> None:
    machine_id = int(record["machine_id"])
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            live = instance_by_id(machine_id)
            assert_owned(record, live)
        except (RuntimeError, subprocess.SubprocessError, OSError) as error:
            last_error = error
            time.sleep(5)
            continue
        if str(live.get("status", "")).lower() == "running":
            try:
                run_json(
                    ["jl", "exec", str(machine_id), "--json", "--", "echo", "barun-ready"],
                    timeout=60,
                )
                return
            except (RuntimeError, subprocess.SubprocessError, OSError) as error:
                last_error = error
        time.sleep(5)
    raise RuntimeError(f"owned machine {machine_id} did not become SSH-ready: {last_error}")


def download_artifact(
    machine_id: int, remote_path: str, local_path: Path, *, recursive: bool
) -> dict[str, Any]:
    command = ["jl", "download", str(machine_id), remote_path, str(local_path)]
    if recursive:
        command.append("--recursive")
    command.append("--json")
    result = run_json(command, timeout=1800)
    return result if isinstance(result, dict) else {"result": result}


def filtered_run_snapshot(record: dict[str, Any], snapshot: Any) -> dict[str, Any]:
    """Normalize a live managed-run status and bind it to the exact owned machine/run."""

    if not isinstance(snapshot, dict):
        fail("run status returned a non-object payload")
    if str(snapshot.get("run_id")) != str(record.get("run_id")):
        fail("run status returned a different run ID")
    raw_machine_id = snapshot.get("machine_id")
    if isinstance(raw_machine_id, bool):
        fail("run status returned an invalid machine ID")
    try:
        machine_id = int(raw_machine_id)
    except (TypeError, ValueError) as error:
        raise SafetyError("run status omitted a valid machine ID") from error
    if machine_id != int(record["machine_id"]):
        fail("run status returned a different machine ID")
    filtered = {key: snapshot.get(key) for key in RUN_STATUS_FIELDS if key in snapshot}
    filtered["run_id"] = str(record["run_id"])
    filtered["machine_id"] = machine_id
    filtered["state"] = str(snapshot.get("state", "")).lower()
    return filtered


def authoritative_exit_code(snapshot: dict[str, Any]) -> int:
    """Derive outcome only from the final live status, never from log metadata."""

    state = str(snapshot.get("state", "")).lower()
    if state not in TERMINAL_RUN_STATES:
        fail(f"run status is not terminal: {state!r}")
    raw_exit_code = snapshot.get("exit_code")
    if isinstance(raw_exit_code, bool):
        fail("terminal run status has an invalid exit code")
    if raw_exit_code is None:
        if state == "succeeded":
            fail("succeeded run status omitted its exit code")
        return 1
    try:
        exit_code = int(raw_exit_code)
    except (TypeError, ValueError) as error:
        raise SafetyError("terminal run status has an invalid exit code") from error
    if exit_code < 0:
        fail("terminal run status has a negative exit code")
    if state in {"failed", "stopped"} and exit_code == 0:
        return 1
    return exit_code


def monitor_owned_run(
    record_path: Path,
    record: dict[str, Any],
    *,
    max_runtime_minutes: int,
    poll_seconds: int,
) -> tuple[dict[str, Any], int]:
    deadline = time.monotonic() + max_runtime_minutes * 60
    while time.monotonic() < deadline:
        raw = run_json(["jl", "run", "status", str(record["run_id"]), "--json"], timeout=90)
        snapshot = filtered_run_snapshot(record, raw)
        record["latest_run_status"] = snapshot
        atomic_write_json(record_path, record)
        print(json.dumps({"event": "status", **snapshot}), flush=True)
        if snapshot["state"] in TERMINAL_RUN_STATES:
            exit_code = authoritative_exit_code(snapshot)
            record["final_run_status"] = snapshot
            record["remote_exit_code"] = exit_code
            persist_event(
                record_path,
                record,
                "run_terminal_status_recorded",
                state=snapshot["state"],
                exit_code=exit_code,
            )
            return snapshot, exit_code
        if snapshot.get("exit_code") is not None:
            fail("run status reported an exit code without a terminal state")
        time.sleep(poll_seconds)
    persist_event(record_path, record, "runtime_deadline_exceeded")
    raise TimeoutError(f"run exceeded {max_runtime_minutes} minutes")


def collect_run_evidence(
    record_path: Path,
    record: dict[str, Any],
    args: argparse.Namespace,
    *,
    reason: str,
) -> list[str]:
    """Attempt artifact and log collection independently, with the artifact first."""

    if not record.get("run_id"):
        return []
    collection = record.setdefault("evidence_collection", {})
    failures: list[str] = []
    machine_id = int(record["machine_id"])

    if args.artifact and collection.get("artifact_status") != "succeeded":
        try:
            artifact_result = download_artifact(
                machine_id,
                args.artifact,
                args.artifact_dest.resolve(),
                recursive=args.artifact_recursive,
            )
        except (RuntimeError, subprocess.SubprocessError, OSError, SafetyError) as error:
            failures.append("artifact")
            collection["artifact_status"] = "failed"
            collection["artifact_error_type"] = type(error).__name__
            persist_event(
                record_path,
                record,
                "artifact_download_failed",
                reason=reason,
                error_type=type(error).__name__,
            )
        else:
            record["artifact_download"] = artifact_result
            collection["artifact_status"] = "succeeded"
            collection.pop("artifact_error_type", None)
            persist_event(
                record_path,
                record,
                "artifact_downloaded",
                reason=reason,
                remote=args.artifact,
                local=str(args.artifact_dest.resolve()),
            )

    if collection.get("log_status") != "succeeded":
        try:
            log_payload = run_json(
                ["jl", "run", "logs", str(record["run_id"]), "--json"], timeout=300
            )
            if not isinstance(log_payload, dict):
                fail("run logs returned a non-object payload")
            run_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(record["run_id"]))
            log_path = record_path.with_name(f"{record_path.stem}-{run_slug}.remote.log")
            log_path.write_text(str(log_payload.get("content", "")), encoding="utf-8")
        except (RuntimeError, subprocess.SubprocessError, OSError, SafetyError) as error:
            failures.append("log")
            collection["log_status"] = "failed"
            collection["log_error_type"] = type(error).__name__
            persist_event(
                record_path,
                record,
                "log_collection_failed",
                reason=reason,
                error_type=type(error).__name__,
            )
        else:
            record["remote_log_file"] = str(log_path)
            record["remote_log_sha256"] = sha256_file(log_path)
            record["log_reported_run_exit_code"] = log_payload.get("run_exit_code")
            collection["log_status"] = "succeeded"
            collection.pop("log_error_type", None)
            persist_event(
                record_path,
                record,
                "log_collected",
                reason=reason,
                path=str(log_path),
            )
    return failures


def stop_owned_run(record: dict[str, Any]) -> dict[str, Any]:
    """Stop only the managed run proven to be attached to the exact owned machine."""

    live = instance_by_id(int(record["machine_id"]))
    assert_owned(record, live)
    raw = run_json(["jl", "run", "status", str(record["run_id"]), "--json"], timeout=90)
    snapshot = filtered_run_snapshot(record, raw)
    if snapshot["state"] in TERMINAL_RUN_STATES:
        return {"action": "not_needed", "status": snapshot}
    result = run_json(["jl", "run", "stop", str(record["run_id"]), "--json"], timeout=180)
    return {"action": "stop_requested", "result": result}


def best_effort_stop_owned_run(record_path: Path, record: dict[str, Any], *, reason: str) -> None:
    if not record.get("run_id"):
        return
    try:
        outcome = stop_owned_run(record)
    except (RuntimeError, subprocess.SubprocessError, OSError, SafetyError) as error:
        record["postlaunch_stop"] = {
            "status": "failed",
            "reason": reason,
            "error_type": type(error).__name__,
        }
        persist_event(
            record_path,
            record,
            "owned_run_stop_failed",
            reason=reason,
            error_type=type(error).__name__,
        )
    else:
        record["postlaunch_stop"] = {"status": "completed", "reason": reason, **outcome}
        persist_event(
            record_path,
            record,
            "owned_run_stop_checked",
            reason=reason,
            action=outcome["action"],
        )


def collect_after_postlaunch_error(
    record_path: Path,
    record: dict[str, Any],
    args: argparse.Namespace,
    *,
    reason: str,
) -> None:
    if not record.get("run_id"):
        return
    best_effort_stop_owned_run(record_path, record, reason=reason)
    collect_run_evidence(record_path, record, args, reason=reason)


def arm_deadline_watchdog(
    record_path: Path,
    record: dict[str, Any],
    *,
    deadline_epoch: float,
) -> dict[str, Any]:
    """Detach a second local process that can stop/pause the exact ID after the deadline."""

    token = f"{os.getpid()}-{time.time_ns()}"
    disarm_path = record_path.with_name(f".{record_path.name}.watchdog-{token}.disarm.json")
    proof_path = record_path.with_name(f"{record_path.name}.watchdog-{token}.proof.json")
    log_path = record_path.with_name(f"{record_path.name}.watchdog-{token}.log")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_deadline-watchdog",
        "--record",
        str(record_path),
        "--deadline-epoch",
        repr(float(deadline_epoch)),
        "--disarm-path",
        str(disarm_path),
        "--proof-path",
        str(proof_path),
    ]
    with log_path.open("ab") as watchdog_log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=watchdog_log,
            stderr=watchdog_log,
            start_new_session=True,
            close_fds=True,
        )
    receipt = {
        "status": "armed",
        "armed_at": utc_now(),
        "deadline_epoch": float(deadline_epoch),
        "pid": process.pid,
        "disarm_path": str(disarm_path),
        "proof_path": str(proof_path),
        "log_path": str(log_path),
        "scope": "exact-owned-machine-and-attached-run-only",
    }
    record["deadline_watchdog"] = receipt
    persist_event(
        record_path,
        record,
        "deadline_watchdog_armed",
        pid=process.pid,
        deadline_epoch=float(deadline_epoch),
    )
    return receipt


def disarm_deadline_watchdog(record_path: Path, record: dict[str, Any]) -> None:
    watchdog = record.get("deadline_watchdog")
    if not isinstance(watchdog, dict) or watchdog.get("status") != "armed":
        return
    disarm_path = Path(str(watchdog["disarm_path"]))
    atomic_write_json(
        disarm_path,
        {
            "schema_version": 1,
            "disarmed_at": utc_now(),
            "machine_id": int(record["machine_id"]),
            "run_id": str(record.get("run_id")),
        },
    )
    watchdog["status"] = "disarmed"
    watchdog["disarmed_at"] = utc_now()
    proof_path = Path(str(watchdog["proof_path"]))
    if proof_path.is_file():
        try:
            watchdog["proof"] = load_json(proof_path)
        except (OSError, json.JSONDecodeError):
            watchdog["proof"] = {"status": "unreadable"}
    persist_event(record_path, record, "deadline_watchdog_disarmed")


def run_deadline_watchdog(args: argparse.Namespace) -> int:
    """Detached deadline worker; it never discovers or mutates an unrecorded resource."""

    record_path = args.record.resolve()
    disarm_path = args.disarm_path.resolve()
    proof_path = args.proof_path.resolve()
    while time.time() < args.deadline_epoch:
        if disarm_path.is_file():
            return 0
        time.sleep(min(WATCHDOG_POLL_SECONDS, max(0.05, args.deadline_epoch - time.time())))
    if disarm_path.is_file():
        return 0

    proof: dict[str, Any] = {
        "schema_version": 1,
        "triggered_at": utc_now(),
        "deadline_epoch": args.deadline_epoch,
        "record": str(record_path),
    }
    try:
        record = load_json(record_path)
        if not isinstance(record, dict) or not record.get("run_id") or "machine_id" not in record:
            fail("watchdog record does not prove an attached owned run")
        try:
            proof["run_stop"] = stop_owned_run(record)
        except (RuntimeError, subprocess.SubprocessError, OSError, SafetyError) as error:
            proof["run_stop"] = {"status": "failed", "error_type": type(error).__name__}
            pause_grace_seconds = 0.0
        else:
            pause_grace_seconds = WATCHDOG_PAUSE_GRACE_SECONDS
        # Stop enforces the managed-run deadline. Keep the instance reachable briefly so a live
        # primary controller can retrieve partial evidence before either controller pauses it.
        grace_deadline = time.time() + pause_grace_seconds
        while time.time() < grace_deadline:
            if disarm_path.is_file():
                return 0
            time.sleep(min(WATCHDOG_POLL_SECONDS, max(0.05, grace_deadline - time.time())))
        pause_proof = pause_owned(record)
        proof["status"] = "pause_verified"
        proof["machine_id"] = int(record["machine_id"])
        proof["run_id"] = str(record["run_id"])
        proof["final_instance"] = pause_proof
        result = 0
    except (
        RuntimeError,
        subprocess.SubprocessError,
        OSError,
        SafetyError,
        KeyError,
        ValueError,
    ) as error:
        proof["status"] = "cleanup_failed"
        proof["error_type"] = type(error).__name__
        result = 2
    atomic_write_json(proof_path, proof)
    return result


def run_owned_retry(args: argparse.Namespace, record_path: Path) -> int:
    """Retry only a project-owned resource recovered from a failed fresh launch."""

    record = load_json(record_path)
    if not isinstance(record, dict) or "machine_id" not in record:
        fail("retry record does not prove a machine ID")
    requested = record.get("requested")
    expected = {
        "gpu": args.gpu,
        "num_gpus": args.num_gpus,
        "template": args.template,
        "python_implementation": args.python_implementation,
        "python_version": args.python_version,
        "region": args.region,
        "spot": args.spot,
        "storage_gb": args.storage,
        "target": str(args.target),
        "script": args.script,
        "requirements": requirements_receipt(args.requirements),
        "setup": managed_setup_command(args),
        "remote_args": args.remote_args,
        "artifact": args.artifact,
        "artifact_dest": str(args.artifact_dest),
        "artifact_recursive": args.artifact_recursive,
        "append_jarvis_machine_id": getattr(args, "append_jarvis_machine_id", False),
        "append_bound_attempt_sha256": getattr(args, "append_bound_attempt_sha256", False),
        "bind_attempt_inventory": (
            str(args.bind_attempt_inventory) if args.bind_attempt_inventory else None
        ),
        "max_runtime_minutes": args.max_runtime_minutes,
    }
    if requested != expected:
        fail("retry arguments differ from the immutable failed-launch record")
    live = instance_by_id(int(record["machine_id"]))
    assert_owned(record, live)
    if str(live.get("status", "")).lower() != "paused":
        fail("owned retry requires the recorded instance to be Paused")

    primary_error: BaseException | None = None

    def on_signal(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    old_handlers = {
        signum: signal.signal(signum, on_signal) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    machine_id = int(record["machine_id"])
    try:
        previous_run_id = record.pop("run_id", None)
        if previous_run_id is not None:
            record.setdefault("run_id_history", []).append(str(previous_run_id))
        persist_event(record_path, record, "owned_retry_requested", machine_id=machine_id)
        machine_id = resume_owned(record_path, record)
        persist_event(record_path, record, "owned_retry_resumed", machine_id=machine_id)
        wait_for_ssh(record)
        persist_event(record_path, record, "owned_retry_ssh_ready", machine_id=machine_id)
        attest_live_hardware(record_path, record, args)
        attest_live_python_runtime(record_path, record, args)

        if args.bind_attempt_inventory:
            bind_attempt_inventory(
                target=args.target,
                relative_path=args.bind_attempt_inventory,
                record_path=record_path,
                record=record,
                machine_id=machine_id,
            )

        if requirements_receipt(args.requirements) != record["requested"]["requirements"]:
            fail("requirements bytes changed after retry validation and before upload")
        summary = run_json(build_attached_run_command(args, machine_id, record), timeout=1800)
        if not isinstance(summary, dict):
            fail("jl run returned a non-object summary")
        if int(summary["machine_id"]) != machine_id:
            fail("attached run returned a different machine ID")
        record.update({"run_id": str(summary["run_id"]), "created_by_safe_run": True})
        record["launch_summary"] = {
            "machine_id": machine_id,
            "run_id": str(summary["run_id"]),
            "remote_log": summary.get("remote_log"),
            "remote_exit_code": summary.get("remote_exit_code"),
            "target_kind": summary.get("target_kind"),
            "instance_origin": summary.get("instance_origin"),
            "lifecycle_policy": summary.get("lifecycle_policy"),
            "owned_retry": True,
        }
        record.pop("final_run_status", None)
        record.pop("remote_exit_code", None)
        record.pop("evidence_collection", None)
        record.pop("artifact_download", None)
        record.pop("remote_log_file", None)
        record.pop("remote_log_sha256", None)
        persist_event(record_path, record, "run_started", machine_id=machine_id)
        print(
            json.dumps(
                {"event": "run_started", "machine_id": machine_id, "run_id": record["run_id"]}
            ),
            flush=True,
        )
        arm_deadline_watchdog(
            record_path,
            record,
            deadline_epoch=time.time() + args.max_runtime_minutes * 60,
        )
        _, exit_code = monitor_owned_run(
            record_path,
            record,
            max_runtime_minutes=args.max_runtime_minutes,
            poll_seconds=args.poll_seconds,
        )
        collection_failures = collect_run_evidence(record_path, record, args, reason="terminal_run")
        if collection_failures:
            raise RuntimeError("one or more terminal evidence collections failed")
        return exit_code
    except BaseException as error:
        primary_error = error
        persist_event(record_path, record, "controller_error", error=repr(error))
        collect_after_postlaunch_error(
            record_path,
            record,
            args,
            reason=type(error).__name__,
        )
        raise
    finally:
        try:
            proof = pause_owned(record)
            persist_verified_pause(record_path, record, proof, event="pause_verified")
            disarm_deadline_watchdog(record_path, record)
            print(
                json.dumps(
                    {
                        "event": "pause_verified",
                        "machine_id": proof["machine_id"],
                        "status": proof.get("status"),
                    }
                ),
                flush=True,
            )
        except (RuntimeError, subprocess.SubprocessError, OSError, KeyError, ValueError) as error:
            record["cleanup_error"] = repr(error)
            atomic_write_json(record_path, record)
            print(str(error), file=sys.stderr, flush=True)
            if primary_error is None:
                raise
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def run_fresh(args: argparse.Namespace) -> int:
    validate_name(args.name)
    if args.remote_args[:1] == ["--"]:
        args.remote_args = args.remote_args[1:]
    validate_no_sensitive_arguments(args.remote_args)
    if args.retry_owned and args.bind_attempt_inventory:
        fail("--retry-owned is forbidden with a fresh-instance attempt binding")
    if getattr(args, "append_bound_attempt_sha256", False) and not args.bind_attempt_inventory:
        fail("--append-bound-attempt-sha256 requires --bind-attempt-inventory")
    target_root = args.target.resolve()
    if not target_root.exists():
        fail(f"run target does not exist: {args.target}")
    if not target_root.is_dir():
        fail("safe_run requires a project-directory target")
    validate_no_sensitive_files(args.target)
    if not args.retry_owned and args.artifact and args.artifact_dest.resolve().exists():
        fail("fresh-run artifact destination must not already exist")
    record_path = args.record.resolve()
    if record_path == target_root or target_root in record_path.parents:
        fail("safe_run record must be outside the uploaded target")
    if record_path.exists():
        if args.retry_owned:
            return run_owned_retry(args, record_path)
        fail(f"refusing to overwrite existing run record: {record_path}")

    attempt_contract = None
    if args.bind_attempt_inventory:
        attempt_contract = validate_unbound_attempt_contract(
            target=args.target,
            relative_path=args.bind_attempt_inventory,
            args=args,
        )

    preexisting = inventory()
    preexisting_ids = {int(item["machine_id"]) for item in preexisting}
    if not permanent_protected_ids().issubset(preexisting_ids):
        # A protected resource may have been destroyed by its owner; it remains denied by ID.
        missing = sorted(permanent_protected_ids() - preexisting_ids)
    else:
        missing = []

    record: dict[str, Any] = {
        "schema_version": 1,
        "controller": "infra/jarvis/safe_run.py",
        "controller_sha256": sha256_file(Path(__file__).resolve()),
        "controller_started_at": utc_now(),
        "created_by_safe_run": False,
        "instance_name": args.name,
        "requested": {
            "gpu": args.gpu,
            "num_gpus": args.num_gpus,
            "template": args.template,
            "python_implementation": args.python_implementation,
            "python_version": args.python_version,
            "region": args.region,
            "spot": args.spot,
            "storage_gb": args.storage,
            "target": str(args.target),
            "script": args.script,
            "requirements": requirements_receipt(args.requirements),
            "setup": managed_setup_command(args),
            "remote_args": args.remote_args,
            "artifact": args.artifact,
            "artifact_dest": str(args.artifact_dest),
            "artifact_recursive": args.artifact_recursive,
            "append_jarvis_machine_id": getattr(args, "append_jarvis_machine_id", False),
            "append_bound_attempt_sha256": getattr(args, "append_bound_attempt_sha256", False),
            "bind_attempt_inventory": (
                str(args.bind_attempt_inventory) if args.bind_attempt_inventory else None
            ),
            "max_runtime_minutes": args.max_runtime_minutes,
        },
        "preexisting_resources": preexisting,
        "protected_ids_missing_from_live_inventory": missing,
        "attempt_contract_precreate_validation": attempt_contract,
        "events": [],
    }
    atomic_write_json(record_path, record)

    machine_claimed = False
    primary_error: BaseException | None = None

    def on_signal(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    old_handlers = {
        signum: signal.signal(signum, on_signal) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    try:
        command = build_create_command(args)
        persist_event(record_path, record, "create_requested")
        try:
            create_summary = run_json(command, timeout=1800)
        except (RuntimeError, subprocess.SubprocessError, OSError) as error:
            persist_event(
                record_path,
                record,
                "create_cli_failed",
                error_type=type(error).__name__,
            )
            discovered = discover_failed_creation(name=args.name, preexisting_ids=preexisting_ids)
            if discovered is None:
                raise
            machine_id = int(discovered["machine_id"])
            record.update(
                {
                    "machine_id": machine_id,
                    "created_by_safe_run": True,
                    "creation_recovered_after_cli_failure": True,
                    "create_summary": discovered,
                }
            )
            machine_claimed = True
            assert_owned(record, discovered)
            persist_event(
                record_path,
                record,
                "machine_discovered_after_create_failure",
                machine_id=machine_id,
            )
        else:
            try:
                machine_id = required_create_machine_id(create_summary)
            except SafetyError as response_error:
                recovery_reason = str(response_error)
                persist_event(
                    record_path,
                    record,
                    "create_response_invalid",
                    reason=recovery_reason,
                    response_type=type(create_summary).__name__,
                )
                discovered = discover_failed_creation(
                    name=args.name, preexisting_ids=preexisting_ids
                )
                if discovered is None:
                    persist_event(
                        record_path,
                        record,
                        "invalid_create_response_recovery_no_candidate",
                    )
                    raise
                machine_id = int(discovered["machine_id"])
                record.update(
                    {
                        "machine_id": machine_id,
                        "created_by_safe_run": True,
                        "creation_recovered_after_invalid_response": True,
                        "creation_recovery_reason": recovery_reason,
                    }
                )
                machine_claimed = True
                assert_owned(record, discovered)
                persist_event(
                    record_path,
                    record,
                    "machine_discovered_after_invalid_create_response",
                    machine_id=machine_id,
                    reason=recovery_reason,
                )
                raise
            if machine_id in preexisting_ids or machine_id in permanent_protected_ids():
                fail(f"jl create returned protected/pre-existing machine ID {machine_id}")
            record.update(
                {
                    "machine_id": machine_id,
                    "created_by_safe_run": True,
                    "create_summary": filtered_instance(create_summary),
                }
            )
            machine_claimed = True
            persist_event(record_path, record, "machine_created", machine_id=machine_id)

        wait_for_ssh(record)
        persist_event(record_path, record, "machine_ssh_ready", machine_id=machine_id)
        attest_live_hardware(record_path, record, args)
        attest_live_python_runtime(record_path, record, args)

        if args.bind_attempt_inventory:
            prebind_contract = validate_unbound_attempt_contract(
                target=args.target,
                relative_path=args.bind_attempt_inventory,
                args=args,
            )
            if (
                not isinstance(attempt_contract, dict)
                or prebind_contract["sha256"] != attempt_contract["sha256"]
            ):
                fail("attempt preregistration changed after pre-create validation")
            record["attempt_contract_prebind_validation"] = prebind_contract
            bind_attempt_inventory(
                target=args.target,
                relative_path=args.bind_attempt_inventory,
                record_path=record_path,
                record=record,
                machine_id=machine_id,
            )

        if requirements_receipt(args.requirements) != record["requested"]["requirements"]:
            fail("requirements bytes changed after inventory and before upload")
        persist_event(record_path, record, "attached_run_requested", machine_id=machine_id)
        summary = run_json(build_attached_run_command(args, machine_id, record), timeout=1800)
        if not isinstance(summary, dict):
            fail("jl run returned a non-object summary")
        if int(summary["machine_id"]) != machine_id:
            fail("attached run returned a different machine ID")
        record.update({"run_id": str(summary["run_id"]), "created_by_safe_run": True})

        record.update(
            {
                "launch_summary": {
                    "machine_id": machine_id,
                    "run_id": str(summary["run_id"]),
                    "remote_log": summary.get("remote_log"),
                    "remote_exit_code": summary.get("remote_exit_code"),
                    "target_kind": summary.get("target_kind"),
                    "instance_origin": summary.get("instance_origin"),
                    "lifecycle_policy": summary.get("lifecycle_policy"),
                },
            }
        )
        persist_event(record_path, record, "run_started", machine_id=machine_id)
        print(
            json.dumps(
                {"event": "run_started", "machine_id": machine_id, "run_id": record["run_id"]}
            ),
            flush=True,
        )
        arm_deadline_watchdog(
            record_path,
            record,
            deadline_epoch=time.time() + args.max_runtime_minutes * 60,
        )
        _, exit_code = monitor_owned_run(
            record_path,
            record,
            max_runtime_minutes=args.max_runtime_minutes,
            poll_seconds=args.poll_seconds,
        )
        collection_failures = collect_run_evidence(record_path, record, args, reason="terminal_run")
        if collection_failures:
            raise RuntimeError("one or more terminal evidence collections failed")
        return exit_code
    except BaseException as error:
        primary_error = error
        persist_event(record_path, record, "controller_error", error=repr(error))
        collect_after_postlaunch_error(
            record_path,
            record,
            args,
            reason=type(error).__name__,
        )
        raise
    finally:
        try:
            if machine_claimed:
                try:
                    proof = pause_owned(record)
                    persist_verified_pause(record_path, record, proof, event="pause_verified")
                    disarm_deadline_watchdog(record_path, record)
                    print(
                        json.dumps(
                            {
                                "event": "pause_verified",
                                "machine_id": proof["machine_id"],
                                "status": proof.get("status"),
                            }
                        ),
                        flush=True,
                    )
                except (
                    RuntimeError,
                    subprocess.SubprocessError,
                    OSError,
                    KeyError,
                    ValueError,
                ) as cleanup_error:
                    record["cleanup_error"] = repr(cleanup_error)
                    atomic_write_json(record_path, record)
                    print(str(cleanup_error), file=sys.stderr, flush=True)
                    if primary_error is None:
                        raise
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def cleanup_record(args: argparse.Namespace) -> int:
    record_path = args.record.resolve()
    record = load_json(record_path)
    if not isinstance(record, dict) or "machine_id" not in record:
        fail("record does not contain a machine_id")
    try:
        instance_by_id(int(record["machine_id"]))
    except RuntimeError:
        if not args.recover_migrated_id:
            raise
        recover_migrated_owned_id(record_path, record)
    proof = pause_owned(record)
    persist_verified_pause(record_path, record, proof, event="manual_pause_verified")
    print(json.dumps(proof, indent=2, sort_keys=True))
    return 0


def show_status(args: argparse.Namespace) -> int:
    record = load_json(args.record.resolve())
    live = instance_by_id(int(record["machine_id"]))
    assert_owned(record, live)
    payload: dict[str, Any] = {"instance": live}
    if record.get("run_id"):
        payload["run"] = run_json(
            ["jl", "run", "status", str(record["run_id"]), "--json"], timeout=90
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory", help="print a non-sensitive inventory")
    inventory_parser.set_defaults(func=lambda _args: print(json.dumps(inventory(), indent=2)) or 0)

    run_parser = subparsers.add_parser(
        "run", help="create, watch, collect, and pause one fresh run"
    )
    run_parser.add_argument("--name", required=True)
    run_parser.add_argument("--record", type=Path, required=True)
    run_parser.add_argument("--target", type=Path, default=Path("."))
    run_parser.add_argument("--script", required=True)
    run_parser.add_argument("--gpu", default="L4")
    run_parser.add_argument("--num-gpus", type=int, default=1)
    run_parser.add_argument("--template", default="pytorch")
    run_parser.add_argument("--python-implementation", choices=("CPython",), required=True)
    run_parser.add_argument("--python-version", choices=("3.11.10",), required=True)
    run_parser.add_argument("--storage", type=int, default=100)
    run_parser.add_argument("--region", choices=("IN1", "IN2", "EU1"))
    run_parser.add_argument("--spot", action="store_true")
    run_parser.add_argument("--requirements", type=Path)
    run_parser.add_argument("--poll-seconds", type=int, default=20)
    run_parser.add_argument("--max-runtime-minutes", type=int, default=240)
    run_parser.add_argument("--artifact", help="remote file/directory to download before pause")
    run_parser.add_argument("--artifact-dest", type=Path, default=Path("remote-artifacts"))
    run_parser.add_argument("--artifact-recursive", action="store_true")
    run_parser.add_argument(
        "--append-jarvis-machine-id",
        action="store_true",
        help="append --jarvis-machine-id <captured-id> to the remote script arguments",
    )
    run_parser.add_argument(
        "--bind-attempt-inventory",
        type=Path,
        help=(
            "target-relative attempt JSON whose machine-ID and preexisting-ID placeholders are "
            "atomically bound after fresh creation and before jl run uploads the target"
        ),
    )
    run_parser.add_argument(
        "--append-bound-attempt-sha256",
        action="store_true",
        help=(
            "append --attempt-preregistration-sha256 with the exact post-binding digest; "
            "valid only with --bind-attempt-inventory"
        ),
    )
    run_parser.add_argument(
        "--retry-owned",
        action="store_true",
        help="resume and retry only the exact project-owned ID in an existing failed record",
    )
    run_parser.add_argument("remote_args", nargs=argparse.REMAINDER)
    run_parser.set_defaults(func=run_fresh)

    cleanup_parser = subparsers.add_parser("cleanup", help="pause the exact ID in an owned record")
    cleanup_parser.add_argument("--record", type=Path, required=True)
    cleanup_parser.add_argument(
        "--recover-migrated-id",
        action="store_true",
        help="recover one exact-name non-preexisting replacement ID before pausing",
    )
    cleanup_parser.set_defaults(func=cleanup_record)

    status_parser = subparsers.add_parser("status", help="inspect the exact ID in an owned record")
    status_parser.add_argument("--record", type=Path, required=True)
    status_parser.set_defaults(func=show_status)

    watchdog_parser = subparsers.add_parser("_deadline-watchdog", help=argparse.SUPPRESS)
    watchdog_parser.add_argument("--record", type=Path, required=True)
    watchdog_parser.add_argument("--deadline-epoch", type=float, required=True)
    watchdog_parser.add_argument("--disarm-path", type=Path, required=True)
    watchdog_parser.add_argument("--proof-path", type=Path, required=True)
    watchdog_parser.set_defaults(func=run_deadline_watchdog)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "poll_seconds") and args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    if hasattr(args, "max_runtime_minutes") and args.max_runtime_minutes < 1:
        parser.error("--max-runtime-minutes must be positive")
    if hasattr(args, "num_gpus") and args.num_gpus < 1:
        parser.error("--num-gpus must be positive")
    if hasattr(args, "storage") and args.storage < 20:
        parser.error("--storage must be at least 20 GB")
    try:
        return int(args.func(args))
    except (SafetyError, RuntimeError, TimeoutError) as error:
        print(f"safe_run: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
