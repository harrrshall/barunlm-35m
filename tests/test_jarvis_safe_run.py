from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_safe_run() -> ModuleType:
    path = ROOT / "infra" / "jarvis" / "safe_run.py"
    spec = importlib.util.spec_from_file_location("barun_safe_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inject_test_protected_resources(
    module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Provide a private denylist fixture without relying on the user's local file."""
    denylist = tmp_path / "test-protected-resources.json"
    module.atomic_write_json(denylist, {"resources": [{"machine_id": 463058}]})
    monkeypatch.setenv(module.PROTECTED_RESOURCES_ENV, str(denylist))
    return denylist


def owned_record(machine_id: int = 999_001) -> dict[str, Any]:
    return {
        "machine_id": machine_id,
        "instance_name": "barun-unit-test-s0",
        "created_by_safe_run": True,
        "preexisting_resources": [{"machine_id": 463058, "name": "Kroda", "status": "Running"}],
    }


def retry_args() -> SimpleNamespace:
    return SimpleNamespace(
        gpu="H200",
        num_gpus=1,
        template="pytorch",
        region="IN2",
        spot=False,
        storage=100,
        target=Path("."),
        script="scripts/run_mobile_probe.py",
        remote_args=["--run-id", "unit-test"],
        max_runtime_minutes=180,
        requirements=None,
        artifact=None,
        artifact_dest=Path("remote-artifacts"),
        artifact_recursive=False,
        append_jarvis_machine_id=False,
        append_bound_attempt_sha256=False,
        bind_attempt_inventory=None,
        poll_seconds=20,
    )


def retry_record(machine_id: int = 999_001) -> dict[str, Any]:
    args = retry_args()
    return {
        **owned_record(machine_id),
        "requested": {
            "gpu": args.gpu,
            "num_gpus": args.num_gpus,
            "template": args.template,
            "region": args.region,
            "spot": args.spot,
            "storage_gb": args.storage,
            "target": str(args.target),
            "script": args.script,
            "requirements": None,
            "setup": "uv pip install -e '.[dev]'",
            "remote_args": args.remote_args,
            "artifact": args.artifact,
            "artifact_dest": str(args.artifact_dest),
            "artifact_recursive": args.artifact_recursive,
            "append_jarvis_machine_id": args.append_jarvis_machine_id,
            "append_bound_attempt_sha256": args.append_bound_attempt_sha256,
            "bind_attempt_inventory": None,
            "max_runtime_minutes": args.max_runtime_minutes,
        },
        "events": [],
    }


def fresh_args(tmp_path: Path, *, append_machine_id: bool = False) -> SimpleNamespace:
    target = tmp_path / "project"
    target.mkdir()
    return SimpleNamespace(
        name="barun-fresh-test-s0",
        record=tmp_path / "jarvis.json",
        target=target,
        script="scripts/run_mobile_probe.py",
        gpu="H200",
        num_gpus=1,
        template="pytorch",
        storage=100,
        region="IN2",
        spot=False,
        requirements=None,
        poll_seconds=20,
        max_runtime_minutes=180,
        artifact=None,
        artifact_dest=tmp_path / "remote-artifacts",
        artifact_recursive=False,
        append_jarvis_machine_id=append_machine_id,
        append_bound_attempt_sha256=False,
        bind_attempt_inventory=None,
        retry_owned=False,
        remote_args=["--run-id", "unit-test"],
    )


def test_name_and_credential_guards() -> None:
    module = load_safe_run()

    module.validate_name("barun-mobile-sft-s0")
    with pytest.raises(module.SafetyError):
        module.validate_name("StrataLM-experiment")
    with pytest.raises(module.SafetyError):
        module.validate_name("kriti-existing")
    with pytest.raises(module.SafetyError):
        module.validate_no_sensitive_arguments(["--api-key=never-put-keys-here"])


def test_private_denylist_can_be_injected_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    denylist = tmp_path / "private-protected-resources.json"
    module.atomic_write_json(
        denylist,
        {"resources": [{"machine_id": 463058}, {"machine_id": "463697"}]},
    )
    monkeypatch.setenv(module.PROTECTED_RESOURCES_ENV, str(denylist))

    assert module.protected_resources_path() == denylist
    assert module.permanent_protected_ids() == {463058, 463697}

    denylist.unlink()
    with pytest.raises(module.SafetyError, match="denylist is unavailable"):
        module.permanent_protected_ids()


def test_attached_run_omits_fresh_instance_lifecycle_flags() -> None:
    module = load_safe_run()
    args = SimpleNamespace(
        target=Path("."),
        script="scripts/run_mobile_probe.py",
        requirements=None,
        remote_args=["--run-id", "unit-test"],
    )

    command = module.build_attached_run_command(args, 999_001)

    assert command[command.index("--on") + 1] == "999001"
    assert not {"--pause", "--destroy", "--keep"}.intersection(command)
    assert "--jarvis-machine-id" not in command
    assert command[command.index("--setup") + 1] == "uv pip install -e '.[dev]'"


def test_explicit_requirements_omit_editable_setup_to_preserve_staged_tree(
    tmp_path: Path,
) -> None:
    module = load_safe_run()
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch==2.13.0\n", encoding="utf-8")
    args = SimpleNamespace(
        target=Path("."),
        script="scripts/run_mobile_temporal_counterfactual.py",
        requirements=requirements,
        remote_args=[],
    )

    command = module.build_attached_run_command(args, 999_001)

    assert "--setup" not in command
    assert command[command.index("--requirements") + 1] == str(requirements)
    assert module.managed_setup_command(args) is None


def test_create_command_preserves_fresh_instance_arguments() -> None:
    module = load_safe_run()
    args = SimpleNamespace(
        gpu="H200",
        template="pytorch",
        storage=240,
        name="barun-create-test-s0",
        num_gpus=2,
        region="IN2",
        spot=True,
    )

    command = module.build_create_command(args)

    assert command == [
        "jl",
        "create",
        "--gpu",
        "H200",
        "--template",
        "pytorch",
        "--storage",
        "240",
        "--name",
        "barun-create-test-s0",
        "--num-gpus",
        "2",
        "--region",
        "IN2",
        "--spot",
        "--yes",
        "--json",
    ]


def test_attached_run_optionally_appends_captured_machine_id() -> None:
    module = load_safe_run()
    args = SimpleNamespace(
        target=Path("."),
        script="scripts/run_mobile_probe.py",
        requirements=None,
        remote_args=["--run-id", "unit-test"],
        append_jarvis_machine_id=True,
    )

    command = module.build_attached_run_command(args, 999_001)

    separator = command.index("--")
    assert command[separator + 1 :] == [
        "--run-id",
        "unit-test",
        "--jarvis-machine-id",
        "999001",
    ]


def test_attached_run_appends_exact_recorded_post_binding_sha() -> None:
    module = load_safe_run()
    digest = "ab" * 32
    args = SimpleNamespace(
        target=Path("."),
        script="scripts/run_mobile_probe.py",
        requirements=None,
        remote_args=["--attempt-preregistration", "attempt-preregistration.json"],
        append_jarvis_machine_id=True,
        bind_attempt_inventory=Path("attempt-preregistration.json"),
        append_bound_attempt_sha256=True,
    )
    record = {"attempt_inventory_binding": {"after_sha256": digest}}

    command = module.build_attached_run_command(args, 999_001, record)

    assert command[command.index("--") + 1 :] == [
        "--attempt-preregistration",
        "attempt-preregistration.json",
        "--jarvis-machine-id",
        "999001",
        "--attempt-preregistration-sha256",
        digest,
    ]
    assert all("token" not in argument.lower() for argument in command)
    assert all("api_key" not in argument.lower() for argument in command)


def test_attached_run_rejects_missing_bound_attempt_sha() -> None:
    module = load_safe_run()
    args = SimpleNamespace(
        target=Path("."),
        script="scripts/run_mobile_probe.py",
        requirements=None,
        remote_args=[],
        bind_attempt_inventory=Path("attempt-preregistration.json"),
        append_bound_attempt_sha256=True,
    )

    with pytest.raises(module.SafetyError, match="bound attempt SHA-256 is unavailable"):
        module.build_attached_run_command(args, 999_001, {})


def test_append_bound_attempt_sha_requires_binding_before_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_safe_run()
    args = fresh_args(tmp_path)
    args.append_bound_attempt_sha256 = True
    monkeypatch.setattr(
        module,
        "inventory",
        lambda: (_ for _ in ()).throw(AssertionError("must fail before Jarvis inventory")),
    )

    with pytest.raises(
        module.SafetyError,
        match="--append-bound-attempt-sha256 requires --bind-attempt-inventory",
    ):
        module.run_fresh(args)


def test_attempt_bound_retry_is_rejected_before_record_or_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_safe_run()
    args = fresh_args(tmp_path)
    args.retry_owned = True
    args.bind_attempt_inventory = Path("attempt-preregistration.json")
    monkeypatch.setattr(
        module,
        "inventory",
        lambda: (_ for _ in ()).throw(AssertionError("must fail before Jarvis inventory")),
    )

    with pytest.raises(
        module.SafetyError,
        match="--retry-owned is forbidden with a fresh-instance attempt binding",
    ):
        module.run_fresh(args)


def test_fresh_artifact_destination_must_not_exist_before_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_safe_run()
    args = fresh_args(tmp_path)
    args.artifact = "/home/barun-artifacts/unit/execution/essential"
    args.artifact_dest.mkdir()
    monkeypatch.setattr(
        module,
        "inventory",
        lambda: (_ for _ in ()).throw(AssertionError("must fail before Jarvis inventory")),
    )

    with pytest.raises(module.SafetyError, match="artifact destination must not already exist"):
        module.run_fresh(args)


def test_record_must_be_outside_uploaded_target_before_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_safe_run()
    args = fresh_args(tmp_path)
    args.record = args.target / "jarvis.json"
    monkeypatch.setattr(
        module,
        "inventory",
        lambda: (_ for _ in ()).throw(AssertionError("must fail before Jarvis inventory")),
    )

    with pytest.raises(module.SafetyError, match="record must be outside the uploaded target"):
        module.run_fresh(args)


def test_attempt_inventory_binding_is_atomic_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_safe_run()
    inject_test_protected_resources(module, tmp_path, monkeypatch)
    target = tmp_path / "stage"
    target.mkdir()
    attempt_path = target / "attempt-preregistration.json"
    module.atomic_write_json(
        attempt_path,
        {
            "schema_version": "unit-test",
            "prelaunch_inventory": {
                "captured_before_project_instance_creation": True,
                "protected_machine_ids": module.PREEXISTING_IDS_PLACEHOLDER,
                "project_machine_id": module.MACHINE_ID_PLACEHOLDER,
                "fresh_project_instance": True,
            },
        },
    )
    record_path = tmp_path / "jarvis.json"
    record = {
        **owned_record(),
        "events": [],
    }

    module.bind_attempt_inventory(
        target=target,
        relative_path=Path("attempt-preregistration.json"),
        record_path=record_path,
        record=record,
        machine_id=999_001,
    )

    bound = module.load_json(attempt_path)
    assert bound["prelaunch_inventory"] == {
        "captured_before_project_instance_creation": True,
        "protected_machine_ids": [463058],
        "project_machine_id": 999_001,
        "fresh_project_instance": True,
    }
    assert record["attempt_inventory_binding"]["bound_before_attached_run"] is True
    assert (
        record["attempt_inventory_binding"]["before_sha256"]
        != record["attempt_inventory_binding"]["after_sha256"]
    )
    assert module.load_json(record_path)["events"][-1]["event"] == "attempt_inventory_bound"


def test_attempt_inventory_binding_rejects_denylist_drift(tmp_path: Path) -> None:
    module = load_safe_run()
    target = tmp_path / "stage"
    target.mkdir()
    attempt_path = target / "attempt-preregistration.json"
    module.atomic_write_json(
        attempt_path,
        {
            "prelaunch_inventory": {
                "captured_before_project_instance_creation": True,
                "protected_machine_ids": [123],
                "project_machine_id": module.MACHINE_ID_PLACEHOLDER,
                "fresh_project_instance": True,
            }
        },
    )

    with pytest.raises(module.SafetyError, match="denylist differs"):
        module.bind_attempt_inventory(
            target=target,
            relative_path=Path("attempt-preregistration.json"),
            record_path=tmp_path / "jarvis.json",
            record={**owned_record(), "events": []},
            machine_id=999_001,
        )


def test_attempt_inventory_binding_includes_absent_durable_protected_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_safe_run()
    denylist = tmp_path / "test-protected-resources.json"
    module.atomic_write_json(
        denylist,
        {"resources": [{"machine_id": 463058}, {"machine_id": 463697}]},
    )
    monkeypatch.setenv(module.PROTECTED_RESOURCES_ENV, str(denylist))
    target = tmp_path / "stage"
    target.mkdir()
    relative_path = Path("attempt-preregistration.json")
    module.atomic_write_json(
        target / relative_path,
        {
            "prelaunch_inventory": {
                "captured_before_project_instance_creation": True,
                "protected_machine_ids": module.PREEXISTING_IDS_PLACEHOLDER,
                "project_machine_id": module.MACHINE_ID_PLACEHOLDER,
                "fresh_project_instance": True,
            }
        },
    )
    record_path = tmp_path / "jarvis.json"
    record = {
        **owned_record(),
        "events": [],
    }

    module.bind_attempt_inventory(
        target=target,
        relative_path=relative_path,
        record_path=record_path,
        record=record,
        machine_id=999_001,
    )

    inventory_block = module.load_json(target / relative_path)["prelaunch_inventory"]
    assert inventory_block["protected_machine_ids"] == [463058, 463697]


def test_sensitive_file_guard_ignores_venv_but_rejects_env(tmp_path: Path) -> None:
    module = load_safe_run()
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "certificate.pem").write_text("not scanned")
    module.validate_no_sensitive_files(tmp_path)

    (tmp_path / ".env.local").write_text("TOKEN=secret")
    with pytest.raises(module.SafetyError, match="sensitive files"):
        module.validate_no_sensitive_files(tmp_path)


def test_ownership_rejects_kroda_preexisting_and_name_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_safe_run()
    inject_test_protected_resources(module, tmp_path, monkeypatch)

    protected = owned_record(machine_id=463058)
    with pytest.raises(module.SafetyError, match="permanently protected"):
        module.assert_owned(
            protected,
            {"machine_id": 463058, "name": "Kroda", "status": "Running"},
        )

    preexisting = owned_record()
    preexisting["preexisting_resources"].append({"machine_id": 999_001})
    with pytest.raises(module.SafetyError, match="existed before"):
        module.assert_owned(
            preexisting,
            {"machine_id": 999_001, "name": "barun-unit-test-s0", "status": "Running"},
        )

    record = owned_record()
    with pytest.raises(module.SafetyError, match="name"):
        module.assert_owned(
            record,
            {"machine_id": 999_001, "name": "unrecognized", "status": "Running"},
        )


def test_pause_owned_mutates_only_the_recorded_exact_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    inject_test_protected_resources(module, tmp_path, monkeypatch)
    record = owned_record()
    state = {"status": "Running"}
    commands: list[list[str]] = []

    def fake_instance(machine_id: int) -> dict[str, Any]:
        assert machine_id == 999_001
        return {
            "machine_id": machine_id,
            "name": "barun-unit-test-s0",
            "status": state["status"],
        }

    def fake_run_json(command: list[str], *, timeout: float | None = None) -> dict[str, Any]:
        del timeout
        commands.append(command)
        assert command == ["jl", "pause", "999001", "--yes", "--json"]
        state["status"] = "Paused"
        return {"success": True, "machine_id": 999_001}

    monkeypatch.setattr(module, "instance_by_id", fake_instance)
    monkeypatch.setattr(module, "run_json", fake_run_json)

    proof = module.pause_owned(record, attempts=1)

    assert proof["status"] == "Paused"
    assert commands == [["jl", "pause", "999001", "--yes", "--json"]]


def test_failed_creation_discovery_requires_one_exact_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    inject_test_protected_resources(module, tmp_path, monkeypatch)
    monkeypatch.setattr(
        module,
        "inventory",
        lambda: [
            {"machine_id": 463058, "name": "Kroda"},
            {"machine_id": 999_002, "name": "barun-wanted-s0"},
            {"machine_id": 999_003, "name": "barun-other-s0"},
        ],
    )

    found = module.discover_failed_creation(name="barun-wanted-s0", preexisting_ids={463058})

    assert found == {"machine_id": 999_002, "name": "barun-wanted-s0"}


def test_fresh_create_persists_id_before_wait_then_runs_attached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    args = fresh_args(tmp_path, append_machine_id=True)
    args.bind_attempt_inventory = Path("attempt-preregistration.json")
    args.append_bound_attempt_sha256 = True
    module.atomic_write_json(
        args.target / args.bind_attempt_inventory,
        {
            "prelaunch_inventory": {
                "captured_before_project_instance_creation": True,
                "protected_machine_ids": module.PREEXISTING_IDS_PLACEHOLDER,
                "project_machine_id": module.MACHINE_ID_PLACEHOLDER,
                "fresh_project_instance": True,
            }
        },
    )
    operations: list[str] = []

    monkeypatch.setattr(
        module,
        "inventory",
        lambda: [{"machine_id": 463058, "name": "Kroda", "status": "Running"}],
    )
    monkeypatch.setattr(module, "permanent_protected_ids", lambda: {463058})

    def fake_wait(record: dict[str, Any]) -> None:
        saved = module.load_json(args.record)
        assert saved["machine_id"] == 999_001
        assert saved["created_by_safe_run"] is True
        assert record["machine_id"] == 999_001
        assert module.load_json(args.target / args.bind_attempt_inventory)[
            "prelaunch_inventory"
        ] == {
            "captured_before_project_instance_creation": True,
            "protected_machine_ids": [463058],
            "project_machine_id": 999_001,
            "fresh_project_instance": True,
        }
        operations.append("wait")

    def fake_run_json(command: list[str], *, timeout: float | None = None) -> dict[str, Any]:
        del timeout
        if command[:2] == ["jl", "create"]:
            operations.append("create")
            return {
                "machine_id": 999_001,
                "name": "barun-fresh-test-s0",
                "status": "Running",
                "gpu_type": "H200",
                "num_gpus": 1,
                "region": "IN2",
                "is_spot": False,
            }
        if command[:2] == ["jl", "get"]:
            operations.append("attest")
            return {
                "machine_id": 999_001,
                "name": "barun-fresh-test-s0",
                "status": "Running",
                "gpu_type": "H200",
                "num_gpus": 1,
                "region": "IN2",
                "is_spot": False,
            }
        if command[:3] == ["jl", "run", "status"]:
            operations.append("status")
            return {
                "run_id": "r_unit",
                "machine_id": 999_001,
                "state": "succeeded",
                "exit_code": 0,
            }
        if command[:3] == ["jl", "run", "logs"]:
            operations.append("logs")
            return {"content": "done", "run_exit_code": 0}
        if command[:2] == ["jl", "run"]:
            operations.append("attached_run")
            assert command[command.index("--on") + 1] == "999001"
            digest = module.load_json(args.record)["attempt_inventory_binding"]["after_sha256"]
            assert command[-4:] == [
                "--jarvis-machine-id",
                "999001",
                "--attempt-preregistration-sha256",
                digest,
            ]
            return {"machine_id": 999_001, "run_id": "r_unit"}
        raise AssertionError(f"unexpected command: {command}")

    def fake_pause(record: dict[str, Any]) -> dict[str, Any]:
        operations.append("pause")
        assert record["machine_id"] == 999_001
        return {
            "machine_id": 999_001,
            "name": "barun-fresh-test-s0",
            "status": "Paused",
        }

    monkeypatch.setattr(module, "wait_for_ssh", fake_wait)
    monkeypatch.setattr(module, "run_json", fake_run_json)
    monkeypatch.setattr(module, "pause_owned", fake_pause)
    monkeypatch.setattr(
        module,
        "arm_deadline_watchdog",
        lambda *_args, **_kwargs: operations.append("watchdog_armed") or {},
    )
    monkeypatch.setattr(
        module,
        "disarm_deadline_watchdog",
        lambda *_args, **_kwargs: operations.append("watchdog_disarmed"),
    )

    assert module.run_fresh(args) == 0

    saved = module.load_json(args.record)
    assert operations == [
        "create",
        "wait",
        "attest",
        "attached_run",
        "watchdog_armed",
        "status",
        "logs",
        "pause",
        "watchdog_disarmed",
    ]
    assert saved["machine_id"] == 999_001
    assert saved["run_id"] == "r_unit"
    assert len(saved["controller_sha256"]) == 64
    assert saved["requested"]["append_jarvis_machine_id"] is True
    assert saved["requested"]["append_bound_attempt_sha256"] is True
    assert saved["requested"]["bind_attempt_inventory"] == "attempt-preregistration.json"
    assert saved["final_instance"]["machine_id"] == 999_001
    events = [item["event"] for item in saved["events"]]
    assert events.index("machine_created") < events.index("machine_ssh_ready")
    assert events.index("attempt_inventory_bound") < events.index("machine_ssh_ready")
    assert events.index("machine_ssh_ready") < events.index("attached_run_requested")
    assert events[-1] == "pause_verified"


def test_fresh_create_requires_explicit_machine_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    args = fresh_args(tmp_path)
    paused = False
    monkeypatch.setattr(
        module,
        "inventory",
        lambda: [{"machine_id": 463058, "name": "Kroda", "status": "Running"}],
    )
    monkeypatch.setattr(module, "permanent_protected_ids", lambda: {463058})
    monkeypatch.setattr(
        module,
        "run_json",
        lambda _command, timeout=None: {"name": "barun-fresh-test-s0"},
    )

    def fake_pause(_record: dict[str, Any]) -> dict[str, Any]:
        nonlocal paused
        paused = True
        raise AssertionError("an unclaimed machine must never be paused")

    monkeypatch.setattr(module, "pause_owned", fake_pause)

    with pytest.raises(module.SafetyError, match="required machine_id"):
        module.run_fresh(args)

    saved = module.load_json(args.record)
    assert "machine_id" not in saved
    assert paused is False
    events = [item["event"] for item in saved["events"]]
    assert "create_response_invalid" in events
    assert "invalid_create_response_recovery_no_candidate" in events


@pytest.mark.parametrize(
    ("create_summary", "expected_reason"),
    [
        ([], "non-object summary"),
        ({"status": "Running"}, "required machine_id"),
        ({"machine_id": None}, "invalid machine_id"),
        ({"machine_id": 1.5}, "invalid machine_id"),
        ({"machine_id": 0}, "invalid machine_id"),
    ],
)
def test_invalid_create_response_recovers_unique_id_only_for_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    create_summary: Any,
    expected_reason: str,
) -> None:
    module = load_safe_run()
    args = fresh_args(tmp_path)
    inventory_calls = 0
    paused_ids: list[int] = []

    def fake_inventory() -> list[dict[str, Any]]:
        nonlocal inventory_calls
        inventory_calls += 1
        resources = [{"machine_id": 463058, "name": "Kroda", "status": "Running"}]
        if inventory_calls > 1:
            resources.append(
                {
                    "machine_id": 999_004,
                    "name": "barun-fresh-test-s0",
                    "status": "Running",
                }
            )
        return resources

    def fake_pause(record: dict[str, Any]) -> dict[str, Any]:
        machine_id = int(record["machine_id"])
        paused_ids.append(machine_id)
        return {
            "machine_id": machine_id,
            "name": "barun-fresh-test-s0",
            "status": "Paused",
        }

    monkeypatch.setattr(module, "inventory", fake_inventory)
    monkeypatch.setattr(module, "permanent_protected_ids", lambda: {463058})
    monkeypatch.setattr(
        module,
        "run_json",
        lambda _command, timeout=None: create_summary,
    )
    monkeypatch.setattr(
        module,
        "wait_for_ssh",
        lambda _record: (_ for _ in ()).throw(
            AssertionError("an invalid create response must not authorize a run")
        ),
    )
    monkeypatch.setattr(module, "pause_owned", fake_pause)

    with pytest.raises(module.SafetyError, match=expected_reason):
        module.run_fresh(args)

    saved = module.load_json(args.record)
    assert saved["machine_id"] == 999_004
    assert saved["creation_recovered_after_invalid_response"] is True
    assert expected_reason in saved["creation_recovery_reason"]
    assert paused_ids == [999_004]
    assert saved["final_instance"]["machine_id"] == 999_004
    events = [item["event"] for item in saved["events"]]
    assert "create_response_invalid" in events
    assert "machine_discovered_after_invalid_create_response" in events
    assert "machine_ssh_ready" not in events
    assert events[-1] == "pause_verified"


def test_invalid_create_response_never_guesses_between_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    args = fresh_args(tmp_path)
    inventory_calls = 0
    paused = False

    def fake_inventory() -> list[dict[str, Any]]:
        nonlocal inventory_calls
        inventory_calls += 1
        resources = [{"machine_id": 463058, "name": "Kroda", "status": "Running"}]
        if inventory_calls > 1:
            resources.extend(
                [
                    {
                        "machine_id": 999_004,
                        "name": "barun-fresh-test-s0",
                        "status": "Running",
                    },
                    {
                        "machine_id": 999_005,
                        "name": "barun-fresh-test-s0",
                        "status": "Running",
                    },
                ]
            )
        return resources

    def fake_pause(_record: dict[str, Any]) -> dict[str, Any]:
        nonlocal paused
        paused = True
        raise AssertionError("ambiguous resources must never be guessed")

    monkeypatch.setattr(module, "inventory", fake_inventory)
    monkeypatch.setattr(module, "permanent_protected_ids", lambda: {463058})
    monkeypatch.setattr(
        module,
        "run_json",
        lambda _command, timeout=None: {"status": "Running"},
    )
    monkeypatch.setattr(module, "pause_owned", fake_pause)

    with pytest.raises(module.SafetyError, match="ambiguous failed creation"):
        module.run_fresh(args)

    saved = module.load_json(args.record)
    assert "machine_id" not in saved
    assert paused is False
    events = [item["event"] for item in saved["events"]]
    assert "create_response_invalid" in events
    assert "machine_discovered_after_invalid_create_response" not in events


def test_create_timeout_recovers_unique_new_id_and_finally_pauses_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    args = fresh_args(tmp_path)
    inventory_calls = 0
    paused_ids: list[int] = []

    def fake_inventory() -> list[dict[str, Any]]:
        nonlocal inventory_calls
        inventory_calls += 1
        resources = [{"machine_id": 463058, "name": "Kroda", "status": "Running"}]
        if inventory_calls > 1:
            resources.append(
                {
                    "machine_id": 999_002,
                    "name": "barun-fresh-test-s0",
                    "status": "Running",
                }
            )
        return resources

    def fake_wait(record: dict[str, Any]) -> None:
        saved = module.load_json(args.record)
        assert saved["machine_id"] == 999_002
        assert record["machine_id"] == 999_002

    def fake_run_json(command: list[str], *, timeout: float | None = None) -> dict[str, Any]:
        if command[:2] == ["jl", "create"]:
            raise subprocess.TimeoutExpired(command, timeout or 1800)
        if command[:2] == ["jl", "get"]:
            return {
                "machine_id": 999_002,
                "name": "barun-fresh-test-s0",
                "status": "Running",
                "gpu_type": "H200",
                "num_gpus": 1,
                "region": "IN2",
                "is_spot": False,
            }
        if command[:2] == ["jl", "run"]:
            raise RuntimeError("attached launch failed")
        raise AssertionError(f"unexpected command: {command}")

    def fake_pause(record: dict[str, Any]) -> dict[str, Any]:
        machine_id = int(record["machine_id"])
        paused_ids.append(machine_id)
        return {
            "machine_id": machine_id,
            "name": "barun-fresh-test-s0",
            "status": "Paused",
        }

    monkeypatch.setattr(module, "inventory", fake_inventory)
    monkeypatch.setattr(module, "permanent_protected_ids", lambda: {463058})
    monkeypatch.setattr(module, "wait_for_ssh", fake_wait)
    monkeypatch.setattr(module, "run_json", fake_run_json)
    monkeypatch.setattr(module, "pause_owned", fake_pause)

    with pytest.raises(RuntimeError, match="attached launch failed"):
        module.run_fresh(args)

    saved = module.load_json(args.record)
    assert saved["machine_id"] == 999_002
    assert saved["creation_recovered_after_cli_failure"] is True
    assert paused_ids == [999_002]
    assert saved["final_instance"]["machine_id"] == 999_002
    events = [item["event"] for item in saved["events"]]
    assert "create_cli_failed" in events
    assert "machine_discovered_after_create_failure" in events
    assert events[-1] == "pause_verified"


def test_create_timeout_never_guesses_between_new_exact_name_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    args = fresh_args(tmp_path)
    inventory_calls = 0

    def fake_inventory() -> list[dict[str, Any]]:
        nonlocal inventory_calls
        inventory_calls += 1
        resources = [{"machine_id": 463058, "name": "Kroda", "status": "Running"}]
        if inventory_calls > 1:
            resources.extend(
                [
                    {
                        "machine_id": 999_002,
                        "name": "barun-fresh-test-s0",
                        "status": "Running",
                    },
                    {
                        "machine_id": 999_003,
                        "name": "barun-fresh-test-s0",
                        "status": "Running",
                    },
                ]
            )
        return resources

    monkeypatch.setattr(module, "inventory", fake_inventory)
    monkeypatch.setattr(module, "permanent_protected_ids", lambda: {463058})
    monkeypatch.setattr(
        module,
        "run_json",
        lambda command, timeout=None: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(command, timeout or 1800)
        ),
    )
    monkeypatch.setattr(
        module,
        "pause_owned",
        lambda _record: (_ for _ in ()).throw(
            AssertionError("ambiguous resources must never be guessed")
        ),
    )

    with pytest.raises(module.SafetyError, match="ambiguous failed creation"):
        module.run_fresh(args)

    saved = module.load_json(args.record)
    assert "machine_id" not in saved
    assert all(
        item["event"] != "machine_discovered_after_create_failure" for item in saved["events"]
    )


def test_successful_resume_requires_an_explicit_machine_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    record_path = tmp_path / "jarvis.json"
    record = retry_record()
    module.atomic_write_json(record_path, record)
    monkeypatch.setattr(module, "run_json", lambda _command, timeout=None: {"status": "Running"})

    with pytest.raises(module.SafetyError, match="required machine_id"):
        module.resume_owned(record_path, record)

    assert record["machine_id"] == 999_001
    assert "machine_id_history" not in record


def test_resume_cli_failure_recovers_unique_replacement_and_finally_pauses_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    args = retry_args()
    record_path = tmp_path / "jarvis.json"
    record = retry_record()
    record["cleanup_error"] = "stale failure from an older attempt"
    module.atomic_write_json(record_path, record)
    paused_ids: list[int] = []

    monkeypatch.setattr(
        module,
        "instance_by_id",
        lambda machine_id: {
            "machine_id": machine_id,
            "name": "barun-unit-test-s0",
            "status": "Paused",
            "gpu_type": "H200",
            "num_gpus": 1,
            "region": "IN2",
            "is_spot": False,
        },
    )
    monkeypatch.setattr(
        module,
        "inventory",
        lambda: [
            {"machine_id": 463058, "name": "Kroda", "status": "Running"},
            {
                "machine_id": 999_002,
                "name": "barun-unit-test-s0",
                "status": "Running",
            },
        ],
    )
    monkeypatch.setattr(module, "permanent_protected_ids", lambda: {463058})
    monkeypatch.setattr(module, "wait_for_ssh", lambda _record: None)

    def fake_run_json(command: list[str], *, timeout: float | None = None) -> dict[str, Any]:
        if command[:2] == ["jl", "resume"]:
            raise subprocess.TimeoutExpired(command, timeout or 300)
        if command[:2] == ["jl", "run"]:
            raise RuntimeError("attached launch failed")
        raise AssertionError(f"unexpected command: {command}")

    def fake_pause(owned: dict[str, Any]) -> dict[str, Any]:
        paused_ids.append(int(owned["machine_id"]))
        return {
            "machine_id": int(owned["machine_id"]),
            "name": owned["instance_name"],
            "status": "Paused",
        }

    monkeypatch.setattr(module, "run_json", fake_run_json)
    monkeypatch.setattr(module, "pause_owned", fake_pause)

    with pytest.raises(RuntimeError, match="attached launch failed"):
        module.run_owned_retry(args, record_path)

    saved = module.load_json(record_path)
    assert saved["machine_id"] == 999_002
    assert saved["machine_id_history"] == [999_001]
    assert paused_ids == [999_002]
    assert saved["final_instance"]["machine_id"] == 999_002
    assert "cleanup_error" not in saved
    events = [item["event"] for item in saved["events"]]
    assert "owned_resume_cli_failed" in events
    assert "owned_resume_id_migrated" in events
    assert "owned_resume_recovered_after_cli_failure" in events
    assert events[-1] == "pause_verified"


def test_owned_retry_rejects_changed_requirements_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    args = retry_args()
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==2.4.6\n", encoding="utf-8")
    args.requirements = requirements
    record = retry_record()
    record["requested"]["requirements"] = module.requirements_receipt(requirements)
    record_path = tmp_path / "jarvis.json"
    module.atomic_write_json(record_path, record)

    requirements.write_text("numpy==2.4.7\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "instance_by_id",
        lambda _machine_id: (_ for _ in ()).throw(
            AssertionError("recipe drift must fail before instance access")
        ),
    )

    with pytest.raises(module.SafetyError, match="retry arguments differ"):
        module.run_owned_retry(args, record_path)


def test_owned_retry_rejects_changed_artifact_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    args = retry_args()
    record = retry_record()
    record_path = tmp_path / "jarvis.json"
    module.atomic_write_json(record_path, record)
    args.artifact = "/home/barun-artifacts/different/essential"
    monkeypatch.setattr(
        module,
        "instance_by_id",
        lambda _machine_id: (_ for _ in ()).throw(
            AssertionError("artifact drift must fail before instance access")
        ),
    )

    with pytest.raises(module.SafetyError, match="retry arguments differ"):
        module.run_owned_retry(args, record_path)


def test_resume_cli_failure_never_guesses_between_replacement_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    record_path = tmp_path / "jarvis.json"
    record = retry_record()
    module.atomic_write_json(record_path, record)
    monkeypatch.setattr(
        module,
        "run_json",
        lambda _command, timeout=None: (_ for _ in ()).throw(RuntimeError("resume failed")),
    )
    monkeypatch.setattr(
        module,
        "inventory",
        lambda: [
            {
                "machine_id": 999_002,
                "name": "barun-unit-test-s0",
                "status": "Running",
            },
            {
                "machine_id": 999_003,
                "name": "barun-unit-test-s0",
                "status": "Running",
            },
        ],
    )
    monkeypatch.setattr(module, "permanent_protected_ids", lambda: {463058})

    with pytest.raises(module.SafetyError, match="found 2 exact-name candidates"):
        module.resume_owned(record_path, record)

    saved = module.load_json(record_path)
    assert saved["machine_id"] == 999_001
    assert "machine_id_history" not in saved
    assert all(item["event"] != "owned_resume_id_migrated" for item in saved["events"])


def test_hardware_mismatch_blocks_upload_but_not_exact_owned_pause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    inject_test_protected_resources(module, tmp_path, monkeypatch)
    args = fresh_args(tmp_path)
    record = owned_record()
    live = {
        "machine_id": 999_001,
        "name": "barun-unit-test-s0",
        "status": "Running",
        "gpu_type": "L4",
        "num_gpus": 1,
        "region": "IN2",
        "is_spot": False,
    }

    module.assert_owned(record, live)
    with pytest.raises(module.SafetyError, match="gpu_type"):
        module.requested_hardware_attestation(args, live)

    state = {"status": "Running"}
    monkeypatch.setattr(
        module,
        "instance_by_id",
        lambda machine_id: {**live, "machine_id": machine_id, "status": state["status"]},
    )

    def fake_pause(command: list[str], *, timeout: float | None = None) -> dict[str, Any]:
        del timeout
        assert command == ["jl", "pause", "999001", "--yes", "--json"]
        state["status"] = "Paused"
        return {"machine_id": 999_001}

    monkeypatch.setattr(module, "run_json", fake_pause)
    assert module.pause_owned(record, attempts=1)["status"] == "Paused"


def test_artifact_collection_is_attempted_even_when_log_collection_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    args = fresh_args(tmp_path)
    args.artifact = "/home/project/essential"
    args.artifact_recursive = True
    record_path = tmp_path / "jarvis.json"
    record = {**owned_record(), "run_id": "r_unit", "events": []}
    module.atomic_write_json(record_path, record)
    operations: list[str] = []

    monkeypatch.setattr(
        module,
        "download_artifact",
        lambda *_args, **_kwargs: operations.append("artifact") or {"success": True},
    )

    def failed_log(command: list[str], *, timeout: float | None = None) -> dict[str, Any]:
        del timeout
        assert command[:3] == ["jl", "run", "logs"]
        operations.append("log")
        raise RuntimeError("log API unavailable")

    monkeypatch.setattr(module, "run_json", failed_log)
    failures = module.collect_run_evidence(record_path, record, args, reason="terminal_run")

    assert operations == ["artifact", "log"]
    assert failures == ["log"]
    saved = module.load_json(record_path)
    assert saved["evidence_collection"]["artifact_status"] == "succeeded"
    assert saved["evidence_collection"]["log_status"] == "failed"


def test_recursive_essential_download_uses_exact_machine_source_and_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    destination = tmp_path / "downloaded-essential"
    observed: list[list[str]] = []

    def fake_run_json(command: list[str], *, timeout: float | None = None) -> dict[str, Any]:
        assert timeout == 1800
        observed.append(command)
        return {"success": True}

    monkeypatch.setattr(module, "run_json", fake_run_json)

    result = module.download_artifact(
        999_001,
        "/home/barun-artifacts/attempt-1/execution/essential",
        destination,
        recursive=True,
    )

    assert result == {"success": True}
    assert observed == [
        [
            "jl",
            "download",
            "999001",
            "/home/barun-artifacts/attempt-1/execution/essential",
            str(destination),
            "--recursive",
            "--json",
        ]
    ]


def test_final_status_is_authoritative_and_failed_zero_cannot_succeed() -> None:
    module = load_safe_run()
    record = {**owned_record(), "run_id": "r_unit"}
    failed = module.filtered_run_snapshot(
        record,
        {
            "run_id": "r_unit",
            "machine_id": 999_001,
            "state": "failed",
            "exit_code": 0,
        },
    )
    succeeded = {**failed, "state": "succeeded", "exit_code": 0}

    assert module.authoritative_exit_code(failed) == 1
    assert module.authoritative_exit_code(succeeded) == 0
    with pytest.raises(module.SafetyError, match="omitted"):
        module.authoritative_exit_code({**succeeded, "exit_code": None})


def test_postlaunch_timeout_stops_then_attempts_artifact_and_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    args = fresh_args(tmp_path)
    args.artifact = "/home/project/essential"
    args.artifact_recursive = True
    record_path = tmp_path / "jarvis.json"
    record = {**owned_record(), "run_id": "r_unit", "events": []}
    module.atomic_write_json(record_path, record)
    operations: list[str] = []

    monkeypatch.setattr(
        module,
        "best_effort_stop_owned_run",
        lambda *_args, **_kwargs: operations.append("stop"),
    )
    monkeypatch.setattr(
        module,
        "download_artifact",
        lambda *_args, **_kwargs: operations.append("artifact") or {"success": True},
    )

    def collect_log(command: list[str], *, timeout: float | None = None) -> dict[str, Any]:
        del timeout
        assert command[:3] == ["jl", "run", "logs"]
        operations.append("log")
        return {"content": "partial", "run_exit_code": None}

    monkeypatch.setattr(module, "run_json", collect_log)
    module.collect_after_postlaunch_error(record_path, record, args, reason="TimeoutError")

    assert operations == ["stop", "artifact", "log"]


def test_detached_watchdog_command_and_worker_remain_exact_owned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_safe_run()
    record_path = tmp_path / "jarvis.json"
    record = {**owned_record(), "run_id": "r_unit", "events": []}
    module.atomic_write_json(record_path, record)
    spawned: dict[str, Any] = {}

    class FakeProcess:
        pid = 765432

    def fake_popen(command, **kwargs):
        spawned["command"] = command
        spawned["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    receipt = module.arm_deadline_watchdog(
        record_path,
        record,
        deadline_epoch=1234567890.0,
    )
    assert spawned["command"][2] == "_deadline-watchdog"
    assert spawned["kwargs"]["start_new_session"] is True
    assert receipt["scope"] == "exact-owned-machine-and-attached-run-only"

    disarm_path = tmp_path / "watchdog.disarm.json"
    proof_path = tmp_path / "watchdog.proof.json"
    stopped: list[tuple[int, str]] = []
    paused: list[int] = []
    monkeypatch.setattr(module, "WATCHDOG_PAUSE_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(
        module,
        "stop_owned_run",
        lambda current: (
            stopped.append((current["machine_id"], current["run_id"]))
            or {"action": "stop_requested"}
        ),
    )
    monkeypatch.setattr(
        module,
        "pause_owned",
        lambda current: (
            paused.append(current["machine_id"])
            or {
                "machine_id": current["machine_id"],
                "name": current["instance_name"],
                "status": "Paused",
            }
        ),
    )
    worker_args = SimpleNamespace(
        record=record_path,
        disarm_path=disarm_path,
        proof_path=proof_path,
        deadline_epoch=0.0,
    )
    assert module.run_deadline_watchdog(worker_args) == 0
    assert stopped == [(999_001, "r_unit")]
    assert paused == [999_001]
    assert module.load_json(proof_path)["status"] == "pause_verified"
