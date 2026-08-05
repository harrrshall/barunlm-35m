"""Continue attempt-6 on the already-created exact H200 after local controller death.

Does not create a GPU. Operates only on jarvis-attempt-6.json machine_id.
Attempt inventory is still unbound; bind after axolotl/3.11.10 attestation.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "infra" / "jarvis"))
import safe_run as sr  # noqa: E402

RUN_DIR = Path(__file__).resolve().parent
RECORD = RUN_DIR / "jarvis-attempt-6.json"
STAGE = Path((RUN_DIR / "stage-attempt-6.path").read_text(encoding="utf-8").strip())
EXPECTED_MACHINE_ID = 465257


def build_args(record: dict) -> argparse.Namespace:
    req = record["requested"]
    return argparse.Namespace(
        name=record["instance_name"],
        record=RECORD,
        target=Path(req["target"]),
        script=req["script"],
        gpu=req["gpu"],
        num_gpus=req["num_gpus"],
        template=req["template"],
        python_implementation=req["python_implementation"],
        python_version=req["python_version"],
        storage=req["storage_gb"],
        region=req["region"],
        spot=req["spot"],
        requirements=Path(req["requirements"]["path"]),
        isolated_project_venv=True,
        poll_seconds=120,
        max_runtime_minutes=req["max_runtime_minutes"],
        artifact=req["artifact"],
        artifact_dest=Path(req["artifact_dest"]),
        artifact_recursive=req["artifact_recursive"],
        append_jarvis_machine_id=req["append_jarvis_machine_id"],
        append_bound_attempt_sha256=req.get("append_bound_attempt_sha256", False),
        bind_attempt_inventory=Path("attempt-preregistration.json"),
        remote_args=list(req["remote_args"]),
        retry_owned=False,
    )


def main() -> int:
    record_path = RECORD.resolve()
    record = sr.load_json(record_path)
    if not isinstance(record, dict) or "machine_id" not in record:
        raise SystemExit("record missing machine_id")
    machine_id = int(record["machine_id"])
    if machine_id != EXPECTED_MACHINE_ID:
        raise SystemExit(f"refusing unexpected machine_id {machine_id}")
    if not record.get("created_by_safe_run"):
        raise SystemExit("refusing non-owned record")
    if Path(record["requested"]["target"]).resolve() != STAGE.resolve():
        raise SystemExit("stage path drifted from frozen record target")
    attempt = json.loads((STAGE / "attempt-preregistration.json").read_text(encoding="utf-8"))
    if attempt["prelaunch_inventory"] != {
        "captured_before_project_instance_creation": True,
        "fresh_project_instance": True,
        "project_machine_id": "__SAFE_RUN_MACHINE_ID__",
        "protected_machine_ids": "__SAFE_RUN_PREEXISTING_IDS__",
    }:
        raise SystemExit("attempt inventory is not unbound; refusing unsafe continue")
    if attempt["compute"]["template"] != "axolotl":
        raise SystemExit("attempt compute template is not axolotl")
    if attempt.get("config_sha256") != "0885b32f14751e77539f0bf58cae6f89b7c72e1d80c1b8a6d3fe61cff9c55540":
        raise SystemExit("attempt config sha drifted")

    args = build_args(record)
    if args.template != "axolotl":
        raise SystemExit("refusing non-axolotl continue")
    if not args.isolated_project_venv:
        raise SystemExit("refusing continue without isolated-project-venv")
    setup = sr.managed_setup_command(args)
    if setup != record["requested"]["setup"]:
        raise SystemExit(f"setup drift: {setup!r} vs {record['requested']['setup']!r}")
    if sr.requirements_receipt(args.requirements) != record["requested"]["requirements"]:
        raise SystemExit("requirements bytes drifted")

    live = sr.instance_by_id(machine_id)
    sr.assert_owned(record, live)
    status = str(live.get("status", "")).lower()
    if status == "paused":
        sr.persist_event(record_path, record, "owned_continue_resume_needed", machine_id=machine_id)
        machine_id = sr.resume_owned(record_path, record)
    elif status != "running":
        raise SystemExit(f"unexpected status {live.get('status')}")

    primary_error: BaseException | None = None

    def on_signal(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    old_handlers = {
        signum: signal.signal(signum, on_signal) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        sr.persist_event(
            record_path,
            record,
            "owned_continue_requested",
            machine_id=machine_id,
            reason="controller_death_after_create_before_ssh_ready",
        )
        sr.wait_for_ssh(record)
        sr.persist_event(record_path, record, "owned_continue_ssh_ready", machine_id=machine_id)
        sr.attest_live_hardware(record_path, record, args)
        sr.attest_live_python_runtime(record_path, record, args)
        py = record.get("python_runtime_attestation", {}).get("observed", {})
        hw = record.get("hardware_attestation", {}).get("observed", {})
        if hw.get("template") != "axolotl" or py.get("version") != "3.11.10":
            raise RuntimeError(f"continue attestation failed: hw={hw} py={py}")

        # Bind unbound attempt inventory to this exact owned ID before upload.
        prebind = sr.validate_unbound_attempt_contract(
            target=args.target,
            relative_path=args.bind_attempt_inventory,
            args=args,
        )
        expected_sha = record["attempt_contract_precreate_validation"]["sha256"]
        if prebind["sha256"] != expected_sha:
            raise RuntimeError("attempt preregistration changed after pre-create validation")
        record["attempt_contract_prebind_validation"] = prebind
        sr.bind_attempt_inventory(
            target=args.target,
            relative_path=args.bind_attempt_inventory,
            record_path=record_path,
            record=record,
            machine_id=machine_id,
        )

        if sr.requirements_receipt(args.requirements) != record["requested"]["requirements"]:
            raise RuntimeError("requirements changed after bind")

        sr.persist_event(record_path, record, "attached_run_requested", machine_id=machine_id)
        summary = sr.run_json(
            sr.build_attached_run_command(args, machine_id, record), timeout=1800
        )
        if not isinstance(summary, dict):
            raise RuntimeError("jl run returned non-object")
        if int(summary["machine_id"]) != machine_id:
            raise RuntimeError("jl run returned different machine_id")
        record.update({"run_id": str(summary["run_id"]), "created_by_safe_run": True})
        record["launch_summary"] = {
            "machine_id": machine_id,
            "run_id": str(summary["run_id"]),
            "remote_log": summary.get("remote_log"),
            "remote_exit_code": summary.get("remote_exit_code"),
            "target_kind": summary.get("target_kind"),
            "instance_origin": summary.get("instance_origin"),
            "lifecycle_policy": summary.get("lifecycle_policy"),
            "owned_continue": True,
        }
        sr.persist_event(record_path, record, "run_started", machine_id=machine_id)
        print(
            json.dumps(
                {"event": "run_started", "machine_id": machine_id, "run_id": record["run_id"]}
            ),
            flush=True,
        )
        sr.arm_deadline_watchdog(
            record_path,
            record,
            deadline_epoch=time.time() + args.max_runtime_minutes * 60,
        )
        _, exit_code = sr.monitor_owned_run(
            record_path,
            record,
            max_runtime_minutes=args.max_runtime_minutes,
            poll_seconds=args.poll_seconds,
        )
        failures = sr.collect_run_evidence(
            record_path, record, args, reason="terminal_run"
        )
        if failures:
            raise RuntimeError(f"evidence collection failed: {failures}")
        return int(exit_code)
    except BaseException as error:
        primary_error = error
        sr.persist_event(record_path, record, "controller_error", error=repr(error))
        try:
            sr.collect_after_postlaunch_error(
                record_path, record, args, reason=type(error).__name__
            )
        except Exception as collect_error:  # noqa: BLE001
            record["collect_after_error"] = repr(collect_error)
            sr.atomic_write_json(record_path, record)
        raise
    finally:
        try:
            try:
                proof = sr.pause_owned(record)
                sr.persist_verified_pause(
                    record_path, record, proof, event="pause_verified"
                )
                sr.disarm_deadline_watchdog(record_path, record)
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
            except Exception as cleanup_error:  # noqa: BLE001
                record["cleanup_error"] = repr(cleanup_error)
                sr.atomic_write_json(record_path, record)
                print(str(cleanup_error), file=sys.stderr, flush=True)
                if primary_error is None:
                    raise
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        print(f"owned_continue: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
