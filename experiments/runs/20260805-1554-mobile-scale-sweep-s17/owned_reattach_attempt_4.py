"""Operational reattach for attempt-4 after local controller death during jl upload.

Uses the already-created exact machine ID from jarvis-attempt-4.json. Does not create
a GPU. Does not mutate protected IDs. Attempt inventory is already bound on the stage.
safe_run CLI forbids --retry-owned together with --bind-attempt-inventory; this driver
performs the owned resume→attest→jl-run→monitor→download→pause path for that one case.
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
RECORD = RUN_DIR / "jarvis-attempt-4.json"
STAGE = Path((RUN_DIR / "stage-attempt-4.path").read_text(encoding="utf-8").strip())


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
        poll_seconds=120,
        max_runtime_minutes=req["max_runtime_minutes"],
        artifact=req["artifact"],
        artifact_dest=Path(req["artifact_dest"]),
        artifact_recursive=req["artifact_recursive"],
        append_jarvis_machine_id=req["append_jarvis_machine_id"],
        append_bound_attempt_sha256=req.get("append_bound_attempt_sha256", False),
        bind_attempt_inventory=None,  # already bound; do not rebind
        remote_args=list(req["remote_args"]),
        retry_owned=True,
    )


def main() -> int:
    record_path = RECORD.resolve()
    record = sr.load_json(record_path)
    if not isinstance(record, dict) or "machine_id" not in record:
        raise SystemExit("record missing machine_id")
    if int(record["machine_id"]) != 465183 and 465183 not in {
        int(x) for x in record.get("machine_id_history", []) if x is not None
    }:
        # Allow resume-migrated replacement of the original owned ID only.
        if "created_by_safe_run" not in record or not record.get("created_by_safe_run"):
            raise SystemExit("refusing to operate on non-owned record")
    if Path(record["requested"]["target"]).resolve() != STAGE.resolve():
        raise SystemExit("stage path drifted from frozen record target")
    # Confirm attempt remains bound to the recorded machine (or placeholder only if rebound later).
    attempt = json.loads((STAGE / "attempt-preregistration.json").read_text(encoding="utf-8"))
    bound = attempt["prelaunch_inventory"]["project_machine_id"]
    if bound not in (465183, record["machine_id"]):
        raise SystemExit(f"attempt binding machine mismatch: {bound}")
    if attempt["compute"]["template"] != "axolotl":
        raise SystemExit("attempt compute template is not axolotl")

    args = build_args(record)
    if args.template != "axolotl":
        raise SystemExit("refusing non-axolotl reattach")
    if sr.requirements_receipt(args.requirements) != record["requested"]["requirements"]:
        raise SystemExit("requirements bytes drifted")

    live = sr.instance_by_id(int(record["machine_id"]))
    sr.assert_owned(record, live)
    if str(live.get("status", "")).lower() != "paused":
        raise SystemExit(f"expected Paused, got {live.get('status')}")

    primary_error: BaseException | None = None

    def on_signal(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    old_handlers = {
        signum: signal.signal(signum, on_signal) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    machine_id = int(record["machine_id"])
    try:
        previous_run_id = record.pop("run_id", None)
        if previous_run_id is not None:
            record.setdefault("run_id_history", []).append(str(previous_run_id))
        sr.persist_event(
            record_path,
            record,
            "owned_reattach_requested",
            machine_id=machine_id,
            reason="controller_death_during_jl_run_upload",
        )
        machine_id = sr.resume_owned(record_path, record)
        # If resume migrated the ID, re-bind attempt inventory to the new owned ID.
        if machine_id != bound:
            sr.bind_attempt_inventory(
                target=args.target,
                relative_path=Path("attempt-preregistration.json"),
                record_path=record_path,
                record=record,
                machine_id=machine_id,
            )
        sr.persist_event(record_path, record, "owned_reattach_resumed", machine_id=machine_id)
        sr.wait_for_ssh(record)
        sr.persist_event(record_path, record, "owned_reattach_ssh_ready", machine_id=machine_id)
        sr.attest_live_hardware(record_path, record, args)
        sr.attest_live_python_runtime(record_path, record, args)
        # Fail closed if attestation drifted off axolotl/3.11.10
        py = record.get("python_runtime_attestation", {}).get("observed", {})
        hw = record.get("hardware_attestation", {}).get("observed", {})
        if hw.get("template") != "axolotl" or py.get("version") != "3.11.10":
            raise RuntimeError(f"reattach attestation failed: hw={hw} py={py}")

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
        print(f"owned_reattach: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
