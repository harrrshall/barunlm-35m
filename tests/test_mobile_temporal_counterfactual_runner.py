from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_barun_mobile_temporal_runner",
        ROOT / "scripts" / "run_mobile_temporal_counterfactual.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load temporal counterfactual runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


@dataclass(frozen=True)
class FrozenFixture:
    config_path: Path
    view_audit_path: Path
    output_dir: Path
    source_rows: int
    source_sha256: str
    source_membership_sha256: str
    shadow_rows: dict[str, int]
    shadow_memberships: dict[str, str]


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        for row in rows
    )


def _write(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _membership(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def _target(example_id: str, month: int = 6) -> str:
    return json.dumps(
        {
            "calls": [
                {
                    "args": {
                        "datetime": f"2025-{month:02d}-06T14:00:00",
                        "title": f"Meeting {example_id}",
                    },
                    "tool": "create_calendar_event",
                }
            ],
            "decision": "CALL",
            "mode": "SINGLE",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _row(
    example_id: str,
    *,
    derived_split: str,
    now: str = "2025-06-04T15:29:23",
    target: str | None = None,
    source_id: str | None = None,
    view_kind: str | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "call_names": ["create_calendar_event"],
        "cluster_id": f"cluster-{example_id}",
        "derived_split": derived_split,
        "family_id": f"family-{example_id}",
        "source_split": "train",
    }
    if source_id is not None:
        metadata["cluster_id"] = f"cluster-{source_id}"
        metadata["family_id"] = f"family-{source_id}"
        metadata["mobile_temporal_view"] = {
            "kind": view_kind,
            "schema_version": "barun-mobile-temporal-view-v1",
            "source_id": source_id,
        }
    return {
        "id": example_id,
        "metadata": metadata,
        "prompt": (
            "<bos><system>\nACTION_IR_V1\n"
            f"NOW {now}\nTOOLS\n"
            "create_calendar_event(title:string!, datetime:string!): Creates an event.\n"
            f"<user>\nSchedule {source_id or example_id}.\n<assistant>\n"
        ),
        "schema_version": "barun-sft-example-v1",
        "target": target if target is not None else _target(source_id or example_id),
    }


def _build_frozen_fixture(tmp_path: Path, *, safe_variants: int = 1_000) -> FrozenFixture:
    materialized = tmp_path / "materialized-screening-v1"
    materialized.mkdir()
    construction = [
        _row(f"source-{index:04d}", derived_split="train") for index in range(safe_variants)
    ]
    selection = [_row("selection-0000", derived_split="dev")]
    confirmation = [_row("confirmation-0000", derived_split="confirmation")]
    repeats = [
        _row(
            f"source-{index:04d}--temporal-repeat-v1",
            derived_split="train",
            source_id=f"source-{index:04d}",
            view_kind="repeat_control",
        )
        for index in range(safe_variants)
    ]
    counterfactuals = [
        _row(
            f"source-{index:04d}--mbcf-v1-deadbeef{index:04d}",
            derived_split="train",
            now="2025-05-31T15:29:23",
            source_id=f"source-{index:04d}",
            view_kind="month_boundary_counterfactual",
        )
        for index in range(safe_variants)
    ]
    receipts = [
        {
            "invariants": {
                "cross_month": True,
                "non_calendar_calls_unchanged": True,
                "source_user_text_unchanged": True,
                "target_byte_identical_for_absolute": True,
                "target_timedelta_preserved_for_relative": False,
            },
            "schema_version": "barun-mobile-temporal-transform-receipt-v1",
            "source_id": f"source-{index:04d}",
            "variant_id": f"source-{index:04d}--mbcf-v1-deadbeef{index:04d}",
        }
        for index in range(safe_variants)
    ]
    artifacts = {
        "construction": construction,
        "selection": selection,
        "confirmation": confirmation,
        "repeat": construction + repeats,
        "counterfactual": construction + counterfactuals,
        "receipts": receipts,
    }
    hashes = {
        name: _write(
            materialized / runner.EXPECTED_ARTIFACT_FILENAMES[name],
            _jsonl_bytes(rows),
        )
        for name, rows in artifacts.items()
    }
    rows = {name: len(records) for name, records in artifacts.items()}
    shadow_rows = {
        "construction_train": len(construction),
        "selection": len(selection),
        "confirmation": len(confirmation),
    }
    shadow_memberships = {
        "construction_train": _membership([str(row["id"]) for row in construction]),
        "selection": _membership([str(row["id"]) for row in selection]),
        "confirmation": _membership([str(row["id"]) for row in confirmation]),
    }
    source_rows = sum(shadow_rows.values())
    source_sha256 = "1" * 64
    source_membership = _membership(
        [str(row["id"]) for row in construction + selection + confirmation]
    )
    shadow_audit = {
        "development_rows_read": 0,
        "official_evaluation_rows_read": 0,
        "overlap": {
            "cluster_id": 0,
            "exact_prompt_target": 0,
            "example_id": 0,
            "family_id": 0,
        },
        "roles": {
            role: {
                "calendar": {"cross_month": 0, "rows": count, "same_month": count},
                "manifest_sha256": hashes[
                    {
                        "construction_train": "construction",
                        "selection": "selection",
                        "confirmation": "confirmation",
                    }[role]
                ],
                "membership_sha256": shadow_memberships[role],
                "rows": count,
            }
            for role, count in shadow_rows.items()
        },
        "schema_version": "barun-mobile-temporal-shadow-audit-v1",
        "source": {
            "membership_sha256": source_membership,
            "rows": source_rows,
            "sha256": source_sha256,
        },
        "split_policy": {"version": "barun-mobile-temporal-shadow-split-v1"},
    }
    shadow_audit_sha256 = _write(materialized / "shadow-audit.json", _json_bytes(shadow_audit))
    safe_ids = [f"source-{index:04d}" for index in range(safe_variants)]
    safe_membership = _membership(safe_ids)
    construction_steps = (len(construction) + 62) // 63
    paired_steps = (len(construction) + safe_variants + 62) // 63
    cross_fraction = safe_variants / (len(construction) + safe_variants)
    view_audit = {
        "calendar_presentations": {
            "added_cross_month": safe_variants,
            "minimum_fraction": 0.45,
            "original_cross_month": 0,
            "original_same_month": len(construction),
            "post_cross_month": safe_variants,
            "post_cross_month_fraction": cross_fraction,
            "post_total": len(construction) + safe_variants,
        },
        "eligibility": {
            "counts": {"absolute_explicit": safe_variants},
            "minimum_required": 1_000,
            "safe_source_membership_sha256": safe_membership,
            "safe_variant_rows": safe_variants,
        },
        "leakage": {
            "confirmation_source_rows_used_for_variants": 0,
            "development_rows_read": 0,
            "official_evaluation_rows_read": 0,
            "selection_source_rows_used_for_variants": 0,
        },
        "matched_control": {
            "same_added_rows": True,
            "same_optimizer_steps": True,
            "same_source_ids": True,
            "source_membership_sha256": safe_membership,
        },
        "receipts": {
            "filename": "transform-receipts.jsonl",
            "rows": safe_variants,
            "sha256": hashes["receipts"],
        },
        "schema_version": "barun-mobile-temporal-view-audit-v1",
        "tokenizer": dict(runner.EXPECTED_TOKENIZER),
        "views": {
            "counterfactual": {
                "added_rows": safe_variants,
                "filename": "counterfactual-train.jsonl",
                "optimizer_steps_at_batch_63": paired_steps,
                "rows": rows["counterfactual"],
                "sha256": hashes["counterfactual"],
            },
            "repeat": {
                "added_rows": safe_variants,
                "filename": "repeat-train.jsonl",
                "optimizer_steps_at_batch_63": paired_steps,
                "rows": rows["repeat"],
                "sha256": hashes["repeat"],
            },
            "standard": {
                "filename": "construction-train.jsonl",
                "optimizer_steps_at_batch_63": construction_steps,
                "rows": rows["construction"],
                "sha256": hashes["construction"],
            },
        },
    }
    view_audit_path = materialized / "view-audit.json"
    view_audit_sha256 = _write(view_audit_path, _json_bytes(view_audit))

    full_dir = tmp_path / "materialized-full-refit-v1"
    full_dir.mkdir()
    full_source = construction + [
        _row("selection-0000", derived_split="train"),
        _row("confirmation-0000", derived_split="train"),
    ]
    full_counterfactuals = counterfactuals
    full_receipts = receipts
    full_train = full_source + full_counterfactuals
    full_train_sha256 = _write(
        full_dir / "full-counterfactual-train.jsonl", _jsonl_bytes(full_train)
    )
    full_receipts_sha256 = _write(
        full_dir / "full-transform-receipts.jsonl", _jsonl_bytes(full_receipts)
    )
    full_steps = (len(full_train) + 62) // 63
    full_calendar_rows = len(full_source) + safe_variants
    full_cross_fraction = safe_variants / full_calendar_rows
    full_audit = {
        "calendar_presentations": {
            "added_cross_month": safe_variants,
            "minimum_fraction": 0.45,
            "original_cross_month": 0,
            "original_same_month": len(full_source),
            "post_cross_month": safe_variants,
            "post_cross_month_fraction": full_cross_fraction,
            "post_total": full_calendar_rows,
        },
        "development_rows_read": 0,
        "official_evaluation_rows_read": 0,
        "output": {
            "filename": "full-counterfactual-train.jsonl",
            "optimizer_steps_at_batch_63": full_steps,
            "rows": len(full_train),
            "sha256": full_train_sha256,
        },
        "receipts": {
            "filename": "full-transform-receipts.jsonl",
            "rows": safe_variants,
            "sha256": full_receipts_sha256,
        },
        "schema_version": "barun-mobile-temporal-full-view-audit-v1",
        "source": {
            "membership_sha256": source_membership,
            "rows": source_rows,
            "sha256": source_sha256,
        },
        "tokenizer": dict(runner.EXPECTED_TOKENIZER),
        "variants": {
            "by_class": {"absolute_explicit": safe_variants},
            "minimum_required": 1_000,
            "rows": safe_variants,
            "source_membership_sha256": safe_membership,
        },
    }
    full_audit_sha256 = _write(full_dir / "full-view-audit.json", _json_bytes(full_audit))
    implementation_path = tmp_path / "implementation.py"
    implementation_sha256 = _write(implementation_path, b"# frozen implementation\n")
    config = {
        "arms": list(runner.EXPECTED_ARMS),
        "checkpoint_policy": "final_only",
        "conditional_full_refit": {
            "audit_sha256": full_audit_sha256,
            "optimizer_steps": full_steps,
            "receipts_sha256": full_receipts_sha256,
            "rows": len(full_train),
            "safe_source_membership_sha256": safe_membership,
            "seed": 17,
            "source_rows": source_rows,
            "train_sha256": full_train_sha256,
            "variant_rows": safe_variants,
        },
        "conditional_full_refit_budget": 1,
        "implementation_contract": {
            "files": {"implementation.py": implementation_sha256},
        },
        "materialization": {
            "artifact_rows": rows,
            "artifact_sha256": hashes,
            "shadow_audit_sha256": shadow_audit_sha256,
            "view_audit_sha256": view_audit_sha256,
        },
        "official_evaluation": dict(runner.EXPECTED_OFFICIAL_EVALUATION),
        "optimization": dict(runner.EXPECTED_OPTIMIZATION),
        "retry_policy": {
            "maximum_byte_identical_infrastructure_retries": 2,
            "retry_after_any_held_out_signal": (
                "forbidden after any usable held-out signal, including a selection diagnostic "
                "loss, selection/confirmation/terminal prediction, or action-evaluation score; "
                "a retry is allowed only when the previous attempt produced no usable held-out "
                "signal of any kind"
            ),
            "scientific_change_on_retry": "forbidden",
            "retry_execution_policy": (
                "fresh_instance_new_output_directory_new_attempt_id_incremented_ordinal_and_"
                "prior_zero_signal_receipt_only; safe_run_owned_resume_retry_forbidden"
            ),
        },
        "remote_validation_contract": {
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
                "template": runner.EXPECTED_JARVIS_TEMPLATE,
                "python_implementation": runner.EXPECTED_PYTHON_IMPLEMENTATION,
                "python_version": runner.EXPECTED_PYTHON_VERSION,
                "repository_root_virtual_environment": ".venv",
                "virtual_environment_must_be_active": True,
                "virtual_environment_in_scientific_tree": False,
                "preupload_runtime_attestation_required": True,
            },
            "required_exit_code": 0,
            "parent_process_torch_imported_before_validation": False,
            "real_terminal_manifest_path_or_contents_passed_in_test_argv_or_environment": False,
            "test_file_allowlist_has_no_real_terminal_reference": True,
        },
        "run_id": "20260803-2350-mobile-temporal-counterfactual-s17",
        "schema_version": runner.CONFIG_SCHEMA_VERSION,
        "screening_fit_budget": 9,
        "seeds": list(runner.EXPECTED_SEEDS),
        "thresholds": dict(runner.EXPECTED_THRESHOLDS),
    }
    config_path = tmp_path / "config.json"
    _write(config_path, _json_bytes(config))
    return FrozenFixture(
        config_path=config_path,
        view_audit_path=view_audit_path,
        output_dir=tmp_path / "plan",
        source_rows=source_rows,
        source_sha256=source_sha256,
        source_membership_sha256=source_membership,
        shadow_rows=shadow_rows,
        shadow_memberships=shadow_memberships,
    )


def _write_bound_launch_provenance(
    fixture: FrozenFixture, *, machine_id: int = 987654
) -> tuple[Path, str, Path]:
    root = fixture.config_path.parent
    terminal = root / "reused-756.jsonl"
    terminal.write_text('{"sealed":"not-read-during-preflight"}\n', encoding="utf-8")
    config = json.loads(fixture.config_path.read_text(encoding="utf-8"))
    exact_exclusions = sorted(
        {
            runner.ATTEMPT_PREREGISTRATION_RELATIVE_PATH,
            runner.SOURCE_SNAPSHOT_RELATIVE_PATH,
            terminal.name,
        }
    )
    tree = runner._recompute_source_tree(root, excluded_exact_paths=exact_exclusions)
    snapshot = {
        "schema_version": runner.SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "run_id": config["run_id"],
        "git": {
            "repository": "https://github.com/harrrshall/kimi_100M",
            "commit": "a" * 40,
            "branch": "barun-mobile-temporal-test",
        },
        "content_tree": {
            "algorithm": runner.SOURCE_TREE_ALGORITHM,
            **tree,
            "excluded_exact_paths": exact_exclusions,
            "excluded_directory_names": list(runner.SOURCE_TREE_EXCLUDED_DIRECTORY_NAMES),
            "excluded_directory_suffixes": list(runner.SOURCE_TREE_EXCLUDED_DIRECTORY_SUFFIXES),
            "excluded_file_names": list(runner.SOURCE_TREE_EXCLUDED_FILE_NAMES),
            "excluded_file_suffixes": list(runner.SOURCE_TREE_EXCLUDED_FILE_SUFFIXES),
        },
        "frozen_files": config["implementation_contract"]["files"],
        "staging_policy": {
            "credentials_present": False,
            "official_mobile_evaluation_present": False,
            "symlinks_present_before_launch": False,
        },
    }
    snapshot_path = root / runner.SOURCE_SNAPSHOT_RELATIVE_PATH
    snapshot_sha256 = _write(snapshot_path, _json_bytes(snapshot))
    screening_path = fixture.view_audit_path.relative_to(root).as_posix()
    full_path = root / "materialized-full-refit-v1" / "full-view-audit.json"
    attempt = {
        "schema_version": runner.ATTEMPT_PREREGISTRATION_SCHEMA_VERSION,
        "status": runner.ATTEMPT_STATUS,
        "run_id": config["run_id"],
        "attempt_id": f"{config['run_id']}-attempt-1",
        "attempt_ordinal": 1,
        "prior_attempts": [],
        "durable_protected_machine_ids": sorted(runner.KNOWN_PROTECTED_JARVIS_IDS),
        "scientific_config": {
            "path": fixture.config_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(fixture.config_path.read_bytes()).hexdigest(),
        },
        "source_snapshot": {
            "path": runner.SOURCE_SNAPSHOT_RELATIVE_PATH,
            "sha256": snapshot_sha256,
        },
        "materialization": {
            "screening_view_audit_path": screening_path,
            "screening_view_audit_sha256": hashlib.sha256(
                fixture.view_audit_path.read_bytes()
            ).hexdigest(),
            "full_view_audit_path": full_path.relative_to(root).as_posix(),
            "full_view_audit_sha256": hashlib.sha256(full_path.read_bytes()).hexdigest(),
        },
        "terminal_population": {
            "path": terminal.name,
            "sha256": runner.PINNED_REUSED_756_SHA256,
            "rows": runner.PINNED_REUSED_756_ROWS,
            "access": "only_after_confirmation_pass_and_full_refit",
        },
        "prelaunch_inventory": {
            "captured_before_project_instance_creation": True,
            "protected_machine_ids": sorted(runner.KNOWN_PROTECTED_JARVIS_IDS),
            "project_machine_id": machine_id,
            "fresh_project_instance": True,
        },
        "compute": {
            "provider": "JarvisLabs",
            "template": runner.EXPECTED_JARVIS_TEMPLATE,
            "python_implementation": runner.EXPECTED_PYTHON_IMPLEMENTATION,
            "python_version": runner.EXPECTED_PYTHON_VERSION,
            "gpu": "H200",
            "num_gpus": 1,
            "region": "IN2",
            "is_spot": False,
            "max_gpu_job_minutes": 30,
        },
        "retry_lock": {
            name: config["retry_policy"][name]
            for name in ("retry_after_any_held_out_signal", "retry_execution_policy")
        },
        "official_evaluation": {
            "policy": "forbidden",
            "rows": 961,
            "rows_present": 0,
            "rows_read": 0,
        },
    }
    attempt_path = root / runner.ATTEMPT_PREREGISTRATION_RELATIVE_PATH
    attempt_sha256 = _write(attempt_path, _json_bytes(attempt))
    return attempt_path, attempt_sha256, terminal


def _install_active_runtime_root_venv(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    venv = root / ".venv"
    executable = venv / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"test interpreter marker\n")
    (venv / "pyvenv.cfg").write_text("include-system-site-packages = true\n", encoding="utf-8")
    base_prefix = root.parent / f"{root.name}-base-python"
    base_prefix.mkdir()
    monkeypatch.setattr(runner.sys, "prefix", str(venv))
    monkeypatch.setattr(runner.sys, "exec_prefix", str(venv))
    monkeypatch.setattr(runner.sys, "base_prefix", str(base_prefix))
    monkeypatch.setattr(runner.sys, "executable", str(executable))
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    return venv


def _patch_expected_runtime_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.platform, "python_implementation", lambda: runner.EXPECTED_PYTHON_IMPLEMENTATION
    )
    monkeypatch.setattr(runner.platform, "python_version", lambda: runner.EXPECTED_PYTHON_VERSION)


def _write_retry_launch_provenance(
    fixture: FrozenFixture,
    *,
    prior_machine_id: int = 987654,
    current_machine_id: int = 987655,
) -> tuple[Path, str, Path, dict[str, Path]]:
    root = fixture.config_path.parent
    attempt_path, _, terminal = _write_bound_launch_provenance(fixture, machine_id=prior_machine_id)
    prior_attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    prior_attempt_bytes = attempt_path.read_bytes()
    prior_snapshot_path = root / runner.SOURCE_SNAPSHOT_RELATIVE_PATH
    prior_snapshot_bytes = prior_snapshot_path.read_bytes()
    attempt_id = str(prior_attempt["attempt_id"])
    evidence_root = root / runner.RETRY_EVIDENCE_ROOT / attempt_id
    (evidence_root / "essential").mkdir(parents=True)
    evidence = {
        "outcome": evidence_root / "outcome.json",
        "attempt": evidence_root / "attempt-preregistration.json",
        "snapshot": evidence_root / "source-snapshot.json",
        "failure": evidence_root / "essential" / "failure.json",
        "manifest": evidence_root / "essential" / "artifact-manifest.json",
        "lifecycle": evidence_root / "lifecycle.json",
    }
    _write(evidence["attempt"], prior_attempt_bytes)
    prior_attempt_sha256 = hashlib.sha256(prior_attempt_bytes).hexdigest()
    prior_snapshot_sha256 = _write(evidence["snapshot"], prior_snapshot_bytes)
    failure = {
        "schema_version": "barun-mobile-temporal-execution-failure-v1",
        "jarvis_resource_id": str(prior_machine_id),
        "artifacts_preserved": True,
        "automatic_retry_attempted": False,
        "usable_held_out_artifacts": [],
        "retry_after_this_attempt": "external_controller_must_verify_eligibility",
        "official_961_rows_read": 0,
        "reused_756_rows_read": 0,
        "reused_756_rows_scored": 0,
    }
    failure_sha256 = _write(evidence["failure"], _json_bytes(failure))
    manifest = {
        "schema_version": "barun-mobile-temporal-essential-v1",
        "status": "prebackend_failure",
        "file_count": 1,
        "total_bytes": evidence["failure"].stat().st_size,
        "files": {
            "failure.json": {
                "bytes": evidence["failure"].stat().st_size,
                "sha256": failure_sha256,
            }
        },
    }
    manifest_sha256 = _write(evidence["manifest"], _json_bytes(manifest))
    relative = {name: path.relative_to(root).as_posix() for name, path in evidence.items()}
    lifecycle = {
        "machine_id": prior_machine_id,
        "final_instance": {"machine_id": prior_machine_id, "status": "paused"},
        "final_run_status": {"machine_id": prior_machine_id, "state": "failed"},
        "evidence_collection": {
            "artifact_status": "succeeded",
            "log_status": "succeeded",
            "artifact_manifest_path": relative["manifest"],
            "artifact_manifest_sha256": manifest_sha256,
        },
        "attempt_inventory_binding": {
            "machine_id": prior_machine_id,
            "after_sha256": prior_attempt_sha256,
        },
    }
    lifecycle_sha256 = _write(evidence["lifecycle"], _json_bytes(lifecycle))
    config_sha256 = hashlib.sha256(fixture.config_path.read_bytes()).hexdigest()
    outcome = {
        "schema_version": "barun-mobile-temporal-prior-attempt-outcome-v1",
        "attempt_id": attempt_id,
        "attempt_ordinal": 1,
        "run_id": prior_attempt["run_id"],
        "machine_id": prior_machine_id,
        "attempt_preregistration": {
            "path": relative["attempt"],
            "sha256": prior_attempt_sha256,
        },
        "scientific_config_sha256": config_sha256,
        "source_snapshot_sha256": prior_snapshot_sha256,
        "source_snapshot": {
            "path": relative["snapshot"],
            "sha256": prior_snapshot_sha256,
        },
        "execution_failure": {"path": relative["failure"], "sha256": failure_sha256},
        "essential_manifest": {"path": relative["manifest"], "sha256": manifest_sha256},
        "lifecycle": {"path": relative["lifecycle"], "sha256": lifecycle_sha256},
        "usable_held_out_signals": 0,
    }
    outcome_sha256 = _write(evidence["outcome"], _json_bytes(outcome))

    current_attempt = dict(prior_attempt)
    current_attempt["attempt_id"] = f"{prior_attempt['run_id']}-attempt-2"
    current_attempt["attempt_ordinal"] = 2
    current_attempt["prior_attempts"] = [
        {
            "attempt_id": attempt_id,
            "attempt_ordinal": 1,
            "outcome_receipt_path": relative["outcome"],
            "outcome_receipt_sha256": outcome_sha256,
            "usable_held_out_signals": 0,
        }
    ]
    protected = sorted({*runner.KNOWN_PROTECTED_JARVIS_IDS, prior_machine_id})
    current_attempt["prelaunch_inventory"] = {
        **current_attempt["prelaunch_inventory"],
        "protected_machine_ids": protected,
        "project_machine_id": current_machine_id,
    }
    base_exclusions = {
        runner.ATTEMPT_PREREGISTRATION_RELATIVE_PATH,
        runner.SOURCE_SNAPSHOT_RELATIVE_PATH,
        terminal.name,
    }
    exact_exclusions = sorted(base_exclusions | set(relative.values()))
    tree = runner._recompute_source_tree(root, excluded_exact_paths=exact_exclusions)
    current_snapshot = json.loads(prior_snapshot_bytes)
    current_snapshot["content_tree"] = {
        "algorithm": runner.SOURCE_TREE_ALGORITHM,
        **tree,
        "excluded_exact_paths": exact_exclusions,
        "excluded_directory_names": list(runner.SOURCE_TREE_EXCLUDED_DIRECTORY_NAMES),
        "excluded_directory_suffixes": list(runner.SOURCE_TREE_EXCLUDED_DIRECTORY_SUFFIXES),
        "excluded_file_names": list(runner.SOURCE_TREE_EXCLUDED_FILE_NAMES),
        "excluded_file_suffixes": list(runner.SOURCE_TREE_EXCLUDED_FILE_SUFFIXES),
    }
    current_snapshot_sha256 = _write(prior_snapshot_path, _json_bytes(current_snapshot))
    current_attempt["source_snapshot"] = {
        "path": runner.SOURCE_SNAPSHOT_RELATIVE_PATH,
        "sha256": current_snapshot_sha256,
    }
    current_attempt_sha256 = _write(attempt_path, _json_bytes(current_attempt))
    return attempt_path, current_attempt_sha256, terminal, evidence


def _patch_shadow_contract(monkeypatch: pytest.MonkeyPatch, fixture: FrozenFixture) -> None:
    monkeypatch.setattr(runner, "PINNED_TRAIN_ROWS", fixture.source_rows)
    monkeypatch.setattr(runner, "PINNED_TRAIN_SHA256", fixture.source_sha256)
    monkeypatch.setattr(runner, "PINNED_TRAIN_MEMBERSHIP_SHA256", fixture.source_membership_sha256)
    monkeypatch.setattr(runner, "EXPECTED_SHADOW_ROWS", fixture.shadow_rows)
    monkeypatch.setattr(runner, "EXPECTED_SHADOW_MEMBERSHIP_SHA256", fixture.shadow_memberships)
    monkeypatch.setattr(runner, "EXPECTED_FULL_MINIMUM_VARIANTS", 1_000)
    payload = json.loads(fixture.config_path.read_text())
    monkeypatch.setattr(
        runner,
        "EXPECTED_SCIENTIFIC_CONTRACT_SHA256",
        runner._scientific_contract_sha256(payload),
    )


def _rewrite_config_for_current_audits(fixture: FrozenFixture) -> None:
    config = json.loads(fixture.config_path.read_text())
    materialized = fixture.view_audit_path.parent
    config["materialization"]["view_audit_sha256"] = hashlib.sha256(
        fixture.view_audit_path.read_bytes()
    ).hexdigest()
    config["materialization"]["shadow_audit_sha256"] = hashlib.sha256(
        (materialized / "shadow-audit.json").read_bytes()
    ).hexdigest()
    _write(fixture.config_path, _json_bytes(config))


def test_runner_imports_without_torch_or_cuda_side_effects() -> None:
    code = """
import runpy
import sys

assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
runpy.run_path("scripts/run_mobile_temporal_counterfactual.py", run_name="_preflight_import")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_scientific_contract_mutation_fails_before_torch_import() -> None:
    code = """
import json
import runpy
import sys

runner = runpy.run_path(
    "scripts/run_mobile_temporal_counterfactual.py", run_name="_contract_mutation"
)
with open("configs/mobile_temporal_counterfactual_v1.json", encoding="utf-8") as handle:
    config = json.load(handle)
config["schema_version"] = runner["CONFIG_SCHEMA_VERSION"]
config["confirmation_gate"]["evaluator_thresholds"]["minimum_positive_seed_count"] = 0
try:
    runner["_validate_config"](config)
except runner["TemporalPreflightError"] as error:
    assert "scientific contract changed" in str(error)
else:
    raise AssertionError("mutated confirmation gate passed stdlib freeze")
with open("configs/mobile_temporal_counterfactual_v1.json", encoding="utf-8") as handle:
    config = json.load(handle)
config["schema_version"] = runner["CONFIG_SCHEMA_VERSION"]
config["run_id"] = "20260803-2354-mobile-temporal-counterfactual-s17"
try:
    runner["_validate_config"](config)
except runner["TemporalPreflightError"] as error:
    assert "scientific contract changed" in str(error)
else:
    raise AssertionError("mutated run identity passed stdlib freeze")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_plan_proves_frozen_design_and_is_byte_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    _patch_shadow_contract(monkeypatch, fixture)

    plan = runner.run_plan(
        config_path=fixture.config_path,
        materialization_receipt=fixture.view_audit_path,
        output_dir=fixture.output_dir,
    )

    assert plan["fit_count"] == 9
    assert plan["maximum_fit_count_if_confirmation_passes"] == 10
    assert plan["conditional_full_refit"]["status"] == (
        "conditional_disabled_until_confirmation_gate_passes"
    )
    assert plan["conditional_full_refit"]["authorization"] == (
        "execute_flag_and_passing_confirmation_gate_required"
    )
    assert plan["execution_backend"] == runner.EXECUTION_BACKEND_VERSION
    assert [(fit["arm_id"], fit["seed"]) for fit in plan["fits"]] == [
        (arm, seed) for seed in (17, 29, 43) for arm in ("A", "B", "C")
    ]
    assert {fit["checkpoint_policy"] for fit in plan["fits"]} == {"final_only"}
    assert plan["fits"][1]["expected_optimizer_steps"] == 32
    assert plan["fits"][2]["expected_optimizer_steps"] == 32
    receipt_path = fixture.output_dir / "preflight-receipt.json"
    plan_path = fixture.output_dir / "execution-plan.json"
    before = (receipt_path.read_bytes(), plan_path.read_bytes())
    receipt = json.loads(receipt_path.read_text())
    assert receipt["paired_control"]["safe_variants"] == 1_000
    assert receipt["conditional_full_refit"]["variant_rows"] == 1_000
    assert receipt["conditional_full_refit"]["status"] == (
        "conditional_disabled_until_confirmation_gate_passes"
    )
    assert receipt["calendar_presentations"]["mbcf"]["cross_month_fraction"] == 0.5
    assert receipt["official_evaluation"]["rows_read"] == 0
    assert receipt["model_or_cuda_loaded"] is False

    repeated = runner.run_plan(
        config_path=fixture.config_path,
        materialization_receipt=fixture.view_audit_path,
        output_dir=fixture.output_dir,
    )
    assert repeated == plan
    assert (receipt_path.read_bytes(), plan_path.read_bytes()) == before


def test_source_order_drift_fails_before_plan_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    _patch_shadow_contract(monkeypatch, fixture)
    materialized = fixture.view_audit_path.parent
    counterfactual_path = materialized / "counterfactual-train.jsonl"
    rows = [json.loads(line) for line in counterfactual_path.read_text().splitlines()]
    rows[-1], rows[-2] = rows[-2], rows[-1]
    counterfactual_sha256 = _write(counterfactual_path, _jsonl_bytes(rows))
    view_audit = json.loads(fixture.view_audit_path.read_text())
    view_audit["views"]["counterfactual"]["sha256"] = counterfactual_sha256
    _write(fixture.view_audit_path, _json_bytes(view_audit))
    config = json.loads(fixture.config_path.read_text())
    config["materialization"]["artifact_sha256"]["counterfactual"] = counterfactual_sha256
    _write(fixture.config_path, _json_bytes(config))
    _rewrite_config_for_current_audits(fixture)
    monkeypatch.setattr(
        runner,
        "EXPECTED_SCIENTIFIC_CONTRACT_SHA256",
        runner._scientific_contract_sha256(json.loads(fixture.config_path.read_text())),
    )

    with pytest.raises(runner.TemporalPreflightError, match="source presentation order"):
        runner.run_plan(
            config_path=fixture.config_path,
            materialization_receipt=fixture.view_audit_path,
            output_dir=fixture.output_dir,
        )

    assert not fixture.output_dir.exists()


def test_hash_drift_and_official_access_fail_before_plan_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    _patch_shadow_contract(monkeypatch, fixture)
    repeat_path = fixture.view_audit_path.parent / "repeat-train.jsonl"
    repeat_path.write_bytes(repeat_path.read_bytes() + b"\n")

    with pytest.raises(runner.TemporalPreflightError, match="SHA-256 changed"):
        runner.run_plan(
            config_path=fixture.config_path,
            materialization_receipt=fixture.view_audit_path,
            output_dir=fixture.output_dir,
        )
    assert not fixture.output_dir.exists()

    config = json.loads(fixture.config_path.read_text())
    config["official_evaluation"]["rows_read"] = 1
    _write(fixture.config_path, _json_bytes(config))
    with pytest.raises(runner.TemporalPreflightError, match="scientific contract changed"):
        runner.run_plan(
            config_path=fixture.config_path,
            materialization_receipt=fixture.view_audit_path,
            output_dir=fixture.output_dir,
        )
    assert not fixture.output_dir.exists()


def test_conditional_full_refit_hash_drift_fails_before_plan_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    _patch_shadow_contract(monkeypatch, fixture)
    full_train = tmp_path / "materialized-full-refit-v1" / "full-counterfactual-train.jsonl"
    full_train.write_bytes(full_train.read_bytes() + b"\n")

    with pytest.raises(runner.TemporalPreflightError, match="SHA-256 changed"):
        runner.run_plan(
            config_path=fixture.config_path,
            materialization_receipt=fixture.view_audit_path,
            output_dir=fixture.output_dir,
        )

    assert not fixture.output_dir.exists()


def test_execute_enters_lazy_backend_only_after_preflight_and_cublas_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    fixture = replace(fixture, output_dir=tmp_path.parent / f"{tmp_path.name}-execution-output")
    _patch_shadow_contract(monkeypatch, fixture)
    attempt_path, attempt_sha256, terminal_path = _write_bound_launch_provenance(fixture)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    observed: dict[str, object] = {}
    original_sha256_file = runner._sha256_file

    def guarded_sha256_file(path: Path) -> str:
        assert path.resolve() != terminal_path.resolve(), "terminal bytes read during preflight"
        return original_sha256_file(path)

    def fake_backend(**kwargs):
        observed.update(kwargs)
        assert (fixture.output_dir / "preflight-receipt.json").is_file()
        assert (fixture.output_dir / "execution-plan.json").is_file()
        assert (fixture.output_dir / "attempt-preregistration.json").is_file()
        assert (fixture.output_dir / "source-snapshot.json").is_file()
        assert runner.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert runner.os.environ["WANDB_MODE"] == "disabled"
        return {"status": "fake-complete"}

    monkeypatch.setattr(runner, "_torch_is_imported", lambda: False)
    monkeypatch.setattr(runner, "_sha256_file", guarded_sha256_file)
    monkeypatch.setattr(
        runner,
        "_run_pre_cuda_validation",
        lambda **_kwargs: {"status": "passed", "log_sha256": "f" * 64},
    )
    monkeypatch.setattr(runner, "_execute_frozen_backend", fake_backend)
    result = runner.run_plan(
        config_path=fixture.config_path,
        materialization_receipt=fixture.view_audit_path,
        output_dir=fixture.output_dir,
        execute=True,
        device="cuda",
        jarvis_resource_id="987654",
        terminal_manifest=terminal_path,
        terminal_manifest_sha256=runner.PINNED_REUSED_756_SHA256,
        attempt_preregistration=attempt_path,
        attempt_preregistration_sha256=attempt_sha256,
    )

    assert result == {"status": "fake-complete"}
    assert observed["jarvis_resource_id"] == "987654"
    assert observed["cublas_configured_before_torch_import"] is True
    assert (
        observed["execution_plan"]["launch_provenance"]["attempt_preregistration"][
            "project_machine_id"
        ]
        == 987654
    )
    assert (fixture.output_dir / "preflight-receipt.json").is_file()
    assert (fixture.output_dir / "execution-plan.json").is_file()


def test_runtime_source_validation_accepts_only_active_root_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    _patch_shadow_contract(monkeypatch, fixture)
    attempt_path, attempt_sha256, terminal_path = _write_bound_launch_provenance(fixture)
    snapshot = json.loads((tmp_path / runner.SOURCE_SNAPSHOT_RELATIVE_PATH).read_text())
    scientific_tree = {
        name: snapshot["content_tree"][name]
        for name in ("sha256", "file_count", "content_bytes", "files")
    }
    _install_active_runtime_root_venv(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)

    result = runner._validate_attempt_preregistration(
        attempt_path=attempt_path,
        config_path=fixture.config_path,
        config=json.loads(fixture.config_path.read_text(encoding="utf-8")),
        materialization_receipt=fixture.view_audit_path,
        terminal_binding={
            "manifest": str(terminal_path.resolve()),
            "manifest_sha256": runner.PINNED_REUSED_756_SHA256,
        },
        jarvis_resource_id="987654",
        expected_attempt_sha256=attempt_sha256,
    )

    assert result["source_snapshot"]["content_tree"] == scientific_tree
    assert all(not record["path"].startswith(".venv/") for record in scientific_tree["files"])


def test_source_tree_builder_default_rejects_active_root_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "source.txt").write_text("scientific source\n", encoding="utf-8")
    _install_active_runtime_root_venv(tmp_path, monkeypatch)

    with pytest.raises(
        runner.TemporalPreflightError,
        match=r"forbidden excluded directory: \.venv",
    ):
        runner._recompute_source_tree(tmp_path, excluded_exact_paths=[])


def test_runtime_source_validation_rejects_inactive_root_venv(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_text("scientific source\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()

    with pytest.raises(runner.TemporalPreflightError, match="not the active Python environment"):
        runner._recompute_source_tree(
            tmp_path,
            excluded_exact_paths=[],
            allow_active_runtime_root_venv=True,
        )


def test_runtime_source_validation_rejects_symlinked_root_venv(tmp_path: Path) -> None:
    provider_venv = tmp_path.parent / f"{tmp_path.name}-provider-venv"
    provider_venv.mkdir()
    (tmp_path / ".venv").symlink_to(provider_venv, target_is_directory=True)

    with pytest.raises(runner.TemporalPreflightError, match=r"symlink: \.venv"):
        runner._recompute_source_tree(
            tmp_path,
            excluded_exact_paths=[],
            allow_active_runtime_root_venv=True,
        )


@pytest.mark.parametrize("relative", [".cache", "venv", "nested/.venv", "nested/__pycache__"])
def test_runtime_venv_exception_rejects_every_other_excluded_directory(
    tmp_path: Path, relative: str
) -> None:
    (tmp_path / relative).mkdir(parents=True)

    with pytest.raises(runner.TemporalPreflightError, match="forbidden excluded directory"):
        runner._recompute_source_tree(
            tmp_path,
            excluded_exact_paths=[],
            allow_active_runtime_root_venv=True,
        )


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("template", "pytorch"),
        ("python_implementation", "PyPy"),
        ("python_version", "3.11.9"),
    ],
)
def test_attempt_rejects_unbound_compute_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    changed: str,
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    _patch_shadow_contract(monkeypatch, fixture)
    attempt_path, _, terminal_path = _write_bound_launch_provenance(fixture)
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt["compute"][field] = changed
    attempt_sha256 = _write(attempt_path, _json_bytes(attempt))
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)

    with pytest.raises(runner.TemporalPreflightError, match="compute contract changed"):
        runner._validate_attempt_preregistration(
            attempt_path=attempt_path,
            config_path=fixture.config_path,
            config=json.loads(fixture.config_path.read_text(encoding="utf-8")),
            materialization_receipt=fixture.view_audit_path,
            terminal_binding={"manifest": str(terminal_path.resolve())},
            jarvis_resource_id="987654",
            expected_attempt_sha256=attempt_sha256,
        )


@pytest.mark.parametrize("missing_id", sorted(runner.KNOWN_PROTECTED_JARVIS_IDS))
def test_attempt_requires_every_durable_protected_machine_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_id: int
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    fixture = replace(fixture, output_dir=tmp_path.parent / f"{tmp_path.name}-execution-output")
    _patch_shadow_contract(monkeypatch, fixture)
    attempt_path, _, terminal_path = _write_bound_launch_provenance(fixture)
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt["prelaunch_inventory"]["protected_machine_ids"].remove(missing_id)
    attempt_sha256 = _write(attempt_path, _json_bytes(attempt))
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)

    with pytest.raises(runner.TemporalPreflightError, match="denylist"):
        runner.run_plan(
            config_path=fixture.config_path,
            materialization_receipt=fixture.view_audit_path,
            output_dir=fixture.output_dir,
            execute=True,
            device="cuda",
            jarvis_resource_id="987654",
            terminal_manifest=terminal_path,
            terminal_manifest_sha256=runner.PINNED_REUSED_756_SHA256,
            attempt_preregistration=attempt_path,
            attempt_preregistration_sha256=attempt_sha256,
        )

    failure = json.loads(
        (fixture.output_dir / "execution" / "essential" / "failure.json").read_text()
    )
    assert failure["retry_after_this_attempt"] == "forbidden_unproven_prebackend_failure"


def test_execute_requires_explicit_cuda_and_jarvis_id_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    fixture = replace(fixture, output_dir=tmp_path.parent / f"{tmp_path.name}-execution-output")
    _patch_shadow_contract(monkeypatch, fixture)

    with pytest.raises(runner.TemporalPreflightError, match="device cuda"):
        runner.run_plan(
            config_path=fixture.config_path,
            materialization_receipt=fixture.view_audit_path,
            output_dir=fixture.output_dir,
            execute=True,
            jarvis_resource_id="987654",
        )

    failure = json.loads(
        (fixture.output_dir / "execution" / "essential" / "failure.json").read_text()
    )
    assert failure["phase"] == "stdlib_prebackend"
    assert failure["jarvis_resource_id"] == "987654"
    assert failure["official_961_rows_read"] == 0
    assert (fixture.output_dir / "execution" / "essential" / "artifact-manifest.json").is_file()


def test_execute_rejects_output_inside_staged_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)

    with pytest.raises(runner.TemporalPreflightError, match="outside"):
        runner.run_plan(
            config_path=fixture.config_path,
            materialization_receipt=fixture.view_audit_path,
            output_dir=fixture.output_dir,
            execute=True,
            device="cuda",
            jarvis_resource_id="987654",
        )

    assert not fixture.output_dir.exists()


def test_parser_accepts_safe_run_jarvis_machine_id_shape(tmp_path: Path) -> None:
    args = runner.build_parser().parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--materialization-receipt",
            str(tmp_path / "view-audit.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--execute",
            "--device",
            "cuda",
            "--jarvis-machine-id",
            "987654",
            "--terminal-manifest",
            str(tmp_path / "reused-756.jsonl"),
            "--terminal-manifest-sha256",
            runner.PINNED_REUSED_756_SHA256,
            "--attempt-preregistration",
            str(tmp_path / "attempt-preregistration.json"),
            "--attempt-preregistration-sha256",
            "a" * 64,
        ]
    )

    assert args.jarvis_resource_id == "987654"
    assert args.attempt_preregistration == tmp_path / "attempt-preregistration.json"
    assert args.attempt_preregistration_sha256 == "a" * 64


def test_cpu_gate_uses_only_exact_staged_src_on_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    staged_src = tmp_path / "src"
    staged_src.mkdir()
    terminal = tmp_path / "terminal.jsonl"
    terminal.write_text("sealed\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout=b"cpu gate passed\n")

    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_torch_is_imported", lambda: False)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    _patch_expected_runtime_identity(monkeypatch)
    config = json.loads(fixture.config_path.read_text(encoding="utf-8"))

    receipt = runner._run_pre_cuda_validation(
        config=config,
        output_dir=tmp_path / "cpu-gate",
        terminal_manifest=terminal,
    )

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["PYTHONPATH"] == str(staged_src)
    assert receipt["controller_safety_overrides"]["PYTHONPATH"] == str(staged_src)
    assert receipt["runtime_identity"] == {
        "status": "matched",
        "expected": {
            "python_implementation": "CPython",
            "python_version": "3.11.10",
            "sys_implementation_name": "cpython",
        },
        "observed": {
            "python_implementation": "CPython",
            "python_version": "3.11.10",
            "sys_implementation_name": "cpython",
        },
    }
    assert not any(tmp_path.rglob("*.egg-info"))


@pytest.mark.parametrize(
    ("implementation", "version"),
    [("PyPy", "3.11.10"), ("CPython", "3.11.9")],
)
def test_cpu_gate_rejects_runtime_identity_before_remote_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    implementation: str,
    version: str,
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    (tmp_path / "src").mkdir()
    terminal = tmp_path / "terminal.jsonl"
    terminal.write_text("sealed\n", encoding="utf-8")

    def forbidden_remote_test(*_args, **_kwargs):
        raise AssertionError("remote tests ran with an unbound Python runtime")

    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_torch_is_imported", lambda: False)
    monkeypatch.setattr(runner.platform, "python_implementation", lambda: implementation)
    monkeypatch.setattr(runner.platform, "python_version", lambda: version)
    monkeypatch.setattr(runner.subprocess, "run", forbidden_remote_test)

    with pytest.raises(runner.TemporalPreflightError, match="runtime Python identity changed"):
        runner._run_pre_cuda_validation(
            config=json.loads(fixture.config_path.read_text(encoding="utf-8")),
            output_dir=tmp_path / "cpu-identity-failure",
            terminal_manifest=terminal,
        )


def test_cpu_gate_timeout_fails_closed_with_durable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    (tmp_path / "src").mkdir()
    terminal = tmp_path / "terminal.jsonl"
    terminal.write_text("sealed\n", encoding="utf-8")

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == runner.REMOTE_VALIDATION_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output=b"partial output\n")

    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_torch_is_imported", lambda: False)
    monkeypatch.setattr(runner.subprocess, "run", timeout)
    _patch_expected_runtime_identity(monkeypatch)
    config = json.loads(fixture.config_path.read_text(encoding="utf-8"))
    output = tmp_path / "cpu-timeout"

    with pytest.raises(runner.TemporalPreflightError, match="exceeded"):
        runner._run_pre_cuda_validation(
            config=config,
            output_dir=output,
            terminal_manifest=terminal,
        )

    receipt = json.loads((output / "pre-cuda-validation.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "timed_out"
    assert receipt["exit_code"] is None
    assert receipt["timeout_seconds"] == 600
    assert b"partial output" in (output / "pre-cuda-validation.log").read_bytes()


def test_backend_imports_from_exact_staged_src_without_editable_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged_src = tmp_path / "src"
    package = staged_src / "barunlm" / "training"
    package.mkdir(parents=True)
    (staged_src / "barunlm" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "mobile_temporal_experiment.py").write_text(
        "def execute_temporal_experiment(**kwargs):\n"
        "    return {'source': __file__, 'kwargs': kwargs}\n",
        encoding="utf-8",
    )
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "barunlm" or name.startswith("barunlm.")
    }
    for name in original_modules:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(sys, "path", [str(staged_src)])
    try:
        observed = runner._execute_frozen_backend(probe="ok")
        assert Path(observed["source"]).resolve().is_relative_to(staged_src)
        assert observed["kwargs"] == {"probe": "ok"}
        assert not any(tmp_path.rglob("*.egg-info"))
    finally:
        for name in list(sys.modules):
            if name == "barunlm" or name.startswith("barunlm."):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)


def test_staged_source_activation_rejects_conflicting_package_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "src").mkdir()
    rogue = tmp_path / "rogue"
    (rogue / "barunlm").mkdir(parents=True)
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "barunlm" or name.startswith("barunlm.")
    }
    for name in original_modules:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(sys, "path", [str(rogue)])

    try:
        with pytest.raises(runner.TemporalPreflightError, match="conflicting"):
            runner._activate_staged_source_import()
    finally:
        sys.modules.update(original_modules)


def test_retry_accepts_distinct_snapshot_bytes_with_only_attempt_scoped_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    (requirements / "mobile-temporal-counterfactual.txt").write_text(
        "torch==2.13.0\n", encoding="utf-8"
    )
    attempt_path, attempt_sha256, terminal, evidence = _write_retry_launch_provenance(fixture)
    _patch_shadow_contract(monkeypatch, fixture)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    config = json.loads(fixture.config_path.read_text(encoding="utf-8"))

    result = runner._validate_attempt_preregistration(
        attempt_path=attempt_path,
        config_path=fixture.config_path,
        config=config,
        materialization_receipt=fixture.view_audit_path,
        terminal_binding={
            "manifest": str(terminal.resolve()),
            "manifest_sha256": runner.PINNED_REUSED_756_SHA256,
        },
        jarvis_resource_id="987655",
        expected_attempt_sha256=attempt_sha256,
    )

    assert result["attempt_preregistration"]["attempt_ordinal"] == 2
    assert (
        hashlib.sha256(evidence["snapshot"].read_bytes()).hexdigest()
        != hashlib.sha256(
            (tmp_path / runner.SOURCE_SNAPSHOT_RELATIVE_PATH).read_bytes()
        ).hexdigest()
    )


@pytest.mark.parametrize(
    ("relative_path", "initial", "changed"),
    [
        ("pyproject.toml", "[build-system]\n", "[build-system]\nrequires=[]\n"),
        (
            "requirements/mobile-temporal-counterfactual.txt",
            "torch==2.13.0\n",
            "torch==2.13.1\n",
        ),
        ("infra/jarvis_safe_run.py", "# frozen\n", "# changed\n"),
    ],
)
def test_retry_rejects_execution_source_mutation_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    initial: str,
    changed: str,
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    changed_path = tmp_path / relative_path
    changed_path.parent.mkdir(parents=True, exist_ok=True)
    changed_path.write_text(initial, encoding="utf-8")
    attempt_path, attempt_sha256, terminal, _ = _write_retry_launch_provenance(fixture)
    changed_path.write_text(changed, encoding="utf-8")
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    config = json.loads(fixture.config_path.read_text(encoding="utf-8"))

    with pytest.raises(runner.TemporalPreflightError, match="content tree"):
        runner._validate_attempt_preregistration(
            attempt_path=attempt_path,
            config_path=fixture.config_path,
            config=config,
            materialization_receipt=fixture.view_audit_path,
            terminal_binding={"manifest": str(terminal.resolve())},
            jarvis_resource_id="987655",
            expected_attempt_sha256=attempt_sha256,
        )


def test_retry_evidence_cannot_hide_requirements_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    requirements_path = requirements / "mobile-temporal-counterfactual.txt"
    requirements_path.write_text("torch==2.13.0\n", encoding="utf-8")
    attempt_path, _, terminal, evidence = _write_retry_launch_provenance(fixture)
    outcome = json.loads(evidence["outcome"].read_text(encoding="utf-8"))
    outcome["source_snapshot"]["path"] = requirements_path.relative_to(tmp_path).as_posix()
    outcome_sha256 = _write(evidence["outcome"], _json_bytes(outcome))
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt["prior_attempts"][0]["outcome_receipt_sha256"] = outcome_sha256
    attempt_sha256 = _write(attempt_path, _json_bytes(attempt))
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    config = json.loads(fixture.config_path.read_text(encoding="utf-8"))

    with pytest.raises(runner.TemporalPreflightError, match="attempt-scoped"):
        runner._validate_attempt_preregistration(
            attempt_path=attempt_path,
            config_path=fixture.config_path,
            config=config,
            materialization_receipt=fixture.view_audit_path,
            terminal_binding={"manifest": str(terminal.resolve())},
            jarvis_resource_id="987655",
            expected_attempt_sha256=attempt_sha256,
        )


def test_retry_rejects_heldout_marker_bound_by_prior_essential_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_frozen_fixture(tmp_path)
    attempt_path, _, terminal, evidence = _write_retry_launch_provenance(fixture)
    marker = (
        evidence["manifest"].parent / "fits" / "a-s17" / "selection-heldout-access-started.json"
    )
    marker.parent.mkdir(parents=True)
    marker_sha256 = _write(marker, _json_bytes({"population": "selection"}))
    marker_relative = marker.relative_to(evidence["manifest"].parent).as_posix()
    manifest = json.loads(evidence["manifest"].read_text(encoding="utf-8"))
    manifest["files"][marker_relative] = {
        "bytes": marker.stat().st_size,
        "sha256": marker_sha256,
    }
    manifest["file_count"] = len(manifest["files"])
    manifest["total_bytes"] = sum(entry["bytes"] for entry in manifest["files"].values())
    manifest_sha256 = _write(evidence["manifest"], _json_bytes(manifest))
    lifecycle = json.loads(evidence["lifecycle"].read_text(encoding="utf-8"))
    lifecycle["evidence_collection"]["artifact_manifest_sha256"] = manifest_sha256
    lifecycle_sha256 = _write(evidence["lifecycle"], _json_bytes(lifecycle))
    outcome = json.loads(evidence["outcome"].read_text(encoding="utf-8"))
    outcome["essential_manifest"]["sha256"] = manifest_sha256
    outcome["lifecycle"]["sha256"] = lifecycle_sha256
    outcome_sha256 = _write(evidence["outcome"], _json_bytes(outcome))
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt["prior_attempts"][0]["outcome_receipt_sha256"] = outcome_sha256
    attempt_sha256 = _write(attempt_path, _json_bytes(attempt))
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    config = json.loads(fixture.config_path.read_text(encoding="utf-8"))

    with pytest.raises(runner.TemporalPreflightError, match="held-out signal"):
        runner._validate_attempt_preregistration(
            attempt_path=attempt_path,
            config_path=fixture.config_path,
            config=config,
            materialization_receipt=fixture.view_audit_path,
            terminal_binding={"manifest": str(terminal.resolve())},
            jarvis_resource_id="987655",
            expected_attempt_sha256=attempt_sha256,
        )


def test_outer_failure_fallback_preserves_backend_error_and_compact_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged_root = tmp_path / "stage"
    staged_root.mkdir()
    output = tmp_path / "output"
    original_error = RuntimeError("backend exploded after scoring")
    failure_bytes: bytes | None = None

    def fake_impl(**kwargs):
        nonlocal failure_bytes
        execution = kwargs["output_dir"] / "execution"
        selection = execution / "fits" / "a-s17" / "selection"
        scores = selection / "scores"
        scores.mkdir(parents=True)
        predictions = selection / "predictions.jsonl"
        sample_scores = scores / "sample_scores.jsonl"
        predictions.write_text('{"prediction":"x"}\n', encoding="utf-8")
        sample_scores.write_text('{"exact":true}\n', encoding="utf-8")
        (execution / "essential-bundle-failure.json").write_text(
            '{"error":"copy failed"}\n', encoding="utf-8"
        )
        promoted = execution / "conditional-full-refit" / "inference-checkpoint"
        promoted.mkdir(parents=True)
        (promoted / "model.safetensors").write_bytes(b"promoted-weights")
        training = execution / "training" / "screening"
        training.mkdir(parents=True)
        (training / "optimizer.pt").write_bytes(b"must-not-copy")
        usable = [
            predictions.relative_to(execution).as_posix(),
            sample_scores.relative_to(execution).as_posix(),
        ]
        failure = {
            "schema_version": "barun-mobile-temporal-execution-failure-v1",
            "jarvis_resource_id": "987654",
            "artifacts_preserved": True,
            "automatic_retry_attempted": False,
            "usable_held_out_artifacts": usable,
            "retry_after_this_attempt": "forbidden_usable_held_out_signal",
            "official_961_rows_read": 0,
            "reused_756_rows_read": 0,
            "reused_756_rows_scored": 0,
        }
        failure_bytes = _json_bytes(failure)
        (execution / "failure.json").write_bytes(failure_bytes)
        (execution / "essential").mkdir()
        raise original_error

    monkeypatch.setattr(runner, "REPOSITORY_ROOT", staged_root)
    monkeypatch.setattr(runner, "_run_plan_impl", fake_impl)
    with pytest.raises(RuntimeError, match="backend exploded") as caught:
        runner.run_plan(
            config_path=staged_root / "unused-config.json",
            materialization_receipt=staged_root / "unused-audit.json",
            output_dir=output,
            execute=True,
            device="cuda",
            jarvis_resource_id="987654",
        )

    assert caught.value is original_error
    assert failure_bytes is not None
    execution = output / "execution"
    essential = execution / "essential"
    assert (execution / "failure.json").read_bytes() == failure_bytes
    assert (essential / "failure.json").read_bytes() == failure_bytes
    assert (essential / "fits/a-s17/selection/predictions.jsonl").is_file()
    assert (essential / "fits/a-s17/selection/scores/sample_scores.jsonl").is_file()
    assert (essential / "essential-bundle-failure.json").is_file()
    assert (essential / "conditional-full-refit/inference-checkpoint/model.safetensors").is_file()
    assert not any(path.name == "optimizer.pt" for path in essential.rglob("*"))
    manifest = json.loads((essential / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "backend_failure_fallback"
