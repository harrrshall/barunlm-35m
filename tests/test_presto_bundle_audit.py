from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import barunlm.presto_bundle_audit as audit
from barunlm.evaluation.presto import score_rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(prefix: str) -> dict[str, str]:
    return {
        "barun_config.json": prefix * 64,
        "model.safetensors": chr(ord(prefix) + 1) * 64,
        "tokenizer.json": chr(ord(prefix) + 2) * 64,
    }


def test_artifact_manifest_is_exact_and_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "essential"
    root.mkdir()
    payload = root / "payload.json"
    payload.write_text("{}\n", encoding="utf-8")
    _write_json(root / "artifact-sha256.json", {"payload.json": _sha(payload)})
    monkeypatch.setattr(
        audit,
        "_REQUIRED_BUNDLE_FILES",
        frozenset({"artifact-sha256.json", "payload.json"}),
    )
    monkeypatch.setattr(audit, "_OPTIONAL_BUNDLE_FILES", frozenset())

    manifest, _ = audit._verify_artifact_manifest(root, {})
    assert manifest == {"payload.json": _sha(payload)}

    payload.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(audit.PrestoBundleAuditError, match="SHA-256 mismatch"):
        audit._verify_artifact_manifest(root, {})

    payload.write_text("{}\n", encoding="utf-8")
    (root / "unknown.bin").write_bytes(b"x")
    with pytest.raises(audit.PrestoBundleAuditError, match="file set mismatch"):
        audit._verify_artifact_manifest(root, {})


def test_sample_audit_counts_schema_invalid_call_conservatively() -> None:
    target = '{"decision":"ABSTAIN"}'
    raw = (
        '{"calls":[{"args":{"slots":[]},"tool":"unknown_tool"}],"decision":"CALL","mode":"SINGLE"}'
    )
    manifest = {
        "id": "unsafe",
        "prompt": (
            "<bos><system>\nACTION_IR_V1\nPRESTO_CONTEXT_V1\n"
            'CONTEXT {"contacts":[],"lists":[],"notes":[]}\n'
            "DIALOGUE []\n<user>\nrequest\n<assistant>\n"
        ),
        "target": target,
        "metadata": {
            "derived_split": "dev",
            "source_split": "dev",
            "policy_decision": "ABSTAIN",
            "root_intent": "Other",
            "linguistic_phenomenon": "none",
        },
    }
    prediction = {
        "id": "unsafe",
        "prediction_raw": raw,
        "truncated": False,
        "generation_failure": None,
        "prompt_tokens": 10,
        "generated_tokens": 10,
    }
    _, samples = score_rows([manifest], [prediction])
    original_sample = dict(samples[0])
    original_sample["schema_version"] = audit.PRESTO_SCORER_VERSION_V1

    audited = audit._audit_sample(original_sample, prediction, label="unsafe")
    assert audited["parse_valid"] is True
    assert audited["schema_valid"] is False
    assert audited["false_call_on_gate"] is True

    tampered = dict(original_sample)
    tampered["false_call_on_gate"] = False
    with pytest.raises(audit.PrestoBundleAuditError, match="false_call_on_gate"):
        audit._audit_sample(tampered, prediction, label="unsafe")


def _metric_row(group: str, decision: str = "CALL") -> dict[str, object]:
    return {
        "sample_id": group,
        "gold_decision": decision,
        "predicted_decision": decision,
        "parse_valid": True,
        "schema_valid": True,
        "ast_exact": True,
        "decision_correct": True,
        "false_call_on_gate": False,
        "phenomenon_group": group,
        "prediction_error_code": None,
        "truncated": False,
        "generation_failed": False,
    }


def test_gate_is_recomputed_from_samples_not_result_claim() -> None:
    rows = [
        _metric_row("no_phenomenon", "ABSTAIN"),
        _metric_row("revision", "CONFIRM"),
        _metric_row("disfluency", "CALL"),
    ]
    metrics = audit._recomputed_metrics(rows)
    recipe = {
        "gate": {
            "derived_ast_exact_at_least": 0.7,
            "schema_valid_at_least": 0.95,
            "abstention_f1_at_least": 0.8,
            "false_call_on_gate_at_most": 0.01,
            "maximum_no_phenomenon_to_revision_or_disfluency_gap": 0.1,
            "require_represented_gap_buckets": True,
        }
    }
    gate = audit._recompute_gate(metrics, recipe)
    assert gate["passed"] is True
    audit._compare_gate(gate, gate)

    tampered = json.loads(json.dumps(gate))
    tampered["observed"]["false_call_on_gate"] = 0.5
    with pytest.raises(audit.PrestoBundleAuditError, match="false_call_on_gate"):
        audit._compare_gate(tampered, gate)


def _anchors(root: Path) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    run_id = "20260803-1920-presto-stage-s17"
    input_hashes = _hashes("a")
    prereg = {
        "schema_version": "barun-experiment-preregistration-v1",
        "status": "frozen_before_remote_launch_or_model_scoring",
        "run_id": run_id,
        "hypothesis_and_decision": {
            "path": "configs/presto_stage_v1.json",
            "sha256": "d" * 64,
        },
        "input_checkpoint": {
            "path": "experiments/input/checkpoint",
            "model_sha256": input_hashes["model.safetensors"],
            "config_sha256": input_hashes["barun_config.json"],
            "tokenizer_sha256": input_hashes["tokenizer.json"],
        },
        "compute": {
            "provider": "JarvisLabs",
            "instance_name": "barun-presto-stage-1920-s17",
            "protected_running_instance": {"machine_id": 463058},
        },
    }
    machine_id = 463622
    managed_run_id = "r_4745576e"
    jarvis = {
        "schema_version": 1,
        "controller": "infra/jarvis/safe_run.py",
        "created_by_safe_run": True,
        "machine_id": machine_id,
        "run_id": managed_run_id,
        "instance_name": "barun-presto-stage-1920-s17",
        "create_summary": {
            "machine_id": machine_id,
            "name": "barun-presto-stage-1920-s17",
        },
        "final_instance": {
            "machine_id": machine_id,
            "name": "barun-presto-stage-1920-s17",
            "status": "Paused",
        },
        "latest_run_status": {
            "machine_id": machine_id,
            "run_id": managed_run_id,
            "state": "succeeded",
            "exit_code": 0,
        },
        "launch_summary": {"machine_id": machine_id, "run_id": managed_run_id},
        "preexisting_resources": [{"machine_id": 463058}],
        "events": [{"event": "pause_verified", "status": "Paused"}],
        "artifact_download": {
            "direction": "download",
            "exit_code": 0,
            "machine_id": machine_id,
            "recursive": True,
            "dest": str(root),
        },
        "requested": {
            "append_jarvis_machine_id": True,
            "script": "scripts/run_presto_stage.py",
            "remote_args": [
                "--run-id",
                run_id,
                "--input-checkpoint-dir",
                "experiments/input/checkpoint",
                "--input-model-sha256",
                input_hashes["model.safetensors"],
                "--input-config-sha256",
                input_hashes["barun_config.json"],
                "--input-tokenizer-sha256",
                input_hashes["tokenizer.json"],
            ],
        },
    }
    return prereg, jarvis, input_hashes


def test_run_and_jarvis_identities_require_success_download_and_pause(
    tmp_path: Path,
) -> None:
    root = tmp_path / "essential"
    root.mkdir()
    prereg, jarvis, input_hashes = _anchors(root)

    observed = audit._audit_external_anchors(prereg, jarvis, root.resolve())
    assert observed == (
        "20260803-1920-presto-stage-s17",
        463622,
        "r_4745576e",
        "d" * 64,
        input_hashes,
    )

    jarvis["final_instance"]["status"] = "Running"  # type: ignore[index]
    with pytest.raises(audit.PrestoBundleAuditError, match="final instance status"):
        audit._audit_external_anchors(prereg, jarvis, root.resolve())


def test_checkpoint_and_recipe_identities_are_anchored(tmp_path: Path) -> None:
    root = tmp_path / "essential"
    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    for name, value in (
        ("barun_config.json", b"config"),
        ("model.safetensors", b"weights"),
        ("tokenizer.json", b"tokenizer"),
    ):
        (checkpoint / name).write_bytes(value)
    output_hashes = {name: _sha(checkpoint / name) for name in audit.CHECKPOINT_FILES}
    input_hashes = dict(output_hashes)
    input_hashes["model.safetensors"] = "f" * 64
    recipe = {
        "schema_version": "barun-presto-stage-preregistration-v1",
        "recipe_id": "presto-context-safety-sft-v1",
        "proposed_input_checkpoint": {
            "model_sha256": input_hashes["model.safetensors"],
            "config_sha256": input_hashes["barun_config.json"],
            "tokenizer_sha256": input_hashes["tokenizer.json"],
        },
        "tokenizer_identity": {"sha256": input_hashes["tokenizer.json"]},
        "evaluation": {
            "native_presto_metric": False,
            "decoding": "unconstrained deterministic greedy",
        },
    }
    _write_json(root / "recipe.json", recipe)
    recipe_sha = _sha(root / "recipe.json")
    _write_json(
        root / "preregistration.json",
        {
            "schema_version": "barun-presto-effective-preregistration-v1",
            "run_id": "20260803-1920-presto-stage-s17",
            "jarvis_machine_id": 463622,
            "recipe_sha256": recipe_sha,
            "recipe": recipe,
            "created_before_dataset_preparation": True,
            "official_test_member_accessed": False,
            "input_checkpoint": {"file_sha256": input_hashes},
        },
    )
    _write_json(
        root / "bundle-manifest.json",
        {
            "schema_version": "barun-presto-essential-bundle-v1",
            "run_id": "20260803-1920-presto-stage-s17",
            "optimizer_state_included": False,
            "sample_level_base_and_post_evidence_included": True,
        },
    )
    _write_json(
        root / "result.json",
        {
            "schema_version": "barun-presto-stage-result-v1",
            "run_id": "20260803-1920-presto-stage-s17",
            "jarvis_machine_id": 463622,
            "recipe_sha256": recipe_sha,
            "metric_scope": audit.METRIC_SCOPE,
            "input_checkpoint": {"file_sha256": input_hashes},
            "output_checkpoint_sha256": output_hashes,
        },
    )
    _write_json(
        checkpoint / "checkpoint_manifest.json",
        {
            "schema_version": "barun-release-checkpoint-v1",
            "run_id": "20260803-1920-presto-stage-s17",
            "input_checkpoint_sha256": input_hashes,
            "file_sha256": output_hashes,
        },
    )
    _write_json(
        root / "environment.json",
        {
            "jarvis_machine_id": 463622,
            "cuda_available": True,
            "determinism_preflight": {
                "configured_before_torch_import": True,
                "cuda_initialized_before_preflight": False,
            },
        },
    )

    _, _, observed = audit._audit_recipe_and_identities(
        root=root,
        cache={},
        run_id="20260803-1920-presto-stage-s17",
        machine_id=463622,
        recipe_sha=recipe_sha,
        input_hashes=input_hashes,
    )
    assert observed == output_hashes

    (checkpoint / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(audit.PrestoBundleAuditError, match="exported checkpoint hash"):
        audit._audit_recipe_and_identities(
            root=root,
            cache={},
            run_id="20260803-1920-presto-stage-s17",
            machine_id=463622,
            recipe_sha=recipe_sha,
            input_hashes=input_hashes,
        )


def test_official_test_firewall_requires_member_level_zero_access(
    tmp_path: Path,
) -> None:
    root = tmp_path / "essential"
    data = root / "data"
    data.mkdir(parents=True)
    prepared = {"train": "a" * 64, "dev": "b" * 64, "audit": "c" * 64}
    official = {
        **audit._FIREWALL_ZERO_ACCESS,
        "rows": 194118,
        "sensitive_member_count": 2,
    }
    member = {
        "runtime_open_count": 0,
        "runtime_bytes_read": 0,
        "verification": "whole_archive_sha256_and_central_directory_only",
    }
    _write_json(
        data / "audit.json",
        {
            "schema_version": "barun-presto-en-audit-v1",
            "source": {
                "repository": "google-research-datasets/presto",
                "revision": "revision",
                "archive": {
                    "actual_sha256": "d" * 64,
                    "expected_sha256": "d" * 64,
                },
                "members": {
                    "presto_dataset.jsonl": member,
                    "presto_test.jsonl": member,
                },
            },
            "official_test": official,
            "artifacts": {
                "manifests": {
                    "train": {"sha256": prepared["train"]},
                    "dev": {"sha256": prepared["dev"]},
                }
            },
            "counts": {
                "english_selected": {"train": 10, "dev": 3},
                "records_dropped": 0,
                "records_truncated": 0,
            },
        },
    )
    prepared["audit"] = _sha(data / "audit.json")
    _write_json(
        data / "train-focus-audit.json",
        {
            "schema_version": "barun-presto-focus-replay-audit-v1",
            "source_sha256": prepared["train"],
            "source_rows": 10,
            "output_rows": 12,
            "output_sha256": "e" * 64,
            "selector_version": "selector",
            "maximum_replays_per_category": 2,
            "development_rows_read_for_exact_duplicate_exclusion": 3,
            "development_rows_read_for_focus_selection": 0,
            "official_test_rows_read": 0,
            "rows_after_exact_dev_exclusion": 9,
            "exact_prompt_target_train_rows_excluded": 1,
            "replay_counts": {"revision": 2, "disfluency": 1},
        },
    )
    recipe = {
        "dataset": {
            "repository": "google-research-datasets/presto",
            "revision": "revision",
            "archive_sha256": "d" * 64,
            "train_manifest_sha256": prepared["train"],
            "dev_manifest_sha256": prepared["dev"],
            "audit_sha256": prepared["audit"],
        },
        "train_view": {
            "selector_version": "selector",
            "maximum_replays_per_category": 2,
            "development_rows_read_for_exact_duplicate_exclusion": 3,
        },
    }
    result = {
        "data": {
            "revision": "revision",
            "train_rows": 10,
            "focused_train_rows": 12,
            "dev_rows": 3,
            "official_test_rows_opaque_unparsed": 194118,
            "official_test_firewall": {
                **audit._FIREWALL_ZERO_ACCESS,
                "official_test_rows_opaque_unparsed": 194118,
            },
            "prepared_hashes": prepared,
            "focus_view_sha256": "e" * 64,
        }
    }
    assert audit._audit_firewall_and_data(root=root, cache={}, recipe=recipe, result=result) == (
        10,
        3,
    )

    tampered = json.loads((data / "audit.json").read_text(encoding="utf-8"))
    tampered["source"]["members"]["presto_test.jsonl"]["runtime_open_count"] = 1
    _write_json(data / "audit.json", tampered)
    with pytest.raises(audit.PrestoBundleAuditError, match="runtime open count"):
        audit._audit_firewall_and_data(root=root, cache={}, recipe=recipe, result=result)


def test_optimizer_files_are_rejected_even_if_renamed_with_suffix(tmp_path: Path) -> None:
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "adam-optimizer-state.bin").write_bytes(b"state")
    with pytest.raises(audit.PrestoBundleAuditError, match="optimizer state"):
        audit._audit_optimizer_exclusion(tmp_path)


def test_training_lineage_connects_input_selected_checkpoint_and_output(
    tmp_path: Path,
) -> None:
    root = tmp_path / "essential"
    (root / "training").mkdir(parents=True)
    (root / "checkpoint").mkdir()
    run_id = "20260803-1920-presto-stage-s17"
    machine_id = 463622
    input_hashes = _hashes("a")
    output_hashes = _hashes("d")
    input_dir = "/remote/project/experiments/input/checkpoint"
    selected = "/remote/artifacts/training/checkpoints/step-00000010"
    focus_hash = "1" * 64
    dev_hash = "2" * 64
    config = {
        "schema_version": "barun-sft-config-v1",
        "run_id": run_id,
        "optimization": {"epochs": 1},
        "base_checkpoint": {
            "source": "local",
            "local_dir": input_dir,
            "expected_sha256": input_hashes,
        },
        "data": {"train_sha256": focus_hash, "dev_sha256": dev_hash},
        "execution": {
            "device": "cuda",
            "precision": "bf16",
            "deterministic": True,
            "jarvis_resource_id": str(machine_id),
        },
    }
    _write_json(root / "training-config.json", config)
    _write_json(
        root / "training" / "run_manifest.json",
        {
            "schema_version": "barun-sft-run-v1",
            "run_id": run_id,
            "config_source_sha256": _sha(root / "training-config.json"),
            "execution": {
                "jarvis_resource_id": str(machine_id),
                "deterministic": True,
            },
            "base_checkpoint": {
                "file_sha256": input_hashes,
                "parameter_counts": {"total": 35072768},
            },
            "data": {
                "train_manifest_sha256": focus_hash,
                "dev_manifest_sha256": dev_hash,
                "train": {"accepted_examples": 12, "rejected_examples": 0},
                "dev": {"accepted_examples": 3, "rejected_examples": 0},
            },
        },
    )
    summary = {
        "run_id": run_id,
        "global_steps": 10,
        "best_checkpoint": selected,
        "final_checkpoint": selected,
    }
    _write_json(root / "training" / "summary.json", summary)
    _write_json(
        root / "training" / "best_checkpoint.json",
        {"checkpoint": selected, "global_step": 10},
    )
    _write_jsonl(
        root / "training" / "metrics.jsonl",
        [{"timestamp": "now", "event": "complete", **summary}],
    )
    _write_json(
        root / "checkpoint" / "checkpoint_manifest.json",
        {"source_checkpoint": selected},
    )
    result = {
        "data": {
            "focus_view_sha256": focus_hash,
            "focused_train_rows": 12,
            "train_rows": 10,
        },
        "training": summary,
        "input_checkpoint": {"directory": input_dir},
        "generation": {
            "base": {"checkpoint_dir": input_dir},
            "post_sft": {
                "checkpoint_dir": selected,
                "checkpoint_sha256": output_hashes,
            },
        },
    }
    recipe = {
        "optimization": {"epochs": 1},
        "dataset": {"dev_manifest_sha256": dev_hash},
    }

    audit._audit_lineage(
        root=root,
        cache={},
        result=result,
        recipe=recipe,
        run_id=run_id,
        machine_id=machine_id,
        input_hashes=input_hashes,
        output_hashes=output_hashes,
        train_rows=10,
        dev_rows=3,
    )

    result["generation"]["post_sft"]["checkpoint_dir"] = "/wrong"  # type: ignore[index]
    with pytest.raises(audit.PrestoBundleAuditError, match="post checkpoint path"):
        audit._audit_lineage(
            root=root,
            cache={},
            result=result,
            recipe=recipe,
            run_id=run_id,
            machine_id=machine_id,
            input_hashes=input_hashes,
            output_hashes=output_hashes,
            train_rows=10,
            dev_rows=3,
        )
