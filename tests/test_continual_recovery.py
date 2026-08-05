from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from barunlm.evaluation.presto import (
    PRESTO_SCORER_VERSION,
    PRESTO_USER_REVISION_RAW_LABELS_V2,
)
from barunlm.recovery import continual_recovery
from barunlm.training.data import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "continual_recovery_v1.json"
SHARED_SELECTOR = ROOT / "configs" / "parallel_rescue_selection_v1.json"
PRESTO_CHECKPOINT = ROOT / "experiments/runs/20260803-1920-presto-stage-s17/essential/checkpoint"
PRESTO_CHECKPOINT_FILES = tuple(
    PRESTO_CHECKPOINT / name
    for name in (
        "barun_config.json",
        "checkpoint_manifest.json",
        "model.safetensors",
        "tokenizer.json",
    )
)


def _load_runner_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_barun_run_continual_recovery", ROOT / "scripts" / "run_continual_recovery.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load run_continual_recovery.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


recovery_runner = _load_runner_module()


def _membership(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> str:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return sha256_file(path)


def _checkpoint_contract(path: str) -> dict[str, Any]:
    config = continual_recovery.load_recovery_config(CONFIG)
    lineage = config["lineage"]
    current = dict(lineage["input_checkpoint_file_sha256"])
    parent = {
        "barun_config.json": current["barun_config.json"],
        "model.safetensors": lineage["input_checkpoint_parent"]["model_sha256"],
        "tokenizer.json": current["tokenizer.json"],
    }
    return {
        "source_run_id": lineage["input_run_id"],
        "parent_run_id": lineage["input_checkpoint_parent"]["run_id"],
        "path": path,
        "checkpoint_manifest_sha256": recovery_runner.PRESTO_CHECKPOINT_MANIFEST_SHA256,
        "file_sha256": current,
        "parent_file_sha256": parent,
    }


def _attempt_payload(
    *,
    run_id: str,
    snapshot_path: str,
    snapshot_sha256: str,
    runner_sha256: str,
    checkpoint: str,
    machine_id: int = 999_001,
) -> dict[str, Any]:
    return {
        "schema_version": recovery_runner.ATTEMPT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "frozen_before_remote_data_prep_training_or_scoring",
        "registered_at": "2026-08-03T22:00:00+05:30",
        "created_before_remote_launch_or_data_prep": True,
        "hypothesis": "The frozen one-shot recovery can restore Mobile and retain PRESTO.",
        "decision": "Keep only if the conjunctive frozen development gates pass.",
        "recovery_config": {
            "path": "configs/continual_recovery_v1.json",
            "sha256": continual_recovery.RECOVERY_CONFIG_SHA256,
        },
        "shared_selector": {
            "path": recovery_runner.SHARED_SELECTOR_RELATIVE_PATH,
            "sha256": recovery_runner.SHARED_SELECTOR_SHA256,
        },
        "source_snapshot": {"path": snapshot_path, "sha256": snapshot_sha256},
        "runner": {
            "path": "scripts/run_continual_recovery.py",
            "sha256": runner_sha256,
        },
        "input_checkpoint": _checkpoint_contract(checkpoint),
        "prelaunch_inventory": {
            "captured_before_project_instance_creation": True,
            "protected_machine_ids": [463058, 700_001],
            "project_machine_id": machine_id,
            "fresh_project_instance": True,
        },
        "evaluation": {
            "terminal_rounds": 1,
            "mobile_development_rows": 756,
            "presto_development_rows": 14_288,
            "official_mobile_evaluation_rows_read": 0,
            "official_presto_test_rows_read": 0,
        },
        "claim_limits": ["Public development recovery evidence only."],
    }


def _snapshot_payload(
    *,
    run_id: str,
    tree: dict[str, Any],
    exact_exclusions: list[str],
    runner_sha256: str,
    checkpoint: str,
) -> dict[str, Any]:
    return {
        "schema_version": recovery_runner.SNAPSHOT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_before_remote_launch_or_data_prep": True,
        "stage_root": ".",
        "content_tree": {
            "schema_version": recovery_runner.CONTENT_TREE_SCHEMA_VERSION,
            "sha256": tree["sha256"],
            "file_count": tree["file_count"],
            "content_bytes": tree["content_bytes"],
            "hash_method": recovery_runner.CONTENT_TREE_HASH_METHOD,
            "excluded_exact_paths": exact_exclusions,
            "excluded_directory_names": list(recovery_runner.TREE_EXCLUDED_DIRECTORY_NAMES),
            "excluded_directory_suffixes": list(recovery_runner.TREE_EXCLUDED_DIRECTORY_SUFFIXES),
            "excluded_file_names": list(recovery_runner.TREE_EXCLUDED_FILE_NAMES),
            "excluded_file_suffixes": list(recovery_runner.TREE_EXCLUDED_FILE_SUFFIXES),
        },
        "frozen_files": {
            "configs/continual_recovery_v1.json": continual_recovery.RECOVERY_CONFIG_SHA256,
            recovery_runner.SHARED_SELECTOR_RELATIVE_PATH: (recovery_runner.SHARED_SELECTOR_SHA256),
            "scripts/run_continual_recovery.py": runner_sha256,
        },
        "input_checkpoint": _checkpoint_contract(checkpoint),
        "staging_policy": {
            "credentials_present": False,
            "cache_directories_present_before_launch": False,
            "symlinks_present_before_launch": False,
            "official_mobile_evaluation_present": False,
            "official_presto_test_present": False,
        },
    }


def _mobile(sample_id: str) -> dict[str, object]:
    return {
        "schema_version": "barun-sft-example-v1",
        "id": sample_id,
        "prompt": f"<bos><system>\nACTION_IR_V1\n<user>\n{sample_id}\n<assistant>\n",
        "target": '{"decision":"ABSTAIN"}',
        "metadata": {"derived_split": "train", "source_split": "train"},
    }


def _presto(
    sample_id: str,
    *,
    phenomenon: str,
    decision: str = "CALL",
    split: str = "train",
    content: str | None = None,
    contextual: bool = False,
) -> dict[str, object]:
    context = {"contacts": ["Asha"] if contextual else [], "lists": [], "notes": []}
    dialogue = [{"assistant": "Which?", "user": "That one"}] if contextual else []
    user = sample_id if content is None else content
    prompt = (
        "<bos><system>\nACTION_IR_V1\nPRESTO_CONTEXT_V1\n"
        f"CONTEXT {json.dumps(context, sort_keys=True, separators=(',', ':'))}\n"
        f"DIALOGUE {json.dumps(dialogue, sort_keys=True, separators=(',', ':'))}\n"
        f"<user>\n{user}\n<assistant>\n"
    )
    return {
        "schema_version": "barun-sft-example-v1",
        "id": sample_id,
        "prompt": prompt,
        "target": '{"decision":"ABSTAIN"}',
        "metadata": {
            "derived_split": split,
            "linguistic_phenomenon": phenomenon,
            "policy_decision": decision,
            "source_split": split,
        },
    }


def test_recovery_config_freezes_one_no_dev_tuning_recipe_and_exact_lineage() -> None:
    config = continual_recovery.load_recovery_config(CONFIG)

    assert sha256_file(CONFIG) == continual_recovery.RECOVERY_CONFIG_SHA256
    assert config["status"] == "preregistered_unlaunched"
    assert config["lineage"]["input_run_id"] == "20260803-1920-presto-stage-s17"
    assert config["lineage"]["input_checkpoint_file_sha256"]["model.safetensors"] == (
        "83c78952e719456574d2bb808fb486d0320519dd4f5df3c67736ae17931ee357"
    )
    assert config["recovery_view"]["mobile_rows"] == 7_937
    assert config["recovery_view"]["presto_replay_rows"] == 2_646
    assert config["optimization"]["learning_rate"] == 5e-5
    assert config["optimization"]["expected_optimizer_steps"] == 168
    assert config["optimization"]["checkpoint_policy"] == "final_only"
    assert config["optimization"]["development_loss_evaluation"] is False
    assert config["terminal_evaluation"]["checkpoint_evaluations"] == 1
    assert config["terminal_evaluation"]["official_mobile_evaluation_rows_read"] == 0
    assert config["terminal_evaluation"]["official_presto_test_rows_read"] == 0


def test_recovery_config_tampering_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "continual-recovery.json"
    changed.write_bytes(CONFIG.read_bytes() + b"\n")

    with pytest.raises(continual_recovery.RecoveryError, match="SHA-256 mismatch"):
        continual_recovery.load_recovery_config(changed)


def test_recovery_view_is_deterministic_train_only_and_full_revision_family(
    tmp_path: Path,
) -> None:
    mobile_rows = [_mobile(f"mobile-{index}") for index in range(8)]
    mobile_path = tmp_path / "mobile.jsonl"
    mobile_hash = _write_manifest(mobile_path, mobile_rows)
    raw_tags = sorted(PRESTO_USER_REVISION_RAW_LABELS_V2)
    presto_rows = [
        _presto(f"revision-{tag}", phenomenon=tag, contextual=index % 2 == 0)
        for index, tag in enumerate(raw_tags)
    ]
    presto_rows.extend(
        [
            _presto("disfluency", phenomenon="disfluency"),
            _presto("abstain", phenomenon="", decision="ABSTAIN"),
            _presto("confirm", phenomenon="", decision="CONFIRM"),
            _presto("extra-1", phenomenon="correct-argument"),
            _presto("extra-2", phenomenon="correct-argument"),
            _presto("extra-3", phenomenon="disfluency"),
            _presto("extra-4", phenomenon="", decision="ABSTAIN"),
            _presto(
                "train-dev-duplicate",
                phenomenon="correct-action",
                content="duplicate-content",
            ),
        ]
    )
    presto_train = tmp_path / "presto-train.jsonl"
    presto_train_hash = _write_manifest(presto_train, presto_rows)
    presto_dev_rows = [
        _presto(
            "dev-duplicate",
            phenomenon="correct-action",
            split="dev",
            content="duplicate-content",
        )
    ]
    presto_dev = tmp_path / "presto-dev.jsonl"
    presto_dev_hash = _write_manifest(presto_dev, presto_dev_rows)

    manifests: list[bytes] = []
    audits: list[dict[str, object]] = []
    for suffix in ("a", "b"):
        output = tmp_path / f"recovery-{suffix}.jsonl"
        audit_path = tmp_path / f"recovery-{suffix}-audit.json"
        audit = continual_recovery.materialize_recovery_view(
            mobile_train_path=mobile_path,
            mobile_train_sha256=mobile_hash,
            mobile_train_rows=8,
            mobile_membership_sha256=_membership([f"mobile-{index}" for index in range(8)]),
            presto_train_path=presto_train,
            presto_train_sha256=presto_train_hash,
            presto_train_rows=len(presto_rows),
            presto_dev_path=presto_dev,
            presto_dev_sha256=presto_dev_hash,
            presto_dev_rows=1,
            replay_rows=8,
            seed=17,
            output_manifest=output,
            output_audit=audit_path,
        )
        manifests.append(output.read_bytes())
        audits.append(audit)
        output_rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(output_rows) == 16
        assert len({row["id"] for row in output_rows}) == 16
        assert sum(row["id"].startswith("recovery-presto-") for row in output_rows) == 8
        assert audit["presto"]["development_correctness_used"] is False
        assert audit["presto"]["exact_prompt_target_train_rows_excluded"] == 1
        assert set(audit["presto"]["selected_user_revision_raw_tag_counts"]) == set(raw_tags)
        assert audit["presto"]["official_test_rows_read"] == 0
    assert manifests[0] == manifests[1]
    assert (
        audits[0]["presto"]["replay_source_membership_sha256"]
        == audits[1]["presto"]["replay_source_membership_sha256"]
    )


def _presto_metrics(*, false_call: float = 0.01) -> dict[str, object]:
    raw = {
        tag: {
            "count": 1,
            "ast_exact_match": {"numerator": 1, "denominator": 1, "value": 1.0},
        }
        for tag in PRESTO_USER_REVISION_RAW_LABELS_V2
    }
    return {
        "schema_version": PRESTO_SCORER_VERSION,
        "sample_count": 14_288,
        "ast_exact_match": {"value": 0.70},
        "schema_valid": {"value": 0.95},
        "abstention": {"f1": 0.80},
        "false_call_on_gate": {"value": false_call},
        "headline_phenomenon_buckets": {
            "no_phenomenon": {"count": 10, "ast_exact_match": {"value": 0.81}},
            "revision": {"count": 10, "ast_exact_match": {"value": 0.72}},
            "disfluency": {"count": 10, "ast_exact_match": {"value": 0.72}},
        },
        "per_phenomenon": raw,
    }


def test_corrected_presto_recovery_gate_is_conjunctive_and_boundary_safe() -> None:
    thresholds = continual_recovery.load_recovery_config(CONFIG)["terminal_evaluation"]["presto"][
        "gates"
    ]
    passed = continual_recovery.evaluate_recovery_presto_gate(_presto_metrics(), thresholds)

    assert passed["passed"] is True
    assert all(passed["checks"].values())
    assert set(passed["observed"]["revision_raw_tag_metrics"]) == set(
        PRESTO_USER_REVISION_RAW_LABELS_V2
    )

    failed = continual_recovery.evaluate_recovery_presto_gate(
        _presto_metrics(false_call=0.010001), thresholds
    )
    assert failed["passed"] is False
    assert failed["checks"]["safety_false_call_on_gate"] is False


def test_recovery_schedule_is_fixed_conservative_warmup_and_cosine_floor() -> None:
    optimization = continual_recovery.load_recovery_config(CONFIG)["optimization"]

    assert continual_recovery._learning_rate(
        1, total_steps=168, optimization=optimization
    ) == pytest.approx(5e-5 / 16)
    assert continual_recovery._learning_rate(
        16, total_steps=168, optimization=optimization
    ) == pytest.approx(5e-5)
    assert continual_recovery._learning_rate(
        168, total_steps=168, optimization=optimization
    ) == pytest.approx(5e-6)


def test_attempt_and_snapshot_schemas_are_strict() -> None:
    attempt = _attempt_payload(
        run_id="20260803-2200-continual-recovery-s17",
        snapshot_path="source-snapshot.json",
        snapshot_sha256="a" * 64,
        runner_sha256="b" * 64,
        checkpoint="checkpoint",
    )
    normalized_attempt = recovery_runner._validate_attempt_schema(attempt)
    assert normalized_attempt["source_snapshot"]["sha256"] == "a" * 64
    assert normalized_attempt["shared_selector"] == {
        "path": "configs/parallel_rescue_selection_v1.json",
        "sha256": recovery_runner.SHARED_SELECTOR_SHA256,
    }
    assert normalized_attempt["prelaunch_inventory"]["project_machine_id"] == 999_001

    changed_attempt = dict(attempt)
    changed_attempt["unregistered"] = True
    with pytest.raises(continual_recovery.RecoveryError, match="fields changed"):
        recovery_runner._validate_attempt_schema(changed_attempt)

    missing_protected = json.loads(json.dumps(attempt))
    missing_protected["prelaunch_inventory"]["protected_machine_ids"] = [700_001]
    with pytest.raises(continual_recovery.RecoveryError, match="463058"):
        recovery_runner._validate_attempt_schema(missing_protected)

    preexisting_project = json.loads(json.dumps(attempt))
    preexisting_project["prelaunch_inventory"]["protected_machine_ids"].append(999_001)
    with pytest.raises(continual_recovery.RecoveryError, match="pre-existing and is protected"):
        recovery_runner._validate_attempt_schema(preexisting_project)

    tree = {"sha256": "c" * 64, "file_count": 2, "content_bytes": 10}
    snapshot = _snapshot_payload(
        run_id="20260803-2200-continual-recovery-s17",
        tree=tree,
        exact_exclusions=["attempt-preregistration.json", "source-snapshot.json"],
        runner_sha256="b" * 64,
        checkpoint="checkpoint",
    )
    normalized_snapshot = recovery_runner._validate_snapshot_schema(snapshot)
    assert normalized_snapshot["content_tree"]["sha256"] == "c" * 64
    assert (
        normalized_snapshot["frozen_files"]["configs/parallel_rescue_selection_v1.json"]
        == recovery_runner.SHARED_SELECTOR_SHA256
    )

    changed_snapshot = json.loads(json.dumps(snapshot))
    changed_snapshot["content_tree"]["excluded_directory_names"].append("secret-hideout")
    with pytest.raises(continual_recovery.RecoveryError, match="excluded_directory_names"):
        recovery_runner._validate_snapshot_schema(changed_snapshot)


def test_content_tree_excludes_only_documented_runtime_paths_and_rejects_credentials(
    tmp_path: Path,
) -> None:
    (tmp_path / "source.txt").write_text("frozen\n", encoding="utf-8")
    cache = tmp_path / ".venv"
    cache.mkdir()
    (cache / "runtime.bin").write_bytes(b"ignored runtime environment")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "source.pyc").write_bytes(b"ignored bytecode")
    exact = ["attempt-preregistration.json", "source-snapshot.json"]

    first = recovery_runner._recompute_staged_content_tree(tmp_path, excluded_exact_paths=exact)
    assert first["file_count"] == 1
    assert first["content_bytes"] == len(b"frozen\n")
    assert ".venv/" in first["excluded_runtime_paths"]

    (tmp_path / "source.txt").write_text("changed\n", encoding="utf-8")
    second = recovery_runner._recompute_staged_content_tree(tmp_path, excluded_exact_paths=exact)
    assert second["sha256"] != first["sha256"]

    (tmp_path / ".env").write_text("TOKEN=must-not-enter-tree\n", encoding="utf-8")
    with pytest.raises(continual_recovery.RecoveryError, match="credential-like"):
        recovery_runner._recompute_staged_content_tree(tmp_path, excluded_exact_paths=exact)


@pytest.mark.skipif(
    not all(path.is_file() for path in PRESTO_CHECKPOINT_FILES),
    reason="requires the downloaded W&B PRESTO checkpoint evidence",
)
def test_actual_presto_checkpoint_manifest_parent_chain_is_frozen() -> None:
    config = continual_recovery.load_recovery_config(CONFIG)
    checkpoint = PRESTO_CHECKPOINT
    contract = _checkpoint_contract(
        "experiments/runs/20260803-1920-presto-stage-s17/essential/checkpoint"
    )

    lineage = recovery_runner._validate_checkpoint_parent_chain(
        checkpoint_dir=checkpoint,
        config=config,
        contract=contract,
    )
    assert lineage["source_run_id"] == "20260803-1920-presto-stage-s17"
    assert lineage["parent_run_id"] == "20260803-1845-mobile-followup-retry-s17"
    assert lineage["selection_step"] == 947

    changed = json.loads(json.dumps(contract))
    changed["parent_file_sha256"]["model.safetensors"] = "0" * 64
    with pytest.raises(continual_recovery.RecoveryError, match="parent checkpoint files"):
        recovery_runner._validate_checkpoint_parent_chain(
            checkpoint_dir=checkpoint,
            config=config,
            contract=changed,
        )


def test_actual_shared_selector_is_hash_pinned_and_has_expected_schema() -> None:
    assert sha256_file(SHARED_SELECTOR) == recovery_runner.SHARED_SELECTOR_SHA256
    selector = json.loads(SHARED_SELECTOR.read_text(encoding="utf-8"))
    assert selector["schema_version"] == recovery_runner.SHARED_SELECTOR_SCHEMA_VERSION


def test_end_to_end_provenance_validation_binds_tree_and_enters_essential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "20260803-2200-continual-recovery-s17"
    stage = tmp_path / "stage"
    (stage / "configs").mkdir(parents=True)
    (stage / "scripts").mkdir()
    shutil.copy2(CONFIG, stage / "configs" / "continual_recovery_v1.json")
    shutil.copy2(
        SHARED_SELECTOR,
        stage / "configs" / "parallel_rescue_selection_v1.json",
    )
    shutil.copy2(
        ROOT / "scripts" / "run_continual_recovery.py",
        stage / "scripts" / "run_continual_recovery.py",
    )
    checkpoint_relative = "checkpoint"
    checkpoint = stage / checkpoint_relative
    checkpoint.mkdir()
    for name in (
        "barun_config.json",
        "checkpoint_manifest.json",
        "model.safetensors",
        "tokenizer.json",
    ):
        (checkpoint / name).write_text(f"synthetic {name}\n", encoding="utf-8")
    (stage / "data.txt").write_text("pinned data\n", encoding="utf-8")
    attempt_path = stage / "attempt-preregistration.json"
    snapshot_path = stage / "source-snapshot.json"
    exact = sorted([attempt_path.name, snapshot_path.name])
    tree = recovery_runner._recompute_staged_content_tree(stage, excluded_exact_paths=exact)
    runner_path = stage / "scripts" / "run_continual_recovery.py"
    runner_sha256 = sha256_file(runner_path)
    snapshot = _snapshot_payload(
        run_id=run_id,
        tree=tree,
        exact_exclusions=exact,
        runner_sha256=runner_sha256,
        checkpoint=checkpoint_relative,
    )
    snapshot_path.write_text(json.dumps(snapshot, sort_keys=True) + "\n", encoding="utf-8")
    attempt = _attempt_payload(
        run_id=run_id,
        snapshot_path=snapshot_path.name,
        snapshot_sha256=sha256_file(snapshot_path),
        runner_sha256=runner_sha256,
        checkpoint=checkpoint_relative,
    )
    attempt_path.write_text(json.dumps(attempt, sort_keys=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(recovery_runner, "REPOSITORY_ROOT", stage)
    monkeypatch.setattr(
        recovery_runner,
        "RECOVERY_CONFIG",
        stage / "configs" / "continual_recovery_v1.json",
    )
    monkeypatch.setattr(
        recovery_runner,
        "SHARED_SELECTOR_PATH",
        stage / "configs" / "parallel_rescue_selection_v1.json",
    )
    monkeypatch.setattr(recovery_runner, "RUNNER_PATH", runner_path)
    monkeypatch.setattr(
        recovery_runner,
        "_validate_checkpoint_parent_chain",
        lambda **_: {"synthetic_parent_chain": True},
    )
    provenance = recovery_runner.validate_recovery_provenance(
        run_id=run_id,
        attempt_path=attempt_path,
        snapshot_path=snapshot_path,
        input_checkpoint=checkpoint,
        config=continual_recovery.load_recovery_config(CONFIG),
        jarvis_machine_id=999_001,
    )
    assert provenance["validated_before_data_prep_training_or_scoring"] is True
    assert provenance["content_tree"]["sha256"] == tree["sha256"]
    assert provenance["shared_selector"]["sha256"] == recovery_runner.SHARED_SELECTOR_SHA256
    assert provenance["prelaunch_inventory"]["protected_machine_ids"] == [463058, 700_001]

    essential = tmp_path / "essential"
    essential.mkdir()
    recovery_runner._copy_provenance_evidence(
        essential=essential,
        attempt_path=attempt_path,
        snapshot_path=snapshot_path,
        provenance=provenance,
    )
    manifest = recovery_runner._tree_manifest(essential)["file_sha256"]
    assert "attempt-preregistration.json" in manifest
    assert "source-snapshot.json" in manifest
    assert "provenance-validation.json" in manifest

    wrong_selector_attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    wrong_selector_attempt["shared_selector"]["sha256"] = "f" * 64
    attempt_path.write_text(
        json.dumps(wrong_selector_attempt, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(continual_recovery.RecoveryError, match="shared-selector identity"):
        recovery_runner.validate_recovery_provenance(
            run_id=run_id,
            attempt_path=attempt_path,
            snapshot_path=snapshot_path,
            input_checkpoint=checkpoint,
            config=continual_recovery.load_recovery_config(CONFIG),
            jarvis_machine_id=999_001,
        )
    attempt_path.write_text(json.dumps(attempt, sort_keys=True) + "\n", encoding="utf-8")

    staged_selector = stage / "configs" / "parallel_rescue_selection_v1.json"
    staged_selector.write_text("{}\n", encoding="utf-8")
    with pytest.raises(continual_recovery.RecoveryError, match="shared rescue selector SHA-256"):
        recovery_runner.validate_recovery_provenance(
            run_id=run_id,
            attempt_path=attempt_path,
            snapshot_path=snapshot_path,
            input_checkpoint=checkpoint,
            config=continual_recovery.load_recovery_config(CONFIG),
            jarvis_machine_id=999_001,
        )
    shutil.copy2(SHARED_SELECTOR, staged_selector)

    with pytest.raises(continual_recovery.RecoveryError, match="pre-existing and protected"):
        recovery_runner.validate_recovery_provenance(
            run_id=run_id,
            attempt_path=attempt_path,
            snapshot_path=snapshot_path,
            input_checkpoint=checkpoint,
            config=continual_recovery.load_recovery_config(CONFIG),
            jarvis_machine_id=463058,
        )

    with pytest.raises(continual_recovery.RecoveryError, match="differs from the CLI"):
        recovery_runner.validate_recovery_provenance(
            run_id=run_id,
            attempt_path=attempt_path,
            snapshot_path=snapshot_path,
            input_checkpoint=checkpoint,
            config=continual_recovery.load_recovery_config(CONFIG),
            jarvis_machine_id=999_002,
        )

    (stage / "data.txt").write_text("post-snapshot mutation\n", encoding="utf-8")
    with pytest.raises(continual_recovery.RecoveryError, match="content-tree"):
        recovery_runner.validate_recovery_provenance(
            run_id=run_id,
            attempt_path=attempt_path,
            snapshot_path=snapshot_path,
            input_checkpoint=checkpoint,
            config=continual_recovery.load_recovery_config(CONFIG),
            jarvis_machine_id=999_001,
        )


def test_recovery_runner_requires_both_provenance_inputs() -> None:
    parser = recovery_runner.build_parser()
    actions = {option: action for action in parser._actions for option in action.option_strings}

    assert actions["--attempt-preregistration"].required is True
    assert actions["--source-snapshot"].required is True


def test_run_checks_machine_provenance_before_runtime_or_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    attempt = stage / "attempt-preregistration.json"
    snapshot = stage / "source-snapshot.json"
    attempt.write_text("{}\n", encoding="utf-8")
    snapshot.write_text("{}\n", encoding="utf-8")

    def reject_provenance(**kwargs: object) -> dict[str, Any]:
        assert kwargs["jarvis_machine_id"] == 999_001
        raise continual_recovery.RecoveryError("inventory rejected before runtime")

    def forbidden_after_provenance(*args: object, **kwargs: object) -> object:
        raise AssertionError("runtime or data work happened before provenance validation")

    monkeypatch.setattr(recovery_runner, "validate_recovery_provenance", reject_provenance)
    monkeypatch.setattr(recovery_runner, "_runtime_environment", forbidden_after_provenance)
    monkeypatch.setattr(recovery_runner, "_run_repository_tests", forbidden_after_provenance)
    monkeypatch.setattr(recovery_runner, "materialize_recovery_view", forbidden_after_provenance)
    args = SimpleNamespace(
        run_id="20260803-2200-continual-recovery-s17",
        jarvis_machine_id=999_001,
        attempt_preregistration=attempt,
        source_snapshot=snapshot,
        input_checkpoint=stage / "checkpoint",
        artifact_root=tmp_path / "artifacts",
    )

    with pytest.raises(continual_recovery.RecoveryError, match="inventory rejected before runtime"):
        recovery_runner.run(args)
