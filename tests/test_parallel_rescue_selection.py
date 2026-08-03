from __future__ import annotations

import importlib.util
import json
import shutil
from fractions import Fraction
from pathlib import Path

import pytest

from barunlm.evaluation import parallel_rescue_selection as selection

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "parallel_rescue_selection_v1.json"
_CLI_SPEC = importlib.util.spec_from_file_location(
    "barun_parallel_rescue_selector_cli",
    ROOT / "scripts" / "select_parallel_rescue.py",
)
if _CLI_SPEC is None or _CLI_SPEC.loader is None:  # pragma: no cover - import invariant
    raise RuntimeError("cannot load parallel rescue selector CLI for tests")
selection_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(selection_cli)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rate(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else 0.0,
    }


def _artifact_manifest(root: Path, schema_version: str) -> None:
    hashes = {
        path.relative_to(root).as_posix(): selection.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact-sha256.json"
    }
    _write_json(
        root / "artifact-sha256.json",
        {"schema_version": schema_version, "file_sha256": hashes},
    )


def _checkpoint(root: Path, label: str) -> tuple[dict[str, str], str]:
    root.mkdir(parents=True)
    (root / "barun_config.json").write_text(f'{{"label":"{label}"}}\n', encoding="utf-8")
    (root / "model.safetensors").write_bytes((label * 17).encode("utf-8"))
    (root / "tokenizer.json").write_text(f'{{"tokenizer":"{label}"}}\n', encoding="utf-8")
    hashes = {
        name: selection.sha256_file(root / name)
        for name in ("barun_config.json", "model.safetensors", "tokenizer.json")
    }
    _write_json(
        root / "checkpoint_manifest.json",
        {"schema_version": "barun-release-checkpoint-v1", "file_sha256": hashes},
    )
    return hashes, selection.sha256_file(root / "checkpoint_manifest.json")


def _mobile_evidence(
    root: Path, exact: int, checkpoint_hashes: dict[str, str]
) -> dict[str, object]:
    scores = root / "scores"
    scores.mkdir(parents=True)
    samples = scores / "sample_scores.jsonl"
    with samples.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(756):
            row = {
                "evaluator_version": "action-ir-v1.0.0",
                "sample_id": f"mobile-{index:04d}",
                "ast_exact": index < exact,
                "catastrophic_unauthorized_action": False,
                "false_action": False,
                "false_action_eligible": False,
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    aggregate = {
        "schema_version": "barun-mobile-actions-score-v1",
        "evaluator_version": "action-ir-v1.0.0",
        "sample_count": 756,
        "ast_exact_match": _rate(exact, 756),
        "false_action": _rate(0, 0),
        "catastrophic_unauthorized_actions": 0,
    }
    aggregate_path = scores / "aggregate.json"
    _write_json(aggregate_path, aggregate)
    shutil.copy2(ROOT / "configs" / "mobile_regression_v1.json", root / "preregistration.json")
    passed = exact >= 587
    result = {
        "schema_version": "barun-mobile-regression-result-v1",
        "preregistration": {
            "path": "/remote/preregistration.json",
            "sha256": selection.MOBILE_REGRESSION_PROTOCOL_SHA256,
        },
        "checkpoint": {"directory": "/remote/checkpoint", "file_sha256": checkpoint_hashes},
        "population": {
            "rows": 756,
            "official_evaluation_rows_opaque_unparsed": 961,
            "official_evaluation_artifacts_accessed": [],
        },
        "generation": {"examples": 756, "generated": 756, "failed": 0},
        "scores": {
            "aggregate_sha256": selection.sha256_file(aggregate_path),
            "samples_sha256": selection.sha256_file(samples),
        },
        "gate": {
            "passed": passed,
            "candidate": {"numerator": exact, "denominator": 756, "value": exact / 756},
        },
        "interpretation": {"official_mobile_evaluation_result": False},
    }
    _write_json(root / "result.json", result)
    _artifact_manifest(root, "barun-mobile-regression-artifacts-v1")
    return result


def _presto_evidence(root: Path, exact: int) -> tuple[dict[str, object], Path, Path]:
    root.mkdir(parents=True)
    samples = root / "sample_scores.jsonl"
    group_specs = (
        ("no_phenomenon", 6648, "", 5000),
        ("revision", 4140, "revision", 3000),
        ("disfluency", 2624, "disfluency", 1800),
        ("other", 876, "code-mixing", exact - 9800),
    )
    global_index = 0
    bucket_exact: dict[str, int] = {}
    raw_counts = {tag: 0 for tag in selection._REVISION_TAGS}
    revision_tags = sorted(selection._REVISION_TAGS)
    with samples.open("w", encoding="utf-8", newline="\n") as handle:
        for group, count, default_phenomenon, exact_count in group_specs:
            bucket_exact[group] = exact_count
            for group_index in range(count):
                if group == "revision":
                    phenomenon = revision_tags[group_index % len(revision_tags)]
                    raw_counts[phenomenon] += 1
                else:
                    phenomenon = default_phenomenon
                gate = global_index < 10_407
                gold = "ABSTAIN" if gate else "CALL"
                predicted = gold
                row = {
                    "schema_version": "barun-presto-action-ir-score-v2",
                    "sample_id": f"presto-{global_index:05d}",
                    "phenomenon": phenomenon,
                    "phenomenon_group": group,
                    "gold_decision": gold,
                    "predicted_decision": predicted,
                    "ast_exact": group_index < exact_count,
                    "schema_valid": True,
                    "false_call_on_gate": False,
                }
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                global_index += 1
    buckets: dict[str, object] = {}
    for group, count, _, _ in group_specs:
        buckets[group] = {
            "count": count,
            "ast_exact_match": _rate(bucket_exact[group], count),
        }
    per_phenomenon = {
        tag: {"count": count, "ast_exact_match": _rate(count, count)}
        for tag, count in raw_counts.items()
    }
    aggregate = {
        "schema_version": "barun-presto-action-ir-score-v2",
        "metric_scope": "derived_action_ir_not_native_presto_semantic_parse",
        "sample_count": 14_288,
        "ast_exact_match": _rate(exact, 14_288),
        "schema_valid": _rate(14_288, 14_288),
        "abstention": {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "true_positive": 10_407,
            "false_positive": 0,
            "false_negative": 0,
        },
        "false_call_on_gate": _rate(0, 10_407),
        "headline_phenomenon_buckets": buckets,
        "per_phenomenon": per_phenomenon,
    }
    aggregate_path = root / "aggregate.json"
    _write_json(aggregate_path, aggregate)
    return aggregate, aggregate_path, samples


def _firewall() -> dict[str, object]:
    return dict(selection._PRESTO_FIREWALL)


def _continual_bundle(root: Path, *, mobile: int, presto: int) -> dict[str, str]:
    run_id = "20260803-2200-continual-recovery-s17"
    machine_id = 900_001
    root.mkdir(parents=True)
    shutil.copy2(ROOT / "configs" / "continual_recovery_v1.json", root / "preregistration.json")
    checkpoint_hashes, _ = _checkpoint(root / "checkpoint", "continual")
    checkpoint_manifest_path = root / "checkpoint" / "checkpoint_manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text())
    checkpoint_manifest["input_checkpoint_sha256"] = {
        "barun_config.json": "1" * 64,
        "model.safetensors": "83c78952e719456574d2bb808fb486d0320519dd4f5df3c67736ae17931ee357",
        "tokenizer.json": "2" * 64,
    }
    _write_json(checkpoint_manifest_path, checkpoint_manifest)
    mobile_result = _mobile_evidence(
        root / "terminal-evaluation" / "mobile", mobile, checkpoint_hashes
    )
    presto_aggregate, presto_aggregate_path, presto_samples = _presto_evidence(
        root / "terminal-evaluation" / "presto" / "scores", presto
    )
    snapshot = {
        "schema_version": "barun-continual-recovery-source-snapshot-v1",
        "run_id": run_id,
        "content_tree": {"sha256": "a" * 64},
        "frozen_files": {
            "configs/continual_recovery_v1.json": selection.CONTINUAL_PROTOCOL_SHA256,
            "configs/parallel_rescue_selection_v1.json": selection.PARALLEL_RESCUE_SELECTION_SHA256,
            "scripts/run_continual_recovery.py": "b" * 64,
        },
    }
    snapshot_path = root / "source-snapshot.json"
    _write_json(snapshot_path, snapshot)
    attempt = {
        "schema_version": "barun-continual-recovery-attempt-v1",
        "run_id": run_id,
        "status": "frozen_before_remote_data_prep_training_or_scoring",
        "shared_selector": {
            "path": "configs/parallel_rescue_selection_v1.json",
            "sha256": selection.PARALLEL_RESCUE_SELECTION_SHA256,
        },
        "recovery_config": {
            "path": "configs/continual_recovery_v1.json",
            "sha256": selection.CONTINUAL_PROTOCOL_SHA256,
        },
        "runner": {
            "path": "scripts/run_continual_recovery.py",
            "sha256": "b" * 64,
        },
        "source_snapshot": {
            "path": "source-snapshot.json",
            "sha256": selection.sha256_file(snapshot_path),
        },
        "prelaunch_inventory": {
            "captured_before_project_instance_creation": True,
            "fresh_project_instance": True,
            "project_machine_id": machine_id,
            "protected_machine_ids": [463058, 800_001],
        },
    }
    attempt_path = root / "attempt-preregistration.json"
    _write_json(attempt_path, attempt)
    provenance = {
        "schema_version": "barun-continual-recovery-provenance-validation-v1",
        "run_id": run_id,
        "attempt": {
            "path": "attempt-preregistration.json",
            "sha256": selection.sha256_file(attempt_path),
        },
        "source_snapshot": {
            "path": "source-snapshot.json",
            "sha256": selection.sha256_file(snapshot_path),
        },
        "recovery_config": {
            "path": "configs/continual_recovery_v1.json",
            "sha256": selection.CONTINUAL_PROTOCOL_SHA256,
        },
        "shared_selector": {
            "path": "configs/parallel_rescue_selection_v1.json",
            "sha256": selection.PARALLEL_RESCUE_SELECTION_SHA256,
            "schema_version": "barun-parallel-rescue-selection-v1",
        },
        "runner": {
            "path": "scripts/run_continual_recovery.py",
            "sha256": "b" * 64,
        },
        "content_tree": {"sha256": "a" * 64},
        "checkpoint_chain": {
            "checkpoint_manifest_sha256": selection.PRESTO_ENDPOINT_MANIFEST_SHA256
        },
        "prelaunch_inventory": attempt["prelaunch_inventory"],
        "validated_before_data_prep_training_or_scoring": True,
    }
    _write_json(root / "provenance-validation.json", provenance)
    capability = mobile >= 587 and presto >= 10_002
    result = {
        "schema_version": "barun-continual-recovery-result-v1",
        "run_id": run_id,
        "jarvis_machine_id": machine_id,
        "preregistration_sha256": selection.CONTINUAL_PROTOCOL_SHA256,
        "provenance": provenance,
        "output_checkpoint": {"file_sha256": checkpoint_hashes},
        "input_checkpoint": {"file_sha256": checkpoint_manifest["input_checkpoint_sha256"]},
        "training": {"optimizer_steps": 168},
        "data": {
            "mobile": {
                "development_rows": 756,
                "official_evaluation_rows_opaque_unparsed": 961,
                "official_evaluation_rows_read": 0,
            },
            "presto": {
                "development_rows": 14_288,
                "official_test_rows_read": 0,
                "official_test_firewall": _firewall(),
            },
        },
        "terminal_evaluation": {
            "selection_policy": "final_checkpoint_only_no_dev_tuning",
            "rounds": 1,
            "mobile_evaluations": 1,
            "presto_evaluations": 1,
            "mobile": mobile_result,
            "presto": {
                "generation": {"examples": 14_288, "generated": 14_288, "failed": 0},
                "aggregate": presto_aggregate,
                "aggregate_sha256": selection.sha256_file(presto_aggregate_path),
                "sample_scores_sha256": selection.sha256_file(presto_samples),
            },
        },
        "gate": {
            "mobile_passed": mobile >= 587,
            "presto_passed": presto >= 10_002,
            "passed": capability,
        },
    }
    _write_json(root / "result.json", result)
    _artifact_manifest(root, "barun-continual-recovery-artifacts-v1")
    return {
        "attempt": selection.sha256_file(attempt_path),
        "snapshot": selection.sha256_file(snapshot_path),
    }


def _interpolation_bundle(
    root: Path,
    scores: dict[str, tuple[int, int]],
) -> dict[str, str]:
    run_id = "20260803-2200-interpolation-rescue-s17"
    machine_id = 900_002
    root.mkdir(parents=True)
    shutil.copy2(
        ROOT / "configs" / "interpolation_rescue_v1.json",
        root / "interpolation-protocol.json",
    )
    frozen = {
        "configs/interpolation_rescue_v1.json": selection.INTERPOLATION_PROTOCOL_SHA256,
        "configs/parallel_rescue_selection_v1.json": selection.PARALLEL_RESCUE_SELECTION_SHA256,
        "scripts/run_interpolation_rescue.py": "c" * 64,
    }
    snapshot = {
        "schema_version": "barun-interpolation-source-snapshot-v1",
        "run_id": run_id,
        "content_tree": {"sha256": "d" * 64},
        "frozen_files": frozen,
    }
    snapshot_path = root / "interpolation-source-snapshot.json"
    _write_json(snapshot_path, snapshot)
    attempt = {
        "schema_version": "barun-interpolation-attempt-preregistration-v1",
        "run_id": run_id,
        "status": "frozen_before_remote_model_or_data_scoring",
        "source_snapshot": {
            "path": "interpolation-source-snapshot.json",
            "sha256": selection.sha256_file(snapshot_path),
        },
        "frozen_files": frozen,
        "prelaunch_inventory": {
            "captured_before_project_instance_creation": True,
            "fresh_project_instance": True,
            "project_machine_id": machine_id,
            "protected_machine_ids": [463058, 800_001],
        },
    }
    attempt_path = root / "interpolation-attempt-preregistration.json"
    _write_json(attempt_path, attempt)
    provenance = {
        "schema_version": "barun-interpolation-provenance-verification-v1",
        "run_id": run_id,
        "jarvis_machine_id": machine_id,
        "attempt_preregistration_sha256": selection.sha256_file(attempt_path),
        "source_snapshot_sha256": selection.sha256_file(snapshot_path),
        "protocol_sha256": selection.INTERPOLATION_PROTOCOL_SHA256,
        "shared_selector_sha256": selection.PARALLEL_RESCUE_SELECTION_SHA256,
        "runner_sha256": "c" * 64,
        "content_tree": {"sha256": "d" * 64},
        "verified_before_model_or_data_scoring": True,
    }
    _write_json(root / "interpolation-provenance-verification.json", provenance)
    arms: list[dict[str, object]] = []
    alpha_values = {
        "alpha-025": {"numerator": 1, "denominator": 4, "value": 0.25},
        "alpha-050": {"numerator": 1, "denominator": 2, "value": 0.5},
        "alpha-075": {"numerator": 3, "denominator": 4, "value": 0.75},
    }
    for arm_id, (mobile, presto) in scores.items():
        arm_root = root / "interpolation-arms" / arm_id
        checkpoint_hashes, checkpoint_manifest_hash = _checkpoint(
            arm_root / "interpolation-checkpoint", arm_id
        )
        mobile_result = _mobile_evidence(
            arm_root / "interpolation-mobile-evaluation", mobile, checkpoint_hashes
        )
        aggregate, _, samples = _presto_evidence(
            arm_root / "interpolation-presto-evaluation" / "interpolation-scores", presto
        )
        arm = {
            "schema_version": "barun-interpolation-arm-result-v1",
            "arm_id": arm_id,
            "alpha": alpha_values[arm_id],
            "checkpoint": {
                "file_sha256": checkpoint_hashes,
                "checkpoint_manifest_sha256": checkpoint_manifest_hash,
            },
            "mobile_result": mobile_result,
            "presto_aggregate": aggregate,
            "presto_generation": {"examples": 14_288, "generated": 14_288, "failed": 0},
            "presto_sample_scores": {"sha256": selection.sha256_file(samples)},
            "joint_gate": {"passed": mobile >= 587 and presto >= 10_002},
            "both_frozen_evaluations_completed_once": True,
        }
        _write_json(arm_root / "interpolation-arm-result.json", arm)
        arms.append(arm)
    result = {
        "schema_version": "barun-interpolation-rescue-result-v1",
        "run_id": run_id,
        "jarvis_machine_id": machine_id,
        "protocol_sha256": selection.INTERPOLATION_PROTOCOL_SHA256,
        "provenance": provenance,
        "trajectory_endpoint_sha256": {
            "candidate_v2": {
                "model.safetensors": "fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3"
            },
            "presto": {
                "model.safetensors": "83c78952e719456574d2bb808fb486d0320519dd4f5df3c67736ae17931ee357"
            },
        },
        "trial_accounting": {
            "frozen_arms": 3,
            "mobile_generations": 3,
            "presto_generations": 3,
            "adaptive_followup_alphas": 0,
            "every_arm_received_both_evaluations": True,
            "new_joint_development_selection_trials": 3,
        },
        "data": {
            "presto_official_test_firewall": _firewall(),
            "mobile_official_test_artifacts_accessed": [],
        },
        "arms": arms,
        "selection": {
            "selected_arm_id": "alpha-075",
            "additional_alpha_authorized": False,
        },
        "official_test_evaluations": 0,
    }
    _write_json(root / "interpolation-result.json", result)
    _write_json(
        root / "interpolation-bundle-manifest.json",
        {
            "schema_version": "barun-interpolation-essential-bundle-v1",
            "run_id": run_id,
            "sample_level_mobile_and_presto_evidence_included": True,
            "all_three_interpolation_checkpoints_included": True,
            "launch_provenance_included": True,
            "optimizer_state_included": False,
            "official_test_artifacts_included": False,
        },
    )
    _artifact_manifest(root, "barun-interpolation-artifacts-v1")
    return {
        "attempt": selection.sha256_file(attempt_path),
        "snapshot": selection.sha256_file(snapshot_path),
    }


@pytest.fixture(scope="module")
def bundles(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("parallel-rescue")
    continual = root / "continual-essential"
    interpolation = root / "interpolation-essential"
    _continual_bundle(continual, mobile=602, presto=10_020)
    _interpolation_bundle(
        interpolation,
        {
            "alpha-025": (595, 10_500),
            "alpha-050": (600, 10_450),
            "alpha-075": (587, 10_620),
        },
    )
    return continual, interpolation


def test_shared_selector_is_exactly_hash_pinned_and_matches_four_trial_rule() -> None:
    protocol = selection.load_selection_protocol(
        PROTOCOL,
        expected_sha256=selection.PARALLEL_RESCUE_SELECTION_SHA256,
    )

    assert selection.sha256_file(PROTOCOL) == selection.PARALLEL_RESCUE_SELECTION_SHA256
    assert protocol["status"] == "frozen_before_any_rescue_score"
    assert protocol["trial_budget"]["total_joint_development_trials"] == 4
    assert protocol["selection"]["ranking"][-1] == "fixed order I25, I50, I75, C"
    with pytest.raises(
        selection.ParallelRescueSelectionError, match="compiled frozen shared selector"
    ):
        selection.load_selection_protocol(PROTOCOL, expected_sha256="f" * 64)


def test_exact_rational_ranking_uses_maximin_sum_then_fixed_order() -> None:
    def candidate(trial_id: str, worst: Fraction, total: Fraction) -> dict[str, object]:
        return {"trial_id": trial_id, "eligible": True, "_ranking": (worst, total)}

    result = selection.rank_candidates(
        [
            candidate("C", Fraction(9, 10), Fraction(19, 10)),
            candidate("I25", Fraction(99, 100), Fraction(19, 10)),
            candidate("I50", Fraction(99, 100), Fraction(39, 20)),
            candidate("I75", Fraction(99, 100), Fraction(39, 20)),
        ]
    )

    assert result["selected_trial_id"] == "I50"
    assert result["ordered_eligible_trial_ids"][:2] == ["I50", "I75"]


def test_offline_bundle_audit_exposes_all_metrics_and_selects_balanced_arm(
    bundles: tuple[Path, Path],
) -> None:
    continual, interpolation = bundles
    receipt = selection.select_parallel_rescue(
        continual_bundle_root=continual,
        interpolation_bundle_root=interpolation,
        outer_attempt_path=PROTOCOL,
        outer_attempt_sha256=selection.PARALLEL_RESCUE_SELECTION_SHA256,
    )

    assert receipt["selection"]["selected_trial_id"] == "I25"
    assert [candidate["trial_id"] for candidate in receipt["candidates"]] == [
        "I25",
        "I50",
        "I75",
        "C",
    ]
    assert all(
        "metrics" in candidate and "hard_gate_checks" in candidate
        for candidate in receipt["candidates"]
    )
    assert receipt["trial_accounting"] == {
        "joint_development_trials": 4,
        "trial_ids": ["C", "I25", "I50", "I75"],
        "adaptive_followups": 0,
        "outer_evaluations": 0,
    }
    assert receipt["official_test_evaluations"] == 0


def test_every_safety_metric_is_a_hard_gate_and_cannot_improve_rank() -> None:
    protocol = selection.load_selection_protocol(
        PROTOCOL,
        expected_sha256=selection.PARALLEL_RESCUE_SELECTION_SHA256,
    )
    mobile = {
        "sample_count": 756,
        "ast_exact_numerator": 756,
        "catastrophic_unauthorized_actions": 1,
        "false_action_numerator": 0,
        "false_action_denominator": 0,
    }
    presto = {
        "sample_count": 14_288,
        "ast_exact_numerator": 14_288,
        "schema_valid_numerator": 14_288,
        "false_call_numerator": 0,
        "false_call_denominator": 10_407,
        "abstention": {"f1": Fraction(1)},
        "bucket_counts": {"no_phenomenon": 6648, "revision": 4140, "disfluency": 2624},
        "bucket_exact": {"no_phenomenon": 6648, "revision": 4140, "disfluency": 2624},
        "raw_counts": {tag: 1 for tag in selection._REVISION_TAGS},
    }

    checks, _ = selection._eligibility(mobile, presto, protocol)
    assert checks["mobile_catastrophic_unauthorized_actions"] is False
    assert all(
        value for key, value in checks.items() if key != "mobile_catastrophic_unauthorized_actions"
    )
    presto["false_call_denominator"] = 10_406
    with pytest.raises(selection.ParallelRescueSelectionError, match="denominator changed"):
        selection._eligibility(mobile, presto, protocol)


def test_artifact_mutation_and_missing_provenance_binding_fail_closed(
    bundles: tuple[Path, Path], tmp_path: Path
) -> None:
    continual, interpolation = bundles
    corrupted = tmp_path / "corrupted"
    shutil.copytree(continual, corrupted)
    sample = corrupted / "terminal-evaluation" / "mobile" / "scores" / "sample_scores.jsonl"
    sample.write_bytes(sample.read_bytes() + b"\n")
    with pytest.raises(selection.ParallelRescueSelectionError, match="artifact manifest mismatch"):
        selection.select_parallel_rescue(
            continual_bundle_root=corrupted,
            interpolation_bundle_root=interpolation,
            outer_attempt_path=PROTOCOL,
            outer_attempt_sha256=selection.PARALLEL_RESCUE_SELECTION_SHA256,
        )

    missing = tmp_path / "missing-binding"
    shutil.copytree(interpolation, missing)
    attempt_path = missing / "interpolation-attempt-preregistration.json"
    attempt = json.loads(attempt_path.read_text())
    del attempt["frozen_files"]["configs/parallel_rescue_selection_v1.json"]
    _write_json(attempt_path, attempt)
    provenance_path = missing / "interpolation-provenance-verification.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["attempt_preregistration_sha256"] = selection.sha256_file(attempt_path)
    _write_json(provenance_path, provenance)
    result_path = missing / "interpolation-result.json"
    result = json.loads(result_path.read_text())
    result["provenance"] = provenance
    _write_json(result_path, result)
    _artifact_manifest(missing, "barun-interpolation-artifacts-v1")
    with pytest.raises(selection.ParallelRescueSelectionError, match="attempt.frozen_files lacks"):
        selection.select_parallel_rescue(
            continual_bundle_root=continual,
            interpolation_bundle_root=missing,
            outer_attempt_path=PROTOCOL,
            outer_attempt_sha256=selection.PARALLEL_RESCUE_SELECTION_SHA256,
        )


def test_cli_receipt_is_optional_exclusive_and_outside_bundles(
    bundles: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    continual, interpolation = bundles
    receipt = tmp_path / "selection-receipt.json"
    args = [
        "--continual-bundle",
        str(continual),
        "--interpolation-bundle",
        str(interpolation),
        "--outer-attempt",
        str(PROTOCOL),
        "--outer-attempt-sha256",
        selection.PARALLEL_RESCUE_SELECTION_SHA256,
        "--receipt",
        str(receipt),
    ]

    assert selection_cli.main(args) == 0
    assert json.loads(receipt.read_text())["selection"]["selected_trial_id"] == "I25"
    capsys.readouterr()
    assert selection_cli.main(args) == 1
    assert "refusing to overwrite" in capsys.readouterr().err

    inside = continual / "forbidden-receipt.json"
    with pytest.raises(selection.ParallelRescueSelectionError, match="outside immutable"):
        selection.write_exclusive_receipt(
            inside,
            {"ok": True},
            forbidden_roots=(continual, interpolation),
        )
