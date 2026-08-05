from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_model, save_file, save_model

from barunlm.config import BarunConfig
from barunlm.evaluation.interpolation_rescue import (
    INTERPOLATION_PROTOCOL_PATH,
    INTERPOLATION_PROTOCOL_SHA256,
    InterpolationRescueError,
    evaluate_joint_gate,
    evaluate_presto_gate,
    interpolate_safetensors,
    interpolation_arms,
    load_interpolation_protocol,
    protocol_sha256,
    select_interpolation_arm,
)
from barunlm.evaluation.presto import PRESTO_SCORER_VERSION
from barunlm.model import BarunLM
from barunlm.training.data import sha256_file

ROOT = Path(__file__).resolve().parents[1]
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "barun_interpolation_rescue_runner",
    ROOT / "scripts" / "run_interpolation_rescue.py",
)
if _RUNNER_SPEC is None or _RUNNER_SPEC.loader is None:  # pragma: no cover - import invariant
    raise RuntimeError("cannot load interpolation rescue runner for tests")
interpolation_runner = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(interpolation_runner)


def _tiny_config() -> BarunConfig:
    return BarunConfig(
        vocab_size=16,
        dim=8,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim=12,
        max_seq_len=32,
        rope_fraction=0.5,
        local_window=8,
        full_attention_every=1,
        residual_select_every=1,
        tie_embeddings=True,
    )


def _fill_model(model: BarunLM, value: float) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)


def _tensor_contract(path: Path) -> tuple[int, int, dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        parameters = sum(handle.get_tensor(key).numel() for key in keys)
        return len(keys), parameters, dict(handle.metadata() or {})


def _presto_aggregate(
    *,
    exact_numerator: int = 10_500,
    revision_value: float = 0.71,
    disfluency_value: float = 0.74,
) -> dict[str, object]:
    raw_counts = {
        "cancel-action": 454,
        "correct-action": 230,
        "correct-argument": 389,
        "within-turn-correction": 3067,
    }
    return {
        "schema_version": PRESTO_SCORER_VERSION,
        "metric_scope": "derived_action_ir_not_native_presto_semantic_parse",
        "sample_count": 14_288,
        "ast_exact_match": {
            "numerator": exact_numerator,
            "denominator": 14_288,
            "value": exact_numerator / 14_288,
        },
        "schema_valid": {"numerator": 14_288, "denominator": 14_288, "value": 1.0},
        "abstention": {"f1": 0.9},
        "false_call_on_gate": {"numerator": 0, "denominator": 10_407, "value": 0.0},
        "headline_phenomenon_buckets": {
            "no_phenomenon": {
                "count": 6648,
                "ast_exact_match": {"value": 0.80},
            },
            "revision": {
                "count": 4140,
                "ast_exact_match": {"value": revision_value},
            },
            "disfluency": {
                "count": 2624,
                "ast_exact_match": {"value": disfluency_value},
            },
        },
        "per_phenomenon": {
            name: {"count": count, "ast_exact_match": {"value": 0.75}}
            for name, count in raw_counts.items()
        },
    }


def _mobile_result(numerator: int, *, passed: bool | None = None) -> dict[str, object]:
    observed_pass = numerator >= 587 if passed is None else passed
    return {
        "gate": {
            "passed": observed_pass,
            "candidate": {
                "numerator": numerator,
                "denominator": 756,
                "value": numerator / 756,
            },
        }
    }


def _arm_result(
    arm_id: str,
    *,
    presto_numerator: int,
    mobile_numerator: int,
    passed: bool = True,
) -> dict[str, object]:
    alpha_by_arm = {
        "alpha-025": {"numerator": 1, "denominator": 4, "value": 0.25},
        "alpha-050": {"numerator": 1, "denominator": 2, "value": 0.5},
        "alpha-075": {"numerator": 3, "denominator": 4, "value": 0.75},
    }
    presto = _presto_aggregate(exact_numerator=presto_numerator)
    mobile = _mobile_result(mobile_numerator)
    return {
        "arm_id": arm_id,
        "alpha": alpha_by_arm[arm_id],
        "joint_gate": {"passed": passed},
        "presto_aggregate": presto,
        "mobile_result": mobile,
    }


def _provenance_stage(
    tmp_path: Path,
    *,
    run_id: str = "20260803-2100-interpolation-rescue-s17",
    machine_id: int = 999_001,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "interpolation-stage"
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "src").mkdir()
    shutil.copy2(
        ROOT / "scripts" / "run_interpolation_rescue.py",
        root / "scripts" / "run_interpolation_rescue.py",
    )
    shutil.copy2(
        INTERPOLATION_PROTOCOL_PATH,
        root / "configs" / "interpolation_rescue_v1.json",
    )
    shutil.copy2(
        ROOT / "configs" / "parallel_rescue_selection_v1.json",
        root / "configs" / "parallel_rescue_selection_v1.json",
    )
    (root / "src" / "safe.txt").write_text("source evidence\n", encoding="utf-8")
    runner_sha256 = sha256_file(root / "scripts" / "run_interpolation_rescue.py")
    frozen_files = {
        "configs/interpolation_rescue_v1.json": INTERPOLATION_PROTOCOL_SHA256,
        "configs/parallel_rescue_selection_v1.json": (
            "68bb6132a486dfbd6c9ad478db9c87ae9f2350bfc1d302193169fd7f5c290130"
        ),
        "scripts/run_interpolation_rescue.py": runner_sha256,
    }
    tree = interpolation_runner.compute_interpolation_content_tree(root)
    snapshot_path = root / "interpolation-source-snapshot.json"
    snapshot = {
        "schema_version": "barun-interpolation-source-snapshot-v1",
        "run_id": run_id,
        "created_at": "2026-08-03T21:00:00+05:30",
        "created_before_remote_launch_or_model_scoring": True,
        "git_commit": "0" * 40,
        "git_dirty": True,
        "content_tree": tree,
        "frozen_files": frozen_files,
        "staging_policy": {
            "credentials_included": False,
            "caches_included": False,
            "official_test_artifacts_included": False,
            "optimizer_state_included": False,
        },
    }
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attempt_path = root / "interpolation-attempt-preregistration.json"
    attempt = {
        "schema_version": "barun-interpolation-attempt-preregistration-v1",
        "run_id": run_id,
        "status": "frozen_before_remote_model_or_data_scoring",
        "created_at": "2026-08-03T21:01:00+05:30",
        "created_before_remote_model_or_data_scoring": True,
        "protocol": {
            "path": "configs/interpolation_rescue_v1.json",
            "sha256": INTERPOLATION_PROTOCOL_SHA256,
        },
        "source_snapshot": {
            "path": "interpolation-source-snapshot.json",
            "sha256": sha256_file(snapshot_path),
        },
        "frozen_files": frozen_files,
        "prelaunch_inventory": {
            "captured_before_project_instance_creation": True,
            "protected_machine_ids": [463058, 700_001],
            "project_machine_id": machine_id,
            "fresh_project_instance": True,
        },
    }
    attempt_path.write_text(
        json.dumps(attempt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root, attempt_path, snapshot_path


def test_interpolation_protocol_is_hash_pinned_and_nonadaptive() -> None:
    protocol = load_interpolation_protocol()

    assert protocol_sha256() == INTERPOLATION_PROTOCOL_SHA256
    assert INTERPOLATION_PROTOCOL_PATH.name == "interpolation_rescue_v1.json"
    assert [(arm["numerator"], arm["denominator"]) for arm in interpolation_arms(protocol)] == [
        (1, 4),
        (1, 2),
        (3, 4),
    ]
    budget = protocol["interpolation"]["trial_budget"]
    assert budget["checkpoint_arms"] == 3
    assert budget["adaptive_followup_alphas"] == 0
    assert budget["early_stopping_before_both_evaluations"] is False
    assert protocol["official_test_firewall"]["accepted_official_test_input_paths"] == []
    assert protocol["remote_execution"]["known_protected_machine_ids"] == [463058]


def test_interpolation_protocol_mutation_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "interpolation-changed.json"
    changed.write_bytes(INTERPOLATION_PROTOCOL_PATH.read_bytes() + b"\n")

    with pytest.raises(InterpolationRescueError, match="SHA-256 mismatch"):
        load_interpolation_protocol(changed)


def test_float64_interpolation_is_exactly_rounded_once_and_deterministic(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.safetensors"
    presto = tmp_path / "presto.safetensors"
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    metadata = {"lm_head.weight": "embedding.weight"}
    save_file(
        {"embedding.weight": torch.tensor([0.0, 4.0], dtype=torch.float32)},
        candidate,
        metadata=metadata,
    )
    save_file(
        {"embedding.weight": torch.tensor([4.0, 0.0], dtype=torch.float32)},
        presto,
        metadata=metadata,
    )

    evidence = interpolate_safetensors(
        candidate,
        presto,
        first,
        numerator=1,
        denominator=4,
        expected_metadata=metadata,
        expected_tensor_count=1,
        expected_parameter_count=2,
    )
    interpolate_safetensors(
        candidate,
        presto,
        second,
        numerator=1,
        denominator=4,
        expected_metadata=metadata,
        expected_tensor_count=1,
        expected_parameter_count=2,
    )

    with safe_open(first, framework="pt", device="cpu") as handle:
        assert handle.get_tensor("embedding.weight").tolist() == [1.0, 3.0]
        assert handle.metadata() == metadata
    assert first.read_bytes() == second.read_bytes()
    assert evidence["accumulator_dtype"] == "torch.float64"
    assert evidence["output_dtype"] == "torch.float32"


def test_interpolation_preserves_real_tied_model_alias(tmp_path: Path) -> None:
    config = _tiny_config()
    candidate_model = BarunLM(config)
    presto_model = BarunLM(config)
    _fill_model(candidate_model, 0.0)
    _fill_model(presto_model, 4.0)
    candidate = tmp_path / "candidate-model.safetensors"
    presto = tmp_path / "presto-model.safetensors"
    output = tmp_path / "interpolated-model.safetensors"
    save_model(candidate_model, candidate)
    save_model(presto_model, presto)
    tensor_count, parameter_count, metadata = _tensor_contract(candidate)

    interpolate_safetensors(
        candidate,
        presto,
        output,
        numerator=1,
        denominator=2,
        expected_metadata=metadata,
        expected_tensor_count=tensor_count,
        expected_parameter_count=parameter_count,
    )
    loaded = BarunLM(config)
    missing, unexpected = load_model(loaded, output, strict=False)

    assert missing == set()
    assert unexpected == []
    assert loaded.lm_head.weight.data_ptr() == loaded.embedding.weight.data_ptr()
    assert torch.all(loaded.embedding.weight == 2.0)


def test_interpolation_rejects_metadata_and_nonfloating_mismatches(tmp_path: Path) -> None:
    metadata = {"lm_head.weight": "embedding.weight"}
    left = tmp_path / "left.safetensors"
    right = tmp_path / "right.safetensors"
    save_file({"counter": torch.tensor([1], dtype=torch.int64)}, left, metadata=metadata)
    save_file({"counter": torch.tensor([2], dtype=torch.int64)}, right, metadata=metadata)

    with pytest.raises(InterpolationRescueError, match="non-floating endpoint tensor differs"):
        interpolate_safetensors(
            left,
            right,
            tmp_path / "output.safetensors",
            numerator=1,
            denominator=2,
            expected_metadata=metadata,
            expected_tensor_count=1,
            expected_parameter_count=1,
        )

    wrong_metadata = tmp_path / "wrong-metadata.safetensors"
    save_file({"counter": torch.tensor([1], dtype=torch.int64)}, wrong_metadata)
    with pytest.raises(InterpolationRescueError, match="tied-weight metadata"):
        interpolate_safetensors(
            left,
            wrong_metadata,
            tmp_path / "second-output.safetensors",
            numerator=1,
            denominator=2,
            expected_metadata=metadata,
            expected_tensor_count=1,
            expected_parameter_count=1,
        )


def test_corrected_full_family_presto_gate_passes_and_fails_closed() -> None:
    protocol = load_interpolation_protocol()
    passing = evaluate_presto_gate(_presto_aggregate(), protocol)

    assert passing["passed"] is True
    assert passing["checks"]["corrected_revision_family_complete"] is True
    assert passing["checks"]["revision_gap_at_most_threshold"] is True

    hard_gap = evaluate_presto_gate(_presto_aggregate(revision_value=0.69), protocol)
    assert hard_gap["passed"] is False
    assert hard_gap["checks"]["revision_gap_at_most_threshold"] is False

    incomplete = _presto_aggregate()
    incomplete["per_phenomenon"]["correct-action"]["count"] = 229
    failed_family = evaluate_presto_gate(incomplete, protocol)
    assert failed_family["passed"] is False
    assert failed_family["checks"]["corrected_revision_family_complete"] is False


def test_joint_gate_uses_mobile_integer_boundary_and_rejects_disagreement() -> None:
    protocol = load_interpolation_protocol()
    passing = evaluate_joint_gate(_mobile_result(587), _presto_aggregate(), protocol)
    below = evaluate_joint_gate(_mobile_result(586), _presto_aggregate(), protocol)

    assert passing["passed"] is True
    assert below["passed"] is False
    with pytest.raises(InterpolationRescueError, match="nested gate disagrees"):
        evaluate_joint_gate(
            _mobile_result(586, passed=True),
            _presto_aggregate(),
            protocol,
        )


def test_selection_is_fixed_presto_then_mobile_then_lower_alpha() -> None:
    protocol = load_interpolation_protocol()
    results = [
        _arm_result("alpha-025", presto_numerator=10_500, mobile_numerator=602),
        _arm_result("alpha-050", presto_numerator=10_600, mobile_numerator=587),
        _arm_result("alpha-075", presto_numerator=10_550, mobile_numerator=600),
    ]
    selection = select_interpolation_arm(results, protocol)
    assert selection["selected_arm_id"] == "alpha-050"

    tied = [
        _arm_result("alpha-025", presto_numerator=10_600, mobile_numerator=590),
        _arm_result("alpha-050", presto_numerator=10_600, mobile_numerator=600),
        _arm_result("alpha-075", presto_numerator=10_600, mobile_numerator=600),
    ]
    tie_selection = select_interpolation_arm(tied, protocol)
    assert tie_selection["selected_arm_id"] == "alpha-050"


def test_selection_rejects_without_joint_passer_and_never_adds_alpha() -> None:
    protocol = load_interpolation_protocol()
    results = [
        _arm_result(
            arm_id,
            presto_numerator=10_600,
            mobile_numerator=586,
            passed=False,
        )
        for arm_id in ("alpha-025", "alpha-050", "alpha-075")
    ]
    selection = select_interpolation_arm(results, protocol)

    assert selection == {
        "status": "reject_interpolation_no_joint_passer",
        "selected_arm_id": None,
        "ordered_joint_passers": [],
        "fallback": "candidate-v2",
        "additional_alpha_authorized": False,
    }


def test_protocol_json_contains_no_unfrozen_alpha_escape_hatch() -> None:
    raw = json.loads(INTERPOLATION_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert raw["selection"]["no_joint_passer"].endswith("run_no_additional_alpha")
    assert raw["interpolation"]["trial_budget"]["adaptive_followup_alphas"] == 0


def test_launch_provenance_binds_attempt_snapshot_runner_protocol_and_tree(
    tmp_path: Path,
) -> None:
    run_id = "20260803-2100-interpolation-rescue-s17"
    machine_id = 999_001
    root, attempt, snapshot = _provenance_stage(
        tmp_path,
        run_id=run_id,
        machine_id=machine_id,
    )

    receipt = interpolation_runner.verify_interpolation_launch_provenance(
        repository_root=root,
        attempt_preregistration_path=attempt,
        source_snapshot_path=snapshot,
        run_id=run_id,
        jarvis_machine_id=machine_id,
    )

    assert receipt["verified_before_model_or_data_scoring"] is True
    assert receipt["source_snapshot_sha256"] == sha256_file(snapshot)
    assert receipt["runner_sha256"] == sha256_file(ROOT / "scripts" / "run_interpolation_rescue.py")
    assert receipt["protocol_sha256"] == INTERPOLATION_PROTOCOL_SHA256
    assert receipt["shared_selector_sha256"] == (
        "68bb6132a486dfbd6c9ad478db9c87ae9f2350bfc1d302193169fd7f5c290130"
    )
    assert receipt["content_tree"] == json.loads(snapshot.read_text())["content_tree"]
    assert receipt["caches_included_in_tree_accounting"] is False


def test_launch_provenance_rejects_attempt_to_snapshot_mismatch(tmp_path: Path) -> None:
    root, attempt_path, snapshot = _provenance_stage(tmp_path)
    attempt = json.loads(attempt_path.read_text())
    attempt["source_snapshot"]["sha256"] = "f" * 64
    attempt_path.write_text(json.dumps(attempt), encoding="utf-8")

    with pytest.raises(InterpolationRescueError, match="attempt-to-snapshot"):
        interpolation_runner.verify_interpolation_launch_provenance(
            repository_root=root,
            attempt_preregistration_path=attempt_path,
            source_snapshot_path=snapshot,
            run_id="20260803-2100-interpolation-rescue-s17",
            jarvis_machine_id=999_001,
        )


def test_launch_provenance_requires_frozen_shared_selector_binding(tmp_path: Path) -> None:
    first_root, first_attempt, first_snapshot = _provenance_stage(tmp_path / "attempt")
    attempt = json.loads(first_attempt.read_text())
    attempt["frozen_files"]["configs/parallel_rescue_selection_v1.json"] = "f" * 64
    first_attempt.write_text(json.dumps(attempt), encoding="utf-8")
    with pytest.raises(InterpolationRescueError, match="shared-selector hashes"):
        interpolation_runner.verify_interpolation_launch_provenance(
            repository_root=first_root,
            attempt_preregistration_path=first_attempt,
            source_snapshot_path=first_snapshot,
            run_id="20260803-2100-interpolation-rescue-s17",
            jarvis_machine_id=999_001,
        )

    second_root, second_attempt, second_snapshot = _provenance_stage(tmp_path / "staged")
    (second_root / "configs" / "parallel_rescue_selection_v1.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(InterpolationRescueError, match="shared rescue selector"):
        interpolation_runner.verify_interpolation_launch_provenance(
            repository_root=second_root,
            attempt_preregistration_path=second_attempt,
            source_snapshot_path=second_snapshot,
            run_id="20260803-2100-interpolation-rescue-s17",
            jarvis_machine_id=999_001,
        )


def test_launch_provenance_rejects_tree_mutation_and_credential_markers(
    tmp_path: Path,
) -> None:
    first_root, first_attempt, first_snapshot = _provenance_stage(tmp_path / "tree")
    (first_root / "src" / "safe.txt").write_text("changed after snapshot\n", encoding="utf-8")
    with pytest.raises(InterpolationRescueError, match="staged content tree differs"):
        interpolation_runner.verify_interpolation_launch_provenance(
            repository_root=first_root,
            attempt_preregistration_path=first_attempt,
            source_snapshot_path=first_snapshot,
            run_id="20260803-2100-interpolation-rescue-s17",
            jarvis_machine_id=999_001,
        )

    second_root, second_attempt, second_snapshot = _provenance_stage(tmp_path / "credential")
    (second_root / "src" / "leak.txt").write_text(
        "WANDB_" + "API_KEY=secret-value\n",
        encoding="utf-8",
    )
    with pytest.raises(InterpolationRescueError, match="credential marker"):
        interpolation_runner.verify_interpolation_launch_provenance(
            repository_root=second_root,
            attempt_preregistration_path=second_attempt,
            source_snapshot_path=second_snapshot,
            run_id="20260803-2100-interpolation-rescue-s17",
            jarvis_machine_id=999_001,
        )


def test_launch_provenance_excludes_runtime_caches_but_not_arbitrary_files(
    tmp_path: Path,
) -> None:
    root, attempt, snapshot = _provenance_stage(tmp_path)
    cache = root / ".pytest_cache"
    cache.mkdir()
    (cache / "runtime.txt").write_text("generated after staging\n", encoding="utf-8")

    receipt = interpolation_runner.verify_interpolation_launch_provenance(
        repository_root=root,
        attempt_preregistration_path=attempt,
        source_snapshot_path=snapshot,
        run_id="20260803-2100-interpolation-rescue-s17",
        jarvis_machine_id=999_001,
    )
    assert receipt["content_tree"]["file_count"] == 4

    (root / "unexpected.txt").write_text("not excluded\n", encoding="utf-8")
    with pytest.raises(InterpolationRescueError, match="staged content tree differs"):
        interpolation_runner.verify_interpolation_launch_provenance(
            repository_root=root,
            attempt_preregistration_path=attempt,
            source_snapshot_path=snapshot,
            run_id="20260803-2100-interpolation-rescue-s17",
            jarvis_machine_id=999_001,
        )


def test_launch_provenance_rejects_schema_extras_and_preexisting_project_id(
    tmp_path: Path,
) -> None:
    first_root, first_attempt, first_snapshot = _provenance_stage(tmp_path / "extra")
    attempt_payload = json.loads(first_attempt.read_text())
    attempt_payload["unregistered"] = True
    first_attempt.write_text(json.dumps(attempt_payload), encoding="utf-8")
    with pytest.raises(InterpolationRescueError, match="fields differ"):
        interpolation_runner.verify_interpolation_launch_provenance(
            repository_root=first_root,
            attempt_preregistration_path=first_attempt,
            source_snapshot_path=first_snapshot,
            run_id="20260803-2100-interpolation-rescue-s17",
            jarvis_machine_id=999_001,
        )

    second_root, second_attempt, second_snapshot = _provenance_stage(tmp_path / "protected")
    second_payload = json.loads(second_attempt.read_text())
    second_payload["prelaunch_inventory"]["protected_machine_ids"].append(999_001)
    second_attempt.write_text(json.dumps(second_payload), encoding="utf-8")
    with pytest.raises(InterpolationRescueError, match="pre-existing and is protected"):
        interpolation_runner.verify_interpolation_launch_provenance(
            repository_root=second_root,
            attempt_preregistration_path=second_attempt,
            source_snapshot_path=second_snapshot,
            run_id="20260803-2100-interpolation-rescue-s17",
            jarvis_machine_id=999_001,
        )


def test_run_checks_provenance_before_environment_or_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "interpolation-stage"
    repository.mkdir()
    environment_called = False

    def forbidden_environment() -> dict[str, object]:
        nonlocal environment_called
        environment_called = True
        raise AssertionError("environment/scoring must follow provenance")

    monkeypatch.setattr(interpolation_runner, "_runtime_environment", forbidden_environment)
    args = SimpleNamespace(
        repository_root=repository,
        artifact_root=tmp_path / "artifacts",
        run_id="20260803-2100-interpolation-rescue-s17",
        jarvis_machine_id=999_001,
        attempt_preregistration=repository / "interpolation-attempt-preregistration.json",
        source_snapshot=repository / "interpolation-source-snapshot.json",
    )

    with pytest.raises(InterpolationRescueError, match="attempt preregistration"):
        interpolation_runner.run(args)
    assert environment_called is False
