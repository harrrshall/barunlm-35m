from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from barunlm.training import mobile_temporal_experiment as backend


def _execution_plan() -> dict[str, Any]:
    fits = []
    for seed in backend.EXPECTED_SEEDS:
        for arm in backend.EXPECTED_ARM_IDS:
            fits.append(
                {
                    "arm_id": arm,
                    "arm_name": {"A": "standard", "B": "repeat", "C": "mbcf"}[arm],
                    "expected_optimizer_steps": 20,
                    "run_id": f"20260804-1200-temporal-{arm.lower()}-s{seed}",
                    "seed": seed,
                    "train_manifest": f"{arm.lower()}-train.jsonl",
                    "train_rows": 1_260,
                    "train_sha256": str(seed).rjust(64, arm.lower()),
                }
            )
    return {
        "fits": fits,
        "conditional_full_refit": {
            "run_id": "20260804-1200-temporal-full-s17",
            "seed": 17,
            "expected_optimizer_steps": 21,
            "train_manifest": "../full/full.jsonl",
            "train_rows": 1_300,
            "train_sha256": "f" * 64,
        },
    }


class FakeOperations:
    def __init__(self, *, selection: bool, confirmation: bool) -> None:
        self.gates = {"selection": selection, "confirmation": confirmation}
        self.trained: list[tuple[str, int]] = []
        self.scored: list[tuple[str, int, str]] = []
        self.gate_calls: list[str] = []
        self.full_calls = 0
        self.terminal_calls = 0
        self.events: list[str] = []

    def train_screening_fit(self, fit):
        identity = (fit["arm_id"], fit["seed"])
        self.trained.append(identity)
        self.events.append(f"train-{identity[0]}-{identity[1]}")
        return {"arm_id": identity[0], "seed": identity[1]}

    def score_screening_fit(self, fit_result, *, population):
        self.scored.append((fit_result["arm_id"], fit_result["seed"], population))
        self.events.append(f"score-{population}-{fit_result['arm_id']}-{fit_result['seed']}")
        return [{"population": population}]

    def evaluate_phase_gate(self, *, phase, evidence):
        self.gate_calls.append(phase)
        self.events.append(f"gate-{phase}")
        assert set(evidence) == {"A", "B", "C"}
        assert all(set(by_seed) == {17, 29, 43} for by_seed in evidence.values())
        return {"passed": self.gates[phase], "phase": phase}

    def train_conditional_full_refit(self, specification):
        self.full_calls += 1
        self.events.append("train-full")
        return {"run_id": specification["run_id"], "development_labels_read": 0}

    def evaluate_terminal_compatibility(self, full_refit):
        self.terminal_calls += 1
        self.events.append("score-terminal")
        assert full_refit["development_labels_read"] == 0
        return {"passed": True, "rows_read": 756, "rows_scored": 756}


def test_selection_failure_stops_before_confirmation_and_full_refit() -> None:
    operations = FakeOperations(selection=False, confirmation=True)
    result = backend.run_gated_schedule(_execution_plan(), operations)

    assert len(operations.trained) == 9
    assert len(operations.scored) == 9
    assert {population for _, _, population in operations.scored} == {"selection"}
    assert operations.gate_calls == ["selection"]
    assert operations.full_calls == 0
    assert operations.terminal_calls == 0
    assert result["status"] == "stopped_after_selection_gate_failure"


def test_confirmation_failure_preserves_all_scores_but_never_runs_full_refit() -> None:
    operations = FakeOperations(selection=True, confirmation=False)
    result = backend.run_gated_schedule(_execution_plan(), operations)

    assert len(operations.trained) == 9
    assert len(operations.scored) == 18
    assert operations.gate_calls == ["selection", "confirmation"]
    assert operations.full_calls == 0
    assert operations.terminal_calls == 0
    assert result["status"] == "stopped_after_confirmation_gate_failure"


def test_full_refit_runs_exactly_once_only_after_both_gates_pass() -> None:
    operations = FakeOperations(selection=True, confirmation=True)
    result = backend.run_gated_schedule(_execution_plan(), operations)

    assert len(operations.trained) == 9
    assert len(operations.scored) == 18
    assert operations.gate_calls == ["selection", "confirmation"]
    assert operations.full_calls == 1
    assert operations.terminal_calls == 1
    assert operations.events[-2:] == ["train-full", "score-terminal"]
    assert result["status"] == "terminal_compatibility_passed"
    assert result["conditional_full_refit"]["development_labels_read"] == 0
    assert result["terminal_compatibility_veto"]["rows_read"] == 756


def test_every_screening_payload_starts_from_the_exact_pinned_base(tmp_path: Path) -> None:
    frozen = {
        "hypothesis": "Frozen hypothesis",
        "decision": "Frozen decision",
        "base_checkpoint": {
            "repo_id": backend.EXPECTED_BASE_REPO,
            "revision": backend.EXPECTED_BASE_REVISION,
            "parameter_count": backend.EXPECTED_PARAMETER_COUNT,
            "file_sha256": dict(backend.EXPECTED_BASE_HASHES),
        },
    }
    payloads = [
        backend.build_screening_training_payload(
            frozen_config=frozen,
            fit=fit,
            train_manifest=tmp_path / fit["train_manifest"],
            selection_manifest=tmp_path / "selection.jsonl",
            selection_sha256="e" * 64,
            output_root=tmp_path / "training",
            jarvis_resource_id="987654",
        )
        for fit in _execution_plan()["fits"]
    ]

    assert len(payloads) == 9
    expected_base = {
        "source": "huggingface",
        "repo_id": backend.EXPECTED_BASE_REPO,
        "revision": backend.EXPECTED_BASE_REVISION,
        "expected_sha256": backend.EXPECTED_BASE_HASHES,
    }
    assert all(payload["base_checkpoint"] == expected_base for payload in payloads)
    assert {payload["optimization"]["seed"] for payload in payloads} == {17, 29, 43}
    assert all(payload["data"]["dev_manifest"].endswith("selection.jsonl") for payload in payloads)
    assert all(payload["optimization"]["max_steps"] is None for payload in payloads)


@pytest.mark.parametrize(
    ("device_count", "device_name", "message"),
    [(2, "NVIDIA H200", "exactly one"), (1, "NVIDIA L4", "requires an H200")],
)
def test_cuda_preflight_rejects_wrong_frozen_hardware(
    monkeypatch: pytest.MonkeyPatch, device_count: int, device_name: str, message: str
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", backend.DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG)
    monkeypatch.setattr(backend, "_force_math_sdpa", lambda: {"math": True})
    monkeypatch.setattr(backend.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(backend.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(backend.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(backend.torch.cuda, "device_count", lambda: device_count)
    monkeypatch.setattr(backend.torch.cuda, "get_device_name", lambda _index: device_name)

    with pytest.raises(backend.TemporalExecutionError, match=message):
        backend._cuda_preflight(configured_before_torch_import=True)


def test_direct_evaluator_thresholds_equal_legacy_mapping_and_keep_two_of_three() -> None:
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads(
        (root / "configs" / "mobile_temporal_counterfactual_v1.json").read_text(encoding="utf-8")
    )

    selection = backend._gate_thresholds(frozen, confirmation=False)
    confirmation = backend._gate_thresholds(frozen, confirmation=True)
    assert selection["minimum_positive_seed_count"] == 2
    assert confirmation["minimum_positive_seed_count"] == 2
    assert confirmation["bootstrap_lower_probability"] == {"numerator": 1, "denominator": 20}

    tampered = copy.deepcopy(frozen)
    tampered["confirmation_gate"]["evaluator_thresholds"]["minimum_positive_seed_count"] = 0
    with pytest.raises(backend.TemporalExecutionError, match="differs from the legacy"):
        backend._gate_thresholds(tampered, confirmation=True)


def test_schedule_has_no_official_or_reused_development_phase() -> None:
    operations = FakeOperations(selection=True, confirmation=True)
    backend.run_gated_schedule(_execution_plan(), operations)

    populations = {population for _, _, population in operations.scored}
    assert populations == {"selection", "confirmation"}
    assert "official_961" not in populations
    assert "reused_756" not in populations

    stopped = backend._data_firewall_receipt(
        terminal=None, reused_rows_read=0, reused_rows_scored=0
    )
    completed = backend._data_firewall_receipt(
        terminal={"status": "pass"}, reused_rows_read=756, reused_rows_scored=756
    )
    assert stopped["terminal_compatibility_veto"] == "not_reached_due_to_shadow_gate"
    assert stopped["reused_756_rows_read"] == 0
    assert completed["terminal_compatibility_veto"] == "pass"
    assert completed["reused_756_rows_scored"] == 756
    assert completed["official_961_rows_read"] == 0


def test_decoding_caps_are_phase_fixed_and_do_not_inspect_gold_lengths(tmp_path: Path) -> None:
    manifest = tmp_path / "shadow.jsonl"
    manifest.write_text('{"target":"x"}\n', encoding="utf-8")
    before = (
        backend._max_new_tokens_for_phase("selection"),
        backend._max_new_tokens_for_phase("confirmation"),
        backend._max_new_tokens_for_phase("terminal"),
    )
    manifest.write_text(json.dumps({"target": "x" * 100_000}) + "\n", encoding="utf-8")
    after = (
        backend._max_new_tokens_for_phase("selection"),
        backend._max_new_tokens_for_phase("confirmation"),
        backend._max_new_tokens_for_phase("terminal"),
    )

    assert before == after == (256, 256, 192)


def test_terminal_veto_enforces_all_frozen_minima_and_failure_maxima() -> None:
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads(
        (root / "configs" / "mobile_temporal_counterfactual_v1.json").read_text(encoding="utf-8")
    )
    thresholds = frozen["terminal_compatibility_veto"]
    samples = [
        {
            "sample_id": f"sample-{index:04d}",
            "prediction_raw": "{}",
            "generation_failure": None,
            "truncated": False,
            "catastrophic_unauthorized_action": False,
        }
        for index in range(756)
    ]
    aggregate = {
        "sample_count": 756,
        "ast_exact_match": {"numerator": 602, "denominator": 756},
        "parse_valid": {"numerator": 756, "denominator": 756},
        "schema_valid": {"numerator": 755, "denominator": 756},
        "missing_prediction": {"numerator": 0, "denominator": 756},
        "truncation": {"numerator": 0, "denominator": 756},
        "catastrophic_unauthorized_actions": 0,
        "per_scenario_success": {
            name: {"numerator": minimum[0], "denominator": minimum[1]}
            for name, minimum in thresholds["minimum_scenario_ast_exact"].items()
        },
    }
    temporal = {
        "subsets": {
            backend.CALENDAR_CROSS_MONTH_SUBSET: {
                backend.CALENDAR_DATETIME_EXACT_METRIC: {
                    "numerator": 25,
                    "denominator": 59,
                }
            }
        }
    }

    passed = backend.evaluate_terminal_veto(
        frozen_config=frozen,
        aggregate=aggregate,
        temporal_subsets=temporal,
        sample_scores=samples,
    )
    assert passed["passed"] is True

    rejected_aggregate = copy.deepcopy(aggregate)
    rejected_aggregate["per_scenario_success"]["show_map"]["numerator"] = 56
    rejected = backend.evaluate_terminal_veto(
        frozen_config=frozen,
        aggregate=rejected_aggregate,
        temporal_subsets=temporal,
        sample_scores=samples,
    )
    assert rejected["passed"] is False
    assert rejected["checks"]["scenario_show_map_at_least_minimum"] is False


def test_essential_bundle_excludes_screening_weights_and_all_optimizers(tmp_path: Path) -> None:
    controller = tmp_path / "controller"
    execution = controller / "execution"
    execution.mkdir(parents=True)
    (controller / "preflight-receipt.json").write_text("{}\n", encoding="utf-8")
    (controller / "attempt-preregistration.json").write_text("{}\n", encoding="utf-8")
    (controller / "source-snapshot.json").write_text("{}\n", encoding="utf-8")
    (controller / "pre-cuda-validation.log").write_text("passed\n", encoding="utf-8")
    (controller / "pre-cuda-validation.json").write_text("{}\n", encoding="utf-8")
    attempt_sha256 = hashlib.sha256(
        (controller / "attempt-preregistration.json").read_bytes()
    ).hexdigest()
    snapshot_sha256 = hashlib.sha256((controller / "source-snapshot.json").read_bytes()).hexdigest()
    validation_log_sha256 = hashlib.sha256(
        (controller / "pre-cuda-validation.log").read_bytes()
    ).hexdigest()
    (controller / "execution-plan.json").write_text(
        json.dumps(
            {
                "launch_provenance": {
                    "attempt_preregistration": {"sha256": attempt_sha256},
                    "source_snapshot": {"sha256": snapshot_sha256},
                },
                "pre_cuda_validation": {
                    "status": "passed",
                    "log_sha256": validation_log_sha256,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (execution / "environment-receipt.json").write_text("{}\n", encoding="utf-8")
    (execution / "result.json").write_text("{}\n", encoding="utf-8")
    (execution / "failure.json").write_text('{"artifacts_preserved":true}\n', encoding="utf-8")
    partial = execution / "essential"
    partial.mkdir()
    (partial / "partial-marker.json").write_text("{}\n", encoding="utf-8")
    fit = execution / "fits" / "a-s17"
    fit.mkdir(parents=True)
    (fit / "training-receipt.json").write_text("{}\n", encoding="utf-8")
    training = execution / "training" / "screening-run"
    checkpoint = training / "checkpoints" / "step-00000020"
    checkpoint.mkdir(parents=True)
    (training / "summary.json").write_text("{}\n", encoding="utf-8")
    (training / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"screening-weight")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    promoted = execution / "conditional-full-refit" / "inference-checkpoint"
    promoted.mkdir(parents=True)
    (promoted / "model.safetensors").write_bytes(b"promoted-weight")
    (promoted / "barun_config.json").write_text("{}\n", encoding="utf-8")
    (promoted / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    receipt = backend._build_essential_bundle(execution_dir=execution, controller_dir=controller)
    essential = Path(receipt["path"])
    files = {
        path.relative_to(essential).as_posix() for path in essential.rglob("*") if path.is_file()
    }
    assert "conditional-full-refit/inference-checkpoint/model.safetensors" in files
    assert "controller/attempt-preregistration.json" in files
    assert "controller/source-snapshot.json" in files
    assert "failure.json" in files
    assert not any(path.endswith("optimizer.pt") for path in files)
    assert not any("checkpoints/step-" in path for path in files)
    manifest = json.loads((essential / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest["optimizer_files"] == 0
    assert manifest["screening_model_weight_files"] == 0
    assert manifest["promoted_model_weight_files"] == 1
    assert (execution / "essential-prior-incomplete" / "partial-marker.json").is_file()


def test_retry_detector_fails_closed_on_dev_loss_and_malformed_metrics(tmp_path: Path) -> None:
    valid = tmp_path / "training" / "valid" / "metrics.jsonl"
    valid.parent.mkdir(parents=True)
    valid.write_text('{"event":"dev","loss":1.25}\n', encoding="utf-8")
    malformed = tmp_path / "training" / "malformed" / "metrics.jsonl"
    malformed.parent.mkdir(parents=True)
    malformed.write_text('{"event":"dev","loss":', encoding="utf-8")
    odd_dev = tmp_path / "training" / "odd-dev" / "metrics.jsonl"
    odd_dev.parent.mkdir(parents=True)
    odd_dev.write_text('{"event":"dev","loss":"torn"}\n', encoding="utf-8")
    marker = tmp_path / "training" / "marked" / "heldout_access_started.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n", encoding="utf-8")
    train_only = tmp_path / "training" / "train-only" / "metrics.jsonl"
    train_only.parent.mkdir(parents=True)
    train_only.write_text('{"event":"train","loss":2.0}\n', encoding="utf-8")

    detected = backend._usable_held_out_artifacts(tmp_path)

    assert "training/valid/metrics.jsonl" in detected
    assert "training/malformed/metrics.jsonl" in detected
    assert "training/odd-dev/metrics.jsonl" in detected
    assert "training/marked/heldout_access_started.json" in detected
    assert "training/train-only/metrics.jsonl" not in detected


def test_screening_access_marker_precedes_trainer_dev_load_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operations = object.__new__(backend._FixedOperations)
    operations.fit_root = tmp_path / "fits"
    operations.fit_root.mkdir()
    operations.config_root = tmp_path / "training-configs"
    operations.config_root.mkdir()
    operations.training_root = tmp_path / "training"
    operations.screening_dir = tmp_path / "screening"
    operations.screening_dir.mkdir()
    operations.frozen_config = {}
    operations.jarvis_resource_id = "987654"
    operations._manifest_paths = {"selection": tmp_path / "selection.jsonl"}
    operations._manifest_hashes = {"selection": "a" * 64}
    monkeypatch.setattr(backend, "build_screening_training_payload", lambda **_kwargs: {})
    monkeypatch.setattr(
        backend.TrainingRunConfig,
        "from_json",
        classmethod(lambda _cls, _path: object()),
    )

    def fail_dev_load(*_args, **_kwargs):
        raise RuntimeError("development load failed")

    monkeypatch.setattr(backend, "train_sft", fail_dev_load)
    fit = {
        "arm_id": "A",
        "seed": 17,
        "run_id": "20260804-0100-marker-s17",
        "train_manifest": "construction-train.jsonl",
    }

    with pytest.raises(RuntimeError, match="development load failed"):
        operations.train_screening_fit(fit)

    marker = tmp_path / "fits" / "a-s17" / "selection-heldout-access-started.json"
    assert marker.is_file()
    assert "fits/a-s17/selection-heldout-access-started.json" in backend._usable_held_out_artifacts(
        tmp_path
    )
