from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from barunlm.evaluation import mobile_regression
from barunlm.evaluation.evaluator import EVALUATOR_VERSION
from barunlm.evaluation.mobile_actions import MOBILE_ACTIONS_SCORER_VERSION

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "configs" / "mobile_regression_v1.json"
DOWNLOADED_REGRESSION_EVIDENCE = (
    ROOT / "experiments/runs/20260803-1810-mobile-blind-s17/remote/data/dev.jsonl",
    ROOT / "experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/"
    "final-eval/scores/aggregate.json",
    ROOT / "experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/"
    "final-eval/scores/sample_scores.jsonl",
)


def _preregistration() -> dict[str, object]:
    return mobile_regression.load_preregistration(PREREGISTRATION)


def _sample(index: int, *, exact: bool) -> dict[str, object]:
    return {
        "sample_id": f"sample-{index:04d}",
        "scenario": "show_map",
        "gold": {
            "calls": [{"args": {"query": str(index)}, "tool": "show_map"}],
            "decision": "CALL",
            "mode": "SINGLE",
        },
        "ast_exact": exact,
    }


def _aggregate(numerator: int) -> dict[str, object]:
    return {
        "schema_version": MOBILE_ACTIONS_SCORER_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "ast_exact_match": {
            "numerator": numerator,
            "denominator": 756,
            "value": numerator / 756,
        },
    }


def test_frozen_preregistration_pins_reference_population_and_integer_gate() -> None:
    payload = _preregistration()
    dataset = payload["dataset"]
    reference = payload["reference"]
    gate = payload["regression_gate"]

    assert isinstance(dataset, dict)
    assert dataset["manifest"] == {
        "relative_path": ("experiments/runs/20260803-1810-mobile-blind-s17/remote/data/dev.jsonl"),
        "sha256": "988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55",
        "rows": 756,
        "required_source_split": "train",
        "required_derived_split": "dev",
    }
    assert isinstance(reference, dict)
    assert reference["candidate_id"] == "candidate-v2"
    assert reference["aggregate"]["ast_exact_match"] == {
        "numerator": 602,
        "denominator": 756,
    }
    assert gate == {
        "metric": "ast_exact_match",
        "direction": "higher_is_better",
        "maximum_absolute_drop": {"numerator": 2, "denominator": 100},
        "minimum_candidate_numerator": 587,
        "required_denominator": 756,
        "maximum_net_correct_rows_lost": 15,
        "gate_only": True,
        "does_not_consume_an_additional_mobile_development_selection_trial": True,
    }
    assert dataset["official_evaluation_firewall"]["accepted_input_paths"] == []


def test_preregistration_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_bytes(PREREGISTRATION.read_bytes() + b"\n")

    with pytest.raises(mobile_regression.MobileRegressionError, match="SHA-256 mismatch"):
        mobile_regression.load_preregistration(changed)


def test_checkpoint_hash_input_must_name_exact_inference_files(tmp_path: Path) -> None:
    hashes = {
        "barun_config.json": "a" * 64,
        "model.safetensors": "b" * 64,
        "tokenizer.json": "c" * 64,
    }
    manifest = tmp_path / "checkpoint_manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": "barun-release-checkpoint-v1", "file_sha256": hashes}),
        encoding="utf-8",
    )
    assert mobile_regression.load_checkpoint_hashes(manifest) == hashes

    manifest.write_text(
        json.dumps({"file_sha256": {**hashes, "optimizer.pt": "d" * 64}}),
        encoding="utf-8",
    )
    with pytest.raises(mobile_regression.MobileRegressionError, match="exactly"):
        mobile_regression.load_checkpoint_hashes(manifest)


def test_paired_gate_passes_at_587_and_fails_at_586() -> None:
    preregistration = _preregistration()
    reference = [_sample(index, exact=index < 602) for index in range(756)]
    boundary = [_sample(index, exact=index < 587) for index in range(756)]

    gate, rows = mobile_regression.compare_paired_samples(
        candidate_aggregate=_aggregate(587),
        candidate_samples=boundary,
        reference_samples=reference,
        preregistration=preregistration,
    )

    assert gate["passed"] is True
    assert gate["delta"]["correct_rows"] == -15
    assert gate["paired_transition_counts"] == {
        "retained_correct": 587,
        "fixed": 0,
        "regressed": 15,
        "retained_incorrect": 154,
    }
    assert len(rows) == 756
    assert rows[601]["transition"] == "regressed"

    below = [_sample(index, exact=index < 586) for index in range(756)]
    failed, _ = mobile_regression.compare_paired_samples(
        candidate_aggregate=_aggregate(586),
        candidate_samples=below,
        reference_samples=reference,
        preregistration=preregistration,
    )
    assert failed["passed"] is False
    assert failed["decision"] == "mobile_regression_gate_failed_reject_broader_checkpoint"


def test_paired_gate_requires_same_gold_and_scenario() -> None:
    preregistration = _preregistration()
    reference = [_sample(index, exact=index < 602) for index in range(756)]
    candidate = [_sample(index, exact=index < 602) for index in range(756)]
    candidate[0]["gold"] = {"decision": "ABSTAIN"}

    with pytest.raises(mobile_regression.MobileRegressionError, match="gold AST mismatch"):
        mobile_regression.compare_paired_samples(
            candidate_aggregate=_aggregate(602),
            candidate_samples=candidate,
            reference_samples=reference,
            preregistration=preregistration,
        )


def test_manifest_hash_is_checked_before_checkpoint_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden_verify(*args: object, **kwargs: object) -> dict[str, str]:
        nonlocal called
        called = True
        raise AssertionError("checkpoint verification must follow frozen-data verification")

    monkeypatch.setattr(mobile_regression, "verify_checkpoint", forbidden_verify)
    wrong_manifest = tmp_path / "dev.jsonl"
    wrong_manifest.write_text("{}\n", encoding="utf-8")
    runtime = _preregistration()["reference"]["environment"]

    with pytest.raises(mobile_regression.MobileRegressionError, match="manifest SHA-256 mismatch"):
        mobile_regression.run_mobile_regression(
            checkpoint_dir=tmp_path / "checkpoint",
            checkpoint_hashes_path=tmp_path / "missing-hashes.json",
            output_dir=tmp_path / "output",
            repository_root=ROOT,
            preregistration_path=PREREGISTRATION,
            runtime_environment=runtime,
            manifest_path=wrong_manifest,
        )
    assert called is False
    assert not (tmp_path / "output").exists()


@pytest.mark.skipif(
    not all(path.is_file() for path in DOWNLOADED_REGRESSION_EVIDENCE),
    reason="requires downloaded W&B Mobile development and candidate-reference evidence",
)
def test_offline_regression_run_emits_paired_and_hashed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preregistration = _preregistration()
    reference = preregistration["reference"]
    hashes = reference["checkpoint_file_sha256"]
    hash_path = tmp_path / "hashes.json"
    hash_path.write_text(json.dumps(hashes), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    def fake_verify(directory: Path, *, expected_sha256: dict[str, str]) -> dict[str, str]:
        assert Path(directory) == checkpoint
        assert expected_sha256 == hashes
        return dict(expected_sha256)

    def fake_generate(**kwargs: object) -> SimpleNamespace:
        assert kwargs["manifest_sha256"] == (
            "988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55"
        )
        assert kwargs["device_name"] == "cuda"
        assert kwargs["batch_size"] == 128
        assert kwargs["max_new_tokens"] == 192
        predictions = Path(kwargs["predictions_path"])
        predictions.write_text('{"id":"evidence-written-by-generator"}\n', encoding="utf-8")
        predictions.with_suffix(".jsonl.manifest.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(to_dict=lambda: {"schema_version": "barun-greedy-generation-v1"})

    reference_aggregate = ROOT / reference["aggregate"]["relative_path"]
    reference_samples = ROOT / reference["sample_scores"]["relative_path"]

    def fake_write_scores(manifest: Path, predictions: Path, output: Path) -> dict[str, Path]:
        assert Path(manifest).name == "dev.jsonl"
        assert Path(predictions).name == "predictions.jsonl"
        output.mkdir(parents=True)
        aggregate = output / "aggregate.json"
        samples = output / "sample_scores.jsonl"
        shutil.copy2(reference_aggregate, aggregate)
        shutil.copy2(reference_samples, samples)
        return {"aggregate": aggregate, "samples": samples}

    monkeypatch.setattr(mobile_regression, "verify_checkpoint", fake_verify)
    monkeypatch.setattr(mobile_regression, "generate_manifest", fake_generate)
    monkeypatch.setattr(mobile_regression, "write_scores", fake_write_scores)

    output = tmp_path / "output"
    result = mobile_regression.run_mobile_regression(
        checkpoint_dir=checkpoint,
        checkpoint_hashes_path=hash_path,
        output_dir=output,
        repository_root=ROOT,
        preregistration_path=PREREGISTRATION,
        runtime_environment=reference["environment"],
    )

    assert result["gate"]["passed"] is True
    assert result["gate"]["candidate"]["numerator"] == 602
    assert result["population"]["official_evaluation_artifacts_accessed"] == []
    assert (output / "scores" / "sample_scores.jsonl").is_file()
    assert (output / "paired-samples.jsonl").is_file()
    assert len((output / "paired-samples.jsonl").read_text().splitlines()) == 756
    artifact_manifest = json.loads((output / "artifact-sha256.json").read_text())
    assert "result.json" in artifact_manifest["file_sha256"]
    assert "paired-samples.jsonl" in artifact_manifest["file_sha256"]
