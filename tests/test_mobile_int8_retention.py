from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from barunlm.evaluation import mobile_int8_retention
from barunlm.evaluation.evaluator import EVALUATOR_VERSION
from barunlm.evaluation.generation import INT8_GENERATION_VERSION
from barunlm.evaluation.mobile_actions import MOBILE_ACTIONS_SCORER_VERSION
from barunlm.training.data import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "barunaction" / "candidate-v2-arm64-int8-retention-v1.json"
DOWNLOADED_RETENTION_EVIDENCE = (
    ROOT / "experiments/runs/20260803-1810-mobile-blind-s17/remote/data/dev.jsonl",
    ROOT / "experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/"
    "final-eval/scores/sample_scores.jsonl",
)


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
        "sample_count": 756,
        "ast_exact_match": {
            "numerator": numerator,
            "denominator": 756,
            "value": numerator / 756,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_frozen_protocol_binds_candidate_artifact_population_and_gates() -> None:
    protocol = mobile_int8_retention.load_retention_protocol(PROTOCOL)

    assert sha256_file(PROTOCOL) == (
        "7229572bea461a7899b09c0310a77b9b4d8af74211d1b9ac8b2b5f77dc6b67cb"
    )
    assert protocol["candidate"]["candidate_id"] == "candidate-v2"
    assert protocol["int8_checkpoint"]["manifest_sha256"] == (
        "f45c391d18d78758b0d62eeb562d139d24cfe0be3c8a409e3b40c25945d95c6b"
    )
    assert protocol["dataset"]["manifest"] == {
        "relative_path": ("experiments/runs/20260803-1810-mobile-blind-s17/remote/data/dev.jsonl"),
        "sha256": "988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55",
        "rows": 756,
    }
    assert protocol["source_float_reference"]["aggregate"]["ast_exact_match"] == {
        "numerator": 602,
        "denominator": 756,
    }
    assert protocol["retention_gate"]["minimum_int8_numerator"] == 587
    assert protocol["retention_gate"]["maximum_float_to_int8_correct_rows_lost"] == 15
    assert protocol["evaluation"]["no_checkpoint_format_fallback"] is True


def test_protocol_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "protocol.json"
    changed.write_bytes(PROTOCOL.read_bytes() + b"\n")
    with pytest.raises(mobile_int8_retention.MobileInt8RetentionError, match="SHA-256 mismatch"):
        mobile_int8_retention.load_retention_protocol(changed)


def test_paired_retention_passes_at_587_and_fails_at_586() -> None:
    protocol = mobile_int8_retention.load_retention_protocol(PROTOCOL)
    float_samples = [_sample(index, exact=index < 602) for index in range(756)]
    boundary = [_sample(index, exact=index < 587) for index in range(756)]

    gate, paired = mobile_int8_retention.compare_retention_samples(
        int8_aggregate=_aggregate(587),
        int8_samples=boundary,
        float_samples=float_samples,
        protocol=protocol,
    )
    assert gate["passed"] is True
    assert gate["float_to_int8"]["correct_rows_lost"] == 15
    assert gate["paired_transition_counts"] == {
        "retained_correct": 587,
        "fixed": 0,
        "regressed": 15,
        "retained_incorrect": 154,
    }
    assert len(paired) == 756

    failed, _ = mobile_int8_retention.compare_retention_samples(
        int8_aggregate=_aggregate(586),
        int8_samples=[_sample(index, exact=index < 586) for index in range(756)],
        float_samples=float_samples,
        protocol=protocol,
    )
    assert failed["passed"] is False
    assert failed["decision"] == "reject_candidate_v2_int8_artifact"


def test_paired_retention_requires_same_ids_scenarios_and_gold() -> None:
    protocol = mobile_int8_retention.load_retention_protocol(PROTOCOL)
    float_samples = [_sample(index, exact=index < 602) for index in range(756)]
    int8_samples = [_sample(index, exact=index < 602) for index in range(756)]
    int8_samples[0]["gold"] = {"decision": "ABSTAIN"}

    with pytest.raises(mobile_int8_retention.MobileInt8RetentionError, match="gold AST mismatch"):
        mobile_int8_retention.compare_retention_samples(
            int8_aggregate=_aggregate(602),
            int8_samples=int8_samples,
            float_samples=float_samples,
            protocol=protocol,
        )


@pytest.mark.skipif(
    not all(path.is_file() for path in DOWNLOADED_RETENTION_EVIDENCE),
    reason="requires downloaded W&B Mobile development and float-reference evidence",
)
def test_receipt_requires_explicit_int8_generation_and_writes_hashed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = mobile_int8_retention.load_retention_protocol(PROTOCOL)
    candidate = protocol["candidate"]
    int8_contract = protocol["int8_checkpoint"]
    dataset = protocol["dataset"]
    manifest_path = ROOT / dataset["manifest"]["relative_path"]
    manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(manifest_rows) == 756
    float_samples_path = ROOT / protocol["source_float_reference"]["sample_scores"]["relative_path"]
    int8_samples = [json.loads(line) for line in float_samples_path.read_text().splitlines()]
    flipped = 0
    for row in int8_samples:
        if row["ast_exact"] and flipped < 15:
            row["ast_exact"] = False
            flipped += 1
    assert flipped == 15

    evaluation = tmp_path / "int8-evaluation"
    predictions = [
        {
            "generated_tokens": 1,
            "generation_failure": row.get("generation_failure"),
            "id": row["sample_id"],
            "prediction_raw": row.get("prediction_raw"),
            "prompt_tokens": 1,
            "truncated": row.get("truncated", False),
        }
        for row in int8_samples
    ]
    predictions_path = evaluation / "predictions.jsonl"
    _write_jsonl(predictions_path, predictions)
    scores = evaluation / "scores"
    _write_jsonl(scores / "sample_scores.jsonl", int8_samples)
    (scores / "aggregate.json").write_text(
        json.dumps(_aggregate(587), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = ROOT / int8_contract["relative_path"]
    runtime = dict(int8_contract["runtime"])
    qengine = runtime.pop("qengine")
    generation = {
        "batch_size": 16,
        "checkpoint_dir": str(checkpoint_path.resolve()),
        "checkpoint_format": "int8",
        "checkpoint_sha256": int8_contract["artifact_sha256"],
        "device": "cpu",
        "dtype": "torch.qint8_dynamic_with_float_remainder",
        "elapsed_seconds": 1.0,
        "examples": 756,
        "failed": 0,
        "generated": 756,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": dataset["manifest"]["sha256"],
        "max_new_tokens": 192,
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "qengine": qengine,
        "quantization_manifest_sha256": int8_contract["manifest_sha256"],
        "runtime": runtime,
        "schema_version": INT8_GENERATION_VERSION,
        "source_checkpoint_sha256": candidate["float_checkpoint_file_sha256"],
        "truncated": 0,
    }
    generation_path = evaluation / "predictions.jsonl.manifest.json"
    generation_path.write_text(
        json.dumps(generation, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    info = SimpleNamespace(
        artifact_sha256=int8_contract["artifact_sha256"],
        source_checkpoint_sha256=candidate["float_checkpoint_file_sha256"],
        manifest_sha256=int8_contract["manifest_sha256"],
        qengine=qengine,
        torch_version=runtime["torch_version"],
    )
    monkeypatch.setattr(
        mobile_int8_retention,
        "verify_int8_checkpoint",
        lambda *args, **kwargs: info,
    )

    receipt = tmp_path / "receipt"
    result = mobile_int8_retention.create_retention_receipt(
        repository_root=ROOT,
        protocol_path=PROTOCOL,
        int8_evaluation_dir=evaluation,
        output_dir=receipt,
    )
    assert result["gate"]["passed"] is True
    assert result["gate"]["int8"]["numerator"] == 587
    assert result["population"]["official_evaluation_artifacts_accessed"] == []
    assert len((receipt / "paired-samples.jsonl").read_text().splitlines()) == 756
    artifact_manifest = json.loads((receipt / "artifact-sha256.json").read_text())
    assert set(artifact_manifest["file_sha256"]) == {
        "paired-samples.jsonl",
        "protocol.json",
        "result.json",
    }

    generation["checkpoint_format"] = "float"
    generation_path.write_text(
        json.dumps(generation, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rejected_output = tmp_path / "rejected-receipt"
    with pytest.raises(
        mobile_int8_retention.MobileInt8RetentionError,
        match="did not explicitly select int8",
    ):
        mobile_int8_retention.create_retention_receipt(
            repository_root=ROOT,
            protocol_path=PROTOCOL,
            int8_evaluation_dir=evaluation,
            output_dir=rejected_output,
        )
    assert not rejected_output.exists()
