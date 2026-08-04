from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import barunlm.training.mobile_planir_experiment as subject

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mobile_planir_construction_screen_v1.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _specs(tmp_path: Path) -> tuple[subject.FitSpec, ...]:
    prefix = subject.RUN_ID.rsplit("-s", 1)[0]
    return tuple(
        subject.FitSpec(
            arm=arm,
            seed=seed,
            run_id=f"{prefix}-{arm.lower()}-s{seed}",
            train_manifest=tmp_path / f"train-{arm.lower()}.jsonl",
            train_sha256=hashlib.sha256(f"train-{arm}".encode()).hexdigest(),
        )
        for arm, seed in subject.FIT_ORDER
    )


def _fit(spec: subject.FitSpec, tmp_path: Path) -> subject.FitResult:
    checkpoint = tmp_path / f"checkpoint-{spec.arm}-{spec.seed}" / "step-00000073"
    checkpoint.mkdir(parents=True)
    for name, content in {
        "model.safetensors": f"model-{spec.arm}-{spec.seed}",
        "barun_config.json": "{}",
        "tokenizer.json": "{}",
    }.items():
        checkpoint.joinpath(name).write_text(content, encoding="utf-8")
    hashes = {
        name: subject.sha256_file(checkpoint / name)
        for name in ("barun_config.json", "model.safetensors", "tokenizer.json")
    }
    _write_json(
        checkpoint / "checkpoint_manifest.json",
        {
            "schema_version": "barun-sft-checkpoint-v1",
            "global_step": subject.EXPECTED_STEPS,
            "file_sha256": hashes,
        },
    )
    model_sha = hashes["model.safetensors"]
    receipt: dict[str, object] = {
        "arm": spec.arm,
        "seed": spec.seed,
        "run_id": spec.run_id,
        "train_rows": subject.TRAIN_ROWS,
        "train_sha256": spec.train_sha256,
        "completed_optimizer_steps": subject.EXPECTED_STEPS,
        "training_mode": "completion_only_no_dev",
        "checkpoint_policy": "unconditional_final_only",
        "development_labels_read": 0,
        "rejected_examples": 0,
        "base_reset_from_canonical_checkpoint": True,
        "checkpoint_file_sha256": hashes,
    }
    return subject.FitResult(
        spec=spec,
        final_checkpoint=checkpoint,
        checkpoint_file_sha256=hashes,
        model_sha256=model_sha,
        training_receipt=receipt,
    )


def _raw(fit: subject.FitResult, tmp_path: Path) -> subject.RawGenerationResult:
    directory = tmp_path / f"raw-{fit.spec.arm}-{fit.spec.seed}"
    directory.mkdir()
    path = directory / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(subject.SCREEN_ROWS):
            handle.write(json.dumps({"id": f"row-{index:04d}"}, sort_keys=True) + "\n")
    prediction_sha = subject.sha256_file(path)
    screen_sha = str(subject.EXPECTED_OUTPUT_SHA256[f"screen_{fit.spec.arm.lower()}"])
    generation = {
        "schema_version": subject.GENERATION_VERSION,
        "manifest_sha256": screen_sha,
        "predictions_sha256": prediction_sha,
        "checkpoint_sha256": dict(fit.checkpoint_file_sha256),
        "batch_size": subject.GENERATION_BATCH_SIZE,
        "max_new_tokens": subject.MAX_NEW_TOKENS,
        "examples": subject.SCREEN_ROWS,
    }
    summary_path = directory / "predictions.jsonl.manifest.json"
    _write_json(summary_path, generation)
    return subject.RawGenerationResult(
        fit=fit,
        screen_manifest_sha256=screen_sha,
        predictions_path=path,
        predictions_sha256=prediction_sha,
        summary_path=summary_path,
        summary_sha256=subject.sha256_file(summary_path),
        generation=generation,
    )


class FakeOperations:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.events: list[str] = []
        self.bad_fit = False
        self.bad_raw = False

    def train_fit(self, spec: subject.FitSpec) -> subject.FitResult:
        # The training capability must not expose any label-bearing screen path/hash.
        assert not hasattr(spec, "screen_manifest")
        assert not hasattr(spec, "screen_sha256")
        self.events.append(f"train:{spec.arm}:{spec.seed}")
        result = _fit(spec, self.tmp_path)
        if self.bad_fit and spec == _specs(self.tmp_path)[0]:
            receipt = dict(result.training_receipt)
            receipt["completed_optimizer_steps"] = 72
            result = replace(result, training_receipt=receipt)
        return result

    def freeze_checkpoints(self, fits: tuple[subject.FitResult, ...]) -> None:
        assert len(fits) == 9
        self.events.append("freeze-checkpoints")

    def begin_screen_access(self) -> None:
        self.events.append("begin-screen-access")

    def generate_raw(self, fit: subject.FitResult) -> subject.RawGenerationResult:
        self.events.append(f"generate:{fit.spec.arm}:{fit.spec.seed}")
        result = _raw(fit, self.tmp_path)
        if self.bad_raw and fit.spec.arm == "A" and fit.spec.seed == 17:
            result = replace(result, screen_manifest_sha256="0" * 64)
        return result

    def freeze_raw_predictions(self, rows: tuple[subject.RawGenerationResult, ...]) -> None:
        assert len(rows) == 9
        self.events.append("freeze-raw")

    def enrich_and_score(self, rows: tuple[subject.RawGenerationResult, ...]) -> dict[str, object]:
        assert len(rows) == 9
        self.events.append("score")
        return {"passed": False}


def test_fixed_schedule_enforces_irreversible_phase_order_and_least_privilege(
    tmp_path: Path,
) -> None:
    operations = FakeOperations(tmp_path)
    result = subject.run_fixed_schedule(_specs(tmp_path), operations)

    assert result == {"passed": False}
    assert operations.events[:9] == [f"train:{arm}:{seed}" for arm, seed in subject.FIT_ORDER]
    assert operations.events[9:11] == ["freeze-checkpoints", "begin-screen-access"]
    assert operations.events[11:20] == [f"generate:{arm}:{seed}" for arm, seed in subject.FIT_ORDER]
    assert operations.events[-2:] == ["freeze-raw", "score"]


def test_schedule_rejects_bad_fit_before_screen_access(tmp_path: Path) -> None:
    operations = FakeOperations(tmp_path)
    operations.bad_fit = True

    with pytest.raises(subject.MobilePlanIRExperimentError, match="training receipt"):
        subject.run_fixed_schedule(_specs(tmp_path), operations)

    assert "freeze-checkpoints" not in operations.events
    assert "begin-screen-access" not in operations.events


def test_schedule_rejects_bad_raw_before_freeze_or_score(tmp_path: Path) -> None:
    operations = FakeOperations(tmp_path)
    operations.bad_raw = True

    with pytest.raises(subject.MobilePlanIRExperimentError, match="wrong arm screen"):
        subject.run_fixed_schedule(_specs(tmp_path), operations)

    assert operations.events.count("begin-screen-access") == 1
    assert "freeze-raw" not in operations.events
    assert "score" not in operations.events


def _mutated_config(tmp_path: Path, mutate: Any) -> Path:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "config.json"
    _write_json(path, payload)
    return path


def test_current_config_is_strictly_accepted() -> None:
    payload, digest = subject.load_frozen_config(CONFIG)
    assert payload["run_id"] == subject.RUN_ID
    assert digest == subject.sha256_file(CONFIG)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda value: value["gate"].__setitem__("ast_margin_points_vs_each_control", 2),
            "gate changed",
        ),
        (
            lambda value: value["resource_policy"]["protected_resource_ids"].remove(463058),
            "protected-ID",
        ),
        (
            lambda value: value["token_compute_audit"]["train"]["C"].__setitem__(
                "full_sequence_tokens", 1
            ),
            "token/compute",
        ),
        (
            lambda value: value["claims"].__setitem__("maximum_positive_claim", "breakthrough"),
            "claim boundary",
        ),
    ],
)
def test_config_changes_fail_closed(tmp_path: Path, mutate: Any, match: str) -> None:
    with pytest.raises(subject.MobilePlanIRExperimentError, match=match):
        subject.load_frozen_config(_mutated_config(tmp_path, mutate))


def test_training_payload_is_completion_only_and_has_no_screen_capability(
    tmp_path: Path,
) -> None:
    frozen, _ = subject.load_frozen_config(CONFIG)
    spec = _specs(tmp_path)[0]
    payload = subject.build_training_payload(
        frozen_config=frozen,
        spec=spec,
        output_root=tmp_path / "training",
        ignored_dev_sentinel=tmp_path / "never-created.jsonl",
        jarvis_resource_id="999999",
    )

    assert payload["optimization"]["max_steps"] is None
    assert payload["optimization"]["batch_size"] == 63
    assert payload["optimization"]["epochs"] == 1
    assert payload["base_checkpoint"]["repo_id"] == subject.BASE_REPO
    assert payload["data"]["dev_sha256"] == "0" * 64
    assert "screen" not in payload["data"]
    assert "resume_from" not in payload


def _tiny_tokenizer(path: Path) -> str:
    tokenizer = Tokenizer(WordLevel(vocab={"<pad>": 0, "<eos>": 1, "hello": 2}, unk_token=None))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))
    return subject.sha256_file(path)


def test_enrichment_losslessly_copies_raw_finish_evidence_and_binds_presentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "SCREEN_ROWS", 1)
    checkpoint = tmp_path / "step-00000073"
    checkpoint.mkdir()
    tokenizer_sha = _tiny_tokenizer(checkpoint / "tokenizer.json")
    monkeypatch.setitem(subject.BASE_HASHES, "tokenizer.json", tokenizer_sha)
    _write_json(checkpoint / "checkpoint_manifest.json", {"ok": True})
    spec = subject.FitSpec(
        arm="B",
        seed=17,
        run_id="20260804-0545-test-b-s17",
        train_manifest=tmp_path / "train.jsonl",
        train_sha256="a" * 64,
    )
    fit = subject.FitResult(
        spec=spec,
        final_checkpoint=checkpoint,
        checkpoint_file_sha256={"model.safetensors": "c" * 64},
        model_sha256="c" * 64,
        training_receipt={},
    )
    generic_path = tmp_path / "generic.jsonl"
    generic_row = {
        "generated_tokens": 3,
        "generation_failure": None,
        "id": "sample-1",
        "prediction_raw": " raw output ",
        "prompt_tokens": 1,
        "truncated": True,
    }
    generic_path.write_text(json.dumps(generic_row) + "\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, {"ok": True})
    raw = subject.RawGenerationResult(
        fit=fit,
        screen_manifest_sha256="d" * 64,
        predictions_path=generic_path,
        predictions_sha256=subject.sha256_file(generic_path),
        summary_path=summary_path,
        summary_sha256=subject.sha256_file(summary_path),
        generation={},
    )
    prompt_sha = hashlib.sha256(b"hello").hexdigest()
    manifest = [
        {
            "id": "sample-1",
            "prompt": "hello",
            "target": "unused",
            "metadata": {
                "mobile_planir_screen": {
                    "arm": "B",
                    "source_id": "sample-1",
                    "output_prompt_sha256": prompt_sha,
                    "prompt_evidence": {
                        "prompt_sha256": prompt_sha,
                        "table_sha256": "e" * 64,
                    },
                }
            },
        }
    ]

    operations = object.__new__(subject.FixedScreenOperations)
    enriched, presentation = operations._enrich_one_run(
        raw, manifest, generation_source_sha256="f" * 64
    )

    assert enriched[0]["prediction_raw"] == " raw output "
    assert enriched[0]["truncated"] is True
    assert enriched[0]["generation_failure"] is None
    assert "catastrophic_unauthorized_action" not in enriched[0]
    assert enriched[0]["checkpoint_sha256"] == "c" * 64
    assert presentation["screen_manifest_sha256"] == "d" * 64
    assert presentation["generic_predictions_sha256"] == raw.predictions_sha256
    assert presentation["generation_summary_sha256"] == raw.summary_sha256
    assert presentation["finish_reasons"] == {
        "eos": 0,
        "max_new_tokens": 1,
        "failure": 0,
    }
    assert presentation["prompt_token_binding_sha256"]


def test_enrichment_rejects_hand_modified_generic_finish_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "SCREEN_ROWS", 1)
    checkpoint = tmp_path / "step-00000073"
    checkpoint.mkdir()
    tokenizer_sha = _tiny_tokenizer(checkpoint / "tokenizer.json")
    monkeypatch.setitem(subject.BASE_HASHES, "tokenizer.json", tokenizer_sha)
    _write_json(checkpoint / "checkpoint_manifest.json", {"ok": True})
    spec = _specs(tmp_path)[0]
    fit = _fit(spec, tmp_path)
    fit = replace(fit, final_checkpoint=checkpoint)
    generic_path = tmp_path / "generic.jsonl"
    generic_path.write_text(
        json.dumps(
            {
                "generated_tokens": 0,
                "generation_failure": None,
                "id": "sample-1",
                "prediction_raw": None,
                "prompt_tokens": 1,
                "truncated": "false",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    _write_json(summary, {})
    raw = subject.RawGenerationResult(
        fit=fit,
        screen_manifest_sha256="d" * 64,
        predictions_path=generic_path,
        predictions_sha256=subject.sha256_file(generic_path),
        summary_path=summary,
        summary_sha256=subject.sha256_file(summary),
        generation={},
    )
    prompt_sha = hashlib.sha256(b"hello").hexdigest()
    manifest = [
        {
            "id": "sample-1",
            "prompt": "hello",
            "metadata": {
                "mobile_planir_screen": {
                    "arm": "A",
                    "source_id": "sample-1",
                    "output_prompt_sha256": prompt_sha,
                    "prompt_evidence": {
                        "prompt_sha256": prompt_sha,
                        "table_sha256": "e" * 64,
                    },
                }
            },
        }
    ]

    operations = object.__new__(subject.FixedScreenOperations)
    with pytest.raises(subject.MobilePlanIRExperimentError, match="finish/failure"):
        operations._enrich_one_run(raw, manifest, generation_source_sha256="f" * 64)


def test_resource_id_rejects_every_durable_protected_machine() -> None:
    for machine_id in subject.PROTECTED_RESOURCE_IDS:
        with pytest.raises(subject.MobilePlanIRExperimentError, match="protected"):
            subject._resource_id(str(machine_id))


def test_observed_training_tokens_must_equal_frozen_arm_audit(tmp_path: Path) -> None:
    frozen, _ = subject.load_frozen_config(CONFIG)
    spec = _specs(tmp_path)[0]
    expected = frozen["token_compute_audit"]["train"]["A"]
    subject._validate_training_token_audit(
        frozen,
        spec=spec,
        observed={
            "encoded_tokens": expected["full_sequence_tokens"],
            "target_tokens": expected["target_tokens"],
        },
    )
    with pytest.raises(subject.MobilePlanIRExperimentError, match="token presentation"):
        subject._validate_training_token_audit(
            frozen,
            spec=spec,
            observed={
                "encoded_tokens": expected["full_sequence_tokens"] + 1,
                "target_tokens": expected["target_tokens"],
            },
        )


def test_raw_files_are_rehashed_against_freeze_receipt_before_scoring(
    tmp_path: Path,
) -> None:
    rows = tuple(_raw(_fit(spec, tmp_path), tmp_path) for spec in _specs(tmp_path))
    receipt = tmp_path / "all-raw-predictions-frozen.json"
    _write_json(receipt, subject._raw_freeze_payload(rows))
    subject._validate_raw_freeze_receipt(receipt, rows)

    with rows[0].predictions_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "post-freeze-mutation"}) + "\n")
    with pytest.raises(subject.MobilePlanIRExperimentError, match="raw generation artifacts"):
        subject._validate_raw_freeze_receipt(receipt, rows)


def test_raw_freeze_receipt_mutation_fails_before_scoring(tmp_path: Path) -> None:
    rows = tuple(_raw(_fit(spec, tmp_path), tmp_path) for spec in _specs(tmp_path))
    receipt = tmp_path / "all-raw-predictions-frozen.json"
    payload = subject._raw_freeze_payload(rows)
    payload["scorer_invoked"] = True
    _write_json(receipt, payload)

    with pytest.raises(subject.MobilePlanIRExperimentError, match="freeze receipt changed"):
        subject._validate_raw_freeze_receipt(receipt, rows)
