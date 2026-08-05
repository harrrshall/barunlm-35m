from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from barunlm import BarunConfig, BarunLM
from barunlm.training import (
    EXAMPLE_SCHEMA_VERSION,
    IGNORE_INDEX,
    SFTExample,
    TrainingRunConfig,
    collate_sft,
    tokenize_examples,
    train_sft,
)
from barunlm.training import trainer as trainer_module
from barunlm.training.data import ManifestError, load_manifest, sha256_file


def toy_tokenizer() -> Tokenizer:
    vocabulary = {
        "<pad>": 0,
        "<unk>": 1,
        "<eos>": 2,
        "train": 3,
        "red": 4,
        "blue": 5,
        "green": 6,
        "eval": 7,
        "answer": 8,
        "yes": 9,
        "no": 10,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def tiny_config() -> BarunConfig:
    return BarunConfig(
        vocab_size=11,
        dim=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim=16,
        max_seq_len=12,
        rope_fraction=0.5,
        local_window=4,
        full_attention_every=1,
        attention_gate=False,
        qk_norm=True,
        residual_select_every=0,
        dropout=0.0,
        tie_embeddings=True,
        mtp_offset=2,
        mtp_loss_weight=0.0,
    )


def example(example_id: str, prompt: str, target: str) -> SFTExample:
    return SFTExample(
        example_id=example_id,
        prompt=prompt,
        target=target,
        metadata={},
        content_sha256=example_id.rjust(64, "0"),
    )


def test_response_only_labels_mask_prompt_and_padding() -> None:
    tokenizer = toy_tokenizer()
    rows, rejected = tokenize_examples(
        [
            example("1", "train red", "answer yes"),
            example("2", "train", "answer no"),
        ],
        tokenizer,
        eos_token_id=2,
        max_seq_len=12,
    )

    assert not rejected
    assert rows[0].input_ids == (3, 4, 8, 9, 2)
    assert rows[0].labels == (IGNORE_INDEX, IGNORE_INDEX, 8, 9, 2)
    batch = collate_sft(rows, pad_token_id=0)
    assert batch.input_ids.shape == (2, 5)
    assert batch.labels[1, -1].item() == IGNORE_INDEX
    assert not batch.attention_mask[1, -1].item()
    assert batch.target_tokens == 6

    torch.manual_seed(3)
    model = BarunLM(tiny_config()).eval()
    output = model(
        batch.input_ids,
        labels=batch.labels,
        attention_mask=batch.attention_mask,
    )
    expected = F.cross_entropy(
        output.logits[:, :-1].reshape(-1, model.config.vocab_size),
        batch.labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    assert output.loss is not None
    torch.testing.assert_close(output.loss, expected)


def test_overlength_rows_are_rejected_without_truncation() -> None:
    rows, rejected = tokenize_examples(
        [example("long", "train red", "answer yes")],
        toy_tokenizer(),
        eos_token_id=2,
        max_seq_len=4,
    )

    assert rows == []
    assert len(rejected) == 1
    assert rejected[0].reason == "overlength"
    assert rejected[0].encoded_tokens == 5
    assert rejected[0].max_tokens == 4


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _manifest_record(example_id: str, prompt: str, target: str) -> dict[str, object]:
    return {
        "schema_version": EXAMPLE_SCHEMA_VERSION,
        "id": example_id,
        "prompt": prompt,
        "target": target,
        "metadata": {"license": "test-only"},
    }


def test_sft_loader_rejects_official_eval_and_wrong_derived_split(tmp_path: Path) -> None:
    manifest = tmp_path / "held-out.jsonl"
    record = _manifest_record("eval-1", "eval green", "answer yes")
    record["metadata"] = {
        "dataset": "google/mobile-actions",
        "source_split": "eval",
        "derived_split": "final_eval",
    }
    _write_jsonl(manifest, [record])

    with pytest.raises(ManifestError, match="official held-out source_split"):
        load_manifest(
            manifest,
            expected_sha256=sha256_file(manifest),
            expected_derived_split="train",
        )

    record["metadata"] = {
        "dataset": "google/mobile-actions",
        "source_split": "train",
        "derived_split": "dev",
    }
    _write_jsonl(manifest, [record])
    with pytest.raises(ManifestError, match="expected metadata.derived_split 'train'"):
        load_manifest(
            manifest,
            expected_sha256=sha256_file(manifest),
            expected_derived_split="train",
        )


def test_tiny_cpu_training_run_lowers_loss_and_resumes_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    torch.manual_seed(11)
    base_model = BarunLM(tiny_config())
    save_model(base_model, base_dir / "model.safetensors")
    base_model.config.save_json(base_dir / "barun_config.json")
    toy_tokenizer().save(str(base_dir / "tokenizer.json"))
    base_hashes = {
        name: sha256_file(base_dir / name)
        for name in ("model.safetensors", "barun_config.json", "tokenizer.json")
    }

    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    _write_jsonl(
        train_path,
        [
            _manifest_record("train-1", "train red", "answer yes"),
            _manifest_record("train-2", "train blue", "answer yes"),
            _manifest_record("train-3", "train green", "answer yes"),
            _manifest_record("train-4", "train red green", "answer yes"),
        ],
    )
    _write_jsonl(
        dev_path,
        [_manifest_record("dev-1", "eval green", "answer yes")],
    )
    config_path = tmp_path / "train_config.json"
    config_payload = {
        "schema_version": "barun-sft-config-v1",
        "run_id": "20260803-1200-cpu-smoke-s1",
        "output_root": str(tmp_path / "runs"),
        "hypothesis": "Response-only SFT should learn a repeated toy target.",
        "decision": "Reject the trainer if held-out response loss does not fall.",
        "base_checkpoint": {
            "source": "local",
            "local_dir": str(base_dir),
            "expected_sha256": base_hashes,
        },
        "data": {
            "train_manifest": str(train_path),
            "train_sha256": sha256_file(train_path),
            "dev_manifest": str(dev_path),
            "dev_sha256": sha256_file(dev_path),
            "max_seq_len": 12,
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "overlength_policy": "error",
        },
        "optimization": {
            "seed": 19,
            "epochs": 3,
            "batch_size": 2,
            "eval_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.02,
            "min_learning_rate_ratio": 0.1,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "adam_epsilon": 1e-8,
            "warmup_steps": 1,
            "max_steps": 4,
            "gradient_clip_norm": 1.0,
            "eval_every_steps": 2,
            "save_every_steps": 2,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 0.0,
        },
        "execution": {
            "device": "cpu",
            "precision": "fp32",
            "deterministic": True,
            "jarvis_resource_id": None,
            "estimated_hourly_cost": None,
        },
    }
    config_path.write_text(
        json.dumps(config_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    original_train_group = trainer_module._train_group
    calls = 0

    def interrupt_before_third_step(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return original_train_group(**kwargs)

    monkeypatch.setattr(trainer_module, "_train_group", interrupt_before_third_step)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_sft(TrainingRunConfig.from_json(config_path))
    monkeypatch.setattr(trainer_module, "_train_group", original_train_group)

    resume_checkpoint = (
        tmp_path / "runs" / "20260803-1200-cpu-smoke-s1" / "checkpoints" / "step-00000002"
    )
    assert resume_checkpoint.is_dir()
    config_payload["resume_from"] = str(resume_checkpoint)
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
    summary = train_sft(TrainingRunConfig.from_json(config_path))

    assert summary.status == "max_steps"
    assert summary.global_steps == 4
    assert summary.best_dev_loss < summary.initial_dev_loss
    assert Path(summary.final_checkpoint, "checkpoint_manifest.json").is_file()
    assert (Path(summary.run_dir) / "metrics.jsonl").is_file()
    assert (Path(summary.run_dir) / "run_manifest.json").is_file()

    del config_payload["resume_from"]
    config_payload["run_id"] = "20260803-1200-cpu-smoke-s2"
    uninterrupted_path = tmp_path / "uninterrupted_config.json"
    uninterrupted_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
    uninterrupted = train_sft(TrainingRunConfig.from_json(uninterrupted_path))
    resumed_tensors = load_file(Path(summary.final_checkpoint) / "model.safetensors")
    uninterrupted_tensors = load_file(Path(uninterrupted.final_checkpoint) / "model.safetensors")
    assert resumed_tensors.keys() == uninterrupted_tensors.keys()
    for name in resumed_tensors:
        assert torch.equal(resumed_tensors[name], uninterrupted_tensors[name]), name


def test_completion_only_training_never_opens_or_scores_development_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    torch.manual_seed(23)
    base_model = BarunLM(tiny_config())
    save_model(base_model, base_dir / "model.safetensors")
    base_model.config.save_json(base_dir / "barun_config.json")
    toy_tokenizer().save(str(base_dir / "tokenizer.json"))

    train_path = tmp_path / "full-train.jsonl"
    _write_jsonl(
        train_path,
        [
            _manifest_record("train-1", "train red", "answer yes"),
            _manifest_record("train-2", "train blue", "answer yes"),
        ],
    )
    forbidden_dev = tmp_path / "FORBIDDEN-DEV-LABELS.jsonl"
    config_path = tmp_path / "completion-only.json"
    config_payload = {
        "schema_version": "barun-sft-config-v1",
        "run_id": "20260804-1200-completion-only-s17",
        "output_root": str(tmp_path / "runs"),
        "hypothesis": "The frozen full population can be fit without a development read.",
        "decision": "Retain only the unconditional final checkpoint.",
        "base_checkpoint": {
            "source": "local",
            "local_dir": str(base_dir),
            "expected_sha256": {
                name: sha256_file(base_dir / name)
                for name in ("model.safetensors", "barun_config.json", "tokenizer.json")
            },
        },
        "data": {
            "train_manifest": str(train_path),
            "train_sha256": sha256_file(train_path),
            "dev_manifest": str(forbidden_dev),
            "dev_sha256": "0" * 64,
            "max_seq_len": 12,
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "overlength_policy": "error",
        },
        "optimization": {
            "seed": 17,
            "epochs": 1,
            "batch_size": 1,
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.01,
            "min_learning_rate_ratio": 0.1,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "adam_epsilon": 1e-8,
            "warmup_steps": 0,
            "max_steps": None,
            "gradient_clip_norm": 1.0,
            "eval_every_steps": 1,
            "save_every_steps": 999,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0,
        },
        "execution": {
            "device": "cpu",
            "precision": "fp32",
            "deterministic": True,
            "jarvis_resource_id": None,
            "estimated_hourly_cost": None,
        },
    }
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    opened: list[Path] = []
    original_load_manifest = trainer_module.load_manifest

    def audited_load_manifest(path, **kwargs):
        resolved = Path(path).resolve()
        opened.append(resolved)
        if resolved == forbidden_dev.resolve():
            raise AssertionError("completion-only mode opened forbidden development labels")
        return original_load_manifest(path, **kwargs)

    monkeypatch.setattr(trainer_module, "load_manifest", audited_load_manifest)
    summary = train_sft(TrainingRunConfig.from_json(config_path), completion_only=True)

    assert opened == [train_path.resolve()]
    assert summary.global_steps == 2
    assert summary.initial_dev_loss is None
    assert summary.best_dev_loss is None
    assert summary.final_dev_loss is None
    assert summary.best_checkpoint is None
    run_dir = Path(summary.run_dir)
    assert not (run_dir / "best_checkpoint.json").exists()
    assert (Path(summary.final_checkpoint) / "checkpoint_manifest.json").is_file()
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["training_mode"] == "completion_only_no_dev"
    assert run_manifest["checkpoint_policy"] == "unconditional_final_only"
    assert run_manifest["data"]["dev_labels_read"] is False
    assert run_manifest["data"]["dev"] is None
    events = [
        json.loads(line)["event"]
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events == ["completion_only_contract", "train", "train", "complete"]


def test_diagnostic_dev_loss_cannot_select_checkpoint_or_change_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    torch.manual_seed(31)
    base_model = BarunLM(tiny_config())
    save_model(base_model, base_dir / "model.safetensors")
    base_model.config.save_json(base_dir / "barun_config.json")
    toy_tokenizer().save(str(base_dir / "tokenizer.json"))
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "selection.jsonl"
    _write_jsonl(
        train_path,
        [
            _manifest_record("train-1", "train red", "answer yes"),
            _manifest_record("train-2", "train blue", "answer yes"),
        ],
    )
    _write_jsonl(dev_path, [_manifest_record("dev-1", "eval green", "answer yes")])
    payload = {
        "schema_version": "barun-sft-config-v1",
        "run_id": "20260804-1300-diagnostic-a-s17",
        "output_root": str(tmp_path / "runs"),
        "hypothesis": "Diagnostic loss cannot influence the final checkpoint.",
        "decision": "Always retain the unconditional final step.",
        "base_checkpoint": {
            "source": "local",
            "local_dir": str(base_dir),
            "expected_sha256": {
                name: sha256_file(base_dir / name)
                for name in ("model.safetensors", "barun_config.json", "tokenizer.json")
            },
        },
        "data": {
            "train_manifest": str(train_path),
            "train_sha256": sha256_file(train_path),
            "dev_manifest": str(dev_path),
            "dev_sha256": sha256_file(dev_path),
            "max_seq_len": 12,
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "overlength_policy": "error",
        },
        "optimization": {
            "seed": 17,
            "epochs": 1,
            "batch_size": 1,
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.01,
            "min_learning_rate_ratio": 0.1,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "adam_epsilon": 1e-8,
            "warmup_steps": 0,
            "max_steps": None,
            "gradient_clip_norm": 1.0,
            "eval_every_steps": 1,
            "save_every_steps": 2,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0,
        },
        "execution": {
            "device": "cpu",
            "precision": "fp32",
            "deterministic": True,
            "jarvis_resource_id": None,
            "estimated_hourly_cost": None,
        },
    }

    def run_with_losses(run_id: str, losses: list[float]):
        payload["run_id"] = run_id
        config_path = tmp_path / f"{run_id}.json"
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        remaining = iter(losses)

        def fake_loss(*args, **kwargs):
            loss = next(remaining)
            return {"loss": loss, "perplexity": loss + 1, "target_tokens": 3, "examples": 1}

        monkeypatch.setattr(trainer_module, "evaluate_response_loss", fake_loss)
        return train_sft(TrainingRunConfig.from_json(config_path), diagnostic_dev_only=True)

    improving = run_with_losses("20260804-1300-diagnostic-a-s17", [100.0, 50.0, 1.0])
    worsening = run_with_losses("20260804-1301-diagnostic-b-s17", [1.0, 50.0, 100.0])

    for summary in (improving, worsening):
        assert summary.status == "max_epochs"
        assert summary.global_steps == 2
        assert summary.best_dev_loss is None
        assert summary.best_checkpoint is None
        run_dir = Path(summary.run_dir)
        assert not (run_dir / "best_checkpoint.json").exists()
        marker = json.loads((run_dir / "heldout_access_started.json").read_text())
        assert marker["retry_lock"] == "forbidden"
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["training_mode"] == "diagnostic_dev_loss_unconditional_final"
        assert manifest["data"]["dev_loss_role"] == "diagnostic_only"
    improving_weights = load_file(Path(improving.final_checkpoint) / "model.safetensors")
    worsening_weights = load_file(Path(worsening.final_checkpoint) / "model.safetensors")
    assert improving_weights.keys() == worsening_weights.keys()
    for name in improving_weights:
        assert torch.equal(improving_weights[name], worsening_weights[name]), name
