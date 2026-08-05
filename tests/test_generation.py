from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from barunlm import BarunConfig, BarunLM
from barunlm.evaluation.generation import (
    INT8_GENERATION_VERSION,
    GenerationError,
    generate_manifest,
)
from barunlm.quantization import export_dynamic_int8_checkpoint
from barunlm.training.data import EXAMPLE_SCHEMA_VERSION, sha256_file


def _tokenizer() -> Tokenizer:
    vocabulary = {
        "<pad>": 0,
        "<unk>": 1,
        "<eos>": 2,
        "prompt": 3,
        "answer": 4,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def _checkpoint(path: Path) -> dict[str, str]:
    config = BarunConfig(
        vocab_size=5,
        dim=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim=16,
        max_seq_len=8,
        rope_fraction=0.5,
        local_window=4,
        full_attention_every=1,
        attention_gate=False,
        residual_select_every=0,
        tie_embeddings=False,
    )
    model = BarunLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    path.mkdir()
    save_model(model, path / "model.safetensors")
    config.save_json(path / "barun_config.json")
    _tokenizer().save(str(path / "tokenizer.json"))
    return {
        name: sha256_file(path / name)
        for name in ("model.safetensors", "barun_config.json", "tokenizer.json")
    }


def test_generation_preserves_raw_special_tokens_and_records_truncation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    hashes = _checkpoint(checkpoint)
    manifest = tmp_path / "dev.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": EXAMPLE_SCHEMA_VERSION,
                "id": "one",
                "prompt": "prompt",
                "target": "answer",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "predictions.jsonl"

    summary = generate_manifest(
        checkpoint_dir=checkpoint,
        manifest_path=manifest,
        manifest_sha256=sha256_file(manifest),
        predictions_path=output,
        device_name="cpu",
        batch_size=1,
        max_new_tokens=2,
        expected_checkpoint_sha256=hashes,
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["prediction_raw"] == "<pad> <pad>"
    assert record["truncated"] is True
    assert record["generation_failure"] is None
    assert summary.generated == 1
    assert summary.truncated == 1
    assert summary.predictions_sha256 == sha256_file(output)
    assert "checkpoint_format" not in summary.to_dict()
    assert "quantization_manifest_sha256" not in summary.to_dict()


def _qengine() -> str:
    supported = tuple(
        engine for engine in torch.backends.quantized.supported_engines if engine != "none"
    )
    if not supported:
        raise AssertionError("the CPU test environment has no quantized engine")
    return "qnnpack" if "qnnpack" in supported else supported[0]


@pytest.mark.filterwarnings("ignore:torch.ao.quantization is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:torch.quantize_per_tensor.*:UserWarning")
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")
def test_generation_loads_explicit_int8_without_float_fallback(tmp_path: Path) -> None:
    source = tmp_path / "source"
    hashes = _checkpoint(source)
    int8 = tmp_path / "int8"
    exported = export_dynamic_int8_checkpoint(
        source,
        int8,
        expected_source_sha256=hashes,
        qengine=_qengine(),
    )
    manifest = tmp_path / "dev.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": EXAMPLE_SCHEMA_VERSION,
                "id": "one",
                "prompt": "prompt",
                "target": "answer",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = generate_manifest(
        checkpoint_dir=int8,
        checkpoint_format="int8",
        expected_int8_manifest_sha256=exported.manifest_sha256,
        manifest_path=manifest,
        manifest_sha256=sha256_file(manifest),
        predictions_path=tmp_path / "predictions.jsonl",
        device_name="cpu",
        batch_size=1,
        max_new_tokens=2,
    )

    receipt = summary.to_dict()
    assert summary.schema_version == INT8_GENERATION_VERSION
    assert receipt["checkpoint_format"] == "int8"
    assert receipt["quantization_manifest_sha256"] == exported.manifest_sha256
    assert receipt["checkpoint_sha256"] == dict(exported.artifact_sha256)
    assert receipt["source_checkpoint_sha256"] == hashes
    assert receipt["qengine"] == exported.qengine
    assert receipt["runtime"]["torch_version"] == str(torch.__version__)


@pytest.mark.parametrize(
    ("checkpoint_format", "device", "float_hashes", "int8_hash", "message"),
    [
        ("int8", "cuda", None, "a" * 64, "requires device_name='cpu'"),
        ("int8", "cpu", None, None, "requires an expected_int8_manifest"),
        ("int8", "cpu", {"model.safetensors": "a" * 64}, "a" * 64, "only with"),
        ("float", "cpu", None, "a" * 64, "only with"),
        ("automatic", "cpu", None, None, "explicitly 'float' or 'int8'"),
    ],
)
def test_generation_rejects_incompatible_checkpoint_arguments_before_writing(
    tmp_path: Path,
    checkpoint_format: str,
    device: str,
    float_hashes: dict[str, str] | None,
    int8_hash: str | None,
    message: str,
) -> None:
    predictions = tmp_path / "nested" / "predictions.jsonl"
    with pytest.raises(GenerationError, match=message):
        generate_manifest(
            checkpoint_dir=tmp_path / "missing-checkpoint",
            checkpoint_format=checkpoint_format,
            expected_checkpoint_sha256=float_hashes,
            expected_int8_manifest_sha256=int8_hash,
            manifest_path=tmp_path / "missing-manifest",
            manifest_sha256="b" * 64,
            predictions_path=predictions,
            device_name=device,
            batch_size=1,
            max_new_tokens=1,
        )
    assert not predictions.parent.exists()
