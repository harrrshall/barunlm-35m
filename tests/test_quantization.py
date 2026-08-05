from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from barunaction import BarunActionCompiler, QuantizationSmokeError, parse_int8_smoke_cases
from barunlm import BarunConfig, BarunLM
from barunlm.evaluation.generation import GenerationError, load_verified_model
from barunlm.quantization import (
    INT8_FORMAT_VERSION,
    INT8_MANIFEST_FILE,
    INT8_WEIGHTS_FILE,
    QuantizationError,
    export_dynamic_int8_checkpoint,
    load_verified_int8_model,
    verify_int8_checkpoint,
)
from barunlm.training.data import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _qengine() -> str:
    supported = tuple(
        engine for engine in torch.backends.quantized.supported_engines if engine != "none"
    )
    if not supported:
        raise AssertionError("the supported CPU test environment has no quantized engine")
    return "qnnpack" if "qnnpack" in supported else supported[0]


def _tiny_checkpoint(path: Path, *, tie_embeddings: bool = True) -> dict[str, str]:
    config = BarunConfig(
        vocab_size=8,
        dim=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim=16,
        max_seq_len=128,
        rope_fraction=0.5,
        local_window=32,
        full_attention_every=1,
        attention_gate=False,
        residual_select_every=0,
        tie_embeddings=tie_embeddings,
    )
    model = BarunLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    tokenizer = Tokenizer(
        WordLevel(
            vocab={
                "<eos>": 0,
                "<pad>": 1,
                "<unk>": 2,
                "ACTION_IR_V1": 3,
                "Turn": 4,
                "on": 5,
                "flashlight": 6,
                "{}": 7,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    path.mkdir()
    save_model(model, path / "model.safetensors")
    config.save_json(path / "barun_config.json")
    tokenizer.save(str(path / "tokenizer.json"))
    hashes = {
        name: sha256_file(path / name)
        for name in ("barun_config.json", "model.safetensors", "tokenizer.json")
    }
    (path / "checkpoint_manifest.json").write_text(
        json.dumps(
            {"file_sha256": hashes, "schema_version": "barun-sft-checkpoint-v1"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return hashes


@pytest.mark.filterwarnings("ignore:torch.ao.quantization is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:torch.quantize_per_tensor.*:UserWarning")
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")
def test_int8_export_load_preserves_tied_head_and_greedy_tokens(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "int8"
    source_hashes = _tiny_checkpoint(source)

    exported = export_dynamic_int8_checkpoint(
        source,
        output,
        expected_source_sha256=source_hashes,
        qengine=_qengine(),
    )
    verified = verify_int8_checkpoint(
        output,
        expected_manifest_sha256=exported.manifest_sha256,
    )
    int8_model, tokenizer, loaded = load_verified_int8_model(
        output,
        expected_manifest_sha256=exported.manifest_sha256,
    )
    float_model, _, _ = load_verified_model(source, expected_sha256=source_hashes)

    assert verified.manifest_sha256 == exported.manifest_sha256 == loaded.manifest_sha256
    assert (
        json.loads((output / INT8_MANIFEST_FILE).read_text())["schema_version"]
        == INT8_FORMAT_VERSION
    )
    assert verified.float_linear_modules == ("lm_head",)
    assert int8_model.lm_head.weight is int8_model.embedding.weight
    assert int8_model.embedding.weight.dtype is torch.float32
    for name in verified.quantized_modules:
        module = dict(int8_model.named_modules())[name]
        assert isinstance(module, torch.ao.nn.quantized.dynamic.Linear)
        assert module.weight().dtype is torch.qint8

    input_ids = torch.tensor([[3]], dtype=torch.long)
    expected = float_model.generate(
        input_ids,
        max_new_tokens=4,
        temperature=0,
        eos_token_id=tokenizer.token_to_id("<eos>"),
        pad_token_id=tokenizer.token_to_id("<pad>"),
    )
    actual = int8_model.generate(
        input_ids,
        max_new_tokens=4,
        temperature=0,
        eos_token_id=tokenizer.token_to_id("<eos>"),
        pad_token_id=tokenizer.token_to_id("<pad>"),
    )
    assert torch.equal(actual, expected)

    compiler = BarunActionCompiler(
        output,
        checkpoint_format="int8",
        expected_int8_manifest_sha256=exported.manifest_sha256,
        device="cpu",
    )
    outcome = compiler.infer(
        request="Turn on the flashlight",
        tool_schemas=[
            {
                "additional_arguments": False,
                "arguments": {},
                "description": "Turn on the flashlight.",
                "name": "turn_on_flashlight",
                "required": [],
                "side_effecting": True,
            }
        ],
        context={},
        now="2026-08-03T20:00:00+05:30",
        max_new_tokens=4,
    )
    assert outcome.checkpoint_format == "int8"
    assert outcome.quantization_manifest_sha256 == exported.manifest_sha256
    assert dict(outcome.source_checkpoint_sha256 or {}) == source_hashes
    assert outcome.error is not None and outcome.error.code == "invalid_json"


@pytest.mark.filterwarnings("ignore:torch.ao.quantization is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:torch.quantize_per_tensor.*:UserWarning")
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")
def test_untied_output_head_is_quantized_and_zero_float_linears_are_valid(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "int8"
    hashes = _tiny_checkpoint(source, tie_embeddings=False)
    exported = export_dynamic_int8_checkpoint(
        source,
        output,
        expected_source_sha256=hashes,
        qengine=_qengine(),
    )

    model, _, info = load_verified_int8_model(
        output,
        expected_manifest_sha256=exported.manifest_sha256,
    )

    assert info.float_linear_modules == ()
    assert "lm_head" in info.quantized_modules
    assert isinstance(model.lm_head, torch.ao.nn.quantized.dynamic.Linear)


@pytest.mark.filterwarnings("ignore:torch.ao.quantization is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:torch.quantize_per_tensor.*:UserWarning")
def test_hash_verification_precedes_torch_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "int8"
    hashes = _tiny_checkpoint(source)
    exported = export_dynamic_int8_checkpoint(
        source,
        output,
        expected_source_sha256=hashes,
        qengine=_qengine(),
    )
    weights = output / INT8_WEIGHTS_FILE
    weights.write_bytes(weights.read_bytes() + b"tampered")

    def forbidden_load(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("torch.load must not run before hash verification")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(QuantizationError, match="SHA-256 mismatch"):
        load_verified_int8_model(
            output,
            expected_manifest_sha256=exported.manifest_sha256,
        )


def test_int8_loader_never_falls_back_to_float_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _tiny_checkpoint(source)

    with pytest.raises(QuantizationError, match="missing 'quantization_manifest.json'"):
        load_verified_int8_model(source, expected_manifest_sha256="0" * 64)
    with pytest.raises(GenerationError, match="missing 'quantization_manifest.json'"):
        BarunActionCompiler(
            source,
            checkpoint_format="int8",
            expected_int8_manifest_sha256="0" * 64,
            device="cpu",
        )
    with pytest.raises(GenerationError, match="require device='cpu'"):
        BarunActionCompiler(
            source,
            checkpoint_format="int8",
            expected_int8_manifest_sha256="0" * 64,
            device="cuda",
        )


def test_qengine_is_explicit_and_export_never_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "int8"
    hashes = _tiny_checkpoint(source)

    with pytest.raises(QuantizationError, match="cannot be 'none'"):
        export_dynamic_int8_checkpoint(
            source,
            output,
            expected_source_sha256=hashes,
            qengine="none",
        )
    assert not output.exists()

    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("owned by caller\n", encoding="utf-8")
    with pytest.raises(QuantizationError, match="refusing to overwrite"):
        export_dynamic_int8_checkpoint(
            source,
            output,
            expected_source_sha256=hashes,
            qengine=_qengine(),
        )
    assert marker.read_text(encoding="utf-8") == "owned by caller\n"


def test_manifest_hash_is_required_and_strict(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "int8"
    hashes = _tiny_checkpoint(source)
    exported = export_dynamic_int8_checkpoint(
        source,
        output,
        expected_source_sha256=hashes,
        qengine=_qengine(),
    )

    with pytest.raises(QuantizationError, match="manifest SHA-256 mismatch"):
        verify_int8_checkpoint(output, expected_manifest_sha256="0" * 64)
    assert exported.package_bytes == sum(
        path.stat().st_size for path in output.iterdir() if path.is_file()
    )
    (output / "unhashed.bin").write_bytes(b"unexpected")
    with pytest.raises(QuantizationError, match="file set differs"):
        verify_int8_checkpoint(
            output,
            expected_manifest_sha256=exported.manifest_sha256,
        )


def test_action_ir_smoke_cases_are_versioned_and_expected_outputs_are_valid() -> None:
    artifact = json.loads(
        (ROOT / "examples/barunaction_int8_smoke.example.json").read_text(encoding="utf-8")
    )
    cases = parse_int8_smoke_cases(artifact)

    assert tuple(case.case_id for case in cases) == (
        "turn-on-flashlight",
        "open-wifi-settings",
    )
    assert cases[0].expected_action["decision"] == "CALL"

    artifact["cases"][0]["expected_action"] = {"decision": "UNKNOWN"}
    with pytest.raises(QuantizationSmokeError, match="expected_action is invalid"):
        parse_int8_smoke_cases(artifact)
