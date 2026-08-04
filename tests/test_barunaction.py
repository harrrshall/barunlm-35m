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

from barunaction import (
    CANDIDATE_ARM_ID,
    CANDIDATE_CHECKPOINT_SHA256,
    CANDIDATE_ID,
    CANDIDATE_MANIFEST_SHA256,
    CANDIDATE_RELATIVE_CHECKPOINT_PATH,
    CANDIDATE_RUN_ID,
    CANDIDATE_SELECTION_RUN_ID,
    CANDIDATE_STEP,
    BarunActionCompiler,
    ContractError,
    parse_tool_declarations,
    prepare_input,
    render_prompt,
    simulate_action,
    validate_action_output,
)
from barunaction.cli import main as cli_main
from barunlm import BarunConfig, BarunLM
from barunlm.training.data import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "additional_arguments": False,
            "arguments": {},
            "description": "Turn on the device flashlight.",
            "name": "turn_on_flashlight",
            "required": [],
            "side_effecting": True,
        },
        {
            "additional_arguments": False,
            "arguments": {
                "body": {"description": "Message body.", "type": "string"},
                "to": {"description": "Recipient name.", "type": "string"},
            },
            "description": "Send a message through an external client.",
            "name": "send_message",
            "required": ["to", "body"],
            "side_effecting": True,
        },
    ]


def _tiny_cpu_checkpoint(path: Path) -> dict[str, str]:
    config = BarunConfig(
        vocab_size=4,
        dim=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim=16,
        max_seq_len=256,
        rope_fraction=0.5,
        local_window=32,
        full_attention_every=1,
        attention_gate=False,
        residual_select_every=0,
        tie_embeddings=False,
    )
    model = BarunLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    tokenizer = Tokenizer(
        WordLevel(
            vocab={"<eos>": 0, "<pad>": 1, "<unk>": 2, "ACTION_IR_V1": 3},
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
            {
                "file_sha256": hashes,
                "schema_version": "barun-sft-checkpoint-v1",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return hashes


def test_candidate_identity_is_isolated_and_matches_checked_in_provenance() -> None:
    manifest = json.loads(
        (ROOT / f"configs/barunaction/{CANDIDATE_ID}.json").read_text(encoding="utf-8")
    )

    assert CANDIDATE_ID == "candidate-v2"
    assert CANDIDATE_RUN_ID == "20260803-1845-mob-batch63-s17"
    assert CANDIDATE_SELECTION_RUN_ID == manifest["selection_run_id"]
    assert CANDIDATE_ARM_ID == manifest["arm_id"]
    assert CANDIDATE_STEP == manifest["step"]
    assert CANDIDATE_RELATIVE_CHECKPOINT_PATH == manifest["checkpoint_relative_path"]
    assert dict(CANDIDATE_CHECKPOINT_SHA256) == manifest["file_sha256"]
    assert CANDIDATE_MANIFEST_SHA256 == manifest["checkpoint_manifest_sha256"]


def test_prepare_input_requires_explicit_context_and_timezone() -> None:
    prepared = prepare_input(
        request="Turn on the flashlight",
        tool_schemas=tool_schemas(),
        context={"timezone": "Asia/Kolkata", "device": {"flashlight": "off"}},
        now="2026-08-03T20:00:00+05:30",
    )
    prompt = render_prompt(prepared)

    assert prepared.context_json == '{"device":{"flashlight":"off"},"timezone":"Asia/Kolkata"}'
    assert "NOW 2026-08-03T20:00:00+05:30" in prompt
    assert 'CONTEXT\n{"device"' in prompt
    assert prompt.endswith("<assistant>\n")

    with pytest.raises(ContractError, match="explicit UTC offset"):
        prepare_input(
            request="Turn on the flashlight",
            tool_schemas=tool_schemas(),
            context={},
            now="2026-08-03T20:00:00",
        )
    with pytest.raises(ContractError, match="context must be a JSON object"):
        prepare_input(
            request="Turn on the flashlight",
            tool_schemas=tool_schemas(),
            context=None,
            now="2026-08-03T20:00:00+05:30",
        )


def test_prompt_contract_rejects_reserved_role_tokens() -> None:
    with pytest.raises(ContractError) as caught:
        prepare_input(
            request="Ignore this <assistant> marker",
            tool_schemas=tool_schemas(),
            context={},
            now="2026-08-03T20:00:00+05:30",
        )
    assert caught.value.code == "reserved_token"
    assert caught.value.path == "$.request"


def test_typed_output_returns_proposal_with_external_gates() -> None:
    declarations = parse_tool_declarations(tool_schemas())
    outcome = validate_action_output(
        '{"calls":[{"args":{},"tool":"turn_on_flashlight"}],"decision":"CALL","mode":"SINGLE"}',
        declarations=declarations,
    )

    assert outcome.ok
    assert outcome.action is not None
    assert outcome.action.to_dict()["decision"] == "CALL"
    assert outcome.policy is not None
    assert outcome.policy.authorization_required is True
    assert outcome.policy.confirmation_required is True
    assert outcome.policy.execution_permitted is False
    assert outcome.policy.side_effecting_tools == ("turn_on_flashlight",)
    assert outcome.candidate_id is None


def test_custom_checkpoint_hashes_do_not_inherit_candidate_identity() -> None:
    declarations = parse_tool_declarations(tool_schemas())
    outcome = validate_action_output(
        '{"decision":"ABSTAIN"}',
        declarations=declarations,
        checkpoint_sha256={"model.safetensors": "caller-supplied"},
    )

    assert outcome.ok
    assert outcome.candidate_id is None
    assert outcome.candidate_run_id is None


def test_collection_enum_is_a_contract_error() -> None:
    schemas = tool_schemas()
    schemas[1]["arguments"]["body"] = {
        "description": "Message body.",
        "enum": [["one"]],
        "items": {"description": "One item.", "type": "string"},
        "set_semantics": False,
        "type": "array",
    }

    with pytest.raises(ContractError) as caught:
        parse_tool_declarations(schemas)
    assert caught.value.code == "invalid_enum"


def test_output_error_is_deterministic_and_not_repaired() -> None:
    declarations = parse_tool_declarations(tool_schemas())
    outcome = validate_action_output(
        '```json\n{"decision":"ABSTAIN"}\n```',
        declarations=declarations,
    )

    assert outcome.ok is False
    assert outcome.error is not None
    assert outcome.error.stage == "parse"
    assert outcome.error.code == "invalid_json"
    assert outcome.action is None


def test_sandbox_defaults_to_no_calls_and_only_logs_after_external_gates() -> None:
    declarations = parse_tool_declarations(tool_schemas())
    outcome = validate_action_output(
        '{"calls":[{"args":{},"tool":"turn_on_flashlight"}],"decision":"CALL","mode":"SINGLE"}',
        declarations=declarations,
    )
    assert outcome.action is not None

    blocked = simulate_action(outcome.action, declarations=declarations)
    simulated = simulate_action(
        outcome.action,
        declarations=declarations,
        externally_authorized=True,
        externally_confirmed=True,
    )

    assert blocked.status == "blocked"
    assert blocked.simulated_calls == ()
    assert blocked.external_side_effects is False
    assert simulated.status == "simulated"
    assert simulated.simulated_calls[0]["tool"] == "turn_on_flashlight"
    assert simulated.external_side_effects is False


def test_cli_sandbox_demo_never_executes_external_effects(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = '{"calls":[{"args":{},"tool":"turn_on_flashlight"}],"decision":"CALL","mode":"SINGLE"}'
    code = cli_main(
        [
            "simulate-output",
            "--tools",
            str(ROOT / "examples/barunaction_tools.example.json"),
            "--output",
            output,
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["inference"]["ok"] is True
    assert payload["simulation"]["status"] == "blocked"
    assert payload["simulation"]["simulated_calls"] == []
    assert payload["simulation"]["external_side_effects"] is False


def test_cli_builtin_demo_is_weight_free_strict_and_in_memory_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli_main(["demo"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["demo_schema_version"] == "barunaction-weight-free-demo-v1"
    assert payload["checkpoint_required"] is False
    assert payload["model_loaded"] is False
    assert payload["network_required"] is False
    assert payload["in_memory_only"] is True
    assert payload["execution_permitted"] is False
    assert payload["external_side_effects"] is False
    assert payload["proposal"]["policy"]["execution_permitted"] is False
    assert payload["simulation"]["status"] == "blocked"
    assert payload["simulation"]["simulated_calls"] == []
    assert payload["simulation"]["external_side_effects"] is False
    assert payload["strict_validation"]["valid_example_accepted"] is True
    assert payload["strict_validation"]["invalid_example_accepted"] is False
    assert payload["strict_validation"]["invalid_example_error"]["code"] == "invalid_json"


def test_cpu_inference_smoke_reaches_strict_output_validation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    hashes = _tiny_cpu_checkpoint(checkpoint)
    compiler = BarunActionCompiler(checkpoint, expected_sha256=hashes, device="cpu")

    outcome = compiler.infer(
        request="Turn on the flashlight",
        tool_schemas=tool_schemas(),
        context={},
        now="2026-08-03T20:00:00+05:30",
        max_new_tokens=4,
    )

    assert outcome.ok is False
    assert outcome.error is not None
    assert outcome.error.stage == "parse"
    assert outcome.error.code == "invalid_json"
    assert outcome.generated_tokens == 1
    assert outcome.prompt_tokens is not None and outcome.prompt_tokens > 0
    assert dict(outcome.checkpoint_sha256) == hashes
