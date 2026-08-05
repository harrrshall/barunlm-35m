from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import barunlm.baselines.mobile_qwen05b_matched as matched
from barunlm.training.data import IGNORE_INDEX, SFTExample


def _prompt(user: str = "Turn on the flashlight.") -> str:
    return (
        "<bos><system>\n"
        "ACTION_IR_V1\n"
        "NOW 2026-08-03T19:30:00+05:30\n"
        "TOOLS\n"
        "turn_on_flashlight(): Turns the flashlight on.\n"
        "<user>\n"
        f"{user}\n"
        "<assistant>\n"
    )


def _example(example_id: str = "example-1", *, derived_split: str = "train") -> SFTExample:
    return SFTExample(
        example_id=example_id,
        prompt=_prompt(),
        target=(
            '{"calls":[{"args":{},"tool":"turn_on_flashlight"}],"decision":"CALL","mode":"SINGLE"}'
        ),
        metadata={"derived_split": derived_split, "source_split": "train"},
        content_sha256="1" * 64,
    )


class _NativeTokenizer:
    chat_template = "test-native-template-v1"
    eos_token_id = 99
    pad_token_id = 0
    emitted_eos_token_id = 99

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert [row["role"] for row in conversation[:2]] == ["system", "user"]
        prefix = [1, len(conversation[0]["content"]), 2, len(conversation[1]["content"]), 3]
        if add_generation_prompt:
            assert len(conversation) == 2
            return prefix + [4]
        assert len(conversation) == 3 and conversation[2]["role"] == "assistant"
        return prefix + [4, 10, len(conversation[2]["content"]), self.emitted_eos_token_id]

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return ",".join(str(value) for value in token_ids)


def _canonical_row(example_id: str, derived_split: str) -> dict[str, object]:
    return {
        "schema_version": "barun-sft-example-v1",
        "id": example_id,
        "prompt": _prompt(),
        "target": _example().target,
        "metadata": {"derived_split": derived_split, "source_split": "train"},
    }


def _write_manifest(path: Path, row: dict[str, object]) -> tuple[str, str]:
    path.write_text(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    artifact_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    membership = hashlib.sha256((str(row["id"]) + "\n").encode()).hexdigest()
    return artifact_hash, membership


def test_parse_action_prompt_removes_only_role_wrapper() -> None:
    parsed = matched.parse_action_prompt(_prompt("  Keep my spaces.  "))

    assert parsed.system_content == (
        "ACTION_IR_V1\n"
        "NOW 2026-08-03T19:30:00+05:30\n"
        "TOOLS\n"
        "turn_on_flashlight(): Turns the flashlight on."
    )
    assert parsed.user_content == "  Keep my spaces.  "
    assert parsed.messages() == (
        {"role": "system", "content": parsed.system_content},
        {"role": "user", "content": "  Keep my spaces.  "},
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "ACTION_IR_V1\nNOW x\nTOOLS\nx\n<user>\ny\n<assistant>\n",
        _prompt().replace("\n<assistant>\n", ""),
        _prompt().replace("\n<user>\n", "\n<user>\nextra\n<user>\n"),
        _prompt().replace("ACTION_IR_V1", "OTHER_IR"),
        _prompt().replace("\nTOOLS\n", "\nSCHEMAS\n"),
    ],
)
def test_parse_action_prompt_rejects_renderer_drift(prompt: str) -> None:
    with pytest.raises(matched.MatchedBaselineError):
        matched.parse_action_prompt(prompt)


def test_native_chat_tokenization_is_prefix_proved_and_response_only() -> None:
    tokenizer = _NativeTokenizer()
    rendered = matched.tokenize_native_chat_example(_example(), tokenizer, max_seq_len=128)

    assert rendered.prompt_ids == (1, 95, 2, 23, 3, 4)
    assert rendered.input_ids[: rendered.prompt_tokens] == rendered.prompt_ids
    assert rendered.input_ids[-1] == tokenizer.eos_token_id
    assert rendered.labels[: rendered.prompt_tokens] == (IGNORE_INDEX,) * rendered.prompt_tokens
    assert rendered.labels[rendered.prompt_tokens :] == rendered.input_ids[rendered.prompt_tokens :]
    assert rendered.response_tokens == 3
    assert rendered.to_training_example().target_tokens == 3


def test_native_chat_tokenization_refuses_missing_template_eos_and_overlength() -> None:
    tokenizer = _NativeTokenizer()
    tokenizer.chat_template = None
    with pytest.raises(matched.MatchedBaselineError, match="native chat template"):
        matched.tokenize_native_chat_example(_example(), tokenizer, max_seq_len=128)

    tokenizer.chat_template = "restored"
    tokenizer.eos_token_id = 123
    with pytest.raises(matched.MatchedBaselineError, match="no EOS"):
        matched.tokenize_native_chat_example(_example(), tokenizer, max_seq_len=128)

    tokenizer.eos_token_id = 99
    with pytest.raises(matched.MatchedBaselineError, match="above the frozen"):
        matched.tokenize_native_chat_example(_example(), tokenizer, max_seq_len=3)


def test_native_chat_tokenization_refuses_nonprefix_full_message() -> None:
    class DriftingTokenizer(_NativeTokenizer):
        def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
            ids = super().apply_chat_template(
                conversation,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
            if not add_generation_prompt:
                ids[0] = 44
            return ids

    with pytest.raises(matched.MatchedBaselineError, match="not prefixed"):
        matched.tokenize_native_chat_example(_example(), DriftingTokenizer(), max_seq_len=128)


def test_count_unique_parameters_deduplicates_tied_storage() -> None:
    class TiedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(7, 3)
            self.output = nn.Linear(3, 7, bias=False)
            self.output.weight = self.embedding.weight
            self.scale = nn.Parameter(torch.ones(2))

    assert matched.count_unique_parameters(TiedModel()) == 7 * 3 + 2


def test_frozen_recipe_is_hash_bound_and_one_recipe() -> None:
    recipe = matched.load_frozen_recipe()

    assert recipe["model"]["id"] == matched.MODEL_ID
    assert recipe["model"]["revision"] == matched.MODEL_REVISION
    assert recipe["model"]["expected_unique_parameters"] == 494_032_768
    assert recipe["trial_budget"]["recipes"] == 1
    assert recipe["optimization"]["effective_batch_size"] == 63
    assert recipe["optimization"]["expected_optimizer_steps"] == 126
    assert recipe["evaluation"]["generation_batch_size"] == 64
    assert recipe["information_budget"]["examples_presented"] == 7_937


def test_frozen_recipe_rejects_copy_even_if_json_is_unchanged(tmp_path: Path) -> None:
    copied = tmp_path / "recipe.json"
    copied.write_text(matched.RECIPE_PATH.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(matched.MatchedBaselineError, match="SHA-256 changed"):
        matched.load_frozen_recipe(copied)


def test_generation_overrides_neutralize_vendor_sampling_and_penalty() -> None:
    recipe = matched.load_frozen_recipe()
    overrides = recipe["evaluation"]["generation_config_overrides"]

    assert overrides == {
        "do_sample": False,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "temperature": None,
        "top_k": None,
        "top_p": None,
    }
    tokenizer = _NativeTokenizer()
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = 151643
    assert matched._greedy_generation_kwargs(
        tokenizer,
        max_new_tokens=192,
        generation_overrides=overrides,
    ) == {
        "do_sample": False,
        "eos_token_id": 151645,
        "length_penalty": 1.0,
        "max_new_tokens": 192,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "pad_token_id": 151643,
        "repetition_penalty": 1.0,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "use_cache": True,
    }


def test_run_preregistration_hashes_the_frozen_implementation() -> None:
    root = matched.RECIPE_PATH.parents[1]
    evidence_root = (
        root / "experiments" / "runs" / "20260803-2122-mobile-qwen05b-matched-s17" / "essential"
    )
    preregistration_path = evidence_root / "attempt-preregistration.json"
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    frozen = preregistration["frozen_implementation"]

    for name in ("module", "entrypoint", "requirements"):
        identity = frozen[name]
        assert matched.sha256_file(root / identity["path"]) == identity["sha256"]

    # Result documentation and independent verification may receive dated post-score
    # addenda/fixes. Preserve their preregistered identities in the immutable source receipt
    # instead of requiring the live handoff copy to masquerade as pre-score source.
    source_snapshot = json.loads(
        (evidence_root / "source-snapshot.json").read_text(encoding="utf-8")
    )
    for name in (
        "tests",
        "documentation",
        "independent_verifier",
        "independent_verifier_tests",
    ):
        identity = frozen[name]
        assert source_snapshot["files_sha256"][identity["path"]] == identity["sha256"]
    assert preregistration["external_model_or_service_contacted"] is True
    assert preregistration["external_metadata_service_contacted"] is True
    assert preregistration["external_model_inference_service_contacted"] is False
    assert preregistration["baseline_downloaded"] is False
    assert preregistration["jarvis_resource_created"] is False


def test_input_firewall_checks_memberships_and_opaque_official_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    train_hash, train_membership = _write_manifest(train_path, _canonical_row("train-1", "train"))
    dev_hash, dev_membership = _write_manifest(dev_path, _canonical_row("dev-1", "dev"))
    audit = {
        "artifacts": {
            "manifests": {
                "train": {
                    "filename": "train.jsonl",
                    "sha256": train_hash,
                    "membership_sha256": train_membership,
                },
                "dev": {
                    "filename": "dev.jsonl",
                    "sha256": dev_hash,
                    "membership_sha256": dev_membership,
                },
            }
        },
        "counts": {
            "derived": {"dev": 1, "train": 1},
            "source": {"official_eval_opaque_unparsed": 2},
        },
        "official_eval": {
            "labels_parsed": False,
            "lengths_computed": False,
            "materialized": False,
            "opaque_unparsed": True,
            "overlaps_computed": False,
            "prompts_parsed": False,
            "rows": 2,
            "source_sha256": "a" * 64,
            "summaries_computed": False,
            "targets_parsed": False,
            "tool_schemas_parsed": False,
        },
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(matched, "TRAIN_MANIFEST_SHA256", train_hash)
    monkeypatch.setattr(matched, "DEV_MANIFEST_SHA256", dev_hash)
    monkeypatch.setattr(matched, "TRAIN_MEMBERSHIP_SHA256", train_membership)
    monkeypatch.setattr(matched, "DEV_MEMBERSHIP_SHA256", dev_membership)
    monkeypatch.setattr(
        matched, "AUDIT_SHA256", hashlib.sha256(audit_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(matched, "TRAIN_ROWS", 1)
    monkeypatch.setattr(matched, "DEV_ROWS", 1)
    monkeypatch.setattr(matched, "OFFICIAL_EVAL_ROWS", 2)
    monkeypatch.setattr(matched, "OFFICIAL_SOURCE_SHA256", "a" * 64)

    verified = matched.verify_frozen_mobile_inputs(
        train_manifest=train_path,
        dev_manifest=dev_path,
        audit_path=audit_path,
    )
    assert [row.example_id for row in verified.train] == ["train-1"]
    assert [row.example_id for row in verified.dev] == ["dev-1"]

    audit["official_eval"]["targets_parsed"] = True
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        matched, "AUDIT_SHA256", hashlib.sha256(audit_path.read_bytes()).hexdigest()
    )
    with pytest.raises(matched.MatchedBaselineError, match="did not stay opaque"):
        matched.verify_frozen_mobile_inputs(
            train_manifest=train_path,
            dev_manifest=dev_path,
            audit_path=audit_path,
        )


def test_runner_has_no_source_or_official_eval_input() -> None:
    destinations = {action.dest for action in matched.build_parser()._actions}

    assert {"train_manifest", "dev_manifest", "audit"}.issubset(destinations)
    assert "source" not in destinations
    assert "test_manifest" not in destinations
    assert "official_eval" not in destinations


def test_compact_bundle_preserves_sample_evidence_but_excludes_optimizer(tmp_path: Path) -> None:
    export = tmp_path / "export"
    export.mkdir()
    for name in (
        "attempt-preregistration.json",
        "environment.json",
        "model-identity.json",
        "preregistration.json",
        "result.json",
        "snapshot-manifest.json",
        "source-snapshot.json",
    ):
        (export / name).write_text("{}\n", encoding="utf-8")
    (export / "repository-tests.log").write_text("passed\n", encoding="utf-8")
    training = export / "training"
    training.mkdir()
    for name in (
        "metrics.jsonl",
        "render-audit.json",
        "rendered-dev.jsonl",
        "rendered-train.jsonl",
        "summary.json",
    ):
        (training / name).write_text("{}\n", encoding="utf-8")
    (training / "optimizer.pt").write_bytes(b"large state")
    checkpoint = training / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    evaluation = export / "evaluation"
    evaluation.mkdir()
    (evaluation / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    data = export / "data"
    data.mkdir()
    (data / "audit.json").write_text("{}\n", encoding="utf-8")
    (data / "identity.json").write_text("{}\n", encoding="utf-8")

    essential = export / "essential"
    matched._copy_essential(export, essential)

    assert (essential / "training" / "rendered-train.jsonl").is_file()
    assert (essential / "evaluation" / "predictions.jsonl").is_file()
    assert (essential / "checkpoint" / "model.safetensors").is_file()
    assert not any(path.name == "optimizer.pt" for path in essential.rglob("*"))
    manifest = json.loads((essential / "bundle-manifest.json").read_text(encoding="utf-8"))
    assert manifest["official_eval_artifacts_included"] is False


def test_schedule_has_exact_warmup_and_final_floor() -> None:
    assert matched._lr_multiplier(12, total_steps=126, warmup_steps=12, minimum_ratio=0.1) == 1
    assert matched._lr_multiplier(126, total_steps=126, warmup_steps=12, minimum_ratio=0.1) == 0.1
