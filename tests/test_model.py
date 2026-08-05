from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from barunlm import BarunConfig, BarunLM
from barunlm import model as model_module

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(**overrides: object) -> BarunConfig:
    values: dict[str, object] = {
        "vocab_size": 32,
        "dim": 16,
        "n_layers": 2,
        "n_heads": 2,
        "n_kv_heads": 1,
        "ffn_dim": 32,
        "max_seq_len": 16,
        "rope_fraction": 0.5,
        "local_window": 3,
        "full_attention_every": 2,
        "attention_gate": True,
        "qk_norm": True,
        "residual_select_every": 0,
        "dropout": 0.0,
        "tie_embeddings": True,
        "mtp_offset": 2,
        "mtp_loss_weight": 0.0,
    }
    values.update(overrides)
    return BarunConfig(**values)


def test_release_config_and_parameter_count() -> None:
    config = BarunConfig.from_json(ROOT / "barun_config.json")
    model = BarunLM(config)

    assert config.ffn_dim == 1_228
    assert config.mtp_loss_weight == 0.0
    assert model.parameter_counts() == {
        "total": 35_072_768,
        "embedding": 7_340_032,
        "non_embedding": 27_732_736,
    }
    assert model.lm_head.weight is model.embedding.weight


def test_default_config_matches_release_json() -> None:
    payload = json.loads((ROOT / "barun_config.json").read_text())
    assert BarunConfig() == BarunConfig(**payload)


def test_forward_shape_and_finite_logits() -> None:
    model = BarunLM(BarunConfig()).eval()
    with torch.inference_mode():
        output = model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    assert output.logits.shape == (1, 4, 16_384)
    assert torch.isfinite(output.logits).all()


def test_causal_loss_uses_conventional_next_token_shift() -> None:
    torch.manual_seed(1)
    model = BarunLM(tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    output = model(input_ids, labels=input_ids)
    expected = F.cross_entropy(
        output.logits[:, :-1].reshape(-1, model.config.vocab_size),
        input_ids[:, 1:].reshape(-1),
    )

    assert output.causal_loss is not None
    torch.testing.assert_close(output.causal_loss, expected)


def test_completion_masks_and_mtp_targets_align_to_label_positions() -> None:
    torch.manual_seed(2)
    model = BarunLM(tiny_config(mtp_loss_weight=0.2)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, 3, -100, 5]], dtype=torch.long)
    captured_mtp_hidden: list[torch.Tensor] = []
    assert model.mtp_norm is not None
    hook = model.mtp_norm.register_forward_hook(
        lambda _module, _inputs, output: captured_mtp_hidden.append(output)
    )

    output = model(input_ids, labels=labels)
    hook.remove()

    expected_causal = F.cross_entropy(
        output.logits[:, :-1].reshape(-1, model.config.vocab_size),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    expected_mtp = F.cross_entropy(
        model.lm_head(captured_mtp_hidden[0]).reshape(-1, model.config.vocab_size),
        labels[:, model.config.mtp_offset :].reshape(-1),
        ignore_index=-100,
    )
    assert output.causal_loss is not None
    assert output.mtp_loss is not None
    torch.testing.assert_close(output.causal_loss, expected_causal)
    torch.testing.assert_close(output.mtp_loss, expected_mtp)
    torch.testing.assert_close(
        output.loss, expected_causal + model.config.mtp_loss_weight * expected_mtp
    )


def test_all_ignored_targets_return_finite_differentiable_zero() -> None:
    torch.manual_seed(3)
    model = BarunLM(tiny_config(mtp_loss_weight=0.2)).train()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)

    output = model(input_ids, labels=labels)

    assert output.loss is not None and output.loss.requires_grad
    assert output.causal_loss is not None and output.causal_loss.item() == 0.0
    assert output.mtp_loss is not None and output.mtp_loss.item() == 0.0
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.embedding.weight.grad is not None
    assert torch.count_nonzero(model.embedding.weight.grad) == 0


def test_single_token_sequence_has_finite_zero_causal_loss() -> None:
    model = BarunLM(tiny_config()).train()
    input_ids = torch.tensor([[1]], dtype=torch.long)

    output = model(input_ids, labels=input_ids)

    assert output.loss is not None and output.loss.item() == 0.0
    output.loss.backward()


def test_cached_local_mask_created_in_inference_mode_is_safe_for_backward() -> None:
    model_module._local_causal_mask.cache_clear()
    torch.manual_seed(4)
    model = BarunLM(tiny_config())
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    model.eval()
    with torch.inference_mode():
        inference_output = model(input_ids)
    assert torch.isfinite(inference_output.logits).all()
    cached_mask = model_module._local_causal_mask(4, 3, "cpu")
    assert not cached_mask.is_inference()

    model.train()
    training_output = model(input_ids, labels=input_ids)
    assert training_output.loss is not None
    training_output.loss.backward()
    assert model.embedding.weight.grad is not None


def test_integer_padding_mask_rejects_values_other_than_zero_and_one() -> None:
    model = BarunLM(tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    invalid = torch.tensor([[1, 1, 2, 1], [1, 1, 1, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="only 0 and 1"):
        model(input_ids, attention_mask=invalid)


def test_standard_batched_padding_mask_matches_unpadded_for_valid_tokens() -> None:
    torch.manual_seed(5)
    model = BarunLM(tiny_config()).eval()
    padded = torch.tensor([[0, 0, 5, 6], [7, 8, 9, 10]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.long)

    with torch.inference_mode():
        batched = model(padded, attention_mask=attention_mask).logits
        short = model(torch.tensor([[5, 6]], dtype=torch.long)).logits
        long = model(torch.tensor([[7, 8, 9, 10]], dtype=torch.long)).logits

    torch.testing.assert_close(batched[0, 2:], short[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(batched[1], long[0], atol=2e-6, rtol=2e-5)


def test_additive_prefix_policy_mask_can_make_prompt_bidirectional() -> None:
    torch.manual_seed(6)
    model = BarunLM(tiny_config()).eval()
    first = torch.tensor([[1, 2, 3]], dtype=torch.long)
    changed_future = torch.tensor([[1, 9, 3]], dtype=torch.long)
    policy = torch.full((3, 3), float("-inf"))
    policy[torch.tril(torch.ones(3, 3, dtype=torch.bool))] = 0.0
    policy[0, 1] = 0.0

    with torch.inference_mode():
        causal_first = model(first).logits[:, 0]
        causal_changed = model(changed_future).logits[:, 0]
        prefix_first = model(first, attention_mask=policy).logits[:, 0]
        prefix_changed = model(changed_future, attention_mask=policy).logits[:, 0]

    torch.testing.assert_close(causal_first, causal_changed)
    assert not torch.allclose(prefix_first, prefix_changed)


def test_kv_cache_logits_match_full_forward() -> None:
    torch.manual_seed(7)
    model = BarunLM(tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

    with torch.inference_mode():
        expected = model(input_ids).logits
        cache = None
        pieces = []
        for position in range(input_ids.shape[1]):
            output = model(
                input_ids[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
                position_offset=position,
            )
            pieces.append(output.logits)
            cache = output.past_key_values
        actual = torch.cat(pieces, dim=1)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_left_padded_batched_generation_matches_individual_generation() -> None:
    torch.manual_seed(8)
    model = BarunLM(tiny_config()).eval()
    padded = torch.tensor([[0, 0, 5, 6], [7, 8, 9, 10]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.long)

    batched = model.generate(padded, max_new_tokens=2, temperature=0, attention_mask=attention_mask)
    short = model.generate(torch.tensor([[5, 6]]), max_new_tokens=2, temperature=0)
    long = model.generate(torch.tensor([[7, 8, 9, 10]]), max_new_tokens=2, temperature=0)

    assert torch.equal(batched[0, -2:], short[0, -2:])
    assert torch.equal(batched[1, -2:], long[0, -2:])


def test_generation_stops_when_every_sequence_emits_eos() -> None:
    model = BarunLM(tiny_config(tie_embeddings=False)).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    prompts = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)

    output = model.generate(
        prompts,
        max_new_tokens=5,
        temperature=0,
        eos_token_id=0,
        pad_token_id=1,
    )

    assert output.shape == (2, 3)
    assert torch.equal(output[:, -1], torch.zeros(2, dtype=torch.long))


def test_batched_generation_rejects_right_padding() -> None:
    model = BarunLM(tiny_config()).eval()
    input_ids = torch.tensor([[5, 6, 0], [7, 8, 9]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="left padding"):
        model.generate(input_ids, max_new_tokens=1, attention_mask=attention_mask)
