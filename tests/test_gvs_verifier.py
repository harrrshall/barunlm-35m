from __future__ import annotations

import copy
import hashlib
import inspect
import json
import types

import pytest
import torch
from safetensors.torch import load as safetensors_load
from torch import nn

import barunlm.evaluation.gvs_verifier as gvs_verifier_module
from barunlm.config import BarunConfig
from barunlm.evaluation.gvs_bridge import (
    GVS_ACTION_IR_CANONICALIZATION_VERSION,
    GVS_VERIFIER_BATCH_CONTRACT_VERSION,
    GVS_VERIFIER_TOKEN_ROW_CONTRACT_VERSION,
    GVS_VERIFIER_VALID_SET_CONTRACT,
)
from barunlm.evaluation.gvs_support import count_complete_gvs_system_parameters
from barunlm.evaluation.gvs_verifier import (
    PRODUCTION_SYSTEM_PARAMETERS,
    PRODUCTION_VERIFIER_PARAMETERS,
    FrozenLinearLoRA,
    GVSVerifier,
    GVSVerifierError,
    audit_verifier_architecture,
    build_verifier_contract,
    export_verifier_adapter,
    load_verifier_adapter,
    load_verifier_receipt_json,
    multi_positive_listwise_loss,
    select_valid_candidate,
    verifier_receipt,
    verify_verifier_receipt,
)
from barunlm.model import BarunLM


def _tiny_config() -> BarunConfig:
    return BarunConfig(
        vocab_size=32,
        dim=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=1,
        ffn_dim=32,
        max_seq_len=32,
        rope_fraction=0.5,
        local_window=8,
        full_attention_every=2,
        attention_gate=False,
        residual_select_every=0,
    )


def _tiny_verifier(seed: int = 17) -> GVSVerifier:
    torch.manual_seed(101)
    return GVSVerifier(BarunLM(_tiny_config()), rank=2, alpha=2, initialization_seed=seed)


def _candidate_identities(valid_mask: torch.Tensor) -> tuple[tuple[str | None, ...], ...]:
    rows: list[tuple[str | None, ...]] = []
    for row_index, validity in enumerate(valid_mask.to(device="cpu").tolist()):
        rows.append(
            tuple(
                hashlib.sha256(f"canonical-{row_index}-{slot}".encode()).hexdigest()
                if valid
                else None
                for slot, valid in enumerate(validity)
            )
        )
    return tuple(rows)


def test_production_contract_has_exact_135617_trainables_and_35208385_system() -> None:
    contract = build_verifier_contract(BarunConfig())
    accounting = count_complete_gvs_system_parameters(
        BarunConfig(), generator_parameters=35_072_768
    )
    assert contract.trainable_parameters == PRODUCTION_VERIFIER_PARAMETERS == 135_617
    assert accounting.complete_system_parameters == PRODUCTION_SYSTEM_PARAMETERS == 35_208_385
    assert len(contract.target_module_names) == 24
    assert contract.target_module_names[0] == "backbone.layers.0.attn.q_proj"
    assert contract.target_module_names[-1] == "backbone.layers.11.attn.v_proj"
    assert (
        contract.candidate_identity
        == "lowercase_sha256_of_lowered_action_ir_semantic_canonical_v1_utf8_bytes_for_valid_candidates"
    )
    assert contract.action_identity_contract_version == GVS_ACTION_IR_CANONICALIZATION_VERSION
    assert contract.valid_candidate_set == GVS_VERIFIER_VALID_SET_CONTRACT
    assert contract.token_row_contract_version == GVS_VERIFIER_TOKEN_ROW_CONTRACT_VERSION
    assert contract.batch_contract_version == GVS_VERIFIER_BATCH_CONTRACT_VERSION
    assert contract.learned_input_fields == ("input_ids", "attention_mask")
    assert contract.selection_tie_break == "lowered_semantic_candidate_sha256_lexicographic_asc"
    assert contract.launch_authorized is False


def test_zero_b_lora_preserves_backbone_logits_and_exact_trainable_roster() -> None:
    torch.manual_seed(7)
    backbone = BarunLM(_tiny_config()).eval()
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    before = backbone(ids, attention_mask=mask).logits.detach().clone()
    verifier = GVSVerifier(backbone, rank=2, alpha=2, initialization_seed=17).eval()
    after = verifier.backbone(ids, attention_mask=mask).logits.detach()
    assert torch.equal(before, after)
    score_before = verifier(ids, mask).detach().clone()
    with torch.no_grad():
        verifier.verifier_layers[0].attn.q_proj.lora_b.fill_(0.1)
    assert torch.equal(before, verifier.backbone(ids, attention_mask=mask).logits.detach())
    assert not torch.equal(score_before, verifier(ids, mask).detach())
    audit = audit_verifier_architecture(verifier)
    expected_count = 2 * 2 * (16 + 16) + 2 * 2 * (16 + 4) + 17
    assert audit["trainable_parameter_count"] == expected_count
    assert audit["generator_parameter_count"] == sum(
        parameter.numel() for parameter in verifier.backbone.parameters()
    )
    assert audit["unique_system_parameter_count"] == (
        audit["generator_parameter_count"] + expected_count
    )
    assert tuple(audit["trainable_parameter_names"])[-2:] == (
        "score_head.weight",
        "score_head.bias",
    )


def test_adapter_and_head_initialization_is_seed_deterministic() -> None:
    first = _tiny_verifier(17)
    second = _tiny_verifier(17)
    third = _tiny_verifier(18)
    assert torch.equal(
        first.verifier_layers[0].attn.q_proj.lora_a,
        second.verifier_layers[0].attn.q_proj.lora_a,
    )
    assert torch.equal(first.score_head.weight, second.score_head.weight)
    assert not torch.equal(
        first.verifier_layers[0].attn.q_proj.lora_a,
        third.verifier_layers[0].attn.q_proj.lora_a,
    )
    assert not torch.equal(first.score_head.weight, third.score_head.weight)
    assert torch.count_nonzero(first.verifier_layers[0].attn.q_proj.lora_b) == 0


def test_verifier_construction_does_not_consume_global_torch_rng() -> None:
    backbone = BarunLM(_tiny_config())
    state_before = torch.random.get_rng_state().clone()
    GVSVerifier(backbone, rank=2, alpha=2, initialization_seed=17)
    assert torch.equal(torch.random.get_rng_state(), state_before)


def test_bfloat16_backbone_keeps_float32_trainables_and_scores() -> None:
    backbone = BarunLM(_tiny_config()).to(dtype=torch.bfloat16)
    verifier = GVSVerifier(backbone, rank=2, alpha=2, initialization_seed=17).eval()
    input_ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    attention_mask = input_ids != 0

    scores = verifier(input_ids, attention_mask)

    assert scores.dtype is torch.float32
    assert {parameter.dtype for parameter in verifier.backbone.parameters()} == {torch.bfloat16}
    assert {parameter.dtype for parameter in verifier.parameters() if parameter.requires_grad} == {
        torch.float32
    }
    audit_verifier_architecture(verifier)


def test_backward_reaches_only_lora_and_head_parameters() -> None:
    verifier = _tiny_verifier()
    ids = torch.tensor(
        [[1, 2, 3, 4], [1, 5, 6, 0], [1, 7, 0, 0], [1, 8, 9, 10]],
        dtype=torch.long,
    )
    mask = ids != 0
    scores = verifier(ids, mask)
    tiled = scores.repeat(2).reshape(1, 8)
    exact = torch.zeros_like(tiled, dtype=torch.bool)
    exact[:, 0] = True
    valid = torch.ones_like(exact)
    loss = multi_positive_listwise_loss(
        tiled,
        exact_mask=exact,
        valid_mask=valid,
        candidate_identities=_candidate_identities(valid),
    )
    loss.backward()
    for name, parameter in verifier.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
        else:
            assert parameter.grad is None, name


def test_constructor_clears_stale_generator_gradients_and_audit_keeps_them_absent() -> None:
    backbone = BarunLM(_tiny_config())
    backbone.embedding.weight.grad = torch.ones_like(backbone.embedding.weight)
    verifier = GVSVerifier(backbone, rank=2, alpha=2, initialization_seed=17)
    assert verifier.backbone.embedding.weight.grad is None
    verifier.backbone.embedding.weight.grad = torch.ones_like(verifier.backbone.embedding.weight)
    with pytest.raises(GVSVerifierError, match="frozen generator parameter retains"):
        audit_verifier_architecture(verifier)


def test_right_padding_and_last_token_pooling_contract_fail_closed() -> None:
    verifier = _tiny_verifier()
    ids = torch.tensor([[1, 2, 0], [1, 3, 4]], dtype=torch.long)
    mask = ids != 0
    assert verifier(ids, mask).shape == (2,)
    hole = torch.tensor([[True, False, True], [True, True, True]])
    with pytest.raises(GVSVerifierError, match="contiguous right padding"):
        verifier(ids, hole)
    with pytest.raises(GVSVerifierError, match="boolean"):
        verifier(ids, mask.to(dtype=torch.int64))
    with pytest.raises(GVSVerifierError, match="non-padding"):
        verifier(ids, torch.zeros_like(mask))
    with pytest.raises(GVSVerifierError, match="embedding index dtype"):
        verifier(ids.to(dtype=torch.int16), mask)


def test_verifier_rejects_inputs_aliasing_frozen_backbone_storage() -> None:
    verifier = _tiny_verifier().eval()
    with torch.no_grad():
        verifier.backbone.embedding.weight.zero_()
    input_ids = (
        verifier.backbone.embedding.weight.detach().view(torch.int64).reshape(-1)[:2].view(1, 2)
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    with pytest.raises(GVSVerifierError, match="must not alias parameter or buffer storage"):
        verifier(input_ids, attention_mask)


def test_hidden_path_and_last_nonpadding_pool_match_the_backbone() -> None:
    verifier = _tiny_verifier().eval()
    ids = torch.tensor([[1, 2, 0], [1, 3, 4]], dtype=torch.long)
    mask = ids != 0
    captured: list[torch.Tensor] = []

    def capture(
        _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        captured.append(output.detach().clone())

    handle = verifier.backbone.final_norm.register_forward_hook(capture)
    try:
        verifier.backbone(ids, attention_mask=mask)
    finally:
        handle.remove()
    hidden = verifier._hidden_states(ids, mask)
    assert torch.equal(hidden, captured[0])
    lengths = mask.sum(dim=1)
    pooled = hidden[torch.arange(ids.shape[0]), lengths - 1].float()
    assert torch.equal(verifier(ids, mask), verifier.score_head(pooled).squeeze(-1))


def test_zero_adapter_hidden_path_matches_attention_gate_and_residual_selectors() -> None:
    config = BarunConfig(
        vocab_size=32,
        dim=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=1,
        ffn_dim=32,
        max_seq_len=32,
        rope_fraction=0.5,
        local_window=8,
        full_attention_every=2,
        attention_gate=True,
        residual_select_every=1,
    )
    verifier = GVSVerifier(BarunLM(config), rank=2, alpha=2, initialization_seed=17).eval()
    ids = torch.tensor([[1, 2, 0], [1, 3, 4]], dtype=torch.int32)
    mask = ids != 0
    captured: list[torch.Tensor] = []
    handle = verifier.backbone.final_norm.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach().clone())
    )
    try:
        verifier.backbone(ids, attention_mask=mask)
    finally:
        handle.remove()
    assert torch.equal(verifier._hidden_states(ids, mask), captured[0])
    assert verifier(ids, mask).dtype is torch.float32


def test_multi_positive_loss_matches_manual_probability_mass() -> None:
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    exact = torch.tensor([[False, True, False, True, False, False, False, False]])
    valid = torch.tensor([[True, True, True, True, False, False, False, False]])
    loss = multi_positive_listwise_loss(
        scores,
        exact_mask=exact,
        valid_mask=valid,
        candidate_identities=_candidate_identities(valid),
    )
    expected = torch.logsumexp(scores[0, :4], dim=0) - torch.logsumexp(scores[0, [1, 3]], dim=0)
    assert torch.equal(loss, expected)


@pytest.mark.parametrize(
    ("exact", "valid", "message"),
    [
        (
            torch.zeros((1, 8), dtype=torch.bool),
            torch.ones((1, 8), dtype=torch.bool),
            "at least one exact",
        ),
        (
            torch.ones((1, 8), dtype=torch.bool),
            torch.ones((1, 8), dtype=torch.bool),
            "at least one valid negative",
        ),
        (
            torch.tensor([[True, False, False, False, False, False, False, False]]),
            torch.tensor([[False, True, True, True, True, True, True, True]]),
            "cannot be invalid",
        ),
    ],
)
def test_listwise_loss_never_silently_drops_untrainable_rows(
    exact: torch.Tensor, valid: torch.Tensor, message: str
) -> None:
    with pytest.raises(GVSVerifierError, match=message):
        multi_positive_listwise_loss(
            torch.zeros((1, 8)),
            exact_mask=exact,
            valid_mask=valid,
            candidate_identities=_candidate_identities(valid),
        )


def test_selection_uses_only_valid_candidates_and_identity_ties() -> None:
    scores = torch.tensor([[5.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([[False, True, True, True, False, False, False, False]])
    identities = (
        (
            None,
            "f" * 64,
            "0" * 64,
            "1" * 64,
            None,
            None,
            None,
            None,
        ),
    )
    assert select_valid_candidate(scores, valid, identities).tolist() == [2]
    with pytest.raises(GVSVerifierError, match="retain a valid"):
        select_valid_candidate(
            scores,
            torch.zeros_like(valid),
            ((None,) * 8,),
        )


def test_selection_identity_tie_break_is_permutation_invariant() -> None:
    scores = torch.tensor([[4.0, 8.0, 8.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, True, True, False, False, False, False]])
    identities = (
        (
            "a" * 64,
            "f" * 64,
            "0" * 64,
            "b" * 64,
            None,
            None,
            None,
            None,
        ),
    )
    selected = select_valid_candidate(scores, valid, identities).item()
    assert identities[0][selected] == "0" * 64

    permutation = torch.tensor([2, 7, 1, 4, 0, 6, 3, 5])
    permuted_identities = (tuple(identities[0][slot] for slot in permutation.tolist()),)
    permuted_selected = select_valid_candidate(
        scores[:, permutation],
        valid[:, permutation],
        permuted_identities,
    ).item()
    assert permuted_identities[0][permuted_selected] == "0" * 64


def test_loss_and_selection_reject_duplicate_or_identity_bearing_invalid_slots() -> None:
    scores = torch.zeros((1, 8), dtype=torch.float32)
    valid = torch.ones((1, 8), dtype=torch.bool)
    duplicate = list(_candidate_identities(valid)[0])
    duplicate[1] = duplicate[0]
    duplicate_identities = (tuple(duplicate),)
    exact = torch.zeros_like(valid)
    exact[0, 0] = True
    with pytest.raises(GVSVerifierError, match="duplicate valid canonical actions"):
        multi_positive_listwise_loss(
            scores,
            exact_mask=exact,
            valid_mask=valid,
            candidate_identities=duplicate_identities,
        )
    with pytest.raises(GVSVerifierError, match="duplicate valid canonical actions"):
        select_valid_candidate(scores, valid, duplicate_identities)

    one_invalid = valid.clone()
    one_invalid[0, 7] = False
    invalid_identity = list(_candidate_identities(one_invalid)[0])
    invalid_identity[7] = "1" * 64
    with pytest.raises(GVSVerifierError, match="invalid/duplicate slot"):
        select_valid_candidate(scores, one_invalid, (tuple(invalid_identity),))


def test_architecture_audit_rejects_unfrozen_or_unshared_base() -> None:
    verifier = _tiny_verifier()
    verifier.backbone.layers[0].attn.k_proj.weight.requires_grad_(True)
    with pytest.raises(GVSVerifierError, match="trainable parameter roster changed"):
        audit_verifier_architecture(verifier)

    verifier = _tiny_verifier()
    verifier.verifier_layers[0].attn.q_proj.base = nn.Linear(16, 16, bias=False)
    with pytest.raises(GVSVerifierError, match="does not share the generator base"):
        audit_verifier_architecture(verifier)


def test_lora_rejects_wrong_base_shape_and_contract_rejects_bool_seed() -> None:
    with pytest.raises(GVSVerifierError, match="bias-free"):
        FrozenLinearLoRA(nn.Linear(4, 4, bias=True), rank=2, alpha=2, seed=1)
    with pytest.raises(GVSVerifierError, match="seed"):
        build_verifier_contract(_tiny_config(), rank=2, alpha=2, initialization_seed=True)
    with pytest.raises(GVSVerifierError, match="no room"):
        build_verifier_contract(_tiny_config(), rank=2, alpha=2, initialization_seed=2**63 - 1)


def test_receipt_binds_trainable_state_and_remains_nonauthorizing() -> None:
    verifier = _tiny_verifier()
    receipt = verifier_receipt(verifier)
    assert receipt["launch_authorized"] is False
    assert receipt["authorizes_model_or_label_access"] is False
    assert receipt["authorizes_cuda_or_jarvis_access"] is False
    assert receipt["backbone_checkpoint_receipt_required"] is True
    assert receipt["runtime"]["python_implementation"]
    assert len(receipt["runtime_sha256"]) == 64
    assert len(receipt["source_sha256"]) == 64
    assert len(receipt["trainable_state_sha256"]) == 64
    assert len(receipt["receipt_sha256"]) == 64
    changed = copy.deepcopy(receipt)
    changed["trainable_state"][0]["sha256"] = "0" * 64
    assert changed["receipt_sha256"] == receipt["receipt_sha256"]
    with pytest.raises(GVSVerifierError, match="live recomputation"):
        verify_verifier_receipt(verifier, changed)

    raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    assert load_verifier_receipt_json(raw, verifier=verifier) == receipt
    with pytest.raises(GVSVerifierError, match="duplicate JSON key"):
        load_verifier_receipt_json(
            b'{"schema_version":"x","schema_version":"y"}', verifier=verifier
        )


def test_receipt_verifier_detects_live_trainable_state_mutation() -> None:
    verifier = _tiny_verifier()
    receipt = verifier_receipt(verifier)
    with torch.no_grad():
        verifier.score_head.bias.add_(1)
    with pytest.raises(GVSVerifierError, match="live recomputation"):
        verify_verifier_receipt(verifier, receipt)


def test_receipt_rejects_behavior_attribute_mutation_without_tensor_changes() -> None:
    verifier = _tiny_verifier()
    receipt = verifier_receipt(verifier)
    verifier.backbone.final_norm.eps = 0.5
    with pytest.raises(GVSVerifierError, match="epsilon differs"):
        verify_verifier_receipt(verifier, receipt)


def test_receipt_binds_frozen_state_and_execution_surface() -> None:
    verifier = _tiny_verifier()
    receipt = verifier_receipt(verifier)
    assert len(receipt["complete_state_sha256"]) == 64
    assert len(receipt["model_source_sha256"]) == 64
    with torch.no_grad():
        verifier.backbone.embedding.weight[0, 0].add_(1)
    with pytest.raises(GVSVerifierError, match="live recomputation"):
        verify_verifier_receipt(verifier, receipt)

    hooked = _tiny_verifier()
    handle = hooked.score_head.register_forward_hook(lambda _m, _i, output: output)
    try:
        with pytest.raises(GVSVerifierError, match="forbidden execution hook"):
            audit_verifier_architecture(hooked)
    finally:
        handle.remove()


def test_global_pre_hook_is_rejected_before_it_can_run_on_verifier() -> None:
    verifier = _tiny_verifier()
    calls: list[str] = []
    handle = torch.nn.modules.module.register_module_forward_pre_hook(
        lambda module, _inputs: calls.append(type(module).__name__)
    )
    try:
        ids = torch.tensor([[1, 2]], dtype=torch.long)
        with pytest.raises(GVSVerifierError, match="global execution hook"):
            verifier(ids, torch.ones_like(ids, dtype=torch.bool))
        assert calls == []
    finally:
        handle.remove()


def test_parameter_gradient_hook_is_rejected_before_scoring() -> None:
    verifier = _tiny_verifier()
    handle = verifier.score_head.weight.register_hook(lambda gradient: gradient * 0)
    try:
        with pytest.raises(GVSVerifierError, match="forbidden tensor hook"):
            audit_verifier_architecture(verifier)
    finally:
        handle.remove()


def test_runtime_dependency_replacement_fails_before_scoring_or_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _tiny_verifier().eval()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    original = gvs_verifier_module.F.linear

    def shifted(*args: object, **kwargs: object) -> torch.Tensor:
        return original(*args, **kwargs) + 1  # type: ignore[arg-type]

    monkeypatch.setattr(gvs_verifier_module.F, "linear", shifted)
    with pytest.raises(GVSVerifierError, match="dependency binding changed"):
        verifier(ids, mask)
    with pytest.raises(GVSVerifierError, match="dependency binding changed"):
        verifier_receipt(verifier)


def test_replacing_origin_and_captured_alias_together_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _tiny_verifier().eval()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    original = gvs_verifier_module.F.linear

    def shifted(*args: object, **kwargs: object) -> torch.Tensor:
        return original(*args, **kwargs) + 1  # type: ignore[arg-type]

    monkeypatch.setattr(gvs_verifier_module.F, "linear", shifted)
    monkeypatch.setattr(gvs_verifier_module, "_F_LINEAR", shifted)
    with pytest.raises(GVSVerifierError, match="dependency binding changed"):
        verifier(ids, mask)


def test_receipt_binds_nonpersistent_buffers_and_shared_alias_topology() -> None:
    verifier = _tiny_verifier()
    receipt = verifier_receipt(verifier)
    topology = receipt["direct_state_topology"]
    assert isinstance(topology, dict)
    slots = topology["tensor_slots"]
    assert isinstance(slots, list)
    nonpersistent = [slot for slot in slots if slot["persistent"] is False]
    assert any(str(slot["slot_path"]).endswith(".rope.cos") for slot in nonpersistent)
    q_base = [
        slot
        for slot in slots
        if str(slot["slot_path"]).endswith("layers.0.attn.q_proj.base.weight")
    ]
    assert q_base
    assert q_base[0]["alias_of"] == "backbone.layers.0.attn.q_proj.weight"


def test_receipt_loader_bounds_huge_integers() -> None:
    verifier = _tiny_verifier()
    with pytest.raises(GVSVerifierError, match="oversized integer"):
        load_verifier_receipt_json(
            b'{"value":' + b"9" * 129 + b"}",
            verifier=verifier,
        )


def test_audit_rejects_distinct_parameter_views_sharing_trainable_storage() -> None:
    verifier = _tiny_verifier()
    lora_a = verifier.verifier_layers[0].attn.q_proj.lora_a
    verifier.verifier_layers[0].attn.q_proj.lora_b = nn.Parameter(lora_a.T)
    assert lora_a.untyped_storage()._cdata == (
        verifier.verifier_layers[0].attn.q_proj.lora_b.untyped_storage()._cdata
    )
    with pytest.raises(GVSVerifierError, match="share one storage"):
        audit_verifier_architecture(verifier)


def test_audit_rejects_distinct_trainable_gradients_sharing_storage() -> None:
    verifier = _tiny_verifier()
    parameters = dict(verifier.named_parameters())
    q_parameter = parameters["verifier_layers.0.attn.q_proj.lora_a"]
    v_parameter = parameters["verifier_layers.0.attn.v_proj.lora_a"]
    shared = torch.zeros_like(q_parameter)
    q_parameter.grad = shared
    v_parameter.grad = shared

    with pytest.raises(GVSVerifierError, match="share one storage"):
        audit_verifier_architecture(verifier)


def test_audit_rejects_behavior_shadow_custom_head_and_extra_buffer() -> None:
    verifier = _tiny_verifier()
    verifier._hidden_states = types.MethodType(  # type: ignore[method-assign]
        lambda _self, ids, _mask: torch.zeros((*ids.shape, 16)),
        verifier,
    )
    with pytest.raises(GVSVerifierError, match="shadows class callable"):
        audit_verifier_architecture(verifier)

    class ShiftedHead(nn.Linear):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return super().forward(inputs) + 1

    verifier = _tiny_verifier()
    shifted = ShiftedHead(16, 1, bias=True)
    with torch.no_grad():
        shifted.weight.copy_(verifier.score_head.weight)
        shifted.bias.copy_(verifier.score_head.bias)
    verifier.score_head = shifted
    with pytest.raises(GVSVerifierError, match="must remain Linear"):
        audit_verifier_architecture(verifier)

    verifier = _tiny_verifier()
    verifier.register_buffer("candidate_rank", torch.tensor(0.0))
    with pytest.raises(GVSVerifierError, match="buffer-slot roster changed"):
        audit_verifier_architecture(verifier)


def test_forward_reaudits_mode_vocab_and_finite_score_contracts() -> None:
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    verifier = _tiny_verifier().train()
    verifier.backbone.eval()
    with pytest.raises(GVSVerifierError, match="divergent modes"):
        verifier(ids, mask)

    verifier.train()
    with pytest.raises(GVSVerifierError, match="outside the backbone vocabulary"):
        verifier(torch.tensor([[1, 32]]), mask)
    with torch.no_grad():
        verifier.score_head.bias.fill_(float("inf"))
    with pytest.raises(GVSVerifierError, match="non-finite value"):
        verifier(ids, mask)


def test_loss_and_selection_reject_empty_nonfloat_and_cross_device_inputs() -> None:
    empty_scores = torch.empty((0, 8), dtype=torch.float32)
    empty_mask = torch.empty((0, 8), dtype=torch.bool)
    with pytest.raises(GVSVerifierError, match="at least one row"):
        multi_positive_listwise_loss(
            empty_scores,
            exact_mask=empty_mask,
            valid_mask=empty_mask,
            candidate_identities=(),
        )
    with pytest.raises(GVSVerifierError, match="at least one row"):
        select_valid_candidate(empty_scores, empty_mask, ())

    integer_scores = torch.zeros((1, 8), dtype=torch.int64)
    mask = torch.ones((1, 8), dtype=torch.bool)
    with pytest.raises(GVSVerifierError, match="float32"):
        multi_positive_listwise_loss(
            integer_scores,
            exact_mask=mask,
            valid_mask=mask,
            candidate_identities=_candidate_identities(mask),
        )
    with pytest.raises(GVSVerifierError, match="float32"):
        select_valid_candidate(integer_scores, mask, _candidate_identities(mask))

    scores = torch.zeros((1, 8), dtype=torch.float32)
    meta_mask = mask.to(device="meta")
    with pytest.raises(GVSVerifierError, match="share one device"):
        select_valid_candidate(scores, meta_mask, _candidate_identities(mask))


def test_full_state_and_dtype_wide_mutation_are_explicitly_forbidden() -> None:
    verifier = _tiny_verifier()
    assert verifier.to(device="cpu") is verifier
    with pytest.raises(GVSVerifierError, match="alias-ambiguous"):
        verifier.state_dict()
    with pytest.raises(GVSVerifierError, match="load_state_dict is forbidden"):
        verifier.load_state_dict({})
    with pytest.raises(GVSVerifierError, match="dtype conversion"):
        verifier.to(dtype=torch.float32)
    with pytest.raises(GVSVerifierError, match="dtype conversion"):
        verifier.half()


def test_inference_mode_allows_scoring_but_not_trainable_construction() -> None:
    backbone = BarunLM(_tiny_config())
    with torch.inference_mode(), pytest.raises(GVSVerifierError, match="construction is forbidden"):
        GVSVerifier(backbone, rank=2, alpha=2, initialization_seed=17)
    verifier = _tiny_verifier().eval()
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    with torch.inference_mode():
        assert verifier(ids, torch.ones_like(ids, dtype=torch.bool)).shape == (1,)


def test_adapter_artifact_is_deterministic_adapter_only_and_exactly_loadable() -> None:
    source = _tiny_verifier()
    with torch.no_grad():
        source.verifier_layers[0].attn.q_proj.lora_b.fill_(0.125)
        source.score_head.bias.fill_(-0.25)
    artifact = export_verifier_adapter(source)
    assert export_verifier_adapter(source) == artifact
    raw_tensors = safetensors_load(artifact)
    assert (
        len(raw_tensors)
        == len(tuple(source.named_parameters()))
        - len(tuple(source.backbone.named_parameters()))
        + 1
    )
    assert all("backbone" not in name for name in raw_tensors)

    destination = _tiny_verifier()
    manifest = load_verifier_adapter(destination, artifact)
    assert manifest["launch_authorized"] is False
    assert manifest["authorizes_model_or_label_access"] is False
    assert manifest["backbone_checkpoint_receipt_required"] is True
    source_trainable = {
        name: parameter for name, parameter in source.named_parameters() if parameter.requires_grad
    }
    destination_trainable = {
        name: parameter
        for name, parameter in destination.named_parameters()
        if parameter.requires_grad
    }
    assert source_trainable.keys() == destination_trainable.keys()
    assert all(
        torch.equal(source_trainable[name], destination_trainable[name])
        for name in source_trainable
    )


def test_adapter_load_rejects_wrong_backbone_and_tampered_tensor_bytes() -> None:
    artifact = export_verifier_adapter(_tiny_verifier())
    wrong_backbone = _tiny_verifier()
    with torch.no_grad():
        wrong_backbone.backbone.embedding.weight[0, 0].add_(1)
    with pytest.raises(GVSVerifierError, match="different frozen backbone bytes"):
        load_verifier_adapter(wrong_backbone, artifact)

    tampered = bytearray(artifact)
    tampered[-1] ^= 1
    with pytest.raises(GVSVerifierError, match="adapter manifest|tensor state differs"):
        load_verifier_adapter(_tiny_verifier(), bytes(tampered))


def test_live_class_callable_and_dependency_guard_replacement_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _tiny_verifier()
    monkeypatch.setattr(
        GVSVerifier,
        "_hidden_states",
        lambda self, ids, _mask: self.backbone.embedding(ids),
    )
    with pytest.raises(GVSVerifierError, match="execution callable binding changed"):
        audit_verifier_architecture(verifier)

    monkeypatch.undo()
    verifier = _tiny_verifier()
    original_linear = gvs_verifier_module.F.linear
    monkeypatch.setattr(
        gvs_verifier_module, "_assert_import_time_dependency_bindings", lambda: None
    )
    monkeypatch.setattr(
        gvs_verifier_module.F,
        "linear",
        lambda *args, **kwargs: original_linear(*args, **kwargs) + 1,
    )
    monkeypatch.setattr(gvs_verifier_module, "_F_LINEAR", gvs_verifier_module.F.linear)
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    with pytest.raises(GVSVerifierError, match="execution callable binding changed"):
        verifier(ids, torch.ones_like(ids, dtype=torch.bool))


def test_torch_function_mode_is_rejected_before_verifier_execution() -> None:
    from torch.overrides import TorchFunctionMode

    class PassthroughMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **({} if kwargs is None else kwargs))

    verifier = _tiny_verifier()
    with PassthroughMode(), pytest.raises(GVSVerifierError, match="function/dispatch modes"):
        audit_verifier_architecture(verifier)


def test_receipt_bounds_in_memory_values_and_records_storage_views() -> None:
    verifier = _tiny_verifier()
    receipt = verifier_receipt(verifier)
    topology = receipt["direct_state_topology"]
    assert topology["storage_alias_count"] <= topology["tensor_alias_count"]
    tensor_slot = next(slot for slot in topology["tensor_slots"] if "sha256" in slot)
    assert {"storage_alias_of", "storage_nbytes", "storage_offset", "stride"} <= tensor_slot.keys()

    oversized = copy.deepcopy(receipt)
    oversized["oversized"] = 10**128
    with pytest.raises(GVSVerifierError, match="oversized integer"):
        verify_verifier_receipt(verifier, oversized)


def test_verifier_batch_scoring_is_permutation_equivariant_and_has_no_rank_argument() -> None:
    verifier = _tiny_verifier().eval()
    ids = torch.tensor(
        [[1, value, 0] if value % 2 else [1, value, value + 1] for value in range(2, 10)]
    )
    mask = ids != 0
    permutation = torch.tensor([7, 0, 5, 2, 1, 6, 4, 3])
    scores = verifier(ids, mask)
    assert torch.equal(verifier(ids[permutation], mask[permutation]), scores[permutation])
    parameters = tuple(inspect.signature(GVSVerifier.forward).parameters)
    assert parameters == ("self", "input_ids", "attention_mask")
