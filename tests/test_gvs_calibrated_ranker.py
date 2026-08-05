from __future__ import annotations

import copy
import inspect
from dataclasses import replace
from itertools import permutations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import barunlm.evaluation.gvs_calibrated_ranker as ranker_module
from barunlm.config import BarunConfig
from barunlm.evaluation.gvs_calibrated_ranker import (
    PRODUCTION_LINEAR_PARAMETERS,
    PRODUCTION_LORA_VERIFIER_PARAMETERS,
    PRODUCTION_SMALL_MLP_PARAMETERS,
    CalibratedRankerContract,
    CalibratedRankerKind,
    CalibrationComponentSplit,
    FrozenHiddenCalibratedRanker,
    GVSCalibratedRankerError,
    assert_calibrated_ranker_runtime_integrity,
    assert_production_config,
    audit_calibrated_ranker,
    audit_calibrated_ranker_contract,
    audit_calibration_component_split,
    build_calibrated_ranker_contract,
    build_calibration_component_split,
    calibrated_ranker_runtime_sha256,
    class_balanced_binary_loss,
    select_calibrated_candidate,
)


def _sha(character: str) -> str:
    return character * 64


def test_production_ranker_parameter_counts_are_exact_and_smaller_than_lora() -> None:
    linear = build_calibrated_ranker_contract(CalibratedRankerKind.LINEAR)
    mlp = build_calibrated_ranker_contract(CalibratedRankerKind.SMALL_MLP)
    assert linear.trainable_parameters == PRODUCTION_LINEAR_PARAMETERS == 449
    assert mlp.trainable_parameters == PRODUCTION_SMALL_MLP_PARAMETERS == 29_793
    assert PRODUCTION_LORA_VERIFIER_PARAMETERS // linear.trainable_parameters == 302
    assert PRODUCTION_LORA_VERIFIER_PARAMETERS / mlp.trainable_parameters == pytest.approx(
        4.5519752962
    )
    assert linear.candidate_metadata_visible is False
    assert mlp.authorizes_model_or_label_access is False


def test_contract_factory_is_sealed_type_exact_and_records_are_detached() -> None:
    contract = build_calibrated_ranker_contract(CalibratedRankerKind.LINEAR)
    assert isinstance(contract, CalibratedRankerContract)
    with pytest.raises(TypeError, match="must be constructed"):
        replace(contract)

    for field_name, forged_value in (
        ("trainable_parameters", 449.0),
        ("transformer_blocks", 12.0),
        ("feature_block_index", 10.0),
    ):
        forged = copy.copy(contract)
        object.__setattr__(forged, field_name, forged_value)
        with pytest.raises(GVSCalibratedRankerError, match="stored.*changed"):
            audit_calibrated_ranker_contract(forged)
        with pytest.raises(GVSCalibratedRankerError, match="stored.*changed"):
            FrozenHiddenCalibratedRanker(forged)

    record = contract.as_record()
    assert record["ranker_runtime_sha256"] == calibrated_ranker_runtime_sha256()
    assert record["certifies_feature_provenance"] is False
    record["hidden_dimensions"].append(999)
    assert contract.hidden_dimensions == ()


@pytest.mark.parametrize("kind", list(CalibratedRankerKind))
def test_ranker_initialization_is_deterministic_without_changing_global_rng(kind: str) -> None:
    contract = build_calibrated_ranker_contract(kind)
    torch.manual_seed(991)
    state_before = torch.random.get_rng_state().clone()
    first = FrozenHiddenCalibratedRanker(contract)
    state_after = torch.random.get_rng_state().clone()
    second = FrozenHiddenCalibratedRanker(contract)
    assert torch.equal(state_before, state_after)
    assert all(
        torch.equal(first_value, second_value)
        for first_value, second_value in zip(
            first.state_dict().values(), second.state_dict().values(), strict=True
        )
    )


@pytest.mark.parametrize("kind", list(CalibratedRankerKind))
def test_ranker_forward_and_architecture_audit(kind: str) -> None:
    ranker = FrozenHiddenCalibratedRanker(build_calibrated_ranker_contract(kind))
    features = torch.arange(3 * 448, dtype=torch.float32).reshape(3, 448) / 1000
    scores = ranker(features)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    audit = audit_calibrated_ranker(ranker)
    assert audit["parameter_count"] == ranker.contract.trainable_parameters
    assert audit["authorizes_model_or_label_access"] is False


def test_ranker_detaches_frozen_features_but_trains_its_own_parameters() -> None:
    ranker = FrozenHiddenCalibratedRanker(
        build_calibrated_ranker_contract(CalibratedRankerKind.LINEAR)
    )
    features = torch.randn((2, 448), requires_grad=True)

    ranker(features).sum().backward()

    assert features.grad is None
    assert all(parameter.grad is not None for parameter in ranker.parameters())


def test_ranker_audit_rejects_gradient_storage_aliasing_a_parameter() -> None:
    ranker = FrozenHiddenCalibratedRanker(
        build_calibrated_ranker_contract(CalibratedRankerKind.LINEAR)
    )
    weight = dict(ranker.named_parameters())["network.0.weight"]
    weight.grad = weight.detach()

    with pytest.raises(GVSCalibratedRankerError, match="gradients must own disjoint storage"):
        audit_calibrated_ranker(ranker)


def test_ranker_audit_rejects_post_accumulate_gradient_hooks() -> None:
    ranker = FrozenHiddenCalibratedRanker(
        build_calibrated_ranker_contract(CalibratedRankerKind.LINEAR)
    )
    handle = next(ranker.parameters()).register_post_accumulate_grad_hook(lambda _parameter: None)
    try:
        with pytest.raises(GVSCalibratedRankerError, match="gradient hooks"):
            audit_calibrated_ranker(ranker)
    finally:
        handle.remove()


def test_ranker_rejects_bad_features_and_postconstruction_mutation() -> None:
    ranker = FrozenHiddenCalibratedRanker(
        build_calibrated_ranker_contract(CalibratedRankerKind.LINEAR)
    )
    with pytest.raises(GVSCalibratedRankerError, match="rank-two"):
        ranker(torch.zeros(448))
    with pytest.raises(GVSCalibratedRankerError, match="hidden dimension"):
        ranker(torch.zeros((1, 447)))
    with pytest.raises(GVSCalibratedRankerError, match="supported floating"):
        ranker(torch.zeros((1, 448), dtype=torch.int64))
    features = torch.zeros((1, 448))
    features[0, 0] = float("nan")
    with pytest.raises(GVSCalibratedRankerError, match="non-finite"):
        ranker(features)
    ranker.network.append(nn.ReLU())
    with pytest.raises(GVSCalibratedRankerError, match="topology"):
        ranker(torch.zeros((1, 448)))


def test_architecture_audit_rejects_parameter_alias_and_forged_contract() -> None:
    mlp = FrozenHiddenCalibratedRanker(
        build_calibrated_ranker_contract(CalibratedRankerKind.SMALL_MLP)
    )
    first = mlp.network[0]
    second = mlp.network[2]
    assert isinstance(first, nn.Linear) and isinstance(second, nn.Linear)
    second.bias = nn.Parameter(first.bias[:16])
    with pytest.raises(GVSCalibratedRankerError, match="alias storage"):
        audit_calibrated_ranker(mlp)

    linear = FrozenHiddenCalibratedRanker(
        build_calibrated_ranker_contract(CalibratedRankerKind.LINEAR)
    )
    forged = copy.copy(linear.contract)
    object.__setattr__(forged, "candidate_metadata_visible", True)
    linear.contract = forged
    with pytest.raises(GVSCalibratedRankerError, match="exact frozen contract|stored.*changed"):
        audit_calibrated_ranker(linear)


def test_architecture_audit_rejects_hooks_dtype_nonfinite_and_internal_views() -> None:
    contract = build_calibrated_ranker_contract(CalibratedRankerKind.LINEAR)
    ranker = FrozenHiddenCalibratedRanker(contract)
    hook = ranker.register_forward_hook(lambda _module, _inputs, output: output + 123)
    try:
        with pytest.raises(GVSCalibratedRankerError, match="runtime hooks"):
            ranker(torch.zeros((1, 448)))
    finally:
        hook.remove()

    prehook_handles: list[torch.utils.hooks.RemovableHandle] = []

    def self_removing_prehook(
        _module: nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        prehook_handles[0].remove()
        return (inputs[0] + 1,)

    prehook_handles.append(ranker.register_forward_pre_hook(self_removing_prehook))
    try:
        with pytest.raises(GVSCalibratedRankerError, match="runtime hooks"):
            ranker(torch.zeros((1, 448)))
    finally:
        prehook_handles[0].remove()

    global_hook = torch.nn.modules.module.register_module_forward_pre_hook(
        lambda _module, inputs: inputs
    )
    try:
        with pytest.raises(GVSCalibratedRankerError, match="global PyTorch module hooks"):
            ranker(torch.zeros((1, 448)))
    finally:
        global_hook.remove()

    ranker.generator_likelihood = torch.tensor([0.9])
    with pytest.raises(GVSCalibratedRankerError, match="uncontracted runtime state"):
        audit_calibrated_ranker(ranker)
    del ranker.generator_likelihood

    parameter_hook = ranker.network[0].weight.register_hook(lambda gradient: gradient)
    try:
        with pytest.raises(GVSCalibratedRankerError, match="gradient hooks"):
            audit_calibrated_ranker(ranker)
    finally:
        parameter_hook.remove()

    half_ranker = FrozenHiddenCalibratedRanker(contract).half()
    with pytest.raises(GVSCalibratedRankerError, match="dense float32 storage"):
        audit_calibrated_ranker(half_ranker)

    nonfinite_ranker = FrozenHiddenCalibratedRanker(contract)
    nonfinite_ranker.network[0].weight.data[0, 0] = float("nan")
    with pytest.raises(GVSCalibratedRankerError, match="non-finite"):
        audit_calibrated_ranker(nonfinite_ranker)

    overlapping_ranker = FrozenHiddenCalibratedRanker(contract)
    backing = torch.zeros(1)
    overlapping_ranker.network[0].weight = nn.Parameter(backing.expand(1, 448))
    with pytest.raises(GVSCalibratedRankerError, match="dense float32 storage"):
        audit_calibrated_ranker(overlapping_ranker)

    alias_ranker = FrozenHiddenCalibratedRanker(contract)
    aliased_features = alias_ranker.network[0].weight.detach()
    assert type(aliased_features) is torch.Tensor and aliased_features.shape == (1, 448)
    with pytest.raises(GVSCalibratedRankerError, match="must not alias"):
        alias_ranker(aliased_features)

    overflowing_ranker = FrozenHiddenCalibratedRanker(contract)
    overflowing_ranker.network[0].weight.data.fill_(torch.finfo(torch.float32).max)
    with pytest.raises(GVSCalibratedRankerError, match="produced a non-finite"):
        overflowing_ranker(torch.full((1, 448), torch.finfo(torch.float32).max))


def test_class_balanced_loss_matches_torch_reference() -> None:
    logits = torch.tensor([-2.0, -1.0, 0.0, 2.0])
    labels = torch.tensor([False, False, False, True])
    actual = class_balanced_binary_loss(logits, labels)
    expected = F.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        pos_weight=torch.tensor(3.0),
    )
    assert torch.equal(actual, expected)
    with pytest.raises(GVSCalibratedRankerError, match="both label classes"):
        class_balanced_binary_loss(logits, torch.ones_like(labels))
    with pytest.raises(GVSCalibratedRankerError, match="boolean"):
        class_balanced_binary_loss(logits, labels.float())
    with pytest.raises(GVSCalibratedRankerError, match="float32"):
        class_balanced_binary_loss(logits.double(), labels)
    sparse_logits = torch.sparse_coo_tensor(
        torch.tensor([[0, 1]]),
        torch.tensor([0.0, 1.0]),
        size=(2,),
        check_invariants=True,
    )
    with pytest.raises(GVSCalibratedRankerError, match="dense strided"):
        class_balanced_binary_loss(sparse_logits, torch.tensor([False, True]))
    with pytest.raises(GVSCalibratedRankerError, match="non-finite"):
        class_balanced_binary_loss(
            torch.tensor([0.0, float("inf")]),
            torch.tensor([False, True]),
        )
    differentiable = logits.clone().requires_grad_(True)
    loss = class_balanced_binary_loss(differentiable, labels)
    loss.backward()
    assert differentiable.grad is not None
    assert torch.isfinite(differentiable.grad).all()


def test_component_split_is_deterministic_disjoint_and_component_level() -> None:
    component_ids = tuple(f"component-{index:02d}" for index in range(12))
    first = build_calibration_component_split(
        component_ids,
        population_sha256=_sha("a"),
    )
    second = build_calibration_component_split(
        tuple(reversed(component_ids)),
        population_sha256=_sha("a"),
    )
    assert first == second
    assert len(first.train_component_ids) == 9
    assert len(first.validation_component_ids) == 3
    assert set(first.train_component_ids).isdisjoint(first.validation_component_ids)
    assert set(first.train_component_ids) | set(first.validation_component_ids) == set(
        component_ids
    )
    assert first.authorizes_model_or_label_access is False
    assert audit_calibration_component_split(first)["component_count"] == len(component_ids)
    assert first.as_record()["certifies_feature_provenance"] is False

    different_population = build_calibration_component_split(
        component_ids,
        population_sha256=_sha("b"),
    )
    assert (
        first.train_component_ids,
        first.validation_component_ids,
    ) != (
        different_population.train_component_ids,
        different_population.validation_component_ids,
    )

    with pytest.raises(TypeError, match="must be constructed"):
        replace(first)
    assert isinstance(first, CalibrationComponentSplit)
    forged = copy.copy(first)
    object.__setattr__(forged, "component_membership_sha256", _sha("0"))
    with pytest.raises(GVSCalibratedRankerError, match="stored.*changed"):
        audit_calibration_component_split(forged)

    detached = first.as_record()
    detached["train_component_ids"].append("component-forged")
    assert "component-forged" not in first.train_component_ids


def test_component_split_rejects_duplicates_bad_hashes_and_too_few_groups() -> None:
    with pytest.raises(GVSCalibratedRankerError, match="at least four"):
        build_calibration_component_split(("a", "b", "c"), population_sha256=_sha("a"))
    with pytest.raises(GVSCalibratedRankerError, match="unique"):
        build_calibration_component_split(("a", "b", "c", "a"), population_sha256=_sha("a"))
    with pytest.raises(GVSCalibratedRankerError, match="lowercase SHA-256"):
        build_calibration_component_split(("a", "b", "c", "d"), population_sha256="bad")
    for invalid_identifier in ("a b", "a\tb", "a\u202eb", "e\u0301"):
        with pytest.raises(GVSCalibratedRankerError, match="whitespace|controls|NFC"):
            build_calibration_component_split(
                (invalid_identifier, "b", "c", "d"),
                population_sha256=_sha("a"),
            )
    with pytest.raises(GVSCalibratedRankerError, match="UTF-8 bytes"):
        build_calibration_component_split(
            ("\U0001f600" * 65, "b", "c", "d"),
            population_sha256=_sha("a"),
        )


def test_candidate_selection_is_permutation_invariant_and_hash_tiebroken() -> None:
    scores = (0.1, 0.9, 0.9)
    identities = (_sha("c"), _sha("b"), _sha("a"))
    selected = select_calibrated_candidate(scores, identities)
    assert identities[selected] == _sha("a")
    for permutation in permutations(range(len(scores))):
        permuted_scores = tuple(scores[index] for index in permutation)
        permuted_ids = tuple(identities[index] for index in permutation)
        selected_permuted = select_calibrated_candidate(permuted_scores, permuted_ids)
        assert permuted_ids[selected_permuted] == _sha("a")
    with pytest.raises(GVSCalibratedRankerError, match="deduplicated"):
        select_calibrated_candidate((0.1, 0.2), (_sha("a"), _sha("a")))
    with pytest.raises(GVSCalibratedRankerError, match="finite"):
        select_calibrated_candidate((float("nan"),), (_sha("a"),))


def test_ranker_api_has_no_rank_likelihood_or_feature_provenance_claim() -> None:
    forward_parameters = tuple(inspect.signature(FrozenHiddenCalibratedRanker.forward).parameters)
    selection_parameters = tuple(inspect.signature(select_calibrated_candidate).parameters)
    assert forward_parameters == ("self", "features")
    assert selection_parameters == ("scores", "candidate_sha256s")
    ranker = FrozenHiddenCalibratedRanker(
        build_calibrated_ranker_contract(CalibratedRankerKind.LINEAR)
    )
    audit = audit_calibrated_ranker(ranker)
    assert audit["authorizes_model_or_label_access"] is False
    assert audit["certifies_feature_provenance"] is False


def test_production_config_assertion_is_exact() -> None:
    assert_production_config(BarunConfig())
    changed = BarunConfig(dim=456, n_heads=8)
    with pytest.raises(GVSCalibratedRankerError, match="canonical"):
        assert_production_config(changed)
    type_confused = BarunConfig()
    type_confused.dim = 448.0  # type: ignore[assignment]
    with pytest.raises(GVSCalibratedRankerError, match="canonical"):
        assert_production_config(type_confused)


def test_runtime_identity_detects_live_code_and_loss_dependency_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_runtime = calibrated_ranker_runtime_sha256()
    ranker = FrozenHiddenCalibratedRanker(
        build_calibrated_ranker_contract(CalibratedRankerKind.LINEAR)
    )
    with monkeypatch.context() as patcher:
        patcher.setattr(ranker_module, "audit_calibrated_ranker", lambda _ranker: {})
        assert calibrated_ranker_runtime_sha256() != baseline_runtime
        with pytest.raises(GVSCalibratedRankerError, match="runtime differs"):
            ranker(torch.zeros((1, 448)))

    with monkeypatch.context() as patcher:
        patcher.setattr(
            ranker_module.F,
            "binary_cross_entropy_with_logits",
            lambda *_args, **_kwargs: torch.tensor(0.0),
        )
        assert calibrated_ranker_runtime_sha256() != baseline_runtime
        with pytest.raises(GVSCalibratedRankerError, match="runtime differs"):
            class_balanced_binary_loss(
                torch.tensor([0.0, 1.0]),
                torch.tensor([False, True]),
            )

    assert calibrated_ranker_runtime_sha256() == baseline_runtime
    assert_calibrated_ranker_runtime_integrity()
