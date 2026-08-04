from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import Tensor, nn

from barunlm.config import BarunConfig
from barunlm.datasets.mobile_actions import render_prompt
from barunlm.evaluation import gvs_decoder
from barunlm.evaluation.action_ir import ToolSchema
from barunlm.evaluation.gvs_decoder import (
    FROZEN_CANDIDATE_COUNT,
    CandidateAnalysis,
    GVSDecoderContract,
    GVSDecoderError,
    LikelihoodTrace,
    analyze_gvs_trace,
    generate_gvs_trace,
    inspect_gvs_runtime,
    parse_candidate_set_analysis_json,
    parse_candidate_set_trace_json,
    parse_likelihood_selection_json,
    recompute_likelihood,
    select_schema_valid_by_likelihood,
)
from barunlm.model import BarunLM

NOW = "2026-08-04T12:00:00"
USER_TEXT = "do the thing"

# These are immutable algorithm vectors, deliberately excluding runtime/source identities.
CONTRACT_GOLDEN_SHA256 = "161607dbac01a9493108ad35aaaa01b09f4bc7f29296c8ed404afe9df08fc3f7"
MULTISTEP_TIE_GOLDEN_SHA256 = "ac4758f53a9f1c088b332cc7cb5c5ab658802baab83f565041e343273bc06d79"
ALL_PAD_GOLDEN_SHA256 = "9026854a299f5efa13383bba3d3571469b4df7ef704ef5c615369a0932632190"


class _ToyTokenizer:
    def __init__(self) -> None:
        self.text = {
            0: "<pad>",
            1: "<eos>",
            2: "PROMPT",
            3: '{"decision":"ABSTAIN"}',
            4: "not-json",
            5: '{"decision":"BOGUS"}',
            6: "x",
            7: "y",
            8: "z",
        }

    def encode(self, sequence: str, add_special_tokens: bool = False) -> SimpleNamespace:
        assert sequence
        assert add_special_tokens is False
        return SimpleNamespace(ids=[2])

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        return "".join(self.text[token_id] for token_id in ids)

    def token_to_id(self, token: str) -> int | None:
        return {"<pad>": 0, "<eos>": 1}.get(token)

    def to_str(self) -> str:
        return json.dumps(self.text, sort_keys=True, separators=(",", ":"))


class _DecodeFailureTokenizer(_ToyTokenizer):
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        if 4 in ids:
            raise ValueError("synthetic decode failure")
        return super().decode(ids, skip_special_tokens=skip_special_tokens)

    def to_str(self) -> str:
        return '{"fixture":"decode-failure-v1"}'


class _LongDecodeTokenizer(_ToyTokenizer):
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del ids
        assert skip_special_tokens is False
        return "x" * 300_000

    def to_str(self) -> str:
        return '{"fixture":"long-decode-v1"}'


class _TransitionModel(nn.Module):
    def __init__(self, transitions: Tensor, *, max_seq_len: int = 32) -> None:
        super().__init__()
        assert transitions.ndim == 2 and transitions.shape[0] == transitions.shape[1]
        self.register_buffer("transitions", transitions.to(dtype=torch.float64))
        self.config = SimpleNamespace(vocab_size=transitions.shape[0], max_seq_len=max_seq_len)

    def forward(self, input_ids: Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.transitions[input_ids])

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        attention_mask: Tensor | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> Tensor:
        del attention_mask, pad_token_id
        assert temperature == 0
        output = input_ids
        for _ in range(max_new_tokens):
            next_token = self(output).logits[:, -1].argmax(dim=-1, keepdim=True)
            output = torch.cat((output, next_token), dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return output


class _FloatGenerateModel(_TransitionModel):
    def generate(self, *args: object, **kwargs: object) -> Tensor:
        return super().generate(*args, **kwargs).to(dtype=torch.float64)


class _Int32GenerateModel(_TransitionModel):
    def generate(self, *args: object, **kwargs: object) -> Tensor:
        return super().generate(*args, **kwargs).to(dtype=torch.int32)


class _EarlyShortModel(_TransitionModel):
    def generate(self, input_ids: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        token = torch.tensor([[3]], dtype=torch.long)
        return torch.cat((input_ids, token), dim=1)


class _TokensAfterEOSModel(_TransitionModel):
    def generate(self, input_ids: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        tokens = torch.tensor([[1, 3]], dtype=torch.long)
        return torch.cat((input_ids, tokens), dim=1)


class _StateMutatingModel(_TransitionModel):
    def generate(self, *args: object, **kwargs: object) -> Tensor:
        with torch.no_grad():
            self.transitions[2, 8] += 0.25
        return super().generate(*args, **kwargs)


class _SelectiveLikelihoodFailureModel(_TransitionModel):
    def forward(self, input_ids: Tensor) -> SimpleNamespace:
        if input_ids.shape[1] == 3 and int(input_ids[0, -1]) == 3:
            raise RuntimeError("one synthetic final likelihood failure")
        return super().forward(input_ids)


class _ModeMutatingModel(_TransitionModel):
    def forward(self, input_ids: Tensor) -> SimpleNamespace:
        self.training = True
        return super().forward(input_ids)


class _DefaultDeviceMutatingModel(_TransitionModel):
    def forward(self, input_ids: Tensor) -> SimpleNamespace:
        torch.set_default_device("meta")
        return super().forward(input_ids)


class _IntegerLogitModel(_TransitionModel):
    def forward(self, input_ids: Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.transitions[input_ids].to(dtype=torch.int64))


class _MutatingTokenizer(_ToyTokenizer):
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        self.text[8] += "!"
        return super().decode(ids, skip_special_tokens=skip_special_tokens)


class _HiddenStateTokenizer(_ToyTokenizer):
    def __init__(self, prompt_token_id: int) -> None:
        super().__init__()
        self.prompt_token_id = prompt_token_id

    def encode(self, sequence: str, add_special_tokens: bool = False) -> SimpleNamespace:
        assert sequence and add_special_tokens is False
        return SimpleNamespace(ids=[self.prompt_token_id])


class _HiddenBehaviorModel(_TransitionModel):
    def __init__(self, transitions: Tensor, *, preferred_token_id: int) -> None:
        super().__init__(transitions)
        self.preferred_token_id = preferred_token_id

    def forward(self, input_ids: Tensor) -> SimpleNamespace:
        logits = self.transitions[input_ids].clone()
        logits[..., self.preferred_token_id] += 100.0
        return SimpleNamespace(logits=logits)


def _alternate_transition_forward(model: _TransitionModel, input_ids: Tensor) -> SimpleNamespace:
    logits = model.transitions[input_ids].clone()
    logits[..., 8] += 100.0
    return SimpleNamespace(logits=logits)


def _transition_model() -> _TransitionModel:
    logits = torch.full((9, 9), -8.0, dtype=torch.float64)
    logits[2] = torch.tensor([-9.0, 4.0, 1.0, 7.0, 8.0, 6.0, 3.0, 2.0, 0.0])
    for token_id in range(9):
        logits[token_id, 1] = 9.0
    logits[2, 4] = 10.0
    logits[2, 3] = 9.5
    logits[2, 1] = 5.0
    return _TransitionModel(logits)


def _uniform_tie_model() -> _TransitionModel:
    logits = torch.zeros((9, 9), dtype=torch.float64)
    logits[:, 0] = -10.0
    return _TransitionModel(logits)


def _all_pad_model() -> _TransitionModel:
    logits = torch.full((9, 9), -20.0, dtype=torch.float64)
    logits[:, 0] = 20.0
    logits[:, 1] = 10.0
    return _TransitionModel(logits)


def _exact_tokenizer() -> Tokenizer:
    return Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<eos>": 1,
                "<unk>": 2,
                "a": 3,
                "b": 4,
                "c": 5,
                "d": 6,
                "e": 7,
                "f": 8,
            },
            unk_token="<unk>",
        )
    )


def _exact_barun_model(*, seed: int = 18, max_seq_len: int = 8) -> BarunLM:
    torch.manual_seed(seed)
    return BarunLM(
        BarunConfig(
            vocab_size=9,
            dim=16,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            ffn_dim=32,
            max_seq_len=max_seq_len,
            local_window=max_seq_len,
            full_attention_every=1,
            attention_gate=False,
            residual_select_every=0,
            tie_embeddings=False,
        )
    ).eval()


def _trace(
    model: nn.Module,
    *,
    tokenizer: _ToyTokenizer | None = None,
    max_new_tokens: int = 2,
):
    tokenizer = _ToyTokenizer() if tokenizer is None else tokenizer
    runtime = inspect_gvs_runtime(model, tokenizer)
    prompt = render_prompt(now=NOW, tools=(), user_text=USER_TEXT)
    return generate_gvs_trace(
        model,
        tokenizer,
        now=NOW,
        tools=(),
        user_text=USER_TEXT,
        model_sha256=runtime.model_state_sha256,
        tokenizer_sha256=runtime.tokenizer_state_sha256,
        source_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        expected_runtime_sha256=runtime.sha256,
        contract=GVSDecoderContract(max_new_tokens=max_new_tokens),
    )


def test_contract_is_non_authorizing_complete_immutable_and_golden() -> None:
    contract = GVSDecoderContract()

    assert contract.status == "reference_only_proposal"
    assert contract.launch_authorized is False
    assert contract.optimized_remote_exact_match_required is True
    assert contract.include_generated_pad_in_score is True
    assert contract.filler_mask_semantics == "explicit_mask_only_token_id_never_implies_filler"
    assert contract.min_content_tokens == 0
    assert contract.max_new_tokens == 192
    assert contract.sha256 == CONTRACT_GOLDEN_SHA256
    json.dumps(contract.to_record(), allow_nan=False)

    with pytest.raises(FrozenInstanceError):
        contract.max_new_tokens = 1  # type: ignore[misc]
    with pytest.raises(GVSDecoderError, match="candidate_count is frozen"):
        GVSDecoderContract(candidate_count=7)


def test_multistep_parent_ties_finished_beams_and_normalization_are_golden() -> None:
    model = _uniform_tie_model()
    trace = _trace(model, max_new_tokens=2)

    assert trace.runtime.reference_impl_only is True
    assert trace.runtime.optimized_remote_exact_match_required is True
    assert trace.model_sha256 == trace.runtime.model_state_sha256
    assert trace.tokenizer_sha256 == trace.runtime.tokenizer_state_sha256
    assert trace.source_sha256 == trace.prompt_sha256
    assert [candidate.generated_token_ids for candidate in trace.candidates] == [
        (1,),
        (1,),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
    ]
    assert [candidate.beam_rank for candidate in trace.candidates[1:]] == list(range(7))
    assert trace.candidates[1].eos_emitted
    assert all(candidate.truncated for candidate in trace.candidates[3:])
    assert trace.candidates[1].likelihood is not None
    assert trace.candidates[2].likelihood is not None
    assert (
        trace.candidates[1].likelihood.normalized_log_likelihood
        == trace.candidates[2].likelihood.normalized_log_likelihood
    )
    assert trace.golden_vector_sha256 == MULTISTEP_TIE_GOLDEN_SHA256


def test_generated_pad_and_all_pad_are_scored_and_retained_golden() -> None:
    trace = _trace(_all_pad_model(), max_new_tokens=2)

    assert len(trace.candidates) == FROZEN_CANDIDATE_COUNT
    assert trace.candidates[0].generated_token_ids == (0, 0)
    assert trace.candidates[0].raw_output == "<pad><pad>"
    assert trace.candidates[0].truncated
    assert trace.candidates[0].likelihood is not None
    assert trace.candidates[0].likelihood.filler_mask == (False, False)
    assert trace.candidates[0].likelihood.score_included == (True, True)
    assert trace.candidates[0].likelihood.scored_token_count == 2
    assert trace.golden_vector_sha256 == ALL_PAD_GOLDEN_SHA256


def test_only_explicit_filler_mask_excludes_a_token_never_pad_identity() -> None:
    model = _uniform_tie_model()
    ordinary = recompute_likelihood(
        model,
        prompt_token_ids=(2,),
        generated_token_ids=(0, 1),
    )
    explicit_filler = recompute_likelihood(
        model,
        prompt_token_ids=(2,),
        generated_token_ids=(0, 1),
        filler_mask=(True, False),
    )

    assert ordinary.score_included == (True, True)
    assert ordinary.scored_token_count == 2
    assert explicit_filler.score_included == (False, True)
    assert explicit_filler.scored_token_count == 1


def test_likelihood_aggregates_require_exact_finite_float_recomputation() -> None:
    with pytest.raises(GVSDecoderError, match="exact finite non-positive float"):
        LikelihoodTrace(
            generated_token_ids=(1,),
            token_log_likelihoods=(0.0,),
            filler_mask=(False,),
            score_included=(True,),
            raw_log_likelihood=0,  # type: ignore[arg-type]
            normalized_log_likelihood=0.0,
            scored_token_count=1,
        )

    with pytest.raises(GVSDecoderError, match="overflowed"):
        LikelihoodTrace(
            generated_token_ids=(1, 2),
            token_log_likelihoods=(-1e308, -1e308),
            filler_mask=(False, False),
            score_included=(True, True),
            raw_log_likelihood=-1.0,
            normalized_log_likelihood=-0.5,
            scored_token_count=2,
        )


def test_runtime_identity_binds_actual_dtype_and_rejects_mismatch() -> None:
    torch.manual_seed(8)
    config = BarunConfig(
        vocab_size=9,
        dim=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim=32,
        max_seq_len=8,
        local_window=8,
        full_attention_every=1,
        attention_gate=False,
        residual_select_every=0,
        tie_embeddings=False,
    )
    float_model = BarunLM(config).eval()
    bf16_model = copy.deepcopy(float_model).to(dtype=torch.bfloat16).eval()
    tokenizer = _ToyTokenizer()
    expected = inspect_gvs_runtime(float_model, tokenizer)

    with pytest.raises(GVSDecoderError, match="runtime disagrees"):
        generate_gvs_trace(
            bf16_model,
            tokenizer,
            now=NOW,
            tools=(),
            user_text=USER_TEXT,
            model_sha256=expected.model_state_sha256,
            tokenizer_sha256=expected.tokenizer_state_sha256,
            source_sha256=hashlib.sha256(
                render_prompt(now=NOW, tools=(), user_text=USER_TEXT).encode("utf-8")
            ).hexdigest(),
            expected_runtime_sha256=expected.sha256,
            contract=GVSDecoderContract(max_new_tokens=2),
        )


def test_runtime_requires_and_explicitly_allocates_cpu_under_changed_default_device() -> None:
    previous = str(torch.get_default_device())
    assert previous == "cpu"
    model = _uniform_tie_model()
    tokenizer = _ToyTokenizer()
    torch.set_default_device("meta")
    try:
        tensor = gvs_decoder._cpu_long_tensor([[2]], label="test tensor")
        assert tensor.device.type == "cpu"
        with pytest.raises(GVSDecoderError, match="default device cpu"):
            inspect_gvs_runtime(model, tokenizer)
    finally:
        # Restoring with ``"cpu"`` leaves a DeviceContext TorchFunctionMode on
        # PyTorch's process-global mode stack.  That leaks into later tests even
        # though ``get_default_device()`` still prints ``cpu``.  ``None`` restores
        # the native CPU default and removes the mode installed above.
        torch.set_default_device(None)
    assert str(torch.get_default_device()) == previous


def test_runtime_binds_hidden_instance_state_and_live_callable_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions = _uniform_tie_model().transitions
    first_model = _HiddenBehaviorModel(transitions, preferred_token_id=3)
    second_model = _HiddenBehaviorModel(transitions, preferred_token_id=8)
    first_runtime = inspect_gvs_runtime(first_model, _ToyTokenizer())
    second_runtime = inspect_gvs_runtime(second_model, _ToyTokenizer())

    assert first_runtime.model_state_sha256 == second_runtime.model_state_sha256
    assert first_runtime.model_instance_state_sha256 != second_runtime.model_instance_state_sha256
    assert first_runtime.sha256 != second_runtime.sha256

    ordinary = _uniform_tie_model()
    before_patch = inspect_gvs_runtime(ordinary, _ToyTokenizer())
    monkeypatch.setattr(_TransitionModel, "forward", _alternate_transition_forward)
    after_patch = inspect_gvs_runtime(ordinary, _ToyTokenizer())
    assert before_patch.model_class_source_sha256 == after_patch.model_class_source_sha256
    assert before_patch.model_callable_sha256 != after_patch.model_callable_sha256
    assert before_patch.sha256 != after_patch.sha256


def test_runtime_binds_test_double_tokenizer_state_and_restricts_production_contract() -> None:
    model = _transition_model()
    first = inspect_gvs_runtime(model, _HiddenStateTokenizer(2))
    second = inspect_gvs_runtime(model, _HiddenStateTokenizer(3))

    assert first.tokenizer_state_sha256 == second.tokenizer_state_sha256
    assert first.tokenizer_instance_state_sha256 != second.tokenizer_instance_state_sha256
    assert first.sha256 != second.sha256
    assert first.tokenizer_contract == "reference_test_double"
    assert first.production_identity_contract_satisfied is False

    torch.manual_seed(18)
    exact_model = BarunLM(
        BarunConfig(
            vocab_size=9,
            dim=16,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            ffn_dim=32,
            max_seq_len=8,
            local_window=8,
            full_attention_every=1,
            attention_gate=False,
            residual_select_every=0,
            tie_embeddings=False,
        )
    ).eval()
    exact_tokenizer = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<eos>": 1,
                "<unk>": 2,
                "a": 3,
                "b": 4,
                "c": 5,
                "d": 6,
                "e": 7,
                "f": 8,
            },
            unk_token="<unk>",
        )
    )
    exact = inspect_gvs_runtime(exact_model, exact_tokenizer)
    assert exact.model_contract == "exact_barunlm_v1"
    assert exact.tokenizer_contract == "exact_tokenizers_reconstructible_v1"
    assert len(exact.tokenizers_implementation_sha256) == 64
    assert exact.adapter_inventory == ()
    assert exact.adapters_enabled is False
    assert exact.production_identity_contract_satisfied is True


def test_exact_tokenizer_rejects_nonserialized_encode_special_tokens_behavior() -> None:
    source = _exact_tokenizer()
    source.pre_tokenizer = Whitespace()
    source.add_special_tokens(["<pad>", "<eos>"])
    ordinary = Tokenizer.from_str(source.to_str())
    contaminated = Tokenizer.from_str(source.to_str())
    contaminated.encode_special_tokens = True

    assert ordinary.to_str() == contaminated.to_str()
    assert (
        ordinary.encode("x<eos>y", add_special_tokens=False).ids
        != contaminated.encode("x<eos>y", add_special_tokens=False).ids
    )
    ordinary_runtime = inspect_gvs_runtime(_exact_barun_model(), ordinary)
    assert ordinary_runtime.production_identity_contract_satisfied is True
    with pytest.raises(GVSDecoderError, match="encode_special_tokens"):
        inspect_gvs_runtime(_exact_barun_model(), contaminated)


def test_runtime_binds_nonpersistent_rotary_buffers_not_only_state_dict() -> None:
    model = _exact_barun_model(seed=72)
    tokenizer = _exact_tokenizer()
    runtime_before = inspect_gvs_runtime(model, tokenizer)
    logits_before = model(torch.tensor([[2, 3, 4, 5]], dtype=torch.long)).logits.detach().clone()

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, gvs_decoder.PartialRotaryEmbedding):
                module.cos.zero_()
                module.sin.fill_(1.0)

    runtime_after = inspect_gvs_runtime(model, tokenizer)
    logits_after = model(torch.tensor([[2, 3, 4, 5]], dtype=torch.long)).logits.detach()
    assert runtime_before.model_state_sha256 == runtime_after.model_state_sha256
    assert runtime_before.model_module_graph_sha256 == runtime_after.model_module_graph_sha256
    assert (
        runtime_before.model_execution_tensor_sha256 != runtime_after.model_execution_tensor_sha256
    )
    assert runtime_before.sha256 != runtime_after.sha256
    assert not torch.equal(logits_before, logits_after)


def test_exact_generation_normalizes_caches_and_selector_observation_state() -> None:
    torch.manual_seed(29)
    model = BarunLM(
        BarunConfig(
            vocab_size=9,
            dim=16,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            ffn_dim=32,
            max_seq_len=8,
            local_window=4,
            full_attention_every=2,
            attention_gate=True,
            residual_select_every=1,
            tie_embeddings=True,
        )
    ).eval()
    dirty_mask = gvs_decoder.barun_model_module._local_causal_mask(1, 4, "cpu")
    dirty_mask.fill_(17.0)
    for selector in model.selectors:
        selector.last_mean_weights = torch.ones(2)

    trace = _trace(model, tokenizer=_exact_tokenizer(), max_new_tokens=2)
    assert trace.runtime.production_identity_contract_satisfied is True
    assert all(selector.last_mean_weights is None for selector in model.selectors)
    assert gvs_decoder.barun_model_module._local_causal_mask.cache_info().currsize == 0


def test_runtime_rejects_ambient_cpu_autocast() -> None:
    model = _exact_barun_model(seed=213)
    tokenizer = _exact_tokenizer()
    inspect_gvs_runtime(model, tokenizer)

    with (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16),
        pytest.raises(GVSDecoderError, match="ambient autocast"),
    ):
        inspect_gvs_runtime(model, tokenizer)


def test_runtime_binds_model_call_and_loaded_decoder_callables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _exact_barun_model(seed=31)
    tokenizer = _exact_tokenizer()
    ordinary = inspect_gvs_runtime(model, tokenizer)
    original_call = BarunLM.__call__

    def changed_call(self: BarunLM, *args: object, **kwargs: object):
        output = original_call(self, *args, **kwargs)
        output.logits = torch.roll(output.logits, shifts=1, dims=-1)
        return output

    with monkeypatch.context() as patch:
        patch.setattr(BarunLM, "__call__", changed_call)
        changed_model_runtime = inspect_gvs_runtime(model, tokenizer)
        assert ordinary.model_module_graph_sha256 != changed_model_runtime.model_module_graph_sha256
        assert ordinary.sha256 != changed_model_runtime.sha256

    with monkeypatch.context() as patch:
        patch.setattr(gvs_decoder, "_native_greedy_tokens", lambda *args, **kwargs: (3, 1))
        changed_decoder_runtime = inspect_gvs_runtime(model, tokenizer)
        assert ordinary.decoder_source_sha256 == changed_decoder_runtime.decoder_source_sha256
        assert ordinary.decoder_callable_sha256 != changed_decoder_runtime.decoder_callable_sha256
        assert ordinary.sha256 != changed_decoder_runtime.sha256


def test_runtime_binds_algorithm_critical_global_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _transition_model()
    tokenizer = _ToyTokenizer()
    ordinary = inspect_gvs_runtime(model, tokenizer)
    prompt = render_prompt(now=NOW, tools=(), user_text=USER_TEXT)

    def zero_log_softmax(values: Tensor, dim: int) -> Tensor:
        del dim
        return torch.zeros_like(values)

    monkeypatch.setattr(torch, "log_softmax", zero_log_softmax)
    changed = inspect_gvs_runtime(model, tokenizer)

    assert ordinary.decoder_source_sha256 == changed.decoder_source_sha256
    assert ordinary.decoder_callable_sha256 != changed.decoder_callable_sha256
    assert ordinary.sha256 != changed.sha256
    with pytest.raises(GVSDecoderError, match="runtime disagrees"):
        generate_gvs_trace(
            model,
            tokenizer,
            now=NOW,
            tools=(),
            user_text=USER_TEXT,
            model_sha256=ordinary.model_state_sha256,
            tokenizer_sha256=ordinary.tokenizer_state_sha256,
            source_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            expected_runtime_sha256=ordinary.sha256,
            contract=GVSDecoderContract(max_new_tokens=2),
        )


def test_runtime_binds_tensor_methods_used_by_exact_barun_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _exact_barun_model(seed=98)
    tokenizer = _exact_tokenizer()
    ordinary = inspect_gvs_runtime(model, tokenizer)
    original_argmax = Tensor.argmax

    def zero_argmax(values: Tensor, *args: object, **kwargs: object) -> Tensor:
        return original_argmax(values, *args, **kwargs).remainder(1)

    monkeypatch.setattr(Tensor, "argmax", zero_argmax)
    changed = inspect_gvs_runtime(model, tokenizer)
    assert ordinary.model_dependency_sha256 != changed.model_dependency_sha256
    assert ordinary.sha256 != changed.sha256


def test_runtime_binds_exact_model_functional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _exact_barun_model(seed=97)
    tokenizer = _exact_tokenizer()
    ordinary = inspect_gvs_runtime(model, tokenizer)
    original_linear = torch.nn.functional.linear

    def shifted_linear(inputs: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
        return original_linear(inputs, weight, bias) + 0.125

    monkeypatch.setattr(torch.nn.functional, "linear", shifted_linear)
    changed = inspect_gvs_runtime(model, tokenizer)

    assert ordinary.model_state_sha256 == changed.model_state_sha256
    assert ordinary.model_dependency_sha256 != changed.model_dependency_sha256
    assert ordinary.sha256 != changed.sha256


def test_exact_barunlm_topology_rejects_marker_free_state_preserving_wrapper() -> None:
    class FixedShiftLinear(nn.Linear):
        def forward(self, inputs: Tensor) -> Tensor:
            return super().forward(inputs) + 0.25

    model = _exact_barun_model(seed=17)
    state_sha256 = gvs_decoder._model_state_sha256(model)
    original = model.layers[0].attn.q_proj
    replacement = FixedShiftLinear(original.in_features, original.out_features, bias=False)
    replacement.weight = original.weight
    model.layers[0].attn.q_proj = replacement

    assert gvs_decoder._model_state_sha256(model) == state_sha256
    assert gvs_decoder._adapter_inventory(model) == ()
    with pytest.raises(GVSDecoderError, match="must be Linear"):
        inspect_gvs_runtime(model, _exact_tokenizer())


def test_exact_barunlm_topology_rejects_marker_free_registry_shape_and_behavior_changes() -> None:
    tokenizer = _exact_tokenizer()

    extra_buffer = _exact_barun_model(seed=81)
    extra_buffer.register_buffer("marker_free_state", torch.zeros(1))
    with pytest.raises(GVSDecoderError, match="registry roster changed"):
        inspect_gvs_runtime(extra_buffer, tokenizer)

    changed_dimension = _exact_barun_model(seed=82)
    changed_dimension.layers[0].attn.q_proj.out_features -= 1
    with pytest.raises(GVSDecoderError, match="dimensions changed"):
        inspect_gvs_runtime(changed_dimension, tokenizer)

    changed_norm = _exact_barun_model(seed=83)
    changed_norm.final_norm.eps = 0.5
    with pytest.raises(GVSDecoderError, match="behavior changed"):
        inspect_gvs_runtime(changed_norm, tokenizer)

    aliased_parameter = _exact_barun_model(seed=84)
    aliased_parameter.layers[1].attn.q_proj.weight = aliased_parameter.layers[0].attn.q_proj.weight
    with pytest.raises(GVSDecoderError, match="alias topology changed"):
        inspect_gvs_runtime(aliased_parameter, tokenizer)

    shared_storage = _exact_barun_model(seed=85)
    source = shared_storage.layers[0].attn.q_proj.weight
    shared_storage.layers[1].attn.q_proj.weight = nn.Parameter(source.view_as(source))
    with pytest.raises(GVSDecoderError, match="distinct tensors sharing one storage"):
        inspect_gvs_runtime(shared_storage, tokenizer)


def test_eval_and_inference_contexts_restore_mixed_modes_and_reject_mode_mutation() -> None:
    model = _transition_model().train()
    model.observer = nn.Identity().eval()
    modes_before = tuple(module.training for module in model.modules())
    default_device_before = str(torch.get_default_device())
    inference_before = torch.is_inference_mode_enabled()

    likelihood = recompute_likelihood(
        model,
        prompt_token_ids=(2,),
        generated_token_ids=(3, 1),
    )
    assert likelihood.scored_token_count == 2
    assert tuple(module.training for module in model.modules()) == modes_before

    trace = _trace(model, max_new_tokens=2)
    assert len(trace.candidates) == 8
    assert tuple(module.training for module in model.modules()) == modes_before
    assert str(torch.get_default_device()) == default_device_before
    assert torch.is_inference_mode_enabled() is inference_before

    mutating = _ModeMutatingModel(_transition_model().transitions).eval()
    with pytest.raises(GVSDecoderError, match="mutated its module training modes"):
        _trace(mutating, max_new_tokens=2)
    assert mutating.training is False
    assert torch.is_inference_mode_enabled() is inference_before


def test_model_global_torch_state_mutation_fails_and_restores_native_cpu_default() -> None:
    model = _DefaultDeviceMutatingModel(_transition_model().transitions).eval()
    with pytest.raises(GVSDecoderError, match="process-global Torch execution state"):
        recompute_likelihood(
            model,
            prompt_token_ids=(2,),
            generated_token_ids=(3, 1),
        )
    assert str(torch.get_default_device()) == "cpu"
    assert gvs_decoder._torch_execution_state_record()["function_modes"] == []


def test_decoder_source_has_no_optimization_sensitive_assert_statements() -> None:
    tree = ast.parse(inspect.getsource(gvs_decoder))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def test_runtime_rejects_adapters_hooks_and_implicit_generation_configuration() -> None:
    adapter_model = _uniform_tie_model()
    adapter_model.active_adapter = "unexpected"
    with pytest.raises(GVSDecoderError, match="adapter inventory must be empty"):
        inspect_gvs_runtime(adapter_model, _ToyTokenizer())

    configured_model = _uniform_tie_model()
    configured_model.generation_config = {"do_sample": True}
    with pytest.raises(GVSDecoderError, match="implicit model generation_config"):
        inspect_gvs_runtime(configured_model, _ToyTokenizer())

    hooked_model = _uniform_tie_model()
    handle = hooked_model.register_forward_hook(lambda _module, _args, output: output)
    try:
        with pytest.raises(GVSDecoderError, match="forbidden execution hook"):
            inspect_gvs_runtime(hooked_model, _ToyTokenizer())
    finally:
        handle.remove()


def test_rank_zero_is_native_barun_generate_token_and_byte_identical() -> None:
    torch.manual_seed(23)
    config = BarunConfig(
        vocab_size=9,
        dim=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim=32,
        max_seq_len=8,
        local_window=8,
        full_attention_every=1,
        attention_gate=False,
        residual_select_every=0,
    )
    model = BarunLM(config).eval()
    expected_ids = model.generate(
        torch.tensor([[2]], dtype=torch.long),
        max_new_tokens=2,
        temperature=0,
        eos_token_id=1,
        pad_token_id=0,
    )[0, 1:]
    tokenizer = _ToyTokenizer()
    expected_raw = tokenizer.decode(
        [token for token in expected_ids.tolist() if token != 1],
        skip_special_tokens=False,
    )

    trace = _trace(model, tokenizer=tokenizer, max_new_tokens=2)

    assert trace.candidates[0].generated_token_ids == tuple(expected_ids.tolist())
    assert trace.candidates[0].raw_output == expected_raw


def test_action_analysis_binds_schema_and_likelihood_selection() -> None:
    trace = _trace(_transition_model(), max_new_tokens=2)
    empty = analyze_gvs_trace(trace, ())
    changed = analyze_gvs_trace(trace, (ToolSchema(name="unused", arguments={}),))

    assert empty.schema_sha256 != changed.schema_sha256
    assert empty.sha256 != changed.sha256
    assert len(empty.candidates) == 8
    assert not empty.candidates[0].parse_valid
    valid = [candidate for candidate in empty.candidates if candidate.schema_valid]
    assert valid
    assert all(candidate.canonical_action_json == '{"decision":"ABSTAIN"}' for candidate in valid)
    selection = select_schema_valid_by_likelihood(empty)
    assert empty.candidates[selection.selected_rank].selection_eligible


def test_analysis_binds_live_evaluator_callable_not_only_source_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _trace(_transition_model(), max_new_tokens=2)
    ordinary = analyze_gvs_trace(trace, ())

    def forged_parser(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(canonical_json=lambda: '{"decision":"ABSTAIN"}')

    monkeypatch.setattr(gvs_decoder, "parse_action_ir", forged_parser)
    changed = analyze_gvs_trace(trace, ())
    assert ordinary.evaluator_source_sha256 == changed.evaluator_source_sha256
    assert ordinary.evaluator_runtime_sha256 != changed.evaluator_runtime_sha256
    assert sum(candidate.schema_valid for candidate in ordinary.candidates) < sum(
        candidate.schema_valid for candidate in changed.candidates
    )
    assert ordinary.sha256 != changed.sha256


def test_truncated_schema_valid_output_is_not_selection_eligible() -> None:
    trace = _trace(_uniform_tie_model(), max_new_tokens=1)
    analysis = analyze_gvs_trace(trace, ())
    valid_truncated = [candidate for candidate in analysis.candidates if candidate.schema_valid]

    assert valid_truncated
    assert all(candidate.truncated for candidate in valid_truncated)
    with pytest.raises(GVSDecoderError, match="no complete schema-valid candidate"):
        select_schema_valid_by_likelihood(analysis)


def test_per_slot_decode_and_native_generation_failures_are_retained() -> None:
    decode_trace = _trace(
        _transition_model(), tokenizer=_DecodeFailureTokenizer(), max_new_tokens=2
    )
    assert len(decode_trace.candidates) == 8
    assert any(candidate.failure_stage == "decode" for candidate in decode_trace.candidates)
    assert all(candidate.candidate_id for candidate in decode_trace.candidates)
    beam_candidates = decode_trace.candidates[1:]
    first_failed_rank = next(
        index for index, candidate in enumerate(beam_candidates) if candidate.failure_stage
    )
    assert all(candidate.failure_stage is None for candidate in beam_candidates[:first_failed_rank])
    assert all(
        candidate.failure_stage is not None for candidate in beam_candidates[first_failed_rank:]
    )
    assert [candidate.beam_rank for candidate in beam_candidates[first_failed_rank:]] == sorted(
        candidate.beam_rank for candidate in beam_candidates[first_failed_rank:]
    )

    float_model = _FloatGenerateModel(_uniform_tie_model().transitions)
    float_trace = _trace(float_model, max_new_tokens=1)
    assert float_trace.candidates[0].failure_stage == "native_generate"
    assert float_trace.candidates[0].output_present is False
    assert all(candidate.failure_stage is None for candidate in float_trace.candidates[1:])

    int32_trace = _trace(_Int32GenerateModel(_uniform_tie_model().transitions), max_new_tokens=1)
    assert int32_trace.candidates[0].failure_stage == "native_generate"
    assert int32_trace.candidates[0].failure_code == "GVSDecoderError"

    with pytest.raises(GVSDecoderError, match="dense floating CPU logits"):
        recompute_likelihood(
            _IntegerLogitModel(_uniform_tie_model().transitions),
            prompt_token_ids=(2,),
            generated_token_ids=(1,),
        )


def test_oversized_decodes_fail_per_slot_and_trace_remains_self_loadable() -> None:
    trace = _trace(
        _transition_model(),
        tokenizer=_LongDecodeTokenizer(),
        max_new_tokens=1,
    )

    assert len(trace.candidates) == 8
    assert all(candidate.failure_stage == "decode" for candidate in trace.candidates)
    assert all(candidate.failure_code == "GVSDecoderError" for candidate in trace.candidates)
    assert all(candidate.generated_token_ids for candidate in trace.candidates)
    assert all(candidate.raw_output is None for candidate in trace.candidates)
    artifact = trace.to_json_bytes()
    assert len(artifact) < 2 * 1024 * 1024
    assert (
        parse_candidate_set_trace_json(
            artifact,
            expected_trace_sha256=trace.sha256,
            expected_runtime_sha256=trace.runtime_sha256,
        )
        == trace
    )


def test_incomplete_native_rank_zero_is_failed_but_preserves_available_evidence() -> None:
    model = _EarlyShortModel(_uniform_tie_model().transitions)
    trace = _trace(model, max_new_tokens=2)
    greedy = trace.candidates[0]

    assert greedy.failure_stage == "native_generate"
    assert greedy.failure_code == "IncompleteNativeGenerationError"
    assert greedy.generated_token_ids == (3,)
    assert greedy.raw_output == '{"decision":"ABSTAIN"}'
    assert greedy.output_present
    assert greedy.likelihood is not None
    assert (
        parse_candidate_set_trace_json(
            trace.to_json_bytes(),
            expected_trace_sha256=trace.sha256,
            expected_runtime_sha256=trace.runtime_sha256,
        )
        == trace
    )
    analysis = analyze_gvs_trace(trace, ())
    assert analysis.candidates[0].generation_failed
    assert not analysis.candidates[0].selection_eligible


def test_native_tokens_after_eos_fail_and_preserve_the_exact_returned_ids() -> None:
    trace = _trace(_TokensAfterEOSModel(_uniform_tie_model().transitions), max_new_tokens=2)
    greedy = trace.candidates[0]

    assert greedy.failure_stage == "native_generate"
    assert greedy.failure_code == "NativeGenerationAfterEOSError"
    assert greedy.generated_token_ids == (1, 3)
    assert greedy.eos_emitted
    assert greedy.eos_position == 0
    assert greedy.content_token_ids == ()


def test_pre_post_runtime_identity_rejects_model_and_tokenizer_mutation() -> None:
    with pytest.raises(GVSDecoderError, match="identity mutated"):
        _trace(_StateMutatingModel(_uniform_tie_model().transitions), max_new_tokens=2)

    with pytest.raises(GVSDecoderError, match="identity mutated"):
        _trace(_uniform_tie_model(), tokenizer=_MutatingTokenizer(), max_new_tokens=2)


def test_caller_identity_claims_must_equal_runtime_and_rendered_prompt() -> None:
    model = _uniform_tie_model()
    tokenizer = _ToyTokenizer()
    runtime = inspect_gvs_runtime(model, tokenizer)
    source_sha256 = hashlib.sha256(
        render_prompt(now=NOW, tools=(), user_text=USER_TEXT).encode("utf-8")
    ).hexdigest()
    common = {
        "now": NOW,
        "tools": (),
        "user_text": USER_TEXT,
        "expected_runtime_sha256": runtime.sha256,
        "contract": GVSDecoderContract(max_new_tokens=2),
    }

    with pytest.raises(GVSDecoderError, match="model_sha256"):
        generate_gvs_trace(
            model,
            tokenizer,
            model_sha256="a" * 64,
            tokenizer_sha256=runtime.tokenizer_state_sha256,
            source_sha256=source_sha256,
            **common,
        )
    with pytest.raises(GVSDecoderError, match="tokenizer_sha256"):
        generate_gvs_trace(
            model,
            tokenizer,
            model_sha256=runtime.model_state_sha256,
            tokenizer_sha256="b" * 64,
            source_sha256=source_sha256,
            **common,
        )
    with pytest.raises(GVSDecoderError, match="source_sha256"):
        generate_gvs_trace(
            model,
            tokenizer,
            model_sha256=runtime.model_state_sha256,
            tokenizer_sha256=runtime.tokenizer_state_sha256,
            source_sha256="c" * 64,
            **common,
        )


def test_one_beam_rescore_failure_retains_only_that_beams_complete_evidence() -> None:
    model = _SelectiveLikelihoodFailureModel(_uniform_tie_model().transitions)
    trace = _trace(model, max_new_tokens=2)
    failures = [
        candidate for candidate in trace.candidates[1:] if candidate.failure_stage is not None
    ]

    assert len(failures) == 1
    assert failures[0].failure_stage == "likelihood"
    assert failures[0].generated_token_ids == (2, 3)
    assert failures[0].content_token_ids == (2, 3)
    assert failures[0].output_present
    assert failures[0].beam_search_raw_log_likelihood is not None
    assert sum(candidate.failure_stage is None for candidate in trace.candidates[1:]) == 6
    assert (
        parse_candidate_set_trace_json(
            trace.to_json_bytes(),
            expected_trace_sha256=trace.sha256,
            expected_runtime_sha256=trace.runtime_sha256,
        )
        == trace
    )

    with pytest.raises(GVSDecoderError, match="finite retained search score"):
        replace(failures[0], beam_search_raw_log_likelihood="not-a-float")
    with pytest.raises(GVSDecoderError, match="non-positive"):
        replace(failures[0], beam_search_raw_log_likelihood=1.0)
    with pytest.raises(GVSDecoderError, match="likelihood failure requires"):
        replace(
            failures[0],
            output_present=False,
            raw_output=None,
            raw_output_sha256=None,
        )


def test_failed_slot_stage_schemas_reject_cross_stage_forgery() -> None:
    trace = _trace(_FloatGenerateModel(_uniform_tie_model().transitions), max_new_tokens=1)
    native_failure = trace.candidates[0]
    assert native_failure.failure_stage == "native_generate"
    with pytest.raises(GVSDecoderError, match="beam-search failure"):
        replace(native_failure, failure_stage="beam_search")

    beam_failure_trace = _trace(
        _SelectiveLikelihoodFailureModel(_uniform_tie_model().transitions), max_new_tokens=2
    )
    likelihood_failure = next(
        candidate
        for candidate in beam_failure_trace.candidates
        if candidate.failure_stage == "likelihood"
    )
    payload = beam_failure_trace.to_artifact_record()
    payload["trace"]["candidates"][likelihood_failure.attempted_rank][
        "beam_search_raw_log_likelihood"
    ] = "not-a-float"
    with pytest.raises(GVSDecoderError, match="finite retained search score"):
        parse_candidate_set_trace_json(
            json.dumps(payload),
            expected_trace_sha256=beam_failure_trace.sha256,
            expected_runtime_sha256=beam_failure_trace.runtime_sha256,
        )


def test_trace_loader_round_trips_and_rejects_hash_rank_duplicate_key_and_size_tampering() -> None:
    trace = _trace(_transition_model(), max_new_tokens=2)
    loaded = parse_candidate_set_trace_json(
        trace.to_json_bytes(),
        expected_trace_sha256=trace.sha256,
        expected_runtime_sha256=trace.runtime_sha256,
    )
    assert loaded == trace

    payload = trace.to_artifact_record()
    payload["candidate_set_trace_sha256"] = "d" * 64
    with pytest.raises(GVSDecoderError, match="embedded candidate-set trace hash"):
        parse_candidate_set_trace_json(
            json.dumps(payload),
            expected_trace_sha256=trace.sha256,
            expected_runtime_sha256=trace.runtime_sha256,
        )

    payload = trace.to_artifact_record()
    payload["trace"]["candidates"][2]["beam_rank"] = payload["trace"]["candidates"][1]["beam_rank"]
    with pytest.raises(GVSDecoderError):
        parse_candidate_set_trace_json(
            json.dumps(payload),
            expected_trace_sha256=trace.sha256,
            expected_runtime_sha256=trace.runtime_sha256,
        )

    payload = trace.to_artifact_record()
    payload["trace"]["model_sha256"] = "d" * 64
    with pytest.raises(GVSDecoderError, match="exact runtime state hash"):
        parse_candidate_set_trace_json(
            json.dumps(payload),
            expected_trace_sha256=trace.sha256,
            expected_runtime_sha256=trace.runtime_sha256,
        )

    payload = trace.to_artifact_record()
    payload["trace"]["source_sha256"] = "d" * 64
    with pytest.raises(GVSDecoderError, match="rendered prompt hash"):
        parse_candidate_set_trace_json(
            json.dumps(payload),
            expected_trace_sha256=trace.sha256,
            expected_runtime_sha256=trace.runtime_sha256,
        )

    with pytest.raises(GVSDecoderError, match="duplicate key"):
        parse_candidate_set_trace_json(
            '{"artifact_schema_version":1,"artifact_schema_version":2}',
            expected_trace_sha256=trace.sha256,
            expected_runtime_sha256=trace.runtime_sha256,
        )
    with pytest.raises(GVSDecoderError, match="size bound"):
        parse_candidate_set_trace_json(
            b" " * (2 * 1024 * 1024 + 1),
            expected_trace_sha256=trace.sha256,
            expected_runtime_sha256=trace.runtime_sha256,
        )

    with pytest.raises(GVSDecoderError, match="oversized integer"):
        parse_candidate_set_trace_json(
            '{"x":' + ("9" * 5000) + "}",
            expected_trace_sha256=trace.sha256,
            expected_runtime_sha256=trace.runtime_sha256,
        )
    with pytest.raises(GVSDecoderError, match="non-finite float"):
        gvs_decoder._decode_bounded_json(
            "1e9999",
            maximum=1024,
            label="synthetic evidence",
        )
    with pytest.raises(GVSDecoderError, match="strict JSON"):
        parse_candidate_set_trace_json(
            ("[" * 1500) + "0" + ("]" * 1500),
            expected_trace_sha256=trace.sha256,
            expected_runtime_sha256=trace.runtime_sha256,
        )
    payload = trace.to_artifact_record()
    successful_rank = next(
        candidate.attempted_rank for candidate in trace.candidates if candidate.output_present
    )
    payload["trace"]["candidates"][successful_rank]["raw_output"] = "\ud800"
    with pytest.raises(GVSDecoderError, match="valid UTF-8"):
        parse_candidate_set_trace_json(
            json.dumps(payload),
            expected_trace_sha256=trace.sha256,
            expected_runtime_sha256=trace.runtime_sha256,
        )


def test_analysis_and_selection_loaders_recompute_every_field() -> None:
    trace = _trace(_transition_model(), max_new_tokens=2)
    analysis = analyze_gvs_trace(trace, ())
    selection = select_schema_valid_by_likelihood(analysis)

    assert (
        parse_candidate_set_analysis_json(
            analysis.to_json_bytes(),
            trace=trace,
            schemas=(),
            expected_analysis_sha256=analysis.sha256,
        )
        == analysis
    )
    assert (
        parse_likelihood_selection_json(
            selection.to_json_bytes(),
            analysis=analysis,
            expected_selection_sha256=selection.sha256,
        )
        == selection
    )

    tampered = analysis.to_artifact_record()
    tampered["analysis"]["candidates"][0]["parse_valid"] = True
    with pytest.raises(GVSDecoderError, match="differs from evaluator recomputation"):
        parse_candidate_set_analysis_json(
            json.dumps(tampered),
            trace=trace,
            schemas=(),
            expected_analysis_sha256=analysis.sha256,
        )

    tampered_selection = selection.to_artifact_record()
    tampered_selection["selection"]["selected_candidate_id"] = "forged-candidate"
    with pytest.raises(GVSDecoderError, match="differs from analysis recomputation"):
        parse_likelihood_selection_json(
            json.dumps(tampered_selection),
            analysis=analysis,
            expected_selection_sha256=selection.sha256,
        )

    with pytest.raises(GVSDecoderError, match="size bound"):
        parse_candidate_set_analysis_json(
            b" " * (2 * 1024 * 1024 + 1),
            trace=trace,
            schemas=(),
            expected_analysis_sha256=analysis.sha256,
        )
    with pytest.raises(GVSDecoderError, match="size bound"):
        parse_likelihood_selection_json(
            b" " * (64 * 1024 + 1),
            analysis=analysis,
            expected_selection_sha256=selection.sha256,
        )


def test_exact_beam_rank_and_analysis_error_invariants_reject_forgery() -> None:
    trace = _trace(_transition_model(), max_new_tokens=2)
    candidates = list(trace.candidates)
    candidates[2] = replace(candidates[2], beam_rank=candidates[1].beam_rank)
    with pytest.raises(GVSDecoderError, match="beam ranks"):
        replace(trace, candidates=tuple(candidates))

    with pytest.raises(GVSDecoderError, match="error stage"):
        CandidateAnalysis(
            attempted_rank=0,
            candidate_id="candidate",
            candidate_trace_sha256="a" * 64,
            raw_output_sha256="b" * 64,
            output_present=True,
            generation_failed=False,
            truncated=False,
            parse_valid=False,
            schema_valid=False,
            canonical_action_json=None,
            canonical_action_sha256=None,
            error_stage=None,
            error_code=None,
            error_path=None,
            normalized_log_likelihood=-1.0,
            raw_log_likelihood=-1.0,
        )


def test_context_overflow_missing_special_tokens_and_invalid_filler_fail_closed() -> None:
    model = _uniform_tie_model()
    model.config.max_seq_len = 2
    with pytest.raises(GVSDecoderError, match="exceeds model context"):
        _trace(model, max_new_tokens=2)

    class _MissingEOS(_ToyTokenizer):
        def token_to_id(self, token: str) -> int | None:
            return 0 if token == "<pad>" else None

    missing = _MissingEOS()
    runtime = inspect_gvs_runtime(_uniform_tie_model(), missing)
    with pytest.raises(GVSDecoderError, match="eos_token_id"):
        generate_gvs_trace(
            _uniform_tie_model(),
            missing,
            now=NOW,
            tools=(),
            user_text=USER_TEXT,
            model_sha256=runtime.model_state_sha256,
            tokenizer_sha256=runtime.tokenizer_state_sha256,
            source_sha256=hashlib.sha256(
                render_prompt(now=NOW, tools=(), user_text=USER_TEXT).encode("utf-8")
            ).hexdigest(),
            expected_runtime_sha256=runtime.sha256,
            contract=GVSDecoderContract(max_new_tokens=1),
        )

    with pytest.raises(GVSDecoderError, match="filler_mask"):
        recompute_likelihood(
            _uniform_tie_model(),
            prompt_token_ids=(2,),
            generated_token_ids=(0, 1),
            filler_mask=(False,),
        )
