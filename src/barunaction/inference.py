"""Fail-closed local inference for the BarunAction-35M proposal compiler."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from barunlm.evaluation.action_ir import (
    ActionIR,
    ActionIRParseError,
    ActionIRValidationError,
    Decision,
    JSONValue,
    ToolSchema,
    parse_action_ir,
)
from barunlm.evaluation.generation import GenerationError, load_verified_model
from barunlm.model import BarunLM
from barunlm.quantization import QuantizationError, load_verified_int8_model

from .candidate import CANDIDATE_CHECKPOINT_SHA256, candidate_identity
from .schema import ContractError, PreparedInput, ToolDeclaration, prepare_input

PROMPT_CONTRACT_VERSION = "barunaction-local-prompt-v1"
RESULT_SCHEMA_VERSION = "barunaction-inference-result-v1"
DEFAULT_MAX_NEW_TOKENS = 192


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    stage: str
    code: str
    message: str
    path: str = "$"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class PolicyAssessment:
    """Deterministic external gates; the model never grants either gate."""

    authorization_required: bool
    confirmation_required: bool
    execution_permitted: bool
    proposed_call_count: int
    side_effecting_tools: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "authorization_required": self.authorization_required,
            "confirmation_required": self.confirmation_required,
            "execution_permitted": self.execution_permitted,
            "proposed_call_count": self.proposed_call_count,
            "reason_codes": list(self.reason_codes),
            "side_effecting_tools": list(self.side_effecting_tools),
        }


@dataclass(frozen=True, slots=True)
class InferenceOutcome:
    """Exactly one parsed action or one deterministic error."""

    action: ActionIR | None
    error: ErrorDetail | None
    policy: PolicyAssessment | None
    raw_output: str | None
    prompt_sha256: str | None
    checkpoint_sha256: Mapping[str, str]
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    candidate_id: str | None = None
    candidate_run_id: str | None = None
    checkpoint_format: str | None = None
    quantization_manifest_sha256: str | None = None
    source_checkpoint_sha256: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if (self.action is None) == (self.error is None):
            raise ValueError("InferenceOutcome requires exactly one of action or error")
        if self.action is not None and self.policy is None:
            raise ValueError("a successful outcome requires a policy assessment")
        if self.action is None and self.policy is not None:
            raise ValueError("an error outcome cannot contain a policy assessment")

    @property
    def ok(self) -> bool:
        return self.action is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict() if self.action is not None else None,
            "checkpoint_sha256": dict(sorted(self.checkpoint_sha256.items())),
            "checkpoint_format": self.checkpoint_format,
            "error": self.error.to_dict() if self.error is not None else None,
            "generated_tokens": self.generated_tokens,
            "ok": self.ok,
            "policy": self.policy.to_dict() if self.policy is not None else None,
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "prompt_sha256": self.prompt_sha256,
            "prompt_tokens": self.prompt_tokens,
            "raw_output": self.raw_output,
            "quantization_manifest_sha256": self.quantization_manifest_sha256,
            "candidate_id": self.candidate_id,
            "candidate_run_id": self.candidate_run_id,
            "schema_version": RESULT_SCHEMA_VERSION,
            "source_checkpoint_sha256": (
                dict(sorted(self.source_checkpoint_sha256.items()))
                if self.source_checkpoint_sha256 is not None
                else None
            ),
        }


def assess_policy(action: ActionIR, schemas: Mapping[str, ToolSchema]) -> PolicyAssessment:
    calls = action.calls if action.decision in (Decision.CALL, Decision.CONFIRM) else ()
    side_effecting = tuple(
        sorted({call.tool for call in calls if schemas[call.tool].side_effecting})
    )
    authorization_required = bool(calls)
    confirmation_required = action.decision is Decision.CONFIRM or bool(side_effecting)
    reasons: list[str] = []
    if authorization_required:
        reasons.append("external_authorization_required")
    if confirmation_required:
        reasons.append("external_confirmation_required")
    if calls:
        reasons.append("model_output_is_proposal_only")
    return PolicyAssessment(
        authorization_required=authorization_required,
        confirmation_required=confirmation_required,
        execution_permitted=False,
        proposed_call_count=len(calls),
        side_effecting_tools=side_effecting,
        reason_codes=tuple(reasons),
    )


def render_prompt(prepared: PreparedInput) -> str:
    tool_lines = "\n".join(declaration.render() for declaration in prepared.declarations)
    context = "" if prepared.context_json == "{}" else f"CONTEXT\n{prepared.context_json}\n"
    return (
        "<bos><system>\n"
        "ACTION_IR_V1\n"
        f"NOW {prepared.now}\n"
        "TOOLS\n"
        f"{tool_lines}\n"
        f"{context}"
        "<user>\n"
        f"{prepared.request}\n"
        "<assistant>\n"
    )


def _error_outcome(
    detail: ErrorDetail,
    *,
    checkpoint_sha256: Mapping[str, str],
    raw_output: str | None = None,
    prompt_sha256: str | None = None,
    prompt_tokens: int | None = None,
    generated_tokens: int | None = None,
    checkpoint_format: str | None = None,
    quantization_manifest_sha256: str | None = None,
    source_checkpoint_sha256: Mapping[str, str] | None = None,
) -> InferenceOutcome:
    identity_hashes = (
        checkpoint_sha256 if source_checkpoint_sha256 is None else source_checkpoint_sha256
    )
    candidate_id, candidate_run_id = candidate_identity(identity_hashes)
    return InferenceOutcome(
        action=None,
        error=detail,
        policy=None,
        raw_output=raw_output,
        prompt_sha256=prompt_sha256,
        checkpoint_sha256=checkpoint_sha256,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        candidate_id=candidate_id,
        candidate_run_id=candidate_run_id,
        checkpoint_format=checkpoint_format,
        quantization_manifest_sha256=quantization_manifest_sha256,
        source_checkpoint_sha256=source_checkpoint_sha256,
    )


def validate_action_output(
    raw_output: str,
    *,
    declarations: tuple[ToolDeclaration, ...],
    checkpoint_sha256: Mapping[str, str] | None = None,
    prompt_sha256: str | None = None,
    prompt_tokens: int | None = None,
    generated_tokens: int | None = None,
    checkpoint_format: str | None = None,
    quantization_manifest_sha256: str | None = None,
    source_checkpoint_sha256: Mapping[str, str] | None = None,
) -> InferenceOutcome:
    hashes = {} if checkpoint_sha256 is None else checkpoint_sha256
    identity_hashes = hashes if source_checkpoint_sha256 is None else source_checkpoint_sha256
    candidate_id, candidate_run_id = candidate_identity(identity_hashes)
    registry = {item.schema.name: item.schema for item in declarations}
    try:
        action = parse_action_ir(raw_output, registry)
    except ActionIRParseError as error:
        return _error_outcome(
            ErrorDetail("parse", error.code, str(error), error.path),
            checkpoint_sha256=hashes,
            raw_output=raw_output,
            prompt_sha256=prompt_sha256,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            checkpoint_format=checkpoint_format,
            quantization_manifest_sha256=quantization_manifest_sha256,
            source_checkpoint_sha256=source_checkpoint_sha256,
        )
    except ActionIRValidationError as error:
        return _error_outcome(
            ErrorDetail("schema", error.code, str(error), error.path),
            checkpoint_sha256=hashes,
            raw_output=raw_output,
            prompt_sha256=prompt_sha256,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            checkpoint_format=checkpoint_format,
            quantization_manifest_sha256=quantization_manifest_sha256,
            source_checkpoint_sha256=source_checkpoint_sha256,
        )
    return InferenceOutcome(
        action=action,
        error=None,
        policy=assess_policy(action, registry),
        raw_output=raw_output,
        prompt_sha256=prompt_sha256,
        checkpoint_sha256=hashes,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        candidate_id=candidate_id,
        candidate_run_id=candidate_run_id,
        checkpoint_format=checkpoint_format,
        quantization_manifest_sha256=quantization_manifest_sha256,
        source_checkpoint_sha256=source_checkpoint_sha256,
    )


class BarunActionCompiler:
    """Verified deterministic inference; this class never invokes a declared tool."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        expected_sha256: Mapping[str, str] | None = None,
        checkpoint_format: str = "float",
        expected_int8_manifest_sha256: str | None = None,
        device: str = "cpu",
    ) -> None:
        if checkpoint_format not in {"float", "int8"}:
            raise GenerationError("checkpoint_format must be explicitly 'float' or 'int8'")
        if device not in {"cpu", "cuda"}:
            raise GenerationError("device must be explicitly 'cpu' or 'cuda'")
        if checkpoint_format == "int8" and device != "cpu":
            raise GenerationError("dynamic-int8 checkpoints require device='cpu'")
        if device == "cuda" and not torch.cuda.is_available():
            raise GenerationError("CUDA was requested but is unavailable")
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            raise GenerationError("the requested CUDA device does not support bfloat16")
        if checkpoint_format == "float":
            if expected_int8_manifest_sha256 is not None:
                raise GenerationError(
                    "expected_int8_manifest_sha256 is valid only for checkpoint_format='int8'"
                )
            float_hashes = (
                CANDIDATE_CHECKPOINT_SHA256 if expected_sha256 is None else expected_sha256
            )
            model, tokenizer, hashes = load_verified_model(
                checkpoint_dir, expected_sha256=float_hashes
            )
            source_hashes: Mapping[str, str] | None = None
            quantization_manifest_sha256 = None
        elif checkpoint_format == "int8":
            if expected_sha256 is not None:
                raise GenerationError(
                    "expected_sha256 is valid only for float checkpoints; "
                    "int8 requires an expected manifest SHA-256"
                )
            if expected_int8_manifest_sha256 is None:
                raise GenerationError(
                    "checkpoint_format='int8' requires expected_int8_manifest_sha256"
                )
            try:
                model, tokenizer, info = load_verified_int8_model(
                    checkpoint_dir,
                    expected_manifest_sha256=expected_int8_manifest_sha256,
                )
            except QuantizationError as error:
                raise GenerationError(str(error)) from error
            hashes = dict(info.artifact_sha256)
            source_hashes = info.source_checkpoint_sha256
            quantization_manifest_sha256 = info.manifest_sha256
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_sha256 = hashes
        self.checkpoint_format = checkpoint_format
        self.source_checkpoint_sha256 = source_hashes
        self.quantization_manifest_sha256 = quantization_manifest_sha256
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        if checkpoint_format == "int8":
            self.model: BarunLM = model.eval()
        else:
            self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self.tokenizer: Tokenizer = tokenizer
        self.eos_token_id = tokenizer.token_to_id("<eos>")
        self.pad_token_id = tokenizer.token_to_id("<pad>")
        if (
            self.eos_token_id is None
            or self.pad_token_id is None
            or self.eos_token_id == self.pad_token_id
        ):
            raise GenerationError("tokenizer must define distinct <eos> and <pad> tokens")

    def infer(
        self,
        *,
        request: Any,
        tool_schemas: Any,
        context: Any,
        now: Any,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> InferenceOutcome:
        provenance: dict[str, Any] = {
            "checkpoint_format": self.checkpoint_format,
            "checkpoint_sha256": self.checkpoint_sha256,
            "quantization_manifest_sha256": self.quantization_manifest_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
        }
        try:
            prepared = prepare_input(
                request=request,
                tool_schemas=tool_schemas,
                context=context,
                now=now,
            )
        except ContractError as error:
            return _error_outcome(
                ErrorDetail("input", error.code, error.message, error.path),
                **provenance,
            )
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            return _error_outcome(
                ErrorDetail("input", "invalid_generation_limit", "max_new_tokens must be positive"),
                **provenance,
            )

        prompt = render_prompt(prepared)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if not prompt_ids:
            return _error_outcome(
                ErrorDetail("generation", "empty_prompt", "rendered prompt encoded to zero tokens"),
                **provenance,
                prompt_sha256=prompt_sha256,
                prompt_tokens=0,
            )
        if len(prompt_ids) + max_new_tokens > self.model.config.max_seq_len:
            return _error_outcome(
                ErrorDetail(
                    "generation",
                    "context_overflow",
                    "prompt plus max_new_tokens exceeds the checkpoint context length",
                ),
                **provenance,
                prompt_sha256=prompt_sha256,
                prompt_tokens=len(prompt_ids),
            )

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.pad_token_id,
                )
        except torch.OutOfMemoryError:
            return _error_outcome(
                ErrorDetail("generation", "oom", "model generation ran out of memory"),
                **provenance,
                prompt_sha256=prompt_sha256,
                prompt_tokens=len(prompt_ids),
            )
        continuation = output_ids[0, len(prompt_ids) :].tolist()
        eos_position = next(
            (index for index, token_id in enumerate(continuation) if token_id == self.eos_token_id),
            None,
        )
        content_ids = continuation if eos_position is None else continuation[:eos_position]
        raw_output = self.tokenizer.decode(content_ids, skip_special_tokens=False)
        generated_tokens = len(continuation) if eos_position is None else eos_position + 1
        if eos_position is None:
            return _error_outcome(
                ErrorDetail(
                    "generation",
                    "generation_truncated",
                    "model did not emit EOS within max_new_tokens",
                ),
                **provenance,
                raw_output=raw_output,
                prompt_sha256=prompt_sha256,
                prompt_tokens=len(prompt_ids),
                generated_tokens=generated_tokens,
            )
        return validate_action_output(
            raw_output,
            declarations=prepared.declarations,
            **provenance,
            prompt_sha256=prompt_sha256,
            prompt_tokens=len(prompt_ids),
            generated_tokens=generated_tokens,
        )


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "PROMPT_CONTRACT_VERSION",
    "RESULT_SCHEMA_VERSION",
    "BarunActionCompiler",
    "ErrorDetail",
    "InferenceOutcome",
    "PolicyAssessment",
    "assess_policy",
    "render_prompt",
    "validate_action_output",
]
