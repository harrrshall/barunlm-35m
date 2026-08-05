from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import barunlm.evaluation.gvs_bridge as bridge_module
import barunlm.evaluation.gvs_shortcuts as shortcuts_module
from barunlm.datasets.mobile_actions import ToolArgument, ToolDefinition, render_prompt
from barunlm.evaluation.action_simulator import (
    SIMULATOR_TOOL_SCHEMAS,
    compile_program_step,
)
from barunlm.evaluation.gvs_bridge import (
    CandidateSupportBridgeInput,
    build_candidate_support_bridge,
    compute_gvs_source_commitment,
)
from barunlm.evaluation.gvs_decoder import (
    GVSDecoderContract,
    analyze_gvs_trace,
    generate_gvs_trace,
    inspect_gvs_runtime,
)
from barunlm.evaluation.gvs_faults import (
    FaultDescriptor,
    FaultKind,
    certify_single_fault,
)
from barunlm.evaluation.gvs_population import (
    AUDITED_BOOLEAN_STRATA,
    GVS_POPULATION_RECORD_SCHEMA_VERSION,
    POPULATION_ROLES,
    build_population_firewall,
    build_scan_evidence,
    seal_population_record,
)
from barunlm.evaluation.gvs_schema_presentation import (
    IDENTITY_PRESENTATION,
    ArgumentNameMap,
    SchemaPresentation,
    ToolNameMap,
    build_schema_presentation,
)
from barunlm.evaluation.gvs_shortcuts import (
    PROVISIONAL_SHORTCUT_PAIR_ACCURACY_REFERENCE,
    ComponentBalancedPairAccuracy,
    ExactRational,
    GVSShortcutError,
    ShortcutCandidateSignal,
    ShortcutControl,
    ShortcutEvaluationStatus,
    ShortcutPairScore,
    ShortcutRowEvidence,
    TNewShortcutCalibrationInput,
    build_d_support_shortcut_evidence,
    build_shortcut_audit,
    build_shortcut_v2_audit,
    compute_component_balanced_pair_accuracy,
    derive_d_support_shortcut_row,
    load_shortcut_audit_json,
    load_shortcut_v2_audit_json,
)
from barunlm.evaluation.gvs_support import (
    SupportClass,
    SupportSampleIdentity,
    SupportStrata,
)
from barunlm.evaluation.sim_program import (
    SIM_PROGRAM_VERSION,
    WORLD_STATE_VERSION,
    parse_sim_program,
)

NOW = "2026-08-04T09:30:00"
USER_TEXT = "Do not take any action."
V2_CUTOFF = "2026-01-01T00:00:00Z"
V2_ROLE_ASSIGNED = "2026-02-01T00:00:00Z"
V2_AUTHORED = "2026-02-02T00:00:00Z"
V2_SCAN_COMPLETED = "2026-02-03T00:00:00Z"
V2_SCAN_FROZEN = "2026-02-04T00:00:00Z"
V2_ELIGIBILITY_DECIDED = "2026-02-05T00:00:00Z"
V2_FIREWALL_FROZEN = "2026-02-06T00:00:00Z"


class _ShortcutTokenizer:
    def __init__(self) -> None:
        unsafe = {
            "decision": "CALL",
            "mode": "SINGLE",
            "calls": [
                {
                    "tool": "reminder_create",
                    "args": {
                        "due_at": "2026-08-06T12:00:00+05:30",
                        "reminder_id": "reminder.new",
                        "title": "Call",
                    },
                }
            ],
        }
        self.text = {
            0: "<pad>",
            1: "<eos>",
            2: "PROMPT",
            3: '{"decision":"ABSTAIN"}',
            4: json.dumps(unsafe, sort_keys=True, separators=(",", ":")),
            5: "not-json",
            6: '{"decision":"CLARIFY","missing":["recipient"]}',
            7: '{"decision":"BOGUS"}',
            8: "x",
        }

    def encode(self, sequence: str, add_special_tokens: bool = False) -> SimpleNamespace:
        assert sequence and add_special_tokens is False
        return SimpleNamespace(ids=[2])

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        return "".join(self.text[token_id] for token_id in ids)

    def token_to_id(self, token: str) -> int | None:
        return {"<pad>": 0, "<eos>": 1}.get(token)

    def to_str(self) -> str:
        return json.dumps(self.text, sort_keys=True, separators=(",", ":"))


class _TransitionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        logits = torch.full((9, 9), -8.0, dtype=torch.float64)
        logits[2] = torch.tensor([-9.0, 5.0, 1.0, 9.5, 10.0, 8.0, 7.0, 6.0, 4.0])
        for token_id in range(9):
            logits[token_id, 1] = 11.0
        logits[2, 4] = 12.0
        logits[2, 3] = 11.5
        self.register_buffer("transitions", logits)
        self.config = SimpleNamespace(vocab_size=9, max_seq_len=32)

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
            token = self(output).logits[:, -1].argmax(dim=-1, keepdim=True)
            output = torch.cat((output, token), dim=1)
            if eos_token_id is not None and token.item() == eos_token_id:
                break
        return output


def _world_payload() -> dict[str, object]:
    return {
        "schema_version": WORLD_STATE_VERSION,
        "reference_time": "2026-08-04T09:30:00+05:30",
        "timezone": "Asia/Kolkata",
        "reminders": {},
        "calendar": {},
        "contacts": {},
        "notes": {},
        "lists": {},
        "places": {},
        "routes": {},
        "outbox": [],
        "media": {"status": "stopped", "track_id": None, "catalog": {}},
        "settings": {"wifi": True, "airplane_mode": False, "brightness": 50},
    }


def _program(case_id: str):
    return parse_sim_program(
        {
            "schema_version": SIM_PROGRAM_VERSION,
            "case_id": case_id,
            "initial_state": _world_payload(),
            "steps": [
                {
                    "step_id": "step.one",
                    "decision": "ABSTAIN",
                    "mode": None,
                    "operations": [],
                    "missing": [],
                }
            ],
        }
    )


def _prompt_tools() -> tuple[ToolDefinition, ...]:
    return tuple(
        ToolDefinition(
            name=schema.name,
            description=f"Canonical {schema.name} operation.",
            arguments=tuple(
                ToolArgument(
                    name=name,
                    type_name=value.kind.value,
                    description=f"Value for {name}.",
                    required=name in schema.required,
                )
                for name, value in schema.arguments.items()
            ),
        )
        for schema in SIMULATOR_TOOL_SCHEMAS
    )


def _identity_schema_receipt(tools: tuple[ToolDefinition, ...]):
    presentation = SchemaPresentation(
        presentation_id="shortcut-identity-v1",
        mode=IDENTITY_PRESENTATION,
        tools=tuple(
            ToolNameMap(
                canonical_name=schema.name,
                presented_name=schema.name,
                arguments=tuple(
                    ArgumentNameMap(canonical_name=name, presented_name=name)
                    for name in sorted(schema.arguments)
                ),
            )
            for schema in sorted(SIMULATOR_TOOL_SCHEMAS, key=lambda value: value.name)
        ),
    )
    return build_schema_presentation(
        canonical_schemas=SIMULATOR_TOOL_SCHEMAS,
        prompt_tools=tools,
        presentation=presentation,
    )


def _tools_sha256(tools: tuple[ToolDefinition, ...]) -> str:
    record = [
        {
            "arguments": [
                {
                    "description": argument.description,
                    "name": argument.name,
                    "required": argument.required,
                    "type_name": argument.type_name,
                }
                for argument in tool.arguments
            ],
            "description": tool.description,
            "name": tool.name,
        }
        for tool in tools
    ]
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _v2_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _v2_bindings() -> dict[str, str]:
    return {
        "exact_scan_manifest_sha256": _v2_sha("v2-exact-scan"),
        "near_scan_manifest_sha256": _v2_sha("v2-near-scan"),
        "scan_config_sha256": _v2_sha("v2-scan-config"),
        "scan_code_sha256": _v2_sha("v2-scan-code"),
        "lineage_config_sha256": _v2_sha("v2-lineage-config"),
        "firewall_code_sha256": _v2_sha("v2-firewall-code"),
    }


def _v2_record(
    *,
    record_id: str,
    role: str,
    source_commitment_sha256: str,
) -> dict[str, object]:
    name = record_id.replace(":", "-").replace(".", "-")
    return seal_population_record(
        {
            "schema_version": GVS_POPULATION_RECORD_SCHEMA_VERSION,
            "record_id": record_id,
            "population_role": role,
            "task_class": "safety",
            "expected_outcome": "ABSTAIN",
            "action_family": "none",
            "strata": {value: False for value in AUDITED_BOOLEAN_STRATA},
            "provenance": {
                "author_id": f"author-{name}",
                "source_id": f"source-{name}",
                "collection_batch": f"batch-{name}",
                "schema_family": f"schema-{name}",
                "program_template_id": f"template-{name}",
                "paraphrase_family": f"paraphrase-{name}",
                "entity_pool_ids": [f"entity-{name}"],
                "temporal_construction_id": f"temporal-{name}",
                "source_commitment_sha256": source_commitment_sha256,
                "role_assigned_at_utc": V2_ROLE_ASSIGNED,
                "authored_at_utc": V2_AUTHORED,
                "license": "Apache-2.0",
                "consent": True,
                "no_model_assistance": role in {"S-new", "C-new"},
                "authoring_protocol_revision": "gvs-authoring-v1",
            },
            "duplicate_evidence": {
                "exact_content_sha256": _v2_sha(f"exact-{name}"),
                "normalized_request_sha256": _v2_sha(f"normalized-{name}"),
                "delexicalized_template_sha256": _v2_sha(f"delexicalized-{name}"),
                "near_duplicate_lineage_id": f"near-{name}",
                "scan_completed_at_utc": V2_SCAN_COMPLETED,
                **_v2_bindings(),
            },
            "eligibility": {
                "eligible": True,
                "decision": "include",
                "reason_code": None,
                "decision_stage": "pre-outcome",
                "evidence_sha256": _v2_sha(f"eligibility-{name}"),
                "decided_at_utc": V2_ELIGIBILITY_DECIDED,
            },
            "integrity": {"record_sha256": "0" * 64},
        },
        compared_release_cutoff_utc=V2_CUTOFF,
    )


@pytest.fixture(scope="module")
def bridge_inputs() -> tuple[CandidateSupportBridgeInput, CandidateSupportBridgeInput]:
    model = _TransitionModel()
    tokenizer = _ShortcutTokenizer()
    tools = _prompt_tools()
    schema_receipt = _identity_schema_receipt(tools)
    runtime = inspect_gvs_runtime(model, tokenizer)
    prompt = render_prompt(now=NOW, tools=tools, user_text=USER_TEXT)
    trace = generate_gvs_trace(
        model,
        tokenizer,
        now=NOW,
        tools=tools,
        user_text=USER_TEXT,
        model_sha256=runtime.model_state_sha256,
        tokenizer_sha256=runtime.tokenizer_state_sha256,
        source_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        expected_runtime_sha256=runtime.sha256,
        contract=GVSDecoderContract(max_new_tokens=2),
    )
    analysis = analyze_gvs_trace(trace, SIMULATOR_TOOL_SCHEMAS)
    values: list[CandidateSupportBridgeInput] = []
    for index, (case_id, component_id) in enumerate(
        (("case.shortcut.a", "a" * 64), ("case.shortcut.b", "b" * 64)),
        start=3,
    ):
        program = _program(case_id)
        commitment = compute_gvs_source_commitment(
            program_sha256=program.sha256(),
            step_id="step.one",
            prompt_sha256=trace.prompt_sha256,
            prompt_tools_sha256=_tools_sha256(tools),
            schema_presentation_sha256=schema_receipt.presentation_sha256,
            schema_presentation_receipt_sha256=schema_receipt.sha256,
            schema_presentation_artifact_sha256=hashlib.sha256(
                schema_receipt.to_json_bytes()
            ).hexdigest(),
            schema_presentation_mode=schema_receipt.presentation.mode,
        )
        sample = SupportSampleIdentity(
            sample_id=f"{case_id}:step.one",
            firewall_sha256="1" * 64,
            scan_evidence_sha256="2" * 64,
            population_record_sha256=str(index) * 64,
            component_id=component_id,
            source_commitment_sha256=commitment,
            task_class=SupportClass.SAFETY,
            expected_outcome="ABSTAIN",
            action_family="none",
            strata=SupportStrata(tuple((name, False) for name in AUDITED_BOOLEAN_STRATA)),
        )
        receipt = build_candidate_support_bridge(
            sample=sample,
            trace=trace,
            analysis=analysis,
            program=program,
            step_id="step.one",
            now=NOW,
            user_text=USER_TEXT,
            prompt_tools=tools,
            schema_presentation_receipt=schema_receipt,
        )
        values.append(
            CandidateSupportBridgeInput(
                receipt=receipt,
                sample=sample,
                trace=trace,
                analysis=analysis,
                program=program,
                step_id="step.one",
                now=NOW,
                user_text=USER_TEXT,
                prompt_tools=tools,
                schema_presentation_receipt=schema_receipt,
            )
        )
    return values[0], values[1]


def _make_v2_fixture(
    base: CandidateSupportBridgeInput,
    *,
    distinct_live_requests: bool,
) -> dict[str, object]:
    programs = {
        "T-new": _program("case.v2.t"),
        "D-support": _program("case.v2.d"),
    }
    live_decoder_evidence = {
        "T-new": (base.trace, base.analysis, USER_TEXT),
        "D-support": (base.trace, base.analysis, USER_TEXT),
    }
    if distinct_live_requests:
        d_user_text = "Leave the D-support device unchanged."
        model = _TransitionModel()
        tokenizer = _ShortcutTokenizer()
        runtime = inspect_gvs_runtime(model, tokenizer)
        prompt = render_prompt(
            now=NOW,
            tools=base.prompt_tools,
            user_text=d_user_text,
        )
        d_trace = generate_gvs_trace(
            model,
            tokenizer,
            now=NOW,
            tools=base.prompt_tools,
            user_text=d_user_text,
            model_sha256=runtime.model_state_sha256,
            tokenizer_sha256=runtime.tokenizer_state_sha256,
            source_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            expected_runtime_sha256=runtime.sha256,
            contract=GVSDecoderContract(max_new_tokens=2),
        )
        live_decoder_evidence["D-support"] = (
            d_trace,
            analyze_gvs_trace(d_trace, SIMULATOR_TOOL_SCHEMAS),
            d_user_text,
        )
    commitments: dict[str, str] = {}
    for role, program in programs.items():
        trace, _, _ = live_decoder_evidence[role]
        commitments[role] = compute_gvs_source_commitment(
            program_sha256=program.sha256(),
            step_id="step.one",
            prompt_sha256=trace.prompt_sha256,
            prompt_tools_sha256=_tools_sha256(base.prompt_tools),
            schema_presentation_sha256=(base.schema_presentation_receipt.presentation_sha256),
            schema_presentation_receipt_sha256=(base.schema_presentation_receipt.sha256),
            schema_presentation_artifact_sha256=hashlib.sha256(
                base.schema_presentation_receipt.to_json_bytes()
            ).hexdigest(),
            schema_presentation_mode=(base.schema_presentation_receipt.presentation.mode),
        )
    populations = {
        "T-new": [
            _v2_record(
                record_id="case.v2.t:step.one",
                role="T-new",
                source_commitment_sha256=commitments["T-new"],
            )
        ],
        "D-support": [
            _v2_record(
                record_id="case.v2.d:step.one",
                role="D-support",
                source_commitment_sha256=commitments["D-support"],
            )
        ],
        "S-new": [
            _v2_record(
                record_id="case.v2.s:step.one",
                role="S-new",
                source_commitment_sha256=_v2_sha("v2-s-source"),
            )
        ],
        "C-new": [
            _v2_record(
                record_id="case.v2.c:step.one",
                role="C-new",
                source_commitment_sha256=_v2_sha("v2-c-source"),
            )
        ],
    }
    membership = {
        role: [str(value["record_id"]) for value in populations[role]] for role in POPULATION_ROLES
    }
    bindings = _v2_bindings()
    scan = build_scan_evidence(
        populations,
        expected_membership=membership,
        expected_scan_bindings=bindings,
        compared_release_cutoff_utc=V2_CUTOFF,
        scan_evidence_frozen_at_utc=V2_SCAN_FROZEN,
    )
    firewall = build_population_firewall(
        populations,
        expected_membership=membership,
        expected_scan_bindings=bindings,
        scan_evidence=scan,
        compared_release_cutoff_utc=V2_CUTOFF,
        scan_evidence_frozen_at_utc=V2_SCAN_FROZEN,
        firewall_frozen_at_utc=V2_FIREWALL_FROZEN,
    )
    firewall_args = {
        "population_records": populations,
        "expected_membership": membership,
        "expected_scan_bindings": bindings,
        "scan_evidence": scan,
        "firewall_manifest": firewall,
        "compared_release_cutoff_utc": V2_CUTOFF,
        "scan_evidence_frozen_at_utc": V2_SCAN_FROZEN,
        "firewall_frozen_at_utc": V2_FIREWALL_FROZEN,
    }
    identities = shortcuts_module.derive_shortcut_v2_population_identities(**firewall_args)
    live_inputs: dict[str, CandidateSupportBridgeInput] = {}
    for role in ("T-new", "D-support"):
        program = programs[role]
        sample = identities[role][0]
        trace, analysis, user_text = live_decoder_evidence[role]
        receipt = build_candidate_support_bridge(
            sample=sample,
            trace=trace,
            analysis=analysis,
            program=program,
            step_id="step.one",
            now=NOW,
            user_text=user_text,
            prompt_tools=base.prompt_tools,
            schema_presentation_receipt=base.schema_presentation_receipt,
        )
        live_inputs[role] = CandidateSupportBridgeInput(
            receipt=receipt,
            sample=sample,
            trace=trace,
            analysis=analysis,
            program=program,
            step_id="step.one",
            now=NOW,
            user_text=user_text,
            prompt_tools=base.prompt_tools,
            schema_presentation_receipt=base.schema_presentation_receipt,
        )
    descriptor = FaultDescriptor(
        kind=FaultKind.DECISION,
        call_index=None,
        argument="decision",
        replacement=json.loads(_ShortcutTokenizer().text[4]),
    )
    t_program = programs["T-new"]
    certification = certify_single_fault(
        t_program.initial_state,
        compile_program_step(t_program.steps[0]),
        descriptor,
    )
    return {
        "firewall_args": firewall_args,
        "t_new_inputs": [
            TNewShortcutCalibrationInput(
                bridge_input=live_inputs["T-new"],
                fault_certification=certification,
            )
        ],
        "d_support_inputs": [live_inputs["D-support"]],
    }


@pytest.fixture(scope="module")
def v2_fixture(
    bridge_inputs: tuple[CandidateSupportBridgeInput, CandidateSupportBridgeInput],
) -> dict[str, object]:
    return _make_v2_fixture(bridge_inputs[0], distinct_live_requests=True)


def _pair(
    component: str,
    sample: str,
    positive: str,
    negative: str,
    positive_score: Fraction,
    negative_score: Fraction,
) -> ShortcutPairScore:
    return ShortcutPairScore(
        component_id=component,
        sample_id=sample,
        positive_candidate_sha256=positive * 64,
        negative_candidate_sha256=negative * 64,
        positive_score=ExactRational.from_fraction(positive_score),
        negative_score=ExactRational.from_fraction(negative_score),
    )


def test_inventory_is_explicitly_not_evaluated_and_nonauthorizing() -> None:
    receipt = build_shortcut_audit()

    assert receipt.d_support_evidence is None
    assert receipt.authenticated_t_new_calibration_present is False
    assert receipt.d_support_population_completeness_authenticated is False
    assert receipt.scientific_result_available is False
    assert receipt.provisional_reference_is_gate is False
    assert receipt.authorizes_model_or_label_access is False
    assert receipt.authorizes_cuda_or_jarvis_access is False
    assert receipt.certifies_external_custody is False
    assert PROVISIONAL_SHORTCUT_PAIR_ACCURACY_REFERENCE == Fraction(3, 5)
    assert tuple(status.control for status in receipt.controls) == tuple(ShortcutControl)

    for status in receipt.controls:
        assert status.calibration_population == "T-new"
        assert status.evaluation_population == "D-support"
        assert status.calibration_status is ShortcutEvaluationStatus.NOT_EVALUATED
        assert status.evaluation_status is ShortcutEvaluationStatus.NOT_EVALUATED
        assert status.pair_accuracy is None
        assert status.below_provisional_reference is None
        assert status.passes_gate is None
        assert status.provisional_reference_is_gate is False
        assert "authenticated_recomputed_T-new_shortcut_bridge_receipt_absent" in (
            status.missing_bindings
        )
        assert "recomputed_D-support_shortcut_evidence_absent" in status.missing_bindings
        assert (
            "authenticated_complete_D-support_population_denominator_binding_absent"
            in status.missing_bindings
        )
    tool_status = next(
        status for status in receipt.controls if status.control is ShortcutControl.TOOL_FREQUENCY
    )
    assert tool_status.feature_derivation_implemented is False
    assert tool_status.d_support_feature_evidence_present is False
    assert "bridge_bound_lowered_candidate_tool_multiset_or_family_absent" in (
        tool_status.missing_bindings
    )


def test_inventory_artifact_round_trip_is_strict_and_deterministic() -> None:
    receipt = build_shortcut_audit()
    raw = receipt.to_json_bytes()

    loaded = load_shortcut_audit_json(raw, expected_receipt_sha256=receipt.sha256)
    assert loaded == receipt
    assert loaded.to_json_bytes() == raw
    assert len(receipt.source_files_sha256) == 64
    assert len(receipt.runtime_identity_sha256) == 64


def test_live_d_support_derives_only_rank_hidden_signals_and_all_exact_roots(
    bridge_inputs: tuple[CandidateSupportBridgeInput, CandidateSupportBridgeInput],
) -> None:
    evidence = build_d_support_shortcut_evidence(list(reversed(bridge_inputs)))

    assert evidence.population_completeness_authenticated is False
    assert evidence.row_count == 2
    assert evidence.component_count == 2
    assert tuple(row.sample_id for row in evidence.rows) == tuple(
        sorted(value.sample.sample_id for value in bridge_inputs)
    )
    for root in (
        evidence.firewall_sha256,
        evidence.scan_evidence_sha256,
        evidence.population_record_root_sha256,
        evidence.component_assignment_sha256,
        evidence.bridge_receipt_root_sha256,
        evidence.rank_hidden_payload_root_sha256,
        evidence.semantic_evidence_root_sha256,
    ):
        assert len(root) == 64
    for row in evidence.rows:
        identities = tuple(candidate.semantic_identity_sha256 for candidate in row.candidates)
        assert identities == tuple(sorted(set(identities)))
        assert len(row.candidates) <= 8
        assert all(candidate.syntax_parse_valid for candidate in row.candidates)
        assert all(candidate.validity_schema_valid_nontruncated for candidate in row.candidates)
        assert all(candidate.tool_identity is None for candidate in row.candidates)
        assert all(candidate.tool_feature_binding_sha256 is None for candidate in row.candidates)
        assert all(
            candidate.exact_semantic_match
            is (candidate.semantic_identity_sha256 == row.gold_semantic_action_ir_sha256)
            for candidate in row.candidates
        )
        candidate_keys = set(row.to_record()["candidates"][0])
        assert candidate_keys.isdisjoint(
            {"attempted_rank", "origin", "likelihood", "list_position", "beam"}
        )

    forward = build_d_support_shortcut_evidence(list(bridge_inputs))
    assert forward == evidence
    assert forward.sha256 == evidence.sha256


def test_live_audit_keeps_all_controls_not_evaluated(
    bridge_inputs: tuple[CandidateSupportBridgeInput, CandidateSupportBridgeInput],
) -> None:
    receipt = build_shortcut_audit(list(bridge_inputs))

    assert receipt.d_support_evidence is not None
    assert receipt.d_support_population_completeness_authenticated is False
    assert receipt.scientific_result_available is False
    for status in receipt.controls:
        assert status.evaluation_status is ShortcutEvaluationStatus.NOT_EVALUATED
        assert status.calibration_status is ShortcutEvaluationStatus.NOT_EVALUATED
        assert status.pair_accuracy is None
        assert status.passes_gate is None
        assert "recomputed_D-support_shortcut_evidence_absent" not in status.missing_bindings
        assert (
            "authenticated_complete_D-support_population_denominator_binding_absent"
            in status.missing_bindings
        )
        if status.control is ShortcutControl.TOOL_FREQUENCY:
            assert status.d_support_feature_evidence_present is False
        else:
            assert status.d_support_feature_evidence_present is True


def test_serialized_d_support_requires_complete_live_reverification(
    bridge_inputs: tuple[CandidateSupportBridgeInput, CandidateSupportBridgeInput],
) -> None:
    receipt = build_shortcut_audit(list(bridge_inputs))
    raw = receipt.to_json_bytes()

    with pytest.raises(GVSShortcutError, match="requires complete live bridge inputs"):
        load_shortcut_audit_json(raw, expected_receipt_sha256=receipt.sha256)
    with pytest.raises(GVSShortcutError, match="requires complete live bridge inputs"):
        load_shortcut_audit_json(
            raw,
            expected_receipt_sha256=receipt.sha256,
            bridge_inputs=[],
        )
    with pytest.raises(GVSShortcutError, match="differs from live bridge recomputation"):
        load_shortcut_audit_json(
            raw,
            expected_receipt_sha256=receipt.sha256,
            bridge_inputs=[bridge_inputs[0]],
        )

    loaded = load_shortcut_audit_json(
        raw,
        expected_receipt_sha256=receipt.sha256,
        bridge_inputs=list(reversed(bridge_inputs)),
    )
    assert loaded == receipt


def test_inventory_artifact_rejects_bridge_inputs(
    bridge_inputs: tuple[CandidateSupportBridgeInput, CandidateSupportBridgeInput],
) -> None:
    receipt = build_shortcut_audit()
    with pytest.raises(GVSShortcutError, match="must not receive bridge inputs"):
        load_shortcut_audit_json(
            receipt.to_json_bytes(),
            expected_receipt_sha256=receipt.sha256,
            bridge_inputs=[bridge_inputs[0]],
        )


def test_bridge_receipt_is_recomputed_instead_of_trusted(
    bridge_inputs: tuple[CandidateSupportBridgeInput, CandidateSupportBridgeInput],
) -> None:
    value = bridge_inputs[0]
    forged = replace(value.receipt, schema_presentation_id="forged-presentation-id")
    forged_input = replace(value, receipt=forged)

    with pytest.raises(GVSShortcutError, match="complete live recomputation"):
        derive_d_support_shortcut_row(forged_input)


def test_bridge_verifier_monkeypatch_fails_before_evidence_derivation(
    bridge_inputs: tuple[CandidateSupportBridgeInput, CandidateSupportBridgeInput],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bridge_module,
        "verify_candidate_support_bridge",
        lambda *args, **kwargs: bridge_inputs[0].receipt,
    )
    with pytest.raises(GVSShortcutError, match="changed identity"):
        derive_d_support_shortcut_row(bridge_inputs[0])


def test_runtime_hash_helper_replacement_cannot_return_the_pinned_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        shortcuts_module,
        "module_runtime_sha256",
        lambda *_args, **_kwargs: shortcuts_module._INITIAL_RUNTIME_SHA256,
    )
    with pytest.raises(GVSShortcutError, match="runtime-hash callable changed identity"):
        shortcuts_module.assert_shortcut_runtime_integrity()


def test_caller_subset_never_authenticates_the_d_support_denominator(
    bridge_inputs: tuple[CandidateSupportBridgeInput, CandidateSupportBridgeInput],
) -> None:
    receipt = build_shortcut_audit([bridge_inputs[0]])
    assert receipt.d_support_evidence is not None
    assert receipt.d_support_evidence.row_count == 1
    assert receipt.d_support_evidence.population_completeness_authenticated is False
    assert receipt.d_support_population_completeness_authenticated is False
    for status in receipt.controls:
        assert status.evaluation_status is ShortcutEvaluationStatus.NOT_EVALUATED
        assert (
            "authenticated_complete_D-support_population_denominator_binding_absent"
            in status.missing_bindings
        )


def test_explicit_empty_or_wrong_typed_population_fails_closed() -> None:
    with pytest.raises(GVSShortcutError, match="must be nonempty"):
        build_shortcut_audit([])
    with pytest.raises(GVSShortcutError, match="None, an exact list, or an exact tuple"):
        build_shortcut_audit(set())  # type: ignore[arg-type]
    with pytest.raises(GVSShortcutError, match="nonempty exact list or tuple"):
        build_d_support_shortcut_evidence([])


def test_pair_accuracy_is_exact_component_balanced_and_permutation_stable() -> None:
    pairs = [
        _pair("component.a", "sample.a1", "1", "2", Fraction(1), Fraction(0)),
        _pair("component.a", "sample.a1", "3", "4", Fraction(1, 2), Fraction(1, 2)),
        _pair("component.a", "sample.a2", "5", "6", Fraction(0), Fraction(1)),
        _pair("component.b", "sample.b1", "7", "8", Fraction(1), Fraction(0)),
    ]

    result = compute_component_balanced_pair_accuracy(pairs)
    reversed_result = compute_component_balanced_pair_accuracy(list(reversed(pairs)))

    assert type(result) is ComponentBalancedPairAccuracy
    assert result == reversed_result
    assert result.accuracy.fraction == Fraction(11, 16)
    assert result.pair_count == 4
    assert result.tie_count == 1
    assert result.sample_count == 3
    assert result.component_count == 2
    assert result.to_record()["aggregation"] == ("pair_mean_then_sample_mean_then_component_mean")
    assert result.to_record()["tie_credit"]["exact"] == "1/2"


def test_ties_receive_half_credit_without_hash_or_position_tie_break() -> None:
    first = _pair(
        "component",
        "sample",
        "f",
        "0",
        Fraction(1, 3),
        Fraction(1, 3),
    )
    second = _pair(
        "component",
        "sample",
        "0",
        "f",
        Fraction(1, 3),
        Fraction(1, 3),
    )

    assert compute_component_balanced_pair_accuracy([first]).accuracy.fraction == Fraction(1, 2)
    assert compute_component_balanced_pair_accuracy([second]).accuracy.fraction == Fraction(1, 2)


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [
        (True, 1),
        (1, True),
        (-1, 1),
        (2, 1),
        (2, 4),
        (1.0, 2),
    ],
)
def test_exact_rational_rejects_nonexact_nonreduced_or_out_of_range_values(
    numerator: object,
    denominator: object,
) -> None:
    with pytest.raises(GVSShortcutError):
        ExactRational(numerator, denominator)  # type: ignore[arg-type]


def test_pair_accuracy_rejects_duplicate_or_invalid_semantic_pairs() -> None:
    pair = _pair("component", "sample", "1", "2", Fraction(1), Fraction(0))
    changed_score = replace(pair, positive_score=ExactRational(1, 2))

    with pytest.raises(GVSShortcutError, match="duplicate semantic comparisons"):
        compute_component_balanced_pair_accuracy([pair, changed_score])
    with pytest.raises(GVSShortcutError, match="nonempty exact list or tuple"):
        compute_component_balanced_pair_accuracy([])
    with pytest.raises(GVSShortcutError, match="nonempty exact list or tuple"):
        compute_component_balanced_pair_accuracy({pair})  # type: ignore[arg-type]
    with pytest.raises(GVSShortcutError, match="distinct semantic candidates"):
        replace(pair, negative_candidate_sha256=pair.positive_candidate_sha256)


def test_factory_only_receipts_and_signal_tool_absence_fail_closed() -> None:
    with pytest.raises(TypeError, match="live support bridge"):
        ShortcutRowEvidence(
            sample_id="sample",
            firewall_sha256="1" * 64,
            scan_evidence_sha256="2" * 64,
            population_record_sha256="3" * 64,
            component_id="component",
            source_commitment_sha256="4" * 64,
            bridge_receipt_sha256="5" * 64,
            bridge_runtime_identity_sha256="6" * 64,
            rank_hidden_payload_sha256="7" * 64,
            rank_hidden_batch_sha256="8" * 64,
            schema_presentation_receipt_sha256="9" * 64,
            gold_semantic_action_ir_sha256="a" * 64,
            candidates=(),
        )
    with pytest.raises(GVSShortcutError, match="cannot supply a candidate tool-frequency"):
        ShortcutCandidateSignal(
            semantic_identity_sha256="a" * 64,
            candidate_evidence_sha256="b" * 64,
            content_token_count=1,
            syntax_parse_valid=True,
            validity_schema_valid_nontruncated=True,
            exact_semantic_match=False,
            tool_identity="reminder_create",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}',
        b'{"x":1.0}',
        b'{"x":NaN}',
        b'{"x":123456789012345678901}',
        b"\xff",
    ],
)
def test_artifact_loader_rejects_duplicate_float_nonfinite_large_integer_or_utf8(
    raw: bytes,
) -> None:
    with pytest.raises(GVSShortcutError):
        load_shortcut_audit_json(raw, expected_receipt_sha256="0" * 64)


def test_artifact_loader_rejects_tamper_wrong_commitment_and_wrong_container() -> None:
    receipt = build_shortcut_audit()
    raw = receipt.to_json_bytes()
    payload = json.loads(raw)
    payload["receipt"]["authorizes_model_or_label_access"] = True
    tampered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(GVSShortcutError):
        load_shortcut_audit_json(tampered, expected_receipt_sha256=receipt.sha256)
    with pytest.raises(GVSShortcutError, match="hash failed caller-bound recomputation"):
        load_shortcut_audit_json(raw, expected_receipt_sha256="f" * 64)
    with pytest.raises(GVSShortcutError, match="empty, non-bytes, or oversized"):
        load_shortcut_audit_json(bytearray(raw), expected_receipt_sha256=receipt.sha256)  # type: ignore[arg-type]
    with pytest.raises(GVSShortcutError, match="empty, non-bytes, or oversized"):
        load_shortcut_audit_json(
            b"0" * (shortcuts_module.MAX_SHORTCUT_ARTIFACT_BYTES + 1),
            expected_receipt_sha256=receipt.sha256,
        )


def test_serialized_semantic_root_tamper_is_rejected_before_live_acceptance(
    bridge_inputs: tuple[CandidateSupportBridgeInput, CandidateSupportBridgeInput],
) -> None:
    receipt = build_shortcut_audit(list(bridge_inputs))
    payload = json.loads(receipt.to_json_bytes())
    payload["receipt"]["d_support_evidence"]["semantic_evidence_root_sha256"] = "0" * 64
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(GVSShortcutError, match="semantic_evidence_root_sha256"):
        load_shortcut_audit_json(
            raw,
            expected_receipt_sha256=receipt.sha256,
            bridge_inputs=list(bridge_inputs),
        )


def test_v2_missing_live_populations_serializes_only_not_evaluated(
    v2_fixture: dict[str, object],
) -> None:
    receipt = build_shortcut_v2_audit(**v2_fixture["firewall_args"])

    assert receipt.structural_result_available is False
    assert receipt.scientific_result_available is False
    assert receipt.external_custody_authenticated is False
    assert receipt.rules is None
    assert receipt.t_new_rows == ()
    assert receipt.d_support_rows == ()
    assert all(
        value.evaluation_status is ShortcutEvaluationStatus.NOT_EVALUATED
        and value.calibration_pair_accuracy is None
        and value.evaluation_pair_accuracy is None
        for value in receipt.controls
    )
    assert "complete_firewall_bound_T-new_calibration_inputs_absent" in receipt.blockers
    assert "complete_firewall_bound_D-support_evaluation_inputs_absent" in receipt.blockers
    assert "external_secret_custody_and_single_use_signer_authentication_absent" in receipt.blockers


def test_v2_complete_structural_controls_bind_tools_and_component_balance(
    v2_fixture: dict[str, object],
) -> None:
    receipt = build_shortcut_v2_audit(
        t_new_inputs=v2_fixture["t_new_inputs"],
        d_support_inputs=v2_fixture["d_support_inputs"],
        **v2_fixture["firewall_args"],
    )

    assert receipt.structural_result_available is True
    assert receipt.scientific_result_available is False
    assert receipt.external_custody_authenticated is False
    assert receipt.authorizes_model_or_label_access is False
    assert receipt.authorizes_cuda_or_jarvis_access is False
    assert receipt.launch_authorized is False
    assert receipt.rules is not None
    assert receipt.t_new_population_evidence is not None
    assert receipt.d_support_population_evidence is not None
    assert receipt.d_support_population_evidence.complete_firewall_bound_denominator is True
    calibration = receipt.t_new_rows[0]
    assert calibration.positive_tool_identities == ()
    assert calibration.negative_tool_identities == ("reminder_create",)
    assert (
        calibration.row.program_sha256
        == v2_fixture["t_new_inputs"][0].bridge_input.program.sha256()
    )
    for result in receipt.controls:
        assert result.calibration_status is ShortcutEvaluationStatus.EVALUATED
        assert result.evaluation_status is ShortcutEvaluationStatus.EVALUATED
        assert result.calibration_pair_accuracy is not None
        assert result.evaluation_pair_accuracy is not None
        assert result.evaluation_pair_accuracy.pair_count == (
            len(receipt.d_support_rows[0].positives) * len(receipt.d_support_rows[0].negatives)
        )
        assert result.evaluation_pair_accuracy.sample_count == 1
        assert result.evaluation_pair_accuracy.component_count == 1
        assert result.passes_gate is None
        assert result.scientific_result_available is False

    forbidden = {"attempted_rank", "origin", "likelihood", "list_position", "beam_rank"}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(receipt.to_record())


def test_v2_requires_complete_membership_and_loader_rebuilds_live(
    v2_fixture: dict[str, object],
) -> None:
    with pytest.raises(GVSShortcutError, match="nonempty bounded exact sequence"):
        build_shortcut_v2_audit(
            t_new_inputs=[],
            d_support_inputs=v2_fixture["d_support_inputs"],
            **v2_fixture["firewall_args"],
        )

    receipt = build_shortcut_v2_audit(
        t_new_inputs=v2_fixture["t_new_inputs"],
        d_support_inputs=v2_fixture["d_support_inputs"],
        **v2_fixture["firewall_args"],
    )
    raw = receipt.to_json_bytes()
    rebuilt = load_shortcut_v2_audit_json(
        raw,
        expected_receipt_sha256=receipt.sha256,
        t_new_inputs=v2_fixture["t_new_inputs"],
        d_support_inputs=v2_fixture["d_support_inputs"],
        **v2_fixture["firewall_args"],
    )
    assert rebuilt.to_json_bytes() == raw

    payload = json.loads(raw)
    payload["receipt"]["scientific_result_available"] = True
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload["receipt"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    tampered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(GVSShortcutError, match="differs from complete live recomputation"):
        load_shortcut_v2_audit_json(
            tampered,
            expected_receipt_sha256=payload["receipt_sha256"],
            t_new_inputs=v2_fixture["t_new_inputs"],
            d_support_inputs=v2_fixture["d_support_inputs"],
            **v2_fixture["firewall_args"],
        )


@pytest.mark.parametrize(
    "binding_name",
    (
        "_ACTION_SIMULATOR_CLASS",
        "SIMULATOR_TOOL_REGISTRY",
        "SIMULATOR_TOOL_SCHEMAS",
    ),
)
def test_v2_runtime_integrity_rejects_local_dependency_substitution(
    monkeypatch: pytest.MonkeyPatch,
    binding_name: str,
) -> None:
    monkeypatch.setattr(shortcuts_module, binding_name, object())
    with pytest.raises(GVSShortcutError, match="changed identity"):
        shortcuts_module.assert_shortcut_runtime_integrity()


def test_v2_rejects_live_request_or_trace_reuse_across_t_and_d(
    bridge_inputs: tuple[CandidateSupportBridgeInput, CandidateSupportBridgeInput],
) -> None:
    reused = _make_v2_fixture(bridge_inputs[0], distinct_live_requests=False)
    with pytest.raises(GVSShortcutError, match="live exact request evidence overlaps"):
        build_shortcut_v2_audit(
            t_new_inputs=reused["t_new_inputs"],
            d_support_inputs=reused["d_support_inputs"],
            **reused["firewall_args"],
        )


def test_v2_row_and_audit_recompute_semantic_labels_and_completeness(
    v2_fixture: dict[str, object],
) -> None:
    receipt = build_shortcut_v2_audit(
        t_new_inputs=v2_fixture["t_new_inputs"],
        d_support_inputs=v2_fixture["d_support_inputs"],
        **v2_fixture["firewall_args"],
    )
    row = receipt.d_support_rows[0]
    positive = row.positives[0]
    forged_positive = replace(
        positive,
        exact_semantic_match=False,
        _factory_token=shortcuts_module._V2_CANDIDATE_FACTORY_TOKEN,
    )
    forged_candidates = tuple(
        forged_positive if value is positive else value for value in row.candidates
    )
    with pytest.raises(GVSShortcutError, match="exactness differs"):
        replace(
            row,
            candidates=forged_candidates,
            _factory_token=shortcuts_module._V2_ROW_FACTORY_TOKEN,
        )

    with pytest.raises(GVSShortcutError, match="blockers differ"):
        replace(
            receipt,
            blockers=tuple(sorted((*receipt.blockers, "forged_binding_absent"))),
            _factory_token=shortcuts_module._V2_AUDIT_FACTORY_TOKEN,
        )


def test_v2_tool_frequency_calibration_weights_components_not_duplicate_rows() -> None:
    tool_a = ("calendar_create",)
    tool_b = ("reminder_create",)
    observations = [("a" * 64, tool_a, tool_b)] * 10 + [("b" * 64, tool_b, tool_a)]

    statistics = shortcuts_module._component_balanced_tool_statistics(observations)

    assert statistics[tool_a][:2] == (10, 1)
    assert statistics[tool_b][:2] == (1, 10)
    assert statistics[tool_a][2:] == (Fraction(1, 2), Fraction(1, 2))
    assert statistics[tool_b][2:] == (Fraction(1, 2), Fraction(1, 2))
