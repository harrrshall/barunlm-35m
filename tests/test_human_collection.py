from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from barunaction.schema import SCHEMA_CONTRACT_VERSION
from barunlm.evaluation.human_collection import (
    FROZEN_SELECTION_GATES,
    HUMAN_COLLECTION_SCHEMA_VERSION,
    LABEL_ACCESS_EVENT_SCHEMA_VERSION,
    LABEL_ENVELOPE_SCHEMA_VERSION,
    SELECTION_RECEIPT_SCHEMA_VERSION,
    TEXT_NORMALIZER_VERSION,
    HumanCollectionError,
    PinnedModelInputRenderer,
    PinnedTokenCounter,
    access_final_confirmation_label,
    audit_population_firewall,
    canonical_sha256,
    prompt_population_sha256,
    seal_label_envelope,
    seal_prompt_record,
    seal_selection_receipt,
    validate_access_ledger,
    validate_label_envelope,
    validate_prompt_record,
    validate_selection_receipt,
)

CUTOFF = "2026-08-03T00:00:00Z"
EXACT_SCAN = "a" * 64
NEAR_SCAN = "b" * 64
SCAN_CODE = "c" * 64
LABEL_KEY = b"L" * 32
RECEIPT_KEY = b"R" * 32
LEDGER_KEY = b"G" * 32


def _byte_counter(text: str) -> int:
    return len(text.encode("utf-8"))


TOKENIZER = PinnedTokenCounter(
    tokenizer_id="harrrshall/BarunLM-35M",
    tokenizer_revision="pinned-revision",
    tokenizer_sha256="d" * 64,
    counter=_byte_counter,
)


def _render_model_input(source: Mapping[str, Any]) -> str:
    return f"NOW {source['reference_timestamp']}\nREQUEST {source['request']}"


MODEL_INPUT_RENDERER = PinnedModelInputRenderer(
    renderer_id="barun-human-test-renderer",
    renderer_revision="renderer-v1",
    renderer_sha256="7" * 64,
    renderer=_render_model_input,
)


def _tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "additional_arguments": False,
            "arguments": {
                "title": {
                    "description": "Calendar event title.",
                    "type": "string",
                }
            },
            "description": "Create a calendar event in the sandbox.",
            "name": "create_calendar_event",
            "required": ["title"],
            "side_effecting": True,
        }
    ]


def _record(
    marker: str = "0",
    *,
    role: str = "selection",
    task_class: str = "efficacy",
    eligible: bool = True,
    request: str | None = None,
    reference_timestamp: str = "2026-08-04T05:30:00+05:30",
    reference_fold: int = 0,
    timezone: str = "Asia/Kolkata",
    authored_at_utc: str = "2026-08-04T00:01:00Z",
) -> dict[str, Any]:
    schemas = _tool_schemas()
    request_text = request or f"Schedule review {marker}"
    raw: dict[str, Any] = {
        "schema_version": HUMAN_COLLECTION_SCHEMA_VERSION,
        "record_id": f"record-{role}-{marker}",
        "cluster_id": f"cluster-{role}-{marker}",
        "role": role,
        "task_class": task_class,
        "provenance": {
            "author_id": f"author-{role}-{marker}",
            "source_id": f"source-{role}-{marker}",
            "collection_batch": f"batch-{role}-{marker}",
            "role_assigned_at_utc": "2026-08-03T23:59:00Z",
            "authored_at_utc": authored_at_utc,
            "license": "CC-BY-4.0",
            "consent": True,
            "no_model_assistance": True,
            "authoring_protocol_revision": "protocol-v2",
            "schema_family": f"schema-{role}-{marker}",
            "paraphrase_family": f"paraphrase-{role}-{marker}",
            "entity_source_ids": [f"entity-{role}-{marker}"],
            "temporal_construction_id": f"temporal-{role}-{marker}",
        },
        "input": {
            "request": request_text,
            "context": {},
            "reference_timestamp": reference_timestamp,
            "reference_fold": reference_fold,
            "timezone": timezone,
            "tool_schemas": schemas,
            "tool_schema_identity": {
                "identifier": "mobile-tools",
                "revision": f"v2-{marker}",
                "contract_version": SCHEMA_CONTRACT_VERSION,
                "sha256": canonical_sha256(schemas),
            },
            "model_input": "filled-by-pinned-renderer",
            "model_input_renderer_identity": {
                "identifier": MODEL_INPUT_RENDERER.renderer_id,
                "revision": MODEL_INPUT_RENDERER.renderer_revision,
                "sha256": MODEL_INPUT_RENDERER.renderer_sha256,
            },
        },
        "eligibility": {
            "eligible": eligible,
            "decision": "include" if eligible else "exclude",
            "reason": None if eligible else "predeclared exclusion",
            "decider_id": f"decider-{role}-{marker}",
            "decided_at_utc": "2026-08-04T00:02:00Z",
        },
        "duplicate_evidence": {
            "input_sha256": "0" * 64,
            "normalized_request_sha256": "0" * 64,
            "delexicalized_template_sha256": canonical_sha256({"delexicalized_template": marker}),
            "near_duplicate_cluster_id": f"near-{role}-{marker}",
            "normalizer_revision": TEXT_NORMALIZER_VERSION,
            "exact_scan_manifest_sha256": EXACT_SCAN,
            "near_scan_manifest_sha256": NEAR_SCAN,
            "scan_code_sha256": SCAN_CODE,
            "scan_completed_at_utc": "2026-08-04T00:01:30Z",
        },
        "tokenization": {
            "tokenizer_id": TOKENIZER.tokenizer_id,
            "tokenizer_revision": TOKENIZER.tokenizer_revision,
            "tokenizer_sha256": TOKENIZER.tokenizer_sha256,
            "input_tokens": 0,
        },
        "integrity": {"input_sha256": "0" * 64, "prompt_record_sha256": "0" * 64},
    }
    return seal_prompt_record(
        raw,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
    )


def _action(title: str = "Review") -> dict[str, Any]:
    return {
        "calls": [{"args": {"title": title}, "tool": "create_calendar_event"}],
        "decision": "CALL",
        "mode": "SINGLE",
    }


def _envelope(record: Mapping[str, Any], *, label: object | None = None) -> dict[str, Any]:
    action = _action(str(record["record_id"])) if label is None else label
    raw: dict[str, Any] = {
        "schema_version": LABEL_ENVELOPE_SCHEMA_VERSION,
        "record_id": "filled-by-sealer",
        "prompt_record_sha256": "0" * 64,
        "annotation": {
            "labeler_id": f"labeler-{record['record_id']}",
            "labeler_no_model_assistance": True,
            "labeler_labeled_at_utc": "2026-08-04T00:03:00Z",
            "labeler_action_ir": copy.deepcopy(action),
            "reviewer_id": f"reviewer-{record['record_id']}",
            "reviewer_no_model_assistance": True,
            "reviewer_labeled_at_utc": "2026-08-04T00:04:00Z",
            "reviewer_action_ir": copy.deepcopy(action),
            "disagreement": False,
            "adjudicator_id": None,
            "adjudicator_no_model_assistance": None,
            "adjudicated_at_utc": None,
            "adjudicated_action_ir": None,
        },
        "tokenization": {
            "tokenizer_id": TOKENIZER.tokenizer_id,
            "tokenizer_revision": TOKENIZER.tokenizer_revision,
            "tokenizer_sha256": TOKENIZER.tokenizer_sha256,
            "output_tokens": 0,
        },
        "sealed_at_utc": "2026-08-04T00:05:00Z",
        "key_id": "label-key-v1",
        "envelope_sha256": "0" * 64,
        "envelope_hmac_sha256": "0" * 64,
    }
    return seal_label_envelope(
        raw,
        prompt_record=record,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
        signing_key=LABEL_KEY,
    )


def _bindings(
    selection: Sequence[Mapping[str, Any]], confirmation: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    return {
        "selection_collection_sha256": prompt_population_sha256(
            selection,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
        ),
        "confirmation_prompt_collection_sha256": prompt_population_sha256(
            confirmation,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
        ),
        "experiment_config_sha256": "e" * 64,
        "candidate_set_sha256": "f" * 64,
        "compiler_sha256": "1" * 64,
        "evaluator_sha256": "2" * 64,
        "source_revision": "source-revision-20260804",
        "code_revision": "code-revision-20260804",
    }


def _receipt(bindings: Mapping[str, str], *, failed_gate: str | None = None) -> dict[str, Any]:
    gates = {name: True for name in FROZEN_SELECTION_GATES}
    if failed_gate is not None:
        gates[failed_gate] = False
    raw: dict[str, Any] = {
        "schema_version": SELECTION_RECEIPT_SCHEMA_VERSION,
        **dict(bindings),
        "gate_results": gates,
        "passed": all(gates.values()),
        "created_at_utc": "2026-08-04T01:00:00Z",
        "key_id": "receipt-key-v1",
        "receipt_hmac_sha256": "0" * 64,
    }
    return seal_selection_receipt(raw, signing_key=RECEIPT_KEY)


def _access(
    record: Mapping[str, Any],
    envelope: Mapping[str, Any],
    receipt: Mapping[str, Any],
    bindings: Mapping[str, str],
    confirmation: Sequence[Mapping[str, Any]],
    *,
    ledger: Sequence[Mapping[str, Any]] = (),
    confirmation_prompt_hashes: Sequence[str] | None = None,
    scoring_session_id: str = "scoring-session-1",
    event_id: str | None = None,
    accessed_at_utc: str = "2026-08-04T02:00:00Z",
    durable_append: Any = None,
) -> dict[str, object]:
    if durable_append is None:
        durable_append = lambda event, expected_head: True
    return access_final_confirmation_label(
        record,
        envelope,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
        label_signing_key=LABEL_KEY,
        expected_label_key_id="label-key-v1",
        selection_receipt=receipt,
        receipt_signing_key=RECEIPT_KEY,
        expected_receipt_key_id="receipt-key-v1",
        expected_receipt_bindings=bindings,
        confirmation_prompt_record_hashes=(
            [item["integrity"]["prompt_record_sha256"] for item in confirmation]
            if confirmation_prompt_hashes is None
            else confirmation_prompt_hashes
        ),
        prior_durable_ledger=ledger,
        ledger_signing_key=LEDGER_KEY,
        expected_ledger_key_id="ledger-key-v1",
        scoring_session_id=scoring_session_id,
        accessor_id="human-eval-scorer",
        purpose="once-only-confirmation-scoring",
        event_id=f"event-{len(ledger)}" if event_id is None else event_id,
        accessed_at_utc=accessed_at_utc,
        durable_append=durable_append,
    )


def test_strict_json_hash_rejects_nested_non_string_keys_and_non_json_types() -> None:
    with pytest.raises(HumanCollectionError, match="keys must be exact strings"):
        canonical_sha256({"nested": {1: "collision"}})
    with pytest.raises(HumanCollectionError, match="non-JSON type tuple"):
        canonical_sha256({"nested": ("not", "json")})
    assert canonical_sha256({"1": "collision"}) == canonical_sha256({"1": "collision"})


def test_public_prompt_record_contains_no_annotation_or_label_data() -> None:
    record = _record()
    validated = validate_prompt_record(
        record,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
    )
    assert validated == record
    assert validated is not record
    assert "annotation" not in record
    assert "access_events" not in record

    def structural_keys(value: object) -> set[str]:
        if type(value) is dict:
            return set(value) | {key for child in value.values() for key in structural_keys(child)}
        if type(value) is list:
            return {key for child in value for key in structural_keys(child)}
        return set()

    assert not {
        key
        for key in structural_keys(record)
        if "label" in key.casefold() or "annotation" in key.casefold()
    }
    with pytest.raises(HumanCollectionError, match="fields must be exact"):
        validate_prompt_record(
            {**record, "label": _action()},
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
        )


def test_model_input_is_recomputed_by_pinned_renderer() -> None:
    record = _record("renderer")
    record["input"]["model_input"] += "\nHIDDEN LABEL: CALL"
    with pytest.raises(HumanCollectionError, match="does not match the pinned renderer"):
        validate_prompt_record(
            record,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
        )


def test_cutoff_is_strict_and_author_attestation_is_required() -> None:
    at_cutoff = _record("cutoff")
    at_cutoff["provenance"]["role_assigned_at_utc"] = "2026-08-02T23:59:00Z"
    at_cutoff["provenance"]["authored_at_utc"] = CUTOFF
    with pytest.raises(HumanCollectionError, match="strictly after compared"):
        seal_prompt_record(
            at_cutoff,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
        )

    record = _record("model-assisted")
    record["provenance"]["no_model_assistance"] = False
    with pytest.raises(HumanCollectionError, match="no_model_assistance"):
        seal_prompt_record(
            record,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
        )


def test_reference_timestamp_rejects_dst_gap_and_accepts_explicit_fold_one() -> None:
    valid_fold = _record(
        "fold",
        reference_timestamp="2026-11-01T01:30:00-05:00",
        reference_fold=1,
        timezone="America/New_York",
    )
    validate_prompt_record(
        valid_fold,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
    )

    with pytest.raises(HumanCollectionError, match="nonexistent local time"):
        _record(
            "gap",
            reference_timestamp="2026-03-08T02:30:00-05:00",
            reference_fold=0,
            timezone="America/New_York",
        )


def test_token_counts_are_recomputed_and_bool_is_not_an_integer() -> None:
    record = _record()
    record["tokenization"]["input_tokens"] += 1
    record["integrity"]["prompt_record_sha256"] = canonical_sha256(
        {
            **record,
            "integrity": {
                **record["integrity"],
                "prompt_record_sha256": "0" * 64,
            },
        }
    )
    with pytest.raises(HumanCollectionError, match="does not match pinned counter"):
        validate_prompt_record(
            record,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
        )

    bool_counter = PinnedTokenCounter(
        tokenizer_id=TOKENIZER.tokenizer_id,
        tokenizer_revision=TOKENIZER.tokenizer_revision,
        tokenizer_sha256=TOKENIZER.tokenizer_sha256,
        counter=lambda text: True,
    )
    with pytest.raises(HumanCollectionError, match="non-negative integer"):
        seal_prompt_record(
            _record(),
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=bool_counter,
            model_input_renderer=MODEL_INPUT_RENDERER,
        )


@pytest.mark.parametrize(
    "bad_label",
    [
        [{"decision": "ABSTAIN"}],
        {"decision": "NONSENSE"},
        {
            "calls": [{"args": {}, "tool": "create_calendar_event"}],
            "decision": "CALL",
            "mode": "SINGLE",
        },
    ],
)
def test_every_human_label_must_be_schema_valid_action_ir(bad_label: object) -> None:
    record = _record()
    with pytest.raises(HumanCollectionError, match="Action IR"):
        _envelope(record, label=bad_label)


def test_labeler_reviewer_and_adjudicator_attestations_are_enforced() -> None:
    record = _record()
    envelope = _envelope(record)
    envelope["annotation"]["reviewer_no_model_assistance"] = False
    with pytest.raises(HumanCollectionError, match="reviewer_no_model_assistance"):
        seal_label_envelope(
            envelope,
            prompt_record=record,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            signing_key=LABEL_KEY,
        )

    disagreement = _envelope(record)
    disagreement["annotation"]["reviewer_action_ir"] = _action("Different")
    disagreement["annotation"]["disagreement"] = True
    disagreement["annotation"]["adjudicator_id"] = "adjudicator-1"
    disagreement["annotation"]["adjudicator_no_model_assistance"] = False
    disagreement["annotation"]["adjudicated_at_utc"] = "2026-08-04T00:05:00Z"
    disagreement["annotation"]["adjudicated_action_ir"] = _action("Final")
    disagreement["sealed_at_utc"] = "2026-08-04T00:06:00Z"
    with pytest.raises(HumanCollectionError, match="adjudicator_no_model_assistance"):
        seal_label_envelope(
            disagreement,
            prompt_record=record,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            signing_key=LABEL_KEY,
        )


def test_label_envelope_hmac_detects_alteration() -> None:
    record = _record()
    envelope = _envelope(record)
    envelope["envelope_hmac_sha256"] = "0" * 64
    with pytest.raises(HumanCollectionError, match="HMAC mismatch"):
        validate_label_envelope(
            envelope,
            prompt_record=record,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            signing_key=LABEL_KEY,
            expected_key_id="label-key-v1",
        )


def test_population_audit_counts_only_eligible_clusters_and_reports_exclusions() -> None:
    assert (
        inspect.signature(audit_population_firewall)
        .parameters["minimum_selection_eligible_clusters"]
        .default
        is inspect.Parameter.empty
    )
    selection = [_record("s", role="selection")]
    confirmation = [
        _record("eff", role="confirmation", task_class="efficacy"),
        _record("safe", role="confirmation", task_class="safety"),
        _record("excluded", role="confirmation", task_class="efficacy", eligible=False),
    ]
    result = audit_population_firewall(
        selection,
        confirmation,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
        expected_exact_scan_manifest_sha256=EXACT_SCAN,
        expected_near_scan_manifest_sha256=NEAR_SCAN,
        expected_scan_code_sha256=SCAN_CODE,
        require_schema_family_separation=True,
        minimum_selection_eligible_clusters=1,
        minimum_confirmation_efficacy_clusters=1,
        minimum_confirmation_safety_clusters=1,
    )
    assert result["confirmation_collected_records"] == 3
    assert result["confirmation_excluded_records"] == 1
    assert result["confirmation_eligible_records"] == 2
    assert result["selection_collected_clusters"] == 1
    assert result["selection_excluded_clusters"] == 0
    assert result["selection_eligible_clusters"] == 1
    assert result["confirmation_collected_clusters"] == 3
    assert result["confirmation_excluded_clusters"] == 1
    assert result["confirmation_eligible_clusters"] == 2
    assert result["confirmation_collected_efficacy_clusters"] == 2
    assert result["confirmation_excluded_efficacy_clusters"] == 1
    assert result["confirmation_eligible_efficacy_clusters"] == 1
    assert result["confirmation_collected_safety_clusters"] == 1
    assert result["confirmation_excluded_safety_clusters"] == 0
    assert result["confirmation_eligible_safety_clusters"] == 1


def test_all_excluded_population_cannot_pass_even_with_collected_clusters() -> None:
    selection = [_record("s", role="selection")]
    confirmation = [
        _record("eff", role="confirmation", task_class="efficacy", eligible=False),
        _record("safe", role="confirmation", task_class="safety", eligible=False),
    ]
    with pytest.raises(HumanCollectionError, match="eligible efficacy clusters below minimum"):
        audit_population_firewall(
            selection,
            confirmation,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            expected_exact_scan_manifest_sha256=EXACT_SCAN,
            expected_near_scan_manifest_sha256=NEAR_SCAN,
            expected_scan_code_sha256=SCAN_CODE,
            require_schema_family_separation=True,
            minimum_selection_eligible_clusters=1,
            minimum_confirmation_efficacy_clusters=1,
            minimum_confirmation_safety_clusters=1,
        )


def test_all_excluded_selection_population_cannot_pass() -> None:
    selection = [_record("s", role="selection", eligible=False)]
    confirmation = [
        _record("eff", role="confirmation", task_class="efficacy"),
        _record("safe", role="confirmation", task_class="safety"),
    ]
    with pytest.raises(HumanCollectionError, match="selection eligible clusters below minimum"):
        audit_population_firewall(
            selection,
            confirmation,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            expected_exact_scan_manifest_sha256=EXACT_SCAN,
            expected_near_scan_manifest_sha256=NEAR_SCAN,
            expected_scan_code_sha256=SCAN_CODE,
            require_schema_family_separation=True,
            minimum_selection_eligible_clusters=1,
            minimum_confirmation_efficacy_clusters=1,
            minimum_confirmation_safety_clusters=1,
        )


def test_duplicate_fingerprints_and_unpinned_scan_evidence_fail_closed() -> None:
    selection = [_record("s", role="selection", request="Same normalized request")]
    confirmation = [
        _record(
            "eff",
            role="confirmation",
            task_class="efficacy",
            request="  SAME   normalized request ",
        ),
        _record("safe", role="confirmation", task_class="safety"),
    ]
    with pytest.raises(HumanCollectionError, match="normalized_request_sha256 overlap"):
        audit_population_firewall(
            selection,
            confirmation,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            expected_exact_scan_manifest_sha256=EXACT_SCAN,
            expected_near_scan_manifest_sha256=NEAR_SCAN,
            expected_scan_code_sha256=SCAN_CODE,
            require_schema_family_separation=True,
            minimum_selection_eligible_clusters=1,
            minimum_confirmation_efficacy_clusters=1,
            minimum_confirmation_safety_clusters=1,
        )

    with pytest.raises(HumanCollectionError, match="different exact-scan manifest"):
        audit_population_firewall(
            [_record("other", role="selection")],
            confirmation,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            expected_exact_scan_manifest_sha256="9" * 64,
            expected_near_scan_manifest_sha256=NEAR_SCAN,
            expected_scan_code_sha256=SCAN_CODE,
            require_schema_family_separation=True,
            minimum_selection_eligible_clusters=1,
            minimum_confirmation_efficacy_clusters=1,
            minimum_confirmation_safety_clusters=1,
        )


def test_exact_input_sha256_duplicate_is_rejected_before_component_counting() -> None:
    selection = [_record("exact-input-selection", role="selection")]
    first = _record("exact-input-first", role="confirmation", task_class="efficacy")
    second = _record("exact-input-second", role="confirmation", task_class="efficacy")
    second["input"] = copy.deepcopy(first["input"])
    second = seal_prompt_record(
        second,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
    )
    assert (
        first["duplicate_evidence"]["input_sha256"] == second["duplicate_evidence"]["input_sha256"]
    )

    with pytest.raises(HumanCollectionError, match="exact duplicate inputs"):
        audit_population_firewall(
            selection,
            [
                first,
                second,
                _record("exact-input-safety", role="confirmation", task_class="safety"),
            ],
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            expected_exact_scan_manifest_sha256=EXACT_SCAN,
            expected_near_scan_manifest_sha256=NEAR_SCAN,
            expected_scan_code_sha256=SCAN_CODE,
            require_schema_family_separation=True,
            minimum_selection_eligible_clusters=1,
            minimum_confirmation_efficacy_clusters=2,
            minimum_confirmation_safety_clusters=1,
        )


def test_schema_family_is_not_a_separation_or_cluster_key_when_disabled() -> None:
    shared_family = "schema-family-shared-by-policy"
    selection_record = _record("schema-disabled-selection", role="selection")
    selection_record["provenance"]["schema_family"] = shared_family
    selection_record = seal_prompt_record(
        selection_record,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
    )
    confirmation = [
        _record("schema-disabled-first", role="confirmation", task_class="efficacy"),
        _record("schema-disabled-second", role="confirmation", task_class="efficacy"),
        _record("schema-disabled-safety", role="confirmation", task_class="safety"),
    ]
    for index, record in enumerate(confirmation):
        record["provenance"]["schema_family"] = shared_family
        confirmation[index] = seal_prompt_record(
            record,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
        )

    result = audit_population_firewall(
        [selection_record],
        confirmation,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
        expected_exact_scan_manifest_sha256=EXACT_SCAN,
        expected_near_scan_manifest_sha256=NEAR_SCAN,
        expected_scan_code_sha256=SCAN_CODE,
        require_schema_family_separation=False,
        minimum_selection_eligible_clusters=1,
        minimum_confirmation_efficacy_clusters=2,
        minimum_confirmation_safety_clusters=1,
    )
    assert result["selection_eligible_clusters"] == 1
    assert result["confirmation_eligible_efficacy_clusters"] == 2
    assert result["confirmation_eligible_safety_clusters"] == 1


def test_receipt_requires_exact_gate_set_hmac_and_all_bindings() -> None:
    selection = [_record("s", role="selection")]
    confirmation = [_record("c", role="confirmation")]
    bindings = _bindings(selection, confirmation)
    receipt = _receipt(bindings)
    validated = validate_selection_receipt(
        receipt,
        signing_key=RECEIPT_KEY,
        expected_key_id="receipt-key-v1",
        expected_bindings=bindings,
    )
    assert validated == receipt
    assert validated is not receipt

    wrong_gates = copy.deepcopy(receipt)
    wrong_gates["gate_results"].pop("overall_vs_a_ge_3")
    wrong_gates["gate_results"]["invented_gate"] = True
    with pytest.raises(HumanCollectionError, match="exact frozen gate set"):
        seal_selection_receipt(wrong_gates, signing_key=RECEIPT_KEY)

    altered = copy.deepcopy(receipt)
    altered["compiler_sha256"] = "9" * 64
    altered_bindings = {**bindings, "compiler_sha256": "9" * 64}
    with pytest.raises(HumanCollectionError, match="HMAC mismatch"):
        validate_selection_receipt(
            altered,
            signing_key=RECEIPT_KEY,
            expected_key_id="receipt-key-v1",
            expected_bindings=altered_bindings,
        )

    forged = copy.deepcopy(receipt)
    forged["receipt_hmac_sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "receipt_hmac_sha256"}
    )
    with pytest.raises(HumanCollectionError, match="HMAC mismatch"):
        validate_selection_receipt(
            forged,
            signing_key=RECEIPT_KEY,
            expected_key_id="receipt-key-v1",
            expected_bindings=bindings,
        )

    wrong_binding = {**bindings, "candidate_set_sha256": "8" * 64}
    with pytest.raises(HumanCollectionError, match="different candidate_set_sha256"):
        validate_selection_receipt(
            receipt,
            signing_key=RECEIPT_KEY,
            expected_key_id="receipt-key-v1",
            expected_bindings=wrong_binding,
        )


def test_failed_selection_receipt_never_unlocks_confirmation() -> None:
    selection = [_record("s", role="selection")]
    record = _record("c", role="confirmation")
    confirmation = [record]
    bindings = _bindings(selection, confirmation)
    receipt = _receipt(bindings, failed_gate="overall_vs_a_ge_3")
    with pytest.raises(HumanCollectionError, match="failed gate"):
        _access(record, _envelope(record), receipt, bindings, confirmation)


def test_access_commits_chain_event_then_returns_only_final_label() -> None:
    selection = [_record("s", role="selection")]
    record = _record("c", role="confirmation")
    confirmation = [record]
    bindings = _bindings(selection, confirmation)
    receipt = _receipt(bindings)
    envelope = _envelope(record)
    committed: list[dict[str, Any]] = []

    def append(event: Mapping[str, Any], expected_head: str) -> bool:
        assert expected_head == "0" * 64
        committed.append(copy.deepcopy(dict(event)))
        return True

    result = _access(
        record,
        envelope,
        receipt,
        bindings,
        confirmation,
        durable_append=append,
    )
    assert set(result) == {"final_action_ir", "access_event"}
    assert result["final_action_ir"] == envelope["annotation"]["labeler_action_ir"]
    assert "labeler_action_ir" not in result
    assert len(committed) == 1
    assert committed[0]["schema_version"] == LABEL_ACCESS_EVENT_SCHEMA_VERSION
    assert (
        validate_access_ledger(
            committed,
            signing_key=LEDGER_KEY,
            expected_key_id="ledger-key-v1",
        )
        == committed[0]["event_sha256"]
    )


def test_repeat_access_is_rejected_even_under_another_scoring_session() -> None:
    selection = [_record("s", role="selection")]
    record = _record("c", role="confirmation")
    confirmation = [record]
    bindings = _bindings(selection, confirmation)
    receipt = _receipt(bindings)
    envelope = _envelope(record)
    first = _access(record, envelope, receipt, bindings, confirmation)
    ledger = [first["access_event"]]
    with pytest.raises(HumanCollectionError, match="already disclosed"):
        _access(
            record,
            {},  # Retirement is checked before the private envelope is read again.
            receipt,
            bindings,
            confirmation,
            ledger=ledger,
            scoring_session_id="different-session",
        )


def test_access_ledger_tampering_and_false_durable_append_fail_closed() -> None:
    selection = [_record("s", role="selection")]
    record = _record("c", role="confirmation")
    confirmation = [record]
    bindings = _bindings(selection, confirmation)
    receipt = _receipt(bindings)
    envelope = _envelope(record)
    first = _access(record, envelope, receipt, bindings, confirmation)
    tampered = copy.deepcopy(first["access_event"])
    tampered["purpose"] = "changed-after-signing"
    with pytest.raises(HumanCollectionError, match="event hash mismatch"):
        validate_access_ledger(
            [tampered],
            signing_key=LEDGER_KEY,
            expected_key_id="ledger-key-v1",
        )

    tampered_mac = copy.deepcopy(first["access_event"])
    tampered_mac["event_hmac_sha256"] = "0" * 64
    with pytest.raises(HumanCollectionError, match="HMAC mismatch"):
        validate_access_ledger(
            [tampered_mac],
            signing_key=LEDGER_KEY,
            expected_key_id="ledger-key-v1",
        )

    with pytest.raises(HumanCollectionError, match="not atomically committed"):
        _access(
            record,
            envelope,
            receipt,
            bindings,
            confirmation,
            durable_append=lambda event, expected_head: False,
        )


def test_confirmation_membership_and_receipt_time_are_enforced() -> None:
    selection = [_record("s", role="selection")]
    record = _record("c", role="confirmation")
    other = _record("other", role="confirmation")
    confirmation = [other]
    bindings = _bindings(selection, confirmation)
    receipt = _receipt(bindings)
    envelope = _envelope(record)
    with pytest.raises(HumanCollectionError, match="not in the receipt-bound"):
        access_final_confirmation_label(
            record,
            envelope,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            label_signing_key=LABEL_KEY,
            expected_label_key_id="label-key-v1",
            selection_receipt=receipt,
            receipt_signing_key=RECEIPT_KEY,
            expected_receipt_key_id="receipt-key-v1",
            expected_receipt_bindings=bindings,
            confirmation_prompt_record_hashes=[
                other["integrity"]["prompt_record_sha256"],
            ],
            prior_durable_ledger=[],
            ledger_signing_key=LEDGER_KEY,
            expected_ledger_key_id="ledger-key-v1",
            scoring_session_id="session",
            accessor_id="scorer",
            purpose="score",
            event_id="event",
            accessed_at_utc="2026-08-04T02:00:00Z",
            durable_append=lambda event, expected_head: True,
        )

    bound_confirmation = [record]
    bound_bindings = _bindings(selection, bound_confirmation)
    late_receipt = _receipt(bound_bindings)
    late_receipt["created_at_utc"] = "2026-08-04T03:00:00Z"
    late_receipt = seal_selection_receipt(late_receipt, signing_key=RECEIPT_KEY)
    with pytest.raises(HumanCollectionError, match="strictly after the passing selection receipt"):
        _access(
            record,
            envelope,
            late_receipt,
            bound_bindings,
            bound_confirmation,
        )


def test_validators_return_detached_snapshots() -> None:
    record = _record("detached", role="confirmation")
    checked_record = validate_prompt_record(
        record,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
    )
    original_request = checked_record["input"]["request"]
    record["input"]["request"] = "mutated after validation"
    assert checked_record["input"]["request"] == original_request

    clean_record = _record("detached-envelope", role="confirmation")
    envelope = _envelope(clean_record)
    checked_envelope = validate_label_envelope(
        envelope,
        prompt_record=clean_record,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
        signing_key=LABEL_KEY,
        expected_key_id="label-key-v1",
    )
    original_title = checked_envelope["annotation"]["labeler_action_ir"]["calls"][0]["args"][
        "title"
    ]
    envelope["annotation"]["labeler_action_ir"]["calls"][0]["args"]["title"] = "mutated"
    assert (
        checked_envelope["annotation"]["labeler_action_ir"]["calls"][0]["args"]["title"]
        == original_title
    )

    selection = [_record("detached-selection", role="selection")]
    confirmation = [clean_record]
    bindings = _bindings(selection, confirmation)
    receipt = _receipt(bindings)
    checked_receipt = validate_selection_receipt(
        receipt,
        signing_key=RECEIPT_KEY,
        expected_key_id="receipt-key-v1",
        expected_bindings=bindings,
    )
    receipt["gate_results"]["overall_vs_a_ge_3"] = False
    assert checked_receipt["gate_results"]["overall_vs_a_ge_3"] is True


def test_custom_sequences_are_rejected_before_authorization() -> None:
    class ChangingSequence(Sequence[Any]):
        def __init__(self, values: list[Any]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> Any:
            return self.values[index]

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self.values)

        def __contains__(self, value: object) -> bool:
            return True

    with pytest.raises(HumanCollectionError, match="exact list or tuple"):
        prompt_population_sha256(
            ChangingSequence([_record("sequence", role="selection")]),
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
        )
    with pytest.raises(HumanCollectionError, match="exact list or tuple"):
        validate_access_ledger(
            ChangingSequence([]),
            signing_key=LEDGER_KEY,
            expected_key_id="ledger-key-v1",
        )

    selection = [_record("sequence-selection", role="selection")]
    record = _record("sequence-confirmation", role="confirmation")
    confirmation = [record]
    bindings = _bindings(selection, confirmation)
    receipt = _receipt(bindings)
    envelope = _envelope(record)
    with pytest.raises(HumanCollectionError, match="exact list or tuple"):
        _access(
            record,
            envelope,
            receipt,
            bindings,
            confirmation,
            confirmation_prompt_hashes=ChangingSequence(
                [record["integrity"]["prompt_record_sha256"]]
            ),
        )

    first = _access(record, envelope, receipt, bindings, confirmation)
    with pytest.raises(HumanCollectionError, match="exact list or tuple"):
        _access(
            record,
            envelope,
            receipt,
            bindings,
            confirmation,
            ledger=ChangingSequence([first["access_event"]]),
            scoring_session_id="sequence-session-2",
        )


def test_effective_clusters_cannot_be_inflated_by_renaming_cluster_id() -> None:
    selection = [_record("cluster-selection", role="selection")]
    first = _record("cluster-first", role="confirmation", task_class="efficacy")
    second = _record("cluster-second", role="confirmation", task_class="efficacy")
    second["provenance"] = copy.deepcopy(first["provenance"])
    second["input"]["request"] = "A distinct surface request in the same semantic lineage"
    second["cluster_id"] = "renamed-cluster-that-must-not-count"
    second = seal_prompt_record(
        second,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
    )
    safety = _record("cluster-safety", role="confirmation", task_class="safety")

    with pytest.raises(HumanCollectionError, match="eligible efficacy clusters below minimum"):
        audit_population_firewall(
            selection,
            [first, second, safety],
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            expected_exact_scan_manifest_sha256=EXACT_SCAN,
            expected_near_scan_manifest_sha256=NEAR_SCAN,
            expected_scan_code_sha256=SCAN_CODE,
            require_schema_family_separation=True,
            minimum_selection_eligible_clusters=1,
            minimum_confirmation_efficacy_clusters=2,
            minimum_confirmation_safety_clusters=1,
        )


def test_duplicate_scan_precedes_eligibility_and_receipt_follows_frozen_labels() -> None:
    bad_prompt = _record("late-scan", role="confirmation")
    bad_prompt["duplicate_evidence"]["scan_completed_at_utc"] = "2026-08-04T00:10:00Z"
    with pytest.raises(HumanCollectionError, match="strictly after duplicate scanning completes"):
        seal_prompt_record(
            bad_prompt,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
        )

    selection = [_record("chronology-selection", role="selection")]
    record = _record("chronology-confirmation", role="confirmation")
    confirmation = [record]
    bindings = _bindings(selection, confirmation)
    envelope = _envelope(record)

    early_receipt = _receipt(bindings)
    early_receipt["created_at_utc"] = "2020-01-01T00:00:00Z"
    early_receipt = seal_selection_receipt(early_receipt, signing_key=RECEIPT_KEY)
    with pytest.raises(HumanCollectionError, match="strictly postdate confirmation prompt"):
        _access(record, envelope, early_receipt, bindings, confirmation)

    late_envelope = copy.deepcopy(envelope)
    late_envelope["sealed_at_utc"] = "2026-08-04T01:30:00Z"
    late_envelope = seal_label_envelope(
        late_envelope,
        prompt_record=record,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
        signing_key=LABEL_KEY,
    )
    with pytest.raises(HumanCollectionError, match="strictly postdate the frozen confirmation"):
        _access(record, late_envelope, _receipt(bindings), bindings, confirmation)


def test_every_chronology_boundary_is_strict() -> None:
    scan_equals_authoring = _record("scan-equals-authoring", role="confirmation")
    scan_equals_authoring["duplicate_evidence"]["scan_completed_at_utc"] = scan_equals_authoring[
        "provenance"
    ]["authored_at_utc"]
    with pytest.raises(HumanCollectionError, match="strictly after authoring"):
        seal_prompt_record(
            scan_equals_authoring,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
        )

    eligibility_equals_scan = _record("eligibility-equals-scan", role="confirmation")
    eligibility_equals_scan["eligibility"]["decided_at_utc"] = eligibility_equals_scan[
        "duplicate_evidence"
    ]["scan_completed_at_utc"]
    with pytest.raises(HumanCollectionError, match="strictly after duplicate scanning"):
        seal_prompt_record(
            eligibility_equals_scan,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
        )

    record = _record("strict-chain", role="confirmation")
    label_equals_eligibility = _envelope(record)
    label_equals_eligibility["annotation"]["labeler_labeled_at_utc"] = record["eligibility"][
        "decided_at_utc"
    ]
    with pytest.raises(HumanCollectionError, match="strictly before labeling"):
        seal_label_envelope(
            label_equals_eligibility,
            prompt_record=record,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            signing_key=LABEL_KEY,
        )

    seal_equals_label = _envelope(record)
    seal_equals_label["sealed_at_utc"] = seal_equals_label["annotation"]["reviewer_labeled_at_utc"]
    with pytest.raises(HumanCollectionError, match="strictly after final label"):
        seal_label_envelope(
            seal_equals_label,
            prompt_record=record,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            signing_key=LABEL_KEY,
        )

    selection = [_record("strict-selection", role="selection")]
    confirmation = [record]
    bindings = _bindings(selection, confirmation)
    envelope = _envelope(record)

    receipt_equals_envelope = _receipt(bindings)
    receipt_equals_envelope["created_at_utc"] = envelope["sealed_at_utc"]
    receipt_equals_envelope = seal_selection_receipt(
        receipt_equals_envelope, signing_key=RECEIPT_KEY
    )
    with pytest.raises(HumanCollectionError, match="strictly postdate the frozen confirmation"):
        _access(record, envelope, receipt_equals_envelope, bindings, confirmation)

    receipt = _receipt(bindings)
    with pytest.raises(HumanCollectionError, match="strictly after the passing selection"):
        _access(
            record,
            envelope,
            receipt,
            bindings,
            confirmation,
            accessed_at_utc=receipt["created_at_utc"],
        )


def test_adjudication_timestamp_must_strictly_follow_the_later_label() -> None:
    record = _record("adjudication-equals-later-label", role="confirmation")
    envelope = _envelope(record)
    envelope["annotation"]["reviewer_action_ir"] = _action("Different")
    envelope["annotation"]["disagreement"] = True
    envelope["annotation"]["adjudicator_id"] = "independent-adjudicator"
    envelope["annotation"]["adjudicator_no_model_assistance"] = True
    envelope["annotation"]["adjudicated_at_utc"] = envelope["annotation"]["reviewer_labeled_at_utc"]
    envelope["annotation"]["adjudicated_action_ir"] = _action("Final")

    with pytest.raises(HumanCollectionError, match="strictly after both independent labels"):
        seal_label_envelope(
            envelope,
            prompt_record=record,
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            signing_key=LABEL_KEY,
        )


@pytest.mark.parametrize(
    "lineage",
    [
        "cluster_id",
        "author_id",
        "source_id",
        "collection_batch",
        "paraphrase_family",
        "entity_source_ids",
        "temporal_construction_id",
        "delexicalized_template_sha256",
        "near_duplicate_cluster_id",
        "schema_family",
        "normalized_request_sha256",
    ],
)
def test_each_documented_lineage_independently_merges_effective_clusters(lineage: str) -> None:
    selection = [_record(f"lineage-selection-{lineage}", role="selection")]
    first = _record(f"lineage-first-{lineage}", role="confirmation", task_class="efficacy")
    second = _record(f"lineage-second-{lineage}", role="confirmation", task_class="efficacy")

    if lineage in {
        "author_id",
        "source_id",
        "collection_batch",
        "paraphrase_family",
        "entity_source_ids",
        "temporal_construction_id",
        "schema_family",
    }:
        second["provenance"][lineage] = copy.deepcopy(first["provenance"][lineage])
    elif lineage in {"delexicalized_template_sha256", "near_duplicate_cluster_id"}:
        second["duplicate_evidence"][lineage] = first["duplicate_evidence"][lineage]
    elif lineage == "normalized_request_sha256":
        second["input"]["request"] = f"  SCHEDULE   REVIEW LINEAGE-FIRST-{lineage.upper()}  "
    else:
        second["cluster_id"] = first["cluster_id"]
    second = seal_prompt_record(
        second,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
    )
    safety = _record(f"lineage-safety-{lineage}", role="confirmation", task_class="safety")

    with pytest.raises(HumanCollectionError, match="eligible efficacy clusters below minimum"):
        audit_population_firewall(
            selection,
            [first, second, safety],
            compared_release_cutoff_utc=CUTOFF,
            tokenizer=TOKENIZER,
            model_input_renderer=MODEL_INPUT_RENDERER,
            expected_exact_scan_manifest_sha256=EXACT_SCAN,
            expected_near_scan_manifest_sha256=NEAR_SCAN,
            expected_scan_code_sha256=SCAN_CODE,
            require_schema_family_separation=True,
            minimum_selection_eligible_clusters=1,
            minimum_confirmation_efficacy_clusters=2,
            minimum_confirmation_safety_clusters=1,
        )


def test_effective_cluster_union_is_transitive_across_distinct_lineage_bridges() -> None:
    selection = [_record("bridge-selection", role="selection")]
    first = _record("bridge-first", role="confirmation", task_class="efficacy")
    middle = _record("bridge-middle", role="confirmation", task_class="efficacy")
    last = _record("bridge-last", role="confirmation", task_class="efficacy")

    middle["provenance"]["source_id"] = first["provenance"]["source_id"]
    last["provenance"]["collection_batch"] = middle["provenance"]["collection_batch"]
    middle = seal_prompt_record(
        middle,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
    )
    last = seal_prompt_record(
        last,
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
    )
    assert first["provenance"]["source_id"] != last["provenance"]["source_id"]
    assert first["provenance"]["collection_batch"] != last["provenance"]["collection_batch"]

    result = audit_population_firewall(
        selection,
        [first, middle, last, _record("bridge-safety", role="confirmation", task_class="safety")],
        compared_release_cutoff_utc=CUTOFF,
        tokenizer=TOKENIZER,
        model_input_renderer=MODEL_INPUT_RENDERER,
        expected_exact_scan_manifest_sha256=EXACT_SCAN,
        expected_near_scan_manifest_sha256=NEAR_SCAN,
        expected_scan_code_sha256=SCAN_CODE,
        require_schema_family_separation=True,
        minimum_selection_eligible_clusters=1,
        minimum_confirmation_efficacy_clusters=1,
        minimum_confirmation_safety_clusters=1,
    )
    assert result["confirmation_collected_efficacy_clusters"] == 1
    assert result["confirmation_eligible_efficacy_clusters"] == 1


def test_known_answer_hash_and_receipt_hmac_vectors() -> None:
    assert canonical_sha256({"b": 2, "a": "é"}) == (
        "06c264c46ad5ada9493abd3aa2383fb205ae99d7d0bad40b03a43bfec8a1b8de"
    )
    selection = [_record("vector-s", role="selection")]
    confirmation = [_record("vector-c", role="confirmation")]
    receipt = _receipt(_bindings(selection, confirmation))
    assert receipt["receipt_hmac_sha256"] == (
        "6151d22fa7ba80fe995f33bfe2415f1f2df5a4b2ad7ceb2acdd9f96c7efb6059"
    )


def test_valid_multi_event_chain_and_callback_mutation_detachment() -> None:
    selection = [_record("chain-selection", role="selection")]
    first_record = _record("chain-first", role="confirmation")
    second_record = _record("chain-second", role="confirmation")
    confirmation = [first_record, second_record]
    bindings = _bindings(selection, confirmation)
    receipt = _receipt(bindings)
    first = _access(
        first_record,
        _envelope(first_record),
        receipt,
        bindings,
        confirmation,
        accessed_at_utc="2026-08-04T02:00:00Z",
    )
    second = _access(
        second_record,
        _envelope(second_record),
        receipt,
        bindings,
        confirmation,
        ledger=[first["access_event"]],
        accessed_at_utc="2026-08-04T02:00:01Z",
    )
    ledger = [first["access_event"], second["access_event"]]
    assert (
        validate_access_ledger(
            ledger,
            signing_key=LEDGER_KEY,
            expected_key_id="ledger-key-v1",
        )
        == second["access_event"]["event_sha256"]
    )

    callback_record = _record("callback-copy", role="confirmation")
    callback_confirmation = [callback_record]
    callback_bindings = _bindings(selection, callback_confirmation)

    def mutate_callback_copy(event: Mapping[str, Any], expected_head: str) -> bool:
        assert expected_head == "0" * 64
        event["purpose"] = "mutated-callback-copy"  # type: ignore[index]
        return True

    result = _access(
        callback_record,
        _envelope(callback_record),
        _receipt(callback_bindings),
        callback_bindings,
        callback_confirmation,
        durable_append=mutate_callback_copy,
    )
    assert result["access_event"]["purpose"] == "once-only-confirmation-scoring"
    validate_access_ledger(
        [result["access_event"]],
        signing_key=LEDGER_KEY,
        expected_key_id="ledger-key-v1",
    )


def test_successive_ledger_event_timestamp_must_strictly_increase() -> None:
    selection = [_record("equal-ledger-selection", role="selection")]
    first_record = _record("equal-ledger-first", role="confirmation")
    second_record = _record("equal-ledger-second", role="confirmation")
    confirmation = [first_record, second_record]
    bindings = _bindings(selection, confirmation)
    receipt = _receipt(bindings)
    accessed_at = "2026-08-04T02:00:00Z"
    first = _access(
        first_record,
        _envelope(first_record),
        receipt,
        bindings,
        confirmation,
        accessed_at_utc=accessed_at,
    )

    with pytest.raises(HumanCollectionError, match="strictly postdate the prior event"):
        _access(
            second_record,
            _envelope(second_record),
            receipt,
            bindings,
            confirmation,
            ledger=[first["access_event"]],
            accessed_at_utc=accessed_at,
        )
