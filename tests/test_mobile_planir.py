from __future__ import annotations

import calendar
import importlib.util
import inspect
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import barunlm.datasets.mobile_planir as mobile_planir_module
from barunlm.datasets.mobile_planir import (
    ACCEPTED_MEMBERSHIP_SHA256,
    ACCEPTED_ROWS,
    CONSTRUCTION_MANIFEST_SHA256,
    CONSTRUCTION_ROWS,
    CONSTRUCTION_USE,
    EXPECTED_CLOCK_REFERENCES,
    EXPECTED_DATE_OPERATOR_COUNTS,
    EXPECTED_REJECTION_CODE,
    EXPECTED_REJECTION_ID,
    PINNED_TOKENIZER_SHA256,
    MobilePlanIRError,
    audit_construction_oracles,
    audit_construction_target_lengths,
    enumerate_grounding_candidates,
)
from barunlm.evaluation.grounded_planir import MOBILE_TOOL_SCHEMAS, compile_mobile_plan
from barunlm.training.data import SFTExample, load_manifest, sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTION_MANIFEST = (
    ROOT
    / "experiments/runs/20260803-2353-mobile-temporal-counterfactual-s17"
    / "materialized-screening-v1/construction-train.jsonl"
)
PINNED_TOKENIZER = ROOT / "experiments/releases/candidate-v2-20260803/staging/float/tokenizer.json"
REQUIRES_CONSTRUCTION = pytest.mark.skipif(
    not CONSTRUCTION_MANIFEST.is_file(),
    reason="private construction-only manifest is absent from this checkout",
)
REQUIRES_CONSTRUCTION_AND_TOKENIZER = pytest.mark.skipif(
    not CONSTRUCTION_MANIFEST.is_file() or not PINNED_TOKENIZER.is_file(),
    reason="private construction manifest or pinned tokenizer is absent from this checkout",
)


def _compile_candidate_pair(
    request: str,
    now: str,
    date_candidate: Any,
    clock_candidate: Any,
):
    plan = {
        "calls": [
            {
                "args": {
                    "datetime": {
                        "date": date_candidate.as_plan_value(),
                        "time": clock_candidate.as_plan_value(),
                    },
                    "title": "Differential probe",
                },
                "tool": "create_calendar_event",
            }
        ],
        "decision": "CALL",
        "mode": "SINGLE",
        "schema_version": "grounded-plan-ir-v1",
    }
    return compile_mobile_plan(
        json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        request,
        now,
        MOBILE_TOOL_SCHEMAS,
    )


def _example_by_id(example_id: str) -> SFTExample:
    examples = load_manifest(
        CONSTRUCTION_MANIFEST,
        expected_sha256=CONSTRUCTION_MANIFEST_SHA256,
        expected_derived_split="train",
    )
    return next(example for example in examples if example.example_id == example_id)


def test_grounding_enumerator_has_no_target_or_gold_input() -> None:
    parameters = inspect.signature(enumerate_grounding_candidates).parameters
    assert tuple(parameters) == ("request", "now")
    assert "derive_oracle" not in mobile_planir_module.__all__


def test_grounding_candidates_cover_frozen_temporal_operators_and_exact_spans() -> None:
    request = (
        "On August 20th, 2026 schedule 2 PM tomorrow; if needed use "
        "August 20th next Tuesday at 14:00."
    )
    candidates = enumerate_grounding_candidates(request, "2026-08-17T09:00:00")

    absolute = [item for item in candidates.dates if item.operator == "ABSOLUTE_DATE"]
    month_day = [item for item in candidates.dates if item.operator == "MONTH_DAY_NEXT"]
    relative = [item for item in candidates.dates if item.operator == "RELATIVE_DAY"]
    weekday = [item for item in candidates.dates if item.operator == "WEEKDAY"]
    assert any(item.ref.quote == "August 20th, 2026" for item in absolute)
    assert {item.ref.occurrence for item in month_day if item.ref.quote == "August 20th"} == {
        0,
        1,
    }
    assert any(item.ref.quote == "tomorrow" for item in relative)
    assert {(item.ref.quote, item.ordinal) for item in weekday} == {
        ("Tuesday", 1),
        ("Tuesday", 2),
    }
    assert any(item.ref.quote == "2 PM" for item in candidates.clocks)
    assert any(item.ref.quote == "14:00" for item in candidates.clocks)


def test_every_enumerated_candidate_differentially_matches_the_quote_v1_compiler() -> None:
    request = (
        "Dates: 2026-08-04; August 5th, 2026; 6th of August, 2026; August 7th; "
        "8th of August; today; tonight; this morning; this afternoon; this evening; "
        "tomorrow; day after tomorrow; in 10 days; 12 days from now; Monday. "
        "Times: 9 AM; 9:07:05 p.m.; 23:59; 00:00:09; nine in the morning; "
        "ten thirty in the morning; 4 in the afternoon; 7 in the evening; "
        "8 in the night; noon; midnight; 9 in the morning."
    )
    now = "2026-08-03T00:00:00"
    candidates = enumerate_grounding_candidates(request, now)

    assert {candidate.operator for candidate in candidates.dates} == {
        "ABSOLUTE_DATE",
        "MONTH_DAY_NEXT",
        "RELATIVE_DAY",
        "WEEKDAY",
    }
    assert {"noon", "midnight", "8 in the night", "ten thirty in the morning"}.issubset(
        {candidate.ref.quote for candidate in candidates.clocks}
    )
    assert candidates.dates and candidates.clocks
    for date_candidate in candidates.dates:
        for clock_candidate in candidates.clocks:
            result = _compile_candidate_pair(request, now, date_candidate, clock_candidate)
            assert result.ok, (
                date_candidate,
                clock_candidate,
                result.error,
            )
            assert result.action_ir is not None
            compiled = json.loads(result.action_ir)
            expected = datetime.combine(
                date_candidate.resolved, clock_candidate.resolved
            ).isoformat(timespec="seconds")
            assert compiled["calls"][0]["args"]["datetime"] == expected


def test_leap_day_and_non_20xx_absolute_candidates_match_compiler_resolution() -> None:
    cases = (
        ("Meet February 29 at noon", "2025-03-01T00:00:00", "2028-02-29T12:00:00"),
        ("Meet 1999-12-31 at noon", "1999-01-01T00:00:00", "1999-12-31T12:00:00"),
    )
    for request, now, expected in cases:
        candidates = enumerate_grounding_candidates(request, now)
        assert candidates.dates and candidates.clocks
        for date_candidate in candidates.dates:
            for clock_candidate in candidates.clocks:
                result = _compile_candidate_pair(request, now, date_candidate, clock_candidate)
                assert result.ok and result.action_ir is not None
                assert json.loads(result.action_ir)["calls"][0]["args"]["datetime"] == expected


def test_enumerator_does_not_emit_lexical_forms_rejected_by_quote_v1_compiler() -> None:
    cases = (
        ("Meet Aug 4th, 2026", "date"),
        ("Meet August 2th, 2026", "date"),
        ("Meet February 30th, 2026", "date"),
        ("Meet 2026-8-04", "date"),
        ("Meet in 367 days", "date"),
        ("Meet 367 days from now", "date"),
        ("Meet at 9 morning", "clock"),
        ("Meet at nine fifteen in the morning", "clock"),
        ("Meet at nine o'clock in the morning", "clock"),
        ("Meet at 9:30 in the morning", "clock"),
        ("Meet at 25 in the morning", "clock"),
        ("Meet at 13 PM", "clock"),
        ("Meet at 24:00", "clock"),
    )
    for request, kind in cases:
        candidates = enumerate_grounding_candidates(request, "2026-08-03T00:00:00")
        observed = candidates.dates if kind == "date" else candidates.clocks
        assert observed == (), (request, observed)


@pytest.mark.parametrize(
    "bad_now",
    [
        "2026-08-03T00:00",
        "2026-8-03T00:00:00",
        "2026-08-03 00:00:00",
        "2026-08-03T00:00:00.000000",
        "2026-08-03T00:00:00+00:00",
        "not-a-date",
        datetime(2026, 8, 3),  # noqa: DTZ001 - deliberately invalid naive object input
        None,
        False,
        7,
    ],
)
def test_enumerator_rejects_every_noncanonical_now_predictably(bad_now: object) -> None:
    with pytest.raises(MobilePlanIRError, match="NOW must be canonical"):
        enumerate_grounding_candidates("Meet tomorrow at noon", bad_now)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_request", [None, False, 7, ["tomorrow"], {"x": "tomorrow"}])
def test_enumerator_rejects_non_string_requests_predictably(bad_request: object) -> None:
    with pytest.raises(MobilePlanIRError, match="request must be a string"):
        enumerate_grounding_candidates(  # type: ignore[arg-type]
            bad_request, "2026-08-03T00:00:00"
        )


def test_enumerator_rejects_non_nfc_and_invalid_utf8_requests_predictably() -> None:
    with pytest.raises(MobilePlanIRError, match="NFC"):
        enumerate_grounding_candidates("Cafe\u0301 tomorrow", "2026-08-03T00:00:00")
    with pytest.raises(MobilePlanIRError, match="valid UTF-8"):
        enumerate_grounding_candidates("\ud800 tomorrow", "2026-08-03T00:00:00")


def test_english_lexicon_is_independent_of_localized_calendar_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calendar, "month_name", ("",) + tuple(f"M{i}" for i in range(1, 13)))
    monkeypatch.setattr(calendar, "day_name", tuple(f"D{i}" for i in range(7)))
    module_name = "barunlm.datasets._mobile_planir_locale_probe"
    spec = importlib.util.spec_from_file_location(module_name, Path(mobile_planir_module.__file__))
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = probe
    try:
        spec.loader.exec_module(probe)
        candidates = probe.enumerate_grounding_candidates(
            "Meet August 4th, 2026 or Monday at noon", "2026-08-03T00:00:00"
        )
    finally:
        sys.modules.pop(module_name, None)

    assert any(candidate.ref.quote == "August 4th, 2026" for candidate in candidates.dates)
    assert any(candidate.ref.quote == "Monday" for candidate in candidates.dates)
    assert any(candidate.ref.quote == "noon" for candidate in candidates.clocks)


@REQUIRES_CONSTRUCTION
def test_construction_manifest_identity_is_training_and_feasibility_only() -> None:
    assert CONSTRUCTION_USE == "training-and-feasibility-only"
    assert sha256_file(CONSTRUCTION_MANIFEST) == CONSTRUCTION_MANIFEST_SHA256
    assert sum(1 for _ in CONSTRUCTION_MANIFEST.open("rb")) == CONSTRUCTION_ROWS


@REQUIRES_CONSTRUCTION
def test_target_is_not_consulted_until_after_target_free_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = _example_by_id("mobile-actions-03338-4e2596cb93faa26b")
    target_consulted = False
    real_enumerator = mobile_planir_module.enumerate_grounding_candidates
    real_target_parser = mobile_planir_module._parse_target

    def tracked_enumerator(request: str, now: str):
        assert not target_consulted
        return real_enumerator(request, now)

    def tracked_target_parser(item: SFTExample):
        nonlocal target_consulted
        target_consulted = True
        return real_target_parser(item)

    monkeypatch.setattr(mobile_planir_module, "enumerate_grounding_candidates", tracked_enumerator)
    monkeypatch.setattr(mobile_planir_module, "_parse_target", tracked_target_parser)

    derivation = mobile_planir_module._derive_oracle(example)
    assert target_consulted
    assert derivation.accepted


@REQUIRES_CONSTRUCTION
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_split", "test"),
        ("derived_split", "evaluation"),
        ("evaluation_role", "confirmation"),
        ("population_role", "selection"),
    ],
)
def test_internal_quote_v1_oracle_rejects_nonconstruction_metadata(field: str, value: str) -> None:
    example = _example_by_id("mobile-actions-03338-4e2596cb93faa26b")
    poisoned = replace(example, metadata={**example.metadata, field: value})
    with pytest.raises(MobilePlanIRError):
        mobile_planir_module._derive_oracle(poisoned)


@REQUIRES_CONSTRUCTION
def test_mobile_actions_03338_keeps_immutable_quote_v1_clock_winner() -> None:
    """New ``noon`` coverage must not silently replace the frozen ``12:00`` target."""

    example = _example_by_id("mobile-actions-03338-4e2596cb93faa26b")
    request, now, _ = mobile_planir_module._parse_prompt(example)
    candidates = enumerate_grounding_candidates(request, now)
    noon = next(candidate for candidate in candidates.clocks if candidate.ref.quote == "noon")
    numeric = next(candidate for candidate in candidates.clocks if candidate.ref.quote == "12:00")
    derivation = mobile_planir_module._derive_oracle(example)

    assert noon.resolved == numeric.resolved
    assert (numeric.compatibility_tier, noon.compatibility_tier) == (0, 1)
    assert derivation.accepted
    assert derivation.selected_clock_ref == numeric.ref


@pytest.mark.parametrize(
    "bad_target",
    ["not-json", "[]", '{"decision":"ABSTAIN"} trailing', '{"x":NaN}'],
)
def test_invalid_oracle_targets_fail_predictably_after_enumeration(bad_target: str) -> None:
    prompt = (
        "<bos><system>\nACTION_IR_V1\nNOW 2026-08-03T00:00:00\n"
        "<user>\nMeet tomorrow at noon\n<assistant>\n"
    )
    example = SFTExample(
        example_id="invalid-target",
        prompt=prompt,
        target=bad_target,
        metadata={
            "adapter_schema_version": "barun-mobile-actions-adapter-v2",
            "dataset": "google/mobile-actions",
            "derived_split": "train",
            "prompt_contract_version": "barun-action-prompt-v1",
            "source_split": "train",
        },
        content_sha256="0" * 64,
    )

    with pytest.raises(MobilePlanIRError):
        mobile_planir_module._derive_oracle(example)


@REQUIRES_CONSTRUCTION
def test_full_construction_oracle_audit_matches_frozen_evidence() -> None:
    audit = audit_construction_oracles(CONSTRUCTION_MANIFEST)

    assert audit.use == "training-and-feasibility-only"
    assert audit.source_rows == CONSTRUCTION_ROWS == 5_745
    assert audit.accepted_rows == ACCEPTED_ROWS == 5_744
    assert audit.rejected_rows == 1
    assert audit.accepted_membership_sha256 == ACCEPTED_MEMBERSHIP_SHA256
    assert (
        audit.date_operator_counts
        == EXPECTED_DATE_OPERATOR_COUNTS
        == {
            "ABSOLUTE_DATE": 1_333,
            "MONTH_DAY_NEXT": 357,
            "RELATIVE_DAY": 138,
            "WEEKDAY": 241,
        }
    )
    assert audit.clock_references == EXPECTED_CLOCK_REFERENCES == 2_070
    assert audit.rejection_codes == {EXPECTED_REJECTION_CODE: 1}
    assert [(item.example_id, item.code) for item in audit.rejections] == [
        (EXPECTED_REJECTION_ID, EXPECTED_REJECTION_CODE)
    ]
    rejection = audit.rejections[0]
    assert rejection.plan is None
    assert rejection.compiled_action_ir is None


@REQUIRES_CONSTRUCTION_AND_TOKENIZER
def test_construction_only_planir_target_token_length_audit() -> None:
    audit = audit_construction_target_lengths(CONSTRUCTION_MANIFEST, PINNED_TOKENIZER)

    assert audit.use == "training-and-feasibility-only"
    assert audit.manifest_sha256 == CONSTRUCTION_MANIFEST_SHA256
    assert audit.tokenizer_sha256 == PINNED_TOKENIZER_SHA256
    assert audit.accepted_membership_sha256 == ACCEPTED_MEMBERSHIP_SHA256
    assert audit.rejection_ids == (EXPECTED_REJECTION_ID,)
    assert (audit.source_rows, audit.compared_rows, audit.rejected_rows) == (5_745, 5_744, 1)
    assert audit.max_sequence_length == 2_048
    assert (
        audit.direct_targets.total,
        audit.direct_targets.median,
        audit.direct_targets.p90,
        audit.direct_targets.maximum,
    ) == (514_097, 94.0, 138, 196)
    assert audit.direct_targets.mean == pytest.approx(89.50156685236769)
    assert (
        audit.planir_targets.total,
        audit.planir_targets.median,
        audit.planir_targets.p90,
        audit.planir_targets.maximum,
    ) == (735_301, 122.0, 203, 267)
    assert audit.planir_targets.mean == pytest.approx(128.01201253481895)
    assert audit.rows_planir_exceeds_direct == 5_744
    assert (audit.maximum_direct_sequence, audit.maximum_planir_sequence) == (660, 679)
    assert audit.direct_sequences_over_limit == 0
    assert audit.planir_sequences_over_limit == 0
