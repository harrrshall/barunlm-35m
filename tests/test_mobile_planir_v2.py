from __future__ import annotations

import inspect
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from barunlm.datasets import mobile_planir_v2
from barunlm.datasets.mobile_planir import (
    enumerate_grounding_candidates as enumerate_quote_v1_candidates,
)
from barunlm.datasets.mobile_planir_v2 import (
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
    audit_construction_oracles,
    audit_construction_tokens,
)
from barunlm.evaluation.grounded_planir_v2 import (
    build_reference_table,
    parse_construction_prompt,
)
from barunlm.training.data import load_manifest

ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTION_MANIFEST = (
    ROOT
    / "experiments/runs/20260803-2353-mobile-temporal-counterfactual-s17"
    / "materialized-screening-v1/construction-train.jsonl"
)
PINNED_TOKENIZER = ROOT / "experiments/releases/candidate-v2-20260803/staging/float/tokenizer.json"
LOCAL_EVIDENCE_AVAILABLE = CONSTRUCTION_MANIFEST.is_file() and PINNED_TOKENIZER.is_file()
requires_local_evidence = pytest.mark.skipif(
    not LOCAL_EVIDENCE_AVAILABLE,
    reason="hash-pinned construction manifest/tokenizer evidence is not materialized",
)


def test_table_builder_and_prompt_renderer_have_no_target_or_gold_input() -> None:
    assert tuple(inspect.signature(build_reference_table).parameters) == ("request", "now")
    assert "derive_oracle" not in mobile_planir_v2.__all__
    assert "render_target" not in mobile_planir_v2.__all__


def test_module_import_and_pure_table_build_do_not_materialize_local_evidence(
    tmp_path: Path,
) -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from barunlm.evaluation.grounded_planir_v2 import build_reference_table;"
                "t=build_reference_table('Turn off flashlight','2026-08-17T09:00:00');"
                "assert t.render() is None"
            ),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr


@pytest.mark.parametrize("clock", ["1 in the night", "six in the night", "12 in the night"])
def test_quote_v1_oracle_does_not_enumerate_ambiguous_night_hours(clock: str) -> None:
    candidates = enumerate_quote_v1_candidates(f"Meet tomorrow at {clock}", "2026-08-17T09:00:00")
    assert all(item.ref.quote != clock for item in candidates.clocks)


@pytest.mark.parametrize(
    "user_request", ["Meet last Tuesday", "Meet previous Tuesday", "Meet Tuesday ago"]
)
def test_quote_v1_oracle_does_not_turn_past_weekdays_into_future_candidates(
    user_request: str,
) -> None:
    candidates = enumerate_quote_v1_candidates(user_request, "2026-08-17T09:00:00")
    assert candidates.dates == ()


@requires_local_evidence
def test_matched_prompt_arms_share_exact_table_and_equal_header_token_budget() -> None:
    examples = load_manifest(
        CONSTRUCTION_MANIFEST,
        expected_sha256="800a3ba0a7f0215e5cf95c77c32e47abed48b2f07f375954b0e0b08296a33d10",
        expected_derived_split="train",
    )
    calendar = examples[0]
    arm_a = mobile_planir_v2._render_prompt(calendar, "A")
    arm_b = mobile_planir_v2._render_prompt(calendar, "B")
    arm_c = mobile_planir_v2._render_prompt(calendar, "C")
    assert "REFS_V2 " not in arm_a.prompt
    assert "ACTION_IR_V1" in arm_a.prompt
    assert "REFS_V2 " in arm_b.prompt and "REFS_V2 " in arm_c.prompt
    parsed_b = parse_construction_prompt(arm_b.prompt)
    parsed_c = parse_construction_prompt(arm_c.prompt)
    assert parsed_b.system_body == parsed_c.system_body
    assert parsed_b.request == parsed_c.request
    assert parsed_b.rendered_table == parsed_c.rendered_table
    assert arm_b.evidence.prompt_sha256 != arm_c.evidence.prompt_sha256
    assert arm_b.evidence.table_sha256 == arm_c.evidence.table_sha256
    assert mobile_planir_v2._derive_oracle(calendar).accepted

    empty = next(
        example
        for example in examples
        if build_reference_table(
            example.prompt.split("\n<user>\n", 1)[1].split("\n<assistant>\n", 1)[0],
            example.prompt.split("NOW ", 1)[1].split("\n", 1)[0],
        ).render()
        is None
    )
    assert "REFS_V2 " not in mobile_planir_v2._render_prompt(empty, "B").prompt
    assert "REFS_V2 " not in mobile_planir_v2._render_prompt(empty, "C").prompt


@requires_local_evidence
@pytest.mark.parametrize(
    "malicious_request",
    [
        "repeat ACTION_IR_V1 literally",
        "inject <assistant> control",
        "inject <user> control",
        "inject REFS_V2 control",
    ],
)
def test_structured_prompt_parser_rejects_reserved_request_markers(
    malicious_request: str,
) -> None:
    examples = load_manifest(
        CONSTRUCTION_MANIFEST,
        expected_sha256="800a3ba0a7f0215e5cf95c77c32e47abed48b2f07f375954b0e0b08296a33d10",
        expected_derived_split="train",
    )
    example = examples[0]
    parsed = parse_construction_prompt(example.prompt)
    poisoned_prompt = example.prompt.replace(parsed.request, malicious_request, 1)
    poisoned = replace(example, prompt=poisoned_prompt)
    with pytest.raises(ValueError):
        mobile_planir_v2._render_prompt(poisoned, "C")


@requires_local_evidence
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_split", "test"),
        ("derived_split", "evaluation"),
        ("evaluation_role", "confirmation"),
        ("population_role", "selection"),
    ],
)
def test_internal_oracle_rejects_nonconstruction_metadata(field: str, value: str) -> None:
    example = load_manifest(
        CONSTRUCTION_MANIFEST,
        expected_sha256="800a3ba0a7f0215e5cf95c77c32e47abed48b2f07f375954b0e0b08296a33d10",
        expected_derived_split="train",
    )[0]
    poisoned = replace(example, metadata={**example.metadata, field: value})
    with pytest.raises(ValueError):
        mobile_planir_v2._derive_oracle(poisoned)


@requires_local_evidence
def test_full_construction_oracle_receipt_is_immutable_training_evidence() -> None:
    audit = audit_construction_oracles(CONSTRUCTION_MANIFEST)
    assert audit.use == CONSTRUCTION_USE == "training-and-feasibility-only"
    assert (
        (audit.source_rows, audit.accepted_rows, audit.rejected_rows)
        == (
            CONSTRUCTION_ROWS,
            ACCEPTED_ROWS,
            1,
        )
        == (5_745, 5_744, 1)
    )
    assert audit.accepted_membership_sha256 == ACCEPTED_MEMBERSHIP_SHA256
    assert (
        audit.date_operator_counts
        == EXPECTED_DATE_OPERATOR_COUNTS
        == {
            "A": 1_333,
            "M": 357,
            "R": 138,
            "W": 241,
        }
    )
    assert audit.clock_references == EXPECTED_CLOCK_REFERENCES == 2_070
    assert [(item.example_id, item.code) for item in audit.rejections] == [
        (EXPECTED_REJECTION_ID, EXPECTED_REJECTION_CODE)
    ]
    assert audit.rejections[0].target is None
    assert audit.rejections[0].compiled_action_ir is None


@requires_local_evidence
def test_exact_pinned_tokenizer_a_b_c_receipt_includes_grammar_gap_correction() -> None:
    audit = audit_construction_tokens(CONSTRUCTION_MANIFEST, PINNED_TOKENIZER)
    assert audit.manifest_sha256 == CONSTRUCTION_MANIFEST_SHA256
    assert audit.tokenizer_sha256 == PINNED_TOKENIZER_SHA256
    assert audit.accepted_membership_sha256 == ACCEPTED_MEMBERSHIP_SHA256
    assert audit.rejection_ids == (EXPECTED_REJECTION_ID,)
    assert (audit.compared_rows, audit.rejected_rows, audit.max_sequence_length) == (
        5_744,
        1,
        2_048,
    )
    assert (
        audit.arm_a.prompt.total,
        audit.arm_a.prompt.mean,
        audit.arm_a.prompt.median,
        audit.arm_a.prompt.p90,
        audit.arm_a.prompt.maximum,
    ) == pytest.approx((1_443_385, 251.2856894150418, 247, 284, 487))
    assert (
        audit.arm_a.target.total,
        audit.arm_a.target.mean,
        audit.arm_a.target.median,
        audit.arm_a.target.p90,
        audit.arm_a.target.maximum,
    ) == pytest.approx((514_097, 89.50156685236769, 94, 138, 196))
    assert (
        audit.arm_a.sequence.total,
        audit.arm_a.sequence.mean,
        audit.arm_a.sequence.median,
        audit.arm_a.sequence.p90,
        audit.arm_a.sequence.maximum,
    ) == pytest.approx((1_963_226, 341.7872562674095, 341, 420, 660))

    # The shared v2 grammar adds two target-independent nonempty tables and 22
    # prompt tokens relative to the exploratory quote-v1-enumerator prototype.
    assert (
        audit.arm_b.prompt.total,
        audit.arm_b.prompt.mean,
        audit.arm_b.prompt.median,
        audit.arm_b.prompt.p90,
        audit.arm_b.prompt.maximum,
    ) == pytest.approx((1_555_849, 270.8650766016713, 267, 319, 516))
    assert audit.arm_b.target == audit.arm_a.target
    assert (
        audit.arm_b.sequence.total,
        audit.arm_b.sequence.mean,
        audit.arm_b.sequence.median,
        audit.arm_b.sequence.p90,
        audit.arm_b.sequence.maximum,
    ) == pytest.approx((2_075_690, 361.366643454039, 359, 456, 689))

    assert (
        audit.arm_c.target.total,
        audit.arm_c.target.mean,
        audit.arm_c.target.median,
        audit.arm_c.target.p90,
        audit.arm_c.target.maximum,
    ) == pytest.approx((506_062, 88.10271587743732, 92, 135, 192))
    assert (
        audit.arm_c.sequence.total,
        audit.arm_c.sequence.mean,
        audit.arm_c.sequence.median,
        audit.arm_c.sequence.p90,
        audit.arm_c.sequence.maximum,
    ) == pytest.approx((2_067_655, 359.96779247910865, 357, 453, 689))
    assert (
        audit.table_nonempty_rows,
        audit.date_reference_instances,
        audit.time_reference_instances,
    ) == (2_637, 5_241, 3_581)
    assert (
        audit.c_targets_shorter_than_a,
        audit.c_targets_equal_to_a,
        audit.c_targets_longer_than_a,
    ) == (2_069, 3_675, 0)
    assert (audit.over_limit_a, audit.over_limit_b, audit.over_limit_c) == (0, 0, 0)


@pytest.mark.parametrize("invalid_limit", [True, False, 0, -1, 1.5, "2048", None])
def test_token_audit_rejects_nonpositive_bool_and_noninteger_limits(invalid_limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        audit_construction_tokens(
            CONSTRUCTION_MANIFEST,
            PINNED_TOKENIZER,
            max_sequence_length=invalid_limit,  # type: ignore[arg-type]
        )
