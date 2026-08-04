from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from barunlm.datasets import action_correction_forge as forge

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/action_correction_forge_screen_v1.json"

EXPECTED_SINGLE_FAULT_BY_STRATUM = {
    "ADD_LIST_ITEM": 492,
    "CREATE_CALENDAR_EVENT": 520,
    "CREATE_NOTE": 523,
    "CREATE_REMINDER": 475,
    "LOOK_UP_CONTACT": 523,
    "LOOK_UP_ROUTE": 491,
    "PAUSE_MEDIA": 492,
    "PLAY_MEDIA": 498,
    "PROPOSE_MESSAGE": 529,
    "RESCHEDULE_CALENDAR_EVENT": 482,
    "SET_BOOLEAN_SETTING": 508,
    "SET_LIST_ITEM_CHECKED": 502,
    "UPDATE_REMINDER": 530,
    "ABSTAIN": 502,
    "CLARIFY": 487,
    "CONFIRM": 522,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _planned_v1_source_id(stratum: str, ordinal: int) -> str:
    families = forge.ROLE_FAMILIES["T-synth"]
    lineage = forge.DeclaredLineage(
        role="T-synth",
        renderer_family=families.renderer_family,
        entity_family=families.entity_family,
        temporal_family=families.temporal_family,
        schema_family=families.schema_family,
        context_family=families.context_family,
        source_family=families.source_family,
        program_skeleton_family=(f"acf.program.train.{stratum.lower().replace('_', '-')}.v1"),
    )
    commitment = _canonical_json(
        {
            "version": forge.FORGE_VERSION,
            "role": "T-synth",
            "stratum": stratum,
            "ordinal": ordinal,
            "lineage": lineage.to_dict(),
        }
    )
    suffix = hashlib.sha256(commitment.encode("utf-8")).hexdigest()[:12]
    slug = stratum.lower().replace("_", "-")
    return f"acf.t.{slug}.{ordinal:02d}.{suffix}"


def test_reproduction_matches_the_two_bounded_source_ids() -> None:
    for stratum in forge.DESCRIPTIVE_STRATA:
        for ordinal in range(forge.PROTOTYPE_MAX_FIXTURE_ROWS_PER_STRATUM):
            assert (
                _planned_v1_source_id(stratum, ordinal)
                == forge.build_forge_source("T-synth", stratum, ordinal).source_id
            )


def test_v1_low_bit_rule_fails_its_planned_exact_50_50_population_gate() -> None:
    observed: dict[str, int] = {}
    for stratum in forge.DESCRIPTIVE_STRATA:
        observed[stratum] = sum(
            forge.c_assignment(_planned_v1_source_id(stratum, ordinal)) == "single_fault"
            for ordinal in range(1024)
        )

    assert observed == EXPECTED_SINGLE_FAULT_BY_STRATUM
    assert sum(observed.values()) == 8076
    assert 16384 - sum(observed.values()) == 8308
    assert any(count != 512 for count in observed.values())

    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert contract["training"]["c_certified_single_fault_draft_rows_per_fit"] == 8192
    assert contract["training"]["c_exact_draft_rows_per_fit"] == 8192
    assert (
        contract["training"]["budget_requirements"]["clean_fault_split_verification_passed"]
        is False
    )


def test_v1_parsed_draft_object_cannot_transport_exact_arbitrary_raw_output() -> None:
    source = forge.build_forge_source("T-synth", "CREATE_NOTE", 0)
    correction_input = json.loads(forge.build_view_c(source).input_json)

    assert set(inspect.signature(forge.build_view_c).parameters) == {"source"}
    assert "draft_raw" not in correction_input
    assert isinstance(correction_input["draft"], dict)

    noncanonical_raw = ' { "decision" : "ABSTAIN" } '
    parsed_then_rendered = _canonical_json(json.loads(noncanonical_raw))
    assert parsed_then_rendered == '{"decision":"ABSTAIN"}'
    assert parsed_then_rendered != noncanonical_raw

    for unrepresentable_raw in (
        "not json",
        '{"decision":"ABSTAIN"} suffix',
        '{"decision":"ABSTAIN"',
        '{"decision":"ABSTAIN","decision":"CALL"}',
    ):
        try:
            parsed = json.loads(unrepresentable_raw)
        except json.JSONDecodeError:
            continue
        assert _canonical_json(parsed) != unrepresentable_raw
