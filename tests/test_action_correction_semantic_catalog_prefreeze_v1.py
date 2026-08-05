from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/action_correction_semantic_catalog_prefreeze_v1.json"
DOC = ROOT / "docs/action-correction-semantic-catalog-v1.md"
EXPECTED_CONFIG_SHA256 = "68955b54e52820aab80f83450ef7aa97da9c50687f703d5b547571b1c1fbc866"


def _payload() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_exact_draft_identity_and_every_authorization_remain_closed() -> None:
    payload = _payload()

    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == EXPECTED_CONFIG_SHA256
    assert payload["schema_version"] == ("barun-action-correction-semantic-catalog-prefreeze-v1")
    assert payload["run_id"] == ("20260805-0505-action-correction-semantic-catalog-prefreeze-s17")
    assert payload["status"] == "cpu_contract_draft_non_authorizing"
    assert payload["authorization"]
    assert all(value is False for value in payload["authorization"].values())


def test_coordinate_decoder_is_a_fixed_bijection_domain_without_enumerating_rows() -> None:
    payload = _payload()
    decoder = payload["coordinate_decoder"]
    axes = decoder["mixed_radix_order_least_significant_first"]
    cardinalities = [axis["cardinality"] for axis in axes]

    assert [axis["axis"] for axis in axes] == [
        "renderer_variant",
        "wording_variant",
        "entity_bundle",
        "temporal_bundle",
        "context_variant",
    ]
    assert math.prod(cardinalities) == decoder["mixed_radix_product"] == 2048
    assert decoder["modulus"] == 2048
    assert decoder["adaptive_or_outcome_dependent_coordinates_authorized"] is False

    roles = payload["roles"]
    assert {role: value["candidate_ordinals_per_stratum"] for role, value in roles.items()} == {
        "T": 2048,
        "P": 128,
        "D": 512,
    }
    for role in ("T", "P", "D"):
        multiplier = roles[role]["affine_multiplier_mod_2048"]
        offset = roles[role]["affine_offset_mod_2048"]
        assert math.gcd(multiplier, decoder["modulus"]) == 1
        assert 0 <= offset < decoder["modulus"]


def test_role_families_and_prompt_shapes_are_substantively_distinct() -> None:
    payload = _payload()
    roles = payload["roles"]
    family_fields = (
        "renderer_family",
        "request_grammar_family",
        "schema_presentation_family",
        "context_shape_family",
        "timezone",
        "temporal_epoch",
        "entity_lexicon_family",
        "logical_source_batch",
    )
    for field in family_fields:
        assert len({roles[role][field] for role in ("T", "P", "D")}) == 3

    shapes = payload["role_prompt_shapes"]
    assert len({tuple(shapes[role]["top_level_fields"]) for role in ("T", "P", "D")}) == 3
    assert len({shapes[role]["state_path"] for role in ("T", "P", "D")}) == 3
    assert shapes["P"]["state_path"] == "$.environment.snapshot"


def test_catalog_claims_no_independence_or_future_receipts() -> None:
    payload = _payload()
    lineage = payload["lineage_and_duplicate_contract"]
    binding = payload["future_roster_binding"]
    render = payload["render_contract"]

    assert lineage["role_or_split_prefix_is_disjointness_evidence"] is False
    assert lineage["slot_or_coordinate_hash_is_disjointness_evidence"] is False
    assert lineage["effective_component_or_power_claimed_by_this_contract"] is False
    assert lineage["semantic_lineage_disjointness_proven"] is False
    assert lineage["complete_near_pair_scan_proven"] is False
    assert render["complete_candidate_enumeration_in_this_run"] is False
    assert render["ordinal_or_hash_only_uniqueness_counts_as_semantics"] is False
    assert binding["global_placeholder_coordinate_permitted"] is False
    for key, value in binding.items():
        if key.endswith("_sha256"):
            assert value is None


def test_fault_controls_and_registered_strata_are_explicit() -> None:
    payload = _payload()
    strata = payload["registered_strata"]
    fault = payload["fault_contract"]

    assert len(strata) == len(set(strata)) == 16
    assert set(fault["decision_only_control_strata"]) == {"PAUSE_MEDIA", "ABSTAIN"}
    assert set(fault["argument_fault_required_except"]) == {"PAUSE_MEDIA", "ABSTAIN"}
    assert fault["failed_certificate_policy"] == ("abort_complete_population_without_backfill")
    assert fault["arbitrary_model_output_fault_distance_scorer_included"] is False
    assert fault["arbitrary_model_output_fault_distance_scorer_required_before_launch"] is True


def test_markdown_uses_canonical_names_and_binds_the_draft() -> None:
    text = DOC.read_text(encoding="utf-8")

    assert "BarunAction-35M" in text
    assert "StrataLM" not in text
    assert EXPECTED_CONFIG_SHA256 in text
    assert "nonauthorizing" in text
    assert "candidate-universe enumeration" in text
