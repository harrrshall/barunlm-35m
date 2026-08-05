from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import barunlm.evaluation.action_correction_contract as subject

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/action_correction_forge_screen_v1.json"
SPEC = ROOT / "docs/generate-correct-sft-v1.md"


def _payload() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _mutated(tmp_path: Path, mutate: Any) -> Path:
    payload = _payload()
    mutate(payload)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_current_prefreeze_contract_is_strictly_accepted_and_nonauthorizing() -> None:
    contract = subject.ActionCorrectionContract.from_json(CONFIG)

    assert contract.run_id == subject.RUN_ID
    assert contract.schema_version == subject.CONTRACT_SCHEMA_VERSION
    assert contract.source_sha256 == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert contract.roles == subject.ALL_ROLES
    assert contract.seeds == subject.SEEDS
    assert contract.rows_per_fit == 38_512
    assert contract.optimizer_updates_per_fit == 602
    assert contract.launch_authorized is False
    assert contract.model_cuda_authorized is False
    assert contract.population_materialized is False


def test_roles_include_required_causal_controls_without_adding_fits() -> None:
    payload = _payload()

    assert {"A", "B", "C", "G0", "A2", "C1", "C2"}.issubset(payload["roles"])
    assert payload["roles"]["G02"] == {
        "kind": "no_fit_two_pass_untrained_prompt_control",
        "passes": 2,
        "training_view": "none",
        "weights": "candidate_v2",
    }
    assert payload["roles"]["B0"]["weights"] == "canonical_base"
    assert payload["training"]["training_arms"] == ["A", "B", "C"]
    assert payload["training"]["total_fits"] == 9


def test_base_and_candidate_checkpoint_identities_are_frozen_without_loading_them() -> None:
    identities = _payload()["checkpoint_identities"]
    assert identities["canonical_base"]["repo_id"] == "harrrshall/BarunLM-35M"
    assert identities["canonical_base"]["parameter_count"] == 35_072_768
    assert identities["canonical_base"]["file_sha256"]["model.safetensors"] == (
        "f2a7c88b9f2c2e3584809081407ab136795d82e30e89b730e007781c45d01447"
    )
    assert identities["candidate_v2"]["parameter_count"] == 35_072_768
    assert identities["candidate_v2"]["file_sha256"]["model.safetensors"] == (
        "fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3"
    )


def test_fit_arithmetic_and_unverified_budget_requirements_are_explicit() -> None:
    payload = _payload()
    population = payload["population_plan"]
    training = payload["training"]

    assert (
        population["t_replay"]["accepted_rows"] + population["t_synth"]["planned_rows"] * 2
        == training["rows_per_fit"]
        == 38_512
    )
    assert (training["rows_per_fit"] + training["batch_size"] - 1) // training["batch_size"] == 602
    assert training["last_batch_rows"] == 48
    requirements = training["budget_requirements"]
    assert requirements["same_source_ids_required"] is True
    assert requirements["same_complete_gold_target_by_source_required"] is True
    assert requirements["same_supervised_target_token_total_required"] is True
    assert requirements["clean_fault_split_verification_passed"] is False
    assert requirements["every_fault_draft_single_fault_certificate_required"] is True
    assert requirements["fault_certification_verification_passed"] is False
    assert requirements["source_and_target_budget_verification_passed"] is False
    assert requirements["input_compute_matched_claimed"] is False
    assert training["c_exact_draft_rows_per_fit"] == 8_192
    assert training["c_certified_single_fault_draft_rows_per_fit"] == 8_192
    assert training["optimization"]["warmup_steps"] == 18


def test_replay_and_internal_holdout_cannot_be_misrepresented_as_fresh_components() -> None:
    population = _payload()["population_plan"]
    replay = population["t_replay"]
    internal = population["d_internal"]

    assert replay["source_rows"] == 5_745
    assert replay["accepted_rows"] == 5_744
    assert replay["source_sha256"] == subject.MOBILE_CONSTRUCTION_SOURCE_SHA256
    assert replay["source_membership_order_sha256"] == (
        subject.MOBILE_CONSTRUCTION_SOURCE_MEMBERSHIP_SHA256
    )
    assert replay["exclusion_ledger_sha256"] == subject.REPLAY_EXCLUSION_LEDGER_SHA256
    assert replay["accepted_membership_order_sha256"] == (subject.REPLAY_ACCEPTED_MEMBERSHIP_SHA256)
    assert replay["known_bad_label_id"] == subject.KNOWN_BAD_REPLAY_ID
    assert replay["direct_action_ir_only"] is True
    assert replay["fresh_or_selection_evidence"] is False
    assert internal["nominal_rows"] == 4_096
    assert internal["registered_descriptive_strata"] == 16
    assert internal["nominal_rows_per_registered_stratum"] == 256
    assert internal["effective_components_claimed"] is None
    assert internal["effective_components_verified"] is False
    assert internal["independent_or_powered_population_claimed"] is False
    assert internal["old_7937_756_v3_planir_presto_dev_lineage_allowed"] is False


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.__setitem__("launch_authorized", True), "launch_authorized"),
        (
            lambda value: value.__setitem__("model_cuda_authorized", True),
            "model_cuda_authorized",
        ),
        (
            lambda value: value.__setitem__("population_materialized", True),
            "population_materialized",
        ),
        (
            lambda value: value["distinctness"].__setitem__("uses_k8_candidates", True),
            "distinctness",
        ),
        (
            lambda value: value["distinctness"].__setitem__("uses_planir_or_placeholder_v2", True),
            "distinctness",
        ),
        (lambda value: value["roles"].pop("G02"), "causal roles"),
        (lambda value: value["training"].__setitem__("seeds", [17]), "training seeds"),
        (lambda value: value["training"].__setitem__("rows_per_fit", 38_511), "rows per fit"),
        (
            lambda value: value["training"].__setitem__("optimizer_updates_per_fit", 601),
            "updates per fit",
        ),
        (
            lambda value: value["forbidden_population_access"].__setitem__(
                "mobile_official_evaluation_961_rows_read", 1
            ),
            "zero-access",
        ),
        (
            lambda value: value["population_plan"]["d_internal"].__setitem__(
                "effective_components_claimed", 4_096
            ),
            "D-internal",
        ),
        (
            lambda value: value["training"]["budget_requirements"].__setitem__(
                "source_and_target_budget_verification_passed", True
            ),
            "matched-budget",
        ),
        (
            lambda value: value["gates"].__setitem__("evaluated", True),
            "internal keep gates",
        ),
        (
            lambda value: value["phase_firewall"].__setitem__(
                "D_access_before_all_nine_final_checkpoints_frozen", True
            ),
            "phase firewall",
        ),
    ],
)
def test_authorization_scientific_and_budget_mutations_fail_closed(
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    with pytest.raises(subject.ActionCorrectionContractError, match=match):
        subject.ActionCorrectionContract.from_json(_mutated(tmp_path, mutate))


def test_all_forbidden_populations_are_bound_to_zero_access() -> None:
    forbidden = _payload()["forbidden_population_access"]
    assert forbidden == {
        "gvs_population_rows_read": 0,
        "mobile_official_evaluation_961_rows_read": 0,
        "mobile_reused_development_756_rows_read": 0,
        "planir_screen_rows_read": 0,
        "presto_development_rows_read": 0,
        "private_human_labels_read": 0,
        "v3_confirmation_rows_read": 0,
        "v3_selection_rows_read": 0,
    }


def test_official_development_retention_gate_is_truthfully_unevaluable() -> None:
    gate = _payload()["gates"]["standing_synthetic_tranche_gate"]
    assert gate["minimum_hard_development_gain"] == [5, 100]
    assert gate["maximum_official_development_loss"] == [2, 100]
    assert gate["official_development_population_authorized"] is False
    assert gate["official_development_retention_evaluable"] is False
    assert gate["official_development_retention_passed"] is None


def test_preview_checkpoint_is_predesignated_without_seed_selection_or_refit() -> None:
    payload = _payload()
    policy = payload["training"]["preview_checkpoint_policy"]
    assert policy == {
        "best_seed_selection_authorized": False,
        "maximum_total_fits": 9,
        "post_D_refit_authorized": False,
        "predesignated_preview_seed": 17,
        "replication_only_seeds": [29, 43],
        "tenth_fit_authorized": False,
        "weight_averaging_authorized": False,
    }
    assert payload["gates"]["preview_checkpoint"] == {
        "predesignated_seed": 17,
        "seed17_must_individually_pass_all_applicable_quality_validity_safety_efficiency": True,
    }


def _output(raw: str | None, *, valid: bool) -> subject.PassOutput:
    return subject.PassOutput(
        raw=raw,
        parse_valid=valid,
        schema_valid=valid,
        policy_conformant_proposal=valid,
    )


def test_two_pass_fallback_prefers_valid_pass2_then_pass1_then_abstains() -> None:
    both_valid = subject.select_two_pass_output(
        pass1=_output('{"decision":"CALL","calls":[]}', valid=True),
        pass2=_output('{"decision":"CLARIFY","missing":["time"]}', valid=True),
    )
    assert both_valid.source == "pass2"
    assert both_valid.raw == '{"decision":"CLARIFY","missing":["time"]}'

    pass1_only = subject.select_two_pass_output(
        pass1=_output('{"decision":"ABSTAIN"}', valid=True),
        pass2=_output("not json", valid=False),
    )
    assert pass1_only == subject.SelectedPassOutput(raw='{"decision":"ABSTAIN"}', source="pass1")

    neither = subject.select_two_pass_output(
        pass1=_output(None, valid=False),
        pass2=_output("not json", valid=False),
    )
    assert neither == subject.SelectedPassOutput(
        raw=subject.CANONICAL_ABSTAIN,
        source="canonical_abstain",
    )


def test_two_pass_caps_are_unresolved_and_policy_conformance_never_executes() -> None:
    contract = _payload()["two_pass_contract"]
    assert contract["decoding_caps_required_before_model_authorization"] is True
    assert contract["decoding_caps_resolved"] is False
    assert contract["pass1_max_new_tokens"] is None
    assert contract["pass2_max_new_tokens"] is None
    assert contract["execution_permitted"] is False
    assert contract["real_tool_execution"] is False
    assert contract["validity_predicate"] == [
        "parse_valid",
        "schema_valid",
        "policy_conformant_proposal",
    ]


def test_fallback_requires_complete_parse_schema_and_policy_evidence() -> None:
    blocked_second = subject.PassOutput(
        raw='{"decision":"CALL"}',
        parse_valid=True,
        schema_valid=True,
        policy_conformant_proposal=False,
    )
    selected = subject.select_two_pass_output(
        pass1=_output('{"decision":"ABSTAIN"}', valid=True),
        pass2=blocked_second,
    )
    assert selected.source == "pass1"

    with pytest.raises(ValueError, match="schema-valid"):
        subject.PassOutput(
            raw="x",
            parse_valid=False,
            schema_valid=True,
            policy_conformant_proposal=False,
        )
    with pytest.raises(ValueError, match="missing output"):
        subject.PassOutput(
            raw=None,
            parse_valid=True,
            schema_valid=True,
            policy_conformant_proposal=True,
        )


def test_draft_diagnostics_accept_only_draft_derived_vocabulary() -> None:
    diagnostics = subject.validate_draft_diagnostics(
        {
            "parse_valid": True,
            "schema_valid": True,
            "policy_status": "requires_confirmation",
            "simulator_status": "accepted",
            "error_codes": ["confirmation_required"],
        }
    )
    assert diagnostics.error_codes == ("confirmation_required",)
    contract = _payload()["diagnostics_contract"]
    assert contract["gold_derived_diagnostics_allowed"] is False
    assert contract["fault_certificate_model_visible"] is False
    assert contract["runtime_recomputation_verified"] is False


@pytest.mark.parametrize(
    "payload,match",
    [
        (
            {
                "parse_valid": True,
                "schema_valid": True,
                "policy_status": "conformant",
                "simulator_status": "accepted",
                "error_codes": [],
                "gold_action": {"decision": "ABSTAIN"},
            },
            "unknown fields",
        ),
        (
            {
                "parse_valid": True,
                "schema_valid": True,
                "policy_status": "conformant",
                "simulator_status": "accepted",
                "error_codes": ["gold_value_mismatch"],
            },
            "forbidden source",
        ),
        (
            {
                "parse_valid": False,
                "schema_valid": True,
                "policy_status": "conformant",
                "simulator_status": "accepted",
                "error_codes": [],
            },
            "schema-valid",
        ),
        (
            {
                "parse_valid": True,
                "schema_valid": True,
                "policy_status": "conformant",
                "simulator_status": "accepted",
                "error_codes": ["z_error", "a_error"],
            },
            "sorted and unique",
        ),
    ],
)
def test_draft_diagnostics_reject_gold_fields_and_forged_shapes(
    payload: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(subject.ActionCorrectionContractError, match=match):
        subject.validate_draft_diagnostics(payload)


def test_duplicate_keys_and_nonfinite_numbers_are_rejected(tmp_path: Path) -> None:
    raw = CONFIG.read_text(encoding="utf-8")
    duplicate = raw.replace(
        '  "run_id": "20260805-0230-action-correction-forge-screen-s17",',
        '  "run_id": "20260805-0230-action-correction-forge-screen-s17",\n'
        '  "run_id": "20260805-0230-action-correction-forge-screen-s17",',
        1,
    )
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(subject.ActionCorrectionContractError, match="duplicate JSON key"):
        subject.ActionCorrectionContract.from_json(duplicate_path)

    nonfinite = raw.replace('"base_parameter_count": 35072768', '"base_parameter_count": NaN')
    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text(nonfinite, encoding="utf-8")
    with pytest.raises(subject.ActionCorrectionContractError, match="non-finite"):
        subject.ActionCorrectionContract.from_json(nonfinite_path)


def test_new_spec_uses_only_canonical_model_names_and_preserves_claim_boundary() -> None:
    spec = SPEC.read_text(encoding="utf-8")
    assert "BarunLM-35M" in spec
    assert "BarunAction-35M" in spec
    assert "StrataLM" not in spec
    assert "stratalm" not in spec.casefold()
    assert "4,096 independent" in spec
    assert (
        "No result from this template-generated internal screen can support a breakthrough" in spec
    )
