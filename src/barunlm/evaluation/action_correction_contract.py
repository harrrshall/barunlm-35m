"""CPU-only contract checks for the nonauthorizing Generate-Correct SFT v1 plan.

This module deliberately contains no model, Torch, CUDA, network, dataset, or
JarvisLabs capability.  It validates the immutable pre-materialization contract and
implements the small deterministic two-pass fallback that a later, separately
authorized runtime may reuse.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA_VERSION = "barun-action-correction-forge-prefreeze-v1"
RUN_ID = "20260805-0230-action-correction-forge-screen-s17"
BASE_MODEL_NAME = "BarunLM-35M"
POST_TRAINED_MODEL_NAME = "BarunAction-35M"
PARAMETER_COUNT = 35_072_768

TRAINING_ARMS = ("A", "B", "C")
SEEDS = (17, 29, 43)
ALL_ROLES = ("A", "B", "C", "G0", "A2", "C1", "C2", "G02", "B0")
T_SYNTH_ROWS = 16_384
T_REPLAY_SOURCE_ROWS = 5_745
T_REPLAY_ACCEPTED_ROWS = 5_744
D_INTERNAL_NOMINAL_ROWS = 4_096
ROWS_PER_FIT = 38_512
BATCH_SIZE = 64
UPDATES_PER_FIT = 602
LAST_BATCH_ROWS = 48
TOTAL_FITS = 9
MOBILE_CONSTRUCTION_SOURCE_SHA256 = (
    "800a3ba0a7f0215e5cf95c77c32e47abed48b2f07f375954b0e0b08296a33d10"
)
MOBILE_CONSTRUCTION_SOURCE_MEMBERSHIP_SHA256 = (
    "b70f6dfe79cfe49792b6e6f5d414fd509c889d18101f75cd8df6d4e4f146c588"
)
REPLAY_EXCLUSION_LEDGER_SHA256 = "7593b556b9c096308ca1a9e9d9b89627ebf19710da930bd4dd5794c58879616a"
REPLAY_ACCEPTED_MEMBERSHIP_SHA256 = (
    "27f9acb8905db132420ddb5d11a20bed138f2e2927aec41672f6852b3895452d"
)
KNOWN_BAD_REPLAY_ID = "mobile-actions-04894-6bd7643b1bc95697"

CANONICAL_ABSTAIN = '{"decision":"ABSTAIN"}'
DIAGNOSTIC_FIELDS = (
    "parse_valid",
    "schema_valid",
    "policy_status",
    "simulator_status",
    "error_codes",
)
_POLICY_STATUSES = frozenset({"blocked", "conformant", "requires_confirmation", "unavailable"})
_SIMULATOR_STATUSES = frozenset({"accepted", "rejected", "not_run", "unavailable"})
_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_FORBIDDEN_DIAGNOSTIC_TERMS = (
    "expected",
    "fault_kind",
    "gold",
    "label",
    "oracle",
    "reference_answer",
    "target",
)


class ActionCorrectionContractError(ValueError):
    """The pre-materialization contract or two-pass evidence is invalid."""


def _reject_constant(value: str) -> None:
    raise ActionCorrectionContractError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ActionCorrectionContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _object(value: object, *, context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ActionCorrectionContractError(f"{context} must be an exact JSON object")
    return value


def _exact_keys(value: dict[str, Any], *, keys: set[str], context: str) -> None:
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing:
        raise ActionCorrectionContractError(f"{context} is missing fields {missing}")
    if unknown:
        raise ActionCorrectionContractError(f"{context} has unknown fields {unknown}")


def _strictly_equal(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        observed_dict = observed
        expected_dict = expected
        assert isinstance(observed_dict, dict) and isinstance(expected_dict, dict)
        return set(observed_dict) == set(expected_dict) and all(
            _strictly_equal(observed_dict[key], expected_dict[key]) for key in expected_dict
        )
    if type(expected) is list:
        observed_list = observed
        expected_list = expected
        assert isinstance(observed_list, list) and isinstance(expected_list, list)
        return len(observed_list) == len(expected_list) and all(
            _strictly_equal(left, right)
            for left, right in zip(observed_list, expected_list, strict=True)
        )
    return observed == expected


def _expect(value: object, expected: object, *, context: str) -> None:
    if not _strictly_equal(value, expected):
        raise ActionCorrectionContractError(f"{context} changed from the immutable v1 contract")


def _expected_distinctness() -> dict[str, object]:
    return {
        "closed_predecessors": {
            "generate_verify_select_v1": "closed_before_population_or_model_access",
            "grounded_planir_placeholder_v2": "closed_terminal_no_retry",
            "month_boundary_counterfactual_v3": "closed_scientific_rejection",
        },
        "output_contract": "ACTION_IR_V1",
        "reopens_closed_run_or_recipe": False,
        "uses_gvs_population_candidate_set_ranker_or_receipt": False,
        "uses_k8_candidates": False,
        "uses_month_boundary_counterfactual_transform": False,
        "uses_planir_or_placeholder_v2": False,
        "uses_verifier_or_learned_ranker": False,
    }


def _expected_checkpoint_identities() -> dict[str, object]:
    return {
        "canonical_base": {
            "file_sha256": {
                "barun_config.json": (
                    "9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565"
                ),
                "model.safetensors": (
                    "f2a7c88b9f2c2e3584809081407ab136795d82e30e89b730e007781c45d01447"
                ),
                "tokenizer.json": (
                    "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6"
                ),
            },
            "parameter_count": PARAMETER_COUNT,
            "repo_id": "harrrshall/BarunLM-35M",
            "revision": "ef3e483a9fd7d906ecf2a7929babeffaf82d1d16",
        },
        "candidate_v2": {
            "candidate_descriptor_sha256": (
                "ec76e48ab1fe5105482aaad902954ccc3f8f3d23fb98226b66d13b292c2a4ac9"
            ),
            "checkpoint_manifest_sha256": (
                "c743ab7c4d33ae75c6b0aa4547458a961b92766da8fcf85fd148fda2ebb5530a"
            ),
            "file_sha256": {
                "barun_config.json": (
                    "9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565"
                ),
                "model.safetensors": (
                    "fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3"
                ),
                "tokenizer.json": (
                    "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6"
                ),
            },
            "parameter_count": PARAMETER_COUNT,
            "source_run_id": "20260803-1845-mobile-followup-retry-s17",
            "step": 126,
        },
    }


def _expected_roles() -> dict[str, object]:
    return {
        "A": {
            "kind": "trained_arm_and_one_pass_evaluation",
            "passes": 1,
            "training_view": "direct_action_ir_repeat",
            "weights": "arm_A",
        },
        "B": {
            "kind": "trained_arm_and_one_pass_evaluation",
            "passes": 1,
            "training_view": "single_semantic_field_masked_full_action_reconstruction",
            "weights": "arm_B",
        },
        "C": {
            "kind": "trained_arm_weights_only",
            "passes": 0,
            "training_view": "exact_or_certified_single_fault_draft_50_50",
            "weights": "arm_C",
        },
        "G0": {
            "kind": "no_fit_one_pass_reference",
            "passes": 1,
            "training_view": "none",
            "weights": "candidate_v2",
        },
        "A2": {
            "kind": "no_additional_fit_two_pass_process_control",
            "passes": 2,
            "training_view": "none",
            "weights": "arm_A",
        },
        "C1": {
            "kind": "no_additional_fit_one_pass_training_effect_control",
            "passes": 1,
            "training_view": "none",
            "weights": "arm_C",
        },
        "C2": {
            "kind": "no_additional_fit_two_pass_treatment",
            "passes": 2,
            "training_view": "none",
            "weights": "arm_C",
        },
        "G02": {
            "kind": "no_fit_two_pass_untrained_prompt_control",
            "passes": 2,
            "training_view": "none",
            "weights": "candidate_v2",
        },
        "B0": {
            "kind": "no_fit_one_pass_base_probe_reference",
            "passes": 1,
            "training_view": "none",
            "weights": "canonical_base",
        },
    }


def _expected_forbidden_access() -> dict[str, int]:
    return {
        "gvs_population_rows_read": 0,
        "mobile_official_evaluation_961_rows_read": 0,
        "mobile_reused_development_756_rows_read": 0,
        "planir_screen_rows_read": 0,
        "presto_development_rows_read": 0,
        "private_human_labels_read": 0,
        "v3_confirmation_rows_read": 0,
        "v3_selection_rows_read": 0,
    }


def _expected_gates() -> dict[str, object]:
    return {
        "evaluated": False,
        "passed": None,
        "hard_conjunction": True,
        "statistical_or_independent_component_claim_authorized": False,
        "base_probe": {
            "comparator": "B0",
            "minimum_schema_validity": [95, 100],
            "minimum_exact_gain_over_base": [25, 100],
        },
        "standing_synthetic_tranche_gate": {
            "hard_development_comparator": "G0",
            "minimum_hard_development_gain": [5, 100],
            "maximum_official_development_loss": [2, 100],
            "official_development_population_authorized": False,
            "official_development_retention_evaluable": False,
            "official_development_retention_passed": None,
        },
        "treatment_quality": {
            "minimum_execution_gain_C2_vs_G0": [5, 100],
            "minimum_execution_gain_C2_vs_each_of_A_B_A2_G02": [3, 100],
            "minimum_ast_exact_gain_C2_vs_each_of_A_B": [3, 100],
            "minimum_execution_gain_C2_vs_C1": [2, 100],
            "minimum_argument_value_exact_gain_C2_vs_each_of_A_B": [5, 100],
            "minimum_positive_seed_comparisons_out_of_three": 2,
            "minimum_positive_registered_strata_out_of_sixteen": 12,
            "minimum_correct_draft_retention": [98, 100],
            "minimum_certified_single_fault_recovery": [40, 100],
            "minimum_fixes_to_regressions_ratio": [3, 1],
            "maximum_registered_stratum_regression": [2, 100],
        },
        "validity_and_safety": {
            "minimum_parse_validity": [995, 1000],
            "minimum_schema_validity": [99, 100],
            "maximum_false_action_increase": [1, 100],
            "maximum_catastrophic_unauthorized_actions": 0,
            "maximum_generation_failures": 0,
            "maximum_missing_predictions": 0,
            "maximum_truncations": 0,
        },
        "efficiency": {
            "device": "MacBook Pro M5 Pro",
            "maximum_p95_latency_ratio_vs_candidate_v2": [5, 2],
            "maximum_peak_rss_ratio_vs_candidate_v2": [5, 4],
            "maximum_checkpoint_size_ratio_vs_candidate_v2": [101, 100],
        },
        "preview_checkpoint": {
            "predesignated_seed": 17,
            "seed17_must_individually_pass_all_applicable_quality_validity_safety_efficiency": True,
        },
        "hierarchical_internal_keep_order": [
            "C2_if_complete_treatment_conjunction_passes",
            "B_if_C2_fails_and_B_beats_G0_by_3_points_and_A_by_2_points_with_all_safety_gates",
            "A_if_C2_and_B_fail_and_A_beats_G0_by_3_points_with_all_safety_gates",
            "otherwise_keep_candidate_v2",
        ],
    }


@dataclass(frozen=True, slots=True)
class ActionCorrectionContract:
    """Validated immutable CPU-only contract identity."""

    source_path: Path
    source_sha256: str
    schema_version: str
    run_id: str
    roles: tuple[str, ...]
    seeds: tuple[int, ...]
    rows_per_fit: int
    optimizer_updates_per_fit: int
    launch_authorized: bool
    model_cuda_authorized: bool
    population_materialized: bool

    @classmethod
    def from_json(cls, path: str | Path) -> ActionCorrectionContract:
        source_path = Path(path).expanduser().resolve()
        raw = source_path.read_bytes()
        try:
            payload = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ActionCorrectionContractError(f"invalid contract JSON: {error}") from error
        payload = _object(payload, context="contract")
        _validate_contract_payload(payload)
        training = _object(payload["training"], context="training")
        return cls(
            source_path=source_path,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            schema_version=payload["schema_version"],
            run_id=payload["run_id"],
            roles=tuple(payload["role_order"]),
            seeds=tuple(training["seeds"]),
            rows_per_fit=training["rows_per_fit"],
            optimizer_updates_per_fit=training["optimizer_updates_per_fit"],
            launch_authorized=payload["launch_authorized"],
            model_cuda_authorized=payload["model_cuda_authorized"],
            population_materialized=payload["population_materialized"],
        )


def _validate_contract_payload(payload: dict[str, Any]) -> None:
    top_level = {
        "schema_version",
        "run_id",
        "status",
        "canonical_names",
        "checkpoint_identities",
        "hypothesis",
        "decision",
        "launch_authorized",
        "model_cuda_authorized",
        "population_materialized",
        "jarvislabs_resource_creation_authorized",
        "human_label_access_authorized",
        "distinctness",
        "population_plan",
        "forbidden_population_access",
        "role_order",
        "roles",
        "training",
        "phase_firewall",
        "two_pass_contract",
        "diagnostics_contract",
        "gates",
        "claims",
        "supersession",
    }
    _exact_keys(payload, keys=top_level, context="contract")
    _expect(payload["schema_version"], CONTRACT_SCHEMA_VERSION, context="schema version")
    _expect(payload["run_id"], RUN_ID, context="run ID")
    _expect(payload["status"], "cpu_prefreeze_non_authorizing", context="status")
    for field in (
        "launch_authorized",
        "model_cuda_authorized",
        "population_materialized",
        "jarvislabs_resource_creation_authorized",
        "human_label_access_authorized",
    ):
        _expect(payload[field], False, context=field)

    _expect(
        payload["canonical_names"],
        {
            "base_model": BASE_MODEL_NAME,
            "post_trained_model": POST_TRAINED_MODEL_NAME,
            "base_parameter_count": PARAMETER_COUNT,
        },
        context="canonical names",
    )
    _expect(
        payload["checkpoint_identities"],
        _expected_checkpoint_identities(),
        context="checkpoint identities",
    )
    for field in ("hypothesis", "decision"):
        if type(payload[field]) is not str or not payload[field].strip():
            raise ActionCorrectionContractError(f"{field} must be a non-empty string")
    _expect(payload["distinctness"], _expected_distinctness(), context="distinctness boundary")
    _validate_population_plan(payload["population_plan"])
    _expect(
        payload["forbidden_population_access"],
        _expected_forbidden_access(),
        context="forbidden-population zero-access boundary",
    )
    _expect(payload["role_order"], list(ALL_ROLES), context="role order")
    _expect(payload["roles"], _expected_roles(), context="causal roles")
    _validate_training(payload["training"])
    _validate_phase_firewall(payload["phase_firewall"])
    _validate_two_pass_contract(payload["two_pass_contract"])
    _validate_diagnostics_contract(payload["diagnostics_contract"])
    _expect(payload["gates"], _expected_gates(), context="internal keep gates")
    _expect(
        payload["claims"],
        {
            "breakthrough_claim_authorized": False,
            "candidate_v2_remains_public_release_checkpoint": True,
            "larger_model_superiority_claim_authorized": False,
            "maximum_claim": "hypothesis_fresh_internal_synthetic_mechanism_evidence",
            "public_release_replacement_authorized": False,
        },
        context="claim boundary",
    )
    _expect(
        payload["supersession"],
        {
            "editing_this_config_can_authorize_launch": False,
            "materialized_or_model_experiment_requires_new_config_and_run_id": True,
            "retry_rescue_or_threshold_edit_authorized": False,
        },
        context="supersession boundary",
    )


def _validate_population_plan(value: object) -> None:
    population = _object(value, context="population_plan")
    _exact_keys(
        population,
        keys={
            "name",
            "status",
            "t_synth",
            "t_replay",
            "d_internal",
            "manifest_sha256",
            "requirements",
        },
        context="population_plan",
    )
    _expect(population["name"], "ActionCorrection-Forge-v1", context="population name")
    _expect(population["status"], "planned_not_materialized", context="population status")
    _expect(
        population["t_synth"],
        {
            "nominal_rows_per_registered_stratum": T_SYNTH_ROWS // 16,
            "planned_rows": T_SYNTH_ROWS,
            "registered_descriptive_strata": 16,
            "text_source": "deterministic_program_first_renderer_only",
        },
        context="T-synth plan",
    )
    _expect(
        population["t_replay"],
        {
            "accepted_rows": T_REPLAY_ACCEPTED_ROWS,
            "accepted_membership_order_sha256": REPLAY_ACCEPTED_MEMBERSHIP_SHA256,
            "direct_action_ir_only": True,
            "exclusion_ledger_sha256": REPLAY_EXCLUSION_LEDGER_SHA256,
            "fresh_or_selection_evidence": False,
            "known_bad_label_excluded": True,
            "known_bad_label_id": KNOWN_BAD_REPLAY_ID,
            "source_rows": T_REPLAY_SOURCE_ROWS,
            "source_sha256": MOBILE_CONSTRUCTION_SOURCE_SHA256,
            "source_membership_order_sha256": (MOBILE_CONSTRUCTION_SOURCE_MEMBERSHIP_SHA256),
        },
        context="T-replay boundary",
    )
    _expect(
        population["d_internal"],
        {
            "effective_components_claimed": None,
            "effective_components_verified": False,
            "freshness_scope": "hypothesis_fresh_internal_synthetic_only",
            "independent_or_powered_population_claimed": False,
            "nominal_rows": D_INTERNAL_NOMINAL_ROWS,
            "nominal_rows_per_registered_stratum": D_INTERNAL_NOMINAL_ROWS // 16,
            "old_7937_756_v3_planir_presto_dev_lineage_allowed": False,
            "registered_descriptive_strata": 16,
        },
        context="D-internal boundary",
    )
    _expect(
        population["manifest_sha256"],
        {"d_internal": None, "t_replay": None, "t_synth": None},
        context="unmaterialized manifest hashes",
    )
    _expect(
        population["requirements"],
        {
            "all_requirements_verified": False,
            "component_split_before_render_required": True,
            "d_frozen_before_model_output_required": True,
            "joint_exact_normalized_delexicalized_near_duplicate_closure_required": True,
            "materialization_and_token_audit_required": True,
            "no_human_or_model_authored_text_required": True,
            "role_exclusive_provenance_required": True,
        },
        context="population requirements",
    )


def _validate_training(value: object) -> None:
    training = _object(value, context="training")
    _exact_keys(
        training,
        keys={
            "training_arms",
            "seeds",
            "total_fits",
            "starting_checkpoint",
            "parameter_count_per_fit",
            "full_parameter",
            "shared_replay_rows_per_fit",
            "shared_direct_synth_rows_per_fit",
            "arm_specific_synth_rows_per_fit",
            "c_certified_single_fault_draft_rows_per_fit",
            "c_exact_draft_rows_per_fit",
            "rows_per_fit",
            "epochs",
            "batch_size",
            "drop_last",
            "last_batch_rows",
            "optimizer_updates_per_fit",
            "budget_requirements",
            "preview_checkpoint_policy",
            "optimization",
        },
        context="training",
    )
    _expect(training["training_arms"], list(TRAINING_ARMS), context="training arms")
    _expect(training["seeds"], list(SEEDS), context="training seeds")
    _expect(training["total_fits"], TOTAL_FITS, context="total fit count")
    _expect(
        training["starting_checkpoint"],
        "BarunAction-35M candidate-v2",
        context="starting checkpoint",
    )
    _expect(training["parameter_count_per_fit"], PARAMETER_COUNT, context="parameter count")
    _expect(training["full_parameter"], True, context="full-parameter training")
    _expect(
        training["shared_replay_rows_per_fit"],
        T_REPLAY_ACCEPTED_ROWS,
        context="shared replay rows",
    )
    _expect(
        training["shared_direct_synth_rows_per_fit"],
        T_SYNTH_ROWS,
        context="shared direct rows",
    )
    _expect(
        training["arm_specific_synth_rows_per_fit"],
        T_SYNTH_ROWS,
        context="arm-specific rows",
    )
    _expect(
        training["c_certified_single_fault_draft_rows_per_fit"],
        T_SYNTH_ROWS // 2,
        context="certified fault-draft rows",
    )
    _expect(
        training["c_exact_draft_rows_per_fit"],
        T_SYNTH_ROWS // 2,
        context="exact draft rows",
    )
    _expect(
        training["c_certified_single_fault_draft_rows_per_fit"]
        + training["c_exact_draft_rows_per_fit"],
        T_SYNTH_ROWS,
        context="clean/fault split",
    )
    expected_rows = T_REPLAY_ACCEPTED_ROWS + 2 * T_SYNTH_ROWS
    _expect(training["rows_per_fit"], expected_rows, context="rows per fit")
    _expect(training["rows_per_fit"], ROWS_PER_FIT, context="rows per fit")
    _expect(training["epochs"], 1, context="epoch count")
    _expect(training["batch_size"], BATCH_SIZE, context="batch size")
    _expect(training["drop_last"], False, context="drop-last policy")
    expected_updates = math.ceil(ROWS_PER_FIT / BATCH_SIZE)
    expected_last = ROWS_PER_FIT - BATCH_SIZE * (expected_updates - 1)
    _expect(training["optimizer_updates_per_fit"], expected_updates, context="updates per fit")
    _expect(training["optimizer_updates_per_fit"], UPDATES_PER_FIT, context="updates per fit")
    _expect(training["last_batch_rows"], expected_last, context="last batch rows")
    _expect(training["last_batch_rows"], LAST_BATCH_ROWS, context="last batch rows")
    _expect(
        training["budget_requirements"],
        {
            "input_compute_matched_claimed": False,
            "same_complete_gold_target_by_source_required": True,
            "same_example_count_required": True,
            "same_optimizer_updates_required": True,
            "same_source_ids_required": True,
            "same_supervised_target_token_total_required": True,
            "clean_fault_split_verification_passed": False,
            "every_fault_draft_single_fault_certificate_required": True,
            "fault_certification_verification_passed": False,
            "source_and_target_budget_verification_passed": False,
        },
        context="unverified matched-budget requirements",
    )
    _expect(
        training["preview_checkpoint_policy"],
        {
            "best_seed_selection_authorized": False,
            "maximum_total_fits": 9,
            "post_D_refit_authorized": False,
            "predesignated_preview_seed": 17,
            "replication_only_seeds": [29, 43],
            "tenth_fit_authorized": False,
            "weight_averaging_authorized": False,
        },
        context="preview checkpoint policy",
    )
    _expect(
        training["optimization"],
        {
            "adam_epsilon": 1e-8,
            "attention_backend": "deterministic_math_sdpa_only",
            "beta1": 0.9,
            "beta2": 0.95,
            "checkpoint_policy": "unconditional_final_only",
            "decoding": "unconstrained_greedy",
            "early_stopping": False,
            "gradient_accumulation_steps": 1,
            "gradient_clip_norm": 1.0,
            "learning_rate": 5e-5,
            "loss": "response_only_cross_entropy",
            "max_seq_len": 2048,
            "min_learning_rate_ratio": 0.1,
            "packing": False,
            "precision": "bf16",
            "scheduler": "cosine",
            "warmup_steps": 18,
            "weight_decay": 0.1,
        },
        context="optimization recipe",
    )


def _validate_phase_firewall(value: object) -> None:
    _expect(
        value,
        {
            "B0_evaluation_requires_frozen_D_and_phase_open": True,
            "D_access_before_all_nine_final_checkpoints_frozen": False,
            "D_loss_used_for_training_early_stop_or_checkpoint_choice": False,
            "all_nine_checkpoints_frozen_before_first_D_decode": True,
            "phase_requirements_verified": False,
        },
        context="phase firewall",
    )


def _validate_two_pass_contract(value: object) -> None:
    _expect(
        value,
        {
            "canonical_abstain": CANONICAL_ABSTAIN,
            "decoding_caps_required_before_model_authorization": True,
            "decoding_caps_resolved": False,
            "execution_permitted": False,
            "fallback_order": ["pass2_valid", "pass1_valid", "canonical_abstain"],
            "maximum_passes": 2,
            "pass1_max_new_tokens": None,
            "pass2_max_new_tokens": None,
            "real_tool_execution": False,
            "retry_count": 0,
            "same_checkpoint_for_both_passes": True,
            "two_pass_roles": ["G02", "A2", "C2"],
            "validity_predicate": [
                "parse_valid",
                "schema_valid",
                "policy_conformant_proposal",
            ],
        },
        context="two-pass contract",
    )


def _validate_diagnostics_contract(value: object) -> None:
    _expect(
        value,
        {
            "allowed_fields": list(DIAGNOSTIC_FIELDS),
            "derivation": "draft_only_recomputed_by_pinned_runtime",
            "fault_certificate_model_visible": False,
            "forbidden_semantic_sources": list(_FORBIDDEN_DIAGNOSTIC_TERMS),
            "gold_derived_diagnostics_allowed": False,
            "runtime_recomputation_verified": False,
        },
        context="diagnostics contract",
    )


@dataclass(frozen=True, slots=True)
class DraftDiagnostics:
    """The complete model-visible diagnostic vocabulary for one draft."""

    parse_valid: bool
    schema_valid: bool
    policy_status: str
    simulator_status: str
    error_codes: tuple[str, ...]


def validate_draft_diagnostics(value: object) -> DraftDiagnostics:
    """Validate diagnostics without accepting a gold, target, label, or oracle field."""

    payload = _object(value, context="draft diagnostics")
    _exact_keys(payload, keys=set(DIAGNOSTIC_FIELDS), context="draft diagnostics")
    parse_valid = payload["parse_valid"]
    schema_valid = payload["schema_valid"]
    if type(parse_valid) is not bool or type(schema_valid) is not bool:
        raise ActionCorrectionContractError("diagnostic validity fields must be booleans")
    if schema_valid and not parse_valid:
        raise ActionCorrectionContractError("schema-valid diagnostics must also be parse-valid")
    policy_status = payload["policy_status"]
    simulator_status = payload["simulator_status"]
    if type(policy_status) is not str or policy_status not in _POLICY_STATUSES:
        raise ActionCorrectionContractError("unknown draft-only policy status")
    if type(simulator_status) is not str or simulator_status not in _SIMULATOR_STATUSES:
        raise ActionCorrectionContractError("unknown draft-only simulator status")
    error_codes = payload["error_codes"]
    if type(error_codes) is not list:
        raise ActionCorrectionContractError("diagnostic error_codes must be an exact JSON list")
    checked: list[str] = []
    for code in error_codes:
        if type(code) is not str or _ERROR_CODE.fullmatch(code) is None:
            raise ActionCorrectionContractError("diagnostic error code has invalid syntax")
        if any(term in code for term in _FORBIDDEN_DIAGNOSTIC_TERMS):
            raise ActionCorrectionContractError("diagnostic error code implies a forbidden source")
        checked.append(code)
    if checked != sorted(set(checked)):
        raise ActionCorrectionContractError("diagnostic error codes must be sorted and unique")
    return DraftDiagnostics(
        parse_valid=parse_valid,
        schema_valid=schema_valid,
        policy_status=policy_status,
        simulator_status=simulator_status,
        error_codes=tuple(checked),
    )


@dataclass(frozen=True, slots=True)
class PassOutput:
    """Only label-free proposal-conformance evidence needed by the fallback.

    Policy conformance validates proposal shape. It never permits execution.
    """

    raw: str | None
    parse_valid: bool
    schema_valid: bool
    policy_conformant_proposal: bool

    def __post_init__(self) -> None:
        for name in ("parse_valid", "schema_valid", "policy_conformant_proposal"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.raw is not None and (type(self.raw) is not str or not self.raw):
            raise ValueError("raw must be None or a non-empty string")
        if self.schema_valid and not self.parse_valid:
            raise ValueError("schema-valid output must be parse-valid")
        if self.policy_conformant_proposal and not self.schema_valid:
            raise ValueError("policy-conformant proposal must be schema-valid")
        if self.raw is None and (
            self.parse_valid or self.schema_valid or self.policy_conformant_proposal
        ):
            raise ValueError("a missing output cannot carry passing validity evidence")

    @property
    def eligible(self) -> bool:
        return bool(
            self.raw is not None
            and self.parse_valid
            and self.schema_valid
            and self.policy_conformant_proposal
        )


@dataclass(frozen=True, slots=True)
class SelectedPassOutput:
    raw: str
    source: str


def select_two_pass_output(
    *,
    pass1: PassOutput,
    pass2: PassOutput,
) -> SelectedPassOutput:
    """Apply the frozen pass-2, pass-1, then canonical-abstain fallback."""

    if type(pass1) is not PassOutput or type(pass2) is not PassOutput:
        raise TypeError("fallback inputs must be exact PassOutput instances")
    if pass2.eligible:
        assert pass2.raw is not None
        return SelectedPassOutput(raw=pass2.raw, source="pass2")
    if pass1.eligible:
        assert pass1.raw is not None
        return SelectedPassOutput(raw=pass1.raw, source="pass1")
    return SelectedPassOutput(raw=CANONICAL_ABSTAIN, source="canonical_abstain")


__all__ = [
    "ALL_ROLES",
    "BASE_MODEL_NAME",
    "BATCH_SIZE",
    "CANONICAL_ABSTAIN",
    "CONTRACT_SCHEMA_VERSION",
    "DIAGNOSTIC_FIELDS",
    "D_INTERNAL_NOMINAL_ROWS",
    "KNOWN_BAD_REPLAY_ID",
    "LAST_BATCH_ROWS",
    "MOBILE_CONSTRUCTION_SOURCE_MEMBERSHIP_SHA256",
    "MOBILE_CONSTRUCTION_SOURCE_SHA256",
    "PARAMETER_COUNT",
    "POST_TRAINED_MODEL_NAME",
    "REPLAY_ACCEPTED_MEMBERSHIP_SHA256",
    "REPLAY_EXCLUSION_LEDGER_SHA256",
    "ROWS_PER_FIT",
    "RUN_ID",
    "SEEDS",
    "TOTAL_FITS",
    "TRAINING_ARMS",
    "T_REPLAY_ACCEPTED_ROWS",
    "T_REPLAY_SOURCE_ROWS",
    "T_SYNTH_ROWS",
    "UPDATES_PER_FIT",
    "ActionCorrectionContract",
    "ActionCorrectionContractError",
    "DraftDiagnostics",
    "PassOutput",
    "SelectedPassOutput",
    "select_two_pass_output",
    "validate_draft_diagnostics",
]
