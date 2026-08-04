from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from math import comb

import pytest

from barunlm.evaluation import gvs_power
from barunlm.evaluation.gvs_power import (
    GVS_POWER_ASSUMPTIONS_SCHEMA_VERSION,
    GVS_POWER_REPORT_SCHEMA_VERSION,
    GVS_POWER_RUNTIME_SELF_TEST_SCHEMA_VERSION,
    ONE_SAMPLE_KIND,
    PAIRED_KIND,
    GVSPowerError,
    build_power_report,
    exact_critical_successes,
    load_power_assumptions_json,
    load_power_report_json,
    verify_power_report,
)


def _ratio(numerator: int, denominator: int) -> dict[str, int]:
    return {"denominator": denominator, "numerator": numerator}


def _probability_fraction(value: dict[str, object]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _assumptions() -> dict[str, object]:
    return {
        "assumption_sources": [
            {
                "precollection": True,
                "role": "external_or_historical_precollection_assumption",
                "sha256": "1" * 64,
                "source_id": "historical-audit",
            }
        ],
        "authorizes_model_or_label_access": False,
        "component_rosters": [
            {
                "component_count": 10,
                "eligible_component_ids_sha256": "4" * 64,
                "eligible_only": True,
                "outcomes_absent": True,
                "population_firewall_report_sha256": "5" * 64,
                "population_role": "D-support",
                "preoutcome_frozen": True,
                "roster_id": "d-support-all",
                "scope": "D-support/all-components",
                "stratum_id": "all-components",
                "unit": "gvs_firewall_connected_component",
            },
            {
                "component_count": 80,
                "eligible_component_ids_sha256": "6" * 64,
                "eligible_only": True,
                "outcomes_absent": True,
                "population_firewall_report_sha256": "7" * 64,
                "population_role": "S-new",
                "preoutcome_frozen": True,
                "roster_id": "s-new-efficacy",
                "scope": "S-new/efficacy-components",
                "stratum_id": "efficacy-components",
                "unit": "gvs_firewall_connected_component",
            },
        ],
        "design_id": "gvs-power-test-v1",
        "endpoints": [
            {
                "alternative_direction": "candidate_greater",
                "alternative_rate": _ratio(4, 5),
                "component_roster_id": "d-support-all",
                "endpoint_id": "oracle-support",
                "kind": ONE_SAMPLE_KIND,
                "metric_id": "oracle-pass-at-8",
                "null_rate": _ratio(1, 2),
                "reference_id": "fixed-null-rate",
                "source_id": "historical-audit",
            },
            {
                "alternative_direction": "candidate_greater",
                "assumed_discordance": _ratio(1, 2),
                "assumed_gain": _ratio(1, 4),
                "component_roster_id": "s-new-efficacy",
                "endpoint_id": "selection-vs-greedy",
                "kind": PAIRED_KIND,
                "metric_id": "exact-execution",
                "reference_id": "frozen-greedy",
                "source_id": "historical-audit",
            },
        ],
        "hypothesis_roster": {
            "complete_for_protocol": True,
            "endpoint_ids": ["oracle-support", "selection-vs-greedy"],
            "family_id": "gvs-primary",
            "protocol_sha256": "3" * 64,
        },
        "independence_contract": {
            "cross_role_component_closure_required": True,
            "effective_n_equals_eligible_components": True,
            "unit": "gvs_firewall_connected_component",
        },
        "multiplicity": {
            "endpoint_count": 2,
            "family_id": "gvs-primary",
            "familywise_alpha": _ratio(1, 20),
            "method": "bonferroni",
        },
        "outcome_access_prohibited": True,
        "population_protocol_sha256": "2" * 64,
        "protocol_sha256": "3" * 64,
        "schema_version": GVS_POWER_ASSUMPTIONS_SCHEMA_VERSION,
        "target_power": _ratio(4, 5),
        "zero_post_freeze_attrition": True,
    }


def test_exact_binomial_cutoff_matches_known_small_vector() -> None:
    # Under Binomial(10, 0.5), P[X>=8]=56/1024 > .05 and P[X>=9]=11/1024 <= .05.
    assert (
        exact_critical_successes(
            trials=10,
            null_probability=Fraction(1, 2),
            one_sided_alpha=Fraction(1, 20),
        )
        == 9
    )


def test_exact_binomial_cutoffs_match_independent_fraction_enumeration() -> None:
    for trials in range(1, 21):
        for probability in (Fraction(1, 10), Fraction(1, 3), Fraction(1, 2), Fraction(9, 10)):
            denominator = probability.denominator**trials
            masses = [
                comb(trials, successes)
                * probability.numerator**successes
                * (probability.denominator - probability.numerator) ** (trials - successes)
                for successes in range(trials + 1)
            ]
            for alpha in (Fraction(1, 100), Fraction(1, 20), Fraction(1, 10)):
                expected = next(
                    cutoff
                    for cutoff in range(trials + 2)
                    if sum(masses[cutoff:]) * alpha.denominator <= alpha.numerator * denominator
                )
                assert (
                    exact_critical_successes(
                        trials=trials,
                        null_probability=probability,
                        one_sided_alpha=alpha,
                    )
                    == expected
                )


def test_report_is_deterministic_bound_and_explicitly_nonauthorizing() -> None:
    assumptions = _assumptions()
    first = build_power_report(assumptions)
    second = build_power_report(copy.deepcopy(assumptions))
    assert first == second
    assert first["schema_version"] == GVS_POWER_REPORT_SCHEMA_VERSION
    assert first["launch_authorized"] is False
    assert first["authorizes_model_or_label_access"] is False
    assert first["authorizes_cuda_or_jarvis_access"] is False
    assert first["structural_planning_only"] is True
    assert first["requires_authenticated_one_shot_custody_receipt"] is True
    assert first["requires_independent_power_review"] is True
    assert first["outcome_blindness_authenticated"] is False
    assert first["assumption_sources_are_caller_attested_only"] is True
    assert first["component_roster_authentication_verified"] is False
    assert first["hypothesis_roster_completeness_authenticated"] is False
    assert first["endpoint_alpha"] == _ratio(1, 40)
    assert len(first["report_sha256"]) == 64
    assert len(first["runtime_identity"]["runtime_identity_sha256"]) == 64
    assert first["runtime_identity"]["arithmetic_contract"].startswith("exact Python")
    dependencies = first["runtime_identity"]["behavior_dependencies"]
    assert dependencies["dependency_count"] == 21
    assert len(dependencies["dependency_roster_sha256"]) == 64
    assert [record["dependency_id"] for record in dependencies["dependencies"]] == [
        "decimal.Decimal",
        "decimal.localcontext",
        "fractions.Fraction",
        "hashlib.sha256",
        "json.JSONDecodeError",
        "json.dumps",
        "json.loads",
        "math.comb",
        "math.gcd",
        "math.isfinite",
        "math.lcm",
        "os.fstat",
        "pathlib.Path",
        "platform.machine",
        "platform.python_implementation",
        "platform.python_version",
        "platform.release",
        "platform.system",
        "types.CodeType",
        "unicodedata.category",
        "unicodedata.normalize",
    ]
    behavior_constants = first["runtime_identity"]["behavior_constants"]
    assert len(behavior_constants["behavior_constant_sha256"]) == 64
    assert behavior_constants["component_unit"] == "gvs_firewall_connected_component"
    assert behavior_constants["limits"]["paired_components"] == 4_000
    exact_self_test = first["runtime_identity"]["exact_runtime_self_test"]
    assert exact_self_test["schema_version"] == GVS_POWER_RUNTIME_SELF_TEST_SCHEMA_VERSION
    assert len(exact_self_test["vectors_sha256"]) == 64
    assert exact_self_test["vectors"]["math_comb_10_3"] == 120
    assert exact_self_test["vectors"]["paired_n10_d1_2_g1_4_alpha1_40"] == {
        "denominator": 1_073_741_824,
        "numerator": 64_116_279,
    }
    assert [row["endpoint_id"] for row in first["endpoints"]] == [
        "oracle-support",
        "selection-vs-greedy",
    ]
    assumptions["component_rosters"][0]["component_count"] = 999
    assumptions["hypothesis_roster"]["endpoint_ids"].clear()
    assert first["component_rosters"][0]["component_count"] == 10
    assert first["hypothesis_roster"]["endpoint_ids"] == [
        "oracle-support",
        "selection-vs-greedy",
    ]
    assert first["hypothesis_roster"]["hypotheses"][1] == {
        "alternative_direction": "candidate_greater",
        "component_roster_id": "s-new-efficacy",
        "endpoint_id": "selection-vs-greedy",
        "kind": PAIRED_KIND,
        "metric_id": "exact-execution",
        "planning_assumptions": {
            "assumed_discordance": _ratio(1, 2),
            "assumed_gain": _ratio(1, 4),
        },
        "reference_id": "frozen-greedy",
        "source_id": "historical-audit",
    }


def test_one_sample_report_retains_exact_tail_and_power() -> None:
    report = build_power_report(_assumptions())
    result = report["endpoints"][0]["calculation"]
    assert result["critical_successes"] == 9
    assert result["attainable_rejection_boundary"] is True
    assert result["null_tail_at_boundary"]["numerator"] == "11"
    assert result["null_tail_at_boundary"]["denominator"] == "1024"
    assert result["meets_planning_power_target"] is False


def test_paired_report_uses_exact_boundaries_and_preserves_mass() -> None:
    report = build_power_report(_assumptions())
    result = report["endpoints"][1]["calculation"]
    assert result["test"] == "exact conditional one-sided McNemar boundaries"
    assert len(result["critical_boundaries_sha256"]) == 64
    assert result["represented_probability_mass"]["numerator"] == "1"
    assert result["represented_probability_mass"]["denominator"] == "1"
    assert result["arithmetic"] == "exact integer multinomial common denominator"
    assert result["assumed_candidate_only_rate"] == _ratio(3, 8)
    assert result["assumed_reference_only_rate"] == _ratio(1, 8)
    assert result["assumed_tie_rate"] == _ratio(1, 2)


def test_paired_power_matches_independent_exact_small_vector() -> None:
    assumptions = _assumptions()
    assumptions["component_rosters"][1]["component_count"] = 10
    report = build_power_report(assumptions)
    power = _probability_fraction(report["endpoints"][1]["calculation"]["alternative_power"])
    assert power == Fraction(64_116_279, 1_073_741_824)


def test_paired_power_does_not_underflow_for_near_certain_candidate_wins() -> None:
    assumptions = _assumptions()
    assumptions["component_rosters"][1]["component_count"] = 500
    assumptions["endpoints"][1]["assumed_discordance"] = _ratio(1, 2)
    assumptions["endpoints"][1]["assumed_gain"] = _ratio(499_999, 1_000_000)
    report = build_power_report(assumptions)
    calculation = report["endpoints"][1]["calculation"]
    power = _probability_fraction(calculation["alternative_power"])
    # The exact subset {reference-only=0, candidate-only>=6} already exceeds this bound.
    assert power > Fraction(99_975, 100_000)
    assert calculation["meets_planning_power_target"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"observed_outcomes": []}), "fields changed"),
        (
            lambda value: value.update({"authorizes_model_or_label_access": True}),
            "must be false",
        ),
        (lambda value: value.update({"outcome_access_prohibited": False}), "must be true"),
        (
            lambda value: value["multiplicity"].update({"familywise_alpha": _ratio(1, 10)}),
            "at most one twentieth",
        ),
        (
            lambda value: value.update({"target_power": _ratio(3, 4)}),
            "four fifths",
        ),
        (
            lambda value: value["multiplicity"].update({"endpoint_count": 1}),
            "exactly multiplicity.endpoint_count",
        ),
        (lambda value: value["endpoints"].reverse(), "canonical endpoint_id order"),
        (
            lambda value: value["endpoints"][0].update({"source_id": "missing"}),
            "unknown assumption source",
        ),
        (
            lambda value: value["independence_contract"].update({"unit": "row"}),
            "firewall connected component",
        ),
        (
            lambda value: value["component_rosters"][0].update({"outcomes_absent": False}),
            "outcomes_absent must be true",
        ),
        (
            lambda value: value["hypothesis_roster"].update({"complete_for_protocol": False}),
            "complete_for_protocol must be true",
        ),
        (
            lambda value: value["endpoints"][0].update({"component_roster_id": "missing"}),
            "unknown component roster",
        ),
        (
            lambda value: value["endpoints"][0].update({"null_rate": _ratio(4, 5)}),
            "null < alternative",
        ),
        (
            lambda value: value["endpoints"][1].update({"assumed_gain": _ratio(1, 2)}),
            "strictly between zero and discordance",
        ),
    ],
)
def test_report_fails_closed_on_contract_changes(mutation, message: str) -> None:
    assumptions = _assumptions()
    mutation(assumptions)
    with pytest.raises(GVSPowerError, match=message):
        build_power_report(assumptions)


def test_rejects_noncanonical_ratios_and_boolean_integer_confusion() -> None:
    assumptions = _assumptions()
    assumptions["target_power"] = _ratio(8, 10)
    with pytest.raises(GVSPowerError, match="lowest terms"):
        build_power_report(assumptions)

    assumptions = _assumptions()
    assumptions["component_rosters"][0]["component_count"] = True
    with pytest.raises(GVSPowerError, match="component_count"):
        build_power_report(assumptions)


def test_strict_loader_rejects_duplicates_nonfinite_and_oversize() -> None:
    raw = json.dumps(_assumptions(), sort_keys=True).encode("utf-8")
    assert load_power_assumptions_json(raw) == json.loads(raw)
    with pytest.raises(GVSPowerError, match="duplicate JSON key"):
        load_power_assumptions_json(b'{"schema_version":"x","schema_version":"y"}')
    with pytest.raises(GVSPowerError, match="non-finite"):
        load_power_assumptions_json(b'{"x":NaN}')
    with pytest.raises(GVSPowerError, match="oversized"):
        load_power_assumptions_json(b"{" + b" " * (2 * 1024 * 1024))


def test_custom_mapping_and_container_cycles_are_rejected() -> None:
    class CustomMapping(dict):
        pass

    with pytest.raises(GVSPowerError, match="exact built-in JSON"):
        build_power_report(CustomMapping(_assumptions()))

    assumptions = _assumptions()
    assumptions["cycle"] = assumptions
    with pytest.raises(GVSPowerError, match="cycle"):
        build_power_report(assumptions)


def test_report_loader_and_recomputation_close_stale_and_rehashed_tampering() -> None:
    assumptions = _assumptions()
    report = build_power_report(assumptions)
    loaded = load_power_report_json(_canonical_json(report))
    assert verify_power_report(loaded, assumptions) == report

    stale = copy.deepcopy(report)
    stale["launch_authorized"] = True
    with pytest.raises(GVSPowerError, match="SHA-256 does not match"):
        verify_power_report(stale, assumptions)

    rehashed = copy.deepcopy(report)
    rehashed["endpoints"][0]["calculation"]["meets_planning_power_target"] = True
    rehashed_body = dict(rehashed)
    del rehashed_body["report_sha256"]
    rehashed["report_sha256"] = hashlib.sha256(_canonical_json(rehashed_body)).hexdigest()
    with pytest.raises(GVSPowerError, match="exact recomputation"):
        verify_power_report(rehashed, assumptions)


def test_report_recomputation_binds_component_and_hypothesis_rosters() -> None:
    assumptions = _assumptions()
    report = build_power_report(assumptions)
    changed = copy.deepcopy(assumptions)
    changed["component_rosters"][0]["eligible_component_ids_sha256"] = "8" * 64
    with pytest.raises(GVSPowerError, match="exact recomputation"):
        verify_power_report(report, changed)
    assert report["hypothesis_roster"]["endpoint_ids"] == [
        "oracle-support",
        "selection-vs-greedy",
    ]
    assert report["component_rosters"][0]["component_count"] == 10


def test_runtime_identity_change_during_calculation_fails_closed(monkeypatch) -> None:
    stable = gvs_power._runtime_identity_snapshot()
    calls = 0

    def changing_identity() -> dict[str, object]:
        nonlocal calls
        calls += 1
        result = copy.deepcopy(stable)
        if calls > 1:
            result["python_version"] = "changed-during-calculation"
        return result

    monkeypatch.setattr(gvs_power, "_runtime_identity_snapshot", changing_identity)
    with pytest.raises(GVSPowerError, match="changed during calculation"):
        build_power_report(_assumptions())


def test_import_time_capture_closes_math_comb_monkeypatch_exploit(monkeypatch) -> None:
    monkeypatch.setattr(gvs_power.math, "comb", lambda _n, _k: 0)
    with pytest.raises(GVSPowerError, match="dependency binding changed: math.comb"):
        build_power_report(_assumptions())


def test_exact_formula_avoids_fraction_transitive_math_gcd_lookup(monkeypatch) -> None:
    arguments = {
        "component_count": 10,
        "assumed_discordance": Fraction(1, 2),
        "assumed_gain": Fraction(1, 4),
        "endpoint_alpha": Fraction(1, 40),
        "target_power": Fraction(4, 5),
    }
    baseline = gvs_power._paired_result(**arguments)
    monkeypatch.setattr(gvs_power.math, "gcd", lambda _a, _b: 1)
    assert gvs_power._paired_result(**arguments) == baseline


@pytest.mark.parametrize(
    ("module", "attribute", "replacement", "dependency_id"),
    [
        (gvs_power.math, "gcd", lambda _a, _b: 1, "math.gcd"),
        (gvs_power.math, "lcm", lambda _a, _b, *_rest: 1, "math.lcm"),
        (gvs_power.hashlib, "sha256", lambda _value=b"": None, "hashlib.sha256"),
        (gvs_power.json, "dumps", lambda *_args, **_kwargs: "{}", "json.dumps"),
        (
            gvs_power.unicodedata,
            "normalize",
            lambda *_args: "changed",
            "unicodedata.normalize",
        ),
    ],
)
def test_originating_module_dependency_tampering_fails_closed(
    monkeypatch,
    module,
    attribute: str,
    replacement,
    dependency_id: str,
) -> None:
    monkeypatch.setattr(module, attribute, replacement)
    with pytest.raises(GVSPowerError, match=f"dependency binding changed: {dependency_id}"):
        build_power_report(_assumptions())


@pytest.mark.parametrize(
    ("dependency_name", "replacement"),
    [
        ("_MATH_COMB", lambda _n, _k: 0),
        ("_MATH_GCD", lambda _a, _b: 1),
        ("_MATH_LCM", lambda _a, _b, *_rest: 1),
        ("_JSON_DUMPS", lambda *_args, **_kwargs: "{}"),
        ("_FRACTION_TYPE", lambda *_args: Fraction(0, 1)),
    ],
)
def test_captured_dependency_tampering_fails_closed(
    monkeypatch,
    dependency_name: str,
    replacement,
) -> None:
    monkeypatch.setattr(gvs_power, dependency_name, replacement)
    with pytest.raises(GVSPowerError, match="import-time dependency binding changed"):
        build_power_report(_assumptions())


def test_exact_self_test_rejects_loaded_arithmetic_function_replacement(monkeypatch) -> None:
    monkeypatch.setattr(
        gvs_power,
        "_weighted_binomial_upper_tail_integer",
        lambda *_args, **_kwargs: 0,
    )
    with pytest.raises(GVSPowerError, match="exact runtime dependency self-test failed"):
        build_power_report(_assumptions())


@pytest.mark.parametrize("value", ["power\tobserved", "power\u202eobserved"])
def test_identifiers_reject_embedded_whitespace_and_format_controls(value: str) -> None:
    assumptions = _assumptions()
    assumptions["design_id"] = value
    with pytest.raises(GVSPowerError, match="forbidden whitespace or control"):
        build_power_report(assumptions)


def test_ratio_and_public_fraction_inputs_have_strict_complexity_bounds() -> None:
    assumptions = _assumptions()
    assumptions["endpoints"][1]["assumed_discordance"] = _ratio(1, 10**400)
    assumptions["endpoints"][1]["assumed_gain"] = _ratio(1, 2 * 10**400)
    with pytest.raises(GVSPowerError, match="fit in 64 bits"):
        build_power_report(assumptions)

    with pytest.raises(GVSPowerError, match="fractions.Fraction"):
        exact_critical_successes(
            trials=10,
            null_probability=0.5,
            one_sided_alpha=Fraction(1, 20),
        )
    with pytest.raises(GVSPowerError, match="fractions.Fraction"):
        exact_critical_successes(
            trials=10,
            null_probability=Fraction(1, 2),
            one_sided_alpha=0.05,
        )


def test_exact_probability_serialization_does_not_depend_on_python_digit_limit() -> None:
    assumptions = _assumptions()
    assumptions["component_rosters"][0]["component_count"] = 1_500
    assumptions["endpoints"][0]["null_rate"] = _ratio(1, 1_000)
    assumptions["endpoints"][0]["alternative_rate"] = _ratio(2, 1_001)
    report = build_power_report(assumptions)
    denominator_text = report["endpoints"][0]["calculation"]["alternative_power"]["denominator"]
    assert len(denominator_text) > 4_300
    assert denominator_text.isdecimal()


def test_loader_normalizes_excessive_integer_digits_to_power_error() -> None:
    raw = b'{"x":' + b"1" * 5_000 + b"}"
    with pytest.raises(GVSPowerError, match="not strict UTF-8 JSON"):
        load_power_assumptions_json(raw)

    deeply_nested = b"[" * 2_000 + b"0" + b"]" * 2_000
    with pytest.raises(GVSPowerError, match="not strict UTF-8 JSON|maximum JSON depth"):
        load_power_assumptions_json(deeply_nested)


def test_paired_component_and_aggregate_work_limits_fail_before_calculation() -> None:
    assumptions = _assumptions()
    assumptions["component_rosters"][1]["component_count"] = 4_001
    with pytest.raises(GVSPowerError, match="paired endpoint component_count exceeds"):
        build_power_report(assumptions)

    assumptions = _assumptions()
    assumptions["component_rosters"][0]["component_count"] = 3_500
    assumptions["component_rosters"][1]["component_count"] = 3_500
    assumptions["endpoints"][0] = {
        "alternative_direction": "candidate_greater",
        "assumed_discordance": _ratio(1, 2),
        "assumed_gain": _ratio(1, 10),
        "component_roster_id": "d-support-all",
        "endpoint_id": "oracle-support",
        "kind": PAIRED_KIND,
        "metric_id": "oracle-pass-at-8",
        "reference_id": "fixed-greedy",
        "source_id": "historical-audit",
    }
    with pytest.raises(GVSPowerError, match="aggregate exact-calculation work budget"):
        build_power_report(assumptions)

    assumptions = _assumptions()
    assumptions["component_rosters"][1]["component_count"] = 2_000
    assumptions["endpoints"][1]["assumed_discordance"] = _ratio(1, 2**63 - 25)
    assumptions["endpoints"][1]["assumed_gain"] = _ratio(1, 2**63 - 1)
    with pytest.raises(GVSPowerError, match="aggregate exact-calculation work budget"):
        build_power_report(assumptions)


def test_aggregate_exact_integer_output_budget_fails_before_calculation() -> None:
    assumptions = _assumptions()
    denominator = 2**63 - 25
    for roster in assumptions["component_rosters"]:
        roster["component_count"] = 10_000
    for endpoint, roster_id, endpoint_id, metric_id, reference_id in (
        (
            assumptions["endpoints"][0],
            "d-support-all",
            "oracle-support",
            "oracle-pass-at-8",
            "fixed-null-rate",
        ),
        (
            assumptions["endpoints"][1],
            "s-new-efficacy",
            "selection-vs-greedy",
            "exact-execution",
            "fixed-null-rate-2",
        ),
    ):
        endpoint.clear()
        endpoint.update(
            {
                "alternative_direction": "candidate_greater",
                "alternative_rate": _ratio(2, denominator),
                "component_roster_id": roster_id,
                "endpoint_id": endpoint_id,
                "kind": ONE_SAMPLE_KIND,
                "metric_id": metric_id,
                "null_rate": _ratio(1, denominator),
                "reference_id": reference_id,
                "source_id": "historical-audit",
            }
        )
    with pytest.raises(GVSPowerError, match="exact-integer report budget"):
        build_power_report(assumptions)


def test_structured_hypotheses_cannot_duplicate_or_hide_rosters() -> None:
    assumptions = _assumptions()
    assumptions["endpoints"][1]["component_roster_id"] = "d-support-all"
    assumptions["endpoints"][1]["metric_id"] = "oracle-pass-at-8"
    assumptions["endpoints"][1]["reference_id"] = "fixed-null-rate"
    with pytest.raises(GVSPowerError, match="every component roster"):
        build_power_report(assumptions)

    assumptions = _assumptions()
    assumptions["hypothesis_roster"]["endpoint_ids"] = ["oracle-support"]
    with pytest.raises(GVSPowerError, match="exactly multiplicity.endpoint_count"):
        build_power_report(assumptions)


def test_stable_loaded_calculation_replacement_cannot_rehash_itself_as_attested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The runtime self-test exercises the exact primitives but not every endpoint
    # report builder. A stable replacement must still differ from import-time code.
    monkeypatch.setattr(
        gvs_power,
        "_one_sample_result",
        lambda **_kwargs: {"meets_planning_power_target": True},
    )
    with pytest.raises(GVSPowerError, match="loaded code changed after import"):
        build_power_report(_assumptions())
