from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from barunlm.evaluation import mobile_temporal_counterfactual as temporal
from barunlm.evaluation.evaluator import EVALUATOR_VERSION
from barunlm.evaluation.mobile_actions import (
    MOBILE_ACTIONS_SCORER_VERSION,
)
from barunlm.evaluation.mobile_actions import (
    score_rows as score_mobile_rows,
)

SEEDS = (17, 29, 43)
CONTROL_A = "C_minus_A"
CONTROL_B = "C_minus_B"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _calendar(value: str, *, title: str = "Review") -> dict[str, Any]:
    return {
        "args": {"datetime": value, "title": title},
        "tool": "create_calendar_event",
    }


def _map(query: str = "coffee") -> dict[str, Any]:
    return {"args": {"query": query}, "tool": "search_maps"}


def _action(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "calls": calls,
        "decision": "CALL",
        "mode": "SINGLE" if len(calls) == 1 else "SERIAL",
    }


def _manifest(
    sample_id: str,
    calls: list[dict[str, Any]],
    *,
    now: str = "2026-08-31T12:00:00",
    cluster_id: str | None = None,
) -> dict[str, Any]:
    prompt = (
        "<bos><system>\nACTION_IR_V1\n"
        f"NOW {now}\n"
        "TOOLS\ncreate_calendar_event(datetime:string!, title:string!): Create event\n"
        "search_maps(query:string!): Search maps\n<user>\nDo it.\n<assistant>\n"
    )
    target = _canonical(_action(calls))
    return {
        "id": sample_id,
        "prompt": prompt,
        "target": target,
        "metadata": {
            "cluster_id": cluster_id or f"cluster-{sample_id}",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
        },
    }


def _score(
    manifest: dict[str, Any],
    prediction: dict[str, Any] | None,
    *,
    ast_exact: bool | None = None,
    parse_valid: bool | None = None,
    schema_valid: bool | None = None,
    generation_failure: str | None = None,
    truncated: bool = False,
    catastrophic: bool | None = None,
) -> dict[str, Any]:
    prediction_raw = None if prediction is None else _canonical(prediction)
    _, scored = score_mobile_rows(
        [manifest],
        [
            {
                "id": manifest["id"],
                "prediction_raw": prediction_raw,
                "generation_failure": generation_failure,
                "truncated": truncated,
            }
        ],
    )
    record = scored[0].to_record()
    if ast_exact is not None:
        record["ast_exact"] = ast_exact
    if parse_valid is not None:
        record["parse_valid"] = parse_valid
    if schema_valid is not None:
        record["schema_valid"] = schema_valid
    if catastrophic is not None:
        record["catastrophic_unauthorized_action"] = catastrophic
    record.update(
        {
            # These malicious/untrusted values must not affect the derived join.
            "cluster_id": "caller-controlled-cluster",
            "temporal_subset": temporal.NON_CALENDAR_SUBSET,
            "calendar_datetime_exact": True,
        }
    )
    return record


def _population() -> list[dict[str, Any]]:
    return [
        _manifest(
            "cross-multi",
            [
                _map(),
                _calendar("2026-09-02T09:00:00", title="First"),
                _map("bookstore"),
            ],
            cluster_id="cross-cluster",
        ),
        _manifest(
            "cross-missing",
            [_calendar("2026-09-04T11:00:00")],
            cluster_id="cross-missing-cluster",
        ),
        _manifest(
            "same-invalid",
            [_calendar("2026-08-30T11:00:00")],
            cluster_id="same-cluster",
        ),
        _manifest("non-calendar", [_map("tea")], cluster_id="map-cluster"),
    ]


def test_temporal_subsets_are_manifest_derived_and_failures_stay_wrong() -> None:
    manifest = _population()
    multi_gold = json.loads(manifest[0]["target"])
    multi_prediction = copy.deepcopy(multi_gold)
    multi_prediction["calls"][1]["args"]["datetime"] = "2026-09-02T10:30:00"
    rows = [
        _score(manifest[0], multi_prediction, ast_exact=False),
        _score(
            manifest[1],
            None,
            ast_exact=False,
            parse_valid=False,
            schema_valid=False,
        ),
        # Schema-invalid output stays in the denominator and is forced wrong.
        _score(
            manifest[2],
            {"decision": "CALL", "mode": "SINGLE", "calls": []},
        ),
        _score(manifest[3], json.loads(manifest[3]["target"])),
    ]

    result = temporal.score_temporal_subsets(manifest, rows)

    assert result["manifest_binding"]["cluster_and_subset_source"] == "joined_manifest_only"
    cross = result["subsets"][temporal.CALENDAR_CROSS_MONTH_SUBSET]
    assert cross["sample_count"] == 2
    assert cross[temporal.CALENDAR_DATETIME_EXACT_METRIC] == {
        "numerator": 0,
        "denominator": 2,
        "value": 0.0,
    }
    assert cross["missing_prediction"]["numerator"] == 1
    same = result["subsets"][temporal.CALENDAR_SAME_MONTH_SUBSET]
    assert same[temporal.AST_EXACT_METRIC]["numerator"] == 0
    assert same["forced_incorrect"]["numerator"] == 0
    assert (
        result["subsets"][temporal.NON_CALENDAR_SUBSET][temporal.AST_EXACT_METRIC]["numerator"] == 1
    )


def test_datetime_endpoint_uses_true_calendar_position_in_multi_action_rows() -> None:
    manifest = [_population()[0]]
    gold = json.loads(manifest[0]["target"])
    exact = temporal.score_temporal_subsets(manifest, [_score(manifest[0], gold)])
    assert (
        exact["subsets"][temporal.CALENDAR_CROSS_MONTH_SUBSET][
            temporal.CALENDAR_DATETIME_EXACT_METRIC
        ]["numerator"]
        == 1
    )

    wrong_later_call = copy.deepcopy(gold)
    wrong_later_call["calls"][1]["args"]["datetime"] = "2026-09-02T09:00:01"
    wrong = temporal.score_temporal_subsets(
        manifest, [_score(manifest[0], wrong_later_call, ast_exact=False)]
    )
    assert (
        wrong["subsets"][temporal.CALENDAR_CROSS_MONTH_SUBSET][
            temporal.CALENDAR_DATETIME_EXACT_METRIC
        ]["numerator"]
        == 0
    )

    shifted_position = copy.deepcopy(gold)
    shifted_position["calls"][0], shifted_position["calls"][1] = (
        shifted_position["calls"][1],
        shifted_position["calls"][0],
    )
    shifted = temporal.score_temporal_subsets(
        manifest, [_score(manifest[0], shifted_position, ast_exact=False)]
    )
    assert (
        shifted["subsets"][temporal.CALENDAR_CROSS_MONTH_SUBSET][
            temporal.CALENDAR_DATETIME_EXACT_METRIC
        ]["numerator"]
        == 0
    )


def test_invalid_predicted_datetime_is_wrong_but_invalid_gold_or_now_fails_closed() -> None:
    manifest = _manifest("cross-invalid-prediction", [_calendar("2026-09-02T09:00:00")])
    prediction = json.loads(manifest["target"])
    prediction["calls"][0]["args"]["datetime"] = "2026-09-31T09:00:00"
    score = _score(manifest, prediction)

    result = temporal.score_temporal_subsets([manifest], [score])
    cross = result["subsets"][temporal.CALENDAR_CROSS_MONTH_SUBSET]
    assert cross[temporal.CALENDAR_DATETIME_EXACT_METRIC]["numerator"] == 0
    assert cross["parse_valid"]["numerator"] == 1
    assert cross["schema_valid"]["numerator"] == 1

    invalid_gold = _manifest("invalid-gold", [_calendar("2026-09-31T09:00:00")])
    with pytest.raises(temporal.MobileTemporalEvaluationError, match="not a valid datetime"):
        temporal.score_temporal_subsets(
            [invalid_gold], [_score(invalid_gold, json.loads(invalid_gold["target"]))]
        )

    invalid_now = _manifest(
        "invalid-now",
        [_calendar("2026-09-02T09:00:00")],
        now="2026-08-32T12:00:00",
    )
    with pytest.raises(temporal.MobileTemporalEvaluationError, match="not a valid datetime"):
        temporal.score_temporal_subsets(
            [invalid_now], [_score(invalid_now, json.loads(invalid_now["target"]))]
        )


def test_manifest_score_join_and_mixed_calendar_classification_fail_closed() -> None:
    manifest = _population()
    rows = [_score(row, json.loads(row["target"])) for row in manifest]
    with pytest.raises(temporal.MobileTemporalEvaluationError, match="ID sets differ"):
        temporal.score_temporal_subsets(manifest, rows[:-1])

    bad_gold = copy.deepcopy(rows)
    bad_gold[0]["gold"] = {"decision": "ABSTAIN"}
    with pytest.raises(temporal.MobileTemporalEvaluationError, match="does not match"):
        temporal.score_temporal_subsets(manifest, bad_gold)

    bad_hash = copy.deepcopy(manifest)
    bad_hash[0]["metadata"]["target_sha256"] = "0" * 64
    with pytest.raises(temporal.MobileTemporalEvaluationError, match="hash mismatch"):
        temporal.score_temporal_subsets(bad_hash, rows)

    multiple = _manifest(
        "multiple",
        [_calendar("2026-08-30T09:00:00"), _calendar("2026-09-02T09:00:00")],
    )
    with pytest.raises(temporal.MobileTemporalEvaluationError, match="multiple calendar calls"):
        temporal.score_temporal_subsets(
            [multiple], [_score(multiple, json.loads(multiple["target"]))]
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row, gold: row.__setitem__("ast_exact", True), "metrics disagree"),
        (lambda row, gold: row.__setitem__("prediction", gold), "prediction.*disagrees"),
        (lambda row, gold: row.pop("truncated"), "raw prediction evidence"),
        (lambda row, gold: row.pop("generation_failure"), "raw prediction evidence"),
        (
            lambda row, gold: row.pop("catastrophic_unauthorized_action"),
            "required evidence fields",
        ),
        (lambda row, gold: row.__setitem__("evaluator_version", "forged"), "frozen evaluator"),
    ],
)
def test_score_fields_cannot_override_frozen_mobile_recomputation(
    mutate: Any, message: str
) -> None:
    manifest = _manifest("adversarial-score", [_map("gold")])
    wrong_prediction = _action([_map("wrong")])
    row = _score(manifest, wrong_prediction)
    mutate(row, json.loads(manifest["target"]))

    with pytest.raises(temporal.MobileTemporalEvaluationError, match=message):
        temporal.score_temporal_subsets([manifest], [row])


def _seed_population() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(4):
        rows.append(
            _manifest(
                f"cross-{index}",
                [_calendar(f"2026-09-{index + 1:02d}T09:00:00")],
                cluster_id=f"cross-cluster-{index // 2}",
            )
        )
    for index in range(2):
        rows.append(
            _manifest(
                f"same-{index}",
                [_calendar(f"2026-08-{index + 1:02d}T09:00:00")],
                cluster_id=f"same-cluster-{index}",
            )
        )
        rows.append(
            _manifest(
                f"map-{index}",
                [_map(str(index))],
                cluster_id=f"map-cluster-{index}",
            )
        )
    return rows


def _seed_scores(
    manifest: list[dict[str, Any]], cross_correct: dict[int, int]
) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        scores: list[dict[str, Any]] = []
        for row in manifest:
            gold = json.loads(row["target"])
            prediction = copy.deepcopy(gold)
            if row["id"].startswith("cross-"):
                index = int(row["id"].split("-")[1])
                if index >= cross_correct[seed]:
                    prediction["calls"][0]["args"]["datetime"] = "2026-10-01T09:00:00"
            scores.append(_score(row, prediction))
        output[seed] = scores
    return output


def test_three_seed_comparison_is_matched_and_reports_positive_seed_count() -> None:
    manifest = _seed_population()
    reference = _seed_scores(manifest, {17: 1, 29: 1, 43: 1})
    candidate = _seed_scores(manifest, {17: 3, 29: 2, 43: 1})

    result = temporal.compare_seeded_arms(manifest, candidate, reference, expected_seeds=SEEDS)
    endpoint = result["subsets"][temporal.CALENDAR_CROSS_MONTH_SUBSET]["metrics"][
        temporal.CALENDAR_DATETIME_EXACT_METRIC
    ]
    assert endpoint["mean_delta"] == {
        "numerator": 1,
        "denominator": 4,
        "value": 0.25,
    }
    assert endpoint["positive_seed_count"] == 2
    assert endpoint["paired_transition_counts"]["candidate_only"] == 3

    missing_seed = dict(candidate)
    missing_seed.pop(43)
    with pytest.raises(temporal.MobileTemporalEvaluationError, match="seeds must be exactly"):
        temporal.compare_seeded_arms(manifest, missing_seed, reference, expected_seeds=SEEDS)


def test_pcg64_cluster_bootstrap_is_deterministic_and_order_invariant() -> None:
    manifest = _seed_population()
    reference = _seed_scores(manifest, {17: 0, 29: 0, 43: 0})
    candidate = _seed_scores(manifest, {17: 3, 29: 2, 43: 1})
    kwargs = {
        "subset": temporal.CALENDAR_CROSS_MONTH_SUBSET,
        "metric": temporal.CALENDAR_DATETIME_EXACT_METRIC,
        "expected_seeds": SEEDS,
        "resamples": 256,
        "rng_seed": 172943,
    }

    first = temporal.paired_cluster_bootstrap(manifest, candidate, reference, **kwargs)
    second = temporal.paired_cluster_bootstrap(
        list(reversed(manifest)),
        {seed: list(reversed(rows)) for seed, rows in candidate.items()},
        {seed: list(reversed(rows)) for seed, rows in reference.items()},
        **kwargs,
    )
    assert first == second
    assert first["rng"] == {
        "library": "numpy",
        "api": "Generator",
        "bit_generator": "PCG64",
        "seed": 172943,
    }
    assert first["point_estimate"] == 0.5
    assert first["resampling_unit"] == "semantic_cluster_within_each_fixed_training_seed"
    assert len(first["replicate_float64_le_sha256"]) == 64

    changed_seed = temporal.paired_cluster_bootstrap(
        manifest, candidate, reference, **{**kwargs, "rng_seed": 172944}
    )
    assert changed_seed["replicate_float64_le_sha256"] != first["replicate_float64_le_sha256"]


def test_cluster_bootstrap_preserves_equal_weight_for_each_fixed_training_seed() -> None:
    manifest = [
        _manifest(
            f"cross-{index}",
            [_calendar(f"2026-09-{index + 1:02d}T09:00:00")],
            cluster_id="large" if index < 3 else "small",
        )
        for index in range(4)
    ]
    reference = _seed_scores(manifest, {17: 0, 29: 0, 43: 0})
    candidate = _seed_scores(manifest, {17: 3, 29: 1, 43: 2})
    resamples = 64
    rng_seed = 91

    result = temporal.paired_cluster_bootstrap(
        manifest,
        candidate,
        reference,
        subset=temporal.CALENDAR_CROSS_MONTH_SUBSET,
        metric=temporal.CALENDAR_DATETIME_EXACT_METRIC,
        expected_seeds=SEEDS,
        resamples=resamples,
        rng_seed=rng_seed,
    )

    # Independently reproduce mean(within-seed sampled row rates). Pooling sampled
    # rows across seeds would randomly reweight seeds when cluster sizes differ.
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    cluster_sums = (
        np.asarray([3, 0], dtype=np.int64),
        np.asarray([1, 0], dtype=np.int64),
        np.asarray([2, 0], dtype=np.int64),
    )
    cluster_counts = np.asarray([3, 1], dtype=np.int64)
    expected = np.empty(resamples, dtype=np.float64)
    for replicate in range(resamples):
        rate_sum = 0.0
        for sums in cluster_sums:
            indexes = rng.integers(0, 2, size=2)
            rate_sum += int(sums[indexes].sum()) / int(cluster_counts[indexes].sum())
        expected[replicate] = rate_sum / 3
    digest = hashlib.sha256(np.asarray(expected, dtype="<f8").tobytes(order="C")).hexdigest()

    assert result["estimator"] == "mean_of_fixed_seed_within_seed_cluster_rates"
    assert result["point_estimate_exact"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }
    assert result["replicate_float64_le_sha256"] == digest


def _ratio(numerator: int, denominator: int) -> dict[str, int]:
    return {"numerator": numerator, "denominator": denominator}


def _thresholds(*, confirmation: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "comparison_names": [CONTROL_A, CONTROL_B],
        "expected_seeds": list(SEEDS),
        "target_subset": temporal.CALENDAR_CROSS_MONTH_SUBSET,
        "target_metric": temporal.CALENDAR_DATETIME_EXACT_METRIC,
        "overall_metric": temporal.AST_EXACT_METRIC,
        "regression_subsets": [
            temporal.CALENDAR_SAME_MONTH_SUBSET,
            temporal.NON_CALENDAR_SUBSET,
        ],
        "regression_metrics": {
            temporal.CALENDAR_SAME_MONTH_SUBSET: temporal.CALENDAR_DATETIME_EXACT_METRIC,
            temporal.NON_CALENDAR_SUBSET: temporal.AST_EXACT_METRIC,
        },
        "minimum_target_mean_gain": _ratio(10 if confirmation else 15, 100),
        "minimum_overall_mean_gain": _ratio(2 if confirmation else 3, 100),
        "minimum_positive_seed_count": 2,
        "maximum_regression_mean_loss": _ratio(2, 100),
        "minimum_candidate_parse_valid": _ratio(995, 1000),
        "minimum_candidate_schema_valid": _ratio(99, 100),
        "maximum_missing_predictions": 0,
        "maximum_generation_failures": 0,
        "maximum_truncations": 0,
        "maximum_catastrophic_unauthorized_actions": 0,
    }
    if confirmation:
        result["bootstrap_lower_bound_strictly_greater_than"] = _ratio(0, 1)
        result["bootstrap_lower_probability"] = _ratio(5, 100)
        result["bootstrap_contract"] = {
            "estimator": "mean_of_fixed_seed_within_seed_cluster_rates",
            "expected_training_seeds": list(SEEDS),
            "quantile_method": "linear",
            "resamples": 10_000,
            "resampling_unit": "semantic_cluster_within_each_fixed_training_seed",
            "rng": {
                "api": "Generator",
                "bit_generator": "PCG64",
                "library": "numpy",
                "seed": 17,
            },
        }
    return result


def _comparison(
    *,
    target_gain: Fraction,
    overall_gain: Fraction,
    same_delta: Fraction = Fraction(-2, 100),
    non_calendar_delta: Fraction = Fraction(-2, 100),
    positive_seeds: int = 2,
    parse: tuple[int, int] = (995, 1000),
    schema: tuple[int, int] = (990, 1000),
    missing: int = 0,
    generation_failure: int = 0,
    truncation: int = 0,
    catastrophic: int = 0,
) -> dict[str, Any]:
    binding = {
        "rows": 1000,
        "content_sha256": "a" * 64,
        "membership_sha256": "b" * 64,
        "cluster_and_subset_source": "joined_manifest_only",
        "sample_metrics_source": "recomputed_from_prediction_raw_with_frozen_mobile_scorer",
        "mobile_actions_scorer_version": MOBILE_ACTIONS_SCORER_VERSION,
        "action_ir_evaluator_version": EVALUATOR_VERSION,
    }

    def endpoint(delta: Fraction) -> dict[str, Any]:
        return {
            "mean_delta": _ratio(delta.numerator, delta.denominator),
            "positive_seed_count": positive_seeds,
            "seed_count": 3,
        }

    overall_metrics = {
        "parse_valid": _ratio(*parse),
        "schema_valid": _ratio(*schema),
        "missing_prediction": _ratio(missing, 1000),
        "generation_failure": _ratio(generation_failure, 1000),
        "truncation": _ratio(truncation, 1000),
        "catastrophic_unauthorized_action": _ratio(catastrophic, 1000),
    }
    return {
        "schema_version": temporal.MOBILE_TEMPORAL_COMPARISON_VERSION,
        "manifest_binding": binding,
        "expected_seeds": list(SEEDS),
        "candidate": {"pooled": {"subsets": {temporal.OVERALL_SUBSET: overall_metrics}}},
        "subsets": {
            temporal.CALENDAR_CROSS_MONTH_SUBSET: {
                "metrics": {temporal.CALENDAR_DATETIME_EXACT_METRIC: endpoint(target_gain)}
            },
            temporal.OVERALL_SUBSET: {
                "metrics": {temporal.AST_EXACT_METRIC: endpoint(overall_gain)}
            },
            temporal.CALENDAR_SAME_MONTH_SUBSET: {
                "metrics": {temporal.CALENDAR_DATETIME_EXACT_METRIC: endpoint(same_delta)}
            },
            temporal.NON_CALENDAR_SUBSET: {
                "metrics": {temporal.AST_EXACT_METRIC: endpoint(non_calendar_delta)}
            },
        },
    }


def _bootstrap(comparison: dict[str, Any], lower: float) -> dict[str, Any]:
    point = comparison["subsets"][temporal.CALENDAR_CROSS_MONTH_SUBSET]["metrics"][
        temporal.CALENDAR_DATETIME_EXACT_METRIC
    ]["mean_delta"]
    return {
        "schema_version": temporal.MOBILE_TEMPORAL_BOOTSTRAP_VERSION,
        "manifest_binding": comparison["manifest_binding"],
        "subset": temporal.CALENDAR_CROSS_MONTH_SUBSET,
        "metric": temporal.CALENDAR_DATETIME_EXACT_METRIC,
        "expected_seeds": list(SEEDS),
        "point_estimate": point["numerator"] / point["denominator"],
        "point_estimate_exact": copy.deepcopy(point),
        "one_sided_lower_percentile": {
            "probability": _ratio(5, 100),
            "value": lower,
        },
        "resamples": 10_000,
        "rng": {
            "library": "numpy",
            "api": "Generator",
            "bit_generator": "PCG64",
            "seed": 17,
        },
        "resampling_unit": "semantic_cluster_within_each_fixed_training_seed",
        "estimator": "mean_of_fixed_seed_within_seed_cluster_rates",
        "quantile_method": "linear",
        "cluster_counts_by_seed": {str(seed): 10 for seed in SEEDS},
        "replicate_float64_le_sha256": "c" * 64,
    }


def test_selection_gate_passes_exact_inclusive_boundaries_and_is_conjunctive() -> None:
    boundary = _comparison(target_gain=Fraction(15, 100), overall_gain=Fraction(3, 100))
    comparisons = {CONTROL_A: boundary, CONTROL_B: copy.deepcopy(boundary)}
    passed = temporal.evaluate_temporal_gates(comparisons, thresholds=_thresholds())
    assert passed["passed"] is True
    assert passed["candidate_quality"]["passed"] is True

    below_target = copy.deepcopy(comparisons)
    below_target[CONTROL_B]["subsets"][temporal.CALENDAR_CROSS_MONTH_SUBSET]["metrics"][
        temporal.CALENDAR_DATETIME_EXACT_METRIC
    ]["mean_delta"] = _ratio(149, 1000)
    assert (
        temporal.evaluate_temporal_gates(below_target, thresholds=_thresholds())["passed"] is False
    )

    excess_regression = {
        CONTROL_A: _comparison(
            target_gain=Fraction(15, 100),
            overall_gain=Fraction(3, 100),
            same_delta=Fraction(-21, 1000),
        ),
        CONTROL_B: copy.deepcopy(boundary),
    }
    assert (
        temporal.evaluate_temporal_gates(excess_regression, thresholds=_thresholds())["passed"]
        is False
    )

    invalid_quality = _comparison(
        target_gain=Fraction(15, 100),
        overall_gain=Fraction(3, 100),
        parse=(994, 1000),
        missing=1,
    )
    assert (
        temporal.evaluate_temporal_gates(
            {CONTROL_A: invalid_quality, CONTROL_B: copy.deepcopy(invalid_quality)},
            thresholds=_thresholds(),
        )["passed"]
        is False
    )


@pytest.mark.parametrize(
    ("overrides", "failed_section", "failed_check"),
    [
        (
            {"overall_gain": Fraction(29, 1000)},
            "comparison",
            "overall_mean_gain_at_least_threshold",
        ),
        (
            {"positive_seeds": 1},
            "comparison",
            "positive_seed_count_at_least_threshold",
        ),
        (
            {"same_delta": Fraction(-21, 1000)},
            "comparison",
            f"{temporal.CALENDAR_SAME_MONTH_SUBSET}_loss_at_most_threshold",
        ),
        (
            {"non_calendar_delta": Fraction(-21, 1000)},
            "comparison",
            f"{temporal.NON_CALENDAR_SUBSET}_loss_at_most_threshold",
        ),
        (
            {"parse": (994, 1000)},
            "quality",
            "parse_valid_at_least_threshold",
        ),
        (
            {"schema": (989, 1000)},
            "quality",
            "schema_valid_at_least_threshold",
        ),
        (
            {"missing": 1},
            "quality",
            "missing_prediction_at_most_threshold",
        ),
        (
            {"generation_failure": 1},
            "quality",
            "generation_failure_at_most_threshold",
        ),
        (
            {"truncation": 1},
            "quality",
            "truncation_at_most_threshold",
        ),
        (
            {"catastrophic": 1},
            "quality",
            "catastrophic_unauthorized_action_at_most_threshold",
        ),
    ],
)
def test_selection_gate_fails_each_independent_condition(
    overrides: dict[str, Any], failed_section: str, failed_check: str
) -> None:
    arguments: dict[str, Any] = {
        "target_gain": Fraction(15, 100),
        "overall_gain": Fraction(3, 100),
    }
    arguments.update(overrides)
    comparison = _comparison(**arguments)
    result = temporal.evaluate_temporal_gates(
        {CONTROL_A: comparison, CONTROL_B: copy.deepcopy(comparison)},
        thresholds=_thresholds(),
    )

    assert result["passed"] is False
    if failed_section == "quality":
        assert result["candidate_quality"]["checks"][failed_check] is False
    else:
        assert result["comparisons"][CONTROL_A]["checks"][failed_check] is False


def test_confirmation_gate_uses_strict_positive_bootstrap_boundary() -> None:
    boundary = _comparison(
        target_gain=Fraction(10, 100),
        overall_gain=Fraction(2, 100),
        positive_seeds=2,
    )
    comparisons = {CONTROL_A: boundary, CONTROL_B: copy.deepcopy(boundary)}
    positive_bootstraps = {
        name: _bootstrap(comparison, 1e-12) for name, comparison in comparisons.items()
    }
    passed = temporal.evaluate_temporal_gates(
        comparisons,
        thresholds=_thresholds(confirmation=True),
        bootstraps=positive_bootstraps,
    )
    assert passed["passed"] is True

    zero_boundary = copy.deepcopy(positive_bootstraps)
    zero_boundary[CONTROL_B]["one_sided_lower_percentile"]["value"] = 0.0
    failed = temporal.evaluate_temporal_gates(
        comparisons,
        thresholds=_thresholds(confirmation=True),
        bootstraps=zero_boundary,
    )
    assert failed["passed"] is False
    assert (
        failed["comparisons"][CONTROL_B]["checks"]["bootstrap_lower_bound_strictly_above_threshold"]
        is False
    )

    wrong_probability = copy.deepcopy(positive_bootstraps)
    wrong_probability[CONTROL_A]["one_sided_lower_percentile"]["probability"] = _ratio(1, 10)
    with pytest.raises(temporal.MobileTemporalEvaluationError, match="wrong lower probability"):
        temporal.evaluate_temporal_gates(
            comparisons,
            thresholds=_thresholds(confirmation=True),
            bootstraps=wrong_probability,
        )

    insufficient_positive_seeds = {
        name: _comparison(
            target_gain=Fraction(10, 100),
            overall_gain=Fraction(2, 100),
            positive_seeds=1,
        )
        for name in (CONTROL_A, CONTROL_B)
    }
    insufficient_bootstraps = {
        name: _bootstrap(comparison, 1e-12)
        for name, comparison in insufficient_positive_seeds.items()
    }
    result = temporal.evaluate_temporal_gates(
        insufficient_positive_seeds,
        thresholds=_thresholds(confirmation=True),
        bootstraps=insufficient_bootstraps,
    )
    assert result["passed"] is False
    assert (
        result["comparisons"][CONTROL_A]["checks"]["positive_seed_count_at_least_threshold"]
        is False
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: artifact.__setitem__("resamples", 9_999),
            "frozen bootstrap contract",
        ),
        (
            lambda artifact: artifact["rng"].__setitem__("seed", 18),
            "frozen bootstrap contract",
        ),
        (
            lambda artifact: artifact.__setitem__("quantile_method", "nearest"),
            "frozen bootstrap contract",
        ),
        (
            lambda artifact: artifact.__setitem__("point_estimate_exact", _ratio(11, 100)),
            "point estimate differs",
        ),
    ],
)
def test_confirmation_gate_binds_complete_bootstrap_identity(mutate: Any, message: str) -> None:
    boundary = _comparison(target_gain=Fraction(10, 100), overall_gain=Fraction(2, 100))
    comparisons = {CONTROL_A: boundary, CONTROL_B: copy.deepcopy(boundary)}
    bootstraps = {name: _bootstrap(comparison, 1e-12) for name, comparison in comparisons.items()}
    mutate(bootstraps[CONTROL_A])

    with pytest.raises(temporal.MobileTemporalEvaluationError, match=message):
        temporal.evaluate_temporal_gates(
            comparisons,
            thresholds=_thresholds(confirmation=True),
            bootstraps=bootstraps,
        )


def test_gate_rejects_float_thresholds_instead_of_approximating_them() -> None:
    thresholds = _thresholds()
    thresholds["minimum_target_mean_gain"] = 0.15
    comparison = _comparison(target_gain=Fraction(15, 100), overall_gain=Fraction(3, 100))
    with pytest.raises(temporal.MobileTemporalEvaluationError, match="exact numerator"):
        temporal.evaluate_temporal_gates(
            {CONTROL_A: comparison, CONTROL_B: copy.deepcopy(comparison)},
            thresholds=thresholds,
        )


def test_gate_requires_both_controls_and_frozen_metric_mapping() -> None:
    comparison = _comparison(target_gain=Fraction(15, 100), overall_gain=Fraction(3, 100))
    with pytest.raises(temporal.MobileTemporalEvaluationError, match="exactly the two"):
        temporal.evaluate_temporal_gates({CONTROL_A: comparison}, thresholds=_thresholds())

    wrong_endpoint = _thresholds()
    wrong_endpoint["target_subset"] = temporal.CALENDAR_SAME_MONTH_SUBSET
    with pytest.raises(temporal.MobileTemporalEvaluationError, match="frozen target endpoint"):
        temporal.evaluate_temporal_gates(
            {CONTROL_A: comparison, CONTROL_B: copy.deepcopy(comparison)},
            thresholds=wrong_endpoint,
        )

    wrong_names = _thresholds()
    wrong_names["comparison_names"] = ["different-A", "different-B"]
    with pytest.raises(temporal.MobileTemporalEvaluationError, match="reference arms"):
        temporal.evaluate_temporal_gates(
            {CONTROL_A: comparison, CONTROL_B: copy.deepcopy(comparison)},
            thresholds=wrong_names,
        )

    wrong_seed_comparison = copy.deepcopy(comparison)
    wrong_seed_comparison["expected_seeds"] = [1, 2, 3]
    with pytest.raises(temporal.MobileTemporalEvaluationError, match="seed identity"):
        temporal.evaluate_temporal_gates(
            {CONTROL_A: wrong_seed_comparison, CONTROL_B: copy.deepcopy(comparison)},
            thresholds=_thresholds(),
        )
