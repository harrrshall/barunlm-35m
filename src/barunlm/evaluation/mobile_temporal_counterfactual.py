"""Strict evaluation for the Mobile month-boundary counterfactual experiment.

Temporal labels and semantic-cluster IDs are derived only from the joined manifest.  A
caller cannot inject them through model score rows.  Missing, invalid, truncated, and
generation-failed predictions remain in every denominator and count as incorrect.

The module contains mechanics, not product policy: all effect, regression, validity,
failure, and bootstrap thresholds are supplied explicitly as exact numerator/denominator
objects by the preregistered caller.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from typing import Any

import numpy as np

from .evaluator import EVALUATOR_VERSION, SampleEvaluation
from .mobile_actions import MOBILE_ACTIONS_SCORER_VERSION, MobileActionsScoreError, score_rows

MOBILE_TEMPORAL_SCORE_VERSION = "barun-mobile-temporal-score-v1"
MOBILE_TEMPORAL_COMPARISON_VERSION = "barun-mobile-temporal-comparison-v1"
MOBILE_TEMPORAL_BOOTSTRAP_VERSION = "barun-mobile-temporal-bootstrap-v1"
MOBILE_TEMPORAL_GATE_VERSION = "barun-mobile-temporal-gate-v1"

OVERALL_SUBSET = "overall"
CALENDAR_CROSS_MONTH_SUBSET = "calendar_cross_month"
CALENDAR_SAME_MONTH_SUBSET = "calendar_same_month"
NON_CALENDAR_SUBSET = "non_calendar"
AST_EXACT_METRIC = "ast_exact_match"
CALENDAR_DATETIME_EXACT_METRIC = "create_calendar_event_datetime_exact"

_SUBSETS = (
    CALENDAR_CROSS_MONTH_SUBSET,
    CALENDAR_SAME_MONTH_SUBSET,
    NON_CALENDAR_SUBSET,
)
_METRICS = (AST_EXACT_METRIC, CALENDAR_DATETIME_EXACT_METRIC)
_NOW_RE = re.compile(r"(?m)^NOW (?P<value>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})$")
_DATETIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REQUIRED_THRESHOLD_FIELDS = frozenset(
    {
        "comparison_names",
        "expected_seeds",
        "target_subset",
        "target_metric",
        "overall_metric",
        "regression_subsets",
        "regression_metrics",
        "minimum_target_mean_gain",
        "minimum_overall_mean_gain",
        "minimum_positive_seed_count",
        "maximum_regression_mean_loss",
        "minimum_candidate_parse_valid",
        "minimum_candidate_schema_valid",
        "maximum_missing_predictions",
        "maximum_generation_failures",
        "maximum_truncations",
        "maximum_catastrophic_unauthorized_actions",
    }
)
_OPTIONAL_THRESHOLD_FIELDS = frozenset(
    {
        "bootstrap_lower_bound_strictly_greater_than",
        "bootstrap_lower_probability",
        "bootstrap_contract",
    }
)
_BOOTSTRAP_ESTIMATOR = "mean_of_fixed_seed_within_seed_cluster_rates"
_BOOTSTRAP_RESAMPLING_UNIT = "semantic_cluster_within_each_fixed_training_seed"
_BOOTSTRAP_CONTRACT_FIELDS = frozenset(
    {
        "estimator",
        "expected_training_seeds",
        "quantile_method",
        "resamples",
        "resampling_unit",
        "rng",
    }
)
_BOOTSTRAP_RNG_FIELDS = frozenset({"api", "bit_generator", "library", "seed"})


class MobileTemporalEvaluationError(ValueError):
    """A manifest, score artifact, comparison, or gate contract is malformed."""


@dataclass(frozen=True, slots=True)
class _ManifestIdentity:
    sample_id: str
    cluster_id: str
    subset: str
    prompt: str
    target: str
    metadata: Mapping[str, Any]
    gold: Mapping[str, Any]
    gold_identity: str
    calendar_datetime_facts: tuple[tuple[int, str], ...]


@dataclass(frozen=True, slots=True)
class _ManifestPopulation:
    rows: tuple[_ManifestIdentity, ...]
    content_sha256: str
    membership_sha256: str


@dataclass(frozen=True, slots=True)
class _Observation:
    sample_id: str
    cluster_id: str
    subset: str
    gold_identity: str
    ast_exact: bool
    effective_ast_exact: bool
    parse_valid: bool
    schema_valid: bool
    missing_prediction: bool
    truncated: bool
    generation_failure: bool
    catastrophic_unauthorized_action: bool
    calendar_datetime_eligible: bool
    calendar_datetime_exact: bool


def _canonical_json(value: object, *, label: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise MobileTemporalEvaluationError(f"{label} is not strict JSON") from error


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strict_bool(value: object, *, name: str, default: bool | None = None) -> bool:
    if value is None and default is not None:
        return default
    if type(value) is not bool:
        raise MobileTemporalEvaluationError(f"{name} must be boolean")
    return value


def _strict_datetime(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or _DATETIME_RE.fullmatch(value) is None:
        raise MobileTemporalEvaluationError(f"{label} must use YYYY-MM-DDTHH:MM:SS")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise MobileTemporalEvaluationError(f"{label} is not a valid datetime") from error
    if parsed.tzinfo is not None:
        raise MobileTemporalEvaluationError(f"{label} must be timezone-naive")
    return parsed


def _action_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise MobileTemporalEvaluationError(f"{label} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise MobileTemporalEvaluationError(f"{label} must be an Action IR object")
    return value


def _calendar_datetime_facts(
    action: Mapping[str, Any], *, label: str
) -> tuple[tuple[int, str], ...]:
    calls = action.get("calls", [])
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
        raise MobileTemporalEvaluationError(f"{label}.calls must be an array")
    facts: list[tuple[int, str]] = []
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            raise MobileTemporalEvaluationError(f"{label}.calls[{index}] must be an object")
        if call.get("tool") != "create_calendar_event":
            continue
        arguments = call.get("args")
        if not isinstance(arguments, Mapping):
            raise MobileTemporalEvaluationError(f"{label}.calls[{index}].args must be an object")
        raw_datetime = arguments.get("datetime")
        _strict_datetime(raw_datetime, label=f"{label}.calls[{index}].args.datetime")
        assert isinstance(raw_datetime, str)
        facts.append((index, raw_datetime))
    return tuple(facts)


def _classify_manifest_row(row: Mapping[str, Any], *, row_number: int) -> _ManifestIdentity:
    sample_id = row.get("id")
    prompt = row.get("prompt")
    target = row.get("target")
    metadata = row.get("metadata")
    if not isinstance(sample_id, str) or not sample_id:
        raise MobileTemporalEvaluationError(f"manifest row {row_number} requires a non-empty id")
    if not isinstance(prompt, str) or not isinstance(target, str):
        raise MobileTemporalEvaluationError(
            f"manifest row {sample_id!r} requires string prompt and target"
        )
    if not isinstance(metadata, Mapping):
        raise MobileTemporalEvaluationError(f"manifest metadata for {sample_id!r} is invalid")
    cluster_id = metadata.get("cluster_id")
    if not isinstance(cluster_id, str) or not cluster_id:
        raise MobileTemporalEvaluationError(f"manifest metadata for {sample_id!r} lacks cluster_id")
    for field, payload in (("prompt_sha256", prompt), ("target_sha256", target)):
        if field in metadata and metadata[field] != _sha256_text(payload):
            raise MobileTemporalEvaluationError(
                f"manifest metadata hash mismatch for {sample_id!r} at {field}"
            )

    now_matches = list(_NOW_RE.finditer(prompt))
    if len(now_matches) != 1:
        raise MobileTemporalEvaluationError(
            f"manifest prompt for {sample_id!r} must contain exactly one NOW line"
        )
    now = _strict_datetime(now_matches[0].group("value"), label=f"NOW for {sample_id!r}")
    gold = _action_mapping(target, label=f"target for {sample_id!r}")
    if _canonical_json(gold, label=f"target for {sample_id!r}") != target:
        raise MobileTemporalEvaluationError(
            f"manifest target for {sample_id!r} is not canonical strict JSON"
        )
    facts = _calendar_datetime_facts(gold, label=f"target for {sample_id!r}")
    if not facts:
        subset = NON_CALENDAR_SUBSET
    else:
        if len(facts) != 1:
            raise MobileTemporalEvaluationError(
                f"manifest row {sample_id!r} has multiple calendar calls unsupported by v1"
            )
        calendar_datetime = _strict_datetime(
            facts[0][1], label=f"calendar datetime for {sample_id!r}"
        )
        crosses_month = (calendar_datetime.year, calendar_datetime.month) != (
            now.year,
            now.month,
        )
        subset = CALENDAR_CROSS_MONTH_SUBSET if crosses_month else CALENDAR_SAME_MONTH_SUBSET
    gold_identity = _canonical_json(gold, label=f"target for {sample_id!r}")
    return _ManifestIdentity(
        sample_id=sample_id,
        cluster_id=cluster_id,
        subset=subset,
        prompt=prompt,
        target=target,
        metadata=metadata,
        gold=gold,
        gold_identity=gold_identity,
        calendar_datetime_facts=facts,
    )


def _prepare_manifest(
    manifest_rows: Sequence[Mapping[str, Any]],
) -> _ManifestPopulation:
    identities = tuple(
        sorted(
            (
                _classify_manifest_row(row, row_number=index)
                for index, row in enumerate(manifest_rows, start=1)
            ),
            key=lambda item: item.sample_id,
        )
    )
    if not identities:
        raise MobileTemporalEvaluationError("temporal evaluation manifest is empty")
    ids = [row.sample_id for row in identities]
    if len(ids) != len(set(ids)):
        raise MobileTemporalEvaluationError("temporal evaluation manifest has duplicate IDs")
    canonical_rows = sorted(_canonical_json(row, label="manifest row") for row in manifest_rows)
    return _ManifestPopulation(
        rows=identities,
        content_sha256=_sha256_text("\n".join(canonical_rows) + "\n"),
        membership_sha256=_sha256_text("\n".join(ids) + "\n"),
    )


def _score_observation(
    identity: _ManifestIdentity,
    row: Mapping[str, Any],
    recomputed: SampleEvaluation,
) -> _Observation:
    sample_id = identity.sample_id
    required_fields = {
        "ast_exact",
        "catastrophic_unauthorized_action",
        "evaluator_version",
        "generation_failure",
        "gold",
        "parse_valid",
        "prediction",
        "prediction_raw",
        "schema_valid",
        "truncated",
    }
    missing_fields = sorted(required_fields.difference(row))
    if missing_fields:
        raise MobileTemporalEvaluationError(
            f"score row {sample_id!r} lacks required evidence fields {missing_fields!r}"
        )
    if row.get("evaluator_version") != EVALUATOR_VERSION:
        raise MobileTemporalEvaluationError(
            f"score row {sample_id!r} does not use frozen evaluator {EVALUATOR_VERSION!r}"
        )
    if "gold" not in row:
        raise MobileTemporalEvaluationError(f"score row {sample_id!r} lacks gold Action IR")
    scored_gold = _canonical_json(
        _action_mapping(row["gold"], label=f"score gold for {sample_id!r}"),
        label=f"score gold for {sample_id!r}",
    )
    if scored_gold != identity.gold_identity:
        raise MobileTemporalEvaluationError(
            f"score gold does not match manifest target for {sample_id!r}"
        )
    ast_exact = _strict_bool(row.get("ast_exact"), name=f"ast_exact for {sample_id!r}")
    parse_valid = _strict_bool(row.get("parse_valid"), name=f"parse_valid for {sample_id!r}")
    schema_valid = _strict_bool(row.get("schema_valid"), name=f"schema_valid for {sample_id!r}")
    truncated = _strict_bool(row.get("truncated"), name=f"truncated for {sample_id!r}")
    catastrophic = _strict_bool(
        row.get("catastrophic_unauthorized_action"),
        name=f"catastrophic_unauthorized_action for {sample_id!r}",
    )
    if (
        ast_exact is not recomputed.ast_exact
        or parse_valid is not recomputed.parse_valid
        or schema_valid is not recomputed.schema_valid
        or truncated is not recomputed.truncated
        or catastrophic is not recomputed.catastrophic_unauthorized_action
        or row.get("generation_failure") != recomputed.generation_failure
    ):
        raise MobileTemporalEvaluationError(
            f"score row {sample_id!r} metrics disagree with frozen Mobile scorer recomputation"
        )
    prediction_raw = row.get("prediction_raw")
    if prediction_raw is not None and not isinstance(prediction_raw, str):
        raise MobileTemporalEvaluationError(
            f"prediction_raw for {sample_id!r} must be a string or null"
        )
    missing_prediction = prediction_raw is None
    if (
        "missing_prediction" in row
        and _strict_bool(
            row.get("missing_prediction"), name=f"missing_prediction for {sample_id!r}"
        )
        is not missing_prediction
    ):
        raise MobileTemporalEvaluationError(
            f"missing_prediction for {sample_id!r} disagrees with prediction_raw"
        )
    recomputed_prediction = (
        None if recomputed.prediction is None else recomputed.prediction.to_dict()
    )
    supplied_prediction = row.get("prediction")
    if recomputed_prediction is None:
        if supplied_prediction is not None:
            raise MobileTemporalEvaluationError(
                f"score prediction for {sample_id!r} disagrees with frozen Mobile scorer"
            )
    else:
        supplied_prediction_identity = _canonical_json(
            _action_mapping(supplied_prediction, label=f"prediction for {sample_id!r}"),
            label=f"prediction for {sample_id!r}",
        )
        if supplied_prediction_identity != _canonical_json(
            recomputed_prediction, label=f"recomputed prediction for {sample_id!r}"
        ):
            raise MobileTemporalEvaluationError(
                f"score prediction for {sample_id!r} disagrees with frozen Mobile scorer"
            )
    generation_failure = recomputed.generation_failure is not None
    valid_prediction = bool(
        parse_valid
        and schema_valid
        and not missing_prediction
        and not truncated
        and not generation_failure
    )
    effective_ast_exact = bool(ast_exact and valid_prediction)
    calendar_eligible = bool(identity.calendar_datetime_facts)
    calendar_exact = False
    if calendar_eligible and valid_prediction:
        prediction_value = recomputed_prediction
        if prediction_value is None:
            calendar_exact = False
        else:
            prediction = _action_mapping(prediction_value, label=f"prediction for {sample_id!r}")
            try:
                predicted_facts = _calendar_datetime_facts(
                    prediction, label=f"prediction for {sample_id!r}"
                )
            except MobileTemporalEvaluationError:
                # Gold and NOW define the population and remain fail-closed above.  A
                # syntactically/schema-valid predicted string can still be an impossible
                # calendar date; that is model error evidence, not an evaluator failure.
                predicted_facts = ()
            # The row passes only if every calendar datetime is at the same call index
            # and byte-exact.  One correct first call cannot mask a later call error.
            calendar_exact = predicted_facts == identity.calendar_datetime_facts
    return _Observation(
        sample_id=sample_id,
        cluster_id=identity.cluster_id,
        subset=identity.subset,
        gold_identity=identity.gold_identity,
        ast_exact=ast_exact,
        effective_ast_exact=effective_ast_exact,
        parse_valid=parse_valid,
        schema_valid=schema_valid,
        missing_prediction=missing_prediction,
        truncated=truncated,
        generation_failure=generation_failure,
        catastrophic_unauthorized_action=catastrophic,
        calendar_datetime_eligible=calendar_eligible,
        calendar_datetime_exact=calendar_exact,
    )


def _join_scores(
    manifest: _ManifestPopulation,
    scored_rows: Sequence[Mapping[str, Any]],
) -> tuple[_Observation, ...]:
    scores: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(scored_rows, start=1):
        if not isinstance(row, Mapping):
            raise MobileTemporalEvaluationError(f"score row {row_number} is not an object")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise MobileTemporalEvaluationError(
                f"score row {row_number} requires a non-empty sample_id"
            )
        if sample_id in scores:
            raise MobileTemporalEvaluationError(f"duplicate score ID {sample_id!r}")
        scores[sample_id] = row
    manifest_ids = {row.sample_id for row in manifest.rows}
    if set(scores) != manifest_ids:
        missing = sorted(manifest_ids.difference(scores))[:5]
        extra = sorted(set(scores).difference(manifest_ids))[:5]
        raise MobileTemporalEvaluationError(
            f"manifest/score ID sets differ; missing={missing!r}, extra={extra!r}"
        )
    mobile_manifest = [
        {
            "id": identity.sample_id,
            "metadata": dict(identity.metadata),
            "prompt": identity.prompt,
            "target": identity.target,
        }
        for identity in manifest.rows
    ]
    mobile_predictions: list[dict[str, Any]] = []
    for identity in manifest.rows:
        row = scores[identity.sample_id]
        required_prediction_fields = {"prediction_raw", "generation_failure", "truncated"}
        missing_fields = sorted(required_prediction_fields.difference(row))
        if missing_fields:
            raise MobileTemporalEvaluationError(
                f"score row {identity.sample_id!r} lacks raw prediction evidence {missing_fields!r}"
            )
        mobile_predictions.append(
            {
                "id": identity.sample_id,
                "prediction_raw": row.get("prediction_raw"),
                "generation_failure": row.get("generation_failure"),
                "truncated": row.get("truncated"),
            }
        )
    try:
        _, recomputed_rows = score_rows(mobile_manifest, mobile_predictions)
    except (MobileActionsScoreError, TypeError, ValueError) as error:
        raise MobileTemporalEvaluationError(
            "raw prediction evidence failed frozen Mobile scorer recomputation"
        ) from error
    recomputed = {row.sample_id: row for row in recomputed_rows}
    return tuple(
        _score_observation(identity, scores[identity.sample_id], recomputed[identity.sample_id])
        for identity in manifest.rows
    )


def _rate(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else 0.0,
    }


def _fraction_record(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "value": float(value),
    }


def _metrics(rows: Sequence[_Observation]) -> dict[str, Any]:
    denominator = len(rows)
    calendar_rows = [row for row in rows if row.calendar_datetime_eligible]
    return {
        "sample_count": denominator,
        "cluster_count": len({row.cluster_id for row in rows}),
        AST_EXACT_METRIC: _rate(sum(row.effective_ast_exact for row in rows), denominator),
        "raw_ast_exact_match": _rate(sum(row.ast_exact for row in rows), denominator),
        CALENDAR_DATETIME_EXACT_METRIC: _rate(
            sum(row.calendar_datetime_exact for row in calendar_rows), len(calendar_rows)
        ),
        "parse_valid": _rate(sum(row.parse_valid for row in rows), denominator),
        "schema_valid": _rate(sum(row.schema_valid for row in rows), denominator),
        "missing_prediction": _rate(sum(row.missing_prediction for row in rows), denominator),
        "truncation": _rate(sum(row.truncated for row in rows), denominator),
        "generation_failure": _rate(sum(row.generation_failure for row in rows), denominator),
        "catastrophic_unauthorized_action": _rate(
            sum(row.catastrophic_unauthorized_action for row in rows), denominator
        ),
        "forced_incorrect": _rate(
            sum(row.ast_exact and not row.effective_ast_exact for row in rows), denominator
        ),
    }


def _manifest_binding(manifest: _ManifestPopulation) -> dict[str, Any]:
    return {
        "rows": len(manifest.rows),
        "content_sha256": manifest.content_sha256,
        "membership_sha256": manifest.membership_sha256,
        "cluster_and_subset_source": "joined_manifest_only",
        "sample_metrics_source": "recomputed_from_prediction_raw_with_frozen_mobile_scorer",
        "mobile_actions_scorer_version": MOBILE_ACTIONS_SCORER_VERSION,
        "action_ir_evaluator_version": EVALUATOR_VERSION,
    }


def _score_population(
    observations: Sequence[_Observation], manifest: _ManifestPopulation
) -> dict[str, Any]:
    grouped = {subset: [row for row in observations if row.subset == subset] for subset in _SUBSETS}
    return {
        "schema_version": MOBILE_TEMPORAL_SCORE_VERSION,
        "sample_count": len(observations),
        "manifest_binding": _manifest_binding(manifest),
        "subset_names": [name for name in _SUBSETS if grouped[name]],
        "subsets": {
            OVERALL_SUBSET: _metrics(observations),
            **{name: _metrics(rows) for name, rows in grouped.items() if rows},
        },
    }


def score_temporal_subsets(
    manifest_rows: Sequence[Mapping[str, Any]],
    scored_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join identical ID sets, derive temporal subsets, and score failed rows in place."""

    manifest = _prepare_manifest(manifest_rows)
    return _score_population(_join_scores(manifest, scored_rows), manifest)


def _expected_seed_tuple(expected_seeds: Sequence[int]) -> tuple[int, int, int]:
    seeds = tuple(expected_seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise MobileTemporalEvaluationError(
            "expected_seeds must contain exactly three unique seeds"
        )
    if any(type(seed) is not int for seed in seeds):
        raise MobileTemporalEvaluationError("expected seeds must be integers")
    return seeds


def _seeded_observations(
    manifest: _ManifestPopulation,
    evidence: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    expected_seeds: tuple[int, int, int],
    arm_name: str,
) -> dict[int, tuple[_Observation, ...]]:
    if not isinstance(evidence, Mapping):
        raise MobileTemporalEvaluationError(f"{arm_name} evidence must be keyed by seed")
    if any(type(seed) is not int for seed in evidence):
        raise MobileTemporalEvaluationError(f"{arm_name} evidence has a non-integer seed")
    if set(evidence) != set(expected_seeds):
        raise MobileTemporalEvaluationError(
            f"{arm_name} evidence seeds must be exactly {list(expected_seeds)!r}"
        )
    return {seed: _join_scores(manifest, evidence[seed]) for seed in expected_seeds}


def _prepare_seeded_pairs(
    manifest_rows: Sequence[Mapping[str, Any]],
    candidate_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    reference_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    expected_seeds: Sequence[int],
) -> tuple[
    _ManifestPopulation,
    tuple[int, int, int],
    dict[int, tuple[_Observation, ...]],
    dict[int, tuple[_Observation, ...]],
]:
    manifest = _prepare_manifest(manifest_rows)
    seeds = _expected_seed_tuple(expected_seeds)
    candidate = _seeded_observations(
        manifest, candidate_by_seed, expected_seeds=seeds, arm_name="candidate"
    )
    reference = _seeded_observations(
        manifest, reference_by_seed, expected_seeds=seeds, arm_name="reference"
    )
    return manifest, seeds, candidate, reference


def _record_fraction(record: Mapping[str, Any], *, name: str) -> Fraction:
    numerator = record.get("numerator")
    denominator = record.get("denominator")
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise MobileTemporalEvaluationError(f"{name} is not an exact numerator/denominator ratio")
    return Fraction(numerator, denominator)


def _metric_outcome(row: _Observation, metric: str) -> bool | None:
    if metric == AST_EXACT_METRIC:
        return row.effective_ast_exact
    if metric == CALENDAR_DATETIME_EXACT_METRIC:
        return row.calendar_datetime_exact if row.calendar_datetime_eligible else None
    raise MobileTemporalEvaluationError(f"unsupported temporal metric {metric!r}")


def _metric_comparison(
    candidate: Mapping[int, tuple[_Observation, ...]],
    reference: Mapping[int, tuple[_Observation, ...]],
    *,
    seeds: Sequence[int],
    subset: str,
    metric: str,
) -> dict[str, Any] | None:
    per_seed: dict[str, Any] = {}
    candidate_rates: list[Fraction] = []
    reference_rates: list[Fraction] = []
    positive = 0
    transitions = {
        "both_correct": 0,
        "candidate_only": 0,
        "reference_only": 0,
        "both_incorrect": 0,
    }
    for seed in seeds:
        reference_index = {row.sample_id: row for row in reference[seed]}
        candidate_values: list[bool] = []
        reference_values: list[bool] = []
        for row in candidate[seed]:
            if subset != OVERALL_SUBSET and row.subset != subset:
                continue
            incumbent = reference_index[row.sample_id]
            candidate_value = _metric_outcome(row, metric)
            reference_value = _metric_outcome(incumbent, metric)
            if (candidate_value is None) != (reference_value is None):
                raise MobileTemporalEvaluationError(
                    f"metric eligibility differs at seed {seed}, sample {row.sample_id!r}"
                )
            if candidate_value is None:
                continue
            candidate_values.append(candidate_value)
            reference_values.append(bool(reference_value))
            if candidate_value and reference_value:
                transitions["both_correct"] += 1
            elif candidate_value:
                transitions["candidate_only"] += 1
            elif reference_value:
                transitions["reference_only"] += 1
            else:
                transitions["both_incorrect"] += 1
        if not candidate_values:
            return None
        candidate_rate = Fraction(sum(candidate_values), len(candidate_values))
        reference_rate = Fraction(sum(reference_values), len(reference_values))
        delta = candidate_rate - reference_rate
        candidate_rates.append(candidate_rate)
        reference_rates.append(reference_rate)
        if delta > 0:
            positive += 1
        per_seed[str(seed)] = {
            "candidate": _fraction_record(candidate_rate),
            "reference": _fraction_record(reference_rate),
            "delta": _fraction_record(delta),
        }
    candidate_mean = sum(candidate_rates, Fraction()) / len(candidate_rates)
    reference_mean = sum(reference_rates, Fraction()) / len(reference_rates)
    return {
        "candidate_mean": _fraction_record(candidate_mean),
        "reference_mean": _fraction_record(reference_mean),
        "mean_delta": _fraction_record(candidate_mean - reference_mean),
        "positive_seed_count": positive,
        "seed_count": len(seeds),
        "per_seed": per_seed,
        "paired_transition_counts": transitions,
    }


def compare_seeded_arms(
    manifest_rows: Sequence[Mapping[str, Any]],
    candidate_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    reference_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    """Compare two arms over exactly three seeds and one manifest-bound population."""

    manifest, seeds, candidate, reference = _prepare_seeded_pairs(
        manifest_rows,
        candidate_by_seed,
        reference_by_seed,
        expected_seeds=expected_seeds,
    )
    subsets: dict[str, Any] = {}
    for subset in (OVERALL_SUBSET, *_SUBSETS):
        metrics = {
            metric: result
            for metric in _METRICS
            if (
                result := _metric_comparison(
                    candidate,
                    reference,
                    seeds=seeds,
                    subset=subset,
                    metric=metric,
                )
            )
            is not None
        }
        if metrics:
            subsets[subset] = {"metrics": metrics}
    candidate_pooled = _score_population(
        tuple(row for seed in seeds for row in candidate[seed]), manifest
    )
    reference_pooled = _score_population(
        tuple(row for seed in seeds for row in reference[seed]), manifest
    )
    return {
        "schema_version": MOBILE_TEMPORAL_COMPARISON_VERSION,
        "manifest_binding": _manifest_binding(manifest),
        "expected_seeds": list(seeds),
        "pairing": {
            "sample_ids_matched": True,
            "gold_cluster_and_subset_identities_manifest_bound": True,
            "samples_per_seed": len(manifest.rows),
        },
        "candidate": {"pooled": candidate_pooled},
        "reference": {"pooled": reference_pooled},
        "subsets": subsets,
    }


def _probability(value: object, *, name: str) -> Fraction:
    if isinstance(value, Mapping):
        result = _record_fraction(value, name=name)
    elif isinstance(value, Fraction):
        result = value
    elif type(value) in (int, float) or isinstance(value, str):
        try:
            result = Fraction(str(value))
        except (ValueError, ZeroDivisionError) as error:
            raise MobileTemporalEvaluationError(f"{name} is not a valid probability") from error
    else:
        raise MobileTemporalEvaluationError(f"{name} is not a valid probability")
    if not 0 < result < 1:
        raise MobileTemporalEvaluationError(f"{name} must lie strictly between zero and one")
    return result


def paired_cluster_bootstrap(
    manifest_rows: Sequence[Mapping[str, Any]],
    candidate_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    reference_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    subset: str,
    metric: str,
    expected_seeds: Sequence[int],
    resamples: int,
    rng_seed: int,
    lower_probability: object = Fraction(1, 20),
) -> dict[str, Any]:
    """Compute a deterministic paired cluster bootstrap within every fixed seed."""

    if subset not in (OVERALL_SUBSET, *_SUBSETS):
        raise MobileTemporalEvaluationError(f"unsupported bootstrap subset {subset!r}")
    if metric not in _METRICS:
        raise MobileTemporalEvaluationError(f"unsupported bootstrap metric {metric!r}")
    if type(resamples) is not int or resamples <= 0:
        raise MobileTemporalEvaluationError("bootstrap resamples must be a positive integer")
    if type(rng_seed) is not int or rng_seed < 0:
        raise MobileTemporalEvaluationError("bootstrap rng_seed must be a non-negative integer")
    probability = _probability(lower_probability, name="lower_probability")
    manifest, seeds, candidate, reference = _prepare_seeded_pairs(
        manifest_rows,
        candidate_by_seed,
        reference_by_seed,
        expected_seeds=expected_seeds,
    )
    cluster_statistics: list[tuple[np.ndarray, np.ndarray]] = []
    point_rates: list[Fraction] = []
    for seed in seeds:
        reference_index = {row.sample_id: row for row in reference[seed]}
        grouped: dict[str, list[int]] = defaultdict(list)
        for row in candidate[seed]:
            if subset != OVERALL_SUBSET and row.subset != subset:
                continue
            incumbent = reference_index[row.sample_id]
            candidate_value = _metric_outcome(row, metric)
            reference_value = _metric_outcome(incumbent, metric)
            if (candidate_value is None) != (reference_value is None):
                raise MobileTemporalEvaluationError(
                    f"bootstrap eligibility differs at seed {seed}, sample {row.sample_id!r}"
                )
            if candidate_value is None:
                continue
            grouped[row.cluster_id].append(int(candidate_value) - int(reference_value))
        if not grouped:
            raise MobileTemporalEvaluationError(
                f"bootstrap endpoint {subset!r}/{metric!r} is empty at seed {seed}"
            )
        sums = np.asarray([sum(grouped[name]) for name in sorted(grouped)], dtype=np.int64)
        counts = np.asarray([len(grouped[name]) for name in sorted(grouped)], dtype=np.int64)
        cluster_statistics.append((sums, counts))
        point_rates.append(Fraction(int(sums.sum()), int(counts.sum())))

    rng = np.random.Generator(np.random.PCG64(rng_seed))
    replicates = np.empty(resamples, dtype=np.float64)
    for replicate_index in range(resamples):
        seed_rate_sum = 0.0
        for sums, counts in cluster_statistics:
            sampled_clusters = rng.integers(0, len(sums), size=len(sums))
            delta_sum = int(sums[sampled_clusters].sum())
            row_count = int(counts[sampled_clusters].sum())
            seed_rate_sum += delta_sum / row_count
        replicates[replicate_index] = seed_rate_sum / len(cluster_statistics)
    percentile = float(np.quantile(replicates, float(probability), method="linear"))
    replicate_bytes = np.asarray(replicates, dtype="<f8").tobytes(order="C")
    point_estimate = sum(point_rates, Fraction()) / len(point_rates)
    return {
        "schema_version": MOBILE_TEMPORAL_BOOTSTRAP_VERSION,
        "manifest_binding": _manifest_binding(manifest),
        "subset": subset,
        "metric": metric,
        "expected_seeds": list(seeds),
        "point_estimate": float(point_estimate),
        "point_estimate_exact": _fraction_record(point_estimate),
        "one_sided_lower_percentile": {
            "probability": _fraction_record(probability),
            "value": percentile,
        },
        "resamples": resamples,
        "rng": {
            "library": "numpy",
            "api": "Generator",
            "bit_generator": "PCG64",
            "seed": rng_seed,
        },
        "resampling_unit": _BOOTSTRAP_RESAMPLING_UNIT,
        "estimator": _BOOTSTRAP_ESTIMATOR,
        "quantile_method": "linear",
        "cluster_counts_by_seed": {
            str(seed): len(cluster_statistics[index][0]) for index, seed in enumerate(seeds)
        },
        "replicate_float64_le_sha256": hashlib.sha256(replicate_bytes).hexdigest(),
    }


def _threshold_ratio(value: object, *, name: str) -> Fraction:
    if not isinstance(value, Mapping):
        raise MobileTemporalEvaluationError(f"{name} must be an exact numerator/denominator object")
    result = _record_fraction(value, name=name)
    if result < 0:
        raise MobileTemporalEvaluationError(f"{name} must be non-negative")
    return result


def _non_negative_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise MobileTemporalEvaluationError(f"{name} must be a non-negative integer")
    return value


def _comparison_metric(
    comparison: Mapping[str, Any], *, subset: str, metric: str, reference_name: str
) -> Mapping[str, Any]:
    subsets = comparison.get("subsets")
    if not isinstance(subsets, Mapping) or not isinstance(subsets.get(subset), Mapping):
        raise MobileTemporalEvaluationError(
            f"comparison {reference_name!r} lacks subset {subset!r}"
        )
    metrics = subsets[subset].get("metrics")
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get(metric), Mapping):
        raise MobileTemporalEvaluationError(
            f"comparison {reference_name!r} lacks metric {subset}.{metric}"
        )
    return metrics[metric]


def _pooled_candidate(comparisons: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    first: Mapping[str, Any] | None = None
    first_binding: Mapping[str, Any] | None = None
    first_seeds: tuple[int, int, int] | None = None
    for reference_name, comparison in comparisons.items():
        if comparison.get("schema_version") != MOBILE_TEMPORAL_COMPARISON_VERSION:
            raise MobileTemporalEvaluationError(
                f"comparison {reference_name!r} has an unsupported schema version"
            )
        raw_seeds = comparison.get("expected_seeds")
        if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, (str, bytes, bytearray)):
            raise MobileTemporalEvaluationError(
                f"comparison {reference_name!r} lacks three-seed evidence"
            )
        seeds = _expected_seed_tuple(raw_seeds)
        binding = comparison.get("manifest_binding")
        candidate = comparison.get("candidate")
        if not isinstance(binding, Mapping) or not isinstance(candidate, Mapping):
            raise MobileTemporalEvaluationError(
                f"comparison {reference_name!r} lacks manifest/candidate evidence"
            )
        expected_binding_fields = {
            "action_ir_evaluator_version": EVALUATOR_VERSION,
            "cluster_and_subset_source": "joined_manifest_only",
            "mobile_actions_scorer_version": MOBILE_ACTIONS_SCORER_VERSION,
            "sample_metrics_source": "recomputed_from_prediction_raw_with_frozen_mobile_scorer",
        }
        if any(binding.get(name) != value for name, value in expected_binding_fields.items()):
            raise MobileTemporalEvaluationError(
                f"comparison {reference_name!r} is not bound to frozen manifest/scorer evidence"
            )
        if (
            type(binding.get("rows")) is not int
            or binding["rows"] <= 0
            or _SHA256_RE.fullmatch(str(binding.get("content_sha256"))) is None
            or _SHA256_RE.fullmatch(str(binding.get("membership_sha256"))) is None
        ):
            raise MobileTemporalEvaluationError(
                f"comparison {reference_name!r} has an invalid manifest binding"
            )
        pooled = candidate.get("pooled")
        if not isinstance(pooled, Mapping):
            raise MobileTemporalEvaluationError(
                f"comparison {reference_name!r} lacks pooled candidate scores"
            )
        if first is None:
            first = pooled
            first_binding = binding
            first_seeds = seeds
        elif pooled != first or binding != first_binding or seeds != first_seeds:
            raise MobileTemporalEvaluationError(
                "candidate, manifest, or seed evidence differs between reference comparisons"
            )
    if first is None:
        raise MobileTemporalEvaluationError("at least one reference comparison is required")
    return first


def _pooled_metric(pooled: Mapping[str, Any], *, metric: str) -> tuple[Fraction, int]:
    subsets = pooled.get("subsets")
    if not isinstance(subsets, Mapping) or not isinstance(subsets.get(OVERALL_SUBSET), Mapping):
        raise MobileTemporalEvaluationError("pooled candidate score lacks overall metrics")
    record = subsets[OVERALL_SUBSET].get(metric)
    if not isinstance(record, Mapping):
        raise MobileTemporalEvaluationError(f"pooled candidate score lacks {metric!r}")
    return _record_fraction(record, name=f"pooled candidate {metric}"), int(record["numerator"])


def _expected_bootstrap_contract(
    value: object, *, expected_seeds: tuple[int, int, int]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BOOTSTRAP_CONTRACT_FIELDS:
        raise MobileTemporalEvaluationError(
            "bootstrap_contract must contain exactly the frozen bootstrap identity fields"
        )
    contract_seeds_value = value.get("expected_training_seeds")
    if not isinstance(contract_seeds_value, Sequence) or isinstance(
        contract_seeds_value, (str, bytes, bytearray)
    ):
        raise MobileTemporalEvaluationError("bootstrap expected_training_seeds is invalid")
    contract_seeds = _expected_seed_tuple(contract_seeds_value)
    if contract_seeds != expected_seeds:
        raise MobileTemporalEvaluationError(
            "bootstrap expected_training_seeds differs from the gate seed identity"
        )
    resamples = value.get("resamples")
    if type(resamples) is not int or resamples <= 0:
        raise MobileTemporalEvaluationError("bootstrap contract resamples must be positive")
    rng = value.get("rng")
    if not isinstance(rng, Mapping) or set(rng) != _BOOTSTRAP_RNG_FIELDS:
        raise MobileTemporalEvaluationError("bootstrap contract rng identity is invalid")
    rng_seed = rng.get("seed")
    if type(rng_seed) is not int or rng_seed < 0:
        raise MobileTemporalEvaluationError("bootstrap contract rng seed is invalid")
    expected_strings = {
        "estimator": _BOOTSTRAP_ESTIMATOR,
        "quantile_method": "linear",
        "resampling_unit": _BOOTSTRAP_RESAMPLING_UNIT,
    }
    if any(value.get(name) != expected for name, expected in expected_strings.items()):
        raise MobileTemporalEvaluationError("bootstrap contract changes the frozen estimator")
    expected_rng = {
        "api": "Generator",
        "bit_generator": "PCG64",
        "library": "numpy",
        "seed": rng_seed,
    }
    if dict(rng) != expected_rng:
        raise MobileTemporalEvaluationError("bootstrap contract must use NumPy Generator/PCG64")
    return {
        **expected_strings,
        "expected_training_seeds": list(contract_seeds),
        "resamples": resamples,
        "rng": expected_rng,
    }


def evaluate_temporal_gates(
    comparisons: Mapping[str, Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any],
    bootstraps: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply explicit exact thresholds as one conjunctive gate across all controls."""

    if not isinstance(comparisons, Mapping) or len(comparisons) != 2:
        raise MobileTemporalEvaluationError(
            "comparisons must contain exactly the two preregistered reference arms"
        )
    if any(not isinstance(name, str) or not name for name in comparisons):
        raise MobileTemporalEvaluationError("comparison names must be non-empty strings")
    if not isinstance(thresholds, Mapping):
        raise MobileTemporalEvaluationError("thresholds must be an object")
    fields = set(thresholds)
    missing_fields = sorted(_REQUIRED_THRESHOLD_FIELDS.difference(fields))
    extra_fields = sorted(
        fields.difference(_REQUIRED_THRESHOLD_FIELDS | _OPTIONAL_THRESHOLD_FIELDS)
    )
    if missing_fields or extra_fields:
        raise MobileTemporalEvaluationError(
            f"threshold fields mismatch; missing={missing_fields!r}, extra={extra_fields!r}"
        )
    comparison_names_value = thresholds.get("comparison_names")
    if not isinstance(comparison_names_value, Sequence) or isinstance(
        comparison_names_value, (str, bytes, bytearray)
    ):
        raise MobileTemporalEvaluationError("comparison_names must be a sequence")
    comparison_names = tuple(comparison_names_value)
    if (
        len(comparison_names) != 2
        or len(set(comparison_names)) != 2
        or any(not isinstance(name, str) or not name for name in comparison_names)
        or set(comparisons) != set(comparison_names)
    ):
        raise MobileTemporalEvaluationError(
            "comparison_names must bind exactly the two preregistered reference arms"
        )
    expected_seeds_value = thresholds.get("expected_seeds")
    if not isinstance(expected_seeds_value, Sequence) or isinstance(
        expected_seeds_value, (str, bytes, bytearray)
    ):
        raise MobileTemporalEvaluationError("expected_seeds must be a sequence")
    expected_seeds = _expected_seed_tuple(expected_seeds_value)
    for reference_name, comparison in comparisons.items():
        if comparison.get("expected_seeds") != list(expected_seeds):
            raise MobileTemporalEvaluationError(
                f"comparison {reference_name!r} does not match the preregistered seed identity"
            )
    target_subset = thresholds.get("target_subset")
    target_metric = thresholds.get("target_metric")
    overall_metric = thresholds.get("overall_metric")
    if (
        target_subset != CALENDAR_CROSS_MONTH_SUBSET
        or target_metric != CALENDAR_DATETIME_EXACT_METRIC
        or overall_metric != AST_EXACT_METRIC
    ):
        raise MobileTemporalEvaluationError(
            "the frozen target endpoint is calendar_cross_month datetime exact with overall AST"
        )
    regression_value = thresholds.get("regression_subsets")
    regression_metrics_value = thresholds.get("regression_metrics")
    if not isinstance(regression_value, Sequence) or isinstance(
        regression_value, (str, bytes, bytearray)
    ):
        raise MobileTemporalEvaluationError("regression_subsets must be a sequence")
    regression_subsets = tuple(regression_value)
    if (
        set(regression_subsets) != {CALENDAR_SAME_MONTH_SUBSET, NON_CALENDAR_SUBSET}
        or len(regression_subsets) != 2
    ):
        raise MobileTemporalEvaluationError(
            "the frozen regression subsets are calendar_same_month and non_calendar"
        )
    if not isinstance(regression_metrics_value, Mapping) or set(regression_metrics_value) != set(
        regression_subsets
    ):
        raise MobileTemporalEvaluationError(
            "regression_metrics must map exactly the configured regression subsets"
        )
    regression_metrics = dict(regression_metrics_value)
    if regression_metrics != {
        CALENDAR_SAME_MONTH_SUBSET: CALENDAR_DATETIME_EXACT_METRIC,
        NON_CALENDAR_SUBSET: AST_EXACT_METRIC,
    }:
        raise MobileTemporalEvaluationError(
            "regression_metrics does not match the frozen endpoints"
        )

    minimum_target_gain = _threshold_ratio(
        thresholds["minimum_target_mean_gain"], name="minimum_target_mean_gain"
    )
    minimum_overall_gain = _threshold_ratio(
        thresholds["minimum_overall_mean_gain"], name="minimum_overall_mean_gain"
    )
    maximum_regression_loss = _threshold_ratio(
        thresholds["maximum_regression_mean_loss"], name="maximum_regression_mean_loss"
    )
    minimum_parse = _threshold_ratio(
        thresholds["minimum_candidate_parse_valid"], name="minimum_candidate_parse_valid"
    )
    minimum_schema = _threshold_ratio(
        thresholds["minimum_candidate_schema_valid"], name="minimum_candidate_schema_valid"
    )
    if minimum_parse > 1 or minimum_schema > 1:
        raise MobileTemporalEvaluationError("validity thresholds cannot exceed one")
    minimum_positive = _non_negative_integer(
        thresholds["minimum_positive_seed_count"], name="minimum_positive_seed_count"
    )
    maxima = {
        name: _non_negative_integer(thresholds[name], name=name)
        for name in (
            "maximum_missing_predictions",
            "maximum_generation_failures",
            "maximum_truncations",
            "maximum_catastrophic_unauthorized_actions",
        )
    }
    bootstrap_value = thresholds.get("bootstrap_lower_bound_strictly_greater_than")
    bootstrap_probability_value = thresholds.get("bootstrap_lower_probability")
    bootstrap_contract_value = thresholds.get("bootstrap_contract")
    bootstrap_threshold = (
        None
        if bootstrap_value is None
        else _threshold_ratio(bootstrap_value, name="bootstrap_lower_bound_strictly_greater_than")
    )
    if bootstrap_threshold is not None:
        if bootstrap_probability_value is None:
            raise MobileTemporalEvaluationError(
                "bootstrap_lower_probability is required with a bootstrap threshold"
            )
        bootstrap_probability = _threshold_ratio(
            bootstrap_probability_value, name="bootstrap_lower_probability"
        )
        if not 0 < bootstrap_probability < 1:
            raise MobileTemporalEvaluationError(
                "bootstrap_lower_probability must lie strictly between zero and one"
            )
        if not isinstance(bootstraps, Mapping) or set(bootstraps) != set(comparisons):
            raise MobileTemporalEvaluationError(
                "bootstrap gate requires one bootstrap per reference comparison"
            )
        bootstrap_contract = _expected_bootstrap_contract(
            bootstrap_contract_value, expected_seeds=expected_seeds
        )
    elif bootstraps:
        raise MobileTemporalEvaluationError(
            "bootstraps were supplied without an explicit bootstrap threshold"
        )
    elif bootstrap_probability_value is not None:
        raise MobileTemporalEvaluationError(
            "bootstrap_lower_probability was supplied without a bootstrap threshold"
        )
    else:
        bootstrap_probability = None
        if bootstrap_contract_value is not None:
            raise MobileTemporalEvaluationError(
                "bootstrap_contract was supplied without a bootstrap threshold"
            )
        bootstrap_contract = None

    pooled = _pooled_candidate(comparisons)
    parse_rate, _ = _pooled_metric(pooled, metric="parse_valid")
    schema_rate, _ = _pooled_metric(pooled, metric="schema_valid")
    quality_checks = {
        "parse_valid_at_least_threshold": parse_rate >= minimum_parse,
        "schema_valid_at_least_threshold": schema_rate >= minimum_schema,
    }
    quality_observed: dict[str, Any] = {
        "parse_valid": _fraction_record(parse_rate),
        "schema_valid": _fraction_record(schema_rate),
    }
    failure_metrics = {
        "maximum_missing_predictions": "missing_prediction",
        "maximum_generation_failures": "generation_failure",
        "maximum_truncations": "truncation",
        "maximum_catastrophic_unauthorized_actions": ("catastrophic_unauthorized_action"),
    }
    for threshold_name, metric in failure_metrics.items():
        rate, numerator = _pooled_metric(pooled, metric=metric)
        quality_observed[metric] = _fraction_record(rate)
        quality_checks[f"{metric}_at_most_threshold"] = numerator <= maxima[threshold_name]

    comparison_results: dict[str, Any] = {}
    all_checks = list(quality_checks.values())
    for reference_name in sorted(comparisons):
        comparison = comparisons[reference_name]
        target = _comparison_metric(
            comparison,
            subset=str(target_subset),
            metric=str(target_metric),
            reference_name=reference_name,
        )
        overall = _comparison_metric(
            comparison,
            subset=OVERALL_SUBSET,
            metric=str(overall_metric),
            reference_name=reference_name,
        )
        target_delta = _record_fraction(
            target.get("mean_delta", {}),
            name=f"{reference_name}.{target_subset}.{target_metric}.mean_delta",
        )
        overall_delta = _record_fraction(
            overall.get("mean_delta", {}),
            name=f"{reference_name}.overall.{overall_metric}.mean_delta",
        )
        positive_seed_count = target.get("positive_seed_count")
        seed_count = target.get("seed_count")
        if (
            type(positive_seed_count) is not int
            or type(seed_count) is not int
            or seed_count != 3
            or not 0 <= positive_seed_count <= seed_count
            or minimum_positive > seed_count
        ):
            raise MobileTemporalEvaluationError(
                f"comparison {reference_name!r} has invalid three-seed counts"
            )
        checks = {
            "target_mean_gain_at_least_threshold": target_delta >= minimum_target_gain,
            "overall_mean_gain_at_least_threshold": overall_delta >= minimum_overall_gain,
            "positive_seed_count_at_least_threshold": positive_seed_count >= minimum_positive,
        }
        observed_regressions: dict[str, Any] = {}
        for subset in regression_subsets:
            metric = str(regression_metrics[subset])
            subgroup = _comparison_metric(
                comparison,
                subset=str(subset),
                metric=metric,
                reference_name=reference_name,
            )
            delta = _record_fraction(
                subgroup.get("mean_delta", {}),
                name=f"{reference_name}.{subset}.{metric}.mean_delta",
            )
            loss = -delta
            observed_regressions[str(subset)] = {
                "metric": metric,
                "loss": _fraction_record(loss),
            }
            checks[f"{subset}_loss_at_most_threshold"] = loss <= maximum_regression_loss

        bootstrap_observation: dict[str, Any] | None = None
        if bootstrap_threshold is not None:
            assert bootstraps is not None
            assert bootstrap_contract is not None
            bootstrap = bootstraps[reference_name]
            if (
                bootstrap.get("schema_version") != MOBILE_TEMPORAL_BOOTSTRAP_VERSION
                or bootstrap.get("subset") != target_subset
                or bootstrap.get("metric") != target_metric
                or bootstrap.get("manifest_binding") != comparison.get("manifest_binding")
                or bootstrap.get("expected_seeds") != comparison.get("expected_seeds")
            ):
                raise MobileTemporalEvaluationError(
                    f"bootstrap {reference_name!r} does not match comparison endpoint"
                )
            actual_contract = {
                "estimator": bootstrap.get("estimator"),
                "expected_training_seeds": bootstrap.get("expected_seeds"),
                "quantile_method": bootstrap.get("quantile_method"),
                "resamples": bootstrap.get("resamples"),
                "resampling_unit": bootstrap.get("resampling_unit"),
                "rng": bootstrap.get("rng"),
            }
            if actual_contract != bootstrap_contract:
                raise MobileTemporalEvaluationError(
                    f"bootstrap {reference_name!r} does not match the frozen bootstrap contract"
                )
            cluster_counts = bootstrap.get("cluster_counts_by_seed")
            if (
                not isinstance(cluster_counts, Mapping)
                or set(cluster_counts) != {str(seed) for seed in expected_seeds}
                or any(type(count) is not int or count <= 0 for count in cluster_counts.values())
                or _SHA256_RE.fullmatch(str(bootstrap.get("replicate_float64_le_sha256"))) is None
            ):
                raise MobileTemporalEvaluationError(
                    f"bootstrap {reference_name!r} lacks complete cluster/replicate evidence"
                )
            point_exact = bootstrap.get("point_estimate_exact")
            point_float = bootstrap.get("point_estimate")
            if (
                not isinstance(point_exact, Mapping)
                or _record_fraction(
                    point_exact, name=f"bootstrap {reference_name}.point_estimate_exact"
                )
                != target_delta
                or not isinstance(point_float, (int, float))
                or not np.isfinite(point_float)
                or float(point_float) != float(target_delta)
            ):
                raise MobileTemporalEvaluationError(
                    f"bootstrap {reference_name!r} point estimate differs from comparison"
                )
            lower = bootstrap.get("one_sided_lower_percentile")
            if not isinstance(lower, Mapping):
                raise MobileTemporalEvaluationError(
                    f"bootstrap {reference_name!r} lacks a lower percentile"
                )
            probability_record = lower.get("probability")
            if (
                not isinstance(probability_record, Mapping)
                or _record_fraction(
                    probability_record,
                    name=f"bootstrap {reference_name}.lower_probability",
                )
                != bootstrap_probability
            ):
                raise MobileTemporalEvaluationError(
                    f"bootstrap {reference_name!r} uses the wrong lower probability"
                )
            lower_value = lower.get("value")
            if not isinstance(lower_value, (int, float)) or not np.isfinite(lower_value):
                raise MobileTemporalEvaluationError(
                    f"bootstrap {reference_name!r} lower percentile is invalid"
                )
            bootstrap_observation = {
                "one_sided_lower_percentile": float(lower_value),
                "strict_threshold": _fraction_record(bootstrap_threshold),
            }
            checks["bootstrap_lower_bound_strictly_above_threshold"] = float(lower_value) > float(
                bootstrap_threshold
            )
        passed = all(checks.values())
        all_checks.extend(checks.values())
        comparison_results[reference_name] = {
            "passed": passed,
            "checks": checks,
            "observed": {
                "target_mean_gain": _fraction_record(target_delta),
                "overall_mean_gain": _fraction_record(overall_delta),
                "positive_seed_count": positive_seed_count,
                "seed_count": seed_count,
                "regression_losses": observed_regressions,
                "bootstrap": bootstrap_observation,
            },
        }

    passed = all(all_checks)
    return {
        "schema_version": MOBILE_TEMPORAL_GATE_VERSION,
        "operator": "all_components_and_all_reference_comparisons_must_pass",
        "passed": passed,
        "status": "pass" if passed else "fail",
        "endpoint": {"target_subset": target_subset, "target_metric": target_metric},
        "candidate_quality": {
            "passed": all(quality_checks.values()),
            "checks": quality_checks,
            "observed": quality_observed,
        },
        "comparisons": comparison_results,
        "thresholds": {
            "comparison_names": list(comparison_names),
            "expected_seeds": list(expected_seeds),
            "target_subset": target_subset,
            "target_metric": target_metric,
            "overall_metric": overall_metric,
            "regression_subsets": list(regression_subsets),
            "regression_metrics": regression_metrics,
            "minimum_target_mean_gain": _fraction_record(minimum_target_gain),
            "minimum_overall_mean_gain": _fraction_record(minimum_overall_gain),
            "minimum_positive_seed_count": minimum_positive,
            "maximum_regression_mean_loss": _fraction_record(maximum_regression_loss),
            "minimum_candidate_parse_valid": _fraction_record(minimum_parse),
            "minimum_candidate_schema_valid": _fraction_record(minimum_schema),
            **maxima,
            "bootstrap_lower_bound_strictly_greater_than": (
                None if bootstrap_threshold is None else _fraction_record(bootstrap_threshold)
            ),
            "bootstrap_lower_probability": (
                None if bootstrap_probability is None else _fraction_record(bootstrap_probability)
            ),
            "bootstrap_contract": bootstrap_contract,
        },
    }


__all__ = [
    "AST_EXACT_METRIC",
    "CALENDAR_CROSS_MONTH_SUBSET",
    "CALENDAR_DATETIME_EXACT_METRIC",
    "CALENDAR_SAME_MONTH_SUBSET",
    "MOBILE_TEMPORAL_BOOTSTRAP_VERSION",
    "MOBILE_TEMPORAL_COMPARISON_VERSION",
    "MOBILE_TEMPORAL_GATE_VERSION",
    "MOBILE_TEMPORAL_SCORE_VERSION",
    "NON_CALENDAR_SUBSET",
    "OVERALL_SUBSET",
    "MobileTemporalEvaluationError",
    "compare_seeded_arms",
    "evaluate_temporal_gates",
    "paired_cluster_bootstrap",
    "score_temporal_subsets",
]
